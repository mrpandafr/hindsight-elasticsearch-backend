"""Native schema migrations for the Elasticsearch backend.

Placement: hindsight_api/engine/db/migrations_elasticsearch.py

This is the Elasticsearch counterpart of the Alembic tree
(``hindsight_api/alembic/``). Alembic drives SQL DDL through SQLAlchemy
dialects (``_dialect.run_for_dialect`` dispatches ``postgresql`` / ``oracle``
/ ``elasticsearch``); Elasticsearch has no SQLAlchemy dialect and its schema
is index mappings, so its migrations run through this module instead:

* ``ElasticsearchBackend.initialize()`` calls ``run_migrations()`` on every
  startup (same behaviour as "migrations run automatically on API startup").
* The admin CLI path (``hindsight-admin run-db-migration``) must branch on
  ``is_elasticsearch_url(database_url)`` and call ``run_migrations()``
  instead of invoking Alembic — see README.

Model
-----
Mirrors Alembic's semantics with ES-native mechanics:

* an ordered list of ``Migration(version, description, apply)`` steps
  (the "migration tree"; linear, like the repo's Alembic tree);
* an ``alembic_version``-equivalent tracking index
  (``{schema}-hindsight_migrations``) with one document per applied version;
* concurrency safety without advisory locks: each version document is
  written with a deterministic ``_id`` and ``op_type=create``, so exactly
  one concurrent starter claims a version and runs it; the others skip it
  (409). Apply functions must therefore be idempotent — which they are by
  construction, since ES index/mapping operations are naturally so
  (``indices.create`` ignoring resource_already_exists,
  ``indices.put_mapping`` being additive).

ES mappings can only be *extended* (new fields, new indexes); a breaking
change (field type change) requires a new index + reindex, which is what a
migration step encodes when needed. There are no downgrades — same stance
as running Alembic ``upgrade head`` only.

Adding a migration
------------------
Append to ``MIGRATIONS`` (never reorder, never edit an applied step):

    async def _0002_add_confidence(client, schema, ctx):
        await put_mapping(client, schema, "memory_units",
                          {"confidence": {"type": "float"}})

    MIGRATIONS.append(Migration("0002", "add confidence to memory_units",
                                _0002_add_confidence))
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from .ops_elasticsearch import INDEX_MAPPINGS, _index, _status_of

logger = logging.getLogger(__name__)

#: alembic_version equivalent (one index per schema prefix).
MIGRATIONS_TABLE = "hindsight_migrations"

_MIGRATIONS_MAPPING = {
    "properties": {
        "version": {"type": "keyword"},
        "description": {"type": "text"},
        "applied_at": {"type": "date"},
        "status": {"type": "keyword"},  # applied | failed
        "error": {"type": "text"},
    }
}


@dataclass
class MigrationContext:
    """Options forwarded to apply functions."""

    embedding_dims: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


ApplyFn = Callable[[Any, str, MigrationContext], Awaitable[None]]


@dataclass(frozen=True)
class Migration:
    version: str  # zero-padded, ordered ("0001", "0002", ...)
    description: str
    apply: ApplyFn


# ---------------------------------------------------------------------------
# Native DDL helpers (the op.create_table / op.add_column of this backend)
# ---------------------------------------------------------------------------

async def create_index(client: Any, schema: str, table: str,
                       mappings: dict, settings: dict | None = None) -> None:
    """Idempotent index creation (CREATE TABLE IF NOT EXISTS equivalent)."""
    idx = _index(f"{schema}.{table}")
    if await client.indices.exists(index=idx):
        return
    try:
        kwargs: dict[str, Any] = {"index": idx, "mappings": mappings}
        if settings:
            kwargs["settings"] = settings
        await client.indices.create(**kwargs)
        logger.info("created index %s", idx)
    except Exception as exc:
        if _status_of(exc) != 400:  # resource_already_exists race
            raise


async def put_mapping(client: Any, schema: str, table: str,
                      properties: dict) -> None:
    """Additive field addition (ALTER TABLE ADD COLUMN equivalent).

    ES put_mapping is additive-only: attempting to change an existing
    field's type raises, exactly like a conflicting ALTER would.
    """
    await client.indices.put_mapping(
        index=_index(f"{schema}.{table}"),
        properties=properties,
    )


async def reindex_to(client: Any, schema: str, table: str,
                     new_mappings: dict, suffix: str = "v2") -> None:
    """Breaking change path: new index + _reindex + alias swap.

    Creates ``{index}-{suffix}`` with the new mappings, copies the data with
    the _reindex API, then re-points the canonical name as an alias. Use for
    field-type changes, which ES forbids in place.
    """
    old = _index(f"{schema}.{table}")
    new = f"{old}-{suffix}"
    await create_index(client, schema, f"{table}-{suffix}", new_mappings)
    await client.reindex(
        body={"source": {"index": old}, "dest": {"index": new}},
        wait_for_completion=True,
        refresh=True,
    )
    await client.indices.delete(index=old)
    await client.indices.update_aliases(body={"actions": [
        {"add": {"index": new, "alias": old}},
    ]})
    logger.info("reindexed %s -> %s (alias swapped)", old, new)


# ---------------------------------------------------------------------------
# Migration tree
# ---------------------------------------------------------------------------

async def _0001_baseline(client: Any, schema: str, ctx: MigrationContext) -> None:
    """Baseline: every index of ops_elasticsearch.INDEX_MAPPINGS.

    Same role as the initial Alembic revision that creates all tables.
    The mappings live next to the ops (single source of truth); this step
    only pins the dense_vector dimension when configured.
    """
    for table, mapping in INDEX_MAPPINGS.items():
        body = json.loads(json.dumps(mapping))  # deep copy
        if table == "memory_units" and ctx.embedding_dims:
            body["properties"]["embedding"]["dims"] = ctx.embedding_dims
        await create_index(client, schema, table, body)


MIGRATIONS: list[Migration] = [
    Migration("0001", "baseline indexes from INDEX_MAPPINGS", _0001_baseline),
    # Append new steps here — never reorder, never edit an applied step.
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def _ensure_tracking_index(client: Any, schema: str) -> str:
    idx = _index(f"{schema}.{MIGRATIONS_TABLE}")
    if not await client.indices.exists(index=idx):
        try:
            await client.indices.create(index=idx, mappings=_MIGRATIONS_MAPPING)
        except Exception as exc:
            if _status_of(exc) != 400:
                raise
    return idx


async def _applied_versions(client: Any, tracking_idx: str) -> set[str]:
    resp = await client.search(
        index=tracking_idx,
        body={
            "query": {"term": {"status": "applied"}},
            "size": 10000,
            "_source": ["version"],
            "track_total_hits": False,
        },
    )
    return {h["_source"]["version"] for h in resp["hits"]["hits"]}


async def run_migrations(
    client: Any,
    schema: str = "public",
    *,
    embedding_dims: int | None = None,
    migrations: list[Migration] | None = None,
) -> list[str]:
    """Apply every pending migration for ``schema``; return applied versions.

    Alembic-``upgrade head`` equivalent. Concurrency-safe across several
    starting API processes: a version is claimed by creating its tracking
    document with ``op_type=create`` (deterministic ``_id`` = version);
    the loser of the race gets a 409 and skips the step.

    A failing step marks its document ``status=failed`` (with the error) and
    aborts the run, so the next startup surfaces the same pending version
    instead of silently continuing past a hole in the tree.
    """
    steps = migrations if migrations is not None else MIGRATIONS
    ordered = sorted(steps, key=lambda m: m.version)
    versions = [m.version for m in ordered]
    if len(set(versions)) != len(versions):
        raise ValueError(f"duplicate migration versions: {versions}")

    tracking_idx = await _ensure_tracking_index(client, schema)
    applied = await _applied_versions(client, tracking_idx)
    ctx = MigrationContext(embedding_dims=embedding_dims)
    ran: list[str] = []

    for mig in ordered:
        if mig.version in applied:
            continue
        # claim the version (advisory-lock equivalent)
        try:
            await client.index(
                index=tracking_idx,
                id=mig.version,
                op_type="create",
                document={
                    "version": mig.version,
                    "description": mig.description,
                    "applied_at": datetime.now(timezone.utc).isoformat(),
                    "status": "applied",
                },
                refresh="wait_for",
            )
        except Exception as exc:
            if _status_of(exc) == 409:
                # another process claimed it; if it previously *failed*,
                # surface that instead of skipping silently.
                doc = await client.get(index=tracking_idx, id=mig.version)
                if doc["_source"].get("status") == "failed":
                    raise RuntimeError(
                        f"migration {mig.version} previously failed: "
                        f"{doc['_source'].get('error')!r} — fix and delete "
                        f"its tracking document to retry."
                    )
                logger.info("migration %s claimed elsewhere, skipping", mig.version)
                continue
            raise

        logger.info("applying migration %s: %s", mig.version, mig.description)
        try:
            await mig.apply(client, schema, ctx)
        except Exception as exc:
            await client.update(
                index=tracking_idx,
                id=mig.version,
                doc={"status": "failed", "error": str(exc)},
                refresh="wait_for",
            )
            raise
        ran.append(mig.version)

    return ran


async def current_version(client: Any, schema: str = "public") -> str | None:
    """Highest applied version (``alembic current`` equivalent)."""
    tracking_idx = await _ensure_tracking_index(client, schema)
    applied = await _applied_versions(client, tracking_idx)
    return max(applied) if applied else None
