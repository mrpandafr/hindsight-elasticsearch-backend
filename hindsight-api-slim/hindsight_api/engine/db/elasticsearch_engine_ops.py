"""Engine-facing native operations for the Elasticsearch backend.

Placement: hindsight_api/engine/db/elasticsearch_engine_ops.py

Covers the last SQL surfaces of memory_engine.py — tag listing, bank stats
(the profile endpoint), and the small async_operations idioms the retain
worker path leans on — with output shapes copied from the SQL verbatim, so
the engine hooks are one-liners. Health needs no hook at all: the
connection's ``fetchval`` answers the ``SELECT 1`` liveness idiom with
``client.ping()``.

Engine patch guide (one line each, same dispatch key as retrieval.py):

    # memory_engine.list_tags internals (~line 9665)
    if getattr(conn, "backend_type", "") == "elasticsearch":
        from .db.elasticsearch_engine_ops import list_tags_paginated
        return await list_tags_paginated(conn, fq_table(table), bank_id,
                                         pattern, limit, offset)

    # memory_engine._compute_bank_stats (~line 9785)
    if getattr(conn, "backend_type", "") == "elasticsearch":
        from .db.elasticsearch_engine_ops import compute_bank_stats
        return await compute_bank_stats(conn, fq_table, bank_id)

    # retain worker idioms (_check_op_alive, cancelled-check,
    # _update_webhook_delivery_metadata, _delete_operation_record):
    #   get_operation_status / set_operation_result_metadata / delete_operation

    # retain re-ingest (delete previous units of a document, first batch):
    #   purge_document_units — returns the deleted unit ids like RETURNING id

Everything here is bulk/aggregation/ping — no SQL, no translation.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from .ops_elasticsearch import _index, _iso, _load_json, _status_of

logger = logging.getLogger(__name__)

_AGG_PAGE = 2000  # composite-aggregation page size


def _client(conn: Any):
    return getattr(conn, "client", conn)


# ---------------------------------------------------------------------------
# Tags listing (memory_engine.list_tags)
# ---------------------------------------------------------------------------

async def list_tags_paginated(
    conn: Any,
    table: str,
    bank_id: str,
    pattern: str | None,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """Native port of the tag-listing SQL, same response shape:
    ``{"items": [{"tag", "count"}...], "total", "limit", "offset"}``.

    SQL: unnest(tags) GROUP BY, COUNT DISTINCT for total, ILIKE pattern,
    ORDER BY count DESC, tag ASC, LIMIT/OFFSET.
    ES: a composite aggregation over the ``tags`` keyword field (arrays
    flatten natively; composite paginates past the terms-agg size ceiling),
    then the case-insensitive wildcard (``*`` = ILIKE ``%``), the exact
    sort, and the offset/limit window applied client-side — tag
    cardinality is human-scale, the buckets are cheap to hold.
    """
    client = _client(conn)
    idx = _index(table)
    buckets: list[tuple[str, int]] = []
    after: dict | None = None
    while True:
        comp: dict[str, Any] = {
            "size": _AGG_PAGE,
            "sources": [{"tag": {"terms": {"field": "tags"}}}],
        }
        if after:
            comp["after"] = after
        resp = await client.search(
            index=idx,
            body={
                "query": {"bool": {"filter": [{"term": {"bank_id": bank_id}}],
                                    "must": [{"exists": {"field": "tags"}}]}},
                "size": 0,
                "aggs": {"tags": {"composite": comp}},
            },
        )
        agg = resp["aggregations"]["tags"]
        for b in agg.get("buckets", []):
            buckets.append((str(b["key"]["tag"]), int(b["doc_count"])))
        after = agg.get("after_key")
        if not after or not agg.get("buckets"):
            break

    if pattern:
        # '*' wildcard, case-insensitive — the ILIKE contract
        pat = pattern.lower()
        buckets = [(t, c) for t, c in buckets if fnmatch.fnmatchcase(t.lower(), pat)]

    total = len(buckets)
    buckets.sort(key=lambda tc: (-tc[1], tc[0]))  # count DESC, tag ASC
    window = buckets[offset:offset + limit]
    return {
        "items": [{"tag": t, "count": c} for t, c in window],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# ---------------------------------------------------------------------------
# Bank stats (memory_engine._compute_bank_stats — the profile endpoint)
# ---------------------------------------------------------------------------

async def compute_bank_stats(
    conn: Any,
    fq_table: Callable[[str], str],
    bank_id: str,
    max_links_per_entity: int = 10,
) -> dict[str, Any]:
    """Native port of _compute_bank_stats, identical response dict.

    * node_counts        — terms agg on fact_type (memory_units)
    * link_counts        — terms agg on link_type (memory_links) plus the
                           derived "entity" total: per-entity unit counts
                           from unit_entities, capped SUM(LEAST(n-1, cap)),
                           scoped to the bank through the entities index
                           (unit_entities has no bank_id, same as SQL's join)
    * operations         — terms agg on status (async_operations)
    * total_documents    — count on documents
    * consolidation      — max(consolidated_at) + pending/failed counts;
                           on indexes created before those fields existed the
                           aggregations degrade to null/0 (unmapped fields),
                           matching a bank that has never consolidated
    """
    client = _client(conn)
    mu_idx = _index(fq_table("memory_units"))
    ml_idx = _index(fq_table("memory_links"))
    ue_idx = _index(fq_table("unit_entities"))
    e_idx = _index(fq_table("entities"))
    ops_idx = _index(fq_table("async_operations"))
    doc_idx = _index(fq_table("documents"))
    bank_filter = {"term": {"bank_id": bank_id}}
    cons_types = {"terms": {"fact_type": ["experience", "world"]}}

    # nodes + consolidation rollup in one request
    resp = await client.search(
        index=mu_idx,
        body={
            "query": bank_filter, "size": 0,
            "aggs": {
                "by_fact_type": {"terms": {"field": "fact_type", "size": 50}},
                "last_consolidated": {"max": {"field": "consolidated_at"}},
                "pending": {"filter": {"bool": {
                    "filter": [cons_types],
                    "must_not": [{"exists": {"field": "consolidated_at"}}],
                }}},
                "failed": {"filter": {"bool": {"filter": [
                    cons_types, {"exists": {"field": "consolidation_failed_at"}},
                ]}}},
            },
        },
    )
    aggs = resp["aggregations"]
    node_counts = {b["key"]: b["doc_count"] for b in aggs["by_fact_type"]["buckets"]}
    last_val = aggs["last_consolidated"].get("value_as_string") or aggs["last_consolidated"].get("value")
    if isinstance(last_val, (int, float)):
        last_val = datetime.fromtimestamp(last_val / 1000, tz=timezone.utc).isoformat()
    pending = aggs["pending"]["doc_count"]
    failed = aggs["failed"]["doc_count"]

    # non-entity links
    resp = await client.search(
        index=ml_idx,
        body={"query": bank_filter, "size": 0,
              "aggs": {"by_link_type": {"terms": {"field": "link_type", "size": 50}}}},
    )
    link_counts: dict[str, int] = {
        b["key"]: b["doc_count"]
        for b in resp["aggregations"]["by_link_type"]["buckets"]
    }

    # derived entity links: SUM(LEAST(n-1, cap)) over per-entity unit counts,
    # bank-scoped through the entities index (unit_entities has no bank_id)
    bank_entities: set[str] = set()
    after: dict | None = None
    while True:
        comp: dict[str, Any] = {"size": _AGG_PAGE,
                                "sources": [{"e": {"terms": {"field": "id"}}}]}
        if after:
            comp["after"] = after
        resp = await client.search(
            index=e_idx,
            body={"query": bank_filter, "size": 0, "aggs": {"ids": {"composite": comp}}},
        )
        agg = resp["aggregations"]["ids"]
        bank_entities.update(str(b["key"]["e"]) for b in agg.get("buckets", []))
        after = agg.get("after_key")
        if not after or not agg.get("buckets"):
            break

    entity_link_total = 0
    if bank_entities:
        after = None
        while True:
            comp = {"size": _AGG_PAGE,
                    "sources": [{"e": {"terms": {"field": "entity_id"}}}]}
            if after:
                comp["after"] = after
            resp = await client.search(
                index=ue_idx,
                body={"size": 0, "aggs": {"per_entity": {"composite": comp}}},
            )
            agg = resp["aggregations"]["per_entity"]
            for b in agg.get("buckets", []):
                if str(b["key"]["e"]) in bank_entities:
                    entity_link_total += min(int(b["doc_count"]) - 1, max_links_per_entity)
            after = agg.get("after_key")
            if not after or not agg.get("buckets"):
                break
    if entity_link_total > 0:
        link_counts["entity"] = entity_link_total

    # operations by status + document count
    resp = await client.search(
        index=ops_idx,
        body={"query": bank_filter, "size": 0,
              "aggs": {"by_status": {"terms": {"field": "status", "size": 20}}}},
    )
    ops_by_status = {b["key"]: b["doc_count"]
                     for b in resp["aggregations"]["by_status"]["buckets"]}
    resp = await client.count(index=doc_idx, query=bank_filter)
    total_documents = int(resp.get("count", 0))

    return {
        "bank_id": bank_id,
        "node_counts": node_counts,
        "link_counts": link_counts,
        "link_counts_by_fact_type": {},
        "link_breakdown": [],
        "operations": ops_by_status,
        "total_documents": total_documents,
        "last_consolidated_at": last_val,
        "pending_consolidation": pending,
        "failed_consolidation": failed,
        "total_observations": node_counts.get("observation", 0),
    }


# ---------------------------------------------------------------------------
# Retain worker idioms over async_operations
# ---------------------------------------------------------------------------
# Mirrors of the four SQL one-liners the retain/worker path uses:
# cancelled-check, _check_op_alive, _update_webhook_delivery_metadata /
# progress writes, and _delete_operation_record. operation_id is the ES _id.

async def get_operation_status(conn: Any, table: str, operation_id: str) -> str | None:
    """``SELECT status FROM async_operations WHERE operation_id=$1`` — None
    when the row is gone (bank deleted), enabling the same cancelled/alive
    logic (`row is not None and row["status"] != "cancelled"`)."""
    try:
        got = await _client(conn).get(index=_index(table), id=str(operation_id),
                                      source_includes=["status"])
    except Exception as exc:
        if _status_of(exc) == 404:
            return None
        raise
    return got.get("_source", {}).get("status")


async def set_operation_result_metadata(
    conn: Any, table: str, operation_id: str, metadata: dict | str,
    *, merge: bool = False,
) -> bool:
    """``UPDATE async_operations SET result_metadata=$2, updated_at=now()``.

    ``merge=True`` gives the jsonb ``||`` behaviour (progress writes layered
    over prior metadata); False replaces, like the webhook-delivery write.
    Returns False when the row no longer exists.
    """
    client, idx = _client(conn), _index(table)
    meta = _load_json(metadata, {}) if isinstance(metadata, str) else dict(metadata)
    doc: dict[str, Any] = {"updated_at": _iso(datetime.now(timezone.utc))}
    try:
        if merge:
            got = await client.get(index=idx, id=str(operation_id),
                                   source_includes=["result_metadata"])
            current = got.get("_source", {}).get("result_metadata") or {}
            if isinstance(current, str):
                current = _load_json(current, {})
            current.update(meta)
            doc["result_metadata"] = current
        else:
            doc["result_metadata"] = meta
        await client.update(index=idx, id=str(operation_id), doc=doc,
                            refresh="wait_for")
        return True
    except Exception as exc:
        if _status_of(exc) == 404:
            return False
        raise


async def delete_operation(conn: Any, table: str, operation_id: str) -> bool:
    """``DELETE FROM async_operations WHERE operation_id=$1``."""
    try:
        resp = await _client(conn).delete(index=_index(table), id=str(operation_id),
                                          refresh="wait_for")
    except Exception as exc:
        if _status_of(exc) == 404:
            return False
        raise
    return resp.get("result") == "deleted"


# ---------------------------------------------------------------------------
# Retain re-ingest: purge a document's previous memory graph
# ---------------------------------------------------------------------------

async def purge_document_units(
    conn: Any,
    fq_table: Callable[[str], str],
    bank_id: str,
    document_id: str,
) -> list[str]:
    """First-batch re-ingest cleanup: remove every memory unit of a document
    plus its graph attachments, returning the deleted unit ids (the SQL
    ``DELETE ... RETURNING id`` shape).

    Where PostgreSQL leans on ``ON DELETE CASCADE``, ES cascades explicitly:
    1. collect the unit ids (search, paginated);
    2. ``delete_by_query`` on memory_links touching them (either direction),
       on unit_entities, and on the graph-maintenance queue;
    3. delete the units themselves.
    Orphaned entities are left to prune_orphan_entities (the documented
    maintenance job), same as the SQL backend.
    """
    client = _client(conn)
    mu_idx = _index(fq_table("memory_units"))
    unit_ids: list[str] = []
    search_after = None
    while True:
        body: dict[str, Any] = {
            "query": {"bool": {"filter": [
                {"term": {"bank_id": bank_id}},
                {"term": {"document_id": document_id}},
            ]}},
            "size": 5000, "_source": ["id"], "sort": [{"_id": "asc"}],
            "track_total_hits": False,
        }
        if search_after is not None:
            body["search_after"] = search_after
        resp = await client.search(index=mu_idx, body=body)
        hits = resp["hits"]["hits"]
        if not hits:
            break
        unit_ids.extend(str(h["_source"].get("id", h["_id"])) for h in hits)
        search_after = hits[-1]["sort"]
    if not unit_ids:
        return []

    for start in range(0, len(unit_ids), 5000):
        chunk = unit_ids[start:start + 5000]
        await client.delete_by_query(
            index=_index(fq_table("memory_links")),
            query={"bool": {"should": [
                {"terms": {"from_unit_id": chunk}},
                {"terms": {"to_unit_id": chunk}},
            ], "minimum_should_match": 1}},
            conflicts="proceed", refresh=True,
        )
        await client.delete_by_query(
            index=_index(fq_table("unit_entities")),
            query={"terms": {"unit_id": chunk}},
            conflicts="proceed", refresh=True,
        )
        await client.delete_by_query(
            index=_index(fq_table("graph_maintenance_queue")),
            query={"terms": {"unit_id": chunk}},
            conflicts="proceed", refresh=True,
        )
        await client.delete_by_query(
            index=mu_idx,
            query={"terms": {"id": chunk}},
            conflicts="proceed", refresh=True,
        )
    return unit_ids


async def finalize_document(
    conn: Any,
    table: str,
    doc_id: str,
    bank_id: str,
    original_text: str,
    content_hash: str,
) -> None:
    """End-of-retain document upsert (text + content hash), the counterpart
    of lock_document_for_write's ``__pending__`` placeholder."""
    await _client(conn).update(
        index=_index(table),
        id=f"{bank_id}::{doc_id}",
        doc={
            "id": doc_id, "bank_id": bank_id,
            "original_text": original_text, "content_hash": content_hash,
            "updated_at": _iso(datetime.now(timezone.utc)),
        },
        doc_as_upsert=True,
        refresh="wait_for",
    )


# ---------------------------------------------------------------------------
# Block 1 — operation completion & parent/sibling rollup (async_operations)
# ---------------------------------------------------------------------------
# Native counterparts of _mark_operation_completed_and_fire_webhook /
# _maybe_update_parent_operation. The SQL transaction + FOR UPDATE become:
# read the parent with seq_no/primary_term, decide from the siblings, then
# conditional-update; a 409 means another child won the race — re-read and
# retry the whole decision (the engine snippet loops, bounded).

def _parse_dt_field(value: Any) -> Any:
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return value


async def get_operation_metadata(conn: Any, table: str, operation_id: str) -> dict | None:
    """``SELECT result_metadata WHERE operation_id=$1`` — parsed dict, or
    None when the row is gone. The outcome-metadata write pairs this with
    set_operation_result_metadata(..., merge=True) (the jsonb ``||``)."""
    try:
        got = await _client(conn).get(index=_index(table), id=str(operation_id),
                                      source_includes=["result_metadata"])
    except Exception as exc:
        if _status_of(exc) == 404:
            return None
        raise
    meta = got.get("_source", {}).get("result_metadata")
    return _load_json(meta, {}) if isinstance(meta, str) else (meta or {})


async def get_operation_fields(
    conn: Any, table: str, operation_id: str, fields: list[str],
    *, with_concurrency: bool = False,
) -> dict | None:
    """Field projection of one operation row; with_concurrency=True also
    returns _seq_no/_primary_term — the FOR UPDATE handle for conditional
    parent updates."""
    try:
        got = await _client(conn).get(index=_index(table), id=str(operation_id),
                                      source_includes=fields)
    except Exception as exc:
        if _status_of(exc) == 404:
            return None
        raise
    row = dict(got.get("_source", {}))
    if isinstance(row.get("result_metadata"), str):
        row["result_metadata"] = _load_json(row["result_metadata"], {})
    if with_concurrency:
        row["_seq_no"], row["_primary_term"] = got.get("_seq_no"), got.get("_primary_term")
    return row


async def mark_operation_completed(conn: Any, table: str, operation_id: str) -> bool:
    """``UPDATE ... SET status='completed', updated_at=NOW(), completed_at=NOW()
    RETURNING operation_id`` — False when the row no longer exists (bank
    deleted), the engine's skip signal. The webhook outbox row goes through
    ops.insert_webhook_delivery_task in the same flow; without transactions
    the ordering (status first, outbox second, both idempotent) is the
    at-least-once guarantee."""
    now = _iso(datetime.now(timezone.utc))
    try:
        await _client(conn).update(
            index=_index(table), id=str(operation_id),
            doc={"status": "completed", "updated_at": now, "completed_at": now},
            refresh="wait_for",
        )
        return True
    except Exception as exc:
        if _status_of(exc) == 404:
            return False
        raise


async def find_child_operations(
    conn: Any, table: str, bank_id: str, parent_operation_id: str,
) -> list[dict]:
    """``WHERE result_metadata @> {"parent_operation_id": X}`` — the sibling
    scan. ES: term on the dynamic subfield (keyword and raw text forms both
    tried; UUIDs single-token lowercase match either). Returns rows with
    status / error_message / result_metadata (parsed)."""
    pid = str(parent_operation_id)
    resp = await _client(conn).search(
        index=_index(table),
        body={
            "query": {"bool": {
                "filter": [{"term": {"bank_id": bank_id}}],
                "should": [
                    {"term": {"result_metadata.parent_operation_id.keyword": pid}},
                    {"term": {"result_metadata.parent_operation_id": pid}},
                ],
                "minimum_should_match": 1,
            }},
            "size": 10000,
            "_source": ["status", "error_message", "result_metadata"],
            "track_total_hits": False,
        },
    )
    rows: list[dict] = []
    for h in resp["hits"]["hits"]:
        src = h["_source"]
        meta = src.get("result_metadata")
        rows.append({
            "status": src.get("status"),
            "error_message": src.get("error_message"),
            "result_metadata": _load_json(meta, {}) if isinstance(meta, str) else (meta or {}),
        })
    return rows


async def update_operation_conditional(
    conn: Any, table: str, operation_id: str, doc: dict,
    *, seq_no: int, primary_term: int,
    merge_metadata: dict | None = None,
) -> bool:
    """Conditional parent update — the FOR UPDATE replacement. False on 409
    (another child finalized concurrently: re-read siblings and retry) or
    404 (parent gone)."""
    client, idx = _client(conn), _index(table)
    payload = dict(doc)
    payload["updated_at"] = _iso(datetime.now(timezone.utc))
    if merge_metadata:
        current = await get_operation_metadata(conn, table, operation_id) or {}
        current.update(merge_metadata)
        payload["result_metadata"] = current
    try:
        await client.update(index=idx, id=str(operation_id), doc=payload,
                            if_seq_no=seq_no, if_primary_term=primary_term,
                            refresh="wait_for")
        return True
    except Exception as exc:
        if _status_of(exc) in (404, 409):
            return False
        raise


# ---------------------------------------------------------------------------
# Block 2 — recall enrichment reads (observations, chunks, source facts,
# entities). All read-only; date fields come back as datetimes because the
# engine calls .isoformat() on them (the asyncpg contract).
# ---------------------------------------------------------------------------

async def fetch_observation_sources(
    conn: Any, table: str, observation_ids: list,
) -> list[dict]:
    """``SELECT id, source_memory_ids WHERE id=ANY($1) AND
    fact_type='observation'`` — serves both the prefer_observations dedup
    (superseded raw facts) and the source-fact enrichment."""
    ids = [str(i) for i in observation_ids]
    if not ids:
        return []
    rows: list[dict] = []
    for start in range(0, len(ids), 10000):
        resp = await _client(conn).search(
            index=_index(table),
            body={
                "query": {"bool": {"filter": [
                    {"terms": {"id": ids[start:start + 10000]}},
                    {"term": {"fact_type": "observation"}},
                ]}},
                "size": 10000,
                "_source": ["id", "source_memory_ids"],
                "track_total_hits": False,
            },
        )
        for h in resp["hits"]["hits"]:
            src = h["_source"]
            rows.append({"id": str(src.get("id", h["_id"])),
                         "source_memory_ids": [str(s) for s in src.get("source_memory_ids") or []]})
    return rows


async def fetch_observation_source_chunks(
    conn: Any, fq_table: Callable[[str], str], observation_ids_ordered: list,
) -> list[dict]:
    """Rows ``{obs_id, chunk_id}`` in observation-rank order (the
    ``array_position`` ORDER BY): observations' source units that carry a
    chunk_id. Two terms queries replace the self-join."""
    obs_rows = await fetch_observation_sources(
        conn, fq_table("memory_units"), observation_ids_ordered
    )
    by_obs = {r["id"]: r["source_memory_ids"] for r in obs_rows}
    all_sources = sorted({s for sids in by_obs.values() for s in sids})
    chunk_by_unit: dict[str, str] = {}
    for start in range(0, len(all_sources), 10000):
        resp = await _client(conn).search(
            index=_index(fq_table("memory_units")),
            body={
                "query": {"bool": {
                    "filter": [{"terms": {"id": all_sources[start:start + 10000]}}],
                    "must": [{"exists": {"field": "chunk_id"}}],
                }},
                "size": 10000,
                "_source": ["id", "chunk_id"],
                "track_total_hits": False,
            },
        )
        for h in resp["hits"]["hits"]:
            src = h["_source"]
            chunk_by_unit[str(src["id"])] = str(src["chunk_id"])
    out: list[dict] = []
    for obs_id in (str(o) for o in observation_ids_ordered):
        for sid in by_obs.get(obs_id, []):
            cid = chunk_by_unit.get(sid)
            if cid:
                out.append({"obs_id": obs_id, "chunk_id": cid})
    return out


async def fetch_chunks_by_ids(conn: Any, table: str, chunk_ids: list[str]) -> list[dict]:
    """``SELECT chunk_id, chunk_text, chunk_index WHERE chunk_id=ANY($1)``."""
    ids = [str(c) for c in chunk_ids]
    rows: list[dict] = []
    for start in range(0, len(ids), 10000):
        resp = await _client(conn).search(
            index=_index(table),
            body={
                "query": {"terms": {"chunk_id": ids[start:start + 10000]}},
                "size": 10000,
                "_source": ["chunk_id", "chunk_text", "chunk_index"],
                "track_total_hits": False,
            },
        )
        rows.extend({"chunk_id": h["_source"].get("chunk_id"),
                     "chunk_text": h["_source"].get("chunk_text"),
                     "chunk_index": h["_source"].get("chunk_index")}
                    for h in resp["hits"]["hits"])
    return rows


_SOURCE_FACT_FIELDS = ["id", "text", "fact_type", "context", "occurred_start",
                       "occurred_end", "mentioned_at", "document_id",
                       "chunk_id", "tags", "metadata"]


async def fetch_units_by_ids(conn: Any, table: str, unit_ids: list) -> list[dict]:
    """Source-fact hydration (``SELECT id, text, ... WHERE id=ANY($1)``).
    Date columns are parsed back to datetimes — the engine calls
    ``.isoformat()`` on them, per the asyncpg row contract."""
    ids = [str(i) for i in unit_ids]
    rows: list[dict] = []
    for start in range(0, len(ids), 10000):
        resp = await _client(conn).search(
            index=_index(table),
            body={
                "query": {"terms": {"id": ids[start:start + 10000]}},
                "size": 10000,
                "_source": _SOURCE_FACT_FIELDS,
                "track_total_hits": False,
            },
        )
        for h in resp["hits"]["hits"]:
            src = h["_source"]
            row = {f: src.get(f) for f in _SOURCE_FACT_FIELDS}
            row["id"] = str(row.get("id") or h["_id"])
            for f in ("occurred_start", "occurred_end", "mentioned_at"):
                row[f] = _parse_dt_field(row[f])
            rows.append(row)
    return rows


async def fetch_entities_for_units(
    conn: Any, fq_table: Callable[[str], str], unit_ids: list,
) -> list[dict]:
    """``_entity_rows_for_units_sql`` equivalent: rows
    ``{unit_id, entity_id, canonical_name}`` resolving direct unit_entities
    rows AND observation inheritance (an observation exposes the entities of
    its source_memory_ids)."""
    client = _client(conn)
    ids = [str(i) for i in unit_ids]
    if not ids:
        return []
    # observation inheritance map: obs -> sources
    obs_rows = await fetch_observation_sources(conn, fq_table("memory_units"), ids)
    sources_of = {r["id"]: r["source_memory_ids"] for r in obs_rows}
    lookup_ids = sorted(set(ids) | {s for sids in sources_of.values() for s in sids})

    entity_ids_by_unit: dict[str, list[str]] = {}
    for start in range(0, len(lookup_ids), 10000):
        resp = await client.search(
            index=_index(fq_table("unit_entities")),
            body={"query": {"terms": {"unit_id": lookup_ids[start:start + 10000]}},
                  "size": 10000, "_source": ["unit_id", "entity_id"],
                  "track_total_hits": False},
        )
        for h in resp["hits"]["hits"]:
            src = h["_source"]
            entity_ids_by_unit.setdefault(str(src["unit_id"]), []).append(str(src["entity_id"]))

    all_entities = sorted({e for es in entity_ids_by_unit.values() for e in es})
    name_of: dict[str, str] = {}
    for start in range(0, len(all_entities), 10000):
        resp = await client.search(
            index=_index(fq_table("entities")),
            body={"query": {"terms": {"id": all_entities[start:start + 10000]}},
                  "size": 10000, "_source": ["id", "canonical_name"],
                  "track_total_hits": False},
        )
        for h in resp["hits"]["hits"]:
            src = h["_source"]
            name_of[str(src["id"])] = src.get("canonical_name")

    out: list[dict] = []
    for uid in ids:
        direct = entity_ids_by_unit.get(uid, [])
        inherited = [e for sid in sources_of.get(uid, [])
                     for e in entity_ids_by_unit.get(sid, [])]
        seen: set[str] = set()
        for eid in direct + inherited:
            if eid in seen:
                continue
            seen.add(eid)
            out.append({"unit_id": uid, "entity_id": eid,
                        "canonical_name": name_of.get(eid)})
    return out


# ---------------------------------------------------------------------------
# @es_native — the dispatch decorator (plan-claude-decorators.md, no new file)
# ---------------------------------------------------------------------------

def es_native(native_fn: Callable) -> Callable:
    """Route a SQL-bodied coroutine to its ES-native twin when the backend is
    Elasticsearch; run the original otherwise.

    The dispatch key is discovered on the call arguments themselves: the
    first positional/keyword value exposing ``backend_type`` (conn, pool or
    backend), unwrapping one level of ``_pool``/``_backend``/``client``
    wrappers (BudgetedPool et al.). Signatures must match — the native twin
    receives the exact same ``*args, **kwargs``.

        @es_native(list_tags_paginated)
        async def _list_tags_from_table(self, conn, table, ...):
            ...  # SQL body, never runs under ES
    """
    import functools
    import inspect

    def _backend_type_of(value: Any) -> str | None:
        bt = getattr(value, "backend_type", None)
        if isinstance(bt, str):
            return bt
        for attr in ("_pool", "_backend", "pool", "backend"):
            inner = getattr(value, attr, None)
            bt = getattr(inner, "backend_type", None)
            if isinstance(bt, str):
                return bt
        return None

    def decorator(sql_fn: Callable) -> Callable:
        @functools.wraps(sql_fn)
        async def wrapper(*args: Any, **kwargs: Any):
            for value in list(args) + list(kwargs.values()):
                bt = _backend_type_of(value)
                if bt is not None:
                    if bt == "elasticsearch":
                        result = native_fn(*args, **kwargs)
                        return await result if inspect.isawaitable(result) else result
                    break
            return await sql_fn(*args, **kwargs)
        return wrapper

    return decorator
