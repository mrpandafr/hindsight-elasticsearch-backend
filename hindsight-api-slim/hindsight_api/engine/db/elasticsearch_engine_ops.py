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
