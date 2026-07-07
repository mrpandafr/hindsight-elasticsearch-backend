"""Elasticsearch-native retrieval arms.

Placement: hindsight_api/engine/search/retrieval_elasticsearch.py
(next to retrieval.py, which dispatches here when
``conn.backend_type == "elasticsearch"``).

Ports the SQL retrieval arms of retrieval.py to native Elasticsearch:

* semantic arm      -> ``conn.knn_search()``     (dense_vector kNN)
* BM25 arm          -> ``conn.text_search()``    (multi_match on text fields)
* temporal arm      -> kNN with the window predicate ported to bool/should,
                       then the SAME coverage selection and spreading logic
                       as the SQL version (formulas copied verbatim:
                       proximity from window midpoint, causal boosts 2.0/1.5,
                       propagation = parent * weight * boost * 0.7,
                       frontier gate combined > 0.2, weight floor 0.1,
                       per-source top-10, batch 20, max 5 iterations),
                       with links fetched from the memory_links index and
                       neighbor similarity computed by a script_score
                       cosineSimilarity query.

Output shape is identical to the SQL arms: dict[fact_type -> ...] of
``RetrievalResult.from_db_row(row)`` where each row carries the same columns
(id, text, context, event_date, occurred_start, occurred_end, mentioned_at,
fact_type, document_id, chunk_id, tags, metadata, proof_count) plus
``similarity`` / ``bm25_score``. Date fields are parsed back to aware
datetimes (ES returns ISO strings; the temporal math and downstream fusion
expect datetime objects, as the SQL driver provides).

Score mapping: ES kNN cosine ``_score`` is ``(1 + cos) / 2``, so
``similarity = 2 * _score - 1`` — the same [~0..1] cosine similarity the SQL
``1 - (embedding <=> $1)`` produces. Retrieval-level floors (min_semantic /
min_keyword) are applied client-side on the mapped values.

Known divergences (documented, not hidden):

* ``created_after/created_before`` filter on ``created_at`` (the field the
  ES schema tracks); the SQL arms filter ``updated_at``. Facts are
  append-mostly, so the practical difference is limited to re-consolidated
  units.
* ``tag_groups`` are translated when they expose ``tags``/``match`` (the
  TagGroup shape); anything else is skipped with a one-time warning rather
  than mistranslated.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from ...config import get_config
from ..db.ops_elasticsearch import _index, _parse_vector
from ..memory_engine import fq_table
from .graph_retrieval import GraphRetriever

logger = logging.getLogger(__name__)


# Mirrors of retrieval.py's temporal tuning constants (kept in sync lazily at
# call time via getattr on the retrieval module, falling back to these).
_TEMPORAL_POOL_SIZE = 60
_TEMPORAL_ENTRY_POINTS = 10
_TEMPORAL_COVERAGE_BUCKETS = 8

_ROW_FIELDS = (
    "id", "text", "context", "event_date", "occurred_start", "occurred_end",
    "mentioned_at", "fact_type", "document_id", "chunk_id", "tags",
    "metadata", "proof_count",
)
_DATE_FIELDS = ("event_date", "occurred_start", "occurred_end", "mentioned_at")
_SPREAD_LINK_TYPES = ["temporal", "causes", "caused_by", "enables", "prevents"]


# ---------------------------------------------------------------------------
# Row plumbing
# ---------------------------------------------------------------------------

def _parse_dt(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _to_row(esrow: Any) -> dict[str, Any]:
    """ESRow -> the dict shape RetrievalResult.from_db_row consumes."""
    row = {f: esrow.get(f) for f in _ROW_FIELDS}
    for f in _DATE_FIELDS:
        row[f] = _parse_dt(row[f])
    if row.get("tags") is None:
        row["tags"] = []
    if row.get("proof_count") is None:
        row["proof_count"] = 0
    return row


def _knn_similarity(score: float | None) -> float:
    """ES cosine kNN _score = (1 + cos) / 2  ->  cos, like `1 - (emb <=> q)`."""
    return 2.0 * float(score or 0.0) - 1.0


# ---------------------------------------------------------------------------
# Filter translation (tags / tag groups / created range)
# ---------------------------------------------------------------------------
# Exact port of tags.py semantics (see build_tags_where_clause_simple and
# _build_group_clause):
#   any        -> overlap  OR untagged
#   all        -> superset OR untagged
#   any_strict -> overlap,  untagged excluded
#   all_strict -> superset, untagged excluded
#   exact      -> set equality; empty scope matches ONLY untagged rows
# "Untagged" (SQL: tags IS NULL OR tags = '{}') is ES "field missing":
# empty arrays are indexed as no value, so must_not exists covers both.

_UNTAGGED = {"bool": {"must_not": [{"exists": {"field": "tags"}}]}}


def _tags_query(tags, match) -> dict | None:
    """ES query for one leaf's semantics. None = no filtering."""
    match = str(match)
    if match == "exact":
        distinct = sorted(set(tags or []))
        if not distinct:
            return _UNTAGGED  # empty scope = global/untagged only
        # @> AND <@ == set equality: superset via term-per-tag, subset via
        # cardinality (tags are de-facto unique per unit).
        return {"bool": {"filter": [{"term": {"tags": t}} for t in distinct] + [
            {"script": {"script": {
                "source": "doc['tags'].size() == params.n",
                "params": {"n": len(distinct)},
            }}},
        ]}}
    if not tags:
        return None
    is_any = match in ("any", "any_strict")
    is_all = match in ("all", "all_strict")
    include_untagged = match not in ("any_strict", "all_strict")
    # Unknown modes: _parse_tags_match falls back to ("&&", True) — overlap
    # with untagged included — so anything that isn't an "all" variant uses
    # the overlap core.
    if is_all:
        core: dict = {"bool": {"filter": [{"term": {"tags": t}} for t in tags]}}
    else:  # any / any_strict / unknown fallback
        core = {"terms": {"tags": list(tags)}}
    if include_untagged:
        return {"bool": {"should": [_UNTAGGED, core], "minimum_should_match": 1}}
    return core


def _group_query(group) -> dict | None:
    """Recursive ES translation of a TagGroup (Leaf / And / Or / Not).

    Accepts the pydantic models and their dict aliases ({"and": [...]},
    {"or": [...]}, {"not": {...}}, {"tags": [...], "match": ...}).
    """
    if isinstance(group, dict):
        if "tags" in group:
            return _tags_query(group["tags"], group.get("match", "any_strict"))
        if "and" in group:
            children = [q for q in (_group_query(c) for c in group["and"]) if q]
            return {"bool": {"filter": children}} if children else None
        if "or" in group:
            children = [q for q in (_group_query(c) for c in group["or"]) if q]
            return {"bool": {"should": children, "minimum_should_match": 1}} if children else None
        if "not" in group:
            child = _group_query(group["not"])
            return {"bool": {"must_not": [child]}} if child else None
        return None
    if hasattr(group, "tags"):  # TagGroupLeaf (default match: any_strict)
        return _tags_query(group.tags, getattr(group, "match", "any_strict"))
    if hasattr(group, "filters"):  # TagGroupAnd / TagGroupOr
        children = [q for q in (_group_query(c) for c in group.filters) if q]
        if not children:
            return None
        if "Or" in type(group).__name__:
            return {"bool": {"should": children, "minimum_should_match": 1}}
        return {"bool": {"filter": children}}
    if hasattr(group, "filter"):  # TagGroupNot
        child = _group_query(group.filter)
        return {"bool": {"must_not": [child]}} if child else None
    logger.warning("Unrecognized TagGroup shape %r — group skipped", type(group).__name__)
    return None


def _tag_filters(tags, tags_match, tag_groups) -> list[dict]:
    filters: list[dict] = []
    # exact + empty scope filters even without tags (global/untagged scope)
    if tags or str(tags_match) == "exact":
        q = _tags_query(tags, tags_match)
        if q is not None:
            filters.append(q)
    for group in tag_groups or []:
        q = _group_query(group)
        if q is not None:
            filters.append(q)
    return filters


def _created_range_filters(created_after, created_before) -> list[dict]:
    filters: list[dict] = []
    if created_after is not None:
        filters.append({"range": {"created_at": {"gt": _iso(created_after)}}})
    if created_before is not None:
        filters.append({"range": {"created_at": {"lt": _iso(created_before)}}})
    return filters


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Semantic + BM25 combined (native port of retrieve_semantic_bm25_combined)
# ---------------------------------------------------------------------------

async def retrieve_semantic_bm25_combined_es(
    conn,
    query_emb_str: str,
    query_text: str,
    bank_id: str,
    fact_types: list[str],
    limit: int,
    tags=None,
    tags_match="any",
    tag_groups=None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    min_semantic: float | None = None,
    min_keyword: float | None = None,
):
    """Native ES equivalent, same return shape:
    dict[fact_type -> (semantic RetrievalResults, bm25 RetrievalResults)].

    Fast path: ONE ``msearch`` round trip carrying every arm (a kNN and a
    BM25 sub-search per fact_type) — the network-level analogue of the SQL
    version's single UNION ALL statement. Falls back to per-fact_type
    ``knn_search``/``text_search`` calls if the cluster or a test double
    doesn't accept the batched form.
    """
    from .retrieval import tokenize_query  # lazy: avoids circular import
    from .types import RetrievalResult

    config = get_config()
    sem_min = min_semantic if min_semantic is not None else config.semantic_min_similarity
    bm25_min = min_keyword if min_keyword is not None else config.bm25_min_score
    tokens = tokenize_query(query_text)

    table = fq_table("memory_units")
    extra = _tag_filters(tags, tags_match, tag_groups) + _created_range_filters(
        created_after, created_before
    )
    # HNSW over-fetch parity: SQL over-fetches 5x (min 100) then trims;
    # ES expresses the same idea as num_candidates.
    num_candidates = max(limit * 5, 100)

    def _convert_sem(rows) -> list:
        out = []
        for r in rows:
            similarity = _knn_similarity(r.get("score"))
            if similarity < sem_min:
                continue
            row = _to_row(r)
            row["similarity"] = similarity
            row["bm25_score"] = None
            out.append(RetrievalResult.from_db_row(row))
        return out

    def _convert_bm25(rows) -> list:
        out = []
        for r in rows:
            score = float(r.get("score") or 0.0)
            if score <= bm25_min:
                continue
            row = _to_row(r)
            row["similarity"] = None
            row["bm25_score"] = score
            out.append(RetrievalResult.from_db_row(row))
        return out

    # --- fast path: every arm in one msearch round trip ---
    try:
        from ..db.elasticsearch import (
            _SEARCH_SOURCE_FIELDS,
            _bank_filters,
            _hits_to_rows,
        )

        vector = _parse_vector(query_emb_str)
        idx = _index(table)
        searches: list[dict] = []
        plan: list[tuple[str, str]] = []
        for ft in fact_types:
            searches.append({"index": idx})
            searches.append({
                "knn": {
                    "field": "embedding",
                    "query_vector": vector,
                    "k": limit,
                    "num_candidates": num_candidates,
                    "filter": _bank_filters(bank_id, ft, extra or None),
                },
                "size": limit,
                "_source": _SEARCH_SOURCE_FIELDS,
                "track_total_hits": False,
            })
            plan.append((ft, "sem"))
            if tokens:
                searches.append({"index": idx})
                body: dict[str, Any] = {
                    "query": {"bool": {
                        "must": [{"multi_match": {
                            "query": query_text,
                            "fields": ["text^2", "context", "text_signals"],
                            "type": "best_fields",
                        }}],
                        "filter": _bank_filters(bank_id, ft, extra or None),
                    }},
                    "size": limit,
                    "_source": _SEARCH_SOURCE_FIELDS,
                    "track_total_hits": False,
                }
                if bm25_min > 0:
                    body["min_score"] = bm25_min  # server-side floor
                searches.append(body)
                plan.append((ft, "bm25"))

        resp = await conn.client.msearch(searches=searches)
        responses = resp["responses"]
        if len(responses) != len(plan):
            raise RuntimeError("msearch response count mismatch")
        result: dict[str, tuple[list, list]] = {ft: ([], []) for ft in fact_types}
        for (ft, kind), sub in zip(plan, responses):
            if sub.get("error"):
                raise RuntimeError(f"msearch arm failed: {sub['error']}")
            if kind == "sem":
                rows = _hits_to_rows(sub, "semantic")
                result[ft] = (_convert_sem(rows), result[ft][1])
            else:
                rows = _hits_to_rows(sub, "bm25")
                result[ft] = (result[ft][0], _convert_bm25(rows))
        return result
    except Exception as exc:
        logger.debug("msearch fast path unavailable (%s); per-fact_type fallback", exc)

    # --- fallback: per-fact_type native calls (2 round trips per ft) ---
    async def _one_ft(ft: str):
        sem_rows = await conn.knn_search(
            table, query_emb_str, bank_id,
            fact_type=ft, k=limit, num_candidates=num_candidates,
            extra_filters=extra or None,
        )
        bm_rows = []
        if tokens:
            bm_rows = await conn.text_search(
                table, query_text, bank_id,
                fact_type=ft, limit=limit, extra_filters=extra or None,
            )
        return ft, _convert_sem(sem_rows), _convert_bm25(bm_rows)

    results = await asyncio.gather(*(_one_ft(ft) for ft in fact_types))
    return {ft: (sem, bm) for ft, sem, bm in results}


# ---------------------------------------------------------------------------
# Temporal combined (native port of retrieve_temporal_combined)
# ---------------------------------------------------------------------------

def _window_filter(start_iso: str, end_iso: str) -> dict:
    """Exact port of the SQL window predicate:
    (occurred_start<=end AND occurred_end>=start) OR mentioned_at IN window
    OR occurred_start IN window OR occurred_end IN window.
    """
    return {"bool": {"should": [
        {"bool": {"filter": [
            {"range": {"occurred_start": {"lte": end_iso}}},
            {"range": {"occurred_end": {"gte": start_iso}}},
        ]}},
        {"range": {"mentioned_at": {"gte": start_iso, "lte": end_iso}}},
        {"range": {"occurred_start": {"gte": start_iso, "lte": end_iso}}},
        {"range": {"occurred_end": {"gte": start_iso, "lte": end_iso}}},
    ], "minimum_should_match": 1}}


def _best_date(row: dict) -> datetime | None:
    """COALESCE logic of the SQL version, midpoints included."""
    os_, oe, ma = row.get("occurred_start"), row.get("occurred_end"), row.get("mentioned_at")
    if os_ is not None and oe is not None:
        return os_ + (oe - os_) / 2
    if os_ is not None:
        return os_
    if oe is not None:
        return oe
    return ma


def _proximity(best: datetime | None, mid: datetime, total_days: float, default: float) -> float:
    if not best:
        return default
    if best.tzinfo is None:
        best = best.replace(tzinfo=UTC)
    days_from_mid = abs((best - mid).total_seconds() / 86400)
    return 1.0 - min(days_from_mid / (total_days / 2), 1.0) if total_days > 0 else 1.0


async def retrieve_temporal_combined_es(
    conn,
    query_emb_str: str,
    bank_id: str,
    fact_types: list[str],
    start_date: datetime,
    end_date: datetime,
    budget: int,
    semantic_threshold: float = 0.1,
    tags=None,
    tags_match="any",
    tag_groups=None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
):
    """Native ES equivalent, same return shape:
    dict[fact_type -> list[RetrievalResult]] with temporal_score /
    temporal_proximity set exactly as the SQL version computes them.
    """
    from . import retrieval as _r  # lazy: shared constants + coverage selection
    from .types import RetrievalResult

    if not fact_types:
        return {}
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=UTC)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=UTC)
    start_iso, end_iso = _iso(start_date), _iso(end_date)

    pool_size = getattr(_r, "_TEMPORAL_POOL_SIZE", _TEMPORAL_POOL_SIZE)
    entry_limit = getattr(_r, "_TEMPORAL_ENTRY_POINTS", _TEMPORAL_ENTRY_POINTS)
    buckets = getattr(_r, "_TEMPORAL_COVERAGE_BUCKETS", _TEMPORAL_COVERAGE_BUCKETS)

    table = fq_table("memory_units")
    base_extra = _tag_filters(tags, tags_match, tag_groups) + _created_range_filters(
        created_after, created_before
    )
    window = _window_filter(start_iso, end_iso)

    # --- Entry-point pools: similarity-ranked, window-filtered kNN per ft ---
    async def _pool(ft: str):
        rows = await conn.knn_search(
            table, query_emb_str, bank_id,
            fact_type=ft, k=pool_size, num_candidates=max(pool_size * 5, 200),
            extra_filters=base_extra + [window],
        )
        out = []
        for r in rows:
            similarity = _knn_similarity(r.get("score"))
            if similarity < semantic_threshold:
                continue
            row = _to_row(r)
            row["similarity"] = similarity
            out.append(row)
        return ft, out

    pool_by_ft = dict(
        (ft, rows) for ft, rows in await asyncio.gather(*(_pool(ft) for ft in fact_types))
    )
    if not any(pool_by_ft.values()):
        return {ft: [] for ft in fact_types}

    entries_by_ft = {
        ft: _r._select_with_temporal_coverage(rows, start_date, end_date, entry_limit, buckets)
        for ft, rows in pool_by_ft.items()
    }

    total_days = (end_date - start_date).total_seconds() / 86400
    mid_date = start_date + (end_date - start_date) / 2
    query_vec = _parse_vector(query_emb_str)
    client = conn.client
    ml_idx = _index(fq_table("memory_links"))
    mu_idx = _index(table)

    results_by_ft: dict[str, list] = {}

    for ft in fact_types:
        ft_entry_points = entries_by_ft.get(ft, [])
        if not ft_entry_points:
            results_by_ft[ft] = []
            continue

        results = []
        visited: set[str] = set()
        node_scores: dict[str, tuple[float, float]] = {}

        for ep in ft_entry_points:
            unit_id = str(ep["id"])
            visited.add(unit_id)
            temporal_proximity = _proximity(_best_date(ep), mid_date, total_days, default=0.5)
            row = dict(ep)
            row.setdefault("bm25_score", None)
            ep_result = RetrievalResult.from_db_row(row)
            ep_result.temporal_score = temporal_proximity
            ep_result.temporal_proximity = temporal_proximity
            results.append(ep_result)
            node_scores[unit_id] = (ep["similarity"], 1.0)

        # --- Spreading: same tuning as SQL (batch 20, top-10/source, w>=0.1,
        #     5 iterations max, budget-bounded) with native link + score queries.
        frontier = list(node_scores.keys())
        budget_remaining = budget - len(ft_entry_points)
        batch_size, per_source_limit, max_iterations = 20, 10, 5
        iteration = 0

        while frontier and budget_remaining > 0 and iteration < max_iterations:
            iteration += 1
            batch_ids = frontier[:batch_size]
            frontier = frontier[batch_size:]

            # LATERAL top-K per source -> one links query, per-source cap client-side
            resp = await client.search(
                index=ml_idx,
                body={
                    "query": {"bool": {"filter": [
                        {"terms": {"from_unit_id": batch_ids}},
                        {"terms": {"link_type": _SPREAD_LINK_TYPES}},
                        {"range": {"weight": {"gte": 0.1}}},
                    ]}},
                    "sort": [{"weight": "desc"}],
                    "size": len(batch_ids) * per_source_limit * 3,
                    "_source": ["from_unit_id", "to_unit_id", "weight", "link_type"],
                    "track_total_hits": False,
                },
            )
            per_source: dict[str, list[dict]] = {}
            for h in resp["hits"]["hits"]:
                src = h["_source"]
                bucket = per_source.setdefault(str(src["from_unit_id"]), [])
                if len(bucket) < per_source_limit:
                    bucket.append(src)
            links = [l for bucket in per_source.values() for l in bucket]
            if not links:
                continue

            # Hydrate + similarity-score the neighbor candidates in one query
            candidate_ids = sorted({str(l["to_unit_id"]) for l in links})
            nresp = await client.search(
                index=mu_idx,
                body={
                    "query": {"script_score": {
                        "query": {"bool": {"filter": [
                            {"terms": {"id": candidate_ids}},
                            {"term": {"bank_id": bank_id}},
                            {"term": {"fact_type": ft}},
                            {"exists": {"field": "embedding"}},
                        ] + base_extra}},
                        "script": {
                            # cosineSimilarity in [-1,1]; +1 keeps ES scores
                            # non-negative. similarity = _score - 1.
                            "source": "cosineSimilarity(params.qv, 'embedding') + 1.0",
                            "params": {"qv": query_vec},
                        },
                    }},
                    # server-side floor on the shifted score; the client-side
                    # threshold check below stays as a safety net.
                    "min_score": semantic_threshold + 1.0,
                    "size": len(candidate_ids),
                    "_source": list(_ROW_FIELDS),
                    "track_total_hits": False,
                },
            )
            neighbor_rows: dict[str, dict] = {}
            for h in nresp["hits"]["hits"]:
                similarity = float(h.get("_score", 0.0)) - 1.0
                if similarity < semantic_threshold:
                    continue
                row = _to_row(h["_source"])
                row["similarity"] = similarity
                neighbor_rows[str(row["id"])] = row

            # One (source, target) pair per link, like the SQL row stream
            for link in links:
                if budget_remaining <= 0:
                    break
                neighbor_id = str(link["to_unit_id"])
                row = neighbor_rows.get(neighbor_id)
                if row is None or neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                budget_remaining -= 1

                _, parent_temporal_score = node_scores.get(
                    str(link["from_unit_id"]), (0.5, 0.5)
                )
                neighbor_temporal_proximity = _proximity(
                    _best_date(row), mid_date, total_days, default=0.3
                )
                link_type = link["link_type"]
                if link_type in ("causes", "caused_by"):
                    causal_boost = 2.0
                elif link_type in ("enables", "prevents"):
                    causal_boost = 1.5
                else:
                    causal_boost = 1.0
                propagated = parent_temporal_score * float(link["weight"]) * causal_boost * 0.7
                combined_temporal = max(neighbor_temporal_proximity, propagated)

                out = dict(row)
                out["weight"] = float(link["weight"])
                out["link_type"] = link_type
                out.setdefault("bm25_score", None)
                neighbor_result = RetrievalResult.from_db_row(out)
                neighbor_result.temporal_score = combined_temporal
                neighbor_result.temporal_proximity = neighbor_temporal_proximity
                results.append(neighbor_result)

                if budget_remaining > 0 and combined_temporal > 0.2:
                    node_scores[neighbor_id] = (row["similarity"], combined_temporal)
                    frontier.append(neighbor_id)

        results_by_ft[ft] = results

    return results_by_ft


# ---------------------------------------------------------------------------
# Native graph retriever (4th arm)
# ---------------------------------------------------------------------------

class ElasticsearchGraphRetriever(GraphRetriever):
    """GraphRetriever implementation backed by the ES ops expansion methods.

    Same contract as LinkExpansionRetriever (graph_retrieval.GraphRetriever):
    ``retrieve(pool, ...) -> (list[RetrievalResult], GraphRetrievalTimings)``.
    Registered by retrieval.py when the pool is Elasticsearch and no explicit
    retriever was passed.

    Pipeline:
      1. seeds  — provided semantic/temporal seeds when available, else a
         kNN entry-point search (tag/created filters applied at query time);
      2. expand — ops.expand_semantic_causal (link weights) and
         ops.expand_entities (shared-entity counts), the native equivalents
         of the SQL expansion CTEs;
      3. score  — activation in [0, 1]: link-based rows keep their weight,
         entity rows get overlap_count / max_overlap; a unit reached several
         ways keeps its max;
      4. filter — tags / tag_groups post-filtered with tags.py's own
         filter_results_by_tags / filter_results_by_tag_groups (the exact
         code path SQL graph results go through), then budget-capped by
         activation. created_after/before are enforced at seed time only:
         expansion rows don't carry created_at (documented divergence).
    """

    @property
    def name(self) -> str:
        return "es_link_expansion"

    async def retrieve(
        self,
        pool,
        query_embedding_str: str,
        bank_id: str,
        fact_type: str,
        budget: int,
        query_text: str | None = None,
        semantic_seeds=None,
        temporal_seeds=None,
        adjacency=None,
        tags=None,
        tags_match="any",
        tag_groups=None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ):
        import time as _time

        from ..db.ops_elasticsearch import ElasticsearchOps
        from .tags import filter_results_by_tag_groups, filter_results_by_tags
        from .types import GraphRetrievalTimings, RetrievalResult

        timings = GraphRetrievalTimings(fact_type=fact_type)
        t0 = _time.time()
        ops = ElasticsearchOps()
        mu_table = fq_table("memory_units")
        ml_table = fq_table("memory_links")
        ue_table = fq_table("unit_entities")

        async with pool.acquire() as conn:
            # 1. seeds
            seeds_start = _time.time()
            seed_ids: list[str] = []
            for provided in (semantic_seeds, temporal_seeds):
                if provided:
                    seed_ids.extend(str(r.id) for r in provided[:10])
            if not seed_ids:
                extra = _tag_filters(tags, tags_match, tag_groups) + \
                    _created_range_filters(created_after, created_before)
                seed_rows = await conn.knn_search(
                    mu_table, query_embedding_str, bank_id,
                    fact_type=fact_type, k=10, num_candidates=200,
                    extra_filters=extra or None,
                )
                seed_ids = [str(r["id"]) for r in seed_rows]
                timings.db_queries += 1
            timings.seeds_time = _time.time() - seeds_start
            if not seed_ids:
                timings.traverse = _time.time() - t0
                return [], timings

            # 2. expansion (native equivalents of the SQL expansion CTEs) —
            #    the two expansions are independent: run them concurrently.
            #    The graph arm is auxiliary: a failure here degrades to an
            #    empty arm (warning logged) instead of killing the whole
            #    recall gather in retrieve_all_fact_types_parallel.
            try:
                (sem_rows, causal_rows), ent_rows = await asyncio.gather(
                    ops.expand_semantic_causal(
                        conn, ml_table, mu_table, seed_ids, fact_type, budget
                    ),
                    ops.expand_entities(
                        conn, mu_table, ue_table, seed_ids, fact_type, budget,
                        per_entity_limit=5,
                    ),
                )
            except Exception:
                logger.warning(
                    "graph expansion failed on the Elasticsearch backend; "
                    "returning an empty graph arm", exc_info=True,
                )
                timings.traverse = _time.time() - t0
                return [], timings
            timings.db_queries += 3
            timings.edge_count = len(sem_rows) + len(causal_rows) + len(ent_rows)

            # 3. activation scoring, max-merged per unit
            max_overlap = max((r["score"] for r in ent_rows), default=1.0) or 1.0
            best: dict[str, tuple[float, Any]] = {}

            def _consider(row, activation: float) -> None:
                uid = str(row["id"])
                if uid in seed_ids:
                    return
                if uid not in best or activation > best[uid][0]:
                    best[uid] = (activation, row)

            for r in sem_rows + causal_rows:
                _consider(r, max(0.0, min(1.0, float(r["score"]))))
            for r in ent_rows:
                _consider(r, float(r["score"]) / max_overlap)

            results = []
            for activation, row in sorted(best.values(), key=lambda x: -x[0]):
                out = _to_row(row)
                out["activation"] = activation
                results.append(RetrievalResult.from_db_row(out))

        # 4. post-filtering with the shared tags.py code path, then budget cap
        results = filter_results_by_tags(results, tags, match=tags_match)
        if tag_groups:
            results = filter_results_by_tag_groups(results, tag_groups)
        results = results[:budget]

        timings.result_count = len(results)
        timings.traverse = _time.time() - t0
        return results, timings
