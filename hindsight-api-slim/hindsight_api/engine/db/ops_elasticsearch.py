"""Elasticsearch implementation of DataAccessOps.

Drop-in backend for ``hindsight_api.engine.db.ops.DataAccessOps`` that targets
an Elasticsearch cluster (8.x / 9.x) instead of a SQL database.

Concept mapping (SQL -> Elasticsearch)
--------------------------------------
=====================================  ==========================================
SQL / PostgreSQL concept               Elasticsearch equivalent used here
=====================================  ==========================================
table name (``schema.table``)          index name (``schema-table``, lowercased)
DatabaseConnection                     AsyncElasticsearch client (or an object
                                       exposing it via a ``.client`` attribute)
INSERT ... unnest() batch              ``_bulk`` API
ON CONFLICT DO NOTHING                 ``op_type=create`` + deterministic ``_id``
ON CONFLICT DO UPDATE (upsert)         bulk ``update`` with ``doc_as_upsert``
row lock / FOR UPDATE SKIP LOCKED      optimistic concurrency control
                                       (``if_seq_no`` / ``if_primary_term``)
RETURNING id                           client-side UUID generation
ANY($1::uuid[])                        ``terms`` query
CROSS JOIN LATERAL fan-out             ``msearch`` (one sub-search per row)
GROUP BY / DISTINCT ON                 aggregations + client-side dedup
unnest(tags)                           ``terms`` aggregation on keyword field
tsvector full-text search              native ``match`` on ``text`` fields
pgvector ``vector`` column             ``dense_vector`` field (kNN)
transaction atomicity                  best-effort: deterministic ids,
                                       ``refresh="wait_for"``, OCC retries
=====================================  ==========================================

Methods of the ABC that *return SQL fragments to be embedded in a larger SQL
query* (``build_entity_expansion_cte``, ``build_semantic_causal_cte``,
``build_tag_listing_parts``) cannot be honoured by a non-SQL backend: there is
no SQL engine to execute the fragment. They raise ``NotImplementedError`` and
this class exposes executable native equivalents instead:

* ``expand_entities()``          <-> build_entity_expansion_cte
* ``expand_semantic_causal()``   <-> build_semantic_causal_cte
* ``list_tags()``                <-> build_tag_listing_parts

Every other abstract method keeps the exact signature, argument semantics and
return types of the ABC (same shapes as the PostgreSQL implementation:
``list[ResultRow]``-like rows supporting ``row["col"]`` access, id lists as
``list[str]``, counts as ``int``...).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

try:  # inside the hindsight package
    from .base import DatabaseConnection  # noqa: F401  (typing only)
    from .ops import DataAccessOps, TagListingParts
    from .result import ResultRow  # noqa: F401  (typing only)
except ImportError:  # standalone usage
    DatabaseConnection = Any  # type: ignore

    class TagListingParts:  # type: ignore
        def __init__(self, tag_source, non_empty_check, tag_col, bank_prefix):
            self.tag_source = tag_source
            self.non_empty_check = non_empty_check
            self.tag_col = tag_col
            self.bank_prefix = bank_prefix

    class DataAccessOps:  # type: ignore
        def _get_mu_table(self) -> str:
            return "public.memory_units"


class ESRow(dict):
    """Duck-typed ResultRow: a mapping supporting ``row["col"]`` access."""

    __slots__ = ()


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _es(conn: Any):
    """Extract the AsyncElasticsearch client from whatever ``conn`` is."""
    return getattr(conn, "client", conn)


def _index(table: str) -> str:
    """Map a fully-qualified SQL table name to an ES index name.

    ``public.memory_units`` -> ``public-memory_units`` (ES index names must be
    lowercase and must not contain ``.`` in a leading position).
    """
    return table.replace('"', "").replace(".", "-").lower()


def _iso(value: Any) -> Any:
    """Serialize datetimes to ISO-8601 for ES; pass through everything else."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return value


def _parse_vector(embedding: Any) -> list[float] | None:
    """Accept pgvector text form ``'[0.1,0.2]'``, JSON, or a float list."""
    if embedding is None:
        return None
    if isinstance(embedding, (list, tuple)):
        return [float(x) for x in embedding]
    s = str(embedding).strip()
    if not s:
        return None
    return [float(x) for x in s.strip("[]() ").split(",") if x.strip()]


def _load_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _epoch_hours_diff(a: str | datetime, b: str | datetime) -> float:
    da = a if isinstance(a, datetime) else datetime.fromisoformat(str(a).replace("Z", "+00:00"))
    db = b if isinstance(b, datetime) else datetime.fromisoformat(str(b).replace("Z", "+00:00"))
    if da.tzinfo is None:
        da = da.replace(tzinfo=timezone.utc)
    if db.tzinfo is None:
        db = db.replace(tzinfo=timezone.utc)
    return abs((da - db).total_seconds()) / 3600.0


_MU_SOURCE_FIELDS = [
    "id", "text", "context", "event_date", "occurred_start", "occurred_end",
    "mentioned_at", "fact_type", "document_id", "chunk_id", "tags",
    "proof_count",
]

_CAUSAL_LINK_TYPES = ["causes", "caused_by", "enables", "prevents"]


def _mu_row(hit: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> ESRow:
    src = hit.get("_source", {})
    row = ESRow({f: src.get(f) for f in _MU_SOURCE_FIELDS})
    row["id"] = src.get("id", hit.get("_id"))
    if extra:
        row.update(extra)
    return row


async def _scan(client, index: str, query: dict, source: list[str] | bool = True,
                sort: list | None = None, page_size: int = 1000):
    """search_after pagination over all hits matching ``query``."""
    sort = sort or [{"_shard_doc": "asc"}]
    # _shard_doc requires a PIT; fall back to _id tiebreaker without one.
    if sort == [{"_shard_doc": "asc"}]:
        sort = [{"_id": "asc"}]
    search_after = None
    while True:
        body: dict[str, Any] = {"query": query, "size": page_size, "sort": sort,
                                "_source": source, "track_total_hits": False}
        if search_after is not None:
            body["search_after"] = search_after
        resp = await client.search(index=index, body=body)
        hits = resp["hits"]["hits"]
        if not hits:
            return
        for h in hits:
            yield h
        search_after = hits[-1]["sort"]


class ElasticsearchOps(DataAccessOps):
    """Elasticsearch-specific data access operations.

    ``conn`` arguments must be (or wrap, via a ``.client`` attribute) an
    ``elasticsearch.AsyncElasticsearch`` instance.
    """

    #: writes that are read back within the same logical operation must be
    #: visible immediately; ``wait_for`` avoids a hard index refresh storm.
    REFRESH = "wait_for"

    @property
    def uses_observation_sources_table(self) -> bool:
        # Like PG: source_memory_ids is stored inline (keyword array) on the
        # memory_units documents, no junction index needed.
        return False

    # -- Bulk insert operations ------------------------------------------

    async def bulk_upsert_chunks(
        self,
        conn: DatabaseConnection,
        table: str,
        chunk_ids: list[str],
        document_ids: list[str],
        bank_ids: list[str],
        chunk_texts: list[str],
        chunk_indices: list[int],
        content_hashes: list[str],
    ) -> None:
        """ON CONFLICT (chunk_id) DO UPDATE -> bulk update + doc_as_upsert."""
        if not chunk_ids:
            return
        client, idx = _es(conn), _index(table)
        ops: list[dict] = []
        for cid, did, bid, text, cidx, chash in zip(
            chunk_ids, document_ids, bank_ids, chunk_texts, chunk_indices, content_hashes
        ):
            ops.append({"update": {"_index": idx, "_id": cid}})
            ops.append({
                "doc": {
                    "chunk_id": cid,
                    "document_id": did,
                    "bank_id": bid,
                    "chunk_text": text,
                    "chunk_index": cidx,
                    "content_hash": chash,
                },
                "doc_as_upsert": True,
            })
        resp = await client.bulk(operations=ops, refresh=self.REFRESH)
        _raise_on_bulk_errors(resp, ignore_statuses=())

    async def lock_document_for_write(
        self,
        conn: DatabaseConnection,
        table: str,
        doc_id: str,
        bank_id: str,
    ) -> str | None:
        """Create-or-read the document row and return its prior content_hash.

        ES has no row locks. Equivalent guarantees are provided by:

        * ``op_type=create`` on the deterministic id ``{bank_id}::{doc_id}``:
          exactly one concurrent writer wins the *creation* race and sees
          ``'__pending__'``;
        * every other writer reads the stored hash back.

        Serialization of *subsequent* writes must be done by the caller with
        optimistic concurrency (``if_seq_no``/``if_primary_term`` on the
        returned document), which is the idiomatic ES replacement for
        ``SELECT ... FOR UPDATE``.
        """
        client, idx = _es(conn), _index(table)
        es_id = f"{bank_id}::{doc_id}"
        try:
            await client.index(
                index=idx,
                id=es_id,
                op_type="create",
                document={
                    "id": doc_id,
                    "bank_id": bank_id,
                    "original_text": "",
                    "content_hash": "__pending__",
                },
                refresh=self.REFRESH,
            )
            return "__pending__"
        except Exception as exc:  # version_conflict_engine_exception (409)
            if _status_of(exc) != 409:
                raise
        try:
            got = await client.get(index=idx, id=es_id, source_includes=["content_hash"])
        except Exception as exc:
            if _status_of(exc) == 404:
                return None
            raise
        return got.get("_source", {}).get("content_hash")

    async def insert_facts_batch(
        self,
        conn: DatabaseConnection,
        bank_id: str,
        fact_texts: list[str],
        embeddings: list[str],
        event_dates: list,
        occurred_starts: list,
        occurred_ends: list,
        mentioned_ats: list,
        contexts: list[str],
        fact_types: list[str],
        metadata_jsons: list[str],
        chunk_ids: list,
        document_ids: list,
        tags_list: list[str],
        observation_scopes_list: list,
        text_signals_list: list,
        text_search_extension: str = "native",
    ) -> list[str]:
        """Batch-insert facts into the memory_units index, returning new IDs.

        ``RETURNING id`` is emulated by generating UUIDs client-side (used as
        both the ES ``_id`` and the ``id`` field). ``search_vector`` has no ES
        equivalent: full-text search is served natively by the ``text``,
        ``context`` and ``text_signals`` mapped fields, so
        ``text_search_extension`` is accepted and ignored.
        """
        if not fact_texts:
            return []
        client = _es(conn)
        idx = _index(self._get_mu_table())
        new_ids = [str(uuid.uuid4()) for _ in fact_texts]
        ops: list[dict] = []
        for i, fact_id in enumerate(new_ids):
            tags = _load_json(tags_list[i] if i < len(tags_list) else None, [])
            scopes = _load_json(
                observation_scopes_list[i] if i < len(observation_scopes_list) else None, None
            )
            doc = {
                "id": fact_id,
                "bank_id": bank_id,
                "text": fact_texts[i],
                "embedding": _parse_vector(embeddings[i] if i < len(embeddings) else None),
                "event_date": _iso(event_dates[i] if i < len(event_dates) else None),
                "occurred_start": _iso(occurred_starts[i] if i < len(occurred_starts) else None),
                "occurred_end": _iso(occurred_ends[i] if i < len(occurred_ends) else None),
                "mentioned_at": _iso(mentioned_ats[i] if i < len(mentioned_ats) else None),
                "context": contexts[i] if i < len(contexts) else None,
                "fact_type": fact_types[i] if i < len(fact_types) else None,
                "metadata": _load_json(metadata_jsons[i] if i < len(metadata_jsons) else None, {}),
                "chunk_id": chunk_ids[i] if i < len(chunk_ids) else None,
                "document_id": document_ids[i] if i < len(document_ids) else None,
                "tags": tags if isinstance(tags, list) else [],
                "observation_scopes": scopes,
                "text_signals": text_signals_list[i] if i < len(text_signals_list) else None,
                "proof_count": 0,
                "source_memory_ids": None,
                "created_at": _now_iso(),
            }
            ops.append({"index": {"_index": idx, "_id": fact_id}})
            ops.append({k: v for k, v in doc.items() if v is not None or k in ("context",)})
        resp = await client.bulk(operations=ops, refresh=self.REFRESH)
        _raise_on_bulk_errors(resp, ignore_statuses=())
        return new_ids

    async def bulk_insert_links(
        self,
        conn: DatabaseConnection,
        table: str,
        sorted_links: list[tuple],
        bank_id: str,
        nil_entity_uuid: str,
        exists_clause: str,
        chunk_size: int = 5000,
    ) -> None:
        """Bulk insert memory_links with ON CONFLICT DO NOTHING semantics.

        The PG unique key ``(from_unit_id, to_unit_id, link_type,
        COALESCE(entity_id, nil_uuid))`` becomes the deterministic ES ``_id``;
        ``op_type=create`` then makes duplicate inserts no-ops (409 ignored).

        The PG FOR KEY SHARE dance (guarding against concurrently deleted
        endpoints) is approximated by verifying endpoint existence with a
        ``terms`` query in the same chunk loop and skipping links whose
        endpoints have vanished — same observable behaviour, without the
        commit-time FK check that ES does not have. ``exists_clause`` is
        therefore accepted and unused (as it is on PostgreSQL).
        """
        if not sorted_links:
            return
        client, idx = _es(conn), _index(table)
        mu_idx = _index(self._get_mu_table())

        for start in range(0, len(sorted_links), chunk_size):
            chunk = sorted_links[start:start + chunk_size]
            referenced = sorted({str(l[0]) for l in chunk} | {str(l[1]) for l in chunk})
            existing: set[str] = set()
            for rstart in range(0, len(referenced), 10000):
                resp = await client.search(
                    index=mu_idx,
                    body={
                        "query": {"terms": {"id": referenced[rstart:rstart + 10000]}},
                        "size": 10000,
                        "_source": False,
                        "track_total_hits": False,
                    },
                )
                existing.update(h["_id"] for h in resp["hits"]["hits"])

            ops: list[dict] = []
            for from_id, to_id, link_type, weight, entity_id in chunk:
                if str(from_id) not in existing or str(to_id) not in existing:
                    continue  # endpoint deleted concurrently -> drop the link
                key_entity = str(entity_id) if entity_id is not None else nil_entity_uuid
                es_id = f"{from_id}::{to_id}::{link_type}::{key_entity}"
                ops.append({"create": {"_index": idx, "_id": es_id}})
                ops.append({
                    "from_unit_id": str(from_id),
                    "to_unit_id": str(to_id),
                    "link_type": link_type,
                    "weight": float(weight) if weight is not None else None,
                    "entity_id": str(entity_id) if entity_id is not None else None,
                    "bank_id": bank_id,
                })
            if ops:
                resp = await client.bulk(operations=ops, refresh=self.REFRESH)
                _raise_on_bulk_errors(resp, ignore_statuses=(409,))

    async def bulk_insert_entities(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        entity_names: list[str],
        entity_dates: list,
    ) -> dict[str, str]:
        """Insert entities, ignoring conflicts; return {lower(name): id} for
        the rows actually inserted (mirrors PG ``RETURNING`` on DO NOTHING).

        The PG unique key ``(bank_id, LOWER(canonical_name))`` becomes the
        deterministic ES ``_id`` ``{bank_id}::{lower(name)}``.
        """
        if not entity_names:
            return {}
        client, idx = _es(conn), _index(table)
        ops: list[dict] = []
        planned: list[tuple[str, str]] = []  # (name_lower, generated_uuid)
        seen: set[str] = set()
        for i, name in enumerate(entity_names):
            name_lower = name.lower()
            if name_lower in seen:
                continue
            seen.add(name_lower)
            eid = str(uuid.uuid4())
            date = _iso(entity_dates[i] if i < len(entity_dates) else None) or _now_iso()
            planned.append((name_lower, eid))
            ops.append({"create": {"_index": idx, "_id": f"{bank_id}::{name_lower}"}})
            ops.append({
                "id": eid,
                "bank_id": bank_id,
                "canonical_name": name,
                "canonical_name_lower": name_lower,
                "first_seen": date,
                "last_seen": date,
                "mention_count": 0,
            })
        resp = await client.bulk(operations=ops, refresh=self.REFRESH)
        inserted: dict[str, str] = {}
        for (name_lower, eid), item in zip(planned, resp.get("items", [])):
            action = item.get("create", {})
            if action.get("status") in (200, 201):
                inserted[name_lower] = eid
            elif action.get("status") != 409:
                raise RuntimeError(f"bulk_insert_entities failed: {action}")
        return inserted

    async def fetch_missing_entity_ids(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        missing_names: list[str],
    ) -> list[ESRow]:
        """Fetch entity ids for names that conflicted during insert.

        PG: unnest + JOIN on LOWER(name). ES: one ``terms`` query on the
        ``canonical_name_lower`` keyword field, then re-attach the original
        input casing (``input_name``) client-side.
        """
        if not missing_names:
            return []
        client, idx = _es(conn), _index(table)
        by_lower: dict[str, str] = {}
        for n in missing_names:
            by_lower.setdefault(n.lower(), n)
        rows: list[ESRow] = []
        lowers = list(by_lower.keys())
        for start in range(0, len(lowers), 10000):
            resp = await client.search(
                index=idx,
                body={
                    "query": {"bool": {"filter": [
                        {"term": {"bank_id": bank_id}},
                        {"terms": {"canonical_name_lower": lowers[start:start + 10000]}},
                    ]}},
                    "size": 10000,
                    "_source": ["id", "canonical_name_lower"],
                    "track_total_hits": False,
                },
            )
            for h in resp["hits"]["hits"]:
                src = h["_source"]
                rows.append(ESRow({
                    "id": src["id"],
                    "name_lower": src["canonical_name_lower"],
                    "input_name": by_lower.get(src["canonical_name_lower"]),
                }))
        return rows

    async def bulk_insert_unit_entities(
        self,
        conn: DatabaseConnection,
        table: str,
        unit_ids: list,
        entity_ids: list,
    ) -> None:
        """ON CONFLICT DO NOTHING -> op_type=create on ``{unit}::{entity}``."""
        if not unit_ids:
            return
        client, idx = _es(conn), _index(table)
        ops: list[dict] = []
        for u, e in zip(unit_ids, entity_ids):
            ops.append({"create": {"_index": idx, "_id": f"{u}::{e}"}})
            ops.append({"unit_id": str(u), "entity_id": str(e)})
        resp = await client.bulk(operations=ops, refresh=self.REFRESH)
        _raise_on_bulk_errors(resp, ignore_statuses=(409,))

    # -- Graph maintenance queue -----------------------------------------

    async def enqueue_graph_maintenance(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        unit_ids: list,
    ) -> None:
        """Dedup on (bank_id, unit_id) -> deterministic _id + op_type=create.

        The PG sort-before-insert deadlock avoidance is unnecessary: ES bulk
        creates take no cross-document locks, and duplicate creates simply
        return 409, which is the DO NOTHING semantics we want.
        """
        if not unit_ids:
            return
        client, idx = _es(conn), _index(table)
        ops: list[dict] = []
        for u in sorted({str(x) for x in unit_ids}):
            ops.append({"create": {"_index": idx, "_id": f"{bank_id}::{u}"}})
            ops.append({"bank_id": bank_id, "unit_id": u, "enqueued_at": _now_iso()})
        resp = await client.bulk(operations=ops, refresh=self.REFRESH)
        _raise_on_bulk_errors(resp, ignore_statuses=(409,))

    async def claim_graph_maintenance_batch(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        limit: int,
    ) -> list[str]:
        """Atomic claim-and-delete of queue rows.

        PG: ``DELETE ... RETURNING`` in one statement. ES: search the oldest
        rows with their ``seq_no``/``primary_term``, then delete each with
        ``if_seq_no``/``if_primary_term``. A row concurrently claimed by
        another worker fails its conditional delete (409) and is *not*
        returned, so each queue row is claimed exactly once.
        """
        if limit <= 0:
            return []
        client, idx = _es(conn), _index(table)
        resp = await client.search(
            index=idx,
            body={
                "query": {"term": {"bank_id": bank_id}},
                "sort": [{"enqueued_at": "asc"}, {"_id": "asc"}],
                "size": limit,
                "seq_no_primary_term": True,
                "_source": ["unit_id"],
                "track_total_hits": False,
            },
        )
        hits = resp["hits"]["hits"]
        if not hits:
            return []
        ops: list[dict] = []
        unit_by_pos: list[str] = []
        for h in hits:
            ops.append({"delete": {
                "_index": idx,
                "_id": h["_id"],
                "if_seq_no": h["_seq_no"],
                "if_primary_term": h["_primary_term"],
            }})
            unit_by_pos.append(str(h["_source"]["unit_id"]))
        dresp = await client.bulk(operations=ops, refresh=self.REFRESH)
        claimed: list[str] = []
        for unit_id, item in zip(unit_by_pos, dresp.get("items", [])):
            action = item.get("delete", {})
            if action.get("status") == 200 and action.get("result") == "deleted":
                claimed.append(unit_id)
            elif action.get("status") not in (404, 409):
                raise RuntimeError(f"claim_graph_maintenance_batch failed: {action}")
        return claimed

    async def prune_orphan_entities(
        self,
        conn: DatabaseConnection,
        entities_table: str,
        ue_table: str,
        bank_id: str,
    ) -> int:
        """Delete entities of the bank with no unit_entities row referencing
        them; return the number of deletions.

        PG: DELETE ... NOT EXISTS. ES has no joins, so: scan entity ids in the
        bank, probe unit_entities with batched ``terms`` aggregations to find
        which entity ids are still referenced, bulk-delete the rest.
        """
        client = _es(conn)
        e_idx, ue_idx = _index(entities_table), _index(ue_table)
        deleted = 0
        batch: list[tuple[str, str]] = []  # (es _id, entity id field)

        async def _flush(entity_batch: list[tuple[str, str]]) -> int:
            if not entity_batch:
                return 0
            ids = [eid for _, eid in entity_batch]
            referenced: set[str] = set()
            resp = await client.search(
                index=ue_idx,
                body={
                    "query": {"terms": {"entity_id": ids}},
                    "size": 0,
                    "aggs": {"ref": {"terms": {"field": "entity_id", "size": len(ids)}}},
                },
            )
            for bucket in resp["aggregations"]["ref"]["buckets"]:
                referenced.add(str(bucket["key"]))
            ops: list[dict] = []
            for es_id, eid in entity_batch:
                if eid not in referenced:
                    ops.append({"delete": {"_index": e_idx, "_id": es_id}})
            if not ops:
                return 0
            dresp = await client.bulk(operations=ops, refresh=self.REFRESH)
            n = sum(
                1 for item in dresp.get("items", [])
                if item.get("delete", {}).get("result") == "deleted"
            )
            return n

        async for hit in _scan(client, e_idx, {"term": {"bank_id": bank_id}},
                               source=["id"]):
            batch.append((hit["_id"], str(hit["_source"]["id"])))
            if len(batch) >= 1000:
                deleted += await _flush(batch)
                batch = []
        deleted += await _flush(batch)

        # FK ON DELETE CASCADE equivalent: also purge cooccurrence rows that
        # point at now-deleted entities is left to prune_stale_cooccurrences /
        # dedicated cascade jobs, exactly as documented in the ABC.
        return deleted

    async def prune_stale_cooccurrences(
        self,
        conn: DatabaseConnection,
        ec_table: str,
        ue_table: str,
        entities_table: str,
        bank_id: str,
    ) -> int:
        """Delete cooccurrence rows whose two entities are no longer witnessed
        together by any unit; return the number of deletions.

        PG: DELETE ... NOT EXISTS(self-join on unit_entities). ES: for each
        cooccurrence in the bank, fetch the unit sets of both entities and
        test the intersection client-side.
        """
        client = _es(conn)
        ec_idx, ue_idx, e_idx = _index(ec_table), _index(ue_table), _index(entities_table)

        # entity_cooccurrences has no bank_id: scope through entities.bank_id.
        bank_entities: set[str] = set()
        async for hit in _scan(client, e_idx, {"term": {"bank_id": bank_id}}, source=["id"]):
            bank_entities.add(str(hit["_source"]["id"]))
        if not bank_entities:
            return 0

        async def _units_of(entity_id: str) -> set[str]:
            units: set[str] = set()
            async for h in _scan(client, ue_idx, {"term": {"entity_id": entity_id}},
                                 source=["unit_id"]):
                units.add(str(h["_source"]["unit_id"]))
            return units

        unit_cache: dict[str, set[str]] = {}
        ops: list[dict] = []
        async for hit in _scan(client, ec_idx, {"match_all": {}},
                               source=["entity_id_1", "entity_id_2"]):
            src = hit["_source"]
            e1, e2 = str(src["entity_id_1"]), str(src["entity_id_2"])
            if e1 not in bank_entities:
                continue
            if e1 not in unit_cache:
                unit_cache[e1] = await _units_of(e1)
            if e2 not in unit_cache:
                unit_cache[e2] = await _units_of(e2)
            if unit_cache[e1] & unit_cache[e2]:
                continue  # still co-witnessed by at least one unit
            ops.append({"delete": {"_index": ec_idx, "_id": hit["_id"]}})

        if not ops:
            return 0
        dresp = await client.bulk(operations=ops, refresh=self.REFRESH)
        return sum(
            1 for item in dresp.get("items", [])
            if item.get("delete", {}).get("result") == "deleted"
        )

    # -- LATERAL / fan-out queries ---------------------------------------

    async def fetch_unit_dates(
        self,
        conn: DatabaseConnection,
        mu_table: str,
        unit_ids: list[str],
    ) -> list[ESRow]:
        """ANY($1) -> terms query. Returns rows with id/event_date/fact_type."""
        if not unit_ids:
            return []
        client, idx = _es(conn), _index(mu_table)
        rows: list[ESRow] = []
        ids = [str(u) for u in unit_ids]
        for start in range(0, len(ids), 10000):
            resp = await client.search(
                index=idx,
                body={
                    "query": {"terms": {"id": ids[start:start + 10000]}},
                    "size": 10000,
                    "_source": ["id", "event_date", "fact_type"],
                    "track_total_hits": False,
                },
            )
            for h in resp["hits"]["hits"]:
                src = h["_source"]
                rows.append(ESRow({
                    "id": src.get("id", h["_id"]),
                    "event_date": src.get("event_date"),
                    "fact_type": src.get("fact_type"),
                }))
        return rows

    async def fetch_temporal_neighbors(
        self,
        conn: DatabaseConnection,
        mu_table: str,
        bank_id: str,
        lateral_unit_ids: list,
        lateral_event_dates: list,
        lateral_fact_types: list,
        half_limit: int,
        batch_size: int = 500,
    ) -> list[ESRow]:
        """Bidirectional temporal neighbor scan.

        PG: unnest + CROSS JOIN LATERAL (one backward + one forward index scan
        per seed). ES: ``msearch`` with two sub-searches per seed (range on
        ``event_date`` <= / >, sorted desc / asc, size=half_limit), then the
        ROW_NUMBER()-style cap at ``half_limit`` per seed computed client-side
        on ``time_diff_hours`` — identical output shape:
        (from_id, id, event_date, time_diff_hours).
        """
        if not lateral_unit_ids:
            return []
        client, idx = _es(conn), _index(mu_table)
        rows: list[ESRow] = []
        # msearch payload budget: 2 sub-searches per seed.
        msearch_seed_batch = min(batch_size, 200)

        for start in range(0, len(lateral_unit_ids), msearch_seed_batch):
            end = min(start + msearch_seed_batch, len(lateral_unit_ids))
            seeds = list(zip(
                lateral_unit_ids[start:end],
                lateral_event_dates[start:end],
                lateral_fact_types[start:end],
            ))
            body: list[dict] = []
            for unit_id, event_date, fact_type in seeds:
                ed = _iso(event_date)
                base_filter = [
                    {"term": {"bank_id": bank_id}},
                    {"term": {"fact_type": fact_type}},
                ]
                must_not = [{"term": {"id": str(unit_id)}}]
                # backward scan: event_date <= seed, newest first
                body.append({"index": idx})
                body.append({
                    "query": {"bool": {
                        "filter": base_filter + [{"range": {"event_date": {"lte": ed}}}],
                        "must_not": must_not,
                    }},
                    "sort": [{"event_date": "desc"}],
                    "size": half_limit,
                    "_source": ["id", "event_date"],
                    "track_total_hits": False,
                })
                # forward scan: event_date > seed, oldest first
                body.append({"index": idx})
                body.append({
                    "query": {"bool": {
                        "filter": base_filter + [{"range": {"event_date": {"gt": ed}}}],
                        "must_not": must_not,
                    }},
                    "sort": [{"event_date": "asc"}],
                    "size": half_limit,
                    "_source": ["id", "event_date"],
                    "track_total_hits": False,
                })
            resp = await client.msearch(searches=body)
            responses = resp["responses"]
            for i, (unit_id, event_date, _ft) in enumerate(seeds):
                candidates: list[ESRow] = []
                for sub in (responses[2 * i], responses[2 * i + 1]):
                    if sub.get("error"):
                        raise RuntimeError(f"fetch_temporal_neighbors: {sub['error']}")
                    for h in sub["hits"]["hits"]:
                        src = h["_source"]
                        candidates.append(ESRow({
                            "from_id": str(unit_id),
                            "id": src.get("id", h["_id"]),
                            "event_date": src.get("event_date"),
                            "time_diff_hours": _epoch_hours_diff(
                                src.get("event_date"), _iso(event_date)
                            ),
                        }))
                # ROW_NUMBER() OVER (PARTITION BY seed ORDER BY time_diff) <= half_limit
                candidates.sort(key=lambda r: r["time_diff_hours"])
                rows.extend(candidates[:half_limit])
        return rows

    # -- CTE builders for graph retrieval --------------------------------
    #
    # These two ABC methods return *SQL text* meant to be spliced into a larger
    # SQL statement by the caller. There is no SQL engine on Elasticsearch, so
    # the fragments cannot exist. The executable equivalents below
    # (expand_entities / expand_semantic_causal) perform the exact computation
    # the CTEs describe and return the same row shape
    # (id, text, context, event_date, occurred_start, occurred_end,
    #  mentioned_at, fact_type, document_id, chunk_id, tags, proof_count,
    #  score, source).

    def build_entity_expansion_cte(
        self,
        mu_table: str,
        ue_table: str,
        per_entity_limit: int,
    ) -> str:
        raise NotImplementedError(
            "Elasticsearch backend cannot emit SQL CTE fragments; call "
            "ElasticsearchOps.expand_entities(conn, mu_table, ue_table, "
            "seed_ids, fact_type, limit, per_entity_limit) instead."
        )

    def build_semantic_causal_cte(
        self,
        ml_table: str,
        mu_table: str,
    ) -> str:
        raise NotImplementedError(
            "Elasticsearch backend cannot emit SQL CTE fragments; call "
            "ElasticsearchOps.expand_semantic_causal(conn, ml_table, mu_table, "
            "seed_ids, fact_type, limit) instead."
        )

    async def expand_entities(
        self,
        conn: DatabaseConnection,
        mu_table: str,
        ue_table: str,
        seed_ids: list,
        fact_type: str,
        limit: int,
        per_entity_limit: int,
    ) -> list[ESRow]:
        """Native equivalent of build_entity_expansion_cte.

        1. seed_entities:   entities linked to the seed units
        2. fan-out:         per entity, up to per_entity_limit other units
        3. score:           COUNT(DISTINCT entity) per candidate unit
        4. hydrate + filter fact_type, ORDER BY score DESC LIMIT $3
        """
        client = _es(conn)
        mu_idx, ue_idx = _index(mu_table), _index(ue_table)
        seed_set = {str(s) for s in seed_ids}
        if not seed_set:
            return []

        # 1. seed entities
        resp = await client.search(
            index=ue_idx,
            body={
                "query": {"terms": {"unit_id": sorted(seed_set)}},
                "size": 0,
                "aggs": {"ents": {"terms": {"field": "entity_id", "size": 10000}}},
            },
        )
        entity_ids = [str(b["key"]) for b in resp["aggregations"]["ents"]["buckets"]]
        if not entity_ids:
            return []

        # 2. per-entity fan-out (LATERAL ... LIMIT per_entity_limit) via msearch
        entity_count_by_unit: dict[str, set[str]] = {}
        for start in range(0, len(entity_ids), 200):
            chunk = entity_ids[start:start + 200]
            body: list[dict] = []
            for eid in chunk:
                body.append({"index": ue_idx})
                body.append({
                    "query": {"bool": {
                        "filter": [{"term": {"entity_id": eid}}],
                        "must_not": [{"terms": {"unit_id": sorted(seed_set)}}],
                    }},
                    "sort": [{"unit_id": "desc"}],
                    "size": per_entity_limit,
                    "_source": ["unit_id"],
                    "track_total_hits": False,
                })
            resp = await client.msearch(searches=body)
            for eid, sub in zip(chunk, resp["responses"]):
                if sub.get("error"):
                    raise RuntimeError(f"expand_entities: {sub['error']}")
                for h in sub["hits"]["hits"]:
                    unit = str(h["_source"]["unit_id"])
                    entity_count_by_unit.setdefault(unit, set()).add(eid)

        if not entity_count_by_unit:
            return []

        # 3-4. hydrate candidates, filter fact_type, score, sort, limit
        candidate_ids = list(entity_count_by_unit.keys())
        rows: list[ESRow] = []
        for start in range(0, len(candidate_ids), 10000):
            resp = await client.search(
                index=mu_idx,
                body={
                    "query": {"bool": {"filter": [
                        {"terms": {"id": candidate_ids[start:start + 10000]}},
                        {"term": {"fact_type": fact_type}},
                    ]}},
                    "size": 10000,
                    "_source": _MU_SOURCE_FIELDS,
                    "track_total_hits": False,
                },
            )
            for h in resp["hits"]["hits"]:
                row = _mu_row(h, {"source": "entity"})
                row["score"] = float(len(entity_count_by_unit.get(str(row["id"]), ())))
                rows.append(row)
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:limit]

    async def expand_semantic_causal(
        self,
        conn: DatabaseConnection,
        ml_table: str,
        mu_table: str,
        seed_ids: list,
        fact_type: str,
        limit: int,
    ) -> tuple[list[ESRow], list[ESRow]]:
        """Native equivalent of build_semantic_causal_cte.

        Returns (semantic_rows, causal_rows):
        * semantic: bidirectional 'semantic' links, dedup by unit with
          MAX(weight) as score, ORDER BY score DESC LIMIT limit;
        * causal:   outgoing causal links, DISTINCT ON (unit) keeping the
          highest weight, LIMIT limit.
        """
        client = _es(conn)
        ml_idx, mu_idx = _index(ml_table), _index(mu_table)
        seeds = sorted({str(s) for s in seed_ids})
        if not seeds:
            return [], []

        # semantic: links touching the seeds in either direction
        sem_score: dict[str, float] = {}
        query = {"bool": {
            "filter": [{"term": {"link_type": "semantic"}}],
            "should": [
                {"terms": {"from_unit_id": seeds}},
                {"terms": {"to_unit_id": seeds}},
            ],
            "minimum_should_match": 1,
        }}
        async for h in _scan(client, ml_idx, query,
                             source=["from_unit_id", "to_unit_id", "weight"]):
            src = h["_source"]
            f, t = str(src["from_unit_id"]), str(src["to_unit_id"])
            w = float(src.get("weight") or 0.0)
            for other in ((t,) if f in seeds else ()) + ((f,) if t in seeds else ()):
                if other in seeds:
                    continue  # mu.id != ALL(seeds)
                sem_score[other] = max(sem_score.get(other, float("-inf")), w)

        # causal: outgoing links only, keep max weight per target
        causal_score: dict[str, float] = {}
        query = {"bool": {"filter": [
            {"terms": {"from_unit_id": seeds}},
            {"terms": {"link_type": _CAUSAL_LINK_TYPES}},
        ]}}
        async for h in _scan(client, ml_idx, query, source=["to_unit_id", "weight"]):
            src = h["_source"]
            t = str(src["to_unit_id"])
            w = float(src.get("weight") or 0.0)
            causal_score[t] = max(causal_score.get(t, float("-inf")), w)

        async def _hydrate(score_by_id: dict[str, float], source_tag: str) -> list[ESRow]:
            if not score_by_id:
                return []
            ids = list(score_by_id.keys())
            out: list[ESRow] = []
            for start in range(0, len(ids), 10000):
                resp = await client.search(
                    index=mu_idx,
                    body={
                        "query": {"bool": {"filter": [
                            {"terms": {"id": ids[start:start + 10000]}},
                            {"term": {"fact_type": fact_type}},
                        ]}},
                        "size": 10000,
                        "_source": _MU_SOURCE_FIELDS,
                        "track_total_hits": False,
                    },
                )
                for h in resp["hits"]["hits"]:
                    row = _mu_row(h, {"source": source_tag})
                    row["score"] = score_by_id[str(row["id"])]
                    out.append(row)
            out.sort(key=lambda r: r["score"], reverse=True)
            return out[:limit]

        return await _hydrate(sem_score, "semantic"), await _hydrate(causal_score, "causal")

    async def expand_observations(
        self,
        conn: DatabaseConnection,
        mu_table: str,
        ue_table: str,
        ml_table: str,
        seed_ids: list,
        budget: int,
        per_entity_limit: int,
    ) -> tuple[list[ESRow], list[ESRow], list[ESRow]]:
        """Observation-specific graph expansion (entity, semantic, causal).

        Mirrors the PG native-array version: source_memory_ids is a keyword
        array on the memory_units documents; the ``&&`` overlap operator
        becomes a ``terms`` query on that field, and the overlap-count score
        (COUNT(DISTINCT s) WHERE s = ANY(connected)) is computed client-side.
        """
        client = _es(conn)
        mu_idx, ue_idx = _index(mu_table), _index(ue_table)
        seeds = sorted({str(s) for s in seed_ids})
        if not seeds:
            return [], [], []

        # seed_sources: distinct unnest(source_memory_ids) of the seeds
        seed_sources: set[str] = set()
        resp = await client.search(
            index=mu_idx,
            body={
                "query": {"bool": {"filter": [{"terms": {"id": seeds}}],
                                    "must": [{"exists": {"field": "source_memory_ids"}}]},
                          },
                "size": len(seeds),
                "_source": ["source_memory_ids"],
                "track_total_hits": False,
            },
        )
        for h in resp["hits"]["hits"]:
            seed_sources.update(str(s) for s in h["_source"].get("source_memory_ids") or [])

        entity_rows: list[ESRow] = []
        if seed_sources:
            # source_entities: entities of those source units
            resp = await client.search(
                index=ue_idx,
                body={
                    "query": {"terms": {"unit_id": sorted(seed_sources)}},
                    "size": 0,
                    "aggs": {"ents": {"terms": {"field": "entity_id", "size": 10000}}},
                },
            )
            source_entities = [str(b["key"]) for b in resp["aggregations"]["ents"]["buckets"]]

            # connected_sources: per-entity fan-out, excluding seed sources
            connected: set[str] = set()
            for start in range(0, len(source_entities), 200):
                chunk = source_entities[start:start + 200]
                body: list[dict] = []
                for eid in chunk:
                    body.append({"index": ue_idx})
                    body.append({
                        "query": {"term": {"entity_id": eid}},
                        "sort": [{"unit_id": "desc"}],
                        "size": per_entity_limit,
                        "_source": ["unit_id"],
                        "track_total_hits": False,
                    })
                mresp = await client.msearch(searches=body)
                for sub in mresp["responses"]:
                    if sub.get("error"):
                        raise RuntimeError(f"expand_observations: {sub['error']}")
                    for h in sub["hits"]["hits"]:
                        uid = str(h["_source"]["unit_id"])
                        if uid not in seed_sources:
                            connected.add(uid)

            if connected:
                connected_list = sorted(connected)
                # observations overlapping the connected source set (&&)
                query = {"bool": {
                    "filter": [
                        {"term": {"fact_type": "observation"}},
                        {"terms": {"source_memory_ids": connected_list}},
                    ],
                    "must_not": [{"terms": {"id": seeds}}],
                }}
                scored: list[ESRow] = []
                async for h in _scan(client, mu_idx, query,
                                     source=_MU_SOURCE_FIELDS + ["source_memory_ids"]):
                    row = _mu_row(h)
                    overlap = len(
                        {str(s) for s in h["_source"].get("source_memory_ids") or []}
                        & connected
                    )
                    row["score"] = float(overlap)
                    scored.append(row)
                scored.sort(key=lambda r: r["score"], reverse=True)
                entity_rows = scored[:budget]

        semantic_rows, causal_rows = await self.expand_semantic_causal(
            conn, ml_table, mu_table, seeds, "observation", budget
        )
        return entity_rows, semantic_rows, causal_rows

    # -- Tag listing -----------------------------------------------------

    def build_tag_listing_parts(self, mu_table: str) -> TagListingParts:
        raise NotImplementedError(
            "Elasticsearch backend cannot emit SQL fragments for tag listing; "
            "call ElasticsearchOps.list_tags(conn, mu_table, bank_id, "
            "prefix=None, limit=1000) instead."
        )

    async def list_tags(
        self,
        conn: DatabaseConnection,
        mu_table: str,
        bank_id: str,
        prefix: str | None = None,
        limit: int = 1000,
    ) -> list[ESRow]:
        """Native equivalent of the tag-listing SQL built from
        build_tag_listing_parts: distinct tags of a bank with usage counts.

        PG: ``unnest(tags)`` + GROUP BY. ES: a ``terms`` aggregation on the
        ``tags`` keyword field (arrays are flattened natively).
        """
        client, idx = _es(conn), _index(mu_table)
        agg: dict[str, Any] = {"field": "tags", "size": limit}
        if prefix:
            agg["include"] = f"{prefix}.*"
        resp = await client.search(
            index=idx,
            body={
                "query": {"bool": {"filter": [
                    {"term": {"bank_id": bank_id}},
                    {"exists": {"field": "tags"}},
                ]}},
                "size": 0,
                "aggs": {"tags": {"terms": agg}},
            },
        )
        return [
            ESRow({"tag": b["key"], "count": b["doc_count"]})
            for b in resp["aggregations"]["tags"]["buckets"]
        ]

    # -- Bank index management -------------------------------------------

    async def create_bank_vector_indexes(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        internal_id: str,
        index_clause: str,
        fact_types: dict[str, str],
    ) -> None:
        """No-op: ES dense_vector fields are indexed globally (HNSW) at the
        mapping level; per-bank partial indexes do not exist. kNN queries are
        scoped with a ``filter`` on bank_id/fact_type instead — same behaviour
        as the non-PG backends described in the ABC.
        """
        return None

    async def drop_bank_vector_indexes(
        self,
        conn: DatabaseConnection,
        schema: str,
        internal_id: str,
        fact_types: dict[str, str],
    ) -> None:
        """No-op (see create_bank_vector_indexes)."""
        return None

    # -- Entity resolution strategy routing ------------------------------

    def get_entity_resolution_strategy(self) -> str:
        """No pg_trgm / UTL_MATCH on ES: fall back to the portable 'full'
        strategy (ES fuzzy/match queries can be layered on top by consumers
        that know about this backend).
        """
        return "full"

    # -- Webhook operations ------------------------------------------------

    _WEBHOOK_FIELDS = [
        "id", "bank_id", "url", "secret", "event_types", "enabled",
        "http_config", "created_at", "updated_at",
    ]

    def _webhook_row(self, hit_or_source: Mapping[str, Any]) -> ESRow:
        src = hit_or_source.get("_source", hit_or_source)
        row = ESRow({f: src.get(f) for f in self._WEBHOOK_FIELDS})
        # PG returns http_config::text and timestamps ::text -> keep strings.
        if isinstance(row.get("http_config"), (dict, list)):
            row["http_config"] = json.dumps(row["http_config"])
        return row

    async def create_webhook(
        self,
        conn,
        table,
        webhook_id,
        bank_id,
        url,
        secret,
        event_types,
        enabled,
        http_config_json,
    ):
        """Insert a webhook row and return the created row (or None)."""
        client, idx = _es(conn), _index(table)
        now = _now_iso()
        doc = {
            "id": str(webhook_id),
            "bank_id": bank_id,
            "url": url,
            "secret": secret,
            "event_types": list(event_types or []),
            "enabled": bool(enabled),
            "http_config": _load_json(http_config_json, {}),
            "created_at": now,
            "updated_at": now,
        }
        await client.index(index=idx, id=str(webhook_id), op_type="create",
                           document=doc, refresh=self.REFRESH)
        return self._webhook_row(doc)

    async def list_webhooks_for_bank(self, conn, table, bank_id):
        """All webhooks of a bank, ORDER BY created_at."""
        client, idx = _es(conn), _index(table)
        resp = await client.search(
            index=idx,
            body={
                "query": {"term": {"bank_id": bank_id}},
                "sort": [{"created_at": "asc"}],
                "size": 10000,
                "track_total_hits": False,
            },
        )
        return [self._webhook_row(h) for h in resp["hits"]["hits"]]

    async def get_webhooks_for_dispatch(self, conn, webhook_table, bank_id):
        """Enabled webhooks matching the bank + global rows (bank_id NULL).

        SQL NULL bank_id -> ES document *missing* the bank_id field
        (must_not exists).
        """
        client, idx = _es(conn), _index(webhook_table)
        resp = await client.search(
            index=idx,
            body={
                "query": {"bool": {
                    "filter": [{"term": {"enabled": True}}],
                    "should": [
                        {"term": {"bank_id": bank_id}},
                        {"bool": {"must_not": [{"exists": {"field": "bank_id"}}]}},
                    ],
                    "minimum_should_match": 1,
                }},
                "size": 10000,
                "track_total_hits": False,
            },
        )
        return [self._webhook_row(h) for h in resp["hits"]["hits"]]

    async def update_webhook(self, conn, table, webhook_id, bank_id, set_clauses, params):
        """Update a webhook and return the updated row, or None if not found.

        The ABC's contract passes SQL ``set_clauses`` + positional ``params``
        (params[0]=webhook_id, params[1]=bank_id, then one value per clause,
        as built by the shared caller). We parse the column name out of each
        ``"col = $N"`` clause and apply a partial-document update.
        """
        client, idx = _es(conn), _index(table)
        try:
            got = await client.get(index=idx, id=str(webhook_id))
        except Exception as exc:
            if _status_of(exc) == 404:
                return None
            raise
        if got["_source"].get("bank_id") != bank_id:
            return None

        doc: dict[str, Any] = {}
        value_params = list(params[2:])  # $1=webhook_id, $2=bank_id
        for clause, value in zip(set_clauses, value_params):
            col = clause.split("=", 1)[0].strip()
            if col == "http_config":
                value = _load_json(value, {})
            if col == "event_types" and value is not None:
                value = list(value)
            doc[col] = value
        doc["updated_at"] = _now_iso()

        try:
            await client.update(
                index=idx,
                id=str(webhook_id),
                doc=doc,
                if_seq_no=got["_seq_no"],
                if_primary_term=got["_primary_term"],
                refresh=self.REFRESH,
            )
        except Exception as exc:
            if _status_of(exc) in (404, 409):
                return None
            raise
        updated = await client.get(index=idx, id=str(webhook_id))
        return self._webhook_row(updated)

    async def delete_webhook(self, conn, table, webhook_id, bank_id):
        """Delete a webhook; True if a row was deleted."""
        client, idx = _es(conn), _index(table)
        try:
            got = await client.get(index=idx, id=str(webhook_id),
                                   source_includes=["bank_id"])
        except Exception as exc:
            if _status_of(exc) == 404:
                return False
            raise
        if got["_source"].get("bank_id") != bank_id:
            return False
        try:
            resp = await client.delete(index=idx, id=str(webhook_id),
                                       refresh=self.REFRESH)
        except Exception as exc:
            if _status_of(exc) == 404:
                return False
            raise
        return resp.get("result") == "deleted"

    async def list_webhook_deliveries(self, conn, ops_table, webhook_id, bank_id, limit, cursor):
        """Delivery operations of a webhook, newest first, cursor-paginated.

        PG filters ``task_payload->>'webhook_id'``: the payload is stored as an
        ES ``object``, so the same filter is a term on
        ``task_payload.webhook_id``. Fetches limit+1 rows like PG so the
        caller can detect the next page.
        """
        client, idx = _es(conn), _index(ops_table)
        filters: list[dict] = [
            {"term": {"operation_type": "webhook_delivery"}},
            {"term": {"bank_id": bank_id}},
            {"term": {"task_payload.webhook_id": str(webhook_id)}},
        ]
        if cursor:
            filters.append({"range": {"created_at": {"lt": _iso(cursor)}}})
        resp = await client.search(
            index=idx,
            body={
                "query": {"bool": {"filter": filters}},
                "sort": [{"created_at": "desc"}],
                "size": limit + 1,
                "track_total_hits": False,
            },
        )
        rows: list[ESRow] = []
        for h in resp["hits"]["hits"]:
            src = h["_source"]
            payload = src.get("task_payload")
            meta = src.get("result_metadata")
            rows.append(ESRow({
                "operation_id": src.get("operation_id", h["_id"]),
                "status": src.get("status"),
                "retry_count": src.get("retry_count", 0),
                "next_retry_at": src.get("next_retry_at"),
                "error_message": src.get("error_message"),
                "task_payload": json.dumps(payload) if isinstance(payload, (dict, list)) else payload,
                "result_metadata": json.dumps(meta) if isinstance(meta, (dict, list)) else meta,
                "created_at": src.get("created_at"),
                "updated_at": src.get("updated_at"),
            }))
        return rows

    async def insert_webhook_delivery_task(self, conn, ops_table, operation_id, bank_id, payload_json, timestamp):
        """Insert a webhook_delivery task into async_operations."""
        client, idx = _es(conn), _index(ops_table)
        ts = _iso(timestamp)
        await client.index(
            index=idx,
            id=str(operation_id),
            op_type="create",
            document={
                "operation_id": str(operation_id),
                "bank_id": bank_id,
                "operation_type": "webhook_delivery",
                "status": "pending",
                "task_payload": _load_json(payload_json, {}),
                "result_metadata": {},
                "retry_count": 0,
                "next_retry_at": None,
                "created_at": ts,
                "updated_at": ts,
            },
            refresh=self.REFRESH,
        )

    # -- Task claiming operations ------------------------------------------

    async def claim_tasks(
        self,
        conn: DatabaseConnection,
        table: str,
        worker_id: str,
        reserved_limits: dict[str, int],
        shared_limit: int,
        *,
        consolidation_bank_priority: dict[str, int] | None = None,
    ) -> list[ESRow]:
        """Claim pending tasks from the async_operations index.

        PG: NOT EXISTS + FOR UPDATE SKIP LOCKED. ES: search candidates with
        their seq_no/primary_term, then conditionally update each to
        ``status='processing'`` with ``if_seq_no``/``if_primary_term``. A task
        grabbed concurrently by another worker fails its conditional update
        (409) and is skipped — the exact SKIP LOCKED semantics, one document
        at a time.

        Consolidation tasks are bank-serialized (banks with a consolidation
        task already 'processing' are excluded) and support the priority-tier
        scheduling described in the ABC (``*`` wildcards -> ES ``wildcard``
        queries; specific patterns take precedence over the ``*`` catch-all).

        Returns rows with operation_id, operation_type, task_payload,
        retry_count (task_payload re-serialized to a JSON string, matching the
        PG driver output consumed by ClaimedTask builders).
        """
        client, idx = _es(conn), _index(table)
        claimed: list[ESRow] = []
        claimed_ids: list[str] = []
        now = _now_iso()

        base_filters = [
            {"term": {"status": "pending"}},
            {"exists": {"field": "task_payload"}},
        ]
        retry_ok = {"bool": {"should": [
            {"bool": {"must_not": [{"exists": {"field": "next_retry_at"}}]}},
            {"range": {"next_retry_at": {"lte": now}}},
        ], "minimum_should_match": 1}}

        async def _busy_consolidation_banks() -> list[str]:
            resp = await client.search(
                index=idx,
                body={
                    "query": {"bool": {"filter": [
                        {"term": {"operation_type": "consolidation"}},
                        {"terms": {"status": ["processing", "claimed"]}},
                    ]}},
                    "size": 0,
                    "aggs": {"banks": {"terms": {"field": "bank_id", "size": 10000}}},
                },
            )
            return [str(b["key"]) for b in resp["aggregations"]["banks"]["buckets"]]

        async def _try_claim(hit: Mapping[str, Any]) -> ESRow | None:
            """Conditional pending->processing transition on one document."""
            try:
                await client.update(
                    index=idx,
                    id=hit["_id"],
                    doc={
                        "status": "processing",
                        "worker_id": worker_id,
                        "updated_at": _now_iso(),
                    },
                    if_seq_no=hit["_seq_no"],
                    if_primary_term=hit["_primary_term"],
                    refresh=self.REFRESH,
                )
            except Exception as exc:
                if _status_of(exc) in (404, 409):
                    return None  # someone else won: SKIP LOCKED
                raise
            src = hit["_source"]
            payload = src.get("task_payload")
            return ESRow({
                "operation_id": src.get("operation_id", hit["_id"]),
                "operation_type": src.get("operation_type"),
                "task_payload": json.dumps(payload) if isinstance(payload, (dict, list)) else payload,
                "retry_count": src.get("retry_count", 0),
            })

        async def _claim_matching(extra_filters: list[dict],
                                  extra_must_not: list[dict],
                                  limit: int) -> list[ESRow]:
            """Claim up to ``limit`` tasks matching the filters, oldest first.

            Over-fetches to compensate for tasks lost to concurrent workers.
            """
            got: list[ESRow] = []
            attempts = 0
            while len(got) < limit and attempts < 5:
                attempts += 1
                must_not = list(extra_must_not)
                if claimed_ids:
                    must_not.append({"terms": {"operation_id": claimed_ids}})
                resp = await client.search(
                    index=idx,
                    body={
                        "query": {"bool": {
                            "filter": base_filters + [retry_ok] + extra_filters,
                            "must_not": must_not,
                        }},
                        "sort": [{"created_at": "asc"}, {"_id": "asc"}],
                        "size": (limit - len(got)) * 2,
                        "seq_no_primary_term": True,
                        "track_total_hits": False,
                    },
                )
                hits = resp["hits"]["hits"]
                if not hits:
                    break
                progressed = False
                for h in hits:
                    if len(got) >= limit:
                        break
                    row = await _try_claim(h)
                    if row is not None:
                        progressed = True
                        got.append(row)
                        claimed_ids.append(str(row["operation_id"]))
                if not progressed:
                    break
            return got

        async def _claim_consolidation(limit: int) -> list[ESRow]:
            if limit <= 0:
                return []
            busy = await _busy_consolidation_banks()
            busy_must_not = [{"terms": {"bank_id": busy}}] if busy else []
            cons_filter = [{"term": {"operation_type": "consolidation"}}]

            if not consolidation_bank_priority:
                return await _claim_matching(cons_filter, busy_must_not, limit)

            # tiered claiming: specific patterns beat the '*' catch-all
            specific_by_priority: dict[int, list[str]] = {}
            all_specific: list[str] = []
            catch_all_priority = 1
            for pattern, priority in consolidation_bank_priority.items():
                if pattern == "*":
                    catch_all_priority = priority
                else:
                    specific_by_priority.setdefault(priority, []).append(pattern)
                    all_specific.append(pattern)

            result: list[ESRow] = []
            remaining = limit
            for pri in sorted(set(specific_by_priority) | {catch_all_priority}, reverse=True):
                if remaining <= 0:
                    break
                if pri in specific_by_priority:
                    like = {"bool": {"should": [
                        {"wildcard": {"bank_id": {"value": p}}}
                        for p in specific_by_priority[pri]
                    ], "minimum_should_match": 1}}
                    rows = await _claim_matching(cons_filter + [like], busy_must_not, remaining)
                    result.extend(rows)
                    remaining -= len(rows)
                if pri == catch_all_priority and remaining > 0:
                    not_like = [
                        {"wildcard": {"bank_id": {"value": p}}} for p in all_specific
                    ]
                    rows = await _claim_matching(
                        cons_filter, busy_must_not + not_like, remaining
                    )
                    result.extend(rows)
                    remaining -= len(rows)
            return result

        # 1. reserved per-type pools
        for op_type, limit in (reserved_limits or {}).items():
            if limit <= 0:
                continue
            if op_type == "consolidation":
                claimed.extend(await _claim_consolidation(limit))
            else:
                claimed.extend(await _claim_matching(
                    [{"term": {"operation_type": op_type}}], [], limit
                ))

        # 2. shared pool: any type, excluding already-claimed ids; consolidation
        #    inside the shared pool still honours bank serialization.
        if shared_limit and shared_limit > 0:
            busy = await _busy_consolidation_banks()
            must_not: list[dict] = []
            if busy:
                must_not.append({"bool": {"filter": [
                    {"term": {"operation_type": "consolidation"}},
                    {"terms": {"bank_id": busy}},
                ]}})
            claimed.extend(await _claim_matching([], must_not, shared_limit))

        return claimed


# ---------------------------------------------------------------------------
# Bulk / error helpers
# ---------------------------------------------------------------------------

def _status_of(exc: Exception) -> int | None:
    """Best-effort HTTP status extraction across elasticsearch-py versions."""
    for attr in ("status_code", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    meta = getattr(exc, "meta", None)
    v = getattr(meta, "status", None)
    return v if isinstance(v, int) else None


def _raise_on_bulk_errors(resp: Mapping[str, Any], ignore_statuses: Iterable[int]) -> None:
    if not resp.get("errors"):
        return
    ignore = set(ignore_statuses)
    failures = []
    for item in resp.get("items", []):
        for _op, action in item.items():
            status = action.get("status")
            if action.get("error") and status not in ignore:
                failures.append(action)
    if failures:
        raise RuntimeError(f"Elasticsearch bulk failures: {failures[:5]}"
                           + (f" (+{len(failures) - 5} more)" if len(failures) > 5 else ""))


# ---------------------------------------------------------------------------
# Index bootstrap (schema equivalent)
# ---------------------------------------------------------------------------

INDEX_MAPPINGS: dict[str, dict] = {
    "memory_units": {
        "properties": {
            "id": {"type": "keyword"},
            "bank_id": {"type": "keyword"},
            "text": {"type": "text"},
            "context": {"type": "text"},
            "text_signals": {"type": "text"},
            "embedding": {"type": "dense_vector", "index": True,
                          "similarity": "cosine"},
            "event_date": {"type": "date"},
            "occurred_start": {"type": "date"},
            "occurred_end": {"type": "date"},
            "mentioned_at": {"type": "date"},
            "fact_type": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": True},
            "chunk_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "tags": {"type": "keyword"},
            "observation_scopes": {"type": "object", "enabled": True},
            "proof_count": {"type": "integer"},
            "source_memory_ids": {"type": "keyword"},
            "created_at": {"type": "date"},
        }
    },
    "memory_links": {
        "properties": {
            "from_unit_id": {"type": "keyword"},
            "to_unit_id": {"type": "keyword"},
            "link_type": {"type": "keyword"},
            "weight": {"type": "float"},
            "entity_id": {"type": "keyword"},
            "bank_id": {"type": "keyword"},
        }
    },
    "entities": {
        "properties": {
            "id": {"type": "keyword"},
            "bank_id": {"type": "keyword"},
            "canonical_name": {"type": "text",
                                "fields": {"raw": {"type": "keyword"}}},
            "canonical_name_lower": {"type": "keyword"},
            "first_seen": {"type": "date"},
            "last_seen": {"type": "date"},
            "mention_count": {"type": "integer"},
        }
    },
    "unit_entities": {
        "properties": {
            "unit_id": {"type": "keyword"},
            "entity_id": {"type": "keyword"},
        }
    },
    "entity_cooccurrences": {
        "properties": {
            "entity_id_1": {"type": "keyword"},
            "entity_id_2": {"type": "keyword"},
            "count": {"type": "integer"},
        }
    },
    "document_chunks": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "bank_id": {"type": "keyword"},
            "chunk_text": {"type": "text"},
            "chunk_index": {"type": "integer"},
            "content_hash": {"type": "keyword"},
        }
    },
    "documents": {
        "properties": {
            "id": {"type": "keyword"},
            "bank_id": {"type": "keyword"},
            "original_text": {"type": "text"},
            "content_hash": {"type": "keyword"},
        }
    },
    "webhooks": {
        "properties": {
            "id": {"type": "keyword"},
            "bank_id": {"type": "keyword"},
            "url": {"type": "keyword"},
            "secret": {"type": "keyword"},
            "event_types": {"type": "keyword"},
            "enabled": {"type": "boolean"},
            "http_config": {"type": "object", "enabled": True},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        }
    },
    "async_operations": {
        "properties": {
            "operation_id": {"type": "keyword"},
            "bank_id": {"type": "keyword"},
            "operation_type": {"type": "keyword"},
            "status": {"type": "keyword"},
            "worker_id": {"type": "keyword"},
            "task_payload": {
                "type": "object",
                "properties": {"webhook_id": {"type": "keyword"}},
            },
            "result_metadata": {"type": "object", "enabled": True},
            "retry_count": {"type": "integer"},
            "next_retry_at": {"type": "date"},
            "error_message": {"type": "text"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        }
    },
    "graph_maintenance_queue": {
        "properties": {
            "bank_id": {"type": "keyword"},
            "unit_id": {"type": "keyword"},
            "enqueued_at": {"type": "date"},
        }
    },
}


async def ensure_indexes(conn: Any, schema: str = "public",
                         embedding_dims: int | None = None) -> None:
    """Create every index of the schema if missing (CREATE TABLE equivalent).

    ``embedding_dims`` pins the dense_vector dimension; when None, ES infers
    it from the first indexed document.
    """
    client = _es(conn)
    for table, mapping in INDEX_MAPPINGS.items():
        idx = _index(f"{schema}.{table}")
        body = json.loads(json.dumps(mapping))  # deep copy
        if table == "memory_units" and embedding_dims:
            body["properties"]["embedding"]["dims"] = embedding_dims
        exists = await client.indices.exists(index=idx)
        if not exists:
            try:
                await client.indices.create(index=idx, mappings=body)
            except Exception as exc:
                if _status_of(exc) != 400:  # resource_already_exists race
                    raise
