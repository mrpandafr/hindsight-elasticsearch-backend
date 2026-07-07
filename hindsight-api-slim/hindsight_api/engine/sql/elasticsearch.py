"""SQL dialect for the Elasticsearch backend.

Placement: hindsight_api/engine/sql/elasticsearch.py

Returned by ``create_sql_dialect("elasticsearch")``, which memory_engine
calls right after migrations. Elasticsearch executes no SQL: every actual
data path goes through ``ElasticsearchOps`` (engine/db/ops_elasticsearch.py)
and the native retrieval primitives (``knn_search`` / ``text_search`` /
``get_by_ids``), and any raw SQL that still reaches the connection raises
``NativeBackendError`` by design.

This dialect therefore exists for exactly one reason: the engine builds its
SQL query plan *before* deciding where to send it, and that construction
must not crash. Every method returns a syntactically well-formed, inert
PG-style fragment — never executed, never meaningful.

Two hard rules for this file:

1. **Exact ABC signatures.** Each overridden method matches
   ``base.SQLDialect`` argument-for-argument (the previous skeleton
   diverged on ``upsert``, ``array_contains``, ``greatest``,
   ``bulk_unnest``, ``returning``, ``limit_offset``, ``advisory_lock`` —
   all of which pass instantiation but explode with TypeError at call
   time, which is the worst failure mode: late and query-dependent).
2. **Fragments must compose.** Callers concatenate these strings into
   larger statements, so each fragment keeps the arity and shape of its
   PG counterpart (placeholders preserved, clauses complete), just
   semantically empty where emptiness is safe (``WHERE 1=0`` arms).
"""

from __future__ import annotations

from typing import Any

from .base import SQLDialect


class ESDialect(SQLDialect):
    """Inert SQL dialect for Elasticsearch (fragments built, never run)."""

    # -- Parameter binding -----------------------------------------------

    def param(self, n: int) -> str:
        return f"${n}"

    # -- Type casting ----------------------------------------------------

    def cast(self, param: str, type_name: str) -> str:
        return f"{param}::{type_name}"

    # -- Vector operations -----------------------------------------------

    def vector_distance(self, col: str, param: str) -> str:
        # Real vector search happens in ElasticsearchConnection.knn_search().
        return f"{col} <=> {param}::vector"

    def vector_similarity(self, col: str, param: str) -> str:
        return f"1 - ({col} <=> {param}::vector)"

    # -- JSON operations -------------------------------------------------

    def json_extract_text(self, col: str, key: str) -> str:
        return f"{col} ->> '{key}'"

    def json_contains(self, col: str, param: str) -> str:
        return f"{col} @> {param}::jsonb"

    def json_merge(self, col: str, param: str) -> str:
        return f"{col} || {param}::jsonb"

    # -- Text search -----------------------------------------------------

    def text_search_score(self, col: str, query_param: str, *, index_name: str | None = None) -> str:
        # Real BM25 happens in ElasticsearchConnection.text_search().
        return f"ts_rank_cd({col}, to_tsquery({query_param}))"

    def text_search_order(self, col: str, query_param: str, *, index_name: str | None = None) -> str:
        return f"ts_rank_cd({col}, to_tsquery({query_param})) DESC"

    # -- Fuzzy string matching -------------------------------------------

    def similarity(self, col: str, param: str) -> str:
        return f"similarity({col}, {param})"

    # -- Upsert ----------------------------------------------------------

    def upsert(
        self,
        table: str,
        columns: list[str],
        conflict_columns: list[str],
        update_columns: list[str],
    ) -> str:
        """Exact ABC signature (the skeleton's 3-arg version raised
        TypeError at call time). Mirrors the PG shape; if this string ever
        reaches a connection, NativeBackendError names the ops-layer
        replacement (deterministic _id + op_type=create / doc_as_upsert).
        """
        col_list = ", ".join(columns)
        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        conflict = ", ".join(conflict_columns)
        if not update_columns:
            return (
                f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict}) DO NOTHING"
            )
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_columns)
        return (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
        )

    # -- Bulk operations -------------------------------------------------

    def bulk_unnest(self, param_types: list[tuple[str, str]]) -> str:
        """ABC signature: list of (placeholder, sql_type) pairs — the
        skeleton's (columns, types) form raised TypeError."""
        args = ", ".join(f"{p}::{t}" for p, t in param_types)
        return f"unnest({args})"

    # -- Pagination ------------------------------------------------------

    def limit_offset(self, limit_param: str, offset_param: str) -> str:
        """ABC passes placeholders (e.g. "$3", "$4"), not ints."""
        return f"LIMIT {limit_param} OFFSET {offset_param}"

    # -- RETURNING clause ------------------------------------------------

    def returning(self, columns: list[str]) -> str:
        """ABC passes a list — the skeleton's str version would have
        emitted ``RETURNING ['a', 'b']``."""
        return f"RETURNING {', '.join(columns)}"

    # -- Pattern matching ------------------------------------------------

    def ilike(self, col: str, param: str) -> str:
        return f"{col} ILIKE {param}"

    # -- Array operations ------------------------------------------------

    def array_any(self, param: str) -> str:
        """ABC contract: expression appended after a column
        (``col {array_any(param)}``) — the skeleton's ``col = ANY(col)``
        self-reference was semantically wrong even as a no-op."""
        return f"= ANY({param})"

    def array_all(self, param: str) -> str:
        return f"!= ALL({param})"

    def array_contains(self, col: str, param: str) -> str:
        """Two-argument ABC signature (the skeleton's one-arg version
        raised TypeError)."""
        return f"{col} @> {param}::varchar[]"

    # -- Locking ---------------------------------------------------------

    def for_update_skip_locked(self) -> str:
        # Real claim semantics: optimistic concurrency in ElasticsearchOps
        # (if_seq_no / if_primary_term).
        return "FOR UPDATE SKIP LOCKED"

    def advisory_lock(self, id_param: str) -> str:
        """Honours its argument (the skeleton hard-coded $1). Real
        cross-process exclusivity: op_type=create claims in the ops layer
        and the migration runner."""
        return f"pg_try_advisory_lock({id_param})"

    # -- UUID generation -------------------------------------------------

    def generate_uuid(self) -> str:
        # Real ids are generated client-side (uuid4) in ElasticsearchOps.
        return "gen_random_uuid()"

    # -- Misc ------------------------------------------------------------

    def greatest(self, *args: str) -> str:
        """Variadic per the ABC (the skeleton's (a, b) raised TypeError on
        the first 3-argument call)."""
        return f"GREATEST({', '.join(args)})"

    def current_timestamp(self) -> str:
        return "now()"

    def array_agg(self, expr: str) -> str:
        return f"array_agg({expr})"

    # -- Retrieval query arms ----------------------------------------------

    def build_semantic_arm(
        self,
        *,
        table: str,
        cols: str,
        fact_type: str,
        embedding_param: str,
        bank_id_param: str,
        fetch_limit: int,
        min_similarity: float,
        tags_clause: str = "",
        groups_clause: str = "",
        extra_where: str = "",
    ) -> str:
        """Empty-result arm with the exact column shape of the PG arm
        (cols, similarity, bm25_score, source), so a UNION ALL that mixes
        arms stays well-formed. Real semantic retrieval:
        ``ElasticsearchConnection.knn_search()``.
        """
        return (
            f"(SELECT {cols},"
            f"        NULL::float AS similarity,"
            f"        NULL::float AS bm25_score,"
            f"        'semantic' AS source"
            f" FROM {table}"
            f" WHERE 1=0)"
        )

    def build_bm25_arm(
        self,
        *,
        table: str,
        cols: str,
        fact_type: str,
        bank_id_param: str,
        limit_param: str,
        text_param: str,
        tags_clause: str = "",
        groups_clause: str = "",
        arm_index: int = 0,
        text_search_extension: str = "native",
        bm25_language: str = "english",
        bm25_min_score: float = 0.0,
        extra_where: str = "",
    ) -> str:
        """Empty-result arm, same shape as the PG arm. Real BM25 retrieval:
        ``ElasticsearchConnection.text_search()``.
        """
        return (
            f"(SELECT {cols},"
            f"        NULL::float AS similarity,"
            f"        NULL::float AS bm25_score,"
            f"        'bm25' AS source"
            f" FROM {table}"
            f" WHERE 1=0)"
        )

    def prepare_bm25_text(
        self,
        tokens: list[str],
        query_text: str,
        *,
        text_search_extension: str = "native",
    ) -> str:
        # Bound as a parameter, never parsed by a SQL text-search engine;
        # the native path re-tokenizes through ES analyzers anyway.
        return query_text

    # -- Extras kept from the team skeleton --------------------------------
    # Not part of the SQLDialect ABC, but engine code paths may call them on
    # the dialect object; kept with fixed arity, inert values.

    def json_extract_path(self, column: str, *path: str) -> str:
        return column

    def json_extract_path_text(self, column: str, *path: str) -> str:
        return column

    def json_array_length(self, column: str) -> str:
        return "0"

    def json_array_element_text(self, column: str, index: int) -> str:
        return "''"

    def unnest(self, column: str) -> str:
        return column

    def regexp_like(self, column: str, pattern: str) -> str:
        return "TRUE"

    def vector_distance_l2sq(self, column: str, embedding: str) -> str:
        return "0.0"

    def vector_distance_cosine(self, column: str, embedding: str) -> str:
        return "0.0"

    def vector_distance_inner_product(self, column: str, embedding: str) -> str:
        return "0.0"

    def now(self) -> str:
        return "NOW()"

    def cast_to_text(self, expr: str) -> str:
        return expr
