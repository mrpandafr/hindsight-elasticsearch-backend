"""Elasticsearch backend — native connection layer for ElasticsearchOps.

Placement: hindsight_api/engine/db/elasticsearch.py
Companion: hindsight_api/engine/db/ops_elasticsearch.py (source of truth)

This module deliberately contains **zero SQL translation**. Elasticsearch is
not a SQL engine and pretending otherwise (rewriting PG queries on the fly)
would only hide semantic mismatches. The contract is native end to end:

* Every data-access operation goes through ``ElasticsearchOps``
  (ops_elasticsearch.py), which speaks bulk / msearch / aggregations /
  optimistic concurrency directly.
* ``ElasticsearchConnection`` is the thin object those ops consume: it
  exposes the raw ``AsyncElasticsearch`` client as ``.client`` (exactly what
  ``ops_elasticsearch._es(conn)`` unwraps) plus the native retrieval
  primitives the ops layer does not cover — kNN vector search and BM25
  full-text search over the ``memory_units`` mapping.
* Legacy SQL entry points (``execute``/``fetch``/``fetchrow``/``fetchval``/
  ``executemany``) exist only to satisfy the ``DatabaseConnection``
  interface and raise ``NativeBackendError`` immediately, naming the native
  replacement. No statement is ever "mapped" — a caller still emitting SQL
  is a caller that must be ported.

Connection URL formats accepted:

    elasticsearch://user:pass@host:9200
    elasticsearch+https://user:pass@host:9243
    https://elastic:changeme@localhost:9200

Optional environment-driven settings (wired by the backend factory):

    HINDSIGHT_API_ES_API_KEY        API key auth (instead of basic auth)
    HINDSIGHT_API_ES_VERIFY_CERTS   "false" to disable TLS verification
    HINDSIGHT_API_ES_EMBEDDING_DIMS pin the dense_vector dimension at bootstrap
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

try:  # inside the hindsight package
    from .base import DatabaseBackend, DatabaseConnection
except ImportError:  # standalone usage / tests outside the package
    class DatabaseBackend:  # type: ignore
        pass

    class DatabaseConnection:  # type: ignore
        pass

from .ops_elasticsearch import (
    ESRow,
    ElasticsearchOps,
    _index,
    _parse_vector,
)
from .migrations_elasticsearch import current_version, run_migrations

logger = logging.getLogger(__name__)

#: memory_units fields hydrated by the native search helpers — kept in sync
#: with ops_elasticsearch._MU_SOURCE_FIELDS plus the retrieval extras.
_SEARCH_SOURCE_FIELDS = [
    "id", "text", "context", "event_date", "occurred_start", "occurred_end",
    "mentioned_at", "fact_type", "document_id", "chunk_id", "tags",
    "proof_count", "metadata", "source_memory_ids",
]


class NativeBackendError(RuntimeError):
    """Raised when a caller reaches this backend with SQL.

    The Elasticsearch backend is native-only: there is no SQL engine behind
    it and nothing is translated. The message names the replacement
    (an ElasticsearchOps method, a native search helper, or ``conn.client``).
    """


def _import_elasticsearch():
    """Lazy import elasticsearch to avoid a hard dependency (oracledb pattern)."""
    try:
        from elasticsearch import AsyncElasticsearch  # type: ignore[import-not-found]
        return AsyncElasticsearch
    except ImportError:
        raise ImportError(
            "elasticsearch (>=8) is required for the Elasticsearch backend. "
            "Install it with: pip install 'elasticsearch>=8'"
        ) from None


def _bank_filters(bank_id: str, fact_type: str | list[str] | None,
                  extra_filters: list[dict] | None) -> list[dict]:
    """Standard scoping filters: bank, optional fact_type(s), extras."""
    filters: list[dict] = [{"term": {"bank_id": bank_id}}]
    if isinstance(fact_type, str):
        filters.append({"term": {"fact_type": fact_type}})
    elif isinstance(fact_type, (list, tuple)) and fact_type:
        filters.append({"terms": {"fact_type": list(fact_type)}})
    if extra_filters:
        filters.extend(extra_filters)
    return filters


def _hits_to_rows(resp: dict, source_tag: str) -> list[ESRow]:
    rows: list[ESRow] = []
    for h in resp["hits"]["hits"]:
        src = h.get("_source", {})
        row = ESRow({f: src.get(f) for f in _SEARCH_SOURCE_FIELDS})
        row["id"] = src.get("id", h.get("_id"))
        row["score"] = h.get("_score")
        row["source"] = source_tag
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# ElasticsearchConnection
# ---------------------------------------------------------------------------

class ElasticsearchConnection(DatabaseConnection):
    """Native connection handle consumed by ElasticsearchOps.

    ``ops_elasticsearch._es(conn)`` unwraps ``.client``; everything the ops
    layer needs is that attribute. On top of it, this class provides the
    retrieval primitives that live outside DataAccessOps: kNN vector search
    (pgvector replacement) and BM25 full-text search (tsvector replacement).
    """

    __slots__ = ("client", "_schema")

    def __init__(self, client: Any, schema: str = "public") -> None:
        self.client = client
        self._schema = schema

    @property
    def backend_type(self) -> str:
        return "elasticsearch"

    @property
    def schema(self) -> str:
        return self._schema

    # -- transaction --------------------------------------------------------

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["ElasticsearchConnection"]:
        """No-op: Elasticsearch has no multi-document transactions.

        Atomicity is provided by ElasticsearchOps through deterministic
        document ids, ``op_type=create`` and optimistic concurrency control
        (``if_seq_no``/``if_primary_term``). A failure inside this block is
        NOT rolled back; every ops-layer operation is individually
        idempotent instead.
        """
        yield self

    # -- native retrieval primitives ------------------------------------------

    async def knn_search(
        self,
        table: str,
        query_vector: Any,
        bank_id: str,
        *,
        fact_type: str | list[str] | None = None,
        k: int = 10,
        num_candidates: int | None = None,
        extra_filters: list[dict] | None = None,
    ) -> list[ESRow]:
        """Vector search on the ``embedding`` dense_vector field.

        Replaces the pgvector ``ORDER BY embedding <=> $1`` path. Per-bank
        scoping is done with a kNN ``filter`` (the ES-native equivalent of
        the per-bank partial indexes that create_bank_vector_indexes no-ops).
        ``query_vector`` accepts a float list or the pgvector text form.
        Returns memory_units rows with ``score`` (cosine similarity mapped to
        (0, 1]) and ``source="semantic"``.
        """
        vector = _parse_vector(query_vector)
        if not vector:
            return []
        resp = await self.client.search(
            index=_index(table),
            knn={
                "field": "embedding",
                "query_vector": vector,
                "k": k,
                "num_candidates": num_candidates or max(k * 10, 100),
                "filter": _bank_filters(bank_id, fact_type, extra_filters),
            },
            source_includes=_SEARCH_SOURCE_FIELDS,
            size=k,
        )
        return _hits_to_rows(resp, "semantic")

    async def text_search(
        self,
        table: str,
        query_text: str,
        bank_id: str,
        *,
        fact_type: str | list[str] | None = None,
        limit: int = 10,
        fields: tuple[str, ...] = ("text^2", "context", "text_signals"),
        extra_filters: list[dict] | None = None,
    ) -> list[ESRow]:
        """BM25 full-text search on the mapped ``text`` fields.

        Replaces the tsvector/``search_vector`` path — no indexed column to
        maintain, scoring is native BM25. Returns memory_units rows with
        ``score`` and ``source="bm25"``.
        """
        if not query_text or not query_text.strip():
            return []
        resp = await self.client.search(
            index=_index(table),
            query={"bool": {
                "must": [{"multi_match": {
                    "query": query_text,
                    "fields": list(fields),
                    "type": "best_fields",
                }}],
                "filter": _bank_filters(bank_id, fact_type, extra_filters),
            }},
            source_includes=_SEARCH_SOURCE_FIELDS,
            size=limit,
        )
        return _hits_to_rows(resp, "bm25")

    async def get_by_ids(self, table: str, ids: list[str]) -> list[ESRow]:
        """Hydrate documents by their ``id`` field (terms query, batched)."""
        rows: list[ESRow] = []
        str_ids = [str(i) for i in ids]
        for start in range(0, len(str_ids), 10000):
            resp = await self.client.search(
                index=_index(table),
                query={"terms": {"id": str_ids[start:start + 10000]}},
                source_includes=_SEARCH_SOURCE_FIELDS,
                size=10000,
                track_total_hits=False,
            )
            rows.extend(_hits_to_rows(resp, "lookup"))
        return rows

    # -- base.py concrete methods whose defaults emit PG SQL: native overrides ----

    async def bulk_insert_from_arrays(
        self,
        table: str,
        columns: list[str],
        arrays: list[list],
        *,
        column_types: list[str] | None = None,
        returning: str | None = None,
    ) -> list[ESRow] | str:
        """Native override of base.DatabaseConnection.bulk_insert_from_arrays.

        The base default builds ``INSERT ... SELECT * FROM unnest(...)`` and
        calls self.fetch/execute — which this backend rejects by design.
        Same contract, native mechanics: one ``_bulk`` request, one document
        per row. ``column_types`` (PG cast hints) drive value coercion:
        vector -> dense_vector list, json/jsonb -> parsed object, timestamps
        -> ISO-8601. The ``id`` column (or a generated UUID) becomes the ES
        ``_id``; ``RETURNING <cols>`` is honoured client-side.
        """
        if not arrays or not arrays[0]:
            return [] if returning else "INSERT 0 0"
        from .ops_elasticsearch import (
            _iso as _ops_iso,
            _load_json as _ops_load_json,
            _parse_vector as _ops_parse_vector,
            _raise_on_bulk_errors,
        )
        import uuid as _uuid

        types = column_types or [""] * len(columns)
        n_rows = len(arrays[0])
        idx = _index(table)
        ops_payload: list[dict] = []
        docs: list[dict] = []
        for r in range(n_rows):
            doc: dict[str, Any] = {}
            for c, col in enumerate(columns):
                value = arrays[c][r]
                ctype = (types[c] if c < len(types) else "").lower()
                if "vector" in ctype:
                    value = _ops_parse_vector(value)
                elif "json" in ctype:
                    value = _ops_load_json(value, value)
                else:
                    value = _ops_iso(value)
                doc[col] = value
            es_id = str(doc.get("id") or _uuid.uuid4())
            doc.setdefault("id", es_id)
            ops_payload.append({"index": {"_index": idx, "_id": es_id}})
            ops_payload.append(doc)
            docs.append(doc)
        resp = await self.client.bulk(operations=ops_payload, refresh="wait_for")
        _raise_on_bulk_errors(resp, ignore_statuses=())
        if returning:
            wanted = [c.strip() for c in returning.split(",")]
            return [ESRow({c: d.get(c) for c in wanted}) for d in docs]
        return f"INSERT 0 {n_rows}"

    async def copy_records_to_table(
        self,
        table_name: str,
        *,
        records: list[tuple[Any, ...]],
        columns: list[str],
        timeout: float | None = None,
    ) -> None:
        """Native override of base.DatabaseConnection.copy_records_to_table.

        The base default builds an INSERT and calls self.executemany — which
        this backend rejects by design. Row tuples become documents via the
        same ``_bulk`` path as bulk_insert_from_arrays (no cast hints here;
        datetimes are ISO-serialized, everything else passes through).
        """
        if not records:
            return
        arrays: list[list] = [[rec[c] for rec in records] for c in range(len(columns))]
        await self.bulk_insert_from_arrays(table_name, columns, arrays)

    # -- legacy SQL surface: hard errors, no translation ----------------------

    def _no_sql(self, method: str, query: str) -> NativeBackendError:
        return NativeBackendError(
            f"{method}() received SQL but the Elasticsearch backend is "
            f"native-only — nothing is translated. Route this call through "
            f"ElasticsearchOps (backend.data_ops), the native search helpers "
            f"(knn_search / text_search / get_by_ids), or conn.client for a "
            f"raw ES request. Offending statement: {query[:120]!r}"
        )

    async def execute(self, query: str, *args: Any, **kwargs: Any) -> str:
        raise self._no_sql("execute", query)

    async def executemany(self, query: str, *args: Any, **kwargs: Any) -> None:
        raise self._no_sql("executemany", query)

    async def fetch(self, query: str, *args: Any, **kwargs: Any) -> list[ESRow]:
        raise self._no_sql("fetch", query)

    async def fetchrow(self, query: str, *args: Any, **kwargs: Any) -> ESRow | None:
        raise self._no_sql("fetchrow", query)

    async def fetchval(self, query: str, *args: Any, **kwargs: Any) -> Any:
        raise self._no_sql("fetchval", query)

    async def close(self) -> None:
        """Connections are virtual (the client pools HTTP itself)."""
        return None


# ---------------------------------------------------------------------------
# ElasticsearchPool
# ---------------------------------------------------------------------------

class ElasticsearchPool:
    """Pool adapter returned by ``ElasticsearchBackend.get_pool()``.

    Elasticsearch has no exclusive connection checkout: AsyncElasticsearch
    multiplexes requests over its own HTTP connection pool. This adapter
    exists so that pool-shaped callers (``pool.acquire()`` /
    ``pool.release()`` / ``pool.close()``, the asyncpg surface the SQL
    backends expose) keep working — every ``acquire()`` hands out a
    lightweight ``ElasticsearchConnection`` over the shared client, and
    ``release()`` is a no-op since nothing was checked out.
    """

    __slots__ = ("_backend",)

    def __init__(self, backend: "ElasticsearchBackend") -> None:
        self._backend = backend

    @property
    def backend_type(self) -> str:
        """Dispatch key for pool-level consumers (retrieval graph guard)."""
        return "elasticsearch"

    @property
    def client(self) -> Any:
        """The underlying AsyncElasticsearch client."""
        return self._backend._require_client()

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[ElasticsearchConnection]:
        async with self._backend.acquire() as conn:
            yield conn

    async def release(self, conn: ElasticsearchConnection) -> None:
        """No-op: connections are virtual, nothing to give back."""
        return None

    async def close(self) -> None:
        """Close the shared client (pool close == backend close)."""
        await self._backend.close()

    #: asyncpg exposes terminate() as the abrupt variant; same thing here.
    async def terminate(self) -> None:
        await self._backend.close()

    def get_size(self) -> int:
        """Nominal size for pool-introspecting callers: one shared client."""
        return 1


# ---------------------------------------------------------------------------
# ElasticsearchBackend
# ---------------------------------------------------------------------------

class ElasticsearchBackend(DatabaseBackend):
    """DatabaseBackend for Elasticsearch clusters.

    * ``initialize()`` replaces the Alembic startup step: ping the cluster,
      then apply the native migration tree
      (``migrations_elasticsearch.MIGRATIONS`` — baseline creates every
      index from ``ops_elasticsearch.INDEX_MAPPINGS``).
    * ``run_migrations()`` / ``migration_version()`` are the admin-CLI
      equivalents of ``alembic upgrade head`` / ``alembic current``.
    * ``acquire()`` hands out lightweight wrappers over the shared client —
      AsyncElasticsearch manages its own HTTP connection pool, there is no
      exclusive checkout.
    * ``data_ops`` is the ``ElasticsearchOps`` instance every data-access
      call must go through.
    """

    #: environment variables read when constructor args are omitted —
    #: enables the repo's zero-arg factory pattern
    #: (``_get_backend_class(backend_type)()``), like the PostgreSQL backend.
    ENV_URL = "HINDSIGHT_API_DATABASE_URL"
    ENV_SCHEMA = "HINDSIGHT_API_DATABASE_SCHEMA"
    ENV_API_KEY = "HINDSIGHT_API_ES_API_KEY"
    ENV_VERIFY_CERTS = "HINDSIGHT_API_ES_VERIFY_CERTS"
    ENV_EMBEDDING_DIMS = "HINDSIGHT_API_ES_EMBEDDING_DIMS"

    def __init__(
        self,
        database_url: str | None = None,
        schema: str | None = None,
        *,
        api_key: str | None = None,
        verify_certs: bool | None = None,
        embedding_dims: int | None = None,
        client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Every argument is optional; omitted ones resolve from environment
        variables (explicit argument > env var > default), so both call
        styles work:

        * zero-arg factory: ``_get_backend_class("elasticsearch")()``
        * explicit wiring:  ``create_elasticsearch_backend(url, schema=...)``

        A missing URL is not an error here — construction must stay cheap
        and infallible for the factory; ``initialize()`` raises the explicit
        configuration error if no URL was provided by either path.
        """
        self._url = database_url or os.getenv(self.ENV_URL, "")
        self._schema = schema or os.getenv(self.ENV_SCHEMA) or "public"
        self._api_key = api_key or os.getenv(self.ENV_API_KEY) or None
        if verify_certs is None:
            verify_certs = os.getenv(self.ENV_VERIFY_CERTS, "true").lower() != "false"
        self._verify_certs = verify_certs
        if embedding_dims is None:
            raw_dims = os.getenv(self.ENV_EMBEDDING_DIMS)
            embedding_dims = int(raw_dims) if raw_dims else None
        self._embedding_dims = embedding_dims
        self._client_kwargs = client_kwargs or {}
        self._client: Any = None
        self._ops = ElasticsearchOps()
        self._pool = ElasticsearchPool(self)

    # -- identity -------------------------------------------------------------

    @property
    def backend_type(self) -> str:
        return "elasticsearch"

    @property
    def schema(self) -> str:
        return self._schema

    @property
    def data_ops(self) -> ElasticsearchOps:
        """The DataAccessOps implementation for this backend.

        Alias kept for direct users; the engine goes through the inherited
        ``ops`` property (base.py), whose factory resolves the same
        ElasticsearchOps class via ``create_data_access_ops("elasticsearch")``.
        """
        return self._ops

    def get_data_ops(self) -> ElasticsearchOps:
        return self._ops

    # -- capabilities (base.py defaults are PG-shaped; override honestly) ---------

    @property
    def supports_partial_indexes(self) -> bool:
        """No CREATE INDEX ... WHERE: kNN scoping uses query-time filters
        (bank_id/fact_type), see create_bank_vector_indexes (no-op)."""
        return False

    @property
    def supports_bm25(self) -> bool:
        """BM25 is native (text fields); no tsvector column, no extension."""
        return True

    @property
    def supports_unnest(self) -> bool:
        """No unnest(): batch fan-out uses _bulk/msearch in the ops layer."""
        return False

    @property
    def supports_pg_trgm(self) -> bool:
        """No pg_trgm; entity resolution strategy is 'full' (see ops)."""
        return False

    @property
    def supports_worker_poller(self) -> bool:
        """WorkerPoller works: claim_tasks is implemented natively with
        optimistic concurrency (the SKIP LOCKED equivalent)."""
        return True

    # -- lifecycle -------------------------------------------------------------

    @staticmethod
    def _parse_url(url: str) -> tuple[str, str | None, str | None]:
        """Return (es_endpoint, username, password) from a database URL."""
        parsed = urlparse(url)
        scheme = parsed.scheme
        if scheme in ("elasticsearch", "es"):
            http_scheme = "http"
        elif scheme in ("elasticsearch+https", "es+https"):
            http_scheme = "https"
        elif scheme in ("http", "https"):
            http_scheme = scheme
        else:
            raise ValueError(
                f"Unsupported Elasticsearch URL scheme: {scheme!r} "
                "(expected elasticsearch://, elasticsearch+https://, "
                "http:// or https://)"
            )
        host = parsed.hostname or "localhost"
        port = parsed.port or 9200
        return f"{http_scheme}://{host}:{port}", parsed.username, parsed.password

    def _make_client(self, url: str, *, request_timeout: float | None = None) -> Any:
        """Build an AsyncElasticsearch client for ``url`` (shared by
        initialize() and the standalone migration runner).

        Production defaults — each overridable through ``client_kwargs``:
        transparent retries on timeouts/connection errors (the pool-level
        resilience asyncpg callers expect), and HTTP compression (embedding
        payloads are large and compress well).
        """
        AsyncElasticsearch = _import_elasticsearch()
        endpoint, user, password = self._parse_url(url)
        kwargs: dict[str, Any] = {
            "hosts": [endpoint],
            "verify_certs": self._verify_certs,
            "retry_on_timeout": True,
            "max_retries": 3,
            "http_compress": True,
            **self._client_kwargs,
        }
        if request_timeout:
            kwargs.setdefault("request_timeout", request_timeout)
        if self._api_key:
            kwargs["api_key"] = self._api_key
        elif user:
            kwargs["basic_auth"] = (user, password or "")
        return AsyncElasticsearch(**kwargs)

    def _resolve_url(self, dsn: str | None) -> str:
        url = dsn or self._url
        if not url:
            raise ValueError(
                "No Elasticsearch URL configured: pass a dsn, pass "
                "database_url to the constructor, or set "
                f"{self.ENV_URL} "
                "(e.g. elasticsearch://elastic:changeme@localhost:9200)."
            )
        return url

    async def initialize(
        self,
        dsn: str | None = None,
        *,
        min_size: int = 5,
        max_size: int = 20,
        command_timeout: float = 300,
        acquire_timeout: float = 30,
        statement_cache_size: int = 0,
        init_callback: Any | None = None,
    ) -> None:
        """Abstract-method implementation: create the shared client.

        Matches the base signature the engine calls
        (``backend.initialize(self.db_url, min_size=..., init_callback=...)``).
        Pool-tuning arguments map as follows on a clusterized HTTP client:

        * ``command_timeout`` -> the client's ``request_timeout``;
        * ``min_size``/``max_size``/``acquire_timeout``/
          ``statement_cache_size`` have no equivalent (AsyncElasticsearch
          multiplexes over its own HTTP pool; there are no prepared
          statements) — accepted and ignored;
        * ``init_callback`` is asyncpg-specific (receives an
          ``asyncpg.Connection`` to run ``SET ...`` on): there are no
          per-connection sessions to initialize, so it is ignored with a
          debug log rather than silently dropped.
        """
        self._url = self._resolve_url(dsn)
        if init_callback is not None:
            logger.debug(
                "init_callback ignored by the Elasticsearch backend "
                "(no per-connection session to initialize)"
            )
        self._client = self._make_client(self._url, request_timeout=command_timeout)

        info = await self._client.info()
        logger.info(
            "Connected to Elasticsearch %s (cluster=%s), schema prefix %r",
            info.get("version", {}).get("number", "?"),
            info.get("cluster_name", "?"),
            self._schema,
        )
        # Idempotent safety net: the engine runs run_migrations() explicitly
        # before initialize(); standalone users get the baseline for free.
        # Alembic/_dialect.py never runs for this backend — SQL-DDL
        # migrations are deliberate no-ops on Elasticsearch.
        applied = await run_migrations(
            self._client,
            schema=self._schema,
            embedding_dims=self._embedding_dims,
        )
        if applied:
            logger.info("applied Elasticsearch migrations: %s", ", ".join(applied))

    def run_migrations(self, dsn: str | None = None, *, schema: str | None = None) -> None:
        """Abstract-contract implementation, matching base.DatabaseBackend:
        **synchronous**, ``dsn`` positional, called by the engine as
        ``backend.run_migrations(self.db_url, schema=normalize_schema(...))``
        once per tenant, *before* ``initialize()``.

        PG delegates to Alembic (sync); this backend runs its native async
        migration runner on a dedicated event loop **in a separate thread**,
        so the call works identically from plain sync code and from inside a
        running event loop (the engine calls it un-awaited from an async
        method — ``asyncio.run()`` in-thread would raise there). It builds
        its own short-lived client from ``dsn`` because the shared client
        does not exist yet at this point in the startup sequence.
        """
        url = self._resolve_url(dsn)
        target_schema = schema or self._schema
        result: dict[str, Any] = {}

        def _worker() -> None:
            async def _inner() -> None:
                client = self._make_client(url)
                try:
                    applied = await run_migrations(
                        client,
                        schema=target_schema,
                        embedding_dims=self._embedding_dims,
                    )
                    if applied:
                        logger.info(
                            "applied Elasticsearch migrations for schema %r: %s",
                            target_schema, ", ".join(applied),
                        )
                finally:
                    await client.close()
            try:
                asyncio.run(_inner())
            except BaseException as exc:  # propagated to the caller below
                result["error"] = exc

        thread = threading.Thread(
            target=_worker, name=f"es-migrations-{target_schema}", daemon=True
        )
        thread.start()
        thread.join()
        if "error" in result:
            raise result["error"]

    async def run_migrations_async(self, schema: str | None = None) -> list[str]:
        """Async variant for callers already holding the initialized backend
        (admin CLI, tests). Returns the versions applied by this call
        (idempotent; empty list when up to date).
        """
        return await run_migrations(
            self._require_client(),
            schema=schema or self._schema,
            embedding_dims=self._embedding_dims,
        )

    async def migration_version(self, schema: str | None = None) -> str | None:
        """Highest applied migration version (``alembic current`` equivalent)."""
        return await current_version(
            self._require_client(), schema=schema or self._schema
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def shutdown(self) -> None:
        """Abstract-method implementation: graceful teardown.

        The SQL backends drain and close their connection pools here; the
        only resource this backend holds is the shared AsyncElasticsearch
        client, so shutdown == close. Idempotent (safe to call twice or
        before initialize()).
        """
        await self.close()

    # -- transaction (abstract-method implementation) ------------------------------

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[ElasticsearchConnection]:
        """Abstract-method implementation: backend-level transaction scope.

        SQL backends implement this as "acquire a pooled connection and open
        a transaction on it" (``async with backend.transaction() as conn:``).
        Elasticsearch has no multi-document transactions, so this yields a
        connection wrapper over the shared client and the scope is a no-op:
        nothing is rolled back on failure. Atomicity lives one level up, in
        ElasticsearchOps (deterministic ids, ``op_type=create``, optimistic
        concurrency) — every ops operation is individually idempotent.
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn

    # -- pool (abstract-method implementation) ------------------------------------

    def get_pool(self) -> ElasticsearchPool:
        """Abstract-method implementation: the pool-shaped adapter.

        There is no real pool — AsyncElasticsearch multiplexes over its own
        HTTP connections — so this returns the ``ElasticsearchPool`` adapter
        whose ``acquire()``/``release()``/``close()`` delegate to the
        backend. Callable before ``initialize()`` (the adapter resolves the
        client lazily and raises the explicit not-initialized error only
        when actually used).
        """
        return self._pool

    # -- connection handout -------------------------------------------------------

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError(
                "ElasticsearchBackend is not initialized — call initialize() "
                "before acquiring connections."
            )
        return self._client

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[ElasticsearchConnection]:
        yield ElasticsearchConnection(self._require_client(), self._schema)

    @asynccontextmanager
    async def acquire_read(self) -> AsyncIterator[ElasticsearchConnection]:
        """Read path: identical to acquire(). ES routes reads to replica
        shards natively; no separate read-replica pool is needed.
        """
        yield ElasticsearchConnection(self._require_client(), self._schema)

    def connection(self) -> ElasticsearchConnection:
        """Direct (non-context-managed) wrapper, for process-lifetime holders."""
        return ElasticsearchConnection(self._require_client(), self._schema)


# ---------------------------------------------------------------------------
# Factory hook
# ---------------------------------------------------------------------------

def is_elasticsearch_url(database_url: str) -> bool:
    """Scheme test for the backend factory (see README for the wiring)."""
    scheme = urlparse(database_url).scheme
    return scheme in ("elasticsearch", "elasticsearch+https", "es", "es+https")


def create_elasticsearch_backend(
    database_url: str,
    schema: str = "public",
    **kwargs: Any,
) -> ElasticsearchBackend:
    """Build (but do not initialize) an ElasticsearchBackend."""
    return ElasticsearchBackend(database_url, schema=schema, **kwargs)
