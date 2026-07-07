# Elasticsearch Backend for Hindsight

A native Elasticsearch storage and retrieval backend for
[Hindsight](https://github.com/vectorize-io/hindsight), the open-source agent
memory system by Vectorize. This driver lets you run the Hindsight API
(`hindsight-api-slim`) against an Elasticsearch 8.x/9.x cluster instead of
PostgreSQL or Oracle.

**Native end to end — zero SQL translation.** Elasticsearch is not a SQL
engine, and this backend never pretends otherwise. Every data path speaks
Elasticsearch natively: the `_bulk` API for batch writes, `msearch` for
fan-out queries, aggregations for grouping, kNN on `dense_vector` for
semantic search, BM25 `multi_match` for full-text, and optimistic
concurrency control (`if_seq_no` / `if_primary_term`) where SQL backends use
row locks. Any legacy SQL that still reaches the connection raises an
explicit `NativeBackendError` naming the native replacement — a deliberate
porting signal, never a silent mistranslation.

## What's included

```
hindsight-api-slim/hindsight_api/
├── engine/
│   ├── db/
│   │   ├── ops_elasticsearch.py          # DataAccessOps implementation (28 methods)
│   │   ├── elasticsearch.py              # Connection layer: backend, pool, native search
│   │   └── migrations_elasticsearch.py   # Native schema migrations (Alembic equivalent)
│   ├── sql/
│   │   └── elasticsearch.py              # Inert SQLDialect (query plans build, never run)
│   └── search/
│       ├── retrieval.py                  # PATCHED: 3 minimal dispatch hooks
│       └── retrieval_elasticsearch.py    # Native 4-arm recall (semantic/BM25/temporal/graph)
└── alembic/
    └── _dialect.py                       # PATCHED: recognizes the elasticsearch dialect
```

Six files are new; two (`retrieval.py`, `_dialect.py`) replace upstream files
with minimal, backward-compatible patches — existing PostgreSQL and Oracle
behavior is untouched.

## How it maps

| SQL / PostgreSQL concept | Elasticsearch equivalent |
|---|---|
| Table (`schema.table`) | Index (`schema-table`) — schemas become index prefixes (multi-tenant) |
| `INSERT ... unnest()` batches | `_bulk` API |
| `ON CONFLICT DO NOTHING` | Deterministic `_id` + `op_type=create` (409 ignored) |
| Upsert | Bulk `update` with `doc_as_upsert` |
| `FOR UPDATE SKIP LOCKED` | Conditional update with `if_seq_no` / `if_primary_term` |
| `RETURNING id` | Client-side UUID generation |
| `CROSS JOIN LATERAL` fan-out | `msearch` (one sub-search per row) |
| pgvector `<=>` + partial HNSW indexes | `dense_vector` kNN with query-time `bank_id`/`fact_type` filters |
| tsvector / `search_vector` column | Native BM25 on mapped `text` fields (no column to maintain) |
| Advisory locks (migrations) | Version claim via `op_type=create` on a tracking index |
| Alembic migration tree | Ordered native migration steps + `{schema}-hindsight_migrations` index |
| Transactions | None — atomicity via deterministic ids, idempotent ops, OCC |

## Feature highlights

- **Full `DataAccessOps` contract** (28 methods): batch fact insertion,
  link/entity/co-occurrence management, graph maintenance queue with
  exactly-once claiming, webhook CRUD, and a native task queue implementing
  the SKIP LOCKED worker pattern with per-type reserved limits, bank-serialized
  consolidation, and wildcard priority tiers.
- **Full `DatabaseBackend` / `DatabaseConnection` contract**, validated
  against the upstream ABCs: zero-arg factory construction (env-driven
  config), pool adapter, backend-level transactions, graceful shutdown, and
  native overrides of the base-class helpers that would otherwise emit SQL
  (`bulk_insert_from_arrays`, `copy_records_to_table`).
- **4-arm native recall**, dispatched inside `retrieval.py` on
  `conn.backend_type` (the same key the dialect selection already uses):
  - *Semantic*: kNN with the score remapped to cosine similarity
    (`2 × _score − 1`), HNSW over-fetch expressed as `num_candidates`.
  - *BM25*: `multi_match` over `text^2`, `context`, `text_signals`.
  - *Temporal*: the SQL window predicate ported verbatim to `bool`/`should`,
    the upstream coverage selection reused as-is, and the full
    link-spreading algorithm with identical formulas (causal boosts 2.0/1.5,
    propagation `parent × weight × boost × 0.7`, frontier gate 0.2,
    weight floor 0.1, top-10 per source, 5 iterations max).
  - *Graph*: `ElasticsearchGraphRetriever` (implements the `GraphRetriever`
    ABC) — kNN seeds, then entity-overlap and semantic/causal link expansion
    through the ops layer, activation-scored and post-filtered by the
    project's own `tags.py` functions.
- **Exact tag semantics**: all five `TagsMatch` modes (`any`/`all` include
  untagged memories; `_strict` variants exclude them; `exact` is
  set-equality with the empty scope matching only untagged rows) and
  recursive `TagGroup` And/Or/Not trees, verified equivalent to the upstream
  Python matcher across the full mode × request × document matrix.
- **Native migrations**: linear versioned tree, concurrent-safe claiming,
  failed-step tracking that blocks subsequent startups instead of leaving
  holes, and DDL helpers (`create_index`, additive `put_mapping`,
  `reindex_to` with alias swap for breaking changes).

## Installation

### 1. Prerequisites

- A running Hindsight `hindsight-api-slim` checkout (this driver targets its
  `engine/db`, `engine/sql`, `engine/search`, and `alembic` packages).
- An Elasticsearch 8.x or 9.x cluster.
- The async Python client:

```bash
pip install 'elasticsearch>=8'
```

The import is lazy (same pattern as the `oracledb` dependency): nothing
breaks if the package is absent and the backend is unused.

### 2. Drop in the files

Unzip at the repository root — the archive mirrors the target tree. Only
`engine/search/retrieval.py` and `alembic/_dialect.py` overwrite existing
files; review those diffs if you carry local patches.

### 3. Wire the factories (two one-line registrations)

The backend follows the repo's zero-argument factory pattern
(`_get_backend_class(backend_type)()`), configuring itself from the
environment like the PostgreSQL backend does.

In `engine/db/__init__.py`, register the class:

```python
if backend_type == "elasticsearch":
    from .elasticsearch import ElasticsearchBackend
    return ElasticsearchBackend
```

and the ops class:

```python
if backend_type == "elasticsearch":
    from .ops_elasticsearch import ElasticsearchOps
    return ElasticsearchOps
```

In `engine/sql/__init__.py`, register the dialect:

```python
if backend_type == "elasticsearch":
    from .elasticsearch import ESDialect
    return ESDialect()
```

Where the backend type is derived from the database URL, route the scheme:

```python
from .elasticsearch import is_elasticsearch_url

if is_elasticsearch_url(database_url):
    backend_type = "elasticsearch"
```

### 4. Configure

```bash
# Any of these URL forms:
export HINDSIGHT_API_DATABASE_URL=elasticsearch://elastic:changeme@localhost:9200
# elasticsearch+https://elastic:pwd@es.prod:9243
# https://elastic:pwd@es.prod:9243

# Optional:
export HINDSIGHT_API_ES_API_KEY=...            # API-key auth (replaces user:pass)
export HINDSIGHT_API_ES_VERIFY_CERTS=false     # self-signed TLS in dev
export HINDSIGHT_API_ES_EMBEDDING_DIMS=384     # pin dense_vector dims
                                               # (bge-small=384, OpenAI=1536/3072)
export HINDSIGHT_API_DATABASE_SCHEMA=public    # becomes the index prefix
```

Explicit constructor arguments always take precedence over environment
variables.

### 5. Run

Start the API as usual. On startup the backend pings the cluster and applies
the native migration tree (baseline: all indexes from `INDEX_MAPPINGS`,
including the cosine `dense_vector` embedding field). Migrations are
idempotent and concurrent-safe across multiple starting processes.

```python
# Programmatic lifecycle, if you embed the backend directly:
backend = create_elasticsearch_backend(url, schema="public")
await backend.initialize()
async with backend.acquire() as conn:
    ids = await backend.data_ops.insert_facts_batch(conn, bank_id, ...)
    hits = await conn.knn_search("public.memory_units", query_vector,
                                 bank_id=bank_id, fact_type="world", k=20)
await backend.shutdown()
```

## Adding a schema migration

Append to `MIGRATIONS` in `migrations_elasticsearch.py` — never reorder,
never edit an applied step:

```python
async def _0002_add_confidence(client, schema, ctx):
    await put_mapping(client, schema, "memory_units",
                      {"confidence": {"type": "float"}})

MIGRATIONS.append(Migration("0002", "add confidence", _0002_add_confidence))
```

`backend.run_migrations(dsn, schema=...)` (sync, matching the base contract
the engine calls per tenant) and `backend.migration_version()` are the
`alembic upgrade head` / `alembic current` equivalents.

## Performance notes

- **Single-round-trip recall.** The semantic + BM25 arms for every fact type
  travel in one `msearch` request (the network-level analogue of the SQL
  backend's single `UNION ALL` statement), with an automatic per-fact-type
  fallback if the batched form is unavailable. BM25 floors are pushed
  server-side (`min_score`); graph expansions run concurrently; the temporal
  spreading's neighbor scoring applies its similarity threshold server-side.
- **Resilient by default.** The client retries on timeouts (3 attempts) and
  compresses HTTP payloads (embeddings compress well) — both overridable via
  `client_kwargs`. A graph-arm failure degrades to an empty arm with a
  warning instead of failing the whole recall.

## Known limitations

- **No transactions.** `transaction()` scopes are no-ops; nothing rolls back
  on failure. Every ops-layer operation is individually idempotent by
  design (deterministic ids + OCC), which is the guarantee this backend
  offers instead.
- **Raw SQL paths raise.** Engine code paths that still emit SQL through
  `conn.fetch`/`execute` (currently: tag listing, profile, health) raise
  `NativeBackendError` with the native replacement named. They are the
  remaining porting surface, on the same pattern as the recall port.
- **Documented divergences.** Time-range filters apply to `created_at` (the
  field the ES schema tracks) where SQL filters `updated_at`; graph-arm
  `created_after/before` apply at seed time only; `exact` tag matching
  assumes de-duplicated tags per unit (de facto true).
- **Entity resolution** uses the portable `"full"` strategy (no
  pg_trgm/UTL_MATCH equivalent); ES fuzzy matching can be layered on by
  consumers aware of this backend.
- Writes default to `refresh="wait_for"` for read-after-write correctness in
  retain/recall flows; tune `ElasticsearchOps.REFRESH` under heavy ingest.

## Testing

The driver was validated against the upstream abstract base classes
(`DataAccessOps`, `DatabaseBackend`, `DatabaseConnection`, `SQLDialect`,
`GraphRetriever`) and the engine's actual call sequences, with mock-cluster
integration tests covering: bulk write semantics and conflict handling,
exactly-once queue claiming under concurrency, migration idempotence and
failure blocking, the full 4-arm recall (temporal spreading formulas checked
to two decimal places), and tag-filter equivalence against the upstream
Python matcher (90 mode/request/document combinations plus recursive
groups). Running against a live cluster requires only the standard
`docker run -p 9200:9200 elasticsearch:8.x` plus the environment variables
above.

## Compatibility

- Elasticsearch **8.x / 9.x** (uses `dense_vector` kNN, `script_score`
  `cosineSimilarity`, `_bulk`, `msearch`, terms aggregations, OCC).
- Python **3.11+**, `elasticsearch>=8` (async client).
- Hindsight `hindsight-api-slim` with the pluggable-backend architecture
  (PostgreSQL/Oracle dispatch via `DataAccessOps` / `SQLDialect`).

## License

MIT — the same license as
[Hindsight](https://github.com/vectorize-io/hindsight/blob/main/LICENSE)
(Copyright © 2025 Vectorize AI, Inc.), which this driver extends.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*Hindsight™ is a trademark of Vectorize AI, Inc. This is a community
backend contribution and is not officially endorsed by Vectorize.*

---

*Built with 🐢 by K1SS Atelier 0 — JS & Kage. Besançon, France.*
