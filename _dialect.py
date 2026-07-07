"""Dialect dispatcher for Alembic migrations.

Each migration file declares a ``_pg_upgrade``/``_oracle_upgrade`` (and matching
downgrades) function and routes ``upgrade()``/``downgrade()`` through
``run_for_dialect``. The helper inspects the live connection's dialect name and
runs the matching function — or no-ops if the migration doesn't apply to the
current backend.

Use ``None`` (or omit the kwarg) when a migration intentionally has no effect
on a dialect; the helper treats it as a no-op.

Elasticsearch note
------------------
The Elasticsearch backend does not normally reach this module at all: it has
no SQLAlchemy dialect, and its schema evolution (index mappings) runs through
the native runner in ``hindsight_api.engine.db.migrations_elasticsearch`` —
invoked by ``ElasticsearchBackend.initialize()`` and by the admin CLI, which
must skip Alembic when ``is_elasticsearch_url(database_url)`` is true.

The ``es`` kwarg exists for the edge case where an Elasticsearch-backed
deployment is nevertheless driven through the Alembic tree (e.g. a custom
env.py registering a shim dialect named ``elasticsearch``): existing SQL-DDL
migrations simply omit it and become no-ops instead of raising, and a
migration that *does* have an ES-side effect can pass a function that
delegates to the native runner.
"""

from __future__ import annotations

from collections.abc import Callable

from alembic import op

DialectFn = Callable[[], None]

_SUPPORTED = ("postgresql", "oracle", "elasticsearch")


def run_for_dialect(
    *,
    pg: DialectFn | None = None,
    oracle: DialectFn | None = None,
    es: DialectFn | None = None,
) -> None:
    """Dispatch to the function matching the current bind's dialect.

    Args:
        pg: Function to run when the active bind is PostgreSQL.
        oracle: Function to run when the active bind is Oracle.
        es: Function to run when the active bind is Elasticsearch. SQL-DDL
            migrations omit it — the migration is then a deliberate no-op on
            Elasticsearch, whose schema lives in native index mappings
            (see migrations_elasticsearch).

    Unrecognized dialects raise; an explicit ``None`` for the active dialect
    is a no-op (the migration deliberately does nothing here).
    """
    name = op.get_bind().dialect.name
    if name not in _SUPPORTED:
        raise RuntimeError(f"Unsupported dialect for migration dispatch: {name!r}. Expected one of {_SUPPORTED}.")
    fn = {"postgresql": pg, "oracle": oracle, "elasticsearch": es}[name]
    if fn is not None:
        fn()
