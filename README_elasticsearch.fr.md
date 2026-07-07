# Backend Elasticsearch pour Hindsight — `engine/db/`

Fichiers à déposer dans `hindsight-api-slim/hindsight_api/engine/db/` :

| Fichier | Rôle | Équivalent existant dans le repo |
|---|---|---|
| `ops_elasticsearch.py` | **Source de vérité.** `ElasticsearchOps` (implémente `DataAccessOps`) + `INDEX_MAPPINGS` + `ensure_indexes()` — tout l'accès aux données, 100 % natif (bulk, msearch, aggregations, OCC) | `ops_postgresql.py` / `ops_oracle.py` |
| `elasticsearch.py` | Couche connexion calée sur `ops_elasticsearch.py` : `ElasticsearchConnection` + `ElasticsearchBackend`. **Aucune traduction SQL** | `oracle.py` |
| `migrations_elasticsearch.py` | Migrations de schéma natives (arbre ordonné + index de tracking + claim concurrent) | l'arbre `hindsight_api/alembic/` |

Un fichier à déposer dans `hindsight-api-slim/hindsight_api/engine/sql/` :

| Fichier | Rôle | Équivalent existant dans le repo |
|---|---|---|
| `elasticsearch.py` (sql) | `ESDialect` conforme à l'ABC `SQLDialect` : fragments PG-style inertes, jamais exécutés (le plan SQL se construit, l'exécution passe en natif) | `postgresql.py` / `oracle.py` |

Et un fichier à **remplacer** dans `hindsight-api-slim/hindsight_api/alembic/` :

| Fichier | Correction |
|---|---|
| `_dialect.py` | `run_for_dialect(*, pg=None, oracle=None, es=None)` : le dialecte `elasticsearch` est reconnu et **no-op par défaut** — les migrations SQL-DDL existantes n'ont pas à changer, elles ne s'appliquent simplement pas à ES |

## Principe : natif de bout en bout, zéro SQL

Contrairement au backend Oracle (qui réécrit le SQL PG car Oracle reste du
SQL), ce backend ne mappe **rien** :

- Tout l'accès aux données passe par `ElasticsearchOps` (`backend.data_ops`),
  qui parle directement `_bulk`, `msearch`, aggregations et concurrence
  optimiste (`if_seq_no`/`if_primary_term`).
- `ElasticsearchConnection` est l'objet que consomme la couche ops : il
  expose le client brut via `.client` (ce que `ops_elasticsearch._es(conn)`
  déballe), plus les primitives de retrieval natives hors DataAccessOps :

  ```python
  # remplaçant pgvector (ORDER BY embedding <=> $1) :
  rows = await conn.knn_search("public.memory_units", query_vector,
                               bank_id="bank1", fact_type="world", k=20)

  # remplaçant tsvector/search_vector (BM25 natif) :
  rows = await conn.text_search("public.memory_units", "alice google",
                                bank_id="bank1", fact_type="world", limit=20)

  # hydratation par ids :
  rows = await conn.get_by_ids("public.memory_units", ids)
  ```

  Chaque row est un `ESRow` (le type de `ops_elasticsearch`) avec les champs
  memory_units + `score` + `source` ("semantic" / "bm25" / "lookup").
- Les points d'entrée SQL hérités (`execute`, `fetch`, `fetchrow`,
  `fetchval`, `executemany`) existent uniquement pour satisfaire l'interface
  `DatabaseConnection` et lèvent immédiatement `NativeBackendError` en
  nommant le remplaçant. Un appelant qui émet encore du SQL doit être porté,
  pas traduit.

## Dépendance

```bash
pip install 'elasticsearch>=8'
```

Import paresseux (pattern `oracledb` de `oracle.py`) : rien ne casse si le
paquet est absent et que le backend n'est pas utilisé.

## Câblage dans la factory de backend

La factory du repo (`engine/db/__init__.py`) instancie les backends **sans
argument** — `_get_backend_class(backend_type)()` — chaque backend se
configurant depuis l'environnement (comme PostgreSQL). `ElasticsearchBackend`
suit ce contrat : tous ses paramètres de constructeur sont optionnels et se
résolvent en `argument explicite > variable d'env > défaut`. Le câblage se
réduit donc à deux lignes :

**1. Enregistrer la classe** dans le registre de `_get_backend_class` :

```python
from .elasticsearch import ElasticsearchBackend
# dans le mapping backend_type -> classe :
"elasticsearch": ElasticsearchBackend,
```

**2. Router le `backend_type`** là où il est déduit de l'URL :

```python
from .elasticsearch import is_elasticsearch_url

if is_elasticsearch_url(database_url):
    backend_type = "elasticsearch"
```

C'est tout : `_get_backend_class("elasticsearch")()` fonctionne alors tel
quel, le backend lisant `HINDSIGHT_API_DATABASE_URL`,
`HINDSIGHT_API_DATABASE_SCHEMA`, `HINDSIGHT_API_ES_API_KEY`,
`HINDSIGHT_API_ES_VERIFY_CERTS` et `HINDSIGHT_API_ES_EMBEDDING_DIMS`.
Une URL absente ne fait pas échouer la construction (la factory doit rester
infaillible) ; c'est `initialize()` qui lève un `ValueError` explicite.

Le câblage explicite reste disponible pour les usages hors factory
(tests, scripts) :

```python
backend = create_elasticsearch_backend(url, schema=schema, api_key=..., ...)
```

Et là où le moteur choisit son `DataAccessOps` (`PostgreSQLOps` vs `OracleOps`) :

```python
ops = backend.data_ops          # -> ElasticsearchOps
```

## Configuration

```bash
export HINDSIGHT_API_DATABASE_URL=elasticsearch://elastic:changeme@localhost:9200
# ou : elasticsearch+https://elastic:pwd@es.prod:9243  /  https://...

export HINDSIGHT_API_ES_API_KEY=...            # optionnel, remplace user:pass
export HINDSIGHT_API_ES_VERIFY_CERTS=false     # optionnel, TLS auto-signé en dev
export HINDSIGHT_API_ES_EMBEDDING_DIMS=384     # optionnel, fige dense_vector
                                               # (bge-small=384, OpenAI=1536/3072)
```

`HINDSIGHT_API_DATABASE_SCHEMA` (ex. `tenant_acme`) devient un **préfixe
d'index** : `tenant_acme.memory_units` → index `tenant_acme-memory_units`
(équivalent du `ALTER SESSION SET CURRENT_SCHEMA` d'Oracle).

## Cycle de vie

```python
backend = create_elasticsearch_backend(url, schema="public")
await backend.initialize()      # ping + application des migrations natives
                                # (baseline = tous les index de INDEX_MAPPINGS)
async with backend.acquire() as conn:
    ops = backend.data_ops
    ids = await ops.insert_facts_batch(conn, bank_id, ...)     # écriture native
    hits = await conn.knn_search("public.memory_units", vec,
                                 bank_id=bank_id, fact_type="world")
await backend.shutdown()        # teardown gracieux (== close, idempotent)
```

### Interface `DatabaseBackend` complète

Passe de conformité faite contre le **vrai** `base.py` du repo (et la séquence
de démarrage de `memory_engine.py`) :

- **`initialize(dsn, *, min_size, max_size, command_timeout, acquire_timeout,
  statement_cache_size, init_callback)`** — signature de base honorée.
  `command_timeout` → `request_timeout` du client ; les paramètres de pool
  n'ont pas d'équivalent (le client multiplexe son propre pool HTTP) et sont
  acceptés/ignorés ; `init_callback` (spécifique asyncpg, il exécute des
  `SET ...`) est ignoré avec un log debug.
- **`run_migrations(dsn, *, schema=None)`** — **synchrone**, conforme à la
  base : le moteur l'appelle sans `await` par tenant, avant `initialize()`.
  Le runner async natif tourne sur une event loop dédiée dans un thread
  séparé (fonctionne depuis du code sync comme depuis l'event loop), avec un
  client éphémère construit depuis le `dsn`. La variante async pour le CLI
  s'appelle désormais `run_migrations_async(schema=None)`.
- **`get_pool()`** → `ElasticsearchPool` (`acquire()`/`release()` no-op/
  `close()`/`get_size()`/`.client`) ; **`shutdown()`** == close, idempotent ;
  **`transaction()`** au niveau backend (acquire + scope no-op documenté).
- **Capacités** surchargées honnêtement : `supports_partial_indexes=False`,
  `supports_bm25=True` (natif), `supports_unnest=False`,
  `supports_pg_trgm=False`, `supports_worker_poller=True` (claim_tasks
  natif en OCC).
- **`ops`** : la propriété héritée de `base.py` fonctionne telle quelle via
  votre factory (`create_data_access_ops("elasticsearch")` →
  `ElasticsearchOps`).
- **Méthodes concrètes de `DatabaseConnection` surchargées** : les défauts de
  `bulk_insert_from_arrays` (unnest PG) et `copy_records_to_table`
  (executemany) génèrent du SQL — réimplémentées nativement en `_bulk`, avec
  coercition guidée par `column_types` (`vector` → dense_vector,
  `json` → objet, timestamps → ISO) et `RETURNING` honoré côté client.
  `parse_json` héritée fonctionne telle quelle (ES renvoie des objets).

### Points chauds restants dans `memory_engine.py` (hors périmètre backend)

1. ~~`create_sql_dialect("elasticsearch")`~~ — **fermé** : le fichier
   `engine/sql/elasticsearch.py` livré fournit `ESDialect`, conforme à
   l'ABC `SQLDialect` (27 abstraites), retourné par la factory déjà câblée.
   Philosophie : fragments PG-style inertes et bien formés — le plan de
   requête SQL se construit sans crasher, mais ce SQL n'atteint jamais un
   moteur (toute exécution lève `NativeBackendError` côté connexion).
   Le squelette initial passait l'instanciation mais levait `TypeError` à
   l'appel sur `upsert` (3 args vs 4), `array_contains` (1 arg vs 2),
   `greatest` (binaire vs variadique) et `bulk_unnest` (2 listes vs liste
   de tuples) — signatures réalignées argument par argument sur `base.py` ;
   les arms sémantique/BM25 gardent la forme de colonnes des arms PG
   (`similarity`, `bm25_score`, `source`) pour rester composables en
   `UNION ALL` ; les méthodes extra du squelette (hors ABC) sont conservées
   avec arité corrigée.
2. ~~Recall (arms SQL de `retrieval.py`)~~ — **fermé** : `retrieval.py`
   (patché, livré) reçoit trois hooks minimaux qui dispatchent vers
   `retrieval_elasticsearch.py` (nouveau) quand
   `conn.backend_type == "elasticsearch"` — même clé de dispatch que la
   sélection de dialecte existante. Le patch de `_search_with_retries`
   dans `memory_engine.py` devient supprimable.
   - **Sémantique** → `conn.knn_search()` ; similarité remappée
     `2*_score - 1` (le `_score` kNN cosinus d'ES vaut `(1+cos)/2`) pour
     retrouver l'échelle du `1 - (embedding <=> $1)` SQL ; floors
     `min_semantic`/`min_keyword` appliqués sur les valeurs remappées ;
     sur-échantillonnage HNSW exprimé en `num_candidates`.
   - **BM25** → `conn.text_search()` (multi_match natif).
   - **Temporel** → kNN filtré par le prédicat de fenêtre porté à
     l'identique en bool/should, réutilisation de
     `_select_with_temporal_coverage` (pur Python), puis **spreading natif
     complet** avec les formules copiées à la virgule près : proximité au
     milieu de fenêtre, boosts causaux 2.0/1.5, propagation
     `parent × weight × boost × 0.7`, seuil frontier 0.2, plancher de
     poids 0.1, top-10 par source, batch 20, 5 itérations max. Les liens
     viennent de l'index `memory_links`, la similarité des voisins d'une
     requête `script_score` cosineSimilarity.
   - **Graphe** → `ElasticsearchGraphRetriever` (hérite de l'ABC
     `GraphRetriever`), substitué par `retrieval.py` quand le pool est ES et
     qu'aucun retriever explicite n'est passé. Pipeline : seeds (fournis ou
     kNN top-10) → `ops.expand_semantic_causal` + `ops.expand_entities` →
     `activation` ∈ [0,1] (poids de lien ; comptage d'entités partagées
     normalisé au max ; max-merge par unité) → post-filtrage via les
     `filter_results_by_tags`/`filter_results_by_tag_groups` de `tags.py`
     (le même chemin de code que les résultats graphe SQL) → cap au budget.
     `GraphRetrievalTimings` renseigné (seeds_time, db_queries, edge_count,
     traverse, result_count).
   - **Tags** : portage exact des 5 modes de `TagsMatch` — `any`/`all`
     **incluent les non-tagués** (should [must_not exists, overlap/superset]),
     `any_strict`/`all_strict` les excluent, `exact` = égalité d'ensembles
     (term-par-tag + script de cardinalité ; scope vide = non-tagués
     uniquement). `TagGroup` récursifs And/Or/Not traduits en bool
     filter/should/must_not, formes pydantic et dict acceptées. Équivalence
     vérifiée contre `filter_results_by_tags`/`_match_group` de `tags.py`
     sur 90 combinaisons + groupes composés.
   - Les rows repassent par `RetrievalResult.from_db_row` (vérifié contre le
     vrai `types.py` : `.get()` tolérant, `id`/`text`/`fact_type` requis) ;
     les dates ISO d'ES sont reparsées en datetimes aware. Divergences
     documentées : filtre temporel sur `created_at` là où le SQL filtre
     `updated_at` ; pour l'arm graphe, `created_after/before` ne s'appliquent
     qu'aux seeds (les rows d'expansion ne portent pas `created_at`) ;
     l'`exact` ES suppose des tags dédupliqués par unité (de facto le cas).
3. Ligne ~9665 : `ops.build_tag_listing_parts(...)` + SQL — lèvera
   `NotImplementedError` sur le listing de tags ; brancher sur
   `ops.list_tags(conn, table, bank_id)` pour ES.
4. Profile et health (signalés par JS & Kage) : chemins SQL →
   `NativeBackendError` explicite, à porter sur la couche ops.

Tout chemin qui appelle `conn.fetch/execute` avec du SQL brut lèvera
`NativeBackendError` en nommant le remplaçant — c'est le signal de portage,
par conception.

## Migrations : `_dialect.py` et le runner natif

Alembic pilote du DDL SQL via SQLAlchemy ; Elasticsearch n'a ni dialecte
SQLAlchemy ni DDL — son schéma est un ensemble de mappings d'index. La
correction se fait donc en deux moitiés cohérentes :

**1. `_dialect.py` (côté Alembic).** `run_for_dialect` accepte maintenant
`es=...` et reconnaît le dialecte `elasticsearch`. Les migrations existantes
(`run_for_dialect(pg=..., oracle=...)`) restent inchangées : sur ES elles
deviennent des no-op délibérés au lieu de lever `RuntimeError`. Ce chemin
n'est qu'un filet de sécurité — en fonctionnement normal, Alembic ne tourne
jamais pour ce backend.

**2. `migrations_elasticsearch.py` (le vrai chemin).** L'équivalent natif de
l'arbre Alembic :

- `MIGRATIONS` : liste ordonnée de `Migration(version, description, apply)`,
  baseline `0001` = création de tous les index depuis `INDEX_MAPPINGS`.
  On **ajoute** en fin de liste, on ne réordonne jamais.
- Index de tracking `{schema}-hindsight_migrations` (l'`alembic_version`) :
  un document par version appliquée.
- Concurrence sans advisory lock : chaque version est *réclamée* par
  `op_type=create` sur un `_id` déterministe — un seul process concurrent
  applique chaque étape, les autres passent (409). Une étape en échec est
  marquée `status=failed` et bloque les démarrages suivants jusqu'à
  correction, au lieu de laisser un trou silencieux dans l'arbre.
- Helpers DDL natifs : `create_index` (idempotent), `put_mapping`
  (ALTER ADD COLUMN — additif seulement, comme ES l'impose), `reindex_to`
  (changement cassant : nouvel index + `_reindex` + bascule d'alias).

Exemple d'ajout de migration :

```python
async def _0002_add_confidence(client, schema, ctx):
    await put_mapping(client, schema, "memory_units",
                      {"confidence": {"type": "float"}})

MIGRATIONS.append(Migration("0002", "add confidence", _0002_add_confidence))
```

**Câblage du CLI admin.** Là où `run-db-migration` construit l'engine
SQLAlchemy et invoque Alembic, brancher d'abord :

```python
if is_elasticsearch_url(database_url):
    backend = create_elasticsearch_backend(database_url, schema=schema, ...)
    await backend.initialize()          # applique les migrations
    print(await backend.migration_version())   # ⇔ alembic current
    await backend.close()
    return
# sinon : chemin Alembic habituel (pg / oracle)
```

`backend.run_migrations()` (⇔ `alembic upgrade head`) et
`backend.migration_version()` (⇔ `alembic current`) sont aussi exposés pour
un usage direct. Pas de downgrade — même posture que le repo (upgrade only).

## Sémantique (portée par `ops_elasticsearch.py`)

- Pas de transactions : `conn.transaction()` est un no-op documenté ;
  l'atomicité vient des `_id` déterministes + `op_type=create`
  (⇔ `ON CONFLICT DO NOTHING`) et des updates/deletes conditionnels
  `if_seq_no`/`if_primary_term` (⇔ `FOR UPDATE SKIP LOCKED`).
- Les méthodes de l'ABC qui retournent des fragments SQL
  (`build_entity_expansion_cte`, `build_semantic_causal_cte`,
  `build_tag_listing_parts`) lèvent `NotImplementedError` ; leurs
  équivalents natifs exécutables sont `expand_entities()`,
  `expand_semantic_causal()`, `list_tags()`.
- `create/drop_bank_vector_indexes` sont des no-op : le kNN se filtre par
  `bank_id`/`fact_type` (c'est ce que fait `conn.knn_search`).
- Écritures en `refresh="wait_for"` pour le read-after-write des flux
  retain/recall (`ElasticsearchOps.REFRESH`, ajustable sous forte charge).
- `get_entity_resolution_strategy()` retourne `"full"`.
