<p align="center">
  <img src="img/logo.jpg" alt="NietzscheDB Logo" width="320"/>
</p>

<h1 align="center">NietzscheDB</h1>

<p align="center">
  <strong>Temporal Hyperbolic Graph Database</strong>
</p>

<p align="center">
  <em>A new category of database — not a combination of existing ones.</em>
</p>

<p align="center">
  <a href="https://github.com/JoseRFJuniorLLMs/NietzscheDB/blob/main/LICENSE_AGPLv3.md"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/built%20with-Rust%20nightly-orange.svg" alt="Rust"></a>
  <img src="https://img.shields.io/badge/crates-38%20workspace-informational.svg" alt="Crates">
  <img src="https://img.shields.io/badge/gRPC-65%2B%20RPCs-blueviolet.svg" alt="RPCs">
  <img src="https://img.shields.io/badge/geometry-Poincar%C3%A9%20Ball-purple.svg" alt="Hyperbolic">
  <img src="https://img.shields.io/badge/category-Temporal%20Hyperbolic%20Graph%20DB-red.svg" alt="Category">
  <img src="https://img.shields.io/badge/GPU-cuVS%20CAGRA-76b900.svg" alt="GPU">
  <img src="https://img.shields.io/badge/TPU-PJRT%20v5e%2Fv6e%2Fv7-4285F4.svg" alt="TPU">
</p>

---

## What Category Is This?

NietzscheDB does not fit any existing database category.

```
Graph Database        Neo4j, ArangoDB
  → has nodes, edges, traversal, query language         ✔ NietzscheDB has this

Vector Database       Qdrant, Pinecone
  → has embeddings, similarity search                   ✔ NietzscheDB has this

Document Store        MongoDB, ArangoDB
  → each node is a rich JSON document                   ✔ NietzscheDB has this

Time-evolving Graph   no mainstream product
  → the graph rewrites itself autonomously              ✔ NietzscheDB has this (L-System)

Hyperbolic Database   no mature product exists
  → non-Euclidean geometry as a native primitive        ✔ NietzscheDB has this
```

The name for what NietzscheDB actually is:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│         TEMPORAL HYPERBOLIC GRAPH DATABASE              │
│                                                         │
│   · Non-Euclidean geometry as the storage primitive     │
│   · Autonomous fractal growth via L-System rules        │
│   · Multi-scale search via hyperbolic heat diffusion    │
│   · Active memory reconsolidation during idle cycles    │
│   · GPU/TPU-accelerated vector search                   │
│   · 11 built-in graph algorithms (PageRank, Louvain,...) │
│   · Hybrid BM25+ANN search with RRF fusion              │
│   · Filtered KNN with Roaring Bitmaps + metadata push    │
│   · Product Quantization (magnitude-preserving)          │
│   · MERGE upsert with edge metadata + atomic counters    │
│   · SET with arithmetic (n.count = n.count + 1)          │
│   · CREATE with TTL + DETACH DELETE                      │
│   · Edge alias property access (r.weight, r.count)       │
│   · Redis-compatible cache (CacheSet/Get/Del + TTL)      │
│   · Per-collection RwLock concurrency                    │
│   · Persistent secondary indexes with query planner      │
│   · RBAC + AES-256-CTR encryption at-rest                │
│   · Schema validation + metadata secondary indexes       │
│   · Multi-vector per node (Named Vectors)                │
│   · Media/blob storage via OpenDAL (S3, GCS, local)      │
│   · Relational tables via SQLite (hybrid graph+SQL)      │
│   · MCP server for AI assistant integration              │
│   · Prometheus/OpenTelemetry metrics export              │
│   · Kafka Connect sink for CDC streaming                 │
│   · Autonomous evolution (Zaratustra cycle)             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**In plain language:**
- *For users:* "A memory database that thinks in hierarchies, grows like a plant, and sleeps to consolidate what it learned."
- *For engineers:* "A hyperbolic graph with native L-System growth, heat kernel diffusion as a search primitive, GPU/TPU vector backends, 65+ gRPC RPCs, MCP server for AI assistants, Prometheus metrics, filtered KNN with metadata push-down, Product Quantization, MERGE upsert semantics, persistent secondary indexes, per-collection RwLock concurrency, Redis-compatible cache layer, and periodic Riemannian reconsolidation."
- *For the market:* The category does not exist yet. This is it.

---

## What is NietzscheDB?

NietzscheDB is a purpose-built database engine for **[EVA-Mind](https://github.com/JoseRFJuniorLLMs)** — an AI memory and reasoning system that requires more than flat vector search can offer.

Standard vector databases store memories in Euclidean space: every concept lives at the same "depth", every relationship is a number, and hierarchical structure collapses into cosine similarity. EVA-Mind needs to think differently.

NietzscheDB organizes knowledge in **hyperbolic space (Poincare ball model)**, where:
- Abstract concepts naturally live near the center
- Specific memories live near the boundary
- Hierarchical distance is intrinsic — not encoded, but *geometric*
- The knowledge graph grows and prunes itself like a fractal organism
- The system "sleeps" and reconsolidates its own memory topology
- An autonomous evolution cycle (Zaratustra) propagates energy, captures temporal echoes, and identifies elite nodes

It is a fork of **[YARlabs/hyperspace-db](https://github.com/YARlabs/hyperspace-db)** — the world's first native hyperbolic HNSW vector database — extended with a full graph engine, query language, L-System growth, hyperbolic diffusion, GPU/TPU acceleration, graph algorithms, cluster support, and an autonomous sleep/reconsolidation cycle.

---

## Why EVA-Mind Needs This

| Problem | Standard Vector DB | NietzscheDB |
|---|---|---|
| Hierarchy representation | Flat — same depth for all | Geometric — depth = abstraction level |
| Semantic search | Cosine similarity | Hyperbolic HNSW + heat kernel diffusion |
| Knowledge growth | Static inserts | L-System: graph grows by production rules |
| Memory pruning | Manual deletion | Hausdorff dimension: self-pruning fractal |
| Memory consolidation | No concept | Sleep cycle: Riemannian perturbation + rollback |
| Query language | k-NN only | NQL — graph, vector, diffusion, CREATE/SET/DELETE/DETACH DELETE/MERGE, EXPLAIN |
| Search | Vector OR text | Hybrid BM25+KNN with RRF fusion |
| Graph analytics | External tool needed | 11 built-in algorithms (PageRank, Louvain, A*, ...) |
| Hardware acceleration | CPU only or proprietary | GPU (cuVS CAGRA) + TPU (PJRT) at runtime |
| Security | API key at best | RBAC (Admin/Writer/Reader) + AES-256-CTR encryption at-rest |
| Data integrity | No schema enforcement | Per-NodeType schema validation (required fields + types) |
| Cache layer | External Redis | Built-in CacheSet/Get/Del with TTL + lazy expiry |
| Concurrency | Global lock or external | Per-collection RwLock — concurrent reads, write-isolated |
| Consistency | Single-store | ACID saga pattern across graph + vector store |
| Vector compression | Scalar/binary quant | Product Quantization (magnitude-preserving — depth safe) |
| Multi-vector | Single embedding only | Named Vectors: multiple embeddings per node with different metrics |
| AI integration | REST API only | MCP server (19 tools) + REST + gRPC |
| Media storage | External service | Built-in OpenDAL store (S3, GCS, local filesystem) |
| Relational data | Separate RDBMS | Built-in SQLite table store with NodeRef foreign keys |
| Observability | External setup | Built-in Prometheus metrics (12 counters/gauges/histograms) |
| CDC streaming | External CDC | Built-in Kafka Connect sink (6 mutation types) |

---

## Architecture

NietzscheDB is built as a **Rust nightly workspace** with 38 crates in two layers:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         NietzscheDB Layer (29 crates)                        │
│                                                                              │
│  Engine:     nietzsche-graph    nietzsche-query     nietzsche-hyp-ops        │
│  Growth:     nietzsche-lsystem  nietzsche-pregel    nietzsche-sleep          │
│  Evolution:  nietzsche-zaratustra                                            │
│  Analytics:  nietzsche-algo     nietzsche-sensory                            │
│  Visionary:  nietzsche-dream    nietzsche-narrative  nietzsche-agency        │
│  Wiederkehr: nietzsche-wiederkehr                                            │
│  Infra:      nietzsche-api      nietzsche-server    nietzsche-cluster        │
│  SDKs:       nietzsche-sdk      nietzsche-mcp                                │
│  Accel:      nietzsche-hnsw-gpu nietzsche-tpu       nietzsche-cugraph        │
│  Search:     nietzsche-filtered-knn  nietzsche-named-vectors  nietzsche-pq   │
│  Index:      nietzsche-secondary-idx                                         │
│  Observe:    nietzsche-metrics                                               │
│  Storage:    nietzsche-table    nietzsche-media      nietzsche-kafka          │
├──────────────────────────────────────────────────────────────────────────────┤
│                     HyperspaceDB Layer (9 crates — fork base)                │
│                                                                              │
│  hyperspace-core   hyperspace-index   hyperspace-store                       │
│  hyperspace-server hyperspace-proto   hyperspace-cli                         │
│  hyperspace-embed  hyperspace-wasm    hyperspace-sdk                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### HyperspaceDB Foundation (fork base)

The storage and indexing foundation, inheriting all of HyperspaceDB v2.0:

- **Poincare Ball HNSW** — native hyperbolic nearest-neighbor index. Not re-ranking, not post-processing: the graph itself navigates in hyperbolic geometry.
- **mmap Vector Store** — memory-mapped, append-only segments (`chunk_N.hyp`) with 1-bit to 8-bit quantization (up to 64x compression).
- **Write-Ahead Log v3** — binary `[Magic][Len][CRC32][Op][Data]` format with configurable durability and automatic crash recovery.
- **gRPC API** — async Command-Query Separation with background indexing. Client gets `OK` as soon as the WAL is written.
- **Leader-Follower Replication** — async anti-entropy via logical clocks and Merkle tree bucket sync (256 buckets).
- **SIMD Acceleration** — Portable SIMD (`std::simd`) for 4-8x distance computation speedup on AVX2/Neon.
- **Multi-tenancy** — namespace isolation per `user_id`, per-user quota and billing accounting.
- **9,087 QPS** insert performance verified under stress test (90x above original target).
- **WASM** — browser-compatible build via `hyperspace-wasm` with IndexedDB storage.
- **Universal Embedder** — `hyperspace-embed` with local ONNX (`ort`) + remote API support.

### NietzscheDB Extensions

Twenty-nine new crates built on top of the foundation:

#### `nietzsche-graph` — Hyperbolic Graph Engine
- `Node` = `NodeMeta` (~100 bytes: id, depth, energy, node_type, hausdorff_local, content) + `PoincareVector` (embedding, stored separately for 10-25x traversal speedup)
- `PoincareVector` with `Vec<f32>` coords (distance kernel promotes to f64 internally for numerical stability near the Poincare boundary)
- `SparseVector` for SPLADE/sparse embeddings: sorted indices + values with O(nnz) dot product, cosine similarity, and L2 norm
- `Edge` typed as `Association`, `LSystemGenerated`, `Hierarchical`, or `Pruned`
- `AdjacencyIndex` using `DashMap` for lock-free concurrent access
- `GraphStorage` over RocksDB with 10 column families: `nodes`, `embeddings`, `edges`, `adj_out`, `adj_in`, `meta`, `sensory`, `energy_idx`, `meta_idx`, `lists`
- Own WAL for graph operations, separate from the vector WAL
- `NietzscheDB` dual-write: every insert goes to both RocksDB (graph) and HyperspaceDB (embedding)
- **Traversal engine** (`traversal.rs`): energy-gated BFS (reads only NodeMeta — ~100 bytes per hop), Poincare-distance Dijkstra, shortest-path reconstruction, energy-biased `DiffusionWalk` with seeded RNG
- **EmbeddedVectorStore** abstraction: CPU (HnswIndex) / GPU (GpuVectorStore) / TPU (TpuVectorStore) / Mock — selected at runtime via `NIETZSCHE_VECTOR_BACKEND`. Default is Embedded (real HNSW); Mock requires explicit opt-in
- **Multi-Metric HNSW**: Cosine, Euclidean, Poincare, and DotProduct distance metrics per collection. Factory routing ensures correct metric type at the HNSW graph topology level
- **KNN metadata filter push-down**: `MetadataFilter` (Eq, In, Range, And) pushed through VectorStore → DynHnsw → HnswIndex with RoaringBitmap pre-filtering for efficient filtered vector search
- **MERGE upsert semantics**: `merge_node` (find-or-create by content), `merge_edge` (find-or-create by from/to/type) with ON CREATE SET / ON MATCH SET
- **Atomic edge metadata increment**: `increment_edge_metadata(edge_id, field, delta)` for counter patterns (e.g. `r.count = r.count + 1`)
- **Persistent secondary indexes**: `create_index(field)` / `drop_index(field)` / `list_indexes()` with automatic backfill and startup recovery from CF_META registry. NQL executor auto-detects indexed fields for O(log N) scans instead of full table scans
- **Encryption at-rest** (`encryption.rs`): AES-256-CTR with HKDF-SHA256 per-CF key derivation from master key (`NIETZSCHE_ENCRYPTION_KEY`)
- **Schema validation** (`schema.rs`): per-NodeType constraints (required fields, field types), persisted in CF_META, enforced on `insert_node`
- **Metadata secondary indexes** (`CF_META_IDX`): arbitrary field indexing with FNV-1a + sortable value encoding for range scans
- **ListStore** (`CF_LISTS`): per-node ordered lists with RPUSH/LRANGE/LLEN semantics, atomic sequence counters
- **TTL / expires_at** enforcement: background reaper scans expired nodes and phantomizes them (topology-preserving). CREATE with `ttl` property auto-computes `expires_at`
- **Redis-compatible cache layer**: `CacheSet`/`CacheGet`/`CacheDel` RPCs using CF_META with "cache:" prefix, TTL as 8-byte expiry timestamp, lazy-delete on expired reads
- **Per-collection `tokio::sync::RwLock` concurrency**: `CollectionManager` with `DashMap` + `Arc<RwLock<NietzscheDB>>` per collection. Reads proceed concurrently; writes block only the affected collection
- **Full-text search + hybrid** (`fulltext.rs`): inverted index with BM25 scoring, plus RRF fusion with KNN vector search

#### `nietzsche-hyp-ops` — Poincare Ball Math
Core hyperbolic geometry primitives: Mobius addition, exponential/logarithmic maps, geodesic distance, parallel transport. Used by all other crates that need Poincare ball operations. Includes criterion benchmarks.

#### `nietzsche-query` — NQL Query Language
Nietzsche Query Language — a declarative query language with first-class hyperbolic primitives. Parser built with `pest` (PEG grammar). Supports arithmetic SET expressions (`n.count = n.count + 1`), edge alias property access (`-[r:TYPE]->` with `r.weight` in WHERE/ORDER BY), CREATE with TTL, DETACH DELETE, and eval_field fallback to `node.content`/`node.metadata` for dynamic properties. 113+ unit + integration tests.

**[Full NQL Reference: docs/NQL.md](docs/NQL.md)**

Query types:

| Type | Description |
|---|---|
| `MATCH` | Pattern matching on nodes/paths with hyperbolic conditions |
| `CREATE` | Insert new nodes with labels, properties, and optional TTL |
| `MATCH … SET` | Update matched nodes' properties (supports arithmetic: `n.count = n.count + 1`) |
| `MATCH … DELETE` | Delete matched nodes |
| `MATCH … DETACH DELETE` | Delete matched nodes and all incident edges |
| `MERGE` | Upsert nodes/edges (ON CREATE SET / ON MATCH SET) |
| `DIFFUSE` | Multi-scale heat-kernel activation propagation |
| `RECONSTRUCT` | Decode sensory data from latent vector |
| `EXPLAIN` | Return execution plan with cost estimates |
| `DREAM FROM` | Speculative graph exploration via hyperbolic diffusion with noise |
| `APPLY/REJECT DREAM` | Accept or discard dream simulation results |
| `TRANSLATE` | Cross-modal projection (Synesthesia) via Poincare ball log/exp map |
| `MATCH ... AS OF CYCLE` | Time-travel query on named snapshots (Eternal Return) |
| `COUNTERFACTUAL` | What-if query with ephemeral property overlays |
| `CREATE/DROP/SHOW DAEMON` | Autonomous daemon agents (Wiederkehr) |
| `SHOW ARCHETYPES` | List shared cross-collection archetypes |
| `SHARE ARCHETYPE` | Publish elite node for cross-collection discovery |
| `NARRATE` | Generate human-readable narrative from graph evolution |

```sql
-- Hyperbolic nearest-neighbor search with depth filter
MATCH (m:Memory)
WHERE HYPERBOLIC_DIST(m.embedding, $q) < 0.5
  AND m.depth > 0.6
  AND NOT m.node_type = "Pruned"
RETURN m
ORDER BY HYPERBOLIC_DIST(m.embedding, $q) ASC
LIMIT 10

-- Graph traversal: hierarchical expansion
MATCH (c:Concept)-[:Hierarchical]->(child)
WHERE c.energy > 0.7
RETURN child ORDER BY child.depth DESC LIMIT 20

-- IN / BETWEEN / string operators
MATCH (n)
WHERE n.node_type IN ("Semantic", "Episodic")
  AND n.energy BETWEEN 0.3 AND 0.9
  AND n.node_type STARTS_WITH "S"
RETURN n LIMIT 50

-- Aggregation with GROUP BY
MATCH (n)
RETURN n.node_type, COUNT(*) AS total, AVG(n.energy) AS avg_e
GROUP BY n.node_type
ORDER BY total DESC

-- Mathematician-named geometric functions
MATCH (n)
WHERE RIEMANN_CURVATURE(n) > 0.3
  AND HAUSDORFF_DIM(n) BETWEEN 1.2 AND 1.8
  AND DIRICHLET_ENERGY(n) < 0.1
RETURN n ORDER BY RIEMANN_CURVATURE(n) DESC LIMIT 10

-- Multi-hop path traversal (BFS 2..4 hops)
MATCH (a)-[:Association*2..4]->(b)
WHERE a.energy > 0.5
RETURN a, b LIMIT 50

-- Create a new node
CREATE (n:Episodic {title: "first meeting", source: "manual"})
RETURN n

-- Create with TTL (auto-expires after 3600 seconds)
CREATE (n:EvaSession {id: "sess_1", turn_count: 0, ttl: 3600})
RETURN n

-- Arithmetic SET (per-node evaluation)
MATCH (n:EvaSession {id: "sess_1"})
SET n.turn_count = n.turn_count + 1, n.status = "active"
RETURN n

-- Edge alias: access edge properties in WHERE/ORDER BY
MATCH (a:Person)-[r:MENTIONED]->(b:Topic)
WHERE r.weight > 0.5
RETURN a, b ORDER BY r.weight DESC LIMIT 10

-- Update matched nodes
MATCH (n:Semantic) WHERE n.energy < 0.1 SET n.energy = 0.5 RETURN n

-- DETACH DELETE (node + all incident edges)
MATCH (n:EvaSession) WHERE n.status = "expired" DETACH DELETE n

-- Delete expired nodes
MATCH (n) WHERE n.energy = 0.0 DELETE n

-- Time-based queries with NOW() and INTERVAL()
MATCH (n) WHERE n.created_at > NOW() - INTERVAL("7d") RETURN n LIMIT 50

-- EXPLAIN with cost estimates
EXPLAIN MATCH (n:Memory) WHERE n.energy > 0.3 RETURN n
-- → NodeScan(label=Memory) -> Filter(conditions=1) | rows=~250, scan=EnergyIndexScan, index=CF_ENERGY_IDX, cost=~500µs

-- Multi-scale heat-kernel diffusion
DIFFUSE FROM $seed
  WITH t = [0.1, 1.0, 10.0]
  MAX_HOPS 6
RETURN path

-- Dream Queries — speculative exploration
DREAM FROM $seed DEPTH 5 NOISE 0.05
SHOW DREAMS
APPLY DREAM "dream_xxx"

-- Daemon Agents — autonomous graph patrols
CREATE DAEMON guardian ON (n:Memory)
  WHEN n.energy > 0.8
  THEN DIFFUSE FROM n WITH t=[0.1, 1.0] MAX_HOPS 5
  EVERY INTERVAL("1h")
  ENERGY 0.8
SHOW DAEMONS

-- Time-travel via named snapshots
MATCH (n:Memory) AS OF CYCLE 3
WHERE n.energy > 0.5
RETURN n

-- Narrative Engine
NARRATE IN "memories" WINDOW 24 FORMAT json
```

**Built-in geometric functions:**

| Function | Named after | Computes |
|---|---|---|
| `HYPERBOLIC_DIST(n.e, $q)` | — | Poincare ball geodesic distance |
| `POINCARE_DIST(n, $q)` | Henri Poincare | Same — explicit model name |
| `KLEIN_DIST(n, $q)` | Felix Klein | Beltrami-Klein distance |
| `RIEMANN_CURVATURE(n)` | Bernhard Riemann | Ollivier-Ricci curvature |
| `HAUSDORFF_DIM(n)` | Felix Hausdorff | Local fractal dimension |
| `GAUSS_KERNEL(n, t)` | Carl Friedrich Gauss | Heat kernel `exp(-d^2/4t)` |
| `CHEBYSHEV_COEFF(n, k)` | Pafnuty Chebyshev | Chebyshev polynomial T_k |
| `DIRICHLET_ENERGY(n)` | P.G.L. Dirichlet | Local Dirichlet energy |
| `EULER_CHAR(n)` | Leonhard Euler | V - E characteristic |
| `LAPLACIAN_SCORE(n)` | P.-S. Laplace | Graph Laplacian diagonal |
| `LOBACHEVSKY_ANGLE(n, $p)` | N. Lobachevsky | Angle of parallelism |
| `MINKOWSKI_NORM(n)` | H. Minkowski | Conformal factor |
| `RAMANUJAN_EXPANSION(n)` | S. Ramanujan | Spectral expansion ratio |
| `FOURIER_COEFF(n, k)` | J. Fourier | Graph Fourier coefficient |
| `NOW()` | — | Current Unix timestamp (seconds, f64) |
| `EPOCH_MS()` | — | Current Unix epoch (milliseconds, f64) |
| `INTERVAL("1h")` | — | Duration to seconds (s/m/h/d/w units) |

#### `nietzsche-lsystem` — Fractal Growth Engine
The knowledge graph is not static — it grows by **L-System production rules**:
- `ProductionRule` fires when `EnergyAbove(t)`, `DepthBelow(t)`, `HausdorffAbove(t)`, or custom conditions (`And`, `Or`, `Not`, `Always`)
- `SpawnChild` places the child node deeper in the Poincare ball (more specific, closer to boundary) via **Mobius addition** `u + v`
- `SpawnSibling` creates lateral associations at a given hyperbolic angle
- `Prune` archives low-complexity regions (not deleted — tagged as `Pruned`)
- **Hausdorff dimension** computed via box-counting on hyperbolic coordinates at scales `[4, 8, 16, 32, 64]`
- Nodes with D < 0.5 or D > 1.9 are pruned automatically; target fractal regime: **1.2 < D < 1.8**
- `LSystemEngine::tick` protocol: Scan -> Hausdorff update -> Rule matching -> Apply mutations -> Report
- 29 unit tests across all four modules

#### `nietzsche-pregel` — Hyperbolic Heat Kernel Diffusion
Multi-scale activation propagation across the hyperbolic graph:
- **Chebyshev polynomial approximation** of the heat kernel `e^(-tL)` — O(K x |E|) complexity
- **Hyperbolic graph Laplacian** — edge weights derived from Poincare distances
- **Modified Bessel functions** `I_k(t)` for computing Chebyshev coefficients analytically
- Multiple diffusion scales: `t=0.1` activates direct neighbors (focused recall), `t=10.0` activates structurally connected but semantically distant nodes (free association)
- `HyperbolicLaplacian` + `apply_heat_kernel` + `chebyshev_coefficients` as stable public API

#### `nietzsche-wiederkehr` — DAEMON Agents
Autonomous agents that live inside the database, patrolling the graph and executing actions when conditions are met:
- **DaemonDef** with configurable WHEN conditions, THEN actions, EVERY interval, and ENERGY budget
- **DaemonEngine** tick loop: evaluate conditions, collect intents, decay energy, reap dead daemons
- **Will to Power** priority scheduler: BinaryHeap-based scheduling with energy × urgency weighting
- NQL: `CREATE DAEMON`, `DROP DAEMON`, `SHOW DAEMONS`
- 18 unit tests (store, evaluator, engine, priority)

#### `nietzsche-dream` — Dream Queries
Speculative graph exploration via hyperbolic diffusion with stochastic noise:
- **DreamEngine**: BFS exploration from seed node with noise-perturbed energy detection
- Energy spike and curvature anomaly event detection
- Pending/Applied/Rejected dream lifecycle with persistent sessions
- NQL: `DREAM FROM`, `APPLY DREAM`, `REJECT DREAM`, `SHOW DREAMS`
- 8 unit tests (store, engine)

#### `nietzsche-narrative` — Narrative Engine
Story arc detection and generation from graph evolution:
- **NarrativeEngine**: scans nodes, computes energy statistics, detects elite emergence and decay events
- Configurable time window, elite/decay thresholds
- JSON and text output formats with auto-generated summaries
- NQL: `NARRATE IN "collection" WINDOW hours FORMAT json|text`
- 4 unit tests

#### Synesthesia (in `nietzsche-sensory`)
Cross-modal projection via hyperbolic parallel transport:
- `translate_modality()`: log_map → modal rotation → exp_map on the Poincare ball
- Preserves hierarchical depth (radius) while changing modality direction
- Quality loss estimation per modality pair
- NQL: `TRANSLATE $node FROM text TO audio`

#### Eternal Return (in `nietzsche-query`)
Temporal queries and counterfactual reasoning:
- `AS OF CYCLE N`: time-travel queries on named snapshots
- `COUNTERFACTUAL SET ... MATCH ...`: what-if queries with ephemeral overlays

#### Collective Unconscious (in `nietzsche-cluster`)
Cross-collection archetype sharing via gossip protocol:
- `ArchetypeRegistry`: DashMap-based registry with merge_peer_archetypes for gossip
- NQL: `SHOW ARCHETYPES`, `SHARE ARCHETYPE $node TO "collection"`
- 4 unit tests

#### `nietzsche-agency` — Autonomous Agency Engine
Graph-level autonomous intelligence with counterfactual reasoning:
- **AgencyEngine** tick loop: runs 3 built-in daemons (Entropy, Gap, Coherence) + MetaObserver
- **EntropyDaemon**: detects Hausdorff variance spikes across angular regions
- **GapDaemon**: identifies knowledge gaps in depth×angle sectors
- **CoherenceDaemon**: measures multi-scale diffusion overlap (Chebyshev heat kernel)
- **MetaObserver**: produces HealthReports with energy percentiles, fractal status, wake-up triggers
- **CounterfactualEngine**: what-if simulations via ShadowGraph (remove/add nodes without mutating real graph)
- **AgencyEventBus**: tokio broadcast channel for cross-system event propagation
- 20+ unit tests (event_bus, engine, observer, daemons, shadow, simulator)

#### `nietzsche-sleep` — Reconsolidation Sleep Cycle
EVA-Mind sleeps. During sleep:
1. Sample high-curvature subgraph via random walk
2. Snapshot current embeddings (rollback point)
3. Perturb embeddings in the tangent space (the "dream")
4. Optimize via **RiemannianAdam** on the Poincare manifold
5. Measure Hausdorff dimension before and after
6. **Commit** if `delta(Hausdorff) < threshold` — identity preserved, reconsolidation accepted
7. **Rollback** if identity was destroyed — dream discarded

This prevents catastrophic forgetting while allowing genuine memory reorganization.

**Time-travel / Versioning:** Named snapshots (`SnapshotRegistry`) allow creating labeled checkpoints of the entire embedding state, listing all snapshots with timestamps, and restoring any previous state — enabling temporal queries and safe experimentation.

#### `nietzsche-zaratustra` — Autonomous Evolution Engine
Three-phase autonomous cycle inspired by Nietzsche's philosophy:
1. **Will to Power** — energy propagation: each node absorbs `alpha x mean(neighbour_energy)`, amplifying high-energy clusters
2. **Eternal Recurrence** — temporal echo snapshots: captures periodic state for pattern detection
3. **Ubermensch** — elite tier identification: nodes in the top energy fraction are promoted to elite status

Configurable via `ZARATUSTRA_INTERVAL_SECS` (default 600s). Invocable via gRPC `InvokeZaratustra` or automatic background scheduler.

#### `nietzsche-algo` — Graph Algorithm Library
Eleven built-in graph algorithms, all available via gRPC and HTTP:

| Algorithm | Type | RPC |
|---|---|---|
| PageRank | Centrality | `RunPageRank` |
| Louvain | Community | `RunLouvain` |
| Label Propagation | Community | `RunLabelProp` |
| Betweenness Centrality | Centrality | `RunBetweenness` |
| Closeness Centrality | Centrality | `RunCloseness` |
| Degree Centrality | Centrality | `RunDegreeCentrality` |
| WCC (Weakly Connected) | Component | `RunWCC` |
| SCC (Strongly Connected) | Component | `RunSCC` |
| A* Pathfinding | Pathfinding | `RunAStar` |
| Triangle Count | Structure | `RunTriangleCount` |
| Jaccard Similarity | Similarity | `RunJaccardSimilarity` |

#### `nietzsche-sensory` — Sensory Compression Layer
Multi-modal latent vector storage with progressive degradation:
- Stores latent representations for: **text, audio, image, fused** modalities
- Progressive degradation: `f32 -> f16 -> int8 -> PQ -> gone`
- Original shape metadata preserved for reconstruction
- Encoder version tracking for backward compatibility
- Persisted in RocksDB via graph storage

#### `nietzsche-cluster` — Distributed Foundation
Gossip-based cluster discovery and shard routing:
- `ClusterNode` — identity, role (primary/replica/coordinator), health
- `ClusterRegistry` — gossip-updated peer view
- `ClusterRouter` — shard selection
- Eventual consistency via gossip (no Raft in current phase)
- Configurable via `NIETZSCHE_CLUSTER_ENABLED`, `NIETZSCHE_CLUSTER_ROLE`, `NIETZSCHE_CLUSTER_SEEDS`

#### `nietzsche-hnsw-gpu` — GPU Vector Backend
NVIDIA cuVS CAGRA acceleration for vector search. See [GPU Acceleration](#gpu-acceleration) section.

#### `nietzsche-tpu` — TPU Vector Backend
Google TPU acceleration via PJRT C API. See [TPU Acceleration](#tpu-acceleration) section.

#### `nietzsche-cugraph` — GPU Graph Traversal
GPU-accelerated graph algorithms via NVIDIA cuGraph:
- cuGraph BFS/Dijkstra/PageRank on GPU
- Poincare GPU k-NN with custom CUDA kernel
- Dynamic FFI loader for `libcugraph.so` at runtime
- cudarc + NVRTC for Poincare kernel compilation
- Feature flag: `--features cuda`

#### `nietzsche-mcp` — Model Context Protocol Server
JSON-RPC 2.0 server for AI assistant integration (Claude, GPT, etc.):
- **19 tools**: graph CRUD, NQL query, KNN search, traversal, graph algorithms, diffusion, stats
- Stdin/stdout transport (standard MCP protocol)
- Parameter validation with typed `ParamValue` (String, Float, Int, Bool, Vector)
- 19 unit tests

#### `nietzsche-metrics` — Prometheus/OpenTelemetry Metrics
Observability export layer:
- **NODES_INSERTED**, **EDGES_INSERTED**, **QUERIES_EXECUTED** (CounterVec by collection)
- **QUERY_DURATION_SECONDS**, **KNN_DURATION_SECONDS**, **DIFFUSION_DURATION_SECONDS** (Histogram)
- **NODE_COUNT**, **EDGE_COUNT**, **DAEMON_COUNT**, **DAEMON_ENERGY_TOTAL** (Gauge)
- Singleton `MetricsRegistry` with Prometheus text format export (`/metrics`)
- 6 unit tests

#### `nietzsche-filtered-knn` — Filtered KNN with Roaring Bitmaps
Pre-filtered nearest-neighbor search using Roaring Bitmaps:
- **NodeFilter** enum: EnergyRange, NodeType, ContentField, ContentFieldExists, And, Or
- Energy range filter leverages `CF_ENERGY_IDX` for efficient range scans
- JSON dot-path navigation for content field filtering
- Poincare distance computation for hyperbolic KNN
- 15 integration tests

#### `nietzsche-named-vectors` — Multi-Vector per Node
Multiple named vector embeddings per node:
- `NamedVector { node_id, name, coordinates, metric }` with VectorMetric (Poincare, Cosine, Euclidean)
- Persisted in CF_META with key `nvec:{node_id}:{name}` (bincode serialization)
- `NamedVectorStore` with put/get/list/delete/delete_all operations
- 8 unit tests

#### `nietzsche-pq` — Product Quantization
Magnitude-preserving vector compression (NOT binary quantization — preserves hyperbolic depth):
- **Codebook** training via k-means clustering per sub-vector partition
- **PQEncoder** with encode/decode: M sub-vectors × K=256 centroids
- **Asymmetric Distance Computation (ADC)** via precomputed DistanceTable
- KEY: `test_magnitude_preservation` proves PQ preserves `‖x‖` = depth in Poincare ball
- Configurable: `PQConfig { m: 8, k: 256, max_iterations: 25 }`
- 12 unit tests

#### `nietzsche-secondary-idx` — Secondary Indexes
Arbitrary JSON field indexing for fast lookups:
- `IndexDef { name, field_path, index_type: String|Float|Int }`
- Persisted in CF_META: definitions at `idx_def:{name}`, entries at `sidx:{name}:{sortable_value}:{node_id}`
- Float encoding: IEEE 754 sign-magnitude to lexicographic order (16 hex chars)
- `SecondaryIndexBuilder` with create_index, drop_index, insert_entry, lookup, range_lookup
- 13 unit tests

#### `nietzsche-kafka` — Kafka Connect Sink
Change data capture sink for streaming mutations:
- `GraphMutation` enum: InsertNode, DeleteNode, InsertEdge, DeleteEdge, SetEnergy, SetContent
- `KafkaSink` with process_message/process_batch (BatchResult with succeeded/failed/errors)
- SetContent merges JSON fields into existing node content
- 9 unit tests

#### `nietzsche-table` — Relational Table Store (SQLite)
Bridging graph and relational paradigms:
- `TableSchema` with `ColumnDef { name, col_type, nullable, default }`
- Column types: Text, Integer, Float, Bool, Uuid, Json, **NodeRef** (FK to graph nodes)
- `TableStore` wrapping rusqlite::Connection with create/drop/insert/query/delete/list/schema
- Schema metadata persisted in `_nietzsche_table_meta` internal table
- File-backed or in-memory operation modes
- 15 unit tests

#### `nietzsche-media` — Media/Blob Store (OpenDAL)
Backend-agnostic media storage for files associated with graph nodes:
- Powered by **Apache OpenDAL** — supports local filesystem, S3, GCS, Azure, and more
- `MediaMeta { id, node_id, filename, media_type, content_type, size_bytes, created_at }`
- Media types: Image, Audio, Video, Document, Binary
- `MediaStore` with put/get/get_meta/delete/list_for_node/exists
- Flat key structure: `{node_id}/{media_id}` and `{node_id}/{media_id}.meta`
- 8 unit tests

#### `nietzsche-api` — Unified gRPC API
Single endpoint for all NietzscheDB capabilities — **65+ RPCs** over a single `NietzscheDB` service. Every data-plane RPC accepts a `collection` field; empty -> `"default"`.

```protobuf
service NietzscheDB {
  // ── Collection management ─────────────────────────────────────
  rpc CreateCollection(CreateCollectionRequest)   returns (CreateCollectionResponse);
  rpc DropCollection(DropCollectionRequest)       returns (StatusResponse);
  rpc ListCollections(Empty)                      returns (ListCollectionsResponse);

  // ── Graph CRUD ────────────────────────────────────────────────
  rpc InsertNode(InsertNodeRequest)     returns (NodeResponse);
  rpc GetNode(NodeIdRequest)            returns (NodeResponse);
  rpc DeleteNode(NodeIdRequest)         returns (StatusResponse);
  rpc UpdateEnergy(UpdateEnergyRequest) returns (StatusResponse);
  rpc InsertEdge(InsertEdgeRequest)     returns (EdgeResponse);
  rpc DeleteEdge(EdgeIdRequest)         returns (StatusResponse);
  rpc MergeNode(MergeNodeRequest)       returns (MergeNodeResponse);
  rpc MergeEdge(MergeEdgeRequest)       returns (MergeEdgeResponse);
  rpc IncrementEdgeMeta(IncrementEdgeMetaRequest) returns (IncrementEdgeMetaResponse);

  // ── Batch Operations ──────────────────────────────────────────
  rpc BatchInsertNodes(BatchInsertNodesRequest)   returns (BatchInsertNodesResponse);
  rpc BatchInsertEdges(BatchInsertEdgesRequest)   returns (BatchInsertEdgesResponse);

  // ── Query & Search ────────────────────────────────────────────
  rpc Query(QueryRequest)               returns (QueryResponse);
  rpc KnnSearch(KnnRequest)             returns (KnnResponse);   // supports metadata filters
  rpc FullTextSearch(FullTextSearchRequest) returns (FullTextSearchResponse);
  rpc HybridSearch(HybridSearchRequest) returns (KnnResponse);

  // ── Traversal ─────────────────────────────────────────────────
  rpc Bfs(TraversalRequest)             returns (TraversalResponse);
  rpc Dijkstra(TraversalRequest)        returns (TraversalResponse);
  rpc Diffuse(DiffusionRequest)         returns (DiffusionResponse);

  // ── Graph Algorithms ──────────────────────────────────────────
  rpc RunPageRank(PageRankRequest)             returns (AlgorithmScoreResponse);
  rpc RunLouvain(LouvainRequest)               returns (AlgorithmCommunityResponse);
  rpc RunLabelProp(LabelPropRequest)           returns (AlgorithmCommunityResponse);
  rpc RunBetweenness(BetweennessRequest)       returns (AlgorithmScoreResponse);
  rpc RunCloseness(ClosenessRequest)           returns (AlgorithmScoreResponse);
  rpc RunDegreeCentrality(DegreeCentralityRequest) returns (AlgorithmScoreResponse);
  rpc RunWCC(WccRequest)                       returns (AlgorithmCommunityResponse);
  rpc RunSCC(SccRequest)                       returns (AlgorithmCommunityResponse);
  rpc RunAStar(AStarRequest)                   returns (AStarResponse);
  rpc RunTriangleCount(TriangleCountRequest)   returns (TriangleCountResponse);
  rpc RunJaccardSimilarity(JaccardRequest)     returns (SimilarityResponse);

  // ── Lifecycle ─────────────────────────────────────────────────
  rpc TriggerSleep(SleepRequest)        returns (SleepResponse);
  rpc InvokeZaratustra(ZaratustraRequest) returns (ZaratustraResponse);

  // ── Sensory Compression ───────────────────────────────────────
  rpc InsertSensory(InsertSensoryRequest) returns (StatusResponse);
  rpc GetSensory(NodeIdRequest)           returns (SensoryResponse);
  rpc Reconstruct(ReconstructRequest)     returns (ReconstructResponse);
  rpc DegradeSensory(NodeIdRequest)       returns (StatusResponse);

  // ── Backup / Restore ──────────────────────────────────────────
  rpc CreateBackup(CreateBackupRequest)   returns (BackupResponse);
  rpc ListBackups(Empty)                  returns (ListBackupsResponse);
  rpc RestoreBackup(RestoreBackupRequest) returns (StatusResponse);

  // ── ListStore ────────────────────────────────────────────────
  rpc ListRPush(ListPushRequest)          returns (ListPushResponse);
  rpc ListLRange(ListRangeRequest)        returns (ListRangeResponse);
  rpc ListLen(ListLenRequest)             returns (ListLenResponse);

  // ── Cache (Redis-compatible) ────────────────────────────────
  rpc CacheSet(CacheSetRequest)           returns (StatusResponse);
  rpc CacheGet(CacheGetRequest)           returns (CacheGetResponse);
  rpc CacheDel(CacheDelRequest)           returns (StatusResponse);
  rpc ReapExpired(ReapExpiredRequest)      returns (ReapExpiredResponse);

  // ── Change Data Capture ───────────────────────────────────────
  rpc SubscribeCDC(CdcRequest)            returns (stream CdcEvent);

  // ── Cluster ─────────────────────────────────────────────────
  rpc ExchangeGossip(GossipRequest)       returns (GossipResponse);

  // ── Schema Validation ───────────────────────────────────────
  rpc SetSchema(SetSchemaRequest)         returns (StatusResponse);
  rpc GetSchema(GetSchemaRequest)         returns (GetSchemaResponse);
  rpc ListSchemas(Empty)                  returns (ListSchemasResponse);

  // ── Secondary Indexes ──────────────────────────────────────
  rpc CreateIndex(CreateIndexRequest)     returns (StatusResponse);
  rpc DropIndex(DropIndexRequest)         returns (StatusResponse);
  rpc ListIndexes(ListIndexesRequest)     returns (ListIndexesResponse);

  // ── Admin ─────────────────────────────────────────────────────
  rpc GetStats(Empty)                     returns (StatsResponse);
  rpc HealthCheck(Empty)                  returns (StatusResponse);
}
```

#### `nietzsche-sdk` — Rust Client SDK
Async gRPC client with seed examples (`seed_100.rs`, `seed_1gb.rs`).

#### `nietzsche-server` — Production Binary
Standalone server binary with env-var-based configuration, background sleep and Zaratustra schedulers, TTL reaper, scheduled backup with auto-pruning, RBAC (Admin/Writer/Reader), cluster gossip, embedded HTTP dashboard, and graceful shutdown.

```bash
NIETZSCHE_DATA_DIR=/data/nietzsche \
NIETZSCHE_PORT=50051 \
NIETZSCHE_DASHBOARD_PORT=8080 \
NIETZSCHE_LOG_LEVEL=info \
NIETZSCHE_SLEEP_INTERVAL_SECS=3600 \
ZARATUSTRA_INTERVAL_SECS=600 \
nietzsche-server
```

---

## HTTP Dashboard & REST API

NietzscheDB ships with an embedded **React + Cosmograph 2.1** dashboard, compiled into the binary as a single HTML file. No external web server needed.

### Dashboard Tech Stack

| Component | Version |
|---|---|
| React | 19.2 |
| TypeScript | 5.9 |
| Cosmograph | 2.1 (GPU graph visualization) |
| Vite | 7.2 + vite-plugin-singlefile |
| Tailwind CSS | 4.1 |
| TanStack React Query | 5.90 |
| Radix UI | Component primitives |
| Recharts | 3.7 |

### Dashboard Pages

| Page | Features |
|---|---|
| Overview | Node/edge counts, uptime, version, system config |
| Collections | List, create, inspect collections with dimension/metric |
| Nodes | Browse, insert, delete nodes |
| Graph Explorer | Full Cosmograph 2.1 visualization with timeline, histograms (energy/depth/hausdorff), categorical bars (node_type/edge_type), search, color legend, selection tools |
| Data Explorer | NQL query editor, CRUD forms |
| Settings | Server configuration |

### REST API Endpoints

**Core:**
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | `{status: "ok"}` |
| GET | `/api/stats` | Node/edge counts, version, uptime |
| GET | `/api/collections` | List all collections |
| GET | `/metrics` | Prometheus metrics |

**Data (CRUD):**
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/graph?collection=NAME&limit=N` | Nodes + edges for visualization |
| GET | `/api/node/:id` | Get single node |
| POST | `/api/node` | Insert node |
| DELETE | `/api/node/:id` | Delete node |
| POST | `/api/edge` | Insert edge |
| DELETE | `/api/edge/:id` | Delete edge |
| POST | `/api/batch/nodes` | Batch insert nodes |
| POST | `/api/batch/edges` | Batch insert edges |

**Query & Traversal:**
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/query` | Execute NQL query |
| POST | `/api/sleep` | Trigger sleep cycle |
| GET | `/api/search` | Full-text search |

**Graph Algorithms:**
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/algo/pagerank` | PageRank centrality |
| GET | `/api/algo/louvain` | Louvain community detection |
| GET | `/api/algo/labelprop` | Label propagation |
| GET | `/api/algo/betweenness` | Betweenness centrality |
| GET | `/api/algo/closeness` | Closeness centrality |
| GET | `/api/algo/degree` | Degree centrality |
| GET | `/api/algo/wcc` | Weakly connected components |
| GET | `/api/algo/scc` | Strongly connected components |
| GET | `/api/algo/triangles` | Triangle count |
| GET | `/api/algo/jaccard` | Jaccard similarity |

**Data Management:**
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/backup` | Create backup |
| GET | `/api/backup` | List backups |
| GET | `/api/export/nodes` | Export all nodes |
| GET | `/api/export/edges` | Export all edges |

---

## SDKs

### Python
```python
from nietzsche_db import NietzscheClient

db = NietzscheClient("localhost:50051")

# Insert a memory into hyperbolic space (||x|| < 1.0 required)
node_id = db.insert_node(embedding=[0.1, 0.2, 0.3], metadata={"text": "Nietzsche on memory"})

# KNN search
results = db.knn_search(embedding=[0.1, 0.2, 0.3], k=10)

# Run NQL query
results = db.query("MATCH (m:Memory) WHERE m.depth > 0.7 RETURN m LIMIT 5")

# Trigger sleep and reconsolidate
report = db.trigger_sleep(noise=0.02, adam_lr=0.005)
print(f"delta_hausdorff={report.hausdorff_delta:.3f}, committed={report.committed}")
```

### Go (sdk-papa-caolho)
```go
import nietzsche "sdk-papa-caolho"

client, _ := nietzsche.ConnectInsecure("localhost:50052")
defer client.Close()

// Insert node with Poincare embedding
node, _ := client.InsertNode(ctx, nietzsche.InsertNodeOpts{
    Coords:   []float64{0.1, 0.2, 0.3},
    Content:  map[string]string{"text": "first memory"},
    NodeType: "Semantic",
})

// KNN search
results, _ := client.KnnSearch(ctx, []float64{0.1, 0.2, 0.3}, 10, "")

// Graph algorithms
scores, _ := client.RunPageRank(ctx, nietzsche.PageRankOpts{Iterations: 20})

// Trigger sleep
sleep, _ := client.TriggerSleep(ctx, nietzsche.SleepOpts{Noise: 0.02})
fmt.Printf("deltaH=%.3f committed=%v\n", sleep.HausdorffDelta, sleep.Committed)
```

Go SDK covers all 65+ RPCs: collections, nodes, edges, batch operations, query, search, traversal, algorithms, backup, CDC, merge, sensory, indexes, lifecycle.

### TypeScript & C++
Located in `sdks/ts/` and `sdks/cpp/`.

---

## Development Roadmap

```
PHASE 0   Foundation & environment          ✅ COMPLETE
PHASE 1   Node and edge model               ✅ COMPLETE
PHASE 2   Graph storage engine              ✅ COMPLETE
PHASE 3   Traversal engine                  ✅ COMPLETE
PHASE 4   NQL query language                ✅ COMPLETE
PHASE 5   L-System engine                   ✅ COMPLETE
PHASE 6   Fractal diffusion / Pregel        ✅ COMPLETE
PHASE 7   ACID transactions on graph        ✅ COMPLETE
PHASE 8   Reconsolidation (sleep cycle)     ✅ COMPLETE
PHASE 9   Public API + SDKs                 ✅ COMPLETE
PHASE 10  Benchmarks, hardening, production ✅ COMPLETE
PHASE 11  Sensory compression layer         ✅ COMPLETE
PHASE Z   Zaratustra evolution engine       ✅ COMPLETE
PHASE A+B Unified gRPC API (65+ RPCs)       ✅ COMPLETE
PHASE D   Merge semantics (upsert)          ✅ COMPLETE
PHASE G   Cluster foundation (gossip)       ✅ COMPLETE
PHASE GPU GPU acceleration (cuVS CAGRA)     ✅ COMPLETE
PHASE TPU TPU acceleration (PJRT)           ✅ COMPLETE

── Production Hardening Roadmap ─────────────────────────────────
P0.1  TTL Reaper (background expiry)       ✅ COMPLETE
P0.2  RBAC (Admin/Writer/Reader roles)     ✅ COMPLETE
P0.3  Backup Hardening (scheduled+prune)   ✅ COMPLETE
P1.4  NQL CREATE / SET / DELETE            ✅ COMPLETE
P1.5  Metadata Secondary Indexes           ✅ COMPLETE
P1.6  Cluster Gossip Wiring                ✅ COMPLETE
P2.7  Encryption at-rest (AES-256-CTR)     ✅ COMPLETE
P2.8  Multi-hop Path NQL (BoundedBFS)      ✅ COMPLETE
P2.9  ListStore (RPUSH/LRANGE/LLEN)        ✅ COMPLETE
P3.10 Query Cost Estimator (EXPLAIN)       ✅ COMPLETE
P3.11 Hybrid BM25+ANN (RRF fusion)         ✅ COMPLETE
P3.12 Schema Validation (per-NodeType)     ✅ COMPLETE

── Consolidation Sprint (2026-02-21) ──────────────────────
C0.1  Bug fixes (5 real bugs)              ✅ COMPLETE
C0.2  Test coverage (+389 new tests)       ✅ COMPLETE
       Sprint 1: +166 (algo, zaratustra, gpu, graph, query, sparse, autotuner, snapshot)
       Sprint 2: +127 (tpu: 28, cugraph: 41, sdk: 28, wasm: 30)
       Sprint 3: +96  (embed: 49, cli: 47) — 100% module coverage
C1.1  NQL Time Functions (NOW/INTERVAL)    ✅ COMPLETE
C1.2  ListStore list_del method            ✅ COMPLETE
C2.1  SparseVector type                    ✅ COMPLETE
C2.2  HNSW Auto-tuner (ef_search)          ✅ COMPLETE
C2.3  Named Snapshots (time-travel)        ✅ COMPLETE

── Expansion Sprint (2026-02-21) ──────────────────────
E0.1  MCP Server (AI assistant tools)      ✅ COMPLETE  (19 tools, 19 tests)
E0.2  Prometheus/OTel metrics export       ✅ COMPLETE  (12 metrics, 6 tests)
E0.3  Filtered KNN + Roaring Bitmaps       ✅ COMPLETE  (5 filter types, 15 tests)
E0.4  Named Vectors (multi-vector/node)    ✅ COMPLETE  (3 metrics, 8 tests)
E0.5  Product Quantization (PQ)            ✅ COMPLETE  (magnitude-preserving, 12 tests)
E0.6  Secondary Indexes (arbitrary field)  ✅ COMPLETE  (3 index types, 13 tests)
E0.7  Kafka Connect Sink (CDC)             ✅ COMPLETE  (6 mutation types, 9 tests)
E0.8  Table Store (SQLite)                 ✅ COMPLETE  (7 column types, 15 tests)
E0.9  Media/Blob Store (OpenDAL)           ✅ COMPLETE  (5 media types, 8 tests)
E1.0  Go SDK batch RPCs                    ✅ COMPLETE  (42/42 RPCs)

── EVA-Mind Compatibility Sprint (2026-02-21) ──────────
A.1   Multi-Metric HNSW fix + DotProduct   ✅ COMPLETE  (Euclidean bug fixed, DotProduct added)
A.2   EmbeddedVectorStore as default       ✅ COMPLETE  (Mock → Embedded, real HNSW by default)
B.2   KNN metadata filter push-down        ✅ COMPLETE  (MetadataFilter → RoaringBitmap pre-filter)
D.1   MergeEdge ON MATCH + edge metadata   ✅ COMPLETE  (update_edge_metadata + WAL entries)
D.2   IncrementEdgeMeta RPC               ✅ COMPLETE  (atomic counter increment on edges)
E.1   Persistent secondary index registry  ✅ COMPLETE  (create/drop/list + backfill + startup load)
E.2   NQL executor index integration       ✅ COMPLETE  (auto O(log N) scan for indexed WHERE)
E.3   Index management gRPC RPCs           ✅ COMPLETE  (CreateIndex/DropIndex/ListIndexes)

── NQL & EVA-Mind Compatibility (2026-02-21) ──────────
NQL-1 MERGE statement (ON CREATE/ON MATCH)  ✅ COMPLETE  (node + edge MERGE with upsert)
NQL-2 Multi-hop typed path (*1..4)          ✅ COMPLETE  (BFS with depth + label filter)
NQL-3 SET with arithmetic expressions       ✅ COMPLETE  (n.count = n.count + 1, per-node eval)
NQL-4 CREATE with TTL support               ✅ COMPLETE  (ttl property → expires_at auto-compute)
NQL-5 DETACH DELETE                         ✅ COMPLETE  (node + all incident edges)
NQL-6 Edge property access in WHERE/RETURN  ✅ COMPLETE  (edge alias -[r:TYPE]-> with r.field)
NQL-7 ORDER BY on edge properties           ✅ COMPLETE  (r.weight, r.created_at, etc.)
Ph.C  Redis-compatible cache RPCs           ✅ COMPLETE  (CacheSet/Get/Del + ReapExpired)
Ph.F  Sensory RPCs (fully connected)        ✅ COMPLETE  (insert/get/reconstruct/degrade)
Ph.G  Per-collection RwLock concurrency     ✅ COMPLETE  (DashMap + tokio::sync::RwLock)
```

---

## Benchmarks

Run all benchmarks:
```bash
cargo bench --workspace
```

Individual suites:

| Suite | Command |
|---|---|
| Graph engine | `cargo bench -p nietzsche-graph` |
| Riemannian ops | `cargo bench -p nietzsche-sleep` |
| Chebyshev / diffusion | `cargo bench -p nietzsche-pregel` |
| Hyperbolic math | `cargo bench -p nietzsche-hyp-ops` |
| Distance metrics | `cargo bench -p hyperspace-core` |

### Representative results (ring graph, Apple M2)

| Benchmark | N | Time |
|---|---|---|
| `graph/insert_node` | 1 | ~18 us |
| `graph/insert_node_batch` | 100 | ~1.4 ms |
| `graph/scan_nodes` | 500 | ~3.8 ms |
| `graph/bfs` | chain-50 | ~52 us |
| `graph/dijkstra` | chain-50 | ~81 us |
| `riemannian/exp_map` | dim=256 | ~420 ns |
| `riemannian/adam_10_steps` | dim=64 | ~6.2 us |
| `chebyshev/apply_heat_kernel` | ring-40 | ~210 us |
| `chebyshev/laplacian_build` | ring-50 | ~390 us |

---

## Production Deployment

### Docker Compose (recommended)

```yaml
# docker-compose.yaml
services:
  nietzsche:
    build: .
    ports:
      - "50052:50051"   # gRPC
      - "8080:8080"     # HTTP dashboard
    environment:
      NIETZSCHE_DATA_DIR:            /data/nietzsche
      NIETZSCHE_PORT:                "50051"
      NIETZSCHE_DASHBOARD_PORT:      "8080"
      NIETZSCHE_SLEEP_INTERVAL_SECS: "300"
      ZARATUSTRA_INTERVAL_SECS:      "600"
    volumes:
      - nietzsche_data:/data/nietzsche

  hyperspace:
    build:
      context: .
      dockerfile: deploy/docker/Dockerfile
    ports:
      - "50051:50051"
      - "50050:50050"
    environment:
      NIETZSCHE_ADDR: http://nietzsche:50052
    depends_on:
      - nietzsche
    volumes:
      - hyperspace_data:/data/hyperspace

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

### Docker (standalone)

```bash
# Build
docker build -t nietzsche-db:latest .

# Run
docker run -d \
  -p 50051:50051 \
  -p 8080:8080 \
  -v /data/nietzsche:/data/nietzsche \
  -e NIETZSCHE_LOG_LEVEL=info \
  -e NIETZSCHE_SLEEP_INTERVAL_SECS=300 \
  -e NIETZSCHE_DASHBOARD_PORT=8080 \
  --name nietzsche-db \
  nietzsche-db:latest
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NIETZSCHE_DATA_DIR` | `/data/nietzsche` | RocksDB + WAL + collections root |
| `NIETZSCHE_PORT` | `50051` | gRPC listen port |
| `NIETZSCHE_DASHBOARD_PORT` | `8080` | HTTP dashboard port (0 = disabled) |
| `NIETZSCHE_LOG_LEVEL` | `info` | Tracing filter (`trace`, `debug`, `info`, `warn`, `error`) |
| `NIETZSCHE_SLEEP_INTERVAL_SECS` | `0` | Sleep cycle interval in seconds (0 = disabled) |
| `NIETZSCHE_SLEEP_NOISE` | `0.02` | Tangent-space perturbation magnitude |
| `NIETZSCHE_SLEEP_ADAM_STEPS` | `10` | RiemannianAdam steps per sleep cycle |
| `NIETZSCHE_HAUSDORFF_THRESHOLD` | `0.15` | Max delta-hausdorff before rollback |
| `NIETZSCHE_MAX_CONNECTIONS` | `1024` | Maximum concurrent gRPC connections |
| `NIETZSCHE_VECTOR_BACKEND` | `embedded` | `embedded` (HNSW), `gpu`, `tpu`, or empty (mock) |
| `NIETZSCHE_API_KEY` | — | Admin auth token for gRPC (backward compat) |
| `NIETZSCHE_API_KEY_ADMIN` | — | Admin role API key |
| `NIETZSCHE_API_KEY_WRITER` | — | Writer role API key (read + mutate) |
| `NIETZSCHE_API_KEY_READER` | — | Reader role API key (read only) |
| `NIETZSCHE_ENCRYPTION_KEY` | — | Base64-encoded 32-byte AES master key (empty = disabled) |
| `NIETZSCHE_TTL_REAPER_INTERVAL_SECS` | `60` | TTL reaper scan interval (0 = disabled) |
| `NIETZSCHE_BACKUP_INTERVAL_SECS` | `0` | Automatic backup interval (0 = disabled) |
| `NIETZSCHE_BACKUP_RETENTION_COUNT` | `5` | Max backups to keep (older ones pruned) |
| `NIETZSCHE_INDEXED_FIELDS` | — | CSV of metadata fields to index (e.g. `created_at,category`) |
| `ZARATUSTRA_INTERVAL_SECS` | `600` | Zaratustra cycle interval (0 = disabled) |
| `NIETZSCHE_CLUSTER_ENABLED` | `false` | Enable cluster mode |
| `NIETZSCHE_CLUSTER_NODE_NAME` | `nietzsche-0` | Human-readable node name |
| `NIETZSCHE_CLUSTER_ROLE` | `primary` | `primary`, `replica`, or `coordinator` |
| `NIETZSCHE_CLUSTER_SEEDS` | — | Comma-separated seed peer addresses |
| `PJRT_PLUGIN_PATH` | — | Path to `libtpu.so` for TPU backend |

### Health Check

```bash
# gRPC health (requires grpcurl)
grpcurl -plaintext localhost:50051 nietzsche.NietzscheDB/HealthCheck

# HTTP dashboard health
curl http://localhost:8080/api/health

# List all collections
curl http://localhost:8080/api/collections

# Graph data for visualization
curl "http://localhost:8080/api/graph?collection=eva_core&limit=500"
```

### CI / CD

**`.github/workflows/ci.yml`** — runs on every PR:

| Job | What it does |
|---|---|
| `lint` | `cargo fmt --check` + `cargo clippy -D warnings` |
| `test` | `cargo test --workspace --all-features` (requires `protoc` + `libclang`) |
| `bench-dry-run` | `cargo bench --no-run --workspace` (compile check) |
| `docker` | `docker build` (validates Dockerfile; no push on PRs) |

**`.github/workflows/deploy-gcp.yml`** — runs on push to `main`:

| Job | What it does |
|---|---|
| `build-and-push` | Builds Docker image, pushes to GCP Artifact Registry via WIF |
| `deploy` | SSH into GCP VM via OS Login, runs `docker compose up -d` |
| Health check | Verifies `GET /api/health` returns 200 |

---

## GPU Acceleration

NietzscheDB supports GPU-accelerated vector search via **NVIDIA cuVS CAGRA** and GPU graph traversal via **NVIDIA cuGraph**.

### Vector Search — `nietzsche-hnsw-gpu`

```
Insert → CPU staging buffer (Vec<f32>)
               │
               ├── n < 1,000 vectors  → CPU linear scan
               └── n >= 1,000 vectors → CAGRA build on GPU (lazy, on first knn)
                                         └── GPU search → results back to CPU
```

- Lazy CAGRA index build: only constructs GPU index when first k-NN query arrives
- Dirty ratio rebuild: reconstructs when >= 10% of index modified
- Automatic fallback to CPU if GPU fails

### Graph Traversal — `nietzsche-cugraph`

- GPU-accelerated BFS, Dijkstra, PageRank via cuGraph FFI
- Custom CUDA kernel for Poincare distance computation (compiled via NVRTC)
- Dynamic `libcugraph.so` loading at runtime

### Build (GCP / Linux)

```bash
# 1. CUDA Toolkit 12.x + cuVS 24.6
apt-get install -y clang libclang-dev

# 2. Build with GPU support
cargo build --release --features gpu

# 3. Run with GPU backend
NIETZSCHE_VECTOR_BACKEND=gpu ./target/release/nietzsche-server
```

### Docker (GPU)

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS builder
RUN apt-get update && apt-get install -y clang libclang-dev curl
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
COPY . .
RUN cargo build --release --features gpu

FROM nvidia/cuda:12.4-runtime-ubuntu22.04
COPY --from=builder /target/release/nietzsche-server /usr/local/bin/
EXPOSE 50051 8080
CMD ["nietzsche-server"]
```

```yaml
# docker-compose.gpu.yml
services:
  nietzsche-server:
    image: nietzsche-server:gpu
    runtime: nvidia
    environment:
      NIETZSCHE_VECTOR_BACKEND: gpu
      NIETZSCHE_DATA_DIR: /data/nietzsche
    ports:
      - "50051:50051"
      - "8080:8080"
    volumes:
      - nietzsche_data:/data/nietzsche
```

### GCP GPU Instance Recommendation

| Instance | GPU | VRAM | Best for |
|---|---|---|---|
| `g2-standard-4` | L4 | 24 GB | Production — best price/perf |
| `n1-standard-4` + T4 | T4 | 16 GB | Budget option |
| `a2-highgpu-1g` | A100 | 40 GB | Large-scale datasets |

### Feature Flags

```toml
# nietzsche-server/Cargo.toml
[features]
gpu = ["dep:nietzsche-hnsw-gpu"]   # enables GPU injection in main.rs

# nietzsche-hnsw-gpu/Cargo.toml
[features]
cuda = ["dep:cuvs", "dep:ndarray"] # enables actual CUDA calls
```

---

## TPU Acceleration

NietzscheDB supports Google **TPU**-accelerated vector search via the **PJRT C API**, targeting Cloud TPU VMs (v5e, v6e Trillium, v7 Ironwood).

### Architecture

```
Insert → CPU staging buffer (Vec<f32>)
               │
               ├── n < 1,000 vectors  → CPU linear scan
               └── n >= 1,000 vectors → lazy MHLO compile (once)
                                         ├── upload %query  → TPU
                                         ├── upload %matrix → TPU
                                         ├── execute MHLO kernel
                                         └── CPU: L2 norm correction
```

### Hardware Targets

| TPU | Generation | HBM |
|---|---|---|
| v5e | 5th gen | 16 GB/chip |
| v6e | Trillium | 32 GB/chip |
| v7 | **Ironwood** | **192 GB/chip** |

### Build (Cloud TPU VM)

```bash
PJRT_PLUGIN_PATH=/lib/libtpu.so \
cargo build --release --features tpu

PJRT_PLUGIN_PATH=/lib/libtpu.so \
NIETZSCHE_VECTOR_BACKEND=tpu \
./target/release/nietzsche-server
```

### Feature Flags

```toml
# nietzsche-server/Cargo.toml
[features]
tpu = ["dep:nietzsche-tpu", "nietzsche-tpu/tpu"]

# nietzsche-tpu/Cargo.toml
[features]
tpu = ["dep:pjrt"]
```

### GPU vs TPU

| Feature | GPU (cuVS CAGRA) | TPU (PJRT MHLO) |
|---|---|---|
| Index type | ANN graph (HNSW-like) | Exact dot-product batch |
| Best for | Ultra-low latency, single query | High throughput, large batch |
| Memory | GPU VRAM (CUDA managed) | TPU HBM (192 GB on Ironwood) |
| Cloud | GCP GPU instances | GCP Cloud TPU VMs |
| Feature flag | `--features gpu` | `--features tpu` |
| Env var | `NIETZSCHE_VECTOR_BACKEND=gpu` | `NIETZSCHE_VECTOR_BACKEND=tpu` |

CPU-only build (default) compiles and runs correctly — GPU/TPU paths simply not activated.

---

## Codebase Structure

```
NietzscheDB/
├── Cargo.toml                ← unified Rust workspace (38 crates)
├── rust-toolchain.toml       ← nightly channel
├── Dockerfile                ← multi-stage production image
├── docker-compose.yaml       ← nietzsche + hyperspace + prometheus + grafana
├── .github/workflows/        ← CI + deploy pipelines
├── crates/
│   ├── hyperspace-core/      ← Poincare HNSW, distance metrics, SIMD
│   ├── hyperspace-store/     ← mmap segments, WAL v3
│   ├── hyperspace-index/     ← HNSW graph, ArcSwap lock-free updates
│   ├── hyperspace-server/    ← gRPC server, multi-tenancy, replication
│   ├── hyperspace-proto/     ← protobuf definitions
│   ├── hyperspace-cli/       ← CLI tools (ratatui TUI)
│   ├── hyperspace-embed/     ← ONNX + remote embedding
│   ├── hyperspace-wasm/      ← WASM / browser / IndexedDB
│   ├── hyperspace-sdk/       ← HyperspaceDB Rust client
│   ├── nietzsche-graph/      ← hyperbolic graph engine         [Phases 1-3]
│   ├── nietzsche-hyp-ops/    ← Poincare ball math primitives
│   ├── nietzsche-query/      ← NQL parser + executor           [Phase 4]
│   ├── nietzsche-lsystem/    ← L-System + Hausdorff pruning    [Phase 5]
│   ├── nietzsche-pregel/     ← heat kernel diffusion           [Phase 6]
│   ├── nietzsche-sleep/      ← sleep/reconsolidation cycle     [Phase 8]
│   ├── nietzsche-zaratustra/ ← autonomous evolution engine     [Phase Z]
│   ├── nietzsche-algo/       ← 11 graph algorithms
│   ├── nietzsche-sensory/    ← sensory compression layer       [Phase 11]
│   ├── nietzsche-cluster/    ← gossip cluster foundation       [Phase G]
│   ├── nietzsche-wiederkehr/ ← DAEMON agents + Will to Power scheduler
│   ├── nietzsche-dream/      ← dream queries (speculative exploration)
│   ├── nietzsche-narrative/  ← narrative engine (story arc detection)
│   ├── nietzsche-agency/     ← autonomous agency + counterfactual engine
│   ├── nietzsche-mcp/        ← MCP server for AI assistants (19 tools)
│   ├── nietzsche-metrics/    ← Prometheus/OpenTelemetry metrics export
│   ├── nietzsche-filtered-knn/ ← filtered KNN with Roaring Bitmaps
│   ├── nietzsche-named-vectors/ ← multi-vector per node
│   ├── nietzsche-pq/         ← Product Quantization (magnitude-preserving)
│   ├── nietzsche-secondary-idx/ ← secondary indexes by arbitrary field
│   ├── nietzsche-kafka/      ← Kafka Connect sink (CDC streaming)
│   ├── nietzsche-table/      ← relational table store (SQLite)
│   ├── nietzsche-media/      ← media/blob store (OpenDAL: S3, GCS, local)
│   ├── nietzsche-api/        ← unified gRPC API (65+ RPCs)
│   ├── nietzsche-sdk/        ← Rust client SDK
│   ├── nietzsche-server/     ← production binary + dashboard
│   ├── nietzsche-hnsw-gpu/   ← GPU vector search (cuVS CAGRA)
│   ├── nietzsche-tpu/        ← TPU vector search (PJRT)
│   └── nietzsche-cugraph/    ← GPU graph traversal (cuGraph)
├── dashboard/                ← React 19 + Cosmograph 2.1 + Tailwind 4
│   ├── src/pages/            ← Overview, Collections, Nodes, Graph, Data, Settings
│   └── dist/                 ← single-file HTML (embedded in binary)
├── sdks/
│   ├── go/                   ← sdk-papa-caolho (65+ RPCs, full coverage)
│   ├── python/               ← gRPC client + proto generation
│   ├── ts/                   ← TypeScript SDK
│   └── cpp/                  ← C++ SDK
├── scripts/                  ← benchmark, build-dashboard, build-wasm, verify
├── benchmarks/               ← reproducible benchmark suite
├── integrations/             ← LangChain Python + JS, LlamaIndex
├── deploy/                   ← Docker, Kubernetes, WIF setup
├── docs/                     ← NQL reference, architecture
└── examples/                 ← HiveMind Tauri app, Python, TypeScript
```

---

## Build Profiles

```toml
# Release (production)
[profile.release]
lto = true
codegen-units = 1
strip = true
panic = "abort"
opt-level = 3

# Bench-fast (CI benchmarks)
[profile.bench-fast]
inherits = "release"
lto = "thin"
codegen-units = 4

# Perf (maximum native CPU optimization)
[profile.perf]
inherits = "release"
# RUSTFLAGS="-C target-cpu=native"
```

---

## Research Context

NietzscheDB closes gaps that no existing database fills:

- **No production HNSW is natively hyperbolic.** hnswlib, FAISS, Qdrant, Milvus, Weaviate — all use Euclidean geometry internally. NietzscheDB (native hyperbolic HNSW) is genuinely original work.
- **No graph database has intrinsic hyperbolic geometry.** Neo4j, ArangoDB, TigerGraph — all flat.
- **No AI memory system has a formal sleep/reconsolidation cycle.** NietzscheDB implements Riemannian optimization with Hausdorff identity verification and automatic rollback.
- **No database has an autonomous fractal growth engine.** The L-System rewrites the graph topology every tick based on production rules and local Hausdorff dimension.
- **No database ships 11 graph algorithms with both gRPC and REST interfaces.** PageRank, Louvain, A*, WCC, SCC, betweenness, closeness, degree, label propagation, triangle count, Jaccard — all built-in.
- **No graph database has built-in hybrid BM25+ANN search.** NietzscheDB fuses full-text BM25 with hyperbolic KNN via Reciprocal Rank Fusion (RRF) in a single API call.
- **No graph database has application-level encryption with per-CF key derivation.** AES-256-CTR with HKDF-SHA256 derives unique keys per column family from a single master key.
- **No graph database has magnitude-preserving Product Quantization.** PQ compresses vectors while preserving `‖x‖` — critical for hyperbolic depth semantics. Binary Quantization is explicitly rejected (destroys hierarchy).
- **No graph database ships with a built-in MCP server.** NietzscheDB exposes 19 tools via the Model Context Protocol for direct AI assistant integration.
- **No graph database bridges graph + relational + blob storage in one engine.** SQLite table store with NodeRef foreign keys + OpenDAL media store (S3/GCS/local) — unified under one API.

Key references:
- Krioukov et al., "Hyperbolic Geometry of Complex Networks" (2010)
- Ganea et al., "Hyperbolic Neural Networks" (NeurIPS 2018)
- Defferrard et al., "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" (2016)
- Becigneul & Ganea, "Riemannian Adaptive Optimization Methods" (ICLR 2019)
- HyperCore (Yale, 2025) — HyperbolicGraphRAG, Lorentz ViT, HypLoRA
- Hyperbolic Graph Wavelet Neural Network — Tsinghua 2025

---

## Git Remotes

```bash
origin    https://github.com/JoseRFJuniorLLMs/NietzscheDB.git   # this repo
upstream  https://github.com/YARlabs/hyperspace-db.git          # upstream sync
```

To pull upstream HNSW improvements:
```bash
git fetch upstream
git merge upstream/main --allow-unrelated-histories
```

---

## License

NietzscheDB inherits the **AGPL-3.0** license from HyperspaceDB.
Commercial licensing available — see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).

---

<p align="center">
  <em>"He who has a why to live can bear almost any how."</em><br>
  — Friedrich Nietzsche
</p>

<p align="center">
  Built for <strong>EVA-Mind</strong> · Powered by <strong>Rust nightly</strong> · <strong>38 crates</strong> · <strong>65+ gRPC RPCs</strong> · <strong>MCP + Prometheus</strong> · <strong>GPU/TPU</strong> · <strong>RBAC + Encryption</strong>
</p>
