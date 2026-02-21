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
  <img src="https://img.shields.io/badge/crates-25%20workspace-informational.svg" alt="Crates">
  <img src="https://img.shields.io/badge/gRPC-55%2B%20RPCs-blueviolet.svg" alt="RPCs">
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
│   · 11 built-in graph algorithms                        │
│   · Hybrid BM25+ANN search with RRF fusion              │
│   · RBAC + AES-256-CTR encryption at-rest                │
│   · Schema validation + metadata secondary indexes       │
│   · Autonomous evolution (Zaratustra cycle)             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**In plain language:**
- *For users:* "A memory database that thinks in hierarchies, grows like a plant, and sleeps to consolidate what it learned."
- *For engineers:* "A hyperbolic graph with native L-System growth, heat kernel diffusion as a search primitive, GPU/TPU vector backends, 45+ gRPC RPCs, and periodic Riemannian reconsolidation."
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
| Query language | k-NN only | NQL — graph, vector, diffusion, CREATE/SET/DELETE, EXPLAIN |
| Search | Vector OR text | Hybrid BM25+KNN with RRF fusion |
| Graph analytics | External tool needed | 11 built-in algorithms (PageRank, Louvain, A*, ...) |
| Hardware acceleration | CPU only or proprietary | GPU (cuVS CAGRA) + TPU (PJRT) at runtime |
| Security | API key at best | RBAC (Admin/Writer/Reader) + AES-256-CTR encryption at-rest |
| Data integrity | No schema enforcement | Per-NodeType schema validation (required fields + types) |
| Consistency | Single-store | ACID saga pattern across graph + vector store |

---

## Architecture

NietzscheDB is built as a **Rust nightly workspace** with 25 crates in two layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NietzscheDB Layer (16 crates)                   │
│                                                                         │
│  Engine:     nietzsche-graph    nietzsche-query     nietzsche-hyp-ops   │
│  Growth:     nietzsche-lsystem  nietzsche-pregel    nietzsche-sleep     │
│  Evolution:  nietzsche-zaratustra                                       │
│  Analytics:  nietzsche-algo     nietzsche-sensory                       │
│  Infra:      nietzsche-api      nietzsche-server    nietzsche-cluster   │
│  SDKs:       nietzsche-sdk                                              │
│  Accel:      nietzsche-hnsw-gpu nietzsche-tpu       nietzsche-cugraph   │
├─────────────────────────────────────────────────────────────────────────┤
│                     HyperspaceDB Layer (9 crates — fork base)           │
│                                                                         │
│  hyperspace-core   hyperspace-index   hyperspace-store                  │
│  hyperspace-server hyperspace-proto   hyperspace-cli                    │
│  hyperspace-embed  hyperspace-wasm    hyperspace-sdk                    │
└─────────────────────────────────────────────────────────────────────────┘
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

Sixteen new crates built on top of the foundation:

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
- **EmbeddedVectorStore** abstraction: CPU (HnswIndex) / GPU (GpuVectorStore) / TPU (TpuVectorStore) / Mock — selected at runtime via `NIETZSCHE_VECTOR_BACKEND`
- **Encryption at-rest** (`encryption.rs`): AES-256-CTR with HKDF-SHA256 per-CF key derivation from master key (`NIETZSCHE_ENCRYPTION_KEY`)
- **Schema validation** (`schema.rs`): per-NodeType constraints (required fields, field types), persisted in CF_META, enforced on `insert_node`
- **Metadata secondary indexes** (`CF_META_IDX`): arbitrary field indexing with FNV-1a + sortable value encoding for range scans
- **ListStore** (`CF_LISTS`): per-node ordered lists with RPUSH/LRANGE/LLEN semantics, atomic sequence counters
- **TTL / expires_at** enforcement: background reaper scans expired nodes and deletes them automatically
- **Full-text search + hybrid** (`fulltext.rs`): inverted index with BM25 scoring, plus RRF fusion with KNN vector search

#### `nietzsche-hyp-ops` — Poincare Ball Math
Core hyperbolic geometry primitives: Mobius addition, exponential/logarithmic maps, geodesic distance, parallel transport. Used by all other crates that need Poincare ball operations. Includes criterion benchmarks.

#### `nietzsche-query` — NQL Query Language
Nietzsche Query Language — a declarative query language with first-class hyperbolic primitives. Parser built with `pest` (PEG grammar). 19 unit + integration tests.

**[Full NQL Reference: docs/NQL.md](docs/NQL.md)**

Query types:

| Type | Description |
|---|---|
| `MATCH` | Pattern matching on nodes/paths with hyperbolic conditions |
| `CREATE` | Insert new nodes with labels and properties |
| `MATCH … SET` | Update matched nodes' properties |
| `MATCH … DELETE` | Delete matched nodes |
| `MERGE` | Upsert nodes/edges (ON CREATE SET / ON MATCH SET) |
| `DIFFUSE` | Multi-scale heat-kernel activation propagation |
| `RECONSTRUCT` | Decode sensory data from latent vector |
| `EXPLAIN` | Return execution plan with cost estimates |

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

-- Update matched nodes
MATCH (n:Semantic) WHERE n.energy < 0.1 SET n.energy = 0.5 RETURN n

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

#### `nietzsche-api` — Unified gRPC API
Single endpoint for all NietzscheDB capabilities — **55+ RPCs** over a single `NietzscheDB` service. Every data-plane RPC accepts a `collection` field; empty -> `"default"`.

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

  // ── Batch Operations ──────────────────────────────────────────
  rpc BatchInsertNodes(BatchInsertNodesRequest)   returns (BatchInsertNodesResponse);
  rpc BatchInsertEdges(BatchInsertEdgesRequest)   returns (BatchInsertEdgesResponse);

  // ── Query & Search ────────────────────────────────────────────
  rpc Query(QueryRequest)               returns (QueryResponse);
  rpc KnnSearch(KnnRequest)             returns (KnnResponse);
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

  // ── Change Data Capture ───────────────────────────────────────
  rpc SubscribeCDC(CdcRequest)            returns (stream CdcEvent);

  // ── Cluster ─────────────────────────────────────────────────
  rpc ExchangeGossip(GossipRequest)       returns (GossipResponse);

  // ── Schema Validation ───────────────────────────────────────
  rpc SetSchema(SetSchemaRequest)         returns (StatusResponse);
  rpc GetSchema(GetSchemaRequest)         returns (GetSchemaResponse);
  rpc ListSchemas(Empty)                  returns (ListSchemasResponse);

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

Go SDK covers all 55+ RPCs: collections, nodes, edges, query, search, traversal, algorithms, backup, CDC, merge, sensory, lifecycle.

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
PHASE A+B Unified gRPC API (45+ RPCs)       ✅ COMPLETE
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
C0.2  Test coverage (+156 new tests)       ✅ COMPLETE
C1.1  NQL Time Functions (NOW/INTERVAL)    ✅ COMPLETE
C1.2  ListStore list_del method            ✅ COMPLETE
C2.1  SparseVector type                    ✅ COMPLETE
C2.2  HNSW Auto-tuner (ef_search)          ✅ COMPLETE
C2.3  Named Snapshots (time-travel)        ✅ COMPLETE
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
├── Cargo.toml                ← unified Rust workspace (25 crates)
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
│   ├── nietzsche-api/        ← unified gRPC API (45+ RPCs)
│   ├── nietzsche-sdk/        ← Rust client SDK
│   ├── nietzsche-server/     ← production binary + dashboard
│   ├── nietzsche-hnsw-gpu/   ← GPU vector search (cuVS CAGRA)
│   ├── nietzsche-tpu/        ← TPU vector search (PJRT)
│   └── nietzsche-cugraph/    ← GPU graph traversal (cuGraph)
├── dashboard/                ← React 19 + Cosmograph 2.1 + Tailwind 4
│   ├── src/pages/            ← Overview, Collections, Nodes, Graph, Data, Settings
│   └── dist/                 ← single-file HTML (embedded in binary)
├── sdks/
│   ├── go/                   ← sdk-papa-caolho (45+ RPCs, full coverage)
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
  Built for <strong>EVA-Mind</strong> · Powered by <strong>Rust nightly</strong> · <strong>25 crates</strong> · <strong>55+ gRPC RPCs</strong> · <strong>GPU/TPU</strong> · <strong>RBAC + Encryption</strong>
</p>
