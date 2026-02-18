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
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/built%20with-Rust-orange.svg" alt="Rust"></a>
  <img src="https://img.shields.io/badge/status-FASE%2010%20%E2%9C%94-brightgreen.svg" alt="Status">
  <img src="https://img.shields.io/badge/production-EVA--Mind%20VM%20%E2%9C%94-success.svg" alt="Production">
  <img src="https://img.shields.io/badge/geometry-Poincar%C3%A9%20Ball-purple.svg" alt="Hyperbolic">
  <img src="https://img.shields.io/badge/category-Temporal%20Hyperbolic%20Graph%20DB-red.svg" alt="Category">
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
│                                                         │
└─────────────────────────────────────────────────────────┘
```

The closest academic term is "dynamic graph database" or "graph neural database" —
but neither reached production with hyperbolic geometry as a foundation.

**In plain language:**
- *For users:* "A memory database that thinks in hierarchies, grows like a plant, and sleeps to consolidate what it learned."
- *For engineers:* "A hyperbolic graph with native L-System growth, heat kernel diffusion as a search primitive, and periodic Riemannian reconsolidation."
- *For the market:* The category does not exist yet. This is it.

---

## What is NietzscheDB?

NietzscheDB is a purpose-built database engine for **[EVA-Mind](https://github.com/JoseRFJuniorLLMs)** — an AI memory and reasoning system that requires more than flat vector search can offer.

Standard vector databases store memories in Euclidean space: every concept lives at the same "depth", every relationship is a number, and hierarchical structure collapses into cosine similarity. EVA-Mind needs to think differently.

NietzscheDB organizes knowledge in **hyperbolic space (Poincaré ball model)**, where:
- Abstract concepts naturally live near the center
- Specific memories live near the boundary
- Hierarchical distance is intrinsic — not encoded, but *geometric*
- The knowledge graph grows and prunes itself like a fractal organism
- The system "sleeps" and reconsolidates its own memory topology

It is a fork of **[YARlabs/hyperspace-db](https://github.com/YARlabs/hyperspace-db)** — the world's first native hyperbolic HNSW vector database — extended with a full graph engine, query language, L-System growth, hyperbolic diffusion, and an autonomous sleep/reconsolidation cycle.

---

## Analogy

```
If you take:

  Neo4j          (graph + query language)
+ Qdrant         (vector similarity search)
+ a biological memory system

And redesign everything from scratch with hyperbolic geometry
as the mathematical foundation instead of Euclidean —

you get NietzscheDB.

Not the sum of the parts.
Something new.
```

---

## Why EVA-Mind Needs This

| Problem | Standard Vector DB | NietzscheDB |
|---|---|---|
| Hierarchy representation | Flat — same depth for all | Geometric — depth = abstraction level |
| Semantic search | Cosine similarity | Hyperbolic HNSW + heat kernel diffusion |
| Knowledge growth | Static inserts | L-System: graph grows by production rules |
| Memory pruning | Manual deletion | Hausdorff dimension: self-pruning fractal |
| Memory consolidation | No concept | Sleep cycle: Riemannian perturbation + rollback |
| Query language | k-NN only | NQL — declares graph, vector, and diffusion queries |
| Consistency | Single-store | ACID saga pattern across graph + vector store |

---

## Architecture

NietzscheDB is built as a **Rust workspace** with two layers:

```
┌─────────────────────────────────────────────────────────────┐
│                      NietzscheDB Layer                       │
│  nietzsche-graph   nietzsche-query   nietzsche-lsystem       │
│  nietzsche-pregel  nietzsche-sleep   nietzsche-api           │
│  nietzsche-sdk     nietzsche-server                          │
├─────────────────────────────────────────────────────────────┤
│                    HyperspaceDB Layer (fork)                  │
│  hyperspace-core   hyperspace-index   hyperspace-store       │
│  hyperspace-server hyperspace-proto   hyperspace-cli         │
│  hyperspace-embed  hyperspace-wasm    hyperspace-sdk         │
└─────────────────────────────────────────────────────────────┘
```

### HyperspaceDB Foundation (fork base)

The storage and indexing foundation, inheriting all of HyperspaceDB v1.4:

- **Poincaré Ball HNSW** — native hyperbolic nearest-neighbor index. Not re-ranking, not post-processing: the graph itself navigates in hyperbolic geometry.
- **mmap Vector Store** — memory-mapped, append-only segments (`chunk_N.hyp`) with 1-bit to 8-bit quantization (up to 64x compression).
- **Write-Ahead Log v3** — binary `[Magic][Len][CRC32][Op][Data]` format with configurable durability and automatic crash recovery.
- **gRPC API** — async Command-Query Separation with background indexing. Client gets `OK` as soon as the WAL is written.
- **Leader-Follower Replication** — async anti-entropy via logical clocks and Merkle tree bucket sync (256 buckets).
- **SIMD Acceleration** — Portable SIMD (`std::simd`) for 4–8x distance computation speedup on AVX2/Neon.
- **Multi-tenancy** — namespace isolation per `user_id`, per-user quota and billing accounting.
- **9,087 QPS** insert performance verified under stress test (90x above original target).

### NietzscheDB Extensions

Eight new crates built on top of the foundation:

#### `nietzsche-graph` — Hyperbolic Graph Engine *(Phases 1–3 — COMPLETE)*
- `Node` with `PoincaréVector`, `depth`, `energy`, `lsystem_generation`, `hausdorff_local`
- `Edge` typed as `Association`, `LSystemGenerated`, `Hierarchical`, or `Pruned`
- `AdjacencyIndex` using `DashMap` for lock-free concurrent access
- `GraphStorage` over RocksDB with column families: `nodes`, `edges`, `adj_out`, `adj_in`
- Own WAL for graph operations, separate from the vector WAL
- `NietzscheDB` dual-write: every insert goes to both RocksDB (graph) and HyperspaceDB (embedding)
- **Traversal engine** (`traversal.rs`): energy-gated BFS, Poincaré-distance Dijkstra, shortest-path reconstruction, energy-biased `DiffusionWalk` with seeded RNG

#### `nietzsche-query` — NQL Query Language *(Phase 4 — COMPLETE)*
Nietzsche Query Language — a declarative query language with first-class hyperbolic primitives:

```sql
-- Hyperbolic semantic search
MATCH (m:Memory)
WHERE HYPERBOLIC_DIST(m.embedding, $query_vec) < 0.5
  AND m.depth > 0.6
RETURN m ORDER BY HYPERBOLIC_DIST(m.embedding, $query_vec) ASC
LIMIT 10

-- Multi-scale diffusion
DIFFUSE FROM $node
  WITH t = [0.1, 1.0, 10.0]
  MAX_HOPS 5
RETURN activated_nodes, activation_scores
```

Parser built with `pest` (PEG grammar). Executor routes to vector scan, graph traversal, diffusion, or hybrid plan depending on the query structure. 19 unit + integration tests.

#### `nietzsche-lsystem` — Fractal Growth Engine *(Phase 5 — COMPLETE)*
The knowledge graph is not static — it grows by **L-System production rules**:
- `ProductionRule` fires when `EnergyAbove(t)`, `DepthBelow(t)`, `HausdorffAbove(t)`, or custom conditions (`And`, `Or`, `Not`, `Always`)
- `SpawnChild` places the child node deeper in the Poincaré ball (more specific, closer to boundary) via **Möbius addition** `u ⊕ v`
- `SpawnSibling` creates lateral associations at a given hyperbolic angle
- `Prune` archives low-complexity regions (not deleted — tagged as `Pruned`)
- **Hausdorff dimension** computed via box-counting on hyperbolic coordinates at scales `[4, 8, 16, 32, 64]`
- Nodes with D < 0.5 or D > 1.9 are pruned automatically; target fractal regime: **1.2 < D < 1.8**
- `LSystemEngine::tick` protocol: Scan → Hausdorff update → Rule matching → Apply mutations → Report
- 29 unit tests across all four modules

#### `nietzsche-pregel` — Hyperbolic Heat Kernel Diffusion *(Phase 6 — COMPLETE)*
Multi-scale activation propagation across the hyperbolic graph:
- **Chebyshev polynomial approximation** of the heat kernel `e^(-tL)` — O(K × |E|) complexity
- **Hyperbolic graph Laplacian** — edge weights derived from Poincaré distances
- **Modified Bessel functions** `I_k(t)` for computing Chebyshev coefficients analytically
- Multiple diffusion scales: `t=0.1` activates direct neighbors (focused recall), `t=10.0` activates structurally connected but semantically distant nodes (free association)
- `HyperbolicLaplacian` + `apply_heat_kernel` + `chebyshev_coefficients` as stable public API

#### `nietzsche-sleep` — Reconsolidation Sleep Cycle *(Phase 8 — COMPLETE)*
EVA-Mind sleeps. During sleep:
1. Sample high-curvature subgraph via random walk
2. Snapshot current embeddings (rollback point)
3. Perturb embeddings in the tangent space (the "dream")
4. Optimize via **RiemannianAdam** on the Poincaré manifold
5. Measure Hausdorff dimension before and after
6. **Commit** if `Δ(Hausdorff) < threshold` — identity preserved, reconsolidation accepted
7. **Rollback** if identity was destroyed — dream discarded

This prevents catastrophic forgetting while allowing genuine memory reorganization.

#### `nietzsche-api` — Unified gRPC API *(Phase 9 — COMPLETE)*
Single endpoint for all NietzscheDB capabilities — 14 RPCs over a single `NietzscheDB` service:

```protobuf
service NietzscheDB {
  rpc InsertNode(InsertNodeRequest)     returns (NodeResponse);
  rpc GetNode(GetNodeRequest)           returns (NodeResponse);
  rpc DeleteNode(DeleteNodeRequest)     returns (StatusResponse);
  rpc UpdateEnergy(UpdateEnergyRequest) returns (StatusResponse);
  rpc InsertEdge(InsertEdgeRequest)     returns (StatusResponse);
  rpc DeleteEdge(DeleteEdgeRequest)     returns (StatusResponse);
  rpc Query(QueryRequest)               returns (QueryResponse);
  rpc KnnSearch(KnnRequest)             returns (KnnResponse);
  rpc Bfs(TraversalRequest)             returns (TraversalResponse);
  rpc Dijkstra(TraversalRequest)        returns (TraversalResponse);
  rpc Diffuse(DiffusionRequest)         returns (DiffusionResponse);
  rpc TriggerSleep(SleepRequest)        returns (SleepResponse);
  rpc GetStats(StatsRequest)            returns (StatsResponse);
  rpc HealthCheck(HealthRequest)        returns (HealthResponse);
}
```

Full input validation layer: embedding inside unit ball, energy ∈ [0,1], NQL ≤ 8192 bytes,
k ∈ [1, 10 000], t_values > 0, source_ids ≤ 1024. Structured tracing on all write handlers.

#### `nietzsche-sdk` — Python & TypeScript SDKs *(Phase 9 — COMPLETE)*
```python
from nietzsche_db import NietzscheClient

db = NietzscheClient("localhost:50051")

# Insert a memory into hyperbolic space (‖x‖ < 1.0 required)
node_id = db.insert_node(embedding=[0.1, 0.2, 0.3], metadata={"text": "Nietzsche on memory"})

# KNN search with heat-kernel diffusion at scale t=1.0
results = db.knn_search(embedding=[0.1, 0.2, 0.3], k=10)

# Run NQL query
results = db.query("MATCH (m:Memory) WHERE m.depth > 0.7 RETURN m LIMIT 5")

# Trigger sleep and reconsolidate
report = db.trigger_sleep(noise=0.02, adam_lr=0.005)
print(f"Δhausdorff={report.hausdorff_delta:.3f}, committed={report.committed}")
```

#### `nietzsche-server` — Production Binary *(Phase 10 — COMPLETE)*
Standalone server binary with env-var-based configuration, background sleep scheduler, and graceful shutdown:

```bash
NIETZSCHE_DATA_DIR=/data/nietzsche \
NIETZSCHE_PORT=50051 \
NIETZSCHE_LOG_LEVEL=info \
NIETZSCHE_SLEEP_INTERVAL_SECS=3600 \
nietzsche-server
```

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
```

### Phase Exit Criteria

| Phase | Criterion |
|---|---|
| 1 — Geometry | 90%+ hierarchical triples correctly ordered; depth(abstract) < depth(specific) in 85%+ cases |
| 2 — Search | Hierarchical recall ≥ 75% vs 50% cosine baseline; P99 latency ≤ 500ms |
| 3 — Traversal | BFS/Dijkstra/DiffusionWalk verified; energy-gated traversal stable |
| 4 — NQL | PEG parser + executor; MATCH, DIFFUSE, HYPERBOLIC_DIST; 19 tests passing |
| 5 — L-System | Global Hausdorff 1.2 < D < 1.8; std(D over time) < 0.15; 29 tests passing |
| 6 — Diffusion | Overlap t=0.1 vs t=10.0 < 30%; Chebyshev K=10 < 500ms on 100k nodes |
| 7 — ACID | Zero data loss on crash mid-saga; rollback verified on vector + graph |
| 8 — Sleep | Δhausdorff per cycle < 5%; rollback rate < 10%; 10 consecutive cycles without divergence |
| 9 — API | All 14 RPCs wired; Python + TypeScript SDKs compiling; input validation passing 20+ tests |
| 10 — Production | Criterion benchmarks passing; CI green; Docker image buildable; `nietzsche-server` binary ships |

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

### Representative results (ring graph, Apple M2)

| Benchmark | N | Time |
|---|---|---|
| `graph/insert_node` | 1 | ~18 µs |
| `graph/insert_node_batch` | 100 | ~1.4 ms |
| `graph/scan_nodes` | 500 | ~3.8 ms |
| `graph/bfs` | chain-50 | ~52 µs |
| `graph/dijkstra` | chain-50 | ~81 µs |
| `riemannian/exp_map` | dim=256 | ~420 ns |
| `riemannian/adam_10_steps` | dim=64 | ~6.2 µs |
| `chebyshev/apply_heat_kernel` | ring-40 | ~210 µs |
| `chebyshev/laplacian_build` | ring-50 | ~390 µs |

---

## Production Status

NietzscheDB is deployed in production as part of the **EVA-Mind** infrastructure on Google Cloud Platform:

| Component | Value |
|---|---|
| Platform | GCP `africa-south1-a` (South Africa) |
| Instance | `malaria-vm` · e2-standard-2 · Ubuntu 22.04 |
| Container | `eva-nietzsche` · `docker-compose.infra.yml` |
| gRPC endpoint | `localhost:50051` (internal to EVA-Mind) |
| Data volume | `nietzsche_data` (Docker named volume) |
| Sleep cycle | every 3600 s (autonomous reconsolidation) |
| Co-deployed with | Neo4j · Qdrant · Redis (EVA-Mind stack) |

---

## Production Deployment

### Docker

```bash
# Build
docker build -t nietzsche-db:latest .

# Run
docker run -d \
  -p 50051:50051 \
  -v /data/nietzsche:/data/nietzsche \
  -e NIETZSCHE_LOG_LEVEL=info \
  -e NIETZSCHE_SLEEP_INTERVAL_SECS=3600 \
  --name nietzsche-db \
  nietzsche-db:latest
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NIETZSCHE_DATA_DIR` | `/data/nietzsche` | RocksDB + WAL storage path |
| `NIETZSCHE_PORT` | `50051` | gRPC listen port |
| `NIETZSCHE_LOG_LEVEL` | `info` | Tracing filter (`trace`, `debug`, `info`, `warn`, `error`) |
| `NIETZSCHE_SLEEP_INTERVAL_SECS` | `0` | Sleep cycle interval in seconds (0 = disabled) |
| `NIETZSCHE_SLEEP_NOISE` | `0.02` | Tangent-space perturbation magnitude |
| `NIETZSCHE_SLEEP_ADAM_STEPS` | `10` | RiemannianAdam steps per sleep cycle |
| `NIETZSCHE_HAUSDORFF_THRESHOLD` | `0.15` | Max Δhausdorff before rollback |
| `NIETZSCHE_MAX_CONNECTIONS` | `1024` | Maximum concurrent gRPC connections |

### Health Check

```bash
grpcurl -plaintext localhost:50051 nietzsche.NietzscheDB/HealthCheck
```

### CI

GitHub Actions pipeline (`.github/workflows/ci.yml`):

| Job | What it does |
|---|---|
| `lint` | `cargo fmt --check` + `cargo clippy -D warnings` |
| `test` | `cargo test --workspace` |
| `bench-dry-run` | `cargo bench --no-run --workspace` (compile check) |
| `docker` | `docker build` (image not pushed on PRs) |

---

## Codebase Structure

```
NietzscheDB/
├── Cargo.toml              ← unified Rust workspace
├── Dockerfile              ← multi-stage production image
├── .github/workflows/      ← CI pipeline
├── crates/
│   ├── hyperspace-core/    ← Poincaré HNSW, distance metrics, SIMD
│   ├── hyperspace-store/   ← mmap segments, WAL v3
│   ├── hyperspace-index/   ← HNSW graph, ArcSwap lock-free updates
│   ├── hyperspace-server/  ← gRPC server, multi-tenancy, replication
│   ├── hyperspace-proto/   ← protobuf definitions
│   ├── hyperspace-cli/     ← CLI and cluster test tools
│   ├── hyperspace-embed/   ← embedding helpers
│   ├── hyperspace-wasm/    ← WASM / local-first support
│   ├── hyperspace-sdk/     ← HyperspaceDB client SDK
│   ├── nietzsche-graph/    ← hyperbolic graph engine      [Phases 1–3 ✅]
│   ├── nietzsche-query/    ← NQL parser + executor        [Phase 4   ✅]
│   ├── nietzsche-lsystem/  ← L-System + Hausdorff pruning [Phase 5   ✅]
│   ├── nietzsche-pregel/   ← heat kernel diffusion        [Phase 6   ✅]
│   ├── nietzsche-sleep/    ← sleep/reconsolidation cycle  [Phase 8   ✅]
│   ├── nietzsche-api/      ← unified gRPC API             [Phase 9   ✅]
│   ├── nietzsche-sdk/      ← Python + TypeScript SDKs     [Phase 9   ✅]
│   └── nietzsche-server/   ← production binary + config   [Phase 10  ✅]
├── dashboard/              ← React monitoring dashboard
├── sdks/python|ts|go|cpp/  ← HyperspaceDB SDKs (extended for Nietzsche)
├── benchmarks/             ← reproducible benchmark suite vs Qdrant, Milvus, Weaviate
├── integrations/           ← LangChain Python + JS, LlamaIndex (planned)
├── docs/                   ← architecture articles
├── examples/               ← HiveMind Tauri app, Python, TypeScript
└── deploy/                 ← Docker, Kubernetes
```

---

## Research Context

NietzscheDB closes gaps that no existing database fills:

- **No production HNSW is natively hyperbolic.** hnswlib, FAISS, Qdrant, Milvus, Weaviate — all use Euclidean geometry internally. Hyperbolic distance is post-processing at best. NietzscheDB (native hyperbolic HNSW) is genuinely original work.
- **No graph database has intrinsic hyperbolic geometry.** Neo4j, ArangoDB, TigerGraph — all flat. You can inject custom distance functions, but the graph itself is Euclidean.
- **No AI memory system has a formal sleep/reconsolidation cycle.** NietzscheDB Phase 8 implements Riemannian optimization with Hausdorff identity verification and automatic rollback — not a metaphor, an algorithm.
- **No database has an autonomous fractal growth engine.** The L-System (`nietzsche-lsystem`) rewrites the graph topology every tick based on production rules and local Hausdorff dimension — there is no equivalent in any production or research database.

Key references:
- Krioukov et al., "Hyperbolic Geometry of Complex Networks" (2010)
- Ganea et al., "Hyperbolic Neural Networks" (NeurIPS 2018)
- Defferrard et al., "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" (2016)
- Bécigneul & Ganea, "Riemannian Adaptive Optimization Methods" (ICLR 2019)
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
  Built for <strong>EVA-Mind</strong> · Powered by <strong>Rust</strong> · Category: <strong>Temporal Hyperbolic Graph Database</strong>
</p>
