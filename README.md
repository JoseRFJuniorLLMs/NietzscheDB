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
- `Node` = `NodeMeta` (~100 bytes: id, depth, energy, node_type, hausdorff_local, content) + `PoincareVector` (embedding, stored separately for 10-25x traversal speedup)
- `PoincareVector` with `Vec<f32>` coords (distance kernel promotes to f64 internally for numerical stability near the Poincaré boundary)
- `Edge` typed as `Association`, `LSystemGenerated`, `Hierarchical`, or `Pruned`
- `AdjacencyIndex` using `DashMap` for lock-free concurrent access
- `GraphStorage` over RocksDB with 8 column families: `nodes` (NodeMeta), `embeddings` (PoincareVector), `edges`, `adj_out`, `adj_in`, `meta`, `sensory`, `energy_idx`
- Own WAL for graph operations, separate from the vector WAL
- `NietzscheDB` dual-write: every insert goes to both RocksDB (graph) and HyperspaceDB (embedding)
- **Traversal engine** (`traversal.rs`): energy-gated BFS (reads only NodeMeta — ~100 bytes per hop), Poincaré-distance Dijkstra, shortest-path reconstruction, energy-biased `DiffusionWalk` with seeded RNG

#### `nietzsche-query` — NQL Query Language *(Phase 4 — COMPLETE)*
Nietzsche Query Language — a declarative query language with first-class hyperbolic primitives. Parser built with `pest` (PEG grammar). 19 unit + integration tests.

**→ [Full NQL Reference: docs/NQL.md](docs/NQL.md)**

Query types:

| Type | Description |
|---|---|
| `MATCH` | Pattern matching on nodes/paths with hyperbolic conditions |
| `DIFFUSE` | Multi-scale heat-kernel activation propagation |
| `RECONSTRUCT` | Decode sensory data from latent vector (Phase 11) |
| `EXPLAIN` | Return execution plan without running |

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
WHERE RIEMANN_CURVATURE(n) > 0.3      -- Ollivier-Ricci curvature
  AND HAUSDORFF_DIM(n) BETWEEN 1.2 AND 1.8
  AND DIRICHLET_ENERGY(n) < 0.1
RETURN n ORDER BY RIEMANN_CURVATURE(n) DESC LIMIT 10

-- Multi-scale heat-kernel diffusion
DIFFUSE FROM $seed
  WITH t = [0.1, 1.0, 10.0]
  MAX_HOPS 6
RETURN path

-- Execution plan inspection
EXPLAIN MATCH (n) WHERE n.energy > 0.5 RETURN n LIMIT 10
```

**Built-in geometric functions:**

| Function | Named after | Computes |
|---|---|---|
| `HYPERBOLIC_DIST(n.e, $q)` | — | Poincaré ball geodesic distance |
| `POINCARE_DIST(n, $q)` | Henri Poincaré | Same — explicit model name |
| `KLEIN_DIST(n, $q)` | Felix Klein | Beltrami-Klein distance |
| `RIEMANN_CURVATURE(n)` | Bernhard Riemann | Ollivier-Ricci curvature |
| `HAUSDORFF_DIM(n)` | Felix Hausdorff | Local fractal dimension |
| `GAUSS_KERNEL(n, t)` | Carl Friedrich Gauss | Heat kernel `exp(-d²/4t)` |
| `CHEBYSHEV_COEFF(n, k)` | Pafnuty Chebyshev | Chebyshev polynomial T_k |
| `DIRICHLET_ENERGY(n)` | P.G.L. Dirichlet | Local Dirichlet energy |
| `EULER_CHAR(n)` | Leonhard Euler | V − E characteristic |
| `LAPLACIAN_SCORE(n)` | P.-S. Laplace | Graph Laplacian diagonal |
| `LOBACHEVSKY_ANGLE(n, $p)` | N. Lobachevsky | Angle of parallelism |
| `MINKOWSKI_NORM(n)` | H. Minkowski | Conformal factor |
| `RAMANUJAN_EXPANSION(n)` | S. Ramanujan | Spectral expansion ratio |
| `FOURIER_COEFF(n, k)` | J. Fourier | Graph Fourier coefficient |

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

#### `nietzsche-api` — Unified gRPC API *(Fase A+B — COMPLETE)*
Single endpoint for all NietzscheDB capabilities — **22 RPCs** over a single `NietzscheDB` service. Every data-plane RPC accepts a `collection` field; empty → `"default"`.

```protobuf
service NietzscheDB {
  // ── Collection management (Fase B) ──────────────────────────────
  rpc CreateCollection(CreateCollectionRequest)   returns (CreateCollectionResponse);
  rpc DropCollection(DropCollectionRequest)       returns (StatusResponse);
  rpc ListCollections(ListCollectionsRequest)     returns (ListCollectionsResponse);

  // ── Graph CRUD ──────────────────────────────────────────────────
  rpc InsertNode(InsertNodeRequest)     returns (NodeResponse);
  rpc GetNode(NodeIdRequest)            returns (NodeResponse);
  rpc DeleteNode(NodeIdRequest)         returns (StatusResponse);
  rpc UpdateEnergy(UpdateEnergyRequest) returns (StatusResponse);
  rpc InsertEdge(InsertEdgeRequest)     returns (EdgeResponse);
  rpc DeleteEdge(EdgeIdRequest)         returns (StatusResponse);

  // ── Query & Search ──────────────────────────────────────────────
  rpc Query(QueryRequest)               returns (QueryResponse);
  rpc KnnSearch(KnnRequest)             returns (KnnResponse);

  // ── Traversal ───────────────────────────────────────────────────
  rpc Bfs(TraversalRequest)             returns (TraversalResponse);
  rpc Dijkstra(TraversalRequest)        returns (TraversalResponse);
  rpc Diffuse(DiffusionRequest)         returns (DiffusionResponse);

  // ── Lifecycle ───────────────────────────────────────────────────
  rpc TriggerSleep(SleepRequest)        returns (SleepResponse);

  // ── Sensory Compression Layer (Phase 11) ──────────────────────
  rpc InsertSensory(InsertSensoryRequest) returns (StatusResponse);
  rpc GetSensory(NodeIdRequest)           returns (SensoryResponse);
  rpc Reconstruct(ReconstructRequest)     returns (ReconstructResponse);
  rpc DegradeSensory(NodeIdRequest)       returns (StatusResponse);

  // ── Zaratustra — autonomous evolution (Phase Z) ───────────────
  rpc InvokeZaratustra(ZaratustraRequest) returns (ZaratustraResponse);

  // ── Admin ─────────────────────────────────────────────────────
  rpc GetStats(Empty)                     returns (StatsResponse);
  rpc HealthCheck(Empty)                  returns (StatusResponse);
}
```

Full input validation: embedding inside unit ball, energy ∈ [0,1], NQL ≤ 8192 bytes,
k ∈ [1, 10 000], t_values > 0, source_ids ≤ 1024. Structured tracing on all write handlers.

**Multi-collection example:**
```python
# Create a named collection with explicit dimension + metric
db.create_collection("patients", dim=1536, metric="cosine")
db.create_collection("core",     dim=768,  metric="cosine")

# All subsequent operations route to named collection
db.insert_node(embedding=[...], collection="patients")
results = db.query("MATCH (n) RETURN n LIMIT 5", collection="core")
```

#### `nietzsche-sdk` — Python, TypeScript & Go SDKs *(Phase 9 — COMPLETE)*

**Python:**
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

**Go (sdk-papa-caolho):**
```go
import nietzsche "sdk-papa-caolho"

client, _ := nietzsche.ConnectInsecure("localhost:50052")
defer client.Close()

// Insert node with Poincaré embedding
node, _ := client.InsertNode(ctx, nietzsche.InsertNodeOpts{
    Coords:   []float64{0.1, 0.2, 0.3},
    Content:  map[string]string{"text": "first memory"},
    NodeType: "Semantic",
})

// KNN search
results, _ := client.KnnSearch(ctx, []float64{0.1, 0.2, 0.3}, 10, "")

// NQL query with typed parameters
qr, _ := client.Query(ctx,
    "MATCH (m:Memory) WHERE m.depth > $d RETURN m LIMIT 5",
    map[string]interface{}{"d": 0.7}, "")

// Trigger sleep
sleep, _ := client.TriggerSleep(ctx, nietzsche.SleepOpts{Noise: 0.02})
fmt.Printf("ΔH=%.3f committed=%v\n", sleep.HausdorffDelta, sleep.Committed)
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
| 9 — API | All 22 RPCs wired; Python + TypeScript + Go SDKs compiling; input validation passing 20+ tests |
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
| Co-deployed with | NietzscheDB (replaces 2×Neo4j + Qdrant + Redis) |

---

## Production Deployment

### Docker Compose (EVA-Mind VM — recommended)

```yaml
# docker-compose.infra.yml — excerpt
services:
  nietzsche-server:
    image: us-central1-docker.pkg.dev/PROJECT/nietzsche/nietzsche-server:latest
    container_name: eva-nietzsche
    restart: unless-stopped
    ports:
      - "50051:50051"   # gRPC
      - "8082:8080"     # HTTP dashboard
    environment:
      NIETZSCHE_DATA_DIR:            /data/nietzsche
      NIETZSCHE_PORT:                "50051"
      NIETZSCHE_DASHBOARD_PORT:      "8080"
      NIETZSCHE_SLEEP_INTERVAL_SECS: "300"
    volumes:
      - nietzsche_data:/data/nietzsche
```

### Docker (standalone)

```bash
# Build
docker build -t nietzsche-db:latest .

# Run
docker run -d \
  -p 50051:50051 \
  -p 8082:8080 \
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
| `NIETZSCHE_HAUSDORFF_THRESHOLD` | `0.15` | Max Δhausdorff before rollback |
| `NIETZSCHE_MAX_CONNECTIONS` | `1024` | Maximum concurrent gRPC connections |
| `NIETZSCHE_VECTOR_BACKEND` | `embedded` | `embedded` (HNSW) or `mock` (in-memory) |

### Health Check

```bash
# gRPC health (requires grpcurl)
grpcurl -plaintext localhost:50051 nietzsche.NietzscheDB/HealthCheck

# HTTP dashboard health (always available)
curl http://localhost:8082/api/health

# List all collections
curl http://localhost:8082/api/collections
```

### CI / CD

**`.github/workflows/ci.yml`** — runs on every PR:

| Job | What it does |
|---|---|
| `lint` | `cargo fmt --check` + `cargo clippy -D warnings` |
| `test` | `cargo test --workspace` (requires `protoc` + `libclang`) |
| `bench-dry-run` | `cargo bench --no-run --workspace` (compile check) |
| `docker` | `docker build` (validates Dockerfile; no push on PRs) |

**`.github/workflows/deploy-gcp.yml`** — runs on push to `main`:

| Job | What it does |
|---|---|
| `build-and-push` | Builds Docker image, pushes to GCP Artifact Registry via WIF |
| `deploy` | SSH into GCP VM via OS Login, runs `docker compose up -d` |
| Health check | Verifies `GET /api/health` returns 200 |

Setup: run `deploy/scripts/setup-wif.sh` once to configure Workload Identity Federation and Artifact Registry, then add the printed values as GitHub Secrets.

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
│   │                          Go SDK: sdks/go/ (sdk-papa-caolho, 22 RPCs)
│   └── nietzsche-server/   ← production binary + config   [Phase 10  ✅]
├── dashboard/              ← React monitoring dashboard
├── sdks/
│   ├── go/                 ← sdk-papa-caolho (Go gRPC SDK, 22 RPCs, 25 tests)
│   ├── python|ts|cpp/      ← HyperspaceDB SDKs (extended for Nietzsche)
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

## GPU Acceleration — `nietzsche-hnsw-gpu`

NietzscheDB supports GPU-accelerated vector search via **NVIDIA cuVS CAGRA**
(Compressed-Adjacency Graph Retrieval Algorithm), delivering up to **10× faster
index construction** compared to CPU HNSW.

### Architecture

```
Insert → CPU staging buffer (Vec<f32>)
               │
               ├── n < 1.000 vectors  → CPU linear scan (GPU transfer overhead not worth it)
               └── n ≥ 1.000 vectors  → CAGRA build on GPU (lazy, on first knn)
                                         └── GPU search → results back to CPU
```

### Build Requirements (GCP / Linux)

```bash
# 1. CUDA Toolkit 12.x
#    https://developer.nvidia.com/cuda-downloads

# 2. cuVS 24.6
#    https://github.com/rapidsai/cuvs

# 3. libclang (needed by bindgen in cuvs-sys)
apt-get install -y clang libclang-dev

# 4. Build the server with GPU support
cargo build --release --features gpu

# 5. Run with GPU backend
NIETZSCHE_VECTOR_BACKEND=gpu ./target/release/nietzsche-server
```

### Docker (GCP GPU instance)

```yaml
# docker-compose.yml
services:
  nietzsche-server:
    image: nietzsche-server:gpu
    runtime: nvidia                          # NVIDIA Container Runtime
    environment:
      NIETZSCHE_VECTOR_BACKEND: gpu
      NIETZSCHE_DATA_DIR: /data/nietzsche
    ports:
      - "50051:50051"
      - "8082:8080"
    volumes:
      - nietzsche_data:/data/nietzsche
```

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

### GCP Instance Recommendation

| Instance | GPU | VRAM | Best for |
|---|---|---|---|
| `g2-standard-4` | L4 | 24 GB | Production — best price/perf |
| `n1-standard-4` + T4 | T4 | 16 GB | Budget option |
| `a2-highgpu-1g` | A100 | 40 GB | Large-scale datasets |

### How It Works at Startup

When the server starts with `--features gpu` and `NIETZSCHE_VECTOR_BACKEND=gpu`,
it automatically replaces the CPU HNSW with a GPU CAGRA store for every collection:

```
CollectionManager::open()
        ↓
For each collection:
  GpuVectorStore::new(dim)      → CAGRA initialised on GPU
  db.set_vector_store(gpu)      → replaces CPU HNSW
        ↓
gRPC server ready                → all knn queries go through GPU
```

### Crate Structure

| Crate | Role |
|---|---|
| `nietzsche-hnsw-gpu` | `GpuVectorStore` implementing `VectorStore` via cuVS CAGRA |
| `nietzsche-server --features gpu` | Injects GPU store at startup |
| `nietzsche-graph` | `AnyVectorStore::Gpu(Box<dyn VectorStore>)` — type-erased slot |

### Feature Flags

```toml
# nietzsche-server/Cargo.toml
[features]
gpu = ["dep:nietzsche-hnsw-gpu"]   # enables GPU injection in main.rs

# nietzsche-hnsw-gpu/Cargo.toml
[features]
cuda = ["dep:cuvs", "dep:ndarray"] # enables actual CUDA calls in GpuVectorStore
```

CPU-only build (default) compiles and runs correctly — GPU path simply not activated.

---

## TPU Acceleration — `nietzsche-tpu`

NietzscheDB supports Google **TPU**-accelerated vector search via the **PJRT C API**,
targeting Cloud TPU VMs (v5e, **v6e Trillium**, **v7 Ironwood**). This is the same
runtime used internally by JAX, PyTorch-XLA, and Google's own Gemini models.

### Architecture

```
Insert → CPU staging buffer (Vec<f32>)
               │
               ├── n < 1.000 vectors  → CPU linear scan (PJRT overhead not worth it)
               └── n ≥ 1.000 vectors  → lazy compact + MHLO compile (once)
                                         ├── upload %query  (D×f32)   → TPU
                                         ├── upload %matrix (N×D×f32) → TPU
                                         ├── execute MHLO: dots[i] = dot(matrix[i,:], query)
                                         └── CPU: L2² = q_norm² − 2·dots + m_norms²
```

The MHLO kernel runs the dominant O(N·D) computation on TPU. L2 norm corrections
are O(N) and applied on CPU using norms precomputed at build time.

### Hardware Targets

| TPU | Generation | FP8 TFLOPs | HBM | GA |
| --- | --- | --- | --- | --- |
| v5e | 5th gen | — | 16 GB/chip | ✅ |
| v6e | Trillium | — | 32 GB/chip | ✅ |
| v7 | **Ironwood** | **4,614** | **192 GB/chip** | ✅ Nov 2025 |

Ironwood (v7) is the current flagship: 42.5 Exaflops per pod, used by Anthropic for Claude.

### Build Requirements (Cloud TPU VM)

```bash
# 1. Provision a Cloud TPU VM (v5e / v6e / v7):
#    gcloud compute tpus tpu-vm create nietzsche-tpu-vm \
#      --zone=us-central2-b --accelerator-type=v5e-1 \
#      --version=tpu-ubuntu2204-base

# 2. SSH into the VM — libtpu.so is pre-installed at /lib/libtpu.so

# 3. Build the server with TPU support:
PJRT_PLUGIN_PATH=/lib/libtpu.so \
cargo build --release --features tpu

# 4. Run with TPU backend:
PJRT_PLUGIN_PATH=/lib/libtpu.so \
NIETZSCHE_VECTOR_BACKEND=tpu \
./target/release/nietzsche-server
```

### Docker (Cloud TPU VM)

```dockerfile
# Dockerfile.tpu — run on Cloud TPU VM
FROM ubuntu:22.04 AS builder
RUN apt-get update && apt-get install -y curl build-essential
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"
COPY . .
RUN cargo build --release --features tpu

FROM ubuntu:22.04
# libtpu.so must be present on the TPU VM host; mount it in:
COPY --from=builder /target/release/nietzsche-server /usr/local/bin/
EXPOSE 50051 8080
CMD ["nietzsche-server"]
```

```yaml
# docker-compose.tpu.yml — run on a TPU VM
services:
  nietzsche-server:
    image: nietzsche-server:tpu
    environment:
      PJRT_PLUGIN_PATH: /lib/libtpu.so    # pre-installed on TPU VMs
      NIETZSCHE_VECTOR_BACKEND: tpu
      NIETZSCHE_DATA_DIR: /data/nietzsche
    volumes:
      - /lib/libtpu.so:/lib/libtpu.so:ro  # host libtpu.so bind-mount
      - nietzsche_data:/data/nietzsche
    ports:
      - "50051:50051"
      - "8082:8080"
```

### TPU Startup Flow

```
CollectionManager::open()
        ↓
For each collection:
  TpuVectorStore::new(dim)      → PJRT client init (loads libtpu.so)
  db.set_vector_store(tpu)      → replaces CPU HNSW
        ↓
gRPC server ready                → all knn() calls compile MHLO lazily on first use
```

### TPU Crate Structure

| Crate | Role |
| --- | --- |
| `nietzsche-tpu` | `TpuVectorStore` implementing `VectorStore` via PJRT + MHLO |
| `nietzsche-server --features tpu` | Injects TPU store at startup |
| `nietzsche-graph` | `AnyVectorStore::Tpu(Box<dyn VectorStore>)` — type-erased slot |

### TPU Feature Flags

```toml
# nietzsche-server/Cargo.toml
[features]
tpu = ["dep:nietzsche-tpu", "nietzsche-tpu/tpu"]   # enables TPU injection

# nietzsche-tpu/Cargo.toml
[features]
tpu = ["dep:pjrt"]   # enables PJRT C API calls in TpuVectorStore
```

CPU-only build (default) compiles and runs correctly — TPU path simply not activated.
Falls back to CPU linear scan if `PJRT_PLUGIN_PATH` is not set or init fails.

### GPU vs TPU

| Feature | GPU (cuVS CAGRA) | TPU (PJRT MHLO) |
| --- | --- | --- |
| Index type | ANN graph (HNSW-like) | Exact dot-product batch |
| Best for | Ultra-low latency, single query | High throughput, large batch |
| Memory | GPU VRAM (CUDA managed) | TPU HBM (192 GB on Ironwood) |
| Scale | Up to ~100M vectors | Ironwood pod: 1.77 PB shared HBM |
| Cloud | GCP GPU instances | GCP Cloud TPU VMs |
| Feature flag | `--features gpu` | `--features tpu` |
| Env var | `NIETZSCHE_VECTOR_BACKEND=gpu` | `NIETZSCHE_VECTOR_BACKEND=tpu` |

For more details see [`docs/npu-deploy.md`](docs/npu-deploy.md).

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
