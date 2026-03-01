# AGENT 10: MASTER BLUEPRINT -- EVA v2.0 Neuro-Symbolic Architecture

**Date**: 2026-02-23
**Status**: Architecture Integration Plan
**Scope**: NietzscheDB (Rust workspace, 41 crates) + EVA (Go, ~30 internal packages)
**Authors**: Agent 10 (Architecture Integration), Claude Opus 4.6

---

## 1. Current Architecture Map

### 1.1 Complete Crate Dependency Graph (Existing)

```
FOUNDATION LAYER (NietzscheDB fork)
==============================================
nietzsche-core        (v2.0 - distance metrics, PoincareMetric, SIMD, vector ops)
    |
    +-- nietzsche-vecstore   (v2.0 - mmap vector storage, segments, WAL)
    |       |
    +-- nietzsche-hnsw   (v2.0 - HNSW graph, Poincare/Cosine/Euclidean, rkyv snapshots)
            |
            depends on: nietzsche-core, nietzsche-vecstore

CORE GRAPH ENGINE
==============================================
nietzsche-graph        (THE HUB - nodes, edges, adjacency, storage, RocksDB, CollectionManager)
    depends on: nietzsche-core, nietzsche-hnsw, nietzsche-vecstore
    |
    +-- nietzsche-query        (NQL parser/executor)
    |       depends on: nietzsche-graph
    |
    +-- nietzsche-lsystem      (L-System fractal growth engine)
    |       depends on: nietzsche-graph
    |
    +-- nietzsche-pregel       (BSP graph compute, Chebyshev heat-kernel diffusion)
    |       depends on: nietzsche-graph
    |
    +-- nietzsche-sleep        (Riemannian Adam optimizer, sleep cycle reconsolidation)
    |       depends on: nietzsche-graph, nietzsche-lsystem
    |
    +-- nietzsche-zaratustra   (Will to Power, Eternal Recurrence, Ubermensch evolution)
    |       depends on: nietzsche-graph, nietzsche-lsystem
    |
    +-- nietzsche-agency       (Autonomous daemons, meta-observer, reactor, desire engine)
    |       depends on: nietzsche-graph, nietzsche-pregel, nietzsche-lsystem
    |
    +-- nietzsche-dream        (Dream simulation engine, associative walks)
    |       depends on: nietzsche-graph, nietzsche-query
    |
    +-- nietzsche-narrative    (Narrative generation from graph events)
    |       depends on: nietzsche-graph
    |
    +-- nietzsche-wiederkehr   (Periodic daemon tasks, intent execution)
    |       depends on: nietzsche-graph, nietzsche-query
    |
    +-- nietzsche-sensory      (Multimodal sensory compression layer)
    |       depends on: nietzsche-graph, nietzsche-hyp-ops
    |
    +-- nietzsche-algo         (Graph algorithms: PageRank, Louvain, Betweenness, etc.)
    |       depends on: nietzsche-graph
    |
    +-- nietzsche-swartz       (Embedded SQL layer via GlueSQL)
    |       depends on: nietzsche-graph
    |
    +-- nietzsche-filtered-knn (Pre-filtered KNN via energy_idx)
    |       depends on: nietzsche-graph
    |
    +-- nietzsche-secondary-idx (Metadata secondary indexes)
    |       depends on: nietzsche-graph
    |
    +-- nietzsche-named-vectors (Multi-vector per node)
    |       depends on: nietzsche-graph
    |
    +-- nietzsche-kafka        (Kafka CDC integration)
    |       depends on: nietzsche-graph
    |
    +-- nietzsche-mcp          (Model Context Protocol integration)
    |       depends on: nietzsche-graph, nietzsche-query, nietzsche-algo
    |
    +-- nietzsche-cugraph      (cuGraph GPU graph analytics)
    |       depends on: nietzsche-graph (+optional libloading, cudarc)

ACCELERATOR LAYER
==============================================
nietzsche-hnsw-gpu     (NVIDIA cuVS CAGRA GPU vector search)
    depends on: nietzsche-graph (+optional cuvs, ndarray)

nietzsche-tpu          (Google TPU PJRT/MHLO vector search)
    depends on: nietzsche-graph (+optional pjrt)

STANDALONE UTILITY CRATES
==============================================
nietzsche-hyp-ops      (Pure hyperbolic math: exp_map, log_map, geodesic, Mobius ops)
    depends on: NOTHING (pure math library)

nietzsche-pq           (Product quantization)
    depends on: NOTHING (standalone)

nietzsche-cluster      (Cluster gossip/registry)
    depends on: NOTHING (standalone networking)

nietzsche-metrics      (Prometheus metrics)
    depends on: prometheus

nietzsche-table        (Embedded SQLite via rusqlite)
    depends on: rusqlite

nietzsche-media        (Blob storage via OpenDAL)
    depends on: opendal

API & SDK LAYER
==============================================
nietzsche-api          (Unified gRPC service - THE AGGREGATOR)
    depends on: nietzsche-graph, nietzsche-query, nietzsche-lsystem,
                nietzsche-pregel, nietzsche-sleep, nietzsche-zaratustra,
                nietzsche-sensory, nietzsche-algo, nietzsche-cluster,
                nietzsche-hyp-ops, nietzsche-wiederkehr, nietzsche-dream,
                nietzsche-narrative, nietzsche-swartz

nietzsche-sdk          (Rust SDK client)
    depends on: nietzsche-api

nietzsche-server       (THE BINARY - production gRPC+HTTP server)
    depends on: nietzsche-api, nietzsche-graph, nietzsche-sleep,
                nietzsche-query, nietzsche-zaratustra, nietzsche-algo,
                nietzsche-cluster, nietzsche-wiederkehr, nietzsche-agency,
                nietzsche-hyp-ops, nietzsche-lsystem, nietzsche-dream,
                nietzsche-narrative
    optional:   nietzsche-hnsw-gpu (feature: gpu)
                nietzsche-tpu      (feature: tpu)
```

### 1.2 Server Entry Point Call Chain

```
main.rs (nietzsche-server)
  |
  +-- Config::from_env()                           -- load all NIETZSCHE_* env vars
  +-- CollectionManager::open(&db_path)            -- nietzsche-graph (RocksDB, HNSW)
  |
  +-- [feature=gpu] GpuVectorStore::new()          -- nietzsche-hnsw-gpu (cuVS CAGRA)
  +-- [feature=tpu] TpuVectorStore::new()          -- nietzsche-tpu (PJRT/MHLO)
  |
  +-- BACKGROUND TASKS (tokio::spawn):
  |   +-- SleepCycle::run()                        -- nietzsche-sleep (Riemannian Adam)
  |   +-- ZaratustraEngine::run_cycle()            -- nietzsche-zaratustra (evolution)
  |   +-- TTL Reaper (db.reap_expired())           -- nietzsche-graph
  |   +-- Backup scheduler                         -- nietzsche-graph (BackupManager)
  |   +-- DaemonEngine::tick()                     -- nietzsche-wiederkehr
  |   +-- AgencyEngine::tick()                     -- nietzsche-agency (the brain)
  |       |-- DreamEngine::dream_from()            -- nietzsche-dream
  |       |-- NarrativeEngine::narrate()           -- nietzsche-narrative
  |       |-- LSystemEngine::tick()                -- nietzsche-lsystem
  |       |-- nietzsche_query::execute()           -- nietzsche-query (Code-as-Data)
  |
  +-- CdcBroadcaster::new()                        -- nietzsche-api (CDC stream)
  +-- AuthInterceptor::from_env()                  -- server auth module
  +-- NietzscheServer::new(cm, cdc)                -- nietzsche-api (gRPC service)
  +-- dashboard::serve()                           -- server dashboard module (HTTP :8080)
  +-- tonic::transport::Server::builder().serve()  -- gRPC :50051
```

### 1.3 EVA Go --> NietzscheDB gRPC Interface

```
EVA (Go, port 8091)
  |
  +-- internal/brainstem/infrastructure/nietzsche/
  |   |-- client.go               -- gRPC connection via nietzsche-sdk (Go)
  |   |-- graph_adapter.go        -- InsertNode, GetNode, InsertEdge, BFS, Dijkstra
  |   |-- vector_adapter.go       -- KnnSearch, HybridSearch
  |   |-- cache_adapter.go        -- CacheSet/Get/Del (NietzscheDB replacement)
  |   |-- manifold_adapter.go     -- Synthesis, CausalNeighbors, KleinPath
  |   |-- algo_adapter.go         -- PageRank, Louvain, etc.
  |   |-- sensory_adapter.go      -- InsertSensory, Reconstruct
  |   |-- narrative_adapter.go    -- Story/narrative from graph
  |   |-- wiederkehr_adapter.go   -- CreateDaemon, ListDaemons
  |   |-- sql_adapter.go          -- SqlQuery, SqlExec
  |   |-- backup_service.go       -- CreateBackup, RestoreBackup
  |   |-- cdc_listener.go         -- SubscribeCDC (streaming)
  |   |-- security_adapter.go     -- Schema validation
  |   |-- audio_buffer.go         -- Audio data handling
  |   |-- named_vectors_adapter.go -- Multi-vector operations
  |
  +-- internal/memory/krylov/
  |   |-- krylov_manager.go       -- Krylov subspace compression (1536D -> 64D)
  |   |-- adaptive_krylov.go      -- Adaptive dimension selection
  |   |-- hierarchical_krylov.go  -- Hierarchical subspace management
  |
  +-- internal/cortex/                -- "The Brain" of EVA
  |   |-- brain/                      -- Context + memory service
  |   |-- consciousness/              -- Global Workspace Theory
  |   |-- learning/                   -- Autonomous learning, meta-learning
  |   |-- prediction/                 -- Bayesian network, crisis predictor
  |   |-- spectral/                   -- Community detection, fractal dimension
  |   |-- attention/                  -- Affect stabilizer, center router
  |   |-- personality/                -- Interpretation validation
  |   |-- voice/speaker/embedder.go   -- ONNX runtime (ECAPA-TDNN, 192-dim)
  |   |-- ram/                        -- Retrieval-Augmented Memory
  |   |-- lacan/                      -- Lacanian narrative analysis
  |   |-- explainability/             -- Clinical decision explainer, PDF gen
  |   |-- selfawareness/              -- Self-model
  |   |-- evolution/                  -- System evolution
  |
  +-- internal/hippocampus/           -- Long-term memory
  |   |-- graph/                      -- Heat kernel, family tracker
  |   |-- memory/                     -- Retrieval, pattern miner, reflective
  |   |-- knowledge/                  -- Wisdom service
  |
  +-- internal/clinical/              -- Clinical safety layer
  |   |-- crisis/                     -- Crisis detection + notification
  |   |-- risk/                       -- Pediatric risk detection
  |   |-- goals/                      -- Goal tracking
  |   |-- notes/                      -- Clinical note generation
  |
  +-- nietzsche-sdk (Go, at ../NietzscheDB/sdks/go/)
      |-- client.go, nodes.go, edges.go, search.go, traversal.go, etc.
      |-- proto/nietzsche.proto       -- Shared proto definition
      |-- pb/nietzsche.pb.go          -- Generated gRPC stubs
```

### 1.4 Data Flow: Request --> Processing --> Storage --> Response

```
[EVA Go Client] --(gRPC)--> [nietzsche-server :50051]
                                    |
                                    v
                            [nietzsche-api]
                            (route to handler)
                                    |
              +---------------------+--------------------+
              |                     |                    |
        [InsertNode]          [KnnSearch]          [Query (NQL)]
              |                     |                    |
              v                     v                    v
       [nietzsche-graph]    [nietzsche-hnsw]    [nietzsche-query]
       (RocksDB CF_NODES)    (HNSW graph walk)    (parse -> plan -> exec)
       (adjacency lists)     (Poincare distance)        |
              |                     |              [nietzsche-graph]
              v                     v              [nietzsche-pregel]
      [nietzsche-vecstore]   [nietzsche-core]            |
      (mmap vector data)   (PoincareMetric)        (heat kernel)
              |                     |                    |
              v                     v                    v
       [RocksDB on disk]   [Memory-mapped files]  [In-memory compute]
```

### 1.5 Hot Paths (Latency-Critical)

| Path | Components | Target Latency |
|------|-----------|----------------|
| **KNN Search** | nietzsche-api -> nietzsche-graph -> nietzsche-hnsw -> nietzsche-core (Poincare distance) | < 10ms p99 |
| **InsertNode** | nietzsche-api -> nietzsche-graph -> RocksDB write + async HNSW indexing | < 5ms p99 |
| **GetNode** | nietzsche-api -> nietzsche-graph -> RocksDB read (currently ~25KB, target ~340B after NodeMeta split) | < 1ms p99 |
| **NQL WHERE filter** | nietzsche-query -> nietzsche-graph -> RocksDB scan with secondary indexes | < 50ms p99 |
| **BFS/Dijkstra** | nietzsche-api -> nietzsche-graph -> adjacency walk (in-memory) | < 20ms p99 |

### 1.6 Cold Paths (Where Neural Inference Can Run Without Blocking)

| Path | Current Owner | Why Cold |
|------|--------------|----------|
| **Sleep Cycle** | nietzsche-sleep (background tokio::spawn, configurable interval) | Riemannian Adam optimizer iterates ~10 steps/node. Perfect for neural-guided perturbation. |
| **Agency Tick** | nietzsche-agency (background, 60s default interval) | Meta-observer + reactor + desire engine. Neural attention scoring fits here. |
| **Zaratustra Cycle** | nietzsche-zaratustra (background, 600s default) | Evolution engine. Neural fitness evaluation natural fit. |
| **L-System Tick** | nietzsche-lsystem (triggered by agency) | Growth rules. Neural rule selection / parameter adaptation. |
| **Dream Engine** | nietzsche-dream (triggered by agency) | Associative walk. Neural graph traversal policy natural fit. |
| **Krylov Subspace** | EVA Go internal/memory/krylov | Compression 1536D->64D. VQ-VAE replacement candidate. |

---

## 2. New Crate Dependency Graph (Neural Extension)

### 2.1 New Crates to Create

```
NEURAL FOUNDATION
==============================================
nietzsche-neural       (NEW - ModelRegistry, ONNX runtime wrapper, tensor ops)
    depends on: ort (onnxruntime crate), ndarray, nietzsche-hyp-ops
    features: [onnx, tensorrt, tflite]
    |
    +-- nietzsche-gnn          (NEW - Graph Neural Networks: GAT, SAGE, HGT)
    |       depends on: nietzsche-neural, nietzsche-pregel, nietzsche-graph
    |
    +-- nietzsche-mcts         (NEW - Monte Carlo Tree Search for clinical decisions)
    |       depends on: nietzsche-neural, nietzsche-agency
    |
    +-- nietzsche-rl           (NEW - Reinforcement Learning: neural-guided L-System)
    |       depends on: nietzsche-neural, nietzsche-lsystem
    |
    +-- nietzsche-vqvae        (NEW - Vector Quantized VAE for latent compression)
    |       depends on: nietzsche-neural, nietzsche-pq, nietzsche-hyp-ops
    |
    +-- nietzsche-dsi          (NEW - Differentiable Search Index)
            depends on: nietzsche-neural, nietzsche-hnsw-gpu, nietzsche-query
```

### 2.2 Complete Dependency Tree (Existing + New)

```
Layer 0: STANDALONE (no internal deps)
    nietzsche-core, nietzsche-hyp-ops, nietzsche-pq,
    nietzsche-cluster, nietzsche-metrics, nietzsche-table,
    nietzsche-media

Layer 1: STORAGE
    nietzsche-vecstore  (depends: nietzsche-core)
    nietzsche-hnsw  (depends: nietzsche-core, nietzsche-vecstore)

Layer 2: GRAPH HUB
    nietzsche-graph   (depends: nietzsche-core/index/store)

Layer 3: GRAPH SERVICES
    nietzsche-query, nietzsche-lsystem, nietzsche-pregel,
    nietzsche-algo, nietzsche-narrative, nietzsche-swartz,
    nietzsche-filtered-knn, nietzsche-secondary-idx,
    nietzsche-named-vectors, nietzsche-kafka, nietzsche-cugraph,
    nietzsche-hnsw-gpu, nietzsche-tpu
    (all depend: nietzsche-graph)

Layer 3b: GRAPH SERVICES WITH CROSS-DEPS
    nietzsche-sensory   (depends: nietzsche-graph, nietzsche-hyp-ops)
    nietzsche-sleep     (depends: nietzsche-graph, nietzsche-lsystem)
    nietzsche-zaratustra(depends: nietzsche-graph, nietzsche-lsystem)
    nietzsche-dream     (depends: nietzsche-graph, nietzsche-query)
    nietzsche-wiederkehr(depends: nietzsche-graph, nietzsche-query)
    nietzsche-mcp       (depends: nietzsche-graph, nietzsche-query, nietzsche-algo)
    nietzsche-agency    (depends: nietzsche-graph, nietzsche-pregel, nietzsche-lsystem)

Layer 4: NEURAL FOUNDATION (NEW)
    nietzsche-neural    (depends: nietzsche-hyp-ops, ort, ndarray)

Layer 5: NEURAL MODULES (NEW)
    nietzsche-gnn       (depends: nietzsche-neural, nietzsche-pregel, nietzsche-graph)
    nietzsche-mcts      (depends: nietzsche-neural, nietzsche-agency)
    nietzsche-rl        (depends: nietzsche-neural, nietzsche-lsystem)
    nietzsche-vqvae     (depends: nietzsche-neural, nietzsche-pq, nietzsche-hyp-ops)
    nietzsche-dsi       (depends: nietzsche-neural, nietzsche-hnsw-gpu, nietzsche-query)

Layer 6: API AGGREGATOR
    nietzsche-api       (depends: ALL service crates + optionally neural crates)

Layer 7: SERVER BINARY
    nietzsche-server    (depends: nietzsche-api + all background engines)
```

### 2.3 Feature Flag Hierarchy

```toml
# In workspace Cargo.toml
[workspace.features]
# Master neural feature - enables ALL neural components
neural = [
    "nietzsche-neural",
    "nietzsche-gnn",
    "nietzsche-mcts",
    "nietzsche-rl",
    "nietzsche-vqvae",
    "nietzsche-dsi",
]

# Individual neural component flags
neural-gnn   = ["nietzsche-neural", "nietzsche-gnn"]
neural-mcts  = ["nietzsche-neural", "nietzsche-mcts"]
neural-rl    = ["nietzsche-neural", "nietzsche-rl"]
neural-vqvae = ["nietzsche-neural", "nietzsche-vqvae"]
neural-dsi   = ["nietzsche-neural", "nietzsche-dsi"]

# Existing accelerator features (unchanged)
gpu = ["nietzsche-hnsw-gpu"]
tpu = ["nietzsche-tpu"]

# In nietzsche-server/Cargo.toml
[features]
default = []
gpu     = ["dep:nietzsche-hnsw-gpu"]
tpu     = ["dep:nietzsche-tpu", "nietzsche-tpu/tpu"]
neural  = [
    "dep:nietzsche-neural",
    "dep:nietzsche-gnn",
    "dep:nietzsche-mcts",
    "dep:nietzsche-rl",
    "dep:nietzsche-vqvae",
    "dep:nietzsche-dsi",
]
neural-gnn   = ["dep:nietzsche-neural", "dep:nietzsche-gnn"]
neural-mcts  = ["dep:nietzsche-neural", "dep:nietzsche-mcts"]
neural-rl    = ["dep:nietzsche-neural", "dep:nietzsche-rl"]
neural-vqvae = ["dep:nietzsche-neural", "dep:nietzsche-vqvae"]
neural-dsi   = ["dep:nietzsche-neural", "dep:nietzsche-dsi"]
```

**Build commands**:
```bash
# Classic behavior (no neural) - default
cargo build --release --bin nietzsche-server

# With GPU
cargo build --release --bin nietzsche-server --features gpu

# With all neural components
cargo build --release --bin nietzsche-server --features neural

# With specific neural component + GPU
cargo build --release --bin nietzsche-server --features "neural-gnn,gpu"

# Full stack
cargo build --release --bin nietzsche-server --features "neural,gpu"
```

---

## 3. EVA Go Integration Architecture

### 3.1 Current EVA --> NietzscheDB Interface

EVA connects to NietzscheDB via:
- **Go SDK** (`nietzsche-sdk` at `../NietzscheDB/sdks/go/`) using `replace` directive in `go.mod`
- **gRPC** on port 50051 (address configured via `NIETZSCHE_GRPC_ADDR` env var)
- **15 adapter files** in `internal/brainstem/infrastructure/nietzsche/`
- **ONNX already used in Go**: `github.com/yalue/onnxruntime_go` for speaker embeddings (ECAPA-TDNN)

### 3.2 New EVA Go Packages for Neural Integration

```
EVA (Go)
+-- internal/neural/                    (NEW - shared ONNX inference utilities)
|   |-- model_registry.go              -- Local model cache, version management
|   |-- inference_client.go            -- gRPC client for NietzscheDB neural RPCs
|   |-- tensor_utils.go                -- Go tensor <-> protobuf conversion
|   |-- poincare_constraint.go         -- Enforce ||x|| < 1-epsilon in Go
|
+-- internal/cortex/mcts/              (NEW - clinical decision trees)
|   |-- tree.go                        -- MCTS node structure
|   |-- policy.go                      -- Neural policy network (via NietzscheDB gRPC)
|   |-- value.go                       -- Value network for state evaluation
|   |-- clinical_adapter.go            -- Map clinical states to MCTS nodes
|
+-- internal/cortex/worldmodel/        (NEW - DreamerV3 dream simulation)
|   |-- dreamer.go                     -- World model interface
|   |-- latent_state.go                -- Latent state representation
|   |-- imagination.go                 -- Trajectory rollout
|   |-- clinical_simulator.go          -- Clinical scenario simulation
|
+-- internal/cortex/attention/neural/  (NEW - neural attention scoring)
|   |-- scorer.go                      -- Replace heuristic attention with neural
|   |-- gnn_context.go                 -- Use GNN node embeddings for context
```

### 3.3 gRPC API Extensions Required

New RPCs to add to `nietzsche.proto`:

```protobuf
service NietzscheDB {
    // ... existing RPCs ...

    // ---- Neural Inference RPCs (Phase N) ----

    // Run GNN inference to get enriched node embeddings
    rpc GnnInfer (GnnInferRequest) returns (GnnInferResponse);

    // MCTS: expand and evaluate a decision tree
    rpc MctsSearch (MctsSearchRequest) returns (MctsSearchResponse);

    // Neural attention: score nodes by relevance to a context
    rpc NeuralAttention (NeuralAttentionRequest) returns (NeuralAttentionResponse);

    // VQ-VAE: encode sensory data to discrete codes
    rpc VqvaeEncode (VqvaeEncodeRequest) returns (VqvaeEncodeResponse);

    // VQ-VAE: decode discrete codes back to continuous
    rpc VqvaeDecode (VqvaeDecodeRequest) returns (VqvaeDecodeResponse);

    // DSI: neural document retrieval (query -> doc IDs directly)
    rpc DsiRetrieve (DsiRetrieveRequest) returns (DsiRetrieveResponse);

    // Model management
    rpc ListModels (Empty) returns (ModelListResponse);
    rpc LoadModel  (LoadModelRequest) returns (StatusResponse);
}

// ---- Neural messages ----

message GnnInferRequest {
    string collection     = 1;
    repeated string node_ids = 2;  // seed nodes for GNN context
    uint32 num_layers     = 3;     // GNN hops (default 2)
    string model_name     = 4;     // registered GNN model
}

message GnnInferResponse {
    map<string, PoincareVector> enriched_embeddings = 1;
    uint64 duration_ms = 2;
}

message MctsSearchRequest {
    string collection    = 1;
    string root_node_id  = 2;     // starting state
    uint32 simulations   = 3;     // number of MCTS rollouts (default 100)
    string policy_model  = 4;     // neural policy model name
    string value_model   = 5;     // neural value model name
}

message MctsSearchResponse {
    repeated MctsAction actions = 1;
    uint64 duration_ms = 2;
}

message MctsAction {
    string node_id    = 1;
    double value      = 2;
    uint32 visit_count = 3;
    double prior      = 4;
}

message NeuralAttentionRequest {
    string collection     = 1;
    repeated double context_embedding = 2;
    repeated string candidate_ids     = 3;
    string model_name     = 4;
}

message NeuralAttentionResponse {
    repeated NodeScore scores = 1;
    uint64 duration_ms = 2;
}

message VqvaeEncodeRequest {
    repeated float input      = 1;
    string model_name         = 2;
}

message VqvaeEncodeResponse {
    repeated uint32 codes     = 1;   // discrete codebook indices
    repeated float  quantized = 2;   // reconstructed continuous vector
    float reconstruction_loss = 3;
}

message VqvaeDecodeRequest {
    repeated uint32 codes     = 1;
    string model_name         = 2;
}

message VqvaeDecodeResponse {
    repeated float output     = 1;
}

message DsiRetrieveRequest {
    string collection     = 1;
    repeated double query = 2;
    uint32 k              = 3;
    string model_name     = 4;
}

message DsiRetrieveResponse {
    repeated string node_ids   = 1;
    repeated double scores     = 2;
    uint64 duration_ms         = 3;
}

message ModelInfo {
    string name       = 1;
    string type       = 2;   // "gnn" | "mcts_policy" | "mcts_value" | "vqvae" | "dsi" | "attention"
    string path       = 3;
    bool   loaded     = 4;
    uint64 size_bytes = 5;
}

message ModelListResponse {
    repeated ModelInfo models = 1;
}

message LoadModelRequest {
    string name       = 1;
    string model_path = 2;   // path to .onnx file
    string model_type = 3;
}
```

### 3.4 Go <--> Rust Interface Design

```
EVA Go                              NietzscheDB Rust
=========                           ================
internal/neural/                    nietzsche-neural/
  inference_client.go  --gRPC-->      ModelRegistry
  model_registry.go                   OnnxSession pool
  tensor_utils.go                     tensor_ops.rs

internal/cortex/mcts/               nietzsche-mcts/
  clinical_adapter.go  --gRPC-->      MctsTree
  policy.go            MctsSearch     NeuralPolicy
  value.go                            NeuralValue

internal/cortex/worldmodel/         nietzsche-dream/ (extended)
  dreamer.go           --gRPC-->      DreamerV3Engine (new)
  latent_state.go      GnnInfer       LatentDynamics
  imagination.go                      TrajectoryRollout

internal/cortex/attention/neural/   nietzsche-agency/ (extended)
  scorer.go            --gRPC-->      NeuralAttentionScorer
  gnn_context.go       NeuralAttention  GnnContextBuilder
```

**Key Design Principle**: Heavy neural inference runs in Rust (NietzscheDB) where ONNX runtime and GPU access live. Go (EVA) orchestrates clinical logic and user interaction, calling NietzscheDB for neural compute.

---

## 4. Implementation Timeline (Phased)

### Phase 1: Foundation (nietzsche-neural) -- MUST BE FIRST
**Duration**: 2-3 weeks
**Dependencies**: None (clean new crate)

| Task | Description | Effort |
|------|-----------|--------|
| Create `nietzsche-neural` crate | Workspace member, Cargo.toml with `ort` (ONNX runtime) | 1 day |
| ModelRegistry | Thread-safe registry: load/unload/list ONNX models | 3 days |
| OnnxSession pool | Connection pool for concurrent inference sessions | 2 days |
| Tensor ops | Rust ndarray <-> ONNX tensor conversion | 2 days |
| Poincare constraint layer | Post-inference projection: ensure all outputs satisfy `\|\|x\|\| < 1 - epsilon` | 1 day |
| Graceful fallback | `#[cfg(feature = "neural")]` gating everywhere; without feature = classic behavior | 2 days |
| Integration into server | Feature-gated model loading at startup, health check | 2 days |
| Tests + benchmarks | Unit tests, model loading benchmark | 2 days |

**Deliverable**: `cargo build --features neural` compiles and can load/run ONNX models.

### Phase 2: Core Neural (nietzsche-gnn + nietzsche-mcts) -- PARALLEL
**Duration**: 3-4 weeks (parallel tracks)
**Dependencies**: Phase 1 (nietzsche-neural)

**Track A: nietzsche-gnn**
| Task | Description | Effort |
|------|-----------|--------|
| Create crate | Depends on nietzsche-neural, nietzsche-pregel, nietzsche-graph | 1 day |
| Neighbor sampling | K-hop neighborhood extraction from adjacency lists | 3 days |
| GAT layer (ONNX) | Graph Attention Network inference via pre-trained ONNX model | 4 days |
| Message passing | Pregel-compatible message passing with neural aggregation | 3 days |
| Poincare-aware pooling | Aggregate GNN outputs in hyperbolic space (Frechet mean on Poincare ball) | 3 days |
| gRPC endpoint | `GnnInfer` RPC implementation in nietzsche-api | 2 days |
| Go client | `internal/neural/gnn_client.go` in EVA | 1 day |

**Track B: nietzsche-mcts**
| Task | Description | Effort |
|------|-----------|--------|
| Create crate | Depends on nietzsche-neural, nietzsche-agency | 1 day |
| MCTS core | UCT tree search with PUCT exploration | 4 days |
| Neural policy | Load policy network, output action probabilities | 3 days |
| Neural value | Load value network, evaluate states | 2 days |
| Graph state adapter | Map NietzscheDB graph neighborhoods to MCTS states | 3 days |
| Clinical guard rails | Ensure MCTS outputs are advisory-only (never autonomous) | 2 days |
| gRPC endpoint | `MctsSearch` RPC in nietzsche-api | 2 days |
| Go client | `internal/cortex/mcts/` in EVA | 2 days |

### Phase 3: Evolution (nietzsche-rl + nietzsche-vqvae) -- PARALLEL
**Duration**: 3-4 weeks (parallel tracks)
**Dependencies**: Phase 1 (nietzsche-neural)

**Track A: nietzsche-rl**
| Task | Description | Effort |
|------|-----------|--------|
| Create crate | Depends on nietzsche-neural, nietzsche-lsystem | 1 day |
| Reward function | Map Hausdorff dimension + energy distribution to reward signal | 3 days |
| PPO agent | Proximal Policy Optimization for L-System rule selection | 5 days |
| L-System integration | Replace static production rules with RL-selected rules | 3 days |
| Hyperbolic reward shaping | Bonus for maintaining Poincare structure (1.2 < D < 1.8) | 2 days |
| Training loop | Online RL with experience replay from sleep cycles | 3 days |

**Track B: nietzsche-vqvae**
| Task | Description | Effort |
|------|-----------|--------|
| Create crate | Depends on nietzsche-neural, nietzsche-pq, nietzsche-hyp-ops | 1 day |
| VQ-VAE encoder | ONNX model for encoding continuous embeddings to discrete codes | 3 days |
| Hyperbolic codebook | Codebook vectors live on Poincare ball; quantize via hyperbolic distance | 4 days |
| Commitment loss | Hyperbolic commitment loss: `d_hyp(z_e, sg[e])` | 2 days |
| Sensory integration | Replace current f32 latent in nietzsche-sensory with VQ codes | 3 days |
| gRPC endpoints | `VqvaeEncode`, `VqvaeDecode` | 2 days |
| Progressive degradation | `f32 -> f16 -> int8 -> pq -> vq -> gone` (current path enhanced) | 3 days |

### Phase 4: Advanced (DreamerV3, DSI, Neural Attention) -- SEQUENTIAL
**Duration**: 4-6 weeks
**Dependencies**: Phase 2 (GNN) + Phase 3 (VQ-VAE)

| Task | Description | Effort |
|------|-----------|--------|
| **DreamerV3 world model** | Extend nietzsche-dream with latent dynamics model | 2 weeks |
| -- Latent state encoder | GNN-enriched graph state -> latent vector | 3 days |
| -- Transition model | Predict next latent state (ONNX RSSM) | 4 days |
| -- Reward predictor | Predict reward from latent state | 2 days |
| -- Dream rollout | Generate imagined trajectories for planning | 3 days |
| **nietzsche-dsi** | Differentiable Search Index | 1.5 weeks |
| -- Document encoder | Map nodes to DSI-compatible representations | 3 days |
| -- Autoregressive decoder | Generate doc IDs from query embedding | 4 days |
| -- Integration with KNN | Fallback: DSI miss -> standard HNSW KNN | 3 days |
| **Neural Attention** | Extend nietzsche-agency attention scoring | 1 week |
| -- Attention scorer | Replace heuristic energy/recency with neural scorer | 3 days |
| -- Context encoding | GNN-enriched context for attention computation | 2 days |
| -- gRPC endpoint | `NeuralAttention` RPC | 1 day |

### Phase 5: Full Integration
**Duration**: 2-3 weeks
**Dependencies**: All previous phases

| Task | Description | Effort |
|------|-----------|--------|
| Neural Thresholds | Agency uses neural confidence to decide sleep/dream/grow triggers | 3 days |
| EVA Go full integration | Wire all neural Go clients into cortex pipeline | 4 days |
| End-to-end testing | Clinical scenario testing with neural components | 3 days |
| Performance tuning | Model quantization, batch inference, GPU memory management | 3 days |
| Dashboard neural panel | Show model status, inference latency, neural health in HTTP dashboard | 2 days |
| Documentation | API docs, deployment guide, model training guide | 2 days |

### Timeline Summary

```
Week  1-3:   Phase 1 (nietzsche-neural foundation)
Week  3-7:   Phase 2 (GNN + MCTS in parallel)
Week  5-9:   Phase 3 (RL + VQ-VAE in parallel, overlaps Phase 2)
Week  8-14:  Phase 4 (DreamerV3, DSI, Neural Attention)
Week 13-16:  Phase 5 (Integration, testing, tuning)

Total: ~16 weeks (4 months) for full neural stack
```

### Phase Dependencies (DAG)

```
Phase 1 (neural) --------+-----> Phase 2A (gnn) ------+
                          |                            |
                          +-----> Phase 2B (mcts) -----+-----> Phase 4 (advanced)
                          |                            |              |
                          +-----> Phase 3A (rl)   -----+              v
                          |                            |       Phase 5 (integration)
                          +-----> Phase 3B (vqvae) ----+
```

---

## 5. Configuration & Deployment

### 5.1 Feature Flag Matrix

| Deployment Target | Features | Description |
|---|---|---|
| **Dev local (Windows/Mac)** | `default` | No GPU, no neural. Classic NietzscheDB. |
| **Dev with neural** | `neural` | ONNX CPU inference. For testing neural components. |
| **Staging (Cloud VM)** | `neural,gpu` | Full neural + GPU vector search. |
| **Production (Cloud TPU)** | `neural,tpu` | Full neural + TPU vector search. |
| **Edge / Lightweight** | `neural-gnn` | Only GNN, no MCTS/RL/VQ-VAE. |

### 5.2 Model File Management

```
/data/nietzsche/
    models/                          # ONNX model directory
        registry.json                # Model metadata registry
        gnn/
            gat_v1.onnx              # Graph Attention Network
            gat_v1.json              # Model config (dims, layers, etc.)
        mcts/
            policy_v1.onnx           # MCTS policy network
            value_v1.onnx            # MCTS value network
        vqvae/
            encoder_v1.onnx          # VQ-VAE encoder
            decoder_v1.onnx          # VQ-VAE decoder
            codebook_v1.bin          # Discrete codebook (Poincare ball)
        dsi/
            index_v1.onnx            # Differentiable Search Index
        attention/
            scorer_v1.onnx           # Neural attention scorer
```

**Environment variables for neural config:**
```bash
# Model directory (all .onnx files)
NIETZSCHE_MODEL_DIR=/data/nietzsche/models

# Enable neural inference (master switch, separate from compile-time feature)
NIETZSCHE_NEURAL_ENABLED=true

# Per-component enable/disable at runtime
NIETZSCHE_GNN_ENABLED=true
NIETZSCHE_MCTS_ENABLED=true
NIETZSCHE_RL_ENABLED=false     # disable RL in production initially
NIETZSCHE_VQVAE_ENABLED=true
NIETZSCHE_DSI_ENABLED=false    # experimental

# Inference settings
NIETZSCHE_NEURAL_THREADS=4            # ONNX intra-op threads
NIETZSCHE_NEURAL_BATCH_SIZE=32        # Max batch for batched inference
NIETZSCHE_NEURAL_TIMEOUT_MS=5000      # Inference timeout per request
NIETZSCHE_NEURAL_GPU_DEVICE=0         # GPU device for ONNX (if available)
```

### 5.3 GPU/TPU Deployment Considerations

| Resource | Requirement | Notes |
|---|---|---|
| **GPU (CUDA)** | NVIDIA GPU + CUDA 12.x + cuVS 24.6 | For nietzsche-hnsw-gpu (CAGRA) + ONNX GPU provider |
| **GPU Memory** | ~2-4GB for all neural models | Models are small; bulk memory is vector data |
| **TPU** | Cloud TPU v5e+ + libtpu.so | For nietzsche-tpu; ONNX runs on CPU alongside |
| **CPU-only** | No special requirements | ONNX CPU provider works everywhere, ~3-5x slower |
| **ONNX Runtime** | v1.16+ dynamically linked | Bundled via `ort` Rust crate |

### 5.4 Backward Compatibility

**Critical invariant**: Building without `--features neural` produces EXACTLY the same binary as today.

```rust
// Pattern used throughout codebase:
#[cfg(feature = "neural")]
use nietzsche_neural::ModelRegistry;

pub fn process_node(&self, node: &Node) -> Result<()> {
    // Classic behavior always runs
    let result = self.classic_process(node)?;

    // Neural enhancement only if feature compiled AND runtime enabled
    #[cfg(feature = "neural")]
    if self.neural_enabled() {
        if let Ok(enhanced) = self.neural_enhance(node) {
            // Neural result augments, never replaces, classic result
            result.merge(enhanced);
        }
        // If neural fails, classic result is used (graceful degradation)
    }

    Ok(result)
}
```

### 5.5 CI/CD

```yaml
# GitHub Actions matrix
jobs:
  build:
    strategy:
      matrix:
        features:
          - ""                    # classic build
          - "neural"              # neural CPU
          - "neural,gpu"          # neural + GPU (needs CUDA runner)
          - "neural-gnn"          # only GNN

  test-neural:
    # Uses pre-trained tiny test models (< 1MB each)
    # Stored in tests/fixtures/models/
    # Tests verify:
    #   1. Model loads without error
    #   2. Inference produces output with correct shape
    #   3. All outputs satisfy ||x|| < 1-epsilon (Poincare constraint)
    #   4. Graceful fallback when model file missing
    #   5. Classic behavior unchanged when neural disabled

  benchmark-neural:
    # Nightly benchmark on GPU runner
    # Measures: inference latency p50/p99, memory usage, throughput
    # Regression alerts if p99 > 2x baseline
```

---

## 6. INTOCAVEIS (Sacred Constraints)

### Permanently Rejected

| Constraint | Reason | Reference |
|---|---|---|
| **Gemini model `gemini-2.5-flash-native-audio-preview-12-2025`** | INTOCAVEL. Used for EVA-Mind native audio (real-time voice). Located in `EVA-Mind/.env` as `MODEL_ID`. ANY change breaks the voice system. | CLAUDE.md |
| **Binary Quantization as HNSW metric** | PERMANENTLY REJECTED. `sign(x)` destroys magnitude = hierarchy in Poincare ball. Unanimous decision (Claude + Grok + Committee, 2026-02-19). Only permitted as pre-filter with oversampling >=30x and mandatory rescore. | `risco_hiperbolico.md` Part 4, CLAUDE.md |
| **Matryoshka Embeddings (dimension truncation)** | REJECTED. Truncating Poincare coordinates changes `\|\|x\|\|`, destroying hierarchical depth. Same mechanism of destruction as Binary Quantization. | `risco_hiperbolico.md` Part 10.1 |
| **Differential Privacy with Euclidean noise** | REJECTED. Gaussian noise `N(0,sigma)` added to Poincare coords can push `\|\|x\|\| >= 1.0` (outside the ball = undefined). Must use Riemannian noise via tangent space. | `risco_hiperbolico.md` Part 10.2 |
| **Serverless / Scale-to-zero** | REJECTED. L-System, Sleep cycle, and Zaratustra are continuous autonomous processes. Scale-to-zero kills the "consciousness" of the database. | `risco_hiperbolico.md` Part 10.3 |

### Mandatory Constraints for All Neural Components

| Constraint | Enforcement | Implementation |
|---|---|---|
| **Poincare ball constraint**: ALL neural outputs must satisfy `\|\|x\|\| < 1 - epsilon` | Post-inference projection layer in `nietzsche-neural` | `project_to_ball(x, epsilon=0.001)` applied after every ONNX inference that produces embeddings |
| **Classic fallback**: EVERY neural component must degrade gracefully | `#[cfg(feature = "neural")]` compile-time + runtime enable flag | If neural fails or is disabled, classic heuristic behavior runs instead |
| **Clinical safety**: Neural components are ADVISORY, never autonomous for critical decisions | MCTS outputs are ranked suggestions, not commands. Agency uses neural scores as one input among many. | Clinical guard rail in `nietzsche-mcts`: output includes confidence + alternatives |
| **Gradients in f64**: Riemannian Adam optimizer always uses f64 for gradient computation | Sleep cycle already uses f64. Neural extensions must follow. | Convert f32 neural outputs to f64 before gradient computation, back to f32 for storage |
| **Soft clamping**: `\|\|x\|\| > 0.999 -> 0.999` | Applied in `project_into_ball()` in model.rs | Prevents underflow in denominator `(1 - \|\|x\|\|^2)` |
| **Hyperbolic Invariant Test Suite**: Nightly CI checks neural outputs | New CI job testing recall@10, drift, % nodes > 0.995, distance error | Run after every neural model update |

### The Unifying Principle

> **In Poincare ball geometry, magnitude `||x||` encodes hierarchical information (depth). ANY operation that destroys, truncates, or corrupts this magnitude destroys the fundamental reason for using hyperbolic geometry.**

```
Center of ball  (||x|| ~ 0)     --> abstract/general concepts (Semantic nodes)
Border of ball  (||x|| ~ 0.999) --> specific/detailed memories (Episodic nodes)
```

Every neural component MUST preserve this property. This is not optional. This is the entire point of NietzscheDB.

---

## Appendix A: Crate Creation Checklist

For each new crate (`nietzsche-neural`, `nietzsche-gnn`, etc.):

1. Create directory: `crates/nietzsche-{name}/`
2. Create `Cargo.toml` with `workspace = true` dependencies
3. Add to workspace `Cargo.toml` members
4. Add to `nietzsche-server/Cargo.toml` as optional dependency
5. Gate all usage with `#[cfg(feature = "neural-{name}")]`
6. Add runtime enable/disable via `NIETZSCHE_{NAME}_ENABLED` env var
7. Implement `trait NeuralFallback` with `fn classic_fallback(&self) -> Result<T>`
8. Add to CI build matrix
9. Create `tests/fixtures/models/{name}_test.onnx` (tiny model for CI)
10. Add to dashboard model status panel

## Appendix B: Key File Paths

| File | Purpose |
|---|---|
| `D:\DEV\NietzscheDB\Cargo.toml` | Workspace root (41 crates) |
| `D:\DEV\NietzscheDB\crates\nietzsche-server\src\main.rs` | Server entry point (~804 lines) |
| `D:\DEV\NietzscheDB\crates\nietzsche-server\src\config.rs` | All NIETZSCHE_* env vars |
| `D:\DEV\NietzscheDB\crates\nietzsche-api\Cargo.toml` | API aggregator (depends on 14 crates) |
| `D:\DEV\NietzscheDB\crates\nietzsche-graph\Cargo.toml` | Core graph engine (THE HUB) |
| `D:\DEV\NietzscheDB\sdks\go\proto\nietzsche.proto` | gRPC API definition (1022 lines) |
| `D:\DEV\NietzscheDB\risco_hiperbolico.md` | Hyperbolic risk analysis (649 lines) |
| `D:\DEV\NietzscheDB\md\FASES.md` | Phase success criteria |
| `D:\DEV\EVA\go.mod` | EVA Go module (references nietzsche-sdk) |
| `D:\DEV\EVA\.env` | EVA configuration (NIETZSCHE_GRPC_ADDR, MODEL_ID, etc.) |
| `D:\DEV\EVA\internal\brainstem\infrastructure\nietzsche\client.go` | NietzscheDB gRPC client |
| `D:\DEV\EVA\internal\memory\krylov\krylov_manager.go` | Krylov subspace compression |
| `D:\DEV\EVA\internal\cortex\voice\speaker\embedder.go` | ONNX runtime usage in Go |
| `D:\DEV\NietzscheDB\docs\book\src\architecture.md` | NietzscheDB architecture guide |

---

*Document prepared by Agent 10 (Architecture Integration) -- 2026-02-23*
*NietzscheDB v0.1.0 | EVA v2.0 Blueprint | Claude Opus 4.6*
