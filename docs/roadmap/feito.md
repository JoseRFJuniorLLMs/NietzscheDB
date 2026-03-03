# NietzscheDB — O que esta FEITO (verificado no codigo 2026-02-22)

> Cada item abaixo foi verificado lendo o codigo Rust real, nao documentacao.

---

## Engine Core (`nietzsche-graph`)

- [x] `NietzscheDB<V: VectorStore>` com dual-write (RocksDB + HNSW) — `db.rs`
- [x] 12 Column Families no RocksDB (CF_NODES, CF_EMBEDDINGS, CF_EDGES, CF_ADJ_OUT, CF_ADJ_IN, CF_META, CF_SENSORY, CF_ENERGY_IDX, CF_META_IDX, CF_LISTS, CF_SQL_SCHEMA, CF_SQL_DATA)
- [x] NodeMeta separado de embedding (100B vs 24KB) — 240x reducao RAM no hot-tier
- [x] PoincareVector com f32 (50% reducao vs f64), math interna em f64
- [x] WAL com CRC32 + crash recovery + 14 tipos de entry
- [x] Transacoes Saga (BEGIN/COMMIT/ROLLBACK) com replay idempotente
- [x] AdjacencyIndex com DashMap lock-free, rebuild on startup
- [x] Migracao V0/V1/V1.5/V2 com parser manual byte-level para V0
- [x] 45+ metodos em db.rs — zero todo!(), zero unimplemented!()

## VectorStore

- [x] EmbeddedVectorStore com metricas: Cosine, Euclidean, Poincare, DotProduct
- [x] GPU optional (cuVS CAGRA, feature-gated)
- [x] TPU optional (PJRT MHLO, feature-gated)
- [x] KNN filtrado push-down (knn_filtered, knn_energy_filtered dual-path)
- [x] Hybrid search (BM25 + KNN via RRF)
- [x] Full-text search (BM25 inverted index)

## Multi-Colecao

- [x] CollectionManager com HNSW + RocksDB independente por colecao
- [x] RPCs: CreateCollection, DropCollection, ListCollections
- [x] RwLock por colecao (reads concorrentes, writes isolados)

## TTL / Cache / Lists (substitui NietzscheDB)

- [x] expires_at em NodeMeta + TTL Reaper background (60s default)
- [x] CacheSet/CacheGet/CacheDel RPCs com TTL
- [x] ListStore: CF_LISTS com RPUSH/LRANGE/LEN RPCs
- [x] ReapExpired RPC para limpeza manual

## Merge (substitui NietzscheDB MERGE)

- [x] MergeNode RPC — get-or-create com on_create_set / on_match_set
- [x] MergeEdge RPC — upsert de aresta com metadata atomico
- [x] IncrementEdgeMeta RPC — `r.count = r.count + 1` atomico

## Indices Secundarios

- [x] CF_META_IDX com field_hash + sortable_value + node_id
- [x] index_scan_eq() e index_scan_range() em db.rs
- [x] RPCs: CreateIndex, DropIndex, ListIndexes
- [x] CF_ENERGY_IDX para range queries em energy

## NQL Query Language

- [x] Grammar PEG: 265 regras
- [x] MATCH com WHERE complexo (AND/OR/NOT, IN, BETWEEN, CONTAINS, STARTS_WITH, ENDS_WITH)
- [x] MATCH path traversal *min..max com label/edge filter
- [x] CREATE node com props + TTL
- [x] MERGE node/edge com ON CREATE SET / ON MATCH SET
- [x] SET com aritmetica (n.field + 1, n.field - 1)
- [x] DELETE / DETACH DELETE
- [x] DIFFUSE FROM $node WITH t=[...] MAX_HOPS n
- [x] RECONSTRUCT $node MODALITY audio QUALITY high
- [x] EXPLAIN <query> com estimativa de custo
- [x] BEGIN / COMMIT / ROLLBACK
- [x] CREATE/DROP/SHOW DAEMON
- [x] DREAM FROM / APPLY DREAM / REJECT DREAM / SHOW DREAMS
- [x] TRANSLATE (synesthesia)
- [x] COUNTERFACTUAL SET ... MATCH ...
- [x] NARRATE IN "col" WINDOW 24
- [x] SHOW/SHARE ARCHETYPES
- [x] INVOKE ZARATUSTRA
- [x] PSYCHOANALYZE $node
- [x] 13 funcoes matematicas (POINCARE_DIST, RIEMANN_CURVATURE, HAUSDORFF_DIM, etc.)
- [x] Agregacoes: COUNT, SUM, AVG, MIN, MAX + GROUP BY + ORDER BY + DISTINCT + SKIP/LIMIT
- [x] Tempo: NOW(), EPOCH_MS(), INTERVAL("1h"/"1d"/"1w")

## Servidor gRPC

- [x] 117 RPCs — TODOS implementados com logica real
- [x] 6 background schedulers (Sleep, Zaratustra, TTL Reaper, Backup, Daemons, Agency)
- [x] RBAC (Admin/Writer roles)
- [x] CDC streaming (SubscribeCDC)
- [x] Backpressure signaling
- [x] Dashboard porta 8080

## 4 Geometrias Simultaneas (`nietzsche-hyp-ops`)

- [x] Poincare Ball: exp_map, log_map, mobius_add, distance, gyromidpoint, parallel_transport
- [x] Klein: to_klein/to_poincare, is_collinear, is_on_shortest_path
- [x] Riemann Sphere: synthesis (2-point + N-ary), frechet_mean, spherical_distance
- [x] Minkowski: interval, classify (Timelike/Spacelike/Lightlike), light_cone_filter
- [x] Manifold health: normalize, sanitize, safe roundtrips

## Sleep Cycle (`nietzsche-sleep`)

- [x] Riemannian Adam optimizer real (exp_map, bias correction, retraction)
- [x] 7-step reconsolidation (scan, hausdorff, snapshot, perturb, adam loop, re-scan, commit/rollback)
- [x] SnapshotRegistry (create/restore/list/delete)
- [x] Episodic-to-Semantic fusion (Union-Find clustering, centroid Poincare, phantomization)

## L-System (`nietzsche-lsystem`)

- [x] Production rules com algebra booleana (And/Or/Not, 8 tipos de condicao)
- [x] Hausdorff box-counting (multi-scale OLS regression)
- [x] Engine tick 7-step com sensory degradation hook
- [x] Mobius spawning (child radial, sibling rotacional)
- [x] Circuit breaker (depth caps, tumor detection BFS, energy dampening)

## Heat Diffusion (`nietzsche-pregel`)

- [x] Modified Bessel I_k(t) (power series ate 80 termos)
- [x] Chebyshev polynomial recurrence (3-term, sem multiplicacao de matrizes)
- [x] Multi-scale diffusion engine

## Zaratustra (`nietzsche-zaratustra`)

- [x] Will to Power: propagacao de energia com circuit breaker depth-aware
- [x] Eternal Recurrence: ring buffer FIFO de snapshots energeticos
- [x] Ubermensch: selecao elite top-N% com threshold + mean comparison

## DAEMON Agents (`nietzsche-wiederkehr`)

- [x] Engine tick loop (scan, evaluate WHEN, generate intents)
- [x] Condition evaluator completo (comparisons, AND/OR/NOT, string ops, math funcs)
- [x] Persistent store em CF_META prefix "daemon:"
- [x] Priority scheduler (BinaryHeap energy*priority weighted)
- [x] Energy decay + reap automatico

## Dream Queries (`nietzsche-dream`)

- [x] BFS speculative exploration com noise
- [x] Energy spike + curvature anomaly detection
- [x] Session management (Pending/Applied/Rejected)
- [x] Apply (persiste deltas) / Reject (descarta)

## AGI Inference Layer (`nietzsche-agi`) — 6,994 LOC, 123 testes

### Layer 1-2: Representacao e Navegacao Verificavel
- [x] `SynthesisNode` — wrapper AGI com metadata de inferencia — `representation.rs`
- [x] `Rationale` com `InferenceType` (6 tipos), `fidelity`, `energy_seal`, `certification` — `rationale.rs`
- [x] `GeodesicTrajectory` com validacao GCS (collinearidade Klein + gradiente radial Poincare) — `trajectory.rs`
- [x] `GeodesicCoherenceScore` — media harmonica por-hop, range [0, 1] — `trajectory.rs`

### Layer 3: Inferencia Explicita
- [x] `InferenceEngine` — motor de regras: trajectory → Generalization/Specialization/Bridge/Analogy/Dialectic/Rupture — `inference_engine.rs`
- [x] `FrechetSynthesizer` — sintese dialetica via media de Frechet na esfera de Riemann — `synthesis.rs`
- [x] `DialecticDetector` — detecao de pares de tensao cross-cluster — `dialectic.rs`

### Layer 4: Atualizacao Dinamica
- [x] `FeedbackLoop` com `simulate()` (off-graph) + `prepare()` (re-insercao) — `feedback_loop.rs`
- [x] `HomeostasisGuard` + `RadialField` — repulsao suave quadratica perto da origem — `homeostasis.rs`
- [x] `RelevanceDecay` — decay baseado em frequencia + boost por acesso — `relevance_decay.rs`
- [x] `EvolutionScheduler` — heartbeat autonomo coordenando todos sub-sistemas — `evolution.rs`

### Layer 5: Motor de Estabilidade (Phase V)
- [x] `StabilityEvaluator` com E(τ) = w₁·H_GCS + w₂·θ_klein + w₃·causal + w₄·entropy — `stability.rs`
- [x] `CertificationLevel` 4-tier: StableInference(≥0.75) / WeakBridge(≥0.50) / MetaphoricDrift(≥0.25) / LogicalRupture(<0.25) — `certification.rs`
- [x] `CertificationSeal` — selo imutavel de qualidade epistemologica — `certification.rs`
- [x] `SpectralMonitor` — λ₂ Fiedler eigenvalue via Jacobi + power iteration — `spectral.rs`
- [x] `ConnectivityClass` — Rigid/Stable/Fragile/Disconnected — `spectral.rs`

### Layer 6: Equilibrio Metabolico (Phase VI)
- [x] `DiscoveryField` com D(τ) = w_g·|∇E| + w_c·θ_cluster — friccao produtiva — `discovery.rs`
- [x] `InnovationEvaluator` com Φ(τ) = αS + βD - γR (α=0.50, β=0.35, γ=0.60) — `innovation.rs`
- [x] `AcceptanceDecision` — Accept (Φ≥0.50) / Sandbox (0.20≤Φ<0.50) / Reject (Φ<0.20) — `innovation.rs`
- [x] `SandboxEvaluator` — quarentena com peso 0.3x, decay 3x, monitorizacao Δλ₂ — `sandbox.rs`
- [x] `DriftTracker` — tracking de evolucao λ₂ para Modelo Evolutivo (drift controlado) — `spectral.rs`

### Exemplo
- [x] `pipeline_ignition.rs` — demo completa do pipeline AGI (detect → traverse → validate → infer → synthesize)

## Agency Engine (`nietzsche-agency`) — 5,734 LOC

- [x] Event Bus (tokio broadcast, 7 event types)
- [x] 4 Daemons: Entropy, Coherence, Gap, NiilistaGc
- [x] Counterfactual Engine (shadow graph, what_if_remove/add)
- [x] Dialectic Synthesis (detect polarities -> tension -> synthesis)
- [x] Motor de Desejo (knowledge gaps -> DesireSignal -> trigger dreams)
- [x] Reactor (events -> intents com cooldown)
- [x] Observer Identity (meta-node auto-consciencia)

## Narrative Engine (`nietzsche-narrative`)

- [x] Graph analysis em time window
- [x] Event detection: EliteEmergence, Decay (2/6 tipos)
- [x] Summary generation human-readable

## MCP Server (`nietzsche-mcp`)

- [x] 9 tools: query, insert_node, get_node, delete_node, insert_edge, get_stats, knn_search, diffuse, run_algorithm
- [x] PageRank + Louvain wired

## Sensory Compression (`nietzsche-sensory`)

- [x] Progressive degradation: f32 -> f16 -> int8 -> PQ (64B) -> Gone
- [x] Encoders: Text (Krylov 64D), Audio (EnCodec 128D), Fusion (gyromidpoint)
- [x] Quantization roundtrips testados

## Table Store (`nietzsche-table`)

- [x] SQLite embutido via Swartz engine
- [x] CF_SQL_SCHEMA + CF_SQL_DATA
- [x] RPCs: SqlQuery (SELECT), SqlExec (CREATE TABLE, INSERT, UPDATE, DELETE, DROP TABLE, ALTER)

## Media Store (`nietzsche-media`)

- [x] OpenDAL backend (fs/s3/gcs)
- [x] CRUD de blobs associados a nodes

## Go SDK

- [x] 48 RPCs completos (nodes, edges, query, search, batch, merge, traversal, algo, manifold, lifecycle, sensory, cache, sql, schema, collections, cdc, backup, admin)

## Algoritmos de Grafo (`nietzsche-algo`)

- [x] PageRank, Louvain, LabelProp, Betweenness, Closeness, DegreeCentrality, WCC, SCC, AStar, TriangleCount, JaccardSimilarity (11 total)

## Outros

- [x] Kafka Connect sink (nietzsche-kafka)
- [x] Filtered KNN com Roaring Bitmaps (nietzsche-filtered-knn)
- [x] Named Vectors / Multi-Vector (nietzsche-named-vectors)
- [x] Product Quantization magnitude-preserving (nietzsche-pq)
- [x] Prometheus/OpenTelemetry metrics (nietzsche-metrics)
- [x] WASM + IndexedDB (nietzsche-wasm)
- [x] TUI ratatui + stress tests (nietzsche-cli)
- [x] ONNX + remote embedder (nietzsche-embed)

## Testes

- [x] ~1080 testes unitarios/integracao (957 engine + 123 AGI)
- [x] 18/18 modulos com cobertura (100%) — inclui nietzsche-agi
- [x] 6 benchmarks Criterion
- [x] CI/CD GitHub Actions

---

## O QUE NAO ESTA FEITO

### NQL Syntax Pendente

| Item | Impacto | Observacao |
|------|---------|------------|
| NQL WITH clause | MEDIO | Pipeline de queries |
| NQL FOREACH | BAIXO | Batch operations |
| NQL IN COLLECTION syntax | MEDIO | Existe no gRPC, nao no NQL |
| NQL CREATE INDEX syntax | BAIXO | Existe via RPC |
| NQL edge property access (r.field) | ALTO | Bloqueador NietzscheDB queries |
| SET/DELETE em path patterns | MEDIO | So funciona em node patterns |

### 5 Features Arquiteturais Faltantes (verificado no codigo 2026-02-22)

| # | Feature | Status | Impacto | Evidencia no Codigo |
|---|---------|--------|---------|---------------------|
| 1 | **Raft/Paxos Consensus** | ❌ | MEDIO | `nietzsche-cluster/src/lib.rs:13` → "No Raft in this phase". Apenas gossip basico |
| 2 | **Semantic Drift Validation** | ✅ | ALTO | `nietzsche-agi/src/spectral.rs` → DriftTracker (λ₂ evolution) + StabilityEvaluator (E(τ)) |
| 3 | **Manifold Visualizer** | 🟡 | BAIXO | `dashboard/GraphExplorerPage.tsx` → Cosmograph Euclidiano, sem Poincare disk |
| 4 | **HNSW Distributed Sharding** | ❌ | BAIXO | `nietzsche-graph/vector_store.rs` → Collection-level apenas, sem cross-shard KNN |
| 5 | **DX Integrations** | ❌ | CRITICO | `sdks/python/` → NietzscheDB (nome antigo), sem LangChain/DSPy/OGM |

### Outros Pendentes

| Item | Impacto | Observacao |
|------|---------|------------|
| Narrative 4/6 event types | BAIXO | 2 tipos funcionam |
| ImageEncoder | BAIXO | Voice-first |
| C++ SDK | BAIXO | Sem demanda |
| Migracao EVA-Mind PG | ALTO | 98 arquivos Go + 65 tabelas Python |
| Managed Cloud / DBaaS | CRITICO | Para adocao externa |
