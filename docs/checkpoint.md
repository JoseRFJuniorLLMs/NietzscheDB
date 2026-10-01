# CHECKPOINT - Nietzsche-Database
**Data:** 2026-10-01
**Status:** release 3.2.0 em hardening - engine core funcional, multi-manifold implementado, EVA em produção

**Baseline 3.2.0:** branch `release/3.2.0`, derivada do `main` em `087a301c4fbafa227b00a0411d99c86a29065151`.

---

## O QUE E O PROJETO
Banco de dados grafico **multi-manifold** em Rust, fork do NietzscheDB com graph engine nativo (NietzscheDB). Opera em 4 geometrias nao-euclidianas simultaneamente: **Poincare** (storage/HNSW), **Klein** (pathfinding), **Riemann** (sintese), **Minkowski** (causalidade). HNSW para busca vetorial, RocksDB para storage, NQL (Nietzsche Query Language) como linguagem propria. Substituiu NietzscheDB + NietzscheDB + NietzscheDB na EVA.

**Tech Stack:** Rust nightly, Tokio, Tonic (gRPC), RocksDB, Axum (HTTP dashboard), Pest (PEG parser), DashMap, rayon, memmap2, React 19 (dashboard SPA), D3.js (dashboard embedded), Docker Compose

---

## O QUE FUNCIONA
### Core Graph Engine (nietzsche-graph)
- PoincareVector com distance(), project_into_ball()
- Node model: UUID, embedding, energy (0-1), depth, Hausdorff dimension, NodeTypes
- Edge model: directed, typed, weighted
- RocksDB storage: 7 column families (nodes, edges, adj_out, adj_in, meta, sensory, energy_idx)
- Energy secondary index para range scans O(log N)
- Write-Ahead Log (binary append-only, crash recovery)
- AdjacencyIndex: DashMap lock-free bidirectional
- CollectionManager: multi-collection namespace isolado
- ACID Saga transactions (Phase 7): buffered ops, WAL, idempotent replay
- EmbeddedVectorStore: HNSW real via nietzsche-hnsw, Cosine + Poincare + Euclidean

### Graph Traversal
- BFS, Dijkstra, Diffusion walk, Shortest path (com filtros energy_min, max_depth)

### NQL (Nietzsche Query Language)
- PEG grammar (265 linhas) com parser Pest
- MATCH com node/path patterns, WHERE, RETURN (ORDER BY, LIMIT, SKIP, DISTINCT)
- Path patterns single-hop: `(a)-[:TYPE]->(b)`
- HYPERBOLIC_DIST e SENSORY_DIST functions
- DIFFUSE FROM, RECONSTRUCT, EXPLAIN, INVOKE ZARATUSTRA
- BEGIN/COMMIT/ROLLBACK transactions
- Aggregations: COUNT, AVG, MIN, MAX, SUM com GROUP BY
- Parallel scan com rayon (threshold 2000 nodes)

### gRPC API (nietzsche-api)
- Collection CRUD, Node CRUD, Edge CRUD, NQL Query
- KnnSearch, Bfs, Dijkstra, Diffuse
- TriggerSleep, InvokeZaratustra
- InsertSensory, GetSensory, Reconstruct, DegradeSensory
- GetStats, HealthCheck, gRPC reflection

### HTTP Dashboard (nietzsche-server)
- REST API via Axum: stats, health, collections, graph, node CRUD, edge CRUD, NQL query, sleep
- D3.js dashboard embeddido

### Sleep Cycle (Phase 8)
- Riemannian Adam optimizer no disco de Poincare
- 7-step reconsolidation protocol
- Embedding checkpoint/rollback baseado em Hausdorff threshold

### Zaratustra Engine (Phase Z)
- Will to Power: energy propagation (graph heat diffusion)
- Eternal Recurrence: temporal echo ring-buffer
- Ubermensch: top-N% elite-tier identification

### L-System Engine (Phase 5)
- Production rules, Mobius addition, Hausdorff dimension (box-counting)

### Diffusion Engine (Phase 6)
- Hyperbolic Laplacian, Chebyshev polynomial approximation

### Sensory Compression (Phase 11)
- Degradacao progressiva baseada em energy: F32 -> F16 -> Int8 -> PQ 64B -> discard
- Modalidades: Text, Audio, Image, Fused

### React Dashboard
- Auth, Overview, Collections, Nodes, DataExplorer (NQL console), GraphExplorer (cosmograph), Settings

### NietzscheDB (fork base)
- HNSW hiperbolico, WAL v3, mmap storage
- Scalar I8 e Binary quantization
- Cluster federation, multi-tenancy
- LangChain Python/JS integrations
- Python, TypeScript SDKs

---

## O QUE FALTA
### Engine
1. ~~KNN com metadata filters pushed-down~~ ✅ FEITO
2. ~~TTL/expires_at em Node~~ ✅ FEITO
3. ~~ListStore~~ ✅ FEITO
4. ~~MergeNode/MergeEdge~~ ✅ FEITO
5. ~~Secondary indexes~~ ✅ FEITO
6. **Cluster nao wired** - nietzsche-cluster implementado mas nao conectado ao server
7. ~~Table Store (SQLite)~~ ✅ FEITO
8. ~~Media Store (OpenDAL)~~ ✅ FEITO

### NQL
1. ~~MERGE statement~~ ✅ FEITO
2. ~~Multi-hop paths~~ ✅ FEITO
3. ~~SET statement~~ ✅ FEITO
4. ~~CREATE statement~~ ✅ FEITO
5. ~~DELETE / DETACH DELETE~~ ✅ FEITO
6. **WITH clause** (pipeline) - pendente
7. ~~Time functions~~ ✅ FEITO

### SDK
- ~~Go SDK~~ ✅ FEITO (48 RPCs, inclui 6 multi-manifold)

### Multi-Manifold (2026-02-22) ✅ COMPLETO
- Klein model (pathfinding com geodesicas retas)
- Riemann sphere (sintese dialetica, Frechet mean)
- Minkowski spacetime (classificacao causal, light cone filter)
- Manifold normalization (health checks, safe roundtrips)
- Edge causality metadata (CausalType, minkowski_interval)
- 6 novos gRPC RPCs + Go SDK methods

---

## BUGS / DIVIDAS REVALIDADOS PARA 3.2.0
1. ~~**EmbeddedVectorStore usa CosineMetric para todas as metricas**~~ **RESOLVIDO** - o codigo atual possui wrappers separados para Cosine, Euclidean e Poincare; `HnswPoincareWrapper` usa `PoincareMetric` nativamente.
2. ~~**Dashboard API mismatch**~~ **RESOLVIDO** - `nietzsche-server` expoe `/api/status` como alias de `/api/stats` e implementa `/api/cluster/status`.
3. ~~**Reconstruct RPC hardcoda collection "default"**~~ **RESOLVIDO** - `ReconstructRequest` carrega `collection = 3` e o handler resolve a collection solicitada.
4. ~~**Dockerfile / Docker Compose divergiam em 50051/50052**~~ **RESOLVIDO NA 3.2.0** - container escuta `50051`; host Docker Compose publica `50052:50051`.
5. **Documentacao historica obsoleta** - auditorias e roadmaps antigos continuam preservados como registro historico; nao devem ser usados como fonte de verdade do estado atual sem confronto com `main`/release.

---

## DEPENDENCIAS PRINCIPAIS
### Rust
tokio 1.35, tonic 0.10, prost 0.12, serde/serde_json 1.0, rocksdb 0.21, dashmap 5.5, pest 2, memmap2 0.9, rayon 1.10, axum 0.7, uuid 1.6, bincode 1.3, tracing 0.1

### Dashboard
react 19.2, react-router-dom 7.13, @cosmograph/cosmograph 2.1, @tanstack/react-query 5.90, recharts 3.7, tailwindcss 4.1, vite 7.2

### Build
Rust nightly, protoc (vendored), libclang-dev, clang, cmake, libssl-dev

---

## DEAD CODE
- md/roadmap2.md (duplicata exata de roadmap.md)
- md/ARCHITECTURE.md (copia de docs/ARCHITECTURE.md)
- md/audit_resultado.md (audit de OUTRO projeto - EVA-Mobile)
- md/feito.md (lista features NietzscheDB, nao NietzscheDB)
- md/TODO_ADOPTION.md (todos items completos, NietzscheDB only)
- md/faze11.md / faze11x.md (superseded por fazer.md)
- ~~sdks/go/~~ → IMPLEMENTADO (48 RPCs + multi-manifold)
- sdks/cpp/ (so README, sem implementacao)
- nietzsche-cluster (implementado mas nao wired no server)

---

## .md PARA DELETAR
- md/roadmap2.md (duplicata)
- md/ARCHITECTURE.md (copia)
- md/audit_resultado.md (projeto errado)
- md/TODO_ADOPTION.md (obsoleto, tudo completo)
- md/faze11.md (superseded)
- md/faze11x.md (superseded)
- md/pesquisar1.md (scratch notes)
- md/pesquisas.md (scratch notes)
- md/hiperbolica.md (notas exploratoria)
- md/hiperbolica2.md (notas exploratoria)
