# CLAUDE.md - NietzscheDB

> [!danger] REGRAS ABSOLUTAS - NUNCA VIOLAR
> Estas regras sao **imutaveis**. Qualquer violacao quebra o sistema.

---

## Regras Absolutas

### Binary Quantization - REJEITADO PERMANENTEMENTE
- **NUNCA** implementar Binary Quantization como metrica do HNSW
- `sign(x)` destroi hierarquia hiperbolica (magnitude = profundidade no Poincare ball)
- Pre-filter com oversampling >=30x e rescore obrigatorio: **UNICA excecao**
- Decisao unanime 2026-02-19 — ref: `docs/analysis/risco_hiperbolico.md` PARTE 4

### Feature GPU e OBRIGATORIA
- O server **SEMPRE** compila com `--features gpu` (default no Cargo.toml)
- **NUNCA** compilar sem GPU — o binary em producao depende de CUDA + cuVS
- Feature `gpu` ativa `nietzsche-neural/cuda` → `ort/cuda` → 12 modelos ONNX usam `CUDAExecutionProvider`
- Sem `ort/cuda`, fallback silencioso para CPU (performance catastrofica)
- Para testar crates individuais: `cargo check -p <crate>` sem o server

### VM (nietzsche-eva-gpu) - JAMAIS DESLIGAR
- IP: `136.111.0.47` — **NUNCA** executar `gcloud compute instances stop` ou `sudo shutdown`
- Roda: NietzscheDB (865K+ nos, 35 collections), EVA-X, Malaria API, nginx
- Reiniciar servicos individuais (`systemctl restart`) e permitido

---

## O Que E o NietzscheDB

> [!info] Multi-Manifold Graph Database
> Primeiro banco de dados do mundo que opera em **4 geometrias nao-Euclidianas simultaneamente** a partir de uma unica camada Poincare.

**Linguagem**: Rust (nightly 1.96.0) | **Workspace**: 48 crates | **Versao**: 3.1.0 (2026-03-08)
**Proposito**: Substrato de conhecimento para o sistema AGI [[EVA-Mind]]

### As 4 Geometrias
| Geometria | Curvatura | Uso | Crate |
|-----------|-----------|-----|-------|
| **Poincare Ball** | K<0 | Storage, KNN, HNSW, difusao | `nietzsche-hnsw` / `nietzsche-hnsw-gpu` |
| **Klein Model** | K<0 | Pathfinding O(1) colinearidade | `nietzsche-hyp-ops::klein` |
| **Riemann Sphere** | K>0 | Sintese dialetica (Hegel) | `nietzsche-hyp-ops::riemann` |
| **Minkowski Spacetime** | Flat | Causalidade (cones de luz) | `nietzsche-hyp-ops::minkowski` |

**Principio fundamental**: No Poincare ball, `||x||` (magnitude) = profundidade hierarquica.
Centro = abstrato, fronteira = especifico. E por isso que Binary Quantization (`sign(x)`) e proibida.

---

## Arquitetura de Crates

### Camada Fundacao
| Crate | Responsabilidade |
|-------|-----------------|
| `nietzsche-core` | Metricas, quantizacao, tipos de vetor, param tuning |
| `nietzsche-proto` | Definicoes protobuf gRPC (71+ RPCs) |
| `nietzsche-graph` | Modelo de grafo: Node/Edge/PoincareVector, RocksDB (10 CFs), WAL, transacoes |
| `nietzsche-query` | NQL (Nietzsche Query Language) — parser PEG + executor |
| `nietzsche-hyp-ops` | 4 geometrias: Poincare, Klein, Riemann, Minkowski |

### Camada Search & Indexing
| Crate | Responsabilidade |
|-------|-----------------|
| `nietzsche-hnsw` | HNSW CPU para espaco hiperbolico |
| `nietzsche-hnsw-gpu` | CAGRA via NVIDIA cuVS 24.6 (CUDA 12.x) |
| `nietzsche-tpu` | Google TPU (PJRT Ironwood) |
| `nietzsche-filtered-knn` | KNN filtrado com pre-filter/rescore |
| `nietzsche-pq` | Product Quantization |
| `nietzsche-secondary-idx` | Indices secundarios em metadata |

### Camada Algoritmos
| Crate | Responsabilidade |
|-------|-----------------|
| `nietzsche-algo` | 11 algoritmos: PageRank, Louvain, Betweenness, WCC, Degree, Triangles, A*, Closeness, Jaccard, Label Propagation |
| `nietzsche-pregel` | Framework Pregel (difusao, heat kernels) |
| `nietzsche-cugraph` | Algoritmos GPU via cuGraph |

### Camada Autonomia (Agency)
| Crate | Responsabilidade |
|-------|-----------------|
| `nietzsche-agency` | Motor de autonomia: 27 fases, daemons, desejos, intents |
| `nietzsche-agi` | Stack de inferencia: 8 camadas (representacao → metabolica) |
| `nietzsche-lsystem` | Crescimento fractal L-System com Mobius + poda Hausdorff |
| `nietzsche-epistemics` | Metricas epistemicas: coerencia, cobertura, freshness |
| `nietzsche-dream` | Ciclos de sonho (raciocinio especulativo) |
| `nietzsche-narrative` | Narrativas auto-geradas sobre o grafo |
| `nietzsche-wiederkehr` | Detecao de anomalias (Eterno Retorno) |
| `nietzsche-zaratustra` | Vontade de Poder → Eterno Retorno → Ubermensch |
| `nietzsche-sleep` | Reconsolidacao Riemanniana (Adam + rollback) |
| `nietzsche-sensory` | Compressao multi-modal: audio/texto/imagem → vetores latentes |

### Camada Neural & ML
| Crate | Responsabilidade |
|-------|-----------------|
| `nietzsche-neural` | Registry ONNX (12 modelos), CUDAExecutionProvider |
| `nietzsche-gnn` | Graph Neural Networks em manifolds hiperbolicos |
| `nietzsche-rl` | Agentes de Reinforcement Learning |
| `nietzsche-vqvae` | Vector Quantized VAE |
| `nietzsche-mcts` | Monte Carlo Tree Search |

### Camada Infra & API
| Crate | Responsabilidade |
|-------|-----------------|
| `nietzsche-api` | Router gRPC unificado |
| `nietzsche-server` | Entry point + dashboard embebido (rust-embed) |
| `nietzsche-cluster` | Replicacao leader-follower, anti-entropy |
| `nietzsche-kafka` | Integracao Kafka |
| `nietzsche-mcp` | Model Context Protocol para Claude |
| `nietzsche-dsi` | Pipeline de ingestao de dados |

---

## RocksDB - Column Families

| CF | Conteudo | Tamanho tipico |
|----|----------|---------------|
| `nodes` | NodeMeta (~100 bytes) | Leve |
| `embeddings` | PoincareVector (dim * 4 bytes) | Pesado |
| `edges` | Edge com CausalType + metadata | Medio |
| `adj_out` | Adjacencia saida (node → edges) | Indice |
| `adj_in` | Adjacencia entrada (node ← edges) | Indice |
| `meta` | Metadata do collection | Leve |
| `sensory` | Dados sensoriais comprimidos | Variavel |
| `energy_idx` | Indice de energia | Leve |
| `meta_idx` | Indice de metadata | Leve |
| `lists` | Estruturas de lista (rpush/lrange) | Variavel |

> [!tip] NodeMeta Separation (v2.1.0)
> `Node` foi separado em `NodeMeta` + `PoincareVector`. Operacoes de metadata (energy, hausdorff) nao tocam embeddings.
> BFS le apenas NodeMeta (~100 bytes vs ~24KB) → **10-25x speedup**.
> `Node` implementa `Deref<Target=NodeMeta>` — backward compatible.

---

## Agency Engine - Fases do Tick

| Fase | Nome | Intervalo | Descricao |
|------|------|-----------|-----------|
| 1-10 | Core L-System | cada tick | Rewrite rules, branching, energy |
| 11-20 | Links, Dream, Shatter | variavel | Link prediction, dream cycles |
| 17 | Ego-Cache | cada tick | Hot-tier identity cache O(1) |
| 18 | Reasoning Engine | cada tick | Multi-hop deductive reasoning |
| 19 | Self-Healing | cada tick | Boundary drift, orfaos, dead edges |
| 20 | Graph Learning | cada tick | Hotspots, sector growth rates |
| 21 | Energy minimization | energy_interval | Gradient descent no Poincare ball |
| 22 | Flywheel | flywheel_interval | Feedback loops unificados |
| 23 | Hyperbolic Training | training_interval | Contrastive loss + Riemannian SGD |
| 24 | Temporal Decay | `AGENCY_TEMPORAL_DECAY_INTERVAL`(10) | w * e^(-lambda*t), prune edges |
| 25 | Graph Growth | `AGENCY_GROWTH_INTERVAL`(20) | Discover + propose new edges |
| 26 | Cognitive Layer | `AGENCY_COGNITIVE_INTERVAL`(30) | Cluster → concept nodes |
| 27 | Epistemic Evolution | `AGENCY_EVOLUTION_27_INTERVAL`(40) | AlphaEvolve-style knowledge evolution |

### Padrao de Intents
Engine produz `AgencyIntent` (read-only) → server handler executa mutations (write lock):
- `ApplyTemporalDecay` → `storage().get_edge()` + `put_edge()`
- `PruneDecayedEdge` → `delete_edge(id)`
- `ProposeEdge` → `insert_edge(Edge::new(...))`
- `ProposeConcept` → `insert_node(concept)` + `insert_edge()` por membro
- `EpistemicMutation` → ProposeEdge/EnergyBoost/Reclassify/PruneEdge (Phase 27)

### Config Agency (env vars)
```env
AGENCY_TICK_SECS=60
AGENCY_TEMPORAL_DECAY_ENABLED=true
AGENCY_TEMPORAL_DECAY_INTERVAL=10
AGENCY_TEMPORAL_DECAY_LAMBDA=0.0000001
AGENCY_GROWTH_ENABLED=true
AGENCY_GROWTH_INTERVAL=20
AGENCY_COGNITIVE_ENABLED=true
AGENCY_COGNITIVE_INTERVAL=30
AGENCY_EVOLUTION_27_ENABLED=true
AGENCY_EVOLUTION_27_INTERVAL=40
AGENCY_EVOLUTION_27_MAX_EVAL=500
AGENCY_EVOLUTION_27_QUALITY_FLOOR=0.4
AGENCY_SKIP_COLLECTIONS=eva_core,eva_self_knowledge,eva_codebase,eva_docs
LSYSTEM_HAUSDORFF_SAMPLE=8000
LSYSTEM_K=12
```

---

## gRPC API (71+ RPCs)

### Collection Management
`CreateCollection` · `DeleteCollection` · `ListCollections` · `GetCollectionStats` · `RebuildIndex`

### Vector Operations
`Insert` · `BatchInsert` · `InsertText` · `Delete` · `Search` · `SearchBatch`

### Graph Navigation
`GetNode` · `GetNeighbors` · `GetConceptParents` · `Traverse` · `FindSemanticClusters`

### Manifold Reasoning (v3.0)
`Synthesis` · `SynthesisMulti` · `CausalNeighbors` · `CausalChain` · `KleinPath` · `IsOnShortestPath`

### Streaming
`Monitor` (stream SystemStats) · `SubscribeToEvents` (stream EventMessage) · `Replicate`

### Filtros
`FilterExpr`: Match | Range | And | Or | Not | Contains
`typed_metadata`: string, int64, double, bool

**Proto**: `crates/nietzsche-proto/proto/nietzsche_db.proto`

---

## HTTP Dashboard API

Base: `http://136.111.0.47:8080` (externo) ou `http://localhost:8080` (VM)

```
GET  /api/collections                     # lista collections
GET  /api/stats                           # totais globais
GET  /api/graph?collection=X&limit=N      # nos e edges
GET  /api/node/<uuid>?collection=X        # no individual
GET  /api/search?q=texto&collection=X     # busca full-text
GET  /api/algo/{pagerank,louvain,betweenness,degree,triangles,wcc}?collection=X
GET  /api/agency/dashboard?collection=X   # dashboard cognitivo
GET  /api/agency/observation?collection=X # termodinamica
GET  /api/agency/health/latest?collection=X
GET  /api/agency/observer?collection=X    # identidade observer
GET  /api/agency/desires?collection=X     # gaps de conhecimento
GET  /api/agency/evolution?collection=X   # L-System evolution
GET  /api/agency/narrative?collection=X   # narrativa do grafo
GET  /api/agency/nietzsche-evolve?collection=X  # AlphaEvolve dashboard
POST /api/reasoning/synthesis             # sintese Riemanniana
POST /api/reasoning/causal-neighbors      # vizinhos causais
POST /api/reasoning/causal-chain          # cadeia causal
POST /api/reasoning/klein-path            # geodesica Klein
GET  /api/export/nodes?format=jsonl&collection=X
GET  /api/export/edges?format=jsonl&collection=X
```

---

## .env - Configuracao Completa

```env
# Poincare hyperbolic storage
HS_DIMENSION=64
HS_METRIC=poincare
HS_HNSW_M=64
HS_HNSW_EF_CONSTRUCT=400
HS_HNSW_EF_SEARCH=400
HS_INDEXER_CONCURRENCY=8
HS_QUANTIZATION_LEVEL=scalar

# WAL Durability
HYPERSPACE_WAL_SYNC_MODE=batch
HYPERSPACE_WAL_BATCH_INTERVAL=100

# NietzscheDB Vector Backend
NIETZSCHE_VECTOR_BACKEND=embedded    # CRITICO: sem isto usa MockVectorStore (linear scan)
NIETZSCHE_VECTOR_DIM=3072            # Gemini embeddings
NIETZSCHE_VECTOR_METRIC=cosine
NIETZSCHE_PORT=50051
NIETZSCHE_DASHBOARD_PORT=8080

# Sleep Cycle
NIETZSCHE_SLEEP_INTERVAL_SECS=0
NIETZSCHE_SLEEP_NOISE=0.02
NIETZSCHE_SLEEP_ADAM_STEPS=10
NIETZSCHE_HAUSDORFF_THRESHOLD=0.15
```

> [!warning] NIETZSCHE_VECTOR_BACKEND=embedded
> Sem esta var, o servidor usa `MockVectorStore` (scan linear O(n)) em vez do HNSW real.

---

## Build & Deploy

### Pipeline completo (local → VM)
```bash
# 1. LOCAL: commit + push
cd D:/DEV/NietzscheDB
git add <files> && git commit -m "feat: ..." && git push origin main

# 2. VM: pull
ssh -i ~/.ssh/google_compute_engine -o ConnectTimeout=30 web2a@136.111.0.47 \
  "cd /home/web2a/NietzscheDB && git pull origin main"

# 3. VM: build (CUDA + cuVS obrigatorio)
ssh -i ~/.ssh/google_compute_engine -o ConnectTimeout=30 web2a@136.111.0.47 "
  source ~/.cargo/env && cd /home/web2a/NietzscheDB && \
  LIBRARY_PATH=/home/web2a/miniforge3/envs/cuvs/lib \
  LD_LIBRARY_PATH=/home/web2a/miniforge3/envs/cuvs/lib \
  cargo build --release -p nietzsche-server"

# 4. VM: deploy
ssh -i ~/.ssh/google_compute_engine -o ConnectTimeout=30 web2a@136.111.0.47 "
  sudo cp /home/web2a/NietzscheDB/target/release/nietzsche-server /usr/local/bin/nietzsche-server && \
  sudo systemctl restart nietzsche-server"
```

**Tempo de build**: ~4-5 min (release) | **Binary**: ~45 MB

### cuVS Environment (CRITICO para build)
```bash
export CUVS_ROOT=/home/web2a/miniforge3/envs/cuvs
export CMAKE_PREFIX_PATH=$CUVS_ROOT
export CPATH=$CUVS_ROOT/include
export LD_LIBRARY_PATH=$CUVS_ROOT/lib
export LIBRARY_PATH=$CUVS_ROOT/lib
```
Sem estas vars: `cuvs/core/c_api.h not found`

### Compilacao LOCAL (Windows, sem GPU)
```bash
# Apenas check/test de crates individuais
cargo check -p nietzsche-agency
cargo check -p nietzsche-hyp-ops
cargo test -p nietzsche-hyp-ops
# NAO compila localmente: nietzsche-server, nietzsche-hnsw-gpu, nietzsche-lsystem(cuda)
```

---

## VM - Servicos & Portas

| Servico | Porta | Protocolo | Systemd |
|---------|-------|-----------|---------|
| nietzsche-server | :50051 | gRPC | `nietzsche-server.service` |
| nietzsche-server | :8080 | HTTP (dashboard) | mesmo |
| eva-x | :8091 / :50052 | REST / gRPC | `eva-x.service` |
| malaria-api | :8082 | REST | `malaria-api.service` |
| nginx | :80 / :443 | HTTP/HTTPS (proxy gRPC) | `nginx.service` |

### Acesso SSH
```bash
ssh -i ~/.ssh/google_compute_engine -o ConnectTimeout=30 web2a@136.111.0.47 "<cmd>"
```

### Controlo de Servicos
```bash
# NietzscheDB
ssh -i ~/.ssh/google_compute_engine -o ConnectTimeout=30 web2a@136.111.0.47 "sudo systemctl {start|stop|restart|status} nietzsche-server"

# Scripts de conveniencia na VM:
#   /home/web2a/nietzsche-ctl.sh {start|stop|restart|status|logs}
#   /home/web2a/vm-ctl.sh {all|ndb|eva|malaria|nginx} {start|stop|restart|status}
```

### gRPC Remoto (HTTPS:443 via nginx)
```python
import grpc, os
from nietzschedb.proto import nietzsche_pb2 as pb, nietzsche_pb2_grpc as rpc

with open(os.path.expanduser('~/AppData/Local/Temp/eva-cert.pem'), 'rb') as f:
    cert = f.read()
creds = grpc.ssl_channel_credentials(root_certificates=cert)
channel = grpc.secure_channel('136.111.0.47:443', creds)
stub = rpc.NietzscheDBStub(channel)
```

### Dados
- Collections: `/var/lib/nietzsche/collections/` (~35 collections, ~10 GB)
- Disco: 87 GB / 194 GB (45%)
- Auth: desabilitada

---

## Testes

### Integration Tests (Python)
```bash
# 64 testes, 20 categorias
NIETZSCHE_HOST=136.111.0.47:50051 pytest tests/test_eva_memory.py -v

# Teste especifico
NIETZSCHE_HOST=136.111.0.47:50051 pytest tests/test_eva_memory.py -v -k "test_episodic"
```
Deps: `pip install grpcio grpcio-tools pytest` (grpcio>=1.78.0)

### Categorias de Teste
| Categoria | # | Cobre |
|-----------|---|-------|
| Collection Management | 4 | create, list, info, drop |
| Node CRUD | 8 | Episodic/Semantic/Concept, UUID, get/delete, energy |
| Edge CRUD | 4 | CONTAINS, TEMPORAL_NEXT, HEBBIAN_VISUAL, delete |
| Merge/Upsert | 3 | MergeNode create+match, MergeEdge |
| KNN Search | 2 | busca vetorial Poincare, filtros |
| NQL Queries | 5 | MATCH, WHERE, COUNT, error handling |
| Sensory Layer | 5 | insert/get/reconstruct/degrade, multimodal |
| Batch Operations | 2 | batch nodes (10), batch edges |
| Cache/TTL | 5 | set/get/del/overwrite, reap |
| Graph Traversal | 2 | BFS, Dijkstra |
| Graph Algorithms | 6 | PageRank, Louvain, WCC, Degree, Betweenness, Triangles |
| Synthesis | 2 | 2-node, multi-node |
| Perception Pipeline | 2 | Scene2D+Objects+edges |
| Episodic Memory | 2 | conversacao EVA-Mind, topic extraction |
| Poincare Constraints | 3 | hierarquia magnitude, zero vec, dim mismatch |
| Admin/Stats | 3 | health, stats, node count |
| Stress | 2 | 100 nodes batch, KNN rapido |

### Gotchas
- Server requer UUID valido nos IDs (nao aceita strings arbitrarias)
- `OriginalShape` no InsertSensory: tagged enum JSON `{"Image": {"width": W, "height": H, "channels": C}}`
- NQL `MATCH (n:Episodic) RETURN n` pode dar vazio em collections novas (HNSW index delay)
- Reconstruct sensory tem perda de precisao (quantizacao interna)
- MergeEdge sempre retorna `created=True` (sem dedup por from+to+type)

---

## Scripts de Dados

### Galaxy Scripts (inserir conhecimento)
```bash
scp -i ~/.ssh/google_compute_engine scripts/<script>.py web2a@136.111.0.47:/tmp/
ssh -i ~/.ssh/google_compute_engine web2a@136.111.0.47 \
  "cd /home/web2a/NietzscheDB && python3 /tmp/<script>.py --host localhost:50051 --collection <nome> --metric poincare --dim 128"
```

| Script | Collection | Galaxias | Nos | Edges |
|--------|-----------|----------|-----|-------|
| `insert_tech_galaxies.py` | `tech_galaxies` | 8 | 760 | 2.8K |
| `insert_knowledge_galaxies.py` | `knowledge_galaxies` | 8 | 970 | 3.5K |
| `insert_culture_galaxies.py` | `culture_galaxies` | 10 | 988 | 3.5K |
| `insert_science_galaxies.py` | `science_galaxies` | 8 | 721 | 2.6K |

> [!note] Servidor demora ~2 min para ficar pronto apos restart.

### Relatorios
```bash
# Memoria EVA (11 secoes, ~5 min completo, ~30s --quick)
python3 scripts/memoria_eva.py --host http://136.111.0.47:8080 --quick

# Super Report (brain scan de todas collections)
python3 scripts/super_report.py --host http://136.111.0.47:8080 --all
```

---

## SDKs

| SDK | Path | Linguagem |
|-----|------|-----------|
| Python | `sdks/python/` | `NietzscheClient` com gRPC |
| TypeScript | `sdks/ts/` | gRPC-web, streaming reativo |
| Go | `sdks/go/` | Cliente nativo, manifold methods |
| C++ | `sdks/cpp/` | Bindings nativos |

---

## NQL (Nietzsche Query Language)

Parser PEG em `crates/nietzsche-query/`. Spec completa: `docs/nql/NQL.md` (77KB).

```nql
-- Exemplos basicos
MATCH (n:Episodic) WHERE n.energy > 0.5 RETURN n LIMIT 10
MATCH (n)-[e:CONTAINS]->(m) WHERE n.label = "AI" RETURN n, e, m
MATCH (n:Concept) RETURN COUNT(n)
```

---

## NietzscheEvolve (AlphaEvolve-inspired)

Evolucao autonoma activa desde 2026-03-19:
- **ParameterEvolve**: `crates/nietzsche-core/src/param_evolve.rs` — HNSW tuning
- **EnergyEvolve**: `crates/nietzsche-agency/src/energy_evolve.rs` — funcoes de energia
- **CogEvolve**: `crates/nietzsche-agency/src/cog_evolve.rs` — estrategias cognitivas
- **HypothesisEvolve**: `nietzsche-lab/prompt_evolve.py` — meta-evolucao LLM
- **Dashboard**: `GET /api/agency/nietzsche-evolve?collection=X`
- Docs: `docs/roadmap/NietzscheEvolve.md`

---

## Camada Sensorial

Degradacao progressiva baseada em energia:
| Energia | Precisao | Tamanho |
|---------|----------|---------|
| E >= 0.7 | f32 | 100% |
| E >= 0.5 | f16 | 50% |
| E >= 0.3 | int8 | 25% |
| E >= 0.1 | PQ 64B | ~2% |
| E < 0.1 | None (descartado) | 0 |

Encoders: Audio, Text, Image, Fusion (multi-modal)

---

## Zaratustra (Cascata de Energia)

Tres fases filosoficas do motor de energia:
1. **Vontade de Poder** (Will to Power): Propagacao de energia pelo grafo
2. **Eterno Retorno** (Eternal Recurrence): Detecao de padroes recorrentes
3. **Ubermensch**: Promocao de nos de elite (alta energia + coerencia)

---

## Dependencias Principais

| Categoria | Crates |
|-----------|--------|
| Runtime | tokio 1.35, tonic 0.10, prost 0.12 |
| Storage | rocksdb 0.21, bincode 1.3, rkyv 0.7 |
| ML | ort 2.0.0-rc.11 (ONNX + CUDA) |
| Concurrent | dashmap 5.5, parking_lot 0.12, rayon 1.10 |
| Crypto | aes-gcm, sha2, hkdf |
| Math | ordered-float 4.1, ndarray 0.17 |
| GPU | cuvs 24.6, cudarc (optional) |

---

## Benchmarks (do .env)

| Config | Dim | Metric | Insert QPS | Search QPS | P99 Lat | Recall |
|--------|-----|--------|-----------|-----------|---------|--------|
| M=64, ef=400 | 1024 | Cosine | 23,662 | 995 | 2.36ms | 99.8% |
| M=64, ef=400 | 64 | Poincare | 158,675 | 2,777 | 1.52ms | 99.9% |

---

## Changelog Resumido

| Versao | Data | Destaque |
|--------|------|----------|
| **3.1.0** | 2026-03-08 | Agency Phases XVII-XXIV (Ego-Cache, Reasoning, Self-Healing, Learning, Compression, Sharding, World Model, Flywheel) |
| **3.0.0** | 2026-02-22 | Multi-Manifold Architecture (Klein, Riemann, Minkowski) + 6 RPCs |
| **2.1.0** | 2026-02-19 | NodeMeta Separation, f32 coords, Binary Quant rejection |
| **2.0.0** | 2026-02-16 | Replication anti-entropy, multi-tenancy, WASM |
| **1.6.0** | 2026-02-15 | Cold storage, lazy loading |

---

## Paths Criticos (Referencia Rapida)

```
D:/DEV/NietzscheDB/                          # Root
  Cargo.toml                                  # Workspace (48 crates)
  .env                                        # Config completa
  crates/
    nietzsche-server/                         # Entry point
    nietzsche-api/                            # Router gRPC
    nietzsche-graph/                          # RocksDB + Node/Edge
    nietzsche-query/                          # NQL parser
    nietzsche-agency/                         # Motor autonomia (27 fases)
    nietzsche-agi/                            # Stack inferencia
    nietzsche-hyp-ops/                        # 4 geometrias
    nietzsche-proto/proto/nietzsche_db.proto  # Definicao gRPC
    nietzsche-hnsw-gpu/                       # CAGRA/cuVS
    nietzsche-neural/                         # ONNX models
    nietzsche-algo/                           # Algoritmos de grafo
    nietzsche-lsystem/                        # L-System fractal
    nietzsche-sensory/                        # Multi-modal compression
  sdks/{python,ts,go,cpp}/                    # Client SDKs
  tests/                                      # Integration tests Python
  scripts/                                    # Galaxy scripts + relatorios
  dashboard/                                  # React frontend (embebido)
  docs/
    architecture/ARCHITECTURE.md              # Visao geral sistema
    architecture/Architecture_Manifolds.md    # 4 geometrias
    roadmap/NietzscheEvolve.md                # AlphaEvolve
    roadmap/NietzscheLab.md                   # Autoresearch
    roadmap/FASES.md                          # 26+ fases
    nql/NQL.md                                # Spec NQL (77KB)
    analysis/risco_hiperbolico.md             # Porque BQ e proibida
```

---

## Scripts Locais (D:\DEV\scripts\)

| Script | Funcao |
|--------|--------|
| `ndb.sh` | `{start\|stop\|restart\|status\|logs\|tunnel}` — atalho NietzscheDB |
| `vm-ssh.sh` | `[cmd]` — SSH rapido a VM |
| `vm-tunnel.sh` | `{start\|stop\|status}` — tunnel gRPC local |

---

> [!abstract] Meta
> Este ficheiro serve como **memoria persistente** do Claude para o projecto NietzscheDB.
> Atualizado: 2026-03-26 | Versao: 3.1.0 | Crates: 48 | Nos: 865K+ | Collections: ~35
