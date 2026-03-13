# CLAUDE.md - NietzscheDB

## REGRAS ABSOLUTAS

### Binary Quantization - REJEITADO PERMANENTEMENTE
- NUNCA implementar Binary Quantization como metrica do HNSW
- sign(x) destroi hierarquia hiperbolica (magnitude = profundidade no Poincare ball)
- Pre-filter com oversampling >=30x e rescore obrigatorio: UNICA excecao

### Feature GPU e OBRIGATORIA
- O server SEMPRE compila com `--features gpu` (default no Cargo.toml)
- NUNCA compilar sem GPU — o binary em producao depende de CUDA + cuVS
- A feature `gpu` ativa `nietzsche-neural/cuda` que habilita `ort/cuda` — todos os 12 modelos ONNX usam `CUDAExecutionProvider` para inferencia na GPU
- Sem `ort/cuda`, o `CUDAExecutionProvider` faz fallback silencioso para CPU
- Se precisar testar apenas crates internos, usar `cargo check -p <crate>` sem o server

---

## Build & Deploy (GPU)

### Requisitos na VM
- **VM**: `nietzsche-eva-gpu` (GCP `us-central1-a`, projeto `malaria-487614`)
- **IP**: `136.111.0.47` (estatico)
- **GPU**: NVIDIA T4 com CUDA 12.x
- **cuVS**: 24.6 instalado via conda em `/home/web2a/miniforge3/envs/cuvs/`
- **Rust**: nightly (instalado na VM)
- **Binary producao**: `/usr/local/bin/nietzsche-server`
- **Env**: `/etc/nietzsche.env`
- **Systemd**: `nietzsche-server.service`

### Pipeline completo (push → pull → build → deploy)

```bash
# 1. LOCAL: commit + push
cd D:/DEV/NietzscheDB
git add <files>
git commit -m "feat: ..."
git push origin main

# 2. VM: pull
gcloud compute ssh web2a@nietzsche-eva-gpu --zone=us-central1-a \
  --command="cd /home/web2a/NietzscheDB && git pull origin main"

# 3. VM: build (OBRIGATORIO incluir env vars do cuVS)
gcloud compute ssh web2a@nietzsche-eva-gpu --zone=us-central1-a --command="
  cd /home/web2a/NietzscheDB && \
  source ~/.cargo/env && \
  export CUVS_ROOT=/home/web2a/miniforge3/envs/cuvs && \
  export CMAKE_PREFIX_PATH=\$CUVS_ROOT && \
  export CPATH=\$CUVS_ROOT/include && \
  export LD_LIBRARY_PATH=\$CUVS_ROOT/lib:\$LD_LIBRARY_PATH && \
  export LIBRARY_PATH=\$CUVS_ROOT/lib && \
  cargo build --release -p nietzsche-server"

# 4. VM: deploy (stop → copy → start)
gcloud compute ssh web2a@nietzsche-eva-gpu --zone=us-central1-a --command="
  sudo systemctl stop nietzsche-server && \
  sudo cp /home/web2a/NietzscheDB/target/release/nietzsche-server /usr/local/bin/nietzsche-server && \
  sudo systemctl start nietzsche-server && \
  sleep 3 && \
  sudo systemctl status nietzsche-server --no-pager"
```

### Vars de ambiente para cuVS (CRITICO)
Sem estas vars, o `cuvs-sys` build.rs falha com `cuvs/core/c_api.h not found`:
```bash
export CUVS_ROOT=/home/web2a/miniforge3/envs/cuvs
export CMAKE_PREFIX_PATH=$CUVS_ROOT
export CPATH=$CUVS_ROOT/include         # headers C/C++
export LD_LIBRARY_PATH=$CUVS_ROOT/lib   # runtime linking
export LIBRARY_PATH=$CUVS_ROOT/lib      # compile-time linking
```

### gcloud SSH no Windows (Git Bash)
O gcloud precisa de Python. Se falhar com "Python not found":
```bash
export CLOUDSDK_PYTHON="/c/Users/web2a/AppData/Local/Programs/Python/Python312/python.exe"
```
SSH direto com chave pode dar timeout — usar sempre `gcloud compute ssh`.

### VM pode estar desligada
Verificar status e ligar se necessario:
```bash
gcloud compute instances describe nietzsche-eva-gpu --zone=us-central1-a --format="value(status)"
# Se TERMINATED:
gcloud compute instances start nietzsche-eva-gpu --zone=us-central1-a
# Esperar ~15s antes do SSH
```

---

## Compilacao LOCAL (Windows, sem GPU)

### Verificar crates individuais (sem CUDA)
```bash
cargo check -p nietzsche-agency
cargo check -p nietzsche-hyp-ops
cargo check -p nietzsche-graph
cargo check -p nietzsche-query
# etc.
```

### Testes locais (sem server)
```bash
cargo test -p nietzsche-hyp-ops
cargo test -p nietzsche-experiment
# nietzsche-agency tests podem falhar por shatter.rs — testar modulos individuais
```

### Teste de integracao EVA Memory (Python)
Teste completo da memoria da EVA via Python SDK — 64 testes, 20 categorias.
```bash
# Contra a VM (requer server rodando)
NIETZSCHE_HOST=136.111.0.47:50051 pytest tests/test_eva_memory.py -v

# Teste especifico
NIETZSCHE_HOST=136.111.0.47:50051 pytest tests/test_eva_memory.py -v -k "test_episodic"
```

Deps: `pip install grpcio grpcio-tools pytest`

| Categoria | Testes | O que cobre |
|-----------|--------|------------|
| Collection Management | 4 | create, list, info, drop |
| Node CRUD | 8 | Episodic/Semantic/Concept, custom UUID, get/delete, energy |
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
| Full-Text Search | 1 | busca textual |
| List Store | 2 | rpush/lrange, len |
| Perception Pipeline | 2 | Scene2D+Objects+edges, temporal sequence |
| Episodic Memory Pipeline | 2 | conversacao EVA-Mind, topic extraction |
| Poincare Constraints | 3 | hierarquia magnitude, zero vec, dim mismatch |
| Admin/Stats | 3 | health, stats, node count |
| Production Collections | 1 | verifica collections EVA (read-only) |
| Stress | 2 | 100 nodes batch, KNN rapido |

**Gotchas descobertos nos testes**:
- Server requer UUID valido nos IDs (nao aceita strings arbitrarias)
- `OriginalShape` no InsertSensory e tagged enum JSON: `{"Image": {"width": W, "height": H, "channels": C}}`, `{"Audio": {"samples": N, "sample_rate": SR}}`, `{"Text": {"tokens": N}}`
- NQL `MATCH (n:Episodic) RETURN n` pode retornar vazio em collections novas (HNSW index delay)
- Reconstruct sensory tem perda de precisao significativa (quantizacao interna)
- MergeEdge sempre retorna `created=True` (server nao faz dedup por from+to+type)
- `grpcio>=1.78.0` obrigatorio (stubs gerados com essa versao)

### Analise Completa da Memoria da EVA (memoria_eva.py)
Relatorio que examina TODAS as collections EVA: inventario, estrutura, aprendizagem (Hebbian/L-System), associacoes (PageRank/Louvain), poda/decay (phantoms/pruned), termodinamica, desejos, shatter, healing, observer, narrativa, busca textual.

```bash
# Na VM (relatorio completo — demora ~5 min pelos algoritmos de grafo)
python3 /tmp/memoria_eva.py --host http://localhost:8080

# Na VM (modo rapido — pula PageRank/Louvain/WCC/Degree, ~30s)
python3 /tmp/memoria_eva.py --host http://localhost:8080 --quick

# Contra a VM (externo)
python3 scripts/memoria_eva.py --host http://136.111.0.47:8080 --quick

# Apenas uma collection
python3 /tmp/memoria_eva.py --host http://localhost:8080 --collection eva_core

# Salvar em ficheiro
python3 /tmp/memoria_eva.py --host http://localhost:8080 --output /tmp/eva_report.txt
```

Copiar para VM: `scp -i ~/.ssh/google_compute_engine scripts/memoria_eva.py web2a@136.111.0.47:/tmp/`

Sem dependencias externas (usa apenas urllib + json da stdlib).

11 secoes: Inventario Global, Estrutura, Aprendizagem, Associacoes, Poda & Decay, Termodinamica, Desejos, Shatter & Healing, Observer, Narrativa, Busca Textual + Resumo Final.

### O que NAO compila localmente
- `nietzsche-server` — depende de `nietzsche-hnsw-gpu` (CUDA/cuVS)
- `nietzsche-hnsw-gpu` — requer CUDA toolkit + cuVS SDK
- `nietzsche-lsystem` com feature `cuda` — requer CUDA

---

## Arquitetura do Agency Engine (Roadmap)

### Fases do tick (AgencyEngine::tick)
| Fase | Nome | Intervalo | Descricao |
|------|------|-----------|-----------|
| 1-10 | Core L-System | cada tick | Rewrite rules, branching, energy |
| 11-20 | Links, Dream, Shatter | variavel | Link prediction, dream cycles |
| 21 | Energy minimization | energy_interval | Gradient descent no Poincare ball |
| 22 | Flywheel | flywheel_interval | Feedback loops |
| 23 | Hyperbolic Training | training_interval | Contrastive loss + Riemannian SGD |
| 24 | Temporal Decay | temporal_decay_interval(10) | w × e^(-lambda*t), prune edges |
| 25 | Graph Growth | growth_interval(20) | Discover + propose new edges |
| 26 | Cognitive Layer | cognitive_interval(30) | Cluster → concept nodes |
| 27 | Epistemic Evolution | evolution_27_interval(40) | autoresearch-style knowledge evolution |

### Padrao de intents
O engine produz `AgencyIntent` (read-only) → server handler executa mutations (write lock):
- `ApplyTemporalDecay` → `storage().get_edge()` + `storage().put_edge()`
- `PruneDecayedEdge` → `delete_edge(id)`
- `ProposeEdge` → `insert_edge(Edge::new(...))`
- `ProposeConcept` → `insert_node(concept)` + `insert_edge()` por membro
- `EpistemicMutation` → avalia qualidade epistemica + aplica/rejeita mutacao

### Config env vars (todas com prefixo AGENCY_)

#### Collection selection (controla ONDE o Agency/L-System corre)
```env
# Collections a EXCLUIR do Agency Engine (comma-separated, exact match)
# Sobrepoe o skip list hardcoded. Impede L-System, sleep, dreams, etc.
AGENCY_SKIP_COLLECTIONS=eva_core,eva_self_knowledge,eva_codebase,eva_docs

# Override do AGENCY_ALWAYS hardcoded (collections que correm mesmo com <10 nodes)
# Se definido, SUBSTITUI a lista hardcoded. Se vazio, usa defaults.
AGENCY_ALWAYS_COLLECTIONS=memories,signifier_chains,eva_mind,patient_graph
```

**Skip list hardcoded** (sempre activo, alem do env var):
`eva_cache`, `eva_perceptions`, `speaker_embeddings`, `eva_sensory`, `lobby_*`, `test_*`

**Always list hardcoded** (default se `AGENCY_ALWAYS_COLLECTIONS` nao definido):
`memories`, `signifier_chains`, `eva_mind`, `eva_core`, `patient_graph`

**Logica**: `AGENCY_SKIP_COLLECTIONS` tem prioridade sobre tudo (skip ganha sobre always).

#### Parametros do Agency Engine
```env
AGENCY_TICK_SECS=60
AGENCY_TEMPORAL_DECAY_ENABLED=true
AGENCY_TEMPORAL_DECAY_INTERVAL=10
AGENCY_TEMPORAL_DECAY_LAMBDA=0.0000001
AGENCY_GROWTH_ENABLED=true
AGENCY_GROWTH_INTERVAL=20
AGENCY_GROWTH_DISTANCE_THRESHOLD=1.5
AGENCY_COGNITIVE_ENABLED=true
AGENCY_COGNITIVE_INTERVAL=30
AGENCY_COGNITIVE_CLUSTER_RADIUS=0.3
AGENCY_EVOLUTION_27_ENABLED=true
AGENCY_EVOLUTION_27_INTERVAL=40
AGENCY_EVOLUTION_27_MAX_EVAL=500
AGENCY_EVOLUTION_27_QUALITY_FLOOR=0.4
AGENCY_EVOLUTION_27_MAX_PROPOSALS=5
AGENCY_EVOLUTION_27_MIN_ENERGY=0.05
```

#### Parametros L-System
```env
LSYSTEM_HAUSDORFF_SAMPLE=8000   # nodes amostrados por tick (0 = todos)
LSYSTEM_K=12                     # k vizinhos para Hausdorff local (min 3)
```

---

## Servicos na VM

| Servico | Porta | Protocolo | Status |
|---------|-------|-----------|--------|
| nietzsche-server | :50051 | gRPC | ATIVO |
| nietzsche-server | :8080 | HTTP (dashboard) | ATIVO |
| eva-x | :8091 / :50052 | REST / gRPC | ATIVO |
| malaria-api | :8082 | REST | ATIVO |
| nginx | :80 / :443 | HTTP/HTTPS | ATIVO |

## Data
- Collections: `/var/lib/nietzsche/collections/` (~35 collections, 10 GB)
- Auth: desabilitada
- Disco: 87 GB / 194 GB (45%)

---

## Scripts de Dados & Relatorios

### Inserir Conhecimento (Galaxy Scripts)
Scripts Python em `scripts/` que inserem galaxias de conhecimento via gRPC (porta 50051).
Requerem `grpcio grpcio-tools` instalados na VM.

```bash
# Copiar script para VM e executar
scp -i ~/.ssh/google_compute_engine scripts/<script>.py web2a@136.111.0.47:/tmp/
ssh -i ~/.ssh/google_compute_engine web2a@136.111.0.47 \
  "cd /home/web2a/NietzscheDB && python3 /tmp/<script>.py --host localhost:50051 --collection <nome> --metric poincare --dim 128"
```

| Script | Collection | Galaxias | Nos | Edges | Conteudo |
|--------|-----------|----------|-----|-------|----------|
| `insert_tech_galaxies.py` | `tech_galaxies` | 8 | 760 | 2.8K | AI, Systems, Data, Languages, Security, Web, Math/CS, Blockchain |
| `insert_knowledge_galaxies.py` | `knowledge_galaxies` | 8 | 970 | 3.5K | Literatura, Fisica, Fractais, Cosmologia, Filosofia, Neurociencia, Musica/Arte, Biologia |
| `insert_culture_galaxies.py` | `culture_galaxies` | 10 | 988 | 3.5K | Mitologia, Historia, Matematica Pura, Psicologia, Linguistica, Civilizacoes, Esoterismo, Cinema/Teatro, Gastronomia, Arquitetura |
| `insert_science_galaxies.py` | `science_galaxies` | 8 | 721 | 2.6K | Quimica, Metalurgia, Genomica, Farmacologia, Ciencias da Terra, Engenharia, Economia/Game Theory, Energia |

**IMPORTANTE**: O servidor demora ~2 min para ficar pronto apos restart. O script `insert_culture_galaxies.py` e `insert_science_galaxies.py` ja incluem `grpc.channel_ready_future()` com timeout de 120s.

### Super Report (Brain Scan Completo)
```bash
# Copiar e executar na VM
scp -i ~/.ssh/google_compute_engine scripts/super_report.py web2a@136.111.0.47:/tmp/
ssh -i ~/.ssh/google_compute_engine web2a@136.111.0.47 \
  "python3 /tmp/super_report.py --host http://localhost:8080 --collections tech_galaxies knowledge_galaxies culture_galaxies science_galaxies"

# Scan COMPLETO de todas as 35 collections:
ssh web2a@136.111.0.47 "python3 /tmp/super_report.py --host http://localhost:8080 --all"

# Modo rapido (sem algoritmos pesados):
ssh web2a@136.111.0.47 "python3 /tmp/super_report.py --host http://localhost:8080 --quick"
```

Requer `pip3 install requests` na VM. O relatorio cobre:

| Secao | O que mostra |
|-------|-------------|
| Global Overview | Total nos/edges, todas collections, densidade |
| Cross-Collection | Comparacao Hausdorff, energia, coerencia, gaps |
| Thermodynamics | Temperatura, fase (Solid/Liquid/Gas), entropia, free energy |
| ECAN | Preco de atencao (inflacao), flow total, Hebbian traces |
| Semantic Gravity | Gravity wells, forca media |
| Observer | ID da consciencia do grafo, energia, profundidade |
| Shatter Protocol | Super-nodes, ghosts, avatars, maior grau |
| Self-Healing | Boundary drift, orfaos, dead edges, health score |
| Desires | O que o banco QUER (gaps de conhecimento a preencher) |
| L-System Evolution | Geracao, estrategia, fitness history |
| Narrative | Historia que o banco conta sobre si mesmo |
| PageRank | Top-15 conceitos mais influentes (com nomes) |
| Louvain | Comunidades detectadas, modularidade |
| Betweenness | Nos-ponte entre clusters |
| Triangles | Densidade de triangulacoes |
| WCC | Componentes conectados, fragmentacao |
| Degree | Hubs mais conectados |
| Growth Analysis | Distribuicao de tipos, energia, nos criados pelo Agency |
| Counterfactual | Impacto de remover top nodes |

### Dashboard HTTP API (endpoints uteis para consultas)
Base URL: `http://localhost:8080` (ou `http://136.111.0.47:8080` externo)

```
GET  /api/collections                          # lista todas collections
GET  /api/stats                                # totais globais
GET  /api/graph?collection=X&limit=N           # nos e edges
GET  /api/node/<uuid>?collection=X             # no individual
GET  /api/search?q=texto&collection=X          # busca full-text
GET  /api/algo/pagerank?collection=X           # conceitos influentes
GET  /api/algo/louvain?collection=X            # comunidades
GET  /api/algo/betweenness?collection=X        # nos-ponte
GET  /api/algo/degree?collection=X             # hubs
GET  /api/algo/triangles?collection=X          # triangulacoes
GET  /api/algo/wcc?collection=X                # componentes conectados
GET  /api/agency/dashboard?collection=X        # dashboard cognitivo completo
GET  /api/agency/observation?collection=X      # frame de observacao (termodinamica)
GET  /api/agency/health/latest?collection=X    # ultimo health report
GET  /api/agency/observer?collection=X         # identidade do observer
GET  /api/agency/shatter?collection=X          # protocolo shatter
GET  /api/agency/healing?collection=X          # self-healing
GET  /api/agency/desires?collection=X          # desejos pendentes
GET  /api/agency/evolution?collection=X        # evolucao L-System
GET  /api/agency/narrative?collection=X        # narrativa do grafo
POST /api/reasoning/synthesis                  # sintese Riemanniana
POST /api/reasoning/causal-neighbors           # vizinhos causais
POST /api/reasoning/causal-chain               # cadeia causal
POST /api/reasoning/klein-path                 # caminho geodesico Klein
POST /api/navigate                             # navegacao hiperbolica
GET  /api/export/nodes?format=jsonl&collection=X  # exportar nos
GET  /api/export/edges?format=jsonl&collection=X  # exportar edges
```
