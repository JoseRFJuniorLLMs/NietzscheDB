# CLAUDE.md - NietzscheDB

## REGRAS ABSOLUTAS

### Binary Quantization - REJEITADO PERMANENTEMENTE
- NUNCA implementar Binary Quantization como metrica do HNSW
- sign(x) destroi hierarquia hiperbolica (magnitude = profundidade no Poincare ball)
- Pre-filter com oversampling >=30x e rescore obrigatorio: UNICA excecao

### Feature GPU e OBRIGATORIA
- O server SEMPRE compila com `--features gpu` (default no Cargo.toml)
- NUNCA compilar sem GPU — o binary em producao depende de CUDA + cuVS
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

### Padrao de intents
O engine produz `AgencyIntent` (read-only) → server handler executa mutations (write lock):
- `ApplyTemporalDecay` → `storage().get_edge()` + `storage().put_edge()`
- `PruneDecayedEdge` → `delete_edge(id)`
- `ProposeEdge` → `insert_edge(Edge::new(...))`
- `ProposeConcept` → `insert_node(concept)` + `insert_edge()` por membro

### Config env vars (todas com prefixo AGENCY_)
```env
AGENCY_TEMPORAL_DECAY_ENABLED=true
AGENCY_TEMPORAL_DECAY_INTERVAL=10
AGENCY_TEMPORAL_DECAY_LAMBDA=0.0000001
AGENCY_GROWTH_ENABLED=true
AGENCY_GROWTH_INTERVAL=20
AGENCY_GROWTH_DISTANCE_THRESHOLD=1.5
AGENCY_COGNITIVE_ENABLED=true
AGENCY_COGNITIVE_INTERVAL=30
AGENCY_COGNITIVE_CLUSTER_RADIUS=0.3
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
- Collections: `/var/lib/nietzsche/collections/` (14 collections)
- Auth: desabilitada
- Disco: monitorar com `df -h` (historico de 99% cheio)
