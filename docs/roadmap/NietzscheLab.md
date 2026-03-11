# NietzscheLab — Autonomous Knowledge Evolution Engine

> Loop evolutivo de descoberta automática no NietzscheDB.
> Inspirado no `autoresearch` (Andrej Karpathy), adaptado para evolução epistémica em grafos hiperbólicos.
>
> **Status**: Todas as 3 fases implementadas (2026-03-11).
> - FASE 1: `nietzsche-lab/` (Python) — hypothesis loop externo ✅
> - FASE 2: `crates/nietzsche-epistemics/` (Rust) — métricas epistémicas ✅
> - FASE 3: Phase 27 no `crates/nietzsche-agency/` (Rust) — integração no Agency Engine ✅

---

## 1. Análise Comparativa: NietzscheDB vs autoresearch

### 1.1 O que o autoresearch faz

O `autoresearch` de Karpathy é um **loop de pesquisa automática controlado por agente LLM**.

```
LLM agent → modifica código (train.py) → treina modelo (5 min)
→ mede métrica (val_bpb) → melhorou? commit : rollback → repete
```

Características:
- ~630 linhas de código
- 1 GPU, cada experimento ~5 minutos
- Até 100 experimentos por noite
- Histórico completo no git
- Essencialmente um "PhD automático" que faz ablation experiments sozinho

### 1.2 O padrão universal escondido

O autoresearch implementa um template universal de descoberta:

```
WORLD STATE → AGENT → MODIFY → TEST → METRIC → SELECT → MEMORY
```

Formalmente: `θ(t+1) = argmax(metric(experiment(θ(t))))`

Com LLM gerando as mutações. É **evolução + reinforcement learning + git history**.

### 1.3 O que o NietzscheDB já tem (e o autoresearch não)

| Capacidade | NietzscheDB | autoresearch |
|---|---|---|
| Geometria hiperbólica (Poincaré ball) | **Sim** | Não |
| Multi-manifold (Klein, Riemann, Minkowski) | **Sim** | Não |
| Representação hierárquica nativa | **Sim** (magnitude = profundidade) | Não |
| Dinâmica cognitiva (L-System) | **Sim** (metabolismo vivo) | Não |
| Reconsolidação (Sleep cycle) | **Sim** (Riemannian Adam + rollback) | Não |
| Hebbian learning | **Sim** | Não |
| Attention economy (ECAN) | **Sim** | Não |
| Community detection (Louvain) | **Sim** | Não |
| Causalidade temporal (Minkowski) | **Sim** | Não |
| Agency Engine (26 fases autônomas) | **Sim** | Não |
| Perspectivismo (projeção por agente/profundidade) | **Sim** | Não |

O NietzscheDB é conceitualmente **mais ambicioso** — é infraestrutura cognitiva, não apenas otimização de modelos.

### 1.4 O que o autoresearch tem (e o NietzscheDB ainda não)

| Capacidade | autoresearch | NietzscheDB |
|---|---|---|
| Loop evolutivo de descoberta | **Sim** | **Não** |
| Geração autônoma de hipóteses (LLM) | **Sim** | **Não** |
| Teste automático de consistência | **Sim** (val_bpb) | Parcial (Hausdorff/energy) |
| Seleção/rejeição sem humano | **Sim** (git commit/rollback) | Parcial (Sleep cycle) |
| Experiment journal estruturado | **Sim** (git history) | **Não** |

### 1.5 Diagnóstico

O NietzscheDB está no **nível 2** (banco de inferência com dinâmica cognitiva).
O autoresearch implementa o **nível 3** (descoberta de conhecimento novo).

A diferença:
- O L-System é **metabolismo** (mantém o organismo vivo), não **evolução** (não gera mutações epistémicas).
- O Sleep cycle faz **reconsolidação** (otimiza o existente), não **exploração** (não propõe ideias novas).
- A Phase 25 (graph growth) propõe edges, mas sem **hipótese + teste + score + accept/reject**.

O NietzscheDB tem um **organismo cognitivo**, mas não um **organismo evolutivo**.

### 1.6 Por que o NietzscheDB pode superar o autoresearch

O autoresearch evolui **código de treinamento**. O NietzscheDB pode evoluir **estrutura do conhecimento** — o que é muito mais geral.

Se juntar:
```
geometria não-euclidiana + evolução de hipóteses + LLM agentes
```

O resultado é **evolução de ontologias**: ideias competindo dentro do banco.

A geometria hiperbólica dá uma **métrica natural de fitness**: edges que respeitam a hierarquia
(parent.magnitude < child.magnitude) são geometricamente corretos.

Comparação com sistemas existentes: mais perto de **OpenCog** / **cognitive architectures**
do que de bancos de dados tradicionais.

### 1.7 Três níveis de sistemas de conhecimento

```
Nível 1 — Banco de dados:     armazenar conhecimento
Nível 2 — Banco de inferência: responder perguntas        ← NietzscheDB HOJE
Nível 3 — Banco evolutivo:     descobrir conhecimento novo ← NietzscheLab (objetivo)
```

### 1.8 Nota sobre o regime térmico

A temperatura cognitiva ~0.03 + modularidade ~0.37 observada nos relatórios indica que o grafo
pode estar num **regime crítico sub-congelado**, similar a redes neurais biológicas na transição
de fase entre ordem e caos. Isso pode explicar por que o sistema está "gelado" — poucas mutações
espontâneas, alta estabilidade. O loop evolutivo pode ser exatamente o que desbloqueia essa dinâmica.

---

## 2. Arquitetura do NietzscheLab

### 2.1 Conceito

```
NietzscheDB (ambiente)
      ↓
  Hypothesis Generator (LLM agent)
      ↓
  Mutation (InsertNode/InsertEdge/DeleteEdge via gRPC)
      ↓
  Epistemic Scorer (métricas antes vs depois)
      ↓
  Selection (accept → commit / reject → rollback)
      ↓
  Experiment Journal (log estruturado)
      ↓
  Repete
```

### 2.2 Métricas de fitness epistémico

| Métrica | Fórmula | Significado |
|---|---|---|
| `hierarchy_consistency` | `Σ(parent.depth < child.depth) / N_edges` | Hierarquia hiperbólica respeitada |
| `redundancy` | `1 - (unique_paths / total_paths)` entre A↔B | Grafo não-degenerado |
| `coverage` | `vol(convex_hull(embeddings)) / vol(ball)` | Espaço conceitual utilizado |
| `coherence` | `mean(cosine_sim(neighbors))` | Vizinhos semanticamente próximos |
| `novelty` | `mean(min_dist_to_existing)` para novos nodes | Não é duplicata |
| `hausdorff_delta` | `H_after - H_before` | Complexidade fractal estável |
| `energy_delta` | `E_avg_after - E_avg_before` | Saúde energética do subgrafo |

### 2.3 Visão de longo prazo

```
100 agentes × 1000 experimentos/dia × 365 dias
= 36.5M mutações epistémicas testadas por ano
```

Cada commit bem-sucedido = **mutação epistemológica**.
O banco torna-se literalmente um **organismo cognitivo evolutivo** —
um "Darwinian Knowledge Engine" onde ideias competem e sobrevivem as melhores.

---

## 3. Plano de Implementação

### FASE 1 — Hypothesis Loop (serviço externo)

**Esforço**: ~3-5 dias | **Risco**: baixo | **Mudanças no Rust**: zero

Serviço Python que usa a gRPC API existente + Claude API para gerar e testar hipóteses.

#### 1.1 Componentes

```
nietzsche-lab/
├── lab_runner.py            # Loop principal (equivalente ao autoresearch)
├── hypothesis_generator.py  # LLM gera hipóteses dado um subgrafo
├── consistency_scorer.py    # Mede métricas antes/depois via gRPC
├── experiment_journal.py    # Log estruturado de experimentos
├── grpc_client.py           # Wrapper para NietzscheDB gRPC API
├── config.yaml              # Collection alvo, budget, thresholds
└── requirements.txt         # anthropic, grpcio, protobuf
```

#### 1.2 lab_runner.py — Loop principal

```python
for i in range(max_experiments):
    # 1. Selecionar subgrafo de interesse
    subgraph = grpc_client.query(nql="MATCH (n) WHERE n.energy > 0.3 RETURN n LIMIT 50")

    # 2. Gerar hipótese via LLM
    hypothesis = hypothesis_generator.generate(subgraph, history=journal)

    # 3. Snapshot (backup)
    grpc_client.create_backup()

    # 4. Medir estado antes
    metrics_before = scorer.evaluate(affected_nodes=hypothesis.nodes)

    # 5. Aplicar mutação
    grpc_client.apply(hypothesis.mutation)

    # 6. Medir estado depois
    metrics_after = scorer.evaluate(affected_nodes=hypothesis.nodes)

    # 7. Aceitar ou rejeitar
    delta = scorer.compute_delta(metrics_before, metrics_after)
    if delta.score > threshold:
        journal.log(hypothesis, metrics_before, metrics_after, accepted=True)
    else:
        grpc_client.restore_backup()
        journal.log(hypothesis, metrics_before, metrics_after, accepted=False)
```

#### 1.3 hypothesis_generator.py — Tipos de hipótese

```python
class HypothesisType(Enum):
    NEW_EDGE       = "edge entre dois nodes existentes que deveriam estar ligados"
    NEW_CONCEPT    = "node conceitual que unifica um cluster"
    REMOVE_EDGE    = "edge inconsistente que deve ser removido"
    MERGE_NODES    = "dois nodes que representam o mesmo conceito"
    SPLIT_NODE     = "node sobrecarregado que deve ser dividido"
    RECLASSIFY     = "node no nível hierárquico errado (profundidade incorreta)"
```

O LLM recebe:
- Subgrafo atual (nodes + edges + métricas)
- Histórico de experimentos anteriores (journal)
- Métricas globais do grafo

E retorna uma hipótese estruturada com tipo + nodes afetados + justificação.

#### 1.4 consistency_scorer.py — Métricas via gRPC

```python
def evaluate(self, affected_nodes: list[str]) -> EpistemicMetrics:
    # Hausdorff global (já existe via API)
    global_h = self.grpc.query("MATCH (n) RETURN hausdorff_global()")

    # Hierarquia local: para cada edge, verificar parent.depth < child.depth
    edges = self.grpc.query(f"MATCH (a)-[e]->(b) WHERE a.id IN {ids} RETURN a, b, e")
    hierarchy = sum(1 for a, b, _ in edges if a.depth < b.depth) / len(edges)

    # Energy média
    energy_avg = mean(n.energy for n in affected_nodes)

    # Coherence: distância média entre vizinhos
    coherence = mean(poincare_dist(a, b) for a, b in neighbor_pairs)

    return EpistemicMetrics(
        hausdorff=global_h,
        hierarchy_consistency=hierarchy,
        energy_avg=energy_avg,
        coherence=coherence,
    )
```

#### 1.5 Por que é viável

- Usa 100% da API gRPC existente (InsertNode, InsertEdge, Query, CreateBackup, RestoreBackup)
- Zero mudanças no código Rust
- Pode correr no Windows local apontando para a VM (136.111.0.47:50051)
- Pode usar uma collection de teste sem tocar nas 14 de produção
- Validação rápida: se as métricas existentes funcionam como fitness, justifica Fases 2-3

---

### FASE 2 — Métricas Epistemológicas (crate Rust)

**Esforço**: ~5-8 dias | **Risco**: médio | **Novo crate**: `nietzsche-epistemics`

#### 2.1 Novo crate

```
crates/nietzsche-epistemics/
├── Cargo.toml
├── src/
│   ├── lib.rs           # EpistemicScore struct + evaluate()
│   ├── hierarchy.rs     # hierarchy_consistency metric
│   ├── redundancy.rs    # path redundancy analysis
│   ├── coverage.rs      # convex hull coverage in Poincaré ball
│   ├── coherence.rs     # neighbor semantic coherence
│   └── novelty.rs       # novelty of new nodes/edges
```

#### 2.2 API pública

```rust
pub struct EpistemicScore {
    pub hierarchy_consistency: f32,
    pub redundancy: f32,
    pub coverage: f32,
    pub coherence: f32,
    pub novelty: f32,
    pub composite: f32,              // weighted sum
}

/// Avalia a qualidade epistémica de um conjunto de nodes
pub fn evaluate_subgraph(
    db: &GraphDB,
    node_ids: &[Uuid],
    weights: &ScoreWeights,
) -> Result<EpistemicScore>

/// Avalia o delta epistémico de uma mutação (antes/depois)
pub fn evaluate_mutation(
    db: &GraphDB,
    snapshot_before: &Snapshot,
    affected_nodes: &[Uuid],
    weights: &ScoreWeights,
) -> Result<EpistemicDelta>
```

#### 2.3 Novo RPC

```protobuf
rpc EvaluateEpistemic(EpistemicRequest) returns (EpistemicResponse);

message EpistemicRequest {
    string collection = 1;
    repeated string node_ids = 2;
    optional ScoreWeights weights = 3;
}

message EpistemicResponse {
    float hierarchy_consistency = 1;
    float redundancy = 2;
    float coverage = 3;
    float coherence = 4;
    float novelty = 5;
    float composite = 6;
}
```

#### 2.4 Benefício

A Fase 1 (Python) passa a chamar `EvaluateEpistemic()` via gRPC em vez de calcular métricas
localmente. Mais rápido (Rust nativo, acesso direto ao RocksDB) e mais preciso.

---

### FASE 3 — Phase 27: Evolution (integração no Agency Engine)

**Esforço**: ~5-10 dias | **Risco**: alto | **Modifica**: `nietzsche-agency`

#### 3.1 Nova fase no AgencyEngine

```rust
// crates/nietzsche-agency/src/evolution.rs

pub struct EvolutionPhase {
    pub experiment_budget: usize,        // max mutações por tick
    pub score_threshold: f32,            // mínimo para aceitar
    pub strategy: EvolutionStrategy,
    pub target_temperature: f32,         // descongelar grafo se T < target
}

pub enum EvolutionStrategy {
    Random,                              // mutações aleatórias (baseline)
    Guided,                              // baseado em métricas epistémicas
    LlmDriven { endpoint: String },      // chamada externa a LLM
}
```

#### 3.2 Novos AgencyIntents

```rust
AgencyIntent::ProposeHypothesis {
    hypothesis_type: HypothesisType,
    affected_nodes: Vec<Uuid>,
    predicted_score: f32,
}
AgencyIntent::AcceptHypothesis { experiment_id: u64 }
AgencyIntent::RejectHypothesis { experiment_id: u64 }
```

#### 3.3 Integração no tick loop

```
Phase 25: Graph Growth (proposição de edges)        ← existente
Phase 26: Cognitive Synthesis (clusters → conceitos) ← existente
Phase 27: Evolution (hipótese → teste → seleção)     ← NOVO
```

A Phase 27 usa internamente:
- `nietzsche-epistemics` para scoring
- Sleep cycle snapshots para rollback
- L-System como operador de mutação (SpawnChild/Sibling/Prune)

#### 3.4 Interação com regime térmico

A Phase 27 monitora a temperatura cognitiva. Se T < 0.05 (grafo congelado),
aumenta automaticamente o `experiment_budget` para injetar mais variação.
Isso cria um **termostato epistémico** que mantém o grafo na edge of chaos.

---

## 4. Cronograma sugerido

```
Semana 1-2:  FASE 1 — Python lab runner + collection de teste
Semana 2:    Validação — as métricas existentes funcionam como fitness?
Semana 3-4:  FASE 2 — nietzsche-epistemics crate + RPC
Semana 5-6:  FASE 3 — Phase 27 no Agency Engine
Semana 6:    Testes end-to-end com collection real
```

## 5. Critério de sucesso

A implementação é considerada bem-sucedida quando:

1. **Fase 1**: O lab runner consegue gerar ≥10 hipóteses, testar, e aceitar/rejeitar com base em métricas mensuráveis
2. **Fase 2**: As métricas epistémicas correlacionam com qualidade percebida do grafo (validação humana)
3. **Fase 3**: O Agency Engine roda a Phase 27 autonomamente e o grafo evolui sem degradação (Hausdorff estável, energia média estável ou crescente)

## 6. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Métricas não capturam "qualidade" real | Fase 1 valida antes de investir em Rust |
| LLM gera hipóteses inúteis | Budget limitado + journal para aprendizado |
| Mutações degradam o grafo | Snapshot/rollback obrigatório (já existe) |
| Loop diverge (aceita tudo ou rejeita tudo) | Threshold adaptativo baseado em running average |
| Custo API Claude | Budget configurável, cache de hipóteses similares |
| Temperatura do grafo sobe descontroladamente | Circuit breaker: pausa se Hausdorff sair de [0.5, 1.9] |
