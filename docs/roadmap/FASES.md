# Critérios de sucesso — todas as fases

## Fase 1 — Geometria hiperbólica

**O que medir:**

```python
# Teste 1: Hierarquia semântica aparece nas distâncias?
# Pega 3 triplas (abstrato, médio, específico) e mede

triplas = [
    ("ser_vivo", "mamífero", "gato_siamês"),
    ("emoção", "medo", "medo_de_altura"),
    ("linguagem", "português", "gíria_carioca_2024"),
]

# Critério: d(abstrato, médio) < d(abstrato, específico)
# E essa relação deve ser MAIS pronunciada no hiperbólico que no cosine

def hierarquia_ok(tripla, manifold):
    abstrato, medio, especifico = tripla
    d1 = manifold.dist(abstrato, medio)      # deve ser menor
    d2 = manifold.dist(abstrato, especifico) # deve ser maior
    return d1 < d2  # simples, mas mensurável

# Critério de sucesso: 90%+ das triplas passam no hiperbólico
# vs baseline cosine (esperado: ~60-70%)
```

**Métricas concretas:**

```
✓ PASSA se:
  → 90%+ das triplas hierárquicas ordenadas corretamente
  → depth(abstrato) < depth(médio) < depth(específico) em 85%+ dos casos
  → distortion score hiperbólico < distortion score euclidiano
    (distortion = quanto as distâncias no embedding refletem
     as distâncias reais na hierarquia)

✗ FALHA se:
  → conceitos do mesmo nível têm depths muito diferentes (std > 0.3)
  → "gato" fica mais perto do centro que "mamífero"
  → a distância hiperbólica não separa melhor que cosine
```

---

## Fase 2 — Busca hiperbólica

**O que medir:**

```python
# Teste: dado um conceito abstrato como query,
# os resultados hiperbólicos são mais hierarquicamente corretos?

def recall_hierarquico(query, resultados, ground_truth_hierarquia):
    """
    ground_truth = lista ordenada por profundidade hierárquica real
    resultados   = o que o sistema retornou
  
    Mede: quantos dos top-K resultados estão no mesmo nível
    hierárquico que a query
    """
    corretos = sum(1 for r in resultados 
                   if mesmo_nivel_hierarquico(r, query, ground_truth_hierarquia))
    return corretos / len(resultados)

# Baseline (cosine/HNSW puro):     ~50% recall hierárquico
# Alvo (HNSW + re-rank hiperbólico): ~75%+
```

**Métricas concretas:**

```
✓ PASSA se:
  → recall hierárquico ≥ 75% (vs 50% baseline cosine)
  → latência de busca com re-rank ≤ 2x latência sem re-rank
  → top-1 result correto hierarquicamente em 80%+ das queries

✗ FALHA se:
  → ganho de recall < 10% sobre cosine puro
    (não vale a complexidade adicionada)
  → latência > 3x (re-rank matou o sistema em produção)
  → resultados hiperbólicos são só cosine reordenado
    (sem diferença semântica real)
```

---

## Fase 3 — Difusão multi-escala

**O que medir:**

```python
# Teste: t pequeno vs t grande produzem resultados diferentes?
# E os resultados fazem sentido semântico?

def teste_difusao(query_node, grafo, t_valores=[0.1, 1.0, 10.0]):
    resultados_por_t = {}
  
    for t in t_valores:
        ativados = heat_kernel(grafo, query_node, t)
        resultados_por_t[t] = ativados
  
    # Critério 1: resultados devem ser DIFERENTES entre escalas
    overlap_01_10 = jaccard(resultados_por_t[0.1], resultados_por_t[10.0])
  
    # Critério 2: t pequeno deve retornar vizinhos diretos
    # t grande deve retornar conceitos estruturalmente conectados
    # mas semanticamente distantes
  
    return overlap_01_10  # deve ser < 0.3 (pouco overlap entre escalas)

# Critério de performance:
# Chebyshev grau K=10 em grafo de 100k nós < 500ms
```

**Métricas concretas:**

```
✓ PASSA se:
  → overlap entre t=0.1 e t=10.0 < 30%
    (escalas diferentes realmente ativam regiões diferentes)
  → t pequeno retorna vizinhos diretos em 90%+ dos casos
  → t grande retorna pelo menos 1 nó a 3+ hops em 80%+ dos casos
  → latência Chebyshev K=10 < 500ms em grafos até 100k nós

✗ FALHA se:
  → todos os t retornam os mesmos resultados
    (difusão não está funcionando — só é busca com custo extra)
  → t grande não ativa nenhum nó além dos vizinhos imediatos
  → latência > 2s (inutilizável em produção)
```

---

## Fase 4 — L-System + Poda

**O que medir:**

```python
# Teste 1: o grafo cresce de forma fractal (não linear)?
def dimensao_hausdorff_global(grafo, epsilon_range):
    # box-counting no espaço de embeddings hiperbólicos
    # resultado esperado: entre 1.2 e 1.8 (fractal)
    # resultado ruim: ~1.0 (linear) ou ~2.0 (preenchimento completo)
    pass

# Teste 2: após N inserções, a dimensão se mantém estável?
dims_ao_longo_do_tempo = []
for checkpoint in [1000, 5000, 10000, 50000]:
    inserir_n_memorias(checkpoint)
    dims_ao_longo_do_tempo.append(dimensao_hausdorff_global(grafo))

# Critério: std(dims) < 0.15 (dimensão estável = auto-similaridade)

# Teste 3: poda está removendo os nós certos?
nos_antes_poda = grafo.nos_com_baixa_hausdorff_local()
podar(grafo)
# Critério: 95%+ dos nós removidos estavam na lista de baixa Hausdorff
```

**Métricas concretas:**

```
✓ PASSA se:
  → dimensão de Hausdorff global: 1.2 < D < 1.8
    (fractal real — nem linear nem preenchido)
  → std da dimensão ao longo do tempo < 0.15
    (auto-similaridade preservada com crescimento)
  → poda remove exatamente os nós de baixa complexidade local
  → após poda, dimensão global muda < 5%
    (poda não destrói a estrutura fractal)
  → ratio nós/arestas se mantém em banda estável ±20%

✗ FALHA se:
  → dimensão < 1.0 (grafo colapsou)
  → dimensão > 1.9 (grafo virou massa densa, perdeu estrutura)
  → poda remove nós aleatoriamente (sem correlação com Hausdorff)
  → grafo explode exponencialmente (L-System sem inibição suficiente)
```

---

## Fase 5 — Ciclo de sono

**O que medir:**

```python
# Teste: o sistema muda MAS mantém identidade?

def ciclo_sono_completo(grafo, manifold):
    # Snapshot antes
    dim_antes = hausdorff_amostral(grafo, n_amostras=1000)
    estrutura_antes = fingerprint_estrutural(grafo)
    # (fingerprint = distribuição de graus + distribuição de depths)
  
    # Executa sono
    reconsolidar(grafo, manifold)
  
    # Snapshot depois
    dim_depois = hausdorff_amostral(grafo, n_amostras=1000)
    estrutura_depois = fingerprint_estrutural(grafo)
  
    delta_dim = abs(dim_antes - dim_depois) / dim_antes
    delta_estrutura = distancia_fingerprint(estrutura_antes, estrutura_depois)
  
    return delta_dim, delta_estrutura

# Critérios:
# delta_dim < 5%          → identidade fractal preservada
# delta_estrutura < 20%   → estrutura mudou mas não colapsou
# delta_estrutura > 1%    → alguma coisa realmente mudou (não foi no-op)
```

**Métricas concretas:**

```
✓ PASSA se:
  → dimensão de Hausdorff antes/depois difere < 5%
  → fingerprint estrutural muda entre 1% e 20%
    (mudou, mas não colapsou — a janela é o critério)
  → pelo menos 30% dos nós do subgrafo amostrado tiveram
    embeddings atualizados (sono não foi no-op)
  → rollback acontece em < 10% dos ciclos
    (sistema é estável, não fica desfazendo tudo)
  → após 10 ciclos consecutivos, o sistema não diverge

✗ FALHA se:
  → delta_dim > 15% (perdeu identidade fractal)
  → delta_estrutura < 0.5% (sono não fez nada — é placebo)
  → rollback > 50% dos ciclos (sistema instável)
  → após 10 ciclos, dimensão está em queda monotônica
    (sistema está morrendo lentamente)
```

---

## Dashboard de saúde — o que monitorar sempre

```python
# Métricas que rodam continuamente, não só nos testes de fase

METRICAS_CONTINUAS = {
    # Geometria
    "depth_mean":          "média das profundidades — não deve derivar",
    "depth_std":           "diversidade hierárquica — não deve colapsar",
  
    # Grafo
    "hausdorff_global":    "identidade fractal — banda 1.2-1.8",
    "ratio_nos_arestas":   "densidade — banda ±20% do baseline",
    "nos_podados_por_dia": "poda está ativa — não deve ser zero",
  
    # Busca
    "recall_hierarquico":  "qualidade da busca — ≥ 75%",
    "latencia_p99":        "performance — ≤ 500ms",
  
    # Sono
    "delta_dim_por_ciclo": "estabilidade — < 5%",
    "rollback_rate":       "saúde do ciclo — < 10%",
}
```

---

## Resumo — o que você tem agora

```
Fase 1: critério claro    → 90% triplas hierárquicas corretas
Fase 2: critério claro    → recall hierárquico ≥ 75%
Fase 3: critério claro    → overlap entre escalas < 30%
Fase 4: critério claro    → Hausdorff 1.2-1.8, std < 0.15
Fase 5: critério claro    → delta_dim < 5%, delta_estrutura 1-20%
```

Agora você pode implementar cada fase **sabendo exatamente quando ela passou**. Quer começar a Fase 1?

---

## Fases de Agency Autónoma (Phases IX–XXIV)

Estas fases transformam o NietzscheDB de um banco reativo num **sistema cognitivo autónomo**.

### Phase IX — Geometric Self (Self Geométrico)
- `CentroidGuardian`: mantém centroide da civilização via Fréchet mean
- `MaturityEvaluator`: promove embeddings estáveis a Axiomas
- `AxiomRegistry`: DashMap L1 + RocksDB L2 para queries O(1)

### Phase X — Hyperbolic Health Monitor
- Radial Density Entropy (RDE): H = -Σ pᵢ ln(pᵢ)
- Detecção de Boundary Crowding e Angular Variance Collapse
- Semantic Attractors ("Black Holes" via bin density > 3μ)

### Phase XII — ECAN + Hebbian LTP
- Attention Economy: bid/ask free-market para alocação de foco
- Hebbian LTP: "nodes that fire together, wire together"

### Phase XIII — Cognitive Thermodynamics
- Temperature T = σ_E / μ_E
- Phase States: Solid, Liquid, Gas, Critical
- Fourier's Law: q_ij = κ·(Eᵢ - Eⱼ)

### Phase XIV — Semantic Gravity
- Semantic Mass: M = E × ln(degree + 1)
- Gravitational force: F_ij = G·Mᵢ·Mⱼ / d²

### Phase XV — Observability (DirtySet + Dashboard)
- CognitiveDashboard: unified JSON snapshot de todos os subsistemas
- DirtySet: O(N) → O(Δ) via temporal adaptive sampling
- ObservationBridge: temperatura cognitiva → HSV/RGB para WebGL

### Phase XVI — Shatter Protocol
```
✓ PASSA se:
  → super-nodes (grau > threshold) são detectados e particionados
  → avatars herdam subconjuntos de edges corretamente
  → phantom ratio não ultrapassa 20% do grafo
  → grau máximo pós-shatter < threshold

✗ FALHA se:
  → edges são perdidas durante o split
  → avatars ficam desconectados do grafo
  → phantom ratio > 30% (grafo fantasma)
```

### Phase XVII — Ego-Cache
```
✓ PASSA se:
  → cache hit rate > 80% para nós de alta frequência
  → latência de acesso cached < 1μs
  → eviction policy mantém working set correto

✗ FALHA se:
  → cache hit rate < 50% (cache inútil)
  → memory footprint > 10% do grafo total
```

### Phase XVIII — Reasoning Engine
```
✓ PASSA se:
  → chains multi-hop (3+ hops) resolvem em < 50ms
  → regras dedutivas produzem inferências corretas
  → ciclos no grafo não causam loops infinitos

✗ FALHA se:
  → reasoning produz contradições não detectadas
  → latência > 500ms para chains simples (2 hops)
```

### Phase XIX — Self-Healing Graph
```
✓ PASSA se:
  → boundary drift detectado e corrigido (nodes retraídos para ||x|| < 1)
  → orphan nodes reconectados ou removidos
  → dead edges (apontando para nodes inexistentes) limpas
  → phantom ratio monitorado e alertas emitidos > 20%
  → exhausted nodes (energy ≈ 0) identificados

✗ FALHA se:
  → healing introduz novos problemas (edges inválidas)
  → false positives > 10% (nodes saudáveis marcados como doentes)
  → healing não detecta problemas conhecidos
```

### Phase XX — Graph Learning Engine
```
✓ PASSA se:
  → access hotspots identificam top-10 nodes mais acedidos
  → mutation hotspots rastreiam nodes com mais modificações
  → sector growth rates detectam regiões em expansão
  → rolling window mantém memória bounded (eviction funciona)
  → decay exponencial previne obsolescência de dados antigos

✗ FALHA se:
  → hotspots são estáticos (não refletem padrões recentes)
  → memória cresce linearmente com o tempo (eviction falhou)
  → growth rates não detectam mudanças abruptas
```

### Phase XXI — Knowledge Compression
```
✓ PASSA se:
  → near-duplicates (d < ε) detectados com precision > 90%
  → stale clusters (energy < threshold, degree < limit) agrupados
  → redundant paths (edges paralelas) identificadas
  → MergeProposals têm confidence scores calibrados
  → archetype selection preserva o node mais representativo

✗ FALHA se:
  → merges destróem informação semântica única
  → false positive rate > 5% (nodes distintos marcados como duplicatas)
  → compression não reduz o grafo em pelo menos 5%
```

### Phase XXII — Hyperbolic Sharding
```
✓ PASSA se:
  → partição respeita geometria Poincaré (radial bands + angular sectors)
  → imbalance ratio < 2.0 (shards razoavelmente equilibrados)
  → rebalance recommendation emitida quando imbalance > threshold
  → nodes assignados ao shard correto baseado em norm e atan2

✗ FALHA se:
  → sharding ignora profundidade hiperbólica (trata como Euclidiano)
  → imbalance ratio > 5.0 (shards completamente desbalanceados)
  → empty shards > 50% (particionamento excessivo)
```

### Phase XXIII — World Model Graph
```
✓ PASSA se:
  → rolling statistics (query_rate, mutation_rate) convergem em < 100 observações
  → anomalias detectadas quando z-score > sensitivity
  → quiet periods identificados (rate < 0.3× mean)
  → busy periods identificados (rate > 2× mean)
  → trend() indica direção correta da tendência

✗ FALHA se:
  → anomaly detection tem false positive rate > 15%
  → rolling window não decai (memória cresce infinitamente)
  → quiet/busy detection falha em padrões óbvios
```

### Phase XXIV — Cognitive Flywheel
```
✓ PASSA se:
  → momentum EMA converge e reflete saúde real do sistema
  → per-subsystem assessment (Healthy/Active/Degraded/Inactive) correto
  → recommendations geradas para subsistemas Degraded
  → flywheel "spins" (momentum > min_threshold) quando sistema saudável
  → healthy_streak incrementa corretamente

✗ FALHA se:
  → momentum não converge (oscila indefinidamente)
  → subsystem status não reflete estado real
  → recommendations são genéricas/inúteis
  → flywheel nunca spins mesmo com sistema saudável
```

---

## Resumo expandido

```
Fase 1:   critério claro → 90% triplas hierárquicas corretas
Fase 2:   critério claro → recall hierárquico ≥ 75%
Fase 3:   critério claro → overlap entre escalas < 30%
Fase 4:   critério claro → Hausdorff 1.2-1.8, std < 0.15
Fase 5:   critério claro → delta_dim < 5%, delta_estrutura 1-20%
Phase IX:  implementado  → CentroidGuardian + MaturityEvaluator + AxiomRegistry
Phase X:   implementado  → HyperbolicHealthMonitor (RDE + attractors)
Phase XII: implementado  → ECAN + Hebbian LTP
Phase XIII: implementado → Cognitive Thermodynamics (T, S, F)
Phase XIV: implementado  → Semantic Gravity (M, F_ij)
Phase XV:  implementado  → CognitiveDashboard + DirtySet + ObservationBridge
Phase XVI: implementado  → Shatter Protocol (super-node splitting)
Phase XVII: implementado → Ego-Cache (identity hot-tier)
Phase XVIII: implementado→ Reasoning Engine (multi-hop inference)
Phase XIX: implementado  → Self-Healing Graph (autonomous repair)
Phase XX:  implementado  → Graph Learning Engine (hotspots + growth)
Phase XXI: implementado  → Knowledge Compression (semantic merge)
Phase XXII: implementado → Hyperbolic Sharding (Poincaré partitions)
Phase XXIII: implementado→ World Model Graph (environmental awareness)
Phase XXIV: implementado → Cognitive Flywheel (unified feedback loop)
```
