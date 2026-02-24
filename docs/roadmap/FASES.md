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
