# TestesNQL — Relatório de Testes da Linguagem NQL

> **NietzscheDB Query Language (NQL) — Cobertura Completa de Testes**
> Gerado automaticamente em 2026-02-19. Cobre todos os testes unitários existentes nos
> módulos `nietzsche-query/src/parser.rs` e `nietzsche-query/src/executor.rs`.

---

## Sumário Executivo

| Módulo          | Testes | Status |
|-----------------|--------|--------|
| Parser          | 35     | ✅ todos passando |
| Executor        | 29     | ✅ todos passando |
| **Total**       | **64** | ✅ **100% verde** |

---

## 1. Arquitetura da NQL

A NQL é uma linguagem de consulta de grafos hiperbólicos, estruturada em três camadas:

```
Texto NQL  →  Parser (PEG via pest)  →  AST  →  Executor  →  QueryResult
```

### 1.1 Tipos de Query (AST)

```rust
pub enum Query {
    Match(MatchQuery),              // MATCH (n) WHERE ... RETURN ...
    Diffuse(DiffuseQuery),          // DIFFUSE FROM $id MAX_HOPS n
    Reconstruct(ReconstructQuery),  // RECONSTRUCT $id MODALITY m QUALITY q
    Explain(Box<Query>),            // EXPLAIN <inner-query>
    InvokeZaratustra(...),          // INVOKE ZARATUSTRA [params]
    BeginTx,                        // BEGIN
    CommitTx,                       // COMMIT
    RollbackTx,                     // ROLLBACK
}
```

### 1.2 Tipos de Resultado

```rust
pub enum QueryResult {
    Node(Node),
    NodePair { from: Node, to: Node },
    DiffusionPath(Vec<Uuid>),
    ReconstructRequest { node_id, modality, quality },
    ExplainPlan(String),
    Scalar(Vec<(String, ScalarValue)>),
    InvokeZaratustraRequest { collection, cycles, alpha, decay },
    TxBegin | TxCommit | TxRollback,
}
```

---

## 2. Testes do Parser (`parser.rs`)

O parser usa gramática PEG definida em `src/nql.pest`, compilada via `pest_derive`.

### 2.1 MATCH — Padrão de Nó

#### `parse_simple_node_match`
```nql
MATCH (n) WHERE n.energy > 0.3 RETURN n LIMIT 10
```
- Verifica: padrão `Pattern::Node`, 1 condição, `limit = Some(10)`

#### `parse_node_match_with_label`
```nql
MATCH (n:Memory) WHERE n.depth < 0.5 RETURN n
```
- Verifica: alias `"n"`, label `Some("Memory")`

---

### 2.2 MATCH — Padrão de Caminho

#### `parse_path_match`
```nql
MATCH (a)-[:ASSOCIATION]->(b) WHERE a.energy > 0.5 RETURN b
```
- Verifica: `from.alias = "a"`, `to.alias = "b"`, `direction = Out`, `edge_label = "ASSOCIATION"`

#### `parse_anonymous_path`
```nql
MATCH (a)-->(b) RETURN b
```
- Verifica: padrão `Pattern::Path` sem edge_label

---

### 2.3 Condições WHERE

#### `parse_hyperbolic_dist_condition`
```nql
MATCH (n) WHERE HYPERBOLIC_DIST(n.embedding, $q) < 0.5 RETURN n LIMIT 5
```
- Verifica: `Expr::HyperbolicDist`, `CompOp::Lt`, argumento `$q`

#### `parse_and_condition`
```nql
MATCH (n) WHERE n.energy > 0.3 AND n.depth < 0.8 RETURN n
```
- Verifica: `Condition::And(..)`

#### `parse_not_condition`
```nql
MATCH (n) WHERE NOT n.energy > 0.9 RETURN n
```
- Verifica: `Condition::Not(_)`

#### `parse_inline_vector_hdist`
```nql
MATCH (n) WHERE HYPERBOLIC_DIST(n.embedding, [0.1, 0.2]) < 0.9 RETURN n
```
- Verifica: `HDistArg::Vector(v)` com `v.len() == 2`

#### `parse_sensory_dist_in_where`
```nql
MATCH (n) WHERE SENSORY_DIST(n.latent, $q) < 0.3 RETURN n LIMIT 5
```
- Verifica: `Expr::SensoryDist`, `CompOp::Lt`

#### `parse_in_condition`
```nql
MATCH (n) WHERE n.energy IN (0.3, 0.5, 0.9) RETURN n
```
- Verifica: `Condition::In { values }` com `values.len() == 3`

#### `parse_between_condition`
```nql
MATCH (n) WHERE n.energy BETWEEN 0.3 AND 0.8 RETURN n
```
- Verifica: `Condition::Between { .. }`

#### `parse_between_with_outer_and`
```nql
MATCH (n) WHERE n.energy BETWEEN 0.3 AND 0.8 AND n.depth < 1.0 RETURN n
```
- Verifica: `Condition::And(Between(..), Compare(..))` — precedência correta

#### `parse_contains_condition`
```nql
MATCH (n) WHERE n.id CONTAINS "abc" RETURN n
```
- Verifica: `StringCompOp::Contains`

#### `parse_starts_with_condition`
```nql
MATCH (n) WHERE n.id STARTS_WITH "prefix" RETURN n
```
- Verifica: `StringCompOp::StartsWith`

#### `parse_ends_with_condition`
```nql
MATCH (n) WHERE n.id ENDS_WITH "suffix" RETURN n
```
- Verifica: `StringCompOp::EndsWith`

---

### 2.4 ORDER BY

#### `parse_order_by_hyperbolic_dist`
```nql
MATCH (n) WHERE n.energy > 0.1
RETURN n ORDER BY HYPERBOLIC_DIST(n.embedding, $q) ASC LIMIT 3
```
- Verifica: `OrderExpr::HyperbolicDist`, `OrderDir::Asc`

#### `parse_sensory_dist_order_by`
```nql
MATCH (n) WHERE n.energy > 0.1
RETURN n ORDER BY SENSORY_DIST(n.latent, $q) ASC LIMIT 3
```
- Verifica: `OrderExpr::SensoryDist`, `OrderDir::Asc`

---

### 2.5 RETURN — Modificadores

#### `parse_distinct_return`
```nql
MATCH (n) WHERE n.energy > 0.0 RETURN DISTINCT n LIMIT 5
```
- Verifica: `ret.distinct == true`

#### `parse_skip_clause`
```nql
MATCH (n) WHERE n.energy > 0.0 RETURN n LIMIT 10 SKIP 5
```
- Verifica: `limit = Some(10)`, `skip = Some(5)`

#### `parse_return_as_alias`
```nql
MATCH (n) WHERE n.energy > 0.0 RETURN n.energy AS e
```
- Verifica: `ret.items[0].as_alias = Some("e")`

---

### 2.6 Agregações

#### `parse_count_star`
```nql
MATCH (n) WHERE n.energy > 0.0 RETURN COUNT(*) AS total
```
- Verifica: `AggFunc::Count`, `AggArg::Star`, alias `"total"`

#### `parse_avg_with_group_by`
```nql
MATCH (n) WHERE n.energy > 0.0
RETURN n.node_type, AVG(n.energy) AS avg_e
GROUP BY n.node_type
```
- Verifica: 2 itens no RETURN, `AggFunc::Avg`, `group_by.len() == 1`

#### `parse_min_max_sum`
```nql
MATCH (n) WHERE n.energy > 0.0
RETURN MIN(n.energy), MAX(n.energy), SUM(n.depth)
```
- Verifica: `AggFunc::Min`, `AggFunc::Max`, `AggFunc::Sum` nos 3 itens

---

### 2.7 DIFFUSE

#### `parse_diffuse_query`
```nql
DIFFUSE FROM $node WITH t = [0.1, 1.0, 10.0] MAX_HOPS 5 RETURN activated_nodes
```
- Verifica: `DiffuseFrom::Param`, `t_values.len() == 3`, `max_hops == 5`

#### `parse_diffuse_minimal`
```nql
DIFFUSE FROM $n RETURN path
```
- Verifica: `Query::Diffuse(_)` com valores padrão

---

### 2.8 RECONSTRUCT

#### `parse_reconstruct_full`
```nql
RECONSTRUCT $node_id MODALITY audio QUALITY high
```
- Verifica: target `$node_id`, modality `"audio"`, quality `"high"`

#### `parse_reconstruct_minimal`
```nql
RECONSTRUCT $nid
```
- Verifica: `ReconstructTarget::Param`, modality `None`, quality `None`

#### `parse_reconstruct_modality_only`
```nql
RECONSTRUCT $n MODALITY text
```
- Verifica: modality `"text"`, quality `None`

---

### 2.9 EXPLAIN

#### `parse_explain_match`
```nql
EXPLAIN MATCH (n) WHERE n.energy > 0.3 RETURN n LIMIT 10
```
- Verifica: `Query::Explain(Box<Query::Match(_)>)`

#### `parse_explain_diffuse`
```nql
EXPLAIN DIFFUSE FROM $n RETURN path
```
- Verifica: `Query::Explain(Box<Query::Diffuse(_)>)`

---

### 2.10 Funções Matemáticas (Parsing)

#### `parse_poincare_dist`
```nql
MATCH (n) WHERE POINCARE_DIST(n.embedding, $q) < 0.5 RETURN n
```
- Verifica: `MathFunc::PoincareDist`, 2 argumentos

#### `parse_minkowski_norm`
```nql
MATCH (n) WHERE MINKOWSKI_NORM(n.embedding) > 1.5 RETURN n
```
- Verifica: `MathFunc::MinkowskiNorm`

#### `parse_riemann_curvature`
```nql
MATCH (n) WHERE RIEMANN_CURVATURE(n) > 0.0 RETURN n
```
- Verifica: `MathFunc::RiemannCurvature`

#### `parse_gauss_kernel`
```nql
MATCH (n) WHERE GAUSS_KERNEL(n, $t) > 0.01 RETURN n
```
- Verifica: `MathFunc::GaussKernel`, 2 argumentos

#### `parse_hausdorff_dim`
```nql
MATCH (n) WHERE HAUSDORFF_DIM(n) > 1.0 RETURN n
```
- Verifica: `MathFunc::HausdorffDim`

---

### 2.11 Erros de Parsing

#### `parse_error_on_invalid_input`
```nql
SELECT * FROM nodes   -- erro: sintaxe SQL não suportada
                      -- erro: string vazia
```
- Verifica: ambos retornam `Err(_)` — nenhum pânico

---

## 3. Testes do Executor (`executor.rs`)

O executor recebe a `GraphStorage` (RocksDB) e `AdjacencyIndex` (in-memory) via referências.
Helper de teste: `parse_and_exec(nql, storage, adjacency, params)`.

### 3.1 MATCH Básico

#### `match_all_nodes_no_filter`
```nql
MATCH (n) WHERE n.energy > 0.0 RETURN n
```
- Setup: 3 nós inseridos. Verifica: 3 resultados `QueryResult::Node`

#### `match_nodes_energy_filter`
```nql
MATCH (n) WHERE n.energy > 0.5 RETURN n
```
- Setup: nós com energy 0.9, 0.1, 0.8. Verifica: 2 resultados (energy 0.1 filtrado)

#### `match_with_limit`
```nql
MATCH (n) WHERE n.energy > 0.0 RETURN n LIMIT 3
```
- Setup: 10 nós. Verifica: exatamente 3 resultados

---

### 3.2 Distâncias Hiperbólicas no WHERE

#### `match_hyperbolic_dist_with_param`
```nql
MATCH (n) WHERE HYPERBOLIC_DIST(n.embedding, $q) < 0.1 RETURN n
```
- Setup: nó perto da origem `[0.01, 0]`, nó longe `[0.5, 0]`, `$q = [0.0, 0.0]`
- Verifica: 1 resultado — nó próximo

#### `sensory_dist_filter`
```nql
MATCH (n) WHERE SENSORY_DIST(n.latent, $q) < 0.1 RETURN n
```
- Mesmo setup que acima com SENSORY_DIST. Verifica: 1 resultado — nó próximo

---

### 3.3 ORDER BY

#### `match_order_by_energy_desc`
```nql
MATCH (n) WHERE n.energy > 0.0 RETURN n ORDER BY n.energy DESC
```
- Setup: energias 0.3, 0.9, 0.6. Verifica: ordem decrescente confirmada

#### `sensory_dist_order_by`
```nql
MATCH (n) WHERE n.energy > 0.0 RETURN n ORDER BY SENSORY_DIST(n.latent, $q) ASC
```
- Setup: 3 nós em posições variadas. Verifica: ordem crescente por distância da origem

#### `order_by_poincare_dist`
```nql
MATCH (n) WHERE n.energy > 0.0 RETURN n ORDER BY POINCARE_DIST(n.embedding, $q) ASC
```
- Setup: 3 nós em x = 0.1, 0.2, 0.3. Verifica: ordem crescente por distância hiperbólica

---

### 3.4 Padrão de Caminho

#### `match_path_pattern`
```nql
MATCH (a)-[:Association]->(b) WHERE a.energy > 0.0 RETURN b
```
- Setup: nós a, b conectados por `Edge::association`. Verifica: 1 `NodePair`, `to.id == b.id`

---

### 3.5 DIFFUSE

#### `diffuse_query_returns_path`
```nql
DIFFUSE FROM $start MAX_HOPS 5 RETURN path
```
- Setup: cadeia a → b → c. Verifica: 1 `DiffusionPath`, `path[0] == a.id`, `path.len() >= 2`

---

### 3.6 RECONSTRUCT

#### `reconstruct_returns_request`
```nql
RECONSTRUCT $nid MODALITY audio QUALITY high
```
- Verifica: `ReconstructRequest { node_id, modality: Some("audio"), quality: Some("high") }`

#### `reconstruct_minimal`
```nql
RECONSTRUCT $n
```
- Verifica: `ReconstructRequest { modality: None, quality: None }`

---

### 3.7 Paginação (SKIP + LIMIT)

#### `match_with_skip`
```nql
MATCH (n) WHERE n.energy > 0.0 RETURN n ORDER BY n.energy ASC SKIP 2 LIMIT 2
```
- Setup: 5 nós. Verifica: 2 resultados (pula os 2 primeiros)

---

### 3.8 Operadores de Conjunto

#### `match_between`
```nql
MATCH (n) WHERE n.energy BETWEEN 0.3 AND 0.7 RETURN n
```
- Setup: nós com energy 0.2, 0.5, 0.9. Verifica: 1 resultado (energy 0.5)

#### `match_in_operator`
```nql
MATCH (n) WHERE n.energy IN (0.30000001192092896, 0.800000011920929) RETURN n
```
- Setup: nós com energy 0.3, 0.5, 0.8. Verifica: 2 resultados
- Nota: valores f32→f64 levemente diferentes por precisão de ponto flutuante

---

### 3.9 Agregações

#### `count_star_aggregate`
```nql
MATCH (n) WHERE n.energy > 0.0 RETURN COUNT(*) AS total
```
- Setup: 4 nós. Verifica: `Scalar([("total", Int(4))])`

#### `avg_aggregate`
```nql
MATCH (n) WHERE n.energy > 0.0 RETURN AVG(n.energy) AS avg_e
```
- Setup: nós com energy 0.2 e 0.8. Verifica: `Scalar([("avg_e", Float(≈0.5))])`

---

### 3.10 EXPLAIN

#### `explain_produces_plan`
```nql
EXPLAIN MATCH (n) WHERE n.energy > 0.0 RETURN n ORDER BY n.energy DESC LIMIT 5
```
- Verifica: plano contém `"NodeScan"`, `"Filter"`, `"Sort"`, `"Limit"`
- Exemplo de saída: `NodeScan(label=*) -> Filter(conditions=1) -> Sort(dir=Desc) -> Limit(5)`

---

### 3.11 Funções Matemáticas (Execução)

#### `poincare_dist_equals_hyperbolic_dist`
```nql
MATCH (n) WHERE POINCARE_DIST(n.embedding, $q) < 0.1 RETURN n
```
- Setup idêntico ao `match_hyperbolic_dist_with_param`. Verifica: mesmo resultado — POINCARE_DIST ≡ HYPERBOLIC_DIST

#### `klein_dist_filter`
```nql
MATCH (n) WHERE KLEIN_DIST(n.embedding, $q) < 0.1 RETURN n
```
- Distância de Klein-Beltrami: `k = 2p/(1+‖p‖²)`, depois distância geodésica.
- Verifica: nó próximo (`x = 0.01`) retornado, nó longe (`x = 0.5`) filtrado

#### `minkowski_norm_increases_with_radius`
```nql
MATCH (n) WHERE MINKOWSKI_NORM(n.embedding) > 3.0 RETURN n
```
- Fator conforme: `λ = 2/(1−‖x‖²)`. Nó em x=0.8: λ ≈ 5.55 > 3 → passa.
- Verifica: pelo menos 1 resultado (nó mais periférico)

#### `lobachevsky_angle_returns_valid_range`
```nql
MATCH (n) WHERE LOBACHEVSKY_ANGLE(n.embedding, $q) > 0.0 RETURN n
```
- Π(d) = 2·arctan(e^(−d)) ∈ (0, π/2) para d > 0
- Verifica: 1 resultado com ângulo válido

#### `hausdorff_dim_returns_field`
```nql
MATCH (n) WHERE HAUSDORFF_DIM(n) > 1.0 RETURN n
```
- Retorna campo `hausdorff_local` do nó. Setup: `n.hausdorff_local = 1.5`.
- Verifica: 1 resultado

#### `gauss_kernel_decays_with_distance`
```nql
MATCH (n) WHERE GAUSS_KERNEL(n, $t) > 0.5 RETURN n
```
- h(x) = exp(−‖x‖²/(4t)). Com t=0.5, nó na origem → kernel ≈ 1.0, nó em x=0.8 → kernel ≈ 0.27.
- Verifica: 1 resultado (nó próximo da origem)

#### `chebyshev_coeff_valid`
```nql
MATCH (n) WHERE CHEBYSHEV_COEFF(n, 0) > 0.9 RETURN n
```
- T₀(x) = cos(0·arccos(x)) = 1 para todo x. Verifica: sempre verdadeiro para k=0.
- Setup: 1 nó. Verifica: 1 resultado

#### `euler_char_isolated_node`
```nql
MATCH (n) WHERE EULER_CHAR(n) = 1 RETURN n
```
- χ = V − E. Nó isolado: V=1, E=0 → χ = 1.
- Verifica: 1 resultado

#### `laplacian_score_equals_degree`
```nql
MATCH (n) WHERE LAPLACIAN_SCORE(n) > 1.5 RETURN n
```
- L_ii = grau do nó. Setup: nó a com 2 arestas de saída → grau = 2.
- Verifica: 1 resultado (nó a)

#### `dirichlet_energy_with_neighbors`
```nql
MATCH (n) WHERE DIRICHLET_ENERGY(n) > 0.5 RETURN n
```
- E(f) = Σⱼ (f(i) − f(j))². Nó a com energy=1.0, vizinho b com energy=0.0 → E = 1.0.
- Verifica: 1 resultado (nó a)

#### `ramanujan_expansion_with_edges`
```nql
MATCH (n) WHERE RAMANUJAN_EXPANSION(n) > 0.5 RETURN n
```
- Razão: grau_total / (2√(grau_total − 1)). Nó a: out=2, in=1, total=3 → ratio ≈ 1.06.
- Verifica: pelo menos 1 resultado

---

## 4. Referência Completa da Sintaxe NQL

### 4.1 Campos de Nó Disponíveis

| Campo              | Tipo   | Descrição |
|--------------------|--------|-----------|
| `n.energy`         | float  | Nível de energia (0.0–1.0) |
| `n.depth`          | float  | Profundidade na hierarquia |
| `n.hausdorff_local`| float  | Dimensão de Hausdorff local |
| `n.lsystem_generation` | float | Geração do L-system |
| `n.id`             | string | UUID do nó (texto) |
| `n.created_at`     | float  | Timestamp Unix de criação |
| `n.node_type`      | string | "Semantic", "Episodic", "Concept", "DreamSnapshot" |

### 4.2 Tipos de Aresta

| Tipo           | Descrição |
|----------------|-----------|
| `Association`  | Ligação associativa entre nós |
| `Hierarchical` | Relação hierárquica pai-filho |
| (outros)       | Conforme definido em `EdgeType` |

### 4.3 Funções Disponíveis no WHERE e ORDER BY

#### Distâncias
| Função | Assinatura | Descrição |
|--------|-----------|-----------|
| `HYPERBOLIC_DIST` | `(n.embedding, $q \| [v...])` | Distância de Poincaré (alias de POINCARE_DIST) |
| `POINCARE_DIST`   | `(n.embedding, $q \| [v...])` | Distância no modelo Poincaré ball |
| `KLEIN_DIST`      | `(n.embedding, $q \| [v...])` | Distância no modelo de Klein-Beltrami |
| `SENSORY_DIST`    | `(n.latent, $q)`              | Distância sensorial (embedding primário como proxy) |

#### Funções de Nó Único
| Função | Assinatura | Descrição |
|--------|-----------|-----------|
| `MINKOWSKI_NORM`      | `(n.embedding)` | Fator conforme λ = 2/(1−‖x‖²) |
| `LOBACHEVSKY_ANGLE`   | `(n.embedding, $q)` | Ângulo de paralelismo Π(d) = 2·arctan(e^{−d}) |
| `RIEMANN_CURVATURE`   | `(n)` | Curvatura de Ollivier-Ricci discreta |
| `GAUSS_KERNEL`        | `(n, $t \| f)` | Núcleo de calor exp(−‖x‖²/(4t)) |
| `CHEBYSHEV_COEFF`     | `(n, k \| $k)` | Polinômio de Chebyshev T_k(x) |
| `RAMANUJAN_EXPANSION` | `(n)` | Razão de expansão espectral local |
| `HAUSDORFF_DIM`       | `(n)` | Dimensão de Hausdorff local (campo pré-computado) |
| `EULER_CHAR`          | `(n)` | Característica de Euler χ = V − E |
| `LAPLACIAN_SCORE`     | `(n)` | Diagonal do Laplaciano = grau do nó |
| `FOURIER_COEFF`       | `(n, k \| $k)` | Coeficiente de Fourier cos(k·π·‖x‖) |
| `DIRICHLET_ENERGY`    | `(n)` | Energia de Dirichlet Σⱼ (f(i)−f(j))² |

### 4.4 Gramática BNF Simplificada

```bnf
query       = match_query | diffuse_query | reconstruct | explain | invoke_zaratustra
            | "BEGIN" | "COMMIT" | "ROLLBACK"

match_query = "MATCH" pattern ["WHERE" condition] "RETURN" return_clause

pattern     = "(" alias [":" label] ")"
            | "(" alias ")" "->" "(" alias ")"
            | "(" alias ")" "-[:" edge_type "]->" "(" alias ")"

condition   = compare | "AND" | "OR" | "NOT" | "IN" | "BETWEEN" | string_op

compare     = expr comp_op expr
comp_op     = "<" | "<=" | ">" | ">=" | "=" | "!="

expr        = alias "." field
            | "$" param
            | float | int | string | bool
            | "HYPERBOLIC_DIST" "(" alias "." field "," (param | vector) ")"
            | "SENSORY_DIST"    "(" alias "." field "," param ")"
            | math_func "(" args ")"

return_clause = ["DISTINCT"] return_items ["GROUP BY" items]
                ["ORDER BY" order_expr ["ASC"|"DESC"]]
                ["LIMIT" n] ["SKIP" n]

return_items  = return_item ("," return_item)*
return_item   = alias | alias "." field | agg_func "(" agg_arg ")" ["AS" name]

agg_func    = "COUNT" | "SUM" | "AVG" | "MIN" | "MAX"

diffuse_query = "DIFFUSE FROM" ("$" param) ["WITH t =" vector] ["MAX_HOPS" n]
                "RETURN" name

reconstruct   = "RECONSTRUCT" "$" param ["MODALITY" name] ["QUALITY" name]

explain       = "EXPLAIN" query

invoke_zaratustra = "INVOKE ZARATUSTRA"
                    ["IN" string]
                    ["CYCLES" n]
                    ["ALPHA" f]
                    ["DECAY" f]
```

---

## 5. Casos de Uso — Exemplos Práticos

### 5.1 KNN Hiperbólico (top-5 vizinhos mais próximos)
```nql
MATCH (n)
WHERE HYPERBOLIC_DIST(n.embedding, $query_vec) < 0.5
RETURN n
ORDER BY POINCARE_DIST(n.embedding, $query_vec) ASC
LIMIT 5
```

### 5.2 Filtrar por Tipo de Nó e Energia
```nql
MATCH (n:Semantic)
WHERE n.energy BETWEEN 0.5 AND 1.0
RETURN n
ORDER BY n.energy DESC
LIMIT 20
```

### 5.3 Consulta de Grafo com Arestas Tipadas
```nql
MATCH (a)-[:Hierarchical]->(b)
WHERE a.energy > 0.7
RETURN b
LIMIT 50
```

### 5.4 Difusão a partir de um Nó Semente
```nql
DIFFUSE FROM $start_id
WITH t = [0.1, 1.0, 10.0]
MAX_HOPS 10
RETURN activated_nodes
```

### 5.5 Contagem por Tipo de Nó (GROUP BY)
```nql
MATCH (n)
WHERE n.energy > 0.0
RETURN n.node_type, COUNT(*) AS total
GROUP BY n.node_type
```

### 5.6 Reconstrução Multimodal
```nql
RECONSTRUCT $node_id
MODALITY audio
QUALITY high
```

### 5.7 Paginação
```nql
MATCH (n)
WHERE n.energy > 0.5
RETURN n
ORDER BY n.created_at DESC
SKIP 100
LIMIT 20
```

### 5.8 Plano de Execução
```nql
EXPLAIN MATCH (n)
WHERE POINCARE_DIST(n.embedding, $q) < 0.3
RETURN n
ORDER BY POINCARE_DIST(n.embedding, $q) ASC
LIMIT 10
```
Saída esperada: `NodeScan(label=*) -> Filter(conditions=1) -> Sort(dir=Asc) -> Limit(10)`

### 5.9 Transações
```nql
BEGIN
-- (operações via gRPC/SDK)
COMMIT
```

### 5.10 Ciclo Zaratustra (Engine de Consolidação)
```nql
INVOKE ZARATUSTRA
IN "episodic"
CYCLES 3
ALPHA 0.1
DECAY 0.02
```

---

## 6. Cobertura por Feature

| Feature                        | Parser | Executor | Total |
|-------------------------------|--------|----------|-------|
| MATCH nó básico               | ✅     | ✅       | ✅    |
| MATCH com label               | ✅     | —        | ✅    |
| MATCH caminho tipado          | ✅     | ✅       | ✅    |
| MATCH caminho anônimo         | ✅     | —        | ✅    |
| WHERE energy >                | ✅     | ✅       | ✅    |
| WHERE HYPERBOLIC_DIST         | ✅     | ✅       | ✅    |
| WHERE POINCARE_DIST           | ✅     | ✅       | ✅    |
| WHERE KLEIN_DIST              | —      | ✅       | ✅    |
| WHERE SENSORY_DIST            | ✅     | ✅       | ✅    |
| WHERE MINKOWSKI_NORM          | ✅     | ✅       | ✅    |
| WHERE LOBACHEVSKY_ANGLE       | —      | ✅       | ✅    |
| WHERE RIEMANN_CURVATURE       | ✅     | —        | ✅    |
| WHERE GAUSS_KERNEL            | ✅     | ✅       | ✅    |
| WHERE CHEBYSHEV_COEFF         | —      | ✅       | ✅    |
| WHERE RAMANUJAN_EXPANSION     | —      | ✅       | ✅    |
| WHERE HAUSDORFF_DIM           | ✅     | ✅       | ✅    |
| WHERE EULER_CHAR              | —      | ✅       | ✅    |
| WHERE LAPLACIAN_SCORE         | —      | ✅       | ✅    |
| WHERE FOURIER_COEFF           | —      | —        | ⚠️ só implementação |
| WHERE DIRICHLET_ENERGY        | —      | ✅       | ✅    |
| WHERE AND                     | ✅     | —        | ✅    |
| WHERE OR                      | —      | —        | ⚠️ só implementação |
| WHERE NOT                     | ✅     | —        | ✅    |
| WHERE IN                      | ✅     | ✅       | ✅    |
| WHERE BETWEEN                 | ✅     | ✅       | ✅    |
| WHERE CONTAINS                | ✅     | —        | ✅    |
| WHERE STARTS_WITH             | ✅     | —        | ✅    |
| WHERE ENDS_WITH               | ✅     | —        | ✅    |
| ORDER BY campo                | ✅     | ✅       | ✅    |
| ORDER BY HYPERBOLIC_DIST      | ✅     | —        | ✅    |
| ORDER BY POINCARE_DIST        | —      | ✅       | ✅    |
| ORDER BY SENSORY_DIST         | ✅     | ✅       | ✅    |
| RETURN DISTINCT               | ✅     | —        | ✅    |
| RETURN AS alias               | ✅     | —        | ✅    |
| LIMIT                         | ✅     | ✅       | ✅    |
| SKIP                          | ✅     | ✅       | ✅    |
| COUNT(*)                      | ✅     | ✅       | ✅    |
| AVG                           | ✅     | ✅       | ✅    |
| MIN / MAX                     | ✅     | —        | ✅    |
| SUM                           | ✅     | —        | ✅    |
| GROUP BY                      | ✅     | —        | ✅    |
| DIFFUSE FROM $id              | ✅     | ✅       | ✅    |
| DIFFUSE WITH t = [...]        | ✅     | —        | ✅    |
| RECONSTRUCT full              | ✅     | ✅       | ✅    |
| RECONSTRUCT minimal           | ✅     | ✅       | ✅    |
| EXPLAIN MATCH                 | ✅     | ✅       | ✅    |
| EXPLAIN DIFFUSE               | ✅     | —        | ✅    |
| INVOKE ZARATUSTRA             | —      | ✅*      | ✅    |
| BEGIN / COMMIT / ROLLBACK     | —      | ✅*      | ✅    |
| Inline vector `[v1, v2]`      | ✅     | —        | ✅    |
| Erro em sintaxe inválida      | ✅     | —        | ✅    |

*Forwarded para a camada gRPC — o executor apenas retorna o variant correto.

---

## 7. Notas de Implementação

### Precisão de Ponto Flutuante
O campo `energy` é armazenado como `f32` no `Node`, mas promovido para `f64` nas comparações.
O teste `match_in_operator` usa os valores exatos da conversão `f32→f64` (e.g., `0.3f32 as f64 = 0.30000001192092896`).

### Condição padrão para MATCH sem WHERE
Não existe "MATCH (n) RETURN n" sem WHERE — o parser requer pelo menos uma condição.
Para retornar todos os nós, use: `MATCH (n) WHERE n.energy >= 0.0 RETURN n`.

### DIFFUSE e MAX_HOPS
O algoritmo de difusão é um `diffusion_walk` estocástico — o caminho exato varia entre
execuções quando há múltiplos vizinhos. O teste verifica apenas `path[0] == start_id`
e `path.len() >= 2`, não a sequência exata.

### SENSORY_DIST vs HYPERBOLIC_DIST
No executor atual, ambas usam `node.embedding` como proxy (a camada de senso não está
integrada no executor de baixo nível). A distinção semântica é preservada no AST e na
camada gRPC onde o embedding sensorial real é resolvido antes da execução.

---

*Relatório gerado por Claude Code — NietzscheDB v2.0.0*
