# Nietzsche Database — Roadmap Completo

## Construindo um banco hiperbólico + grafo + fractal em Rust sobre HyperspaceDB

> **Premissa**: HyperspaceDB como camada vetorial hiperbólica nativa (Poincaré + HNSW). Tudo o mais é construído em Rust por cima, resultando num banco de dados completo original.

---

## Visão geral das fases

```
FASE 0   Fundação e ambiente                          2-3 semanas
FASE 1   Modelo de nós e arestas                      4-6 semanas
FASE 2   Storage engine de grafo                      6-10 semanas
FASE 3   Traversal engine                             4-6 semanas
FASE 4   Linguagem de query (NQL)                     8-12 semanas
FASE 5   L-System engine                              4-6 semanas
FASE 6   Difusão fractal / Pregel hiperbólico         6-8 semanas
FASE 7   Transações ACID em grafo                     6-10 semanas
FASE 8   Reconsolidação (ciclo de sono)               4-6 semanas
FASE 9   API pública + SDKs                           4-6 semanas
FASE 10  Benchmark, hardening, produção               4-8 semanas

TOTAL ESTIMADO:  ~18-24 meses (solo, tempo integral)
                 ~10-14 meses (equipe de 3-4 engenheiros Rust sênior)
```

---

## FASE 0 — Fundação e ambiente

**Duração:** 2-3 semanas **Objetivo:** Entender o código do HyperspaceDB e preparar o workspace

### 0.1 — Ler e mapear o HyperspaceDB

```
Repositório:  github.com/YARlabs/hyperspace-db
Arquivos-chave a estudar:
  crates/                    ← core do banco
  crates/hyperspace-core/    ← HNSW hiperbólico
  crates/hyperspace-server/  ← gRPC server
  ARCHITECTURE.md            ← visão geral
```

**O que mapear:**

* Como o HNSW hiperbólico está implementado (distância de Poincaré)
* Como o WAL (Write-Ahead Log) funciona
* Como o mmap storage é organizado
* Onde injetar o grafo sem quebrar a vector store

### 0.2 — Setup do workspace Rust

```toml
# Cargo.toml (workspace)
[workspace]
members = [
    "crates/hyperspace-db",     # fork do HyperspaceDB
    "crates/nietzsche-graph",   # grafo que você vai construir
    "crates/nietzsche-query",   # linguagem de query (NQL)
    "crates/nietzsche-lsystem", # L-System engine
    "crates/nietzsche-pregel",  # difusão fractal
    "crates/nietzsche-sleep",   # ciclo de reconsolidação
    "crates/nietzsche-api",     # API pública unificada
    "crates/nietzsche-sdk",     # SDK Python/TS gerado
]
```

### 0.3 — Fork e branching strategy

```bash
git clone github.com/YARlabs/hyperspace-db
# Criar fork próprio — AGPL-3.0 permite
# Manter upstream sync para pegar updates de HNSW hiperbólico
git remote add upstream github.com/YARlabs/hyperspace-db
```

### Critério de saída da Fase 0

```
✓ Consegue compilar e rodar HyperspaceDB localmente
✓ Consegue inserir vetores no Poincaré ball via Python SDK
✓ Entendeu onde o HNSW hiperbólico vive no código
✓ Workspace Rust configurado com todos os crates
```

---

## FASE 1 — Modelo de nós e arestas

**Duração:** 4-6 semanas **Objetivo:** Definir as estruturas de dados que representam o grafo

### 1.1 — Estruturas core

```rust
// crates/nietzsche-graph/src/model.rs

use uuid::Uuid;
use std::collections::HashMap;

/// Nó no grafo hiperbólico
/// Vive tanto no HyperspaceDB (embedding) quanto no grafo (relações)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Node {
    pub id: Uuid,
    pub embedding: PoincaréVector,    // coordenadas no ball ‖x‖ < 1
    pub depth: f32,                   // distância ao centro (hierarquia)
    pub content: serde_json::Value,   // documento JSON livre
    pub node_type: NodeType,
    pub energy: f32,                  // 1.0 → decai → poda
    pub lsystem_generation: u32,      // qual geração do L-System criou este nó
    pub hausdorff_local: f32,         // dimensão fractal local (poda)
    pub created_at: i64,              // unix timestamp
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PoincaréVector {
    pub coords: Vec<f64>,
    pub dim: usize,
    // invariante: coords.iter().map(|x| x*x).sum::<f64>().sqrt() < 1.0
}

impl PoincaréVector {
    pub fn norm(&self) -> f64 {
        self.coords.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    pub fn is_valid(&self) -> bool {
        self.norm() < 1.0
    }

    /// Distância hiperbólica de Poincaré entre dois pontos
    pub fn distance(&self, other: &Self) -> f64 {
        let diff_sq: f64 = self.coords.iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let denom = (1.0 - self.norm().powi(2)) * (1.0 - other.norm().powi(2));
        (1.0 + 2.0 * diff_sq / denom).acosh()
    }
}

/// Aresta no grafo
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Edge {
    pub id: Uuid,
    pub from: Uuid,
    pub to: Uuid,
    pub edge_type: EdgeType,
    pub weight: f32,
    pub lsystem_rule: Option<String>,  // qual regra de produção criou esta aresta
    pub created_at: i64,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum NodeType {
    Episodic,      // memória episódica (específica, perto do boundary)
    Semantic,      // memória semântica (abstrata, perto do centro)
    Concept,       // conceito puro
    DreamSnapshot, // estado capturado durante reconsolidação
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum EdgeType {
    Association,      // aresta normal
    LSystemGenerated, // criada por regra de produção
    Hierarchical,     // pai → filho na hierarquia
    Pruned,           // foi podada (arquivada, não deletada)
}
```

### 1.2 — Adjacency structure

```rust
// crates/nietzsche-graph/src/adjacency.rs

use dashmap::DashMap; // concurrent hashmap
use uuid::Uuid;

/// Estrutura de adjacência em memória
/// Separada do storage — é o índice de navegação do grafo
pub struct AdjacencyIndex {
    // node_id → lista de (edge_id, neighbor_id, weight)
    outgoing: DashMap<Uuid, Vec<(Uuid, Uuid, f32)>>,
    incoming: DashMap<Uuid, Vec<(Uuid, Uuid, f32)>>,
    // para traversal bidirecional eficiente
}

impl AdjacencyIndex {
    pub fn add_edge(&self, edge: &Edge) {
        self.outgoing
            .entry(edge.from)
            .or_default()
            .push((edge.id, edge.to, edge.weight));

        self.incoming
            .entry(edge.to)
            .or_default()
            .push((edge.id, edge.from, edge.weight));
    }

    pub fn neighbors_out(&self, node_id: &Uuid) -> Vec<Uuid> {
        self.outgoing
            .get(node_id)
            .map(|v| v.iter().map(|(_, n, _)| *n).collect())
            .unwrap_or_default()
    }
}
```

### Critério de saída da Fase 1

```
✓ Node e Edge compilam com serde (serialização)
✓ PoincaréVector.distance() produz distâncias corretas (teste unitário)
✓ AdjacencyIndex suporta inserção e busca concorrente
✓ Invariante ‖x‖ < 1.0 validada em todos os pontos de entrada
```

---

## FASE 2 — Storage engine de grafo

**Duração:** 6-10 semanas **Objetivo:** Persistir nós e arestas em disco, separado do HyperspaceDB

### 2.1 — Escolha do storage backend

```
Opção A: RocksDB (LSM-tree)
  ✓ Alta performance de escrita
  ✓ Compressão nativa
  ✓ Usado pelo ArangoDB internamente
  → crate: rocksdb

Opção B: Sled (pure Rust B-tree)
  ✓ 100% Rust, sem dependência C++
  ✓ Mais simples de integrar
  ✗ Menos maduro que RocksDB
  → crate: sled

Recomendação: RocksDB (mesmo trade-off que o ArangoDB fez)
```

### 2.2 — Layout do storage

```rust
// crates/nietzsche-graph/src/storage.rs

use rocksdb::{DB, Options, ColumnFamily};

/// Column Families (tabelas dentro do RocksDB)
const CF_NODES: &str = "nodes";       // key: node_id → value: Node (bincode)
const CF_EDGES: &str = "edges";       // key: edge_id → value: Edge (bincode)
const CF_ADJ_OUT: &str = "adj_out";   // key: node_id → value: Vec<edge_id>
const CF_ADJ_IN: &str = "adj_in";     // key: node_id → value: Vec<edge_id>
const CF_META: &str = "meta";         // key: string → value: metadata global

pub struct GraphStorage {
    db: DB,
}

impl GraphStorage {
    pub fn open(path: &str) -> Result<Self, StorageError> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cfs = vec![CF_NODES, CF_EDGES, CF_ADJ_OUT, CF_ADJ_IN, CF_META];
        let db = DB::open_cf(&opts, path, &cfs)?;
        Ok(Self { db })
    }

    pub fn put_node(&self, node: &Node) -> Result<(), StorageError> {
        let cf = self.db.cf_handle(CF_NODES).unwrap();
        let key = node.id.as_bytes();
        let value = bincode::serialize(node)?;
        self.db.put_cf(cf, key, value)?;
        Ok(())
    }

    pub fn get_node(&self, id: &Uuid) -> Result<Option<Node>, StorageError> {
        let cf = self.db.cf_handle(CF_NODES).unwrap();
        match self.db.get_cf(cf, id.as_bytes())? {
            Some(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn put_edge(&self, edge: &Edge) -> Result<(), StorageError> {
        // 1. Salva a aresta
        let cf_edges = self.db.cf_handle(CF_EDGES).unwrap();
        self.db.put_cf(cf_edges, edge.id.as_bytes(), bincode::serialize(edge)?)?;

        // 2. Atualiza adjacência outgoing
        self.append_adjacency(CF_ADJ_OUT, &edge.from, &edge.id)?;

        // 3. Atualiza adjacência incoming
        self.append_adjacency(CF_ADJ_IN, &edge.to, &edge.id)?;

        Ok(())
    }
}
```

### 2.3 — WAL próprio para o grafo

```rust
// Write-Ahead Log para operações de grafo
// Separado do WAL do HyperspaceDB (que só cobre vetores)

pub enum GraphWalEntry {
    InsertNode(Node),
    InsertEdge(Edge),
    DeleteNode(Uuid),
    PruneEdge(Uuid),              // arquiva, não deleta
    UpdateNodeEnergy(Uuid, f32),
    UpdateHausdorff(Uuid, f32),
}
```

### 2.4 — Sincronização com HyperspaceDB

```rust
// O ponto crítico: quando você insere um nó no Nietzsche DB,
// você precisa inserir em DOIS lugares:
//   1. HyperspaceDB → embedding hiperbólico (busca vetorial)
//   2. GraphStorage → nó + arestas (traversal, L-System, etc)

pub struct NietzscheDB {
    hyperspace: HyperspaceClient,  // cliente gRPC pro HyperspaceDB
    graph: GraphStorage,           // RocksDB local
    adjacency: AdjacencyIndex,     // índice em memória
}

impl NietzscheDB {
    pub async fn insert_node(&self, node: Node) -> Result<Uuid, Error> {
        // Validação hiperbólica
        assert!(node.embedding.is_valid(), "‖x‖ deve ser < 1.0");

        // 1. Persiste no grafo (RocksDB)
        self.graph.put_node(&node)?;

        // 2. Atualiza adjacência em memória
        // (nova inserção não tem arestas ainda)

        // 3. Insere embedding no HyperspaceDB
        self.hyperspace.insert(
            node.id.to_string(),
            node.embedding.coords.clone(),
        ).await?;

        // 4. WAL entry
        self.graph.wal_append(GraphWalEntry::InsertNode(node.clone()))?;

        Ok(node.id)
    }
}
```

### Critério de saída da Fase 2

```
✓ Inserção de nó persiste em RocksDB E HyperspaceDB atomicamente
✓ Recuperação de nó por ID funciona após restart
✓ Adjacência reconstruída do RocksDB na inicialização
✓ WAL permite recovery após crash
✓ Teste de 100k nós inseridos sem perda de dados
```

---

## FASE 3 — Traversal engine

**Duração:** 4-6 semanas **Objetivo:** Navegar o grafo em profundidade/largura, com custo hiperbólico

### 3.1 — Traversal básico

```rust
// crates/nietzsche-graph/src/traversal.rs

pub struct TraversalConfig {
    pub max_depth: usize,
    pub max_nodes: usize,
    pub direction: Direction,          // Outgoing, Incoming, Both
    pub edge_filter: Option<EdgeType>, // filtrar por tipo de aresta
    pub cost_fn: CostFunction,         // como calcular custo de cada passo
}

pub enum CostFunction {
    Uniform,                    // todos os passos custam 1
    HyperbolicDistance,         // custo = d(u,v) hiperbólica
    EnergyWeighted,             // custo = 1 / edge.weight
}

pub enum Direction {
    Outgoing,
    Incoming,
    Both,
}

pub struct TraversalResult {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub total_cost: f64,
}

impl NietzscheDB {
    /// BFS com custo hiperbólico
    pub fn traverse_bfs(
        &self,
        start: Uuid,
        config: TraversalConfig,
    ) -> TraversalResult {
        let mut queue = std::collections::VecDeque::new();
        let mut visited = std::collections::HashSet::new();
        let mut result_nodes = vec![];
        let mut result_edges = vec![];
        let mut total_cost = 0.0;

        queue.push_back((start, 0usize, 0.0f64));

        while let Some((node_id, depth, cost)) = queue.pop_front() {
            if depth > config.max_depth { continue; }
            if result_nodes.len() >= config.max_nodes { break; }
            if visited.contains(&node_id) { continue; }

            visited.insert(node_id);

            let node = self.graph.get_node(&node_id).unwrap().unwrap();
            result_nodes.push(node.clone());
            total_cost += cost;

            let neighbors = match config.direction {
                Direction::Outgoing => self.adjacency.neighbors_out(&node_id),
                Direction::Incoming => self.adjacency.neighbors_in(&node_id),
                Direction::Both => {
                    let mut n = self.adjacency.neighbors_out(&node_id);
                    n.extend(self.adjacency.neighbors_in(&node_id));
                    n
                }
            };

            for neighbor_id in neighbors {
                if visited.contains(&neighbor_id) { continue; }

                let neighbor = self.graph.get_node(&neighbor_id).unwrap().unwrap();
                let step_cost = match config.cost_fn {
                    CostFunction::Uniform => 1.0,
                    CostFunction::HyperbolicDistance =>
                        node.embedding.distance(&neighbor.embedding),
                    CostFunction::EnergyWeighted =>
                        1.0 / neighbor.energy as f64,
                };

                queue.push_back((neighbor_id, depth + 1, step_cost));
            }
        }

        TraversalResult { nodes: result_nodes, edges: result_edges, total_cost }
    }

    /// Dijkstra hiperbólico — menor caminho em distância hiperbólica
    pub fn shortest_path_hyperbolic(
        &self,
        from: Uuid,
        to: Uuid,
    ) -> Option<Vec<Uuid>> {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;

        let mut dist: HashMap<Uuid, f64> = HashMap::new();
        let mut prev: HashMap<Uuid, Uuid> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(from, 0.0);
        heap.push(Reverse((ordered_float::OrderedFloat(0.0), from)));

        while let Some(Reverse((cost, node_id))) = heap.pop() {
            if node_id == to {
                // Reconstruir caminho
                let mut path = vec![to];
                let mut current = to;
                while let Some(&p) = prev.get(&current) {
                    path.push(p);
                    current = p;
                }
                path.reverse();
                return Some(path);
            }

            let node = self.graph.get_node(&node_id).unwrap()?;

            for neighbor_id in self.adjacency.neighbors_out(&node_id) {
                let neighbor = self.graph.get_node(&neighbor_id).unwrap()?;
                let d = cost.0 + node.embedding.distance(&neighbor.embedding);

                if d < *dist.get(&neighbor_id).unwrap_or(&f64::MAX) {
                    dist.insert(neighbor_id, d);
                    prev.insert(neighbor_id, node_id);
                    heap.push(Reverse((ordered_float::OrderedFloat(d), neighbor_id)));
                }
            }
        }
        None
    }
}
```

### Critério de saída da Fase 3

```
✓ BFS retorna nós corretos com profundidade limitada
✓ Dijkstra hiperbólico encontra o caminho mais curto
✓ Traversal funciona com 1M+ nós sem OOM
✓ Custo hiperbólico é matematicamente correto (testes unitários)
```

---

## FASE 4 — Linguagem de query (NQL — Nietzsche Query Language)

**Duração:** 8-12 semanas **Objetivo:** Query language declarativa que combina grafo + hiperbólico

### 4.1 — Design da linguagem

```
NQL é inspirada em AQL/Cypher mas com primitivas hiperbólicas nativas:

// Busca semântica hiperbólica
MATCH (m:Memory)
WHERE HYPERBOLIC_DIST(m.embedding, $query_vec) < 0.5
AND m.depth > 0.6
RETURN m ORDER BY HYPERBOLIC_DIST(m.embedding, $query_vec) ASC
LIMIT 10

// Traversal com custo hiperbólico
FOR v, e, p IN 1..3 OUTBOUND $start_node associations
  FILTER e.weight > 0.3
  SORT HYPERBOLIC_DIST(v.embedding, $query_vec) ASC
  RETURN {node: v, path_cost: p.cost}

// Difusão multi-escala
DIFFUSE FROM $node
  WITH t = [0.1, 1.0, 10.0]   -- escalas de difusão
  MAX_HOPS 5
RETURN activated_nodes, activation_scores

// L-System query
LSYSTEM GROW FROM $seed_node
  RULE "node(t, i) -> node(t, i*0.9) + child(i*0.7) IF i > 0.1"
  GENERATIONS 3
RETURN new_nodes, new_edges
```

### 4.2 — Parser em Rust

```toml
# Cargo.toml
[dependencies]
pest = "2"          # PEG parser generator
pest_derive = "2"
```

```pest
// grammar/nql.pest
query = { match_clause | for_clause | diffuse_clause | lsystem_clause }

match_clause = {
    "MATCH" ~ pattern ~
    ("WHERE" ~ condition)? ~
    ("RETURN" ~ return_expr) ~
    ("ORDER BY" ~ order_expr)? ~
    ("LIMIT" ~ number)?
}

hyperbolic_dist = {
    "HYPERBOLIC_DIST" ~ "(" ~ expr ~ "," ~ expr ~ ")"
}

diffuse_clause = {
    "DIFFUSE FROM" ~ expr ~
    "WITH t = [" ~ number_list ~ "]" ~
    "MAX_HOPS" ~ number ~
    "RETURN" ~ return_expr
}
```

### 4.3 — Planner e executor

```rust
// crates/nietzsche-query/src/planner.rs

pub enum QueryPlan {
    VectorScan {
        embedding: PoincaréVector,
        radius: f64,
        top_k: usize,
    },
    GraphTraversal {
        start: Uuid,
        config: TraversalConfig,
    },
    HyperbolicDiffusion {
        start: Uuid,
        t_values: Vec<f64>,
        max_hops: usize,
    },
    LSystemGrow {
        seed: Uuid,
        rules: Vec<ProductionRule>,
        generations: u32,
    },
    Hybrid {                         // combina vector + graph
        vector_phase: Box<QueryPlan>,
        graph_phase: Box<QueryPlan>,
    },
}

pub struct QueryPlanner;

impl QueryPlanner {
    pub fn plan(&self, ast: NqlAst) -> QueryPlan {
        // Análise do AST → escolha do plano mais eficiente
        // Similar ao query optimizer do PostgreSQL mas muito mais simples
        match ast {
            NqlAst::Match { pattern, condition, .. } => {
                if condition.uses_hyperbolic_dist() {
                    // Pode usar HyperspaceDB para candidate retrieval
                    QueryPlan::VectorScan { .. }
                } else {
                    // Traversal puro
                    QueryPlan::GraphTraversal { .. }
                }
            }
            NqlAst::Diffuse { .. } => QueryPlan::HyperbolicDiffusion { .. },
            NqlAst::LSystem { .. } => QueryPlan::LSystemGrow { .. },
        }
    }
}
```

### Critério de saída da Fase 4

```
✓ Parser aceita queries NQL básicas sem panic
✓ MATCH + WHERE HYPERBOLIC_DIST retorna resultados corretos
✓ FOR v IN traversal funciona com filtros
✓ DIFFUSE retorna ativações em múltiplas escalas
✓ Planner escolhe vector scan quando há filtro hiperbólico
```

---

## FASE 5 — L-System engine

**Duração:** 4-6 semanas **Objetivo:** Grafo que cresce e se poda por regras de produção

### 5.1 — Estrutura de regras

```rust
// crates/nietzsche-lsystem/src/lib.rs

#[derive(Debug, Clone)]
pub struct ProductionRule {
    pub name: String,
    pub condition: RuleCondition,    // quando esta regra dispara
    pub action: RuleAction,          // o que ela faz
}

pub enum RuleCondition {
    EnergyAbove(f32),                // node.energy > threshold
    DepthBelow(f32),                 // node.depth < threshold (perto do centro)
    HausdorffAbove(f32),             // dimensão fractal local > threshold
    GenerationBelow(u32),            // geração < max
    Custom(Box<dyn Fn(&Node) -> bool + Send + Sync>),
}

pub enum RuleAction {
    SpawnChild {
        energy_factor: f32,          // filho herda energy * factor
        depth_delta: f32,            // filho fica mais perto do boundary
        edge_type: EdgeType,
    },
    SpawnSibling {
        angle: f32,                  // ângulo no espaço hiperbólico
        energy_factor: f32,
    },
    Prune,                           // remove este nó
    UpdateEnergy(f32),               // multiplica energy por fator
}

pub struct LSystemEngine {
    rules: Vec<ProductionRule>,
    max_nodes_per_tick: usize,       // controle de explosão
    hausdorff_budget: f32,           // dimensão total máxima do grafo
}

impl LSystemEngine {
    /// Executa um tick do L-System sobre o grafo inteiro
    pub async fn tick(&self, db: &mut NietzscheDB) -> LSystemResult {
        let mut nodes_created = 0;
        let mut nodes_pruned = 0;

        // Itera sobre todos os nós — disparo assíncrono
        let all_nodes = db.graph.scan_nodes();

        for node in all_nodes {
            for rule in &self.rules {
                if rule.condition.evaluate(&node) {
                    match &rule.action {
                        RuleAction::SpawnChild { energy_factor, depth_delta, edge_type } => {
                            if nodes_created >= self.max_nodes_per_tick { break; }

                            // Posição do filho no Poincaré ball:
                            // mais perto do boundary que o pai (depth + delta)
                            let child_embedding = self.compute_child_position(
                                &node.embedding,
                                *depth_delta,
                            );

                            let child = Node {
                                id: Uuid::new_v4(),
                                embedding: child_embedding,
                                energy: node.energy * energy_factor,
                                lsystem_generation: node.lsystem_generation + 1,
                                ..node.clone()
                            };

                            db.insert_node(child.clone()).await?;
                            db.insert_edge(Edge {
                                from: node.id,
                                to: child.id,
                                edge_type: *edge_type,
                                lsystem_rule: Some(rule.name.clone()),
                                ..Default::default()
                            }).await?;

                            nodes_created += 1;
                        }
                        RuleAction::Prune => {
                            db.prune_node(node.id).await?;
                            nodes_pruned += 1;
                        }
                        _ => {}
                    }
                }
            }
        }

        LSystemResult { nodes_created, nodes_pruned }
    }

    /// Calcula posição do filho no espaço hiperbólico
    /// O filho fica mais afastado do centro (mais específico)
    fn compute_child_position(
        &self,
        parent: &PoincaréVector,
        depth_delta: f32,
    ) -> PoincaréVector {
        // Usa Möbius addition para mover no manifold
        // em direção ao boundary (afasta do centro)
        let scale = 1.0 + depth_delta as f64;
        let new_coords: Vec<f64> = parent.coords.iter()
            .map(|x| x * scale)
            .collect();

        // Garante que ainda está dentro do ball
        let norm: f64 = new_coords.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1.0 {
            let coords = new_coords.iter().map(|x| x / (norm + 1e-5) * 0.95).collect();
            PoincaréVector { coords, dim: parent.dim }
        } else {
            PoincaréVector { coords: new_coords, dim: parent.dim }
        }
    }
}
```

### 5.2 — Poda via dimensão de Hausdorff

```rust
// Calcula dimensão de Hausdorff local via box-counting
pub fn hausdorff_local(db: &NietzscheDB, node_id: Uuid, radius_hops: usize) -> f32 {
    // 1. Pega vizinhança do nó até radius_hops hops
    let neighborhood = db.traverse_bfs(node_id, TraversalConfig {
        max_depth: radius_hops,
        cost_fn: CostFunction::Uniform,
        ..Default::default()
    });

    let coords: Vec<Vec<f64>> = neighborhood.nodes.iter()
        .map(|n| n.embedding.coords.clone())
        .collect();

    // 2. Box-counting nas coordenadas hiperbólicas
    let epsilon_range = vec![0.5, 0.25, 0.125, 0.0625];
    let counts: Vec<f64> = epsilon_range.iter().map(|&eps| {
        let mut boxes = std::collections::HashSet::new();
        for coord in &coords {
            let key: Vec<i64> = coord.iter()
                .map(|x| (x / eps).floor() as i64)
                .collect();
            boxes.insert(key);
        }
        boxes.len() as f64
    }).collect();

    // 3. Regressão log-log → dimensão
    let log_eps: Vec<f64> = epsilon_range.iter().map(|e| e.log2()).collect();
    let log_cnt: Vec<f64> = counts.iter().map(|c| c.log2()).collect();

    linear_regression_slope(&log_eps, &log_cnt) as f32
}

// Critério de poda:
// Se hausdorff_local < 0.5 → região colapsou → prune
// Se hausdorff_local > 1.9 → região densa demais → prune também
pub fn should_prune(hausdorff: f32) -> bool {
    hausdorff < 0.5 || hausdorff > 1.9
}
```

### Critério de saída da Fase 5

```
✓ L-System gera novos nós por regras de produção
✓ Poda baseada em Hausdorff remove nós corretos
✓ Dimensão de Hausdorff global fica estável em 1.2-1.8
✓ Grafo não explode (max_nodes_per_tick é respeitado)
✓ Todos os nós gerados têm ‖x‖ < 1.0 (invariante hiperbólica)
```

---

## FASE 6 — Difusão fractal / Pregel hiperbólico

**Duração:** 6-8 semanas **Objetivo:** Heat kernel em múltiplas escalas sobre o grafo hiperbólico

### 6.1 — Heat kernel discreto

```rust
// crates/nietzsche-pregel/src/heat_kernel.rs

pub struct HeatKernelConfig {
    pub t_values: Vec<f64>,          // escalas de difusão
    pub max_iterations: usize,        // Chebyshev approximation degree
    pub convergence_threshold: f64,
}

pub struct HeatKernelResult {
    /// Para cada t, a ativação de cada nó
    pub activations_by_t: HashMap<ordered_float::OrderedFloat<f64>, HashMap<Uuid, f64>>,
}

impl NietzscheDB {
    /// Difusão hiperbólica multi-escala via Chebyshev approximation
    /// Complexidade: O(K * |E|) onde K = grau do polinômio
    pub fn hyperbolic_diffusion(
        &self,
        source: Uuid,
        config: HeatKernelConfig,
    ) -> HeatKernelResult {
        // 1. Constrói sinal inicial (delta no nó source)
        let mut signal: HashMap<Uuid, f64> = HashMap::new();
        signal.insert(source, 1.0);

        let mut result = HeatKernelResult { activations_by_t: HashMap::new() };

        // 2. Para cada escala t, aplica o heat kernel
        for &t in &config.t_values {
            // K_t(signal) ≈ Σ_k c_k(t) T_k(L̃) signal
            // onde T_k são os polinômios de Chebyshev e L̃ é o Laplaciano normalizado
            let activated = self.chebyshev_propagation(
                &signal,
                t,
                config.max_iterations,
            );

            result.activations_by_t.insert(
                ordered_float::OrderedFloat(t),
                activated,
            );
        }

        result
    }

    fn chebyshev_propagation(
        &self,
        signal: &HashMap<Uuid, f64>,
        t: f64,
        k: usize,
    ) -> HashMap<Uuid, f64> {
        // Aproximação do heat kernel e^(-tL) via polinômios de Chebyshev
        // Referência: "Convolutional Neural Networks on Graphs with
        //              Fast Localized Spectral Filtering" (Defferrard 2016)

        let mut x_prev: HashMap<Uuid, f64> = signal.clone();
        let mut x_curr = self.laplacian_multiply(&x_prev); // L * signal
        let mut result: HashMap<Uuid, f64> = HashMap::new();

        // c_0 = e^(-t) * I_0(t), c_1 = 2*e^(-t)*I_1(t), ...
        // (coeficientes de Chebyshev do heat kernel)
        let coeffs = chebyshev_heat_coefficients(t, k);

        for node_id in signal.keys() {
            let v = coeffs[0] * x_prev.get(node_id).unwrap_or(&0.0)
                  + coeffs[1] * x_curr.get(node_id).unwrap_or(&0.0);
            result.insert(*node_id, v);
        }

        // Iterações restantes
        for i in 2..k {
            let x_next = self.laplacian_multiply_and_combine(&x_curr, &x_prev, 2.0);
            for node_id in signal.keys() {
                *result.entry(*node_id).or_insert(0.0) +=
                    coeffs[i] * x_next.get(node_id).unwrap_or(&0.0);
            }
            x_prev = x_curr;
            x_curr = x_next;
        }

        result
    }

    /// L * x — multiplicação pelo Laplaciano hiperbólico
    /// L_hyp(i,j) = d(i,j)_hyp se (i,j) é aresta, -grau(i) se i=j
    fn laplacian_multiply(&self, x: &HashMap<Uuid, f64>) -> HashMap<Uuid, f64> {
        let mut result: HashMap<Uuid, f64> = HashMap::new();

        for (node_id, &val) in x {
            let node = self.graph.get_node(node_id).unwrap().unwrap();
            let neighbors = self.adjacency.neighbors_out(node_id);

            let degree = neighbors.len() as f64;
            *result.entry(*node_id).or_insert(0.0) -= degree * val;

            for neighbor_id in neighbors {
                let neighbor = self.graph.get_node(&neighbor_id).unwrap().unwrap();
                // peso hiperbólico da aresta
                let w = node.embedding.distance(&neighbor.embedding);
                *result.entry(neighbor_id).or_insert(0.0) += w * val;
            }
        }

        result
    }
}
```

### Critério de saída da Fase 6

```
✓ t=0.1 ativa apenas vizinhos diretos (busca local)
✓ t=10.0 ativa nós a 3+ hops (associação livre)
✓ Overlap entre t=0.1 e t=10.0 é < 30%
✓ Latência < 500ms em grafos de 100k nós (Chebyshev K=10)
✓ Ativações somam ≤ 1.0 (conservação de energia)
```

---

## FASE 7 — Transações ACID em grafo

**Duração:** 6-10 semanas **Objetivo:** Garantir consistência entre GraphStorage e HyperspaceDB

### 7.1 — O problema de duas escritas

```
Cada inserção no Nietzsche DB escreve em DOIS sistemas:
  1. HyperspaceDB (vetores hiperbólicos)
  2. RocksDB (nós + arestas)

Se 1 succeeds e 2 falha → inconsistência
Se 2 succeeds e 1 falha → nó existe no grafo sem embedding

Solução: Saga pattern (2-phase commit simplificado)
```

### 7.2 — Saga pattern

```rust
// crates/nietzsche-graph/src/transaction.rs

pub struct Transaction {
    id: Uuid,
    operations: Vec<TxOperation>,
    compensations: Vec<TxOperation>,  // rollback ops
    state: TxState,
}

pub enum TxOperation {
    InsertNodeGraph(Node),
    InsertNodeVector(Uuid, Vec<f64>),
    InsertEdge(Edge),
    PruneNode(Uuid),
    UpdateEmbedding(Uuid, PoincaréVector),
}

pub enum TxState {
    Pending,
    Committed,
    RolledBack,
}

impl NietzscheDB {
    pub async fn begin_tx(&self) -> Transaction {
        Transaction {
            id: Uuid::new_v4(),
            operations: vec![],
            compensations: vec![],
            state: TxState::Pending,
        }
    }

    pub async fn commit(&self, tx: Transaction) -> Result<(), TxError> {
        // Phase 1: executa todas as operações
        for op in &tx.operations {
            match op {
                TxOperation::InsertNodeGraph(node) => {
                    self.graph.put_node(node)?;
                }
                TxOperation::InsertNodeVector(id, coords) => {
                    // Se falhar aqui, executa compensations
                    self.hyperspace.insert(id.to_string(), coords.clone())
                        .await
                        .map_err(|e| {
                            // Rollback das ops de grafo já feitas
                            self.rollback(&tx);
                            TxError::VectorInsertFailed(e)
                        })?;
                }
                _ => {}
            }
        }

        // Phase 2: marca como committed no WAL
        self.graph.wal_append(GraphWalEntry::TxCommitted(tx.id))?;
        Ok(())
    }

    fn rollback(&self, tx: &Transaction) {
        for comp in tx.compensations.iter().rev() {
            // executa operações de compensação em ordem reversa
        }
    }
}
```

### Critério de saída da Fase 7

```
✓ Insert de nó é atômico: ou persiste em ambos ou em nenhum
✓ Crash durante commit → recovery via WAL
✓ Rollback limpa ambos os sistemas
✓ Teste de chaos: kill -9 durante inserção → sem dados órfãos
```

---

## FASE 8 — Reconsolidação (ciclo de sono)

**Duração:** 4-6 semanas **Objetivo:** Reescrita ativa do grafo com perturbação + verificação fractal

### 8.1 — Ciclo completo

```rust
// crates/nietzsche-sleep/src/lib.rs

pub struct SleepConfig {
    pub noise_scale: f64,            // perturbação dos embeddings
    pub sample_fraction: f64,        // fração do grafo por ciclo (0.1 = 10%)
    pub riemannian_lr: f64,          // learning rate do RiemannianAdam
    pub riemannian_iterations: u32,
    pub hausdorff_tolerance: f64,    // delta máximo permitido (0.05 = 5%)
}

pub struct SleepResult {
    pub nodes_updated: usize,
    pub cycles_committed: usize,
    pub cycles_rolled_back: usize,
    pub hausdorff_before: f32,
    pub hausdorff_after: f32,
}

impl NietzscheDB {
    pub async fn sleep_cycle(&self, config: SleepConfig) -> SleepResult {
        // 1. Mede dimensão de Hausdorff global (amostral)
        let hausdorff_before = self.sample_hausdorff(1000);

        // 2. Seleciona subgrafo de alta curvatura via random walk
        let seed = self.find_high_curvature_node();
        let subgraph = self.traverse_bfs(seed, TraversalConfig {
            max_depth: 4,
            max_nodes: (self.node_count() as f64 * config.sample_fraction) as usize,
            cost_fn: CostFunction::HyperbolicDistance,
            ..Default::default()
        });

        // 3. Snapshot dos embeddings atuais (para rollback)
        let snapshot: HashMap<Uuid, PoincaréVector> = subgraph.nodes.iter()
            .map(|n| (n.id, n.embedding.clone()))
            .collect();

        // 4. Perturbação no espaço tangente (sonho)
        let mut perturbed = snapshot.clone();
        for (_, emb) in perturbed.iter_mut() {
            for coord in emb.coords.iter_mut() {
                *coord += rand::random::<f64>() * config.noise_scale * 2.0
                         - config.noise_scale;
            }
            // Re-projeta para dentro do ball
            if emb.norm() >= 1.0 {
                let n = emb.norm();
                for c in emb.coords.iter_mut() { *c = *c / (n + 1e-5) * 0.95; }
            }
        }

        // 5. Otimização Riemanniana (minimiza energia do subgrafo)
        // Chamada ao Python via PyO3 ou reimplementação em Rust
        let optimized = self.riemannian_optimize(perturbed, &config);

        // 6. Aplica embeddings otimizados temporariamente
        for (node_id, new_emb) in &optimized {
            self.update_embedding_temp(*node_id, new_emb.clone()).await?;
        }

        // 7. Verifica preservação de Hausdorff
        let hausdorff_after = self.sample_hausdorff(1000);
        let delta = (hausdorff_after - hausdorff_before).abs() / hausdorff_before;

        if delta < config.hausdorff_tolerance as f32 {
            // Commit — reconsolidação preservou a identidade fractal
            for (node_id, new_emb) in &optimized {
                self.commit_embedding(*node_id, new_emb.clone()).await?;
                self.hyperspace.update(node_id.to_string(), new_emb.coords.clone()).await?;
            }
            SleepResult {
                nodes_updated: optimized.len(),
                hausdorff_before,
                hausdorff_after,
                cycles_committed: 1,
                cycles_rolled_back: 0,
            }
        } else {
            // Rollback — reconsolidação destruiu a estrutura fractal
            for (node_id, orig_emb) in &snapshot {
                self.commit_embedding(*node_id, orig_emb.clone()).await?;
            }
            SleepResult {
                nodes_updated: 0,
                hausdorff_before,
                hausdorff_after,
                cycles_committed: 0,
                cycles_rolled_back: 1,
            }
        }
    }

    /// Amostra dimensão de Hausdorff em n nós aleatórios
    fn sample_hausdorff(&self, n: usize) -> f32 {
        let sample_ids = self.random_sample_nodes(n);
        let dims: Vec<f32> = sample_ids.iter()
            .map(|&id| hausdorff_local(self, id, 2))
            .collect();
        dims.iter().sum::<f32>() / dims.len() as f32
    }
}
```

### Critério de saída da Fase 8

```
✓ delta_hausdorff < 5% em ciclos normais (commit)
✓ Rollback acontece em < 10% dos ciclos
✓ Embeddings realmente mudam (sem no-op)
✓ HyperspaceDB e RocksDB ficam em sync após cada ciclo
✓ 10 ciclos consecutivos não divergem
```

---

## FASE 9 — API pública + SDKs

**Duração:** 4-6 semanas **Objetivo:** Interface unificada que esconde toda a complexidade interna

### 9.1 — gRPC API unificada

```protobuf
// proto/nietzsche.proto

syntax = "proto3";

service NietzscheDB {
  // CRUD básico
  rpc InsertNode(InsertNodeRequest) returns (InsertNodeResponse);
  rpc GetNode(GetNodeRequest) returns (NodeResponse);
  rpc InsertEdge(InsertEdgeRequest) returns (InsertEdgeResponse);

  // Busca hiperbólica
  rpc SearchHyperbolic(SearchRequest) returns (SearchResponse);

  // Query NQL
  rpc Query(NqlRequest) returns (stream NqlResult);

  // L-System
  rpc LSystemTick(LSystemTickRequest) returns (LSystemTickResponse);

  // Difusão
  rpc Diffuse(DiffuseRequest) returns (DiffuseResponse);

  // Ciclo de sono
  rpc SleepCycle(SleepRequest) returns (SleepResponse);

  // Monitoramento
  rpc GetHealth(HealthRequest) returns (HealthResponse);
  rpc GetMetrics(MetricsRequest) returns (MetricsResponse);
}

message InsertNodeRequest {
  repeated double embedding = 1;   // coordenadas no Poincaré ball
  bytes content = 2;               // JSON do conteúdo
  string node_type = 3;
  map<string, bytes> metadata = 4;
}
```

### 9.2 — Python SDK

```python
# sdks/python/nietzsche_db/__init__.py

class NietzscheClient:
    def __init__(self, host: str = "localhost", port: int = 9090):
        self._channel = grpc.insecure_channel(f"{host}:{port}")
        self._stub = nietzsche_pb2_grpc.NietzscheDBStub(self._channel)

    def insert(self, embedding: list[float], content: dict, **kwargs) -> str:
        """Insere uma memória no espaço hiperbólico"""
        assert max(sum(x**2 for x in embedding)**0.5, 0) < 1.0, \
            "Embedding deve satisfazer ‖x‖ < 1.0"
        request = nietzsche_pb2.InsertNodeRequest(
            embedding=embedding,
            content=json.dumps(content).encode(),
        )
        response = self._stub.InsertNode(request)
        return response.node_id

    def search(self, query: list[float], top_k: int = 10,
               t: float = 1.0) -> list[dict]:
        """Busca hiperbólica com heat kernel"""
        request = nietzsche_pb2.SearchRequest(
            embedding=query, top_k=top_k, diffusion_t=t
        )
        return [self._parse_node(n) for n in self._stub.SearchHyperbolic(request).nodes]

    def query(self, nql: str) -> list[dict]:
        """Executa uma query NQL"""
        request = nietzsche_pb2.NqlRequest(query=nql)
        return [self._parse_result(r) for r in self._stub.Query(request)]

    def sleep(self, noise: float = 0.01) -> dict:
        """Executa um ciclo de reconsolidação"""
        response = self._stub.SleepCycle(
            nietzsche_pb2.SleepRequest(noise_scale=noise)
        )
        return {
            "nodes_updated": response.nodes_updated,
            "hausdorff_delta": response.hausdorff_delta,
            "committed": response.committed,
        }
```

### Critério de saída da Fase 9

```
✓ Python SDK funciona com pip install nietzsche-db
✓ gRPC API responde a todos os endpoints sem panic
✓ Documentação básica (README com exemplos)
✓ Docker image funcional (server + dashboard)
```

---

## FASE 10 — Benchmark, hardening, produção

**Duração:** 4-8 semanas **Objetivo:** Validar os critérios de sucesso definidos no B (antes da implementação)

### 10.1 — Critérios de sucesso (revisão)

```
Fase 1 (geometria):
  □ 90%+ das triplas hierárquicas ordenadas corretamente
  □ depth(abstrato) < depth(específico) em 85%+ dos casos

Fase 2 (busca):
  □ Recall hierárquico ≥ 75% vs 50% baseline cosine
  □ Latência P99 ≤ 500ms

Fase 3 (difusão):
  □ Overlap t=0.1 vs t=10.0 < 30%
  □ t pequeno = vizinhos, t grande = conceitos distantes

Fase 4 (L-System):
  □ Hausdorff global 1.2 < D < 1.8
  □ std(D ao longo do tempo) < 0.15

Fase 5 (sono):
  □ delta_hausdorff < 5% por ciclo
  □ rollback_rate < 10%
```

### 10.2 — Benchmark suite

```rust
// benches/full_stack.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_insert(c: &mut Criterion) {
    c.bench_function("insert 1k nodes", |b| {
        b.iter(|| {
            // insert 1000 nodes with random Poincaré embeddings
        })
    });
}

fn benchmark_search(c: &mut Criterion) {
    c.bench_function("hyperbolic search top-10", |b| {
        b.iter(|| {
            // search with diffusion t=1.0
        })
    });
}

fn benchmark_lsystem(c: &mut Criterion) {
    c.bench_function("lsystem tick 10k nodes", |b| {
        b.iter(|| {
            // one L-System tick on 10k node graph
        })
    });
}

criterion_group!(benches, benchmark_insert, benchmark_search, benchmark_lsystem);
criterion_main!(benches);
```

---

## Resumo de esforço total

```
FASE  COMPONENTE                    MESES (solo)    MESES (equipe 3)
  0   Fundação                      0.5             0.2
  1   Modelo nós/arestas            1.5             0.5
  2   Storage engine                2.5             0.8
  3   Traversal engine              1.5             0.5
  4   Linguagem NQL                 3.0             1.0
  5   L-System + Hausdorff          1.5             0.5
  6   Difusão / Pregel              2.0             0.7
  7   Transações ACID               2.5             0.8
  8   Reconsolidação                1.5             0.5
  9   API + SDKs                    1.5             0.5
 10   Benchmark + hardening         2.0             0.7
─────────────────────────────────────────────────────
      TOTAL                        ~20 meses       ~7 meses

Stack de pré-requisitos Rust:
  - Tokio (async)
  - Tonic (gRPC)
  - RocksDB bindings
  - Serde + bincode
  - Criterion (benchmarks)
  - PyO3 (se quiser chamar geoopt do Python)
  - pest (parser NQL)
  - dashmap (concurrent hashmap)
  - uuid, ordered-float, rand
```

---

## O que você tem no final

```
NietzscheDB v1.0 — um banco de dados original em Rust:

  ✓ Vector store hiperbólica nativa (HNSW no Poincaré ball)
    → base: HyperspaceDB
  ✓ Grafo de conhecimento com traversal e Dijkstra hiperbólico
    → construído do zero em RocksDB
  ✓ Linguagem de query declarativa (NQL)
    → parser pest + planner + executor
  ✓ L-System vivo com poda via dimensão de Hausdorff
    → implementado do zero
  ✓ Difusão fractal multi-escala (heat kernel hiperbólico)
    → Chebyshev approximation
  ✓ Transações ACID com saga pattern
    → consistência entre HyperspaceDB e RocksDB
  ✓ Ciclo de reconsolidação (sono) com verificação fractal
    → perturbação + RiemannianAdam + rollback
  ✓ API gRPC + Python/TS SDKs
  ✓ Dashboard web (herdado do HyperspaceDB + extensões)

Licença: AGPL-3.0 (compatível com HyperspaceDB)
         ou compra de licença comercial da YARlabs
```
