# NietzscheDB — Plano de Compatibilidade com EVA-Mind

> **Contexto**: Auditoria realizada em 2026-02-18 comparando EVA-Mind (2× Neo4j, Redis, Qdrant, PostgreSQL)
> com NietzscheDB. Este documento lista **tudo que falta** no NietzscheDB + NQL para absorver o stack
> do EVA-Mind, organizado por prioridade e viabilidade.

---

## Resumo da Auditoria de Compatibilidade

| Database EVA-Mind | Compatibilidade Atual | Bloqueador Principal |
|---|---|---|
| Neo4j #1 (Grafo Paciente) | 45% | MERGE, path patterns `*1..4`, filtros em aresta |
| Neo4j #2 (Meta-Cognitivo EVA) | 70% | MERGE/upsert; resto mapeável |
| Redis (Cache + Audio) | 0% | TTL, List structures, latência sub-ms |
| Qdrant (Vetores) | 15% | Cosine ≠ Poincaré, multi-dim, multi-coleção |
| PostgreSQL (Relacional) | 0% | Sem relacional, PL/pgSQL, LGPD, ACID |

**Compatibilidade global atual: ~26%**
**Meta pós-plano: ~85%** (PostgreSQL e Redis permanecem como satélites opcionais)

---

## O que o EVA-Mind usa de cada banco (referência rápida)

### Neo4j #1 — Grafo do Paciente
```
(:Person)-[:EXPERIENCED]->(:Event {content, emotion, importance, activation_score})
(:Event)-[:RELATED_TO]->(:Topic)
(:Person)-[r:MENTIONED {count, first_mention, last_mention}]->(:Topic)
(:Person)-[r:FEELS {count, first_felt, last_felt}]->(:Emotion)
(:Person)-[:KNOWS]->(:PersonInLife {names[], avg_sentiment, gender})

Queries críticas:
  MERGE (p:Person {id: $id}) ON CREATE SET ... ON MATCH SET ...
  MATCH path = (start)-[:RELATED_TO*1..4]-(related:Event)
  ORDER BY p.last_mention DESC LIMIT 1   ← ORDER BY em prop de aresta
  MERGE (p)-[r:MENTIONED]->(t) ON MATCH SET r.count = r.count + 1
```

### Neo4j #2 — Meta-cognitivo EVA
```
(:EvaSession)-[:HAS_TURN]->(:EvaTurn)-[:ABOUT]->(:EvaTopic)
(:EvaTopic)-[:RELATED_TO]->(:EvaTopic)

Queries: MERGE com ON CREATE/MATCH SET; leitura de tópicos frequentes
```

### Redis
```
audio:<sessionID>            → List  RPUSH/LRANGE  TTL 1h   (buffer PCM streaming)
emb:<md5>                    → String JSON          TTL 24h  (cache embeddings)
sig:<idosoID>:<hash>         → String JSON          TTL 5min (cadeias significantes)
situation:<userID>           → String JSON          TTL 5min (contexto situacional)
cognitive_load:<id>:state    → String JSON          TTL 5min (carga cognitiva)
fdpn:<userID>:<hash>         → String JSON          TTL 5min (subgrafo FDPN)
```

### Qdrant
```
memories          dim=3072  Cosine  Gemini embedding-001   (memórias episódicas)
speaker_embeddings dim=192  Cosine  ECAPA-TDNN ONNX        (impressões digitais de voz)
signifier_chains  dim=3072  Cosine  Gemini                 (cadeias Lacanianas)
eva_self_knowledge dim=3072 Cosine  Gemini                 (auto-conhecimento EVA)
eva_codebase      dim=3072  Cosine  Gemini                 (código-fonte indexado)
nasrudin_stories  dim=768   Cosine  Vertex AI              (histórias Sufi)
aesop_fables      dim=768   Cosine  Vertex AI              (fábulas de Esopo)
+ 9 coleções de sabedoria   dim=768                        (filosofia, meditação, etc.)
eva_learnings     dim=3072  Cosine  Gemini                 (aprendizados contínuos)
```

### PostgreSQL
```
40+ tabelas: idosos, agendamentos, episodic_memories, patient_enneagram,
patient_master_signifiers, lacan_*, clinical_assessments, lgpd_audit_log,
cognitive_load_state, speaker_profiles, personas, ...

PL/pgSQL: calculate_risk_score(), apply_temporal_decay(), audit_response()
Views:    v_patient_mirror_profile, v_weighted_memories, v_active_metaphors, ...
RLS:      Row-Level Security em 11 tabelas (multi-tenancy)
```

---

## PLANO DE DESENVOLVIMENTO — NietzscheDB Engine (Rust)

---

### FASE A — Distância Cosine + Multi-Métrica no VectorStore
**Prioridade: CRÍTICA** | Bloqueia: substituição do Qdrant

#### A.1 — Adicionar Cosine distance ao HNSW
**Arquivo alvo**: `crates/hyperspace-index/src/hnsw.rs`, `crates/hyperspace-core/src/metrics.rs`

```rust
// Adicionar novo enum de métrica
pub enum DistanceMetric {
    PoincareBall,    // existente
    SquaredL2,       // existente
    Cosine,          // NOVO — para embeddings de texto (Gemini, Vertex AI)
    DotProduct,      // NOVO — para bi-encoders
}

// Cosine: 1 - (u·v / (||u|| ||v||))
// Normalizar vetores na inserção → cosine vira produto interno (mais rápido)
// Pré-normalização na ingestão elimina sqrt em runtime
```

O que implementar:
- [ ] `CosineMetric` com pré-normalização de vetores em `upsert()`
- [ ] `DotProductMetric` (vetores já normalizados → produto interno = cosine)
- [ ] Métrica configurável por instância de VectorStore (não só em compile time)
- [ ] Conversão de vetores Gemini (float32[3072]) sem projeção para Poincaré
- [ ] Testes: recall@10 ≥ 0.95 em benchmark MS-MARCO ou similar

#### A.2 — Injetar VectorStore real no servidor (remover MockVectorStore)
**Arquivo alvo**: `crates/nietzsche-server/src/main.rs`, `crates/nietzsche-graph/src/lib.rs`

```rust
// Hoje: MockVectorStore (linear scan) está hardcoded no binário do servidor
// Meta: injetar HyperspaceDB real via gRPC client ou embedding direto

let vector_store = HyperspaceVectorStore::new(
    hyperspace_endpoint,
    metric: DistanceMetric::Cosine,
).await?;
let db = NietzscheDB::open(data_dir, vector_store)?;
```

- [ ] `HyperspaceVectorStore` que implementa `VectorStore` trait via gRPC
- [ ] `EmbeddedVectorStore` — HNSW embutido no mesmo processo (sem rede)
- [ ] Configuração via env vars: `NIETZSCHE_VECTOR_BACKEND=embedded|grpc`

---

### FASE B — Multi-Coleção e Multi-Dimensionalidade
**Prioridade: CRÍTICA** | Bloqueia: substituição do Qdrant

O EVA-Mind tem 17+ coleções com 3 dimensionalidades diferentes (192, 768, 3072).

#### B.1 — Collections namespace no NietzscheDB
**Arquivo alvo**: `crates/nietzsche-graph/src/storage.rs`, novo `crates/nietzsche-graph/src/collection.rs`

```rust
// Cada Collection = um espaço isolado de nós + HNSW próprio + métrica própria
pub struct CollectionConfig {
    pub name: String,
    pub vector_dim: usize,
    pub metric: DistanceMetric,
    pub hnsw_ef_construction: usize,
    pub hnsw_m: usize,
}

pub struct NietzscheDB<V: VectorStore> {
    collections: HashMap<String, Collection<V>>,  // NOVO
    // ...
}

impl NietzscheDB {
    pub fn create_collection(&mut self, config: CollectionConfig) -> Result<()>;
    pub fn drop_collection(&mut self, name: &str) -> Result<()>;
    pub fn collection(&self, name: &str) -> Option<&Collection>;
    pub fn list_collections(&self) -> Vec<CollectionInfo>;
}
```

- [ ] RocksDB Column Family por coleção (prefixo no key: `col:{name}:nodes:`, `col:{name}:edges:`)
- [ ] HNSW index independente por coleção (dimensão + métrica configurável)
- [ ] WAL com `collection_id` no header de cada entrada
- [ ] gRPC: `CreateCollection`, `DropCollection`, `ListCollections`
- [ ] `InsertNode` recebe `collection: Option<String>` (default = "default")
- [ ] `KnnSearch` recebe `collection: String`

#### B.2 — Filtro por metadata durante KNN (pushed-down)
**Arquivo alvo**: `crates/hyperspace-index/src/hnsw.rs`

```rust
// Hoje: KNN retorna top-K e aplicação filtra depois
// Meta: filtro durante busca HNSW (pré-filtragem ou pós-filtragem eficiente)

pub fn knn_filtered(
    &self,
    query: &Vector,
    k: usize,
    filter: &MetadataFilter,  // NOVO
) -> Result<Vec<(Uuid, f64)>>;

pub enum MetadataFilter {
    FieldEq { field: String, value: serde_json::Value },
    FieldIn { field: String, values: Vec<serde_json::Value> },
    And(Vec<MetadataFilter>),
    Or(Vec<MetadataFilter>),
    None,
}
```

- [ ] Roaring Bitmap por valor de campo (como HyperspaceDB já tem em `feito.md`)
- [ ] Pré-filtro: candpool apenas de nós que passam no filtro
- [ ] gRPC: `KnnSearch` recebe campo `filter: MetadataFilterProto`
- [ ] Equivale ao `SearchWithScore(ctx, col, emb, k, 0.7, userID)` do Qdrant

---

### FASE C — Substituição Completa do Redis
**Prioridade: ALTA** | Meta: eliminar Redis do stack do EVA-Mind totalmente

#### Análise: o que o Redis faz no EVA-Mind

| Key Pattern | Estrutura | TTL | Frequência | Substituível? |
|---|---|---|---|---|
| `emb:<md5>` | String JSON | 24h | Altíssima (toda query) | ✅ C.1 + block cache |
| `sig:<id>:<hash>` | String JSON | 5min | Alta | ✅ C.1 |
| `situation:<id>` | String JSON | 5min | Alta | ✅ C.1 |
| `cognitive_load:<id>:state` | String JSON | 5min | Média | ✅ C.1 |
| `fdpn:<id>:<hash>` | String JSON | 5min | Média | ✅ C.1 |
| `audio:<sessionID>` | **List** RPUSH/LRANGE | 1h | Alta (20 chunks/s) | ✅ C.2 |

**Conclusão**: Redis é 100% substituível após C.1 + C.2 + block cache configurado.

#### Análise de latência — RocksDB vs Redis

```
Redis (in-memory):          0.05 – 0.1 ms
RocksDB block cache hit:    0.1  – 0.5 ms   ← diferença irrelevante
RocksDB disk miss:          1    – 5   ms   ← nunca acontece para hot keys

Hot keys do EVA-Mind (emb:, sig:, situation:) têm acesso frequente → ficam
no block cache do RocksDB. Gemini API = 50-200ms. Economizar 0.4ms no cache
é irrelevante. A troca é viável.
```

#### C.1 — TTL por nó + JanitorTask (substitui 5 de 6 usos do Redis)
**Arquivo alvo**: `crates/nietzsche-graph/src/model.rs`, `crates/nietzsche-graph/src/storage.rs`

```rust
pub struct Node {
    // ... campos existentes ...
    pub expires_at: Option<i64>,  // NOVO — unix timestamp; None = eterno
}
```

Como o EVA-Mind usaria (Go SDK):
```go
// Antes (Redis):
cache.Set(ctx, "emb:"+hash, jsonBytes, 24*time.Hour)

// Depois (NietzscheDB):
client.InsertNode(ctx, &nietzsche.Node{
    Collection: "cache",
    Content:    jsonPayload,
    Metadata:   map[string]any{"key": "emb:" + hash},
    Ttl:        86400,  // 24h em segundos → vira expires_at = now() + 86400
})
// Leitura:
client.Query(ctx, `
    MATCH (n) IN COLLECTION 'cache'
    WHERE n.metadata.key = $key
    RETURN n.content
`, map[string]any{"key": "emb:" + hash})
```

- [ ] Campo `expires_at: Option<i64>` no `Node` e no `TableRow`
- [ ] WAL entry `SetExpiry { id, expires_at }`
- [ ] `GraphStorage::scan_expired()` → iterator de IDs expirados (índice em `expires_at`)
- [ ] Background task `JanitorTask` — configurable interval (default 30s para TTL curtos)
- [ ] `NIETZSCHE_JANITOR_INTERVAL_SECS` env var
- [ ] RocksDB block cache configurado para 512MB–2GB: `NIETZSCHE_BLOCK_CACHE_MB` (default 512)
- [ ] NQL: `CREATE (n {ttl: 300})` → seta `expires_at = now() + 300`
- [ ] NQL: `MATCH (n) WHERE n.ttl_remaining > 0 RETURN n` — campo virtual calculado
- [ ] Collection `cache` dedicada (dimensão mínima, sem HNSW, só KV lookup)
- [ ] Índice automático em `expires_at` para o janitor não fazer full scan

#### C.2 — ListStore (substitui `audio:<sessionID>` RPUSH/LRANGE)
**Arquivo alvo**: novo `crates/nietzsche-graph/src/list_store.rs`

```rust
// RocksDB CF "lists"
// key = "{list_name}:{seq:08x}" — seq monotônico por lista (AtomicU64)
// RPUSH: incrementa seq, escreve
// LRANGE: scan por prefixo ordenado, coleta range
// TTL: a própria lista tem expires_at; janitor deleta todo o prefixo

pub struct ListStore {
    db: Arc<DB>,
    counters: DashMap<String, AtomicU64>,  // seq por lista (in-memory, rebuilt on startup)
}

impl ListStore {
    pub fn rpush(&self, list: &str, data: &[u8]) -> Result<u64>;
    // key = "list:{name}:{seq:016x}" → garante ordem lexicográfica = inserção
    pub fn lrange(&self, list: &str, start: i64, stop: i64) -> Result<Vec<Vec<u8>>>;
    pub fn llen(&self, list: &str) -> Result<u64>;
    pub fn del(&mut self, list: &str) -> Result<()>;  // apaga todo o prefixo
    pub fn expire(&mut self, list: &str, ttl_secs: u64) -> Result<()>;
}
```

Como o EVA-Mind usaria (Go SDK):
```go
// Antes (Redis):
pipe.RPush(ctx, "audio:"+sessionID, pcmChunk)
pipe.Expire(ctx, "audio:"+sessionID, 1*time.Hour)
chunks := client.LRange(ctx, "audio:"+sessionID, 0, -1)

// Depois (NietzscheDB):
client.ListRPush(ctx, "audio:"+sessionID, pcmChunk, 3600) // ttl=1h
chunks := client.ListLRange(ctx, "audio:"+sessionID, 0, -1)
```

- [ ] RocksDB CF `lists` com prefixo `{list_name}:{seq:016x}`
- [ ] `AtomicU64` por lista para seq monotônico (rebuilt from CF scan no startup)
- [ ] `lrange` com `ReadOptions::iterate_range(prefix)` — O(k) onde k = range size
- [ ] TTL de lista: chave meta `list_ttl:{name}` → janitor apaga todo o prefixo quando expirar
- [ ] gRPC: `ListRPush`, `ListLRange`, `ListLen`, `ListDel`
- [ ] Throughput alvo: 20 RPUSH/segundo por sessão sem contenção (audio PCM rate)

#### C.3 — Block Cache e Collection "cache" dedicada
**Arquivo alvo**: `crates/nietzsche-server/src/main.rs`, `crates/nietzsche-graph/src/storage.rs`

```rust
// RocksDB com block cache generoso para hot keys
let mut opts = Options::default();
opts.set_block_based_table_factory(&BlockBasedOptions {
    block_cache: Cache::new_lru_cache(block_cache_bytes),  // default 512MB
    ..Default::default()
});

// Collection "cache" — sem HNSW, apenas KV lookup por metadata.key
// Nós de cache não têm embedding (embedding = zeros de dimensão 1)
// Índice em metadata.key para lookup O(1)
```

- [ ] Collection `cache` criada automaticamente no startup
- [ ] Lookup `metadata.key = $key` usa índice secundário (Fase E) → O(log n)
- [ ] `NIETZSCHE_BLOCK_CACHE_MB` env var (default: 512)
- [ ] Benchmark: latência p99 de get/set no collection `cache` vs Redis local

#### Checklist Fase C — Redis completo

- [ ] **C.1** `expires_at` em Node + JanitorTask
- [ ] **C.1** Collection `cache` com índice em `metadata.key`
- [ ] **C.1** NQL `ttl:` shorthand em CREATE
- [ ] **C.2** `ListStore` (RPUSH/LRANGE/DEL/TTL) em RocksDB CF `lists`
- [ ] **C.2** gRPC: `ListRPush`, `ListLRange`, `ListLen`, `ListDel`
- [ ] **C.3** RocksDB block cache configurável
- [ ] **C.X** Go SDK: `CacheGet`, `CacheSet`, `CacheDelete`, `ListRPush`, `ListLRange`

#### Migração do EVA-Mind (Redis → NietzscheDB)

| Componente EVA | Redis key | Substituto NietzscheDB |
|---|---|---|
| `embedding_cache.go` | `emb:<md5>` TTL 24h | `CacheSet("emb:"+hash, data, 86400)` |
| `embedding_cache.go` | `sig:<id>:<hash>` TTL 5min | `CacheSet("sig:"+..., data, 300)` |
| `fdpn_situational.go` | `situation:<id>` TTL 5min | `CacheSet("situation:"+id, data, 300)` |
| `cognitive/` | `cognitive_load:<id>:state` TTL 5min | `CacheSet("cog:"+id, data, 300)` |
| `fdpn_engine.go` | `fdpn:<id>:<hash>` TTL 5min | `CacheSet("fdpn:"+..., data, 300)` |
| `audio_analysis.go` | `audio:<sid>` List TTL 1h | `ListRPush("audio:"+sid, chunk, 3600)` |

#### Métricas de sucesso — Redis substituído
```
✓ Redis eliminado quando:
  → CacheGet latência p99 < 2ms (vs Redis 0.1ms — aceitável dado Gemini 50-200ms)
  → ListRPush throughput ≥ 50 ops/s por sessão (audio PCM rate = 20/s)
  → JanitorTask não interfere com reads (background isolado)
  → Zero perda de chunks de áudio em stress test de 1h
  → EVA-Mind rodando sem variável REDIS_HOST no .env
```

---

### FASE D — MERGE Semântico e Upsert
**Prioridade: ALTA** | Bloqueia: substituição dos dois Neo4j

O MERGE é o padrão mais usado no EVA-Mind em Neo4j.

#### D.1 — MERGE no gRPC API
**Arquivo alvo**: `crates/nietzsche-api/src/handlers.rs`, `proto/nietzsche.proto`

```protobuf
// NOVO RPC
rpc MergeNode(MergeNodeRequest) returns (MergeNodeResponse);
rpc MergeEdge(MergeEdgeRequest) returns (MergeEdgeResponse);

message MergeNodeRequest {
    string collection = 1;
    string match_field = 2;       // campo de lookup (ex: "name")
    google.protobuf.Value match_value = 3;
    NodeData on_create = 4;       // dados a usar na criação
    map<string, google.protobuf.Value> on_match_set = 5; // campos a atualizar se existir
}

message MergeNodeResponse {
    string id = 1;
    bool created = 2;  // true = criado, false = já existia (matched)
}
```

- [ ] `GraphStorage::find_by_field(field, value)` — scan com early-exit
- [ ] `NietzscheDB::merge_node(match_field, match_value, on_create, on_match_set)`
- [ ] `NietzscheDB::merge_edge(from, to, edge_type, on_create, on_match_set)`
- [ ] Atomicidade: lock do nó durante check + write para evitar race condition
- [ ] Índice secundário por campo para `merge` eficiente (ver Fase E)
- [ ] Testes: merge idempotente; contadores incrementam corretamente

#### D.2 — Incremento atômico em metadata de aresta
```rust
// Para: MERGE (p)-[r:MENTIONED]->(t) ON MATCH SET r.count = r.count + 1
pub fn increment_edge_metadata(
    &mut self,
    edge_id: Uuid,
    field: &str,
    delta: i64,
) -> Result<i64>; // retorna novo valor
```

---

### FASE E — Índices Secundários por Propriedade
**Prioridade: ALTA** | Bloqueia: MERGE eficiente + NQL sem full scan

Hoje todo MATCH em NQL faz full scan O(n). Para o EVA-Mind com milhares de memórias por paciente, isso é inaceitável.

#### E.1 — B-Tree index em campos de metadata
**Arquivo alvo**: novo `crates/nietzsche-graph/src/secondary_index.rs`

```rust
pub struct SecondaryIndex {
    // RocksDB CF "idx:{field_name}": key = field_value_bytes:node_id, value = empty
    // Suporta: range scan, equality, IN list
}

impl NietzscheDB {
    pub fn create_index(&mut self, field: &str) -> Result<()>;
    pub fn drop_index(&mut self, field: &str) -> Result<()>;
    pub fn list_indexes(&self) -> Vec<String>;

    // Usado internamente pelo NQL executor
    pub(crate) fn index_scan(
        &self,
        field: &str,
        op: IndexOp,
    ) -> Result<impl Iterator<Item = Uuid>>;
}
```

Campos a indexar inicialmente (baseado no uso do EVA-Mind):
- `energy` — range scan frequente
- `node_type` — equality filter
- `created_at` — range + ORDER BY
- `expires_at` — range (janitor)
- Qualquer campo de `metadata` declarado pelo usuário

- [ ] gRPC: `CreateIndex(field)`, `DropIndex(field)`, `ListIndexes()`
- [ ] NQL: `CREATE INDEX ON :Node(energy)` — syntax inspirada em Cypher
- [ ] NQL executor usa índice automaticamente quando disponível (query planner simples)

---

### FASE F — Sensory RPCs (completar stubs)
**Prioridade: MÉDIA** | Hoje: gRPC handlers retornam `unimplemented`

**Arquivos alvo**: `crates/nietzsche-api/src/handlers.rs`
Os tipos e o storage já existem em `nietzsche-sensory`. Só falta conectar.

- [ ] `InsertSensory` — recebe `SensoryMemoryProto`, desserializa, chama `storage.put_sensory()`
- [ ] `GetSensory` — busca por node_id no CF "sensory" do RocksDB
- [ ] `Reconstruct` — busca sensory, roda decode pipeline (F32/F16/Int8/PQ), retorna bytes + quality
- [ ] `DegradeSensory` — aplica quantização forcada para nível especificado
- [ ] Testes end-to-end via nietzsche-sdk

---

### FASE G — Concorrência: Mutex → RwLock por Collection
**Prioridade: MÉDIA** | Hoje: mutex único serializa tudo

**Arquivo alvo**: `crates/nietzsche-server/src/main.rs`

```rust
// Hoje:
Arc<tokio::sync::Mutex<NietzscheDB<V>>>  // ← gargalo

// Meta:
Arc<NietzscheDB<V>>  // NietzscheDB internamente usa DashMap por collection
                     // reads em paralelo, writes com RwLock por shard
```

- [ ] `NietzscheDB` usa `tokio::sync::RwLock` por collection em vez de mutex global
- [ ] Reads concorrentes sem bloqueio entre si
- [ ] Writes bloqueiam só a collection afetada (não o banco inteiro)
- [ ] Benchmark: throughput com 100 clientes concorrentes antes e depois

---

## PLANO DE DESENVOLVIMENTO — NQL (Nietzsche Query Language)

---

### NQL-1 — MERGE Statement
**Prioridade: CRÍTICA** | Equivale ao MERGE do Cypher

```sql
-- Criar ou encontrar por campo único
MERGE (n:EvaSession {id: $sessionId})
ON CREATE SET n.started_at = $ts, n.turn_count = 0, n.status = 'active'
ON MATCH SET n.turn_count = n.turn_count + 1

-- Merge de aresta entre dois nós existentes
MATCH (p:Person {id: $pid}), (t:Topic {name: $name})
MERGE (p)-[r:MENTIONED]->(t)
ON CREATE SET r.count = 1, r.first_mention = $ts
ON MATCH SET r.count = r.count + 1, r.last_mention = $ts
```

**Arquivo alvo**: `crates/nietzsche-query/src/ast.rs`, `parser.pest`, `executor.rs`

AST a adicionar:
```rust
pub enum Statement {
    Match(MatchQuery),       // existente
    Diffuse(DiffuseQuery),   // existente
    Explain(Box<Statement>), // existente
    Reconstruct(...),        // existente
    Merge(MergeQuery),       // NOVO
    Create(CreateQuery),     // NOVO
    Set(SetQuery),           // NOVO
    Delete(DeleteQuery),     // NOVO
    CreateIndex(IndexQuery), // NOVO
}

pub struct MergeQuery {
    pub pattern: MergePattern,           // (n:Label {field: $val})
    pub on_create: Vec<SetClause>,       // ON CREATE SET ...
    pub on_match: Vec<SetClause>,        // ON MATCH SET ...
    pub return_clause: Option<Return>,
}
```

- [ ] Grammar `pest`: regra `merge_statement`, `on_create_clause`, `on_match_clause`
- [ ] AST: `MergeQuery`, `SetClause`, `MergePattern`
- [ ] Executor: lookup por campo → branch create/update → aplicar SET clauses
- [ ] Suporte a `MERGE (a)-[:TYPE]->(b)` (aresta MERGE)
- [ ] Testes: idempotência, contadores atômicos, branch create vs match

---

### NQL-2 — Multi-hop Typed Path Pattern
**Prioridade: CRÍTICA** | Equivale ao `*1..4` do Cypher

```sql
-- Traversal tipado com label e profundidade variável
MATCH (start:Event {id: $id})-[:RELATED_TO*1..4]-(related:Event)
WHERE related.id <> $id
RETURN DISTINCT related.id, related.importance
ORDER BY related.importance DESC
LIMIT 10

-- Com direção e múltiplos tipos
MATCH (p:Person {id: $pid})-[:EXPERIENCED|RELATED_TO*1..3]->(e:Event)
WHERE e.emotion = 'tristeza'
RETURN e
```

**Arquivo alvo**: `crates/nietzsche-query/src/ast.rs`, `executor.rs`

```rust
pub struct PathPattern {
    pub start: NodePattern,
    pub steps: Vec<PathStep>,
}

pub struct PathStep {
    pub edge_types: Vec<String>,     // [:RELATED_TO] ou [:A|B]
    pub min_hops: usize,             // *1..4 → min=1
    pub max_hops: usize,             // *1..4 → max=4
    pub direction: Direction,        // Outgoing | Incoming | Both
    pub target: NodePattern,
}

pub struct NodePattern {
    pub alias: Option<String>,
    pub label: Option<String>,       // :Event, :Person, etc.
    pub properties: HashMap<String, Expr>,
}
```

Executor:
- [ ] BFS iterativo com depth tracking e label filter em cada hop
- [ ] Edge type filter durante traversal (só seguir arestas do tipo especificado)
- [ ] `DISTINCT` por padrão nos resultados de path
- [ ] `length(path)` como campo calculável no RETURN
- [ ] Limite de hops para evitar explosão combinatória (máx configurável, default 10)

---

### NQL-3 — SET Statement (Update de propriedades)
**Prioridade: ALTA**

```sql
-- Update simples
MATCH (n:EvaSession {id: $id})
SET n.turn_count = n.turn_count + 1, n.status = 'completed'
RETURN n

-- Update condicional
MATCH (n:Node)
WHERE n.energy < 0.1
SET n.energy = 0.0
```

```rust
pub struct SetQuery {
    pub pattern: NodePattern,
    pub where_clause: Option<Condition>,
    pub assignments: Vec<Assignment>,
}

pub struct Assignment {
    pub field: String,
    pub value: Expr,   // pode ser n.count + 1 (aritmética)
}
```

- [ ] Grammar: `SET alias.field = expr (COMMA alias.field = expr)*`
- [ ] Expressões aritméticas em SET: `n.count = n.count + 1`
- [ ] Batching: `SET` em múltiplos nós de uma query MATCH
- [ ] WAL entry para cada node atualizado

---

### NQL-4 — CREATE Statement Explícito
**Prioridade: ALTA**

```sql
-- Criar nó com coleção e TTL
CREATE (n:EvaSession {
    id: $id,
    started_at: $ts,
    turn_count: 0,
    ttl: 3600
})
RETURN n.id

-- Criar aresta entre dois nós existentes
MATCH (s:EvaSession {id: $sid}), (t:EvaTurn {id: $tid})
CREATE (s)-[:HAS_TURN {created_at: $ts}]->(t)
```

```rust
pub struct CreateQuery {
    pub pattern: CreatePattern,   // nó ou aresta
    pub return_clause: Option<Return>,
}
```

- [ ] Grammar: `CREATE (alias:Label {props}) RETURN ...`
- [ ] Suporte a criar aresta via `MATCH ... CREATE (a)-[:TYPE]->(b)`
- [ ] `ttl: 3600` em props → converte para `expires_at = now() + 3600`

---

### NQL-5 — DELETE by Pattern
**Prioridade: ALTA**

```sql
-- Delete por padrão
MATCH (n:EvaSession {id: $id})
DELETE n

-- Delete aresta
MATCH (p:Person {id: $pid})-[r:MENTIONED]->(t:Topic {name: $name})
DELETE r

-- Delete em cascata
MATCH (s:EvaSession {id: $id})
DETACH DELETE s   -- deleta nó + todas arestas incidentes
```

- [ ] Grammar: `DELETE alias` e `DETACH DELETE alias`
- [ ] Executor: busca nó pelo padrão, chama `delete_node()` ou `delete_edge()`
- [ ] `DETACH DELETE`: remove nó + todas as arestas incidentes (já existe na engine, só expor via NQL)

---

### NQL-6 — Acesso a Propriedades de Aresta no WHERE/ORDER BY
**Prioridade: ALTA** | Bloqueia: pronoun resolution do EVA-Mind

```sql
-- ORDER BY em propriedade de aresta
MATCH (patient:Person {id: $pid})-[r:KNOWS]->(p:PersonInLife)
WHERE $gender = 'any' OR p.gender = $gender
RETURN p.id, p.names, r.last_mention
ORDER BY r.last_mention DESC
LIMIT 1

-- WHERE em propriedade de aresta
MATCH (p:Person)-[r:MENTIONED {count > 5}]->(t:Topic)
RETURN t.name, r.count
ORDER BY r.count DESC
```

- [ ] Parser: `r.field` onde `r` é alias de aresta no MATCH pattern
- [ ] Executor: ao resolver path pattern, carregar aresta + expor `r.*` no contexto de avaliação
- [ ] ORDER BY em campos de aresta: sort sobre `Vec<(Node, Edge, Node)>`
- [ ] WHERE em campos de aresta: filtro em `edge.metadata[field]`

---

### NQL-7 — COLLECTION Qualifier (Multi-coleção)
**Prioridade: ALTA** | Depende de Fase B (engine)

```sql
-- Query em coleção específica
MATCH (n:Memory) IN COLLECTION 'memories'
WHERE n.idoso_id = $pid
RETURN n LIMIT 20

-- KNN em coleção específica
MATCH KNN(n, $embedding, k=10) IN COLLECTION 'memories'
WHERE n.idoso_id = $pid
RETURN n, COSINE_DIST(n.embedding, $embedding) AS score
ORDER BY score DESC

-- KNN em coleção de sabedoria
MATCH KNN(n, $embedding, k=5) IN COLLECTION 'nasrudin_stories'
WHERE COSINE_DIST(n.embedding, $embedding) > 0.7
RETURN n
```

- [ ] Grammar: `IN COLLECTION 'name'` após padrão de nó
- [ ] `MATCH KNN(alias, $vec, k=N)` — syntax para busca vetorial
- [ ] `COSINE_DIST(n.embedding, $vec)` — função de distância Cosine (além de HYPERBOLIC_DIST)
- [ ] `L2_DIST(n.embedding, $vec)` — distância Euclidiana
- [ ] Executor roteia para collection correta

---

### NQL-8 — WITH Clause (Pipeline de queries)
**Prioridade: MÉDIA**

```sql
-- Pipeline: primeiro encontra nós, depois usa como input
MATCH (p:Person {id: $pid})-[:EXPERIENCED]->(e:Event)
WHERE e.importance > 0.7
WITH e
ORDER BY e.importance DESC
LIMIT 5
MATCH (e)-[:RELATED_TO*1..3]-(related:Event)
RETURN related
```

- [ ] Grammar: `WITH alias (COMMA alias)* (ORDER BY ...)? (LIMIT n)?`
- [ ] Executor: resultado do estágio anterior vira input do próximo MATCH
- [ ] Equivale ao `WITH` do Cypher para pipeline de traversal

---

### NQL-9 — CREATE INDEX Syntax
**Prioridade: MÉDIA** | Depende de Fase E (engine)

```sql
-- Criar índice em campo de metadata
CREATE INDEX ON :Node(energy)
CREATE INDEX ON :Node(created_at)
CREATE INDEX ON :EvaSession(started_at)
CREATE INDEX ON :Event(importance)

-- Listar índices
SHOW INDEXES

-- Remover índice
DROP INDEX ON :Node(energy)
```

- [ ] Grammar: `create_index_statement`, `show_indexes_statement`, `drop_index_statement`
- [ ] Executor: chama `db.create_index(field)`, `db.list_indexes()`, `db.drop_index(field)`

---

### NQL-10 — Funções de Tempo e Datetime
**Prioridade: MÉDIA** | O EVA-Mind usa `datetime()` extensivamente

```sql
-- Funções de tempo
MATCH (n:EvaSession)
WHERE n.created_at > NOW() - INTERVAL 7 DAYS
RETURN n

-- datetime literal
CREATE (s:EvaSession {started_at: datetime($ts)})

-- Agora
MATCH (n) WHERE n.expires_at < TIMESTAMP()
```

Funções a adicionar:
- [ ] `NOW()` → unix timestamp atual como i64
- [ ] `TIMESTAMP()` → alias de NOW()
- [ ] `INTERVAL N DAYS/HOURS/MINUTES` → duração em segundos como i64
- [ ] `datetime($iso_string)` → parse RFC3339 para i64
- [ ] Operadores aritméticos em timestamps: `$ts - INTERVAL 7 DAYS`

---

### NQL-11 — FOREACH / Batch Operations
**Prioridade: BAIXA**

```sql
-- Aplicar operação em múltiplos resultados de subquery
MATCH (p:Person {id: $pid})-[:EXPERIENCED]->(e:Event)
WHERE e.energy < 0.1
FOREACH (n IN [e] | SET n.energy = 0.0)
```

- [ ] Grammar: `FOREACH (alias IN expr | statement)`
- [ ] Executor: itera sobre lista, aplica statement a cada elemento

---

## PLANO DE COMPATIBILIDADE COM EVA-MIND

### O que migrar para NietzscheDB (após fases acima)

| Componente EVA-Mind | Fases necessárias | Esforço estimado |
|---|---|---|
| `graph_store.go` (Neo4j #1 escrita) | NQL-1 (MERGE), NQL-3 (SET) | Médio — reescrever em Go SDK |
| `retrieval.go` (Neo4j #1 leitura) | NQL-2 (path *1..4), NQL-6 (edge props) | Alto — padrões complexos |
| `eva_memory.go` (Neo4j #2) | NQL-1 (MERGE), NQL-3 (SET) | Baixo — grafo simples |
| `fdpn_engine.go` (spreading activation) | NQL-2 (path), NQL-7 (collection) | Alto — lógica de ativação |
| `wisdom_service.go` (Qdrant multi-col) | Fase B (multi-col) + NQL-7 | Alto — 11 coleções |
| `storage.go` (memórias episódicas) | Fase A (Cosine) + Fase B (multi-col) | Alto — pipeline completo |
| `embedding_cache.go` `emb:<md5>` TTL 24h | Fase C.1 + C.3 (block cache) | Baixo — `CacheSet/Get` |
| `embedding_cache.go` `sig:<id>:<hash>` TTL 5min | Fase C.1 | Baixo — `CacheSet/Get` |
| `fdpn_situational.go` `situation:<id>` TTL 5min | Fase C.1 | Baixo — `CacheSet/Get` |
| `cognitive/` `cognitive_load:<id>:state` TTL 5min | Fase C.1 | Baixo — `CacheSet/Get` |
| `fdpn_engine.go` `fdpn:<id>:<hash>` TTL 5min | Fase C.1 | Baixo — `CacheSet/Get` |
| `audio_analysis.go` `audio:<sid>` List TTL 1h | Fase C.2 (ListStore) | Médio — `ListRPush/LRange` |

### O que NÃO migrar (manter como está — decisão definitiva)

| Database | Decisão | Motivo |
|---|---|---|
| **PostgreSQL** | ✅ Manter permanentemente | É a fonte de verdade clínica — 40+ tabelas, LGPD, PL/pgSQL, multi-tenancy RLS. NietzscheDB complementa, não substitui. |
| **Redis** | ✅ Substituível após Fase C completa | Fase C.1 (TTL) substitui 5 de 6 usos. Fase C.2 (List) substitui o último (audio PCM). Redis eliminado totalmente. |

---

## Ordem de Execução Recomendada

```
FASE A  →  FASE B  →  FASE D  →  FASE E  →  NQL-1 + NQL-2 + NQL-3
  ↑             ↑          ↑          ↑             ↑
Cosine      Multi-col   MERGE    Índices      Linguagem completa
(Qdrant)    (Qdrant)   (Neo4j)  (perf)       (Neo4j + Qdrant)

Depois:
  NQL-4 → NQL-5 → NQL-6 → NQL-7 → FASE C → FASE F → FASE G
  CREATE   DELETE  Edge    Coll    TTL+List Sensory  Concurr.
                                   Redis↓

Paralelo (não bloqueia o caminho crítico):
  FASE C.2 → FASE I.1 → FASE I.2 → FASE I.3 → FASE I.4
  ListStore   OpenDAL    PCM flow   Fotos/Vídeo  gRPC Media
  (audio buf) (MediaStore) (consolida) (CLIP/MAE) (5 RPCs)

Por último (não bloqueia o resto):
  FASE H.1 → FASE H.2 → FASE H.3 → FASE H.4 → FASE H.5
  TableStore  NQL TABLE  gRPC RPCs  Migração    Hybrid JOIN
  (SQLite)    statements 8 métodos  EVA-Mind    TABLE+NODE+VEC
```

### Critérios de "pronto para migrar Neo4j"
- [ ] MERGE idempotente funcionando via gRPC e NQL
- [ ] Path pattern `*1..4` com label filter executando em <100ms para 10k nós
- [ ] ORDER BY em propriedades de aresta
- [ ] Índice secundário em pelo menos 3 campos
- [ ] `nietzsche-sdk` Go client funcional (hoje só Rust)

### Critérios de "pronto para migrar Qdrant"
- [ ] Cosine distance implementado e validado (recall@10 ≥ 0.95)
- [ ] Pelo menos 5 coleções independentes com dimensões diferentes
- [ ] KNN com filtro pushed-down por metadata
- [ ] Ingestão de 3072-dim vetores Gemini sem projeção

---

## Go SDK — ~~Pré-requisito para Migração~~ FEITO (2026-02-20)

**Nome**: `sdk-papa-caolho`
**Localização**: `sdks/go/`
**Módulo Go**: `sdk-papa-caolho`
**Cobertura**: 22/22 RPCs | 25 testes unitários | go build + go vet limpos

```go
import nietzsche "sdk-papa-caolho"

client, _ := nietzsche.ConnectInsecure("localhost:50052")
defer client.Close()

// Insert node
node, _ := client.InsertNode(ctx, nietzsche.InsertNodeOpts{
    Coords:     embedding,
    Content:    map[string]string{"text": "memory"},
    NodeType:   "Semantic",
    Collection: "memories",
})

// KNN search
results, _ := client.KnnSearch(ctx, embedding, 20, "memories")

// NQL query with typed params
qr, _ := client.Query(ctx,
    "MATCH (m:Memory) WHERE m.energy > $min RETURN m LIMIT 10",
    map[string]interface{}{"min": 0.5}, "memories")

// Sleep cycle
sleep, _ := client.TriggerSleep(ctx, nietzsche.SleepOpts{Collection: "memories"})

// Zaratustra evolution
z, _ := client.InvokeZaratustra(ctx, nietzsche.ZaratustraOpts{Cycles: 1})
```

- [x] Gerado a partir do `.proto` via `protoc-gen-go-grpc`
- [x] Wrapper idiomático Go (sem expor proto diretamente)
- [x] Métodos: `Connect`, `ConnectInsecure`, `Close`, `InsertNode`, `GetNode`, `DeleteNode`,
      `UpdateEnergy`, `InsertEdge`, `DeleteEdge`, `Query`, `KnnSearch`, `Bfs`, `Dijkstra`,
      `Diffuse`, `CreateCollection`, `DropCollection`, `ListCollections`, `TriggerSleep`,
      `InvokeZaratustra`, `InsertSensory`, `GetSensory`, `Reconstruct`, `DegradeSensory`,
      `GetStats`, `HealthCheck`
- [x] Context propagation nativo (todos os métodos aceitam `context.Context`)
- [ ] MergeNode / MergeEdge (aguarda FASE D no servidor)
- [ ] CacheGet/CacheSet/ListRPush/ListLRange (aguarda FASE C no servidor)
- [ ] Exemplos em `sdks/go/examples/`

---

## Checklist Resumido

### Engine (Rust)
- [ ] **A.1** Cosine distance no HNSW
- [ ] **A.2** HyperspaceVectorStore real (remover MockVectorStore do servidor)
- [ ] **B.1** Collections namespace (HNSW + RocksDB por coleção)
- [ ] **B.2** KNN com filtro de metadata pushed-down
- [ ] **C.1** `expires_at` em Node + JanitorTask + Collection `cache` com índice
- [ ] **C.2** `ListStore` RPUSH/LRANGE em RocksDB CF `lists` + gRPC
- [ ] **C.3** RocksDB block cache configurável (512MB–2GB)
- [ ] **C.X** Go SDK: `CacheGet`, `CacheSet`, `ListRPush`, `ListLRange`
- [ ] **D.1** MergeNode + MergeEdge gRPC
- [ ] **D.2** Incremento atômico em metadata de aresta
- [ ] **E.1** Índices secundários por campo (RocksDB B-tree)
- [ ] **F**   Sensory RPCs (InsertSensory, GetSensory, Reconstruct, DegradeSensory)
- [ ] **G**   RwLock por collection (remover mutex global)

### NQL
- [ ] **NQL-1**  MERGE ... ON CREATE SET ... ON MATCH SET
- [ ] **NQL-2**  Path pattern `(a:L)-[:T*min..max]->(b:L)`
- [ ] **NQL-3**  SET alias.field = expr (com aritmética)
- [ ] **NQL-4**  CREATE (n:Label {props}) e CREATE aresta
- [ ] **NQL-5**  DELETE alias e DETACH DELETE alias
- [ ] **NQL-6**  Acesso a propriedades de aresta `r.field` em WHERE/ORDER BY
- [ ] **NQL-7**  `IN COLLECTION 'name'` + `MATCH KNN(n, $v, k=N)`
- [ ] **NQL-8**  WITH clause (pipeline)
- [ ] **NQL-9**  CREATE/DROP/SHOW INDEX
- [ ] **NQL-10** Funções de tempo: NOW(), INTERVAL, datetime()
- [ ] **NQL-11** FOREACH batch (baixa prioridade)

### Table Store — Fase H
- [ ] **H.1** `TableStore` com SQLite embutido via `rusqlite` (feature `bundled`) — CRUD + WAL + ACID grátis
- [ ] **H.2** NQL: `CREATE TABLE`, `INSERT TABLE`, `MERGE TABLE`, `MATCH TABLE`, `DROP TABLE`, `SHOW TABLES`
- [ ] **H.3** gRPC: 8 novos RPCs de Table Store
- [ ] **H.4** Scripts de migração das 8 tabelas candidatas do EVA-Mind
- [ ] **H.5** Hybrid JOIN `TABLE + NODE + VECTOR` na NQL

### Media Store — Fase I
- [ ] **I.1** `MediaStore` com OpenDAL — backend `fs` (dev) + `gcs`/`s3` (prod)
- [ ] **I.1** `NIETZSCHE_MEDIA_BACKEND` env var (`fs` | `s3` | `gcs`)
- [ ] **I.2** Fluxo PCM: `ListStore → ConsolidateAudio → MediaStore → Graph node`
- [ ] **I.3** Coleção `speaker_embeddings` dim=192 Cosine + `patient_faces` dim=768
- [ ] **I.4** gRPC: `MediaPut`, `MediaGet`, `MediaDelete`, `MediaList`, `ConsolidateAudio`

### SDK
- [x] **SDK-Go** Cliente Go idiomático gerado do proto + wrapper — `sdk-papa-caolho` (22 RPCs, 25 testes)
- [ ] **SDK-Go** Adicionar métodos Table Store + Media Store (quando FASE H/I estiverem prontas no servidor)

### Métricas de Sucesso Finais
```
✓ Neo4j #1 substituível quando:
  → MERGE + path *1..4 + edge props + índices funcionando
  → Recall de memórias episódicas ≥ Neo4j em benchmark A/B

✓ Qdrant substituível quando:
  → Cosine recall@10 ≥ 0.95 vs Qdrant
  → Multi-coleção com dim 192 + 768 + 3072 funcionando
  → KNN filtrado por idoso_id < 10ms p95

✓ Redis ELIMINADO quando:
  → CacheGet p99 < 2ms (RocksDB block cache quente)
  → ListRPush ≥ 50 ops/s por sessão (PCM audio rate ok)
  → JanitorTask rodando sem impacto em leituras
  → EVA-Mind sem REDIS_HOST no .env — zero dependência

✓ PostgreSQL: PERMANECE — decisão definitiva, não entra no escopo de migração

✓ Table Store (Fase H) pronto quando:
  → CRUD de tabelas leves via NQL funcionando
  → Dados de config migrados do PostgreSQL para NietzscheDB
  → PostgreSQL aliviado de ~8 tabelas de configuração não-clínica
```

---

## FASE H — Table Store Mínimo (dados leves de configuração)

> **Decisão**: PostgreSQL permanece para tudo clínico, regulado e relacional.
> A Fase H cobre apenas dados de **configuração leve** que hoje vivem no PostgreSQL
> mas não precisam de ACID completo, JOINs complexos ou LGPD.
> Objetivo: aliviar o PostgreSQL de tabelas simples e centralizar mais dados no NietzscheDB.

---

### Tabelas do EVA-Mind candidatas à migração para NietzscheDB

| Tabela PostgreSQL | Motivo para migrar | Complexidade |
|---|---|---|
| `personas` | Configuração estática, 8 registros, sem FK crítica | Baixa |
| `persona_configs` | Config por paciente, acesso simples por `idoso_id` | Baixa |
| `system_prompts` | Versionamento de prompts, leitura frequente | Baixa |
| `dynamic_tools` | Registry de ferramentas dinâmicas, KV simples | Baixa |
| `lacan_config` | Thresholds e TTLs do motor Lacan, raramente muda | Baixa |
| `lacan_ethical_principles` | Tabela de referência estática | Baixa |
| `lacan_elaboration_markers` | Tabela de referência estática | Baixa |
| `enneagram_types` | 9 registros fixos com keywords/patterns | Baixa |
| `treatment_goals` | Metas de tratamento por paciente, sem JOINs complexos | Média |

**NÃO candidatas** (permanecem no PostgreSQL):
```
idosos, agendamentos, historico_ligacoes — core operacional + scheduling
episodic_memories — FK âncora do Qdrant (point ID = postgres ID)
clinical_assessments — dados clínicos regulados (PHQ-9, GAD-7, C-SSRS)
lgpd_audit_log — obrigação legal, requer integridade garantida
cognitive_load_state — PL/pgSQL functions dependentes
patient_* (12 sistemas de memória) — JOINs complexos + triggers
speaker_profiles / speaker_identifications — FK em episodic_memories
```

---

### H.1 — Table Store com SQLite embutido
**Decisão de backend**: SQLite via crate `rusqlite` (feature `bundled`)
**Arquivo alvo**: novo `crates/nietzsche-graph/src/table_store.rs`

> **Por que SQLite e não RocksDB custom?**
> Implementar CRUD + índices + schema + WAL sobre RocksDB do zero levaria semanas.
> O SQLite já tem tudo isso pronto, battle-tested, e o crate `rusqlite` com feature
> `bundled` compila o SQLite junto no binário — zero dependência de sistema.
> NietzscheDB continua sendo um binário único. RocksDB para grafo+vetor, SQLite para tabelas.

```toml
# crates/nietzsche-graph/Cargo.toml
[dependencies]
rusqlite = { version = "0.31", features = ["bundled"] }
# "bundled" → compila libsqlite3 junto no binário, sem precisar instalar no sistema
```

```rust
// NÃO reescrevemos SQLite. Usamos rusqlite como wrapper seguro da libsqlite3 (C).
// TableStore é apenas uma struct Rust que encapsula a conexão SQLite.

use rusqlite::{Connection, params};

pub struct TableStore {
    conn: Connection,   // sqlite3 file: nietzsche_data_dir/tables.db
}

pub struct TableSchema {
    pub name: String,
    pub fields: Vec<FieldDef>,
    pub primary_key: String,
    pub indexes: Vec<String>,   // campos com índice automático
}

pub struct FieldDef {
    pub name: String,
    pub field_type: FieldType,  // Text | Int | Float | Bool | Json | Timestamp
    pub nullable: bool,
    pub default: Option<serde_json::Value>,
}

// FieldType → tipo SQLite (Text→TEXT, Int→INTEGER, Float→REAL, Bool→INTEGER, Json→TEXT, Timestamp→INTEGER)

impl TableStore {
    pub fn open(data_dir: &Path) -> Result<Self> {
        let conn = Connection::open(data_dir.join("tables.db"))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        Ok(Self { conn })
    }

    pub fn create_table(&self, schema: &TableSchema) -> Result<()>;
    pub fn drop_table(&self, name: &str) -> Result<()>;
    pub fn list_tables(&self) -> Result<Vec<String>>;

    pub fn insert(&self, table: &str, row: &serde_json::Value) -> Result<String>;    // retorna pk
    pub fn get(&self, table: &str, pk: &str) -> Result<Option<serde_json::Value>>;
    pub fn update(&self, table: &str, pk: &str, fields: &serde_json::Value) -> Result<()>;
    pub fn delete(&self, table: &str, pk: &str) -> Result<bool>;
    pub fn scan(&self, table: &str, filter: Option<&TableFilter>) -> Result<Vec<serde_json::Value>>;
    pub fn find_by(&self, table: &str, field: &str, value: &serde_json::Value)
        -> Result<Vec<serde_json::Value>>;
}

pub enum TableFilter {
    FieldEq(String, serde_json::Value),
    FieldGt(String, serde_json::Value),
    FieldLt(String, serde_json::Value),
    And(Vec<TableFilter>),
    Or(Vec<TableFilter>),
}
```

Como o SQLite cuida de tudo automaticamente:
```
→ Schema:    CREATE TABLE personas (id TEXT PRIMARY KEY, name TEXT, active INTEGER DEFAULT 1)
→ Índices:   CREATE INDEX IF NOT EXISTS idx_personas_active ON personas(active)
→ CRUD:      INSERT, SELECT WHERE, UPDATE SET WHERE, DELETE WHERE — SQL interno (não exposto ao usuário)
→ WAL:       PRAGMA journal_mode=WAL → durabilidade sem fsync em cada write
→ ACID:      SQLite garante atomicidade de transações — grátis
→ Binário:   feature "bundled" → libsqlite3 compilada junto com o NietzscheDB
```

O usuário do NietzscheDB nunca vê SQL. O TableStore traduz NQL TABLE → SQL internamente:
```rust
// NQL: MATCH TABLE (r:personas) WHERE r.active = true RETURN r.name ORDER BY r.name ASC
// ↓ executor detecta MatchTable → chama TableStore::scan()
// ↓ TableStore::scan() executa internamente:
//   SELECT * FROM personas WHERE active = 1 ORDER BY name ASC
// ↓ resultado volta como Vec<serde_json::Value> → NQL RETURN clause aplica projeção
```

- [ ] `TableStore::open()` — abre `tables.db` no data_dir, seta WAL mode
- [ ] `TableStore::create_table()` — gera e executa `CREATE TABLE` + `CREATE INDEX` via rusqlite
- [ ] `TableStore::schema_to_sql()` — converte `TableSchema` em DDL SQLite
- [ ] `TableStore::filter_to_sql()` — converte `TableFilter` em cláusula `WHERE` parametrizada
- [ ] `TableStore::row_from_sqlite()` — converte `rusqlite::Row` para `serde_json::Value`
- [ ] `PRAGMA journal_mode=WAL` — write-ahead log do próprio SQLite (não usar WAL do NietzscheDB para tabelas)
- [ ] Thread safety: `Connection` não é `Send` — usar `Mutex<Connection>` ou `r2d2-sqlite` connection pool
- [ ] `NIETZSCHE_TABLE_DB_PATH` env var para override do path do `tables.db`

---

### H.2 — NQL: TABLE Statements (NQL puro — sem SQL)

> **Decisão arquitetural**: NQL é a linguagem única do NietzscheDB.
> Não existe dialeto SQL separado. Tables são um storage backend a mais —
> a linguagem para acessá-las é a mesma NQL que já existe para grafos e vetores.
> Isso elimina fricção cognitiva e mantém o parser/executor unificados.

**Arquivo alvo**: `crates/nietzsche-query/src/ast.rs`, `parser.pest`, `executor.rs`

#### Princípio de design

```
NQL trata TABLE como um "espaço de nós estruturados":
  - MATCH TABLE  →  lê linhas       (equivale a SELECT)
  - INSERT TABLE →  cria linha      (equivale a INSERT INTO)
  - MERGE TABLE  →  upsert linha    (equivale a INSERT ON CONFLICT)
  - SET TABLE    →  atualiza linha  (equivale a UPDATE)
  - DELETE TABLE →  remove linha    (equivale a DELETE FROM)

O resultado de qualquer query TABLE é sempre compatível com
o resultado de MATCH grafo — podem ser combinados no mesmo RETURN.
```

#### Sintaxe NQL para Tables

```nql
-- ── SCHEMA ──────────────────────────────────────────────────────

-- Criar tabela (schema declarativo)
CREATE TABLE personas {
    id:     TEXT     PRIMARY KEY,
    name:   TEXT,
    role:   TEXT,
    prompt: TEXT,
    active: BOOL     DEFAULT true
}

-- Listar tabelas existentes
SHOW TABLES

-- Remover tabela
DROP TABLE personas

-- Criar índice numa coluna
CREATE INDEX ON TABLE personas(active)
CREATE INDEX ON TABLE personas(role)


-- ── ESCRITA ──────────────────────────────────────────────────────

-- Inserir linha
INSERT TABLE personas {
    id:     $id,
    name:   $name,
    role:   $role,
    prompt: $prompt,
    active: true
}

-- Upsert: cria se não existe, atualiza se já existe (pelo PK)
MERGE TABLE personas { id: $id }
ON CREATE SET name = $name, role = $role, prompt = $prompt, active = true
ON MATCH SET  prompt = $prompt

-- Atualizar campos de linhas que casam o filtro
MATCH TABLE (r:personas) WHERE r.id = $id
SET r.prompt = $prompt, r.active = false

-- Deletar linha por PK (ou por qualquer filtro)
MATCH TABLE (r:personas) WHERE r.id = $id
DELETE r


-- ── LEITURA ──────────────────────────────────────────────────────

-- Buscar por PK
MATCH TABLE (r:personas) WHERE r.id = $id
RETURN r

-- Buscar por campo (usa índice se existir)
MATCH TABLE (r:personas) WHERE r.active = true
RETURN r.name, r.role
ORDER BY r.name ASC

-- Agregação (mesma syntax que MATCH grafo)
MATCH TABLE (r:personas)
RETURN COUNT(*) AS total, r.role
GROUP BY r.role

-- Com LIMIT e SKIP
MATCH TABLE (r:personas) WHERE r.active = true
RETURN r LIMIT 10 SKIP 20


-- ── HYBRID JOIN: TABLE + NODE + VECTOR ───────────────────────────

-- Une uma row de tabela com nós de grafo e vetores
MATCH TABLE (p:personas) WHERE p.active = true
MATCH (e:Event) IN COLLECTION 'memories'
WHERE e.metadata.persona_id = p.id
  AND e.importance > 0.7
RETURN p.name AS persona, e.content, e.emotion
ORDER BY e.created_at DESC
LIMIT 20

-- Pipelining com WITH (lê tabela → usa resultado no grafo)
MATCH TABLE (pc:persona_configs) WHERE pc.idoso_id = $pid
WITH pc
MATCH TABLE (p:personas) WHERE p.id = pc.persona_id
WITH p
MATCH KNN(e, $embedding, k=10) IN COLLECTION 'memories'
WHERE e.metadata.idoso_id = $pid
RETURN p.name AS persona, e.content, COSINE_DIST(e.embedding, $embedding) AS score
ORDER BY score DESC
```

#### AST a adicionar (extensão do enum existente)

```rust
pub enum Statement {
    Match(MatchQuery),            // existente — grafo
    Diffuse(DiffuseQuery),        // existente
    Explain(Box<Statement>),      // existente
    Reconstruct(ReconstructQuery),// existente
    Merge(MergeQuery),            // NQL-1 — grafo
    Create(CreateQuery),          // NQL-4 — grafo
    Set(SetQuery),                // NQL-3 — grafo
    Delete(DeleteQuery),          // NQL-5 — grafo

    // NOVO — TABLE (mesmo executor, backend diferente)
    CreateTable(CreateTableQuery),   // CREATE TABLE nome { schema }
    DropTable(DropTableQuery),       // DROP TABLE nome
    ShowTables,                      // SHOW TABLES
    MatchTable(MatchTableQuery),     // MATCH TABLE (r:nome) WHERE ... RETURN/SET/DELETE
    InsertTable(InsertTableQuery),   // INSERT TABLE nome { campos }
    MergeTable(MergeTableQuery),     // MERGE TABLE ... ON CREATE/MATCH SET
}

// MatchTable é o nó central — leitura, update e delete de tabelas
// todos partem de MATCH TABLE (r:nome) e depois aplicam RETURN / SET / DELETE
pub struct MatchTableQuery {
    pub alias: String,                        // "r"
    pub table: String,                        // "personas"
    pub where_clause: Option<Condition>,      // reusa Condition existente
    pub action: MatchTableAction,
}

pub enum MatchTableAction {
    Return(ReturnClause),                     // RETURN r.name, ...
    Set(Vec<Assignment>),                     // SET r.field = expr
    Delete,                                   // DELETE r
}
```

- [ ] Grammar `pest`: `create_table_stmt`, `drop_table_stmt`, `show_tables_stmt`,
      `match_table_stmt`, `insert_table_stmt`, `merge_table_stmt`
- [ ] `MATCH TABLE` reutiliza `Condition` e `ReturnClause` existentes (zero duplicação)
- [ ] `MATCH TABLE ... SET` → update; `MATCH TABLE ... DELETE` → delete
- [ ] `MATCH TABLE` + `MATCH (n:Node)` no mesmo statement → Hybrid JOIN
- [ ] Executor detecta `MatchTable` → roteia para `TableStore`; `Match` → `GraphStorage`
- [ ] `WITH` propaga resultados de TABLE para MATCH grafo e vice-versa
- [ ] Testes: CRUD completo, Hybrid JOIN, índice vs full scan, MERGE idempotente

---

### H.3 — gRPC: Table RPCs
**Arquivo alvo**: `proto/nietzsche.proto`, `crates/nietzsche-api/src/handlers.rs`

```protobuf
// NOVOS RPCs
rpc CreateTable(CreateTableRequest) returns (CreateTableResponse);
rpc DropTable(DropTableRequest) returns (DropTableResponse);
rpc TableInsert(TableInsertRequest) returns (TableInsertResponse);
rpc TableGet(TableGetRequest) returns (TableGetResponse);
rpc TableUpdate(TableUpdateRequest) returns (TableUpdateResponse);
rpc TableDelete(TableDeleteRequest) returns (TableDeleteResponse);
rpc TableScan(TableScanRequest) returns (TableScanResponse);
rpc ListTables(ListTablesRequest) returns (ListTablesResponse);
```

- [ ] Tipos proto para `TableSchema`, `TableFilter`, `TableRow` (google.protobuf.Struct)
- [ ] Handlers implementados e testados
- [ ] SDK Go inclui métodos de Table Store

---

### H.4 — Script de migração EVA-Mind: PostgreSQL → NietzscheDB Table Store

Para cada tabela candidata, criar um script Go de migração única:

```go
// cmd/migrate_to_nietzsche/main.go (no EVA-Mind)

func migratePersonas(pg *sql.DB, ndb *nietzsche.Client) error {
    rows, _ := pg.Query(`SELECT id, name, role, prompt_template, ativo FROM personas`)
    for rows.Next() {
        var p Persona
        rows.Scan(&p.ID, &p.Name, &p.Role, &p.Prompt, &p.Active)
        ndb.TableInsert(ctx, "personas", map[string]any{
            "id": p.ID, "name": p.Name, "role": p.Role,
            "prompt": p.Prompt, "active": p.Active,
        })
    }
    // Após validação: DROP TABLE personas no PostgreSQL
}
```

- [ ] `migratePersonas()`
- [ ] `migrateSystemPrompts()`
- [ ] `migrateDynamicTools()`
- [ ] `migrateLacanConfig()`
- [ ] `migrateEnneagramTypes()`
- [ ] Validação: count antes = count depois; rollback automático se divergir

---

### H.5 — Hybrid JOIN entre Table Store e Graph
**Valor diferencial**: nenhum outro banco faz isso nativamente.

```nql
-- NQL puro: tabela + grafo + vetor numa única query
MATCH TABLE (pc:persona_configs) WHERE pc.idoso_id = $pid
MATCH TABLE (p:personas)         WHERE p.id = pc.persona_id AND p.active = true
MATCH (e:Event) IN COLLECTION 'memories'
WHERE e.metadata.session_persona = pc.persona_id
  AND e.importance > 0.7
RETURN p.name AS persona, e.content, e.emotion
ORDER BY e.created_at DESC
LIMIT 20

-- Com busca semântica (KNN) integrada ao JOIN
MATCH TABLE (p:personas) WHERE p.active = true
MATCH KNN(e, $query_embedding, k=15) IN COLLECTION 'memories'
WHERE e.metadata.idoso_id = $pid
  AND e.metadata.persona_id = p.id
RETURN p.name, e.content, COSINE_DIST(e.embedding, $query_embedding) AS score
ORDER BY score DESC
```

Isso é o que torna o NietzscheDB único: **dados estruturados + grafo + vetores numa query NQL só**.
Nenhum outro banco faz isso. ArangoDB e SurrealDB têm multi-model mas não têm HNSW hiperbólico
nem o ciclo de sono Riemanniano integrado ao planner de queries.

- [ ] Planner de hybrid join: decide ordem de execução (table-first ou graph-first)
- [ ] Resultado do JOIN é um stream de `(TableRow, GraphNode)` pares
- [ ] Filtros de cada lado executados no storage correto (pushdown)
- [ ] Testes de performance: hybrid join vs aplicação fazendo 2 queries separadas

---

### Checklist Fase H

- [ ] **H.1** `TableStore` com SQLite via `rusqlite` (feature `bundled`) — sem RocksDB custom
- [ ] **H.1** `Mutex<Connection>` ou pool `r2d2-sqlite` para thread safety
- [ ] **H.1** `PRAGMA journal_mode=WAL; synchronous=NORMAL` no open
- [ ] **H.2** NQL: `CREATE TABLE`, `INSERT TABLE`, `MERGE TABLE`, `MATCH TABLE`, `DROP TABLE`, `SHOW TABLES`
- [ ] **H.3** gRPC: 8 novos RPCs de Table Store
- [ ] **H.4** Scripts de migração para 8 tabelas candidatas do EVA-Mind
- [ ] **H.5** Hybrid JOIN TABLE + NODE + VECTOR na NQL

### Métricas de Sucesso da Fase H
```
✓ Table Store pronto quando:
  → CRUD em tabelas leves < 2ms p99
  → Scan com índice < 5ms para 10k registros
  → 8 tabelas de config migradas do PostgreSQL com zero perda de dados
  → Hybrid JOIN funcionando em benchmark sintético
  → PostgreSQL aliviado de ~15% das queries de configuração do EVA-Mind
```

---

## FASE I — Media Store (OpenDAL)
**Prioridade: MÉDIA** | Depende de: Fase C.2 (ListStore) + Fase A (Cosine/VectorStore)

> **Contexto**: EVA-Mind já usa áudio (PCM streaming via Redis List → Fase C.2 substitui).
> Fase I adiciona armazenamento **permanente** de áudio, imagem e vídeo, com metadados
> no grafo e embeddings no VectorStore. OpenDAL abstrai o backend de storage.

### I.1 — OpenDAL como camada de storage de media
**Arquivo alvo**: novo `crates/nietzsche-graph/src/media_store.rs`

```toml
# crates/nietzsche-graph/Cargo.toml
[dependencies]
opendal = { version = "0.48", features = ["services-s3", "services-gcs", "services-fs"] }
# services-fs  → filesystem local (dev/default)
# services-s3  → AWS S3 (produção)
# services-gcs → Google Cloud Storage (EVA-Mind usa GCP)
```

```rust
use opendal::{Operator, Scheme};

pub struct MediaStore {
    op: Operator,   // backend configurável: fs | s3 | gcs | azblob
}

impl MediaStore {
    /// Configura via env vars:
    /// NIETZSCHE_MEDIA_BACKEND=fs|s3|gcs  (default: fs)
    /// NIETZSCHE_MEDIA_ROOT=/data/media   (para fs)
    /// NIETZSCHE_MEDIA_BUCKET=meu-bucket  (para s3/gcs)
    pub fn from_env() -> Result<Self>;

    /// Salva arquivo de media — retorna URL canônica
    pub async fn put(&self, path: &str, data: Bytes) -> Result<String>;
    // path = "audio/{session_id}.pcm" → URL = "s3://bucket/audio/{id}.pcm" ou "file:///data/media/audio/{id}.pcm"

    pub async fn get(&self, path: &str) -> Result<Bytes>;
    pub async fn delete(&self, path: &str) -> Result<()>;
    pub async fn exists(&self, path: &str) -> Result<bool>;
    pub async fn list(&self, prefix: &str) -> Result<Vec<String>>;
}
```

- [ ] `MediaStore::from_env()` — detecta `NIETZSCHE_MEDIA_BACKEND` e inicializa Operator
- [ ] Backend `fs` como default para dev (sem config de cloud necessária)
- [ ] Backend `gcs` configurado via `GOOGLE_APPLICATION_CREDENTIALS` (EVA-Mind já usa GCP)
- [ ] Backend `s3` configurado via `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`
- [ ] Streaming upload para arquivos grandes (PCM longo): `op.writer(path).await?`

### I.2 — Fluxo completo: PCM temporário → Media permanente

```
Fase C.2 (ListStore)                    Fase I (MediaStore)
──────────────────────                  ──────────────────────────────────────
1. Audio chunk chega                    4. Sessão termina
   → ListRPush("audio:sid", chunk, 3600)   → consolidar chunks
                                           → MediaStore::put("audio/{sid}.pcm", bytes)
2. Buffer acumula (TTL 1h)              5. URL salva no grafo:
   → lrange("audio:sid", 0, -1)            (session)-[:HAS_AUDIO {
3. EVA processa PCM                          url: "gcs://bucket/audio/{sid}.pcm",
   → Speaker embedding via ECAPA-TDNN        duration_secs: 183,
   → Salva em VectorStore                    sample_rate: 16000
                                           }]->(audio_node)
                                        6. ListDel("audio:sid") — limpa buffer
```

- [ ] RPC `ConsolidateAudio(session_id)` → lê ListStore → grava MediaStore → cria nó+aresta no grafo
- [ ] NQL: `MATCH (s:EvaSession {id: $sid})-[:HAS_AUDIO]->(a) RETURN a.url, a.duration_secs`

### I.3 — Casos de uso EVA-Mind

| Media | Fluxo | Embedding | Coleção VectorStore |
|---|---|---|---|
| Áudio PCM da sessão | ListStore (temp) → MediaStore (perm) | ECAPA-TDNN dim=192 | `speaker_embeddings` |
| Foto do idoso | Upload direto → MediaStore | CLIP dim=768 | `patient_faces` (nova) |
| Vídeo de sessão | Chunks → MediaStore | VideoMAE dim=768 | `session_videos` (nova) |

### I.4 — gRPC RPCs de Media

```protobuf
rpc MediaPut(MediaPutRequest) returns (MediaPutResponse);    // upload + retorna URL
rpc MediaGet(MediaGetRequest) returns (stream MediaChunk);   // streaming download
rpc MediaDelete(MediaDeleteRequest) returns (MediaDeleteResponse);
rpc MediaList(MediaListRequest) returns (MediaListResponse);
rpc ConsolidateAudio(ConsolidateAudioRequest) returns (ConsolidateAudioResponse);
// ConsolidateAudio: lê ListStore → grava MediaStore → cria nó no grafo automaticamente
```

- [ ] Streaming gRPC para upload/download de arquivos grandes (server-side streaming)
- [ ] `MediaPutRequest`: path, content_type, metadata (session_id, patient_id, etc.)
- [ ] `MediaPutResponse`: url canônica + node_id criado no grafo

### Checklist Fase I

- [ ] **I.1** `MediaStore` com OpenDAL — backend `fs` (dev) + `gcs` (prod)
- [ ] **I.1** `NIETZSCHE_MEDIA_BACKEND` env var (`fs` | `s3` | `gcs`)
- [ ] **I.2** Fluxo PCM: `ListStore → consolidate → MediaStore → Graph node`
- [ ] **I.2** RPC `ConsolidateAudio` — fecha ciclo de áudio da sessão
- [ ] **I.3** Coleção `speaker_embeddings` (dim=192 Cosine) criada na Fase B
- [ ] **I.4** 5 gRPC RPCs de Media (MediaPut, MediaGet, MediaDelete, MediaList, ConsolidateAudio)
- [ ] **I.X** NQL: `MATCH (s:Session)-[:HAS_AUDIO]->(a) RETURN a.url`

### Métricas de Sucesso da Fase I
```
✓ Media Store pronto quando:
  → Upload de PCM 10MB < 500ms (filesystem local)
  → ConsolidateAudio: ListStore → MediaStore → Graph node em < 2s
  → EVA-Mind sem Redis para áudio E com audio permanente em GCS
  → speaker_embeddings no VectorStore (dim=192) substituindo Qdrant
```

---

### Visão Final da Arquitetura após todas as Fases

```
┌──────────────────────────────────────────────────────────────────┐
│                        NietzscheDB                                │
│                                                                    │
│  ┌──────────┐ ┌──────────────────┐ ┌──────────┐ ┌──────┐ ┌──────┐ │
│  │  Graph   │ │   Vector Store   │ │  Table   │ │Cache │ │Media │ │
│  │  Store   │ │  HNSW multi-col  │ │  Store   │ │Store │ │Store │ │
│  │ (RocksDB)│ │  Cosine+Poincaré │ │ (SQLite) │ │+List │ │(Open │ │
│  │ Node/Edge│ │  dim 192/768/3072│ │ personas │ │Store │ │ DAL) │ │
│  │ WAL+Adj  │ │  17+ collections │ │ prompts  │ │ TTL  │ │fs/gcs│ │
│  └────┬─────┘ └────────┬─────────┘ └────┬─────┘ └──┬───┘ └──┬───┘ │
│       │                │                │           │         │     │
│  absorve Neo4j #1+#2   │   absorve Qdrant  absorve Redis  audio/img │
│  ┌─────▼────────────────▼────────────────▼───────────▼─────────▼──┐ │
│  │                      NQL Engine                                  │ │
│  │  MATCH · MERGE · CREATE · SET · DELETE · DETACH DELETE          │ │
│  │  MATCH TABLE · INSERT TABLE · MERGE TABLE  (Table Store)        │ │
│  │  MATCH KNN · IN COLLECTION · COSINE_DIST   (Vector Store)       │ │
│  │  CACHE GET/SET · LIST RPUSH/LRANGE          (Cache Store)       │ │
│  │  Hybrid JOIN (TABLE + NODE + VECTOR em uma query só)            │ │
│  │  DIFFUSE · RECONSTRUCT · EXPLAIN                                 │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                gRPC API (tonic) + OpenDAL (fs | gcs | s3)            │
└────────────────────────────────────────────────────────────────────┘
                          ↕ coexiste com
┌────────────────────────────────────────────────────────────────────┐
│                         PostgreSQL                                   │
│   idosos · agendamentos · episodic_memories · lgpd_audit_log        │
│   clinical_assessments · patient_* (12 sistemas de memória)         │
│   speaker_profiles · cognitive_load · ethical_audit                 │
│   medicamentos · escalation_logs · family_changes                   │
└────────────────────────────────────────────────────────────────────┘

Stack EVA-Mind:   ANTES  5 bancos  →  DEPOIS  2 bancos
                  Neo4j #1             NietzscheDB  (grafo + vetor + cache + table + media)
                  Neo4j #2      ──►
                  Redis
                  Qdrant
                  PostgreSQL           PostgreSQL   (clínico + LGPD)
```
