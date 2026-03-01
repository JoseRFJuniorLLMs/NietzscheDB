# Auditoria de Performance — NietzscheDB vs NietzscheDB
## Análise Completa Antes de Qualquer Implementação

> **Metodologia**: código NietzscheDB clonado (`github.com/NietzscheDB/NietzscheDB`, commit `9f433b1`, Fev 2026)
> e analisado linha a linha. NietzscheDB inspecionado em profundidade.
> **Nenhum código foi alterado neste documento.**

---

## PARTE 1 — ARQUITETURA ATUAL DO NIETZSCHEDB

### 1.1 Stack completo

```
nietzsche-server
  └─ CollectionManager → Arc<RwLock<NietzscheDB<AnyVectorStore>>>
       └─ NietzscheDB<V>
            ├─ GraphStorage (RocksDB, 7 CFs)
            ├─ GraphWal     (append-only file)
            ├─ AdjacencyIndex (DashMap em RAM)
            ├─ V: VectorStore
            │    └─ EmbeddedVectorStore
            │         ├─ Box<dyn DynHnsw>  (vtable dispatch)
            │         │    └─ HnswIndex<N, CosineMetric>  ← nietzsche-hnsw
            │         └─ DashMap<Uuid, u32>  (UUID→HNSW id)
            └─ hot_tier: Arc<DashMap<Uuid, Node>>
```

### 1.2 Tamanho real de um Node no RocksDB

Para `dim = 3072` (default):

```
Node struct serializado (bincode) no CF_NODES:
  id:                  16 bytes   (UUID)
  embedding.coords:    24.576 bytes  ← Vec<f64> × 3072  ← DUPLICADO no HNSW mmap!
  embedding.dim:        8 bytes
  depth:                4 bytes   (f32)
  content:           ~200 bytes   (serde_json::Value típico)
  node_type:            1 byte
  energy:               4 bytes   (f32)
  lsystem_generation:   4 bytes   (u32)
  hausdorff_local:      4 bytes   (f32)
  created_at:           8 bytes   (i64)
  metadata:           ~100 bytes  (HashMap típico)
  ─────────────────────────────
  TOTAL:             ~25 KB por nó  ← 98% é o embedding
```

**Para 1 M de nós:** ~25 GB no RocksDB, sendo ~24 GB de embeddings já presentes no HNSW mmap store (`data_dir/hnsw/`).

---

## PARTE 2 — ANÁLISE DE CADA OTIMIZAÇÃO

---

### ITEM A — Separar embedding de CF_NODES
**Criticidade: CRÍTICO | Ganho: 10–20× get_node() | Risco hiperbólico: NENHUM**

#### O problema

`Node` completo, incluindo `Vec<f64>` de 24 KB, é gravado no CF_NODES. O mesmo embedding já existe no `nietzsche-vecstore` mmap. É **duplicação**.

Cada `get_node()` (BFS, traversal, NQL filter, sleep, zaratustra) desserializa 25 KB desnecessariamente quando só precisa de `energy`, `hausdorff_local`, `depth`, `content` — ~400 bytes.

#### Solução — NodeMeta separado do Embedding

```rust
// CF_NODES armazenaria apenas:
struct NodeMeta {
    id:                  Uuid,        // 16 bytes
    depth:               f32,         //  4 bytes
    energy:              f32,         //  4 bytes
    node_type:           NodeType,    //  1 byte
    hausdorff_local:     f32,         //  4 bytes
    lsystem_generation:  u32,         //  4 bytes
    created_at:          i64,         //  8 bytes
    content:             Value,       // ~200 bytes
    metadata:            HashMap,     // ~100 bytes
    // SEM embedding
}
// TOTAL: ~340 bytes por nó = 73× menor
```

O embedding fica **somente** no HNSW mmap store, lido on-demand (sleep cycle, distance computation).

#### Impacto por operação

| Operação | Antes | Depois |
|----------|-------|--------|
| BFS/traversal get_node | 25 KB desserializado | 340 bytes |
| NQL WHERE n.energy > X | 25 KB por nó scaneado | 340 bytes |
| Zaratustra energy scan | 25 KB × N | 340 bytes × N |
| hot_tier RAM (1M elite) | 25 GB | 340 MB |
| Sleep cycle (precisa embedding) | igual | igual (lê do HNSW mmap) |

#### Risco hiperbólico

NENHUM. O embedding continua existindo, só muda onde vive. A geometria Poincaré não é afetada.

#### Dependências antes de implementar

- Migração de dados (CF_NODES existentes têm Node completo)
- API de leitura de embedding do nietzsche-vecstore (actualmente não exposta)
- Possível refactoring de todos os callers de `get_node()` que acessam `node.embedding`

---

### ITEM B — True Poincaré HNSW Metric (BUG-EVS-001)
**Criticidade: CRÍTICO (correctness) | Ganho: Recall correto | Risco: NENHUM (é bug fix)**

#### O bug

```rust
// embedded_vector_store.rs:307-312
VectorMetric::Euclidean | VectorMetric::PoincareBall => {
    make_raw_hnsw(dim, &storage_dir)?
    // HnswRawWrapper<N> usa HnswIndex<N, CosineMetric> internamente!
}
```

O grafo HNSW é construído com `CosineMetric` — vizinhos são determinados por similaridade cosseno. Mas a métrica real do NietzscheDB é:

```
d(u,v) = acosh(1 + 2‖u-v‖² / ((1-‖u‖²)(1-‖v‖²)))
```

**Consequência:** Um nó próximo na geometria hiperbólica pode não ser vizinho no grafo HNSW. O recall para KNN hiperbólico é sistematicamente inferior ao possível. Nós próximos à fronteira (episódicos) são os mais afetados — onde `‖x‖ → 1`, distâncias hiperbólicas divergem de distâncias coseno.

#### Solução

`HnswPoincareWrapper<N>` com `PoincareMetric` como parâmetro genérico do `HnswIndex`. Requer suporte a métricas customizadas no `nietzsche-hnsw`.

#### Risco hiperbólico

NENHUM — é um bug fix. O banco só pode melhorar.

---

### ITEM C — f64 → f32 para armazenamento de embeddings
**Criticidade: ALTO | Ganho: 2× memória, 2× SIMD | Risco hiperbólico: BAIXO (mitigável)**

#### Situação atual

```rust
pub struct PoincareVector {
    pub coords: Vec<f64>,  // 8 bytes por coordenada
    pub dim: usize,
}
```

Para `dim=3072`: 24.576 bytes por embedding em f64.

#### Análise para Poincaré ball

- Invariante ‖x‖ < 1.0 garante coords ∈ (-1.0, +1.0)
- Modelos ML produzem embeddings em f32 (7 dígitos decimais de precisão)
- f32 precisão: ±5.96e-8 por coordenada
- A distância Poincaré usa `acosh` que amplifica erros perto da fronteira:

```
(1 - ‖x‖²) para ‖x‖ = 0.99:   = 0.0199
Erro f32 em uma coord = 0.008:
(1 - (0.99+0.008)²) = 0.0040   → diferença de 5× no denominador
```

#### Mitigação

Armazenar em f32, computar em f64:

```rust
pub struct PoincareVector {
    pub coords: Vec<f32>,   // storage: 4 bytes por coordenada
    pub dim: usize,
}

// No kernel de distância: promove f32→f64 antes do acosh
fn poincare_sums_f32(u: &[f32], v: &[f32]) -> (f64, f64, f64) {
    let mut diff_sq = 0.0f64;
    let mut norm_u  = 0.0f64;
    let mut norm_v  = 0.0f64;
    for i in 0..u.len() {
        let a = u[i] as f64;  // promoção aqui
        let b = v[i] as f64;
        let d = a - b;
        diff_sq += d * d;
        norm_u  += a * a;
        norm_v  += b * b;
    }
    (diff_sq, norm_u, norm_v)
}
```

AVX2: `_mm256_cvtps_pd` converte 4 f32 → 4 f64 por instrução. Overhead mínimo.

#### Impacto

| Métrica | Antes (f64) | Depois (f32 storage) |
|---------|-------------|---------------------|
| Memória embedding | 24.576 bytes (3072-dim) | 12.288 bytes |
| SIMD throughput | 4 f64/ciclo (AVX2) | 8 f32/ciclo durante accumulate |
| Precisão distância | f64 nativa | f64 (promovido na hora) |
| Risco boundary nodes | — | baixo com mitigação |

#### Impacto em outros módulos

- `sleep/riemannian.rs` — Adam optimizer: atualiza coords com gradientes; precisa ser revisado para f32
- `model.rs`, `storage.rs` — mudança de tipo afeta bincode serialization (migração de dados)
- `nietzsche-lsystem` — gera novas coordenadas; precisaria ajustar
- **Breaking change**: todos os callers de `embedding.coords` (Vec<f64> → Vec<f32>)

---

### ITEM D — Scalar Quantization Int8
**Criticidade: MÉDIO | Ganho: 8× memória (f64→u8), KNN mais rápido | Risco hiperbólico: MÉDIO**

#### Implementação NietzscheDB (código real estudado)

```rust
// lib/quantization/src/encoded_vectors_u8.rs

// Calibração: alpha e offset definem o mapeamento
fn alpha_offset_from_min_max(min: f32, max: f32) -> (f32, f32) {
    let alpha = (max - min) / 127.0;  // step size
    let offset = min;                  // zero point
    (alpha, offset)
}

// Encoding por elemento
fn encode_value(&self, value: f32) -> u8 {
    let i = (value - self.offset) / self.alpha;
    i.clamp(0.0, 127.0).round() as u8
}

// Layout por vetor no disco:
// [f32_correction(4 bytes)][u8 * dim]
// A correção por vetor é pré-computada e armazenada inline

// Scoring assimétrico:
// score = multiplier * dot_int8(query_u8, vector_u8)  [SIMD AVX2]
//       + query.offset_f32                             [pre-computado na query]
//       + vector.correction_f32                        [lido inline do vetor]
```

#### Adaptação para Poincaré

A invariante ‖x‖ < 1.0 significa coords ∈ (-1.0, +1.0). Range fixo e conhecido:

```rust
// Para Poincaré ball: calibração trivial, sem necessidade de P2 quantile estimator
const POINCARE_ALPHA: f32 = 2.0 / 127.0;  // range 2.0 / 127 levels
const POINCARE_OFFSET: f32 = -1.0;

fn encode_poincare(x: f64) -> u8 {
    let q = (x as f32 - POINCARE_OFFSET) / POINCARE_ALPHA;
    q.clamp(0.0, 127.0).round() as u8
}
```

#### Risco específico — nós de fronteira

Erro de quantização: `±(POINCARE_ALPHA / 2) = ±0.0079` por coordenada.

Para ‖x‖ > 0.9 (nós episódicos), o denominador `(1-‖x‖²)` é pequeno. Um erro de 0.008 em uma coordenada pode mudar a distância hiperbólica de 2× a 5×.

**Mitigação (idêntica ao NietzscheDB):**
- Busca `k × oversampling_factor` candidatos no índice quantizado
- Rescore dos candidatos com coords originais (f32 ou f64)
- `oversampling_factor = 3` para nós uniformes, `= 8` se maioria na fronteira

#### SIMD dispatch (NietzscheDB pattern)

```rust
fn score(&self, q: &[u8], v: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { impl_score_dot_avx(q.as_ptr(), v.as_ptr(), q.len() as u32) };
    }
    // SSE, NEON, scalar fallbacks...
}
// AVX2: 32 u8 multiplications per instruction vs 8 f32 per instruction = 4× throughput
```

#### Impacto total

| Métrica | f64 original | Int8 quantizado |
|---------|-------------|-----------------|
| Memória por vetor (3072-dim) | 24.576 bytes | 3.072 bytes + 4 bytes correction = 3.076 bytes |
| Redução de memória | — | **8× (f64→u8)** |
| KNN throughput | baseline | ~4× (SIMD u8 vs f64) |
| Recall (com oversampling×3) | 100% | ~97-99% |
| Recall sem oversampling | 100% | ~90-95% (boundary degradada) |

---

### ITEM E — Filtered KNN via energy_idx (pre-filter antes do HNSW)
**Criticidade: ALTO | Ganho: 5–50× em queries filtradas | Risco hiperbólico: NENHUM**

#### O problema atual

```
Query: MATCH (n) WHERE n.energy > 0.5 KNN (n, $query, k=10)

Fluxo atual:
1. HNSW.search(query, k=10) → 10 nós sem filtro de energy
2. WHERE n.energy > 0.5      → talvez só 2 passam
3. Resultado: apenas 2 resultados (precisa de k=500 para obter 10 com energy>0.5)
```

#### O que o NietzscheDB faz (código real)

```rust
// hnsw.rs — dispatcher de busca filtrada
let query_cardinality = payload_index.estimate_cardinality(query_filter);

if query_cardinality.max < self.config.full_scan_threshold {
    // Poucos candidatos: linear scan nos filtrados
    // Custo: O(cardinalidade × distância) — ignora HNSW
    return self.search_vectors_plain(vectors, query_filter, top, params);
}
if query_cardinality.min > self.config.full_scan_threshold {
    // Muitos candidatos: HNSW com filter check em cada vizinho
    return self.search_vectors_with_graph(vectors, filter, top, params);
}
// Cardinalidade incerta: sample 1000 pontos, mede fração que passa
```

#### Adaptação para NietzscheDB

Já temos `energy_idx` em RocksDB (range scan O(log N) para `WHERE energy > X`). Precisamos:

1. Expor `knn_filtered(query, k, allowed_ids: &[Uuid])` no `VectorStore` trait
2. Routing logic no NQL executor:

```rust
// executor.rs — ao processar KNN com filtro de energy
let energy_threshold = 0.5;
let candidate_ids = storage.energy_range(energy_threshold..)?; // O(log N)

let k = 10;
let result = if candidate_ids.len() < PLAIN_SCAN_THRESHOLD {
    // Poucos candidatos: linear scan computando distância diretamente
    vector_store.knn_from_subset(&query, k, &candidate_ids)?
} else {
    // Muitos candidatos: HNSW com máscara
    vector_store.knn_filtered(&query, k, &candidate_ids)?
};
```

#### Impacto por workload

| Cenário | Antes | Depois |
|---------|-------|--------|
| 50% nós com energy > 0.5, k=10 | k=500 no HNSW, 490 descartados | k=10 em 50% dos nós |
| 10% nós com energy > 0.5, k=10 | k=5000 no HNSW | linear scan nos 10% → 50× mais rápido |
| Zaratustra (scan por energy) | full scan CF_NODES | energy_idx range → O(log N) |

---

### ITEM F — Binary Quantization (XOR + POPCOUNT)
**Criticidade: BAIXO para Poincaré | Ganho: 30–40× teórico | Risco hiperbólico: ALTO**

#### Implementação NietzscheDB (código real)

```rust
// encoded_vectors_binary.rs — encoding
fn encode_one_bit_vector(vector: &[f32], encoded: &mut [u8]) {
    for (i, &v) in vector.iter().enumerate() {
        if v > 0.0 {
            encoded[i / 8] |= 1 << (i % 8);
        }
    }
    // 768-dim → 96 bytes (vs 3072 bytes f32)
}

// Scoring via XOR + POPCOUNT
fn xor_popcnt(v1: &[u8], v2: &[u8]) -> usize {
    v1.iter().zip(v2).map(|(&a, &b)| (a ^ b).count_ones() as usize).sum()
}
// score = dim - 2 * xor_popcnt  ≈ cosine similarity
```

#### Por que NÃO funciona bem para Poincaré ball

A aproximação `cosine ≈ 1 - 2*Hamming/dim` funciona porque cosseno é baseado em ângulo entre vetores — e `sign(x)` captura o semiespaço.

A distância Poincaré **não é baseada em ângulo**:
```
d(u,v) = acosh(1 + 2‖u-v‖² / ((1-‖u‖²)(1-‖v‖²)))
```

O denominador `(1-‖u‖²)(1-‖v‖²)` depende criticamente da magnitude, que `sign(x)` ignora completamente.

**Caso concreto:**
```
u = [0.01, 0.01, ..., 0.01]  → nó semântico perto do centro, ‖u‖ ≈ 0.017
v = [0.02, 0.01, ..., 0.01]  → nó semântico próximo
w = [-0.01, 0.01, ..., 0.01] → nó semântico, sinal diferente

Hamming(sign(u), sign(v)) = 0  → BQ diz: distância mínima
Hamming(sign(u), sign(w)) = 1  → BQ diz: diferente

d_hyperbolic(u, v) ≈ 0.010   → muito próximos (ambos perto do centro)
d_hyperbolic(u, w) ≈ 0.010   → igualmente próximos

BQ está ERRADO: v não é mais próximo de u que w em Poincaré ball.
```

Pior: a maioria dos nós semânticos (Semantic, Concept) vive perto do centro com coords ∈ (-0.1, 0.1). O sinal dessas coordenadas é quase aleatório em relação à distância hiperbólica real.

#### Quando pode ser usado

APENAS como pre-filter grosseiro para dimensões altas (dim ≥ 768) com oversampling agressivo (×20) e rescore obrigatório com distância Poincaré real.

**Verdict: NÃO implementar como métrica primária. Apenas como pre-filter opcional.**

---

### ITEM G — HNSW Link Compression (Delta Encoding + Bit-Packing)
**Criticidade: BAIXO agora | Ganho: 2–4× RAM no grafo HNSW | Risco hiperbólico: NENHUM**

#### O que o NietzscheDB faz (código real estudado)

```rust
// graph_links/view.rs — três formatos
pub enum GraphLinksFormat {
    Plain,                  // u32 IDs diretos, sem compressão
    Compressed,             // bit-packing com ceil(log2(N)) bits/ID + varint offsets
    CompressedWithVectors,  // Compressed + vetores quantizados inline por vizinho
}

// Compressed: para 1M pontos = ceil(log2(1M)) = 20 bits/ID vs 32 bits plain
// Redução: 20/32 = 37.5% menor nos IDs

// Acesso zero-copy via slice de bytes — sem desserialização
let idx = if level == 0 { point_id as usize } else {
    level_offsets[level] + reindex[point_id]  // O(1) lookup
};
let bit_range = offsets.get(idx)..offsets.get(idx+1);
iterate_packed_links(&neighbors[bit_range], bits_per_link, M)
```

#### Por que é mais complexo no NietzscheDB

O NietzscheDB usa IDs internos `u32` sequenciais. `ceil(log2(1M)) = 20 bits`. Compressão real.

O NietzscheDB usa `Uuid::new_v4()` — 128-bit aleatórios. O `nietzsche-hnsw` usa `u32` internamente mas mapeia para UUID via HashMap de metadata.

Para bit-packing funcionar, precisamos saber a quantidade total de pontos no índice. Com u32 internos já temos isso. O problema é que o `nietzsche-hnsw` é uma caixa preta — não podemos modificar seus internals sem forkar.

**O formato `CompressedWithVectors` é o mais valioso:**
```
Layout por nó (nível 0):
[vetor_quantizado_base | count | links_bit_packed | padding | vetores_quantizados_vizinhos]
                                                             ↑
                              Score calculado sem cache miss extra!
```

#### Dependências

- Forkar ou substituir `nietzsche-hnsw`
- Ou implementar nosso próprio HNSW (após resolver item B)
- UUID → u32 remapping para bit-packing eficiente

**Verdict: FASE 4. Primeiro resolver A, B, C, D, E.**

---

### ITEM H — Gridstore (substituir RocksDB para embeddings)
**Criticidade: MÉDIO (futuro) | Ganho: latência previsível, sem compaction pauses | Risco: NENHUM**

#### O que é o Gridstore do NietzscheDB

```rust
// Estrutura: páginas de 32MB em mmap
pub struct Gridstore<V> {
    pages: Vec<Page>,               // arquivos mmap de 32MB
    tracker: Tracker,               // point_id → (page_id, block_offset, length)
    bitmask: Bitmask,               // 1 bit por bloco de 128 bytes (livre/usado)
}

// Read: O(1) — tracker lookup + mmap deref
// Write: O(1) — bitmask find + mmap write (sem WAL para este storage)
// Crash safety: write data → update bitmask → update tracker (flush atomicamente)
```

**Por que é relevante para NietzscheDB:**
- Após item A (separar embedding de CF_NODES), os embeddings precisam de um storage eficiente
- O `nietzsche-vecstore` já é mmap-based, mas é gerenciado pelo HNSW
- Para embeddings de nós que saíram do HNSW (pruned, archived), precisamos de storage separado
- Gridstore = zero WAL overhead, zero compaction, latência previsível

**Verdict: FASE 4 após item A.**

---

## PARTE 3 — ACHADOS EXCLUSIVOS DO CÓDIGO NietzscheDB

### Insights de arquitetura não óbvios

#### Pool de visited lists

```rust
// graph_layers.rs — evita alloc por busca
fn get_visited_list_from_pool(&self) -> VisitedList {
    // Reutiliza bitset com O(1) reset via generation counter
    // SEM Vec::fill(false) — sem O(N) por busca
}
```
**Para NietzscheDB:** Cada BFS/DFS aloca um HashSet novo. Com pools, o overhead por busca cai 10–30% em alta frequência.

#### Batch scoring

```rust
// Score um BATCH de IDs, não um por um
points_scorer.score_points(&mut points_ids, limit)
    .for_each(|score_point| { ... });
// Permite auto-vetorização SIMD sobre o batch
```
**Para NietzscheDB:** A distância Poincaré é calculada um nó por vez em todos os traversals. Batch scoring com SIMD seria 2–4× mais rápido.

#### ACORN-1 — busca filtrada com 2-hop expansion

```rust
// Para filtros muito seletivos (1–5% dos nós passam):
// Vizinho filtrado → explorar os vizinhos dos vizinhos
// Mantém recall quando HNSW normal teria recall < 80%
self.try_for_each_link(hop1, level, |hop2| {
    if passes_filter(hop2) { to_score.push(hop2); }
});
```
**Para NietzscheDB:** Essencial para queries com `WHERE energy > 0.9 AND hausdorff < 0.5` (muito seletivas). Implementável no executor NQL para KNN + filtro duplo.

#### Histograma de cardinalidade

```rust
// Estima cardinalidade de range query em O(log B), onde B=num_buckets
// Sem precisar varrer o índice real
struct Histogram<T> {
    borders: BTreeMap<Point<T>, Counts>,  // ~2KB por campo indexado
}
fn estimate(&self, from: T, to: T) -> CardinalityEstimation { ... }
```
**Para NietzscheDB:** O `energy_idx` já existe. Adicionar um histograma de `energy` (2KB) permitiria routing sem precisar escanear o range primeiro.

---

## PARTE 4 — MATRIZ DE PRIORIDADE

```
┌────┬──────────────────────────────────────┬──────────┬────────────────────┬──────────────┬──────────┐
│ ID │ Otimização                           │  Ganho   │ Risco Hiperbólico  │ Complexidade │ Prioridade│
├────┼──────────────────────────────────────┼──────────┼────────────────────┼──────────────┼──────────┤
│  A │ Separar embedding de CF_NODES        │ 10–20×   │ NENHUM             │ ALTA         │ CRÍTICO  │
│    │ (NodeMeta vs full Node em RocksDB)   │ get_node │                    │ breaking API │          │
├────┼──────────────────────────────────────┼──────────┼────────────────────┼──────────────┼──────────┤
│  B │ True Poincaré HNSW metric            │ Recall ✓ │ NENHUM (é bug fix) │ ALTA         │ CRÍTICO  │
│    │ (BUG-EVS-001)                        │          │                    │ fork hspace  │          │
├────┼──────────────────────────────────────┼──────────┼────────────────────┼──────────────┼──────────┤
│  E │ Filtered KNN via energy_idx          │ 5–50×    │ NENHUM             │ MÉDIA        │ ALTO     │
│    │ (routing: plain/hnsw/sample)         │ queries  │                    │              │          │
├────┼──────────────────────────────────────┼──────────┼────────────────────┼──────────────┼──────────┤
│  C │ f64 → f32 para embedding storage     │ 2× mem   │ BAIXO              │ BAIXA        │ ALTO     │
│    │ (compute em f64, store em f32)       │ 2× SIMD  │ (mitigável)        │ migration    │          │
├────┼──────────────────────────────────────┼──────────┼────────────────────┼──────────────┼──────────┤
│  D │ Scalar Quantization Int8             │ 8× mem   │ MÉDIO              │ MÉDIA        │ MÉDIO    │
│    │ (alpha=(max-min)/127, oversampling)  │ 4× KNN   │ (boundary nodes)   │              │          │
├────┼──────────────────────────────────────┼──────────┼────────────────────┼──────────────┼──────────┤
│ D+ │ Pool de visited lists (BFS/HNSW)     │ 10–30%   │ NENHUM             │ BAIXA        │ MÉDIO    │
│    │ (elimina alloc por busca)            │ latência │                    │              │          │
├────┼──────────────────────────────────────┼──────────┼────────────────────┼──────────────┼──────────┤
│ E+ │ Histograma de cardinalidade energy   │ routing  │ NENHUM             │ BAIXA        │ MÉDIO    │
│    │ (evita scan para estimativa)         │ melhor   │                    │              │          │
├────┼──────────────────────────────────────┼──────────┼────────────────────┼──────────────┼──────────┤
│ E+ │ ACORN-1 (2-hop expansion filtrada)   │ recall   │ NENHUM             │ MÉDIA        │ MÉDIO    │
│    │ (para energy < 5% dos nós)           │ alto     │                    │              │          │
├────┼──────────────────────────────────────┼──────────┼────────────────────┼──────────────┼──────────┤
│  F │ Binary Quantization                  │ 30–40×*  │ ALTO               │ ALTA         │ BAIXO    │
│    │ (XOR+POPCOUNT, só pre-filter)        │ teórico  │ (métrica errada     │              │          │
│    │                                      │          │  p/ Poincaré)       │              │          │
├────┼──────────────────────────────────────┼──────────┼────────────────────┼──────────────┼──────────┤
│  G │ HNSW Link Bit-packing                │ 2–4×     │ NENHUM             │ MUITO ALTA   │ FASE 4   │
│    │ (CompressedWithVectors format)       │ HNSW RAM │                    │ requer fork  │          │
├────┼──────────────────────────────────────┼──────────┼────────────────────┼──────────────┼──────────┤
│  H │ Gridstore (mmap pages para embeddings│ latência │ NENHUM             │ ALTA         │ FASE 4   │
│    │ sem compaction pauses do RocksDB)    │ previsív.│                    │ após item A  │          │
└────┴──────────────────────────────────────┴──────────┴────────────────────┴──────────────┴──────────┘
```

---

## PARTE 5 — ROADMAP POR FASE

### Fase 3a — Quick wins sem breaking changes

```
C. f64 → f32 para embedding storage
   Arquivos: model.rs, storage.rs, sleep/*.rs, lsystem/engine.rs
   Risco: BAIXO | Migração de dados necessária

E. Filtered KNN API — expor energy_idx ao VectorStore trait
   Arquivos: db.rs, embedded_vector_store.rs, executor.rs
   Risco: NENHUM | energy_idx já existe

D+. Pool de visited lists para BFS/DFS
    Arquivos: traversal.rs, executor.rs
    Risco: NENHUM
```

### Fase 3b — Grandes mudanças arquiteturais

```
A. Separar embedding de CF_NODES (NodeMeta + EmbeddingStore separados)
   Arquivos: model.rs, storage.rs, db.rs, + todos os callers
   Risco: MÉDIO (breaking API, migração de dados)

B. True Poincaré HNSW metric (HnswPoincareWrapper<N>)
   Arquivos: embedded_vector_store.rs, nietzsche-hnsw (fork?)
   Risco: MÉDIO (rebuild do índice HNSW necessário)
```

### Fase 3c — Quantização

```
D. Scalar Quantization Int8
   Arquivos: nova crate nietzsche-quantization
   Dependência: item A (embedding separado) facilita muito
   Risco: MÉDIO (oversampling obrigatório para boundary nodes)

E+. Histograma de cardinalidade energy (CardinalityEstimation)
    Arquivos: storage.rs, executor.rs
    Risco: NENHUM
```

### Fase 4 — Arquitetura avançada (futuro)

```
G. HNSW Link Bit-packing (CompressedWithVectors format)
   Dependência: fork de nietzsche-hnsw ou HNSW próprio

H. Gridstore para embeddings
   Dependência: item A completo

F. Binary Quantization (APENAS como pre-filter)
   Dependência: item D completo, alta dimensão (≥768)
```

---

## PARTE 6 — CONCLUSÃO EXECUTIVA

### Os 2 problemas mais críticos (específicos de ser banco multi-manifold)

1. **Embedding duplicado em CF_NODES** — NÃO é um problema do NietzscheDB (que separa payload de vetores desde o início). É uma dívida técnica específica do NietzscheDB. Para `dim=3072`, cada `get_node()` processa 25KB sendo 24KB de embedding já disponível no HNSW mmap.

2. **CosineMetric em vez de PoincareMetric no HNSW** — O banco se chama NietzscheDB, opera em geometria hiperbólica, mas o grafo HNSW usa métrica cosseno. Os vizinhos no grafo HNSW são incorretos para a geometria que o banco anuncia suportar.

### O que o NietzscheDB tem que NietzscheDB não tem (e que funciona para Poincaré)

| Feature NietzscheDB | Impacto em NietzscheDB | Aplicável? |
|----------------|------------------------|------------|
| Payload separado de vetores | 10-20× get_node() | SIM (item A) |
| Cardinality routing (plain/hnsw/sample) | 5-50× queries filtradas | SIM (item E) |
| Scalar Quantization Int8 | 8× memória + 4× KNN | SIM com oversampling |
| Pool de visited lists | 10-30% latência BFS | SIM |
| Histograma de cardinalidade | routing melhor | SIM |
| ACORN-1 (2-hop filtered) | recall em queries seletivas | SIM |
| CompressedWithVectors HNSW | 2-4× RAM grafo | SIM (fase 4) |
| Binary Quantization | 30-40× KNN | NÃO como métrica primária |
| Gridstore mmap pages | latência previsível | SIM (fase 4) |

### O que o NietzscheDB NÃO tem e o NietzscheDB TEM (vantagens a preservar)

| Feature NietzscheDB | Descrição |
|---------------------|-----------|
| Geometria hiperbólica nativa | Poincaré ball com invariante ‖x‖ < 1.0 |
| Grafo com semântica | BFS, DFS, diffusion, L-System, Hausdorff |
| Energy / Zaratustra / Sleep | Ciclos de vida dos nós |
| NQL — linguagem de query | `MATCH (n) WHERE n.energy > 0.5` |
| WAL + transações (saga) | Durabilidade com rollback |
| Hierarquia hiperbólica | Depth = ‖embedding‖, nós Semantic/Episodic |

---

*Auditoria gerada em 2026-02-19 — pesquisa do código NietzscheDB commit 9f433b1*
*Nenhum código foi modificado neste documento. Aguardando aprovação para iniciar implementação.*
