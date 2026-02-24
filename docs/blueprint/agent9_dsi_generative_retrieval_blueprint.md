# AGENT 9: DSI Generative Retrieval Blueprint

## 1. Current Search/Retrieval Infrastructure

### 1.1 HNSW Implementation (hyperspace-index)

NietzscheDB has a full multi-layer HNSW (Hierarchical Navigable Small World) graph implementation in `hyperspace-index`.

**Core structure** (`crates/hyperspace-index/src/lib.rs`):
- `HnswIndex<const N: usize, M: Metric<N>>` — generic over dimension `N` and metric `M`
- Multi-layer navigable small world graph with up to `MAX_LAYERS = 16` levels
- Topology stored as `Vec<Node>` where each `Node` has `Vec<RwLock<Vec<NodeId>>>` per layer
- Uses `parking_lot::RwLock` for concurrent read/write (fine-grained per-layer locks)
- Thread-local `VisitedScratch` with generation-based visited tracking (no per-query allocation)

**HNSW Parameters** (from `crates/hyperspace-core/src/config.rs`):
- `M` (max connections per layer): default `16`, configurable via `GlobalConfig::set_m()`
- `ef_construction`: default `100`, configurable at runtime via atomics
- `ef_search`: default `100`, configurable at runtime via atomics
- Layer 0 connectivity: `2 * M` (double density for better recall)
- Layer selection: geometric distribution with `p = 0.5`

**Search flow** (`HnswIndex::search()`, line ~638):
1. Build filter bitmap from metadata (RoaringBitmap intersection of tag filters, complex filters, minus deleted)
2. Create `HyperVector<N>` from query, validate against metric
3. Zoom-in phase: greedy descent from entry point through layers `max_layer` down to layer 1
4. Local search: `search_layer0()` with ef-bounded beam search over layer 0 neighbors
5. Filter: only `is_valid(id)` candidates enter the result set (bitmap or deleted check)
6. Return top-k sorted by distance ascending

**Neighbor selection heuristic** (`select_neighbors()`, line ~1129):
- Diversity-aware: rejects candidates closer to existing selected neighbors than to the query
- This is the standard HNSW heuristic for graph quality

### 1.2 GPU Acceleration (cuVS CAGRA)

**Crate**: `nietzsche-hnsw-gpu` (`crates/nietzsche-hnsw-gpu/src/lib.rs`)

- Wraps NVIDIA cuVS CAGRA (Compressed-Adjacency Graph Retrieval Algorithm)
- Feature-gated behind `--features cuda`
- `GPU_THRESHOLD = 1_000` — below this, CPU linear scan (transfer overhead not worthwhile)
- `REBUILD_DELTA_RATIO = 0.10` — rebuild GPU index when 10% of entries are dirty
- Lazy build: GPU index built on first qualifying `knn()` call
- Soft-delete with compaction on rebuild
- **Fallback**: if GPU build or search fails, falls back to CPU linear scan with `eprintln!` warning
- Implements the `VectorStore` trait from `nietzsche-graph`

**GPU compute shaders** (`crates/hyperspace-core/src/gpu.rs`):
- WGSL compute shader for batch Lorentz SQ8 distance computation
- 256 threads per workgroup, dequantizes i8 coords with per-vector scale factor
- CPU reference implementation for validation

### 1.3 Metrics (Distance Functions)

All metrics implement `Metric<N>` trait (`crates/hyperspace-core/src/lib.rs`):

| Metric | Distance | Used For |
|--------|----------|----------|
| `PoincareMetric` | `acosh(1 + 2*||u-v||^2 / ((1-||u||^2)(1-||v||^2)))` | Knowledge graph hierarchy |
| `EuclideanMetric` | L2 squared distance (f32 math) | General embeddings |
| `CosineMetric` | L2 on normalized vectors | Text/audio/image embeddings |
| `LorentzMetric` | `acosh(-<a,b>_L)` Minkowski inner product | Hyperboloid model |

Each metric supports three quantization modes:
- `None` — full f64 precision
- `ScalarI8` — 8-bit scalar quantization (`QuantizedHyperVector`)
- `Binary` — 1-bit per dimension (`BinaryHyperVector`) **NOTE: Binary quantization panics for LorentzMetric — sign(x) destroys hierarchical info (per CLAUDE.md REJECTED rule)**

**Poincare distance in nietzsche-graph** (`crates/nietzsche-graph/src/model.rs`, line ~103):
- `PoincareVector::distance()` — single-pass computation of diff_sq, norm_u_sq, norm_v_sq
- f32 storage, f64 internal math (ITEM C committee decision — 50% memory saving)
- `project_into_ball()` — soft-clamp to ||x|| < 0.999 for numerical stability

### 1.4 BM25 Full-Text Search

**Crate**: `nietzsche-graph/src/fulltext.rs`

- Complete BM25 implementation with inverted index
- Storage in RocksDB `meta` column family with `fts:` prefix
- Key format: `fts:term:{token}` -> `Vec<(Uuid, u32)>` posting lists (bincode)
- Document length: `fts:dl:{node_id}` -> `u32`
- Document count: `fts:doc_count` -> `u64`
- BM25 parameters: k1 = 1.2, b = 0.75
- Tokenizer: whitespace split + lowercase + alphanumeric filter + stop-word removal
- `auto_index_content()` — recursively extracts string values from JSON content
- Returns `Vec<FtsResult>` sorted by BM25 score descending

### 1.5 Hybrid Search (RRF Fusion)

**Location**: `HnswIndex::search_hybrid()` (`crates/hyperspace-index/src/lib.rs`, line ~1740)

- Activated when `hybrid_query` is present in `SearchParams`
- Vector path: HNSW search for top `k*2` candidates (recall buffer)
- Keyword path: tokenize query, scan `_txt:{token}` inverted index, score by token overlap
- Fusion: **Reciprocal Rank Fusion (RRF)** — `RRF_score = 1/(alpha + rank_vec) + 1/(alpha + rank_key)`
- Default alpha = 60.0 (standard RRF constant)
- Output: `(id, 10.0 - score)` to maintain "smaller is better" distance convention

### 1.6 Filtered KNN Search

**Crate**: `nietzsche-filtered-knn` (`crates/nietzsche-filtered-knn/src/`)

- Brute-force KNN over Roaring Bitmap-filtered subsets
- Uses **Poincare distance** (not Euclidean) for the actual distance computation
- Max-heap of size k with early pruning (skip candidates farther than worst in heap)
- `NodeFilter` enum: `EnergyRange`, `NodeType`, `ContentField`, `ContentFieldExists`, `And`, `Or`
- `build_filter_bitmap()` converts NQL WHERE clauses to RoaringBitmap
- Designed for small filtered subsets (< 10k nodes); HNSW for large unfiltered searches
- Can compose with HNSW: pass bitmap into `search_layer0`'s `allowed` parameter

### 1.7 NQL Query Pipeline

**Crate**: `nietzsche-query` (`crates/nietzsche-query/src/`)

**Query execution flow**:
1. **Parse** (`parser.rs`): NQL text -> `Query` AST
2. **Cost estimate** (`cost.rs`): AST -> `CostEstimate` (scan type, estimated rows, cost)
3. **Execute** (`executor.rs`): AST + storage + params -> `Vec<QueryResult>`

**Scan strategies** (from cost.rs):
- `FullScan` — CF_NODES scan, ~5 us/row
- `EnergyIndexScan` — CF_ENERGY_IDX range scan, ~2 us/row
- `MetaIndexScan` — CF_META_IDX prefix scan, ~2 us/row
- `EdgeScan` — CF_ADJ_OUT/CF_ADJ_IN, ~3 us/row
- `BoundedBFS` — multi-hop traversal, ~10 us/hop
- `DiffusionWalk` — energy-biased random walk

**HYPERBOLIC_DIST execution** (`executor.rs`, line ~1495):
- `Expr::HyperbolicDist { alias, field, arg }` -> resolve node embedding + query vector -> `node.embedding.distance(&query_vec)`
- Only supports `.embedding` field
- Used in WHERE clause filtering and ORDER BY sorting
- Also supports `SENSORY_DIST`, `POINCARE_DIST` (alias), and 13 mathematician-named functions

**Parallel execution** (`executor.rs`, line ~648):
- `PARALLEL_SCAN_THRESHOLD = 2_000` nodes triggers rayon parallel filter
- Gas tracking with `GasTracker` (50,000 default budget, 1 gas/node, 2 gas/edge)

### 1.8 Secondary Indexes

**Crate**: `nietzsche-secondary-idx` (`crates/nietzsche-secondary-idx/src/`)

- Indexes on arbitrary JSON content fields (dot-separated paths)
- Three types: `String` (lexicographic), `Float` (IEEE 754 sign-magnitude ordered), `Int` (big-endian sign-flip)
- Storage in CF_META: definitions under `idx_def:{name}`, entries under `sidx:{name}:{sortable_value}:{node_id}`
- Operations: `create_index` (with backfill), `drop_index`, `insert_entry`, `lookup`, `range_lookup`
- LRU cache for `list_indexes` to avoid repeated CF_META scans

### 1.9 Metadata Index (HNSW Layer)

**Location**: `MetadataIndex` in `hyperspace-index/src/lib.rs`

- `inverted: DashMap<String, RoaringBitmap>` — tag-based filtering (`key:value` and `_txt:token`)
- `numeric: DashMap<String, BTreeMap<i64, RoaringBitmap>>` — numeric range filtering
- `deleted: RwLock<RoaringBitmap>` — soft-delete bitfield
- `forward: DashMap<u32, HashMap<String, String>>` — ID -> metadata for reverse lookup

### 1.10 AutoTuner

**Location**: `crates/hyperspace-core/src/auto_tuner.rs`

- Rolling window of query latencies and result counts
- Auto-adjusts `ef_search` based on p95 latency vs target and estimated recall vs target
- Clamped to `[EF_MIN=10, EF_MAX=1000]`

### 1.11 Current Retrieval Paths Summary

```
Query arrives (NQL or gRPC)
    |
    +-- MATCH (n) WHERE HYPERBOLIC_DIST(...) < threshold
    |       -> scan_nodes() or energy_index_scan()
    |       -> filter by Poincare distance in executor
    |       -> ORDER BY distance, LIMIT k
    |
    +-- VectorStore::knn(query, k)
    |       -> EmbeddedVectorStore -> HnswIndex::search()
    |       -> Zoom-in (greedy top layers)
    |       -> Beam search layer 0 with RoaringBitmap filter
    |       -> Return top-k (uuid, distance)
    |
    +-- VectorStore::knn_filtered(query, k, filter)
    |       -> HNSW with metadata pre-filter pushed down
    |
    +-- GpuVectorStore::knn(query, k)
    |       -> cuVS CAGRA GPU search (if n >= 1000 + cuda feature)
    |       -> CPU linear scan fallback
    |
    +-- filtered_knn(query, k, bitmap, node_ids, storage)
    |       -> Brute-force Poincare KNN over bitmap-selected nodes
    |
    +-- FullTextIndex::search(query, limit)
    |       -> BM25 scoring over inverted index
    |
    +-- HnswIndex::search_hybrid(query, k, ef, filter, text, alpha)
            -> HNSW vector results + keyword results
            -> RRF Fusion
```

---

## 2. ONNX Tensor Specification

### 2.1 DSI Model (Generative Retrieval)

**Input**:
```
query_embedding: float32[batch_size, query_dim]
    # Poincare ball coordinates (f32, dim matches NIETZSCHE_VECTOR_DIM)
    # Default: [B, 3072] for text embeddings, [B, 1024] for smaller models
```

**Output (Option A — Autoregressive Doc ID Generation)**:
```
doc_id_tokens: int32[batch_size, max_doc_id_length]
    # Hierarchical document ID as token sequence
    # Each position = cluster assignment at that level
    # max_doc_id_length = ceil(log(N_docs) / log(cluster_size))
    # Example: 3-level hierarchy with 256 clusters -> [B, 3]

eos_mask: bool[batch_size, max_doc_id_length]
    # True at the position where generation should stop
```

**Output (Option B — Score-based Candidate Selection)**:
```
relevance_scores: float32[batch_size, K]
    # Scores for top-K candidates pre-selected by HNSW
    # Higher = more relevant

candidate_indices: int32[batch_size, K]
    # Which HNSW candidates these scores correspond to
```

### 2.2 Neural Re-ranker Model

**Input**:
```
query_doc_pairs: float32[batch_size, query_dim + doc_dim]
    # Concatenation of query embedding and document embedding
    # For Poincare: [B, 3072 + 3072] = [B, 6144]

# Alternative cross-encoder input:
query_tokens: int32[batch_size, max_query_tokens]
doc_tokens: int32[batch_size, max_doc_tokens]
```

**Output**:
```
relevance_score: float32[batch_size, 1]
    # Scalar relevance score per query-document pair
    # Used to re-rank HNSW candidates
```

### 2.3 Hyperbolic-Aware DSI Extension

```
query_poincare: float32[batch_size, dim]
    # Query point in Poincare ball
poincare_norms: float32[batch_size, 1]
    # ||query||^2 precomputed for distance kernel
hierarchy_level: int32[batch_size, 1]
    # Depth in the Poincare ball (derived from norm)
    # Closer to boundary = deeper/more specific
```

---

## 3. Neural Architecture

### 3.1 DSI (Differentiable Search Index)

A DSI is an encoder-decoder Transformer that **memorizes** the mapping from queries to document identifiers. Instead of building an external index, the model IS the index.

**Architecture**:
```
Encoder: TransformerEncoder(
    input_dim = query_dim,      # 3072 (Poincare embedding dim)
    hidden_dim = 768,           # Internal representation
    num_layers = 6,             # Depth of encoder
    num_heads = 12,             # Multi-head attention
    dropout = 0.1
)

Decoder: TransformerDecoder(
    vocab_size = cluster_size,  # Size of ID token vocabulary (e.g., 256)
    max_length = id_depth,      # Max hierarchical ID levels (e.g., 4)
    hidden_dim = 768,
    num_layers = 6,
    num_heads = 12
)
```

**Document ID Representation** (three options, recommend Option C for NietzscheDB):

- **Option A — Atomic IDs**: Each document gets a unique integer ID. Simple but does not scale (vocabulary = corpus size).
- **Option B — Semantic IDs**: Cluster documents by embedding similarity, assign hierarchical cluster path as ID. E.g., `[cluster_level1=42, cluster_level2=17, cluster_level3=8]`.
- **Option C — Hyperbolic Hierarchical IDs** (recommended): Leverage Poincare ball structure. Partition the ball into nested annular rings by norm (depth), then angular sectors. The ID encodes the hierarchical position:
  ```
  ID = [depth_band, angular_sector_1, angular_sector_2, ...]
  ```
  This naturally aligns with NietzscheDB's hyperbolic geometry where ||x|| encodes depth/specificity.

**Training Data**:
- Source: NQL query logs + HNSW search results
- Pairs: `(query_embedding, relevant_doc_id)` from successful retrievals
- Negative sampling: random documents from other depth bands
- Fine-tuning signal: user feedback (click/dwell on returned results)

**Training Protocol**:
1. Pre-train on (random_embedding, hierarchical_id) pairs from existing corpus
2. Fine-tune on (query, clicked_doc_id) pairs from usage logs
3. Periodic re-indexing: as corpus grows, retrain to memorize new documents

### 3.2 Neural Re-ranker (Recommended First Step)

A simpler, more practical first implementation:

```
CrossEncoder(
    input: concatenate(query_emb, doc_emb)  # [B, 2*dim]
    layers:
        Linear(2*dim, 512) + LayerNorm + GELU
        Linear(512, 256) + LayerNorm + GELU
        Linear(256, 1) + Sigmoid
    output: relevance_score  # [B, 1] in [0, 1]
)
```

**Workflow**:
1. HNSW returns top-100 candidates with distances
2. For each candidate: load embedding, concatenate with query, run through re-ranker
3. Re-rank by neural score, return top-10

**Hyperbolic-Aware Variant**:
```
HyperbolicReranker(
    input: [query_emb, doc_emb, poincare_dist, depth_query, depth_doc]
    # poincare_dist = precomputed hyperbolic distance
    # depth_* = norm of embedding (encodes hierarchy level)
    layers:
        Linear(2*dim + 3, 512) + LayerNorm + GELU
        Linear(512, 256) + LayerNorm + GELU
        Linear(256, 1) + Sigmoid
)
```

### 3.3 Hybrid Fusion Architecture

```
Query arrives
    |
    +--[HNSW Path]---> top-100 candidates (recall-optimized)
    |                     |
    +--[BM25 Path]---> top-50 keyword matches
    |                     |
    +--[DSI Path]----> top-20 generative predictions (precision-optimized)
    |                     |
    v                     v
  +---------------------------+
  | Fusion Layer              |
  |   - RRF across 3 sources  |
  |   - Neural re-ranker      |
  |   - Poincare distance     |
  |     as tiebreaker         |
  +---------------------------+
              |
              v
        Final top-K results
```

---

## 4. Integration Point

### 4.1 Option A: DSI as Additional Retrieval Path (Recommended)

**Where**: Add a `NeuralRetriever` alongside `VectorStore` in `NietzscheDB`.

```rust
// crates/nietzsche-neural/src/lib.rs (new crate)

pub trait NeuralRetriever: Send + Sync {
    /// Generative retrieval: query -> predicted document IDs
    fn retrieve(&self, query: &PoincareVector, k: usize) -> Result<Vec<(Uuid, f64)>, NeuralError>;

    /// Re-rank candidates from HNSW
    fn rerank(
        &self,
        query: &PoincareVector,
        candidates: &[(Uuid, f64)],  // (id, hnsw_distance)
    ) -> Result<Vec<(Uuid, f64)>, NeuralError>;

    /// Check if model is loaded and ready
    fn is_ready(&self) -> bool;
}
```

**Integration in NietzscheDB** (`crates/nietzsche-graph/src/db.rs`):
```rust
pub struct NietzscheDB<V: VectorStore> {
    storage:      Arc<GraphStorage>,
    wal:          GraphWal,
    adjacency:    AdjacencyIndex,
    vector_store: RwLock<V>,
    // NEW: Optional neural retriever
    #[cfg(feature = "neural")]
    neural:       Option<Arc<dyn NeuralRetriever>>,
}
```

**Integration in VectorStore trait** (`crates/nietzsche-graph/src/db.rs`, after line ~131):
```rust
/// Neural-augmented KNN: HNSW recall + neural re-ranking for precision.
///
/// Only available with the `neural` feature flag.
#[cfg(feature = "neural")]
fn knn_neural(
    &self,
    query: &PoincareVector,
    k: usize,
    neural: &dyn NeuralRetriever,
) -> Result<Vec<(Uuid, f64)>, GraphError> {
    // 1. HNSW returns top-100 (recall buffer)
    let candidates = self.knn(query, k * 10)?;
    // 2. Neural re-ranker selects top-k
    let reranked = neural.rerank(query, &candidates)
        .map_err(|e| GraphError::Storage(format!("neural rerank: {e}")))?;
    Ok(reranked.into_iter().take(k).collect())
}
```

### 4.2 Option B: Neural Re-ranker After HNSW (Simplest)

**Where**: Insert between HNSW search and result return in `HnswIndex::search()`.

This requires the least structural change. After `search_layer0()` returns candidates, feed them through the ONNX re-ranker.

```rust
// In HnswIndex::search() after line ~834:
let mut results = self.search_layer0(curr_node, &q_vec, k, ef_search, allowed_bitmap.as_ref());

// Neural re-ranking (feature-gated)
#[cfg(feature = "neural")]
if let Some(ref reranker) = self.neural_reranker {
    results = reranker.rerank_pairs(&query, &results);
}
```

### 4.3 Option C: DSI for Generative Queries in NQL

**Where**: New NQL query form that routes to DSI:

```sql
-- Standard HNSW path (unchanged)
MATCH (n) WHERE HYPERBOLIC_DIST(n.embedding, $q) < 0.5 RETURN n LIMIT 10

-- New: Neural retrieval path
MATCH (n) WHERE NEURAL_RETRIEVE(n.embedding, $q) RETURN n LIMIT 10

-- New: Hybrid with neural re-ranking
MATCH (n) WHERE HYPERBOLIC_DIST(n.embedding, $q) < 0.5
RERANK NEURAL RETURN n LIMIT 10
```

**AST additions** (`crates/nietzsche-query/src/ast.rs`):
```rust
pub enum Expr {
    // ... existing variants ...
    /// Neural retrieval: NEURAL_RETRIEVE(alias.embedding, $query)
    NeuralRetrieve {
        alias: String,
        field: String,
        arg: HDistArg,
    },
}

pub struct MatchQuery {
    // ... existing fields ...
    /// Optional neural re-ranking strategy
    pub rerank: Option<RerankStrategy>,
}

pub enum RerankStrategy {
    Neural,
    // Future: CrossEncoder, ColBERT, etc.
}
```

### 4.4 NQL Query Pipeline Integration Point

The most natural integration point is in `execute_node_match()` (`crates/nietzsche-query/src/executor.rs`, line ~604):

```
execute_node_match()
    |
    +-- [1] Scan: energy index / meta index / full scan
    |       (existing, unchanged)
    |
    +-- [2] Filter: eval_conditions with HYPERBOLIC_DIST
    |       (existing, unchanged)
    |
    +-- [NEW 2.5] Neural augmentation:
    |       if neural retriever available AND conditions contain NEURAL_RETRIEVE:
    |           dsi_results = neural.retrieve(query, k * 3)
    |           merge dsi_results with filtered candidates
    |       elif rerank == Neural:
    |           pass filtered candidates through neural.rerank()
    |
    +-- [3] ORDER BY
    +-- [4] LIMIT
```

### 4.5 Feature Flag

```toml
# crates/nietzsche-neural/Cargo.toml
[package]
name = "nietzsche-neural"

[dependencies]
ort = { version = "2.0", optional = true }  # ONNX Runtime
nietzsche-graph = { path = "../nietzsche-graph" }

[features]
default = []
onnx = ["ort"]
```

```toml
# Root Cargo.toml addition
[features]
neural = ["nietzsche-neural/onnx"]
```

### 4.6 Latency Budget

| Component | Target Latency | Notes |
|-----------|---------------|-------|
| HNSW search (top-100) | 1-5 ms | Existing, well-optimized |
| BM25 search | 1-3 ms | Existing |
| DSI inference (top-20) | 5-15 ms | ONNX CPU, batch=1 |
| Re-ranker (100 candidates) | 3-10 ms | ONNX CPU, batch=100 |
| RRF fusion | < 0.1 ms | Simple rank merge |
| **Total hybrid** | **10-30 ms** | **Acceptable for interactive** |

GPU inference (if CUDA available): DSI < 2 ms, re-ranker < 1 ms.

---

## 5. Safety and Fallback

### 5.1 Mandatory Fallback

```rust
impl OnnxNeuralRetriever {
    pub fn retrieve(&self, query: &PoincareVector, k: usize) -> Result<Vec<(Uuid, f64)>, NeuralError> {
        match self.run_inference(query, k) {
            Ok(results) => Ok(results),
            Err(e) => {
                // MANDATORY: log error and return empty so caller falls back to HNSW
                tracing::warn!(error = %e, "Neural retrieval failed, falling back to HNSW");
                Err(NeuralError::InferenceFailed(e.to_string()))
            }
        }
    }
}

// In the calling code:
let neural_results = match neural.retrieve(&query, k) {
    Ok(r) => r,
    Err(_) => vec![],  // Empty = pure HNSW path, no degradation
};
```

### 5.2 Recall Safety (Critical for Medical Context)

DSI must **NEVER** be the sole retrieval path. The architecture is strictly **additive**:

```
Final_Results = HNSW_Results UNION DSI_Results
                    |
                    v
              Neural_Rerank (precision improvement only)
```

- HNSW is the **recall backbone** — its results are always included
- DSI can only ADD candidates that HNSW missed, never remove HNSW candidates
- If DSI returns a candidate not in HNSW top-K, it is added to the candidate pool and re-ranked
- If DSI contradicts HNSW ordering, HNSW distance is used as tiebreaker

**Recall monitoring**:
```rust
struct RetrievalMetrics {
    hnsw_candidates: usize,
    dsi_candidates: usize,
    dsi_unique_additions: usize,  // DSI found but HNSW missed
    final_results: usize,
    hnsw_recall_at_k: f64,       // How many final results came from HNSW
}
```

### 5.3 Poincare-Aware Retrieval

All neural components must respect hyperbolic geometry:

1. **DSI training**: Use Poincare distance (not Euclidean) for similarity labels
2. **Re-ranker input**: Include Poincare distance as an explicit feature alongside embeddings
3. **Hierarchical ID assignment**: Based on Poincare ball depth bands (norm = hierarchy level)
4. **Distance output**: Neural scores must be convertible to Poincare-compatible distances for fusion

```rust
/// Convert neural relevance score to Poincare-compatible distance.
/// Higher score = more relevant = shorter distance.
fn neural_score_to_distance(score: f64) -> f64 {
    // Inverse logistic mapping: score in [0,1] -> distance in [0, inf)
    // Calibrated against Poincare distance distribution
    (-score.ln()).max(0.0)
}
```

### 5.4 NEVER Use Binary Quantization for DSI Embeddings

Per CLAUDE.md absolute rule (ITEM F — REJECTED PERMANENTLY):
- Binary quantization via `sign(x)` destroys the magnitude information in Poincare ball coordinates
- Magnitude encodes depth/hierarchy (distance from origin = specificity)
- DSI embeddings that represent Poincare positions MUST use full precision or ScalarI8 (with oversampling)
- The ONLY exception: pre-filter with oversampling >= 30x and mandatory rescore

### 5.5 Model Versioning and Rollback

```rust
struct NeuralModelConfig {
    model_path: PathBuf,          // Path to .onnx file
    model_version: String,        // Semantic version
    fallback_enabled: bool,       // Always true in production
    max_inference_ms: u64,        // Timeout for ONNX session
    min_recall_threshold: f64,    // If recall drops below, disable neural
    canary_percentage: f64,       // % of queries routed through neural (gradual rollout)
}
```

### 5.6 Monitoring and Circuit Breaker

```rust
struct NeuralCircuitBreaker {
    failure_count: AtomicU32,
    total_count: AtomicU32,
    last_failure: AtomicU64,      // Unix timestamp
    state: AtomicU8,              // 0=closed, 1=half-open, 2=open
}

impl NeuralCircuitBreaker {
    fn should_allow(&self) -> bool {
        match self.state.load(Ordering::Relaxed) {
            0 => true,                                    // Closed: allow
            1 => self.total_count.load(Ordering::Relaxed) % 10 == 0,  // Half-open: 10%
            2 => {                                        // Open: check cooldown
                let elapsed = now() - self.last_failure.load(Ordering::Relaxed);
                if elapsed > 30_000 {                     // 30s cooldown
                    self.state.store(1, Ordering::Relaxed);
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}
```

---

## 6. Implementation Roadmap

### Phase 1: Neural Re-ranker (Lowest Risk, Highest Value)
1. Create `nietzsche-neural` crate with ONNX Runtime integration
2. Train a simple MLP cross-encoder on (query, doc, relevance) triples from HNSW logs
3. Export to ONNX, integrate as post-HNSW re-ranker
4. Feature flag: `cargo features = ["neural"]`
5. A/B test: measure precision@10 improvement over pure HNSW

### Phase 2: Hyperbolic Hierarchical IDs
1. Partition Poincare ball into hierarchical zones (depth bands + angular sectors)
2. Assign every node a hierarchical ID based on its embedding position
3. Store as node metadata (persisted in RocksDB)
4. This becomes the target vocabulary for DSI training

### Phase 3: DSI Encoder-Decoder
1. Train Transformer DSI on (query_embedding, hierarchical_id) pairs
2. Export encoder-decoder to ONNX
3. Integrate as parallel retrieval path alongside HNSW
4. Fusion via RRF (extending existing `search_hybrid` pattern)

### Phase 4: Online Learning
1. Capture query-click feedback from gRPC responses
2. Fine-tune DSI and re-ranker incrementally
3. Periodic model refresh with circuit breaker rollback

---

## 7. File Reference Index

| Component | File | Key Functions/Structs |
|-----------|------|----------------------|
| HNSW Index | `crates/hyperspace-index/src/lib.rs` | `HnswIndex::search()`, `search_layer0()`, `search_hybrid()`, `index_node()` |
| HNSW Config | `crates/hyperspace-core/src/config.rs` | `GlobalConfig` (M=16, ef=100) |
| Metrics | `crates/hyperspace-core/src/lib.rs` | `PoincareMetric`, `EuclideanMetric`, `LorentzMetric`, `CosineMetric` |
| Vector Types | `crates/hyperspace-core/src/vector.rs` | `HyperVector<N>`, `QuantizedHyperVector<N>`, `BinaryHyperVector<N>` |
| GPU Compute | `crates/hyperspace-core/src/gpu.rs` | `LORENTZ_DISTANCE_WGSL`, `batch_lorentz_distance_cpu()` |
| AutoTuner | `crates/hyperspace-core/src/auto_tuner.rs` | `AutoTuner` |
| GPU VectorStore | `crates/nietzsche-hnsw-gpu/src/lib.rs` | `GpuVectorStore`, `GpuState::build_gpu_index()`, `gpu_search()` |
| Filtered KNN | `crates/nietzsche-filtered-knn/src/search.rs` | `filtered_knn()`, `brute_force_knn()` |
| Filter Builder | `crates/nietzsche-filtered-knn/src/filter.rs` | `NodeFilter`, `build_filter_bitmap()` |
| NQL AST | `crates/nietzsche-query/src/ast.rs` | `Query`, `MatchQuery`, `Expr::HyperbolicDist` |
| NQL Parser | `crates/nietzsche-query/src/parser.rs` | `parse()` |
| NQL Executor | `crates/nietzsche-query/src/executor.rs` | `execute()`, `execute_node_match()`, `eval_condition_with_edges()` |
| NQL Cost | `crates/nietzsche-query/src/cost.rs` | `estimate_cost()`, `ScanType` |
| Secondary Idx | `crates/nietzsche-secondary-idx/src/builder.rs` | `SecondaryIndexBuilder` |
| Secondary Store | `crates/nietzsche-secondary-idx/src/store.rs` | `SecondaryIndexStore`, `encode_sortable_value()` |
| VectorStore Trait | `crates/nietzsche-graph/src/db.rs` | `trait VectorStore`, `knn()`, `knn_filtered()` |
| PoincareVector | `crates/nietzsche-graph/src/model.rs` | `PoincareVector::distance()`, `project_into_ball()` |
| Embedded HNSW | `crates/nietzsche-graph/src/embedded_vector_store.rs` | `EmbeddedVectorStore`, `DynHnsw` |
| BM25 Full-text | `crates/nietzsche-graph/src/fulltext.rs` | `FullTextIndex::search()`, `tokenize()` |
| GraphStorage | `crates/nietzsche-graph/src/storage.rs` | `GraphStorage` (RocksDB) |
| NietzscheDB | `crates/nietzsche-graph/src/db.rs` | `NietzscheDB<V: VectorStore>` |
