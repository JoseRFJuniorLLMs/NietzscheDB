//! EmbeddedVectorStore — real HNSW-backed vector store using CosineMetric.
//!
//! Replaces `MockVectorStore` in production. Wraps `hyperspace-index`'s
//! `HnswIndex<N, CosineMetric>` with type erasure so the dimension is
//! chosen at runtime via an environment variable.
//!
//! ## Environment variables
//! - `NIETZSCHE_VECTOR_DIM`    — vector dimension (default: `3072`)
//! - `NIETZSCHE_VECTOR_METRIC` — `"cosine"` | `"euclidean"` (default: `"cosine"`)
//!
//! ## Supported dimensions
//! 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 3072
//! (matches common embedding model outputs used in EVA-Mind)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dashmap::DashMap;
use uuid::Uuid;

use hyperspace_core::{CosineMetric, GlobalConfig, QuantizationMode};
use hyperspace_index::HnswIndex;
use hyperspace_store::VectorStore as RawStore;

use crate::db::VectorStore;
use crate::error::GraphError;
use crate::model::PoincareVector;

// ─── Internal type-erased HNSW ───────────────────────────────────────────────

/// Type-erased HNSW operations that erase the `const N: usize` generic.
/// All concrete implementations live in `HnswCosineWrapper<N>`.
trait DynHnsw: Send + Sync {
    /// Insert a (normalized) vector and return the internal HNSW node id.
    fn hnsw_insert(&self, vector: &[f64], uuid_str: &str) -> Result<u32, String>;

    /// Search for `k` nearest neighbors. Returns `(hnsw_id, distance)` pairs.
    fn hnsw_search(&self, query: &[f64], k: usize) -> Vec<(u32, f64)>;

    /// Soft-delete a node by its internal HNSW id.
    fn hnsw_delete(&self, id: u32);

    /// Recover the UUID string stored in the HNSW forward metadata index.
    fn hnsw_get_uuid_str(&self, id: u32) -> Option<String>;

    /// Nominal vector dimension this index was created for.
    fn dim(&self) -> usize;
}

// ─── Concrete Cosine HNSW wrapper ────────────────────────────────────────────

struct HnswCosineWrapper<const N: usize> {
    index: HnswIndex<N, CosineMetric>,
}

impl<const N: usize> HnswCosineWrapper<N> {
    fn new(storage_dir: &Path) -> Self {
        // element_size = size_of::<HyperVector<N>>() via the public SIZE const.
        let element_size = hyperspace_core::vector::HyperVector::<N>::SIZE;
        let storage = Arc::new(RawStore::new(storage_dir, element_size));
        let config = Arc::new(GlobalConfig::default());
        Self {
            index: HnswIndex::new(storage, QuantizationMode::None, config),
        }
    }

    /// L2-normalize a vector to the unit sphere.
    ///
    /// `CosineMetric` delegates to `EuclideanMetric`, so ranking is preserved
    /// when all stored vectors and queries are unit-normalized.
    /// Mirrors the `normalize_if_cosine` logic in `hyperspace-server`.
    #[inline]
    fn normalize(vector: &[f64]) -> Vec<f64> {
        let norm_sq: f64 = vector.iter().map(|x| x * x).sum();
        // If already unit length (within ε) or zero vector, skip allocation.
        if (norm_sq - 1.0).abs() < 1e-9 || norm_sq <= 1e-18 {
            return vector.to_vec();
        }
        let inv = 1.0 / norm_sq.sqrt();
        vector.iter().map(|x| x * inv).collect()
    }
}

impl<const N: usize> DynHnsw for HnswCosineWrapper<N> {
    fn hnsw_insert(&self, vector: &[f64], uuid_str: &str) -> Result<u32, String> {
        let normalized = Self::normalize(vector);
        let mut meta = HashMap::new();
        // "nid" = nietzsche id — used for reverse lookup after search.
        meta.insert("nid".to_string(), uuid_str.to_string());
        self.index.insert(&normalized, meta)
    }

    fn hnsw_search(&self, query: &[f64], k: usize) -> Vec<(u32, f64)> {
        let normalized = Self::normalize(query);
        self.index.search(
            &normalized,
            k,
            100,                // ef_search — balance recall vs latency
            &HashMap::new(),    // legacy tag filters (unused here)
            &[],                // complex filters (unused here)
            None,               // hybrid_query
            None,               // hybrid_alpha
        )
    }

    fn hnsw_delete(&self, id: u32) {
        self.index.delete(id);
    }

    fn hnsw_get_uuid_str(&self, id: u32) -> Option<String> {
        self.index
            .metadata
            .forward
            .get(&id)
            .and_then(|m| m.get("nid").cloned())
    }

    fn dim(&self) -> usize {
        N
    }
}

// ─── Concrete Raw (non-normalizing) HNSW wrapper — BUG-EVS-001 fix ───────────
//
// Euclidean and PoincaréBall metrics must NOT pre-normalize vectors before
// insertion into the HNSW index.  `HnswRawWrapper<N>` is identical to
// `HnswCosineWrapper<N>` except `hnsw_insert` and `hnsw_search` pass the
// vector through unchanged.

struct HnswRawWrapper<const N: usize> {
    index: HnswIndex<N, CosineMetric>,
}

impl<const N: usize> HnswRawWrapper<N> {
    fn new(storage_dir: &Path) -> Self {
        let element_size = hyperspace_core::vector::HyperVector::<N>::SIZE;
        let storage = Arc::new(RawStore::new(storage_dir, element_size));
        let config = Arc::new(GlobalConfig::default());
        Self {
            index: HnswIndex::new(storage, QuantizationMode::None, config),
        }
    }
}

impl<const N: usize> DynHnsw for HnswRawWrapper<N> {
    fn hnsw_insert(&self, vector: &[f64], uuid_str: &str) -> Result<u32, String> {
        // No L2 normalization — raw vectors preserved for true Euclidean distance.
        let mut meta = HashMap::new();
        meta.insert("nid".to_string(), uuid_str.to_string());
        self.index.insert(vector, meta)
    }

    fn hnsw_search(&self, query: &[f64], k: usize) -> Vec<(u32, f64)> {
        self.index.search(
            query,
            k,
            100,
            &HashMap::new(),
            &[],
            None,
            None,
        )
    }

    fn hnsw_delete(&self, id: u32) {
        self.index.delete(id);
    }

    fn hnsw_get_uuid_str(&self, id: u32) -> Option<String> {
        self.index
            .metadata
            .forward
            .get(&id)
            .and_then(|m| m.get("nid").cloned())
    }

    fn dim(&self) -> usize {
        N
    }
}

// ─── Factory ─────────────────────────────────────────────────────────────────

fn make_cosine_hnsw(dim: usize, storage_dir: &Path) -> Result<Box<dyn DynHnsw>, String> {
    match dim {
        64   => Ok(Box::new(HnswCosineWrapper::<64>::new(storage_dir))),
        128  => Ok(Box::new(HnswCosineWrapper::<128>::new(storage_dir))),
        192  => Ok(Box::new(HnswCosineWrapper::<192>::new(storage_dir))),
        256  => Ok(Box::new(HnswCosineWrapper::<256>::new(storage_dir))),
        384  => Ok(Box::new(HnswCosineWrapper::<384>::new(storage_dir))),
        512  => Ok(Box::new(HnswCosineWrapper::<512>::new(storage_dir))),
        768  => Ok(Box::new(HnswCosineWrapper::<768>::new(storage_dir))),
        1024 => Ok(Box::new(HnswCosineWrapper::<1024>::new(storage_dir))),
        1536 => Ok(Box::new(HnswCosineWrapper::<1536>::new(storage_dir))),
        3072 => Ok(Box::new(HnswCosineWrapper::<3072>::new(storage_dir))),
        n => Err(format!(
            "unsupported vector dimension {n}; \
             supported: 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 3072"
        )),
    }
}

fn make_raw_hnsw(dim: usize, storage_dir: &Path) -> Result<Box<dyn DynHnsw>, String> {
    match dim {
        64   => Ok(Box::new(HnswRawWrapper::<64>::new(storage_dir))),
        128  => Ok(Box::new(HnswRawWrapper::<128>::new(storage_dir))),
        192  => Ok(Box::new(HnswRawWrapper::<192>::new(storage_dir))),
        256  => Ok(Box::new(HnswRawWrapper::<256>::new(storage_dir))),
        384  => Ok(Box::new(HnswRawWrapper::<384>::new(storage_dir))),
        512  => Ok(Box::new(HnswRawWrapper::<512>::new(storage_dir))),
        768  => Ok(Box::new(HnswRawWrapper::<768>::new(storage_dir))),
        1024 => Ok(Box::new(HnswRawWrapper::<1024>::new(storage_dir))),
        1536 => Ok(Box::new(HnswRawWrapper::<1536>::new(storage_dir))),
        3072 => Ok(Box::new(HnswRawWrapper::<3072>::new(storage_dir))),
        n => Err(format!(
            "unsupported vector dimension {n}; \
             supported: 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 3072"
        )),
    }
}

// ─── Public types ─────────────────────────────────────────────────────────────

/// Distance metric for the embedded HNSW vector store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorMetric {
    /// Cosine similarity via L2 on unit-normalized vectors (default).
    /// Recommended for all text/audio/image embeddings (Gemini, Vertex AI, ECAPA-TDNN).
    Cosine,
    /// Squared Euclidean L2 (for raw feature vectors, no normalization).
    Euclidean,
    /// Legacy Poincaré ball distance (for hyperbolic knowledge graph embeddings).
    PoincareBall,
}

/// Real HNSW-backed `VectorStore` that replaces `MockVectorStore` in production.
///
/// Uses `CosineMetric` from `hyperspace-index` (squared L2 on unit-normalized vectors).
/// Supports runtime-configurable dimensions via env vars.
///
/// ## Storage
/// The HNSW backing store writes raw vector data to `{data_dir}/hnsw/chunk_N.hyp`
/// files (mmap backend) or keeps everything in RAM (ram backend).
///
/// ## Thread safety
/// Internally uses `DashMap` and `HnswIndex` (which uses `RwLock` + atomics),
/// so `upsert`/`delete`/`knn` are safe under the outer `Mutex<NietzscheDB<V>>`.
pub struct EmbeddedVectorStore {
    inner: Box<dyn DynHnsw>,
    /// UUID → HNSW internal id, for O(1) soft-delete without scanning forward index.
    uuid_to_hnsw: DashMap<Uuid, u32>,
    metric: VectorMetric,
    dim: usize,
}

impl EmbeddedVectorStore {
    /// Create from environment variables, storing data under `data_dir/hnsw/`:
    /// - `NIETZSCHE_VECTOR_DIM`    (default: `3072`)
    /// - `NIETZSCHE_VECTOR_METRIC` (default: `"cosine"`)
    pub fn from_env(data_dir: &Path) -> Result<Self, String> {
        let dim: usize = std::env::var("NIETZSCHE_VECTOR_DIM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3072);

        let metric = match std::env::var("NIETZSCHE_VECTOR_METRIC")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "euclidean" | "l2" => VectorMetric::Euclidean,
            "poincare" | "hyperbolic" => VectorMetric::PoincareBall,
            _ => VectorMetric::Cosine,
        };

        Self::new(data_dir, dim, metric)
    }

    /// Create with explicit dimension and metric.
    /// Vector data is persisted under `data_dir/hnsw/`.
    ///
    /// # Known limitation — metric is advisory only
    /// The underlying HNSW index always uses `CosineMetric` (L2 on unit-normalised
    /// vectors) regardless of the requested `metric`.  Euclidean and Poincaré metrics
    /// are **not** yet backed by dedicated HNSW wrappers.  For `Euclidean`, vectors
    /// are L2-normalised before insertion (changing their semantics).  For
    /// `PoincareBall`, cosine distance is used instead of the true Poincaré metric.
    ///
    /// This is tracked as `BUG-EVS-001`.  The fix requires adding
    /// `HnswEuclideanWrapper<N>` and `HnswPoincareWrapper<N>` variants to
    /// `make_*_hnsw()` and updating `DynHnsw::hnsw_insert` to skip normalisation
    /// for Euclidean data.
    pub fn new(data_dir: &Path, dim: usize, metric: VectorMetric) -> Result<Self, String> {
        // Keep HNSW files in a dedicated sub-directory to avoid collisions with RocksDB.
        let storage_dir: PathBuf = data_dir.join("hnsw");
        std::fs::create_dir_all(&storage_dir)
            .map_err(|e| format!("cannot create HNSW storage dir {}: {e}", storage_dir.display()))?;

        // BUG-EVS-001 fix: route Euclidean and PoincaréBall to the raw wrapper
        // that does NOT L2-normalize vectors before insertion.
        let inner = match metric {
            VectorMetric::Cosine => make_cosine_hnsw(dim, &storage_dir)?,
            VectorMetric::Euclidean | VectorMetric::PoincareBall => {
                make_raw_hnsw(dim, &storage_dir)?
            }
        };
        Ok(Self {
            inner,
            uuid_to_hnsw: DashMap::new(),
            metric,
            dim,
        })
    }

    /// Distance metric this store was configured with.
    pub fn metric(&self) -> VectorMetric {
        self.metric
    }

    /// Vector dimension this store was configured with.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl VectorStore for EmbeddedVectorStore {
    /// Upsert a vector embedding.
    ///
    /// If the UUID already has an HNSW slot, the previous entry is soft-deleted
    /// before inserting the new one. The HNSW slot counter monotonically increases;
    /// compaction (if ever needed) would require an index rebuild.
    fn upsert(&mut self, id: Uuid, vector: &PoincareVector) -> Result<(), GraphError> {
        // Soft-delete previous HNSW entry for this UUID (if any).
        if let Some(old_id) = self.uuid_to_hnsw.get(&id) {
            self.inner.hnsw_delete(*old_id);
        }

        let hnsw_id = self
            .inner
            .hnsw_insert(&vector.coords, &id.to_string())
            .map_err(|e| GraphError::Storage(format!("HNSW upsert: {e}")))?;

        self.uuid_to_hnsw.insert(id, hnsw_id);
        Ok(())
    }

    fn delete(&mut self, id: Uuid) -> Result<(), GraphError> {
        if let Some((_, hnsw_id)) = self.uuid_to_hnsw.remove(&id) {
            self.inner.hnsw_delete(hnsw_id);
        }
        Ok(())
    }

    fn knn(&self, query: &PoincareVector, k: usize) -> Result<Vec<(Uuid, f64)>, GraphError> {
        let raw = self.inner.hnsw_search(&query.coords, k);

        // Convert HNSW internal ids back to UUIDs via the forward metadata index.
        let results: Vec<(Uuid, f64)> = raw
            .into_iter()
            .filter_map(|(hnsw_id, dist)| {
                self.inner
                    .hnsw_get_uuid_str(hnsw_id)
                    .and_then(|s| Uuid::parse_str(&s).ok())
                    .map(|uuid| (uuid, dist))
            })
            .collect();

        Ok(results)
    }
}

// ─── AnyVectorStore (enum dispatch for main.rs) ───────────────────────────────

/// Runtime-selectable vector store backend.
///
/// Allows the server to pick between the real HNSW (`Embedded`) and the
/// in-memory linear scan (`Mock`) without compile-time branching.
/// Controlled by `NIETZSCHE_VECTOR_BACKEND` env var:
/// - `"embedded"` or `"hnsw"` → `EmbeddedVectorStore`
/// - anything else (or unset) → `MockVectorStore` (legacy default)
pub enum AnyVectorStore {
    Embedded(EmbeddedVectorStore),
    Mock(crate::db::MockVectorStore),
}

impl AnyVectorStore {
    /// Build from env, storing HNSW data under `data_dir/hnsw/`:
    /// - `NIETZSCHE_VECTOR_BACKEND=embedded` → real HNSW (reads DIM + METRIC vars)
    /// - default → MockVectorStore (for development / backward compat)
    pub fn from_env(data_dir: &Path) -> Self {
        match std::env::var("NIETZSCHE_VECTOR_BACKEND")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "embedded" | "hnsw" | "cosine" => {
                match EmbeddedVectorStore::from_env(data_dir) {
                    Ok(vs) => AnyVectorStore::Embedded(vs),
                    Err(e) => {
                        eprintln!(
                            "[nietzsche] WARN: EmbeddedVectorStore init failed ({e}); \
                             falling back to MockVectorStore"
                        );
                        AnyVectorStore::Mock(crate::db::MockVectorStore::default())
                    }
                }
            }
            _ => AnyVectorStore::Mock(crate::db::MockVectorStore::default()),
        }
    }

    /// Build for a named collection with an explicit `dim` and `metric`.
    ///
    /// Used by [`CollectionManager`] so each collection can have its own
    /// dimension without depending on `NIETZSCHE_VECTOR_DIM`.
    ///
    /// Respects `NIETZSCHE_VECTOR_BACKEND`:
    /// - `"embedded"` | `"hnsw"` | `"cosine"` → `EmbeddedVectorStore`
    /// - anything else / unset → `MockVectorStore`
    pub fn for_collection(
        data_dir: &Path,
        dim: usize,
        metric: VectorMetric,
    ) -> Result<Self, String> {
        match std::env::var("NIETZSCHE_VECTOR_BACKEND")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "embedded" | "hnsw" | "cosine" => {
                match EmbeddedVectorStore::new(data_dir, dim, metric) {
                    Ok(vs) => Ok(AnyVectorStore::Embedded(vs)),
                    Err(e) => {
                        eprintln!(
                            "[nietzsche] WARN: EmbeddedVectorStore(dim={dim}) failed ({e}); \
                             falling back to MockVectorStore"
                        );
                        Ok(AnyVectorStore::Mock(crate::db::MockVectorStore::default()))
                    }
                }
            }
            _ => Ok(AnyVectorStore::Mock(crate::db::MockVectorStore::default())),
        }
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Embedded(_) => "EmbeddedHnsw(Cosine)",
            Self::Mock(_)     => "MockVectorStore(LinearScan)",
        }
    }
}

impl VectorStore for AnyVectorStore {
    fn upsert(&mut self, id: Uuid, vector: &PoincareVector) -> Result<(), GraphError> {
        match self {
            Self::Embedded(s) => s.upsert(id, vector),
            Self::Mock(s)     => s.upsert(id, vector),
        }
    }

    fn delete(&mut self, id: Uuid) -> Result<(), GraphError> {
        match self {
            Self::Embedded(s) => s.delete(id),
            Self::Mock(s)     => s.delete(id),
        }
    }

    fn knn(&self, query: &PoincareVector, k: usize) -> Result<Vec<(Uuid, f64)>, GraphError> {
        match self {
            Self::Embedded(s) => s.knn(query, k),
            Self::Mock(s)     => s.knn(query, k),
        }
    }
}
