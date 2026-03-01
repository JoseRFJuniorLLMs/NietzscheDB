//! EmbeddedVectorStore — real HNSW-backed vector store.
//!
//! Replaces `MockVectorStore` in production. Wraps `nietzsche-hnsw`'s
//! `HnswIndex<N, M>` with type erasure so the dimension and metric are
//! chosen at runtime via environment variables.
//!
//! ## Supported metrics
//! - `CosineMetric`   — L2 on unit-normalised vectors (text/audio/image embeddings)
//! - `PoincareMetric` — true hyperbolic distance (NietzscheDB knowledge graph nodes)
//! - `HnswRawWrapper` — raw Euclidean L2 without normalisation
//!
//! ## Environment variables
//! - `NIETZSCHE_VECTOR_DIM`    — vector dimension (default: `3072`)
//! - `NIETZSCHE_VECTOR_METRIC` — `"cosine"` | `"euclidean"` | `"poincare"` (default: `"cosine"`)
//!
//! ## Supported dimensions
//! 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 3072
//! (matches common embedding model outputs used in EVA-Mind)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dashmap::DashMap;
use uuid::Uuid;

use nietzsche_core::{CosineMetric, EuclideanMetric, PoincareMetric, GlobalConfig, QuantizationMode};
use nietzsche_hnsw::HnswIndex;
use nietzsche_vecstore::VectorStore as RawStore;

use crate::db::VectorStore;
use crate::error::GraphError;
use crate::model::PoincareVector;

// ─── Internal type-erased HNSW ───────────────────────────────────────────────

/// Type-erased HNSW operations that erase the `const N: usize` generic.
/// All concrete implementations live in `HnswCosineWrapper<N>`.
trait DynHnsw: Send + Sync {
    /// Insert a (normalized) vector and return the internal HNSW node id.
    fn hnsw_insert(&self, vector: &[f64], uuid_str: &str) -> Result<u32, String>;

    /// Insert a vector with additional metadata fields to index in the HNSW
    /// metadata index (RoaringBitmap inverted + numeric). The `nid` key is
    /// always added automatically — callers provide extra fields like
    /// `idoso_id`, `node_type`, etc.
    fn hnsw_insert_with_meta(
        &self,
        vector: &[f64],
        uuid_str: &str,
        extra_meta: HashMap<String, String>,
    ) -> Result<u32, String>;

    /// Search for `k` nearest neighbors. Returns `(hnsw_id, distance)` pairs.
    fn hnsw_search(&self, query: &[f64], k: usize) -> Vec<(u32, f64)>;

    /// Filtered search: push `filter` and `complex_filters` into the HNSW
    /// layer so RoaringBitmap pre-filtering prunes candidates before distance
    /// computation. Returns `(hnsw_id, distance)` pairs.
    fn hnsw_search_filtered(
        &self,
        query: &[f64],
        k: usize,
        filter: &HashMap<String, String>,
        complex_filters: &[nietzsche_core::FilterExpr],
    ) -> Vec<(u32, f64)>;

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
        let element_size = nietzsche_core::vector::HyperVector::<N>::SIZE;
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
    /// Mirrors the `normalize_if_cosine` logic in `nietzsche-baseserver`.
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
        meta.insert("nid".to_string(), uuid_str.to_string());
        self.index.insert(&normalized, meta)
    }

    fn hnsw_insert_with_meta(
        &self,
        vector: &[f64],
        uuid_str: &str,
        mut extra_meta: HashMap<String, String>,
    ) -> Result<u32, String> {
        let normalized = Self::normalize(vector);
        extra_meta.insert("nid".to_string(), uuid_str.to_string());
        self.index.insert(&normalized, extra_meta)
    }

    fn hnsw_search(&self, query: &[f64], k: usize) -> Vec<(u32, f64)> {
        let normalized = Self::normalize(query);
        let ef = (k * 4).max(16).min(512);
        self.index.search(&normalized, k, ef, &HashMap::new(), &[], None, None)
    }

    fn hnsw_search_filtered(
        &self,
        query: &[f64],
        k: usize,
        filter: &HashMap<String, String>,
        complex_filters: &[nietzsche_core::FilterExpr],
    ) -> Vec<(u32, f64)> {
        let normalized = Self::normalize(query);
        let ef = (k * 4).max(16).min(512);
        self.index.search(&normalized, k, ef, filter, complex_filters, None, None)
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
    index: HnswIndex<N, EuclideanMetric>,
}

impl<const N: usize> HnswRawWrapper<N> {
    fn new(storage_dir: &Path) -> Self {
        let element_size = nietzsche_core::vector::HyperVector::<N>::SIZE;
        let storage = Arc::new(RawStore::new(storage_dir, element_size));
        let config = Arc::new(GlobalConfig::default());
        Self {
            index: HnswIndex::new(storage, QuantizationMode::None, config),
        }
    }
}

impl<const N: usize> DynHnsw for HnswRawWrapper<N> {
    fn hnsw_insert(&self, vector: &[f64], uuid_str: &str) -> Result<u32, String> {
        let mut meta = HashMap::new();
        meta.insert("nid".to_string(), uuid_str.to_string());
        self.index.insert(vector, meta)
    }

    fn hnsw_insert_with_meta(
        &self,
        vector: &[f64],
        uuid_str: &str,
        mut extra_meta: HashMap<String, String>,
    ) -> Result<u32, String> {
        extra_meta.insert("nid".to_string(), uuid_str.to_string());
        self.index.insert(vector, extra_meta)
    }

    fn hnsw_search(&self, query: &[f64], k: usize) -> Vec<(u32, f64)> {
        let ef = (k * 4).max(16).min(512);
        self.index.search(query, k, ef, &HashMap::new(), &[], None, None)
    }

    fn hnsw_search_filtered(
        &self,
        query: &[f64],
        k: usize,
        filter: &HashMap<String, String>,
        complex_filters: &[nietzsche_core::FilterExpr],
    ) -> Vec<(u32, f64)> {
        let ef = (k * 4).max(16).min(512);
        self.index.search(query, k, ef, filter, complex_filters, None, None)
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

// ─── Concrete Poincaré HNSW wrapper ──────────────────────────────────────────
//
// Uses `PoincareMetric` from `nietzsche-core` — the geometrically correct metric
// for NietzscheDB's hyperbolic knowledge graph.
//
// HNSW neighbours are built with d(u,v) = acosh(1 + 2‖u-v‖²/((1-‖u‖²)(1-‖v‖²))),
// so proximity in the index reflects true hyperbolic distance, not cosine angle.
// Vectors must satisfy ‖x‖ < 1.0 (Poincaré ball invariant). No L2 normalisation
// is applied: coordinates are inserted raw, preserving their position in the ball.

struct HnswPoincareWrapper<const N: usize> {
    index: HnswIndex<N, PoincareMetric>,
}

impl<const N: usize> HnswPoincareWrapper<N> {
    fn new(storage_dir: &Path) -> Self {
        let element_size = nietzsche_core::vector::HyperVector::<N>::SIZE;
        let storage = Arc::new(RawStore::new(storage_dir, element_size));
        let config = Arc::new(GlobalConfig::default());
        Self {
            index: HnswIndex::new(storage, QuantizationMode::None, config),
        }
    }
}

impl<const N: usize> DynHnsw for HnswPoincareWrapper<N> {
    fn hnsw_insert(&self, vector: &[f64], uuid_str: &str) -> Result<u32, String> {
        let mut meta = HashMap::new();
        meta.insert("nid".to_string(), uuid_str.to_string());
        self.index.insert(vector, meta)
    }

    fn hnsw_insert_with_meta(
        &self,
        vector: &[f64],
        uuid_str: &str,
        mut extra_meta: HashMap<String, String>,
    ) -> Result<u32, String> {
        extra_meta.insert("nid".to_string(), uuid_str.to_string());
        self.index.insert(vector, extra_meta)
    }

    fn hnsw_search(&self, query: &[f64], k: usize) -> Vec<(u32, f64)> {
        let ef = (k * 4).max(16).min(512);
        self.index.search(query, k, ef, &HashMap::new(), &[], None, None)
    }

    fn hnsw_search_filtered(
        &self,
        query: &[f64],
        k: usize,
        filter: &HashMap<String, String>,
        complex_filters: &[nietzsche_core::FilterExpr],
    ) -> Vec<(u32, f64)> {
        let ef = (k * 4).max(16).min(512);
        self.index.search(query, k, ef, filter, complex_filters, None, None)
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

fn make_poincare_hnsw(dim: usize, storage_dir: &Path) -> Result<Box<dyn DynHnsw>, String> {
    match dim {
        64   => Ok(Box::new(HnswPoincareWrapper::<64>::new(storage_dir))),
        128  => Ok(Box::new(HnswPoincareWrapper::<128>::new(storage_dir))),
        192  => Ok(Box::new(HnswPoincareWrapper::<192>::new(storage_dir))),
        256  => Ok(Box::new(HnswPoincareWrapper::<256>::new(storage_dir))),
        384  => Ok(Box::new(HnswPoincareWrapper::<384>::new(storage_dir))),
        512  => Ok(Box::new(HnswPoincareWrapper::<512>::new(storage_dir))),
        768  => Ok(Box::new(HnswPoincareWrapper::<768>::new(storage_dir))),
        1024 => Ok(Box::new(HnswPoincareWrapper::<1024>::new(storage_dir))),
        1536 => Ok(Box::new(HnswPoincareWrapper::<1536>::new(storage_dir))),
        3072 => Ok(Box::new(HnswPoincareWrapper::<3072>::new(storage_dir))),
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
    /// Dot product similarity for pre-normalized vectors (bi-encoders).
    /// Internally uses the same Cosine wrapper (L2 on unit sphere) since for
    /// already-normalized vectors: L2²(u,v) = 2 − 2·dot(u,v).
    DotProduct,
    /// Squared Euclidean L2 (for raw feature vectors, no normalization).
    Euclidean,
    /// Legacy Poincaré ball distance (for hyperbolic knowledge graph embeddings).
    PoincareBall,
}

/// Real HNSW-backed `VectorStore` that replaces `MockVectorStore` in production.
///
/// Uses `CosineMetric` from `nietzsche-hnsw` (squared L2 on unit-normalized vectors).
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
            "dotproduct" | "dot" | "inner_product" => VectorMetric::DotProduct,
            _ => VectorMetric::Cosine,
        };

        Self::new(data_dir, dim, metric)
    }

    /// Create with explicit dimension and metric.
    /// Vector data is persisted under `data_dir/hnsw/`.
    ///
    /// Each metric routes to a dedicated HNSW wrapper:
    /// - `Cosine`      → `HnswCosineWrapper`   (L2 on unit-normalised vectors)
    /// - `Euclidean`   → `HnswRawWrapper`       (raw L2, no normalisation)
    /// - `PoincareBall`→ `HnswPoincareWrapper`  (true hyperbolic distance via acosh)
    pub fn new(data_dir: &Path, dim: usize, metric: VectorMetric) -> Result<Self, String> {
        // Keep HNSW files in a dedicated sub-directory to avoid collisions with RocksDB.
        let storage_dir: PathBuf = data_dir.join("hnsw");
        std::fs::create_dir_all(&storage_dir)
            .map_err(|e| format!("cannot create HNSW storage dir {}: {e}", storage_dir.display()))?;

        let inner = match metric {
            VectorMetric::Cosine | VectorMetric::DotProduct => make_cosine_hnsw(dim, &storage_dir)?,
            VectorMetric::Euclidean   => make_raw_hnsw(dim, &storage_dir)?,
            VectorMetric::PoincareBall => make_poincare_hnsw(dim, &storage_dir)?,
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

/// Convert a [`MetadataFilter`] into the HNSW-level filter types:
/// `(legacy_tags, complex_filters)`.
fn metadata_filter_to_hnsw(
    filter: &crate::db::MetadataFilter,
) -> (HashMap<String, String>, Vec<nietzsche_core::FilterExpr>) {
    let mut tags = HashMap::new();
    let mut exprs = Vec::new();
    collect_filter(filter, &mut tags, &mut exprs);
    (tags, exprs)
}

fn collect_filter(
    filter: &crate::db::MetadataFilter,
    tags: &mut HashMap<String, String>,
    exprs: &mut Vec<nietzsche_core::FilterExpr>,
) {
    match filter {
        crate::db::MetadataFilter::Eq { field, value } => {
            // Use the complex filter Match for exact equality (inverted index lookup).
            exprs.push(nietzsche_core::FilterExpr::Match {
                key: field.clone(),
                value: value.clone(),
            });
        }
        crate::db::MetadataFilter::In { field, values } => {
            // HNSW has no native IN; emit one Match per value.
            // The HNSW search intersects all masks, so we'd need OR semantics.
            // Workaround: use legacy tag filter for the first value (best effort).
            // For full IN support, post-filter is needed.
            if let Some(first) = values.first() {
                tags.insert(field.clone(), first.clone());
            }
        }
        crate::db::MetadataFilter::Range { field, gte, lte } => {
            exprs.push(nietzsche_core::FilterExpr::Range {
                key: field.clone(),
                gte: *gte,
                lte: *lte,
            });
        }
        crate::db::MetadataFilter::And(subs) => {
            let mut sub_exprs = Vec::new();
            for sub in subs {
                collect_filter(sub, tags, &mut sub_exprs);
            }
            if !sub_exprs.is_empty() {
                exprs.push(nietzsche_core::FilterExpr::And(sub_exprs));
            }
        }
        crate::db::MetadataFilter::Or(subs) => {
            let mut sub_exprs = Vec::new();
            for sub in subs {
                collect_filter(sub, tags, &mut sub_exprs);
            }
            if !sub_exprs.is_empty() {
                exprs.push(nietzsche_core::FilterExpr::Or(sub_exprs));
            }
        }
        crate::db::MetadataFilter::Not(sub) => {
            let mut sub_exprs = Vec::new();
            collect_filter(sub, tags, &mut sub_exprs);
            if let Some(expr) = sub_exprs.pop() {
                exprs.push(nietzsche_core::FilterExpr::Not(Box::new(expr)));
            }
        }
        crate::db::MetadataFilter::Contains { field, value } => {
            exprs.push(nietzsche_core::FilterExpr::Contains {
                key: field.clone(),
                value: value.clone(),
            });
        }
        crate::db::MetadataFilter::Exists { field } => {
            exprs.push(nietzsche_core::FilterExpr::Exists {
                key: field.clone(),
            });
        }
        crate::db::MetadataFilter::None => {}
    }
}

impl VectorStore for EmbeddedVectorStore {
    fn upsert(&mut self, id: Uuid, vector: &PoincareVector) -> Result<(), GraphError> {
        if let Some(old_id) = self.uuid_to_hnsw.get(&id) {
            self.inner.hnsw_delete(*old_id);
        }

        let coords_f64 = vector.coords_f64();
        let hnsw_id = self
            .inner
            .hnsw_insert(&coords_f64, &id.to_string())
            .map_err(|e| GraphError::Storage(format!("HNSW upsert: {e}")))?;

        self.uuid_to_hnsw.insert(id, hnsw_id);
        Ok(())
    }

    fn upsert_with_meta(
        &mut self,
        id: Uuid,
        vector: &PoincareVector,
        meta: HashMap<String, String>,
    ) -> Result<(), GraphError> {
        if let Some(old_id) = self.uuid_to_hnsw.get(&id) {
            self.inner.hnsw_delete(*old_id);
        }

        let coords_f64 = vector.coords_f64();
        let hnsw_id = self
            .inner
            .hnsw_insert_with_meta(&coords_f64, &id.to_string(), meta)
            .map_err(|e| GraphError::Storage(format!("HNSW upsert_with_meta: {e}")))?;

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
        let coords_f64 = query.coords_f64();
        let raw = self.inner.hnsw_search(&coords_f64, k);
        Ok(self.hnsw_results_to_uuids(raw))
    }

    fn knn_filtered(
        &self,
        query: &PoincareVector,
        k: usize,
        filter: &crate::db::MetadataFilter,
    ) -> Result<Vec<(Uuid, f64)>, GraphError> {
        if matches!(filter, crate::db::MetadataFilter::None) {
            return self.knn(query, k);
        }

        let (tags, exprs) = metadata_filter_to_hnsw(filter);
        let coords_f64 = query.coords_f64();
        let raw = self.inner.hnsw_search_filtered(&coords_f64, k, &tags, &exprs);
        Ok(self.hnsw_results_to_uuids(raw))
    }
}

impl EmbeddedVectorStore {
    /// Convert HNSW (id, dist) pairs back to (Uuid, dist).
    fn hnsw_results_to_uuids(&self, raw: Vec<(u32, f64)>) -> Vec<(Uuid, f64)> {
        raw.into_iter()
            .filter_map(|(hnsw_id, dist)| {
                self.inner
                    .hnsw_get_uuid_str(hnsw_id)
                    .and_then(|s| Uuid::parse_str(&s).ok())
                    .map(|uuid| (uuid, dist))
            })
            .collect()
    }
}

// ─── AnyVectorStore (enum dispatch for main.rs) ───────────────────────────────

/// Runtime-selectable vector store backend.
///
/// Allows the server to pick between the real HNSW (`Embedded`) and the
/// in-memory linear scan (`Mock`) without compile-time branching.
/// Controlled by `NIETZSCHE_VECTOR_BACKEND` env var:
/// - default (unset / `"embedded"` / `"hnsw"`) → `EmbeddedVectorStore` (real HNSW)
/// - `"mock"` or `"linear"`  → `MockVectorStore` (linear scan, tests only)
/// - `"gpu"`                 → GPU backend (inject via [`AnyVectorStore::gpu`])
/// - `"tpu"`                 → TPU backend (inject via [`AnyVectorStore::tpu`])
pub enum AnyVectorStore {
    Embedded(EmbeddedVectorStore),
    Mock(crate::db::MockVectorStore),
    /// GPU-accelerated backend (e.g. nietzsche-hnsw-gpu with NVIDIA cuVS CAGRA).
    /// Injected at server startup — nietzsche-graph itself does not depend on CUDA.
    Gpu(Box<dyn crate::db::VectorStore>),
    /// Google TPU-accelerated backend via PJRT C API (MHLO dot-product kernel).
    /// Runs on Cloud TPU VMs (v5e / v6e Trillium / v7 Ironwood).
    /// Injected at server startup — nietzsche-graph itself does not depend on pjrt.
    Tpu(Box<dyn crate::db::VectorStore>),
}

impl AnyVectorStore {
    /// Build from env, storing HNSW data under `data_dir/hnsw/`:
    /// - `NIETZSCHE_VECTOR_BACKEND=mock` → `MockVectorStore` (linear scan, for tests only)
    /// - `NIETZSCHE_VECTOR_BACKEND=gpu`  → inject GPU store via [`AnyVectorStore::gpu`]
    /// - `NIETZSCHE_VECTOR_BACKEND=tpu`  → inject TPU store via [`AnyVectorStore::tpu`]
    /// - default (unset or `"embedded"` / `"hnsw"`) → real `EmbeddedVectorStore` (HNSW)
    pub fn from_env(data_dir: &Path) -> Self {
        match std::env::var("NIETZSCHE_VECTOR_BACKEND")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "mock" | "linear" => AnyVectorStore::Mock(crate::db::MockVectorStore::default()),
            _ => {
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
        }
    }

    /// Build for a named collection with an explicit `dim` and `metric`.
    ///
    /// Used by [`CollectionManager`] so each collection can have its own
    /// dimension without depending on `NIETZSCHE_VECTOR_DIM`.
    ///
    /// Respects `NIETZSCHE_VECTOR_BACKEND`:
    /// - `"mock"` | `"linear"` → `MockVectorStore` (tests only)
    /// - default (unset or `"embedded"` / `"hnsw"`) → real `EmbeddedVectorStore`
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
            "mock" | "linear" => Ok(AnyVectorStore::Mock(crate::db::MockVectorStore::default())),
            _ => {
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
        }
    }

    /// Inject a GPU-accelerated [`VectorStore`] implementation.
    ///
    /// ```rust,no_run
    /// use nietzsche_hnsw_gpu::GpuVectorStore;
    /// use nietzsche_graph::embedded_vector_store::AnyVectorStore;
    ///
    /// let gpu = GpuVectorStore::new(1024).expect("CUDA init failed");
    /// let vs  = AnyVectorStore::gpu(Box::new(gpu));
    /// ```
    pub fn gpu(store: Box<dyn crate::db::VectorStore>) -> Self {
        Self::Gpu(store)
    }

    /// Inject a Google TPU-accelerated [`VectorStore`] implementation.
    ///
    /// ```rust,no_run
    /// use nietzsche_tpu::TpuVectorStore;
    /// use nietzsche_graph::embedded_vector_store::AnyVectorStore;
    ///
    /// let tpu = TpuVectorStore::new(1536).expect("PJRT init failed");
    /// let vs  = AnyVectorStore::tpu(Box::new(tpu));
    /// ```
    pub fn tpu(store: Box<dyn crate::db::VectorStore>) -> Self {
        Self::Tpu(store)
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Embedded(s) => match s.metric() {
                VectorMetric::Cosine       => "EmbeddedHnsw(Cosine)",
                VectorMetric::DotProduct   => "EmbeddedHnsw(DotProduct)",
                VectorMetric::Euclidean    => "EmbeddedHnsw(Euclidean)",
                VectorMetric::PoincareBall => "EmbeddedHnsw(Poincaré)",
            },
            Self::Mock(_) => "MockVectorStore(LinearScan)",
            Self::Gpu(_)  => "GpuVectorStore(CAGRA/cuVS)",
            Self::Tpu(_)  => "TpuVectorStore(PJRT/MHLO)",
        }
    }
}

impl VectorStore for AnyVectorStore {
    fn upsert(&mut self, id: Uuid, vector: &PoincareVector) -> Result<(), GraphError> {
        match self {
            Self::Embedded(s) => s.upsert(id, vector),
            Self::Mock(s)     => s.upsert(id, vector),
            Self::Gpu(s)      => s.upsert(id, vector),
            Self::Tpu(s)      => s.upsert(id, vector),
        }
    }

    fn upsert_with_meta(
        &mut self,
        id: Uuid,
        vector: &PoincareVector,
        meta: HashMap<String, String>,
    ) -> Result<(), GraphError> {
        match self {
            Self::Embedded(s) => s.upsert_with_meta(id, vector, meta),
            Self::Mock(s)     => s.upsert_with_meta(id, vector, meta),
            Self::Gpu(s)      => s.upsert_with_meta(id, vector, meta),
            Self::Tpu(s)      => s.upsert_with_meta(id, vector, meta),
        }
    }

    fn delete(&mut self, id: Uuid) -> Result<(), GraphError> {
        match self {
            Self::Embedded(s) => s.delete(id),
            Self::Mock(s)     => s.delete(id),
            Self::Gpu(s)      => s.delete(id),
            Self::Tpu(s)      => s.delete(id),
        }
    }

    fn knn(&self, query: &PoincareVector, k: usize) -> Result<Vec<(Uuid, f64)>, GraphError> {
        match self {
            Self::Embedded(s) => s.knn(query, k),
            Self::Mock(s)     => s.knn(query, k),
            Self::Gpu(s)      => s.knn(query, k),
            Self::Tpu(s)      => s.knn(query, k),
        }
    }

    fn knn_filtered(
        &self,
        query: &PoincareVector,
        k: usize,
        filter: &crate::db::MetadataFilter,
    ) -> Result<Vec<(Uuid, f64)>, GraphError> {
        match self {
            Self::Embedded(s) => s.knn_filtered(query, k, filter),
            Self::Mock(s)     => s.knn_filtered(query, k, filter),
            Self::Gpu(s)      => s.knn_filtered(query, k, filter),
            Self::Tpu(s)      => s.knn_filtered(query, k, filter),
        }
    }
}
