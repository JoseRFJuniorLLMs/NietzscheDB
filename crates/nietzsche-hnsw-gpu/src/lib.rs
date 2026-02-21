//! # nietzsche-hnsw-gpu
//!
//! GPU-accelerated vector store for NietzscheDB using NVIDIA **cuVS CAGRA**.
//!
//! CAGRA (Compressed-Adjacency Graph Retrieval Algorithm) builds the ANN index
//! on the GPU — up to **10× faster** than CPU HNSW construction — and searches
//! with state-of-the-art GPU throughput (millions of QPS on large datasets).
//!
//! ## Features
//! - `cuda` *(default: off)* — enables GPU via NVIDIA cuVS. Requires CUDA 12.x,
//!   cuVS 24.6, and `libclang` installed on the host machine.
//!
//! ## Build with GPU
//! ```bash
//! cargo build -p nietzsche-hnsw-gpu --features cuda
//! ```
//!
//! ## Drop-in usage
//! ```rust,no_run
//! use nietzsche_hnsw_gpu::GpuVectorStore;
//! use nietzsche_graph::embedded_vector_store::AnyVectorStore;
//!
//! let gpu = GpuVectorStore::new(1024).expect("CUDA init failed");
//! let vs  = AnyVectorStore::gpu(Box::new(gpu));
//! ```
//!
//! ## Architecture
//! ```text
//! Insert → CPU staging buffer (Vec<f32>)
//!                │
//!                ├── feature `cuda` OFF  → always CPU linear scan
//!                │
//!                └── feature `cuda` ON
//!                        ├── n < GPU_THRESHOLD  → CPU linear scan (transfer overhead)
//!                        └── n ≥ GPU_THRESHOLD  → CAGRA build on GPU (lazy)
//!                                                  └── CAGRA search on GPU → results
//! ```

use std::collections::HashMap;
use std::sync::Mutex;

use uuid::Uuid;

use nietzsche_graph::db::VectorStore;
use nietzsche_graph::error::GraphError;
use nietzsche_graph::model::PoincareVector;

// ── GPU imports (only when `cuda` feature is enabled) ────────────────────────

#[cfg(feature = "cuda")]
use {
    cuvs::cagra::{Index as CagraIndex, IndexParams, SearchParams},
    cuvs::{ManagedTensor, Resources},
    ndarray::Array2,
};

// ── Tuning constants ──────────────────────────────────────────────────────────

/// Minimum active vectors before GPU index is used (below → CPU linear scan).
const GPU_THRESHOLD: usize = 1_000;

/// Fraction of dirty entries (inserts+deletes) relative to index size
/// that triggers a GPU index rebuild.
const REBUILD_DELTA_RATIO: f64 = 0.10;

// ── Safety: cuVS wraps raw CUDA pointers ─────────────────────────────────────
//
// All access to GpuState is serialised through Mutex<GpuState>.
// The CUDA context is pinned to a single device (the one active when
// Resources::new() is called). Callers must not migrate threads to a
// different GPU between operations.
//
// SAFETY: Mutex guarantees exclusive access; CAGRA is safe for multi-stream
// concurrent use on the same device.
#[cfg(feature = "cuda")]
unsafe impl Send for GpuState {}
#[cfg(feature = "cuda")]
unsafe impl Sync for GpuState {}

// ── Internal state ────────────────────────────────────────────────────────────

struct GpuState {
    /// Staging buffer: row i = Vec<f32> of `dim` coordinates.
    /// Deleted rows are kept (not compacted) until next GPU rebuild.
    vectors: Vec<Vec<f32>>,

    /// UUID → current row index in `vectors`.
    uuid_to_row: HashMap<Uuid, usize>,

    /// Row index → UUID (for decoding CAGRA neighbor indices back to UUIDs).
    row_to_uuid: Vec<Uuid>,

    /// Soft-delete bitfield (true = logically deleted).
    deleted: Vec<bool>,

    /// Count of soft-deleted rows not yet compacted.
    n_deleted: usize,

    /// Inserts + deletes since the last GPU build.
    dirty_count: usize,

    /// CAGRA GPU index — None until first lazy build. Only present when `cuda`.
    #[cfg(feature = "cuda")]
    gpu_index: Option<CagraIndex>,

    /// Rows included in `gpu_index` (post-compaction count).
    gpu_index_size: usize,

    /// cuVS resource handle — one per GPU device, reused across builds.
    #[cfg(feature = "cuda")]
    resources: Resources,
}

impl GpuState {
    fn new() -> Result<Self, String> {
        #[cfg(feature = "cuda")]
        let resources = Resources::new()
            .map_err(|e| format!("cuVS Resources::new() failed: {e:?}"))?;

        Ok(Self {
            vectors: Vec::new(),
            uuid_to_row: HashMap::new(),
            row_to_uuid: Vec::new(),
            deleted: Vec::new(),
            n_deleted: 0,
            dirty_count: 0,
            #[cfg(feature = "cuda")]
            gpu_index: None,
            gpu_index_size: 0,
            #[cfg(feature = "cuda")]
            resources,
        })
    }

    fn n_active(&self) -> usize {
        self.vectors.len() - self.n_deleted
    }

    fn should_rebuild(&self) -> bool {
        let n = self.n_active();
        if n < GPU_THRESHOLD {
            return false;
        }
        #[cfg(feature = "cuda")]
        {
            if self.gpu_index.is_none() {
                return true;
            }
            let ratio = self.dirty_count as f64 / self.gpu_index_size.max(1) as f64;
            return ratio >= REBUILD_DELTA_RATIO;
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    fn reset_empty(&mut self) {
        self.vectors.clear();
        self.uuid_to_row.clear();
        self.row_to_uuid.clear();
        self.deleted.clear();
        self.n_deleted = 0;
        self.dirty_count = 0;
        self.gpu_index_size = 0;
        #[cfg(feature = "cuda")]
        {
            self.gpu_index = None;
        }
    }

    /// Build (or rebuild) the CAGRA GPU index. Compacts deleted rows first.
    #[cfg(feature = "cuda")]
    fn build_gpu_index(&mut self, dim: usize) -> Result<(), String> {
        let active: Vec<(usize, &Vec<f32>, Uuid)> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.deleted[*i])
            .map(|(i, v)| (i, v, self.row_to_uuid[i]))
            .collect();

        let n = active.len();
        if n == 0 {
            self.reset_empty();
            return Ok(());
        }

        // Build flat f32 matrix [n × dim] for cuVS.
        let mut flat = vec![0f32; n * dim];
        for (new_idx, (_, v, _)) in active.iter().enumerate() {
            flat[new_idx * dim..(new_idx + 1) * dim].copy_from_slice(v);
        }
        let dataset = Array2::from_shape_vec((n, dim), flat)
            .map_err(|e| format!("ndarray shape error: {e}"))?;

        let build_params = IndexParams::new()
            .map_err(|e| format!("IndexParams::new() failed: {e:?}"))?;
        let index = CagraIndex::build(&self.resources, &build_params, &dataset)
            .map_err(|e| format!("CAGRA::build() failed: {e:?}"))?;

        // Compact: rebuild all CPU-side mappings with contiguous post-build indices.
        let new_vectors: Vec<Vec<f32>> = active.iter().map(|(_, v, _)| (*v).clone()).collect();
        let new_row_to_uuid: Vec<Uuid> = active.iter().map(|(_, _, uuid)| *uuid).collect();
        let mut new_uuid_to_row: HashMap<Uuid, usize> = HashMap::with_capacity(n);
        for (new_idx, &uuid) in new_row_to_uuid.iter().enumerate() {
            new_uuid_to_row.insert(uuid, new_idx);
        }

        self.vectors = new_vectors;
        self.uuid_to_row = new_uuid_to_row;
        self.row_to_uuid = new_row_to_uuid;
        self.deleted = vec![false; n];
        self.n_deleted = 0;
        self.dirty_count = 0;
        self.gpu_index = Some(index);
        self.gpu_index_size = n;

        Ok(())
    }

    /// GPU CAGRA search — returns `(row_idx, distance)` pairs.
    #[cfg(feature = "cuda")]
    fn gpu_search(&self, query: &[f32], k: usize, dim: usize) -> Result<Vec<(usize, f32)>, String> {
        let index = self
            .gpu_index
            .as_ref()
            .ok_or_else(|| "GPU index not built".to_string())?;

        let actual_k = k.min(self.gpu_index_size);
        if actual_k == 0 {
            return Ok(vec![]);
        }

        let query_arr = Array2::from_shape_vec((1, dim), query.to_vec())
            .map_err(|e| format!("query shape error: {e}"))?;

        let mut neighbors_host = Array2::<u32>::zeros((1, actual_k));
        let mut distances_host = Array2::<f32>::zeros((1, actual_k));

        let queries_dev = ManagedTensor::from(&query_arr)
            .to_device(&self.resources)
            .map_err(|e| format!("to_device(query): {e:?}"))?;
        let neighbors_dev = ManagedTensor::from(&neighbors_host)
            .to_device(&self.resources)
            .map_err(|e| format!("to_device(neighbors): {e:?}"))?;
        let distances_dev = ManagedTensor::from(&distances_host)
            .to_device(&self.resources)
            .map_err(|e| format!("to_device(distances): {e:?}"))?;

        let search_params = SearchParams::new()
            .map_err(|e| format!("SearchParams::new(): {e:?}"))?;

        index
            .search(
                &self.resources,
                &search_params,
                &queries_dev,
                &neighbors_dev,
                &distances_dev,
            )
            .map_err(|e| format!("CAGRA::search(): {e:?}"))?;

        distances_dev
            .to_host(&self.resources, &mut distances_host)
            .map_err(|e| format!("to_host(distances): {e:?}"))?;
        neighbors_dev
            .to_host(&self.resources, &mut neighbors_host)
            .map_err(|e| format!("to_host(neighbors): {e:?}"))?;

        Ok(neighbors_host
            .row(0)
            .iter()
            .zip(distances_host.row(0).iter())
            .map(|(&idx, &dist)| (idx as usize, dist))
            .collect())
    }

    /// CPU linear scan — O(n·d). Used when `cuda` feature is off or as fallback.
    fn cpu_search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.deleted[*i])
            .map(|(i, v)| {
                let dist: f32 = v
                    .iter()
                    .zip(query.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                (i, dist)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// GPU-accelerated vector store using NVIDIA cuVS CAGRA.
///
/// Implements [`VectorStore`] — drop-in for [`EmbeddedVectorStore`].
/// Inject into NietzscheDB via [`AnyVectorStore::gpu`].
///
/// Without the `cuda` feature the store still works correctly using CPU
/// linear scan (useful for development without a GPU).
pub struct GpuVectorStore {
    dim: usize,
    state: Mutex<GpuState>,
}

impl GpuVectorStore {
    /// Create a new GPU vector store for vectors of the given `dim`ension.
    ///
    /// With `cuda` feature: initialises the CUDA device and cuVS resources.
    /// Fails if no NVIDIA GPU is available or cuVS is not installed.
    ///
    /// Without `cuda` feature: always succeeds, uses CPU linear scan.
    pub fn new(dim: usize) -> Result<Self, String> {
        Ok(Self {
            dim,
            state: Mutex::new(GpuState::new()?),
        })
    }

    /// Explicitly trigger a GPU index build from the current staging buffer.
    ///
    /// Call before a batch of `knn` searches to ensure the GPU index is warm.
    /// Otherwise the index is built lazily on the first qualifying `knn` call.
    ///
    /// No-op when the `cuda` feature is disabled.
    pub fn build_gpu_index(&self) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        {
            let mut state = self.state.lock().unwrap();
            return state.build_gpu_index(self.dim);
        }
        #[cfg(not(feature = "cuda"))]
        Ok(())
    }

    /// Number of active (non-deleted) vectors.
    pub fn len(&self) -> usize {
        self.state.lock().unwrap().n_active()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Whether the crate was compiled with CUDA / GPU support.
    pub fn is_gpu_enabled() -> bool {
        cfg!(feature = "cuda")
    }
}

impl VectorStore for GpuVectorStore {
    fn upsert(&mut self, id: Uuid, vector: &PoincareVector) -> Result<(), GraphError> {
        if vector.coords.len() != self.dim {
            return Err(GraphError::Storage(format!(
                "GpuVectorStore: dim {} ≠ expected {}",
                vector.coords.len(),
                self.dim
            )));
        }

        let coords: Vec<f32> = vector.coords.clone();
        let mut state = self.state.lock().unwrap();

        // Soft-delete any existing entry for this UUID.
        if let Some(&old_row) = state.uuid_to_row.get(&id) {
            if !state.deleted[old_row] {
                state.deleted[old_row] = true;
                state.n_deleted += 1;
                state.dirty_count += 1;
            }
        }

        let new_row = state.vectors.len();
        state.vectors.push(coords);
        state.deleted.push(false);
        state.row_to_uuid.push(id);
        state.uuid_to_row.insert(id, new_row);
        state.dirty_count += 1;

        Ok(())
    }

    fn delete(&mut self, id: Uuid) -> Result<(), GraphError> {
        let mut state = self.state.lock().unwrap();
        if let Some(&row) = state.uuid_to_row.get(&id) {
            if !state.deleted[row] {
                state.deleted[row] = true;
                state.n_deleted += 1;
                state.dirty_count += 1;
            }
            state.uuid_to_row.remove(&id);
        }
        Ok(())
    }

    fn knn(&self, query: &PoincareVector, k: usize) -> Result<Vec<(Uuid, f64)>, GraphError> {
        let query_f32: Vec<f32> = query.coords.clone();
        let mut state = self.state.lock().unwrap();

        // Lazy GPU rebuild when dirty ratio exceeds threshold.
        if state.should_rebuild() {
            #[cfg(feature = "cuda")]
            if let Err(e) = state.build_gpu_index(self.dim) {
                eprintln!("[nietzsche-hnsw-gpu] GPU build failed, CPU fallback: {e}");
            }
        }

        // GPU path (cuda feature) or CPU fallback.
        let raw: Vec<(usize, f32)> = {
            #[cfg(feature = "cuda")]
            {
                if state.gpu_index.is_some() {
                    match state.gpu_search(&query_f32, k, self.dim) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!("[nietzsche-hnsw-gpu] GPU search error, CPU fallback: {e}");
                            state.cpu_search(&query_f32, k)
                        }
                    }
                } else {
                    state.cpu_search(&query_f32, k)
                }
            }
            #[cfg(not(feature = "cuda"))]
            state.cpu_search(&query_f32, k)
        };

        Ok(raw
            .into_iter()
            .filter_map(|(row, dist)| {
                state.row_to_uuid.get(row).map(|&uuid| (uuid, dist as f64))
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::db::VectorStore;
    use nietzsche_graph::model::PoincareVector;

    /// Helper: create a PoincareVector from f32 coords.
    fn pv(coords: Vec<f32>) -> PoincareVector {
        PoincareVector::new(coords)
    }

    // ── Basic upsert and knn ─────────────────────────────────────────────────

    #[test]
    fn upsert_and_knn_single_vector() {
        let mut store = GpuVectorStore::new(3).unwrap();
        let id = Uuid::new_v4();
        let vec = pv(vec![0.1, 0.2, 0.3]);

        store.upsert(id, &vec).unwrap();
        assert_eq!(store.len(), 1);

        let results = store.knn(&vec, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
        // Distance to self should be 0
        assert!(results[0].1 < 1e-6, "distance to self should be ~0, got {}", results[0].1);
    }

    #[test]
    fn upsert_and_knn_returns_nearest() {
        let mut store = GpuVectorStore::new(2).unwrap();

        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        let id_c = Uuid::new_v4();

        // A at (0.1, 0.0), B at (0.2, 0.0), C at (0.9, 0.0)
        store.upsert(id_a, &pv(vec![0.1, 0.0])).unwrap();
        store.upsert(id_b, &pv(vec![0.2, 0.0])).unwrap();
        store.upsert(id_c, &pv(vec![0.9, 0.0])).unwrap();

        // Query near A
        let query = pv(vec![0.1, 0.0]);
        let results = store.knn(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // Nearest to (0.1, 0.0) should be A (dist=0), then B (dist=0.1)
        assert_eq!(results[0].0, id_a);
        assert_eq!(results[1].0, id_b);
    }

    #[test]
    fn knn_returns_sorted_by_distance() {
        let mut store = GpuVectorStore::new(2).unwrap();

        let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
        let coords = [
            vec![0.1, 0.0],
            vec![0.3, 0.0],
            vec![0.5, 0.0],
            vec![0.7, 0.0],
            vec![0.9, 0.0],
        ];

        for (i, id) in ids.iter().enumerate() {
            store.upsert(*id, &pv(coords[i].clone())).unwrap();
        }

        let query = pv(vec![0.0, 0.0]); // origin
        let results = store.knn(&query, 5).unwrap();

        // Verify results are sorted ascending by distance
        for i in 1..results.len() {
            assert!(
                results[i].1 >= results[i - 1].1 - 1e-6,
                "results not sorted: dist[{}]={} < dist[{}]={}",
                i,
                results[i].1,
                i - 1,
                results[i - 1].1,
            );
        }
    }

    #[test]
    fn knn_k_greater_than_n() {
        let mut store = GpuVectorStore::new(2).unwrap();
        let id = Uuid::new_v4();
        store.upsert(id, &pv(vec![0.1, 0.1])).unwrap();

        let results = store.knn(&pv(vec![0.0, 0.0]), 10).unwrap();
        // Should return only 1 result even though k=10
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn knn_empty_store() {
        let store = GpuVectorStore::new(2).unwrap();
        let results = store.knn(&pv(vec![0.0, 0.0]), 5).unwrap();
        assert!(results.is_empty());
    }

    // ── Delete ───────────────────────────────────────────────────────────────

    #[test]
    fn delete_marks_as_soft_deleted() {
        let mut store = GpuVectorStore::new(2).unwrap();
        let id = Uuid::new_v4();
        store.upsert(id, &pv(vec![0.1, 0.1])).unwrap();
        assert_eq!(store.len(), 1);

        store.delete(id).unwrap();
        assert_eq!(store.len(), 0);

        // knn should not return the deleted vector
        let results = store.knn(&pv(vec![0.1, 0.1]), 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn delete_nonexistent_is_noop() {
        let mut store = GpuVectorStore::new(2).unwrap();
        let id = Uuid::new_v4();
        // Should not error
        store.delete(id).unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn delete_then_reinsert() {
        let mut store = GpuVectorStore::new(2).unwrap();
        let id = Uuid::new_v4();

        store.upsert(id, &pv(vec![0.1, 0.0])).unwrap();
        store.delete(id).unwrap();
        assert_eq!(store.len(), 0);

        // Re-insert with different coords
        store.upsert(id, &pv(vec![0.5, 0.0])).unwrap();
        assert_eq!(store.len(), 1);

        let results = store.knn(&pv(vec![0.5, 0.0]), 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
        assert!(results[0].1 < 1e-6, "should find the re-inserted vector");
    }

    #[test]
    fn delete_excludes_from_knn_results() {
        let mut store = GpuVectorStore::new(2).unwrap();

        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        let id_c = Uuid::new_v4();

        store.upsert(id_a, &pv(vec![0.1, 0.0])).unwrap();
        store.upsert(id_b, &pv(vec![0.2, 0.0])).unwrap();
        store.upsert(id_c, &pv(vec![0.3, 0.0])).unwrap();

        // Delete B (the middle one)
        store.delete(id_b).unwrap();
        assert_eq!(store.len(), 2);

        let results = store.knn(&pv(vec![0.2, 0.0]), 3).unwrap();
        assert_eq!(results.len(), 2);
        // B should not appear in results
        assert!(
            results.iter().all(|(uuid, _)| *uuid != id_b),
            "deleted ID should not appear in results",
        );
    }

    // ── Upsert overwrites ────────────────────────────────────────────────────

    #[test]
    fn upsert_same_id_overwrites() {
        let mut store = GpuVectorStore::new(2).unwrap();
        let id = Uuid::new_v4();

        store.upsert(id, &pv(vec![0.1, 0.0])).unwrap();
        store.upsert(id, &pv(vec![0.9, 0.0])).unwrap();
        assert_eq!(store.len(), 1);

        // Query near (0.9, 0.0) — should find the updated vector
        let results = store.knn(&pv(vec![0.9, 0.0]), 1).unwrap();
        assert_eq!(results[0].0, id);
        assert!(results[0].1 < 1e-6, "should match the new position");
    }

    // ── Below GPU_THRESHOLD uses linear scan ─────────────────────────────────

    #[test]
    fn below_gpu_threshold_uses_cpu_linear_scan() {
        // Insert fewer than GPU_THRESHOLD vectors (1000) and verify correctness.
        // On CPU fallback (no cuda feature), this is always the path.
        let dim = 4;
        let mut store = GpuVectorStore::new(dim).unwrap();

        let n = 50; // well below GPU_THRESHOLD=1000
        let mut ids = Vec::with_capacity(n);

        for i in 0..n {
            let id = Uuid::new_v4();
            let val = (i as f32 + 1.0) * 0.01; // 0.01..0.50
            store.upsert(id, &pv(vec![val, 0.0, 0.0, 0.0])).unwrap();
            ids.push((id, val));
        }

        assert_eq!(store.len(), n);
        assert!(!GpuVectorStore::is_gpu_enabled(), "cuda feature should be off in test");

        // knn for a query at origin should return closest vectors (smallest coords)
        let query = pv(vec![0.0, 0.0, 0.0, 0.0]);
        let results = store.knn(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // Results should be sorted by distance (ascending)
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-6);
        }
    }

    // ── Dimension mismatch ───────────────────────────────────────────────────

    #[test]
    fn upsert_dimension_mismatch_returns_error() {
        let mut store = GpuVectorStore::new(3).unwrap();
        let id = Uuid::new_v4();
        let wrong_dim = pv(vec![0.1, 0.2]); // 2 != 3
        let result = store.upsert(id, &wrong_dim);
        assert!(result.is_err(), "should error on dimension mismatch");
    }

    // ── Helper methods ───────────────────────────────────────────────────────

    #[test]
    fn is_empty_on_new_store() {
        let store = GpuVectorStore::new(2).unwrap();
        assert!(store.is_empty());
        assert_eq!(store.dim(), 2);
    }

    #[test]
    fn len_tracks_active_count() {
        let mut store = GpuVectorStore::new(2).unwrap();
        assert_eq!(store.len(), 0);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        store.upsert(id1, &pv(vec![0.1, 0.0])).unwrap();
        assert_eq!(store.len(), 1);
        store.upsert(id2, &pv(vec![0.2, 0.0])).unwrap();
        assert_eq!(store.len(), 2);

        store.delete(id1).unwrap();
        assert_eq!(store.len(), 1);
        store.delete(id2).unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn gpu_not_enabled_in_test() {
        assert!(!GpuVectorStore::is_gpu_enabled());
    }

    #[test]
    fn build_gpu_index_is_noop_without_cuda() {
        let store = GpuVectorStore::new(2).unwrap();
        // Should not error
        store.build_gpu_index().unwrap();
    }

    // ── Correctness: knn distances are Euclidean ─────────────────────────────

    #[test]
    fn knn_distance_is_euclidean() {
        let mut store = GpuVectorStore::new(2).unwrap();
        let id = Uuid::new_v4();

        store.upsert(id, &pv(vec![0.3, 0.4])).unwrap();

        // Query from origin: Euclidean distance = sqrt(0.09 + 0.16) = 0.5
        let results = store.knn(&pv(vec![0.0, 0.0]), 1).unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            (results[0].1 - 0.5).abs() < 1e-5,
            "expected distance 0.5, got {}",
            results[0].1,
        );
    }
}
