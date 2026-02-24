//! GPU-accelerated graph traversal for NietzscheDB via NVIDIA cuGraph (RAPIDS).
//!
//! Provides:
//! - **BFS** — breadth-first traversal on GPU (cuGraph)
//! - **Dijkstra / SSSP** — weighted shortest paths on GPU (cuGraph)
//! - **PageRank** — centrality scoring on GPU (cuGraph)
//! - **Poincaré kNN** — hyperbolic distance brute-force kNN on GPU (NVRTC kernel)
//!
//! # Feature flags
//! - `cuda` — enables GPU execution via cuGraph + NVRTC.
//!   Without this, all operations fall back to CPU implementations.
//!
//! # Requirements (`cuda` feature)
//! - CUDA 12.x
//! - RAPIDS cuGraph 24.6 (`libcugraph.so` on `LD_LIBRARY_PATH`)
//! - NVRTC (included with CUDA toolkit)
//!
//! # GCP setup
//! ```bash
//! # Install RAPIDS (includes cuGraph + cuVS)
//! pip install --extra-index-url=https://pypi.nvidia.com \
//!     cudf-cu12==24.6.* cugraph-cu12==24.6.*
//!
//! # Or via conda
//! conda install -c rapidsai -c conda-forge cugraph=24.6 cuda-version=12.4
//! ```
//!
//! # Example
//! ```rust,no_run
//! use nietzsche_cugraph::CuGraphIndex;
//!
//! // Build from live adjacency
//! let index = CuGraphIndex::from_adjacency(&adjacency).unwrap();
//!
//! // GPU BFS (falls back to CPU without `cuda` feature)
//! let bfs = index.bfs(start_id, 6).unwrap();
//! println!("Visited {} nodes", bfs.visited.len());
//!
//! // GPU PageRank
//! let pr = index.pagerank(0.85, 100, 1e-6).unwrap();
//! ```

pub mod csr;
pub mod error;
pub mod traversal;

#[cfg(feature = "cuda")]
pub mod gpu;

pub use error::CuGraphError;
pub use traversal::{BfsResult, DijkstraResult, PageRankResult};

use nietzsche_graph::AdjacencyIndex;
use uuid::Uuid;

// ── CuGraphIndex ──────────────────────────────────────────────────────────────

/// GPU graph index derived from a NietzscheDB adjacency.
///
/// Converting the adjacency to CSR is O(E).  With the `cuda` feature, the CSR
/// is kept in host memory and uploaded to the GPU per-operation (cuGraph
/// manages GPU memory internally).
pub struct CuGraphIndex {
    pub(crate) csr: csr::Csr,
}

impl CuGraphIndex {
    /// Build the index from a live `AdjacencyIndex`.
    ///
    /// This is a pure CPU operation — it snapshots the adjacency and builds
    /// the CSR representation. Safe to call while the DB holds a read lock.
    pub fn from_adjacency(adjacency: &AdjacencyIndex) -> Result<Self, CuGraphError> {
        let csr = csr::build(adjacency)?;
        Ok(Self { csr })
    }

    /// GPU (or CPU fallback) BFS from `start`, up to `max_depth` hops.
    ///
    /// With `cuda` feature: runs via cuGraph on the L4 GPU.
    /// Without `cuda`: runs a standard CPU BFS over the CSR.
    pub fn bfs(&self, start: Uuid, max_depth: usize) -> Result<BfsResult, CuGraphError> {
        traversal::bfs(self, start, max_depth)
    }

    /// GPU (or CPU fallback) Dijkstra / SSSP from `start`.
    ///
    /// Edge costs are the edge weights stored in the adjacency (`AdjEntry::weight`).
    /// With `cuda`: cuGraph SSSP (optimised for positive-weight graphs).
    pub fn dijkstra(&self, start: Uuid) -> Result<DijkstraResult, CuGraphError> {
        traversal::dijkstra(self, start)
    }

    /// GPU (or CPU fallback) PageRank.
    ///
    /// `alpha` — damping factor (typical: 0.85).
    /// `max_iter` — max power-iteration steps.
    /// `tol` — L1 convergence threshold.
    pub fn pagerank(
        &self,
        alpha: f32,
        max_iter: u32,
        tol: f64,
    ) -> Result<PageRankResult, CuGraphError> {
        traversal::pagerank(self, alpha, max_iter, tol)
    }

    /// **GPU-only** Poincaré ball kNN.
    ///
    /// Computes exact Poincaré distances between `query` and all `db_embeddings`
    /// in parallel on the GPU using an NVRTC-compiled CUDA kernel.
    ///
    /// `db_embeddings[i]` must correspond to `self.node_ids()[i]`.
    ///
    /// Only available with `cuda` feature. Without it, use the CPU Poincaré
    /// search in `nietzsche-graph`.
    #[cfg(feature = "cuda")]
    pub fn poincare_knn(
        &self,
        query: &[f32],
        k: usize,
        db_embeddings: &[Vec<f32>],
    ) -> Result<Vec<(Uuid, f64)>, CuGraphError> {
        gpu::poincare::poincare_knn(self, query, k, db_embeddings)
    }

    /// **GPU batch** Poincaré kNN for ALL nodes simultaneously.
    ///
    /// Computes the full N×N pairwise distance matrix on GPU in a single
    /// kernel launch, then extracts top-k neighbours for each node.
    ///
    /// This is the L-System accelerator: replaces the O(n²) sequential
    /// Hausdorff loop with a single GPU burst.
    #[cfg(feature = "cuda")]
    pub fn poincare_batch_knn(
        embeddings: &[f32],
        n: usize,
        dim: usize,
        k: usize,
    ) -> Result<gpu::poincare_batch::BatchKnnResult, CuGraphError> {
        gpu::poincare_batch::poincare_batch_knn(embeddings, n, dim, k)
    }

    /// Number of vertices in the index.
    pub fn vertex_count(&self) -> usize {
        self.csr.vertex_count()
    }

    /// Number of directed edges in the index.
    pub fn edge_count(&self) -> usize {
        self.csr.edge_count()
    }

    /// Ordered list of node UUIDs (`node_ids[i]` = UUID for vertex index `i`).
    pub fn node_ids(&self) -> &[Uuid] {
        &self.csr.node_ids
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    pub(crate) fn start_idx(&self, id: Uuid) -> Option<u32> {
        self.csr.uuid_to_idx.get(&id).copied()
    }

    pub(crate) fn idx_to_uuid(&self, idx: u32) -> Option<Uuid> {
        self.csr.node_ids.get(idx as usize).copied()
    }
}
