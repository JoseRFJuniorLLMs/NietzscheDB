//! GPU execution layer: cuGraph BFS / Dijkstra / PageRank + Poincaré kNN.
//!
//! Uses `libloading` to dynamically load `libcugraph.so` (RAPIDS 24.6) at
//! runtime, so the crate links cleanly even if RAPIDS is not installed at
//! *build* time — only at *runtime* on a CUDA-capable machine.

pub mod ffi;
pub mod poincare;

use std::collections::HashMap;
use uuid::Uuid;

use crate::{CuGraphError, CuGraphIndex};
use crate::traversal::{BfsResult, DijkstraResult, PageRankResult};

// ── Library path resolution ───────────────────────────────────────────────────

fn cugraph_lib_path() -> String {
    std::env::var("CUGRAPH_LIB")
        .unwrap_or_else(|_| "/usr/lib/x86_64-linux-gnu/libcugraph.so".to_string())
}

// ── BFS ───────────────────────────────────────────────────────────────────────

pub fn bfs_gpu(
    index: &CuGraphIndex,
    start: Uuid,
    max_depth: usize,
) -> Result<BfsResult, CuGraphError> {
    let start_idx = index
        .start_idx(start)
        .ok_or(CuGraphError::NodeNotFound(start))?;

    let lib_path = cugraph_lib_path();
    let session = ffi::CuGraphSession::load(&lib_path)?;

    let n = index.csr.vertex_count();
    let (vertices, distances) = session.bfs(
        &index.csr.offsets,
        &index.csr.col_idx,
        n,
        start_idx,
        max_depth as u64,
    )?;

    let visited = vertices
        .iter()
        .filter_map(|&v| index.idx_to_uuid(v))
        .collect();
    let dist_out = distances
        .iter()
        .map(|&d| if d == u32::MAX { u32::MAX } else { d })
        .collect();

    Ok(BfsResult {
        visited,
        distances: dist_out,
    })
}

// ── Dijkstra / SSSP ───────────────────────────────────────────────────────────

pub fn dijkstra_gpu(
    index: &CuGraphIndex,
    start: Uuid,
) -> Result<DijkstraResult, CuGraphError> {
    let start_idx = index
        .start_idx(start)
        .ok_or(CuGraphError::NodeNotFound(start))?;

    let lib_path = cugraph_lib_path();
    let session = ffi::CuGraphSession::load(&lib_path)?;

    let n = index.csr.vertex_count();
    let (vertices, distances) = session.sssp(
        &index.csr.offsets,
        &index.csr.col_idx,
        &index.csr.weights,
        n,
        start_idx,
    )?;

    let map: HashMap<Uuid, f64> = vertices
        .iter()
        .zip(distances.iter())
        .filter(|(_, &d)| d.is_finite())
        .filter_map(|(&v, &d)| index.idx_to_uuid(v).map(|id| (id, d)))
        .collect();

    Ok(DijkstraResult { distances: map })
}

// ── PageRank ──────────────────────────────────────────────────────────────────

pub fn pagerank_gpu(
    index: &CuGraphIndex,
    alpha: f32,
    max_iter: u32,
    tol: f64,
) -> Result<PageRankResult, CuGraphError> {
    let lib_path = cugraph_lib_path();
    let session = ffi::CuGraphSession::load(&lib_path)?;

    let n = index.csr.vertex_count();
    let (vertices, scores) = session.pagerank(
        &index.csr.offsets,
        &index.csr.col_idx,
        &index.csr.weights,
        n,
        alpha as f64,
        tol,
        max_iter as usize,
    )?;

    let map: HashMap<Uuid, f64> = vertices
        .iter()
        .zip(scores.iter())
        .filter_map(|(&v, &s)| index.idx_to_uuid(v).map(|id| (id, s)))
        .collect();

    Ok(PageRankResult { scores: map })
}
