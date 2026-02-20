//! BFS, Dijkstra (SSSP), PageRank — dispatches to GPU or CPU fallback.

use std::collections::HashMap;
use uuid::Uuid;

use crate::{CuGraphError, CuGraphIndex};

// ── Result types ──────────────────────────────────────────────────────────────

/// Nodes visited by BFS in breadth-first order, with their distances (hop count).
pub struct BfsResult {
    /// Visited node UUIDs in discovery order.
    pub visited: Vec<Uuid>,
    /// Hop distance from source for each visited node.
    pub distances: Vec<u32>,
}

/// SSSP result: shortest-path distances from the source node.
pub struct DijkstraResult {
    /// Map from node UUID to shortest path distance (sum of edge weights).
    pub distances: HashMap<Uuid, f64>,
}

/// PageRank centrality scores.
pub struct PageRankResult {
    /// Map from node UUID to PageRank score (sums to ~1.0).
    pub scores: HashMap<Uuid, f64>,
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

pub fn bfs(
    index: &CuGraphIndex,
    start: Uuid,
    max_depth: usize,
) -> Result<BfsResult, CuGraphError> {
    #[cfg(feature = "cuda")]
    {
        crate::gpu::bfs_gpu(index, start, max_depth)
    }
    #[cfg(not(feature = "cuda"))]
    {
        bfs_cpu(index, start, max_depth)
    }
}

pub fn dijkstra(
    index: &CuGraphIndex,
    start: Uuid,
) -> Result<DijkstraResult, CuGraphError> {
    #[cfg(feature = "cuda")]
    {
        crate::gpu::dijkstra_gpu(index, start)
    }
    #[cfg(not(feature = "cuda"))]
    {
        dijkstra_cpu(index, start)
    }
}

pub fn pagerank(
    index: &CuGraphIndex,
    alpha: f32,
    max_iter: u32,
    tol: f64,
) -> Result<PageRankResult, CuGraphError> {
    #[cfg(feature = "cuda")]
    {
        crate::gpu::pagerank_gpu(index, alpha, max_iter, tol)
    }
    #[cfg(not(feature = "cuda"))]
    {
        pagerank_cpu(index, alpha, max_iter, tol)
    }
}

// ── CPU fallbacks (no CUDA dependency) ───────────────────────────────────────

fn bfs_cpu(
    index: &CuGraphIndex,
    start: Uuid,
    max_depth: usize,
) -> Result<BfsResult, CuGraphError> {
    use std::collections::VecDeque;

    let start_idx = index
        .start_idx(start)
        .ok_or(CuGraphError::NodeNotFound(start))?;

    let n = index.csr.vertex_count();
    let mut dist = vec![u32::MAX; n];
    dist[start_idx as usize] = 0;

    let mut queue: VecDeque<u32> = VecDeque::new();
    queue.push_back(start_idx);

    let mut visited = Vec::new();
    let mut distances = Vec::new();

    while let Some(u) = queue.pop_front() {
        let d = dist[u as usize];
        if d as usize > max_depth {
            continue;
        }
        visited.push(index.idx_to_uuid(u).unwrap());
        distances.push(d);

        let off_start = index.csr.offsets[u as usize] as usize;
        let off_end   = index.csr.offsets[u as usize + 1] as usize;
        for &v in &index.csr.col_idx[off_start..off_end] {
            if dist[v as usize] == u32::MAX {
                dist[v as usize] = d + 1;
                queue.push_back(v);
            }
        }
    }

    Ok(BfsResult { visited, distances })
}

fn dijkstra_cpu(
    index: &CuGraphIndex,
    start: Uuid,
) -> Result<DijkstraResult, CuGraphError> {
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;

    let start_idx = index
        .start_idx(start)
        .ok_or(CuGraphError::NodeNotFound(start))?;

    let n = index.csr.vertex_count();
    let mut dist = vec![f64::INFINITY; n];
    dist[start_idx as usize] = 0.0;

    // Min-heap: (dist * 1e9 as u64, vertex_idx) for integer ordering
    let mut heap: BinaryHeap<Reverse<(ordered_float::OrderedFloat<f64>, u32)>> =
        BinaryHeap::new();
    heap.push(Reverse((ordered_float::OrderedFloat(0.0), start_idx)));

    while let Some(Reverse((d, u))) = heap.pop() {
        let d: f64 = d.into();
        if d > dist[u as usize] {
            continue;
        }
        let off_start = index.csr.offsets[u as usize] as usize;
        let off_end   = index.csr.offsets[u as usize + 1] as usize;
        for i in off_start..off_end {
            let v   = index.csr.col_idx[i];
            let w   = index.csr.weights[i] as f64;
            let alt = d + w;
            if alt < dist[v as usize] {
                dist[v as usize] = alt;
                heap.push(Reverse((ordered_float::OrderedFloat(alt), v)));
            }
        }
    }

    let distances = index
        .csr
        .node_ids
        .iter()
        .enumerate()
        .filter(|(i, _)| dist[*i].is_finite())
        .map(|(i, &id)| (id, dist[i]))
        .collect();

    Ok(DijkstraResult { distances })
}

fn pagerank_cpu(
    index: &CuGraphIndex,
    alpha: f32,
    max_iter: u32,
    tol: f64,
) -> Result<PageRankResult, CuGraphError> {
    let n = index.csr.vertex_count();
    if n == 0 {
        return Ok(PageRankResult { scores: HashMap::new() });
    }

    let alpha = alpha as f64;
    let mut rank = vec![1.0f64 / n as f64; n];
    let mut new_rank = vec![0.0f64; n];

    // Compute out-degree for dangling node handling
    let out_degree: Vec<u32> = (0..n)
        .map(|i| index.csr.offsets[i + 1] - index.csr.offsets[i])
        .collect();

    for _ in 0..max_iter {
        let dangling_sum: f64 = (0..n)
            .filter(|&i| out_degree[i] == 0)
            .map(|i| rank[i])
            .sum();

        new_rank.fill((1.0 - alpha) / n as f64 + alpha * dangling_sum / n as f64);

        for u in 0..n {
            if out_degree[u] == 0 {
                continue;
            }
            let contribution = alpha * rank[u] / out_degree[u] as f64;
            let off_start = index.csr.offsets[u] as usize;
            let off_end   = index.csr.offsets[u + 1] as usize;
            for &v in &index.csr.col_idx[off_start..off_end] {
                new_rank[v as usize] += contribution;
            }
        }

        // Convergence check (L1 norm)
        let delta: f64 = rank.iter().zip(&new_rank).map(|(a, b)| (a - b).abs()).sum();
        std::mem::swap(&mut rank, &mut new_rank);
        if delta < tol {
            break;
        }
    }

    let scores = index
        .csr
        .node_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, rank[i]))
        .collect();

    Ok(PageRankResult { scores })
}
