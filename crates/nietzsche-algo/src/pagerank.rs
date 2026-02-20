//! PageRank via power iteration.

use std::collections::HashMap;
use std::time::Instant;
use uuid::Uuid;
use nietzsche_graph::{GraphStorage, AdjacencyIndex, GraphError};

pub struct PageRankConfig {
    pub damping_factor: f64,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 20,
            convergence_threshold: 1e-7,
        }
    }
}

pub struct PageRankResult {
    pub scores: Vec<(Uuid, f64)>,
    pub iterations: usize,
    pub converged: bool,
    pub duration_ms: u64,
}

/// PageRank via power iteration.
///
/// Scores sum to 1.0.  Convergence is measured by L1 norm of the
/// score-delta vector falling below `convergence_threshold`.
pub fn pagerank(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    config: &PageRankConfig,
) -> Result<PageRankResult, GraphError> {
    let start = Instant::now();

    let node_ids: Vec<Uuid> = storage
        .scan_nodes_meta()?
        .into_iter()
        .map(|n| n.id)
        .collect();
    let n = node_ids.len();

    if n == 0 {
        return Ok(PageRankResult {
            scores: vec![],
            iterations: 0,
            converged: true,
            duration_ms: 0,
        });
    }

    let id_to_idx: HashMap<Uuid, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    let d = config.damping_factor;
    let base = (1.0 - d) / n as f64;

    let mut scores = vec![1.0 / n as f64; n];
    let mut new_scores = vec![0.0_f64; n];
    let mut converged = false;
    let mut iterations = 0;

    // Pre-compute out-degree for each node
    let out_deg: Vec<usize> = node_ids
        .iter()
        .map(|id| adjacency.degree_out(id))
        .collect();

    for _ in 0..config.max_iterations {
        iterations += 1;

        // Reset
        for s in new_scores.iter_mut() {
            *s = base;
        }

        // Distribute rank
        for u in 0..n {
            if out_deg[u] == 0 {
                // Dangling node: distribute evenly
                let share = d * scores[u] / n as f64;
                for s in new_scores.iter_mut() {
                    *s += share;
                }
            } else {
                let share = d * scores[u] / out_deg[u] as f64;
                for entry in adjacency.entries_out(&node_ids[u]) {
                    if let Some(&v) = id_to_idx.get(&entry.neighbor_id) {
                        new_scores[v] += share;
                    }
                }
            }
        }

        // Check convergence (L1 norm)
        let diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        std::mem::swap(&mut scores, &mut new_scores);

        if diff < config.convergence_threshold {
            converged = true;
            break;
        }
    }

    let mut result: Vec<(Uuid, f64)> = node_ids
        .into_iter()
        .zip(scores.into_iter())
        .collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(PageRankResult {
        scores: result,
        iterations,
        converged,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}
