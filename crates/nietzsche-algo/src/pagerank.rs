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

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::model::{Edge, Node, PoincareVector};
    use nietzsche_graph::{AdjacencyIndex, GraphStorage};
    use tempfile::TempDir;

    /// Open a temporary RocksDB-backed GraphStorage.
    fn open_temp_db() -> (GraphStorage, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();
        (storage, dir)
    }

    /// Create a minimal node with a 2-D Poincare embedding.
    fn make_node(x: f32, y: f32) -> Node {
        Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, y]),
            serde_json::json!({"label": "test"}),
        )
    }

    /// Insert nodes and edges, returning (storage, adjacency, node_ids, _dir).
    fn build_graph(
        nodes: &[Node],
        edges: &[(usize, usize, f32)],
    ) -> (GraphStorage, AdjacencyIndex, Vec<Uuid>, TempDir) {
        let (storage, dir) = open_temp_db();
        let adj = AdjacencyIndex::new();
        let ids: Vec<Uuid> = nodes.iter().map(|n| n.id).collect();

        for node in nodes {
            storage.put_node(node).unwrap();
        }
        for &(from, to, weight) in edges {
            let edge = Edge::association(ids[from], ids[to], weight);
            storage.put_edge(&edge).unwrap();
            adj.add_edge(&edge);
        }
        (storage, adj, ids, dir)
    }

    // ── Test: empty graph ──────────────────────────────────────────────────

    #[test]
    fn pagerank_empty_graph() {
        let (storage, _dir) = open_temp_db();
        let adj = AdjacencyIndex::new();
        let config = PageRankConfig::default();

        let result = pagerank(&storage, &adj, &config).unwrap();
        assert!(result.scores.is_empty());
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    // ── Test: single node (dangling) ───────────────────────────────────────

    #[test]
    fn pagerank_single_node() {
        let node = make_node(0.1, 0.1);
        let (storage, adj, ids, _dir) = build_graph(&[node], &[]);
        let config = PageRankConfig::default();

        let result = pagerank(&storage, &adj, &config).unwrap();
        assert_eq!(result.scores.len(), 1);
        assert_eq!(result.scores[0].0, ids[0]);
        // Single node gets all the rank
        assert!((result.scores[0].1 - 1.0).abs() < 1e-6);
    }

    // ── Test: star topology (hub should have highest rank) ─────────────────

    #[test]
    fn pagerank_star_topology() {
        // Node 0 is the hub; nodes 1..4 point to node 0
        let nodes: Vec<Node> = (0..5).map(|i| make_node(0.1 * i as f32, 0.05)).collect();
        let edges: Vec<(usize, usize, f32)> = (1..5).map(|i| (i, 0, 1.0)).collect();
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let config = PageRankConfig { max_iterations: 100, ..Default::default() };
        let result = pagerank(&storage, &adj, &config).unwrap();

        // Hub (node 0) should have the highest rank
        assert_eq!(result.scores[0].0, ids[0]);
        // Hub rank should be significantly higher than any spoke
        let hub_score = result.scores[0].1;
        for &(_, score) in result.scores.iter().skip(1) {
            assert!(hub_score > score, "hub should have higher rank than spokes");
        }
    }

    // ── Test: cycle (all ranks should be equal) ────────────────────────────

    #[test]
    fn pagerank_cycle_equal_ranks() {
        // 0 -> 1 -> 2 -> 3 -> 0
        let nodes: Vec<Node> = (0..4).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)];
        let (storage, adj, _ids, _dir) = build_graph(&nodes, &edges);

        let config = PageRankConfig {
            max_iterations: 200,
            convergence_threshold: 1e-10,
            ..Default::default()
        };
        let result = pagerank(&storage, &adj, &config).unwrap();

        // All scores should be approximately equal (1/4 = 0.25)
        let expected = 1.0 / 4.0;
        for &(_, score) in &result.scores {
            assert!(
                (score - expected).abs() < 1e-4,
                "expected {expected}, got {score} in a 4-cycle"
            );
        }
        assert!(result.converged);
    }

    // ── Test: scores sum to 1.0 ────────────────────────────────────────────

    #[test]
    fn pagerank_scores_sum_to_one() {
        // A -> B -> C, A -> C
        let nodes: Vec<Node> = (0..3).map(|i| make_node(0.1 + 0.1 * i as f32, 0.05)).collect();
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let (storage, adj, _ids, _dir) = build_graph(&nodes, &edges);

        let config = PageRankConfig::default();
        let result = pagerank(&storage, &adj, &config).unwrap();

        let sum: f64 = result.scores.iter().map(|(_, s)| s).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "pagerank scores should sum to 1.0, got {sum}"
        );
    }
}
