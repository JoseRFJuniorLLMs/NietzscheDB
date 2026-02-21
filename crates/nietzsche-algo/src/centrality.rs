//! Centrality algorithms: Betweenness (Brandes), Closeness, Degree.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use uuid::Uuid;
use nietzsche_graph::{GraphStorage, AdjacencyIndex, GraphError};

pub enum Direction { In, Out, Both }

pub struct BetweennessResult {
    pub scores: Vec<(Uuid, f64)>,
    pub duration_ms: u64,
}

/// Betweenness centrality via Brandes' algorithm.
///
/// If `sample_size` is `Some(k)`, only samples `k` source nodes for an
/// approximation (O(k*E) instead of O(V*E)).
pub fn betweenness_centrality(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    sample_size: Option<usize>,
) -> Result<BetweennessResult, GraphError> {
    let start = Instant::now();

    let node_ids: Vec<Uuid> = storage
        .scan_nodes_meta()?
        .into_iter()
        .map(|n| n.id)
        .collect();
    let n = node_ids.len();
    if n == 0 {
        return Ok(BetweennessResult { scores: vec![], duration_ms: 0 });
    }

    let id_to_idx: HashMap<Uuid, usize> = node_ids.iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    let mut bc = vec![0.0_f64; n];

    // Select sources
    let sources: Vec<usize> = if let Some(k) = sample_size {
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rand::thread_rng());
        indices.truncate(k);
        indices
    } else {
        (0..n).collect()
    };

    for &s in &sources {
        // BFS from source
        let mut stack: Vec<usize> = Vec::new();
        let mut predecessors: Vec<Vec<usize>> = vec![vec![]; n];
        let mut sigma = vec![0.0_f64; n];  // number of shortest paths
        let mut dist = vec![-1i64; n];
        let mut delta = vec![0.0_f64; n];

        sigma[s] = 1.0;
        dist[s] = 0;

        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let v_id = &node_ids[v];

            for entry in adjacency.entries_out(v_id) {
                if let Some(&w) = id_to_idx.get(&entry.neighbor_id) {
                    if dist[w] < 0 {
                        queue.push_back(w);
                        dist[w] = dist[v] + 1;
                    }
                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v];
                        predecessors[w].push(v);
                    }
                }
            }
        }

        // Back-propagation
        while let Some(w) = stack.pop() {
            for &v in &predecessors[w] {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            if w != s {
                bc[w] += delta[w];
            }
        }

        // Reset delta
        for d in delta.iter_mut() { *d = 0.0; }
    }

    // Normalize for undirected approximation
    if sample_size.is_some() {
        let scale = n as f64 / sources.len() as f64;
        for b in bc.iter_mut() { *b *= scale; }
    }

    let mut result: Vec<(Uuid, f64)> = node_ids.into_iter()
        .zip(bc.into_iter())
        .collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(BetweennessResult {
        scores: result,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}

/// Closeness centrality: `(N-1) / sum(shortest_distances)` for each node.
pub fn closeness_centrality(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
) -> Result<Vec<(Uuid, f64)>, GraphError> {
    let node_ids: Vec<Uuid> = storage
        .scan_nodes_meta()?
        .into_iter()
        .map(|n| n.id)
        .collect();
    let n = node_ids.len();
    if n == 0 { return Ok(vec![]); }

    let id_to_idx: HashMap<Uuid, usize> = node_ids.iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    let mut result = Vec::with_capacity(n);

    for s in 0..n {
        // BFS to compute shortest distances
        let mut dist = vec![-1i64; n];
        dist[s] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(s);
        let mut total_dist = 0i64;
        let mut reachable = 0usize;

        while let Some(v) = queue.pop_front() {
            let v_id = &node_ids[v];
            for entry in adjacency.entries_out(v_id) {
                if let Some(&w) = id_to_idx.get(&entry.neighbor_id) {
                    if dist[w] < 0 {
                        dist[w] = dist[v] + 1;
                        total_dist += dist[w];
                        reachable += 1;
                        queue.push_back(w);
                    }
                }
            }
        }

        let closeness = if reachable > 0 && total_dist > 0 {
            reachable as f64 / total_dist as f64
        } else {
            0.0
        };
        result.push((node_ids[s], closeness));
    }

    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(result)
}

/// Degree centrality: normalized degree for each node.
pub fn degree_centrality(
    adjacency: &AdjacencyIndex,
    direction: Direction,
    node_ids: &[Uuid],
) -> Vec<(Uuid, f64)> {
    let n = node_ids.len();
    if n <= 1 { return node_ids.iter().map(|id| (*id, 0.0)).collect(); }

    let denom = (n - 1) as f64;
    let mut result: Vec<(Uuid, f64)> = node_ids.iter().map(|id| {
        let deg = match direction {
            Direction::Out  => adjacency.degree_out(id),
            Direction::In   => adjacency.degree_in(id),
            Direction::Both => adjacency.degree_out(id) + adjacency.degree_in(id),
        };
        (*id, deg as f64 / denom)
    }).collect();

    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::model::{Edge, Node, PoincareVector};
    use nietzsche_graph::{AdjacencyIndex, GraphStorage};
    use tempfile::TempDir;

    fn open_temp_db() -> (GraphStorage, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();
        (storage, dir)
    }

    fn make_node(x: f32, y: f32) -> Node {
        Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, y]),
            serde_json::json!({"label": "test"}),
        )
    }

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

    // ── Betweenness centrality tests ─────────────────────────────────────

    #[test]
    fn betweenness_empty_graph() {
        let (storage, _dir) = open_temp_db();
        let adj = AdjacencyIndex::new();

        let result = betweenness_centrality(&storage, &adj, None).unwrap();
        assert!(result.scores.is_empty());
    }

    #[test]
    fn betweenness_path_graph() {
        // Path: 0 -> 1 -> 2 -> 3
        // Node 1 and node 2 are on all shortest paths between the endpoints
        let nodes: Vec<Node> = (0..4).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let result = betweenness_centrality(&storage, &adj, None).unwrap();
        let score_map: HashMap<Uuid, f64> = result.scores.into_iter().collect();

        // Endpoints (0, 3) should have 0 betweenness
        assert!(
            score_map[&ids[0]].abs() < 1e-6,
            "endpoint node 0 should have 0 betweenness"
        );
        assert!(
            score_map[&ids[3]].abs() < 1e-6,
            "endpoint node 3 should have 0 betweenness"
        );

        // Interior nodes (1, 2) should have positive betweenness
        assert!(
            score_map[&ids[1]] > 0.0,
            "interior node 1 should have positive betweenness"
        );
        assert!(
            score_map[&ids[2]] > 0.0,
            "interior node 2 should have positive betweenness"
        );
    }

    #[test]
    fn betweenness_star_center_highest() {
        // Star: 0 is the center, 1..4 connect to center
        // Center should have highest betweenness because all shortest paths go through it
        let nodes: Vec<Node> = (0..5).map(|i| make_node(0.05 * i as f32, 0.05)).collect();
        let edges = vec![
            (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0),
            (1, 0, 1.0), (2, 0, 1.0), (3, 0, 1.0), (4, 0, 1.0),
        ];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let result = betweenness_centrality(&storage, &adj, None).unwrap();
        let score_map: HashMap<Uuid, f64> = result.scores.into_iter().collect();

        let center_bc = score_map[&ids[0]];
        for i in 1..5 {
            assert!(
                center_bc >= score_map[&ids[i]],
                "center should have highest betweenness"
            );
        }
    }

    // ── Closeness centrality tests ───────────────────────────────────────

    #[test]
    fn closeness_empty_graph() {
        let (storage, _dir) = open_temp_db();
        let adj = AdjacencyIndex::new();

        let result = closeness_centrality(&storage, &adj).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn closeness_star_center_highest() {
        // Star: center -> all spokes, and spokes -> center
        // Center has distance 1 to every other node
        let nodes: Vec<Node> = (0..5).map(|i| make_node(0.05 * i as f32, 0.05)).collect();
        let edges = vec![
            (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0),
        ];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let result = closeness_centrality(&storage, &adj).unwrap();
        let score_map: HashMap<Uuid, f64> = result.into_iter().collect();

        let center_cc = score_map[&ids[0]];
        // Center can reach all 4 spokes at distance 1 => closeness = 4/4 = 1.0
        assert!(
            (center_cc - 1.0).abs() < 1e-6,
            "center closeness should be 1.0, got {center_cc}"
        );
    }

    #[test]
    fn closeness_path_graph() {
        // Path: 0 -> 1 -> 2
        // Node 0 can reach 1 (dist 1) and 2 (dist 2), closeness = 2/3
        // Node 1 can reach 2 (dist 1) only, closeness = 1/1 = 1.0
        // Node 2 is a sink with no outgoing, closeness = 0
        let nodes: Vec<Node> = (0..3).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let result = closeness_centrality(&storage, &adj).unwrap();
        let score_map: HashMap<Uuid, f64> = result.into_iter().collect();

        // Node 0: reaches 2 nodes, total dist = 1+2 = 3, closeness = 2/3
        assert!(
            (score_map[&ids[0]] - 2.0 / 3.0).abs() < 1e-6,
            "node 0 closeness expected 2/3, got {}",
            score_map[&ids[0]]
        );
        // Node 1: reaches 1 node at dist 1, closeness = 1/1 = 1.0
        assert!(
            (score_map[&ids[1]] - 1.0).abs() < 1e-6,
            "node 1 closeness expected 1.0, got {}",
            score_map[&ids[1]]
        );
        // Node 2: reaches 0 nodes, closeness = 0
        assert!(
            score_map[&ids[2]].abs() < 1e-6,
            "node 2 (sink) closeness should be 0, got {}",
            score_map[&ids[2]]
        );
    }

    // ── Degree centrality tests ──────────────────────────────────────────

    #[test]
    fn degree_centrality_single_node() {
        let node = make_node(0.1, 0.1);
        let adj = AdjacencyIndex::new();
        let ids = vec![node.id];

        let result = degree_centrality(&adj, Direction::Both, &ids);
        assert_eq!(result.len(), 1);
        assert!((result[0].1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn degree_centrality_star_out() {
        // Star: center (node 0) has 4 outgoing edges
        let nodes: Vec<Node> = (0..5).map(|i| make_node(0.05 * i as f32, 0.05)).collect();
        let ids: Vec<Uuid> = nodes.iter().map(|n| n.id).collect();
        let adj = AdjacencyIndex::new();

        for i in 1..5 {
            let edge = Edge::association(ids[0], ids[i], 1.0);
            adj.add_edge(&edge);
        }

        let result = degree_centrality(&adj, Direction::Out, &ids);
        let score_map: HashMap<Uuid, f64> = result.into_iter().collect();

        // Center: 4 outgoing / (5-1) = 1.0
        assert!(
            (score_map[&ids[0]] - 1.0).abs() < 1e-6,
            "center out-degree centrality should be 1.0"
        );
        // Spokes: 0 outgoing / (5-1) = 0.0
        for i in 1..5 {
            assert!(
                score_map[&ids[i]].abs() < 1e-6,
                "spoke out-degree centrality should be 0.0"
            );
        }
    }

    #[test]
    fn degree_centrality_both_directions() {
        // Triangle: 0->1, 1->2, 2->0
        let nodes: Vec<Node> = (0..3).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let ids: Vec<Uuid> = nodes.iter().map(|n| n.id).collect();
        let adj = AdjacencyIndex::new();

        let e01 = Edge::association(ids[0], ids[1], 1.0);
        let e12 = Edge::association(ids[1], ids[2], 1.0);
        let e20 = Edge::association(ids[2], ids[0], 1.0);
        adj.add_edge(&e01);
        adj.add_edge(&e12);
        adj.add_edge(&e20);

        let result = degree_centrality(&adj, Direction::Both, &ids);
        let score_map: HashMap<Uuid, f64> = result.into_iter().collect();

        // Each node has 1 out + 1 in = 2 total, denom = 2 => centrality = 1.0
        for i in 0..3 {
            assert!(
                (score_map[&ids[i]] - 1.0).abs() < 1e-6,
                "node {i} both-degree centrality should be 1.0, got {}",
                score_map[&ids[i]]
            );
        }
    }
}
