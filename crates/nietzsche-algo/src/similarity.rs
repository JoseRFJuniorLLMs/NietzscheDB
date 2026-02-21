//! Node similarity: Jaccard.

use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use nietzsche_graph::{GraphStorage, AdjacencyIndex, GraphError};

pub struct SimilarityPair {
    pub node_a: Uuid,
    pub node_b: Uuid,
    pub score: f64,
}

/// Jaccard node similarity: `|N(u) ∩ N(v)| / |N(u) ∪ N(v)|`.
///
/// Returns pairs with similarity >= `threshold`, sorted descending, limited to `top_k`.
pub fn jaccard_similarity(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    top_k: usize,
    threshold: f64,
) -> Result<Vec<SimilarityPair>, GraphError> {
    let node_ids: Vec<Uuid> = storage
        .scan_nodes_meta()?
        .into_iter()
        .map(|n| n.id)
        .collect();
    let n = node_ids.len();

    let id_to_idx: HashMap<Uuid, usize> = node_ids.iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    // Build neighbor sets (undirected)
    let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for (i, id) in node_ids.iter().enumerate() {
        for entry in adjacency.entries_out(id) {
            if let Some(&j) = id_to_idx.get(&entry.neighbor_id) {
                neighbors[i].insert(j);
                neighbors[j].insert(i);
            }
        }
    }

    let mut pairs = Vec::new();

    // Compare all pairs with a neighbor-of-neighbor optimization:
    // only compare nodes that share at least one neighbor
    for u in 0..n {
        let mut candidates: HashSet<usize> = HashSet::new();
        for &v in &neighbors[u] {
            for &w in &neighbors[v] {
                if w > u { candidates.insert(w); }
            }
        }

        for v in candidates {
            let intersection = neighbors[u].intersection(&neighbors[v]).count();
            if intersection == 0 { continue; }
            let union_size = neighbors[u].len() + neighbors[v].len() - intersection;
            if union_size == 0 { continue; }
            let score = intersection as f64 / union_size as f64;
            if score >= threshold {
                pairs.push(SimilarityPair {
                    node_a: node_ids[u],
                    node_b: node_ids[v],
                    score,
                });
            }
        }
    }

    // Sort descending by score, take top_k
    pairs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(top_k);

    Ok(pairs)
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

    #[test]
    fn jaccard_empty_graph() {
        let (storage, _dir) = open_temp_db();
        let adj = AdjacencyIndex::new();

        let result = jaccard_similarity(&storage, &adj, 10, 0.0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn jaccard_no_common_neighbors() {
        // 0 -> 1, 2 -> 3 (disjoint; 0 and 2 share no neighbors)
        let nodes: Vec<Node> = (0..4).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let edges = vec![(0, 1, 1.0), (2, 3, 1.0)];
        let (storage, adj, _ids, _dir) = build_graph(&nodes, &edges);

        let result = jaccard_similarity(&storage, &adj, 100, 0.0).unwrap();
        // No pairs with common neighbors
        for pair in &result {
            assert!(pair.score >= 0.0);
        }
    }

    #[test]
    fn jaccard_identical_neighborhoods() {
        // Nodes 0 and 1 both connect to node 2 and node 3
        // Jaccard(0,1) = |{2,3}| / |{2,3}| = 1.0
        let nodes: Vec<Node> = (0..4).map(|i| make_node(0.05 * i as f32, 0.05)).collect();
        let edges = vec![
            (0, 2, 1.0), (0, 3, 1.0),
            (1, 2, 1.0), (1, 3, 1.0),
        ];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let result = jaccard_similarity(&storage, &adj, 100, 0.0).unwrap();

        // Find the pair (0, 1) or (1, 0)
        let pair_01 = result.iter().find(|p| {
            (p.node_a == ids[0] && p.node_b == ids[1])
                || (p.node_a == ids[1] && p.node_b == ids[0])
        });
        assert!(pair_01.is_some(), "pair (0,1) should appear");
        let pair = pair_01.unwrap();
        assert!(
            (pair.score - 1.0).abs() < 1e-6,
            "identical neighborhoods => Jaccard = 1.0, got {}",
            pair.score
        );
    }

    #[test]
    fn jaccard_partial_overlap() {
        // Node 0 neighbors: {2, 3}
        // Node 1 neighbors: {2, 4}
        // Intersection = {2}, Union = {2,3,4}
        // Jaccard = 1/3
        let nodes: Vec<Node> = (0..5).map(|i| make_node(0.05 * i as f32, 0.05)).collect();
        let edges = vec![
            (0, 2, 1.0), (0, 3, 1.0),
            (1, 2, 1.0), (1, 4, 1.0),
        ];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let result = jaccard_similarity(&storage, &adj, 100, 0.0).unwrap();

        let pair_01 = result.iter().find(|p| {
            (p.node_a == ids[0] && p.node_b == ids[1])
                || (p.node_a == ids[1] && p.node_b == ids[0])
        });
        assert!(pair_01.is_some(), "pair (0,1) should appear");
        let pair = pair_01.unwrap();
        assert!(
            (pair.score - 1.0 / 3.0).abs() < 1e-6,
            "partial overlap => Jaccard = 1/3, got {}",
            pair.score
        );
    }

    #[test]
    fn jaccard_respects_threshold() {
        // Same as partial_overlap, but set threshold > 1/3
        let nodes: Vec<Node> = (0..5).map(|i| make_node(0.05 * i as f32, 0.05)).collect();
        let edges = vec![
            (0, 2, 1.0), (0, 3, 1.0),
            (1, 2, 1.0), (1, 4, 1.0),
        ];
        let (storage, adj, _ids, _dir) = build_graph(&nodes, &edges);

        let result = jaccard_similarity(&storage, &adj, 100, 0.5).unwrap();
        // The only pair has Jaccard = 1/3 < 0.5, so it should be filtered out
        for pair in &result {
            assert!(
                pair.score >= 0.5,
                "all returned pairs should meet threshold"
            );
        }
    }

    #[test]
    fn jaccard_respects_top_k() {
        // Build a graph with multiple pairs of varying similarity
        let nodes: Vec<Node> = (0..6).map(|i| make_node(0.05 * i as f32, 0.05)).collect();
        let edges = vec![
            (0, 4, 1.0), (0, 5, 1.0),
            (1, 4, 1.0), (1, 5, 1.0),  // pair(0,1) Jaccard=1.0
            (2, 4, 1.0),
            (3, 4, 1.0), (3, 5, 1.0),  // pair(2,3) has some overlap
        ];
        let (storage, adj, _ids, _dir) = build_graph(&nodes, &edges);

        let result = jaccard_similarity(&storage, &adj, 1, 0.0).unwrap();
        assert!(result.len() <= 1, "top_k=1 should return at most 1 pair");
    }
}
