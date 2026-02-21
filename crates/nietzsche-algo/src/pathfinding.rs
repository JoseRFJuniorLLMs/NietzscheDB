//! Pathfinding: A* and Triangle Count.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use uuid::Uuid;
use nietzsche_graph::{GraphStorage, AdjacencyIndex, GraphError};

// ── A* ──────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct AStarNode {
    idx: usize,
    f_score: f64,
}

impl PartialEq for AStarNode {
    fn eq(&self, other: &Self) -> bool { self.idx == other.idx }
}
impl Eq for AStarNode {}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f_score.partial_cmp(&self.f_score)
            .unwrap_or(Ordering::Equal) // min-heap
    }
}

/// A* pathfinding using Poincare distance as heuristic.
///
/// Returns `Ok(Some((path, cost)))` if a path exists, `Ok(None)` if unreachable.
pub fn astar(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    start: Uuid,
    goal: Uuid,
) -> Result<Option<(Vec<Uuid>, f64)>, GraphError> {
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

    let start_idx = match id_to_idx.get(&start) {
        Some(&i) => i,
        None => return Ok(None),
    };
    let goal_idx = match id_to_idx.get(&goal) {
        Some(&i) => i,
        None => return Ok(None),
    };

    // Load goal embedding for heuristic
    let goal_embedding = storage.get_embedding(&goal)?;

    let mut g_score = vec![f64::INFINITY; n];
    let mut came_from = vec![usize::MAX; n];
    let mut open = BinaryHeap::new();
    let mut closed = vec![false; n];

    g_score[start_idx] = 0.0;

    let h = |idx: usize| -> f64 {
        if let Some(ref ge) = goal_embedding {
            if let Ok(Some(emb)) = storage.get_embedding(&node_ids[idx]) {
                return emb.distance(ge);
            }
        }
        0.0 // fallback: no heuristic (degrades to Dijkstra)
    };

    open.push(AStarNode { idx: start_idx, f_score: h(start_idx) });

    while let Some(current) = open.pop() {
        if current.idx == goal_idx {
            // Reconstruct path
            let mut path = Vec::new();
            let mut c = goal_idx;
            while c != usize::MAX {
                path.push(node_ids[c]);
                c = came_from[c];
            }
            path.reverse();
            return Ok(Some((path, g_score[goal_idx])));
        }

        if closed[current.idx] { continue; }
        closed[current.idx] = true;

        let v_id = &node_ids[current.idx];
        for entry in adjacency.entries_out(v_id) {
            if let Some(&w) = id_to_idx.get(&entry.neighbor_id) {
                if closed[w] { continue; }
                let tentative = g_score[current.idx] + (entry.weight as f64).max(0.001);
                if tentative < g_score[w] {
                    g_score[w] = tentative;
                    came_from[w] = current.idx;
                    open.push(AStarNode { idx: w, f_score: tentative + h(w) });
                }
            }
        }
    }

    Ok(None) // unreachable
}

// ── Triangle Count ──────────────────────────────────────────────────────────

/// Count the total number of triangles in the graph.
///
/// A triangle is three nodes (u, v, w) with edges u-v, v-w, u-w (in any direction).
pub fn triangle_count(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
) -> Result<u64, GraphError> {
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

    // Build undirected neighbor sets (index-based) for intersection counting
    let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for (i, id) in node_ids.iter().enumerate() {
        for entry in adjacency.entries_out(id) {
            if let Some(&j) = id_to_idx.get(&entry.neighbor_id) {
                neighbors[i].insert(j);
                neighbors[j].insert(i);
            }
        }
    }

    let mut count = 0u64;
    for u in 0..n {
        for &v in &neighbors[u] {
            if v <= u { continue; } // avoid double counting
            // Count common neighbors w > v
            for &w in &neighbors[u] {
                if w <= v { continue; }
                if neighbors[v].contains(&w) {
                    count += 1;
                }
            }
        }
    }

    Ok(count)
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

    // ── A* tests ─────────────────────────────────────────────────────────

    #[test]
    fn astar_same_node() {
        let node = make_node(0.1, 0.1);
        let (storage, adj, ids, _dir) = build_graph(&[node], &[]);

        let result = astar(&storage, &adj, ids[0], ids[0]).unwrap();
        assert!(result.is_some(), "path from node to itself should exist");
        let (path, cost) = result.unwrap();
        assert_eq!(path, vec![ids[0]]);
        assert!((cost - 0.0).abs() < 1e-6, "self-path cost should be 0");
    }

    #[test]
    fn astar_direct_edge() {
        let nodes: Vec<Node> = (0..2).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let edges = vec![(0, 1, 1.0)];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let result = astar(&storage, &adj, ids[0], ids[1]).unwrap();
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(path, vec![ids[0], ids[1]]);
        assert!(cost > 0.0);
    }

    #[test]
    fn astar_unreachable() {
        // Two isolated nodes
        let nodes: Vec<Node> = (0..2).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let (storage, adj, ids, _dir) = build_graph(&nodes, &[]);

        let result = astar(&storage, &adj, ids[0], ids[1]).unwrap();
        assert!(result.is_none(), "disconnected nodes should return None");
    }

    #[test]
    fn astar_multi_hop_path() {
        // Path: 0 -> 1 -> 2 -> 3
        let nodes: Vec<Node> = (0..4).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let result = astar(&storage, &adj, ids[0], ids[3]).unwrap();
        assert!(result.is_some());
        let (path, _cost) = result.unwrap();
        assert_eq!(path.len(), 4);
        assert_eq!(path[0], ids[0]);
        assert_eq!(path[3], ids[3]);
    }

    #[test]
    fn astar_nonexistent_node() {
        let node = make_node(0.1, 0.1);
        let (storage, adj, _ids, _dir) = build_graph(&[node], &[]);
        let fake_id = Uuid::new_v4();

        let result = astar(&storage, &adj, fake_id, fake_id).unwrap();
        assert!(result.is_none(), "nonexistent start should return None");
    }

    // ── Triangle count tests ─────────────────────────────────────────────

    #[test]
    fn triangle_count_empty_graph() {
        let (storage, _dir) = open_temp_db();
        let adj = AdjacencyIndex::new();

        let count = triangle_count(&storage, &adj).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn triangle_count_no_triangles() {
        // Path: 0 -> 1 -> 2 (no closing edge => 0 triangles)
        let nodes: Vec<Node> = (0..3).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];
        let (storage, adj, _ids, _dir) = build_graph(&nodes, &edges);

        let count = triangle_count(&storage, &adj).unwrap();
        assert_eq!(count, 0, "path graph has no triangles");
    }

    #[test]
    fn triangle_count_single_triangle() {
        // Triangle: 0 -> 1, 1 -> 2, 0 -> 2
        let nodes: Vec<Node> = (0..3).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let (storage, adj, _ids, _dir) = build_graph(&nodes, &edges);

        let count = triangle_count(&storage, &adj).unwrap();
        assert_eq!(count, 1, "three nodes with closing edge = 1 triangle");
    }

    #[test]
    fn triangle_count_complete_4() {
        // K4 (complete graph on 4 nodes): C(4,3) = 4 triangles
        let nodes: Vec<Node> = (0..4).map(|i| make_node(0.05 * i as f32, 0.05)).collect();
        // All directed edges (complete)
        let mut edges = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    edges.push((i, j, 1.0));
                }
            }
        }
        let (storage, adj, _ids, _dir) = build_graph(&nodes, &edges);

        let count = triangle_count(&storage, &adj).unwrap();
        assert_eq!(count, 4, "K4 has C(4,3) = 4 triangles");
    }
}
