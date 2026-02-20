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
