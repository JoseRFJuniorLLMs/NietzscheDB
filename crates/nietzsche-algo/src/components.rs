//! Connected components: WCC (Union-Find) and SCC (Tarjan).

use std::collections::HashMap;
use std::time::Instant;
use uuid::Uuid;
use nietzsche_graph::{GraphStorage, AdjacencyIndex, GraphError};

pub struct ComponentResult {
    pub components: Vec<(Uuid, u64)>,
    pub component_count: usize,
    pub largest_component_size: usize,
    pub duration_ms: u64,
}

// ── Union-Find ──────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // path compression
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry { return; }
        // union by rank
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
    }
}

/// Weakly Connected Components using Union-Find.
///
/// Treats the graph as undirected: edges go both ways.
pub fn weakly_connected_components(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
) -> Result<ComponentResult, GraphError> {
    let start = Instant::now();

    let node_ids: Vec<Uuid> = storage
        .scan_nodes_meta()?
        .into_iter()
        .map(|n| n.id)
        .collect();
    let n = node_ids.len();
    if n == 0 {
        return Ok(ComponentResult {
            components: vec![], component_count: 0, largest_component_size: 0, duration_ms: 0,
        });
    }

    let id_to_idx: HashMap<Uuid, usize> = node_ids.iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    let mut uf = UnionFind::new(n);

    // Union all edges
    for (i, id) in node_ids.iter().enumerate() {
        for entry in adjacency.entries_out(id) {
            if let Some(&j) = id_to_idx.get(&entry.neighbor_id) {
                uf.union(i, j);
            }
        }
    }

    // Collect components
    let mut comp_map: HashMap<usize, u64> = HashMap::new();
    let mut next_id = 0u64;
    let mut comp_sizes: HashMap<u64, usize> = HashMap::new();

    let components: Vec<(Uuid, u64)> = node_ids.iter().enumerate().map(|(i, id)| {
        let root = uf.find(i);
        let comp_id = *comp_map.entry(root).or_insert_with(|| { let c = next_id; next_id += 1; c });
        *comp_sizes.entry(comp_id).or_default() += 1;
        (*id, comp_id)
    }).collect();

    let largest = comp_sizes.values().copied().max().unwrap_or(0);

    Ok(ComponentResult {
        components,
        component_count: comp_map.len(),
        largest_component_size: largest,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}

/// Strongly Connected Components using Tarjan's algorithm.
pub fn strongly_connected_components(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
) -> Result<ComponentResult, GraphError> {
    let start = Instant::now();

    let node_ids: Vec<Uuid> = storage
        .scan_nodes_meta()?
        .into_iter()
        .map(|n| n.id)
        .collect();
    let n = node_ids.len();
    if n == 0 {
        return Ok(ComponentResult {
            components: vec![], component_count: 0, largest_component_size: 0, duration_ms: 0,
        });
    }

    let id_to_idx: HashMap<Uuid, usize> = node_ids.iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    // Build adjacency list (index-based) from outgoing entries
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, id) in node_ids.iter().enumerate() {
        for entry in adjacency.entries_out(id) {
            if let Some(&j) = id_to_idx.get(&entry.neighbor_id) {
                adj[i].push(j);
            }
        }
    }

    // Tarjan's iterative
    let mut index_counter = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    let mut on_stack = vec![false; n];
    let mut indices = vec![usize::MAX; n];
    let mut lowlinks = vec![usize::MAX; n];
    let mut component_id = vec![u64::MAX; n];
    let mut next_comp = 0u64;
    let mut comp_sizes: HashMap<u64, usize> = HashMap::new();

    // Use iterative DFS to avoid stack overflow on large graphs
    for start_node in 0..n {
        if indices[start_node] != usize::MAX { continue; }

        let mut dfs_stack: Vec<(usize, usize)> = vec![(start_node, 0)]; // (node, neighbor_idx)
        indices[start_node] = index_counter;
        lowlinks[start_node] = index_counter;
        index_counter += 1;
        stack.push(start_node);
        on_stack[start_node] = true;

        while let Some((v, ni)) = dfs_stack.last_mut() {
            let v = *v;
            if *ni < adj[v].len() {
                let w = adj[v][*ni];
                *ni += 1;
                if indices[w] == usize::MAX {
                    indices[w] = index_counter;
                    lowlinks[w] = index_counter;
                    index_counter += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    dfs_stack.push((w, 0));
                } else if on_stack[w] {
                    lowlinks[v] = lowlinks[v].min(indices[w]);
                }
            } else {
                // Done with v's neighbors
                if lowlinks[v] == indices[v] {
                    // v is root of an SCC
                    let cid = next_comp;
                    next_comp += 1;
                    let mut size = 0;
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack[w] = false;
                        component_id[w] = cid;
                        size += 1;
                        if w == v { break; }
                    }
                    comp_sizes.insert(cid, size);
                }
                let lv = lowlinks[v];
                dfs_stack.pop();
                if let Some((parent, _)) = dfs_stack.last() {
                    lowlinks[*parent] = lowlinks[*parent].min(lv);
                }
            }
        }
    }

    let largest = comp_sizes.values().copied().max().unwrap_or(0);

    let components: Vec<(Uuid, u64)> = node_ids.into_iter()
        .zip(component_id.into_iter())
        .collect();

    Ok(ComponentResult {
        components,
        component_count: comp_sizes.len(),
        largest_component_size: largest,
        duration_ms: start.elapsed().as_millis() as u64,
    })
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

    // ── WCC tests ────────────────────────────────────────────────────────

    #[test]
    fn wcc_empty_graph() {
        let (storage, _dir) = open_temp_db();
        let adj = AdjacencyIndex::new();

        let result = weakly_connected_components(&storage, &adj).unwrap();
        assert!(result.components.is_empty());
        assert_eq!(result.component_count, 0);
        assert_eq!(result.largest_component_size, 0);
    }

    #[test]
    fn wcc_single_component() {
        // Path: 0 -> 1 -> 2 -> 3 (all weakly connected)
        let nodes: Vec<Node> = (0..4).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let result = weakly_connected_components(&storage, &adj).unwrap();
        assert_eq!(result.component_count, 1);
        assert_eq!(result.largest_component_size, 4);

        // All nodes should be in the same component
        let comp_map: HashMap<Uuid, u64> = result.components.into_iter().collect();
        let comp0 = comp_map[&ids[0]];
        for id in &ids {
            assert_eq!(comp_map[id], comp0, "all nodes should be in same WCC");
        }
    }

    #[test]
    fn wcc_two_components() {
        // Component A: 0 -> 1
        // Component B: 2 -> 3
        // (disconnected)
        let nodes: Vec<Node> = (0..4).map(|i| make_node(0.1 * i as f32, 0.05)).collect();
        let edges = vec![(0, 1, 1.0), (2, 3, 1.0)];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let result = weakly_connected_components(&storage, &adj).unwrap();
        assert_eq!(result.component_count, 2);
        assert_eq!(result.largest_component_size, 2);

        let comp_map: HashMap<Uuid, u64> = result.components.into_iter().collect();
        assert_eq!(comp_map[&ids[0]], comp_map[&ids[1]]);
        assert_eq!(comp_map[&ids[2]], comp_map[&ids[3]]);
        assert_ne!(comp_map[&ids[0]], comp_map[&ids[2]]);
    }

    #[test]
    fn wcc_isolated_nodes() {
        // 4 isolated nodes = 4 components
        let nodes: Vec<Node> = (0..4).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let (storage, adj, _ids, _dir) = build_graph(&nodes, &[]);

        let result = weakly_connected_components(&storage, &adj).unwrap();
        assert_eq!(result.component_count, 4);
        assert_eq!(result.largest_component_size, 1);
    }

    // ── SCC tests ────────────────────────────────────────────────────────

    #[test]
    fn scc_empty_graph() {
        let (storage, _dir) = open_temp_db();
        let adj = AdjacencyIndex::new();

        let result = strongly_connected_components(&storage, &adj).unwrap();
        assert!(result.components.is_empty());
        assert_eq!(result.component_count, 0);
    }

    #[test]
    fn scc_cycle_is_single_component() {
        // Cycle: 0 -> 1 -> 2 -> 0 (all mutually reachable)
        let nodes: Vec<Node> = (0..3).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let result = strongly_connected_components(&storage, &adj).unwrap();
        assert_eq!(result.component_count, 1);
        assert_eq!(result.largest_component_size, 3);

        let comp_map: HashMap<Uuid, u64> = result.components.into_iter().collect();
        let c0 = comp_map[&ids[0]];
        assert_eq!(comp_map[&ids[1]], c0);
        assert_eq!(comp_map[&ids[2]], c0);
    }

    #[test]
    fn scc_path_has_n_components() {
        // Path: 0 -> 1 -> 2 -> 3 (no back edges => each node is its own SCC)
        let nodes: Vec<Node> = (0..4).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
        let (storage, adj, _ids, _dir) = build_graph(&nodes, &edges);

        let result = strongly_connected_components(&storage, &adj).unwrap();
        assert_eq!(result.component_count, 4, "each node in a DAG is its own SCC");
        assert_eq!(result.largest_component_size, 1);

        // All nodes should be in different components
        let comp_map: HashMap<Uuid, u64> = result.components.into_iter().collect();
        let comps: std::collections::HashSet<u64> = comp_map.values().copied().collect();
        assert_eq!(comps.len(), 4);
    }

    #[test]
    fn scc_two_cycles_with_bridge() {
        // Cycle A: 0 -> 1 -> 0
        // Cycle B: 2 -> 3 -> 2
        // Bridge: 1 -> 2 (one-way, so cycles remain separate SCCs)
        let nodes: Vec<Node> = (0..4).map(|i| make_node(0.05 * i as f32, 0.05)).collect();
        let edges = vec![
            (0, 1, 1.0), (1, 0, 1.0),  // Cycle A
            (2, 3, 1.0), (3, 2, 1.0),  // Cycle B
            (1, 2, 1.0),               // One-way bridge
        ];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let result = strongly_connected_components(&storage, &adj).unwrap();
        assert_eq!(result.component_count, 2, "two separate cycles = 2 SCCs");
        assert_eq!(result.largest_component_size, 2);

        let comp_map: HashMap<Uuid, u64> = result.components.into_iter().collect();
        assert_eq!(comp_map[&ids[0]], comp_map[&ids[1]]);
        assert_eq!(comp_map[&ids[2]], comp_map[&ids[3]]);
        assert_ne!(comp_map[&ids[0]], comp_map[&ids[2]]);
    }
}
