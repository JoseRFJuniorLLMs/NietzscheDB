use std::cell::RefCell;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::cmp::Reverse;

use ordered_float::OrderedFloat;
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;
use uuid::Uuid;

use crate::adjacency::AdjacencyIndex;
use crate::error::GraphError;
use crate::storage::GraphStorage;

// ─────────────────────────────────────────────
// Visited-list pool
// ─────────────────────────────────────────────

// Thread-local pool of reusable HashSet<Uuid> instances.
// Each BFS call acquires a pre-allocated set and returns it when done —
// eliminating ~1–3 µs of allocator overhead per traversal on a warm pool.
// Based on Qdrant's get_visited_list_from_pool pattern (hnsw/graph_layers.rs).
thread_local! {
    static VISITED_POOL: RefCell<Vec<HashSet<Uuid>>> = RefCell::new(Vec::with_capacity(4));
}

/// Acquire a cleared `HashSet<Uuid>` from the thread-local pool.
/// If the pool is empty, allocates a new set.
#[inline]
fn acquire_visited() -> HashSet<Uuid> {
    VISITED_POOL.with(|pool| {
        pool.borrow_mut().pop().unwrap_or_default()
    })
}

/// Return a visited set to the pool for reuse.
/// The set is cleared before storage.
#[inline]
fn release_visited(mut set: HashSet<Uuid>) {
    set.clear();
    VISITED_POOL.with(|pool| {
        let mut p = pool.borrow_mut();
        // Cap pool depth at 8 sets per thread to avoid unbounded growth.
        if p.len() < 8 {
            p.push(set);
        }
    });
}

// ─────────────────────────────────────────────
// Config structs
// ─────────────────────────────────────────────

/// Configuration for breadth-first traversal.
#[derive(Debug, Clone)]
pub struct BfsConfig {
    /// Maximum hop depth from the start node (inclusive).
    pub max_depth: usize,
    /// Skip nodes whose `energy` is below this threshold.
    pub energy_min: f32,
    /// Stop after visiting this many nodes.
    pub max_nodes: usize,
}

impl Default for BfsConfig {
    fn default() -> Self {
        Self { max_depth: 10, energy_min: 0.0, max_nodes: 1_000 }
    }
}

/// Configuration for Dijkstra / shortest-path traversal.
///
/// Edge weights are the Poincaré ball distance between the two node embeddings.
#[derive(Debug, Clone)]
pub struct DijkstraConfig {
    /// Skip nodes whose `energy` is below this threshold.
    pub energy_min: f32,
    /// Stop after settling this many nodes.
    pub max_nodes: usize,
    /// Do not explore beyond this hyperbolic distance from the start.
    pub max_distance: f64,
}

impl Default for DijkstraConfig {
    fn default() -> Self {
        Self { energy_min: 0.0, max_nodes: 1_000, max_distance: f64::INFINITY }
    }
}

/// Configuration for the energy-biased random walk.
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Maximum number of steps.
    pub steps: usize,
    /// Temperature parameter for energy-weighted sampling.
    /// Higher values make the walk greedier toward high-energy neighbours.
    pub energy_bias: f32,
    /// Optional RNG seed for reproducible walks.
    pub seed: Option<u64>,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self { steps: 50, energy_bias: 1.0, seed: None }
    }
}

// ─────────────────────────────────────────────
// BFS
// ─────────────────────────────────────────────

/// Breadth-first traversal from `start`.
///
/// Returns node IDs in discovery order. The start node is always included
/// regardless of its energy level. Nodes with `energy < config.energy_min`
/// are skipped (neither visited nor expanded).
pub fn bfs(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    start: Uuid,
    config: &BfsConfig,
) -> Result<Vec<Uuid>, GraphError> {
    // Pool: acquire pre-allocated HashSet (zero alloc on warm pool).
    let mut visited = acquire_visited();
    let mut order   = Vec::new();

    // queue entries: (node_id, depth)
    let mut queue: VecDeque<(Uuid, usize)> = VecDeque::new();
    queue.push_back((start, 0));
    visited.insert(start);

    while let Some((node_id, depth)) = queue.pop_front() {
        if order.len() >= config.max_nodes {
            break;
        }
        order.push(node_id);

        if depth >= config.max_depth {
            continue;
        }

        for neighbor_id in adjacency.neighbors_out(&node_id) {
            if visited.contains(&neighbor_id) {
                continue;
            }
            visited.insert(neighbor_id);

            // Energy gate — load node only to check energy
            match storage.get_node(&neighbor_id)? {
                Some(n) if n.energy >= config.energy_min => {
                    queue.push_back((neighbor_id, depth + 1));
                }
                _ => {} // below threshold or missing — skip
            }
        }
    }

    // Return the set to the pool for the next BFS call on this thread.
    release_visited(visited);
    Ok(order)
}

// ─────────────────────────────────────────────
// Dijkstra
// ─────────────────────────────────────────────

/// Dijkstra's algorithm over the hyperbolic graph.
///
/// Edge cost is the **Poincaré ball distance** between the embeddings of
/// the two endpoint nodes. Returns a map `node_id → distance_from_start`
/// for every reachable node within the config constraints.
pub fn dijkstra(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    start: Uuid,
    config: &DijkstraConfig,
) -> Result<HashMap<Uuid, f64>, GraphError> {
    // min-heap: (distance, node_id)
    let mut heap: BinaryHeap<(Reverse<OrderedFloat<f64>>, Uuid)> = BinaryHeap::new();
    // Dijkstra uses `dist` as its settled-set — no separate HashSet needed.
    let mut dist: HashMap<Uuid, f64> = HashMap::new();

    dist.insert(start, 0.0);
    heap.push((Reverse(OrderedFloat(0.0)), start));

    while let Some((Reverse(OrderedFloat(d)), node_id)) = heap.pop() {
        // All remaining entries have distance ≥ d — safe to prune
        if d > config.max_distance {
            break;
        }
        if dist.len() > config.max_nodes {
            break;
        }
        // Stale entry (a shorter path was already settled)
        if d > *dist.get(&node_id).unwrap_or(&f64::INFINITY) {
            continue;
        }

        let node = match storage.get_node(&node_id)? {
            Some(n) => n,
            None    => continue,
        };

        for neighbor_id in adjacency.neighbors_out(&node_id) {
            let neighbor = match storage.get_node(&neighbor_id)? {
                Some(n) => n,
                None    => continue,
            };
            if neighbor.energy < config.energy_min {
                continue;
            }

            let edge_cost = node.embedding.distance(&neighbor.embedding);
            let new_dist  = d + edge_cost;

            if new_dist > config.max_distance {
                continue;
            }

            let best = dist.entry(neighbor_id).or_insert(f64::INFINITY);
            if new_dist < *best {
                *best = new_dist;
                heap.push((Reverse(OrderedFloat(new_dist)), neighbor_id));
            }
        }
    }

    Ok(dist)
}

/// Shortest path from `start` to `end` using Poincaré distance.
///
/// Returns the ordered list of node IDs from `start` to `end` (inclusive),
/// or `None` if `end` is unreachable within the config constraints.
pub fn shortest_path(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    start: Uuid,
    end: Uuid,
    config: &DijkstraConfig,
) -> Result<Option<Vec<Uuid>>, GraphError> {
    let mut heap: BinaryHeap<(Reverse<OrderedFloat<f64>>, Uuid)> = BinaryHeap::new();
    let mut dist: HashMap<Uuid, f64> = HashMap::new();
    let mut prev: HashMap<Uuid, Uuid> = HashMap::new();

    dist.insert(start, 0.0);
    heap.push((Reverse(OrderedFloat(0.0)), start));

    while let Some((Reverse(OrderedFloat(d)), node_id)) = heap.pop() {
        if node_id == end {
            // Reconstruct path by following parent pointers
            let mut path = vec![end];
            let mut cur  = end;
            while let Some(&p) = prev.get(&cur) {
                path.push(p);
                cur = p;
            }
            path.reverse();
            return Ok(Some(path));
        }

        if d > config.max_distance {
            break;
        }
        if dist.len() > config.max_nodes {
            break;
        }
        if d > *dist.get(&node_id).unwrap_or(&f64::INFINITY) {
            continue;
        }

        let node = match storage.get_node(&node_id)? {
            Some(n) => n,
            None    => continue,
        };

        for neighbor_id in adjacency.neighbors_out(&node_id) {
            let neighbor = match storage.get_node(&neighbor_id)? {
                Some(n) => n,
                None    => continue,
            };
            if neighbor.energy < config.energy_min {
                continue;
            }

            let edge_cost = node.embedding.distance(&neighbor.embedding);
            let new_dist  = d + edge_cost;

            if new_dist > config.max_distance {
                continue;
            }

            let best = dist.entry(neighbor_id).or_insert(f64::INFINITY);
            if new_dist < *best {
                *best = new_dist;
                prev.insert(neighbor_id, node_id);
                heap.push((Reverse(OrderedFloat(new_dist)), neighbor_id));
            }
        }
    }

    Ok(None) // end not reached
}

// ─────────────────────────────────────────────
// Diffusion Walk
// ─────────────────────────────────────────────

/// Energy-biased random walk from `start`.
///
/// At each step the next node is sampled from the outgoing neighbours with
/// probability proportional to:
///
/// ```text
/// w_i = edge.weight * exp(neighbor.energy * energy_bias)
/// ```
///
/// The walk terminates early if the current node has no eligible neighbours.
/// Returns the sequence of visited node IDs (including `start`).
pub fn diffusion_walk(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    start: Uuid,
    config: &DiffusionConfig,
) -> Result<Vec<Uuid>, GraphError> {
    // Optionally seeded RNG for reproducibility
    let mut rng: Box<dyn rand::RngCore> = match config.seed {
        Some(s) => Box::new(rand::rngs::StdRng::seed_from_u64(s)),
        None    => Box::new(rand::thread_rng()),
    };

    let mut path    = vec![start];
    let mut current = start;

    for _ in 0..config.steps {
        let entries = adjacency.entries_out(&current);
        if entries.is_empty() {
            break;
        }

        // Build (candidate, weight) pairs
        let mut candidates: Vec<(Uuid, f64)> = Vec::with_capacity(entries.len());
        for entry in &entries {
            if let Some(neighbor) = storage.get_node(&entry.neighbor_id)? {
                let w = (entry.weight as f64)
                    * ((neighbor.energy as f64 * config.energy_bias as f64).exp());
                if w > 0.0 {
                    candidates.push((entry.neighbor_id, w));
                }
            }
        }

        if candidates.is_empty() {
            break;
        }

        let weights: Vec<f64> = candidates.iter().map(|(_, w)| *w).collect();
        let sampler = WeightedIndex::new(&weights)
            .map_err(|e| GraphError::Storage(format!("diffusion weight error: {e}")))?;

        let next = candidates[sampler.sample(&mut rng)].0;
        path.push(next);
        current = next;
    }

    Ok(path)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Edge, Node, PoincareVector};
    use tempfile::TempDir;

    // ── helpers ──────────────────────────────────────────

    fn tmp() -> TempDir { TempDir::new().unwrap() }

    fn open_storage(dir: &TempDir) -> GraphStorage {
        let p = dir.path().join("rocksdb");
        GraphStorage::open(p.to_str().unwrap()).unwrap()
    }

    fn node_at(x: f64) -> Node {
        Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, 0.0]),
            serde_json::json!({}),
        )
    }

    /// Build a linear chain: a → b → c → d
    /// Returns (storage, adjacency, [a, b, c, d])
    fn linear_chain(dir: &TempDir) -> (GraphStorage, AdjacencyIndex, Vec<Node>) {
        let mut storage = open_storage(dir);
        let adjacency   = AdjacencyIndex::new();

        let nodes: Vec<Node> = [0.1, 0.2, 0.3, 0.4]
            .iter()
            .map(|&x| node_at(x))
            .collect();

        for n in &nodes {
            storage.put_node(n).unwrap();
        }

        // a→b, b→c, c→d
        for i in 0..nodes.len() - 1 {
            let edge = Edge::association(nodes[i].id, nodes[i + 1].id, 0.9);
            storage.put_edge(&edge).unwrap();
            adjacency.add_edge(&edge);
        }

        (storage, adjacency, nodes)
    }

    // ── BFS ──────────────────────────────────────────────

    #[test]
    fn bfs_visits_all_nodes_in_linear_chain() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);
        let config = BfsConfig::default();

        let visited = bfs(&storage, &adjacency, nodes[0].id, &config).unwrap();
        assert_eq!(visited.len(), 4);
        assert_eq!(visited[0], nodes[0].id);
        assert_eq!(visited[3], nodes[3].id);
    }

    #[test]
    fn bfs_respects_max_depth() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);
        let config = BfsConfig { max_depth: 1, ..Default::default() };

        let visited = bfs(&storage, &adjacency, nodes[0].id, &config).unwrap();
        // depth=0: a, depth=1: b — then max_depth reached
        assert_eq!(visited.len(), 2);
    }

    #[test]
    fn bfs_respects_max_nodes() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);
        let config = BfsConfig { max_nodes: 2, ..Default::default() };

        let visited = bfs(&storage, &adjacency, nodes[0].id, &config).unwrap();
        assert_eq!(visited.len(), 2);
    }

    #[test]
    fn bfs_skips_low_energy_nodes() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        let a = node_at(0.1);                             // energy = 1.0 (default)
        let mut b = node_at(0.2); b.energy = 0.0;        // pruned
        let c = node_at(0.3);

        for n in [&a, &b, &c] { storage.put_node(n).unwrap(); }

        let ab = Edge::association(a.id, b.id, 0.9);
        let bc = Edge::association(b.id, c.id, 0.9);
        for e in [&ab, &bc] {
            storage.put_edge(e).unwrap();
            adjacency.add_edge(e);
        }

        // energy_min = 0.1 → b (energy=0) is skipped, c is not reachable
        let config = BfsConfig { energy_min: 0.1, ..Default::default() };
        let visited = bfs(&storage, &adjacency, a.id, &config).unwrap();
        assert_eq!(visited.len(), 1);
        assert!(!visited.contains(&c.id));
    }

    #[test]
    fn bfs_start_node_only_when_isolated() {
        let dir = tmp();
        let storage   = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        let node = node_at(0.5);
        storage.put_node(&node).unwrap();

        let visited = bfs(&storage, &adjacency, node.id, &BfsConfig::default()).unwrap();
        assert_eq!(visited, vec![node.id]);
    }

    // ── Dijkstra ─────────────────────────────────────────

    #[test]
    fn dijkstra_start_has_zero_distance() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);

        let dist = dijkstra(&storage, &adjacency, nodes[0].id, &DijkstraConfig::default()).unwrap();
        assert_eq!(*dist.get(&nodes[0].id).unwrap(), 0.0);
    }

    #[test]
    fn dijkstra_distances_increase_along_chain() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);

        let dist = dijkstra(&storage, &adjacency, nodes[0].id, &DijkstraConfig::default()).unwrap();

        let d0 = dist[&nodes[0].id];
        let d1 = dist[&nodes[1].id];
        let d2 = dist[&nodes[2].id];
        let d3 = dist[&nodes[3].id];

        assert!(d0 < d1);
        assert!(d1 < d2);
        assert!(d2 < d3);
    }

    #[test]
    fn dijkstra_respects_max_distance() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);

        // Very small cutoff — only start node should appear
        let config = DijkstraConfig { max_distance: 1e-10, ..Default::default() };
        let dist = dijkstra(&storage, &adjacency, nodes[0].id, &config).unwrap();
        assert_eq!(dist.len(), 1);
        assert!(dist.contains_key(&nodes[0].id));
    }

    // ── Shortest path ─────────────────────────────────────

    #[test]
    fn shortest_path_linear_chain() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);

        let path = shortest_path(
            &storage,
            &adjacency,
            nodes[0].id,
            nodes[3].id,
            &DijkstraConfig::default(),
        )
        .unwrap()
        .expect("path should exist");

        assert_eq!(path.len(), 4);
        assert_eq!(path[0], nodes[0].id);
        assert_eq!(path[3], nodes[3].id);
    }

    #[test]
    fn shortest_path_returns_none_when_unreachable() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);

        // Reverse direction — no path from d to a in a directed chain
        let result = shortest_path(
            &storage,
            &adjacency,
            nodes[3].id,
            nodes[0].id,
            &DijkstraConfig::default(),
        )
        .unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn shortest_path_start_equals_end() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);

        let path = shortest_path(
            &storage,
            &adjacency,
            nodes[0].id,
            nodes[0].id,
            &DijkstraConfig::default(),
        )
        .unwrap()
        .expect("trivial path exists");

        assert_eq!(path, vec![nodes[0].id]);
    }

    // ── Diffusion walk ────────────────────────────────────

    #[test]
    fn diffusion_walk_starts_at_start_node() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);

        let path = diffusion_walk(
            &storage,
            &adjacency,
            nodes[0].id,
            &DiffusionConfig { seed: Some(42), ..Default::default() },
        )
        .unwrap();

        assert_eq!(path[0], nodes[0].id);
    }

    #[test]
    fn diffusion_walk_stays_within_graph() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);
        let ids: HashSet<Uuid> = nodes.iter().map(|n| n.id).collect();

        let path = diffusion_walk(
            &storage,
            &adjacency,
            nodes[0].id,
            &DiffusionConfig { steps: 20, seed: Some(7), ..Default::default() },
        )
        .unwrap();

        for id in &path {
            assert!(ids.contains(id), "walk escaped the graph");
        }
    }

    #[test]
    fn diffusion_walk_with_same_seed_is_deterministic() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);

        let cfg = DiffusionConfig { steps: 10, seed: Some(99), ..Default::default() };
        let p1 = diffusion_walk(&storage, &adjacency, nodes[0].id, &cfg).unwrap();
        let p2 = diffusion_walk(&storage, &adjacency, nodes[0].id, &cfg).unwrap();

        assert_eq!(p1, p2);
    }

    #[test]
    fn diffusion_walk_stops_at_sink_node() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        let a = node_at(0.1);
        let b = node_at(0.2); // sink — no outgoing edges

        for n in [&a, &b] { storage.put_node(n).unwrap(); }
        let e = Edge::association(a.id, b.id, 1.0);
        storage.put_edge(&e).unwrap();
        adjacency.add_edge(&e);

        let path = diffusion_walk(
            &storage,
            &adjacency,
            a.id,
            &DiffusionConfig { steps: 50, seed: Some(1), ..Default::default() },
        )
        .unwrap();

        // Walk reaches b (sink) and must stop there
        assert!(path.len() <= 3); // [a, b] — then stops
        assert_eq!(*path.last().unwrap(), b.id);
    }
}
