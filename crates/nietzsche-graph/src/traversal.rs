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
// Based on NietzscheDB's get_visited_list_from_pool pattern (hnsw/graph_layers.rs).
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
///
/// ## BUG A optimized
/// Uses `get_node_meta()` (~100 bytes) for the energy gate check instead of
/// `get_node()` (~24 KB). This gives 10–25× speedup in BFS traversal by
/// avoiding unnecessary embedding deserialization.
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

            // Energy gate — load ONLY NodeMeta (~100 bytes, no embedding)
            match storage.get_node_meta(&neighbor_id)? {
                Some(m) if m.energy >= config.energy_min => {
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
///
/// ## BUG A optimized
/// Energy filter uses `get_node_meta()` (~100 bytes). The embedding is loaded
/// via `get_embedding()` only for neighbours that pass the energy gate.
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

        // Load current node's embedding for distance calculation
        let node_emb = match storage.get_embedding(&node_id)? {
            Some(e) => e,
            None    => continue,
        };

        for neighbor_id in adjacency.neighbors_out(&node_id) {
            // Energy gate — NodeMeta only (~100 bytes)
            let neighbor_meta = match storage.get_node_meta(&neighbor_id)? {
                Some(m) => m,
                None    => continue,
            };
            if neighbor_meta.energy < config.energy_min {
                continue;
            }

            // Only load embedding for neighbours that pass the energy gate
            let neighbor_emb = match storage.get_embedding(&neighbor_id)? {
                Some(e) => e,
                None    => continue,
            };

            let edge_cost = node_emb.distance(&neighbor_emb);
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

        let node_emb = match storage.get_embedding(&node_id)? {
            Some(e) => e,
            None    => continue,
        };

        for neighbor_id in adjacency.neighbors_out(&node_id) {
            // Energy gate — NodeMeta only
            let neighbor_meta = match storage.get_node_meta(&neighbor_id)? {
                Some(m) => m,
                None    => continue,
            };
            if neighbor_meta.energy < config.energy_min {
                continue;
            }

            let neighbor_emb = match storage.get_embedding(&neighbor_id)? {
                Some(e) => e,
                None    => continue,
            };

            let edge_cost = node_emb.distance(&neighbor_emb);
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

        // Build (candidate, weight) pairs — uses NodeMeta only (no embedding).
        // Arousal amplifies energy_bias: heat travels faster through
        // emotionally charged memories than neutral facts.
        let mut candidates: Vec<(Uuid, f64)> = Vec::with_capacity(entries.len());
        for entry in &entries {
            if let Some(meta) = storage.get_node_meta(&entry.neighbor_id)? {
                let effective_bias = crate::valence::arousal_modulated_bias(
                    config.energy_bias,
                    meta.arousal,
                );
                let w = (entry.weight as f64)
                    * ((meta.energy as f64 * effective_bias as f64).exp());
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
// Hyperbolic Greedy Router
// ─────────────────────────────────────────────

/// Configuration for hyperbolic greedy routing.
#[derive(Debug, Clone)]
pub struct GreedyRouteConfig {
    /// Skip nodes whose `energy` is below this threshold.
    pub energy_min: f32,
    /// Maximum hops before giving up (prevents infinite loops).
    pub max_hops: usize,
    /// If true, fall back to A* when greedy gets stuck in a local minimum.
    pub astar_fallback: bool,
    /// Maximum nodes to explore during A* fallback (bounds memory/time).
    pub astar_max_nodes: usize,
}

impl Default for GreedyRouteConfig {
    fn default() -> Self {
        Self {
            energy_min: 0.0,
            max_hops: 200,
            astar_fallback: true,
            astar_max_nodes: 5_000,
        }
    }
}

/// Result of a greedy routing attempt.
#[derive(Debug, Clone)]
pub struct RouteResult {
    /// Ordered list of node IDs from start to closest-to-target.
    pub path: Vec<Uuid>,
    /// Whether the route reached the target node exactly.
    pub reached_target: bool,
    /// Whether A* fallback was used (greedy hit a local minimum).
    pub used_fallback: bool,
    /// Poincaré distance from the last node in the path to the target.
    pub final_distance: f64,
    /// Number of hops taken.
    pub hops: usize,
    /// Total hyperbolic distance traversed along the path.
    pub path_distance: f64,
}

/// Greedy hyperbolic routing from `start` toward `target`.
///
/// At each step, moves to the outgoing neighbour that minimises the
/// Poincaré distance to `target`. Terminates when:
///
/// - The target is reached (success).
/// - No neighbour is closer than the current node (local minimum).
/// - `max_hops` is exceeded.
///
/// When a local minimum is hit and `astar_fallback` is enabled, the
/// router switches to A* with `h(n) = poincare_distance(n, target)` to
/// find the optimal path from the stuck node to the target.
///
/// ## Complexity
///
/// - Greedy phase: `O(hops × avg_degree)` — typically `O(log N)` hops
///   in well-formed hyperbolic graphs.
/// - A* fallback: `O(N log N)` worst case, bounded by `astar_max_nodes`.
///
/// ## BUG A optimized
///
/// Energy filter uses `get_node_meta()` (~100 bytes). Embeddings are loaded
/// via `get_embedding()` only for neighbours that pass the energy gate.
pub fn greedy_route(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    start: Uuid,
    target: Uuid,
    config: &GreedyRouteConfig,
) -> Result<RouteResult, GraphError> {
    // Trivial case: start == target
    if start == target {
        return Ok(RouteResult {
            path: vec![start],
            reached_target: true,
            used_fallback: false,
            final_distance: 0.0,
            hops: 0,
            path_distance: 0.0,
        });
    }

    // Load target embedding once (shared across all distance comparisons)
    let target_emb = storage.get_embedding(&target)?
        .ok_or_else(|| GraphError::Storage(format!("target node {target} has no embedding")))?;

    let mut path = vec![start];
    let mut visited = acquire_visited();
    visited.insert(start);
    let mut current = start;
    let mut path_distance = 0.0;

    // Load current node's distance to target
    let current_emb = storage.get_embedding(&current)?
        .ok_or_else(|| GraphError::Storage(format!("start node {current} has no embedding")))?;
    let mut current_dist = current_emb.distance(&target_emb);

    for _hop in 0..config.max_hops {
        if current == target {
            release_visited(visited);
            return Ok(RouteResult {
                path,
                reached_target: true,
                used_fallback: false,
                final_distance: 0.0,
                hops: _hop,
                path_distance,
            });
        }

        // Find the unvisited neighbour closest to target
        let mut best_id: Option<Uuid> = None;
        let mut best_dist = current_dist;
        let mut best_edge_cost = 0.0;

        let current_emb = storage.get_embedding(&current)?
            .ok_or_else(|| GraphError::Storage(format!("node {current} has no embedding")))?;

        for neighbor_id in adjacency.neighbors_out(&current) {
            if visited.contains(&neighbor_id) {
                continue;
            }

            // Energy gate — NodeMeta only (~100 bytes)
            match storage.get_node_meta(&neighbor_id)? {
                Some(m) if m.energy >= config.energy_min => {}
                _ => continue,
            }

            let neighbor_emb = match storage.get_embedding(&neighbor_id)? {
                Some(e) => e,
                None => continue,
            };

            let d = neighbor_emb.distance(&target_emb);
            if d < best_dist {
                best_dist = d;
                best_edge_cost = current_emb.distance(&neighbor_emb);
                best_id = Some(neighbor_id);
            }
        }

        match best_id {
            Some(next) => {
                visited.insert(next);
                path.push(next);
                path_distance += best_edge_cost;
                current = next;
                current_dist = best_dist;
            }
            None => {
                // Local minimum — no neighbour is closer
                break;
            }
        }
    }

    // Did we reach the target via greedy?
    if current == target {
        let hops = path.len() - 1;
        release_visited(visited);
        return Ok(RouteResult {
            path,
            reached_target: true,
            used_fallback: false,
            final_distance: 0.0,
            hops,
            path_distance,
        });
    }

    // ── A* fallback from the stuck node ────────────────────
    if config.astar_fallback {
        if let Some(astar_path) = astar_hyperbolic(
            storage,
            adjacency,
            current,
            target,
            &target_emb,
            config.energy_min,
            config.astar_max_nodes,
        )? {
            // Splice: greedy prefix + A* suffix (skip current which is already in path)
            let greedy_hops = path.len() - 1;
            for &node in astar_path.iter().skip(1) {
                path.push(node);
            }

            // Compute total path distance
            let mut total = 0.0;
            for w in path.windows(2) {
                if let (Some(a), Some(b)) = (
                    storage.get_embedding(&w[0])?,
                    storage.get_embedding(&w[1])?,
                ) {
                    total += a.distance(&b);
                }
            }

            release_visited(visited);
            return Ok(RouteResult {
                hops: path.len() - 1,
                reached_target: *path.last().unwrap() == target,
                used_fallback: true,
                final_distance: if *path.last().unwrap() == target { 0.0 } else { current_dist },
                path,
                path_distance: total,
            });
        }
    }

    // No fallback or A* also failed
    let final_dist = current_dist;
    let hops = path.len() - 1;
    release_visited(visited);
    Ok(RouteResult {
        path,
        reached_target: false,
        used_fallback: false,
        final_distance: final_dist,
        hops,
        path_distance,
    })
}

/// Navigate toward a **target embedding** (not a specific node).
///
/// Same as [`greedy_route`] but the target is specified as coordinates in
/// the Poincaré ball rather than a node ID. Terminates when no neighbour
/// reduces the distance (local minimum). No A* fallback since there is
/// no specific destination node.
///
/// Useful for semantic navigation: "find the closest reachable node to
/// this embedding".
pub fn greedy_route_to_embedding(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    start: Uuid,
    target_coords: &[f32],
    config: &GreedyRouteConfig,
) -> Result<RouteResult, GraphError> {
    use crate::model::PoincareVector;

    let target_emb = PoincareVector::new(target_coords.to_vec());

    let mut path = vec![start];
    let mut visited = acquire_visited();
    visited.insert(start);
    let mut current = start;
    let mut path_distance = 0.0;

    let current_emb = storage.get_embedding(&current)?
        .ok_or_else(|| GraphError::Storage(format!("start node {current} has no embedding")))?;
    let mut current_dist = current_emb.distance(&target_emb);

    for _hop in 0..config.max_hops {
        let mut best_id: Option<Uuid> = None;
        let mut best_dist = current_dist;
        let mut best_edge_cost = 0.0;

        let cur_emb = storage.get_embedding(&current)?
            .ok_or_else(|| GraphError::Storage(format!("node {current} has no embedding")))?;

        for neighbor_id in adjacency.neighbors_out(&current) {
            if visited.contains(&neighbor_id) {
                continue;
            }

            match storage.get_node_meta(&neighbor_id)? {
                Some(m) if m.energy >= config.energy_min => {}
                _ => continue,
            }

            let neighbor_emb = match storage.get_embedding(&neighbor_id)? {
                Some(e) => e,
                None => continue,
            };

            let d = neighbor_emb.distance(&target_emb);
            if d < best_dist {
                best_dist = d;
                best_edge_cost = cur_emb.distance(&neighbor_emb);
                best_id = Some(neighbor_id);
            }
        }

        match best_id {
            Some(next) => {
                visited.insert(next);
                path.push(next);
                path_distance += best_edge_cost;
                current = next;
                current_dist = best_dist;
            }
            None => break,
        }
    }

    let hops = path.len() - 1;
    release_visited(visited);
    Ok(RouteResult {
        path,
        reached_target: false, // embedding target — never "reached"
        used_fallback: false,
        final_distance: current_dist,
        hops,
        path_distance,
    })
}

/// A* search with Poincaré distance heuristic.
///
/// Internal helper for the greedy router fallback. Uses
/// `h(n) = poincare_distance(n, target)` as the admissible heuristic
/// (always underestimates because graph paths are at least as long as
/// the geodesic distance).
fn astar_hyperbolic(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    start: Uuid,
    target: Uuid,
    target_emb: &crate::model::PoincareVector,
    energy_min: f32,
    max_nodes: usize,
) -> Result<Option<Vec<Uuid>>, GraphError> {
    // min-heap: (f = g + h, g, node_id)
    let mut heap: BinaryHeap<(Reverse<OrderedFloat<f64>>, OrderedFloat<f64>, Uuid)> = BinaryHeap::new();
    let mut g_score: HashMap<Uuid, f64> = HashMap::new();
    let mut prev: HashMap<Uuid, Uuid> = HashMap::new();

    let start_emb = match storage.get_embedding(&start)? {
        Some(e) => e,
        None => return Ok(None),
    };
    let h_start = start_emb.distance(target_emb);

    g_score.insert(start, 0.0);
    heap.push((Reverse(OrderedFloat(h_start)), OrderedFloat(0.0), start));

    let mut expanded = 0usize;

    while let Some((_, OrderedFloat(g), node_id)) = heap.pop() {
        if node_id == target {
            // Reconstruct path
            let mut path = vec![target];
            let mut cur = target;
            while let Some(&p) = prev.get(&cur) {
                path.push(p);
                cur = p;
            }
            path.reverse();
            return Ok(Some(path));
        }

        expanded += 1;
        if expanded > max_nodes {
            break;
        }

        // Stale entry
        if g > *g_score.get(&node_id).unwrap_or(&f64::INFINITY) {
            continue;
        }

        let node_emb = match storage.get_embedding(&node_id)? {
            Some(e) => e,
            None => continue,
        };

        for neighbor_id in adjacency.neighbors_out(&node_id) {
            // Energy gate
            match storage.get_node_meta(&neighbor_id)? {
                Some(m) if m.energy >= energy_min => {}
                _ => continue,
            }

            let neighbor_emb = match storage.get_embedding(&neighbor_id)? {
                Some(e) => e,
                None => continue,
            };

            let edge_cost = node_emb.distance(&neighbor_emb);
            let tentative_g = g + edge_cost;

            let best = g_score.entry(neighbor_id).or_insert(f64::INFINITY);
            if tentative_g < *best {
                *best = tentative_g;
                prev.insert(neighbor_id, node_id);
                let h = neighbor_emb.distance(target_emb);
                let f = tentative_g + h;
                heap.push((Reverse(OrderedFloat(f)), OrderedFloat(tentative_g), neighbor_id));
            }
        }
    }

    Ok(None) // target not found within budget
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
            PoincareVector::new(vec![x as f32, 0.0]),
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

    // ── Greedy Router ──────────────────────────────────

    #[test]
    fn greedy_route_trivial_start_equals_target() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);

        let result = greedy_route(
            &storage, &adjacency, nodes[0].id, nodes[0].id,
            &GreedyRouteConfig::default(),
        ).unwrap();

        assert!(result.reached_target);
        assert_eq!(result.path, vec![nodes[0].id]);
        assert_eq!(result.hops, 0);
        assert_eq!(result.final_distance, 0.0);
    }

    #[test]
    fn greedy_route_linear_chain_reaches_end() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);

        let result = greedy_route(
            &storage, &adjacency, nodes[0].id, nodes[3].id,
            &GreedyRouteConfig::default(),
        ).unwrap();

        assert!(result.reached_target);
        assert_eq!(result.path.len(), 4);
        assert_eq!(result.path[0], nodes[0].id);
        assert_eq!(*result.path.last().unwrap(), nodes[3].id);
        assert!(result.path_distance > 0.0);
    }

    #[test]
    fn greedy_route_unreachable_reverse_direction() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);

        // d → a is impossible in a directed a→b→c→d chain
        let result = greedy_route(
            &storage, &adjacency, nodes[3].id, nodes[0].id,
            &GreedyRouteConfig { astar_fallback: false, ..Default::default() },
        ).unwrap();

        assert!(!result.reached_target);
        assert_eq!(result.path, vec![nodes[3].id]); // stuck at start
    }

    #[test]
    fn greedy_route_skips_low_energy() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let a = node_at(0.1);
        let mut b = node_at(0.2); b.energy = 0.0; // dead node
        let c = node_at(0.3);

        for n in [&a, &b, &c] { storage.put_node(n).unwrap(); }

        let ab = Edge::association(a.id, b.id, 0.9);
        let bc = Edge::association(b.id, c.id, 0.9);
        for e in [&ab, &bc] {
            storage.put_edge(e).unwrap();
            adjacency.add_edge(e);
        }

        let result = greedy_route(
            &storage, &adjacency, a.id, c.id,
            &GreedyRouteConfig { energy_min: 0.1, astar_fallback: false, ..Default::default() },
        ).unwrap();

        // b is blocked by energy gate → c unreachable
        assert!(!result.reached_target);
    }

    #[test]
    fn greedy_route_with_shortcut() {
        // Build: a → b → c → d, plus a direct shortcut a → d
        // Greedy should take the shortcut since d is closest to d
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let nodes: Vec<Node> = [0.1, 0.3, 0.5, 0.7]
            .iter()
            .map(|&x| node_at(x))
            .collect();

        for n in &nodes { storage.put_node(n).unwrap(); }

        // Chain edges
        for i in 0..3 {
            let e = Edge::association(nodes[i].id, nodes[i+1].id, 0.9);
            storage.put_edge(&e).unwrap();
            adjacency.add_edge(&e);
        }

        // Shortcut: a → d
        let shortcut = Edge::association(nodes[0].id, nodes[3].id, 0.9);
        storage.put_edge(&shortcut).unwrap();
        adjacency.add_edge(&shortcut);

        let result = greedy_route(
            &storage, &adjacency, nodes[0].id, nodes[3].id,
            &GreedyRouteConfig::default(),
        ).unwrap();

        assert!(result.reached_target);
        // Should take the shortcut: [a, d]
        assert_eq!(result.path.len(), 2);
        assert_eq!(result.hops, 1);
    }

    #[test]
    fn greedy_route_astar_fallback_finds_path() {
        // Build a graph where greedy gets stuck but A* succeeds:
        // a → b (b is closer to d than a)
        // b → c (c is FARTHER from d — greedy stops here)
        // c → d (but c is the only path to d)
        //
        // Layout on x-axis: a=0.1, b=0.4, d=0.6, c=0.8
        // Greedy from a: goes to b (closer to d). From b, c is farther from d.
        // Greedy stops at b. A* should find b→c→d.
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let a = node_at(0.1);
        let b = node_at(0.4);
        let c = node_at(0.8); // farther from d than b
        let d = node_at(0.6); // target

        for n in [&a, &b, &c, &d] { storage.put_node(n).unwrap(); }

        let edges = [
            Edge::association(a.id, b.id, 0.9),
            Edge::association(b.id, c.id, 0.9), // c is farther from d
            Edge::association(c.id, d.id, 0.9),
        ];
        for e in &edges {
            storage.put_edge(e).unwrap();
            adjacency.add_edge(e);
        }

        let result = greedy_route(
            &storage, &adjacency, a.id, d.id,
            &GreedyRouteConfig::default(),
        ).unwrap();

        assert!(result.reached_target, "A* fallback should find the path");
        assert!(result.used_fallback, "should have used A* fallback");
        assert_eq!(*result.path.last().unwrap(), d.id);
    }

    #[test]
    fn greedy_route_to_embedding_navigates_closer() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);

        // Target embedding is at x=0.45 — between node c (0.3) and d (0.4)
        let target_coords = vec![0.45_f32, 0.0];

        let result = greedy_route_to_embedding(
            &storage, &adjacency, nodes[0].id, &target_coords,
            &GreedyRouteConfig::default(),
        ).unwrap();

        // Should navigate toward the target, ending at d (0.4) which is closest
        assert!(result.path.len() > 1, "should move toward target");
        assert!(result.final_distance < 1.0, "should get close to target");
        // Last node should be d (x=0.4) — closest to 0.45
        assert_eq!(*result.path.last().unwrap(), nodes[3].id);
    }

    #[test]
    fn greedy_route_respects_max_hops() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);

        let result = greedy_route(
            &storage, &adjacency, nodes[0].id, nodes[3].id,
            &GreedyRouteConfig { max_hops: 1, astar_fallback: false, ..Default::default() },
        ).unwrap();

        // Should only take 1 hop: a → b
        assert!(result.path.len() <= 2);
    }

    #[test]
    fn greedy_route_reports_metrics() {
        let dir = tmp();
        let (storage, adjacency, nodes) = linear_chain(&dir);

        let result = greedy_route(
            &storage, &adjacency, nodes[0].id, nodes[3].id,
            &GreedyRouteConfig::default(),
        ).unwrap();

        assert!(result.reached_target);
        assert_eq!(result.hops, result.path.len() - 1);
        assert!(result.path_distance > 0.0);
        assert_eq!(result.final_distance, 0.0);
        assert!(!result.used_fallback);
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
