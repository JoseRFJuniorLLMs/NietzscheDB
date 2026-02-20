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
