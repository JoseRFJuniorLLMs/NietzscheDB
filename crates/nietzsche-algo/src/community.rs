//! Community detection: Louvain modularity optimization + Label Propagation.

use std::collections::HashMap;
use std::time::Instant;
use uuid::Uuid;
use nietzsche_graph::{GraphStorage, AdjacencyIndex, GraphError};

// ── Louvain ──────────────────────────────────────────────────────────────────

pub struct LouvainConfig {
    pub max_iterations: usize,
    pub resolution: f64,
}

impl Default for LouvainConfig {
    fn default() -> Self {
        Self { max_iterations: 10, resolution: 1.0 }
    }
}

pub struct LouvainResult {
    pub communities: Vec<(Uuid, u64)>,
    pub modularity: f64,
    pub community_count: usize,
    pub iterations: usize,
    pub duration_ms: u64,
}

/// Louvain community detection.
///
/// Phase 1: Greedily moves each node to the neighboring community that yields
///          the best modularity gain. Repeats until no improvement.
pub fn louvain(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    config: &LouvainConfig,
) -> Result<LouvainResult, GraphError> {
    let start = Instant::now();

    let node_ids: Vec<Uuid> = storage
        .scan_nodes_meta()?
        .into_iter()
        .map(|n| n.id)
        .collect();
    let n = node_ids.len();
    if n == 0 {
        return Ok(LouvainResult {
            communities: vec![], modularity: 0.0, community_count: 0, iterations: 0, duration_ms: 0,
        });
    }

    let id_to_idx: HashMap<Uuid, usize> = node_ids.iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    // Each node starts in its own community
    let mut community: Vec<u64> = (0..n as u64).collect();

    // Compute total edge weight (m) and per-node weighted degree (k_i)
    let mut k = vec![0.0_f64; n];
    let mut total_weight = 0.0_f64;

    for (i, id) in node_ids.iter().enumerate() {
        for entry in adjacency.entries_out(id) {
            let w = entry.weight as f64;
            k[i] += w;
            total_weight += w;
        }
        for entry in adjacency.entries_in(id) {
            k[i] += entry.weight as f64;
        }
    }

    if total_weight == 0.0 {
        total_weight = 1.0; // avoid division by zero
    }

    let mut iterations = 0;

    for _ in 0..config.max_iterations {
        iterations += 1;
        let mut improved = false;

        for i in 0..n {
            let id = &node_ids[i];
            let current_comm = community[i];

            // Count edges to each neighboring community
            let mut comm_weights: HashMap<u64, f64> = HashMap::new();

            for entry in adjacency.entries_out(id) {
                if let Some(&j) = id_to_idx.get(&entry.neighbor_id) {
                    *comm_weights.entry(community[j]).or_default() += entry.weight as f64;
                }
            }
            for entry in adjacency.entries_in(id) {
                if let Some(&j) = id_to_idx.get(&entry.neighbor_id) {
                    *comm_weights.entry(community[j]).or_default() += entry.weight as f64;
                }
            }

            // Find the community with the best modularity gain
            let mut best_comm = current_comm;
            let mut best_gain = 0.0_f64;

            for (&c, &w_ic) in &comm_weights {
                if c == current_comm { continue; }
                // Simplified modularity gain
                let sigma_c: f64 = community.iter()
                    .enumerate()
                    .filter(|(_, &cc)| cc == c)
                    .map(|(j, _)| k[j])
                    .sum();
                let gain = w_ic / total_weight
                    - config.resolution * k[i] * sigma_c / (2.0 * total_weight * total_weight);
                if gain > best_gain {
                    best_gain = gain;
                    best_comm = c;
                }
            }

            if best_comm != current_comm {
                community[i] = best_comm;
                improved = true;
            }
        }

        if !improved { break; }
    }

    // Renumber communities to be contiguous
    let mut comm_map: HashMap<u64, u64> = HashMap::new();
    let mut next_id = 0u64;
    for c in &mut community {
        let entry = comm_map.entry(*c).or_insert_with(|| { let id = next_id; next_id += 1; id });
        *c = *entry;
    }

    let community_count = comm_map.len();

    // Compute modularity Q
    let mut q = 0.0_f64;
    for (i, id) in node_ids.iter().enumerate() {
        for entry in adjacency.entries_out(id) {
            if let Some(&j) = id_to_idx.get(&entry.neighbor_id) {
                if community[i] == community[j] {
                    q += (entry.weight as f64) - k[i] * k[j] / (2.0 * total_weight);
                }
            }
        }
    }
    q /= 2.0 * total_weight;

    let communities: Vec<(Uuid, u64)> = node_ids.into_iter()
        .zip(community.into_iter())
        .collect();

    Ok(LouvainResult {
        communities,
        modularity: q,
        community_count,
        iterations,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}

// ── Label Propagation ────────────────────────────────────────────────────────

pub struct LabelPropResult {
    pub labels: Vec<(Uuid, u64)>,
    pub community_count: usize,
    pub iterations: usize,
    pub duration_ms: u64,
}

/// Label Propagation community detection.
///
/// Each node adopts the most frequent label among its neighbors.
pub fn label_propagation(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    max_iterations: usize,
) -> Result<LabelPropResult, GraphError> {
    let start = Instant::now();

    let node_ids: Vec<Uuid> = storage
        .scan_nodes_meta()?
        .into_iter()
        .map(|n| n.id)
        .collect();
    let n = node_ids.len();
    if n == 0 {
        return Ok(LabelPropResult { labels: vec![], community_count: 0, iterations: 0, duration_ms: 0 });
    }

    let id_to_idx: HashMap<Uuid, usize> = node_ids.iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    let mut labels: Vec<u64> = (0..n as u64).collect();
    let mut iterations = 0;

    for _ in 0..max_iterations {
        iterations += 1;
        let mut changed = false;

        for i in 0..n {
            let id = &node_ids[i];
            let mut label_counts: HashMap<u64, usize> = HashMap::new();

            // Count neighbor labels (both directions)
            for entry in adjacency.entries_out(id) {
                if let Some(&j) = id_to_idx.get(&entry.neighbor_id) {
                    *label_counts.entry(labels[j]).or_default() += 1;
                }
            }
            for entry in adjacency.entries_in(id) {
                if let Some(&j) = id_to_idx.get(&entry.neighbor_id) {
                    *label_counts.entry(labels[j]).or_default() += 1;
                }
            }

            // Pick the most frequent label
            if let Some((&best_label, _)) = label_counts.iter().max_by_key(|(_, &c)| c) {
                if labels[i] != best_label {
                    labels[i] = best_label;
                    changed = true;
                }
            }
        }

        if !changed { break; }
    }

    // Count communities
    let mut unique: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for &l in &labels { unique.insert(l); }

    let result: Vec<(Uuid, u64)> = node_ids.into_iter()
        .zip(labels.into_iter())
        .collect();

    Ok(LabelPropResult {
        labels: result,
        community_count: unique.len(),
        iterations,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}
