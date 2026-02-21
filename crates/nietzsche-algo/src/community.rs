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

    // ── Louvain tests ────────────────────────────────────────────────────

    #[test]
    fn louvain_empty_graph() {
        let (storage, _dir) = open_temp_db();
        let adj = AdjacencyIndex::new();
        let config = LouvainConfig::default();

        let result = louvain(&storage, &adj, &config).unwrap();
        assert!(result.communities.is_empty());
        assert_eq!(result.community_count, 0);
        assert_eq!(result.modularity, 0.0);
    }

    #[test]
    fn louvain_single_node() {
        let node = make_node(0.1, 0.1);
        let (storage, adj, ids, _dir) = build_graph(&[node], &[]);
        let config = LouvainConfig::default();

        let result = louvain(&storage, &adj, &config).unwrap();
        assert_eq!(result.communities.len(), 1);
        assert_eq!(result.communities[0].0, ids[0]);
        assert_eq!(result.community_count, 1);
    }

    #[test]
    fn louvain_reduces_community_count() {
        // Two dense cliques with a weak bridge.
        // We test that Louvain merges nodes into fewer communities than n,
        // and that the resulting modularity is non-negative.
        let nodes: Vec<Node> = (0..6).map(|i| make_node(0.05 * i as f32, 0.05)).collect();
        let edges = vec![
            // Clique A: fully connected 0-1-2
            (0, 1, 1.0), (1, 0, 1.0),
            (0, 2, 1.0), (2, 0, 1.0),
            (1, 2, 1.0), (2, 1, 1.0),
            // Clique B: fully connected 3-4-5
            (3, 4, 1.0), (4, 3, 1.0),
            (3, 5, 1.0), (5, 3, 1.0),
            (4, 5, 1.0), (5, 4, 1.0),
            // Weak bridge
            (2, 3, 0.1),
        ];
        let (storage, adj, _ids, _dir) = build_graph(&nodes, &edges);

        let config = LouvainConfig::default();
        let result = louvain(&storage, &adj, &config).unwrap();

        // The algorithm should find structure (fewer communities than nodes)
        assert!(
            result.community_count < 6,
            "expected Louvain to merge into fewer than 6 communities, got {}",
            result.community_count
        );
        // Modularity should be non-negative for a graph with community structure
        assert!(
            result.modularity >= 0.0,
            "modularity should be non-negative, got {}",
            result.modularity
        );
        // All 6 nodes should be assigned
        assert_eq!(result.communities.len(), 6);
    }

    #[test]
    fn louvain_all_nodes_assigned() {
        let nodes: Vec<Node> = (0..4).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let edges = vec![(0, 1, 1.0), (2, 3, 1.0)];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let config = LouvainConfig::default();
        let result = louvain(&storage, &adj, &config).unwrap();

        // Every node should appear exactly once
        let assigned_ids: std::collections::HashSet<Uuid> =
            result.communities.iter().map(|(id, _)| *id).collect();
        for id in &ids {
            assert!(assigned_ids.contains(id), "node {id} not assigned a community");
        }
    }

    // ── Label Propagation tests ──────────────────────────────────────────

    #[test]
    fn label_propagation_empty_graph() {
        let (storage, _dir) = open_temp_db();
        let adj = AdjacencyIndex::new();

        let result = label_propagation(&storage, &adj, 10).unwrap();
        assert!(result.labels.is_empty());
        assert_eq!(result.community_count, 0);
    }

    #[test]
    fn label_propagation_single_node() {
        let node = make_node(0.1, 0.1);
        let (storage, adj, _ids, _dir) = build_graph(&[node], &[]);

        let result = label_propagation(&storage, &adj, 10).unwrap();
        assert_eq!(result.labels.len(), 1);
        assert_eq!(result.community_count, 1);
    }

    #[test]
    fn label_propagation_connected_pair() {
        // Two nodes connected bidirectionally should end up in the same community
        let nodes: Vec<Node> = (0..2).map(|i| make_node(0.1 * i as f32, 0.1)).collect();
        let edges = vec![(0, 1, 1.0), (1, 0, 1.0)];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let result = label_propagation(&storage, &adj, 20).unwrap();
        let label_map: HashMap<Uuid, u64> = result.labels.into_iter().collect();
        assert_eq!(
            label_map[&ids[0]], label_map[&ids[1]],
            "connected pair should share the same label"
        );
    }

    #[test]
    fn label_propagation_disconnected_components() {
        // Two disconnected pairs: (0,1) and (2,3)
        let nodes: Vec<Node> = (0..4).map(|i| make_node(0.1 * i as f32, 0.05)).collect();
        let edges = vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)];
        let (storage, adj, ids, _dir) = build_graph(&nodes, &edges);

        let result = label_propagation(&storage, &adj, 20).unwrap();
        let label_map: HashMap<Uuid, u64> = result.labels.into_iter().collect();

        // Within each pair, labels should match
        assert_eq!(label_map[&ids[0]], label_map[&ids[1]]);
        assert_eq!(label_map[&ids[2]], label_map[&ids[3]]);
        // Between pairs, labels should differ
        assert_ne!(label_map[&ids[0]], label_map[&ids[2]]);
    }
}
