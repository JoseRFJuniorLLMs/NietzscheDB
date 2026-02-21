use std::collections::{HashMap, VecDeque};

use uuid::Uuid;
use nietzsche_graph::NodeMeta;

use super::shadow::ShadowGraph;
use crate::config::AgencyConfig;

/// Predicted energy change for a single node.
#[derive(Debug, Clone)]
pub struct EnergyChange {
    pub node_id: Uuid,
    pub before: f32,
    pub after: f32,
}

/// Result of a counterfactual simulation.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Nodes most affected, sorted by impact score descending.
    pub impact_scores: Vec<(Uuid, f64)>,
    /// Predicted energy changes.
    pub energy_changes: Vec<EnergyChange>,
    /// Predicted change in mean graph energy.
    pub mean_energy_delta: f32,
    /// Number of unique nodes within max_hops of the target.
    pub affected_radius: usize,
}

/// Simulate the impact of removing a node using BFS energy propagation.
///
/// The removed node's energy is "lost" from its neighbors proportionally
/// to edge weight / degree. The loss propagates outward with 0.5x decay
/// per hop, up to `max_hops`.
pub fn simulate_remove(
    shadow: &mut ShadowGraph,
    target_id: Uuid,
    config: &AgencyConfig,
) -> SimulationResult {
    let removed_energy = shadow
        .get_meta(&target_id)
        .map(|m| m.energy)
        .unwrap_or(0.0);

    // Collect direct neighbors before removal
    let neighbors = shadow.neighbors(&target_id);
    shadow.simulate_remove_node(target_id);

    // BFS impact propagation
    let mut impact: HashMap<Uuid, f64> = HashMap::new();
    let mut queue: VecDeque<(Uuid, f64, usize)> = VecDeque::new();

    for nid in &neighbors {
        let degree = shadow.degree_out(nid).max(1);
        let loss = removed_energy as f64 / degree as f64;
        queue.push_back((*nid, loss, 1));
    }

    while let Some((nid, loss, hop)) = queue.pop_front() {
        if hop > config.counterfactual_max_hops || loss < 1e-6 {
            continue;
        }

        let current = impact.entry(nid).or_insert(0.0);
        if *current >= loss {
            continue; // already accounted for with larger impact
        }
        *current = loss;

        for next_id in shadow.neighbors(&nid) {
            if next_id != target_id && !impact.contains_key(&next_id) {
                let degree = shadow.degree_out(&next_id).max(1);
                queue.push_back((next_id, loss * 0.5 / degree as f64, hop + 1));
            }
        }
    }

    build_result(shadow, &impact, removed_energy)
}

/// Simulate the impact of adding a node connected to `connect_to` nodes.
///
/// The new node's energy radiates outward from its connected nodes.
pub fn simulate_add(
    shadow: &mut ShadowGraph,
    new_meta: NodeMeta,
    connect_to: &[Uuid],
    config: &AgencyConfig,
) -> SimulationResult {
    let new_energy = new_meta.energy;
    let new_id = new_meta.id;

    shadow.simulate_add_node(new_meta);
    for &target in connect_to {
        shadow.simulate_add_edge(new_id, target, 0.8);
    }

    let mut impact: HashMap<Uuid, f64> = HashMap::new();
    let mut queue: VecDeque<(Uuid, f64, usize)> = VecDeque::new();

    for &target in connect_to {
        let degree = shadow.degree_out(&target).max(1);
        let gain = new_energy as f64 * 0.1 / degree as f64;
        queue.push_back((target, gain, 1));
    }

    while let Some((nid, gain, hop)) = queue.pop_front() {
        if hop > config.counterfactual_max_hops || gain < 1e-6 {
            continue;
        }

        let current = impact.entry(nid).or_insert(0.0);
        if *current >= gain {
            continue;
        }
        *current = gain;

        for next_id in shadow.neighbors(&nid) {
            if next_id != new_id && !impact.contains_key(&next_id) {
                let degree = shadow.degree_out(&next_id).max(1);
                queue.push_back((next_id, gain * 0.5 / degree as f64, hop + 1));
            }
        }
    }

    build_result(shadow, &impact, -new_energy) // negative = energy added
}

fn build_result(
    shadow: &ShadowGraph,
    impact: &HashMap<Uuid, f64>,
    removed_energy: f32,
) -> SimulationResult {
    let mut impact_scores: Vec<(Uuid, f64)> = impact
        .iter()
        .map(|(&id, &score)| (id, score))
        .collect();
    impact_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let energy_changes: Vec<EnergyChange> = impact_scores
        .iter()
        .filter_map(|&(id, delta)| {
            shadow.get_meta(&id).map(|m| EnergyChange {
                node_id: id,
                before: m.energy,
                after: (m.energy - delta as f32).max(0.0),
            })
        })
        .collect();

    let total_energy_before: f64 = shadow.nodes.values().map(|m| m.energy as f64).sum();
    let total_after = total_energy_before - removed_energy as f64;
    let n = shadow.nodes.len().max(1) as f64;
    let mean_energy_delta = ((total_after / n) - (total_energy_before / n)) as f32;

    SimulationResult {
        affected_radius: impact_scores.len(),
        impact_scores,
        energy_changes,
        mean_energy_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{AdjacencyIndex, Edge, EdgeType, GraphStorage, Node, PoincareVector};
    use tempfile::TempDir;

    fn open_storage(dir: &TempDir) -> GraphStorage {
        GraphStorage::open(dir.path().to_str().unwrap()).unwrap()
    }

    fn make_node(x: f32, y: f32, energy: f32) -> Node {
        let mut node = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, y]),
            serde_json::json!({}),
        );
        node.meta.energy = energy;
        node
    }

    #[test]
    fn remove_hub_high_impact() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        // Star: hub -> leaf1, hub -> leaf2, hub -> leaf3
        let hub = make_node(0.1, 0.0, 0.9);
        let hub_id = hub.id;
        storage.put_node(&hub).unwrap();

        let mut leaf_ids = Vec::new();
        for i in 0..3 {
            let leaf = make_node(0.2 + i as f32 * 0.1, 0.0, 0.5);
            let lid = leaf.id;
            storage.put_node(&leaf).unwrap();
            let edge = Edge::new(hub_id, lid, EdgeType::Association, 0.9);
            storage.put_edge(&edge).unwrap();
            adjacency.add_edge(&edge);
            leaf_ids.push(lid);
        }

        let mut shadow = ShadowGraph::snapshot(&storage, &adjacency).unwrap();
        let config = AgencyConfig::default();
        let result = simulate_remove(&mut shadow, hub_id, &config);

        assert!(result.affected_radius > 0, "should affect neighbors");
        assert!(result.impact_scores.len() >= 3, "should affect all leaves");
    }

    #[test]
    fn remove_leaf_low_impact() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let hub = make_node(0.1, 0.0, 0.9);
        let leaf = make_node(0.3, 0.0, 0.2);
        let hub_id = hub.id;
        let leaf_id = leaf.id;

        storage.put_node(&hub).unwrap();
        storage.put_node(&leaf).unwrap();

        let edge = Edge::new(hub_id, leaf_id, EdgeType::Association, 0.5);
        storage.put_edge(&edge).unwrap();
        adjacency.add_edge(&edge);

        let mut shadow = ShadowGraph::snapshot(&storage, &adjacency).unwrap();
        let config = AgencyConfig::default();
        let result = simulate_remove(&mut shadow, leaf_id, &config);

        // Removing a low-energy leaf should have minimal impact
        if !result.impact_scores.is_empty() {
            assert!(result.impact_scores[0].1 < 0.5, "low-energy leaf removal should have low impact");
        }
    }

    #[test]
    fn add_node_predicts_impact() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let a = make_node(0.1, 0.0, 0.5);
        let b = make_node(0.2, 0.0, 0.5);
        let aid = a.id;
        let bid = b.id;
        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();
        let edge = Edge::new(aid, bid, EdgeType::Association, 0.8);
        storage.put_edge(&edge).unwrap();
        adjacency.add_edge(&edge);

        let mut shadow = ShadowGraph::snapshot(&storage, &adjacency).unwrap();
        let config = AgencyConfig::default();

        let new_meta = NodeMeta {
            id: Uuid::new_v4(),
            depth: 0.15,
            content: serde_json::json!({}),
            node_type: nietzsche_graph::NodeType::Semantic,
            energy: 0.9,
            lsystem_generation: 0,
            hausdorff_local: 1.0,
            created_at: 0,
            expires_at: None,
            metadata: std::collections::HashMap::new(),
        };

        let result = simulate_add(&mut shadow, new_meta, &[aid, bid], &config);
        assert!(result.affected_radius > 0);
    }
}
