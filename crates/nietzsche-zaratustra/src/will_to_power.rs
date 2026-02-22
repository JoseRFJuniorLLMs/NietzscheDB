//! **Will to Power** — energy propagation engine.
//!
//! # Philosophy
//! Nietzsche's *Wille zur Macht* holds that all life is driven by a fundamental
//! drive to grow, extend, and overcome.  In NietzscheDB this translates to a
//! graph-diffusion process: energy flows from strongly-connected nodes outward,
//! amplifying knowledge hubs while draining isolated or stagnant nodes.
//!
//! # Algorithm
//! For each propagation step:
//! ```text
//! new_energy(v) = clip(
//!     old_energy(v) × (1 − δ)           ← temporal decay
//!     + α × mean(energy(u) for u ∈ N_out(v)),  ← neighbour influence
//!     0, cap
//! )
//! ```
//! Nodes with no out-neighbours still decay by `(1 − δ)`.
//!
//! # Complexity
//! O(V + E) per propagation step.

use std::collections::HashMap;
use uuid::Uuid;

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use nietzsche_lsystem::CircuitBreakerConfig;

use crate::config::ZaratustraConfig;
use crate::error::ZaratustraError;

/// Statistics produced by a single Will-to-Power phase.
#[derive(Debug, Clone, Default)]
pub struct WillToPowerReport {
    /// Number of nodes whose energy changed.
    pub nodes_updated: u64,
    /// Mean energy before propagation.
    pub mean_energy_before: f32,
    /// Mean energy after propagation.
    pub mean_energy_after: f32,
    /// Total energy delta (sum of |new − old|).
    pub total_energy_delta: f32,
}

/// Run `config.propagation_steps` energy-propagation iterations over the
/// graph, writing updated energies back via `GraphStorage::put_node`.
///
/// `GraphStorage::put_node` takes `&self` (RocksDB is internally thread-safe),
/// so no mutable borrow is required.
pub fn run_will_to_power(
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
    config:    &ZaratustraConfig,
) -> Result<WillToPowerReport, ZaratustraError> {
    let nodes = storage.scan_nodes()
        .map_err(|e| ZaratustraError::Graph(e.to_string()))?;

    if nodes.is_empty() {
        return Ok(WillToPowerReport::default());
    }

    // Build an in-memory energy map for synchronous batch updates.
    let mut energy_map: HashMap<Uuid, f32> = nodes
        .iter()
        .map(|n| (n.id, n.energy))
        .collect();

    let mean_before: f32 = {
        let sum: f32 = energy_map.values().copied().sum();
        sum / energy_map.len() as f32
    };

    for _step in 0..config.propagation_steps {
        // Snapshot current energies — synchronous update semantics.
        let snapshot: HashMap<Uuid, f32> = energy_map.clone();

        for (&node_id, cur_energy) in energy_map.iter_mut() {
            let neighbours = adjacency.neighbors_out(&node_id);

            // Mean energy of out-neighbours (0 if isolated)
            let neighbour_mean: f32 = if neighbours.is_empty() {
                0.0
            } else {
                let sum: f32 = neighbours
                    .iter()
                    .filter_map(|nid| snapshot.get(nid).copied())
                    .sum();
                sum / neighbours.len() as f32
            };

            // Will-to-Power update equation
            *cur_energy = (*cur_energy * (1.0 - config.decay)
                + config.alpha * neighbour_mean)
                .clamp(0.0, config.energy_cap);
        }
    }

    // Write updated nodes back to storage via put_node (full node overwrite).
    let mut nodes_updated = 0u64;
    let mut total_delta   = 0.0f32;

    for mut node in nodes {
        let new_energy = energy_map[&node.id];
        let delta      = (new_energy - node.energy).abs();

        if delta > f32::EPSILON {
            node.energy = new_energy;
            storage.put_node(&node)
                .map_err(|e| ZaratustraError::Graph(e.to_string()))?;
            nodes_updated += 1;
            total_delta   += delta;
        }
    }

    let mean_after: f32 = {
        let sum: f32 = energy_map.values().copied().sum();
        sum / energy_map.len() as f32
    };

    Ok(WillToPowerReport {
        nodes_updated,
        mean_energy_before: mean_before,
        mean_energy_after:  mean_after,
        total_energy_delta: total_delta,
    })
}

/// Enhanced Will-to-Power with circuit breaker integration.
///
/// Same as [`run_will_to_power`] but additionally:
/// - Applies **depth-aware energy caps** (boundary nodes get lower caps)
/// - Applies **rate limiting** (max energy delta per node per cycle)
///
/// Returns `(WillToPowerReport, nodes_depth_capped, nodes_rate_limited)`.
pub fn run_will_to_power_guarded(
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
    config:    &ZaratustraConfig,
    cb_config: &CircuitBreakerConfig,
) -> Result<(WillToPowerReport, usize, usize), ZaratustraError> {
    let nodes = storage.scan_nodes()
        .map_err(|e| ZaratustraError::Graph(e.to_string()))?;

    if nodes.is_empty() {
        return Ok((WillToPowerReport::default(), 0, 0));
    }

    // Snapshot: old energies + depths
    let old_energies: HashMap<Uuid, f32> = nodes
        .iter()
        .map(|n| (n.id, n.energy))
        .collect();
    let depths: HashMap<Uuid, f32> = nodes
        .iter()
        .map(|n| (n.id, n.depth))
        .collect();

    let mean_before: f32 = {
        let sum: f32 = old_energies.values().copied().sum();
        sum / old_energies.len() as f32
    };

    // Standard propagation
    let mut energy_map: HashMap<Uuid, f32> = old_energies.clone();

    for _step in 0..config.propagation_steps {
        let snapshot: HashMap<Uuid, f32> = energy_map.clone();

        for (&node_id, cur_energy) in energy_map.iter_mut() {
            let neighbours = adjacency.neighbors_out(&node_id);
            let neighbour_mean: f32 = if neighbours.is_empty() {
                0.0
            } else {
                let sum: f32 = neighbours
                    .iter()
                    .filter_map(|nid| snapshot.get(nid).copied())
                    .sum();
                sum / neighbours.len() as f32
            };

            *cur_energy = (*cur_energy * (1.0 - config.decay)
                + config.alpha * neighbour_mean)
                .clamp(0.0, config.energy_cap);
        }
    }

    // Apply depth-aware caps
    let mut nodes_depth_capped = 0usize;
    for (&id, energy) in energy_map.iter_mut() {
        let depth = depths.get(&id).copied().unwrap_or(0.0);
        let cap = nietzsche_lsystem::depth_aware_cap(depth, cb_config);
        if *energy > cap {
            *energy = cap;
            nodes_depth_capped += 1;
        }
    }

    // Apply rate limiting
    let mut nodes_rate_limited = 0usize;
    if cb_config.max_energy_delta > 0.0 {
        for (&id, energy) in energy_map.iter_mut() {
            let old_e = old_energies.get(&id).copied().unwrap_or(0.0);
            let delta = *energy - old_e;
            if delta > cb_config.max_energy_delta {
                *energy = old_e + cb_config.max_energy_delta;
                nodes_rate_limited += 1;
            }
        }
    }

    // Persist updates
    let mut nodes_updated = 0u64;
    let mut total_delta = 0.0f32;

    for mut node in nodes {
        let new_energy = energy_map[&node.id];
        let delta = (new_energy - node.energy).abs();

        if delta > f32::EPSILON {
            node.energy = new_energy;
            storage.put_node(&node)
                .map_err(|e| ZaratustraError::Graph(e.to_string()))?;
            nodes_updated += 1;
            total_delta += delta;
        }
    }

    let mean_after: f32 = {
        let sum: f32 = energy_map.values().copied().sum();
        sum / energy_map.len() as f32
    };

    Ok((
        WillToPowerReport {
            nodes_updated,
            mean_energy_before: mean_before,
            mean_energy_after: mean_after,
            total_energy_delta: total_delta,
        },
        nodes_depth_capped,
        nodes_rate_limited,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{AdjacencyIndex, GraphStorage, Edge, EdgeType, Node, PoincareVector};
    use tempfile::TempDir;

    fn open_temp_db() -> (GraphStorage, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();
        (storage, dir)
    }

    fn make_node(x: f32, y: f32, energy: f32) -> Node {
        let mut node = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, y]),
            serde_json::json!({"label": "test"}),
        );
        node.energy = energy;
        node
    }

    fn default_config() -> ZaratustraConfig {
        ZaratustraConfig::default()
    }

    #[test]
    fn empty_graph_returns_default_report() {
        let (storage, _dir) = open_temp_db();
        let adjacency = AdjacencyIndex::new();
        let report = run_will_to_power(&storage, &adjacency, &default_config()).unwrap();
        assert_eq!(report.nodes_updated, 0);
        assert!((report.mean_energy_before - 0.0).abs() < f32::EPSILON);
        assert!((report.mean_energy_after - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn single_isolated_node_decays() {
        let (storage, _dir) = open_temp_db();
        let adjacency = AdjacencyIndex::new();
        let node = make_node(0.1, 0.1, 0.5);
        let id = node.id;
        storage.put_node(&node).unwrap();

        let cfg = ZaratustraConfig {
            propagation_steps: 1,
            decay: 0.1,
            ..default_config()
        };
        let report = run_will_to_power(&storage, &adjacency, &cfg).unwrap();

        assert_eq!(report.nodes_updated, 1);
        // After 1 step: 0.5 * (1 - 0.1) = 0.45
        let updated = storage.scan_nodes().unwrap();
        let n = updated.iter().find(|n| n.id == id).unwrap();
        assert!((n.energy - 0.45).abs() < 1e-5, "expected 0.45, got {}", n.energy);
    }

    #[test]
    fn hub_gains_more_energy_than_leaf() {
        // Create a star topology: hub -> leaf1, hub -> leaf2, hub -> leaf3
        // All start with energy 0.5. Hub has energetic neighbours; leaves are isolated.
        let (storage, _dir) = open_temp_db();
        let adjacency = AdjacencyIndex::new();

        let hub = make_node(0.1, 0.0, 0.5);
        let leaf1 = make_node(0.2, 0.0, 0.8);
        let leaf2 = make_node(0.0, 0.2, 0.8);
        let leaf3 = make_node(0.0, 0.1, 0.8);

        let hub_id = hub.id;
        let leaf1_id = leaf1.id;

        // Hub points to leaves (hub -> leaf_i)
        // So hub's out-neighbours are the leaves (energetic).
        let e1 = Edge::new(hub.id, leaf1.id, EdgeType::Association, 1.0);
        let e2 = Edge::new(hub.id, leaf2.id, EdgeType::Association, 1.0);
        let e3 = Edge::new(hub.id, leaf3.id, EdgeType::Association, 1.0);

        storage.put_node(&hub).unwrap();
        storage.put_node(&leaf1).unwrap();
        storage.put_node(&leaf2).unwrap();
        storage.put_node(&leaf3).unwrap();

        adjacency.add_edge(&e1);
        adjacency.add_edge(&e2);
        adjacency.add_edge(&e3);

        let cfg = ZaratustraConfig {
            alpha: 0.5,
            decay: 0.0,
            propagation_steps: 1,
            ..default_config()
        };

        run_will_to_power(&storage, &adjacency, &cfg).unwrap();

        let nodes = storage.scan_nodes().unwrap();
        let hub_after = nodes.iter().find(|n| n.id == hub_id).unwrap();
        let leaf_after = nodes.iter().find(|n| n.id == leaf1_id).unwrap();

        // Hub: 0.5 * (1-0) + 0.5 * mean(0.8, 0.8, 0.8) = 0.5 + 0.4 = 0.9
        // Leaf1 (no out-neighbours): 0.8 * (1-0) + 0 = 0.8 (unchanged)
        assert!(
            hub_after.energy > leaf_after.energy,
            "hub energy ({}) should exceed leaf energy ({})",
            hub_after.energy,
            leaf_after.energy,
        );
    }

    #[test]
    fn energy_is_capped_at_energy_cap() {
        let (storage, _dir) = open_temp_db();
        let adjacency = AdjacencyIndex::new();

        let a = make_node(0.1, 0.0, 0.95);
        let b = make_node(0.0, 0.1, 0.99);
        let a_id = a.id;

        // a -> b (a receives energy from b)
        let edge = Edge::new(a.id, b.id, EdgeType::Association, 1.0);
        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();
        adjacency.add_edge(&edge);

        let cfg = ZaratustraConfig {
            alpha: 0.5,
            decay: 0.0,
            energy_cap: 1.0,
            propagation_steps: 5,
            ..default_config()
        };

        run_will_to_power(&storage, &adjacency, &cfg).unwrap();

        let nodes = storage.scan_nodes().unwrap();
        let a_after = nodes.iter().find(|n| n.id == a_id).unwrap();
        assert!(
            a_after.energy <= 1.0 + f32::EPSILON,
            "energy {} should not exceed cap 1.0",
            a_after.energy,
        );
    }

    #[test]
    fn decay_reduces_energy_every_step() {
        let (storage, _dir) = open_temp_db();
        let adjacency = AdjacencyIndex::new();

        let node = make_node(0.1, 0.1, 1.0);
        let id = node.id;
        storage.put_node(&node).unwrap();

        let cfg = ZaratustraConfig {
            alpha: 0.0,  // no neighbour influence
            decay: 0.10, // 10% decay per step
            propagation_steps: 3,
            ..default_config()
        };

        run_will_to_power(&storage, &adjacency, &cfg).unwrap();

        // After 3 steps: 1.0 * (0.9)^3 = 0.729
        let nodes = storage.scan_nodes().unwrap();
        let n = nodes.iter().find(|n| n.id == id).unwrap();
        assert!(
            (n.energy - 0.729).abs() < 1e-4,
            "expected ~0.729, got {}",
            n.energy,
        );
    }

    #[test]
    fn report_mean_energy_before_and_after() {
        let (storage, _dir) = open_temp_db();
        let adjacency = AdjacencyIndex::new();

        let n1 = make_node(0.1, 0.0, 0.4);
        let n2 = make_node(0.0, 0.1, 0.6);
        storage.put_node(&n1).unwrap();
        storage.put_node(&n2).unwrap();

        let cfg = ZaratustraConfig {
            propagation_steps: 1,
            decay: 0.0,
            alpha: 0.0,
            ..default_config()
        };

        let report = run_will_to_power(&storage, &adjacency, &cfg).unwrap();
        // mean_before = (0.4+0.6)/2 = 0.5
        assert!((report.mean_energy_before - 0.5).abs() < 1e-5);
        // With alpha=0, decay=0, no change
        assert!((report.mean_energy_after - 0.5).abs() < 1e-5);
    }

    #[test]
    fn multiple_propagation_steps_accumulate() {
        let (storage, _dir) = open_temp_db();
        let adjacency = AdjacencyIndex::new();

        // a -> b, both start at 0.5
        let a = make_node(0.1, 0.0, 0.5);
        let b = make_node(0.0, 0.1, 0.5);
        let a_id = a.id;
        let edge = Edge::new(a.id, b.id, EdgeType::Association, 1.0);
        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();
        adjacency.add_edge(&edge);

        let cfg1 = ZaratustraConfig {
            alpha: 0.2,
            decay: 0.0,
            propagation_steps: 1,
            ..default_config()
        };
        let cfg3 = ZaratustraConfig {
            propagation_steps: 3,
            ..cfg1.clone()
        };

        // Run 1-step first
        run_will_to_power(&storage, &adjacency, &cfg1).unwrap();
        let nodes_1 = storage.scan_nodes().unwrap();
        let a_1 = nodes_1.iter().find(|n| n.id == a_id).unwrap().energy;

        // Reset
        let (storage2, _dir2) = open_temp_db();
        let a2 = make_node(0.1, 0.0, 0.5);
        let b2 = make_node(0.0, 0.1, 0.5);
        // Re-use same IDs won't matter — different db
        let edge2 = Edge::new(a2.id, b2.id, EdgeType::Association, 1.0);
        let adj2 = AdjacencyIndex::new();
        storage2.put_node(&a2).unwrap();
        storage2.put_node(&b2).unwrap();
        adj2.add_edge(&edge2);

        run_will_to_power(&storage2, &adj2, &cfg3).unwrap();
        let nodes_3 = storage2.scan_nodes().unwrap();
        let a_3 = nodes_3.iter().find(|n| n.id == a2.id).unwrap().energy;

        // 3 steps should accumulate more energy than 1 step
        assert!(
            a_3 > a_1,
            "3-step energy {} should exceed 1-step energy {}",
            a_3,
            a_1,
        );
    }

    // ── Guarded (circuit-breaker-integrated) tests ──────────────────────────

    #[test]
    fn guarded_depth_cap_limits_boundary_node() {
        let (storage, _dir) = open_temp_db();
        let adjacency = AdjacencyIndex::new();

        // Deep node (depth ≈ 0.89) at high energy
        let deep = make_node(0.8, 0.4, 0.95); // ‖[0.8, 0.4]‖ ≈ 0.894
        let deep_id = deep.id;
        storage.put_node(&deep).unwrap();

        let cfg = ZaratustraConfig {
            alpha: 0.0,
            decay: 0.0,
            propagation_steps: 1,
            ..default_config()
        };
        let cb = CircuitBreakerConfig {
            depth_cap_gradient: 0.5,
            base_cap: 1.0,
            max_energy_delta: 0.0, // disabled
            ..CircuitBreakerConfig::default()
        };

        let (report, depth_capped, _rate_limited) =
            run_will_to_power_guarded(&storage, &adjacency, &cfg, &cb).unwrap();

        assert!(depth_capped >= 1, "deep node should be depth-capped");

        let nodes = storage.scan_nodes().unwrap();
        let n = nodes.iter().find(|n| n.id == deep_id).unwrap();
        // cap = 1.0 - 0.5 * 0.894 ≈ 0.553
        assert!(
            n.energy < 0.6,
            "deep node energy {} should be capped below 0.6",
            n.energy,
        );
    }

    #[test]
    fn guarded_rate_limiting_clamps_large_increase() {
        let (storage, _dir) = open_temp_db();
        let adjacency = AdjacencyIndex::new();

        // a -> b, a starts low, b starts very high
        let a = make_node(0.1, 0.0, 0.2);
        let b = make_node(0.0, 0.1, 1.0);
        let a_id = a.id;

        let edge = Edge::new(a.id, b.id, EdgeType::Association, 1.0);
        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();
        adjacency.add_edge(&edge);

        let cfg = ZaratustraConfig {
            alpha: 0.8, // aggressive propagation
            decay: 0.0,
            propagation_steps: 1,
            ..default_config()
        };
        let cb = CircuitBreakerConfig {
            max_energy_delta: 0.15,
            depth_cap_gradient: 0.0, // disabled
            ..CircuitBreakerConfig::default()
        };

        let (_report, _depth_capped, rate_limited) =
            run_will_to_power_guarded(&storage, &adjacency, &cfg, &cb).unwrap();

        assert!(rate_limited >= 1, "node a should be rate-limited");

        let nodes = storage.scan_nodes().unwrap();
        let a_after = nodes.iter().find(|n| n.id == a_id).unwrap();
        // Without rate limiting: 0.2 + 0.8 * 1.0 = 1.0 (delta 0.8)
        // With rate limiting:    0.2 + 0.15 = 0.35
        assert!(
            a_after.energy <= 0.36,
            "rate-limited energy {} should be ≤ 0.36",
            a_after.energy,
        );
    }
}
