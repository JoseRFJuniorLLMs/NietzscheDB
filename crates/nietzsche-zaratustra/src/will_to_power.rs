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
