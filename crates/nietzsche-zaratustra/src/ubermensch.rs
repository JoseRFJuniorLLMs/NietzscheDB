//! **Übermensch** — elite-tier analytics and reporting.
//!
//! # Philosophy
//! Nietzsche's *Übermensch* is not a fixed entity but a direction — the
//! ideal of self-overcoming.  In NietzscheDB the Übermensch tier is the
//! top fraction of nodes by energy at any given moment: those that have
//! "overcome" their peers through repeated activation and propagation.
//!
//! # Current capabilities
//! This module computes the elite-tier stats and identifies the Übermensch
//! node IDs.  Future work: hot-tier promotion (in-memory pinning).

use uuid::Uuid;

use nietzsche_graph::GraphStorage;

use crate::config::ZaratustraConfig;
use crate::error::ZaratustraError;

/// Statistics produced by the Übermensch phase.
#[derive(Debug, Clone, Default)]
pub struct UbermenschReport {
    /// Number of nodes in the elite tier.
    pub elite_count: u64,
    /// Energy threshold at which the elite tier begins.
    pub energy_threshold: f32,
    /// Mean energy across elite nodes.
    pub mean_elite_energy: f32,
    /// Mean energy across non-elite nodes.
    pub mean_base_energy: f32,
    /// Node IDs of the elite tier (sorted descending by energy).
    pub elite_node_ids: Vec<Uuid>,
}

/// Compute the Übermensch tier statistics from the current node energies.
pub fn run_ubermensch(
    storage: &GraphStorage,
    config:  &ZaratustraConfig,
) -> Result<UbermenschReport, ZaratustraError> {
    let nodes = storage.scan_nodes()
        .map_err(|e| ZaratustraError::Graph(e.to_string()))?;

    if nodes.is_empty() {
        return Ok(UbermenschReport::default());
    }

    // Sort descending by energy
    let mut ranked: Vec<(Uuid, f32)> = nodes
        .iter()
        .map(|n| (n.id, n.energy))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let total       = ranked.len();
    let elite_count = ((total as f32 * config.ubermensch_top_fraction).ceil() as usize)
        .max(1)
        .min(total);

    let elite      = &ranked[..elite_count];
    let base       = &ranked[elite_count..];
    let threshold  = elite.last().map(|(_, e)| *e).unwrap_or(0.0);

    let mean_elite = if elite.is_empty() { 0.0 } else {
        elite.iter().map(|(_, e)| e).sum::<f32>() / elite.len() as f32
    };
    let mean_base = if base.is_empty() { 0.0 } else {
        base.iter().map(|(_, e)| e).sum::<f32>() / base.len() as f32
    };

    Ok(UbermenschReport {
        elite_count:      elite_count as u64,
        energy_threshold: threshold,
        mean_elite_energy: mean_elite,
        mean_base_energy:  mean_base,
        elite_node_ids:    elite.iter().map(|(id, _)| *id).collect(),
    })
}
