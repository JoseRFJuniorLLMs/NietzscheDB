//! Phase XII — Curiosity Engine for NietzscheDB.
//!
//! Drives **exploration** of unknown regions in the Poincaré ball by
//! computing novelty signals from local density and converting them
//! into curiosity-weighted attention bids.
//!
//! ## How it works
//!
//! 1. **Novelty**: inversely proportional to local angular density.
//!    Nodes in sparse regions of the disk are "novel" and attract exploration.
//!    `novelty = 1 / (1 + density)`
//!
//! 2. **Curiosity**: the impulse to explore. Combines energy with novelty.
//!    `curiosity = energy × novelty`
//!
//! 3. **Exploration vs Exploitation**: each cycle computes a global
//!    `exploration_ratio` from mean curiosity. High curiosity = more
//!    exploration bids; low curiosity = more exploitation bids.
//!
//! ## Integration with ECAN
//!
//! The curiosity engine produces **exploration bids** that compete in
//! the same auction as exploitation bids. This creates a natural
//! explore/exploit balance without any hard-coded switching logic.

use std::collections::HashMap;
use uuid::Uuid;

use crate::attention_economy::{AttentionBid, AttentionState};

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

/// Per-node curiosity state.
#[derive(Debug, Clone)]
pub struct CuriosityState {
    pub node_id: Uuid,
    /// Local angular density around this node (0 = empty, 1 = saturated).
    pub density: f32,
    /// Novelty score: 1 / (1 + density).
    pub novelty: f32,
    /// Curiosity drive: energy × novelty.
    pub curiosity: f32,
}

/// Configuration for the curiosity engine.
#[derive(Debug, Clone)]
pub struct CuriosityConfig {
    /// Number of angular bins for density estimation.
    pub density_bins: usize,
    /// Weight of curiosity in the attention budget (additive).
    pub curiosity_budget_weight: f32,
    /// Weight of novelty in exploration bid computation.
    pub novelty_bid_weight: f32,
    /// Minimum curiosity to generate an exploration bid.
    pub curiosity_threshold: f32,
    /// Maximum exploration ratio (caps explore budget).
    pub max_exploration_ratio: f32,
}

impl Default for CuriosityConfig {
    fn default() -> Self {
        Self {
            density_bins: 32,
            curiosity_budget_weight: 0.5,
            novelty_bid_weight: 1.0,
            curiosity_threshold: 0.1,
            max_exploration_ratio: 0.6,
        }
    }
}

// ─────────────────────────────────────────────
// Core functions
// ─────────────────────────────────────────────

/// Compute novelty from local density.
///
/// `novelty = 1 / (1 + density)`
///
/// - density ≈ 0 → novelty ≈ 1.0 (frontier region, unexplored)
/// - density ≈ 1 → novelty ≈ 0.5 (well-explored)
/// - density >> 1 → novelty → 0.0 (saturated)
#[inline]
pub fn compute_novelty(density: f32) -> f32 {
    1.0 / (1.0 + density)
}

/// Compute curiosity drive for a node.
///
/// `curiosity = energy × novelty`
///
/// High-energy nodes in novel regions have the strongest exploration drive.
#[inline]
pub fn compute_curiosity(energy: f32, novelty: f32) -> f32 {
    energy * novelty
}

/// Compute angular density for each node from their theta values.
///
/// Uses a circular histogram: each node's theta is binned into
/// `n_bins` angular slices. The density at each node is the
/// normalized count of its bin.
///
/// Returns: map from node_id → density (count / max_count).
pub fn compute_angular_density(
    nodes: &[(Uuid, f32)], // (id, theta in radians)
    n_bins: usize,
) -> HashMap<Uuid, f32> {
    if nodes.is_empty() || n_bins == 0 {
        return HashMap::new();
    }

    let bin_width = std::f64::consts::TAU as f32 / n_bins as f32;

    // Build histogram
    let mut histogram = vec![0usize; n_bins];
    let mut node_bins: Vec<(Uuid, usize)> = Vec::with_capacity(nodes.len());

    for &(id, theta) in nodes {
        let theta_pos = if theta < 0.0 {
            theta + std::f64::consts::TAU as f32
        } else {
            theta
        };
        let bin = ((theta_pos / bin_width) as usize).min(n_bins - 1);
        histogram[bin] += 1;
        node_bins.push((id, bin));
    }

    // Max count for normalization
    let max_count = *histogram.iter().max().unwrap_or(&1);
    let max_f = max_count.max(1) as f32;

    // Assign density to each node
    let mut density_map = HashMap::with_capacity(nodes.len());
    for (id, bin) in node_bins {
        let density = histogram[bin] as f32 / max_f;
        density_map.insert(id, density);
    }

    density_map
}

/// Compute curiosity states for all participating nodes.
pub fn compute_curiosity_states(
    attention_states: &[AttentionState],
    density_map: &HashMap<Uuid, f32>,
) -> Vec<CuriosityState> {
    attention_states
        .iter()
        .map(|state| {
            let density = density_map.get(&state.node_id).copied().unwrap_or(0.5);
            let novelty = compute_novelty(density);
            let curiosity = compute_curiosity(
                state.semantic_mass.max(state.budget),
                novelty,
            );
            CuriosityState {
                node_id: state.node_id,
                density,
                novelty,
                curiosity,
            }
        })
        .collect()
}

/// Compute the global exploration ratio.
///
/// `exploration_ratio = clamp(mean_curiosity, 0, max_exploration_ratio)`
///
/// High mean curiosity = the graph wants to explore.
/// Low mean curiosity = the graph is satisfied and exploits.
pub fn exploration_ratio(
    curiosity_states: &[CuriosityState],
    max_ratio: f32,
) -> f32 {
    if curiosity_states.is_empty() {
        return 0.0;
    }
    let mean = curiosity_states.iter().map(|c| c.curiosity).sum::<f32>()
        / curiosity_states.len() as f32;
    mean.clamp(0.0, max_ratio)
}

/// Generate exploration bids from curiosity states.
///
/// Each curious node bids on the node with highest novelty among
/// its known neighbours (passed in via `neighbour_novelties`).
///
/// Exploration bid: `source.curiosity × target.novelty × weight / (1 + distance)`
pub fn generate_exploration_bids(
    curiosity_states: &[CuriosityState],
    // For each source, its candidate targets with (target_id, novelty, distance_proxy)
    neighbour_candidates: &HashMap<Uuid, Vec<(Uuid, f32, f32)>>,
    config: &CuriosityConfig,
) -> Vec<AttentionBid> {
    let mut bids = Vec::new();

    for cstate in curiosity_states {
        if cstate.curiosity < config.curiosity_threshold {
            continue;
        }

        if let Some(candidates) = neighbour_candidates.get(&cstate.node_id) {
            // Pick top candidate by novelty-weighted score
            if let Some(best) = candidates.iter().max_by(|a, b| {
                let sa = a.1 / (1.0 + a.2); // novelty / (1 + distance)
                let sb = b.1 / (1.0 + b.2);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                let bid_value = cstate.curiosity
                    * best.1  // target novelty
                    * config.novelty_bid_weight
                    / (1.0 + best.2); // distance penalty

                if bid_value > 0.0 {
                    bids.push(AttentionBid {
                        source: cstate.node_id,
                        target: best.0,
                        value: bid_value,
                    });
                }
            }
        }
    }

    bids
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn novelty_inverse_of_density() {
        assert!((compute_novelty(0.0) - 1.0).abs() < 0.01);
        assert!((compute_novelty(1.0) - 0.5).abs() < 0.01);
        assert!(compute_novelty(10.0) < 0.15);
    }

    #[test]
    fn curiosity_zero_when_no_energy() {
        assert_eq!(compute_curiosity(0.0, 1.0), 0.0);
    }

    #[test]
    fn curiosity_high_for_energetic_novel_nodes() {
        let c = compute_curiosity(0.9, 0.95);
        assert!(c > 0.8, "high energy + high novelty = high curiosity: {c}");
    }

    #[test]
    fn angular_density_detects_clusters() {
        let nodes: Vec<(Uuid, f32)> = (0..20)
            .map(|i| {
                let theta = if i < 15 {
                    0.5 + (i as f32) * 0.01 // cluster at θ ≈ 0.5
                } else {
                    2.5 + (i as f32) * 0.1 // sparse at θ ≈ 2.5+
                };
                (Uuid::from_u128(i as u128), theta)
            })
            .collect();

        let density = compute_angular_density(&nodes, 16);

        // Nodes in the cluster should have higher density
        let cluster_d = density.get(&Uuid::from_u128(0)).unwrap();
        let sparse_d = density.get(&Uuid::from_u128(18)).unwrap();
        assert!(
            cluster_d > sparse_d,
            "cluster density {cluster_d} should > sparse {sparse_d}"
        );
    }

    #[test]
    fn exploration_ratio_bounded() {
        let states: Vec<CuriosityState> = (0..10)
            .map(|i| CuriosityState {
                node_id: Uuid::from_u128(i),
                density: 0.1,
                novelty: 0.9,
                curiosity: 0.9,
            })
            .collect();

        let ratio = exploration_ratio(&states, 0.6);
        assert!(ratio <= 0.6, "ratio should be capped at max: {ratio}");
        assert!(ratio > 0.0, "ratio should be positive");
    }

    #[test]
    fn exploration_ratio_zero_when_saturated() {
        let states: Vec<CuriosityState> = (0..10)
            .map(|i| CuriosityState {
                node_id: Uuid::from_u128(i),
                density: 1.0,
                novelty: 0.0,
                curiosity: 0.0,
            })
            .collect();

        let ratio = exploration_ratio(&states, 0.6);
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn exploration_bids_generated_for_curious_nodes() {
        let curiosity_states = vec![
            CuriosityState {
                node_id: Uuid::from_u128(1),
                density: 0.1,
                novelty: 0.9,
                curiosity: 0.8,
            },
            CuriosityState {
                node_id: Uuid::from_u128(2),
                density: 0.9,
                novelty: 0.1,
                curiosity: 0.05, // below threshold
            },
        ];

        let mut candidates = HashMap::new();
        candidates.insert(
            Uuid::from_u128(1),
            vec![(Uuid::from_u128(10), 0.95, 0.2)], // high novelty target
        );
        candidates.insert(
            Uuid::from_u128(2),
            vec![(Uuid::from_u128(20), 0.5, 0.5)],
        );

        let config = CuriosityConfig::default();
        let bids = generate_exploration_bids(&curiosity_states, &candidates, &config);

        // Only node 1 should generate a bid (node 2 below threshold)
        assert_eq!(bids.len(), 1);
        assert_eq!(bids[0].source, Uuid::from_u128(1));
        assert!(bids[0].value > 0.0);
    }

    #[test]
    fn empty_inputs_no_crash() {
        let density = compute_angular_density(&[], 16);
        assert!(density.is_empty());

        let ratio = exploration_ratio(&[], 0.6);
        assert_eq!(ratio, 0.0);
    }
}
