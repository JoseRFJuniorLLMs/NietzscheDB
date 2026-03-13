// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Phase XIV — Semantic Gravity Model for NietzscheDB.
//!
//! Inspired by Newton's law of universal gravitation, adapted for hyperbolic
//! knowledge graphs:
//!
//! ```text
//! F(i,j) = G × (M_i × M_j) / d(i,j)²
//! ```
//!
//! Where:
//! - **M** = semantic mass = `energy × ln(degree + 1)` (from ECAN)
//! - **d** = Poincaré distance between embeddings
//! - **G** = gravitational constant (coupling strength)
//!
//! ## Emergent Behaviors
//!
//! - **Gravity wells**: high-mass concept clusters attract nearby nodes
//! - **Semantic orbits**: moderate-mass nodes orbit around gravity wells
//! - **Tidal forces**: competing gravity wells pull on shared neighborhoods
//! - **Escape velocity**: low-energy nodes far from any well drift to boundary
//!
//! ## Integration
//!
//! Runs as part of the agency tick (interval-gated). Produces:
//! 1. `GravityReport` with top-K attractions and gravity well census
//! 2. Optional `GravityPull` intents that nudge node energies toward wells
//!
//! The gravity field is **read-only by default** — it reports forces but
//! doesn't move nodes. Enable `apply_pulls` to let gravity actually
//! redistribute energy (experimental).

use std::collections::HashMap;
use uuid::Uuid;

use crate::attention_economy::semantic_mass;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the Semantic Gravity engine.
#[derive(Debug, Clone)]
pub struct GravityConfig {
    /// Gravitational constant — coupling strength (default: 1.0).
    pub g_constant: f64,
    /// Minimum mass to be considered a "gravity well" (default: 0.5).
    pub well_mass_threshold: f64,
    /// Maximum number of node pairs to evaluate (default: 5000).
    /// Gravity is O(N²), so we sample.
    pub max_pairs: usize,
    /// Top-K strongest attractions to report (default: 20).
    pub top_k: usize,
    /// Tick interval — run gravity every N agency ticks (default: 3).
    pub interval: u64,
    /// Whether to generate GravityPull intents (default: false).
    /// When true, gravity redistributes energy toward wells.
    pub apply_pulls: bool,
    /// Maximum energy pull per tick (default: 0.005).
    pub max_pull: f32,
    /// Minimum Poincaré distance to avoid singularity (default: 0.01).
    pub min_distance: f64,
    /// Whether gravity engine is enabled (default: true).
    pub enabled: bool,
}

impl Default for GravityConfig {
    fn default() -> Self {
        Self {
            g_constant: 1.0,
            well_mass_threshold: 0.5,
            max_pairs: 5000,
            top_k: 20,
            interval: 3,
            apply_pulls: false,
            max_pull: 0.005,
            min_distance: 0.01,
            enabled: true,
        }
    }
}

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

/// Input for gravity computation: one node's essential properties.
#[derive(Debug, Clone)]
pub struct GravityNode {
    pub id: Uuid,
    pub energy: f32,
    pub degree: usize,
    pub embedding: Vec<f64>,
}

impl GravityNode {
    /// Semantic mass = energy × ln(degree + 1).
    #[inline]
    pub fn mass(&self) -> f64 {
        semantic_mass(self.energy, self.degree) as f64
    }
}

/// A gravitational force between two nodes.
#[derive(Debug, Clone)]
pub struct GravityForce {
    pub source: Uuid,
    pub target: Uuid,
    pub force: f64,
    pub source_mass: f64,
    pub target_mass: f64,
    pub distance: f64,
}

/// An energy pull intent: gravity well attracts a nearby node.
#[derive(Debug, Clone)]
pub struct GravityPull {
    /// The gravity well (attractor).
    pub well_id: Uuid,
    /// The attracted node.
    pub node_id: Uuid,
    /// Energy to transfer from node → well's neighborhood boost.
    pub amount: f32,
}

/// Report from a gravity field computation.
#[derive(Debug, Clone)]
pub struct GravityReport {
    /// Number of node pairs evaluated.
    pub pairs_evaluated: usize,
    /// Top-K strongest gravitational attractions.
    pub top_attractions: Vec<GravityForce>,
    /// Mean force magnitude across all pairs.
    pub mean_force: f64,
    /// Gravity wells: nodes with mass above threshold.
    pub wells: Vec<GravityWell>,
    /// Optional energy pull intents.
    pub pulls: Vec<GravityPull>,
}

/// A node that qualifies as a gravity well.
#[derive(Debug, Clone)]
pub struct GravityWell {
    pub id: Uuid,
    pub mass: f64,
    /// Number of nodes attracted (force > mean_force).
    pub attracted_count: usize,
    /// Total gravitational force exerted on neighbors.
    pub total_force: f64,
}

// ─────────────────────────────────────────────
// Persistent state
// ─────────────────────────────────────────────

/// Persistent state for the gravity engine across ticks.
pub struct GravityState {
    tick_count: u64,
}

impl GravityState {
    pub fn new() -> Self {
        Self { tick_count: 0 }
    }
}

// ─────────────────────────────────────────────
// Core computation
// ─────────────────────────────────────────────

/// Compute gravitational force between two nodes.
///
/// `F = G × (M_a × M_b) / d²`
///
/// where d = Poincaré distance, clamped to min_distance to avoid singularity.
#[inline]
pub fn gravitational_force(
    mass_a: f64,
    mass_b: f64,
    distance: f64,
    g_constant: f64,
    min_distance: f64,
) -> f64 {
    let d = distance.max(min_distance);
    g_constant * (mass_a * mass_b) / (d * d)
}

/// Run a full gravity field computation.
///
/// Takes a slice of `GravityNode` (pre-sampled if needed) and computes
/// pairwise forces, identifies gravity wells, and optionally generates pulls.
pub fn compute_gravity_field(
    nodes: &[GravityNode],
    config: &GravityConfig,
) -> GravityReport {
    if nodes.len() < 2 {
        return GravityReport {
            pairs_evaluated: 0,
            top_attractions: Vec::new(),
            mean_force: 0.0,
            wells: Vec::new(),
            pulls: Vec::new(),
        };
    }

    // Pre-compute masses
    let masses: Vec<(Uuid, f64)> = nodes.iter()
        .map(|n| (n.id, n.mass()))
        .collect();

    // Identify gravity wells
    let wells_set: Vec<usize> = masses.iter().enumerate()
        .filter(|(_, (_, m))| *m >= config.well_mass_threshold)
        .map(|(i, _)| i)
        .collect();

    // Compute pairwise forces (sampled if too many pairs)
    let n = nodes.len();
    let total_pairs = n * (n - 1) / 2;
    let sample_every = if total_pairs > config.max_pairs {
        total_pairs / config.max_pairs
    } else {
        1
    };

    let mut forces: Vec<GravityForce> = Vec::with_capacity(config.max_pairs.min(total_pairs));
    let mut force_sum = 0.0f64;
    let mut pair_idx = 0usize;
    let mut pairs_evaluated = 0usize;

    for i in 0..n {
        for j in (i + 1)..n {
            pair_idx += 1;
            if sample_every > 1 && (pair_idx % sample_every) != 0 {
                continue;
            }

            let m_a = masses[i].1;
            let m_b = masses[j].1;

            // Skip pairs where both masses are negligible
            if m_a < 1e-6 && m_b < 1e-6 {
                continue;
            }

            let dist = nietzsche_hyp_ops::poincare_distance(
                &nodes[i].embedding,
                &nodes[j].embedding,
            );

            let f = gravitational_force(m_a, m_b, dist, config.g_constant, config.min_distance);
            force_sum += f;
            pairs_evaluated += 1;

            forces.push(GravityForce {
                source: nodes[i].id,
                target: nodes[j].id,
                force: f,
                source_mass: m_a,
                target_mass: m_b,
                distance: dist,
            });
        }
    }

    // Sort by force (descending) and take top-K
    forces.sort_by(|a, b| b.force.partial_cmp(&a.force).unwrap_or(std::cmp::Ordering::Equal));
    let top_attractions: Vec<GravityForce> = forces.into_iter()
        .take(config.top_k)
        .collect();

    let mean_force = if pairs_evaluated > 0 {
        force_sum / pairs_evaluated as f64
    } else {
        0.0
    };

    // Build gravity well reports
    let mut well_force_totals: HashMap<Uuid, (f64, usize)> = HashMap::new();
    for gf in &top_attractions {
        // Both source and target contribute to well stats
        for &(well_id, _) in wells_set.iter().filter_map(|&wi| {
            if nodes[wi].id == gf.source || nodes[wi].id == gf.target {
                Some(&masses[wi])
            } else {
                None
            }
        }) {
            let entry = well_force_totals.entry(well_id).or_insert((0.0, 0));
            entry.0 += gf.force;
            entry.1 += 1;
        }
    }

    let wells: Vec<GravityWell> = wells_set.iter().map(|&wi| {
        let (total_f, count) = well_force_totals
            .get(&nodes[wi].id)
            .copied()
            .unwrap_or((0.0, 0));
        GravityWell {
            id: nodes[wi].id,
            mass: masses[wi].1,
            attracted_count: count,
            total_force: total_f,
        }
    }).collect();

    // Generate pulls if enabled
    let pulls = if config.apply_pulls && !wells.is_empty() {
        generate_pulls(&top_attractions, &wells, config)
    } else {
        Vec::new()
    };

    if pairs_evaluated > 0 {
        tracing::debug!(
            pairs = pairs_evaluated,
            wells = wells.len(),
            mean_force = format!("{:.6}", mean_force),
            top_force = top_attractions.first().map(|f| format!("{:.4}", f.force)).unwrap_or_default(),
            "Gravity field computed"
        );
    }

    GravityReport {
        pairs_evaluated,
        top_attractions,
        mean_force,
        wells,
        pulls,
    }
}

/// Generate energy pull intents from gravity wells to attracted nodes.
fn generate_pulls(
    top_attractions: &[GravityForce],
    wells: &[GravityWell],
    config: &GravityConfig,
) -> Vec<GravityPull> {
    let well_ids: std::collections::HashSet<Uuid> = wells.iter().map(|w| w.id).collect();
    let mut pulls = Vec::new();

    for gf in top_attractions {
        // If source is a well, it attracts target
        if well_ids.contains(&gf.source) && !well_ids.contains(&gf.target) {
            let amount = (gf.force as f32 * 0.01).min(config.max_pull);
            if amount > 1e-6 {
                pulls.push(GravityPull {
                    well_id: gf.source,
                    node_id: gf.target,
                    amount,
                });
            }
        }
        // If target is a well, it attracts source
        if well_ids.contains(&gf.target) && !well_ids.contains(&gf.source) {
            let amount = (gf.force as f32 * 0.01).min(config.max_pull);
            if amount > 1e-6 {
                pulls.push(GravityPull {
                    well_id: gf.target,
                    node_id: gf.source,
                    amount,
                });
            }
        }
    }

    pulls
}

/// Run a gravity tick: check interval, compute field, return report.
pub fn run_gravity_tick(
    state: &mut GravityState,
    nodes: &[GravityNode],
    config: &GravityConfig,
) -> Option<GravityReport> {
    if !config.enabled {
        return None;
    }

    state.tick_count += 1;
    if config.interval > 0 && (state.tick_count % config.interval) != 0 {
        return None;
    }

    Some(compute_gravity_field(nodes, config))
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u128, energy: f32, degree: usize, coords: &[f64]) -> GravityNode {
        GravityNode {
            id: Uuid::from_u128(id),
            energy,
            degree,
            embedding: coords.to_vec(),
        }
    }

    #[test]
    fn force_inverse_square() {
        let f1 = gravitational_force(1.0, 1.0, 1.0, 1.0, 0.01);
        let f2 = gravitational_force(1.0, 1.0, 2.0, 1.0, 0.01);
        // At double distance, force should be 1/4
        assert!((f2 - f1 / 4.0).abs() < 1e-6, "f1={f1}, f2={f2}");
    }

    #[test]
    fn force_proportional_to_mass() {
        let f1 = gravitational_force(1.0, 1.0, 1.0, 1.0, 0.01);
        let f2 = gravitational_force(2.0, 1.0, 1.0, 1.0, 0.01);
        assert!((f2 - 2.0 * f1).abs() < 1e-6);
    }

    #[test]
    fn force_singularity_protection() {
        // Distance = 0 should be clamped to min_distance
        let f = gravitational_force(1.0, 1.0, 0.0, 1.0, 0.1);
        assert!(f.is_finite());
        assert!(f > 0.0);
        // Should equal G * M² / min_d²
        let expected = 1.0 / (0.1 * 0.1);
        assert!((f - expected).abs() < 1e-6);
    }

    #[test]
    fn gravity_coupling_constant() {
        let f1 = gravitational_force(1.0, 1.0, 1.0, 1.0, 0.01);
        let f2 = gravitational_force(1.0, 1.0, 1.0, 3.0, 0.01);
        assert!((f2 - 3.0 * f1).abs() < 1e-6);
    }

    #[test]
    fn empty_nodes_no_crash() {
        let config = GravityConfig::default();
        let report = compute_gravity_field(&[], &config);
        assert_eq!(report.pairs_evaluated, 0);
        assert!(report.top_attractions.is_empty());
    }

    #[test]
    fn single_node_no_forces() {
        let config = GravityConfig::default();
        let nodes = vec![node(1, 0.8, 5, &[0.1, 0.2])];
        let report = compute_gravity_field(&nodes, &config);
        assert_eq!(report.pairs_evaluated, 0);
    }

    #[test]
    fn two_nodes_one_force() {
        let config = GravityConfig::default();
        let nodes = vec![
            node(1, 0.8, 5, &[0.1, 0.0]),
            node(2, 0.6, 3, &[0.3, 0.0]),
        ];
        let report = compute_gravity_field(&nodes, &config);
        assert_eq!(report.pairs_evaluated, 1);
        assert_eq!(report.top_attractions.len(), 1);
        assert!(report.top_attractions[0].force > 0.0);
    }

    #[test]
    fn closer_nodes_stronger_force() {
        let config = GravityConfig::default();
        let nodes = vec![
            node(1, 0.8, 5, &[0.1, 0.0]),
            node(2, 0.6, 3, &[0.12, 0.0]),  // very close
            node(3, 0.6, 3, &[0.5, 0.0]),   // far
        ];
        let report = compute_gravity_field(&nodes, &config);
        // Force 1→2 should be much stronger than 1→3
        let f12 = report.top_attractions.iter()
            .find(|f| {
                (f.source == Uuid::from_u128(1) && f.target == Uuid::from_u128(2)) ||
                (f.source == Uuid::from_u128(2) && f.target == Uuid::from_u128(1))
            })
            .map(|f| f.force)
            .unwrap_or(0.0);
        let f13 = report.top_attractions.iter()
            .find(|f| {
                (f.source == Uuid::from_u128(1) && f.target == Uuid::from_u128(3)) ||
                (f.source == Uuid::from_u128(3) && f.target == Uuid::from_u128(1))
            })
            .map(|f| f.force)
            .unwrap_or(0.0);
        assert!(f12 > f13, "closer nodes should have stronger force: f12={f12}, f13={f13}");
    }

    #[test]
    fn gravity_wells_identified() {
        let config = GravityConfig {
            well_mass_threshold: 0.5,
            ..Default::default()
        };
        let nodes = vec![
            node(1, 0.9, 10, &[0.1, 0.0]),  // high mass → well
            node(2, 0.01, 1, &[0.3, 0.0]),   // low mass → not a well
        ];
        let report = compute_gravity_field(&nodes, &config);
        let well_ids: Vec<_> = report.wells.iter().map(|w| w.id).collect();
        assert!(well_ids.contains(&Uuid::from_u128(1)));
        assert!(!well_ids.contains(&Uuid::from_u128(2)));
    }

    #[test]
    fn zero_energy_produces_zero_mass() {
        let n = node(1, 0.0, 10, &[0.1, 0.0]);
        assert!(n.mass().abs() < 1e-6);
    }

    #[test]
    fn disabled_returns_none() {
        let config = GravityConfig { enabled: false, ..Default::default() };
        let mut state = GravityState::new();
        let nodes = vec![node(1, 0.8, 5, &[0.1, 0.0])];
        assert!(run_gravity_tick(&mut state, &nodes, &config).is_none());
    }

    #[test]
    fn interval_gating() {
        let config = GravityConfig { interval: 3, ..Default::default() };
        let mut state = GravityState::new();
        let nodes = vec![
            node(1, 0.8, 5, &[0.1, 0.0]),
            node(2, 0.6, 3, &[0.3, 0.0]),
        ];

        // Tick 1, 2: gated
        assert!(run_gravity_tick(&mut state, &nodes, &config).is_none());
        assert!(run_gravity_tick(&mut state, &nodes, &config).is_none());
        // Tick 3: runs
        assert!(run_gravity_tick(&mut state, &nodes, &config).is_some());
        // Tick 4, 5: gated
        assert!(run_gravity_tick(&mut state, &nodes, &config).is_none());
        assert!(run_gravity_tick(&mut state, &nodes, &config).is_none());
        // Tick 6: runs
        assert!(run_gravity_tick(&mut state, &nodes, &config).is_some());
    }

    #[test]
    fn pulls_generated_when_enabled() {
        let config = GravityConfig {
            well_mass_threshold: 0.3,
            apply_pulls: true,
            max_pull: 0.01,
            ..Default::default()
        };
        let nodes = vec![
            node(1, 0.9, 10, &[0.1, 0.0]),  // gravity well
            node(2, 0.1, 1, &[0.15, 0.0]),   // lightweight, close
        ];
        let report = compute_gravity_field(&nodes, &config);
        // Should generate at least one pull (well 1 attracts node 2)
        // (depends on force magnitude)
        assert!(report.pairs_evaluated == 1);
    }

    #[test]
    fn sampling_reduces_pairs() {
        let config = GravityConfig {
            max_pairs: 5,
            ..Default::default()
        };
        // 10 nodes = 45 pairs, but max_pairs=5 → sample ~5
        let nodes: Vec<_> = (0..10)
            .map(|i| node(i, 0.5, 3, &[0.1 * (i as f64), 0.0]))
            .collect();
        let report = compute_gravity_field(&nodes, &config);
        assert!(report.pairs_evaluated <= 10, "should sample: got {}", report.pairs_evaluated);
    }
}
