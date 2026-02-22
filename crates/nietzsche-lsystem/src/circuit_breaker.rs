//! **EnergyCircuitBreaker** — anti-tumor mechanism for the hyperbolic graph.
//!
//! The basic L-System circuit breaker (μ + kσ spawn-blocking) operates only
//! within the L-System tick.  This module provides a **cross-system** circuit
//! breaker that:
//!
//! 1. **Depth-aware energy caps** — nodes near the Poincaré boundary (specific
//!    memories) have a lower energy ceiling than hub nodes near the centre.
//! 2. **Tumor detection** — BFS over the adjacency graph to find connected
//!    clusters of "overheated" nodes (energy > threshold).
//! 3. **Energy dampening** — force-drain tumors by multiplying their energy
//!    by a dampening factor, preventing runaway feedback loops between
//!    DAEMONs, L-System, and Will-to-Power.
//!
//! # Integration
//!
//! - Called **after** Will-to-Power and **before** L-System tick during the
//!   Zaratustra cycle.
//! - Emits `TumorDetected` events on the `AgencyEventBus` when clusters are
//!   found, allowing the MetaObserver to react.

use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the cross-system energy circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Global energy threshold above which a node is considered "overheated".
    ///
    /// Nodes with `energy > overheated_threshold` are candidates for tumor
    /// clustering.  Default: `0.92`.
    pub overheated_threshold: f32,

    /// Minimum cluster size to qualify as a tumor.
    ///
    /// Isolated hot nodes are normal.  A *tumor* is a connected component of
    /// overheated nodes of size ≥ this value.  Default: `3`.
    pub min_tumor_size: usize,

    /// Dampening factor applied to tumor nodes.
    ///
    /// Each tumor node's energy is multiplied by this value:
    /// `new_energy = old_energy × dampening_factor`.
    /// Default: `0.7` (30% drain per cycle).
    pub dampening_factor: f32,

    /// Gradient for depth-aware energy caps.
    ///
    /// `cap(depth) = base_cap - depth_cap_gradient × depth`
    ///
    /// Deeper nodes (higher ‖embedding‖, closer to boundary) get a lower cap.
    /// This prevents boundary nodes from accumulating excessive energy.
    ///
    /// Default: `0.3` → at depth=0.9, cap = 1.0 - 0.3×0.9 = 0.73.
    pub depth_cap_gradient: f32,

    /// Base energy cap (at depth=0).  Default: `1.0`.
    pub base_cap: f32,

    /// Maximum energy delta (absolute) allowed per node per cycle.
    ///
    /// If Will-to-Power would increase a node's energy by more than this
    /// value in a single cycle, the increase is clamped.
    /// `0.0` = disabled (no rate limiting).
    /// Default: `0.25`.
    pub max_energy_delta: f32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            overheated_threshold: 0.92,
            min_tumor_size: 3,
            dampening_factor: 0.7,
            depth_cap_gradient: 0.3,
            base_cap: 1.0,
            max_energy_delta: 0.25,
        }
    }
}

// ─────────────────────────────────────────────
// Reports
// ─────────────────────────────────────────────

/// A connected cluster of overheated nodes (a "semantic tumor").
#[derive(Debug, Clone)]
pub struct TumorCluster {
    /// Node IDs in this cluster.
    pub node_ids: Vec<Uuid>,
    /// Mean energy of cluster nodes (before dampening).
    pub mean_energy: f32,
    /// Max energy in the cluster.
    pub max_energy: f32,
}

/// Report returned by [`scan_and_dampen`].
#[derive(Debug, Clone, Default)]
pub struct TumorReport {
    /// Number of tumor clusters detected.
    pub tumors_found: usize,
    /// Total nodes across all tumors.
    pub total_tumor_nodes: usize,
    /// Number of nodes whose energy was dampened.
    pub nodes_dampened: usize,
    /// Number of nodes whose energy was capped by the depth-aware cap.
    pub nodes_depth_capped: usize,
    /// Number of nodes whose energy delta was rate-limited.
    pub nodes_rate_limited: usize,
    /// Details of each detected tumor.
    pub clusters: Vec<TumorCluster>,
}

// ─────────────────────────────────────────────
// Core functions
// ─────────────────────────────────────────────

/// Compute the depth-aware energy cap for a node at the given `depth`
/// (‖embedding‖ in the Poincaré ball, ∈ [0, 1)).
///
/// Deeper (more specific) nodes get a lower cap, preventing boundary
/// nodes from accumulating excessive energy.
pub fn depth_aware_cap(depth: f32, config: &CircuitBreakerConfig) -> f32 {
    (config.base_cap - config.depth_cap_gradient * depth).max(0.1)
}

/// Apply depth-aware energy caps to a set of nodes.
///
/// Returns a map of `node_id → new_energy` for nodes that were capped.
pub fn apply_depth_caps(
    nodes: &[(Uuid, f32, f32)], // (id, energy, depth)
    config: &CircuitBreakerConfig,
) -> Vec<(Uuid, f32)> {
    let mut capped = Vec::new();
    for &(id, energy, depth) in nodes {
        let cap = depth_aware_cap(depth, config);
        if energy > cap {
            capped.push((id, cap));
        }
    }
    capped
}

/// Apply energy delta rate limiting.
///
/// Given `(node_id, old_energy, new_energy)` triples, clamp any increase
/// that exceeds `config.max_energy_delta`.
///
/// Returns adjusted `(node_id, clamped_energy)` for nodes that were limited.
pub fn apply_rate_limiting(
    deltas: &[(Uuid, f32, f32)], // (id, old_energy, new_energy)
    config: &CircuitBreakerConfig,
) -> Vec<(Uuid, f32)> {
    if config.max_energy_delta <= 0.0 {
        return Vec::new();
    }

    let mut limited = Vec::new();
    for &(id, old_e, new_e) in deltas {
        let delta = new_e - old_e;
        if delta > config.max_energy_delta {
            limited.push((id, old_e + config.max_energy_delta));
        }
    }
    limited
}

/// Detect tumor clusters via BFS over the adjacency graph.
///
/// A tumor is a connected component of nodes whose energy exceeds
/// `config.overheated_threshold` and whose size ≥ `config.min_tumor_size`.
///
/// # Arguments
///
/// - `node_energies` — map of `node_id → energy` for all live nodes.
/// - `neighbors_fn` — function returning the bidirectional neighbors of a node.
/// - `config` — circuit breaker configuration.
pub fn detect_tumors<F>(
    node_energies: &HashMap<Uuid, f32>,
    neighbors_fn: F,
    config: &CircuitBreakerConfig,
) -> Vec<TumorCluster>
where
    F: Fn(&Uuid) -> Vec<Uuid>,
{
    // Collect overheated nodes
    let overheated: HashSet<Uuid> = node_energies
        .iter()
        .filter(|(_, &e)| e > config.overheated_threshold)
        .map(|(&id, _)| id)
        .collect();

    if overheated.len() < config.min_tumor_size {
        return Vec::new();
    }

    // BFS to find connected components among overheated nodes
    let mut visited: HashSet<Uuid> = HashSet::new();
    let mut clusters: Vec<TumorCluster> = Vec::new();

    for &seed in &overheated {
        if visited.contains(&seed) {
            continue;
        }

        // BFS from seed
        let mut queue = VecDeque::new();
        let mut component = Vec::new();
        queue.push_back(seed);
        visited.insert(seed);

        while let Some(current) = queue.pop_front() {
            component.push(current);

            for neighbor in neighbors_fn(&current) {
                if overheated.contains(&neighbor) && visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        if component.len() >= config.min_tumor_size {
            let energies: Vec<f32> = component
                .iter()
                .filter_map(|id| node_energies.get(id).copied())
                .collect();
            let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;
            let max_energy = energies.iter().cloned().fold(0.0f32, f32::max);

            clusters.push(TumorCluster {
                node_ids: component,
                mean_energy,
                max_energy,
            });
        }
    }

    clusters
}

/// Full scan-and-dampen cycle: detect tumors, apply depth caps, and dampen.
///
/// Returns `(TumorReport, Vec<(Uuid, f32)>)` — the report and a list of
/// `(node_id, new_energy)` pairs that should be persisted.
///
/// The caller is responsible for persisting the energy updates (via
/// `NietzscheDB::update_energy` or `GraphStorage::put_node`).
pub fn scan_and_dampen<F>(
    node_energies: &HashMap<Uuid, f32>,
    node_depths: &HashMap<Uuid, f32>,
    neighbors_fn: F,
    config: &CircuitBreakerConfig,
) -> (TumorReport, Vec<(Uuid, f32)>)
where
    F: Fn(&Uuid) -> Vec<Uuid>,
{
    let mut report = TumorReport::default();
    let mut updates: HashMap<Uuid, f32> = HashMap::new();

    // Phase 1: Depth-aware caps
    for (&id, &energy) in node_energies {
        let depth = node_depths.get(&id).copied().unwrap_or(0.0);
        let cap = depth_aware_cap(depth, config);
        if energy > cap {
            updates.insert(id, cap);
            report.nodes_depth_capped += 1;
        }
    }

    // Phase 2: Tumor detection
    let clusters = detect_tumors(node_energies, neighbors_fn, config);
    report.tumors_found = clusters.len();
    report.total_tumor_nodes = clusters.iter().map(|c| c.node_ids.len()).sum();

    // Phase 3: Dampening
    for cluster in &clusters {
        for &node_id in &cluster.node_ids {
            let current = updates
                .get(&node_id)
                .copied()
                .or_else(|| node_energies.get(&node_id).copied())
                .unwrap_or(0.0);
            let dampened = current * config.dampening_factor;
            updates.insert(node_id, dampened);
            report.nodes_dampened += 1;
        }
    }

    report.clusters = clusters;

    let result: Vec<(Uuid, f32)> = updates.into_iter().collect();
    (report, result)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> CircuitBreakerConfig {
        CircuitBreakerConfig::default()
    }

    #[test]
    fn depth_aware_cap_center_node() {
        let cfg = default_config();
        // Depth 0 (center) → cap = 1.0
        assert!((depth_aware_cap(0.0, &cfg) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn depth_aware_cap_boundary_node() {
        let cfg = default_config();
        // Depth 0.9 → cap = 1.0 - 0.3*0.9 = 0.73
        let cap = depth_aware_cap(0.9, &cfg);
        assert!((cap - 0.73).abs() < 1e-5, "expected ~0.73, got {cap}");
    }

    #[test]
    fn depth_aware_cap_never_below_minimum() {
        let cfg = CircuitBreakerConfig {
            depth_cap_gradient: 5.0, // extreme gradient
            ..default_config()
        };
        // Even at depth 0.9: 1.0 - 5.0*0.9 = -3.5 → clamped to 0.1
        assert!((depth_aware_cap(0.9, &cfg) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn no_tumors_when_below_threshold() {
        let cfg = default_config();
        let mut energies = HashMap::new();
        for i in 0..10u32 {
            energies.insert(Uuid::new_v4(), 0.5); // all below 0.92
        }

        let tumors = detect_tumors(&energies, |_| Vec::new(), &cfg);
        assert!(tumors.is_empty());
    }

    #[test]
    fn isolated_hot_nodes_not_a_tumor() {
        let cfg = default_config(); // min_tumor_size = 3
        let mut energies = HashMap::new();

        // 2 hot nodes, but not connected
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        energies.insert(a, 0.99);
        energies.insert(b, 0.98);

        let tumors = detect_tumors(&energies, |_| Vec::new(), &cfg);
        assert!(tumors.is_empty(), "2 isolated hot nodes should not form a tumor");
    }

    #[test]
    fn connected_hot_cluster_is_tumor() {
        let cfg = default_config(); // min_tumor_size = 3

        let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
        let mut energies = HashMap::new();
        for &id in &ids {
            energies.insert(id, 0.95); // all overheated
        }

        // Build a chain: 0-1-2-3-4
        let adjacency: HashMap<Uuid, Vec<Uuid>> = ids
            .windows(2)
            .flat_map(|w| vec![(w[0], w[1]), (w[1], w[0])])
            .fold(HashMap::new(), |mut acc, (from, to)| {
                acc.entry(from).or_default().push(to);
                acc
            });

        let tumors = detect_tumors(&energies, |id| {
            adjacency.get(id).cloned().unwrap_or_default()
        }, &cfg);

        assert_eq!(tumors.len(), 1, "expected one tumor cluster");
        assert_eq!(tumors[0].node_ids.len(), 5);
        assert!(tumors[0].mean_energy > 0.94);
    }

    #[test]
    fn dampening_reduces_tumor_energy() {
        let cfg = CircuitBreakerConfig {
            overheated_threshold: 0.9,
            min_tumor_size: 2,
            dampening_factor: 0.5,
            ..default_config()
        };

        let ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
        let mut energies = HashMap::new();
        let mut depths = HashMap::new();
        for &id in &ids {
            energies.insert(id, 0.95);
            depths.insert(id, 0.1); // near center, cap ~ 0.97
        }

        // Chain: 0-1-2
        let adjacency: HashMap<Uuid, Vec<Uuid>> = vec![
            (ids[0], vec![ids[1]]),
            (ids[1], vec![ids[0], ids[2]]),
            (ids[2], vec![ids[1]]),
        ]
        .into_iter()
        .collect();

        let (report, updates) = scan_and_dampen(
            &energies,
            &depths,
            |id| adjacency.get(id).cloned().unwrap_or_default(),
            &cfg,
        );

        assert_eq!(report.tumors_found, 1);
        assert_eq!(report.nodes_dampened, 3);

        // All nodes should be dampened to ~0.475 (0.95 * 0.5)
        for (id, new_e) in &updates {
            assert!(*new_e < 0.5, "node {id} should be dampened, got {new_e}");
        }
    }

    #[test]
    fn rate_limiting_clamps_large_delta() {
        let cfg = CircuitBreakerConfig {
            max_energy_delta: 0.2,
            ..default_config()
        };

        let id = Uuid::new_v4();
        let deltas = vec![(id, 0.3, 0.8)]; // delta = 0.5 > 0.2

        let limited = apply_rate_limiting(&deltas, &cfg);
        assert_eq!(limited.len(), 1);
        assert!((limited[0].1 - 0.5).abs() < 1e-6, "expected 0.3 + 0.2 = 0.5");
    }

    #[test]
    fn rate_limiting_allows_small_delta() {
        let cfg = CircuitBreakerConfig {
            max_energy_delta: 0.2,
            ..default_config()
        };

        let id = Uuid::new_v4();
        let deltas = vec![(id, 0.3, 0.4)]; // delta = 0.1 < 0.2

        let limited = apply_rate_limiting(&deltas, &cfg);
        assert!(limited.is_empty(), "small delta should not be rate limited");
    }

    #[test]
    fn rate_limiting_disabled_when_zero() {
        let cfg = CircuitBreakerConfig {
            max_energy_delta: 0.0,
            ..default_config()
        };

        let id = Uuid::new_v4();
        let deltas = vec![(id, 0.1, 0.9)]; // delta = 0.8

        let limited = apply_rate_limiting(&deltas, &cfg);
        assert!(limited.is_empty(), "rate limiting should be disabled when max_delta=0");
    }

    #[test]
    fn full_scan_and_dampen_pipeline() {
        let cfg = CircuitBreakerConfig {
            overheated_threshold: 0.85,
            min_tumor_size: 2,
            dampening_factor: 0.6,
            depth_cap_gradient: 0.4,
            base_cap: 1.0,
            ..default_config()
        };

        // 3 hot connected nodes + 2 cool nodes
        let hot: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
        let cool: Vec<Uuid> = (0..2).map(|_| Uuid::new_v4()).collect();

        let mut energies = HashMap::new();
        let mut depths = HashMap::new();
        for &id in &hot {
            energies.insert(id, 0.95);
            depths.insert(id, 0.8); // deep → cap = 1.0 - 0.4*0.8 = 0.68
        }
        for &id in &cool {
            energies.insert(id, 0.4);
            depths.insert(id, 0.2);
        }

        let adjacency: HashMap<Uuid, Vec<Uuid>> = vec![
            (hot[0], vec![hot[1]]),
            (hot[1], vec![hot[0], hot[2]]),
            (hot[2], vec![hot[1]]),
        ]
        .into_iter()
        .collect();

        let (report, updates) = scan_and_dampen(
            &energies,
            &depths,
            |id| adjacency.get(id).cloned().unwrap_or_default(),
            &cfg,
        );

        assert_eq!(report.tumors_found, 1);
        assert_eq!(report.total_tumor_nodes, 3);
        assert!(report.nodes_depth_capped >= 3, "hot deep nodes should be depth-capped");
        assert_eq!(report.nodes_dampened, 3);

        // Updates map
        let update_map: HashMap<Uuid, f32> = updates.into_iter().collect();

        // Hot nodes: depth-capped to 0.68, then dampened to 0.68 * 0.6 = 0.408
        for &id in &hot {
            let e = update_map[&id];
            assert!(e < 0.5, "hot node should be severely dampened, got {e}");
        }

        // Cool nodes: not affected (not in updates or unchanged)
        for &id in &cool {
            assert!(
                !update_map.contains_key(&id),
                "cool node should not be in updates"
            );
        }
    }

    #[test]
    fn multiple_separate_tumors() {
        let cfg = CircuitBreakerConfig {
            overheated_threshold: 0.9,
            min_tumor_size: 2,
            ..default_config()
        };

        // Two separate clusters of 2 hot nodes each
        let cluster_a: Vec<Uuid> = (0..2).map(|_| Uuid::new_v4()).collect();
        let cluster_b: Vec<Uuid> = (0..2).map(|_| Uuid::new_v4()).collect();

        let mut energies = HashMap::new();
        for &id in cluster_a.iter().chain(cluster_b.iter()) {
            energies.insert(id, 0.95);
        }

        let mut adjacency: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        adjacency.insert(cluster_a[0], vec![cluster_a[1]]);
        adjacency.insert(cluster_a[1], vec![cluster_a[0]]);
        adjacency.insert(cluster_b[0], vec![cluster_b[1]]);
        adjacency.insert(cluster_b[1], vec![cluster_b[0]]);

        let tumors = detect_tumors(&energies, |id| {
            adjacency.get(id).cloned().unwrap_or_default()
        }, &cfg);

        assert_eq!(tumors.len(), 2, "expected two separate tumors");
    }
}
