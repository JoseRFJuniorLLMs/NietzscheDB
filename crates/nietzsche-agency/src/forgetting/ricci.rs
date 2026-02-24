//! Ricci Curvature Shield — Geometric veto for the Forgetting Engine.
//!
//! Before any hard delete, the system simulates the removal and checks
//! if the local Ricci curvature would collapse beyond the threshold.
//!
//! ## Approximation
//!
//! True Ollivier-Ricci curvature requires optimal transport (Wasserstein
//! distance between neighbor distributions), which is O(n³) per edge.
//! Instead we use a lightweight **degree-variance proxy**:
//!
//! κ_proxy(v) = 1 - Var(deg(N(v))) / E[deg(N(v))]²
//!
//! - Uniform neighborhoods → high proxy (flat/positive curvature)
//! - Highly irregular neighborhoods → low proxy (negative curvature)
//!
//! The ΔRicci is the change in mean proxy curvature when a node is removed.

use uuid::Uuid;
use std::collections::{HashMap, HashSet};

/// Result of a Ricci curvature simulation.
#[derive(Debug, Clone)]
pub struct RicciSimulation {
    /// The node being evaluated for removal.
    pub node_id: Uuid,
    /// Proxy curvature before removal.
    pub curvature_before: f32,
    /// Proxy curvature after simulated removal.
    pub curvature_after: f32,
    /// ΔRicci = after - before. Negative = topology collapse.
    pub delta: f32,
    /// Whether the veto was triggered (delta < -epsilon).
    pub veto_triggered: bool,
}

/// Lightweight Ricci curvature approximator using degree-variance proxy.
///
/// Operates on an adjacency snapshot (no graph mutation needed).
pub struct RicciShield {
    /// The threshold ε: if ΔRicci < -ε, veto the deletion.
    pub epsilon: f32,
}

impl RicciShield {
    pub fn new(epsilon: f32) -> Self {
        Self { epsilon: epsilon.abs() }
    }

    /// Compute the degree-variance proxy curvature for the neighborhood
    /// of a given node.
    ///
    /// κ_proxy = 1 - Var(degrees) / Mean(degrees)² (coefficient of variation squared)
    ///
    /// Returns 0.0 if no neighbors.
    fn local_curvature_proxy(
        &self,
        node_id: &Uuid,
        adjacency_out: &HashMap<Uuid, Vec<Uuid>>,
        adjacency_in: &HashMap<Uuid, Vec<Uuid>>,
        excluded: &HashSet<Uuid>,
    ) -> f32 {
        // Get all neighbors (both directions)
        let mut neighbors: HashSet<Uuid> = HashSet::new();

        if let Some(outs) = adjacency_out.get(node_id) {
            for n in outs {
                if !excluded.contains(n) {
                    neighbors.insert(*n);
                }
            }
        }
        if let Some(ins) = adjacency_in.get(node_id) {
            for n in ins {
                if !excluded.contains(n) {
                    neighbors.insert(*n);
                }
            }
        }

        if neighbors.is_empty() {
            return 0.0;
        }

        // Collect degrees of neighbors
        let degrees: Vec<f32> = neighbors.iter().map(|n| {
            let out_deg = adjacency_out.get(n)
                .map(|v| v.iter().filter(|x| !excluded.contains(x)).count())
                .unwrap_or(0);
            let in_deg = adjacency_in.get(n)
                .map(|v| v.iter().filter(|x| !excluded.contains(x)).count())
                .unwrap_or(0);
            (out_deg + in_deg) as f32
        }).collect();

        let n = degrees.len() as f32;
        let mean = degrees.iter().sum::<f32>() / n;

        if mean < 1e-6 {
            return 0.0;
        }

        let variance = degrees.iter()
            .map(|d| (d - mean).powi(2))
            .sum::<f32>() / n;

        // κ_proxy = 1 - CV² where CV = σ/μ
        1.0 - (variance / (mean * mean))
    }

    /// Compute the mean curvature proxy across all active nodes.
    fn mean_curvature(
        &self,
        active_nodes: &[Uuid],
        adjacency_out: &HashMap<Uuid, Vec<Uuid>>,
        adjacency_in: &HashMap<Uuid, Vec<Uuid>>,
        excluded: &HashSet<Uuid>,
    ) -> f32 {
        if active_nodes.is_empty() {
            return 0.0;
        }

        let sum: f32 = active_nodes.iter()
            .filter(|n| !excluded.contains(n))
            .map(|n| self.local_curvature_proxy(n, adjacency_out, adjacency_in, excluded))
            .sum();

        let count = active_nodes.iter().filter(|n| !excluded.contains(n)).count();
        if count == 0 { 0.0 } else { sum / count as f32 }
    }

    /// Simulate the removal of a node and compute ΔRicci.
    ///
    /// Returns `RicciSimulation` with the delta and whether the veto fired.
    pub fn simulate_removal(
        &self,
        node_id: &Uuid,
        active_nodes: &[Uuid],
        adjacency_out: &HashMap<Uuid, Vec<Uuid>>,
        adjacency_in: &HashMap<Uuid, Vec<Uuid>>,
    ) -> RicciSimulation {
        let excluded_empty = HashSet::new();

        // Curvature before removal
        let curvature_before = self.mean_curvature(
            active_nodes, adjacency_out, adjacency_in, &excluded_empty,
        );

        // Curvature after simulated removal
        let mut excluded_with_node = HashSet::new();
        excluded_with_node.insert(*node_id);
        let curvature_after = self.mean_curvature(
            active_nodes, adjacency_out, adjacency_in, &excluded_with_node,
        );

        let delta = curvature_after - curvature_before;
        let veto_triggered = delta < -self.epsilon;

        RicciSimulation {
            node_id: *node_id,
            curvature_before,
            curvature_after,
            delta,
            veto_triggered,
        }
    }

    /// Quick check: should we veto this deletion?
    ///
    /// For performance, this only checks the LOCAL neighborhood rather
    /// than recomputing global mean curvature.
    pub fn quick_veto(
        &self,
        node_id: &Uuid,
        adjacency_out: &HashMap<Uuid, Vec<Uuid>>,
        adjacency_in: &HashMap<Uuid, Vec<Uuid>>,
    ) -> bool {
        let excluded_empty = HashSet::new();
        let local_before = self.local_curvature_proxy(
            node_id, adjacency_out, adjacency_in, &excluded_empty,
        );

        // If this node has very high local curvature (it's a hub connecting
        // uniform neighborhoods), removing it would likely cause collapse.
        // Threshold: if local curvature > 0.5 and the node has many edges,
        // trigger veto without full simulation.
        let total_edges = adjacency_out.get(node_id).map(|v| v.len()).unwrap_or(0)
            + adjacency_in.get(node_id).map(|v| v.len()).unwrap_or(0);

        local_before > 0.5 && total_edges > 5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_star_graph(center: Uuid, leaves: &[Uuid]) -> (HashMap<Uuid, Vec<Uuid>>, HashMap<Uuid, Vec<Uuid>>) {
        let mut adj_out = HashMap::new();
        let mut adj_in = HashMap::new();

        let leaf_vec: Vec<Uuid> = leaves.to_vec();
        adj_out.insert(center, leaf_vec.clone());

        for &leaf in leaves {
            adj_in.entry(leaf).or_insert_with(Vec::new).push(center);
        }

        (adj_out, adj_in)
    }

    #[test]
    fn removing_hub_triggers_veto() {
        let center = Uuid::new_v4();
        let leaves: Vec<Uuid> = (0..10).map(|_| Uuid::new_v4()).collect();
        let (adj_out, adj_in) = make_star_graph(center, &leaves);

        let shield = RicciShield::new(0.05);
        let mut all_nodes = vec![center];
        all_nodes.extend_from_slice(&leaves);

        let sim = shield.simulate_removal(&center, &all_nodes, &adj_out, &adj_in);
        // Removing the hub of a star should cause significant curvature change
        assert!(sim.delta < 0.0, "removing star center should decrease curvature: delta={}", sim.delta);
    }

    #[test]
    fn removing_leaf_no_veto() {
        let center = Uuid::new_v4();
        let leaves: Vec<Uuid> = (0..10).map(|_| Uuid::new_v4()).collect();
        let (adj_out, adj_in) = make_star_graph(center, &leaves);

        let shield = RicciShield::new(0.15);
        let mut all_nodes = vec![center];
        all_nodes.extend_from_slice(&leaves);

        // Removing one leaf from a 10-leaf star should be safe
        let sim = shield.simulate_removal(&leaves[0], &all_nodes, &adj_out, &adj_in);
        assert!(!sim.veto_triggered, "removing single leaf should not trigger veto");
    }

    #[test]
    fn isolated_node_no_veto() {
        let node = Uuid::new_v4();
        let adj_out = HashMap::new();
        let adj_in = HashMap::new();

        let shield = RicciShield::new(0.1);
        let sim = shield.simulate_removal(&node, &[node], &adj_out, &adj_in);
        assert!(!sim.veto_triggered, "isolated node removal is always safe");
    }
}
