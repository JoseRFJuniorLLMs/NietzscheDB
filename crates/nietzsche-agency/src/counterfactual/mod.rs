pub mod shadow;
pub mod simulator;

use uuid::Uuid;
use nietzsche_graph::{AdjacencyIndex, GraphStorage, NodeMeta};

use crate::config::AgencyConfig;
use crate::error::AgencyError;

use shadow::ShadowGraph;
use simulator::EnergyChange;

/// A hypothetical operation to simulate.
#[derive(Debug, Clone)]
pub enum CounterfactualOp {
    RemoveNode { id: Uuid },
    AddNode { meta: NodeMeta, connect_to: Vec<Uuid> },
}

/// Result of a counterfactual simulation.
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    pub operation: CounterfactualOp,
    pub impact_scores: Vec<(Uuid, f64)>,
    pub energy_changes: Vec<EnergyChange>,
    pub mean_energy_delta: f32,
    pub affected_radius: usize,
}

/// Stateless counterfactual engine for "what-if" analysis.
///
/// Creates a lightweight shadow copy of the graph, applies a hypothetical
/// mutation, and predicts impact via BFS energy propagation. The real
/// graph is never modified.
pub struct CounterfactualEngine;

impl CounterfactualEngine {
    /// Simulate "what if we remove node X?"
    pub fn what_if_remove(
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        node_id: Uuid,
        config: &AgencyConfig,
    ) -> Result<CounterfactualResult, AgencyError> {
        let mut shadow = ShadowGraph::snapshot(storage, adjacency)?;
        let sim = simulator::simulate_remove(&mut shadow, node_id, config);

        Ok(CounterfactualResult {
            operation: CounterfactualOp::RemoveNode { id: node_id },
            impact_scores: sim.impact_scores,
            energy_changes: sim.energy_changes,
            mean_energy_delta: sim.mean_energy_delta,
            affected_radius: sim.affected_radius,
        })
    }

    /// Simulate "what if we add a node connected to these nodes?"
    pub fn what_if_add(
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        new_meta: NodeMeta,
        connect_to: Vec<Uuid>,
        config: &AgencyConfig,
    ) -> Result<CounterfactualResult, AgencyError> {
        let mut shadow = ShadowGraph::snapshot(storage, adjacency)?;
        let meta_clone = new_meta.clone();
        let sim = simulator::simulate_add(&mut shadow, new_meta, &connect_to, config);

        Ok(CounterfactualResult {
            operation: CounterfactualOp::AddNode {
                meta: meta_clone,
                connect_to,
            },
            impact_scores: sim.impact_scores,
            energy_changes: sim.energy_changes,
            mean_energy_delta: sim.mean_energy_delta,
            affected_radius: sim.affected_radius,
        })
    }
}
