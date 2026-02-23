use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use crate::daemons::{AgencyDaemon, DaemonReport};
use crate::config::AgencyConfig;
use crate::error::AgencyError;
use crate::event_bus::{AgencyEvent, AgencyEventBus};
use std::time::Instant;

/// EvolutionDaemon (Ag.12)
///
/// Implements the "Will to Power" energy dynamics.
/// Periodically boosts the energy of nodes that have accumulated 
/// high structural importance or successful reasoning hits (MCTS).
pub struct EvolutionDaemon;

impl AgencyDaemon for EvolutionDaemon {
    fn name(&self) -> &str {
        "EvolutionDaemon"
    }

    fn tick(
        &self,
        storage: &GraphStorage,
        _adjacency: &AdjacencyIndex,
        bus: &AgencyEventBus,
        _config: &AgencyConfig,
    ) -> Result<DaemonReport, AgencyError> {
        let start = Instant::now();
        let mut nodes_scanned = 0;
        let mut events_emitted = 0;
        let mut details = Vec::new();

        // Scan nodes for success metadata (e.g. mcts_hits)
        for result in storage.iter_nodes() {
            let node = result.map_err(|e| AgencyError::Internal(e.to_string()))?;
            nodes_scanned += 1;

            // Check if node has accumulated structural success
            let mcts_hits = node.content.get("mcts_hits")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            if mcts_hits > 0 {
                // Boost energy based on hits
                let boost = (0.05 * mcts_hits as f32).min(0.2);
                
                bus.publish(AgencyEvent::NeuralProtection {
                    node_id: node.id,
                    importance: 1.0, // High importance because it's a hit
                    description: format!("EvolutionBoost: Node favored by MCTS (hits={})", mcts_hits),
                });
                
                events_emitted += 1;
                details.push(format!("Node {}: boost={:.2} (hits={})", node.id, boost, mcts_hits));
            }
        }

        Ok(DaemonReport {
            daemon_name: self.name().to_string(),
            events_emitted,
            nodes_scanned,
            duration_us: start.elapsed().as_micros() as u64,
            details,
        })
    }
}
