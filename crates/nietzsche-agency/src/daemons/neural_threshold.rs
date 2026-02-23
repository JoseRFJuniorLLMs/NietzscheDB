use nietzsche_graph::{AdjacencyIndex, GraphStorage, VectorStore};
use nietzsche_gnn::{GnnEngine, NeighborSampler};
use crate::daemons::{AgencyDaemon, DaemonReport};
use crate::config::AgencyConfig;
use crate::error::AgencyError;
use crate::event_bus::{AgencyEvent, AgencyEventBus};
use std::time::Instant;

/// NeuralThresholdDaemon (Ag.8)
///
/// This daemon uses GNN inference to adaptively tune energy thresholds
/// for other daemons. It identifies "critical" nodes that may have low energy
/// but high structural importance (GNN score), and signals the agency
/// to prevent their pruning.
pub struct NeuralThresholdDaemon {
    model_name: String,
}

impl NeuralThresholdDaemon {
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
        }
    }
}

impl AgencyDaemon for NeuralThresholdDaemon {
    fn name(&self) -> &str {
        "NeuralThresholdDaemon"
    }

    fn tick(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        bus: &AgencyEventBus,
        _config: &AgencyConfig,
    ) -> Result<DaemonReport, AgencyError> {
        let start = Instant::now();
        let mut nodes_scanned = 0;
        let mut events_emitted = 0;
        
        // 1. Initialize GNN engine
        let engine = GnnEngine::new(&self.model_name);
        let sampler = NeighborSampler::new(storage, adjacency);

        // 2. Identify candidates for neural evaluation
        // We look for nodes with low energy that might be pruned soon.
        let mut candidates = Vec::new();
        for result in storage.iter_nodes_meta() {
            let meta = result.map_err(|e| AgencyError::Internal(e.to_string()))?;
            if meta.energy < 0.2 && !meta.is_phantom {
                candidates.push(meta.id);
            }
            nodes_scanned += 1;
            if candidates.len() >= 10 { // Batch size
                break;
            }
        }

        // 3. Perform inference
        for id in candidates {
            if let Ok(subgraph) = sampler.sample_k_hop(id, 2) {
                // prediction is async, but we are in a sync trait method.
                // We need to bridge async to sync here, or change the trait.
                // Since this runs in a background thread of the server, 
                // we can use a block_on or run it in a tokio task.
                
                // For now, let's assume we can block_on (common in these server daemons handle loops).
                let rt = tokio::runtime::Handle::current();
                let predictions = rt.block_on(engine.predict(&subgraph))
                    .map_err(|e| AgencyError::Internal(e.to_string()))?;
                
                if let Some(best) = predictions.first() {
                    // If GNN importance is high (> 0.7) but energy is low, 
                    // we signal a "NeuralProtection" event.
                    if best.score > 0.7 {
                        bus.publish(AgencyEvent::NeuralProtection {
                            node_id: id,
                            importance: best.score,
                            description: format!("NeuralProtection: High structural importance ({:.2}) for low-energy node", best.score),
                        });
                        events_emitted += 1;
                    }
                }
            }
        }

        Ok(DaemonReport {
            daemon_name: self.name().to_string(),
            events_emitted,
            nodes_scanned,
            duration_us: start.elapsed().as_micros() as u64,
            details: vec![format!("Scanned candidates for neural protection")],
        })
    }
}
