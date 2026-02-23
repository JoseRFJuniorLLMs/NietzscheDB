use nietzsche_graph::GraphStorage;
use crate::error::AgencyError;

/// Prevents "energy storms" or pathological activation clusters in the graph.
/// Implements the safety guidelines from EVA's clinical cortex.
pub struct EnergyCircuitBreaker {
    pub max_active_reflexes: usize,
    pub energy_sum_threshold: f32,
}

impl Default for EnergyCircuitBreaker {
    fn default() -> Self {
        Self {
            max_active_reflexes: 20,
            energy_sum_threshold: 50.0,
        }
    }
}

impl EnergyCircuitBreaker {
    /// Checks if the current state of the graph allows for more autonomous actions.
    pub fn check_safety(&self, storage: &GraphStorage, activated_count: usize) -> Result<bool, AgencyError> {
        // 1. Absolute count check
        if activated_count > self.max_active_reflexes {
            tracing::warn!(activated_count, "Circuit breaker tripped: too many active reflexes");
            return Ok(false);
        }

        // 2. Global energy density check (sum of energy across all nodes)
        let mut total_energy = 0.0;
        let mut node_count = 0;
        
        for result in storage.iter_nodes_meta() {
            if let Ok(meta) = result {
                total_energy += meta.energy;
                node_count += 1;
            }
        }

        if total_energy > self.energy_sum_threshold {
            tracing::warn!(
                total_energy, 
                node_count, 
                "Circuit breaker tripped: pathological energy accumulation"
            );
            return Ok(false);
        }

        Ok(true)
    }
}
