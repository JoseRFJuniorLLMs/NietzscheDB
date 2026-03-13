// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! # Relevance Decay — frequency-based weight adjustment
//!
//! Adjusts the energy of synthesis nodes based on how frequently they are
//! referenced by subsequent inferences.
//!
//! ## Mechanism
//!
//! - Nodes referenced frequently get an energy boost
//! - Nodes never referenced after creation decay faster
//!
//! This creates a natural "survival of the fittest" for synthesis nodes:
//! useful abstractions persist, while dead-end syntheses fade away.
//!
//! ## Design
//!
//! The [`RelevanceDecay`] is a **pure computation engine** — it computes
//! energy deltas but does not access the graph directly. The caller is
//! responsible for reading/writing energy values.

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for relevance decay.
#[derive(Debug, Clone)]
pub struct RelevanceConfig {
    /// Energy boost per reference. Added to node energy on each reference.
    /// Default: 0.05
    pub boost_per_reference: f32,

    /// Base decay rate per cycle. Subtracted from energy each cycle.
    /// Default: 0.01
    pub base_decay: f32,

    /// Maximum energy a node can have (clamped).
    /// Default: 1.0
    pub max_energy: f32,

    /// Minimum energy before a node is considered prunable.
    /// Default: 0.1
    pub prune_threshold: f32,
}

impl Default for RelevanceConfig {
    fn default() -> Self {
        Self {
            boost_per_reference: 0.05,
            base_decay: 0.01,
            max_energy: 1.0,
            prune_threshold: 0.1,
        }
    }
}

// ─────────────────────────────────────────────
// EnergyDelta — what the caller should apply
// ─────────────────────────────────────────────

/// An energy change to be applied to a node.
///
/// The caller reads the current energy from the graph, computes the delta
/// via `RelevanceDecay`, and then writes the new energy back via
/// `NietzscheDB::update_energy(node_id, new_energy)`.
#[derive(Debug, Clone, Copy)]
pub struct EnergyDelta {
    /// The node to update.
    pub node_id: uuid::Uuid,
    /// The new energy value (already clamped).
    pub new_energy: f32,
    /// Whether the node is now below the prune threshold.
    pub is_prunable: bool,
}

// ─────────────────────────────────────────────
// RelevanceDecay
// ─────────────────────────────────────────────

/// Pure computation engine for frequency-based energy adjustment.
pub struct RelevanceDecay {
    config: RelevanceConfig,
}

impl RelevanceDecay {
    pub fn new(config: RelevanceConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(RelevanceConfig::default())
    }

    /// Compute the energy boost for a node that was just referenced.
    ///
    /// # Arguments
    /// - `node_id`: the node being referenced
    /// - `current_energy`: the node's current energy
    ///
    /// # Returns
    /// An [`EnergyDelta`] with the new energy value.
    pub fn compute_boost(
        &self,
        node_id: uuid::Uuid,
        current_energy: f32,
    ) -> EnergyDelta {
        let new_energy = (current_energy + self.config.boost_per_reference)
            .min(self.config.max_energy);
        EnergyDelta {
            node_id,
            new_energy,
            is_prunable: new_energy < self.config.prune_threshold,
        }
    }

    /// Compute the energy decay for a node during periodic maintenance.
    ///
    /// # Arguments
    /// - `node_id`: the node to decay
    /// - `current_energy`: the node's current energy
    ///
    /// # Returns
    /// An [`EnergyDelta`] with the new energy value.
    pub fn compute_decay(
        &self,
        node_id: uuid::Uuid,
        current_energy: f32,
    ) -> EnergyDelta {
        let new_energy = (current_energy - self.config.base_decay).max(0.0);
        EnergyDelta {
            node_id,
            new_energy,
            is_prunable: new_energy < self.config.prune_threshold,
        }
    }

    /// Compute the projected energy after `n` references and `cycles` decay cycles.
    /// Pure function — does not touch the graph.
    pub fn projected_energy(&self, initial: f32, references: u32, cycles: u32) -> f32 {
        let boost = self.config.boost_per_reference * references as f32;
        let decay = self.config.base_decay * cycles as f32;
        (initial + boost - decay).clamp(0.0, self.config.max_energy)
    }

    /// Batch compute decay for multiple nodes.
    ///
    /// # Arguments
    /// - `nodes`: Vec of (node_id, current_energy)
    ///
    /// # Returns
    /// Vec of [`EnergyDelta`] for each node.
    pub fn batch_decay(&self, nodes: &[(uuid::Uuid, f32)]) -> Vec<EnergyDelta> {
        nodes
            .iter()
            .map(|(id, energy)| self.compute_decay(*id, *energy))
            .collect()
    }

    pub fn config(&self) -> &RelevanceConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_compute_boost() {
        let rd = RelevanceDecay::with_defaults();
        let delta = rd.compute_boost(Uuid::new_v4(), 0.5);
        assert!((delta.new_energy - 0.55).abs() < 1e-5);
        assert!(!delta.is_prunable);
    }

    #[test]
    fn test_compute_boost_capped() {
        let rd = RelevanceDecay::with_defaults();
        let delta = rd.compute_boost(Uuid::new_v4(), 0.98);
        assert!((delta.new_energy - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_decay() {
        let rd = RelevanceDecay::with_defaults();
        let delta = rd.compute_decay(Uuid::new_v4(), 0.5);
        assert!((delta.new_energy - 0.49).abs() < 1e-5);
        assert!(!delta.is_prunable);
    }

    #[test]
    fn test_compute_decay_prunable() {
        let rd = RelevanceDecay::with_defaults();
        let delta = rd.compute_decay(Uuid::new_v4(), 0.05);
        assert!((delta.new_energy - 0.04).abs() < 1e-5);
        assert!(delta.is_prunable);
    }

    #[test]
    fn test_compute_decay_floor() {
        let rd = RelevanceDecay::with_defaults();
        let delta = rd.compute_decay(Uuid::new_v4(), 0.005);
        assert!((delta.new_energy - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_projected_energy() {
        let rd = RelevanceDecay::with_defaults();

        // Initial 1.0, 0 refs, 10 cycles → 1.0 - 0.1 = 0.9
        assert!((rd.projected_energy(1.0, 0, 10) - 0.9).abs() < 1e-5);

        // Initial 0.5, 5 refs, 0 cycles → 0.5 + 0.25 = 0.75
        assert!((rd.projected_energy(0.5, 5, 0) - 0.75).abs() < 1e-5);

        // Initial 0.1, 0 refs, 20 cycles → max(0.1 - 0.2, 0) = 0.0
        assert!((rd.projected_energy(0.1, 0, 20) - 0.0).abs() < 1e-5);

        // Capped at max
        assert!((rd.projected_energy(0.9, 10, 0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_decay() {
        let rd = RelevanceDecay::with_defaults();
        let nodes = vec![
            (Uuid::new_v4(), 0.5),
            (Uuid::new_v4(), 0.05),
            (Uuid::new_v4(), 1.0),
        ];
        let deltas = rd.batch_decay(&nodes);
        assert_eq!(deltas.len(), 3);
        assert!(!deltas[0].is_prunable);
        assert!(deltas[1].is_prunable);
        assert!(!deltas[2].is_prunable);
    }
}
