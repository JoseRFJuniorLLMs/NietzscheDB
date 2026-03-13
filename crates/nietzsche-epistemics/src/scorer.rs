// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Unified epistemic scorer.
//!
//! Combines all individual metrics into a composite `EpistemicScore`
//! and provides delta comparison for mutation evaluation.

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use uuid::Uuid;

use crate::coherence;
use crate::coverage;
use crate::hierarchy;
use crate::redundancy;

/// Weights for computing the composite epistemic score.
#[derive(Debug, Clone)]
pub struct ScoreWeights {
    pub hierarchy: f32,
    pub coherence: f32,
    pub coverage: f32,
    pub redundancy: f32,
    pub energy: f32,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            hierarchy: 0.25,
            coherence: 0.25,
            coverage: 0.15,
            redundancy: 0.15,
            energy: 0.20,
        }
    }
}

/// Comprehensive epistemic quality score for a subgraph.
#[derive(Debug, Clone)]
pub struct EpistemicScore {
    /// Fraction of edges respecting depth hierarchy [0, 1].
    pub hierarchy_consistency: f32,
    /// Semantic coherence of connected nodes [0, 1].
    pub coherence: f32,
    /// Spatial coverage of the Poincaré ball [0, 1].
    pub coverage: f32,
    /// Path diversity / connectivity [0, 1].
    pub redundancy: f32,
    /// Average energy of evaluated nodes [0, 1].
    pub energy_avg: f32,
    /// Hausdorff local average.
    pub hausdorff_avg: f32,
    /// Number of nodes evaluated.
    pub node_count: usize,
    /// Weighted composite score [0, 1].
    pub composite: f32,
}

/// Delta between two epistemic scores (after - before).
#[derive(Debug, Clone)]
pub struct EpistemicDelta {
    pub hierarchy_delta: f32,
    pub coherence_delta: f32,
    pub coverage_delta: f32,
    pub redundancy_delta: f32,
    pub energy_delta: f32,
    /// Weighted composite delta (positive = improvement).
    pub composite_delta: f32,
}

impl EpistemicDelta {
    /// Whether the mutation improved the graph.
    pub fn is_improvement(&self, threshold: f32) -> bool {
        self.composite_delta >= threshold
    }
}

/// Evaluate epistemic quality of a subgraph.
///
/// Computes all individual metrics and produces a weighted composite score.
pub fn evaluate_subgraph(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    node_ids: &[Uuid],
    weights: &ScoreWeights,
) -> EpistemicScore {
    let hierarchy_consistency = hierarchy::hierarchy_consistency_global(storage);
    let coh = coherence::combined_coherence(storage, adjacency, node_ids);
    let cov = coverage::combined_coverage(storage, node_ids);
    let red = redundancy::path_diversity(adjacency, node_ids, 50);

    // Energy average
    let mut energy_sum = 0.0f32;
    let mut hausdorff_sum = 0.0f32;
    let mut count = 0usize;
    for nid in node_ids {
        if let Ok(Some(meta)) = storage.get_node_meta(nid) {
            energy_sum += meta.energy;
            hausdorff_sum += meta.hausdorff_local;
            count += 1;
        }
    }
    let energy_avg = if count > 0 { energy_sum / count as f32 } else { 0.0 };
    let hausdorff_avg = if count > 0 { hausdorff_sum / count as f32 } else { 1.0 };

    let composite = weights.hierarchy * hierarchy_consistency
        + weights.coherence * coh
        + weights.coverage * cov
        + weights.redundancy * red
        + weights.energy * energy_avg;

    EpistemicScore {
        hierarchy_consistency,
        coherence: coh,
        coverage: cov,
        redundancy: red,
        energy_avg,
        hausdorff_avg,
        node_count: count,
        composite,
    }
}

/// Evaluate the epistemic delta caused by a mutation.
///
/// Compares the score before and after the mutation was applied.
pub fn evaluate_mutation(
    before: &EpistemicScore,
    after: &EpistemicScore,
    weights: &ScoreWeights,
) -> EpistemicDelta {
    let h_delta = after.hierarchy_consistency - before.hierarchy_consistency;
    let c_delta = after.coherence - before.coherence;
    let cov_delta = after.coverage - before.coverage;
    let r_delta = after.redundancy - before.redundancy;
    let e_delta = after.energy_avg - before.energy_avg;

    let composite_delta = weights.hierarchy * h_delta
        + weights.coherence * c_delta
        + weights.coverage * cov_delta
        + weights.redundancy * r_delta
        + weights.energy * e_delta;

    EpistemicDelta {
        hierarchy_delta: h_delta,
        coherence_delta: c_delta,
        coverage_delta: cov_delta,
        redundancy_delta: r_delta,
        energy_delta: e_delta,
        composite_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weights_default_sum() {
        let w = ScoreWeights::default();
        let sum = w.hierarchy + w.coherence + w.coverage + w.redundancy + w.energy;
        assert!((sum - 1.0).abs() < 0.01, "weights should sum to ~1.0, got {sum}");
    }

    #[test]
    fn test_delta_improvement() {
        let delta = EpistemicDelta {
            hierarchy_delta: 0.05,
            coherence_delta: 0.02,
            coverage_delta: 0.0,
            redundancy_delta: 0.01,
            energy_delta: 0.0,
            composite_delta: 0.03,
        };
        assert!(delta.is_improvement(0.02));
        assert!(!delta.is_improvement(0.05));
    }
}
