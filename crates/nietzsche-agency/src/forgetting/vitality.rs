//! Vitality Sigmoid Function — The mathematical heart of the Forgetting Engine.
//!
//! Implements V(n) = σ(w₁e + w₂H − w₃ξ + w₄π + w₅κ − w₆τ) where:
//! - e: Node energy ∈ [0,1]
//! - H: Hausdorff local (structural complexity)
//! - ξ: Entropy delta (contribution to order)
//! - π: Elite proximity (closeness to high-value nodes in Poincaré ball)
//! - κ: Causal centrality (Minkowski timelike edge count)
//! - τ: Emotional toxicity ∈ [0,1]
//!
//! Named after Rashid Nezhmetdinov's positional sacrifice — the system
//! sacrifices data persistence for topological clarity.

use serde::{Deserialize, Serialize};

/// The six weights of the Vitality function.
///
/// These control how much each factor contributes to a node's survival score.
/// Higher vitality = more likely to survive the forgetting cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitalityWeights {
    /// w₁: Weight for node energy (positive contributor)
    pub w1_energy: f32,
    /// w₂: Weight for Hausdorff local complexity (positive contributor)
    pub w2_hausdorff: f32,
    /// w₃: Weight for entropy delta (NEGATIVE contributor — high entropy = death)
    pub w3_entropy: f32,
    /// w₄: Weight for elite proximity (positive contributor)
    pub w4_elite_prox: f32,
    /// w₅: Weight for causal centrality — Minkowski edges (positive, strongest shield)
    pub w5_causal: f32,
    /// w₆: Weight for emotional toxicity (NEGATIVE contributor)
    pub w6_toxicity: f32,
}

impl Default for VitalityWeights {
    fn default() -> Self {
        Self {
            w1_energy: 1.0,
            w2_hausdorff: 0.8,
            w3_entropy: 1.2,
            w4_elite_prox: 1.5,
            w5_causal: 2.0,   // Causal edges are the strongest shield
            w6_toxicity: 1.0,
        }
    }
}

/// Input features for evaluating a single node's vitality.
#[derive(Debug, Clone)]
pub struct VitalityInput {
    /// e(n): Node energy ∈ [0, 1]
    pub energy: f32,
    /// H(n): Hausdorff local dimension (structural complexity)
    pub hausdorff_local: f32,
    /// ξ(n): Entropy delta — how much this node contributes to local entropy
    pub entropy_delta: f32,
    /// π(n): Elite proximity — Poincaré distance to nearest elite centroid
    /// Lower = closer to elites = higher survival chance
    pub elite_proximity: f32,
    /// κ(n): Causal centrality — count of Minkowski timelike/lightlike edges
    pub causal_centrality: f32,
    /// τ(n): Emotional toxicity — negative valence * arousal
    pub toxicity: f32,
}

impl VitalityInput {
    /// Create from raw node data.
    ///
    /// `elite_proximity` is inverted: we pass the raw distance and convert
    /// it to a "closeness" score (1.0 - distance.min(1.0)) so that nodes
    /// NEAR elites get a bonus.
    pub fn new(
        energy: f32,
        hausdorff_local: f32,
        entropy_delta: f32,
        elite_distance: f32,
        causal_edge_count: usize,
        toxicity: f32,
    ) -> Self {
        Self {
            energy: energy.clamp(0.0, 1.0),
            hausdorff_local: hausdorff_local.max(0.0),
            entropy_delta,
            // Invert: close to elite = high proximity score
            elite_proximity: (1.0 - elite_distance.min(1.0)).clamp(0.0, 1.0),
            causal_centrality: causal_edge_count as f32,
            toxicity: toxicity.clamp(0.0, 1.0),
        }
    }
}

/// Classic sigmoid: σ(z) = 1 / (1 + e^(-z))
#[inline]
pub fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Calculate the Vitality score V(n) for a node.
///
/// V(n) = σ(w₁e + w₂H − w₃ξ + w₄π + w₅κ − w₆τ)
///
/// Returns a value in (0, 1) where:
/// - V(n) close to 0 = candidate for forgetting
/// - V(n) close to 1 = sacred, protected
pub fn nezhmetdinov_vitality(weights: &VitalityWeights, input: &VitalityInput) -> f32 {
    let z = weights.w1_energy * input.energy
          + weights.w2_hausdorff * input.hausdorff_local
          - weights.w3_entropy * input.entropy_delta
          + weights.w4_elite_prox * input.elite_proximity
          + weights.w5_causal * input.causal_centrality
          - weights.w6_toxicity * input.toxicity;

    sigmoid(z)
}

/// Batch-evaluate vitality for multiple nodes.
///
/// Returns (vitality_scores, mean, variance) — the variance is used
/// for the Var(V) health metric.
pub fn nezhmetdinov_vitality_batch(
    weights: &VitalityWeights,
    inputs: &[VitalityInput],
) -> (Vec<f32>, f32, f32) {
    if inputs.is_empty() {
        return (vec![], 0.0, 0.0);
    }

    let scores: Vec<f32> = inputs
        .iter()
        .map(|input| nezhmetdinov_vitality(weights, input))
        .collect();

    let n = scores.len() as f32;
    let sum: f32 = scores.iter().sum();
    let mean = sum / n;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;

    (scores, mean, variance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn high_energy_high_vitality() {
        let w = VitalityWeights::default();
        let input = VitalityInput {
            energy: 0.9,
            hausdorff_local: 0.8,
            entropy_delta: 0.1,
            elite_proximity: 0.9,
            causal_centrality: 5.0,
            toxicity: 0.0,
        };
        let v = nezhmetdinov_vitality(&w, &input);
        assert!(v > 0.95, "high energy elite node should have very high vitality: {}", v);
    }

    #[test]
    fn low_energy_toxic_low_vitality() {
        let w = VitalityWeights::default();
        let input = VitalityInput {
            energy: 0.05,
            hausdorff_local: 0.1,
            entropy_delta: 0.9,
            elite_proximity: 0.0,
            causal_centrality: 0.0,
            toxicity: 0.9,
        };
        let v = nezhmetdinov_vitality(&w, &input);
        assert!(v < 0.2, "low energy toxic node should have low vitality: {}", v);
    }

    #[test]
    fn causal_centrality_protects() {
        let w = VitalityWeights::default();
        // Same weak node but with causal edges
        let weak_no_causal = VitalityInput {
            energy: 0.1, hausdorff_local: 0.1, entropy_delta: 0.5,
            elite_proximity: 0.0, causal_centrality: 0.0, toxicity: 0.3,
        };
        let weak_with_causal = VitalityInput {
            energy: 0.1, hausdorff_local: 0.1, entropy_delta: 0.5,
            elite_proximity: 0.0, causal_centrality: 5.0, toxicity: 0.3,
        };
        let v_no = nezhmetdinov_vitality(&w, &weak_no_causal);
        let v_yes = nezhmetdinov_vitality(&w, &weak_with_causal);
        assert!(v_yes > v_no + 0.3, "causal edges should dramatically increase vitality");
    }

    #[test]
    fn batch_statistics() {
        let w = VitalityWeights::default();
        let inputs: Vec<VitalityInput> = (0..100).map(|i| VitalityInput {
            energy: i as f32 / 100.0,
            hausdorff_local: 0.5,
            entropy_delta: 0.3,
            elite_proximity: 0.2,
            causal_centrality: 0.0, // No causal links → wider vitality spread
            toxicity: 0.1,
        }).collect();

        let (scores, mean, variance) = nezhmetdinov_vitality_batch(&w, &inputs);
        assert_eq!(scores.len(), 100);
        assert!(mean > 0.1 && mean < 0.95, "mean out of range: {}", mean);
        assert!(variance > 0.0);
    }

    #[test]
    fn vitality_input_clamps() {
        let input = VitalityInput::new(1.5, 0.5, 0.3, -0.5, 3, 2.0);
        assert_eq!(input.energy, 1.0);
        assert_eq!(input.toxicity, 1.0);
        assert_eq!(input.elite_proximity, 1.0); // -0.5 distance -> clamped
    }
}
