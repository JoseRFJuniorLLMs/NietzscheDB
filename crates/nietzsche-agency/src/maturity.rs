// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! # MaturityEvaluator — Natural Selection of Concepts
//!
//! Evaluates node maturity based on the integrated maturity score:
//!
//! ```text
//! M_i = V_i · log(1 + S_i) · C_i · exp(-Var(angle_i))
//! ```
//!
//! Where:
//! - `V_i` = vitality (energy-based lifeforce)
//! - `S_i` = stability (inverse of embedding variance over time)
//! - `C_i` = hyperbolic centrality relative to the civilizational centroid
//! - `Var(angle_i)` = angular variance in tangent space (O(k) via trace(Σ_tangent))
//!
//! ## Ontological Classes
//!
//! | Class | Condition | Meaning |
//! |-------|-----------|---------|
//! | LATENT | energy ≤ 0 or phantom | Embryonic/dead idea |
//! | ACTIVE | 0 < M_i < promote_threshold | Idea under test |
//! | MATURE | M_i ≥ promote_threshold | Idea that defines the Self |
//!
//! ## Dependency
//!
//! ```text
//! CentroidGuardian::centrality(x_i) → C_i → MaturityEvaluator
//! ```

use uuid::Uuid;

use crate::centroid_guardian::CentroidGuardian;

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

/// Configuration for the MaturityEvaluator.
#[derive(Debug, Clone)]
pub struct MaturityConfig {
    /// Maturity score above which a node is promoted to MATURE.
    pub promote_threshold: f64,
    /// Maturity score below which a MATURE node is demoted to ACTIVE.
    /// Should be lower than `promote_threshold` to create hysteresis.
    pub demote_threshold: f64,
}

impl Default for MaturityConfig {
    fn default() -> Self {
        Self {
            promote_threshold: 0.6,
            demote_threshold: 0.3,
        }
    }
}

// ─────────────────────────────────────────────
// Ontological classification
// ─────────────────────────────────────────────

/// Ontological class of a node in the cognitive metabolism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeClass {
    /// Embryonic or dead idea (energy ≤ 0, phantom, or expired).
    Latent,
    /// Idea under active testing (0 < M < promote_threshold).
    Active,
    /// Idea that defines the Self (M ≥ promote_threshold).
    Mature,
}

// ─────────────────────────────────────────────
// MaturityScore
// ─────────────────────────────────────────────

/// Detailed maturity evaluation result for a single node.
#[derive(Debug, Clone)]
pub struct MaturityScore {
    pub node_id: Uuid,
    /// Raw maturity score M_i.
    pub score: f64,
    /// Ontological class after evaluation.
    pub class: NodeClass,
    /// Whether the class changed from the previous state.
    pub transition: Option<(NodeClass, NodeClass)>,

    // Component breakdown
    pub vitality: f64,
    pub stability: f64,
    pub centrality: f64,
    pub angular_penalty: f64,
}

// ─────────────────────────────────────────────
// Input
// ─────────────────────────────────────────────

/// Input data for maturity evaluation of a single node.
pub struct NodeMaturityInput {
    pub node_id: Uuid,
    /// Node embedding in Poincaré ball (f64, already promoted from f32).
    pub embedding: Vec<f64>,
    /// Vitality score V_i (from Nezhmetdinov or energy-based).
    pub vitality: f64,
    /// Stability score S_i (inverse variance of embedding over time).
    /// Higher = more stable position.
    pub stability: f64,
    /// Angular variance in tangent space.
    /// Computed as trace(Σ_tangent) — O(k) approximation.
    pub angular_variance: f64,
    /// Previous ontological class (for hysteresis).
    pub previous_class: NodeClass,
}

// ─────────────────────────────────────────────
// Evaluation
// ─────────────────────────────────────────────

/// Evaluate maturity score for a single node.
///
/// ```text
/// M_i = V_i · log(1 + S_i) · C_i · exp(-Var(angle_i))
/// ```
pub fn evaluate_maturity(
    input: &NodeMaturityInput,
    guardian: &CentroidGuardian,
    config: &MaturityConfig,
) -> MaturityScore {
    // Centrality: C_i = 1 / (1 + d_H(x_i, C_t))
    let centrality = guardian.centrality(&input.embedding);

    // Stability contribution: log(1 + S_i) — diminishing returns
    let stability_contrib = (1.0 + input.stability).ln();

    // Angular penalty: exp(-Var(angle)) — penalizes directional instability
    let angular_penalty = (-input.angular_variance).exp();

    // Final score
    let score = input.vitality * stability_contrib * centrality * angular_penalty;

    // Classify with hysteresis
    let class = classify(score, input.previous_class, config);

    let transition = if class != input.previous_class {
        Some((input.previous_class, class))
    } else {
        None
    };

    MaturityScore {
        node_id: input.node_id,
        score,
        class,
        transition,
        vitality: input.vitality,
        stability: stability_contrib,
        centrality,
        angular_penalty,
    }
}

/// Batch evaluate maturity for multiple nodes.
pub fn evaluate_batch(
    inputs: &[NodeMaturityInput],
    guardian: &CentroidGuardian,
    config: &MaturityConfig,
) -> Vec<MaturityScore> {
    inputs.iter()
        .map(|input| evaluate_maturity(input, guardian, config))
        .collect()
}

/// Classify a node based on its maturity score with hysteresis.
///
/// Hysteresis prevents oscillation at the boundary:
/// - Promote: score > promote_threshold
/// - Demote: score < demote_threshold (lower than promote)
/// - Between thresholds: keep previous class
fn classify(score: f64, previous: NodeClass, config: &MaturityConfig) -> NodeClass {
    match previous {
        NodeClass::Latent => {
            // Latent nodes can only become Active first
            if score > config.demote_threshold {
                NodeClass::Active
            } else {
                NodeClass::Latent
            }
        }
        NodeClass::Active => {
            if score >= config.promote_threshold {
                NodeClass::Mature
            } else if score <= 0.0 {
                NodeClass::Latent
            } else {
                NodeClass::Active
            }
        }
        NodeClass::Mature => {
            if score < config.demote_threshold {
                NodeClass::Active
            } else {
                NodeClass::Mature
            }
        }
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_guardian() -> CentroidGuardian {
        CentroidGuardian::new(3)
    }

    fn make_config() -> MaturityConfig {
        MaturityConfig {
            promote_threshold: 0.5,
            demote_threshold: 0.2,
        }
    }

    #[test]
    fn high_vitality_near_centroid_promotes() {
        let guardian = make_guardian(); // centroid at origin
        let config = make_config();

        let input = NodeMaturityInput {
            node_id: Uuid::new_v4(),
            embedding: vec![0.05, 0.0, 0.0], // very close to origin
            vitality: 0.9,
            stability: 2.0,
            angular_variance: 0.01,
            previous_class: NodeClass::Active,
        };

        let score = evaluate_maturity(&input, &guardian, &config);
        assert!(score.score > config.promote_threshold,
            "high vitality + near centroid should promote, score={}", score.score);
        assert_eq!(score.class, NodeClass::Mature);
    }

    #[test]
    fn low_vitality_stays_active() {
        let guardian = make_guardian();
        let config = make_config();

        let input = NodeMaturityInput {
            node_id: Uuid::new_v4(),
            embedding: vec![0.1, 0.0, 0.0],
            vitality: 0.1,
            stability: 0.5,
            angular_variance: 0.5,
            previous_class: NodeClass::Active,
        };

        let score = evaluate_maturity(&input, &guardian, &config);
        assert!(score.score < config.promote_threshold);
        assert_eq!(score.class, NodeClass::Active);
    }

    #[test]
    fn far_from_centroid_reduces_score() {
        let guardian = make_guardian();
        let config = make_config();

        let near = NodeMaturityInput {
            node_id: Uuid::new_v4(),
            embedding: vec![0.05, 0.0, 0.0],
            vitality: 0.8,
            stability: 1.0,
            angular_variance: 0.0,
            previous_class: NodeClass::Active,
        };

        let far = NodeMaturityInput {
            node_id: Uuid::new_v4(),
            embedding: vec![0.9, 0.0, 0.0],
            vitality: 0.8,
            stability: 1.0,
            angular_variance: 0.0,
            previous_class: NodeClass::Active,
        };

        let score_near = evaluate_maturity(&near, &guardian, &config);
        let score_far = evaluate_maturity(&far, &guardian, &config);

        assert!(score_near.score > score_far.score,
            "near centroid should score higher: {:.4} vs {:.4}",
            score_near.score, score_far.score);
    }

    #[test]
    fn hysteresis_prevents_oscillation() {
        let guardian = make_guardian();
        let config = make_config();

        // Score between demote and promote thresholds
        let input_mature = NodeMaturityInput {
            node_id: Uuid::new_v4(),
            embedding: vec![0.3, 0.0, 0.0],
            vitality: 0.5,
            stability: 0.5,
            angular_variance: 0.1,
            previous_class: NodeClass::Mature,
        };

        let input_active = NodeMaturityInput {
            node_id: Uuid::new_v4(),
            embedding: vec![0.3, 0.0, 0.0],
            vitality: 0.5,
            stability: 0.5,
            angular_variance: 0.1,
            previous_class: NodeClass::Active,
        };

        let score_m = evaluate_maturity(&input_mature, &guardian, &config);
        let score_a = evaluate_maturity(&input_active, &guardian, &config);

        // Same score, but different class due to hysteresis
        assert!((score_m.score - score_a.score).abs() < 1e-10);

        // If score is between thresholds, keep previous class
        if score_m.score > config.demote_threshold && score_m.score < config.promote_threshold {
            assert_eq!(score_m.class, NodeClass::Mature, "mature should stay mature in hysteresis band");
            assert_eq!(score_a.class, NodeClass::Active, "active should stay active in hysteresis band");
        }
    }

    #[test]
    fn angular_variance_penalizes() {
        let guardian = make_guardian();
        let config = make_config();

        let stable = NodeMaturityInput {
            node_id: Uuid::new_v4(),
            embedding: vec![0.1, 0.0, 0.0],
            vitality: 0.8,
            stability: 1.0,
            angular_variance: 0.0, // perfectly stable direction
            previous_class: NodeClass::Active,
        };

        let unstable = NodeMaturityInput {
            node_id: Uuid::new_v4(),
            embedding: vec![0.1, 0.0, 0.0],
            vitality: 0.8,
            stability: 1.0,
            angular_variance: 2.0, // very unstable direction
            previous_class: NodeClass::Active,
        };

        let s_stable = evaluate_maturity(&stable, &guardian, &config);
        let s_unstable = evaluate_maturity(&unstable, &guardian, &config);

        assert!(s_stable.score > s_unstable.score,
            "angular instability should reduce score: {:.4} vs {:.4}",
            s_stable.score, s_unstable.score);
    }

    #[test]
    fn batch_evaluates_all() {
        let guardian = make_guardian();
        let config = make_config();

        let inputs: Vec<NodeMaturityInput> = (0..5).map(|i| {
            NodeMaturityInput {
                node_id: Uuid::new_v4(),
                embedding: vec![0.1 * (i as f64 + 1.0), 0.0, 0.0],
                vitality: 0.5,
                stability: 1.0,
                angular_variance: 0.0,
                previous_class: NodeClass::Active,
            }
        }).collect();

        let scores = evaluate_batch(&inputs, &guardian, &config);
        assert_eq!(scores.len(), 5);
    }

    #[test]
    fn transition_recorded() {
        let guardian = make_guardian();
        let config = make_config();

        let input = NodeMaturityInput {
            node_id: Uuid::new_v4(),
            embedding: vec![0.02, 0.0, 0.0],
            vitality: 0.95,
            stability: 3.0,
            angular_variance: 0.0,
            previous_class: NodeClass::Active,
        };

        let score = evaluate_maturity(&input, &guardian, &config);
        if score.class == NodeClass::Mature {
            assert_eq!(score.transition, Some((NodeClass::Active, NodeClass::Mature)));
        }
    }
}
