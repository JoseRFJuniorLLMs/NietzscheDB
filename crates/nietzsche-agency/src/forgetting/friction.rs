//! External Friction Score — Epistemological Closure Prevention.
//!
//! From the Nezhmetdinov spec (CAMADA 3):
//! Validation against frozen external models (LLMs) to ensure the
//! forgetting process doesn't create an echo chamber where the system
//! only reinforces its own biases.
//!
//! F(h) measures how well newly generated hypotheses align with
//! external knowledge sources. Too much alignment = echo chamber.
//! Too little = disconnection from reality.

use serde::{Deserialize, Serialize};

/// External friction score for a single hypothesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrictionScore {
    /// The hypothesis being evaluated (void ID or generated node ID).
    pub hypothesis_id: String,
    /// Alignment score with external model [0, 1].
    /// 1.0 = perfect alignment, 0.0 = complete disagreement.
    pub alignment: f32,
    /// Novelty score [0, 1].
    /// 1.0 = completely novel (not in external model), 0.0 = already known.
    pub novelty: f32,
    /// Combined friction score.
    /// Healthy range: 0.3 - 0.7 (some alignment, some novelty).
    pub friction: f32,
}

impl FrictionScore {
    /// Compute friction from alignment and novelty.
    ///
    /// Friction = alignment × (1 - |alignment - 0.5| × 2)
    /// This peaks at alignment = 0.5 (balanced between agreement and novelty).
    pub fn compute(hypothesis_id: impl Into<String>, alignment: f32, novelty: f32) -> Self {
        let a = alignment.clamp(0.0, 1.0);
        let n = novelty.clamp(0.0, 1.0);

        // Optimal friction: balanced between alignment and novelty
        // Penalize extremes (pure echo = bad, pure divergence = bad)
        let balance_penalty = (a - 0.5).abs() * 2.0; // 0 at 0.5, 1 at extremes
        let friction = (1.0 - balance_penalty) * (a + n) / 2.0;

        Self {
            hypothesis_id: hypothesis_id.into(),
            alignment: a,
            novelty: n,
            friction: friction.clamp(0.0, 1.0),
        }
    }
}

/// Aggregate friction metrics for a cycle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CycleFriction {
    pub cycle: u64,
    pub scores: Vec<FrictionScore>,
    pub mean_friction: f32,
    pub mean_alignment: f32,
    pub mean_novelty: f32,
    /// True if the system is in echo chamber territory (alignment > 0.9).
    pub echo_chamber_risk: bool,
    /// True if the system is disconnected (alignment < 0.1).
    pub disconnection_risk: bool,
}

/// External Friction Calculator.
///
/// For MVP, this uses placeholder scores. In production, it would
/// call frozen LLM APIs to validate hypotheses.
#[derive(Debug, Clone)]
pub struct FrictionCalculator {
    /// History of cycle-level friction metrics.
    pub history: Vec<CycleFriction>,
    /// Echo chamber threshold (mean alignment above this = risk).
    pub echo_threshold: f32,
    /// Disconnection threshold (mean alignment below this = risk).
    pub disconnect_threshold: f32,
}

impl Default for FrictionCalculator {
    fn default() -> Self {
        Self {
            history: Vec::new(),
            echo_threshold: 0.85,
            disconnect_threshold: 0.15,
        }
    }
}

impl FrictionCalculator {
    /// Record friction scores for a cycle.
    pub fn record_cycle(&mut self, cycle: u64, scores: Vec<FrictionScore>) -> CycleFriction {
        let n = scores.len().max(1) as f32;
        let mean_friction = scores.iter().map(|s| s.friction).sum::<f32>() / n;
        let mean_alignment = scores.iter().map(|s| s.alignment).sum::<f32>() / n;
        let mean_novelty = scores.iter().map(|s| s.novelty).sum::<f32>() / n;

        let echo_chamber_risk = mean_alignment > self.echo_threshold;
        let disconnection_risk = mean_alignment < self.disconnect_threshold;

        let result = CycleFriction {
            cycle,
            scores,
            mean_friction,
            mean_alignment,
            mean_novelty,
            echo_chamber_risk,
            disconnection_risk,
        };

        self.history.push(result.clone());
        result
    }

    /// Generate placeholder friction scores for MVP (no external API).
    ///
    /// Uses a simple heuristic based on void plausibility.
    pub fn placeholder_scores(
        hypothesis_ids: &[String],
        plausibilities: &[f32],
    ) -> Vec<FrictionScore> {
        hypothesis_ids.iter().zip(plausibilities.iter())
            .map(|(id, &plaus)| {
                // Heuristic: plausibility correlates with alignment
                let alignment = plaus * 0.7 + 0.15; // Range ~[0.15, 0.85]
                let novelty = 1.0 - plaus * 0.5;    // Higher plausibility = less novel
                FrictionScore::compute(id.clone(), alignment, novelty)
            })
            .collect()
    }

    /// Current mean friction (latest cycle).
    pub fn current_friction(&self) -> f32 {
        self.history.last().map(|c| c.mean_friction).unwrap_or(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_friction_is_highest() {
        let balanced = FrictionScore::compute("test", 0.5, 0.5);
        let echo = FrictionScore::compute("test", 0.95, 0.1);
        let divergent = FrictionScore::compute("test", 0.05, 0.9);

        assert!(balanced.friction > echo.friction,
            "balanced ({:.3}) should beat echo ({:.3})", balanced.friction, echo.friction);
        assert!(balanced.friction > divergent.friction,
            "balanced ({:.3}) should beat divergent ({:.3})", balanced.friction, divergent.friction);
    }

    #[test]
    fn echo_chamber_detection() {
        let mut calc = FrictionCalculator::default();
        let scores: Vec<FrictionScore> = (0..10)
            .map(|i| FrictionScore::compute(format!("h{}", i), 0.95, 0.05))
            .collect();
        let result = calc.record_cycle(1, scores);
        assert!(result.echo_chamber_risk);
        assert!(!result.disconnection_risk);
    }

    #[test]
    fn disconnection_detection() {
        let mut calc = FrictionCalculator::default();
        let scores: Vec<FrictionScore> = (0..10)
            .map(|i| FrictionScore::compute(format!("h{}", i), 0.05, 0.95))
            .collect();
        let result = calc.record_cycle(1, scores);
        assert!(!result.echo_chamber_risk);
        assert!(result.disconnection_risk);
    }

    #[test]
    fn placeholder_scores_in_range() {
        let ids: Vec<String> = (0..5).map(|i| format!("void_{}", i)).collect();
        let plaus = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let scores = FrictionCalculator::placeholder_scores(&ids, &plaus);

        for s in &scores {
            assert!(s.alignment >= 0.0 && s.alignment <= 1.0);
            assert!(s.novelty >= 0.0 && s.novelty <= 1.0);
            assert!(s.friction >= 0.0 && s.friction <= 1.0);
        }
    }
}
