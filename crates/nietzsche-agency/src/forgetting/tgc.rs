//! TGC — Topological Generative Capacity.
//!
//! The "gold metric" of the Forgetting Engine (CAMADA 4):
//!
//! TGC(t) = (G_t / V_t) × Quality
//!
//! Where:
//! - G_t = number of nodes generated from voids in cycle t
//! - V_t = number of voids available in cycle t
//! - Quality = mean plausibility × external friction score
//!
//! ## Sub-metrics
//!
//! - **Intrinsic TGC**: Structural plausibility of the voids themselves
//! - **Decoder TGC**: Quality of neural decoding (future: VQ-VAE)

use serde::{Deserialize, Serialize};

/// Snapshot of TGC metrics for a single cycle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TgcSnapshot {
    /// Cycle number.
    pub cycle: u64,
    /// Number of voids available at start of cycle.
    pub voids_available: usize,
    /// Number of nodes generated from voids this cycle.
    pub nodes_generated: usize,
    /// Mean structural plausibility of available voids.
    pub mean_plausibility: f32,
    /// External friction score (validation against frozen models).
    /// 1.0 = perfect alignment, 0.0 = complete disagreement.
    /// Default 0.5 until external validation is connected.
    pub external_friction: f32,
    /// Intrinsic TGC: plausibility-weighted yield ratio.
    pub tgc_intrinsic: f32,
    /// Decoder TGC: quality of neural generation (future).
    pub tgc_decoder: f32,
    /// Combined TGC score.
    pub tgc_combined: f32,
}

/// TGC Calculator — computes the Topological Generative Capacity.
#[derive(Debug, Clone)]
pub struct TgcCalculator {
    /// History of TGC snapshots for trend analysis.
    pub history: Vec<TgcSnapshot>,
    /// Running EMA (exponential moving average) of TGC.
    pub ema_tgc: f32,
    /// EMA smoothing factor (0 < α < 1, default 0.1).
    pub ema_alpha: f32,
}

impl Default for TgcCalculator {
    fn default() -> Self {
        Self {
            history: Vec::new(),
            ema_tgc: 0.0,
            ema_alpha: 0.1,
        }
    }
}

impl TgcCalculator {
    pub fn new(ema_alpha: f32) -> Self {
        Self {
            ema_alpha: ema_alpha.clamp(0.01, 0.99),
            ..Self::default()
        }
    }

    /// Compute TGC for the current cycle.
    ///
    /// # Arguments
    /// - `cycle`: Current cycle number
    /// - `voids_available`: Number of unconsumed voids
    /// - `nodes_generated`: Nodes created from voids this cycle
    /// - `mean_plausibility`: Mean plausibility of available voids
    /// - `external_friction`: External validation score [0, 1]
    pub fn compute(
        &mut self,
        cycle: u64,
        voids_available: usize,
        nodes_generated: usize,
        mean_plausibility: f32,
        external_friction: f32,
    ) -> TgcSnapshot {
        // Yield ratio: what fraction of voids produced new nodes?
        let yield_ratio = if voids_available > 0 {
            (nodes_generated as f32 / voids_available as f32).min(1.0)
        } else {
            0.0
        };

        // Intrinsic TGC: structural plausibility × yield
        let tgc_intrinsic = yield_ratio * mean_plausibility;

        // Decoder TGC: placeholder for VQ-VAE quality (future)
        // For MVP, we use a heuristic based on generation success rate
        let tgc_decoder = if nodes_generated > 0 {
            mean_plausibility * external_friction
        } else {
            0.0
        };

        // Combined TGC = (G_t / V_t) × Quality
        let quality = mean_plausibility * external_friction.max(0.01);
        let tgc_combined = yield_ratio * quality;

        // Update EMA
        self.ema_tgc = self.ema_alpha * tgc_combined + (1.0 - self.ema_alpha) * self.ema_tgc;

        let snapshot = TgcSnapshot {
            cycle,
            voids_available,
            nodes_generated,
            mean_plausibility,
            external_friction,
            tgc_intrinsic,
            tgc_decoder,
            tgc_combined,
        };

        self.history.push(snapshot.clone());
        snapshot
    }

    /// Get the current EMA of TGC.
    pub fn current_ema(&self) -> f32 {
        self.ema_tgc
    }

    /// Check if TGC is trending down (potential stagnation).
    /// Returns true if the last N snapshots show a declining trend.
    pub fn is_declining(&self, window: usize) -> bool {
        if self.history.len() < window || window < 2 {
            return false;
        }

        let recent = &self.history[self.history.len() - window..];
        let first_half: f32 = recent[..window / 2].iter().map(|s| s.tgc_combined).sum::<f32>()
            / (window / 2).max(1) as f32;
        let second_half: f32 = recent[window / 2..].iter().map(|s| s.tgc_combined).sum::<f32>()
            / (window - window / 2).max(1) as f32;

        second_half < first_half * 0.9 // 10% decline threshold
    }

    /// Compute TGC variance over the last N cycles (for anti-gaming detection).
    pub fn tgc_variance(&self, window: usize) -> f32 {
        let n = self.history.len().min(window);
        if n < 2 {
            return 0.0;
        }

        let recent = &self.history[self.history.len() - n..];
        let mean = recent.iter().map(|s| s.tgc_combined).sum::<f32>() / n as f32;
        recent.iter()
            .map(|s| (s.tgc_combined - mean).powi(2))
            .sum::<f32>() / n as f32
    }

    /// Latest snapshot.
    pub fn latest(&self) -> Option<&TgcSnapshot> {
        self.history.last()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tgc_zero_voids() {
        let mut calc = TgcCalculator::default();
        let snap = calc.compute(1, 0, 0, 0.0, 0.5);
        assert_eq!(snap.tgc_combined, 0.0);
        assert_eq!(snap.tgc_intrinsic, 0.0);
    }

    #[test]
    fn tgc_perfect_generation() {
        let mut calc = TgcCalculator::default();
        // All voids generated nodes, high plausibility, good external friction
        let snap = calc.compute(1, 10, 10, 0.9, 0.8);
        assert!(snap.tgc_combined > 0.5, "perfect generation should have high TGC: {}", snap.tgc_combined);
        assert!(snap.tgc_intrinsic > 0.5);
    }

    #[test]
    fn tgc_partial_generation() {
        let mut calc = TgcCalculator::default();
        let snap = calc.compute(1, 100, 10, 0.5, 0.5);
        // yield = 0.1, quality = 0.25, tgc = 0.025
        assert!(snap.tgc_combined > 0.0);
        assert!(snap.tgc_combined < 0.1);
    }

    #[test]
    fn ema_smoothing() {
        let mut calc = TgcCalculator::new(0.5);
        calc.compute(1, 10, 10, 0.8, 0.8);
        let first_ema = calc.current_ema();
        calc.compute(2, 10, 0, 0.0, 0.0);
        let second_ema = calc.current_ema();
        // EMA should have dropped but not to zero
        assert!(second_ema < first_ema);
        assert!(second_ema > 0.0);
    }

    #[test]
    fn declining_detection() {
        let mut calc = TgcCalculator::default();
        // 10 high TGC cycles
        for i in 0..10 {
            calc.compute(i, 10, 8, 0.8, 0.8);
        }
        assert!(!calc.is_declining(8));

        // 10 low TGC cycles
        for i in 10..20 {
            calc.compute(i, 10, 1, 0.2, 0.3);
        }
        // Use window=20 to span the high→low transition
        assert!(calc.is_declining(20));
    }

    #[test]
    fn tgc_variance_computation() {
        let mut calc = TgcCalculator::default();
        // Constant TGC → low variance
        for i in 0..10 {
            calc.compute(i, 10, 5, 0.5, 0.5);
        }
        let var = calc.tgc_variance(10);
        assert!(var < 0.01, "constant TGC should have near-zero variance: {}", var);
    }
}
