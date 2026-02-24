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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  TgcMonitor — Production-Grade TGC with δH and δE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// TGC(t) = intensity * mean_quality * (1 + α*δH) * (1 + β*δE)
//
// Where:
//   intensity   = nodes_created / active_nodes
//   mean_quality = mean vitality of generated nodes
//   δH = structural_entropy(t) - structural_entropy(t-1)
//   δE = global_efficiency(t) - global_efficiency(t-1)
//   α = 2.0  (entropy amplifier)
//   β = 3.0  (efficiency amplifier)
//
// This formula rewards generation that IMPROVES topological health:
//   - δH > 0 → new nodes increased structural diversity    → TGC boosted
//   - δE > 0 → new nodes improved information routing      → TGC boosted
//   - δH < 0 or δE < 0 → topology degraded                → TGC penalized
//
// Phase rupture detection: TGC > 1.5 indicates regime change.

/// Production TGC monitor with structural entropy and global efficiency deltas.
#[derive(Debug, Clone)]
pub struct TgcMonitor {
    /// Previous cycle's structural entropy.
    pub prev_hs: f32,
    /// Previous cycle's global efficiency.
    pub prev_eg: f32,
    /// EMA-smoothed TGC.
    pub ema_tgc: f32,
    /// Entropy amplifier (default 2.0).
    pub alpha: f32,
    /// Efficiency amplifier (default 3.0).
    pub beta: f32,
    /// EMA decay for zero-intensity cycles (default 0.8).
    pub decay: f32,
    /// History of raw TGC values per cycle.
    pub history: Vec<f32>,
}

impl Default for TgcMonitor {
    fn default() -> Self {
        Self {
            prev_hs: 0.0,
            prev_eg: 0.0,
            ema_tgc: 0.0,
            alpha: 2.0,
            beta: 3.0,
            decay: 0.8,
            history: Vec::new(),
        }
    }
}

impl TgcMonitor {
    pub fn new(alpha: f32, beta: f32) -> Self {
        Self {
            alpha,
            beta,
            ..Self::default()
        }
    }

    /// Compute TGC for the current cycle.
    ///
    /// # Arguments
    /// - `nodes_created`: Nodes generated from voids this cycle
    /// - `active_nodes`: Total active nodes after generation
    /// - `mean_quality`: Mean vitality of generated nodes
    /// - `current_hs`: Structural entropy H_s of the graph NOW
    /// - `current_eg`: Global efficiency E_g of the graph NOW
    ///
    /// # Returns
    /// Raw TGC value (not EMA-smoothed). Use `ema()` for smoothed.
    pub fn compute(
        &mut self,
        nodes_created: usize,
        active_nodes: usize,
        mean_quality: f32,
        current_hs: f32,
        current_eg: f32,
    ) -> f32 {
        let intensity = if active_nodes > 0 {
            nodes_created as f32 / active_nodes as f32
        } else {
            0.0
        };

        if intensity == 0.0 {
            self.prev_hs = current_hs;
            self.prev_eg = current_eg;
            self.ema_tgc *= self.decay;
            self.history.push(0.0);
            return 0.0;
        }

        let delta_h = current_hs - self.prev_hs;
        let delta_e = current_eg - self.prev_eg;

        let mut tgc = intensity
            * mean_quality.clamp(0.0, 1.0)
            * (1.0 + self.alpha * delta_h)
            * (1.0 + self.beta * delta_e);

        tgc = tgc.max(0.0);

        // Update state
        self.prev_hs = current_hs;
        self.prev_eg = current_eg;
        self.ema_tgc = 0.2 * tgc + 0.8 * self.ema_tgc;
        self.history.push(tgc);

        tgc
    }

    /// EMA-smoothed TGC.
    pub fn ema(&self) -> f32 {
        self.ema_tgc
    }

    /// Check for phase rupture (TGC > threshold).
    pub fn is_phase_rupture(&self, threshold: f32) -> bool {
        self.history.last().map_or(false, |&v| v > threshold)
    }

    /// Mean TGC over last N cycles.
    pub fn mean_last(&self, n: usize) -> f32 {
        if self.history.is_empty() { return 0.0; }
        let start = self.history.len().saturating_sub(n);
        let slice = &self.history[start..];
        slice.iter().sum::<f32>() / slice.len().max(1) as f32
    }
}

#[cfg(test)]
mod monitor_tests {
    use super::*;

    #[test]
    fn monitor_zero_creation() {
        let mut m = TgcMonitor::default();
        let tgc = m.compute(0, 1000, 0.5, 1.0, 0.5);
        assert_eq!(tgc, 0.0);
    }

    #[test]
    fn monitor_basic_creation() {
        let mut m = TgcMonitor::default();
        // First cycle: set baseline
        m.compute(0, 1000, 0.5, 1.0, 0.5);
        // Second cycle: create 100 nodes, entropy increased, efficiency increased
        let tgc = m.compute(100, 1000, 0.8, 1.2, 0.6);
        // intensity = 0.1, quality = 0.8, (1+2*0.2)=1.4, (1+3*0.1)=1.3
        // tgc = 0.1 * 0.8 * 1.4 * 1.3 = 0.1456
        assert!(tgc > 0.1, "should produce positive TGC, got {}", tgc);
    }

    #[test]
    fn monitor_degradation_penalizes() {
        let mut m = TgcMonitor::default();
        m.prev_hs = 1.5;
        m.prev_eg = 0.8;
        // Entropy dropped, efficiency dropped → delta_h < 0, delta_e < 0
        let tgc = m.compute(50, 1000, 0.7, 1.0, 0.5);
        // (1+2*(-0.5))=0.0, (1+3*(-0.3))=0.1 → tgc = intensity * 0.7 * 0.0 * 0.1 = 0
        assert!(tgc < 0.01, "degradation should heavily penalize TGC, got {}", tgc);
    }

    #[test]
    fn monitor_ema_smoothing() {
        let mut m = TgcMonitor::default();
        m.compute(100, 1000, 0.8, 1.0, 0.5);
        let e1 = m.ema();
        m.compute(0, 1000, 0.0, 1.0, 0.5);
        let e2 = m.ema();
        assert!(e2 < e1, "EMA should decay on zero-creation cycle");
        assert!(e2 > 0.0, "EMA should not hit zero immediately");
    }

    #[test]
    fn monitor_phase_rupture() {
        let mut m = TgcMonitor::default();
        m.prev_hs = 0.1;
        m.prev_eg = 0.1;
        // Massive creation + huge entropy/efficiency gain
        let tgc = m.compute(500, 1000, 0.9, 2.0, 0.9);
        // intensity=0.5, quality=0.9, (1+2*1.9)=4.8, (1+3*0.8)=3.4
        // tgc = 0.5*0.9*4.8*3.4 = 7.34
        assert!(m.is_phase_rupture(1.5), "massive gains should trigger phase rupture, tgc={}", tgc);
    }
}
