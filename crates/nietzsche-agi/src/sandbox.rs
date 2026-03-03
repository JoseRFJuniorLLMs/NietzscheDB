//! # Sandbox — Quarantine and promotion for innovative inferences
//!
//! The sandbox is the **immune system** of the AGI layer. When the
//! [`InnovationEvaluator`](crate::innovation::InnovationEvaluator) decides
//! an inference should be sandboxed (promising but uncertain), the
//! [`SandboxEvaluator`] manages its lifecycle:
//!
//! ## Quarantine Flow
//!
//! ```text
//! ┌─────────────┐      ┌──────────────┐      ┌──────────────┐
//! │  Inference   │─────▶│  Quarantine   │─────▶│   Decision   │
//! │  (Sandbox)   │      │  N cycles     │      │              │
//! └─────────────┘      └──────────────┘      └──────┬───────┘
//!                                                    │
//!                                          ┌─────────┼─────────┐
//!                                          ▼         ▼         ▼
//!                                       Promote   Continue   Reject
//!                                     (full wt)  (observe)  (remove)
//! ```
//!
//! 1. **Quarantine**: Insert with reduced weight (0.3×) and accelerated decay (3×)
//! 2. **Observation**: Run N diffusion cycles, measuring Δλ₂ each time
//! 3. **Promotion**: If |Δλ₂| stays below threshold for N cycles → promote to full weight
//! 4. **Rejection**: If Δλ₂ drops significantly or max cycles exceeded → remove
//!
//! ## λ₂ Drift Model (Evolutionary — Option B)
//!
//! The sandbox allows **slow controlled drift** of λ₂, not rigid constancy.
//! This transforms the system from a static library into an evolving organism
//! that re-maps its own intelligence as new synthesis nodes create new
//! centers of gravity.
//!
//! ```text
//! |Δλ₂| ≤ max_drift_per_cycle  →  acceptable (evolution)
//! |Δλ₂| > max_drift_per_cycle  →  destabilizing (rejection)
//! ```

use uuid::Uuid;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the sandbox evaluator.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Factor to reduce edge weights during quarantine.
    /// quarantine_weight = original_weight × weight_reduction
    /// Default: 0.3 (30% of normal weight)
    pub weight_reduction: f32,

    /// Multiplier for relevance decay during quarantine.
    /// quarantine_decay = base_decay × decay_multiplier
    /// Default: 3.0 (3× faster decay)
    pub decay_multiplier: f64,

    /// Number of stable cycles required for promotion.
    /// Default: 3
    pub promotion_cycles: usize,

    /// Maximum |Δλ₂| per cycle considered stable.
    /// This is the core of the Evolutionary Model (Option B):
    /// allows slow drift, rejects sudden destabilization.
    /// Default: 0.05
    pub max_drift_per_cycle: f64,

    /// Maximum cycles before auto-rejection.
    /// If the node can't stabilize in this many cycles, reject it.
    /// Default: 10
    pub max_quarantine_cycles: usize,

    /// Minimum Δλ₂ to consider the insertion beneficial.
    /// If λ₂ actually increases, the insertion strengthens the graph.
    /// Default: -0.01 (allow tiny drops)
    pub min_beneficial_drift: f64,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            weight_reduction: 0.3,
            decay_multiplier: 3.0,
            promotion_cycles: 3,
            max_drift_per_cycle: 0.05,
            max_quarantine_cycles: 10,
            min_beneficial_drift: -0.01,
        }
    }
}

// ─────────────────────────────────────────────
// SandboxVerdict
// ─────────────────────────────────────────────

/// The three possible outcomes for a sandboxed inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SandboxVerdict {
    /// Keep observing — not enough data yet for a decision.
    Continue,

    /// Promote to permanent manifold: restore full weight, normal decay.
    /// The insertion has proven spectrally stable.
    Promote,

    /// Reject and remove: the insertion destabilized the graph.
    Reject,
}

impl std::fmt::Display for SandboxVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxVerdict::Continue => write!(f, "CONTINUE"),
            SandboxVerdict::Promote => write!(f, "PROMOTE"),
            SandboxVerdict::Reject => write!(f, "REJECT"),
        }
    }
}

// ─────────────────────────────────────────────
// SandboxEntry — state of a quarantined node
// ─────────────────────────────────────────────

/// State of a node in quarantine.
///
/// Tracks the spectral history and quarantine parameters for a
/// single sandboxed inference.
#[derive(Debug, Clone)]
pub struct SandboxEntry {
    /// ID of the quarantined synthesis node.
    pub node_id: Uuid,

    /// Original edge weight (before reduction).
    pub original_weight: f32,

    /// Quarantine edge weight (reduced).
    pub quarantine_weight: f32,

    /// λ₂ snapshots taken after each diffusion cycle.
    /// The first snapshot is the baseline (before insertion).
    pub lambda2_snapshots: Vec<f64>,

    /// Number of observation cycles completed.
    pub cycle_count: usize,

    /// Unix timestamp when the node entered quarantine.
    pub created_at: i64,

    /// The Φ(τ) score that sent this inference to sandbox.
    pub phi_score: f64,
}

impl SandboxEntry {
    /// Compute Δλ₂ for each cycle (difference from baseline).
    pub fn delta_lambda2(&self) -> Vec<f64> {
        if self.lambda2_snapshots.len() < 2 {
            return vec![];
        }
        let baseline = self.lambda2_snapshots[0];
        self.lambda2_snapshots[1..]
            .iter()
            .map(|&l| l - baseline)
            .collect()
    }

    /// Compute per-cycle drift (difference between consecutive snapshots).
    pub fn per_cycle_drift(&self) -> Vec<f64> {
        if self.lambda2_snapshots.len() < 2 {
            return vec![];
        }
        self.lambda2_snapshots
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect()
    }
}

// ─────────────────────────────────────────────
// PromotionReport — detailed evaluation
// ─────────────────────────────────────────────

/// Detailed report of a sandbox evaluation cycle.
#[derive(Debug, Clone)]
pub struct PromotionReport {
    /// The verdict for this evaluation.
    pub verdict: SandboxVerdict,

    /// Per-cycle Δλ₂ from baseline.
    pub delta_lambda2: Vec<f64>,

    /// Per-cycle drift (consecutive differences).
    pub per_cycle_drift: Vec<f64>,

    /// Mean absolute drift per cycle.
    pub mean_drift: f64,

    /// Maximum absolute drift in any single cycle.
    pub max_drift: f64,

    /// Number of cycles completed.
    pub cycles_completed: usize,

    /// Number of consecutive stable cycles (|Δλ₂| < threshold).
    pub consecutive_stable: usize,

    /// Reason for the verdict.
    pub reason: String,
}

impl std::fmt::Display for PromotionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} [cycles={}, mean_drift={:.6}, max_drift={:.6}, stable={}/{}]: {}",
            self.verdict,
            self.cycles_completed,
            self.mean_drift,
            self.max_drift,
            self.consecutive_stable,
            self.cycles_completed,
            self.reason,
        )
    }
}

// ─────────────────────────────────────────────
// SandboxEvaluator
// ─────────────────────────────────────────────

/// Manages the quarantine lifecycle for sandboxed inferences.
///
/// **Pure computation** — returns decisions, the caller (server)
/// performs the actual insertion/removal/weight-change operations.
pub struct SandboxEvaluator {
    config: SandboxConfig,
}

impl SandboxEvaluator {
    pub fn new(config: SandboxConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(SandboxConfig::default())
    }

    /// Create a quarantine entry for a new sandboxed inference.
    ///
    /// # Arguments
    /// - `node_id`: ID of the synthesis node
    /// - `original_weight`: the normal edge weight (before reduction)
    /// - `baseline_lambda2`: λ₂ of the graph BEFORE insertion
    /// - `phi_score`: the Φ(τ) that triggered sandboxing
    ///
    /// # Returns
    /// A [`SandboxEntry`] ready for observation cycles.
    pub fn quarantine(
        &self,
        node_id: Uuid,
        original_weight: f32,
        baseline_lambda2: f64,
        phi_score: f64,
    ) -> SandboxEntry {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        SandboxEntry {
            node_id,
            original_weight,
            quarantine_weight: original_weight * self.config.weight_reduction,
            lambda2_snapshots: vec![baseline_lambda2], // baseline
            cycle_count: 0,
            created_at: now,
            phi_score,
        }
    }

    /// Record a new observation cycle.
    ///
    /// Call this after each diffusion cycle with the current λ₂.
    pub fn record_cycle(&self, entry: &mut SandboxEntry, current_lambda2: f64) {
        entry.lambda2_snapshots.push(current_lambda2);
        entry.cycle_count += 1;
    }

    /// Evaluate the current state of a sandboxed entry.
    ///
    /// Returns a [`PromotionReport`] with the verdict:
    /// - **Promote**: N consecutive stable cycles → integrate permanently
    /// - **Reject**: destabilization detected or max cycles exceeded
    /// - **Continue**: keep observing
    pub fn evaluate(&self, entry: &SandboxEntry) -> PromotionReport {
        let deltas = entry.delta_lambda2();
        let drifts = entry.per_cycle_drift();

        let mean_drift = if drifts.is_empty() {
            0.0
        } else {
            drifts.iter().map(|d| d.abs()).sum::<f64>() / drifts.len() as f64
        };

        let max_drift = drifts
            .iter()
            .map(|d| d.abs())
            .fold(0.0_f64, f64::max);

        // Count consecutive stable cycles (from the end)
        let consecutive_stable = drifts
            .iter()
            .rev()
            .take_while(|&&d| d.abs() <= self.config.max_drift_per_cycle)
            .count();

        // ── Decision Logic ──

        // 1. Check for catastrophic destabilization
        if max_drift > self.config.max_drift_per_cycle * 3.0 {
            return PromotionReport {
                verdict: SandboxVerdict::Reject,
                delta_lambda2: deltas,
                per_cycle_drift: drifts,
                mean_drift,
                max_drift,
                cycles_completed: entry.cycle_count,
                consecutive_stable,
                reason: format!(
                    "Catastrophic destabilization: max |Δλ₂| = {:.6} > 3× threshold {:.6}",
                    max_drift,
                    self.config.max_drift_per_cycle * 3.0
                ),
            };
        }

        // 2. Check if max quarantine cycles exceeded
        if entry.cycle_count >= self.config.max_quarantine_cycles {
            return PromotionReport {
                verdict: SandboxVerdict::Reject,
                delta_lambda2: deltas,
                per_cycle_drift: drifts,
                mean_drift,
                max_drift,
                cycles_completed: entry.cycle_count,
                consecutive_stable,
                reason: format!(
                    "Max quarantine cycles exceeded: {} >= {}",
                    entry.cycle_count, self.config.max_quarantine_cycles
                ),
            };
        }

        // 3. Check for promotion: N consecutive stable cycles
        if consecutive_stable >= self.config.promotion_cycles {
            return PromotionReport {
                verdict: SandboxVerdict::Promote,
                delta_lambda2: deltas,
                per_cycle_drift: drifts,
                mean_drift,
                max_drift,
                cycles_completed: entry.cycle_count,
                consecutive_stable,
                reason: format!(
                    "Spectrally stable for {} consecutive cycles (threshold: {})",
                    consecutive_stable, self.config.promotion_cycles
                ),
            };
        }

        // 4. Not enough data yet → continue
        PromotionReport {
            verdict: SandboxVerdict::Continue,
            delta_lambda2: deltas,
            per_cycle_drift: drifts,
            mean_drift,
            max_drift,
            cycles_completed: entry.cycle_count,
            consecutive_stable,
            reason: format!(
                "Observing: {}/{} stable cycles needed",
                consecutive_stable, self.config.promotion_cycles
            ),
        }
    }

    /// Compute the promoted edge weight (restored from quarantine).
    pub fn promoted_weight(&self, entry: &SandboxEntry) -> f32 {
        entry.original_weight
    }

    /// Compute the quarantine decay rate.
    pub fn quarantine_decay_rate(&self, base_decay: f64) -> f64 {
        base_decay * self.config.decay_multiplier
    }

    pub fn config(&self) -> &SandboxConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(baseline: f64) -> SandboxEntry {
        let eval = SandboxEvaluator::with_defaults();
        eval.quarantine(Uuid::new_v4(), 0.8, baseline, 0.35)
    }

    #[test]
    fn test_quarantine_reduces_weight() {
        let eval = SandboxEvaluator::with_defaults();
        let entry = eval.quarantine(Uuid::new_v4(), 0.8, 1.0, 0.35);
        assert!((entry.quarantine_weight - 0.24).abs() < 0.01); // 0.8 × 0.3
        assert_eq!(entry.lambda2_snapshots.len(), 1); // baseline only
        assert_eq!(entry.cycle_count, 0);
    }

    #[test]
    fn test_promotion_after_stable_cycles() {
        let eval = SandboxEvaluator::with_defaults();
        let mut entry = make_entry(1.0);

        // Record 3 stable cycles (small drift)
        eval.record_cycle(&mut entry, 1.01);
        eval.record_cycle(&mut entry, 1.02);
        eval.record_cycle(&mut entry, 1.015);

        let report = eval.evaluate(&entry);
        assert_eq!(report.verdict, SandboxVerdict::Promote);
        assert_eq!(report.consecutive_stable, 3);
    }

    #[test]
    fn test_continue_when_not_enough_cycles() {
        let eval = SandboxEvaluator::with_defaults();
        let mut entry = make_entry(1.0);

        // Only 2 stable cycles (need 3)
        eval.record_cycle(&mut entry, 1.01);
        eval.record_cycle(&mut entry, 1.02);

        let report = eval.evaluate(&entry);
        assert_eq!(report.verdict, SandboxVerdict::Continue);
        assert_eq!(report.consecutive_stable, 2);
    }

    #[test]
    fn test_reject_on_catastrophic_drift() {
        let eval = SandboxEvaluator::with_defaults();
        let mut entry = make_entry(1.0);

        // Catastrophic drop: Δ = -0.5 > 3 × 0.05 = 0.15
        eval.record_cycle(&mut entry, 0.5);

        let report = eval.evaluate(&entry);
        assert_eq!(report.verdict, SandboxVerdict::Reject);
        assert!(report.reason.contains("Catastrophic"));
    }

    #[test]
    fn test_reject_on_max_cycles() {
        let config = SandboxConfig {
            max_quarantine_cycles: 3,
            ..Default::default()
        };
        let eval = SandboxEvaluator::new(config);
        let mut entry = make_entry(1.0);

        // 3 unstable cycles (drift = 0.1 > 0.05)
        eval.record_cycle(&mut entry, 1.1);
        eval.record_cycle(&mut entry, 1.2);
        eval.record_cycle(&mut entry, 1.3);

        let report = eval.evaluate(&entry);
        assert_eq!(report.verdict, SandboxVerdict::Reject);
        assert!(report.reason.contains("Max quarantine"));
    }

    #[test]
    fn test_mixed_stability() {
        let eval = SandboxEvaluator::with_defaults();
        let mut entry = make_entry(1.0);

        // Unstable → stabilizes
        eval.record_cycle(&mut entry, 1.1);  // drift 0.1 > 0.05 (unstable)
        eval.record_cycle(&mut entry, 1.12); // drift 0.02 (stable)
        eval.record_cycle(&mut entry, 1.11); // drift 0.01 (stable)
        eval.record_cycle(&mut entry, 1.115);// drift 0.005 (stable)

        let report = eval.evaluate(&entry);
        assert_eq!(report.verdict, SandboxVerdict::Promote);
        assert_eq!(report.consecutive_stable, 3); // last 3 are stable
    }

    #[test]
    fn test_delta_lambda2_computation() {
        let mut entry = make_entry(1.0);
        entry.lambda2_snapshots.push(1.05);
        entry.lambda2_snapshots.push(0.98);

        let deltas = entry.delta_lambda2();
        assert_eq!(deltas.len(), 2);
        assert!((deltas[0] - 0.05).abs() < 1e-10);
        assert!((deltas[1] - (-0.02)).abs() < 1e-10);
    }

    #[test]
    fn test_per_cycle_drift() {
        let mut entry = make_entry(1.0);
        entry.lambda2_snapshots.push(1.05);
        entry.lambda2_snapshots.push(1.02);

        let drifts = entry.per_cycle_drift();
        assert_eq!(drifts.len(), 2);
        assert!((drifts[0] - 0.05).abs() < 1e-10);  // 1.0 → 1.05
        assert!((drifts[1] - (-0.03)).abs() < 1e-10); // 1.05 → 1.02
    }

    #[test]
    fn test_quarantine_decay_rate() {
        let eval = SandboxEvaluator::with_defaults();
        let rate = eval.quarantine_decay_rate(0.01);
        assert!((rate - 0.03).abs() < 1e-10); // 0.01 × 3.0
    }

    #[test]
    fn test_promoted_weight_restored() {
        let eval = SandboxEvaluator::with_defaults();
        let entry = eval.quarantine(Uuid::new_v4(), 0.8, 1.0, 0.35);
        assert!((eval.promoted_weight(&entry) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_promotion_report_display() {
        let report = PromotionReport {
            verdict: SandboxVerdict::Promote,
            delta_lambda2: vec![0.01, 0.02, 0.015],
            per_cycle_drift: vec![0.01, 0.01, -0.005],
            mean_drift: 0.0083,
            max_drift: 0.01,
            cycles_completed: 3,
            consecutive_stable: 3,
            reason: "Spectrally stable for 3 cycles".to_string(),
        };
        let s = format!("{report}");
        assert!(s.contains("PROMOTE"));
        assert!(s.contains("stable=3/3"));
    }
}
