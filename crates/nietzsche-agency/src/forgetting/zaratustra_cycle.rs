//! Zaratustra Cycle — The Master Orchestrator.
//!
//! Ties all four Camadas together:
//! 1. JULGAMENTO LOCAL: Evaluate vitality → Triple Condition → Ricci Shield
//! 2. REGISTRO HISTÓRICO: Record deletions in Merkle-based ledger
//! 3. METABOLISMO GENERATIVO: Track voids for future generation
//! 4. SAÚDE GLOBAL: Update TGC, Var(V), Elite Drift, Anti-Gaming
//!
//! One complete Zaratustra cycle = one tick of the forgetting clock.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::anti_gaming::AntiGamingMonitor;
use super::bounds::NezhmetdinovConfig;
use super::elite_drift::{EliteCentroid, EliteDriftTracker};
use super::judgment::{ForgetteringJudgment, MikhailThallReport, Verdict};
use super::ledger::{DeletionLedger, DeletionReceipt, StructuralHash};
use super::stability::{CollapseAlert, StabilityConfig, StabilityMonitor};
use super::telemetry::CycleTelemetry;
use super::tgc::TgcCalculator;
use super::vitality::{nezhmetdinov_vitality, VitalityInput, VitalityWeights};
use super::void_tracker::{VoidCoordinate, VoidTracker};

/// Complete state of the Zaratustra forgetting cycle.
#[derive(Debug)]
pub struct ZaratustraCycle {
    /// Current cycle number.
    pub cycle: u64,
    /// Vitality weights.
    pub weights: VitalityWeights,
    /// Configuration with hard bounds.
    pub config: NezhmetdinovConfig,
    /// Deletion ledger.
    pub ledger: DeletionLedger,
    /// Void tracker.
    pub void_tracker: VoidTracker,
    /// TGC calculator.
    pub tgc: TgcCalculator,
    /// Elite drift tracker.
    pub elite_drift: EliteDriftTracker,
    /// Anti-gaming monitor.
    pub anti_gaming: AntiGamingMonitor,
    /// Stability monitor.
    pub stability: StabilityMonitor,
}

impl ZaratustraCycle {
    /// Create a new Zaratustra cycle orchestrator.
    pub fn new(config: NezhmetdinovConfig) -> Self {
        Self {
            cycle: 0,
            weights: VitalityWeights::default(),
            config,
            ledger: DeletionLedger::new(),
            void_tracker: VoidTracker::new(100), // Voids expire after 100 cycles
            tgc: TgcCalculator::default(),
            elite_drift: EliteDriftTracker::default(),
            anti_gaming: AntiGamingMonitor::default(),
            stability: StabilityMonitor::new(StabilityConfig::default()),
        }
    }

    /// Execute one complete Zaratustra cycle on a set of judgments.
    ///
    /// This is called AFTER the NezhmetdinovDaemon has evaluated all nodes.
    /// It processes the judgments through all four Camadas.
    pub fn execute_cycle(
        &mut self,
        judgments: &[ForgetteringJudgment],
        total_nodes_before: usize,
    ) -> ZaratustraCycleReport {
        self.cycle += 1;
        let cycle = self.cycle;

        // ── CAMADA 1: Collect condemned nodes ────────────────────────
        let thall_report = MikhailThallReport::from_judgments(cycle, judgments, 0);
        let condemned: Vec<&ForgetteringJudgment> = judgments.iter()
            .filter(|j| j.verdict.is_death_sentence())
            .collect();

        // Check bounds: max deletion rate
        let mut actual_deletions = condemned.len();
        if self.config.would_exceed_deletion_rate(total_nodes_before, actual_deletions) {
            let max_allowed = (total_nodes_before as f32 * self.config.bounds.max_deletion_rate) as usize;
            actual_deletions = max_allowed;
        }

        // Check bounds: min universe size
        if self.config.would_violate_min_universe(total_nodes_before, actual_deletions) {
            actual_deletions = total_nodes_before.saturating_sub(self.config.bounds.min_universe_size);
        }

        let condemned_to_delete = &condemned[..actual_deletions.min(condemned.len())];

        // ── CAMADA 2: Record in ledger ───────────────────────────────
        for j in condemned_to_delete {
            let receipt = DeletionReceipt {
                node_id: j.node_id,
                cycle,
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
                vitality_at_death: j.vitality_score,
                energy_at_death: j.energy,
                verdict: j.verdict.label().into(),
                structural_hash: StructuralHash::from_node_topology(
                    &j.node_id,
                    "unknown",
                    0.0,
                    j.causal_centrality,
                    &[],
                ),
                poincare_coords: vec![], // Would come from storage
                merkle_index: 0,
            };
            self.ledger.record_deletion(receipt);
        }

        let merkle_root = self.ledger.finalize_cycle(cycle);

        // ── CAMADA 3: Track voids ────────────────────────────────────
        for j in condemned_to_delete {
            let void = VoidCoordinate::from_deletion(
                j.node_id,
                vec![], // Would come from storage
                cycle,
                vec![], // Would come from adjacency
            );
            self.void_tracker.record_void(void);
        }
        self.void_tracker.expire_old_voids(cycle);

        // ── CAMADA 4: Health metrics ─────────────────────────────────
        let total_after = total_nodes_before.saturating_sub(actual_deletions);

        // TGC
        let tgc_snap = self.tgc.compute(
            cycle,
            self.void_tracker.available_count(),
            0, // No generation in pure forgetting mode
            self.void_tracker.mean_plausibility(),
            0.5, // Placeholder external friction
        );

        // Elite Drift
        let elite_nodes: Vec<(f32, Vec<f32>)> = judgments.iter()
            .filter(|j| j.verdict == Verdict::Sacred)
            .map(|j| (j.vitality_score, vec![]))
            .collect();
        let centroid = self.elite_drift.compute_elite_centroid(cycle, &elite_nodes);
        let drift = self.elite_drift.record_centroid(centroid);

        // Anti-Gaming
        let gaming_report = self.anti_gaming.analyze(
            cycle,
            tgc_snap.tgc_combined,
            actual_deletions,
            total_nodes_before,
            0,
            self.config.vitality_threshold,
            thall_report.mean_vitality,
            thall_report.mean_vitality, // Simplified: before ≈ after for daemon tick
        );

        // Stability
        let collapse_alerts = self.stability.check(
            cycle,
            total_after,
            actual_deletions,
            thall_report.mean_vitality,
            thall_report.vitality_variance,
        );

        // Build telemetry
        let telemetry = CycleTelemetry {
            cycle,
            total_nodes: total_after,
            sacrificed_this_cycle: actual_deletions,
            condemned_count: thall_report.condemned_count,
            toxic_count: thall_report.toxic_count,
            ricci_shielded_count: thall_report.ricci_shielded_count,
            sacred_count: thall_report.sacred_count,
            dormant_count: thall_report.dormant_count,
            mean_vitality: thall_report.mean_vitality,
            vitality_variance: thall_report.vitality_variance,
            mean_energy: 0.0, // Would come from observer
            tgc_intrinsic: tgc_snap.tgc_intrinsic,
            tgc_combined: gaming_report.normalized_tgc,
            elite_drift: drift,
            voids_available: self.void_tracker.available_count(),
            gaming_suspected: gaming_report.gaming_suspected,
            false_positive_count: 0, // Only known in simulation
        };

        ZaratustraCycleReport {
            cycle,
            thall_report,
            telemetry,
            merkle_root,
            actual_deletions,
            collapse_alerts,
            tgc_combined: gaming_report.normalized_tgc,
            elite_drift: drift,
            gaming_suspected: gaming_report.gaming_suspected,
        }
    }
}

/// Complete report from one Zaratustra cycle.
#[derive(Debug, Clone)]
pub struct ZaratustraCycleReport {
    pub cycle: u64,
    pub thall_report: MikhailThallReport,
    pub telemetry: CycleTelemetry,
    pub merkle_root: [u8; 32],
    pub actual_deletions: usize,
    pub collapse_alerts: Vec<CollapseAlert>,
    pub tgc_combined: f32,
    pub elite_drift: f32,
    pub gaming_suspected: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_judgments(sacred: usize, condemned: usize, dormant: usize) -> Vec<ForgetteringJudgment> {
        let mut judgments = Vec::new();
        for _ in 0..sacred {
            judgments.push(ForgetteringJudgment::sacred(
                Uuid::new_v4(), 0.9, 0.8, 5, 1, "test sacred",
            ));
        }
        for _ in 0..condemned {
            judgments.push(ForgetteringJudgment::condemned(
                Uuid::new_v4(), 0.1, 0.05, 0.0, 1,
            ));
        }
        for _ in 0..dormant {
            judgments.push(ForgetteringJudgment {
                node_id: Uuid::new_v4(),
                vitality_score: 0.4,
                energy: 0.3,
                causal_centrality: 1,
                ricci_delta: 0.0,
                toxicity: 0.2,
                verdict: Verdict::Dormant,
                cycle: 1,
                reason: "test dormant".into(),
            });
        }
        judgments
    }

    #[test]
    fn zaratustra_cycle_basic() {
        let config = NezhmetdinovConfig::default();
        let mut zc = ZaratustraCycle::new(config);

        let judgments = make_judgments(100, 50, 850);
        let report = zc.execute_cycle(&judgments, 1000);

        assert_eq!(report.cycle, 1);
        assert!(report.actual_deletions <= 50);
        assert_eq!(report.telemetry.sacred_count, 100);
    }

    #[test]
    fn max_deletion_rate_enforced() {
        let mut config = NezhmetdinovConfig::default();
        config.bounds.max_deletion_rate = 0.05; // Only 5% per cycle
        let mut zc = ZaratustraCycle::new(config);

        // 500 condemned out of 1000 = 50%, but max is 5%
        let judgments = make_judgments(100, 500, 400);
        let report = zc.execute_cycle(&judgments, 1000);

        assert!(report.actual_deletions <= 50, "max 5% of 1000 = 50, got {}", report.actual_deletions);
    }

    #[test]
    fn min_universe_protected() {
        let mut config = NezhmetdinovConfig::default();
        config.bounds.min_universe_size = 900;
        let mut zc = ZaratustraCycle::new(config);

        // 500 condemned, but universe can't drop below 900
        let judgments = make_judgments(100, 500, 400);
        let report = zc.execute_cycle(&judgments, 1000);

        assert!(report.actual_deletions <= 100, "can only delete to 900, got {} deletions", report.actual_deletions);
    }

    #[test]
    fn ledger_accumulates() {
        let config = NezhmetdinovConfig::default();
        let mut zc = ZaratustraCycle::new(config);

        // Use large universe so min_universe_size (100) doesn't block all deletions
        let j1 = make_judgments(50, 20, 30);
        zc.execute_cycle(&j1, 10000);

        let j2 = make_judgments(40, 15, 25);
        zc.execute_cycle(&j2, 9980);

        assert!(zc.ledger.total_deletions() > 0);
        assert_eq!(zc.ledger.merkle_roots.len(), 2);
    }
}
