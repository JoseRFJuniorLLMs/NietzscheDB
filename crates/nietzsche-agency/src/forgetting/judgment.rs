//! Judgment system — Verdict and ForgetteringJudgment for the Forgetting Engine.
//!
//! Each node receives a verdict after evaluation by the Nezhmetdinov daemon:
//! - **Sacred**: Protected by vitality or causal immunity
//! - **Dormant**: Waiting for more cycles to prove its value
//! - **Condemned**: Failed the Triple Condition — scheduled for Hard Delete
//! - **Toxic**: High negative emotion without structural value
//! - **RicciShielded**: Would have been condemned but ΔRicci veto saved it
//!
//! Named in honor of Mikhail Tal (escudeiro) — the relentless tactician
//! who assists in executing the Nezhmetdinov sacrifice.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The verdict of the Forgetting Engine on a single node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Verdict {
    /// Protected by high Vitality V(n) > θ_sacred or causal immunity κ(n) > threshold.
    Sacred,
    /// Neither condemned nor sacred — awaiting more cycles.
    Dormant,
    /// Failed the Triple Condition: V(n) < θ AND e(n) < θ_e AND κ(n) = 0.
    /// The guillotine falls. Prepared for Hard Delete.
    Condemned,
    /// High emotional toxicity τ > 0.8 without causal anchoring.
    /// Priority deletion target.
    Toxic,
    /// Would have been Condemned but the Ricci curvature veto (ΔRicci < -ε)
    /// prevented deletion to protect local topology.
    RicciShielded,
}

impl Verdict {
    /// Returns true if this node should be hard-deleted.
    pub fn is_death_sentence(&self) -> bool {
        matches!(self, Verdict::Condemned | Verdict::Toxic)
    }

    /// Returns true if this node is protected from deletion.
    pub fn is_protected(&self) -> bool {
        matches!(self, Verdict::Sacred | Verdict::RicciShielded)
    }

    /// Human-readable label for telemetry.
    pub fn label(&self) -> &'static str {
        match self {
            Verdict::Sacred => "SACRED",
            Verdict::Dormant => "DORMANT",
            Verdict::Condemned => "CONDEMNED",
            Verdict::Toxic => "TOXIC",
            Verdict::RicciShielded => "RICCI_SHIELDED",
        }
    }
}

/// Complete judgment record for a single node after Nezhmetdinov evaluation.
///
/// This is the "Mikhail Thall Report" — the squire's detailed account of
/// each node's trial before the Forgetting Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetteringJudgment {
    /// The node being judged.
    pub node_id: Uuid,
    /// Computed vitality score V(n) ∈ (0, 1).
    pub vitality_score: f32,
    /// Current node energy e(n) ∈ [0, 1].
    pub energy: f32,
    /// Causal centrality κ(n) — count of Minkowski timelike/lightlike edges.
    pub causal_centrality: usize,
    /// Simulated Ricci curvature change if this node were removed.
    /// Negative = topology would collapse. ΔRicci < -ε triggers veto.
    pub ricci_delta: f32,
    /// Emotional toxicity τ ∈ [0, 1].
    pub toxicity: f32,
    /// The final verdict.
    pub verdict: Verdict,
    /// Cycle number when this judgment was made.
    pub cycle: u64,
    /// Reason string for audit trail.
    pub reason: String,
}

impl ForgetteringJudgment {
    /// Create a Sacred judgment (node protected).
    pub fn sacred(node_id: Uuid, vitality: f32, energy: f32, causal: usize, cycle: u64, reason: impl Into<String>) -> Self {
        Self {
            node_id,
            vitality_score: vitality,
            energy,
            causal_centrality: causal,
            ricci_delta: 0.0,
            toxicity: 0.0,
            verdict: Verdict::Sacred,
            cycle,
            reason: reason.into(),
        }
    }

    /// Create a Condemned judgment (node scheduled for deletion).
    pub fn condemned(node_id: Uuid, vitality: f32, energy: f32, ricci_delta: f32, cycle: u64) -> Self {
        Self {
            node_id,
            vitality_score: vitality,
            energy,
            causal_centrality: 0,
            ricci_delta,
            toxicity: 0.0,
            verdict: Verdict::Condemned,
            cycle,
            reason: format!(
                "Triple condition met: V={:.4} < θ, e={:.4} < θ_e, κ=0",
                vitality, energy
            ),
        }
    }
}

/// Summary of all judgments in a forgetting cycle.
///
/// The "Mikhail Thall Cycle Report" — aggregate statistics for telemetry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MikhailThallReport {
    /// Cycle number.
    pub cycle: u64,
    /// Total nodes evaluated.
    pub nodes_evaluated: usize,
    /// Nodes condemned (will be deleted).
    pub condemned_count: usize,
    /// Nodes marked toxic (will be deleted).
    pub toxic_count: usize,
    /// Nodes saved by Ricci shield.
    pub ricci_shielded_count: usize,
    /// Nodes declared sacred.
    pub sacred_count: usize,
    /// Nodes dormant (waiting).
    pub dormant_count: usize,
    /// Mean vitality of all evaluated nodes.
    pub mean_vitality: f32,
    /// Variance of vitality scores.
    pub vitality_variance: f32,
    /// Duration of evaluation in microseconds.
    pub duration_us: u64,
}

impl MikhailThallReport {
    /// Build from a list of judgments.
    pub fn from_judgments(cycle: u64, judgments: &[ForgetteringJudgment], duration_us: u64) -> Self {
        let nodes_evaluated = judgments.len();
        let mut condemned = 0usize;
        let mut toxic = 0usize;
        let mut ricci_shielded = 0usize;
        let mut sacred = 0usize;
        let mut dormant = 0usize;
        let mut vitality_sum = 0.0f32;

        for j in judgments {
            vitality_sum += j.vitality_score;
            match j.verdict {
                Verdict::Condemned => condemned += 1,
                Verdict::Toxic => toxic += 1,
                Verdict::RicciShielded => ricci_shielded += 1,
                Verdict::Sacred => sacred += 1,
                Verdict::Dormant => dormant += 1,
            }
        }

        let mean_vitality = if nodes_evaluated > 0 {
            vitality_sum / nodes_evaluated as f32
        } else {
            0.0
        };

        let vitality_variance = if nodes_evaluated > 0 {
            judgments.iter()
                .map(|j| (j.vitality_score - mean_vitality).powi(2))
                .sum::<f32>() / nodes_evaluated as f32
        } else {
            0.0
        };

        Self {
            cycle,
            nodes_evaluated,
            condemned_count: condemned,
            toxic_count: toxic,
            ricci_shielded_count: ricci_shielded,
            sacred_count: sacred,
            dormant_count: dormant,
            mean_vitality,
            vitality_variance,
            duration_us,
        }
    }

    /// Total nodes that will be deleted (condemned + toxic).
    pub fn total_deletions(&self) -> usize {
        self.condemned_count + self.toxic_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_is_death_sentence() {
        assert!(Verdict::Condemned.is_death_sentence());
        assert!(Verdict::Toxic.is_death_sentence());
        assert!(!Verdict::Sacred.is_death_sentence());
        assert!(!Verdict::Dormant.is_death_sentence());
        assert!(!Verdict::RicciShielded.is_death_sentence());
    }

    #[test]
    fn verdict_is_protected() {
        assert!(Verdict::Sacred.is_protected());
        assert!(Verdict::RicciShielded.is_protected());
        assert!(!Verdict::Condemned.is_protected());
        assert!(!Verdict::Toxic.is_protected());
        assert!(!Verdict::Dormant.is_protected());
    }

    #[test]
    fn mikhail_thall_report_from_judgments() {
        let id = Uuid::new_v4();
        let judgments = vec![
            ForgetteringJudgment::sacred(id, 0.9, 0.8, 5, 1, "high vitality"),
            ForgetteringJudgment::condemned(id, 0.1, 0.05, 0.0, 1),
            ForgetteringJudgment {
                node_id: id, vitality_score: 0.3, energy: 0.2,
                causal_centrality: 0, ricci_delta: -0.2, toxicity: 0.9,
                verdict: Verdict::Toxic, cycle: 1, reason: "toxic".into(),
            },
        ];
        let report = MikhailThallReport::from_judgments(1, &judgments, 500);
        assert_eq!(report.nodes_evaluated, 3);
        assert_eq!(report.sacred_count, 1);
        assert_eq!(report.condemned_count, 1);
        assert_eq!(report.toxic_count, 1);
        assert_eq!(report.total_deletions(), 2);
    }
}
