//! Morning Report — post-sleep summary of overnight cognitive activity.
//!
//! Aggregates data from the sleep cycle (embedding reconsolidation) and
//! epistemology evaluations (Code-as-Data mutations) into a human-readable
//! report that EVA can present after waking up.
//!
//! # Design
//!
//! The morning report is **dependency-free** — it takes plain data structs,
//! not references to SleepReport or NarrativeEngine. This avoids coupling
//! `nietzsche-agency` to `nietzsche-sleep` or `nietzsche-narrative`.

use serde::{Deserialize, Serialize};

/// Input data for generating a morning report.
///
/// The caller (typically the server's sleep cycle handler) fills this in
/// from the SleepReport, EpistemologyResult, and any narrative data.
#[derive(Debug, Clone, Default)]
pub struct MorningReportInput {
    // ── Sleep cycle data ──────────────────────────
    /// Hausdorff dimension before reconsolidation.
    pub hausdorff_before: f32,
    /// Hausdorff dimension after reconsolidation.
    pub hausdorff_after: f32,
    /// |H_after - H_before|.
    pub hausdorff_delta: f32,
    /// Average semantic drift (Poincaré distance).
    pub semantic_drift_avg: f64,
    /// Whether embeddings were committed or rolled back.
    pub sleep_committed: bool,
    /// Number of node embeddings perturbed.
    pub nodes_perturbed: usize,

    // ── Epistemology data ─────────────────────────
    /// Number of pending mutations evaluated during sleep.
    pub epistemology_evaluated: usize,
    /// Number of mutations accepted (ΔTGC > threshold).
    pub epistemology_accepted: usize,
    /// Short descriptions of accepted mutations.
    pub epistemology_summaries: Vec<String>,

    // ── Graph snapshot ────────────────────────────
    /// Total nodes at time of report.
    pub total_nodes: usize,
    /// Total edges at time of report.
    pub total_edges: usize,
    /// Collection name.
    pub collection: String,
}

/// The generated morning report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorningReport {
    /// When the report was generated (Unix timestamp).
    pub generated_at: i64,
    /// Collection name.
    pub collection: String,
    /// Human-readable summary of overnight activity.
    pub summary: String,
    /// Sleep quality score: 0.0 (rollback, high drift) to 1.0 (committed, low drift).
    pub sleep_quality: f64,
    /// Whether epistemology discovered any improvements.
    pub has_discoveries: bool,
    /// Number of accepted mutations.
    pub discoveries_count: usize,
    /// Individual discovery descriptions.
    pub discoveries: Vec<String>,
}

/// Generate a morning report from the given input data.
pub fn generate_morning_report(input: &MorningReportInput) -> MorningReport {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // ── Sleep quality score ──────────────────────
    // 1.0 = perfect (committed, zero drift)
    // 0.0 = worst (rollback)
    let sleep_quality = if input.sleep_committed {
        // Scale by drift: lower drift = better quality
        let drift_penalty = (input.semantic_drift_avg * 2.0).min(0.5);
        let hausdorff_penalty = (input.hausdorff_delta as f64 * 2.0).min(0.3);
        (1.0 - drift_penalty - hausdorff_penalty).max(0.2)
    } else {
        0.0 // rollback = sleep failed
    };

    // ── Build summary text ───────────────────────
    let mut parts = Vec::new();

    // Sleep status
    if input.sleep_committed {
        parts.push(format!(
            "Sleep cycle committed. {} nodes reconsolidated, Hausdorff delta={:.4}, drift={:.4}.",
            input.nodes_perturbed, input.hausdorff_delta, input.semantic_drift_avg,
        ));
    } else {
        parts.push(format!(
            "Sleep cycle rolled back (drift={:.4} or geometry exceeded threshold). {} nodes were restored to pre-sleep state.",
            input.semantic_drift_avg, input.nodes_perturbed,
        ));
    }

    // Epistemology results
    if input.epistemology_evaluated > 0 {
        parts.push(format!(
            "Epistemology: {}/{} mutations accepted.",
            input.epistemology_accepted, input.epistemology_evaluated,
        ));
        for desc in &input.epistemology_summaries {
            parts.push(format!("  - {}", desc));
        }
    } else {
        parts.push("No epistemology mutations pending during sleep.".to_string());
    }

    // Graph state
    parts.push(format!(
        "Graph state: {} nodes, {} edges in '{}'.",
        input.total_nodes, input.total_edges, input.collection,
    ));

    // Quality assessment
    let quality_label = match sleep_quality {
        q if q >= 0.8 => "excellent",
        q if q >= 0.5 => "good",
        q if q >= 0.2 => "fair",
        _ => "poor (rollback)",
    };
    parts.push(format!("Sleep quality: {} ({:.0}%).", quality_label, sleep_quality * 100.0));

    let summary = parts.join(" ");
    let has_discoveries = input.epistemology_accepted > 0;

    MorningReport {
        generated_at: now,
        collection: input.collection.clone(),
        summary,
        sleep_quality,
        has_discoveries,
        discoveries_count: input.epistemology_accepted,
        discoveries: input.epistemology_summaries.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_sleep_positive_quality() {
        let input = MorningReportInput {
            sleep_committed: true,
            hausdorff_delta: 0.02,
            semantic_drift_avg: 0.05,
            nodes_perturbed: 100,
            collection: "test".to_string(),
            ..Default::default()
        };
        let report = generate_morning_report(&input);
        assert!(report.sleep_quality > 0.5);
        assert!(report.summary.contains("committed"));
    }

    #[test]
    fn rollback_sleep_zero_quality() {
        let input = MorningReportInput {
            sleep_committed: false,
            semantic_drift_avg: 0.8,
            nodes_perturbed: 50,
            collection: "test".to_string(),
            ..Default::default()
        };
        let report = generate_morning_report(&input);
        assert_eq!(report.sleep_quality, 0.0);
        assert!(report.summary.contains("rolled back"));
    }

    #[test]
    fn epistemology_discoveries_reported() {
        let input = MorningReportInput {
            sleep_committed: true,
            epistemology_evaluated: 5,
            epistemology_accepted: 2,
            epistemology_summaries: vec![
                "Optimized decay query: ΔTGC=+0.03".to_string(),
                "Improved edge pruning threshold: ΔTGC=+0.01".to_string(),
            ],
            collection: "main".to_string(),
            ..Default::default()
        };
        let report = generate_morning_report(&input);
        assert!(report.has_discoveries);
        assert_eq!(report.discoveries_count, 2);
        assert!(report.summary.contains("2/5 mutations accepted"));
    }

    #[test]
    fn no_epistemology_no_discoveries() {
        let input = MorningReportInput {
            sleep_committed: true,
            collection: "test".to_string(),
            ..Default::default()
        };
        let report = generate_morning_report(&input);
        assert!(!report.has_discoveries);
        assert!(report.summary.contains("No epistemology mutations"));
    }

    #[test]
    fn report_serializes_to_json() {
        let input = MorningReportInput {
            sleep_committed: true,
            collection: "test".to_string(),
            ..Default::default()
        };
        let report = generate_morning_report(&input);
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("sleep_quality"));
    }
}
