//! Phase XIX — Self-Healing Graph: autonomous graph maintenance daemon.
//!
//! Large knowledge graphs inevitably degrade over time:
//!
//! - **Boundary drift**: embeddings approach ‖x‖ → 1.0, causing numerical instability
//! - **Orphan nodes**: nodes with zero edges (disconnected from the manifold)
//! - **Dead edges**: edges pointing to phantom/deleted nodes that weren't cleaned up
//! - **Energy decay**: nodes with energy → 0 that accumulate as dead weight
//! - **Ghost accumulation**: too many phantom nodes wasting storage
//!
//! The Self-Healing daemon detects these pathologies during agency ticks and
//! emits repair events for the reactor to convert into mutation intents.
//!
//! ## Healing operations
//!
//! | Pathology | Detection | Repair |
//! |-----------|-----------|--------|
//! | Boundary drift | ‖embedding‖ > `norm_threshold` | Re-project to `target_norm` |
//! | Orphan nodes | degree(n) = 0, age > threshold | Phantomize or delete |
//! | Dead edges | edge.to or edge.from is phantom | Delete edge |
//! | Energy exhaustion | energy ≤ 0 for > N ticks | Phantomize |
//! | Ghost accumulation | phantom_ratio > threshold | Hard-delete oldest ghosts |
//! | Neural anomaly | anomaly_detector score > threshold | Flag for inspection |
//!
//! - **Neural anomalies**: ONNX anomaly_detector flags structurally degenerate embeddings
//!
//! ## Design
//!
//! Read-only daemon — emits `HealingRequired` events. The reactor converts
//! them into `HealNode` / `HealEdge` intents executed under write lock.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use nietzsche_neural::REGISTRY;

use crate::config::AgencyConfig;
use crate::daemons::{AgencyDaemon, DaemonReport};
use crate::error::AgencyError;
use crate::event_bus::AgencyEventBus;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the Self-Healing daemon.
#[derive(Debug, Clone)]
pub struct HealingConfig {
    /// Whether self-healing is enabled.
    pub enabled: bool,
    /// Tick interval between healing scans.
    pub interval: u64,
    /// Maximum nodes to scan per tick.
    pub max_scan: usize,
    /// Embedding norm above which re-projection is triggered (default: 0.995).
    pub norm_threshold: f64,
    /// Target norm after re-projection (default: 0.98).
    pub target_norm: f64,
    /// Minimum age (seconds) for an orphan node to be considered for cleanup.
    pub orphan_min_age_secs: i64,
    /// Maximum number of phantom nodes to hard-delete per tick.
    pub ghost_cleanup_batch: usize,
    /// Phantom ratio above which ghost cleanup triggers (default: 0.15).
    pub ghost_ratio_threshold: f64,
    /// Whether neural anomaly detection is enabled (default: false).
    pub neural_enabled: bool,
    /// Anomaly score threshold for flagging a node (default: 0.7).
    pub neural_threshold: f64,
    /// Maximum nodes to run through neural inference per tick (default: 200).
    pub neural_max_infer: usize,
}

impl Default for HealingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: 10,
            max_scan: 5000,
            norm_threshold: 0.995,
            target_norm: 0.98,
            orphan_min_age_secs: 3600, // 1 hour
            ghost_cleanup_batch: 100,
            ghost_ratio_threshold: 0.15,
            neural_enabled: false,
            neural_threshold: 0.7,
            neural_max_infer: 200,
        }
    }
}

// ─────────────────────────────────────────────
// Healing report
// ─────────────────────────────────────────────

/// Diagnostic report from a self-healing scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealingReport {
    /// Number of nodes scanned.
    pub nodes_scanned: usize,
    /// Nodes with boundary-drifted embeddings (‖x‖ > threshold).
    pub boundary_drift_count: usize,
    /// Orphan nodes (zero edges).
    pub orphan_count: usize,
    /// Dead edges (pointing to phantom/missing nodes).
    pub dead_edge_count: usize,
    /// Exhausted nodes (energy ≤ 0).
    pub exhausted_count: usize,
    /// Ghost (phantom) nodes in the graph.
    pub ghost_count: usize,
    /// Total live nodes.
    pub live_count: usize,
    /// Phantom ratio (ghost_count / total).
    pub phantom_ratio: f64,
    /// IDs of nodes needing embedding re-projection.
    pub drift_node_ids: Vec<Uuid>,
    /// IDs of orphan nodes to clean up.
    pub orphan_node_ids: Vec<Uuid>,
    /// IDs of dead edges to remove.
    pub dead_edge_ids: Vec<Uuid>,
    /// IDs of exhausted nodes to phantomize.
    pub exhausted_node_ids: Vec<Uuid>,
    /// IDs of oldest ghosts to hard-delete.
    pub ghost_delete_ids: Vec<Uuid>,
    /// Nodes flagged by neural anomaly detector (score > threshold).
    pub neural_anomaly_count: usize,
    /// IDs of nodes flagged as anomalous by the neural detector.
    pub neural_anomaly_ids: Vec<Uuid>,
    /// Whether neural detection was actually used this scan.
    pub neural_detection_active: bool,
}

impl Default for HealingReport {
    fn default() -> Self {
        Self {
            nodes_scanned: 0,
            boundary_drift_count: 0,
            orphan_count: 0,
            dead_edge_count: 0,
            exhausted_count: 0,
            ghost_count: 0,
            live_count: 0,
            phantom_ratio: 0.0,
            drift_node_ids: vec![],
            orphan_node_ids: vec![],
            dead_edge_ids: vec![],
            exhausted_node_ids: vec![],
            ghost_delete_ids: vec![],
            neural_anomaly_count: 0,
            neural_anomaly_ids: vec![],
            neural_detection_active: false,
        }
    }
}

// ─────────────────────────────────────────────
// Healing scan logic
// ─────────────────────────────────────────────

/// Run a self-healing scan on the graph.
///
/// This is read-only — it identifies pathologies but does not mutate.
/// The caller (reactor) is responsible for executing repairs.
pub fn scan_healing(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    config: &HealingConfig,
) -> HealingReport {
    let mut report = HealingReport::default();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let mut scanned = 0usize;
    let mut phantom_nodes: Vec<(Uuid, i64)> = Vec::new(); // (id, created_at) for age-sorting

    // ── Phase 1: Scan nodes ──────────────────────────

    for result in storage.iter_nodes_meta() {
        let meta = match result {
            Ok(m) => m,
            Err(_) => continue,
        };

        if scanned >= config.max_scan {
            break;
        }
        scanned += 1;

        // Track phantoms
        if meta.is_phantom {
            report.ghost_count += 1;
            phantom_nodes.push((meta.id, meta.created_at));
            continue;
        }

        report.live_count += 1;

        // Check 1: Orphan detection (zero edges)
        let degree = adjacency.degree_in(&meta.id) + adjacency.degree_out(&meta.id);
        if degree == 0 {
            let age = now - meta.created_at;
            if age > config.orphan_min_age_secs {
                report.orphan_count += 1;
                if report.orphan_node_ids.len() < 100 {
                    report.orphan_node_ids.push(meta.id);
                }
            }
        }

        // Check 2: Energy exhaustion
        if meta.energy <= 0.0 {
            report.exhausted_count += 1;
            if report.exhausted_node_ids.len() < 100 {
                report.exhausted_node_ids.push(meta.id);
            }
        }

        // Check 3: Boundary drift (need full node for embedding)
        if let Ok(Some(node)) = storage.get_node(&meta.id) {
            let norm: f64 = node.embedding.coords.iter()
                .map(|c| (*c as f64) * (*c as f64))
                .sum::<f64>()
                .sqrt();

            if norm > config.norm_threshold {
                report.boundary_drift_count += 1;
                if report.drift_node_ids.len() < 100 {
                    report.drift_node_ids.push(meta.id);
                }
            }
        }
    }

    report.nodes_scanned = scanned;

    // ── Phase 2: Dead edge detection ─────────────────

    // Collect phantom node IDs for fast lookup
    let phantom_set: HashSet<Uuid> = phantom_nodes.iter().map(|(id, _)| *id).collect();

    // Sample edges for dead references
    let mut edge_count = 0usize;
    for result in storage.iter_edges() {
        let edge = match result {
            Ok(e) => e,
            Err(_) => continue,
        };

        edge_count += 1;
        if edge_count > config.max_scan {
            break;
        }

        // Check if either endpoint is phantom
        if phantom_set.contains(&edge.from) || phantom_set.contains(&edge.to) {
            report.dead_edge_count += 1;
            if report.dead_edge_ids.len() < 100 {
                report.dead_edge_ids.push(edge.id);
            }
        }
    }

    // ── Phase 3: Ghost cleanup candidates ────────────

    let total = report.ghost_count + report.live_count;
    report.phantom_ratio = if total > 0 {
        report.ghost_count as f64 / total as f64
    } else {
        0.0
    };

    if report.phantom_ratio > config.ghost_ratio_threshold {
        // Sort by creation date (oldest first) and select batch
        phantom_nodes.sort_by_key(|(_, created_at)| *created_at);
        report.ghost_delete_ids = phantom_nodes.iter()
            .take(config.ghost_cleanup_batch)
            .map(|(id, _)| *id)
            .collect();
    }

    // ── Phase 4: Neural anomaly detection ─────────
    //
    // If AGENCY_HEALING_NEURAL_ENABLED=true and the anomaly_detector model
    // is loaded in the ONNX registry, run inference on live node embeddings
    // to detect structurally degenerate patterns that heuristics miss.
    //
    // Input:  [1, 64]  — first 64 components of the node embedding (zero-padded)
    // Output: [1, 65]  — 64D reconstruction + 1D anomaly score
    // The combined score (learned + reconstruction error) is checked against
    // `config.neural_threshold`. Nodes above the threshold are flagged.

    if config.neural_enabled {
        match run_neural_anomaly_scan(storage, config) {
            Ok((ids, count)) => {
                report.neural_detection_active = true;
                report.neural_anomaly_count = count;
                report.neural_anomaly_ids = ids;
            }
            Err(reason) => {
                // Graceful fallback: log and continue with heuristic-only results
                tracing::debug!(reason = %reason, "neural anomaly detection skipped");
            }
        }
    }

    report
}

// ─────────────────────────────────────────────
// Neural anomaly detection (ONNX)
// ─────────────────────────────────────────────

/// Run the `anomaly_detector` ONNX model over live node embeddings.
///
/// Returns `(flagged_ids, flagged_count)` on success, or a reason string on
/// failure (model not loaded, lock poisoned, etc.). The caller should treat
/// failure as a non-fatal fallback to heuristic-only mode.
///
/// Uses `REGISTRY.infer_f32()` convenience wrapper to avoid depending on
/// `ort` / `ndarray` crate types directly.
fn run_neural_anomaly_scan(
    storage: &GraphStorage,
    config: &HealingConfig,
) -> Result<(Vec<Uuid>, usize), String> {
    // 1. Check that the model is loaded — bail early if not
    if !REGISTRY.has_model("anomaly_detector") {
        return Err("anomaly_detector model not loaded in REGISTRY".into());
    }

    let mut flagged_ids: Vec<Uuid> = Vec::new();
    let mut flagged_count = 0usize;
    let mut inferred = 0usize;

    // 2. Iterate live nodes, run inference on each
    for result in storage.iter_nodes_meta() {
        let meta = match result {
            Ok(m) => m,
            Err(_) => continue,
        };

        // Skip phantoms and exhausted nodes (already handled by heuristics)
        if meta.is_phantom || meta.energy <= 0.0 {
            continue;
        }

        if inferred >= config.neural_max_infer {
            break;
        }

        // Load the full node to get the embedding
        let node = match storage.get_node(&meta.id) {
            Ok(Some(n)) => n,
            _ => continue,
        };

        // Build the 64D input: take first 64 coords, zero-pad if shorter
        let mut input_data = vec![0.0f32; 64];
        let copy_len = node.embedding.coords.len().min(64);
        for i in 0..copy_len {
            input_data[i] = node.embedding.coords[i] as f32;
        }

        // 3. Run inference via REGISTRY.infer_f32()
        //    Input:  [1, 64]  (node embedding, zero-padded)
        //    Output: [1, 65]  (64D reconstruction + 1D anomaly score)
        let raw = match REGISTRY.infer_f32("anomaly_detector", vec![1, 64], input_data.clone()) {
            Ok(output) => output,
            Err(e) => {
                tracing::trace!(node_id = %meta.id, error = %e, "anomaly inference failed");
                continue;
            }
        };

        inferred += 1;

        // 4. Parse the output: first 64 = reconstruction, element 64 = learned anomaly score
        if raw.len() < 65 {
            tracing::trace!(
                node_id = %meta.id,
                output_len = raw.len(),
                "anomaly_detector output too short, expected 65"
            );
            continue;
        }

        let reconstructed = &raw[..64];
        let anomaly_score = raw[64];

        // Compute reconstruction error (MSE)
        let reconstruction_error: f32 = input_data
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 64.0;

        // Combined score: 50% learned + 50% sigmoid(reconstruction_error * 10)
        let recon_sigmoid = 1.0 / (1.0 + (-reconstruction_error * 10.0).exp());
        let combined_score = anomaly_score * 0.5 + recon_sigmoid * 0.5;

        // 5. Flag if above threshold
        if (combined_score as f64) > config.neural_threshold {
            flagged_count += 1;
            if flagged_ids.len() < 100 {
                flagged_ids.push(meta.id);
            }

            tracing::debug!(
                node_id = %meta.id,
                anomaly_score,
                reconstruction_error,
                combined_score,
                "neural anomaly detected"
            );
        }
    }

    tracing::info!(
        inferred,
        flagged = flagged_count,
        threshold = config.neural_threshold,
        "neural anomaly scan complete"
    );

    Ok((flagged_ids, flagged_count))
}

// ─────────────────────────────────────────────
// Daemon implementation
// ─────────────────────────────────────────────

/// Self-Healing daemon — detects graph pathologies during agency tick.
///
/// Emits `HealingRequired` events for the reactor to process.
pub struct SelfHealingDaemon;

impl AgencyDaemon for SelfHealingDaemon {
    fn name(&self) -> &str { "self_healing" }

    fn tick(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        bus: &AgencyEventBus,
        config: &AgencyConfig,
    ) -> Result<DaemonReport, AgencyError> {
        let t0 = std::time::Instant::now();

        if !config.healing_enabled {
            return Ok(DaemonReport {
                daemon_name: "self_healing".into(),
                events_emitted: 0,
                nodes_scanned: 0,
                duration_us: t0.elapsed().as_micros() as u64,
                details: vec!["disabled".into()],
            });
        }

        let healing_config = build_healing_config(config);
        let report = scan_healing(storage, adjacency, &healing_config);

        let mut events_emitted = 0;
        let mut details = Vec::new();

        let total_issues = report.boundary_drift_count
            + report.orphan_count
            + report.dead_edge_count
            + report.exhausted_count
            + report.ghost_delete_ids.len()
            + report.neural_anomaly_count;

        if total_issues > 0 {
            details.push(format!(
                "drift={}, orphans={}, dead_edges={}, exhausted={}, ghosts={} (ratio={:.2}%)",
                report.boundary_drift_count,
                report.orphan_count,
                report.dead_edge_count,
                report.exhausted_count,
                report.ghost_count,
                report.phantom_ratio * 100.0,
            ));

            if report.neural_detection_active {
                details.push(format!(
                    "neural_anomalies={} (threshold={:.2})",
                    report.neural_anomaly_count,
                    config.healing_neural_threshold,
                ));
            }

            bus.publish(crate::event_bus::AgencyEvent::HealingRequired {
                report: Box::new(report.clone()),
            });
            events_emitted = 1;
        } else {
            let neural_status = if report.neural_detection_active {
                ", neural=clean"
            } else {
                ""
            };
            details.push(format!(
                "healthy — {} nodes, {} ghosts ({:.1}%){}",
                report.live_count,
                report.ghost_count,
                report.phantom_ratio * 100.0,
                neural_status,
            ));
        }

        Ok(DaemonReport {
            daemon_name: "self_healing".into(),
            events_emitted,
            nodes_scanned: report.nodes_scanned,
            duration_us: t0.elapsed().as_micros() as u64,
            details,
        })
    }
}

/// Build a `HealingConfig` from the global `AgencyConfig`.
pub fn build_healing_config(config: &AgencyConfig) -> HealingConfig {
    HealingConfig {
        enabled: config.healing_enabled,
        interval: config.healing_interval,
        max_scan: config.healing_max_scan,
        norm_threshold: config.healing_norm_threshold,
        target_norm: 0.98,
        orphan_min_age_secs: config.healing_orphan_min_age,
        ghost_cleanup_batch: 100,
        ghost_ratio_threshold: config.healing_ghost_ratio,
        neural_enabled: config.healing_neural_enabled,
        neural_threshold: config.healing_neural_threshold,
        neural_max_infer: config.healing_neural_max_infer,
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = HealingConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.interval, 10);
        assert_eq!(cfg.max_scan, 5000);
        assert!((cfg.norm_threshold - 0.995).abs() < 1e-6);
        assert_eq!(cfg.orphan_min_age_secs, 3600);
    }

    #[test]
    fn healing_report_serializes() {
        let report = HealingReport {
            nodes_scanned: 1000,
            boundary_drift_count: 5,
            orphan_count: 10,
            dead_edge_count: 3,
            exhausted_count: 2,
            ghost_count: 50,
            live_count: 950,
            phantom_ratio: 0.05,
            drift_node_ids: vec![],
            orphan_node_ids: vec![],
            dead_edge_ids: vec![],
            exhausted_node_ids: vec![],
            ghost_delete_ids: vec![],
            neural_anomaly_count: 0,
            neural_anomaly_ids: vec![],
            neural_detection_active: false,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"nodes_scanned\":1000"));
        assert!(json.contains("\"orphan_count\":10"));
    }

    #[test]
    fn empty_report() {
        let report = HealingReport::default();
        assert_eq!(report.nodes_scanned, 0);
        assert_eq!(report.boundary_drift_count, 0);
        assert_eq!(report.phantom_ratio, 0.0);
    }
}
