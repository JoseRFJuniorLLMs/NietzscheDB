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
//!
//! ## Design
//!
//! Read-only daemon — emits `HealingRequired` events. The reactor converts
//! them into `HealNode` / `HealEdge` intents executed under write lock.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use nietzsche_graph::{AdjacencyIndex, GraphStorage};

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

    report
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
            + report.ghost_delete_ids.len();

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

            bus.publish(crate::event_bus::AgencyEvent::HealingRequired {
                report: Box::new(report.clone()),
            });
            events_emitted = 1;
        } else {
            details.push(format!(
                "healthy — {} nodes, {} ghosts ({:.1}%)",
                report.live_count,
                report.ghost_count,
                report.phantom_ratio * 100.0,
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
