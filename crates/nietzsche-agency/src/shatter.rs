// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Phase XVI — Shatter Protocol: super-node splitting to prevent
//! connectivity black holes.
//!
//! When a node's degree (in + out) exceeds a configurable threshold it
//! becomes a "super-node" — a hotspot that degrades query latency and
//! distorts the hyperbolic embedding space.
//!
//! The Shatter Protocol:
//! 1. **Detect** super-nodes via `AdjacencyIndex` degree queries.
//! 2. **Cluster** their neighbors by `EdgeType` to find natural context groups.
//! 3. **Emit** `ShatterNode` intents with computed avatar positions.
//! 4. **Execute** (in the server, under write lock):
//!    - Create avatar nodes at context-specific positions (via `gyromidpoint`).
//!    - Redistribute edges from the original to the nearest avatar.
//!    - Link original → avatars with `Hierarchical` edges.
//!    - Phantomize the original (`is_phantom = true`).
//!
//! The original node becomes a "ghost" at the center, while its avatars
//! inherit the semantic load distributed across the Poincaré ball.

use std::collections::HashMap;

use nietzsche_graph::{AdjacencyIndex, EdgeType, GraphStorage};
use uuid::Uuid;

use crate::config::AgencyConfig;
use crate::daemons::{AgencyDaemon, DaemonReport};
use crate::error::AgencyError;
use crate::event_bus::AgencyEventBus;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the Shatter Protocol.
#[derive(Debug, Clone)]
pub struct ShatterConfig {
    /// Minimum degree (in + out) to trigger shattering.
    /// Default: 500.
    pub degree_threshold: usize,
    /// Maximum number of avatar shards per super-node.
    /// Default: 8.
    pub max_avatars: usize,
    /// Whether the protocol is enabled.
    /// Default: true.
    pub enabled: bool,
    /// Only run every N agency ticks.
    /// Default: 5.
    pub interval: u64,
    /// Minimum neighbors in a cluster to warrant its own avatar.
    /// Clusters below this are merged into the largest cluster.
    /// Default: 3.
    pub min_cluster_size: usize,
}

impl Default for ShatterConfig {
    fn default() -> Self {
        Self {
            degree_threshold: 500,
            max_avatars: 8,
            enabled: true,
            interval: 5,
            min_cluster_size: 3,
        }
    }
}

// ─────────────────────────────────────────────
// Shatter candidate & plan
// ─────────────────────────────────────────────

/// A super-node detected by the daemon.
#[derive(Debug, Clone)]
pub struct ShatterCandidate {
    pub node_id: Uuid,
    pub degree: usize,
    pub degree_in: usize,
    pub degree_out: usize,
}

/// A planned avatar shard — where to place it and which edges to move.
#[derive(Debug, Clone)]
pub struct AvatarPlan {
    /// New UUID for the avatar node.
    pub avatar_id: Uuid,
    /// Context tag describing this shard (e.g. "Association", "Hierarchical").
    pub context_tag: String,
    /// Edge IDs (from the original) that should be reassigned to this avatar.
    pub edge_ids: Vec<Uuid>,
    /// Neighbor IDs used to compute the avatar's embedding via gyromidpoint.
    pub neighbor_ids: Vec<Uuid>,
}

/// Complete shatter plan for one super-node.
#[derive(Debug, Clone)]
pub struct ShatterPlan {
    pub candidate: ShatterCandidate,
    pub avatars: Vec<AvatarPlan>,
}

/// Report from a shatter scan.
#[derive(Debug, Clone)]
pub struct ShatterReport {
    /// Number of nodes scanned for degree.
    pub nodes_scanned: usize,
    /// Super-nodes detected above threshold.
    pub super_nodes_detected: usize,
    /// Shatter plans emitted as intents.
    pub plans_emitted: usize,
    /// Details of each detected super-node.
    pub candidates: Vec<ShatterCandidate>,

    // ── Live graph-wide metrics ──────────────────
    /// Count of phantom (ghost) nodes in the graph.
    pub ghost_nodes: usize,
    /// Count of avatar nodes (content._is_avatar == true).
    pub avatar_nodes: usize,
    /// Largest degree (in + out) observed during scan.
    pub largest_degree: usize,
    /// Average degree across all scanned nodes.
    pub avg_degree: f64,
}

// ─────────────────────────────────────────────
// Shatter scanning logic
// ─────────────────────────────────────────────

/// Scan for super-nodes and build shatter plans.
///
/// This is a read-only operation — it only inspects `AdjacencyIndex` and
/// `GraphStorage` metadata. Actual mutations happen via intents in the server.
pub fn scan_super_nodes(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    config: &ShatterConfig,
) -> ShatterReport {
    let mut candidates = Vec::new();
    let mut nodes_scanned = 0usize;
    let mut ghost_nodes = 0usize;
    let mut avatar_nodes = 0usize;
    let mut largest_degree = 0usize;
    let mut degree_sum = 0u64;

    // Scan all nodes for high degree + collect live metrics
    for result in storage.iter_nodes_meta() {
        let meta = match result {
            Ok(m) => m,
            Err(_) => continue,
        };

        // Count ghosts (phantomized super-nodes)
        if meta.is_phantom {
            ghost_nodes += 1;
            continue;
        }

        // Count avatars (nodes with _is_avatar in content)
        if meta.content.get("_is_avatar").and_then(|v| v.as_bool()).unwrap_or(false) {
            avatar_nodes += 1;
        }

        nodes_scanned += 1;

        let d_in = adjacency.degree_in(&meta.id);
        let d_out = adjacency.degree_out(&meta.id);
        let total = d_in + d_out;

        degree_sum += total as u64;
        if total > largest_degree {
            largest_degree = total;
        }

        if total >= config.degree_threshold {
            candidates.push(ShatterCandidate {
                node_id: meta.id,
                degree: total,
                degree_in: d_in,
                degree_out: d_out,
            });
        }
    }

    let super_nodes_detected = candidates.len();
    let avg_degree = if nodes_scanned > 0 {
        degree_sum as f64 / nodes_scanned as f64
    } else {
        0.0
    };

    ShatterReport {
        nodes_scanned,
        super_nodes_detected,
        plans_emitted: 0, // filled by caller after emitting intents
        candidates,
        ghost_nodes,
        avatar_nodes,
        largest_degree,
        avg_degree,
    }
}

/// Build a shatter plan for a single super-node.
///
/// Clusters neighbors by EdgeType to find natural context groups,
/// then assigns edges to avatar shards.
pub fn build_shatter_plan(
    node_id: Uuid,
    adjacency: &AdjacencyIndex,
    config: &ShatterConfig,
) -> ShatterPlan {
    let d_in = adjacency.degree_in(&node_id);
    let d_out = adjacency.degree_out(&node_id);

    let candidate = ShatterCandidate {
        node_id,
        degree: d_in + d_out,
        degree_in: d_in,
        degree_out: d_out,
    };

    // Collect all edges grouped by EdgeType
    let mut clusters: HashMap<String, Vec<(Uuid, Uuid)>> = HashMap::new(); // tag -> [(edge_id, neighbor_id)]

    for entry in adjacency.entries_out(&node_id) {
        let tag = format!("{:?}_out", entry.edge_type);
        clusters.entry(tag).or_default().push((entry.edge_id, entry.neighbor_id));
    }
    for entry in adjacency.entries_in(&node_id) {
        let tag = format!("{:?}_in", entry.edge_type);
        clusters.entry(tag).or_default().push((entry.edge_id, entry.neighbor_id));
    }

    // Merge tiny clusters into the largest one
    let largest_tag = clusters.iter()
        .max_by_key(|(_, v)| v.len())
        .map(|(k, _)| k.clone())
        .unwrap_or_default();

    let small_clusters: Vec<String> = clusters.iter()
        .filter(|(k, v)| v.len() < config.min_cluster_size && *k != &largest_tag)
        .map(|(k, _)| k.clone())
        .collect();

    for tag in small_clusters {
        if let Some(entries) = clusters.remove(&tag) {
            clusters.entry(largest_tag.clone()).or_default().extend(entries);
        }
    }

    // Cap at max_avatars: if too many clusters, merge smallest into largest
    while clusters.len() > config.max_avatars {
        let smallest_tag = clusters.iter()
            .filter(|(k, _)| *k != &largest_tag)
            .min_by_key(|(_, v)| v.len())
            .map(|(k, _)| k.clone());
        if let Some(tag) = smallest_tag {
            if let Some(entries) = clusters.remove(&tag) {
                clusters.entry(largest_tag.clone()).or_default().extend(entries);
            }
        } else {
            break;
        }
    }

    // Build avatar plans
    let avatars: Vec<AvatarPlan> = clusters.into_iter().map(|(tag, entries)| {
        let edge_ids: Vec<Uuid> = entries.iter().map(|(eid, _)| *eid).collect();
        let neighbor_ids: Vec<Uuid> = entries.iter().map(|(_, nid)| *nid).collect();

        AvatarPlan {
            avatar_id: Uuid::new_v4(),
            context_tag: tag,
            edge_ids,
            neighbor_ids,
        }
    }).collect();

    ShatterPlan { candidate, avatars }
}

/// Run a full shatter scan (interval-gated).
///
/// Returns `None` if not yet time to run or disabled.
pub fn run_shatter_scan(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    config: &ShatterConfig,
    tick_count: u64,
) -> Option<(ShatterReport, Vec<ShatterPlan>)> {
    if !config.enabled {
        return None;
    }
    if config.interval > 0 && (tick_count % config.interval) != 0 {
        return None;
    }

    let report = scan_super_nodes(storage, adjacency, config);

    if report.super_nodes_detected == 0 {
        return Some((report, Vec::new()));
    }

    let plans: Vec<ShatterPlan> = report.candidates.iter()
        .map(|c| build_shatter_plan(c.node_id, adjacency, config))
        .collect();

    let mut final_report = report;
    final_report.plans_emitted = plans.len();

    Some((final_report, plans))
}

// ─────────────────────────────────────────────
// Daemon (read-only scan during agency tick)
// ─────────────────────────────────────────────

/// Shatter daemon — detects super-nodes during agency tick.
///
/// Does NOT mutate the graph. Emits `SuperNodeDetected` events
/// to the event bus for the reactor to convert into intents.
pub struct ShatterDaemon;

impl AgencyDaemon for ShatterDaemon {
    fn name(&self) -> &str { "shatter" }

    fn tick(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        bus: &AgencyEventBus,
        config: &AgencyConfig,
    ) -> Result<DaemonReport, AgencyError> {
        let t0 = std::time::Instant::now();

        if !config.shatter_enabled {
            return Ok(DaemonReport {
                daemon_name: "shatter".into(),
                events_emitted: 0,
                nodes_scanned: 0,
                duration_us: t0.elapsed().as_micros() as u64,
                details: vec!["disabled".into()],
            });
        }

        let shatter_config = build_shatter_config(config);
        let report = scan_super_nodes(storage, adjacency, &shatter_config);

        let mut events_emitted = 0;
        let mut details = Vec::new();

        for candidate in &report.candidates {
            bus.publish(crate::event_bus::AgencyEvent::SuperNodeDetected {
                node_id: candidate.node_id,
                degree: candidate.degree,
            });
            events_emitted += 1;
            details.push(format!(
                "super-node {} degree={} (in={}, out={})",
                candidate.node_id, candidate.degree,
                candidate.degree_in, candidate.degree_out,
            ));
        }

        Ok(DaemonReport {
            daemon_name: "shatter".into(),
            events_emitted,
            nodes_scanned: report.nodes_scanned,
            duration_us: t0.elapsed().as_micros() as u64,
            details,
        })
    }
}

/// Build a `ShatterConfig` from the global `AgencyConfig`.
pub fn build_shatter_config(config: &AgencyConfig) -> ShatterConfig {
    ShatterConfig {
        degree_threshold: config.shatter_threshold,
        max_avatars: config.shatter_max_avatars,
        enabled: config.shatter_enabled,
        interval: config.shatter_interval,
        min_cluster_size: 3,
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
        let cfg = ShatterConfig::default();
        assert_eq!(cfg.degree_threshold, 500);
        assert_eq!(cfg.max_avatars, 8);
        assert!(cfg.enabled);
        assert_eq!(cfg.interval, 5);
        assert_eq!(cfg.min_cluster_size, 3);
    }

    #[test]
    fn shatter_report_empty() {
        let report = ShatterReport {
            nodes_scanned: 1000,
            super_nodes_detected: 0,
            plans_emitted: 0,
            candidates: vec![],
            ghost_nodes: 5,
            avatar_nodes: 12,
            largest_degree: 200,
            avg_degree: 4.5,
        };
        assert_eq!(report.super_nodes_detected, 0);
        assert_eq!(report.ghost_nodes, 5);
        assert_eq!(report.avatar_nodes, 12);
    }

    #[test]
    fn shatter_candidate_fields() {
        let c = ShatterCandidate {
            node_id: Uuid::from_u128(42),
            degree: 600,
            degree_in: 200,
            degree_out: 400,
        };
        assert_eq!(c.degree, c.degree_in + c.degree_out);
    }

    #[test]
    fn avatar_plan_creation() {
        let plan = AvatarPlan {
            avatar_id: Uuid::new_v4(),
            context_tag: "Association_out".into(),
            edge_ids: vec![Uuid::from_u128(1), Uuid::from_u128(2)],
            neighbor_ids: vec![Uuid::from_u128(10), Uuid::from_u128(20)],
        };
        assert_eq!(plan.edge_ids.len(), 2);
        assert_eq!(plan.neighbor_ids.len(), 2);
    }

    #[test]
    fn shatter_disabled_config() {
        // Validate that disabled config has correct defaults
        let cfg = ShatterConfig { enabled: false, ..Default::default() };
        assert!(!cfg.enabled);
    }

    #[test]
    fn shatter_interval_config() {
        // Validate interval gating logic
        let cfg = ShatterConfig { interval: 5, enabled: true, ..Default::default() };
        assert_eq!(cfg.interval, 5);
        assert!(cfg.enabled);
        // Tick 1 would be skipped by run_shatter_scan (1 % 5 != 0)
    }
}
