//! Phase XXI — Knowledge Compression: semantic deduplication and merge engine.
//!
//! Large knowledge graphs accumulate redundancy over time:
//!
//! - **Near-duplicate nodes**: embeddings within ε distance
//! - **Redundant paths**: multiple routes encoding the same semantic relationship
//! - **Stale clusters**: groups of low-energy nodes that can be summarized
//!
//! Knowledge Compression detects these patterns and proposes merge operations
//! that preserve semantic content while reducing graph size.
//!
//! ## Compression operations
//!
//! | Pattern | Detection | Compression |
//! |---------|-----------|-------------|
//! | Near-duplicates | d(a,b) < ε | Merge into archetype, redirect edges |
//! | Stale clusters | energy < threshold, degree < limit | Summarize into single node |
//! | Redundant paths | same endpoints, similar weights | Keep strongest, prune rest |
//!
//! ## Design
//!
//! Read-only scan → produce `CompressionPlan` → reactor executes merges.
//! The Niilista GC daemon already detects near-duplicates; this module goes
//! further by analyzing structural redundancy and proposing merges.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use nietzsche_hyp_ops::poincare_distance;

use crate::config::AgencyConfig;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for Knowledge Compression.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Whether compression is enabled.
    pub enabled: bool,
    /// Tick interval between compression scans (default: 20).
    pub interval: u64,
    /// Maximum nodes to scan per tick (default: 2000).
    pub max_scan: usize,
    /// Poincaré distance threshold for near-duplicate detection (default: 0.005).
    pub duplicate_epsilon: f64,
    /// Energy threshold below which a node is considered "stale" (default: 0.05).
    pub stale_energy_threshold: f32,
    /// Maximum degree for a node to be considered for cluster merge (default: 5).
    pub stale_max_degree: usize,
    /// Minimum cluster size to trigger compression (default: 3).
    pub min_cluster_size: usize,
    /// Maximum merges to propose per tick (default: 50).
    pub max_merges_per_tick: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: 20,
            max_scan: 2000,
            duplicate_epsilon: 0.005,
            stale_energy_threshold: 0.05,
            stale_max_degree: 5,
            min_cluster_size: 3,
            max_merges_per_tick: 50,
        }
    }
}

/// Build a `CompressionConfig` from the global `AgencyConfig`.
pub fn build_compression_config(config: &AgencyConfig) -> CompressionConfig {
    CompressionConfig {
        enabled: config.compression_enabled,
        interval: config.compression_interval,
        max_scan: config.compression_max_scan,
        duplicate_epsilon: config.compression_duplicate_epsilon,
        stale_energy_threshold: config.compression_stale_energy,
        stale_max_degree: config.compression_stale_max_degree,
        min_cluster_size: config.compression_min_cluster,
        max_merges_per_tick: config.compression_max_merges,
    }
}

// ─────────────────────────────────────────────
// Compression report
// ─────────────────────────────────────────────

/// Report from a compression scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionReport {
    /// Nodes scanned.
    pub nodes_scanned: usize,
    /// Near-duplicate pairs found.
    pub duplicate_pairs: usize,
    /// Stale clusters identified.
    pub stale_clusters: usize,
    /// Total nodes in stale clusters.
    pub stale_cluster_nodes: usize,
    /// Redundant paths detected.
    pub redundant_paths: usize,
    /// Proposed merge operations.
    pub merge_proposals: Vec<MergeProposal>,
    /// Estimated node reduction if all merges execute.
    pub estimated_reduction: usize,
}

/// A proposed merge operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeProposal {
    /// The archetype (surviving) node.
    pub archetype_id: Uuid,
    /// Nodes to merge into the archetype.
    pub merge_ids: Vec<Uuid>,
    /// Reason for the merge.
    pub reason: MergeReason,
    /// Confidence score (0.0–1.0).
    pub confidence: f64,
}

/// Why a merge was proposed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeReason {
    /// Near-duplicate embeddings (d < ε).
    NearDuplicate { distance: f64 },
    /// Stale low-energy cluster.
    StaleCluster { mean_energy: f32, cluster_size: usize },
    /// Redundant parallel paths between same endpoints.
    RedundantPath { endpoint_a: Uuid, endpoint_b: Uuid },
}

// ─────────────────────────────────────────────
// Compression scan
// ─────────────────────────────────────────────

/// Run a compression scan on the graph.
///
/// Read-only — produces `CompressionReport` with proposed merges.
pub fn scan_compression(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    config: &CompressionConfig,
) -> CompressionReport {
    let mut report = CompressionReport {
        nodes_scanned: 0,
        duplicate_pairs: 0,
        stale_clusters: 0,
        stale_cluster_nodes: 0,
        redundant_paths: 0,
        merge_proposals: Vec::new(),
        estimated_reduction: 0,
    };

    // ── Phase 1: Collect candidate nodes ────────
    struct NodeInfo {
        id: Uuid,
        energy: f32,
        embedding: Vec<f32>,
        degree: usize,
    }

    let mut candidates: Vec<NodeInfo> = Vec::new();
    let mut scanned = 0usize;

    for result in storage.iter_nodes_meta() {
        let meta = match result {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.is_phantom { continue; }
        if scanned >= config.max_scan { break; }
        scanned += 1;

        let degree = adjacency.degree_in(&meta.id) + adjacency.degree_out(&meta.id);

        // Load embedding for duplicate detection
        if let Ok(Some(node)) = storage.get_node(&meta.id) {
            candidates.push(NodeInfo {
                id: meta.id,
                energy: meta.energy,
                embedding: node.embedding.coords.iter().map(|c| *c).collect(),
                degree,
            });
        }
    }
    report.nodes_scanned = scanned;

    // ── Phase 2: Near-duplicate detection ───────
    // Brute-force pairwise (bounded by max_scan)
    let mut merged_into: HashMap<Uuid, Uuid> = HashMap::new();

    for i in 0..candidates.len() {
        if merged_into.contains_key(&candidates[i].id) { continue; }
        if report.merge_proposals.len() >= config.max_merges_per_tick { break; }

        let ei: Vec<f64> = candidates[i].embedding.iter().map(|c| *c as f64).collect();

        let mut group = Vec::new();

        for j in (i + 1)..candidates.len() {
            if merged_into.contains_key(&candidates[j].id) { continue; }

            let ej: Vec<f64> = candidates[j].embedding.iter().map(|c| *c as f64).collect();

            if ei.len() != ej.len() || ei.is_empty() { continue; }

            let dist = poincare_distance(&ei, &ej);
            if dist < config.duplicate_epsilon {
                group.push((candidates[j].id, dist));
                merged_into.insert(candidates[j].id, candidates[i].id);
            }
        }

        if !group.is_empty() {
            let min_dist = group.iter().map(|(_, d)| *d).fold(f64::MAX, f64::min);
            report.duplicate_pairs += group.len();
            report.merge_proposals.push(MergeProposal {
                archetype_id: candidates[i].id,
                merge_ids: group.iter().map(|(id, _)| *id).collect(),
                reason: MergeReason::NearDuplicate { distance: min_dist },
                confidence: 1.0 - (min_dist / config.duplicate_epsilon).min(1.0),
            });
        }
    }

    // ── Phase 3: Stale cluster detection ────────
    // Find groups of low-energy, low-degree nodes
    let stale_nodes: Vec<&NodeInfo> = candidates.iter()
        .filter(|n| {
            n.energy <= config.stale_energy_threshold
                && n.degree <= config.stale_max_degree
                && !merged_into.contains_key(&n.id)
        })
        .collect();

    if stale_nodes.len() >= config.min_cluster_size
        && report.merge_proposals.len() < config.max_merges_per_tick
    {
        // Group stale nodes by proximity
        let mut stale_groups: Vec<Vec<&NodeInfo>> = Vec::new();
        let mut assigned: HashMap<Uuid, usize> = HashMap::new();

        for node in &stale_nodes {
            let ne: Vec<f64> = node.embedding.iter().map(|c| *c as f64).collect();

            let mut found_group = None;
            for (gi, group) in stale_groups.iter().enumerate() {
                if let Some(repr) = group.first() {
                    let re: Vec<f64> = repr.embedding.iter().map(|c| *c as f64).collect();
                    if ne.len() == re.len() && !ne.is_empty() {
                        let dist = poincare_distance(&ne, &re);
                        if dist < config.duplicate_epsilon * 10.0 {
                            found_group = Some(gi);
                            break;
                        }
                    }
                }
            }

            match found_group {
                Some(gi) => {
                    stale_groups[gi].push(node);
                    assigned.insert(node.id, gi);
                }
                None => {
                    let gi = stale_groups.len();
                    stale_groups.push(vec![node]);
                    assigned.insert(node.id, gi);
                }
            }
        }

        for group in &stale_groups {
            if group.len() >= config.min_cluster_size
                && report.merge_proposals.len() < config.max_merges_per_tick
            {
                let mean_energy = group.iter().map(|n| n.energy).sum::<f32>() / group.len() as f32;

                // Pick highest-energy node as archetype
                let archetype = group.iter()
                    .max_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();

                let merge_ids: Vec<Uuid> = group.iter()
                    .filter(|n| n.id != archetype.id)
                    .map(|n| n.id)
                    .collect();

                report.stale_clusters += 1;
                report.stale_cluster_nodes += group.len();
                report.merge_proposals.push(MergeProposal {
                    archetype_id: archetype.id,
                    merge_ids,
                    reason: MergeReason::StaleCluster {
                        mean_energy,
                        cluster_size: group.len(),
                    },
                    confidence: 1.0 - mean_energy as f64,
                });
            }
        }
    }

    // ── Phase 4: Redundant path detection ───────
    // Find nodes with multiple edges to the same target
    for node in &candidates {
        if report.merge_proposals.len() >= config.max_merges_per_tick { break; }

        let out_edges: Vec<Uuid> = adjacency.neighbors_out(&node.id);
        if out_edges.len() < 2 { continue; }

        // Count edges per target
        let mut target_counts: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        for edge_id in &out_edges {
            if let Ok(Some(edge)) = storage.get_edge(edge_id) {
                target_counts.entry(edge.to).or_default().push(edge.id);
            }
        }

        for (target, edge_ids) in &target_counts {
            if edge_ids.len() >= 2 {
                report.redundant_paths += 1;
                // Keep first edge, mark rest as redundant
                let redundant: Vec<Uuid> = edge_ids[1..].to_vec();
                report.merge_proposals.push(MergeProposal {
                    archetype_id: edge_ids[0],
                    merge_ids: redundant,
                    reason: MergeReason::RedundantPath {
                        endpoint_a: node.id,
                        endpoint_b: *target,
                    },
                    confidence: 0.9,
                });
            }
        }
    }

    report.estimated_reduction = report.merge_proposals.iter()
        .map(|p| p.merge_ids.len())
        .sum();

    report
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = CompressionConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.interval, 20);
        assert_eq!(cfg.max_scan, 2000);
        assert!((cfg.duplicate_epsilon - 0.005).abs() < 1e-6);
    }

    #[test]
    fn merge_proposal_serializes() {
        let p = MergeProposal {
            archetype_id: Uuid::nil(),
            merge_ids: vec![Uuid::new_v4()],
            reason: MergeReason::NearDuplicate { distance: 0.003 },
            confidence: 0.95,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("NearDuplicate"));
        assert!(json.contains("0.95"));
    }

    #[test]
    fn report_serializes() {
        let report = CompressionReport {
            nodes_scanned: 500,
            duplicate_pairs: 10,
            stale_clusters: 2,
            stale_cluster_nodes: 15,
            redundant_paths: 3,
            merge_proposals: vec![],
            estimated_reduction: 25,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"nodes_scanned\":500"));
    }
}
