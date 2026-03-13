// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Phase XXII — Hyperbolic Sharding: partition-aware locality analysis.
//!
//! In a Poincaré ball, data has natural hierarchical structure:
//!
//! - **Core** (‖x‖ < 0.3): Abstract concepts, roots of taxonomies
//! - **Middle** (0.3 ≤ ‖x‖ < 0.7): Mid-level categories
//! - **Boundary** (‖x‖ ≥ 0.7): Specific instances, leaves
//!
//! Hyperbolic Sharding analyzes this radial + angular distribution to
//! produce a `ShardMap` — a logical partition plan that respects the
//! Poincaré geometry. Each shard contains nodes that are close in
//! hyperbolic distance, minimizing cross-shard traversals.
//!
//! ## Design
//!
//! - **Radial bands**: Divide the Poincaré ball into concentric shells
//! - **Angular sectors**: Within each band, divide by angle (2D projection)
//! - **Shard assignment**: Each (band, sector) pair defines a shard
//! - **Affinity score**: How well a node fits its assigned shard
//! - **Balance metric**: How evenly nodes distribute across shards
//!
//! This is a planning/analysis module — it does NOT redistribute data.
//! Output feeds into future distributed deployment or parallel processing.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use nietzsche_graph::GraphStorage;

use crate::config::AgencyConfig;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for Hyperbolic Sharding analysis.
#[derive(Debug, Clone)]
pub struct ShardingConfig {
    /// Whether sharding analysis is enabled.
    pub enabled: bool,
    /// Tick interval between sharding scans (default: 30).
    pub interval: u64,
    /// Maximum nodes to sample for shard analysis (default: 5000).
    pub max_sample: usize,
    /// Number of radial bands (default: 4).
    pub radial_bands: usize,
    /// Number of angular sectors per band (default: 8).
    pub angular_sectors: usize,
    /// Imbalance ratio above which rebalancing is suggested (default: 3.0).
    pub imbalance_threshold: f64,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: 30,
            max_sample: 5000,
            radial_bands: 4,
            angular_sectors: 8,
            imbalance_threshold: 3.0,
        }
    }
}

/// Build a `ShardingConfig` from the global `AgencyConfig`.
pub fn build_sharding_config(config: &AgencyConfig) -> ShardingConfig {
    ShardingConfig {
        enabled: config.sharding_enabled,
        interval: config.sharding_interval,
        max_sample: config.sharding_max_sample,
        radial_bands: config.sharding_radial_bands,
        angular_sectors: config.sharding_angular_sectors,
        imbalance_threshold: config.sharding_imbalance_threshold,
    }
}

// ─────────────────────────────────────────────
// Shard map
// ─────────────────────────────────────────────

/// Logical shard identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShardId {
    /// Radial band index (0 = core, max = boundary).
    pub band: usize,
    /// Angular sector index.
    pub sector: usize,
}

impl std::fmt::Display for ShardId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "shard({},{})", self.band, self.sector)
    }
}

/// Statistics for a single shard.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardStats {
    /// Number of nodes in this shard.
    pub node_count: usize,
    /// Mean radial distance (‖x‖) of nodes in this shard.
    pub mean_radius: f64,
    /// Mean energy of nodes in this shard.
    pub mean_energy: f64,
    /// Sum of energy.
    pub total_energy: f64,
    /// Node IDs (sampled, up to 10).
    pub sample_ids: Vec<Uuid>,
}

/// Sharding analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingReport {
    /// Total nodes analyzed.
    pub nodes_analyzed: usize,
    /// Number of shards (bands × sectors).
    pub total_shards: usize,
    /// Number of non-empty shards.
    pub occupied_shards: usize,
    /// Empty shards (potential knowledge gaps).
    pub empty_shards: usize,
    /// Imbalance ratio (max_count / mean_count).
    pub imbalance_ratio: f64,
    /// Whether rebalancing is recommended.
    pub rebalance_recommended: bool,
    /// Per-shard statistics.
    pub shards: HashMap<String, ShardStats>,
    /// Radial band boundaries (norm thresholds).
    pub band_boundaries: Vec<f64>,
    /// Most populated shard.
    pub largest_shard: Option<String>,
    /// Least populated non-empty shard.
    pub smallest_shard: Option<String>,
}

// ─────────────────────────────────────────────
// Shard assignment
// ─────────────────────────────────────────────

/// Compute the shard for a given embedding.
fn assign_shard(embedding: &[f32], config: &ShardingConfig) -> ShardId {
    // Compute norm (radius in Poincaré ball)
    let norm: f64 = embedding.iter()
        .map(|c| (*c as f64) * (*c as f64))
        .sum::<f64>()
        .sqrt();

    // Radial band (uniform partition of [0, 1))
    let band = ((norm * config.radial_bands as f64) as usize)
        .min(config.radial_bands - 1);

    // Angular sector using first two coordinates
    let (x, y) = if embedding.len() >= 2 {
        (embedding[0] as f64, embedding[1] as f64)
    } else {
        (0.0, 0.0)
    };

    let angle = y.atan2(x); // [-π, π]
    let normalized = (angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI); // [0, 1)
    let sector = ((normalized * config.angular_sectors as f64) as usize)
        .min(config.angular_sectors - 1);

    ShardId { band, sector }
}

/// Compute radial band boundaries.
fn band_boundaries(bands: usize) -> Vec<f64> {
    (0..=bands).map(|i| i as f64 / bands as f64).collect()
}

// ─────────────────────────────────────────────
// Sharding scan
// ─────────────────────────────────────────────

/// Run a sharding analysis on the graph.
pub fn scan_sharding(
    storage: &GraphStorage,
    config: &ShardingConfig,
) -> ShardingReport {
    let mut shard_stats: HashMap<ShardId, ShardStats> = HashMap::new();
    let mut analyzed = 0usize;

    for result in storage.iter_nodes_meta() {
        let meta = match result {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.is_phantom { continue; }
        if analyzed >= config.max_sample { break; }
        analyzed += 1;

        // Load embedding
        let embedding = match storage.get_node(&meta.id) {
            Ok(Some(node)) => node.embedding.coords.clone(),
            _ => continue,
        };

        let shard = assign_shard(&embedding, config);
        let norm: f64 = embedding.iter()
            .map(|c| (*c as f64) * (*c as f64))
            .sum::<f64>()
            .sqrt();

        let entry = shard_stats.entry(shard).or_default();
        entry.node_count += 1;
        entry.mean_radius += norm;
        entry.mean_energy += meta.energy as f64;
        entry.total_energy += meta.energy as f64;
        if entry.sample_ids.len() < 10 {
            entry.sample_ids.push(meta.id);
        }
    }

    // Finalize means
    for stats in shard_stats.values_mut() {
        if stats.node_count > 0 {
            stats.mean_radius /= stats.node_count as f64;
            stats.mean_energy /= stats.node_count as f64;
        }
    }

    let total_shards = config.radial_bands * config.angular_sectors;
    let occupied = shard_stats.len();
    let empty = total_shards.saturating_sub(occupied);

    // Compute imbalance
    let counts: Vec<usize> = shard_stats.values().map(|s| s.node_count).collect();
    let mean_count = if counts.is_empty() {
        1.0
    } else {
        counts.iter().sum::<usize>() as f64 / counts.len() as f64
    };
    let max_count = counts.iter().copied().max().unwrap_or(0) as f64;
    let imbalance_ratio = if mean_count > 0.0 { max_count / mean_count } else { 0.0 };

    // Find largest/smallest shards
    let largest = shard_stats.iter()
        .max_by_key(|(_, s)| s.node_count)
        .map(|(id, _)| id.to_string());
    let smallest = shard_stats.iter()
        .filter(|(_, s)| s.node_count > 0)
        .min_by_key(|(_, s)| s.node_count)
        .map(|(id, _)| id.to_string());

    // Convert to string keys for JSON serialization
    let shards_map: HashMap<String, ShardStats> = shard_stats.into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

    ShardingReport {
        nodes_analyzed: analyzed,
        total_shards,
        occupied_shards: occupied,
        empty_shards: empty,
        imbalance_ratio,
        rebalance_recommended: imbalance_ratio > config.imbalance_threshold,
        shards: shards_map,
        band_boundaries: band_boundaries(config.radial_bands),
        largest_shard: largest,
        smallest_shard: smallest,
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
        let cfg = ShardingConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.radial_bands, 4);
        assert_eq!(cfg.angular_sectors, 8);
        assert_eq!(cfg.interval, 30);
    }

    #[test]
    fn shard_assignment_core() {
        let config = ShardingConfig::default();
        // Near origin → band 0
        let embedding = vec![0.01f32, 0.01, 0.0];
        let shard = assign_shard(&embedding, &config);
        assert_eq!(shard.band, 0);
    }

    #[test]
    fn shard_assignment_boundary() {
        let config = ShardingConfig::default();
        // Near boundary → high band
        let embedding = vec![0.9f32, 0.0, 0.0];
        let shard = assign_shard(&embedding, &config);
        assert!(shard.band >= 3);
    }

    #[test]
    fn shard_id_display() {
        let shard = ShardId { band: 2, sector: 5 };
        assert_eq!(format!("{}", shard), "shard(2,5)");
    }

    #[test]
    fn band_boundaries_count() {
        let bounds = band_boundaries(4);
        assert_eq!(bounds.len(), 5);
        assert!((bounds[0] - 0.0).abs() < 1e-6);
        assert!((bounds[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn report_serializes() {
        let report = ShardingReport {
            nodes_analyzed: 1000,
            total_shards: 32,
            occupied_shards: 28,
            empty_shards: 4,
            imbalance_ratio: 2.5,
            rebalance_recommended: false,
            shards: HashMap::new(),
            band_boundaries: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            largest_shard: Some("shard(3,0)".into()),
            smallest_shard: Some("shard(0,3)".into()),
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"nodes_analyzed\":1000"));
        assert!(json.contains("\"imbalance_ratio\":2.5"));
    }
}
