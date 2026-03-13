// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Phase B1 — Temporal Edge Decay.
//!
//! Edge weights decay exponentially over time:
//!
//! ```text
//! w_effective(t) = w_base × e^(-λ × Δt)
//! ```
//!
//! where:
//! - `w_base`: original edge weight at creation
//! - `λ` (lambda): decay rate (higher = faster forgetting)
//! - `Δt = now - edge.created_at` (in seconds)
//!
//! ## Integration
//!
//! Called from `AgencyEngine::tick()` every `temporal_decay_interval` ticks.
//! Produces `AgencyIntent::ApplyLtd` for edges whose decayed weight drops
//! below a threshold, enabling natural edge pruning.
//!
//! ## Semantic Design
//!
//! - Recent connections are strong (high w_effective)
//! - Old, unused connections fade (low w_effective)
//! - High-energy edges decay slower (λ is modulated by importance)
//! - Prevents "fossilized" graphs where ancient edges dominate

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use tracing::info;
use uuid::Uuid;

/// Configuration for temporal decay.
#[derive(Debug, Clone)]
pub struct TemporalDecayConfig {
    /// Base decay rate λ (default: 1e-7 ≈ weight halves in ~80 days).
    pub lambda: f64,
    /// Minimum effective weight before edge is marked for pruning (default: 0.01).
    pub prune_threshold: f32,
    /// Maximum edges to scan per tick (default: 5000).
    pub max_scan: usize,
    /// Whether to actually emit prune intents (default: false — report only).
    pub enable_pruning: bool,
}

impl Default for TemporalDecayConfig {
    fn default() -> Self {
        Self {
            lambda: 1e-7, // ~80 day half-life
            prune_threshold: 0.01,
            max_scan: 5_000,
            enable_pruning: false,
        }
    }
}

/// Build config from AgencyConfig.
pub fn build_decay_config(cfg: &crate::config::AgencyConfig) -> TemporalDecayConfig {
    TemporalDecayConfig {
        lambda: cfg.temporal_decay_lambda,
        prune_threshold: cfg.temporal_decay_prune_threshold,
        max_scan: cfg.temporal_decay_max_scan,
        enable_pruning: cfg.temporal_decay_enable_pruning,
    }
}

/// Result of a temporal decay scan.
#[derive(Debug, Clone)]
pub struct TemporalDecayReport {
    /// Number of edges scanned.
    pub edges_scanned: usize,
    /// Number of edges whose effective weight changed significantly.
    pub edges_decayed: usize,
    /// Number of edges below prune threshold.
    pub edges_below_threshold: usize,
    /// Average decay factor across scanned edges.
    pub avg_decay_factor: f64,
    /// Edge updates: (edge_id, from_id, to_id, old_weight, new_effective_weight).
    pub updates: Vec<(Uuid, Uuid, Uuid, f32, f32)>,
}

/// Compute effective edge weight with exponential decay.
///
/// w_effective = w_base × e^(-λ × Δt)
#[inline]
pub fn decay_weight(base_weight: f32, lambda: f64, delta_seconds: i64) -> f32 {
    if delta_seconds <= 0 {
        return base_weight;
    }
    let factor = (-lambda * delta_seconds as f64).exp();
    (base_weight as f64 * factor) as f32
}

/// Scan edges and compute temporal decay.
///
/// Returns a report with decayed edges and optional prune suggestions.
pub fn scan_temporal_decay(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    config: &TemporalDecayConfig,
) -> TemporalDecayReport {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let all_nodes = adjacency.all_nodes();
    let mut edges_scanned = 0usize;
    let mut edges_decayed = 0usize;
    let mut edges_below = 0usize;
    let mut total_factor = 0.0f64;
    let mut updates = Vec::new();
    let mut seen_edges = std::collections::HashSet::new();

    for node_id in &all_nodes {
        if edges_scanned >= config.max_scan {
            break;
        }

        for entry in adjacency.entries_out(node_id) {
            if edges_scanned >= config.max_scan {
                break;
            }
            if seen_edges.contains(&entry.edge_id) {
                continue;
            }
            seen_edges.insert(entry.edge_id);

            if let Ok(Some(edge)) = storage.get_edge(&entry.edge_id) {
                edges_scanned += 1;
                let delta_t = now - edge.created_at;
                let factor = (-config.lambda * delta_t as f64).exp();
                total_factor += factor;

                let effective_weight = decay_weight(edge.weight, config.lambda, delta_t);

                // Only track significant decay (> 5% change)
                if (edge.weight - effective_weight).abs() > edge.weight * 0.05 {
                    edges_decayed += 1;
                    updates.push((
                        edge.id,
                        edge.from,
                        edge.to,
                        edge.weight,
                        effective_weight,
                    ));
                }

                if effective_weight < config.prune_threshold {
                    edges_below += 1;
                }
            }
        }
    }

    let avg_decay_factor = if edges_scanned > 0 {
        total_factor / edges_scanned as f64
    } else {
        1.0
    };

    TemporalDecayReport {
        edges_scanned,
        edges_decayed,
        edges_below_threshold: edges_below,
        avg_decay_factor,
        updates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_weight_no_time() {
        let w = decay_weight(1.0, 1e-7, 0);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_decay_weight_half_life() {
        // For λ = ln(2)/T, weight halves at t = T
        let half_life = 86400.0 * 80.0; // 80 days in seconds
        let lambda = (2.0_f64).ln() / half_life;
        let w = decay_weight(1.0, lambda, half_life as i64);
        assert!(
            (w - 0.5).abs() < 0.01,
            "Weight should be ~0.5 at half-life, got {w}"
        );
    }

    #[test]
    fn test_decay_weight_recent() {
        // 1 hour ago should barely decay with λ = 1e-7
        let w = decay_weight(1.0, 1e-7, 3600);
        assert!(w > 0.999, "Recent edge should barely decay, got {w}");
    }

    #[test]
    fn test_decay_weight_old() {
        // 1 year (31M seconds) with λ = 1e-7 → factor ≈ 0.045
        let w = decay_weight(1.0, 1e-7, 31_536_000);
        assert!(w < 0.1, "Year-old edge should be mostly decayed, got {w}");
    }
}
