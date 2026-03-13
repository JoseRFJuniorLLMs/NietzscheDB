// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Phase XII.5 — Hebbian Long-Term Potentiation (LTP) for NietzscheDB.
//!
//! Complements the LTD Daemon (which *weakens* edges on caregiver corrections)
//! with **Hebbian strengthening**: edges that carry repeated attention flow
//! become stronger — "nodes that fire together wire together".
//!
//! ## How it works
//!
//! 1. **Co-activation**: During ECAN, when a bid from A→B wins an auction,
//!    both A and B are "co-activated" in the same tick.
//!
//! 2. **LTP trace**: Each winning bid increments a per-edge activation counter.
//!    The counter decays exponentially between ticks.
//!
//! 3. **Weight reinforcement**: When the trace exceeds a threshold, the edge
//!    weight increases by `ltp_rate × trace`. This is bounded by `max_weight`.
//!
//! 4. **Homeostatic scaling**: To prevent runaway potentiation, total outgoing
//!    weight from any node is normalized if it exceeds `max_total_weight`.
//!
//! ## Integration
//!
//! The Hebbian module runs as part of the ECAN post-processing:
//! `ECAN auction results → hebbian traces → weight deltas → AgencyIntents`
//!
//! This creates **structural plasticity** in the knowledge graph:
//! frequently co-activated concepts form stronger connections over time.

use std::collections::HashMap;
use uuid::Uuid;

use crate::attention_economy::AttentionBid;

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

/// Configuration for Hebbian LTP.
#[derive(Debug, Clone)]
pub struct HebbianConfig {
    /// Weight increase per unit of LTP trace (default: 0.02 = 2% per activation).
    pub ltp_rate: f32,
    /// Trace decay factor per tick (default: 0.9 — traces fade in ~10 ticks).
    pub trace_decay: f32,
    /// Minimum trace to trigger potentiation (default: 0.1).
    pub trace_threshold: f32,
    /// Maximum edge weight (cap to prevent runaway, default: 5.0).
    pub max_weight: f32,
    /// Maximum total outgoing weight per node (homeostatic scaling, default: 50.0).
    pub max_total_weight: f32,
    /// Whether LTP is enabled (default: true).
    pub enabled: bool,
}

impl Default for HebbianConfig {
    fn default() -> Self {
        Self {
            ltp_rate: 0.02,
            trace_decay: 0.9,
            trace_threshold: 0.1,
            max_weight: 5.0,
            max_total_weight: 50.0,
            enabled: true,
        }
    }
}

/// Persistent Hebbian state across ticks.
///
/// Tracks activation traces per edge (source, target) → trace value.
/// Traces decay each tick and accumulate on co-activation.
pub struct HebbianState {
    /// Per-edge activation traces: (source, target) → trace.
    traces: HashMap<(Uuid, Uuid), f32>,
}

impl HebbianState {
    pub fn new() -> Self {
        Self {
            traces: HashMap::new(),
        }
    }

    /// Get trace count (for diagnostics).
    pub fn trace_count(&self) -> usize {
        self.traces.len()
    }

    /// Update traces from auction winners and compute weight deltas.
    ///
    /// Returns: Vec<(source, target, weight_delta)> for the server to apply.
    pub fn tick(
        &mut self,
        winning_bids: &[AttentionBid],
        config: &HebbianConfig,
    ) -> Vec<HebbianDelta> {
        if !config.enabled {
            return Vec::new();
        }

        // 1. Decay all existing traces
        self.traces.retain(|_, trace| {
            *trace *= config.trace_decay;
            *trace > 1e-6 // prune near-zero traces
        });

        // 2. Accumulate new activations from winning bids
        for bid in winning_bids {
            let key = (bid.source, bid.target);
            let trace = self.traces.entry(key).or_insert(0.0);
            *trace += bid.value; // stronger bids leave stronger traces
        }

        // 3. Compute weight deltas for traces above threshold
        let mut deltas = Vec::new();
        for (&(source, target), &trace) in &self.traces {
            if trace >= config.trace_threshold {
                let delta = (config.ltp_rate * trace).min(config.max_weight * 0.1);
                deltas.push(HebbianDelta {
                    source,
                    target,
                    weight_delta: delta,
                    trace,
                });
            }
        }

        // 4. Homeostatic scaling: cap total outgoing weight per node
        let mut outgoing_totals: HashMap<Uuid, f32> = HashMap::new();
        for delta in &deltas {
            *outgoing_totals.entry(delta.source).or_insert(0.0) += delta.weight_delta;
        }

        for delta in &mut deltas {
            if let Some(&total) = outgoing_totals.get(&delta.source) {
                if total > config.max_total_weight * 0.1 {
                    // Scale down proportionally
                    let scale = (config.max_total_weight * 0.1) / total;
                    delta.weight_delta *= scale;
                }
            }
        }

        deltas
    }
}

/// A Hebbian weight reinforcement delta.
#[derive(Debug, Clone)]
pub struct HebbianDelta {
    /// Source node of the edge.
    pub source: Uuid,
    /// Target node of the edge.
    pub target: Uuid,
    /// Weight increase to apply.
    pub weight_delta: f32,
    /// Current activation trace (for diagnostics).
    pub trace: f32,
}

/// Report from a Hebbian tick.
#[derive(Debug, Clone)]
pub struct HebbianReport {
    /// Number of edges with active traces.
    pub active_traces: usize,
    /// Number of edges that received potentiation.
    pub potentiated: usize,
    /// Total weight delta applied.
    pub total_delta: f32,
    /// Individual deltas for the server to apply.
    pub deltas: Vec<HebbianDelta>,
}

/// Run a Hebbian tick: update traces from winning bids, compute weight deltas.
pub fn run_hebbian_tick(
    state: &mut HebbianState,
    winning_bids: &[AttentionBid],
    config: &HebbianConfig,
) -> HebbianReport {
    let deltas = state.tick(winning_bids, config);

    let potentiated = deltas.len();
    let total_delta: f32 = deltas.iter().map(|d| d.weight_delta).sum();

    if potentiated > 0 {
        tracing::debug!(
            active_traces = state.trace_count(),
            potentiated,
            total_delta = format!("{:.4}", total_delta),
            "Hebbian LTP tick"
        );
    }

    HebbianReport {
        active_traces: state.trace_count(),
        potentiated,
        total_delta,
        deltas,
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bid(src: u128, tgt: u128, value: f32) -> AttentionBid {
        AttentionBid {
            source: Uuid::from_u128(src),
            target: Uuid::from_u128(tgt),
            value,
        }
    }

    #[test]
    fn traces_accumulate_on_repeated_activation() {
        let mut state = HebbianState::new();
        let config = HebbianConfig::default();
        let bids = vec![bid(1, 2, 0.5)];

        // Tick 1: trace = 0.5
        state.tick(&bids, &config);
        assert!(*state.traces.get(&(Uuid::from_u128(1), Uuid::from_u128(2))).unwrap() > 0.4);

        // Tick 2: trace = 0.5*0.9 + 0.5 = 0.95
        state.tick(&bids, &config);
        let trace = *state.traces.get(&(Uuid::from_u128(1), Uuid::from_u128(2))).unwrap();
        assert!(trace > 0.9, "trace should accumulate: {trace}");
    }

    #[test]
    fn traces_decay_without_activation() {
        let mut state = HebbianState::new();
        let config = HebbianConfig::default();
        let bids = vec![bid(1, 2, 1.0)];

        // Activate once
        state.tick(&bids, &config);
        let t0 = *state.traces.get(&(Uuid::from_u128(1), Uuid::from_u128(2))).unwrap();

        // No more activation — trace should decay
        state.tick(&[], &config);
        let t1 = *state.traces.get(&(Uuid::from_u128(1), Uuid::from_u128(2))).unwrap();
        assert!(t1 < t0, "trace should decay: {t1} < {t0}");
        assert!((t1 - t0 * 0.9).abs() < 0.01);
    }

    #[test]
    fn potentiation_above_threshold() {
        let mut state = HebbianState::new();
        let config = HebbianConfig {
            trace_threshold: 0.3,
            ..Default::default()
        };

        // Below threshold — no potentiation
        let bids_small = vec![bid(1, 2, 0.1)];
        let deltas = state.tick(&bids_small, &config);
        assert!(deltas.is_empty(), "below threshold, no potentiation");

        // Above threshold — potentiation
        let bids_big = vec![bid(1, 2, 0.5)];
        let deltas = state.tick(&bids_big, &config);
        assert!(!deltas.is_empty(), "above threshold, should potentiate");
        assert!(deltas[0].weight_delta > 0.0);
    }

    #[test]
    fn disabled_produces_no_deltas() {
        let mut state = HebbianState::new();
        let config = HebbianConfig {
            enabled: false,
            ..Default::default()
        };

        let bids = vec![bid(1, 2, 1.0)];
        let deltas = state.tick(&bids, &config);
        assert!(deltas.is_empty());
    }

    #[test]
    fn homeostatic_scaling_caps_total() {
        let mut state = HebbianState::new();
        let config = HebbianConfig {
            trace_threshold: 0.01,
            max_total_weight: 1.0, // very low cap to trigger scaling
            ..Default::default()
        };

        // Many strong bids from same source
        let bids: Vec<AttentionBid> = (0..20)
            .map(|i| bid(1, 100 + i, 2.0))
            .collect();

        let deltas = state.tick(&bids, &config);
        let total: f32 = deltas.iter().map(|d| d.weight_delta).sum();

        // Total should be capped by homeostatic scaling
        assert!(
            total <= config.max_total_weight * 0.1 + 0.01,
            "total {total} should be capped at {}",
            config.max_total_weight * 0.1
        );
    }

    #[test]
    fn report_aggregates_correctly() {
        let mut state = HebbianState::new();
        let config = HebbianConfig {
            trace_threshold: 0.05,
            ..Default::default()
        };

        let bids = vec![bid(1, 2, 0.5), bid(3, 4, 0.3)];
        let report = run_hebbian_tick(&mut state, &bids, &config);

        assert!(report.active_traces >= 2);
        assert_eq!(report.potentiated, report.deltas.len());
        assert!(report.total_delta > 0.0);
    }

    #[test]
    fn near_zero_traces_pruned() {
        let mut state = HebbianState::new();
        let config = HebbianConfig {
            trace_decay: 0.001, // very aggressive decay
            ..Default::default()
        };

        let bids = vec![bid(1, 2, 0.01)];
        state.tick(&bids, &config);

        // After aggressive decay (0.01 * 0.001 = 0.00001 < 1e-6), trace pruned
        state.tick(&[], &config);
        state.tick(&[], &config); // second decay to ensure below threshold
        assert!(
            state.traces.is_empty(),
            "near-zero traces should be pruned, got {} traces",
            state.traces.len()
        );
    }
}
