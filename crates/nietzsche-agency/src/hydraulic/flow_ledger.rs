// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! FlowLedger — per-edge CPU cost accounting (the "ATP meter").
//!
//! Every edge traversal deposits a measurement: `(+1 traversal, +N cpu_ns)`.
//! The Agency tick and SleepCycle consume these statistics to decide where
//! to widen (myelinate) or narrow (atrophy) conductivity channels.
//!
//! ## Concurrency
//!
//! - `DashMap` for per-edge stats (lock-free sharded)
//! - `AtomicU64` for global counters (no mutex)
//! - `record()` is on the hot path — must be < 100ns

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

// ─────────────────────────────────────────────
// FlowStats — per-edge accumulator
// ─────────────────────────────────────────────

/// Accumulated flow statistics for a single edge.
#[derive(Debug, Clone)]
pub struct FlowStats {
    /// Total traversals since last epoch reset.
    pub traversals: u64,
    /// Cumulative CPU time (nanoseconds).
    pub total_cpu_ns: u64,
    /// Peak CPU cost observed (single worst traversal).
    pub peak_cpu_ns: u64,
    /// Exponential moving average of CPU cost (ns).
    pub ema_cpu_ns: f64,
    /// Last traversal timestamp (Unix nanos).
    pub last_traversed_ns: u64,
}

impl FlowStats {
    fn new() -> Self {
        Self {
            traversals: 0,
            total_cpu_ns: 0,
            peak_cpu_ns: 0,
            ema_cpu_ns: 0.0,
            last_traversed_ns: 0,
        }
    }

    /// Mean CPU cost per traversal. Returns 0 if no traversals.
    pub fn mean_cpu_ns(&self) -> f64 {
        if self.traversals == 0 {
            0.0
        } else {
            self.total_cpu_ns as f64 / self.traversals as f64
        }
    }
}

// ─────────────────────────────────────────────
// FlowLedger — the metabolic accounting system
// ─────────────────────────────────────────────

/// Per-edge flow accounting. Lock-free, concurrent-safe.
///
/// Every traversal deposits a measurement via `record()`.
/// The Agency tick reads via `pressure()`, `flow_rate()`, etc.
/// The SleepCycle drains via `drain_epoch()`.
pub struct FlowLedger {
    /// edge_id → FlowStats
    stats: DashMap<Uuid, FlowStats>,
    /// Global traversal counter (monotonic).
    global_traversals: AtomicU64,
    /// Epoch start time (Unix nanos).
    epoch_start_ns: AtomicU64,
    /// EMA smoothing factor (default: 0.1).
    ema_alpha: f64,
}

/// Configuration for the FlowLedger.
#[derive(Debug, Clone)]
pub struct FlowConfig {
    /// Master switch for the Hydraulic Flow Engine (default: true).
    pub enabled: bool,
    /// Agency ticks between flow analysis passes (default: 5).
    pub interval: u64,
    /// Record 1-in-N traversals (1 = all, default: 1).
    pub sample_rate: u64,
    /// EMA smoothing factor (default: 0.1).
    pub ema_alpha: f64,
    /// Flow-proportional conductivity learning rate (default: 0.01).
    pub conductivity_rate: f64,
    /// Minimum conductivity (default: 0.01).
    pub conductivity_min: f32,
    /// Maximum conductivity (default: 10.0).
    pub conductivity_max: f32,
    /// Conductivity decay rate toward baseline (default: 0.00001).
    pub decay_lambda: f64,
    /// Max new shortcuts per epoch (default: 5).
    pub max_shortcuts_per_epoch: usize,
    /// Minimum CPU savings (ns) to justify a shortcut (default: 100_000).
    pub shortcut_min_savings_ns: u64,
    /// Minimum chain length for shortcut consideration (default: 3).
    pub shortcut_min_hops: usize,
    /// Initial conductivity for new shortcuts (default: 3.0).
    pub shortcut_initial_conductivity: f32,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: 5,
            sample_rate: 1,
            ema_alpha: 0.1,
            conductivity_rate: 0.01,
            conductivity_min: 0.01,
            conductivity_max: 10.0,
            decay_lambda: 0.00001,
            max_shortcuts_per_epoch: 5,
            shortcut_min_savings_ns: 100_000,
            shortcut_min_hops: 3,
            shortcut_initial_conductivity: 3.0,
        }
    }
}

/// Snapshot of one epoch's flow data, returned by `drain_epoch()`.
#[derive(Debug)]
pub struct FlowEpochReport {
    /// Per-edge snapshots (edge_id → stats clone).
    pub edge_stats: Vec<(Uuid, FlowStats)>,
    /// Total traversals in this epoch.
    pub total_traversals: u64,
    /// Epoch duration in seconds.
    pub epoch_duration_secs: f64,
    /// Global mean CPU cost per traversal (ns).
    pub global_mean_cpu_ns: f64,
}

impl FlowLedger {
    /// Create a new FlowLedger with the given EMA alpha.
    pub fn new(ema_alpha: f64) -> Self {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self {
            stats: DashMap::new(),
            global_traversals: AtomicU64::new(0),
            epoch_start_ns: AtomicU64::new(now_ns),
            ema_alpha,
        }
    }

    /// Record a traversal on the given edge.
    ///
    /// Called on the hot path — must be fast (< 100ns typical).
    /// Uses DashMap's per-shard locking for minimal contention.
    pub fn record(&self, edge_id: Uuid, cpu_ns: u64) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        self.global_traversals.fetch_add(1, Ordering::Relaxed);

        let mut entry = self.stats.entry(edge_id).or_insert_with(FlowStats::new);
        let s = entry.value_mut();
        s.traversals += 1;
        s.total_cpu_ns += cpu_ns;
        if cpu_ns > s.peak_cpu_ns {
            s.peak_cpu_ns = cpu_ns;
        }
        // EMA update: ema = α·sample + (1-α)·ema
        s.ema_cpu_ns = self.ema_alpha * cpu_ns as f64 + (1.0 - self.ema_alpha) * s.ema_cpu_ns;
        s.last_traversed_ns = now_ns;
    }

    /// Semantic pressure of an edge: ratio of its cost to global mean.
    ///
    /// Returns values > 1.0 for bottlenecks, < 1.0 for efficient edges.
    /// Returns 1.0 if no data exists.
    pub fn pressure(&self, edge_id: Uuid) -> f64 {
        let global_mean = self.global_mean_cpu_ns();
        if global_mean < 1e-9 {
            return 1.0;
        }
        match self.stats.get(&edge_id) {
            Some(s) => s.ema_cpu_ns / global_mean,
            None => 1.0,
        }
    }

    /// Flow rate of an edge: traversals per second since epoch start.
    pub fn flow_rate(&self, edge_id: Uuid) -> f64 {
        let epoch_secs = self.epoch_duration_secs();
        if epoch_secs < 1e-9 {
            return 0.0;
        }
        match self.stats.get(&edge_id) {
            Some(s) => s.traversals as f64 / epoch_secs,
            None => 0.0,
        }
    }

    /// Global mean CPU cost per traversal (nanoseconds).
    pub fn global_mean_cpu_ns(&self) -> f64 {
        let total_t: u64 = self.global_traversals.load(Ordering::Relaxed);
        if total_t == 0 {
            return 0.0;
        }
        let total_cpu: u64 = self.stats.iter().map(|e| e.total_cpu_ns).sum();
        total_cpu as f64 / total_t as f64
    }

    /// Mean flow rate across all tracked edges (traversals/sec).
    pub fn mean_flow_rate(&self) -> f64 {
        let epoch_secs = self.epoch_duration_secs();
        let n = self.stats.len();
        if n == 0 || epoch_secs < 1e-9 {
            return 0.0;
        }
        let total: u64 = self.stats.iter().map(|e| e.traversals).sum();
        total as f64 / (n as f64 * epoch_secs)
    }

    /// Top-K edges by pressure (highest cost/traversal ratio).
    pub fn top_bottlenecks(&self, k: usize) -> Vec<(Uuid, f64)> {
        let global_mean = self.global_mean_cpu_ns();
        if global_mean < 1e-9 {
            return Vec::new();
        }
        let mut pairs: Vec<(Uuid, f64)> = self.stats.iter()
            .filter(|e| e.traversals > 0)
            .map(|e| (*e.key(), e.ema_cpu_ns / global_mean))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(k);
        pairs
    }

    /// Top-K edges by flow rate (most traversed).
    pub fn top_highways(&self, k: usize) -> Vec<(Uuid, f64)> {
        let epoch_secs = self.epoch_duration_secs();
        if epoch_secs < 1e-9 {
            return Vec::new();
        }
        let mut pairs: Vec<(Uuid, f64)> = self.stats.iter()
            .filter(|e| e.traversals > 0)
            .map(|e| (*e.key(), e.traversals as f64 / epoch_secs))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(k);
        pairs
    }

    /// Edges not traversed for at least `min_idle_secs`.
    pub fn dormant_edges(&self, min_idle_secs: u64) -> Vec<Uuid> {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let threshold_ns = min_idle_secs * 1_000_000_000;

        self.stats.iter()
            .filter(|e| {
                e.last_traversed_ns > 0
                    && now_ns.saturating_sub(e.last_traversed_ns) > threshold_ns
            })
            .map(|e| *e.key())
            .collect()
    }

    /// Drain the current epoch: snapshot all stats, reset epoch counter.
    ///
    /// Called by the SleepCycle or periodic flow analysis.
    /// Stats are NOT cleared — they accumulate across epochs.
    /// Only the epoch timer and global counter are reset.
    pub fn drain_epoch(&self) -> FlowEpochReport {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let epoch_start = self.epoch_start_ns.swap(now_ns, Ordering::Relaxed);
        let total_t = self.global_traversals.swap(0, Ordering::Relaxed);
        let epoch_secs = (now_ns.saturating_sub(epoch_start)) as f64 / 1e9;

        let edge_stats: Vec<(Uuid, FlowStats)> = self.stats.iter()
            .map(|e| (*e.key(), e.value().clone()))
            .collect();

        let total_cpu: u64 = edge_stats.iter().map(|(_, s)| s.total_cpu_ns).sum();
        let global_mean = if total_t > 0 {
            total_cpu as f64 / total_t as f64
        } else {
            0.0
        };

        // Reset per-edge counters for next epoch
        for mut entry in self.stats.iter_mut() {
            entry.traversals = 0;
            entry.total_cpu_ns = 0;
            entry.peak_cpu_ns = 0;
            // Keep ema_cpu_ns and last_traversed_ns (they carry forward)
        }

        FlowEpochReport {
            edge_stats,
            total_traversals: total_t,
            epoch_duration_secs: epoch_secs,
            global_mean_cpu_ns: global_mean,
        }
    }

    /// Constructal flow energy: E = Σ(f² / κ).
    ///
    /// Lower values = more efficient flow network.
    /// `get_conductivity` is a closure that returns edge conductivity.
    pub fn constructal_energy<F>(&self, get_conductivity: F) -> f64
    where
        F: Fn(Uuid) -> f32,
    {
        let epoch_secs = self.epoch_duration_secs();
        if epoch_secs < 1e-9 {
            return 0.0;
        }
        self.stats.iter()
            .filter(|e| e.traversals > 0)
            .map(|e| {
                let flow_rate = e.traversals as f64 / epoch_secs;
                let kappa = get_conductivity(*e.key()).max(0.01) as f64;
                (flow_rate * flow_rate) / kappa
            })
            .sum()
    }

    /// Number of tracked edges.
    pub fn tracked_edges(&self) -> usize {
        self.stats.len()
    }

    /// Get stats for a specific edge (if tracked).
    pub fn get_stats(&self, edge_id: Uuid) -> Option<FlowStats> {
        self.stats.get(&edge_id).map(|e| e.value().clone())
    }

    // ── Internal ──

    fn epoch_duration_secs(&self) -> f64 {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let start = self.epoch_start_ns.load(Ordering::Relaxed);
        (now_ns.saturating_sub(start)) as f64 / 1e9
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  NodeLedger — per-node (action-level) CPU cost accounting (P5)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// Parallel to FlowLedger (which tracks edges), NodeLedger tracks
// the CPU cost of executing CodeAsData ActionNodes. This enables
// the EpistemologyDaemon to rank mutation candidates by friction.

/// Per-node accumulated statistics for action execution cost.
#[derive(Debug, Clone)]
pub struct NodeFlowStats {
    /// Total executions since last epoch reset.
    pub executions: u64,
    /// Cumulative CPU time (nanoseconds).
    pub total_cpu_ns: u64,
    /// Peak CPU cost observed (single worst execution).
    pub peak_cpu_ns: u64,
    /// Exponential moving average of CPU cost (ns).
    pub ema_cpu_ns: f64,
    /// Last execution timestamp (Unix nanos).
    pub last_executed_ns: u64,
}

impl NodeFlowStats {
    fn new() -> Self {
        Self {
            executions: 0,
            total_cpu_ns: 0,
            peak_cpu_ns: 0,
            ema_cpu_ns: 0.0,
            last_executed_ns: 0,
        }
    }

    /// Mean CPU cost per execution.
    pub fn mean_cpu_ns(&self) -> f64 {
        if self.executions == 0 { 0.0 } else { self.total_cpu_ns as f64 / self.executions as f64 }
    }
}

/// Per-node (action-level) flow accounting. Lock-free, concurrent-safe.
///
/// Every CodeAsData action execution deposits a measurement via `record()`.
/// The EpistemologyDaemon reads via `top_friction_nodes()` to find
/// the most expensive actions to optimize.
pub struct NodeLedger {
    stats: DashMap<Uuid, NodeFlowStats>,
    global_executions: AtomicU64,
    ema_alpha: f64,
}

impl NodeLedger {
    pub fn new(ema_alpha: f64) -> Self {
        Self {
            stats: DashMap::new(),
            global_executions: AtomicU64::new(0),
            ema_alpha,
        }
    }

    /// Record an action execution.
    pub fn record(&self, node_id: Uuid, cpu_ns: u64) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        self.global_executions.fetch_add(1, Ordering::Relaxed);

        let mut entry = self.stats.entry(node_id).or_insert_with(NodeFlowStats::new);
        let s = entry.value_mut();
        s.executions += 1;
        s.total_cpu_ns += cpu_ns;
        if cpu_ns > s.peak_cpu_ns {
            s.peak_cpu_ns = cpu_ns;
        }
        s.ema_cpu_ns = self.ema_alpha * cpu_ns as f64 + (1.0 - self.ema_alpha) * s.ema_cpu_ns;
        s.last_executed_ns = now_ns;
    }

    /// Get friction (pressure) for a specific node: ratio of its cost to global mean.
    pub fn friction(&self, node_id: Uuid) -> f64 {
        let global_mean = self.global_mean_cpu_ns();
        if global_mean < 1e-9 {
            return 1.0;
        }
        match self.stats.get(&node_id) {
            Some(s) => s.ema_cpu_ns / global_mean,
            None => 1.0,
        }
    }

    /// Top-K nodes by friction (most expensive actions).
    pub fn top_friction_nodes(&self, k: usize) -> Vec<(Uuid, f64)> {
        let global_mean = self.global_mean_cpu_ns();
        if global_mean < 1e-9 {
            return Vec::new();
        }
        let mut pairs: Vec<(Uuid, f64)> = self.stats.iter()
            .filter(|e| e.executions > 0)
            .map(|e| (*e.key(), e.ema_cpu_ns / global_mean))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(k);
        pairs
    }

    /// Global mean CPU cost per execution (nanoseconds).
    pub fn global_mean_cpu_ns(&self) -> f64 {
        let total = self.global_executions.load(Ordering::Relaxed);
        if total == 0 { return 0.0; }
        let cpu: u64 = self.stats.iter().map(|e| e.total_cpu_ns).sum();
        cpu as f64 / total as f64
    }

    /// Get stats for a specific node.
    pub fn get_stats(&self, node_id: Uuid) -> Option<NodeFlowStats> {
        self.stats.get(&node_id).map(|e| e.value().clone())
    }

    /// Number of tracked nodes.
    pub fn tracked_nodes(&self) -> usize {
        self.stats.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_stats() {
        let ledger = FlowLedger::new(0.1);
        let edge = Uuid::new_v4();

        ledger.record(edge, 1000);
        ledger.record(edge, 2000);
        ledger.record(edge, 3000);

        let stats = ledger.get_stats(edge).unwrap();
        assert_eq!(stats.traversals, 3);
        assert_eq!(stats.total_cpu_ns, 6000);
        assert_eq!(stats.peak_cpu_ns, 3000);
        assert!(stats.ema_cpu_ns > 0.0);
    }

    #[test]
    fn test_pressure_no_data() {
        let ledger = FlowLedger::new(0.1);
        let edge = Uuid::new_v4();
        assert_eq!(ledger.pressure(edge), 1.0);
    }

    #[test]
    fn test_drain_epoch_resets_counters() {
        let ledger = FlowLedger::new(0.1);
        let edge = Uuid::new_v4();

        ledger.record(edge, 500);
        ledger.record(edge, 500);

        let report = ledger.drain_epoch();
        assert_eq!(report.total_traversals, 2);
        assert_eq!(report.edge_stats.len(), 1);

        // After drain, counters are reset
        let stats = ledger.get_stats(edge).unwrap();
        assert_eq!(stats.traversals, 0);
        // But EMA carries forward
        assert!(stats.ema_cpu_ns > 0.0);
    }

    #[test]
    fn test_constructal_energy() {
        let ledger = FlowLedger::new(0.1);
        let e1 = Uuid::new_v4();
        let e2 = Uuid::new_v4();

        // Record some traversals
        for _ in 0..10 {
            ledger.record(e1, 100);
        }
        for _ in 0..5 {
            ledger.record(e2, 200);
        }

        let energy = ledger.constructal_energy(|_| 1.0);
        assert!(energy > 0.0);

        // Higher conductivity → lower energy
        let energy_high_k = ledger.constructal_energy(|_| 5.0);
        assert!(energy_high_k < energy);
    }

    #[test]
    fn test_top_bottlenecks_and_highways() {
        let ledger = FlowLedger::new(0.5);
        let fast_edge = Uuid::new_v4();
        let slow_edge = Uuid::new_v4();

        // Fast edge: many cheap traversals
        for _ in 0..100 {
            ledger.record(fast_edge, 10);
        }
        // Slow edge: few expensive traversals
        for _ in 0..5 {
            ledger.record(slow_edge, 10_000);
        }

        let bottlenecks = ledger.top_bottlenecks(2);
        assert!(!bottlenecks.is_empty());
        // Slow edge should be the top bottleneck
        assert_eq!(bottlenecks[0].0, slow_edge);

        let highways = ledger.top_highways(2);
        assert!(!highways.is_empty());
        // Fast edge should be the top highway
        assert_eq!(highways[0].0, fast_edge);
    }
}
