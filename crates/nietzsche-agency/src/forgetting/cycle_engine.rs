//! Cycle Engine — Orchestrates one tick of the forgetting metabolism.
//!
//! Ties together:
//! 1. Structural metrics (Hs, Eg)
//! 2. TGC computation (intensity × quality × δH × δE)
//! 3. Telemetry output
//!
//! This is the REAL cycle runner for production data.
//! The simulator uses this engine's logic with synthetic graphs.
//!
//! ## Usage
//! ```rust,ignore
//! let mut engine = CycleEngine::new(CycleConfig::default());
//! let report = engine.run_cycle(&adj, &nodes, created, mean_quality, cycle);
//! ```

use std::collections::{HashMap, HashSet};

use super::structural_metrics::{degree_distribution, global_efficiency, structural_entropy};
use super::tgc::TgcMonitor;

/// Configuration for the cycle engine.
#[derive(Debug, Clone)]
pub struct CycleConfig {
    /// BFS sample size for global efficiency computation.
    /// Higher = more accurate but slower. 32 is good for <100k nodes.
    pub efficiency_sample_size: usize,
    /// Compute global efficiency every N cycles (expensive).
    /// Set to 1 for every cycle, 5 for every 5th cycle.
    pub efficiency_interval: usize,
    /// TGC entropy amplifier (α).
    pub tgc_alpha: f32,
    /// TGC efficiency amplifier (β).
    pub tgc_beta: f32,
}

impl Default for CycleConfig {
    fn default() -> Self {
        Self {
            efficiency_sample_size: 32,
            efficiency_interval: 1,
            tgc_alpha: 2.0,
            tgc_beta: 3.0,
        }
    }
}

/// Report from one cycle of the engine.
#[derive(Debug, Clone)]
pub struct CycleReport {
    pub cycle: usize,
    pub active_nodes: usize,
    pub structural_entropy: f32,
    pub global_efficiency: f32,
    pub tgc_raw: f32,
    pub tgc_ema: f32,
    pub phase_rupture: bool,
}

/// The Cycle Engine — production-grade metabolic tick.
#[derive(Debug)]
pub struct CycleEngine {
    pub config: CycleConfig,
    pub tgc: TgcMonitor,
    /// Cached global efficiency for non-computation cycles.
    pub cached_eg: f32,
    /// Current cycle counter.
    pub cycle: usize,
}

impl CycleEngine {
    pub fn new(config: CycleConfig) -> Self {
        let tgc = TgcMonitor::new(config.tgc_alpha, config.tgc_beta);
        Self {
            config,
            tgc,
            cached_eg: 0.0,
            cycle: 0,
        }
    }

    /// Run one metabolic cycle.
    ///
    /// # Arguments
    /// - `adj`: Current adjacency list of the graph
    /// - `node_ids`: All active node IDs
    /// - `nodes_created`: How many nodes were born this cycle
    /// - `mean_quality`: Mean vitality of newborn nodes
    ///
    /// # Returns
    /// `CycleReport` with all topological vital signs.
    pub fn run_cycle(
        &mut self,
        adj: &HashMap<usize, HashSet<usize>>,
        node_ids: &[usize],
        nodes_created: usize,
        mean_quality: f32,
    ) -> CycleReport {
        self.cycle += 1;
        let cycle = self.cycle;

        let active = node_ids.len();

        // ── Structural Entropy (every cycle, O(N)) ──
        let deg_dist = degree_distribution(adj);
        let hs = structural_entropy(&deg_dist, active);

        // ── Global Efficiency (sampled BFS, possibly cached) ──
        let eg = if cycle % self.config.efficiency_interval == 0 || cycle == 1 {
            let e = global_efficiency(
                adj,
                node_ids,
                self.config.efficiency_sample_size,
                42u64.wrapping_mul(cycle as u64),
            );
            self.cached_eg = e;
            e
        } else {
            self.cached_eg
        };

        // ── TGC ──
        let tgc_raw = self.tgc.compute(
            nodes_created,
            active,
            mean_quality,
            hs,
            eg,
        );
        let tgc_ema = self.tgc.ema();
        let phase_rupture = self.tgc.is_phase_rupture(1.5);

        CycleReport {
            cycle,
            active_nodes: active,
            structural_entropy: hs,
            global_efficiency: eg,
            tgc_raw,
            tgc_ema,
            phase_rupture,
        }
    }

    /// Get current state for persistence.
    pub fn state(&self) -> (f32, f32, f32) {
        (self.tgc.prev_hs, self.tgc.prev_eg, self.tgc.ema_tgc)
    }

    /// Restore state from persistence.
    pub fn restore_state(&mut self, prev_hs: f32, prev_eg: f32, ema_tgc: f32) {
        self.tgc.prev_hs = prev_hs;
        self.tgc.prev_eg = prev_eg;
        self.tgc.ema_tgc = ema_tgc;
        self.cached_eg = prev_eg;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ring(n: usize) -> (HashMap<usize, HashSet<usize>>, Vec<usize>) {
        let mut adj = HashMap::new();
        for i in 0..n {
            let mut neighbors = HashSet::new();
            neighbors.insert((i + 1) % n);
            neighbors.insert((i + n - 1) % n);
            adj.insert(i, neighbors);
        }
        let ids: Vec<usize> = (0..n).collect();
        (adj, ids)
    }

    /// Build a heterogeneous graph: star (hub deg=N-1) + ring (deg=2) + leaves (deg=1).
    fn make_diverse_graph() -> (HashMap<usize, HashSet<usize>>, Vec<usize>) {
        let mut adj: HashMap<usize, HashSet<usize>> = HashMap::new();
        // Hub node 0 connected to 1..20 (degree 20)
        for i in 1..=20 {
            adj.entry(0).or_default().insert(i);
            adj.entry(i).or_default().insert(0);
        }
        // Ring among 1..10 (adds degree 2 to these nodes)
        for i in 1..10 {
            adj.entry(i).or_default().insert(i + 1);
            adj.entry(i + 1).or_default().insert(i);
        }
        // Leaves 21..30 each connected to node 10 only (degree 1)
        for i in 21..=30 {
            adj.entry(10).or_default().insert(i);
            adj.entry(i).or_default().insert(10);
        }
        let ids: Vec<usize> = adj.keys().cloned().collect();
        (adj, ids)
    }

    #[test]
    fn cycle_engine_basic() {
        let config = CycleConfig::default();
        let mut engine = CycleEngine::new(config);
        let (adj, ids) = make_diverse_graph();
        let report = engine.run_cycle(&adj, &ids, 5, 0.7);

        assert_eq!(report.cycle, 1);
        assert_eq!(report.active_nodes, ids.len());
        // Diverse graph has multiple degree classes → entropy > 0
        assert!(report.structural_entropy > 0.0, "diverse graph should have Hs>0, got {}", report.structural_entropy);
        assert!(report.global_efficiency > 0.0);
    }

    #[test]
    fn cycle_engine_ring_zero_entropy() {
        // Ring = uniform degree → Shannon entropy = 0 (mathematically correct)
        let config = CycleConfig::default();
        let mut engine = CycleEngine::new(config);
        let (adj, ids) = make_ring(50);
        let report = engine.run_cycle(&adj, &ids, 0, 0.0);
        assert!(report.structural_entropy.abs() < 1e-6, "ring should have Hs=0, got {}", report.structural_entropy);
    }

    #[test]
    fn cycle_engine_no_creation() {
        let mut engine = CycleEngine::new(CycleConfig::default());
        let (adj, ids) = make_ring(50);
        let report = engine.run_cycle(&adj, &ids, 0, 0.0);
        assert_eq!(report.tgc_raw, 0.0);
    }

    #[test]
    fn cycle_engine_multiple_cycles() {
        let mut engine = CycleEngine::new(CycleConfig::default());
        let (adj, ids) = make_ring(100);

        for _ in 0..10 {
            let report = engine.run_cycle(&adj, &ids, 5, 0.6);
            assert!(report.tgc_ema >= 0.0);
        }
        assert_eq!(engine.cycle, 10);
    }

    #[test]
    fn cycle_engine_state_persistence() {
        let mut engine = CycleEngine::new(CycleConfig::default());
        let (adj, ids) = make_ring(100);
        engine.run_cycle(&adj, &ids, 10, 0.8);

        let (hs, eg, ema) = engine.state();

        let mut engine2 = CycleEngine::new(CycleConfig::default());
        engine2.restore_state(hs, eg, ema);

        assert_eq!(engine2.tgc.prev_hs, hs);
        assert_eq!(engine2.tgc.prev_eg, eg);
        assert_eq!(engine2.tgc.ema_tgc, ema);
    }
}
