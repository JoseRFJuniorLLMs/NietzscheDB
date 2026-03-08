//! Phase XX — Graph Learning Engine: the graph learns from its own operations.
//!
//! Tracks operational patterns across agency ticks:
//!
//! - **Access patterns**: which nodes/edges are read most frequently
//! - **Mutation patterns**: which areas of the graph change most
//! - **Query hotspots**: which subgraphs are traversed repeatedly
//! - **Decay patterns**: which nodes lose energy fastest
//! - **Growth patterns**: which sectors receive the most new nodes
//!
//! The Learning Engine aggregates these signals into `LearningInsight`s
//! that other subsystems can query to make smarter decisions:
//!
//! - Healing can prioritize high-traffic nodes
//! - ECAN can pre-allocate attention to hot zones
//! - Gravity can strengthen wells near access hotspots
//! - Shatter can proactively split nodes trending toward super-node status
//!
//! ## Design
//!
//! Append-only counters per tick → rolling window statistics → insight emission.
//! All state is in-memory with periodic CF_META snapshots.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::AgencyConfig;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the Graph Learning Engine.
#[derive(Debug, Clone)]
pub struct LearningConfig {
    /// Whether learning is enabled.
    pub enabled: bool,
    /// Tick interval between learning analysis (default: 5).
    pub interval: u64,
    /// Rolling window size in ticks for pattern detection (default: 50).
    pub window_size: usize,
    /// Maximum entries to track per category (default: 1000).
    pub max_tracked: usize,
    /// Minimum access count to qualify as a hotspot (default: 10).
    pub hotspot_threshold: u64,
    /// Growth rate (nodes/tick) above which a sector is "booming" (default: 5.0).
    pub growth_rate_threshold: f64,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: 5,
            window_size: 50,
            max_tracked: 1000,
            hotspot_threshold: 10,
            growth_rate_threshold: 5.0,
        }
    }
}

/// Build a `LearningConfig` from the global `AgencyConfig`.
pub fn build_learning_config(config: &AgencyConfig) -> LearningConfig {
    LearningConfig {
        enabled: config.learning_enabled,
        interval: config.learning_interval,
        window_size: config.learning_window_size,
        max_tracked: config.learning_max_tracked,
        hotspot_threshold: config.learning_hotspot_threshold,
        growth_rate_threshold: config.learning_growth_rate_threshold,
    }
}

// ─────────────────────────────────────────────
// Tick observation
// ─────────────────────────────────────────────

/// A single tick's operational observation — fed into the learning engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TickObservation {
    /// Nodes that were read/queried this tick.
    pub accessed_nodes: Vec<Uuid>,
    /// Nodes that were mutated this tick.
    pub mutated_nodes: Vec<Uuid>,
    /// Edges traversed during queries.
    pub traversed_edges: Vec<Uuid>,
    /// New nodes created this tick.
    pub created_nodes: Vec<Uuid>,
    /// Nodes that lost energy this tick.
    pub decayed_nodes: Vec<(Uuid, f32)>, // (id, energy_delta)
    /// Number of intents executed.
    pub intents_executed: usize,
    /// Tick number.
    pub tick: u64,
}

// ─────────────────────────────────────────────
// Learning state
// ─────────────────────────────────────────────

/// Rolling counters for pattern detection.
#[derive(Debug, Clone, Default)]
pub struct LearningState {
    /// Per-node access frequency (rolling window).
    access_counts: HashMap<Uuid, u64>,
    /// Per-node mutation frequency.
    mutation_counts: HashMap<Uuid, u64>,
    /// Per-node decay accumulator.
    decay_totals: HashMap<Uuid, f32>,
    /// Creation counts per angular sector (8 sectors).
    sector_creation: [u64; 8],
    /// Total observations ingested.
    observations: u64,
    /// Tick of last analysis.
    last_analysis_tick: u64,
}

impl LearningState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest a tick observation into rolling counters.
    pub fn ingest(&mut self, obs: &TickObservation, config: &LearningConfig) {
        for id in &obs.accessed_nodes {
            let entry = self.access_counts.entry(*id).or_insert(0);
            *entry += 1;
        }
        for id in &obs.mutated_nodes {
            let entry = self.mutation_counts.entry(*id).or_insert(0);
            *entry += 1;
        }
        for &(id, delta) in &obs.decayed_nodes {
            let entry = self.decay_totals.entry(id).or_insert(0.0);
            *entry += delta.abs();
        }
        // Track creation by sector (hash UUID into 8 sectors)
        for id in &obs.created_nodes {
            let sector = (id.as_u128() % 8) as usize;
            self.sector_creation[sector] += 1;
        }
        self.observations += 1;

        // Evict low-frequency entries when exceeding max_tracked
        if self.access_counts.len() > config.max_tracked * 2 {
            self.evict_low_frequency(config.max_tracked);
        }
    }

    /// Remove low-frequency entries to keep memory bounded.
    fn evict_low_frequency(&mut self, max: usize) {
        // Keep top-N by count
        if self.access_counts.len() > max {
            let mut entries: Vec<_> = self.access_counts.drain().collect();
            entries.sort_by(|a, b| b.1.cmp(&a.1));
            entries.truncate(max);
            self.access_counts = entries.into_iter().collect();
        }
        if self.mutation_counts.len() > max {
            let mut entries: Vec<_> = self.mutation_counts.drain().collect();
            entries.sort_by(|a, b| b.1.cmp(&a.1));
            entries.truncate(max);
            self.mutation_counts = entries.into_iter().collect();
        }
        if self.decay_totals.len() > max {
            let mut entries: Vec<_> = self.decay_totals.drain().collect();
            entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            entries.truncate(max);
            self.decay_totals = entries.into_iter().collect();
        }
    }

    /// Decay all counters by a factor (for rolling window approximation).
    pub fn decay_counters(&mut self, factor: f64) {
        for v in self.access_counts.values_mut() {
            *v = (*v as f64 * factor) as u64;
        }
        for v in self.mutation_counts.values_mut() {
            *v = (*v as f64 * factor) as u64;
        }
        for v in self.decay_totals.values_mut() {
            *v *= factor as f32;
        }
        // Decay sector creation
        for s in &mut self.sector_creation {
            *s = (*s as f64 * factor) as u64;
        }
        // Remove zeros
        self.access_counts.retain(|_, v| *v > 0);
        self.mutation_counts.retain(|_, v| *v > 0);
        self.decay_totals.retain(|_, v| *v > 0.001);
    }
}

// ─────────────────────────────────────────────
// Learning report
// ─────────────────────────────────────────────

/// Insight produced by the Learning Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningReport {
    /// Total observations ingested.
    pub total_observations: u64,
    /// Access hotspots — nodes queried/read most frequently.
    pub access_hotspots: Vec<Hotspot>,
    /// Mutation hotspots — nodes mutated most frequently.
    pub mutation_hotspots: Vec<Hotspot>,
    /// Fastest-decaying nodes.
    pub decay_hotspots: Vec<DecayEntry>,
    /// Sector growth rates (creations per observation).
    pub sector_growth: [f64; 8],
    /// Booming sectors (growth rate above threshold).
    pub booming_sectors: Vec<usize>,
    /// Number of tracked access entries.
    pub tracked_access: usize,
    /// Number of tracked mutation entries.
    pub tracked_mutations: usize,
}

/// A frequently-accessed or frequently-mutated node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hotspot {
    pub node_id: Uuid,
    pub count: u64,
    /// Frequency per observation.
    pub rate: f64,
}

/// A node experiencing rapid energy decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayEntry {
    pub node_id: Uuid,
    pub total_decay: f32,
    /// Decay per observation.
    pub rate: f32,
}

// ─────────────────────────────────────────────
// Analysis
// ─────────────────────────────────────────────

/// Run the learning analysis on accumulated state.
pub fn run_learning_analysis(
    state: &LearningState,
    config: &LearningConfig,
) -> LearningReport {
    let obs = state.observations.max(1) as f64;

    // Top access hotspots
    let mut access: Vec<_> = state.access_counts.iter()
        .filter(|(_, &count)| count >= config.hotspot_threshold)
        .map(|(&id, &count)| Hotspot {
            node_id: id,
            count,
            rate: count as f64 / obs,
        })
        .collect();
    access.sort_by(|a, b| b.count.cmp(&a.count));
    access.truncate(20);

    // Top mutation hotspots
    let mut mutations: Vec<_> = state.mutation_counts.iter()
        .filter(|(_, &count)| count >= config.hotspot_threshold)
        .map(|(&id, &count)| Hotspot {
            node_id: id,
            count,
            rate: count as f64 / obs,
        })
        .collect();
    mutations.sort_by(|a, b| b.count.cmp(&a.count));
    mutations.truncate(20);

    // Fastest-decaying nodes
    let mut decay: Vec<_> = state.decay_totals.iter()
        .map(|(&id, &total)| DecayEntry {
            node_id: id,
            total_decay: total,
            rate: total / obs as f32,
        })
        .collect();
    decay.sort_by(|a, b| b.total_decay.partial_cmp(&a.total_decay).unwrap_or(std::cmp::Ordering::Equal));
    decay.truncate(20);

    // Sector growth rates
    let mut sector_growth = [0.0f64; 8];
    let mut booming = Vec::new();
    for (i, &count) in state.sector_creation.iter().enumerate() {
        let rate = count as f64 / obs;
        sector_growth[i] = rate;
        if rate > config.growth_rate_threshold {
            booming.push(i);
        }
    }

    LearningReport {
        total_observations: state.observations,
        access_hotspots: access,
        mutation_hotspots: mutations,
        decay_hotspots: decay,
        sector_growth,
        booming_sectors: booming,
        tracked_access: state.access_counts.len(),
        tracked_mutations: state.mutation_counts.len(),
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
        let cfg = LearningConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.interval, 5);
        assert_eq!(cfg.window_size, 50);
    }

    #[test]
    fn ingest_and_analyze() {
        let config = LearningConfig {
            hotspot_threshold: 2,
            ..Default::default()
        };
        let mut state = LearningState::new();
        let node = Uuid::new_v4();

        for _ in 0..5 {
            state.ingest(&TickObservation {
                accessed_nodes: vec![node],
                ..Default::default()
            }, &config);
        }

        let report = run_learning_analysis(&state, &config);
        assert_eq!(report.total_observations, 5);
        assert!(!report.access_hotspots.is_empty());
        assert_eq!(report.access_hotspots[0].count, 5);
    }

    #[test]
    fn decay_counters() {
        let config = LearningConfig::default();
        let mut state = LearningState::new();
        let node = Uuid::new_v4();

        state.ingest(&TickObservation {
            accessed_nodes: vec![node; 10],
            ..Default::default()
        }, &config);

        assert_eq!(*state.access_counts.get(&node).unwrap(), 10);

        state.decay_counters(0.5);
        assert_eq!(*state.access_counts.get(&node).unwrap(), 5);
    }

    #[test]
    fn sector_growth_detection() {
        let config = LearningConfig {
            growth_rate_threshold: 1.0,
            ..Default::default()
        };
        let mut state = LearningState::new();

        // Create nodes that hash into sector 0
        for _ in 0..20 {
            let id = Uuid::new_v4();
            state.ingest(&TickObservation {
                created_nodes: vec![id],
                ..Default::default()
            }, &config);
        }

        let report = run_learning_analysis(&state, &config);
        // At least some sectors should have growth
        let total_growth: f64 = report.sector_growth.iter().sum();
        assert!(total_growth > 0.0);
    }

    #[test]
    fn report_serializes() {
        let report = LearningReport {
            total_observations: 100,
            access_hotspots: vec![],
            mutation_hotspots: vec![],
            decay_hotspots: vec![],
            sector_growth: [0.0; 8],
            booming_sectors: vec![],
            tracked_access: 50,
            tracked_mutations: 30,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"total_observations\":100"));
    }

    #[test]
    fn eviction_keeps_top_n() {
        let config = LearningConfig {
            max_tracked: 5,
            hotspot_threshold: 1,
            ..Default::default()
        };
        let mut state = LearningState::new();

        // Insert 15 different nodes (over 2x max_tracked)
        for _ in 0..15 {
            state.ingest(&TickObservation {
                accessed_nodes: vec![Uuid::new_v4()],
                ..Default::default()
            }, &config);
        }

        // After eviction, should have at most max_tracked entries
        assert!(state.access_counts.len() <= config.max_tracked);
    }
}
