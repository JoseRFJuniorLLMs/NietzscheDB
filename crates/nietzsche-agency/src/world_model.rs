// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Phase XXIII — World Model Graph: meta-representation of external state.
//!
//! The World Model maintains a compact internal representation of the
//! "external world" that the knowledge graph interacts with:
//!
//! - **Query patterns**: What users ask about, how often, when
//! - **Data sources**: Where new knowledge comes from
//! - **Temporal patterns**: Daily/weekly cycles in access and mutation
//! - **Environmental signals**: Temperature, load, resource usage
//!
//! The World Model allows the agency engine to anticipate needs:
//!
//! - Pre-warm ego-caches for nodes likely to be queried
//! - Allocate attention budget based on predicted demand
//! - Schedule healing during low-traffic windows
//! - Detect anomalous access patterns (potential attacks or bugs)
//!
//! ## Design
//!
//! Lightweight in-memory model updated every tick with environmental
//! observations. Periodically produces a `WorldSnapshot` persisted to CF_META.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::config::AgencyConfig;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the World Model.
#[derive(Debug, Clone)]
pub struct WorldModelConfig {
    /// Whether the world model is enabled.
    pub enabled: bool,
    /// Tick interval between world model snapshots (default: 10).
    pub snapshot_interval: u64,
    /// Maximum history length for rolling statistics (default: 100).
    pub history_length: usize,
    /// Anomaly detection sensitivity (std devs from mean, default: 3.0).
    pub anomaly_sensitivity: f64,
}

impl Default for WorldModelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            snapshot_interval: 10,
            history_length: 100,
            anomaly_sensitivity: 3.0,
        }
    }
}

/// Build a `WorldModelConfig` from the global `AgencyConfig`.
pub fn build_world_model_config(config: &AgencyConfig) -> WorldModelConfig {
    WorldModelConfig {
        enabled: config.world_model_enabled,
        snapshot_interval: config.world_model_snapshot_interval,
        history_length: config.world_model_history_length,
        anomaly_sensitivity: config.world_model_anomaly_sensitivity,
    }
}

// ─────────────────────────────────────────────
// Environmental observation
// ─────────────────────────────────────────────

/// A single tick's environmental observation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvironmentObservation {
    /// Timestamp (Unix millis).
    pub timestamp_ms: u64,
    /// Number of queries received this tick.
    pub query_count: u64,
    /// Number of mutations this tick.
    pub mutation_count: u64,
    /// Number of new nodes created.
    pub nodes_created: u64,
    /// Number of nodes deleted/phantomized.
    pub nodes_removed: u64,
    /// Current total node count.
    pub total_nodes: u64,
    /// Agency tick duration (ms).
    pub tick_duration_ms: u64,
    /// Number of active intents.
    pub active_intents: u64,
}

// ─────────────────────────────────────────────
// World Model state
// ─────────────────────────────────────────────

/// Rolling statistics for a single metric.
#[derive(Debug, Clone, Default)]
struct RollingStat {
    values: VecDeque<f64>,
    max_len: usize,
}

impl RollingStat {
    fn new(max_len: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(max_len),
            max_len,
        }
    }

    fn push(&mut self, v: f64) {
        if self.values.len() >= self.max_len {
            self.values.pop_front();
        }
        self.values.push_back(v);
    }

    fn mean(&self) -> f64 {
        if self.values.is_empty() { return 0.0; }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    fn std_dev(&self) -> f64 {
        if self.values.len() < 2 { return 0.0; }
        let m = self.mean();
        let variance = self.values.iter()
            .map(|v| (v - m) * (v - m))
            .sum::<f64>() / (self.values.len() - 1) as f64;
        variance.sqrt()
    }

    fn last(&self) -> f64 {
        self.values.back().copied().unwrap_or(0.0)
    }

    fn trend(&self) -> f64 {
        if self.values.len() < 2 { return 0.0; }
        let n = self.values.len();
        let recent = self.values.iter().rev().take(n / 2).sum::<f64>();
        let older = self.values.iter().take(n / 2).sum::<f64>();
        let half = (n / 2).max(1) as f64;
        (recent / half) - (older / half)
    }
}

/// In-memory world model state.
#[derive(Debug)]
pub struct WorldModelState {
    query_rate: RollingStat,
    mutation_rate: RollingStat,
    creation_rate: RollingStat,
    tick_duration: RollingStat,
    total_nodes_history: RollingStat,
    observations: u64,
}

impl WorldModelState {
    pub fn new(history_length: usize) -> Self {
        Self {
            query_rate: RollingStat::new(history_length),
            mutation_rate: RollingStat::new(history_length),
            creation_rate: RollingStat::new(history_length),
            tick_duration: RollingStat::new(history_length),
            total_nodes_history: RollingStat::new(history_length),
            observations: 0,
        }
    }

    /// Ingest an environmental observation.
    pub fn observe(&mut self, obs: &EnvironmentObservation) {
        self.query_rate.push(obs.query_count as f64);
        self.mutation_rate.push(obs.mutation_count as f64);
        self.creation_rate.push(obs.nodes_created as f64);
        self.tick_duration.push(obs.tick_duration_ms as f64);
        self.total_nodes_history.push(obs.total_nodes as f64);
        self.observations += 1;
    }

    /// Generate a world snapshot with anomaly detection.
    pub fn snapshot(&self, config: &WorldModelConfig) -> WorldSnapshot {
        let mut anomalies = Vec::new();

        // Check each metric for anomalies
        let checks = [
            ("query_rate", &self.query_rate),
            ("mutation_rate", &self.mutation_rate),
            ("creation_rate", &self.creation_rate),
            ("tick_duration", &self.tick_duration),
        ];

        for (name, stat) in &checks {
            let mean = stat.mean();
            let std = stat.std_dev();
            let current = stat.last();

            if std > 0.0 && ((current - mean) / std).abs() > config.anomaly_sensitivity {
                anomalies.push(Anomaly {
                    metric: name.to_string(),
                    current,
                    mean,
                    std_dev: std,
                    z_score: (current - mean) / std,
                });
            }
        }

        WorldSnapshot {
            observations: self.observations,
            query_rate_mean: self.query_rate.mean(),
            query_rate_trend: self.query_rate.trend(),
            mutation_rate_mean: self.mutation_rate.mean(),
            mutation_rate_trend: self.mutation_rate.trend(),
            creation_rate_mean: self.creation_rate.mean(),
            tick_duration_mean: self.tick_duration.mean(),
            tick_duration_last: self.tick_duration.last(),
            total_nodes_last: self.total_nodes_history.last() as u64,
            node_growth_trend: self.total_nodes_history.trend(),
            anomalies,
            is_quiet_period: self.query_rate.last() < self.query_rate.mean() * 0.3,
            is_busy_period: self.query_rate.last() > self.query_rate.mean() * 2.0,
        }
    }
}

// ─────────────────────────────────────────────
// World snapshot
// ─────────────────────────────────────────────

/// A snapshot of the world model — persisted to CF_META.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSnapshot {
    /// Total observations ingested.
    pub observations: u64,
    /// Mean query rate (queries/tick).
    pub query_rate_mean: f64,
    /// Query rate trend (positive = increasing).
    pub query_rate_trend: f64,
    /// Mean mutation rate.
    pub mutation_rate_mean: f64,
    /// Mutation rate trend.
    pub mutation_rate_trend: f64,
    /// Mean creation rate.
    pub creation_rate_mean: f64,
    /// Mean tick duration (ms).
    pub tick_duration_mean: f64,
    /// Last tick duration (ms).
    pub tick_duration_last: f64,
    /// Last known total nodes.
    pub total_nodes_last: u64,
    /// Node growth trend.
    pub node_growth_trend: f64,
    /// Detected anomalies.
    pub anomalies: Vec<Anomaly>,
    /// Whether this appears to be a quiet/maintenance window.
    pub is_quiet_period: bool,
    /// Whether this appears to be a high-traffic period.
    pub is_busy_period: bool,
}

/// A detected anomaly in an operational metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Which metric spiked.
    pub metric: String,
    /// Current value.
    pub current: f64,
    /// Rolling mean.
    pub mean: f64,
    /// Rolling standard deviation.
    pub std_dev: f64,
    /// Z-score (how many std devs from mean).
    pub z_score: f64,
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = WorldModelConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.snapshot_interval, 10);
        assert_eq!(cfg.history_length, 100);
    }

    #[test]
    fn rolling_stat_basics() {
        let mut stat = RollingStat::new(5);
        for i in 0..10 {
            stat.push(i as f64);
        }
        // Only last 5 values retained
        assert_eq!(stat.values.len(), 5);
        assert_eq!(stat.last(), 9.0);
        assert!((stat.mean() - 7.0).abs() < 1e-6); // (5+6+7+8+9)/5 = 7
    }

    #[test]
    fn anomaly_detection() {
        let config = WorldModelConfig {
            anomaly_sensitivity: 2.0,
            ..Default::default()
        };
        let mut state = WorldModelState::new(100);

        // Feed stable data
        for _ in 0..50 {
            state.observe(&EnvironmentObservation {
                query_count: 10,
                ..Default::default()
            });
        }

        // Feed a spike
        state.observe(&EnvironmentObservation {
            query_count: 1000,
            ..Default::default()
        });

        let snap = state.snapshot(&config);
        assert!(!snap.anomalies.is_empty());
        assert_eq!(snap.anomalies[0].metric, "query_rate");
    }

    #[test]
    fn quiet_period_detection() {
        let config = WorldModelConfig::default();
        let mut state = WorldModelState::new(100);

        // Feed active data
        for _ in 0..20 {
            state.observe(&EnvironmentObservation {
                query_count: 100,
                ..Default::default()
            });
        }

        // Feed quiet data
        state.observe(&EnvironmentObservation {
            query_count: 1,
            ..Default::default()
        });

        let snap = state.snapshot(&config);
        assert!(snap.is_quiet_period);
    }

    #[test]
    fn snapshot_serializes() {
        let snap = WorldSnapshot {
            observations: 100,
            query_rate_mean: 50.0,
            query_rate_trend: 1.0,
            mutation_rate_mean: 10.0,
            mutation_rate_trend: -0.5,
            creation_rate_mean: 5.0,
            tick_duration_mean: 42.0,
            tick_duration_last: 45.0,
            total_nodes_last: 10000,
            node_growth_trend: 2.0,
            anomalies: vec![],
            is_quiet_period: false,
            is_busy_period: false,
        };
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("\"observations\":100"));
    }

    #[test]
    fn trend_calculation() {
        let mut stat = RollingStat::new(10);
        // Increasing trend
        for i in 0..10 {
            stat.push(i as f64);
        }
        assert!(stat.trend() > 0.0);

        // Decreasing trend
        let mut stat2 = RollingStat::new(10);
        for i in (0..10).rev() {
            stat2.push(i as f64);
        }
        assert!(stat2.trend() < 0.0);
    }
}
