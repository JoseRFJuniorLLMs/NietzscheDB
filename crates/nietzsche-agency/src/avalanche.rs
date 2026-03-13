// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Avalanche size tracking for Self-Organized Criticality (SOC) monitoring.
//!
//! In SOC theory, an **avalanche** is a cascade of mutations triggered by a
//! single event. In our agency engine, each tick produces a set of `AgencyIntent`
//! mutations — the count of intents per tick is the avalanche size **S**.
//!
//! Over many ticks the distribution P(S) should follow a power law P(S) ~ S^{-τ}
//! if the system is at criticality (τ ≈ 1.5 for mean-field universality class).
//!
//! This module is **observation-only** — it records sizes but never alters behavior.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Default capacity of the rolling window (number of ticks to remember).
pub const DEFAULT_WINDOW: usize = 1000;

/// Serializable snapshot of avalanche statistics for HTTP API / CF_META persistence.
///
/// Includes power-law analysis results computed at snapshot time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvalancheSnapshot {
    pub total_ticks: u64,
    pub avalanche_sizes_histogram: HashMap<u32, u32>,
    pub mean_size: f64,
    pub max_size: u32,
    pub estimated_tau: f64,
    pub is_power_law: bool,
    pub ks_distance: f64,
}

impl AvalancheSnapshot {
    /// Create a snapshot from live `AvalancheStats`, running power-law analysis.
    pub fn from_stats(stats: &AvalancheStats) -> Self {
        let sizes = stats.sizes_chronological();
        let pl = crate::powerlaw::is_power_law(&sizes);

        Self {
            total_ticks: stats.total_ticks,
            avalanche_sizes_histogram: stats.distribution(),
            mean_size: stats.mean_size(),
            max_size: stats.max_size,
            estimated_tau: pl.tau,
            is_power_law: pl.is_significant,
            ks_distance: pl.ks_distance,
        }
    }

    /// CF_META key for persistence.
    pub fn meta_key() -> &'static str {
        "agency:avalanche_stats"
    }
}

/// Lightweight circular buffer that tracks avalanche sizes per tick.
///
/// All operations are O(1) except `distribution()` which is O(window).
#[derive(Debug, Clone)]
pub struct AvalancheStats {
    /// Fixed-size circular buffer of avalanche sizes.
    sizes: Vec<u32>,
    /// Write cursor into `sizes` (next slot to overwrite).
    cursor: usize,
    /// How many entries have been written (capped at capacity for stats).
    count: usize,
    /// Total ticks observed (never wraps — monotonic counter).
    pub total_ticks: u64,
    /// Largest avalanche size ever observed.
    pub max_size: u32,
    /// Running sum of all sizes in the buffer (for O(1) mean computation).
    running_sum: u64,
}

impl AvalancheStats {
    /// Create a new tracker with the given window capacity.
    pub fn new(capacity: usize) -> Self {
        let cap = if capacity == 0 { DEFAULT_WINDOW } else { capacity };
        Self {
            sizes: vec![0; cap],
            cursor: 0,
            count: 0,
            total_ticks: 0,
            max_size: 0,
            running_sum: 0,
        }
    }

    /// Record a single avalanche (one tick's worth of intents).
    ///
    /// O(1) — overwrites the oldest entry in the circular buffer.
    #[inline]
    pub fn record(&mut self, size: u32) {
        let cap = self.sizes.len();

        // Subtract the value we are about to overwrite from the running sum.
        if self.count >= cap {
            self.running_sum -= self.sizes[self.cursor] as u64;
        }

        self.sizes[self.cursor] = size;
        self.running_sum += size as u64;
        self.cursor = (self.cursor + 1) % cap;
        self.total_ticks += 1;

        if self.count < cap {
            self.count += 1;
        }

        if size > self.max_size {
            self.max_size = size;
        }
    }

    /// Mean avalanche size over the rolling window.
    ///
    /// Returns 0.0 if no ticks have been recorded yet.
    #[inline]
    pub fn mean_size(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.running_sum as f64 / self.count as f64
        }
    }

    /// Number of entries currently stored (min of total_ticks and capacity).
    #[inline]
    pub fn window_len(&self) -> usize {
        self.count
    }

    /// Capacity of the rolling window.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.sizes.len()
    }

    /// Build a histogram: avalanche size S -> count of ticks with that size.
    ///
    /// O(window) — iterates over the buffer once.
    pub fn distribution(&self) -> HashMap<u32, u32> {
        let mut hist: HashMap<u32, u32> = HashMap::new();
        for &s in self.sizes.iter().take(self.count) {
            *hist.entry(s).or_insert(0) += 1;
        }
        hist
    }

    /// Return a snapshot of the raw sizes in chronological order (oldest first).
    ///
    /// Useful for external analysis. O(window).
    pub fn sizes_chronological(&self) -> Vec<u32> {
        if self.count < self.sizes.len() {
            // Buffer not yet full — data starts at 0
            self.sizes[..self.count].to_vec()
        } else {
            // Buffer full — oldest entry is at cursor
            let mut out = Vec::with_capacity(self.count);
            out.extend_from_slice(&self.sizes[self.cursor..]);
            out.extend_from_slice(&self.sizes[..self.cursor]);
            out
        }
    }
}

impl Default for AvalancheStats {
    fn default() -> Self {
        Self::new(DEFAULT_WINDOW)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stats() {
        let stats = AvalancheStats::default();
        assert_eq!(stats.total_ticks, 0);
        assert_eq!(stats.max_size, 0);
        assert!((stats.mean_size() - 0.0).abs() < f64::EPSILON);
        assert!(stats.distribution().is_empty());
        assert!(stats.sizes_chronological().is_empty());
    }

    #[test]
    fn single_record() {
        let mut stats = AvalancheStats::new(5);
        stats.record(7);
        assert_eq!(stats.total_ticks, 1);
        assert_eq!(stats.max_size, 7);
        assert!((stats.mean_size() - 7.0).abs() < f64::EPSILON);
        assert_eq!(stats.window_len(), 1);

        let dist = stats.distribution();
        assert_eq!(dist.get(&7), Some(&1));
    }

    #[test]
    fn rolling_window_overwrites() {
        let mut stats = AvalancheStats::new(3);
        stats.record(1);
        stats.record(2);
        stats.record(3);
        // Buffer full: [1, 2, 3]
        assert_eq!(stats.window_len(), 3);
        assert!((stats.mean_size() - 2.0).abs() < f64::EPSILON);

        // Overwrite oldest (1)
        stats.record(10);
        // Buffer: [10, 2, 3] (cursor-wise: [2, 3, 10])
        assert_eq!(stats.total_ticks, 4);
        assert_eq!(stats.max_size, 10);
        // Mean: (2 + 3 + 10) / 3 = 5.0
        assert!((stats.mean_size() - 5.0).abs() < f64::EPSILON);

        let chrono = stats.sizes_chronological();
        assert_eq!(chrono, vec![2, 3, 10]);
    }

    #[test]
    fn distribution_counts() {
        let mut stats = AvalancheStats::new(10);
        stats.record(0);
        stats.record(3);
        stats.record(3);
        stats.record(5);
        stats.record(3);

        let dist = stats.distribution();
        assert_eq!(dist.get(&0), Some(&1));
        assert_eq!(dist.get(&3), Some(&3));
        assert_eq!(dist.get(&5), Some(&1));
        assert_eq!(dist.len(), 3);
    }

    #[test]
    fn max_tracks_globally() {
        let mut stats = AvalancheStats::new(2);
        stats.record(100);
        stats.record(1);
        stats.record(2); // overwrites 100 in buffer
        // max_size should still be 100 (global, not windowed)
        assert_eq!(stats.max_size, 100);
    }

    #[test]
    fn zero_capacity_defaults() {
        let stats = AvalancheStats::new(0);
        assert_eq!(stats.capacity(), DEFAULT_WINDOW);
    }
}
