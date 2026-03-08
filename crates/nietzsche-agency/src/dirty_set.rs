//! Phase XV.2 — Temporal Adaptive Sampling via DirtySet.
//!
//! Instead of scanning every node on every agency tick (O(N)), the DirtySet
//! tracks which node IDs were mutated since the last scan, reducing work to
//! O(Δ) where Δ ≪ N.
//!
//! ## How it works
//!
//! 1. **Mark dirty**: After any mutation (energy change, Hebbian LTP, heat flow,
//!    ECAN attention, gravity pull, insert/delete), the node ID is added to the set.
//!
//! 2. **Drain**: At the start of an agency tick, the engine drains the dirty set.
//!    If |Δ| < `full_scan_threshold`, only those nodes are scanned.
//!    If |Δ| ≥ threshold, a full scan is triggered (too many changes = scan all).
//!
//! 3. **Stability sample**: Even when Δ is small, a random sample of
//!    `stability_sample` clean nodes is included to detect silent drift.
//!
//! ## Thread safety
//!
//! Uses `DashSet` for lock-free concurrent inserts from multiple tokio tasks
//! (e.g., gRPC handlers writing nodes while the agency tick reads).

use dashmap::DashSet;
use uuid::Uuid;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the DirtySet adaptive sampling.
#[derive(Debug, Clone)]
pub struct DirtySetConfig {
    /// If |dirty| >= this fraction of total nodes, do a full scan instead.
    /// Default: 0.3 (30% changed = just scan everything).
    pub full_scan_ratio: f64,
    /// Minimum number of stability samples from clean nodes.
    /// Prevents the system from ignoring drift in untouched regions.
    /// Default: 50.
    pub stability_sample: usize,
    /// Whether adaptive sampling is enabled.
    /// When false, always does a full scan (safe default).
    /// Default: true.
    pub enabled: bool,
}

impl Default for DirtySetConfig {
    fn default() -> Self {
        Self {
            full_scan_ratio: 0.3,
            stability_sample: 50,
            enabled: true,
        }
    }
}

// ─────────────────────────────────────────────
// DirtySet
// ─────────────────────────────────────────────

/// Concurrent set of mutated node IDs for adaptive scanning.
///
/// Thread-safe: multiple writers (gRPC handlers, agency intents) can mark
/// nodes dirty concurrently. The agency engine drains the set once per tick.
pub struct DirtySet {
    inner: DashSet<Uuid>,
    /// Number of times a full scan was forced (diagnostics).
    full_scan_count: u64,
    /// Number of adaptive scans performed (diagnostics).
    adaptive_scan_count: u64,
}

impl DirtySet {
    pub fn new() -> Self {
        Self {
            inner: DashSet::new(),
            full_scan_count: 0,
            adaptive_scan_count: 0,
        }
    }

    /// Mark a node as dirty (mutated since last scan).
    ///
    /// O(1) amortized, lock-free. Safe to call from any thread.
    #[inline]
    pub fn mark(&self, id: Uuid) {
        self.inner.insert(id);
    }

    /// Mark multiple nodes as dirty.
    #[inline]
    pub fn mark_many(&self, ids: &[Uuid]) {
        for &id in ids {
            self.inner.insert(id);
        }
    }

    /// Number of dirty nodes currently tracked.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the dirty set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Drain the dirty set, returning all dirty node IDs.
    ///
    /// After this call, the set is empty and ready for the next tick.
    pub fn drain(&self) -> Vec<Uuid> {
        let mut ids = Vec::with_capacity(self.inner.len());
        // DashSet doesn't have drain(), so we iterate + clear
        for entry in self.inner.iter() {
            ids.push(*entry.key());
        }
        self.inner.clear();
        ids
    }

    /// Decide whether to do adaptive scan or full scan.
    ///
    /// Returns `ScanDecision::Adaptive(dirty_ids)` when |dirty| is small
    /// relative to total_nodes, or `ScanDecision::FullScan` otherwise.
    pub fn decide(
        &mut self,
        total_nodes: usize,
        config: &DirtySetConfig,
    ) -> ScanDecision {
        if !config.enabled || total_nodes == 0 {
            self.inner.clear();
            self.full_scan_count += 1;
            return ScanDecision::FullScan;
        }

        let dirty_count = self.inner.len();
        let threshold = (total_nodes as f64 * config.full_scan_ratio) as usize;

        if dirty_count >= threshold || dirty_count == 0 {
            // Too many changes or nothing changed — full scan
            self.inner.clear();
            self.full_scan_count += 1;
            ScanDecision::FullScan
        } else {
            // Adaptive: drain dirty set and scan only those nodes
            let ids = self.drain();
            self.adaptive_scan_count += 1;
            tracing::debug!(
                dirty = ids.len(),
                total = total_nodes,
                ratio = format!("{:.2}%", (ids.len() as f64 / total_nodes as f64) * 100.0),
                "DirtySet: adaptive scan"
            );
            ScanDecision::Adaptive(ids)
        }
    }

    /// Diagnostic: total full scans since creation.
    pub fn full_scan_count(&self) -> u64 {
        self.full_scan_count
    }

    /// Diagnostic: total adaptive scans since creation.
    pub fn adaptive_scan_count(&self) -> u64 {
        self.adaptive_scan_count
    }
}

/// Result of `DirtySet::decide()`.
#[derive(Debug)]
pub enum ScanDecision {
    /// Scan only these node IDs (+ stability sample from the rest).
    Adaptive(Vec<Uuid>),
    /// Full scan of all nodes (too many changes or disabled).
    FullScan,
}

impl ScanDecision {
    /// Whether this is a full scan.
    pub fn is_full_scan(&self) -> bool {
        matches!(self, ScanDecision::FullScan)
    }

    /// Number of dirty nodes (0 for full scan).
    pub fn dirty_count(&self) -> usize {
        match self {
            ScanDecision::Adaptive(ids) => ids.len(),
            ScanDecision::FullScan => 0,
        }
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_and_drain() {
        let ds = DirtySet::new();
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);

        ds.mark(a);
        ds.mark(b);
        assert_eq!(ds.len(), 2);

        let drained = ds.drain();
        assert_eq!(drained.len(), 2);
        assert!(ds.is_empty());
    }

    #[test]
    fn dedup_on_double_mark() {
        let ds = DirtySet::new();
        let a = Uuid::from_u128(42);

        ds.mark(a);
        ds.mark(a); // duplicate
        assert_eq!(ds.len(), 1);
    }

    #[test]
    fn mark_many() {
        let ds = DirtySet::new();
        let ids: Vec<Uuid> = (0..10).map(Uuid::from_u128).collect();
        ds.mark_many(&ids);
        assert_eq!(ds.len(), 10);
    }

    #[test]
    fn decide_full_scan_when_disabled() {
        let mut ds = DirtySet::new();
        ds.mark(Uuid::from_u128(1));
        let config = DirtySetConfig { enabled: false, ..Default::default() };

        let decision = ds.decide(100, &config);
        assert!(decision.is_full_scan());
        assert!(ds.is_empty()); // cleared
    }

    #[test]
    fn decide_full_scan_when_empty() {
        let mut ds = DirtySet::new();
        let config = DirtySetConfig::default();

        let decision = ds.decide(100, &config);
        assert!(decision.is_full_scan());
    }

    #[test]
    fn decide_adaptive_when_few_dirty() {
        let mut ds = DirtySet::new();
        let config = DirtySetConfig {
            full_scan_ratio: 0.3,
            ..Default::default()
        };

        // 5 dirty out of 1000 total = 0.5% → adaptive
        for i in 0..5 {
            ds.mark(Uuid::from_u128(i));
        }

        let decision = ds.decide(1000, &config);
        match decision {
            ScanDecision::Adaptive(ids) => assert_eq!(ids.len(), 5),
            ScanDecision::FullScan => panic!("expected adaptive scan"),
        }
        assert!(ds.is_empty());
    }

    #[test]
    fn decide_full_scan_when_many_dirty() {
        let mut ds = DirtySet::new();
        let config = DirtySetConfig {
            full_scan_ratio: 0.3,
            ..Default::default()
        };

        // 400 dirty out of 1000 = 40% → full scan
        for i in 0..400 {
            ds.mark(Uuid::from_u128(i));
        }

        let decision = ds.decide(1000, &config);
        assert!(decision.is_full_scan());
        assert!(ds.is_empty());
    }

    #[test]
    fn decide_tracks_diagnostics() {
        let mut ds = DirtySet::new();
        let config = DirtySetConfig::default();

        // Full scan (empty)
        ds.decide(100, &config);
        assert_eq!(ds.full_scan_count(), 1);
        assert_eq!(ds.adaptive_scan_count(), 0);

        // Adaptive scan
        ds.mark(Uuid::from_u128(1));
        ds.decide(1000, &config);
        assert_eq!(ds.full_scan_count(), 1);
        assert_eq!(ds.adaptive_scan_count(), 1);
    }

    #[test]
    fn concurrent_marks() {
        use std::sync::Arc;

        let ds = Arc::new(DirtySet::new());
        let mut handles = Vec::new();

        for t in 0..4 {
            let ds = Arc::clone(&ds);
            handles.push(std::thread::spawn(move || {
                for i in 0..100 {
                    ds.mark(Uuid::from_u128(t * 1000 + i));
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(ds.len(), 400);
    }

    #[test]
    fn boundary_full_scan_ratio() {
        let mut ds = DirtySet::new();
        let config = DirtySetConfig {
            full_scan_ratio: 0.1, // 10%
            ..Default::default()
        };

        // Exactly at threshold: 10 dirty out of 100 = 10% → full scan
        for i in 0..10 {
            ds.mark(Uuid::from_u128(i));
        }
        let decision = ds.decide(100, &config);
        assert!(decision.is_full_scan());

        // Just below threshold: 9 dirty out of 100 = 9% → adaptive
        for i in 0..9 {
            ds.mark(Uuid::from_u128(i));
        }
        let decision = ds.decide(100, &config);
        assert!(!decision.is_full_scan());
    }
}
