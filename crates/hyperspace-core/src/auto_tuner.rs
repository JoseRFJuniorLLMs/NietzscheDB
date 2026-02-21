use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Minimum allowed ef_search value.
const EF_MIN: usize = 10;
/// Maximum allowed ef_search value.
const EF_MAX: usize = 1000;

/// Tracks query performance and suggests `ef_search` adjustments.
///
/// `AutoTuner` maintains a rolling window of recent query latencies and
/// result counts, then uses these to recommend increasing or decreasing
/// `ef_search` to meet latency and recall targets.
///
/// # Tuning logic
///
/// - If p95 latency exceeds `latency_target * 1.2`, decrease ef by 10% (floor `EF_MIN`).
/// - If estimated recall is below `recall_target`, increase ef by 10%.
/// - Otherwise keep the current value.
/// - Final value is always clamped to `[EF_MIN, EF_MAX]`.
pub struct AutoTuner {
    /// Rolling window of recent query latencies (microseconds).
    latencies: VecDeque<u64>,
    /// Rolling window of result counts `(k_requested, k_returned)` for recall estimation.
    result_counts: VecDeque<(usize, usize)>,
    /// Maximum window size.
    window_size: usize,
    /// Target p95 latency in microseconds.
    latency_target_us: u64,
    /// Target recall in `[0.0, 1.0]`.
    recall_target: f32,
    /// Current suggested `ef_search` (atomically readable from other threads).
    suggested_ef: AtomicUsize,
}

impl AutoTuner {
    /// Create a new `AutoTuner`.
    ///
    /// # Arguments
    ///
    /// * `latency_target_ms` - Target p95 latency in **milliseconds** (converted internally to microseconds).
    /// * `recall_target` - Target recall ratio in `[0.0, 1.0]`.
    /// * `window_size` - Number of recent queries to keep in the rolling window.
    pub fn new(latency_target_ms: u64, recall_target: f32, window_size: usize) -> Self {
        let recall_target = recall_target.clamp(0.0, 1.0);
        let window_size = window_size.max(1);
        Self {
            latencies: VecDeque::with_capacity(window_size),
            result_counts: VecDeque::with_capacity(window_size),
            window_size,
            latency_target_us: latency_target_ms.saturating_mul(1000),
            recall_target,
            suggested_ef: AtomicUsize::new(100),
        }
    }

    /// Record one query's performance metrics.
    ///
    /// # Arguments
    ///
    /// * `latency_us` - Wall-clock latency of the query in microseconds.
    /// * `k_requested` - The `top_k` value the caller asked for.
    /// * `k_returned` - The number of results actually returned.
    pub fn record(&mut self, latency_us: u64, k_requested: usize, k_returned: usize) {
        if self.latencies.len() >= self.window_size {
            self.latencies.pop_front();
        }
        if self.result_counts.len() >= self.window_size {
            self.result_counts.pop_front();
        }
        self.latencies.push_back(latency_us);
        self.result_counts.push_back((k_requested, k_returned));
    }

    /// Compute the p95 latency from the current rolling window.
    ///
    /// Returns `None` if no latencies have been recorded yet.
    pub fn p95_latency_us(&self) -> Option<u64> {
        if self.latencies.is_empty() {
            return None;
        }
        let mut sorted: Vec<u64> = self.latencies.iter().copied().collect();
        sorted.sort_unstable();
        // p95 index: for n items the 95th percentile is at index ceil(0.95 * n) - 1
        let idx = ((sorted.len() as f64 * 0.95).ceil() as usize).saturating_sub(1);
        let idx = idx.min(sorted.len() - 1);
        Some(sorted[idx])
    }

    /// Compute the estimated recall from the rolling window.
    ///
    /// Recall is estimated as `avg(k_returned / k_requested)`.
    /// Returns `None` if no results have been recorded or all requests had `k_requested == 0`.
    fn estimated_recall(&self) -> Option<f32> {
        if self.result_counts.is_empty() {
            return None;
        }
        let mut sum = 0.0f64;
        let mut count = 0u64;
        for &(requested, returned) in &self.result_counts {
            if requested > 0 {
                sum += returned as f64 / requested as f64;
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        Some((sum / count as f64) as f32)
    }

    /// Suggest an adjusted `ef_search` value based on collected metrics.
    ///
    /// The suggested value is also stored internally and can be read via
    /// [`suggested_ef_search`](Self::suggested_ef_search).
    ///
    /// # Arguments
    ///
    /// * `current_ef` - The current `ef_search` value in use.
    ///
    /// # Returns
    ///
    /// The recommended `ef_search`, clamped to `[EF_MIN, EF_MAX]`.
    pub fn suggest_ef_search(&self, current_ef: usize) -> usize {
        let p95 = self.p95_latency_us();
        let recall = self.estimated_recall();

        let new_ef = match (p95, recall) {
            // Not enough data yet -- keep current.
            (None, _) | (_, None) => current_ef,

            (Some(p95_val), Some(recall_val)) => {
                let latency_threshold = (self.latency_target_us as f64 * 1.2) as u64;

                if p95_val > latency_threshold {
                    // Latency is too high -- reduce ef by 10%.
                    let reduced = (current_ef as f64 * 0.9) as usize;
                    reduced.max(EF_MIN)
                } else if recall_val < self.recall_target {
                    // Recall is below target -- increase ef by 10%.
                    let increased = (current_ef as f64 * 1.1).ceil() as usize;
                    increased.min(EF_MAX)
                } else {
                    // Both targets met -- keep current.
                    current_ef
                }
            }
        };

        let clamped = new_ef.clamp(EF_MIN, EF_MAX);
        self.suggested_ef.store(clamped, Ordering::Relaxed);
        clamped
    }

    /// Read the last suggested `ef_search` value without recomputing.
    ///
    /// This is safe to call from any thread.
    pub fn suggested_ef_search(&self) -> usize {
        self.suggested_ef.load(Ordering::Relaxed)
    }

    /// Return the number of samples currently in the window.
    pub fn sample_count(&self) -> usize {
        self.latencies.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_valid_tuner() {
        let tuner = AutoTuner::new(50, 0.95, 100);
        assert_eq!(tuner.latency_target_us, 50_000);
        assert!((tuner.recall_target - 0.95).abs() < f32::EPSILON);
        assert_eq!(tuner.window_size, 100);
        assert_eq!(tuner.sample_count(), 0);
        assert_eq!(tuner.suggested_ef_search(), 100);
    }

    #[test]
    fn recall_target_clamped() {
        let tuner = AutoTuner::new(10, 1.5, 10);
        assert!((tuner.recall_target - 1.0).abs() < f32::EPSILON);

        let tuner2 = AutoTuner::new(10, -0.5, 10);
        assert!((tuner2.recall_target - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn window_size_at_least_one() {
        let tuner = AutoTuner::new(10, 0.9, 0);
        assert_eq!(tuner.window_size, 1);
    }

    #[test]
    fn p95_empty_returns_none() {
        let tuner = AutoTuner::new(50, 0.95, 100);
        assert!(tuner.p95_latency_us().is_none());
    }

    #[test]
    fn p95_single_sample() {
        let mut tuner = AutoTuner::new(50, 0.95, 100);
        tuner.record(1234, 10, 10);
        assert_eq!(tuner.p95_latency_us(), Some(1234));
    }

    #[test]
    fn p95_multiple_samples() {
        let mut tuner = AutoTuner::new(50, 0.95, 200);
        // Insert 100 samples: 1..=100 (in microseconds)
        for i in 1..=100 {
            tuner.record(i, 10, 10);
        }
        // p95 of 1..=100 should be 95
        let p95 = tuner.p95_latency_us().unwrap();
        assert_eq!(p95, 95);
    }

    #[test]
    fn rolling_window_evicts_old_entries() {
        let mut tuner = AutoTuner::new(50, 0.95, 5);
        for i in 1..=10 {
            tuner.record(i * 100, 10, 10);
        }
        // Window should only contain the last 5: 600, 700, 800, 900, 1000
        assert_eq!(tuner.sample_count(), 5);
        let p95 = tuner.p95_latency_us().unwrap();
        assert_eq!(p95, 1000);
    }

    #[test]
    fn estimated_recall_computation() {
        let mut tuner = AutoTuner::new(50, 0.95, 100);
        tuner.record(100, 10, 10);
        tuner.record(100, 10, 8);
        tuner.record(100, 10, 9);
        // recall = (1.0 + 0.8 + 0.9) / 3 = 0.9
        let recall = tuner.estimated_recall().unwrap();
        assert!((recall - 0.9).abs() < 0.001);
    }

    #[test]
    fn estimated_recall_ignores_zero_k_requested() {
        let mut tuner = AutoTuner::new(50, 0.95, 100);
        tuner.record(100, 0, 0);
        tuner.record(100, 10, 10);
        let recall = tuner.estimated_recall().unwrap();
        assert!((recall - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn estimated_recall_all_zero_returns_none() {
        let mut tuner = AutoTuner::new(50, 0.95, 100);
        tuner.record(100, 0, 0);
        assert!(tuner.estimated_recall().is_none());
    }

    #[test]
    fn suggest_ef_no_data_keeps_current() {
        let tuner = AutoTuner::new(50, 0.95, 100);
        assert_eq!(tuner.suggest_ef_search(200), 200);
    }

    #[test]
    fn suggest_ef_decreases_when_latency_high() {
        let mut tuner = AutoTuner::new(10, 0.95, 100);
        // target = 10ms = 10_000us, threshold = 12_000us
        // Record latencies well above threshold
        for _ in 0..20 {
            tuner.record(20_000, 10, 10); // 20ms >> 12ms threshold
        }
        let suggested = tuner.suggest_ef_search(100);
        // 100 * 0.9 = 90
        assert_eq!(suggested, 90);
    }

    #[test]
    fn suggest_ef_decreases_min_clamp() {
        let mut tuner = AutoTuner::new(1, 0.95, 100);
        // target = 1ms = 1_000us, threshold = 1_200us
        for _ in 0..20 {
            tuner.record(5_000, 10, 10);
        }
        let suggested = tuner.suggest_ef_search(EF_MIN);
        // 10 * 0.9 = 9, but clamped to EF_MIN = 10
        assert_eq!(suggested, EF_MIN);
    }

    #[test]
    fn suggest_ef_increases_when_recall_low() {
        let mut tuner = AutoTuner::new(100, 0.95, 100);
        // target = 100ms = 100_000us, threshold = 120_000us
        // Latency well under threshold, but recall is low
        for _ in 0..20 {
            tuner.record(1_000, 10, 7); // recall = 0.7 < 0.95
        }
        let suggested = tuner.suggest_ef_search(100);
        // 100 * 1.1 = 110, ceil rounds up to 111 due to f64 precision
        assert_eq!(suggested, 111);
    }

    #[test]
    fn suggest_ef_increases_max_clamp() {
        let mut tuner = AutoTuner::new(100, 0.95, 100);
        for _ in 0..20 {
            tuner.record(1_000, 10, 5); // low recall
        }
        let suggested = tuner.suggest_ef_search(EF_MAX);
        // 1000 * 1.1 = 1100, clamped to EF_MAX = 1000
        assert_eq!(suggested, EF_MAX);
    }

    #[test]
    fn suggest_ef_keeps_current_when_targets_met() {
        let mut tuner = AutoTuner::new(100, 0.90, 100);
        // Latency well under target, recall above target
        for _ in 0..20 {
            tuner.record(5_000, 10, 10); // recall = 1.0 > 0.9, 5ms < 100ms
        }
        let suggested = tuner.suggest_ef_search(150);
        assert_eq!(suggested, 150);
    }

    #[test]
    fn suggest_ef_latency_priority_over_recall() {
        // When latency is too high, we reduce ef even if recall is also low.
        // Latency check runs first.
        let mut tuner = AutoTuner::new(1, 0.95, 100);
        for _ in 0..20 {
            tuner.record(50_000, 10, 5); // high latency AND low recall
        }
        let suggested = tuner.suggest_ef_search(100);
        // Latency is over threshold => decrease: 100 * 0.9 = 90
        assert_eq!(suggested, 90);
    }

    #[test]
    fn suggested_ef_search_stored_atomically() {
        let mut tuner = AutoTuner::new(100, 0.90, 100);
        for _ in 0..20 {
            tuner.record(5_000, 10, 10);
        }
        let suggested = tuner.suggest_ef_search(200);
        assert_eq!(tuner.suggested_ef_search(), suggested);
    }

    #[test]
    fn iterative_convergence_decreasing() {
        // Simulate iterative tuning where latency is persistently high.
        let mut tuner = AutoTuner::new(5, 0.90, 50);
        let mut ef = 500usize;
        for _ in 0..50 {
            tuner.record(100_000, 10, 10); // 100ms >> 5ms target
        }
        // Apply several rounds of suggestions
        for _ in 0..50 {
            ef = tuner.suggest_ef_search(ef);
        }
        // Should have decreased substantially but not below EF_MIN
        assert!(ef >= EF_MIN);
        assert!(ef < 500);
    }

    #[test]
    fn iterative_convergence_increasing() {
        // Simulate iterative tuning where recall is persistently low.
        let mut tuner = AutoTuner::new(1000, 0.99, 50);
        let mut ef = 20usize;
        for _ in 0..50 {
            tuner.record(100, 10, 5); // low recall, low latency
        }
        for _ in 0..50 {
            ef = tuner.suggest_ef_search(ef);
        }
        // Should have increased substantially but not above EF_MAX
        assert!(ef <= EF_MAX);
        assert!(ef > 20);
    }
}
