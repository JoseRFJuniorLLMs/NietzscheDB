//! Elite Drift — Identity Defense Monitor.
//!
//! Monitors ||centroid(E_t) - centroid(E_0)|| to ensure the system
//! doesn't lose its original identity through progressive forgetting.
//!
//! If the elite centroid drifts too far from the original, the system
//! may be forgetting its core knowledge (catastrophic forgetting).

use serde::{Deserialize, Serialize};

/// Snapshot of elite centroid at a given cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EliteCentroid {
    /// Cycle number.
    pub cycle: u64,
    /// Centroid coordinates in Poincaré ball.
    pub coords: Vec<f32>,
    /// Number of elite nodes used to compute this centroid.
    pub elite_count: usize,
    /// Mean vitality of the elite cohort.
    pub mean_vitality: f32,
}

/// Elite Drift Tracker — monitors centroid deviation over time.
#[derive(Debug, Clone)]
pub struct EliteDriftTracker {
    /// The original centroid (E_0) — set during initialization.
    pub baseline: Option<EliteCentroid>,
    /// History of centroid snapshots.
    pub history: Vec<EliteCentroid>,
    /// Maximum allowed drift before alarm.
    pub max_drift: f32,
    /// Number of top nodes to consider as "elite".
    pub elite_top_n: usize,
}

impl Default for EliteDriftTracker {
    fn default() -> Self {
        Self {
            baseline: None,
            history: Vec::new(),
            max_drift: 0.3,  // 30% of Poincaré ball radius
            elite_top_n: 50,
        }
    }
}

impl EliteDriftTracker {
    pub fn new(max_drift: f32, elite_top_n: usize) -> Self {
        Self {
            max_drift,
            elite_top_n,
            ..Self::default()
        }
    }

    /// Set the baseline centroid (E_0). Called once at initialization.
    pub fn set_baseline(&mut self, centroid: EliteCentroid) {
        self.baseline = Some(centroid.clone());
        self.history.push(centroid);
    }

    /// Record a new centroid snapshot and compute drift.
    ///
    /// Returns the drift distance from baseline, or 0.0 if no baseline set.
    pub fn record_centroid(&mut self, centroid: EliteCentroid) -> f32 {
        let drift = if let Some(ref baseline) = self.baseline {
            euclidean_distance(&baseline.coords, &centroid.coords)
        } else {
            // First recording becomes baseline
            self.baseline = Some(centroid.clone());
            0.0
        };

        self.history.push(centroid);
        drift
    }

    /// Compute centroid from elite node features (vitality scores + coords).
    ///
    /// Takes a list of (vitality, poincare_coords) and selects the top N.
    pub fn compute_elite_centroid(
        &self,
        cycle: u64,
        nodes: &[(f32, Vec<f32>)], // (vitality, coords)
    ) -> EliteCentroid {
        if nodes.is_empty() {
            return EliteCentroid {
                cycle,
                coords: vec![],
                elite_count: 0,
                mean_vitality: 0.0,
            };
        }

        // Sort by vitality (descending) and take top N
        let mut sorted: Vec<&(f32, Vec<f32>)> = nodes.iter().collect();
        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let elites: Vec<&&(f32, Vec<f32>)> = sorted.iter().take(self.elite_top_n).collect();

        if elites.is_empty() {
            return EliteCentroid {
                cycle,
                coords: vec![],
                elite_count: 0,
                mean_vitality: 0.0,
            };
        }

        let dim = elites[0].1.len();
        let n = elites.len() as f32;
        let mut centroid = vec![0.0f32; dim];
        let mut vitality_sum = 0.0f32;

        for elite in &elites {
            vitality_sum += elite.0;
            for (i, coord) in elite.1.iter().enumerate() {
                if i < dim {
                    centroid[i] += coord / n;
                }
            }
        }

        EliteCentroid {
            cycle,
            coords: centroid,
            elite_count: elites.len(),
            mean_vitality: vitality_sum / n,
        }
    }

    /// Current drift from baseline.
    pub fn current_drift(&self) -> f32 {
        match (&self.baseline, self.history.last()) {
            (Some(baseline), Some(current)) => {
                euclidean_distance(&baseline.coords, &current.coords)
            }
            _ => 0.0,
        }
    }

    /// Check if drift exceeds the maximum allowed.
    pub fn is_drifting(&self) -> bool {
        self.current_drift() > self.max_drift
    }

    /// Drift velocity: rate of change in the last N cycles.
    pub fn drift_velocity(&self, window: usize) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let n = self.history.len().min(window + 1);
        let recent = &self.history[self.history.len() - n..];

        if recent.len() < 2 {
            return 0.0;
        }

        let first = &recent[0];
        let last = &recent[recent.len() - 1];
        let dist = euclidean_distance(&first.coords, &last.coords);
        let cycles = (last.cycle - first.cycle).max(1) as f32;

        dist / cycles
    }
}

/// Euclidean distance between two coordinate vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let min_len = a.len().min(b.len());
    if min_len == 0 {
        return 0.0;
    }

    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_drift_from_identical_centroids() {
        let mut tracker = EliteDriftTracker::default();
        let c1 = EliteCentroid { cycle: 0, coords: vec![0.1, 0.2], elite_count: 10, mean_vitality: 0.8 };
        let c2 = EliteCentroid { cycle: 1, coords: vec![0.1, 0.2], elite_count: 10, mean_vitality: 0.8 };

        tracker.set_baseline(c1);
        let drift = tracker.record_centroid(c2);
        assert!(drift < 1e-6, "identical centroids should have zero drift");
        assert!(!tracker.is_drifting());
    }

    #[test]
    fn drift_detection() {
        let mut tracker = EliteDriftTracker::new(0.1, 10);
        let c1 = EliteCentroid { cycle: 0, coords: vec![0.0, 0.0], elite_count: 10, mean_vitality: 0.8 };
        let c2 = EliteCentroid { cycle: 10, coords: vec![0.3, 0.4], elite_count: 10, mean_vitality: 0.7 };

        tracker.set_baseline(c1);
        let drift = tracker.record_centroid(c2);
        assert!(drift > 0.4); // sqrt(0.09 + 0.16) = 0.5
        assert!(tracker.is_drifting());
    }

    #[test]
    fn compute_elite_centroid_from_nodes() {
        let tracker = EliteDriftTracker::new(0.3, 2);
        let nodes = vec![
            (0.9, vec![0.1, 0.2]),
            (0.8, vec![0.3, 0.4]),
            (0.1, vec![0.9, 0.9]), // Low vitality, excluded from top 2
        ];

        let centroid = tracker.compute_elite_centroid(1, &nodes);
        assert_eq!(centroid.elite_count, 2);
        assert!((centroid.coords[0] - 0.2).abs() < 0.01);
        assert!((centroid.coords[1] - 0.3).abs() < 0.01);
    }

    #[test]
    fn drift_velocity() {
        let mut tracker = EliteDriftTracker::default();
        tracker.set_baseline(EliteCentroid { cycle: 0, coords: vec![0.0, 0.0], elite_count: 10, mean_vitality: 0.8 });
        tracker.record_centroid(EliteCentroid { cycle: 5, coords: vec![0.1, 0.0], elite_count: 10, mean_vitality: 0.8 });
        tracker.record_centroid(EliteCentroid { cycle: 10, coords: vec![0.2, 0.0], elite_count: 10, mean_vitality: 0.8 });

        let velocity = tracker.drift_velocity(10);
        assert!(velocity > 0.0);
    }

    #[test]
    fn empty_nodes_centroid() {
        let tracker = EliteDriftTracker::default();
        let centroid = tracker.compute_elite_centroid(1, &[]);
        assert_eq!(centroid.elite_count, 0);
        assert!(centroid.coords.is_empty());
    }
}
