//! # Minkowski spacetime operations
//!
//! The Minkowski metric defines causal structure on the NietzscheDB edge graph.
//! Every edge carries a spacetime interval ds² that classifies the relationship
//! between two events:
//!
//! | Interval | Classification | Meaning |
//! |---|---|---|
//! | ds² < 0 | **Timelike** | Event A **caused** event B |
//! | ds² > 0 | **Spacelike** | Events are causally independent |
//! | ds² ≈ 0 | **Lightlike** | Events at the causal boundary |
//!
//! ## The causal filter
//!
//! When EVA asks "Why did I conclude X?", NietzscheDB filters edges by
//! `minkowski_interval < 0.0`, instantly discarding causally unrelated events
//! and returning only the chain of events that **provably** led to X.
//!
//! ## Formula
//!
//! ```text
//! ds² = −c²·Δt² + ‖Δx‖²
//! ```
//!
//! Where:
//! - `Δt` = difference in timestamps (seconds)
//! - `‖Δx‖²` = squared Euclidean distance between embeddings in Poincaré ball
//! - `c` = causal speed constant (configurable, controls the light cone angle)
//!
//! ## Causal speed `c`
//!
//! `c` determines how "fast" causality propagates through the semantic space.
//! - **Large c**: wide light cone, most events are timelike (conservative)
//! - **Small c**: narrow light cone, strict causality (aggressive filtering)
//! - Default: `c = 1.0` (balanced — Δt and Δx contribute equally)

use serde::{Deserialize, Serialize};

/// Default causal speed constant.
///
/// With c=1.0, a 1-second time difference covers 1 unit of semantic distance.
/// Adjust based on the domain's temporal scale.
pub const DEFAULT_CAUSAL_SPEED: f64 = 1.0;

// ─────────────────────────────────────────────
// Causal classification
// ─────────────────────────────────────────────

/// Classification of a spacetime interval.
///
/// Derives from the sign of ds² = −c²·Δt² + ‖Δx‖².
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CausalType {
    /// ds² < 0: Event A caused event B (within the light cone).
    /// The temporal separation dominates the spatial separation.
    Timelike,

    /// ds² > 0: Events are causally independent (outside the light cone).
    /// The spatial separation dominates the temporal separation.
    Spacelike,

    /// ds² ≈ 0: Events at the boundary of the light cone.
    /// Causal connection at the speed limit.
    Lightlike,
}

impl Default for CausalType {
    fn default() -> Self {
        Self::Spacelike
    }
}

impl std::fmt::Display for CausalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CausalType::Timelike => write!(f, "Timelike"),
            CausalType::Spacelike => write!(f, "Spacelike"),
            CausalType::Lightlike => write!(f, "Lightlike"),
        }
    }
}

// ─────────────────────────────────────────────
// Minkowski interval computation
// ─────────────────────────────────────────────

/// Compute the Minkowski spacetime interval between two events.
///
/// ```text
/// ds² = −c²·(t_b − t_a)² + ‖emb_a − emb_b‖²
/// ```
///
/// # Arguments
///
/// - `embedding_a`, `embedding_b`: Poincaré ball coordinates (f32 storage)
/// - `timestamp_a`, `timestamp_b`: Unix timestamps (seconds, i64)
/// - `causal_speed`: the `c` constant (use [`DEFAULT_CAUSAL_SPEED`] if unsure)
///
/// # Returns
///
/// The interval ds² as f64. Negative = timelike, positive = spacelike.
#[inline]
pub fn minkowski_interval(
    embedding_a: &[f32],
    embedding_b: &[f32],
    timestamp_a: i64,
    timestamp_b: i64,
    causal_speed: f64,
) -> f64 {
    debug_assert_eq!(
        embedding_a.len(),
        embedding_b.len(),
        "dimension mismatch in minkowski_interval"
    );

    // Spatial component: ‖Δx‖² (computed in f64 for precision)
    let spatial_sq: f64 = embedding_a
        .iter()
        .zip(embedding_b.iter())
        .map(|(&a, &b)| {
            let d = (a as f64) - (b as f64);
            d * d
        })
        .sum();

    // Temporal component: c²·Δt²
    let dt = (timestamp_b - timestamp_a) as f64;
    let temporal_sq = causal_speed * causal_speed * dt * dt;

    // ds² = −c²Δt² + ‖Δx‖²
    spatial_sq - temporal_sq
}

/// Convenience: compute interval from f64 embeddings.
#[inline]
pub fn minkowski_interval_f64(
    embedding_a: &[f64],
    embedding_b: &[f64],
    timestamp_a: i64,
    timestamp_b: i64,
    causal_speed: f64,
) -> f64 {
    debug_assert_eq!(embedding_a.len(), embedding_b.len());

    let spatial_sq: f64 = embedding_a
        .iter()
        .zip(embedding_b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    let dt = (timestamp_b - timestamp_a) as f64;
    let temporal_sq = causal_speed * causal_speed * dt * dt;

    spatial_sq - temporal_sq
}

// ─────────────────────────────────────────────
// Classification
// ─────────────────────────────────────────────

/// Classify a spacetime interval.
///
/// `epsilon` controls the Lightlike tolerance zone.
/// Recommended: `1e-6` for typical usage.
#[inline]
pub fn classify(interval: f64, epsilon: f64) -> CausalType {
    if interval < -epsilon {
        CausalType::Timelike
    } else if interval > epsilon {
        CausalType::Spacelike
    } else {
        CausalType::Lightlike
    }
}

/// Check if an interval is timelike (causal).
#[inline]
pub fn is_timelike(interval: f64) -> bool {
    interval < 0.0
}

/// Check if an interval is spacelike (non-causal).
#[inline]
pub fn is_spacelike(interval: f64) -> bool {
    interval > 0.0
}

/// Check if an interval is lightlike (boundary).
#[inline]
pub fn is_lightlike(interval: f64, epsilon: f64) -> bool {
    interval.abs() <= epsilon
}

// ─────────────────────────────────────────────
// Light cone filtering
// ─────────────────────────────────────────────

/// Directions for light cone traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConeDirection {
    /// Future light cone: events caused BY the origin event (t_target > t_origin).
    Future,
    /// Past light cone: events that CAUSED the origin event (t_target < t_origin).
    Past,
    /// Both directions: full causal cone.
    Both,
}

/// Filter a list of candidate edges by their Minkowski interval.
///
/// Returns only edges with timelike intervals (ds² < 0) in the requested
/// cone direction.
///
/// # Arguments
///
/// - `origin_timestamp`: the timestamp of the origin node
/// - `candidates`: `(edge_interval, edge_target_timestamp)` pairs
/// - `direction`: which half of the light cone to include
///
/// Returns indices of candidates that pass the causal filter.
pub fn light_cone_filter(
    origin_timestamp: i64,
    candidates: &[(f64, i64)],
    direction: ConeDirection,
) -> Vec<usize> {
    candidates
        .iter()
        .enumerate()
        .filter(|(_, (interval, target_ts))| {
            // Must be timelike
            if *interval >= 0.0 {
                return false;
            }
            // Direction filter
            match direction {
                ConeDirection::Future => *target_ts > origin_timestamp,
                ConeDirection::Past => *target_ts < origin_timestamp,
                ConeDirection::Both => true,
            }
        })
        .map(|(i, _)| i)
        .collect()
}

/// Compute and classify an edge between two nodes, returning
/// `(interval, causal_type)`.
///
/// This is the function called during edge insertion to populate
/// the `minkowski_interval` and `causal_type` fields on the Edge struct.
pub fn compute_edge_causality(
    embedding_a: &[f32],
    embedding_b: &[f32],
    timestamp_a: i64,
    timestamp_b: i64,
    causal_speed: f64,
    epsilon: f64,
) -> (f64, CausalType) {
    let interval = minkowski_interval(
        embedding_a,
        embedding_b,
        timestamp_a,
        timestamp_b,
        causal_speed,
    );
    let causal_type = classify(interval, epsilon);
    (interval, causal_type)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_point_same_time_is_lightlike() {
        let emb = vec![0.3f32, 0.4];
        let interval = minkowski_interval(&emb, &emb, 1000, 1000, 1.0);
        assert!(interval.abs() < 1e-12);
        assert_eq!(classify(interval, 1e-6), CausalType::Lightlike);
    }

    #[test]
    fn large_time_diff_is_timelike() {
        let a = vec![0.1f32, 0.2];
        let b = vec![0.15f32, 0.25]; // small spatial distance
        // Large temporal difference → timelike
        let interval = minkowski_interval(&a, &b, 0, 1000, 1.0);
        assert!(interval < 0.0, "expected timelike, got ds²={interval}");
        assert_eq!(classify(interval, 1e-6), CausalType::Timelike);
    }

    #[test]
    fn large_spatial_diff_is_spacelike() {
        let a = vec![0.1f32, 0.1];
        let b = vec![0.9f32, 0.9]; // large spatial distance
        // Small temporal difference → spacelike
        let interval = minkowski_interval(&a, &b, 0, 1, 1.0);
        assert!(interval > 0.0, "expected spacelike, got ds²={interval}");
        assert_eq!(classify(interval, 1e-6), CausalType::Spacelike);
    }

    #[test]
    fn causal_speed_widens_light_cone() {
        let a = vec![0.0f32, 0.0];
        let b = vec![0.5f32, 0.5]; // moderate spatial distance
        let dt = 1;

        // With c=1.0, might be spacelike
        let interval_c1 = minkowski_interval(&a, &b, 0, dt, 1.0);
        // With c=10.0, temporal term dominates → timelike
        let interval_c10 = minkowski_interval(&a, &b, 0, dt, 10.0);

        assert!(
            interval_c10 < interval_c1,
            "larger c should make interval more negative"
        );
    }

    #[test]
    fn future_cone_filter() {
        let candidates = vec![
            (-5.0, 200_i64), // timelike, future → PASS
            (-3.0, 50),      // timelike, past → FAIL
            (2.0, 300),      // spacelike → FAIL
            (-1.0, 150),     // timelike, future → PASS
        ];
        let indices = light_cone_filter(100, &candidates, ConeDirection::Future);
        assert_eq!(indices, vec![0, 3]);
    }

    #[test]
    fn past_cone_filter() {
        let candidates = vec![
            (-5.0, 200_i64), // timelike, future → FAIL
            (-3.0, 50),      // timelike, past → PASS
            (-1.0, 80),      // timelike, past → PASS
            (2.0, 10),       // spacelike → FAIL
        ];
        let indices = light_cone_filter(100, &candidates, ConeDirection::Past);
        assert_eq!(indices, vec![1, 2]);
    }

    #[test]
    fn both_cone_filter() {
        let candidates = vec![
            (-5.0, 200_i64), // timelike → PASS
            (-3.0, 50),      // timelike → PASS
            (2.0, 300),      // spacelike → FAIL
        ];
        let indices = light_cone_filter(100, &candidates, ConeDirection::Both);
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn compute_edge_causality_works() {
        let a = vec![0.1f32, 0.2];
        let b = vec![0.12f32, 0.22];
        let (interval, causal_type) = compute_edge_causality(&a, &b, 0, 100, 1.0, 1e-6);
        assert!(interval < 0.0);
        assert_eq!(causal_type, CausalType::Timelike);
    }

    #[test]
    fn f64_matches_f32() {
        let a_f32 = vec![0.3f32, 0.4];
        let b_f32 = vec![0.5f32, 0.6];
        let a_f64 = vec![0.3f64, 0.4];
        let b_f64 = vec![0.5f64, 0.6];

        let i_f32 = minkowski_interval(&a_f32, &b_f32, 0, 10, 1.0);
        let i_f64 = minkowski_interval_f64(&a_f64, &b_f64, 0, 10, 1.0);

        assert!(
            (i_f32 - i_f64).abs() < 1e-6,
            "f32 interval {i_f32} ≠ f64 interval {i_f64}"
        );
    }
}
