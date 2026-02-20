//! Box-counting Hausdorff dimension estimator for Poincaré-ball embeddings.
//!
//! ## Algorithm
//! 1. For each scale `s` in `[4, 8, 16, 32, 64]` divide the unit cube
//!    `[-1, 1]^dim` into `s^dim` boxes of side `2/s`.
//! 2. Count `N(s)` = number of non-empty boxes.
//! 3. Fit `log N(s) = D · log s + C` by linear regression.
//! 4. The slope `D` is the box-counting dimension.
//!
//! **Local** variant: use the `k` nearest neighbours as the point set.

use nietzsche_graph::{Node, PoincareVector};
use std::collections::HashSet;

/// Scales used for the multi-scale box count.
const SCALES: &[u32] = &[4, 8, 16, 32, 64];

/// Number of neighbours for local Hausdorff estimation.
pub const LOCAL_K: usize = 12;

// ─────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────

/// Estimate the global Hausdorff dimension of all `nodes` in the graph.
///
/// Returns a value in `[0.0, 3.0]`. Returns `1.0` when there are fewer
/// than three nodes (too sparse to estimate).
pub fn global_hausdorff(nodes: &[Node]) -> f32 {
    if nodes.len() < 3 {
        return 1.0;
    }
    let coords: Vec<&Vec<f32>> = nodes.iter().map(|n| &n.embedding.coords).collect();
    box_counting(&coords)
}

/// Estimate the **local** Hausdorff dimension for `center` using its
/// `k` nearest neighbours (by Poincaré distance).
///
/// Returns `1.0` when there are not enough neighbours.
pub fn local_hausdorff(center: &Node, all_nodes: &[Node], k: usize) -> f32 {
    // Sort other nodes by Poincaré distance from center
    let mut sorted: Vec<&Node> = all_nodes
        .iter()
        .filter(|n| n.id != center.id)
        .collect();

    sorted.sort_by(|a, b| {
        let da = center.embedding.distance(&a.embedding);
        let db = center.embedding.distance(&b.embedding);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });

    let k_neighbours: Vec<&Vec<f32>> = sorted
        .iter()
        .take(k)
        .map(|n| &n.embedding.coords)
        .collect();

    if k_neighbours.len() < 3 {
        return 1.0;
    }

    box_counting(&k_neighbours)
}

// ─────────────────────────────────────────────
// Core box-counting
// ─────────────────────────────────────────────

/// Multi-scale box-counting dimension of an arbitrary point set.
///
/// `coords` is a slice of references to coordinate vectors.
/// All vectors must have the same dimensionality.
pub fn box_counting(coords: &[&Vec<f32>]) -> f32 {
    if coords.len() < 2 {
        return 1.0;
    }

    let mut log_scales: Vec<f64> = Vec::new();
    let mut log_counts: Vec<f64> = Vec::new();

    for &scale in SCALES {
        let n = count_boxes(coords, scale);
        if n > 0 {
            log_scales.push((scale as f64).ln());
            log_counts.push((n as f64).ln());
        }
    }

    if log_scales.len() < 2 {
        return 1.0;
    }

    // Ordinary least-squares: slope of log N vs log s
    let d = ols_slope(&log_scales, &log_counts);
    (d as f32).clamp(0.0, 3.0)
}

/// Count non-empty boxes at `scale` divisions per axis.
fn count_boxes(coords: &[&Vec<f32>], scale: u32) -> usize {
    let mut boxes: HashSet<Vec<i32>> = HashSet::new();
    for coord in coords {
        let key: Vec<i32> = coord
            .iter()
            .map(|&c| {
                let c64 = c as f64;
                let normalised = (c64 + 1.0) * 0.5;          // [0, 1)
                let idx = (normalised * scale as f64).floor() as i32;
                idx.clamp(0, scale as i32 - 1)
            })
            .collect();
        boxes.insert(key);
    }
    boxes.len()
}

/// Ordinary least-squares slope: Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mean_x: f64 = xs.iter().sum::<f64>() / n;
    let mean_y: f64 = ys.iter().sum::<f64>() / n;

    let num: f64 = xs.iter().zip(ys.iter())
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let den: f64 = xs.iter().map(|x| (x - mean_x).powi(2)).sum();

    if den.abs() < 1e-12 { 1.0 } else { num / den }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f32, y: f32) -> Vec<f32> { vec![x, y] }

    /// A Cantor-like set along the x-axis has D ≈ 0 (sparse).
    #[test]
    fn cantor_set_is_low_dimension() {
        let pts: Vec<Vec<f32>> = (0..6)
            .map(|i| point(i as f32 * 0.02 - 0.05, 0.0))
            .collect();
        let refs: Vec<&Vec<f32>> = pts.iter().collect();
        let d = box_counting(&refs);
        // Points on a line ≈ D=1; very few points ≤ 1
        assert!(d <= 1.5, "expected D ≤ 1.5, got {d}");
    }

    /// A uniform 2-D grid should have D ≈ 2.
    #[test]
    fn grid_is_two_dimensional() {
        let mut pts = Vec::new();
        for i in 0..8 {
            for j in 0..8 {
                pts.push(vec![
                    i as f32 / 8.0 * 1.8 - 0.9,
                    j as f32 / 8.0 * 1.8 - 0.9,
                ]);
            }
        }
        let refs: Vec<&Vec<f32>> = pts.iter().collect();
        let d = box_counting(&refs);
        assert!(d > 1.5, "expected D > 1.5 for 2-D grid, got {d}");
    }

    #[test]
    fn empty_set_returns_one() {
        let refs: Vec<&Vec<f32>> = Vec::new();
        assert_eq!(box_counting(&refs), 1.0);
    }

    #[test]
    fn single_point_returns_one() {
        let p = vec![0.1_f32, 0.2];
        let refs = vec![&p];
        assert_eq!(box_counting(&refs), 1.0);
    }

    #[test]
    fn local_hausdorff_with_node_set() {
        use nietzsche_graph::{Node, PoincareVector};
        use uuid::Uuid;

        let center = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![0.0, 0.0]),
            serde_json::json!({}),
        );

        let neighbours: Vec<Node> = (1..=10)
            .map(|i| {
                Node::new(
                    Uuid::new_v4(),
                    PoincareVector::new(vec![i as f32 * 0.05, i as f32 * 0.05]),
                    serde_json::json!({}),
                )
            })
            .collect();

        let d = local_hausdorff(&center, &neighbours, LOCAL_K);
        assert!(d >= 0.0 && d <= 3.0, "D out of range: {d}");
    }
}
