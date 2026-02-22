//! Integration tests for the multi-manifold pipeline.
//!
//! These tests verify the same operations that the 6 gRPC RPC handlers execute:
//! - Synthesis (Riemann sphere)
//! - SynthesisMulti (Riemann sphere, N-ary)
//! - CausalNeighbors / CausalChain (Minkowski spacetime)
//! - KleinPath / IsOnShortestPath (Klein model)
//! - manifold_sanitize (post-query normalization)
//!
//! All tests use the same f32 → f64 → manifold → f64 → f32 pipeline that the
//! production server follows.

use nietzsche_hyp_ops::{
    exp_map_zero, poincare_distance, assert_poincare,
    klein, riemann, minkowski, manifold,
};

// ─────────────────────────────────────────────────────
// Helpers: simulate node embeddings as they exist in DB
// ─────────────────────────────────────────────────────

/// Create a Poincaré ball point at a given "depth" in a given direction.
/// depth ∈ (0, 1): how close to the boundary (0 = center, 1 = boundary).
fn make_poincare_point(dim: usize, direction_seed: u64, depth: f64) -> Vec<f64> {
    assert!(depth > 0.0 && depth < 1.0, "depth must be in (0, 1)");
    let mut v: Vec<f64> = (0..dim)
        .map(|i| {
            let angle = ((i as u64).wrapping_mul(direction_seed).wrapping_add(7)) as f64 * 0.01;
            angle.sin()
        })
        .collect();
    // Normalize direction, then scale to desired depth
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x = (*x / norm) * depth;
        }
    }
    v
}

/// Simulate f32 storage → f64 promotion (as server does).
fn f32_roundtrip(v: &[f64]) -> (Vec<f32>, Vec<f64>) {
    let f32_coords: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    let f64_coords: Vec<f64> = f32_coords.iter().map(|&x| x as f64).collect();
    (f32_coords, f64_coords)
}

// ─────────────────────────────────────────────────────
// Test: Synthesis (binary)
// ─────────────────────────────────────────────────────

#[test]
fn synthesis_produces_shallower_point() {
    let a = make_poincare_point(128, 1, 0.6);
    let b = make_poincare_point(128, 2, 0.7);

    let (_, a_f64) = f32_roundtrip(&a);
    let (_, b_f64) = f32_roundtrip(&b);

    let synthesis = riemann::synthesis(&a_f64, &b_f64).unwrap();
    let sanitized = manifold::sanitize_poincare_f64(&synthesis);

    // Verify inside ball
    assert_poincare(&sanitized).unwrap();

    // Verify more abstract (closer to center)
    let depth_a: f64 = a_f64.iter().map(|x| x * x).sum::<f64>().sqrt();
    let depth_b: f64 = b_f64.iter().map(|x| x * x).sum::<f64>().sqrt();
    let depth_s: f64 = sanitized.iter().map(|x| x * x).sum::<f64>().sqrt();

    assert!(
        depth_s < depth_a.min(depth_b),
        "synthesis depth {depth_s:.4} should be < min({depth_a:.4}, {depth_b:.4})"
    );
}

#[test]
fn synthesis_through_f32_roundtrip_is_stable() {
    // Test the full pipeline: f32 → f64 → synthesis → sanitize → f32
    let a = make_poincare_point(3072, 42, 0.5);
    let b = make_poincare_point(3072, 99, 0.6);

    let (a_f32, a_f64) = f32_roundtrip(&a);
    let (b_f32, b_f64) = f32_roundtrip(&b);

    let synthesis = riemann::synthesis(&a_f64, &b_f64).unwrap();
    let sanitized = manifold::sanitize_poincare_f64(&synthesis);

    // Convert back to f32 (as server would store)
    let result_f32: Vec<f32> = sanitized.iter().map(|&x| x as f32).collect();
    let norm: f64 = result_f32.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    assert!(norm < 1.0, "f32 result norm {norm} >= 1.0");
}

// ─────────────────────────────────────────────────────
// Test: SynthesisMulti (N-ary)
// ─────────────────────────────────────────────────────

#[test]
fn synthesis_multi_3_points() {
    let points: Vec<Vec<f64>> = (1..=3)
        .map(|i| make_poincare_point(64, i * 13, 0.5 + i as f64 * 0.1))
        .collect();

    let (_, f64s): (Vec<_>, Vec<_>) = points.iter()
        .map(|p| f32_roundtrip(p))
        .unzip();

    let refs: Vec<&[f64]> = f64s.iter().map(|v| v.as_slice()).collect();
    let synthesis = riemann::synthesis_multi(&refs).unwrap();
    let sanitized = manifold::sanitize_poincare_f64(&synthesis);

    assert_poincare(&sanitized).unwrap();

    // Should be shallower than all inputs
    let min_depth: f64 = f64s.iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f64>().sqrt())
        .fold(f64::INFINITY, f64::min);
    let synth_depth: f64 = sanitized.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(synth_depth < min_depth, "multi-synthesis should be shallower");
}

#[test]
fn synthesis_multi_single_point_returns_identity() {
    let p = make_poincare_point(64, 7, 0.5);
    let (_, p_f64) = f32_roundtrip(&p);

    let result = riemann::synthesis_multi(&[&p_f64]).unwrap();
    let error: f64 = p_f64.iter().zip(result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(error < 1e-8, "single-point synthesis should be identity, error = {error}");
}

// ─────────────────────────────────────────────────────
// Test: Minkowski causal classification
// ─────────────────────────────────────────────────────

#[test]
fn minkowski_timelike_for_nearby_nodes_with_large_dt() {
    let a_f32: Vec<f32> = vec![0.1; 128];
    let b_f32: Vec<f32> = vec![0.11; 128]; // very close spatially

    let interval = minkowski::minkowski_interval(
        &a_f32, &b_f32,
        0, 1000, // large dt
        minkowski::DEFAULT_CAUSAL_SPEED,
    );

    assert!(minkowski::is_timelike(interval), "close nodes with large Δt should be timelike");
    let ct = minkowski::classify(interval, 1e-6);
    assert_eq!(ct, minkowski::CausalType::Timelike);
}

#[test]
fn minkowski_spacelike_for_distant_nodes_with_small_dt() {
    let a_f32: Vec<f32> = vec![0.1; 128];
    let b_f32: Vec<f32> = vec![0.9; 128]; // very far spatially

    let interval = minkowski::minkowski_interval(
        &a_f32, &b_f32,
        0, 1, // small dt
        minkowski::DEFAULT_CAUSAL_SPEED,
    );

    assert!(minkowski::is_spacelike(interval), "distant nodes with small Δt should be spacelike");
}

#[test]
fn light_cone_filter_direction() {
    // Simulate edges with pre-computed intervals
    let candidates = vec![
        (-5.0, 200_i64),  // timelike, future (t > 100)
        (-3.0, 50),       // timelike, past (t < 100)
        (2.0, 300),       // spacelike → filtered out
        (-1.0, 150),      // timelike, future
        (-0.5, 30),       // timelike, past
    ];

    let future = minkowski::light_cone_filter(100, &candidates, minkowski::ConeDirection::Future);
    assert_eq!(future, vec![0, 3], "future cone should only include t > origin");

    let past = minkowski::light_cone_filter(100, &candidates, minkowski::ConeDirection::Past);
    assert_eq!(past, vec![1, 4], "past cone should only include t < origin");

    let both = minkowski::light_cone_filter(100, &candidates, minkowski::ConeDirection::Both);
    assert_eq!(both, vec![0, 1, 3, 4], "both cone should include all timelike");
}

// ─────────────────────────────────────────────────────
// Test: Klein pathfinding pipeline
// ─────────────────────────────────────────────────────

#[test]
fn klein_projection_preserves_distances() {
    let p1 = make_poincare_point(64, 1, 0.4);
    let p2 = make_poincare_point(64, 2, 0.5);

    let d_poincare = poincare_distance(&p1, &p2);

    let k1 = klein::to_klein(&p1).unwrap();
    let k2 = klein::to_klein(&p2).unwrap();
    let d_klein = klein::klein_distance(&k1, &k2);

    assert!(
        (d_poincare - d_klein).abs() < 1e-6,
        "Poincaré dist {d_poincare:.6} ≠ Klein dist {d_klein:.6}"
    );
}

#[test]
fn klein_shortest_path_check_collinear_points() {
    // Create 3 points on a line in Klein space
    let a = make_poincare_point(8, 1, 0.3);
    let b = make_poincare_point(8, 1, 0.6); // same direction, deeper

    // Project to Klein
    let k_a = klein::to_klein(&a).unwrap();
    let k_b = klein::to_klein(&b).unwrap();

    // Midpoint in Klein (straight line!) should be on shortest path
    let k_mid: Vec<f64> = k_a.iter().zip(k_b.iter())
        .map(|(a, b)| (a + b) / 2.0)
        .collect();

    assert!(
        klein::is_on_shortest_path(&k_a, &k_b, &k_mid, 1e-6),
        "midpoint should be on shortest path"
    );
}

#[test]
fn klein_off_geodesic_not_on_path() {
    let a = vec![0.1, 0.2, 0.0, 0.0];
    let b = vec![0.4, 0.5, 0.0, 0.0];

    let k_a = klein::to_klein(&a).unwrap();
    let k_b = klein::to_klein(&b).unwrap();

    // Point clearly off the geodesic
    let off = vec![0.0, 0.0, 0.3, 0.3];
    let k_off = klein::to_klein(&off).unwrap();

    assert!(
        !klein::is_on_shortest_path(&k_a, &k_b, &k_off, 1e-7),
        "off-geodesic point should NOT be on shortest path"
    );
}

// ─────────────────────────────────────────────────────
// Test: manifold_sanitize batch
// ─────────────────────────────────────────────────────

#[test]
fn manifold_sanitize_repairs_drifted_vectors() {
    let good = vec![0.3f32, 0.4, 0.1];
    let drifted = vec![0.9f32, 0.9, 0.9]; // ‖x‖ ≈ 1.56 → needs repair
    let nan = vec![f32::NAN, 0.5, 0.2];

    let input = vec![good, drifted, nan];
    let (results, report) = manifold::manifold_sanitize(&input);

    assert_eq!(report.healthy, 1, "1 vector should be healthy");
    assert_eq!(report.repaired, 1, "1 vector should be repaired");
    assert_eq!(report.reset, 1, "1 vector should be reset (NaN)");

    // All results should be inside the ball
    for v in &results {
        let norm: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        assert!(norm < 1.0, "sanitized vector norm {norm} >= 1.0");
    }
}

#[test]
fn sanitize_poincare_f64_preserves_healthy() {
    let v = vec![0.3, 0.4, 0.1];
    let sanitized = manifold::sanitize_poincare_f64(&v);
    assert_eq!(v, sanitized, "healthy vector should be unchanged");
}

#[test]
fn sanitize_poincare_f64_clamps_outside() {
    let v = vec![0.9, 0.9]; // ‖x‖ ≈ 1.27
    let sanitized = manifold::sanitize_poincare_f64(&v);
    let norm: f64 = sanitized.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(norm < 1.0, "sanitized norm {norm} should be < 1.0");
}

// ─────────────────────────────────────────────────────
// Test: Full pipeline (simulating a complete RPC flow)
// ─────────────────────────────────────────────────────

#[test]
fn full_synthesis_pipeline_3072d() {
    // Simulate what the Synthesis RPC does with production dimensions
    let dim = 3072;

    // Create two nodes at different depths (abstract vs specific)
    let abstract_concept = make_poincare_point(dim, 1, 0.3); // near center = abstract
    let specific_concept = make_poincare_point(dim, 2, 0.7); // near boundary = specific

    // f32 storage roundtrip (as in production)
    let (_, a_f64) = f32_roundtrip(&abstract_concept);
    let (_, b_f64) = f32_roundtrip(&specific_concept);

    // Riemann synthesis
    let synthesis = riemann::synthesis(&a_f64, &b_f64).unwrap();

    // Post-query sanitize
    let sanitized = manifold::sanitize_poincare_f64(&synthesis);

    // Verify invariants
    assert_poincare(&sanitized).unwrap();
    let depth: f64 = sanitized.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(depth < 0.3, "synthesis of abstract(0.3) + specific(0.7) should be very abstract, got {depth:.4}");

    // Compute distance from synthesis to both parents
    let d_to_abstract = poincare_distance(&sanitized, &a_f64);
    let d_to_specific = poincare_distance(&sanitized, &b_f64);
    assert!(d_to_abstract > 0.0 && d_to_specific > 0.0, "distances should be positive");
}

#[test]
fn full_causal_pipeline() {
    // Simulate CausalNeighbors/CausalChain edge checking
    let dim = 128;
    let node_a = make_poincare_point(dim, 1, 0.5);
    let node_b = make_poincare_point(dim, 1, 0.51); // very close to A spatially

    let (a_f32, _) = f32_roundtrip(&node_a);
    let (b_f32, _) = f32_roundtrip(&node_b);

    // Scenario 1: Large time difference → timelike (causal)
    let (interval, ct) = minkowski::compute_edge_causality(
        &a_f32, &b_f32,
        1000000, 1001000, // 1000 seconds apart
        minkowski::DEFAULT_CAUSAL_SPEED,
        1e-6,
    );
    assert_eq!(ct, minkowski::CausalType::Timelike);
    assert!(interval < 0.0, "should be timelike (negative interval)");

    // Scenario 2: Small time difference → spacelike (not causal)
    let (interval2, ct2) = minkowski::compute_edge_causality(
        &a_f32, &b_f32,
        1000000, 1000000, // same time
        minkowski::DEFAULT_CAUSAL_SPEED,
        1e-6,
    );
    // Same position, same time → lightlike
    assert!(
        matches!(ct2, minkowski::CausalType::Lightlike | minkowski::CausalType::Spacelike),
        "same-time nearby nodes should be lightlike or spacelike"
    );
}
