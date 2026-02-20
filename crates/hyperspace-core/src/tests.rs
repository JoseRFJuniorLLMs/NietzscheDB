use super::*;

#[test]
fn test_euclidean_distance() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    // diffs: -3, -3, -3. sq: 9, 9, 9. sum: 27.
    let dist = EuclideanMetric::distance(&a, &b);
    assert!((dist - 27.0).abs() < f64::EPSILON);
}

#[test]
fn test_cosine_distance() {
    // Vectors must be normalized for CosineMetric usually, but logic is sum((a-b)^2)
    // If a=[1,0], b=[0,1]. diff=[1, -1]. sq=[1, 1]. sum=2.
    // 2*(1 - cos(90)) = 2*(1-0) = 2. Correct.
    let a = [1.0, 0.0];
    let b = [0.0, 1.0];
    let dist = CosineMetric::distance(&a, &b);
    assert!((dist - 2.0).abs() < f64::EPSILON);

    // a=[1,0], b=[1,0]. diff=[0,0]. sum=0.
    let dist_same = CosineMetric::distance(&a, &a);
    assert!(dist_same.abs() < f64::EPSILON);

    // a=[1,0], b=[-1,0]. diff=[2,0]. sq=[4,0]. sum=4.
    // 2(1 - cos(180)) = 2(1 - (-1)) = 4. Correct.
    let c = [-1.0, 0.0];
    let dist_opp = CosineMetric::distance(&a, &c);
    assert!((dist_opp - 4.0).abs() < f64::EPSILON);
}

#[test]
fn test_poincare_validation() {
    let v_valid = [0.1, 0.2];
    assert!(PoincareMetric::validate(&v_valid).is_ok());

    let v_invalid = [1.0, 0.0]; // Norm=1. Boundary.
    assert!(PoincareMetric::validate(&v_invalid).is_err());

    let v_invalid2 = [0.8, 0.8]; // Norm sq = 0.64+0.64 = 1.28
    assert!(PoincareMetric::validate(&v_invalid2).is_err());
}

// ── Lorentz Metric Tests ──────────────────────────────────────────────────────

#[test]
fn test_lorentz_distance_identity() {
    // acosh(1 + 1e-12) ~ 1.4e-6 due to numerical stability clamping
    let origin = [1.0, 0.0, 0.0];
    let dist = LorentzMetric::distance(&origin, &origin);
    assert!(dist < 1e-5, "Self-distance should be ~0, got {dist}");
}

#[test]
fn test_lorentz_distance_known() {
    // Origin on H^2: (1, 0, 0)
    // Point at geodesic distance r=1.5: (cosh(1.5), sinh(1.5), 0)
    let r = 1.5_f64;
    let a = [1.0, 0.0, 0.0];
    let b = [r.cosh(), r.sinh(), 0.0];
    let dist = LorentzMetric::distance(&a, &b);
    assert!(
        (dist - r).abs() < 1e-9,
        "Expected distance {r}, got {dist}"
    );
}

#[test]
fn test_lorentz_validation_valid() {
    // (1, 0, 0) is on the hyperboloid: 1 - 0 = 1
    assert!(LorentzMetric::validate(&[1.0, 0.0, 0.0]).is_ok());
    // (cosh(1), sinh(1), 0): cosh^2 - sinh^2 = 1
    let v = [1.0_f64.cosh(), 1.0_f64.sinh(), 0.0];
    assert!(LorentzMetric::validate(&v).is_ok());
}

#[test]
fn test_lorentz_validation_lower_sheet() {
    // Lower sheet: t < 0
    let v = [-1.0, 0.0, 0.0];
    assert!(LorentzMetric::validate(&v).is_err());
}

#[test]
fn test_lorentz_validation_off_hyperboloid() {
    // t^2 - |x|^2 = 4 - 1 = 3 != 1
    let v = [2.0, 1.0, 0.0];
    assert!(LorentzMetric::validate(&v).is_err());
}

#[test]
#[should_panic(expected = "Scalar quantization is not supported")]
fn test_lorentz_quantized_panics() {
    use crate::vector::{HyperVector, QuantizedHyperVector};
    let v = HyperVector::<3> {
        coords: [1.0_f64.cosh(), 1.0_f64.sinh(), 0.0],
        alpha: 0.0, // unused for Lorentz
    };
    let q = QuantizedHyperVector::from_float(&v);
    let _ = LorentzMetric::distance_quantized(&q, &v);
}

#[test]
#[should_panic(expected = "Binary quantization is permanently rejected")]
fn test_lorentz_binary_panics() {
    use crate::vector::{BinaryHyperVector, HyperVector};
    let v = HyperVector::<3> {
        coords: [1.0_f64.cosh(), 1.0_f64.sinh(), 0.0],
        alpha: 0.0,
    };
    let b = BinaryHyperVector::from_float(&v);
    let _ = LorentzMetric::distance_binary(&b, &v);
}
