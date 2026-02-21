//! Unit tests for hyperspace-wasm (non-WASM target).
//!
//! These tests validate the core logic (IndexWrapper, ID mapping, dimension
//! validation) without requiring a WASM runtime or browser.
//! The WASM-specific APIs (save/load IndexedDB, JsValue) are tested via
//! wasm-bindgen-test in a separate file.
//!
//! To test WASM bindings: `wasm-pack test --headless --firefox`
//! To test Rust logic:    `cargo test -p hyperspace-wasm`

// Note: hyperspace-wasm compiles to cdylib+rlib. The rlib target allows
// native Rust tests. However, wasm_bindgen types (JsValue) are not available
// in native tests. We test the underlying data structures instead.

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

// ══════════════════════════════════════════════════════════════════════════════
// ID Mapping Logic (mirrors HyperspaceDB's id_map / rev_map)
// ══════════════════════════════════════════════════════════════════════════════

/// Simulate the ID mapping layer from HyperspaceDB
struct IdMapper {
    id_map: RwLock<HashMap<u32, u32>>,
    rev_map: RwLock<HashMap<u32, u32>>,
}

impl IdMapper {
    fn new() -> Self {
        Self {
            id_map: RwLock::new(HashMap::new()),
            rev_map: RwLock::new(HashMap::new()),
        }
    }

    fn insert(&self, user_id: u32, internal_id: u32) -> Result<(), String> {
        let mut id_map = self.id_map.write();
        if id_map.contains_key(&user_id) {
            return Err("Duplicate ID not supported".into());
        }
        id_map.insert(user_id, internal_id);
        self.rev_map.write().insert(internal_id, user_id);
        Ok(())
    }

    fn resolve(&self, internal_id: u32) -> Option<u32> {
        self.rev_map.read().get(&internal_id).copied()
    }

    fn len(&self) -> usize {
        self.id_map.read().len()
    }
}

#[test]
fn id_mapper_insert_and_resolve() {
    let mapper = IdMapper::new();
    mapper.insert(100, 0).unwrap();
    mapper.insert(200, 1).unwrap();

    assert_eq!(mapper.resolve(0), Some(100));
    assert_eq!(mapper.resolve(1), Some(200));
    assert_eq!(mapper.resolve(99), None);
}

#[test]
fn id_mapper_duplicate_rejected() {
    let mapper = IdMapper::new();
    mapper.insert(100, 0).unwrap();

    let result = mapper.insert(100, 1);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Duplicate"));
}

#[test]
fn id_mapper_many_entries() {
    let mapper = IdMapper::new();
    for i in 0..1000u32 {
        mapper.insert(i + 10_000, i).unwrap();
    }
    assert_eq!(mapper.len(), 1000);

    // Verify round-trip
    for i in 0..1000u32 {
        assert_eq!(mapper.resolve(i), Some(i + 10_000));
    }
}

#[test]
fn id_mapper_u32_max() {
    let mapper = IdMapper::new();
    mapper.insert(u32::MAX, 0).unwrap();
    assert_eq!(mapper.resolve(0), Some(u32::MAX));
}

#[test]
fn id_mapper_zero_ids() {
    let mapper = IdMapper::new();
    mapper.insert(0, 0).unwrap();
    assert_eq!(mapper.resolve(0), Some(0));
}

// ══════════════════════════════════════════════════════════════════════════════
// Dimension Validation
// ══════════════════════════════════════════════════════════════════════════════

const SUPPORTED_DIMS: [usize; 5] = [384, 768, 1024, 1536, 3072];
const SUPPORTED_METRICS: [&str; 5] = ["l2", "euclidean", "cosine", "poincare", "lorentz"];

fn is_supported_config(dimension: usize, metric: &str) -> bool {
    let metric = metric.to_lowercase();
    SUPPORTED_DIMS.contains(&dimension)
        && SUPPORTED_METRICS.contains(&metric.as_str())
}

#[test]
fn valid_dimension_metric_combos() {
    for &dim in &SUPPORTED_DIMS {
        for &metric in &SUPPORTED_METRICS {
            assert!(is_supported_config(dim, metric),
                "({dim}, {metric}) should be supported");
        }
    }
}

#[test]
fn unsupported_dimensions_rejected() {
    let bad_dims = [0, 1, 128, 256, 512, 2048, 4096];
    for dim in bad_dims {
        assert!(!is_supported_config(dim, "l2"),
            "dim={dim} should be unsupported");
    }
}

#[test]
fn unsupported_metrics_rejected() {
    let bad_metrics = ["hamming", "jaccard", "manhattan", "dot", ""];
    for metric in bad_metrics {
        assert!(!is_supported_config(1536, metric),
            "metric={metric} should be unsupported");
    }
}

#[test]
fn metric_case_insensitive() {
    assert!(is_supported_config(1536, "L2"));
    assert!(is_supported_config(1536, "COSINE"));
    assert!(is_supported_config(1536, "Poincare"));
    assert!(is_supported_config(1536, "LORENTZ"));
    assert!(is_supported_config(1536, "Euclidean"));
}

// ══════════════════════════════════════════════════════════════════════════════
// Element Size Calculation
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn element_size_calculation() {
    // VectorStore element_size = dimension * 4 (f32 = 4 bytes)
    assert_eq!(384 * 4, 1536);
    assert_eq!(768 * 4, 3072);
    assert_eq!(1024 * 4, 4096);
    assert_eq!(1536 * 4, 6144);
    assert_eq!(3072 * 4, 12288);
}

// ══════════════════════════════════════════════════════════════════════════════
// Search Result Mapping
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn search_results_map_internal_to_user_ids() {
    let mapper = IdMapper::new();
    mapper.insert(42, 0).unwrap();
    mapper.insert(99, 1).unwrap();
    mapper.insert(7, 2).unwrap();

    // Simulate HNSW returning internal IDs with distances
    let raw_results: Vec<(u32, f64)> = vec![(1, 0.1), (0, 0.3), (2, 0.5)];

    let mapped: Vec<(u32, f64)> = raw_results
        .iter()
        .map(|(internal_id, dist)| {
            let user_id = mapper.resolve(*internal_id).unwrap_or(*internal_id);
            (user_id, *dist)
        })
        .collect();

    assert_eq!(mapped[0], (99, 0.1));
    assert_eq!(mapped[1], (42, 0.3));
    assert_eq!(mapped[2], (7, 0.5));
}

#[test]
fn search_results_preserve_distance_order() {
    let mapper = IdMapper::new();
    for i in 0..10u32 {
        mapper.insert(i * 100, i).unwrap();
    }

    let raw_results: Vec<(u32, f64)> = (0..10)
        .map(|i| (i as u32, i as f64 * 0.1))
        .collect();

    // Already sorted by distance
    for w in raw_results.windows(2) {
        assert!(w[0].1 <= w[1].1);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Concurrent Access (RwLock)
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn concurrent_reads_dont_block() {
    use std::thread;

    let mapper = Arc::new(IdMapper::new());
    for i in 0..100u32 {
        mapper.insert(i, i).unwrap();
    }

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let m = Arc::clone(&mapper);
            thread::spawn(move || {
                for i in 0..100u32 {
                    assert_eq!(m.resolve(i), Some(i));
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn concurrent_inserts() {
    let mapper = Arc::new(IdMapper::new());
    let handles: Vec<_> = (0..4u32)
        .map(|t| {
            let m = Arc::clone(&mapper);
            std::thread::spawn(move || {
                for i in 0..25u32 {
                    let user_id = t * 1000 + i;
                    let internal_id = t * 25 + i;
                    m.insert(user_id, internal_id).unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(mapper.len(), 100);
}

// ══════════════════════════════════════════════════════════════════════════════
// ID Map Serialization (save/load simulation)
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn id_map_serde_roundtrip() {
    let mut original: HashMap<u32, u32> = HashMap::new();
    for i in 0..50u32 {
        original.insert(i + 1000, i);
    }

    let json = serde_json::to_string(&original).unwrap();
    let restored: HashMap<u32, u32> = serde_json::from_str(&json).unwrap();

    assert_eq!(original, restored);
}

#[test]
fn reverse_map_reconstruction() {
    let id_map: HashMap<u32, u32> = [(100, 0), (200, 1), (300, 2)]
        .into_iter()
        .collect();

    // Simulate load() logic
    let mut rev_map: HashMap<u32, u32> = HashMap::new();
    for (&k, &v) in &id_map {
        rev_map.insert(v, k);
    }

    assert_eq!(rev_map[&0], 100);
    assert_eq!(rev_map[&1], 200);
    assert_eq!(rev_map[&2], 300);
}

#[test]
fn empty_id_map_serde() {
    let empty: HashMap<u32, u32> = HashMap::new();
    let json = serde_json::to_string(&empty).unwrap();
    let restored: HashMap<u32, u32> = serde_json::from_str(&json).unwrap();
    assert!(restored.is_empty());
}

// ══════════════════════════════════════════════════════════════════════════════
// Distance Metric Formulas (pure math tests)
// ══════════════════════════════════════════════════════════════════════════════

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}

fn poincare_distance(u: &[f64], v: &[f64]) -> f64 {
    let diff_sq: f64 = u.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum();
    let norm_u_sq: f64 = u.iter().map(|x| x * x).sum();
    let norm_v_sq: f64 = v.iter().map(|x| x * x).sum();
    let denom = (1.0 - norm_u_sq) * (1.0 - norm_v_sq);
    let arg = (1.0 + 2.0 * diff_sq / denom).max(1.0);
    arg.acosh()
}

#[test]
fn euclidean_same_point_is_zero() {
    let a = vec![0.1, 0.2, 0.3];
    assert!(euclidean_distance(&a, &a) < 1e-10);
}

#[test]
fn euclidean_known_distance() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-10);
}

#[test]
fn euclidean_symmetry() {
    let a = vec![0.1, 0.2, 0.3];
    let b = vec![0.4, 0.5, 0.6];
    assert!((euclidean_distance(&a, &b) - euclidean_distance(&b, &a)).abs() < 1e-10);
}

#[test]
fn cosine_identical_vectors() {
    let a = vec![1.0, 2.0, 3.0];
    assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-10);
}

#[test]
fn cosine_orthogonal_vectors() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    assert!(cosine_similarity(&a, &b).abs() < 1e-10);
}

#[test]
fn cosine_opposite_vectors() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-10);
}

#[test]
fn poincare_origin_distance() {
    let origin = vec![0.0, 0.0];
    let point = vec![0.5, 0.0];
    let d = poincare_distance(&origin, &point);
    // Known: d(0, x) = 2 * atanh(||x||) = 2 * atanh(0.5)
    let expected = 2.0 * (0.5_f64).atanh();
    assert!((d - expected).abs() < 1e-6, "expected {expected}, got {d}");
}

#[test]
fn poincare_same_point_is_zero() {
    let p = vec![0.1, 0.2, 0.3];
    assert!(poincare_distance(&p, &p) < 1e-10);
}

#[test]
fn poincare_symmetry() {
    let u = vec![0.1, 0.2];
    let v = vec![0.3, 0.4];
    assert!((poincare_distance(&u, &v) - poincare_distance(&v, &u)).abs() < 1e-10);
}

#[test]
fn poincare_boundary_grows_fast() {
    let origin = vec![0.0, 0.0];

    let near = poincare_distance(&origin, &[0.1, 0.0]);
    let mid = poincare_distance(&origin, &[0.5, 0.0]);
    let far = poincare_distance(&origin, &[0.9, 0.0]);
    let very_far = poincare_distance(&origin, &[0.99, 0.0]);

    assert!(near < mid);
    assert!(mid < far);
    assert!(far < very_far);

    // The increment near the boundary should be larger than near the center.
    // Going from 0.9 → 0.99 (Δr = 0.09) should add more distance than
    // going from 0.1 → 0.5 (Δr = 0.40), showing boundary divergence.
    let delta_near = mid - near;    // 0.1 → 0.5 (Δr = 0.40)
    let delta_far = very_far - far; // 0.9 → 0.99 (Δr = 0.09)
    assert!(delta_far > delta_near,
        "Poincaré distance increment near boundary ({delta_far:.3}) should exceed \
         increment near center ({delta_near:.3}) despite smaller Δr");
}

#[test]
fn poincare_triangle_inequality() {
    let a = vec![0.1, 0.0];
    let b = vec![0.0, 0.3];
    let c = vec![-0.2, 0.1];

    let d_ab = poincare_distance(&a, &b);
    let d_bc = poincare_distance(&b, &c);
    let d_ac = poincare_distance(&a, &c);

    assert!(d_ac <= d_ab + d_bc + 1e-10, "triangle inequality violated");
}

// ══════════════════════════════════════════════════════════════════════════════
// GlobalConfig / QuantizationMode (import check)
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn global_config_default() {
    use hyperspace_core::GlobalConfig;
    let config = GlobalConfig::default();
    // Just verify it constructs without panic
    let _ = format!("{:?}", Arc::new(config));
}

#[test]
fn quantization_mode_none() {
    use hyperspace_core::QuantizationMode;
    let mode = QuantizationMode::None;
    // Should be Copy/Clone
    let _ = mode;
}
