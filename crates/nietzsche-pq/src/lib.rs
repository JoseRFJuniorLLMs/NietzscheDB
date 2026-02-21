//! Product Quantization (PQ) for NietzscheDB.
//!
//! Compresses high-dimensional Poincare embeddings by splitting
//! the vector into M sub-vectors and quantizing each sub-vector
//! to its nearest centroid from K=256 centroids.
//!
//! This preserves magnitude (||x|| = depth in the Poincare ball)
//! unlike Binary Quantization which destroys it.
//!
//! ## Why PQ, not BQ?
//!
//! In hyperbolic space, the norm ||x|| encodes hierarchical depth.
//! BQ (sign-based) discards all magnitude information.
//! PQ preserves it through learned centroids that capture both
//! direction AND magnitude.

pub mod codebook;
pub mod config;
pub mod distance;
pub mod encoder;
pub mod error;

pub use codebook::Codebook;
pub use config::PQConfig;
pub use distance::asymmetric_distance;
pub use encoder::{PQCode, PQEncoder};
pub use error::PQError;

#[cfg(test)]
mod tests {
    use super::*;
    use distance::DistanceTable;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Helper: generate `n` random vectors of dimension `dim` with components in [lo, hi].
    fn random_vectors(rng: &mut StdRng, n: usize, dim: usize, lo: f32, hi: f32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(lo..hi)).collect())
            .collect()
    }

    /// Helper: build a small trained codebook for testing.
    /// Uses M=4, K=16, dim=16 (sub_dim=4) to keep tests fast.
    fn small_config() -> PQConfig {
        PQConfig {
            m: 4,
            k: 16,
            max_iterations: 50,
            convergence_threshold: 1e-6,
        }
    }

    fn train_small_codebook(rng: &mut StdRng) -> Codebook {
        let config = small_config();
        let vectors = random_vectors(rng, 200, 16, -1.0, 1.0);
        Codebook::train(&vectors, &config).expect("training should succeed")
    }

    // -----------------------------------------------------------------------
    // Test 1: test_config_default
    // -----------------------------------------------------------------------
    #[test]
    fn test_config_default() {
        let cfg = PQConfig::default();
        assert_eq!(cfg.m, 8);
        assert_eq!(cfg.k, 256);
        assert_eq!(cfg.max_iterations, 25);
        assert!((cfg.convergence_threshold - 1e-5).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Test 2: test_train_codebook
    // -----------------------------------------------------------------------
    #[test]
    fn test_train_codebook() {
        let mut rng = StdRng::seed_from_u64(42);
        let config = small_config();
        let vectors = random_vectors(&mut rng, 200, 16, -1.0, 1.0);

        let codebook = Codebook::train(&vectors, &config).unwrap();

        // Verify shape: M sub-quantizers
        assert_eq!(codebook.centroids.len(), config.m);
        // Each sub-quantizer has K centroids
        for sub_centroids in &codebook.centroids {
            assert_eq!(sub_centroids.len(), config.k);
            // Each centroid has sub_dim dimensions
            for centroid in sub_centroids {
                assert_eq!(centroid.len(), 16 / config.m);
            }
        }
        assert_eq!(codebook.dim, 16);
        assert_eq!(codebook.sub_dim, 4);
    }

    // -----------------------------------------------------------------------
    // Test 3: test_encode_decode_roundtrip
    // -----------------------------------------------------------------------
    #[test]
    fn test_encode_decode_roundtrip() {
        let mut rng = StdRng::seed_from_u64(123);
        let codebook = train_small_codebook(&mut rng);
        let encoder = PQEncoder::new(&codebook);

        // Generate a test vector
        let vector: Vec<f32> = (0..16).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let code = encoder.encode(&vector).unwrap();
        let decoded = encoder.decode(&code);

        // Decoded vector should be the same dimension
        assert_eq!(decoded.len(), vector.len());

        // Compute reconstruction error (should be bounded, not zero but not huge)
        let mse: f64 = vector
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>()
            / vector.len() as f64;

        // With 16 centroids per sub-vector and 200 training vectors,
        // MSE should be well under 1.0 for [-1, 1] data.
        assert!(
            mse < 1.0,
            "reconstruction MSE too high: {mse}. PQ should approximate well."
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: test_pq_code_byte_size
    // -----------------------------------------------------------------------
    #[test]
    fn test_pq_code_byte_size() {
        let mut rng = StdRng::seed_from_u64(456);
        let codebook = train_small_codebook(&mut rng);
        let encoder = PQEncoder::new(&codebook);

        let vector: Vec<f32> = (0..16).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let code = encoder.encode(&vector).unwrap();

        // Code size should equal M (number of sub-vectors)
        assert_eq!(code.byte_size(), codebook.config.m);
        assert_eq!(code.codes.len(), 4);
    }

    // -----------------------------------------------------------------------
    // Test 5: test_dimension_mismatch
    // -----------------------------------------------------------------------
    #[test]
    fn test_dimension_mismatch() {
        // Training with dimension not divisible by M
        let config = PQConfig {
            m: 3,
            k: 4,
            max_iterations: 10,
            convergence_threshold: 1e-5,
        };
        let vectors: Vec<Vec<f32>> = (0..10).map(|_| vec![1.0; 10]).collect();
        let result = Codebook::train(&vectors, &config);
        assert!(result.is_err());
        match result.unwrap_err() {
            PQError::DimensionMismatch { dim, m } => {
                assert_eq!(dim, 10);
                assert_eq!(m, 3);
            }
            other => panic!("expected DimensionMismatch, got: {other}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 6: test_insufficient_training
    // -----------------------------------------------------------------------
    #[test]
    fn test_insufficient_training() {
        let config = small_config(); // k=16
        // Provide only 5 vectors (less than k=16)
        let vectors: Vec<Vec<f32>> = (0..5).map(|_| vec![1.0; 16]).collect();
        let result = Codebook::train(&vectors, &config);
        assert!(result.is_err());
        match result.unwrap_err() {
            PQError::InsufficientTraining { need, got } => {
                assert_eq!(need, 16);
                assert_eq!(got, 5);
            }
            other => panic!("expected InsufficientTraining, got: {other}"),
        }

        // Also test empty vector set
        let empty: Vec<Vec<f32>> = vec![];
        let result = Codebook::train(&empty, &config);
        assert!(result.is_err());
        match result.unwrap_err() {
            PQError::InsufficientTraining { got: 0, .. } => {}
            other => panic!("expected InsufficientTraining with got=0, got: {other}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7: test_asymmetric_distance
    // -----------------------------------------------------------------------
    #[test]
    fn test_asymmetric_distance() {
        let mut rng = StdRng::seed_from_u64(789);
        let codebook = train_small_codebook(&mut rng);
        let encoder = PQEncoder::new(&codebook);

        let v1: Vec<f32> = (0..16).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let v2: Vec<f32> = (0..16).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let code1 = encoder.encode(&v1).unwrap();
        let code2 = encoder.encode(&v2).unwrap();

        // Distance from v1 to itself (via its own code) should be small
        let self_dist = asymmetric_distance(&v1, &code1, &codebook);
        // Distance from v1 to v2 can be anything, but typically > self_dist
        let cross_dist = asymmetric_distance(&v1, &code2, &codebook);

        // Self-distance should be small (it's the quantization error)
        assert!(
            self_dist < 2.0,
            "self-distance too large: {self_dist}. Should be close to 0."
        );

        // The ADC distance should be non-negative
        assert!(self_dist >= 0.0);
        assert!(cross_dist >= 0.0);

        // Self distance should generally be less than cross distance
        // (unless v1 and v2 happen to be very close, which is unlikely with random)
        assert!(
            self_dist < cross_dist,
            "self_dist ({self_dist}) should be less than cross_dist ({cross_dist})"
        );
    }

    // -----------------------------------------------------------------------
    // Test 8: test_distance_table_build
    // -----------------------------------------------------------------------
    #[test]
    fn test_distance_table_build() {
        let mut rng = StdRng::seed_from_u64(1010);
        let codebook = train_small_codebook(&mut rng);

        let query: Vec<f32> = (0..16).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let table = DistanceTable::build(&query, &codebook);

        // The table should have M entries
        assert_eq!(table.num_sub_quantizers(), codebook.config.m);
        // Each entry should have K distances
        for m in 0..codebook.config.m {
            assert_eq!(table.num_centroids(m), codebook.config.k);
            // All distances should be non-negative (squared L2)
            for k in 0..codebook.config.k {
                let d = table.get(m, k);
                assert!(d >= 0.0, "distance should be non-negative, got {d}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 9: test_nearest_neighbor_quality
    // -----------------------------------------------------------------------
    #[test]
    fn test_nearest_neighbor_quality() {
        let mut rng = StdRng::seed_from_u64(2024);
        let codebook = train_small_codebook(&mut rng);
        let encoder = PQEncoder::new(&codebook);

        // Create a database of 50 vectors
        let db_vectors = random_vectors(&mut rng, 50, 16, -1.0, 1.0);
        let db_codes: Vec<PQCode> = db_vectors
            .iter()
            .map(|v| encoder.encode(v).unwrap())
            .collect();

        // Pick a query that is one of the database vectors (with small perturbation)
        let target_idx = 10;
        let query: Vec<f32> = db_vectors[target_idx]
            .iter()
            .map(|x| x + rng.gen_range(-0.01..0.01))
            .collect();

        // Find nearest neighbor using ADC
        let table = DistanceTable::build(&query, &codebook);
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;
        for (i, code) in db_codes.iter().enumerate() {
            let d = table.distance(code);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }

        // Find true nearest neighbor using exact L2
        let mut true_best_idx = 0;
        let mut true_best_dist = f64::MAX;
        for (i, v) in db_vectors.iter().enumerate() {
            let d: f64 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum();
            if d < true_best_dist {
                true_best_dist = d;
                true_best_idx = i;
            }
        }

        // The PQ-based NN should match the true NN (for this simple case with
        // a very close query). We check that PQ found the correct target.
        assert_eq!(
            best_idx, true_best_idx,
            "PQ NN (idx={best_idx}) should match exact NN (idx={true_best_idx})"
        );
        // The target should be the perturbed vector's original
        assert_eq!(true_best_idx, target_idx);
    }

    // -----------------------------------------------------------------------
    // Test 10: test_magnitude_preservation  [KEY TEST]
    // -----------------------------------------------------------------------
    /// This is the critical test proving PQ preserves magnitude information,
    /// which is essential for NietzscheDB's Poincare ball embeddings where
    /// ||x|| encodes hierarchical depth.
    ///
    /// We create three groups of vectors with DIFFERENT magnitudes:
    ///   - "shallow" vectors (small norm, near origin of Poincare ball)
    ///   - "mid" vectors (medium norm)
    ///   - "deep" vectors (large norm, near boundary of Poincare ball)
    ///
    /// After PQ encoding/decoding, the magnitude ordering must be preserved:
    /// vectors that were "deep" must still have larger norms than "shallow" ones.
    ///
    /// Binary Quantization (sign-based) would DESTROY this because sign(x)
    /// discards all magnitude info. PQ preserves it through centroids.
    #[test]
    fn test_magnitude_preservation() {
        let mut rng = StdRng::seed_from_u64(2026);
        let dim = 16;
        let config = small_config();

        // Generate training data with vectors at various magnitudes
        // to ensure centroids learn the magnitude structure
        let mut training = Vec::new();
        for scale in [0.1f32, 0.3, 0.5, 0.7, 0.9] {
            for _ in 0..50 {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0) * scale).collect();
                training.push(v);
            }
        }

        let codebook = Codebook::train(&training, &config).unwrap();
        let encoder = PQEncoder::new(&codebook);

        // Create vectors at three distinct depth levels
        let n_per_group = 20;

        // "Shallow" vectors: norm ~ 0.1 (near Poincare ball origin)
        let shallow: Vec<Vec<f32>> = (0..n_per_group)
            .map(|_| {
                (0..dim)
                    .map(|_| rng.gen_range(-1.0..1.0) * 0.1)
                    .collect()
            })
            .collect();

        // "Mid" vectors: norm ~ 0.5
        let mid: Vec<Vec<f32>> = (0..n_per_group)
            .map(|_| {
                (0..dim)
                    .map(|_| rng.gen_range(-1.0..1.0) * 0.5)
                    .collect()
            })
            .collect();

        // "Deep" vectors: norm ~ 0.9 (near Poincare ball boundary)
        let deep: Vec<Vec<f32>> = (0..n_per_group)
            .map(|_| {
                (0..dim)
                    .map(|_| rng.gen_range(-1.0..1.0) * 0.9)
                    .collect()
            })
            .collect();

        // Helper to compute L2 norm
        let norm = |v: &[f32]| -> f64 {
            v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
        };

        // Encode and decode each group, compute average norm before and after
        let avg_norm = |vectors: &[Vec<f32>]| -> (f64, f64) {
            let mut orig_sum = 0.0;
            let mut recon_sum = 0.0;
            for v in vectors {
                orig_sum += norm(v);
                let code = encoder.encode(v).unwrap();
                let decoded = encoder.decode(&code);
                recon_sum += norm(&decoded);
            }
            (
                orig_sum / vectors.len() as f64,
                recon_sum / vectors.len() as f64,
            )
        };

        let (orig_shallow, recon_shallow) = avg_norm(&shallow);
        let (orig_mid, recon_mid) = avg_norm(&mid);
        let (orig_deep, recon_deep) = avg_norm(&deep);

        // Verify original ordering
        assert!(
            orig_shallow < orig_mid,
            "original shallow ({orig_shallow}) < mid ({orig_mid})"
        );
        assert!(
            orig_mid < orig_deep,
            "original mid ({orig_mid}) < deep ({orig_deep})"
        );

        // KEY ASSERTION: PQ-reconstructed vectors preserve magnitude ordering.
        // This is what Binary Quantization CANNOT do (sign destroys magnitude).
        assert!(
            recon_shallow < recon_mid,
            "PQ-reconstructed shallow ({recon_shallow}) must be < mid ({recon_mid}). \
             Magnitude ordering must be preserved for Poincare ball depth semantics."
        );
        assert!(
            recon_mid < recon_deep,
            "PQ-reconstructed mid ({recon_mid}) must be < deep ({recon_deep}). \
             Magnitude ordering must be preserved for Poincare ball depth semantics."
        );

        // Additional: check that ADC distances also distinguish magnitude groups.
        // A shallow query should be closer to other shallow vectors than to deep ones.
        let shallow_query = &shallow[0];
        let shallow_code = encoder.encode(&shallow[1]).unwrap();
        let deep_code = encoder.encode(&deep[0]).unwrap();

        let dist_shallow_to_shallow = asymmetric_distance(shallow_query, &shallow_code, &codebook);
        let dist_shallow_to_deep = asymmetric_distance(shallow_query, &deep_code, &codebook);

        assert!(
            dist_shallow_to_shallow < dist_shallow_to_deep,
            "ADC: shallow-to-shallow ({dist_shallow_to_shallow}) should be < \
             shallow-to-deep ({dist_shallow_to_deep}). PQ preserves magnitude-based distances."
        );
    }

    // -----------------------------------------------------------------------
    // Test 11: test_encode_wrong_dimension
    // -----------------------------------------------------------------------
    #[test]
    fn test_encode_wrong_dimension() {
        let mut rng = StdRng::seed_from_u64(9999);
        let codebook = train_small_codebook(&mut rng);
        let encoder = PQEncoder::new(&codebook);

        // Try encoding a vector with wrong dimension (8 instead of 16)
        let wrong_vec = vec![1.0f32; 8];
        let result = encoder.encode(&wrong_vec);
        assert!(result.is_err());
        match result.unwrap_err() {
            PQError::DimensionMismatch { dim: 8, .. } => {}
            other => panic!("expected DimensionMismatch with dim=8, got: {other}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 12: test_codebook_serialization
    // -----------------------------------------------------------------------
    #[test]
    fn test_codebook_serialization() {
        let mut rng = StdRng::seed_from_u64(5555);
        let codebook = train_small_codebook(&mut rng);

        // Serialize to JSON
        let json = serde_json::to_string(&codebook).expect("json serialization");
        let deserialized: Codebook =
            serde_json::from_str(&json).expect("json deserialization");

        assert_eq!(deserialized.dim, codebook.dim);
        assert_eq!(deserialized.sub_dim, codebook.sub_dim);
        assert_eq!(deserialized.centroids.len(), codebook.centroids.len());

        // Serialize to bincode
        let bytes = bincode::serialize(&codebook).expect("bincode serialization");
        let deserialized2: Codebook =
            bincode::deserialize(&bytes).expect("bincode deserialization");

        assert_eq!(deserialized2.dim, codebook.dim);
        assert_eq!(deserialized2.centroids.len(), codebook.centroids.len());
    }
}
