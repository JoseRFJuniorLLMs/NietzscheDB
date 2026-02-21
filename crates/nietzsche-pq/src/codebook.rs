use serde::{Deserialize, Serialize};

use crate::{PQConfig, PQError};

/// The PQ codebook: M sub-quantizers, each with K centroids.
/// centroids[m][k] is a Vec<f32> of dimension D/M.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    pub config: PQConfig,
    pub dim: usize,
    pub sub_dim: usize,
    /// Shape: [M][K][sub_dim]
    pub centroids: Vec<Vec<Vec<f32>>>,
}

impl Codebook {
    /// Train a codebook from training vectors using k-means.
    pub fn train(vectors: &[Vec<f32>], config: &PQConfig) -> Result<Self, PQError> {
        // Validate
        if vectors.is_empty() {
            return Err(PQError::InsufficientTraining {
                need: config.k,
                got: 0,
            });
        }
        let dim = vectors[0].len();
        if dim % config.m != 0 {
            return Err(PQError::DimensionMismatch { dim, m: config.m });
        }
        if vectors.len() < config.k {
            return Err(PQError::InsufficientTraining {
                need: config.k,
                got: vectors.len(),
            });
        }

        let sub_dim = dim / config.m;
        let mut centroids = Vec::with_capacity(config.m);

        for m_idx in 0..config.m {
            // Extract sub-vectors for this partition
            let sub_vectors: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| v[m_idx * sub_dim..(m_idx + 1) * sub_dim].to_vec())
                .collect();

            // Run k-means on this partition
            let partition_centroids = kmeans(
                &sub_vectors,
                config.k,
                config.max_iterations,
                config.convergence_threshold,
            );
            centroids.push(partition_centroids);
        }

        Ok(Self {
            config: config.clone(),
            dim,
            sub_dim,
            centroids,
        })
    }

    /// Get the centroid for sub-vector m, code k.
    pub fn centroid(&self, m: usize, k: usize) -> &[f32] {
        &self.centroids[m][k]
    }
}

/// Simple k-means implementation. Returns K centroids.
fn kmeans(vectors: &[Vec<f32>], k: usize, max_iter: usize, threshold: f64) -> Vec<Vec<f32>> {
    let dim = vectors[0].len();
    let mut centroids: Vec<Vec<f32>> = vectors.iter().take(k).cloned().collect();

    // Pad with zeros if not enough unique vectors
    while centroids.len() < k {
        centroids.push(vec![0.0; dim]);
    }

    for _ in 0..max_iter {
        // Assign each vector to nearest centroid
        let mut assignments = vec![0usize; vectors.len()];
        for (i, v) in vectors.iter().enumerate() {
            let mut best_dist = f64::MAX;
            let mut best_k = 0;
            for (ki, c) in centroids.iter().enumerate() {
                let d: f64 = v
                    .iter()
                    .zip(c.iter())
                    .map(|(a, b)| ((*a - *b) as f64).powi(2))
                    .sum();
                if d < best_dist {
                    best_dist = d;
                    best_k = ki;
                }
            }
            assignments[i] = best_k;
        }

        // Recompute centroids
        let mut new_centroids = vec![vec![0.0f64; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, v) in vectors.iter().enumerate() {
            let ki = assignments[i];
            counts[ki] += 1;
            for (j, val) in v.iter().enumerate() {
                new_centroids[ki][j] += *val as f64;
            }
        }

        let mut max_shift = 0.0f64;
        for ki in 0..k {
            if counts[ki] > 0 {
                let new_c: Vec<f32> = new_centroids[ki]
                    .iter()
                    .map(|s| (*s / counts[ki] as f64) as f32)
                    .collect();
                let shift: f64 = centroids[ki]
                    .iter()
                    .zip(new_c.iter())
                    .map(|(a, b)| ((*a - *b) as f64).powi(2))
                    .sum();
                max_shift = max_shift.max(shift);
                centroids[ki] = new_c;
            }
        }

        if max_shift < threshold {
            break;
        }
    }

    centroids
}
