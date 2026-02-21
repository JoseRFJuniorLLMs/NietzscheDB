use serde::{Deserialize, Serialize};

use crate::{Codebook, PQError};

/// A PQ-compressed code: M bytes, one per sub-vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PQCode {
    pub codes: Vec<u8>,
}

impl PQCode {
    /// Byte size of this code.
    pub fn byte_size(&self) -> usize {
        self.codes.len()
    }
}

/// Encoder that compresses vectors to PQ codes using a trained codebook.
pub struct PQEncoder<'a> {
    codebook: &'a Codebook,
}

impl<'a> PQEncoder<'a> {
    pub fn new(codebook: &'a Codebook) -> Self {
        Self { codebook }
    }

    /// Encode a full-dimensional vector into a PQ code.
    pub fn encode(&self, vector: &[f32]) -> Result<PQCode, PQError> {
        if vector.len() != self.codebook.dim {
            return Err(PQError::DimensionMismatch {
                dim: vector.len(),
                m: self.codebook.config.m,
            });
        }

        let sub_dim = self.codebook.sub_dim;
        let mut codes = Vec::with_capacity(self.codebook.config.m);

        for m in 0..self.codebook.config.m {
            let sub_vec = &vector[m * sub_dim..(m + 1) * sub_dim];
            let mut best_dist = f64::MAX;
            let mut best_k = 0u8;

            for (k, centroid) in self.codebook.centroids[m].iter().enumerate() {
                let d: f64 = sub_vec
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| ((*a - *b) as f64).powi(2))
                    .sum();
                if d < best_dist {
                    best_dist = d;
                    best_k = k as u8;
                }
            }
            codes.push(best_k);
        }

        Ok(PQCode { codes })
    }

    /// Decode a PQ code back to an approximate vector.
    pub fn decode(&self, code: &PQCode) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.codebook.dim);
        for (m, &k) in code.codes.iter().enumerate() {
            result.extend_from_slice(self.codebook.centroid(m, k as usize));
        }
        result
    }
}
