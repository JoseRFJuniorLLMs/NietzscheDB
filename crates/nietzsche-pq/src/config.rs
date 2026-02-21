use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// Number of sub-vectors (M). Dimension must be divisible by M.
    pub m: usize,
    /// Number of centroids per sub-vector (K). Typically 256 (fits in u8).
    pub k: usize,
    /// Max iterations for k-means training.
    pub max_iterations: usize,
    /// Convergence threshold for k-means.
    pub convergence_threshold: f64,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            m: 8,
            k: 256,
            max_iterations: 25,
            convergence_threshold: 1e-5,
        }
    }
}
