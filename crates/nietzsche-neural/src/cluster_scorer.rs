//! Cluster Scorer — thin wrapper around the `cluster_scorer` ONNX model.
//!
//! Provides a high-level API that downstream crates (e.g. `nietzsche-agency`)
//! can call without depending on `ort` or `ndarray` directly.
//!
//! Input:  261 f32 features (128D centroid + 128D variance + 5 scalar stats)
//! Output: 3 softmax probabilities [keep, split, merge]

use crate::REGISTRY;

/// Result of cluster scoring: softmax probabilities for keep/split/merge.
#[derive(Debug, Clone, Copy)]
pub struct ClusterScoreResult {
    pub keep: f32,
    pub split: f32,
    pub merge: f32,
}

/// Check whether the cluster_scorer model is loaded in the global registry.
pub fn is_loaded() -> bool {
    REGISTRY.has_model("cluster_scorer")
}

/// Run inference on the cluster_scorer model.
///
/// `features` must be exactly 261 f32 values:
///   - \[0..128\]:   cluster centroid embedding (128D)
///   - \[128..256\]: per-dimension variance (128D)
///   - \[256..261\]: scalar stats \[size_norm, density, avg_energy, edge_count_norm, coherence\]
///
/// Returns `Err` if the model is not loaded, the input is malformed,
/// or inference fails.
pub fn score(features: &[f32; 261]) -> Result<ClusterScoreResult, String> {
    let probs = REGISTRY
        .infer_f32("cluster_scorer", vec![1, 261], features.to_vec())
        .map_err(|e| format!("cluster_scorer inference failed: {e}"))?;

    Ok(ClusterScoreResult {
        keep:  probs.get(0).copied().unwrap_or(1.0),
        split: probs.get(1).copied().unwrap_or(0.0),
        merge: probs.get(2).copied().unwrap_or(0.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_loaded_false_by_default() {
        // No model loaded in test environment
        assert!(!is_loaded());
    }

    #[test]
    fn test_score_without_model_returns_error() {
        let features = [0.0f32; 261];
        let result = score(&features);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cluster_scorer"));
    }
}
