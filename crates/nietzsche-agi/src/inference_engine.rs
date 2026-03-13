// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! # Inference Engine — rule engine that classifies trajectories
//!
//! The [`InferenceEngine`] takes a validated [`GeodesicTrajectory`] and produces
//! a [`Rationale`] by:
//!
//! 1. Extracting the radial gradient (depth change)
//! 2. Counting cluster transitions along the path
//! 3. Computing causal fraction (% of Timelike edges)
//! 4. Classifying the inference type via the rule hierarchy
//! 5. Building the Rationale proof certificate
//!
//! # Design
//!
//! The inference engine is a **pure computation layer** — it does not read from
//! or write to the graph. All required data (edge causality, cluster membership)
//! is provided by the caller. This keeps the AGI crate independent of storage.
//!
//! # Usage
//! ```ignore
//! let engine = InferenceEngine::new(config);
//! let rationale = engine.analyze(&trajectory, &edge_info, cluster_map)?;
//! match rationale.inference_type {
//!     InferenceType::DialecticalSynthesis => { /* trigger Fréchet synthesis */ }
//!     InferenceType::LogicalRupture => { /* reject */ }
//!     _ => { /* record the rationale */ }
//! }
//! ```

use uuid::Uuid;
use nietzsche_graph::CausalType;

use crate::error::AgiResult;
use crate::rationale::{InferenceType, Rationale, RationaleBuilder};
use crate::trajectory::GeodesicTrajectory;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the inference engine.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Minimum aggregate GCS to accept a trajectory (below → LogicalRupture).
    /// Default: 0.5
    pub gcs_threshold: f64,

    /// Minimum |radial_gradient| to classify as Generalization/Specialization.
    /// Default: 0.05
    pub radial_threshold: f64,

    /// Minimum cluster transitions for DialecticalSynthesis.
    /// Default: 2
    pub min_cluster_transitions_for_synthesis: usize,

    /// Whether to automatically trigger synthesis for DialecticalSynthesis inferences.
    /// Default: true
    pub auto_synthesize: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            gcs_threshold: 0.5,
            radial_threshold: 0.05,
            min_cluster_transitions_for_synthesis: 2,
            auto_synthesize: true,
        }
    }
}

// ─────────────────────────────────────────────
// EdgeInfo — causal metadata for edges along a path
// ─────────────────────────────────────────────

/// Causal information about edges along a trajectory.
///
/// Provided by the caller (who has access to the graph).
/// One entry per hop (path.len() - 1).
#[derive(Debug, Clone)]
pub struct EdgeInfo {
    /// Causal type of the edge.
    pub causal_type: CausalType,
}

// ─────────────────────────────────────────────
// InferenceEngine
// ─────────────────────────────────────────────

/// Rule engine that classifies validated trajectories into inference types.
///
/// The engine uses geometric properties of the trajectory (radial gradient,
/// cluster transitions, GCS) to determine what kind of reasoning occurred.
///
/// **Pure computation** — does not access the graph storage directly.
pub struct InferenceEngine {
    config: InferenceConfig,
}

impl InferenceEngine {
    pub fn new(config: InferenceConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(InferenceConfig::default())
    }

    /// Analyze a validated trajectory and produce a Rationale.
    ///
    /// # Arguments
    /// - `trajectory`: a GCS-validated trajectory
    /// - `edge_infos`: causal type for each hop (len = path.len() - 1), or empty
    /// - `cluster_map`: optional function mapping node ID → cluster ID
    ///
    /// # Returns
    /// A [`Rationale`] with the classified inference type and all metrics.
    pub fn analyze(
        &self,
        trajectory: &GeodesicTrajectory,
        edge_infos: &[EdgeInfo],
        cluster_map: Option<&dyn Fn(&Uuid) -> Option<u32>>,
    ) -> AgiResult<Rationale> {
        let path = &trajectory.path;
        let hop_gcs: Vec<f64> = trajectory.hop_scores.iter().map(|h| h.score).collect();

        // For 2-node paths, hop_gcs is empty → GCS = 1.0
        let effective_gcs = if hop_gcs.is_empty() {
            vec![1.0]
        } else {
            hop_gcs
        };

        // Count cluster transitions
        let cluster_transitions = count_cluster_transitions(path, cluster_map);

        // Compute causal fraction from provided edge info
        let causal_fraction = compute_causal_fraction(edge_infos);

        // Build the rationale
        let rationale = RationaleBuilder::new()
            .path(path.clone())
            .hop_gcs(effective_gcs)
            .radial_gradient(trajectory.radial_gradient)
            .cluster_transitions(cluster_transitions)
            .causal_fraction(causal_fraction)
            .build(self.config.gcs_threshold, self.config.radial_threshold);

        tracing::debug!(
            inference_type = %rationale.inference_type,
            gcs = rationale.gcs,
            fidelity = rationale.fidelity,
            cluster_transitions = cluster_transitions,
            radial_gradient = trajectory.radial_gradient,
            "inference classified"
        );

        Ok(rationale)
    }

    /// Convenience: analyze a trajectory and return the inference type directly.
    pub fn classify(
        &self,
        trajectory: &GeodesicTrajectory,
        edge_infos: &[EdgeInfo],
        cluster_map: Option<&dyn Fn(&Uuid) -> Option<u32>>,
    ) -> AgiResult<InferenceType> {
        let rationale = self.analyze(trajectory, edge_infos, cluster_map)?;
        Ok(rationale.inference_type)
    }

    /// Returns the config for inspection.
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

/// Count how many times the cluster changes along the path.
fn count_cluster_transitions(
    path: &[Uuid],
    cluster_map: Option<&dyn Fn(&Uuid) -> Option<u32>>,
) -> usize {
    let cluster_fn = match cluster_map {
        Some(f) => f,
        None => return 0, // No cluster info → 0 transitions
    };

    let clusters: Vec<Option<u32>> = path.iter().map(|id| cluster_fn(id)).collect();
    let mut transitions = 0;
    let mut last_cluster: Option<u32> = None;

    for c in &clusters {
        if let Some(current) = c {
            if let Some(prev) = last_cluster {
                if *current != prev {
                    transitions += 1;
                }
            }
            last_cluster = Some(*current);
        }
    }

    transitions
}

/// Compute the fraction of edges along the path that are Timelike (causal).
fn compute_causal_fraction(edge_infos: &[EdgeInfo]) -> f64 {
    if edge_infos.is_empty() {
        return 0.0;
    }

    let timelike = edge_infos
        .iter()
        .filter(|e| e.causal_type == CausalType::Timelike)
        .count();

    timelike as f64 / edge_infos.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{GcsConfig, GeodesicCoherenceScore, GeodesicTrajectory};

    fn make_trajectory(path_len: usize, gcs_scores: Vec<f64>, radial: f64) -> GeodesicTrajectory {
        let path: Vec<Uuid> = (0..path_len).map(|_| Uuid::new_v4()).collect();
        let embeddings: Vec<Vec<f64>> = (0..path_len).map(|_| vec![0.0; 3]).collect();
        let hop_scores: Vec<GeodesicCoherenceScore> = gcs_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| GeodesicCoherenceScore {
                hop: i + 1,
                score,
                is_coherent: score >= 0.5,
            })
            .collect();

        GeodesicTrajectory {
            path,
            embeddings,
            hop_scores,
            aggregate_gcs: if gcs_scores.is_empty() {
                1.0
            } else {
                gcs_scores.iter().sum::<f64>() / gcs_scores.len() as f64
            },
            is_valid: true,
            radial_gradient: radial,
        }
    }

    #[test]
    fn test_classify_generalization() {
        let engine = InferenceEngine::with_defaults();
        let traj = make_trajectory(3, vec![0.9], -0.2);
        let result = engine.classify(&traj, &[], None).unwrap();
        assert_eq!(result, InferenceType::Generalization);
    }

    #[test]
    fn test_classify_specialization() {
        let engine = InferenceEngine::with_defaults();
        let traj = make_trajectory(3, vec![0.85], 0.3);
        let result = engine.classify(&traj, &[], None).unwrap();
        assert_eq!(result, InferenceType::Specialization);
    }

    #[test]
    fn test_classify_with_cluster_transitions() {
        let engine = InferenceEngine::with_defaults();
        let traj = make_trajectory(4, vec![0.8, 0.85], -0.15);

        let a = traj.path[0];
        let b = traj.path[1];
        let c = traj.path[2];
        let d = traj.path[3];

        let cluster_fn = move |id: &Uuid| -> Option<u32> {
            if *id == a || *id == b { Some(0) }
            else { Some(1) }
        };

        let result = engine.classify(&traj, &[], Some(&cluster_fn)).unwrap();
        // 1 cluster transition + negative gradient → still just generalization
        // (need ≥2 transitions for dialectical synthesis)
        assert_eq!(result, InferenceType::Generalization);
    }

    #[test]
    fn test_causal_fraction() {
        let edges = vec![
            EdgeInfo { causal_type: CausalType::Timelike },
            EdgeInfo { causal_type: CausalType::Spacelike },
            EdgeInfo { causal_type: CausalType::Timelike },
        ];
        let frac = compute_causal_fraction(&edges);
        assert!((frac - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cluster_transitions_no_map() {
        let path = vec![Uuid::new_v4(), Uuid::new_v4()];
        assert_eq!(count_cluster_transitions(&path, None), 0);
    }

    #[test]
    fn test_cluster_transitions_multiple() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();

        let map = move |id: &Uuid| -> Option<u32> {
            if *id == a { Some(0) }
            else if *id == b { Some(1) }
            else { Some(2) }
        };

        let path = vec![a, b, c];
        let transitions = count_cluster_transitions(&path, Some(&map));
        assert_eq!(transitions, 2); // 0→1→2
    }
}
