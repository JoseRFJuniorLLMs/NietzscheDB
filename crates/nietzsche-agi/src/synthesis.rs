//! # Synthesis — Fréchet mean dialectical synthesis
//!
//! Implements the core AGI operation: **Thesis + Antithesis → Synthesis**.
//!
//! ## Algorithm
//!
//! 1. Collect the Poincaré embeddings of the source nodes
//! 2. Project to the Riemann sphere (preserving magnitude)
//! 3. Compute the Fréchet mean on the sphere
//! 4. Project back to the Poincaré ball with `min_magnitude * DEPTH_FACTOR`
//! 5. Apply homeostasis check (prevent center collapse)
//! 6. Create the synthesis node with full provenance
//!
//! The depth factor ensures the synthesis point is always more abstract
//! (closer to center) than the most abstract input, preserving the
//! hierarchical semantics of the Poincaré ball.
//!
//! ## Why Riemann sphere?
//!
//! Computing Fréchet mean directly in the Poincaré ball is numerically
//! unstable near the boundary (distances → ∞). The sphere model avoids
//! this by stripping magnitude (which encodes depth) and computing the
//! directional centroid separately, then re-attaching a depth that is
//! more abstract than either input.

use uuid::Uuid;
use serde_json;

use nietzsche_hyp_ops::riemann;
use nietzsche_graph::{Node, NodeMeta, NodeType, PoincareVector};

use crate::error::{AgiError, AgiResult};
use crate::homeostasis::HomeostasisGuard;
use crate::rationale::Rationale;
use crate::representation::{SynthesisNode, AGI_CONTENT_KEY};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the Fréchet synthesizer.
#[derive(Debug, Clone)]
pub struct SynthesisConfig {
    /// Depth reduction factor for synthesis nodes.
    /// The synthesis point's depth = `min_input_depth * depth_factor`.
    /// Default: 0.7 (30% more abstract than the most abstract input).
    pub depth_factor: f64,

    /// Maximum Fréchet mean iterations on the sphere.
    /// Default: 50
    pub max_frechet_iterations: usize,

    /// Fréchet mean convergence tolerance.
    /// Default: 1e-8
    pub frechet_tolerance: f64,

    /// Minimum radius for synthesis nodes (homeostasis).
    /// Default: 0.05 (never collapse to true center).
    pub min_synthesis_radius: f64,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            depth_factor: 0.7,
            max_frechet_iterations: 50,
            frechet_tolerance: 1e-8,
            min_synthesis_radius: 0.05,
        }
    }
}

// ─────────────────────────────────────────────
// FrechetSynthesizer
// ─────────────────────────────────────────────

/// Dialectical synthesis engine using Fréchet mean on the Riemann sphere.
pub struct FrechetSynthesizer {
    config: SynthesisConfig,
    homeostasis: HomeostasisGuard,
}

impl FrechetSynthesizer {
    pub fn new(config: SynthesisConfig) -> Self {
        let min_radius = config.min_synthesis_radius;
        Self {
            config,
            homeostasis: HomeostasisGuard::new(min_radius),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(SynthesisConfig::default())
    }

    /// Synthesize a new point from multiple source embeddings.
    ///
    /// # Arguments
    /// - `source_embeddings`: Poincaré coordinates of source nodes (f32 storage format)
    /// - `source_ids`: UUIDs of the source nodes
    /// - `rationale`: the Rationale from the inference engine
    ///
    /// # Returns
    /// A tuple of:
    /// - `Vec<f64>`: the synthesis point in Poincaré ball (f64 for precision)
    /// - `SynthesisNode`: AGI metadata for the new node
    ///
    /// # Errors
    /// - `AgiError::EmptyInput` if fewer than 2 source nodes
    /// - `AgiError::HomeostasisViolation` if the result is too close to origin
    pub fn synthesize(
        &self,
        source_embeddings: &[&PoincareVector],
        source_ids: &[Uuid],
        mut rationale: Rationale,
    ) -> AgiResult<(Vec<f64>, SynthesisNode)> {
        if source_embeddings.len() < 2 {
            return Err(AgiError::EmptyInput {
                context: "synthesis requires at least 2 source nodes".into(),
            });
        }

        // 1. Promote f32 → f64
        let points_f64: Vec<Vec<f64>> = source_embeddings
            .iter()
            .map(|pv| pv.coords.iter().map(|&x| x as f64).collect())
            .collect();

        let dim = points_f64[0].len();

        // 2. Compute magnitudes (depths) of each input
        let magnitudes: Vec<f64> = points_f64
            .iter()
            .map(|p| p.iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect();

        let min_magnitude = magnitudes
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        // 3. Project all points to Riemann sphere
        let sphere_points: Vec<Vec<f64>> = points_f64
            .iter()
            .filter_map(|p| riemann::to_sphere(p).ok())
            .collect();

        if sphere_points.len() < 2 {
            return Err(AgiError::EmptyInput {
                context: "failed to project points to sphere".into(),
            });
        }

        // 4. Compute Fréchet mean on the sphere
        let sphere_refs: Vec<&[f64]> = sphere_points.iter().map(|v| v.as_slice()).collect();
        let frechet_sphere = riemann::frechet_mean_sphere(
            &sphere_refs,
            self.config.max_frechet_iterations,
            self.config.frechet_tolerance,
        )?;

        // 5. Project back to Poincaré ball with reduced magnitude
        let target_magnitude = (min_magnitude * self.config.depth_factor)
            .max(self.config.min_synthesis_radius);

        let synthesis_point = riemann::from_sphere(&frechet_sphere, target_magnitude);

        // 6. Homeostasis check
        let actual_norm: f64 = synthesis_point.iter().map(|x| x * x).sum::<f64>().sqrt();
        self.homeostasis.check(actual_norm)?;

        // 7. Ensure dimension matches
        if synthesis_point.len() != dim {
            return Err(AgiError::DimensionMismatch {
                expected: dim,
                got: synthesis_point.len(),
            });
        }

        // 8. Update rationale with Fréchet point
        rationale.frechet_point = Some(synthesis_point.clone());

        // 9. Create SynthesisNode metadata
        let node_id = Uuid::new_v4();
        rationale.synthesis_node_id = Some(node_id);

        let synthesis_node = SynthesisNode::new(
            node_id,
            rationale.inference_type,
            source_ids.to_vec(),
            rationale,
        );

        tracing::info!(
            node_id = %node_id,
            depth = actual_norm,
            source_count = source_embeddings.len(),
            "Fréchet synthesis produced"
        );

        Ok((synthesis_point, synthesis_node))
    }

    /// Convert a synthesis result into a NietzscheDB Node ready for insertion.
    ///
    /// The AGI metadata is embedded in the node's `content._agi` field.
    pub fn to_graph_node(
        &self,
        synthesis_point: &[f64],
        synthesis_node: &SynthesisNode,
        label: &str,
    ) -> Node {
        let depth = synthesis_point
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt() as f32;

        let coords_f32: Vec<f32> = synthesis_point.iter().map(|&x| x as f32).collect();
        let dim = coords_f32.len();

        let mut content = serde_json::json!({
            "node_label": label,
            "type": "synthesis",
            "synthesis_depth": synthesis_node.synthesis_depth,
        });
        content[AGI_CONTENT_KEY] = synthesis_node.to_agi_content();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let meta = NodeMeta {
            id: synthesis_node.node_id,
            depth,
            content,
            node_type: NodeType::Semantic, // Synthesis nodes are always Semantic (abstract)
            energy: 1.0,                   // Born with full energy
            lsystem_generation: 0,
            hausdorff_local: 1.0, // Will be recomputed by the L-System
            created_at: now,
            expires_at: None,
            metadata: std::collections::HashMap::new(),
            valence: 0.0,
            arousal: 0.0,
            is_phantom: false,
        };

        Node {
            meta,
            embedding: PoincareVector { coords: coords_f32, dim },
        }
    }

    pub fn config(&self) -> &SynthesisConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rationale::RationaleBuilder;

    fn make_pv(coords: &[f32]) -> PoincareVector {
        PoincareVector {
            coords: coords.to_vec(),
            dim: coords.len(),
        }
    }

    fn make_rationale() -> Rationale {
        RationaleBuilder::new()
            .path(vec![Uuid::new_v4(), Uuid::new_v4()])
            .hop_gcs(vec![0.9])
            .radial_gradient(-0.2)
            .cluster_transitions(2)
            .causal_fraction(1.0)
            .build(0.5, 0.05)
    }

    #[test]
    fn test_synthesize_two_points() {
        let synth = FrechetSynthesizer::with_defaults();
        let a = make_pv(&[0.3, 0.0, 0.0]);
        let b = make_pv(&[0.0, 0.3, 0.0]);
        let ids = vec![Uuid::new_v4(), Uuid::new_v4()];
        let rationale = make_rationale();

        let (point, node) = synth.synthesize(&[&a, &b], &ids, rationale).unwrap();

        // Synthesis point should be more abstract (lower norm) than inputs
        let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 0.3, "Synthesis norm {norm} should be < 0.3");
        assert!(norm >= 0.05, "Synthesis norm {norm} should be >= min_radius");
        assert_eq!(point.len(), 3);
        assert_eq!(node.source_nodes.len(), 2);
    }

    #[test]
    fn test_synthesize_preserves_dimension() {
        let synth = FrechetSynthesizer::with_defaults();
        let a = make_pv(&[0.2, 0.1, 0.0, 0.0, 0.0]);
        let b = make_pv(&[0.0, 0.2, 0.1, 0.0, 0.0]);
        let ids = vec![Uuid::new_v4(), Uuid::new_v4()];
        let rationale = make_rationale();

        let (point, _) = synth.synthesize(&[&a, &b], &ids, rationale).unwrap();
        assert_eq!(point.len(), 5, "Dimension should be preserved");
    }

    #[test]
    fn test_to_graph_node() {
        let synth = FrechetSynthesizer::with_defaults();
        let a = make_pv(&[0.3, 0.0, 0.0]);
        let b = make_pv(&[0.0, 0.3, 0.0]);
        let ids = vec![Uuid::new_v4(), Uuid::new_v4()];
        let rationale = make_rationale();

        let (point, snode) = synth.synthesize(&[&a, &b], &ids, rationale).unwrap();
        let graph_node = synth.to_graph_node(&point, &snode, "test_synthesis");

        assert_eq!(graph_node.meta.node_type, NodeType::Semantic);
        assert_eq!(graph_node.meta.energy, 1.0);
        assert!(!graph_node.meta.is_phantom);
        assert!(graph_node.meta.content.get("_agi").is_some());
    }
}
