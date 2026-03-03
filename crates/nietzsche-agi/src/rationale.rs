//! # Rationale — proof object for every inference
//!
//! Every AGI inference produces a [`Rationale`] that captures:
//! - The path through the hyperbolic graph (node IDs)
//! - The Geodesic Coherence Score (GCS) of the trajectory
//! - The radial gradient (depth change from start to end)
//! - How many cluster boundaries were crossed
//! - The Fréchet synthesis point (if a synthesis was produced)
//! - A fidelity score ∈ [0, 1] combining all metrics
//!
//! The Rationale is the "proof certificate" — without it, no inference is accepted.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ─────────────────────────────────────────────
// InferenceType — classification of reasoning patterns
// ─────────────────────────────────────────────

/// Classification of a reasoning trajectory based on geometric properties.
///
/// Determined by the [`InferenceEngine`](crate::inference_engine::InferenceEngine)
/// after analyzing a validated trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferenceType {
    /// **Generalization**: trajectory moves inward (depth decreases).
    /// Concrete → Abstract. The path moves from the periphery toward the center
    /// of the Poincaré ball, indicating abstraction/generalization.
    ///
    /// Detection: `radial_gradient < -threshold` (default: -0.05)
    Generalization,

    /// **Specialization**: trajectory moves outward (depth increases).
    /// Abstract → Concrete. The path moves from the center toward the boundary,
    /// indicating concretization/specialization.
    ///
    /// Detection: `radial_gradient > +threshold` (default: +0.05)
    Specialization,

    /// **Dialectical Synthesis**: trajectory crosses ≥2 cluster boundaries
    /// and produces a Fréchet mean point that is more abstract than both inputs.
    ///
    /// This is the core AGI operation: Thesis + Antithesis → Synthesis.
    /// Uses the Riemann sphere model for synthesis to avoid Poincaré distortion.
    ///
    /// Detection: `cluster_transitions >= 2 && radial_gradient < 0`
    DialecticalSynthesis,

    /// **Structural Bridge**: trajectory connects two clusters without
    /// significant depth change. A lateral connection between domains.
    ///
    /// Detection: `cluster_transitions >= 1 && |radial_gradient| < threshold`
    StructuralBridge,

    /// **Analogical Mapping**: trajectory finds structural similarity between
    /// distant graph regions via parallel transport comparison.
    ///
    /// Detection: `cluster_transitions >= 2 && parallel_transport_fidelity > threshold`
    AnalogicalMapping,

    /// **Logical Rupture**: the trajectory failed GCS validation.
    /// The path is geometrically incoherent — at least one hop violates
    /// the geodesic constraint. This inference is REJECTED.
    ///
    /// Detection: `min_gcs < gcs_threshold`
    LogicalRupture,
}

impl InferenceType {
    /// Returns true if this inference type represents a valid (non-rejected) inference.
    pub fn is_valid(&self) -> bool {
        !matches!(self, InferenceType::LogicalRupture)
    }

    /// Returns true if this inference produces a synthesis node.
    pub fn produces_synthesis(&self) -> bool {
        matches!(self, InferenceType::DialecticalSynthesis)
    }
}

impl std::fmt::Display for InferenceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceType::Generalization => write!(f, "Generalization"),
            InferenceType::Specialization => write!(f, "Specialization"),
            InferenceType::DialecticalSynthesis => write!(f, "DialecticalSynthesis"),
            InferenceType::StructuralBridge => write!(f, "StructuralBridge"),
            InferenceType::AnalogicalMapping => write!(f, "AnalogicalMapping"),
            InferenceType::LogicalRupture => write!(f, "LogicalRupture"),
        }
    }
}

// ─────────────────────────────────────────────
// Rationale — the proof certificate
// ─────────────────────────────────────────────

/// Proof certificate accompanying every AGI inference.
///
/// A Rationale captures the full geometric evidence for an inference:
/// what path was taken, how coherent it was, what type of reasoning occurred,
/// and where the synthesis point landed.
///
/// # Invariants
/// - `path` always has ≥ 2 nodes (source + target)
/// - `hop_gcs` has len = `path.len() - 1` (one score per hop)
/// - `gcs` is the harmonic mean of `hop_gcs`
/// - `fidelity ∈ [0.0, 1.0]`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rationale {
    /// Ordered sequence of node IDs forming the trajectory.
    pub path: Vec<Uuid>,

    /// Per-hop Geodesic Coherence Scores.
    /// `hop_gcs[i]` = GCS between `path[i]` and `path[i+1]`.
    pub hop_gcs: Vec<f64>,

    /// Aggregate GCS for the entire trajectory (harmonic mean of hop_gcs).
    /// Range: [0.0, 1.0] where 1.0 = perfect geodesic alignment.
    pub gcs: f64,

    /// Radial gradient: `depth(last) - depth(first)`.
    /// Negative = moving toward center (generalization).
    /// Positive = moving toward boundary (specialization).
    pub radial_gradient: f64,

    /// Number of cluster boundaries crossed along the trajectory.
    pub cluster_transitions: usize,

    /// Classification of the reasoning pattern.
    pub inference_type: InferenceType,

    /// Combined quality metric ∈ [0.0, 1.0].
    ///
    /// `fidelity = gcs * causal_bonus * (1 - rupture_penalty)`
    ///
    /// where:
    /// - `causal_bonus` = 1.2 if all edges are Timelike (capped at 1.0)
    /// - `rupture_penalty` = number of hops below GCS threshold / total hops
    pub fidelity: f64,

    /// Fréchet synthesis point in the Poincaré ball (if synthesis occurred).
    /// `None` for non-synthesis inferences.
    pub frechet_point: Option<Vec<f64>>,

    /// ID of the synthesis node inserted into the graph (if any).
    pub synthesis_node_id: Option<Uuid>,

    /// Unix timestamp of when this rationale was produced.
    pub created_at: i64,
}

impl Rationale {
    /// Returns true if this rationale represents a valid (non-ruptured) inference.
    pub fn is_valid(&self) -> bool {
        self.inference_type.is_valid()
    }

    /// Returns the minimum per-hop GCS (the weakest link in the chain).
    pub fn min_hop_gcs(&self) -> f64 {
        self.hop_gcs
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Returns the depth of the start node (from radial_gradient context).
    /// This is a convenience — the actual depth comes from the node's ‖embedding‖.
    pub fn depth_start(&self) -> Option<f64> {
        // The radial gradient alone doesn't tell us the absolute start depth.
        // Callers should look up the node for the exact value.
        None
    }
}

/// Builder for constructing a Rationale step by step.
pub struct RationaleBuilder {
    path: Vec<Uuid>,
    hop_gcs: Vec<f64>,
    radial_gradient: f64,
    cluster_transitions: usize,
    frechet_point: Option<Vec<f64>>,
    synthesis_node_id: Option<Uuid>,
    causal_fraction: f64,
}

impl RationaleBuilder {
    pub fn new() -> Self {
        Self {
            path: Vec::new(),
            hop_gcs: Vec::new(),
            radial_gradient: 0.0,
            cluster_transitions: 0,
            frechet_point: None,
            synthesis_node_id: None,
            causal_fraction: 0.0,
        }
    }

    pub fn path(mut self, path: Vec<Uuid>) -> Self {
        self.path = path;
        self
    }

    pub fn hop_gcs(mut self, scores: Vec<f64>) -> Self {
        self.hop_gcs = scores;
        self
    }

    pub fn radial_gradient(mut self, gradient: f64) -> Self {
        self.radial_gradient = gradient;
        self
    }

    pub fn cluster_transitions(mut self, count: usize) -> Self {
        self.cluster_transitions = count;
        self
    }

    pub fn frechet_point(mut self, point: Vec<f64>) -> Self {
        self.frechet_point = Some(point);
        self
    }

    pub fn synthesis_node_id(mut self, id: Uuid) -> Self {
        self.synthesis_node_id = Some(id);
        self
    }

    /// Fraction of edges that are Timelike (causal). ∈ [0, 1].
    pub fn causal_fraction(mut self, frac: f64) -> Self {
        self.causal_fraction = frac;
        self
    }

    /// Build the Rationale, computing aggregate GCS, inference type, and fidelity.
    pub fn build(self, gcs_threshold: f64, radial_threshold: f64) -> Rationale {
        let gcs = harmonic_mean(&self.hop_gcs);

        // Classify inference type
        let inference_type = classify_inference(
            gcs,
            gcs_threshold,
            self.radial_gradient,
            radial_threshold,
            self.cluster_transitions,
        );

        // Compute fidelity
        let rupture_count = self
            .hop_gcs
            .iter()
            .filter(|&&s| s < gcs_threshold)
            .count();
        let rupture_penalty = if self.hop_gcs.is_empty() {
            0.0
        } else {
            rupture_count as f64 / self.hop_gcs.len() as f64
        };
        let causal_bonus = (1.0 + 0.2 * self.causal_fraction).min(1.2);
        let fidelity = (gcs * causal_bonus * (1.0 - rupture_penalty)).clamp(0.0, 1.0);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        Rationale {
            path: self.path,
            hop_gcs: self.hop_gcs,
            gcs,
            radial_gradient: self.radial_gradient,
            cluster_transitions: self.cluster_transitions,
            inference_type,
            fidelity,
            frechet_point: self.frechet_point,
            synthesis_node_id: self.synthesis_node_id,
            created_at: now,
        }
    }
}

impl Default for RationaleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

/// Harmonic mean of a slice of positive values.
/// Returns 0.0 for empty slices or if any value is ≤ 0.
fn harmonic_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_reciprocals: f64 = values.iter().map(|&v| {
        if v <= 0.0 { return f64::INFINITY; }
        1.0 / v
    }).sum();
    if sum_reciprocals == 0.0 || sum_reciprocals.is_infinite() {
        return 0.0;
    }
    values.len() as f64 / sum_reciprocals
}

/// Classify the inference type from geometric properties.
fn classify_inference(
    gcs: f64,
    gcs_threshold: f64,
    radial_gradient: f64,
    radial_threshold: f64,
    cluster_transitions: usize,
) -> InferenceType {
    // 1. Check for logical rupture first
    if gcs < gcs_threshold {
        return InferenceType::LogicalRupture;
    }

    // 2. Dialectical synthesis: crosses ≥2 clusters and moves inward
    if cluster_transitions >= 2 && radial_gradient < -radial_threshold {
        return InferenceType::DialecticalSynthesis;
    }

    // 3. Analogical mapping: crosses ≥2 clusters but stays at same depth
    if cluster_transitions >= 2 && radial_gradient.abs() < radial_threshold {
        return InferenceType::AnalogicalMapping;
    }

    // 4. Structural bridge: crosses 1+ cluster at same depth
    if cluster_transitions >= 1 && radial_gradient.abs() < radial_threshold {
        return InferenceType::StructuralBridge;
    }

    // 5. Generalization: moves inward
    if radial_gradient < -radial_threshold {
        return InferenceType::Generalization;
    }

    // 6. Specialization: moves outward
    if radial_gradient > radial_threshold {
        return InferenceType::Specialization;
    }

    // Default: structural bridge (minimal depth change, no cluster crossing)
    InferenceType::StructuralBridge
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_mean() {
        assert!((harmonic_mean(&[1.0, 1.0, 1.0]) - 1.0).abs() < 1e-10);
        assert!((harmonic_mean(&[2.0, 2.0]) - 2.0).abs() < 1e-10);
        // Harmonic mean of 1 and 3 = 2 / (1/1 + 1/3) = 2 / (4/3) = 1.5
        assert!((harmonic_mean(&[1.0, 3.0]) - 1.5).abs() < 1e-10);
        assert_eq!(harmonic_mean(&[]), 0.0);
    }

    #[test]
    fn test_classify_logical_rupture() {
        let t = classify_inference(0.3, 0.5, -0.1, 0.05, 3);
        assert_eq!(t, InferenceType::LogicalRupture);
    }

    #[test]
    fn test_classify_dialectical_synthesis() {
        let t = classify_inference(0.8, 0.5, -0.2, 0.05, 2);
        assert_eq!(t, InferenceType::DialecticalSynthesis);
    }

    #[test]
    fn test_classify_generalization() {
        let t = classify_inference(0.9, 0.5, -0.3, 0.05, 0);
        assert_eq!(t, InferenceType::Generalization);
    }

    #[test]
    fn test_classify_specialization() {
        let t = classify_inference(0.85, 0.5, 0.4, 0.05, 0);
        assert_eq!(t, InferenceType::Specialization);
    }

    #[test]
    fn test_classify_structural_bridge() {
        let t = classify_inference(0.9, 0.5, 0.01, 0.05, 1);
        assert_eq!(t, InferenceType::StructuralBridge);
    }

    #[test]
    fn test_classify_analogical_mapping() {
        let t = classify_inference(0.85, 0.5, 0.02, 0.05, 2);
        assert_eq!(t, InferenceType::AnalogicalMapping);
    }

    #[test]
    fn test_rationale_builder() {
        let r = RationaleBuilder::new()
            .path(vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()])
            .hop_gcs(vec![0.9, 0.8])
            .radial_gradient(-0.15)
            .cluster_transitions(0)
            .causal_fraction(1.0)
            .build(0.5, 0.05);

        assert_eq!(r.inference_type, InferenceType::Generalization);
        assert!(r.fidelity > 0.0);
        assert!(r.is_valid());
        assert_eq!(r.path.len(), 3);
        assert_eq!(r.hop_gcs.len(), 2);
    }
}
