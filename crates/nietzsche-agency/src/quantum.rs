//! Neuromorphic/Quantum Bridge — Poincaré ball ↔ Bloch sphere mapping.
//!
//! Maps points in the hyperbolic Poincaré ball to quantum states on
//! the Bloch sphere, enabling future neuromorphic and quantum hardware
//! integration.
//!
//! ## Mathematical Foundation
//!
//! The Poincaré ball model of hyperbolic space maps to the Bloch sphere
//! representation of qubit states:
//!
//! - **Radial coordinate** `||p|| ∈ [0,1)` → polar angle `θ ∈ [0,π)`
//!   via `θ = 2·arctan(||p||)`, preserving the conformal structure
//! - **Angular coordinate** `atan2(p₁, p₀)` → azimuthal angle `φ ∈ [0,2π)`
//! - **Node energy** `∈ [0,1]` → state purity (mixed vs pure state)
//!
//! This mapping is conformal (angle-preserving), meaning hyperbolic
//! distances approximately correspond to quantum state fidelities.
//!
//! ## Practical Use
//!
//! - Hardware mapping: Poincaré embeddings → qubit rotations for
//!   neuromorphic or quantum processing
//! - Fidelity-based similarity: quantum fidelity as distance metric
//! - Entanglement proxy: cross-cluster fidelity as coupling measure

use std::f64::consts::PI;

// ── Configurable Entanglement Thresholds ────────────────────────────────────

/// Configuration for quantum entanglement thresholds.
///
/// Controls when the entanglement proxy triggers forced edge materialisation
/// in Schrödinger collapse. Different clinical/operational contexts may
/// require stricter or more relaxed coupling thresholds.
///
/// # Examples
///
/// ```
/// use nietzsche_agency::quantum::QuantumConfig;
///
/// // Default: 0.85 (general purpose)
/// let cfg = QuantumConfig::default();
/// assert!((cfg.default_entanglement_threshold - 0.85).abs() < 1e-9);
///
/// // Strict mode for medication dosage (fewer false positives)
/// assert!(cfg.strict_threshold > cfg.default_entanglement_threshold);
///
/// // Relaxed mode for psychological support (more associative recall)
/// assert!(cfg.relaxed_threshold < cfg.default_entanglement_threshold);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantumConfig {
    /// Default entanglement threshold for general-purpose collapse.
    /// Fidelity above this value forces edge materialisation.
    pub default_entanglement_threshold: f64,

    /// Strict threshold for safety-critical contexts (e.g. medication dosage).
    /// Higher value = fewer forced materialisations = fewer false positives.
    pub strict_threshold: f64,

    /// Relaxed threshold for exploratory contexts (e.g. psychological support).
    /// Lower value = more associative recall = broader semantic connections.
    pub relaxed_threshold: f64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            default_entanglement_threshold: 0.85,
            strict_threshold: 0.90,
            relaxed_threshold: 0.65,
        }
    }
}

impl QuantumConfig {
    /// Build from environment variables, falling back to defaults.
    ///
    /// Reads:
    /// - `QUANTUM_ENTANGLEMENT_THRESHOLD` (default: 0.85)
    /// - `QUANTUM_STRICT_THRESHOLD` (default: 0.90)
    /// - `QUANTUM_RELAXED_THRESHOLD` (default: 0.65)
    pub fn from_env() -> Self {
        fn env_f64(key: &str, default: f64) -> f64 {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        }
        Self {
            default_entanglement_threshold: env_f64("QUANTUM_ENTANGLEMENT_THRESHOLD", 0.85),
            strict_threshold:              env_f64("QUANTUM_STRICT_THRESHOLD", 0.90),
            relaxed_threshold:             env_f64("QUANTUM_RELAXED_THRESHOLD", 0.65),
        }
    }

    /// Select the appropriate threshold for a given context.
    ///
    /// - `"strict"` → `strict_threshold` (0.90)
    /// - `"relaxed"` → `relaxed_threshold` (0.65)
    /// - anything else → `default_entanglement_threshold` (0.85)
    pub fn threshold_for_context(&self, context: &str) -> f64 {
        match context {
            "strict" => self.strict_threshold,
            "relaxed" => self.relaxed_threshold,
            _ => self.default_entanglement_threshold,
        }
    }
}

/// A quantum state on the Bloch sphere.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlochState {
    /// Polar angle θ ∈ [0, π]
    pub theta: f64,
    /// Azimuthal angle φ ∈ [0, 2π)
    pub phi: f64,
    /// State purity ∈ [0, 1] (1 = pure, 0 = maximally mixed)
    pub purity: f64,
    /// Bloch vector (x, y, z) with ||v|| = purity
    pub vector: [f64; 3],
}

/// A single-qubit quantum gate on the Bloch sphere.
#[derive(Debug, Clone)]
pub enum QuantumGate {
    /// Rotation around X axis by angle (radians).
    RotX(f64),
    /// Rotation around Y axis.
    RotY(f64),
    /// Rotation around Z axis.
    RotZ(f64),
    /// Hadamard gate: maps |0⟩ → |+⟩, |1⟩ → |−⟩.
    Hadamard,
}

impl BlochState {
    /// Create from polar coordinates + purity.
    pub fn new(theta: f64, phi: f64, purity: f64) -> Self {
        let p = purity.clamp(0.0, 1.0);
        let vector = [
            p * theta.sin() * phi.cos(),
            p * theta.sin() * phi.sin(),
            p * theta.cos(),
        ];
        Self { theta, phi, purity: p, vector }
    }

    /// The |0⟩ state (north pole).
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }

    /// The |1⟩ state (south pole).
    pub fn one() -> Self {
        Self::new(PI, 0.0, 1.0)
    }

    /// Quantum fidelity between two states.
    /// For pure states: F = cos²(α/2) where α is the Bloch vector angle.
    pub fn fidelity(&self, other: &BlochState) -> f64 {
        let dot = self.vector[0] * other.vector[0]
            + self.vector[1] * other.vector[1]
            + self.vector[2] * other.vector[2];
        let norm_a = vec_norm(&self.vector);
        let norm_b = vec_norm(&other.vector);

        if norm_a < 1e-12 || norm_b < 1e-12 {
            return 0.5; // Maximally mixed
        }

        let cos_angle = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
        (1.0 + cos_angle) / 2.0
    }

    /// Trace distance: D(ρ,σ) = ||r⃗ − s⃗|| / 2.
    pub fn trace_distance(&self, other: &BlochState) -> f64 {
        let dx = self.vector[0] - other.vector[0];
        let dy = self.vector[1] - other.vector[1];
        let dz = self.vector[2] - other.vector[2];
        (dx * dx + dy * dy + dz * dz).sqrt() / 2.0
    }

    /// Apply a quantum gate, returning the transformed state.
    pub fn apply_gate(&self, gate: &QuantumGate) -> BlochState {
        match gate {
            QuantumGate::RotX(a) => self.rotate_x(*a),
            QuantumGate::RotY(a) => self.rotate_y(*a),
            QuantumGate::RotZ(a) => self.rotate_z(*a),
            QuantumGate::Hadamard => {
                // Hadamard on Bloch sphere: (x,y,z) → (z,−y,x)
                BlochState::from_vector(
                    [self.vector[2], -self.vector[1], self.vector[0]],
                    self.purity,
                )
            }
        }
    }

    fn rotate_x(&self, angle: f64) -> BlochState {
        let (c, s) = (angle.cos(), angle.sin());
        BlochState::from_vector(
            [
                self.vector[0],
                c * self.vector[1] - s * self.vector[2],
                s * self.vector[1] + c * self.vector[2],
            ],
            self.purity,
        )
    }

    fn rotate_y(&self, angle: f64) -> BlochState {
        let (c, s) = (angle.cos(), angle.sin());
        BlochState::from_vector(
            [
                c * self.vector[0] + s * self.vector[2],
                self.vector[1],
                -s * self.vector[0] + c * self.vector[2],
            ],
            self.purity,
        )
    }

    fn rotate_z(&self, angle: f64) -> BlochState {
        let (c, s) = (angle.cos(), angle.sin());
        BlochState::from_vector(
            [
                c * self.vector[0] - s * self.vector[1],
                s * self.vector[0] + c * self.vector[1],
                self.vector[2],
            ],
            self.purity,
        )
    }

    fn from_vector(vector: [f64; 3], purity: f64) -> BlochState {
        let norm = vec_norm(&vector);
        let theta = if norm < 1e-12 { 0.0 } else { (vector[2] / norm).clamp(-1.0, 1.0).acos() };
        let phi = {
            let raw = vector[1].atan2(vector[0]);
            if raw < 0.0 { raw + 2.0 * PI } else { raw }
        };
        BlochState { theta, phi, purity, vector }
    }
}

fn vec_norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// ── Public mapping functions ────────────────────────────────────────────────

/// Map a Poincaré ball point to a Bloch sphere state.
///
/// - `r = ||embedding||` → `θ = 2·arctan(r)` (conformal map)
/// - angle from first two dims → `φ` (azimuthal)
/// - `energy` → purity (coherence of quantum state)
pub fn poincare_to_bloch(embedding: &[f64], energy: f32) -> BlochState {
    if embedding.is_empty() {
        return BlochState::zero();
    }

    let r: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
    let r = r.min(0.9999);

    let theta = 2.0 * r.atan();

    let phi = if embedding.len() >= 2 {
        let angle = embedding[1].atan2(embedding[0]);
        if angle < 0.0 { angle + 2.0 * PI } else { angle }
    } else {
        0.0
    };

    let purity = (energy as f64).clamp(0.0, 1.0);
    BlochState::new(theta, phi, purity)
}

/// Map a Bloch sphere state back to a 2D Poincaré ball point.
/// Returns `(embedding, energy)`.
pub fn bloch_to_poincare(state: &BlochState) -> (Vec<f64>, f32) {
    let r = (state.theta / 2.0).tan().min(0.9999);
    let x = r * state.phi.cos();
    let y = r * state.phi.sin();
    (vec![x, y], state.purity as f32)
}

/// Batch-map nodes to Bloch states.
pub fn batch_poincare_to_bloch(
    nodes: &[(Vec<f64>, f32)], // (embedding, energy)
) -> Vec<BlochState> {
    nodes
        .iter()
        .map(|(emb, energy)| poincare_to_bloch(emb, *energy))
        .collect()
}

/// Entanglement proxy: average fidelity between two groups.
/// High fidelity ≈ strong coupling between groups.
pub fn entanglement_proxy(group_a: &[BlochState], group_b: &[BlochState]) -> f64 {
    if group_a.is_empty() || group_b.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0u64;
    for a in group_a {
        for b in group_b {
            total += a.fidelity(b);
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { total / count as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_maps_to_north_pole() {
        let state = poincare_to_bloch(&[0.0, 0.0], 1.0);
        assert!(state.theta.abs() < 0.01, "origin → θ≈0 (north pole)");
        assert!((state.purity - 1.0).abs() < 0.01);
    }

    #[test]
    fn boundary_maps_near_equator() {
        let state = poincare_to_bloch(&[0.99, 0.0], 1.0);
        // θ = 2·arctan(0.99) ≈ 1.56 ≈ π/2
        assert!(
            (state.theta - PI / 2.0).abs() < 0.1,
            "boundary → equator, got θ={}",
            state.theta
        );
    }

    #[test]
    fn round_trip_preserves_point() {
        let emb = vec![0.3, 0.4];
        let energy = 0.7f32;
        let state = poincare_to_bloch(&emb, energy);
        let (recovered, rec_e) = bloch_to_poincare(&state);
        assert!((recovered[0] - emb[0]).abs() < 0.01);
        assert!((recovered[1] - emb[1]).abs() < 0.01);
        assert!((rec_e - energy).abs() < 0.01);
    }

    #[test]
    fn fidelity_same_state_is_one() {
        let state = BlochState::new(1.0, 0.5, 1.0);
        assert!((state.fidelity(&state) - 1.0).abs() < 0.01);
    }

    #[test]
    fn fidelity_orthogonal_states_near_zero() {
        let zero = BlochState::zero();
        let one = BlochState::one();
        assert!(zero.fidelity(&one) < 0.01);
    }

    #[test]
    fn hadamard_moves_to_equator() {
        let zero = BlochState::zero();
        let plus = zero.apply_gate(&QuantumGate::Hadamard);
        assert!(
            (plus.theta - PI / 2.0).abs() < 0.1,
            "H|0⟩ → equator, got θ={}",
            plus.theta
        );
    }

    #[test]
    fn rotation_z_changes_phi() {
        let state = BlochState::new(PI / 2.0, 0.0, 1.0);
        let rotated = state.apply_gate(&QuantumGate::RotZ(PI / 2.0));
        assert!(
            (rotated.phi - PI / 2.0).abs() < 0.1,
            "RotZ(π/2) shifts φ, got φ={}",
            rotated.phi
        );
    }

    #[test]
    fn low_energy_gives_mixed_state() {
        let state = poincare_to_bloch(&[0.5, 0.0], 0.1);
        assert!(state.purity < 0.2);
        assert!(vec_norm(&state.vector) < 0.2);
    }

    #[test]
    fn entanglement_proxy_self_high() {
        let states = vec![
            BlochState::new(0.5, 0.5, 1.0),
            BlochState::new(0.6, 0.5, 1.0),
        ];
        assert!(entanglement_proxy(&states, &states) > 0.8);
    }

    #[test]
    fn trace_distance_opposite_poles() {
        let zero = BlochState::zero();
        let one = BlochState::one();
        let d = zero.trace_distance(&one);
        assert!((d - 1.0).abs() < 0.01);
    }

    #[test]
    fn batch_mapping() {
        let nodes = vec![
            (vec![0.1, 0.2], 0.8f32),
            (vec![0.5, -0.3], 0.5f32),
        ];
        let states = batch_poincare_to_bloch(&nodes);
        assert_eq!(states.len(), 2);
        assert!(states[0].purity > states[1].purity);
    }
}
