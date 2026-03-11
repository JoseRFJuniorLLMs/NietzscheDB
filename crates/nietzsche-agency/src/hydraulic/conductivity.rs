//! ConductivityTensor — adaptive metric that makes frequently-used paths shorter.
//!
//! Each edge carries a `conductivity: f32` field that modifies the effective
//! Poincaré distance: `d_eff = d_poincaré / conductivity`.
//!
//! ## Update Rules
//!
//! 1. **Hebbian LTP** (Phase XII.5): co-activation → conductivity increase
//! 2. **Flow reinforcement** (Phase XVII): high flow rate → conductivity increase
//! 3. **Temporal decay**: unused edges → conductivity decays toward 1.0
//! 4. **Murray rebalancer**: fractal equilibrium during sleep

use uuid::Uuid;

use super::flow_ledger::{FlowConfig, FlowLedger};

/// Compute effective distance accounting for edge conductivity.
///
/// This is the "myelinated" distance: well-trodden paths feel shorter.
/// Raw Poincaré distance is preserved for similarity queries (HNSW KNN).
///
/// # Usage
///
/// - DIFFUSE walk: use `effective_distance` (flow follows myelinated paths)
/// - Gravity force: use `effective_distance` (pulls through conductive channels)
/// - Heat flow: use `effective_distance` (Fourier's law: q = κ·ΔE/d_eff)
/// - HNSW KNN: use raw Poincaré distance (similarity must be pure geometric)
#[inline]
pub fn effective_distance(poincare_dist: f64, conductivity: f32) -> f64 {
    poincare_dist / (conductivity.clamp(0.01, 10.0) as f64)
}

/// Compute flow-proportional conductivity deltas for all tracked edges.
///
/// Edges above mean flow rate get wider; edges below get narrower.
///
/// Formula: Δκ = α_flow × (f_edge / f_mean - 1) × κ_current
///
/// Returns: Vec<(edge_id, old_κ, new_κ)>
pub fn compute_flow_conductivity_updates(
    ledger: &FlowLedger,
    config: &FlowConfig,
    get_conductivity: impl Fn(Uuid) -> f32,
) -> Vec<ConductivityUpdate> {
    if !config.enabled {
        return Vec::new();
    }

    let mean_flow = ledger.mean_flow_rate();
    if mean_flow < 1e-9 {
        return Vec::new();
    }

    let mut updates = Vec::new();

    // Only process edges with actual flow data
    let highways = ledger.top_highways(usize::MAX);
    for (edge_id, flow_rate) in highways {
        let current_k = get_conductivity(edge_id);
        let flow_ratio = flow_rate / mean_flow;

        // Δκ = α × (f/f_mean - 1) × κ
        let delta = config.conductivity_rate as f32 * (flow_ratio as f32 - 1.0) * current_k;
        if delta.abs() < 1e-6 {
            continue;
        }

        let new_k = (current_k + delta).clamp(config.conductivity_min, config.conductivity_max);
        if (new_k - current_k).abs() < 1e-6 {
            continue;
        }

        updates.push(ConductivityUpdate {
            edge_id,
            old_conductivity: current_k,
            new_conductivity: new_k,
            flow_rate,
            pressure: ledger.pressure(edge_id),
        });
    }

    updates
}

/// Decay conductivity toward baseline (1.0) for dormant edges.
///
/// Formula: κ(t) = 1.0 + (κ₀ - 1.0) × e^(-λ × Δt)
///
/// Unlike edge weight decay (toward 0), conductivity decays toward 1.0:
/// unused edges don't die, they just lose their myelination.
pub fn decay_conductivity(
    current: f32,
    idle_secs: f64,
    lambda: f64,
) -> f32 {
    let excess = current - 1.0;
    if excess.abs() < 1e-6 {
        return 1.0;
    }
    let decayed = 1.0 + excess * (-lambda * idle_secs).exp() as f32;
    decayed.clamp(0.01, 10.0)
}

/// A conductivity update produced by flow analysis.
#[derive(Debug, Clone)]
pub struct ConductivityUpdate {
    pub edge_id: Uuid,
    pub old_conductivity: f32,
    pub new_conductivity: f32,
    pub flow_rate: f64,
    pub pressure: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_distance_baseline() {
        // conductivity = 1.0 → no change
        assert!((effective_distance(5.0, 1.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_effective_distance_myelinated() {
        // conductivity = 2.0 → distance halved
        assert!((effective_distance(5.0, 2.0) - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_effective_distance_constricted() {
        // conductivity = 0.5 → distance doubled
        assert!((effective_distance(5.0, 0.5) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_effective_distance_clamp() {
        // conductivity = 0 → clamped to 0.01
        let d = effective_distance(1.0, 0.0);
        assert!((d - 100.0).abs() < 1e-9);

        // conductivity = 100 → clamped to 10.0
        let d = effective_distance(10.0, 100.0);
        assert!((d - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_decay_conductivity_toward_baseline() {
        // High conductivity decays toward 1.0
        let decayed = decay_conductivity(5.0, 100_000.0, 0.00001);
        assert!(decayed < 5.0);
        assert!(decayed > 1.0);

        // Low conductivity also approaches 1.0
        let decayed = decay_conductivity(0.5, 100_000.0, 0.00001);
        assert!(decayed > 0.5);
        assert!(decayed < 1.0);
    }

    #[test]
    fn test_decay_conductivity_at_baseline() {
        // Already at 1.0 → stays at 1.0
        let decayed = decay_conductivity(1.0, 999_999.0, 0.1);
        assert!((decayed - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_decay_conductivity_zero_time() {
        // No time passed → no change
        let decayed = decay_conductivity(5.0, 0.0, 0.1);
        assert!((decayed - 5.0).abs() < 1e-5);
    }
}
