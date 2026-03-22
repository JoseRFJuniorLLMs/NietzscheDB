// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! NietzscheEvolve — Genome Persistence & Auto-Apply.
//!
//! Closes the evolutionary loop: when an evolver finds better parameters,
//! they are persisted to CF_META and automatically applied to the running
//! AgencyEngine on the next tick.
//!
//! ## Flow
//!
//! ```text
//! EnergyEvolver/CogEvolver/ParamEvolver
//!   → finds best genome
//!   → persist_genome(storage, key, json)    ← writes to RocksDB CF_META
//!
//! AgencyEngine::tick()
//!   → apply_evolved_params(storage, config) ← reads from CF_META, patches config
//!   → phases run with evolved parameters
//! ```
//!
//! ## No code rewriting
//!
//! The evolved parameters are applied at **runtime** via config mutation.
//! No Rust source code is modified, no recompilation needed.
//! On server restart, the genome is re-loaded from CF_META.

use nietzsche_graph::GraphStorage;
use crate::config::AgencyConfig;

// ── Meta keys for CF_META ─────────────────────────────────────

/// CF_META key for the evolved energy genome.
pub const META_ENERGY_GENOME: &str = "evolve:energy_genome";
/// CF_META key for the evolved cognitive genome.
pub const META_COG_GENOME: &str = "evolve:cog_genome";
/// CF_META key for the evolved HNSW genome.
pub const META_HNSW_GENOME: &str = "evolve:hnsw_genome";
/// CF_META key for the evolution generation counter.
pub const META_EVOLVE_GENERATION: &str = "evolve:generation";

// ── Persist ───────────────────────────────────────────────────

/// Persist a genome to CF_META as JSON bytes.
pub fn persist_genome(
    storage: &GraphStorage,
    key: &str,
    json: &str,
) -> Result<(), String> {
    storage.put_meta(key, json.as_bytes())
        .map_err(|e| format!("persist_genome({key}): {e}"))
}

/// Load a genome from CF_META as a JSON string.
pub fn load_genome(
    storage: &GraphStorage,
    key: &str,
) -> Option<String> {
    storage.get_meta(key)
        .ok()
        .flatten()
        .and_then(|bytes| String::from_utf8(bytes).ok())
}

// ── Apply evolved energy parameters ──────────────────────────

/// Parsed energy genome values (deserialized from JSON).
#[derive(Debug, Clone)]
pub struct EvolvedEnergyParams {
    pub decay_rate: f32,
    pub boost_on_access: f32,
    pub hub_bonus_factor: f32,
    pub depth_scaling: f32,
    pub coherence_bonus: f32,
    pub thermal_conductivity: f32,
}

impl EvolvedEnergyParams {
    /// Parse from JSON string. Returns None if parsing fails.
    pub fn from_json(json: &str) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_str(json).ok()?;
        Some(Self {
            decay_rate: v.get("decay_rate")?.as_f64()? as f32,
            boost_on_access: v.get("boost_on_access")?.as_f64()? as f32,
            hub_bonus_factor: v.get("hub_bonus_factor")?.as_f64()? as f32,
            depth_scaling: v.get("depth_scaling")?.as_f64()? as f32,
            coherence_bonus: v.get("coherence_bonus")?.as_f64()? as f32,
            thermal_conductivity: v.get("thermal_conductivity")?.as_f64()? as f32,
        })
    }
}

/// Apply evolved energy parameters to the running AgencyConfig.
///
/// Modifies thermal conductivity and related params in-place.
/// Returns true if parameters were applied.
pub fn apply_evolved_energy(
    storage: &GraphStorage,
    config: &mut AgencyConfig,
) -> bool {
    let json = match load_genome(storage, META_ENERGY_GENOME) {
        Some(j) => j,
        None => return false,
    };

    let params = match EvolvedEnergyParams::from_json(&json) {
        Some(p) => p,
        None => return false,
    };

    // Apply thermal conductivity (Phase XIII)
    config.thermo_conductivity = params.thermal_conductivity as f64;

    tracing::info!(
        conductivity = params.thermal_conductivity,
        decay_rate = params.decay_rate,
        hub_bonus = params.hub_bonus_factor,
        "NietzscheEvolve: applied evolved energy params from CF_META"
    );

    true
}

// ── Apply evolved cognitive parameters ───────────────────────

/// Parsed cognitive genome values.
#[derive(Debug, Clone)]
pub struct EvolvedCogParams {
    pub thermal_conductivity: f32,
    pub growth_distance: f32,
    pub decay_lambda: f64,
    pub cluster_radius: f32,
}

impl EvolvedCogParams {
    pub fn from_json(json: &str) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_str(json).ok()?;
        Some(Self {
            thermal_conductivity: v.get("thermal_conductivity")?.as_f64()? as f32,
            growth_distance: v.get("growth_distance")?.as_f64()? as f32,
            decay_lambda: v.get("decay_lambda")?.as_f64()?,
            cluster_radius: v.get("cluster_radius")?.as_f64()? as f32,
        })
    }
}

/// Apply evolved cognitive parameters to AgencyConfig.
///
/// Modifies growth, decay, and cognitive params in-place.
/// Returns true if parameters were applied.
pub fn apply_evolved_cog(
    storage: &GraphStorage,
    config: &mut AgencyConfig,
) -> bool {
    let json = match load_genome(storage, META_COG_GENOME) {
        Some(j) => j,
        None => return false,
    };

    let params = match EvolvedCogParams::from_json(&json) {
        Some(p) => p,
        None => return false,
    };

    // Apply to config fields
    config.growth_distance_threshold = params.growth_distance as f64;
    config.temporal_decay_lambda = params.decay_lambda;
    config.cognitive_cluster_radius = params.cluster_radius as f64;
    config.thermo_conductivity = params.thermal_conductivity as f64;

    tracing::info!(
        growth_dist = params.growth_distance,
        decay_lambda = params.decay_lambda,
        cluster_radius = params.cluster_radius,
        "NietzscheEvolve: applied evolved cognitive params from CF_META"
    );

    true
}

// ── Apply evolved HNSW parameters ────────────────────────────

/// Parsed HNSW genome values.
#[derive(Debug, Clone)]
pub struct EvolvedHnswParams {
    pub ef_search: usize,
    pub ef_construction: usize,
    pub m: usize,
}

impl EvolvedHnswParams {
    pub fn from_json(json: &str) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_str(json).ok()?;
        Some(Self {
            ef_search: v.get("ef_search")?.as_u64()? as usize,
            ef_construction: v.get("ef_construction")?.as_u64()? as usize,
            m: v.get("m")?.as_u64()? as usize,
        })
    }
}

/// Apply evolved HNSW parameters to a GlobalConfig.
///
/// This updates the atomic values that the HNSW search reads at query time.
/// Changes take effect immediately for new queries — no restart needed.
pub fn apply_evolved_hnsw(
    storage: &GraphStorage,
    global_config: &nietzsche_core::GlobalConfig,
) -> bool {
    let json = match load_genome(storage, META_HNSW_GENOME) {
        Some(j) => j,
        None => return false,
    };

    let params = match EvolvedHnswParams::from_json(&json) {
        Some(p) => p,
        None => return false,
    };

    global_config.set_ef_search(params.ef_search);
    global_config.set_ef_construction(params.ef_construction);
    global_config.set_m(params.m);

    tracing::info!(
        ef_search = params.ef_search,
        ef_construction = params.ef_construction,
        m = params.m,
        "NietzscheEvolve: applied evolved HNSW params to GlobalConfig"
    );

    true
}

// ── Unified apply-all ────────────────────────────────────────

/// Apply all evolved parameters from CF_META to the running config.
///
/// Called once at startup and periodically (e.g., every 100 ticks).
/// Safe to call repeatedly — only overwrites if genome exists in CF_META.
pub fn apply_all_evolved_params(
    storage: &GraphStorage,
    config: &mut AgencyConfig,
) -> usize {
    let mut applied = 0;
    if apply_evolved_energy(storage, config) { applied += 1; }
    if apply_evolved_cog(storage, config) { applied += 1; }
    applied
}

// ── Persist from evolver results ─────────────────────────────

/// Persist an EnergyGenome to CF_META.
pub fn persist_energy_genome(
    storage: &GraphStorage,
    genome: &crate::energy_evolve::EnergyGenome,
) -> Result<(), String> {
    let json = serde_json::json!({
        "decay_rate": genome.decay_rate,
        "boost_on_access": genome.boost_on_access,
        "hub_bonus_factor": genome.hub_bonus_factor,
        "depth_scaling": genome.depth_scaling,
        "coherence_bonus": genome.coherence_bonus,
        "thermal_conductivity": genome.thermal_conductivity,
        "generation": genome.generation,
        "fitness": genome.fitness,
    });
    persist_genome(storage, META_ENERGY_GENOME, &json.to_string())
}

/// Persist a CognitiveGenome to CF_META.
pub fn persist_cog_genome(
    storage: &GraphStorage,
    genome: &crate::cog_evolve::CognitiveGenome,
) -> Result<(), String> {
    persist_genome(storage, META_COG_GENOME, &genome.to_json())
}

/// Persist HNSW parameters to CF_META.
pub fn persist_hnsw_genome(
    storage: &GraphStorage,
    ef_search: usize,
    ef_construction: usize,
    m: usize,
    fitness: f64,
) -> Result<(), String> {
    let json = serde_json::json!({
        "ef_search": ef_search,
        "ef_construction": ef_construction,
        "m": m,
        "fitness": fitness,
    });
    persist_genome(storage, META_HNSW_GENOME, &json.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_params_parse() {
        let json = r#"{"decay_rate":0.02,"boost_on_access":0.05,"hub_bonus_factor":0.1,"depth_scaling":1.0,"coherence_bonus":0.05,"thermal_conductivity":0.07}"#;
        let params = EvolvedEnergyParams::from_json(json).unwrap();
        assert!((params.decay_rate - 0.02).abs() < 0.001);
        assert!((params.thermal_conductivity - 0.07).abs() < 0.001);
    }

    #[test]
    fn test_cog_params_parse() {
        let json = r#"{"thermal_conductivity":0.05,"growth_distance":1.2,"decay_lambda":0.0000005,"cluster_radius":0.25}"#;
        let params = EvolvedCogParams::from_json(json).unwrap();
        assert!((params.growth_distance - 1.2).abs() < 0.01);
    }

    #[test]
    fn test_hnsw_params_parse() {
        let json = r#"{"ef_search":200,"ef_construction":150,"m":24,"fitness":0.85}"#;
        let params = EvolvedHnswParams::from_json(json).unwrap();
        assert_eq!(params.ef_search, 200);
        assert_eq!(params.m, 24);
    }

    #[test]
    fn test_invalid_json_returns_none() {
        assert!(EvolvedEnergyParams::from_json("not json").is_none());
        assert!(EvolvedCogParams::from_json("{}").is_none());
        assert!(EvolvedHnswParams::from_json("").is_none());
    }
}
