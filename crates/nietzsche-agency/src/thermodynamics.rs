//! Phase XIII — Cognitive Thermodynamics for NietzscheDB.
//!
//! Formalises the knowledge graph as a **thermodynamic system** where energy,
//! entropy, and temperature govern emergent behaviour.
//!
//! ## Core Concepts
//!
//! ### 1. Cognitive Temperature (T)
//!
//! ```text
//! T = σ_E / mean_E   (coefficient of variation of energy distribution)
//! ```
//!
//! - **High T** → chaotic, high variance, lots of exploration
//! - **Low T** → crystallised, uniform energy, pure exploitation
//! - **Optimal T** → liquid state, balanced self-organisation
//!
//! ### 2. Shannon Entropy (S)
//!
//! ```text
//! S = −Σ p_i · ln(p_i)   where p_i = energy_i / total_energy
//! ```
//!
//! Measures disorder in the energy distribution. High entropy = energy
//! spread uniformly. Low entropy = concentrated in a few hubs.
//!
//! ### 3. Helmholtz Free Energy (F)
//!
//! ```text
//! F = U − T·S   where U = mean_energy
//! ```
//!
//! The system **minimises free energy** to self-organise. This is the
//! variational free energy principle (Friston): the graph seeks states
//! that minimise surprise while maintaining complexity.
//!
//! ### 4. Phase State Classification
//!
//! | Phase    | Temperature  | Entropy  | Behaviour                     |
//! |----------|-------------|----------|-------------------------------|
//! | Solid    | T < T_cold  | Low      | Rigid, no exploration         |
//! | Liquid   | T_cold..T_hot| Medium  | Balanced, self-organising     |
//! | Gas      | T > T_hot   | High     | Chaotic, fragmented attention |
//! | Critical | T ≈ T_c     | Peak     | Phase transition, emergent    |
//!
//! ### 5. Heat Flow (Fourier's Law)
//!
//! Energy flows from hot (high energy) to cold (low energy) regions:
//!
//! ```text
//! q_{ij} = κ · (E_i − E_j) / d_{ij}
//! ```
//!
//! where κ = thermal conductivity, d = graph distance.
//! This creates thermal equilibrium over time.
//!
//! ### 6. Entropy Production Rate
//!
//! ```text
//! dS/dt = S(t) − S(t−1)
//! ```
//!
//! - dS/dt > 0 → system becoming more disordered (Second Law)
//! - dS/dt < 0 → local order creation (attention creating structure)
//! - dS/dt ≈ 0 → equilibrium
//!
//! ## Integration with ECAN (Phase XII)
//!
//! Temperature modulates the explore/exploit ratio:
//!
//! ```text
//! effective_explore_ratio = base_ratio × T / T_optimal
//! ```
//!
//! High temperature → more exploration (the system is "hot" and restless).
//! Low temperature → more exploitation (the system is "cold" and settled).

use uuid::Uuid;

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

/// Configuration for cognitive thermodynamics.
#[derive(Debug, Clone)]
pub struct ThermodynamicsConfig {
    /// Cold threshold: below this temperature, the system is "solid" (too rigid).
    pub t_cold: f64,
    /// Hot threshold: above this temperature, the system is "gas" (too chaotic).
    pub t_hot: f64,
    /// Thermal conductivity κ for heat flow between connected nodes.
    pub thermal_conductivity: f64,
    /// Maximum heat flow per edge per tick (prevents energy teleportation).
    pub max_heat_flow: f32,
    /// Maximum nodes to scan for thermodynamic computation.
    pub max_scan: usize,
    /// Tick interval for thermodynamic analysis (0 = disabled).
    pub interval: u64,
    /// Whether to generate heat flow intents (energy redistribution).
    pub enable_heat_flow: bool,
}

impl Default for ThermodynamicsConfig {
    fn default() -> Self {
        Self {
            t_cold: 0.15,
            t_hot: 0.85,
            thermal_conductivity: 0.05,
            max_heat_flow: 0.02,
            max_scan: 10_000,
            interval: 5,    // every 5 agency ticks
            enable_heat_flow: true,
        }
    }
}

/// Thermodynamic phase classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseState {
    /// T < T_cold: rigid, crystallised, no exploration.
    Solid,
    /// T_cold ≤ T ≤ T_hot: balanced, self-organising, optimal cognition.
    Liquid,
    /// T > T_hot: chaotic, fragmented attention, too much exploration.
    Gas,
    /// T near phase boundary: emerging structure, critical phenomena.
    Critical,
}

impl std::fmt::Display for PhaseState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhaseState::Solid => write!(f, "solid"),
            PhaseState::Liquid => write!(f, "liquid"),
            PhaseState::Gas => write!(f, "gas"),
            PhaseState::Critical => write!(f, "critical"),
        }
    }
}

/// A heat flow proposal: transfer energy between connected nodes.
#[derive(Debug, Clone)]
pub struct HeatFlow {
    /// Node losing energy (hotter).
    pub from: Uuid,
    /// Node gaining energy (colder).
    pub to: Uuid,
    /// Energy amount to transfer.
    pub amount: f32,
}

/// Full thermodynamic report from one analysis cycle.
#[derive(Debug, Clone)]
pub struct ThermodynamicReport {
    /// Number of nodes analysed.
    pub nodes_analysed: usize,
    /// Total energy in the system (U).
    pub total_energy: f64,
    /// Mean energy per node.
    pub mean_energy: f64,
    /// Energy standard deviation.
    pub energy_std: f64,
    /// Cognitive temperature: T = σ_E / mean_E.
    pub temperature: f64,
    /// Shannon entropy of energy distribution.
    pub entropy: f64,
    /// Helmholtz free energy: F = U − T·S.
    pub free_energy: f64,
    /// Entropy production rate: dS/dt = S(t) − S(t−1).
    pub entropy_rate: f64,
    /// Phase state classification.
    pub phase: PhaseState,
    /// Heat flow proposals (energy redistribution from hot → cold).
    pub heat_flows: Vec<HeatFlow>,
    /// Temperature-adjusted exploration ratio modifier.
    /// Multiply the ECAN explore_ratio by this value.
    pub explore_modifier: f64,
}

/// Persistent thermodynamic state across ticks.
pub struct ThermodynamicState {
    tick_count: u64,
    /// Previous tick's entropy (for rate computation).
    last_entropy: Option<f64>,
    /// Previous tick's temperature (for trend detection).
    last_temperature: Option<f64>,
    /// Running average temperature (EMA for smoothing).
    ema_temperature: f64,
}

impl ThermodynamicState {
    pub fn new() -> Self {
        Self {
            tick_count: 0,
            last_entropy: None,
            last_temperature: None,
            ema_temperature: 0.5, // initial neutral
        }
    }

    /// Get current EMA temperature.
    pub fn ema_temperature(&self) -> f64 {
        self.ema_temperature
    }
}

// ─────────────────────────────────────────────
// Core Functions
// ─────────────────────────────────────────────

/// Compute cognitive temperature from energy distribution.
///
/// `T = σ_E / mean_E` (coefficient of variation)
///
/// Returns 0.0 if mean_energy ≈ 0 (dead graph).
#[inline]
pub fn cognitive_temperature(mean_energy: f64, energy_std: f64) -> f64 {
    if mean_energy < 1e-9 {
        return 0.0;
    }
    energy_std / mean_energy
}

/// Compute Shannon entropy of the energy distribution.
///
/// `S = −Σ p_i · ln(p_i)` where `p_i = energy_i / total_energy`
///
/// Returns 0.0 for empty or zero-energy graphs.
pub fn shannon_entropy(energies: &[f32]) -> f64 {
    if energies.is_empty() {
        return 0.0;
    }

    let total: f64 = energies.iter().map(|&e| e.max(0.0) as f64).sum();
    if total < 1e-12 {
        return 0.0;
    }

    let mut entropy = 0.0f64;
    for &e in energies {
        let p = e.max(0.0) as f64 / total;
        if p > 1e-15 {
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Maximum possible Shannon entropy for N nodes (uniform distribution).
///
/// `S_max = ln(N)`
#[inline]
pub fn max_entropy(n: usize) -> f64 {
    if n <= 1 { 0.0 } else { (n as f64).ln() }
}

/// Normalised entropy: S / S_max ∈ [0, 1].
///
/// 0.0 = all energy concentrated in one node.
/// 1.0 = perfectly uniform distribution.
#[inline]
pub fn normalised_entropy(entropy: f64, n: usize) -> f64 {
    let s_max = max_entropy(n);
    if s_max < 1e-12 { 0.0 } else { (entropy / s_max).clamp(0.0, 1.0) }
}

/// Compute Helmholtz free energy.
///
/// `F = U − T·S`
///
/// Where U = mean_energy (internal energy per node),
/// T = cognitive temperature, S = Shannon entropy.
///
/// The system seeks to **minimise** F: this means either:
/// - Lowering internal energy (reducing mean_energy)
/// - Increasing entropy at constant T (spreading energy more uniformly)
/// - Finding the optimal T×S trade-off
#[inline]
pub fn helmholtz_free_energy(mean_energy: f64, temperature: f64, entropy: f64) -> f64 {
    mean_energy - temperature * entropy
}

/// Classify the phase state from temperature.
pub fn classify_phase(temperature: f64, config: &ThermodynamicsConfig) -> PhaseState {
    let margin = (config.t_hot - config.t_cold) * 0.1;

    if temperature < config.t_cold {
        PhaseState::Solid
    } else if temperature > config.t_hot {
        PhaseState::Gas
    } else if (temperature - config.t_cold).abs() < margin
        || (temperature - config.t_hot).abs() < margin
    {
        PhaseState::Critical
    } else {
        PhaseState::Liquid
    }
}

/// Compute temperature-adjusted exploration modifier.
///
/// In the liquid phase (optimal), modifier ≈ 1.0.
/// In solid phase, modifier > 1.0 (force more exploration to "melt").
/// In gas phase, modifier < 1.0 (reduce exploration to "cool down").
///
/// ```text
/// modifier = clamp(T_optimal / T, 0.2, 3.0)
/// ```
///
/// where T_optimal = midpoint of [T_cold, T_hot].
pub fn exploration_modifier(temperature: f64, config: &ThermodynamicsConfig) -> f64 {
    let t_optimal = (config.t_cold + config.t_hot) / 2.0;
    if temperature < 1e-9 {
        return 3.0; // dead system → max exploration
    }
    (t_optimal / temperature).clamp(0.2, 3.0)
}

/// Compute heat flows between connected nodes (Fourier's law on graphs).
///
/// For each edge (i, j):
/// ```text
/// q = κ · (E_i − E_j)   (simplified, no distance denominator on graph)
/// ```
///
/// Energy flows from hot to cold. Capped by max_heat_flow.
///
/// `edges`: list of (node_a, node_b, energy_a, energy_b).
pub fn compute_heat_flows(
    edges: &[(Uuid, Uuid, f32, f32)],
    config: &ThermodynamicsConfig,
) -> Vec<HeatFlow> {
    let kappa = config.thermal_conductivity as f32;
    let max_flow = config.max_heat_flow;

    let mut flows = Vec::new();

    for &(a, b, ea, eb) in edges {
        let diff = ea - eb;
        if diff.abs() < 1e-4 {
            continue; // near equilibrium, skip
        }

        let raw_flow = kappa * diff;
        let clamped = raw_flow.clamp(-max_flow, max_flow);

        if clamped > 0.0 {
            // Energy flows from a → b
            flows.push(HeatFlow {
                from: a,
                to: b,
                amount: clamped,
            });
        } else if clamped < 0.0 {
            // Energy flows from b → a
            flows.push(HeatFlow {
                from: b,
                to: a,
                amount: -clamped,
            });
        }
    }

    flows
}

/// Run a full thermodynamic analysis cycle.
///
/// Takes energy samples and edge energy pairs, returns a complete
/// thermodynamic report with temperature, entropy, free energy,
/// phase state, and heat flow proposals.
pub fn run_thermodynamic_cycle(
    state: &mut ThermodynamicState,
    energies: &[f32],
    edge_energies: &[(Uuid, Uuid, f32, f32)],
    config: &ThermodynamicsConfig,
) -> ThermodynamicReport {
    let n = energies.len();

    if n == 0 {
        return ThermodynamicReport {
            nodes_analysed: 0,
            total_energy: 0.0,
            mean_energy: 0.0,
            energy_std: 0.0,
            temperature: 0.0,
            entropy: 0.0,
            free_energy: 0.0,
            entropy_rate: 0.0,
            phase: PhaseState::Solid,
            heat_flows: Vec::new(),
            explore_modifier: 1.0,
        };
    }

    // ── Energy statistics ──────────────────────
    let total_energy: f64 = energies.iter().map(|&e| e as f64).sum();
    let mean_energy = total_energy / n as f64;
    let variance: f64 = energies
        .iter()
        .map(|&e| {
            let d = e as f64 - mean_energy;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let energy_std = variance.sqrt();

    // ── Temperature ────────────────────────────
    let temperature = cognitive_temperature(mean_energy, energy_std);

    // Update EMA temperature (smoothing factor α = 0.2)
    state.ema_temperature = 0.8 * state.ema_temperature + 0.2 * temperature;

    // ── Entropy ────────────────────────────────
    let entropy = shannon_entropy(energies);

    // ── Entropy rate ───────────────────────────
    let entropy_rate = match state.last_entropy {
        Some(prev) => entropy - prev,
        None => 0.0,
    };

    // ── Free energy ────────────────────────────
    let free_energy = helmholtz_free_energy(mean_energy, temperature, entropy);

    // ── Phase classification ───────────────────
    let phase = classify_phase(temperature, config);

    // ── Exploration modifier ───────────────────
    let explore_modifier = exploration_modifier(temperature, config);

    // ── Heat flow ──────────────────────────────
    let heat_flows = if config.enable_heat_flow {
        compute_heat_flows(edge_energies, config)
    } else {
        Vec::new()
    };

    // ── Update state ───────────────────────────
    state.last_entropy = Some(entropy);
    state.last_temperature = Some(temperature);
    state.tick_count += 1;

    let report = ThermodynamicReport {
        nodes_analysed: n,
        total_energy,
        mean_energy,
        energy_std,
        temperature,
        entropy,
        free_energy,
        entropy_rate,
        phase,
        heat_flows,
        explore_modifier,
    };

    tracing::info!(
        nodes = n,
        T = format!("{:.4}", temperature),
        S = format!("{:.4}", entropy),
        F = format!("{:.4}", free_energy),
        dS = format!("{:.4}", entropy_rate),
        phase = %report.phase,
        modifier = format!("{:.2}", explore_modifier),
        heat_flows = report.heat_flows.len(),
        "thermodynamic cycle"
    );

    report
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_coefficient_of_variation() {
        // Uniform energy: std ≈ 0, T ≈ 0
        let t_uniform = cognitive_temperature(0.5, 0.0);
        assert!(t_uniform < 0.01, "uniform energy → low T: {t_uniform}");

        // High variance: T > 0
        let t_varied = cognitive_temperature(0.5, 0.4);
        assert!(t_varied > 0.5, "high variance → high T: {t_varied}");
    }

    #[test]
    fn temperature_zero_for_dead_graph() {
        assert_eq!(cognitive_temperature(0.0, 0.0), 0.0);
    }

    #[test]
    fn entropy_uniform_is_maximum() {
        // 10 nodes, all energy = 1.0 → maximum entropy
        let energies: Vec<f32> = vec![1.0; 10];
        let s = shannon_entropy(&energies);
        let s_max = max_entropy(10);
        assert!(
            (s - s_max).abs() < 0.01,
            "uniform dist should give max entropy: s={s}, s_max={s_max}"
        );
    }

    #[test]
    fn entropy_concentrated_is_low() {
        // 10 nodes, one has all energy
        let mut energies = vec![0.0f32; 10];
        energies[0] = 1.0;
        let s = shannon_entropy(&energies);
        assert!(s < 0.01, "concentrated energy → low entropy: {s}");
    }

    #[test]
    fn entropy_empty_is_zero() {
        assert_eq!(shannon_entropy(&[]), 0.0);
        assert_eq!(shannon_entropy(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn normalised_entropy_bounds() {
        let energies: Vec<f32> = vec![1.0; 100];
        let s = shannon_entropy(&energies);
        let ns = normalised_entropy(s, 100);
        assert!(
            (ns - 1.0).abs() < 0.01,
            "uniform → normalised ≈ 1.0: {ns}"
        );

        let mut conc = vec![0.0f32; 100];
        conc[0] = 1.0;
        let s2 = shannon_entropy(&conc);
        let ns2 = normalised_entropy(s2, 100);
        assert!(ns2 < 0.01, "concentrated → normalised ≈ 0.0: {ns2}");
    }

    #[test]
    fn free_energy_decreases_with_entropy() {
        let f1 = helmholtz_free_energy(0.5, 0.3, 1.0);
        let f2 = helmholtz_free_energy(0.5, 0.3, 2.0);
        assert!(f2 < f1, "higher entropy → lower free energy");
    }

    #[test]
    fn phase_classification() {
        let config = ThermodynamicsConfig::default();
        assert_eq!(classify_phase(0.05, &config), PhaseState::Solid);
        assert_eq!(classify_phase(0.50, &config), PhaseState::Liquid);
        assert_eq!(classify_phase(1.50, &config), PhaseState::Gas);
    }

    #[test]
    fn critical_near_boundary() {
        let config = ThermodynamicsConfig {
            t_cold: 0.2,
            t_hot: 0.8,
            ..Default::default()
        };
        // Within 10% of boundary → critical
        let phase = classify_phase(0.21, &config);
        assert_eq!(phase, PhaseState::Critical);
    }

    #[test]
    fn exploration_modifier_varies_with_temperature() {
        let config = ThermodynamicsConfig::default();

        // Cold → high modifier (explore more to melt)
        let mod_cold = exploration_modifier(0.05, &config);
        assert!(mod_cold > 1.5, "cold → high explore modifier: {mod_cold}");

        // Hot → low modifier (explore less to cool)
        let mod_hot = exploration_modifier(2.0, &config);
        assert!(mod_hot < 0.5, "hot → low explore modifier: {mod_hot}");

        // Optimal → ≈ 1.0
        let t_opt = (config.t_cold + config.t_hot) / 2.0;
        let mod_opt = exploration_modifier(t_opt, &config);
        assert!(
            (mod_opt - 1.0).abs() < 0.01,
            "optimal → modifier ≈ 1.0: {mod_opt}"
        );
    }

    #[test]
    fn heat_flow_hot_to_cold() {
        let config = ThermodynamicsConfig::default();
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);

        let edges = vec![(a, b, 0.9, 0.1)]; // a is hot, b is cold
        let flows = compute_heat_flows(&edges, &config);

        assert_eq!(flows.len(), 1);
        assert_eq!(flows[0].from, a, "energy should flow from hot a");
        assert_eq!(flows[0].to, b, "energy should flow to cold b");
        assert!(flows[0].amount > 0.0);
    }

    #[test]
    fn heat_flow_capped() {
        let config = ThermodynamicsConfig {
            thermal_conductivity: 1.0, // very high κ
            max_heat_flow: 0.01,       // but capped
            ..Default::default()
        };

        let edges = vec![(Uuid::from_u128(1), Uuid::from_u128(2), 1.0, 0.0)];
        let flows = compute_heat_flows(&edges, &config);

        assert_eq!(flows.len(), 1);
        assert!(
            flows[0].amount <= 0.01 + 1e-6,
            "flow should be capped: {}",
            flows[0].amount
        );
    }

    #[test]
    fn heat_flow_near_equilibrium_skipped() {
        let config = ThermodynamicsConfig::default();
        let edges = vec![(Uuid::from_u128(1), Uuid::from_u128(2), 0.5, 0.5)];
        let flows = compute_heat_flows(&edges, &config);
        assert!(flows.is_empty(), "equilibrium → no heat flow");
    }

    #[test]
    fn full_cycle_produces_report() {
        let mut state = ThermodynamicState::new();
        let config = ThermodynamicsConfig::default();
        let energies: Vec<f32> = (0..20).map(|i| 0.1 + (i as f32) * 0.04).collect();
        let edges = vec![
            (Uuid::from_u128(0), Uuid::from_u128(19), 0.1, 0.86),
        ];

        let report = run_thermodynamic_cycle(&mut state, &energies, &edges, &config);

        assert_eq!(report.nodes_analysed, 20);
        assert!(report.total_energy > 0.0);
        assert!(report.temperature > 0.0);
        assert!(report.entropy > 0.0);
        assert!(report.entropy_rate == 0.0, "first tick → no rate");
    }

    #[test]
    fn entropy_rate_computed_across_ticks() {
        let mut state = ThermodynamicState::new();
        let config = ThermodynamicsConfig::default();

        // Tick 1: uniform → high entropy
        let e1: Vec<f32> = vec![0.5; 20];
        let r1 = run_thermodynamic_cycle(&mut state, &e1, &[], &config);
        assert_eq!(r1.entropy_rate, 0.0, "first tick → rate = 0");

        // Tick 2: concentrated → low entropy → negative rate
        let mut e2 = vec![0.01f32; 20];
        e2[0] = 0.95;
        let r2 = run_thermodynamic_cycle(&mut state, &e2, &[], &config);
        assert!(r2.entropy_rate < 0.0, "entropy decreased → negative rate: {}", r2.entropy_rate);
    }

    #[test]
    fn empty_graph_no_crash() {
        let mut state = ThermodynamicState::new();
        let config = ThermodynamicsConfig::default();
        let report = run_thermodynamic_cycle(&mut state, &[], &[], &config);
        assert_eq!(report.nodes_analysed, 0);
        assert_eq!(report.phase, PhaseState::Solid);
    }

    #[test]
    fn ema_temperature_smooths() {
        let mut state = ThermodynamicState::new();
        let config = ThermodynamicsConfig::default();

        // Several ticks with varying temperature
        for i in 0..10 {
            let energies: Vec<f32> = (0..20)
                .map(|j| if j % 2 == 0 { 0.1 * (i as f32 + 1.0) } else { 0.05 })
                .collect();
            run_thermodynamic_cycle(&mut state, &energies, &[], &config);
        }

        // EMA should be between 0 and the last raw temperature
        assert!(state.ema_temperature > 0.0, "EMA should be positive");
        assert!(state.ema_temperature < 10.0, "EMA should be bounded");
    }
}
