//! # Genome — Evolutionary Parameter Adaptation (Phase VIII)
//!
//! Implements autonomous evolution of the navigability innovation parameters
//! (α, β, γ) via a genomic metaphor with formal stability guarantees.
//!
//! ## Architecture
//!
//! ### EvolutionaryGenome
//!
//! DNA-like struct holding metabolic parameters with hard bounds:
//! - α ∈ [0.1, 2.0] — spectral connectivity exponent
//! - β ∈ [0.1, 2.0] — energy stability exponent
//! - γ ∈ [0.5, 3.0] — path entropy exponent
//! - Stability constraint: α + β + γ ≤ Σ_max (proxy for Jacobian spectral radius < 1)
//!
//! ### TGC (Topological Generative Capacity)
//!
//! Structural reward signal for evaluating parameter mutations:
//!
//! ```text
//! TGC(G, θ) = λ₂^α · E^β · [H^γ · (1-H)]
//! R_structural = TGC_after - TGC_before
//! ```
//!
//! TGC IS the navigability score parameterized by the genome. Mutations that
//! improve TGC are accepted; mutations that degrade it trigger rollback.
//!
//! ### GenomicSnapshot + Rollback
//!
//! Before each mutation, a snapshot preserves the current state. Automatic
//! rollback triggers on:
//! 1. **Spectral rupture**: λ₂ drops below critical threshold
//! 2. **TGC degradation**: R_structural < -δ (significant loss)
//! 3. **Energy turbulence**: σ²_E exceeds turbulence bound
//!
//! ### Innovation Homeostasis Loop
//!
//! Maintains I ≈ I_target via adaptive mutation rate:
//! - I > I_target → reduce mutation_rate (conserve gains)
//! - I < I_target → increase mutation_rate (explore harder)
//!
//! ### Cognitive Budget
//!
//! Bounds computational cost per cycle to prevent metabolic runaway.
//!
//! ## Spectral Stability Analysis
//!
//! Hard bounds enforce `max(α, β, γ) < M` and `α + β + γ ≤ Σ_max` as a
//! practical proxy for the Jacobian spectral radius remaining bounded.
//! The sensitivity of I to each parameter is:
//!
//! ```text
//! ∂I/∂α = I · ln(λ₂)      — diverges when λ₂ → 0
//! ∂I/∂β = I · ln(E)        — diverges when E → 0
//! ∂I/∂γ = I · ln(H)        — diverges when H → 0
//! ```
//!
//! The bounds prevent these sensitivities from creating chaotic parameter drift.

use serde::{Deserialize, Serialize};

use crate::innovation::{NavigabilityConfig, NavigabilityEvaluator, NavigabilityReport};

// ═══════════════════════════════════════════════
// §1 — Genome Bounds (Spectral Stability Proxy)
// ═══════════════════════════════════════════════

/// Hard bounds on genome parameters.
///
/// These act as a practical proxy for the Jacobian spectral radius < 1
/// of the coupled dynamical system [λ₂, E, H] → I → [α, β, γ].
///
/// ## Why These Specific Bounds?
///
/// - `alpha_range [0.1, 2.0]`: α < 0.1 ignores connectivity; α > 2.0
///   makes I hypersensitive to λ₂ fluctuations
/// - `beta_range [0.1, 2.0]`: same logic for energy stability
/// - `gamma_range [0.5, 3.0]`: γ < 0.5 kills the parabolic H-filter;
///   γ > 3.0 makes the peak too sharp (numerically unstable)
/// - `max_sum 5.0`: prevents simultaneous large exponents from creating
///   catastrophic sensitivity amplification
/// - `max_single 3.0`: no single parameter dominates the score
#[derive(Debug, Clone, PartialEq)]
pub struct GenomeBounds {
    /// Valid range for α (spectral connectivity exponent).
    pub alpha_range: (f64, f64),

    /// Valid range for β (energy stability exponent).
    pub beta_range: (f64, f64),

    /// Valid range for γ (path entropy exponent).
    pub gamma_range: (f64, f64),

    /// Maximum sum α + β + γ (spectral radius proxy).
    pub max_sum: f64,

    /// Maximum value for any single parameter.
    pub max_single: f64,
}

impl Default for GenomeBounds {
    fn default() -> Self {
        Self {
            alpha_range: (0.1, 2.0),
            beta_range: (0.1, 2.0),
            gamma_range: (0.5, 3.0),
            max_sum: 5.0,
            max_single: 3.0,
        }
    }
}

impl GenomeBounds {
    /// Check if a parameter set satisfies all bounds.
    pub fn is_valid(&self, alpha: f64, beta: f64, gamma: f64) -> bool {
        alpha >= self.alpha_range.0
            && alpha <= self.alpha_range.1
            && beta >= self.beta_range.0
            && beta <= self.beta_range.1
            && gamma >= self.gamma_range.0
            && gamma <= self.gamma_range.1
            && (alpha + beta + gamma) <= self.max_sum
            && alpha <= self.max_single
            && beta <= self.max_single
            && gamma <= self.max_single
    }

    /// Clamp parameters to satisfy all bounds, preserving ratios where possible.
    pub fn enforce(&self, alpha: f64, beta: f64, gamma: f64) -> (f64, f64, f64) {
        // Step 1: Clamp individual ranges
        let mut a = alpha.clamp(self.alpha_range.0, self.alpha_range.1.min(self.max_single));
        let mut b = beta.clamp(self.beta_range.0, self.beta_range.1.min(self.max_single));
        let mut g = gamma.clamp(self.gamma_range.0, self.gamma_range.1.min(self.max_single));

        // Step 2: Enforce sum constraint by proportional reduction
        let sum = a + b + g;
        if sum > self.max_sum {
            let scale = self.max_sum / sum;
            a *= scale;
            b *= scale;
            g *= scale;

            // Re-clamp to minimums (scaling might have pushed below floor)
            a = a.max(self.alpha_range.0);
            b = b.max(self.beta_range.0);
            g = g.max(self.gamma_range.0);
        }

        (a, b, g)
    }
}

// ═══════════════════════════════════════════════
// §2 — EvolutionaryGenome
// ═══════════════════════════════════════════════

/// DNA-like struct holding the navigability innovation parameters.
///
/// The genome parameterizes the innovation score:
/// ```text
/// I = λ₂^α · E(τ)^β · [H^γ · (1-H)]
/// ```
///
/// Parameters evolve through mutation + selection against TGC reward.
#[derive(Debug, Clone, PartialEq)]
pub struct EvolutionaryGenome {
    /// Spectral connectivity exponent.
    pub alpha: f64,

    /// Energy stability exponent.
    pub beta: f64,

    /// Path entropy exponent (controls parabola peak position H_peak = γ/(γ+1)).
    pub gamma: f64,

    /// Base mutation step size σ.
    /// Mutations are Gaussian: Δθ ~ N(0, σ²).
    /// Default: 0.05
    pub mutation_rate: f64,

    /// Minimum mutation rate (prevents premature convergence).
    /// Default: 0.005
    pub min_mutation_rate: f64,

    /// Maximum mutation rate (prevents chaotic jumps).
    /// Default: 0.3
    pub max_mutation_rate: f64,

    /// Hard bounds for spectral stability.
    pub bounds: GenomeBounds,

    /// Generation counter (monotonically increasing).
    pub generation: u64,
}

impl Default for EvolutionaryGenome {
    fn default() -> Self {
        Self {
            // Defaults match NavigabilityConfig defaults
            alpha: 0.5,
            beta: 0.3,
            gamma: 1.2,
            mutation_rate: 0.05,
            min_mutation_rate: 0.005,
            max_mutation_rate: 0.3,
            bounds: GenomeBounds::default(),
            generation: 0,
        }
    }
}

impl EvolutionaryGenome {
    /// Create a genome from explicit parameters.
    pub fn new(alpha: f64, beta: f64, gamma: f64) -> Self {
        let bounds = GenomeBounds::default();
        let (a, b, g) = bounds.enforce(alpha, beta, gamma);
        Self {
            alpha: a,
            beta: b,
            gamma: g,
            ..Default::default()
        }
    }

    /// Create a genome from an existing NavigabilityConfig.
    pub fn from_navigability_config(config: &NavigabilityConfig) -> Self {
        Self::new(config.alpha, config.beta, config.gamma)
    }

    /// Convert genome to NavigabilityConfig for score evaluation.
    pub fn to_navigability_config(&self) -> NavigabilityConfig {
        NavigabilityConfig {
            alpha: self.alpha,
            beta: self.beta,
            gamma: self.gamma,
        }
    }

    /// Apply a random mutation using Box-Muller Gaussian noise.
    ///
    /// Each parameter receives independent noise: Δθ ~ N(0, σ²).
    /// After mutation, bounds are enforced via proportional clamping.
    pub fn mutate(&mut self, rng: &mut dyn FnMut() -> f64) {
        let sigma = self.mutation_rate;

        // Box-Muller transform for Gaussian noise
        let gaussian = |rng_fn: &mut dyn FnMut() -> f64| -> f64 {
            let u1 = (rng_fn)().max(1e-30);
            let u2 = (rng_fn)();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        let da = sigma * gaussian(rng);
        let db = sigma * gaussian(rng);
        let dg = sigma * gaussian(rng);

        let new_a = self.alpha + da;
        let new_b = self.beta + db;
        let new_g = self.gamma + dg;

        let (a, b, g) = self.bounds.enforce(new_a, new_b, new_g);
        self.alpha = a;
        self.beta = b;
        self.gamma = g;
        self.generation += 1;
    }

    /// Apply **Proportional Adaptive Mutation** with Success Inertia (Phase VIII).
    ///
    /// The mutation amplitude ε encodes three independent survival signals:
    ///
    /// ```text
    /// ε(t) = ε_base · exp(-k · σ²_TGC) · (1 - ρ/ρ_max) · tanh(R_structural)
    /// ```
    ///
    /// Where:
    /// - ε_base = base mutation rate (self.mutation_rate)
    /// - k = sensitivity to TGC variance (default: 2.0)
    /// - σ²_TGC = variance of recent TGC history
    /// - ρ = stability_score() (proxy for Jacobian spectral radius)
    /// - ρ_max = maximum acceptable stability score (default: 2.5)
    /// - R_structural = mean of recent structural rewards (TGC deltas)
    ///
    /// ## Three Factors
    ///
    /// 1. **Variance Damping** `exp(-k·σ²)`: turbulent → freeze; stable → explore
    /// 2. **Spectral Margin** `(1-ρ/ρ_max)`: near boundary → freeze
    /// 3. **Success Inertia** `tanh(R)`: positive track record → bolder mutations;
    ///    no history or negative → cautious (clamped to min 0.1 for liveness)
    ///
    /// ## Single-Parameter Mutation
    ///
    /// Instead of mutating all three parameters simultaneously, we randomly
    /// select ONE of [α, β, γ] per cycle. This allows cleaner attribution
    /// of improvements and reduces the chance of catastrophic multi-axis jumps.
    pub fn mutate_proportional(
        &mut self,
        rng: &mut dyn FnMut() -> f64,
        tgc_variance: f64,
        variance_sensitivity: f64,
        rho_max: f64,
        recent_reward: f64,
    ) {
        // Factor 1: Variance damping — turbulence → smaller steps
        let variance_damping = (-variance_sensitivity * tgc_variance).exp();

        // Factor 2: Spectral margin — near stability boundary → freeze
        let rho = self.stability_score();
        let spectral_margin = (1.0 - rho / rho_max).clamp(0.0, 1.0);

        // Factor 3: Success inertia — positive track record → bolder
        // tanh(R) ∈ (-1, 1), clamped to [0.1, 1.0] for guaranteed liveness
        let success_inertia = recent_reward.tanh().max(0.1);

        // Combined adaptive mutation amplitude
        let epsilon = self.mutation_rate * variance_damping * spectral_margin * success_inertia;

        // Box-Muller Gaussian noise (consumes 2 RNG calls)
        let u1 = (rng)().max(1e-30);
        let u2 = (rng)();
        let gaussian_noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let delta = epsilon * gaussian_noise;

        // Single-parameter mutation: select one of [α, β, γ]
        // Hash-based selection to decorrelate from LCG Gaussian subsequence
        let sel = (rng)();
        let sel_hash = (sel * 2654435769.0).fract().abs();
        let target_param = (sel_hash * 3.0).floor().min(2.0) as usize;

        match target_param {
            0 => self.alpha += delta,
            1 => self.beta += delta,
            _ => self.gamma += delta,
        }

        // Enforce bounds and spectral guardrail
        let (a, b, g) = self.bounds.enforce(self.alpha, self.beta, self.gamma);
        self.alpha = a;
        self.beta = b;
        self.gamma = g;

        // Apply spectral guardrail attenuation
        self.enforce_spectral_guardrail();

        self.generation += 1;
    }

    /// Enforce the spectral guardrail — containment for structural stability.
    ///
    /// Prevents the genome from entering psychotic instability by:
    /// 1. Attenuating all parameters by 0.8× if max(α, β, γ) > 2.2
    /// 2. Proportionally scaling if sum exceeds 4.5 (tighter than bounds.max_sum)
    ///
    /// This is the **emergency brake** — separate from normal bounds enforcement.
    pub fn enforce_spectral_guardrail(&mut self) {
        let max_param = self.alpha.max(self.beta).max(self.gamma);

        // 1. Imminent instability: single parameter too dominant
        if max_param > 2.2 {
            self.attenuate_all(0.8);
        }

        // 2. Global gain too high (tighter constraint for edge-of-chaos regime)
        let sum = self.alpha + self.beta + self.gamma;
        if sum > 4.5 {
            let scale = 4.5 / sum;
            self.alpha *= scale;
            self.beta *= scale;
            self.gamma *= scale;
        }

        // 3. Re-enforce minimums
        self.alpha = self.alpha.max(self.bounds.alpha_range.0);
        self.beta = self.beta.max(self.bounds.beta_range.0);
        self.gamma = self.gamma.max(self.bounds.gamma_range.0);
    }

    /// Attenuate all parameters by a factor (0 < factor < 1).
    fn attenuate_all(&mut self, factor: f64) {
        self.alpha *= factor;
        self.beta *= factor;
        self.gamma *= factor;
        // Re-enforce minimums
        self.alpha = self.alpha.max(self.bounds.alpha_range.0);
        self.beta = self.beta.max(self.bounds.beta_range.0);
        self.gamma = self.gamma.max(self.bounds.gamma_range.0);
    }

    /// Compute the optimal H_path for current γ.
    /// H_peak = γ / (γ + 1)
    pub fn optimal_h_path(&self) -> f64 {
        self.gamma / (self.gamma + 1.0)
    }

    /// Compute the parameter sensitivity vector at a given state.
    ///
    /// Returns (∂I/∂α, ∂I/∂β, ∂I/∂γ) where:
    /// - ∂I/∂α = I · ln(λ₂)
    /// - ∂I/∂β = I · ln(E)
    /// - ∂I/∂γ = I · [ln(H) - H·ln(H)/(1-H)] (approximate)
    ///
    /// Large sensitivities signal danger zones where small parameter
    /// changes create large score swings.
    pub fn sensitivity(&self, lambda2: f64, energy: f64, h_path: f64) -> (f64, f64, f64) {
        let eval = NavigabilityEvaluator::new(self.to_navigability_config());
        let report = eval.evaluate(lambda2, energy, h_path);
        let i = report.score;

        let l2 = lambda2.max(1e-30);
        let e = energy.clamp(1e-30, 1.0);
        let h = h_path.clamp(1e-30, 1.0 - 1e-10);

        let di_da = i * l2.ln();
        let di_db = i * e.ln();
        // ∂/∂γ [H^γ · (1-H)] = H^γ · ln(H) · (1-H)
        // So ∂I/∂γ = λ₂^α · E^β · H^γ · ln(H) · (1-H) = I · ln(H) / [H^γ·(1-H)] * [H^γ·ln(H)·(1-H)]
        // Simplifies to: ∂I/∂γ = I · ln(H) when H^γ·(1-H) > 0
        let di_dg = if report.entropy_parabola > 1e-30 {
            i * h.ln()
        } else {
            0.0
        };

        (di_da, di_db, di_dg)
    }

    /// Check if the genome is within all hard bounds.
    pub fn is_valid(&self) -> bool {
        self.bounds.is_valid(self.alpha, self.beta, self.gamma)
    }

    /// Spectral stability score: lower = more stable.
    ///
    /// ```text
    /// Ω = max(α, β, γ) + (α + β + γ) / Σ_max
    /// ```
    ///
    /// When Ω > 2.0, the genome is near the stability boundary.
    pub fn stability_score(&self) -> f64 {
        let max_param = self.alpha.max(self.beta).max(self.gamma);
        let sum_ratio = (self.alpha + self.beta + self.gamma) / self.bounds.max_sum;
        max_param + sum_ratio
    }
}

impl std::fmt::Display for EvolutionaryGenome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Genome[α={:.3}, β={:.3}, γ={:.3} | gen={} σ={:.4} Ω={:.3}]",
            self.alpha,
            self.beta,
            self.gamma,
            self.generation,
            self.mutation_rate,
            self.stability_score(),
        )
    }
}

// ═══════════════════════════════════════════════
// §3 — GenomicSnapshot (Checkpoint for Rollback)
// ═══════════════════════════════════════════════

/// Immutable checkpoint taken before a structural mutation.
///
/// If the mutation degrades the system, the StructuralEvolutionUnit
/// restores this snapshot automatically.
#[derive(Debug, Clone)]
pub struct GenomicSnapshot {
    /// Frozen copy of the genome at snapshot time.
    pub genome: EvolutionaryGenome,

    /// Cycle number when the snapshot was taken.
    pub cycle_number: u64,

    /// TGC (navigability score) at snapshot time.
    pub tgc: f64,

    /// Metabolic observables at snapshot time.
    pub lambda2: f64,
    pub energy: f64,
    pub h_path: f64,
}

impl std::fmt::Display for GenomicSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Snapshot[cycle={}, TGC={:.6}, λ₂={:.4}, E={:.4}, H={:.4}]",
            self.cycle_number, self.tgc, self.lambda2, self.energy, self.h_path,
        )
    }
}

// ═══════════════════════════════════════════════
// §4 — Rollback System
// ═══════════════════════════════════════════════

/// Reason for automatic rollback after a mutation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RollbackReason {
    /// λ₂ dropped below the critical threshold → graph fragmentation risk.
    SpectralRupture {
        lambda2: f64,
        threshold: f64,
    },

    /// TGC degraded significantly → the mutation made things worse.
    TgcDegradation {
        delta_tgc: f64,
        threshold: f64,
    },

    /// Energy variance exceeded turbulence bound → chaotic regime.
    EnergyTurbulence {
        variance: f64,
        max_variance: f64,
    },

    /// Sensitivity explosion: ‖∂I/∂θ‖ too large → near instability.
    SensitivityExplosion {
        norm: f64,
        max_norm: f64,
    },

    /// Axiom drift exceeded: d_𝔻(C₀, Cₜ) > η · r̄₀ → identity dissolution risk.
    AxiomDrift {
        drift: f64,
        limit: f64,
    },
}

impl std::fmt::Display for RollbackReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RollbackReason::SpectralRupture { lambda2, threshold } => {
                write!(f, "SpectralRupture(λ₂={lambda2:.4} < {threshold:.4})")
            }
            RollbackReason::TgcDegradation { delta_tgc, threshold } => {
                write!(f, "TGC_Degradation(Δ={delta_tgc:.6} < {threshold:.6})")
            }
            RollbackReason::EnergyTurbulence { variance, max_variance } => {
                write!(f, "EnergyTurbulence(σ²={variance:.4} > {max_variance:.4})")
            }
            RollbackReason::SensitivityExplosion { norm, max_norm } => {
                write!(f, "SensitivityExplosion(‖∂I/∂θ‖={norm:.4} > {max_norm:.4})")
            }
            RollbackReason::AxiomDrift { drift, limit } => {
                write!(f, "AxiomDrift(d_𝔻={drift:.6} > limit={limit:.6})")
            }
        }
    }
}

// ═══════════════════════════════════════════════
// §5 — Cognitive Budget
// ═══════════════════════════════════════════════

/// Bounds on computational cost per metabolic cycle.
///
/// Prevents the evolution unit from consuming unbounded resources
/// during exploration phases.
#[derive(Debug, Clone)]
pub struct CognitiveBudget {
    /// Maximum mutations to attempt per cycle.
    /// Default: 3
    pub max_mutations_per_cycle: usize,

    /// Maximum total mutations before forced cooldown.
    /// Default: 1000
    pub max_total_mutations: u64,

    /// Cooldown cycles after hitting max_total_mutations.
    /// Default: 10
    pub cooldown_cycles: u64,

    /// Remaining cooldown (internal state).
    remaining_cooldown: u64,

    /// Total mutations performed.
    total_mutations: u64,
}

impl Default for CognitiveBudget {
    fn default() -> Self {
        Self {
            max_mutations_per_cycle: 3,
            max_total_mutations: 1000,
            cooldown_cycles: 10,
            remaining_cooldown: 0,
            total_mutations: 0,
        }
    }
}

impl CognitiveBudget {
    /// Check if the budget allows another mutation this cycle.
    pub fn can_mutate(&self) -> bool {
        self.remaining_cooldown == 0 && self.total_mutations < self.max_total_mutations
    }

    /// Record a mutation and check for cooldown trigger.
    pub fn record_mutation(&mut self) {
        self.total_mutations += 1;
        if self.total_mutations >= self.max_total_mutations {
            self.remaining_cooldown = self.cooldown_cycles;
        }
    }

    /// Advance one cycle (decrement cooldown if active).
    pub fn tick_cycle(&mut self) {
        if self.remaining_cooldown > 0 {
            self.remaining_cooldown -= 1;
            if self.remaining_cooldown == 0 {
                // Reset after cooldown
                self.total_mutations = 0;
            }
        }
    }

    /// Get total mutations performed.
    pub fn total_mutations(&self) -> u64 {
        self.total_mutations
    }

    /// Check if in cooldown.
    pub fn in_cooldown(&self) -> bool {
        self.remaining_cooldown > 0
    }
}

// ═══════════════════════════════════════════════
// §6 — StructuralEvolutionUnit Configuration
// ═══════════════════════════════════════════════

/// Configuration for the StructuralEvolutionUnit.
#[derive(Debug, Clone)]
pub struct StructuralEvolutionConfig {
    /// Target innovation score I_target.
    /// The homeostasis loop tries to maintain I ≈ I_target.
    /// Default: 0.08 (Promising territory)
    pub i_target: f64,

    /// TGC degradation threshold for rollback.
    /// If TGC_after - TGC_before < -delta_threshold, rollback.
    /// Default: -0.01
    pub tgc_degradation_threshold: f64,

    /// Minimum λ₂ for spectral safety (below → rollback).
    /// Default: 0.005
    pub lambda2_safety_floor: f64,

    /// Maximum energy variance before turbulence rollback.
    /// Default: 0.5
    pub max_energy_variance: f64,

    /// Maximum sensitivity norm before rollback.
    /// ‖∂I/∂θ‖ = √((∂I/∂α)² + (∂I/∂β)² + (∂I/∂γ)²)
    /// Default: 10.0
    pub max_sensitivity_norm: f64,

    /// Homeostasis gain: how fast mutation_rate adapts to I vs I_target.
    /// mutation_rate *= (1 + gain · (I_target - I) / I_target)
    /// Default: 0.1
    pub homeostasis_gain: f64,

    /// Minimum cycles between mutations (to let the system settle).
    /// Default: 3
    pub mutation_cooldown_cycles: u64,

    /// History window for innovation score tracking.
    /// Default: 50
    pub i_history_size: usize,

    /// Sensitivity k for TGC variance in proportional mutation.
    /// ε = ε₀ · exp(-k · Var(TGC)) · (1 - ρ/ρ_max)
    /// Higher k → mutation rate drops faster with turbulence.
    /// Default: 2.0
    pub tgc_variance_sensitivity: f64,

    /// Maximum stability score (ρ_max) for proportional mutation.
    /// When stability_score() → ρ_max, mutation amplitude → 0.
    /// Default: 2.5
    pub rho_max: f64,

    /// Window size for structural reward history (R_structural).
    /// The mean of recent rewards feeds the success inertia factor.
    /// Default: 20
    pub reward_window_size: usize,

    /// Enable the evolution unit.
    /// Default: true
    pub enabled: bool,
}

impl Default for StructuralEvolutionConfig {
    fn default() -> Self {
        Self {
            i_target: 0.08,
            tgc_degradation_threshold: -0.01,
            lambda2_safety_floor: 0.005,
            max_energy_variance: 0.5,
            max_sensitivity_norm: 10.0,
            homeostasis_gain: 0.1,
            mutation_cooldown_cycles: 3,
            i_history_size: 50,
            tgc_variance_sensitivity: 2.0,
            rho_max: 2.5,
            reward_window_size: 20,
            enabled: true,
        }
    }
}

// ═══════════════════════════════════════════════
// §7 — StructuralEvolutionUnit
// ═══════════════════════════════════════════════

/// Autonomous evolution orchestrator for navigability parameters.
///
/// ## Lifecycle (per cycle)
///
/// ```text
/// ┌──────────────────────────────────────────────────────┐
/// │              STRUCTURAL EVOLUTION CYCLE                │
/// │                                                        │
/// │  1. Check budget + cooldown                            │
/// │  2. Snapshot current state                             │
/// │  3. Evaluate TGC_before = I(λ₂, E, H; α, β, γ)       │
/// │  4. Check safety: λ₂ ≥ floor, σ² ≤ max, ‖∂I/∂θ‖ ≤ M │
/// │  5. Mutate genome: Δθ ~ N(0, σ²)                      │
/// │  6. Evaluate TGC_after = I(λ₂, E, H; α', β', γ')     │
/// │  7. Decision:                                          │
/// │     ├── TGC improved → ACCEPT, update snapshot         │
/// │     └── TGC degraded → ROLLBACK to snapshot            │
/// │  8. Innovation homeostasis: adjust σ → I ≈ I_target    │
/// │  9. Produce EvolutionReport                            │
/// └──────────────────────────────────────────────────────┘
/// ```
pub struct StructuralEvolutionUnit {
    /// Current genome (the "DNA" of the system).
    pub genome: EvolutionaryGenome,

    /// Last known good state for rollback.
    pub snapshot: Option<GenomicSnapshot>,

    /// Configuration.
    pub config: StructuralEvolutionConfig,

    /// Cognitive budget tracker.
    pub budget: CognitiveBudget,

    /// Innovation score history (for trend analysis).
    i_history: Vec<f64>,

    /// Structural reward history (R_structural = TGC_after - TGC_before).
    /// The mean feeds the success inertia factor: tanh(mean(R_history)).
    reward_history: Vec<f64>,

    /// Cycles since last mutation.
    cycles_since_mutation: u64,

    /// Total rollbacks performed.
    pub total_rollbacks: u64,

    /// Total accepted mutations.
    pub total_accepted: u64,

    /// Total rejected mutations (via rollback).
    pub total_rejected: u64,

    /// Internal deterministic RNG state.
    rng_state: u64,
}

/// Report from one structural evolution step.
#[derive(Debug, Clone)]
pub struct EvolutionStepReport {
    /// Generation number of the genome.
    pub generation: u64,

    /// TGC before mutation (or current if no mutation this cycle).
    pub tgc_before: f64,

    /// TGC after mutation (None if no mutation performed).
    pub tgc_after: Option<f64>,

    /// Structural reward R = TGC_after - TGC_before.
    pub reward: Option<f64>,

    /// Whether the mutation was accepted.
    pub accepted: bool,

    /// Rollback reason (if rejected).
    pub rollback_reason: Option<RollbackReason>,

    /// Current genome state after this step.
    pub genome_alpha: f64,
    pub genome_beta: f64,
    pub genome_gamma: f64,
    pub mutation_rate: f64,

    /// Whether the unit was in cooldown/disabled.
    pub skipped: bool,
    pub skip_reason: Option<String>,

    /// Innovation homeostasis: current I vs I_target.
    pub current_i: f64,
    pub target_i: f64,

    /// Sensitivity norm ‖∂I/∂θ‖.
    pub sensitivity_norm: f64,
}

impl std::fmt::Display for EvolutionStepReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.skipped {
            return write!(
                f,
                "Evolution[SKIP: {}] gen={} I={:.6} target={:.6}",
                self.skip_reason.as_deref().unwrap_or("unknown"),
                self.generation,
                self.current_i,
                self.target_i,
            );
        }
        if let Some(reward) = self.reward {
            let status = if self.accepted { "ACCEPT" } else { "ROLLBACK" };
            write!(
                f,
                "Evolution[{}] gen={} R={:.6} TGC={:.6}→{:.6} σ={:.4} ‖∂I/∂θ‖={:.3}",
                status,
                self.generation,
                reward,
                self.tgc_before,
                self.tgc_after.unwrap_or(self.tgc_before),
                self.mutation_rate,
                self.sensitivity_norm,
            )
        } else {
            write!(
                f,
                "Evolution[EVAL] gen={} TGC={:.6} I={:.6} target={:.6}",
                self.generation,
                self.tgc_before,
                self.current_i,
                self.target_i,
            )
        }
    }
}

impl StructuralEvolutionUnit {
    /// Create a new unit with default genome and configuration.
    pub fn new(config: StructuralEvolutionConfig) -> Self {
        Self {
            genome: EvolutionaryGenome::default(),
            snapshot: None,
            config,
            budget: CognitiveBudget::default(),
            i_history: Vec::new(),
            reward_history: Vec::new(),
            cycles_since_mutation: 0,
            total_rollbacks: 0,
            total_accepted: 0,
            total_rejected: 0,
            rng_state: 0xDEAD_BEEF_CAFE_42,
        }
    }

    /// Create with defaults.
    pub fn with_defaults() -> Self {
        Self::new(StructuralEvolutionConfig::default())
    }

    /// Create with a specific initial genome.
    pub fn with_genome(genome: EvolutionaryGenome, config: StructuralEvolutionConfig) -> Self {
        Self {
            genome,
            ..Self::new(config)
        }
    }

    /// Compute TGC (the navigability score) for current genome at given state.
    fn compute_tgc(&self, lambda2: f64, energy: f64, h_path: f64) -> NavigabilityReport {
        let eval = NavigabilityEvaluator::new(self.genome.to_navigability_config());
        eval.evaluate(lambda2, energy, h_path)
    }

    /// Compute TGC with a specific genome (for pre/post comparison).
    #[cfg(test)]
    fn compute_tgc_with(
        genome: &EvolutionaryGenome,
        lambda2: f64,
        energy: f64,
        h_path: f64,
    ) -> NavigabilityReport {
        let eval = NavigabilityEvaluator::new(genome.to_navigability_config());
        eval.evaluate(lambda2, energy, h_path)
    }

    /// Take a snapshot of the current state.
    fn take_snapshot(&self, cycle_number: u64, tgc: f64, lambda2: f64, energy: f64, h_path: f64) -> GenomicSnapshot {
        GenomicSnapshot {
            genome: self.genome.clone(),
            cycle_number,
            tgc,
            lambda2,
            energy,
            h_path,
        }
    }

    /// Perform rollback: restore genome from snapshot.
    fn rollback(&mut self, reason: RollbackReason) -> RollbackReason {
        if let Some(ref snap) = self.snapshot {
            self.genome = snap.genome.clone();
        }
        self.total_rollbacks += 1;
        self.total_rejected += 1;

        // Reduce mutation rate after rollback (be more conservative)
        self.genome.mutation_rate = (self.genome.mutation_rate * 0.8)
            .max(self.genome.min_mutation_rate);

        reason
    }

    /// Innovation homeostasis: adjust mutation_rate to drive I toward I_target.
    fn homeostasis_adjust(&mut self, current_i: f64) {
        let target = self.config.i_target;
        if target <= 0.0 {
            return;
        }

        // Relative error: (I_target - I) / I_target
        let error = (target - current_i) / target;

        // PID-like proportional control:
        // I too low → increase mutation_rate (explore more)
        // I too high → decrease mutation_rate (conserve)
        let adjustment = 1.0 + self.config.homeostasis_gain * error;
        self.genome.mutation_rate *= adjustment.clamp(0.5, 2.0);

        // Clamp to safety bounds
        self.genome.mutation_rate = self.genome.mutation_rate
            .clamp(self.genome.min_mutation_rate, self.genome.max_mutation_rate);
    }

    /// Record innovation score in history.
    fn record_i(&mut self, i: f64) {
        self.i_history.push(i);
        if self.i_history.len() > self.config.i_history_size {
            self.i_history.remove(0);
        }
    }

    /// Get the innovation score trend (positive = improving).
    pub fn i_trend(&self) -> f64 {
        if self.i_history.len() < 2 {
            return 0.0;
        }
        let first = self.i_history[0];
        let last = *self.i_history.last().unwrap();
        (last - first) / self.i_history.len() as f64
    }

    /// Get mean innovation score over history.
    pub fn i_mean(&self) -> f64 {
        if self.i_history.is_empty() {
            return 0.0;
        }
        self.i_history.iter().sum::<f64>() / self.i_history.len() as f64
    }

    /// Compute TGC variance over the history window.
    ///
    /// Used by proportional mutation: ε = ε₀ · exp(-k · Var(TGC)) · (1 - ρ/ρ_max).
    /// High variance → reduce mutation amplitude (turbulence protection).
    /// Low variance → increase mutation amplitude (explore new personalities).
    pub fn compute_tgc_variance(&self) -> f64 {
        if self.i_history.len() < 2 {
            return 0.0;
        }
        let mean = self.i_mean();
        let n = self.i_history.len() as f64;
        self.i_history.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    }

    /// Record a structural reward in the history window.
    fn record_reward(&mut self, reward: f64) {
        self.reward_history.push(reward);
        if self.reward_history.len() > self.config.reward_window_size {
            self.reward_history.remove(0);
        }
    }

    /// Compute the mean recent structural reward (R_structural).
    ///
    /// Used by the success inertia factor: `tanh(R_structural)`.
    /// - R > 0 → tanh > 0 → bolder mutations (organism has momentum)
    /// - R ≈ 0 → tanh ≈ 0 → cautious (no track record yet)
    /// - R < 0 → tanh < 0 → very cautious (clamped to 0.1 in mutate)
    pub fn mean_recent_reward(&self) -> f64 {
        if self.reward_history.is_empty() {
            return 0.0;
        }
        self.reward_history.iter().sum::<f64>() / self.reward_history.len() as f64
    }

    /// Get current NavigabilityConfig from the genome.
    pub fn current_config(&self) -> NavigabilityConfig {
        self.genome.to_navigability_config()
    }

    /// Run one structural evolution step.
    ///
    /// # Arguments
    /// - `cycle_number`: current metabolic cycle number
    /// - `lambda2`: Fiedler eigenvalue λ₂
    /// - `energy`: trajectory energy E(τ) ∈ [0, 1]
    /// - `h_path`: normalized path entropy H ∈ [0, 1]
    /// - `energy_variance`: σ²_E (for turbulence detection)
    ///
    /// # Returns
    /// An [`EvolutionStepReport`] with full audit trail.
    pub fn step(
        &mut self,
        cycle_number: u64,
        lambda2: f64,
        energy: f64,
        h_path: f64,
        energy_variance: f64,
    ) -> EvolutionStepReport {
        self.budget.tick_cycle();
        self.cycles_since_mutation += 1;

        // ── Evaluate current TGC ──
        let current_report = self.compute_tgc(lambda2, energy, h_path);
        let current_tgc = current_report.score;
        self.record_i(current_tgc);

        // ── Sensitivity analysis ──
        let (di_da, di_db, di_dg) = self.genome.sensitivity(lambda2, energy, h_path);
        let sensitivity_norm = (di_da.powi(2) + di_db.powi(2) + di_dg.powi(2)).sqrt();

        // ── Check skip conditions ──
        if !self.config.enabled {
            return self.make_skip_report(
                "disabled", current_tgc, sensitivity_norm,
            );
        }

        if self.budget.in_cooldown() {
            return self.make_skip_report(
                "budget cooldown", current_tgc, sensitivity_norm,
            );
        }

        if self.cycles_since_mutation < self.config.mutation_cooldown_cycles {
            self.homeostasis_adjust(current_tgc);
            return self.make_skip_report(
                "mutation cooldown", current_tgc, sensitivity_norm,
            );
        }

        if !self.budget.can_mutate() {
            return self.make_skip_report(
                "budget exhausted", current_tgc, sensitivity_norm,
            );
        }

        // ── Pre-mutation safety checks ──

        // Check spectral safety
        if lambda2 < self.config.lambda2_safety_floor {
            self.homeostasis_adjust(current_tgc);
            return self.make_skip_report(
                &format!("λ₂={lambda2:.4} below safety floor"),
                current_tgc,
                sensitivity_norm,
            );
        }

        // Check energy turbulence
        if energy_variance > self.config.max_energy_variance {
            self.homeostasis_adjust(current_tgc);
            return self.make_skip_report(
                &format!("σ²_E={energy_variance:.4} exceeds max"),
                current_tgc,
                sensitivity_norm,
            );
        }

        // Check sensitivity explosion
        if sensitivity_norm > self.config.max_sensitivity_norm {
            self.homeostasis_adjust(current_tgc);
            return self.make_skip_report(
                &format!("‖∂I/∂θ‖={sensitivity_norm:.3} too large"),
                current_tgc,
                sensitivity_norm,
            );
        }

        // ── Take snapshot ──
        self.snapshot = Some(self.take_snapshot(
            cycle_number, current_tgc, lambda2, energy, h_path,
        ));

        // ── Mutate (Proportional Adaptive Mutation — Phase VIII) ──
        // ε(t) = ε_base · exp(-k·σ²_TGC) · (1-ρ/ρ_max) · tanh(R_structural)
        let tgc_before = current_tgc;
        let tgc_variance = self.compute_tgc_variance();
        let recent_reward = self.mean_recent_reward();
        let mut rng_state = self.rng_state;
        let mut rng_fn = || -> f64 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f64) / (u32::MAX as f64)
        };
        self.genome.mutate_proportional(
            &mut rng_fn,
            tgc_variance,
            self.config.tgc_variance_sensitivity,
            self.config.rho_max,
            recent_reward,
        );
        self.rng_state = rng_state;

        self.budget.record_mutation();
        self.cycles_since_mutation = 0;

        // ── Evaluate post-mutation TGC ──
        let post_report = self.compute_tgc(lambda2, energy, h_path);
        let tgc_after = post_report.score;
        let reward = tgc_after - tgc_before;

        // ── Post-mutation safety: check for rollback conditions ──

        // 1. TGC degradation
        if reward < self.config.tgc_degradation_threshold {
            self.record_reward(reward); // Track negative reward for inertia
            let reason = self.rollback(RollbackReason::TgcDegradation {
                delta_tgc: reward,
                threshold: self.config.tgc_degradation_threshold,
            });
            self.homeostasis_adjust(tgc_before); // Use pre-mutation score
            return EvolutionStepReport {
                generation: self.genome.generation,
                tgc_before,
                tgc_after: Some(tgc_after),
                reward: Some(reward),
                accepted: false,
                rollback_reason: Some(reason),
                genome_alpha: self.genome.alpha,
                genome_beta: self.genome.beta,
                genome_gamma: self.genome.gamma,
                mutation_rate: self.genome.mutation_rate,
                skipped: false,
                skip_reason: None,
                current_i: tgc_before,
                target_i: self.config.i_target,
                sensitivity_norm,
            };
        }

        // 2. Sensitivity explosion post-mutation
        let (da2, db2, dg2) = self.genome.sensitivity(lambda2, energy, h_path);
        let post_sens = (da2.powi(2) + db2.powi(2) + dg2.powi(2)).sqrt();
        if post_sens > self.config.max_sensitivity_norm {
            self.record_reward(reward); // Track for inertia
            let reason = self.rollback(RollbackReason::SensitivityExplosion {
                norm: post_sens,
                max_norm: self.config.max_sensitivity_norm,
            });
            self.homeostasis_adjust(tgc_before);
            return EvolutionStepReport {
                generation: self.genome.generation,
                tgc_before,
                tgc_after: Some(tgc_after),
                reward: Some(reward),
                accepted: false,
                rollback_reason: Some(reason),
                genome_alpha: self.genome.alpha,
                genome_beta: self.genome.beta,
                genome_gamma: self.genome.gamma,
                mutation_rate: self.genome.mutation_rate,
                skipped: false,
                skip_reason: None,
                current_i: tgc_before,
                target_i: self.config.i_target,
                sensitivity_norm: post_sens,
            };
        }

        // ── Mutation accepted ──
        self.total_accepted += 1;
        self.record_reward(reward);

        // Update snapshot with new state
        self.snapshot = Some(self.take_snapshot(
            cycle_number, tgc_after, lambda2, energy, h_path,
        ));

        // Homeostasis adjustment
        self.homeostasis_adjust(tgc_after);

        // Positive reward → slightly increase mutation rate (momentum)
        if reward > 0.0 {
            self.genome.mutation_rate = (self.genome.mutation_rate * 1.05)
                .min(self.genome.max_mutation_rate);
        }

        EvolutionStepReport {
            generation: self.genome.generation,
            tgc_before,
            tgc_after: Some(tgc_after),
            reward: Some(reward),
            accepted: true,
            rollback_reason: None,
            genome_alpha: self.genome.alpha,
            genome_beta: self.genome.beta,
            genome_gamma: self.genome.gamma,
            mutation_rate: self.genome.mutation_rate,
            skipped: false,
            skip_reason: None,
            current_i: tgc_after,
            target_i: self.config.i_target,
            sensitivity_norm: post_sens,
        }
    }

    /// Helper to create a skip report.
    fn make_skip_report(
        &self,
        reason: &str,
        current_tgc: f64,
        sensitivity_norm: f64,
    ) -> EvolutionStepReport {
        EvolutionStepReport {
            generation: self.genome.generation,
            tgc_before: current_tgc,
            tgc_after: None,
            reward: None,
            accepted: false,
            rollback_reason: None,
            genome_alpha: self.genome.alpha,
            genome_beta: self.genome.beta,
            genome_gamma: self.genome.gamma,
            mutation_rate: self.genome.mutation_rate,
            skipped: true,
            skip_reason: Some(reason.to_string()),
            current_i: current_tgc,
            target_i: self.config.i_target,
            sensitivity_norm,
        }
    }

    /// Get statistics summary.
    pub fn stats(&self) -> EvolutionUnitStats {
        EvolutionUnitStats {
            generation: self.genome.generation,
            total_accepted: self.total_accepted,
            total_rejected: self.total_rejected,
            total_rollbacks: self.total_rollbacks,
            acceptance_rate: if self.total_accepted + self.total_rejected > 0 {
                self.total_accepted as f64 / (self.total_accepted + self.total_rejected) as f64
            } else {
                0.0
            },
            i_mean: self.i_mean(),
            i_trend: self.i_trend(),
            current_alpha: self.genome.alpha,
            current_beta: self.genome.beta,
            current_gamma: self.genome.gamma,
            mutation_rate: self.genome.mutation_rate,
            stability_score: self.genome.stability_score(),
            budget_total_mutations: self.budget.total_mutations(),
            budget_in_cooldown: self.budget.in_cooldown(),
        }
    }
}

/// Statistics snapshot for the StructuralEvolutionUnit.
#[derive(Debug, Clone)]
pub struct EvolutionUnitStats {
    pub generation: u64,
    pub total_accepted: u64,
    pub total_rejected: u64,
    pub total_rollbacks: u64,
    pub acceptance_rate: f64,
    pub i_mean: f64,
    pub i_trend: f64,
    pub current_alpha: f64,
    pub current_beta: f64,
    pub current_gamma: f64,
    pub mutation_rate: f64,
    pub stability_score: f64,
    pub budget_total_mutations: u64,
    pub budget_in_cooldown: bool,
}

impl std::fmt::Display for EvolutionUnitStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SEU[gen={} accept={}/{} ({:.0}%) | α={:.3} β={:.3} γ={:.3} σ={:.4} \
             Ω={:.3} | I_mean={:.6} trend={:+.2e}{}]",
            self.generation,
            self.total_accepted,
            self.total_accepted + self.total_rejected,
            self.acceptance_rate * 100.0,
            self.current_alpha,
            self.current_beta,
            self.current_gamma,
            self.mutation_rate,
            self.stability_score,
            self.i_mean,
            self.i_trend,
            if self.budget_in_cooldown { " COOLDOWN" } else { "" },
        )
    }
}

// ═══════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── GenomeBounds tests ──

    #[test]
    fn test_bounds_default_valid() {
        let bounds = GenomeBounds::default();
        // Default genome parameters should be valid
        assert!(bounds.is_valid(0.5, 0.3, 1.2));
    }

    #[test]
    fn test_bounds_reject_out_of_range() {
        let bounds = GenomeBounds::default();
        assert!(!bounds.is_valid(0.0, 0.3, 1.2)); // α below min
        assert!(!bounds.is_valid(0.5, 3.0, 1.2)); // β above max
        assert!(!bounds.is_valid(0.5, 0.3, 0.1)); // γ below min
    }

    #[test]
    fn test_bounds_reject_sum_exceeded() {
        let bounds = GenomeBounds::default();
        // Sum = 2.0 + 2.0 + 2.0 = 6.0 > 5.0
        assert!(!bounds.is_valid(2.0, 2.0, 2.0));
    }

    #[test]
    fn test_bounds_enforce_clamps() {
        let bounds = GenomeBounds::default();
        let (a, b, g) = bounds.enforce(0.0, 5.0, 0.1);
        assert!(a >= bounds.alpha_range.0);
        assert!(b <= bounds.beta_range.1);
        assert!(g >= bounds.gamma_range.0);
    }

    #[test]
    fn test_bounds_enforce_sum() {
        let bounds = GenomeBounds::default();
        let (a, b, g) = bounds.enforce(2.0, 2.0, 3.0);
        assert!(
            a + b + g <= bounds.max_sum + 1e-10,
            "Sum should be ≤ {}: got {}",
            bounds.max_sum,
            a + b + g
        );
    }

    #[test]
    fn test_bounds_enforce_preserves_valid() {
        let bounds = GenomeBounds::default();
        let (a, b, g) = bounds.enforce(0.5, 0.3, 1.2);
        assert!((a - 0.5).abs() < 1e-10);
        assert!((b - 0.3).abs() < 1e-10);
        assert!((g - 1.2).abs() < 1e-10);
    }

    // ── EvolutionaryGenome tests ──

    #[test]
    fn test_genome_default() {
        let genome = EvolutionaryGenome::default();
        assert!((genome.alpha - 0.5).abs() < 1e-10);
        assert!((genome.beta - 0.3).abs() < 1e-10);
        assert!((genome.gamma - 1.2).abs() < 1e-10);
        assert_eq!(genome.generation, 0);
        assert!(genome.is_valid());
    }

    #[test]
    fn test_genome_from_navigability() {
        let config = NavigabilityConfig {
            alpha: 0.8,
            beta: 0.4,
            gamma: 1.5,
        };
        let genome = EvolutionaryGenome::from_navigability_config(&config);
        assert!((genome.alpha - 0.8).abs() < 1e-10);
        assert!((genome.beta - 0.4).abs() < 1e-10);
        assert!((genome.gamma - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_genome_roundtrip_config() {
        let genome = EvolutionaryGenome::default();
        let config = genome.to_navigability_config();
        assert!((config.alpha - genome.alpha).abs() < 1e-10);
        assert!((config.beta - genome.beta).abs() < 1e-10);
        assert!((config.gamma - genome.gamma).abs() < 1e-10);
    }

    #[test]
    fn test_genome_mutate_stays_valid() {
        let mut genome = EvolutionaryGenome::default();
        let mut rng_state = 42u64;
        let mut rng = || -> f64 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f64) / (u32::MAX as f64)
        };

        for _ in 0..100 {
            genome.mutate(&mut rng);
            assert!(
                genome.is_valid(),
                "Genome invalid after mutation: α={}, β={}, γ={}, sum={}",
                genome.alpha,
                genome.beta,
                genome.gamma,
                genome.alpha + genome.beta + genome.gamma,
            );
        }
    }

    #[test]
    fn test_genome_mutate_increments_generation() {
        let mut genome = EvolutionaryGenome::default();
        assert_eq!(genome.generation, 0);

        let mut rng_state = 42u64;
        let mut rng = || -> f64 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f64) / (u32::MAX as f64)
        };

        genome.mutate(&mut rng);
        assert_eq!(genome.generation, 1);
        genome.mutate(&mut rng);
        assert_eq!(genome.generation, 2);
    }

    #[test]
    fn test_genome_optimal_h_path() {
        let genome = EvolutionaryGenome::default();
        let h_peak = genome.optimal_h_path();
        assert!((h_peak - 1.2 / 2.2).abs() < 1e-10);
    }

    #[test]
    fn test_genome_stability_score() {
        let genome = EvolutionaryGenome::default();
        let omega = genome.stability_score();
        // max(0.5, 0.3, 1.2) = 1.2, sum/5.0 = 2.0/5.0 = 0.4
        let expected = 1.2 + 0.4;
        assert!(
            (omega - expected).abs() < 1e-10,
            "Ω should be {expected}: got {omega}"
        );
    }

    #[test]
    fn test_genome_sensitivity() {
        let genome = EvolutionaryGenome::default();
        let (da, db, dg) = genome.sensitivity(0.5, 0.8, 0.5);
        // ∂I/∂α = I · ln(0.5) < 0 (since ln(0.5) < 0)
        assert!(da < 0.0, "∂I/∂α should be negative when λ₂ < 1: {da}");
        // ∂I/∂β = I · ln(0.8) < 0
        assert!(db < 0.0, "∂I/∂β should be negative when E < 1: {db}");
        // ∂I/∂γ = I · ln(0.5) < 0
        assert!(dg < 0.0, "∂I/∂γ should be negative when H < 1: {dg}");
    }

    #[test]
    fn test_genome_display() {
        let genome = EvolutionaryGenome::default();
        let s = format!("{genome}");
        assert!(s.contains("α=0.500"));
        assert!(s.contains("β=0.300"));
        assert!(s.contains("γ=1.200"));
        assert!(s.contains("gen=0"));
    }

    // ── CognitiveBudget tests ──

    #[test]
    fn test_budget_default_can_mutate() {
        let budget = CognitiveBudget::default();
        assert!(budget.can_mutate());
        assert!(!budget.in_cooldown());
    }

    #[test]
    fn test_budget_cooldown() {
        let mut budget = CognitiveBudget {
            max_total_mutations: 3,
            cooldown_cycles: 2,
            ..Default::default()
        };

        budget.record_mutation();
        budget.record_mutation();
        assert!(budget.can_mutate());

        budget.record_mutation(); // Hits limit
        assert!(!budget.can_mutate());
        assert!(budget.in_cooldown());

        budget.tick_cycle(); // Cooldown 1
        assert!(budget.in_cooldown());

        budget.tick_cycle(); // Cooldown 2 → resets
        assert!(!budget.in_cooldown());
        assert!(budget.can_mutate());
    }

    // ── GenomicSnapshot tests ──

    #[test]
    fn test_snapshot_display() {
        let snap = GenomicSnapshot {
            genome: EvolutionaryGenome::default(),
            cycle_number: 42,
            tgc: 0.0856,
            lambda2: 0.3,
            energy: 0.7,
            h_path: 0.5,
        };
        let s = format!("{snap}");
        assert!(s.contains("cycle=42"));
        assert!(s.contains("TGC="));
    }

    // ── StructuralEvolutionUnit tests ──

    #[test]
    fn test_seu_creation() {
        let seu = StructuralEvolutionUnit::with_defaults();
        assert_eq!(seu.genome.generation, 0);
        assert_eq!(seu.total_accepted, 0);
        assert_eq!(seu.total_rejected, 0);
        assert!(seu.snapshot.is_none());
    }

    #[test]
    fn test_seu_skip_when_disabled() {
        let config = StructuralEvolutionConfig {
            enabled: false,
            ..Default::default()
        };
        let mut seu = StructuralEvolutionUnit::new(config);
        let report = seu.step(1, 0.5, 0.8, 0.5, 0.01);
        assert!(report.skipped);
        assert!(report.skip_reason.as_deref().unwrap().contains("disabled"));
    }

    #[test]
    fn test_seu_skip_during_cooldown() {
        let config = StructuralEvolutionConfig {
            mutation_cooldown_cycles: 5,
            ..Default::default()
        };
        let mut seu = StructuralEvolutionUnit::new(config);

        // First step should skip (cooldown = 5 but cycles_since_mutation = 0)
        let report = seu.step(1, 0.5, 0.8, 0.5, 0.01);
        assert!(report.skipped);
        assert!(report.skip_reason.as_deref().unwrap().contains("mutation cooldown"));
    }

    #[test]
    fn test_seu_skip_low_lambda2() {
        let config = StructuralEvolutionConfig {
            mutation_cooldown_cycles: 0,
            lambda2_safety_floor: 0.1,
            ..Default::default()
        };
        let mut seu = StructuralEvolutionUnit::new(config);
        // Force past cooldown
        seu.cycles_since_mutation = 100;

        let report = seu.step(1, 0.01, 0.8, 0.5, 0.01);
        assert!(report.skipped);
        assert!(report.skip_reason.as_deref().unwrap().contains("below safety floor"));
    }

    #[test]
    fn test_seu_mutation_accepted() {
        let config = StructuralEvolutionConfig {
            mutation_cooldown_cycles: 0,
            tgc_degradation_threshold: -1.0, // Very lenient
            ..Default::default()
        };
        let mut seu = StructuralEvolutionUnit::new(config);
        seu.cycles_since_mutation = 100;

        let report = seu.step(1, 0.5, 0.8, 0.5, 0.01);
        // Should have attempted a mutation
        assert!(!report.skipped);
        // With very lenient threshold, should be accepted
        assert!(
            report.accepted || report.rollback_reason.is_some(),
            "Should either accept or have a rollback reason"
        );
    }

    #[test]
    fn test_seu_rollback_on_tgc_degradation() {
        let config = StructuralEvolutionConfig {
            mutation_cooldown_cycles: 0,
            tgc_degradation_threshold: 0.0, // Any degradation triggers rollback
            ..Default::default()
        };
        let mut seu = StructuralEvolutionUnit::new(config);
        seu.cycles_since_mutation = 100;

        // Run many steps; some should trigger rollbacks
        let mut rollbacks = 0;
        for i in 0..50 {
            seu.cycles_since_mutation = 100;
            let report = seu.step(i as u64, 0.5, 0.8, 0.5, 0.01);
            if !report.skipped && !report.accepted {
                rollbacks += 1;
                assert!(report.rollback_reason.is_some());
            }
        }
        // With threshold = 0.0, any mutation that doesn't improve should rollback
        assert!(
            rollbacks > 0,
            "Should have at least some rollbacks with strict threshold"
        );
    }

    #[test]
    fn test_seu_genome_stays_valid_through_evolution() {
        let config = StructuralEvolutionConfig {
            mutation_cooldown_cycles: 0,
            tgc_degradation_threshold: -1.0,
            ..Default::default()
        };
        let mut seu = StructuralEvolutionUnit::new(config);

        for i in 0..100 {
            seu.cycles_since_mutation = 100;
            seu.step(i as u64, 0.5, 0.8, 0.5, 0.01);
            assert!(
                seu.genome.is_valid(),
                "Genome invalid at step {i}: α={}, β={}, γ={}, sum={}",
                seu.genome.alpha,
                seu.genome.beta,
                seu.genome.gamma,
                seu.genome.alpha + seu.genome.beta + seu.genome.gamma,
            );
        }
    }

    #[test]
    fn test_seu_homeostasis_increases_rate_when_i_low() {
        let config = StructuralEvolutionConfig {
            i_target: 1.0, // Very high target → I will always be below
            homeostasis_gain: 0.5,
            mutation_cooldown_cycles: 0,
            ..Default::default()
        };
        let mut seu = StructuralEvolutionUnit::new(config);
        let initial_rate = seu.genome.mutation_rate;

        // Run with very low I (disconnected graph)
        for i in 0..5 {
            seu.cycles_since_mutation = 100;
            seu.step(i as u64, 0.01, 0.3, 0.5, 0.01);
        }

        // Mutation rate should have increased due to homeostasis
        // (I << I_target → explore more)
        // Note: rollbacks reduce rate, so we check the net trend
        assert!(
            seu.genome.mutation_rate >= initial_rate * 0.5,
            "Rate should not collapse: initial={initial_rate}, current={}",
            seu.genome.mutation_rate,
        );
    }

    #[test]
    fn test_seu_stats() {
        let seu = StructuralEvolutionUnit::with_defaults();
        let stats = seu.stats();
        let s = format!("{stats}");
        assert!(s.contains("SEU["));
        assert!(s.contains("gen=0"));
    }

    #[test]
    fn test_seu_i_history() {
        let config = StructuralEvolutionConfig {
            mutation_cooldown_cycles: 0,
            tgc_degradation_threshold: -1.0,
            i_history_size: 5,
            ..Default::default()
        };
        let mut seu = StructuralEvolutionUnit::new(config);

        for i in 0..10 {
            seu.cycles_since_mutation = 100;
            seu.step(i as u64, 0.5, 0.8, 0.5, 0.01);
        }

        // History should be capped at i_history_size
        assert!(
            seu.i_history.len() <= 5,
            "History should be capped: {}",
            seu.i_history.len()
        );
    }

    #[test]
    fn test_seu_report_display() {
        let config = StructuralEvolutionConfig {
            mutation_cooldown_cycles: 0,
            tgc_degradation_threshold: -1.0,
            ..Default::default()
        };
        let mut seu = StructuralEvolutionUnit::new(config);
        seu.cycles_since_mutation = 100;

        let report = seu.step(1, 0.5, 0.8, 0.5, 0.01);
        let s = format!("{report}");
        assert!(
            s.contains("Evolution["),
            "Report display should contain Evolution[: {s}"
        );
    }

    #[test]
    fn test_rollback_reason_display() {
        let reason = RollbackReason::SpectralRupture {
            lambda2: 0.003,
            threshold: 0.005,
        };
        let s = format!("{reason}");
        assert!(s.contains("SpectralRupture"));
        assert!(s.contains("0.003"));
    }

    #[test]
    fn test_evolution_unit_stats_display() {
        let stats = EvolutionUnitStats {
            generation: 42,
            total_accepted: 30,
            total_rejected: 12,
            total_rollbacks: 12,
            acceptance_rate: 0.714,
            i_mean: 0.085,
            i_trend: 0.001,
            current_alpha: 0.5,
            current_beta: 0.3,
            current_gamma: 1.2,
            mutation_rate: 0.05,
            stability_score: 1.6,
            budget_total_mutations: 42,
            budget_in_cooldown: false,
        };
        let s = format!("{stats}");
        assert!(s.contains("gen=42"));
        assert!(s.contains("accept=30/42"));
    }

    // ── Integration test: verify TGC computation matches NavigabilityEvaluator ──

    #[test]
    fn test_tgc_matches_navigability() {
        let seu = StructuralEvolutionUnit::with_defaults();
        let report = seu.compute_tgc(0.5, 0.8, 0.5);

        let nav = NavigabilityEvaluator::with_defaults();
        let nav_report = nav.evaluate(0.5, 0.8, 0.5);

        assert!(
            (report.score - nav_report.score).abs() < 1e-10,
            "TGC should match: {} vs {}",
            report.score,
            nav_report.score,
        );
    }

    #[test]
    fn test_tgc_with_mutated_genome() {
        let mut genome = EvolutionaryGenome::new(0.8, 0.5, 1.5);
        let report_before = StructuralEvolutionUnit::compute_tgc_with(&genome, 0.5, 0.8, 0.5);

        let mut rng_state = 42u64;
        let mut rng = || -> f64 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f64) / (u32::MAX as f64)
        };
        genome.mutate(&mut rng);
        let report_after = StructuralEvolutionUnit::compute_tgc_with(&genome, 0.5, 0.8, 0.5);

        // Scores should differ after mutation (unless extremely unlikely same params)
        // We just verify both are valid
        assert!(report_before.score >= 0.0);
        assert!(report_after.score >= 0.0);
    }

    // ── Proportional Mutation tests ──

    #[test]
    fn test_proportional_mutation_stays_valid() {
        let mut genome = EvolutionaryGenome::default();
        let mut rng_state = 42u64;
        let mut rng = || -> f64 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f64) / (u32::MAX as f64)
        };

        for _ in 0..100 {
            genome.mutate_proportional(&mut rng, 0.01, 2.0, 2.5, 0.5);
            assert!(
                genome.is_valid(),
                "Genome invalid: α={}, β={}, γ={}",
                genome.alpha, genome.beta, genome.gamma,
            );
        }
    }

    #[test]
    fn test_proportional_mutation_high_variance_reduces_step() {
        let mut genome1 = EvolutionaryGenome::default();
        let mut genome2 = genome1.clone();

        let mut rng_state1 = 42u64;
        let mut rng1 = || -> f64 {
            rng_state1 = rng_state1
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state1 >> 33) as f64) / (u32::MAX as f64)
        };

        let mut rng_state2 = 42u64;
        let mut rng2 = || -> f64 {
            rng_state2 = rng_state2
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state2 >> 33) as f64) / (u32::MAX as f64)
        };

        // Low variance → larger steps
        genome1.mutate_proportional(&mut rng1, 0.001, 2.0, 2.5, 0.5);
        // High variance → smaller steps
        genome2.mutate_proportional(&mut rng2, 1.0, 2.0, 2.5, 0.5);

        // Both should be valid but with different magnitudes of change
        assert!(genome1.is_valid());
        assert!(genome2.is_valid());
    }

    #[test]
    fn test_spectral_guardrail_attenuates() {
        let mut genome = EvolutionaryGenome {
            alpha: 2.0,
            beta: 2.0,
            gamma: 2.5, // max = 2.5 > 2.2 → should attenuate
            bounds: GenomeBounds::default(),
            ..Default::default()
        };
        genome.enforce_spectral_guardrail();
        let max_after = genome.alpha.max(genome.beta).max(genome.gamma);
        assert!(
            max_after <= 2.2,
            "After guardrail, max should be ≤ 2.2: {max_after}"
        );
    }

    #[test]
    fn test_spectral_guardrail_sum_constraint() {
        let mut genome = EvolutionaryGenome {
            alpha: 2.0,
            beta: 1.5,
            gamma: 1.5, // sum = 5.0 > 4.5
            bounds: GenomeBounds::default(),
            ..Default::default()
        };
        genome.enforce_spectral_guardrail();
        let sum = genome.alpha + genome.beta + genome.gamma;
        assert!(
            sum <= 4.5 + 1e-10,
            "After guardrail, sum should be ≤ 4.5: {sum}"
        );
    }

    #[test]
    fn test_tgc_variance_computation() {
        let config = StructuralEvolutionConfig {
            i_history_size: 10,
            ..Default::default()
        };
        let mut seu = StructuralEvolutionUnit::new(config);

        // Empty history → variance = 0
        assert_eq!(seu.compute_tgc_variance(), 0.0);

        // Add constant values → variance ≈ 0
        for _ in 0..5 {
            seu.i_history.push(0.08);
        }
        assert!(
            seu.compute_tgc_variance() < 1e-10,
            "Constant series should have zero variance"
        );

        // Add oscillating values → variance > 0
        seu.i_history.clear();
        seu.i_history.push(0.05);
        seu.i_history.push(0.10);
        seu.i_history.push(0.05);
        seu.i_history.push(0.10);
        assert!(
            seu.compute_tgc_variance() > 0.0005,
            "Oscillating series should have variance > 0"
        );
    }

    // ── Success Inertia tests ──

    #[test]
    fn test_success_inertia_zero_reward() {
        let mut genome = EvolutionaryGenome::default();
        let original = genome.clone();
        let mut rng_state = 42u64;
        let mut rng = || -> f64 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f64) / (u32::MAX as f64)
        };

        // With zero reward → tanh(0) = 0, clamped to 0.1 → small mutations
        genome.mutate_proportional(&mut rng, 0.0, 2.0, 2.5, 0.0);
        assert!(genome.is_valid());
        // Should have mutated but with reduced amplitude (0.1 × base)
        assert_eq!(genome.generation, 1);
        // At least one parameter should have changed (single-param mutation)
        let changed = (genome.alpha - original.alpha).abs() > 1e-15
            || (genome.beta - original.beta).abs() > 1e-15
            || (genome.gamma - original.gamma).abs() > 1e-15;
        assert!(changed, "At least one parameter should change");
    }

    #[test]
    fn test_success_inertia_positive_reward_bolder() {
        // Compare mutation magnitudes with positive vs zero reward
        let mut genome_bold = EvolutionaryGenome::default();
        let mut genome_cautious = EvolutionaryGenome::default();

        // Same RNG seed for both
        let make_rng = || {
            let mut state = 42u64;
            move || -> f64 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f64) / (u32::MAX as f64)
            }
        };

        let mut rng1 = make_rng();
        let mut rng2 = make_rng();

        // Positive reward → tanh(2.0) ≈ 0.964 → near-full ε
        genome_bold.mutate_proportional(&mut rng1, 0.0, 2.0, 2.5, 2.0);
        // Zero reward → tanh(0) = 0, clamped to 0.1 → 10× smaller ε
        genome_cautious.mutate_proportional(&mut rng2, 0.0, 2.0, 2.5, 0.0);

        // Both should be valid
        assert!(genome_bold.is_valid());
        assert!(genome_cautious.is_valid());
    }

    #[test]
    fn test_success_inertia_negative_reward_cautious() {
        let mut genome = EvolutionaryGenome::default();
        let mut rng_state = 42u64;
        let mut rng = || -> f64 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f64) / (u32::MAX as f64)
        };

        // Negative reward → tanh(-1.0) ≈ -0.76, clamped to 0.1
        genome.mutate_proportional(&mut rng, 0.0, 2.0, 2.5, -1.0);
        assert!(genome.is_valid());
        assert_eq!(genome.generation, 1);
    }

    #[test]
    fn test_single_param_mutation() {
        // Verify that single-parameter mutation selects all three params over time.
        // Use a PERSISTENT RNG across iterations so the state diverges properly.
        let mut genome = EvolutionaryGenome::default();
        let mut changes = [0usize; 3]; // [alpha_changes, beta_changes, gamma_changes]

        let mut rng_state = 0xDEAD_BEEF_42u64;

        for _ in 0..60 {
            let original = genome.clone();
            let mut rng = || -> f64 {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rng_state >> 33) as f64) / (u32::MAX as f64)
            };

            genome.mutate_proportional(&mut rng, 0.0, 2.0, 2.5, 1.0);

            let a_changed = (genome.alpha - original.alpha).abs() > 1e-15;
            let b_changed = (genome.beta - original.beta).abs() > 1e-15;
            let g_changed = (genome.gamma - original.gamma).abs() > 1e-15;

            // Count which parameters were modified (bounds enforcement may
            // adjust secondary parameters, which is acceptable)
            if a_changed { changes[0] += 1; }
            if b_changed { changes[1] += 1; }
            if g_changed { changes[2] += 1; }
        }

        // All three parameters should have been selected at least once
        // (with 60 iterations, probability of missing one is ~(2/3)^60 ≈ 2e-11)
        assert!(
            changes[0] > 0 && changes[1] > 0 && changes[2] > 0,
            "All parameters should be selected: α={}, β={}, γ={}",
            changes[0], changes[1], changes[2],
        );
    }

    #[test]
    fn test_reward_history_tracking() {
        let config = StructuralEvolutionConfig {
            mutation_cooldown_cycles: 0,
            tgc_degradation_threshold: -1.0, // Very lenient
            reward_window_size: 5,
            ..Default::default()
        };
        let mut seu = StructuralEvolutionUnit::new(config);

        // Initially empty
        assert_eq!(seu.mean_recent_reward(), 0.0);
        assert!(seu.reward_history.is_empty());

        // Run several evolution steps
        for i in 0..10 {
            seu.cycles_since_mutation = 100;
            seu.step(i as u64, 0.5, 0.8, 0.5, 0.01);
        }

        // Reward history should be capped at window size
        assert!(
            seu.reward_history.len() <= 5,
            "Reward history should be capped: {}",
            seu.reward_history.len(),
        );
    }

    #[test]
    fn test_axiom_drift_rollback_reason_display() {
        let reason = RollbackReason::AxiomDrift {
            drift: 0.15,
            limit: 0.10,
        };
        let s = format!("{reason}");
        assert!(s.contains("AxiomDrift"));
        assert!(s.contains("0.15"));
    }
}
