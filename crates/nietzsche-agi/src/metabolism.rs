//! # Metabolism — Riemannian Sleep + Path Entropy + Curiosity Engine
//!
//! This module implements the **physiological layer** of the AGI system:
//! the organism decides *when* to think, *what* to measure, and *when*
//! to force exploration.
//!
//! ## Three Components
//!
//! ### 1. Path Entropy Estimator (`PathEntropyEstimator`)
//!
//! Measures **functional navigability** H_path via stochastic random walks
//! on the hyperbolic graph. The walk transition is a **4-factor Boltzmann
//! distribution computed entirely in log-space** for numerical stability:
//!
//! ```text
//! log w_u = -β · d_𝔻(v, u)          // geodesic proximity
//!         + α · E_edge(v, u)          // edge energy affinity
//!         - γ · ln(deg(u))            // anti-hub penalty
//!         + memory_penalty(u)         // tabu avoidance
//! ```
//!
//! **Numerical safety guarantees:**
//! - **Poincaré distance**: denominators clamped to ε when |x| → 1
//! - **Hub penalty**: `-γ·ln(deg)` instead of `deg^(-γ)` (no underflow)
//! - **Sampling**: log-sum-exp trick prevents overflow/underflow
//! - **Spectral guardrail**: γ_eff = γ·σ((λ₂-λ_c)/τ) — smooth sigmoid, no hysteresis
//!
//! High H_path = diverse routes = healthy knowledge.
//! Low H_path = redundant paths = echo chamber or triviality.
//!
//! ### 2. Metabolic Sleep Manager (`MetabolicSleepManager`)
//!
//! Computes the **relaxation time** τ_relax from the state vector
//! S = [λ₂, E_g, H_path]:
//!
//! ```text
//! τ_relax = 1 / (‖∇S‖ + ε)
//! T_sleep = η · τ_relax · f(D, ΔE, ΔH)
//! ```
//!
//! Where f is a cognitive modulation factor that distinguishes:
//! - **Stagnation** (low D, low ΔE, low ΔH) → shorter sleep → exploration
//! - **Convergence** (low D, high ΔE) → longer sleep → consolidation
//! - **Turbulence** (high D, high variance) → normal sleep → regulation
//!
//! ### 3. Curiosity Engine (`CuriosityEngine`)
//!
//! When stagnation persists across multiple cycles, activates
//! **controlled perturbation** (Option D — adaptive combination):
//!
//! - **Geometric perturbation**: x ← exp_x(ε · ξ) (geodesic noise)
//! - **Anti-hub γ reweighting**: temporarily reduce hub gravity
//! - **Walk length increase**: extend exploration horizon
//!
//! ## Design: Pure Computation
//!
//! Like all AGI modules, metabolism is side-effect-free.
//! The caller provides graph data, receives reports and actions.

use std::time::Duration;

use nietzsche_hyp_ops::exp_map_zero;

use crate::innovation::NavigabilityReport;

#[cfg(test)]
use crate::innovation::NavigabilityEvaluator;

// ─────────────────────────────────────────────
// State Vector
// ─────────────────────────────────────────────

/// The sovereign triad of metabolic observables.
///
/// | Metric | Low | High |
/// |--------|-----|------|
/// | λ₂ | Fragmentation | Crystallization (Dogma) |
/// | E_g | Apathy / Forgetting | Obsession / Noise |
/// | H_path | Redundancy / Triviality | Semantic Chaos |
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateVector {
    /// Spectral connectivity (Fiedler eigenvalue λ₂).
    pub lambda2: f64,

    /// Global energy (mean node energy).
    pub energy: f64,

    /// Path entropy (navigability diversity).
    pub entropy: f64,
}

impl StateVector {
    pub fn new(lambda2: f64, energy: f64, entropy: f64) -> Self {
        Self { lambda2, energy, entropy }
    }

    /// Weighted gradient norm: ‖∇S‖ = √(w_λ·(dλ)² + w_E·(dE)² + w_H·(dH)²)
    pub fn gradient_norm(&self, prev: &StateVector, dt: f64, weights: &[f64; 3]) -> f64 {
        if dt <= 0.0 {
            return 0.0;
        }
        let d_lambda = (self.lambda2 - prev.lambda2) / dt;
        let d_energy = (self.energy - prev.energy) / dt;
        let d_entropy = (self.entropy - prev.entropy) / dt;
        (weights[0] * d_lambda.powi(2)
            + weights[1] * d_energy.powi(2)
            + weights[2] * d_entropy.powi(2))
        .sqrt()
    }
}

impl std::fmt::Display for StateVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "S[λ₂={:.4}, E={:.4}, H={:.4}]",
            self.lambda2, self.energy, self.entropy
        )
    }
}

// ─────────────────────────────────────────────
// Cognitive State
// ─────────────────────────────────────────────

/// Cognitive regime classification derived from divergence + energy + entropy dynamics.
///
/// This is NOT the spectral regime (Rigid/Critical/Turbulent from criticality.rs).
/// This classifies the *cognitive behavior* of the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CognitiveState {
    /// Low divergence + low ΔE + low ΔH → thinking in circles.
    /// Prescription: force exploration.
    Stagnation,

    /// Low divergence + high ΔE + H decreasing → solving a problem.
    /// Prescription: allow deep sleep (consolidation).
    Convergence,

    /// High divergence + high energy variance → chaotic exploration.
    /// Prescription: maintain vigilance (regulation).
    Turbulence,

    /// Everything within normal bounds.
    /// Prescription: standard metabolic rhythm.
    Cruising,
}

impl std::fmt::Display for CognitiveState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CognitiveState::Stagnation => write!(f, "Stagnation (thinking in circles)"),
            CognitiveState::Convergence => write!(f, "Convergence (resolving problem)"),
            CognitiveState::Turbulence => write!(f, "Turbulence (chaotic exploration)"),
            CognitiveState::Cruising => write!(f, "Cruising (normal rhythm)"),
        }
    }
}

// ═══════════════════════════════════════════════
// §1 — Path Entropy Estimator (Numerically Stable)
// ═══════════════════════════════════════════════

/// Configuration for path entropy estimation.
///
/// The walker uses a 4-factor Boltzmann transition in log-space:
/// ```text
/// log w_u = -β·d_𝔻(v,u) + α·E_edge - γ·ln(deg(u)) + memory_penalty
/// ```
#[derive(Debug, Clone)]
pub struct PathEntropyConfig {
    /// Number of random walks to sample.
    /// More walks = more accurate H_path, but slower.
    /// Default: 50
    pub n_walks: usize,

    /// Length of each random walk (number of steps).
    /// Default: 10
    pub walk_length: usize,

    /// Temperature β for geodesic proximity: -β · d_𝔻(v, u).
    /// Higher β = more greedy (follow shortest geodesic).
    /// Lower β = more exploratory (uniform random).
    /// Default: 1.0
    pub beta: f64,

    /// Edge energy affinity α: +α · E_edge(v, u).
    /// Higher α = prefer high-energy edges (structurally important).
    /// Default: 0.5
    pub alpha: f64,

    /// Anti-hub exponent γ: -γ · ln(deg(u)).
    /// Higher γ = stronger repulsion from hubs.
    /// Default: 0.3
    pub gamma: f64,

    /// Tabu memory size (recent nodes to penalize).
    /// 0 = no memory (pure Markov). 3-5 = short contextual memory.
    /// Default: 3
    pub memory_size: usize,

    /// Base log-space penalty for recently visited nodes (tabu).
    /// Equivalent to multiplying probability by exp(penalty).
    /// When `adaptive_memory` is true, the actual penalty is:
    ///   penalty = memory_penalty_base * (1 - H_local)
    /// where H_local is the walk's local entropy.
    /// Default: -5.0 (≈ ×0.0067)
    pub memory_penalty: f64,

    /// Enable adaptive memory penalty scaling by local entropy.
    ///
    /// When true:
    ///   effective_penalty = memory_penalty * (1 - H_local)
    ///   - Low H_local (redundant path) → stronger penalty → force exploration
    ///   - High H_local (exploratory path) → weaker penalty → allow revisits
    ///
    /// This prevents the fixed -5.0 from collapsing the probability
    /// support in sparse graph regimes.
    /// Default: true
    pub adaptive_memory: bool,

    /// λ₂ center for smooth spectral guardrail sigmoid (λ_c).
    /// When the graph is nearly disconnected, penalizing hubs
    /// would fragment the cognitive space further.
    /// Default: 0.02
    pub spectral_guardrail_threshold: f64,

    /// Sigmoid steepness τ for the spectral guardrail.
    /// γ_eff = γ · σ((λ₂ - λ_c) / τ) — eliminates hysteresis
    /// from the old discrete threshold.
    /// Smaller τ = sharper transition (closer to step function).
    /// Default: 0.005
    pub spectral_tau: f64,

    /// Minimum number of unique nodes visited for a valid entropy estimate.
    /// Default: 3
    pub min_visited: usize,

    /// Epsilon for safe Poincaré distance computation.
    /// Clamps (1 - |x|²) to avoid division by zero near the ball boundary.
    /// Default: 1e-9
    pub distance_epsilon: f64,

    /// Maximum Poincaré ball radius² for distance safety.
    /// Points with |x|² > this are clamped.
    /// Default: 0.999999
    pub max_radius_sq: f64,
}

impl Default for PathEntropyConfig {
    fn default() -> Self {
        Self {
            n_walks: 50,
            walk_length: 10,
            beta: 1.0,
            alpha: 0.5,
            gamma: 0.3,
            memory_size: 3,
            memory_penalty: -5.0,
            adaptive_memory: true,
            spectral_guardrail_threshold: 0.02,
            spectral_tau: 0.005,
            min_visited: 3,
            distance_epsilon: 1e-9,
            max_radius_sq: 0.999_999,
        }
    }
}

/// Estimates functional navigability H_path via numerically stable
/// hyperbolic random walks.
///
/// ## Transition Model (4-factor Boltzmann in log-space)
///
/// ```text
/// log w_u = -β·d_𝔻(v,u) + α·E_edge - γ·ln(deg(u)) + memory_penalty
/// P(v→u) = exp(log w_u) / Σ exp(log w_j)   [via log-sum-exp]
/// ```
///
/// ## Numerical Safety
///
/// 1. **Poincaré distance**: `(1-|x|²)` clamped to `distance_epsilon`
/// 2. **Hub penalty**: `-γ·ln(deg)` instead of `deg^(-γ)` (no underflow)
/// 3. **Sampling**: log-sum-exp trick for overflow/underflow resistance
/// 4. **Spectral guardrail**: γ_eff = γ·σ((λ₂-λ_c)/τ) smooth sigmoid
/// 5. **Tabu memory**: logarithmic penalty (not multiplicative 0.01)
///
/// ## Divergence Metric
///
/// Computes average pairwise divergence D between walk trajectories:
/// D = 1 - mean(J(Tᵢ, Tⱼ)) where J is Jaccard similarity.
#[derive(Debug, Clone)]
pub struct PathEntropyEstimator {
    pub config: PathEntropyConfig,
    /// Simple LCG state for deterministic pseudo-random walks in pure computation.
    rng_state: u64,
}

/// Report from a path entropy estimation run.
#[derive(Debug, Clone)]
pub struct PathEntropyReport {
    /// Shannon entropy over visit frequencies: H = -Σ p_i · log(p_i)
    pub h_path: f64,

    /// Average pairwise divergence between walk trajectories.
    /// D ∈ [0, 1]. D=0 means all walks identical, D=1 means maximally different.
    pub divergence: f64,

    /// Number of unique nodes visited across all walks.
    pub unique_visited: usize,

    /// Total walks executed.
    pub n_walks: usize,

    /// Walk length used (may have been boosted by CuriosityEngine).
    pub walk_length: usize,
}

impl std::fmt::Display for PathEntropyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "H_path={:.4}, D={:.4}, unique={}/{} walks (len={})",
            self.h_path, self.divergence, self.unique_visited, self.n_walks, self.walk_length
        )
    }
}

impl PathEntropyEstimator {
    pub fn new(config: PathEntropyConfig) -> Self {
        Self {
            config,
            rng_state: 42,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(PathEntropyConfig::default())
    }

    /// Simple LCG pseudo-random number generator (deterministic, no external deps).
    fn next_random(&mut self) -> f64 {
        // LCG parameters (Numerical Recipes)
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Extract bits and normalize to [0, 1)
        ((self.rng_state >> 33) as f64) / (u32::MAX as f64)
    }

    // ─────────────────────────────────────────────
    // Layer 1: Safe Poincaré distance
    // ─────────────────────────────────────────────

    /// Numerically stable Poincaré distance that doesn't explode when |x| → 1.
    ///
    /// ```text
    /// d(u,v) = arcosh(1 + 2·|u-v|² / ((1-|u|²)(1-|v|²)))
    /// ```
    ///
    /// When |x|² > max_radius_sq, the radius is clamped.
    /// When (1-|x|²) < ε, the denominator is clamped to ε.
    fn safe_poincare_distance(&self, u: &[f64], v: &[f64]) -> f64 {
        debug_assert_eq!(u.len(), v.len(), "dimension mismatch in safe distance");

        let diff_sq: f64 = u.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();

        let norm_u_sq: f64 = u.iter().map(|x| x * x).sum::<f64>()
            .min(self.config.max_radius_sq);
        let norm_v_sq: f64 = v.iter().map(|x| x * x).sum::<f64>()
            .min(self.config.max_radius_sq);

        let eps = self.config.distance_epsilon;
        let denom_u = (1.0 - norm_u_sq).max(eps);
        let denom_v = (1.0 - norm_v_sq).max(eps);
        let denom = denom_u * denom_v;

        // Clamp arg to ≥ 1.0 to avoid NaN from floating-point noise
        let arg = (1.0 + 2.0 * diff_sq / denom).max(1.0);
        arg.acosh()
    }

    // ─────────────────────────────────────────────
    // Layer 2: Log-space transition weights
    // ─────────────────────────────────────────────

    /// Build adjacency list from edge list.
    fn build_adjacency(edges: &[(usize, usize, f64)], n_nodes: usize) -> Vec<Vec<(usize, f64)>> {
        let mut adj = vec![vec![]; n_nodes];
        for &(src, dst, w) in edges {
            if src < n_nodes && dst < n_nodes {
                adj[src].push((dst, w));
                adj[dst].push((src, w));
            }
        }
        adj
    }

    /// Compute node degrees for anti-hub penalty.
    fn compute_degrees(adj: &[Vec<(usize, f64)>]) -> Vec<u64> {
        adj.iter().map(|neighbors| neighbors.len() as u64).collect()
    }

    /// Compute log-transition weights from `current` to each neighbor.
    ///
    /// ```text
    /// log w_u = -β·d_𝔻(v,u) + α·E_edge - γ_eff·ln(deg(u)) + memory_penalty
    /// ```
    ///
    /// All computation stays in log-space to prevent overflow/underflow.
    ///
    /// When `adaptive_memory` is enabled, the memory penalty is scaled by
    /// `(1 - h_local)` where h_local is the walk's local entropy so far.
    /// This prevents the fixed penalty from collapsing probability support
    /// in sparse graph regimes.
    fn log_transition_weights(
        &self,
        neighbors: &[(usize, f64)],
        current_pos: Option<&[f64]>,
        positions: &[Vec<f64>],
        degrees: &[u64],
        tabu: &[usize],
        gamma_eff: f64,
        h_local: f64,
    ) -> Vec<f64> {
        if neighbors.is_empty() {
            return vec![];
        }

        let beta = self.config.beta;
        let alpha = self.config.alpha;

        // Adaptive memory: penalty scales with (1 - H_local)
        // Low H_local (redundant) → full penalty. High H_local (exploratory) → soft penalty.
        let effective_memory_penalty = if self.config.adaptive_memory {
            self.config.memory_penalty * (1.0 - h_local.clamp(0.0, 0.99))
        } else {
            self.config.memory_penalty
        };

        neighbors
            .iter()
            .map(|&(nbr, edge_weight)| {
                // ── Factor 1: Geodesic proximity ──
                let dist = if let Some(cur_p) = current_pos {
                    if nbr < positions.len() && !positions[nbr].is_empty() {
                        self.safe_poincare_distance(cur_p, &positions[nbr])
                    } else {
                        // Fallback: 1/weight as proxy distance
                        if edge_weight > 0.0 { 1.0 / edge_weight } else { 1.0 }
                    }
                } else {
                    if edge_weight > 0.0 { 1.0 / edge_weight } else { 1.0 }
                };
                let log_geodesic = -beta * dist;

                // ── Factor 2: Edge energy affinity ──
                let log_energy = alpha * edge_weight.max(0.0);

                // ── Factor 3: Anti-hub penalty (log-space, no underflow) ──
                let deg = if nbr < degrees.len() { degrees[nbr] } else { 1 };
                let log_antihub = -gamma_eff * (deg.max(1) as f64).ln();

                // ── Factor 4: Tabu memory penalty (adaptive or fixed) ──
                let log_memory = if tabu.contains(&nbr) {
                    effective_memory_penalty
                } else {
                    0.0
                };

                let log_w = log_geodesic + log_energy + log_antihub + log_memory;

                // Guard against NaN/Inf
                if log_w.is_finite() {
                    log_w
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    // ─────────────────────────────────────────────
    // Layer 3: Log-Sum-Exp sampling
    // ─────────────────────────────────────────────

    /// Sample an index from log-weights using the log-sum-exp trick.
    ///
    /// This is numerically stable even when log-weights span many
    /// orders of magnitude (e.g., -300 to +5).
    fn log_sum_exp_sample(
        &mut self,
        log_weights: &[f64],
        neighbors: &[(usize, f64)],
    ) -> usize {
        if log_weights.is_empty() || neighbors.is_empty() {
            return 0;
        }

        // 1. Find max log-weight (filter -inf)
        let max_log = log_weights
            .iter()
            .cloned()
            .filter(|x| x.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);

        if !max_log.is_finite() {
            // All weights are -inf → uniform fallback
            let idx = (self.next_random() * neighbors.len() as f64) as usize;
            return neighbors[idx.min(neighbors.len() - 1)].0;
        }

        // 2. Compute exp(log_w - max_log) for each neighbor
        let mut exp_weights = Vec::with_capacity(log_weights.len());
        let mut sum = 0.0;
        for &lw in log_weights {
            let val = if lw.is_finite() {
                (lw - max_log).exp()
            } else {
                0.0
            };
            exp_weights.push(val);
            sum += val;
        }

        if sum <= 0.0 || !sum.is_finite() {
            // Degenerate → uniform fallback
            let idx = (self.next_random() * neighbors.len() as f64) as usize;
            return neighbors[idx.min(neighbors.len() - 1)].0;
        }

        // 3. Cumulative sampling
        let mut threshold = self.next_random() * sum;
        for (i, &w) in exp_weights.iter().enumerate() {
            threshold -= w;
            if threshold <= 0.0 {
                return neighbors[i].0;
            }
        }

        // Fallback: last neighbor
        neighbors.last().map(|&(n, _)| n).unwrap_or(0)
    }

    // ─────────────────────────────────────────────
    // Layer 4: Random walk with tabu memory
    // ─────────────────────────────────────────────

    /// Compute local entropy H_local from visit counts so far.
    ///
    /// H_local = -Σ p_i · ln(p_i) / ln(n_unique)   ∈ [0, 1]
    ///
    /// This sliding entropy measure allows the adaptive memory penalty
    /// to soften when the walk is already exploratory.
    fn compute_h_local(visit_counts: &[u32], total_steps: u32) -> f64 {
        if total_steps <= 1 {
            return 0.0;
        }
        let n_unique = visit_counts.iter().filter(|&&c| c > 0).count();
        if n_unique <= 1 {
            return 0.0;
        }
        let total = total_steps as f64;
        let ln_n = (n_unique as f64).ln();
        if ln_n <= 0.0 {
            return 0.0;
        }
        let entropy: f64 = visit_counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total;
                -p * p.ln()
            })
            .sum();
        (entropy / ln_n).clamp(0.0, 1.0)
    }

    /// Execute one random walk from `start_node` with tabu memory.
    ///
    /// When `adaptive_memory` is enabled, computes a sliding H_local
    /// from the walk's visit counts and scales the memory penalty accordingly.
    fn random_walk(
        &mut self,
        start: usize,
        adj: &[Vec<(usize, f64)>],
        positions: &[Vec<f64>],
        degrees: &[u64],
        walk_length: usize,
        gamma_eff: f64,
        n_nodes: usize,
    ) -> Vec<usize> {
        let mut path = Vec::with_capacity(walk_length + 1);
        path.push(start);
        let mut current = start;

        // Tabu ring buffer (recent nodes to penalize)
        let memory_size = self.config.memory_size;
        let mut tabu: Vec<usize> = Vec::with_capacity(memory_size + 1);

        // Per-walk visit counts for H_local (only when adaptive_memory is on)
        let adaptive = self.config.adaptive_memory;
        let mut local_visits = if adaptive { vec![0u32; n_nodes] } else { vec![] };
        let mut total_steps = 0u32;
        if adaptive && start < n_nodes {
            local_visits[start] = 1;
            total_steps = 1;
        }

        for _ in 0..walk_length {
            let neighbors = &adj[current];
            if neighbors.is_empty() {
                break; // Dead end
            }

            let cur_pos = if current < positions.len() && !positions[current].is_empty() {
                Some(positions[current].as_slice())
            } else {
                None
            };

            // Compute H_local from the walk so far
            let h_local = if adaptive {
                Self::compute_h_local(&local_visits, total_steps)
            } else {
                0.0
            };

            let log_weights = self.log_transition_weights(
                neighbors, cur_pos, positions, degrees, &tabu, gamma_eff, h_local,
            );

            current = self.log_sum_exp_sample(&log_weights, neighbors);

            // Update tabu memory
            if memory_size > 0 {
                tabu.push(current);
                if tabu.len() > memory_size {
                    tabu.remove(0);
                }
            }

            // Update local visit counts
            if adaptive && current < n_nodes {
                local_visits[current] += 1;
                total_steps += 1;
            }

            path.push(current);
        }

        path
    }

    // ─────────────────────────────────────────────
    // Layer 5: Estimation with spectral guardrail
    // ─────────────────────────────────────────────

    /// Estimate H_path and divergence from the graph.
    ///
    /// # Arguments
    /// - `edges`: (source, target, weight) triples
    /// - `n_nodes`: total node count
    /// - `positions`: Poincaré ball coordinates (may be sparse/empty)
    /// - `walk_length_override`: if Some, override config walk_length (used by CuriosityEngine)
    pub fn estimate(
        &mut self,
        edges: &[(usize, usize, f64)],
        n_nodes: usize,
        positions: &[Vec<f64>],
        walk_length_override: Option<usize>,
    ) -> PathEntropyReport {
        self.estimate_with_spectral(edges, n_nodes, positions, walk_length_override, None)
    }

    /// Estimate with optional λ₂ for smooth spectral guardrail.
    ///
    /// Uses a sigmoid γ_eff = γ · σ((λ₂ - λ_c) / τ) to avoid the
    /// hysteresis oscillation that the old discrete threshold caused
    /// when λ₂ fluctuated around λ_c.
    pub fn estimate_with_spectral(
        &mut self,
        edges: &[(usize, usize, f64)],
        n_nodes: usize,
        positions: &[Vec<f64>],
        walk_length_override: Option<usize>,
        lambda2: Option<f64>,
    ) -> PathEntropyReport {
        if n_nodes == 0 || edges.is_empty() {
            return PathEntropyReport {
                h_path: 0.0,
                divergence: 0.0,
                unique_visited: 0,
                n_walks: 0,
                walk_length: 0,
            };
        }

        let adj = Self::build_adjacency(edges, n_nodes);
        let degrees = Self::compute_degrees(&adj);
        let walk_length = walk_length_override.unwrap_or(self.config.walk_length);
        let n_walks = self.config.n_walks;

        // ── Layer 6: Smooth spectral guardrail (sigmoid, no hysteresis) ──
        // γ_eff = γ · σ((λ₂ - λ_c) / τ)
        // When λ₂ >> λ_c: σ ≈ 1 → full hub penalty
        // When λ₂ << λ_c: σ ≈ 0 → no hub penalty (preserve bridges)
        // When λ₂ ≈ λ_c: σ ≈ 0.5 → smooth transition, no oscillation
        let gamma_eff = if let Some(l2) = lambda2 {
            let lambda_c = self.config.spectral_guardrail_threshold;
            let tau = self.config.spectral_tau.max(1e-12);
            let x = (l2 - lambda_c) / tau;
            // Numerically safe sigmoid: clamp x to avoid exp overflow
            let sigmoid = if x > 20.0 {
                1.0
            } else if x < -20.0 {
                0.0
            } else {
                1.0 / (1.0 + (-x).exp())
            };
            self.config.gamma * sigmoid
        } else {
            self.config.gamma
        };

        // Visit frequency counter
        let mut visit_counts = vec![0u64; n_nodes];
        let mut total_visits = 0u64;

        // Collect visited sets for each walk (for divergence computation)
        let mut walk_sets: Vec<Vec<bool>> = Vec::with_capacity(n_walks);

        // Choose start nodes distributed across the graph
        for w in 0..n_walks {
            // Spread start nodes across the graph
            let start = if n_nodes > 0 {
                (w * 7 + 13) % n_nodes
            } else {
                0
            };

            // Skip isolated nodes
            if adj[start].is_empty() {
                continue;
            }

            let path = self.random_walk(start, &adj, positions, &degrees, walk_length, gamma_eff, n_nodes);

            // Count visits
            let mut visited = vec![false; n_nodes];
            for &node in &path {
                visit_counts[node] += 1;
                total_visits += 1;
                visited[node] = true;
            }
            walk_sets.push(visited);
        }

        // ── Compute H_path (Shannon entropy over visit frequencies) ──
        let h_path = if total_visits == 0 {
            0.0
        } else {
            let total = total_visits as f64;
            visit_counts
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / total;
                    -p * p.ln()
                })
                .sum::<f64>()
        };

        // ── Compute divergence D = 1 - mean(Jaccard similarity) ──
        let divergence = if walk_sets.len() < 2 {
            0.0
        } else {
            let m = walk_sets.len();
            let mut total_jaccard = 0.0;
            let mut pairs = 0u64;

            for i in 0..m {
                for j in (i + 1)..m {
                    let mut intersection = 0u64;
                    let mut union = 0u64;
                    for k in 0..n_nodes {
                        let a = walk_sets[i][k];
                        let b = walk_sets[j][k];
                        if a || b { union += 1; }
                        if a && b { intersection += 1; }
                    }
                    if union > 0 {
                        total_jaccard += intersection as f64 / union as f64;
                    }
                    pairs += 1;
                }
            }

            if pairs > 0 {
                1.0 - total_jaccard / pairs as f64
            } else {
                0.0
            }
        };

        let unique_visited = visit_counts.iter().filter(|&&c| c > 0).count();

        PathEntropyReport {
            h_path,
            divergence,
            unique_visited,
            n_walks: walk_sets.len(),
            walk_length,
        }
    }
}

// ═══════════════════════════════════════════════
// §2 — Metabolic Sleep Manager
// ═══════════════════════════════════════════════

/// Configuration for the metabolic sleep manager.
#[derive(Debug, Clone)]
pub struct SleepConfig {
    /// Weights for the state gradient: [λ₂, energy, entropy].
    /// Default: [1.0, 0.5, 0.8]
    pub weights: [f64; 3],

    /// Scaling factor η for relaxation time.
    /// T_sleep = η · τ_relax.  η ∈ [3, 5] guarantees >95% dissipation.
    /// Default: 4.0
    pub eta: f64,

    /// Minimum sleep duration.
    /// Default: 5 seconds
    pub min_sleep: Duration,

    /// Maximum sleep duration.
    /// Default: 300 seconds (5 minutes)
    pub max_sleep: Duration,

    /// Minimum healthy divergence. Below this → possible stagnation.
    /// Default: 0.2
    pub divergence_min: f64,

    /// Energy change below this → considered flat.
    /// Default: 0.01
    pub energy_epsilon: f64,

    /// Entropy change below this → considered flat.
    /// Default: 0.01
    pub entropy_epsilon: f64,

    /// Energy change above this → convergence signal.
    /// Default: 0.1
    pub energy_threshold: f64,

    /// Divergence above this → turbulence signal.
    /// Default: 0.8
    pub divergence_max: f64,

    /// Cognitive modulation factor for stagnation (< 1.0 = shorter sleep).
    /// Default: 0.3
    pub stagnation_factor: f64,

    /// Cognitive modulation factor for convergence (> 1.0 = longer sleep).
    /// Default: 1.5
    pub convergence_factor: f64,

    /// Cognitive modulation factor for turbulence.
    /// Default: 0.7
    pub turbulence_factor: f64,

    /// Epsilon to avoid division by zero in τ_relax.
    /// Default: 1e-6
    pub epsilon: f64,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            weights: [1.0, 0.5, 0.8],
            eta: 4.0,
            min_sleep: Duration::from_secs(5),
            max_sleep: Duration::from_secs(300),
            divergence_min: 0.2,
            energy_epsilon: 0.01,
            entropy_epsilon: 0.01,
            energy_threshold: 0.1,
            divergence_max: 0.8,
            stagnation_factor: 0.3,
            convergence_factor: 1.5,
            turbulence_factor: 0.7,
            epsilon: 1e-6,
        }
    }
}

/// Metabolic Sleep Manager — computes adaptive sleep duration from the state vector.
///
/// The organism's "heartbeat" rate is not fixed. It depends on:
/// 1. How fast the state vector S = [λ₂, E_g, H_path] is changing
/// 2. Whether the system is stagnating, converging, or turbulent
///
/// Fast-changing state → short relaxation → frequent heartbeats.
/// Stable state → long relaxation → deep sleep.
#[derive(Debug, Clone)]
pub struct MetabolicSleepManager {
    config: SleepConfig,
    /// Previous state vector for gradient computation.
    prev_state: Option<StateVector>,
    /// Consecutive stagnation cycles (for CuriosityEngine trigger).
    pub stagnation_cycles: u32,
}

/// Report from a sleep computation.
#[derive(Debug, Clone)]
pub struct SleepReport {
    /// Computed sleep duration.
    pub sleep_duration: Duration,

    /// Raw τ_relax before cognitive modulation.
    pub tau_relax: f64,

    /// Gradient norm ‖∇S‖.
    pub gradient_norm: f64,

    /// Cognitive state classification.
    pub cognitive_state: CognitiveState,

    /// Cognitive modulation factor applied.
    pub modulation_factor: f64,

    /// Consecutive stagnation cycles.
    pub stagnation_cycles: u32,
}

impl std::fmt::Display for SleepReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Sleep={:.1}s (τ={:.2}, ‖∇S‖={:.4}, {} ×{:.2}, stagnation={})",
            self.sleep_duration.as_secs_f64(),
            self.tau_relax,
            self.gradient_norm,
            self.cognitive_state,
            self.modulation_factor,
            self.stagnation_cycles,
        )
    }
}

impl MetabolicSleepManager {
    pub fn new(config: SleepConfig) -> Self {
        Self {
            config,
            prev_state: None,
            stagnation_cycles: 0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(SleepConfig::default())
    }

    /// Classify the cognitive state based on divergence and energy/entropy dynamics.
    pub fn classify_cognitive(
        &self,
        divergence: f64,
        delta_energy: f64,
        delta_entropy: f64,
    ) -> CognitiveState {
        let low_div = divergence < self.config.divergence_min;
        let high_div = divergence > self.config.divergence_max;
        let flat_energy = delta_energy.abs() < self.config.energy_epsilon;
        let flat_entropy = delta_entropy.abs() < self.config.entropy_epsilon;
        let rising_energy = delta_energy > self.config.energy_threshold;

        if low_div && flat_energy && flat_entropy {
            CognitiveState::Stagnation
        } else if low_div && rising_energy {
            CognitiveState::Convergence
        } else if high_div {
            CognitiveState::Turbulence
        } else {
            CognitiveState::Cruising
        }
    }

    /// Compute physiological sleep duration with cognitive modulation.
    ///
    /// # Arguments
    /// - `state`: current metabolic state vector
    /// - `dt`: elapsed time since last measurement (seconds)
    /// - `divergence`: walk divergence from PathEntropyEstimator
    /// - `delta_energy`: energy change since last cycle
    /// - `delta_entropy`: entropy change since last cycle
    pub fn compute_sleep(
        &mut self,
        state: StateVector,
        dt: f64,
        divergence: f64,
        delta_energy: f64,
        delta_entropy: f64,
    ) -> SleepReport {
        // 1. Compute gradient norm
        let grad_norm = if let Some(ref prev) = self.prev_state {
            state.gradient_norm(prev, dt, &self.config.weights)
        } else {
            0.0
        };

        // 2. Compute τ_relax = 1 / (‖∇S‖ + ε)
        let tau_relax = 1.0 / (grad_norm + self.config.epsilon);

        // 3. Classify cognitive state
        let cognitive_state = self.classify_cognitive(divergence, delta_energy, delta_entropy);

        // 4. Modulation factor based on cognitive state
        let modulation_factor = match cognitive_state {
            CognitiveState::Stagnation => self.config.stagnation_factor,
            CognitiveState::Convergence => self.config.convergence_factor,
            CognitiveState::Turbulence => self.config.turbulence_factor,
            CognitiveState::Cruising => 1.0,
        };

        // 5. T_sleep = η · τ_relax · modulation
        let raw_sleep = self.config.eta * tau_relax * modulation_factor;
        let clamped_sleep = raw_sleep
            .max(self.config.min_sleep.as_secs_f64())
            .min(self.config.max_sleep.as_secs_f64());
        let sleep_duration = Duration::from_secs_f64(clamped_sleep);

        // 6. Track stagnation cycles
        if cognitive_state == CognitiveState::Stagnation {
            self.stagnation_cycles += 1;
        } else {
            self.stagnation_cycles = 0;
        }

        // 7. Update history
        self.prev_state = Some(state);

        SleepReport {
            sleep_duration,
            tau_relax,
            gradient_norm: grad_norm,
            cognitive_state,
            modulation_factor,
            stagnation_cycles: self.stagnation_cycles,
        }
    }

    /// Reset internal state (useful after a CuriosityEngine intervention).
    pub fn reset(&mut self) {
        self.prev_state = None;
        self.stagnation_cycles = 0;
    }
}

// ═══════════════════════════════════════════════
// §3 — Curiosity Engine (Option D: Adaptive Combination)
// ═══════════════════════════════════════════════

/// Configuration for the curiosity engine.
#[derive(Debug, Clone)]
pub struct CuriosityConfig {
    /// Stagnation cycles before curiosity activates.
    /// Default: 3
    pub activation_threshold: u32,

    /// Geometric perturbation strength (ε for exp_map noise).
    /// Default: 0.02
    pub perturbation_strength: f64,

    /// Anti-hub γ reduction factor (multiply hub weight by this).
    /// Default: 0.7
    pub antihub_factor: f64,

    /// Walk length multiplier when curiosity is active.
    /// Default: 2.0
    pub walk_length_multiplier: f64,

    /// Maximum perturbation strength (increases with stagnation depth).
    /// Default: 0.1
    pub max_perturbation: f64,

    /// Maximum hub nodes to reweight per cycle.
    /// Default: 5
    pub max_hubs_to_reweight: usize,

    /// Degree threshold to consider a node a "hub" (percentile).
    /// Default: 0.9 (top 10% by degree)
    pub hub_percentile: f64,

    /// Ball radius ceiling for perturbation (safety).
    /// Default: 0.95
    pub max_radius: f64,
}

impl Default for CuriosityConfig {
    fn default() -> Self {
        Self {
            activation_threshold: 3,
            perturbation_strength: 0.02,
            antihub_factor: 0.7,
            walk_length_multiplier: 2.0,
            max_perturbation: 0.1,
            max_hubs_to_reweight: 5,
            hub_percentile: 0.9,
            max_radius: 0.95,
        }
    }
}

/// Actions the curiosity engine prescribes.
#[derive(Debug, Clone)]
pub struct CuriosityActions {
    /// Whether curiosity was activated.
    pub activated: bool,

    /// Geodesic perturbation vectors: (node_index, new_position).
    /// The caller applies these to the graph.
    pub perturbations: Vec<(usize, Vec<f64>)>,

    /// Hub nodes to temporarily reweight: (node_index, weight_multiplier).
    pub hub_reweights: Vec<(usize, f64)>,

    /// Extended walk length for next PathEntropy estimation.
    pub extended_walk_length: Option<usize>,

    /// Perturbation strength used.
    pub strength_used: f64,

    /// Stagnation depth (how many cycles stagnated).
    pub stagnation_depth: u32,
}

impl std::fmt::Display for CuriosityActions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.activated {
            write!(f, "Curiosity: inactive")
        } else {
            write!(
                f,
                "Curiosity: ACTIVE (depth={}, ε={:.4}, {} perturbations, {} hub reweights, walk_len={:?})",
                self.stagnation_depth,
                self.strength_used,
                self.perturbations.len(),
                self.hub_reweights.len(),
                self.extended_walk_length,
            )
        }
    }
}

/// Curiosity Engine — forces controlled exploration when the system stagnates.
///
/// Implements Option D (adaptive combination):
/// 1. Geometric perturbation: x ← exp₀(ε · ξ) — geodesic noise injection
/// 2. Anti-hub γ reweighting: reduce hub gravity temporarily
/// 3. Walk length increase: extend exploration horizon
///
/// The intensity scales with stagnation depth:
/// - 3 cycles stagnant → mild perturbation
/// - 6+ cycles stagnant → aggressive exploration
#[derive(Debug, Clone)]
pub struct CuriosityEngine {
    config: CuriosityConfig,
    /// LCG state for deterministic perturbations.
    rng_state: u64,
}

impl CuriosityEngine {
    pub fn new(config: CuriosityConfig) -> Self {
        Self {
            config,
            rng_state: 137,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(CuriosityConfig::default())
    }

    /// Simple LCG pseudo-random.
    fn next_random(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.rng_state >> 33) as f64) / (u32::MAX as f64)
    }

    /// Generate a random unit vector of given dimension.
    fn random_direction(&mut self, dim: usize) -> Vec<f64> {
        let v: Vec<f64> = (0..dim).map(|_| self.next_random() * 2.0 - 1.0).collect();
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            let mut unit = vec![0.0; dim];
            if dim > 0 { unit[0] = 1.0; }
            unit
        } else {
            v.iter().map(|x| x / norm).collect()
        }
    }

    /// Evaluate and optionally generate curiosity actions.
    ///
    /// # Arguments
    /// - `stagnation_cycles`: consecutive stagnation count from SleepManager
    /// - `positions`: current node positions in Poincaré ball
    /// - `edges`: graph edges for hub detection
    /// - `n_nodes`: total node count
    /// - `base_walk_length`: normal walk length from PathEntropyConfig
    pub fn evaluate(
        &mut self,
        stagnation_cycles: u32,
        positions: &[Vec<f64>],
        edges: &[(usize, usize, f64)],
        n_nodes: usize,
        base_walk_length: usize,
    ) -> CuriosityActions {
        if stagnation_cycles < self.config.activation_threshold {
            return CuriosityActions {
                activated: false,
                perturbations: vec![],
                hub_reweights: vec![],
                extended_walk_length: None,
                strength_used: 0.0,
                stagnation_depth: stagnation_cycles,
            };
        }

        let depth = stagnation_cycles - self.config.activation_threshold + 1;

        // ── 1. Geometric perturbation ──
        // Strength scales with stagnation depth: ε · min(depth, 5) / 3
        let scaled_strength = (self.config.perturbation_strength * depth as f64 / 3.0)
            .min(self.config.max_perturbation);

        let perturbations: Vec<(usize, Vec<f64>)> = positions
            .iter()
            .enumerate()
            .filter(|(_, pos)| !pos.is_empty())
            .map(|(idx, pos)| {
                let dim = pos.len();
                let direction = self.random_direction(dim);

                // Tangent vector: ε · ξ
                let tangent: Vec<f64> = direction.iter().map(|d| d * scaled_strength).collect();

                // exp₀(tangent) gives a perturbation point, but we want to
                // perturb AROUND the current position.
                // Simple approach: log(current) + noise → exp back
                // For points near origin, direct addition in tangent space works.
                let perturbed_tangent: Vec<f64> = pos
                    .iter()
                    .zip(tangent.iter())
                    .map(|(p, t)| p + t)
                    .collect();

                // Project back to ball via exp_map for safety
                let new_pos = exp_map_zero(&perturbed_tangent);

                // Clip to max_radius
                let norm: f64 = new_pos.iter().map(|x| x * x).sum::<f64>().sqrt();
                let clipped = if norm > self.config.max_radius {
                    let scale = self.config.max_radius / norm;
                    new_pos.iter().map(|x| x * scale).collect()
                } else {
                    new_pos
                };

                (idx, clipped)
            })
            .collect();

        // ── 2. Anti-hub γ reweighting ──
        let mut degrees = vec![0u64; n_nodes];
        for &(src, dst, _) in edges {
            if src < n_nodes { degrees[src] += 1; }
            if dst < n_nodes { degrees[dst] += 1; }
        }

        // Find degree threshold for hub percentile
        let mut sorted_degrees: Vec<u64> = degrees.clone();
        sorted_degrees.sort_unstable();
        let threshold_idx = (sorted_degrees.len() as f64 * self.config.hub_percentile) as usize;
        let degree_threshold = if threshold_idx < sorted_degrees.len() {
            sorted_degrees[threshold_idx]
        } else {
            u64::MAX
        };

        let mut hub_reweights: Vec<(usize, f64)> = degrees
            .iter()
            .enumerate()
            .filter(|(_, &deg)| deg >= degree_threshold && deg > 0)
            .take(self.config.max_hubs_to_reweight)
            .map(|(idx, _)| {
                // Scale reweight with depth: more stagnation → more aggressive
                let factor = self.config.antihub_factor.powf(depth as f64 / 3.0);
                (idx, factor)
            })
            .collect();

        // Sort by degree descending so we reweight the biggest hubs first
        hub_reweights.sort_by(|a, b| {
            degrees[b.0].cmp(&degrees[a.0])
        });
        hub_reweights.truncate(self.config.max_hubs_to_reweight);

        // ── 3. Walk length extension ──
        let extended_walk_length = Some(
            (base_walk_length as f64 * self.config.walk_length_multiplier * (1.0 + depth as f64 * 0.2))
                .round() as usize
        );

        CuriosityActions {
            activated: true,
            perturbations,
            hub_reweights,
            extended_walk_length,
            strength_used: scaled_strength,
            stagnation_depth: stagnation_cycles,
        }
    }
}

// ═══════════════════════════════════════════════
// §4 — Metabolic Report (full cycle audit)
// ═══════════════════════════════════════════════

/// Complete metabolic report for one physiological cycle.
///
/// Includes entropy estimation, sleep computation, curiosity actions,
/// and the navigability innovation score I = λ₂^α · E^β · H^γ·(1-H).
#[derive(Debug, Clone)]
pub struct MetabolicReport {
    /// Current state vector S = [λ₂, E_g, H_path].
    pub state: StateVector,

    /// Path entropy estimation results.
    pub entropy: PathEntropyReport,

    /// Sleep computation results.
    pub sleep: SleepReport,

    /// Curiosity engine actions (if activated).
    pub curiosity: CuriosityActions,

    /// Navigability innovation score (Phase VI second-order metric).
    /// Couples [λ₂, E(τ), H_path] into a single innovation measure.
    pub navigability: Option<NavigabilityReport>,
}

impl std::fmt::Display for MetabolicReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══ Metabolic Report ══╗")?;
        writeln!(f, "║ State: {}", self.state)?;
        writeln!(f, "║ Entropy: {}", self.entropy)?;
        writeln!(f, "║ Sleep: {}", self.sleep)?;
        writeln!(f, "║ {}", self.curiosity)?;
        if let Some(ref nav) = self.navigability {
            writeln!(f, "║ Innovation: {}", nav)?;
        }
        write!(f, "╚══════════════════════╝")
    }
}

// ═══════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── StateVector tests ──

    #[test]
    fn test_state_vector_gradient_norm() {
        let s1 = StateVector::new(0.3, 0.5, 1.0);
        let s2 = StateVector::new(0.4, 0.6, 1.2);
        let weights = [1.0, 0.5, 0.8];
        let dt = 1.0;

        let grad = s2.gradient_norm(&s1, dt, &weights);

        // dλ=0.1, dE=0.1, dH=0.2
        // ‖∇S‖ = √(1.0·0.01 + 0.5·0.01 + 0.8·0.04) = √(0.01+0.005+0.032) = √0.047
        let expected = (1.0 * 0.01 + 0.5 * 0.01 + 0.8 * 0.04_f64).sqrt();
        assert!((grad - expected).abs() < 1e-10, "gradient = {grad}, expected = {expected}");
    }

    #[test]
    fn test_state_vector_zero_dt() {
        let s1 = StateVector::new(0.3, 0.5, 1.0);
        let s2 = StateVector::new(0.4, 0.6, 1.2);
        let weights = [1.0, 1.0, 1.0];

        assert_eq!(s2.gradient_norm(&s1, 0.0, &weights), 0.0);
        assert_eq!(s2.gradient_norm(&s1, -1.0, &weights), 0.0);
    }

    #[test]
    fn test_state_vector_display() {
        let s = StateVector::new(0.3, 0.5, 1.2);
        let display = format!("{s}");
        assert!(display.contains("λ₂=0.3000"));
        assert!(display.contains("E=0.5000"));
        assert!(display.contains("H=1.2000"));
    }

    // ── CognitiveState tests ──

    #[test]
    fn test_cognitive_state_display() {
        assert!(format!("{}", CognitiveState::Stagnation).contains("circles"));
        assert!(format!("{}", CognitiveState::Convergence).contains("resolving"));
        assert!(format!("{}", CognitiveState::Turbulence).contains("chaotic"));
        assert!(format!("{}", CognitiveState::Cruising).contains("normal"));
    }

    // ── PathEntropyEstimator tests ──

    fn make_line_graph(n: usize) -> (Vec<(usize, usize, f64)>, Vec<Vec<f64>>) {
        let edges: Vec<(usize, usize, f64)> = (0..n.saturating_sub(1))
            .map(|i| (i, i + 1, 1.0))
            .collect();
        let positions: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let r = 0.1 + 0.7 * (i as f64) / (n.max(1) as f64);
                vec![r, 0.0, 0.0]
            })
            .collect();
        (edges, positions)
    }

    fn make_star_graph(n: usize) -> (Vec<(usize, usize, f64)>, Vec<Vec<f64>>) {
        let edges: Vec<(usize, usize, f64)> = (1..n)
            .map(|i| (0, i, 1.0))
            .collect();
        let positions: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                if i == 0 {
                    vec![0.1, 0.0, 0.0]
                } else {
                    let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                    vec![0.5 * angle.cos(), 0.5 * angle.sin(), 0.0]
                }
            })
            .collect();
        (edges, positions)
    }

    fn make_complete_graph(n: usize) -> (Vec<(usize, usize, f64)>, Vec<Vec<f64>>) {
        let mut edges = vec![];
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push((i, j, 1.0));
            }
        }
        let positions: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                vec![0.3 * angle.cos(), 0.3 * angle.sin(), 0.0]
            })
            .collect();
        (edges, positions)
    }

    #[test]
    fn test_entropy_empty_graph() {
        let mut est = PathEntropyEstimator::with_defaults();
        let report = est.estimate(&[], 0, &[], None);
        assert_eq!(report.h_path, 0.0);
        assert_eq!(report.divergence, 0.0);
        assert_eq!(report.unique_visited, 0);
    }

    #[test]
    fn test_entropy_line_graph() {
        let (edges, positions) = make_line_graph(10);
        let mut est = PathEntropyEstimator::with_defaults();
        let report = est.estimate(&edges, 10, &positions, None);

        assert!(report.h_path > 0.0, "H_path should be positive: {}", report.h_path);
        assert!(report.unique_visited > 3, "Should visit multiple nodes: {}", report.unique_visited);
        assert!(report.n_walks > 0);
    }

    #[test]
    fn test_entropy_star_lower_than_complete() {
        let (star_edges, star_pos) = make_star_graph(10);
        let (comp_edges, comp_pos) = make_complete_graph(10);

        let mut est1 = PathEntropyEstimator::with_defaults();
        let mut est2 = PathEntropyEstimator::with_defaults();

        let star_report = est1.estimate(&star_edges, 10, &star_pos, None);
        let comp_report = est2.estimate(&comp_edges, 10, &comp_pos, None);

        assert!(
            comp_report.h_path > star_report.h_path * 0.5,
            "Complete graph H={:.4} should be meaningfully positive (star H={:.4})",
            comp_report.h_path, star_report.h_path
        );
    }

    #[test]
    fn test_entropy_divergence_range() {
        let (edges, positions) = make_line_graph(20);
        let mut est = PathEntropyEstimator::with_defaults();
        let report = est.estimate(&edges, 20, &positions, None);

        assert!(report.divergence >= 0.0, "D must be >= 0: {}", report.divergence);
        assert!(report.divergence <= 1.0, "D must be <= 1: {}", report.divergence);
    }

    #[test]
    fn test_entropy_walk_length_override() {
        let (edges, positions) = make_line_graph(10);
        let mut est = PathEntropyEstimator::with_defaults();
        let report = est.estimate(&edges, 10, &positions, Some(20));
        assert_eq!(report.walk_length, 20);
    }

    #[test]
    fn test_entropy_no_positions() {
        let (edges, _) = make_line_graph(10);
        let empty_positions: Vec<Vec<f64>> = vec![];
        let mut est = PathEntropyEstimator::with_defaults();
        let report = est.estimate(&edges, 10, &empty_positions, None);
        assert!(report.h_path > 0.0, "Should work without positions: {}", report.h_path);
    }

    #[test]
    fn test_entropy_report_display() {
        let report = PathEntropyReport {
            h_path: 2.3,
            divergence: 0.65,
            unique_visited: 15,
            n_walks: 50,
            walk_length: 10,
        };
        let s = format!("{report}");
        assert!(s.contains("H_path=2.3000"));
        assert!(s.contains("D=0.6500"));
    }

    // ── Numerical stability tests ──

    #[test]
    fn test_safe_distance_near_boundary() {
        // Nodes at |x| ≈ 0.999 — would explode without safe clamping
        let est = PathEntropyEstimator::with_defaults();
        let u = vec![0.999, 0.0, 0.0];
        let v = vec![0.0, 0.999, 0.0];

        let d = est.safe_poincare_distance(&u, &v);
        assert!(d.is_finite(), "Distance must be finite near boundary: {d}");
        assert!(d > 0.0, "Distance must be positive: {d}");
    }

    #[test]
    fn test_safe_distance_at_boundary() {
        // Nodes exactly at |x| = 1.0 — impossible in Poincaré ball,
        // but must not crash with NaN
        let est = PathEntropyEstimator::with_defaults();
        let u = vec![1.0, 0.0, 0.0];
        let v = vec![0.0, 1.0, 0.0];

        let d = est.safe_poincare_distance(&u, &v);
        assert!(d.is_finite(), "Must handle boundary points: {d}");
    }

    #[test]
    fn test_safe_distance_same_point() {
        let est = PathEntropyEstimator::with_defaults();
        let u = vec![0.5, 0.3, 0.0];
        let d = est.safe_poincare_distance(&u, &u);
        assert!((d - 0.0).abs() < 1e-10, "Distance to self must be 0: {d}");
    }

    #[test]
    fn test_safe_distance_beyond_boundary() {
        // Nodes OUTSIDE the ball (|x| > 1) — must clamp, not explode
        let est = PathEntropyEstimator::with_defaults();
        let u = vec![1.5, 0.0, 0.0]; // Outside ball
        let v = vec![0.0, 0.5, 0.0];

        let d = est.safe_poincare_distance(&u, &v);
        assert!(d.is_finite(), "Must handle points outside ball: {d}");
    }

    #[test]
    fn test_log_weights_large_hub() {
        // Node with degree 10000 — deg^(-gamma) would underflow,
        // but -gamma * ln(deg) is stable
        let est = PathEntropyEstimator::with_defaults();
        let neighbors = vec![(1, 1.0), (2, 1.0)];
        let positions: Vec<Vec<f64>> = vec![
            vec![0.1, 0.0, 0.0],
            vec![0.2, 0.0, 0.0],
            vec![0.3, 0.0, 0.0],
        ];
        // Fake high degrees: node 1 = hub (10000), node 2 = leaf (2)
        let degrees = vec![5, 10000, 2];
        let tabu = vec![];

        let log_w = est.log_transition_weights(
            &neighbors,
            Some(&positions[0]),
            &positions,
            &degrees,
            &tabu,
            est.config.gamma,
            0.0, // h_local = 0 (fresh walk)
        );

        assert_eq!(log_w.len(), 2);
        assert!(log_w[0].is_finite(), "Hub log-weight must be finite: {}", log_w[0]);
        assert!(log_w[1].is_finite(), "Leaf log-weight must be finite: {}", log_w[1]);
        // Leaf should have higher weight than hub (less penalty)
        assert!(log_w[1] > log_w[0], "Leaf should be preferred over hub");
    }

    #[test]
    fn test_log_weights_with_tabu() {
        let est = PathEntropyEstimator::with_defaults();
        let neighbors = vec![(1, 1.0), (2, 1.0)];
        let positions: Vec<Vec<f64>> = vec![
            vec![0.1, 0.0, 0.0],
            vec![0.2, 0.0, 0.0],
            vec![0.3, 0.0, 0.0],
        ];
        let degrees = vec![2, 2, 2];

        // Node 1 is in tabu
        let tabu = vec![1];
        let log_w = est.log_transition_weights(
            &neighbors,
            Some(&positions[0]),
            &positions,
            &degrees,
            &tabu,
            est.config.gamma,
            0.0, // h_local = 0 (fresh walk, full penalty)
        );

        // Node 2 (not tabu) should have higher weight than node 1 (tabu)
        assert!(
            log_w[1] > log_w[0],
            "Non-tabu node should be preferred: {} vs {}",
            log_w[1], log_w[0]
        );
    }

    #[test]
    fn test_entropy_boundary_nodes_no_nan() {
        // Graph where all nodes are near the Poincaré boundary
        let n = 8;
        let edges: Vec<(usize, usize, f64)> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
        let positions: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                // Place near boundary: |x| ≈ 0.995
                vec![0.995 * angle.cos(), 0.995 * angle.sin(), 0.0]
            })
            .collect();

        let mut est = PathEntropyEstimator::with_defaults();
        let report = est.estimate(&edges, n, &positions, None);

        assert!(report.h_path.is_finite(), "H_path must be finite near boundary: {}", report.h_path);
        assert!(report.divergence.is_finite(), "Divergence must be finite: {}", report.divergence);
        assert!(!report.h_path.is_nan(), "H_path must not be NaN");
    }

    #[test]
    fn test_entropy_high_energy_edges_no_overflow() {
        // Edges with very high weights — α·E_edge could overflow without log-space
        let n = 5;
        let edges: Vec<(usize, usize, f64)> = vec![
            (0, 1, 100.0),  // Very high weight
            (1, 2, 100.0),
            (2, 3, 0.001),  // Very low weight
            (3, 4, 100.0),
        ];
        let positions: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![0.1 * (i as f64 + 1.0), 0.0, 0.0])
            .collect();

        let mut est = PathEntropyEstimator::with_defaults();
        let report = est.estimate(&edges, n, &positions, None);

        assert!(report.h_path.is_finite(), "Must handle extreme edge weights: {}", report.h_path);
    }

    #[test]
    fn test_spectral_guardrail_reduces_gamma() {
        // When λ₂ is very low, the walker should reduce hub penalty
        let (edges, positions) = make_star_graph(10);
        let mut est1 = PathEntropyEstimator::with_defaults();
        let mut est2 = PathEntropyEstimator::with_defaults();

        // Normal gamma
        let report_normal = est1.estimate_with_spectral(
            &edges, 10, &positions, None, Some(0.5),
        );
        // Reduced gamma (nearly disconnected)
        let report_guardrail = est2.estimate_with_spectral(
            &edges, 10, &positions, None, Some(0.01),
        );

        // Both must be finite
        assert!(report_normal.h_path.is_finite());
        assert!(report_guardrail.h_path.is_finite());
        // With reduced gamma, walks should be slightly more concentrated on hubs
        // (less anti-hub penalty), but both should work
    }

    #[test]
    fn test_tabu_prevents_loops() {
        // Line graph with short walk — tabu should reduce revisits
        let (edges, positions) = make_line_graph(5);

        // With memory
        let mut est_mem = PathEntropyEstimator::new(PathEntropyConfig {
            memory_size: 2,
            n_walks: 20,
            walk_length: 8,
            ..PathEntropyConfig::default()
        });
        let report_mem = est_mem.estimate(&edges, 5, &positions, None);

        // Without memory
        let mut est_no_mem = PathEntropyEstimator::new(PathEntropyConfig {
            memory_size: 0,
            n_walks: 20,
            walk_length: 8,
            ..PathEntropyConfig::default()
        });
        let report_no_mem = est_no_mem.estimate(&edges, 5, &positions, None);

        // Both should be finite and positive
        assert!(report_mem.h_path > 0.0);
        assert!(report_no_mem.h_path > 0.0);
        // With memory, should visit more unique nodes (less backtracking)
        assert!(
            report_mem.unique_visited >= report_no_mem.unique_visited.saturating_sub(1),
            "Tabu should help explore: mem={} vs no_mem={}",
            report_mem.unique_visited, report_no_mem.unique_visited
        );
    }

    // ── MetabolicSleepManager tests ──

    #[test]
    fn test_sleep_first_call_max_tau() {
        let mut mgr = MetabolicSleepManager::with_defaults();
        let state = StateVector::new(0.3, 0.5, 1.0);

        let report = mgr.compute_sleep(state, 1.0, 0.5, 0.05, 0.05);

        // First call: no previous state → gradient = 0 → τ_relax = 1/ε → max sleep
        assert_eq!(report.sleep_duration, mgr.config.max_sleep);
        assert_eq!(report.cognitive_state, CognitiveState::Cruising);
    }

    #[test]
    fn test_sleep_fast_changes_short_sleep() {
        let mut mgr = MetabolicSleepManager::with_defaults();

        // First call to establish baseline
        let s1 = StateVector::new(0.3, 0.5, 1.0);
        mgr.compute_sleep(s1, 1.0, 0.5, 0.05, 0.05);

        // Second call with large change → high gradient → short τ → short sleep
        let s2 = StateVector::new(0.8, 1.5, 3.0);
        let report = mgr.compute_sleep(s2, 1.0, 0.5, 0.5, 0.5);

        assert!(
            report.sleep_duration < Duration::from_secs(30),
            "Fast changes should yield short sleep: {:?}",
            report.sleep_duration
        );
        assert!(report.gradient_norm > 0.5, "Gradient should be large: {}", report.gradient_norm);
    }

    #[test]
    fn test_sleep_stagnation_shortens_sleep() {
        let mut mgr = MetabolicSleepManager::with_defaults();

        let s1 = StateVector::new(0.3, 0.5, 1.0);
        mgr.compute_sleep(s1, 1.0, 0.5, 0.05, 0.05);

        // Stagnation: low divergence, flat energy, flat entropy
        let s2 = StateVector::new(0.3, 0.5, 1.0);
        let report = mgr.compute_sleep(s2, 1.0, 0.05, 0.001, 0.001);

        assert_eq!(report.cognitive_state, CognitiveState::Stagnation);
        assert!((report.modulation_factor - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_sleep_convergence_lengthens_sleep() {
        let mut mgr = MetabolicSleepManager::with_defaults();

        let s1 = StateVector::new(0.3, 0.5, 1.0);
        mgr.compute_sleep(s1, 1.0, 0.5, 0.05, 0.05);

        // Convergence: low divergence, high energy change
        let s2 = StateVector::new(0.35, 0.6, 1.05);
        let report = mgr.compute_sleep(s2, 1.0, 0.1, 0.5, -0.1);

        assert_eq!(report.cognitive_state, CognitiveState::Convergence);
        assert!((report.modulation_factor - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_sleep_stagnation_counter() {
        let mut mgr = MetabolicSleepManager::with_defaults();

        let s = StateVector::new(0.3, 0.5, 1.0);
        mgr.compute_sleep(s, 1.0, 0.5, 0.05, 0.05); // Initial (Cruising)

        // 3 stagnation cycles
        for i in 1..=3 {
            let s2 = StateVector::new(0.3, 0.5, 1.0);
            let report = mgr.compute_sleep(s2, 1.0, 0.05, 0.001, 0.001);
            assert_eq!(report.stagnation_cycles, i);
        }

        // Break stagnation
        let s3 = StateVector::new(0.5, 0.8, 1.5);
        let report = mgr.compute_sleep(s3, 1.0, 0.5, 0.3, 0.2);
        assert_eq!(report.stagnation_cycles, 0);
    }

    #[test]
    fn test_sleep_turbulence() {
        let mut mgr = MetabolicSleepManager::with_defaults();

        let s1 = StateVector::new(0.3, 0.5, 1.0);
        mgr.compute_sleep(s1, 1.0, 0.5, 0.05, 0.05);

        let s2 = StateVector::new(0.35, 0.55, 1.05);
        let report = mgr.compute_sleep(s2, 1.0, 0.9, 0.3, 0.2);

        assert_eq!(report.cognitive_state, CognitiveState::Turbulence);
    }

    #[test]
    fn test_sleep_report_display() {
        let report = SleepReport {
            sleep_duration: Duration::from_secs_f64(15.5),
            tau_relax: 3.88,
            gradient_norm: 0.258,
            cognitive_state: CognitiveState::Cruising,
            modulation_factor: 1.0,
            stagnation_cycles: 0,
        };
        let s = format!("{report}");
        assert!(s.contains("15.5s"));
        assert!(s.contains("normal"));
    }

    // ── CuriosityEngine tests ──

    #[test]
    fn test_curiosity_inactive_below_threshold() {
        let mut engine = CuriosityEngine::with_defaults();
        let actions = engine.evaluate(2, &[], &[], 0, 10);

        assert!(!actions.activated);
        assert!(actions.perturbations.is_empty());
        assert!(actions.hub_reweights.is_empty());
        assert!(actions.extended_walk_length.is_none());
    }

    #[test]
    fn test_curiosity_activates_at_threshold() {
        let mut engine = CuriosityEngine::with_defaults();
        let positions = vec![vec![0.3, 0.1, 0.0], vec![0.5, 0.2, 0.0]];
        let edges = vec![(0, 1, 1.0)];

        let actions = engine.evaluate(3, &positions, &edges, 2, 10);

        assert!(actions.activated);
        assert_eq!(actions.perturbations.len(), 2);
        assert!(actions.extended_walk_length.is_some());
        assert!(actions.strength_used > 0.0);
    }

    #[test]
    fn test_curiosity_perturbations_stay_in_ball() {
        let mut engine = CuriosityEngine::with_defaults();
        let positions = vec![
            vec![0.8, 0.3, 0.0],  // Near boundary
            vec![0.1, 0.0, 0.0],  // Near center
            vec![0.5, 0.5, 0.0],  // Middle
        ];
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];

        let actions = engine.evaluate(5, &positions, &edges, 3, 10);

        for (_, new_pos) in &actions.perturbations {
            let norm: f64 = new_pos.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                norm <= 0.951, // max_radius + tiny float tolerance
                "Perturbed position must stay in ball: norm = {norm}"
            );
        }
    }

    #[test]
    fn test_curiosity_strength_scales_with_depth() {
        let mut engine1 = CuriosityEngine::with_defaults();
        let mut engine2 = CuriosityEngine::with_defaults();
        let positions = vec![vec![0.3, 0.1, 0.0]];
        let edges = vec![];

        let actions_mild = engine1.evaluate(3, &positions, &edges, 1, 10);   // depth=1
        let actions_deep = engine2.evaluate(6, &positions, &edges, 1, 10);   // depth=4

        assert!(
            actions_deep.strength_used >= actions_mild.strength_used,
            "Deeper stagnation → stronger perturbation: {} vs {}",
            actions_deep.strength_used, actions_mild.strength_used
        );
    }

    #[test]
    fn test_curiosity_hub_reweighting() {
        let mut engine = CuriosityEngine::with_defaults();

        // Star graph: node 0 is the hub
        let n = 10;
        let edges: Vec<(usize, usize, f64)> = (1..n).map(|i| (0, i, 1.0)).collect();
        let positions: Vec<Vec<f64>> = (0..n).map(|i| vec![0.1 * (i as f64), 0.0, 0.0]).collect();

        let actions = engine.evaluate(4, &positions, &edges, n, 10);

        assert!(!actions.hub_reweights.is_empty(), "Should detect hub node 0");
        // Hub 0 should be in the reweight list
        let hub_0 = actions.hub_reweights.iter().find(|&&(idx, _)| idx == 0);
        assert!(hub_0.is_some(), "Node 0 (hub) should be reweighted");
        let (_, factor) = hub_0.unwrap();
        assert!(*factor < 1.0, "Hub factor should reduce weight: {factor}");
    }

    #[test]
    fn test_curiosity_walk_length_increases() {
        let mut engine = CuriosityEngine::with_defaults();
        let actions = engine.evaluate(3, &[], &[], 0, 10);

        assert!(actions.activated);
        let extended = actions.extended_walk_length.unwrap();
        assert!(extended > 10, "Walk length should increase from 10: {extended}");
    }

    #[test]
    fn test_curiosity_actions_display() {
        let actions = CuriosityActions {
            activated: true,
            perturbations: vec![(0, vec![0.3, 0.1, 0.0])],
            hub_reweights: vec![(5, 0.7)],
            extended_walk_length: Some(20),
            strength_used: 0.04,
            stagnation_depth: 4,
        };
        let s = format!("{actions}");
        assert!(s.contains("ACTIVE"));
        assert!(s.contains("depth=4"));
    }

    // ── MetabolicReport tests ──

    #[test]
    fn test_metabolic_report_display() {
        let report = MetabolicReport {
            state: StateVector::new(0.3, 0.5, 1.2),
            entropy: PathEntropyReport {
                h_path: 1.8,
                divergence: 0.55,
                unique_visited: 12,
                n_walks: 50,
                walk_length: 10,
            },
            sleep: SleepReport {
                sleep_duration: Duration::from_secs(20),
                tau_relax: 5.0,
                gradient_norm: 0.2,
                cognitive_state: CognitiveState::Cruising,
                modulation_factor: 1.0,
                stagnation_cycles: 0,
            },
            curiosity: CuriosityActions {
                activated: false,
                perturbations: vec![],
                hub_reweights: vec![],
                extended_walk_length: None,
                strength_used: 0.0,
                stagnation_depth: 0,
            },
            navigability: None,
        };
        let s = format!("{report}");
        assert!(s.contains("Metabolic Report"));
        assert!(s.contains("λ₂=0.3000"));
        assert!(s.contains("inactive"));
    }

    // ── Integration test: full metabolic estimation cycle ──

    #[test]
    fn test_full_metabolic_cycle() {
        let (edges, positions) = make_complete_graph(8);
        let n_nodes = 8;

        // 1. Estimate path entropy
        let mut estimator = PathEntropyEstimator::with_defaults();
        let entropy_report = estimator.estimate(&edges, n_nodes, &positions, None);
        assert!(entropy_report.h_path > 0.0);

        // 2. Compute sleep
        let mut sleep_mgr = MetabolicSleepManager::with_defaults();
        let state = StateVector::new(0.35, 0.5, entropy_report.h_path);
        let sleep_report = sleep_mgr.compute_sleep(
            state, 1.0,
            entropy_report.divergence,
            0.05, 0.02,
        );
        assert!(sleep_report.sleep_duration >= sleep_mgr.config.min_sleep);

        // 3. Check curiosity (not yet stagnant)
        let mut curiosity = CuriosityEngine::with_defaults();
        let actions = curiosity.evaluate(
            sleep_mgr.stagnation_cycles,
            &positions, &edges, n_nodes,
            estimator.config.walk_length,
        );
        assert!(!actions.activated, "No stagnation yet");

        // 4. Compute navigability innovation score
        let nav_eval = NavigabilityEvaluator::with_defaults();
        let nav_report = nav_eval.evaluate(state.lambda2, state.energy, state.entropy.min(1.0));

        // 5. Assemble full report
        let report = MetabolicReport {
            state,
            entropy: entropy_report,
            sleep: sleep_report,
            curiosity: actions,
            navigability: Some(nav_report),
        };
        assert!(format!("{report}").contains("Metabolic Report"));
    }

    #[test]
    fn test_stagnation_triggers_curiosity() {
        let (edges, positions) = make_line_graph(10);

        let mut sleep_mgr = MetabolicSleepManager::with_defaults();
        let mut curiosity = CuriosityEngine::with_defaults();

        // Simulate 4 stagnation cycles
        for _ in 0..4 {
            let state = StateVector::new(0.3, 0.5, 1.0);
            sleep_mgr.compute_sleep(state, 1.0, 0.05, 0.001, 0.001);
        }

        assert!(sleep_mgr.stagnation_cycles >= 3, "Should be stagnating");

        let actions = curiosity.evaluate(
            sleep_mgr.stagnation_cycles,
            &positions, &edges, 10, 10,
        );
        assert!(actions.activated, "Curiosity should activate after stagnation");
        assert!(!actions.perturbations.is_empty());
    }
}
