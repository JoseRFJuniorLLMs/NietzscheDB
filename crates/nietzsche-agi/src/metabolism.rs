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
//! on the hyperbolic graph. The walks are biased by the Poincaré metric:
//!
//! ```text
//! P(v → u) ∝ exp(-β · d_𝔻(v, u))
//! ```
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

use nietzsche_hyp_ops::{exp_map_zero, poincare_distance};

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
// §1 — Path Entropy Estimator
// ═══════════════════════════════════════════════

/// Configuration for path entropy estimation.
#[derive(Debug, Clone)]
pub struct PathEntropyConfig {
    /// Number of random walks to sample.
    /// More walks = more accurate H_path, but slower.
    /// Default: 50
    pub n_walks: usize,

    /// Length of each random walk (number of steps).
    /// Default: 10
    pub walk_length: usize,

    /// Temperature parameter β for the Boltzmann distribution.
    /// Higher β = more greedy (follow shortest geodesic).
    /// Lower β = more exploratory (uniform random).
    /// Default: 1.0
    pub beta: f64,

    /// Minimum number of unique nodes visited for a valid entropy estimate.
    /// Default: 3
    pub min_visited: usize,
}

impl Default for PathEntropyConfig {
    fn default() -> Self {
        Self {
            n_walks: 50,
            walk_length: 10,
            beta: 1.0,
            min_visited: 3,
        }
    }
}

/// Estimates functional navigability H_path via hyperbolic random walks.
///
/// The walk is biased by the Poincaré distance metric:
/// P(v → u) ∝ exp(-β · d_𝔻(v, u))
///
/// This measures *how diverse the routes are* through the knowledge graph.
/// If all walks converge to the same hub → H_path ≈ 0 (echo chamber).
/// If walks explore uniformly → H_path → log(N) (maximum diversity).
///
/// ## Divergence Metric
///
/// Also computes average pairwise divergence D between walk trajectories:
/// D = 1 - (1/m²) · Σ J(Tᵢ, Tⱼ)
///
/// Where J(Tᵢ, Tⱼ) is the Jaccard similarity between visited node sets.
#[derive(Debug, Clone)]
pub struct PathEntropyEstimator {
    config: PathEntropyConfig,
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

    /// Compute transition probabilities from node `current` to its neighbors.
    ///
    /// P(current → neighbor) ∝ exp(-β · d_𝔻(current, neighbor))
    ///
    /// Using Poincaré distance if positions are available, else fall back
    /// to edge weight (1/weight as "distance").
    fn transition_probabilities(
        &self,
        neighbors: &[(usize, f64)],
        current_pos: Option<&[f64]>,
        positions: &[Vec<f64>],
        beta: f64,
    ) -> Vec<f64> {
        if neighbors.is_empty() {
            return vec![];
        }

        let probs: Vec<f64> = neighbors
            .iter()
            .map(|&(nbr, edge_weight)| {
                // Use hyperbolic distance if positions available, else edge weight
                let dist = if let Some(cur_p) = current_pos {
                    if nbr < positions.len() && !positions[nbr].is_empty() {
                        poincare_distance(cur_p, &positions[nbr])
                    } else {
                        // Fallback: use 1/weight as proxy distance
                        if edge_weight > 0.0 { 1.0 / edge_weight } else { 1.0 }
                    }
                } else {
                    if edge_weight > 0.0 { 1.0 / edge_weight } else { 1.0 }
                };
                (-beta * dist).exp()
            })
            .collect();

        // Normalize
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 {
            vec![1.0 / neighbors.len() as f64; neighbors.len()]
        } else {
            probs.iter().map(|p| p / sum).collect()
        }
    }

    /// Sample a neighbor using cumulative distribution.
    fn sample_neighbor(&mut self, neighbors: &[(usize, f64)], probs: &[f64]) -> usize {
        let r = self.next_random();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return neighbors[i].0;
            }
        }
        // Fallback: last neighbor
        neighbors.last().map(|&(n, _)| n).unwrap_or(0)
    }

    /// Execute one random walk from `start_node`.
    fn random_walk(
        &mut self,
        start: usize,
        adj: &[Vec<(usize, f64)>],
        positions: &[Vec<f64>],
        walk_length: usize,
        beta: f64,
    ) -> Vec<usize> {
        let mut path = Vec::with_capacity(walk_length + 1);
        path.push(start);
        let mut current = start;

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
            let probs = self.transition_probabilities(neighbors, cur_pos, positions, beta);
            current = self.sample_neighbor(neighbors, &probs);
            path.push(current);
        }

        path
    }

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
        let walk_length = walk_length_override.unwrap_or(self.config.walk_length);
        let n_walks = self.config.n_walks;
        let beta = self.config.beta;

        // Visit frequency counter
        let mut visit_counts = vec![0u64; n_nodes];
        let mut total_visits = 0u64;

        // Collect visited sets for each walk (for divergence computation)
        let mut walk_sets: Vec<Vec<bool>> = Vec::with_capacity(n_walks);

        // Choose start nodes distributed across the graph
        for w in 0..n_walks {
            // Spread start nodes across the graph
            let start = if n_nodes > 0 {
                (w * 7 + 13) % n_nodes // Simple deterministic spread
            } else {
                0
            };

            // Skip isolated nodes
            if adj[start].is_empty() {
                continue;
            }

            let path = self.random_walk(start, &adj, positions, walk_length, beta);

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
                    // Jaccard similarity = |A ∩ B| / |A ∪ B|
                    let mut intersection = 0u64;
                    let mut union = 0u64;
                    for k in 0..n_nodes {
                        let a = walk_sets[i][k];
                        let b = walk_sets[j][k];
                        if a || b {
                            union += 1;
                        }
                        if a && b {
                            intersection += 1;
                        }
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
/// Includes entropy estimation, sleep computation, and curiosity actions.
/// This is the "consciousness" snapshot of the system.
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
}

impl std::fmt::Display for MetabolicReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══ Metabolic Report ══╗")?;
        writeln!(f, "║ State: {}", self.state)?;
        writeln!(f, "║ Entropy: {}", self.entropy)?;
        writeln!(f, "║ Sleep: {}", self.sleep)?;
        writeln!(f, "║ {}", self.curiosity)?;
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
        // Simple line: 0-1-2-...-n
        let edges: Vec<(usize, usize, f64)> = (0..n.saturating_sub(1))
            .map(|i| (i, i + 1, 1.0))
            .collect();
        // Place nodes along a line in Poincaré ball
        let positions: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let r = 0.1 + 0.7 * (i as f64) / (n.max(1) as f64);
                vec![r, 0.0, 0.0]
            })
            .collect();
        (edges, positions)
    }

    fn make_star_graph(n: usize) -> (Vec<(usize, usize, f64)>, Vec<Vec<f64>>) {
        // Hub at node 0, all others connect to 0
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
        // Star graph should have lower entropy than complete graph
        // (all walks funnel through hub)
        let (star_edges, star_pos) = make_star_graph(10);
        let (comp_edges, comp_pos) = make_complete_graph(10);

        let mut est1 = PathEntropyEstimator::with_defaults();
        let mut est2 = PathEntropyEstimator::with_defaults();

        let star_report = est1.estimate(&star_edges, 10, &star_pos, None);
        let comp_report = est2.estimate(&comp_edges, 10, &comp_pos, None);

        // Complete graph has more diverse routes → higher entropy
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
        // Should work even without Poincaré positions (falls back to edge weight)
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

        // 4. Assemble full report
        let report = MetabolicReport {
            state,
            entropy: entropy_report,
            sleep: sleep_report,
            curiosity: actions,
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
