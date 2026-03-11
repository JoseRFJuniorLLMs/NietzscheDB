//! # SemanticQudit — N-Dimensional Cognitive Superposition Unit
//!
//! The atomic building block of the Orch-OR emulation layer.
//! Named in homage to **Stuart Hameroff**'s tubulin model, generalized
//! from a binary qubit to an N-dimensional qudit capable of sustaining
//! arbitrary numbers of concurrent hypotheses.
//!
//! ## Naming Convention
//!
//! Method names in this module pay tribute to the original theorists:
//!
//! | Method | Homage | Concept |
//! |---|---|---|
//! | `penrose_gravity` | **Roger Penrose** | Semantic gravity as Bayesian evidence accumulation |
//! | `penrose_reduction` | **Roger Penrose** | Objective Reduction — probabilistic wave function collapse |
//! | `hameroff_resuperpose` | **Stuart Hameroff** | Microtubule re-coherence after collapse |
//! | `shannon_entropy` | **Claude Shannon** | Normalized information entropy as decision readiness metric |
//!
//! ## Mathematical Foundation
//!
//! The state vector is an N-dimensional categorical distribution:
//!
//! ```text
//! |ψ⟩ = Σᵢ cᵢ |i⟩   where Σ |cᵢ|² = 1
//! ```
//!
//! Evidence accumulation follows Bayes' theorem:
//!
//! ```text
//! P(H|E) ∝ P(E|H) · P(H)
//! ```
//!
//! Collapse uses weighted categorical sampling (Born rule analogue).
//!
//! ## References
//!
//! 1. Penrose, R. (1994). *Shadows of the Mind*. Oxford University Press.
//! 2. Hameroff, S. & Penrose, R. (2014). Consciousness in the universe:
//!    A review of the 'Orch OR' theory. *Physics of Life Reviews*, 11(1), 39-78.
//! 3. Shannon, C.E. (1948). A Mathematical Theory of Communication.
//!    *Bell System Technical Journal*, 27(3), 379-423.

use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

/// Represents an N-dimensional cognitive superposition unit.
///
/// Inspired by **Hameroff**'s tubulin model — each `SemanticQudit` maintains
/// a probability distribution over N hypotheses that coexist until
/// **Penrose**'s Objective Reduction collapses them to a single observation.
///
/// # Invariants
///
/// 1. `Σ amplitudes_squared = 1.0` (normalization) — always holds after any operation
/// 2. All values in `amplitudes_squared` are `≥ 0.0`
/// 3. Once collapsed (`is_collapsed = true`), `penrose_gravity()` has no effect
/// 4. The RNG is never instantiated internally — always injected by the caller
///
/// # Example
///
/// ```
/// use nietzsche_agency::orch_or::SemanticQudit;
/// use rand::thread_rng;
///
/// let mut rng = thread_rng();
///
/// // 3 competing hypotheses in uniform superposition
/// let mut qudit = SemanticQudit::new(3);
/// assert!((qudit.shannon_entropy() - 1.0).abs() < 1e-6); // max uncertainty
///
/// // Bayesian evidence favoring hypothesis 0
/// qudit.penrose_gravity(&[2.0, 1.0, 0.5]);
/// assert!(qudit.probabilities()[0] > qudit.probabilities()[2]);
///
/// // Collapse
/// let winner = qudit.penrose_reduction(&mut rng);
/// assert!(winner < 3);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticQudit {
    /// Probability distribution over N hypotheses.
    /// Each value represents |cᵢ|² (Born rule probability).
    amplitudes_squared: Vec<f64>,

    /// Whether Objective Reduction has occurred.
    is_collapsed: bool,

    /// The observed state after collapse (if any).
    observed_state: Option<usize>,
}

impl SemanticQudit {
    /// Initialize in uniform superposition over `n_states` hypotheses.
    ///
    /// All hypotheses are equally probable: P(i) = 1/N.
    /// This represents maximum epistemic uncertainty (Shannon entropy = 1.0).
    ///
    /// # Panics
    ///
    /// Panics if `n_states == 0` — a qudit must have at least one state.
    pub fn new(n_states: usize) -> Self {
        assert!(n_states > 0, "SemanticQudit requires at least 1 state");
        let p = 1.0 / n_states as f64;
        SemanticQudit {
            amplitudes_squared: vec![p; n_states],
            is_collapsed: false,
            observed_state: None,
        }
    }

    /// Number of hypotheses this qudit can represent.
    pub fn n_states(&self) -> usize {
        self.amplitudes_squared.len()
    }

    /// Read-only access to the probability distribution.
    pub fn probabilities(&self) -> &[f64] {
        &self.amplitudes_squared
    }

    /// The collapsed state, if Objective Reduction has occurred.
    pub fn observed(&self) -> Option<usize> {
        self.observed_state
    }

    /// Whether the qudit has already undergone Objective Reduction.
    pub fn is_collapsed(&self) -> bool {
        self.is_collapsed
    }

    // ──────────────────────────────────────────────────────────────────
    // CLAUDE SHANNON — Information Entropy
    // ──────────────────────────────────────────────────────────────────

    /// Normalized Shannon entropy of the probability distribution.
    ///
    /// Named after **Claude Shannon** (1916-2001), father of information theory.
    ///
    /// ```text
    /// H_norm = -Σ pᵢ·ln(pᵢ) / ln(N)
    /// ```
    ///
    /// Returns a value in `[0.0, 1.0]`:
    /// - `0.0` = fully decided (one hypothesis has P=1)
    /// - `1.0` = maximum uncertainty (uniform distribution)
    ///
    /// This metric tells the system **when** to collapse: when entropy
    /// drops below a threshold, the qudit has accumulated enough evidence
    /// to make a principled decision.
    ///
    /// *Ref: Shannon, C.E. (1948). A Mathematical Theory of Communication.*
    pub fn shannon_entropy(&self) -> f64 {
        let n = self.amplitudes_squared.len() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        let max_entropy = n.ln();
        let h: f64 = self
            .amplitudes_squared
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        h / max_entropy
    }

    // ──────────────────────────────────────────────────────────────────
    // ROGER PENROSE — Semantic Gravity & Objective Reduction
    // ──────────────────────────────────────────────────────────────────

    /// Apply semantic gravity — Bayesian evidence accumulation.
    ///
    /// Named after **Roger Penrose** (b. 1931, Nobel Prize Physics 2020).
    /// In the Orch-OR theory, gravitational self-energy determines when
    /// a quantum superposition becomes unstable and must collapse.
    /// Here, "gravity" is the contextual evidence from the NietzscheDB
    /// semantic graph that pulls the distribution toward certain hypotheses.
    ///
    /// Implements Bayes' theorem:
    /// ```text
    /// P(Hᵢ|E) = P(E|Hᵢ) · P(Hᵢ) / Σⱼ P(E|Hⱼ) · P(Hⱼ)
    /// ```
    ///
    /// Where `evidence[i]` = P(E|Hᵢ) (likelihood) and
    /// `amplitudes_squared[i]` = P(Hᵢ) (prior).
    ///
    /// # Safety: Zero-Evidence Fallback
    ///
    /// If all evidence weights produce a zero sum (total annihilation),
    /// the distribution resets to uniform instead of leaving all-zero
    /// weights — preventing a fatal panic in `penrose_reduction()`.
    ///
    /// # Returns
    ///
    /// - `true` if evidence was successfully applied
    /// - `false` if rejected (qudit already collapsed, dimension mismatch,
    ///   or evidence was annihilating and triggered uniform reset)
    ///
    /// *Ref: Penrose, R. (1994). Shadows of the Mind, Ch. 6-7.*
    pub fn penrose_gravity(&mut self, evidence: &[f64]) -> bool {
        if self.is_collapsed || evidence.len() != self.amplitudes_squared.len() {
            return false;
        }

        // Bayesian update: posterior ∝ likelihood × prior
        for (p, &e) in self.amplitudes_squared.iter_mut().zip(evidence) {
            *p *= e.max(0.0); // clamp negative evidence to zero
        }

        let sum: f64 = self.amplitudes_squared.iter().sum();
        if sum > 0.0 {
            // Normalize: Σ P(i) = 1
            for p in &mut self.amplitudes_squared {
                *p /= sum;
            }
            true
        } else {
            // Zero-evidence fallback: reset to uniform (homeostasis)
            // This prevents WeightedIndex panic in penrose_reduction()
            let uniform = 1.0 / self.amplitudes_squared.len() as f64;
            for p in &mut self.amplitudes_squared {
                *p = uniform;
            }
            false
        }
    }

    /// Objective Reduction — probabilistic collapse of the superposition.
    ///
    /// Named after **Roger Penrose**'s central mechanism in Orch-OR:
    /// when the gravitational self-energy of a superposition exceeds a
    /// threshold (here: Shannon entropy < collapse_threshold), the system
    /// must "choose" — collapsing to a single observed state.
    ///
    /// Uses weighted categorical sampling (the **Born rule** analogue):
    /// hypothesis i is selected with probability |cᵢ|².
    ///
    /// # Idempotency
    ///
    /// If already collapsed, returns the previously observed state
    /// without re-sampling. This is physically correct: observation
    /// is irreversible in the Orch-OR framework.
    ///
    /// # Arguments
    ///
    /// * `rng` — External random number generator (injected for efficiency
    ///   and deterministic testing). Never instantiated internally.
    ///
    /// *Ref: Penrose, R. & Hameroff, S. (2014). Physics of Life Reviews, 11(1), 39-78.*
    pub fn penrose_reduction(&mut self, rng: &mut impl Rng) -> usize {
        if let Some(state) = self.observed_state {
            return state;
        }

        let dist = WeightedIndex::new(&self.amplitudes_squared)
            .expect("amplitudes invalid after normalization — this is a bug");

        let result = dist.sample(rng);
        self.is_collapsed = true;
        self.observed_state = Some(result);
        result
    }

    // ──────────────────────────────────────────────────────────────────
    // STUART HAMEROFF — Microtubule Re-Superposition
    // ──────────────────────────────────────────────────────────────────

    /// Re-superpose the qudit after collapse, preserving learned bias.
    ///
    /// Named after **Stuart Hameroff** (b. 1947), who proposed that
    /// microtubules undergo cycles of quantum coherence → collapse →
    /// re-coherence, with each cycle informed by the previous result.
    ///
    /// The previously observed state receives a `prior_boost` of extra
    /// probability, creating a **cognitive prior** — the system "remembers"
    /// what worked before without being locked into it.
    ///
    /// ```text
    /// P(i) = 1/N + boost·δ(i, prev_winner)     (then normalized)
    /// ```
    ///
    /// This is analogous to Hameroff's proposal that microtubule
    /// re-coherence is not a blank reset but carries forward structural
    /// information from the previous conscious moment.
    ///
    /// # Arguments
    ///
    /// * `prior_boost` — Extra probability mass for the previously
    ///   observed state. Clamped to `≥ 0.0`. Typical values: 0.1-0.3.
    ///
    /// *Ref: Hameroff, S. (2007). The brain is both neurocomputer and quantum
    /// computer. Cognitive Science, 31(6), 1035-1045.*
    pub fn hameroff_resuperpose(&mut self, prior_boost: f64) {
        let n = self.amplitudes_squared.len();
        let uniform = 1.0 / n as f64;

        for p in &mut self.amplitudes_squared {
            *p = uniform;
        }

        // If there was a previous observation, it becomes a biased prior
        if let Some(prev) = self.observed_state {
            self.amplitudes_squared[prev] += prior_boost.max(0.0);
            let sum: f64 = self.amplitudes_squared.iter().sum();
            for p in &mut self.amplitudes_squared {
                *p /= sum;
            }
        }

        self.is_collapsed = false;
        self.observed_state = None;
    }
}

// ══════════════════════════════════════════════════════════════════════
// Backward-compatible aliases
// ══════════════════════════════════════════════════════════════════════

impl SemanticQudit {
    /// Alias for [`penrose_gravity`]. Provided for backward compatibility
    /// with code referencing the original design documents.
    pub fn apply_semantic_gravity(&mut self, evidence: &[f64]) -> bool {
        self.penrose_gravity(evidence)
    }

    /// Alias for [`penrose_reduction`]. Provided for backward compatibility.
    pub fn objective_reduction(&mut self, rng: &mut impl Rng) -> usize {
        self.penrose_reduction(rng)
    }

    /// Alias for [`hameroff_resuperpose`]. Provided for backward compatibility.
    pub fn resuperpose(&mut self, prior_boost: f64) {
        self.hameroff_resuperpose(prior_boost)
    }

    /// Alias for [`shannon_entropy`]. Provided for backward compatibility.
    pub fn entropy(&self) -> f64 {
        self.shannon_entropy()
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn test_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn new_creates_uniform_distribution() {
        let q = SemanticQudit::new(4);
        for &p in q.probabilities() {
            assert!((p - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn shannon_entropy_uniform_is_one() {
        let q = SemanticQudit::new(5);
        assert!((q.shannon_entropy() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn shannon_entropy_collapsed_is_zero() {
        let mut q = SemanticQudit::new(3);
        // Force all probability to state 0
        q.penrose_gravity(&[1.0, 0.0, 0.0]);
        assert!(q.shannon_entropy() < 1e-6);
    }

    #[test]
    fn penrose_gravity_bayesian_update() {
        let mut q = SemanticQudit::new(3);
        // Evidence strongly favors hypothesis 1
        q.penrose_gravity(&[0.1, 10.0, 0.1]);
        assert!(q.probabilities()[1] > 0.9);
        // Sum still 1.0
        let sum: f64 = q.probabilities().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn penrose_gravity_accumulates() {
        let mut q = SemanticQudit::new(3);
        // Two rounds of evidence — both should accumulate
        q.penrose_gravity(&[2.0, 1.0, 1.0]);
        let after_first = q.probabilities()[0];

        q.penrose_gravity(&[2.0, 1.0, 1.0]);
        let after_second = q.probabilities()[0];

        assert!(after_second > after_first, "evidence should accumulate");
    }

    #[test]
    fn penrose_gravity_zero_evidence_resets_to_uniform() {
        let mut q = SemanticQudit::new(3);
        q.penrose_gravity(&[2.0, 1.0, 1.0]); // bias first
        let result = q.penrose_gravity(&[0.0, 0.0, 0.0]); // annihilate
        assert!(!result, "should return false on annihilation");

        // Should be uniform again
        for &p in q.probabilities() {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn penrose_gravity_rejected_after_collapse() {
        let mut q = SemanticQudit::new(2);
        let mut rng = test_rng();
        q.penrose_reduction(&mut rng);
        assert!(!q.penrose_gravity(&[1.0, 2.0]));
    }

    #[test]
    fn penrose_gravity_dimension_mismatch() {
        let mut q = SemanticQudit::new(3);
        assert!(!q.penrose_gravity(&[1.0, 2.0])); // wrong size
    }

    #[test]
    fn penrose_reduction_returns_valid_state() {
        let mut q = SemanticQudit::new(5);
        let mut rng = test_rng();
        let result = q.penrose_reduction(&mut rng);
        assert!(result < 5);
        assert!(q.is_collapsed());
        assert_eq!(q.observed(), Some(result));
    }

    #[test]
    fn penrose_reduction_idempotent() {
        let mut q = SemanticQudit::new(3);
        let mut rng = test_rng();
        let first = q.penrose_reduction(&mut rng);
        let second = q.penrose_reduction(&mut rng);
        assert_eq!(first, second);
    }

    #[test]
    fn penrose_reduction_respects_bias() {
        // With overwhelming evidence for state 0, collapse should
        // almost always select state 0
        let mut wins = 0;
        for seed in 0..100 {
            let mut q = SemanticQudit::new(3);
            q.penrose_gravity(&[1000.0, 1.0, 1.0]);
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            if q.penrose_reduction(&mut rng) == 0 {
                wins += 1;
            }
        }
        assert!(wins > 95, "state 0 should win >95% of the time, got {}", wins);
    }

    #[test]
    fn hameroff_resuperpose_resets_with_bias() {
        let mut q = SemanticQudit::new(3);
        let mut rng = test_rng();

        // Collapse to some state
        q.penrose_gravity(&[10.0, 1.0, 1.0]);
        let winner = q.penrose_reduction(&mut rng);

        // Re-superpose with boost
        q.hameroff_resuperpose(0.5);

        assert!(!q.is_collapsed());
        assert_eq!(q.observed(), None);

        // Winner should have higher probability than others
        let winner_prob = q.probabilities()[winner];
        for (i, &p) in q.probabilities().iter().enumerate() {
            if i != winner {
                assert!(winner_prob > p, "winner should have boosted probability");
            }
        }

        // Still normalized
        let sum: f64 = q.probabilities().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn hameroff_resuperpose_without_prior_is_uniform() {
        let mut q = SemanticQudit::new(4);
        // Never collapsed — resuperpose should just be uniform
        q.hameroff_resuperpose(0.3);
        for &p in q.probabilities() {
            assert!((p - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn full_lifecycle_penrose_hameroff() {
        let mut rng = test_rng();
        let mut q = SemanticQudit::new(4);

        // Phase 1: Accumulate evidence (Penrose gravity)
        q.penrose_gravity(&[1.0, 3.0, 1.0, 1.0]);
        q.penrose_gravity(&[1.0, 2.0, 1.0, 1.0]);
        assert!(q.shannon_entropy() < 1.0); // Entropy should decrease

        // Phase 2: Objective Reduction (Penrose collapse)
        let winner = q.penrose_reduction(&mut rng);
        assert!(q.is_collapsed());

        // Phase 3: Re-superposition (Hameroff re-coherence)
        q.hameroff_resuperpose(0.2);
        assert!(!q.is_collapsed());
        assert!(q.probabilities()[winner] > 0.25); // winner has prior boost

        // Phase 4: New evidence cycle
        assert!(q.penrose_gravity(&[1.0, 1.0, 5.0, 1.0]));
        let second_winner = q.penrose_reduction(&mut rng);
        assert!(q.is_collapsed());
        assert!(second_winner < 4);
    }

    #[test]
    fn backward_compatible_aliases() {
        let mut q = SemanticQudit::new(3);
        let mut rng = test_rng();

        // Old names should work
        assert!(q.apply_semantic_gravity(&[2.0, 1.0, 1.0]));
        assert!(q.entropy() < 1.0);
        let _ = q.objective_reduction(&mut rng);
        q.resuperpose(0.1);
        assert!(!q.is_collapsed());
    }

    #[test]
    #[should_panic(expected = "at least 1 state")]
    fn zero_states_panics() {
        SemanticQudit::new(0);
    }

    #[test]
    fn negative_evidence_clamped_to_zero() {
        let mut q = SemanticQudit::new(3);
        q.penrose_gravity(&[-5.0, 1.0, 1.0]);
        // State 0 should have zero probability
        assert!(q.probabilities()[0] < 1e-10);
        // Still normalized
        let sum: f64 = q.probabilities().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
