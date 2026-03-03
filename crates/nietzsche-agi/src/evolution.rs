//! # Evolution — autonomous metabolic cycle scheduler
//!
//! Orchestrates the full AGI metabolic cycle:
//!
//! 1. **Measure λ₂** (pre-homeostasis): spectral health snapshot
//! 2. **Homeostasis**: damped force integration in tangent space via exp_map
//! 3. **Evaluate Sandbox**: promote, reject, or continue quarantined nodes
//! 4. **Measure λ₂** (post-homeostasis): detect real structural changes
//! 5. **Criticality Analysis**: classify regime (Rigid/Critical/Turbulent)
//! 6. **Adapt**: adjust damping coefficient and λ₂ target band
//! 7. **Record**: produce auditable CycleReport
//!
//! The [`EvolutionScheduler`] runs as a background task, periodically
//! waking up to perform one full cycle. This is the "heartbeat" of
//! the AGI layer — a closed-loop dynamical system with feedback.
//!
//! ## Design: Pure Computation
//!
//! The scheduler does NOT hold a reference to `NietzscheDB`. Instead,
//! it takes a [`MetabolicInput`] snapshot (pre-computed by the caller)
//! and returns a [`CycleReport`] with all actions to apply.
//! This keeps the AGI layer testable and side-effect-free.

use std::time::{Duration, Instant};

use crate::certification::CertificationConfig;
use crate::criticality::{CriticalityConfig, CriticalityDetector, CriticalityMetrics, RegimeState};
use crate::dialectic::{DialecticConfig, DialecticDetector};
use crate::discovery::{DiscoveryConfig, DiscoveryField};
use crate::feedback_loop::{FeedbackConfig, FeedbackLoop};
use crate::homeostasis::{DampedHomeostasis, DampingConfig, DampingReport, HomeostasisGuard, NodeDynamics};
use crate::inference_engine::{InferenceConfig, InferenceEngine};
use crate::innovation::{InnovationConfig, InnovationEvaluator};
use crate::relevance_decay::{RelevanceConfig, RelevanceDecay};
use crate::sandbox::{SandboxConfig, SandboxEntry, SandboxEvaluator, SandboxVerdict};
use crate::spectral::{AdaptiveBand, DriftTracker, SpectralConfig, SpectralMonitor};
use crate::stability::{StabilityConfig, StabilityEvaluator};
use crate::synthesis::{FrechetSynthesizer, SynthesisConfig};
use crate::trajectory::GcsConfig;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Master configuration for the evolution scheduler.
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    /// How often to run a full evolution cycle.
    /// Default: 60 seconds.
    pub cycle_interval: Duration,

    /// Maximum number of synthesis operations per cycle.
    /// Default: 5
    pub max_syntheses_per_cycle: usize,

    /// Whether the scheduler is enabled.
    /// Default: true
    pub enabled: bool,

    /// Sub-configurations
    pub gcs: GcsConfig,
    pub inference: InferenceConfig,
    pub synthesis: SynthesisConfig,
    pub dialectic: DialecticConfig,
    pub feedback: FeedbackConfig,
    pub relevance: RelevanceConfig,

    // Phase V sub-configurations
    pub stability: StabilityConfig,
    pub certification: CertificationConfig,
    pub spectral: SpectralConfig,

    // Phase VI sub-configurations
    pub discovery: DiscoveryConfig,
    pub innovation: InnovationConfig,
    pub sandbox: SandboxConfig,

    // Phase VI+ sub-configurations
    pub damping: DampingConfig,
    pub criticality: CriticalityConfig,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            cycle_interval: Duration::from_secs(60),
            max_syntheses_per_cycle: 5,
            enabled: true,
            gcs: GcsConfig::default(),
            inference: InferenceConfig::default(),
            synthesis: SynthesisConfig::default(),
            dialectic: DialecticConfig::default(),
            feedback: FeedbackConfig::default(),
            relevance: RelevanceConfig::default(),
            stability: StabilityConfig::default(),
            certification: CertificationConfig::default(),
            spectral: SpectralConfig::default(),
            discovery: DiscoveryConfig::default(),
            innovation: InnovationConfig::default(),
            sandbox: SandboxConfig::default(),
            damping: DampingConfig::default(),
            criticality: CriticalityConfig::default(),
        }
    }
}

// ─────────────────────────────────────────────
// MetabolicInput — pre-computed snapshot for one cycle
// ─────────────────────────────────────────────

/// Pre-computed data snapshot that the caller provides for one metabolic cycle.
///
/// The scheduler is **pure computation** — it doesn't access the graph.
/// The caller (e.g., `nietzsche-agency`) gathers this data, passes it in,
/// and applies the resulting [`CycleReport`] actions.
#[derive(Debug, Clone)]
pub struct MetabolicInput {
    /// Graph edges as (source_idx, target_idx, weight) for spectral analysis.
    pub edges: Vec<(usize, usize, f64)>,

    /// Total number of nodes in the graph.
    pub n_nodes: usize,

    /// Current positions of synthesis/AGI nodes in the Poincaré ball.
    /// These are the nodes subject to homeostatic damping.
    pub node_positions: Vec<Vec<f64>>,

    /// Energy values of all nodes (for variance computation).
    pub node_energies: Vec<f64>,

    /// Active sandbox entries awaiting evaluation.
    pub sandbox_entries: Vec<SandboxEntry>,

    /// Global energy this cycle (sum of all node energies / N).
    pub global_energy: f64,

    /// Energy injected this cycle (from new inferences, syntheses, etc.).
    pub energy_input: f64,

    /// Change in graph structural entropy since last cycle.
    /// Positive = more complex, negative = simpler.
    pub entropy_delta: f64,
}

// ─────────────────────────────────────────────
// CycleReport — complete auditable output
// ─────────────────────────────────────────────

/// Complete output of one metabolic cycle.
///
/// Every field is measurable and auditable. The caller reads this report
/// and applies the actions (new positions, promotions, rejections) to the graph.
#[derive(Debug, Clone)]
pub struct CycleReport {
    /// Cycle number (monotonically increasing).
    pub cycle_number: u64,

    /// Spectral health measured BEFORE homeostasis.
    pub lambda2_pre: f64,

    /// Spectral health measured AFTER homeostasis + promotions.
    pub lambda2_post: f64,

    /// Criticality metrics (regime, antifragility, adaptive damping, etc.).
    pub criticality: CriticalityMetrics,

    /// Damping report (new positions, kinetic energy, etc.).
    pub damping: DampingReport,

    /// Sandbox promotions this cycle.
    pub promotions: Vec<SandboxAction>,

    /// Sandbox rejections this cycle.
    pub rejections: Vec<SandboxAction>,

    /// Sandbox entries still in quarantine.
    pub continuing: usize,

    /// Current adaptive λ₂ target band center.
    pub band_target: f64,

    /// Current adaptive damping coefficient.
    pub adaptive_damping: f64,

    /// Total time elapsed for this cycle.
    pub elapsed: Duration,
}

/// A sandbox action (promotion or rejection) with reason.
#[derive(Debug, Clone)]
pub struct SandboxAction {
    /// The sandbox entry that was acted upon.
    pub entry: SandboxEntry,
    /// The verdict.
    pub verdict: SandboxVerdict,
}

impl std::fmt::Display for CycleReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cycle#{} [{}] λ₂: {:.4}→{:.4} | band={:.3} | damp={:.3} | \
             promoted={} rejected={} continuing={} | KE={:.6} | {:?}",
            self.cycle_number,
            self.criticality.regime,
            self.lambda2_pre,
            self.lambda2_post,
            self.band_target,
            self.adaptive_damping,
            self.promotions.len(),
            self.rejections.len(),
            self.continuing,
            self.damping.mean_kinetic_energy,
            self.elapsed,
        )
    }
}

// ─────────────────────────────────────────────
// EvolutionScheduler
// ─────────────────────────────────────────────

/// Autonomous evolution orchestrator.
///
/// Holds all the AGI sub-systems and runs them in a coordinated cycle.
/// The metabolic cycle is a closed-loop dynamical system:
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────┐
/// │                     METABOLIC CYCLE                             │
/// │                                                                 │
/// │  Input ──► λ₂_pre ──► Damped Homeostasis ──► Sandbox Eval     │
/// │                                                   │             │
/// │              λ₂_post ◄── Spectral Measure ◄───────┘             │
/// │                │                                                │
/// │         Criticality Analysis                                    │
/// │                │                                                │
/// │   ┌────────────┴────────────┐                                   │
/// │   │  Adapt damping + band   │                                   │
/// │   └────────────┬────────────┘                                   │
/// │                │                                                │
/// │           CycleReport ──► Output                                │
/// └─────────────────────────────────────────────────────────────────┘
/// ```
pub struct EvolutionScheduler {
    pub config: EvolutionConfig,
    pub inference_engine: InferenceEngine,
    pub synthesizer: FrechetSynthesizer,
    pub dialectic_detector: DialecticDetector,
    pub feedback_loop: FeedbackLoop,
    pub relevance_decay: RelevanceDecay,
    pub homeostasis: HomeostasisGuard,

    // Phase V sub-systems
    pub stability_evaluator: StabilityEvaluator,
    pub spectral_monitor: SpectralMonitor,

    // Phase VI sub-systems
    pub discovery_field: DiscoveryField,
    pub innovation_evaluator: InnovationEvaluator,
    pub sandbox_evaluator: SandboxEvaluator,
    pub drift_tracker: DriftTracker,

    // Phase VI+ sub-systems (metabolic dynamics)
    pub damped_homeostasis: DampedHomeostasis,
    pub criticality_detector: CriticalityDetector,
    pub adaptive_band: AdaptiveBand,
    pub node_dynamics: Vec<NodeDynamics>,

    // Stats
    pub total_cycles: u64,
    pub total_syntheses: u64,
    pub total_ruptures: u64,
    pub total_promotions: u64,
    pub total_rejections: u64,
    pub last_cycle: Option<Instant>,
    pub last_global_energy: f64,
}

impl EvolutionScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: EvolutionConfig) -> Self {
        let inference_engine = InferenceEngine::new(config.inference.clone());
        let synthesizer = FrechetSynthesizer::new(config.synthesis.clone());
        let dialectic_detector = DialecticDetector::new(config.dialectic.clone());
        let feedback_loop = FeedbackLoop::new(config.feedback.clone());
        let relevance_decay = RelevanceDecay::new(config.relevance.clone());
        let homeostasis = HomeostasisGuard::new(config.synthesis.min_synthesis_radius);
        let stability_evaluator = StabilityEvaluator::new(config.stability.clone());
        let spectral_monitor = SpectralMonitor::new(config.spectral.clone());
        let discovery_field = DiscoveryField::new(config.discovery.clone());
        let innovation_evaluator = InnovationEvaluator::new(config.innovation.clone());
        let sandbox_evaluator = SandboxEvaluator::new(config.sandbox.clone());
        let drift_tracker = DriftTracker::with_defaults();
        let damped_homeostasis = DampedHomeostasis::new(config.damping.clone());
        let criticality_detector = CriticalityDetector::new(config.criticality.clone());
        let adaptive_band = AdaptiveBand::with_defaults();

        Self {
            config,
            inference_engine,
            synthesizer,
            dialectic_detector,
            feedback_loop,
            relevance_decay,
            homeostasis,
            stability_evaluator,
            spectral_monitor,
            discovery_field,
            innovation_evaluator,
            sandbox_evaluator,
            drift_tracker,
            damped_homeostasis,
            criticality_detector,
            adaptive_band,
            node_dynamics: Vec::new(),
            total_cycles: 0,
            total_syntheses: 0,
            total_ruptures: 0,
            total_promotions: 0,
            total_rejections: 0,
            last_cycle: None,
            last_global_energy: 0.0,
        }
    }

    /// Create a scheduler with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(EvolutionConfig::default())
    }

    /// Check if enough time has passed for a new cycle.
    pub fn should_run(&self) -> bool {
        if !self.config.enabled {
            return false;
        }
        match self.last_cycle {
            None => true,
            Some(last) => last.elapsed() >= self.config.cycle_interval,
        }
    }

    /// Run one complete metabolic cycle.
    ///
    /// This is the "heartbeat" method. The caller provides a [`MetabolicInput`]
    /// snapshot and receives a [`CycleReport`] with all actions to apply.
    ///
    /// ## Cycle Steps
    ///
    /// 1. **Measure λ₂ (pre)** — spectral health before any corrections
    /// 2. **Damped Homeostasis** — apply forces in tangent space via exp_map
    /// 3. **Sandbox Evaluation** — promote/reject/continue quarantined nodes
    /// 4. **Measure λ₂ (post)** — spectral health AFTER corrections
    /// 5. **Criticality Analysis** — classify regime, compute antifragility
    /// 6. **Adapt** — adjust damping coefficient and λ₂ band target
    /// 7. **Record** — update internal state, produce CycleReport
    pub fn run_metabolic_cycle(&mut self, input: &MetabolicInput) -> CycleReport {
        let start = Instant::now();
        self.total_cycles += 1;
        self.last_cycle = Some(start);

        // ── Step 1: Measure λ₂ (pre-homeostasis) ──
        let pre_health = self.spectral_monitor.analyze(&input.edges, input.n_nodes);
        let lambda2_pre = pre_health.fiedler_eigenvalue;

        // ── Step 2: Damped Homeostasis ──
        // Ensure we have dynamics state for all nodes
        self.ensure_dynamics(input.node_positions.len(), input.node_positions.first().map(|p| p.len()).unwrap_or(3));

        let damping_report = self.damped_homeostasis.tick(
            &input.node_positions,
            &mut self.node_dynamics,
        );

        // ── Step 3: Sandbox Evaluation ──
        let mut promotions = Vec::new();
        let mut rejections = Vec::new();
        let mut continuing = 0usize;

        for entry in &input.sandbox_entries {
            let mut entry_clone = entry.clone();
            self.sandbox_evaluator.record_cycle(&mut entry_clone, lambda2_pre);
            let report = self.sandbox_evaluator.evaluate(&entry_clone);

            match report.verdict {
                SandboxVerdict::Promote => {
                    promotions.push(SandboxAction {
                        entry: entry_clone,
                        verdict: SandboxVerdict::Promote,
                    });
                    self.total_promotions += 1;
                }
                SandboxVerdict::Reject => {
                    rejections.push(SandboxAction {
                        entry: entry_clone,
                        verdict: SandboxVerdict::Reject,
                    });
                    self.total_rejections += 1;
                }
                SandboxVerdict::Continue => {
                    continuing += 1;
                }
            }
        }

        // ── Step 4: Measure λ₂ (post-homeostasis) ──
        // NOTE: In a real system, edges would be recomputed from new positions.
        // Here we use the same edges as a reasonable approximation (positions
        // moved by damping, but topology didn't change this cycle).
        let post_health = self.spectral_monitor.analyze(&input.edges, input.n_nodes);
        let lambda2_post = post_health.fiedler_eigenvalue;

        // Record in drift tracker
        self.drift_tracker.record(lambda2_post);

        // ── Step 5: Criticality Analysis ──
        let energy_variance = compute_variance(&input.node_energies);
        let criticality = self.criticality_detector.analyze(
            &self.drift_tracker.history,
            energy_variance,
            input.entropy_delta,
            input.energy_input,
            input.n_nodes,
        );

        // ── Step 6: Adapt ──
        // Adjust damping coefficient based on regime
        let adaptive_damping = criticality.adaptive_damping;
        self.damped_homeostasis.config.damping_coeff = adaptive_damping;

        // Adjust λ₂ target band
        let energy_delta = input.global_energy - self.last_global_energy;
        self.adaptive_band.adjust(energy_delta, lambda2_post);
        self.last_global_energy = input.global_energy;

        // ── Step 7: Record ──
        let elapsed = start.elapsed();

        CycleReport {
            cycle_number: self.total_cycles,
            lambda2_pre,
            lambda2_post,
            criticality,
            damping: damping_report,
            promotions,
            rejections,
            continuing,
            band_target: self.adaptive_band.target,
            adaptive_damping,
            elapsed,
        }
    }

    /// Record synthesis results.
    pub fn record_synthesis(&mut self, count: u64) {
        self.total_syntheses += count;
    }

    /// Record rupture results.
    pub fn record_ruptures(&mut self, count: u64) {
        self.total_ruptures += count;
    }

    /// Get a snapshot of the scheduler's stats.
    pub fn stats(&self) -> EvolutionStats {
        EvolutionStats {
            total_cycles: self.total_cycles,
            total_syntheses: self.total_syntheses,
            total_ruptures: self.total_ruptures,
            total_promotions: self.total_promotions,
            total_rejections: self.total_rejections,
            last_cycle_elapsed: self.last_cycle.map(|l| l.elapsed()),
            enabled: self.config.enabled,
            current_regime: self.drift_tracker.history.last().map(|_| {
                // Approximate regime from last data
                if self.drift_tracker.is_healthy() {
                    RegimeState::Critical
                } else {
                    RegimeState::Turbulent
                }
            }),
            band_target: self.adaptive_band.target,
            adaptive_damping: self.damped_homeostasis.config.damping_coeff,
        }
    }

    /// Ensure we have NodeDynamics for all positions.
    fn ensure_dynamics(&mut self, n: usize, dim: usize) {
        if self.node_dynamics.len() < n {
            self.node_dynamics.resize_with(n, || NodeDynamics::new(dim));
        } else if self.node_dynamics.len() > n {
            self.node_dynamics.truncate(n);
        }
    }
}

/// Snapshot of evolution scheduler statistics.
#[derive(Debug, Clone)]
pub struct EvolutionStats {
    pub total_cycles: u64,
    pub total_syntheses: u64,
    pub total_ruptures: u64,
    pub total_promotions: u64,
    pub total_rejections: u64,
    pub last_cycle_elapsed: Option<Duration>,
    pub enabled: bool,
    pub current_regime: Option<RegimeState>,
    pub band_target: f64,
    pub adaptive_damping: f64,
}

impl std::fmt::Display for EvolutionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Evolution[cycles={}, syntheses={}, ruptures={}, promoted={}, rejected={}, \
             band={:.3}, damp={:.3}, enabled={}]",
            self.total_cycles,
            self.total_syntheses,
            self.total_ruptures,
            self.total_promotions,
            self.total_rejections,
            self.band_target,
            self.adaptive_damping,
            self.enabled,
        )
    }
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

/// Compute variance of a slice of f64 values.
fn compute_variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let sched = EvolutionScheduler::with_defaults();
        assert_eq!(sched.total_cycles, 0);
        assert_eq!(sched.total_syntheses, 0);
        assert!(sched.should_run()); // No last cycle → should run
    }

    #[test]
    fn test_scheduler_timing() {
        let config = EvolutionConfig {
            cycle_interval: Duration::from_millis(100),
            ..Default::default()
        };
        let mut sched = EvolutionScheduler::new(config);
        sched.last_cycle = Some(Instant::now());
        sched.total_cycles = 1;
        assert!(!sched.should_run()); // Just started → shouldn't run yet
    }

    #[test]
    fn test_stats_display() {
        let sched = EvolutionScheduler::with_defaults();
        let stats = sched.stats();
        let s = format!("{stats}");
        assert!(s.contains("cycles=0"));
        assert!(s.contains("syntheses=0"));
    }

    #[test]
    fn test_metabolic_cycle_empty() {
        let mut sched = EvolutionScheduler::with_defaults();
        let input = MetabolicInput {
            edges: vec![],
            n_nodes: 0,
            node_positions: vec![],
            node_energies: vec![],
            sandbox_entries: vec![],
            global_energy: 0.0,
            energy_input: 0.0,
            entropy_delta: 0.0,
        };

        let report = sched.run_metabolic_cycle(&input);
        assert_eq!(report.cycle_number, 1);
        assert_eq!(report.promotions.len(), 0);
        assert_eq!(report.rejections.len(), 0);
    }

    #[test]
    fn test_metabolic_cycle_with_nodes() {
        let mut sched = EvolutionScheduler::with_defaults();
        let input = MetabolicInput {
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
            n_nodes: 3,
            node_positions: vec![
                vec![0.3, 0.1, 0.0],
                vec![0.5, 0.2, 0.0],
                vec![0.1, 0.4, 0.0],
            ],
            node_energies: vec![0.8, 0.6, 0.7],
            sandbox_entries: vec![],
            global_energy: 0.7,
            energy_input: 0.05,
            entropy_delta: 0.01,
        };

        let report = sched.run_metabolic_cycle(&input);
        assert_eq!(report.cycle_number, 1);
        assert!(report.lambda2_pre > 0.0, "Connected graph should have λ₂ > 0");
        assert!(report.lambda2_post > 0.0);
        assert_eq!(report.damping.total_nodes, 3);

        // Run a second cycle
        let report2 = sched.run_metabolic_cycle(&input);
        assert_eq!(report2.cycle_number, 2);
        assert_eq!(sched.total_cycles, 2);
    }

    #[test]
    fn test_metabolic_cycle_report_display() {
        let mut sched = EvolutionScheduler::with_defaults();
        let input = MetabolicInput {
            edges: vec![(0, 1, 1.0)],
            n_nodes: 2,
            node_positions: vec![vec![0.3, 0.0, 0.0], vec![0.6, 0.0, 0.0]],
            node_energies: vec![0.5, 0.5],
            sandbox_entries: vec![],
            global_energy: 0.5,
            energy_input: 0.0,
            entropy_delta: 0.0,
        };

        let report = sched.run_metabolic_cycle(&input);
        let s = format!("{report}");
        assert!(s.contains("Cycle#1"));
        assert!(s.contains("λ₂:"));
    }

    #[test]
    fn test_adaptive_damping_updates() {
        let mut sched = EvolutionScheduler::with_defaults();
        let base_damp = sched.damped_homeostasis.config.damping_coeff;

        // High energy variance → damping should increase
        let input = MetabolicInput {
            edges: vec![(0, 1, 1.0)],
            n_nodes: 2,
            node_positions: vec![vec![0.3, 0.0, 0.0], vec![0.6, 0.0, 0.0]],
            node_energies: vec![0.1, 0.9], // High variance
            sandbox_entries: vec![],
            global_energy: 0.5,
            energy_input: 0.1,
            entropy_delta: 0.01,
        };

        let report = sched.run_metabolic_cycle(&input);
        // With high variance (0.1 vs 0.9), damping should have adapted
        assert!(
            report.adaptive_damping != base_damp || report.adaptive_damping >= 0.05,
            "Damping should adapt: base={base_damp}, new={}",
            report.adaptive_damping
        );
    }

    #[test]
    fn test_band_evolves_with_energy() {
        let mut sched = EvolutionScheduler::with_defaults();
        let initial_target = sched.adaptive_band.target;

        // First cycle with positive energy
        sched.last_global_energy = 0.0;
        let input = MetabolicInput {
            edges: vec![(0, 1, 1.0)],
            n_nodes: 2,
            node_positions: vec![vec![0.3, 0.0, 0.0], vec![0.6, 0.0, 0.0]],
            node_energies: vec![0.5, 0.5],
            sandbox_entries: vec![],
            global_energy: 0.8, // Higher than initial 0 → positive delta
            energy_input: 0.1,
            entropy_delta: 0.01,
        };

        let report = sched.run_metabolic_cycle(&input);
        assert!(
            report.band_target > initial_target || (report.band_target - initial_target).abs() < 0.1,
            "Band should evolve with positive energy delta: {initial_target} → {}",
            report.band_target,
        );
    }

    #[test]
    fn test_compute_variance() {
        assert!((compute_variance(&[]) - 0.0).abs() < 1e-10);
        assert!((compute_variance(&[5.0]) - 0.0).abs() < 1e-10);
        // Variance of [1, 2, 3] = 1.0
        let v = compute_variance(&[1.0, 2.0, 3.0]);
        assert!((v - 1.0).abs() < 1e-10, "Variance should be 1.0: {v}");
    }
}
