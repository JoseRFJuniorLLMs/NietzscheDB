//! # Evolution — autonomous evolution scheduler
//!
//! Orchestrates the full AGI cycle:
//!
//! 1. **Detect**: Run dialectic detection to find tension pairs
//! 2. **Traverse**: Build trajectories between tension pairs
//! 3. **Validate**: GCS-validate each trajectory
//! 4. **Infer**: Classify each trajectory via the inference engine
//! 5. **Synthesize**: Produce Fréchet synthesis for dialectical inferences
//! 6. **Insert**: Feed back synthesis nodes into the graph
//! 7. **Decay**: Apply relevance decay to old synthesis nodes
//! 8. **Repeat**: Schedule the next cycle
//!
//! The [`EvolutionScheduler`] runs as a background task, periodically
//! waking up to perform one full cycle. This is the "heartbeat" of
//! the AGI layer.

use std::time::{Duration, Instant};

use crate::certification::CertificationConfig;
use crate::dialectic::{DialecticConfig, DialecticDetector};
use crate::discovery::{DiscoveryConfig, DiscoveryField};
use crate::feedback_loop::{FeedbackConfig, FeedbackLoop};
use crate::homeostasis::HomeostasisGuard;
use crate::inference_engine::{InferenceConfig, InferenceEngine};
use crate::innovation::{InnovationConfig, InnovationEvaluator};
use crate::relevance_decay::{RelevanceConfig, RelevanceDecay};
use crate::sandbox::{SandboxConfig, SandboxEvaluator};
use crate::spectral::{DriftTracker, SpectralConfig, SpectralMonitor};
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
        }
    }
}

// ─────────────────────────────────────────────
// EvolutionScheduler
// ─────────────────────────────────────────────

/// Autonomous evolution orchestrator.
///
/// Holds all the AGI sub-systems and runs them in a coordinated cycle.
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

    // Stats
    pub total_cycles: u64,
    pub total_syntheses: u64,
    pub total_ruptures: u64,
    pub last_cycle: Option<Instant>,
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
            total_cycles: 0,
            total_syntheses: 0,
            total_ruptures: 0,
            last_cycle: None,
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

    /// Record that a cycle has started.
    pub fn mark_cycle_start(&mut self) {
        self.last_cycle = Some(Instant::now());
        self.total_cycles += 1;
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
            last_cycle_elapsed: self.last_cycle.map(|l| l.elapsed()),
            enabled: self.config.enabled,
        }
    }
}

/// Snapshot of evolution scheduler statistics.
#[derive(Debug, Clone)]
pub struct EvolutionStats {
    pub total_cycles: u64,
    pub total_syntheses: u64,
    pub total_ruptures: u64,
    pub last_cycle_elapsed: Option<Duration>,
    pub enabled: bool,
}

impl std::fmt::Display for EvolutionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Evolution[cycles={}, syntheses={}, ruptures={}, enabled={}]",
            self.total_cycles, self.total_syntheses, self.total_ruptures, self.enabled
        )
    }
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
        sched.mark_cycle_start();
        assert!(!sched.should_run()); // Just started → shouldn't run yet

        // After the interval, should_run would return true
        // (Can't test timing reliably in unit tests without sleeping)
    }

    #[test]
    fn test_stats_display() {
        let sched = EvolutionScheduler::with_defaults();
        let stats = sched.stats();
        let s = format!("{stats}");
        assert!(s.contains("cycles=0"));
        assert!(s.contains("syntheses=0"));
    }
}
