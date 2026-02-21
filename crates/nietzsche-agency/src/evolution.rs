//! Open Evolution — L-System rule mutation and selection.
//!
//! Transforms health metrics into adaptive L-System rule strategies.
//! Instead of fixed production rules, the evolution module observes
//! graph health over time and suggests rule modifications that better
//! fit the graph's current needs.
//!
//! ## Philosophy
//!
//! In Nietzsche's concept of *Eternal Recurrence*, patterns repeat
//! with variation — each cycle is an opportunity for refinement. The
//! evolution module embodies this: L-System rules are not fixed axioms
//! but living parameters that adapt through observation.

use nietzsche_graph::GraphStorage;

use crate::error::AgencyError;
use crate::observer::HealthReport;

/// High-level evolution strategy determined by health metrics.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum EvolutionStrategy {
    /// Many gaps, energy is healthy — grow to fill.
    FavorGrowth,
    /// High entropy or energy inflation — prune excess.
    FavorPruning,
    /// Steady state — moderate growth with maintenance pruning.
    Balanced,
    /// Hausdorff out of fractal range — structural repair needed.
    Consolidate,
}

/// A specification for an evolved production rule.
/// The server converts this into an actual `ProductionRule`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvolvedRule {
    pub name: String,
    pub rule_type: EvolvedRuleType,
    pub energy_threshold: f32,
    pub max_generation: u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum EvolvedRuleType {
    GrowthChild,
    LateralAssociation,
    PruneFading,
    EnergyBoost { delta: f32 },
}

/// Fitness tracking for rule sets across generations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvolutionState {
    pub generation: u64,
    pub last_strategy: String,
    pub fitness_history: Vec<FitnessEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FitnessEntry {
    pub generation: u64,
    pub strategy: String,
    pub hausdorff: f32,
    pub mean_energy: f32,
    pub gap_count: usize,
    pub fitness: f64,
}

impl Default for EvolutionState {
    fn default() -> Self {
        Self {
            generation: 0,
            last_strategy: "Balanced".to_string(),
            fitness_history: Vec::new(),
        }
    }
}

const EVOLUTION_META_KEY: &str = "agency:evolution_state";

/// Rule evolution engine.
pub struct RuleEvolution;

impl RuleEvolution {
    /// Determine the best evolution strategy from a health report.
    pub fn suggest_strategy(report: &HealthReport) -> EvolutionStrategy {
        let gap_ratio = report.gap_count as f64 / 80.0;
        let energy_ok = report.mean_energy > 0.3 && report.mean_energy < 0.8;
        let fractal_ok = report.is_fractal;
        let entropy_high = report.entropy_spike_count > 2;

        if !fractal_ok {
            return EvolutionStrategy::Consolidate;
        }
        if gap_ratio > 0.3 && energy_ok {
            return EvolutionStrategy::FavorGrowth;
        }
        if entropy_high || report.mean_energy > 0.8 {
            return EvolutionStrategy::FavorPruning;
        }

        EvolutionStrategy::Balanced
    }

    /// Generate evolved rules for a given strategy.
    pub fn evolve_rules(strategy: &EvolutionStrategy, generation: u64) -> Vec<EvolvedRule> {
        match strategy {
            EvolutionStrategy::FavorGrowth => vec![
                EvolvedRule {
                    name: format!("evo-growth-g{}", generation),
                    rule_type: EvolvedRuleType::GrowthChild,
                    energy_threshold: 0.3,
                    max_generation: 5,
                },
                EvolvedRule {
                    name: format!("evo-lateral-g{}", generation),
                    rule_type: EvolvedRuleType::LateralAssociation,
                    energy_threshold: 0.4,
                    max_generation: 4,
                },
                EvolvedRule {
                    name: format!("evo-prune-g{}", generation),
                    rule_type: EvolvedRuleType::PruneFading,
                    energy_threshold: 0.05,
                    max_generation: u32::MAX,
                },
            ],
            EvolutionStrategy::FavorPruning => vec![
                EvolvedRule {
                    name: format!("evo-growth-g{}", generation),
                    rule_type: EvolvedRuleType::GrowthChild,
                    energy_threshold: 0.7,
                    max_generation: 2,
                },
                EvolvedRule {
                    name: format!("evo-prune-g{}", generation),
                    rule_type: EvolvedRuleType::PruneFading,
                    energy_threshold: 0.2,
                    max_generation: u32::MAX,
                },
                EvolvedRule {
                    name: format!("evo-drain-g{}", generation),
                    rule_type: EvolvedRuleType::EnergyBoost { delta: -0.05 },
                    energy_threshold: 0.9,
                    max_generation: u32::MAX,
                },
            ],
            EvolutionStrategy::Balanced => vec![
                EvolvedRule {
                    name: format!("evo-growth-g{}", generation),
                    rule_type: EvolvedRuleType::GrowthChild,
                    energy_threshold: 0.5,
                    max_generation: 3,
                },
                EvolvedRule {
                    name: format!("evo-lateral-g{}", generation),
                    rule_type: EvolvedRuleType::LateralAssociation,
                    energy_threshold: 0.5,
                    max_generation: 3,
                },
                EvolvedRule {
                    name: format!("evo-prune-g{}", generation),
                    rule_type: EvolvedRuleType::PruneFading,
                    energy_threshold: 0.1,
                    max_generation: u32::MAX,
                },
            ],
            EvolutionStrategy::Consolidate => vec![
                EvolvedRule {
                    name: format!("evo-boost-g{}", generation),
                    rule_type: EvolvedRuleType::EnergyBoost { delta: 0.05 },
                    energy_threshold: 0.3,
                    max_generation: u32::MAX,
                },
                EvolvedRule {
                    name: format!("evo-prune-g{}", generation),
                    rule_type: EvolvedRuleType::PruneFading,
                    energy_threshold: 0.08,
                    max_generation: u32::MAX,
                },
            ],
        }
    }

    /// Compute fitness score for a health report.
    pub fn compute_fitness(report: &HealthReport) -> f64 {
        let energy_score = (report.mean_energy as f64).clamp(0.0, 1.0);
        let coherence_score = report.coherence_score.clamp(0.0, 1.0);
        let fractal_score = if report.is_fractal { 1.0 } else { 0.3 };
        let gap_penalty = (report.gap_count as f64 / 80.0).min(0.5);
        let entropy_penalty = (report.entropy_spike_count as f64 / 10.0).min(0.3);

        (energy_score * 0.25 + coherence_score * 0.25 + fractal_score * 0.3)
            - gap_penalty * 0.1
            - entropy_penalty * 0.1
    }

    /// Persist evolution state to CF_META.
    pub fn save_state(
        storage: &GraphStorage,
        state: &EvolutionState,
    ) -> Result<(), AgencyError> {
        let json = serde_json::to_vec(state)
            .map_err(|e| AgencyError::Internal(e.to_string()))?;
        storage.put_meta(EVOLUTION_META_KEY, &json)?;
        Ok(())
    }

    /// Load evolution state from CF_META.
    pub fn load_state(
        storage: &GraphStorage,
    ) -> Result<EvolutionState, AgencyError> {
        match storage.get_meta(EVOLUTION_META_KEY)? {
            Some(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| AgencyError::Internal(e.to_string())),
            None => Ok(EvolutionState::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_graph_suggests_balanced() {
        let report = HealthReport {
            mean_energy: 0.6,
            is_fractal: true,
            gap_count: 5,
            entropy_spike_count: 0,
            coherence_score: 0.8,
            ..HealthReport::default()
        };
        assert_eq!(
            RuleEvolution::suggest_strategy(&report),
            EvolutionStrategy::Balanced
        );
    }

    #[test]
    fn many_gaps_suggests_growth() {
        let report = HealthReport {
            mean_energy: 0.5,
            is_fractal: true,
            gap_count: 30,
            entropy_spike_count: 0,
            ..HealthReport::default()
        };
        assert_eq!(
            RuleEvolution::suggest_strategy(&report),
            EvolutionStrategy::FavorGrowth
        );
    }

    #[test]
    fn high_entropy_suggests_pruning() {
        let report = HealthReport {
            mean_energy: 0.9,
            is_fractal: true,
            gap_count: 2,
            entropy_spike_count: 5,
            ..HealthReport::default()
        };
        assert_eq!(
            RuleEvolution::suggest_strategy(&report),
            EvolutionStrategy::FavorPruning
        );
    }

    #[test]
    fn non_fractal_suggests_consolidate() {
        let report = HealthReport {
            mean_energy: 0.5,
            is_fractal: false,
            global_hausdorff: 0.3,
            ..HealthReport::default()
        };
        assert_eq!(
            RuleEvolution::suggest_strategy(&report),
            EvolutionStrategy::Consolidate
        );
    }

    #[test]
    fn favor_growth_rules_have_low_threshold() {
        let rules = RuleEvolution::evolve_rules(&EvolutionStrategy::FavorGrowth, 1);
        let growth = rules
            .iter()
            .find(|r| matches!(r.rule_type, EvolvedRuleType::GrowthChild))
            .unwrap();
        assert!(growth.energy_threshold < 0.5);
        assert!(growth.max_generation >= 4);
    }

    #[test]
    fn fitness_higher_for_healthy_graph() {
        let healthy = HealthReport {
            mean_energy: 0.7,
            coherence_score: 0.9,
            is_fractal: true,
            gap_count: 2,
            entropy_spike_count: 0,
            ..HealthReport::default()
        };
        let unhealthy = HealthReport {
            mean_energy: 0.1,
            coherence_score: 0.2,
            is_fractal: false,
            gap_count: 30,
            entropy_spike_count: 8,
            ..HealthReport::default()
        };
        assert!(
            RuleEvolution::compute_fitness(&healthy)
                > RuleEvolution::compute_fitness(&unhealthy)
        );
    }

    #[test]
    fn evolution_state_persistence() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();

        let state = EvolutionState {
            generation: 5,
            last_strategy: "FavorGrowth".to_string(),
            fitness_history: vec![FitnessEntry {
                generation: 4,
                strategy: "Balanced".to_string(),
                hausdorff: 1.5,
                mean_energy: 0.6,
                gap_count: 3,
                fitness: 0.72,
            }],
        };

        RuleEvolution::save_state(&storage, &state).unwrap();
        let loaded = RuleEvolution::load_state(&storage).unwrap();
        assert_eq!(loaded.generation, 5);
        assert_eq!(loaded.fitness_history.len(), 1);
    }
}
