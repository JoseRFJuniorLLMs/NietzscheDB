// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! NietzscheEvolve — Phase 4: Cognitive Strategy Evolution for EVA.
//!
//! Evolves the cognitive parameters that govern EVA's memory consolidation,
//! attention, learning, and perception systems. These parameters control
//! how the AgencyEngine's various phases interact.
//!
//! ## What is evolved
//!
//! A `CognitiveGenome` encodes:
//! - Memory consolidation threshold
//! - Attention decay rate (ECAN)
//! - Hebbian learning rate
//! - Novelty weight (curiosity vs familiarity)
//! - Perception confidence threshold
//! - Dream cycle interval
//! - Thermodynamic thermal conductivity
//! - Growth distance threshold
//!
//! ## Fitness
//!
//! Measured by:
//! 1. Epistemic score improvement over simulated ticks
//! 2. Node utilization (accessed vs total)
//! 3. Information surprise (entropy of predictions vs reality)
//! 4. Energy stability (avoiding oscillation)

use std::collections::VecDeque;

/// Cognitive strategy genome for EVA.
///
/// Each parameter maps to a specific AgencyConfig field or
/// module parameter in the agency engine.
#[derive(Debug, Clone)]
pub struct CognitiveGenome {
    /// When to consolidate episodic → semantic memory [0.1..0.9].
    /// Lower = consolidate aggressively, higher = keep more episodic.
    pub consolidation_threshold: f32,

    /// ECAN attention decay rate [0.01..0.2].
    /// How fast attention fades from nodes.
    pub attention_decay: f32,

    /// Hebbian learning rate [0.01..0.3].
    /// Strength of association by co-activation.
    pub hebbian_rate: f32,

    /// Novelty weight [0.1..0.9].
    /// Balance between curiosity (explore) and familiarity (exploit).
    pub novelty_weight: f32,

    /// Minimum confidence for perception to create nodes [0.3..0.9].
    pub perception_threshold: f32,

    /// Ticks between dream consolidation cycles [10..200].
    pub dream_interval: u32,

    /// Thermal conductivity for energy redistribution [0.01..0.2].
    pub thermal_conductivity: f32,

    /// Distance threshold for autonomous graph growth [0.5..3.0].
    pub growth_distance: f32,

    /// Temporal decay lambda for edge weight [0.0000001..0.001].
    pub decay_lambda: f64,

    /// Cognitive cluster radius for concept detection [0.1..0.8].
    pub cluster_radius: f32,

    /// Generation and fitness tracking.
    pub generation: u32,
    pub fitness: f64,
}

impl Default for CognitiveGenome {
    fn default() -> Self {
        Self {
            consolidation_threshold: 0.3,
            attention_decay: 0.05,
            hebbian_rate: 0.1,
            novelty_weight: 0.4,
            perception_threshold: 0.6,
            dream_interval: 100,
            thermal_conductivity: 0.05,
            growth_distance: 1.5,
            decay_lambda: 0.0000001,
            cluster_radius: 0.3,
            generation: 0,
            fitness: 0.0,
        }
    }
}

/// Configuration for cognitive evolution.
#[derive(Debug, Clone)]
pub struct CogEvolveConfig {
    pub population_size: usize,
    pub max_generations: u32,
    pub tournament_size: usize,
    pub mutation_rate: f64,
    pub mutation_step: f64,
    /// Number of simulated ticks per evaluation.
    pub eval_ticks: usize,
}

impl Default for CogEvolveConfig {
    fn default() -> Self {
        Self {
            population_size: 8,
            max_generations: 6,
            tournament_size: 3,
            mutation_rate: 0.35,
            mutation_step: 0.12,
            eval_ticks: 30,
        }
    }
}

/// Simulated cognitive metrics after applying a genome for N ticks.
#[derive(Debug, Clone)]
pub struct CogSimResult {
    /// Epistemic score delta.
    pub epistemic_delta: f64,
    /// Fraction of nodes accessed at least once.
    pub utilization: f64,
    /// Information entropy of energy distribution.
    pub entropy: f64,
    /// Energy oscillation (std of energy changes per tick).
    pub stability: f64,
    /// New edges created (growth effectiveness).
    pub edges_grown: usize,
    /// Concepts detected.
    pub concepts_found: usize,
}

/// Compute fitness from cognitive simulation result.
pub fn cognitive_fitness(result: &CogSimResult) -> f64 {
    // Epistemic improvement is primary
    let epistemic_score = result.epistemic_delta.max(0.0).min(1.0);

    // Utilization: want 50-80%
    let util_score = 1.0 - (result.utilization - 0.65).abs() * 3.0;

    // Entropy: want moderate (not too uniform, not too concentrated)
    let entropy_score = if result.entropy > 0.3 && result.entropy < 0.85 {
        1.0
    } else {
        0.3
    };

    // Stability: lower oscillation is better
    let stability_score = (1.0 - result.stability * 5.0).max(0.0);

    // Growth: some edges is good, too many is bloat
    let growth_score = if result.edges_grown > 0 && result.edges_grown < 20 {
        1.0
    } else if result.edges_grown == 0 {
        0.3
    } else {
        0.5
    };

    0.30 * epistemic_score
        + 0.20 * util_score.max(0.0)
        + 0.15 * entropy_score
        + 0.20 * stability_score
        + 0.15 * growth_score
}

/// Report from cognitive evolution.
#[derive(Debug, Clone)]
pub struct CogEvolveReport {
    pub best: CognitiveGenome,
    pub fitness_history: Vec<f64>,
    pub generations_completed: u32,
    pub population_size: usize,
}

/// Evolutionary engine for cognitive strategies.
pub struct CogEvolver {
    config: CogEvolveConfig,
    population: Vec<CognitiveGenome>,
    fitness_history: VecDeque<f64>,
    generation: u32,
    rng_state: u64,
}

impl CogEvolver {
    pub fn new(config: CogEvolveConfig) -> Self {
        let mut evolver = Self {
            population: Vec::with_capacity(config.population_size),
            fitness_history: VecDeque::new(),
            generation: 0,
            rng_state: 31337,
            config,
        };
        evolver.init_population();
        evolver
    }

    fn init_population(&mut self) {
        let n = self.config.population_size;
        for i in 0..n {
            let f = i as f32 / n.max(1) as f32;
            self.population.push(CognitiveGenome {
                consolidation_threshold: 0.1 + f * 0.8,
                attention_decay: 0.01 + f * 0.19,
                hebbian_rate: 0.01 + f * 0.29,
                novelty_weight: 0.1 + f * 0.8,
                perception_threshold: 0.3 + f * 0.6,
                dream_interval: 10 + (f * 190.0) as u32,
                thermal_conductivity: 0.01 + f * 0.19,
                growth_distance: 0.5 + f * 2.5,
                decay_lambda: 0.0000001 + f as f64 * 0.000999,
                cluster_radius: 0.1 + f * 0.7,
                generation: 0,
                fitness: 0.0,
            });
        }
    }

    pub fn population(&self) -> &[CognitiveGenome] {
        &self.population
    }

    pub fn set_fitness(&mut self, idx: usize, fitness: f64) {
        if let Some(g) = self.population.get_mut(idx) {
            g.fitness = fitness;
        }
    }

    pub fn evolve_generation(&mut self) {
        self.generation += 1;

        let best_fitness = self.population.iter()
            .map(|g| g.fitness)
            .fold(f64::NEG_INFINITY, f64::max);
        self.fitness_history.push_back(best_fitness);

        let pop_size = self.config.population_size;
        let mut new_pop = Vec::with_capacity(pop_size);

        // Elitism
        if let Some(best) = self.population.iter().max_by(|a, b| {
            a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let mut elite = best.clone();
            elite.generation = self.generation;
            new_pop.push(elite);
        }

        while new_pop.len() < pop_size {
            let a = self.tournament_select();
            let b = self.tournament_select();
            let mut child = self.crossover(&a, &b);
            self.mutate(&mut child);
            child.generation = self.generation;
            child.fitness = 0.0;
            new_pop.push(child);
        }

        self.population = new_pop;
    }

    pub fn best_genome(&self) -> CognitiveGenome {
        self.population.iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap_or_default()
    }

    pub fn report(&self) -> CogEvolveReport {
        CogEvolveReport {
            best: self.best_genome(),
            fitness_history: self.fitness_history.iter().copied().collect(),
            generations_completed: self.generation,
            population_size: self.config.population_size,
        }
    }

    fn next_rand(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    fn tournament_select(&mut self) -> CognitiveGenome {
        let mut best: Option<&CognitiveGenome> = None;
        for _ in 0..self.config.tournament_size {
            let idx = (self.next_rand() * self.population.len() as f64) as usize;
            let idx = idx.min(self.population.len().saturating_sub(1));
            let candidate = &self.population[idx];
            if best.map_or(true, |b| candidate.fitness > b.fitness) {
                best = Some(candidate);
            }
        }
        best.cloned().unwrap_or_default()
    }

    fn crossover(&mut self, a: &CognitiveGenome, b: &CognitiveGenome) -> CognitiveGenome {
        CognitiveGenome {
            consolidation_threshold: if self.next_rand() < 0.5 { a.consolidation_threshold } else { b.consolidation_threshold },
            attention_decay: if self.next_rand() < 0.5 { a.attention_decay } else { b.attention_decay },
            hebbian_rate: if self.next_rand() < 0.5 { a.hebbian_rate } else { b.hebbian_rate },
            novelty_weight: if self.next_rand() < 0.5 { a.novelty_weight } else { b.novelty_weight },
            perception_threshold: if self.next_rand() < 0.5 { a.perception_threshold } else { b.perception_threshold },
            dream_interval: if self.next_rand() < 0.5 { a.dream_interval } else { b.dream_interval },
            thermal_conductivity: if self.next_rand() < 0.5 { a.thermal_conductivity } else { b.thermal_conductivity },
            growth_distance: if self.next_rand() < 0.5 { a.growth_distance } else { b.growth_distance },
            decay_lambda: if self.next_rand() < 0.5 { a.decay_lambda } else { b.decay_lambda },
            cluster_radius: if self.next_rand() < 0.5 { a.cluster_radius } else { b.cluster_radius },
            generation: 0,
            fitness: 0.0,
        }
    }

    fn mutate(&mut self, g: &mut CognitiveGenome) {
        let s = self.config.mutation_step as f32;

        if self.next_rand() < self.config.mutation_rate {
            g.consolidation_threshold += (self.next_rand() as f32 - 0.5) * 2.0 * s * 0.8;
            g.consolidation_threshold = g.consolidation_threshold.clamp(0.1, 0.9);
        }
        if self.next_rand() < self.config.mutation_rate {
            g.attention_decay += (self.next_rand() as f32 - 0.5) * 2.0 * s * 0.2;
            g.attention_decay = g.attention_decay.clamp(0.01, 0.2);
        }
        if self.next_rand() < self.config.mutation_rate {
            g.hebbian_rate += (self.next_rand() as f32 - 0.5) * 2.0 * s * 0.3;
            g.hebbian_rate = g.hebbian_rate.clamp(0.01, 0.3);
        }
        if self.next_rand() < self.config.mutation_rate {
            g.novelty_weight += (self.next_rand() as f32 - 0.5) * 2.0 * s * 0.8;
            g.novelty_weight = g.novelty_weight.clamp(0.1, 0.9);
        }
        if self.next_rand() < self.config.mutation_rate {
            g.perception_threshold += (self.next_rand() as f32 - 0.5) * 2.0 * s * 0.6;
            g.perception_threshold = g.perception_threshold.clamp(0.3, 0.9);
        }
        if self.next_rand() < self.config.mutation_rate {
            let delta = ((self.next_rand() - 0.5) * 2.0 * self.config.mutation_step * 190.0) as i32;
            g.dream_interval = (g.dream_interval as i32 + delta).max(10).min(200) as u32;
        }
        if self.next_rand() < self.config.mutation_rate {
            g.thermal_conductivity += (self.next_rand() as f32 - 0.5) * 2.0 * s * 0.2;
            g.thermal_conductivity = g.thermal_conductivity.clamp(0.01, 0.2);
        }
        if self.next_rand() < self.config.mutation_rate {
            g.growth_distance += (self.next_rand() as f32 - 0.5) * 2.0 * s * 2.5;
            g.growth_distance = g.growth_distance.clamp(0.5, 3.0);
        }
        if self.next_rand() < self.config.mutation_rate {
            g.decay_lambda *= 1.0 + (self.next_rand() - 0.5) * self.config.mutation_step * 2.0;
            g.decay_lambda = g.decay_lambda.clamp(0.0000001, 0.001);
        }
        if self.next_rand() < self.config.mutation_rate {
            g.cluster_radius += (self.next_rand() as f32 - 0.5) * 2.0 * s * 0.7;
            g.cluster_radius = g.cluster_radius.clamp(0.1, 0.8);
        }
    }
}

/// Convert a CognitiveGenome to AgencyConfig overrides (env var format).
///
/// Returns a Vec of (key, value) pairs for environment variables.
impl CognitiveGenome {
    pub fn to_env_vars(&self) -> Vec<(String, String)> {
        vec![
            ("AGENCY_THERMAL_CONDUCTIVITY".into(), format!("{}", self.thermal_conductivity)),
            ("AGENCY_GROWTH_DISTANCE_THRESHOLD".into(), format!("{}", self.growth_distance)),
            ("AGENCY_TEMPORAL_DECAY_LAMBDA".into(), format!("{}", self.decay_lambda)),
            ("AGENCY_COGNITIVE_CLUSTER_RADIUS".into(), format!("{}", self.cluster_radius)),
        ]
    }

    /// Serialize genome to JSON for storage in NietzscheDB.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"consolidation_threshold":{:.4},"attention_decay":{:.4},"hebbian_rate":{:.4},"novelty_weight":{:.4},"perception_threshold":{:.4},"dream_interval":{},"thermal_conductivity":{:.4},"growth_distance":{:.4},"decay_lambda":{:.10},"cluster_radius":{:.4},"generation":{},"fitness":{:.6}}}"#,
            self.consolidation_threshold,
            self.attention_decay,
            self.hebbian_rate,
            self.novelty_weight,
            self.perception_threshold,
            self.dream_interval,
            self.thermal_conductivity,
            self.growth_distance,
            self.decay_lambda,
            self.cluster_radius,
            self.generation,
            self.fitness,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_genome() {
        let g = CognitiveGenome::default();
        assert!((g.consolidation_threshold - 0.3).abs() < 0.01);
        assert!((g.novelty_weight - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_cognitive_fitness() {
        let result = CogSimResult {
            epistemic_delta: 0.5,
            utilization: 0.65,
            entropy: 0.5,
            stability: 0.05,
            edges_grown: 5,
            concepts_found: 2,
        };
        let f = cognitive_fitness(&result);
        assert!(f > 0.0);
        assert!(f <= 1.0);
    }

    #[test]
    fn test_evolution_runs() {
        let config = CogEvolveConfig {
            population_size: 4,
            max_generations: 2,
            ..Default::default()
        };
        let mut evolver = CogEvolver::new(config);
        assert_eq!(evolver.population().len(), 4);

        // Set some fitness
        for i in 0..4 {
            evolver.set_fitness(i, i as f64 * 0.25);
        }

        evolver.evolve_generation();
        assert_eq!(evolver.population().len(), 4);
        assert!(evolver.best_genome().fitness >= 0.0);
    }

    #[test]
    fn test_genome_json() {
        let g = CognitiveGenome::default();
        let json = g.to_json();
        assert!(json.contains("consolidation_threshold"));
        assert!(json.contains("fitness"));
    }

    #[test]
    fn test_env_vars() {
        let g = CognitiveGenome::default();
        let vars = g.to_env_vars();
        assert!(!vars.is_empty());
        assert!(vars.iter().any(|(k, _)| k == "AGENCY_THERMAL_CONDUCTIVITY"));
    }
}
