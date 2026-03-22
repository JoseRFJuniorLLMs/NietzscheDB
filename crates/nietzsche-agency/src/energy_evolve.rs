// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! NietzscheEvolve — Phase 2: Evolutionary Energy Functions.
//!
//! Evolves the energy decay/boost heuristics used by the Thermodynamics engine
//! (Phase XIII) and the general energy management system.
//!
//! Instead of fixed constants for decay rates, hub bonuses, and depth scaling,
//! we evolve these parameters via a population-based search. The fitness function
//! measures epistemic quality improvement over N simulated ticks.
//!
//! ## AlphaEvolve Analogy
//!
//! AlphaEvolve evolves algorithm code → we evolve the **energy landscape function**
//! that governs how knowledge nodes gain/lose importance over time.

use std::collections::VecDeque;

/// A single energy function genome.
///
/// These parameters control how energy flows in the knowledge graph.
#[derive(Debug, Clone)]
pub struct EnergyGenome {
    /// Base decay rate per tick [0.001..0.1].
    /// Higher = faster forgetting.
    pub decay_rate: f32,
    /// Energy boost on access [0.01..0.2].
    /// How much energy a node gains when queried/accessed.
    pub boost_on_access: f32,
    /// Hub bonus factor [0.0..0.5].
    /// Proportional to degree: hubs decay slower.
    pub hub_bonus_factor: f32,
    /// Depth scaling [0.5..2.0].
    /// > 1.0 means deeper nodes (more specific) decay faster.
    /// < 1.0 means deeper nodes are more persistent.
    pub depth_scaling: f32,
    /// Coherence bonus [0.0..0.3].
    /// Nodes with high local coherence get energy bonus.
    pub coherence_bonus: f32,
    /// Thermal conductivity override [0.01..0.2].
    /// Controls heat flow rate between connected nodes.
    pub thermal_conductivity: f32,
    /// Generation and fitness.
    pub generation: u32,
    pub fitness: f64,
}

impl Default for EnergyGenome {
    fn default() -> Self {
        Self {
            decay_rate: 0.01,
            boost_on_access: 0.05,
            hub_bonus_factor: 0.1,
            depth_scaling: 1.0,
            coherence_bonus: 0.05,
            thermal_conductivity: 0.05,
            generation: 0,
            fitness: 0.0,
        }
    }
}

impl EnergyGenome {
    /// Compute the effective decay for a node given its properties.
    ///
    /// ```text
    /// effective_decay = decay_rate * depth_scaling^depth * (1 - hub_bonus * log2(degree+1))
    /// ```
    pub fn effective_decay(&self, depth: f32, degree: usize) -> f32 {
        let depth_factor = self.depth_scaling.powf(depth.clamp(0.0, 1.0));
        let hub_factor = 1.0 - self.hub_bonus_factor * (degree as f32 + 1.0).log2().min(5.0) / 5.0;
        self.decay_rate * depth_factor * hub_factor.max(0.01)
    }

    /// Compute the effective boost for a node on access.
    ///
    /// Nodes with high local coherence get an extra bonus.
    pub fn effective_boost(&self, local_coherence: f32) -> f32 {
        self.boost_on_access + self.coherence_bonus * local_coherence.clamp(0.0, 1.0)
    }
}

/// Configuration for energy evolution.
#[derive(Debug, Clone)]
pub struct EnergyEvolveConfig {
    /// Population size.
    pub population_size: usize,
    /// Maximum generations.
    pub max_generations: u32,
    /// Tournament selection size.
    pub tournament_size: usize,
    /// Mutation rate.
    pub mutation_rate: f64,
    /// Mutation step (fraction of parameter range).
    pub mutation_step: f64,
    /// Number of simulated ticks for fitness evaluation.
    pub sim_ticks: usize,
}

impl Default for EnergyEvolveConfig {
    fn default() -> Self {
        Self {
            population_size: 10,
            max_generations: 8,
            tournament_size: 3,
            mutation_rate: 0.35,
            mutation_step: 0.15,
            sim_ticks: 20,
        }
    }
}

/// Simulated graph state for fitness evaluation.
#[derive(Debug, Clone)]
pub struct SimNode {
    pub energy: f32,
    pub depth: f32,
    pub degree: usize,
    pub coherence: f32,
    pub accessed: bool,
}

/// Result of simulating a genome on a set of nodes.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Mean energy after simulation.
    pub mean_energy: f64,
    /// Energy standard deviation.
    pub energy_std: f64,
    /// Fraction of nodes with energy > 0.01 (alive).
    pub alive_ratio: f64,
    /// Fraction of hub nodes (degree > 5) with energy > 0.1.
    pub hub_survival: f64,
    /// Energy coefficient of variation (temperature proxy).
    pub temperature: f64,
}

/// Simulate applying an energy genome to a set of nodes for N ticks.
pub fn simulate_energy_genome(
    genome: &EnergyGenome,
    nodes: &mut [SimNode],
    ticks: usize,
) -> SimulationResult {
    for _tick in 0..ticks {
        for node in nodes.iter_mut() {
            // Apply decay
            let decay = genome.effective_decay(node.depth, node.degree);
            node.energy = (node.energy - decay).max(0.0);

            // Apply access boost (simulated: 10% chance per tick)
            if node.accessed {
                let boost = genome.effective_boost(node.coherence);
                node.energy = (node.energy + boost).min(1.0);
                node.accessed = false;
            }
        }
    }

    // Compute statistics
    let n = nodes.len() as f64;
    if n < 1.0 {
        return SimulationResult {
            mean_energy: 0.0,
            energy_std: 0.0,
            alive_ratio: 0.0,
            hub_survival: 0.0,
            temperature: 0.0,
        };
    }

    let mean: f64 = nodes.iter().map(|n| n.energy as f64).sum::<f64>() / n;
    let var: f64 = nodes.iter().map(|n| {
        let d = n.energy as f64 - mean;
        d * d
    }).sum::<f64>() / n;
    let std = var.sqrt();

    let alive = nodes.iter().filter(|n| n.energy > 0.01).count() as f64 / n;
    let hubs: Vec<&SimNode> = nodes.iter().filter(|n| n.degree > 5).collect();
    let hub_survival = if hubs.is_empty() {
        1.0
    } else {
        hubs.iter().filter(|n| n.energy > 0.1).count() as f64 / hubs.len() as f64
    };

    let temperature = if mean > 1e-9 { std / mean } else { 0.0 };

    SimulationResult {
        mean_energy: mean,
        energy_std: std,
        alive_ratio: alive,
        hub_survival,
        temperature,
    }
}

/// Compute fitness from simulation result.
///
/// Optimal state: liquid phase (moderate temperature), high hub survival,
/// reasonable alive ratio (not all dead, not all maxed out).
pub fn energy_fitness(result: &SimulationResult) -> f64 {
    // Alive ratio: want ~70-90% alive
    let alive_score = 1.0 - (result.alive_ratio - 0.8).abs() * 2.0;

    // Hub survival: want > 80%
    let hub_score = result.hub_survival;

    // Temperature: want liquid phase (0.15 < T < 0.85)
    let temp_score = if result.temperature > 0.15 && result.temperature < 0.85 {
        1.0
    } else if result.temperature < 0.05 || result.temperature > 1.5 {
        0.1
    } else {
        0.5
    };

    // Mean energy: want moderate (not too high, not too low)
    let energy_score = 1.0 - (result.mean_energy - 0.4).abs() * 2.0;

    0.30 * alive_score.max(0.0)
        + 0.25 * hub_score
        + 0.25 * temp_score
        + 0.20 * energy_score.max(0.0)
}

/// Report from energy evolution.
#[derive(Debug, Clone)]
pub struct EnergyEvolveReport {
    pub best: EnergyGenome,
    pub fitness_history: Vec<f64>,
    pub generations_completed: u32,
}

/// Evolutionary engine for energy functions.
pub struct EnergyEvolver {
    config: EnergyEvolveConfig,
    population: Vec<EnergyGenome>,
    fitness_history: VecDeque<f64>,
    generation: u32,
    rng_state: u64,
}

impl EnergyEvolver {
    pub fn new(config: EnergyEvolveConfig) -> Self {
        let mut evolver = Self {
            population: Vec::with_capacity(config.population_size),
            fitness_history: VecDeque::new(),
            generation: 0,
            rng_state: 7919, // prime seed
            config,
        };
        evolver.init_population();
        evolver
    }

    fn init_population(&mut self) {
        let n = self.config.population_size;
        for i in 0..n {
            let frac = i as f32 / n.max(1) as f32;
            self.population.push(EnergyGenome {
                decay_rate: 0.001 + frac * 0.099,
                boost_on_access: 0.01 + frac * 0.19,
                hub_bonus_factor: frac * 0.5,
                depth_scaling: 0.5 + frac * 1.5,
                coherence_bonus: frac * 0.3,
                thermal_conductivity: 0.01 + frac * 0.19,
                generation: 0,
                fitness: 0.0,
            });
        }
    }

    pub fn population(&self) -> &[EnergyGenome] {
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

    pub fn best_genome(&self) -> EnergyGenome {
        self.population.iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap_or_default()
    }

    pub fn report(&self) -> EnergyEvolveReport {
        EnergyEvolveReport {
            best: self.best_genome(),
            fitness_history: self.fitness_history.iter().copied().collect(),
            generations_completed: self.generation,
        }
    }

    fn next_rand(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    fn tournament_select(&mut self) -> EnergyGenome {
        let pop_len = self.population.len();
        let indices: Vec<usize> = (0..self.config.tournament_size)
            .map(|_| {
                let idx = (self.next_rand() * pop_len as f64) as usize;
                idx.min(pop_len.saturating_sub(1))
            })
            .collect();
        let mut best: Option<&EnergyGenome> = None;
        for idx in indices {
            let candidate = &self.population[idx];
            if best.map_or(true, |b| candidate.fitness > b.fitness) {
                best = Some(candidate);
            }
        }
        best.cloned().unwrap_or_default()
    }

    fn crossover(&mut self, a: &EnergyGenome, b: &EnergyGenome) -> EnergyGenome {
        EnergyGenome {
            decay_rate: if self.next_rand() < 0.5 { a.decay_rate } else { b.decay_rate },
            boost_on_access: if self.next_rand() < 0.5 { a.boost_on_access } else { b.boost_on_access },
            hub_bonus_factor: if self.next_rand() < 0.5 { a.hub_bonus_factor } else { b.hub_bonus_factor },
            depth_scaling: if self.next_rand() < 0.5 { a.depth_scaling } else { b.depth_scaling },
            coherence_bonus: if self.next_rand() < 0.5 { a.coherence_bonus } else { b.coherence_bonus },
            thermal_conductivity: if self.next_rand() < 0.5 { a.thermal_conductivity } else { b.thermal_conductivity },
            generation: 0,
            fitness: 0.0,
        }
    }

    fn mutate(&mut self, g: &mut EnergyGenome) {
        let s = self.config.mutation_step as f32;
        if self.next_rand() < self.config.mutation_rate {
            g.decay_rate += (self.next_rand() as f32 - 0.5) * 2.0 * s * 0.1;
            g.decay_rate = g.decay_rate.clamp(0.001, 0.1);
        }
        if self.next_rand() < self.config.mutation_rate {
            g.boost_on_access += (self.next_rand() as f32 - 0.5) * 2.0 * s * 0.2;
            g.boost_on_access = g.boost_on_access.clamp(0.01, 0.2);
        }
        if self.next_rand() < self.config.mutation_rate {
            g.hub_bonus_factor += (self.next_rand() as f32 - 0.5) * 2.0 * s * 0.5;
            g.hub_bonus_factor = g.hub_bonus_factor.clamp(0.0, 0.5);
        }
        if self.next_rand() < self.config.mutation_rate {
            g.depth_scaling += (self.next_rand() as f32 - 0.5) * 2.0 * s * 1.5;
            g.depth_scaling = g.depth_scaling.clamp(0.5, 2.0);
        }
        if self.next_rand() < self.config.mutation_rate {
            g.coherence_bonus += (self.next_rand() as f32 - 0.5) * 2.0 * s * 0.3;
            g.coherence_bonus = g.coherence_bonus.clamp(0.0, 0.3);
        }
        if self.next_rand() < self.config.mutation_rate {
            g.thermal_conductivity += (self.next_rand() as f32 - 0.5) * 2.0 * s * 0.2;
            g.thermal_conductivity = g.thermal_conductivity.clamp(0.01, 0.2);
        }
    }
}

/// Run a complete energy evolution using a node snapshot.
///
/// `initial_nodes` should be a representative sample of the graph.
pub fn run_energy_evolution(
    config: EnergyEvolveConfig,
    initial_nodes: &[SimNode],
) -> EnergyEvolveReport {
    let mut evolver = EnergyEvolver::new(config.clone());

    for _gen in 0..config.max_generations {
        for i in 0..evolver.population().len() {
            let genome = evolver.population()[i].clone();
            let mut sim_nodes: Vec<SimNode> = initial_nodes.to_vec();
            let result = simulate_energy_genome(&genome, &mut sim_nodes, config.sim_ticks);
            let fitness = energy_fitness(&result);
            evolver.set_fitness(i, fitness);
        }
        evolver.evolve_generation();
    }

    // Final evaluation
    for i in 0..evolver.population().len() {
        let genome = evolver.population()[i].clone();
        let mut sim_nodes: Vec<SimNode> = initial_nodes.to_vec();
        let result = simulate_energy_genome(&genome, &mut sim_nodes, config.sim_ticks);
        let fitness = energy_fitness(&result);
        evolver.set_fitness(i, fitness);
    }

    evolver.report()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_nodes() -> Vec<SimNode> {
        vec![
            SimNode { energy: 0.8, depth: 0.1, degree: 10, coherence: 0.7, accessed: true },
            SimNode { energy: 0.5, depth: 0.3, degree: 3, coherence: 0.5, accessed: false },
            SimNode { energy: 0.2, depth: 0.7, degree: 1, coherence: 0.3, accessed: false },
            SimNode { energy: 0.9, depth: 0.05, degree: 20, coherence: 0.9, accessed: true },
            SimNode { energy: 0.1, depth: 0.9, degree: 0, coherence: 0.1, accessed: false },
        ]
    }

    #[test]
    fn test_effective_decay() {
        let g = EnergyGenome::default();
        let shallow_hub = g.effective_decay(0.1, 15);
        let deep_leaf = g.effective_decay(0.9, 1);
        // Hubs should decay slower
        assert!(shallow_hub < deep_leaf || (shallow_hub - deep_leaf).abs() < 0.001);
    }

    #[test]
    fn test_simulation_runs() {
        let genome = EnergyGenome::default();
        let mut nodes = sample_nodes();
        let result = simulate_energy_genome(&genome, &mut nodes, 10);
        assert!(result.mean_energy >= 0.0);
        assert!(result.alive_ratio >= 0.0 && result.alive_ratio <= 1.0);
    }

    #[test]
    fn test_energy_evolution() {
        let config = EnergyEvolveConfig {
            population_size: 6,
            max_generations: 3,
            sim_ticks: 5,
            ..Default::default()
        };
        let nodes = sample_nodes();
        let report = run_energy_evolution(config, &nodes);
        assert!(report.generations_completed > 0);
        assert!(report.best.fitness >= 0.0);
    }
}
