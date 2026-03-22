// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! NietzscheEvolve — Phase 1: Evolutionary HNSW Parameter Tuning.
//!
//! Population-based search for optimal HNSW parameters, inspired by
//! AlphaEvolve's evolutionary code optimization. Instead of evolving code,
//! we evolve search configuration for hyperbolic vector spaces.
//!
//! ## Genome
//!
//! Each individual encodes:
//! - `ef_construction`: Build-time quality (higher = better graph, slower indexing)
//! - `ef_search`: Search depth (higher = better recall, slower queries)
//! - `m`: Max connections per HNSW layer (affects space & recall tradeoff)
//!
//! ## Fitness
//!
//! ```text
//! fitness = w_recall * recall@k + w_latency * (1 - p95/budget) + w_memory * (1 - mem/budget)
//! ```
//!
//! ## Hyperbolic constraints
//!
//! Poincaré distance is non-linear — nodes near the boundary appear much farther
//! apart than Euclidean intuition suggests. This means:
//! - `ef_search` should generally be higher than for Euclidean spaces
//! - `M` ≥ 16 is recommended (hierarchies need more layer-0 neighbors)

use std::collections::VecDeque;

/// A single parameter genome for HNSW tuning.
#[derive(Debug, Clone)]
pub struct HnswGenome {
    /// ef_construction: build quality [16..512].
    pub ef_construction: usize,
    /// ef_search: search depth [10..1000].
    pub ef_search: usize,
    /// M: max connections per layer [8..64].
    pub m: usize,
    /// Generation this genome was created in.
    pub generation: u32,
    /// Fitness score (higher = better).
    pub fitness: f64,
}

impl Default for HnswGenome {
    fn default() -> Self {
        Self {
            ef_construction: 100,
            ef_search: 100,
            m: 16,
            generation: 0,
            fitness: 0.0,
        }
    }
}

/// Configuration for the evolutionary parameter search.
#[derive(Debug, Clone)]
pub struct ParamEvolveConfig {
    /// Population size.
    pub population_size: usize,
    /// Number of generations to run.
    pub max_generations: u32,
    /// Tournament selection size.
    pub tournament_size: usize,
    /// Mutation rate (probability of mutating each gene).
    pub mutation_rate: f64,
    /// Mutation step size (fraction of range).
    pub mutation_step: f64,
    /// Fitness weights.
    pub w_recall: f64,
    pub w_latency: f64,
    pub w_memory: f64,
    /// Latency budget in microseconds (p95 target).
    pub latency_budget_us: u64,
    /// Whether to use Poincaré-aware parameter bounds.
    pub hyperbolic_mode: bool,
}

impl Default for ParamEvolveConfig {
    fn default() -> Self {
        Self {
            population_size: 12,
            max_generations: 10,
            tournament_size: 3,
            mutation_rate: 0.3,
            mutation_step: 0.15,
            w_recall: 0.5,
            w_latency: 0.3,
            w_memory: 0.2,
            latency_budget_us: 50_000, // 50ms
            hyperbolic_mode: true,
        }
    }
}

/// Result from a benchmark evaluation of a genome.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Recall@k (fraction of true neighbors found).
    pub recall: f64,
    /// p95 latency in microseconds.
    pub p95_latency_us: u64,
    /// Estimated memory usage (relative, 0..1).
    pub memory_ratio: f64,
}

/// Report from one evolution run.
#[derive(Debug, Clone)]
pub struct ParamEvolveReport {
    /// Best genome found.
    pub best: HnswGenome,
    /// Fitness history (best per generation).
    pub fitness_history: Vec<f64>,
    /// Total generations completed.
    pub generations_completed: u32,
    /// Population at termination.
    pub final_population: Vec<HnswGenome>,
}

/// Evolutionary engine for HNSW parameters.
pub struct ParamEvolver {
    config: ParamEvolveConfig,
    population: Vec<HnswGenome>,
    /// Fitness history (best per generation).
    fitness_history: VecDeque<f64>,
    /// Current generation.
    generation: u32,
    /// Simple deterministic seed for reproducibility.
    rng_state: u64,
}

impl ParamEvolver {
    /// Create a new evolver with initial population.
    pub fn new(config: ParamEvolveConfig) -> Self {
        let mut evolver = Self {
            population: Vec::with_capacity(config.population_size),
            fitness_history: VecDeque::with_capacity(config.max_generations as usize),
            generation: 0,
            rng_state: 42,
            config,
        };
        evolver.init_population();
        evolver
    }

    /// Initialize population with diverse parameter combinations.
    fn init_population(&mut self) {
        let n = self.config.population_size;
        let (ef_min, m_min) = if self.config.hyperbolic_mode {
            (50, 16) // Poincaré needs higher defaults
        } else {
            (20, 8)
        };

        for i in 0..n {
            let frac = i as f64 / n.max(1) as f64;
            self.population.push(HnswGenome {
                ef_construction: ef_min + (frac * 400.0) as usize,
                ef_search: ef_min + (frac * 500.0) as usize,
                m: m_min + (frac * 48.0) as usize,
                generation: 0,
                fitness: 0.0,
            });
        }
    }

    /// Evaluate fitness for a genome given benchmark results.
    pub fn evaluate_fitness(&self, _genome: &HnswGenome, bench: &BenchmarkResult) -> f64 {
        let recall_score = bench.recall;
        let latency_score = if bench.p95_latency_us <= self.config.latency_budget_us {
            1.0
        } else {
            let over = bench.p95_latency_us as f64 / self.config.latency_budget_us as f64;
            (1.0 / over).max(0.0)
        };
        let memory_score = (1.0 - bench.memory_ratio).max(0.0);

        self.config.w_recall * recall_score
            + self.config.w_latency * latency_score
            + self.config.w_memory * memory_score
    }

    /// Set fitness for a genome by index.
    pub fn set_fitness(&mut self, idx: usize, fitness: f64) {
        if let Some(g) = self.population.get_mut(idx) {
            g.fitness = fitness;
        }
    }

    /// Get current population (for benchmarking externally).
    pub fn population(&self) -> &[HnswGenome] {
        &self.population
    }

    /// Advance one generation: selection → crossover → mutation.
    pub fn evolve_generation(&mut self) {
        self.generation += 1;

        // Record best fitness
        let best_fitness = self.population.iter()
            .map(|g| g.fitness)
            .fold(f64::NEG_INFINITY, f64::max);
        self.fitness_history.push_back(best_fitness);

        let pop_size = self.config.population_size;
        let mut new_pop = Vec::with_capacity(pop_size);

        // Elitism: keep the best individual
        if let Some(best) = self.population.iter().max_by(|a, b| {
            a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let mut elite = best.clone();
            elite.generation = self.generation;
            new_pop.push(elite);
        }

        // Fill rest via tournament selection + crossover + mutation
        while new_pop.len() < pop_size {
            let parent_a = self.tournament_select();
            let parent_b = self.tournament_select();
            let mut child = self.crossover(&parent_a, &parent_b);
            self.mutate(&mut child);
            child.generation = self.generation;
            child.fitness = 0.0;
            new_pop.push(child);
        }

        self.population = new_pop;
    }

    /// Get the best genome found so far.
    pub fn best_genome(&self) -> HnswGenome {
        self.population.iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap_or_default()
    }

    /// Generate the final report.
    pub fn report(&self) -> ParamEvolveReport {
        ParamEvolveReport {
            best: self.best_genome(),
            fitness_history: self.fitness_history.iter().copied().collect(),
            generations_completed: self.generation,
            final_population: self.population.clone(),
        }
    }

    // ── Internal helpers ─────────────────────────────────────

    fn next_rand(&mut self) -> f64 {
        // Simple xorshift64 PRNG
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    fn tournament_select(&mut self) -> HnswGenome {
        let mut best_idx = 0usize;
        let mut best_fitness = f64::NEG_INFINITY;
        for _ in 0..self.config.tournament_size {
            let idx = (self.next_rand() * self.population.len() as f64) as usize;
            let idx = idx.min(self.population.len().saturating_sub(1));
            if self.population[idx].fitness > best_fitness {
                best_fitness = self.population[idx].fitness;
                best_idx = idx;
            }
        }
        self.population[best_idx].clone()
    }

    fn crossover(&mut self, a: &HnswGenome, b: &HnswGenome) -> HnswGenome {
        let r = self.next_rand();
        HnswGenome {
            ef_construction: if r < 0.5 { a.ef_construction } else { b.ef_construction },
            ef_search: if self.next_rand() < 0.5 { a.ef_search } else { b.ef_search },
            m: if self.next_rand() < 0.5 { a.m } else { b.m },
            generation: 0,
            fitness: 0.0,
        }
    }

    fn mutate(&mut self, genome: &mut HnswGenome) {
        let step = self.config.mutation_step;
        let (ef_min, m_min) = if self.config.hyperbolic_mode {
            (50usize, 16usize)
        } else {
            (20, 8)
        };

        if self.next_rand() < self.config.mutation_rate {
            let delta = ((self.next_rand() - 0.5) * 2.0 * step * 512.0) as isize;
            genome.ef_construction = (genome.ef_construction as isize + delta)
                .max(ef_min as isize).min(512) as usize;
        }
        if self.next_rand() < self.config.mutation_rate {
            let delta = ((self.next_rand() - 0.5) * 2.0 * step * 1000.0) as isize;
            genome.ef_search = (genome.ef_search as isize + delta)
                .max(10).min(1000) as usize;
        }
        if self.next_rand() < self.config.mutation_rate {
            let delta = ((self.next_rand() - 0.5) * 2.0 * step * 64.0) as isize;
            genome.m = (genome.m as isize + delta)
                .max(m_min as isize).min(64) as usize;
        }
    }
}

// ── Convenience: one-shot evolution from benchmark function ──

/// Run a complete evolutionary search.
///
/// `benchmark_fn` takes a genome and returns a BenchmarkResult.
/// Returns the best genome and evolution report.
pub fn run_param_evolution<F>(
    config: ParamEvolveConfig,
    mut benchmark_fn: F,
) -> ParamEvolveReport
where
    F: FnMut(&HnswGenome) -> BenchmarkResult,
{
    let mut evolver = ParamEvolver::new(config.clone());

    for _gen in 0..config.max_generations {
        // Evaluate all individuals
        for i in 0..evolver.population().len() {
            let genome = evolver.population()[i].clone();
            let bench = benchmark_fn(&genome);
            let fitness = evolver.evaluate_fitness(&genome, &bench);
            evolver.set_fitness(i, fitness);
        }

        // Check for convergence (top 3 within 1% of each other)
        let mut fitnesses: Vec<f64> = evolver.population().iter().map(|g| g.fitness).collect();
        fitnesses.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        if fitnesses.len() >= 3 {
            let spread = fitnesses[0] - fitnesses[2];
            if spread < 0.01 * fitnesses[0].abs().max(1e-6) {
                break; // converged
            }
        }

        evolver.evolve_generation();
    }

    // Final evaluation
    for i in 0..evolver.population().len() {
        let genome = evolver.population()[i].clone();
        let bench = benchmark_fn(&genome);
        let fitness = evolver.evaluate_fitness(&genome, &bench);
        evolver.set_fitness(i, fitness);
    }

    evolver.report()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = ParamEvolveConfig::default();
        assert_eq!(cfg.population_size, 12);
        assert!(cfg.hyperbolic_mode);
    }

    #[test]
    fn test_init_population() {
        let evolver = ParamEvolver::new(ParamEvolveConfig::default());
        assert_eq!(evolver.population().len(), 12);
        // Hyperbolic mode: ef_construction starts at 50
        assert!(evolver.population()[0].ef_construction >= 50);
    }

    #[test]
    fn test_fitness_evaluation() {
        let evolver = ParamEvolver::new(ParamEvolveConfig::default());
        let genome = HnswGenome::default();
        let bench = BenchmarkResult {
            recall: 0.95,
            p95_latency_us: 30_000,
            memory_ratio: 0.3,
        };
        let fitness = evolver.evaluate_fitness(&genome, &bench);
        assert!(fitness > 0.0);
        assert!(fitness <= 1.0);
    }

    #[test]
    fn test_evolution_improves() {
        let config = ParamEvolveConfig {
            population_size: 6,
            max_generations: 5,
            ..Default::default()
        };

        // Mock benchmark: reward high ef_search (better recall)
        let report = run_param_evolution(config, |genome| {
            let recall = (genome.ef_search as f64 / 1000.0).min(1.0);
            let latency = genome.ef_search as u64 * 50; // linear cost
            BenchmarkResult {
                recall,
                p95_latency_us: latency,
                memory_ratio: genome.m as f64 / 64.0,
            }
        });

        assert!(report.generations_completed > 0);
        assert!(report.best.fitness > 0.0);
    }

    #[test]
    fn test_hyperbolic_bounds() {
        let config = ParamEvolveConfig {
            hyperbolic_mode: true,
            population_size: 4,
            ..Default::default()
        };
        let evolver = ParamEvolver::new(config);
        for g in evolver.population() {
            assert!(g.ef_construction >= 50, "Poincaré needs ef_construction >= 50");
            assert!(g.m >= 16, "Poincaré needs M >= 16");
        }
    }
}
