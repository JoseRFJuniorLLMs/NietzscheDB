"""
NietzscheEvolve — Phase 3: Hypothesis Prompt Evolution.

Inspired by AlphaEvolve's approach of evolving the code that generates solutions,
this module evolves the **system prompt** used by the hypothesis generator.

Instead of hardcoding the researcher persona and instructions, we maintain a
population of prompts and evolve them based on how well their generated
hypotheses perform (measured by epistemic quality delta).

The evolved prompts are stored as nodes in NietzscheDB itself — the system
literally evolves inside its own substrate.

Usage:
    python prompt_evolve.py \\
        --collection my_test \\
        --generations 5 \\
        --population 4 \\
        --experiments-per-prompt 3
"""

from __future__ import annotations

import json
import time
import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

from grpc_client import NietzscheClient
from hypothesis_generator import (
    Hypothesis, HypothesisType, Mutation, SYSTEM_PROMPT,
    _format_subgraph_context, generate_hypothesis_random,
)
from consistency_scorer import ConsistencyScorer, ScoreWeights
from lab_runner import apply_mutations, rollback_mutations

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("prompt_evolve")


# ── Prompt Genome ─────────────────────────────────────────────


@dataclass
class PromptGenome:
    """A single prompt individual in the population."""
    system_prompt: str
    generation: int = 0
    fitness: float = 0.0
    experiments_run: int = 0
    accepted_count: int = 0
    total_delta: float = 0.0
    id: str = ""  # NietzscheDB node ID if persisted


@dataclass
class PromptEvolveConfig:
    """Configuration for prompt evolution."""
    population_size: int = 4
    max_generations: int = 5
    experiments_per_prompt: int = 3
    score_threshold: float = 0.02
    llm_model: str = "claude-sonnet-4-20250514"
    mutation_temperature: float = 0.9
    cooldown: float = 2.0  # seconds between experiments


@dataclass
class PromptEvolveReport:
    """Report from a prompt evolution run."""
    best_prompt: str
    best_fitness: float
    generations_completed: int
    fitness_history: list[float] = field(default_factory=list)
    population_summary: list[dict[str, Any]] = field(default_factory=list)


# ── Core Evolution Loop ──────────────────────────────────────


def generate_hypothesis_with_prompt(
    system_prompt: str,
    nodes, edges,
    history=None,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.7,
) -> Hypothesis | None:
    """Generate a hypothesis using a custom system prompt."""
    try:
        import anthropic
    except ImportError:
        return generate_hypothesis_random(nodes, edges)

    client = anthropic.Anthropic()
    context = _format_subgraph_context(nodes, edges, history)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=temperature,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": f"{context}\n\nPropose ONE hypothesis to improve this graph."
            }],
        )

        text = response.content[0].text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)
        return Hypothesis(
            hypothesis_type=HypothesisType(data["hypothesis_type"]),
            description=data["description"],
            justification=data["justification"],
            mutations=[
                Mutation(action=m["action"], params=m.get("params", {}))
                for m in data.get("mutations", [])
            ],
            affected_node_ids=data.get("affected_node_ids", []),
            predicted_improvement=data.get("predicted_improvement", ""),
        )
    except Exception as e:
        log.warning("Hypothesis generation failed: %s", e)
        return None


def evaluate_prompt(
    genome: PromptGenome,
    client: NietzscheClient,
    scorer: ConsistencyScorer,
    config: PromptEvolveConfig,
    collection: str,
) -> float:
    """Evaluate a prompt genome by running experiments and measuring deltas."""
    total_delta = 0.0
    accepted = 0

    for exp in range(config.experiments_per_prompt):
        nodes, edges = client.sample_subgraph(limit=50, min_energy=0.1)
        if len(nodes) < 5:
            continue

        # Measure before
        metrics_before = scorer.evaluate(nodes, edges)

        # Generate hypothesis with this prompt
        hypothesis = generate_hypothesis_with_prompt(
            system_prompt=genome.system_prompt,
            nodes=nodes,
            edges=edges,
            model=config.llm_model,
        )

        if hypothesis is None:
            continue

        # Apply mutations
        created_ids = apply_mutations(client, hypothesis.mutations, collection)
        if not created_ids:
            continue

        # Measure after
        nodes_after, edges_after = client.sample_subgraph(limit=50, min_energy=0.1)
        metrics_after = scorer.evaluate(nodes_after, edges_after)
        delta = scorer.compute_delta(metrics_before, metrics_after)

        if delta.composite_score >= config.score_threshold:
            accepted += 1
            total_delta += delta.composite_score
            log.info(
                "  [gen=%d exp=%d] ACCEPTED delta=%.4f (%s)",
                genome.generation, exp, delta.composite_score, hypothesis.description[:60],
            )
        else:
            rollback_mutations(client, created_ids, collection)
            log.info(
                "  [gen=%d exp=%d] REJECTED delta=%.4f",
                genome.generation, exp, delta.composite_score,
            )

        time.sleep(config.cooldown)

    # Fitness = acceptance rate * mean delta
    experiments_run = config.experiments_per_prompt
    genome.experiments_run += experiments_run
    genome.accepted_count += accepted
    genome.total_delta += total_delta

    if accepted > 0:
        fitness = (accepted / experiments_run) * (total_delta / accepted)
    else:
        fitness = 0.0

    genome.fitness = fitness
    return fitness


def mutate_prompt(
    parent_prompt: str,
    accepted_history: list[str],
    rejected_history: list[str],
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.9,
) -> str:
    """Use an LLM to mutate a system prompt based on experiment history."""
    try:
        import anthropic
    except ImportError:
        return parent_prompt  # can't mutate without LLM

    client = anthropic.Anthropic()

    mutation_instructions = f"""\
You are a meta-researcher evolving system prompts for a graph mutation engine.

Given the parent system prompt and experiment history, create an IMPROVED version
of the prompt that will generate better graph hypotheses.

## Parent Prompt:
{parent_prompt}

## Accepted Hypotheses (these worked):
{chr(10).join(accepted_history[-5:]) if accepted_history else "(none yet)"}

## Rejected Hypotheses (these failed):
{chr(10).join(rejected_history[-5:]) if rejected_history else "(none yet)"}

## Instructions:
1. Keep the core structure (JSON output format, hypothesis types)
2. Improve the guidance based on what worked/failed
3. Add or refine heuristics for better graph analysis
4. The output should be ONLY the new system prompt text (no meta-commentary)
5. Keep it under 1500 characters
"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=temperature,
            messages=[{"role": "user", "content": mutation_instructions}],
        )
        new_prompt = response.content[0].text.strip()
        # Basic validation
        if len(new_prompt) < 100 or "hypothesis" not in new_prompt.lower():
            return parent_prompt
        return new_prompt
    except Exception as e:
        log.warning("Prompt mutation failed: %s", e)
        return parent_prompt


def run_prompt_evolution(
    client: NietzscheClient,
    collection: str,
    config: PromptEvolveConfig | None = None,
) -> PromptEvolveReport:
    """Run the full prompt evolution loop."""
    if config is None:
        config = PromptEvolveConfig()

    scorer = ConsistencyScorer(ScoreWeights())

    # Initialize population with variations of the base prompt
    population: list[PromptGenome] = []
    population.append(PromptGenome(system_prompt=SYSTEM_PROMPT, generation=0))

    # Create variations by adjusting focus areas
    focus_areas = [
        "\n\nPrioritize NEW_EDGE hypotheses that improve coherence between clusters.",
        "\n\nFocus on RECLASSIFY hypotheses to fix hierarchy violations.",
        "\n\nLook for NEW_CONCEPT opportunities where clusters lack unifying nodes.",
    ]
    for i, suffix in enumerate(focus_areas):
        if len(population) >= config.population_size:
            break
        population.append(PromptGenome(
            system_prompt=SYSTEM_PROMPT + suffix,
            generation=0,
        ))

    # Fill remaining slots with the base prompt
    while len(population) < config.population_size:
        population.append(PromptGenome(system_prompt=SYSTEM_PROMPT, generation=0))

    fitness_history = []
    accepted_history: list[str] = []
    rejected_history: list[str] = []

    for gen in range(config.max_generations):
        log.info("=== Generation %d ===", gen)

        # Evaluate each prompt
        for genome in population:
            genome.generation = gen
            evaluate_prompt(genome, client, scorer, config, collection)
            log.info(
                "  Prompt (%.30s...) fitness=%.4f accepted=%d/%d",
                genome.system_prompt[:30], genome.fitness,
                genome.accepted_count, genome.experiments_run,
            )

        # Record best fitness
        best = max(population, key=lambda g: g.fitness)
        fitness_history.append(best.fitness)
        log.info("Best fitness gen %d: %.4f", gen, best.fitness)

        # Selection + mutation for next generation
        if gen < config.max_generations - 1:
            # Sort by fitness descending
            population.sort(key=lambda g: g.fitness, reverse=True)

            # Keep top half
            survivors = population[:max(1, len(population) // 2)]

            # Mutate survivors to create children
            new_pop = []
            for survivor in survivors:
                new_pop.append(PromptGenome(
                    system_prompt=survivor.system_prompt,
                    generation=gen + 1,
                ))
                # Create mutated child
                if len(new_pop) < config.population_size:
                    mutated = mutate_prompt(
                        survivor.system_prompt,
                        accepted_history,
                        rejected_history,
                        model=config.llm_model,
                        temperature=config.mutation_temperature,
                    )
                    new_pop.append(PromptGenome(
                        system_prompt=mutated,
                        generation=gen + 1,
                    ))

            # Fill remaining
            while len(new_pop) < config.population_size:
                new_pop.append(PromptGenome(
                    system_prompt=SYSTEM_PROMPT,
                    generation=gen + 1,
                ))

            population = new_pop

    # Final report
    best = max(population, key=lambda g: g.fitness)
    return PromptEvolveReport(
        best_prompt=best.system_prompt,
        best_fitness=best.fitness,
        generations_completed=config.max_generations,
        fitness_history=fitness_history,
        population_summary=[
            {
                "prompt_preview": g.system_prompt[:100],
                "fitness": g.fitness,
                "accepted": g.accepted_count,
                "experiments": g.experiments_run,
            }
            for g in population
        ],
    )


# ── CLI ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="NietzscheEvolve — Prompt Evolution")
    parser.add_argument("--grpc", default="136.111.0.47:443", help="gRPC endpoint")
    parser.add_argument("--http", default="http://136.111.0.47:8080", help="HTTP endpoint")
    parser.add_argument("--collection", default="tech_galaxies", help="Collection to evolve on")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations")
    parser.add_argument("--population", type=int, default=4, help="Population size")
    parser.add_argument("--experiments-per-prompt", type=int, default=3, help="Experiments per prompt")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="LLM model")
    parser.add_argument("--threshold", type=float, default=0.02, help="Score threshold")
    parser.add_argument("--output", default=None, help="Save report to file")
    args = parser.parse_args()

    client = NietzscheClient(
        grpc_host=args.grpc,
        http_base=args.http,
        collection=args.collection,
    )

    config = PromptEvolveConfig(
        population_size=args.population,
        max_generations=args.generations,
        experiments_per_prompt=args.experiments_per_prompt,
        score_threshold=args.threshold,
        llm_model=args.model,
    )

    log.info("Starting prompt evolution on collection '%s'", args.collection)
    log.info("Config: %s generations, %s population, %s experiments/prompt",
             config.max_generations, config.population_size, config.experiments_per_prompt)

    report = run_prompt_evolution(client, args.collection, config)

    log.info("\n=== EVOLUTION COMPLETE ===")
    log.info("Best fitness: %.4f", report.best_fitness)
    log.info("Generations: %d", report.generations_completed)
    log.info("Fitness history: %s", [f"{f:.4f}" for f in report.fitness_history])
    log.info("Best prompt preview: %.200s...", report.best_prompt[:200])

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "best_prompt": report.best_prompt,
                "best_fitness": report.best_fitness,
                "fitness_history": report.fitness_history,
                "population": report.population_summary,
            }, f, indent=2)
        log.info("Report saved to %s", args.output)


if __name__ == "__main__":
    main()
