#!/usr/bin/env python3
"""
NietzscheLab — Autonomous Knowledge Evolution Engine.

Main entry point for running epistemic evolution experiments on NietzscheDB.
Implements the autoresearch-style loop: hypothesize → mutate → measure → select.

Usage:
    python lab_runner.py                          # default config
    python lab_runner.py --config config.yaml     # custom config
    python lab_runner.py --collection my_test     # target collection
    python lab_runner.py --max-experiments 50     # run 50 experiments
    python lab_runner.py --random-only            # no LLM, random baseline
    python lab_runner.py --dry-run                # don't apply mutations
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import yaml

from grpc_client import NietzscheClient
from hypothesis_generator import (
    Hypothesis, HypothesisType, Mutation,
    generate_hypothesis_llm, generate_hypothesis_random,
)
from consistency_scorer import ConsistencyScorer, EpistemicMetrics, ScoreWeights
from experiment_journal import ExperimentJournal


def load_config(path: str | None = None) -> dict:
    """Load configuration from YAML file."""
    config_path = Path(path) if path else Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def apply_mutations(client: NietzscheClient, hypothesis: Hypothesis,
                    dry_run: bool = False) -> list[str]:
    """Apply hypothesis mutations to the graph.

    Returns list of created entity IDs (for rollback tracking).
    """
    created_ids = []

    for mutation in hypothesis.mutations:
        if dry_run:
            print(f"  [DRY-RUN] Would apply: {mutation.action} {mutation.params}")
            continue

        action = mutation.action
        params = mutation.params

        if action == "insert_edge":
            eid = client.insert_edge_grpc(
                from_id=params["from_id"],
                to_id=params["to_id"],
                edge_type=params.get("edge_type", "Association"),
                weight=float(params.get("weight", 1.0)),
            )
            if eid:
                created_ids.append(("edge", eid))
                print(f"  [+] Edge created: {eid[:8]}")

        elif action == "insert_node":
            nid = client.insert_node_grpc(
                content=params.get("content", {}),
                node_type=params.get("node_type", "Semantic"),
                energy=float(params.get("energy", 0.5)),
            )
            if nid:
                created_ids.append(("node", nid))
                print(f"  [+] Node created: {nid[:8]}")

        elif action == "delete_edge":
            ok = client.delete_edge_grpc(params["edge_id"])
            if ok:
                print(f"  [-] Edge deleted: {params['edge_id'][:8]}")

        elif action == "delete_node":
            ok = client.delete_node_grpc(params["node_id"])
            if ok:
                print(f"  [-] Node deleted: {params['node_id'][:8]}")

    return created_ids


def rollback_mutations(client: NietzscheClient, created_ids: list[tuple[str, str]]):
    """Rollback created entities (best-effort)."""
    for entity_type, entity_id in reversed(created_ids):
        try:
            if entity_type == "edge":
                client.delete_edge_grpc(entity_id)
                print(f"  [↩] Rolled back edge: {entity_id[:8]}")
            elif entity_type == "node":
                client.delete_node_grpc(entity_id)
                print(f"  [↩] Rolled back node: {entity_id[:8]}")
        except Exception as e:
            print(f"  [!] Rollback failed for {entity_type} {entity_id[:8]}: {e}")


def run_experiment_loop(
    client: NietzscheClient,
    scorer: ConsistencyScorer,
    journal: ExperimentJournal,
    max_experiments: int = 20,
    score_threshold: float = 0.02,
    cooldown: float = 2.0,
    random_only: bool = False,
    dry_run: bool = False,
    llm_model: str = "claude-sonnet-4-20250514",
    llm_temperature: float = 0.7,
    sample_size: int = 50,
    min_energy: float = 0.1,
    allowed_types: list[str] | None = None,
    enable_geometric_uncertainty: bool = False,
):
    """Run the main experiment loop.

    Args:
        enable_geometric_uncertainty: If True, uses GeometricKernels GP-based
            uncertainty estimation to guide hypothesis generation toward
            knowledge gaps. Requires: pip install geometric-kernels scipy
    """
    print("\n" + "=" * 60)
    print("  NietzscheLab — Autonomous Knowledge Evolution")
    print("=" * 60)
    print(f"  Collection:      {client.collection}")
    print(f"  Max experiments: {max_experiments}")
    print(f"  Score threshold: {score_threshold}")
    print(f"  Mode:            {'random' if random_only else 'LLM'}")
    print(f"  Dry run:         {dry_run}")
    print(f"  Geometric uncertainty: {enable_geometric_uncertainty}")
    print("=" * 60 + "\n")

    # Verify connection
    try:
        stats = client.get_stats()
        print(f"[*] Connected to NietzscheDB")
        print(f"    Stats: {stats}\n")
    except Exception as e:
        print(f"[ERROR] Cannot connect to NietzscheDB: {e}")
        return

    accepted_count = 0
    rejected_count = 0

    for i in range(1, max_experiments + 1):
        print(f"\n{'─' * 50}")
        print(f"  Experiment {i}/{max_experiments}")
        print(f"{'─' * 50}")

        try:
            # 1. Sample subgraph
            print("[1/5] Sampling subgraph...")
            nodes, edges = client.sample_subgraph(limit=sample_size, min_energy=min_energy)
            print(f"      {len(nodes)} nodes, {len(edges)} edges")

            if len(nodes) < 2:
                print("[SKIP] Not enough nodes in collection. Aborting.")
                break

            # 1b. GeometricKernels uncertainty analysis (optional)
            knowledge_gaps = None
            if enable_geometric_uncertainty and len(nodes) >= 10:
                try:
                    import numpy as _np
                    from geometric_service.graph_bridge import (
                        ndb_to_geometric_graph, NodeInfo as _NI, EdgeInfo as _EI,
                    )
                    from geometric_service.uncertainty import EpistemicUncertaintyEstimator

                    _gk_nodes = [
                        _NI(n.id, n.energy, getattr(n, "depth", 0.0))
                        for n in nodes
                    ]
                    _gk_edges = [
                        _EI(e.id, e.from_id, e.to_id, getattr(e, "weight", 1.0))
                        for e in edges
                    ]
                    _space, _nids, _idx = ndb_to_geometric_graph(_gk_nodes, _gk_edges)
                    _estimator = EpistemicUncertaintyEstimator(_space, _nids)

                    _obs_idx = _np.array([j for j, n in enumerate(nodes) if n.energy > 0.1])
                    _obs_val = _np.array([nodes[j].energy for j in _obs_idx], dtype=_np.float64)

                    if len(_obs_idx) >= 3:
                        _estimator.fit(_obs_idx, _obs_val)
                        knowledge_gaps = _estimator.find_knowledge_gaps(top_k=5)
                        print(f"      [GK] Found {len(knowledge_gaps)} knowledge gaps:")
                        for _g in knowledge_gaps[:3]:
                            print(f"           - {_g.node_id[:12]}... "
                                  f"uncertainty={_g.uncertainty:.4f} → {_g.suggested_action}")
                except ImportError:
                    print("      [GK] geometric-kernels not installed, skipping uncertainty")
                except Exception as _e:
                    print(f"      [GK] Uncertainty analysis failed: {_e}")

            # 2. Generate hypothesis
            print("[2/5] Generating hypothesis...")
            if random_only:
                hypothesis = generate_hypothesis_random(nodes, edges)
            else:
                hypothesis = generate_hypothesis_llm(
                    nodes, edges,
                    history=journal.recent_history(5),
                    model=llm_model,
                    temperature=llm_temperature,
                    allowed_types=allowed_types,
                )

            if hypothesis is None:
                print("[SKIP] No hypothesis generated.")
                continue

            print(f"      Type: {hypothesis.hypothesis_type.value}")
            print(f"      Desc: {hypothesis.description}")
            print(f"      Mutations: {len(hypothesis.mutations)}")

            # 3. Measure before
            print("[3/5] Measuring metrics (before)...")
            metrics_before = scorer.evaluate(nodes, edges)
            print(f"      Hierarchy: {metrics_before.hierarchy_consistency:.4f}")
            print(f"      Coherence: {metrics_before.coherence:.4f}")
            print(f"      Energy:    {metrics_before.energy_avg:.4f}")

            # 4. Apply mutations
            print("[4/5] Applying mutations...")
            created_ids = apply_mutations(client, hypothesis, dry_run=dry_run)

            # 5. Measure after & decide
            print("[5/5] Measuring metrics (after)...")
            if dry_run:
                metrics_after = metrics_before  # no actual change
            else:
                # Re-sample to get updated state
                nodes_after, edges_after = client.sample_subgraph(
                    limit=sample_size, min_energy=0.0
                )
                metrics_after = scorer.evaluate(nodes_after, edges_after)

            delta = scorer.compute_delta(metrics_before, metrics_after)

            print(f"\n      Delta composite: {delta.composite_score:+.6f}")
            print(f"        Hierarchy:  {delta.hierarchy_delta:+.6f}")
            print(f"        Coherence:  {delta.coherence_delta:+.6f}")
            print(f"        Energy:     {delta.energy_delta:+.6f}")
            print(f"        Hausdorff:  {delta.hausdorff_global_delta:+.6f}")

            accepted = delta.composite_score >= score_threshold or dry_run

            if accepted:
                accepted_count += 1
                print(f"\n  ✓ ACCEPTED (delta={delta.composite_score:+.6f} >= {score_threshold})")
            else:
                rejected_count += 1
                print(f"\n  ✗ REJECTED (delta={delta.composite_score:+.6f} < {score_threshold})")
                if not dry_run:
                    rollback_mutations(client, created_ids)

            # Log to journal
            journal.log(
                hypothesis=hypothesis,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                delta=delta,
                accepted=accepted,
            )

        except KeyboardInterrupt:
            print("\n\n[!] Interrupted by user.")
            break
        except Exception as e:
            print(f"\n[ERROR] Experiment {i} failed: {e}")
            traceback.print_exc()
            # Log error
            if 'hypothesis' in dir() and hypothesis is not None:
                journal.log(
                    hypothesis=hypothesis,
                    metrics_before=metrics_before if 'metrics_before' in dir() else EpistemicMetrics(0, 0, 0, 1.0, 0, 0, 0),
                    metrics_after=metrics_before if 'metrics_before' in dir() else EpistemicMetrics(0, 0, 0, 1.0, 0, 0, 0),
                    delta=delta if 'delta' in dir() else type('D', (), {'hierarchy_delta': 0, 'coherence_delta': 0, 'energy_delta': 0, 'hausdorff_global_delta': 0, 'hausdorff_local_delta': 0, 'composite_score': 0})(),
                    accepted=False,
                    error=str(e),
                )

        if cooldown > 0 and i < max_experiments:
            time.sleep(cooldown)

    # Final summary
    print(f"\n\n{'=' * 60}")
    print(f"  Run complete: {accepted_count} accepted, {rejected_count} rejected")
    print(f"{'=' * 60}")
    journal.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="NietzscheLab — Autonomous Knowledge Evolution Engine"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--host", type=str, default=None,
                        help="NietzscheDB HTTP host (e.g., http://localhost:8080)")
    parser.add_argument("--grpc", type=str, default=None,
                        help="NietzscheDB gRPC target (e.g., localhost:50051)")
    parser.add_argument("--collection", type=str, default=None,
                        help="Target collection name")
    parser.add_argument("--max-experiments", type=int, default=None,
                        help="Maximum number of experiments")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Minimum score delta to accept")
    parser.add_argument("--random-only", action="store_true",
                        help="Use random hypotheses only (no LLM)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't apply mutations (just generate hypotheses)")
    parser.add_argument("--journal", type=str, default=None,
                        help="Path to experiment journal (JSONL)")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    server_cfg = cfg.get("server", {})
    exp_cfg = cfg.get("experiments", {})
    llm_cfg = cfg.get("llm", {})
    hyp_cfg = cfg.get("hypothesis", {})
    score_cfg = cfg.get("scoring", {})

    # Override with CLI args
    http_host = args.host or f"http://{server_cfg.get('host', 'localhost')}:{server_cfg.get('http_port', 8080)}"
    grpc_target = args.grpc or f"{server_cfg.get('host', 'localhost')}:{server_cfg.get('grpc_port', 50051)}"
    collection = args.collection or cfg.get("collection", "lab_experiments")
    max_experiments = args.max_experiments or exp_cfg.get("max_per_run", 20)
    threshold = args.threshold if args.threshold is not None else exp_cfg.get("score_threshold", 0.02)
    journal_path = args.journal or cfg.get("journal", {}).get("path", "./experiments.jsonl")

    # Init components
    client = NietzscheClient(
        http_base=http_host,
        grpc_target=grpc_target,
        collection=collection,
    )

    weights = ScoreWeights(
        hierarchy=score_cfg.get("hierarchy_weight", 0.30),
        coherence=score_cfg.get("coherence_weight", 0.25),
        energy=score_cfg.get("energy_weight", 0.20),
        hausdorff=score_cfg.get("hausdorff_weight", 0.15),
        novelty=score_cfg.get("novelty_weight", 0.10),
    )

    scorer = ConsistencyScorer(client, weights)
    journal = ExperimentJournal(journal_path)

    # Run
    run_experiment_loop(
        client=client,
        scorer=scorer,
        journal=journal,
        max_experiments=max_experiments,
        score_threshold=threshold,
        cooldown=exp_cfg.get("cooldown_seconds", 2.0),
        random_only=args.random_only,
        dry_run=args.dry_run,
        llm_model=llm_cfg.get("model", "claude-sonnet-4-20250514"),
        llm_temperature=llm_cfg.get("temperature", 0.7),
        sample_size=hyp_cfg.get("subgraph_sample_size", 50),
        min_energy=hyp_cfg.get("min_energy", 0.1),
        allowed_types=hyp_cfg.get("types"),
    )


if __name__ == "__main__":
    main()
