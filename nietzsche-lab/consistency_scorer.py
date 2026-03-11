"""
NietzscheLab — Consistency Scorer.

Measures epistemic quality of a subgraph before and after a mutation.
Uses metrics available from NietzscheDB's HTTP and gRPC API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from grpc_client import NietzscheClient, NodeInfo, EdgeInfo


@dataclass
class EpistemicMetrics:
    """Epistemic quality metrics for a subgraph."""
    hierarchy_consistency: float  # % of edges where parent.depth < child.depth
    coherence: float              # mean inverse distance between connected nodes
    energy_avg: float             # average energy of affected nodes
    hausdorff_global: float       # global Hausdorff dimension
    hausdorff_local_avg: float    # average local Hausdorff of affected nodes
    node_count: int               # number of nodes measured
    edge_count: int               # number of edges measured


@dataclass
class EpistemicDelta:
    """Change in epistemic quality after a mutation."""
    hierarchy_delta: float
    coherence_delta: float
    energy_delta: float
    hausdorff_global_delta: float
    hausdorff_local_delta: float
    composite_score: float  # weighted sum of all deltas

    @property
    def accepted(self) -> bool:
        return self.composite_score > 0


@dataclass
class ScoreWeights:
    """Weights for computing composite score."""
    hierarchy: float = 0.30
    coherence: float = 0.25
    energy: float = 0.20
    hausdorff: float = 0.15
    novelty: float = 0.10


class ConsistencyScorer:
    """Computes epistemic quality metrics for a NietzscheDB subgraph."""

    def __init__(self, client: NietzscheClient, weights: ScoreWeights | None = None):
        self.client = client
        self.weights = weights or ScoreWeights()

    def evaluate(self, nodes: list[NodeInfo] | None = None,
                 edges: list[EdgeInfo] | None = None) -> EpistemicMetrics:
        """Evaluate epistemic quality of the current subgraph state."""
        if nodes is None or edges is None:
            sampled_nodes, sampled_edges = self.client.sample_subgraph(limit=200)
            nodes = nodes or sampled_nodes
            edges = edges or sampled_edges

        # 1. Hierarchy consistency
        hierarchy = self._compute_hierarchy(nodes, edges)

        # 2. Coherence (connected nodes should have proportional depth differences)
        coherence = self._compute_coherence(nodes, edges)

        # 3. Energy average
        energy_avg = sum(n.energy for n in nodes) / max(len(nodes), 1)

        # 4. Hausdorff (from dashboard if available)
        hausdorff_global = self._get_global_hausdorff()
        hausdorff_local_avg = (
            sum(n.hausdorff_local for n in nodes) / max(len(nodes), 1)
        )

        return EpistemicMetrics(
            hierarchy_consistency=hierarchy,
            coherence=coherence,
            energy_avg=energy_avg,
            hausdorff_global=hausdorff_global,
            hausdorff_local_avg=hausdorff_local_avg,
            node_count=len(nodes),
            edge_count=len(edges),
        )

    def compute_delta(self, before: EpistemicMetrics,
                      after: EpistemicMetrics) -> EpistemicDelta:
        """Compute the epistemic delta between two measurements."""
        h_delta = after.hierarchy_consistency - before.hierarchy_consistency
        c_delta = after.coherence - before.coherence
        e_delta = after.energy_avg - before.energy_avg

        # For Hausdorff, being closer to the ideal range [0.5, 1.9] is better
        h_before_dist = self._hausdorff_distance_to_ideal(before.hausdorff_global)
        h_after_dist = self._hausdorff_distance_to_ideal(after.hausdorff_global)
        hg_delta = h_before_dist - h_after_dist  # positive = improvement

        hl_before_dist = self._hausdorff_distance_to_ideal(before.hausdorff_local_avg)
        hl_after_dist = self._hausdorff_distance_to_ideal(after.hausdorff_local_avg)
        hl_delta = hl_before_dist - hl_after_dist

        composite = (
            self.weights.hierarchy * h_delta +
            self.weights.coherence * c_delta +
            self.weights.energy * e_delta +
            self.weights.hausdorff * (hg_delta + hl_delta) / 2 +
            self.weights.novelty * 0.01  # small novelty bonus for any mutation
        )

        return EpistemicDelta(
            hierarchy_delta=h_delta,
            coherence_delta=c_delta,
            energy_delta=e_delta,
            hausdorff_global_delta=hg_delta,
            hausdorff_local_delta=hl_delta,
            composite_score=composite,
        )

    def _compute_hierarchy(self, nodes: list[NodeInfo],
                           edges: list[EdgeInfo]) -> float:
        """Compute hierarchy consistency.

        For each directed edge A→B, checks if A.depth <= B.depth
        (parent/abstract → child/specific). Returns fraction of correct edges.
        """
        if not edges:
            return 1.0

        node_map = {n.id: n for n in nodes}
        correct = 0
        total = 0

        for e in edges:
            src = node_map.get(e.from_id)
            dst = node_map.get(e.to_id)
            if src is not None and dst is not None:
                total += 1
                if src.depth <= dst.depth + 0.05:  # small tolerance
                    correct += 1

        return correct / max(total, 1)

    def _compute_coherence(self, nodes: list[NodeInfo],
                           edges: list[EdgeInfo]) -> float:
        """Compute semantic coherence.

        Measures whether connected nodes have appropriate depth relationships.
        Higher coherence = edges connect nodes at similar or hierarchically
        related depth levels.
        """
        if not edges:
            return 1.0

        node_map = {n.id: n for n in nodes}
        scores = []

        for e in edges:
            src = node_map.get(e.from_id)
            dst = node_map.get(e.to_id)
            if src is not None and dst is not None:
                depth_diff = abs(src.depth - dst.depth)
                # Coherence: small depth differences for association edges
                # are more coherent. Large differences OK for hierarchical edges.
                if e.edge_type in ("Hierarchical", "LSystemGenerated"):
                    # Hierarchical: depth difference should exist
                    score = min(depth_diff * 2, 1.0)
                else:
                    # Association: similar depth = more coherent
                    score = max(1.0 - depth_diff * 2, 0.0)
                scores.append(score)

        return sum(scores) / max(len(scores), 1)

    def _get_global_hausdorff(self) -> float:
        """Get global Hausdorff dimension from the agency dashboard."""
        try:
            dashboard = self.client.get_agency_dashboard()
            # Try various paths in the dashboard response
            if isinstance(dashboard, dict):
                # Health report path
                health = dashboard.get("health", {})
                if isinstance(health, dict):
                    h = health.get("hausdorff_global", health.get("hausdorff", 0))
                    if h:
                        return float(h)
                # Direct path
                h = dashboard.get("hausdorff_global", 0)
                if h:
                    return float(h)
        except Exception:
            pass
        return 1.0  # safe default in ideal range

    @staticmethod
    def _hausdorff_distance_to_ideal(h: float) -> float:
        """Distance from a Hausdorff value to the ideal range [0.5, 1.9]."""
        if 0.5 <= h <= 1.9:
            return 0.0
        if h < 0.5:
            return 0.5 - h
        return h - 1.9
