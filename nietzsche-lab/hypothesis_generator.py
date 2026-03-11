"""
NietzscheLab — Hypothesis Generator.

Uses an LLM (Claude) to propose epistemic mutations on a knowledge graph.
Each hypothesis is a structured mutation proposal with type, affected nodes,
and justification.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from grpc_client import NodeInfo, EdgeInfo


class HypothesisType(str, Enum):
    NEW_EDGE = "NEW_EDGE"           # connect two existing nodes
    NEW_CONCEPT = "NEW_CONCEPT"     # create a unifying concept node
    REMOVE_EDGE = "REMOVE_EDGE"     # remove an inconsistent edge
    RECLASSIFY = "RECLASSIFY"       # move node to correct depth level
    MERGE_NODES = "MERGE_NODES"     # merge duplicate nodes
    SPLIT_NODE = "SPLIT_NODE"       # split an overloaded node


@dataclass
class Mutation:
    """A concrete graph mutation to apply."""
    action: str  # "insert_edge", "insert_node", "delete_edge", "delete_node", "update_energy"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """A structured hypothesis with type, mutations, and justification."""
    hypothesis_type: HypothesisType
    description: str
    justification: str
    mutations: list[Mutation]
    affected_node_ids: list[str]
    predicted_improvement: str = ""


RESEARCHER = "web2ajax@gmail.com"

SYSTEM_PROMPT = """\
You are NietzscheLab, an autonomous epistemic evolution engine for NietzscheDB — \
a multi-manifold hyperbolic graph database. Researcher: web2ajax@gmail.com.

Your task: Given a subgraph of a knowledge graph, propose ONE hypothesis that would \
improve the epistemic quality of the graph.

The graph uses Poincaré ball geometry:
- Node depth (0.0-1.0) = abstraction level. Low depth = abstract/general, high depth = specific.
- Nodes have energy (0.0-1.0) = vitality. Low energy nodes are decaying.
- Edges connect related concepts with weights.
- Hausdorff dimension measures local fractal complexity (ideal: 0.5-1.9).

Types of hypotheses you can propose:
1. NEW_EDGE: Two nodes should be connected but aren't.
2. NEW_CONCEPT: A cluster of nodes would benefit from a unifying concept node.
3. REMOVE_EDGE: An edge is inconsistent or redundant.
4. RECLASSIFY: A node is at the wrong depth level.

Rules:
- Be specific: reference actual node IDs from the subgraph.
- Justify WHY the mutation improves epistemic coherence.
- Only propose ONE hypothesis per call.
- Output valid JSON only.

Output format:
{
  "hypothesis_type": "NEW_EDGE|NEW_CONCEPT|REMOVE_EDGE|RECLASSIFY",
  "description": "Short description of the hypothesis",
  "justification": "Why this improves the graph",
  "mutations": [
    {"action": "insert_edge", "params": {"from_id": "...", "to_id": "...", "edge_type": "Association", "weight": 1.0}},
    {"action": "insert_node", "params": {"content": {...}, "node_type": "Concept", "energy": 0.7}},
    {"action": "delete_edge", "params": {"edge_id": "..."}},
    {"action": "update_energy", "params": {"node_id": "...", "energy": 0.5}}
  ],
  "affected_node_ids": ["uuid1", "uuid2"],
  "predicted_improvement": "Expected effect on graph coherence"
}
"""


def _format_subgraph_context(nodes: list[NodeInfo], edges: list[EdgeInfo],
                              history: list[dict] | None = None) -> str:
    """Format a subgraph into a context string for the LLM."""
    lines = ["## Current Subgraph\n"]
    lines.append(f"Nodes ({len(nodes)}):\n")
    for n in nodes[:60]:  # limit to avoid token overflow
        content_str = json.dumps(n.content, ensure_ascii=False)
        if len(content_str) > 200:
            content_str = content_str[:200] + "..."
        lines.append(
            f"- [{n.id[:8]}] type={n.node_type} depth={n.depth:.3f} "
            f"energy={n.energy:.3f} hausdorff={n.hausdorff_local:.3f} "
            f"content={content_str}"
        )

    lines.append(f"\nEdges ({len(edges)}):\n")
    for e in edges[:80]:
        lines.append(
            f"- [{e.id[:8]}] {e.from_id[:8]}→{e.to_id[:8]} "
            f"type={e.edge_type} weight={e.weight:.3f}"
        )

    if history:
        lines.append(f"\n## Recent Experiment History (last {min(len(history), 5)}):\n")
        for h in history[-5:]:
            status = "ACCEPTED" if h.get("accepted") else "REJECTED"
            lines.append(
                f"- [{status}] {h.get('hypothesis_type', '?')}: "
                f"{h.get('description', '?')} (delta={h.get('delta_score', 0):.4f})"
            )

    return "\n".join(lines)


def generate_hypothesis_llm(
    nodes: list[NodeInfo],
    edges: list[EdgeInfo],
    history: list[dict] | None = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    allowed_types: list[str] | None = None,
) -> Hypothesis | None:
    """Generate a hypothesis using Claude API.

    Returns a Hypothesis object or None if generation fails.
    """
    try:
        import anthropic
    except ImportError:
        print("[NietzscheLab] anthropic package not installed. Using random fallback.")
        return generate_hypothesis_random(nodes, edges)

    client = anthropic.Anthropic()

    context = _format_subgraph_context(nodes, edges, history)

    type_hint = ""
    if allowed_types:
        type_hint = f"\n\nFocus on these hypothesis types: {', '.join(allowed_types)}"

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"{context}{type_hint}\n\nPropose ONE hypothesis to improve this graph."
            }],
        )

        text = response.content[0].text.strip()

        # Extract JSON from response (handle markdown code blocks)
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
        print(f"[NietzscheLab] LLM hypothesis generation failed: {e}")
        return generate_hypothesis_random(nodes, edges)


def generate_hypothesis_random(
    nodes: list[NodeInfo],
    edges: list[EdgeInfo],
) -> Hypothesis | None:
    """Generate a random hypothesis (baseline / fallback).

    Used when LLM is unavailable or for baseline comparison.
    """
    if len(nodes) < 2:
        return None

    # Build adjacency set for quick lookup
    connected = set()
    for e in edges:
        connected.add((e.from_id, e.to_id))
        connected.add((e.to_id, e.from_id))

    # Find unconnected pairs
    unconnected = []
    node_list = list(nodes)
    for i in range(min(len(node_list), 30)):
        for j in range(i + 1, min(len(node_list), 30)):
            a, b = node_list[i], node_list[j]
            if (a.id, b.id) not in connected:
                unconnected.append((a, b))

    if not unconnected:
        return None

    # Pick a random unconnected pair
    a, b = random.choice(unconnected)

    return Hypothesis(
        hypothesis_type=HypothesisType.NEW_EDGE,
        description=f"Connect {a.id[:8]} to {b.id[:8]}",
        justification="Random baseline hypothesis: connect unlinked nodes",
        mutations=[Mutation(
            action="insert_edge",
            params={"from_id": a.id, "to_id": b.id, "edge_type": "Association", "weight": 0.5},
        )],
        affected_node_ids=[a.id, b.id],
        predicted_improvement="Random connection for baseline testing",
    )
