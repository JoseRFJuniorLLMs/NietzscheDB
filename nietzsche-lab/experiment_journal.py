"""
NietzscheLab — Experiment Journal.

Records all hypothesis experiments with before/after metrics
in a structured JSONL file for analysis and LLM context.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from consistency_scorer import EpistemicMetrics, EpistemicDelta
from hypothesis_generator import Hypothesis, HypothesisType


class ExperimentJournal:
    """Append-only experiment journal stored as JSONL."""

    def __init__(self, path: str = "./experiments.jsonl"):
        self.path = Path(path)
        self._entries: list[dict] = []
        self._load_existing()

    def _load_existing(self):
        """Load existing entries from the JSONL file."""
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

    @property
    def experiment_count(self) -> int:
        return len(self._entries)

    @property
    def accepted_count(self) -> int:
        return sum(1 for e in self._entries if e.get("accepted", False))

    @property
    def rejected_count(self) -> int:
        return sum(1 for e in self._entries if not e.get("accepted", False))

    @property
    def acceptance_rate(self) -> float:
        if not self._entries:
            return 0.0
        return self.accepted_count / len(self._entries)

    def log(self, hypothesis: Hypothesis,
            metrics_before: EpistemicMetrics,
            metrics_after: EpistemicMetrics,
            delta: EpistemicDelta,
            accepted: bool,
            error: str | None = None) -> dict:
        """Log an experiment result."""
        entry = {
            "id": self.experiment_count + 1,
            "researcher": "web2ajax@gmail.com",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hypothesis_type": hypothesis.hypothesis_type.value,
            "description": hypothesis.description,
            "justification": hypothesis.justification,
            "predicted_improvement": hypothesis.predicted_improvement,
            "affected_node_ids": hypothesis.affected_node_ids,
            "mutations": [{"action": m.action, "params": m.params}
                          for m in hypothesis.mutations],
            "metrics_before": {
                "hierarchy_consistency": round(metrics_before.hierarchy_consistency, 4),
                "coherence": round(metrics_before.coherence, 4),
                "energy_avg": round(metrics_before.energy_avg, 4),
                "hausdorff_global": round(metrics_before.hausdorff_global, 4),
                "hausdorff_local_avg": round(metrics_before.hausdorff_local_avg, 4),
                "node_count": metrics_before.node_count,
                "edge_count": metrics_before.edge_count,
            },
            "metrics_after": {
                "hierarchy_consistency": round(metrics_after.hierarchy_consistency, 4),
                "coherence": round(metrics_after.coherence, 4),
                "energy_avg": round(metrics_after.energy_avg, 4),
                "hausdorff_global": round(metrics_after.hausdorff_global, 4),
                "hausdorff_local_avg": round(metrics_after.hausdorff_local_avg, 4),
                "node_count": metrics_after.node_count,
                "edge_count": metrics_after.edge_count,
            },
            "delta": {
                "hierarchy": round(delta.hierarchy_delta, 6),
                "coherence": round(delta.coherence_delta, 6),
                "energy": round(delta.energy_delta, 6),
                "hausdorff_global": round(delta.hausdorff_global_delta, 6),
                "hausdorff_local": round(delta.hausdorff_local_delta, 6),
                "composite_score": round(delta.composite_score, 6),
            },
            "accepted": accepted,
            "error": error,
        }

        self._entries.append(entry)

        # Append to file
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return entry

    def recent_history(self, n: int = 10) -> list[dict]:
        """Get the N most recent experiments."""
        return self._entries[-n:]

    def summary(self) -> dict:
        """Generate a summary of all experiments."""
        if not self._entries:
            return {"total": 0}

        accepted = [e for e in self._entries if e.get("accepted")]
        rejected = [e for e in self._entries if not e.get("accepted")]

        avg_delta = 0.0
        if accepted:
            avg_delta = sum(
                e.get("delta", {}).get("composite_score", 0) for e in accepted
            ) / len(accepted)

        type_counts: dict[str, int] = {}
        type_accepted: dict[str, int] = {}
        for e in self._entries:
            ht = e.get("hypothesis_type", "UNKNOWN")
            type_counts[ht] = type_counts.get(ht, 0) + 1
            if e.get("accepted"):
                type_accepted[ht] = type_accepted.get(ht, 0) + 1

        return {
            "total": len(self._entries),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "acceptance_rate": round(self.acceptance_rate, 4),
            "avg_accepted_delta": round(avg_delta, 6),
            "by_type": {
                t: {
                    "total": type_counts[t],
                    "accepted": type_accepted.get(t, 0),
                    "rate": round(type_accepted.get(t, 0) / type_counts[t], 4),
                }
                for t in type_counts
            },
        }

    def print_summary(self):
        """Print a formatted summary."""
        s = self.summary()
        print("\n" + "=" * 60)
        print("  NietzscheLab — Experiment Summary")
        print("=" * 60)
        print(f"  Total experiments:  {s['total']}")
        print(f"  Accepted:           {s.get('accepted', 0)}")
        print(f"  Rejected:           {s.get('rejected', 0)}")
        print(f"  Acceptance rate:    {s.get('acceptance_rate', 0):.1%}")
        print(f"  Avg accepted delta: {s.get('avg_accepted_delta', 0):.6f}")
        if "by_type" in s:
            print("\n  By hypothesis type:")
            for t, info in s["by_type"].items():
                print(f"    {t}: {info['accepted']}/{info['total']} ({info['rate']:.0%})")
        print("=" * 60 + "\n")
