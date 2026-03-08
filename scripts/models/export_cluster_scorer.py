#!/usr/bin/env python3
"""
export_cluster_scorer.py — Cluster Quality Scorer for NietzscheDB.

Evaluates cluster health and recommends split/merge/keep actions.

Architecture:
    Input:  [B, 261]  (128D centroid + 128D variance + 5 scalar stats)
        Stats: [cluster_size, density, avg_edge_weight, diameter, coherence]
    Hidden: 261 → 128 → 64
    Output: [B, 3]   (softmax: [keep, split, merge])
"""

import torch
import torch.nn as nn
import os


class ClusterScorer(nn.Module):
    """MLP that scores cluster quality and recommends actions."""

    def __init__(self, centroid_dim: int = 128):
        super().__init__()
        # centroid (128) + variance (128) + 5 scalar stats
        input_dim = centroid_dim * 2 + 5

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),

            nn.Linear(64, 3),  # keep, split, merge
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 261] → [B, 3] (action probabilities)"""
        return torch.softmax(self.net(x), dim=-1)


def export_onnx(output_dir: str = "../../models"):
    os.makedirs(output_dir, exist_ok=True)
    model = ClusterScorer()
    model.eval()

    dummy = torch.randn(1, 261)
    path = os.path.join(output_dir, "cluster_scorer.onnx")

    torch.onnx.export(
        model, dummy, path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["cluster_features"],
        output_names=["action_probs"],
        dynamic_axes={"cluster_features": {0: "batch"}, "action_probs": {0: "batch"}},
    )
    print(f"[ClusterScorer] Exported → {path}")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(path))
        print("[ClusterScorer] ONNX validation passed.")
    except ImportError:
        pass


if __name__ == "__main__":
    export_onnx()
