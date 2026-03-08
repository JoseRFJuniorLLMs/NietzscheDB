#!/usr/bin/env python3
"""
export_edge_predictor.py — Link Prediction Network for NietzscheDB.

Predicts whether an edge should exist between two nodes based on their
embeddings. Used by the L-System to decide where to create new connections.

Architecture:
    Input:  [B, 256]  (128D node_a ⊕ 128D node_b concatenation)
    Hidden: 256 → 128 → 64
    Output: [B, 1]    (sigmoid: probability of edge)
"""

import torch
import torch.nn as nn
import os


class EdgePredictor(nn.Module):
    """Link prediction: two node embeddings → edge probability."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        input_dim = embed_dim * 2  # concatenation of two nodes

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 256] (node_a ⊕ node_b) → [B, 1] edge probability"""
        return self.net(x)


def export_onnx(output_dir: str = "../../models"):
    os.makedirs(output_dir, exist_ok=True)
    model = EdgePredictor()
    model.eval()

    dummy = torch.randn(1, 256)
    path = os.path.join(output_dir, "edge_predictor.onnx")

    torch.onnx.export(
        model, dummy, path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["node_pair"],
        output_names=["edge_prob"],
        dynamic_axes={"node_pair": {0: "batch"}, "edge_prob": {0: "batch"}},
    )
    print(f"[EdgePredictor] Exported → {path}")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(path))
        print("[EdgePredictor] ONNX validation passed.")
    except ImportError:
        pass


if __name__ == "__main__":
    export_onnx()
