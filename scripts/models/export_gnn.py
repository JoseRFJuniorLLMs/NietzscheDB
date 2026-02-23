#!/usr/bin/env python3
"""
export_gnn.py — Export a Graph Neural Network (GNN) as an ONNX model.

Architecture: 2-layer MLP that simulates a GNN message-passing step.
  - Input:  node features [batch, node_dim=3072] (Poincaré ball embeddings)
  - Output: refined embeddings [batch, node_dim=3072] + importance score [batch, 1]

Usage:
    pip install torch onnx numpy
    python export_gnn.py --output ../../models/gnn_diffusion.onnx
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import os


class GnnDiffusionModel(nn.Module):
    """
    Simulates a GNN diffusion step over a Poincaré embedding space.

    In production this would be trained on NietzscheDB graph traversal data.
    For the MVP we export a learnable 2-layer MLP that can be fine-tuned later.
    """

    def __init__(self, node_dim: int = 3072, hidden_dim: int = 512):
        super().__init__()

        # Message-passing layer (neighbour aggregation simulation)
        self.message_fc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Update layer (refine node embedding)
        self.update_fc = nn.Sequential(
            nn.Linear(hidden_dim, node_dim),
            nn.Tanh(),   # Keep embeddings bounded (Poincaré ball has norm < 1.0)
        )

        # Importance scorer (used by NeuralThresholdDaemon to protect nodes)
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output: 0.0 (prunable) → 1.0 (must keep)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Node feature matrix [batch, node_dim]
        Returns:
            refined: Refined embeddings [batch, node_dim]
            importance: Node importance scores [batch, 1]
        """
        h = self.message_fc(x)
        refined = self.update_fc(h)
        importance = self.importance_head(h)
        return refined, importance


def export(output_path: str, node_dim: int = 3072, batch_size: int = 1):
    """Export GNN model to ONNX."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    model = GnnDiffusionModel(node_dim=node_dim)
    model.eval()

    # Dummy input: one node with Poincaré coords (norm < 0.9)
    dummy_input = torch.randn(batch_size, node_dim) * 0.5

    print(f"[GNN] Exporting model to: {output_path}")
    print(f"[GNN] Input shape:  [{batch_size}, {node_dim}]")
    print(f"[GNN] Output shape: refined=[{batch_size}, {node_dim}], importance=[{batch_size}, 1]")

    with torch.no_grad():
        refined, importance = model(dummy_input)
        print(f"[GNN] Sample refined embedding norm: {refined.norm().item():.4f}")
        print(f"[GNN] Sample importance score:       {importance[0].item():.4f}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["node_features"],
        output_names=["refined_embeddings", "importance_scores"],
        dynamic_axes={
            "node_features":       {0: "batch_size"},
            "refined_embeddings":  {0: "batch_size"},
            "importance_scores":   {0: "batch_size"},
        },
    )
    print(f"[GNN] ✅ Model exported successfully → {output_path}")

    # Quick validation
    try:
        import onnx
        m = onnx.load(output_path)
        onnx.checker.check_model(m)
        print(f"[GNN] ✅ ONNX model validation passed.")
    except ImportError:
        print("[GNN] ⚠️  onnx package not installed — skipping validation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GNN diffusion model to ONNX")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "../../models/gnn_diffusion.onnx"),
        help="Output path for the ONNX file",
    )
    parser.add_argument("--node-dim", type=int, default=3072, help="Node embedding dimension")
    args = parser.parse_args()

    export(output_path=args.output, node_dim=args.node_dim)
