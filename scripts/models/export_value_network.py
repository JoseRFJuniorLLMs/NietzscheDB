#!/usr/bin/env python3
"""
export_value_network.py — Export a MCTS Value Network as an ONNX model.

Architecture: 3-layer MLP that evaluates a compressed graph state and returns
a value score (0.0 = bad state / 1.0 = excellent clinical hypothesis).

  - Input:  graph state vector [batch, state_dim=64] (Krylov subspace projection)
  - Output: value score [batch, 1]

This Value Network is the "Monstrinho Avaliador" from the System 2 architecture:
it receives the NietzscheDB Minkowski/GNN-scored features and gives a final scalar
judgement used by the MCTS `advise()` method to rank candidate actions.

Usage:
    pip install torch onnx numpy
    python export_value_network.py --output ../../models/value_network.onnx
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import os


class ValueNetwork(nn.Module):
    """
    MCTS Value Network — evaluates a clinical hypothesis / graph state.

    Input features (state_dim=64) represent:
      - Causal connectedness score (Minkowski interval)
      - Mean energy of affected nodes
      - Hausdorff dimension delta (from sleep cycle)
      - Temporal coherence (are events in causal order?)
      - GNN importance score of the seed node
      - ... and 59 other learned latent features

    In production this is trained using outcome data from EVA's clinical sessions.
    """

    def __init__(self, state_dim: int = 64, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1: Feature extraction
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),

            # Layer 2: Reasoning
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.05),

            # Layer 3: Policy head
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),

            # Output: scalar value
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),  # Bounded 0.0 → 1.0
        )

        # Init weights for stable training start
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Graph state features [batch, state_dim]
        Returns:
            value: State value score [batch, 1], 0=bad, 1=excellent
        """
        return self.net(state)


def export(output_path: str, state_dim: int = 64, batch_size: int = 1):
    """Export Value Network model to ONNX."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    model = ValueNetwork(state_dim=state_dim)
    model.eval()

    # Dummy input: random clinical graph state
    dummy_input = torch.randn(batch_size, state_dim)

    print(f"[ValueNet] Exporting model to: {output_path}")
    print(f"[ValueNet] Input shape:  [{batch_size}, {state_dim}]")
    print(f"[ValueNet] Output shape: [{batch_size}, 1]")

    with torch.no_grad():
        value = model(dummy_input)
        print(f"[ValueNet] Sample value score: {value[0].item():.4f}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["graph_state"],
        output_names=["value_score"],
        dynamic_axes={
            "graph_state":  {0: "batch_size"},
            "value_score":  {0: "batch_size"},
        },
    )
    print(f"[ValueNet] ✅ Model exported successfully → {output_path}")

    # Quick validation
    try:
        import onnx
        m = onnx.load(output_path)
        onnx.checker.check_model(m)
        print(f"[ValueNet] ✅ ONNX model validation passed.")
    except ImportError:
        print("[ValueNet] ⚠️  onnx package not installed — skipping validation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MCTS Value Network to ONNX")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "../../models/value_network.onnx"),
        help="Output path for the ONNX file",
    )
    parser.add_argument("--state-dim", type=int, default=64, help="State feature dimension")
    args = parser.parse_args()

    export(output_path=args.output, state_dim=args.state_dim)
