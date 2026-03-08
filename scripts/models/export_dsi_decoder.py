#!/usr/bin/env python3
"""
export_dsi_decoder.py — Differentiable Search Index Decoder for NietzscheDB.

Maps a query embedding to a hierarchical document ID (sequence of VQ codes).
This is the neural retrieval component of DSI (Differentiable Search Index):
    query → decoder → sequence of codebook indices → node lookup

Architecture:
    Input:  [B, 128]   (query embedding, 128D Krylov-compressed)
    Hidden: 128 → 256 → 512
    Output: [B, 4, 1024]  (4-level hierarchy, each level selects from 1024 codes)

At inference: argmax each level → 4 VQ code indices → hierarchical node address.
"""

import torch
import torch.nn as nn
import os


class DsiDecoder(nn.Module):
    """Neural DSI: query embedding → hierarchical code sequence."""

    def __init__(self, query_dim: int = 128, num_levels: int = 4, codebook_size: int = 1024):
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size

        self.shared = nn.Sequential(
            nn.Linear(query_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        # One classification head per hierarchy level
        self.level_heads = nn.ModuleList([
            nn.Linear(512, codebook_size) for _ in range(num_levels)
        ])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        query: [B, 128]
        → [B, num_levels, codebook_size] (logits per level)
        """
        h = self.shared(query)
        logits = torch.stack([head(h) for head in self.level_heads], dim=1)
        return logits


def export_onnx(output_dir: str = "../../models"):
    os.makedirs(output_dir, exist_ok=True)
    model = DsiDecoder()
    model.eval()

    dummy = torch.randn(1, 128)
    path = os.path.join(output_dir, "dsi_decoder.onnx")

    torch.onnx.export(
        model, dummy, path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["query"],
        output_names=["level_logits"],
        dynamic_axes={"query": {0: "batch"}, "level_logits": {0: "batch"}},
    )
    print(f"[DsiDecoder] Exported → {path}")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(path))
        print("[DsiDecoder] ONNX validation passed.")
    except ImportError:
        pass


if __name__ == "__main__":
    export_onnx()
