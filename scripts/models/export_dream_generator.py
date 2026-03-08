#!/usr/bin/env python3
"""
export_dream_generator.py — Hyperbolic Dream Generator for NietzscheDB.

Generates new node embeddings by diffusing from a seed embedding with
stochastic noise — like the brain's dream synthesis.

Architecture:
    Input:  [B, 192]  (128D seed embedding + 64D noise vector)
    Hidden: 192 → 256 → 256 → 128
    Output: [B, 128]  (generated node embedding, Euclidean → project to Poincaré)

The noise vector controls the "creativity" of the dream. Low noise = nearby
exploration; high noise = far-flung associative leaps.
"""

import torch
import torch.nn as nn
import os


class DreamGenerator(nn.Module):
    """Conditional generator: seed + noise → new node embedding."""

    def __init__(self, embed_dim: int = 128, noise_dim: int = 64):
        super().__init__()
        input_dim = embed_dim + noise_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, embed_dim),
            nn.Tanh(),  # bound output to [-1, 1] for Poincaré projection
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, seed: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        seed:  [B, 128] — anchor embedding
        noise: [B, 64]  — stochastic noise
        → [B, 128] generated embedding
        """
        x = torch.cat([seed, noise], dim=1)
        return self.net(x)


class DreamGeneratorSingleInput(nn.Module):
    """ONNX-friendly wrapper: single [B, 192] input."""

    def __init__(self, generator: DreamGenerator):
        super().__init__()
        self.generator = generator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seed = x[:, :128]
        noise = x[:, 128:]
        return self.generator(seed, noise)


def export_onnx(output_dir: str = "../../models"):
    os.makedirs(output_dir, exist_ok=True)
    gen = DreamGenerator()
    gen.eval()

    wrapper = DreamGeneratorSingleInput(gen)
    wrapper.eval()

    dummy = torch.randn(1, 192)
    path = os.path.join(output_dir, "dream_generator.onnx")

    torch.onnx.export(
        wrapper, dummy, path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["seed_noise"],
        output_names=["generated"],
        dynamic_axes={"seed_noise": {0: "batch"}, "generated": {0: "batch"}},
    )
    print(f"[DreamGenerator] Exported → {path}")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(path))
        print("[DreamGenerator] ONNX validation passed.")
    except ImportError:
        pass


if __name__ == "__main__":
    export_onnx()
