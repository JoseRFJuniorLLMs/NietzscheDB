#!/usr/bin/env python3
"""
export_image_encoder.py — Small CNN Image Encoder for NietzscheDB Sensory Layer.

Compresses image patches into 128D latent vectors suitable for Poincaré ball
projection via exp_map_zero.

Architecture:
    Input:  [B, 3, 64, 64]  (small patches — clinical thumbnails, avatars)
    Conv:   3→32→64→128 with BatchNorm + GELU + MaxPool
    FC:     128*4*4=2048 → 512 → 128
    Output: [B, 128]  (Euclidean, projected to Poincaré in Rust)
"""

import torch
import torch.nn as nn
import os


class ImageEncoder(nn.Module):
    """CNN encoder: 64×64 RGB image → 128D latent vector."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()

        # Conv backbone: 64×64 → 32×32 → 16×16 → 8×8 → 4×4
        self.features = nn.Sequential(
            # Block 1: 3 → 32, 64→32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 2: 32 → 64, 32→16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 3: 64 → 128, 16→8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 4: 128 → 128, 8→4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        # FC head: 128*4*4 = 2048 → 512 → 128
        self.head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 64, 64] → [B, 128]"""
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.head(h)


def export_onnx(output_dir: str = "../../models"):
    os.makedirs(output_dir, exist_ok=True)
    model = ImageEncoder()
    model.eval()

    dummy = torch.randn(1, 3, 64, 64)
    path = os.path.join(output_dir, "image_encoder.onnx")

    torch.onnx.export(
        model, dummy, path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["latent"],
        dynamic_axes={"image": {0: "batch"}, "latent": {0: "batch"}},
    )
    print(f"[ImageEncoder] Exported → {path}")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(path))
        print("[ImageEncoder] ONNX validation passed.")
    except ImportError:
        pass


if __name__ == "__main__":
    export_onnx()
