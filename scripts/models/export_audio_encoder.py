#!/usr/bin/env python3
"""
export_audio_encoder.py — Neural Audio Encoder for NietzscheDB Sensory Layer.

Compresses mel-spectrogram features into 128D latent vectors for Poincaré
ball projection. Replaces the passthrough AudioEncoder with learned features.

Architecture:
    Input:  [B, 1, 64, 32]  (64 mel bins × 32 time frames ≈ 1s of audio)
    Conv1D-style 2D convs:  1→32→64→128
    FC:     128*4*2=1024 → 256 → 128
    Output: [B, 128]
"""

import torch
import torch.nn as nn
import os


class AudioNeuralEncoder(nn.Module):
    """Conv encoder: mel-spectrogram [B, 1, 64, 32] → 128D latent."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            # 1 → 32, 64×32 → 32×16
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            # 32 → 64, 32×16 → 16×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),

            # 64 → 128, 16×8 → 8×4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),

            # 128 → 128, 8×4 → 4×2
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        # 128*4*2 = 1024 → 256 → 128
        self.head = nn.Sequential(
            nn.Linear(128 * 4 * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
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
        """x: [B, 1, 64, 32] → [B, 128]"""
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.head(h)


def export_onnx(output_dir: str = "../../models"):
    os.makedirs(output_dir, exist_ok=True)
    model = AudioNeuralEncoder()
    model.eval()

    dummy = torch.randn(1, 1, 64, 32)
    path = os.path.join(output_dir, "audio_encoder.onnx")

    torch.onnx.export(
        model, dummy, path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["mel_spectrogram"],
        output_names=["latent"],
        dynamic_axes={"mel_spectrogram": {0: "batch"}, "latent": {0: "batch"}},
    )
    print(f"[AudioEncoder] Exported → {path}")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(path))
        print("[AudioEncoder] ONNX validation passed.")
    except ImportError:
        pass


if __name__ == "__main__":
    export_onnx()
