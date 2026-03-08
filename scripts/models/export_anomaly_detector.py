#!/usr/bin/env python3
"""
export_anomaly_detector.py — Anomaly Detector Autoencoder for NietzscheDB.

Detects degenerative patterns in graph health by learning "normal" state
distributions. High reconstruction error = anomaly.

Used by the Wiederkehr daemon engine to flag unhealthy graph regions.

Architecture:
    Input:   [B, 64]  (graph health state — same format as PPO/ValueNetwork)
    Encoder: 64 → 32 → 16  (bottleneck)
    Decoder: 16 → 32 → 64
    Output:  [B, 65]  ([reconstructed_64D, anomaly_score_1D])
"""

import torch
import torch.nn as nn
import os


class AnomalyDetector(nn.Module):
    """Autoencoder anomaly detector for graph health states."""

    def __init__(self, state_dim: int = 64, bottleneck: int = 16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, bottleneck),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, state_dim),
        )

        # Anomaly scoring head (from bottleneck)
        self.anomaly_head = nn.Sequential(
            nn.Linear(bottleneck, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 64] → [B, 65] (reconstructed + anomaly_score)
        """
        z = self.encoder(x)
        recon = self.decoder(z)
        anomaly = self.anomaly_head(z)
        return torch.cat([recon, anomaly], dim=1)

    def encode(self, x):
        return self.encoder(x)

    def anomaly_score(self, x):
        """Compute both reconstruction error and learned anomaly score."""
        z = self.encoder(x)
        recon = self.decoder(z)
        learned_score = self.anomaly_head(z)
        recon_error = ((x - recon) ** 2).mean(dim=1, keepdim=True)
        # Combine: learned + reconstruction error
        return learned_score * 0.5 + torch.sigmoid(recon_error * 10) * 0.5


def export_onnx(output_dir: str = "../../models"):
    os.makedirs(output_dir, exist_ok=True)
    model = AnomalyDetector()
    model.eval()

    dummy = torch.randn(1, 64)
    path = os.path.join(output_dir, "anomaly_detector.onnx")

    torch.onnx.export(
        model, dummy, path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["state"],
        output_names=["recon_and_score"],
        dynamic_axes={"state": {0: "batch"}, "recon_and_score": {0: "batch"}},
    )
    print(f"[AnomalyDetector] Exported → {path}")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(path))
        print("[AnomalyDetector] ONNX validation passed.")
    except ImportError:
        pass


if __name__ == "__main__":
    export_onnx()
