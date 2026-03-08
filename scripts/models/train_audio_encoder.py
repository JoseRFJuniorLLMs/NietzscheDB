#!/usr/bin/env python3
"""
train_audio_encoder.py — Train the Neural Audio Encoder for NietzscheDB.

Self-supervised: reconstructive + contrastive objective on mel spectrograms.
The encoder learns to map audio segments into compact 128D representations
that preserve prosodic and timbral similarity.

Data: loads from checkpoints/audio_dataset.pt or generates synthetic spectrograms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from export_audio_encoder import AudioNeuralEncoder


class AudioDecoder(nn.Module):
    """Mirror decoder for reconstruction objective."""

    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128 * 4 * 2),
            nn.GELU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 4×2 → 8×4
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8×4 → 16×8
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16×8 → 32×16
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),     # 32×16 → 64×32
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 128, 4, 2)
        return self.deconv(h)


class MelSpectrogramDataset(Dataset):
    """Synthetic or real mel spectrogram dataset."""

    def __init__(self, num_samples=2000, data_path="../../checkpoints/audio_dataset.pt"):
        if os.path.exists(data_path):
            print(f"[DATASET] Loading real audio data from {data_path}")
            data = torch.load(data_path, weights_only=False)
            self.specs = data["spectrograms"]  # [N, 1, 64, 32]
        else:
            print("[DATASET] No real data. Generating synthetic mel spectrograms.")
            self.specs = torch.zeros(num_samples, 1, 64, 32)
            for i in range(num_samples):
                # Simulate mel spectrogram patterns (harmonic + noise)
                f0 = np.random.uniform(2, 10)
                harmonics = np.random.randint(2, 6)
                spec = np.zeros((64, 32))
                for h in range(1, harmonics + 1):
                    freq_bin = min(int(f0 * h), 63)
                    bandwidth = np.random.randint(1, 4)
                    for b in range(max(0, freq_bin - bandwidth), min(64, freq_bin + bandwidth)):
                        spec[b, :] = np.random.uniform(0.3, 1.0) / h
                spec += np.random.randn(64, 32) * 0.05
                self.specs[i, 0] = torch.from_numpy(spec).float().clamp(0, 1)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        return self.specs[idx]


def train(epochs=20, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AudioEncoder] Training on {device}")

    encoder = AudioNeuralEncoder().to(device)
    decoder = AudioDecoder().to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    dataset = MelSpectrogramDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            z = encoder(batch)
            recon = decoder(z)
            loss = F.mse_loss(recon, batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Recon Loss: {avg:.6f}")

    os.makedirs("../../checkpoints", exist_ok=True)
    torch.save(encoder.state_dict(), "../../checkpoints/audio_encoder.pt")
    print("[AudioEncoder] Weights saved to checkpoints/audio_encoder.pt")

    # Export ONNX (encoder only)
    encoder.eval()
    dummy = torch.randn(1, 1, 64, 32).to(device)
    os.makedirs("../../models", exist_ok=True)
    torch.onnx.export(
        encoder, dummy, "../../models/audio_encoder.onnx",
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["mel_spectrogram"], output_names=["latent"],
        dynamic_axes={"mel_spectrogram": {0: "batch"}, "latent": {0: "batch"}},
    )
    print("[AudioEncoder] ONNX exported → models/audio_encoder.onnx")


if __name__ == "__main__":
    train()
