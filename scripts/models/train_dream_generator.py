#!/usr/bin/env python3
"""
train_dream_generator.py — Train the Hyperbolic Dream Generator.

Training objective: generate embeddings that are:
1. Close to the seed (reconstruction fidelity)
2. On the data manifold (adversarial / MMD regularization)
3. Inside the Poincaré ball (norm < 1.0)

Uses a simple MMD (Maximum Mean Discrepancy) approach instead of full GAN
to keep training stable with small datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from export_dream_generator import DreamGenerator


class EmbeddingDataset(Dataset):
    """Real or synthetic node embeddings for dream training."""

    def __init__(self, num_samples=2000, embed_dim=128,
                 data_path="../../checkpoints/clinical_dataset.pt"):
        if os.path.exists(data_path):
            print(f"[DATASET] Loading real embeddings from {data_path}")
            data = torch.load(data_path, weights_only=False)
            emb = data["embeddings"]
            # If 3072D, project down to 128D via random projection
            if emb.shape[1] > embed_dim:
                proj = torch.randn(emb.shape[1], embed_dim)
                proj = proj / proj.norm(dim=0, keepdim=True)
                emb = emb @ proj
            self.embeddings = emb
        else:
            print("[DATASET] No real data. Generating synthetic embeddings.")
            # Points on a hyperbolic-like manifold (norm < 1)
            raw = torch.randn(num_samples, embed_dim)
            norms = raw.norm(dim=1, keepdim=True)
            self.embeddings = raw / norms * torch.rand(num_samples, 1) * 0.95

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


def mmd_loss(x, y, sigma=1.0):
    """Maximum Mean Discrepancy between two distributions."""
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    K_xx = torch.exp(-sigma * (rx.t() + rx - 2 * xx))
    K_yy = torch.exp(-sigma * (ry.t() + ry - 2 * yy))
    K_xy = torch.exp(-sigma * (rx.t() + ry - 2 * xy))

    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()


def poincare_penalty(x, max_norm=0.99):
    """Penalize points outside the Poincaré ball."""
    norms = x.norm(dim=1)
    violations = F.relu(norms - max_norm)
    return violations.mean()


def train(epochs=30, batch_size=64, lr=1e-3, noise_dim=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DreamGenerator] Training on {device}")

    model = DreamGenerator(embed_dim=128, noise_dim=noise_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = EmbeddingDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        total_loss = 0
        for real in loader:
            real = real.to(device)
            b = real.size(0)
            noise = torch.randn(b, noise_dim, device=device)

            optimizer.zero_grad()

            generated = model(real, noise)

            # 1. Proximity: generated should be near seed
            proximity = F.mse_loss(generated, real) * 0.5

            # 2. Manifold: generated distribution should match real
            manifold = mmd_loss(generated, real) * 2.0

            # 3. Poincaré constraint
            poincare = poincare_penalty(generated) * 10.0

            loss = proximity + manifold + poincare
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg:.6f}")

    os.makedirs("../../checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "../../checkpoints/dream_generator.pt")
    print("[DreamGenerator] Weights saved to checkpoints/dream_generator.pt")

    # Export ONNX (single-input wrapper)
    from export_dream_generator import DreamGeneratorSingleInput
    model.eval()
    wrapper = DreamGeneratorSingleInput(model).to(device)
    wrapper.eval()
    dummy = torch.randn(1, 192).to(device)
    os.makedirs("../../models", exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, "../../models/dream_generator.onnx",
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["seed_noise"], output_names=["generated"],
        dynamic_axes={"seed_noise": {0: "batch"}, "generated": {0: "batch"}},
    )
    print("[DreamGenerator] ONNX exported → models/dream_generator.onnx")


if __name__ == "__main__":
    train()
