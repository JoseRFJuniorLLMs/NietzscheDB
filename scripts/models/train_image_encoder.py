#!/usr/bin/env python3
"""
train_image_encoder.py — Train the CNN Image Encoder for NietzscheDB sensory layer.

Self-supervised contrastive learning: pairs of augmented crops from the same
image should map to nearby points; crops from different images should be far apart.

Data: loads from checkpoints/image_dataset.pt or generates synthetic patches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from export_image_encoder import ImageEncoder


class ImagePatchDataset(Dataset):
    """Synthetic or real image patches for contrastive training."""

    def __init__(self, num_samples=2000, data_path="../../checkpoints/image_dataset.pt"):
        if os.path.exists(data_path):
            print(f"[DATASET] Loading real image patches from {data_path}")
            data = torch.load(data_path, weights_only=False)
            self.images = data["images"]  # [N, 3, 64, 64]
        else:
            print("[DATASET] No real data. Generating synthetic image patches.")
            # Synthetic: random colored patches with structure (gradients, edges)
            self.images = torch.zeros(num_samples, 3, 64, 64)
            for i in range(num_samples):
                # Create structured patterns (not pure noise)
                freq = torch.randint(1, 8, (1,)).item()
                phase = torch.rand(3, 1, 1) * 6.28
                x = torch.linspace(0, freq * 3.14, 64).view(1, 1, 64).expand(3, 64, 64)
                y = torch.linspace(0, freq * 3.14, 64).view(1, 64, 1).expand(3, 64, 64)
                self.images[i] = (torch.sin(x + phase) * torch.cos(y + phase) + 1) / 2

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # Create two augmented views
        view1 = self._augment(img)
        view2 = self._augment(img)
        return view1, view2

    @staticmethod
    def _augment(img):
        """Simple augmentation: noise + horizontal flip."""
        aug = img + torch.randn_like(img) * 0.05
        if torch.rand(1).item() > 0.5:
            aug = aug.flip(-1)
        return aug.clamp(0, 1)


def contrastive_loss(z1, z2, temperature=0.07):
    """NT-Xent (SimCLR-style) contrastive loss."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size),
    ]).to(z.device)

    return F.cross_entropy(sim, labels)


def train(epochs=20, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ImageEncoder] Training on {device}")

    model = ImageEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = ImagePatchDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for view1, view2 in loader:
            view1, view2 = view1.to(device), view2.to(device)
            optimizer.zero_grad()
            z1 = model(view1)
            z2 = model(view2)
            loss = contrastive_loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Contrastive Loss: {avg:.6f}")

    os.makedirs("../../checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "../../checkpoints/image_encoder.pt")
    print("[ImageEncoder] Weights saved to checkpoints/image_encoder.pt")

    # Export ONNX
    model.eval()
    dummy = torch.randn(1, 3, 64, 64).to(device)
    os.makedirs("../../models", exist_ok=True)
    torch.onnx.export(
        model, dummy, "../../models/image_encoder.onnx",
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["image"], output_names=["latent"],
        dynamic_axes={"image": {0: "batch"}, "latent": {0: "batch"}},
    )
    print("[ImageEncoder] ONNX exported → models/image_encoder.onnx")


if __name__ == "__main__":
    train()
