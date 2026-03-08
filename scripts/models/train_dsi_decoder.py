#!/usr/bin/env python3
"""
train_dsi_decoder.py — Train the DSI Decoder (Differentiable Search Index).

The decoder learns to map query embeddings directly to hierarchical document
addresses (VQ code sequences), enabling O(1) neural retrieval.

Training: for each document, the VQ-VAE encoder produces a 4-level code
sequence. The DSI decoder learns to predict those codes from the document's
query embedding.

Data: loads from checkpoints/dsi_dataset.pt or generates synthetic mappings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from export_dsi_decoder import DsiDecoder


class DsiDataset(Dataset):
    """Query embeddings paired with hierarchical VQ code targets."""

    def __init__(self, num_docs=5000, query_dim=128, num_levels=4, codebook_size=1024,
                 data_path="../../checkpoints/dsi_dataset.pt"):
        self.num_levels = num_levels
        if os.path.exists(data_path):
            print(f"[DATASET] Loading real DSI data from {data_path}")
            data = torch.load(data_path, weights_only=False)
            self.queries = data["queries"]
            self.codes = data["codes"]
        else:
            print("[DATASET] Generating synthetic DSI data.")
            # Create clustered embeddings to simulate hierarchical structure
            self.queries = torch.zeros(num_docs, query_dim)
            self.codes = torch.zeros(num_docs, num_levels, dtype=torch.long)

            # Create hierarchical clusters
            for i in range(num_docs):
                # Top-level cluster
                l0 = np.random.randint(codebook_size)
                l1 = np.random.randint(codebook_size)
                l2 = np.random.randint(codebook_size)
                l3 = np.random.randint(codebook_size)

                # Generate embedding consistent with cluster membership
                base = torch.randn(query_dim) * 0.1
                base[:32] += (l0 / codebook_size) * 2 - 1   # level 0 influence
                base[32:64] += (l1 / codebook_size) * 2 - 1  # level 1 influence
                base[64:96] += (l2 / codebook_size) * 2 - 1  # level 2 influence
                base[96:] += (l3 / codebook_size) * 2 - 1    # level 3 influence

                norm = base.norm()
                if norm > 0.99:
                    base = base / norm * 0.99

                self.queries[i] = base
                self.codes[i] = torch.tensor([l0, l1, l2, l3])

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.codes[idx]


def train(epochs=30, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DsiDecoder] Training on {device}")

    model = DsiDecoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = DsiDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_preds = 0

        for queries, codes in loader:
            queries, codes = queries.to(device), codes.to(device)
            optimizer.zero_grad()

            logits = model(queries)  # [B, 4, 1024]

            # Loss across all levels
            loss = 0
            for level in range(model.num_levels):
                loss += criterion(logits[:, level, :], codes[:, level])
            loss /= model.num_levels

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Accuracy
            preds = logits.argmax(dim=2)  # [B, 4]
            total_correct += (preds == codes).sum().item()
            total_preds += codes.numel()

        acc = total_correct / total_preds * 100
        avg = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg:.4f} | Level Accuracy: {acc:.1f}%")

    os.makedirs("../../checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "../../checkpoints/dsi_decoder.pt")
    print("[DsiDecoder] Weights saved to checkpoints/dsi_decoder.pt")

    # Export ONNX
    model.eval()
    dummy = torch.randn(1, 128).to(device)
    os.makedirs("../../models", exist_ok=True)
    torch.onnx.export(
        model, dummy, "../../models/dsi_decoder.onnx",
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["query"], output_names=["level_logits"],
        dynamic_axes={"query": {0: "batch"}, "level_logits": {0: "batch"}},
    )
    print("[DsiDecoder] ONNX exported → models/dsi_decoder.onnx")


if __name__ == "__main__":
    train()
