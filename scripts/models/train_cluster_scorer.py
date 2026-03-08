#!/usr/bin/env python3
"""
train_cluster_scorer.py — Train the Cluster Quality Scorer.

Supervised: learns from cluster health heuristics.
    - Clusters too large (>500 nodes) or low coherence → split
    - Clusters too small (<5 nodes) or high overlap → merge
    - Otherwise → keep

Data: loads from checkpoints/cluster_dataset.pt or generates synthetic examples.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from export_cluster_scorer import ClusterScorer


class ClusterDataset(Dataset):
    """Synthetic cluster statistics with heuristic labels."""

    def __init__(self, num_samples=3000, centroid_dim=128,
                 data_path="../../checkpoints/cluster_dataset.pt"):
        if os.path.exists(data_path):
            print(f"[DATASET] Loading real cluster data from {data_path}")
            data = torch.load(data_path, weights_only=False)
            self.features = data["features"]
            self.labels = data["labels"]
        else:
            print("[DATASET] Generating synthetic cluster data with heuristic labels.")
            self.features = torch.zeros(num_samples, centroid_dim * 2 + 5)
            self.labels = torch.zeros(num_samples, dtype=torch.long)

            for i in range(num_samples):
                centroid = torch.randn(centroid_dim) * 0.5
                variance = torch.abs(torch.randn(centroid_dim)) * 0.3

                size = np.random.exponential(100)
                density = np.random.uniform(0.01, 1.0)
                avg_weight = np.random.uniform(0.1, 1.0)
                diameter = np.random.uniform(0.5, 5.0)
                coherence = np.random.uniform(0.0, 1.0)

                stats = torch.tensor([size / 1000, density, avg_weight, diameter / 5, coherence])
                self.features[i] = torch.cat([centroid, variance, stats])

                # Heuristic labels
                if size > 500 or coherence < 0.3:
                    self.labels[i] = 1  # split
                elif size < 5 or density > 0.8:
                    self.labels[i] = 2  # merge
                else:
                    self.labels[i] = 0  # keep

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train(epochs=15, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ClusterScorer] Training on {device}")

    model = ClusterScorer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = ClusterDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            probs = model(features)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (probs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        acc = correct / total * 100
        avg = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg:.4f} | Accuracy: {acc:.1f}%")

    os.makedirs("../../checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "../../checkpoints/cluster_scorer.pt")
    print("[ClusterScorer] Weights saved to checkpoints/cluster_scorer.pt")

    # Export ONNX
    model.eval()
    dummy = torch.randn(1, 261).to(device)
    os.makedirs("../../models", exist_ok=True)
    torch.onnx.export(
        model, dummy, "../../models/cluster_scorer.onnx",
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["cluster_features"], output_names=["action_probs"],
        dynamic_axes={"cluster_features": {0: "batch"}, "action_probs": {0: "batch"}},
    )
    print("[ClusterScorer] ONNX exported → models/cluster_scorer.onnx")


if __name__ == "__main__":
    train()
