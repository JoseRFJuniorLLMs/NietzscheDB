#!/usr/bin/env python3
"""
train_edge_predictor.py — Train the Link Prediction Network.

Supervised binary classification:
    Positive pairs: real edges from the graph
    Negative pairs: random node pairs with no edge (negative sampling)

Data: loads from checkpoints/edge_dataset.pt or generates synthetic pairs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from export_edge_predictor import EdgePredictor


class EdgeDataset(Dataset):
    """Edge pairs with binary labels (1=edge exists, 0=no edge)."""

    def __init__(self, num_nodes=500, num_edges=2000, embed_dim=128,
                 data_path="../../checkpoints/edge_dataset.pt"):
        if os.path.exists(data_path):
            print(f"[DATASET] Loading real edge data from {data_path}")
            data = torch.load(data_path, weights_only=False)
            self.pairs = data["pairs"]   # [N, 256]
            self.labels = data["labels"]  # [N]
        else:
            print("[DATASET] Generating synthetic edge data.")
            # Create node embeddings on hyperbolic-like manifold
            nodes = torch.randn(num_nodes, embed_dim) * 0.3
            norms = nodes.norm(dim=1, keepdim=True)
            nodes = nodes / norms * torch.rand(num_nodes, 1) * 0.95

            pairs = []
            labels = []

            # Positive edges: nearby nodes (hyperbolic distance < threshold)
            for _ in range(num_edges):
                i = np.random.randint(num_nodes)
                # Find a nearby node
                dists = (nodes - nodes[i]).norm(dim=1)
                nearby = (dists < 0.5).nonzero(as_tuple=True)[0]
                if len(nearby) > 1:
                    j = nearby[np.random.randint(len(nearby))].item()
                    if j != i:
                        pairs.append(torch.cat([nodes[i], nodes[j]]))
                        labels.append(1.0)

            # Negative edges: random pairs (equal count)
            num_pos = len(pairs)
            for _ in range(num_pos):
                i, j = np.random.randint(num_nodes, size=2)
                while i == j:
                    j = np.random.randint(num_nodes)
                pairs.append(torch.cat([nodes[i], nodes[j]]))
                labels.append(0.0)

            self.pairs = torch.stack(pairs)
            self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]


def train(epochs=20, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EdgePredictor] Training on {device}")

    model = EdgePredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    dataset = EdgeDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for pairs, labels in loader:
            pairs, labels = pairs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(pairs).squeeze(1)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += ((preds > 0.5).float() == labels).sum().item()
            total += labels.size(0)

        acc = correct / total * 100
        avg = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg:.4f} | Accuracy: {acc:.1f}%")

    os.makedirs("../../checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "../../checkpoints/edge_predictor.pt")
    print("[EdgePredictor] Weights saved to checkpoints/edge_predictor.pt")

    # Export ONNX
    model.eval()
    dummy = torch.randn(1, 256).to(device)
    os.makedirs("../../models", exist_ok=True)
    torch.onnx.export(
        model, dummy, "../../models/edge_predictor.onnx",
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["node_pair"], output_names=["edge_prob"],
        dynamic_axes={"node_pair": {0: "batch"}, "edge_prob": {0: "batch"}},
    )
    print("[EdgePredictor] ONNX exported → models/edge_predictor.onnx")


if __name__ == "__main__":
    train()
