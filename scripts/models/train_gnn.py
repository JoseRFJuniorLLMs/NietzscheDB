#!/usr/bin/env python3
"""
train_gnn.py — Training harness for the GNN Diffusion Model.

This script implements the training loop for the GNN used in NietzscheDB.
It optimizes the model to learn graph structure importance and embedding diffusion.

Goal: Minimise 'Free Energy' by predicting future node states based on neighbors.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from export_gnn import GnnDiffusionModel

class NietzscheGraphDataset(Dataset):
    """
    Dataset representing graph nodes and their 'future' energy states.
    Attempts to load real data from clinical_dataset.pt, falls back to mock.

    When distilled data is available (via json_to_pt.py from the Go distiller),
    the target_diffusion field contains the ground truth from the classical
    Chebyshev heat kernel — the "Professor" that teaches the GNN "Student".
    """
    def __init__(self, num_samples=1000, node_dim=3072, data_path="../../checkpoints/clinical_dataset.pt"):
        if os.path.exists(data_path):
            print(f"[DATASET] Loading real clinical data from {data_path}")
            data = torch.load(data_path, weights_only=False)
            self.node_features = data["embeddings"]
            self.importance_labels = data["labels"]

            # Check if the Professor (Chebyshev distiller) provided real targets
            if "target_diffusion" in data:
                print(f"[DATASET] Professor targets found! Shape: {data['target_diffusion'].shape}")
                print(f"[DATASET] Training with REAL Chebyshev diffusion labels (zero-data method)")
                # Expand target_diffusion (128D) to match node_dim (3072D)
                # by padding with the original embedding (the diffusion refines, not replaces)
                td = data["target_diffusion"]  # [N, 128]
                pad_dim = node_dim - td.shape[1]
                if pad_dim > 0:
                    # First 128D = diffusion targets, rest = original embedding (subtle shift)
                    self.target_features = self.node_features.clone()
                    self.target_features[:, :td.shape[1]] = td
                else:
                    self.target_features = td[:, :node_dim]
            else:
                print(f"[DATASET] No distilled targets — using self-supervised diffusion shift")
                self.target_features = self.node_features + torch.randn_like(self.node_features) * 0.05
        else:
            print(f"[DATASET] No real data found at {data_path}. Using mock manifolds.")
            self.node_features = torch.randn(num_samples, node_dim) * 0.5
            self.target_features = self.node_features + torch.randn(num_samples, node_dim) * 0.1
            self.importance_labels = torch.rand(num_samples, 1)

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        return self.node_features[idx], self.target_features[idx], self.importance_labels[idx]

def train(epochs=10, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")

    # Initialize model from architecture definition
    model = GnnDiffusionModel(node_dim=3072).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss functions
    embedding_loss_fn = nn.MSELoss()
    importance_loss_fn = nn.BCELoss()

    dataset = NietzscheGraphDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (feat, target_feat, target_imp) in enumerate(dataloader):
            feat, target_feat, target_imp = feat.to(device), target_feat.to(device), target_imp.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            refined, importance = model(feat)
            
            # Multi-task loss: Diffusion accuracy + Importance prediction
            loss_diff = embedding_loss_fn(refined, target_feat)
            loss_imp = importance_loss_fn(importance, target_imp)
            
            loss = loss_diff + 0.5 * loss_imp
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

    # Save trained weights
    os.makedirs("../../checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "../../checkpoints/gnn_diffusion.pt")
    print("[TRAIN] ✅ GNN training complete. Weights saved to checkpoints/gnn_diffusion.pt")

if __name__ == "__main__":
    train()
