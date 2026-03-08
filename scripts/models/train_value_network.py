#!/usr/bin/env python3
"""
train_value_network.py — Training harness for MCTS Value Network.

The Value Network evaluates graph states for the MctsAdvisor, scoring how
"good" a particular node exploration path is.

Architecture defined in export_value_network.py:
    Input:  [B, 64]  (Krylov subspace projection of graph state)
    Output: [B, 1]   (value score 0=bad, 1=excellent)

Training: self-play with simulated graph exploration rewards.
Also supports real clinical data from clinical_dataset.pt (projected to 64D).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from export_value_network import ValueNetwork


class GraphStateDataset(Dataset):
    """Graph states with value labels from simulation or real MCTS rollouts."""

    def __init__(self, num_samples=5000, state_dim=64,
                 data_path="../../checkpoints/clinical_dataset.pt"):
        if os.path.exists(data_path):
            print(f"[VALUE-NET] Loading real clinical data from {data_path}")
            data = torch.load(data_path, weights_only=False)
            emb = data["embeddings"]
            labels = data["labels"]
            # Project from 3072D to 64D via random projection if needed
            if emb.shape[1] > state_dim:
                proj = torch.randn(emb.shape[1], state_dim)
                proj = proj / proj.norm(dim=0, keepdim=True)
                emb = emb @ proj
            self.states = emb
            self.values = labels if labels.dim() == 2 else labels.unsqueeze(1)
        else:
            print(f"[VALUE-NET] No real data at {data_path}. Using synthetic graph states.")
            self.states = torch.zeros(num_samples, state_dim)
            self.values = torch.zeros(num_samples, 1)

            for i in range(num_samples):
                state = torch.randn(state_dim) * 0.5
                # Poincare constraint
                norm = state.norm()
                if norm > 0.99:
                    state = state / norm * 0.99
                self.states[i] = state

                # Value heuristic: graph health features
                mean_energy = abs(state[0].item())
                is_fractal = 1.0 if state[1].item() > 0 else 0.0
                coherence = (state[4].item() + 1) / 2
                value = 0.4 * mean_energy + 0.3 * is_fractal + 0.3 * coherence
                self.values[i, 0] = np.clip(value, 0, 1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.values[idx]


def train(epochs=25, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VALUE-NET] Training on {device}")

    model = ValueNetwork(state_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = GraphStateDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    for epoch in range(epochs):
        total_loss = 0
        for states, values in loader:
            states, values = states.to(device), values.to(device)
            optimizer.zero_grad()
            preds = model(states)
            loss = criterion(preds, values)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] - MSE Loss: {avg:.6f}")

        if avg < best_loss:
            best_loss = avg
            os.makedirs("../../checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "../../checkpoints/value_network.pt")

    print(f"[VALUE-NET] Best loss: {best_loss:.6f}")
    print("[VALUE-NET] Weights saved to checkpoints/value_network.pt")

    # Export ONNX
    model.eval()
    dummy = torch.randn(1, 64).to(device)
    os.makedirs("../../models", exist_ok=True)
    torch.onnx.export(
        model, dummy, "../../models/value_network.onnx",
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["state"], output_names=["value"],
        dynamic_axes={"state": {0: "batch"}, "value": {0: "batch"}},
    )
    print("[VALUE-NET] ONNX exported -> models/value_network.onnx")


if __name__ == "__main__":
    train()
