#!/usr/bin/env python3
"""
train_anomaly_detector.py — Train the Anomaly Detector for NietzscheDB.

Two-phase training:
    Phase 1: Reconstruction — learn normal graph state distribution (autoencoder)
    Phase 2: Anomaly classification — fine-tune anomaly head with injected anomalies

Data: loads from checkpoints/health_trajectories.pt or generates synthetic states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from export_anomaly_detector import AnomalyDetector


class HealthStateDataset(Dataset):
    """Normal + anomalous graph health states."""

    def __init__(self, num_normal=3000, num_anomaly=1000, state_dim=64,
                 data_path="../../checkpoints/health_trajectories.pt"):
        if os.path.exists(data_path):
            print(f"[DATASET] Loading real health data from {data_path}")
            data = torch.load(data_path, weights_only=False)
            if "states" in data:
                normal = data["states"]
            else:
                normal = torch.randn(num_normal, state_dim) * 0.3
            num_normal = len(normal)
        else:
            print("[DATASET] Generating synthetic health states.")
            # Normal states: centered, bounded, Poincaré-constrained
            normal = torch.randn(num_normal, state_dim) * 0.3
            for i in range(num_normal):
                norm = normal[i].norm()
                if norm > 0.95:
                    normal[i] = normal[i] / norm * 0.95

        # Generate anomalous states
        anomalies = torch.zeros(num_anomaly, state_dim)
        for i in range(num_anomaly):
            atype = np.random.choice(["spike", "collapse", "drift", "oscillate"])
            if atype == "spike":
                # Energy spike: extreme values in first few features
                anomalies[i] = torch.randn(state_dim) * 0.3
                anomalies[i, :5] = torch.randn(5) * 3.0
            elif atype == "collapse":
                # Collapse: all features near zero (dead graph region)
                anomalies[i] = torch.randn(state_dim) * 0.01
            elif atype == "drift":
                # Drift: systematic bias in one direction
                anomalies[i] = torch.ones(state_dim) * 0.8 + torch.randn(state_dim) * 0.05
            else:
                # Oscillation: alternating extreme values
                anomalies[i] = torch.randn(state_dim) * 0.3
                anomalies[i, ::2] *= 5.0

        self.states = torch.cat([normal, anomalies], dim=0)
        self.labels = torch.cat([
            torch.zeros(num_normal, 1),   # 0 = normal
            torch.ones(num_anomaly, 1),   # 1 = anomaly
        ], dim=0)

        # Shuffle
        perm = torch.randperm(len(self.states))
        self.states = self.states[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]


def train(epochs_recon=15, epochs_anomaly=10, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AnomalyDetector] Training on {device}")

    model = AnomalyDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = HealthStateDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Phase 1: Reconstruction (autoencoder)
    print("\n--- Phase 1: Reconstruction Training ---")
    for epoch in range(epochs_recon):
        total_loss = 0
        for states, _ in loader:
            states = states.to(device)
            optimizer.zero_grad()

            output = model(states)
            recon = output[:, :64]
            loss = F.mse_loss(recon, states)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Phase 1 Epoch [{epoch+1}/{epochs_recon}] - Recon Loss: {avg:.6f}")

    # Phase 2: Anomaly classification (fine-tune anomaly head)
    print("\n--- Phase 2: Anomaly Classification ---")
    anomaly_criterion = nn.BCELoss()
    for epoch in range(epochs_anomaly):
        total_loss = 0
        correct = 0
        total = 0
        for states, labels in loader:
            states, labels = states.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model(states)
            recon = output[:, :64]
            anomaly_score = output[:, 64:]

            # Combined loss
            recon_loss = F.mse_loss(recon, states)
            anomaly_loss = anomaly_criterion(anomaly_score, labels)
            loss = recon_loss * 0.3 + anomaly_loss * 0.7
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += ((anomaly_score > 0.5).float() == labels).sum().item()
            total += labels.size(0)

        acc = correct / total * 100
        avg = total_loss / len(loader)
        print(f"Phase 2 Epoch [{epoch+1}/{epochs_anomaly}] - Loss: {avg:.4f} | Accuracy: {acc:.1f}%")

    os.makedirs("../../checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "../../checkpoints/anomaly_detector.pt")
    print("[AnomalyDetector] Weights saved to checkpoints/anomaly_detector.pt")

    # Export ONNX
    model.eval()
    dummy = torch.randn(1, 64).to(device)
    os.makedirs("../../models", exist_ok=True)
    torch.onnx.export(
        model, dummy, "../../models/anomaly_detector.onnx",
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["state"], output_names=["recon_and_score"],
        dynamic_axes={"state": {0: "batch"}, "recon_and_score": {0: "batch"}},
    )
    print("[AnomalyDetector] ONNX exported → models/anomaly_detector.onnx")


if __name__ == "__main__":
    train()
