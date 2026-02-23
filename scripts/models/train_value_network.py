#!/usr/bin/env python3
"""
train_value_network.py — Training harness for MCTS Value Network.

This model learns to score 'clinical states'. It is the 'Critic' inside
the System 2 Engine loop. It predicts the probability of a positive clinical
outcome if a certain reasoning node is chosen.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from export_value_network import ValueNetwork

class MctsOutcomeDataset(Dataset):
    """
    Attempts to load real clinical data from clinical_dataset.pt, falls back to mock.
    X: Current latent state (3072D)
    y: Final outcome score [0, 1] (0=crisis, 1=stability)
    """
    def __init__(self, samples=1500, data_path="../../checkpoints/clinical_dataset.pt"):
        if os.path.exists(data_path):
            print(f"[VALUE-NET] Loading real clinical data from {data_path}")
            data = torch.load(data_path)
            self.states = data["embeddings"]
            self.outcomes = data["labels"] # Using prioritized labels as outcome targets
        else:
            print(f"[VALUE-NET] No real data found at {data_path}. Using mock rollouts.")
            self.states = torch.randn(samples, 3072)
            # Higher energy nodes (simulated) lead to better outcomes
            self.outcomes = torch.sigmoid(self.states.mean(dim=1, keepdim=True) * 5.0)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.outcomes[idx]

def train(epochs=12, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VALUE-NET] Training on {device}")

    model = ValueNetwork(input_dim=3072).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss() # Since we want a probability [0, 1]

    loader = DataLoader(MctsOutcomeDataset(), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0
        for states, outcomes in loader:
            states, outcomes = states.to(device), outcomes.to(device)
            
            optimizer.zero_grad()
            preds = model(states)
            loss = criterion(preds, outcomes)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(loader):.6f}")

    os.makedirs("../../checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "../../checkpoints/value_network.pt")
    print("[VALUE-NET] ✅ Weights saved. System 2 'Critic' is ready for fine-tuning.")

if __name__ == "__main__":
    train()
