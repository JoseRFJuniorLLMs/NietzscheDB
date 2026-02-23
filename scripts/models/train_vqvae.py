#!/usr/bin/env python3
"""
train_vqvae.py — Training harness for VQ-VAE (Vector Quantized Variational Autoencoder).
Task 3.2 - Latent Compression for NietzscheDB.

This model compresses high-dimensional (3072D) embeddings into discrete codes,
enabling Hierarchical DSI indexing (Phase 4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # Convert inputs from [B, D]
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity, encoding_indices

class VqVae(nn.Module):
    def __init__(self, input_dim=3072, latent_dim=512, num_embeddings=1024):
        super(VqVae, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        self.vq = VectorQuantizer(num_embeddings, latent_dim, 0.25)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, indices = self.vq(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity

def train(epochs=15, batch_size=64, lr=1e-4, data_path="../../checkpoints/clinical_dataset.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VQ-VAE] Training on {device}")

    model = VqVae().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load real data or fallback to mock
    if os.path.exists(data_path):
        print(f"[VQ-VAE] Loading real clinical data from {data_path}")
        data_dict = torch.load(data_path)
        data = data_dict["embeddings"]
    else:
        print(f"[VQ-VAE] No real data found at {data_path}. Using mock latent embeddings.")
        data = torch.randn(2000, 3072)

    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_recon_loss = 0
        total_vq_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            vq_loss, x_recon, _ = model(batch)
            recon_loss = F.mse_loss(x_recon, batch)
            
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Recon Loss: {total_recon_loss/len(loader):.6f}, VQ Loss: {total_vq_loss/len(loader):.6f}")

    os.makedirs("../../checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "../../checkpoints/vqvae.pt")
    
    # Export to ONNX for NietzscheDB ingestion
    model.eval()
    dummy_input = torch.randn(1, 3072).to(device)
    torch.onnx.export(model, dummy_input, "../../models/vqvae.onnx", 
                     input_names=["input"], output_names=["vq_loss", "recon", "perplexity"],
                     opset_version=12)
    print("[VQ-VAE] ✅ Model exported to models/vqvae.onnx")

if __name__ == "__main__":
    train()
