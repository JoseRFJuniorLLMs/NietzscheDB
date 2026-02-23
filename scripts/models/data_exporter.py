#!/usr/bin/env python3
"""
data_exporter.py — Bridge from NietzscheDB Graph to PyTorch Tensors.

This script connects to the local NietzscheDB instance, fetches all nodes
from the 'Patient' collection, and exports their PoE (Poincaré) embeddings
and metadata to a .pt file for neural training.
"""

import os
import torch
import numpy as np
from nietzschedb.client import NietzscheClient

def export_clinical_data(collection="Patient", output_file="../../checkpoints/clinical_dataset.pt"):
    print(f"[EXPORT] Connecting to NietzscheDB...")
    
    # Initialize connection (defaults to localhost:50051)
    client = NietzscheClient()
    
    try:
        # 1. Fetch all nodes in the target collection using NQL
        nql = f"MATCH (n) RETURN n"
        print(f"[EXPORT] Executing NQL: {nql} on collection '{collection}'")
        
        result = client.query(nql, collection=collection)
        
        nodes = result.nodes
        if not nodes:
            print(f"[WARNING] No nodes found in collection '{collection}'. Proceeding with empty dataset.")
            return

        print(f"[EXPORT] Found {len(nodes)} nodes. Converting to tensors...")

        embeddings = []
        labels = []
        metadata = []

        for node in nodes:
            # NietzscheDB stores embeddings as float64 (Poincaré manifold)
            embeddings.append(node.embedding)
            
            # Extract clinical importance (energy or content-based)
            importance = node.energy / 10.0 # Normalized energy as priority
            labels.append([importance])
            
            # Keep metadata for debugging/explainability
            metadata.append({
                "id": node.id,
                "type": node.node_type,
                "content": node.content
            })

        # 2. Convert to PyTorch tensors
        tensor_embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
        tensor_labels = torch.tensor(np.array(labels), dtype=torch.float32)

        dataset = {
            "embeddings": tensor_embeddings,
            "labels": tensor_labels,
            "metadata": metadata,
            "dim": 3072
        }

        # 3. Save to disk
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save(dataset, output_file)
        print(f"[EXPORT] ✅ Dataset saved to {output_file} ({len(nodes)} samples)")

    except Exception as e:
        print(f"[ERROR] Failed to export data: {e}")
        print("[TIP] Ensure NietzscheDB is running on port 50051 and has data in the 'Patient' collection.")

if __name__ == "__main__":
    export_clinical_data()
