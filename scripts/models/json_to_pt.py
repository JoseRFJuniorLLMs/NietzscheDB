#!/usr/bin/env python3
"""
json_to_pt.py — Converte distilled_data.json (saída do Professor Go) para clinical_dataset.pt (PyTorch).

Este script é a Ponte entre o pipeline Go (synthetic_generator + distiller) e os scripts
de treino Python (train_gnn.py, train_vqvae.py, train_value_network.py).

Fluxo completo:
  1. Go: cmd/seed_synthetic → gera pacientes + roda Chebyshev → distilled_data.json
  2. Python: json_to_pt.py → lê JSON → empacota em Tensors → clinical_dataset.pt
  3. Python: train_gnn.py → lê clinical_dataset.pt → treina GNN → gnn_diffusion.onnx

Uso:
  python json_to_pt.py --input distilled_data.json --output ../../checkpoints/clinical_dataset.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def convert(input_path: str, output_path: str) -> None:
    if not os.path.exists(input_path):
        print(f"[ERROR] Arquivo '{input_path}' nao encontrado.")
        print("[TIP]   Rode o pipeline Go primeiro:")
        print("        cd EVA && go run ./cmd/seed_synthetic -count 100 -output distilled_data.json")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not samples:
        print("[ERROR] Arquivo JSON vazio — nenhuma amostra para converter.")
        sys.exit(1)

    print(f"[JSON->PT] Lendo {len(samples)} amostras do Professor (Chebyshev)...")

    embeddings = []
    energies = []
    targets = []
    labels = []  # importance/energy normalizada (para BCELoss)

    skipped = 0
    for item in samples:
        emb = item.get("embedding")
        if not emb or len(emb) == 0:
            skipped += 1
            continue

        embeddings.append(emb)
        energies.append(item.get("energy", 0.5))

        # Target do Professor: resultado da difusao de calor (Chebyshev)
        target = item.get("target_diffusion", [])
        targets.append(target)

        # Label de importancia (energy normalizada 0-1)
        importance = min(max(item.get("energy", 0.5), 0.0), 10.0) / 10.0
        labels.append([importance])

    if skipped > 0:
        print(f"[WARN] {skipped} amostras sem embedding foram ignoradas.")

    n = len(embeddings)
    emb_dim = len(embeddings[0])
    target_dim = len(targets[0]) if targets and len(targets[0]) > 0 else 0

    print(f"[JSON->PT] Convertendo {n} amostras...")
    print(f"           Embedding dim: {emb_dim}")
    print(f"           Target dim:    {target_dim}")

    # Montar tensors
    t_embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
    t_labels = torch.tensor(np.array(labels), dtype=torch.float32)

    dataset = {
        "embeddings": t_embeddings,           # [N, 3072] - features de entrada (Poincare)
        "labels": t_labels,                   # [N, 1]    - importance (energy/10)
        "dim": emb_dim,
    }

    # Se o Professor gerou targets de difusao, incluir
    if target_dim > 0:
        t_targets = torch.tensor(np.array(targets), dtype=torch.float32)
        dataset["target_diffusion"] = t_targets  # [N, 128] - ground truth do Chebyshev
        print(f"           Target diffusion: {t_targets.shape}")

    # Energias brutas (para Arousal→Purity mapping)
    t_energies = torch.tensor(np.array(energies), dtype=torch.float32)
    dataset["energies"] = t_energies  # [N] - energia bruta

    # Salvar
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(dataset, output_path)

    print(f"[JSON->PT] Dataset salvo em {output_path}")
    print(f"           Embeddings: {t_embeddings.shape}")
    print(f"           Labels:     {t_labels.shape}")
    print(f"           Energies:   {t_energies.shape}")
    print(f"[JSON->PT] Pronto para treino: python train_gnn.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converte distilled_data.json para clinical_dataset.pt")
    parser.add_argument("--input", default="distilled_data.json",
                        help="Caminho do JSON gerado pelo distiller Go")
    parser.add_argument("--output", default="../../checkpoints/clinical_dataset.pt",
                        help="Caminho de saida do .pt para PyTorch")
    args = parser.parse_args()
    convert(args.input, args.output)
