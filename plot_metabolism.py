#!/usr/bin/env python3
"""
NietzscheDB — Benchmark: Real Graph Topology + Structural Metrics

Compares 3 birth modes on real graph (10k nodes, ~50k edges):
  D: FOAM          (void-born orphans, degree=0)
  E: ANCHORED      (elite-parented, degree=2, no polarization)
  F: DIALECTICAL   (elite-parented, degree=2, entropy polarization)

Input:  telemetry_D_foam.csv, telemetry_E_anchored.csv, telemetry_F_dialectical.csv
Output: metabolism_report.png (300 DPI, 8-panel dashboard)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

if os.environ.get("DISPLAY") is None and sys.platform != "win32":
    matplotlib.use("Agg")

plt.rcParams.update({
    "figure.figsize": (20, 18),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

COLORS = {
    "D": "#F44336",
    "E": "#2196F3",
    "F": "#4CAF50",
}


def load_data():
    files = {
        "D": "telemetry_D_foam.csv",
        "E": "telemetry_E_anchored.csv",
        "F": "telemetry_F_dialectical.csv",
    }
    data = {}
    for key, path in files.items():
        try:
            data[key] = pd.read_csv(path)
            print(f"  [{key}] {path}: {len(data[key])} ciclos")
        except FileNotFoundError:
            print(f"  [{key}] {path}: NAO ENCONTRADO")
    return data


def plot_metabolism():
    print("=" * 64)
    print("  NietzscheDB: Real Graph Benchmark (3 Birth Modes)")
    print("=" * 64)
    print()

    data = load_data()
    if not data:
        print("Nenhum CSV encontrado.")
        return

    fig, axes = plt.subplots(4, 2)
    fig.suptitle(
        "Real Graph Benchmark — 10k nodes, ~50k edges, 100 cycles\n"
        "TGC = intensity * quality * (1 + 2*dH) * (1 + 3*dE)",
        fontsize=15,
        fontweight="bold",
    )

    labels = {
        "D": "D: Foam (orphans, deg=0)",
        "E": "E: Anchored (deg=2, no pol.)",
        "F": "F: Dialectical (deg=2 + pol.)",
    }

    # ── Q1: Structural Entropy Hs ──
    ax = axes[0, 0]
    for key, df in data.items():
        ax.plot(df["cycle"], df["structural_entropy"], color=COLORS[key],
                linewidth=1.8, label=labels[key], alpha=0.9)
    ax.set_title("Structural Entropy (Hs)")
    ax.set_ylabel("Shannon Entropy")
    ax.set_xlabel("Ciclo")
    ax.legend(fontsize=8)

    # ── Q2: Global Efficiency Eg ──
    ax = axes[0, 1]
    for key, df in data.items():
        ax.plot(df["cycle"], df["global_efficiency"], color=COLORS[key],
                linewidth=1.8, label=labels[key], alpha=0.9)
    ax.set_title("Global Efficiency (Eg) — sampled BFS")
    ax.set_ylabel("Avg 1/d")
    ax.set_xlabel("Ciclo")
    ax.legend(fontsize=8)

    # ── Q3: TGC Raw ──
    ax = axes[1, 0]
    for key, df in data.items():
        ax.plot(df["cycle"], df["tgc_raw"], color=COLORS[key],
                linewidth=1.8, label=labels[key], alpha=0.9)
    ax.set_title("TGC Raw (dH * dE formula)")
    ax.set_ylabel("TGC")
    ax.set_xlabel("Ciclo")
    ax.legend(fontsize=8)

    # ── Q4: TGC EMA ──
    ax = axes[1, 1]
    for key, df in data.items():
        ax.plot(df["cycle"], df["tgc_ema"], color=COLORS[key],
                linewidth=1.8, label=labels[key], alpha=0.9)
    ax.set_title("TGC EMA (smoothed)")
    ax.set_ylabel("TGC EMA")
    ax.set_xlabel("Ciclo")
    ax.legend(fontsize=8)

    # ── Q5: Elite Drift ──
    ax = axes[2, 0]
    for key, df in data.items():
        ax.plot(df["cycle"], df["elite_drift"], color=COLORS[key],
                linewidth=1.8, label=labels[key], alpha=0.9)
    ax.axhline(y=0.1, color="black", linestyle="--", alpha=0.4, label="Limiar (0.1)")
    ax.set_title("Elite Drift (Desvio de Identidade)")
    ax.set_ylabel("Distancia")
    ax.set_xlabel("Ciclo")
    ax.legend(fontsize=8)

    # ── Q6: Var(V) ──
    ax = axes[2, 1]
    for key, df in data.items():
        ax.plot(df["cycle"], df["variance_vitality"], color=COLORS[key],
                linewidth=1.8, label=labels[key], alpha=0.9)
    ax.axhline(y=0.05, color="black", linestyle=":", alpha=0.4, label="Monocultura (0.05)")
    ax.set_title("Var(V) — Variancia de Vitalidade")
    ax.set_ylabel("Variancia")
    ax.set_xlabel("Ciclo")
    ax.legend(fontsize=8)

    # ── Q7: Edges over time ──
    ax = axes[3, 0]
    for key, df in data.items():
        ax.plot(df["cycle"], df["total_edges"], color=COLORS[key],
                linewidth=1.8, label=labels[key], alpha=0.9)
    ax.set_title("Total Edges (Graph Connectivity)")
    ax.set_ylabel("Edges")
    ax.set_xlabel("Ciclo")
    ax.legend(fontsize=8)

    # ── Q8: ms/cycle (Performance) ──
    ax = axes[3, 1]
    for key, df in data.items():
        ax.plot(df["cycle"], df["ms_per_cycle"], color=COLORS[key],
                linewidth=1.2, label=labels[key], alpha=0.7)
    ax.set_title("Performance (ms/cycle)")
    ax.set_ylabel("Milliseconds")
    ax.set_xlabel("Ciclo")
    ax.legend(fontsize=8)

    plt.tight_layout()
    output_path = "metabolism_report.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print()
    print(f"Dashboard gerado: {output_path}")
    print()

    # ── Summary Table ──
    print("=" * 72)
    print("  COMPARACAO: MEDIA COMPLETA (ALL CYCLES)")
    print("=" * 72)
    header = f"{'Metrica':<24} {'D:Foam':>14} {'E:Anchored':>14} {'F:Dialect.':>14}"
    print(header)
    print("-" * len(header))

    metrics = [
        ("avg_Hs", "structural_entropy"),
        ("avg_Eg", "global_efficiency"),
        ("avg_TGC_raw", "tgc_raw"),
        ("avg_TGC_ema", "tgc_ema"),
        ("avg_Elite_Drift", "elite_drift"),
        ("avg_Var(V)", "variance_vitality"),
        ("avg_Sacrificed", "sacrificed"),
        ("avg_Created", "nodes_created"),
        ("avg_Newborn_Surv", "newborn_survival"),
        ("avg_Mean_V", "mean_vitality"),
        ("avg_Mean_E", "mean_energy"),
        ("avg_Nodes", "total_nodes"),
        ("avg_Edges", "total_edges"),
        ("avg_ms/cycle", "ms_per_cycle"),
    ]

    for label, col in metrics:
        vals = []
        for key in ["D", "E", "F"]:
            if key in data and col in data[key].columns:
                vals.append(f"{data[key][col].mean():.4f}")
            else:
                vals.append("N/A")
        print(f"  {label:<22} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")

    # FP + noise
    for metric, col in [("total_FP", "signal_killed"), ("total_noise_del", "noise_killed")]:
        vals = []
        for key in ["D", "E", "F"]:
            if key in data:
                vals.append(str(int(data[key][col].iloc[-1])))
            else:
                vals.append("N/A")
        print(f"  {metric:<22} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")

    print()
    print("=" * 72)


if __name__ == "__main__":
    plot_metabolism()
