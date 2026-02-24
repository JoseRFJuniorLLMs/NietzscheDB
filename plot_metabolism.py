#!/usr/bin/env python3
"""
NietzscheDB — Protocolo Experimental: Termodinamica do Esquecimento

Compares 3 experimental variations:
  A: DELETE_ONLY   (control — pure catabolism)
  B: LOW_GEN       (~30% void-seeded generation)
  C: MATCHED_GEN   (~100% void-seeded generation)

Input:  telemetry_A_delete_only.csv, telemetry_B_low_gen.csv, telemetry_C_matched_gen.csv
Output: metabolism_report.png (300 DPI, 4-quadrant x 3 overlaid)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

if os.environ.get("DISPLAY") is None and sys.platform != "win32":
    matplotlib.use("Agg")

plt.rcParams.update({
    "figure.figsize": (16, 12),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})

COLORS = {
    "A": "#F44336",  # Red — delete only
    "B": "#2196F3",  # Blue — low gen
    "C": "#4CAF50",  # Green — matched gen
}


def load_data():
    files = {
        "A": "telemetry_A_delete_only.csv",
        "B": "telemetry_B_low_gen.csv",
        "C": "telemetry_C_matched_gen.csv",
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
    print("  NietzscheDB: Protocolo Experimental (3 Variacoes)")
    print("=" * 64)
    print()

    data = load_data()
    if not data:
        print("Nenhum CSV encontrado. Rode o simulador Rust primeiro.")
        return

    fig, axes = plt.subplots(2, 2)
    fig.suptitle(
        "Termodinamica do Esquecimento Ativo (500 Ciclos x 3 Experimentos)",
        fontsize=16,
        fontweight="bold",
    )

    labels = {
        "A": "A: Delete Only",
        "B": "B: Low Gen (30%)",
        "C": "C: Matched Gen (100%)",
    }

    # ── Q1: TGC ──
    ax1 = axes[0, 0]
    for key, df in data.items():
        ax1.plot(df["cycle"], df["tgc"], color=COLORS[key], linewidth=1.8,
                 label=labels[key], alpha=0.9)
    ax1.set_title("TGC (Capacidade Generativa Topologica)")
    ax1.set_ylabel("Score TGC (EMA)")
    ax1.set_xlabel("Ciclo")
    ax1.legend(fontsize=9)

    # ── Q2: Var(V) ──
    ax2 = axes[0, 1]
    for key, df in data.items():
        ax2.plot(df["cycle"], df["variance_vitality"], color=COLORS[key],
                 linewidth=1.8, label=labels[key], alpha=0.9)
    ax2.axhline(y=0.05, color="black", linestyle=":", alpha=0.4, label="Monocultura (0.05)")
    ax2.axhline(y=0.25, color="black", linestyle="--", alpha=0.4, label="Caos (0.25)")
    ax2.set_title("Var(V) - Variancia de Vitalidade")
    ax2.set_ylabel("Variancia")
    ax2.set_xlabel("Ciclo")
    ax2.legend(fontsize=8)

    # ── Q3: Elite Drift ──
    ax3 = axes[1, 0]
    for key, df in data.items():
        ax3.plot(df["cycle"], df["elite_drift"], color=COLORS[key],
                 linewidth=1.8, label=labels[key], alpha=0.9)
    ax3.axhline(y=0.1, color="black", linestyle="--", alpha=0.4, label="Limiar (0.1)")
    ax3.set_title("Elite Drift (Desvio de Identidade)")
    ax3.set_ylabel("Distancia Euclidiana")
    ax3.set_xlabel("Ciclo")
    ax3.legend(fontsize=9)

    # ── Q4: Sacrifices per Cycle ──
    ax4 = axes[1, 1]
    for key, df in data.items():
        ax4.plot(df["cycle"], df["sacrificed"], color=COLORS[key],
                 linewidth=1.2, label=labels[key], alpha=0.7)
    ax4.set_title("Sacrificios por Ciclo (Hard Deletes)")
    ax4.set_ylabel("Nos Deletados")
    ax4.set_xlabel("Ciclo")
    ax4.legend(fontsize=9)

    plt.tight_layout()
    output_path = "metabolism_report.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print()
    print(f"Dashboard gerado: {output_path}")
    print()

    # ── Raw Summary Table ──
    print("=" * 64)
    print("  COMPARACAO: MEDIA DOS ULTIMOS 100 CICLOS")
    print("=" * 64)
    header = f"{'Metrica':<20} {'A:DelOnly':>12} {'B:LowGen':>12} {'C:Matched':>12}"
    print(header)
    print("-" * len(header))

    metrics = [
        ("avg_TGC", "tgc"),
        ("avg_Var(V)", "variance_vitality"),
        ("avg_Elite_Drift", "elite_drift"),
        ("avg_Sacrificed", "sacrificed"),
        ("avg_Created", "nodes_created"),
        ("avg_Mean_V", "mean_vitality"),
        ("avg_Mean_E", "mean_energy"),
        ("avg_Nodes", "total_nodes"),
    ]

    for label, col in metrics:
        vals = []
        for key in ["A", "B", "C"]:
            if key in data and col in data[key].columns:
                last100 = data[key][col].iloc[-100:]
                vals.append(f"{last100.mean():.4f}")
            else:
                vals.append("N/A")
        print(f"  {label:<18} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # FP row
    fp_vals = []
    for key in ["A", "B", "C"]:
        if key in data:
            fp_vals.append(str(int(data[key]["signal_killed"].iloc[-1])))
        else:
            fp_vals.append("N/A")
    print(f"  {'total_FP':<18} {fp_vals[0]:>12} {fp_vals[1]:>12} {fp_vals[2]:>12}")

    print()
    print("=" * 64)


if __name__ == "__main__":
    plot_metabolism()
