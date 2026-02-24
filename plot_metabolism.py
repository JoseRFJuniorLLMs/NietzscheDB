#!/usr/bin/env python3
"""
NietzscheDB — Protocolo Experimental: O Sistema Sabe Nascer?

Compares 3 birth modes (isolated variable):
  D: FOAM          (void-born orphans, kappa=0 — structural foam baseline)
  E: ANCHORED      (elite-parented, kappa=2, no polarization)
  F: DIALECTICAL   (elite-parented, kappa=2, entropy polarization)

Input:  telemetry_D_foam.csv, telemetry_E_anchored.csv, telemetry_F_dialectical.csv
Output: metabolism_report.png (300 DPI, 6-panel dashboard)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

if os.environ.get("DISPLAY") is None and sys.platform != "win32":
    matplotlib.use("Agg")

plt.rcParams.update({
    "figure.figsize": (18, 14),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})

COLORS = {
    "D": "#F44336",  # Red — foam baseline
    "E": "#2196F3",  # Blue — anchored only
    "F": "#4CAF50",  # Green — dialectical (full Option A)
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
    print("  NietzscheDB: O Sistema Sabe Nascer? (3 Birth Modes)")
    print("=" * 64)
    print()

    data = load_data()
    if not data:
        print("Nenhum CSV encontrado. Rode o simulador Rust primeiro.")
        return

    fig, axes = plt.subplots(3, 2)
    fig.suptitle(
        "O Sistema Sabe Nascer? — 500 Ciclos x 3 Modos de Nascimento",
        fontsize=16,
        fontweight="bold",
    )

    labels = {
        "D": "D: Foam (orphans, k=0)",
        "E": "E: Anchored (k=2, no pol.)",
        "F": "F: Dialectical (k=2 + pol.)",
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
    ax2.set_title("Var(V) — Variancia de Vitalidade")
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

    # ── Q5: Newborn Survival Rate (5th vital sign) ──
    ax5 = axes[2, 0]
    for key, df in data.items():
        ax5.plot(df["cycle"], df["newborn_survival"], color=COLORS[key],
                 linewidth=1.8, label=labels[key], alpha=0.9)
    ax5.axhline(y=0.5, color="black", linestyle=":", alpha=0.4, label="50% survival")
    ax5.set_title("Newborn Survival Rate (Sobrevivencia Neonatal)")
    ax5.set_ylabel("Fracao sobrevivente")
    ax5.set_xlabel("Ciclo")
    ax5.set_ylim(-0.05, 1.10)
    ax5.legend(fontsize=9)

    # ── Q6: Mean Energy ──
    ax6 = axes[2, 1]
    for key, df in data.items():
        ax6.plot(df["cycle"], df["mean_energy"], color=COLORS[key],
                 linewidth=1.8, label=labels[key], alpha=0.9)
    ax6.set_title("Energia Media (Reserva Metabolica)")
    ax6.set_ylabel("Mean Energy")
    ax6.set_xlabel("Ciclo")
    ax6.legend(fontsize=9)

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
    header = f"{'Metrica':<22} {'D:Foam':>12} {'E:Anchored':>12} {'F:Dialect.':>12}"
    print(header)
    print("-" * len(header))

    metrics = [
        ("avg_TGC", "tgc"),
        ("avg_Var(V)", "variance_vitality"),
        ("avg_Elite_Drift", "elite_drift"),
        ("avg_Sacrificed", "sacrificed"),
        ("avg_Created", "nodes_created"),
        ("avg_Newborn_Surv", "newborn_survival"),
        ("avg_Mean_V", "mean_vitality"),
        ("avg_Mean_E", "mean_energy"),
        ("avg_Nodes", "total_nodes"),
    ]

    for label, col in metrics:
        vals = []
        for key in ["D", "E", "F"]:
            if key in data and col in data[key].columns:
                last100 = data[key][col].iloc[-100:]
                vals.append(f"{last100.mean():.4f}")
            else:
                vals.append("N/A")
        print(f"  {label:<20} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # FP row
    fp_vals = []
    for key in ["D", "E", "F"]:
        if key in data:
            fp_vals.append(str(int(data[key]["signal_killed"].iloc[-1])))
        else:
            fp_vals.append("N/A")
    print(f"  {'total_FP':<20} {fp_vals[0]:>12} {fp_vals[1]:>12} {fp_vals[2]:>12}")

    # Noise killed
    nk_vals = []
    for key in ["D", "E", "F"]:
        if key in data:
            nk_vals.append(str(int(data[key]["noise_killed"].iloc[-1])))
        else:
            nk_vals.append("N/A")
    print(f"  {'total_noise_del':<20} {nk_vals[0]:>12} {nk_vals[1]:>12} {nk_vals[2]:>12}")

    print()
    print("=" * 64)


if __name__ == "__main__":
    plot_metabolism()
