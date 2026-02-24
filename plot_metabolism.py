#!/usr/bin/env python3
"""
NietzscheDB — Autopsia do Metabolismo Cognitivo

Generates a 4-quadrant clinical dashboard from the Nezhmetdinov
Forgetting Engine simulation telemetry.

Usage:
    cargo run --release --bin simulate_forgetting
    python plot_metabolism.py

Input:  forgetting_telemetry.csv
Output: metabolism_report.png (300 DPI)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

# Use non-interactive backend if no display
if os.environ.get("DISPLAY") is None and sys.platform != "win32":
    matplotlib.use("Agg")

# Clinical visual style
plt.rcParams.update({
    "figure.figsize": (14, 10),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


def plot_metabolism():
    print("=" * 60)
    print("  NietzscheDB: Autopsia do Metabolismo Cognitivo")
    print("=" * 60)
    print()

    csv_path = "forgetting_telemetry.csv"
    print(f"Lendo {csv_path}...")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: {csv_path} nao encontrado.")
        print("Rode primeiro: cargo run --release --bin simulate_forgetting")
        return

    print(f"  Ciclos: {len(df)}")
    print(f"  Colunas: {list(df.columns)}")
    print()

    fig, axes = plt.subplots(2, 2)
    fig.suptitle(
        "NietzscheDB: Autopsia do Metabolismo Cognitivo (500 Ciclos)",
        fontsize=16,
        fontweight="bold",
    )

    # ── Q1: TGC (Topological Generative Capacity) ──
    ax1 = axes[0, 0]
    ax1.plot(df["cycle"], df["tgc"], color="#2196F3", linewidth=2)
    if len(df) > 0:
        baseline = df["tgc"].iloc[0]
        ax1.axhline(y=baseline, color="red", linestyle="--", alpha=0.5, label=f"Baseline={baseline:.3f}")
    ax1.set_title("TGC (Capacidade Generativa Topologica)")
    ax1.set_ylabel("Score TGC")
    ax1.set_xlabel("Ciclo")
    ax1.legend()

    # ── Q2: Vitality Variance (Cognitive Health) ──
    ax2 = axes[0, 1]
    ax2.plot(df["cycle"], df["variance_vitality"], color="#9C27B0", linewidth=2)
    ax2.axhline(y=0.05, color="red", linestyle=":", alpha=0.5, label="Risco Monocultura (0.05)")
    ax2.axhline(y=0.25, color="orange", linestyle=":", alpha=0.5, label="Risco Caos (0.25)")
    ax2.set_title("Var(V) - Variancia de Vitalidade")
    ax2.set_ylabel("Variancia")
    ax2.set_xlabel("Ciclo")
    ax2.legend(fontsize=9)

    # ── Q3: Elite Drift (Identity Preservation) ──
    ax3 = axes[1, 0]
    ax3.plot(df["cycle"], df["elite_drift"], color="#FF9800", linewidth=2)
    ax3.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Limiar Identidade (0.1)")
    ax3.set_title("Elite Drift (Desvio de Identidade)")
    ax3.set_ylabel("Distancia Euclidiana")
    ax3.set_xlabel("Ciclo")
    ax3.legend()

    # ── Q4: Sacrifices per Cycle (The Guillotine) ──
    ax4 = axes[1, 1]
    ax4.bar(df["cycle"], df["sacrificed"], color="#F44336", alpha=0.7, width=1.0)
    ax4.set_title("Sacrificios por Ciclo (Hard Deletes)")
    ax4.set_ylabel("Nos Deletados")
    ax4.set_xlabel("Ciclo")

    plt.tight_layout()
    output_path = "metabolism_report.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Dashboard gerado: {output_path}")
    print()

    # ── Clinical Summary ──
    print("=" * 60)
    print("  AVALIACAO CLINICA")
    print("=" * 60)

    # 1. TGC trend
    if len(df) > 10:
        tgc_start = df["tgc"].iloc[:10].mean()
        tgc_end = df["tgc"].iloc[-10:].mean()
        if tgc_end > tgc_start:
            print(f"  [OK] TGC SUBIU: {tgc_start:.4f} -> {tgc_end:.4f} (fertilidade estrutural)")
        else:
            print(f"  [!!] TGC DESCEU: {tgc_start:.4f} -> {tgc_end:.4f} (sistema empobreceu)")

    # 2. Variance plateau
    if len(df) > 10:
        var_end = df["variance_vitality"].iloc[-10:].mean()
        if var_end < 0.001:
            print(f"  [!!] Var(V) ESMAGOU em zero ({var_end:.6f}): COLAPSO ELITISTA")
        elif var_end < 0.05:
            print(f"  [OK] Var(V) baixa mas estavel ({var_end:.4f}): monocultura controlada")
        else:
            print(f"  [OK] Var(V) saudavel ({var_end:.4f}): diversidade cognitiva mantida")

    # 3. Elite drift
    if len(df) > 0:
        drift_final = df["elite_drift"].iloc[-1]
        if drift_final < 0.05:
            print(f"  [OK] Elite Drift minimo ({drift_final:.4f}): identidade preservada")
        elif drift_final < 0.1:
            print(f"  [OK] Elite Drift aceitavel ({drift_final:.4f}): identidade estavel")
        else:
            print(f"  [!!] Elite Drift alto ({drift_final:.4f}): identidade comprometida")

    # 4. Sacrifice convergence
    if len(df) > 20:
        early_sacr = df["sacrificed"].iloc[:20].mean()
        late_sacr = df["sacrificed"].iloc[-20:].mean()
        if late_sacr < early_sacr * 0.1:
            print(f"  [OK] Sacrificios convergem a zero: {early_sacr:.0f}/ciclo -> {late_sacr:.0f}/ciclo")
        elif late_sacr < early_sacr * 0.5:
            print(f"  [OK] Sacrificios diminuiram: {early_sacr:.0f}/ciclo -> {late_sacr:.0f}/ciclo")
        else:
            print(f"  [!!] Sacrificios NAO convergiram: {early_sacr:.0f}/ciclo -> {late_sacr:.0f}/ciclo")

    # 5. False positives
    if len(df) > 0:
        fp = df["signal_killed"].iloc[-1]
        if fp == 0:
            print("  [OK] ZERO falsos positivos: elites 100% preservadas")
        else:
            print(f"  [!!] {fp} falsos positivos: elites danificadas")

    print()
    print("=" * 60)


if __name__ == "__main__":
    plot_metabolism()
