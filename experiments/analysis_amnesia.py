#!/usr/bin/env python3
"""
NietzscheDB — Controlled Amnesia Experiment: Analysis & Visualization

Reads telemetry CSVs from the 3 experimental conditions (Normal, Off, Inverted),
plots the sculpt-amnesia-recovery curves, and generates a 4-panel dashboard.

Input:  experiments/telemetry_amnesia_Normal.csv, _Off.csv, _Inverted.csv
Output: experiments/amnesia_dashboard.png (300 DPI, 4-panel dashboard)

Usage:
  py -3 experiments/analysis_amnesia.py
"""

import os
import sys
import warnings

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = SCRIPT_DIR

if os.environ.get("DISPLAY") is None and sys.platform != "win32":
    matplotlib.use("Agg")

COLORS = {"Normal": "#2196F3", "Off": "#FF9800", "Inverted": "#F44336"}
SCULPT_CYCLES = 80

def load_data():
    dfs = {}
    for mode in ["Normal", "Off", "Inverted"]:
        path = os.path.join(EXPERIMENTS_DIR, f"telemetry_amnesia_{mode}.csv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {mode}")
            continue
        df = pd.read_csv(path)
        dfs[mode] = df
    return dfs

def compute_global_cycle(df):
    """Assign a global x-axis: sculpt cycles 1-80, then amnesia=81, then recovery 82-131."""
    x = []
    for _, row in df.iterrows():
        if row["phase"] == "sculpt":
            x.append(int(row["cycle"]))
        elif row["phase"] == "amnesia":
            x.append(SCULPT_CYCLES + 1)
        elif row["phase"] == "recovery":
            x.append(SCULPT_CYCLES + 1 + int(row["cycle"]))
    return x

def main():
    print("=" * 60)
    print("  NietzscheDB — Controlled Amnesia: Analysis Dashboard")
    print("=" * 60)

    dfs = load_data()
    if not dfs:
        print("  ERROR: No data files found. Run the experiment first.")
        sys.exit(1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("NietzscheDB — Controlled Amnesia: Topology is the True Memory",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── Panel 1: NCA over time (sculpt + amnesia + recovery) ──
    ax = axes[0, 0]
    for mode, df in dfs.items():
        x = compute_global_cycle(df)
        nca = df["nca"].values
        ax.plot(x, nca, color=COLORS[mode], label=mode, linewidth=1.5)
        # Horizontal dashed: pre-amnesia NCA
        sculpt = df[df["phase"] == "sculpt"]
        if len(sculpt) > 0:
            pre_nca = sculpt["nca"].iloc[-1]
            ax.axhline(y=pre_nca, color=COLORS[mode], linestyle=":", alpha=0.5, linewidth=0.8)

    ax.axvline(x=SCULPT_CYCLES + 1, color="red", linestyle="--", linewidth=2, alpha=0.7, label="AMNESIA")
    ax.set_xlabel("Global Cycle")
    ax.set_ylabel("NCA (Node Classification Accuracy)")
    ax.set_title("NCA: Sculpt → Amnesia → Recovery")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Mean embedding norm during recovery ──
    ax = axes[0, 1]
    for mode, df in dfs.items():
        rec = df[df["phase"] == "recovery"]
        if len(rec) > 0:
            ax.plot(rec["cycle"].values, rec["mean_norm"].values,
                    color=COLORS[mode], label=mode, linewidth=1.5)
    ax.set_xlabel("Recovery Cycle")
    ax.set_ylabel("Mean ||embedding||")
    ax.set_title("Signal Growth from Origin (Recovery Phase)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Cosine recovery curve ──
    ax = axes[1, 0]
    for mode, df in dfs.items():
        rec = df[df["phase"] == "recovery"]
        if len(rec) > 0:
            cos_vals = rec["cosine_recovery"].values
            ax.plot(rec["cycle"].values, cos_vals,
                    color=COLORS[mode], label=mode, linewidth=1.5)
    ax.set_xlabel("Recovery Cycle")
    ax.set_ylabel("Cosine Similarity to Pre-Amnesia")
    ax.set_title("Cosine Recovery (convergence to pre-amnesia structure)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    # ── Panel 4: Recovery% bar chart (final verdict) ──
    ax = axes[1, 1]
    modes_found = []
    recovery_pcts = []
    bar_colors = []

    for mode in ["Normal", "Off", "Inverted"]:
        if mode not in dfs:
            continue
        df = dfs[mode]
        sculpt = df[df["phase"] == "sculpt"]
        amnesia = df[df["phase"] == "amnesia"]
        recovery = df[df["phase"] == "recovery"]

        if len(sculpt) == 0 or len(amnesia) == 0 or len(recovery) == 0:
            continue

        nca_pre = sculpt["nca"].iloc[-1]
        nca_post0 = amnesia["nca"].iloc[0]
        nca_rec = recovery["nca"].iloc[-1]

        denom = nca_pre - nca_post0
        if abs(denom) > 1e-6:
            pct = (nca_rec - nca_post0) / denom * 100.0
        else:
            pct = 0.0

        modes_found.append(mode)
        recovery_pcts.append(pct)
        bar_colors.append(COLORS[mode])

    bars = ax.bar(modes_found, recovery_pcts, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, pct in zip(bars, recovery_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Recovery %")
    ax.set_title("NCA Recovery % (Final Verdict)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(EXPERIMENTS_DIR, "amnesia_dashboard.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n  Dashboard saved: {out_path}")

    # Print verdict table
    print()
    print("  VERDICT TABLE:")
    print(f"  {'Mode':12} | {'NCA_pre':>8} | {'NCA_post0':>9} | {'NCA_rec50':>9} | {'Recovery%':>9}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for mode in ["Normal", "Off", "Inverted"]:
        if mode not in dfs:
            continue
        df = dfs[mode]
        sculpt = df[df["phase"] == "sculpt"]
        amnesia = df[df["phase"] == "amnesia"]
        recovery = df[df["phase"] == "recovery"]
        if len(sculpt) == 0 or len(amnesia) == 0 or len(recovery) == 0:
            continue
        nca_pre = sculpt["nca"].iloc[-1]
        nca_post0 = amnesia["nca"].iloc[0]
        nca_rec = recovery["nca"].iloc[-1]
        denom = nca_pre - nca_post0
        pct = (nca_rec - nca_post0) / denom * 100.0 if abs(denom) > 1e-6 else 0.0
        print(f"  {mode:12} | {nca_pre:8.4f} | {nca_post0:9.4f} | {nca_rec:9.4f} | {pct:8.1f}%")
    print()

if __name__ == "__main__":
    main()
