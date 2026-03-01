#!/usr/bin/env python3
"""
NietzscheDB — TGC Experiment: Statistical Analysis & Visualization (v2)

Reads telemetry CSVs from the 3 experimental conditions (Normal, Off, Inverted),
computes correlations for both fossil and adaptive metrics, generates 9-panel dashboard.

Input:  experiments/telemetry_lp_Normal.csv, _Off.csv, _Inverted.csv
Output: experiments/tgc_link_prediction_report.png (300 DPI, 9-panel dashboard)

Usage:
  py -3 experiments/analysis_link_prediction.py
"""

import os
import sys
import warnings

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = SCRIPT_DIR

if os.environ.get("DISPLAY") is None and sys.platform != "win32":
    matplotlib.use("Agg")

plt.rcParams.update({
    "figure.figsize": (20, 22),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "font.family": "monospace",
})

COLORS = {
    "Normal": "#4CAF50",
    "Off": "#FF9800",
    "Inverted": "#F44336",
}

LABELS = {
    "Normal": "N: Normal (elite-anchored)",
    "Off": "O: Off (foam orphans)",
    "Inverted": "I: Inverted (anti-anchored)",
}


def load_data():
    data = {}
    for key in ["Normal", "Off", "Inverted"]:
        path = os.path.join(EXPERIMENTS_DIR, f"telemetry_lp_{key}.csv")
        try:
            df = pd.read_csv(path)
            data[key] = df
            print(f"  [{key}] {os.path.basename(path)}: {len(df)} cycles, {len(df.columns)} cols")
        except FileNotFoundError:
            print(f"  [{key}] NOT FOUND: {path}")
    return data


def has_adaptive(data):
    """Check if adaptive metric columns exist."""
    for df in data.values():
        if "auc_synthetic" in df.columns:
            return True
    return False


def compute_correlations(data):
    """Compute Pearson r between TGC_ema and various metrics."""
    results = {}

    for metric_col, label in [("auc", "Fossil AUC"), ("auc_synthetic", "AUC_synth"), ("nca", "NCA")]:
        results[label] = {}
        for key, df in data.items():
            if "tgc_ema" in df.columns and metric_col in df.columns:
                valid = df[["tgc_ema", metric_col]].dropna()
                valid = valid[np.isfinite(valid[metric_col])]
                if len(valid) > 2:
                    r, p = stats.pearsonr(valid["tgc_ema"], valid[metric_col])
                    results[label][key] = {"r": r, "p": p, "n": len(valid)}

        # Pooled
        all_tgc, all_metric = [], []
        for df in data.values():
            if "tgc_ema" in df.columns and metric_col in df.columns:
                valid = df[["tgc_ema", metric_col]].dropna()
                valid = valid[np.isfinite(valid[metric_col])]
                all_tgc.extend(valid["tgc_ema"].tolist())
                all_metric.extend(valid[metric_col].tolist())
        if len(all_tgc) > 2:
            r, p = stats.pearsonr(all_tgc, all_metric)
            results[label]["Pooled"] = {"r": r, "p": p, "n": len(all_tgc)}

    return results


def plot_report(data):
    adaptive = has_adaptive(data)
    nrows = 3 if not adaptive else 4
    fig, axes = plt.subplots(nrows, 3, figsize=(20, 6 * nrows))
    fig.suptitle(
        "NietzscheDB: TGC vs Link Prediction — Adaptive Metrics v2\n"
        r"Hypothesis: $\partial$Performance / $\partial$TGC > 0    |    Fossil + Generative + Semantic",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    # ── Row 0: Fossil AUC, TGC EMA, TGC vs Fossil AUC scatter ──

    # Panel 0,0: Fossil AUC over cycles
    ax = axes[0, 0]
    for key, df in data.items():
        ax.plot(df["cycle"], df["auc"], color=COLORS[key],
                linewidth=2.0, label=LABELS[key], alpha=0.9)
    ax.set_title("Fossil AUC over Cycles (static test edges)")
    ax.set_ylabel("AUC")
    ax.set_xlabel("Cycle")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=7)
    ax.set_ylim(0.3, 1.0)

    # Panel 0,1: TGC EMA over cycles
    ax = axes[0, 1]
    for key, df in data.items():
        ax.plot(df["cycle"], df["tgc_ema"], color=COLORS[key],
                linewidth=2.0, label=LABELS[key], alpha=0.9)
    ax.set_title("TGC (EMA) over Cycles")
    ax.set_ylabel("TGC EMA")
    ax.set_xlabel("Cycle")
    ax.legend(fontsize=7)

    # Panel 0,2: TGC vs Fossil AUC scatter pooled
    ax = axes[0, 2]
    all_tgc, all_auc = [], []
    for key, df in data.items():
        ax.scatter(df["tgc_ema"], df["auc"], color=COLORS[key],
                   alpha=0.3, s=12, label=LABELS[key])
        all_tgc.extend(df["tgc_ema"].tolist())
        all_auc.extend(df["auc"].tolist())
    if len(all_tgc) > 2:
        z = np.polyfit(all_tgc, all_auc, 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(min(all_tgc), max(all_tgc), 100)
        ax.plot(x_range, p_line(x_range), color="black", linewidth=2.5, alpha=0.9)
        r, p = stats.pearsonr(all_tgc, all_auc)
        ax.text(0.05, 0.95, f"r = {r:.4f}\np = {p:.2e}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.set_title("TGC vs Fossil AUC (pooled)")
    ax.set_xlabel("TGC EMA")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=7)

    if adaptive:
        # ── Row 1: Adaptive — Generative Coherence AUC ──

        # Panel 1,0: AUC_synthetic over cycles
        ax = axes[1, 0]
        for key, df in data.items():
            if "auc_synthetic" in df.columns:
                valid = df[df["auc_synthetic"].notna() & np.isfinite(df["auc_synthetic"])]
                ax.plot(valid["cycle"], valid["auc_synthetic"], color=COLORS[key],
                        linewidth=2.0, label=LABELS[key], alpha=0.9)
        ax.set_title("Generative Coherence AUC (edges born this cycle)")
        ax.set_ylabel("AUC_synth")
        ax.set_xlabel("Cycle")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)
        ax.set_ylim(0.3, 1.0)

        # Panel 1,1: n_synthetic_pos over cycles
        ax = axes[1, 1]
        for key, df in data.items():
            if "n_synthetic_pos" in df.columns:
                ax.plot(df["cycle"], df["n_synthetic_pos"], color=COLORS[key],
                        linewidth=1.5, label=LABELS[key], alpha=0.9)
        ax.set_title("New Edges per Cycle (positive samples)")
        ax.set_ylabel("Count")
        ax.set_xlabel("Cycle")
        ax.legend(fontsize=7)

        # Panel 1,2: TGC vs AUC_synthetic scatter
        ax = axes[1, 2]
        all_tgc_s, all_aucs = [], []
        for key, df in data.items():
            if "auc_synthetic" in df.columns:
                valid = df[df["auc_synthetic"].notna() & np.isfinite(df["auc_synthetic"])]
                ax.scatter(valid["tgc_ema"], valid["auc_synthetic"], color=COLORS[key],
                           alpha=0.3, s=12, label=LABELS[key])
                all_tgc_s.extend(valid["tgc_ema"].tolist())
                all_aucs.extend(valid["auc_synthetic"].tolist())
        if len(all_tgc_s) > 2:
            z = np.polyfit(all_tgc_s, all_aucs, 1)
            p_line = np.poly1d(z)
            x_range = np.linspace(min(all_tgc_s), max(all_tgc_s), 100)
            ax.plot(x_range, p_line(x_range), color="black", linewidth=2.5, alpha=0.9)
            r, p = stats.pearsonr(all_tgc_s, all_aucs)
            ax.text(0.05, 0.95, f"r = {r:.4f}\np = {p:.2e}",
                    transform=ax.transAxes, fontsize=10, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.set_title("TGC vs Generative AUC (pooled)")
        ax.set_xlabel("TGC EMA")
        ax.set_ylabel("AUC_synth")
        ax.legend(fontsize=7)

        # ── Row 2: Adaptive — Homophily & NCA ──

        # Panel 2,0: Homophily new edges over cycles
        ax = axes[2, 0]
        for key, df in data.items():
            if "homophily_new" in df.columns:
                valid = df[df["homophily_new"].notna() & np.isfinite(df["homophily_new"])]
                ax.plot(valid["cycle"], valid["homophily_new"], color=COLORS[key],
                        linewidth=2.0, label=f"{key} h(new)", alpha=0.9)
            if "homophily_baseline" in df.columns:
                valid = df[df["homophily_baseline"].notna() & np.isfinite(df["homophily_baseline"])]
                ax.plot(valid["cycle"], valid["homophily_baseline"], color=COLORS[key],
                        linewidth=1.0, linestyle="--", label=f"{key} h(base)", alpha=0.5)
        ax.set_title("Edge Homophily: New Edges vs Baseline")
        ax.set_ylabel("Homophily ratio")
        ax.set_xlabel("Cycle")
        ax.legend(fontsize=6)

        # Panel 2,1: NCA over cycles
        ax = axes[2, 1]
        for key, df in data.items():
            if "nca" in df.columns:
                valid = df[df["nca"].notna() & np.isfinite(df["nca"])]
                ax.plot(valid["cycle"], valid["nca"], color=COLORS[key],
                        linewidth=2.0, label=LABELS[key], alpha=0.9)
        ax.set_title("Node Classification Accuracy (NCA)")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Cycle")
        ax.legend(fontsize=7)

        # Panel 2,2: Delta Homophily over cycles
        ax = axes[2, 2]
        for key, df in data.items():
            if "delta_homophily" in df.columns:
                valid = df[df["delta_homophily"].notna() & np.isfinite(df["delta_homophily"])]
                ax.plot(valid["cycle"], valid["delta_homophily"], color=COLORS[key],
                        linewidth=2.0, label=LABELS[key], alpha=0.9)
        ax.set_title("Delta Homophily (h_new - h_baseline)")
        ax.set_ylabel("Δh")
        ax.set_xlabel("Cycle")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)

        # ── Row 3: Summary bar charts ──
        row_summary = 3
    else:
        row_summary = 1

    # Panel summary,0: Final AUC comparison bars
    ax = axes[row_summary, 0]
    conditions = list(data.keys())
    x = np.arange(len(conditions))
    width = 0.25

    final_fossil = [data[k]["auc"].iloc[-1] for k in conditions]
    bars1 = ax.bar(x - width, final_fossil, width, label="Fossil AUC",
                   color=[COLORS[k] for k in conditions], alpha=0.6, edgecolor="black")

    if adaptive:
        final_synth = []
        for k in conditions:
            v = data[k]["auc_synthetic"].iloc[-1]
            final_synth.append(v if np.isfinite(v) else 0)
        bars2 = ax.bar(x, final_synth, width, label="AUC_synth",
                       color=[COLORS[k] for k in conditions], alpha=0.9, edgecolor="black")

        final_nca = []
        for k in conditions:
            v = data[k]["nca"].iloc[-1]
            final_nca.append(v if np.isfinite(v) else 0)
        bars3 = ax.bar(x + width, final_nca, width, label="NCA",
                       color=[COLORS[k] for k in conditions], alpha=0.4, edgecolor="black",
                       hatch="//")
    else:
        mean_aucs = [data[k]["auc"].mean() for k in conditions]
        ax.bar(x, mean_aucs, width, label="Mean AUC",
               color=[COLORS[k] for k in conditions], alpha=0.5, edgecolor="black")

    ax.set_title("Final Metrics by Condition")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend(fontsize=8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    # Panel summary,1: Structural Entropy
    ax = axes[row_summary, 1]
    for key, df in data.items():
        ax.plot(df["cycle"], df["structural_entropy"], color=COLORS[key],
                linewidth=1.8, alpha=0.9, label=f"{key} Hs")
    ax.set_title("Structural Entropy (Hs)")
    ax.set_ylabel("Shannon Entropy")
    ax.set_xlabel("Cycle")
    ax.legend(fontsize=7)

    # Panel summary,2: GPU/timing
    ax = axes[row_summary, 2]
    for key, df in data.items():
        if "ms_per_cycle" in df.columns:
            ax.plot(df["cycle"], df["ms_per_cycle"], color=COLORS[key],
                    linewidth=1.5, label=f"{key} total", alpha=0.9)
        if "gpu_ms" in df.columns:
            ax.plot(df["cycle"], df["gpu_ms"], color=COLORS[key],
                    linewidth=1.0, linestyle=":", label=f"{key} GPU", alpha=0.6)
    ax.set_title("Cycle Timing (ms)")
    ax.set_ylabel("ms/cycle")
    ax.set_xlabel("Cycle")
    ax.legend(fontsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = os.path.join(EXPERIMENTS_DIR, "tgc_link_prediction_report.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n  Report saved: {output_path}")
    return output_path


def print_summary(data, correlations):
    print()
    print("=" * 78)
    print("  STATISTICAL SUMMARY — Adaptive Metrics v2")
    print("=" * 78)

    adaptive = has_adaptive(data)

    # Per-condition stats
    header = f"  {'Metric':<28} {'Normal':>14} {'Off':>14} {'Inverted':>14}"
    print(header)
    print("  " + "-" * 74)

    def safe_val(df, col, fn):
        try:
            s = df[col].dropna()
            s = s[np.isfinite(s)]
            if len(s) == 0:
                return "N/A"
            return fn(s)
        except Exception:
            return "N/A"

    metrics = [
        ("Final Fossil AUC", lambda df: f"{df['auc'].iloc[-1]:.4f}"),
        ("Mean Fossil AUC", lambda df: f"{df['auc'].mean():.4f}"),
    ]

    if adaptive:
        metrics.extend([
            ("Final AUC_synth", lambda df: safe_val(df, "auc_synthetic", lambda s: f"{s.iloc[-1]:.4f}")),
            ("Mean AUC_synth", lambda df: safe_val(df, "auc_synthetic", lambda s: f"{s.mean():.4f}")),
            ("Mean h(new edges)", lambda df: safe_val(df, "homophily_new", lambda s: f"{s.mean():.4f}")),
            ("Mean Δhomophily", lambda df: safe_val(df, "delta_homophily", lambda s: f"{s.mean():+.4f}")),
            ("Final NCA", lambda df: safe_val(df, "nca", lambda s: f"{s.iloc[-1]:.4f}")),
            ("Mean NCA", lambda df: safe_val(df, "nca", lambda s: f"{s.mean():.4f}")),
        ])

    metrics.extend([
        ("Mean TGC EMA", lambda df: f"{df['tgc_ema'].mean():.6f}"),
        ("Mean Hs", lambda df: f"{df['structural_entropy'].mean():.4f}"),
        ("Mean Eg", lambda df: f"{df['global_efficiency'].mean():.4f}"),
        ("Mean nodes", lambda df: f"{df['total_nodes'].mean():.0f}"),
        ("Mean edges", lambda df: f"{df['total_edges'].mean():.0f}"),
        ("Mean ms/cycle", lambda df: f"{df['ms_per_cycle'].mean():.1f}"),
    ])

    for label, fn in metrics:
        vals = []
        for key in ["Normal", "Off", "Inverted"]:
            if key in data:
                try:
                    vals.append(fn(data[key]))
                except Exception:
                    vals.append("N/A")
            else:
                vals.append("N/A")
        print(f"  {label:<28} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")

    # Correlations
    print()
    for metric_label, conds in correlations.items():
        print(f"  PEARSON r(TGC, {metric_label})")
        print("  " + "-" * 60)
        for key, res in conds.items():
            sig = "***" if res["p"] < 0.001 else "**" if res["p"] < 0.01 else "*" if res["p"] < 0.05 else "ns"
            print(f"    {key:<12} r={res['r']:+.4f}  p={res['p']:.2e}  n={res['n']}  {sig}")
        print()

    # Hypothesis test
    print("  HYPOTHESIS: Normal > Off > Inverted")
    print("  " + "-" * 60)

    if "Normal" in data and "Off" in data and "Inverted" in data:
        tests = [("Fossil AUC", "auc"), ("AUC_synth", "auc_synthetic"), ("NCA", "nca")]
        for test_name, col in tests:
            if col not in data["Normal"].columns:
                continue
            vn = data["Normal"][col].dropna()
            vo = data["Off"][col].dropna()
            vi = data["Inverted"][col].dropna()
            vn = vn[np.isfinite(vn)]
            vo = vo[np.isfinite(vo)]
            vi = vi[np.isfinite(vi)]

            if len(vn) < 2 or len(vo) < 2 or len(vi) < 2:
                continue

            fn = vn.iloc[-1]
            fo = vo.iloc[-1]
            fi = vi.iloc[-1]

            u_no, p_no = stats.mannwhitneyu(vn, vo, alternative="greater")
            u_oi, p_oi = stats.mannwhitneyu(vo, vi, alternative="greater")

            holds = fn > fo and fo > fi
            print(f"    {test_name}:")
            print(f"      Final: N={fn:.4f}  O={fo:.4f}  I={fi:.4f}  {'N>O>I' if holds else 'NOT N>O>I'}")
            print(f"      MW U(N>O): p={p_no:.2e}  MW U(O>I): p={p_oi:.2e}")
            print()

        # Homophily Delta (higher = better, average across cycles)
        if "delta_homophily" in data["Normal"].columns:
            dh_n = data["Normal"]["delta_homophily"].dropna().mean()
            dh_o = data["Off"]["delta_homophily"].dropna().mean()
            dh_i = data["Inverted"]["delta_homophily"].dropna().mean()
            holds = dh_n > dh_o and dh_o > dh_i
            print(f"    Δhomophily:")
            print(f"      Mean: N={dh_n:+.4f}  O={dh_o:+.4f}  I={dh_i:+.4f}  {'N>O>I' if holds else 'NOT N>O>I'}")
            print()

    print("=" * 78)


def main():
    print("=" * 78)
    print("  NietzscheDB: TGC vs Link Prediction — Adaptive Metrics v2 Analysis")
    print("=" * 78)
    print()

    data = load_data()
    if not data:
        print("  No telemetry CSVs found. Run the experiment first:")
        print("    cargo run --release --bin experiment_link_prediction --features cuda")
        return

    correlations = compute_correlations(data)
    plot_report(data)
    print_summary(data, correlations)


if __name__ == "__main__":
    main()
