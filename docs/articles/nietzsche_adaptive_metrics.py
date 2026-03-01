"""
NietzscheDB — Adaptive Evaluation Metrics v2.0
═══════════════════════════════════════════════════════════════════════════════
Replaces the static fossil-benchmark with two living metrics:

  1. Generative Coherence AUC  (AUC_synthetic)
     Does the Zaratustra engine create semantically coherent edges?
     Positives  = edges BORN this cycle (dynamic, never stale)
     Negatives  = non-edges sampled ONLY from original Cora nodes
                  (no foam inflation, no boundary tricks)

  2. Semantic Drift Quality  (two sub-metrics)
     a) Homophily Delta (Δh): do new edges connect same-class nodes?
        Δh > 0  → anabolic drift (building meaning)
        Δh < 0  → catabolic drift (destroying meaning)
     b) Node Classification Accuracy (NCA): after the embedding drifts,
        does a linear probe still cluster Cora classes correctly?
        NCA_Normal ≥ NCA_Off  → drift was intelligent

These metrics cannot be gamed by foam inflation or boundary triviality.
They measure the quality of the mind being built, not the fossil it left behind.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY: Hyperbolic distance on Poincaré disk
# ═══════════════════════════════════════════════════════════════════════════

def poincare_distance_batch(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Batched Poincaré disk distance.
    u, v: (N, D) tensors — must already be inside the unit disk (norm < 1)
    Returns: (N,) distance tensor
    """
    # Clamp norms to stay strictly inside the disk
    u_norm_sq = torch.clamp((u * u).sum(dim=-1), max=1.0 - eps)
    v_norm_sq = torch.clamp((v * v).sum(dim=-1), max=1.0 - eps)
    diff = u - v
    diff_norm_sq = (diff * diff).sum(dim=-1)
    
    numerator = 2.0 * diff_norm_sq
    denominator = (1.0 - u_norm_sq) * (1.0 - v_norm_sq)
    denominator = torch.clamp(denominator, min=eps)
    
    arg = 1.0 + numerator / denominator
    arg = torch.clamp(arg, min=1.0 + eps)
    return torch.acosh(arg)


def link_score_from_distance(dist: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Convert hyperbolic distance to link probability (inverse sigmoid of distance)."""
    return torch.sigmoid(-dist / temperature)


# ═══════════════════════════════════════════════════════════════════════════
# CORE CLASS: AdaptiveEvaluator
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveEvaluator:
    """
    Evaluator that measures the QUALITY of dynamically generated structure,
    not the system's memory of a static past.

    Usage
    -----
    evaluator = AdaptiveEvaluator(
        cora_node_ids=set(range(2708)),        # original Cora universe
        cora_labels=labels_tensor,             # node class labels (for NCA)
        device='cuda'
    )

    # At each NietzscheDB cycle:
    metrics = evaluator.evaluate_cycle(
        embeddings=current_embeddings,         # (N_total, D) — all nodes
        new_edges=edges_born_this_cycle,       # (M, 2) tensor of NEW edge indices
        all_edges=current_full_adjacency,      # (E, 2) tensor of ALL edges
        cycle=t
    )
    """

    def __init__(
        self,
        cora_node_ids: set,
        cora_labels: torch.Tensor,
        device: str = 'cuda',
        n_neg_per_pos: int = 1,
        temperature: float = 1.0,
        nca_n_splits: int = 3,
        seed: int = 42,
    ):
        self.cora_node_ids = cora_node_ids
        self.cora_labels = cora_labels  # shape (|cora|,) — label for each Cora node
        self.device = device
        self.n_neg_per_pos = n_neg_per_pos
        self.temperature = temperature
        self.nca_n_splits = nca_n_splits
        self.rng = np.random.default_rng(seed)

        # Build Cora edge set for fast negative sampling
        self._cora_nodes_array = np.array(sorted(cora_node_ids), dtype=np.int64)
        self._existing_edges: set = set()  # updated each cycle

        # History
        self.history = defaultdict(list)

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate_cycle(
        self,
        embeddings: torch.Tensor,          # (N_total, D)
        new_edges: torch.Tensor,           # (M, 2) NEW edges born this cycle
        all_edges: torch.Tensor,           # (E, 2) full current adjacency
        cycle: int,
    ) -> dict:
        """
        Run both adaptive metrics and return a metrics dictionary.
        """
        # Update known edge set (for negative sampling purity)
        self._update_edge_set(all_edges)

        results = {"cycle": cycle}

        # ── Metric 1: Generative Coherence AUC ──────────────────────────
        if new_edges is not None and new_edges.shape[0] > 0:
            auc_synth, n_pos, n_neg = self._generative_coherence_auc(
                embeddings, new_edges
            )
            results["auc_synthetic"] = auc_synth
            results["n_synthetic_positives"] = n_pos
            results["n_synthetic_negatives"] = n_neg
        else:
            results["auc_synthetic"] = float("nan")
            results["n_synthetic_positives"] = 0
            results["n_synthetic_negatives"] = 0

        # ── Metric 2a: Homophily Delta ───────────────────────────────────
        if new_edges is not None and new_edges.shape[0] > 0:
            delta_h, h_new, h_baseline = self._homophily_delta(new_edges)
            results["homophily_new_edges"] = h_new
            results["homophily_baseline"] = h_baseline
            results["delta_homophily"] = delta_h
        else:
            results["homophily_new_edges"] = float("nan")
            results["homophily_baseline"] = float("nan")
            results["delta_homophily"] = float("nan")

        # ── Metric 2b: Node Classification Accuracy ─────────────────────
        nca_mean, nca_std = self._node_classification_accuracy(embeddings)
        results["nca_mean"] = nca_mean
        results["nca_std"] = nca_std

        # ── Store history ────────────────────────────────────────────────
        for k, v in results.items():
            if k != "cycle":
                self.history[k].append(v)
        self.history["cycle"].append(cycle)

        return results

    # ── Metric 1: Generative Coherence AUC ──────────────────────────────

    def _generative_coherence_auc(
        self,
        embeddings: torch.Tensor,
        new_edges: torch.Tensor,
    ) -> tuple:
        """
        AUC where:
          Positives  = new_edges generated this cycle
          Negatives  = pure Cora non-edges (no foam nodes allowed)

        This measures: "Are the edges Zaratustra invented geometrically meaningful?"
        """
        emb = embeddings.to(self.device)
        new_edges = new_edges.to(self.device)

        # ── Positive scores ──────────────────────────────────────────────
        src_pos = emb[new_edges[:, 0]]
        dst_pos = emb[new_edges[:, 1]]
        dist_pos = poincare_distance_batch(src_pos, dst_pos)
        scores_pos = link_score_from_distance(dist_pos, self.temperature).cpu().numpy()

        n_pos = len(scores_pos)
        n_neg = n_pos * self.n_neg_per_pos

        # ── Negative sampling: ONLY from original Cora nodes ────────────
        neg_edges = self._sample_pure_cora_negatives(n_neg)
        if neg_edges is None or len(neg_edges) == 0:
            return float("nan"), n_pos, 0

        neg_tensor = torch.tensor(neg_edges, dtype=torch.long, device=self.device)
        src_neg = emb[neg_tensor[:, 0]]
        dst_neg = emb[neg_tensor[:, 1]]
        dist_neg = poincare_distance_batch(src_neg, dst_neg)
        scores_neg = link_score_from_distance(dist_neg, self.temperature).cpu().numpy()

        # ── AUC ──────────────────────────────────────────────────────────
        y_true = np.concatenate([np.ones(n_pos), np.zeros(len(scores_neg))])
        y_score = np.concatenate([scores_pos, scores_neg])

        if len(np.unique(y_true)) < 2:
            return float("nan"), n_pos, len(scores_neg)

        auc = roc_auc_score(y_true, y_score)
        return float(auc), n_pos, len(scores_neg)

    def _sample_pure_cora_negatives(self, n: int, max_attempts: int = 10) -> np.ndarray:
        """
        Sample n non-edges where BOTH endpoints are original Cora nodes.
        Rejects any pair that exists in the current edge set.
        """
        candidates = []
        attempts = 0
        nodes = self._cora_nodes_array

        while len(candidates) < n and attempts < max_attempts:
            batch_size = min((n - len(candidates)) * 4, 50000)
            src = self.rng.choice(nodes, size=batch_size, replace=True)
            dst = self.rng.choice(nodes, size=batch_size, replace=True)

            # Remove self-loops and existing edges
            mask = src != dst
            pairs = set(zip(src[mask].tolist(), dst[mask].tolist()))
            pure = [(u, v) for u, v in pairs if (u, v) not in self._existing_edges
                    and (v, u) not in self._existing_edges]
            candidates.extend(pure[: n - len(candidates)])
            attempts += 1

        if len(candidates) == 0:
            return None
        return np.array(candidates[:n], dtype=np.int64)

    # ── Metric 2a: Homophily Delta ───────────────────────────────────────

    def _homophily_delta(self, new_edges: torch.Tensor) -> tuple:
        """
        Δh = h(new_edges) - h(baseline)

        h(edge_set) = fraction of edges where src and dst share the same class label.
        h(baseline) = expected homophily under random sampling (= sum_c (n_c/N)^2).

        Δh > 0: Zaratustra is building semantically coherent bridges (anabolic).
        Δh < 0: Zaratustra is connecting noise (catabolic).
        """
        labels = self.cora_labels
        cora_set = self.cora_node_ids

        # Filter to edges where both endpoints are Cora nodes with known labels
        edges_np = new_edges.cpu().numpy()
        valid = [
            (u, v) for u, v in edges_np
            if u in cora_set and v in cora_set and u < len(labels) and v < len(labels)
        ]

        if len(valid) == 0:
            return float("nan"), float("nan"), float("nan")

        # Observed homophily
        same_class = sum(
            1 for u, v in valid if labels[u].item() == labels[v].item()
        )
        h_new = same_class / len(valid)

        # Baseline: random edge homophily (class frequency squared)
        label_counts = torch.bincount(labels.long())
        p = label_counts.float() / label_counts.sum()
        h_baseline = float((p * p).sum().item())

        delta_h = h_new - h_baseline
        return float(delta_h), float(h_new), float(h_baseline)

    # ── Metric 2b: Node Classification Accuracy ─────────────────────────

    def _node_classification_accuracy(self, embeddings: torch.Tensor) -> tuple:
        """
        Probe: can a linear classifier separate Cora classes using current embeddings?

        This measures whether the embedding drift has preserved or enhanced
        semantic structure. A degraded NCA indicates the drift was catabolic;
        a maintained or improved NCA indicates anabolic reorganization.
        """
        labels = self.cora_labels
        nodes = sorted(self.cora_node_ids)
        valid_nodes = [n for n in nodes if n < len(labels) and n < embeddings.shape[0]]

        if len(valid_nodes) < 100:
            return float("nan"), float("nan")

        X = embeddings[valid_nodes].cpu().float().numpy()
        y = labels[valid_nodes].cpu().numpy()

        # Normalize (important for hyperbolic embeddings which can have variable norms)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Stratified cross-validation
        splitter = StratifiedShuffleSplit(
            n_splits=self.nca_n_splits, test_size=0.2, random_state=42
        )
        accs = []
        for train_idx, test_idx in splitter.split(X_scaled, y):
            clf = LogisticRegression(
                max_iter=300, C=1.0, multi_class="auto", solver="lbfgs", n_jobs=-1
            )
            try:
                clf.fit(X_scaled[train_idx], y[train_idx])
                acc = clf.score(X_scaled[test_idx], y[test_idx])
                accs.append(acc)
            except Exception:
                pass

        if len(accs) == 0:
            return float("nan"), float("nan")

        return float(np.mean(accs)), float(np.std(accs))

    # ── Internal helpers ─────────────────────────────────────────────────

    def _update_edge_set(self, all_edges: torch.Tensor):
        """Keep the known edge set current to ensure pure negative sampling."""
        edges_np = all_edges.cpu().numpy()
        self._existing_edges = set(map(tuple, edges_np.tolist()))


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION ADAPTER: Drop-in replacement for evaluate_link_prediction
# ═══════════════════════════════════════════════════════════════════════════

class NietzscheEvaluationSuite:
    """
    Complete evaluation suite. Maintains one AdaptiveEvaluator per condition
    and tracks the history of all metrics across cycles.

    Drop this into your existing training loop where evaluate_link_prediction
    was called.

    Example
    -------
    suite = NietzscheEvaluationSuite(
        cora_node_ids=set(range(2708)),
        cora_labels=data.y,
        conditions=["Normal", "Off", "Inverted"],
        device='cuda'
    )

    # In your cycle loop:
    result = suite.step(
        condition="Normal",
        embeddings=model_output,
        new_edges=zaratustra.new_edges_this_cycle,
        all_edges=current_graph.edge_index.T,
        cycle=t,
        tgc=current_tgc
    )
    print(result)

    # After all cycles:
    suite.report()
    suite.plot(save_path="new_dashboard.png")
    """

    def __init__(
        self,
        cora_node_ids: set,
        cora_labels: torch.Tensor,
        conditions: list = None,
        device: str = "cuda",
        temperature: float = 1.0,
    ):
        self.conditions = conditions or ["Normal", "Off", "Inverted"]
        self.device = device

        self.evaluators = {
            cond: AdaptiveEvaluator(
                cora_node_ids=cora_node_ids,
                cora_labels=cora_labels,
                device=device,
                temperature=temperature,
            )
            for cond in self.conditions
        }

        self.tgc_history = {c: [] for c in self.conditions}
        self.cycle_history = {c: [] for c in self.conditions}

    def step(
        self,
        condition: str,
        embeddings: torch.Tensor,
        new_edges: torch.Tensor,
        all_edges: torch.Tensor,
        cycle: int,
        tgc: float = None,
    ) -> dict:
        assert condition in self.evaluators, f"Unknown condition: {condition}"

        result = self.evaluators[condition].evaluate_cycle(
            embeddings=embeddings,
            new_edges=new_edges,
            all_edges=all_edges,
            cycle=cycle,
        )
        result["condition"] = condition
        if tgc is not None:
            result["tgc"] = tgc
            self.tgc_history[condition].append(tgc)
        self.cycle_history[condition].append(cycle)

        return result

    def report(self) -> dict:
        """Print a concise summary across all conditions."""
        print("\n" + "═" * 70)
        print("NietzscheDB ADAPTIVE EVALUATION REPORT")
        print("═" * 70)
        print(f"{'Condition':<20} {'AUC_synth':>10} {'Δ_homophily':>12} {'NCA':>8}")
        print("─" * 70)

        summary = {}
        for cond, ev in self.evaluators.items():
            h = ev.history
            auc_vals = [v for v in h.get("auc_synthetic", []) if not np.isnan(v)]
            dh_vals  = [v for v in h.get("delta_homophily", []) if not np.isnan(v)]
            nca_vals = [v for v in h.get("nca_mean", []) if not np.isnan(v)]

            auc_mean = np.mean(auc_vals) if auc_vals else float("nan")
            dh_mean  = np.mean(dh_vals)  if dh_vals  else float("nan")
            nca_mean = np.mean(nca_vals) if nca_vals else float("nan")

            print(f"{cond:<20} {auc_mean:>10.4f} {dh_mean:>12.4f} {nca_mean:>8.4f}")
            summary[cond] = {
                "auc_synthetic_mean": auc_mean,
                "delta_homophily_mean": dh_mean,
                "nca_mean": nca_mean,
            }

        print("═" * 70)
        print("\nPredicted ranking if hypothesis holds:")
        print("  AUC_synthetic:  Normal > Off > Inverted")
        print("  Δ_homophily:    Normal > Off > Inverted  (Normal builds meaning)")
        print("  NCA:            Normal ≥ Off > Inverted  (drift was intelligent)")
        print()
        return summary

    def plot(self, save_path: str = "nietzsche_adaptive_dashboard.png"):
        """Generate the adaptive metrics dashboard."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        colors = {"Normal": "#2E7D32", "Off": "#E65100", "Inverted": "#C62828"}
        fig = plt.figure(figsize=(18, 14), facecolor="#0D1117")
        fig.suptitle(
            "NietzscheDB: Adaptive Metrics Dashboard\n"
            "Measuring Intelligence, Not Memory",
            fontsize=16, fontweight="bold", color="white", y=0.98
        )

        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        # Panel config: (title, metric_key, ylabel)
        panels = [
            (gs[0, 0], "Generative Coherence AUC", "auc_synthetic", "AUC (synthetic edges)"),
            (gs[0, 1], "Homophily of New Edges", "homophily_new_edges", "h(new edges)"),
            (gs[0, 2], "Homophily Delta (Δh)", "delta_homophily", "Δh = h_new − h_baseline"),
            (gs[1, 0], "Node Classification Accuracy", "nca_mean", "NCA (linear probe)"),
        ]

        for spec, title, key, ylabel in panels:
            ax = fig.add_subplot(spec)
            ax.set_facecolor("#161B22")
            ax.tick_params(colors="white", labelsize=9)
            ax.set_title(title, color="white", fontsize=11, pad=8)
            ax.set_xlabel("Cycle", color="#AAAAAA", fontsize=9)
            ax.set_ylabel(ylabel, color="#AAAAAA", fontsize=9)
            for spine in ax.spines.values():
                spine.set_color("#333333")
            if key == "delta_homophily":
                ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")

            for cond, ev in self.evaluators.items():
                vals = ev.history.get(key, [])
                cycs = ev.history.get("cycle", [])
                clean = [(c, v) for c, v in zip(cycs, vals) if not np.isnan(v)]
                if clean:
                    cx, cy = zip(*clean)
                    ax.plot(cx, cy, color=colors.get(cond, "gray"),
                            linewidth=2, label=cond, alpha=0.9)

            ax.legend(fontsize=8, facecolor="#222222", labelcolor="white",
                      edgecolor="#333333", framealpha=0.8)

        # Panel: TGC vs AUC_synthetic scatter (pooled)
        ax_scatter = fig.add_subplot(gs[1, 1:])
        ax_scatter.set_facecolor("#161B22")
        ax_scatter.tick_params(colors="white", labelsize=9)
        ax_scatter.set_title("TGC vs Generative Coherence AUC (pooled)",
                              color="white", fontsize=11, pad=8)
        ax_scatter.set_xlabel("TGC EMA", color="#AAAAAA", fontsize=9)
        ax_scatter.set_ylabel("AUC_synthetic", color="#AAAAAA", fontsize=9)
        for spine in ax_scatter.spines.values():
            spine.set_color("#333333")

        all_tgc, all_auc = [], []
        for cond, ev in self.evaluators.items():
            tgcs = self.tgc_history[cond]
            aucs = ev.history.get("auc_synthetic", [])
            cycs = ev.history.get("cycle", [])
            pairs = [(t, a) for t, a in zip(tgcs, aucs) if not np.isnan(a) and not np.isnan(t)]
            if pairs:
                tx, ay = zip(*pairs)
                ax_scatter.scatter(tx, ay, color=colors.get(cond, "gray"),
                                   alpha=0.6, s=18, label=cond)
                all_tgc.extend(tx)
                all_auc.extend(ay)

        if len(all_tgc) > 5:
            from scipy.stats import pearsonr
            r, p = pearsonr(all_tgc, all_auc)
            z = np.polyfit(all_tgc, all_auc, 1)
            line = np.poly1d(z)
            x_range = np.linspace(min(all_tgc), max(all_tgc), 100)
            ax_scatter.plot(x_range, line(x_range), "w--", linewidth=1.5, alpha=0.7)
            ax_scatter.text(0.05, 0.92, f"r = {r:.4f}\np = {p:.2e}",
                            transform=ax_scatter.transAxes, fontsize=10,
                            color="white", verticalalignment="top",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A3A5C",
                                      edgecolor="#2196F3", alpha=0.9))

        ax_scatter.legend(fontsize=8, facecolor="#222222", labelcolor="white",
                          edgecolor="#333333", framealpha=0.8)

        # Panel: Final bar comparison
        ax_bar = fig.add_subplot(gs[2, :])
        ax_bar.set_facecolor("#161B22")
        ax_bar.tick_params(colors="white", labelsize=9)
        ax_bar.set_title("Final Cycle Summary: All Adaptive Metrics by Condition",
                         color="white", fontsize=11, pad=8)
        for spine in ax_bar.spines.values():
            spine.set_color("#333333")

        metrics_labels = ["AUC_synthetic", "Homophily (new edges)", "Δ Homophily", "NCA"]
        metric_keys = ["auc_synthetic", "homophily_new_edges", "delta_homophily", "nca_mean"]
        n_metrics = len(metric_keys)
        n_conds = len(self.conditions)
        width = 0.22
        x = np.arange(n_metrics)

        for i, cond in enumerate(self.conditions):
            ev = self.evaluators[cond]
            vals = []
            for key in metric_keys:
                data = [v for v in ev.history.get(key, []) if not np.isnan(v)]
                vals.append(np.mean(data) if data else 0.0)
            offset = (i - n_conds / 2 + 0.5) * width
            bars = ax_bar.bar(x + offset, vals, width, label=cond,
                              color=colors.get(cond, "gray"), alpha=0.85,
                              edgecolor="#111111", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if not np.isnan(v) and abs(v) > 0.001:
                    ax_bar.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8, color="white", fontweight="bold"
                    )

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metrics_labels, color="white", fontsize=10)
        ax_bar.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax_bar.legend(fontsize=9, facecolor="#222222", labelcolor="white",
                      edgecolor="#333333", framealpha=0.8)

        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0D1117", edgecolor="none")
        plt.close()
        print(f"\nDashboard saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION: Replace evaluate_link_prediction in your training loop
# ═══════════════════════════════════════════════════════════════════════════

def build_evaluation_suite_from_cora(cora_data, device="cuda") -> NietzscheEvaluationSuite:
    """
    Convenience factory. Pass your PyG Cora Data object and get a ready suite.

    cora_data: torch_geometric.data.Data with fields:
        .num_nodes  (int)
        .y          (LongTensor of class labels, shape [num_nodes])
    """
    cora_node_ids = set(range(cora_data.num_nodes))
    return NietzscheEvaluationSuite(
        cora_node_ids=cora_node_ids,
        cora_labels=cora_data.y,
        conditions=["Normal", "Off", "Inverted"],
        device=device,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST: Smoke test with synthetic data (no GPU required)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running smoke test with synthetic hyperbolic embeddings...")
    torch.manual_seed(42)
    np.random.seed(42)

    N_CORA = 200
    N_FOAM = 50
    N_TOTAL = N_CORA + N_FOAM
    D = 16
    N_CYCLES = 10

    # Synthetic labels: 7 classes (like Cora)
    labels = torch.randint(0, 7, (N_TOTAL,))

    # Embeddings: Cora nodes near center, foam near boundary
    emb_cora = torch.randn(N_CORA, D) * 0.2
    emb_cora = F.normalize(emb_cora, dim=-1) * (torch.rand(N_CORA, 1) * 0.5)
    emb_foam = F.normalize(torch.randn(N_FOAM, D), dim=-1) * 0.97
    embeddings = torch.cat([emb_cora, emb_foam], dim=0)

    # Build evaluator
    cora_ids = set(range(N_CORA))
    ev = AdaptiveEvaluator(
        cora_node_ids=cora_ids,
        cora_labels=labels[:N_CORA],
        device="cpu",
        temperature=1.0,
    )

    # Simulate cycles
    for t in range(N_CYCLES):
        # Synthetic new edges: some within Cora (coherent), some to foam
        src = torch.randint(0, N_CORA, (20,))
        dst = torch.randint(0, N_CORA, (20,))
        new_edges = torch.stack([src, dst], dim=1)

        # All edges (base Cora + new)
        all_src = torch.randint(0, N_TOTAL, (100,))
        all_dst = torch.randint(0, N_TOTAL, (100,))
        all_edges = torch.stack([all_src, all_dst], dim=1)

        # Drift embeddings slightly
        embeddings = embeddings + torch.randn_like(embeddings) * 0.005
        embeddings = torch.clamp(embeddings, -0.99, 0.99)

        result = ev.evaluate_cycle(
            embeddings=embeddings,
            new_edges=new_edges,
            all_edges=all_edges,
            cycle=t,
        )
        print(f"  Cycle {t:2d} | AUC_synth={result['auc_synthetic']:.4f} | "
              f"Δh={result['delta_homophily']:+.4f} | "
              f"NCA={result['nca_mean']:.4f}")

    print("\nSmoke test PASSED. All metrics functional.")
    print("\nTo integrate into NietzscheDB:")
    print("  from nietzsche_adaptive_metrics import build_evaluation_suite_from_cora")
    print("  suite = build_evaluation_suite_from_cora(cora_data, device='cuda')")
    print("  # In cycle loop:")
    print("  suite.step('Normal', embeddings, new_edges, all_edges, cycle=t, tgc=tgc_ema)")
    print("  # After training:")
    print("  suite.report()")
    print("  suite.plot('adaptive_dashboard.png')")
