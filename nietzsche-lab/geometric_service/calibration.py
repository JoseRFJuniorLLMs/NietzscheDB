"""
Chebyshev calibration module.

Compares NietzscheDB's Chebyshev polynomial approximation of the heat
kernel (nietzsche-pregel) against GeometricKernels' exact spectral
decomposition to find optimal K_max per cognitive scale.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

from .kernel_service import NietzscheKernelService

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Metrics comparing Chebyshev approximation vs exact kernel."""

    t: float
    k_max: int
    l2_error: float
    spearman_rho: float
    overlap_at_k: float
    k_for_overlap: int
    mae: float  # mean absolute error
    max_error: float  # worst-case error


@dataclass
class CalibrationReport:
    """Full calibration report across scales and K_max values."""

    collection: str
    n_nodes: int
    n_edges: int
    num_eigenpairs: int
    metrics: list[CalibrationMetrics] = field(default_factory=list)
    optimal_k_max: dict[float, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "collection": self.collection,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "num_eigenpairs": self.num_eigenpairs,
            "metrics": [
                {
                    "t": m.t, "k_max": m.k_max, "l2_error": m.l2_error,
                    "spearman_rho": m.spearman_rho,
                    "overlap_at_k": m.overlap_at_k,
                    "k_for_overlap": m.k_for_overlap,
                    "mae": m.mae, "max_error": m.max_error,
                }
                for m in self.metrics
            ],
            "optimal_k_max": {str(k): v for k, v in self.optimal_k_max.items()},
        }

    def summary(self) -> str:
        lines = [
            f"=== Chebyshev Calibration Report ===",
            f"Collection: {self.collection}",
            f"Nodes: {self.n_nodes}, Edges: {self.n_edges}",
            f"Eigenpairs: {self.num_eigenpairs}",
            "",
            f"{'t':>6} {'K_max':>6} {'L2 err':>10} {'Spearman':>10} "
            f"{'Overlap@20':>12} {'MAE':>10} {'MaxErr':>10}",
            "-" * 70,
        ]
        for m in self.metrics:
            lines.append(
                f"{m.t:>6.1f} {m.k_max:>6d} {m.l2_error:>10.6f} "
                f"{m.spearman_rho:>10.4f} {m.overlap_at_k:>12.2%} "
                f"{m.mae:>10.6f} {m.max_error:>10.6f}"
            )
        lines.append("")
        lines.append("Optimal K_max per scale:")
        for t, k in sorted(self.optimal_k_max.items()):
            lines.append(f"  t={t:.1f} → K_max={k}")
        return "\n".join(lines)


class ChebyshevCalibrator:
    """
    Compare Chebyshev-approximated diffusion (Rust) against exact
    heat kernel (GeometricKernels).

    The calibrator simulates what nietzsche-pregel computes using
    a local Chebyshev implementation, then compares against the
    exact kernel from GeometricKernels.
    """

    def __init__(self, gk_service: NietzscheKernelService):
        """
        Args:
            gk_service: Initialized NietzscheKernelService
        """
        self.gk = gk_service
        self.space = gk_service.space
        self.n = self.space.num_vertices

    def _chebyshev_heat_kernel(
        self, source_idx: int, t: float, k_max: int
    ) -> np.ndarray:
        """
        Local implementation of Chebyshev-approximated heat kernel.

        Reproduces the logic of nietzsche-pregel/src/chebyshev.rs:
            K_t f ≈ Σ_{k=0}^{K} c_k(t) · T_k(L̂) · f

        where:
            c_k(t) = e^{-t} · (2 - δ_{k,0}) · I_k(t)
            L̂ = 2/λ_max · L̃ - I
        """
        # Get eigenvalues from the GeometricKernels space
        eigvals = self.space.cache["eigvals"]
        eigvecs = self.space.cache["eigvecs"]
        lambda_max = float(eigvals[-1]) if len(eigvals) > 0 else 2.0
        lambda_max = max(lambda_max, 1e-10)

        # Initial impulse vector
        f = np.zeros(self.n)
        f[source_idx] = 1.0

        # Compute Chebyshev coefficients via modified Bessel function
        coeffs = self._chebyshev_coefficients(t, k_max)

        # Scaled Laplacian: L̂ = 2/λ_max · L̃ - I
        # We work in the Laplacian eigenbasis:
        # L̃ = Σ λ_i φ_i φ_i^T
        # L̂ = Σ (2λ_i/λ_max - 1) φ_i φ_i^T
        scaled_eigvals = 2.0 * eigvals / lambda_max - 1.0

        # Project f into eigenbasis
        f_hat = eigvecs.T @ f  # coefficients in eigenbasis

        # Apply Chebyshev polynomials T_k(scaled_eigvals) to each eigenvalue
        # T_0(x) = 1, T_1(x) = x, T_k(x) = 2x·T_{k-1}(x) - T_{k-2}(x)
        result_hat = np.zeros_like(f_hat)

        T_prev = np.ones_like(scaled_eigvals)   # T_0 = 1
        T_curr = scaled_eigvals.copy()           # T_1 = x

        result_hat += coeffs[0] * T_prev * f_hat

        if k_max >= 1:
            result_hat += coeffs[1] * T_curr * f_hat

        for k in range(2, k_max + 1):
            T_next = 2.0 * scaled_eigvals * T_curr - T_prev
            result_hat += coeffs[k] * T_next * f_hat
            T_prev = T_curr
            T_curr = T_next

        # Project back from eigenbasis
        result = eigvecs @ result_hat
        return np.maximum(result, 0)  # clamp negatives

    @staticmethod
    def _chebyshev_coefficients(t: float, k_max: int) -> np.ndarray:
        """
        Compute Chebyshev expansion coefficients for heat kernel.

        c_k(t) = e^{-t} · (2 - δ_{k,0}) · I_k(t)

        where I_k(t) is the modified Bessel function of the first kind.
        Matches nietzsche-pregel/src/chebyshev.rs::chebyshev_coefficients().
        """
        from scipy.special import iv  # modified Bessel I_k

        coeffs = np.zeros(k_max + 1)
        exp_neg_t = np.exp(-t)

        for k in range(k_max + 1):
            bessel = iv(k, t)
            factor = 1.0 if k == 0 else 2.0
            coeffs[k] = exp_neg_t * factor * bessel

        return coeffs

    def compare_diffusion(
        self,
        source_idx: int,
        t_values: list[float],
        k_max_values: list[int],
        overlap_k: int = 20,
    ) -> list[CalibrationMetrics]:
        """
        Compare Chebyshev vs exact kernel for given scales and K_max values.

        Args:
            source_idx: Source node index for diffusion
            t_values: Diffusion time values to test
            k_max_values: Chebyshev truncation orders to test
            overlap_k: K for Overlap@K metric

        Returns:
            List of CalibrationMetrics for each (t, k_max) combination
        """
        results = []

        for t in t_values:
            # Exact activation via GeometricKernels
            exact_result = self.gk.diffuse_activation([source_idx], t, top_k=0)
            exact = exact_result.activation
            exact_norm = exact / (exact.max() + 1e-15)

            top_exact = set(np.argsort(-exact_norm)[:overlap_k])

            for k_max in k_max_values:
                # Chebyshev approximation
                cheb = self._chebyshev_heat_kernel(source_idx, t, k_max)
                cheb_norm = cheb / (cheb.max() + 1e-15)

                # L2 error (normalized)
                l2_error = float(np.linalg.norm(exact_norm - cheb_norm) / self.n)

                # Spearman rank correlation
                rho, _ = spearmanr(exact_norm, cheb_norm)

                # Overlap@K
                top_cheb = set(np.argsort(-cheb_norm)[:overlap_k])
                overlap = len(top_exact & top_cheb) / overlap_k

                # MAE
                mae = float(np.mean(np.abs(exact_norm - cheb_norm)))

                # Max error
                max_error = float(np.max(np.abs(exact_norm - cheb_norm)))

                metrics = CalibrationMetrics(
                    t=t, k_max=k_max, l2_error=l2_error,
                    spearman_rho=float(rho), overlap_at_k=overlap,
                    k_for_overlap=overlap_k, mae=mae, max_error=max_error,
                )
                results.append(metrics)

                logger.info(
                    "t=%.1f k_max=%d: L2=%.6f Spearman=%.4f Overlap@%d=%.2f%%",
                    t, k_max, l2_error, rho, overlap_k, overlap * 100,
                )

        return results

    def find_optimal_k_max(
        self,
        source_indices: list[int] | None = None,
        t_values: list[float] | None = None,
        target_overlap: float = 0.90,
        k_max_range: list[int] | None = None,
    ) -> CalibrationReport:
        """
        Find minimum K_max that achieves target Overlap@20 for each scale.

        Args:
            source_indices: Nodes to test diffusion from (default: 5 random)
            t_values: Cognitive scales (default: [0.1, 1.0, 10.0])
            target_overlap: Target Overlap@20 (default: 90%)
            k_max_range: K_max values to test (default: [5,10,15,20,30,40,50])

        Returns:
            CalibrationReport with all metrics and optimal K_max per scale
        """
        if t_values is None:
            t_values = [0.1, 1.0, 10.0]
        if k_max_range is None:
            k_max_range = [5, 10, 15, 20, 30, 40, 50]
        if source_indices is None:
            rng = np.random.default_rng(42)
            source_indices = rng.choice(
                self.n, size=min(5, self.n), replace=False
            ).tolist()

        all_metrics = []
        for src in source_indices:
            metrics = self.compare_diffusion(src, t_values, k_max_range)
            all_metrics.extend(metrics)

        # Average metrics per (t, k_max) across sources
        averaged: dict[tuple[float, int], list[CalibrationMetrics]] = {}
        for m in all_metrics:
            key = (m.t, m.k_max)
            averaged.setdefault(key, []).append(m)

        final_metrics = []
        for (t, k_max), group in sorted(averaged.items()):
            avg = CalibrationMetrics(
                t=t, k_max=k_max,
                l2_error=float(np.mean([m.l2_error for m in group])),
                spearman_rho=float(np.mean([m.spearman_rho for m in group])),
                overlap_at_k=float(np.mean([m.overlap_at_k for m in group])),
                k_for_overlap=group[0].k_for_overlap,
                mae=float(np.mean([m.mae for m in group])),
                max_error=float(np.mean([m.max_error for m in group])),
            )
            final_metrics.append(avg)

        # Find optimal K_max per scale
        optimal = {}
        for t in t_values:
            for k_max in sorted(k_max_range):
                candidates = [
                    m for m in final_metrics
                    if m.t == t and m.k_max == k_max
                ]
                if candidates and candidates[0].overlap_at_k >= target_overlap:
                    optimal[t] = k_max
                    break
            else:
                # If none reached target, use the largest
                optimal[t] = max(k_max_range)

        report = CalibrationReport(
            collection="",  # set by caller
            n_nodes=self.n,
            n_edges=0,  # set by caller
            num_eigenpairs=self.gk.num_eigenpairs,
            metrics=final_metrics,
            optimal_k_max=optimal,
        )

        return report
