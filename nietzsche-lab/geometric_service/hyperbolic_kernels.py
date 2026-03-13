"""
Hyperbolic kernel service for NietzscheDB.

Provides exact Matérn and heat kernels on the hyperbolic space
(Poincaré ball model used by NietzscheDB), using GeometricKernels'
hyperboloid representation internally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from geometric_kernels.kernels import MaternGeometricKernel
from geometric_kernels.spaces import Hyperbolic

logger = logging.getLogger(__name__)


@dataclass
class HyperbolicSimilarityResult:
    """Result of hyperbolic kernel similarity computation."""

    index: int
    similarity: float
    poincare_distance: float


class HyperbolicKernelService:
    """
    Kernels on hyperbolic space for NietzscheDB Poincaré embeddings.

    NietzscheDB stores embeddings as PoincareVector (f32 coords in the
    Poincaré ball). GeometricKernels uses the hyperboloid model internally.
    This service handles the conversion transparently.

    Feature maps use rejection sampling on the hyperboloid — more
    computationally expensive than graph kernels but works with
    continuous coordinates (no eigendecomposition needed).

    IMPORTANT: For dim=128 (NietzscheDB default), the feature map
    computation is expensive. Use num_features=64-128 for interactive
    use, 256+ for calibration/training.
    """

    def __init__(self, dim: int = 128, num_features: int = 128):
        """
        Args:
            dim: Hyperbolic space dimension (must match PoincareVector dim)
            num_features: Number of random features for kernel approximation
        """
        self.dim = dim
        self.space = Hyperbolic(dim=dim)
        self.kernel = MaternGeometricKernel(self.space, num=num_features)
        self.params = self.kernel.init_params()
        self.num_features = num_features

        logger.info("HyperbolicKernelService: dim=%d, features=%d", dim, num_features)

    @staticmethod
    def poincare_to_hyperboloid(poincare_coords: np.ndarray) -> np.ndarray:
        """
        Convert Poincaré ball coordinates to hyperboloid model.

        Poincaré ball: x ∈ R^d, ||x|| < 1
        Hyperboloid: y ∈ R^{d+1}, y₀² - y₁² - ... - y_d² = 1, y₀ > 0

        Formula:
            y₀ = (1 + ||x||²) / (1 - ||x||²)
            yᵢ = 2xᵢ / (1 - ||x||²)    for i = 1..d

        Args:
            poincare_coords: Shape (..., d) array of Poincaré ball coordinates

        Returns:
            Shape (..., d+1) array of hyperboloid coordinates
        """
        norm_sq = np.sum(poincare_coords ** 2, axis=-1, keepdims=True)
        # Clamp to avoid division by zero at boundary
        norm_sq = np.clip(norm_sq, 0, 1 - 1e-7)

        denom = 1.0 - norm_sq
        y0 = (1.0 + norm_sq) / denom
        yi = 2.0 * poincare_coords / denom

        return np.concatenate([y0, yi], axis=-1)

    @staticmethod
    def hyperboloid_to_poincare(hyperboloid_coords: np.ndarray) -> np.ndarray:
        """
        Convert hyperboloid model coordinates to Poincaré ball.

        Formula: xᵢ = yᵢ / (1 + y₀)    for i = 1..d
        """
        y0 = hyperboloid_coords[..., :1]
        yi = hyperboloid_coords[..., 1:]
        return yi / (1.0 + y0)

    @staticmethod
    def poincare_distance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Poincaré distance between two points.

        d_P(x,y) = acosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
        """
        diff_sq = np.sum((x - y) ** 2)
        nx = np.sum(x ** 2)
        ny = np.sum(y ** 2)
        denom = (1 - nx) * (1 - ny)
        if denom <= 0:
            return float("inf")
        arg = 1 + 2 * diff_sq / denom
        return float(np.arccosh(max(arg, 1.0)))

    def compute_similarity_matrix(
        self,
        poincare_points: np.ndarray,
        nu: float = 2.5,
        lengthscale: float = 1.0,
    ) -> np.ndarray:
        """
        Compute Matérn kernel matrix on hyperbolic space.

        Args:
            poincare_points: Shape (n, dim) Poincaré ball coordinates
            nu: Smoothness parameter
            lengthscale: Length scale

        Returns:
            (n, n) kernel matrix respecting hyperbolic geometry
        """
        self.params["nu"] = np.array([nu])
        self.params["lengthscale"] = np.array([lengthscale])

        X = self.poincare_to_hyperboloid(poincare_points)
        K = self.kernel.K(self.params, X)
        return np.array(K)

    def compute_heat_kernel_matrix(
        self,
        poincare_points: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Compute heat kernel matrix on hyperbolic space.

        Args:
            poincare_points: Shape (n, dim) Poincaré ball coordinates
            t: Diffusion time

        Returns:
            (n, n) heat kernel matrix
        """
        return self.compute_similarity_matrix(
            poincare_points, nu=np.inf, lengthscale=np.sqrt(2 * t)
        )

    def kernel_knn(
        self,
        query_poincare: np.ndarray,
        candidates_poincare: np.ndarray,
        k: int = 10,
        nu: float = 2.5,
        lengthscale: float = 1.0,
    ) -> list[HyperbolicSimilarityResult]:
        """
        K-nearest neighbors using hyperbolic Matérn kernel.

        Advantage over raw poincare_distance:
        - Matérn captures multi-scale locality (ν controls smoothness)
        - More robust for nodes near the disk boundary
        - Can be tuned via lengthscale for different semantic scales

        Args:
            query_poincare: Shape (dim,) query point
            candidates_poincare: Shape (m, dim) candidate points
            k: Number of nearest neighbors
            nu: Matérn smoothness
            lengthscale: Length scale

        Returns:
            List of k HyperbolicSimilarityResult sorted by similarity desc
        """
        self.params["nu"] = np.array([nu])
        self.params["lengthscale"] = np.array([lengthscale])

        query_hyp = self.poincare_to_hyperboloid(
            query_poincare.reshape(1, -1)
        )
        cands_hyp = self.poincare_to_hyperboloid(candidates_poincare)

        K = np.array(self.kernel.K(self.params, cands_hyp, query_hyp)).flatten()

        top_k_idx = np.argsort(-K)[:k]

        results = []
        for idx in top_k_idx:
            pdist = self.poincare_distance(
                query_poincare, candidates_poincare[idx]
            )
            results.append(HyperbolicSimilarityResult(
                index=int(idx),
                similarity=float(K[idx]),
                poincare_distance=pdist,
            ))

        return results

    def validate_gauss_kernel(
        self,
        poincare_point: np.ndarray,
        t: float,
    ) -> dict[str, float]:
        """
        Compare NQL GAUSS_KERNEL (Euclidean) vs correct hyperbolic heat kernel.

        NietzscheDB's current GAUSS_KERNEL uses:
            h_t(x) = exp(-||x||² / (4t))     ← Euclidean!

        The correct hyperbolic version should use:
            h_t(x) = exp(-d_P(x, 0)² / (4t))   ← Poincaré distance from origin

        Returns:
            Dict with both values and the error
        """
        norm_sq = float(np.sum(poincare_point ** 2))

        # Current NQL implementation (Euclidean)
        euclidean_kernel = np.exp(-norm_sq / (4.0 * max(t, 1e-15)))

        # Correct hyperbolic distance from origin
        # d_P(x, 0) = 2 * atanh(||x||) for Poincaré ball
        norm = np.sqrt(norm_sq)
        norm_clamped = min(norm, 1 - 1e-7)
        poincare_dist = 2.0 * np.arctanh(norm_clamped)
        hyperbolic_kernel = np.exp(-(poincare_dist ** 2) / (4.0 * max(t, 1e-15)))

        return {
            "euclidean_kernel": float(euclidean_kernel),
            "hyperbolic_kernel": float(hyperbolic_kernel),
            "absolute_error": abs(float(euclidean_kernel - hyperbolic_kernel)),
            "relative_error": abs(float(euclidean_kernel - hyperbolic_kernel))
                / max(float(hyperbolic_kernel), 1e-15),
            "poincare_norm": float(norm),
            "poincare_distance_from_origin": float(poincare_dist),
        }
