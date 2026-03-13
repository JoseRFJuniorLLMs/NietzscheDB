"""
Core kernel computation service for NietzscheDB.

Provides heat kernels and Matérn kernels on NietzscheDB graphs using
GeometricKernels' mathematically exact spectral decomposition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from geometric_kernels.kernels import MaternGeometricKernel
from geometric_kernels.spaces import Graph

logger = logging.getLogger(__name__)


@dataclass
class DiffusionResult:
    """Result of heat diffusion from source nodes."""

    activation: np.ndarray  # shape: (n_nodes,)
    source_indices: list[int]
    t: float
    top_k_indices: list[int]  # indices of top-K activated nodes
    top_k_values: list[float]  # activation values for top-K


class NietzscheKernelService:
    """
    Geometric kernel computation on NietzscheDB graph structures.

    Wraps GeometricKernels' MaternGeometricKernel for Graph spaces,
    providing heat kernel (ν=∞), Matérn (ν=0.5..5.0), and activation
    diffusion computations.

    The Graph space uses Karhunen-Loève expansion:
        k(x,y) = Σ_l S(√λ_l) Σ_s φ_{ls}(x) φ_{ls}(y)
    where S is the Matérn spectrum and φ are Laplacian eigenfunctions.
    """

    def __init__(self, graph_space: Graph, num_eigenpairs: int | None = None):
        """
        Args:
            graph_space: GeometricKernels Graph space
            num_eigenpairs: Number of Laplacian eigenpairs to use.
                More = more accurate but slower. Default: min(200, n_nodes).
        """
        self.space = graph_space
        n = graph_space.num_vertices

        if num_eigenpairs is None:
            num_eigenpairs = min(200, n)

        self.num_eigenpairs = num_eigenpairs
        self.kernel = MaternGeometricKernel(graph_space, num=num_eigenpairs)
        self.params = self.kernel.init_params()

        logger.info(
            "KernelService initialized: %d vertices, %d eigenpairs",
            n, num_eigenpairs,
        )

    def _set_params(self, nu: float, lengthscale: float) -> None:
        """Update kernel parameters."""
        self.params["nu"] = np.array([nu])
        self.params["lengthscale"] = np.array([lengthscale])

    def heat_kernel_matrix(
        self, node_indices: np.ndarray, t: float
    ) -> np.ndarray:
        """
        Compute exact heat kernel matrix K_t(i,j) for selected nodes.

        Equivalent to e^{-tL̃} from nietzsche-pregel, but exact
        (not Chebyshev-approximated).

        Args:
            node_indices: Array of node indices (integers)
            t: Diffusion time. Cognitive scales:
                - t=0.1: focused recall (immediate neighbors)
                - t=1.0: associative thinking (2-3 hops)
                - t=10.0: free association (distant structural relatives)

        Returns:
            Kernel matrix of shape (len(node_indices), len(node_indices))
        """
        # Heat kernel: ν=∞, lengthscale=√(2t)
        self._set_params(nu=np.inf, lengthscale=np.sqrt(2 * t))

        X = node_indices.reshape(-1, 1).astype(np.int64)
        K = self.kernel.K(self.params, X)
        return np.array(K)

    def matern_kernel_matrix(
        self,
        node_indices: np.ndarray,
        nu: float = 2.5,
        lengthscale: float = 1.0,
    ) -> np.ndarray:
        """
        Compute Matérn kernel matrix for selected nodes.

        More flexible than heat kernel — ν controls smoothness:
            - ν=0.5: Laplacian kernel (very rough, sharp boundaries)
            - ν=1.5: once differentiable
            - ν=2.5: twice differentiable (good default)
            - ν=∞: heat kernel (infinitely smooth)

        Args:
            node_indices: Array of node indices
            nu: Smoothness parameter
            lengthscale: Length scale parameter

        Returns:
            Kernel matrix of shape (len(node_indices), len(node_indices))
        """
        self._set_params(nu=nu, lengthscale=lengthscale)

        X = node_indices.reshape(-1, 1).astype(np.int64)
        K = self.kernel.K(self.params, X)
        return np.array(K)

    def kernel_diagonal(
        self,
        node_indices: np.ndarray,
        nu: float = 2.5,
        lengthscale: float = 1.0,
    ) -> np.ndarray:
        """
        Compute kernel diagonal K(i,i) — useful for normalization.

        Returns:
            Array of shape (len(node_indices),) with diagonal entries.
        """
        self._set_params(nu=nu, lengthscale=lengthscale)
        X = node_indices.reshape(-1, 1).astype(np.int64)
        return np.array(self.kernel.K_diag(self.params, X))

    def diffuse_activation(
        self,
        source_indices: list[int],
        t: float,
        top_k: int = 20,
    ) -> DiffusionResult:
        """
        Diffuse activation from source nodes to all nodes in the graph.

        Equivalent to DiffusionEngine::diffuse() in nietzsche-pregel,
        but using exact spectral heat kernel instead of Chebyshev.

        Args:
            source_indices: Node indices to activate
            t: Diffusion time (0.1=focused, 1.0=associative, 10.0=free)
            top_k: Number of top-activated nodes to return

        Returns:
            DiffusionResult with activation vector and top-K info
        """
        n = self.space.num_vertices
        self._set_params(nu=np.inf, lengthscale=np.sqrt(2 * t))

        all_indices = np.arange(n).reshape(-1, 1).astype(np.int64)
        sources = np.array(source_indices).reshape(-1, 1).astype(np.int64)

        # K[i, j] = heat_kernel(all_node_i, source_j)
        K = np.array(self.kernel.K(self.params, all_indices, sources))

        # Sum activation from all sources
        activation = K.sum(axis=1)

        # Get top-K
        top_k_actual = min(top_k, n)
        top_indices = np.argsort(-activation)[:top_k_actual]

        return DiffusionResult(
            activation=activation,
            source_indices=source_indices,
            t=t,
            top_k_indices=top_indices.tolist(),
            top_k_values=activation[top_indices].tolist(),
        )

    def pairwise_similarity(
        self,
        idx_a: int,
        idx_b: int,
        kernel_type: Literal["heat", "matern"] = "matern",
        nu: float = 2.5,
        lengthscale: float = 1.0,
        t: float = 1.0,
    ) -> float:
        """
        Compute kernel similarity between two nodes.

        Args:
            idx_a, idx_b: Node indices
            kernel_type: "heat" or "matern"
            nu: Matérn smoothness (ignored for heat)
            lengthscale: Matérn lengthscale (ignored for heat)
            t: Heat diffusion time (ignored for matern)

        Returns:
            Kernel value k(idx_a, idx_b)
        """
        if kernel_type == "heat":
            self._set_params(nu=np.inf, lengthscale=np.sqrt(2 * t))
        else:
            self._set_params(nu=nu, lengthscale=lengthscale)

        X = np.array([[idx_a]], dtype=np.int64)
        Y = np.array([[idx_b]], dtype=np.int64)
        K = self.kernel.K(self.params, X, Y)
        return float(np.array(K)[0, 0])

    def multi_scale_diffusion(
        self,
        source_indices: list[int],
        t_values: list[float] | None = None,
        top_k: int = 20,
    ) -> dict[float, DiffusionResult]:
        """
        Run diffusion at multiple cognitive scales.

        Default scales match nietzsche-pregel:
            t=0.1 → focused recall
            t=1.0 → associative thinking
            t=10.0 → free association

        Returns:
            Dict mapping each t to its DiffusionResult
        """
        if t_values is None:
            t_values = [0.1, 1.0, 10.0]

        results = {}
        for t in t_values:
            results[t] = self.diffuse_activation(source_indices, t, top_k)

        return results
