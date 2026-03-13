"""
GP-based epistemic uncertainty estimation for NietzscheDB.

Uses Gaussian Processes with geometric kernels to estimate
epistemic uncertainty across the knowledge graph. High-variance
regions indicate knowledge gaps — candidates for autonomous
research by the EpistemologyDaemon / Evolution27.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import gpytorch
    from geometric_kernels.frontends import GPyTorchGeometricKernel
    from geometric_kernels.kernels import MaternGeometricKernel
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False
    logger.warning(
        "GPyTorch not available — EpistemicUncertaintyEstimator will use "
        "spectral fallback (less accurate but no extra dependencies)"
    )


@dataclass
class UncertaintyResult:
    """Uncertainty estimation for a single node."""

    node_id: str
    node_index: int
    mean: float
    variance: float
    uncertainty: float  # √variance


@dataclass
class KnowledgeGap:
    """A detected gap in the knowledge graph."""

    node_id: str
    node_index: int
    uncertainty: float
    suggested_action: str  # "research", "consolidate", "prune"


class _ExactGPModel(gpytorch.models.ExactGP if HAS_GPYTORCH else object):
    """GP model with geometric kernel for graph spaces."""

    def __init__(self, train_x, train_y, likelihood, graph_space):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = MaternGeometricKernel(graph_space)
        self.covar_module = GPyTorchGeometricKernel(base_kernel)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class EpistemicUncertaintyEstimator:
    """
    Estimates epistemic uncertainty on the NietzscheDB knowledge graph.

    Workflow:
    1. Treat node energies as "observations" (high energy = well-known)
    2. Train GP with geometric kernel on the graph
    3. Predict variance for ALL nodes
    4. High variance = knowledge gap → trigger EpistemologyDaemon

    Two modes:
    - GPyTorch mode: Full GP with trainable hyperparameters (requires torch)
    - Spectral fallback: Uses kernel eigendecomposition directly (numpy only)
    """

    def __init__(self, graph_space, node_ids: list[str]):
        """
        Args:
            graph_space: GeometricKernels Graph space
            node_ids: Ordered list of node UUID strings
        """
        self.space = graph_space
        self.node_ids = node_ids
        self.n = len(node_ids)
        self.model = None
        self.likelihood = None
        self._fitted = False

    def fit(
        self,
        observed_indices: np.ndarray,
        observed_values: np.ndarray,
        training_iterations: int = 50,
        learning_rate: float = 0.1,
        noise_constraint_lb: float = 1e-4,
    ) -> dict:
        """
        Train the GP model on observed nodes.

        Args:
            observed_indices: Node indices with known values
            observed_values: Observed signal (e.g., energy, activation)
            training_iterations: SGD steps
            learning_rate: Adam learning rate
            noise_constraint_lb: Lower bound on observation noise

        Returns:
            Training summary dict with final loss and hyperparams
        """
        if HAS_GPYTORCH:
            return self._fit_gpytorch(
                observed_indices, observed_values,
                training_iterations, learning_rate, noise_constraint_lb,
            )
        else:
            return self._fit_spectral(observed_indices, observed_values)

    def _fit_gpytorch(
        self, obs_idx, obs_val, iters, lr, noise_lb
    ) -> dict:
        """Full GP training with GPyTorch."""
        train_x = torch.tensor(obs_idx, dtype=torch.long).unsqueeze(-1)
        train_y = torch.tensor(obs_val, dtype=torch.float32)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(noise_lb)
        )
        self.model = _ExactGPModel(
            train_x, train_y, self.likelihood, self.space
        )

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model
        )

        final_loss = 0.0
        for i in range(iters):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

            if (i + 1) % 25 == 0:
                logger.info("GP training iter %d/%d, loss=%.4f", i + 1, iters, final_loss)

        self._fitted = True

        return {
            "mode": "gpytorch",
            "iterations": iters,
            "final_loss": final_loss,
            "n_observed": len(obs_idx),
            "noise": float(self.likelihood.noise.item()),
        }

    def _fit_spectral(self, obs_idx, obs_val) -> dict:
        """
        Spectral fallback: estimate uncertainty without GPyTorch.

        Uses the Laplacian eigenvectors directly to compute a
        kernel-smoothed variance estimate.
        """
        eigvals = np.array(self.space.cache.get("eigvals", []))
        eigvecs = np.array(self.space.cache.get("eigvecs", np.eye(self.n)))

        # Matérn spectrum: S(λ) = (1 + λ)^{-ν-d/2}, using ν=2.5, d=1
        spectrum = (1.0 + eigvals) ** (-3.0)  # ν=2.5, d=1 → exponent=-3.0

        # Kernel matrix (low-rank)
        K = (eigvecs * spectrum) @ eigvecs.T

        # GP posterior with noise σ²=0.01
        noise = 0.01
        K_obs = K[np.ix_(obs_idx, obs_idx)] + noise * np.eye(len(obs_idx))
        K_star = K[:, obs_idx]

        try:
            L = np.linalg.cholesky(K_obs)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, obs_val))
        except np.linalg.LinAlgError:
            # Fallback: use pseudoinverse
            alpha = np.linalg.lstsq(K_obs, obs_val, rcond=None)[0]

        self._spectral_mean = K_star @ alpha
        self._spectral_var = np.diag(K) - np.sum(
            np.linalg.solve(K_obs, K_star.T) * K_star.T, axis=0
        )
        self._spectral_var = np.maximum(self._spectral_var, 0)

        self._fitted = True

        return {
            "mode": "spectral_fallback",
            "n_observed": len(obs_idx),
            "n_eigenpairs": len(eigvals),
        }

    def predict_uncertainty(self) -> list[UncertaintyResult]:
        """
        Predict epistemic uncertainty for all nodes.

        Returns:
            List of UncertaintyResult for each node, sorted by index.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_uncertainty()")

        if HAS_GPYTORCH and self.model is not None:
            return self._predict_gpytorch()
        else:
            return self._predict_spectral()

    def _predict_gpytorch(self) -> list[UncertaintyResult]:
        self.model.eval()
        self.likelihood.eval()

        all_idx = torch.arange(self.n).unsqueeze(-1)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(all_idx))
            means = pred.mean.numpy()
            variances = pred.variance.numpy()

        return [
            UncertaintyResult(
                node_id=self.node_ids[i],
                node_index=i,
                mean=float(means[i]),
                variance=float(variances[i]),
                uncertainty=float(np.sqrt(variances[i])),
            )
            for i in range(self.n)
        ]

    def _predict_spectral(self) -> list[UncertaintyResult]:
        return [
            UncertaintyResult(
                node_id=self.node_ids[i],
                node_index=i,
                mean=float(self._spectral_mean[i]),
                variance=float(self._spectral_var[i]),
                uncertainty=float(np.sqrt(self._spectral_var[i])),
            )
            for i in range(self.n)
        ]

    def find_knowledge_gaps(
        self,
        top_k: int = 20,
        high_uncertainty_threshold: float = 0.7,
        low_energy_threshold: float = 0.1,
        node_energies: dict[str, float] | None = None,
    ) -> list[KnowledgeGap]:
        """
        Identify top-K nodes with highest epistemic uncertainty.

        Classifies each gap into suggested actions:
        - "research": High uncertainty, enough energy → seek new knowledge
        - "consolidate": Low uncertainty, high energy → strengthen connections
        - "prune": Low uncertainty, zero energy → candidate for removal

        Args:
            top_k: Number of gaps to return
            high_uncertainty_threshold: Quantile above which uncertainty is "high"
            low_energy_threshold: Energy below which a node is "low energy"
            node_energies: Optional dict {node_id: energy} for action classification

        Returns:
            List of KnowledgeGap sorted by uncertainty (descending)
        """
        results = self.predict_uncertainty()

        # Sort by variance descending
        results.sort(key=lambda r: r.variance, reverse=True)

        # Compute threshold from quantile
        all_uncertainties = [r.uncertainty for r in results]
        threshold = float(np.quantile(all_uncertainties, high_uncertainty_threshold))

        gaps = []
        for r in results[:top_k]:
            energy = 0.0
            if node_energies and r.node_id in node_energies:
                energy = node_energies[r.node_id]

            if r.uncertainty >= threshold:
                if energy >= low_energy_threshold:
                    action = "research"
                else:
                    action = "research"  # high uncertainty always → research
            elif energy < low_energy_threshold:
                action = "prune"
            else:
                action = "consolidate"

            gaps.append(KnowledgeGap(
                node_id=r.node_id,
                node_index=r.node_index,
                uncertainty=r.uncertainty,
                suggested_action=action,
            ))

        return gaps
