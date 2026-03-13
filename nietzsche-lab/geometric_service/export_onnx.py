"""
ONNX export module for GeometricKernels feature maps.

Exports pre-computed feature maps to ONNX format for fast inference
in NietzscheDB's Rust engine via nietzsche-neural (ort/ONNX Runtime).

Feature map φ(x) ∈ R^M such that k(x,y) ≈ ⟨φ(x), φ(y)⟩
In Rust: similarity = dot(feature(node_a), feature(node_b))
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from geometric_kernels.kernels import MaternGeometricKernel
from geometric_kernels.feature_maps import default_feature_map


class _FeatureLookupModule(nn.Module if HAS_TORCH else object):
    """PyTorch module wrapping a pre-computed feature table for ONNX export."""

    def __init__(self, feature_table: np.ndarray):
        super().__init__()
        self.features = nn.Embedding.from_pretrained(
            torch.tensor(feature_table, dtype=torch.float32),
            freeze=True,
        )

    def forward(self, node_idx: torch.Tensor) -> torch.Tensor:
        return self.features(node_idx)


class GeometricFeatureMapExporter:
    """
    Export GeometricKernels feature maps to ONNX.

    Two export modes:

    1. Graph feature maps (discrete):
       Pre-compute features for ALL nodes → embedding table → ONNX lookup.
       Very fast inference: O(1) per node.

    2. Hyperbolic feature maps (continuous):
       Export the feature computation as a model that takes hyperboloid
       coordinates as input. Slower but works with new/unseen points.
    """

    def __init__(self, space, num_features: int = 256):
        """
        Args:
            space: GeometricKernels space (Graph or Hyperbolic)
            num_features: Number of random features for approximation
        """
        self.space = space
        self.num_features = num_features
        self.kernel = MaternGeometricKernel(space, num=num_features)
        self.feature_map = default_feature_map(space=space, num=num_features)

    def export_graph_features(
        self,
        output_path: str,
        nu: float = 2.5,
        lengthscale: float = 1.0,
        seed: int = 42,
    ) -> dict:
        """
        Export graph node features as ONNX embedding lookup table.

        The ONNX model: node_index (int64) → feature_vector (float32[M])

        In Rust (nietzsche-neural):
            feat_a = infer_f32("gk_features", [1], [node_idx_a])
            feat_b = infer_f32("gk_features", [1], [node_idx_b])
            similarity = dot(feat_a, feat_b)

        Args:
            output_path: Path for .onnx file
            nu: Matérn smoothness
            lengthscale: Length scale
            seed: Random seed for feature map sampling

        Returns:
            Summary dict with model info
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for ONNX export. pip install torch")

        params = self.kernel.init_params()
        params["nu"] = np.array([nu])
        params["lengthscale"] = np.array([lengthscale])

        n_nodes = self.space.num_vertices
        all_indices = np.arange(n_nodes).reshape(-1, 1).astype(np.int64)

        key = np.random.RandomState(seed)
        _, features = self.feature_map(all_indices, params, key=key)
        features = np.array(features, dtype=np.float32)

        # Normalize features for unit-variance kernel approximation
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features_normalized = features / norms

        logger.info(
            "Feature table: %d nodes × %d features, "
            "mean_norm=%.4f, std_norm=%.4f",
            n_nodes, features.shape[1],
            float(norms.mean()), float(norms.std()),
        )

        # Create ONNX model
        model = _FeatureLookupModule(features_normalized)
        dummy_input = torch.tensor([0], dtype=torch.long)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["node_index"],
            output_names=["feature_vector"],
            dynamic_axes={"node_index": {0: "batch_size"}},
            opset_version=17,
        )

        # Verify
        file_size = os.path.getsize(output_path)
        logger.info("Exported ONNX model: %s (%.2f MB)", output_path, file_size / 1e6)

        return {
            "path": output_path,
            "n_nodes": n_nodes,
            "n_features": int(features.shape[1]),
            "nu": nu,
            "lengthscale": lengthscale,
            "file_size_bytes": file_size,
            "mean_feature_norm": float(norms.mean()),
        }

    def export_kernel_matrix(
        self,
        output_path: str,
        nu: float = 2.5,
        lengthscale: float = 1.0,
    ) -> dict:
        """
        Export full kernel matrix as numpy file (.npz).

        Useful for calibration and analysis. Not for runtime inference.

        Args:
            output_path: Path for .npz file
            nu: Matérn smoothness
            lengthscale: Length scale

        Returns:
            Summary dict
        """
        params = self.kernel.init_params()
        params["nu"] = np.array([nu])
        params["lengthscale"] = np.array([lengthscale])

        n_nodes = self.space.num_vertices
        all_indices = np.arange(n_nodes).reshape(-1, 1).astype(np.int64)

        K = np.array(self.kernel.K(params, all_indices), dtype=np.float32)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savez_compressed(output_path, kernel_matrix=K, nu=nu, lengthscale=lengthscale)

        logger.info("Exported kernel matrix: %s (%d×%d)", output_path, n_nodes, n_nodes)

        return {
            "path": output_path,
            "shape": K.shape,
            "nu": nu,
            "lengthscale": lengthscale,
        }

    def verify_approximation(
        self,
        nu: float = 2.5,
        lengthscale: float = 1.0,
        n_pairs: int = 100,
        seed: int = 42,
    ) -> dict:
        """
        Verify that feature map approximation is accurate.

        Compares k(x,y) vs ⟨φ(x), φ(y)⟩ for random pairs.

        Returns:
            Dict with correlation, MAE, max error
        """
        params = self.kernel.init_params()
        params["nu"] = np.array([nu])
        params["lengthscale"] = np.array([lengthscale])

        n = self.space.num_vertices
        rng = np.random.RandomState(seed)

        # Random pairs
        pairs_a = rng.randint(0, n, size=n_pairs)
        pairs_b = rng.randint(0, n, size=n_pairs)

        # Exact kernel values
        exact_values = []
        for a, b in zip(pairs_a, pairs_b):
            X = np.array([[a]], dtype=np.int64)
            Y = np.array([[b]], dtype=np.int64)
            k = float(np.array(self.kernel.K(params, X, Y))[0, 0])
            exact_values.append(k)
        exact_values = np.array(exact_values)

        # Feature map approximation
        all_indices = np.arange(n).reshape(-1, 1).astype(np.int64)
        _, features = self.feature_map(all_indices, params, key=rng)
        features = np.array(features, dtype=np.float64)

        approx_values = np.array([
            float(features[a] @ features[b])
            for a, b in zip(pairs_a, pairs_b)
        ])

        correlation = float(np.corrcoef(exact_values, approx_values)[0, 1])
        mae = float(np.mean(np.abs(exact_values - approx_values)))
        max_error = float(np.max(np.abs(exact_values - approx_values)))

        return {
            "correlation": correlation,
            "mae": mae,
            "max_error": max_error,
            "n_pairs": n_pairs,
            "n_features": self.num_features,
            "nu": nu,
            "lengthscale": lengthscale,
        }
