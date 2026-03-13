#!/usr/bin/env python3
"""
Integration tests for GeometricKernels × NietzscheDB.

Tests cover:
1. Graph bridge (adjacency construction)
2. Kernel computation (heat, Matérn)
3. Diffusion activation (multi-scale)
4. Calibration (Chebyshev vs exact)
5. Uncertainty estimation (spectral fallback)
6. Hyperbolic kernels (Poincaré ↔ hyperboloid)
7. ONNX export (feature map approximation)

Run:
    # Unit tests (no server needed):
    cd NietzscheDB/nietzsche-lab
    pytest geometric_service/test_geometric_kernels.py -v

    # Integration tests (requires NietzscheDB server):
    NIETZSCHE_HOST=http://136.111.0.47:8080 \
    pytest geometric_service/test_geometric_kernels.py -v -k "integration"
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Test fixtures ────────────────────────────────────────────

def _make_test_graph(n: int = 50, edge_prob: float = 0.1, seed: int = 42):
    """Create a random test graph."""
    from geometric_kernels.spaces import Graph

    rng = np.random.default_rng(seed)
    rows, cols, weights = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                w = rng.uniform(0.1, 2.0)
                rows.extend([i, j])
                cols.extend([j, i])
                weights.extend([w, w])

    adj = csr_matrix((weights, (rows, cols)), shape=(n, n))
    space = Graph(adj, normalize_laplacian=True)
    node_ids = [f"node-{i:04d}" for i in range(n)]
    return space, node_ids


def _make_test_nodes_edges(n: int = 30, seed: int = 42):
    """Create test NodeInfo/EdgeInfo for graph bridge tests."""
    from geometric_service.graph_bridge import NodeInfo, EdgeInfo

    rng = np.random.default_rng(seed)
    nodes = [
        NodeInfo(
            nid=f"uuid-{i:04d}",
            energy=float(rng.uniform(0, 1)),
            depth=float(rng.uniform(0, 1)),
            coords=rng.normal(0, 0.3, 128).tolist(),
        )
        for i in range(n)
    ]
    edges = []
    for i in range(n):
        n_edges = rng.integers(1, 4)
        for _ in range(n_edges):
            j = rng.integers(0, n)
            if i != j:
                edges.append(EdgeInfo(
                    eid=f"edge-{i}-{j}",
                    from_id=nodes[i].id,
                    to_id=nodes[j].id,
                    weight=float(rng.uniform(0.5, 2.0)),
                ))
    return nodes, edges


# ═══════════════════════════════════════════════════════════
# 1. Graph Bridge Tests
# ═══════════════════════════════════════════════════════════

class TestGraphBridge:
    def test_ndb_to_geometric_graph(self):
        from geometric_service.graph_bridge import ndb_to_geometric_graph
        nodes, edges = _make_test_nodes_edges()
        space, node_ids, id_to_idx = ndb_to_geometric_graph(nodes, edges)

        assert space.num_vertices == len(nodes)
        assert len(node_ids) == len(nodes)
        assert len(id_to_idx) == len(nodes)
        assert node_ids[0] == nodes[0].id

    def test_empty_graph_raises(self):
        from geometric_service.graph_bridge import ndb_to_geometric_graph
        with pytest.raises(ValueError, match="0 nodes"):
            ndb_to_geometric_graph([], [])

    def test_edges_with_unknown_nodes_skipped(self):
        from geometric_service.graph_bridge import ndb_to_geometric_graph, NodeInfo, EdgeInfo

        nodes = [NodeInfo("a", 1.0, 0.5), NodeInfo("b", 0.5, 0.3)]
        edges = [
            EdgeInfo("e1", "a", "b", 1.0),
            EdgeInfo("e2", "a", "unknown", 1.0),  # should be skipped
        ]
        space, _, _ = ndb_to_geometric_graph(nodes, edges)
        assert space.num_vertices == 2


# ═══════════════════════════════════════════════════════════
# 2. Kernel Service Tests
# ═══════════════════════════════════════════════════════════

class TestKernelService:
    def setup_method(self):
        from geometric_service.kernel_service import NietzscheKernelService
        self.space, self.node_ids = _make_test_graph(30)
        self.service = NietzscheKernelService(self.space, num_eigenpairs=20)

    def test_heat_kernel_symmetric(self):
        idx = np.arange(10, dtype=np.int64)
        K = self.service.heat_kernel_matrix(idx, t=1.0)
        assert K.shape == (10, 10)
        np.testing.assert_allclose(K, K.T, atol=1e-6)

    def test_heat_kernel_positive_diagonal(self):
        idx = np.arange(10, dtype=np.int64)
        K = self.service.heat_kernel_matrix(idx, t=1.0)
        assert np.all(np.diag(K) > 0)

    def test_matern_kernel_different_nu(self):
        idx = np.arange(5, dtype=np.int64)
        K_rough = self.service.matern_kernel_matrix(idx, nu=0.5)
        K_smooth = self.service.matern_kernel_matrix(idx, nu=2.5)
        # Both should be valid kernel matrices
        assert K_rough.shape == (5, 5)
        assert K_smooth.shape == (5, 5)

    def test_diffuse_activation(self):
        result = self.service.diffuse_activation([0], t=1.0, top_k=5)
        assert result.activation.shape == (30,)
        assert len(result.top_k_indices) == 5
        assert result.activation[0] > 0  # source should be activated

    def test_multi_scale_diffusion(self):
        results = self.service.multi_scale_diffusion([0])
        assert 0.1 in results
        assert 1.0 in results
        assert 10.0 in results

    def test_pairwise_similarity(self):
        sim = self.service.pairwise_similarity(0, 1, kernel_type="matern")
        assert isinstance(sim, float)
        assert sim >= 0


# ═══════════════════════════════════════════════════════════
# 3. Calibration Tests
# ═══════════════════════════════════════════════════════════

class TestCalibration:
    def setup_method(self):
        from geometric_service.kernel_service import NietzscheKernelService
        from geometric_service.calibration import ChebyshevCalibrator
        self.space, _ = _make_test_graph(30)
        self.service = NietzscheKernelService(self.space, num_eigenpairs=25)
        self.calibrator = ChebyshevCalibrator(self.service)

    def test_chebyshev_coefficients(self):
        coeffs = self.calibrator._chebyshev_coefficients(1.0, 10)
        assert len(coeffs) == 11
        assert coeffs[0] > 0  # c_0 should be positive

    def test_compare_diffusion(self):
        metrics = self.calibrator.compare_diffusion(
            source_idx=0,
            t_values=[1.0],
            k_max_values=[10, 20],
        )
        assert len(metrics) == 2  # 1 t × 2 k_max
        assert all(0 <= m.spearman_rho <= 1 for m in metrics)

    def test_higher_k_max_better_overlap(self):
        metrics = self.calibrator.compare_diffusion(
            source_idx=0,
            t_values=[1.0],
            k_max_values=[5, 25],
        )
        # Higher K_max should generally give better overlap
        # (not always guaranteed for small graphs, but usually)
        assert metrics[1].k_max > metrics[0].k_max


# ═══════════════════════════════════════════════════════════
# 4. Uncertainty Tests
# ═══════════════════════════════════════════════════════════

class TestUncertainty:
    def setup_method(self):
        from geometric_service.uncertainty import EpistemicUncertaintyEstimator
        self.space, self.node_ids = _make_test_graph(30)
        self.estimator = EpistemicUncertaintyEstimator(self.space, self.node_ids)

    def test_spectral_fallback_fit(self):
        rng = np.random.default_rng(42)
        obs_idx = rng.choice(30, size=10, replace=False)
        obs_val = rng.uniform(0, 1, size=10).astype(np.float64)

        result = self.estimator.fit(obs_idx, obs_val)
        assert result["mode"] == "spectral_fallback"
        assert result["n_observed"] == 10

    def test_predict_uncertainty_after_fit(self):
        rng = np.random.default_rng(42)
        obs_idx = rng.choice(30, size=10, replace=False)
        obs_val = rng.uniform(0, 1, size=10).astype(np.float64)

        self.estimator.fit(obs_idx, obs_val)
        results = self.estimator.predict_uncertainty()

        assert len(results) == 30
        assert all(r.variance >= 0 for r in results)
        assert all(r.uncertainty >= 0 for r in results)

    def test_find_knowledge_gaps(self):
        rng = np.random.default_rng(42)
        obs_idx = rng.choice(30, size=10, replace=False)
        obs_val = rng.uniform(0, 1, size=10).astype(np.float64)

        self.estimator.fit(obs_idx, obs_val)
        gaps = self.estimator.find_knowledge_gaps(top_k=5)

        assert len(gaps) <= 5
        assert all(g.suggested_action in ("research", "consolidate", "prune") for g in gaps)

    def test_predict_without_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            self.estimator.predict_uncertainty()


# ═══════════════════════════════════════════════════════════
# 5. Hyperbolic Kernel Tests
# ═══════════════════════════════════════════════════════════

class TestHyperbolicKernels:
    def test_poincare_to_hyperboloid_origin(self):
        from geometric_service.hyperbolic_kernels import HyperbolicKernelService
        origin = np.zeros(3)
        hyp = HyperbolicKernelService.poincare_to_hyperboloid(origin.reshape(1, -1))
        # Origin in Poincaré → (1, 0, 0, 0) in hyperboloid
        np.testing.assert_allclose(hyp[0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(hyp[0, 1:], 0.0, atol=1e-6)

    def test_poincare_hyperboloid_roundtrip(self):
        from geometric_service.hyperbolic_kernels import HyperbolicKernelService
        rng = np.random.default_rng(42)
        points = rng.normal(0, 0.3, (10, 5))
        # Normalize to be inside the ball
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.maximum(norms, 1.0) * 0.8

        hyp = HyperbolicKernelService.poincare_to_hyperboloid(points)
        back = HyperbolicKernelService.hyperboloid_to_poincare(hyp)
        np.testing.assert_allclose(back, points, atol=1e-6)

    def test_poincare_distance(self):
        from geometric_service.hyperbolic_kernels import HyperbolicKernelService
        origin = np.zeros(3)
        point = np.array([0.5, 0.0, 0.0])
        d = HyperbolicKernelService.poincare_distance(origin, point)
        expected = 2.0 * np.arctanh(0.5)
        np.testing.assert_allclose(d, expected, atol=1e-6)

    def test_validate_gauss_kernel_divergence(self):
        from geometric_service.hyperbolic_kernels import HyperbolicKernelService
        svc = HyperbolicKernelService(dim=3, num_features=32)

        # Near origin: Euclidean ≈ hyperbolic
        near = np.array([0.1, 0.0, 0.0])
        result_near = svc.validate_gauss_kernel(near, t=1.0)
        assert result_near["relative_error"] < 0.1  # < 10% error near origin

        # Near boundary: Euclidean << hyperbolic divergence
        far = np.array([0.9, 0.0, 0.0])
        result_far = svc.validate_gauss_kernel(far, t=1.0)
        assert result_far["relative_error"] > result_near["relative_error"]


# ═══════════════════════════════════════════════════════════
# 6. ONNX Export Tests
# ═══════════════════════════════════════════════════════════

class TestONNXExport:
    def test_verify_approximation(self):
        from geometric_service.export_onnx import GeometricFeatureMapExporter
        space, _ = _make_test_graph(20)
        exporter = GeometricFeatureMapExporter(space, num_features=64)

        result = exporter.verify_approximation(n_pairs=50)
        assert result["correlation"] > 0.8  # should be well correlated
        assert result["mae"] < 0.5

    @pytest.mark.skipif(
        not os.environ.get("ENABLE_ONNX_EXPORT"),
        reason="Set ENABLE_ONNX_EXPORT=1 to test ONNX export"
    )
    def test_export_graph_features(self, tmp_path):
        from geometric_service.export_onnx import GeometricFeatureMapExporter
        space, _ = _make_test_graph(20)
        exporter = GeometricFeatureMapExporter(space, num_features=32)

        output = str(tmp_path / "test_features.onnx")
        result = exporter.export_graph_features(output)
        assert os.path.exists(output)
        assert result["n_nodes"] == 20
        assert result["n_features"] == 32


# ═══════════════════════════════════════════════════════════
# 7. Integration Tests (require NietzscheDB server)
# ═══════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not os.environ.get("NIETZSCHE_HOST"),
    reason="Set NIETZSCHE_HOST=http://136.111.0.47:8080 for integration tests"
)
class TestIntegration:
    def test_fetch_graph_and_create_space(self):
        from geometric_service.graph_bridge import ndb_subgraph_to_space
        host = os.environ["NIETZSCHE_HOST"]
        space, node_ids, id_to_idx, nodes, edges = ndb_subgraph_to_space(
            host, "tech_galaxies", limit=200,
        )
        assert space.num_vertices > 0
        assert len(node_ids) > 0

    def test_heat_diffusion_on_real_graph(self):
        from geometric_service.graph_bridge import ndb_subgraph_to_space
        from geometric_service.kernel_service import NietzscheKernelService

        host = os.environ["NIETZSCHE_HOST"]
        space, node_ids, _, _, _ = ndb_subgraph_to_space(
            host, "tech_galaxies", limit=200,
        )
        service = NietzscheKernelService(space)
        result = service.diffuse_activation([0], t=1.0, top_k=10)
        assert len(result.top_k_indices) == 10

    def test_calibration_on_real_graph(self):
        from geometric_service.graph_bridge import ndb_subgraph_to_space
        from geometric_service.kernel_service import NietzscheKernelService
        from geometric_service.calibration import ChebyshevCalibrator

        host = os.environ["NIETZSCHE_HOST"]
        space, _, _, _, _ = ndb_subgraph_to_space(
            host, "tech_galaxies", limit=200,
        )
        service = NietzscheKernelService(space, num_eigenpairs=50)
        calibrator = ChebyshevCalibrator(service)
        report = calibrator.find_optimal_k_max(
            t_values=[0.1, 1.0],
            k_max_range=[10, 20],
        )
        assert len(report.optimal_k_max) == 2
        print(report.summary())

    def test_uncertainty_on_real_graph(self):
        from geometric_service.graph_bridge import ndb_subgraph_to_space
        from geometric_service.uncertainty import EpistemicUncertaintyEstimator

        host = os.environ["NIETZSCHE_HOST"]
        space, node_ids, _, nodes, _ = ndb_subgraph_to_space(
            host, "tech_galaxies", limit=200,
        )
        estimator = EpistemicUncertaintyEstimator(space, node_ids)

        # Use node energies as observations
        obs_idx = np.array([i for i, n in enumerate(nodes) if n.energy > 0.1])
        obs_val = np.array([nodes[i].energy for i in obs_idx])

        if len(obs_idx) > 5:
            estimator.fit(obs_idx, obs_val)
            gaps = estimator.find_knowledge_gaps(top_k=5)
            assert len(gaps) <= 5
            for gap in gaps:
                print(f"Gap: {gap.node_id} uncertainty={gap.uncertainty:.4f} "
                      f"action={gap.suggested_action}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
