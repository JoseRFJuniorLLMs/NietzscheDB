#!/usr/bin/env python3
"""
Runner script for GeometricKernels × NietzscheDB integration.

Usage:
    # Full pipeline on a collection (calibration + uncertainty + report):
    python run_geometric_kernels.py --host http://136.111.0.47:8080 --collection tech_galaxies

    # Calibration only:
    python run_geometric_kernels.py --host http://136.111.0.47:8080 --collection tech_galaxies --mode calibrate

    # Uncertainty analysis only:
    python run_geometric_kernels.py --host http://136.111.0.47:8080 --collection tech_galaxies --mode uncertainty

    # Export ONNX features:
    python run_geometric_kernels.py --host http://136.111.0.47:8080 --collection tech_galaxies --mode export

    # Quick test (small subgraph):
    python run_geometric_kernels.py --host http://136.111.0.47:8080 --collection tech_galaxies --limit 100 --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometric_service.graph_bridge import ndb_subgraph_to_space
from geometric_service.kernel_service import NietzscheKernelService
from geometric_service.calibration import ChebyshevCalibrator
from geometric_service.uncertainty import EpistemicUncertaintyEstimator
from geometric_service.hyperbolic_kernels import HyperbolicKernelService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("gk_runner")


def run_calibration(service, space, nodes, edges, collection, args):
    """Run Chebyshev calibration analysis."""
    logger.info("=== Phase 2: Chebyshev Calibration ===")
    calibrator = ChebyshevCalibrator(service)

    t_values = [0.1, 1.0, 10.0] if not args.quick else [1.0]
    k_max_range = [5, 10, 15, 20, 30, 40, 50] if not args.quick else [10, 20, 30]

    report = calibrator.find_optimal_k_max(
        t_values=t_values,
        k_max_range=k_max_range,
        target_overlap=0.90,
    )
    report.collection = collection
    report.n_edges = len(edges)

    print()
    print(report.summary())
    print()

    # Save report
    output_path = f"gk_calibration_{collection}.json"
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Saved calibration report: %s", output_path)

    return report


def run_uncertainty(estimator, nodes, node_ids, collection, args):
    """Run epistemic uncertainty analysis."""
    logger.info("=== Phase 3: Epistemic Uncertainty ===")

    # Use node energies as observations
    obs_idx = np.array([i for i, n in enumerate(nodes) if n.energy > 0.1])
    obs_val = np.array([nodes[i].energy for i in obs_idx], dtype=np.float64)

    if len(obs_idx) < 3:
        logger.warning("Too few observed nodes (%d) for GP training", len(obs_idx))
        return None

    t0 = time.time()
    fit_result = estimator.fit(obs_idx, obs_val, training_iterations=50)
    fit_time = time.time() - t0
    logger.info("GP fit: mode=%s, time=%.2fs", fit_result["mode"], fit_time)

    # Find knowledge gaps
    node_energies = {n.id: n.energy for n in nodes}
    gaps = estimator.find_knowledge_gaps(
        top_k=args.top_k,
        node_energies=node_energies,
    )

    print()
    print(f"=== Knowledge Gaps ({collection}) ===")
    print(f"{'Rank':>4} {'Node ID':>40} {'Uncertainty':>12} {'Action':>12}")
    print("-" * 72)
    for i, gap in enumerate(gaps):
        print(f"{i+1:>4} {gap.node_id:>40} {gap.uncertainty:>12.6f} {gap.suggested_action:>12}")
    print()

    # Save gaps
    output_path = f"gk_gaps_{collection}.json"
    with open(output_path, "w") as f:
        json.dump([
            {
                "node_id": g.node_id,
                "node_index": g.node_index,
                "uncertainty": g.uncertainty,
                "suggested_action": g.suggested_action,
            }
            for g in gaps
        ], f, indent=2)
    logger.info("Saved knowledge gaps: %s", output_path)

    return gaps


def run_diffusion_demo(service, node_ids, collection, args):
    """Run multi-scale diffusion demo."""
    logger.info("=== Diffusion Demo ===")

    results = service.multi_scale_diffusion([0], top_k=10)

    for t, result in sorted(results.items()):
        print(f"\n--- t={t:.1f} (", end="")
        if t <= 0.1:
            print("focused recall) ---")
        elif t <= 1.0:
            print("associative thinking) ---")
        else:
            print("free association) ---")

        for i, (idx, val) in enumerate(zip(result.top_k_indices, result.top_k_values)):
            node_id = node_ids[idx] if idx < len(node_ids) else f"idx-{idx}"
            print(f"  {i+1:>2}. [{idx:>4}] {node_id[:36]:>36} activation={val:.6f}")


def run_hyperbolic_validation(nodes, args):
    """Validate GAUSS_KERNEL vs correct hyperbolic heat kernel."""
    logger.info("=== Phase 4: Hyperbolic Kernel Validation ===")

    svc = HyperbolicKernelService(dim=128, num_features=32)

    print()
    print("=== GAUSS_KERNEL Validation (Euclidean vs Hyperbolic) ===")
    print(f"{'‖x‖':>6} {'d_P(x,0)':>10} {'Eucl kernel':>12} {'Hyp kernel':>12} {'Rel Error':>10}")
    print("-" * 56)

    for node in nodes[:20]:
        if not node.coords or len(node.coords) < 128:
            continue
        coords = np.array(node.coords[:128], dtype=np.float64)
        norm = float(np.linalg.norm(coords))
        if norm >= 1.0:
            coords = coords / norm * 0.95  # project back into ball

        result = svc.validate_gauss_kernel(coords, t=1.0)
        print(
            f"{result['poincare_norm']:>6.3f} "
            f"{result['poincare_distance_from_origin']:>10.4f} "
            f"{result['euclidean_kernel']:>12.6f} "
            f"{result['hyperbolic_kernel']:>12.6f} "
            f"{result['relative_error']:>10.2%}"
        )
    print()


def run_export(service, space, collection, args):
    """Export feature maps to ONNX."""
    logger.info("=== Phase 5: ONNX Export ===")

    try:
        from geometric_service.export_onnx import GeometricFeatureMapExporter
    except ImportError:
        logger.error("PyTorch required for ONNX export: pip install torch")
        return

    exporter = GeometricFeatureMapExporter(space, num_features=128)

    # Verify approximation quality first
    verify = exporter.verify_approximation(n_pairs=100)
    print(f"\nFeature map approximation quality:")
    print(f"  Correlation: {verify['correlation']:.4f}")
    print(f"  MAE: {verify['mae']:.6f}")
    print(f"  Max Error: {verify['max_error']:.6f}")

    if verify["correlation"] < 0.9:
        logger.warning("Low feature map correlation (%.4f) — increase num_features", verify["correlation"])

    # Export
    output_path = f"gk_features_{collection}.onnx"
    result = exporter.export_graph_features(output_path)
    print(f"\nExported ONNX model: {output_path}")
    print(f"  Nodes: {result['n_nodes']}")
    print(f"  Features: {result['n_features']}")
    print(f"  Size: {result['file_size_bytes'] / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="GeometricKernels × NietzscheDB Runner"
    )
    parser.add_argument(
        "--host", default="http://136.111.0.47:8080",
        help="NietzscheDB HTTP API host"
    )
    parser.add_argument(
        "--collection", default="tech_galaxies",
        help="Collection to analyze"
    )
    parser.add_argument(
        "--limit", type=int, default=1000,
        help="Max nodes to fetch"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["all", "calibrate", "uncertainty", "diffusion", "hyperbolic", "export"],
        help="What to run"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode (fewer scales, fewer K_max values)"
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Top-K knowledge gaps to find"
    )
    parser.add_argument(
        "--eigenpairs", type=int, default=None,
        help="Number of Laplacian eigenpairs (default: min(200, n))"
    )

    args = parser.parse_args()

    # Fetch graph
    logger.info("Fetching graph from %s/%s (limit=%d)...", args.host, args.collection, args.limit)
    t0 = time.time()
    space, node_ids, id_to_idx, nodes, edges = ndb_subgraph_to_space(
        args.host, args.collection, args.limit,
    )
    fetch_time = time.time() - t0
    logger.info("Fetched %d nodes, %d edges in %.2fs", len(nodes), len(edges), fetch_time)

    # Create kernel service
    service = NietzscheKernelService(space, num_eigenpairs=args.eigenpairs)

    # Run requested analyses
    if args.mode in ("all", "calibrate"):
        run_calibration(service, space, nodes, edges, args.collection, args)

    if args.mode in ("all", "uncertainty"):
        estimator = EpistemicUncertaintyEstimator(space, node_ids)
        run_uncertainty(estimator, nodes, node_ids, args.collection, args)

    if args.mode in ("all", "diffusion"):
        run_diffusion_demo(service, node_ids, args.collection, args)

    if args.mode in ("all", "hyperbolic"):
        run_hyperbolic_validation(nodes, args)

    if args.mode in ("all", "export"):
        run_export(service, space, args.collection, args)

    logger.info("Done!")


if __name__ == "__main__":
    main()
