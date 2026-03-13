"""
GeometricKernels integration for NietzscheDB.

Provides mathematically exact heat/Matérn kernels on graphs and hyperbolic
spaces, GP-based epistemic uncertainty estimation, Chebyshev calibration,
and ONNX export for Rust runtime inference.
"""

from .graph_bridge import ndb_to_geometric_graph, ndb_subgraph_to_space
from .kernel_service import NietzscheKernelService
from .calibration import ChebyshevCalibrator
from .uncertainty import EpistemicUncertaintyEstimator
from .hyperbolic_kernels import HyperbolicKernelService
from .export_onnx import GeometricFeatureMapExporter

__all__ = [
    "ndb_to_geometric_graph",
    "ndb_subgraph_to_space",
    "NietzscheKernelService",
    "ChebyshevCalibrator",
    "EpistemicUncertaintyEstimator",
    "HyperbolicKernelService",
    "GeometricFeatureMapExporter",
]
