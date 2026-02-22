"""NietzscheDB Python SDK — multi-manifold graph database client."""

from nietzschedb.client import NietzscheClient
from nietzschedb.types import (
    AlgorithmResult,
    BackupInfo,
    CausalEdgeInfo,
    CausalType,
    CollectionInfo,
    DiffusionScale,
    Edge,
    EdgeType,
    InsertEdgeOpts,
    InsertNodeOpts,
    KnnResult,
    Node,
    QueryResult,
    SensoryInfo,
    SleepReport,
    StatsResponse,
    SynthesisResult,
    ZaratustraReport,
)

__version__ = "0.1.0"
__all__ = [
    "NietzscheClient",
    "AlgorithmResult",
    "BackupInfo",
    "CausalEdgeInfo",
    "CausalType",
    "CollectionInfo",
    "DiffusionScale",
    "Edge",
    "EdgeType",
    "InsertEdgeOpts",
    "InsertNodeOpts",
    "KnnResult",
    "Node",
    "QueryResult",
    "SensoryInfo",
    "SleepReport",
    "StatsResponse",
    "SynthesisResult",
    "ZaratustraReport",
]
