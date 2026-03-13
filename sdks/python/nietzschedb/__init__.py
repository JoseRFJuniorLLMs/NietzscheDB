"""NietzscheDB Python SDK — multi-manifold graph database client."""

from nietzschedb.client import NietzscheClient
from nietzschedb.gemini_embed import GeminiEmbedder, exp_map_zero
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
    "GeminiEmbedder",
    "exp_map_zero",
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
