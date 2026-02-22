"""Type definitions for the NietzscheDB Python SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enums ──────────────────────────────────────────────────────────────────

class EdgeType(str, Enum):
    ASSOCIATION = "Association"
    HIERARCHICAL = "Hierarchical"
    LSYSTEM_GENERATED = "LSystemGenerated"
    PRUNED = "Pruned"


class CausalType(str, Enum):
    UNKNOWN = "Unknown"
    TIMELIKE = "Timelike"
    SPACELIKE = "Spacelike"
    LIGHTLIKE = "Lightlike"


# ── Core Types ─────────────────────────────────────────────────────────────

@dataclass
class Node:
    id: str
    coords: List[float]
    content: Dict[str, Any]
    node_type: str = ""
    energy: float = 1.0
    depth: float = 0.0
    hausdorff_local: float = 0.0
    created_at: int = 0
    expires_at: int = 0


@dataclass
class Edge:
    id: str
    from_id: str
    to_id: str
    edge_type: str = "Association"
    weight: float = 1.0


@dataclass
class QueryResult:
    """Result of an NQL query.

    Fields populated depend on the query type:
    - MATCH queries → nodes (and node_pairs for edge traversals)
    - DIFFUSE queries → path_ids
    - Aggregation queries → scalar_rows
    - EXPLAIN → explain
    """
    nodes: List[Node] = field(default_factory=list)
    node_pairs: List[tuple] = field(default_factory=list)
    path_ids: List[str] = field(default_factory=list)
    scalar_rows: List[Dict[str, Any]] = field(default_factory=list)
    explain: str = ""
    error: str = ""


@dataclass
class KnnResult:
    id: str
    distance: float


@dataclass
class SleepReport:
    hausdorff_before: float
    hausdorff_after: float
    hausdorff_delta: float
    semantic_drift_avg: float
    semantic_drift_max: float
    committed: bool
    nodes_perturbed: int
    snapshot_nodes: int


@dataclass
class ZaratustraReport:
    nodes_updated: int = 0
    mean_energy_before: float = 0.0
    mean_energy_after: float = 0.0
    total_energy_delta: float = 0.0
    echoes_created: int = 0
    echoes_evicted: int = 0
    total_echoes: int = 0
    elite_count: int = 0
    elite_threshold: float = 0.0
    mean_elite_energy: float = 0.0
    mean_base_energy: float = 0.0
    elite_node_ids: List[str] = field(default_factory=list)
    duration_ms: int = 0
    cycles_run: int = 0


@dataclass
class SynthesisResult:
    coords: List[float]
    nearest_node_id: str
    nearest_distance: float


@dataclass
class CausalEdgeInfo:
    edge_id: str
    from_node_id: str
    to_node_id: str
    minkowski_interval: float
    causal_type: str
    edge_type: str


@dataclass
class AlgorithmResult:
    scores: Dict[str, float] = field(default_factory=dict)
    communities: Dict[str, int] = field(default_factory=dict)
    duration_ms: int = 0
    iterations: int = 0


@dataclass
class BackupInfo:
    label: str
    path: str
    created_at: int
    size_bytes: int


@dataclass
class CollectionInfo:
    name: str
    dim: int = 0
    metric: str = ""
    node_count: int = 0
    edge_count: int = 0


@dataclass
class StatsResponse:
    node_count: int
    edge_count: int
    version: str = ""
    sensory_count: int = 0


@dataclass
class SensoryInfo:
    found: bool
    node_id: str
    modality: str
    dim: int = 0
    quant_level: str = ""
    reconstruction_quality: float = 0.0
    compression_ratio: float = 0.0
    encoder_version: int = 0
    byte_size: int = 0


@dataclass
class DiffusionScale:
    t: float
    scores: Dict[str, float]


# ── Option Types (for method parameters) ──────────────────────────────────

@dataclass
class InsertNodeOpts:
    id: Optional[str] = None
    coords: Optional[List[float]] = None
    content: Optional[Dict[str, Any]] = None
    node_type: str = ""
    energy: float = 1.0
    collection: str = ""
    ttl_seconds: int = 0


@dataclass
class InsertEdgeOpts:
    from_id: str = ""
    to_id: str = ""
    edge_type: str = "Association"
    weight: float = 1.0
    collection: str = ""
