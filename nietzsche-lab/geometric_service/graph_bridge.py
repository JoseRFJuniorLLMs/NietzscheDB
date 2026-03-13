"""
Bridge between NietzscheDB graph data and GeometricKernels Graph space.

Converts NietzscheDB nodes/edges (via gRPC or HTTP) into scipy sparse
adjacency matrices suitable for GeometricKernels' Graph space.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from geometric_kernels.spaces import Graph

logger = logging.getLogger(__name__)


class NodeInfo:
    """Lightweight node representation from NietzscheDB."""

    __slots__ = ("id", "energy", "depth", "coords", "content")

    def __init__(self, nid: str, energy: float, depth: float,
                 coords: list[float] | None = None,
                 content: dict | None = None):
        self.id = nid
        self.energy = energy
        self.depth = depth
        self.coords = coords or []
        self.content = content or {}


class EdgeInfo:
    """Lightweight edge representation from NietzscheDB."""

    __slots__ = ("id", "from_id", "to_id", "weight", "edge_type")

    def __init__(self, eid: str, from_id: str, to_id: str,
                 weight: float = 1.0, edge_type: str = ""):
        self.id = eid
        self.from_id = from_id
        self.to_id = to_id
        self.weight = weight
        self.edge_type = edge_type


def _fetch_json(url: str, timeout: int = 30) -> Any:
    """Fetch JSON from NietzscheDB HTTP API."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_graph_http(host: str, collection: str,
                     limit: int = 5000) -> tuple[list[NodeInfo], list[EdgeInfo]]:
    """
    Fetch nodes and edges from NietzscheDB HTTP API.

    Args:
        host: HTTP base URL (e.g. "http://136.111.0.47:8080")
        collection: Collection name
        limit: Maximum nodes to fetch

    Returns:
        (nodes, edges) lists
    """
    url = f"{host}/api/graph?collection={collection}&limit={limit}"
    data = _fetch_json(url)

    nodes = []
    for n in data.get("nodes", []):
        nodes.append(NodeInfo(
            nid=n.get("id", ""),
            energy=float(n.get("energy", 0.0)),
            depth=float(n.get("depth", 0.0)),
            coords=n.get("coords", []),
            content=n.get("content", {}),
        ))

    edges = []
    for e in data.get("edges", []):
        edges.append(EdgeInfo(
            eid=e.get("id", ""),
            from_id=e.get("from", ""),
            to_id=e.get("to", ""),
            weight=float(e.get("weight", 1.0)),
            edge_type=e.get("edge_type", ""),
        ))

    return nodes, edges


def ndb_to_geometric_graph(
    nodes: list[NodeInfo],
    edges: list[EdgeInfo],
    normalize_laplacian: bool = True,
) -> tuple[Graph, list[str], dict[str, int]]:
    """
    Convert NietzscheDB nodes/edges into a GeometricKernels Graph space.

    Args:
        nodes: List of NodeInfo from NietzscheDB
        edges: List of EdgeInfo from NietzscheDB
        normalize_laplacian: Use symmetric normalized Laplacian
            (consistent with nietzsche-pregel's L̃ = I − D^{−½}AD^{−½})

    Returns:
        (graph_space, node_ids, id_to_idx) where:
        - graph_space: GeometricKernels Graph object
        - node_ids: ordered list of UUID strings
        - id_to_idx: mapping from UUID string to integer index
    """
    node_ids = [n.id for n in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    if n == 0:
        raise ValueError("Cannot create Graph space with 0 nodes")

    rows, cols, weights = [], [], []
    skipped = 0

    for e in edges:
        if e.from_id not in id_to_idx or e.to_id not in id_to_idx:
            skipped += 1
            continue
        if e.from_id == e.to_id:
            continue  # skip self-loops

        i = id_to_idx[e.from_id]
        j = id_to_idx[e.to_id]
        w = max(e.weight, 1e-6)  # avoid zero weights

        # Undirected: add both directions
        rows.extend([i, j])
        cols.extend([j, i])
        weights.extend([w, w])

    if skipped > 0:
        logger.warning("Skipped %d edges with unknown node IDs", skipped)

    adj = csr_matrix((weights, (rows, cols)), shape=(n, n))

    # Remove duplicate entries (sum them)
    adj.sum_duplicates()

    space = Graph(adj, normalize_laplacian=normalize_laplacian)

    logger.info(
        "Created Graph space: %d nodes, %d edges (undirected), "
        "normalize_laplacian=%s",
        n, len(edges) - skipped, normalize_laplacian,
    )

    return space, node_ids, id_to_idx


def ndb_subgraph_to_space(
    host: str,
    collection: str,
    limit: int = 5000,
    normalize_laplacian: bool = True,
) -> tuple[Graph, list[str], dict[str, int], list[NodeInfo], list[EdgeInfo]]:
    """
    Convenience: fetch from HTTP API and create Graph space in one call.

    Returns:
        (graph_space, node_ids, id_to_idx, nodes, edges)
    """
    nodes, edges = fetch_graph_http(host, collection, limit)
    space, node_ids, id_to_idx = ndb_to_geometric_graph(
        nodes, edges, normalize_laplacian
    )
    return space, node_ids, id_to_idx, nodes, edges
