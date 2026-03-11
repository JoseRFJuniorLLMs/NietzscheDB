"""
NietzscheLab — gRPC client wrapper for NietzscheDB.

Provides typed access to the NietzscheDB gRPC API for the lab runner.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import grpc

# Generated proto stubs (run: python -m grpc_tools.protoc ...)
# For now we use the HTTP/REST API as fallback via requests
import requests


@dataclass
class NodeInfo:
    """Lightweight node representation from API responses."""
    id: str
    depth: float
    energy: float
    node_type: str
    content: dict[str, Any]
    hausdorff_local: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeInfo:
    """Lightweight edge representation."""
    id: str
    from_id: str
    to_id: str
    edge_type: str
    weight: float
    metadata: dict[str, Any] = field(default_factory=dict)


class NietzscheClient:
    """Client for NietzscheDB using the HTTP dashboard API.

    Uses the REST endpoints (port 8080) which are simpler to use
    from Python without proto compilation. The gRPC port (50051)
    is used for mutations via the grpcio library when proto stubs
    are available.
    """

    def __init__(self, http_base: str = "http://localhost:8080",
                 grpc_target: str = "localhost:50051",
                 collection: str = "default"):
        self.http_base = http_base.rstrip("/")
        self.grpc_target = grpc_target
        self.collection = collection
        self._grpc_channel = None
        self._grpc_stub = None

    def _http_get(self, path: str, params: dict | None = None) -> dict:
        """Make a GET request to the HTTP API."""
        if params is None:
            params = {}
        params.setdefault("collection", self.collection)
        resp = requests.get(f"{self.http_base}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _http_post(self, path: str, data: dict | None = None) -> dict:
        """Make a POST request to the HTTP API."""
        resp = requests.post(f"{self.http_base}{path}", json=data or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ── Node operations ─────────────────────────────────────────────

    def get_graph(self, limit: int = 100) -> dict:
        """Get nodes and edges from a collection."""
        return self._http_get("/api/graph", {"limit": limit})

    def get_node(self, node_id: str) -> NodeInfo | None:
        """Get a single node by ID."""
        try:
            data = self._http_get(f"/api/node/{node_id}")
            return self._parse_node(data)
        except requests.HTTPError:
            return None

    def search_nodes(self, query: str, limit: int = 50) -> list[NodeInfo]:
        """Full-text search for nodes."""
        data = self._http_get("/api/search", {"q": query, "limit": limit})
        return [self._parse_node(n) for n in data.get("nodes", [])]

    def get_stats(self) -> dict:
        """Get collection statistics."""
        return self._http_get("/api/stats")

    def get_collections(self) -> list[dict]:
        """List all collections."""
        return self._http_get("/api/collections")

    # ── Algorithm endpoints ─────────────────────────────────────────

    def run_pagerank(self) -> list[dict]:
        """Run PageRank and return scored nodes."""
        return self._http_get("/api/algo/pagerank")

    def run_louvain(self) -> dict:
        """Run Louvain community detection."""
        return self._http_get("/api/algo/louvain")

    def run_degree(self) -> list[dict]:
        """Get degree centrality."""
        return self._http_get("/api/algo/degree")

    # ── Agency endpoints ────────────────────────────────────────────

    def get_agency_dashboard(self) -> dict:
        """Get full cognitive dashboard."""
        return self._http_get("/api/agency/dashboard")

    def get_observation(self) -> dict:
        """Get thermodynamic observation frame."""
        return self._http_get("/api/agency/observation")

    def get_health(self) -> dict:
        """Get latest health report."""
        return self._http_get("/api/agency/health/latest")

    # ── Backup/Restore (via HTTP — requires gRPC for full support) ──

    def create_backup_http(self) -> dict:
        """Trigger a backup snapshot via HTTP (if endpoint exists)."""
        try:
            return self._http_post("/api/backup/create")
        except (requests.HTTPError, Exception):
            # Fallback: backup not available via HTTP, would need gRPC
            return {"status": "backup_not_available_via_http"}

    # ── Mutations via gRPC ──────────────────────────────────────────

    def _ensure_grpc(self):
        """Lazy-init gRPC channel and stub."""
        if self._grpc_channel is None:
            self._grpc_channel = grpc.insecure_channel(self.grpc_target)
            try:
                from nietzsche_pb2_grpc import NietzscheDBStub
                self._grpc_stub = NietzscheDBStub(self._grpc_channel)
            except ImportError:
                self._grpc_stub = None

    def insert_node_grpc(self, content: dict, node_type: str = "Semantic",
                         coords: list[float] | None = None,
                         energy: float = 0.5) -> str | None:
        """Insert a node via gRPC. Returns node ID or None."""
        self._ensure_grpc()
        if self._grpc_stub is None:
            return self._insert_node_http(content, node_type, coords, energy)
        try:
            from nietzsche_pb2 import InsertNodeRequest
            req = InsertNodeRequest(
                collection=self.collection,
                content=json.dumps(content),
                node_type=node_type,
                coords=coords or [0.0] * 128,
                energy=energy,
            )
            resp = self._grpc_stub.InsertNode(req)
            return resp.id
        except Exception as e:
            print(f"[gRPC insert_node error] {e}")
            return None

    def insert_edge_grpc(self, from_id: str, to_id: str,
                         edge_type: str = "Association",
                         weight: float = 1.0) -> str | None:
        """Insert an edge via gRPC. Returns edge ID or None."""
        self._ensure_grpc()
        if self._grpc_stub is None:
            return self._insert_edge_http(from_id, to_id, edge_type, weight)
        try:
            from nietzsche_pb2 import InsertEdgeRequest
            req = InsertEdgeRequest(
                collection=self.collection,
                from_id=from_id,
                to_id=to_id,
                edge_type=edge_type,
                weight=weight,
            )
            resp = self._grpc_stub.InsertEdge(req)
            return resp.id
        except Exception as e:
            print(f"[gRPC insert_edge error] {e}")
            return None

    def delete_edge_grpc(self, edge_id: str) -> bool:
        """Delete an edge via gRPC."""
        self._ensure_grpc()
        if self._grpc_stub is None:
            return False
        try:
            from nietzsche_pb2 import EdgeIdRequest
            req = EdgeIdRequest(collection=self.collection, id=edge_id)
            resp = self._grpc_stub.DeleteEdge(req)
            return resp.status == "ok"
        except Exception as e:
            print(f"[gRPC delete_edge error] {e}")
            return False

    def delete_node_grpc(self, node_id: str) -> bool:
        """Delete a node via gRPC."""
        self._ensure_grpc()
        if self._grpc_stub is None:
            return False
        try:
            from nietzsche_pb2 import NodeIdRequest
            req = NodeIdRequest(collection=self.collection, id=node_id)
            resp = self._grpc_stub.DeleteNode(req)
            return resp.status == "ok"
        except Exception as e:
            print(f"[gRPC delete_node error] {e}")
            return False

    # ── HTTP mutation fallbacks ─────────────────────────────────────

    def _insert_node_http(self, content: dict, node_type: str,
                          coords: list[float] | None, energy: float) -> str | None:
        """Fallback: insert node via NQL query over HTTP."""
        nql = (
            f'CREATE (n:{node_type} '
            f'{{content: {json.dumps(json.dumps(content))}, '
            f'energy: {energy}}}) RETURN n'
        )
        try:
            resp = self._http_post("/api/query", {"nql": nql, "collection": self.collection})
            nodes = resp.get("nodes", [])
            if nodes:
                return nodes[0].get("id")
        except Exception as e:
            print(f"[HTTP insert_node error] {e}")
        return None

    def _insert_edge_http(self, from_id: str, to_id: str,
                          edge_type: str, weight: float) -> str | None:
        """Fallback: insert edge via NQL."""
        nql = (
            f'MATCH (a), (b) WHERE a.id = "{from_id}" AND b.id = "{to_id}" '
            f'CREATE (a)-[:{edge_type} {{weight: {weight}}}]->(b) RETURN a, b'
        )
        try:
            resp = self._http_post("/api/query", {"nql": nql, "collection": self.collection})
            return str(uuid.uuid4())  # NQL doesn't return edge ID
        except Exception as e:
            print(f"[HTTP insert_edge error] {e}")
        return None

    # ── Helpers ─────────────────────────────────────────────────────

    def _parse_node(self, data: dict) -> NodeInfo:
        """Parse a node dict from API response."""
        content = data.get("content", {})
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                content = {"text": content}
        return NodeInfo(
            id=data.get("id", ""),
            depth=float(data.get("depth", 0)),
            energy=float(data.get("energy", 0)),
            node_type=data.get("node_type", data.get("type", "Semantic")),
            content=content,
            hausdorff_local=float(data.get("hausdorff_local", 0)),
            valence=float(data.get("valence", 0)),
            arousal=float(data.get("arousal", 0)),
            metadata=data.get("metadata", {}),
        )

    def sample_subgraph(self, limit: int = 50,
                        min_energy: float = 0.0) -> tuple[list[NodeInfo], list[EdgeInfo]]:
        """Sample a subgraph for hypothesis generation."""
        graph = self.get_graph(limit=limit)

        nodes = []
        for n in graph.get("nodes", []):
            node = self._parse_node(n)
            if node.energy >= min_energy:
                nodes.append(node)

        edges = []
        for e in graph.get("edges", []):
            edges.append(EdgeInfo(
                id=e.get("id", ""),
                from_id=e.get("from", e.get("from_id", "")),
                to_id=e.get("to", e.get("to_id", "")),
                edge_type=e.get("edge_type", e.get("type", "Association")),
                weight=float(e.get("weight", 1.0)),
                metadata=e.get("metadata", {}),
            ))

        return nodes, edges
