"""NietzscheDB Python gRPC client SDK.

Generated from ``proto/nietzsche.proto``.  Requires the ``grpcio`` and
``grpcio-tools`` packages.

Installation
------------
.. code-block:: bash

    pip install grpcio grpcio-tools

Regenerate stubs
----------------
.. code-block:: bash

    python -m grpc_tools.protoc \\
        -I../../crates/nietzsche-api/proto \\
        --python_out=. \\
        --grpc_python_out=. \\
        nietzsche.proto

Example
-------
.. code-block:: python

    from nietzsche_db import NietzscheClient

    client = NietzscheClient("localhost:50051")

    # Insert a node
    node = client.insert_node(coords=[0.1, 0.2, 0.3], content={"text": "Zarathustra"})
    print(f"inserted: {node.id}")

    # NQL query
    result = client.query("MATCH (n) WHERE n.energy > 0.5 RETURN n LIMIT 10")
    for n in result.nodes:
        print(n.id, n.energy)

    # Reconsolidation sleep
    report = client.trigger_sleep()
    print(f"committed={report.committed}, delta_H={report.hausdorff_delta:.4f}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import grpc

# ---------------------------------------------------------------------------
# NOTE: The stubs below reference generated classes from nietzsche_pb2 and
# nietzsche_pb2_grpc.  Run the grpc_tools.protoc command above to generate
# those files alongside this module.
# ---------------------------------------------------------------------------
try:
    import nietzsche_pb2        as pb
    import nietzsche_pb2_grpc   as pb_grpc
except ImportError:  # pragma: no cover
    pb        = None  # type: ignore[assignment]
    pb_grpc   = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InsertNodeParams:
    """Parameters for :meth:`NietzscheClient.insert_node`."""
    coords:    List[float]      = field(default_factory=list)
    content:   dict             = field(default_factory=dict)
    node_type: str              = "Semantic"
    energy:    float            = 1.0
    node_id:   Optional[str]   = None   # auto-generated if None


@dataclass
class InsertEdgeParams:
    """Parameters for :meth:`NietzscheClient.insert_edge`."""
    from_id:   str
    to_id:     str
    edge_type: str   = "Association"
    weight:    float = 1.0
    edge_id:   Optional[str] = None     # auto-generated if None


@dataclass
class SleepParams:
    """Parameters for :meth:`NietzscheClient.trigger_sleep`."""
    noise:               float = 0.02
    adam_steps:          int   = 10
    adam_lr:             float = 5e-3
    hausdorff_threshold: float = 0.15
    rng_seed:            int   = 0      # 0 = non-deterministic


@dataclass
class TraversalParams:
    """Parameters for BFS / Dijkstra traversal."""
    max_depth:  int   = 10
    max_nodes:  int   = 1_000
    max_cost:   float = 0.0
    energy_min: float = 0.0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class NietzscheClient:
    """Synchronous gRPC client for NietzscheDB.

    For async usage, wrap calls with ``asyncio.to_thread`` or use the native
    Rust SDK (``nietzsche-sdk`` crate).

    Parameters
    ----------
    address:
        Server address in the form ``"host:port"`` (e.g. ``"localhost:50051"``).
    secure:
        Use TLS/SSL if ``True`` (default ``False`` for local dev).
    """

    def __init__(self, address: str, *, secure: bool = False) -> None:
        if secure:
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.secure_channel(address, credentials)
        else:
            self._channel = grpc.insecure_channel(address)

        if pb_grpc is not None:
            self._stub = pb_grpc.NietzscheDBStub(self._channel)

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        self._channel.close()

    def __enter__(self) -> "NietzscheClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Node CRUD ──────────────────────────────────────────────────────────

    def insert_node(self, params: InsertNodeParams) -> "pb.NodeResponse":
        """Insert a node into the graph."""
        req = pb.InsertNodeRequest(
            id        = params.node_id or "",
            embedding = pb.PoincareVector(coords=params.coords, dim=len(params.coords)),
            content   = json.dumps(params.content).encode(),
            node_type = params.node_type,
            energy    = params.energy,
        )
        return self._stub.InsertNode(req)

    def get_node(self, node_id: str) -> "pb.NodeResponse":
        """Retrieve a node by UUID string."""
        return self._stub.GetNode(pb.NodeIdRequest(id=node_id))

    def delete_node(self, node_id: str) -> "pb.StatusResponse":
        """Hard-delete a node and all its incident edges."""
        return self._stub.DeleteNode(pb.NodeIdRequest(id=node_id))

    def update_energy(self, node_id: str, energy: float) -> "pb.StatusResponse":
        """Update a node's energy level."""
        return self._stub.UpdateEnergy(
            pb.UpdateEnergyRequest(node_id=node_id, energy=energy)
        )

    # ── Edge CRUD ──────────────────────────────────────────────────────────

    def insert_edge(self, params: InsertEdgeParams) -> "pb.EdgeResponse":
        """Insert a directed edge."""
        req = pb.InsertEdgeRequest(
            id        = params.edge_id or "",
            **{"from": params.from_id},   # 'from' is a Python keyword
            to        = params.to_id,
            edge_type = params.edge_type,
            weight    = params.weight,
        )
        return self._stub.InsertEdge(req)

    def delete_edge(self, edge_id: str) -> "pb.StatusResponse":
        """Delete an edge by UUID."""
        return self._stub.DeleteEdge(pb.EdgeIdRequest(id=edge_id))

    # ── NQL query ──────────────────────────────────────────────────────────

    def query(self, nql: str) -> "pb.QueryResponse":
        """Execute a Nietzsche Query Language statement.

        Examples
        --------
        .. code-block:: python

            result = client.query(
                "MATCH (n:Memory) WHERE n.energy > 0.3 RETURN n LIMIT 20"
            )
            result = client.query(
                "MATCH (a)-[:Association]->(b) WHERE a.energy > 0.5 RETURN b"
            )
            result = client.query(
                "DIFFUSE FROM $node WITH t = [0.1, 1.0, 10.0] MAX_HOPS 5 RETURN path"
            )
        """
        return self._stub.Query(pb.QueryRequest(nql=nql))

    # ── KNN search ─────────────────────────────────────────────────────────

    def knn_search(
        self,
        coords: List[float],
        k: int = 10,
    ) -> "pb.KnnResponse":
        """Hyperbolic k-nearest-neighbour search."""
        return self._stub.KnnSearch(pb.KnnRequest(query_coords=coords, k=k))

    # ── Traversal ──────────────────────────────────────────────────────────

    def bfs(
        self,
        start_id: str,
        params: TraversalParams = TraversalParams(),
    ) -> "pb.TraversalResponse":
        """Breadth-first traversal from ``start_id``."""
        req = pb.TraversalRequest(
            start_node_id = start_id,
            max_depth     = params.max_depth,
            max_nodes     = params.max_nodes,
            energy_min    = params.energy_min,
        )
        return self._stub.Bfs(req)

    def dijkstra(
        self,
        start_id: str,
        params: TraversalParams = TraversalParams(),
    ) -> "pb.TraversalResponse":
        """Dijkstra shortest-path from ``start_id``."""
        req = pb.TraversalRequest(
            start_node_id = start_id,
            max_nodes     = params.max_nodes,
            max_cost      = params.max_cost,
            energy_min    = params.energy_min,
        )
        return self._stub.Dijkstra(req)

    # ── Diffusion ──────────────────────────────────────────────────────────

    def diffuse(
        self,
        source_ids:   List[str],
        t_values:     Optional[List[float]] = None,
        k_chebyshev:  int = 10,
    ) -> "pb.DiffusionResponse":
        """Run Chebyshev heat-kernel diffusion.

        Parameters
        ----------
        source_ids:
            Seed node UUIDs.
        t_values:
            Diffusion time scales.  Defaults to ``[0.1, 1.0, 10.0]``.
        k_chebyshev:
            Chebyshev polynomial order.
        """
        req = pb.DiffusionRequest(
            source_ids  = source_ids,
            t_values    = t_values or [0.1, 1.0, 10.0],
            k_chebyshev = k_chebyshev,
        )
        return self._stub.Diffuse(req)

    # ── Sleep cycle ────────────────────────────────────────────────────────

    def trigger_sleep(
        self,
        params: SleepParams = SleepParams(),
    ) -> "pb.SleepResponse":
        """Trigger a Riemannian reconsolidation sleep cycle on the server."""
        req = pb.SleepRequest(
            noise               = params.noise,
            adam_steps          = params.adam_steps,
            adam_lr             = params.adam_lr,
            hausdorff_threshold = params.hausdorff_threshold,
            rng_seed            = params.rng_seed,
        )
        return self._stub.TriggerSleep(req)

    # ── Admin ──────────────────────────────────────────────────────────────

    def get_stats(self) -> "pb.StatsResponse":
        """Retrieve node/edge counts and server version."""
        return self._stub.GetStats(pb.Empty())

    def health_check(self) -> "pb.StatusResponse":
        """Check server liveness."""
        return self._stub.HealthCheck(pb.Empty())
