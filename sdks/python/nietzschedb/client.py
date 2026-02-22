"""NietzscheDB gRPC client — wraps all 65 RPCs from nietzsche.proto.

Usage::

    from nietzschedb import NietzscheClient

    client = NietzscheClient("localhost:50051")
    node_id = client.insert_node(coords=[0.1, 0.2, 0.3], content={"text": "hello"})
    results = client.knn_search(query=[0.1, 0.2, 0.3], k=10)
    client.close()

Or as a context manager::

    with NietzscheClient("localhost:50051") as client:
        client.query("MATCH (n) RETURN n LIMIT 10")
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Iterator, List, Optional, Tuple

import grpc

from nietzschedb.proto import nietzsche_pb2 as pb
from nietzschedb.proto import nietzsche_pb2_grpc as pb_grpc
from nietzschedb.types import (
    AlgorithmResult,
    BackupInfo,
    CausalEdgeInfo,
    CollectionInfo,
    DiffusionScale,
    Edge,
    KnnResult,
    Node,
    QueryResult,
    SensoryInfo,
    SleepReport,
    StatsResponse,
    SynthesisResult,
    ZaratustraReport,
)


def _to_json_bytes(content: Any) -> bytes:
    if content is None:
        return b"{}"
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")
    return json.dumps(content).encode("utf-8")


def _from_json_bytes(data: bytes) -> Dict[str, Any]:
    if not data:
        return {}
    return json.loads(data)


def _make_poincare(coords: List[float]) -> "pb.PoincareVector":
    return pb.PoincareVector(coords=coords, dim=len(coords))


def _node_from_pb(resp) -> Node:
    coords = list(resp.embedding.coords) if resp.embedding else []
    return Node(
        id=resp.id,
        coords=coords,
        content=_from_json_bytes(resp.content),
        node_type=resp.node_type,
        energy=resp.energy,
        depth=getattr(resp, "depth", 0.0),
        hausdorff_local=getattr(resp, "hausdorff_local", 0.0),
        created_at=resp.created_at,
        expires_at=getattr(resp, "expires_at", 0),
    )


def _convert_params(params: Optional[Dict[str, Any]]) -> Dict[str, "pb.QueryParamValue"]:
    if not params:
        return {}
    result = {}
    for k, v in params.items():
        pv = pb.QueryParamValue()
        if isinstance(v, str):
            pv.string_val = v
        elif isinstance(v, bool):
            # bool before int since bool is subclass of int
            pv.int_val = int(v)
        elif isinstance(v, int):
            pv.int_val = v
        elif isinstance(v, float):
            pv.float_val = v
        elif isinstance(v, (list, tuple)):
            pv.vec_val.CopyFrom(pb.VectorParam(coords=list(v)))
        else:
            pv.string_val = json.dumps(v)
        result[k] = pv
    return result


class NietzscheClient:
    """Synchronous gRPC client for NietzscheDB.

    Wraps all RPCs defined in nietzsche.proto with Pythonic types.
    """

    def __init__(
        self,
        addr: str = "localhost:50051",
        *,
        secure: bool = False,
        api_key: Optional[str] = None,
        max_message_size: int = 64 * 1024 * 1024,
    ):
        options = [
            ("grpc.max_send_message_length", max_message_size),
            ("grpc.max_receive_message_length", max_message_size),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
        ]

        if secure:
            creds = grpc.ssl_channel_credentials()
            self._channel = grpc.secure_channel(addr, creds, options=options)
        else:
            self._channel = grpc.insecure_channel(addr, options=options)

        self._stub = pb_grpc.NietzscheDBStub(self._channel)
        self._api_key = api_key

    def close(self):
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Collection Management ──────────────────────────────────────────────

    def create_collection(
        self, name: str, dimension: int = 0, metric: str = ""
    ) -> bool:
        resp = self._stub.CreateCollection(
            pb.CreateCollectionRequest(collection=name, dim=dimension, metric=metric)
        )
        return resp.created

    def drop_collection(self, name: str) -> bool:
        resp = self._stub.DropCollection(
            pb.DropCollectionRequest(collection=name)
        )
        return resp.status == "ok"

    def list_collections(self) -> List[CollectionInfo]:
        resp = self._stub.ListCollections(pb.Empty())
        return [
            CollectionInfo(
                name=c.collection,
                dim=c.dim,
                metric=c.metric,
                node_count=c.node_count,
                edge_count=c.edge_count,
            )
            for c in resp.collections
        ]

    # ── Node CRUD ──────────────────────────────────────────────────────────

    def insert_node(
        self,
        coords: List[float],
        content: Any = None,
        *,
        id: Optional[str] = None,
        node_type: str = "",
        energy: float = 1.0,
        collection: str = "",
        ttl_seconds: int = 0,
    ) -> str:
        node_id = id or str(uuid.uuid4())
        self._stub.InsertNode(pb.InsertNodeRequest(
            id=node_id,
            embedding=_make_poincare(coords),
            content=_to_json_bytes(content),
            node_type=node_type,
            energy=energy,
            collection=collection,
            expires_at=ttl_seconds,
        ))
        return node_id

    def get_node(self, node_id: str, *, collection: str = "") -> Optional[Node]:
        try:
            resp = self._stub.GetNode(pb.NodeIdRequest(
                id=node_id, collection=collection
            ))
            if not resp.found:
                return None
            return _node_from_pb(resp)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            raise

    def delete_node(self, node_id: str, *, collection: str = "") -> bool:
        resp = self._stub.DeleteNode(pb.NodeIdRequest(
            id=node_id, collection=collection
        ))
        return resp.status == "ok"

    def update_energy(
        self, node_id: str, energy: float, *, collection: str = ""
    ) -> bool:
        resp = self._stub.UpdateEnergy(pb.UpdateEnergyRequest(
            node_id=node_id, energy=energy, collection=collection
        ))
        return resp.status == "ok"

    # ── Edge CRUD ──────────────────────────────────────────────────────────

    def insert_edge(
        self,
        from_id: str,
        to_id: str,
        *,
        edge_type: str = "Association",
        weight: float = 1.0,
        collection: str = "",
    ) -> str:
        edge_id = str(uuid.uuid4())
        # Proto field `from` is a Python reserved word — use **kwargs
        self._stub.InsertEdge(pb.InsertEdgeRequest(
            id=edge_id,
            edge_type=edge_type,
            weight=weight,
            collection=collection,
            **{"from": from_id, "to": to_id},
        ))
        return edge_id

    def delete_edge(self, edge_id: str, *, collection: str = "") -> bool:
        resp = self._stub.DeleteEdge(pb.EdgeIdRequest(
            id=edge_id, collection=collection
        ))
        return resp.status == "ok"

    # ── Merge / Upsert ─────────────────────────────────────────────────────

    def merge_node(
        self,
        node_type: str,
        match_keys: Dict[str, Any],
        *,
        on_create: Optional[Dict[str, Any]] = None,
        on_match: Optional[Dict[str, Any]] = None,
        coords: Optional[List[float]] = None,
        energy: float = 1.0,
        collection: str = "",
    ) -> Tuple[str, bool]:
        """Merge (get-or-create) a node.

        Returns (node_id, created) tuple.
        """
        req = pb.MergeNodeRequest(
            collection=collection,
            node_type=node_type,
            match_keys=_to_json_bytes(match_keys),
            on_create_set=_to_json_bytes(on_create),
            on_match_set=_to_json_bytes(on_match),
            energy=energy,
        )
        if coords:
            req.embedding.CopyFrom(_make_poincare(coords))
        resp = self._stub.MergeNode(req)
        return resp.node_id, resp.created

    def merge_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: str = "Association",
        *,
        on_create: Optional[Dict[str, Any]] = None,
        on_match: Optional[Dict[str, Any]] = None,
        collection: str = "",
    ) -> Tuple[str, bool]:
        """Merge (upsert) an edge.

        Returns (edge_id, created) tuple.
        """
        resp = self._stub.MergeEdge(pb.MergeEdgeRequest(
            collection=collection,
            from_node_id=from_id,
            to_node_id=to_id,
            edge_type=edge_type,
            on_create_set=_to_json_bytes(on_create),
            on_match_set=_to_json_bytes(on_match),
        ))
        return resp.edge_id, resp.created

    def increment_edge_meta(
        self,
        edge_id: str,
        field: str,
        delta: float = 1.0,
        *,
        collection: str = "",
    ) -> float:
        """Atomically increment an edge metadata field. Returns new value."""
        resp = self._stub.IncrementEdgeMeta(pb.IncrementEdgeMetaRequest(
            collection=collection,
            edge_id=edge_id,
            field=field,
            delta=delta,
        ))
        return resp.new_value

    # ── Query (NQL) ────────────────────────────────────────────────────────

    def query(
        self,
        nql: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        collection: str = "",
    ) -> QueryResult:
        resp = self._stub.Query(pb.QueryRequest(
            nql=nql,
            params=_convert_params(params),
            collection=collection,
        ))
        nodes = [_node_from_pb(n) for n in resp.nodes]
        node_pairs = []
        for p in resp.node_pairs:
            pair_from = _node_from_pb(getattr(p, "from"))
            pair_to = _node_from_pb(p.to)
            node_pairs.append((pair_from, pair_to))
        scalar_rows = []
        for sr in resp.scalar_rows:
            row = {}
            for entry in sr.entries:
                val = None
                which = entry.WhichOneof("value")
                if entry.is_null:
                    val = None
                elif which == "float_val":
                    val = entry.float_val
                elif which == "int_val":
                    val = entry.int_val
                elif which == "string_val":
                    val = entry.string_val
                elif which == "bool_val":
                    val = entry.bool_val
                row[entry.column] = val
            scalar_rows.append(row)
        return QueryResult(
            nodes=nodes,
            node_pairs=node_pairs,
            path_ids=list(resp.path_ids),
            scalar_rows=scalar_rows,
            explain=resp.explain,
            error=resp.error,
        )

    # ── Search ─────────────────────────────────────────────────────────────

    def knn_search(
        self,
        query: List[float],
        k: int = 10,
        *,
        filters: Optional[List[Dict[str, Any]]] = None,
        collection: str = "",
    ) -> List[KnnResult]:
        pb_filters = []
        if filters:
            for f in filters:
                if "value" in f:
                    pb_filters.append(pb.KnnFilter(
                        match_filter=pb.KnnFilterMatch(
                            field=f["field"], value=str(f["value"])
                        )
                    ))
                elif "gte" in f or "lte" in f:
                    pb_filters.append(pb.KnnFilter(
                        range_filter=pb.KnnFilterRange(
                            field=f["field"],
                            gte=f.get("gte"),
                            lte=f.get("lte"),
                        )
                    ))
        resp = self._stub.KnnSearch(pb.KnnRequest(
            query_coords=query,
            k=k,
            collection=collection,
            filters=pb_filters,
        ))
        return [
            KnnResult(id=r.id, distance=r.distance)
            for r in resp.results
        ]

    def full_text_search(
        self,
        query_text: str,
        limit: int = 10,
        *,
        collection: str = "",
    ) -> List[KnnResult]:
        resp = self._stub.FullTextSearch(pb.FullTextSearchRequest(
            query=query_text,
            limit=limit,
            collection=collection,
        ))
        return [
            KnnResult(id=r.node_id, distance=r.score)
            for r in resp.results
        ]

    def hybrid_search(
        self,
        query_coords: List[float],
        text_query: str,
        k: int = 10,
        *,
        text_weight: float = 1.0,
        vector_weight: float = 1.0,
        collection: str = "",
    ) -> List[KnnResult]:
        resp = self._stub.HybridSearch(pb.HybridSearchRequest(
            text_query=text_query,
            query_coords=query_coords,
            k=k,
            text_weight=text_weight,
            vector_weight=vector_weight,
            collection=collection,
        ))
        return [
            KnnResult(id=r.id, distance=r.distance)
            for r in resp.results
        ]

    # ── Batch Operations ───────────────────────────────────────────────────

    def batch_insert_nodes(
        self,
        nodes: List[Dict[str, Any]],
        *,
        collection: str = "",
    ) -> Tuple[int, List[str]]:
        """Batch insert nodes.

        Each dict should have: coords (list), content (any), and optionally
        id, node_type, energy, expires_at.

        Returns (inserted_count, node_ids).
        """
        pb_nodes = []
        for n in nodes:
            node_id = n.get("id") or str(uuid.uuid4())
            pb_nodes.append(pb.InsertNodeRequest(
                id=node_id,
                embedding=_make_poincare(n.get("coords", [])),
                content=_to_json_bytes(n.get("content")),
                node_type=n.get("node_type", ""),
                energy=n.get("energy", 1.0),
                collection=collection,
                expires_at=n.get("expires_at", 0),
            ))
        resp = self._stub.BatchInsertNodes(pb.BatchInsertNodesRequest(
            nodes=pb_nodes, collection=collection
        ))
        return resp.inserted, list(resp.node_ids)

    def batch_insert_edges(
        self,
        edges: List[Dict[str, Any]],
        *,
        collection: str = "",
    ) -> Tuple[int, List[str]]:
        """Batch insert edges.

        Each dict should have: from_id, to_id, and optionally edge_type, weight.

        Returns (inserted_count, edge_ids).
        """
        pb_edges = []
        for e in edges:
            edge_id = e.get("id") or str(uuid.uuid4())
            pb_edges.append(pb.InsertEdgeRequest(
                id=edge_id,
                edge_type=e.get("edge_type", "Association"),
                weight=e.get("weight", 1.0),
                collection=collection,
                **{"from": e["from_id"], "to": e["to_id"]},
            ))
        resp = self._stub.BatchInsertEdges(pb.BatchInsertEdgesRequest(
            edges=pb_edges, collection=collection
        ))
        return resp.inserted, list(resp.edge_ids)

    # ── Graph Traversal ────────────────────────────────────────────────────

    def bfs(
        self,
        start_id: str,
        *,
        max_depth: int = 10,
        max_nodes: int = 1000,
        energy_min: float = 0.0,
        collection: str = "",
    ) -> List[str]:
        """BFS traversal. Returns visited node IDs in visit order."""
        resp = self._stub.Bfs(pb.TraversalRequest(
            start_node_id=start_id,
            max_depth=max_depth,
            max_nodes=max_nodes,
            energy_min=energy_min,
            collection=collection,
        ))
        return list(resp.visited_ids)

    def dijkstra(
        self,
        start_id: str,
        *,
        max_depth: int = 0,
        max_cost: float = 0.0,
        max_nodes: int = 0,
        energy_min: float = 0.0,
        collection: str = "",
    ) -> Tuple[List[str], List[float]]:
        """Dijkstra traversal. Returns (visited_ids, costs)."""
        resp = self._stub.Dijkstra(pb.TraversalRequest(
            start_node_id=start_id,
            max_depth=max_depth,
            max_cost=max_cost,
            max_nodes=max_nodes,
            energy_min=energy_min,
            collection=collection,
        ))
        return list(resp.visited_ids), list(resp.costs)

    def diffuse(
        self,
        source_ids: List[str],
        t_values: List[float],
        *,
        k_chebyshev: int = 0,
        collection: str = "",
    ) -> List[DiffusionScale]:
        """Run heat-kernel diffusion from source nodes.

        Returns a list of DiffusionScale (one per t_value).
        """
        resp = self._stub.Diffuse(pb.DiffusionRequest(
            source_ids=source_ids,
            t_values=t_values,
            k_chebyshev=k_chebyshev,
            collection=collection,
        ))
        results = []
        for scale in resp.scales:
            scores = {}
            for nid, score in zip(scale.node_ids, scale.scores):
                scores[nid] = score
            results.append(DiffusionScale(t=scale.t, scores=scores))
        return results

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def trigger_sleep(
        self,
        *,
        noise: float = 0.02,
        adam_steps: int = 10,
        adam_lr: float = 5e-3,
        hausdorff_threshold: float = 0.15,
        rng_seed: int = 0,
        collection: str = "",
    ) -> SleepReport:
        resp = self._stub.TriggerSleep(pb.SleepRequest(
            noise=noise,
            adam_steps=adam_steps,
            adam_lr=adam_lr,
            hausdorff_threshold=hausdorff_threshold,
            rng_seed=rng_seed,
            collection=collection,
        ))
        return SleepReport(
            hausdorff_before=resp.hausdorff_before,
            hausdorff_after=resp.hausdorff_after,
            hausdorff_delta=resp.hausdorff_delta,
            semantic_drift_avg=resp.semantic_drift_avg,
            semantic_drift_max=resp.semantic_drift_max,
            committed=resp.committed,
            nodes_perturbed=resp.nodes_perturbed,
            snapshot_nodes=resp.snapshot_nodes,
        )

    def invoke_zaratustra(
        self,
        *,
        alpha: float = 0.0,
        decay: float = 0.0,
        cycles: int = 0,
        collection: str = "",
    ) -> ZaratustraReport:
        resp = self._stub.InvokeZaratustra(pb.ZaratustraRequest(
            collection=collection,
            alpha=alpha,
            decay=decay,
            cycles=cycles,
        ))
        return ZaratustraReport(
            nodes_updated=resp.nodes_updated,
            mean_energy_before=resp.mean_energy_before,
            mean_energy_after=resp.mean_energy_after,
            total_energy_delta=resp.total_energy_delta,
            echoes_created=resp.echoes_created,
            echoes_evicted=resp.echoes_evicted,
            total_echoes=resp.total_echoes,
            elite_count=resp.elite_count,
            elite_threshold=resp.elite_threshold,
            mean_elite_energy=resp.mean_elite_energy,
            mean_base_energy=resp.mean_base_energy,
            elite_node_ids=list(resp.elite_node_ids),
            duration_ms=resp.duration_ms,
            cycles_run=resp.cycles_run,
        )

    # ── Graph Algorithms ───────────────────────────────────────────────────

    def run_pagerank(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 20,
        convergence_threshold: float = 0.0,
        collection: str = "",
    ) -> AlgorithmResult:
        resp = self._stub.RunPageRank(pb.PageRankRequest(
            collection=collection,
            damping_factor=damping,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        ))
        return AlgorithmResult(
            scores={s.node_id: s.score for s in resp.scores},
            duration_ms=resp.duration_ms,
            iterations=resp.iterations,
        )

    def run_louvain(
        self, *, max_iterations: int = 0, resolution: float = 0.0, collection: str = ""
    ) -> AlgorithmResult:
        resp = self._stub.RunLouvain(pb.LouvainRequest(
            collection=collection,
            max_iterations=max_iterations,
            resolution=resolution,
        ))
        return AlgorithmResult(
            communities={a.node_id: a.community_id for a in resp.assignments},
            duration_ms=resp.duration_ms,
            iterations=resp.iterations,
        )

    def run_label_prop(
        self, *, max_iterations: int = 10, collection: str = ""
    ) -> AlgorithmResult:
        resp = self._stub.RunLabelProp(pb.LabelPropRequest(
            collection=collection,
            max_iterations=max_iterations,
        ))
        return AlgorithmResult(
            communities={a.node_id: a.community_id for a in resp.assignments},
            duration_ms=resp.duration_ms,
            iterations=resp.iterations,
        )

    def run_betweenness(
        self, *, sample_size: int = 0, collection: str = ""
    ) -> AlgorithmResult:
        resp = self._stub.RunBetweenness(pb.BetweennessRequest(
            collection=collection, sample_size=sample_size,
        ))
        return AlgorithmResult(
            scores={s.node_id: s.score for s in resp.scores},
            duration_ms=resp.duration_ms,
        )

    def run_closeness(self, *, collection: str = "") -> AlgorithmResult:
        resp = self._stub.RunCloseness(pb.ClosenessRequest(collection=collection))
        return AlgorithmResult(
            scores={s.node_id: s.score for s in resp.scores},
            duration_ms=resp.duration_ms,
        )

    def run_degree_centrality(
        self, *, direction: str = "", collection: str = ""
    ) -> AlgorithmResult:
        resp = self._stub.RunDegreeCentrality(pb.DegreeCentralityRequest(
            collection=collection, direction=direction,
        ))
        return AlgorithmResult(
            scores={s.node_id: s.score for s in resp.scores},
            duration_ms=resp.duration_ms,
        )

    def run_wcc(self, *, collection: str = "") -> AlgorithmResult:
        resp = self._stub.RunWCC(pb.WccRequest(collection=collection))
        return AlgorithmResult(
            communities={a.node_id: a.community_id for a in resp.assignments},
            duration_ms=resp.duration_ms,
        )

    def run_scc(self, *, collection: str = "") -> AlgorithmResult:
        resp = self._stub.RunSCC(pb.SccRequest(collection=collection))
        return AlgorithmResult(
            communities={a.node_id: a.community_id for a in resp.assignments},
            duration_ms=resp.duration_ms,
        )

    def run_astar(
        self, start_id: str, end_id: str, *, collection: str = ""
    ) -> Tuple[List[str], float]:
        """A* pathfinding. Returns (path, cost)."""
        resp = self._stub.RunAStar(pb.AStarRequest(
            collection=collection,
            start_node_id=start_id,
            goal_node_id=end_id,
        ))
        return list(resp.path), resp.cost

    def run_triangle_count(self, *, collection: str = "") -> int:
        resp = self._stub.RunTriangleCount(pb.TriangleCountRequest(
            collection=collection
        ))
        return resp.count

    def run_jaccard(
        self, *, top_k: int = 0, threshold: float = 0.0, collection: str = ""
    ) -> List[Tuple[str, str, float]]:
        """Jaccard similarity. Returns list of (node_a, node_b, score)."""
        resp = self._stub.RunJaccardSimilarity(pb.JaccardRequest(
            collection=collection, top_k=top_k, threshold=threshold,
        ))
        return [(p.node_a, p.node_b, p.score) for p in resp.pairs]

    # ── Sensory Layer (Phase 11) ───────────────────────────────────────────

    def insert_sensory(
        self,
        node_id: str,
        modality: str,
        latent: List[float],
        *,
        original_shape: Optional[Dict[str, Any]] = None,
        original_bytes: int = 0,
        encoder_version: int = 0,
        modality_meta: Optional[Dict[str, Any]] = None,
        collection: str = "",
    ) -> bool:
        resp = self._stub.InsertSensory(pb.InsertSensoryRequest(
            node_id=node_id,
            modality=modality,
            latent=latent,
            original_shape=_to_json_bytes(original_shape),
            original_bytes=original_bytes,
            encoder_version=encoder_version,
            modality_meta=_to_json_bytes(modality_meta),
            collection=collection,
        ))
        return resp.status == "ok"

    def get_sensory(
        self, node_id: str, *, collection: str = ""
    ) -> Optional[SensoryInfo]:
        try:
            resp = self._stub.GetSensory(pb.NodeIdRequest(
                id=node_id, collection=collection
            ))
            if not resp.found:
                return None
            return SensoryInfo(
                found=resp.found,
                node_id=resp.node_id,
                modality=resp.modality,
                dim=resp.dim,
                quant_level=resp.quant_level,
                reconstruction_quality=resp.reconstruction_quality,
                compression_ratio=resp.compression_ratio,
                encoder_version=resp.encoder_version,
                byte_size=resp.byte_size,
            )
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            raise

    def reconstruct(
        self,
        node_id: str,
        quality: str = "best_available",
        *,
        collection: str = "",
    ) -> Optional[List[float]]:
        """Reconstruct sensory data. Returns latent vector or None."""
        resp = self._stub.Reconstruct(pb.ReconstructRequest(
            node_id=node_id,
            quality=quality,
            collection=collection,
        ))
        if not resp.found:
            return None
        return list(resp.latent)

    def degrade_sensory(
        self, node_id: str, *, collection: str = ""
    ) -> bool:
        resp = self._stub.DegradeSensory(pb.NodeIdRequest(
            id=node_id, collection=collection
        ))
        return resp.status == "ok"

    # ── Multi-Manifold Operations (Phase M) ────────────────────────────────

    def synthesis(
        self, node_a: str, node_b: str, *, collection: str = ""
    ) -> SynthesisResult:
        resp = self._stub.Synthesis(pb.SynthesisRequest(
            node_id_a=node_a, node_id_b=node_b, collection=collection
        ))
        return SynthesisResult(
            coords=list(resp.synthesis_coords),
            nearest_node_id=resp.nearest_node_id,
            nearest_distance=resp.nearest_distance,
        )

    def synthesis_multi(
        self, node_ids: List[str], *, collection: str = ""
    ) -> SynthesisResult:
        resp = self._stub.SynthesisMulti(pb.SynthesisMultiRequest(
            node_ids=node_ids, collection=collection
        ))
        return SynthesisResult(
            coords=list(resp.synthesis_coords),
            nearest_node_id=resp.nearest_node_id,
            nearest_distance=resp.nearest_distance,
        )

    def causal_neighbors(
        self,
        node_id: str,
        *,
        direction: str = "both",
        collection: str = "",
    ) -> List[CausalEdgeInfo]:
        resp = self._stub.CausalNeighbors(pb.CausalNeighborsRequest(
            node_id=node_id,
            direction=direction,
            collection=collection,
        ))
        return [
            CausalEdgeInfo(
                edge_id=e.edge_id,
                from_node_id=e.from_node_id,
                to_node_id=e.to_node_id,
                minkowski_interval=e.minkowski_interval,
                causal_type=e.causal_type,
                edge_type=e.edge_type,
            )
            for e in resp.edges
        ]

    def causal_chain(
        self,
        node_id: str,
        *,
        max_depth: int = 0,
        direction: str = "",
        collection: str = "",
    ) -> Tuple[List[str], List[CausalEdgeInfo]]:
        """Causal chain traversal. Returns (chain_ids, edges)."""
        resp = self._stub.CausalChain(pb.CausalChainRequest(
            node_id=node_id,
            max_depth=max_depth,
            direction=direction,
            collection=collection,
        ))
        edges = [
            CausalEdgeInfo(
                edge_id=e.edge_id,
                from_node_id=e.from_node_id,
                to_node_id=e.to_node_id,
                minkowski_interval=e.minkowski_interval,
                causal_type=e.causal_type,
                edge_type=e.edge_type,
            )
            for e in resp.edges
        ]
        return list(resp.chain_ids), edges

    def klein_path(
        self,
        start_id: str,
        end_id: str,
        *,
        collection: str = "",
    ) -> Tuple[List[str], float]:
        """Klein geodesic pathfinding. Returns (path, cost)."""
        resp = self._stub.KleinPath(pb.KleinPathRequest(
            start_node_id=start_id, goal_node_id=end_id, collection=collection
        ))
        return list(resp.path), resp.cost

    def is_on_shortest_path(
        self,
        node_a: str,
        node_b: str,
        candidate: str,
        *,
        collection: str = "",
    ) -> Tuple[bool, float]:
        """Check if candidate lies on Klein geodesic between A and B.

        Returns (on_path, distance).
        """
        resp = self._stub.IsOnShortestPath(pb.ShortestPathCheckRequest(
            node_id_a=node_a,
            node_id_b=node_b,
            node_id_c=candidate,
            collection=collection,
        ))
        return resp.on_path, resp.distance

    # ── Cache / TTL (Redis replacement) ────────────────────────────────────

    def cache_set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: int = 0,
        collection: str = "",
    ) -> bool:
        resp = self._stub.CacheSet(pb.CacheSetRequest(
            collection=collection,
            key=key,
            value=_to_json_bytes(value),
            ttl_secs=ttl_seconds,
        ))
        return resp.status == "ok"

    def cache_get(self, key: str, *, collection: str = "") -> Optional[Any]:
        try:
            resp = self._stub.CacheGet(pb.CacheGetRequest(
                collection=collection, key=key
            ))
            if not resp.found:
                return None
            return _from_json_bytes(resp.value)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            raise

    def cache_del(self, key: str, *, collection: str = "") -> bool:
        resp = self._stub.CacheDel(pb.CacheDelRequest(
            collection=collection, key=key
        ))
        return resp.status == "ok"

    def reap_expired(self, *, collection: str = "") -> int:
        resp = self._stub.ReapExpired(pb.ReapExpiredRequest(
            collection=collection
        ))
        return resp.reaped_count

    # ── List Store ─────────────────────────────────────────────────────────

    def list_rpush(
        self,
        node_id: str,
        list_name: str,
        value: bytes,
        *,
        collection: str = "",
    ) -> int:
        """Append a value to a list. Returns new list length."""
        resp = self._stub.ListRPush(pb.ListPushRequest(
            node_id=node_id,
            list_name=list_name,
            value=value,
            collection=collection,
        ))
        return resp.new_length

    def list_lrange(
        self,
        node_id: str,
        list_name: str,
        start: int = 0,
        stop: int = -1,
        *,
        collection: str = "",
    ) -> List[bytes]:
        resp = self._stub.ListLRange(pb.ListRangeRequest(
            node_id=node_id,
            list_name=list_name,
            start=start,
            stop=stop,
            collection=collection,
        ))
        return list(resp.values)

    def list_len(
        self, node_id: str, list_name: str, *, collection: str = ""
    ) -> int:
        resp = self._stub.ListLen(pb.ListLenRequest(
            node_id=node_id, list_name=list_name, collection=collection
        ))
        return resp.length

    # ── Backup / Restore ───────────────────────────────────────────────────

    def create_backup(self, label: str = "") -> BackupInfo:
        resp = self._stub.CreateBackup(pb.CreateBackupRequest(
            label=label
        ))
        return BackupInfo(
            label=resp.label,
            path=resp.path,
            created_at=resp.created_at,
            size_bytes=resp.size_bytes,
        )

    def list_backups(self) -> List[BackupInfo]:
        resp = self._stub.ListBackups(pb.Empty())
        return [
            BackupInfo(
                label=b.label,
                path=b.path,
                created_at=b.created_at,
                size_bytes=b.size_bytes,
            )
            for b in resp.backups
        ]

    def restore_backup(self, backup_path: str, target_path: str = "") -> bool:
        resp = self._stub.RestoreBackup(pb.RestoreBackupRequest(
            backup_path=backup_path, target_path=target_path
        ))
        return resp.status == "ok"

    # ── CDC (Change Data Capture) ──────────────────────────────────────────

    def subscribe_cdc(
        self, *, from_lsn: int = 0, collection: str = ""
    ) -> Iterator[Dict[str, Any]]:
        stream = self._stub.SubscribeCDC(pb.CdcRequest(
            from_lsn=from_lsn, collection=collection
        ))
        for event in stream:
            yield {
                "lsn": event.lsn,
                "event_type": event.event_type,
                "timestamp_ms": event.timestamp_ms,
                "entity_id": event.entity_id,
                "collection": event.collection,
                "batch_count": event.batch_count,
            }

    # ── Schema Validation ──────────────────────────────────────────────────

    def set_schema(
        self,
        node_type: str,
        required_fields: List[str],
        field_types: Optional[Dict[str, str]] = None,
        *,
        collection: str = "",
    ) -> bool:
        pb_field_types = []
        if field_types:
            for fname, ftype in field_types.items():
                pb_field_types.append(pb.SchemaFieldType(
                    field_name=fname, field_type=ftype
                ))
        resp = self._stub.SetSchema(pb.SetSchemaRequest(
            node_type=node_type,
            required_fields=required_fields,
            field_types=pb_field_types,
            collection=collection,
        ))
        return resp.status == "ok"

    def get_schema(
        self, node_type: str, *, collection: str = ""
    ) -> Optional[Dict[str, Any]]:
        resp = self._stub.GetSchema(pb.GetSchemaRequest(
            node_type=node_type, collection=collection
        ))
        if not resp.found:
            return None
        return {
            "node_type": resp.node_type,
            "required_fields": list(resp.required_fields),
            "field_types": {ft.field_name: ft.field_type for ft in resp.field_types},
        }

    def list_schemas(self, *, collection: str = "") -> List[Dict[str, Any]]:
        resp = self._stub.ListSchemas(pb.Empty())
        schemas = []
        for s in resp.schemas:
            schemas.append({
                "node_type": s.node_type,
                "required_fields": list(s.required_fields),
                "field_types": {ft.field_name: ft.field_type for ft in s.field_types},
            })
        return schemas

    # ── Secondary Indexes ──────────────────────────────────────────────────

    def create_index(
        self, field: str, *, collection: str = ""
    ) -> bool:
        resp = self._stub.CreateIndex(pb.CreateIndexRequest(
            collection=collection, field=field
        ))
        return resp.status == "ok"

    def drop_index(self, field: str, *, collection: str = "") -> bool:
        resp = self._stub.DropIndex(pb.DropIndexRequest(
            collection=collection, field=field
        ))
        return resp.status == "ok"

    def list_indexes(self, *, collection: str = "") -> List[str]:
        resp = self._stub.ListIndexes(pb.ListIndexesRequest(
            collection=collection
        ))
        return list(resp.fields)

    # ── SQL Layer (Swartz/GlueSQL) ─────────────────────────────────────────

    def sql_query(self, sql: str, *, collection: str = "") -> List[Dict[str, Any]]:
        """Execute a SELECT query. Returns list of row dicts."""
        resp = self._stub.SqlQuery(pb.SqlRequest(
            sql=sql, collection=collection
        ))
        columns = [c.name for c in resp.columns]
        rows = []
        for row in resp.rows:
            row_dict = {}
            for i, val_bytes in enumerate(row.values):
                if i < len(columns):
                    row_dict[columns[i]] = json.loads(val_bytes) if val_bytes else None
            rows.append(row_dict)
        return rows

    def sql_exec(self, sql: str, *, collection: str = "") -> int:
        """Execute a DDL/DML statement. Returns affected rows."""
        resp = self._stub.SqlExec(pb.SqlRequest(
            sql=sql, collection=collection
        ))
        return resp.affected_rows

    # ── Admin ──────────────────────────────────────────────────────────────

    def get_stats(self) -> StatsResponse:
        resp = self._stub.GetStats(pb.Empty())
        return StatsResponse(
            node_count=resp.node_count,
            edge_count=resp.edge_count,
            version=resp.version,
            sensory_count=resp.sensory_count,
        )

    def health_check(self) -> bool:
        try:
            resp = self._stub.HealthCheck(pb.Empty())
            return resp.status == "ok"
        except grpc.RpcError:
            return False
