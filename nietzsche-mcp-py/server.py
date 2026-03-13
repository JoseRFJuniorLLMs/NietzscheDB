#!/usr/bin/env python3
"""NietzscheDB MCP Server — exposes NietzscheDB as tools for Claude Code/Desktop.

Runs as a stdio MCP server. Configure in .mcp.json:
{
    "mcpServers": {
        "nietzschedb": {
            "command": "python",
            "args": ["d:/DEV/NietzscheDB/nietzsche-mcp-py/server.py"],
            "env": {
                "NIETZSCHE_HOST": "136.111.0.47:443"
            }
        }
    }
}
"""

from __future__ import annotations

import json
import os
import sys

# Add SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdks", "python"))

from mcp.server.fastmcp import FastMCP

from nietzschedb import NietzscheClient

# ── Config ────────────────────────────────────────────────────────────────

NIETZSCHE_HOST = os.getenv("NIETZSCHE_HOST", "136.111.0.47:443")
NIETZSCHE_INSECURE = os.getenv("NIETZSCHE_INSECURE", "false").lower() == "true"
NIETZSCHE_CERT_PATH = os.getenv(
    "NIETZSCHE_CERT_PATH",
    os.path.expanduser("~/AppData/Local/Temp/eva-cert.pem"),
)

# ── Client singleton ─────────────────────────────────────────────────────

_client: NietzscheClient | None = None


def get_client() -> NietzscheClient:
    global _client
    if _client is None:
        if NIETZSCHE_INSECURE:
            _client = NietzscheClient(NIETZSCHE_HOST)
        else:
            _client = _make_secure_client(NIETZSCHE_HOST, NIETZSCHE_CERT_PATH)
    return _client


def _make_secure_client(host: str, cert_path: str) -> NietzscheClient:
    """Create client with self-signed cert for gRPC over TLS."""
    import grpc as _grpc
    from nietzschedb.proto import nietzsche_pb2_grpc as _pb_grpc

    with open(cert_path, "rb") as f:
        cert = f.read()
    creds = _grpc.ssl_channel_credentials(root_certificates=cert)
    options = [
        ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 10000),
    ]
    channel = _grpc.secure_channel(host, creds, options=options)
    client = NietzscheClient.__new__(NietzscheClient)
    client._channel = channel
    client._stub = _pb_grpc.NietzscheDBStub(channel)
    client._api_key = None
    return client


# ── MCP Server ────────────────────────────────────────────────────────────

mcp = FastMCP(
    "NietzscheDB",
    instructions="Hyperbolic knowledge graph database. Query, search, and manage nodes/edges in Poincare ball space. Use list_collections to see available data, query_nql for NQL queries, run_pagerank/run_louvain for graph analysis.",
)


# ── Collection tools ──────────────────────────────────────────────────────


@mcp.tool()
def list_collections() -> str:
    """List all NietzscheDB collections with node/edge counts, dimension, and metric."""
    cols = get_client().list_collections()
    result = [
        {
            "name": c.name,
            "dim": c.dim,
            "metric": c.metric,
            "node_count": c.node_count,
            "edge_count": c.edge_count,
        }
        for c in cols
    ]
    return json.dumps(result, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get global NietzscheDB statistics (total nodes, edges, version)."""
    s = get_client().get_stats()
    return json.dumps(
        {
            "node_count": s.node_count,
            "edge_count": s.edge_count,
            "version": s.version,
            "sensory_count": s.sensory_count,
        },
        indent=2,
    )


@mcp.tool()
def health_check() -> str:
    """Check if NietzscheDB server is healthy."""
    ok = get_client().health_check()
    return json.dumps({"healthy": ok})


# ── Query & Search ────────────────────────────────────────────────────────


@mcp.tool()
def query_nql(nql: str, collection: str) -> str:
    """Execute an NQL (Nietzsche Query Language) query.

    Examples:
    - MATCH (n:Semantic) RETURN n LIMIT 20
    - MATCH (n)-[e]->(m) WHERE n.energy > 0.5 RETURN n,m LIMIT 50
    - MATCH (n:Concept) RETURN COUNT(n)

    Node types: Episodic, Semantic, Concept, DreamSnapshot
    """
    r = get_client().query(nql, collection=collection)
    nodes = [_node_dict(n) for n in r.nodes]
    return json.dumps(
        {"nodes": nodes, "scalar_rows": r.scalar_rows, "error": r.error},
        default=str,
        indent=2,
    )


@mcp.tool()
def full_text_search(query_text: str, collection: str, limit: int = 10) -> str:
    """Full-text search across node content in a collection."""
    results = get_client().full_text_search(
        query_text=query_text, limit=limit, collection=collection
    )
    return json.dumps([{"id": r.id, "distance": r.distance} for r in results], indent=2)


@mcp.tool()
def knn_search(query_coords: list[float], collection: str, k: int = 10) -> str:
    """K-nearest-neighbor search in the Poincaré ball. Coords must be 128-dim with magnitude < 1.0."""
    results = get_client().knn_search(
        query=query_coords, k=k, collection=collection
    )
    return json.dumps([{"id": r.id, "distance": r.distance} for r in results], indent=2)


# ── Node CRUD ─────────────────────────────────────────────────────────────


@mcp.tool()
def get_node(node_id: str, collection: str) -> str:
    """Get a single node by UUID, returns content, coords, energy, type."""
    n = get_client().get_node(node_id, collection=collection)
    if n is None:
        return json.dumps({"error": "node not found"})
    return json.dumps(_node_dict(n), default=str, indent=2)


@mcp.tool()
def insert_node(
    content: dict,
    collection: str,
    node_type: str = "Semantic",
    coords: list[float] | None = None,
    energy: float = 1.0,
) -> str:
    """Insert a new node. Coords are 128-dim Poincaré vectors (magnitude < 1.0).

    Magnitude encodes depth: ~0.1=root/abstract, ~0.5=mid-level, ~0.8=leaf/specific.
    If coords not provided, uses zero vector.
    """
    if coords is None:
        coords = [0.0] * 128
    nid = get_client().insert_node(
        coords=coords,
        content=content,
        node_type=node_type,
        energy=energy,
        collection=collection,
    )
    return json.dumps({"node_id": nid})


@mcp.tool()
def delete_node(node_id: str, collection: str) -> str:
    """Delete a node by UUID."""
    ok = get_client().delete_node(node_id, collection=collection)
    return json.dumps({"deleted": ok})


@mcp.tool()
def update_energy(node_id: str, energy: float, collection: str) -> str:
    """Update a node's energy. Energy 0=dormant, 1=fully active."""
    ok = get_client().update_energy(node_id, energy=energy, collection=collection)
    return json.dumps({"updated": ok})


# ── Edge CRUD ─────────────────────────────────────────────────────────────


@mcp.tool()
def insert_edge(
    from_id: str,
    to_id: str,
    collection: str,
    edge_type: str = "Association",
    weight: float = 1.0,
) -> str:
    """Create a directed edge between two nodes.

    Edge types: Association (semantic link), Hierarchical (parent→child), LSystemGenerated (auto).
    """
    eid = get_client().insert_edge(
        from_id=from_id,
        to_id=to_id,
        edge_type=edge_type,
        weight=weight,
        collection=collection,
    )
    return json.dumps({"edge_id": eid})


@mcp.tool()
def delete_edge(edge_id: str, collection: str) -> str:
    """Delete an edge by UUID."""
    ok = get_client().delete_edge(edge_id, collection=collection)
    return json.dumps({"deleted": ok})


# ── Merge/Upsert ─────────────────────────────────────────────────────────


@mcp.tool()
def merge_node(
    node_type: str,
    match_keys: dict,
    on_create: dict,
    collection: str,
    on_match: dict | None = None,
    coords: list[float] | None = None,
    energy: float = 1.0,
) -> str:
    """Upsert a node — find by match_keys or create new. Use for deduplication.

    node_type must be: Semantic, Episodic, or Concept (wire format).
    match_keys: fields to match on (e.g. {"name": "Python"}).
    on_create: fields to set if creating. on_match: fields to update if found.
    """
    if coords is None:
        coords = [0.0] * 128
    nid, created = get_client().merge_node(
        node_type=node_type,
        match_keys=match_keys,
        on_create=on_create,
        on_match=on_match or {},
        coords=coords,
        energy=energy,
        collection=collection,
    )
    return json.dumps({"node_id": nid, "created": created})


# ── Graph Algorithms ──────────────────────────────────────────────────────


@mcp.tool()
def run_pagerank(collection: str, damping: float = 0.85) -> str:
    """Run PageRank — finds the most influential/central nodes in the graph. Returns top 30."""
    r = get_client().run_pagerank(damping=damping, collection=collection)
    sorted_scores = sorted(r.scores.items(), key=lambda x: x[1], reverse=True)[:30]
    return json.dumps(
        {"top_nodes": sorted_scores, "duration_ms": r.duration_ms}, indent=2
    )


@mcp.tool()
def run_louvain(collection: str, resolution: float = 1.0) -> str:
    """Run Louvain community detection — finds clusters of related nodes."""
    r = get_client().run_louvain(resolution=resolution, collection=collection)
    # Group by community
    communities: dict[int, list[str]] = {}
    for node_id, comm_id in r.communities.items():
        communities.setdefault(comm_id, []).append(node_id)
    summary = {
        "num_communities": len(communities),
        "sizes": {k: len(v) for k, v in sorted(communities.items())},
        "duration_ms": r.duration_ms,
    }
    return json.dumps(summary, indent=2)


@mcp.tool()
def run_wcc(collection: str) -> str:
    """Find weakly connected components — identifies disconnected subgraphs."""
    r = get_client().run_wcc(collection=collection)
    components: dict[int, int] = {}
    for _, comp_id in r.communities.items():
        components[comp_id] = components.get(comp_id, 0) + 1
    return json.dumps(
        {
            "num_components": len(components),
            "component_sizes": dict(
                sorted(components.items(), key=lambda x: x[1], reverse=True)[:20]
            ),
            "duration_ms": r.duration_ms,
        },
        indent=2,
    )


@mcp.tool()
def run_betweenness(collection: str, sample_size: int = 100) -> str:
    """Run betweenness centrality — finds bridge nodes connecting clusters. Returns top 20."""
    r = get_client().run_betweenness(
        sample_size=sample_size, collection=collection
    )
    sorted_scores = sorted(r.scores.items(), key=lambda x: x[1], reverse=True)[:20]
    return json.dumps(
        {"top_bridges": sorted_scores, "duration_ms": r.duration_ms}, indent=2
    )


@mcp.tool()
def run_triangle_count(collection: str) -> str:
    """Count triangles in the graph — measures clustering density."""
    count = get_client().run_triangle_count(collection=collection)
    return json.dumps({"triangle_count": count})


# ── Graph Traversal ───────────────────────────────────────────────────────


@mcp.tool()
def bfs(start_id: str, collection: str, max_depth: int = 3, max_nodes: int = 50) -> str:
    """Breadth-first search from a node — explore local neighborhoods."""
    ids = get_client().bfs(
        start_id=start_id,
        max_depth=max_depth,
        max_nodes=max_nodes,
        collection=collection,
    )
    return json.dumps({"node_ids": ids, "count": len(ids)})


@mcp.tool()
def dijkstra(
    start_id: str, collection: str, max_depth: int = 5, max_nodes: int = 50
) -> str:
    """Dijkstra shortest-path from a node — finds closest nodes by weighted distance."""
    ids, costs = get_client().dijkstra(
        start_id=start_id,
        max_depth=max_depth,
        max_nodes=max_nodes,
        collection=collection,
    )
    return json.dumps(
        {"path": list(zip(ids, costs)), "count": len(ids)}, indent=2
    )


# ── Multi-Manifold ────────────────────────────────────────────────────────


@mcp.tool()
def synthesis(node_a: str, node_b: str, collection: str) -> str:
    """Geodesic synthesis — find the Riemannian midpoint between two nodes in the Poincaré ball."""
    r = get_client().synthesis(node_a=node_a, node_b=node_b, collection=collection)
    return json.dumps(
        {
            "coords_preview": r.coords[:5],
            "nearest_node_id": r.nearest_node_id,
            "nearest_distance": r.nearest_distance,
        },
        indent=2,
    )


# ── Lifecycle ─────────────────────────────────────────────────────────────


@mcp.tool()
def trigger_sleep(
    collection: str, noise: float = 0.01, adam_steps: int = 50
) -> str:
    """Trigger reconsolidation sleep — optimizes Poincaré embeddings via gradient descent."""
    r = get_client().trigger_sleep(
        noise=noise, adam_steps=adam_steps, collection=collection
    )
    return json.dumps(
        {
            "hausdorff_before": r.hausdorff_before,
            "hausdorff_after": r.hausdorff_after,
            "hausdorff_delta": r.hausdorff_delta,
            "nodes_perturbed": r.nodes_perturbed,
            "committed": r.committed,
        },
        indent=2,
    )


@mcp.tool()
def reap_expired(collection: str) -> str:
    """Clean up expired nodes/edges (TTL reaper)."""
    count = get_client().reap_expired(collection=collection)
    return json.dumps({"reaped_count": count})


# ── Backup ────────────────────────────────────────────────────────────────


@mcp.tool()
def create_backup(label: str) -> str:
    """Create a named backup of the database."""
    b = get_client().create_backup(label=label)
    return json.dumps(
        {"label": b.label, "path": b.path, "size_bytes": b.size_bytes}, indent=2
    )


@mcp.tool()
def list_backups() -> str:
    """List all available backups."""
    backups = get_client().list_backups()
    return json.dumps(
        [
            {
                "label": b.label,
                "path": b.path,
                "created_at": b.created_at,
                "size_bytes": b.size_bytes,
            }
            for b in backups
        ],
        indent=2,
    )


# ── Cache ─────────────────────────────────────────────────────────────────


@mcp.tool()
def cache_get(key: str, collection: str) -> str:
    """Get a cached value by key."""
    val = get_client().cache_get(key, collection=collection)
    return json.dumps({"key": key, "value": val}, default=str)


@mcp.tool()
def cache_set(key: str, value: str, collection: str, ttl_seconds: int = 0) -> str:
    """Set a cache key with optional TTL in seconds."""
    ok = get_client().cache_set(key, value, ttl_seconds=ttl_seconds, collection=collection)
    return json.dumps({"set": ok})


# ── Helpers ───────────────────────────────────────────────────────────────


def _node_dict(n) -> dict:
    return {
        "id": n.id,
        "content": n.content,
        "node_type": n.node_type,
        "energy": n.energy,
        "depth": n.depth,
        "coords_magnitude": sum(c * c for c in n.coords) ** 0.5 if n.coords else 0,
        "created_at": n.created_at,
    }


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
