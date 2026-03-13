"""NietzscheDB tools for the Claude Agent — wraps the Python SDK as Claude tool definitions."""

from __future__ import annotations

import json
import sys
import os
import traceback
from typing import Any, Dict, List

# Add SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdks", "python"))

from nietzschedb import NietzscheClient
from config import NIETZSCHE_HOST, NIETZSCHE_INSECURE, NIETZSCHE_CERT_PATH

# ── Singleton client ──────────────────────────────────────────────────────

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


# ── Tool definitions (Claude tool_use format) ────────────────────────────

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "list_collections",
        "description": "List all NietzscheDB collections with node/edge counts, dimension, and metric.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_stats",
        "description": "Get global NietzscheDB statistics (total nodes, edges, version).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "query_nql",
        "description": "Execute an NQL (Nietzsche Query Language) query. Examples: 'MATCH (n:Semantic) RETURN n LIMIT 20', 'MATCH (n)-[e]->(m) WHERE n.energy > 0.5 RETURN n,m LIMIT 50'. Node types: Episodic, Semantic, Concept, DreamSnapshot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "nql": {"type": "string", "description": "NQL query string"},
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["nql", "collection"],
        },
    },
    {
        "name": "get_node",
        "description": "Get a single node by UUID, returns its content, coords, energy, type.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Node UUID"},
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["node_id", "collection"],
        },
    },
    {
        "name": "insert_node",
        "description": "Insert a new node into the graph. Coords are Poincaré ball vectors (magnitude < 1.0). Magnitude encodes depth: ~0.1 = root/abstract, ~0.5 = mid-level, ~0.8 = leaf/specific.",
        "input_schema": {
            "type": "object",
            "properties": {
                "coords": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Poincaré ball coordinates (128-dim, magnitude < 1.0)",
                },
                "content": {
                    "type": "object",
                    "description": "JSON content (must include 'text' or 'name' field)",
                },
                "node_type": {
                    "type": "string",
                    "enum": ["Semantic", "Episodic", "Concept", "DreamSnapshot"],
                    "description": "Node type",
                },
                "energy": {
                    "type": "number",
                    "description": "Energy value (0.0-1.0, default 1.0)",
                },
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["coords", "content", "node_type", "collection"],
        },
    },
    {
        "name": "insert_edge",
        "description": "Create a directed edge between two nodes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_id": {"type": "string", "description": "Source node UUID"},
                "to_id": {"type": "string", "description": "Target node UUID"},
                "edge_type": {
                    "type": "string",
                    "enum": ["Association", "Hierarchical", "LSystemGenerated"],
                    "description": "Edge type",
                },
                "weight": {"type": "number", "description": "Edge weight (default 1.0)"},
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["from_id", "to_id", "edge_type", "collection"],
        },
    },
    {
        "name": "delete_edge",
        "description": "Delete an edge by UUID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "edge_id": {"type": "string", "description": "Edge UUID"},
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["edge_id", "collection"],
        },
    },
    {
        "name": "delete_node",
        "description": "Delete a node by UUID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Node UUID"},
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["node_id", "collection"],
        },
    },
    {
        "name": "knn_search",
        "description": "K-nearest-neighbor search in the Poincaré ball. Returns closest nodes by hyperbolic distance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Query vector (128-dim Poincaré coords)",
                },
                "k": {"type": "integer", "description": "Number of neighbors (default 10)"},
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["query", "k", "collection"],
        },
    },
    {
        "name": "full_text_search",
        "description": "Full-text search across node content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query_text": {"type": "string", "description": "Search query text"},
                "limit": {"type": "integer", "description": "Max results (default 10)"},
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["query_text", "collection"],
        },
    },
    {
        "name": "run_pagerank",
        "description": "Run PageRank to find the most influential/central nodes in the graph.",
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Collection name"},
                "damping": {"type": "number", "description": "Damping factor (default 0.85)"},
            },
            "required": ["collection"],
        },
    },
    {
        "name": "run_louvain",
        "description": "Run Louvain community detection to find clusters of related nodes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Collection name"},
                "resolution": {"type": "number", "description": "Resolution (default 1.0, higher = more communities)"},
            },
            "required": ["collection"],
        },
    },
    {
        "name": "run_wcc",
        "description": "Find weakly connected components — identifies disconnected subgraphs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["collection"],
        },
    },
    {
        "name": "run_betweenness",
        "description": "Run betweenness centrality — finds bridge nodes connecting different clusters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Collection name"},
                "sample_size": {"type": "integer", "description": "Sample size (default 100)"},
            },
            "required": ["collection"],
        },
    },
    {
        "name": "run_triangle_count",
        "description": "Count triangles in the graph — measures clustering density.",
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["collection"],
        },
    },
    {
        "name": "bfs",
        "description": "Breadth-first search from a starting node — explore local neighborhoods.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_id": {"type": "string", "description": "Starting node UUID"},
                "max_depth": {"type": "integer", "description": "Max depth (default 3)"},
                "max_nodes": {"type": "integer", "description": "Max nodes to return (default 50)"},
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["start_id", "collection"],
        },
    },
    {
        "name": "merge_node",
        "description": "Upsert a node — find by match_keys or create new. Use for deduplication.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_type": {
                    "type": "string",
                    "enum": ["Semantic", "Episodic", "Concept"],
                    "description": "Node type (wire format, not custom labels)",
                },
                "match_keys": {
                    "type": "object",
                    "description": "Fields to match on (e.g. {\"name\": \"Python\"})",
                },
                "on_create": {
                    "type": "object",
                    "description": "Fields to set if creating new node",
                },
                "on_match": {
                    "type": "object",
                    "description": "Fields to update if node already exists",
                },
                "coords": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Poincaré coords (128-dim)",
                },
                "energy": {"type": "number", "description": "Energy (default 1.0)"},
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["node_type", "match_keys", "on_create", "collection"],
        },
    },
    {
        "name": "synthesis",
        "description": "Geodesic synthesis — find the Riemannian midpoint between two nodes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_a": {"type": "string", "description": "First node UUID"},
                "node_b": {"type": "string", "description": "Second node UUID"},
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["node_a", "node_b", "collection"],
        },
    },
    {
        "name": "update_energy",
        "description": "Update a node's energy value. Energy 0 = dormant, 1 = fully active.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Node UUID"},
                "energy": {"type": "number", "description": "New energy value (0.0-1.0)"},
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["node_id", "energy", "collection"],
        },
    },
    {
        "name": "reap_expired",
        "description": "Clean up expired nodes/edges (TTL reaper).",
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Collection name"},
            },
            "required": ["collection"],
        },
    },
    {
        "name": "trigger_sleep",
        "description": "Trigger reconsolidation sleep cycle — optimizes Poincaré embeddings via Adam steps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Collection name"},
                "noise": {"type": "number", "description": "Noise magnitude (default 0.01)"},
                "adam_steps": {"type": "integer", "description": "Adam optimizer steps (default 50)"},
            },
            "required": ["collection"],
        },
    },
    {
        "name": "create_backup",
        "description": "Create a backup of the database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Backup label/name"},
            },
            "required": ["label"],
        },
    },
]


# ── Tool execution ────────────────────────────────────────────────────────

def execute_tool(name: str, input_data: Dict[str, Any]) -> str:
    """Execute a tool by name and return JSON result string."""
    try:
        client = get_client()
        result = _dispatch(client, name, input_data)
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


def _dispatch(client: NietzscheClient, name: str, inp: Dict[str, Any]) -> Any:
    """Route tool call to the appropriate SDK method."""

    if name == "list_collections":
        cols = client.list_collections()
        return [
            {"name": c.name, "dim": c.dim, "metric": c.metric,
             "node_count": c.node_count, "edge_count": c.edge_count}
            for c in cols
        ]

    if name == "get_stats":
        s = client.get_stats()
        return {"node_count": s.node_count, "edge_count": s.edge_count,
                "version": s.version, "sensory_count": s.sensory_count}

    if name == "query_nql":
        r = client.query(inp["nql"], collection=inp["collection"])
        nodes = [_node_dict(n) for n in r.nodes]
        return {"nodes": nodes, "scalar_rows": r.scalar_rows, "error": r.error}

    if name == "get_node":
        n = client.get_node(inp["node_id"], collection=inp["collection"])
        return _node_dict(n) if n else {"error": "node not found"}

    if name == "insert_node":
        nid = client.insert_node(
            coords=inp["coords"],
            content=inp["content"],
            node_type=inp.get("node_type", "Semantic"),
            energy=inp.get("energy", 1.0),
            collection=inp["collection"],
        )
        return {"node_id": nid}

    if name == "insert_edge":
        eid = client.insert_edge(
            from_id=inp["from_id"],
            to_id=inp["to_id"],
            edge_type=inp.get("edge_type", "Association"),
            weight=inp.get("weight", 1.0),
            collection=inp["collection"],
        )
        return {"edge_id": eid}

    if name == "delete_edge":
        ok = client.delete_edge(inp["edge_id"], collection=inp["collection"])
        return {"deleted": ok}

    if name == "delete_node":
        ok = client.delete_node(inp["node_id"], collection=inp["collection"])
        return {"deleted": ok}

    if name == "knn_search":
        results = client.knn_search(
            query=inp["query"], k=inp.get("k", 10), collection=inp["collection"]
        )
        return [{"id": r.id, "distance": r.distance} for r in results]

    if name == "full_text_search":
        results = client.full_text_search(
            query_text=inp["query_text"],
            limit=inp.get("limit", 10),
            collection=inp["collection"],
        )
        return [{"id": r.id, "distance": r.distance} for r in results]

    if name == "run_pagerank":
        r = client.run_pagerank(
            damping=inp.get("damping", 0.85), collection=inp["collection"]
        )
        # Sort by score descending, return top 30
        sorted_scores = sorted(r.scores.items(), key=lambda x: x[1], reverse=True)[:30]
        return {"top_nodes": sorted_scores, "duration_ms": r.duration_ms}

    if name == "run_louvain":
        r = client.run_louvain(
            resolution=inp.get("resolution", 1.0), collection=inp["collection"]
        )
        return {"communities": r.communities, "duration_ms": r.duration_ms}

    if name == "run_wcc":
        r = client.run_wcc(collection=inp["collection"])
        return {"communities": r.communities, "duration_ms": r.duration_ms}

    if name == "run_betweenness":
        r = client.run_betweenness(
            sample_size=inp.get("sample_size", 100), collection=inp["collection"]
        )
        sorted_scores = sorted(r.scores.items(), key=lambda x: x[1], reverse=True)[:30]
        return {"top_bridges": sorted_scores, "duration_ms": r.duration_ms}

    if name == "run_triangle_count":
        count = client.run_triangle_count(collection=inp["collection"])
        return {"triangle_count": count}

    if name == "bfs":
        ids = client.bfs(
            start_id=inp["start_id"],
            max_depth=inp.get("max_depth", 3),
            max_nodes=inp.get("max_nodes", 50),
            collection=inp["collection"],
        )
        return {"node_ids": ids, "count": len(ids)}

    if name == "merge_node":
        nid, created = client.merge_node(
            node_type=inp["node_type"],
            match_keys=inp["match_keys"],
            on_create=inp["on_create"],
            on_match=inp.get("on_match", {}),
            coords=inp.get("coords", [0.0] * 128),
            energy=inp.get("energy", 1.0),
            collection=inp["collection"],
        )
        return {"node_id": nid, "created": created}

    if name == "synthesis":
        r = client.synthesis(
            node_a=inp["node_a"], node_b=inp["node_b"], collection=inp["collection"]
        )
        return {"coords": r.coords[:5], "nearest_node_id": r.nearest_node_id,
                "nearest_distance": r.nearest_distance}

    if name == "update_energy":
        ok = client.update_energy(
            node_id=inp["node_id"], energy=inp["energy"], collection=inp["collection"]
        )
        return {"updated": ok}

    if name == "reap_expired":
        count = client.reap_expired(collection=inp["collection"])
        return {"reaped_count": count}

    if name == "trigger_sleep":
        r = client.trigger_sleep(
            noise=inp.get("noise", 0.01),
            adam_steps=inp.get("adam_steps", 50),
            collection=inp["collection"],
        )
        return {
            "hausdorff_before": r.hausdorff_before,
            "hausdorff_after": r.hausdorff_after,
            "hausdorff_delta": r.hausdorff_delta,
            "nodes_perturbed": r.nodes_perturbed,
            "committed": r.committed,
        }

    if name == "create_backup":
        b = client.create_backup(label=inp["label"])
        return {"label": b.label, "path": b.path, "size_bytes": b.size_bytes}

    return {"error": f"Unknown tool: {name}"}


def _node_dict(n) -> Dict[str, Any]:
    """Convert a Node to a serializable dict."""
    return {
        "id": n.id,
        "content": n.content,
        "node_type": n.node_type,
        "energy": n.energy,
        "depth": n.depth,
        "coords_magnitude": sum(c * c for c in n.coords) ** 0.5 if n.coords else 0,
        "created_at": n.created_at,
    }
