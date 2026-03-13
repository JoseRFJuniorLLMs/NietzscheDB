#!/usr/bin/env python3
"""
Audit mini collections in NietzscheDB (collections with <= 5 nodes).

Connects via gRPC to NietzscheDB, lists collections, identifies small ones,
fetches all nodes, and recommends: DELETE / KEEP / MERGE for each.

Usage:
    NIETZSCHE_HOST=136.111.0.47:443 python scripts/audit_mini_collections.py
    # or with local tunnel:
    NIETZSCHE_HOST=localhost:50051 python scripts/audit_mini_collections.py
"""

import os
import sys
import json
import grpc

# Add SDK path
SDK_PATH = os.path.join(os.path.dirname(__file__), "..", "sdks", "python")
sys.path.insert(0, SDK_PATH)

from nietzschedb.proto import nietzsche_pb2 as pb, nietzsche_pb2_grpc as rpc
from google.protobuf import empty_pb2

# Target collections to audit (known to have ~2 nodes each)
TARGET_COLLECTIONS = [
    "memories",
    "stories",
    "signifier_chains",
    "speaker_embeddings",
    "eva_cache",
]

# Threshold: collections with <= this many nodes are "mini"
MAX_NODES_THRESHOLD = 5


def connect():
    """Connect to NietzscheDB via gRPC."""
    host = os.environ.get("NIETZSCHE_HOST", "136.111.0.47:443")

    if ":443" in host:
        # TLS connection via nginx proxy
        cert_path = os.path.expanduser("~/AppData/Local/Temp/eva-cert.pem")
        if not os.path.exists(cert_path):
            # Try Linux path
            cert_path = "/tmp/eva-cert.pem"
        if os.path.exists(cert_path):
            with open(cert_path, "rb") as f:
                cert = f.read()
            creds = grpc.ssl_channel_credentials(root_certificates=cert)
            channel = grpc.secure_channel(host, creds)
        else:
            print(f"WARNING: No cert found, trying insecure connection to {host}")
            channel = grpc.insecure_channel(host)
    else:
        channel = grpc.insecure_channel(host)

    stub = rpc.NietzscheDBStub(channel)
    return stub, channel


def get_all_collections(stub):
    """List all collections with their stats."""
    resp = stub.ListCollections(empty_pb2.Empty(), timeout=30)
    collections = {}
    for c in resp.collections:
        collections[c.name] = {
            "node_count": c.node_count,
            "edge_count": c.edge_count,
            "dimension": c.dimension,
            "geometry": c.geometry,
        }
    return collections


def fetch_nodes(stub, collection, limit=50):
    """Fetch nodes from a collection using NQL query."""
    try:
        req = pb.NQLRequest(
            nql=f"MATCH (n) IN {collection} RETURN n LIMIT {limit}",
        )
        resp = stub.QueryNQL(req, timeout=30)
        nodes = []
        for row in resp.rows:
            try:
                node_data = json.loads(row.data)
                nodes.append(node_data)
            except (json.JSONDecodeError, AttributeError):
                nodes.append({"raw": str(row)})
        return nodes
    except grpc.RpcError as e:
        print(f"  ERROR fetching nodes from {collection}: {e.code()} {e.details()}")
        return []


def analyze_collection(name, stats, nodes):
    """Analyze a collection and return a recommendation."""
    node_count = stats.get("node_count", 0)

    result = {
        "name": name,
        "node_count": node_count,
        "edge_count": stats.get("edge_count", 0),
        "dimension": stats.get("dimension", 0),
        "geometry": stats.get("geometry", ""),
        "nodes_sample": nodes,
        "recommendation": "UNKNOWN",
        "reason": "",
    }

    # Empty collection
    if node_count == 0:
        result["recommendation"] = "DELETE"
        result["reason"] = "Empty collection with 0 nodes"
        return result

    # Check if nodes have meaningful content
    has_real_data = False
    has_test_data = False
    content_summary = []

    for node in nodes:
        content = node.get("content", {})
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                content = {"raw": content}

        # Check for test/debug markers
        content_str = json.dumps(content, default=str).lower()
        if any(kw in content_str for kw in ["test", "debug", "dummy", "example", "lorem"]):
            has_test_data = True
        elif len(content_str) > 20:
            has_real_data = True

        # Summarize content
        if isinstance(content, dict):
            keys = list(content.keys())[:5]
            summary = f"keys={keys}"
        else:
            summary = str(content)[:100]
        content_summary.append(summary)

    result["content_summary"] = content_summary

    # Recommendations based on analysis
    if has_real_data and not has_test_data:
        if node_count <= 2:
            result["recommendation"] = "MERGE"
            result["reason"] = f"Only {node_count} nodes with real data — consider merging into parent collection"
        else:
            result["recommendation"] = "KEEP"
            result["reason"] = f"Has {node_count} nodes with real data"
    elif has_test_data and not has_real_data:
        result["recommendation"] = "DELETE"
        result["reason"] = "Contains only test/debug data"
    elif node_count <= 2:
        result["recommendation"] = "MERGE"
        result["reason"] = f"Only {node_count} nodes — too small to justify a separate collection"
    else:
        result["recommendation"] = "KEEP"
        result["reason"] = f"Has {node_count} nodes, needs manual review"

    return result


def print_report(results, all_collections):
    """Print the audit report."""
    print("\n" + "=" * 80)
    print("  NietzscheDB Mini Collections Audit Report")
    print("=" * 80)

    # Summary of all collections
    mini = {k: v for k, v in all_collections.items() if v["node_count"] <= MAX_NODES_THRESHOLD}
    print(f"\nTotal collections: {len(all_collections)}")
    print(f"Mini collections (<= {MAX_NODES_THRESHOLD} nodes): {len(mini)}")

    if mini:
        print("\nAll mini collections:")
        for name, stats in sorted(mini.items(), key=lambda x: x[1]["node_count"]):
            marker = " *" if name in TARGET_COLLECTIONS else ""
            print(f"  {name}: {stats['node_count']} nodes, {stats['edge_count']} edges{marker}")

    # Detailed audit of target collections
    print("\n" + "-" * 80)
    print("  DETAILED AUDIT (target collections)")
    print("-" * 80)

    delete_list = []
    merge_list = []
    keep_list = []

    for r in results:
        rec = r["recommendation"]
        print(f"\n{'#' * 60}")
        print(f"  Collection: {r['name']}")
        print(f"  Nodes: {r['node_count']}  |  Edges: {r['edge_count']}  |  Dim: {r['dimension']}  |  Geo: {r['geometry']}")
        print(f"  Recommendation: {rec}")
        print(f"  Reason: {r['reason']}")

        if r.get("content_summary"):
            print(f"  Content preview:")
            for i, cs in enumerate(r["content_summary"][:3]):
                print(f"    [{i}] {cs[:120]}")

        if r.get("nodes_sample"):
            print(f"  Raw node sample (first node):")
            sample = json.dumps(r["nodes_sample"][0], indent=2, default=str, ensure_ascii=False)
            for line in sample.split("\n")[:15]:
                print(f"    {line}")
            if len(sample.split("\n")) > 15:
                print(f"    ... ({len(sample.split(chr(10)))} lines total)")

        if rec == "DELETE":
            delete_list.append(r["name"])
        elif rec == "MERGE":
            merge_list.append(r["name"])
        else:
            keep_list.append(r["name"])

    # Action summary
    print("\n" + "=" * 80)
    print("  ACTION SUMMARY")
    print("=" * 80)

    if delete_list:
        print(f"\n  DELETE ({len(delete_list)}):")
        for name in delete_list:
            print(f"    - {name}")
        print(f"\n  To delete via gRPC:")
        for name in delete_list:
            print(f"    stub.DropCollection(pb.DropCollectionRequest(collection='{name}'))")

    if merge_list:
        print(f"\n  MERGE ({len(merge_list)}):")
        for name in merge_list:
            print(f"    - {name}  -->  suggest merge into eva_mind or parent collection")

    if keep_list:
        print(f"\n  KEEP ({len(keep_list)}):")
        for name in keep_list:
            print(f"    - {name}")

    print("\n" + "=" * 80)


def main():
    print("Connecting to NietzscheDB...")
    stub, channel = connect()

    # Health check
    try:
        health = stub.HealthCheck(empty_pb2.Empty(), timeout=10)
        print(f"Health: {health.status}")
    except grpc.RpcError as e:
        print(f"Health check failed: {e.code()} {e.details()}")
        sys.exit(1)

    # Get all collections
    print("Fetching collections...")
    all_collections = get_all_collections(stub)
    print(f"Found {len(all_collections)} collections")

    # Audit target collections
    results = []
    for name in TARGET_COLLECTIONS:
        if name not in all_collections:
            print(f"  SKIP: {name} not found in server")
            results.append({
                "name": name,
                "node_count": 0,
                "edge_count": 0,
                "dimension": 0,
                "geometry": "",
                "recommendation": "DELETE",
                "reason": "Collection does not exist on server",
            })
            continue

        stats = all_collections[name]
        print(f"  Auditing: {name} ({stats['node_count']} nodes)...")

        nodes = []
        if stats["node_count"] > 0:
            nodes = fetch_nodes(stub, name)

        result = analyze_collection(name, stats, nodes)
        results.append(result)

    # Also find any OTHER mini collections not in target list
    for name, stats in all_collections.items():
        if name not in TARGET_COLLECTIONS and stats["node_count"] <= MAX_NODES_THRESHOLD:
            print(f"  Also mini: {name} ({stats['node_count']} nodes) — not in target list")

    print_report(results, all_collections)

    channel.close()


if __name__ == "__main__":
    main()
