#!/usr/bin/env python3
"""
Ingest Gene Ontology (GO) into NietzscheDB.

Downloads the GO OBO file (~45K terms) and builds a hierarchical DAG
with is_a and part_of relationships — perfect for Poincaré ball embeddings.

Usage:
  python scripts/ingest_gene_ontology.py [--host HOST:PORT] [--collection NAME]

Requirements:
  pip install grpcio grpcio-tools requests
"""

import grpc
import json
import uuid
import math
import hashlib
import sys
import os
import argparse
import re
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))
from grpc_tools import protoc
import importlib

# ═══════════════════════════════════════════════════════════════════════════
# PROTO COMPILATION
# ═══════════════════════════════════════════════════════════════════════════

def ensure_proto_compiled(repo_root=None):
    if repo_root is None:
        for candidate in [
            os.path.join(os.path.dirname(__file__), '..'),
            '/home/web2a/NietzscheDB',
            os.path.expanduser('~/NietzscheDB'),
        ]:
            if os.path.isdir(os.path.join(candidate, 'crates', 'nietzsche-api', 'proto')):
                repo_root = candidate
                break
        if repo_root is None:
            raise RuntimeError("Cannot find NietzscheDB repo root.")

    proto_dir = os.path.join(repo_root, 'crates', 'nietzsche-api', 'proto')
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_gen')
    if not os.access(os.path.dirname(os.path.abspath(__file__)), os.W_OK):
        out_dir = '/tmp/_nietzsche_gen'
    os.makedirs(out_dir, exist_ok=True)

    pb2_file = os.path.join(out_dir, 'nietzsche_pb2.py')
    if not os.path.exists(pb2_file):
        print(f"[proto] Compiling nietzsche.proto from {proto_dir}...")
        protoc.main([
            'grpc_tools.protoc',
            f'-I{proto_dir}',
            f'--python_out={out_dir}',
            f'--grpc_python_out={out_dir}',
            'nietzsche.proto',
        ])
        with open(os.path.join(out_dir, '__init__.py'), 'w') as f:
            f.write('')
        print(f"[proto] Compiled to {out_dir}")

    sys.path.insert(0, out_dir)
    pb2 = importlib.import_module('nietzsche_pb2')
    pb2_grpc = importlib.import_module('nietzsche_pb2_grpc')
    return pb2, pb2_grpc


# ═══════════════════════════════════════════════════════════════════════════
# OBO PARSER
# ═══════════════════════════════════════════════════════════════════════════

def download_go_obo(cache_path="/tmp/go-basic.obo"):
    """Download GO OBO file if not cached."""
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1_000_000:
        print(f"[+] Using cached OBO: {cache_path}")
        return cache_path

    import urllib.request
    urls = [
        "https://release.geneontology.org/2024-06-17/ontology/go-basic.obo",
        "https://current.geneontology.org/ontology/go-basic.obo",
        "http://purl.obolibrary.org/obo/go/go-basic.obo",
    ]
    for url in urls:
        try:
            print(f"[*] Trying {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "NietzscheDB/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(cache_path, "wb") as out:
                out.write(resp.read())
            size_mb = os.path.getsize(cache_path) / 1024 / 1024
            if size_mb < 1:
                print(f"[!] File too small ({size_mb:.2f} MB), trying next...")
                continue
            print(f"[+] Downloaded {size_mb:.1f} MB → {cache_path}")
            return cache_path
        except Exception as e:
            print(f"[!] Failed: {e}")
    raise RuntimeError("Could not download GO OBO from any mirror")


def parse_obo(filepath):
    """Parse OBO format into terms and relationships."""
    terms = {}
    current = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == '[Term]':
                current = {
                    'id': None, 'name': None, 'namespace': None,
                    'definition': '', 'is_a': [], 'part_of': [],
                    'is_obsolete': False, 'alt_ids': [], 'synonyms': [],
                }
                continue
            elif line == '[Typedef]' or line == '':
                if current and current['id'] and not current['is_obsolete']:
                    terms[current['id']] = current
                current = None if line == '[Typedef]' else current
                continue

            if current is None:
                continue

            if line.startswith('id: '):
                current['id'] = line[4:]
            elif line.startswith('name: '):
                current['name'] = line[6:]
            elif line.startswith('namespace: '):
                current['namespace'] = line[11:]
            elif line.startswith('def: '):
                # Extract definition between quotes
                m = re.match(r'def: "(.+?)"', line)
                if m:
                    current['definition'] = m.group(1)[:500]
            elif line.startswith('is_a: '):
                parent_id = line[6:].split(' ! ')[0].strip()
                current['is_a'].append(parent_id)
            elif line.startswith('relationship: part_of '):
                parent_id = line[22:].split(' ! ')[0].strip()
                current['part_of'].append(parent_id)
            elif line == 'is_obsolete: true':
                current['is_obsolete'] = True
            elif line.startswith('synonym: '):
                m = re.match(r'synonym: "(.+?)"', line)
                if m:
                    current['synonyms'].append(m.group(1))

    # Capture last term
    if current and current['id'] and not current['is_obsolete']:
        terms[current['id']] = current

    return terms


def compute_depths(terms):
    """BFS from roots to compute depth of each term."""
    children = defaultdict(list)
    for tid, term in terms.items():
        for parent in term['is_a']:
            if parent in terms:
                children[parent].append(tid)
        for parent in term['part_of']:
            if parent in terms:
                children[parent].append(tid)

    # Find roots (no parents)
    roots = [tid for tid, t in terms.items()
             if not t['is_a'] and not t['part_of']]

    depths = {}
    queue = [(r, 0) for r in roots]
    visited = set()

    while queue:
        tid, depth = queue.pop(0)
        if tid in visited:
            continue
        visited.add(tid)
        depths[tid] = depth
        for child in children[tid]:
            if child not in visited:
                queue.append((child, depth + 1))

    # Assign depth 0 to any unvisited (disconnected)
    for tid in terms:
        if tid not in depths:
            depths[tid] = 0

    return depths


# ═══════════════════════════════════════════════════════════════════════════
# POINCARÉ EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════

# GO namespaces → angular sectors for separation
NAMESPACE_ANGLES = {
    'biological_process': 0.0,
    'molecular_function': 2.094,   # 2π/3
    'cellular_component': 4.189,   # 4π/3
}

def make_poincare_embedding(term_id, namespace, depth, max_depth, dim=128):
    """Generate Poincaré embedding with depth-based magnitude."""
    h = hashlib.sha256(term_id.encode()).digest()

    # Magnitude: deeper = further from origin (0.02 to 0.92)
    if max_depth > 0:
        radius = 0.02 + 0.90 * min(depth / max_depth, 1.0)
    else:
        radius = 0.5

    # Add slight jitter
    jitter = (h[0] / 255.0 - 0.5) * 0.03
    radius = max(0.02, min(0.95, radius + jitter))

    # Direction: namespace angle + hash-based spread
    ns_angle = NAMESPACE_ANGLES.get(namespace, 0.0)
    coords = []
    for i in range(dim):
        base = h[i % 32] / 255.0 - 0.5
        # First 2 dims use namespace angle for clustering
        if i == 0:
            base = math.cos(ns_angle) + base * 0.3
        elif i == 1:
            base = math.sin(ns_angle) + base * 0.3
        coords.append(base)

    # Normalize to target radius
    norm = math.sqrt(sum(c * c for c in coords))
    if norm > 0:
        coords = [c / norm * radius for c in coords]

    return coords


# ═══════════════════════════════════════════════════════════════════════════
# INGESTION
# ═══════════════════════════════════════════════════════════════════════════

def make_edge_request(pb2, **kwargs):
    from_val = kwargs.pop('from_node')
    req = pb2.InsertEdgeRequest(**kwargs)
    setattr(req, 'from', from_val)
    return req


def ingest(host, collection, metric="poincare", dim=128):
    pb2, pb2_grpc = ensure_proto_compiled()

    # Download & parse
    obo_path = download_go_obo()
    terms = parse_obo(obo_path)
    depths = compute_depths(terms)
    max_depth = max(depths.values()) if depths else 1

    print(f"\n[+] Parsed {len(terms)} GO terms")
    print(f"[+] Max depth: {max_depth}")

    # Namespace stats
    ns_counts = defaultdict(int)
    for t in terms.values():
        ns_counts[t['namespace'] or 'unknown'] += 1
    for ns, c in sorted(ns_counts.items()):
        print(f"    {ns}: {c} terms")

    # Connect to NietzscheDB
    channel = grpc.insecure_channel(host, options=[
        ('grpc.max_send_message_length', 256 * 1024 * 1024),
        ('grpc.max_receive_message_length', 256 * 1024 * 1024),
    ])
    stub = pb2_grpc.NietzscheDBStub(channel)

    print(f"\n[*] Waiting for gRPC server at {host}...")
    try:
        grpc.channel_ready_future(channel).result(timeout=120)
        print(f"[+] gRPC server ready!")
    except grpc.FutureTimeoutError:
        print(f"[!] Timeout. Aborting.")
        return

    print(f"\n{'='*60}")
    print(f"  Gene Ontology → NietzscheDB Ingestion")
    print(f"  Host: {host}")
    print(f"  Collection: {collection}")
    print(f"  Terms: {len(terms)}")
    print(f"  Metric: {metric} | Dim: {dim}")
    print(f"{'='*60}\n")

    # Create collection
    try:
        stub.CreateCollection(pb2.CreateCollectionRequest(
            collection=collection, dim=dim, metric=metric))
        print(f"[+] Collection '{collection}' created")
    except grpc.RpcError as e:
        print(f"[!] CreateCollection: {e.details() if hasattr(e, 'details') else e}")

    # Build node ID map
    node_ids = {}
    for tid in terms:
        node_ids[tid] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"go:{tid}"))

    # Insert nodes in batches
    all_nodes = []
    for tid, term in terms.items():
        depth = depths.get(tid, 0)
        ns = term['namespace'] or 'unknown'
        emb = make_poincare_embedding(tid, ns, depth, max_depth, dim)

        content = {
            "go_id": tid,
            "name": term['name'],
            "namespace": ns,
            "definition": term['definition'][:300] if term['definition'] else "",
            "depth": depth,
            "dataset": "gene_ontology",
        }
        if term['synonyms']:
            content["synonyms"] = term['synonyms'][:5]

        # Roots = Concept, leaves = Semantic
        node_type = "Concept" if depth <= 2 else "Semantic"
        energy = max(0.3, 1.0 - depth / max_depth * 0.7)

        all_nodes.append({
            "id": node_ids[tid],
            "content": json.dumps(content).encode('utf-8'),
            "node_type": node_type,
            "energy": energy,
            "embedding": emb,
        })

    batch_size = 100
    inserted = 0
    total = len(all_nodes)
    print(f"[*] Inserting {total} nodes in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = all_nodes[i:i + batch_size]
        requests = []
        for n in batch:
            pv = pb2.PoincareVector(coords=n["embedding"])
            req = pb2.InsertNodeRequest(
                id=n["id"], embedding=pv, content=n["content"],
                node_type=n["node_type"], energy=n["energy"],
                collection=collection,
            )
            requests.append(req)
        try:
            stub.BatchInsertNodes(pb2.BatchInsertNodesRequest(
                nodes=requests, collection=collection))
            inserted += len(batch)
            pct = inserted / total * 100
            print(f"  [{pct:5.1f}%] {inserted}/{total} nodes", end='\r')
        except grpc.RpcError as e:
            print(f"\n  [!] Batch {i // batch_size}: {e.details() if hasattr(e, 'details') else e}")

    print(f"\n[+] Nodes inserted: {inserted}/{total}")

    # Insert edges (is_a = Hierarchical, part_of = Association)
    print(f"\n[*] Inserting edges...")
    edge_count = 0
    edge_errors = 0
    edge_batch = []

    for tid, term in terms.items():
        child_id = node_ids.get(tid)
        if not child_id:
            continue

        for parent_id_go in term['is_a']:
            parent_uuid = node_ids.get(parent_id_go)
            if parent_uuid:
                edge_batch.append({
                    "from": parent_uuid, "to": child_id,
                    "edge_type": "Hierarchical", "weight": 1.0,
                })

        for parent_id_go in term['part_of']:
            parent_uuid = node_ids.get(parent_id_go)
            if parent_uuid:
                edge_batch.append({
                    "from": parent_uuid, "to": child_id,
                    "edge_type": "Association", "weight": 0.8,
                })

        # Flush edge batch
        if len(edge_batch) >= 100:
            reqs = []
            for e in edge_batch:
                eid = str(uuid.uuid4())
                req = make_edge_request(pb2,
                    id=eid, from_node=e["from"], to=e["to"],
                    edge_type=e["edge_type"], weight=e["weight"],
                    collection=collection)
                reqs.append(req)
            try:
                stub.BatchInsertEdges(pb2.BatchInsertEdgesRequest(
                    edges=reqs, collection=collection))
                edge_count += len(edge_batch)
            except grpc.RpcError as e:
                edge_errors += 1
                if edge_errors <= 3:
                    print(f"\n  [!] Edge batch error: {e.details() if hasattr(e, 'details') else e}")
            edge_batch = []
            print(f"  Edges: {edge_count}", end='\r')

    # Final batch
    if edge_batch:
        reqs = []
        for e in edge_batch:
            eid = str(uuid.uuid4())
            req = make_edge_request(pb2,
                id=eid, from_node=e["from"], to=e["to"],
                edge_type=e["edge_type"], weight=e["weight"],
                collection=collection)
            reqs.append(req)
        try:
            stub.BatchInsertEdges(pb2.BatchInsertEdgesRequest(
                edges=reqs, collection=collection))
            edge_count += len(edge_batch)
        except grpc.RpcError:
            pass

    print(f"\n[+] Edges inserted: {edge_count}")

    # Summary
    is_a_count = sum(len(t['is_a']) for t in terms.values())
    part_of_count = sum(len(t['part_of']) for t in terms.values())
    print(f"\n{'='*60}")
    print(f"  GENE ONTOLOGY INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Collection:       {collection}")
    print(f"  Terms (nodes):    {inserted}")
    print(f"  is_a edges:       ~{is_a_count}")
    print(f"  part_of edges:    ~{part_of_count}")
    print(f"  Total edges:      {edge_count}")
    print(f"  Max DAG depth:    {max_depth}")
    print(f"  Namespaces:       {len(ns_counts)}")
    print(f"{'='*60}")
    print(f"\n  The Poincaré ball now contains the tree of life's")
    print(f"  molecular functions, biological processes, and")
    print(f"  cellular components. L-System will evolve connections!")
    print(f"\n  Dashboard: http://{host.split(':')[0]}:8080")

    channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Gene Ontology into NietzscheDB")
    parser.add_argument("--host", default="localhost:50051")
    parser.add_argument("--collection", default="gene_ontology")
    parser.add_argument("--metric", default="poincare")
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    ingest(args.host, args.collection, args.metric, args.dim)
