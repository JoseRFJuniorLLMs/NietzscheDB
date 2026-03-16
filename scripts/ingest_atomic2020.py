#!/usr/bin/env python3
"""
Ingest ATOMIC 2020 into NietzscheDB.

ATOMIC 2020 is a commonsense knowledge graph focused on If-Then social
and physical causality. It gives EVA "Theory of Mind" — the ability to
predict human behavior, intentions, and emotional reactions.

Relations include:
  - xIntent: Why does X do this?
  - xNeed: What does X need before doing this?
  - xWant: What does X want after this?
  - xEffect: What happens to X?
  - xReact: How does X feel?
  - oWant/oEffect/oReact: Same for other people
  - isAfter/isBefore: Temporal ordering
  - Causes/xReason: Physical/logical causality
  - HasSubEvent: Event decomposition

Download:
  The ATOMIC 2020 dataset is available at:
  https://allenai.org/data/atomic-2020
  Download the file and place it in --data-dir

Usage:
  # First download ATOMIC 2020 TSV from Allen AI:
  # Place atomic2020_data-feb2021/train.tsv in data dir

  python scripts/ingest_atomic2020.py --host localhost:50051 --collection eva_core
  python scripts/ingest_atomic2020.py --host localhost:50051 --limit 100000

Requirements:
  pip install grpcio grpcio-tools
"""

import grpc
import json
import uuid
import math
import hashlib
import sys
import os
import argparse
import time
import csv
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))
from grpc_tools import protoc
import importlib


# ═══════════════════════════════════════════════════════════════════════════
# PROTO
# ═══════════════════════════════════════════════════════════════════════════

def ensure_proto_compiled(repo_root=None):
    if repo_root is None:
        for candidate in [
            os.path.join(os.path.dirname(__file__), '..'),
            '/home/web2a/NietzscheDB',
            os.path.expanduser('~/NietzscheDB'),
            'd:/DEV/NietzscheDB',
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
        print(f"[proto] Compiling nietzsche.proto...", flush=True)
        protoc.main([
            'grpc_tools.protoc',
            f'-I{proto_dir}',
            f'--python_out={out_dir}',
            f'--grpc_python_out={out_dir}',
            'nietzsche.proto',
        ])
        with open(os.path.join(out_dir, '__init__.py'), 'w') as f:
            f.write('')

    sys.path.insert(0, out_dir)
    return importlib.import_module('nietzsche_pb2'), importlib.import_module('nietzsche_pb2_grpc')


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# ATOMIC relation categories for Poincare depth and Minkowski classification
RELATION_CONFIG = {
    # Physical/logical causality (timelike in Minkowski) → deeper in Poincare
    "Causes":          {"depth": 0.50, "minkowski": "timelike", "category": "causal", "weight": 0.95},
    "xReason":         {"depth": 0.45, "minkowski": "timelike", "category": "causal", "weight": 0.85},
    "HasSubEvent":     {"depth": 0.42, "minkowski": "timelike", "category": "causal", "weight": 0.80},
    "isAfter":         {"depth": 0.48, "minkowski": "timelike", "category": "temporal", "weight": 0.75},
    "isBefore":        {"depth": 0.48, "minkowski": "timelike", "category": "temporal", "weight": 0.75},
    "HinderedBy":      {"depth": 0.45, "minkowski": "timelike", "category": "causal", "weight": 0.80},

    # Intentional/mental states (Theory of Mind)
    "xIntent":         {"depth": 0.35, "minkowski": "timelike", "category": "mental", "weight": 0.85},
    "xNeed":           {"depth": 0.38, "minkowski": "timelike", "category": "mental", "weight": 0.80},
    "xWant":           {"depth": 0.36, "minkowski": "timelike", "category": "mental", "weight": 0.80},
    "xAttr":           {"depth": 0.30, "minkowski": "spacelike", "category": "mental", "weight": 0.70},

    # Effects and reactions
    "xEffect":         {"depth": 0.50, "minkowski": "timelike", "category": "effect", "weight": 0.90},
    "xReact":          {"depth": 0.45, "minkowski": "timelike", "category": "emotion", "weight": 0.85},
    "oWant":           {"depth": 0.36, "minkowski": "timelike", "category": "mental", "weight": 0.75},
    "oEffect":         {"depth": 0.50, "minkowski": "timelike", "category": "effect", "weight": 0.85},
    "oReact":          {"depth": 0.45, "minkowski": "timelike", "category": "emotion", "weight": 0.80},

    # Attributes and properties
    "ObjectUse":       {"depth": 0.32, "minkowski": "spacelike", "category": "property", "weight": 0.70},
    "AtLocation":      {"depth": 0.30, "minkowski": "spacelike", "category": "property", "weight": 0.65},
    "MadeUpOf":        {"depth": 0.28, "minkowski": "spacelike", "category": "property", "weight": 0.70},
    "HasProperty":     {"depth": 0.30, "minkowski": "spacelike", "category": "property", "weight": 0.75},
    "CapableOf":       {"depth": 0.35, "minkowski": "spacelike", "category": "property", "weight": 0.75},
    "Desires":         {"depth": 0.33, "minkowski": "timelike", "category": "mental", "weight": 0.70},
    "NotDesires":      {"depth": 0.33, "minkowski": "timelike", "category": "mental", "weight": 0.65},

    # Hierarchy
    "IsA":             {"depth": 0.15, "minkowski": "spacelike", "category": "hierarchy", "weight": 0.90},
    "InstanceOf":      {"depth": 0.18, "minkowski": "spacelike", "category": "hierarchy", "weight": 0.85},
}

# Default for unknown relations
DEFAULT_CONFIG = {"depth": 0.35, "minkowski": "spacelike", "category": "social", "weight": 0.60}

# Map ATOMIC categories to angular sectors in Poincare ball
SECTOR_MAP = {
    "causal": 0,
    "temporal": 1,
    "mental": 2,
    "effect": 3,
    "emotion": 4,
    "property": 5,
    "hierarchy": 6,
    "social": 7,
}

# Edge type mapping
EDGE_TYPE_MAP = {
    "hierarchy": "Hierarchical",
}  # everything else → Association


# ═══════════════════════════════════════════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════

def make_poincare_embedding(concept, relation, dim=128):
    """Create Poincare embedding with depth from relation config."""
    config = RELATION_CONFIG.get(relation, DEFAULT_CONFIG)
    depth = config["depth"]
    category = config["category"]

    h = hashlib.sha256(concept.encode('utf-8')).digest()
    coords = [0.0] * dim
    for i in range(dim):
        coords[i] = (h[i % len(h)] / 127.5) - 1.0

    # Sector bias
    sector = SECTOR_MAP.get(category, 7)
    sector_start = sector * 8
    for i in range(sector_start, min(sector_start + 8, dim)):
        coords[i] += 2.0

    norm = math.sqrt(sum(c * c for c in coords))
    if norm > 0:
        coords = [c / norm * depth for c in coords]
    return coords


# ═══════════════════════════════════════════════════════════════════════════
# PARSING
# ═══════════════════════════════════════════════════════════════════════════

def find_atomic_file(data_dir):
    """Find ATOMIC 2020 TSV file in data directory."""
    candidates = [
        os.path.join(data_dir, "train.tsv"),
        os.path.join(data_dir, "atomic2020_data-feb2021", "train.tsv"),
        os.path.join(data_dir, "atomic2020", "train.tsv"),
        os.path.join(data_dir, "ATOMIC2020", "train.tsv"),
    ]
    # Also check for any .tsv file
    if os.path.isdir(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith('.tsv') and 'atomic' in f.lower():
                candidates.append(os.path.join(data_dir, f))
            if f.endswith('.tsv') and f == 'train.tsv':
                candidates.append(os.path.join(data_dir, f))

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ═══════════════════════════════════════════════════════════════════════════
# INGESTION
# ═══════════════════════════════════════════════════════════════════════════

def make_edge_request(pb2, **kwargs):
    from_val = kwargs.pop('from_node')
    req = pb2.InsertEdgeRequest(**kwargs)
    setattr(req, 'from', from_val)
    return req


def ingest(host, collection, metric="poincare", dim=128,
           limit=0, data_dir=None, ssl_cert=None, use_ssl=False,
           batch_size=100):
    """
    Streaming ingestion of ATOMIC 2020.

    ATOMIC TSV format: head_event \t relation \t tail_event
    Each head/tail becomes a node, each row becomes an edge.
    """
    pb2, pb2_grpc = ensure_proto_compiled()

    if use_ssl or ssl_cert:
        cert_path = ssl_cert or os.path.expanduser('~/AppData/Local/Temp/eva-cert.pem')
        with open(cert_path, 'rb') as f:
            cert = f.read()
        creds = grpc.ssl_channel_credentials(root_certificates=cert)
        channel = grpc.secure_channel(host, creds, options=[
            ('grpc.max_send_message_length', 256 * 1024 * 1024),
            ('grpc.max_receive_message_length', 256 * 1024 * 1024),
        ])
    else:
        channel = grpc.insecure_channel(host, options=[
            ('grpc.max_send_message_length', 256 * 1024 * 1024),
            ('grpc.max_receive_message_length', 256 * 1024 * 1024),
        ])

    stub = pb2_grpc.NietzscheDBStub(channel)

    print(f"[*] Waiting for gRPC server at {host}...", flush=True)
    try:
        grpc.channel_ready_future(channel).result(timeout=120)
        print(f"[+] gRPC server ready!", flush=True)
    except grpc.FutureTimeoutError:
        print(f"[!] Timeout. Aborting.", flush=True)
        return

    print(f"\n{'='*70}", flush=True)
    print(f"  ATOMIC 2020 → NietzscheDB Streaming Ingestion", flush=True)
    print(f"  Host: {host} | Collection: {collection}", flush=True)
    print(f"{'='*70}\n", flush=True)

    try:
        stub.CreateCollection(pb2.CreateCollectionRequest(
            collection=collection, dim=dim, metric=metric))
        print(f"[+] Collection '{collection}' ensured", flush=True)
    except grpc.RpcError as e:
        print(f"[!] CreateCollection: {e.details() if hasattr(e, 'details') else e}", flush=True)

    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    filepath = find_atomic_file(data_dir)
    if not filepath:
        print(f"[!] ATOMIC 2020 TSV not found in {data_dir}", flush=True)
        print(f"    Download from: https://allenai.org/data/atomic-2020", flush=True)
        print(f"    Place train.tsv in {data_dir}/", flush=True)
        return

    print(f"[+] Found ATOMIC file: {filepath}", flush=True)

    # ─── Streaming parse + insert ────────────────────────────────────────
    concept_ids = {}
    node_batch = []
    edge_batch = []
    rel_counts = defaultdict(int)
    inserted_nodes = 0
    inserted_edges = 0
    total_rows = 0
    skipped = 0
    t0 = time.time()

    def flush_nodes():
        nonlocal inserted_nodes
        if not node_batch:
            return
        requests = []
        for n in node_batch:
            pv = pb2.PoincareVector(coords=n["embedding"])
            req = pb2.InsertNodeRequest(
                id=n["id"], embedding=pv, content=n["content"],
                node_type=n["node_type"], energy=n["energy"],
                collection=collection)
            requests.append(req)
        try:
            resp = stub.BatchInsertNodes(pb2.BatchInsertNodesRequest(
                nodes=requests, collection=collection))
            inserted_nodes += len(node_batch)
            if hasattr(resp, 'backpressure') and resp.backpressure.suggested_delay_ms > 0:
                time.sleep(resp.backpressure.suggested_delay_ms / 1000.0)
        except grpc.RpcError as e:
            print(f"\n  [!] Node error: {e.details() if hasattr(e, 'details') else e}", flush=True)
        node_batch.clear()

    def flush_edges():
        nonlocal inserted_edges
        if not edge_batch:
            return
        reqs = []
        for eb in edge_batch:
            req = make_edge_request(pb2,
                id=str(uuid.uuid4()),
                from_node=eb["from"], to=eb["to"],
                edge_type=eb["type"], weight=eb["weight"],
                collection=collection)
            reqs.append(req)
        try:
            stub.BatchInsertEdges(pb2.BatchInsertEdgesRequest(
                edges=reqs, collection=collection))
            inserted_edges += len(edge_batch)
        except grpc.RpcError as e:
            print(f"\n  [!] Edge error: {e.details() if hasattr(e, 'details') else e}", flush=True)
        edge_batch.clear()

    def ensure_node(concept_text, relation):
        """Get or create a node for this concept text."""
        if concept_text in concept_ids:
            return concept_ids[concept_text]

        nid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"atomic2020:{concept_text}"))
        concept_ids[concept_text] = nid

        config = RELATION_CONFIG.get(relation, DEFAULT_CONFIG)

        content = {
            "name": concept_text,
            "galaxy": "Social Causality",
            "type": "event" if len(concept_text) > 20 else "concept",
            "source": "atomic2020",
            "minkowski_class": config["minkowski"],
        }

        node_batch.append({
            "id": nid,
            "content": json.dumps(content).encode('utf-8'),
            "node_type": "Semantic",
            "energy": 0.7,
            "embedding": make_poincare_embedding(concept_text, relation, dim),
        })

        if len(node_batch) >= batch_size:
            flush_nodes()

        return nid

    print(f"[*] Streaming ATOMIC 2020...", flush=True)

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')

        # Skip header if present
        first_row = next(reader, None)
        if first_row and first_row[0].lower() in ('head', 'event', 'head_event'):
            pass  # was header, skip
        else:
            # Not a header, process it
            if first_row and len(first_row) >= 3:
                head, relation, tail = first_row[0].strip(), first_row[1].strip(), first_row[2].strip()
                if tail and tail != "none" and relation in RELATION_CONFIG:
                    head_id = ensure_node(head, relation)
                    tail_id = ensure_node(tail, relation)
                    config = RELATION_CONFIG[relation]
                    edge_type = EDGE_TYPE_MAP.get(config["category"], "Association")
                    edge_batch.append({
                        "from": head_id, "to": tail_id,
                        "type": edge_type, "weight": config["weight"],
                    })
                    rel_counts[relation] += 1
                    total_rows += 1

        for row in reader:
            if len(row) < 3:
                skipped += 1
                continue

            head = row[0].strip()
            relation = row[1].strip()
            tail = row[2].strip()

            # Skip "none" tails and unknown relations
            if not tail or tail.lower() == "none":
                skipped += 1
                continue

            if relation not in RELATION_CONFIG:
                skipped += 1
                continue

            total_rows += 1
            rel_counts[relation] += 1

            head_id = ensure_node(head, relation)
            tail_id = ensure_node(tail, relation)

            config = RELATION_CONFIG[relation]
            edge_type = EDGE_TYPE_MAP.get(config["category"], "Association")

            edge_batch.append({
                "from": head_id, "to": tail_id,
                "type": edge_type, "weight": config["weight"],
            })

            if len(edge_batch) >= batch_size:
                flush_edges()

            if total_rows % 10000 == 0:
                elapsed = time.time() - t0
                rate = total_rows / elapsed if elapsed > 0 else 0
                print(f"  [{total_rows:>9,}] nodes={inserted_nodes:,} edges={inserted_edges:,} "
                      f"({rate:.0f}/s)", flush=True)

            if limit > 0 and total_rows >= limit:
                break

    # Final flush
    flush_nodes()
    flush_edges()

    elapsed = time.time() - t0

    # ─── Summary ─────────────────────────────────────────────────────────
    n_causal = sum(rel_counts[r] for r in rel_counts
                   if RELATION_CONFIG.get(r, {}).get("category") in ("causal", "temporal"))
    n_mental = sum(rel_counts[r] for r in rel_counts
                   if RELATION_CONFIG.get(r, {}).get("category") == "mental")
    n_emotion = sum(rel_counts[r] for r in rel_counts
                    if RELATION_CONFIG.get(r, {}).get("category") == "emotion")
    n_effect = sum(rel_counts[r] for r in rel_counts
                   if RELATION_CONFIG.get(r, {}).get("category") == "effect")

    print(f"\n{'='*70}", flush=True)
    print(f"  ATOMIC 2020 Ingestion Complete!  ({elapsed:.1f}s)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Collection:    {collection}", flush=True)
    print(f"  Total rows:    {total_rows:,} (skipped {skipped:,})", flush=True)
    print(f"  Unique nodes:  {inserted_nodes:,}", flush=True)
    print(f"  Total edges:   {inserted_edges:,}", flush=True)
    print(f"  ─────────────────────────────────", flush=True)
    print(f"  Causal/temporal: {n_causal:>8,}  (Causes, isAfter, ...)", flush=True)
    print(f"  Mental states:   {n_mental:>8,}  (xIntent, xNeed, xWant, ...)", flush=True)
    print(f"  Emotions:        {n_emotion:>8,}  (xReact, oReact)", flush=True)
    print(f"  Effects:         {n_effect:>8,}  (xEffect, oEffect)", flush=True)
    print(f"  ─────────────────────────────────", flush=True)

    print(f"\n    Relation breakdown:", flush=True)
    for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        config = RELATION_CONFIG.get(rel, DEFAULT_CONFIG)
        print(f"      [{config['minkowski'][:4]:4s}] {rel:20s} {count:>8,}", flush=True)

    print(f"{'='*70}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Ingest ATOMIC 2020 into NietzscheDB")
    parser.add_argument('--host', default='localhost:50051')
    parser.add_argument('--collection', default='eva_core')
    parser.add_argument('--metric', default='poincare')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--ssl', action='store_true')
    parser.add_argument('--cert', default=None)
    args = parser.parse_args()

    ingest(
        host=args.host, collection=args.collection,
        metric=args.metric, dim=args.dim, limit=args.limit,
        data_dir=args.data_dir, ssl_cert=args.cert, use_ssl=args.ssl,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
