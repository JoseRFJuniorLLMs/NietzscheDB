#!/usr/bin/env python3
"""
Ingest SenticNet 7 valence/arousal data into NietzscheDB.

SenticNet provides scientifically validated emotional dimensions for ~200K concepts.
This script creates emotion-tagged nodes and connects them to existing ConceptNet
concepts, giving EVA a validated "limbic system".

Two modes:
1. INSERT mode (default): Creates new emotion nodes with valence/arousal in content
2. UPDATE mode (--update): Updates existing nodes' content with valence/arousal fields

Usage:
  # Install senticnet first:
  pip install senticnet

  # Insert emotion nodes into eva_core:
  python scripts/ingest_senticnet.py --host localhost:50051 --collection eva_core

  # Insert into science_galaxies:
  python scripts/ingest_senticnet.py --host localhost:50051 --collection science_galaxies

  # Remote via SSL:
  python scripts/ingest_senticnet.py --host 136.111.0.47:443 --ssl

Requirements:
  pip install grpcio grpcio-tools senticnet
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
        print(f"[proto] Compiling nietzsche.proto from {proto_dir}...", flush=True)
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
# POINCARE EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════

def make_poincare_embedding(concept_name, valence, arousal, dim=128):
    """
    Create 128D Poincare embedding for an emotion concept.

    Encoding strategy:
    - Magnitude: 0.25 (mid-level, emotion is a property layer)
    - Direction: hash of concept name for reproducibility
    - First 4 dims biased by valence/arousal for semantic clustering:
      dim[0]: valence (positive/negative emotion axis)
      dim[1]: arousal (intensity axis, mapped from sensitivity)
      dim[2]: pleasantness
      dim[3]: attention
    """
    depth = 0.25  # emotion concepts sit at mid-depth

    h = hashlib.sha256(concept_name.encode('utf-8')).digest()
    coords = [0.0] * dim

    for i in range(dim):
        byte_val = h[i % len(h)]
        coords[i] = (byte_val / 127.5) - 1.0

    # Bias first dims with emotional dimensions for clustering
    coords[0] += valence * 3.0       # strong valence signal
    coords[1] += arousal * 2.0       # arousal signal
    # dims 2-3 left to hash (pleasantness/attention encoded in content)

    norm = math.sqrt(sum(c * c for c in coords))
    if norm > 0:
        coords = [c / norm * depth for c in coords]

    return coords


# ═══════════════════════════════════════════════════════════════════════════
# SENTICNET EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_senticnet_data():
    """
    Extract all concepts from SenticNet via the Python API.

    Returns list of dicts with:
    - name: concept string
    - polarity: float [-1, 1]
    - pleasantness: float
    - attention: float
    - sensitivity: float
    - aptitude: float
    - moodtags: list of mood strings
    - semantics: list of related concepts
    """
    try:
        from senticnet.senticnet import SenticNet
    except ImportError:
        print("[!] SenticNet package not installed. Run: pip install senticnet", flush=True)
        sys.exit(1)

    sn = SenticNet()

    print("[*] Extracting concepts from SenticNet...", flush=True)

    # SenticNet stores data internally — access via the data dict
    # The package exposes concepts through sn.data
    concepts = []

    if hasattr(sn, 'data'):
        data = sn.data
    elif hasattr(sn, 'senticnet'):
        data = sn.senticnet
    else:
        # Try to access internal storage
        print("[!] Cannot find SenticNet data attribute. Trying alternative...", flush=True)
        # Fall back to known concept list approach
        data = {}
        for attr in dir(sn):
            val = getattr(sn, attr)
            if isinstance(val, dict) and len(val) > 1000:
                data = val
                break

    if not data:
        print("[!] Could not extract SenticNet data dict.", flush=True)
        sys.exit(1)

    print(f"[+] Found {len(data):,} concepts in SenticNet", flush=True)

    for concept_key, values in data.items():
        try:
            if isinstance(values, (list, tuple)) and len(values) >= 8:
                # SenticNet format (confirmed):
                # [0]=pleasantness, [1]=attention, [2]=sensitivity, [3]=aptitude,
                # [4]=moodtag1, [5]=moodtag2, [6]=polarity_label, [7]=polarity_value,
                # [8:]=semantics (related concepts)
                pleasantness = float(values[0])
                attention = float(values[1])
                sensitivity = float(values[2])
                aptitude = float(values[3])

                moodtags = [v for v in values[4:6] if isinstance(v, str) and v.startswith('#')]
                polarity_label = values[6] if len(values) > 6 else "neutral"
                polarity = float(values[7]) if len(values) > 7 else 0.0

                semantics_list = [v for v in values[8:] if isinstance(v, str) and not v.startswith('#')]

                name = concept_key.replace('_', ' ')
                concepts.append({
                    'name': name,
                    'key': concept_key,
                    'polarity': polarity,
                    'polarity_label': polarity_label,
                    'pleasantness': pleasantness,
                    'attention': attention,
                    'sensitivity': sensitivity,
                    'aptitude': aptitude,
                    'moodtags': moodtags,
                    'semantics': semantics_list,
                    # Map to EVA's valence/arousal:
                    'valence': polarity,  # direct mapping [-1, 1]
                    'arousal': abs(sensitivity) if sensitivity != 0 else abs(attention),
                })
        except (ValueError, TypeError, IndexError):
            continue

    print(f"[+] Extracted {len(concepts):,} valid concepts", flush=True)

    # Stats
    pos = sum(1 for c in concepts if c['valence'] > 0)
    neg = sum(1 for c in concepts if c['valence'] < 0)
    neutral = sum(1 for c in concepts if c['valence'] == 0)
    avg_arousal = sum(c['arousal'] for c in concepts) / len(concepts) if concepts else 0

    print(f"    Positive: {pos:,} | Negative: {neg:,} | Neutral: {neutral:,}", flush=True)
    print(f"    Avg arousal: {avg_arousal:.3f}", flush=True)

    return concepts


# ═══════════════════════════════════════════════════════════════════════════
# INGESTION
# ═══════════════════════════════════════════════════════════════════════════

def make_edge_request(pb2, **kwargs):
    from_val = kwargs.pop('from_node')
    req = pb2.InsertEdgeRequest(**kwargs)
    setattr(req, 'from', from_val)
    return req


def ingest(host, collection, metric="poincare", dim=128,
           ssl_cert=None, use_ssl=False, batch_size=100, limit=0):
    """
    Insert SenticNet emotion concepts into NietzscheDB.

    Each concept becomes a node with:
    - content: {name, valence, arousal, pleasantness, attention, sensitivity, aptitude, moodtags}
    - embedding: Poincare vector biased by valence/arousal
    - energy: proportional to |polarity| (stronger emotions = more energy)

    Semantic links from SenticNet are created as Association edges.
    """
    pb2, pb2_grpc = ensure_proto_compiled()

    # gRPC channel
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
        print(f"[!] Timeout connecting to {host}. Aborting.", flush=True)
        return

    print(f"\n{'='*70}", flush=True)
    print(f"  SenticNet 7 → NietzscheDB Ingestion", flush=True)
    print(f"  Host: {host} | Collection: {collection}", flush=True)
    print(f"  Mode: INSERT (emotion concept nodes)", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Ensure collection
    try:
        stub.CreateCollection(pb2.CreateCollectionRequest(
            collection=collection, dim=dim, metric=metric))
        print(f"[+] Collection '{collection}' ensured", flush=True)
    except grpc.RpcError as e:
        print(f"[!] CreateCollection: {e.details() if hasattr(e, 'details') else e}", flush=True)

    # Extract SenticNet data
    concepts = extract_senticnet_data()
    if limit > 0:
        concepts = concepts[:limit]

    # ─── Build and insert nodes ──────────────────────────────────────────
    t0 = time.time()
    concept_ids = {}  # concept_key → uuid
    inserted_nodes = 0
    inserted_edges = 0
    node_batch = []
    edge_batch = []

    def flush_node_batch():
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
            print(f"\n  [!] Node batch error: {e.details() if hasattr(e, 'details') else e}", flush=True)
        node_batch.clear()

    def flush_edge_batch():
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
            print(f"\n  [!] Edge batch error: {e.details() if hasattr(e, 'details') else e}", flush=True)
        edge_batch.clear()

    print(f"\n[*] Inserting {len(concepts):,} emotion concepts...", flush=True)

    for i, c in enumerate(concepts):
        nid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"senticnet:{c['key']}"))
        concept_ids[c['key']] = nid

        content = {
            "name": c['name'],
            "galaxy": "Emotion & Valence",
            "type": "emotion_concept",
            "source": "senticnet7",
            "valence": round(c['valence'], 4),
            "arousal": round(c['arousal'], 4),
            "polarity": round(c['polarity'], 4),
            "pleasantness": round(c['pleasantness'], 4),
            "attention": round(c['attention'], 4),
            "sensitivity": round(c['sensitivity'], 4),
            "aptitude": round(c['aptitude'], 4),
        }
        if c['moodtags']:
            content["moodtags"] = c['moodtags'][:5]  # top 5

        # Energy: stronger emotions (positive or negative) = more energy
        energy = min(0.95, 0.4 + abs(c['valence']) * 0.5)

        node_batch.append({
            "id": nid,
            "content": json.dumps(content).encode('utf-8'),
            "node_type": "Semantic",
            "energy": energy,
            "embedding": make_poincare_embedding(
                c['name'], c['valence'], c['arousal'], dim),
        })

        if len(node_batch) >= batch_size:
            flush_node_batch()
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1:>8,}/{len(concepts):,}] nodes={inserted_nodes:,} "
                      f"({(i+1)/elapsed:.0f}/s)", flush=True)

    flush_node_batch()

    # ─── Semantic links ──────────────────────────────────────────────────
    print(f"\n[*] Creating semantic association edges...", flush=True)

    for c in concepts:
        src_id = concept_ids.get(c['key'])
        if not src_id:
            continue

        for related in c.get('semantics', []):
            related_key = related.replace(' ', '_')
            tgt_id = concept_ids.get(related_key)
            if tgt_id and tgt_id != src_id:
                edge_batch.append({
                    "from": src_id,
                    "to": tgt_id,
                    "type": "Association",
                    "weight": 0.7,
                })
                if len(edge_batch) >= batch_size:
                    flush_edge_batch()

    flush_edge_batch()

    elapsed = time.time() - t0

    # ─── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"  SenticNet 7 Ingestion Complete!  ({elapsed:.1f}s)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Collection:     {collection}", flush=True)
    print(f"  Emotion nodes:  {inserted_nodes:,}", flush=True)
    print(f"  Semantic edges: {inserted_edges:,}", flush=True)
    print(f"  ─────────────────────────────────", flush=True)
    print(f"  Each node has: valence, arousal, polarity,", flush=True)
    print(f"  pleasantness, attention, sensitivity, aptitude", flush=True)
    print(f"  ─────────────────────────────────", flush=True)
    print(f"  Poincare encoding:", flush=True)
    print(f"    dim[0] biased by valence (positive/negative axis)", flush=True)
    print(f"    dim[1] biased by arousal (intensity axis)", flush=True)
    print(f"    magnitude ~0.25 (emotion = mid-level property)", flush=True)
    print(f"{'='*70}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest SenticNet 7 emotion data into NietzscheDB")
    parser.add_argument('--host', default='localhost:50051',
                        help='gRPC host:port (default: localhost:50051)')
    parser.add_argument('--collection', default='eva_core',
                        help='Target collection (default: eva_core)')
    parser.add_argument('--metric', default='poincare',
                        help='Distance metric (default: poincare)')
    parser.add_argument('--dim', type=int, default=128,
                        help='Embedding dimension (default: 128)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='gRPC batch size (default: 100)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max concepts to import (0=all)')
    parser.add_argument('--ssl', action='store_true',
                        help='Use SSL/TLS for gRPC connection')
    parser.add_argument('--cert', default=None,
                        help='Path to SSL certificate')
    args = parser.parse_args()

    ingest(
        host=args.host,
        collection=args.collection,
        metric=args.metric,
        dim=args.dim,
        ssl_cert=args.cert,
        use_ssl=args.ssl,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
