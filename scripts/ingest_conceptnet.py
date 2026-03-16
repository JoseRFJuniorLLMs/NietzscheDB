#!/usr/bin/env python3
"""
Ingest ConceptNet 5.8 into NietzscheDB.

Downloads and imports the ConceptNet commonsense knowledge graph,
filtering for useful causal/physical/social relations in English and Portuguese.

This gives EVA physical intuition and common sense:
- "glass is fragile", "if you drop something, it falls"
- "water extinguishes fire", "you can't walk through a wall"

Usage:
  python scripts/ingest_conceptnet.py [--host HOST:PORT] [--collection NAME]
  python scripts/ingest_conceptnet.py --host localhost:50051 --collection science_galaxies
  python scripts/ingest_conceptnet.py --host localhost:50051 --limit 50000

  # Remote via SSL (443):
  python scripts/ingest_conceptnet.py --host 136.111.0.47:443 --ssl --cert ~/AppData/Local/Temp/eva-cert.pem

Requirements:
  pip install grpcio grpcio-tools
"""

import grpc
import json
import uuid
import math
import hashlib
import gzip
import sys
import os
import argparse
import urllib.request
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))
from grpc_tools import protoc
import importlib


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

CONCEPTNET_URL = "https://conceptnet.s3.amazonaws.com/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
# Note: 5.8.1 URL may differ; 5.7 CSV is the most reliable download format

# Relations we care about for EVA's world model, grouped by semantic role
CAUSAL_RELATIONS = {
    # Physical causality (Livro 3 - Genesis Sintetico)
    "Causes",           # X causes Y
    "CausesDesire",     # X makes you want Y
    "HasPrerequisite",  # X requires Y first
    "HasFirstSubevent", # X starts with Y
    "HasLastSubevent",  # X ends with Y
    "HasSubevent",      # X involves Y
}

PROPERTY_RELATIONS = {
    # Material / physical properties (Livro 1 - Senso Comum)
    "HasProperty",      # X has property Y (glass HasProperty fragile)
    "CapableOf",        # X can do Y (bird CapableOf fly)
    "MadeOf",           # X is made of Y
    "PartOf",           # X is part of Y
    "HasA",             # X has a Y
    "UsedFor",          # X is used for Y
    "AtLocation",       # X is found at Y
    "ReceivesAction",   # X can be Y-ed
}

HIERARCHY_RELATIONS = {
    # Taxonomic / ontological (Livro 2 - SUMO style)
    "IsA",              # X is a type of Y
    "InstanceOf",       # X is an instance of Y
    "DefinedAs",        # X means Y
    "MannerOf",         # X is a way to Y
}

SOCIAL_RELATIONS = {
    # Social / emotional (bonus: proto-ATOMIC)
    "MotivatedByGoal",  # X is motivated by Y
    "Desires",          # X wants Y
    "CreatedBy",        # X is created by Y
    "SymbolOf",         # X symbolizes Y
    "DistinctFrom",     # X is NOT Y
    "Antonym",          # X is opposite of Y
    "Synonym",          # X means Y
    "SimilarTo",        # X is similar to Y
    "RelatedTo",        # X is related to Y (most generic)
}

ALL_RELATIONS = CAUSAL_RELATIONS | PROPERTY_RELATIONS | HIERARCHY_RELATIONS | SOCIAL_RELATIONS

# Languages to keep
LANGUAGES = {"en", "pt"}

# Poincare depth mapping by relation type
# Abstract/structural relations → closer to center
# Specific/causal relations → further from center
DEPTH_MAP = {
    # Hierarchy (abstract) → low magnitude
    "IsA": 0.15,
    "InstanceOf": 0.18,
    "DefinedAs": 0.15,
    "MannerOf": 0.20,
    # Properties → mid magnitude
    "HasProperty": 0.30,
    "CapableOf": 0.35,
    "MadeOf": 0.28,
    "PartOf": 0.25,
    "HasA": 0.30,
    "UsedFor": 0.35,
    "AtLocation": 0.32,
    "ReceivesAction": 0.35,
    # Causal → high magnitude (specific/concrete)
    "Causes": 0.50,
    "CausesDesire": 0.48,
    "HasPrerequisite": 0.42,
    "HasFirstSubevent": 0.45,
    "HasLastSubevent": 0.45,
    "HasSubevent": 0.43,
    # Social/generic → mid magnitude
    "MotivatedByGoal": 0.40,
    "Desires": 0.38,
    "CreatedBy": 0.32,
    "SymbolOf": 0.35,
    "DistinctFrom": 0.30,
    "Antonym": 0.28,
    "Synonym": 0.20,
    "SimilarTo": 0.25,
    "RelatedTo": 0.30,
}

# Edge type mapping for NietzscheDB
EDGE_TYPE_MAP = {
    # Hierarchical edges
    "IsA": "Hierarchical",
    "InstanceOf": "Hierarchical",
    "DefinedAs": "Hierarchical",
    "MannerOf": "Hierarchical",
    "PartOf": "Hierarchical",
    # Everything else → Association
}

# Weight mapping based on relation importance for EVA
WEIGHT_MAP = {
    "Causes": 0.95,
    "IsA": 0.90,
    "HasProperty": 0.85,
    "CapableOf": 0.85,
    "HasPrerequisite": 0.80,
    "MadeOf": 0.80,
    "UsedFor": 0.80,
    "PartOf": 0.85,
    "HasA": 0.75,
    "AtLocation": 0.70,
    "ReceivesAction": 0.70,
    "CausesDesire": 0.75,
    "HasSubevent": 0.70,
    "HasFirstSubevent": 0.70,
    "HasLastSubevent": 0.70,
    "MotivatedByGoal": 0.65,
    "Desires": 0.60,
    "CreatedBy": 0.65,
    "SymbolOf": 0.55,
    "DefinedAs": 0.80,
    "MannerOf": 0.70,
    "InstanceOf": 0.85,
    "DistinctFrom": 0.50,
    "Antonym": 0.60,
    "Synonym": 0.75,
    "SimilarTo": 0.55,
    "RelatedTo": 0.40,
}


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

    sys.path.insert(0, out_dir)
    return importlib.import_module('nietzsche_pb2'), importlib.import_module('nietzsche_pb2_grpc')


# ═══════════════════════════════════════════════════════════════════════════
# POINCARE EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════

# Angular sectors for relation categories (radians in 128D hypersphere)
SECTOR_MAP = {
    "causal": 0,
    "property": 1,
    "hierarchy": 2,
    "social": 3,
}


def get_relation_category(relation):
    """Map a ConceptNet relation to a category for angular sector encoding."""
    if relation in CAUSAL_RELATIONS:
        return "causal"
    elif relation in PROPERTY_RELATIONS:
        return "property"
    elif relation in HIERARCHY_RELATIONS:
        return "hierarchy"
    elif relation in SOCIAL_RELATIONS:
        return "social"
    return "social"


def make_poincare_embedding(concept_name, relation, dim=128):
    """
    Create a 128D Poincare embedding for a ConceptNet concept.

    Encodes:
    - Magnitude (depth in Poincare ball) from DEPTH_MAP based on relation type
    - Direction (angular position) from a hash of the concept name
    - Sector (first few dims) from relation category
    """
    depth = DEPTH_MAP.get(relation, 0.30)

    # Hash the concept name for reproducible angular direction
    h = hashlib.sha256(concept_name.encode('utf-8')).digest()

    # Build direction vector from hash bytes
    coords = [0.0] * dim
    for i in range(dim):
        byte_val = h[i % len(h)]
        coords[i] = (byte_val / 127.5) - 1.0  # normalize to [-1, 1]

    # Boost sector dimensions for relation category grouping
    sector = SECTOR_MAP.get(get_relation_category(relation), 0)
    sector_start = sector * 8  # 4 sectors × 8 dims = 32 dims for sector encoding
    for i in range(sector_start, min(sector_start + 8, dim)):
        coords[i] += 2.0  # bias toward sector

    # Normalize to unit sphere, then scale to desired depth
    norm = math.sqrt(sum(c * c for c in coords))
    if norm > 0:
        coords = [c / norm * depth for c in coords]

    return coords


# ═══════════════════════════════════════════════════════════════════════════
# CONCEPTNET PARSING
# ═══════════════════════════════════════════════════════════════════════════

def clean_concept(uri):
    """
    Extract clean concept name from ConceptNet URI.
    /c/en/broken_glass -> broken glass
    /c/pt/vidro_partido -> vidro partido
    """
    parts = uri.strip('/').split('/')
    if len(parts) < 3:
        return None, None
    lang = parts[1]
    name = parts[2].replace('_', ' ')
    return lang, name


def parse_relation(uri):
    """Extract relation name from URI: /r/Causes -> Causes"""
    return uri.strip('/').split('/')[-1]


def download_conceptnet(data_dir):
    """Download ConceptNet assertions file if not present."""
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.basename(CONCEPTNET_URL)
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        print(f"[+] ConceptNet file already exists: {filepath}")
        return filepath

    print(f"[*] Downloading ConceptNet from {CONCEPTNET_URL}...")
    print(f"    This is ~550 MB, may take a while...")

    def progress_hook(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r    [{pct:5.1f}%] {mb:.0f} / {total_mb:.0f} MB", end='', flush=True)

    urllib.request.urlretrieve(CONCEPTNET_URL, filepath, reporthook=progress_hook)
    print(f"\n[+] Downloaded to {filepath}")
    return filepath


def parse_conceptnet(filepath, limit=0, languages=None):
    """
    Parse ConceptNet CSV/TSV assertions file.

    Each line: URI\trelation\tstart\tend\tjson_metadata

    Returns list of (relation, subj_lang, subj, obj_lang, obj, weight, source_count) tuples.
    """
    if languages is None:
        languages = LANGUAGES

    assertions = []
    skipped = 0
    total_lines = 0

    opener = gzip.open if filepath.endswith('.gz') else open

    print(f"[*] Parsing ConceptNet from {filepath}...")
    print(f"    Languages: {languages}")
    print(f"    Relations: {len(ALL_RELATIONS)} types")
    if limit > 0:
        print(f"    Limit: {limit:,} assertions")

    with opener(filepath, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            total_lines += 1
            if total_lines % 1_000_000 == 0:
                print(f"    Scanned {total_lines:,} lines, kept {len(assertions):,}...", end='\r')

            parts = line.strip().split('\t')
            if len(parts) < 5:
                skipped += 1
                continue

            _uri, rel_uri, start_uri, end_uri, meta_json = parts[:5]

            # Parse relation
            relation = parse_relation(rel_uri)
            if relation not in ALL_RELATIONS:
                skipped += 1
                continue

            # Parse subject and object
            subj_lang, subj = clean_concept(start_uri)
            obj_lang, obj = clean_concept(end_uri)

            if subj is None or obj is None:
                skipped += 1
                continue

            # Language filter: at least one side must be in our languages
            if subj_lang not in languages and obj_lang not in languages:
                skipped += 1
                continue

            # Parse weight from metadata
            weight = 1.0
            try:
                meta = json.loads(meta_json)
                weight = meta.get('weight', 1.0)
            except (json.JSONDecodeError, ValueError):
                pass

            assertions.append((
                relation, subj_lang, subj, obj_lang, obj,
                weight, subj_lang in languages and obj_lang in languages
            ))

            if limit > 0 and len(assertions) >= limit:
                break

    print(f"\n[+] Parsed {len(assertions):,} assertions from {total_lines:,} lines (skipped {skipped:,})")
    return assertions


# ═══════════════════════════════════════════════════════════════════════════
# INGESTION
# ═══════════════════════════════════════════════════════════════════════════

def make_edge_request(pb2, **kwargs):
    """Helper: create InsertEdgeRequest (handles 'from' reserved word)."""
    from_val = kwargs.pop('from_node')
    req = pb2.InsertEdgeRequest(**kwargs)
    setattr(req, 'from', from_val)
    return req


def ingest(host, collection, metric="poincare", dim=128,
           limit=0, data_dir=None, ssl_cert=None, use_ssl=False,
           batch_size=100, min_weight=0.5):
    """
    Streaming ingestion pipeline (memory-efficient):
    1. Download ConceptNet (if needed)
    2. Stream-parse, build node/edge batches, insert on the fly
    3. Only keeps concept_ids dict in memory (small: just uuid strings)
    """
    import time as _time

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
    print(f"  ConceptNet 5.7/5.8 → NietzscheDB Streaming Ingestion", flush=True)
    print(f"  Host: {host} | Collection: {collection}", flush=True)
    print(f"  Dim: {dim} | Metric: {metric} | Min weight: {min_weight}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Ensure collection exists
    try:
        stub.CreateCollection(pb2.CreateCollectionRequest(
            collection=collection, dim=dim, metric=metric))
        print(f"[+] Collection '{collection}' created (or already exists)", flush=True)
    except grpc.RpcError as e:
        print(f"[!] CreateCollection: {e.details() if hasattr(e, 'details') else e}", flush=True)

    # Download
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    filepath = download_conceptnet(data_dir)

    # ─── Streaming parse + insert ────────────────────────────────────────
    opener = gzip.open if filepath.endswith('.gz') else open

    concept_ids = {}     # (lang, name) → uuid  (lightweight: ~50 bytes per entry)
    node_batch = []      # pending nodes to insert
    edge_queue = []      # pending edges to insert
    rel_counts = defaultdict(int)

    inserted_nodes = 0
    inserted_edges = 0
    total_kept = 0
    total_lines = 0
    skipped = 0
    t0 = _time.time()

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
            # Respect backpressure
            if hasattr(resp, 'backpressure') and resp.backpressure.suggested_delay_ms > 0:
                _time.sleep(resp.backpressure.suggested_delay_ms / 1000.0)
        except grpc.RpcError as e:
            print(f"\n  [!] Node batch error: {e.details() if hasattr(e, 'details') else e}", flush=True)
        node_batch.clear()

    def flush_edges():
        nonlocal inserted_edges
        if not edge_queue:
            return
        reqs = []
        for eb in edge_queue:
            req = make_edge_request(pb2,
                id=str(uuid.uuid4()),
                from_node=eb["from"], to=eb["to"],
                edge_type=eb["type"], weight=eb["weight"],
                collection=collection)
            reqs.append(req)
        try:
            stub.BatchInsertEdges(pb2.BatchInsertEdgesRequest(
                edges=reqs, collection=collection))
            inserted_edges += len(edge_queue)
        except grpc.RpcError as e:
            print(f"\n  [!] Edge batch error: {e.details() if hasattr(e, 'details') else e}", flush=True)
        edge_queue.clear()

    print(f"[*] Streaming parse + insert from {os.path.basename(filepath)}...", flush=True)
    print(f"    Languages: {LANGUAGES} | Relations: {len(ALL_RELATIONS)}", flush=True)
    if limit > 0:
        print(f"    Limit: {limit:,} assertions", flush=True)

    with opener(filepath, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            total_lines += 1

            parts = line.strip().split('\t')
            if len(parts) < 5:
                skipped += 1
                continue

            _uri, rel_uri, start_uri, end_uri, meta_json = parts[:5]

            relation = parse_relation(rel_uri)
            if relation not in ALL_RELATIONS:
                skipped += 1
                continue

            subj_lang, subj = clean_concept(start_uri)
            obj_lang, obj = clean_concept(end_uri)
            if subj is None or obj is None:
                skipped += 1
                continue

            if subj_lang not in LANGUAGES and obj_lang not in LANGUAGES:
                skipped += 1
                continue

            # Parse weight
            weight = 1.0
            try:
                meta = json.loads(meta_json)
                weight = meta.get('weight', 1.0)
            except (json.JSONDecodeError, ValueError):
                pass

            if weight < min_weight:
                skipped += 1
                continue

            total_kept += 1
            rel_counts[relation] += 1

            # Create/lookup subject node
            subj_key = (subj_lang, subj)
            if subj_key not in concept_ids:
                nid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"conceptnet:{subj_lang}:{subj}"))
                concept_ids[subj_key] = nid
                node_batch.append({
                    "id": nid,
                    "content": json.dumps({
                        "name": subj, "lang": subj_lang,
                        "galaxy": "Causal Physics", "type": "concept",
                        "source": "conceptnet",
                    }).encode('utf-8'),
                    "node_type": "Semantic",
                    "energy": min(0.9, 0.5 + weight * 0.1),
                    "embedding": make_poincare_embedding(subj, relation, dim),
                })

            # Create/lookup object node
            obj_key = (obj_lang, obj)
            if obj_key not in concept_ids:
                nid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"conceptnet:{obj_lang}:{obj}"))
                concept_ids[obj_key] = nid
                node_batch.append({
                    "id": nid,
                    "content": json.dumps({
                        "name": obj, "lang": obj_lang,
                        "galaxy": "Causal Physics", "type": "concept",
                        "source": "conceptnet",
                    }).encode('utf-8'),
                    "node_type": "Semantic",
                    "energy": min(0.9, 0.5 + weight * 0.1),
                    "embedding": make_poincare_embedding(obj, relation, dim),
                })

            # Queue edge
            edge_type = EDGE_TYPE_MAP.get(relation, "Association")
            edge_weight = WEIGHT_MAP.get(relation, 0.5) * min(1.0, weight)
            edge_queue.append({
                "from": concept_ids[subj_key],
                "to": concept_ids[obj_key],
                "type": edge_type,
                "weight": edge_weight,
            })

            # Flush when batch is full
            if len(node_batch) >= batch_size:
                flush_nodes()
            if len(edge_queue) >= batch_size:
                flush_edges()

            # Progress every 10K assertions
            if total_kept % 10000 == 0:
                elapsed = _time.time() - t0
                rate = total_kept / elapsed if elapsed > 0 else 0
                print(f"  [{total_kept:>9,} kept] nodes={inserted_nodes:,} edges={inserted_edges:,} "
                      f"scanned={total_lines:,} ({rate:.0f}/s)", flush=True)

            if limit > 0 and total_kept >= limit:
                break

    # Final flush
    flush_nodes()
    flush_edges()

    elapsed = _time.time() - t0

    # ─── Summary ─────────────────────────────────────────────────────────
    n_causal = sum(rel_counts[r] for r in CAUSAL_RELATIONS if r in rel_counts)
    n_property = sum(rel_counts[r] for r in PROPERTY_RELATIONS if r in rel_counts)
    n_hierarchy = sum(rel_counts[r] for r in HIERARCHY_RELATIONS if r in rel_counts)
    n_social = sum(rel_counts[r] for r in SOCIAL_RELATIONS if r in rel_counts)

    print(f"\n{'='*70}", flush=True)
    print(f"  ConceptNet Ingestion Complete!  ({elapsed:.1f}s)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Collection:    {collection}", flush=True)
    print(f"  Lines scanned: {total_lines:,}", flush=True)
    print(f"  Assertions:    {total_kept:,} (skipped {skipped:,})", flush=True)
    print(f"  Unique nodes:  {inserted_nodes:,}", flush=True)
    print(f"  Total edges:   {inserted_edges:,}", flush=True)
    print(f"  ─────────────────────────────────", flush=True)
    print(f"  Causal edges:    {n_causal:>8,}  (Causes, HasPrerequisite, ...)", flush=True)
    print(f"  Property edges:  {n_property:>8,}  (HasProperty, CapableOf, ...)", flush=True)
    print(f"  Hierarchy edges: {n_hierarchy:>8,}  (IsA, PartOf, ...)", flush=True)
    print(f"  Social edges:    {n_social:>8,}  (RelatedTo, Desires, ...)", flush=True)
    print(f"  ─────────────────────────────────", flush=True)

    print(f"\n    Relation breakdown:", flush=True)
    for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        cat = get_relation_category(rel)
        marker = {"causal": "C", "property": "P", "hierarchy": "H", "social": "S"}[cat]
        print(f"      [{marker}] {rel:25s} {count:>8,}", flush=True)

    print(f"\n  Poincare depth encoding:", flush=True)
    print(f"    Hierarchy nodes: ~0.15-0.20 (near center, abstract)", flush=True)
    print(f"    Property nodes:  ~0.28-0.35 (mid-level)", flush=True)
    print(f"    Causal nodes:    ~0.42-0.50 (outer, specific)", flush=True)
    print(f"{'='*70}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ingest ConceptNet 5.7/5.8 into NietzscheDB")
    parser.add_argument('--host', default='localhost:50051',
                        help='gRPC host:port (default: localhost:50051)')
    parser.add_argument('--collection', default='science_galaxies',
                        help='Target collection (default: science_galaxies)')
    parser.add_argument('--metric', default='poincare',
                        help='Distance metric (default: poincare)')
    parser.add_argument('--dim', type=int, default=128,
                        help='Embedding dimension (default: 128)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max assertions to import (0=all, ~34M total)')
    parser.add_argument('--min-weight', type=float, default=1.0,
                        help='Min ConceptNet weight to include (default: 1.0)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='gRPC batch size (default: 100)')
    parser.add_argument('--data-dir', default=None,
                        help='Directory to store downloaded data')
    parser.add_argument('--ssl', action='store_true',
                        help='Use SSL/TLS for gRPC connection')
    parser.add_argument('--cert', default=None,
                        help='Path to SSL certificate (for self-signed certs)')
    args = parser.parse_args()

    ingest(
        host=args.host,
        collection=args.collection,
        metric=args.metric,
        dim=args.dim,
        limit=args.limit,
        data_dir=args.data_dir,
        ssl_cert=args.cert,
        use_ssl=args.ssl,
        batch_size=args.batch_size,
        min_weight=args.min_weight,
    )


if __name__ == '__main__':
    main()
