#!/usr/bin/env python3
"""
Link Malaria-Vision Nodes — Creates diagnostic edges in the malaria-vision collection.

Connects 3,564+ slides that currently have 0 edges by grouping them by species
and parasite stage, then creating three types of edges:

  - SIMILAR_DIAGNOSIS: same species + same stage (weight=1.0)
  - SAME_SPECIES:      same species, different stage (weight=0.7)
  - SAME_STAGE:        same stage, different species (weight=0.5)

To avoid combinatorial explosion (N*(N-1)/2 edges per group), each node is
connected to at most MAX_PEERS_PER_NODE random peers within each edge type.
Negative slides (is_negative=true) are linked among themselves as SIMILAR_DIAGNOSIS.

Usage:
    python scripts/link_malaria_vision.py                # execute
    python scripts/link_malaria_vision.py --dry-run      # preview only
    python scripts/link_malaria_vision.py --max-peers 10  # more connections per node
"""

import sys
import os
import json
import uuid
import random
import time
import argparse
from collections import defaultdict
from datetime import datetime

# Add SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))

import grpc
from nietzschedb.proto import nietzsche_pb2 as pb, nietzsche_pb2_grpc as rpc

# --- Config ---
CERT_PATH = os.path.expanduser('~/AppData/Local/Temp/eva-cert.pem')
HOST = '136.111.0.47:443'
COLLECTION = 'malaria-vision'
BATCH_SIZE = 100  # edges per batch insert


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def get_stub():
    """Create a gRPC stub with TLS to the remote NietzscheDB."""
    with open(CERT_PATH, 'rb') as f:
        cert = f.read()
    creds = grpc.ssl_channel_credentials(root_certificates=cert)
    channel = grpc.secure_channel(HOST, creds, options=[
        ('grpc.max_receive_message_length', 256 * 1024 * 1024),
        ('grpc.max_send_message_length', 64 * 1024 * 1024),
        ('grpc.keepalive_time_ms', 30000),
        ('grpc.keepalive_timeout_ms', 10000),
    ])
    return rpc.NietzscheDBStub(channel), channel


def fetch_all_nodes(stub):
    """Fetch all nodes from malaria-vision using NQL pagination.

    NQL MATCH returns all nodes mapped to Semantic type.
    We paginate by fetching batches with LIMIT + OFFSET via repeated queries.
    """
    log(f"Fetching all nodes from '{COLLECTION}'...")
    all_nodes = []
    offset = 0
    page_size = 500

    while True:
        nql = f"MATCH (n) RETURN n LIMIT {page_size} OFFSET {offset}"
        try:
            resp = stub.Query(pb.QueryRequest(
                nql=nql,
                collection=COLLECTION,
            ), timeout=120)
        except grpc.RpcError as e:
            log(f"  NQL query failed at offset {offset}: {e}")
            break

        if resp.error:
            log(f"  NQL error: {resp.error}")
            break

        batch = []
        for n in resp.nodes:
            content = {}
            if n.content:
                try:
                    content = json.loads(n.content)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            batch.append({
                'id': n.id,
                'content': content,
            })

        all_nodes.extend(batch)
        log(f"  Fetched {len(all_nodes)} nodes so far (batch={len(batch)})...")

        if len(batch) < page_size:
            break
        offset += page_size

    log(f"Total nodes fetched: {len(all_nodes)}")
    return all_nodes


def classify_nodes(nodes):
    """Group nodes by species and stage for edge creation.

    Returns:
        by_species_stage: dict[(species, stage)] -> [node_ids]
        by_species:       dict[species] -> [node_ids]
        by_stage:         dict[stage] -> [node_ids]
        negatives:        [node_ids]  (is_negative=true or no species/stage)
    """
    by_species_stage = defaultdict(list)
    by_species = defaultdict(list)
    by_stage = defaultdict(list)
    negatives = []

    for node in nodes:
        c = node['content']
        nid = node['id']

        is_negative = c.get('is_negative', False)
        species = c.get('species', '')
        stage = c.get('stage', '')

        if is_negative or (not species and not stage):
            negatives.append(nid)
            continue

        if species and stage:
            by_species_stage[(species, stage)].append(nid)
        if species:
            by_species[species].append(nid)
        if stage:
            by_stage[stage].append(nid)

    return by_species_stage, by_species, by_stage, negatives


def select_peers(node_ids, max_peers):
    """For each node in the list, select up to max_peers random peers.

    Returns a set of (from_id, to_id) tuples with from_id < to_id
    to avoid duplicate edges.
    """
    edges = set()
    if len(node_ids) <= 1:
        return edges

    for i, nid in enumerate(node_ids):
        # Select peers from the rest of the list
        candidates = node_ids[:i] + node_ids[i+1:]
        k = min(max_peers, len(candidates))
        peers = random.sample(candidates, k)
        for peer in peers:
            edge = tuple(sorted([nid, peer]))
            edges.add(edge)

    return edges


def batch_insert_edges(stub, edge_list, edge_type, weight, dry_run):
    """Insert edges in batches. Returns count of edges inserted."""
    if not edge_list:
        return 0

    total = len(edge_list)
    inserted = 0
    errors = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = edge_list[batch_start:batch_start + BATCH_SIZE]

        if dry_run:
            inserted += len(batch)
            continue

        pb_edges = []
        for from_id, to_id in batch:
            edge_id = str(uuid.uuid4())
            pb_edges.append(pb.InsertEdgeRequest(
                id=edge_id,
                edge_type=edge_type,
                weight=weight,
                collection=COLLECTION,
                **{"from": from_id, "to": to_id},
            ))

        try:
            resp = stub.BatchInsertEdges(pb.BatchInsertEdgesRequest(
                edges=pb_edges,
                collection=COLLECTION,
            ), timeout=120)
            inserted += resp.inserted
        except grpc.RpcError as e:
            log(f"  Batch insert error ({edge_type}): {e}")
            errors += len(batch)
            # Fallback: try one-by-one for this batch
            for from_id, to_id in batch:
                try:
                    edge_id = str(uuid.uuid4())
                    stub.InsertEdge(pb.InsertEdgeRequest(
                        id=edge_id,
                        edge_type=edge_type,
                        weight=weight,
                        collection=COLLECTION,
                        **{"from": from_id, "to": to_id},
                    ), timeout=30)
                    inserted += 1
                    errors -= 1
                except grpc.RpcError:
                    pass

        if (batch_start + BATCH_SIZE) % 500 == 0 and batch_start > 0:
            log(f"  {edge_type}: {inserted}/{total} inserted...")

    if errors > 0:
        log(f"  {edge_type}: {errors} errors during insertion")

    return inserted


def main():
    parser = argparse.ArgumentParser(description='Link malaria-vision slides by species and stage')
    parser.add_argument('--dry-run', action='store_true', help='Preview only, do not insert edges')
    parser.add_argument('--max-peers', type=int, default=5,
                        help='Max peers per node per edge type (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)
    max_peers = args.max_peers
    dry_run = args.dry_run

    if dry_run:
        log("=== DRY RUN MODE — no edges will be created ===")

    log(f"Connecting to {HOST} (collection={COLLECTION})...")
    stub, channel = get_stub()

    # Verify connection
    try:
        resp = stub.ListCollections(pb.Empty(), timeout=30)
        found = False
        for c in resp.collections:
            if c.collection == COLLECTION:
                log(f"Collection '{COLLECTION}': {c.node_count} nodes, {c.edge_count} edges, "
                    f"dim={c.dim}, metric={c.metric}")
                found = True
                break
        if not found:
            log(f"ERROR: Collection '{COLLECTION}' not found!")
            sys.exit(1)
    except grpc.RpcError as e:
        log(f"ERROR: Cannot connect to NietzscheDB: {e}")
        sys.exit(1)

    # Fetch all nodes
    nodes = fetch_all_nodes(stub)
    if not nodes:
        log("No nodes found. Exiting.")
        sys.exit(0)

    # Classify
    by_species_stage, by_species, by_stage, negatives = classify_nodes(nodes)

    log(f"\n{'='*60}")
    log("CLASSIFICATION SUMMARY")
    log(f"{'='*60}")
    log(f"Total nodes: {len(nodes)}")
    log(f"Negative slides: {len(negatives)}")
    log(f"\nBy species+stage:")
    for (sp, st), ids in sorted(by_species_stage.items()):
        log(f"  {sp} / {st}: {len(ids)} nodes")
    log(f"\nBy species:")
    for sp, ids in sorted(by_species.items()):
        log(f"  {sp}: {len(ids)} nodes")
    log(f"\nBy stage:")
    for st, ids in sorted(by_stage.items()):
        log(f"  {st}: {len(ids)} nodes")

    # --- Build edge sets ---
    stats = {
        'SIMILAR_DIAGNOSIS': 0,
        'SAME_SPECIES': 0,
        'SAME_STAGE': 0,
    }

    # 1. SIMILAR_DIAGNOSIS: same species + same stage (weight=1.0)
    log(f"\n{'='*60}")
    log("EDGE TYPE: SIMILAR_DIAGNOSIS (same species + stage, weight=1.0)")
    log(f"{'='*60}")

    similar_edges = set()
    for (sp, st), ids in by_species_stage.items():
        pairs = select_peers(ids, max_peers)
        log(f"  {sp} / {st}: {len(ids)} nodes -> {len(pairs)} edges")
        similar_edges.update(pairs)

    # Also link negatives among themselves
    if negatives:
        neg_pairs = select_peers(negatives, max_peers)
        log(f"  Negatives: {len(negatives)} nodes -> {len(neg_pairs)} edges")
        similar_edges.update(neg_pairs)

    similar_list = list(similar_edges)
    log(f"  Total SIMILAR_DIAGNOSIS edges: {len(similar_list)}")
    count = batch_insert_edges(stub, similar_list, 'SIMILAR_DIAGNOSIS', 1.0, dry_run)
    stats['SIMILAR_DIAGNOSIS'] = count

    # 2. SAME_SPECIES: same species, different stage (weight=0.7)
    # For each species, connect nodes across different stages
    log(f"\n{'='*60}")
    log("EDGE TYPE: SAME_SPECIES (same species, different stage, weight=0.7)")
    log(f"{'='*60}")

    species_edges = set()
    species_groups = defaultdict(dict)  # species -> {stage -> [ids]}
    for (sp, st), ids in by_species_stage.items():
        species_groups[sp][st] = ids

    for sp, stage_dict in species_groups.items():
        stages = list(stage_dict.keys())
        if len(stages) < 2:
            log(f"  {sp}: only 1 stage, skipping cross-stage edges")
            continue
        for i, st_a in enumerate(stages):
            for st_b in stages[i+1:]:
                ids_a = stage_dict[st_a]
                ids_b = stage_dict[st_b]
                # Sample cross-stage connections
                for nid in ids_a:
                    k = min(max_peers, len(ids_b))
                    peers = random.sample(ids_b, k)
                    for peer in peers:
                        edge = tuple(sorted([nid, peer]))
                        species_edges.add(edge)
                log(f"  {sp}: {st_a} <-> {st_b}: "
                    f"{len(ids_a)}x{len(ids_b)} nodes")

    # Remove edges already in similar_edges (same species+stage already covered)
    species_edges -= similar_edges
    species_list = list(species_edges)
    log(f"  Total SAME_SPECIES edges (deduplicated): {len(species_list)}")
    count = batch_insert_edges(stub, species_list, 'SAME_SPECIES', 0.7, dry_run)
    stats['SAME_SPECIES'] = count

    # 3. SAME_STAGE: same stage, different species (weight=0.5)
    log(f"\n{'='*60}")
    log("EDGE TYPE: SAME_STAGE (same stage, different species, weight=0.5)")
    log(f"{'='*60}")

    stage_edges = set()
    stage_groups = defaultdict(dict)  # stage -> {species -> [ids]}
    for (sp, st), ids in by_species_stage.items():
        stage_groups[st][sp] = ids

    for st, species_dict in stage_groups.items():
        species_list_local = list(species_dict.keys())
        if len(species_list_local) < 2:
            log(f"  {st}: only 1 species, skipping cross-species edges")
            continue
        for i, sp_a in enumerate(species_list_local):
            for sp_b in species_list_local[i+1:]:
                ids_a = species_dict[sp_a]
                ids_b = species_dict[sp_b]
                for nid in ids_a:
                    k = min(max_peers, len(ids_b))
                    peers = random.sample(ids_b, k)
                    for peer in peers:
                        edge = tuple(sorted([nid, peer]))
                        stage_edges.add(edge)
                log(f"  {st}: {sp_a} <-> {sp_b}: "
                    f"{len(ids_a)}x{len(ids_b)} nodes")

    # Remove already-created edges
    stage_edges -= similar_edges
    stage_edges -= species_edges
    stage_list = list(stage_edges)
    log(f"  Total SAME_STAGE edges (deduplicated): {len(stage_list)}")
    count = batch_insert_edges(stub, stage_list, 'SAME_STAGE', 0.5, dry_run)
    stats['SAME_STAGE'] = count

    # --- Summary ---
    total_edges = sum(stats.values())
    log(f"\n{'='*60}")
    log("FINAL SUMMARY")
    log(f"{'='*60}")
    log(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    log(f"Max peers per node: {max_peers}")
    log(f"Nodes processed: {len(nodes)}")
    log(f"")
    log(f"Edges created by type:")
    for etype, count in stats.items():
        log(f"  {etype}: {count}")
    log(f"  -------------------------")
    log(f"  TOTAL: {total_edges}")

    if dry_run:
        log(f"\nRe-run without --dry-run to create these edges.")

    channel.close()


if __name__ == '__main__':
    main()
