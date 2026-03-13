#!/usr/bin/env python3
"""
cleanup_spam.py -- Delete spam nodes from eva_core and patient_graph

Targets:
  - Demand nodes with desire="indefinido" (empty/junk desires)
  - EvaSession nodes with turn_count=0 (abandoned sessions)

Uses NQL to find low-energy Semantic nodes, then filters by content.

Usage:
  python cleanup_spam.py                  # LIVE mode
  python cleanup_spam.py --dry-run        # preview only
  python cleanup_spam.py --limit 500      # process more per batch
  python cleanup_spam.py --collection eva_core  # single collection
"""
import sys
import os
import time
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))

import grpc
from nietzschedb.proto import nietzsche_pb2 as pb, nietzsche_pb2_grpc as rpc

# --- Config ---
CERT_PATH = os.path.expanduser('~/AppData/Local/Temp/eva-cert.pem')
HOST = '136.111.0.47:443'
TARGET_COLLECTIONS = ['eva_core', 'patient_graph']


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def get_stub():
    with open(CERT_PATH, 'rb') as f:
        cert = f.read()
    creds = grpc.ssl_channel_credentials(root_certificates=cert)
    channel = grpc.secure_channel(HOST, creds, options=[
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ('grpc.keepalive_time_ms', 30000),
    ])
    return rpc.NietzscheDBStub(channel), channel


def is_spam_demand(content):
    """Demand with desire='indefinido' or empty desire."""
    if content.get('node_label') != 'Demand':
        return False
    desire = content.get('desire', '')
    return desire in ('indefinido', '', None)


def is_empty_session(content):
    """EvaSession with turn_count=0 (abandoned session)."""
    if content.get('node_label') != 'EvaSession':
        return False
    turn_count = content.get('turn_count', -1)
    # Accept int or string "0"
    try:
        return int(turn_count) == 0
    except (ValueError, TypeError):
        return False


def scan_and_delete(stub, collection, batch_limit, dry_run):
    """Scan low-energy Semantic nodes and delete spam matches.

    Returns (demands_deleted, sessions_deleted, errors).
    """
    demands_deleted = 0
    sessions_deleted = 0
    errors = 0
    offset = 0
    total_scanned = 0

    # We scan in rounds -- NQL LIMIT caps at batch_limit per query
    # Use progressively higher energy thresholds to catch more spam
    energy_thresholds = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]

    seen_ids = set()

    for energy_max in energy_thresholds:
        # NQL: energy = X finds exact match; use energy < X for ranges
        if energy_max == 0.0:
            nql = f'MATCH (n:Semantic) WHERE n.energy = 0 RETURN n LIMIT {batch_limit}'
        elif energy_max < 1.0:
            nql = f'MATCH (n:Semantic) WHERE n.energy < {energy_max} RETURN n LIMIT {batch_limit}'
        else:
            # Final pass: scan all Semantic nodes
            nql = f'MATCH (n:Semantic) RETURN n LIMIT {batch_limit}'

        log(f"  Scanning: {nql}")

        try:
            resp = stub.Query(pb.QueryRequest(
                nql=nql,
                collection=collection,
            ), timeout=120)
        except grpc.RpcError as e:
            log(f"  ERROR querying: {e.code()} {e.details()}")
            errors += 1
            continue

        if not resp.nodes:
            log(f"  No nodes found at energy<{energy_max}")
            continue

        batch_demands = 0
        batch_sessions = 0

        for node in resp.nodes:
            if node.id in seen_ids:
                continue
            seen_ids.add(node.id)
            total_scanned += 1

            try:
                content = json.loads(node.content) if node.content else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            should_delete = False
            reason = ""

            if is_spam_demand(content):
                should_delete = True
                reason = f"Demand desire='{content.get('desire', '')}'"
                batch_demands += 1
            elif is_empty_session(content):
                should_delete = True
                reason = f"EvaSession turn_count=0"
                batch_sessions += 1

            if should_delete:
                if dry_run:
                    if total_scanned <= 5 or (batch_demands + batch_sessions) <= 10:
                        log(f"    [DRY-RUN] Would delete {node.id[:12]}... ({reason}, energy={node.energy:.4f})")
                else:
                    try:
                        stub.DeleteNode(pb.NodeIdRequest(
                            id=node.id,
                            collection=collection,
                        ), timeout=10)
                    except grpc.RpcError as e:
                        log(f"    ERROR deleting {node.id[:12]}...: {e.code()}")
                        errors += 1

        demands_deleted += batch_demands
        sessions_deleted += batch_sessions
        log(f"  Batch result: {batch_demands} Demands + {batch_sessions} Sessions found "
            f"(scanned {len(resp.nodes)} nodes)")

    return demands_deleted, sessions_deleted, errors, total_scanned


def main():
    parser = argparse.ArgumentParser(description='Delete spam Demand and EvaSession nodes')
    parser.add_argument('--dry-run', action='store_true', help='Preview only, do not delete')
    parser.add_argument('--limit', type=int, default=200, help='Batch size per NQL query (default: 200)')
    parser.add_argument('--collection', type=str, help='Single collection to clean (default: all targets)')
    args = parser.parse_args()

    collections = [args.collection] if args.collection else TARGET_COLLECTIONS
    mode = "DRY-RUN" if args.dry_run else "LIVE"

    log(f"=== NietzscheDB Spam Cleanup ({mode}) ===")
    log(f"Host: {HOST}")
    log(f"Collections: {collections}")
    log(f"Batch limit: {args.limit}")
    log("")

    # Test connectivity
    stub, channel = get_stub()
    try:
        resp = stub.HealthCheck(pb.Empty(), timeout=10)
        log(f"Server health: {resp.status}")
    except grpc.RpcError as e:
        log(f"FATAL: Cannot connect to server: {e.code()} {e.details()}")
        sys.exit(1)

    start = time.time()
    total_demands = 0
    total_sessions = 0
    total_errors = 0
    total_scanned = 0

    for collection in collections:
        log(f"\n{'='*50}")
        log(f"Collection: {collection}")
        log(f"{'='*50}")

        demands, sessions, errors, scanned = scan_and_delete(
            stub, collection, args.limit, args.dry_run
        )
        total_demands += demands
        total_sessions += sessions
        total_errors += errors
        total_scanned += scanned

        log(f"\n  Summary for {collection}:")
        log(f"    Scanned: {scanned}")
        log(f"    Demands (desire=indefinido): {demands}")
        log(f"    Sessions (turn_count=0):     {sessions}")
        log(f"    Errors: {errors}")

    elapsed = time.time() - start
    channel.close()

    log(f"\n{'='*50}")
    log(f"FINAL RESULTS ({mode})")
    log(f"{'='*50}")
    log(f"  Total scanned:  {total_scanned}")
    log(f"  Demands deleted: {total_demands}")
    log(f"  Sessions deleted: {total_sessions}")
    log(f"  Total deleted:   {total_demands + total_sessions}")
    log(f"  Errors:          {total_errors}")
    log(f"  Duration:        {elapsed:.1f}s")

    if args.dry_run:
        log(f"\n  Run without --dry-run to actually delete these nodes.")


if __name__ == '__main__':
    main()
