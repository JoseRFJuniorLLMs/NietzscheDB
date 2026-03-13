#!/usr/bin/env python3
"""
cleanup_orphans.py -- Delete orphan nodes (no edges) from a collection

Uses WCC (Weakly Connected Components) to find isolated nodes (component size=1),
then deletes them in batches.

For `default` collection: ~315K nodes with only ~21K edges means ~283K orphans.

Usage:
  python cleanup_orphans.py --dry-run                     # preview
  python cleanup_orphans.py --collection default           # LIVE on default
  python cleanup_orphans.py --collection default --batch 200  # bigger batches
  python cleanup_orphans.py --collection eva_core --dry-run   # check eva_core
"""
import sys
import os
import time
import json
import argparse
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))

import grpc
from nietzschedb.proto import nietzsche_pb2 as pb, nietzsche_pb2_grpc as rpc

# --- Config ---
CERT_PATH = os.path.expanduser('~/AppData/Local/Temp/eva-cert.pem')
HOST = '136.111.0.47:443'

# Estimated bytes per node (content + embedding + metadata overhead)
# Conservative estimate based on typical NietzscheDB node size
EST_BYTES_PER_NODE = 512


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def get_stub():
    with open(CERT_PATH, 'rb') as f:
        cert = f.read()
    creds = grpc.ssl_channel_credentials(root_certificates=cert)
    channel = grpc.secure_channel(HOST, creds, options=[
        ('grpc.max_receive_message_length', 256 * 1024 * 1024),
        ('grpc.keepalive_time_ms', 30000),
    ])
    return rpc.NietzscheDBStub(channel), channel


def get_collection_info(stub, collection):
    """Get node_count and edge_count for a collection."""
    try:
        resp = stub.ListCollections(pb.Empty(), timeout=30)
        for c in resp.collections:
            if c.collection == collection:
                return c.node_count, c.edge_count
    except grpc.RpcError as e:
        log(f"  ERROR listing collections: {e.code()}")
    return 0, 0


def find_orphans_via_wcc(stub, collection):
    """Use WCC to find all nodes in singleton components (no edges).

    Returns list of orphan node IDs.
    """
    log(f"  Running WCC algorithm on `{collection}`...")
    log(f"  (This may take a while on large collections)")

    try:
        resp = stub.RunWCC(pb.WccRequest(collection=collection), timeout=600)
    except grpc.RpcError as e:
        log(f"  ERROR running WCC: {e.code()} {e.details()}")
        return []

    # Count component sizes
    component_members = {}
    for assignment in resp.assignments:
        cid = assignment.community_id
        nid = assignment.node_id
        component_members.setdefault(cid, []).append(nid)

    # Find singleton components (size == 1 means orphan)
    orphan_ids = []
    component_sizes = Counter()
    for cid, members in component_members.items():
        size = len(members)
        component_sizes[size] += 1
        if size == 1:
            orphan_ids.extend(members)

    log(f"  WCC results:")
    log(f"    Total components: {len(component_members)}")
    log(f"    Duration: {resp.duration_ms}ms")
    log(f"    Component size distribution:")
    for size in sorted(component_sizes.keys()):
        count = component_sizes[size]
        label = "ORPHANS" if size == 1 else ""
        log(f"      size={size}: {count} components {label}")

    return orphan_ids


def delete_batch(stub, collection, node_ids, dry_run):
    """Delete a batch of nodes. Returns (deleted, errors)."""
    deleted = 0
    errors = 0

    for nid in node_ids:
        if dry_run:
            deleted += 1
            continue
        try:
            stub.DeleteNode(pb.NodeIdRequest(
                id=nid,
                collection=collection,
            ), timeout=10)
            deleted += 1
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                # Already gone, count as success
                deleted += 1
            else:
                errors += 1

    return deleted, errors


def human_bytes(n):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description='Delete orphan nodes (no edges) from a collection')
    parser.add_argument('--dry-run', action='store_true', help='Preview only, do not delete')
    parser.add_argument('--collection', type=str, default='default',
                        help='Collection to clean (default: "default")')
    parser.add_argument('--batch', type=int, default=100, help='Delete batch size (default: 100)')
    parser.add_argument('--max-delete', type=int, default=0,
                        help='Maximum nodes to delete (0=unlimited)')
    args = parser.parse_args()

    mode = "DRY-RUN" if args.dry_run else "LIVE"

    log(f"=== NietzscheDB Orphan Cleanup ({mode}) ===")
    log(f"Host: {HOST}")
    log(f"Collection: {args.collection}")
    log(f"Batch size: {args.batch}")
    if args.max_delete > 0:
        log(f"Max delete: {args.max_delete}")
    log("")

    # Connect
    stub, channel = get_stub()
    try:
        resp = stub.HealthCheck(pb.Empty(), timeout=10)
        log(f"Server health: {resp.status}")
    except grpc.RpcError as e:
        log(f"FATAL: Cannot connect to server: {e.code()} {e.details()}")
        sys.exit(1)

    # Get collection info
    node_count, edge_count = get_collection_info(stub, args.collection)
    if node_count == 0:
        log(f"Collection `{args.collection}` not found or empty (node_count=0).")
        log(f"Note: node_count may show 0 if agency holds a lock (try_read).")
        log(f"Proceeding with WCC anyway...")

    log(f"Collection `{args.collection}`: {node_count} nodes, {edge_count} edges")
    if edge_count > 0 and node_count > 0:
        density = (2 * edge_count) / (node_count * (node_count - 1)) if node_count > 1 else 0
        log(f"  Edge density: {density:.8f}")
        estimated_orphans = max(0, node_count - 2 * edge_count)
        log(f"  Estimated orphans (rough): ~{estimated_orphans}")
    log("")

    start = time.time()

    # Find orphans via WCC
    orphan_ids = find_orphans_via_wcc(stub, args.collection)

    if not orphan_ids:
        log(f"\nNo orphan nodes found. Collection is fully connected or WCC failed.")
        channel.close()
        return

    orphan_count = len(orphan_ids)
    estimated_savings = orphan_count * EST_BYTES_PER_NODE

    log(f"\nFound {orphan_count} orphan nodes")
    log(f"Estimated disk savings: {human_bytes(estimated_savings)}")

    if args.max_delete > 0 and orphan_count > args.max_delete:
        orphan_ids = orphan_ids[:args.max_delete]
        log(f"Limiting to {args.max_delete} deletions (--max-delete)")

    # Delete in batches
    total_deleted = 0
    total_errors = 0
    to_delete = len(orphan_ids)

    log(f"\n{'Simulating' if args.dry_run else 'Deleting'} {to_delete} orphan nodes "
        f"in batches of {args.batch}...")

    for i in range(0, to_delete, args.batch):
        batch = orphan_ids[i:i + args.batch]
        batch_num = (i // args.batch) + 1
        total_batches = (to_delete + args.batch - 1) // args.batch

        deleted, errors = delete_batch(stub, args.collection, batch, args.dry_run)
        total_deleted += deleted
        total_errors += errors

        pct = (total_deleted / to_delete) * 100
        log(f"  Batch {batch_num}/{total_batches}: "
            f"{'would delete' if args.dry_run else 'deleted'} {deleted}, "
            f"errors {errors} "
            f"[{total_deleted}/{to_delete} = {pct:.0f}%]")

    elapsed = time.time() - start
    channel.close()

    log(f"\n{'='*50}")
    log(f"FINAL RESULTS ({mode})")
    log(f"{'='*50}")
    log(f"  Collection:       {args.collection}")
    log(f"  Orphans found:    {orphan_count}")
    log(f"  Nodes deleted:    {total_deleted}")
    log(f"  Errors:           {total_errors}")
    log(f"  Est. disk saved:  {human_bytes(total_deleted * EST_BYTES_PER_NODE)}")
    log(f"  Duration:         {elapsed:.1f}s")

    if args.dry_run:
        log(f"\n  Run without --dry-run to actually delete these nodes.")
    else:
        log(f"\n  Tip: Run `ReapExpired` or restart the server to reclaim disk space.")


if __name__ == '__main__':
    main()
