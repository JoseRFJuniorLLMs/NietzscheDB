#!/usr/bin/env python3
"""
NietzscheDB Cleanup Army -- 30 parallel workers
Cleans junk data identified in the 2026-03-13 audit.
"""
import sys, os, time, json, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdks', 'python'))

import grpc
from nietzschedb.proto import nietzsche_pb2 as pb, nietzsche_pb2_grpc as rpc

# --- Config ---
CERT_PATH = os.path.expanduser('~/AppData/Local/Temp/eva-cert.pem')
HOST = '136.111.0.47:443'
MAX_WORKERS = 30
DRY_RUN = '--dry-run' in sys.argv

# Thread-safe stats
_lock = threading.Lock()
stats = {
    'deleted_nodes': 0,
    'edges_created': 0,
    'dropped_collections': [],
    'reaped': [],
    'errors': 0,
}

def inc(key, val=1):
    with _lock:
        stats[key] += val

def append(key, val):
    with _lock:
        stats[key].append(val)

def get_stub():
    with open(CERT_PATH, 'rb') as f:
        cert = f.read()
    creds = grpc.ssl_channel_credentials(root_certificates=cert)
    channel = grpc.secure_channel(HOST, creds, options=[
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ])
    return rpc.NietzscheDBStub(channel)

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    safe = msg.encode('ascii', 'replace').decode()
    print(f"[{ts}] {safe}", flush=True)

# ===== TASK: Drop collection =====
def task_drop(name, reason):
    log(f"DROP: `{name}` ({reason})...")
    if DRY_RUN:
        log(f"  [DRY-RUN] Would drop `{name}`")
        return
    try:
        stub = get_stub()
        stub.DropCollection(pb.DropCollectionRequest(collection=name), timeout=120)
        append('dropped_collections', name)
        log(f"  OK: Dropped `{name}`")
    except Exception as e:
        log(f"  FAIL dropping {name}: {e}")
        inc('errors')

# ===== TASK: Clean Demand junk (energy=0, text="") =====
def task_clean_demands(collection):
    log(f"CLEAN: Demand junk in `{collection}`...")
    try:
        stub = get_stub()
        resp = stub.Query(pb.QueryRequest(
            nql='MATCH (n:Semantic) WHERE n.energy = 0 RETURN n LIMIT 500',
            collection=collection
        ), timeout=60)

        deleted = 0
        for node in resp.nodes:
            content = json.loads(node.content) if node.content else {}
            if content.get('node_label') == 'Demand' and content.get('text', '') == '':
                if DRY_RUN:
                    deleted += 1
                    continue
                try:
                    stub.DeleteNode(pb.NodeIdRequest(
                        id=node.id,
                        collection=collection
                    ), timeout=10)
                    deleted += 1
                    inc('deleted_nodes')
                except:
                    inc('errors')

        log(f"  OK: {'Would delete' if DRY_RUN else 'Deleted'} {deleted} Demand junk from `{collection}`")
    except Exception as e:
        log(f"  FAIL: {e}")
        inc('errors')

# ===== TASK: Reap expired =====
def task_reap(collection):
    log(f"REAP: expired in `{collection}`...")
    if DRY_RUN:
        log(f"  [DRY-RUN] Would reap `{collection}`")
        return
    try:
        stub = get_stub()
        resp = stub.ReapExpired(pb.ReapExpiredRequest(collection=collection), timeout=120)
        count = resp.reaped_count if hasattr(resp, 'reaped_count') else 0
        append('reaped', (collection, count))
        log(f"  OK: Reaped {count} from `{collection}`")
    except Exception as e:
        log(f"  FAIL reaping {collection}: {e}")
        inc('errors')

# ===== TASK: Connect malaria-vision slides =====
def task_connect_malaria_vision():
    log("CONNECT: Building edges in `malaria-vision`...")
    try:
        stub = get_stub()
        resp = stub.Query(pb.QueryRequest(
            nql='MATCH (n:Semantic) RETURN n LIMIT 500',
            collection='malaria-vision'
        ), timeout=60)

        # Group by diagnosis
        groups = {}
        for node in resp.nodes:
            content = json.loads(node.content) if node.content else {}
            level = content.get('infection_level', '')
            stage = content.get('dominant_stage', '')
            key = f"{level}|{stage}"
            if level:
                groups.setdefault(key, []).append(node.id)

        edges = 0
        for key, ids in groups.items():
            for i in range(min(len(ids) - 1, 10)):
                if DRY_RUN:
                    edges += 1
                    continue
                try:
                    stub.InsertEdge(pb.InsertEdgeRequest(
                        from_id=ids[i],
                        to_id=ids[i + 1],
                        edge_type='Association',
                        weight=0.8,
                        collection='malaria-vision'
                    ), timeout=10)
                    edges += 1
                    inc('edges_created')
                except:
                    inc('errors')

        log(f"  OK: {'Would create' if DRY_RUN else 'Created'} {edges} edges in malaria-vision")
    except Exception as e:
        log(f"  FAIL: {e}")
        inc('errors')

# ===== MAIN =====
def main():
    mode = "DRY-RUN" if DRY_RUN else "LIVE"
    log(f"=== NietzscheDB Cleanup Army -- {MAX_WORKERS} workers ({mode}) ===")
    start = time.time()

    # Collections to DROP (total junk)
    drops = [
        ('default',            '303K dead zaratustra echoes'),
        ('eva_learnings',      '15 nodes, only IDs, no content'),
        ('eva_perceptions',    'empty, 0 nodes'),
        ('memories',           '2 nodes, 0 edges'),
        ('stories',            '2 nodes, 0 edges'),
        ('signifier_chains',   '2 nodes, 0 edges'),
        ('speaker_embeddings', '2 nodes, 0 edges'),
        ('eva_cache',          '2 nodes, 0 edges'),
    ]

    # Collections to clean Demand junk
    demand_clean = ['eva_core', 'eva_mind']

    # Collections to reap expired
    reap_targets = [
        'eva_core', 'eva_self_knowledge', 'eva_docs', 'eva_codebase',
        'eva_mind', 'patient_graph', 'malaria', 'gene_ontology',
        'tech_galaxies', 'knowledge_galaxies', 'culture_galaxies',
        'science_galaxies', 'icd10', 'snomed_ct', 'pubmed_kg', 'osm_angola',
        'malaria_atlas', 'malaria-vision',
    ]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = []

        # Drops (Tasks 1-8)
        for name, reason in drops:
            futures.append(pool.submit(task_drop, name, reason))

        # Demand cleanup (Tasks 9-10)
        for col in demand_clean:
            futures.append(pool.submit(task_clean_demands, col))

        # Reap expired (Tasks 11-28)
        for col in reap_targets:
            futures.append(pool.submit(task_reap, col))

        # Connect malaria-vision (Tasks 29-30)
        futures.append(pool.submit(task_connect_malaria_vision))

        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                log(f"  FAIL: Worker exception: {e}")
                inc('errors')

    elapsed = time.time() - start

    log("")
    log("============= RESULTS =============")
    log(f"  Dropped: {stats['dropped_collections']}")
    log(f"  Nodes deleted: {stats['deleted_nodes']}")
    log(f"  Edges created: {stats['edges_created']}")
    log(f"  Reaped: {stats['reaped']}")
    log(f"  Errors: {stats['errors']}")
    log(f"  Duration: {elapsed:.1f}s")
    log("====================================")

if __name__ == '__main__':
    main()
