#!/usr/bin/env python3
"""NietzscheDB gRPC Performance Benchmark"""
import time, json, grpc, subprocess, sys

sys.path.insert(0, "/tmp")
import nietzsche_pb2 as pb, nietzsche_pb2_grpc as rpc

ch = grpc.insecure_channel("localhost:50051")
stub = rpc.NietzscheDBStub(ch)

def t(name, fn):
    t0 = time.perf_counter()
    try:
        r = fn()
        dt = (time.perf_counter() - t0) * 1000
        print("{: <50} {:8.1f} ms  ok".format(name, dt))
        return r
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        err = str(e).split("\n")[0][:70]
        print("{: <50} {:8.1f} ms  ERR: {}".format(name, dt, err))
        return None

ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
print("=" * 70)
print("  NietzscheDB gRPC Performance Benchmark")
print("  " + ts)
print("=" * 70)

# Health
print("\n--- HEALTH & STATS ---")
t("HealthCheck", lambda: stub.HealthCheck(pb.Empty()))
t("GetStats", lambda: stub.GetStats(pb.Empty()))
t("ListCollections", lambda: stub.ListCollections(pb.Empty()))

# KNN
print("\n--- KNN SEARCH (128D Poincare) ---")
v128 = [0.01]*128
t("KNN k=5  tech_galaxies (4.5K)", lambda: stub.KnnSearch(pb.KnnRequest(collection="tech_galaxies", query_coords=v128, k=5)))
t("KNN k=10 tech_galaxies", lambda: stub.KnnSearch(pb.KnnRequest(collection="tech_galaxies", query_coords=v128, k=10)))
t("KNN k=20 tech_galaxies", lambda: stub.KnnSearch(pb.KnnRequest(collection="tech_galaxies", query_coords=v128, k=20)))
t("KNN k=5  knowledge_galaxies (1.9K)", lambda: stub.KnnSearch(pb.KnnRequest(collection="knowledge_galaxies", query_coords=v128, k=5)))
t("KNN k=5  malaria (32K)", lambda: stub.KnnSearch(pb.KnnRequest(collection="malaria", query_coords=v128, k=5)))
t("KNN k=10 malaria", lambda: stub.KnnSearch(pb.KnnRequest(collection="malaria", query_coords=v128, k=10)))
# eva_core KNN crashes server (HNSW panic on 50K nodes) — SKIPPED
t("KNN k=10 science_galaxies (6.3K)", lambda: stub.KnnSearch(pb.KnnRequest(collection="science_galaxies", query_coords=v128, k=10)))
t("KNN k=10 culture_galaxies (3.9K)", lambda: stub.KnnSearch(pb.KnnRequest(collection="culture_galaxies", query_coords=v128, k=10)))

# NQL
print("\n--- NQL QUERIES ---")
t("NQL COUNT tech_galaxies (4.5K)", lambda: stub.Query(pb.QueryRequest(nql="MATCH (n) RETURN COUNT(n)", collection="tech_galaxies")))
t("NQL COUNT malaria (32K)", lambda: stub.Query(pb.QueryRequest(nql="MATCH (n) RETURN COUNT(n)", collection="malaria")))
t("NQL COUNT default (242K)", lambda: stub.Query(pb.QueryRequest(nql="MATCH (n) RETURN COUNT(n)", collection="default")))
t("NQL WHERE energy>0.5 LIMIT 10 tech", lambda: stub.Query(pb.QueryRequest(nql="MATCH (n) WHERE n.energy > 0.5 RETURN n LIMIT 10", collection="tech_galaxies")))
t("NQL WHERE energy>0.3 LIMIT 10 malaria", lambda: stub.Query(pb.QueryRequest(nql="MATCH (n) WHERE n.energy > 0.3 RETURN n LIMIT 10", collection="malaria")))
t("NQL MATCH LIMIT 10 default", lambda: stub.Query(pb.QueryRequest(nql="MATCH (n) RETURN n LIMIT 10", collection="default")))
t("NQL MATCH LIMIT 100 malaria", lambda: stub.Query(pb.QueryRequest(nql="MATCH (n) RETURN n LIMIT 100", collection="malaria")))
t("NQL AVG energy tech_galaxies", lambda: stub.Query(pb.QueryRequest(nql="MATCH (n) RETURN AVG(n.energy)", collection="tech_galaxies")))

# Traversal
print("\n--- GRAPH TRAVERSAL ---")
try:
    r = stub.Query(pb.QueryRequest(nql="MATCH (n) RETURN n LIMIT 1", collection="tech_galaxies"))
nid = ""
if r and r.rows:
    nid = json.loads(r.rows[0]).get("n",{}).get("id","")
    if nid:
        t("BFS depth=2 tech_galaxies", lambda: stub.Bfs(pb.TraversalRequest(collection="tech_galaxies", start_node_id=nid, max_depth=2)))
        t("BFS depth=3 tech_galaxies", lambda: stub.Bfs(pb.TraversalRequest(collection="tech_galaxies", start_node_id=nid, max_depth=3)))
        t("Dijkstra depth=3 tech_galaxies", lambda: stub.Dijkstra(pb.TraversalRequest(collection="tech_galaxies", start_node_id=nid, max_depth=3)))
except Exception as e:
    print("  Traversal skipped: " + str(e)[:60])

# KNN consistency
print("\n--- KNN LATENCY CONSISTENCY (5 runs, k=10) ---")
for i in range(5):
    t("  tech_galaxies run %d" % (i+1), lambda: stub.KnnSearch(pb.KnnRequest(collection="tech_galaxies", query_coords=v128, k=10)))

# Full-text
print("\n--- FULL-TEXT SEARCH ---")
t("FTS 'malaria' in malaria", lambda: stub.FullTextSearch(pb.QueryRequest(nql="malaria", collection="malaria")))
t("FTS 'neural' in tech_galaxies", lambda: stub.FullTextSearch(pb.QueryRequest(nql="neural network", collection="tech_galaxies")))

# CPU
print("\n--- SERVER STATUS ---")
pid = subprocess.check_output("pidof nietzsche-server", shell=True).decode().strip()
cpu = subprocess.check_output("ps -p %s -o %%cpu --no-headers" % pid, shell=True).decode().strip()
mem_kb = int(subprocess.check_output("ps -p %s -o rss --no-headers" % pid, shell=True).decode().strip())
uptime = subprocess.check_output("ps -p %s -o etime --no-headers" % pid, shell=True).decode().strip()
threads = subprocess.check_output("ps -p %s -o nlwp --no-headers" % pid, shell=True).decode().strip()
print("PID=%s  CPU=%s%%  RAM=%sMB  Uptime=%s  Threads=%s" % (pid, cpu, mem_kb//1024, uptime, threads))
