#!/usr/bin/env python3
"""
memory_eva_test.py - Testa TUDO sobre a memoria da EVA no NietzscheDB.
Mostra: collections, nodes, edges, o que pensou, armazenou, conexoes, esquecimentos.
"""

import subprocess
import json
import sys
from datetime import datetime

GRPC = "localhost:50051"

def grpc(method, data=None):
    cmd = ["grpcurl", "-plaintext", "-max-msg-sz", "10000000"]
    if data:
        cmd += ["-d", json.dumps(data)]
    else:
        cmd += ["-d", "{}"]
    cmd += [GRPC, f"nietzsche.NietzscheDB.{method}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode == 0 and r.stdout.strip():
        return json.loads(r.stdout)
    return None

def nql(query, collection="default"):
    return grpc("Query", {"nql": query, "collection": collection})

def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def section(title):
    print(f"\n--- {title} ---")

def main():
    print("")
    print("+" + "="*68 + "+")
    print("|          EVA MEMORY FULL DIAGNOSTIC TEST                          |")
    print("|          " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "                                    |")
    print("+" + "="*68 + "+")

    # 1. HEALTH & GLOBAL STATS
    header("1. SERVER HEALTH & GLOBAL STATS")

    health = grpc("HealthCheck")
    if not health:
        print("  FAIL: Server not responding!")
        sys.exit(1)
    print(f"  Status: {health.get('status', '?')}")

    stats = grpc("GetStats")
    if stats:
        print(f"  Total nodes:   {int(stats.get('nodeCount', 0)):,}")
        print(f"  Total edges:   {int(stats.get('edgeCount', 0)):,}")
        print(f"  Total sensory: {int(stats.get('sensoryCount', 0)):,}")
        print(f"  Version:       {stats.get('version', '?')}")

    # 2. ALL COLLECTIONS
    header("2. ALL COLLECTIONS (EVA Memory Banks)")

    resp = grpc("ListCollections")
    if not resp:
        print("  FAIL: Cannot list collections")
        sys.exit(1)

    collections = resp.get("collections", [])
    print(f"  Total collections: {len(collections)}\n")

    collections.sort(key=lambda c: int(c.get("nodeCount", 0)), reverse=True)

    total_nodes = 0
    total_edges = 0
    eva_collections = []
    other_collections = []

    print(f"  {'Collection':<30} {'Nodes':>10} {'Edges':>10} {'Dim':>6} {'Metric':<12}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*6} {'-'*12}")

    for c in collections:
        name = c.get("collection", "?")
        nodes = int(c.get("nodeCount", 0))
        edges = int(c.get("edgeCount", 0))
        dim = c.get("dim", "?")
        metric = c.get("metric", "?")
        total_nodes += nodes
        total_edges += edges

        marker = ""
        if name.startswith("eva_"):
            eva_collections.append(c)
            marker = " [BRAIN]"
        else:
            other_collections.append(c)

        print(f"  {name:<30} {nodes:>10,} {edges:>10,} {dim:>6} {metric:<12}{marker}")

    print(f"  {'_'*70}")
    print(f"  {'TOTAL':<30} {total_nodes:>10,} {total_edges:>10,}")

    # 3. EVA CORE MEMORY - deep dive
    header("3. EVA CORE MEMORY (eva_* collections)")

    for c in eva_collections:
        name = c.get("collection", "?")
        nodes = int(c.get("nodeCount", 0))
        edges = int(c.get("edgeCount", 0))

        section(f"{name} ({nodes:,} nodes, {edges:,} edges)")

        # Sample recent nodes
        res = nql("MATCH (n) RETURN n LIMIT 5", name)
        if res and res.get("nodes"):
            for node in res["nodes"][:5]:
                nid = node.get("id", "?")[:20]
                ntype = node.get("nodeType", "?")
                energy = node.get("energy", "?")
                content = node.get("content", "{}")
                try:
                    cdata = json.loads(content) if isinstance(content, str) else content
                    text = str(cdata.get("text", cdata.get("summary", cdata.get("description", ""))))[:80]
                except Exception:
                    text = str(content)[:80]
                print(f"    [{ntype}] id={nid}.. energy={energy}")
                if text:
                    print(f"      -> {text}")

        # Count by node type
        for ntype in ["Episodic", "Semantic", "Concept"]:
            res2 = nql(f"MATCH (n:{ntype}) RETURN n LIMIT 1", name)
            count_label = "found" if res2 and res2.get("nodes") else "none"
            print(f"    {ntype}: {count_label}")

    # 4. EVA CONNECTIONS - what she linked together
    header("4. EVA CONNECTIONS (Edge sampling)")

    for c in eva_collections[:3]:
        name = c.get("collection", "?")
        section(f"Edges in {name}")

        res = nql("MATCH (a)-[e]->(b) RETURN a, b LIMIT 5", name)
        if res and res.get("nodePairs"):
            for pair in res["nodePairs"][:5]:
                src = pair.get("source", {})
                tgt = pair.get("target", {})
                etype = pair.get("edgeType", pair.get("label", "?"))
                sid = src.get("id", "?")[:16]
                tid = tgt.get("id", "?")[:16]
                print(f"    {sid}.. --[{etype}]--> {tid}..")
        elif res and res.get("nodes"):
            print(f"    (got {len(res['nodes'])} nodes from edge query)")
        else:
            print("    (no edge data returned)")

    # 5. FORGOTTEN MEMORIES - expired/reaped
    header("5. FORGOTTEN MEMORIES (ReapExpired check)")

    for c in eva_collections:
        name = c.get("collection", "?")
        res = grpc("ReapExpired", {"collection": name, "dryRun": True})
        if res:
            reaped = int(res.get("reapedCount", 0))
            if reaped > 0:
                print(f"  {name}: {reaped:,} expired nodes ready to reap")
            else:
                print(f"  {name}: 0 expired (all memories retained)")
        else:
            print(f"  {name}: (reap not supported or failed)")

    # 6. SPECIAL COLLECTIONS
    header("6. SPECIAL COLLECTIONS")

    special = {
        "memories": "EVA personal memories",
        "stories": "Stories EVA knows",
        "speaker_embeddings": "Voice fingerprints",
        "signifier_chains": "Semiotic chains",
        "malaria": "Malaria domain knowledge",
        "patient_graph": "Patient medical graph",
    }

    for sname, desc in special.items():
        found = next((c for c in collections if c.get("collection") == sname), None)
        if found:
            nodes = int(found.get("nodeCount", 0))
            edges = int(found.get("edgeCount", 0))
            print(f"  OK {sname}: {nodes:,} nodes / {edges:,} edges -- {desc}")

            res = nql("MATCH (n) RETURN n LIMIT 1", sname)
            if res and res.get("nodes"):
                node = res["nodes"][0]
                content = node.get("content", "")
                try:
                    cdata = json.loads(content) if isinstance(content, str) else content
                    preview = str(cdata)[:100]
                except Exception:
                    preview = str(content)[:100]
                print(f"     Sample: {preview}")
        else:
            print(f"  MISSING {sname}: NOT FOUND -- {desc}")

    # 7. KNOWLEDGE GALAXIES & PHILOSOPHICAL COLLECTIONS
    header("7. KNOWLEDGE GALAXIES & PHILOSOPHY")

    knowledge_keywords = [
        "galaxies", "nietzsche", "zen", "stoic", "rumi", "osho", "jung",
        "gurdjieff", "ouspensky", "nasrudin", "aesop", "hypnosis",
        "somatic", "breathing", "wim_hof", "gestalt"
    ]
    knowledge_colls = [c for c in collections if any(
        k in c.get("collection", "") for k in knowledge_keywords
    )]

    for c in knowledge_colls:
        name = c.get("collection", "?")
        nodes = int(c.get("nodeCount", 0))
        print(f"  {name}: {nodes:,} nodes")

    # SUMMARY
    header("SUMMARY")

    eva_nodes = sum(int(c.get("nodeCount", 0)) for c in eva_collections)
    eva_edges = sum(int(c.get("edgeCount", 0)) for c in eva_collections)

    print(f"  EVA Brain:")
    print(f"     Collections:  {len(eva_collections)}")
    print(f"     Nodes:        {eva_nodes:,}")
    print(f"     Edges:        {eva_edges:,}")
    print()
    print(f"  Total System:")
    print(f"     Collections:  {len(collections)}")
    print(f"     Nodes:        {total_nodes:,}")
    print(f"     Edges:        {total_edges:,}")
    print()
    if total_nodes > 0:
        print(f"  Memory density:  {total_edges/total_nodes:.2f} edges/node")
        print(f"  EVA share:       {eva_nodes/total_nodes*100:.1f}% of all nodes")
    print()
    print("+" + "="*68 + "+")
    print("|                    TEST COMPLETE                                  |")
    print("+" + "="*68 + "+")

if __name__ == "__main__":
    main()
