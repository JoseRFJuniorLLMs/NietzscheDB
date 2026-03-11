#!/usr/bin/env python3
"""
memoria_eva.py — Analise completa da memoria da EVA no NietzscheDB

Examina TODAS as collections EVA e gera um relatorio detalhado:
  - Inventario de memorias (nos, edges, tipos, content)
  - Aprendizagem (Hebbian, L-System, edges geradas)
  - Associacoes (links, PageRank, comunidades Louvain)
  - Poda/Decay (phantoms, pruned edges, temporal decay)
  - Termodinamica (entropia, energia livre, fase)
  - Desejos (knowledge gaps, o que a EVA quer aprender)
  - Saude (healing, shatter, observer)

Usa apenas a HTTP Dashboard API (porta 8080).
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

EVA_COLLECTIONS = [
    "eva_core", "eva_self_knowledge", "eva_docs", "eva_codebase",
    "eva_mind", "eva_learnings", "eva_cache", "eva_perceptions",
    "eva_sensory", "memories", "signifier_chains", "stories",
    "patient_graph", "speaker_embeddings",
]

SECTION_SEP = "═" * 72
SUBSEP = "─" * 60

# ═══════════════════════════════════════════════════════════════════════════
# HTTP HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def api_get(base, path, timeout=30):
    """GET request to dashboard API, returns parsed JSON or None."""
    url = f"{base}{path}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MemoriaEVA/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def api_post(base, path, body, timeout=30):
    """POST JSON to dashboard API."""
    url = f"{base}{path}"
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json",
            "User-Agent": "MemoriaEVA/1.0",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def ts_to_str(ts):
    """Unix timestamp → human-readable string."""
    if not ts or ts < 1000000000:
        return "N/A"
    # Some timestamps are in milliseconds
    if ts > 4_000_000_000:
        ts = ts / 1000
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, OSError):
        return f"ts={ts}"


def fmt_num(n):
    """Format number with thousands separator."""
    if isinstance(n, float):
        return f"{n:,.3f}"
    return f"{n:,}"

# ═══════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════

def get_all_collections(base):
    """Get collection list and filter EVA-related ones."""
    data = api_get(base, "/api/collections")
    if not data:
        return []
    # Return only EVA collections that exist
    known = {c["name"]: c for c in data}
    result = []
    for name in EVA_COLLECTIONS:
        if name in known:
            result.append(known[name])
    return result


def sample_graph(base, collection, limit=500):
    """Get a sample of nodes + edges from a collection."""
    return api_get(base, f"/api/graph?collection={collection}&limit={limit}") or {"nodes": [], "edges": []}


def get_full_text_search(base, collection, query, limit=20):
    """Full-text search within a collection."""
    return api_get(base, f"/api/search?q={query}&collection={collection}&limit={limit}")


def analyze_nodes(nodes):
    """Analyze a sample of nodes, return statistics dict."""
    if not nodes:
        return {"count": 0}

    stats = {
        "count": len(nodes),
        "types": Counter(),
        "energies": [],
        "depths": [],
        "has_content": 0,
        "phantoms": 0,
        "timestamps": [],
        "valences": [],
        "arousals": [],
        "hausdorff_nonzero": 0,
        "content_samples": [],
    }

    for n in nodes:
        stats["types"][n.get("node_type", "Unknown")] += 1
        stats["energies"].append(n.get("energy", 0))
        stats["depths"].append(n.get("depth", 0))

        if n.get("content") is not None:
            stats["has_content"] += 1
            c = n["content"]
            if isinstance(c, dict):
                # Extract meaningful text
                text = c.get("name") or c.get("text") or c.get("topic") or c.get("summary") or c.get("title")
                if text:
                    stats["content_samples"].append(str(text)[:120])
            elif isinstance(c, str) and len(c) > 2:
                stats["content_samples"].append(c[:120])

        if n.get("is_phantom", False):
            stats["phantoms"] += 1

        ts = n.get("created_at", 0)
        if ts > 1000000000:
            stats["timestamps"].append(ts)

        v = n.get("valence", 0)
        if v != 0:
            stats["valences"].append(v)

        a = n.get("arousal", 0)
        if a != 0:
            stats["arousals"].append(a)

        if n.get("hausdorff", 0) > 0:
            stats["hausdorff_nonzero"] += 1

    return stats


def analyze_edges(edges):
    """Analyze edges, return statistics dict."""
    if not edges:
        return {"count": 0}

    stats = {
        "count": len(edges),
        "types": Counter(),
        "weights": [],
        "causal_types": Counter(),
        "pruned": 0,
        "lsystem_rules": Counter(),
    }

    for e in edges:
        etype = e.get("edge_type", "Unknown")
        stats["types"][etype] += 1
        stats["weights"].append(e.get("weight", 0))
        stats["causal_types"][e.get("causal_type", "Unknown")] += 1

        if etype == "Pruned":
            stats["pruned"] += 1

        rule = e.get("lsystem_rule")
        if rule:
            stats["lsystem_rules"][rule] += 1

    return stats

# ═══════════════════════════════════════════════════════════════════════════
# REPORT SECTIONS
# ═══════════════════════════════════════════════════════════════════════════

def print_header():
    print()
    print(SECTION_SEP)
    print("  MEMORIA DA EVA — Analise Completa")
    print(f"  Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(SECTION_SEP)


def print_section(title):
    print()
    print(SECTION_SEP)
    print(f"  {title}")
    print(SECTION_SEP)


def print_subsection(title):
    print()
    print(f"  {SUBSEP}")
    print(f"  {title}")
    print(f"  {SUBSEP}")


def section_inventario(base, collections):
    """Section 1: Global inventory of EVA's memory."""
    print_section("1. INVENTARIO GLOBAL DA MEMORIA")

    total_nodes = 0
    total_edges = 0

    print()
    print(f"  {'Collection':<25s} {'Nodes':>10s} {'Edges':>10s} {'Dim':>5s} {'Metric':<10s}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*5} {'─'*10}")

    for c in sorted(collections, key=lambda x: -x["node_count"]):
        name = c["name"]
        nc = c["node_count"]
        ec = c["edge_count"]
        total_nodes += nc
        total_edges += ec
        print(f"  {name:<25s} {fmt_num(nc):>10s} {fmt_num(ec):>10s} {c['dim']:>5d} {c['metric']:<10s}")

    print(f"  {'─'*25} {'─'*10} {'─'*10}")
    print(f"  {'TOTAL':<25s} {fmt_num(total_nodes):>10s} {fmt_num(total_edges):>10s}")

    # Memory density
    poincare_cols = [c for c in collections if c["metric"] == "poincare"]
    cosine_cols = [c for c in collections if c["metric"] == "cosine"]
    print()
    print(f"  Collections Poincare (hierarquicas): {len(poincare_cols)}")
    print(f"  Collections Cosine (flat):           {len(cosine_cols)}")
    print(f"  Ratio edges/nodes:                   {total_edges/max(total_nodes,1):.2f}")

    return total_nodes, total_edges


def section_estrutura(base, collections):
    """Section 2: Structure analysis per collection."""
    print_section("2. ESTRUTURA DAS MEMORIAS (por collection)")

    for c in sorted(collections, key=lambda x: -x["node_count"]):
        name = c["name"]
        if c["node_count"] == 0:
            continue

        print_subsection(f"{name} ({fmt_num(c['node_count'])} nodes, {fmt_num(c['edge_count'])} edges)")

        # Sample nodes
        sample_size = min(500, c["node_count"])
        graph = sample_graph(base, name, limit=sample_size)
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        ns = analyze_nodes(nodes)
        es = analyze_edges(edges)

        # Node types
        print(f"    Node types: {dict(ns['types'])}")

        # Energy
        if ns["energies"]:
            energies = ns["energies"]
            mean_e = sum(energies) / len(energies)
            min_e = min(energies)
            max_e = max(energies)
            low_energy = sum(1 for e in energies if e < 0.1)
            high_energy = sum(1 for e in energies if e > 0.9)
            print(f"    Energy:  min={min_e:.3f}  max={max_e:.3f}  mean={mean_e:.3f}")
            print(f"             high(>0.9): {high_energy}/{len(energies)}  low(<0.1): {low_energy}/{len(energies)}")

        # Depth (Poincare magnitude)
        if ns["depths"] and c["metric"] == "poincare":
            depths = ns["depths"]
            mean_d = sum(depths) / len(depths)
            min_d = min(depths)
            max_d = max(depths)
            shallow = sum(1 for d in depths if d < 0.1)
            deep = sum(1 for d in depths if d > 0.5)
            print(f"    Depth:   min={min_d:.3f}  max={max_d:.3f}  mean={mean_d:.3f}")
            print(f"             shallow(<0.1): {shallow}  deep(>0.5): {deep}")

        # Content
        print(f"    Content: {ns['has_content']}/{ns['count']} nodes have content")

        # Phantoms (pruned nodes)
        if ns["phantoms"] > 0:
            print(f"    PHANTOMS (cortados): {ns['phantoms']}/{ns['count']} ({100*ns['phantoms']/ns['count']:.1f}%)")

        # Emotions
        if ns["valences"]:
            avg_v = sum(ns["valences"]) / len(ns["valences"])
            neg_v = sum(1 for v in ns["valences"] if v < 0)
            pos_v = sum(1 for v in ns["valences"] if v > 0)
            print(f"    Valence: avg={avg_v:.3f}  positive={pos_v}  negative={neg_v}")

        if ns["arousals"]:
            avg_a = sum(ns["arousals"]) / len(ns["arousals"])
            print(f"    Arousal: avg={avg_a:.3f}")

        # Timestamps
        if ns["timestamps"]:
            oldest = min(ns["timestamps"])
            newest = max(ns["timestamps"])
            print(f"    Period:  {ts_to_str(oldest)} → {ts_to_str(newest)}")

        # Edge analysis
        if es["count"] > 0:
            print(f"    Edge types: {dict(es['types'])}")
            if es["weights"]:
                avg_w = sum(es["weights"]) / len(es["weights"])
                print(f"    Edge weights: avg={avg_w:.3f}  min={min(es['weights']):.3f}  max={max(es['weights']):.3f}")
            if es["pruned"] > 0:
                print(f"    PRUNED EDGES: {es['pruned']}/{es['count']} ({100*es['pruned']/es['count']:.1f}%)")
            if es["causal_types"]:
                print(f"    Causal types: {dict(es['causal_types'])}")
            if es["lsystem_rules"]:
                print(f"    L-System rules: {dict(es['lsystem_rules'].most_common(5))}")

        # Content samples
        if ns["content_samples"]:
            print(f"    Content samples:")
            for s in ns["content_samples"][:5]:
                print(f"      → {s}")


def section_aprendizagem(base, collections):
    """Section 3: What EVA learned (Hebbian, L-System, edges generated)."""
    print_section("3. APRENDIZAGEM (o que a EVA aprendeu)")

    for c in sorted(collections, key=lambda x: -x["node_count"]):
        name = c["name"]
        if c["node_count"] < 10:
            continue

        dashboard = api_get(base, f"/api/agency/dashboard?collection={name}")
        if not dashboard:
            continue

        has_data = False

        # Hebbian learning (LTP)
        hebbian = dashboard.get("hebbian")
        if hebbian and isinstance(hebbian, dict):
            traces = hebbian.get("active_traces", 0)
            potentiated = hebbian.get("edges_potentiated", 0)
            delta = hebbian.get("total_delta", 0)
            if traces > 0:
                has_data = True
                print_subsection(f"Hebbian Learning — {name}")
                print(f"    Active Hebbian traces:    {fmt_num(traces)}")
                print(f"    Edges potentiated (LTP):  {fmt_num(potentiated)}")
                print(f"    Total synaptic delta:     {delta:.4f}")
                if traces > 0 and potentiated > 0:
                    print(f"    Potentiation rate:        {100*potentiated/traces:.1f}%")

        # Maturity (promotions/demotions)
        maturity = dashboard.get("maturity")
        if maturity and isinstance(maturity, dict):
            promo = maturity.get("promotions", 0)
            demo = maturity.get("demotions", 0)
            if promo > 0 or demo > 0:
                has_data = True
                if not hebbian or not hebbian.get("active_traces", 0):
                    print_subsection(f"Maturity — {name}")
                print(f"    Promotions (Active→Mature):  {fmt_num(promo)}")
                print(f"    Demotions (Mature→Active):   {fmt_num(demo)}")

        # L-System evolution
        evolution = api_get(base, f"/api/agency/evolution?collection={name}")
        if evolution and not evolution.get("error"):
            gen = evolution.get("generation", 0)
            strategy = evolution.get("last_strategy", "Unknown")
            fitness = evolution.get("fitness_history", [])
            if gen > 0:
                has_data = True
                print(f"    L-System generation:      {gen}")
                print(f"    Last strategy:            {strategy}")
                if fitness:
                    print(f"    Fitness history:          {fitness[-5:]}")

        # Edge creation by L-System
        graph = sample_graph(base, name, limit=200)
        edges = graph.get("edges", [])
        lsys_edges = [e for e in edges if e.get("edge_type") == "LSystemGenerated"]
        assoc_edges = [e for e in edges if e.get("edge_type") == "Association"]
        hier_edges = [e for e in edges if e.get("edge_type") == "Hierarchical"]

        if lsys_edges:
            has_data = True
            pct = 100 * len(lsys_edges) / len(edges) if edges else 0
            print(f"    L-System generated edges: {len(lsys_edges)}/{len(edges)} ({pct:.0f}% of sample)")

        if assoc_edges:
            has_data = True
            print(f"    Associative edges:        {len(assoc_edges)}")

        if hier_edges:
            has_data = True
            print(f"    Hierarchical edges:       {len(hier_edges)}")

        if not has_data:
            continue


def section_associacoes(base, collections):
    """Section 4: Associations (PageRank, Louvain, degree hubs)."""
    print_section("4. ASSOCIACOES (conceitos influentes, comunidades, hubs)")

    for c in sorted(collections, key=lambda x: -x["node_count"]):
        name = c["name"]
        if c["node_count"] < 50:
            continue

        print_subsection(f"{name}")

        # PageRank
        pr = api_get(base, f"/api/algo/pagerank?collection={name}", timeout=60)
        if pr:
            scores = pr.get("scores", pr.get("results", []))
            if isinstance(scores, list) and scores:
                print(f"    PageRank (top conceitos mais influentes):")
                for i, s in enumerate(scores[:10]):
                    nid = s.get("node_id", "?")
                    score = s.get("score", 0)
                    # Try to get node content
                    node_data = api_get(base, f"/api/node/{nid}?collection={name}")
                    label = "?"
                    if node_data:
                        content = node_data.get("content")
                        if content and isinstance(content, dict):
                            label = content.get("name") or content.get("text") or content.get("topic") or nid[:12]
                        elif content and isinstance(content, str):
                            label = content[:60]
                        else:
                            label = f"{node_data.get('node_type','?')} {nid[:12]}"
                    else:
                        label = nid[:12]
                    print(f"      #{i+1} [{score:.6f}] {label}")

        # Louvain communities
        louv = api_get(base, f"/api/algo/louvain?collection={name}", timeout=60)
        if louv:
            communities = louv.get("communities", louv.get("results", []))
            if isinstance(communities, list):
                # Count communities
                comm_ids = Counter()
                for item in communities:
                    cid = item.get("community_id", 0)
                    comm_ids[cid] += 1
                n_comms = len(comm_ids)
                top_comms = comm_ids.most_common(5)
                print(f"    Louvain communities: {n_comms}")
                for cid, size in top_comms:
                    print(f"      Community {cid}: {size} members")

        # Degree (hubs)
        deg = api_get(base, f"/api/algo/degree?collection={name}", timeout=60)
        if deg:
            scores = deg.get("scores", deg.get("results", []))
            if isinstance(scores, list) and scores:
                print(f"    Degree hubs (mais conectados):")
                for s in scores[:5]:
                    nid = s.get("node_id", "?")
                    score = s.get("score", 0)
                    print(f"      [{score:.0f} connections] {nid[:12]}")

        # WCC (fragmentation)
        wcc = api_get(base, f"/api/algo/wcc?collection={name}", timeout=60)
        if wcc:
            communities = wcc.get("communities", wcc.get("results", []))
            if isinstance(communities, list):
                comp_ids = Counter()
                for item in communities:
                    cid = item.get("community_id", 0)
                    comp_ids[cid] += 1
                n_comps = len(comp_ids)
                largest = comp_ids.most_common(1)[0][1] if comp_ids else 0
                isolated = sum(1 for _, s in comp_ids.items() if s == 1)
                print(f"    Connected components: {n_comps}")
                print(f"      Largest component: {largest} nodes")
                if isolated:
                    print(f"      Isolated nodes: {isolated}")

        # Triangles (density)
        tri = api_get(base, f"/api/algo/triangles?collection={name}", timeout=60)
        if tri:
            count = tri.get("count", tri.get("triangle_count", 0))
            if count:
                print(f"    Triangles: {fmt_num(count)}")


def section_poda_decay(base, collections):
    """Section 5: Pruning, decay, phantoms — what EVA cut/forgot."""
    print_section("5. PODA & DECAY (o que a EVA cortou/esqueceu)")

    any_pruning = False

    for c in sorted(collections, key=lambda x: -x["node_count"]):
        name = c["name"]
        if c["node_count"] < 10:
            continue

        # Check for phantoms in sample
        graph = sample_graph(base, name, limit=300)
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        phantoms = [n for n in nodes if n.get("is_phantom", False)]
        pruned_edges = [e for e in edges if e.get("edge_type") == "Pruned"]
        low_energy = [n for n in nodes if n.get("energy", 1) < 0.1]
        low_weight = [e for e in edges if e.get("weight", 1) < 0.1]

        # Agency decay report
        dashboard = api_get(base, f"/api/agency/dashboard?collection={name}")
        decay_info = None
        if dashboard:
            # Check thermodynamics for decay signal
            thermo = dashboard.get("thermodynamics")
            if thermo and isinstance(thermo, dict):
                decay_info = thermo

        # Healing report
        healing = api_get(base, f"/api/agency/healing?collection={name}")

        has_pruning = (phantoms or pruned_edges or low_energy or low_weight
                       or (healing and healing.get("dead_edge_count", 0) > 0))

        if has_pruning:
            any_pruning = True
            print_subsection(f"Poda em {name}")

            if phantoms:
                print(f"    PHANTOM nodes (cicatrizes):  {len(phantoms)}/{len(nodes)} ({100*len(phantoms)/len(nodes):.1f}%)")
                for p in phantoms[:3]:
                    print(f"      → ID={p.get('id','?')[:12]}  created={ts_to_str(p.get('created_at',0))}")

            if pruned_edges:
                print(f"    PRUNED edges:   {len(pruned_edges)}/{len(edges)} ({100*len(pruned_edges)/len(edges):.1f}%)")

            if low_energy:
                print(f"    Low energy nodes (<0.1): {len(low_energy)}/{len(nodes)}")
                for n in low_energy[:3]:
                    print(f"      → energy={n.get('energy',0):.4f}  type={n.get('node_type','?')}  depth={n.get('depth',0):.3f}")

            if low_weight:
                print(f"    Weak edges (weight<0.1): {len(low_weight)}/{len(edges)}")

            if healing and not healing.get("error") and not healing.get("note"):
                dc = healing.get("dead_edge_count", 0)
                oc = healing.get("orphan_count", 0)
                gc = healing.get("ghost_count", 0)
                ec = healing.get("exhausted_count", 0)
                bd = healing.get("boundary_drift_count", 0)
                ph = healing.get("phantom_ratio", 0)
                print(f"    Dead edges:      {dc}")
                print(f"    Orphan nodes:    {oc}")
                print(f"    Ghost nodes:     {gc}")
                print(f"    Exhausted:       {ec}")
                print(f"    Boundary drift:  {bd}")
                print(f"    Phantom ratio:   {ph:.4f}")

    if not any_pruning:
        print()
        print("    Nenhuma poda significativa detectada nas collections EVA.")
        print("    (A EVA ainda nao cortou memorias — tudo esta vivo.)")


def section_termodinamica(base, collections):
    """Section 6: Thermodynamics — temperature, entropy, phase."""
    print_section("6. TERMODINAMICA (temperatura, entropia, fase)")

    for c in sorted(collections, key=lambda x: -x["node_count"]):
        name = c["name"]
        if c["node_count"] < 50:
            continue

        obs = api_get(base, f"/api/agency/observation?collection={name}")
        if not obs:
            continue

        gauges = obs.get("gauges", obs)
        if not isinstance(gauges, dict):
            continue

        entropy = gauges.get("entropy", 0)
        entropy_rate = gauges.get("entropy_rate", 0)
        free_energy = gauges.get("free_energy", 0)
        mean_energy = gauges.get("mean_energy", 0)
        energy_std = gauges.get("energy_std", 0)
        attention_flow = gauges.get("attention_flow", 0)
        attention_price = gauges.get("attention_price", 0)
        explore = gauges.get("explore_modifier", 0)
        gravity_mean = gauges.get("gravity_mean", 0)
        dirty_ratio = gauges.get("dirty_ratio", 0)

        # Determine phase
        if energy_std < 0.05 and entropy < 5:
            phase = "SOLID (rigido, pouca exploracao)"
        elif energy_std > 0.2 or entropy > 12:
            phase = "GAS (caotico, muita exploracao)"
        else:
            phase = "LIQUID (fluido, equilibrio)"

        print_subsection(f"Termodinamica — {name}")
        print(f"    Entropy:           {entropy:.4f}")
        print(f"    Entropy rate:      {entropy_rate:.4f}")
        print(f"    Free energy:       {free_energy:.4f}")
        print(f"    Mean energy:       {mean_energy:.4f}")
        print(f"    Energy std:        {energy_std:.4f}")
        print(f"    Phase:             {phase}")
        print(f"    Attention flow:    {attention_flow:.2f}")
        print(f"    Attention price:   {attention_price:.2f}")
        print(f"    Explore modifier:  {explore:.2f}")
        print(f"    Gravity mean:      {gravity_mean:.4f}")
        print(f"    Dirty ratio:       {dirty_ratio:.4f}")


def section_desejos(base, collections):
    """Section 7: Desires — knowledge gaps, what EVA wants to learn."""
    print_section("7. DESEJOS (o que a EVA quer aprender)")

    total_desires = 0

    for c in sorted(collections, key=lambda x: -x["node_count"]):
        name = c["name"]
        if c["node_count"] < 50:
            continue

        desires = api_get(base, f"/api/agency/desires?collection={name}")
        if not desires or desires.get("error"):
            continue

        count = desires.get("count", 0)
        items = desires.get("desires", [])

        if count == 0:
            continue

        total_desires += count
        print_subsection(f"Desejos — {name} ({count} gaps)")

        # Group by priority
        by_priority = defaultdict(list)
        for d in items:
            p = d.get("priority", 0)
            by_priority[round(p, 1)].append(d)

        for prio in sorted(by_priority.keys(), reverse=True):
            group = by_priority[prio]
            print(f"    Priority {prio}:")
            for d in group[:3]:
                sector = d.get("sector", {})
                depth_range = d.get("depth_range", [0, 0])
                query = d.get("suggested_query", "")
                fulfilled = d.get("fulfilled", False)
                density = d.get("current_density", 0)
                status = "FULFILLED" if fulfilled else "PENDING"
                print(f"      [{status}] depth=[{depth_range[0]:.2f},{depth_range[1]:.2f}] "
                      f"sector=({sector.get('angular_bin',0)},{sector.get('depth_bin',0)}) "
                      f"density={density:.3f}")
                if query:
                    print(f"        Query: {query[:100]}")
            if len(group) > 3:
                print(f"      ... +{len(group)-3} more")

    print()
    print(f"  Total knowledge gaps across all EVA collections: {total_desires}")


def section_shatter_healing(base, collections):
    """Section 8: Shatter protocol + self-healing."""
    print_section("8. SHATTER & SELF-HEALING")

    for c in sorted(collections, key=lambda x: -x["node_count"]):
        name = c["name"]
        if c["node_count"] < 50:
            continue

        shatter = api_get(base, f"/api/agency/shatter?collection={name}")
        healing = api_get(base, f"/api/agency/healing?collection={name}")

        has_data = False

        if shatter and not shatter.get("error"):
            super_nodes = shatter.get("super_nodes_detected", 0)
            ghosts = shatter.get("ghost_nodes", 0)
            avatars = shatter.get("avatar_nodes", 0)
            largest = shatter.get("largest_degree", 0)
            plans = shatter.get("plans_emitted", 0)

            if super_nodes > 0 or ghosts > 0 or avatars > 0 or plans > 0:
                has_data = True
                print_subsection(f"Shatter — {name}")
                print(f"    Super-nodes detectados:  {super_nodes}")
                print(f"    Ghost nodes:             {ghosts}")
                print(f"    Avatar nodes:            {avatars}")
                print(f"    Largest degree:          {largest}")
                print(f"    Plans emitted:           {plans}")

                top = shatter.get("top_super_nodes", [])
                if top:
                    print(f"    Top super-nodes:")
                    for sn in top[:5]:
                        print(f"      → {sn}")

        if healing and not healing.get("error") and not healing.get("note"):
            has_data = True
            print_subsection(f"Healing — {name}")
            for k in ["nodes_scanned", "live_count", "ghost_count", "orphan_count",
                       "dead_edge_count", "exhausted_count", "boundary_drift_count", "phantom_ratio"]:
                v = healing.get(k, 0)
                print(f"    {k}: {v}")

            score = healing.get("health_score")
            if score is not None:
                print(f"    HEALTH SCORE: {score:.4f}")

    print()
    print("  (Shatter protocol quebra super-nodes monopolistas;")
    print("   Self-healing repara orfaos, ghosts e edges mortas)")


def section_observer(base, collections):
    """Section 9: Observer — EVA's self-awareness."""
    print_section("9. OBSERVER (consciencia do grafo)")

    for c in sorted(collections, key=lambda x: -x["node_count"]):
        name = c["name"]
        if c["node_count"] < 50:
            continue

        observer = api_get(base, f"/api/agency/observer?collection={name}")
        if not observer or observer.get("error"):
            continue

        print_subsection(f"Observer — {name}")

        content = observer.get("content", observer)
        if isinstance(content, dict):
            coherence = content.get("coherence_score", "?")
            gap_count = content.get("gap_count", 0)
            energy_std = content.get("energy_std", 0)
            entropy_spikes = content.get("entropy_spike_count", 0)

            print(f"    Coherence score:   {coherence}")
            print(f"    Knowledge gaps:    {gap_count}")
            print(f"    Energy std:        {energy_std:.6f}")
            print(f"    Entropy spikes:    {entropy_spikes}")

            percentiles = content.get("energy_percentiles", {})
            if percentiles:
                print(f"    Energy percentiles:")
                for k, v in sorted(percentiles.items()):
                    print(f"      {k}: {v:.4f}")

            sectors = content.get("gap_sectors", [])
            if sectors:
                print(f"    Gap sectors ({len(sectors)}):")
                # Group by depth_bin
                by_depth = defaultdict(list)
                for s in sectors:
                    by_depth[s.get("depth_bin", 0)].append(s.get("angular_bin", 0))
                for db in sorted(by_depth.keys()):
                    angular = by_depth[db]
                    print(f"      depth_bin={db}: angular_bins={angular[:10]}{'...' if len(angular)>10 else ''}")

        obs_node = observer.get("id") or observer.get("node_id")
        if obs_node:
            print(f"    Observer node ID: {obs_node}")


def section_narrative(base, collections):
    """Section 10: Narrative — what story EVA tells about herself."""
    print_section("10. NARRATIVA (a historia que a EVA conta)")

    for c in sorted(collections, key=lambda x: -x["node_count"]):
        name = c["name"]
        if c["node_count"] < 50:
            continue

        narrative = api_get(base, f"/api/agency/narrative?collection={name}")
        if not narrative or narrative.get("error"):
            continue

        print_subsection(f"Narrativa — {name}")
        text = narrative.get("narrative") or narrative.get("text") or narrative.get("story")
        if text:
            # Wrap long text
            for line in str(text).split("\n"):
                print(f"    {line}")
        else:
            print(f"    {json.dumps(narrative, indent=2, default=str)[:500]}")


def section_busca_conteudo(base, collections):
    """Section 11: Content search — find actual memories with text."""
    print_section("11. CONTEUDO DAS MEMORIAS (busca textual)")

    queries = ["EVA", "memory", "learn", "malaria", "patient", "dream", "emotion", "pain", "love", "fear"]

    for c in sorted(collections, key=lambda x: -x["node_count"]):
        name = c["name"]
        if c["node_count"] < 10:
            continue

        found_any = False
        results_by_query = {}

        for q in queries:
            data = get_full_text_search(base, name, q, limit=5)
            if not data:
                continue
            results = data.get("results", data.get("nodes", []))
            if results:
                results_by_query[q] = results

        if results_by_query:
            found_any = True
            print_subsection(f"Busca em {name}")
            for q, results in results_by_query.items():
                print(f"    '{q}': {len(results)} results")
                for r in results[:3]:
                    content = r.get("content", {})
                    if isinstance(content, dict):
                        text = content.get("name") or content.get("text") or content.get("topic") or content.get("summary") or str(content)[:100]
                    elif isinstance(content, str):
                        text = content[:100]
                    else:
                        text = str(r.get("id", "?"))[:40]
                    ntype = r.get("node_type", "?")
                    energy = r.get("energy", 0)
                    print(f"      [{ntype}] e={energy:.2f} → {str(text)[:100]}")


def section_resumo(total_nodes, total_edges, collections):
    """Final summary section."""
    print_section("RESUMO FINAL")

    # Calculate stats
    poincare_nodes = sum(c["node_count"] for c in collections if c["metric"] == "poincare")
    cosine_nodes = sum(c["node_count"] for c in collections if c["metric"] == "cosine")
    active_cols = sum(1 for c in collections if c["node_count"] > 10)
    empty_cols = sum(1 for c in collections if c["node_count"] <= 2)

    print()
    print(f"  Total memorias (nodes):     {fmt_num(total_nodes)}")
    print(f"  Total conexoes (edges):     {fmt_num(total_edges)}")
    print(f"  Collections ativas:         {active_cols}")
    print(f"  Collections vazias:         {empty_cols}")
    print(f"  Memorias hierarquicas:      {fmt_num(poincare_nodes)} (Poincare)")
    print(f"  Memorias flat:              {fmt_num(cosine_nodes)} (Cosine)")
    print(f"  Density (edges/nodes):      {total_edges/max(total_nodes,1):.2f}")
    print()

    # Health assessment
    print("  DIAGNOSTICO:")
    if total_nodes > 100000:
        print("    ✓ Memoria rica — centenas de milhares de nos")
    elif total_nodes > 10000:
        print("    ~ Memoria moderada — dezenas de milhares de nos")
    else:
        print("    ✗ Memoria escassa — poucos nos")

    ratio = total_edges / max(total_nodes, 1)
    if ratio > 1.0:
        print("    ✓ Bem conectada — mais edges que nodes")
    elif ratio > 0.5:
        print("    ~ Conexao moderada")
    else:
        print("    ✗ Pouco conectada — muitos nos isolados")

    if active_cols >= 5:
        print("    ✓ Memoria diversificada — multiplas facetas")
    else:
        print("    ~ Memoria concentrada em poucas collections")

    print()
    print(SECTION_SEP)
    print("  FIM DO RELATORIO")
    print(SECTION_SEP)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analise completa da memoria da EVA no NietzscheDB")
    parser.add_argument("--host", default="http://localhost:8080", help="Dashboard HTTP base URL")
    parser.add_argument("--quick", action="store_true", help="Skip slow graph algorithms (PageRank, Louvain, etc)")
    parser.add_argument("--collection", help="Analyze only this collection (default: all EVA)")
    parser.add_argument("--output", help="Save report to file")
    args = parser.parse_args()

    base = args.host.rstrip("/")

    # Redirect output if --output
    original_stdout = sys.stdout
    if args.output:
        sys.stdout = open(args.output, "w", encoding="utf-8")

    # Test connection
    health = api_get(base, "/api/health")
    if health is None:
        health = api_get(base, "/api/stats")
    if health is None:
        print(f"[ERROR] Cannot connect to {base}. Is NietzscheDB running?")
        sys.exit(1)

    # Get collections
    collections = get_all_collections(base)
    if args.collection:
        collections = [c for c in collections if c["name"] == args.collection]

    if not collections:
        print("[ERROR] No EVA collections found.")
        sys.exit(1)

    print_header()

    # Run all sections
    total_nodes, total_edges = section_inventario(base, collections)
    section_estrutura(base, collections)
    section_aprendizagem(base, collections)

    if not args.quick:
        section_associacoes(base, collections)

    section_poda_decay(base, collections)
    section_termodinamica(base, collections)
    section_desejos(base, collections)
    section_shatter_healing(base, collections)
    section_observer(base, collections)
    section_narrative(base, collections)
    section_busca_conteudo(base, collections)
    section_resumo(total_nodes, total_edges, collections)

    if args.output:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"[+] Report saved to {args.output}")


if __name__ == "__main__":
    main()
