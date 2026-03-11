#!/usr/bin/env python3
"""
NietzscheDB SUPER REPORT — Complete Brain Scan

Queries ALL agency subsystems, algorithms, thermodynamics, dreams,
forgetting, growth, energy, pruning, discoveries, and network health.

Usage:
  python scripts/super_report.py [--host HOST] [--collections NAMES]
"""

import requests
import json
import sys
import time
import argparse
from datetime import datetime, timezone
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_HOST = "http://localhost:8080"
GALAXY_COLLECTIONS = ["tech_galaxies", "knowledge_galaxies", "culture_galaxies", "science_galaxies"]
ALL_COLLECTIONS = None  # will be populated


def api(host, path, method="GET", json_data=None, timeout=30):
    """Call dashboard API."""
    url = f"{host}{path}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=timeout)
        else:
            r = requests.post(url, json=json_data, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    except Exception as e:
        return {"error": str(e)[:200]}


def hr(char="═", width=70):
    return char * width


def section(title, char="═"):
    print(f"\n{hr(char)}")
    print(f"  {title}")
    print(f"{hr(char)}")


def subsection(title, char="─"):
    print(f"\n  {char*3} {title} {char*3}")


def bar(value, max_val, width=30, fill="█", empty="░"):
    """ASCII progress bar."""
    if max_val == 0:
        return empty * width
    ratio = min(value / max_val, 1.0)
    filled = int(ratio * width)
    return fill * filled + empty * (width - filled)


def sparkline(values, chars="▁▂▃▄▅▆▇█"):
    """Mini sparkline from values."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    return "".join(chars[min(int((v - mn) / rng * (len(chars) - 1)), len(chars) - 1)] for v in values)


def format_num(n):
    """Format number with K/M suffix."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def safe_get(d, *keys, default=None):
    """Safely navigate nested dicts."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


# ═══════════════════════════════════════════════════════════════════════════
# REPORT SECTIONS
# ═══════════════════════════════════════════════════════════════════════════

def report_overview(host):
    """Section 1: Global Overview."""
    section("1. GLOBAL BRAIN OVERVIEW")

    stats = api(host, "/api/stats")
    collections = api(host, "/api/collections")

    if "error" in stats:
        print(f"  [!] Stats error: {stats['error']}")
        return collections

    total_nodes = stats.get("node_count", 0)
    total_edges = stats.get("edge_count", 0)
    version = stats.get("version", "?")
    n_collections = stats.get("collections", 0)

    print(f"""
  Version:      {version}
  Collections:  {n_collections}
  Total Nodes:  {format_num(total_nodes)}
  Total Edges:  {format_num(total_edges)}
  Ratio E/N:    {total_edges/max(total_nodes,1):.2f} edges/node
  Timestamp:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
""")

    if isinstance(collections, list):
        # Sort by node_count desc
        collections.sort(key=lambda c: c.get("node_count", 0), reverse=True)

        print(f"  {'Collection':<30} {'Nodes':>8} {'Edges':>8} {'Dim':>5} {'Metric':<10} {'Density'}")
        print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*5} {'─'*10} {'─'*10}")

        for c in collections:
            name = c.get("name", "?")
            nodes = c.get("node_count", 0)
            edges = c.get("edge_count", 0)
            dim = c.get("dim", 0)
            metric = c.get("metric", "?")
            density = edges / max(nodes, 1)
            active = "●" if nodes > 0 else "○"
            print(f"  {active} {name:<28} {format_num(nodes):>8} {format_num(edges):>8} {dim:>5} {metric:<10} {density:.2f}")

    return collections


def report_agency_health(host, collection):
    """Section 2: Agency Health — Thermodynamics, Energy, Fractal Health."""
    subsection(f"Agency Health: {collection}")

    health = api(host, f"/api/agency/health/latest?collection={collection}")
    if not health or "error" in health:
        print(f"    No health data available")
        return

    # Extract fields
    total_nodes = health.get("total_nodes", 0)
    total_edges = health.get("total_edges", 0)
    hausdorff = health.get("global_hausdorff", 0)
    is_fractal = health.get("is_fractal", False)
    mean_energy = health.get("mean_energy", 0)
    energy_std = health.get("energy_std", 0)
    coherence = health.get("coherence_score", 0)
    gap_count = health.get("gap_count", 0)
    entropy_spikes = health.get("entropy_spike_count", 0)
    tick = health.get("tick_number", 0)

    # Energy percentiles
    ep = health.get("energy_percentiles", {})
    p10 = ep.get("p10", 0)
    p25 = ep.get("p25", 0)
    p50 = ep.get("p50", 0)
    p75 = ep.get("p75", 0)
    p90 = ep.get("p90", 0)

    # Fractal health assessment
    if 1.2 <= hausdorff <= 1.8:
        fractal_status = "HEALTHY (self-similar)"
        fractal_icon = "✦"
    elif hausdorff < 1.2:
        fractal_status = "TOO LINEAR (needs branching)"
        fractal_icon = "⚠"
    else:
        fractal_status = "TOO CHAOTIC (needs pruning)"
        fractal_icon = "⚠"

    print(f"""
    Tick:             #{tick}
    Nodes/Edges:      {format_num(total_nodes)} / {format_num(total_edges)}

    ┌─ FRACTAL DIMENSION ──────────────────────────────────┐
    │ Hausdorff:      {hausdorff:.4f}  {fractal_icon} {fractal_status:<30} │
    │ Target Range:   [1.2000 ─── 1.8000]                  │
    │ Position:       {bar(hausdorff - 0.5, 2.0, 40)} │
    │ Is Fractal:     {'YES' if is_fractal else 'NO':<42} │
    └──────────────────────────────────────────────────────┘

    ┌─ ENERGY DISTRIBUTION ────────────────────────────────┐
    │ Mean:           {mean_energy:.6f}                     │
    │ Std Dev:        {energy_std:.6f}                      │
    │ P10:  {p10:.4f}  {bar(p10, 1.0, 20)}                 │
    │ P25:  {p25:.4f}  {bar(p25, 1.0, 20)}                 │
    │ P50:  {p50:.4f}  {bar(p50, 1.0, 20)}  (median)       │
    │ P75:  {p75:.4f}  {bar(p75, 1.0, 20)}                 │
    │ P90:  {p90:.4f}  {bar(p90, 1.0, 20)}                 │
    └──────────────────────────────────────────────────────┘

    ┌─ COHERENCE & GAPS ───────────────────────────────────┐
    │ Coherence:      {coherence:.4f}  {bar(coherence, 1.0, 30)} │
    │ Knowledge Gaps: {gap_count}                           │
    │ Entropy Spikes: {entropy_spikes}                      │
    └──────────────────────────────────────────────────────┘""")


def report_cognitive_dashboard(host, collection):
    """Section 3: Full Cognitive Dashboard — Thermodynamics, ECAN, Gravity."""
    subsection(f"Cognitive State: {collection}")

    dash = api(host, f"/api/agency/dashboard?collection={collection}")
    if not dash or "error" in dash:
        print(f"    No cognitive dashboard data")
        return dash

    # Observation frame / gauges
    obs = api(host, f"/api/agency/observation?collection={collection}")
    if obs and "error" not in obs:
        gauges = obs.get("gauges", {})
        temp = gauges.get("temperature", 0)
        entropy = gauges.get("entropy", 0)
        free_energy = gauges.get("free_energy", 0)
        phase = gauges.get("phase", "?")
        entropy_rate = gauges.get("entropy_rate", 0)
        attn_price = gauges.get("attention_price", 0)
        attn_flow = gauges.get("attention_flow", 0)
        hebbian = gauges.get("hebbian_traces", 0)
        explore = gauges.get("explore_modifier", 0)
        wells = gauges.get("gravity_wells", 0)
        mean_grav = gauges.get("mean_gravity", 0)
        dirty = gauges.get("dirty_ratio", 0)
        total_n = gauges.get("total_nodes", 0)
        mean_e = gauges.get("mean_energy", 0)

        # Phase color
        phase_colors = {"Solid": "❄ FROZEN", "Liquid": "💧 OPTIMAL", "Gas": "🔥 CHAOTIC", "Critical": "⚡ PHASE TRANSITION"}
        phase_display = phase_colors.get(phase, phase)

        print(f"""
    ┌─ THERMODYNAMICS ─────────────────────────────────────┐
    │ Temperature:    {temp:.6f}                            │
    │ Phase:          {phase_display:<42}│
    │ Entropy (S):    {entropy:.6f}                         │
    │ Free Energy(F): {free_energy:.6f}                     │
    │ Entropy Rate:   {entropy_rate:+.6f} (dS/dt)           │
    │ Explore Mod:    {explore:.4f}                         │
    └──────────────────────────────────────────────────────┘

    ┌─ ATTENTION ECONOMY (ECAN) ───────────────────────────┐
    │ Price (inflation): {attn_price:.4f}                   │
    │ Total Flow:        {attn_flow:.4f}                    │
    │ Hebbian Traces:    {hebbian}                          │
    └──────────────────────────────────────────────────────┘

    ┌─ SEMANTIC GRAVITY ───────────────────────────────────┐
    │ Gravity Wells:     {wells}                            │
    │ Mean Force:        {mean_grav:.6f}                    │
    └──────────────────────────────────────────────────────┘""")

    return dash


def report_observer(host, collection):
    """Section 4: Observer Identity — The graph's consciousness."""
    subsection(f"Observer (Consciousness): {collection}")

    obs = api(host, f"/api/agency/observer?collection={collection}")
    if not obs or "error" in obs:
        print(f"    No observer data")
        return

    obs_id = obs.get("observer_id", "?")
    energy = obs.get("energy", 0)
    depth = obs.get("depth", 0)
    hausdorff = obs.get("hausdorff_local", 0)

    print(f"""
    Observer ID:    {obs_id}
    Energy:         {energy:.4f}  {bar(energy, 1.0, 30)}
    Depth:          {depth:.4f}  (distance from center)
    Hausdorff:      {hausdorff:.4f}""")


def report_shatter(host, collection):
    """Section 5: Shatter Protocol — Super-node splitting."""
    subsection(f"Shatter Protocol: {collection}")

    sh = api(host, f"/api/agency/shatter?collection={collection}")
    if not sh or "error" in sh:
        print(f"    No shatter data")
        return

    scanned = sh.get("nodes_scanned", 0)
    super_nodes = sh.get("super_nodes_detected", 0)
    plans = sh.get("plans_emitted", 0)
    ghosts = sh.get("ghost_nodes", 0)
    avatars = sh.get("avatar_nodes", 0)
    largest = sh.get("largest_degree", 0)
    avg_deg = sh.get("avg_degree", 0)

    print(f"""
    Nodes Scanned:      {format_num(scanned)}
    Super-Nodes:        {super_nodes} (degree > 500)
    Shatter Plans:      {plans}
    Ghost Nodes:        {ghosts} (phantomized)
    Avatar Nodes:       {avatars}
    Largest Degree:     {largest}
    Avg Degree:         {avg_deg:.2f}""")


def report_healing(host, collection):
    """Section 6: Self-Healing — Boundary drift, orphans, dead edges."""
    subsection(f"Self-Healing: {collection}")

    heal = api(host, f"/api/agency/healing?collection={collection}")
    if not heal or "error" in heal:
        print(f"    No healing data")
        return

    scanned = heal.get("nodes_scanned", 0)
    boundary = heal.get("boundary_drift_count", 0)
    orphans = heal.get("orphan_count", 0)
    dead_edges = heal.get("dead_edge_count", 0)
    exhausted = heal.get("exhausted_count", 0)
    ghosts = heal.get("ghost_count", 0)
    live = heal.get("live_count", 0)
    phantom_ratio = heal.get("phantom_ratio", 0)

    total_issues = boundary + orphans + dead_edges + exhausted
    health_pct = (1.0 - total_issues / max(scanned, 1)) * 100

    print(f"""
    Nodes Scanned:      {format_num(scanned)}
    Live Nodes:         {format_num(live)}
    Ghost Nodes:        {ghosts}
    Phantom Ratio:      {phantom_ratio:.4f}

    ┌─ ISSUES DETECTED ────────────────────────────────────┐
    │ Boundary Drift:   {boundary:>6}  (nodes escaping ball)       │
    │ Orphan Nodes:     {orphans:>6}  (no connections)             │
    │ Dead Edges:       {dead_edges:>6}  (endpoints missing)        │
    │ Exhausted Nodes:  {exhausted:>6}  (energy = 0)               │
    │ Total Issues:     {total_issues:>6}                          │
    │ Health Score:     {health_pct:.1f}%  {bar(health_pct, 100, 30)} │
    └──────────────────────────────────────────────────────┘""")


def report_desires(host, collection):
    """Section 7: Desires — What the graph wants."""
    subsection(f"Desires (Motor de Desejo): {collection}")

    desires = api(host, f"/api/agency/desires?collection={collection}")
    if not desires or "error" in desires:
        print(f"    No desires data")
        return

    count = desires.get("count", 0)
    items = desires.get("desires", [])

    print(f"    Pending Desires: {count}")
    for d in items[:10]:
        print(f"    → {d}")


def report_evolution(host, collection):
    """Section 8: L-System Evolution — Rule generation changes."""
    subsection(f"L-System Evolution: {collection}")

    evo = api(host, f"/api/agency/evolution?collection={collection}")
    if not evo or "error" in evo:
        print(f"    No evolution data")
        return

    gen = evo.get("generation", 0)
    strategy = evo.get("last_strategy", "?")
    fitness = evo.get("fitness_history", [])

    print(f"""
    Generation:     {gen}
    Last Strategy:  {strategy}
    Fitness History: {sparkline(fitness[-30:]) if fitness else 'no data'}""")
    if fitness:
        print(f"    Latest Fitness: {fitness[-1]:.6f}")
        if len(fitness) > 1:
            delta = fitness[-1] - fitness[0]
            print(f"    Total Change:   {delta:+.6f} ({'improving' if delta > 0 else 'declining'})")


def report_narrative(host, collection):
    """Section 9: Narrative — The graph's story of itself."""
    subsection(f"Narrative: {collection}")

    narr = api(host, f"/api/agency/narrative?collection={collection}")
    if not narr or "error" in narr:
        print(f"    No narrative available")
        return

    text = narr.get("narrative", "")
    if text:
        # Word-wrap at 60 chars
        words = text.split()
        lines = []
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > 64:
                lines.append(line)
                line = "    "
            line += w + " "
        if line.strip():
            lines.append(line)
        for l in lines[:20]:
            print(l)


def report_algorithms(host, collection):
    """Section 10: Graph Algorithms — PageRank, Communities, Centrality."""
    subsection(f"Graph Algorithms: {collection}")

    # PageRank — Top influential concepts
    pr = api(host, f"/api/algo/pagerank?collection={collection}&iterations=20", timeout=60)
    if pr and "error" not in pr:
        scores = pr.get("scores", [])
        duration = pr.get("duration_ms", 0)
        iterations = pr.get("iterations", 0)
        converged = pr.get("converged", False)

        print(f"\n    PageRank (iters={iterations}, {'converged' if converged else 'max-iters'}, {duration}ms)")
        if scores:
            # Need to resolve node IDs to names
            top = scores[:20]
            print(f"    {'Rank':<6} {'Score':>10} {'Node ID'}")
            print(f"    {'─'*6} {'─'*10} {'─'*36}")
            for i, s in enumerate(top):
                nid = s.get("node_id", "?")
                score = s.get("score", 0)
                print(f"    #{i+1:<5} {score:>10.6f} {nid}")

    # Louvain Communities
    louv = api(host, f"/api/algo/louvain?collection={collection}&resolution=1.0", timeout=60)
    if louv and "error" not in louv:
        n_communities = louv.get("community_count", 0)
        modularity = louv.get("modularity", 0)
        largest = louv.get("largest_size", 0)
        duration = louv.get("duration_ms", 0)

        print(f"""
    Louvain Communities ({duration}ms)
    Communities:   {n_communities}
    Modularity:    {modularity:.4f}  {'(excellent)' if modularity > 0.4 else '(moderate)' if modularity > 0.2 else '(weak)'}
    Largest:       {largest} nodes""")

        # Community size distribution
        assignments = louv.get("communities", louv.get("assignments", []))
        if assignments:
            community_sizes = defaultdict(int)
            for a in assignments:
                cid = a.get("community_id", 0)
                community_sizes[cid] += 1
            sizes = sorted(community_sizes.values(), reverse=True)
            top_sizes = sizes[:10]
            print(f"    Size Distribution: {' '.join(str(s) for s in top_sizes)}{'...' if len(sizes) > 10 else ''}")
            print(f"    Size Sparkline:    {sparkline(top_sizes)}")

    # Betweenness — Bridge nodes
    bet = api(host, f"/api/algo/betweenness?collection={collection}&sample=200", timeout=60)
    if bet and "error" not in bet:
        scores = bet.get("scores", [])
        duration = bet.get("duration_ms", 0)

        print(f"\n    Betweenness Centrality (bridges, {duration}ms)")
        top = scores[:10]
        for i, s in enumerate(top):
            nid = s.get("node_id", "?")
            score = s.get("score", 0)
            print(f"    #{i+1}: {score:.6f} → {nid}")

    # Triangles — Clustering density
    tri = api(host, f"/api/algo/triangles?collection={collection}", timeout=60)
    if tri and "error" not in tri:
        count = tri.get("count", 0)
        print(f"\n    Triangle Count: {format_num(count)}")

    # WCC — Connectivity
    wcc = api(host, f"/api/algo/wcc?collection={collection}", timeout=60)
    if wcc and "error" not in wcc:
        components = wcc.get("component_count", 0)
        largest = wcc.get("largest_component_size", 0)
        duration = wcc.get("duration_ms", 0)

        print(f"""
    Weak Connected Components ({duration}ms)
    Components:    {components}
    Largest:       {format_num(largest)} nodes
    Fragmentation: {'CONNECTED' if components == 1 else f'{components} fragments'}""")

    # Degree — Hub nodes
    deg = api(host, f"/api/algo/degree?collection={collection}&direction=both", timeout=60)
    if deg and "error" not in deg:
        scores = deg.get("scores", [])
        if scores:
            top = scores[:10]
            print(f"\n    Top-10 Hub Nodes (by degree)")
            for i, s in enumerate(top):
                nid = s.get("node_id", "?")
                score = s.get("score", 0)
                print(f"    #{i+1}: degree={int(score)} → {nid}")


def report_node_details(host, collection, node_ids):
    """Resolve node IDs to readable names."""
    names = {}
    for nid in node_ids[:50]:  # limit API calls
        node = api(host, f"/api/node/{nid}?collection={collection}")
        if node and "error" not in node:
            content = node.get("content", {})
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except:
                    pass
            name = content.get("name", nid[:12]) if isinstance(content, dict) else nid[:12]
            energy = node.get("energy", 0)
            depth = node.get("depth", 0)
            ntype = node.get("node_type", "?")
            names[nid] = {"name": name, "energy": energy, "depth": depth, "type": ntype}
    return names


def report_enriched_algorithms(host, collection):
    """Enhanced algorithm report with resolved node names."""
    subsection(f"Top Concepts (Enriched): {collection}")

    # PageRank top nodes
    pr = api(host, f"/api/algo/pagerank?collection={collection}&iterations=20", timeout=60)
    if not pr or "error" in pr:
        print(f"    PageRank not available")
        return

    scores = pr.get("scores", [])[:15]
    if not scores:
        return

    node_ids = [s["node_id"] for s in scores]
    names = report_node_details(host, collection, node_ids)

    print(f"\n    {'Rank':<5} {'PageRank':>10} {'Energy':>8} {'Depth':>7} {'Name'}")
    print(f"    {'─'*5} {'─'*10} {'─'*8} {'─'*7} {'─'*35}")

    for i, s in enumerate(scores):
        nid = s["node_id"]
        score = s.get("score", 0)
        info = names.get(nid, {})
        name = info.get("name", nid[:20])
        energy = info.get("energy", 0)
        depth = info.get("depth", 0)
        print(f"    #{i+1:<4} {score:>10.6f} {energy:>8.4f} {depth:>7.4f} {name[:35]}")


def report_collection_deep(host, collection):
    """Full deep report for a single collection."""
    section(f"COLLECTION: {collection}", "▓")

    # Basic info
    graph = api(host, f"/api/graph?collection={collection}&limit=1")
    if graph and "error" not in graph:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

    # Agency subsystems
    report_agency_health(host, collection)
    report_cognitive_dashboard(host, collection)
    report_observer(host, collection)
    report_shatter(host, collection)
    report_healing(host, collection)
    report_desires(host, collection)
    report_evolution(host, collection)
    report_narrative(host, collection)

    # Algorithms
    report_algorithms(host, collection)
    report_enriched_algorithms(host, collection)


def report_cross_collection(host, collections):
    """Cross-collection comparison."""
    section("CROSS-COLLECTION COMPARISON")

    data = []
    for c in collections:
        info = {"name": c["name"]}
        info["nodes"] = c.get("node_count", 0)
        info["edges"] = c.get("edge_count", 0)
        info["dim"] = c.get("dim", 0)
        info["metric"] = c.get("metric", "?")

        # Get health
        health = api(host, f"/api/agency/health/latest?collection={c['name']}")
        if health and "error" not in health:
            info["hausdorff"] = health.get("global_hausdorff", 0)
            info["mean_energy"] = health.get("mean_energy", 0)
            info["coherence"] = health.get("coherence_score", 0)
            info["gaps"] = health.get("gap_count", 0)
        else:
            info["hausdorff"] = 0
            info["mean_energy"] = 0
            info["coherence"] = 0
            info["gaps"] = 0

        data.append(info)

    # Only show collections with nodes
    data = [d for d in data if d["nodes"] > 0]
    data.sort(key=lambda d: d["nodes"], reverse=True)

    print(f"\n  {'Collection':<28} {'Nodes':>7} {'Edges':>7} {'E/N':>5} {'Hausdorff':>10} {'Energy':>8} {'Coherence':>10} {'Gaps':>5}")
    print(f"  {'─'*28} {'─'*7} {'─'*7} {'─'*5} {'─'*10} {'─'*8} {'─'*10} {'─'*5}")

    for d in data:
        en = d["edges"] / max(d["nodes"], 1)
        print(f"  {d['name']:<28} {format_num(d['nodes']):>7} {format_num(d['edges']):>7} {en:>5.2f} {d['hausdorff']:>10.4f} {d['mean_energy']:>8.4f} {d['coherence']:>10.4f} {d['gaps']:>5}")

    # Find healthiest / sickest
    active = [d for d in data if d["hausdorff"] > 0]
    if active:
        healthiest = max(active, key=lambda d: d["coherence"])
        sickest = min(active, key=lambda d: d["coherence"])
        most_energetic = max(active, key=lambda d: d["mean_energy"])
        most_fractal = min(active, key=lambda d: abs(d["hausdorff"] - 1.5))

        print(f"""
  ┌─ RANKINGS ─────────────────────────────────────────────┐
  │ Most Coherent:    {healthiest['name']:<38} │
  │ Least Coherent:   {sickest['name']:<38} │
  │ Most Energetic:   {most_energetic['name']:<38} │
  │ Most Fractal:     {most_fractal['name']:<38} │
  └────────────────────────────────────────────────────────┘""")


def report_growth_analysis(host, collection):
    """Analyze what the Agency has grown/created."""
    subsection(f"Growth & Discovery Analysis: {collection}")

    # Sample nodes to check types and L-System generations
    graph = api(host, f"/api/graph?collection={collection}&limit=5000", timeout=60)
    if not graph or "error" in graph:
        print(f"    Cannot fetch graph data")
        return

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        print(f"    Empty collection")
        return

    # Analyze node types
    type_counts = defaultdict(int)
    energy_by_type = defaultdict(list)
    depth_by_type = defaultdict(list)
    lsystem_gen_counts = defaultdict(int)
    total_energy = 0
    low_energy = 0
    high_energy = 0

    for n in nodes:
        ntype = n.get("node_type", "?")
        energy = n.get("energy", 0)
        depth = n.get("depth", 0)
        gen = n.get("lsystem_generation", n.get("generation", 0))

        type_counts[ntype] += 1
        energy_by_type[ntype].append(energy)
        depth_by_type[ntype].append(depth)
        lsystem_gen_counts[gen] += 1
        total_energy += energy

        if energy < 0.1:
            low_energy += 1
        elif energy > 0.8:
            high_energy += 1

    # Analyze edge types
    edge_type_counts = defaultdict(int)
    for e in edges:
        etype = e.get("edge_type", "?")
        edge_type_counts[etype] += 1

    print(f"""
    Total Nodes Sampled: {len(nodes)}
    Total Edges Sampled: {len(edges)}

    ┌─ NODE TYPE DISTRIBUTION ─────────────────────────────┐""")
    for ntype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / len(nodes) * 100
        avg_e = sum(energy_by_type[ntype]) / len(energy_by_type[ntype])
        print(f"    │ {ntype:<20} {count:>6} ({pct:>5.1f}%) avg_energy={avg_e:.4f} │")
    print(f"    └──────────────────────────────────────────────────┘")

    print(f"""
    ┌─ EDGE TYPE DISTRIBUTION ─────────────────────────────┐""")
    for etype, count in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
        pct = count / len(edges) * 100 if edges else 0
        print(f"    │ {etype:<20} {count:>6} ({pct:>5.1f}%)                   │")
    print(f"    └──────────────────────────────────────────────────┘")

    # L-System generation analysis
    if any(g > 0 for g in lsystem_gen_counts):
        print(f"""
    ┌─ L-SYSTEM GENERATIONS (Agency Discoveries) ──────────┐""")
        for gen in sorted(lsystem_gen_counts.keys()):
            count = lsystem_gen_counts[gen]
            label = "ORIGINAL (inserted)" if gen == 0 else f"GEN {gen} (Agency created)"
            print(f"    │ {label:<35} {count:>6} nodes      │")
        print(f"    └──────────────────────────────────────────────────┘")

        agency_nodes = sum(c for g, c in lsystem_gen_counts.items() if g > 0)
        print(f"\n    Agency Discoveries: {agency_nodes} nodes created autonomously!")

    # Energy health
    print(f"""
    ┌─ ENERGY HEALTH ──────────────────────────────────────┐
    │ Total Energy:     {total_energy:.2f}                  │
    │ Mean Energy:      {total_energy/max(len(nodes),1):.6f}│
    │ Low Energy (<0.1): {low_energy:>5} ({low_energy/max(len(nodes),1)*100:.1f}%)              │
    │ High Energy (>0.8):{high_energy:>5} ({high_energy/max(len(nodes),1)*100:.1f}%)             │
    └──────────────────────────────────────────────────────┘""")


def report_counterfactual(host, collection):
    """Counterfactual analysis — what would happen if we removed top nodes."""
    subsection(f"Counterfactual Impact Analysis: {collection}")

    # Get top PageRank node
    pr = api(host, f"/api/algo/pagerank?collection={collection}&iterations=20", timeout=60)
    if not pr or "error" in pr:
        return

    scores = pr.get("scores", [])[:3]
    for s in scores:
        nid = s["node_id"]
        cf = api(host, f"/api/agency/counterfactual/remove/{nid}?collection={collection}", timeout=30)
        if cf and "error" not in cf:
            delta = cf.get("mean_energy_delta", 0)
            radius = cf.get("affected_radius", 0)
            print(f"    If removed {nid[:20]}... → energy delta: {delta:+.6f}, affected radius: {radius}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NietzscheDB Super Report")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Dashboard host")
    parser.add_argument("--collections", nargs="*", default=None,
                        help="Collections to analyze (default: all galaxy collections)")
    parser.add_argument("--all", action="store_true", help="Analyze ALL collections")
    parser.add_argument("--quick", action="store_true", help="Quick mode (skip algorithms)")
    args = parser.parse_args()

    host = args.host
    start = time.time()

    print(f"""
{'█'*70}
{'█'*70}
██                                                              ██
██   NietzscheDB SUPER REPORT — Complete Brain Scan             ██
██   Thermodynamics · Agency · Dreams · Growth · Pruning        ██
██   Energy · Fractals · Attention · Gravity · Coherence        ██
██                                                              ██
{'█'*70}
{'█'*70}
""")

    # 1. Global Overview
    collections = report_overview(host)
    if not isinstance(collections, list):
        print("\n[!] Cannot reach NietzscheDB dashboard. Aborting.")
        return

    # 2. Cross-collection comparison
    report_cross_collection(host, collections)

    # 3. Determine which collections to deep-dive
    if args.all:
        target_collections = [c["name"] for c in collections if c.get("node_count", 0) > 0]
    elif args.collections:
        target_collections = args.collections
    else:
        target_collections = GALAXY_COLLECTIONS

    # 4. Deep dive each collection
    for cname in target_collections:
        report_collection_deep(host, cname)
        report_growth_analysis(host, cname)
        if not args.quick:
            report_counterfactual(host, cname)

    # 5. Final summary
    elapsed = time.time() - start

    section("REPORT COMPLETE")
    print(f"""
  Collections Analyzed: {len(target_collections)}
  Total Time:           {elapsed:.1f}s
  Generated:            {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

  Next Steps:
  → Check Dashboard: {host}
  → Re-run with --all for full scan of {len(collections)} collections
  → The Agency engine continues evolving autonomously
""")


if __name__ == "__main__":
    main()
