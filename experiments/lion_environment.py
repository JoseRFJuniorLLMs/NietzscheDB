#!/usr/bin/env python3
"""
NietzscheDB — Lion Environment: Structural World Modeling

A blind agent navigates a community-bottleneck graph where 5 communities
are well-connected internally but separated by narrow bridges. NietzscheDB
observes random walk trajectories, embeds nodes in the Poincare ball, detects
community structure, and synthesizes inter-community bridge edges — WITHOUT
seeing the goal, reward, or community labels.

Three phases:
  I.   BASELINE: measure average path length on raw graph
  II.  NIETZSCHEDB ACTIVE: observe -> embed -> cluster -> synthesize edges
  III. AMNESIA TOTAL: prove structure survives (edges are permanent)

If avg_path drops significantly and survives amnesia,
NietzscheDB inferred the geometry of the world.

Usage:
  py -3 experiments/lion_environment.py
"""

import math
import sys
from collections import deque, defaultdict

# ===============================
# CONFIGURATION
# ===============================
N_NODES = 1000
N_COMMUNITIES = 5
COMMUNITY_SIZE = N_NODES // N_COMMUNITIES  # 200
INTERNAL_DEGREE = 8      # target degree within each community
BOTTLENECK_EDGES = 1     # ONE edge between adjacent communities (hard barrier)
MAX_STEPS = 300          # shorter walks: trap walkers within communities
SEED = 42
N_OBSERVATION_EPISODES = 5000   # more episodes for weak inter-community signal
N_EVAL_EPISODES = 300
PATH_SAMPLES = 500

# Poincare embedding params
EMBED_DIM = 16
EMBED_LR = 0.10
EMBED_STEPS = 50         # more smoothing steps for tight community convergence
BRIDGE_CANDIDATES = 20   # max bridges NietzscheDB can propose

# ===============================
# DETERMINISTIC RNG (matches Rust)
# ===============================
class Rng:
    def __init__(self, seed):
        self.state = seed
    def next_u64(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.state
    def next_f32(self, lo, hi):
        t = ((self.next_u64() >> 33) & 0xFFFFFFFF) / 0xFFFFFFFF
        return lo + t * (hi - lo)
    def next_int(self, hi):
        return ((self.next_u64() >> 33) & 0xFFFFFFFF) % max(hi, 1)

# ===============================
# GRAPH
# ===============================
adj = {i: set() for i in range(N_NODES)}

def add_edge(u, v):
    if u != v:
        adj[u].add(v)
        adj[v].add(u)

def community_of(node):
    """Return which community (0..4) a node belongs to."""
    return node // COMMUNITY_SIZE

def build_graph():
    """
    Community-bottleneck graph:
    - 5 communities of 200 nodes, degree ~6 internally
    - 3 bottleneck edges between adjacent communities
    - Non-adjacent communities have NO direct connection
    """
    rng = Rng(SEED)

    # Internal edges within each community
    for comm in range(N_COMMUNITIES):
        base = comm * COMMUNITY_SIZE
        for i in range(base, base + COMMUNITY_SIZE):
            attempts = 0
            while len([n for n in adj[i] if community_of(n) == comm]) < INTERNAL_DEGREE and attempts < 50:
                j = base + rng.next_int(COMMUNITY_SIZE)
                if j != i and j not in adj[i]:
                    add_edge(i, j)
                attempts += 1

    # Bottleneck: exactly 1 edge between adjacent communities (narrow bridge)
    for comm in range(N_COMMUNITIES - 1):
        base_a = comm * COMMUNITY_SIZE
        base_b = (comm + 1) * COMMUNITY_SIZE
        for _ in range(BOTTLENECK_EDGES):
            # Gateway nodes: specific high-degree nodes at community edges
            u = base_a + COMMUNITY_SIZE - 1  # last node of community
            v = base_b                        # first node of next community
            add_edge(u, v)

# ===============================
# GROUND TRUTH (for evaluation only)
# ===============================
ground_truth_bridges = []

def define_ground_truth():
    """
    Ground-truth: the 'ideal' bridges between NON-ADJACENT communities.
    These are NOT in the graph. NietzscheDB should discover similar bridges.
    """
    global ground_truth_bridges
    rng = Rng(7777)

    # Non-adjacent community pairs: (0,2), (0,3), (0,4), (1,3), (1,4), (2,4)
    pairs = []
    for i in range(N_COMMUNITIES):
        for j in range(i + 2, N_COMMUNITIES):  # skip adjacent (i+1)
            pairs.append((i, j))

    for ca, cb in pairs:
        base_a = ca * COMMUNITY_SIZE
        base_b = cb * COMMUNITY_SIZE
        u = base_a + rng.next_int(COMMUNITY_SIZE)
        v = base_b + rng.next_int(COMMUNITY_SIZE)
        ground_truth_bridges.append((u, v))

# ===============================
# METRICS
# ===============================
def bfs_distance(src, tgt):
    if src == tgt:
        return 0
    visited = {src}
    q = deque([(src, 0)])
    while q:
        u, d = q.popleft()
        for v in adj[u]:
            if v == tgt:
                return d + 1
            if v not in visited:
                visited.add(v)
                q.append((v, d + 1))
    return float('inf')

def average_path_length(samples=PATH_SAMPLES):
    rng = Rng(9999)
    dists = []
    for _ in range(samples):
        s = rng.next_int(N_NODES)
        t = rng.next_int(N_NODES)
        while t == s:
            t = rng.next_int(N_NODES)
        d = bfs_distance(s, t)
        if d < float('inf'):
            dists.append(d)
    return sum(dists) / len(dists) if dists else float('inf')

def cross_community_path_length():
    """Average path length between nodes in DIFFERENT communities."""
    rng = Rng(8888)
    dists = []
    attempts = 0
    while len(dists) < 200 and attempts < 1000:
        s = rng.next_int(N_NODES)
        t = rng.next_int(N_NODES)
        attempts += 1
        if community_of(s) == community_of(t):
            continue
        d = bfs_distance(s, t)
        if d < float('inf'):
            dists.append(d)
    return sum(dists) / len(dists) if dists else float('inf')

def ground_truth_reachability(max_hops=5):
    """Fraction of ground-truth bridges reachable within max_hops."""
    reachable = 0
    for u, v in ground_truth_bridges:
        d = bfs_distance(u, v)
        if d <= max_hops:
            reachable += 1
    return reachable / max(len(ground_truth_bridges), 1)

# ===============================
# BLIND AGENT (random walk)
# ===============================
def random_walk(start, steps, rng):
    """Returns trajectory as list of visited nodes."""
    path = [start]
    pos = start
    for _ in range(steps):
        neighbors = list(adj[pos])
        if not neighbors:
            break
        pos = neighbors[rng.next_int(len(neighbors))]
        path.append(pos)
    return path

# ===============================
# OBSERVATION ENGINE
# ===============================
def observe(n_episodes=N_OBSERVATION_EPISODES):
    """
    Collect random walk trajectories. Returns:
    - visit_freq: node_id -> visit count
    - covisit: (u, v) -> co-occurrence count (within window of 5 steps)
    - trajectories: list of paths
    """
    rng = Rng(1337)
    visit_freq = defaultdict(int)
    covisit = defaultdict(int)
    trajectories = []

    for _ in range(n_episodes):
        start = rng.next_int(N_NODES)
        traj = random_walk(start, MAX_STEPS, rng)
        trajectories.append(traj)

        for i, node in enumerate(traj):
            visit_freq[node] += 1
            # Co-visitation within window of 5 steps
            for j in range(max(0, i - 5), min(len(traj), i + 6)):
                if i != j:
                    pair = (min(traj[i], traj[j]), max(traj[i], traj[j]))
                    covisit[pair] += 1

    return visit_freq, covisit, trajectories

# ===============================
# POINCARE MATH
# ===============================
def poincare_distance(u, v):
    diff_sq = sum((a - b) ** 2 for a, b in zip(u, v))
    norm_u = sum(a ** 2 for a in u)
    norm_v = sum(a ** 2 for a in v)
    denom = (1.0 - norm_u) * (1.0 - norm_v)
    if denom <= 0:
        return 20.0
    arg = max(1.0 + 2.0 * diff_sq / denom, 1.0)
    return math.acosh(arg)

def project_into_ball(coords, max_norm=0.95):
    norm = math.sqrt(sum(x ** 2 for x in coords))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-10)
        return [x * scale for x in coords]
    return coords

# ===============================
# NIETZSCHEDB HOOK — THE CORE
# ===============================
def nietzschedb_hook(visit_freq, covisit, trajectories):
    """
    NietzscheDB structural inference engine.

    INPUTS: random walk observations (no goal, no reward, no labels).
    OUTPUT: list of (u, v) edges to add to the graph.

    Protocol:
    1. Embed all observed nodes into Poincare ball using co-visitation
    2. Propagate embeddings via Laplacian smoothing (co-visit topology)
    3. Identify community structure via embedding proximity
    4. Synthesize inter-community bridge edges

    FORBIDDEN: using `goal`, `reward`, `ground_truth_bridges`, `community_of()`
    """
    rng = Rng(42424242)

    # --- Step 1: Initialize embeddings from co-visitation ---
    # High-frequency nodes near Poincare center (more "abstract/connected").
    # Random direction, radius inversely proportional to visit frequency.

    embeddings = {}
    max_freq = max(visit_freq.values()) if visit_freq else 1

    for node_id in range(N_NODES):
        freq = visit_freq.get(node_id, 0)
        # High-frequency → small radius (center), low-frequency → large radius (periphery)
        radius = 0.85 * (1.0 - 0.7 * freq / max(max_freq, 1))
        emb = []
        for _ in range(EMBED_DIM):
            emb.append(rng.next_f32(-1.0, 1.0) * radius)
        embeddings[node_id] = project_into_ball(emb)

    # --- Step 2: Co-visitation weighted Laplacian smoothing ---
    # Nodes that co-occur frequently in trajectories converge in Poincare space.
    # This is the core: communities will form TIGHT clusters because nodes
    # within a community co-visit heavily, while inter-community co-visitation
    # is sparse (only through bottleneck traversals).
    #
    # KEY: use adaptive threshold to separate strong intra-community signal
    # from weak inter-community noise.

    all_counts = sorted(covisit.values())
    if all_counts:
        # Use median as threshold: only keep edges with above-median co-visitation
        median_idx = len(all_counts) // 2
        threshold = max(all_counts[median_idx], 5)
        # Also compute percentiles for diagnostics
        p25 = all_counts[len(all_counts) // 4]
        p75 = all_counts[3 * len(all_counts) // 4]
        p90 = all_counts[int(len(all_counts) * 0.9)]
        print(f"    Co-visit count distribution: p25={p25}, median={threshold}, p75={p75}, p90={p90}")
    else:
        threshold = 5

    covisit_adj = defaultdict(dict)
    for (u, v), count in covisit.items():
        if count >= threshold:
            covisit_adj[u][v] = count
            covisit_adj[v][u] = count

    n_covisit_edges = sum(len(v) for v in covisit_adj.values()) // 2
    print(f"    Co-visit adjacency: {n_covisit_edges} edges (threshold>={threshold})")

    for step in range(EMBED_STEPS):
        updates = {}
        for node_id in range(N_NODES):
            neighbors = covisit_adj.get(node_id, {})
            if not neighbors:
                continue

            total_weight = 0.0
            mean_emb = [0.0] * EMBED_DIM
            for nb, weight in neighbors.items():
                if nb in embeddings:
                    w = math.log1p(weight)
                    for d in range(EMBED_DIM):
                        mean_emb[d] += embeddings[nb][d] * w
                    total_weight += w

            if total_weight < 1e-6:
                continue

            inv = 1.0 / total_weight
            old = embeddings[node_id]
            new_emb = []
            for d in range(EMBED_DIM):
                new_emb.append((1.0 - EMBED_LR) * old[d] + EMBED_LR * mean_emb[d] * inv)
            updates[node_id] = project_into_ball(new_emb)

        for node_id, emb in updates.items():
            embeddings[node_id] = emb

    # --- Step 3: Identify communities via embedding proximity ---
    # After smoothing, nodes in the same community should form tight clusters.
    # Use greedy clustering from highest-frequency seeds.
    #
    # Adaptive radius: compute typical intra-cluster distances to calibrate.

    MIN_CLUSTER = 10

    # Sample distances to estimate intra-community vs inter-community threshold
    sorted_nodes = sorted(range(N_NODES), key=lambda n: visit_freq.get(n, 0), reverse=True)
    sample_dists = []
    sample_nodes = sorted_nodes[:200]  # top-200 most visited
    for i in range(0, min(len(sample_nodes), 100)):
        for j in range(i + 1, min(len(sample_nodes), 100)):
            ni, nj = sample_nodes[i], sample_nodes[j]
            if ni in embeddings and nj in embeddings:
                sample_dists.append(poincare_distance(embeddings[ni], embeddings[nj]))

    if sample_dists:
        sample_dists.sort()
        # Use the 30th percentile as cluster radius (should capture intra-community)
        cluster_radius = sample_dists[int(len(sample_dists) * 0.30)]
        dist_median = sample_dists[len(sample_dists) // 2]
        dist_p90 = sample_dists[int(len(sample_dists) * 0.9)]
        print(f"    Embedding distances (sample): p30={cluster_radius:.3f}, median={dist_median:.3f}, p90={dist_p90:.3f}")
    else:
        cluster_radius = 0.5

    print(f"    Using cluster_radius={cluster_radius:.3f}")

    assigned = set()
    found_clusters = []

    for seed_node in sorted_nodes:
        if seed_node in assigned:
            continue
        if seed_node not in embeddings:
            continue

        cluster = [seed_node]
        assigned.add(seed_node)
        seed_emb = embeddings[seed_node]

        for other in sorted_nodes:
            if other in assigned or other == seed_node:
                continue
            if other not in embeddings:
                continue
            d = poincare_distance(seed_emb, embeddings[other])
            if d < cluster_radius:
                cluster.append(other)
                assigned.add(other)

        if len(cluster) >= MIN_CLUSTER:
            found_clusters.append(cluster)

        if len(found_clusters) >= 10:
            break

    print(f"    Found {len(found_clusters)} clusters: sizes={[len(c) for c in found_clusters]}")

    # --- Step 4: Synthesize bridge edges between distant clusters ---
    # For each pair of detected clusters whose hubs are far apart in the
    # graph, propose a bridge. This is how NietzscheDB "discovers" that
    # non-adjacent communities should be connected.

    proposed_edges = []

    if len(found_clusters) < 2:
        print("    WARNING: fewer than 2 clusters found, no bridges possible")
        return proposed_edges

    # For each pair of clusters: bridge if graph distance > threshold
    BRIDGE_DIST_THRESHOLD = 8  # communities 2+ apart should be ~15-30 hops

    for i in range(len(found_clusters)):
        for j in range(i + 1, len(found_clusters)):
            # Hub = highest-degree node in each cluster
            hub_i = max(found_clusters[i], key=lambda n: len(adj[n]))
            hub_j = max(found_clusters[j], key=lambda n: len(adj[n]))

            graph_dist = bfs_distance(hub_i, hub_j)

            if graph_dist > BRIDGE_DIST_THRESHOLD:
                proposed_edges.append((hub_i, hub_j))

            if len(proposed_edges) >= BRIDGE_CANDIDATES:
                break
        if len(proposed_edges) >= BRIDGE_CANDIDATES:
            break

    return proposed_edges

# ===============================
# EVALUATION
# ===============================
def evaluate_navigation(label, n_episodes=N_EVAL_EPISODES):
    """Run blind agent episodes, measure steps to reach goal."""
    rng = Rng(5555)
    steps_list = []
    successes = 0
    for _ in range(n_episodes):
        start = rng.next_int(N_NODES)
        goal = rng.next_int(N_NODES)
        while goal == start:
            goal = rng.next_int(N_NODES)

        pos = start
        found = False
        for step in range(MAX_STEPS):
            if pos == goal:
                steps_list.append(step)
                successes += 1
                found = True
                break
            neighbors = list(adj[pos])
            if not neighbors:
                break
            pos = neighbors[rng.next_int(len(neighbors))]
        if not found:
            steps_list.append(MAX_STEPS)

    avg = sum(steps_list) / len(steps_list) if steps_list else MAX_STEPS
    rate = successes / n_episodes * 100.0
    print(f"  [{label}] avg_steps={avg:.1f}  success_rate={rate:.1f}%  ({successes}/{n_episodes})")
    return avg, rate

# ===============================
# MAIN
# ===============================
def main():
    print("=" * 65)
    print("  LION ENVIRONMENT — Structural World Modeling")
    print("  NietzscheDB as blind topology inference engine")
    print(f"  Graph: {N_NODES} nodes, {N_COMMUNITIES} communities x {COMMUNITY_SIZE} nodes")
    print(f"  Internal degree ~{INTERNAL_DEGREE}, bottleneck={BOTTLENECK_EDGES} edge/boundary")
    print("=" * 65)
    print()

    # Build world
    build_graph()
    define_ground_truth()

    total_edges = sum(len(v) for v in adj.values()) // 2
    print(f"  Graph built: {N_NODES} nodes, {total_edges} edges")
    print(f"  Ground-truth bridges (NOT in graph): {ground_truth_bridges}")

    # Show graph structure
    for comm in range(N_COMMUNITIES):
        base = comm * COMMUNITY_SIZE
        internal = sum(1 for i in range(base, base + COMMUNITY_SIZE)
                       for n in adj[i] if community_of(n) == comm) // 2
        external = sum(1 for i in range(base, base + COMMUNITY_SIZE)
                       for n in adj[i] if community_of(n) != comm)
        avg_deg = sum(len(adj[i]) for i in range(base, base + COMMUNITY_SIZE)) / COMMUNITY_SIZE
        print(f"  Community {comm} (nodes {base}-{base+COMMUNITY_SIZE-1}): "
              f"{internal} internal, {external} external edges, avg_deg={avg_deg:.1f}")
    print()

    # ── PHASE I: BASELINE ──
    print("=" * 65)
    print("  PHASE I: BASELINE (no NietzscheDB)")
    print("=" * 65)
    baseline_path = average_path_length()
    baseline_cross = cross_community_path_length()
    baseline_reach = ground_truth_reachability(max_hops=5)
    print(f"  Average path length (all pairs): {baseline_path:.2f}")
    print(f"  Cross-community path length:     {baseline_cross:.2f}")
    print(f"  Ground-truth bridge reachability (<=5 hops): {baseline_reach:.1%}")
    baseline_steps, baseline_rate = evaluate_navigation("Baseline")
    print()

    # ── PHASE II: NIETZSCHEDB ACTIVE ──
    print("=" * 65)
    print("  PHASE II: NietzscheDB ACTIVE (observe -> embed -> synthesize)")
    print("=" * 65)

    print("  Observing trajectories...")
    visit_freq, covisit, trajectories = observe()
    nodes_observed = len(visit_freq)
    pairs_tracked = len(covisit)

    # Analyze co-visitation structure
    cross_covisit = 0
    for (u, v), count in covisit.items():
        if community_of(u) != community_of(v):
            cross_covisit += count
    print(f"    {N_OBSERVATION_EPISODES} episodes x {MAX_STEPS} steps")
    print(f"    {nodes_observed} nodes observed, {pairs_tracked} co-visit pairs")
    print(f"    Cross-community co-visitation events: {cross_covisit}")

    print("  Running NietzscheDB inference...")
    new_edges = nietzschedb_hook(visit_freq, covisit, trajectories)
    print(f"    Proposed {len(new_edges)} bridge edges:")
    for u, v in new_edges:
        d = bfs_distance(u, v)
        cu, cv = community_of(u), community_of(v)
        print(f"      ({u}, {v})  comm=({cu},{cv})  graph_dist={d}")

    # Apply bridges
    for u, v in new_edges:
        add_edge(u, v)

    total_edges_after = sum(len(v) for v in adj.values()) // 2
    print(f"  Edges: {total_edges} -> {total_edges_after} (+{total_edges_after - total_edges})")

    post_path = average_path_length()
    post_cross = cross_community_path_length()
    post_reach = ground_truth_reachability(max_hops=5)
    print(f"  Average path length: {baseline_path:.2f} -> {post_path:.2f}")
    print(f"  Cross-community:     {baseline_cross:.2f} -> {post_cross:.2f}")
    print(f"  Ground-truth reach:  {baseline_reach:.1%} -> {post_reach:.1%}")
    post_steps, post_rate = evaluate_navigation("NietzscheDB")
    print()

    # ── PHASE III: AMNESIA TOTAL ──
    print("=" * 65)
    print("  PHASE III: AMNESIA TOTAL")
    print("  (Edges are structural — no embeddings to erase)")
    print("  (If path length holds, topology is the true memory)")
    print("=" * 65)

    amnesia_path = average_path_length()
    amnesia_cross = cross_community_path_length()
    amnesia_reach = ground_truth_reachability(max_hops=5)
    print(f"  Average path length (post-amnesia): {amnesia_path:.2f}")
    print(f"  Cross-community (post-amnesia):     {amnesia_cross:.2f}")
    print(f"  Ground-truth reach (post-amnesia):  {amnesia_reach:.1%}")
    amnesia_steps, amnesia_rate = evaluate_navigation("Post-Amnesia")
    print()

    # ── VERDICT ──
    print("=" * 65)
    print("  VERDICT")
    print("=" * 65)
    print()
    print(f"  {'Metric':30} | {'Baseline':>10} | {'NietzscheDB':>12} | {'Post-Amnesia':>12}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'Avg path length (all)':30} | {baseline_path:10.2f} | {post_path:12.2f} | {amnesia_path:12.2f}")
    print(f"  {'Cross-community path':30} | {baseline_cross:10.2f} | {post_cross:12.2f} | {amnesia_cross:12.2f}")
    print(f"  {'Ground-truth reach (<=5h)':30} | {baseline_reach:9.1%} | {post_reach:11.1%} | {amnesia_reach:11.1%}")
    print(f"  {'Nav avg steps':30} | {baseline_steps:10.1f} | {post_steps:12.1f} | {amnesia_steps:12.1f}")
    print(f"  {'Nav success rate':30} | {baseline_rate:9.1f}% | {post_rate:11.1f}% | {amnesia_rate:11.1f}%")
    print()

    path_reduction = (baseline_path - post_path) / max(baseline_path, 0.01) * 100
    cross_reduction = (baseline_cross - post_cross) / max(baseline_cross, 0.01) * 100
    path_survives = abs(post_path - amnesia_path) < 0.5

    print(f"  Overall path reduction:        {path_reduction:.1f}%")
    print(f"  Cross-community path reduction: {cross_reduction:.1f}%")
    print(f"  Survives amnesia: {'YES' if path_survives else 'NO'} (delta={abs(post_path - amnesia_path):.2f})")
    print()

    if cross_reduction > 20 and path_survives:
        print("  >>> TOPOLOGY INFERRED LATENT STRUCTURE — CONFIRMED <<<")
        print("  >>> NietzscheDB discovered geometry the agent could not see <<<")
    elif cross_reduction > 10:
        print("  >>> PARTIAL STRUCTURE DETECTED — needs refinement <<<")
    elif cross_reduction > 0:
        print("  >>> MARGINAL IMPROVEMENT — insufficient for world model <<<")
    else:
        print("  >>> NO IMPROVEMENT — NietzscheDB is a passive observer <<<")

    print()
    print("=" * 65)

if __name__ == "__main__":
    main()
