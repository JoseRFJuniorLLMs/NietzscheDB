#!/usr/bin/env python3
"""
NietzscheDB — Protocolo de Falsificabilidade

3 Laminas destrutivas contra o resultado do Lion Environment.
O objetivo e DESTRUIR o resultado. Se sobreviver, nao ha mais o que provar.

Fase 1: Cicatrizacao — deletar ponte mais critica, ver se regenera
Fase 2A: Permutacao — embaralhar IDs, ver se topologia e invariante
Fase 2B: Veneno — injetar 50 arestas falsas, ver se sistema imune funciona
Fase 3: Abismo — escalar para 1K, 10K, 100K nos

Uso:
  py -3 experiments/lion_falsificability.py
"""

import math
import sys
import time
import tracemalloc
from collections import deque, defaultdict

# ===============================
# RNG DETERMINISTICO
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
# MATEMATICA POINCARE
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
# GRAFO PARAMETRIZADO
# ===============================
def add_edge(adj, u, v):
    if u != v:
        adj[u].add(v)
        adj[v].add(u)

def remove_edge(adj, u, v):
    adj[u].discard(v)
    adj[v].discard(u)

def community_of(node, community_size):
    return node // community_size

def make_graph(n_nodes, n_communities=5, internal_degree=8, bottleneck=1, seed=42):
    community_size = n_nodes // n_communities
    adj = {i: set() for i in range(n_nodes)}
    rng = Rng(seed)

    for comm in range(n_communities):
        base = comm * community_size
        for i in range(base, base + community_size):
            attempts = 0
            while len([n for n in adj[i] if community_of(n, community_size) == comm]) < internal_degree and attempts < 50:
                j = base + rng.next_int(community_size)
                if j != i and j not in adj[i]:
                    add_edge(adj, i, j)
                attempts += 1

    for comm in range(n_communities - 1):
        base_a = comm * community_size
        base_b = (comm + 1) * community_size
        for _ in range(bottleneck):
            u = base_a + community_size - 1
            v = base_b
            add_edge(adj, u, v)

    return adj

# ===============================
# BFS E METRICAS
# ===============================
def bfs_distance(adj, src, tgt, max_visited=0):
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
                if max_visited > 0 and len(visited) >= max_visited:
                    return float('inf')
    return float('inf')

def average_path_length(adj, n_nodes, samples=500, max_visited=0):
    rng = Rng(9999)
    dists = []
    for _ in range(samples):
        s = rng.next_int(n_nodes)
        t = rng.next_int(n_nodes)
        while t == s:
            t = rng.next_int(n_nodes)
        d = bfs_distance(adj, s, t, max_visited)
        if d < float('inf'):
            dists.append(d)
    return sum(dists) / len(dists) if dists else float('inf')

# ===============================
# RANDOM WALK + OBSERVACAO
# ===============================
def random_walk(adj, start, steps, rng):
    path = [start]
    pos = start
    for _ in range(steps):
        neighbors = list(adj[pos])
        if not neighbors:
            break
        pos = neighbors[rng.next_int(len(neighbors))]
        path.append(pos)
    return path

def observe(adj, n_nodes, n_episodes=5000, max_steps=300):
    rng = Rng(1337)
    visit_freq = defaultdict(int)
    covisit = defaultdict(int)
    trajectories = []

    for _ in range(n_episodes):
        start = rng.next_int(n_nodes)
        traj = random_walk(adj, start, max_steps, rng)
        trajectories.append(traj)
        for i, node in enumerate(traj):
            visit_freq[node] += 1
            for j in range(max(0, i - 5), min(len(traj), i + 6)):
                if i != j:
                    pair = (min(traj[i], traj[j]), max(traj[i], traj[j]))
                    covisit[pair] += 1

    return visit_freq, covisit, trajectories

# ===============================
# HOOK NIETZSCHEDB (PARAMETRIZADO)
# ===============================
def nietzschedb_hook(adj, n_nodes, visit_freq, covisit, embed_dim=16,
                     embed_lr=0.10, embed_steps=50, bridge_candidates=20,
                     elite_pct=0.0, quiet=False):
    rng = Rng(42424242)

    # Passo 1: Inicializar embeddings
    embeddings = {}
    max_freq = max(visit_freq.values()) if visit_freq else 1

    node_pool = range(n_nodes)
    if elite_pct > 0:
        sorted_by_deg = sorted(range(n_nodes), key=lambda n: len(adj[n]), reverse=True)
        top_k = max(int(n_nodes * elite_pct), 100)
        node_pool = sorted_by_deg[:top_k]

    for node_id in node_pool:
        freq = visit_freq.get(node_id, 0)
        radius = 0.85 * (1.0 - 0.7 * freq / max(max_freq, 1))
        emb = [rng.next_f32(-1.0, 1.0) * radius for _ in range(embed_dim)]
        embeddings[node_id] = project_into_ball(emb)

    # Passo 2: Threshold co-visit adaptativo
    all_counts = sorted(covisit.values())
    if all_counts:
        median_idx = len(all_counts) // 2
        threshold = max(all_counts[median_idx], 5)
    else:
        threshold = 5

    covisit_adj_raw = defaultdict(dict)
    for (u, v), count in covisit.items():
        if count >= threshold:
            if u in embeddings or elite_pct == 0:
                covisit_adj_raw[u][v] = count
                covisit_adj_raw[v][u] = count

    # Limitar vizinhos por no (top-K por peso) para performance O(n*K*steps)
    MAX_NEIGHBORS = 30
    covisit_adj = defaultdict(dict)
    for node, neighbors in covisit_adj_raw.items():
        if len(neighbors) <= MAX_NEIGHBORS:
            covisit_adj[node] = neighbors
        else:
            top = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:MAX_NEIGHBORS]
            covisit_adj[node] = dict(top)

    # Passo 3: Suavizacao Laplaciana
    for step in range(embed_steps):
        updates = {}
        for node_id in embeddings:
            neighbors = covisit_adj.get(node_id, {})
            if not neighbors:
                continue
            total_weight = 0.0
            mean_emb = [0.0] * embed_dim
            for nb, weight in neighbors.items():
                if nb in embeddings:
                    w = math.log1p(weight)
                    for d in range(embed_dim):
                        mean_emb[d] += embeddings[nb][d] * w
                    total_weight += w
            if total_weight < 1e-6:
                continue
            inv = 1.0 / total_weight
            old = embeddings[node_id]
            new_emb = [(1.0 - embed_lr) * old[d] + embed_lr * mean_emb[d] * inv for d in range(embed_dim)]
            updates[node_id] = project_into_ball(new_emb)
        for node_id, emb in updates.items():
            embeddings[node_id] = emb

    # Passo 4: Clustering
    MIN_CLUSTER = max(10, n_nodes // 200)
    embed_nodes = list(embeddings.keys())
    sorted_nodes = sorted(embed_nodes, key=lambda n: visit_freq.get(n, 0), reverse=True)

    sample_dists = []
    sample_nodes = sorted_nodes[:min(200, len(sorted_nodes))]
    for i in range(min(len(sample_nodes), 100)):
        for j in range(i + 1, min(len(sample_nodes), 100)):
            ni, nj = sample_nodes[i], sample_nodes[j]
            sample_dists.append(poincare_distance(embeddings[ni], embeddings[nj]))

    if sample_dists:
        sample_dists.sort()
        cluster_radius = sample_dists[int(len(sample_dists) * 0.30)]
    else:
        cluster_radius = 0.5

    assigned = set()
    found_clusters = []

    # Para grafos grandes, comparar apenas com amostra para evitar O(n²)
    MAX_COMPARE = 2000
    compare_pool = sorted_nodes[:MAX_COMPARE] if len(sorted_nodes) > MAX_COMPARE else sorted_nodes

    for seed_node in sorted_nodes:
        if seed_node in assigned:
            continue
        cluster = [seed_node]
        assigned.add(seed_node)
        seed_emb = embeddings[seed_node]
        for other in compare_pool:
            if other in assigned:
                continue
            d = poincare_distance(seed_emb, embeddings[other])
            if d < cluster_radius:
                cluster.append(other)
                assigned.add(other)
        if len(cluster) >= MIN_CLUSTER:
            found_clusters.append(cluster)
        if len(found_clusters) >= 10:
            break

    if not quiet:
        print(f"    {len(found_clusters)} clusters: tamanhos={[len(c) for c in found_clusters]}")

    # Passo 5: Sintetizar pontes
    proposed_edges = []
    if len(found_clusters) < 2:
        return proposed_edges

    BRIDGE_DIST_THRESHOLD = 8
    max_vis = 5000 if n_nodes >= 10000 else 0

    for i in range(len(found_clusters)):
        for j in range(i + 1, len(found_clusters)):
            hub_i = max(found_clusters[i], key=lambda n: len(adj[n]))
            hub_j = max(found_clusters[j], key=lambda n: len(adj[n]))
            graph_dist = bfs_distance(adj, hub_i, hub_j, max_vis)
            if graph_dist > BRIDGE_DIST_THRESHOLD:
                proposed_edges.append((hub_i, hub_j))
            if len(proposed_edges) >= bridge_candidates:
                break
        if len(proposed_edges) >= bridge_candidates:
            break

    return proposed_edges

# ===============================
# PIPELINE COMPLETO
# ===============================
def full_pipeline(adj, n_nodes, n_episodes=5000, max_steps=300,
                  embed_steps=50, elite_pct=0.0, quiet=False):
    """Observar + Hook + Aplicar pontes. Retorna (pontes, adj)."""
    visit_freq, covisit, traj = observe(adj, n_nodes, n_episodes, max_steps)
    bridges = nietzschedb_hook(adj, n_nodes, visit_freq, covisit,
                               embed_steps=embed_steps, elite_pct=elite_pct,
                               quiet=quiet)
    for u, v in bridges:
        add_edge(adj, u, v)
    return bridges

# ================================================================
#  FASE 1 — CICATRIZACAO (Ablacao Ontologica)
# ================================================================
def measure_bridge_impact(adj, n_nodes, bridges, samples=300, max_visited=0):
    """Remove cada ponte temporariamente, mede degradacao do caminho."""
    base_path = average_path_length(adj, n_nodes, samples, max_visited)
    impacts = []
    for u, v in bridges:
        remove_edge(adj, u, v)
        wounded_path = average_path_length(adj, n_nodes, samples, max_visited)
        add_edge(adj, u, v)
        delta = wounded_path - base_path
        impacts.append(((u, v), delta))
    impacts.sort(key=lambda x: x[1], reverse=True)
    return impacts, base_path

def fase1_cicatrizacao():
    print("=" * 70)
    print("  FASE 1 — CICATRIZACAO (Ablacao Ontologica)")
    print("  Deletar a ponte mais critica. O sistema regenera?")
    print("=" * 70)
    print()

    n_nodes = 1000
    n_communities = 5
    community_size = n_nodes // n_communities

    # Construir grafo fresco e rodar pipeline
    adj = make_graph(n_nodes, n_communities)
    baseline_path = average_path_length(adj, n_nodes)
    print(f"  Baseline (sem pontes): caminho medio = {baseline_path:.2f}")

    bridges = full_pipeline(adj, n_nodes)
    post_path = average_path_length(adj, n_nodes)
    print(f"  Pos-pontes: caminho medio = {post_path:.2f}")
    print(f"  Pontes originais: {bridges}")
    print(f"  Pares de comunidades:")
    original_pairs = set()
    for u, v in bridges:
        cu = community_of(u, community_size)
        cv = community_of(v, community_size)
        pair = (min(cu, cv), max(cu, cv))
        original_pairs.add(pair)
        print(f"    ({u},{v}) -> comm ({cu},{cv})")
    print()

    if not bridges:
        print("  MORTE: Nenhuma ponte para ablacionar.")
        return {"veredito": "MORTE", "motivo": "nenhuma ponte original"}

    # Medir impacto de cada ponte
    impacts, path_with_all = measure_bridge_impact(adj, n_nodes, bridges)
    print("  Impacto de cada ponte:")
    for (u, v), delta in impacts:
        print(f"    ({u},{v}): delta caminho = +{delta:.2f}")

    # Amputar a mais critica
    victim = impacts[0][0]
    print(f"\n  AMPUTACAO: removendo ponte mais critica ({victim[0]},{victim[1]})")
    remove_edge(adj, victim[0], victim[1])

    wounded_path = average_path_length(adj, n_nodes)
    print(f"  Caminho pos-amputacao: {wounded_path:.2f} (baseline era {baseline_path:.2f})")

    # Remover TODAS as pontes restantes para forcar re-descoberta total
    remaining_bridges = [b for b in bridges if b != victim]
    for u, v in remaining_bridges:
        remove_edge(adj, u, v)
    print(f"  Removidas tambem as {len(remaining_bridges)} pontes restantes para re-descoberta total")

    naked_path = average_path_length(adj, n_nodes)
    print(f"  Caminho sem nenhuma ponte: {naked_path:.2f}")

    # Re-observar e re-sintetizar do zero
    print("\n  Re-observando grafo nu...")
    regen_bridges = full_pipeline(adj, n_nodes)
    regen_path = average_path_length(adj, n_nodes)
    print(f"  Pontes regeneradas: {regen_bridges}")
    print(f"  Caminho pos-regeneracao: {regen_path:.2f}")

    # Verificar: as pontes regeneradas conectam o mesmo par de comunidades?
    regen_pairs = set()
    for u, v in regen_bridges:
        cu = community_of(u, community_size)
        cv = community_of(v, community_size)
        pair = (min(cu, cv), max(cu, cv))
        regen_pairs.add(pair)
        print(f"    ({u},{v}) -> comm ({cu},{cv})")

    # Avaliar
    victim_cu = community_of(victim[0], community_size)
    victim_cv = community_of(victim[1], community_size)
    victim_pair = (min(victim_cu, victim_cv), max(victim_cu, victim_cv))

    pair_recovered = victim_pair in regen_pairs
    path_recovered = regen_path < wounded_path

    print(f"\n  Par amputado: comm ({victim_pair[0]},{victim_pair[1]})")
    print(f"  Par recuperado nas regeneradas: {'SIM' if pair_recovered else 'NAO'}")
    print(f"  Caminho melhorou: {'SIM' if path_recovered else 'NAO'} ({wounded_path:.2f} -> {regen_path:.2f})")

    if pair_recovered and path_recovered:
        veredito = "VIDA"
        print(f"\n  >>> FASE 1: VIDA — Cicatriz isomorfica. Sistema regenerou. <<<")
    elif path_recovered:
        veredito = "VIDA_PARCIAL"
        print(f"\n  >>> FASE 1: VIDA PARCIAL — Caminho recuperou mas par diferente. <<<")
    else:
        veredito = "MORTE"
        print(f"\n  >>> FASE 1: MORTE — Ponte nao regenerou. <<<")

    print()
    return {
        "veredito": veredito,
        "ponte_amputada": victim,
        "par_amputado": victim_pair,
        "pontes_regeneradas": regen_bridges,
        "pares_regenerados": regen_pairs,
        "par_recuperado": pair_recovered,
        "caminho_baseline": baseline_path,
        "caminho_pos_pontes": post_path,
        "caminho_amputado": wounded_path,
        "caminho_regenerado": regen_path,
    }

# ================================================================
#  FASE 2A — PERMUTACAO (Rotacao do Mundo)
# ================================================================
def permute_graph(adj, n_nodes, seed=777):
    """Fisher-Yates shuffle de todos os IDs, reconstroi adjacencia."""
    rng = Rng(seed)
    ids = list(range(n_nodes))
    for i in range(n_nodes - 1, 0, -1):
        j = rng.next_int(i + 1)
        ids[i], ids[j] = ids[j], ids[i]

    fwd_map = {old: new for old, new in enumerate(ids)}
    inv_map = {new: old for old, new in enumerate(ids)}

    new_adj = {i: set() for i in range(n_nodes)}
    for u in adj:
        for v in adj[u]:
            new_adj[fwd_map[u]].add(fwd_map[v])

    return new_adj, fwd_map, inv_map

def fase2a_permutacao():
    print("=" * 70)
    print("  FASE 2A — PERMUTACAO (Rotacao do Mundo)")
    print("  Embaralhar todos os IDs. Topologia e invariante?")
    print("=" * 70)
    print()

    n_nodes = 1000
    n_communities = 5
    community_size = n_nodes // n_communities

    # Pipeline no grafo original
    adj_orig = make_graph(n_nodes, n_communities)
    bridges_orig = full_pipeline(adj_orig, n_nodes)
    print(f"  Pontes originais: {bridges_orig}")

    original_pairs = set()
    for u, v in bridges_orig:
        cu = community_of(u, community_size)
        cv = community_of(v, community_size)
        original_pairs.add((min(cu, cv), max(cu, cv)))
    print(f"  Pares de comunidades originais: {sorted(original_pairs)}")
    print()

    # Permutar grafo
    adj_fresh = make_graph(n_nodes, n_communities)
    adj_perm, fwd_map, inv_map = permute_graph(adj_fresh, n_nodes, seed=777)
    print(f"  Grafo permutado (Fisher-Yates com seed=777)")

    # Pipeline no grafo permutado
    bridges_perm = full_pipeline(adj_perm, n_nodes)
    print(f"  Pontes no grafo permutado: {bridges_perm}")

    # Mapear de volta via inverse map
    perm_pairs = set()
    for u, v in bridges_perm:
        orig_u = inv_map[u]
        orig_v = inv_map[v]
        cu = community_of(orig_u, community_size)
        cv = community_of(orig_v, community_size)
        perm_pairs.add((min(cu, cv), max(cu, cv)))
    print(f"  Pares de comunidades (mapeados de volta): {sorted(perm_pairs)}")

    # Comparar
    match = original_pairs == perm_pairs
    overlap = original_pairs & perm_pairs
    print(f"\n  Pares originais: {sorted(original_pairs)}")
    print(f"  Pares permutados: {sorted(perm_pairs)}")
    print(f"  Intersecao: {sorted(overlap)}")
    print(f"  Match exato: {'SIM' if match else 'NAO'}")

    if len(original_pairs) == 0 and len(perm_pairs) == 0:
        veredito = "MORTE"
        print(f"\n  >>> FASE 2A: MORTE — Nenhuma ponte em nenhum grafo. <<<")
    elif match:
        veredito = "VIDA"
        print(f"\n  >>> FASE 2A: VIDA — Isomorfismo topologico perfeito. <<<")
    elif overlap:
        # Metrica: quantos dos pares originais foram reencontrados?
        recall = len(overlap) / max(len(original_pairs), 1)
        print(f"  Recall (originais reencontrados): {recall:.0%}")
        if recall >= 1.0:
            veredito = "VIDA"
            print(f"\n  >>> FASE 2A: VIDA — Todos os pares originais reencontrados (+ {len(perm_pairs) - len(overlap)} extras). <<<")
        elif recall >= 0.5:
            veredito = "VIDA_PARCIAL"
            print(f"\n  >>> FASE 2A: VIDA PARCIAL — {recall:.0%} dos pares originais reencontrados. <<<")
        else:
            veredito = "MORTE"
            print(f"\n  >>> FASE 2A: MORTE — Recall insuficiente ({recall:.0%}). <<<")
    else:
        veredito = "MORTE"
        print(f"\n  >>> FASE 2A: MORTE — Zero sobreposicao. <<<")

    print()
    return {
        "veredito": veredito,
        "pontes_originais": bridges_orig,
        "pontes_permutadas": bridges_perm,
        "pares_originais": sorted(original_pairs),
        "pares_permutados": sorted(perm_pairs),
        "intersecao": sorted(overlap),
    }

# ================================================================
#  FASE 2B — VENENO (Mundo Adversarial)
# ================================================================
def inject_poison_edges(adj, n_nodes, community_size, n_poison=50, seed=666):
    """Injeta arestas aleatorias entre comunidades diferentes."""
    rng = Rng(seed)
    poison = []
    attempts = 0
    while len(poison) < n_poison and attempts < n_poison * 20:
        u = rng.next_int(n_nodes)
        v = rng.next_int(n_nodes)
        attempts += 1
        if u == v:
            continue
        if community_of(u, community_size) == community_of(v, community_size):
            continue
        if v in adj[u]:
            continue
        add_edge(adj, u, v)
        poison.append((u, v))
    return poison

def compute_edge_vitality(adj, n_nodes, n_walks=2000, walk_len=100, seed=4444):
    """Random walks para medir trafego por aresta. Retorna {(u,v): freq normalizada}."""
    rng = Rng(seed)
    edge_traffic = defaultdict(int)

    for _ in range(n_walks):
        pos = rng.next_int(n_nodes)
        for _ in range(walk_len):
            neighbors = list(adj[pos])
            if not neighbors:
                break
            nxt = neighbors[rng.next_int(len(neighbors))]
            edge_key = (min(pos, nxt), max(pos, nxt))
            edge_traffic[edge_key] += 1
            pos = nxt

    if not edge_traffic:
        return {}

    max_traffic = max(edge_traffic.values())
    return {e: c / max_traffic for e, c in edge_traffic.items()}

def prune_dead_edges(adj, vitality, threshold):
    """Remove arestas com vitalidade abaixo do threshold."""
    pruned = []
    for (u, v), vit in vitality.items():
        if vit < threshold:
            if v in adj.get(u, set()):
                remove_edge(adj, u, v)
                pruned.append((u, v))
    return pruned

def shannon_entropy_degree(adj, n_nodes):
    """Entropia de Shannon da distribuicao de grau."""
    degrees = [len(adj[i]) for i in range(n_nodes)]
    total = sum(degrees)
    if total == 0:
        return 0.0
    freq = defaultdict(int)
    for d in degrees:
        freq[d] += 1
    entropy = 0.0
    for count in freq.values():
        p = count / n_nodes
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def compute_edge_impact(adj, n_nodes, edges, samples=150, max_visited=0):
    """Mede impacto estrutural de cada aresta: delta caminho ao remover."""
    base_path = average_path_length(adj, n_nodes, samples, max_visited)
    impacts = {}
    for u, v in edges:
        e = (min(u, v), max(u, v))
        if v not in adj.get(u, set()):
            impacts[e] = 0.0
            continue
        remove_edge(adj, u, v)
        wounded = average_path_length(adj, n_nodes, samples, max_visited)
        add_edge(adj, u, v)
        impacts[e] = wounded - base_path
    return impacts, base_path

def fase2b_veneno():
    print("=" * 70)
    print("  FASE 2B — VENENO (Mundo Adversarial)")
    print("  Injetar 50 arestas falsas. Sistema imune funciona?")
    print("=" * 70)
    print()

    n_nodes = 1000
    n_communities = 5
    community_size = n_nodes // n_communities

    # Construir e sintetizar pontes boas
    adj = make_graph(n_nodes, n_communities)
    good_bridges = full_pipeline(adj, n_nodes)
    print(f"  Pontes boas: {good_bridges}")

    good_set = set()
    for u, v in good_bridges:
        good_set.add((min(u, v), max(u, v)))

    entropy_pre = shannon_entropy_degree(adj, n_nodes)
    print(f"  Entropia grau pre-veneno: {entropy_pre:.4f}")
    print()

    # Injetar veneno
    poison = inject_poison_edges(adj, n_nodes, community_size, 50)
    print(f"  Veneno injetado: {len(poison)} arestas inter-comunidade")

    poison_set = set()
    for u, v in poison:
        poison_set.add((min(u, v), max(u, v)))

    entropy_envenenado = shannon_entropy_degree(adj, n_nodes)
    print(f"  Entropia grau pos-veneno: {entropy_envenenado:.4f}")

    # Medir impacto estrutural: pontes boas vs veneno
    all_test_edges = list(good_set | poison_set)
    print(f"  Calculando impacto estrutural de {len(all_test_edges)} arestas...")
    impacts, base_path = compute_edge_impact(adj, n_nodes, all_test_edges, samples=150)

    good_impacts = [impacts.get(e, 0) for e in good_set]
    poison_impacts = [impacts.get(e, 0) for e in poison_set]

    avg_good_impact = sum(good_impacts) / len(good_impacts) if good_impacts else 0
    avg_poison_impact = sum(poison_impacts) / len(poison_impacts) if poison_impacts else 0
    print(f"  Impacto medio pontes boas: +{avg_good_impact:.4f}")
    print(f"  Impacto medio veneno: +{avg_poison_impact:.4f}")

    # Threshold: mediana do impacto de todas as arestas testadas
    all_impacts = sorted(impacts.values())
    threshold = all_impacts[len(all_impacts) // 2] if all_impacts else 0
    print(f"  Threshold de poda (mediana impacto): {threshold:.4f}")

    # Podar arestas com baixo impacto estrutural
    pruned = []
    for e, impact in impacts.items():
        if impact <= threshold:
            u, v = e
            if v in adj.get(u, set()):
                remove_edge(adj, u, v)
                pruned.append(e)

    print(f"  Arestas podadas: {len(pruned)}")

    # Contar resultados
    veneno_eliminado = sum(1 for e in pruned if e in poison_set)
    pontes_mortas = sum(1 for e in pruned if e in good_set)

    veneno_vivo = len(poison) - veneno_eliminado
    pontes_vivas = len(good_bridges) - pontes_mortas

    pct_veneno_eliminado = veneno_eliminado / max(len(poison), 1) * 100
    pct_pontes_vivas = pontes_vivas / max(len(good_bridges), 1) * 100

    entropy_post = shannon_entropy_degree(adj, n_nodes)

    print(f"\n  Veneno eliminado: {veneno_eliminado}/{len(poison)} ({pct_veneno_eliminado:.0f}%)")
    print(f"  Veneno ainda vivo: {veneno_vivo}/{len(poison)}")
    print(f"  Pontes boas sobreviventes: {pontes_vivas}/{len(good_bridges)} ({pct_pontes_vivas:.0f}%)")
    print(f"  Entropia grau pos-poda: {entropy_post:.4f}")

    # Veredito
    if pct_veneno_eliminado >= 80 and pontes_mortas == 0:
        veredito = "VIDA"
        print(f"\n  >>> FASE 2B: VIDA — {pct_veneno_eliminado:.0f}% veneno eliminado, 100% pontes intactas. <<<")
    elif pct_veneno_eliminado >= 60 and pontes_mortas == 0:
        veredito = "VIDA_PARCIAL"
        print(f"\n  >>> FASE 2B: VIDA PARCIAL — {pct_veneno_eliminado:.0f}% veneno eliminado. <<<")
    elif pontes_mortas > 0 and pct_veneno_eliminado >= 80:
        veredito = "VIDA_PARCIAL"
        print(f"\n  >>> FASE 2B: VIDA PARCIAL — {pct_veneno_eliminado:.0f}% veneno eliminado mas {pontes_mortas} ponte(s) boa(s) perdida(s). <<<")
    elif pontes_mortas > 0:
        veredito = "MORTE"
        print(f"\n  >>> FASE 2B: MORTE — {pontes_mortas} pontes boas destruidas com apenas {pct_veneno_eliminado:.0f}% veneno eliminado! <<<")
    else:
        veredito = "MORTE"
        print(f"\n  >>> FASE 2B: MORTE — Apenas {pct_veneno_eliminado:.0f}% veneno eliminado. <<<")

    print()
    return {
        "veredito": veredito,
        "pontes_boas": good_bridges,
        "veneno_injetado": len(poison),
        "veneno_eliminado": veneno_eliminado,
        "veneno_vivo": veneno_vivo,
        "pontes_mortas": pontes_mortas,
        "pontes_vivas": pontes_vivas,
        "entropia_pre": entropy_pre,
        "entropia_envenenado": entropy_envenenado,
        "entropia_post": entropy_post,
        "impacto_medio_boas": avg_good_impact,
        "impacto_medio_veneno": avg_poison_impact,
    }

# ================================================================
#  FASE 3 — ABISMO (Escala Cruel)
# ================================================================
def run_at_scale(n_nodes, seed=42):
    """Pipeline completo numa escala arbitraria, com timing e memoria."""
    n_communities = 5
    community_size = n_nodes // n_communities
    max_visited = 5000 if n_nodes >= 10000 else 0
    n_episodes = min(8000, max(500, n_nodes * 3))
    max_steps = 300
    embed_steps = 30 if n_nodes >= 10000 else 50
    elite_pct = 0.15 if n_nodes >= 10000 else 0.0

    tracemalloc.start()
    t0 = time.perf_counter()

    # Construir
    t_graph = time.perf_counter()
    adj = make_graph(n_nodes, n_communities, seed=seed)
    t_graph = time.perf_counter() - t_graph

    # Baseline
    t_base = time.perf_counter()
    baseline_path = average_path_length(adj, n_nodes, samples=200, max_visited=max_visited)
    t_base = time.perf_counter() - t_base

    # Observar
    t_obs = time.perf_counter()
    visit_freq, covisit, traj = observe(adj, n_nodes, n_episodes, max_steps)
    t_obs = time.perf_counter() - t_obs

    # Hook
    t_hook = time.perf_counter()
    bridges = nietzschedb_hook(adj, n_nodes, visit_freq, covisit,
                                embed_steps=embed_steps, elite_pct=elite_pct,
                                quiet=True)
    for u, v in bridges:
        add_edge(adj, u, v)
    t_hook = time.perf_counter() - t_hook

    # Pos
    t_post = time.perf_counter()
    post_path = average_path_length(adj, n_nodes, samples=200, max_visited=max_visited)
    t_post = time.perf_counter() - t_post

    t_total = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    reduction = (baseline_path - post_path) / max(baseline_path, 0.01) * 100

    return {
        "n_nodes": n_nodes,
        "n_edges": sum(len(adj[i]) for i in range(n_nodes)) // 2,
        "bridges": len(bridges),
        "baseline_path": baseline_path,
        "post_path": post_path,
        "reduction_pct": reduction,
        "t_graph": t_graph,
        "t_baseline": t_base,
        "t_observe": t_obs,
        "t_hook": t_hook,
        "t_post": t_post,
        "t_total": t_total,
        "mem_peak_mb": peak / (1024 * 1024),
    }

def fase3_abismo(scales=None):
    if scales is None:
        scales = [1000, 10000, 100000]

    MAX_TIME = 120  # segundos, limite para Python puro

    print("=" * 70)
    print("  FASE 3 — ABISMO (Escala Cruel)")
    print(f"  Escalas: {scales}")
    print("=" * 70)
    print()

    results = {}
    for n in scales:
        print(f"  --- N = {n:,} ---")
        try:
            r = run_at_scale(n)
            results[n] = r
            print(f"    Arestas: {r['n_edges']:,}")
            print(f"    Pontes: {r['bridges']}")
            print(f"    Caminho: {r['baseline_path']:.2f} -> {r['post_path']:.2f} ({r['reduction_pct']:.1f}%)")
            print(f"    Tempo: grafo={r['t_graph']:.2f}s obs={r['t_observe']:.2f}s hook={r['t_hook']:.2f}s TOTAL={r['t_total']:.2f}s")
            print(f"    Memoria pico: {r['mem_peak_mb']:.1f} MB")

            if r['t_total'] > MAX_TIME:
                print(f"    AVISO: >{MAX_TIME}s — nao pode escalar mais")
            print()
        except Exception as e:
            print(f"    FALHA: {e}")
            results[n] = {"veredito": "MORTE", "erro": str(e)}
            print()
            continue

    # Veredito geral
    passed = 0
    total = len(scales)
    for n in scales:
        r = results.get(n, {})
        if isinstance(r, dict) and "bridges" in r and r["bridges"] > 0 and r.get("t_total", 999) <= MAX_TIME:
            passed += 1

    if passed == total:
        veredito = "VIDA"
        print(f"  >>> FASE 3: VIDA — {passed}/{total} escalas passaram. <<<")
    elif passed >= total - 1:
        veredito = "VIDA_PARCIAL"
        print(f"  >>> FASE 3: VIDA PARCIAL — {passed}/{total} escalas passaram. <<<")
    else:
        veredito = "MORTE"
        print(f"  >>> FASE 3: MORTE — Apenas {passed}/{total} escalas passaram. <<<")

    print()
    results["veredito"] = veredito
    return results

# ================================================================
#  MAIN
# ================================================================
def main():
    print()
    print("#" * 70)
    print("#")
    print("#  PROTOCOLO DE FALSIFICABILIDADE")
    print("#  3 Laminas contra o NietzscheDB")
    print("#")
    print("#  Objetivo: DESTRUIR o resultado do Lion Environment.")
    print("#  Se sobreviver, nao ha mais o que provar.")
    print("#")
    print("#" * 70)
    print()

    resultados = {}

    # Fase 1
    r1 = fase1_cicatrizacao()
    resultados["fase1"] = r1

    # Fase 2A
    r2a = fase2a_permutacao()
    resultados["fase2a"] = r2a

    # Fase 2B
    r2b = fase2b_veneno()
    resultados["fase2b"] = r2b

    # Fase 3 — sem 100K por defeito (Python puro e lento)
    r3 = fase3_abismo([1000, 10000])
    resultados["fase3"] = r3

    # ── VEREDICTO FINAL ──
    print("#" * 70)
    print("#  VEREDICTO FINAL")
    print("#" * 70)
    print()

    testes = [
        ("Fase 1: Cicatrizacao", r1["veredito"]),
        ("Fase 2A: Permutacao", r2a["veredito"]),
        ("Fase 2B: Veneno", r2b["veredito"]),
        ("Fase 3: Abismo", r3["veredito"]),
    ]

    vivos = 0
    mortos = 0
    parciais = 0

    for nome, v in testes:
        simbolo = "VIDA" if v == "VIDA" else ("~" if "PARCIAL" in v else "MORTE")
        if v == "VIDA":
            vivos += 1
        elif "PARCIAL" in v:
            parciais += 1
        else:
            mortos += 1
        print(f"  {nome:30} [{simbolo}]")

    print()
    total = len(testes)
    print(f"  VIDA: {vivos}/{total}  |  PARCIAL: {parciais}/{total}  |  MORTE: {mortos}/{total}")
    print()

    if mortos == 0 and vivos >= 3:
        print("  ================================================================")
        print("  O NietzscheDB SOBREVIVEU ao Protocolo de Falsificabilidade.")
        print("  A topologia e genuina. Nao ha mais o que provar.")
        print("  ================================================================")
    elif mortos == 0:
        print("  ================================================================")
        print("  O NietzscheDB sobreviveu com ressalvas.")
        print("  Nenhuma morte, mas vida parcial nao e certeza.")
        print("  ================================================================")
    elif mortos <= 1:
        print("  ================================================================")
        print("  O NietzscheDB quase sobreviveu.")
        print(f"  {mortos} morte(s) — precisa investigacao.")
        print("  ================================================================")
    else:
        print("  ================================================================")
        print("  O NietzscheDB FALHOU no Protocolo de Falsificabilidade.")
        print(f"  {mortos} mortes — resultado do Lion e fragil.")
        print("  ================================================================")

    print()

if __name__ == "__main__":
    main()
