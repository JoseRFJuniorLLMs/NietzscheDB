//! Experiment: Controlled Amnesia — Topology is the True Memory.
//!
//! ## Protocol
//!
//! 1. SCULPT (80 cycles): build topology under Normal/Off/Inverted TGC.
//! 2. SNAPSHOT: record pre-amnesia NCA, AUC, embedding state.
//! 3. AMNESIA: zero ALL non-seed embeddings. Topology untouched.
//!    Seeds = top-10% highest-degree original nodes ("sensory cortex").
//! 4. RECOVERY (50 cycles): propagation-only. Seeds broadcast, topology recovers.
//! 5. VERDICT: Normal recovery >> Off ≈ Inverted proves topology = true memory.
//!
//! ## Usage
//! ```sh
//! python3 experiments/dataset_prepare.py
//! cargo run --release --bin experiment_amnesia
//! python3 experiments/analysis_amnesia.py
//! ```

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;

// ══════════════════════════════════════════════════════════════════
//  Constants
// ══════════════════════════════════════════════════════════════════

const EMBED_DIM: usize = 16;
const SCULPT_CYCLES: usize = 80;
const RECOVERY_CYCLES: usize = 50;
const EMBED_LR: f32 = 0.08;
const PROPAG_STEPS: usize = 3;
const WARMUP_STEPS: usize = 10;
const NOISE_INJECT_PER_CYCLE: usize = 15;
const MAX_DELETION_RATE: f32 = 0.05;
const VIT_THRESHOLD: f32 = 0.30;
const ENG_THRESHOLD: f32 = 0.10;
const SEED_FRACTION: f64 = 0.10;

// ══════════════════════════════════════════════════════════════════
//  Poincaré Math
// ══════════════════════════════════════════════════════════════════

fn poincare_distance(u: &[f32], v: &[f32]) -> f64 {
    let mut diff_sq = 0.0f64;
    let mut norm_u_sq = 0.0f64;
    let mut norm_v_sq = 0.0f64;
    let n = u.len().min(v.len());
    for i in 0..n {
        let a = u[i] as f64;
        let b = v[i] as f64;
        let d = a - b;
        diff_sq += d * d;
        norm_u_sq += a * a;
        norm_v_sq += b * b;
    }
    let denom = (1.0 - norm_u_sq) * (1.0 - norm_v_sq);
    if denom <= 0.0 {
        return 20.0;
    }
    let arg = (1.0 + 2.0 * diff_sq / denom).max(1.0);
    arg.acosh()
}

fn project_into_ball(coords: &mut [f32]) {
    let norm_sq: f64 = coords.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let norm = norm_sq.sqrt();
    if norm > 0.95 {
        let scale = (0.95 / (norm + 1e-10)) as f32;
        for c in coords.iter_mut() {
            *c *= scale;
        }
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

fn embedding_norm(emb: &[f32]) -> f64 {
    emb.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

// ══════════════════════════════════════════════════════════════════
//  Deterministic PRNG
// ══════════════════════════════════════════════════════════════════

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let t = ((self.next_u64() >> 33) as f32) / (u32::MAX as f32);
        lo + t * (hi - lo)
    }
    fn next_usize(&mut self, hi: usize) -> usize {
        ((self.next_u64() >> 33) as usize) % hi.max(1)
    }
    fn chance(&mut self, p: f32) -> bool { self.next_f32(0.0, 1.0) < p }
}

// ══════════════════════════════════════════════════════════════════
//  TGC Mode
// ══════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq)]
enum TgcMode { Normal, Off, Inverted }
impl TgcMode {
    fn label(&self) -> &'static str {
        match self { TgcMode::Normal => "Normal", TgcMode::Off => "Off", TgcMode::Inverted => "Inverted" }
    }
}

// ══════════════════════════════════════════════════════════════════
//  CSV Loading
// ══════════════════════════════════════════════════════════════════

fn load_labels(path: &str) -> HashMap<usize, usize> {
    let mut labels = HashMap::new();
    if let Ok(contents) = std::fs::read_to_string(path) {
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("node_id") || line.starts_with('#') { continue; }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                if let (Ok(id), Ok(label)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    labels.insert(id, label);
                }
            }
        }
    }
    labels
}

fn load_edges(path: &str) -> Vec<(usize, usize)> {
    let contents = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("Cannot read {}", path));
    let mut edges = Vec::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("source") || line.starts_with('#') { continue; }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let (Ok(u), Ok(v)) = (parts[0].trim().parse::<usize>(), parts[1].trim().parse::<usize>()) {
                if u != v { edges.push((u, v)); }
            }
        }
    }
    edges
}

fn generate_synthetic(r: &mut Rng) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let communities = 7usize;
    let nodes_per_comm = 386;
    let total = communities * nodes_per_comm;
    let p_intra = 0.008_f32;
    let p_inter = 0.0004_f32;
    println!("    [SYNTHETIC] Generating SBM: {} nodes, {} communities", total, communities);
    let mut all_edges: Vec<(usize, usize)> = Vec::new();
    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
    for i in 0..total {
        let ci = i / nodes_per_comm;
        for j in (i + 1)..total {
            let cj = j / nodes_per_comm;
            let p = if ci == cj { p_intra } else { p_inter };
            if r.chance(p) {
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                if edge_set.insert((a, b)) { all_edges.push((a, b)); }
            }
        }
    }
    let n_test = (all_edges.len() as f32 * 0.1).round() as usize;
    for i in (1..all_edges.len()).rev() {
        let j = r.next_usize(i + 1);
        all_edges.swap(i, j);
    }
    let test_edges = all_edges.split_off(all_edges.len() - n_test);
    (all_edges, test_edges)
}

// ══════════════════════════════════════════════════════════════════
//  Data Structures
// ══════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct SimNode {
    energy: f32,
    hausdorff: f32,
    entropy_delta: f32,
    elite_proximity: f32,
    toxicity: f32,
    is_original: bool,
    embedding: Vec<f32>,
}

struct VoidSeed { embedding: Vec<f32> }

struct PreAmnesiaSnapshot {
    nca: f64,
    auc: f64,
    mean_norm: f64,
    embeddings: HashMap<usize, Vec<f32>>,
}

#[derive(Clone)]
struct RecoveryRow {
    mode: String,
    phase: String,
    cycle: usize,
    nca: f64,
    mean_norm: f64,
    cosine_recovery: f64,
    auc: f64,
    total_nodes: usize,
    total_edges: usize,
}

// ══════════════════════════════════════════════════════════════════
//  The Amnesia Simulator
// ══════════════════════════════════════════════════════════════════

struct AmnesiaSim {
    nodes: HashMap<usize, SimNode>,
    adj: HashMap<usize, HashSet<usize>>,
    next_id: usize,
    voids: Vec<VoidSeed>,
    mode: TgcMode,
    labels: HashMap<usize, usize>,
    node_ages: HashMap<usize, usize>,
    test_edges: Vec<(usize, usize)>,
    negative_edges: Vec<(usize, usize)>,
    n_original: usize,
    seed_ids: HashSet<usize>,
    pre_amnesia: Option<PreAmnesiaSnapshot>,
    new_edges_this_cycle: Vec<(usize, usize)>,
}

impl AmnesiaSim {
    fn new(
        mode: TgcMode,
        train_edges: &[(usize, usize)],
        test_edges: Vec<(usize, usize)>,
        labels: HashMap<usize, usize>,
    ) -> Self {
        let mut node_ids: HashSet<usize> = HashSet::new();
        for &(u, v) in train_edges { node_ids.insert(u); node_ids.insert(v); }
        for &(u, v) in &test_edges { node_ids.insert(u); node_ids.insert(v); }

        let max_id = node_ids.iter().cloned().max().unwrap_or(0);
        let mut r = Rng::new(12345);
        let mut nodes = HashMap::new();
        let mut adj: HashMap<usize, HashSet<usize>> = HashMap::new();

        for &id in &node_ids {
            let mut emb = Vec::with_capacity(EMBED_DIM);
            for _ in 0..EMBED_DIM { emb.push(r.next_f32(-0.1, 0.1)); }
            project_into_ball(&mut emb);
            nodes.insert(id, SimNode {
                energy: r.next_f32(0.7, 1.0), hausdorff: r.next_f32(0.6, 1.2),
                entropy_delta: r.next_f32(0.0, 0.2), elite_proximity: r.next_f32(0.0, 0.3),
                toxicity: r.next_f32(0.0, 0.05), is_original: true,
                embedding: emb,            });
            adj.entry(id).or_default();
        }
        for &(u, v) in train_edges {
            adj.entry(u).or_default().insert(v);
            adj.entry(v).or_default().insert(u);
        }

        let n_original = nodes.len();
        let mut node_ages: HashMap<usize, usize> = HashMap::new();
        for &id in &node_ids { node_ages.insert(id, WARMUP_STEPS); }

        let mut sim = AmnesiaSim {
            nodes, adj, next_id: max_id + 1, voids: Vec::new(),
            mode, labels, node_ages,
            test_edges, negative_edges: Vec::new(), n_original,
            seed_ids: HashSet::new(), pre_amnesia: None,
            new_edges_this_cycle: Vec::new(),
        };
        sim.generate_negatives(&mut Rng::new(777));

        // Warmup propagation
        for _ in 0..WARMUP_STEPS { sim.propagate_embeddings(false); }
        sim
    }

    fn alloc_id(&mut self) -> usize { let id = self.next_id; self.next_id += 1; id }

    fn add_edge(&mut self, a: usize, b: usize) {
        self.adj.entry(a).or_default().insert(b);
        self.adj.entry(b).or_default().insert(a);
    }

    fn remove_node(&mut self, id: usize) {
        if let Some(neighbors) = self.adj.remove(&id) {
            for nb in neighbors { if let Some(nbs) = self.adj.get_mut(&nb) { nbs.remove(&id); } }
        }
        self.nodes.remove(&id);
        self.node_ages.remove(&id);
        self.labels.remove(&id);
    }

    fn degree(&self, id: usize) -> usize { self.adj.get(&id).map_or(0, |s| s.len()) }

    fn generate_negatives(&mut self, r: &mut Rng) {
        let original_ids: Vec<usize> = self.nodes.iter().filter(|(_, n)| n.is_original).map(|(&id, _)| id).collect();
        let n = original_ids.len();
        if n == 0 { return; }
        self.negative_edges.clear();
        for &(u, _) in &self.test_edges {
            let neighbors = self.adj.get(&u).cloned().unwrap_or_default();
            let mut attempts = 0;
            loop {
                let w = original_ids[r.next_usize(n)];
                if w != u && !neighbors.contains(&w) { self.negative_edges.push((u, w)); break; }
                attempts += 1;
                if attempts > 100 {
                    let w = original_ids[r.next_usize(n)];
                    if w != u { self.negative_edges.push((u, w)); }
                    break;
                }
            }
        }
    }

    fn link_score(&self, u: usize, v: usize) -> f64 {
        match (self.nodes.get(&u), self.nodes.get(&v)) {
            (Some(nu), Some(nv)) => -poincare_distance(&nu.embedding, &nv.embedding),
            _ => -100.0,
        }
    }

    fn compute_auc(&self) -> f64 {
        let n = self.test_edges.len().min(self.negative_edges.len());
        if n == 0 { return 0.5; }
        let mut concordant = 0u64;
        let mut tied = 0u64;
        for i in 0..n {
            let (pu, pv) = self.test_edges[i];
            let (nu, nv) = self.negative_edges[i];
            let pos = self.link_score(pu, pv);
            let neg = self.link_score(nu, nv);
            if pos > neg { concordant += 1; } else if (pos - neg).abs() < 1e-12 { tied += 1; }
        }
        (concordant as f64 + 0.5 * tied as f64) / n as f64
    }

    fn compute_nca(&self) -> f64 {
        if self.labels.is_empty() { return f64::NAN; }
        let mut labeled_data: Vec<(usize, usize, &[f32])> = Vec::new();
        for (&id, &label) in &self.labels {
            if let Some(node) = self.nodes.get(&id) {
                labeled_data.push((id, label, &node.embedding));
            }
        }
        if labeled_data.len() < 20 { return f64::NAN; }

        let n_classes = *self.labels.values().max().unwrap_or(&0) + 1;
        let mut centroids: Vec<Vec<f64>> = vec![vec![0.0; EMBED_DIM]; n_classes];
        let mut counts: Vec<usize> = vec![0; n_classes];
        for &(_, label, emb) in &labeled_data {
            if label < n_classes {
                for d in 0..EMBED_DIM { centroids[label][d] += emb[d] as f64; }
                counts[label] += 1;
            }
        }
        for c in 0..n_classes {
            if counts[c] > 0 { for d in 0..EMBED_DIM { centroids[c][d] /= counts[c] as f64; } }
        }

        let mut correct = 0usize;
        let mut total = 0usize;
        for &(_, true_label, emb) in &labeled_data {
            if true_label >= n_classes || counts[true_label] == 0 { continue; }
            let mut best_class = 0;
            let mut best_dist = f64::MAX;
            for c in 0..n_classes {
                if counts[c] == 0 { continue; }
                let mut dist = 0.0f64;
                if c == true_label && counts[c] > 1 {
                    let adj_count = (counts[c] - 1) as f64;
                    for d in 0..EMBED_DIM {
                        let adj_c = (centroids[c][d] * counts[c] as f64 - emb[d] as f64) / adj_count;
                        let diff = emb[d] as f64 - adj_c;
                        dist += diff * diff;
                    }
                } else {
                    for d in 0..EMBED_DIM {
                        let diff = emb[d] as f64 - centroids[c][d];
                        dist += diff * diff;
                    }
                }
                if dist < best_dist { best_dist = dist; best_class = c; }
            }
            if best_class == true_label { correct += 1; }
            total += 1;
        }
        if total == 0 { f64::NAN } else { correct as f64 / total as f64 }
    }

    fn mean_embedding_norm(&self) -> f64 {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for n in self.nodes.values() {
            sum += embedding_norm(&n.embedding);
            count += 1;
        }
        if count == 0 { 0.0 } else { sum / count as f64 }
    }

    fn cosine_recovery(&self) -> f64 {
        let snap = match &self.pre_amnesia {
            Some(s) => s,
            None => return f64::NAN,
        };
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for (&id, pre_emb) in &snap.embeddings {
            if self.seed_ids.contains(&id) { continue; } // skip seeds (they were never zeroed)
            if let Some(node) = self.nodes.get(&id) {
                let norm = embedding_norm(&node.embedding);
                if norm < 1e-8 {
                    // still at origin → cosine = 0
                    count += 1;
                } else {
                    sum += cosine_sim(&node.embedding, pre_emb);
                    count += 1;
                }
            }
        }
        if count == 0 { 0.0 } else { sum / count as f64 }
    }

    // ── Propagation with optional anchor freezing ──────────────

    fn propagate_embeddings(&mut self, freeze_seeds: bool) {
        let ids: Vec<usize> = self.nodes.keys().cloned().collect();
        let mut updates: Vec<(usize, Vec<f32>)> = Vec::with_capacity(ids.len());

        for &id in &ids {
            // Seeds are frozen during recovery — they broadcast but don't update
            if freeze_seeds && self.seed_ids.contains(&id) { continue; }

            let neighbors = match self.adj.get(&id) {
                Some(nbs) if !nbs.is_empty() => nbs,
                _ => continue,
            };

            let mut mean_emb = vec![0.0f32; EMBED_DIM];
            let mut count = 0usize;
            for &nb in neighbors {
                if let Some(nb_node) = self.nodes.get(&nb) {
                    for d in 0..EMBED_DIM { mean_emb[d] += nb_node.embedding[d]; }
                    count += 1;
                }
            }
            if count == 0 { continue; }

            let inv = 1.0 / count as f32;
            let node = &self.nodes[&id];
            let mut new_emb = Vec::with_capacity(EMBED_DIM);
            for d in 0..EMBED_DIM {
                let neighbor_mean = mean_emb[d] * inv;
                new_emb.push((1.0 - EMBED_LR) * node.embedding[d] + EMBED_LR * neighbor_mean);
            }
            project_into_ball(&mut new_emb);
            updates.push((id, new_emb));
        }

        for (id, emb) in updates {
            if let Some(node) = self.nodes.get_mut(&id) { node.embedding = emb; }
        }
    }

    // ── Vitality & Selection ──────────────────────────────────

    fn vitality(&self, n: &SimNode, id: usize) -> f32 {
        let prox = (1.0 - n.elite_proximity).max(0.0);
        let deg = self.degree(id);
        sigmoid(1.0 * n.energy + 0.8 * n.hausdorff - 1.2 * n.entropy_delta
            + 1.5 * prox + 2.0 * deg as f32 - 1.0 * n.toxicity)
    }

    fn elite_ids(&self) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = self.nodes.iter().map(|(&id, n)| (id, n.energy)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top = (self.nodes.len() / 20).max(2);
        indexed[..top.min(indexed.len())].iter().map(|(id, _)| *id).collect()
    }

    fn anti_elite_ids(&self) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = self.nodes.iter()
            .filter(|(_, n)| !n.is_original).map(|(&id, n)| (id, n.energy)).collect();
        if indexed.is_empty() {
            indexed = self.nodes.iter().map(|(&id, n)| (id, n.energy)).collect();
        }
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let bottom = (self.nodes.len() / 5).max(2);
        indexed[..bottom.min(indexed.len())].iter().map(|(id, _)| *id).collect()
    }

    // ── Phase 0: Energy decay + noise injection ──────────────

    fn evolve(&mut self, r: &mut Rng, _cycle: usize) {
        let ids_and_info: Vec<(usize, usize, bool)> = self.nodes.keys()
            .map(|&id| (id, self.degree(id), self.nodes[&id].is_original)).collect();
        for (id, deg, is_orig) in &ids_and_info {
            if let Some(n) = self.nodes.get_mut(id) {
                if *is_orig {
                    n.energy = (n.energy - r.next_f32(0.001, 0.005)).max(0.3);
                    if *deg > 2 { n.energy = (n.energy + r.next_f32(0.002, 0.008)).min(1.0); }
                } else {
                    n.energy = (n.energy - r.next_f32(0.005, 0.02)).max(0.0);
                    if *deg == 0 { n.toxicity = (n.toxicity + r.next_f32(0.005, 0.02)).min(1.0); }
                }
            }
        }
        for _ in 0..NOISE_INJECT_PER_CYCLE {
            let id = self.alloc_id();
            let mut emb = Vec::with_capacity(EMBED_DIM);
            for _ in 0..EMBED_DIM { emb.push(r.next_f32(-0.5, 0.5)); }
            project_into_ball(&mut emb);
            self.nodes.insert(id, SimNode {
                energy: r.next_f32(0.0, 0.20), hausdorff: r.next_f32(0.05, 0.25),
                entropy_delta: r.next_f32(0.5, 1.0), elite_proximity: r.next_f32(0.7, 1.0),
                toxicity: r.next_f32(0.3, 0.8), is_original: false,
                embedding: emb,            });
            self.adj.entry(id).or_default();

            // Inherit label from nearest labeled node
            let node_emb = self.nodes[&id].embedding.clone();
            let mut best_label = None;
            let mut best_dist = f64::MAX;
            let sample: Vec<(usize, usize)> = self.labels.iter().take(100).map(|(&lid, &lbl)| (lid, lbl)).collect();
            for (lid, lbl) in &sample {
                if let Some(lnode) = self.nodes.get(lid) {
                    let d = poincare_distance(&node_emb, &lnode.embedding);
                    if d < best_dist { best_dist = d; best_label = Some(*lbl); }
                }
            }
            if let Some(lbl) = best_label { self.labels.insert(id, lbl); }
            self.node_ages.insert(id, 0);
        }
    }

    // ── Birth modes ──────────────────────────────────────────

    fn birth_foam(&mut self, r: &mut Rng, n_create: usize, _cycle: usize) -> usize {
        let mut created = 0;
        for _ in 0..n_create {
            if let Some(seed) = self.voids.pop() {
                let id = self.alloc_id();
                let mut emb = seed.embedding;
                for d in 0..EMBED_DIM { emb[d] += r.next_f32(-0.03, 0.03); }
                project_into_ball(&mut emb);
                self.nodes.insert(id, SimNode {
                    energy: r.next_f32(0.15, 0.50), hausdorff: r.next_f32(0.3, 0.8),
                    entropy_delta: r.next_f32(0.1, 0.4), elite_proximity: r.next_f32(0.3, 0.6),
                    toxicity: r.next_f32(0.0, 0.2), is_original: false,
                    embedding: emb,                });
                self.adj.entry(id).or_default();
                // Inherit label
                let mut best_label = None;
                let mut best_dist = f64::MAX;
                for (&lid, &lbl) in &self.labels {
                    if let Some(lnode) = self.nodes.get(&lid) {
                        let d = poincare_distance(&self.nodes[&id].embedding, &lnode.embedding);
                        if d < best_dist { best_dist = d; best_label = Some(lbl); }
                    }
                }
                if let Some(lbl) = best_label { self.labels.insert(id, lbl); }
                self.node_ages.insert(id, 0);
                created += 1;
            }
        }
        created
    }

    fn birth_anchored(&mut self, r: &mut Rng, n_create: usize, _cycle: usize) -> usize {
        let elites = self.elite_ids();
        if elites.len() < 2 { return 0; }
        let mut created = 0;
        for _ in 0..n_create {
            if self.voids.is_empty() { break; }
            self.voids.pop();
            let i1 = elites[r.next_usize(elites.len())];
            let mut i2 = elites[r.next_usize(elites.len())];
            let mut att = 0;
            while i2 == i1 && att < 5 { i2 = elites[r.next_usize(elites.len())]; att += 1; }

            let p1 = self.nodes[&i1].clone();
            let p2 = self.nodes[&i2].clone();
            let mut emb = Vec::with_capacity(EMBED_DIM);
            for d in 0..EMBED_DIM { emb.push((p1.embedding[d] + p2.embedding[d]) / 2.0 + r.next_f32(-0.02, 0.02)); }
            project_into_ball(&mut emb);

            let id = self.alloc_id();
            self.nodes.insert(id, SimNode {
                energy: 0.8 * ((p1.energy + p2.energy) / 2.0),
                hausdorff: (p1.hausdorff + p2.hausdorff) / 2.0,
                entropy_delta: (p1.entropy_delta + p2.entropy_delta) / 2.0,
                elite_proximity: (p1.elite_proximity + p2.elite_proximity) / 2.0,
                toxicity: (p1.toxicity + p2.toxicity) / 2.0,
                is_original: false, embedding: emb,            });
            self.adj.entry(id).or_default();
            self.add_edge(id, i1);
            self.add_edge(id, i2);
            self.new_edges_this_cycle.push((id, i1));
            self.new_edges_this_cycle.push((id, i2));

            // Inherit label from nearest parent
            let l1 = self.labels.get(&i1).cloned();
            let l2 = self.labels.get(&i2).cloned();
            match (l1, l2) {
                (Some(la), Some(lb)) => {
                    let d1 = poincare_distance(&self.nodes[&id].embedding, &p1.embedding);
                    let d2 = poincare_distance(&self.nodes[&id].embedding, &p2.embedding);
                    self.labels.insert(id, if d1 <= d2 { la } else { lb });
                }
                (Some(l), None) | (None, Some(l)) => { self.labels.insert(id, l); }
                _ => {}
            }
            self.node_ages.insert(id, 0);
            created += 1;
        }
        created
    }

    fn birth_anti_anchored(&mut self, r: &mut Rng, n_create: usize, _cycle: usize) -> usize {
        let worst = self.anti_elite_ids();
        if worst.len() < 2 { return self.birth_foam(r, n_create, _cycle); }
        let mut created = 0;
        for _ in 0..n_create {
            if self.voids.is_empty() { break; }
            self.voids.pop();
            let i1 = worst[r.next_usize(worst.len())];
            let mut i2 = worst[r.next_usize(worst.len())];
            let mut att = 0;
            while i2 == i1 && att < 5 { i2 = worst[r.next_usize(worst.len())]; att += 1; }

            let mut emb = Vec::with_capacity(EMBED_DIM);
            for _ in 0..EMBED_DIM { emb.push(r.next_f32(-0.6, 0.6)); }
            project_into_ball(&mut emb);

            let id = self.alloc_id();
            self.nodes.insert(id, SimNode {
                energy: r.next_f32(0.05, 0.25), hausdorff: r.next_f32(0.05, 0.3),
                entropy_delta: r.next_f32(0.5, 1.0), elite_proximity: r.next_f32(0.7, 1.0),
                toxicity: r.next_f32(0.2, 0.6), is_original: false,
                embedding: emb,            });
            self.adj.entry(id).or_default();
            self.add_edge(id, i1);
            self.add_edge(id, i2);
            self.new_edges_this_cycle.push((id, i1));
            self.new_edges_this_cycle.push((id, i2));

            let l1 = self.labels.get(&i1).cloned();
            let l2 = self.labels.get(&i2).cloned();
            match (l1, l2) {
                (Some(la), Some(_)) => { self.labels.insert(id, la); }
                (Some(l), None) | (None, Some(l)) => { self.labels.insert(id, l); }
                _ => {}
            }
            self.node_ages.insert(id, 0);
            created += 1;
        }
        created
    }

    // ══════════════════════════════════════════════════════════
    //  SCULPT CYCLE (Phase 1)
    // ══════════════════════════════════════════════════════════

    fn sculpt_cycle(&mut self, cycle: usize) -> RecoveryRow {
        let mut r = Rng::new(42u64.wrapping_mul(cycle as u64 + 7919));

        // Evolve
        self.evolve(&mut r, cycle);

        // Judgment + pruning
        let mut to_remove = Vec::new();
        let mut vits = Vec::new();
        let ids: Vec<usize> = self.nodes.keys().cloned().collect();
        for &id in &ids {
            let n = &self.nodes[&id];
            let v = self.vitality(n, id);
            vits.push(v);
            if n.is_original { continue; }
            let deg = self.degree(id);
            if (v < VIT_THRESHOLD && n.energy < ENG_THRESHOLD && deg == 0) || (n.toxicity > 0.8 && deg == 0) {
                to_remove.push(id);
            }
        }
        let max_del = (self.nodes.len() as f32 * MAX_DELETION_RATE).round() as usize;
        if to_remove.len() > max_del { to_remove.truncate(max_del); }
        let sacrificed = to_remove.len();
        for &id in &to_remove {
            if let Some(n) = self.nodes.get(&id) {
                self.voids.push(VoidSeed { embedding: n.embedding.clone() });
            }
            self.remove_node(id);
        }
        if self.voids.len() > 500 { self.voids.drain(0..self.voids.len() - 500); }

        // Generation
        self.new_edges_this_cycle.clear();
        let gen_target = ((sacrificed as f32 * 0.9).round() as usize + 5).min(self.voids.len());
        let _created = match self.mode {
            TgcMode::Normal => self.birth_anchored(&mut r, gen_target, cycle),
            TgcMode::Off => self.birth_foam(&mut r, gen_target, cycle),
            TgcMode::Inverted => self.birth_anti_anchored(&mut r, gen_target, cycle),
        };

        // Propagation
        for _ in 0..PROPAG_STEPS { self.propagate_embeddings(false); }
        for age in self.node_ages.values_mut() { *age += 1; }

        let total_edges: usize = self.adj.values().map(|s| s.len()).sum::<usize>() / 2;
        RecoveryRow {
            mode: self.mode.label().to_string(),
            phase: "sculpt".to_string(),
            cycle,
            nca: self.compute_nca(),
            mean_norm: self.mean_embedding_norm(),
            cosine_recovery: f64::NAN,
            auc: self.compute_auc(),
            total_nodes: self.nodes.len(),
            total_edges,
        }
    }

    // ══════════════════════════════════════════════════════════
    //  SNAPSHOT (Phase 2) — Select seeds + record state
    // ══════════════════════════════════════════════════════════

    fn phase_snapshot(&mut self) -> PreAmnesiaSnapshot {
        let n_seeds = ((self.n_original as f64 * SEED_FRACTION) as usize).max(1);

        // Rank original nodes by degree (highest = best hub)
        let mut original_by_degree: Vec<(usize, usize)> = self.nodes.iter()
            .filter(|(_, n)| n.is_original)
            .map(|(&id, _)| (id, self.degree(id)))
            .collect();
        original_by_degree.sort_by(|a, b| b.1.cmp(&a.1));

        for &(id, _) in original_by_degree.iter().take(n_seeds) {
            self.seed_ids.insert(id);
        }

        let nca = self.compute_nca();
        let auc = self.compute_auc();
        let mean_norm = self.mean_embedding_norm();

        // Save ALL node embeddings for cosine recovery (only original non-seeds measured)
        let embeddings: HashMap<usize, Vec<f32>> = self.nodes.iter()
            .filter(|(_, n)| n.is_original)
            .map(|(&id, n)| (id, n.embedding.clone()))
            .collect();

        PreAmnesiaSnapshot { nca, auc, mean_norm, embeddings }
    }

    // ══════════════════════════════════════════════════════════
    //  AMNESIA (Phase 3) — The 30-line core
    // ══════════════════════════════════════════════════════════

    fn apply_amnesia(&mut self) {
        let zero = vec![0.0f32; EMBED_DIM];
        let mut zeroed = 0usize;
        let mut kept = 0usize;

        let ids: Vec<usize> = self.nodes.keys().cloned().collect();
        for id in ids {
            if self.seed_ids.contains(&id) {
                kept += 1;
                continue;
            }
            if let Some(node) = self.nodes.get_mut(&id) {
                node.embedding = zero.clone();
                zeroed += 1;
            }
        }

        println!("    AMNESIA: zeroed {} embeddings, kept {} seeds intact", zeroed, kept);
    }

    // ══════════════════════════════════════════════════════════
    //  RECOVERY CYCLE (Phase 4) — Propagation only
    // ══════════════════════════════════════════════════════════

    fn recovery_cycle(&mut self, rec_cycle: usize) -> RecoveryRow {
        // Pure propagation — no evolve, no birth, no death
        for _ in 0..PROPAG_STEPS {
            self.propagate_embeddings(true); // seeds frozen
        }

        let total_edges: usize = self.adj.values().map(|s| s.len()).sum::<usize>() / 2;
        RecoveryRow {
            mode: self.mode.label().to_string(),
            phase: "recovery".to_string(),
            cycle: rec_cycle,
            nca: self.compute_nca(),
            mean_norm: self.mean_embedding_norm(),
            cosine_recovery: self.cosine_recovery(),
            auc: self.compute_auc(),
            total_nodes: self.nodes.len(),
            total_edges,
        }
    }
}

// ══════════════════════════════════════════════════════════════════
//  Experiment Runner
// ══════════════════════════════════════════════════════════════════

fn run_amnesia(
    mode: TgcMode,
    train_edges: &[(usize, usize)],
    test_edges: &[(usize, usize)],
    labels: &HashMap<usize, usize>,
) -> Vec<RecoveryRow> {
    let label = mode.label();
    let _ = std::fs::create_dir_all("experiments");
    let csv_path = format!("experiments/telemetry_amnesia_{}.csv", label);

    println!("  [{}] Phase 1: Sculpting topology ({} cycles)...", label, SCULPT_CYCLES);
    let mut sim = AmnesiaSim::new(mode, train_edges, test_edges.to_vec(), labels.clone());

    let auc0 = sim.compute_auc();
    let nca0 = sim.compute_nca();
    println!("    Initial: {} nodes, AUC_0={:.4}, NCA_0={:.4}", sim.nodes.len(), auc0, nca0);

    let mut all_rows: Vec<RecoveryRow> = Vec::new();

    // ── PHASE 1: SCULPTING ──
    for cycle in 1..=SCULPT_CYCLES {
        let row = sim.sculpt_cycle(cycle);
        if cycle <= 3 || cycle % 20 == 0 || cycle == SCULPT_CYCLES {
            println!("    sculpt {:03} N={:5} E={:5} NCA={:.4} AUC={:.4} norm={:.4}",
                     cycle, row.total_nodes, row.total_edges, row.nca, row.auc, row.mean_norm);
        }
        all_rows.push(row);
    }

    // ── PHASE 2: SNAPSHOT ──
    println!("  [{}] Phase 2: Taking pre-amnesia snapshot...", label);
    let snapshot = sim.phase_snapshot();
    println!("    Pre-amnesia: NCA={:.4}  AUC={:.4}  mean_norm={:.4}  seeds={}",
             snapshot.nca, snapshot.auc, snapshot.mean_norm, sim.seed_ids.len());
    sim.pre_amnesia = Some(snapshot);

    // ── PHASE 3: AMNESIA ──
    println!("  [{}] Phase 3: CORTEX WIPE — zeroing all non-seed embeddings...", label);
    sim.apply_amnesia();

    let total_edges: usize = sim.adj.values().map(|s| s.len()).sum::<usize>() / 2;
    let amnesia_row = RecoveryRow {
        mode: label.to_string(),
        phase: "amnesia".to_string(),
        cycle: 0,
        nca: sim.compute_nca(),
        mean_norm: sim.mean_embedding_norm(),
        cosine_recovery: 0.0,
        auc: sim.compute_auc(),
        total_nodes: sim.nodes.len(),
        total_edges,
    };
    println!("    Post-wipe: NCA={:.4}  AUC={:.4}  mean_norm={:.4}",
             amnesia_row.nca, amnesia_row.auc, amnesia_row.mean_norm);
    all_rows.push(amnesia_row);

    // ── PHASE 4: RECOVERY ──
    println!("  [{}] Phase 4: Recovery ({} propagation cycles, seeds frozen)...", label, RECOVERY_CYCLES);
    for rec in 1..=RECOVERY_CYCLES {
        let row = sim.recovery_cycle(rec);
        if rec <= 5 || rec % 10 == 0 || rec == RECOVERY_CYCLES {
            println!("    rec {:03} NCA={:.4} AUC={:.4} norm={:.4} cos_rec={:.4}",
                     rec, row.nca, row.auc, row.mean_norm, row.cosine_recovery);
        }
        all_rows.push(row);
    }

    // Write CSV
    let mut f = OpenOptions::new().write(true).create(true).truncate(true).open(&csv_path).unwrap();
    writeln!(f, "mode,phase,cycle,nca,mean_norm,cosine_recovery,auc,total_nodes,total_edges").unwrap();
    for row in &all_rows {
        writeln!(f, "{},{},{},{:.6},{:.6},{:.6},{:.6},{},{}",
                 row.mode, row.phase, row.cycle, row.nca, row.mean_norm,
                 row.cosine_recovery, row.auc, row.total_nodes, row.total_edges).unwrap();
    }
    println!("  Saved: {}", csv_path);
    println!();

    all_rows
}

// ══════════════════════════════════════════════════════════════════
//  Main
// ══════════════════════════════════════════════════════════════════

fn main() {
    println!("=================================================================");
    println!("  EXPERIMENT: Controlled Amnesia");
    println!("  Hypothesis: Topology is the True Memory");
    println!("  Protocol: Sculpt({}) -> Snapshot -> Amnesia -> Recovery({})",
             SCULPT_CYCLES, RECOVERY_CYCLES);
    println!("  Seeds: top-{:.0}% highest-degree original nodes", SEED_FRACTION * 100.0);
    println!("  Conditions: Normal | Off | Inverted");
    println!("=================================================================");
    println!();

    // Load Cora
    let (train_edges, test_edges) =
        if std::path::Path::new("experiments/train_edges.csv").exists()
            && std::path::Path::new("experiments/test_edges.csv").exists()
        {
            println!("  Loading Cora dataset from experiments/...");
            let train = load_edges("experiments/train_edges.csv");
            let test = load_edges("experiments/test_edges.csv");
            println!("    Train: {} edges, Test: {} edges", train.len(), test.len());
            (train, test)
        } else {
            println!("  Cora CSVs not found. Generating synthetic SBM...");
            let mut r = Rng::new(42);
            generate_synthetic(&mut r)
        };

    let labels = load_labels("experiments/node_labels.csv");
    if labels.is_empty() {
        println!("  WARNING: No labels loaded. NCA will be NaN.");
    } else {
        let n_classes = labels.values().collect::<HashSet<_>>().len();
        println!("  Loaded {} node labels ({} classes)", labels.len(), n_classes);
    }
    println!();

    // Run all 3 conditions
    println!("--- N: NORMAL (elite-anchored topology) ---");
    let rn = run_amnesia(TgcMode::Normal, &train_edges, &test_edges, &labels);

    println!("--- O: OFF (foam topology, no structural guidance) ---");
    let ro = run_amnesia(TgcMode::Off, &train_edges, &test_edges, &labels);

    println!("--- I: INVERTED (anti-anchored topology) ---");
    let ri = run_amnesia(TgcMode::Inverted, &train_edges, &test_edges, &labels);

    // ══════════════════════════════════════════════════════════
    //  VERDICT
    // ══════════════════════════════════════════════════════════

    let extract = |rows: &[RecoveryRow]| -> (f64, f64, f64, f64, f64) {
        // Find pre-amnesia (last sculpt row)
        let nca_pre = rows.iter().filter(|r| r.phase == "sculpt").last().map_or(f64::NAN, |r| r.nca);
        let _auc_pre = rows.iter().filter(|r| r.phase == "sculpt").last().map_or(f64::NAN, |r| r.auc);
        // Post-amnesia (the amnesia row)
        let nca_post0 = rows.iter().find(|r| r.phase == "amnesia").map_or(f64::NAN, |r| r.nca);
        // Final recovery
        let nca_rec = rows.iter().filter(|r| r.phase == "recovery").last().map_or(f64::NAN, |r| r.nca);
        let cos_rec = rows.iter().filter(|r| r.phase == "recovery").last().map_or(f64::NAN, |r| r.cosine_recovery);
        // Recovery%
        let denom = nca_pre - nca_post0;
        let recovery_pct = if denom.abs() > 1e-6 { (nca_rec - nca_post0) / denom * 100.0 } else { 0.0 };
        (nca_pre, nca_post0, nca_rec, recovery_pct, cos_rec)
    };

    let (nca_pre_n, nca_post0_n, nca_rec_n, recovery_n, cos_n) = extract(&rn);
    let (nca_pre_o, nca_post0_o, nca_rec_o, recovery_o, cos_o) = extract(&ro);
    let (nca_pre_i, nca_post0_i, nca_rec_i, recovery_i, cos_i) = extract(&ri);

    println!("=================================================================");
    println!("  VERDICT: Topology as True Memory");
    println!("=================================================================");
    println!();
    println!("  {:12} | {:>8} | {:>9} | {:>9} | {:>9} | {:>7}", "Mode", "NCA_pre", "NCA_post0", "NCA_rec50", "Recovery%", "Cos_rec");
    println!("  {:-<12}-+-{:-<8}-+-{:-<9}-+-{:-<9}-+-{:-<9}-+-{:-<7}", "", "", "", "", "", "");
    println!("  {:12} | {:8.4} | {:9.4} | {:9.4} | {:8.1}% | {:7.4}", "Normal", nca_pre_n, nca_post0_n, nca_rec_n, recovery_n, cos_n);
    println!("  {:12} | {:8.4} | {:9.4} | {:9.4} | {:8.1}% | {:7.4}", "Off", nca_pre_o, nca_post0_o, nca_rec_o, recovery_o, cos_o);
    println!("  {:12} | {:8.4} | {:9.4} | {:9.4} | {:8.1}% | {:7.4}", "Inverted", nca_pre_i, nca_post0_i, nca_rec_i, recovery_i, cos_i);
    println!();

    // Hypothesis: Normal recovery >> Off and Inverted
    let h_recovery = recovery_n > recovery_o && recovery_n > recovery_i;
    let h_nca = nca_rec_n > nca_rec_o && nca_rec_o > nca_rec_i;
    let h_cosine = cos_n > cos_o && cos_n > cos_i;
    let score = [h_recovery, h_nca, h_cosine].iter().filter(|&&v| v).count();

    println!("  ╔══════════════════════════════════════════════════════╗");
    if h_recovery {
        println!("  ║ Recovery%:  Normal >> Off,Inverted  — CONFIRMED     ║");
    } else {
        println!("  ║ Recovery%:  N>O={} N>I={}  — NOT CONFIRMED           ║",
            recovery_n > recovery_o, recovery_n > recovery_i);
    }
    if h_nca {
        println!("  ║ NCA_rec50:  N > O > I              — CONFIRMED     ║");
    } else {
        println!("  ║ NCA_rec50:  N>O={} O>I={}  — NOT CONFIRMED           ║",
            nca_rec_n > nca_rec_o, nca_rec_o > nca_rec_i);
    }
    if h_cosine {
        println!("  ║ Cos_rec:    Normal >> Off,Inverted  — CONFIRMED     ║");
    } else {
        println!("  ║ Cos_rec:    N>O={} N>I={}  — NOT CONFIRMED           ║",
            cos_n > cos_o, cos_n > cos_i);
    }
    println!("  ╚══════════════════════════════════════════════════════╝");
    println!();

    if score == 3 {
        println!("  >>> TOPOLOGY IS THE TRUE MEMORY — FULLY CONFIRMED ({}/3) <<<", score);
    } else if score >= 2 {
        println!("  >>> TOPOLOGY IS THE TRUE MEMORY — STRONGLY CONFIRMED ({}/3) <<<", score);
    } else if score >= 1 {
        println!("  >>> PARTIALLY CONFIRMED ({}/3) <<<", score);
    } else {
        println!("  >>> NOT CONFIRMED (0/3) <<<");
    }

    println!();
    println!("=================================================================");
    println!("  Run: python3 experiments/analysis_amnesia.py");
    println!("=================================================================");
}
