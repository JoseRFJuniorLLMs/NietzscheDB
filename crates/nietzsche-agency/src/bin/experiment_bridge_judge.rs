// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Experiment: Bridge Judge — Angular Fidelity of DecompressionChamber 128D
//!
//! Loads 90 realistic 1536D embeddings from embeddings_realistas.json
//! (Sistemas/Filosofia/Cibernética), applies Mean Centering + JL 128D,
//! measures Spearman ρ, Recall@10, and cluster structure preservation.

use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;

const DIM_HIGH: usize = 1024;
const DIM_LOW: usize = 128;
const SEED: u64 = 42;
const RECALL_K: usize = 10;

// Sculpting parameters
const SCULPT_EPOCHS: usize = 100;
const TRIPLET_MARGIN: f64 = 0.1; // reduced for real diffuse data
const SCULPT_LR: f64 = 0.01;
const INITIAL_RADIUS: f64 = 0.4;
const MAX_NORM: f64 = 0.95; // hard clamp to keep inside ball

// Geodesic navigation parameters
const KNN_K: usize = 15;
const RADIAL_LAMBDA: f64 = 2.0; // radial penalty factor

// ══════════════════════════════════════════════════════════════════
//  JSON Parsing (minimal, no serde dependency)
// ══════════════════════════════════════════════════════════════════

struct EmbEntry {
    id: usize,
    tag: String,
    text: String,
    vector: Vec<f64>,
}

fn parse_embeddings(path: &str) -> Vec<EmbEntry> {
    let raw = fs::read_to_string(path).expect("Failed to read embeddings JSON");
    let mut entries: Vec<EmbEntry> = Vec::new();

    // Simple state-machine parser for the specific JSON structure:
    // [{"id": N, "tag": "...", "vector": [f, f, ...]}, ...]
    let mut pos = 0;
    let bytes = raw.as_bytes();
    let len = bytes.len();

    while pos < len {
        // Find next "id"
        match raw[pos..].find("\"id\"") {
            None => break,
            Some(offset) => pos += offset + 4,
        }

        // Skip to colon and whitespace
        while pos < len && bytes[pos] != b':' { pos += 1; }
        pos += 1; // skip ':'
        while pos < len && (bytes[pos] == b' ' || bytes[pos] == b'\n' || bytes[pos] == b'\r') { pos += 1; }

        // Read id number
        let id_start = pos;
        while pos < len && bytes[pos].is_ascii_digit() { pos += 1; }
        let id: usize = raw[id_start..pos].parse().unwrap();

        // Find "tag"
        let tag_offset = raw[pos..].find("\"tag\"").unwrap();
        pos += tag_offset + 4;
        while pos < len && bytes[pos] != b':' { pos += 1; }
        pos += 1;
        while pos < len && bytes[pos] != b'"' { pos += 1; }
        pos += 1; // skip opening quote
        let tag_start = pos;
        while pos < len && bytes[pos] != b'"' { pos += 1; }
        let tag = raw[tag_start..pos].to_string();
        pos += 1; // skip closing quote

        // Find "text"
        let text_offset = raw[pos..].find("\"text\"").unwrap();
        pos += text_offset + 6;
        while pos < len && bytes[pos] != b':' { pos += 1; }
        pos += 1;
        while pos < len && bytes[pos] != b'"' { pos += 1; }
        pos += 1; // skip opening quote
        let text_start = pos;
        while pos < len && bytes[pos] != b'"' { pos += 1; }
        let text = raw[text_start..pos].to_string();
        pos += 1; // skip closing quote

        // Find "vector"
        let vec_offset = raw[pos..].find("\"vector\"").unwrap();
        pos += vec_offset + 8;
        while pos < len && bytes[pos] != b'[' { pos += 1; }
        pos += 1; // skip '['

        // Parse array of floats
        let mut vector: Vec<f64> = Vec::with_capacity(DIM_HIGH);
        loop {
            // Skip whitespace
            while pos < len && (bytes[pos] == b' ' || bytes[pos] == b'\n' || bytes[pos] == b'\r' || bytes[pos] == b',') {
                pos += 1;
            }
            if pos >= len || bytes[pos] == b']' { pos += 1; break; }

            // Read number (may include minus, digits, dot, e, E, +, -)
            let num_start = pos;
            while pos < len && bytes[pos] != b',' && bytes[pos] != b']' && bytes[pos] != b' ' && bytes[pos] != b'\n' {
                pos += 1;
            }
            let num_str = &raw[num_start..pos];
            let val: f64 = num_str.parse().unwrap_or_else(|e| panic!("Parse error '{}': {}", num_str, e));
            vector.push(val);
        }

        entries.push(EmbEntry { id, tag, text, vector });
    }

    entries
}

const NUM_CLUSTERS: usize = 7;

fn tag_to_cluster(tag: &str) -> usize {
    match tag {
        t if t.starts_with("Sis") => 0,
        t if t.starts_with("Fil") => 1,
        t if t.starts_with("Cib") => 2,
        t if t.starts_with("Uns") => 3, // UnsafeRust
        t if t.starts_with("Con") => 4, // Continental
        t if t.starts_with("Dia") => 5, // Diary
        t if t.starts_with("Inf") => 6, // InfoTheory
        _ => 2,
    }
}

const CLUSTER_NAMES: [&str; 7] = [
    "Sistemas", "Filosofia", "Cibernetica", "UnsafeRust",
    "Continental", "Diary", "InfoTheory",
];

// ══════════════════════════════════════════════════════════════════
//  RNG / Vector Ops / Stats (same as before)
// ══════════════════════════════════════════════════════════════════

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_gaussian(&mut self) -> f64 {
        let u1 = (((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)).max(1e-15);
        let u2 = ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn l2_norm(v: &[f64]) -> f64 { v.iter().map(|&x| x * x).sum::<f64>().sqrt() }

fn l2_normalize(v: &mut [f64]) {
    let n = l2_norm(v);
    if n > 1e-15 { for x in v.iter_mut() { *x /= n; } }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0; let mut na = 0.0; let mut nb = 0.0;
    for i in 0..a.len() { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    let d = na.sqrt() * nb.sqrt();
    if d < 1e-15 { 0.0 } else { dot / d }
}

fn rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut r = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (idx[j].1 - idx[i].1).abs() < 1e-15 { j += 1; }
        let avg = (i + 1 + j) as f64 / 2.0;
        for k in i..j { r[idx[k].0] = avg; }
        i = j;
    }
    r
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut num, mut dx2, mut dy2) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let dx = x[i] - mx; let dy = y[i] - my;
        num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
    }
    let d = dx2.sqrt() * dy2.sqrt();
    if d < 1e-15 { 0.0 } else { num / d }
}

fn spearman(x: &[f64], y: &[f64]) -> f64 { pearson(&rank(x), &rank(y)) }

fn recall_at_k(orig: &[Vec<f64>], proj: &[Vec<f64>], k: usize) -> f64 {
    let n = orig.len();
    let mut total = 0.0;
    for i in 0..n {
        let mut os: Vec<(usize, f64)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, cosine_similarity(&orig[i], &orig[j]))).collect();
        let mut ps: Vec<(usize, f64)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, cosine_similarity(&proj[i], &proj[j]))).collect();
        os.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let otk: HashSet<usize> = os.iter().take(k).map(|&(j, _)| j).collect();
        let ptk: HashSet<usize> = ps.iter().take(k).map(|&(j, _)| j).collect();
        total += otk.intersection(&ptk).count() as f64 / k as f64;
    }
    total / n as f64
}

fn cluster_purity_at_k(vecs: &[Vec<f64>], labels: &[usize], k: usize) -> f64 {
    let n = vecs.len();
    let mut total = 0.0;
    for i in 0..n {
        let mut sims: Vec<(usize, f64)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, cosine_similarity(&vecs[i], &vecs[j]))).collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let same = sims.iter().take(k).filter(|&&(j, _)| labels[j] == labels[i]).count();
        total += same as f64 / k as f64;
    }
    total / n as f64
}

fn generate_proj_matrix(rng: &mut Rng, din: usize, dout: usize) -> Vec<Vec<f64>> {
    let s = 1.0 / (dout as f64).sqrt();
    (0..din).map(|_| (0..dout).map(|_| rng.next_gaussian() * s).collect()).collect()
}

fn project(v: &[f64], m: &[Vec<f64>], d: usize) -> Vec<f64> {
    let mut r = vec![0.0; d];
    for i in 0..v.len() { for j in 0..d { r[j] += v[i] * m[i][j]; } }
    r
}

fn pairwise_sims(vecs: &[Vec<f64>]) -> Vec<f64> {
    let n = vecs.len();
    let mut s = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n { for j in (i+1)..n { s.push(cosine_similarity(&vecs[i], &vecs[j])); } }
    s
}

fn stats(s: &[f64]) -> (f64, f64, f64, f64) {
    let n = s.len() as f64;
    let m = s.iter().sum::<f64>() / n;
    let sd = (s.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n).sqrt();
    (m, sd, s.iter().cloned().fold(f64::MAX, f64::min), s.iter().cloned().fold(f64::MIN, f64::max))
}

// ══════════════════════════════════════════════════════════════════
//  Poincaré Ball Geometry
// ══════════════════════════════════════════════════════════════════

fn poincare_distance(u: &[f64], v: &[f64]) -> f64 {
    let mut diff_sq = 0.0;
    let mut nu = 0.0;
    let mut nv = 0.0;
    for i in 0..u.len() {
        diff_sq += (u[i] - v[i]).powi(2);
        nu += u[i] * u[i];
        nv += v[i] * v[i];
    }
    let denom = (1.0 - nu).max(1e-10) * (1.0 - nv).max(1e-10);
    let arg = 1.0 + 2.0 * diff_sq / denom;
    arg.max(1.0).acosh()
}

fn project_into_poincare(v: &mut [f64]) {
    let n = l2_norm(v);
    if n >= MAX_NORM {
        let s = (MAX_NORM - 1e-5) / n;
        for x in v.iter_mut() { *x *= s; }
    }
}

/// Gradient of poincaré distance d(u,v) with respect to `u` (if `wrt_first=true`) or `v`.
/// Scaled by the inverse conformal factor (Riemannian → Euclidean) for SGD in ambient coords.
fn poincare_grad(u: &[f64], v: &[f64], wrt_first: bool) -> Vec<f64> {
    let dim = u.len();
    let mut diff_sq = 0.0;
    let mut nu = 0.0;
    let mut nv = 0.0;
    for i in 0..dim {
        diff_sq += (u[i] - v[i]).powi(2);
        nu += u[i] * u[i];
        nv += v[i] * v[i];
    }

    let alpha = 1.0 - nu;
    let beta = 1.0 - nv;
    let gamma = 1.0 + 2.0 * diff_sq / (alpha.max(1e-10) * beta.max(1e-10));

    // d/d(arg) of acosh(arg) = 1/sqrt(arg²-1)
    let d_acosh = 1.0 / (gamma * gamma - 1.0).max(1e-10).sqrt();

    let mut grad = vec![0.0; dim];

    if wrt_first {
        // ∂γ/∂u_i = (2/(α·β)) * [2(u_i - v_i) + 2u_i·||u-v||²/α]
        //         = (2/(α·β)) * 2 * [(u_i - v_i) + u_i·||u-v||²/α]
        let ab = (alpha * beta).max(1e-10);
        for i in 0..dim {
            let dg = 2.0 / ab * (2.0 * (u[i] - v[i]) + 2.0 * u[i] * diff_sq / alpha.max(1e-10));
            // Euclidean grad of d = d_acosh * dγ/du
            // Riemannian SGD: multiply by (α²/4) — the conformal factor squared
            let conf = alpha * alpha / 4.0;
            grad[i] = conf * d_acosh * dg;
        }
    } else {
        // ∂γ/∂v_i = (2/(α·β)) * [-2(u_i - v_i) + 2v_i·||u-v||²/β]
        let ab = (alpha * beta).max(1e-10);
        for i in 0..dim {
            let dg = 2.0 / ab * (-2.0 * (u[i] - v[i]) + 2.0 * v[i] * diff_sq / beta.max(1e-10));
            let conf = beta * beta / 4.0;
            grad[i] = conf * d_acosh * dg;
        }
    }

    grad
}

fn generate_triplets(labels: &[usize], n: usize, rng: &mut Rng) -> Vec<(usize, usize, usize)> {
    let mut triplets = Vec::new();
    let mut by_label: Vec<Vec<usize>> = vec![Vec::new(); NUM_CLUSTERS];
    for i in 0..n { by_label[labels[i]].push(i); }

    // Multiple passes for more triplets
    for _pass in 0..3 {
        for a in 0..n {
            let la = labels[a];
            let pos_pool = &by_label[la];
            if pos_pool.len() < 2 { continue; }
            let mut pi = a;
            while pi == a {
                pi = pos_pool[(rng.next_u64() as usize) % pos_pool.len()];
            }
            // Pick random negative from any different cluster
            let mut neg_label = la;
            while neg_label == la {
                neg_label = (rng.next_u64() as usize) % NUM_CLUSTERS;
                if by_label[neg_label].is_empty() { neg_label = la; }
            }
            let neg_pool = &by_label[neg_label];
            let ni = neg_pool[(rng.next_u64() as usize) % neg_pool.len()];
            triplets.push((a, pi, ni));
        }
    }
    triplets
}

fn mean_distances(ball: &[Vec<f64>], labels: &[usize]) -> (f64, f64) {
    let n = ball.len();
    let (mut intra, mut inter) = (0.0, 0.0);
    let (mut ci, mut ce) = (0usize, 0usize);
    for i in 0..n {
        for j in (i+1)..n {
            let d = poincare_distance(&ball[i], &ball[j]);
            if labels[i] == labels[j] { intra += d; ci += 1; }
            else { inter += d; ce += 1; }
        }
    }
    (if ci > 0 { intra / ci as f64 } else { 0.0 },
     if ce > 0 { inter / ce as f64 } else { 0.0 })
}

fn print_ball_stats(ball: &[Vec<f64>], labels: &[usize], names: &[&str], title: &str) {
    println!("  {} — Poincaré ball stats:", title);
    let norms: Vec<f64> = ball.iter().map(|v| l2_norm(v)).collect();
    let mean_r = norms.iter().sum::<f64>() / norms.len() as f64;
    let max_r = norms.iter().cloned().fold(0.0_f64, f64::max);
    let min_r = norms.iter().cloned().fold(f64::MAX, f64::min);
    println!("    Norms: mean={:.4}, min={:.4}, max={:.4}", mean_r, min_r, max_r);

    let (intra_d, inter_d) = mean_distances(ball, labels);
    println!("    Distances: intra={:.4}, inter={:.4}, ratio={:.4}",
             intra_d, inter_d, if intra_d > 0.0 { inter_d / intra_d } else { 0.0 });

    for c in 0..names.len() {
        let cn: Vec<f64> = norms.iter().enumerate()
            .filter(|&(i, _)| labels[i] == c).map(|(_, &n)| n).collect();
        if cn.is_empty() { continue; }
        let cr = cn.iter().sum::<f64>() / cn.len() as f64;
        println!("    {:12} — mean_r={:.4} (n={})", names[c], cr, cn.len());
    }
}

fn poincare_purity_at_k(ball: &[Vec<f64>], labels: &[usize], k: usize) -> f64 {
    let n = ball.len();
    let mut total = 0.0;
    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, poincare_distance(&ball[i], &ball[j]))).collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let same = dists.iter().take(k).filter(|&&(j, _)| labels[j] == labels[i]).count();
        total += same as f64 / k as f64;
    }
    total / n as f64
}

// ══════════════════════════════════════════════════════════════════
//  Geodesic Navigation — KNN Graph + Dijkstra
// ══════════════════════════════════════════════════════════════════

#[derive(PartialEq)]
struct DijkState {
    cost: f64,
    node: usize,
}

impl Eq for DijkState {}

impl PartialOrd for DijkState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap (BinaryHeap is max-heap by default)
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

/// Build KNN graph with Poincaré distances.
/// Returns adjacency list: adj[u] = vec![(v, weight), ...]
/// Weight = d_P(u,v) * (1 + λ * |‖u‖ - ‖v‖|)
fn build_knn_graph(ball: &[Vec<f64>], k: usize, lambda: f64) -> Vec<Vec<(usize, f64)>> {
    let n = ball.len();
    let norms: Vec<f64> = ball.iter().map(|v| l2_norm(v)).collect();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

    for u in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&v| v != u)
            .map(|v| (v, poincare_distance(&ball[u], &ball[v])))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for &(v, dp) in dists.iter().take(k) {
            let radial_penalty = 1.0 + lambda * (norms[u] - norms[v]).abs();
            let weight = dp * radial_penalty;
            adj[u].push((v, weight));
        }
    }

    // Make symmetric: if u→v exists but v→u doesn't, add v→u
    let mut to_add: Vec<(usize, usize, f64)> = Vec::new();
    for u in 0..n {
        for &(v, w) in &adj[u] {
            if !adj[v].iter().any(|&(nb, _)| nb == u) {
                to_add.push((v, u, w));
            }
        }
    }
    for (from, to, w) in to_add {
        adj[from].push((to, w));
    }

    adj
}

/// Dijkstra shortest path. Returns (path, total_cost) or None.
fn dijkstra(adj: &[Vec<(usize, f64)>], src: usize, dst: usize) -> Option<(Vec<usize>, f64)> {
    let n = adj.len();
    let mut dist = vec![f64::MAX; n];
    let mut prev = vec![usize::MAX; n];
    let mut heap = BinaryHeap::new();

    dist[src] = 0.0;
    heap.push(DijkState { cost: 0.0, node: src });

    while let Some(DijkState { cost, node }) = heap.pop() {
        if node == dst {
            // Reconstruct path
            let mut path = Vec::new();
            let mut cur = dst;
            while cur != usize::MAX {
                path.push(cur);
                cur = prev[cur];
            }
            path.reverse();
            return Some((path, cost));
        }

        if cost > dist[node] { continue; }

        for &(next, w) in &adj[node] {
            let new_cost = cost + w;
            if new_cost < dist[next] {
                dist[next] = new_cost;
                prev[next] = node;
                heap.push(DijkState { cost: new_cost, node: next });
            }
        }
    }

    None
}

/// Find entry index by original ID
fn find_by_id(entries: &[EmbEntry], target_id: usize) -> Option<usize> {
    entries.iter().position(|e| e.id == target_id)
}

// ══════════════════════════════════════════════════════════════════
//  Main
// ══════════════════════════════════════════════════════════════════

fn main() {
    println!("=================================================================");
    println!("  BRIDGE JUDGE: Biosfera 389 — DecompressionChamber 128D");
    println!("  Source: biosfera_390.json (389 × 1024D, 7 clusters)");
    println!("  Pipeline: MeanCenter → JL(128D) → L2Norm → Sculpt");
    println!("  Criteria: Spearman ρ ≥ 0.85, Recall@{} ≥ 0.80", RECALL_K);
    println!("=================================================================");
    println!();

    // ── STEP 1: Load embeddings ──
    println!("  Step 1: Loading embeddings...");
    let entries = parse_embeddings("d:/DEV/biosfera_390.json");
    let n = entries.len();
    let labels: Vec<usize> = entries.iter().map(|e| tag_to_cluster(&e.tag)).collect();
    let mut vectors: Vec<Vec<f64>> = entries.iter().map(|e| e.vector.clone()).collect();
    let names = &CLUSTER_NAMES[..];

    println!("    Loaded: {} vectors, {}D, {} clusters", n, vectors[0].len(), NUM_CLUSTERS);
    let norms: Vec<f64> = vectors.iter().map(|v| l2_norm(v)).collect();
    let nm = norms.iter().sum::<f64>() / n as f64;
    println!("    Mean L2 norm: {:.6} (should be ~1.0)", nm);
    for c in 0..NUM_CLUSTERS {
        let cnt = labels.iter().filter(|&&l| l == c).count();
        println!("      {:12}: {} nodes", CLUSTER_NAMES[c], cnt);
    }
    println!();

    // ── STEP 2: Original cluster structure ──
    println!("  Step 2: Cluster structure in original {}D", DIM_HIGH);
    let mut intra: Vec<f64> = Vec::new();
    let mut inter: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in (i+1)..n {
            let s = cosine_similarity(&vectors[i], &vectors[j]);
            if labels[i] == labels[j] { intra.push(s); } else { inter.push(s); }
        }
    }
    let (im, is, imin, imax) = stats(&intra);
    let (em, es, emin, emax) = stats(&inter);
    println!("    Intra-cluster: mean={:.4} std={:.4} [{:.4}, {:.4}]", im, is, imin, imax);
    println!("    Inter-cluster: mean={:.4} std={:.4} [{:.4}, {:.4}]", em, es, emin, emax);
    println!("    Separation gap: {:.4}", im - em);

    for c in 0..NUM_CLUSTERS {
        let mut cs: Vec<f64> = Vec::new();
        for i in 0..n { for j in (i+1)..n {
            if labels[i] == c && labels[j] == c {
                cs.push(cosine_similarity(&vectors[i], &vectors[j]));
            }
        }}
        if !cs.is_empty() {
            let (cm, _, cmin, cmax) = stats(&cs);
            println!("      {:12} (intra): mean={:.4} [{:.4}, {:.4}]", names[c], cm, cmin, cmax);
        }
    }
    for a in 0..NUM_CLUSTERS { for b in (a+1)..NUM_CLUSTERS {
        let mut xs: Vec<f64> = Vec::new();
        for i in 0..n { for j in 0..n {
            if labels[i] == a && labels[j] == b {
                xs.push(cosine_similarity(&vectors[i], &vectors[j]));
            }
        }}
        if !xs.is_empty() {
            let (xm, _, _, _) = stats(&xs);
            println!("      {:12} ↔ {:12}: mean={:.4}", names[a], names[b], xm);
        }
    }}

    let orig_purity = cluster_purity_at_k(&vectors, &labels, RECALL_K);
    println!("    Cluster Purity@{} (original): {:.4} ({:.1}%)", RECALL_K, orig_purity, orig_purity * 100.0);
    println!();

    // ── STEP 3: Mean Centering ──
    println!("  Step 3: Mean centering...");
    let mean_vec: Vec<f64> = {
        let mut m = vec![0.0; DIM_HIGH];
        for v in &vectors { for d in 0..DIM_HIGH { m[d] += v[d]; } }
        for d in 0..DIM_HIGH { m[d] /= n as f64; }
        m
    };
    println!("    Mean vector norm: {:.6} (anisotropy magnitude)", l2_norm(&mean_vec));
    for v in vectors.iter_mut() {
        for d in 0..DIM_HIGH { v[d] -= mean_vec[d]; }
        l2_normalize(v);
    }

    let mut intra_c: Vec<f64> = Vec::new();
    let mut inter_c: Vec<f64> = Vec::new();
    for i in 0..n { for j in (i+1)..n {
        let s = cosine_similarity(&vectors[i], &vectors[j]);
        if labels[i] == labels[j] { intra_c.push(s); } else { inter_c.push(s); }
    }}
    let (icm, _, _, _) = stats(&intra_c);
    let (ecm, _, _, _) = stats(&inter_c);
    println!("    Post-center: intra={:.4} inter={:.4} gap={:.4} (was {:.4})",
             icm, ecm, icm - ecm, im - em);
    println!();

    // ── STEP 4: JL Projection ──
    println!("  Step 4: JL projection → {}D (seed={})", DIM_LOW, SEED);
    let mut rng = Rng::new(SEED);
    let proj_matrix = generate_proj_matrix(&mut rng, DIM_HIGH, DIM_LOW);
    let vectors_low: Vec<Vec<f64>> = vectors.iter()
        .map(|v| { let mut p = project(v, &proj_matrix, DIM_LOW); l2_normalize(&mut p); p })
        .collect();

    let mut intra_p: Vec<f64> = Vec::new();
    let mut inter_p: Vec<f64> = Vec::new();
    for i in 0..n { for j in (i+1)..n {
        let s = cosine_similarity(&vectors_low[i], &vectors_low[j]);
        if labels[i] == labels[j] { intra_p.push(s); } else { inter_p.push(s); }
    }}
    let (ipm, _, _, _) = stats(&intra_p);
    let (epm, _, _, _) = stats(&inter_p);
    println!("    Projected: intra={:.4} inter={:.4} gap={:.4}", ipm, epm, ipm - epm);

    let proj_purity = cluster_purity_at_k(&vectors_low, &labels, RECALL_K);
    println!("    Cluster Purity@{} (projected): {:.4} ({:.1}%)", RECALL_K, proj_purity, proj_purity * 100.0);
    println!();

    // ── STEP 5: Spearman & Pearson ──
    println!("  Step 5: Correlation metrics...");
    let centered_sims = pairwise_sims(&vectors);
    let projected_sims = pairwise_sims(&vectors_low);
    let rho = spearman(&centered_sims, &projected_sims);
    let r = pearson(&centered_sims, &projected_sims);
    println!("    Spearman ρ = {:.6}", rho);
    println!("    Pearson  r = {:.6}", r);
    println!();

    // ── STEP 6: Recall@10 ──
    println!("  Step 6: Recall@{}...", RECALL_K);
    let recall = recall_at_k(&vectors, &vectors_low, RECALL_K);
    println!("    Recall@{} = {:.4} ({:.1}%)", RECALL_K, recall, recall * 100.0);
    for c in 0..NUM_CLUSTERS {
        let idx: Vec<usize> = labels.iter().enumerate()
            .filter(|&(_, &l)| l == c).map(|(i, _)| i).collect();
        if idx.is_empty() { continue; }
        let mut cr = 0.0;
        for &i in &idx {
            let mut os: Vec<(usize, f64)> = (0..n).filter(|&j| j != i)
                .map(|j| (j, cosine_similarity(&vectors[i], &vectors[j]))).collect();
            let mut ps: Vec<(usize, f64)> = (0..n).filter(|&j| j != i)
                .map(|j| (j, cosine_similarity(&vectors_low[i], &vectors_low[j]))).collect();
            os.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            ps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let otk: HashSet<usize> = os.iter().take(RECALL_K).map(|&(j, _)| j).collect();
            let ptk: HashSet<usize> = ps.iter().take(RECALL_K).map(|&(j, _)| j).collect();
            cr += otk.intersection(&ptk).count() as f64 / RECALL_K as f64;
        }
        cr /= idx.len() as f64;
        println!("      {:12}: {:.4} ({:.1}%)", names[c], cr, cr * 100.0);
    }
    println!();

    // ── STEP 7: Distortion ──
    println!("  ═══════════════════════════════════════════════════════");
    println!("  DISTORTION ANALYSIS");
    println!("  ═══════════════════════════════════════════════════════");
    let np = centered_sims.len();
    let mut errs: Vec<f64> = (0..np).map(|i| (centered_sims[i] - projected_sims[i]).abs()).collect();
    let em2 = errs.iter().sum::<f64>() / np as f64;
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = errs[(np as f64 * 0.95) as usize];
    let p99 = errs[(np as f64 * 0.99) as usize];
    let mx = errs[np - 1];
    println!("    |Δcos|: mean={:.4} p95={:.4} p99={:.4} max={:.4}", em2, p95, p99, mx);
    println!();

    // ── VERDICT ──
    println!("=================================================================");
    println!("  BRIDGE JUDGE — VERDICT (Biosfera 389)");
    println!("=================================================================");
    println!();
    println!("  Pipeline:  {}D → MeanCenter → JL → {}D → Sculpt", DIM_HIGH, DIM_LOW);
    println!("  Source:    biosfera_390.json ({} vectors, {}D BGE-large, {} clusters)", n, DIM_HIGH, NUM_CLUSTERS);
    println!();
    println!("  Spearman ρ          = {:.6}", rho);
    println!("  Pearson  r          = {:.6}", r);
    println!("  Recall@{}           = {:.4} ({:.1}%)", RECALL_K, recall, recall * 100.0);
    println!("  Cluster Purity@{}   = {:.4} ({:.1}%)", RECALL_K, proj_purity, proj_purity * 100.0);
    println!();
    println!("  Gap (centered):  {:.4}", icm - ecm);
    println!("  Gap (projected): {:.4}", ipm - epm);
    println!();

    let pass_rho = rho >= 0.85;
    let pass_recall = recall >= 0.80;

    println!("  ╔══════════════════════════════════════════════════════════════╗");
    if pass_rho {
        println!("  ║  Spearman ρ ≥ 0.85:  PASS  (got {:.4})                    ║", rho);
    } else {
        println!("  ║  Spearman ρ ≥ 0.85:  FAIL  (got {:.4})                    ║", rho);
    }
    if pass_recall {
        println!("  ║  Recall@{} ≥ 0.80:   PASS  (got {:.4})                    ║", RECALL_K, recall);
    } else {
        println!("  ║  Recall@{} ≥ 0.80:   FAIL  (got {:.4})                    ║", RECALL_K, recall);
    }
    println!("  ╚══════════════════════════════════════════════════════════════╝");
    println!();

    if pass_rho && pass_recall {
        println!("  >>> DECOMPRESSION CHAMBER OPERATIONAL <<<");
        println!("  >>> 128D preserves both rank order and neighborhood structure <<<");
    } else if proj_purity >= 0.85 && (pass_rho || pass_recall) {
        println!("  >>> BRIDGE FUNCTIONAL — cluster structure strongly preserved <<<");
    } else if proj_purity >= 0.80 {
        println!("  >>> BRIDGE PARTIAL — macro-structure intact <<<");
    } else {
        println!("  >>> BRIDGE COMPROMISED <<<");
    }
    println!();

    // ══════════════════════════════════════════════════════════════════
    //  PHASE 2: SCULPTING — Triplet Loss in Poincaré Ball
    // ══════════════════════════════════════════════════════════════════

    println!();
    println!("=================================================================");
    println!("  SCULPTING: Triplet Loss in Poincaré Ball");
    println!("  Epochs: {}, Margin: {}, LR: {}", SCULPT_EPOCHS, TRIPLET_MARGIN, SCULPT_LR);
    println!("  Initial norm: r={}", INITIAL_RADIUS);
    println!("=================================================================");
    println!();

    // ── STEP S1: Inject into Poincaré ball ──
    println!("  Step S1: Injecting 128D vectors into Poincaré ball (r={})...", INITIAL_RADIUS);
    let mut ball: Vec<Vec<f64>> = vectors_low.iter().map(|v| {
        let mut p = v.clone();
        // Scale to initial radius inside the ball
        let n = l2_norm(&p);
        if n > 1e-15 {
            for x in p.iter_mut() { *x *= INITIAL_RADIUS / n; }
        }
        p
    }).collect();

    print_ball_stats(&ball, &labels, names, "Initial injection");

    // ── STEP S2: Generate triplets ──
    println!();
    println!("  Step S2: Generating triplets...");
    let triplets = generate_triplets(&labels, n, &mut rng);
    println!("    Generated {} triplets", triplets.len());

    // ── STEP S3: Sculpt ──
    println!();
    println!("  Step S3: Sculpting ({} epochs)...", SCULPT_EPOCHS);
    println!("    {:>5}  {:>10}  {:>10}  {:>10}  {:>10}", "Epoch", "Loss", "IntraDist", "InterDist", "MaxNorm");

    for epoch in 0..SCULPT_EPOCHS {
        let mut total_loss = 0.0;
        let mut n_active = 0usize;

        for &(a, p, ng) in &triplets {
            let d_ap = poincare_distance(&ball[a], &ball[p]);
            let d_an = poincare_distance(&ball[a], &ball[ng]);
            let loss = (d_ap - d_an + TRIPLET_MARGIN).max(0.0);

            if loss > 0.0 {
                total_loss += loss;
                n_active += 1;

                // Riemannian gradient: ∂d/∂x scaled by conformal factor (1-||x||²)²/4
                let grad_ap_a = poincare_grad(&ball[a], &ball[p], true);
                let grad_ap_p = poincare_grad(&ball[a], &ball[p], false);
                let grad_an_a = poincare_grad(&ball[a], &ball[ng], true);
                let grad_an_n = poincare_grad(&ball[a], &ball[ng], false);

                // Update: minimize d(a,p), maximize d(a,n)
                // Loss = d(a,p) - d(a,n) + margin
                // ∂L/∂a = ∂d_ap/∂a - ∂d_an/∂a
                // ∂L/∂p = ∂d_ap/∂p
                // ∂L/∂n = -∂d_an/∂n
                for d in 0..DIM_LOW {
                    ball[a][d] -= SCULPT_LR * (grad_ap_a[d] - grad_an_a[d]);
                    ball[p][d] -= SCULPT_LR * grad_ap_p[d];
                    ball[ng][d] += SCULPT_LR * grad_an_n[d];
                }

                // Project back into ball (||x|| < 1 - eps)
                project_into_poincare(&mut ball[a]);
                project_into_poincare(&mut ball[p]);
                project_into_poincare(&mut ball[ng]);
            }
        }

        if epoch % 10 == 0 || epoch == SCULPT_EPOCHS - 1 {
            let (intra_d, inter_d) = mean_distances(&ball, &labels);
            let max_norm = ball.iter().map(|v| l2_norm(v)).fold(0.0_f64, f64::max);
            let avg_loss = if n_active > 0 { total_loss / n_active as f64 } else { 0.0 };
            println!("    {:>5}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.6}",
                     epoch, avg_loss, intra_d, inter_d, max_norm);
        }
    }

    println!();
    print_ball_stats(&ball, &labels, names, "After sculpting");

    // ── STEP S4: Final analysis ──
    println!();
    println!("  ═══════════════════════════════════════════════════════════════");
    println!("  SCULPTING RESULTS — Poincaré Disk Structure");
    println!("  ═══════════════════════════════════════════════════════════════");
    println!();

    // Cluster centroids (Euclidean mean in ball, then check)
    let mut centroids = vec![vec![0.0; DIM_LOW]; NUM_CLUSTERS];
    let mut counts = vec![0usize; NUM_CLUSTERS];
    for i in 0..n {
        let c = labels[i];
        for d in 0..DIM_LOW { centroids[c][d] += ball[i][d]; }
        counts[c] += 1;
    }
    for c in 0..NUM_CLUSTERS {
        if counts[c] > 0 {
            for d in 0..DIM_LOW { centroids[c][d] /= counts[c] as f64; }
        }
    }

    println!("  Cluster centroids (Euclidean in ball):");
    for c in 0..NUM_CLUSTERS {
        let r = l2_norm(&centroids[c]);
        println!("    {:12} ||c|| = {:.6}  (n={}, depth proxy)", names[c], r, counts[c]);
    }
    println!();

    println!("  Poincaré distances between ALL centroids:");
    // Full distance matrix
    println!("    {:>12}  {}", "", names.iter().map(|n| format!("{:>10}", n)).collect::<Vec<_>>().join("  "));
    for a in 0..NUM_CLUSTERS {
        let mut row = format!("    {:>12}", names[a]);
        for b in 0..NUM_CLUSTERS {
            if a == b {
                row.push_str(&format!("  {:>10}", "—"));
            } else {
                let d = poincare_distance(&centroids[a], &centroids[b]);
                row.push_str(&format!("  {:>10.4}", d));
            }
        }
        println!("{}", row);
    }
    println!();

    // Original BRIDGE HYPOTHESIS: Cibernética equidistant from Sistemas and Filosofia
    let d_sc = poincare_distance(&centroids[0], &centroids[2]); // Sistemas ↔ Cibernética
    let d_fc = poincare_distance(&centroids[1], &centroids[2]); // Filosofia ↔ Cibernética
    let d_sf = poincare_distance(&centroids[0], &centroids[1]); // Sistemas ↔ Filosofia
    let bridge_ratio = d_sc.max(d_fc) / d_sc.min(d_fc);
    let bridge_gap = (d_sc - d_fc).abs();

    println!("  BRIDGE HYPOTHESIS (original 3 clusters):");
    println!("    Sistemas ↔ Filosofia:    {:.4} (should be LARGEST)", d_sf);
    println!("    Sistemas ↔ Cibernética:  {:.4}", d_sc);
    println!("    Filosofia ↔ Cibernética: {:.4}", d_fc);
    println!("    Bridge ratio (max/min):  {:.4} (1.0 = perfect bridge)", bridge_ratio);
    println!("    Bridge gap |d_SC-d_FC|:  {:.4}", bridge_gap);
    println!();

    let bridge_ok = d_sf > d_sc && d_sf > d_fc && bridge_ratio < 2.0;
    if bridge_ok {
        println!("  ╔══════════════════════════════════════════════════════════════╗");
        println!("  ║  BRIDGE: CONSTRUCTED                                        ║");
        println!("  ║  Cibernética sits between Sistemas and Filosofia            ║");
        println!("  ╚══════════════════════════════════════════════════════════════╝");
    } else {
        println!("  ╔══════════════════════════════════════════════════════════════╗");
        println!("  ║  BRIDGE: BROKEN — universe split in two                     ║");
        println!("  ╚══════════════════════════════════════════════════════════════╝");
    }
    println!();

    // SATELLITE ANALYSIS: new clusters vs their parent clusters
    println!("  SATELLITE PROXIMITY ANALYSIS:");
    // UnsafeRust(3) → Sistemas(0), Continental(4) → Filosofia(1), InfoTheory(6) → Cibernetica(2)
    let satellites = [(3, 0, "UnsafeRust", "Sistemas"),
                      (4, 1, "Continental", "Filosofia"),
                      (6, 2, "InfoTheory", "Cibernetica")];
    for &(sat, parent, sat_name, par_name) in &satellites {
        let d_to_parent = poincare_distance(&centroids[sat], &centroids[parent]);
        // Find nearest OTHER centroid (not parent)
        let mut nearest_other = f64::MAX;
        let mut nearest_name = "";
        for c in 0..NUM_CLUSTERS {
            if c == sat || c == parent { continue; }
            let d = poincare_distance(&centroids[sat], &centroids[c]);
            if d < nearest_other { nearest_other = d; nearest_name = names[c]; }
        }
        println!("    {:12} → {:12}: d_P={:.4}  (nearest other: {} @ {:.4})",
                 sat_name, par_name, d_to_parent, nearest_name, nearest_other);
    }
    println!();

    // DIARY DENSITY ANALYSIS: where did diary nodes land?
    println!("  DIARY DENSITY ANALYSIS:");
    let diary_idx: Vec<usize> = labels.iter().enumerate()
        .filter(|&(_, &l)| l == 5).map(|(i, _)| i).collect();
    if !diary_idx.is_empty() {
        // Count nearest centroid for each Diary node
        let mut diary_nearest = vec![0usize; NUM_CLUSTERS];
        for &i in &diary_idx {
            let mut min_d = f64::MAX;
            let mut min_c = 0;
            for c in 0..NUM_CLUSTERS {
                if c == 5 { continue; } // skip Diary itself
                let d = poincare_distance(&ball[i], &centroids[c]);
                if d < min_d { min_d = d; min_c = c; }
            }
            diary_nearest[min_c] += 1;
        }
        println!("    Diary nodes nearest to (excluding Diary centroid):");
        for c in 0..NUM_CLUSTERS {
            if c == 5 { continue; }
            if diary_nearest[c] > 0 {
                println!("      {:12}: {}/{}", names[c], diary_nearest[c], diary_idx.len());
            }
        }
        let diary_r: Vec<f64> = diary_idx.iter().map(|&i| l2_norm(&ball[i])).collect();
        let mean_diary_r = diary_r.iter().sum::<f64>() / diary_r.len() as f64;
        let max_diary_r = diary_r.iter().cloned().fold(0.0_f64, f64::max);
        println!("    Diary radial: mean_r={:.4}, max_r={:.4} (high=peripheral)", mean_diary_r, max_diary_r);
    }
    println!();

    // Per-cluster purity analysis (Poincaré neighbors)
    println!("  PER-CLUSTER PURITY@{} (Poincaré):", RECALL_K);
    for c in 0..NUM_CLUSTERS {
        let idx: Vec<usize> = labels.iter().enumerate()
            .filter(|&(_, &l)| l == c).map(|(i, _)| i).collect();
        if idx.is_empty() { continue; }
        let mut purity_sum = 0.0;
        for &i in &idx {
            let mut dists: Vec<(usize, f64)> = (0..n).filter(|&j| j != i)
                .map(|j| (j, poincare_distance(&ball[i], &ball[j]))).collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let same = dists.iter().take(RECALL_K).filter(|&&(j, _)| labels[j] == c).count();
            purity_sum += same as f64 / RECALL_K as f64;
        }
        let cpurity = purity_sum / idx.len() as f64;
        println!("    {:12}: {:.4} ({:.1}%)", names[c], cpurity, cpurity * 100.0);
    }
    println!();

    // Cluster purity in Poincaré (using hyperbolic distances)
    let purity_ball = poincare_purity_at_k(&ball, &labels, RECALL_K);
    println!("  Cluster Purity@{} (Poincaré): {:.4} ({:.1}%)", RECALL_K, purity_ball, purity_ball * 100.0);
    println!();

    // ══════════════════════════════════════════════════════════════════
    //  PHASE 3: PROBE TEST — Manifold-Informed Synthesis
    // ══════════════════════════════════════════════════════════════════

    println!("=================================================================");
    println!("  PROBE TEST: Manifold-Informed Synthesis");
    println!("  Query: \"Como a segurança de memória (Rust) pode ser vista");
    println!("          como uma forma de vontade de poder sobre o caos");
    println!("          do hardware?\"");
    println!("=================================================================");
    println!();

    // Load REAL probe embedding from BGE-large (generated by sentence-transformers)
    let probe_raw = fs::read_to_string("d:/DEV/probe_query.json")
        .expect("Failed to read probe_query.json");
    let mut probe_high = Vec::with_capacity(DIM_HIGH);
    {
        // Parse "vector": [...]
        let vstart = probe_raw.find("\"vector\"").unwrap();
        let arr_start = probe_raw[vstart..].find('[').unwrap() + vstart + 1;
        let arr_end = probe_raw[arr_start..].find(']').unwrap() + arr_start;
        for num_str in probe_raw[arr_start..arr_end].split(',') {
            let v: f64 = num_str.trim().parse().unwrap();
            probe_high.push(v);
        }
    }
    println!("  Probe loaded from BGE-large ({}D, real semantic embedding)", probe_high.len());

    // Apply same pipeline: Mean Center → JL → L2 Norm → Poincaré injection
    for d in 0..DIM_HIGH { probe_high[d] -= mean_vec[d]; }
    l2_normalize(&mut probe_high);

    let mut probe_128 = project(&probe_high, &proj_matrix, DIM_LOW);
    l2_normalize(&mut probe_128);

    // Inject into ball at initial radius
    let probe_norm = l2_norm(&probe_128);
    if probe_norm > 1e-15 {
        for x in probe_128.iter_mut() { *x *= INITIAL_RADIUS / probe_norm; }
    }

    println!("    ||probe|| in ball = {:.6}", l2_norm(&probe_128));
    println!();

    // Compute distances from probe to all 90 nodes in the sculpted ball
    let mut probe_dists: Vec<(usize, f64)> = (0..n)
        .map(|i| (i, poincare_distance(&probe_128, &ball[i])))
        .collect();
    probe_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Distance to ALL cluster centroids
    println!("  Probe → Cluster centroids (Poincaré):");
    for c in 0..NUM_CLUSTERS {
        let pd = poincare_distance(&probe_128, &centroids[c]);
        println!("    → {:12}: {:.4}", names[c], pd);
    }
    println!();

    // Top-5 nearest neighbors
    println!("  ╔══════════════════════════════════════════════════════════════╗");
    println!("  ║  TOP-5 NEAREST NEIGHBORS (Poincaré distance)               ║");
    println!("  ╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  {:>4}  {:>8}  {:>12}  {}", "Rank", "d_P", "Cluster", "Text");
    println!("  {}  {}  {}  {}", "─".repeat(4), "─".repeat(8), "─".repeat(12), "─".repeat(50));
    for rank in 0..5 {
        let (idx, dist) = probe_dists[rank];
        let tag = &entries[idx].tag;
        let text = &entries[idx].text;
        println!("  {:>4}  {:>8.4}  {:>12}  {}", rank + 1, dist, tag, text);
    }
    println!();

    // Top-10 distribution by cluster
    println!("  Top-10 distribution:");
    for c in 0..NUM_CLUSTERS {
        let cnt = probe_dists.iter().take(10).filter(|&&(i, _)| labels[i] == c).count();
        if cnt > 0 {
            println!("    {:12}: {}", names[c], cnt);
        }
    }
    println!();

    // The 5 anchors formatted for EVA context injection
    println!("  ═══════════════════════════════════════════════════════════════");
    println!("  GEOMETRIC ANCHORS FOR EVA CONTEXT INJECTION");
    println!("  ═══════════════════════════════════════════════════════════════");
    println!();
    for rank in 0..5 {
        let (idx, dist) = probe_dists[rank];
        let tag = &entries[idx].tag;
        let text = &entries[idx].text;
        println!("  Anchor #{} [d_P={:.4}, cluster={}]:", rank + 1, dist, tag);
        println!("    \"{}\"", text);
        println!();
    }

    // ══════════════════════════════════════════════════════════════════
    //  PHASE 5: GEODESIC NAVIGATION — KNN-15 + Dijkstra + GCS
    // ══════════════════════════════════════════════════════════════════

    println!("=================================================================");
    println!("  GEODESIC NAVIGATION: KNN-{} Graph + Dijkstra", KNN_K);
    println!("  Weight(u,v) = d_P(u,v) * (1 + {} * |‖u‖ - ‖v‖|)", RADIAL_LAMBDA);
    println!("=================================================================");
    println!();

    println!("  Building KNN-{} graph ({} nodes)...", KNN_K, n);
    let adj = build_knn_graph(&ball, KNN_K, RADIAL_LAMBDA);
    let total_edges: usize = adj.iter().map(|a| a.len()).sum();
    println!("    Total edges (symmetric): {}", total_edges);
    let avg_degree = total_edges as f64 / n as f64;
    println!("    Average degree: {:.1}", avg_degree);
    println!();

    // Run geodesic for both directions
    let probes: [(usize, usize, &str); 2] = [
        (2, 30, "FORWARD: Sistemas → Filosofia"),
        (30, 2, "REVERSE: Filosofia → Sistemas"),
    ];

    // Store results for asymmetry comparison
    let mut fwd_path_ids: Vec<usize> = Vec::new();
    let mut rev_path_ids: Vec<usize> = Vec::new();
    let mut fwd_gcs = 0.0_f64;
    let mut rev_gcs = 0.0_f64;
    let mut fwd_stretch = 0.0_f64;
    let mut rev_stretch = 0.0_f64;

    for (probe_idx, &(oid, did, label)) in probes.iter().enumerate() {
        let oi = find_by_id(&entries, oid);
        let di = find_by_id(&entries, did);

        if let (Some(src), Some(dst)) = (oi, di) {
            println!("  ═══════════════════════════════════════════════════════════════");
            println!("  GEODESIC {}", label);
            println!("  ═══════════════════════════════════════════════════════════════");
            println!("    Origin: ID {} [{}] \"{}\"", oid, entries[src].tag, entries[src].text);
            println!("    Dest:   ID {} [{}] \"{}\"", did, entries[dst].tag, entries[dst].text);
            println!();

            match dijkstra(&adj, src, dst) {
                Some((path, total_cost)) => {
                    let direct_dp = poincare_distance(&ball[src], &ball[dst]);
                    let stretch = total_cost / direct_dp;

                    println!("    Direct d_P       = {:.4}", direct_dp);
                    println!("    Geodesic cost    = {:.4}", total_cost);
                    println!("    Stretch ratio    = {:.4}", stretch);
                    println!("    Hops             = {}", path.len() - 1);
                    println!();

                    // Path table with BGE cosine between consecutive hops
                    println!("  {:>4}  {:>4}  {:>12}  {:>6}  {:>8}  {:>8}  {}",
                             "Step", "ID", "Cluster", "‖x‖", "d_P→nxt", "cos_BGE", "Text");
                    println!("  {}  {}  {}  {}  {}  {}  {}",
                             "─".repeat(4), "─".repeat(4), "─".repeat(12),
                             "─".repeat(6), "─".repeat(8), "─".repeat(8), "─".repeat(48));

                    let mut hop_cosines: Vec<f64> = Vec::new();
                    let mut hop_radial_deltas: Vec<f64> = Vec::new();

                    for (step, &idx) in path.iter().enumerate() {
                        let eid = entries[idx].id;
                        let tag = &entries[idx].tag;
                        let r = l2_norm(&ball[idx]);
                        let text = &entries[idx].text;
                        let short = if text.len() > 48 { &text[..48] } else { text };

                        let (dp_str, cos_str) = if step < path.len() - 1 {
                            let next_idx = path[step + 1];
                            let dp = poincare_distance(&ball[idx], &ball[next_idx]);
                            // Cosine in ORIGINAL BGE-large space (pre-centering)
                            let cos_bge = cosine_similarity(&entries[idx].vector, &entries[next_idx].vector);
                            hop_cosines.push(cos_bge);
                            let r_next = l2_norm(&ball[next_idx]);
                            hop_radial_deltas.push((r - r_next).abs());
                            (format!("{:.4}", dp), format!("{:.4}", cos_bge))
                        } else {
                            ("—".to_string(), "—".to_string())
                        };

                        println!("  {:>4}  {:>4}  {:>12}  {:>6.3}  {:>8}  {:>8}  {}",
                                 step, eid, tag, r, dp_str, cos_str, short);
                    }
                    println!();

                    // GCS: Geodesic Coherence Score
                    // C(u,v) = (w1*S + w2*R) * T
                    // S = cosine(BGE_u, BGE_v)
                    // R = 1 - |‖u‖ - ‖v‖|
                    // T = 1.0 if same cluster, 0.85 if transition
                    let w1 = 0.7;
                    let w2 = 0.3;
                    let mut coherences: Vec<f64> = Vec::new();
                    for hop in 0..path.len()-1 {
                        let u_idx = path[hop];
                        let v_idx = path[hop + 1];
                        let s = hop_cosines[hop];
                        let r = 1.0 - hop_radial_deltas[hop];
                        let t = if labels[u_idx] == labels[v_idx] { 1.0 } else { 0.85 };
                        let c = (w1 * s + w2 * r) * t;
                        coherences.push(c);
                    }
                    // Harmonic mean of coherences
                    let gcs = if coherences.is_empty() { 0.0 } else {
                        let n_h = coherences.len() as f64;
                        let inv_sum: f64 = coherences.iter().map(|&c| 1.0 / c.max(1e-10)).sum();
                        n_h / inv_sum
                    };

                    let mean_cos = if hop_cosines.is_empty() { 0.0 }
                        else { hop_cosines.iter().sum::<f64>() / hop_cosines.len() as f64 };
                    let min_cos = hop_cosines.iter().cloned().fold(f64::MAX, f64::min);
                    let mean_radial = if hop_radial_deltas.is_empty() { 0.0 }
                        else { hop_radial_deltas.iter().sum::<f64>() / hop_radial_deltas.len() as f64 };

                    println!("  GCS (Geodesic Coherence Score):");
                    println!("    Per-hop coherences: {:?}", coherences.iter().map(|c| format!("{:.4}", c)).collect::<Vec<_>>());
                    println!("    GCS (harmonic mean) = {:.4}", gcs);
                    println!();
                    println!("  BGE Cosine profile:");
                    println!("    Mean cos(BGE) = {:.4}", mean_cos);
                    println!("    Min  cos(BGE) = {:.4}", min_cos);
                    let fluent = mean_cos > 0.5;
                    if fluent {
                        println!("    FLUENT: geodesic is linguistically smooth");
                    } else {
                        println!("    ROUGH: semantic jumps detected");
                    }
                    println!();
                    println!("  Radial stability:");
                    println!("    Mean |Δr| = {:.4}", mean_radial);
                    let path_norms: Vec<f64> = path.iter().map(|&i| l2_norm(&ball[i])).collect();
                    let min_r = path_norms.iter().cloned().fold(f64::MAX, f64::min);
                    let max_r = path_norms.iter().cloned().fold(0.0_f64, f64::max);
                    let mean_r = path_norms.iter().sum::<f64>() / path_norms.len() as f64;
                    println!("    Radial range: [{:.4}, {:.4}], mean={:.4}", min_r, max_r, mean_r);
                    if min_r < mean_r * 0.7 {
                        println!("    WARNING: hub teleport risk");
                    } else {
                        println!("    OK: stays in manifold tissue");
                    }
                    println!();

                    // Cluster transitions
                    let mut transitions: Vec<String> = Vec::new();
                    let mut prev_lbl = labels[path[0]];
                    transitions.push(names[prev_lbl].to_string());
                    for &idx in &path[1..] {
                        let lbl = labels[idx];
                        if lbl != prev_lbl {
                            transitions.push(names[lbl].to_string());
                            prev_lbl = lbl;
                        }
                    }
                    println!("  Cluster path: {}", transitions.join(" → "));
                    println!();

                    // Store for comparison
                    if probe_idx == 0 {
                        fwd_path_ids = path.iter().map(|&i| entries[i].id).collect();
                        fwd_gcs = gcs;
                        fwd_stretch = stretch;
                    } else {
                        rev_path_ids = path.iter().map(|&i| entries[i].id).collect();
                        rev_gcs = gcs;
                        rev_stretch = stretch;
                    }
                }
                None => {
                    println!("  NO PATH FOUND (graph disconnected)");
                    println!();
                }
            }
        }
    }

    // ── ASYMMETRY ANALYSIS ──
    println!("  ═══════════════════════════════════════════════════════════════");
    println!("  ASYMMETRY ANALYSIS");
    println!("  ═══════════════════════════════════════════════════════════════");
    println!();
    println!("    Forward IDs:  {:?}", fwd_path_ids);
    println!("    Reverse IDs:  {:?}", rev_path_ids);
    println!();

    // Check if reverse is just forward reversed
    let rev_reversed: Vec<usize> = rev_path_ids.iter().rev().cloned().collect();
    let is_mirror = fwd_path_ids == rev_reversed;
    let shared: usize = fwd_path_ids.iter().filter(|id| rev_path_ids.contains(id)).count();
    let jaccard = if fwd_path_ids.is_empty() && rev_path_ids.is_empty() { 0.0 }
        else { shared as f64 / (fwd_path_ids.len() + rev_path_ids.len() - shared) as f64 };

    println!("    Mirror path?     {}", if is_mirror { "YES (identical reversed)" } else { "NO (asymmetric)" });
    println!("    Shared nodes:    {}/{} (Jaccard={:.4})", shared,
             fwd_path_ids.len().max(rev_path_ids.len()), jaccard);
    println!("    Forward GCS:     {:.4}", fwd_gcs);
    println!("    Reverse GCS:     {:.4}", rev_gcs);
    println!("    GCS asymmetry:   {:.4} (|fwd-rev|/max)", (fwd_gcs - rev_gcs).abs() / fwd_gcs.max(rev_gcs).max(1e-10));
    println!("    Forward stretch: {:.4}", fwd_stretch);
    println!("    Reverse stretch: {:.4}", rev_stretch);
    println!();

    if is_mirror {
        println!("  ╔══════════════════════════════════════════════════════════════╗");
        println!("  ║  SYMMETRIC: Same road in both directions                    ║");
        println!("  ╚══════════════════════════════════════════════════════════════╝");
    } else if jaccard > 0.5 {
        println!("  ╔══════════════════════════════════════════════════════════════╗");
        println!("  ║  QUASI-SYMMETRIC: Mostly shared nodes, minor detours        ║");
        println!("  ╚══════════════════════════════════════════════════════════════╝");
    } else {
        println!("  ╔══════════════════════════════════════════════════════════════╗");
        println!("  ║  ASYMMETRIC: Different routes discovered                    ║");
        println!("  ║  BGE model encodes directional semantic bias                ║");
        println!("  ╚══════════════════════════════════════════════════════════════╝");
    }
    println!();

    // ══════════════════════════════════════════════════════════════════
    //  CSV
    // ══════════════════════════════════════════════════════════════════
    let _ = std::fs::create_dir_all("experiments");
    let p = "experiments/telemetry_bridge_judge.csv";
    let mut f = OpenOptions::new().write(true).create(true).truncate(true).open(p).unwrap();
    writeln!(f, "metric,value").unwrap();
    writeln!(f, "source,biosfera_390_bge_large").unwrap();
    writeln!(f, "num_clusters,{}", NUM_CLUSTERS).unwrap();
    writeln!(f, "dim_high,{}", DIM_HIGH).unwrap();
    writeln!(f, "dim_low,{}", DIM_LOW).unwrap();
    writeln!(f, "n_vectors,{}", n).unwrap();
    writeln!(f, "spearman_rho,{:.8}", rho).unwrap();
    writeln!(f, "pearson_r,{:.8}", r).unwrap();
    writeln!(f, "recall_at_{},{:.8}", RECALL_K, recall).unwrap();
    writeln!(f, "cluster_purity_orig,{:.8}", orig_purity).unwrap();
    writeln!(f, "cluster_purity_proj,{:.8}", proj_purity).unwrap();
    writeln!(f, "intra_orig,{:.8}", im).unwrap();
    writeln!(f, "inter_orig,{:.8}", em).unwrap();
    writeln!(f, "gap_orig,{:.8}", im - em).unwrap();
    writeln!(f, "intra_centered,{:.8}", icm).unwrap();
    writeln!(f, "inter_centered,{:.8}", ecm).unwrap();
    writeln!(f, "gap_centered,{:.8}", icm - ecm).unwrap();
    writeln!(f, "intra_projected,{:.8}", ipm).unwrap();
    writeln!(f, "inter_projected,{:.8}", epm).unwrap();
    writeln!(f, "gap_projected,{:.8}", ipm - epm).unwrap();
    writeln!(f, "error_mean,{:.8}", em2).unwrap();
    writeln!(f, "error_p95,{:.8}", p95).unwrap();
    writeln!(f, "error_max,{:.8}", mx).unwrap();
    writeln!(f, "sculpt_epochs,{}", SCULPT_EPOCHS).unwrap();
    writeln!(f, "triplet_margin,{}", TRIPLET_MARGIN).unwrap();
    writeln!(f, "poincare_purity,{:.8}", purity_ball).unwrap();
    writeln!(f, "bridge_d_sf,{:.8}", d_sf).unwrap();
    writeln!(f, "bridge_d_sc,{:.8}", d_sc).unwrap();
    writeln!(f, "bridge_d_fc,{:.8}", d_fc).unwrap();
    writeln!(f, "bridge_ratio,{:.8}", bridge_ratio).unwrap();
    writeln!(f, "bridge_ok,{}", bridge_ok).unwrap();
    println!("  Saved: {}", p);
}
