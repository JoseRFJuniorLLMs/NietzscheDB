//! Benchmark: Real Graph Topology with Structural Metrics.
//!
//! This is NOT an abstract simulation. It builds a REAL graph with:
//! - 10,000 nodes (1,000 signal + 9,000 noise)
//! - ~50,000 undirected edges
//! - Full adjacency list (HashMap<usize, HashSet<usize>>)
//! - BFS-sampled global efficiency
//! - Shannon entropy of degree distribution
//! - TGC with δH × δE formula
//!
//! 3 experiments, SAME isolated variable (birth mode):
//!   D: FOAM        — void-born orphans, degree=0
//!   E: ANCHORED    — elite-parented, degree=2 (structural edges)
//!   F: DIALECTICAL — elite-parented, degree=2 + entropy polarization
//!
//! ## Usage
//! ```sh
//! cargo run --release --bin simulate_forgetting
//! ```
//!
//! ## Output
//! - `telemetry_D_foam.csv`
//! - `telemetry_E_anchored.csv`
//! - `telemetry_F_dialectical.csv`
//!
//! ## Benchmark target
//! 10k nodes, ~50k edges, 100 cycles, measure time per cycle.

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

// ──────────────────────────────────────────────────
//  Core Math
// ──────────────────────────────────────────────────

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

struct VitalityWeights {
    w1_energy: f32,
    w2_hausdorff: f32,
    w3_entropy: f32,
    w4_elite_prox: f32,
    w5_causal: f32,
    w6_toxicity: f32,
}

impl Default for VitalityWeights {
    fn default() -> Self {
        Self {
            w1_energy: 1.0,
            w2_hausdorff: 0.8,
            w3_entropy: 1.2,
            w4_elite_prox: 1.5,
            w5_causal: 2.0,
            w6_toxicity: 1.0,
        }
    }
}

// ──────────────────────────────────────────────────
//  Data Structures
// ──────────────────────────────────────────────────

#[derive(Clone)]
struct SimNode {
    energy: f32,
    hausdorff: f32,
    entropy_delta: f32,
    elite_proximity: f32,
    toxicity: f32,
    is_signal: bool,
    px: f32,
    py: f32,
    born_cycle: usize,
}

#[derive(Clone)]
struct VoidSeed {
    px: f32,
    py: f32,
}

struct CycleTelemetry {
    cycle: usize,
    total_nodes: usize,
    total_edges: usize,
    sacrificed: usize,
    nodes_created: usize,
    signal_killed_total: usize,
    noise_killed_total: usize,
    mean_vitality: f32,
    variance_vitality: f32,
    mean_energy: f32,
    structural_entropy: f32,
    global_efficiency: f32,
    tgc_raw: f32,
    tgc_ema: f32,
    elite_drift: f32,
    newborn_survival: f32,
    ms_per_cycle: f32,
}

#[derive(Clone, Copy)]
struct EliteCentroid {
    cx: f32,
    cy: f32,
}

impl EliteCentroid {
    fn zero() -> Self {
        Self { cx: 0.0, cy: 0.0 }
    }
    fn distance(&self, o: &EliteCentroid) -> f32 {
        ((self.cx - o.cx).powi(2) + (self.cy - o.cy).powi(2)).sqrt()
    }
}

/// Birth mode — the ONLY experimental variable.
#[derive(Clone, Copy, PartialEq)]
enum BirthMode {
    Foam,
    Anchored,
    Dialectical,
}

impl BirthMode {
    fn label(&self) -> &'static str {
        match self {
            BirthMode::Foam => "D_foam",
            BirthMode::Anchored => "E_anchored",
            BirthMode::Dialectical => "F_dialectical",
        }
    }
}

// ──────────────────────────────────────────────────
//  TGC Monitor (production formula)
// ──────────────────────────────────────────────────

const ALPHA: f32 = 2.0;
const BETA: f32 = 3.0;

struct TgcMonitor {
    prev_hs: f32,
    prev_eg: f32,
    ema_tgc: f32,
}

impl TgcMonitor {
    fn new() -> Self {
        Self {
            prev_hs: 0.0,
            prev_eg: 0.0,
            ema_tgc: 0.0,
        }
    }

    fn compute(
        &mut self,
        nodes_created: usize,
        active_nodes: usize,
        mean_quality: f32,
        current_hs: f32,
        current_eg: f32,
    ) -> (f32, f32) {
        let intensity = if active_nodes > 0 {
            nodes_created as f32 / active_nodes as f32
        } else {
            0.0
        };

        if intensity == 0.0 {
            self.prev_hs = current_hs;
            self.prev_eg = current_eg;
            self.ema_tgc *= 0.8;
            return (0.0, self.ema_tgc);
        }

        let delta_h = current_hs - self.prev_hs;
        let delta_e = current_eg - self.prev_eg;

        let mut tgc = intensity
            * mean_quality.clamp(0.0, 1.0)
            * (1.0 + ALPHA * delta_h)
            * (1.0 + BETA * delta_e);
        tgc = tgc.max(0.0);

        self.prev_hs = current_hs;
        self.prev_eg = current_eg;
        self.ema_tgc = 0.2 * tgc + 0.8 * self.ema_tgc;

        (tgc, self.ema_tgc)
    }
}

// ──────────────────────────────────────────────────
//  Structural Metrics (inline for binary)
// ──────────────────────────────────────────────────

fn structural_entropy(adj: &HashMap<usize, HashSet<usize>>, total: usize) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let mut deg_counts: HashMap<usize, usize> = HashMap::new();
    for neighbors in adj.values() {
        *deg_counts.entry(neighbors.len()).or_insert(0) += 1;
    }
    let n = total as f32;
    let mut h = 0.0f32;
    for &count in deg_counts.values() {
        if count > 0 {
            let p = count as f32 / n;
            h -= p * p.ln();
        }
    }
    h
}

fn global_efficiency_sampled(
    adj: &HashMap<usize, HashSet<usize>>,
    node_ids: &[usize],
    sample: usize,
    seed: u64,
) -> f32 {
    let n = node_ids.len();
    if n < 2 {
        return 0.0;
    }
    let sample = sample.min(n);
    let mut total = 0.0f64;
    let mut pairs = 0u64;
    let mut state = seed;

    for _ in 0..sample {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let src = node_ids[((state >> 33) as usize) % n];

        // BFS from src
        let mut visited: HashMap<usize, usize> = HashMap::new();
        let mut queue = std::collections::VecDeque::new();
        visited.insert(src, 0);
        queue.push_back(src);
        while let Some(cur) = queue.pop_front() {
            let d = visited[&cur];
            if let Some(neighbors) = adj.get(&cur) {
                for &nb in neighbors {
                    if !visited.contains_key(&nb) {
                        visited.insert(nb, d + 1);
                        queue.push_back(nb);
                    }
                }
            }
        }
        for (&t, &d) in &visited {
            if t != src && d > 0 {
                total += 1.0 / d as f64;
                pairs += 1;
            }
        }
    }

    if pairs == 0 {
        0.0
    } else {
        (total / pairs as f64) as f32
    }
}

// ──────────────────────────────────────────────────
//  Deterministic PRNG
// ──────────────────────────────────────────────────

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let t = ((self.state >> 33) as f32) / (u32::MAX as f32);
        lo + t * (hi - lo)
    }
    fn next_usize(&mut self, hi: usize) -> usize {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 33) as usize) % hi.max(1)
    }
    fn chance(&mut self, p: f32) -> bool {
        self.next_f32(0.0, 1.0) < p
    }
}

// ──────────────────────────────────────────────────
//  The Simulator (with real graph)
// ──────────────────────────────────────────────────

struct Sim {
    w: VitalityWeights,
    nodes: HashMap<usize, SimNode>,
    adj: HashMap<usize, HashSet<usize>>,
    next_id: usize,
    voids: Vec<VoidSeed>,
    noise_killed: usize,
    signal_killed: usize,
    vit_threshold: f32,
    eng_threshold: f32,
    tgc: TgcMonitor,
    centroid_0: EliteCentroid,
    mode: BirthMode,
    last_born_count: usize,
}

impl Sim {
    fn new(mode: BirthMode) -> Self {
        Self {
            w: VitalityWeights::default(),
            nodes: HashMap::new(),
            adj: HashMap::new(),
            next_id: 0,
            voids: Vec::new(),
            noise_killed: 0,
            signal_killed: 0,
            vit_threshold: 0.30,
            eng_threshold: 0.10,
            tgc: TgcMonitor::new(),
            centroid_0: EliteCentroid::zero(),
            mode,
            last_born_count: 0,
        }
    }

    fn alloc_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn add_edge(&mut self, a: usize, b: usize) {
        self.adj.entry(a).or_default().insert(b);
        self.adj.entry(b).or_default().insert(a);
    }

    fn remove_node(&mut self, id: usize) {
        // Remove from adjacency
        if let Some(neighbors) = self.adj.remove(&id) {
            for nb in neighbors {
                if let Some(nbs) = self.adj.get_mut(&nb) {
                    nbs.remove(&id);
                }
            }
        }
        self.nodes.remove(&id);
    }

    fn degree(&self, id: usize) -> usize {
        self.adj.get(&id).map_or(0, |s| s.len())
    }

    /// Seed the universe: 1000 signal + 9000 noise, ~50k edges.
    fn seed(&mut self) {
        let mut r = Rng::new(42);

        // ── Signal nodes: tightly connected core ──
        let mut signal_ids = Vec::new();
        for _ in 0..1000 {
            let id = self.alloc_id();
            self.nodes.insert(
                id,
                SimNode {
                    energy: r.next_f32(0.6, 1.0),
                    hausdorff: r.next_f32(0.6, 1.2),
                    entropy_delta: r.next_f32(0.0, 0.2),
                    elite_proximity: r.next_f32(0.0, 0.3),
                    toxicity: r.next_f32(0.0, 0.1),
                    is_signal: true,
                    px: r.next_f32(-0.3, 0.3),
                    py: r.next_f32(-0.3, 0.3),
                    born_cycle: 0,
                },
            );
            self.adj.entry(id).or_default();
            signal_ids.push(id);
        }

        // Connect signal nodes: ~5 edges per signal node (power-law-ish)
        // Total signal edges: ~2500
        for i in 0..signal_ids.len() {
            let connections = r.next_usize(4) + 3; // 3-6 edges
            for _ in 0..connections {
                let j = r.next_usize(signal_ids.len());
                if i != j {
                    self.add_edge(signal_ids[i], signal_ids[j]);
                }
            }
        }

        // ── Noise nodes: sparse, peripheral ──
        let mut noise_ids = Vec::new();
        for _ in 0..9000 {
            let id = self.alloc_id();
            self.nodes.insert(
                id,
                SimNode {
                    energy: r.next_f32(0.0, 0.25),
                    hausdorff: r.next_f32(0.05, 0.3),
                    entropy_delta: r.next_f32(0.5, 1.0),
                    elite_proximity: r.next_f32(0.7, 1.0),
                    toxicity: r.next_f32(0.3, 0.8),
                    is_signal: false,
                    px: r.next_f32(-0.9, 0.9),
                    py: r.next_f32(-0.9, 0.9),
                    born_cycle: 0,
                },
            );
            self.adj.entry(id).or_default();
            noise_ids.push(id);
        }

        // Connect noise nodes: ~8 edges each → ~36k noise edges
        for i in 0..noise_ids.len() {
            let connections = r.next_usize(6) + 5; // 5-10 edges
            for _ in 0..connections {
                // 70% connect to other noise, 30% to signal
                if r.chance(0.3) && !signal_ids.is_empty() {
                    let j = r.next_usize(signal_ids.len());
                    self.add_edge(noise_ids[i], signal_ids[j]);
                } else {
                    let j = r.next_usize(noise_ids.len());
                    if i != j {
                        self.add_edge(noise_ids[i], noise_ids[j]);
                    }
                }
            }
        }

        // Signal-signal additional backbone edges: ~12k
        for i in 0..signal_ids.len() {
            let extras = r.next_usize(8) + 5;
            for _ in 0..extras {
                let j = r.next_usize(signal_ids.len());
                if i != j {
                    self.add_edge(signal_ids[i], signal_ids[j]);
                }
            }
        }

        let total_edges: usize = self.adj.values().map(|s| s.len()).sum::<usize>() / 2;
        println!(
            "    Seeded: {} nodes, {} edges",
            self.nodes.len(),
            total_edges
        );

        self.centroid_0 = self.elite_centroid();
    }

    fn elite_centroid(&self) -> EliteCentroid {
        if self.nodes.is_empty() {
            return EliteCentroid::zero();
        }
        let mut energies: Vec<f32> = self.nodes.values().map(|n| n.energy).collect();
        energies.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let thr = energies[energies.len() / 5]; // top 20%
        let (mut sx, mut sy, mut c) = (0.0f32, 0.0f32, 0usize);
        for n in self.nodes.values() {
            if n.energy >= thr {
                sx += n.px;
                sy += n.py;
                c += 1;
            }
        }
        if c == 0 {
            return EliteCentroid::zero();
        }
        EliteCentroid {
            cx: sx / c as f32,
            cy: sy / c as f32,
        }
    }

    /// Indices of top 5% nodes by energy.
    fn elite_ids(&self) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = self
            .nodes
            .iter()
            .map(|(&id, n)| (id, n.energy))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top = (self.nodes.len() / 20).max(2);
        indexed[..top.min(indexed.len())]
            .iter()
            .map(|(id, _)| *id)
            .collect()
    }

    fn vitality(&self, n: &SimNode, id: usize) -> f32 {
        let prox = (1.0 - n.elite_proximity).max(0.0);
        let deg = self.degree(id);
        let z = self.w.w1_energy * n.energy
            + self.w.w2_hausdorff * n.hausdorff
            - self.w.w3_entropy * n.entropy_delta
            + self.w.w4_elite_prox * prox
            + self.w.w5_causal * deg as f32
            - self.w.w6_toxicity * n.toxicity;
        sigmoid(z)
    }

    /// Phase 0: Energy decay + noise injection.
    fn evolve(&mut self, r: &mut Rng) {
        // Pre-collect degrees (avoids borrow conflict with nodes.get_mut)
        let ids_and_degs: Vec<(usize, usize)> = self
            .nodes
            .keys()
            .map(|&id| (id, self.adj.get(&id).map_or(0, |s| s.len())))
            .collect();

        for (id, deg) in &ids_and_degs {
            if let Some(n) = self.nodes.get_mut(id) {
                n.energy = (n.energy - r.next_f32(0.005, 0.02)).max(0.0);
                if *deg == 0 {
                    n.toxicity = (n.toxicity + r.next_f32(0.001, 0.01)).min(1.0);
                }
                if n.is_signal && *deg > 0 {
                    n.energy = (n.energy + r.next_f32(0.005, 0.015)).min(1.0);
                }
            }
        }

        // Inject 10 noise nodes per cycle
        for _ in 0..10 {
            let id = self.alloc_id();
            self.nodes.insert(
                id,
                SimNode {
                    energy: r.next_f32(0.0, 0.20),
                    hausdorff: r.next_f32(0.05, 0.25),
                    entropy_delta: r.next_f32(0.5, 1.0),
                    elite_proximity: r.next_f32(0.7, 1.0),
                    toxicity: r.next_f32(0.3, 0.8),
                    is_signal: false,
                    px: r.next_f32(-0.9, 0.9),
                    py: r.next_f32(-0.9, 0.9),
                    born_cycle: 0,
                },
            );
            self.adj.entry(id).or_default(); // isolated — degree 0
        }
    }

    /// Birth: FOAM — void-born orphans, NO edges.
    fn birth_foam(&mut self, r: &mut Rng, n_create: usize, cycle: usize) -> usize {
        let mut created = 0;
        for _ in 0..n_create {
            if let Some(seed) = self.voids.pop() {
                let id = self.alloc_id();
                self.nodes.insert(
                    id,
                    SimNode {
                        energy: r.next_f32(0.15, 0.50),
                        hausdorff: r.next_f32(0.3, 0.8),
                        entropy_delta: r.next_f32(0.1, 0.4),
                        elite_proximity: r.next_f32(0.3, 0.6),
                        toxicity: r.next_f32(0.0, 0.2),
                        is_signal: false,
                        px: seed.px + r.next_f32(-0.05, 0.05),
                        py: seed.py + r.next_f32(-0.05, 0.05),
                        born_cycle: cycle,
                    },
                );
                self.adj.entry(id).or_default(); // ORPHAN — degree 0
                created += 1;
            }
        }
        created
    }

    /// Birth: ANCHORED — elite-parented, 2 REAL edges to parents.
    fn birth_anchored(
        &mut self,
        r: &mut Rng,
        n_create: usize,
        cycle: usize,
        polarize: bool,
    ) -> usize {
        let elites = self.elite_ids();
        if elites.len() < 2 {
            return 0;
        }

        let mut created = 0;
        for _ in 0..n_create {
            if self.voids.is_empty() {
                break;
            }
            self.voids.pop();

            // Select 2 elite parents
            let i1 = elites[r.next_usize(elites.len())];
            let mut i2 = elites[r.next_usize(elites.len())];
            let mut attempts = 0;
            while i2 == i1 && attempts < 5 {
                i2 = elites[r.next_usize(elites.len())];
                attempts += 1;
            }

            let p1 = self.nodes[&i1].clone();
            let p2 = self.nodes[&i2].clone();

            // Geodesic midpoint
            let mid_h = (p1.hausdorff + p2.hausdorff) / 2.0;
            let mid_pi = (p1.elite_proximity + p2.elite_proximity) / 2.0;
            let mid_tau = (p1.toxicity + p2.toxicity) / 2.0;
            let mid_xi_raw = (p1.entropy_delta + p2.entropy_delta) / 2.0;
            let mid_px = (p1.px + p2.px) / 2.0 + r.next_f32(-0.02, 0.02);
            let mid_py = (p1.py + p2.py) / 2.0 + r.next_f32(-0.02, 0.02);
            let inherited_e = 0.8 * ((p1.energy + p2.energy) / 2.0);

            // Entropy polarization (Dialectical only)
            let mid_xi = if polarize {
                let delta = 0.3 * (1.0 - (mid_xi_raw - 0.5).abs());
                if r.chance(0.5) {
                    (mid_xi_raw - delta).clamp(0.0, 1.0)
                } else {
                    (mid_xi_raw + delta).clamp(0.0, 1.0)
                }
            } else {
                mid_xi_raw
            };

            let id = self.alloc_id();
            self.nodes.insert(
                id,
                SimNode {
                    energy: inherited_e,
                    hausdorff: mid_h,
                    entropy_delta: mid_xi,
                    elite_proximity: mid_pi,
                    toxicity: mid_tau,
                    is_signal: false,
                    px: mid_px,
                    py: mid_py,
                    born_cycle: cycle,
                },
            );
            self.adj.entry(id).or_default();

            // κ=2: REAL structural edges to parents
            self.add_edge(id, i1);
            self.add_edge(id, i2);

            created += 1;
        }
        created
    }

    /// Execute one Zaratustra cycle.
    fn tick(&mut self, cycle: usize) -> CycleTelemetry {
        let t_start = Instant::now();
        let mut r = Rng::new(42u64.wrapping_mul(cycle as u64 + 7919));

        // Count newborns from LAST cycle that are still alive BEFORE purge
        let prev_born = self.last_born_count;
        let alive_before_purge = if cycle > 1 && prev_born > 0 {
            self.nodes
                .values()
                .filter(|n| n.born_cycle == cycle - 1)
                .count()
        } else {
            0
        };

        // Phase 0: evolve
        self.evolve(&mut r);

        // Phase 1: judgment — collect vitalities and condemn
        let mut sacrificed = 0usize;
        let mut to_remove = Vec::new();
        let mut vits = Vec::with_capacity(self.nodes.len());
        let mut esum = 0.0f32;

        let ids: Vec<usize> = self.nodes.keys().cloned().collect();
        for &id in &ids {
            let n = &self.nodes[&id];
            let v = self.vitality(n, id);
            vits.push(v);
            esum += n.energy;

            let deg = self.degree(id);
            let triple = v < self.vit_threshold && n.energy < self.eng_threshold && deg == 0;
            let toxic = n.toxicity > 0.8 && deg == 0;

            if triple || toxic {
                to_remove.push(id);
                sacrificed += 1;
                if n.is_signal {
                    self.signal_killed += 1;
                } else {
                    self.noise_killed += 1;
                }
            }
        }

        let total_before = self.nodes.len();

        // Remove condemned nodes (captures void seeds)
        for &id in &to_remove {
            if let Some(n) = self.nodes.get(&id) {
                self.voids.push(VoidSeed { px: n.px, py: n.py });
            }
            self.remove_node(id);
        }
        if self.voids.len() > 500 {
            self.voids.drain(0..self.voids.len() - 500);
        }

        // Newborn survival
        let alive_after_purge = if cycle > 1 && prev_born > 0 {
            self.nodes
                .values()
                .filter(|n| n.born_cycle == cycle - 1)
                .count()
        } else {
            0
        };
        let newborn_survival = if prev_born > 0 {
            alive_after_purge as f32 / prev_born as f32
        } else if alive_before_purge > 0 {
            alive_after_purge as f32 / alive_before_purge as f32
        } else {
            1.0
        };

        // Phase 3: generation
        let gen_target = ((sacrificed as f32 * 0.9).round() as usize + 5).min(self.voids.len());
        let nodes_created = match self.mode {
            BirthMode::Foam => self.birth_foam(&mut r, gen_target, cycle),
            BirthMode::Anchored => self.birth_anchored(&mut r, gen_target, cycle, false),
            BirthMode::Dialectical => self.birth_anchored(&mut r, gen_target, cycle, true),
        };
        self.last_born_count = nodes_created;

        // ── Structural Metrics ──
        let node_ids: Vec<usize> = self.nodes.keys().cloned().collect();
        let hs = structural_entropy(&self.adj, self.nodes.len());
        // Sample 32 BFS sources for efficiency
        let eg = global_efficiency_sampled(
            &self.adj,
            &node_ids,
            32,
            42u64.wrapping_mul(cycle as u64),
        );

        // ── Stats ──
        let n_f = vits.len().max(1) as f32;
        let mean_v: f32 = vits.iter().sum::<f32>() / n_f;
        let var_v: f32 = vits.iter().map(|v| (v - mean_v).powi(2)).sum::<f32>() / n_f;
        let mean_e = esum / total_before.max(1) as f32;

        // ── TGC with production formula ──
        let (tgc_raw, tgc_ema) = self.tgc.compute(
            nodes_created,
            self.nodes.len(),
            mean_v, // use mean vitality as quality proxy
            hs,
            eg,
        );

        let drift = self.centroid_0.distance(&self.elite_centroid());
        let total_edges: usize = self.adj.values().map(|s| s.len()).sum::<usize>() / 2;
        let ms = t_start.elapsed().as_secs_f32() * 1000.0;

        CycleTelemetry {
            cycle,
            total_nodes: self.nodes.len(),
            total_edges,
            sacrificed,
            nodes_created,
            signal_killed_total: self.signal_killed,
            noise_killed_total: self.noise_killed,
            mean_vitality: mean_v,
            variance_vitality: var_v,
            mean_energy: mean_e,
            structural_entropy: hs,
            global_efficiency: eg,
            tgc_raw,
            tgc_ema,
            elite_drift: drift,
            newborn_survival,
            ms_per_cycle: ms,
        }
    }
}

// ──────────────────────────────────────────────────
//  Experiment runner
// ──────────────────────────────────────────────────

fn run(mode: BirthMode, cycles: usize) -> Vec<CycleTelemetry> {
    let label = mode.label();
    // Output to experiments/ directory
    let _ = std::fs::create_dir_all("experiments");
    let csv = format!("experiments/telemetry_{}.csv", label);

    let mut s = Sim::new(mode);
    s.seed();

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&csv)
        .unwrap();
    writeln!(
        f,
        "cycle,total_nodes,total_edges,sacrificed,nodes_created,signal_killed,noise_killed,mean_vitality,variance_vitality,mean_energy,structural_entropy,global_efficiency,tgc_raw,tgc_ema,elite_drift,newborn_survival,ms_per_cycle"
    )
    .unwrap();

    let mut res = Vec::with_capacity(cycles);
    println!("  [{}] {} ciclos...", label, cycles);

    for i in 1..=cycles {
        let t = s.tick(i);
        writeln!(
            f,
            "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{:.2}",
            t.cycle,
            t.total_nodes,
            t.total_edges,
            t.sacrificed,
            t.nodes_created,
            t.signal_killed_total,
            t.noise_killed_total,
            t.mean_vitality,
            t.variance_vitality,
            t.mean_energy,
            t.structural_entropy,
            t.global_efficiency,
            t.tgc_raw,
            t.tgc_ema,
            t.elite_drift,
            t.newborn_survival,
            t.ms_per_cycle,
        )
        .unwrap();

        if i <= 3 || i % 25 == 0 || i == cycles {
            println!(
                "    {:03} N={:5} E={:5} Del={:3} Gen={:3} Surv={:.2} Hs={:.3} Eg={:.3} TGCr={:.4} TGCe={:.4} Drift={:.4} FP={} {:.1}ms",
                i, t.total_nodes, t.total_edges, t.sacrificed, t.nodes_created,
                t.newborn_survival, t.structural_entropy, t.global_efficiency,
                t.tgc_raw, t.tgc_ema, t.elite_drift,
                t.signal_killed_total, t.ms_per_cycle,
            );
        }

        let collapse = t.total_nodes < 100;
        res.push(t);
        if collapse {
            println!("    [COLAPSO] ciclo {}", i);
            break;
        }
    }

    println!(
        "  [{}] FP={} noise_del={} final_N={} final_E={}",
        label,
        s.signal_killed,
        s.noise_killed,
        res.last().map_or(0, |r| r.total_nodes),
        res.last().map_or(0, |r| r.total_edges),
    );
    println!();
    res
}

// ──────────────────────────────────────────────────
//  Raw numbers
// ──────────────────────────────────────────────────

fn raw(label: &str, r: &[CycleTelemetry]) {
    let tail: Vec<&CycleTelemetry> = if r.len() > 100 {
        r[r.len() - 100..].iter().collect()
    } else {
        r.iter().collect()
    };
    let n = tail.len() as f32;
    if n == 0.0 {
        return;
    }

    let avg = |f: fn(&CycleTelemetry) -> f32| -> f32 {
        tail.iter().map(|t| f(t)).sum::<f32>() / n
    };

    println!(
        "  ┌─── {} ─── LAST {} CYCLES ───┐",
        label,
        tail.len()
    );
    println!("  │ avg_nodes        = {:.1}", avg(|t| t.total_nodes as f32));
    println!("  │ avg_edges        = {:.1}", avg(|t| t.total_edges as f32));
    println!("  │ avg_sacrificed   = {:.2}", avg(|t| t.sacrificed as f32));
    println!("  │ avg_created      = {:.2}", avg(|t| t.nodes_created as f32));
    println!("  │ avg_newborn_srv  = {:.4}", avg(|t| t.newborn_survival));
    println!("  │ avg_mean_V       = {:.4}", avg(|t| t.mean_vitality));
    println!("  │ avg_Var(V)       = {:.6}", avg(|t| t.variance_vitality));
    println!("  │ avg_mean_E       = {:.4}", avg(|t| t.mean_energy));
    println!("  │ avg_Hs           = {:.4}", avg(|t| t.structural_entropy));
    println!("  │ avg_Eg           = {:.4}", avg(|t| t.global_efficiency));
    println!("  │ avg_TGC_raw      = {:.6}", avg(|t| t.tgc_raw));
    println!("  │ avg_TGC_ema      = {:.6}", avg(|t| t.tgc_ema));
    println!("  │ avg_elite_drift  = {:.6}", avg(|t| t.elite_drift));
    println!("  │ avg_ms/cycle     = {:.2}", avg(|t| t.ms_per_cycle));
    println!(
        "  │ total_FP         = {}",
        tail.last().map_or(0, |t| t.signal_killed_total)
    );
    println!(
        "  │ total_noise_del  = {}",
        tail.last().map_or(0, |t| t.noise_killed_total)
    );
    println!("  └──────────────────────────────────────────┘");
}

// ──────────────────────────────────────────────────
//  Main
// ──────────────────────────────────────────────────

fn main() {
    println!("=================================================================");
    println!("  BENCHMARK: Real Graph Topology + Structural Metrics");
    println!("  10,000 nodes | ~50,000 edges | 100 cycles");
    println!("  TGC = intensity * quality * (1+{}*dH) * (1+{}*dE)", ALPHA, BETA);
    println!("  Variavel isolada: modo de nascimento");
    println!("=================================================================");
    println!();

    let cycles = 100;

    println!("━━━ D: FOAM (baseline — orphans, degree=0) ━━━");
    let rd = run(BirthMode::Foam, cycles);

    println!("━━━ E: ANCHORED (elite-parented, degree=2, no pol.) ━━━");
    let re = run(BirthMode::Anchored, cycles);

    println!("━━━ F: DIALECTICAL (elite-parented, degree=2, +pol.) ━━━");
    let rf = run(BirthMode::Dialectical, cycles);

    println!("=================================================================");
    println!("  DADOS BRUTOS — ULTIMOS CICLOS");
    println!("=================================================================");
    println!();
    raw("D_FOAM", &rd);
    println!();
    raw("E_ANCHORED", &re);
    println!();
    raw("F_DIALECTICAL", &rf);
    println!();
    println!("=================================================================");
    println!("  Rode: py -3 plot_metabolism.py");
    println!("=================================================================");
}
