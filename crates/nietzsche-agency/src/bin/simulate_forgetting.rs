//! Protocolo Experimental: O Sistema Sabe Nascer?
//!
//! 3 cenarios sobre 5,000 nos x 500 ciclos, variavel isolada = modo de nascimento:
//!
//! | Mode | Birth | kappa | Polarization | Purpose |
//! |------|-------|-------|-------------|---------|
//! | D: FOAM | void-born orphans | 0 | none | Baseline — structural foam |
//! | E: ANCHORED | elite-parented | 2 | none | Isolate connectivity effect |
//! | F: DIALECTICAL | elite-parented | 2 | +-delta | Full Option A |
//!
//! Identical seeding, energy decay, noise injection, deletion logic.
//! ONLY variable: how new nodes are born after purge.
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

use std::fs::OpenOptions;
use std::io::Write;

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
    causal_edges: usize,
    toxicity: f32,
    is_signal: bool,
    px: f32,
    py: f32,
    born_cycle: usize, // cycle this node was created (0 = seed)
}

#[derive(Clone)]
struct VoidSeed {
    px: f32,
    py: f32,
}

struct CycleTelemetry {
    cycle: usize,
    total_nodes: usize,
    sacrificed: usize,
    nodes_created: usize,
    signal_killed_total: usize,
    noise_killed_total: usize,
    mean_vitality: f32,
    variance_vitality: f32,
    mean_energy: f32,
    tgc: f32,
    elite_drift: f32,
    newborn_survival: f32, // fraction of last-cycle newborns that survived this cycle
}

#[derive(Clone, Copy)]
struct EliteCentroid { cx: f32, cy: f32 }

impl EliteCentroid {
    fn zero() -> Self { Self { cx: 0.0, cy: 0.0 } }
    fn distance(&self, o: &EliteCentroid) -> f32 {
        ((self.cx - o.cx).powi(2) + (self.cy - o.cy).powi(2)).sqrt()
    }
}

/// Birth mode — the ONLY experimental variable.
#[derive(Clone, Copy, PartialEq)]
enum BirthMode {
    /// D: Void-born orphans (kappa=0, random mid-quality). The "foam" baseline.
    Foam,
    /// E: Elite-parented, kappa=2, NO entropy polarization. Isolates connectivity.
    Anchored,
    /// F: Elite-parented, kappa=2, WITH entropy polarization. Full Option A.
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
//  TGC Tracker
// ──────────────────────────────────────────────────

struct TgcTracker { ema: f32, alpha: f32 }

impl TgcTracker {
    fn new() -> Self { Self { ema: 0.0, alpha: 0.3 } }

    fn update(&mut self, created: usize, total_after: usize, mean_vitality: f32) -> f32 {
        let raw = (created as f32 / total_after.max(1) as f32) * mean_vitality.clamp(0.0, 1.0);
        self.ema = self.alpha * raw + (1.0 - self.alpha) * self.ema;
        self.ema
    }
}

// ──────────────────────────────────────────────────
//  Deterministic PRNG
// ──────────────────────────────────────────────────

struct Rng { state: u64 }

impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let t = ((self.state >> 33) as f32) / (u32::MAX as f32);
        lo + t * (hi - lo)
    }

    fn next_usize(&mut self, hi: usize) -> usize {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.state >> 33) as usize) % hi.max(1)
    }

    /// Returns true with probability p.
    fn chance(&mut self, p: f32) -> bool {
        self.next_f32(0.0, 1.0) < p
    }
}

// ──────────────────────────────────────────────────
//  The Simulator
// ──────────────────────────────────────────────────

struct Sim {
    w: VitalityWeights,
    graph: Vec<SimNode>,
    voids: Vec<VoidSeed>,
    noise_killed: usize,
    signal_killed: usize,
    vit_threshold: f32,
    eng_threshold: f32,
    tgc: TgcTracker,
    centroid_0: EliteCentroid,
    mode: BirthMode,
    last_born_count: usize, // how many nodes were born last cycle
}

impl Sim {
    fn new(mode: BirthMode) -> Self {
        Self {
            w: VitalityWeights::default(),
            graph: Vec::new(),
            voids: Vec::new(),
            noise_killed: 0,
            signal_killed: 0,
            vit_threshold: 0.30,
            eng_threshold: 0.10,
            tgc: TgcTracker::new(),
            centroid_0: EliteCentroid::zero(),
            mode,
            last_born_count: 0,
        }
    }

    fn seed(&mut self) {
        let mut r = Rng::new(42);
        for _ in 0..1000 {
            self.graph.push(SimNode {
                energy: r.next_f32(0.6, 1.0),
                hausdorff: r.next_f32(0.6, 1.2),
                entropy_delta: r.next_f32(0.0, 0.2),
                elite_proximity: r.next_f32(0.0, 0.3),
                causal_edges: 3,
                toxicity: r.next_f32(0.0, 0.1),
                is_signal: true,
                px: r.next_f32(-0.3, 0.3),
                py: r.next_f32(-0.3, 0.3),
                born_cycle: 0,
            });
        }
        for _ in 0..4000 {
            self.graph.push(SimNode {
                energy: r.next_f32(0.0, 0.25),
                hausdorff: r.next_f32(0.05, 0.3),
                entropy_delta: r.next_f32(0.5, 1.0),
                elite_proximity: r.next_f32(0.7, 1.0),
                causal_edges: 0,
                toxicity: r.next_f32(0.3, 0.8),
                is_signal: false,
                px: r.next_f32(-0.9, 0.9),
                py: r.next_f32(-0.9, 0.9),
                born_cycle: 0,
            });
        }
        self.centroid_0 = self.elite_centroid();
    }

    fn elite_centroid(&self) -> EliteCentroid {
        if self.graph.is_empty() { return EliteCentroid::zero(); }
        let mut e: Vec<f32> = self.graph.iter().map(|n| n.energy).collect();
        e.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let thr = e[e.len() / 5];
        let (mut sx, mut sy, mut c) = (0.0f32, 0.0f32, 0usize);
        for n in &self.graph {
            if n.energy >= thr { sx += n.px; sy += n.py; c += 1; }
        }
        if c == 0 { return EliteCentroid::zero(); }
        EliteCentroid { cx: sx / c as f32, cy: sy / c as f32 }
    }

    /// Collect indices of top 5% nodes by energy — the elite pool for parenting.
    fn elite_indices(&self) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = self.graph.iter()
            .enumerate().map(|(i, n)| (i, n.energy)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top = (self.graph.len() / 20).max(2); // top 5%, min 2
        indexed[..top.min(indexed.len())].iter().map(|(i, _)| *i).collect()
    }

    fn vitality(&self, n: &SimNode) -> f32 {
        let prox = (1.0 - n.elite_proximity).max(0.0);
        let z = self.w.w1_energy * n.energy
              + self.w.w2_hausdorff * n.hausdorff
              - self.w.w3_entropy * n.entropy_delta
              + self.w.w4_elite_prox * prox
              + self.w.w5_causal * n.causal_edges as f32
              - self.w.w6_toxicity * n.toxicity;
        sigmoid(z)
    }

    /// Phase 0: Energy decay + noise injection. Identical for all modes.
    fn evolve(&mut self, r: &mut Rng) {
        for n in &mut self.graph {
            n.energy = (n.energy - r.next_f32(0.005, 0.02)).max(0.0);
            if n.causal_edges == 0 {
                n.toxicity = (n.toxicity + r.next_f32(0.001, 0.01)).min(1.0);
            }
        }
        for n in &mut self.graph {
            if n.is_signal && n.causal_edges > 0 {
                n.energy = (n.energy + r.next_f32(0.005, 0.015)).min(1.0);
            }
        }
        // Constant noise injection: 10/cycle
        for _ in 0..10 {
            self.graph.push(SimNode {
                energy: r.next_f32(0.0, 0.20),
                hausdorff: r.next_f32(0.05, 0.25),
                entropy_delta: r.next_f32(0.5, 1.0),
                elite_proximity: r.next_f32(0.7, 1.0),
                causal_edges: 0,
                toxicity: r.next_f32(0.3, 0.8),
                is_signal: false,
                px: r.next_f32(-0.9, 0.9),
                py: r.next_f32(-0.9, 0.9),
                born_cycle: 0, // noise injection, not tracked
            });
        }
    }

    /// Birth: FOAM — old behavior. Void-born orphans, kappa=0.
    fn birth_foam(&mut self, r: &mut Rng, n_create: usize, cycle: usize) -> usize {
        let mut created = 0;
        for _ in 0..n_create {
            if let Some(seed) = self.voids.pop() {
                self.graph.push(SimNode {
                    energy: r.next_f32(0.15, 0.50),
                    hausdorff: r.next_f32(0.3, 0.8),
                    entropy_delta: r.next_f32(0.1, 0.4),
                    elite_proximity: r.next_f32(0.3, 0.6),
                    causal_edges: 0, // ORPHAN — the foam problem
                    toxicity: r.next_f32(0.0, 0.2),
                    is_signal: false,
                    px: seed.px + r.next_f32(-0.05, 0.05),
                    py: seed.py + r.next_f32(-0.05, 0.05),
                    born_cycle: cycle,
                });
                created += 1;
            }
        }
        created
    }

    /// Birth: ANCHORED — elite parented, kappa=2, NO polarization.
    fn birth_anchored(&mut self, r: &mut Rng, n_create: usize, cycle: usize, polarize: bool) -> usize {
        let elites = self.elite_indices();
        if elites.len() < 2 { return 0; }

        let mut created = 0;
        for _ in 0..n_create {
            if self.voids.is_empty() { break; }
            self.voids.pop(); // consume void seed (for accounting parity)

            // Select 2 elite parents
            let i1 = elites[r.next_usize(elites.len())];
            let mut i2 = elites[r.next_usize(elites.len())];
            // Ensure different parents
            let mut attempts = 0;
            while i2 == i1 && attempts < 5 {
                i2 = elites[r.next_usize(elites.len())];
                attempts += 1;
            }

            let p1 = &self.graph[i1];
            let p2 = &self.graph[i2];

            // Geodesic midpoint (Euclidean proxy in Poincare 2D)
            let mid_h = (p1.hausdorff + p2.hausdorff) / 2.0;
            let mid_pi = (p1.elite_proximity + p2.elite_proximity) / 2.0;
            let mid_tau = (p1.toxicity + p2.toxicity) / 2.0;
            let mid_xi_raw = (p1.entropy_delta + p2.entropy_delta) / 2.0;
            let mid_px = (p1.px + p2.px) / 2.0 + r.next_f32(-0.02, 0.02);
            let mid_py = (p1.py + p2.py) / 2.0 + r.next_f32(-0.02, 0.02);

            // Inherited energy: beta = 0.8
            let inherited_e = 0.8 * ((p1.energy + p2.energy) / 2.0);

            // Entropy polarization (only if mode == Dialectical)
            let mid_xi = if polarize {
                // Adaptive delta: amplifies when near center, reduces at extremes
                let delta = 0.3 * (1.0 - (mid_xi_raw - 0.5).abs());
                if r.chance(0.5) {
                    (mid_xi_raw - delta).clamp(0.0, 1.0) // low entropy — born near top
                } else {
                    (mid_xi_raw + delta).clamp(0.0, 1.0) // high entropy — born as explorer
                }
            } else {
                mid_xi_raw // no polarization
            };

            self.graph.push(SimNode {
                energy: inherited_e,
                hausdorff: mid_h,
                entropy_delta: mid_xi,
                elite_proximity: mid_pi,
                causal_edges: 2,    // ANCHORED — kappa=2 (structural edges to parents)
                toxicity: mid_tau,
                is_signal: false,
                px: mid_px,
                py: mid_py,
                born_cycle: cycle,
            });
            created += 1;
        }
        created
    }

    /// Execute one Zaratustra cycle.
    fn tick(&mut self, cycle: usize) -> CycleTelemetry {
        let mut r = Rng::new(42u64.wrapping_mul(cycle as u64 + 7919));

        // Count newborns from LAST cycle that are still alive BEFORE this cycle's purge
        let prev_born = self.last_born_count;
        let alive_before = if cycle > 1 && prev_born > 0 {
            self.graph.iter().filter(|n| n.born_cycle == cycle - 1).count()
        } else {
            0
        };

        // Phase 0: evolve
        self.evolve(&mut r);

        // Phase 1: judgment
        let mut sacrificed = 0usize;
        let mut vits = Vec::with_capacity(self.graph.len());
        let mut surviving = Vec::with_capacity(self.graph.len());
        let mut esum = 0.0f32;

        for n in &self.graph {
            let v = self.vitality(n);
            vits.push(v);
            esum += n.energy;

            let triple = v < self.vit_threshold && n.energy < self.eng_threshold && n.causal_edges == 0;
            let toxic = n.toxicity > 0.8 && n.causal_edges == 0;

            if triple || toxic {
                self.voids.push(VoidSeed { px: n.px, py: n.py });
                sacrificed += 1;
                if n.is_signal { self.signal_killed += 1; } else { self.noise_killed += 1; }
            } else {
                surviving.push(n.clone());
            }
        }

        let total_before = self.graph.len();
        self.graph = surviving;
        if self.voids.len() > 500 { self.voids.drain(0..self.voids.len() - 500); }

        // Newborn survival: of last cycle's newborns, how many survived THIS purge?
        let alive_after = if cycle > 1 && prev_born > 0 {
            self.graph.iter().filter(|n| n.born_cycle == cycle - 1).count()
        } else {
            0
        };
        let newborn_survival = if prev_born > 0 {
            alive_after as f32 / prev_born as f32
        } else {
            1.0 // no newborns = vacuously all survived
        };

        // Phase 3: generation — ~90% of deleted + 5 baseline, capped by voids
        let gen_target = ((sacrificed as f32 * 0.9).round() as usize + 5).min(self.voids.len());
        let nodes_created = match self.mode {
            BirthMode::Foam => self.birth_foam(&mut r, gen_target, cycle),
            BirthMode::Anchored => self.birth_anchored(&mut r, gen_target, cycle, false),
            BirthMode::Dialectical => self.birth_anchored(&mut r, gen_target, cycle, true),
        };
        self.last_born_count = nodes_created;

        // Stats
        let n = vits.len().max(1) as f32;
        let mean_v: f32 = vits.iter().sum::<f32>() / n;
        let var_v: f32 = vits.iter().map(|v| (v - mean_v).powi(2)).sum::<f32>() / n;
        let mean_e = esum / total_before.max(1) as f32;
        let tgc = self.tgc.update(nodes_created, self.graph.len(), mean_v);
        let drift = self.centroid_0.distance(&self.elite_centroid());

        CycleTelemetry {
            cycle,
            total_nodes: self.graph.len(),
            sacrificed,
            nodes_created,
            signal_killed_total: self.signal_killed,
            noise_killed_total: self.noise_killed,
            mean_vitality: mean_v,
            variance_vitality: var_v,
            mean_energy: mean_e,
            tgc,
            elite_drift: drift,
            newborn_survival,
        }
    }
}

// ──────────────────────────────────────────────────
//  Experiment runner
// ──────────────────────────────────────────────────

fn run(mode: BirthMode) -> Vec<CycleTelemetry> {
    let label = mode.label();
    let csv = format!("telemetry_{}.csv", label);

    let mut s = Sim::new(mode);
    s.seed();

    let mut f = OpenOptions::new().write(true).create(true).truncate(true).open(&csv).unwrap();
    writeln!(f,
        "cycle,total_nodes,sacrificed,nodes_created,signal_killed,noise_killed,mean_vitality,variance_vitality,mean_energy,tgc,elite_drift,newborn_survival"
    ).unwrap();

    let mut res = Vec::with_capacity(500);

    println!("  [{}] 500 ciclos...", label);

    for i in 1..=500 {
        let t = s.tick(i);
        writeln!(f, "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4}",
            t.cycle, t.total_nodes, t.sacrificed, t.nodes_created,
            t.signal_killed_total, t.noise_killed_total,
            t.mean_vitality, t.variance_vitality, t.mean_energy,
            t.tgc, t.elite_drift, t.newborn_survival,
        ).unwrap();

        if i <= 3 || i % 100 == 0 || i == 500 {
            println!(
                "    {:03} N={:5} Del={:3} Gen={:3} Surv={:.2} V={:.3} Var={:.4} E={:.3} TGC={:.4} Drift={:.4} FP={}",
                i, t.total_nodes, t.sacrificed, t.nodes_created, t.newborn_survival,
                t.mean_vitality, t.variance_vitality, t.mean_energy, t.tgc, t.elite_drift,
                t.signal_killed_total,
            );
        }

        let collapse = t.total_nodes < 100;
        res.push(t);
        if collapse { println!("    [COLAPSO] ciclo {}", i); break; }
    }

    println!("  [{}] FP={} noise_del={} final_N={}",
        label, s.signal_killed, s.noise_killed,
        res.last().map_or(0, |r| r.total_nodes));
    println!();
    res
}

// ──────────────────────────────────────────────────
//  Raw numbers
// ──────────────────────────────────────────────────

fn raw(label: &str, r: &[CycleTelemetry]) {
    let tail: Vec<&CycleTelemetry> = if r.len() > 100 { r[r.len()-100..].iter().collect() } else { r.iter().collect() };
    let n = tail.len() as f32;
    if n == 0.0 { return; }

    let avg = |f: fn(&CycleTelemetry) -> f32| -> f32 { tail.iter().map(|t| f(t)).sum::<f32>() / n };
    let minmax = |f: fn(&CycleTelemetry) -> f32| -> (f32, f32) {
        (tail.iter().map(|t| f(t)).fold(f32::INFINITY, f32::min),
         tail.iter().map(|t| f(t)).fold(f32::NEG_INFINITY, f32::max))
    };

    let (tmin, tmax) = minmax(|t| t.tgc);
    let (vmin, vmax) = minmax(|t| t.variance_vitality);

    println!("  ┌─── {} ─── LAST {} CYCLES ───┐", label, tail.len());
    println!("  │ avg_nodes       = {:.1}", avg(|t| t.total_nodes as f32));
    println!("  │ avg_sacrificed  = {:.2}", avg(|t| t.sacrificed as f32));
    println!("  │ avg_created     = {:.2}", avg(|t| t.nodes_created as f32));
    println!("  │ avg_newborn_srv = {:.4}", avg(|t| t.newborn_survival));
    println!("  │ avg_mean_V      = {:.4}", avg(|t| t.mean_vitality));
    println!("  │ avg_Var(V)      = {:.6}  [{:.6}..{:.6}]", avg(|t| t.variance_vitality), vmin, vmax);
    println!("  │ avg_mean_E      = {:.4}", avg(|t| t.mean_energy));
    println!("  │ avg_TGC         = {:.6}  [{:.6}..{:.6}]", avg(|t| t.tgc), tmin, tmax);
    println!("  │ avg_elite_drift = {:.6}", avg(|t| t.elite_drift));
    println!("  │ total_FP        = {}", tail.last().map_or(0, |t| t.signal_killed_total));
    println!("  │ total_noise_del = {}", tail.last().map_or(0, |t| t.noise_killed_total));
    println!("  └────────────────────────────────────────┘");
}

// ──────────────────────────────────────────────────
//  Main
// ──────────────────────────────────────────────────

fn main() {
    println!("=================================================================");
    println!("  PROTOCOLO: O SISTEMA SABE NASCER?");
    println!("  Variavel isolada: modo de nascimento");
    println!("  3 cenarios x 500 ciclos x 5000 nos");
    println!("=================================================================");
    println!();

    println!("━━━ D: FOAM (baseline — orfaos void-born, kappa=0) ━━━");
    let rd = run(BirthMode::Foam);

    println!("━━━ E: ANCHORED (elite-parented, kappa=2, sem polarizacao) ━━━");
    let re = run(BirthMode::Anchored);

    println!("━━━ F: DIALECTICAL (elite-parented, kappa=2, COM polarizacao) ━━━");
    let rf = run(BirthMode::Dialectical);

    println!("=================================================================");
    println!("  DADOS BRUTOS — ULTIMOS 100 CICLOS");
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
