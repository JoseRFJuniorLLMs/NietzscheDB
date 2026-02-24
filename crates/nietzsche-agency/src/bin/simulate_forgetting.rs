//! Protocolo Experimental: Termodinamica do Esquecimento Ativo
//!
//! 3 variacoes experimentais sobre 5,000 nos x 500 ciclos Zaratustra:
//!
//! | Mode | Generation | Purpose |
//! |------|-----------|---------|
//! | A: DELETE_ONLY | 0 | Control — pure catabolism |
//! | B: LOW_GEN | ~30% of deleted | Mild anabolism |
//! | C: MATCHED_GEN | ~100% of deleted | Balanced metabolism |
//!
//! All 3 share identical seeding, energy decay, and noise injection.
//! The ONLY variable is void-seeded node generation after each purge.
//!
//! ## Usage
//! ```sh
//! cargo run --release --bin simulate_forgetting
//! ```
//!
//! ## Output
//! - `telemetry_A_delete_only.csv`
//! - `telemetry_B_low_gen.csv`
//! - `telemetry_C_matched_gen.csv`
//! - Raw numbers from last 100 cycles printed to stdout.

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
}

/// Void coordinate — captures WHERE a node died in Poincare space.
#[derive(Clone)]
struct VoidSeed {
    px: f32,
    py: f32,
    energy_at_death: f32,
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
}

#[derive(Clone, Copy)]
struct EliteCentroid {
    cx: f32,
    cy: f32,
}

impl EliteCentroid {
    fn zero() -> Self { Self { cx: 0.0, cy: 0.0 } }

    fn distance(&self, other: &EliteCentroid) -> f32 {
        let dx = self.cx - other.cx;
        let dy = self.cy - other.cy;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Generation mode for the experiment.
#[derive(Clone, Copy, PartialEq)]
enum GenMode {
    DeleteOnly,   // A: zero generation
    LowGen,       // B: ~30% of deleted
    MatchedGen,   // C: ~100% of deleted
}

impl GenMode {
    fn label(&self) -> &'static str {
        match self {
            GenMode::DeleteOnly => "A_delete_only",
            GenMode::LowGen => "B_low_gen",
            GenMode::MatchedGen => "C_matched_gen",
        }
    }

    fn gen_ratio(&self) -> f32 {
        match self {
            GenMode::DeleteOnly => 0.0,
            GenMode::LowGen => 0.30,
            GenMode::MatchedGen => 1.0,
        }
    }
}

// ──────────────────────────────────────────────────
//  TGC Tracker — uses ACTUAL generation count
// ──────────────────────────────────────────────────

struct TgcTracker {
    ema: f32,
    alpha: f32,
    baseline: f32,
}

impl TgcTracker {
    fn new() -> Self {
        Self { ema: 0.0, alpha: 0.3, baseline: 0.0 }
    }

    /// TGC(t) = (G_t / V_t) * Quality, EMA smoothed.
    /// G_t = actual nodes created this cycle.
    /// V_t = total nodes after cycle.
    /// Quality = mean_vitality of survivors.
    fn update(&mut self, created: usize, total_after: usize, mean_vitality: f32) -> f32 {
        let v_t = total_after.max(1) as f32;
        let g_t = created as f32;
        let quality = mean_vitality.clamp(0.0, 1.0);
        let raw_tgc = (g_t / v_t) * quality;

        self.ema = self.alpha * raw_tgc + (1.0 - self.alpha) * self.ema;
        self.ema
    }

    fn set_baseline(&mut self, val: f32) { self.baseline = val; }
}

// ──────────────────────────────────────────────────
//  Deterministic PRNG (no external dependency)
// ──────────────────────────────────────────────────

struct Rng { state: u64 }

impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let t = ((self.state >> 33) as f32) / (u32::MAX as f32);
        lo + t * (hi - lo)
    }

    fn next_usize(&mut self, lo: usize, hi: usize) -> usize {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let range = (hi - lo).max(1);
        lo + ((self.state >> 33) as usize) % range
    }
}

// ──────────────────────────────────────────────────
//  The Simulator
// ──────────────────────────────────────────────────

struct ForgettingSimulator {
    weights: VitalityWeights,
    graph: Vec<SimNode>,
    void_seeds: Vec<VoidSeed>,
    noise_killed: usize,
    signal_killed: usize,
    vitality_threshold: f32,
    energy_threshold: f32,
    tgc_tracker: TgcTracker,
    initial_centroid: EliteCentroid,
    gen_mode: GenMode,
}

impl ForgettingSimulator {
    fn new(mode: GenMode) -> Self {
        Self {
            weights: VitalityWeights::default(),
            graph: Vec::new(),
            void_seeds: Vec::new(),
            noise_killed: 0,
            signal_killed: 0,
            vitality_threshold: 0.30,
            energy_threshold: 0.10,
            tgc_tracker: TgcTracker::new(),
            initial_centroid: EliteCentroid::zero(),
            gen_mode: mode,
        }
    }

    fn seed_graph(&mut self) {
        let mut rng = Rng::new(42);

        // 1,000 Signal (high energy, causal anchoring)
        for _ in 0..1000 {
            self.graph.push(SimNode {
                energy: rng.next_f32(0.6, 1.0),
                hausdorff: rng.next_f32(0.6, 1.2),
                entropy_delta: rng.next_f32(0.0, 0.2),
                elite_proximity: rng.next_f32(0.0, 0.3),
                causal_edges: 3,
                toxicity: rng.next_f32(0.0, 0.1),
                is_signal: true,
                px: rng.next_f32(-0.3, 0.3),
                py: rng.next_f32(-0.3, 0.3),
            });
        }

        // 4,000 Noise (low energy, no causal protection)
        for _ in 0..4000 {
            self.graph.push(SimNode {
                energy: rng.next_f32(0.0, 0.25),
                hausdorff: rng.next_f32(0.05, 0.3),
                entropy_delta: rng.next_f32(0.5, 1.0),
                elite_proximity: rng.next_f32(0.7, 1.0),
                causal_edges: 0,
                toxicity: rng.next_f32(0.3, 0.8),
                is_signal: false,
                px: rng.next_f32(-0.9, 0.9),
                py: rng.next_f32(-0.9, 0.9),
            });
        }

        self.initial_centroid = self.compute_elite_centroid();
    }

    fn compute_elite_centroid(&self) -> EliteCentroid {
        if self.graph.is_empty() { return EliteCentroid::zero(); }

        let mut energies: Vec<f32> = self.graph.iter().map(|n| n.energy).collect();
        energies.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = energies[energies.len() / 5];

        let mut sx = 0.0f32;
        let mut sy = 0.0f32;
        let mut c = 0usize;

        for n in &self.graph {
            if n.energy >= threshold {
                sx += n.px;
                sy += n.py;
                c += 1;
            }
        }
        if c == 0 { return EliteCentroid::zero(); }
        EliteCentroid { cx: sx / c as f32, cy: sy / c as f32 }
    }

    fn vitality(&self, node: &SimNode) -> f32 {
        let w = &self.weights;
        let prox = (1.0 - node.elite_proximity).max(0.0);
        let z = w.w1_energy * node.energy
              + w.w2_hausdorff * node.hausdorff
              - w.w3_entropy * node.entropy_delta
              + w.w4_elite_prox * prox
              + w.w5_causal * node.causal_edges as f32
              - w.w6_toxicity * node.toxicity;
        sigmoid(z)
    }

    /// Phase 0: Energy decay + noise injection (identical across all 3 experiments).
    fn evolve_universe(&mut self, rng: &mut Rng) {
        // Energy decay
        for node in &mut self.graph {
            let decay = rng.next_f32(0.005, 0.02);
            node.energy = (node.energy - decay).max(0.0);
            if node.causal_edges == 0 {
                node.toxicity = (node.toxicity + rng.next_f32(0.001, 0.01)).min(1.0);
            }
        }

        // Will-to-Power recovery for signal
        for node in &mut self.graph {
            if node.is_signal && node.causal_edges > 0 {
                let recovery = rng.next_f32(0.005, 0.015);
                node.energy = (node.energy + recovery).min(1.0);
            }
        }

        // Constant noise injection: 10/cycle (same for all modes)
        for _ in 0..10 {
            self.graph.push(SimNode {
                energy: rng.next_f32(0.0, 0.20),
                hausdorff: rng.next_f32(0.05, 0.25),
                entropy_delta: rng.next_f32(0.5, 1.0),
                elite_proximity: rng.next_f32(0.7, 1.0),
                causal_edges: 0,
                toxicity: rng.next_f32(0.3, 0.8),
                is_signal: false,
                px: rng.next_f32(-0.9, 0.9),
                py: rng.next_f32(-0.9, 0.9),
            });
        }
    }

    /// Phase 3: Void-seeded generation. Creates new nodes at void coordinates.
    /// These are HEALTHY nodes — higher quality than random noise injection.
    fn generate_from_void(&mut self, rng: &mut Rng, deleted: usize) -> usize {
        let ratio = self.gen_mode.gen_ratio();
        if ratio == 0.0 { return 0; }

        let to_create = ((deleted as f32 * ratio).round() as usize).min(self.void_seeds.len());
        if to_create == 0 { return 0; }

        let mut created = 0;
        for _ in 0..to_create {
            if let Some(seed) = self.void_seeds.pop() {
                // Generate a MID-quality node at the void position.
                // Not elite (no causal), but better than pure noise.
                // Represents structural regrowth from the void.
                self.graph.push(SimNode {
                    energy: rng.next_f32(0.15, 0.50),         // mid-range energy
                    hausdorff: rng.next_f32(0.3, 0.8),        // mid-range fractal
                    entropy_delta: rng.next_f32(0.1, 0.4),    // low entropy (healthy)
                    elite_proximity: rng.next_f32(0.3, 0.6),  // moderate proximity
                    causal_edges: 0,                           // no causal — must earn it
                    toxicity: rng.next_f32(0.0, 0.2),         // low toxicity
                    is_signal: false,                          // void-born, not signal
                    px: seed.px + rng.next_f32(-0.05, 0.05),  // near void position
                    py: seed.py + rng.next_f32(-0.05, 0.05),
                });
                created += 1;
            }
        }
        created
    }

    /// Run one complete Zaratustra cycle.
    fn run_cycle(&mut self, cycle_id: usize) -> CycleTelemetry {
        let mut rng = Rng::new(42u64.wrapping_mul(cycle_id as u64 + 7919));

        // Phase 0: Universe evolution (identical for all modes)
        self.evolve_universe(&mut rng);

        // Phase 1: Judgment — compute vitality and apply Triple Condition
        let mut sacrificed = 0usize;
        let mut vitalities = Vec::with_capacity(self.graph.len());
        let mut surviving = Vec::with_capacity(self.graph.len());
        let mut energy_sum = 0.0f32;

        for node in &self.graph {
            let v = self.vitality(node);
            vitalities.push(v);
            energy_sum += node.energy;

            let cond1 = v < self.vitality_threshold;
            let cond2 = node.energy < self.energy_threshold;
            let cond3 = node.causal_edges == 0;
            let toxic = node.toxicity > 0.8 && node.causal_edges == 0;

            if (cond1 && cond2 && cond3) || toxic {
                // Capture void seed before deletion
                self.void_seeds.push(VoidSeed {
                    px: node.px,
                    py: node.py,
                    energy_at_death: node.energy,
                });
                sacrificed += 1;
                if node.is_signal {
                    self.signal_killed += 1;
                } else {
                    self.noise_killed += 1;
                }
            } else {
                surviving.push(node.clone());
            }
        }

        let total_before = self.graph.len();
        self.graph = surviving;

        // Cap void seeds at 500
        if self.void_seeds.len() > 500 {
            self.void_seeds.drain(0..self.void_seeds.len() - 500);
        }

        // Phase 3: Void-seeded generation (varies by mode)
        let nodes_created = self.generate_from_void(&mut rng, sacrificed);

        // ── Statistics ──
        let n = vitalities.len().max(1) as f32;
        let mean_v: f32 = vitalities.iter().sum::<f32>() / n;
        let var_v: f32 = vitalities.iter()
            .map(|v| (v - mean_v).powi(2))
            .sum::<f32>() / n;
        let mean_e = energy_sum / total_before.max(1) as f32;

        // TGC — uses ACTUAL nodes_created
        let tgc = self.tgc_tracker.update(nodes_created, self.graph.len(), mean_v);

        // Elite Drift
        let current_centroid = self.compute_elite_centroid();
        let elite_drift = self.initial_centroid.distance(&current_centroid);

        CycleTelemetry {
            cycle: cycle_id,
            total_nodes: self.graph.len(),
            sacrificed,
            nodes_created,
            signal_killed_total: self.signal_killed,
            noise_killed_total: self.noise_killed,
            mean_vitality: mean_v,
            variance_vitality: var_v,
            mean_energy: mean_e,
            tgc,
            elite_drift,
        }
    }
}

// ──────────────────────────────────────────────────
//  Run one experiment
// ──────────────────────────────────────────────────

fn run_experiment(mode: GenMode) -> Vec<CycleTelemetry> {
    let label = mode.label();
    let csv_path = format!("telemetry_{}.csv", label);

    let mut sim = ForgettingSimulator::new(mode);
    sim.seed_graph();

    // Capture TGC baseline at cycle 0 BEFORE any purge.
    // Baseline = hypothetical TGC if we had 100% generation ratio on first tick.
    // We use 1.0 as a normalized reference.
    sim.tgc_tracker.set_baseline(1.0);

    let mut file = OpenOptions::new()
        .write(true).create(true).truncate(true)
        .open(&csv_path).unwrap();
    writeln!(file,
        "cycle,total_nodes,sacrificed,nodes_created,signal_killed,noise_killed,mean_vitality,variance_vitality,mean_energy,tgc,elite_drift"
    ).unwrap();

    let mut results = Vec::with_capacity(500);
    let total_cycles = 500;

    println!("  [{}] Rodando {} ciclos...", label, total_cycles);

    for i in 1..=total_cycles {
        let t = sim.run_cycle(i);

        writeln!(file, "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            t.cycle, t.total_nodes, t.sacrificed, t.nodes_created,
            t.signal_killed_total, t.noise_killed_total,
            t.mean_vitality, t.variance_vitality, t.mean_energy,
            t.tgc, t.elite_drift,
        ).unwrap();

        if i <= 3 || i % 100 == 0 || i == total_cycles {
            println!(
                "    Ciclo {:03} | N={:5} | Del={:3} Gen={:3} | V={:.3} Var={:.4} | E={:.3} | TGC={:.4} | Drift={:.4} | FP={}",
                i, t.total_nodes, t.sacrificed, t.nodes_created,
                t.mean_vitality, t.variance_vitality,
                t.mean_energy, t.tgc, t.elite_drift,
                t.signal_killed_total,
            );
        }

        let collapse = t.total_nodes < 100;
        results.push(t);

        if collapse {
            println!("    [COLAPSO MINIMALISTA] ciclo {}", i);
            break;
        }
    }

    println!("  [{}] CSV: {}", label, csv_path);
    println!("  [{}] Sobreviventes: {} | FP: {} | Ruido morto: {}",
        label,
        results.last().map_or(0, |r| r.total_nodes),
        sim.signal_killed,
        sim.noise_killed,
    );
    println!();

    results
}

// ──────────────────────────────────────────────────
//  Raw numbers: last 100 cycles
// ──────────────────────────────────────────────────

fn print_raw_numbers(label: &str, results: &[CycleTelemetry]) {
    let last_100: Vec<&CycleTelemetry> = if results.len() > 100 {
        results[results.len()-100..].iter().collect()
    } else {
        results.iter().collect()
    };

    let n = last_100.len() as f32;
    if n == 0.0 { return; }

    let avg_tgc: f32 = last_100.iter().map(|t| t.tgc).sum::<f32>() / n;
    let avg_var_v: f32 = last_100.iter().map(|t| t.variance_vitality).sum::<f32>() / n;
    let avg_drift: f32 = last_100.iter().map(|t| t.elite_drift).sum::<f32>() / n;
    let avg_sacr: f32 = last_100.iter().map(|t| t.sacrificed as f32).sum::<f32>() / n;
    let avg_created: f32 = last_100.iter().map(|t| t.nodes_created as f32).sum::<f32>() / n;
    let avg_mean_v: f32 = last_100.iter().map(|t| t.mean_vitality).sum::<f32>() / n;
    let avg_mean_e: f32 = last_100.iter().map(|t| t.mean_energy).sum::<f32>() / n;
    let avg_nodes: f32 = last_100.iter().map(|t| t.total_nodes as f32).sum::<f32>() / n;
    let total_fp = last_100.last().map_or(0, |t| t.signal_killed_total);
    let total_noise = last_100.last().map_or(0, |t| t.noise_killed_total);

    // Min/Max TGC
    let min_tgc = last_100.iter().map(|t| t.tgc).fold(f32::INFINITY, f32::min);
    let max_tgc = last_100.iter().map(|t| t.tgc).fold(f32::NEG_INFINITY, f32::max);

    // Min/Max Var(V)
    let min_var = last_100.iter().map(|t| t.variance_vitality).fold(f32::INFINITY, f32::min);
    let max_var = last_100.iter().map(|t| t.variance_vitality).fold(f32::NEG_INFINITY, f32::max);

    println!("  ┌─── {} ─── ULTIMOS {} CICLOS ───┐", label, last_100.len());
    println!("  │ avg_nodes       = {:.1}", avg_nodes);
    println!("  │ avg_sacrificed  = {:.2}", avg_sacr);
    println!("  │ avg_created     = {:.2}", avg_created);
    println!("  │ avg_mean_V      = {:.4}", avg_mean_v);
    println!("  │ avg_Var(V)      = {:.6}  [min={:.6} max={:.6}]", avg_var_v, min_var, max_var);
    println!("  │ avg_mean_E      = {:.4}", avg_mean_e);
    println!("  │ avg_TGC         = {:.6}  [min={:.6} max={:.6}]", avg_tgc, min_tgc, max_tgc);
    println!("  │ avg_elite_drift = {:.6}", avg_drift);
    println!("  │ total_FP        = {}", total_fp);
    println!("  │ total_noise_del = {}", total_noise);
    println!("  └────────────────────────────────────────┘");
}

// ──────────────────────────────────────────────────
//  Main
// ──────────────────────────────────────────────────

fn main() {
    println!("=================================================================");
    println!("  PROTOCOLO EXPERIMENTAL: TERMODINAMICA DO ESQUECIMENTO ATIVO");
    println!("  NietzscheDB — 3 Variacoes x 500 Ciclos x 5000 Nos");
    println!("=================================================================");
    println!();

    // ── Experiment A: DELETE ONLY (Control) ──
    println!("━━━ EXPERIMENTO A: DELETE ONLY (Controle — catabolismo puro) ━━━");
    let results_a = run_experiment(GenMode::DeleteOnly);

    // ── Experiment B: LOW GENERATION (~30%) ──
    println!("━━━ EXPERIMENTO B: LOW GEN (~30%% dos deletados) ━━━");
    let results_b = run_experiment(GenMode::LowGen);

    // ── Experiment C: MATCHED GENERATION (~100%) ──
    println!("━━━ EXPERIMENTO C: MATCHED GEN (~100%% dos deletados) ━━━");
    let results_c = run_experiment(GenMode::MatchedGen);

    // ── Raw Numbers: Last 100 cycles ──
    println!("=================================================================");
    println!("  DADOS BRUTOS — ULTIMOS 100 CICLOS (SEM INTERPRETACAO)");
    println!("=================================================================");
    println!();
    print_raw_numbers("A_DELETE_ONLY", &results_a);
    println!();
    print_raw_numbers("B_LOW_GEN", &results_b);
    println!();
    print_raw_numbers("C_MATCHED_GEN", &results_c);
    println!();
    println!("=================================================================");
    println!("  Rode: py -3 plot_metabolism.py");
    println!("=================================================================");
}
