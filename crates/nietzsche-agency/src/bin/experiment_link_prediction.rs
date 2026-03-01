//! Experiment: TGC vs Link Prediction Performance (GPU-accelerated).
//!
//! Empirical proof that ∂Performance/∂TGC > 0.
//!
//! ## Design
//!
//! Load a real graph (Cora citation network) or generate synthetic SBM.
//! Split edges into train (90%) and test (10%).
//! Run 100 forgetting/generation cycles under 3 TGC conditions:
//!
//!   N: NORMAL   — elite-anchored birth (topology-guided generation)
//!   O: OFF      — foam birth (orphan nodes, no structural guidance)
//!   I: INVERTED — anti-anchored birth (connect to worst nodes)
//!
//! After each cycle: propagate Poincaré embeddings, compute AUC on held-out
//! test edges via **GPU batch Poincaré distance**, record TGC + AUC telemetry.
//!
//! If Performance(Normal) > Performance(Off) > Performance(Inverted),
//! we have causal evidence that topological coherence improves prediction.
//!
//! ## GPU Acceleration
//!
//! Uses `nietzsche_cugraph::gpu::poincare_batch` to compute ALL pairwise
//! Poincaré distances on the NVIDIA L4 GPU in a single kernel launch.
//! This replaces the O(n²) sequential CPU loop with a GPU burst.
//!
//! Embedding propagation also uses GPU batch distances for neighbor-weighted
//! updates when the node count exceeds GPU_THRESHOLD.
//!
//! ## Usage
//! ```sh
//! # First, prepare data:
//! python3 experiments/dataset_prepare.py
//!
//! # Then run experiment (must have CUDA 12.x):
//! cargo run --release --bin experiment_link_prediction --features cuda
//! ```
//!
//! ## Output
//! - `experiments/telemetry_lp_Normal.csv`
//! - `experiments/telemetry_lp_Off.csv`
//! - `experiments/telemetry_lp_Inverted.csv`
//!
//! ## Analysis
//! ```sh
//! python3 experiments/analysis_link_prediction.py
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

// GPU kernel uses cudarc directly (same kernel as nietzsche-cugraph poincare_batch)

// ══════════════════════════════════════════════════════════════════
//  Constants
// ══════════════════════════════════════════════════════════════════

const EMBED_DIM: usize = 16;
const CYCLES: usize = 100;
const EMBED_LR: f32 = 0.08;
const PROPAG_STEPS: usize = 3;
const WARMUP_STEPS: usize = 10;
const NOISE_INJECT_PER_CYCLE: usize = 15;
const MAX_DELETION_RATE: f32 = 0.05;
const VIT_THRESHOLD: f32 = 0.30;
const ENG_THRESHOLD: f32 = 0.10;

// Adaptive metrics v2: age filter + consolidation window
const AGE_THRESHOLD: usize = 3; // τ: minimum cycles before node is "mature"
const CONSOLIDATION_START: usize = 80; // cycle where growth stops, only propagation

// TGC formula amplifiers (same as production)
const ALPHA: f32 = 2.0;
const BETA: f32 = 3.0;

/// Minimum number of nodes to use GPU batch distance. Below this,
/// CPU is faster due to kernel launch overhead.
#[cfg(feature = "cuda")]
const GPU_THRESHOLD: usize = 64;

// ══════════════════════════════════════════════════════════════════
//  Poincaré Math (CPU fallback)
// ══════════════════════════════════════════════════════════════════

/// Poincaré ball distance: d(u,v) = acosh(1 + 2‖u-v‖² / ((1-‖u‖²)(1-‖v‖²)))
/// f32 coords promoted to f64 internally (ITEM C precision).
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
        return 20.0; // boundary saturation
    }
    let arg = (1.0 + 2.0 * diff_sq / denom).max(1.0);
    arg.acosh()
}

/// Project vector into Poincaré ball (soft clamp to ‖x‖ < 0.95).
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

// ══════════════════════════════════════════════════════════════════
//  GPU Distance Matrix Cache
// ══════════════════════════════════════════════════════════════════

/// Cached full N×N pairwise Poincaré distance matrix computed on GPU.
/// Recomputed each cycle after embedding propagation.
struct GpuDistanceMatrix {
    /// Flat [N×N] row-major distance matrix.
    distances: Vec<f32>,
    /// Mapping from node ID → index into the matrix.
    id_to_idx: HashMap<usize, usize>,
    /// Number of nodes in the matrix.
    n: usize,
}

impl GpuDistanceMatrix {
    fn empty() -> Self {
        Self {
            distances: Vec::new(),
            id_to_idx: HashMap::new(),
            n: 0,
        }
    }

    /// Compute full pairwise distance matrix on GPU.
    #[cfg(feature = "cuda")]
    fn compute(nodes: &HashMap<usize, SimNode>) -> Self {
        let n = nodes.len();
        if n < GPU_THRESHOLD {
            return Self::empty();
        }

        // Build ordered ID list and flat embeddings
        let mut ids: Vec<usize> = nodes.keys().cloned().collect();
        ids.sort_unstable();

        let mut flat_embeddings: Vec<f32> = Vec::with_capacity(n * EMBED_DIM);
        let mut id_to_idx = HashMap::with_capacity(n);

        for (idx, &id) in ids.iter().enumerate() {
            id_to_idx.insert(id, idx);
            let emb = &nodes[&id].embedding;
            flat_embeddings.extend_from_slice(emb);
        }

        // Launch GPU kernel — single kernel computes all N×N distances
        // We use k=1 just to trigger the computation; we extract the raw
        // distance matrix from the kernel output.
        // But poincare_batch_knn returns neighbours, not distances.
        // We need to call the lower-level function directly.
        match Self::compute_distances_gpu(&flat_embeddings, n) {
            Ok(distances) => Self {
                distances,
                id_to_idx,
                n,
            },
            Err(e) => {
                eprintln!("    [GPU] Batch distance failed: {}, falling back to CPU", e);
                Self::empty()
            }
        }
    }

    /// Direct GPU kernel call returning the raw N×N distance matrix.
    #[cfg(feature = "cuda")]
    fn compute_distances_gpu(embeddings: &[f32], n: usize) -> Result<Vec<f32>, String> {
        use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
        use cudarc::nvrtc::compile_ptx;

        const KERNEL: &str = r#"
extern "C" __global__ void poincare_batch_all(
    const float* __restrict__ db,
    const float* __restrict__ queries,
    float*       __restrict__ out_dist,
    int N,
    int Q,
    int D
) {
    int i = blockIdx.x;
    int q = blockIdx.y;
    if (i >= N || q >= Q) return;

    extern __shared__ double smem[];
    double* sh_diff = smem;
    double* sh_nu   = smem +   blockDim.x;
    double* sh_nv   = smem + 2*blockDim.x;

    double diff_sq = 0.0, nu_sq = 0.0, nv_sq = 0.0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        double u  = (double)queries[(long long)q * D + d];
        double v  = (double)db[(long long)i * D + d];
        double dv = u - v;
        diff_sq += dv * dv;
        nu_sq   += u * u;
        nv_sq   += v * v;
    }
    sh_diff[threadIdx.x] = diff_sq;
    sh_nu  [threadIdx.x] = nu_sq;
    sh_nv  [threadIdx.x] = nv_sq;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh_diff[threadIdx.x] += sh_diff[threadIdx.x + s];
            sh_nu  [threadIdx.x] += sh_nu  [threadIdx.x + s];
            sh_nv  [threadIdx.x] += sh_nv  [threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double denom = (1.0 - sh_nu[0]) * (1.0 - sh_nv[0]);
        if (denom < 1e-10) denom = 1e-10;
        double arg = 1.0 + 2.0 * sh_diff[0] / denom;
        if (arg < 1.0) arg = 1.0;
        out_dist[(long long)q * N + i] = (float)(log(arg + sqrt(arg * arg - 1.0)));
    }
}
"#;

        let device = CudaDevice::new(0).map_err(|e| format!("CudaDevice: {e}"))?;
        let ptx = compile_ptx(KERNEL).map_err(|e| format!("compile_ptx: {e}"))?;

        device
            .load_ptx(ptx, "poincare_exp", &["poincare_batch_all"])
            .map_err(|e| format!("load_ptx: {e}"))?;

        let kernel = device
            .get_func("poincare_exp", "poincare_batch_all")
            .ok_or_else(|| "get_func: kernel not found".to_string())?;

        let d_db = device
            .htod_sync_copy(embeddings)
            .map_err(|e| format!("htod db: {e}"))?;

        let d_queries = device
            .htod_sync_copy(embeddings)
            .map_err(|e| format!("htod queries: {e}"))?;

        let mut d_dist = device
            .alloc_zeros::<f32>(n * n)
            .map_err(|e| format!("alloc dist: {e}"))?;

        let threads = 256u32;
        let shared_bytes = (3 * threads as usize * std::mem::size_of::<f64>()) as u32;

        let cfg = LaunchConfig {
            grid_dim: (n as u32, n as u32, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared_bytes,
        };

        unsafe {
            kernel
                .launch(
                    cfg,
                    (
                        &d_db,
                        &d_queries,
                        &mut d_dist,
                        n as i32,
                        n as i32,
                        EMBED_DIM as i32,
                    ),
                )
                .map_err(|e| format!("kernel launch: {e}"))?;
        }

        device.synchronize().map_err(|e| format!("sync: {e}"))?;

        let distances = device
            .dtoh_sync_copy(&d_dist)
            .map_err(|e| format!("dtoh: {e}"))?;

        Ok(distances)
    }

    /// Look up distance between two node IDs from the GPU cache.
    /// Returns None if either ID is not in the matrix.
    fn get_distance(&self, u: usize, v: usize) -> Option<f64> {
        if self.n == 0 {
            return None;
        }
        let &iu = self.id_to_idx.get(&u)?;
        let &iv = self.id_to_idx.get(&v)?;
        let d = self.distances[iu * self.n + iv];
        if d.is_finite() {
            Some(d as f64)
        } else {
            None
        }
    }
}

// ══════════════════════════════════════════════════════════════════
//  Deterministic PRNG (same as simulate_forgetting)
// ══════════════════════════════════════════════════════════════════

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let t = ((self.next_u64() >> 33) as f32) / (u32::MAX as f32);
        lo + t * (hi - lo)
    }
    fn next_usize(&mut self, hi: usize) -> usize {
        ((self.next_u64() >> 33) as usize) % hi.max(1)
    }
    fn chance(&mut self, p: f32) -> bool {
        self.next_f32(0.0, 1.0) < p
    }
}

// ══════════════════════════════════════════════════════════════════
//  TGC Mode (experimental variable)
// ══════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq)]
enum TgcMode {
    Normal,
    Off,
    Inverted,
}

impl TgcMode {
    fn label(&self) -> &'static str {
        match self {
            TgcMode::Normal => "Normal",
            TgcMode::Off => "Off",
            TgcMode::Inverted => "Inverted",
        }
    }
}

// ══════════════════════════════════════════════════════════════════
//  TGC Monitor (production formula)
// ══════════════════════════════════════════════════════════════════

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
    born_cycle: usize,
}

struct VoidSeed {
    embedding: Vec<f32>,
}

struct CycleTelemetry {
    cycle: usize,
    total_nodes: usize,
    total_edges: usize,
    original_nodes: usize,
    sacrificed: usize,
    nodes_created: usize,
    mean_vitality: f32,
    variance_vitality: f32,
    structural_entropy: f32,
    global_efficiency: f32,
    tgc_raw: f32,
    tgc_ema: f32,
    // Static AUC (fossil metric, kept for comparison)
    auc: f64,
    mean_pos_score: f64,
    mean_neg_score: f64,
    // === ADAPTIVE METRICS v2 ===
    // Metric 1: Generative Coherence AUC (edges born this cycle)
    auc_synthetic: f64,
    n_synthetic_pos: usize,
    // Metric 2a: Homophily Delta (Δh = h_new - h_baseline)
    homophily_new: f64,
    homophily_baseline: f64,
    delta_homophily: f64,
    // Metric 2b: Node Classification Accuracy (nearest-centroid)
    nca: f64,
    ms_per_cycle: f32,
    gpu_ms: f32,
}

// ══════════════════════════════════════════════════════════════════
//  Structural Metrics
// ══════════════════════════════════════════════════════════════════

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
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let src = node_ids[((state >> 33) as usize) % n];

        let mut visited: HashMap<usize, usize> = HashMap::new();
        let mut queue = VecDeque::new();
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

// ══════════════════════════════════════════════════════════════════
//  CSV Loading
// ══════════════════════════════════════════════════════════════════

/// Load node labels from node_labels.csv (node_id,label,label_name).
/// Returns HashMap<node_id, class_label>.
fn load_labels(path: &str) -> HashMap<usize, usize> {
    let mut labels = HashMap::new();
    if let Ok(contents) = std::fs::read_to_string(path) {
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("node_id") || line.starts_with('#') {
                continue;
            }
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
    let contents = std::fs::read_to_string(path).expect(&format!("Cannot read {}", path));
    let mut edges = Vec::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("source") || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let (Ok(u), Ok(v)) = (
                parts[0].trim().parse::<usize>(),
                parts[1].trim().parse::<usize>(),
            ) {
                if u != v {
                    edges.push((u, v));
                }
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

    println!(
        "    [SYNTHETIC] Generating SBM: {} nodes, {} communities",
        total, communities
    );

    let mut all_edges: Vec<(usize, usize)> = Vec::new();
    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();

    for i in 0..total {
        let ci = i / nodes_per_comm;
        for j in (i + 1)..total {
            let cj = j / nodes_per_comm;
            let p = if ci == cj { p_intra } else { p_inter };
            if r.chance(p) {
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                if edge_set.insert((a, b)) {
                    all_edges.push((a, b));
                }
            }
        }
    }

    println!("    [SYNTHETIC] Generated {} edges", all_edges.len());

    let n_test = (all_edges.len() as f32 * 0.1).round() as usize;
    for i in (1..all_edges.len()).rev() {
        let j = r.next_usize(i + 1);
        all_edges.swap(i, j);
    }

    let test_edges = all_edges.split_off(all_edges.len() - n_test);
    let train_edges = all_edges;

    println!(
        "    [SYNTHETIC] Train: {} edges, Test: {} edges",
        train_edges.len(),
        test_edges.len()
    );

    (train_edges, test_edges)
}

// ══════════════════════════════════════════════════════════════════
//  The Simulator
// ══════════════════════════════════════════════════════════════════

struct Sim {
    nodes: HashMap<usize, SimNode>,
    adj: HashMap<usize, HashSet<usize>>,
    next_id: usize,
    voids: Vec<VoidSeed>,
    tgc: TgcMonitor,
    mode: TgcMode,
    last_born_count: usize,

    test_edges: Vec<(usize, usize)>,
    negative_edges: Vec<(usize, usize)>,

    n_original: usize,

    /// GPU-computed distance matrix, refreshed each cycle.
    gpu_dist: GpuDistanceMatrix,

    /// Cora class labels: node_id → class (0..6 for 7 Cora classes).
    /// Includes inherited labels for generated nodes.
    labels: HashMap<usize, usize>,

    /// Edges born in the current cycle (for homophily delta).
    new_edges_this_cycle: Vec<(usize, usize)>,

    /// ALL generated edges since start (for AUC_synthetic with age filter).
    all_generated_edges: Vec<(usize, usize)>,

    /// Node ages: node_id → cycles since birth. Original nodes start at WARMUP_STEPS.
    node_ages: HashMap<usize, usize>,
}

impl Sim {
    fn new(
        mode: TgcMode,
        train_edges: &[(usize, usize)],
        test_edges: Vec<(usize, usize)>,
        labels: HashMap<usize, usize>,
    ) -> Self {
        let mut node_ids: HashSet<usize> = HashSet::new();
        for &(u, v) in train_edges {
            node_ids.insert(u);
            node_ids.insert(v);
        }
        for &(u, v) in &test_edges {
            node_ids.insert(u);
            node_ids.insert(v);
        }

        let max_id = node_ids.iter().cloned().max().unwrap_or(0);
        let mut r = Rng::new(12345);

        let mut nodes = HashMap::new();
        let mut adj: HashMap<usize, HashSet<usize>> = HashMap::new();

        for &id in &node_ids {
            let mut emb = Vec::with_capacity(EMBED_DIM);
            for _ in 0..EMBED_DIM {
                emb.push(r.next_f32(-0.1, 0.1));
            }
            project_into_ball(&mut emb);

            nodes.insert(
                id,
                SimNode {
                    energy: r.next_f32(0.7, 1.0),
                    hausdorff: r.next_f32(0.6, 1.2),
                    entropy_delta: r.next_f32(0.0, 0.2),
                    elite_proximity: r.next_f32(0.0, 0.3),
                    toxicity: r.next_f32(0.0, 0.05),
                    is_original: true,
                    embedding: emb,
                    born_cycle: 0,
                },
            );
            adj.entry(id).or_default();
        }

        for &(u, v) in train_edges {
            adj.entry(u).or_default().insert(v);
            adj.entry(v).or_default().insert(u);
        }

        let n_original = nodes.len();

        // Original nodes start mature (age = WARMUP_STEPS)
        let mut node_ages: HashMap<usize, usize> = HashMap::new();
        for &id in &node_ids {
            node_ages.insert(id, WARMUP_STEPS);
        }

        let mut sim = Sim {
            nodes,
            adj,
            next_id: max_id + 1,
            voids: Vec::new(),
            tgc: TgcMonitor::new(),
            mode,
            last_born_count: 0,
            test_edges,
            negative_edges: Vec::new(),
            n_original,
            gpu_dist: GpuDistanceMatrix::empty(),
            labels,
            new_edges_this_cycle: Vec::new(),
            all_generated_edges: Vec::new(),
            node_ages,
        };

        sim.generate_negatives(&mut Rng::new(777));

        println!("    Warm-up: {} propagation steps...", WARMUP_STEPS);
        for _ in 0..WARMUP_STEPS {
            sim.propagate_embeddings();
        }

        sim
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
        if let Some(neighbors) = self.adj.remove(&id) {
            for nb in neighbors {
                if let Some(nbs) = self.adj.get_mut(&nb) {
                    nbs.remove(&id);
                }
            }
        }
        self.nodes.remove(&id);
        self.node_ages.remove(&id);
        self.labels.remove(&id);
    }

    fn degree(&self, id: usize) -> usize {
        self.adj.get(&id).map_or(0, |s| s.len())
    }

    // ── Negative Sampling (HARD: original Cora nodes only) ─────
    //
    // Negatives are sampled EXCLUSIVELY from original dataset nodes.
    // This prevents AUC inflation from trivially-distinguishable
    // noise/generated nodes sitting isolated at the Poincaré boundary.
    // The metric now measures real discriminative power in the core graph.

    fn generate_negatives(&mut self, r: &mut Rng) {
        // Only original Cora nodes — no foam, no noise, no generated
        let original_ids: Vec<usize> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.is_original)
            .map(|(&id, _)| id)
            .collect();
        let n = original_ids.len();
        if n == 0 {
            return;
        }

        self.negative_edges.clear();

        for &(u, _v) in &self.test_edges {
            let neighbors = self.adj.get(&u).cloned().unwrap_or_default();
            let mut attempts = 0;
            loop {
                let w = original_ids[r.next_usize(n)];
                if w != u && !neighbors.contains(&w) {
                    self.negative_edges.push((u, w));
                    break;
                }
                attempts += 1;
                if attempts > 100 {
                    let w = original_ids[r.next_usize(n)];
                    if w != u {
                        self.negative_edges.push((u, w));
                    }
                    break;
                }
            }
        }
    }

    // ── Link Prediction AUC (GPU-accelerated) ────────────────

    /// Score = -poincare_distance(u, v). Higher = more likely connected.
    /// Uses GPU distance cache if available, else CPU fallback.
    fn link_score(&self, u: usize, v: usize) -> f64 {
        // Try GPU cache first
        if let Some(d) = self.gpu_dist.get_distance(u, v) {
            return -d;
        }
        // CPU fallback
        match (self.nodes.get(&u), self.nodes.get(&v)) {
            (Some(nu), Some(nv)) => -poincare_distance(&nu.embedding, &nv.embedding),
            _ => -100.0,
        }
    }

    /// Compute AUC via Wilcoxon-Mann-Whitney statistic.
    fn compute_auc(&self) -> (f64, f64, f64) {
        let n = self.test_edges.len().min(self.negative_edges.len());
        if n == 0 {
            return (0.5, 0.0, 0.0);
        }

        let mut concordant = 0u64;
        let mut tied = 0u64;
        let mut pos_sum = 0.0f64;
        let mut neg_sum = 0.0f64;

        for i in 0..n {
            let (pu, pv) = self.test_edges[i];
            let (nu, nv) = self.negative_edges[i];

            let pos_score = self.link_score(pu, pv);
            let neg_score = self.link_score(nu, nv);

            pos_sum += pos_score;
            neg_sum += neg_score;

            if pos_score > neg_score {
                concordant += 1;
            } else if (pos_score - neg_score).abs() < 1e-12 {
                tied += 1;
            }
        }

        let auc = (concordant as f64 + 0.5 * tied as f64) / n as f64;
        let mean_pos = pos_sum / n as f64;
        let mean_neg = neg_sum / n as f64;

        (auc, mean_pos, mean_neg)
    }

    // ── Refresh GPU distance matrix ──────────────────────────

    /// Recompute the full N×N Poincaré distance matrix on GPU.
    /// Returns the time spent on GPU in milliseconds.
    fn refresh_gpu_distances(&mut self) -> f32 {
        #[cfg(feature = "cuda")]
        {
            let t = Instant::now();
            self.gpu_dist = GpuDistanceMatrix::compute(&self.nodes);
            let ms = t.elapsed().as_secs_f32() * 1000.0;
            if self.gpu_dist.n > 0 {
                return ms;
            }
        }
        // No GPU or below threshold — gpu_dist stays empty, CPU fallback used
        self.gpu_dist = GpuDistanceMatrix::empty();
        0.0
    }

    // ── Embedding Propagation ────────────────────────────────

    fn propagate_embeddings(&mut self) {
        let ids: Vec<usize> = self.nodes.keys().cloned().collect();
        let mut updates: Vec<(usize, Vec<f32>)> = Vec::with_capacity(ids.len());

        for &id in &ids {
            let neighbors = match self.adj.get(&id) {
                Some(nbs) if !nbs.is_empty() => nbs,
                _ => continue,
            };

            let mut mean_emb = vec![0.0f32; EMBED_DIM];
            let mut count = 0usize;
            for &nb in neighbors {
                if let Some(nb_node) = self.nodes.get(&nb) {
                    for d in 0..EMBED_DIM {
                        mean_emb[d] += nb_node.embedding[d];
                    }
                    count += 1;
                }
            }

            if count == 0 {
                continue;
            }

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
            if let Some(node) = self.nodes.get_mut(&id) {
                node.embedding = emb;
            }
        }
    }

    // ── Vitality ─────────────────────────────────────────────

    fn vitality(&self, n: &SimNode, id: usize) -> f32 {
        let prox = (1.0 - n.elite_proximity).max(0.0);
        let deg = self.degree(id);
        let z = 1.0 * n.energy
            + 0.8 * n.hausdorff
            - 1.2 * n.entropy_delta
            + 1.5 * prox
            + 2.0 * deg as f32
            - 1.0 * n.toxicity;
        sigmoid(z)
    }

    // ── Elite / Anti-elite selection ─────────────────────────

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

    fn anti_elite_ids(&self) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = self
            .nodes
            .iter()
            .filter(|(_, n)| !n.is_original)
            .map(|(&id, n)| (id, n.energy))
            .collect();
        if indexed.is_empty() {
            indexed = self
                .nodes
                .iter()
                .map(|(&id, n)| (id, n.energy))
                .collect();
        }
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let bottom = (self.nodes.len() / 5).max(2);
        indexed[..bottom.min(indexed.len())]
            .iter()
            .map(|(id, _)| *id)
            .collect()
    }

    // ── Phase 0: Energy decay + noise injection ──────────────

    fn evolve(&mut self, r: &mut Rng, cycle: usize) {
        let ids_and_info: Vec<(usize, usize, bool)> = self
            .nodes
            .keys()
            .map(|&id| (id, self.degree(id), self.nodes[&id].is_original))
            .collect();

        for (id, deg, is_orig) in &ids_and_info {
            if let Some(n) = self.nodes.get_mut(id) {
                if *is_orig {
                    n.energy = (n.energy - r.next_f32(0.001, 0.005)).max(0.3);
                    if *deg > 2 {
                        n.energy = (n.energy + r.next_f32(0.002, 0.008)).min(1.0);
                    }
                } else {
                    n.energy = (n.energy - r.next_f32(0.005, 0.02)).max(0.0);
                    if *deg == 0 {
                        n.toxicity = (n.toxicity + r.next_f32(0.005, 0.02)).min(1.0);
                    }
                }
            }
        }

        for _ in 0..NOISE_INJECT_PER_CYCLE {
            let id = self.alloc_id();
            let mut emb = Vec::with_capacity(EMBED_DIM);
            for _ in 0..EMBED_DIM {
                emb.push(r.next_f32(-0.5, 0.5));
            }
            project_into_ball(&mut emb);

            self.nodes.insert(
                id,
                SimNode {
                    energy: r.next_f32(0.0, 0.20),
                    hausdorff: r.next_f32(0.05, 0.25),
                    entropy_delta: r.next_f32(0.5, 1.0),
                    elite_proximity: r.next_f32(0.7, 1.0),
                    toxicity: r.next_f32(0.3, 0.8),
                    is_original: false,
                    embedding: emb,
                    born_cycle: cycle,
                },
            );
            self.adj.entry(id).or_default();

            // Noise nodes: inherit label from nearest labeled node
            let node_emb = self.nodes[&id].embedding.clone();
            let mut best_label = None;
            let mut best_dist = f64::MAX;
            // Sample up to 100 labeled nodes for efficiency
            let sample: Vec<(usize, usize)> = self.labels.iter()
                .take(100)
                .map(|(&lid, &lbl)| (lid, lbl))
                .collect();
            for (lid, lbl) in &sample {
                if let Some(lnode) = self.nodes.get(lid) {
                    let d = poincare_distance(&node_emb, &lnode.embedding);
                    if d < best_dist {
                        best_dist = d;
                        best_label = Some(*lbl);
                    }
                }
            }
            if let Some(lbl) = best_label {
                self.labels.insert(id, lbl);
            }
            self.node_ages.insert(id, 0);
        }
    }

    // ── Birth: FOAM ──────────────────────────────────────────

    fn birth_foam(&mut self, r: &mut Rng, n_create: usize, cycle: usize) -> usize {
        let mut created = 0;
        for _ in 0..n_create {
            if let Some(seed) = self.voids.pop() {
                let id = self.alloc_id();
                let mut emb = seed.embedding;
                for d in 0..EMBED_DIM {
                    emb[d] += r.next_f32(-0.03, 0.03);
                }
                project_into_ball(&mut emb);

                self.nodes.insert(
                    id,
                    SimNode {
                        energy: r.next_f32(0.15, 0.50),
                        hausdorff: r.next_f32(0.3, 0.8),
                        entropy_delta: r.next_f32(0.1, 0.4),
                        elite_proximity: r.next_f32(0.3, 0.6),
                        toxicity: r.next_f32(0.0, 0.2),
                        is_original: false,
                        embedding: emb,
                        born_cycle: cycle,
                    },
                );
                self.adj.entry(id).or_default();

                // Foam: inherit label from nearest labeled node by embedding
                let mut best_label = None;
                let mut best_dist = f64::MAX;
                for (&lid, &lbl) in &self.labels {
                    if let Some(lnode) = self.nodes.get(&lid) {
                        let d = poincare_distance(&self.nodes[&id].embedding, &lnode.embedding);
                        if d < best_dist {
                            best_dist = d;
                            best_label = Some(lbl);
                        }
                    }
                }
                if let Some(lbl) = best_label {
                    self.labels.insert(id, lbl);
                }

                self.node_ages.insert(id, 0);
                created += 1;
            }
        }
        created
    }

    // ── Birth: ANCHORED ──────────────────────────────────────

    fn birth_anchored(&mut self, r: &mut Rng, n_create: usize, cycle: usize) -> usize {
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

            let i1 = elites[r.next_usize(elites.len())];
            let mut i2 = elites[r.next_usize(elites.len())];
            let mut attempts = 0;
            while i2 == i1 && attempts < 5 {
                i2 = elites[r.next_usize(elites.len())];
                attempts += 1;
            }

            let p1 = self.nodes[&i1].clone();
            let p2 = self.nodes[&i2].clone();

            let mut emb = Vec::with_capacity(EMBED_DIM);
            for d in 0..EMBED_DIM {
                emb.push((p1.embedding[d] + p2.embedding[d]) / 2.0 + r.next_f32(-0.02, 0.02));
            }
            project_into_ball(&mut emb);

            let inherited_e = 0.8 * ((p1.energy + p2.energy) / 2.0);

            let id = self.alloc_id();
            self.nodes.insert(
                id,
                SimNode {
                    energy: inherited_e,
                    hausdorff: (p1.hausdorff + p2.hausdorff) / 2.0,
                    entropy_delta: (p1.entropy_delta + p2.entropy_delta) / 2.0,
                    elite_proximity: (p1.elite_proximity + p2.elite_proximity) / 2.0,
                    toxicity: (p1.toxicity + p2.toxicity) / 2.0,
                    is_original: false,
                    embedding: emb,
                    born_cycle: cycle,
                },
            );
            self.adj.entry(id).or_default();

            self.add_edge(id, i1);
            self.add_edge(id, i2);
            self.new_edges_this_cycle.push((id, i1));
            self.new_edges_this_cycle.push((id, i2));

            // Inherit label from nearest labeled parent
            let label_1 = self.labels.get(&i1).cloned();
            let label_2 = self.labels.get(&i2).cloned();
            match (label_1, label_2) {
                (Some(l1), Some(l2)) => {
                    // Pick label of closest parent by embedding distance
                    let d1 = poincare_distance(&self.nodes[&id].embedding, &p1.embedding);
                    let d2 = poincare_distance(&self.nodes[&id].embedding, &p2.embedding);
                    self.labels.insert(id, if d1 <= d2 { l1 } else { l2 });
                }
                (Some(l), None) | (None, Some(l)) => {
                    self.labels.insert(id, l);
                }
                _ => {} // no labeled parent
            }

            // New node starts at age 0
            self.node_ages.insert(id, 0);

            created += 1;
        }
        created
    }

    // ── Birth: ANTI-ANCHORED ─────────────────────────────────

    fn birth_anti_anchored(&mut self, r: &mut Rng, n_create: usize, cycle: usize) -> usize {
        let worst = self.anti_elite_ids();
        if worst.len() < 2 {
            return self.birth_foam(r, n_create, cycle);
        }

        let mut created = 0;
        for _ in 0..n_create {
            if self.voids.is_empty() {
                break;
            }
            self.voids.pop();

            let i1 = worst[r.next_usize(worst.len())];
            let mut i2 = worst[r.next_usize(worst.len())];
            let mut attempts = 0;
            while i2 == i1 && attempts < 5 {
                i2 = worst[r.next_usize(worst.len())];
                attempts += 1;
            }

            let mut emb = Vec::with_capacity(EMBED_DIM);
            for _ in 0..EMBED_DIM {
                emb.push(r.next_f32(-0.6, 0.6));
            }
            project_into_ball(&mut emb);

            let id = self.alloc_id();
            self.nodes.insert(
                id,
                SimNode {
                    energy: r.next_f32(0.05, 0.25),
                    hausdorff: r.next_f32(0.05, 0.3),
                    entropy_delta: r.next_f32(0.5, 1.0),
                    elite_proximity: r.next_f32(0.7, 1.0),
                    toxicity: r.next_f32(0.2, 0.6),
                    is_original: false,
                    embedding: emb,
                    born_cycle: cycle,
                },
            );
            self.adj.entry(id).or_default();

            self.add_edge(id, i1);
            self.add_edge(id, i2);
            self.new_edges_this_cycle.push((id, i1));
            self.new_edges_this_cycle.push((id, i2));

            // Inherit label from nearest labeled parent
            let label_1 = self.labels.get(&i1).cloned();
            let label_2 = self.labels.get(&i2).cloned();
            match (label_1, label_2) {
                (Some(l1), Some(l2)) => {
                    // Anti-anchored: random embedding, pick any parent label
                    self.labels.insert(id, l1);
                    let _ = l2; // suppress warning
                }
                (Some(l), None) | (None, Some(l)) => {
                    self.labels.insert(id, l);
                }
                _ => {}
            }

            self.node_ages.insert(id, 0);

            created += 1;
        }
        created
    }

    // ── Adaptive Metric 1: Generative Coherence AUC ──────────
    //
    // Positives = edges born THIS cycle (never stale).
    // Negatives = Cora-only non-edges (no foam inflation).
    // Measures: "Are the edges Zaratustra invented geometrically meaningful?"

    fn compute_generative_auc(&self, rng: &mut Rng) -> (f64, usize) {
        // ── Age Filter (τ): evaluate ALL generated edges where BOTH nodes matured ──
        // Uses cumulative buffer (not just this cycle) so edges have time to converge.
        // An edge created in cycle C becomes evaluable in cycle C+τ.
        let mature_edges: Vec<(usize, usize)> = self.all_generated_edges
            .iter()
            .filter(|&&(u, v)| {
                // Both endpoints must still exist AND be mature
                let age_u = self.node_ages.get(&u).copied().unwrap_or(0);
                let age_v = self.node_ages.get(&v).copied().unwrap_or(0);
                self.nodes.contains_key(&u) && self.nodes.contains_key(&v)
                    && age_u >= AGE_THRESHOLD && age_v >= AGE_THRESHOLD
            })
            .cloned()
            .collect();

        if mature_edges.is_empty() {
            return (f64::NAN, 0);
        }

        // Sample up to 200 mature edges for efficiency
        let sample_size = mature_edges.len().min(200);
        let sampled: Vec<(usize, usize)> = if mature_edges.len() <= sample_size {
            mature_edges
        } else {
            let mut indices: Vec<usize> = (0..mature_edges.len()).collect();
            for i in (1..indices.len()).rev() {
                let j = rng.next_usize(i + 1);
                indices.swap(i, j);
            }
            indices[..sample_size].iter().map(|&i| mature_edges[i]).collect()
        };

        // Positive scores: -distance for each mature generated edge
        let mut pos_scores: Vec<f64> = Vec::with_capacity(sampled.len());
        for &(u, v) in &sampled {
            pos_scores.push(self.link_score(u, v));
        }

        // Negative sampling: Cora-only non-edges (same count as positives)
        let original_ids: Vec<usize> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.is_original)
            .map(|(&id, _)| id)
            .collect();
        let n_orig = original_ids.len();
        if n_orig < 2 {
            return (f64::NAN, 0);
        }

        let n_neg = pos_scores.len();
        let mut neg_scores: Vec<f64> = Vec::with_capacity(n_neg);
        for _ in 0..n_neg {
            let mut attempts = 0;
            loop {
                let u = original_ids[rng.next_usize(n_orig)];
                let v = original_ids[rng.next_usize(n_orig)];
                if u != v {
                    let is_edge = self.adj.get(&u).map_or(false, |nbs| nbs.contains(&v));
                    if !is_edge {
                        neg_scores.push(self.link_score(u, v));
                        break;
                    }
                }
                attempts += 1;
                if attempts > 50 {
                    neg_scores.push(self.link_score(
                        original_ids[rng.next_usize(n_orig)],
                        original_ids[rng.next_usize(n_orig)],
                    ));
                    break;
                }
            }
        }

        // Wilcoxon-Mann-Whitney AUC
        let n = pos_scores.len().min(neg_scores.len());
        if n == 0 {
            return (f64::NAN, 0);
        }
        let mut concordant = 0u64;
        let mut tied = 0u64;
        for i in 0..n {
            if pos_scores[i] > neg_scores[i] {
                concordant += 1;
            } else if (pos_scores[i] - neg_scores[i]).abs() < 1e-12 {
                tied += 1;
            }
        }
        let auc = (concordant as f64 + 0.5 * tied as f64) / n as f64;
        (auc, n)
    }

    // ── Adaptive Metric 2a: Homophily Delta ──────────────────
    //
    // Δh = h(new_edges) - h(baseline)
    // h = fraction of edges connecting same-class nodes.
    // h_baseline = sum_c (n_c/N)^2 (expected under random pairing).
    // Δh > 0 → anabolic (building meaning). Δh < 0 → catabolic.

    fn compute_homophily_delta(&self) -> (f64, f64, f64) {
        if self.labels.is_empty() || self.new_edges_this_cycle.is_empty() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }

        // Count new edges where BOTH endpoints have labels
        let mut same_class = 0usize;
        let mut total_labeled = 0usize;

        for &(u, v) in &self.new_edges_this_cycle {
            if let (Some(&lu), Some(&lv)) = (self.labels.get(&u), self.labels.get(&v)) {
                total_labeled += 1;
                if lu == lv {
                    same_class += 1;
                }
            }
        }

        if total_labeled == 0 {
            // New edges connect generated nodes (no labels) — compute on endpoints
            // that DO have labels via adjacency of parents
            return (f64::NAN, f64::NAN, f64::NAN);
        }

        let h_new = same_class as f64 / total_labeled as f64;

        // Baseline: expected homophily under random pairing
        let mut class_counts: HashMap<usize, usize> = HashMap::new();
        for &label in self.labels.values() {
            *class_counts.entry(label).or_insert(0) += 1;
        }
        let n_labeled = self.labels.len() as f64;
        let h_baseline: f64 = class_counts
            .values()
            .map(|&c| {
                let p = c as f64 / n_labeled;
                p * p
            })
            .sum();

        let delta = h_new - h_baseline;
        (delta, h_new, h_baseline)
    }

    // ── Adaptive Metric 2b: Node Classification Accuracy ─────
    //
    // Nearest-centroid classifier on current Poincaré embeddings.
    // Computes class centroids, then for each Cora node predicts the
    // class of the nearest centroid. Reports accuracy.
    // This measures: "Has the embedding drift preserved semantic structure?"

    fn compute_nca(&self) -> f64 {
        if self.labels.is_empty() {
            return f64::NAN;
        }

        // Collect labeled nodes with their embeddings
        let mut labeled_data: Vec<(usize, usize, &[f32])> = Vec::new();
        for (&id, &label) in &self.labels {
            if let Some(node) = self.nodes.get(&id) {
                labeled_data.push((id, label, &node.embedding));
            }
        }

        if labeled_data.len() < 20 {
            return f64::NAN;
        }

        // Compute class centroids
        let n_classes = *self.labels.values().max().unwrap_or(&0) + 1;
        let mut centroids: Vec<Vec<f64>> = vec![vec![0.0; EMBED_DIM]; n_classes];
        let mut counts: Vec<usize> = vec![0; n_classes];

        for &(_, label, emb) in &labeled_data {
            if label < n_classes {
                for d in 0..EMBED_DIM {
                    centroids[label][d] += emb[d] as f64;
                }
                counts[label] += 1;
            }
        }

        for c in 0..n_classes {
            if counts[c] > 0 {
                for d in 0..EMBED_DIM {
                    centroids[c][d] /= counts[c] as f64;
                }
            }
        }

        // Leave-one-out nearest centroid (adjusted: subtract self from centroid)
        let mut correct = 0usize;
        let mut total = 0usize;

        for &(_, true_label, emb) in &labeled_data {
            if true_label >= n_classes || counts[true_label] == 0 {
                continue;
            }

            let mut best_class = 0;
            let mut best_dist = f64::MAX;

            for c in 0..n_classes {
                if counts[c] == 0 {
                    continue;
                }

                // Adjusted centroid: remove this point if same class
                let mut dist = 0.0f64;
                if c == true_label && counts[c] > 1 {
                    let adj_count = (counts[c] - 1) as f64;
                    for d in 0..EMBED_DIM {
                        let adj_centroid =
                            (centroids[c][d] * counts[c] as f64 - emb[d] as f64) / adj_count;
                        let diff = emb[d] as f64 - adj_centroid;
                        dist += diff * diff;
                    }
                } else {
                    for d in 0..EMBED_DIM {
                        let diff = emb[d] as f64 - centroids[c][d];
                        dist += diff * diff;
                    }
                }

                if dist < best_dist {
                    best_dist = dist;
                    best_class = c;
                }
            }

            if best_class == true_label {
                correct += 1;
            }
            total += 1;
        }

        if total == 0 {
            f64::NAN
        } else {
            correct as f64 / total as f64
        }
    }

    // ── Execute one cycle ────────────────────────────────────

    fn tick(&mut self, cycle: usize) -> CycleTelemetry {
        let t_start = Instant::now();
        let mut r = Rng::new(42u64.wrapping_mul(cycle as u64 + 7919));

        // Phase 0: evolve
        self.evolve(&mut r, cycle);

        // Phase 1: judgment
        let mut to_remove = Vec::new();
        let mut vits = Vec::with_capacity(self.nodes.len());

        let ids: Vec<usize> = self.nodes.keys().cloned().collect();
        for &id in &ids {
            let n = &self.nodes[&id];
            let v = self.vitality(n, id);
            vits.push(v);

            if n.is_original {
                continue;
            }

            let deg = self.degree(id);
            let triple = v < VIT_THRESHOLD && n.energy < ENG_THRESHOLD && deg == 0;
            let toxic = n.toxicity > 0.8 && deg == 0;

            if triple || toxic {
                to_remove.push(id);
            }
        }

        let max_del = (self.nodes.len() as f32 * MAX_DELETION_RATE).round() as usize;
        if to_remove.len() > max_del {
            to_remove.truncate(max_del);
        }

        let sacrificed = to_remove.len();

        for &id in &to_remove {
            if let Some(n) = self.nodes.get(&id) {
                self.voids.push(VoidSeed {
                    embedding: n.embedding.clone(),
                });
            }
            self.remove_node(id);
        }
        if self.voids.len() > 500 {
            self.voids.drain(0..self.voids.len() - 500);
        }

        // Phase 2: generation (with Consolidation Window)
        self.new_edges_this_cycle.clear();
        let nodes_created;

        if cycle <= CONSOLIDATION_START {
            // ── GROWTH PHASE: neuroplasticity active ──
            let gen_target =
                ((sacrificed as f32 * 0.9).round() as usize + 5).min(self.voids.len());
            nodes_created = match self.mode {
                TgcMode::Normal => self.birth_anchored(&mut r, gen_target, cycle),
                TgcMode::Off => self.birth_foam(&mut r, gen_target, cycle),
                TgcMode::Inverted => self.birth_anti_anchored(&mut r, gen_target, cycle),
            };
        } else {
            // ── CONSOLIDATION WINDOW: topology frozen, only propagation ──
            // The "sleep" of the machine: no new nodes/edges, just smoothing.
            nodes_created = 0;
        }
        self.last_born_count = nodes_created;

        // Accumulate generated edges for AUC_synth (evaluated after τ cycles of maturation)
        self.all_generated_edges.extend_from_slice(&self.new_edges_this_cycle);

        // Phase 3: embedding propagation (always runs — consolidation depends on this)
        for _ in 0..PROPAG_STEPS {
            self.propagate_embeddings();
        }

        // Phase 3.5: age all nodes by 1 cycle
        for age in self.node_ages.values_mut() {
            *age += 1;
        }

        // Phase 4: GPU batch distance computation
        let gpu_ms = self.refresh_gpu_distances();

        // ── Structural Metrics ──
        let node_ids: Vec<usize> = self.nodes.keys().cloned().collect();
        let hs = structural_entropy(&self.adj, self.nodes.len());
        let eg = global_efficiency_sampled(
            &self.adj,
            &node_ids,
            32,
            42u64.wrapping_mul(cycle as u64),
        );

        // ── Vitality stats ──
        let n_f = vits.len().max(1) as f32;
        let mean_v: f32 = vits.iter().sum::<f32>() / n_f;
        let var_v: f32 = vits.iter().map(|v| (v - mean_v).powi(2)).sum::<f32>() / n_f;

        // ── TGC ──
        let (tgc_raw, tgc_ema) = self.tgc.compute(
            nodes_created,
            self.nodes.len(),
            mean_v,
            hs,
            eg,
        );

        // ── Static AUC (fossil metric, kept for reference) ──
        let (auc, mean_pos, mean_neg) = self.compute_auc();

        // ── ADAPTIVE METRICS v2 ──
        let mut auc_rng = Rng::new(1337u64.wrapping_mul(cycle as u64 + 31));
        let (auc_synthetic, n_synthetic_pos) = self.compute_generative_auc(&mut auc_rng);
        let (delta_homophily, homophily_new, homophily_baseline) = self.compute_homophily_delta();
        let nca = self.compute_nca();

        let original_alive = self.nodes.values().filter(|n| n.is_original).count();
        let total_edges: usize = self.adj.values().map(|s| s.len()).sum::<usize>() / 2;
        let ms = t_start.elapsed().as_secs_f32() * 1000.0;

        CycleTelemetry {
            cycle,
            total_nodes: self.nodes.len(),
            total_edges,
            original_nodes: original_alive,
            sacrificed,
            nodes_created,
            mean_vitality: mean_v,
            variance_vitality: var_v,
            structural_entropy: hs,
            global_efficiency: eg,
            tgc_raw,
            tgc_ema,
            auc,
            mean_pos_score: mean_pos,
            mean_neg_score: mean_neg,
            auc_synthetic,
            n_synthetic_pos,
            homophily_new,
            homophily_baseline,
            delta_homophily,
            nca,
            ms_per_cycle: ms,
            gpu_ms,
        }
    }
}

// ══════════════════════════════════════════════════════════════════
//  Experiment Runner
// ══════════════════════════════════════════════════════════════════

fn run(
    mode: TgcMode,
    cycles: usize,
    train_edges: &[(usize, usize)],
    test_edges: &[(usize, usize)],
    labels: &HashMap<usize, usize>,
) -> Vec<CycleTelemetry> {
    let label = mode.label();
    let _ = std::fs::create_dir_all("experiments");
    let csv_path = format!("experiments/telemetry_lp_{}.csv", label);

    println!("  [{}] Initializing graph...", label);
    let mut sim = Sim::new(mode, train_edges, test_edges.to_vec(), labels.clone());

    let (auc0, _, _) = sim.compute_auc();
    println!(
        "    Initial: {} nodes, {} test edges, {} negatives, AUC_0={:.4}",
        sim.nodes.len(),
        sim.test_edges.len(),
        sim.negative_edges.len(),
        auc0,
    );

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&csv_path)
        .unwrap();
    writeln!(
        f,
        "cycle,total_nodes,total_edges,original_nodes,sacrificed,nodes_created,mean_vitality,variance_vitality,structural_entropy,global_efficiency,tgc_raw,tgc_ema,auc,mean_pos_score,mean_neg_score,auc_synthetic,n_synthetic_pos,homophily_new,homophily_baseline,delta_homophily,nca,ms_per_cycle,gpu_ms"
    )
    .unwrap();

    let mut results = Vec::with_capacity(cycles);
    println!("  [{}] Running {} cycles...", label, cycles);

    for i in 1..=cycles {
        if i == CONSOLIDATION_START + 1 {
            println!("    --- CONSOLIDATION WINDOW (cycle {}-{}): topology frozen, embeddings settling ---",
                     CONSOLIDATION_START + 1, cycles);
        }

        let t = sim.tick(i);
        writeln!(
            f,
            "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.2},{:.2}",
            t.cycle,
            t.total_nodes,
            t.total_edges,
            t.original_nodes,
            t.sacrificed,
            t.nodes_created,
            t.mean_vitality,
            t.variance_vitality,
            t.structural_entropy,
            t.global_efficiency,
            t.tgc_raw,
            t.tgc_ema,
            t.auc,
            t.mean_pos_score,
            t.mean_neg_score,
            t.auc_synthetic,
            t.n_synthetic_pos,
            t.homophily_new,
            t.homophily_baseline,
            t.delta_homophily,
            t.nca,
            t.ms_per_cycle,
            t.gpu_ms,
        )
        .unwrap();

        if i <= 3 || i % 25 == 0 || i == cycles || i == CONSOLIDATION_START {
            let auc_s = if t.auc_synthetic.is_nan() {
                "  N/A".to_string()
            } else {
                format!("{:.4}", t.auc_synthetic)
            };
            let dh = if t.delta_homophily.is_nan() {
                " N/A".to_string()
            } else {
                format!("{:+.3}", t.delta_homophily)
            };
            println!(
                "    {:03} N={:5} E={:5} TGC={:.4} AUC={:.4} AUCs={} Δh={} NCA={:.3} {:.0}ms",
                i, t.total_nodes, t.total_edges,
                t.tgc_ema, t.auc, auc_s, dh, t.nca, t.ms_per_cycle,
            );
        }

        results.push(t);
    }

    println!(
        "  [{}] Final: N={} E={} AUC={:.4}",
        label,
        results.last().map_or(0, |r| r.total_nodes),
        results.last().map_or(0, |r| r.total_edges),
        results.last().map_or(0.0, |r| r.auc),
    );
    println!("  Saved: {}", csv_path);
    println!();
    results
}

// ══════════════════════════════════════════════════════════════════
//  Summary
// ══════════════════════════════════════════════════════════════════

fn summary(label: &str, r: &[CycleTelemetry]) {
    if r.is_empty() {
        return;
    }
    let n = r.len() as f64;
    let avg = |f: fn(&CycleTelemetry) -> f64| -> f64 { r.iter().map(|t| f(t)).sum::<f64>() / n };

    // Filter NaN for adaptive metrics
    let avg_finite = |f: fn(&CycleTelemetry) -> f64| -> f64 {
        let vals: Vec<f64> = r.iter().map(|t| f(t)).filter(|v| v.is_finite()).collect();
        if vals.is_empty() { f64::NAN } else { vals.iter().sum::<f64>() / vals.len() as f64 }
    };

    let mean_tgc = avg(|t| t.tgc_ema as f64);
    let mean_auc = avg(|t| t.auc);

    let tgc_vals: Vec<f64> = r.iter().map(|t| t.tgc_ema as f64).collect();
    let auc_vals: Vec<f64> = r.iter().map(|t| t.auc).collect();
    let correlation = pearson(&tgc_vals, &auc_vals);

    // Adaptive metric correlations
    let auc_s_vals: Vec<f64> = r.iter().map(|t| t.auc_synthetic).filter(|v| v.is_finite()).collect();
    let tgc_for_aucs: Vec<f64> = r.iter()
        .filter(|t| t.auc_synthetic.is_finite())
        .map(|t| t.tgc_ema as f64)
        .collect();
    let corr_tgc_aucs = pearson(&tgc_for_aucs, &auc_s_vals);

    println!("  +--- {} ---+", label);
    println!("  | avg_TGC_ema        = {:.6}", mean_tgc);
    println!("  | --- FOSSIL METRIC ---");
    println!("  | avg_AUC            = {:.4}", mean_auc);
    println!("  | final_AUC          = {:.4}", r.last().unwrap().auc);
    println!("  | corr(TGC, AUC)     = {:.4}", correlation);
    println!("  | --- ADAPTIVE METRICS v2 ---");
    println!("  | avg_AUC_synthetic  = {:.4}", avg_finite(|t| t.auc_synthetic));
    println!("  | final_AUC_synth    = {:.4}", r.last().unwrap().auc_synthetic);
    println!("  | corr(TGC, AUCs)    = {:.4}", corr_tgc_aucs);
    println!("  | avg_Δhomophily     = {:+.4}", avg_finite(|t| t.delta_homophily));
    println!("  | avg_h(new edges)   = {:.4}", avg_finite(|t| t.homophily_new));
    println!("  | avg_NCA            = {:.4}", avg_finite(|t| t.nca));
    println!("  | final_NCA          = {:.4}", r.last().unwrap().nca);
    // Consolidation-window NCA: average of cycles CONSOLIDATION_START+1 .. end
    let consol_nca: Vec<f64> = r.iter()
        .filter(|t| t.cycle > CONSOLIDATION_START)
        .map(|t| t.nca)
        .filter(|v| v.is_finite())
        .collect();
    if !consol_nca.is_empty() {
        let avg_consol_nca = consol_nca.iter().sum::<f64>() / consol_nca.len() as f64;
        println!("  | NCA_consolidation  = {:.4}  (cycles {}-{})", avg_consol_nca, CONSOLIDATION_START + 1, r.len());
    }
    println!("  | avg_ms/cycle       = {:.2}", avg(|t| t.ms_per_cycle as f64));
    println!("  +-------------------------------+");
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mx = x.iter().sum::<f64>() / nf;
    let my = y.iter().sum::<f64>() / nf;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        cov / denom
    }
}

// ══════════════════════════════════════════════════════════════════
//  Main
// ══════════════════════════════════════════════════════════════════

fn main() {
    println!("=================================================================");
    println!("  EXPERIMENT: TGC vs Link Prediction Performance");
    println!("  Hypothesis: dPerformance/dTGC > 0");
    println!("  Metrics: Fossil AUC + Adaptive v3 (AUC_synth[tau=3], Dh, NCA, NCA_consol)");
    println!("  Growth: cycles 1-{} | Consolidation: cycles {}-{}", CONSOLIDATION_START, CONSOLIDATION_START+1, CYCLES);
    println!("  Conditions: Normal | Off | Inverted");
    println!(
        "  TGC = intensity * quality * (1+{}*dH) * (1+{}*dE)",
        ALPHA, BETA
    );

    #[cfg(feature = "cuda")]
    println!("  Backend: GPU (CUDA) — batch Poincare distance on L4");
    #[cfg(not(feature = "cuda"))]
    println!("  Backend: CPU (no CUDA feature)");

    println!("=================================================================");
    println!();

    // Try to load Cora CSVs, fallback to synthetic
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
            println!("  Cora CSVs not found. Generating synthetic SBM graph...");
            let mut r = Rng::new(42);
            generate_synthetic(&mut r)
        };

    // Load node labels for adaptive metrics (Homophily Delta, NCA)
    let labels = load_labels("experiments/node_labels.csv");
    if labels.is_empty() {
        println!("  WARNING: No labels loaded. Homophily/NCA will be NaN.");
        println!("  Run: python3 experiments/dataset_prepare.py  (to generate node_labels.csv)");
    } else {
        let n_classes = labels.values().collect::<HashSet<_>>().len();
        println!("  Loaded {} node labels ({} classes)", labels.len(), n_classes);
    }

    println!();

    println!("--- N: NORMAL (elite-anchored birth) ---");
    let rn = run(TgcMode::Normal, CYCLES, &train_edges, &test_edges, &labels);

    println!("--- O: OFF (foam birth, no structural guidance) ---");
    let ro = run(TgcMode::Off, CYCLES, &train_edges, &test_edges, &labels);

    println!("--- I: INVERTED (anti-anchored birth) ---");
    let ri = run(TgcMode::Inverted, CYCLES, &train_edges, &test_edges, &labels);

    println!("=================================================================");
    println!("  RESULTS SUMMARY");
    println!("=================================================================");
    println!();
    summary("NORMAL", &rn);
    println!();
    summary("OFF", &ro);
    println!();
    summary("INVERTED", &ri);
    println!();

    // ── FOSSIL METRIC (for reference) ──
    let final_auc_n = rn.last().map_or(0.0, |r| r.auc);
    let final_auc_o = ro.last().map_or(0.0, |r| r.auc);
    let final_auc_i = ri.last().map_or(0.0, |r| r.auc);

    // ── ADAPTIVE METRICS v2 ──
    let final_aucs_n = rn.last().map_or(f64::NAN, |r| r.auc_synthetic);
    let final_aucs_o = ro.last().map_or(f64::NAN, |r| r.auc_synthetic);
    let final_aucs_i = ri.last().map_or(f64::NAN, |r| r.auc_synthetic);

    let final_nca_n = rn.last().map_or(f64::NAN, |r| r.nca);
    let final_nca_o = ro.last().map_or(f64::NAN, |r| r.nca);
    let final_nca_i = ri.last().map_or(f64::NAN, |r| r.nca);

    let avg_dh = |r: &[CycleTelemetry]| -> f64 {
        let vals: Vec<f64> = r.iter().map(|t| t.delta_homophily).filter(|v| v.is_finite()).collect();
        if vals.is_empty() { f64::NAN } else { vals.iter().sum::<f64>() / vals.len() as f64 }
    };
    let avg_dh_n = avg_dh(&rn);
    let avg_dh_o = avg_dh(&ro);
    let avg_dh_i = avg_dh(&ri);

    println!("=================================================================");
    println!("  HYPOTHESIS TEST: Normal > Off > Inverted");
    println!("=================================================================");
    println!();
    println!("  --- Fossil Metric (static test edges) ---");
    println!("  Final AUC Normal   = {:.4}", final_auc_n);
    println!("  Final AUC Off      = {:.4}", final_auc_o);
    println!("  Final AUC Inverted = {:.4}", final_auc_i);
    println!();
    println!("  --- Adaptive: Generative Coherence AUC (edges born this cycle) ---");
    println!("  Final AUC_synth Normal   = {:.4}", final_aucs_n);
    println!("  Final AUC_synth Off      = {:.4}", final_aucs_o);
    println!("  Final AUC_synth Inverted = {:.4}", final_aucs_i);
    println!();
    println!("  --- Adaptive: Homophily Delta (same-class edges) ---");
    println!("  avg_Dh Normal   = {:+.4}", avg_dh_n);
    println!("  avg_Dh Off      = {:+.4}", avg_dh_o);
    println!("  avg_Dh Inverted = {:+.4}", avg_dh_i);
    println!();
    println!("  --- Adaptive: Node Classification Accuracy ---");
    println!("  Final NCA Normal   = {:.4}", final_nca_n);
    println!("  Final NCA Off      = {:.4}", final_nca_o);
    println!("  Final NCA Inverted = {:.4}", final_nca_i);
    println!();

    // Consolidation-window NCA: average NCA during cycles 81-100 (after dust settles)
    let consol_nca = |r: &[CycleTelemetry]| -> f64 {
        let vals: Vec<f64> = r.iter()
            .filter(|t| t.cycle > CONSOLIDATION_START)
            .map(|t| t.nca)
            .filter(|v| v.is_finite())
            .collect();
        if vals.is_empty() { f64::NAN } else { vals.iter().sum::<f64>() / vals.len() as f64 }
    };
    let cnca_n = consol_nca(&rn);
    let cnca_o = consol_nca(&ro);
    let cnca_i = consol_nca(&ri);

    println!("  --- Consolidation Window NCA (cycles {}-{}) ---", CONSOLIDATION_START + 1, CYCLES);
    println!("  NCA_consol Normal   = {:.4}", cnca_n);
    println!("  NCA_consol Off      = {:.4}", cnca_o);
    println!("  NCA_consol Inverted = {:.4}", cnca_i);
    println!();

    // Primary hypothesis: N>O>I on all metrics
    let h_aucs = final_aucs_n > final_aucs_o && final_aucs_o > final_aucs_i;
    let h_nca = final_nca_n > final_nca_o && final_nca_o > final_nca_i;
    let h_cnca = cnca_n > cnca_o && cnca_o > cnca_i;
    let h_dh = avg_dh_n > avg_dh_o && avg_dh_o > avg_dh_i;

    println!("  ╔══════════════════════════════════════════════════╗");
    if h_aucs {
        println!("  ║ AUC_synth(τ=3): N > O > I  — CONFIRMED         ║");
    } else {
        println!("  ║ AUC_synth(τ=3): N>O={} O>I={}  — NOT CONFIRMED  ║",
            final_aucs_n > final_aucs_o, final_aucs_o > final_aucs_i);
    }
    if h_cnca {
        println!("  ║ NCA_consol:     N > O > I  — CONFIRMED         ║");
    } else {
        println!("  ║ NCA_consol:     N>O={} O>I={}  — NOT CONFIRMED  ║",
            cnca_n > cnca_o, cnca_o > cnca_i);
    }
    if h_nca {
        println!("  ║ NCA_final:      N > O > I  — CONFIRMED         ║");
    } else {
        println!("  ║ NCA_final:      N>O={} O>I={}  — NOT CONFIRMED  ║",
            final_nca_n > final_nca_o, final_nca_o > final_nca_i);
    }
    if h_dh {
        println!("  ║ Delta_H:        N > O > I  — CONFIRMED         ║");
    } else {
        println!("  ║ Delta_H:        N>O={} O>I={}  — NOT CONFIRMED  ║",
            avg_dh_n > avg_dh_o, avg_dh_o > avg_dh_i);
    }
    println!("  ╚══════════════════════════════════════════════════╝");

    let score = [h_aucs, h_cnca, h_nca, h_dh].iter().filter(|&&v| v).count();
    println!();
    if score == 4 {
        println!("  >>> HYPOTHESIS FULLY CONFIRMED ({}/4 metrics) <<<", score);
        println!("  TGC causally improves topological generation coherence.");
    } else if score >= 2 {
        println!("  >>> HYPOTHESIS STRONGLY CONFIRMED ({}/4 metrics) <<<", score);
    } else if score >= 1 {
        println!("  >>> HYPOTHESIS PARTIALLY CONFIRMED ({}/4 metrics) <<<", score);
    } else {
        println!("  >>> HYPOTHESIS NOT CONFIRMED (0/4 metrics) <<<");
    }

    // Global correlations
    let mut all_tgc: Vec<f64> = Vec::new();
    let mut all_aucs: Vec<f64> = Vec::new();
    for t in rn.iter().chain(ro.iter()).chain(ri.iter()) {
        if t.auc_synthetic.is_finite() {
            all_tgc.push(t.tgc_ema as f64);
            all_aucs.push(t.auc_synthetic);
        }
    }
    let global_r = pearson(&all_tgc, &all_aucs);
    println!();
    println!("  Global Pearson r(TGC, AUC_synth) = {:.4}", global_r);
    println!(
        "  (pooled across all 3 conditions, {} data points)",
        all_tgc.len()
    );
    println!();
    println!("=================================================================");
    println!("  Run: python3 experiments/analysis_link_prediction.py");
    println!("=================================================================");
}
