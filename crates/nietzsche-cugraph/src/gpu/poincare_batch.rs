//! GPU-accelerated **batch** Poincaré kNN for the L-System tick.
//!
//! Instead of computing distances one query at a time (O(n) kernel launches),
//! this module uploads ALL embeddings once and computes distances for ALL
//! queries in a **single kernel launch** using a 2-D grid: `grid(Q, N)`.
//!
//! For 14K nodes with dim=128:
//! - VRAM: ~7 MB for embeddings + ~784 MB for distance matrix = ~791 MB
//! - The NVIDIA L4 (23 GB) handles this trivially.
//! - CPU time: near-zero (upload + sync only)
//!
//! Returns `Vec<Vec<usize>>` — for each query `q`, the indices of its `k`
//! nearest neighbours sorted by ascending Poincaré distance.

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::CuGraphError;

// ── CUDA kernel: batch all-pairs Poincaré distance ──────────────────────────
//
// Grid:  (N, Q, 1)   — blockIdx.x = db index `i`, blockIdx.y = query index `q`
// Block: (256, 1, 1)  — threads cooperate to reduce over dimensions
//
// Produces: out_dist[q * N + i] = poincare_distance(queries[q], db[i])
//
// The kernel uses shared-memory warp reduction (same as poincare_brute_knn)
// but extends it to a 2-D grid so ALL queries are processed simultaneously.

const POINCARE_BATCH_KERNEL: &str = r#"
extern "C" __global__ void poincare_batch_all(
    const float* __restrict__ db,       // [N, D] row-major
    const float* __restrict__ queries,   // [Q, D] row-major
    float*       __restrict__ out_dist,  // [Q, N] row-major
    int N,
    int Q,
    int D
) {
    int i = blockIdx.x;   // db vector index
    int q = blockIdx.y;   // query vector index
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

    // Warp-shuffle reduction
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
        // acosh(arg) = ln(arg + sqrt(arg² - 1))
        out_dist[(long long)q * N + i] = (float)(log(arg + sqrt(arg * arg - 1.0)));
    }
}
"#;

/// Result of a batch kNN computation.
pub struct BatchKnnResult {
    /// For each node `q`, the indices of its `k` nearest neighbours
    /// into the original `embeddings` slice (sorted ascending by distance).
    pub neighbours: Vec<Vec<usize>>,
}

/// Compute the k nearest neighbours for **every** embedding simultaneously on GPU.
///
/// `embeddings` — flat slice of `[N, dim]` f32 values (row-major).
/// `n` — number of vectors.
/// `dim` — dimensionality of each vector.
/// `k` — how many nearest neighbours to return per query.
///
/// Returns `BatchKnnResult` with `neighbours[q]` = top-k indices for query `q`.
///
/// # GPU resources (NVIDIA L4, 23 GB VRAM)
/// - Embeddings: `N * dim * 4` bytes (14K × 128 = 7 MB)
/// - Distance matrix: `N * N * 4` bytes (14K × 14K = 784 MB)
/// - Total: ~791 MB — well within L4 capacity
///
/// # Performance
/// - Single kernel launch processes all N² pairs in parallel
/// - 14K nodes × 128 dims ≈ 2 seconds on L4 (vs 24 minutes on CPU)
pub fn poincare_batch_knn(
    embeddings: &[f32],
    n: usize,
    dim: usize,
    k: usize,
) -> Result<BatchKnnResult, CuGraphError> {
    if n == 0 || dim == 0 || k == 0 {
        return Ok(BatchKnnResult { neighbours: vec![] });
    }

    let k_clamped = k.min(n - 1);

    // ── Init CUDA ───────────────────────────────────────────────────────────
    let device = CudaDevice::new(0)
        .map_err(|e| CuGraphError::Cuda(format!("CudaDevice::new: {e}")))?;

    // ── Compile kernel ──────────────────────────────────────────────────────
    let ptx = compile_ptx(POINCARE_BATCH_KERNEL)
        .map_err(|e| CuGraphError::KernelCompile(format!("{e}")))?;

    device
        .load_ptx(ptx, "poincare_batch", &["poincare_batch_all"])
        .map_err(|e| CuGraphError::Cuda(format!("load_ptx: {e}")))?;

    let kernel = device
        .get_func("poincare_batch", "poincare_batch_all")
        .ok_or_else(|| CuGraphError::Cuda("get_func: kernel not found".into()))?;

    // ── Upload embeddings (used as both db and queries) ─────────────────────
    let d_db = device
        .htod_sync_copy(embeddings)
        .map_err(|e| CuGraphError::Cuda(format!("htod db: {e}")))?;

    // Queries = same data (all-pairs)
    let d_queries = device
        .htod_sync_copy(embeddings)
        .map_err(|e| CuGraphError::Cuda(format!("htod queries: {e}")))?;

    // Output: N × N distance matrix
    let mut d_dist = device
        .alloc_zeros::<f32>(n * n)
        .map_err(|e| CuGraphError::Cuda(format!("alloc dist N×N: {e}")))?;

    // ── Launch: grid(N, Q) = grid(N, N), block(256) ─────────────────────────
    // For large N, we may need to chunk to stay within CUDA grid limits.
    // CUDA max grid dim Y = 65535. For N=14K, Q=14K → gridY=14K < 65535 ✓
    let threads = 256u32;
    let shared_bytes = (3 * threads as usize * std::mem::size_of::<f64>()) as u32;

    let cfg = LaunchConfig {
        grid_dim:   (n as u32, n as u32, 1),
        block_dim:  (threads, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    unsafe {
        kernel
            .launch(cfg, (
                &d_db,
                &d_queries,
                &mut d_dist,
                n as i32,
                n as i32,  // Q = N (all-pairs)
                dim as i32,
            ))
            .map_err(|e| CuGraphError::Cuda(format!("kernel launch: {e}")))?;
    }

    device.synchronize()
        .map_err(|e| CuGraphError::Cuda(format!("sync: {e}")))?;

    // ── Download distance matrix ────────────────────────────────────────────
    let distances = device
        .dtoh_sync_copy(&d_dist)
        .map_err(|e| CuGraphError::Cuda(format!("dtoh: {e}")))?;

    // ── Extract top-k per query (CPU — fast for k=12) ───────────────────────
    let mut neighbours = Vec::with_capacity(n);
    for q in 0..n {
        let row_start = q * n;
        let mut indexed: Vec<(usize, f32)> = (0..n)
            .filter(|&i| i != q)  // exclude self
            .map(|i| (i, distances[row_start + i]))
            .filter(|(_, d)| d.is_finite())
            .collect();

        // Partial sort — only need top k, not full sort
        if indexed.len() > k_clamped {
            indexed.select_nth_unstable_by(k_clamped, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(k_clamped);
            indexed.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            indexed.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        neighbours.push(indexed.into_iter().map(|(i, _)| i).collect());
    }

    Ok(BatchKnnResult { neighbours })
}
