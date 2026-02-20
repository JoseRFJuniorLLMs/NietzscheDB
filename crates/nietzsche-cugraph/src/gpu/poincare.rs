//! GPU-accelerated Poincaré ball kNN.
//!
//! Compiles a CUDA kernel at runtime via NVRTC (NVIDIA Runtime Compilation)
//! and runs it on the device via `cudarc`.
//!
//! Formula:  d(u,v) = acosh(1 + 2‖u−v‖² / ((1−‖u‖²)(1−‖v‖²)))
//!
//! Each CUDA thread block computes the Poincaré distance between the query
//! and one database vector, with cooperative reduction over dimensions.

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;
use uuid::Uuid;

use crate::{CuGraphError, CuGraphIndex};

// ── CUDA kernel source ────────────────────────────────────────────────────────

const POINCARE_KNN_KERNEL: &str = r#"
extern "C" __global__ void poincare_brute_knn(
    const float* __restrict__ db,
    const float* __restrict__ query,
    float*       __restrict__ out_dist,
    int N,
    int D
) {
    int i = blockIdx.x;
    if (i >= N) return;

    extern __shared__ double smem[];
    double* sh_diff  = smem;
    double* sh_nu    = smem +   blockDim.x;
    double* sh_nv    = smem + 2*blockDim.x;

    double diff_sq = 0.0, nu_sq = 0.0, nv_sq = 0.0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        double u  = (double)query[d];
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
        // acosh(arg) = ln(arg + sqrt(arg^2 - 1))
        out_dist[i] = (float)(log(arg + sqrt(arg * arg - 1.0)));
    }
}
"#;

// ── Public entry point ────────────────────────────────────────────────────────

/// Compute Poincaré kNN on GPU for a single query vector.
///
/// `db_embeddings[i]` is the stored embedding for `index.csr.node_ids[i]`.
/// Returns up to `k` `(Uuid, distance)` pairs sorted ascending by distance.
pub fn poincare_knn(
    index: &CuGraphIndex,
    query: &[f32],
    k: usize,
    db_embeddings: &[Vec<f32>],
) -> Result<Vec<(Uuid, f64)>, CuGraphError> {
    let n = db_embeddings.len();
    if n == 0 || k == 0 {
        return Ok(vec![]);
    }

    let dim = query.len();
    if dim == 0 {
        return Ok(vec![]);
    }

    // ── Flatten db embeddings into contiguous f32 array ───────────────────────
    let mut flat_db: Vec<f32> = Vec::with_capacity(n * dim);
    for emb in db_embeddings {
        let slice = if emb.len() == dim {
            emb.as_slice()
        } else {
            // Pad or truncate to dim
            flat_db.extend_from_slice(&emb[..emb.len().min(dim)]);
            flat_db.resize(flat_db.len() + dim.saturating_sub(emb.len()), 0.0);
            continue;
        };
        flat_db.extend_from_slice(slice);
    }

    // ── Init CUDA device ──────────────────────────────────────────────────────
    let device = CudaDevice::new(0)
        .map_err(|e| CuGraphError::Cuda(format!("CudaDevice::new: {e}")))?;

    // ── Compile kernel ────────────────────────────────────────────────────────
    let ptx = compile_ptx(POINCARE_KNN_KERNEL)
        .map_err(|e| CuGraphError::KernelCompile(format!("{e}")))?;

    let module = device
        .load_ptx(ptx, "poincare", &["poincare_brute_knn"])
        .map_err(|e| CuGraphError::Cuda(format!("load_ptx: {e}")))?;

    let kernel = module
        .get_function("poincare_brute_knn")
        .map_err(|e| CuGraphError::Cuda(format!("get_function: {e}")))?;

    // ── Upload data ───────────────────────────────────────────────────────────
    let d_db = device
        .htod_sync_copy(&flat_db)
        .map_err(|e| CuGraphError::Cuda(format!("htod db: {e}")))?;

    let d_query = device
        .htod_sync_copy(query)
        .map_err(|e| CuGraphError::Cuda(format!("htod query: {e}")))?;

    let mut d_dist = device
        .alloc_zeros::<f32>(n)
        .map_err(|e| CuGraphError::Cuda(format!("alloc dist: {e}")))?;

    // ── Launch: one block per db vector, 256 threads per block ────────────────
    let threads = 256u32;
    let shared_bytes = (3 * threads as usize * std::mem::size_of::<f64>()) as u32;
    let cfg = LaunchConfig {
        grid_dim:    (n as u32, 1, 1),
        block_dim:   (threads, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    unsafe {
        kernel
            .launch(cfg, (&d_db, &d_query, &mut d_dist, n as i32, dim as i32))
            .map_err(|e| CuGraphError::Cuda(format!("kernel launch: {e}")))?;
    }

    device.synchronize()
        .map_err(|e| CuGraphError::Cuda(format!("sync: {e}")))?;

    // ── Download distances ────────────────────────────────────────────────────
    let distances = device
        .dtoh_sync_copy(&d_dist)
        .map_err(|e| CuGraphError::Cuda(format!("dtoh: {e}")))?;

    // ── Top-k selection ───────────────────────────────────────────────────────
    let mut indexed: Vec<(usize, f32)> = distances
        .into_iter()
        .enumerate()
        .filter(|(_, d)| d.is_finite())
        .collect();

    indexed.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    indexed.truncate(k);

    let result = indexed
        .into_iter()
        .filter_map(|(i, d)| {
            index.csr.node_ids.get(i).map(|&id| (id, d as f64))
        })
        .collect();

    Ok(result)
}
