//! # nietzsche-tpu
//!
//! Google **TPU**-accelerated vector store for NietzscheDB via the **PJRT C API**.
//!
//! Uses `libtpu.so` (pre-installed on Cloud TPU VMs) through the `pjrt` crate to
//! execute batched dot-product computations on TPU hardware (Trillium / Ironwood),
//! with L2 norm correction applied on CPU.
//!
//! ## Features
//! - `tpu` *(default: off)* — enables TPU via PJRT C API. Requires:
//!   - A Cloud TPU VM (v5e, v6e Trillium, or v7 Ironwood), **OR**
//!   - `PJRT_PLUGIN_PATH` pointing to a compatible `libtpu.so` / PJRT plugin
//!
//! ## Build
//! ```bash
//! cargo build -p nietzsche-tpu --features tpu
//! ```
//!
//! ## Drop-in usage
//! ```rust,no_run
//! use nietzsche_tpu::TpuVectorStore;
//! use nietzsche_graph::embedded_vector_store::AnyVectorStore;
//!
//! let tpu = TpuVectorStore::new(1536).expect("PJRT init failed");
//! let vs  = AnyVectorStore::tpu(Box::new(tpu));
//! ```
//!
//! ## Architecture
//! ```text
//! Insert → CPU staging buffer (Vec<f32>)
//!                │
//!                ├── feature `tpu` OFF  → always CPU linear scan (O(n·d))
//!                │
//!                └── feature `tpu` ON
//!                        ├── n < TPU_THRESHOLD  → CPU linear scan (avoids PJRT overhead)
//!                        └── n ≥ TPU_THRESHOLD  → lazy compact + MHLO compile
//!                                                  ├── Upload %query  (D floats)  → TPU
//!                                                  ├── Upload %matrix (N×D floats) → TPU
//!                                                  ├── Execute: dots[i] = dot(matrix[i,:], query)
//!                                                  └── CPU: L2² = q_norm² − 2·dots + m_norms²
//! ```
//!
//! ## MHLO MLIR Program
//! The compiled program computes the dominant **O(N·D)** dot-product batch:
//! ```text
//!   output[i] = Σⱼ matrix[i,j] · query[j]   for i in 0..N
//! ```
//! L2 norm corrections (O(N), precomputed at build time) are applied on CPU to produce
//! squared Euclidean distances. The final `sqrt` is applied only for the top-k results.
//!
//! ## Optimisation roadmap
//! - TODO: Keep `%matrix` as a persistent `pjrt::Buffer` on TPU between searches.
//!   Currently the matrix is re-uploaded on every `knn()` call.  For large N this
//!   dominates latency. Requires a reference-based `ExecutionInputs` impl in pjrt-rs.
//! - TODO: Support batched queries (multiple query vectors per PJRT call).
//! - TODO: Implement cosine similarity path (normalize before dot product).

use std::collections::HashMap;
use std::sync::Mutex;

use uuid::Uuid;

use nietzsche_graph::db::VectorStore;
use nietzsche_graph::error::GraphError;
use nietzsche_graph::model::PoincareVector;

// ── PJRT imports (only when `tpu` feature is enabled) ────────────────────────

#[cfg(feature = "tpu")]
use pjrt::{Client, HostBuffer, LoadedExecutable, Program, ProgramFormat};

// ── Tuning constants ──────────────────────────────────────────────────────────

/// Minimum active vectors before the TPU kernel is used.
/// Below this threshold, CPU linear scan avoids the PJRT / data-transfer overhead.
const TPU_THRESHOLD: usize = 1_000;

/// Fraction of dirty entries (inserts + deletes since last compact) relative to
/// compact_n that triggers a matrix rebuild + TPU recompile.
const REBUILD_DELTA_RATIO: f64 = 0.10;

/// When the compacted N deviates from the compiled N by this fraction, recompile.
/// Avoids recompiling on every single insert while still catching large size changes.
const RECOMPILE_RATIO: f64 = 0.20;

// ── Safety: PJRT wraps raw C pointers ────────────────────────────────────────
//
// All TPU state is serialised through `Mutex<TpuState>`.
// The PJRT client is created once and reused; it must not be shared across
// processes. On Cloud TPU VMs, `libtpu.so` is process-local.
#[cfg(feature = "tpu")]
unsafe impl Send for TpuState {}
#[cfg(feature = "tpu")]
unsafe impl Sync for TpuState {}

// ── MHLO MLIR template ───────────────────────────────────────────────────────

/// Generate MHLO MLIR text for batched dot products.
///
/// The resulting program computes:
/// ```text
///   output[i] = dot(matrix[i, :], query)   for i in 0..n
/// ```
///
/// Inputs (in order):
/// - `%query`  : `tensor<{d}xf32>` — the query vector
/// - `%matrix` : `tensor<{n}x{d}xf32>` — candidate matrix, row-major
///
/// Output:
/// - `tensor<{n}xf32>` — one dot product per row
///
/// The program is compiled once per (n, d) pair and cached in [`TpuState::executable`].
/// Recompilation is triggered by [`RECOMPILE_RATIO`].
fn mhlo_dot_program(n: usize, d: usize) -> String {
    // `{{` / `}}` in format strings produce literal `{` / `}`.
    format!(
        r#"module @nietzsche_dot_{n}x{d} {{
  func.func @main(
    %query:  tensor<{d}xf32>,
    %matrix: tensor<{n}x{d}xf32>
  ) -> tensor<{n}xf32> {{
    %dots = mhlo.dot_general %matrix, %query,
        contracting_dims = [1] x [0]
        : (tensor<{n}x{d}xf32>, tensor<{d}xf32>) -> tensor<{n}xf32>
    return %dots : tensor<{n}xf32>
  }}
}}"#
    )
}

// ── Internal state ────────────────────────────────────────────────────────────

struct TpuState {
    // ── Staging buffer (always present) ──────────────────────────────────────

    /// Staging buffer: row i = Vec<f32> of `dim` coordinates.
    /// Deleted rows are kept (soft-deleted) until the next compact.
    vectors: Vec<Vec<f32>>,

    /// UUID → current row index in `vectors`.
    uuid_to_row: HashMap<Uuid, usize>,

    /// Row index → UUID (for CPU fallback result decoding).
    row_to_uuid: Vec<Uuid>,

    /// Soft-delete bitfield (true = logically deleted, filtered in CPU scan).
    deleted: Vec<bool>,

    /// Count of soft-deleted rows in the staging buffer.
    n_deleted: usize,

    /// Inserts + deletes since last `compact_matrix()`.
    dirty_count: usize,

    // ── Compacted matrix (CPU-side, rebuilt when dirty ratio exceeds threshold) ─

    /// Flat, compacted matrix for TPU upload.
    /// Layout: row-major, `compact_n × dim` elements.
    /// Rebuilt by `compact_matrix()` — excludes all soft-deleted rows.
    matrix_flat: Vec<f32>,

    /// Precomputed squared L2 norms of each compacted row.
    /// `m_norms_sq[i] = Σⱼ matrix[i,j]²`
    /// Avoids recomputing per-search (O(N·D) → O(N) correction on CPU).
    m_norms_sq: Vec<f32>,

    /// Compacted row index → UUID (for TPU result decoding).
    compact_row_to_uuid: Vec<Uuid>,

    /// Number of rows in the compacted matrix.
    compact_n: usize,

    // ── PJRT / TPU state (feature-gated) ─────────────────────────────────────

    #[cfg(feature = "tpu")]
    /// PJRT client — one per process, initialised at construction time.
    /// `None` if PJRT init failed (falls back to CPU scan).
    client: Option<Client>,

    #[cfg(feature = "tpu")]
    /// Compiled MHLO dot program. Tuple of `(executable, N_it_was_compiled_for)`.
    /// Recompiled when `compact_n` deviates from the stored N by [`RECOMPILE_RATIO`].
    executable: Option<(LoadedExecutable, usize)>,
}

impl TpuState {
    fn new() -> Result<Self, String> {
        #[cfg(feature = "tpu")]
        let (client, executable) = {
            match Self::init_pjrt() {
                Ok(c) => (Some(c), None),
                Err(e) => {
                    tracing::warn!("[nietzsche-tpu] PJRT init failed — CPU fallback active: {e}");
                    (None, None)
                }
            }
        };

        Ok(Self {
            vectors: Vec::new(),
            uuid_to_row: HashMap::new(),
            row_to_uuid: Vec::new(),
            deleted: Vec::new(),
            n_deleted: 0,
            dirty_count: 0,
            matrix_flat: Vec::new(),
            m_norms_sq: Vec::new(),
            compact_row_to_uuid: Vec::new(),
            compact_n: 0,
            #[cfg(feature = "tpu")]
            client,
            #[cfg(feature = "tpu")]
            executable,
        })
    }

    /// Load the PJRT plugin from `PJRT_PLUGIN_PATH` and create a PJRT client.
    ///
    /// On Cloud TPU VMs, `libtpu.so` is pre-installed and the path is typically
    /// `/lib/libtpu.so` or discoverable via `PJRT_PLUGIN_PATH`.
    #[cfg(feature = "tpu")]
    fn init_pjrt() -> Result<Client, String> {
        let plugin_path = std::env::var("PJRT_PLUGIN_PATH").map_err(|_| {
            "PJRT_PLUGIN_PATH env var not set. \
             On Cloud TPU VMs this is usually /lib/libtpu.so. \
             Set PJRT_PLUGIN_PATH=/path/to/libtpu.so"
                .to_string()
        })?;

        let api = pjrt::plugin(&plugin_path)
            .load()
            .map_err(|e| format!("pjrt::plugin(\"{plugin_path}\").load() failed: {e:?}"))?;

        let client = Client::builder(&api)
            .build()
            .map_err(|e| format!("PJRT Client::builder().build() failed: {e:?}"))?;

        let platform = client.platform_name().unwrap_or_else(|_| "unknown".to_string());
        tracing::info!(
            platform = %platform,
            plugin   = %plugin_path,
            "[nietzsche-tpu] PJRT client initialised"
        );

        Ok(client)
    }

    fn n_active(&self) -> usize {
        self.vectors.len() - self.n_deleted
    }

    /// True if the compacted matrix is stale enough to warrant a rebuild.
    fn should_rebuild(&self) -> bool {
        let n = self.n_active();
        if n < TPU_THRESHOLD {
            return false;
        }
        if self.compact_n == 0 {
            return true;
        }
        let ratio = self.dirty_count as f64 / self.compact_n.max(1) as f64;
        ratio >= REBUILD_DELTA_RATIO
    }

    /// True if the compiled TPU program's N differs too much from current compact_n.
    #[cfg(feature = "tpu")]
    fn should_recompile(&self) -> bool {
        match &self.executable {
            None => true,
            Some((_, compiled_n)) => {
                let cn = *compiled_n;
                let ratio = (self.compact_n as f64 - cn as f64).abs() / cn.max(1) as f64;
                ratio >= RECOMPILE_RATIO
            }
        }
    }

    /// Compact the staging buffer: filter deleted rows, rebuild flat matrix + norms.
    ///
    /// Updates `matrix_flat`, `m_norms_sq`, `compact_row_to_uuid`, `compact_n`,
    /// and resets `dirty_count`.
    fn compact_matrix(&mut self, dim: usize) {
        let active: Vec<(usize, &Vec<f32>, Uuid)> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.deleted[*i])
            .map(|(i, v)| (i, v, self.row_to_uuid[i]))
            .collect();

        let n = active.len();
        if n == 0 {
            self.reset_empty();
            return;
        }

        let mut flat = vec![0f32; n * dim];
        let mut norms = vec![0f32; n];
        let mut compact_uuids = Vec::with_capacity(n);

        for (new_idx, (_, v, uuid)) in active.iter().enumerate() {
            let row_start = new_idx * dim;
            flat[row_start..row_start + dim].copy_from_slice(v);
            norms[new_idx] = v.iter().map(|x| x * x).sum();
            compact_uuids.push(*uuid);
        }

        self.matrix_flat = flat;
        self.m_norms_sq = norms;
        self.compact_row_to_uuid = compact_uuids;
        self.compact_n = n;
        self.dirty_count = 0;
    }

    /// Compile the MHLO dot program for current `compact_n` and `dim`.
    ///
    /// Replaces `self.executable`. No-op if PJRT client is not available.
    #[cfg(feature = "tpu")]
    fn compile_tpu(&mut self, dim: usize) -> Result<(), String> {
        let client = match &self.client {
            Some(c) => c,
            None => return Err("PJRT client not initialised".to_string()),
        };

        let n = self.compact_n;
        let mlir = mhlo_dot_program(n, dim);
        let program = Program::new(ProgramFormat::MLIR, mlir.as_bytes());

        let exe = LoadedExecutable::builder(client, &program)
            .build()
            .map_err(|e| format!("PJRT compile failed (n={n}, d={dim}): {e:?}"))?;

        tracing::debug!(n, dim, "[nietzsche-tpu] MHLO dot program compiled successfully");
        self.executable = Some((exe, n));
        Ok(())
    }

    /// Execute the TPU dot-product kernel.
    ///
    /// Uploads `%query` and `%matrix` to TPU, executes the compiled MHLO program,
    /// and downloads the result.
    ///
    /// Returns `dots` where `dots[i] = dot(matrix[i,:], query)`.
    ///
    /// # Performance note
    /// Matrix upload is O(N·D) and currently dominates latency for large N.
    /// A future optimisation will keep the matrix as a persistent TPU buffer.
    #[cfg(feature = "tpu")]
    fn tpu_dots(&self, query: &[f32]) -> Result<Vec<f32>, String> {
        let (exe, _) = self
            .executable
            .as_ref()
            .ok_or_else(|| "TPU executable not compiled".to_string())?;

        let client = self
            .client
            .as_ref()
            .ok_or_else(|| "PJRT client not initialised".to_string())?;

        let n = self.compact_n;
        let d = query.len();

        // ── Upload %query: tensor<{d}xf32> ───────────────────────────────────
        let q_host = HostBuffer::from_data(query.to_vec(), Some(vec![d as i64]), None);
        let q_dev = q_host
            .to_sync(client)
            .copy()
            .map_err(|e| format!("query upload to TPU failed: {e:?}"))?;

        // ── Upload %matrix: tensor<{n}x{d}xf32> ──────────────────────────────
        // TODO(perf): cache this as a persistent TPU Buffer between searches.
        let m_host = HostBuffer::from_data(
            self.matrix_flat.clone(),
            Some(vec![n as i64, d as i64]),
            None,
        );
        let m_dev = m_host
            .to_sync(client)
            .copy()
            .map_err(|e| format!("matrix upload to TPU failed: {e:?}"))?;

        // ── Execute: inputs = [%query, %matrix] ──────────────────────────────
        // Argument order matches the MLIR func signature:
        //   func.func @main(%query: tensor<{d}xf32>, %matrix: tensor<{n}x{d}xf32>)
        let results = exe
            .execution(vec![q_dev, m_dev])
            .run_sync()
            .map_err(|e| format!("TPU execution failed: {e:?}"))?;

        // ── Download result: tensor<{n}xf32> → Vec<f32> ──────────────────────
        let out_buf = results
            .into_iter()
            .next()
            .and_then(|replica| replica.into_iter().next())
            .ok_or_else(|| "TPU returned no output buffers".to_string())?;

        let host_out = out_buf
            .to_host_sync(None)
            .map_err(|e| format!("result download from TPU failed: {e:?}"))?;

        let dots = host_out
            .read_f32()
            .map_err(|e| format!("read_f32 on TPU output failed: {e:?}"))?
            .to_vec();

        Ok(dots)
    }

    /// CPU linear scan — O(n·d). Used as fallback when TPU is off or unavailable.
    fn cpu_search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.deleted[*i])
            .map(|(i, v)| {
                let dist: f32 = v
                    .iter()
                    .zip(query)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                (i, dist)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    fn reset_empty(&mut self) {
        self.vectors.clear();
        self.uuid_to_row.clear();
        self.row_to_uuid.clear();
        self.deleted.clear();
        self.n_deleted = 0;
        self.dirty_count = 0;
        self.matrix_flat.clear();
        self.m_norms_sq.clear();
        self.compact_row_to_uuid.clear();
        self.compact_n = 0;
        #[cfg(feature = "tpu")]
        {
            self.executable = None;
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// TPU-accelerated vector store using Google's PJRT C API.
///
/// Implements [`VectorStore`] — drop-in for [`EmbeddedVectorStore`] or [`GpuVectorStore`].
/// Inject into NietzscheDB via [`AnyVectorStore::tpu`].
///
/// ## Computation
/// The expensive dot products are executed as a single MHLO program on TPU:
/// ```text
///   dots[i] = Σⱼ matrix[i,j] · query[j]   (O(N·D), on TPU)
///   dist[i] = sqrt(q_norm² − 2·dots[i] + m_norms²[i])   (O(N), on CPU)
/// ```
///
/// ## Fallback
/// Without the `tpu` feature (or if PJRT init fails), the store silently falls
/// back to CPU linear scan — identical to [`MockVectorStore`] but with staging
/// buffer management for future TPU migration.
pub struct TpuVectorStore {
    dim: usize,
    state: Mutex<TpuState>,
}

impl TpuVectorStore {
    /// Create a new TPU vector store for vectors of the given `dim`ension.
    ///
    /// With `tpu` feature: loads the PJRT plugin from `PJRT_PLUGIN_PATH` and
    /// initialises a PJRT client. If init fails, logs a warning and falls back to CPU.
    ///
    /// Without `tpu` feature: always succeeds, uses CPU linear scan.
    pub fn new(dim: usize) -> Result<Self, String> {
        Ok(Self {
            dim,
            state: Mutex::new(TpuState::new()?),
        })
    }

    /// Pre-build the TPU index: compact the staging buffer and compile the MHLO program.
    ///
    /// Call before a burst of `knn()` searches to warm the TPU program.
    /// Otherwise the build happens lazily on the first qualifying `knn()` call.
    ///
    /// No-op when the `tpu` feature is disabled.
    pub fn build_tpu_index(&self) -> Result<(), String> {
        #[cfg(feature = "tpu")]
        {
            let mut state = self.state.lock().unwrap();
            state.compact_matrix(self.dim);
            if state.compact_n == 0 {
                return Ok(());
            }
            if state.should_recompile() {
                state.compile_tpu(self.dim)?;
            }
            return Ok(());
        }
        #[cfg(not(feature = "tpu"))]
        Ok(())
    }

    /// Number of active (non-deleted) vectors in the staging buffer.
    pub fn len(&self) -> usize {
        self.state.lock().unwrap().n_active()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vector dimension this store was created for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Whether the crate was compiled with TPU / PJRT support.
    pub fn is_tpu_enabled() -> bool {
        cfg!(feature = "tpu")
    }
}

impl VectorStore for TpuVectorStore {
    fn upsert(&mut self, id: Uuid, vector: &PoincareVector) -> Result<(), GraphError> {
        if vector.coords.len() != self.dim {
            return Err(GraphError::Storage(format!(
                "TpuVectorStore: vector dim {} ≠ expected {}",
                vector.coords.len(),
                self.dim
            )));
        }

        let coords: Vec<f32> = vector.coords.clone();
        let mut state = self.state.lock().unwrap();

        // Soft-delete any existing entry for this UUID (avoid duplicates).
        if let Some(&old_row) = state.uuid_to_row.get(&id) {
            if !state.deleted[old_row] {
                state.deleted[old_row] = true;
                state.n_deleted += 1;
                state.dirty_count += 1;
            }
        }

        let new_row = state.vectors.len();
        state.vectors.push(coords);
        state.deleted.push(false);
        state.row_to_uuid.push(id);
        state.uuid_to_row.insert(id, new_row);
        state.dirty_count += 1;

        Ok(())
    }

    fn delete(&mut self, id: Uuid) -> Result<(), GraphError> {
        let mut state = self.state.lock().unwrap();
        if let Some(&row) = state.uuid_to_row.get(&id) {
            if !state.deleted[row] {
                state.deleted[row] = true;
                state.n_deleted += 1;
                state.dirty_count += 1;
            }
            state.uuid_to_row.remove(&id);
        }
        Ok(())
    }

    fn knn(&self, query: &PoincareVector, k: usize) -> Result<Vec<(Uuid, f64)>, GraphError> {
        let query_f32: Vec<f32> = query.coords.clone();
        let mut state = self.state.lock().unwrap();

        // ── Lazy rebuild when dirty ratio exceeds threshold ───────────────────
        if state.should_rebuild() {
            state.compact_matrix(self.dim);

            #[cfg(feature = "tpu")]
            if state.compact_n > 0 && state.should_recompile() {
                if let Err(e) = state.compile_tpu(self.dim) {
                    tracing::warn!("[nietzsche-tpu] MHLO compile failed, CPU fallback: {e}");
                }
            }
        }

        // ── TPU path ──────────────────────────────────────────────────────────
        #[cfg(feature = "tpu")]
        if state.compact_n >= TPU_THRESHOLD && state.executable.is_some() {
            match state.tpu_dots(&query_f32) {
                Ok(dots) => {
                    // L2 distance: dist[i]² = ||q||² − 2·dots[i] + ||m_i||²
                    // Using L2² for ranking (sqrt only on final top-k) would be faster,
                    // but we return actual L2 distances for consistency with other backends.
                    let q_norm_sq: f32 = query_f32.iter().map(|x| x * x).sum();

                    let mut scored: Vec<(usize, f32)> = dots
                        .into_iter()
                        .enumerate()
                        .map(|(i, dot)| {
                            // Clamp to 0.0 to avoid sqrt(negative) from float rounding.
                            let l2_sq = (q_norm_sq - 2.0 * dot + state.m_norms_sq[i]).max(0.0);
                            (i, l2_sq.sqrt())
                        })
                        .collect();

                    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    scored.truncate(k);

                    return Ok(scored
                        .into_iter()
                        .filter_map(|(compact_row, dist)| {
                            state
                                .compact_row_to_uuid
                                .get(compact_row)
                                .map(|&uuid| (uuid, dist as f64))
                        })
                        .collect());
                }
                Err(e) => {
                    tracing::warn!("[nietzsche-tpu] TPU search error, CPU fallback: {e}");
                    // Fall through to CPU scan below.
                }
            }
        }

        // ── CPU fallback (staging buffer, includes all non-deleted rows) ──────
        let raw = state.cpu_search(&query_f32, k);

        Ok(raw
            .into_iter()
            .filter_map(|(staging_row, dist)| {
                state
                    .row_to_uuid
                    .get(staging_row)
                    .map(|&uuid| (uuid, dist as f64))
            })
            .collect())
    }
}
