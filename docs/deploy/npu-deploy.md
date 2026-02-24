# NietzscheDB — TPU / NPU Deployment Manual

> **Status:** `nietzsche-tpu` crate implemented (Feb 2026). Requires Cloud TPU VM for full activation.
> Without a TPU VM, the crate compiles and falls back to CPU linear scan transparently.

---

## Table of Contents

1. [Hardware Overview](#1-hardware-overview)
2. [Architecture — How TPU Search Works](#2-architecture)
3. [Quick Start — Google Cloud TPU VM](#3-quick-start)
4. [Environment Variables](#4-environment-variables)
5. [Build Matrix](#5-build-matrix)
6. [Dockerfile.tpu](#6-dockerfiletpu)
7. [MHLO MLIR Program](#7-mhlo-mlir-program)
8. [Performance Tuning](#8-performance-tuning)
9. [Troubleshooting](#9-troubleshooting)
10. [NPU Research — Current State (2026)](#10-npu-research)
11. [Roadmap](#11-roadmap)

---

## 1. Hardware Overview

### Google TPU Line (available Feb 2026)

| Version | Name | Status | FP8 TFLOPs/chip | HBM/chip | Best use case |
| --- | --- | --- | --- | --- | --- |
| v5e | TPU v5e | GA | — | 16 GB | Inference, cost-optimised |
| v5p | TPU v5p | GA | — | 95 GB | Large model training |
| v6e | **Trillium** | GA Dec 2024 | — | 32 GB | 2.1× perf/$ over v5e |
| v7 | **Ironwood** | GA Nov 2025 | **4,614** | **192 GB HBM3E** | Inference at scale |

**Ironwood pod scale:** 9,216 chips × 192 GB = **1.77 PB shared HBM**, 9.6 Tb/s ICI, 42.5 Exaflops.

> Anthropic committed to 1 million Ironwood TPUs for Claude inference.

### Why TPU for NietzscheDB?

- EVA uses **Gemini** (Google) for voice/reasoning → same infrastructure as the vector DB
- TPU HBM is co-located with Google's AI stack → lower latency than cross-cloud
- Ironwood's 192 GB HBM per chip fits **~125M vectors at D=1536** per chip
- Pod-scale: 1.77 PB → **~1.15 billion vectors** in shared HBM across a full pod

### What NPU Does Google Offer?

Google does **not** offer a separate "NPU" product. The TPU **is** Google's NPU equivalent
for cloud workloads. On-device (Pixel phones, etc.) Google uses custom cores inside their
Tensor SoC, but these are not customer-accessible in Cloud.

---

## 2. Architecture

### Data Flow (TPU path)

```
knn(query, k)
    │
    ├── n_active < 1,000           → CPU linear scan O(n·d)
    │
    └── n_active ≥ 1,000
            │
            ├── dirty_count / compact_n > 10%   → compact_matrix()
            │       - filters deleted rows
            │       - precomputes m_norms_sq[i] = ||matrix[i]||²
            │       - resets dirty_count = 0
            │
            ├── compiled_n deviates > 20%        → compile_tpu()
            │       - generates MHLO MLIR text (n, d baked in)
            │       - pjrt::LoadedExecutable::builder().build()
            │       - caches executable + compiled_n
            │
            └── tpu_dots(query)
                    ├── upload %query  (D × 4 bytes)         → TPU HBM
                    ├── upload %matrix (n × d × 4 bytes)      → TPU HBM   ← current bottleneck
                    ├── execute MHLO kernel                   → on TPU
                    └── download dots[i] (n × 4 bytes)        ← CPU
                            │
                            └── CPU: dist[i] = sqrt(q_norm² − 2·dots[i] + m_norms²[i])
                                    → sort, truncate top-k
                                    → map compact_row → UUID
```

### PJRT Plugin Chain

```
nietzsche-tpu (Rust)
    └── pjrt crate (v0.2.0)
            └── pjrt::plugin(PJRT_PLUGIN_PATH).load()
                    └── dlopen("libtpu.so")
                            └── GetPjrtApi() → PJRT_Api C struct
                                    └── PJRT C API (same ABI as JAX/PyTorch-XLA)
                                            └── XLA compiler → TPU hardware
```

The `libtpu.so` library is **pre-installed on every Cloud TPU VM** at `/lib/libtpu.so`.
It is the same binary that Google's JAX uses internally.

---

## 3. Quick Start

### Step 1 — Provision a Cloud TPU VM

```bash
# Trillium (v6e) — best price/performance
gcloud compute tpus tpu-vm create nietzsche-tpu \
  --zone=us-central2-b \
  --accelerator-type=v6e-1 \
  --version=tpu-ubuntu2204-base

# OR Ironwood (v7) — maximum scale
gcloud compute tpus tpu-vm create nietzsche-tpu \
  --zone=us-central2-b \
  --accelerator-type=v7-1 \
  --version=tpu-ubuntu2204-base
```

### Step 2 — SSH and verify libtpu

```bash
gcloud compute tpus tpu-vm ssh nietzsche-tpu --zone=us-central2-b

# Verify libtpu.so is present
ls -lh /lib/libtpu.so
# Expected: -rwxr-xr-x 1 root root ~200MB /lib/libtpu.so
```

### Step 3 — Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
```

### Step 4 — Clone and build NietzscheDB with TPU

```bash
git clone https://github.com/JoseRFJuniorLLMs/NietzscheDB.git
cd NietzscheDB

PJRT_PLUGIN_PATH=/lib/libtpu.so \
cargo build --release --features tpu
```

### Step 5 — Run

```bash
PJRT_PLUGIN_PATH=/lib/libtpu.so \
NIETZSCHE_VECTOR_BACKEND=tpu \
NIETZSCHE_DATA_DIR=/data/nietzsche \
NIETZSCHE_PORT=50051 \
./target/release/nietzsche-server
```

Expected startup log:
```
INFO nietzsche_server: NietzscheDB starting version=0.1.0
INFO nietzsche_tpu: PJRT client initialised platform=TPU plugin=/lib/libtpu.so
INFO nietzsche_server: TPU backend active collection=default dim=1536 backend=TpuVectorStore(PJRT/MHLO)
INFO nietzsche_server: gRPC server listening addr=[::]:50051
```

---

## 4. Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `NIETZSCHE_VECTOR_BACKEND` | `mock` | Set to `tpu` to activate TPU backend |
| `PJRT_PLUGIN_PATH` | *(required for TPU)* | Path to `libtpu.so` or other PJRT plugin |
| `NIETZSCHE_DATA_DIR` | `./data` | RocksDB + WAL data directory |
| `NIETZSCHE_PORT` | `50051` | gRPC listen port |
| `NIETZSCHE_LOG_LEVEL` | `info` | Tracing level (`debug` shows MHLO compile events) |

---

## 5. Build Matrix

| Command | GPU | TPU | CPU fallback |
| --- | --- | --- | --- |
| `cargo build --release` | ✗ | ✗ | ✅ (Mock/HNSW) |
| `cargo build --release --features gpu` | ✅ CAGRA | ✗ | ✅ |
| `cargo build --release --features tpu` | ✗ | ✅ PJRT | ✅ |
| `cargo build --release --features gpu,tpu` | ✅ | ✅ | ✅ |

Both `gpu` and `tpu` features can be compiled simultaneously. Which backend activates
at runtime depends on `NIETZSCHE_VECTOR_BACKEND`.

---

## 6. Dockerfile.tpu

```dockerfile
# syntax=docker/dockerfile:1
# Build: docker build -f Dockerfile.tpu -t nietzsche-server:tpu .
# Run on Cloud TPU VM with libtpu.so bind-mounted from host.

FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    curl build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /build
COPY . .

# libtpu.so is NOT present at build time; the pjrt crate only dlopen()s it at runtime.
RUN cargo build --release --features tpu

# ── Runtime image ────────────────────────────────────────────────────────────
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y libssl3 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/nietzsche-server /usr/local/bin/nietzsche-server

EXPOSE 50051 8080
VOLUME ["/data/nietzsche"]

# libtpu.so is bind-mounted at runtime from the TPU VM host (see docker-compose below)
ENV PJRT_PLUGIN_PATH=/lib/libtpu.so
ENV NIETZSCHE_VECTOR_BACKEND=tpu
ENV NIETZSCHE_DATA_DIR=/data/nietzsche

CMD ["nietzsche-server"]
```

```yaml
# docker-compose.tpu.yml — run on a Cloud TPU VM
services:
  nietzsche-server:
    image: nietzsche-server:tpu
    volumes:
      - /lib/libtpu.so:/lib/libtpu.so:ro   # bind-mount from TPU VM host
      - nietzsche_data:/data/nietzsche
    ports:
      - "50051:50051"
      - "8082:8080"
    restart: unless-stopped

volumes:
  nietzsche_data:
```

---

## 7. MHLO MLIR Program

The TPU kernel is generated at runtime by `mhlo_dot_program(n, d)` in
[crates/nietzsche-tpu/src/lib.rs](../crates/nietzsche-tpu/src/lib.rs).

For `n=10_000`, `d=1536`, the generated program is:

```mlir
module @nietzsche_dot_10000x1536 {
  func.func @main(
    %query:  tensor<1536xf32>,
    %matrix: tensor<10000x1536xf32>
  ) -> tensor<10000xf32> {
    %dots = mhlo.dot_general %matrix, %query,
        contracting_dims = [1] x [0]
        : (tensor<10000x1536xf32>, tensor<1536xf32>) -> tensor<10000xf32>
    return %dots : tensor<10000xf32>
  }
}
```

**What it computes:**

```
output[i] = Σⱼ matrix[i, j] × query[j]   for i in 0..n
```

This is a single `mhlo.dot_general` — the most basic XLA operation, maximally
optimised by the XLA compiler for each TPU generation.

**L2 distance reconstruction (CPU, after TPU returns `dots`):**

```rust
dist[i]² = ||query||² - 2 × dots[i] + ||matrix[i]||²
dist[i]  = sqrt(dist[i]².max(0))
```

`||matrix[i]||²` is precomputed once at `compact_matrix()` time and stored in `m_norms_sq`.

**Recompilation policy:**
- First use: compile for current `(n, d)`
- If `n` grows or shrinks by more than 20% (`RECOMPILE_RATIO`): recompile
- Rebuild (compact_matrix): triggered when `dirty_count / compact_n > 10%` (`REBUILD_DELTA_RATIO`)

---

## 8. Performance Tuning

### Matrix Upload Bottleneck

Currently, on every `knn()` call, the full matrix is uploaded to TPU HBM. This is the
dominant cost for large `n`:

| n vectors | D=1536 | Matrix size | Upload @ 100 GB/s PCIe | TPU compute |
| --- | --- | --- | --- | --- |
| 10K | 1536 | 61 MB | ~0.6 ms | ~0.003 ms |
| 100K | 1536 | 610 MB | ~6 ms | ~0.03 ms |
| 1M | 1536 | 6.1 GB | ~61 ms | ~0.3 ms |

**Optimization (TODO):** Keep the matrix as a persistent `pjrt::Buffer` in TPU HBM
between searches. Only upload the query vector (6 KB) per search. This requires a
reference-based `ExecutionInputs` implementation in `pjrt-rs`, which is being tracked.

### Batch Queries

For EVA's inference workload (multiple embeddings per Gemini response), batch all
queries into a single PJRT call:

```
TODO: implement knn_batch(queries: &[PoincareVector], k: usize) -> Vec<Vec<(Uuid, f64)>>
```

This would change the MLIR signature to:
```mlir
%query_batch: tensor<Bx{d}xf32>   (B = batch size)
→ %result: tensor<Bx{n}xf32>
```

### TPU_THRESHOLD Tuning

Default: `1_000` vectors. Below this, CPU linear scan is faster due to PJRT overhead.
Adjust in [lib.rs](../crates/nietzsche-tpu/src/lib.rs):

```rust
const TPU_THRESHOLD: usize = 1_000;   // lower for faster TPU VMs
```

### REBUILD_DELTA_RATIO

Default: `0.10` (rebuild after 10% mutations). For write-heavy workloads, increase to
reduce rebuild frequency:

```rust
const REBUILD_DELTA_RATIO: f64 = 0.20;   // rebuild after 20% mutations
```

---

## 9. Troubleshooting

### `PJRT_PLUGIN_PATH env var not set`

```
WARN nietzsche_tpu: PJRT init failed — CPU fallback active: PJRT_PLUGIN_PATH env var not set.
```

**Fix:** Set the env var before starting the server:
```bash
export PJRT_PLUGIN_PATH=/lib/libtpu.so
```

### `pjrt::plugin().load() failed`

```
WARN nietzsche_tpu: PJRT init failed — CPU fallback active: pjrt::plugin("/lib/libtpu.so").load() failed
```

**Fix:** Verify `libtpu.so` exists and is readable:
```bash
ls -lh /lib/libtpu.so
ldd /lib/libtpu.so | head -10
```

If you're not on a Cloud TPU VM, `libtpu.so` is not available. The server will run
correctly with CPU fallback.

### `PJRT compile failed (n=X, d=Y)`

The MHLO program failed to compile on the target TPU generation.

**Debug:** Set `NIETZSCHE_LOG_LEVEL=debug` and check for XLA compiler errors.
The MLIR text can be dumped by adding a `tracing::debug!("{mlir}")` before compile.

### `TPU search error, CPU fallback`

A runtime execution error occurred (OOM, shape mismatch, etc.). The server automatically
falls back to CPU for that query. Check logs for the underlying error.

---

## 10. NPU Research — Current State (2026)

This section documents why TPU was chosen over other accelerators for NietzscheDB.

### Google's Accelerator Landscape

| Hardware | Type | Customer-accessible | Rust support | Verdict |
| --- | --- | --- | --- | --- |
| TPU v7 Ironwood | ML accelerator | ✅ Cloud TPU VM | `pjrt` crate (0.2.0) | **Used** |
| TPU v6e Trillium | ML accelerator | ✅ Cloud TPU VM | `pjrt` crate | **Used** |
| Axion CPU | ARM CPU (no NPU) | ✅ C4A VMs | Standard Rust | CPU fallback |
| Titanium | Infrastructure offload | ✗ Internal only | N/A | N/A |
| Argos VPU | Video processing | ✗ Internal only | N/A | N/A |

**There is no Google Cloud NPU separate from the TPU line.** The TPU is Google's NPU.

### Why Not Other NPUs?

| NPU | Rust crate | Status | Decision |
| --- | --- | --- | --- |
| Intel NPU (Core Ultra) | `openvino` 0.9.1 | Model-level only, no raw BLAS | Skip — no vector primitives |
| Apple ANE | `candle-coreml` | macOS only, no ANN search | Skip — wrong platform |
| Qualcomm Hexagon | `ort + qnn` EP | Requires Qualcomm QNN SDK, complex setup | Skip — not GCP |
| Rockchip RKNN | `rknpu2-rs` | Edge/embedded only (RK3588) | Skip — not datacenter |

### PJRT C API — The Right Abstraction

PJRT (Portable JAX Runtime) is the stable C ABI that XLA exposes for all hardware:

```
Python (JAX) ──┐
Go (gomlx)  ──┤── PJRT C API ──── libtpu.so  → Google TPU
Rust (pjrt) ──┤                ── libpjrt_gpu → NVIDIA GPU
C++ (XLA)   ──┘                ── CPU plugin  → CPU
```

The `pjrt` Rust crate wraps this C API. Version 0.2.0 was released February 10, 2026
and is actively maintained at [rai-explorers/pjrt-rs](https://github.com/rai-explorers/pjrt-rs).

---

## 11. Roadmap

| Priority | Item | Notes |
| --- | --- | --- |
| P0 | Test on real TPU VM (v5e or v6e) | Validate MHLO MLIR compiles and runs |
| P0 | Validate `mhlo.dot_general` syntax | May need adjustment vs XLA version |
| P1 | Persistent matrix Buffer on TPU | Eliminate per-search upload bottleneck |
| P1 | Batch query support | `knn_batch()` → `tensor<BxNxf32>` MLIR |
| P2 | StableHLO migration | Replace MHLO with StableHLO (forward-compatible) |
| P2 | Cosine similarity kernel | Normalize vectors before dot product on TPU |
| P2 | INT8 quantized path | 4× throughput via `mhlo.convert` + quantized matmul |
| P3 | Persistent TPU Buffer cache | Keep matrix in HBM across requests |
| P3 | Multi-chip sharding | XLA SPMD for pod-scale datasets |
| P3 | `pjrt-rs` upstream contribution | Reference-based `ExecutionInputs` for buffer reuse |

---

## References

- [pjrt-rs GitHub](https://github.com/rai-explorers/pjrt-rs) — Rust PJRT bindings
- [PJRT C API header](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h) — Stable ABI
- [OpenXLA StableHLO](https://openxla.org/stablehlo) — Next-gen HLO IR
- [MHLO Dialect](https://tensorflow.github.io/mlir-hlo/) — Current MLIR dialect
- [Cloud TPU docs](https://cloud.google.com/tpu/docs) — VM provisioning
- [Ironwood announcement](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/) — TPU v7 blog post
- [gomlx/go-xla](https://github.com/gomlx/gopjrt) — Go PJRT reference implementation (most tested non-Python TPU path)
