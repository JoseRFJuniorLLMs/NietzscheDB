# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-02-22

### Added — Multi-Manifold Architecture
NietzscheDB is now the world's first **Multi-Manifold Graph Database**, operating across 4 non-Euclidean geometries simultaneously from a single Poincaré storage layer.

*   **Klein Model** (`nietzsche-hyp-ops::klein`): Pathfinding via straight-line geodesics. `to_klein()`, `to_poincare()`, `klein_distance()`, `is_collinear()`, `is_on_shortest_path()`, batch operations. O(1) colinearity checks via determinant instead of expensive trig. 10 tests including 10,000-vector roundtrip with error < 1e-6.
*   **Riemann Sphere** (`nietzsche-hyp-ops::riemann`): Dialectical synthesis and aggregation. `synthesis()`, `synthesis_multi()`, `spherical_midpoint()`, `frechet_mean_sphere()`, exp/log maps. Synthesis produces a point MORE ABSTRACT (closer to center) than either input. 10 tests including 10,000-vector roundtrip.
*   **Minkowski Spacetime** (`nietzsche-hyp-ops::minkowski`): Causal edge classification. `minkowski_interval()`, `classify()` (Timelike/Spacelike/Lightlike), `light_cone_filter()`, `compute_edge_causality()`. ds² < 0 = timelike = causal. 8 tests.
*   **Manifold Normalization** (`nietzsche-hyp-ops::manifold`): Post-projection normalization to prevent floating-point drift. `normalize_poincare()`, `normalize_klein()`, `normalize_sphere()`, `health_check_poincare()`, `safe_klein_roundtrip()`, `safe_sphere_roundtrip()`. Cascaded 10x roundtrip error < 1e-4.
*   **Edge Causality Metadata**: `CausalType` enum (Timelike/Spacelike/Lightlike/Unknown) and `minkowski_interval: f32` on Edge struct. Backward compatible via `#[serde(default)]`.
*   **6 New gRPC RPCs**: `Synthesis`, `SynthesisMulti`, `CausalNeighbors`, `CausalChain`, `KleinPath`, `IsOnShortestPath`.
*   **Go SDK Manifold Methods**: `Synthesis()`, `SynthesisMulti()`, `CausalNeighbors()`, `CausalChain()`, `KleinPath()`, `IsOnShortestPath()` with full type definitions.

### Changed
*   **Category**: NietzscheDB is now a "Multi-Manifold Graph Database" (previously "Temporal Hyperbolic Graph Database").
*   **`nietzsche-hyp-ops`**: Description updated to "Multi-manifold geometry operations (Poincaré · Klein · Riemann · Minkowski)".
*   **RPC count**: 65+ → 71+ RPCs.

## [2.1.0] - 2026-02-19

### Added
*   **NodeMeta Separation (BUG A)**: `Node` is now split into `NodeMeta` (~100 bytes) + `PoincareVector` (embedding), stored in separate RocksDB column families (`nodes` and `embeddings`).
    *   `update_energy()` and `update_hausdorff()` no longer deserialize the embedding — zero embedding I/O for metadata-only operations.
    *   BFS traversal reads only `NodeMeta` per hop (~100 bytes vs ~24KB), yielding 10-25x speedup on large graphs.
    *   Dijkstra splits reads: `get_node_meta()` for energy gate, `get_embedding()` only for distance computation.
    *   `hot_tier` cache stores `NodeMeta` instead of full `Node` (240x less RAM per entry at 3072 dims).
    *   New public API: `get_node_meta()`, `get_embedding()`, `scan_nodes_meta()`, `iter_nodes_meta()`.
    *   `Node` implements `Deref<Target=NodeMeta>` — all existing field access (`node.energy`, `node.id`, etc.) works without code changes.
*   **f32 Embedding Coordinates (ITEM C)**: `PoincareVector.coords` migrated from `Vec<f64>` to `Vec<f32>` (50% memory reduction).
    *   Distance kernel (`poincare_sums`) promotes to f64 internally for numerical stability near the Poincaré boundary.
    *   New helpers: `PoincareVector::from_f64()` (narrowing constructor), `coords_f64()` (widening accessor).
    *   Proto wire format (`repeated double`) unchanged — server converts at the gRPC boundary.
    *   Sleep cycle and L-System math remain in f64 precision — promote/narrow at function boundaries.
*   **Binary Quantization Rejection (ITEM F)**: Formally documented that Binary Quantization (`sign(x)` → 1-bit) is **permanently rejected** for NietzscheDB's multi-manifold embeddings.
    *   `sign(x)` destroys magnitude, which encodes hierarchical position in the Poincaré ball (center = abstract, boundary = specific).
    *   Decision recorded in `lib.rs`, `risco_hiperbolico.md`, and `CLAUDE.md`.

### Changed
*   **RocksDB Column Families**: Expanded from 7 to 8 CFs with the addition of `CF_EMBEDDINGS` for separated embedding storage.
*   **Storage I/O**: `put_node()` now atomically writes to both `CF_NODES` (NodeMeta) and `CF_EMBEDDINGS` (PoincareVector) via `WriteBatch`.

## [2.0.0] - 2026-02-16

### Added
*   **Replication Anti-Entropy**: Implemented catch-up mechanism for follower nodes using logical clocks in the Write-Ahead Log (WAL).
    *   Followers now report their last persisted clock state upon connection.
    *   Leaders replay missing operations from WAL to ensure consistency.
*   **Multi-Tenancy**: Native support for SaaS-style multi-tenancy.
    *   **Namespace Isolation**: Collections are prefixed with `user_id` (e.g., `{user_id}_{collection_name}`).
    *   **Context Propagation**: `x-nietzsche-user-id` header is propagated through HTTP and gRPC layers.
    *   **Billing Foundations**: New `/api/admin/usage` endpoint provides disk and vector usage breakdown per user.
*   **WASM Flexibility**: Completely refactored `nietzsche-wasm` to support dynamic configurations.
    *   Supports multiple dimensions (384, 768, 1024, 1536) and metrics (Euclidean, Cosine).
    *   Automatic index type selection based on initialization parameters.
*   **Persistence Upgrades**:
    *   **Metadata Persistence**: Filters and deleted items are now correctly saved and restored in snapshots.
    *   **Logical Clocks**: WAL entries now include logical timestamps for precise state restoration.

### Changed
*   **Major Version Bump**: All crates updated to v2.0.0.
*   **API Updates**:
    *   `Replicate` gRPC method now accepts `ReplicationRequest` instead of `Empty`.
    *   Collection listing now filters by `user_id` context.

### Fixed
*   **WAL Replay**: Fixed issue where legacy WAL entries (OpCode 2) could cause replay failures; implemented backward compatibility.
*   **Docker Build**: Optimized Docker images with `strip` and LTO for smaller footprint.

## [1.6.0] - 2026-02-15

### Added
*   **Cold Storage Architecture**: Implemented lazy loading mechanism where collections are loads from disk only upon first access, optimizing startup time and resource usage.
*   **Idle Eviction**: Introduced a background monitor (Reaper) that automatically unloads collections inactive for more than 1 hour, freeing up RAM.
*   **Graceful Shutdown**: Implemented `Drop` trait for Collections to ensure immediate cancellation of background tasks (indexing, snapshots) upon deletion or eviction, preventing memory leaks and panics.
*   **Manual Vacuum**: Enhanced `trigger_vacuum` endpoint to explicitly trigger memory cleanup routines.
*   **Index Rebuild**: Added `rebuild_index` API to defragment and optimize collections live without downtime.
*   **Queue Monitoring**: Exposed `indexing_queue` size in collection stats for real-time visibility into ingestion backlog.

### Changed
*   **Async Access**: Refactored `CollectionManager` to use asynchronous retrieval (`get().await`), enabling non-blocking disk I/O for cold collections.
*   **Stability**: Fixed "Snapshot Error" panics caused by orphaned background tasks.


## [1.5.0] - 2026-02-09

### Added
*   **Multi-Manifold Efficiency**: Optimized Poincaré ball model implementation for 64d vectors, achieving 2.47ms p99 latency with significant storage savings (64d vs 1024d is 16x compression).
*   **Benchmarks**: Added comprehensive benchmarking suite comparisons against Milvus, NietzscheDB, and Weaviate.
    *   `run_benchmark_hyperbolic.py`: Specific script for demonstrating Multi-Manifold vs Euclidean efficiency.
    *   `BENCHMARK_RESULTS.md` and `HYPERBOLIC_BENCHMARK_RESULTS.md`: Official performance reports.

### Performance
*   **Instant Startup**: Implemented `mmap` (memory-mapped file I/O) for snapshot loading.
    *   Replaces synchronous read-all-at-once approach.
    *   Added visual progress bar for graph reconstruction.
    *   Significantly reduced memory pressure during startup.
*   **High-Throughput Ingestion**: Replaced bounded channels with **Unbounded Channels** in the ingestion pipeline.
    *   Eliminated backpressure bottlenecks that caused performance degradation after 100k vectors.
    *   Ingestion stability improved to consistent ~156k QPS for 64d vectors.
*   **Zero-Copy Deserialization**: Enhanced `rkyv` usage with `mmap` for true zero-copy snapshot restoration.

### Fixed
*   **Panic in Search**: Resolved `Index out of bounds` panic in `search_layer` caused by empty layers in edge cases.
*   **WASM Compatibility**: Fixed missing `export` and `from_bytes` methods in `nietzsche-wasm` when using `mmap` feature.
*   **Benchmark Script**: Fixed API key authentication issues and Weaviate deprecation warnings in benchmark scripts.

## [1.4.0] - 2026-02-05

### Added
*   **WebAssembly Core**: `nietzsche-core` and indexes now compile to WASM (`wasm32-unknown-unknown`).
*   **Edge Database**: New `nietzsche-wasm` crate for running the database purely in-browser (RamStore backend).
*   **Architecture**:
    *   **RAM Vector Store**: In-memory storage backend for runtime environments without disk access.
    *   **Feature Gating**: Optional `mmap` and `persistence` features for `no_std` / WASM compatibility.

## [1.2.0] - 2026-02-04

### Added
* **Multi-Tenancy (Collections)**: Full support for named collections within a single instance.
    * Each collection has independent dimension, metric (Poincaré/Euclidean), and quantization settings.
    * Persistent metadata storage (`meta.json`) for collection configuration.
    * gRPC APIs: `CreateCollection`, `DeleteCollection`, `ListCollections`, `GetCollectionStats`.
* **Web Dashboard**: Professional React-based management interface.
    * **Authentication**: API key-based access control (default: `I_LOVE_NIETZSCHEDB`).
    * **Collection Management**: Create/delete collections with preset configurations:
        * Hyperbolic: 16D, 32D, 64D, 128D (Poincaré metric)
        * Euclidean: 1024D, 1536D, 2048D (L2 metric)
    * **Poincaré Disk Visualizer**: Interactive canvas-based visualization of hyperbolic vector space.
    * **System Metrics**: Real-time monitoring of collections, vectors, memory, and QPS.
    * **Responsive UI**: Modern design with tab-based navigation.
* **Euclidean Metric**: Added `EuclideanMetric` implementation for standard L2 distance.
* **Extended Dimension Support**: Added support for 16D, 32D, 64D, 128D, 2048D vectors.
* **HTTP API**: RESTful endpoints for dashboard integration:
    * `GET /api/collections` - List all collections with detailed stats (count, dimension, metric)
    * `POST /api/collections` - Create new collection
    * `DELETE /api/collections/{name}` - Delete collection
    * `GET /api/collections/{name}/stats` - Get collection statistics
    * `GET /api/collections/{name}/peek?limit=N` - View recent vectors with metadata
    * `POST /api/collections/{name}/search` - Search vectors via HTTP (top_k configurable)
    * `GET /api/status` - System status and configuration
    * `GET /api/metrics` - Real-time metrics (vectors, collections, RAM, CPU)
    * `GET /api/logs` - Live server logs
* **Data Explorer**: New dashboard page for inspecting raw vector data and testing search queries.
* **Search Playground**: Interactive UI for validating search functionality with custom vectors.
* **Federated Clustering (Beta)**: Initial implementation of distributed state managment.
    *   **Node Identity**: Each node has a persistent `node_id`.
    *   **Cluster Topology**: HTTP API `/api/cluster/status` to view mesh topology.
    *   **Logical Clocks**: Lamport clocks added to replication logs for causal ordering.
* **SDK Ecosystem Expansion**:
    *   **Python SDK**: Complete API coverage including collection management (`create_collection`, `list_collections`).
    *   **TypeScript SDK (Beta)**: Native Node.js client with Promise-based API.
    *   **Rust SDK**: Updated for v1.2.0 with cluster awareness.
* **shadcn/ui Components**: Production-ready UI component library integration.


### Changed
* **Default HTTP Port**: Changed from 3000 to 50050 to avoid conflicts.
* **Collection-Scoped Operations**: All data operations (insert/search/delete) now support `collection` field.
* **Backward Compatibility**: Empty `collection` field defaults to `"default"` collection.

### Fixed
* **Blocking Send Panic**: Wrapped `blocking_send` in `tokio::task::block_in_place` to prevent runtime panics.
* **Collection Metadata**: Proper persistence and loading of collection configuration.

### Security
* **Dashboard Authentication**: API key validation middleware for all HTTP endpoints.
* **SHA-256 Hashing**: Secure API key comparison using cryptographic hashing.

## [1.1.0] - 2026-01-27
### Added
* **Generic Dimensions**: Support for 1024, 1536, and 768 dimensional vectors (previously limited to 8).
* **Runtime Config**: Configuration via `.env` files (Dispatcher pattern) for dimensions and HNSW params.
* **Metric Abstraction**: Generic `Metric` trait for swappable distance formulas.
* **Client-Side Vectorization (Fat Client)**: SDKs now support built-in embedding generation.
    *   **Python SDK**: Support for OpenAI, Cohere, Voyage, Google (Gemini), and local SentenceTransformers (`bge-m3`).
    *   **Rust SDK**: Added `Embedder` trait with implementations for OpenAI, OpenRouter, Cohere, Voyage, and Google (Gemini).

## [1.0.0] - 2026-01-25

### 🚀 Initial Release ("NietzscheDB One")

NietzscheDB v1.0 is the first production-ready release of a native Poincaré-ball vector database (now evolved into the NietzscheDB multi-manifold graph database).

### Features
*   **Core Engine**: Poincaré Ball HNSW implementation (multi-manifold foundation).
*   **Performance**: Sub-millisecond search at 1M+ vector scale.
*   **Storage**: Segmented memory-mapped storage with `ScalarI8` and `Binary` quantization (8x and 32x-64x compression respectively).
*   **Persistence**: Write-Ahead Log (WAL) and Zero-Copy Snapshots (Rkyv).
*   **Concurrency**: Async Write Buffer handling 15k+ inserts/sec.
*   **Monitoring**: Real-time TUI dashboard (ratatui) for QPS and system health.
*   **Deployment**: Docker/Docker-Compose support.
*   **SDKs**: Initial Beta support for Python and Rust.

### Improvements
*   Use `std::simd` (Portable SIMD) for distance calculations on nightly Rust.
*   Dynamic configuration of `ef_search` and `ef_construction` via gRPC.
