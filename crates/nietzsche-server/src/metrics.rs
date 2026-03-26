// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Prometheus-compatible operation metrics for NietzscheDB.
//!
//! Tracks counters and latency sums for all gRPC operations using lock-free
//! `AtomicU64` fields (no mutex contention on the hot path).
//!
//! The `/metrics` dashboard endpoint calls [`OperationMetrics::to_prometheus`]
//! to emit Prometheus text exposition format.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use nietzsche_graph::CollectionManager;
use nietzsche_sensory::SensoryMetrics;

/// Histogram bucket boundaries in microseconds for latency tracking.
/// Buckets: 100us, 500us, 1ms, 5ms, 10ms, 50ms, 100ms, 500ms, 1s, 5s, +Inf
const HISTOGRAM_BOUNDS_US: [u64; 10] = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000];

/// Lock-free histogram with fixed buckets.
pub struct LatencyHistogram {
    /// Counts per bucket: bucket[i] counts values <= HISTOGRAM_BOUNDS_US[i].
    buckets: [AtomicU64; 10],
    /// Count for values > last bucket boundary (+Inf).
    overflow: AtomicU64,
    /// Total count (sum of all buckets + overflow).
    total_count: AtomicU64,
    /// Total sum of latencies in us.
    total_sum: AtomicU64,
}

impl LatencyHistogram {
    pub fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            overflow: AtomicU64::new(0),
            total_count: AtomicU64::new(0),
            total_sum: AtomicU64::new(0),
        }
    }

    /// Record a latency observation in microseconds.
    #[inline]
    pub fn observe(&self, value_us: u64) {
        self.total_count.fetch_add(1, Ordering::Relaxed);
        self.total_sum.fetch_add(value_us, Ordering::Relaxed);
        for (i, &bound) in HISTOGRAM_BOUNDS_US.iter().enumerate() {
            if value_us <= bound {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        self.overflow.fetch_add(1, Ordering::Relaxed);
    }

    /// Format as Prometheus histogram buckets.
    pub fn to_prometheus(&self, name: &str) -> String {
        let mut out = format!(
            "# HELP {name} Latency histogram in microseconds\n\
             # TYPE {name} histogram\n"
        );
        let mut cumulative = 0u64;
        for (i, &bound) in HISTOGRAM_BOUNDS_US.iter().enumerate() {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            out.push_str(&format!("{name}_bucket{{le=\"{bound}\"}} {cumulative}\n"));
        }
        cumulative += self.overflow.load(Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {cumulative}\n"));
        out.push_str(&format!("{name}_sum {}\n", self.total_sum.load(Ordering::Relaxed)));
        out.push_str(&format!("{name}_count {}\n", self.total_count.load(Ordering::Relaxed)));
        out
    }
}

/// Lock-free operation counters for the NietzscheDB gRPC server.
pub struct OperationMetrics {
    // ── Node operations ──
    pub insert_node_count:      AtomicU64,
    pub insert_node_latency_us: AtomicU64,
    pub get_node_count:         AtomicU64,
    pub delete_node_count:      AtomicU64,

    // ── Edge operations ──
    pub insert_edge_count:      AtomicU64,
    pub delete_edge_count:      AtomicU64,

    // ── Batch operations ──
    pub batch_insert_node_count: AtomicU64,
    pub batch_insert_edge_count: AtomicU64,

    // ── Query / Search ──
    pub query_count:            AtomicU64,
    pub query_latency_us:       AtomicU64,
    pub knn_count:              AtomicU64,
    pub knn_latency_us:         AtomicU64,

    // ── Latency histograms (P50/P95/P99 support) ──
    pub knn_histogram:          LatencyHistogram,
    pub query_histogram:        LatencyHistogram,
    pub insert_histogram:       LatencyHistogram,

    // ── Traversal ──
    pub traversal_count:        AtomicU64,

    // ── Lifecycle ──
    pub sleep_count:            AtomicU64,
    pub zaratustra_count:       AtomicU64,

    // ── Errors ──
    pub error_count:            AtomicU64,

    // ── Blocking pool ──
    /// Number of spawn_blocking tasks currently active.
    pub blocking_tasks_active:  AtomicU64,
    /// High-water mark for active blocking tasks.
    pub blocking_tasks_peak:    AtomicU64,

    // ── Startup ──
    pub start_time:             Instant,
}

impl OperationMetrics {
    pub fn new() -> Self {
        Self {
            insert_node_count:       AtomicU64::new(0),
            insert_node_latency_us:  AtomicU64::new(0),
            get_node_count:          AtomicU64::new(0),
            delete_node_count:       AtomicU64::new(0),
            insert_edge_count:       AtomicU64::new(0),
            delete_edge_count:       AtomicU64::new(0),
            batch_insert_node_count: AtomicU64::new(0),
            batch_insert_edge_count: AtomicU64::new(0),
            query_count:             AtomicU64::new(0),
            query_latency_us:        AtomicU64::new(0),
            knn_count:               AtomicU64::new(0),
            knn_latency_us:          AtomicU64::new(0),
            knn_histogram:           LatencyHistogram::new(),
            query_histogram:         LatencyHistogram::new(),
            insert_histogram:        LatencyHistogram::new(),
            traversal_count:         AtomicU64::new(0),
            sleep_count:             AtomicU64::new(0),
            zaratustra_count:        AtomicU64::new(0),
            error_count:             AtomicU64::new(0),
            blocking_tasks_active:   AtomicU64::new(0),
            blocking_tasks_peak:     AtomicU64::new(0),
            start_time:              Instant::now(),
        }
    }

    /// Increment a counter by 1.
    #[inline]
    pub fn inc(&self, counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment a counter and accumulate latency.
    #[inline]
    pub fn record(&self, counter: &AtomicU64, latency: &AtomicU64, elapsed_us: u64) {
        counter.fetch_add(1, Ordering::Relaxed);
        latency.fetch_add(elapsed_us, Ordering::Relaxed);
    }

    /// Track a blocking task entering the pool.  Call when `spawn_blocking` starts.
    #[inline]
    pub fn blocking_task_start(&self) {
        let active = self.blocking_tasks_active.fetch_add(1, Ordering::Relaxed) + 1;
        // Update peak (non-atomic CAS loop, but contention is very low here)
        let mut peak = self.blocking_tasks_peak.load(Ordering::Relaxed);
        while active > peak {
            match self.blocking_tasks_peak.compare_exchange_weak(
                peak, active, Ordering::Relaxed, Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
        if active > 100 {
            tracing::warn!(active, "blocking thread pool pressure: {active} tasks active");
        }
    }

    /// Track a blocking task leaving the pool.
    #[inline]
    pub fn blocking_task_end(&self) {
        self.blocking_tasks_active.fetch_sub(1, Ordering::Relaxed);
    }

    /// Format all metrics as Prometheus text exposition.
    pub fn to_prometheus(&self, cm: &CollectionManager) -> String {
        let infos = cm.list();
        let (node_count, edge_count) = infos
            .iter()
            .fold((0u64, 0u64), |(n, e), i| {
                (n + i.node_count as u64, e + i.edge_count as u64)
            });

        let uptime = self.start_time.elapsed().as_secs();
        let collections = infos.len();

        // Helper to safely compute average
        let avg = |sum: u64, count: u64| -> f64 {
            if count == 0 { 0.0 } else { sum as f64 / count as f64 }
        };

        let insert_n = self.insert_node_count.load(Ordering::Relaxed);
        let insert_n_lat = self.insert_node_latency_us.load(Ordering::Relaxed);
        let query_n = self.query_count.load(Ordering::Relaxed);
        let query_lat = self.query_latency_us.load(Ordering::Relaxed);
        let knn_n = self.knn_count.load(Ordering::Relaxed);
        let knn_lat = self.knn_latency_us.load(Ordering::Relaxed);

        let mut result = format!(
            "\
# HELP nietzsche_node_count Total nodes across all collections
# TYPE nietzsche_node_count gauge
nietzsche_node_count {node_count}
# HELP nietzsche_edge_count Total edges across all collections
# TYPE nietzsche_edge_count gauge
nietzsche_edge_count {edge_count}
# HELP nietzsche_collection_count Active collections
# TYPE nietzsche_collection_count gauge
nietzsche_collection_count {collections}
# HELP nietzsche_uptime_seconds Server uptime in seconds
# TYPE nietzsche_uptime_seconds gauge
nietzsche_uptime_seconds {uptime}
# HELP nietzsche_insert_node_total Total InsertNode operations
# TYPE nietzsche_insert_node_total counter
nietzsche_insert_node_total {insert_n}
# HELP nietzsche_insert_node_latency_avg_us Average InsertNode latency in microseconds
# TYPE nietzsche_insert_node_latency_avg_us gauge
nietzsche_insert_node_latency_avg_us {avg_insert_node}
# HELP nietzsche_get_node_total Total GetNode operations
# TYPE nietzsche_get_node_total counter
nietzsche_get_node_total {get_node}
# HELP nietzsche_delete_node_total Total DeleteNode operations
# TYPE nietzsche_delete_node_total counter
nietzsche_delete_node_total {delete_node}
# HELP nietzsche_insert_edge_total Total InsertEdge operations
# TYPE nietzsche_insert_edge_total counter
nietzsche_insert_edge_total {insert_edge}
# HELP nietzsche_delete_edge_total Total DeleteEdge operations
# TYPE nietzsche_delete_edge_total counter
nietzsche_delete_edge_total {delete_edge}
# HELP nietzsche_batch_insert_node_total Total BatchInsertNodes calls
# TYPE nietzsche_batch_insert_node_total counter
nietzsche_batch_insert_node_total {batch_node}
# HELP nietzsche_batch_insert_edge_total Total BatchInsertEdges calls
# TYPE nietzsche_batch_insert_edge_total counter
nietzsche_batch_insert_edge_total {batch_edge}
# HELP nietzsche_query_total Total NQL Query operations
# TYPE nietzsche_query_total counter
nietzsche_query_total {query_n}
# HELP nietzsche_query_latency_avg_us Average Query latency in microseconds
# TYPE nietzsche_query_latency_avg_us gauge
nietzsche_query_latency_avg_us {avg_query}
# HELP nietzsche_knn_total Total KNN search operations
# TYPE nietzsche_knn_total counter
nietzsche_knn_total {knn_n}
# HELP nietzsche_knn_latency_avg_us Average KNN latency in microseconds
# TYPE nietzsche_knn_latency_avg_us gauge
nietzsche_knn_latency_avg_us {avg_knn}
# HELP nietzsche_traversal_total Total BFS+Dijkstra traversal operations
# TYPE nietzsche_traversal_total counter
nietzsche_traversal_total {traversal}
# HELP nietzsche_sleep_total Total Sleep cycle operations
# TYPE nietzsche_sleep_total counter
nietzsche_sleep_total {sleep}
# HELP nietzsche_zaratustra_total Total Zaratustra cycle operations
# TYPE nietzsche_zaratustra_total counter
nietzsche_zaratustra_total {zaratustra}
# HELP nietzsche_error_total Total gRPC errors
# TYPE nietzsche_error_total counter
nietzsche_error_total {errors}
# HELP nietzsche_sensory_neural_image_total Sensory inserts via ImageNeuralEncoder (ONNX)
# TYPE nietzsche_sensory_neural_image_total counter
nietzsche_sensory_neural_image_total {sensory_neural_image}
# HELP nietzsche_sensory_neural_audio_total Sensory inserts via AudioNeuralEncoder (ONNX)
# TYPE nietzsche_sensory_neural_audio_total counter
nietzsche_sensory_neural_audio_total {sensory_neural_audio}
# HELP nietzsche_sensory_passthrough_total Sensory inserts via exp_map_zero passthrough
# TYPE nietzsche_sensory_passthrough_total counter
nietzsche_sensory_passthrough_total {sensory_passthrough}
# HELP nietzsche_sensory_neural_fallback_total Neural encoding failures that fell back to passthrough
# TYPE nietzsche_sensory_neural_fallback_total counter
nietzsche_sensory_neural_fallback_total {sensory_neural_fallback}
",
            node_count = node_count,
            edge_count = edge_count,
            collections = collections,
            uptime = uptime,
            insert_n = insert_n,
            avg_insert_node = format!("{:.1}", avg(insert_n_lat, insert_n)),
            get_node = self.get_node_count.load(Ordering::Relaxed),
            delete_node = self.delete_node_count.load(Ordering::Relaxed),
            insert_edge = self.insert_edge_count.load(Ordering::Relaxed),
            delete_edge = self.delete_edge_count.load(Ordering::Relaxed),
            batch_node = self.batch_insert_node_count.load(Ordering::Relaxed),
            batch_edge = self.batch_insert_edge_count.load(Ordering::Relaxed),
            query_n = query_n,
            avg_query = format!("{:.1}", avg(query_lat, query_n)),
            knn_n = knn_n,
            avg_knn = format!("{:.1}", avg(knn_lat, knn_n)),
            traversal = self.traversal_count.load(Ordering::Relaxed),
            sleep = self.sleep_count.load(Ordering::Relaxed),
            zaratustra = self.zaratustra_count.load(Ordering::Relaxed),
            errors = self.error_count.load(Ordering::Relaxed),
            sensory_neural_image = SensoryMetrics::neural_image_count(),
            sensory_neural_audio = SensoryMetrics::neural_audio_count(),
            sensory_passthrough = SensoryMetrics::passthrough_count(),
            sensory_neural_fallback = SensoryMetrics::neural_fallback_count(),
        );

        // Blocking thread pool metrics.
        result.push_str(&format!(
            "# HELP nietzsche_blocking_tasks_active Currently active spawn_blocking tasks\n\
             # TYPE nietzsche_blocking_tasks_active gauge\n\
             nietzsche_blocking_tasks_active {}\n\
             # HELP nietzsche_blocking_tasks_peak Peak active spawn_blocking tasks\n\
             # TYPE nietzsche_blocking_tasks_peak gauge\n\
             nietzsche_blocking_tasks_peak {}\n",
            self.blocking_tasks_active.load(Ordering::Relaxed),
            self.blocking_tasks_peak.load(Ordering::Relaxed),
        ));

        // Append latency histograms for P50/P95/P99 support.
        result.push_str(&self.knn_histogram.to_prometheus("nietzsche_knn_latency_us"));
        result.push_str(&self.query_histogram.to_prometheus("nietzsche_query_latency_us"));
        result.push_str(&self.insert_histogram.to_prometheus("nietzsche_insert_latency_us"));

        result
    }
}
