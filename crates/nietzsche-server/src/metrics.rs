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

    // ── Traversal ──
    pub traversal_count:        AtomicU64,

    // ── Lifecycle ──
    pub sleep_count:            AtomicU64,
    pub zaratustra_count:       AtomicU64,

    // ── Errors ──
    pub error_count:            AtomicU64,

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
            traversal_count:         AtomicU64::new(0),
            sleep_count:             AtomicU64::new(0),
            zaratustra_count:        AtomicU64::new(0),
            error_count:             AtomicU64::new(0),
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

        format!(
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
        )
    }
}
