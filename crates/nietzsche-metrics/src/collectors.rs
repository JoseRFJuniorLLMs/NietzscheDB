//! Metric collectors for NietzscheDB.
//!
//! Every public `static` in this module is a lazily-initialised Prometheus
//! metric that is automatically registered in the global
//! [`MetricsRegistry`](crate::registry::MetricsRegistry).

use once_cell::sync::Lazy;
use prometheus::{
    Gauge, Histogram, HistogramOpts, IntCounter, IntCounterVec, Opts,
};

use crate::registry::registry;

// ---------------------------------------------------------------------------
// Helper — shortcut to register a metric in the global registry.
// ---------------------------------------------------------------------------

fn register<C: prometheus::core::Collector + Clone + 'static>(c: C) -> C {
    registry()
        .prometheus_registry()
        .register(Box::new(c.clone()))
        .expect("metric registration should not fail");
    c
}

// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------

/// Total number of nodes inserted into the graph.
pub static NODES_INSERTED: Lazy<IntCounter> = Lazy::new(|| {
    register(IntCounter::new("nietzsche_nodes_inserted_total", "Total nodes inserted").unwrap())
});

/// Total number of nodes deleted from the graph.
pub static NODES_DELETED: Lazy<IntCounter> = Lazy::new(|| {
    register(IntCounter::new("nietzsche_nodes_deleted_total", "Total nodes deleted").unwrap())
});

/// Total number of edges inserted into the graph.
pub static EDGES_INSERTED: Lazy<IntCounter> = Lazy::new(|| {
    register(IntCounter::new("nietzsche_edges_inserted_total", "Total edges inserted").unwrap())
});

/// Total number of queries executed, broken down by query type.
///
/// Label `type` can be: `match`, `create`, `merge`, `diffuse`, etc.
pub static QUERIES_EXECUTED: Lazy<IntCounterVec> = Lazy::new(|| {
    register(
        IntCounterVec::new(
            Opts::new("nietzsche_queries_executed_total", "Total queries executed by type"),
            &["type"],
        )
        .unwrap(),
    )
});

/// Total number of failed queries.
pub static QUERIES_FAILED: Lazy<IntCounter> = Lazy::new(|| {
    register(IntCounter::new("nietzsche_queries_failed_total", "Total failed queries").unwrap())
});

// ---------------------------------------------------------------------------
// Histograms
// ---------------------------------------------------------------------------

/// Duration of query execution in seconds.
pub static QUERY_DURATION_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    let buckets = vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0];
    register(
        Histogram::with_opts(
            HistogramOpts::new(
                "nietzsche_query_duration_seconds",
                "Query execution time in seconds",
            )
            .buckets(buckets),
        )
        .unwrap(),
    )
});

/// Duration of KNN search in seconds.
pub static KNN_DURATION_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    let buckets = vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0];
    register(
        Histogram::with_opts(
            HistogramOpts::new(
                "nietzsche_knn_duration_seconds",
                "KNN search time in seconds",
            )
            .buckets(buckets),
        )
        .unwrap(),
    )
});

/// Duration of diffusion execution in seconds.
pub static DIFFUSION_DURATION_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    let buckets = vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0];
    register(
        Histogram::with_opts(
            HistogramOpts::new(
                "nietzsche_diffusion_duration_seconds",
                "Diffusion execution time in seconds",
            )
            .buckets(buckets),
        )
        .unwrap(),
    )
});

// ---------------------------------------------------------------------------
// Gauges
// ---------------------------------------------------------------------------

/// Current number of nodes in the graph.
pub static NODE_COUNT: Lazy<Gauge> = Lazy::new(|| {
    register(Gauge::new("nietzsche_node_count", "Current node count").unwrap())
});

/// Current number of edges in the graph.
pub static EDGE_COUNT: Lazy<Gauge> = Lazy::new(|| {
    register(Gauge::new("nietzsche_edge_count", "Current edge count").unwrap())
});

/// Number of active daemons.
pub static DAEMON_COUNT: Lazy<Gauge> = Lazy::new(|| {
    register(Gauge::new("nietzsche_daemon_count", "Active daemon count").unwrap())
});

/// Sum of all daemon energies.
pub static DAEMON_ENERGY_TOTAL: Lazy<Gauge> = Lazy::new(|| {
    register(Gauge::new("nietzsche_daemon_energy_total", "Sum of all daemon energies").unwrap())
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::encode_metrics;

    #[test]
    fn test_counters_increment() {
        NODES_INSERTED.inc();
        NODES_INSERTED.inc();
        assert_eq!(NODES_INSERTED.get(), 2);
    }

    #[test]
    fn test_histogram_observe() {
        QUERY_DURATION_SECONDS.observe(0.042);
        QUERY_DURATION_SECONDS.observe(0.13);
        assert_eq!(QUERY_DURATION_SECONDS.get_sample_count(), 2);
    }

    #[test]
    fn test_gauges_set() {
        NODE_COUNT.set(1234.0);
        assert!((NODE_COUNT.get() - 1234.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_query_counter_vec() {
        QUERIES_EXECUTED.with_label_values(&["match"]).inc();
        QUERIES_EXECUTED.with_label_values(&["match"]).inc();
        QUERIES_EXECUTED.with_label_values(&["create"]).inc();
        QUERIES_EXECUTED.with_label_values(&["diffuse"]).inc();

        assert_eq!(QUERIES_EXECUTED.with_label_values(&["match"]).get(), 2);
        assert_eq!(QUERIES_EXECUTED.with_label_values(&["create"]).get(), 1);
        assert_eq!(QUERIES_EXECUTED.with_label_values(&["diffuse"]).get(), 1);
    }

    #[test]
    fn test_encode_metrics() {
        // Force lazy initialisation of at least one metric so the registry
        // has something to encode.
        NODES_DELETED.inc();

        let output = encode_metrics();
        assert!(!output.is_empty(), "encoded metrics must not be empty");
        assert!(
            output.contains("nietzsche_nodes_deleted_total"),
            "output must contain the nodes_deleted metric"
        );
    }
}
