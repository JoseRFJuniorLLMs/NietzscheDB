//! Prometheus-compatible metrics for NietzscheDB.
//!
//! Exports counters, histograms, and gauges that the gRPC server
//! can serve at `/metrics` in Prometheus text format.

pub mod collectors;
pub mod registry;

pub use collectors::*;
pub use registry::{encode_metrics, MetricsRegistry};
