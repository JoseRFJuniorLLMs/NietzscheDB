//! Global metrics registry singleton backed by `prometheus::Registry`.

use once_cell::sync::Lazy;
use prometheus::{Encoder, Registry, TextEncoder};
use thiserror::Error;

/// Errors that can occur during metrics operations.
#[derive(Debug, Error)]
pub enum MetricsError {
    #[error("prometheus error: {0}")]
    Prometheus(#[from] prometheus::Error),

    #[error("encoding error: {0}")]
    Encode(String),
}

/// Wrapper around a `prometheus::Registry` that acts as the single
/// source of truth for all NietzscheDB metrics.
pub struct MetricsRegistry {
    inner: Registry,
}

impl MetricsRegistry {
    /// Create a new `MetricsRegistry` with a fresh `prometheus::Registry`.
    fn new() -> Self {
        Self {
            inner: Registry::new(),
        }
    }

    /// Return a reference to the underlying `prometheus::Registry`.
    ///
    /// Collectors (counters, histograms, gauges) use this to register
    /// themselves during lazy initialisation.
    pub fn prometheus_registry(&self) -> &Registry {
        &self.inner
    }
}

/// Global singleton instance.
static REGISTRY: Lazy<MetricsRegistry> = Lazy::new(MetricsRegistry::new);

/// Return a reference to the global [`MetricsRegistry`] singleton.
///
/// The registry is created once on first access and lives for the
/// remainder of the process.
pub fn registry() -> &'static MetricsRegistry {
    &REGISTRY
}

/// Encode every metric that has been registered into the Prometheus
/// text exposition format.
///
/// This is the string you serve at the `/metrics` HTTP endpoint.
pub fn encode_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.inner.gather();
    let mut buffer = Vec::new();
    encoder
        .encode(&metric_families, &mut buffer)
        .expect("encoding to a Vec<u8> should never fail");
    String::from_utf8(buffer).expect("prometheus text format is always valid UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_registry_singleton() {
        let r1 = registry();
        let r2 = registry();
        // Both references must point to the exact same allocation.
        assert!(std::ptr::eq(r1, r2));
    }
}
