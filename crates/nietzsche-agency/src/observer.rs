use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use nietzsche_lsystem::global_hausdorff;
use tokio::sync::broadcast;

use crate::config::AgencyConfig;
use crate::error::AgencyError;
use crate::event_bus::{AgencyEvent, AgencyEventBus, SectorId, WakeUpReason};

/// Energy distribution percentiles.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EnergyPercentiles {
    pub p10: f32,
    pub p25: f32,
    pub p50: f32,
    pub p75: f32,
    pub p90: f32,
}

/// Comprehensive health report for the graph.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct HealthReport {
    pub timestamp_ms: u64,
    pub tick_number: u64,

    // Global metrics
    pub total_nodes: usize,
    pub total_edges: usize,
    pub global_hausdorff: f32,
    /// True if `global_hausdorff` is in the healthy fractal range [1.2, 1.8].
    pub is_fractal: bool,

    // Energy distribution
    pub mean_energy: f32,
    pub energy_std: f32,
    pub energy_percentiles: EnergyPercentiles,

    // Coherence (from CoherenceDaemon events)
    pub coherence_score: f64,

    // Gaps & entropy (from daemon events)
    pub gap_count: usize,
    pub gap_sectors: Vec<SectorId>,
    pub entropy_spike_count: usize,
    pub max_regional_variance: f32,
}

/// Meta-observer that aggregates daemon events and produces periodic health
/// reports. This is the "consciousness" of the graph — it watches everything.
///
/// The observer is the **only stateful** component in the agency crate:
/// it tracks tick count and buffers recent events between reports.
pub struct MetaObserver {
    tick_count: u64,
    rx: broadcast::Receiver<AgencyEvent>,
    recent_gap_count: usize,
    recent_entropy_spikes: usize,
    recent_max_variance: f32,
    recent_coherence_overlap: Option<f64>,
    recent_gap_sectors: Vec<SectorId>,
}

impl MetaObserver {
    pub fn new(bus: &AgencyEventBus) -> Self {
        Self {
            tick_count: 0,
            rx: bus.subscribe(),
            recent_gap_count: 0,
            recent_entropy_spikes: 0,
            recent_max_variance: 0.0,
            recent_coherence_overlap: None,
            recent_gap_sectors: Vec::new(),
        }
    }

    /// Called once per agency tick. Drains events, aggregates, and may
    /// produce a HealthReport and/or wake-up events.
    pub fn tick(
        &mut self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        bus: &AgencyEventBus,
        config: &AgencyConfig,
    ) -> Result<Option<HealthReport>, AgencyError> {
        self.tick_count += 1;

        // Drain all pending events from the bus
        self.drain_events();

        // Quick-check triggers (every tick)
        let total_sectors = config.gap_depth_bins * config.gap_sector_count;
        if self.recent_gap_count > total_sectors / 2 {
            bus.publish(AgencyEvent::DaemonWakeUp {
                reason: WakeUpReason::GapCountExceeded(self.recent_gap_count),
            });
        }

        // Full report only at intervals
        if self.tick_count % config.observer_report_interval as u64 != 0 {
            return Ok(None);
        }

        let report = self.generate_report(storage, adjacency, config)?;

        // Wake-up triggers
        if report.mean_energy < config.observer_wake_energy_threshold {
            bus.publish(AgencyEvent::DaemonWakeUp {
                reason: WakeUpReason::MeanEnergyBelow(report.mean_energy),
            });
        }
        if report.global_hausdorff < config.observer_wake_hausdorff_lo
            || report.global_hausdorff > config.observer_wake_hausdorff_hi
        {
            bus.publish(AgencyEvent::DaemonWakeUp {
                reason: WakeUpReason::HausdorffOutOfRange(report.global_hausdorff),
            });
        }

        bus.publish(AgencyEvent::HealthReport(Box::new(report.clone())));

        // Reset accumulators
        self.recent_gap_count = 0;
        self.recent_entropy_spikes = 0;
        self.recent_max_variance = 0.0;
        self.recent_coherence_overlap = None;
        self.recent_gap_sectors.clear();

        Ok(Some(report))
    }

    fn drain_events(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(event) => match event {
                    AgencyEvent::EntropySpike { variance, .. } => {
                        self.recent_entropy_spikes += 1;
                        if variance > self.recent_max_variance {
                            self.recent_max_variance = variance;
                        }
                    }
                    AgencyEvent::CoherenceDrop { overlap_01_10, .. } => {
                        self.recent_coherence_overlap = Some(overlap_01_10);
                    }
                    AgencyEvent::KnowledgeGap { sector, .. } => {
                        self.recent_gap_count += 1;
                        self.recent_gap_sectors.push(sector);
                    }
                    _ => {}
                },
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
                Err(broadcast::error::TryRecvError::Closed) => break,
            }
        }
    }

    fn generate_report(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        _config: &AgencyConfig,
    ) -> Result<HealthReport, AgencyError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Collect energies for statistics
        let mut energies: Vec<f32> = Vec::new();
        let mut nodes_for_hausdorff: Vec<nietzsche_graph::Node> = Vec::new();
        let mut total_nodes = 0usize;

        for result in storage.iter_nodes() {
            let node = match result {
                Ok(n) => n,
                Err(_) => continue,
            };
            total_nodes += 1;
            energies.push(node.energy);
            nodes_for_hausdorff.push(node);
        }

        let total_edges = adjacency.edge_count();

        // Energy statistics
        let (mean_energy, energy_std, percentiles) = compute_energy_stats(&energies);

        // Global Hausdorff
        let global_h = global_hausdorff(&nodes_for_hausdorff);
        let is_fractal = global_h >= 1.2 && global_h <= 1.8;

        // Coherence score: 1.0 - overlap (higher = better)
        let coherence_score = match self.recent_coherence_overlap {
            Some(overlap) => 1.0 - overlap,
            None => 1.0, // no data = assume good
        };

        Ok(HealthReport {
            timestamp_ms: now,
            tick_number: self.tick_count,
            total_nodes,
            total_edges,
            global_hausdorff: global_h,
            is_fractal,
            mean_energy,
            energy_std,
            energy_percentiles: percentiles,
            coherence_score,
            gap_count: self.recent_gap_count,
            gap_sectors: self.recent_gap_sectors.clone(),
            entropy_spike_count: self.recent_entropy_spikes,
            max_regional_variance: self.recent_max_variance,
        })
    }
}

fn compute_energy_stats(energies: &[f32]) -> (f32, f32, EnergyPercentiles) {
    if energies.is_empty() {
        return (0.0, 0.0, EnergyPercentiles::default());
    }

    let n = energies.len() as f64;
    let sum: f64 = energies.iter().map(|&e| e as f64).sum();
    let mean = (sum / n) as f32;

    let var: f64 = energies.iter().map(|&e| {
        let d = e as f64 - mean as f64;
        d * d
    }).sum::<f64>() / n;
    let std = var.sqrt() as f32;

    let mut sorted = energies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile = |p: f64| -> f32 {
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };

    let percentiles = EnergyPercentiles {
        p10: percentile(10.0),
        p25: percentile(25.0),
        p50: percentile(50.0),
        p75: percentile(75.0),
        p90: percentile(90.0),
    };

    (mean, std, percentiles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{Node, PoincareVector};
    use tempfile::TempDir;
    use uuid::Uuid;

    fn open_storage(dir: &TempDir) -> GraphStorage {
        GraphStorage::open(dir.path().to_str().unwrap()).unwrap()
    }

    fn make_node(x: f32, y: f32, energy: f32) -> Node {
        let mut node = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, y]),
            serde_json::json!({}),
        );
        node.meta.energy = energy;
        node
    }

    #[test]
    fn report_generated_at_interval() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = nietzsche_graph::AdjacencyIndex::new();
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig {
            observer_report_interval: 3,
            ..AgencyConfig::default()
        };

        for i in 0..10 {
            let node = make_node(0.01 * (i + 1) as f32, 0.01, 0.5);
            storage.put_node(&node).unwrap();
        }

        let mut observer = MetaObserver::new(&bus);

        // Ticks 1, 2: no report
        assert!(observer.tick(&storage, &adjacency, &bus, &config).unwrap().is_none());
        assert!(observer.tick(&storage, &adjacency, &bus, &config).unwrap().is_none());

        // Tick 3: report generated
        let report = observer.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert!(report.is_some());
        let r = report.unwrap();
        assert_eq!(r.total_nodes, 10);
        assert_eq!(r.tick_number, 3);
    }

    #[test]
    fn wake_up_on_low_energy() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = nietzsche_graph::AdjacencyIndex::new();
        let bus = AgencyEventBus::new(64);
        let mut rx = bus.subscribe();
        let config = AgencyConfig {
            observer_report_interval: 1, // report every tick
            observer_wake_energy_threshold: 0.5,
            ..AgencyConfig::default()
        };

        // Low-energy nodes
        for i in 0..5 {
            let node = make_node(0.01 * (i + 1) as f32, 0.01, 0.1);
            storage.put_node(&node).unwrap();
        }

        let mut observer = MetaObserver::new(&bus);
        let report = observer.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert!(report.is_some());

        // Check for wake-up event
        let mut found_wake_up = false;
        while let Ok(event) = rx.try_recv() {
            if matches!(event, AgencyEvent::DaemonWakeUp { reason: WakeUpReason::MeanEnergyBelow(_) }) {
                found_wake_up = true;
            }
        }
        assert!(found_wake_up, "expected MeanEnergyBelow wake-up");
    }

    #[test]
    fn energy_percentiles_correct() {
        let energies: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let (mean, _std, p) = compute_energy_stats(&energies);
        assert!((mean - 0.495).abs() < 0.01);
        assert!((p.p50 - 0.50).abs() < 0.02);
        assert!(p.p10 < p.p25);
        assert!(p.p25 < p.p50);
        assert!(p.p50 < p.p75);
        assert!(p.p75 < p.p90);
    }
}
