use tokio::sync::broadcast;
use uuid::Uuid;

use crate::observer::HealthReport;

/// Sector identifier in the Poincare ball (depth band x angular slice).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SectorId {
    pub depth_bin: usize,
    pub angular_bin: usize,
}

/// Reason the observer triggers a daemon wake-up.
#[derive(Debug, Clone)]
pub enum WakeUpReason {
    MeanEnergyBelow(f32),
    HausdorffOutOfRange(f32),
    GapCountExceeded(usize),
}

/// Events produced by agency daemons and the observer.
#[derive(Debug, Clone)]
pub enum AgencyEvent {
    /// EntropyDaemon detected high Hausdorff variance in a region.
    EntropySpike {
        region_id: usize,
        variance: f32,
        sample_node_ids: Vec<Uuid>,
    },

    /// CoherenceDaemon detected excessive diffusion overlap across scales.
    CoherenceDrop {
        /// Jaccard(t=0.1, t=10.0)
        overlap_01_10: f64,
        /// Jaccard(t=0.1, t=1.0)
        overlap_01_1: f64,
    },

    /// GapDaemon found a sparse sector in the Poincare ball.
    KnowledgeGap {
        sector: SectorId,
        density: f64,
        suggested_depth_range: (f32, f32),
    },

    /// Observer generated a periodic health report.
    HealthReport(Box<HealthReport>),

    /// NiilistaGcDaemon detected a cluster of near-duplicate nodes.
    SemanticRedundancy {
        /// Total nodes in the redundancy group.
        group_size: usize,
        /// Node selected as the surviving archetype (highest energy).
        archetype_id: Uuid,
        /// Nodes to be phantomized (all except archetype).
        redundant_ids: Vec<Uuid>,
    },

    /// Observer triggered a daemon wake-up due to critical thresholds.
    DaemonWakeUp { reason: WakeUpReason },
}

/// Internal pub/sub bus for inter-daemon communication.
///
/// Separate from the CDC system (which broadcasts mutations to external gRPC
/// subscribers). This bus carries low-frequency analytical signals between
/// agency components.
pub struct AgencyEventBus {
    tx: broadcast::Sender<AgencyEvent>,
}

impl AgencyEventBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Publish an event. No-op if there are no subscribers.
    pub fn publish(&self, event: AgencyEvent) {
        let _ = self.tx.send(event);
    }

    /// Create a new subscriber.
    pub fn subscribe(&self) -> broadcast::Receiver<AgencyEvent> {
        self.tx.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_and_receive() {
        let bus = AgencyEventBus::new(16);
        let mut rx = bus.subscribe();

        bus.publish(AgencyEvent::EntropySpike {
            region_id: 0,
            variance: 0.5,
            sample_node_ids: vec![],
        });

        let event = rx.try_recv().unwrap();
        match event {
            AgencyEvent::EntropySpike { variance, .. } => {
                assert!((variance - 0.5).abs() < 1e-6);
            }
            _ => panic!("expected EntropySpike"),
        }
    }

    #[test]
    fn no_subscribers_no_panic() {
        let bus = AgencyEventBus::new(16);
        bus.publish(AgencyEvent::DaemonWakeUp {
            reason: WakeUpReason::MeanEnergyBelow(0.1),
        });
    }

    #[test]
    fn multiple_subscribers() {
        let bus = AgencyEventBus::new(16);
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        bus.publish(AgencyEvent::CoherenceDrop {
            overlap_01_10: 0.8,
            overlap_01_1: 0.5,
        });

        assert!(rx1.try_recv().is_ok());
        assert!(rx2.try_recv().is_ok());
    }
}
