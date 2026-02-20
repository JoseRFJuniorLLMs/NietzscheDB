//! Change Data Capture (CDC) broadcaster.
//!
//! Broadcasts mutation events via `tokio::broadcast` to all active subscribers.
//! Each gRPC `SubscribeCDC` stream holds a `Receiver<CdcEvent>`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use uuid::Uuid;

/// A CDC event emitted on every mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcEvent {
    /// Monotonically increasing logical sequence number.
    pub lsn: u64,
    /// Type of mutation.
    pub event_type: CdcEventType,
    /// Unix milliseconds when the event was created.
    pub timestamp_ms: u64,
    /// Primary entity ID (node or edge UUID).
    pub entity_id: Uuid,
    /// Collection where the mutation occurred.
    pub collection: String,
}

/// Type of mutation that occurred.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CdcEventType {
    InsertNode,
    UpdateNode,
    DeleteNode,
    InsertEdge,
    DeleteEdge,
    BatchInsertNodes { count: u32 },
    BatchInsertEdges { count: u32 },
    SleepCycle,
    Zaratustra,
}

impl CdcEventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InsertNode => "INSERT_NODE",
            Self::UpdateNode => "UPDATE_NODE",
            Self::DeleteNode => "DELETE_NODE",
            Self::InsertEdge => "INSERT_EDGE",
            Self::DeleteEdge => "DELETE_EDGE",
            Self::BatchInsertNodes { .. } => "BATCH_INSERT_NODES",
            Self::BatchInsertEdges { .. } => "BATCH_INSERT_EDGES",
            Self::SleepCycle => "SLEEP_CYCLE",
            Self::Zaratustra => "ZARATUSTRA",
        }
    }
}

/// CDC broadcaster that wraps a tokio broadcast channel.
///
/// Cheap to clone (inner Arc via broadcast::Sender).
#[derive(Clone)]
pub struct CdcBroadcaster {
    tx: broadcast::Sender<CdcEvent>,
    lsn: std::sync::Arc<AtomicU64>,
}

impl CdcBroadcaster {
    /// Create a new broadcaster with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self {
            tx,
            lsn: std::sync::Arc::new(AtomicU64::new(1)),
        }
    }

    /// Publish a CDC event. If no subscribers are listening, the event is dropped.
    pub fn publish(&self, event_type: CdcEventType, entity_id: Uuid, collection: &str) {
        let lsn = self.lsn.fetch_add(1, Ordering::Relaxed);
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let event = CdcEvent {
            lsn,
            event_type,
            timestamp_ms,
            entity_id,
            collection: collection.to_string(),
        };

        // Ignore error — means no active subscribers
        let _ = self.tx.send(event);
    }

    /// Subscribe to the CDC stream. Returns a receiver that yields events.
    pub fn subscribe(&self) -> broadcast::Receiver<CdcEvent> {
        self.tx.subscribe()
    }

    /// Current LSN (next event will have this LSN).
    pub fn current_lsn(&self) -> u64 {
        self.lsn.load(Ordering::Relaxed)
    }
}
