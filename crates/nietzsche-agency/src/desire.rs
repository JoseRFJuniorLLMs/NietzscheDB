//! Motor de Desejo — Desire Engine for NietzscheDB.
//!
//! Transforms knowledge gaps detected by the GapDaemon into structured
//! **desire signals** that describe what the graph "wants to know". These
//! signals can be consumed by:
//!
//! - **EVA** (external AI agent) to seek missing knowledge
//! - **NQL pipeline** to generate targeted queries
//! - **L-System** to guide growth toward sparse regions
//!
//! ## Philosophy
//!
//! In Nietzsche's *Will to Power*, desire is not a lack but a creative
//! force — the impulse to overcome and expand. The Motor de Desejo
//! embodies this: knowledge gaps are not deficiencies but *opportunities
//! for growth*. Each desire signal carries the coordinates in hyperbolic
//! space where the graph is hungry for new connections.
//!
//! ## Architecture
//!
//! ```text
//! GapDaemon → KnowledgeGap events → DesireEngine → DesireSignal
//!                                                       │
//!                                         ┌─────────────┼──────────────┐
//!                                         │             │              │
//!                                    CF_META        AgencyEvent    Log/EVA
//!                                  (persisted)     (broadcast)    (external)
//! ```

use uuid::Uuid;

use nietzsche_graph::GraphStorage;

use crate::config::AgencyConfig;
use crate::error::AgencyError;
use crate::event_bus::{AgencyEventBus, SectorId};

/// A structured desire signal — what the graph "wants to know".
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DesireSignal {
    /// Unique ID for tracking this desire.
    pub id: Uuid,
    /// When this desire was generated (ms since epoch).
    pub timestamp_ms: u64,
    /// The sector in the Poincaré ball where the gap was detected.
    pub sector: SectorId,
    /// Suggested depth range in the hyperbolic space for new knowledge.
    pub depth_range: (f32, f32),
    /// Current density in the sector (fraction of expected).
    pub current_density: f64,
    /// Estimated priority (0.0 = low, 1.0 = critical).
    pub priority: f32,
    /// Suggested query template for EVA/NQL to fill this gap.
    pub suggested_query: String,
    /// Whether this desire has been fulfilled.
    pub fulfilled: bool,
}

const DESIRE_PREFIX: &str = "agency:desire:";

/// The Desire Engine transforms gap events into structured desire signals.
///
/// On each tick, it:
/// 1. Scans pending gap events from the event bus
/// 2. Generates desire signals with priority and query templates
/// 3. Persists them to CF_META for external consumption
/// 4. Broadcasts them via the agency event bus
pub struct DesireEngine {
    /// Counter for generating monotonic signal IDs.
    tick_count: u64,
}

impl DesireEngine {
    pub fn new() -> Self {
        Self { tick_count: 0 }
    }

    /// Process knowledge gaps and generate desire signals.
    ///
    /// Called by the AgencyEngine after daemons have ticked.
    /// Takes the gaps detected in this cycle and converts them
    /// into persisted, prioritized desire signals.
    pub fn process_gaps(
        &mut self,
        gaps: &[(SectorId, f64, (f32, f32))], // (sector, density, depth_range)
        storage: &GraphStorage,
        _bus: &AgencyEventBus,
        config: &AgencyConfig,
    ) -> Result<Vec<DesireSignal>, AgencyError> {
        self.tick_count += 1;

        if gaps.is_empty() {
            return Ok(Vec::new());
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let total_sectors = (config.gap_depth_bins * config.gap_sector_count) as f64;
        let mut signals = Vec::with_capacity(gaps.len());

        for (sector, density, depth_range) in gaps {
            // Priority: inversely proportional to density, weighted by sector importance
            // Center sectors (low depth) are more critical than boundary sectors
            let depth_weight = 1.0 - (depth_range.0 + depth_range.1) / 2.0;
            let density_weight = 1.0 - (*density as f32).min(1.0);
            let priority = (depth_weight * 0.4 + density_weight * 0.6).clamp(0.0, 1.0);

            let suggested_query = generate_query_template(sector, depth_range);

            let signal = DesireSignal {
                id: Uuid::new_v4(),
                timestamp_ms: now,
                sector: sector.clone(),
                depth_range: *depth_range,
                current_density: *density,
                priority,
                suggested_query,
                fulfilled: false,
            };

            // Persist to CF_META
            let key = format!(
                "{}{:010}_{}",
                DESIRE_PREFIX, self.tick_count, signal.id
            );
            if let Ok(json) = serde_json::to_vec(&signal) {
                let _ = storage.put_meta(&key, &json);
            }

            signals.push(signal);
        }

        // Sort by priority descending
        signals.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

        // Prune old desires (keep max 50)
        let _ = prune_old_desires(storage, 50);

        tracing::info!(
            desires = signals.len(),
            top_priority = signals.first().map(|s| s.priority).unwrap_or(0.0),
            gap_fraction = gaps.len() as f64 / total_sectors,
            "Motor de Desejo: desires generated"
        );

        Ok(signals)
    }
}

/// Generate a human-readable query template for a given gap sector.
fn generate_query_template(sector: &SectorId, depth_range: &(f32, f32)) -> String {
    let depth_desc = if depth_range.0 < 0.3 {
        "abstract/conceptual"
    } else if depth_range.0 < 0.6 {
        "mid-level/relational"
    } else {
        "specific/episodic"
    };

    format!(
        "SEARCH {} knowledge near depth [{:.2}, {:.2}] in angular sector {}, depth band {}",
        depth_desc, depth_range.0, depth_range.1, sector.angular_bin, sector.depth_bin
    )
}

/// List all pending (unfulfilled) desires from CF_META.
pub fn list_pending_desires(
    storage: &GraphStorage,
) -> Result<Vec<DesireSignal>, AgencyError> {
    let entries = storage.scan_meta_prefix(DESIRE_PREFIX.as_bytes())?;
    let mut desires = Vec::with_capacity(entries.len());
    for (_key, value) in entries {
        match serde_json::from_slice::<DesireSignal>(&value) {
            Ok(d) if !d.fulfilled => desires.push(d),
            _ => continue,
        }
    }
    desires.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));
    Ok(desires)
}

/// Mark a desire as fulfilled by its ID.
pub fn fulfill_desire(
    storage: &GraphStorage,
    desire_id: Uuid,
) -> Result<bool, AgencyError> {
    let entries = storage.scan_meta_prefix(DESIRE_PREFIX.as_bytes())?;
    for (key, value) in entries {
        if let Ok(mut desire) = serde_json::from_slice::<DesireSignal>(&value) {
            if desire.id == desire_id {
                desire.fulfilled = true;
                let json = serde_json::to_vec(&desire)
                    .map_err(|e| AgencyError::Internal(e.to_string()))?;
                let key_str = String::from_utf8_lossy(&key);
                storage.put_meta(&key_str, &json)?;
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Prune old desires keeping only the most recent `max` entries.
fn prune_old_desires(
    storage: &GraphStorage,
    max: usize,
) -> Result<usize, AgencyError> {
    let entries = storage.scan_meta_prefix(DESIRE_PREFIX.as_bytes())?;
    if entries.len() <= max {
        return Ok(0);
    }
    let to_remove = entries.len() - max;
    let mut removed = 0;
    for (key, _) in entries.into_iter().take(to_remove) {
        let key_str = String::from_utf8_lossy(&key);
        if storage.delete_meta(&key_str).is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_storage(dir: &TempDir) -> GraphStorage {
        GraphStorage::open(dir.path().to_str().unwrap()).unwrap()
    }

    #[test]
    fn generates_desires_from_gaps() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();
        let mut engine = DesireEngine::new();

        let gaps = vec![
            (SectorId { depth_bin: 1, angular_bin: 3 }, 0.02, (0.2, 0.4)),
            (SectorId { depth_bin: 3, angular_bin: 7 }, 0.05, (0.6, 0.8)),
        ];

        let signals = engine.process_gaps(&gaps, &storage, &bus, &config).unwrap();
        assert_eq!(signals.len(), 2);

        // First signal should be higher priority (lower depth = more central)
        assert!(signals[0].priority >= signals[1].priority);
        assert!(!signals[0].fulfilled);
        assert!(!signals[0].suggested_query.is_empty());
    }

    #[test]
    fn persisted_and_retrievable() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();
        let mut engine = DesireEngine::new();

        let gaps = vec![
            (SectorId { depth_bin: 0, angular_bin: 1 }, 0.01, (0.1, 0.2)),
        ];

        engine.process_gaps(&gaps, &storage, &bus, &config).unwrap();

        let pending = list_pending_desires(&storage).unwrap();
        assert_eq!(pending.len(), 1);
        assert!(!pending[0].fulfilled);
    }

    #[test]
    fn fulfill_marks_as_done() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();
        let mut engine = DesireEngine::new();

        let gaps = vec![
            (SectorId { depth_bin: 2, angular_bin: 5 }, 0.03, (0.4, 0.6)),
        ];

        let signals = engine.process_gaps(&gaps, &storage, &bus, &config).unwrap();
        let id = signals[0].id;

        let fulfilled = fulfill_desire(&storage, id).unwrap();
        assert!(fulfilled);

        // Should no longer appear in pending
        let pending = list_pending_desires(&storage).unwrap();
        assert_eq!(pending.len(), 0);
    }

    #[test]
    fn empty_gaps_no_desires() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();
        let mut engine = DesireEngine::new();

        let signals = engine.process_gaps(&[], &storage, &bus, &config).unwrap();
        assert!(signals.is_empty());
    }

    #[test]
    fn priority_central_higher() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();
        let mut engine = DesireEngine::new();

        // Central gap (low depth) vs peripheral gap (high depth), same density
        let gaps = vec![
            (SectorId { depth_bin: 0, angular_bin: 0 }, 0.05, (0.05, 0.15)),
            (SectorId { depth_bin: 4, angular_bin: 0 }, 0.05, (0.85, 0.95)),
        ];

        let signals = engine.process_gaps(&gaps, &storage, &bus, &config).unwrap();
        // Central gap should have higher priority
        assert!(signals[0].depth_range.0 < 0.5, "first desire should be central");
        assert!(signals[0].priority > signals[1].priority);
    }
}
