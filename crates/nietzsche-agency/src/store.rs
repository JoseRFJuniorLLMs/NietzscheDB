//! Persistence layer for agency data in RocksDB CF_META.
//!
//! Uses the existing `GraphStorage::put_meta` / `get_meta` / `scan_meta_prefix`
//! API to store and retrieve `HealthReport`s, following the same pattern as
//! the Wiederkehr daemon store.

use nietzsche_graph::GraphStorage;

use crate::error::AgencyError;
use crate::observer::HealthReport;

const HEALTH_REPORT_PREFIX: &str = "agency:health:";
const MAX_STORED_REPORTS: usize = 100;

/// Persist a HealthReport to RocksDB CF_META.
///
/// Key format: `agency:health:{tick_number:010}_{timestamp_ms}`
/// The zero-padded tick number ensures lexicographic ordering = chronological.
pub fn put_health_report(
    storage: &GraphStorage,
    report: &HealthReport,
) -> Result<(), AgencyError> {
    let key = format!(
        "{}{:010}_{}",
        HEALTH_REPORT_PREFIX, report.tick_number, report.timestamp_ms
    );
    let json = serde_json::to_vec(report)
        .map_err(|e| AgencyError::Internal(format!("serialize health report: {e}")))?;
    storage.put_meta(&key, &json)?;
    Ok(())
}

/// Retrieve all stored HealthReports, sorted by tick number ascending.
pub fn list_health_reports(
    storage: &GraphStorage,
) -> Result<Vec<HealthReport>, AgencyError> {
    let entries = storage.scan_meta_prefix(HEALTH_REPORT_PREFIX.as_bytes())?;
    let mut reports = Vec::with_capacity(entries.len());
    for (_key, value) in entries {
        match serde_json::from_slice::<HealthReport>(&value) {
            Ok(r) => reports.push(r),
            Err(_) => continue, // skip corrupted entries
        }
    }
    // Already sorted by key (tick_number is zero-padded)
    Ok(reports)
}

/// Retrieve the most recent HealthReport, if any.
pub fn get_latest_health_report(
    storage: &GraphStorage,
) -> Result<Option<HealthReport>, AgencyError> {
    let reports = list_health_reports(storage)?;
    Ok(reports.into_iter().last())
}

/// Prune old reports, keeping only the most recent `MAX_STORED_REPORTS`.
pub fn prune_health_reports(
    storage: &GraphStorage,
) -> Result<usize, AgencyError> {
    let entries = storage.scan_meta_prefix(HEALTH_REPORT_PREFIX.as_bytes())?;
    if entries.len() <= MAX_STORED_REPORTS {
        return Ok(0);
    }

    let to_remove = entries.len() - MAX_STORED_REPORTS;
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
    fn persist_and_retrieve_report() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        let report = HealthReport {
            tick_number: 5,
            timestamp_ms: 1700000000000,
            total_nodes: 100,
            mean_energy: 0.65,
            global_hausdorff: 1.45,
            ..HealthReport::default()
        };

        put_health_report(&storage, &report).unwrap();

        let reports = list_health_reports(&storage).unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].tick_number, 5);
        assert_eq!(reports[0].total_nodes, 100);
    }

    #[test]
    fn reports_sorted_by_tick() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        for tick in [10, 5, 15, 1] {
            let report = HealthReport {
                tick_number: tick,
                timestamp_ms: 1700000000000 + tick,
                ..HealthReport::default()
            };
            put_health_report(&storage, &report).unwrap();
        }

        let reports = list_health_reports(&storage).unwrap();
        assert_eq!(reports.len(), 4);
        assert_eq!(reports[0].tick_number, 1);
        assert_eq!(reports[1].tick_number, 5);
        assert_eq!(reports[2].tick_number, 10);
        assert_eq!(reports[3].tick_number, 15);
    }

    #[test]
    fn get_latest() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        for tick in 1..=3 {
            let report = HealthReport {
                tick_number: tick,
                ..HealthReport::default()
            };
            put_health_report(&storage, &report).unwrap();
        }

        let latest = get_latest_health_report(&storage).unwrap().unwrap();
        assert_eq!(latest.tick_number, 3);
    }

    #[test]
    fn prune_keeps_max() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        // Insert MAX_STORED_REPORTS + 5
        for tick in 1..=(MAX_STORED_REPORTS as u64 + 5) {
            let report = HealthReport {
                tick_number: tick,
                ..HealthReport::default()
            };
            put_health_report(&storage, &report).unwrap();
        }

        let removed = prune_health_reports(&storage).unwrap();
        assert_eq!(removed, 5);

        let reports = list_health_reports(&storage).unwrap();
        assert_eq!(reports.len(), MAX_STORED_REPORTS);
        // Oldest remaining should be tick 6
        assert_eq!(reports[0].tick_number, 6);
    }
}
