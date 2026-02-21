//! **Eternal Recurrence** — temporal echo system.
//!
//! # Philosophy
//! Nietzsche's *Ewige Wiederkehr* is the thought that all events recur
//! infinitely — the heaviest weight, the ultimate affirmation of life.
//! In NietzscheDB this becomes: every node that reaches peak energy leaves
//! a *temporal echo* — a lightweight snapshot of its state at that moment.
//! These echoes form a living memory-of-memory, allowing the graph to
//! *remember* which knowledge was once most vital, even after energy decays.
//!
//! # Implementation
//! Echoes are stored as a JSON array inside `node.content["_zaratustra_echoes"]`.
//! Each echo records:
//! ```json
//! {
//!   "timestamp_secs": 1700000000,
//!   "energy":         0.85,
//!   "echo_index":     3
//! }
//! ```
//! When the echo ring-buffer fills (`max_echoes_per_node`), the oldest echo
//! is evicted (FIFO).

use std::time::{SystemTime, UNIX_EPOCH};

use nietzsche_graph::GraphStorage;

use crate::config::ZaratustraConfig;
use crate::error::ZaratustraError;

const ECHO_KEY: &str = "_zaratustra_echoes";

/// Statistics produced by the Eternal Recurrence phase.
#[derive(Debug, Clone, Default)]
pub struct EternalRecurrenceReport {
    /// Number of new echoes created in this cycle.
    pub echoes_created: u64,
    /// Number of echoes evicted (ring-buffer overflow).
    pub echoes_evicted: u64,
    /// Total echo count across all nodes after this cycle.
    pub total_echoes: u64,
}

/// Create echoes for all nodes whose current energy exceeds
/// `config.echo_threshold`.  Echoes are stored inside each node's JSON
/// `content` blob under the `_zaratustra_echoes` key.
///
/// `GraphStorage::put_node` takes `&self`, so no mutable borrow is required.
pub fn run_eternal_recurrence(
    storage: &GraphStorage,
    config:  &ZaratustraConfig,
) -> Result<EternalRecurrenceReport, ZaratustraError> {
    let nodes = storage.scan_nodes()
        .map_err(|e| ZaratustraError::Graph(e.to_string()))?;

    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut echoes_created = 0u64;
    let mut echoes_evicted = 0u64;
    let mut total_echoes   = 0u64;

    for mut node in nodes {
        // Count pre-existing echoes regardless of threshold
        let pre_count = echo_count(&node.content);
        total_echoes += pre_count as u64;

        // Only snapshot if above echo threshold
        if node.energy < config.echo_threshold {
            continue;
        }

        // Build the new echo entry
        let new_echo = serde_json::json!({
            "timestamp_secs": now_secs,
            "energy":         node.energy,
        });

        // Mutate the content JSON in place
        let evicted = push_echo(&mut node.content, new_echo, config.max_echoes_per_node);
        echoes_created += 1;
        if evicted {
            echoes_evicted += 1;
        }
        total_echoes += 1;

        // Persist the full updated node back to storage.
        // put_node overwrites all fields including content.
        storage.put_node(&node)
            .map_err(|e| ZaratustraError::Graph(e.to_string()))?;
    }

    Ok(EternalRecurrenceReport {
        echoes_created,
        echoes_evicted,
        total_echoes,
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Count the number of echoes stored in `content[ECHO_KEY]`.
fn echo_count(content: &serde_json::Value) -> usize {
    content
        .get(ECHO_KEY)
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0)
}

/// Append `echo` to `content[ECHO_KEY]`, evicting the oldest entry if the
/// ring-buffer is full.  Returns `true` if an entry was evicted.
fn push_echo(
    content:   &mut serde_json::Value,
    echo:      serde_json::Value,
    max:       usize,
) -> bool {
    // Ensure content is a JSON object.
    if !content.is_object() {
        *content = serde_json::json!({});
    }

    let echoes = content
        .as_object_mut()
        .unwrap()
        .entry(ECHO_KEY)
        .or_insert_with(|| serde_json::json!([]));

    let arr = match echoes.as_array_mut() {
        Some(a) => a,
        None    => {
            *echoes = serde_json::json!([]);
            echoes.as_array_mut().unwrap()
        }
    };

    let evicted = if arr.len() >= max {
        arr.remove(0); // evict oldest
        true
    } else {
        false
    };

    arr.push(echo);
    evicted
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{GraphStorage, Node, PoincareVector};
    use tempfile::TempDir;
    use uuid::Uuid;

    fn open_temp_db() -> (GraphStorage, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();
        (storage, dir)
    }

    fn make_node(x: f32, y: f32, energy: f32) -> Node {
        let mut node = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, y]),
            serde_json::json!({"label": "test"}),
        );
        node.energy = energy;
        node
    }

    fn default_config() -> ZaratustraConfig {
        ZaratustraConfig::default()
    }

    // ── push_echo / echo_count unit tests ────────────────────────────────────

    #[test]
    fn echo_count_empty_content() {
        let content = serde_json::json!({});
        assert_eq!(echo_count(&content), 0);
    }

    #[test]
    fn echo_count_with_existing_echoes() {
        let content = serde_json::json!({
            "_zaratustra_echoes": [
                {"timestamp_secs": 1, "energy": 0.8},
                {"timestamp_secs": 2, "energy": 0.9},
            ]
        });
        assert_eq!(echo_count(&content), 2);
    }

    #[test]
    fn push_echo_appends_to_empty() {
        let mut content = serde_json::json!({});
        let echo = serde_json::json!({"timestamp_secs": 100, "energy": 0.8});
        let evicted = push_echo(&mut content, echo, 5);
        assert!(!evicted);
        assert_eq!(echo_count(&content), 1);
    }

    #[test]
    fn push_echo_evicts_oldest_when_full() {
        let mut content = serde_json::json!({
            "_zaratustra_echoes": [
                {"timestamp_secs": 1, "energy": 0.7},
                {"timestamp_secs": 2, "energy": 0.8},
            ]
        });
        let echo = serde_json::json!({"timestamp_secs": 3, "energy": 0.9});
        let evicted = push_echo(&mut content, echo, 2);
        assert!(evicted, "should evict when at capacity");
        assert_eq!(echo_count(&content), 2);

        // The oldest (timestamp_secs=1) should be evicted
        let echoes = content[ECHO_KEY].as_array().unwrap();
        assert_eq!(echoes[0]["timestamp_secs"], 2);
        assert_eq!(echoes[1]["timestamp_secs"], 3);
    }

    #[test]
    fn push_echo_handles_non_object_content() {
        // Content is a string, not an object — should be replaced with {}
        let mut content = serde_json::json!("not an object");
        let echo = serde_json::json!({"timestamp_secs": 1, "energy": 0.5});
        let evicted = push_echo(&mut content, echo, 5);
        assert!(!evicted);
        assert!(content.is_object());
        assert_eq!(echo_count(&content), 1);
    }

    // ── Integration tests with GraphStorage ──────────────────────────────────

    #[test]
    fn empty_graph_produces_no_echoes() {
        let (storage, _dir) = open_temp_db();
        let report = run_eternal_recurrence(&storage, &default_config()).unwrap();
        assert_eq!(report.echoes_created, 0);
        assert_eq!(report.echoes_evicted, 0);
        assert_eq!(report.total_echoes, 0);
    }

    #[test]
    fn node_above_threshold_gets_echo() {
        let (storage, _dir) = open_temp_db();
        let cfg = ZaratustraConfig {
            echo_threshold: 0.5,
            max_echoes_per_node: 5,
            ..default_config()
        };

        let node = make_node(0.1, 0.1, 0.8); // energy 0.8 > threshold 0.5
        let id = node.id;
        storage.put_node(&node).unwrap();

        let report = run_eternal_recurrence(&storage, &cfg).unwrap();
        assert_eq!(report.echoes_created, 1);

        // Verify echo is stored in content
        let updated = storage.scan_nodes().unwrap();
        let n = updated.iter().find(|n| n.id == id).unwrap();
        let echoes = n.content[ECHO_KEY].as_array().unwrap();
        assert_eq!(echoes.len(), 1);
        assert!((echoes[0]["energy"].as_f64().unwrap() - 0.8).abs() < 1e-3);
    }

    #[test]
    fn node_below_threshold_gets_no_echo() {
        let (storage, _dir) = open_temp_db();
        let cfg = ZaratustraConfig {
            echo_threshold: 0.9,
            ..default_config()
        };

        let node = make_node(0.1, 0.1, 0.5); // energy 0.5 < threshold 0.9
        let id = node.id;
        storage.put_node(&node).unwrap();

        let report = run_eternal_recurrence(&storage, &cfg).unwrap();
        assert_eq!(report.echoes_created, 0);

        let updated = storage.scan_nodes().unwrap();
        let n = updated.iter().find(|n| n.id == id).unwrap();
        assert_eq!(echo_count(&n.content), 0);
    }

    #[test]
    fn echo_ring_buffer_eviction() {
        let (storage, _dir) = open_temp_db();
        let cfg = ZaratustraConfig {
            echo_threshold: 0.3,
            max_echoes_per_node: 2,
            ..default_config()
        };

        let node = make_node(0.1, 0.1, 0.8);
        let id = node.id;
        storage.put_node(&node).unwrap();

        // Run 3 times — max_echoes_per_node=2, so 3rd run should evict the 1st echo.
        let r1 = run_eternal_recurrence(&storage, &cfg).unwrap();
        assert_eq!(r1.echoes_created, 1);
        assert_eq!(r1.echoes_evicted, 0);

        let r2 = run_eternal_recurrence(&storage, &cfg).unwrap();
        assert_eq!(r2.echoes_created, 1);
        assert_eq!(r2.echoes_evicted, 0);

        let r3 = run_eternal_recurrence(&storage, &cfg).unwrap();
        assert_eq!(r3.echoes_created, 1);
        assert_eq!(r3.echoes_evicted, 1); // ring buffer overflow

        // Verify final state has exactly 2 echoes
        let updated = storage.scan_nodes().unwrap();
        let n = updated.iter().find(|n| n.id == id).unwrap();
        assert_eq!(echo_count(&n.content), 2);
    }

    #[test]
    fn multiple_nodes_mixed_threshold() {
        let (storage, _dir) = open_temp_db();
        let cfg = ZaratustraConfig {
            echo_threshold: 0.6,
            max_echoes_per_node: 10,
            ..default_config()
        };

        // Node above threshold
        let high = make_node(0.1, 0.0, 0.9);
        // Node below threshold
        let low = make_node(0.0, 0.1, 0.3);
        // Node exactly at threshold (energy < threshold, NOT >=)
        let borderline = make_node(0.1, 0.1, 0.6);

        storage.put_node(&high).unwrap();
        storage.put_node(&low).unwrap();
        storage.put_node(&borderline).unwrap();

        let report = run_eternal_recurrence(&storage, &cfg).unwrap();
        // high (0.9 >= 0.6) and borderline (0.6 >= 0.6) get echoes; low (0.3) does not.
        assert_eq!(report.echoes_created, 2);
    }

    #[test]
    fn echo_contains_timestamp() {
        let (storage, _dir) = open_temp_db();
        let cfg = ZaratustraConfig {
            echo_threshold: 0.0,
            ..default_config()
        };

        let node = make_node(0.1, 0.1, 0.5);
        let id = node.id;
        storage.put_node(&node).unwrap();

        run_eternal_recurrence(&storage, &cfg).unwrap();

        let updated = storage.scan_nodes().unwrap();
        let n = updated.iter().find(|n| n.id == id).unwrap();
        let echoes = n.content[ECHO_KEY].as_array().unwrap();
        assert!(echoes[0].get("timestamp_secs").is_some());
        let ts = echoes[0]["timestamp_secs"].as_u64().unwrap();
        // Timestamp should be recent (within the last minute)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!(
            ts <= now && ts >= now - 60,
            "timestamp {ts} should be close to now {now}",
        );
    }
}
