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
