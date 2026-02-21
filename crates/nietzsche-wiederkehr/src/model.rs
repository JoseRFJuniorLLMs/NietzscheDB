use serde::{Serialize, Deserialize};
use nietzsche_query::ast::{Condition, DaemonAction, Expr, NodePattern};

/// Persistent daemon definition stored in CF_META with prefix `daemon:`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonDef {
    /// Unique daemon name (e.g. "guardian").
    pub name: String,
    /// Node pattern the daemon scans — alias + optional label.
    pub on_pattern: NodePattern,
    /// Condition that must be true for a node to trigger the daemon.
    pub when_cond: Condition,
    /// Action to execute when the condition fires.
    pub then_action: DaemonAction,
    /// Interval expression (serialized Expr, e.g. `INTERVAL("1h")`).
    pub every: Expr,
    /// Current energy level (0.0–1.0). When it reaches `min_energy` the daemon is reaped.
    pub energy: f64,
    /// Unix timestamp (seconds) of the last time this daemon ran.
    pub last_run: f64,
    /// Interval in seconds (resolved from `every` at creation time).
    pub interval_secs: f64,
}

impl DaemonDef {
    /// Check if this daemon is due to run given the current time.
    pub fn is_due(&self, now: f64) -> bool {
        now - self.last_run >= self.interval_secs
    }
}
