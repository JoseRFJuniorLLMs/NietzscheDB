//! Will to Power — priority-based daemon scheduling.
//!
//! Daemons compete for execution slots based on their energy level and
//! urgency (time since last run). Higher-energy daemons run first,
//! implementing Nietzsche's "will to power" as computational priority.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::model::DaemonDef;

/// A daemon entry in the priority queue, ordered by priority score.
#[derive(Debug, Clone)]
pub struct PriorityEntry {
    pub daemon_name: String,
    pub priority:    f64,
    pub energy:      f64,
    pub overdue_secs: f64,
}

impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PriorityEntry {}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.partial_cmp(&other.priority)
            .unwrap_or(Ordering::Equal)
    }
}

/// Configuration for Will to Power scheduling.
#[derive(Debug, Clone)]
pub struct WillToPowerConfig {
    /// Weight for energy in priority calculation.
    pub energy_weight: f64,
    /// Weight for urgency (overdue time) in priority calculation.
    pub urgency_weight: f64,
    /// Maximum daemons to run per tick (CPU-aware throttle).
    pub max_per_tick: usize,
}

impl Default for WillToPowerConfig {
    fn default() -> Self {
        Self {
            energy_weight: 0.6,
            urgency_weight: 0.4,
            max_per_tick: 5,
        }
    }
}

impl WillToPowerConfig {
    pub fn from_env() -> Self {
        Self {
            energy_weight: std::env::var("DAEMON_ENERGY_WEIGHT")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(0.6),
            urgency_weight: std::env::var("DAEMON_URGENCY_WEIGHT")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(0.4),
            max_per_tick: std::env::var("DAEMON_MAX_PER_TICK")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(5),
        }
    }
}

/// Calculate priority score for a daemon.
///
/// Priority = energy_weight × energy + urgency_weight × normalized_overdue
/// where normalized_overdue = min(overdue_secs / interval_secs, 2.0)
pub fn calculate_priority(
    daemon: &DaemonDef,
    now: f64,
    config: &WillToPowerConfig,
) -> f64 {
    let overdue = (now - daemon.last_run - daemon.interval_secs).max(0.0);
    let normalized_overdue = (overdue / daemon.interval_secs.max(1.0)).min(2.0);

    config.energy_weight * daemon.energy
        + config.urgency_weight * normalized_overdue
}

/// Build a priority queue of due daemons, sorted by priority.
/// Returns at most `max_per_tick` entries.
pub fn prioritize_daemons(
    daemons: &[DaemonDef],
    now: f64,
    config: &WillToPowerConfig,
) -> Vec<PriorityEntry> {
    let mut heap = BinaryHeap::new();

    for d in daemons {
        if !d.is_due(now) {
            continue;
        }
        let overdue = (now - d.last_run - d.interval_secs).max(0.0);
        let priority = calculate_priority(d, now, config);

        heap.push(PriorityEntry {
            daemon_name: d.name.clone(),
            priority,
            energy: d.energy,
            overdue_secs: overdue,
        });
    }

    let mut result = Vec::with_capacity(config.max_per_tick);
    for _ in 0..config.max_per_tick {
        match heap.pop() {
            Some(entry) => result.push(entry),
            None => break,
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_query::ast::*;

    fn make_daemon(name: &str, energy: f64, last_run: f64, interval: f64) -> DaemonDef {
        DaemonDef {
            name: name.to_string(),
            on_pattern: NodePattern { alias: "n".into(), label: Some("Memory".into()) },
            when_cond: Condition::Compare {
                left:  Expr::Property { alias: "n".into(), field: "energy".into() },
                op:    CompOp::Gt,
                right: Expr::Float(0.5),
            },
            then_action: DaemonAction::Delete { alias: "n".into() },
            every: Expr::MathFunc {
                func: MathFunc::Interval,
                args: vec![MathFuncArg::Str("1h".into())],
            },
            energy,
            last_run,
            interval_secs: interval,
        }
    }

    #[test]
    fn high_energy_daemon_runs_first() {
        let daemons = vec![
            make_daemon("low",  0.3, 0.0, 100.0),
            make_daemon("high", 0.9, 0.0, 100.0),
            make_daemon("mid",  0.6, 0.0, 100.0),
        ];
        let config = WillToPowerConfig::default();
        let result = prioritize_daemons(&daemons, 200.0, &config);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].daemon_name, "high");
    }

    #[test]
    fn max_per_tick_limits() {
        let daemons: Vec<_> = (0..10)
            .map(|i| make_daemon(&format!("d{i}"), 0.5, 0.0, 100.0))
            .collect();
        let config = WillToPowerConfig { max_per_tick: 3, ..Default::default() };
        let result = prioritize_daemons(&daemons, 200.0, &config);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn overdue_daemon_gets_priority_boost() {
        let daemons = vec![
            make_daemon("recent",  0.9, 90.0,  100.0),  // barely due
            make_daemon("overdue", 0.5, 0.0,   100.0),  // very overdue
        ];
        let config = WillToPowerConfig {
            energy_weight: 0.3,
            urgency_weight: 0.7,
            max_per_tick: 5,
        };
        let result = prioritize_daemons(&daemons, 200.0, &config);
        assert_eq!(result[0].daemon_name, "overdue");
    }

    #[test]
    fn not_due_daemons_excluded() {
        let daemons = vec![
            make_daemon("not_due", 0.9, 150.0, 100.0),  // last_run=150, interval=100, now=200 → next at 250
        ];
        let config = WillToPowerConfig::default();
        let result = prioritize_daemons(&daemons, 200.0, &config);
        assert!(result.is_empty());
    }
}
