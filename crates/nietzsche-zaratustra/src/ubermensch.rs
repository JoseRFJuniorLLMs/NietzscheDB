//! **Übermensch** — elite-tier analytics and reporting.
//!
//! # Philosophy
//! Nietzsche's *Übermensch* is not a fixed entity but a direction — the
//! ideal of self-overcoming.  In NietzscheDB the Übermensch tier is the
//! top fraction of nodes by energy at any given moment: those that have
//! "overcome" their peers through repeated activation and propagation.
//!
//! # Current capabilities
//! This module computes the elite-tier stats and identifies the Übermensch
//! node IDs.  Future work: hot-tier promotion (in-memory pinning).

use uuid::Uuid;

use nietzsche_graph::GraphStorage;

use crate::config::ZaratustraConfig;
use crate::error::ZaratustraError;

/// Statistics produced by the Übermensch phase.
#[derive(Debug, Clone, Default)]
pub struct UbermenschReport {
    /// Number of nodes in the elite tier.
    pub elite_count: u64,
    /// Energy threshold at which the elite tier begins.
    pub energy_threshold: f32,
    /// Mean energy across elite nodes.
    pub mean_elite_energy: f32,
    /// Mean energy across non-elite nodes.
    pub mean_base_energy: f32,
    /// Node IDs of the elite tier (sorted descending by energy).
    pub elite_node_ids: Vec<Uuid>,
}

/// Compute the Übermensch tier statistics from the current node energies.
pub fn run_ubermensch(
    storage: &GraphStorage,
    config:  &ZaratustraConfig,
) -> Result<UbermenschReport, ZaratustraError> {
    let nodes = storage.scan_nodes()
        .map_err(|e| ZaratustraError::Graph(e.to_string()))?;

    if nodes.is_empty() {
        return Ok(UbermenschReport::default());
    }

    // Sort descending by energy
    let mut ranked: Vec<(Uuid, f32)> = nodes
        .iter()
        .map(|n| (n.id, n.energy))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let total       = ranked.len();
    let elite_count = ((total as f32 * config.ubermensch_top_fraction).ceil() as usize)
        .max(1)
        .min(total);

    let elite      = &ranked[..elite_count];
    let base       = &ranked[elite_count..];
    let threshold  = elite.last().map(|(_, e)| *e).unwrap_or(0.0);

    let mean_elite = if elite.is_empty() { 0.0 } else {
        elite.iter().map(|(_, e)| e).sum::<f32>() / elite.len() as f32
    };
    let mean_base = if base.is_empty() { 0.0 } else {
        base.iter().map(|(_, e)| e).sum::<f32>() / base.len() as f32
    };

    Ok(UbermenschReport {
        elite_count:      elite_count as u64,
        energy_threshold: threshold,
        mean_elite_energy: mean_elite,
        mean_base_energy:  mean_base,
        elite_node_ids:    elite.iter().map(|(id, _)| *id).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{GraphStorage, Node, PoincareVector};
    use tempfile::TempDir;

    fn open_temp_db() -> (GraphStorage, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();
        (storage, dir)
    }

    fn make_node(x: f32, y: f32, energy: f32) -> Node {
        let mut node = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, y]),
            serde_json::json!({}),
        );
        node.energy = energy;
        node
    }

    fn default_config() -> ZaratustraConfig {
        ZaratustraConfig::default()
    }

    #[test]
    fn empty_graph_returns_default_report() {
        let (storage, _dir) = open_temp_db();
        let report = run_ubermensch(&storage, &default_config()).unwrap();
        assert_eq!(report.elite_count, 0);
        assert!(report.elite_node_ids.is_empty());
    }

    #[test]
    fn single_node_is_always_elite() {
        let (storage, _dir) = open_temp_db();
        let node = make_node(0.1, 0.1, 0.5);
        let id = node.id;
        storage.put_node(&node).unwrap();

        let cfg = ZaratustraConfig {
            ubermensch_top_fraction: 0.10,
            ..default_config()
        };
        let report = run_ubermensch(&storage, &cfg).unwrap();

        // ceil(1 * 0.10) = 1, clamped to max(1, min(1, 1)) = 1
        assert_eq!(report.elite_count, 1);
        assert_eq!(report.elite_node_ids, vec![id]);
    }

    #[test]
    fn top_10_percent_of_10_nodes_is_1() {
        let (storage, _dir) = open_temp_db();
        let mut ids = Vec::new();
        // Create 10 nodes with energy 0.1, 0.2, ..., 1.0
        for i in 1..=10 {
            let energy = i as f32 * 0.1;
            let node = make_node(0.05 * i as f32, 0.0, energy);
            ids.push((node.id, energy));
            storage.put_node(&node).unwrap();
        }

        let cfg = ZaratustraConfig {
            ubermensch_top_fraction: 0.10,
            ..default_config()
        };
        let report = run_ubermensch(&storage, &cfg).unwrap();

        // ceil(10 * 0.10) = 1 elite node
        assert_eq!(report.elite_count, 1);

        // The elite node should be the one with energy 1.0
        let top_node = ids.iter().find(|(_, e)| (*e - 1.0).abs() < 1e-5).unwrap();
        assert_eq!(report.elite_node_ids.len(), 1);
        assert_eq!(report.elite_node_ids[0], top_node.0);
    }

    #[test]
    fn top_50_percent_of_4_nodes() {
        let (storage, _dir) = open_temp_db();

        let low1 = make_node(0.05, 0.0, 0.1);
        let low2 = make_node(0.0, 0.05, 0.2);
        let high1 = make_node(0.1, 0.0, 0.8);
        let high2 = make_node(0.0, 0.1, 0.9);

        let high1_id = high1.id;
        let high2_id = high2.id;

        storage.put_node(&low1).unwrap();
        storage.put_node(&low2).unwrap();
        storage.put_node(&high1).unwrap();
        storage.put_node(&high2).unwrap();

        let cfg = ZaratustraConfig {
            ubermensch_top_fraction: 0.50,
            ..default_config()
        };
        let report = run_ubermensch(&storage, &cfg).unwrap();

        // ceil(4 * 0.50) = 2 elite nodes
        assert_eq!(report.elite_count, 2);

        // Elite should contain high1 and high2
        assert!(report.elite_node_ids.contains(&high1_id));
        assert!(report.elite_node_ids.contains(&high2_id));
    }

    #[test]
    fn elite_ids_sorted_descending_by_energy() {
        let (storage, _dir) = open_temp_db();

        let n1 = make_node(0.1, 0.0, 0.7);
        let n2 = make_node(0.0, 0.1, 0.9);
        let n3 = make_node(0.1, 0.1, 0.5);

        let n1_id = n1.id;
        let n2_id = n2.id;

        storage.put_node(&n1).unwrap();
        storage.put_node(&n2).unwrap();
        storage.put_node(&n3).unwrap();

        let cfg2 = ZaratustraConfig {
            ubermensch_top_fraction: 0.50,
            ..default_config()
        };
        let report2 = run_ubermensch(&storage, &cfg2).unwrap();

        // ceil(3 * 0.50) = 2
        assert_eq!(report2.elite_count, 2);
        // Descending: n2(0.9), n1(0.7) — n3(0.5) excluded
        assert_eq!(report2.elite_node_ids[0], n2_id);
        assert_eq!(report2.elite_node_ids[1], n1_id);
    }

    #[test]
    fn energy_threshold_is_minimum_elite_energy() {
        let (storage, _dir) = open_temp_db();

        let n1 = make_node(0.1, 0.0, 0.3);
        let n2 = make_node(0.0, 0.1, 0.6);
        let n3 = make_node(0.1, 0.1, 0.9);

        storage.put_node(&n1).unwrap();
        storage.put_node(&n2).unwrap();
        storage.put_node(&n3).unwrap();

        let cfg = ZaratustraConfig {
            ubermensch_top_fraction: 0.34, // ceil(3*0.34) = ceil(1.02) = 2
            ..default_config()
        };
        let report = run_ubermensch(&storage, &cfg).unwrap();
        assert_eq!(report.elite_count, 2);

        // Threshold should be the energy of the least-energetic elite node
        // Ranked: 0.9, 0.6, 0.3 → elite = [0.9, 0.6] → threshold = 0.6
        assert!((report.energy_threshold - 0.6).abs() < 1e-5);
    }

    #[test]
    fn mean_elite_and_base_energy_computed_correctly() {
        let (storage, _dir) = open_temp_db();

        let n1 = make_node(0.1, 0.0, 0.2);
        let n2 = make_node(0.0, 0.1, 0.4);
        let n3 = make_node(0.1, 0.1, 0.8);
        let n4 = make_node(0.15, 0.0, 1.0);

        storage.put_node(&n1).unwrap();
        storage.put_node(&n2).unwrap();
        storage.put_node(&n3).unwrap();
        storage.put_node(&n4).unwrap();

        let cfg = ZaratustraConfig {
            ubermensch_top_fraction: 0.50, // ceil(4*0.5) = 2 elite
            ..default_config()
        };
        let report = run_ubermensch(&storage, &cfg).unwrap();

        // Elite: [1.0, 0.8], base: [0.4, 0.2]
        assert!((report.mean_elite_energy - 0.9).abs() < 1e-5);
        assert!((report.mean_base_energy - 0.3).abs() < 1e-5);
    }

    #[test]
    fn all_nodes_same_energy() {
        let (storage, _dir) = open_temp_db();
        for i in 0..5 {
            let node = make_node(0.01 * (i + 1) as f32, 0.0, 0.5);
            storage.put_node(&node).unwrap();
        }

        let cfg = ZaratustraConfig {
            ubermensch_top_fraction: 0.20, // ceil(5*0.2) = 1
            ..default_config()
        };
        let report = run_ubermensch(&storage, &cfg).unwrap();

        assert_eq!(report.elite_count, 1);
        assert!((report.energy_threshold - 0.5).abs() < 1e-5);
        assert!((report.mean_elite_energy - 0.5).abs() < 1e-5);
        assert!((report.mean_base_energy - 0.5).abs() < 1e-5);
    }
}
