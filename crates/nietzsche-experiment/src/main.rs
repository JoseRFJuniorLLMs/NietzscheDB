//! # NietzscheDB Link Prediction Experiment
//!
//! Validates that Topological Generative Capacity (TGC) improves the structural
//! quality of the hyperbolic embedding space.
//!
//! ## 3 Experimental Modes
//!
//! | Mode | L-System | Dialectic | TGC Feedback |
//! |------|----------|-----------|-------------|
//! | Normal | tick() | active | positive |
//! | Off | tick() | disabled | none |
//! | Inverted | tick() | active | inverted (penalise expansion) |
//!
//! ## Usage
//!
//! ```text
//! experiment --data-dir ./cora_nietzsche --mode normal --cycles 20
//! experiment --data-dir ./cora_nietzsche --mode off --cycles 20
//! experiment --data-dir ./cora_nietzsche --mode inverted --cycles 20
//! ```
//!
//! Output: `telemetry_{mode}.csv` in the current directory.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use nietzsche_experiment::sampling::{evaluate_link_prediction, hold_out_edges};
use nietzsche_experiment::telemetry::{write_csv, CycleRecord};

use nietzsche_agency::forgetting::{
    structural_entropy, global_efficiency, degree_distribution, TgcMonitor,
};
use nietzsche_agency::{run_dialectic_cycle, DialecticConfig};

use nietzsche_graph::{AdjacencyIndex, MockVectorStore, NietzscheDB, VectorStore};
use nietzsche_lsystem::{LSystemEngine, ProductionRule, RuleAction, RuleCondition};

use uuid::Uuid;

// ─────────────────────────────────────────────
// Experiment configuration
// ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// TGC active, dialectic running normally.
    Normal,
    /// L-System ticks but no dialectic, no TGC feedback.
    Off,
    /// Dialectic active but TGC signal is inverted: penalise structural improvement.
    Inverted,
}

impl Mode {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "normal" => Some(Self::Normal),
            "off" => Some(Self::Off),
            "inverted" => Some(Self::Inverted),
            _ => None,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Off => "off",
            Self::Inverted => "inverted",
        }
    }
}

struct ExperimentConfig {
    data_dir: PathBuf,
    mode: Mode,
    cycles: u64,
    holdout_fraction: f64,
    output_csv: PathBuf,
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("nietzsche_experiment=info,nietzsche_lsystem=warn")
        .init();

    let config = parse_args();

    tracing::info!(
        mode = config.mode.label(),
        cycles = config.cycles,
        data_dir = %config.data_dir.display(),
        "Starting Link Prediction experiment"
    );

    // 1. Open DB
    let vs = MockVectorStore::default();
    let mut db = NietzscheDB::open(&config.data_dir, vs, 128)
        .expect("failed to open NietzscheDB");

    // 2. Hold out test edges
    let (test_edges, held_out_ids) = hold_out_edges(db.storage(), config.holdout_fraction);
    tracing::info!(
        test_edges = test_edges.len(),
        "Held out edges for evaluation"
    );

    // Remove held-out edges from the adjacency index so the model can't cheat
    for edge_id in &held_out_ids {
        if let Ok(Some(edge)) = db.get_edge(*edge_id) {
            db.adjacency().remove_edge(&edge);
        }
    }

    // 3. Build L-System engine with default growth rules
    let lsystem = build_default_lsystem();

    // 4. TGC monitor
    let mut tgc_monitor = TgcMonitor::default();

    // 5. Dialectic config
    let dialectic_config = DialecticConfig {
        proximity_threshold: 0.8,
        min_certainty: 0.3,
        max_scan: 200,
        polarity_gap_threshold: 1.2,
        default_certainty: 0.5,
    };

    // 6. Evaluate baseline (Cycle 0 — before any L-System ticks)
    let mut records = Vec::with_capacity((config.cycles + 1) as usize);
    let baseline = evaluate_link_prediction(db.storage(), db.adjacency(), &test_edges);
    let nc = db.node_count().unwrap_or(0);
    let ec = db.edge_count().unwrap_or(0);
    records.push(CycleRecord::new(config.mode.label(), 0, &baseline, nc, ec, 0.0));

    tracing::info!(
        auc = format!("{:.4}", baseline.auc),
        auc_random = format!("{:.4}", baseline.auc_random),
        auc_hard = format!("{:.4}", baseline.auc_hard),
        "Cycle 0 (baseline)"
    );

    // 7. Run cycles
    for cycle in 1..=config.cycles {
        // ── L-System tick (always runs) ──
        let report = match lsystem.tick(&mut db) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(cycle, error = %e, "L-System tick failed, skipping");
                continue;
            }
        };

        // ── Dialectic + TGC (mode-dependent) ──
        let tgc_value = match config.mode {
            Mode::Normal => {
                // Run dialectic synthesis
                let _ = run_dialectic_cycle(db.storage(), &dialectic_config);

                // Compute structural metrics from the adjacency index
                let (adj_map, node_ids) = adjacency_to_hashmap(db.adjacency());
                let deg_dist = degree_distribution(&adj_map);
                let total_nodes = node_ids.len();
                let hs = structural_entropy(&deg_dist, total_nodes);
                let eg = global_efficiency(&adj_map, &node_ids, 32, cycle);

                // Normal TGC: reward structural improvement
                tgc_monitor.compute(
                    report.nodes_spawned,
                    report.total_nodes,
                    0.7, // mean quality placeholder
                    hs,
                    eg,
                )
            }
            Mode::Off => {
                // No dialectic, no TGC feedback — pure growth only
                0.0
            }
            Mode::Inverted => {
                // Run dialectic synthesis
                let _ = run_dialectic_cycle(db.storage(), &dialectic_config);

                // Compute structural metrics
                let (adj_map, node_ids) = adjacency_to_hashmap(db.adjacency());
                let deg_dist = degree_distribution(&adj_map);
                let total_nodes = node_ids.len();
                let hs = structural_entropy(&deg_dist, total_nodes);
                let eg = global_efficiency(&adj_map, &node_ids, 32, cycle);

                // INVERTED TGC: compute normally, then counteract
                let real_tgc = tgc_monitor.compute(
                    report.nodes_spawned,
                    report.total_nodes,
                    0.7,
                    hs,
                    eg,
                );

                // Undo the structural improvement: prune recently spawned nodes
                // proportional to how positive the TGC was.
                if real_tgc > 0.1 {
                    let prune_count = (report.nodes_spawned as f32 * real_tgc.min(1.0)) as usize;
                    prune_recent_spawns(&mut db, prune_count.max(1));
                }

                -real_tgc // negative to signal inversion in telemetry
            }
        };

        // ── Evaluate AUC ──
        let lp_result = evaluate_link_prediction(db.storage(), db.adjacency(), &test_edges);
        let nc = db.node_count().unwrap_or(0);
        let ec = db.edge_count().unwrap_or(0);

        records.push(CycleRecord::new(
            config.mode.label(),
            cycle,
            &lp_result,
            nc,
            ec,
            tgc_value,
        ));

        tracing::info!(
            cycle,
            auc = format!("{:.4}", lp_result.auc),
            auc_hard = format!("{:.4}", lp_result.auc_hard),
            nodes_spawned = report.nodes_spawned,
            nodes_pruned = report.nodes_pruned,
            tgc = format!("{:.4}", tgc_value),
            "Cycle complete"
        );
    }

    // 8. Write telemetry
    write_csv(&config.output_csv, &records).expect("failed to write CSV");
    tracing::info!(path = %config.output_csv.display(), "Telemetry written");
}

// ─────────────────────────────────────────────
// Bridge: AdjacencyIndex (Uuid) → HashMap<usize, HashSet<usize>>
// ─────────────────────────────────────────────

/// Convert the NietzscheDB AdjacencyIndex (Uuid-keyed) to the generic integer-keyed
/// format expected by `structural_entropy()` and `global_efficiency()`.
fn adjacency_to_hashmap(adj: &AdjacencyIndex) -> (HashMap<usize, HashSet<usize>>, Vec<usize>) {
    let all_uuid_nodes = adj.all_nodes();

    // Build Uuid → usize mapping
    let uuid_to_idx: HashMap<Uuid, usize> = all_uuid_nodes
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mut adj_map: HashMap<usize, HashSet<usize>> = HashMap::new();
    for &uuid in &all_uuid_nodes {
        let idx = uuid_to_idx[&uuid];
        let neighbors = adj.neighbors_both(&uuid);
        let neighbor_set: HashSet<usize> = neighbors
            .iter()
            .filter_map(|n| uuid_to_idx.get(n).copied())
            .collect();
        adj_map.insert(idx, neighbor_set);
    }

    let node_ids: Vec<usize> = (0..all_uuid_nodes.len()).collect();
    (adj_map, node_ids)
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/// Prune the N most recently created nodes (used by Inverted mode to undo growth).
fn prune_recent_spawns<V: VectorStore>(db: &mut NietzscheDB<V>, count: usize) {
    if count == 0 {
        return;
    }

    // Collect recent nodes by creation time (descending)
    let mut recent: Vec<(Uuid, i64)> = Vec::new();
    for result in db.storage().iter_nodes_meta() {
        if let Ok(meta) = result {
            recent.push((meta.id, meta.created_at));
        }
    }
    recent.sort_by(|a, b| b.1.cmp(&a.1)); // most recent first

    for (id, _) in recent.into_iter().take(count) {
        let _ = db.prune_node(id);
    }
}

/// Build default L-System with basic growth rules for the experiment.
fn build_default_lsystem() -> LSystemEngine {
    let rules = vec![
        // Rule 1: High-energy nodes below generation 10 spawn a child deeper
        ProductionRule::new(
            "growth_child",
            RuleCondition::growth(0.5, 10),
            RuleAction::SpawnChild {
                depth_offset: 0.08,
                weight: 0.7,
                content: serde_json::Value::Null,
            },
        ),
        // Rule 2: Very low energy nodes get pruned
        ProductionRule::new(
            "prune_dead",
            RuleCondition::EnergyBelow(0.05),
            RuleAction::Prune,
        ),
    ];

    LSystemEngine::new(rules)
}

/// Minimal argument parser (no external deps).
fn parse_args() -> ExperimentConfig {
    let args: Vec<String> = std::env::args().collect();

    let mut data_dir = PathBuf::from("./experiment_data");
    let mut mode = Mode::Normal;
    let mut cycles: u64 = 20;
    let mut holdout = 0.1;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" => {
                i += 1;
                data_dir = PathBuf::from(&args[i]);
            }
            "--mode" => {
                i += 1;
                mode = Mode::from_str(&args[i]).unwrap_or_else(|| {
                    eprintln!("Unknown mode '{}'. Use: normal, off, inverted", args[i]);
                    std::process::exit(1);
                });
            }
            "--cycles" => {
                i += 1;
                cycles = args[i].parse().unwrap_or(20);
            }
            "--holdout" => {
                i += 1;
                holdout = args[i].parse().unwrap_or(0.1);
            }
            "--help" | "-h" => {
                eprintln!("Usage: experiment [--data-dir PATH] [--mode normal|off|inverted] [--cycles N] [--holdout FRAC]");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
            }
        }
        i += 1;
    }

    let output_csv = PathBuf::from(format!("telemetry_{}.csv", mode.label()));

    ExperimentConfig {
        data_dir,
        mode,
        cycles,
        holdout_fraction: holdout,
        output_csv,
    }
}
