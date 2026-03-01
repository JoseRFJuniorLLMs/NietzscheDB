//! Telemetry output for the experiment: CSV writer for per-cycle metrics.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::sampling::LinkPredictionResult;

/// One row in the telemetry CSV.
#[derive(Debug, Clone)]
pub struct CycleRecord {
    pub mode: String,
    pub cycle: u64,
    pub auc: f64,
    pub auc_random: f64,
    pub auc_hard: f64,
    pub edges_evaluated: usize,
    pub edges_skipped: usize,
    pub node_count: usize,
    pub edge_count: usize,
    pub tgc: f32,
}

impl CycleRecord {
    pub fn new(
        mode: &str,
        cycle: u64,
        lp: &LinkPredictionResult,
        node_count: usize,
        edge_count: usize,
        tgc: f32,
    ) -> Self {
        Self {
            mode: mode.to_string(),
            cycle,
            auc: lp.auc,
            auc_random: lp.auc_random,
            auc_hard: lp.auc_hard,
            edges_evaluated: lp.edges_evaluated,
            edges_skipped: lp.edges_skipped,
            node_count,
            edge_count,
            tgc,
        }
    }
}

/// Write a full experiment run to CSV.
pub fn write_csv(path: &Path, records: &[CycleRecord]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "mode,cycle,auc,auc_random,auc_hard,edges_evaluated,edges_skipped,node_count,edge_count,tgc"
    )?;

    for r in records {
        writeln!(
            w,
            "{},{},{:.6},{:.6},{:.6},{},{},{},{},{:.6}",
            r.mode,
            r.cycle,
            r.auc,
            r.auc_random,
            r.auc_hard,
            r.edges_evaluated,
            r.edges_skipped,
            r.node_count,
            r.edge_count,
            r.tgc,
        )?;
    }

    w.flush()
}
