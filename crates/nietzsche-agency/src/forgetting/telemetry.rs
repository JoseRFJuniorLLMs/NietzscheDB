//! Telemetry — The blood of numbers from the Forgetting Engine.
//!
//! Outputs CSV data tracking all vital signs across cycles for
//! plotting and analysis. This is the evidence that proves or
//! disproves the metabolic hypothesis.

use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

/// Complete telemetry snapshot for one forgetting cycle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CycleTelemetry {
    pub cycle: u64,
    pub total_nodes: usize,
    pub sacrificed_this_cycle: usize,
    pub condemned_count: usize,
    pub toxic_count: usize,
    pub ricci_shielded_count: usize,
    pub sacred_count: usize,
    pub dormant_count: usize,
    pub mean_vitality: f32,
    pub vitality_variance: f32,
    pub mean_energy: f32,
    pub tgc_intrinsic: f32,
    pub tgc_combined: f32,
    pub elite_drift: f32,
    pub voids_available: usize,
    pub gaming_suspected: bool,
    pub false_positive_count: usize, // Signal nodes incorrectly killed
}

/// CSV writer for forgetting telemetry.
pub struct TelemetryWriter {
    path: String,
    header_written: bool,
}

impl TelemetryWriter {
    /// Create a new telemetry writer. Truncates the file if it exists.
    pub fn new(path: &str) -> std::io::Result<Self> {
        let mut writer = Self {
            path: path.to_string(),
            header_written: false,
        };
        writer.write_header()?;
        Ok(writer)
    }

    fn write_header(&mut self) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)?;

        writeln!(file, "cycle,total_nodes,sacrificed,condemned,toxic,ricci_shielded,sacred,dormant,mean_vitality,vitality_variance,mean_energy,tgc_intrinsic,tgc_combined,elite_drift,voids,gaming_suspected,false_positives")?;
        self.header_written = true;
        Ok(())
    }

    /// Append a telemetry record to the CSV file.
    pub fn write_record(&self, t: &CycleTelemetry) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .append(true)
            .open(&self.path)?;

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{}",
            t.cycle,
            t.total_nodes,
            t.sacrificed_this_cycle,
            t.condemned_count,
            t.toxic_count,
            t.ricci_shielded_count,
            t.sacred_count,
            t.dormant_count,
            t.mean_vitality,
            t.vitality_variance,
            t.mean_energy,
            t.tgc_intrinsic,
            t.tgc_combined,
            t.elite_drift,
            t.voids_available,
            if t.gaming_suspected { 1 } else { 0 },
            t.false_positive_count,
        )?;
        Ok(())
    }

    /// Get the path to the telemetry file.
    pub fn path(&self) -> &str {
        &self.path
    }
}

/// Compact one-line summary for console output.
pub fn format_cycle_summary(t: &CycleTelemetry) -> String {
    format!(
        "Cycle {:04} | Nodes: {:5} | Killed: {:4} (C:{} T:{} RS:{}) | V_mean: {:.3} V_var: {:.4} | TGC: {:.3} | Drift: {:.3} | FP: {}",
        t.cycle,
        t.total_nodes,
        t.sacrificed_this_cycle,
        t.condemned_count,
        t.toxic_count,
        t.ricci_shielded_count,
        t.mean_vitality,
        t.vitality_variance,
        t.tgc_combined,
        t.elite_drift,
        t.false_positive_count,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn telemetry_writer_creates_csv() {
        let path = "test_telemetry_forgetting.csv";
        {
            let writer = TelemetryWriter::new(path).unwrap();
            writer.write_record(&CycleTelemetry {
                cycle: 1,
                total_nodes: 5000,
                sacrificed_this_cycle: 100,
                condemned_count: 80,
                toxic_count: 20,
                mean_vitality: 0.654,
                vitality_variance: 0.0321,
                mean_energy: 0.45,
                tgc_combined: 0.23,
                elite_drift: 0.05,
                ..Default::default()
            }).unwrap();

            writer.write_record(&CycleTelemetry {
                cycle: 2,
                total_nodes: 4900,
                sacrificed_this_cycle: 50,
                ..Default::default()
            }).unwrap();
        }

        let content = fs::read_to_string(path).unwrap();
        assert!(content.starts_with("cycle,total_nodes"));
        assert!(content.contains("5000"));
        assert!(content.contains("4900"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn format_summary() {
        let t = CycleTelemetry {
            cycle: 42,
            total_nodes: 3500,
            sacrificed_this_cycle: 75,
            condemned_count: 60,
            toxic_count: 15,
            ricci_shielded_count: 5,
            mean_vitality: 0.678,
            vitality_variance: 0.0234,
            tgc_combined: 0.345,
            elite_drift: 0.067,
            ..Default::default()
        };
        let summary = format_cycle_summary(&t);
        assert!(summary.contains("Cycle 0042"));
        assert!(summary.contains("3500"));
    }
}
