//! MurrayRebalancer — fractal vascular equilibrium during sleep.
//!
//! Applies Murray's Law (1926) to the conductivity network:
//!
//! ```text
//! κ_parent³ = Σ κ_child_i³
//! ```
//!
//! This ensures the "vessel network" forms a fractal tree that minimizes
//! total flow resistance — the same shape as rivers, lungs, and blood vessels.
//!
//! ## When It Runs
//!
//! Exclusively during the SleepCycle (REM phase), between Riemannian
//! re-optimization and Hausdorff adjustment. Mirrors biology: vascular
//! remodeling happens during rest, not during active cognition.

use uuid::Uuid;

use super::flow_ledger::FlowLedger;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for Murray's Law rebalancing.
#[derive(Debug, Clone)]
pub struct MurrayConfig {
    /// Damping factor α ∈ (0, 1].
    /// 0.1 = slow/stable (production). 1.0 = instant (testing).
    pub damping: f32,
    /// Minimum outgoing edges to qualify as a "branch point".
    pub min_fanout: usize,
    /// Maximum conductivity change per sleep cycle.
    pub max_delta: f32,
    /// Also balance incoming edges (bidirectional Murray).
    pub bidirectional: bool,
    /// Maximum nodes to process per sleep cycle.
    pub max_nodes: usize,
    /// Whether Murray rebalancing is enabled.
    pub enabled: bool,
}

impl Default for MurrayConfig {
    fn default() -> Self {
        Self {
            damping: 0.1,
            min_fanout: 2,
            max_delta: 0.5,
            bidirectional: false,
            max_nodes: 5000,
            enabled: true,
        }
    }
}

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

/// A single conductivity adjustment from Murray rebalancing.
#[derive(Debug, Clone)]
pub struct MurrayDelta {
    pub edge_id: Uuid,
    pub old_conductivity: f32,
    pub new_conductivity: f32,
    pub murray_target: f32,
    pub child_count: usize,
}

/// Report from a Murray rebalancing pass.
#[derive(Debug)]
pub struct MurrayReport {
    /// Branch points evaluated.
    pub branches_evaluated: usize,
    /// Edges whose conductivity was adjusted.
    pub edges_adjusted: usize,
    /// Mean Murray deviation before adjustment (0 = perfect fractal).
    pub mean_deviation_before: f64,
    /// Mean Murray deviation after adjustment.
    pub mean_deviation_after: f64,
    /// Top-K largest adjustments.
    pub top_adjustments: Vec<MurrayDelta>,
    /// Global Murray compliance score ∈ [0, 1] (1 = perfect fractal).
    pub compliance_score: f64,
}

// ─────────────────────────────────────────────
// MurrayRebalancer
// ─────────────────────────────────────────────

/// Fractal vascular equilibrium engine.
///
/// For each node with outgoing edges (a "branch point"):
/// 1. Compute the cubic sum of children conductivities
/// 2. Adjust incoming edge conductivity to match Murray's Law
/// 3. Apply damping to prevent oscillation
pub struct MurrayRebalancer {
    pub config: MurrayConfig,
}

/// A branch point: incoming edge + outgoing edges with conductivities.
pub struct BranchPoint {
    /// The edge feeding into this branch point (parent → node).
    pub incoming_edge_id: Uuid,
    pub incoming_conductivity: f32,
    /// Outgoing edges (node → children) with their conductivities.
    pub children: Vec<ChildEdge>,
}

pub struct ChildEdge {
    pub edge_id: Uuid,
    pub conductivity: f32,
}

impl MurrayRebalancer {
    pub fn new(config: MurrayConfig) -> Self {
        Self { config }
    }

    /// Rebalance a single branch point using standard Murray's Law.
    ///
    /// Returns None if no adjustment needed or node doesn't qualify.
    pub fn rebalance_branch(&self, branch: &BranchPoint) -> Option<MurrayDelta> {
        if branch.children.len() < self.config.min_fanout {
            return None;
        }

        // Step 1: Cubic sum of children
        let child_cubic_sum: f64 = branch.children.iter()
            .map(|c| (c.conductivity as f64).powi(3))
            .sum();

        // Step 2: Murray target for parent
        let murray_target = child_cubic_sum.cbrt() as f32;

        // Step 3: Damped adjustment
        let current = branch.incoming_conductivity;
        let delta = (murray_target - current) * self.config.damping;
        let clamped_delta = delta.clamp(-self.config.max_delta, self.config.max_delta);
        let new_conductivity = (current + clamped_delta).clamp(0.01, 10.0);

        if (new_conductivity - current).abs() < 1e-6 {
            return None;
        }

        Some(MurrayDelta {
            edge_id: branch.incoming_edge_id,
            old_conductivity: current,
            new_conductivity,
            murray_target,
            child_count: branch.children.len(),
        })
    }

    /// Flow-weighted Murray: children with more traffic get more "diameter".
    ///
    /// κ_parent³ = Σ κ_child_i³ × (f_i / f_mean)
    pub fn rebalance_branch_flow_weighted(
        &self,
        branch: &BranchPoint,
        ledger: &FlowLedger,
    ) -> Option<MurrayDelta> {
        if branch.children.len() < self.config.min_fanout {
            return None;
        }

        // Get flow rates
        let flows: Vec<f64> = branch.children.iter()
            .map(|c| ledger.flow_rate(c.edge_id).max(1e-9))
            .collect();
        let mean_flow = flows.iter().sum::<f64>() / flows.len() as f64;
        if mean_flow < 1e-9 {
            // No flow data — fall back to standard Murray
            return self.rebalance_branch(branch);
        }

        // Flow-weighted cubic sum
        let child_cubic_sum: f64 = branch.children.iter()
            .zip(flows.iter())
            .map(|(c, &f)| {
                let flow_weight = f / mean_flow;
                (c.conductivity as f64).powi(3) * flow_weight
            })
            .sum();

        let murray_target = child_cubic_sum.cbrt() as f32;
        let current = branch.incoming_conductivity;
        let delta = (murray_target - current) * self.config.damping;
        let clamped_delta = delta.clamp(-self.config.max_delta, self.config.max_delta);
        let new_conductivity = (current + clamped_delta).clamp(0.01, 10.0);

        if (new_conductivity - current).abs() < 1e-6 {
            return None;
        }

        Some(MurrayDelta {
            edge_id: branch.incoming_edge_id,
            old_conductivity: current,
            new_conductivity,
            murray_target,
            child_count: branch.children.len(),
        })
    }

    /// Run Murray rebalancing over a set of branch points.
    ///
    /// Returns a report with all deltas and compliance metrics.
    pub fn run(
        &self,
        branches: &[BranchPoint],
        ledger: Option<&FlowLedger>,
    ) -> MurrayReport {
        if !self.config.enabled {
            return MurrayReport {
                branches_evaluated: 0,
                edges_adjusted: 0,
                mean_deviation_before: 0.0,
                mean_deviation_after: 0.0,
                top_adjustments: Vec::new(),
                compliance_score: 1.0,
            };
        }

        let max = branches.len().min(self.config.max_nodes);
        let mut deltas = Vec::new();
        let mut deviations_before = Vec::new();

        for branch in branches.iter().take(max) {
            if branch.children.len() < self.config.min_fanout {
                continue;
            }

            // Compute deviation before
            let child_cubic_sum: f64 = branch.children.iter()
                .map(|c| (c.conductivity as f64).powi(3))
                .sum();
            let parent_cubic = (branch.incoming_conductivity as f64).powi(3);
            let deviation = if parent_cubic > 1e-9 {
                ((parent_cubic - child_cubic_sum) / parent_cubic).abs()
            } else {
                1.0
            };
            deviations_before.push(deviation);

            // Compute delta
            let delta = if let Some(l) = ledger {
                self.rebalance_branch_flow_weighted(branch, l)
            } else {
                self.rebalance_branch(branch)
            };

            if let Some(d) = delta {
                deltas.push(d);
            }
        }

        let mean_dev_before = if deviations_before.is_empty() {
            0.0
        } else {
            deviations_before.iter().sum::<f64>() / deviations_before.len() as f64
        };

        // Compliance = 1 - mean_deviation
        let compliance = (1.0 - mean_dev_before).max(0.0).min(1.0);

        // Compute mean deviation after (simulate adjustments)
        let mean_dev_after = if deltas.is_empty() {
            mean_dev_before
        } else {
            // Rough estimate: each delta reduces its branch's deviation by damping factor
            mean_dev_before * (1.0 - self.config.damping as f64)
        };

        let edges_adjusted = deltas.len();

        // Keep top-10 largest adjustments
        deltas.sort_by(|a, b| {
            let da = (a.new_conductivity - a.old_conductivity).abs();
            let db = (b.new_conductivity - b.old_conductivity).abs();
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
        let top = deltas.iter().take(10).cloned().collect();

        MurrayReport {
            branches_evaluated: max.min(deviations_before.len()),
            edges_adjusted,
            mean_deviation_before: mean_dev_before,
            mean_deviation_after: mean_dev_after,
            top_adjustments: top,
            compliance_score: compliance,
        }
    }
}

/// Compute Murray compliance score for a set of branch points.
///
/// Score ∈ [0, 1] where 1.0 = perfect fractal (all branches satisfy Murray's Law).
pub fn murray_compliance(branches: &[BranchPoint]) -> f64 {
    if branches.is_empty() {
        return 1.0;
    }

    let qualifying: Vec<&BranchPoint> = branches.iter()
        .filter(|b| b.children.len() >= 2)
        .collect();

    if qualifying.is_empty() {
        return 1.0;
    }

    let total_deviation: f64 = qualifying.iter()
        .map(|b| {
            let child_cubic_sum: f64 = b.children.iter()
                .map(|c| (c.conductivity as f64).powi(3))
                .sum();
            let parent_cubic = (b.incoming_conductivity as f64).powi(3);
            if parent_cubic > 1e-9 {
                ((parent_cubic - child_cubic_sum) / parent_cubic).abs()
            } else {
                1.0
            }
        })
        .sum();

    let mean_deviation = total_deviation / qualifying.len() as f64;
    (1.0 - mean_deviation).max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_branch(incoming_k: f32, children_k: &[f32]) -> BranchPoint {
        BranchPoint {
            incoming_edge_id: Uuid::new_v4(),
            incoming_conductivity: incoming_k,
            children: children_k.iter().map(|&k| ChildEdge {
                edge_id: Uuid::new_v4(),
                conductivity: k,
            }).collect(),
        }
    }

    #[test]
    fn test_perfect_murray_no_adjustment() {
        // If κ_parent³ = Σ κ_child_i³, no adjustment needed
        // 2³ = 8, children: cbrt(4)³ + cbrt(4)³ = 4 + 4 = 8 ✓
        let k_child = 4.0_f64.cbrt() as f32; // ≈ 1.587
        let branch = make_branch(2.0, &[k_child, k_child]);

        let rebalancer = MurrayRebalancer::new(MurrayConfig::default());
        let delta = rebalancer.rebalance_branch(&branch);

        // Should be very close to equilibrium (small or no delta)
        match delta {
            None => {} // Perfect — no adjustment needed
            Some(d) => {
                assert!((d.new_conductivity - d.old_conductivity).abs() < 0.1);
            }
        }
    }

    #[test]
    fn test_murray_adjusts_oversized_parent() {
        // Parent κ=5.0, children κ=[1.0, 1.0]
        // Murray target: (1³ + 1³)^(1/3) = 2^(1/3) ≈ 1.26
        // Parent is way too big → should decrease
        let branch = make_branch(5.0, &[1.0, 1.0]);

        let rebalancer = MurrayRebalancer::new(MurrayConfig {
            damping: 0.5,
            ..MurrayConfig::default()
        });
        let delta = rebalancer.rebalance_branch(&branch).unwrap();

        assert!(delta.new_conductivity < 5.0, "Parent should decrease");
        assert!(delta.murray_target < 2.0, "Target should be ~1.26");
    }

    #[test]
    fn test_murray_adjusts_undersized_parent() {
        // Parent κ=0.5, children κ=[3.0, 3.0, 3.0]
        // Murray target: (27+27+27)^(1/3) = 81^(1/3) ≈ 4.33
        // Parent is too small → should increase
        let branch = make_branch(0.5, &[3.0, 3.0, 3.0]);

        let rebalancer = MurrayRebalancer::new(MurrayConfig {
            damping: 0.5,
            ..MurrayConfig::default()
        });
        let delta = rebalancer.rebalance_branch(&branch).unwrap();

        assert!(delta.new_conductivity > 0.5, "Parent should increase");
    }

    #[test]
    fn test_murray_skips_low_fanout() {
        let branch = make_branch(1.0, &[1.0]); // Only 1 child
        let rebalancer = MurrayRebalancer::new(MurrayConfig::default());
        assert!(rebalancer.rebalance_branch(&branch).is_none());
    }

    #[test]
    fn test_murray_compliance_perfect() {
        // Create branches that satisfy Murray's Law
        let k = 2.0_f64.cbrt() as f32;
        let branches = vec![make_branch(k, &[1.0, 1.0])];
        let score = murray_compliance(&branches);
        assert!(score > 0.95, "Perfect Murray should have high compliance: {}", score);
    }

    #[test]
    fn test_murray_compliance_poor() {
        // Parent way off from Murray target
        let branches = vec![make_branch(10.0, &[0.1, 0.1])];
        let score = murray_compliance(&branches);
        assert!(score < 0.5, "Poor Murray should have low compliance: {}", score);
    }

    #[test]
    fn test_run_produces_report() {
        let branches = vec![
            make_branch(5.0, &[1.0, 1.0]),
            make_branch(0.5, &[3.0, 3.0]),
            make_branch(1.0, &[0.5]), // Skipped (low fanout)
        ];

        let rebalancer = MurrayRebalancer::new(MurrayConfig::default());
        let report = rebalancer.run(&branches, None);

        assert_eq!(report.branches_evaluated, 2); // 3rd skipped
        assert!(report.edges_adjusted > 0);
        assert!(report.compliance_score >= 0.0 && report.compliance_score <= 1.0);
    }
}
