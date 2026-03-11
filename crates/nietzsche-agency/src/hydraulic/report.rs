//! Reports for the Hydraulic Flow Engine.
//!
//! These structs are added to `AgencyTickReport` and `SleepCycleReport`
//! for observability and dashboard integration.

use uuid::Uuid;

/// Report from a flow analysis pass (Phase XVII of agency tick).
#[derive(Debug)]
pub struct FlowReport {
    /// Total traversals this epoch.
    pub total_traversals: u64,
    /// Mean CPU cost per traversal (nanoseconds).
    pub mean_cpu_ns: f64,
    /// Constructal flow energy E = Σ(f²/κ). Lower = more efficient.
    pub constructal_energy: f64,
    /// Change in E_flow since last epoch (negative = improving).
    pub delta_energy: f64,
    /// Number of conductivity updates emitted.
    pub conductivity_updates: usize,
    /// Number of shortcut proposals emitted.
    pub shortcuts_proposed: usize,
    /// Top-K bottleneck edges (highest pressure).
    pub bottlenecks: Vec<(Uuid, f64)>,
    /// Top-K highway edges (highest flow rate).
    pub highways: Vec<(Uuid, f64)>,
    /// Murray compliance score [0, 1] (1 = perfect fractal).
    pub murray_compliance: f64,
}
