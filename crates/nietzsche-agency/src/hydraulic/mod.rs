//! Phase XVII — Hydraulic Flow Engine for NietzscheDB.
//!
//! Transforms the Poincaré knowledge graph from a static geometric archive
//! into a **self-eroding computational organism** where information flows
//! along paths of least resistance, carved by repeated use.
//!
//! ## Three Primitives
//!
//! 1. **FlowLedger** — measures real CPU cost per edge traversal (ATP meter)
//! 2. **ConductivityTensor** — adaptive metric that shortens frequently-used paths
//! 3. **MurrayRebalancer** — fractal equilibrium of "vessel diameters" during sleep
//!
//! ## Emergent Behavior
//!
//! The combination of measurement (FlowLedger), adaptation (conductivity),
//! and equilibrium (Murray) causes Bejan's Constructal Law to emerge:
//! the graph's conductivity network self-organizes into a dendritic fractal
//! that minimizes total flow resistance.
//!
//! ## Integration Points
//!
//! - **FlowLedger**: called on every edge traversal (NQL, DIFFUSE, daemon scans)
//! - **Conductivity**: updated by Hebbian LTP (Phase XII.5) and flow analysis
//! - **Murray**: runs during SleepCycle (reconsolidation phase)

pub mod flow_ledger;
pub mod conductivity;
pub mod murray;
pub mod report;
