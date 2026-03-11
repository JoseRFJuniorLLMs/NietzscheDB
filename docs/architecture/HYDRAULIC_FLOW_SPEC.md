# Hydraulic Flow Engine — Technical Specification

> **Module**: `nietzsche-agency` → Hydraulic Substrate (Layer 0)
> **Version**: 1.0
> **Date**: 2026-03-10
> **Status**: Design Complete — Pre-Implementation
> **Authors**: Architecture Committee (Claude + Arquiteto)
> **Depends on**: Phase XII (ECAN), Phase XII.5 (Hebbian LTP), Phase XIII (Thermodynamics), Phase XIV (Gravity)

---

## 1. Overview

The Hydraulic Flow Engine transforms NietzscheDB from a **static geometric archive** into a **self-eroding computational organism**. Information does not get "searched" — it **flows** along paths of least resistance, carved by repeated use and governed by thermodynamic cost.

### The Core Insight

The Poincaré ball provides excellent **static geometry** (hierarchy, depth, similarity). But biology doesn't use static maps — it uses **dynamic flow**. Neurons don't "search" for answers; signals propagate along myelinated axons toward regions of lower potential. Blood vessels don't have fixed diameters; they remodel according to Murray's Law to minimize pumping cost.

This spec adds three primitives that convert the Poincaré geometry into a **living terrain**:

| Primitive | Biological Analog | What It Does |
|---|---|---|
| **FlowLedger** | ATP metabolism | Measures real CPU cost per edge traversal |
| **ConductivityTensor** | Myelination | Adaptive metric that shortens frequently-used paths |
| **MurrayRebalancer** | Vascular remodeling | Fractal equilibrium of "vessel diameters" during sleep |

### What This Is NOT

- NOT a replacement for Poincaré distance (which remains the ground truth for similarity)
- NOT a query optimizer (it's a substrate that makes the graph itself optimize over time)
- NOT a new daemon (it integrates into existing phases: Hebbian, Thermodynamics, SleepCycle)

### Relationship to Existing Phases

```
         ┌─────────────────────────────────────────────────────────────┐
         │              EXISTING AGENCY ENGINE                         │
         │                                                             │
         │  Phase XII (ECAN)  ──→  attention bids  ──→  co-activation │
         │  Phase XII.5 (Hebbian) ──→  weight Δ  ──→  edge.weight    │
         │  Phase XIII (Thermo)  ──→  heat flow  ──→  energy Δ       │
         │  Phase XIV (Gravity)  ──→  force field  ──→  attraction   │
         │                                                             │
         ├─────────── NEW: HYDRAULIC FLOW ENGINE ─────────────────────┤
         │                                                             │
         │  FlowLedger  ←── measures ──  every traversal             │
         │       │                                                     │
         │       ▼                                                     │
         │  ConductivityTensor  ←── Hebbian LTP now writes here      │
         │       │                                                     │
         │       ▼                                                     │
         │  MurrayRebalancer  ←── runs during SleepCycle             │
         │       │                                                     │
         │       ▼                                                     │
         │  effective_distance() = poincaré_d / conductivity          │
         │  (used by HNSW, DIFFUSE, traversal, gravity)               │
         │                                                             │
         └─────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Stack

```
┌──────────────────────────────────────────────────────────────────────┐
│              LAYER 3: MURRAY REBALANCER (Sleep Phase)                │
│  Fractal equilibrium · r³ conservation · Vessel pruning/widening    │
├──────────────────────────────────────────────────────────────────────┤
│              LAYER 2: CONDUCTIVITY TENSOR (Agency Tick)              │
│  Per-edge adaptive metric · Hebbian → conductivity · Decay          │
├──────────────────────────────────────────────────────────────────────┤
│              LAYER 1: FLOW LEDGER (Query Path)                       │
│  Per-edge CPU cost · Traversal counter · Pressure gradient          │
├──────────────────────────────────────────────────────────────────────┤
│              LAYER 0: POINCARÉ SUBSTRATE (Existing)                  │
│  Hyperbolic distance · f32 coords · HNSW index                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1: FlowLedger

The **metabolic accounting system**. Every edge traversal registers its real computational cost, creating a map of where energy is spent.

### The Problem It Solves

Today, the Agency Engine has no concept of **actual cost**. The Gravity module computes force fields using Poincaré distance, but doesn't know that traversing edge A→B costs 400μs while B→C costs 2μs. Without cost measurement, there's no "pain signal" to drive optimization.

### Data Structure

```rust
use dashmap::DashMap;
use uuid::Uuid;
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-edge flow statistics. Lock-free, concurrent-safe.
///
/// This is the "ATP meter" of the graph. Every traversal deposits
/// a measurement. The SleepCycle and Agency tick consume these
/// statistics to decide where to erode shortcuts.
pub struct FlowLedger {
    /// edge_id → FlowStats
    stats: DashMap<Uuid, FlowStats>,
    /// Global traversal counter (monotonic, for epoch computation).
    global_traversals: AtomicU64,
    /// Epoch marker — reset at each SleepCycle to track per-cycle flow.
    epoch_start: AtomicU64,
}

/// Accumulated flow statistics for a single edge.
#[derive(Debug, Clone)]
pub struct FlowStats {
    /// Total number of traversals since last epoch reset.
    pub traversals: u64,
    /// Cumulative CPU time spent traversing this edge (nanoseconds).
    pub total_cpu_ns: u64,
    /// Mean CPU cost per traversal (derived: total_cpu_ns / traversals).
    pub mean_cpu_ns: f64,
    /// Peak CPU cost observed (single worst traversal).
    pub peak_cpu_ns: u64,
    /// Exponential moving average of CPU cost (α = 0.1).
    /// More responsive to recent trends than raw mean.
    pub ema_cpu_ns: f64,
    /// Last traversal timestamp (Unix nanos) — for decay computation.
    pub last_traversed_ns: u64,
}
```

### Key Methods

| Method | Signature | Behavior |
|---|---|---|
| `record` | `(edge_id: Uuid, cpu_ns: u64)` | Atomic increment of traversals + cpu_ns. Updates EMA. |
| `pressure` | `(edge_id: Uuid) -> f64` | Returns `ema_cpu_ns / global_mean_cpu_ns`. Values > 1.0 = bottleneck. |
| `flow_rate` | `(edge_id: Uuid) -> f64` | Returns `traversals / epoch_duration_secs`. Higher = busier route. |
| `drain_epoch` | `() -> FlowEpochReport` | Snapshots all stats, resets epoch counter. Called by SleepCycle. |
| `top_bottlenecks` | `(k: usize) -> Vec<(Uuid, f64)>` | Top-K edges by pressure (cost/traversal ratio). |
| `top_highways` | `(k: usize) -> Vec<(Uuid, f64)>` | Top-K edges by flow rate (most traversed). |
| `dormant_edges` | `(min_idle_secs: u64) -> Vec<Uuid>` | Edges not traversed for N seconds. |

### Recording Points (Where `record()` Is Called)

The FlowLedger hooks into **existing traversal paths** — no new query engine needed:

| Traversal Type | Where | How |
|---|---|---|
| **NQL edge traversal** | `executor.rs` — `MATCH (a)-[:Type]->(b)` | Wrap edge resolution in `Instant::now()` → `ledger.record()` |
| **DIFFUSE walk** | `executor.rs` — `DIFFUSE FROM` | Each hop records its cost |
| **HNSW neighbor scan** | `hnsw/lib.rs` — `search_layer()` | Each distance computation records per-edge cost |
| **Agency daemon scan** | `daemons/*.rs` — `get_neighbors()` | Daemon traversals are "unconscious" flow |
| **Cascade propagation** | `entanglement.rs` (future Quantum Kernel) | Collapse propagation records per-hop cost |

### Pressure Gradient Formula

The **Semantic Pressure** at node A relative to node B:

$$P(A \to B) = \frac{E_A - E_B}{d_{eff}(A, B)}$$

where:
- $E_A, E_B$ = node energies (existing `energy: f32` field)
- $d_{eff}$ = effective distance (Poincaré distance / conductivity)

Information flows from **high pressure to low pressure** — just like water, blood, or electricity. This gradient is already implicit in the DIFFUSE query (energy-biased traversal), but now it becomes explicit and measurable.

### Concurrency Design

- `DashMap` provides lock-free concurrent reads/writes per shard
- `AtomicU64` for global counters — no mutex needed
- `record()` is called on the **hot path** (every traversal), so it must be < 100ns
- EMA update: `ema = α * new_sample + (1 - α) * ema` — single f64 multiply

### Memory Budget

| Collection Size | Edges | FlowLedger RAM |
|---|---|---|
| 1K nodes | ~5K edges | ~400 KB |
| 10K nodes | ~50K edges | ~4 MB |
| 100K nodes | ~500K edges | ~40 MB |
| 1M nodes | ~5M edges | ~400 MB |

Each `FlowStats` = ~64 bytes. `DashMap` overhead ≈ 16 bytes/entry. Total ≈ 80 bytes/edge.

For collections > 100K edges, consider **sampling**: only record 1-in-N traversals (configurable via `AGENCY_FLOW_SAMPLE_RATE`).

---

## 4. Layer 2: ConductivityTensor

The **myelination layer**. Each edge carries an adaptive multiplier that modifies the effective metric, creating "wide highways" and "narrow trails".

### The Problem It Solves

Today, Poincaré distance is **immutable** — `d(A,B)` is always the same regardless of how many times the EVA traverses A→B. But in biology, myelin sheaths grow thicker with repeated activation, making signal propagation faster. The Hebbian LTP module (Phase XII.5) already strengthens `edge.weight`, but this doesn't affect distance calculations.

### New Field on Edge

```rust
// In nietzsche-graph/src/model.rs — Edge struct

/// Hydraulic conductivity of this edge.
///
/// Modifies effective distance: d_eff = d_poincaré / conductivity
///
/// - 1.0 = baseline (default, backward-compatible)
/// - > 1.0 = "myelinated" — frequently used, low resistance
/// - < 1.0 = "constricted" — rarely used, high resistance
/// - Range: [0.01, 10.0] (hard-clamped)
///
/// Updated by:
/// - Hebbian LTP (potentiation → conductivity increase)
/// - Temporal Decay (unused edges → conductivity decrease)
/// - Murray Rebalancer (fractal equilibrium during SleepCycle)
///
/// Legacy edges: 0.0 deserialized as 1.0 via serde(default).
#[serde(default = "default_conductivity")]
pub conductivity: f32,

fn default_conductivity() -> f32 { 1.0 }
```

### Backward Compatibility

- `#[serde(default = "default_conductivity")]` ensures all legacy edges deserialize with `conductivity = 1.0`
- Bincode: field is appended at end of struct. Old data → `0.0` via `#[serde(default)]` → mapped to `1.0` in `deserialize_edge_compat()`
- **V2 → V3 edge migration**: Same pattern as NodeMeta V1→V2. Try V3 first, fallback to V2 and inject `conductivity: 1.0`

### Effective Distance

The core equation that replaces raw Poincaré distance in all flow-sensitive operations:

$$d_{eff}(A, B) = \frac{d_{\mathbb{H}}(A, B)}{\kappa_{AB}}$$

where:
- $d_{\mathbb{H}}(A, B)$ = hyperbolic Poincaré distance (unchanged)
- $\kappa_{AB}$ = `edge.conductivity` ∈ [0.01, 10.0]

```rust
/// Compute effective distance accounting for edge conductivity.
///
/// This is the "myelinated" distance: well-trodden paths feel shorter.
/// Raw Poincaré distance is preserved for similarity queries.
#[inline]
pub fn effective_distance(
    poincare_dist: f64,
    conductivity: f32,
) -> f64 {
    poincare_dist / (conductivity.clamp(0.01, 10.0) as f64)
}
```

### Where Effective Distance Is Used

| Operation | Uses `d_eff`? | Rationale |
|---|---|---|
| **DIFFUSE walk** | YES | Flow follows myelinated paths |
| **Gravity force** | YES | Gravity pulls harder through conductive channels |
| **Heat flow** | YES | Fourier's law: `q = κ · ΔE / d_eff` |
| **HNSW KNN search** | NO | Similarity must remain pure geometric |
| **Coherence scoring** | NO | Structural integrity uses raw distance |
| **Hausdorff dimension** | NO | Fractal measurement uses raw geometry |
| **Contrastive training** | NO | Embedding optimization uses raw metric |

**Critical distinction**: `d_eff` governs **flow** (how information travels). Raw Poincaré distance governs **structure** (where things are). The map doesn't change — only the roads do.

### Conductivity Update Rules

#### Rule 1: Hebbian Potentiation (existing Phase XII.5, redirected)

Today, Hebbian LTP produces `HebbianLTP { weight_delta }` intents. We extend this:

```rust
// In hebbian.rs — tick()
// BEFORE (existing):
AgencyIntent::HebbianLTP {
    from_id, to_id,
    weight_delta: trace * config.ltp_rate,
    trace,
}

// AFTER (extended):
AgencyIntent::HebbianLTP {
    from_id, to_id,
    weight_delta: trace * config.ltp_rate,
    conductivity_delta: trace * config.conductivity_ltp_rate,  // NEW
    trace,
}
```

Server handler applies:
```rust
edge.weight = (edge.weight + intent.weight_delta).min(config.hebbian_max_weight);
edge.conductivity = (edge.conductivity + intent.conductivity_delta).clamp(0.01, 10.0);
```

#### Rule 2: Flow-Proportional Reinforcement (new, from FlowLedger)

During the Agency tick, edges with high flow rate get a conductivity boost:

$$\Delta\kappa = \alpha_{flow} \cdot \left(\frac{f_{edge}}{f_{mean}} - 1\right) \cdot \kappa_{current}$$

where:
- $\alpha_{flow}$ = `AGENCY_FLOW_CONDUCTIVITY_RATE` (default: 0.01)
- $f_{edge}$ = flow rate of this edge (traversals/sec)
- $f_{mean}$ = mean flow rate across all edges

Edges above average flow get wider. Edges below average get narrower. The system self-organizes.

#### Rule 3: Temporal Decay (existing Phase B1, extended)

Today, `scan_temporal_decay` decays `edge.weight`. We extend to also decay conductivity:

$$\kappa(t) = 1.0 + (\kappa_0 - 1.0) \cdot e^{-\lambda_\kappa \cdot \Delta t}$$

This decays conductivity **toward 1.0** (not toward 0), meaning unused edges return to baseline — they don't die, they just lose their myelination.

#### Rule 4: Murray Rebalancer (Layer 3, during SleepCycle)

See Section 5.

### Conductivity Dynamics Summary

```
             ┌──────────────────┐
             │  conductivity    │
             │  κ ∈ [0.01, 10]  │
             └────────┬─────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐  ┌──────────┐  ┌─────────┐
   │ INCREASE│  │ DECREASE │  │EQUILIB. │
   │         │  │          │  │         │
   │ Hebbian │  │ Temporal │  │ Murray  │
   │ LTP     │  │ Decay    │  │ Rebal.  │
   │         │  │          │  │         │
   │ Flow    │  │ Dormancy │  │ Sleep   │
   │ Reinf.  │  │ penalty  │  │ Cycle   │
   └─────────┘  └──────────┘  └─────────┘

   AWAKE (tick)  AWAKE (tick)  ASLEEP (REM)
```

---

## 5. Layer 3: MurrayRebalancer

The **vascular remodeling system**. During sleep, the graph's "vessel network" is rebalanced so that parent conduits match the cubic sum of their children — exactly as biological vasculature does.

### Murray's Law

In 1926, Cecil D. Murray proved that the optimal branching of blood vessels follows:

$$r_{parent}^3 = \sum_{i=1}^{n} r_{child_i}^3$$

This minimizes the total work of pumping fluid through a branching network. The key insight: **the cube root relationship is not arbitrary — it's the unique exponent that minimizes friction + metabolic cost simultaneously.**

In NietzscheDB, the "vessel radius" is `edge.conductivity`:

$$\kappa_{parent}^3 = \sum_{i=1}^{n} \kappa_{child_i}^3$$

### When It Runs

The MurrayRebalancer runs **exclusively during the SleepCycle** — the graph's REM phase. This mirrors biology: vascular remodeling happens during rest, not during active cognition.

Integration point: `nietzsche-sleep/src/cycle.rs`, as a new step between reconsolidation and Hausdorff adjustment.

```
SleepCycle Steps (updated):
  1. Perturbation (existing)
  2. Riemannian re-optimization (existing)
  3. ★ Murray Rebalancing (NEW) ← HERE
  4. Hausdorff adjustment (existing)
  5. Embedding checkpoint (existing)
```

### Algorithm

```rust
/// Murray Rebalancer — fractal vascular equilibrium.
///
/// For each node with outgoing edges (a "branch point"):
/// 1. Compute the cubic sum of children conductivities
/// 2. Adjust incoming edge conductivity to match Murray's Law
/// 3. Apply damping to prevent oscillation
///
/// The result: the conductivity network self-organizes into a
/// fractal tree that minimizes total computational flow resistance.
pub struct MurrayRebalancer {
    config: MurrayConfig,
}

pub struct MurrayConfig {
    /// Damping factor α ∈ (0, 1]. How fast we converge to Murray equilibrium.
    /// 0.1 = slow/stable (production). 1.0 = instant (testing).
    pub damping: f32,
    /// Minimum number of outgoing edges to qualify as a "branch point".
    /// Leaf nodes (degree < min_fanout) are skipped.
    pub min_fanout: usize,
    /// Maximum conductivity change per sleep cycle (prevents jumps).
    pub max_delta: f32,
    /// Whether to also balance incoming edges (bidirectional Murray).
    pub bidirectional: bool,
    /// Maximum nodes to process per sleep cycle (caps cost).
    pub max_nodes: usize,
    /// Whether Murray rebalancing is enabled (default: true).
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
```

### Core Logic

```rust
/// Process one branch point (node with outgoing edges).
///
/// Returns: Vec of (edge_id, old_conductivity, new_conductivity)
fn rebalance_branch(
    &self,
    incoming_edge: &Edge,
    outgoing_edges: &[&Edge],
    flow_stats: &FlowLedger,
) -> Option<MurrayDelta> {
    if outgoing_edges.len() < self.config.min_fanout {
        return None;
    }

    // Step 1: Cubic sum of children
    let child_cubic_sum: f64 = outgoing_edges.iter()
        .map(|e| (e.conductivity as f64).powi(3))
        .sum();

    // Step 2: Murray target for parent
    let murray_target = child_cubic_sum.cbrt() as f32;

    // Step 3: Damped adjustment
    let current = incoming_edge.conductivity;
    let delta = (murray_target - current) * self.config.damping;
    let clamped_delta = delta.clamp(-self.config.max_delta, self.config.max_delta);
    let new_conductivity = (current + clamped_delta).clamp(0.01, 10.0);

    if (new_conductivity - current).abs() < 1e-6 {
        return None; // Already at equilibrium
    }

    Some(MurrayDelta {
        edge_id: incoming_edge.id,
        old_conductivity: current,
        new_conductivity,
        murray_target,
        child_count: outgoing_edges.len(),
    })
}
```

### Flow-Weighted Murray (Enhancement)

Standard Murray assumes all children carry equal flow. But in NietzscheDB, some branches are highways and others are trails. We extend Murray with flow weighting:

$$\kappa_{parent}^3 = \sum_{i=1}^{n} \kappa_{child_i}^3 \cdot \frac{f_i}{\bar{f}}$$

where $f_i$ = flow rate from FlowLedger, $\bar{f}$ = mean flow rate of siblings.

This means: if one child carries 90% of the traffic, Murray allocates more "diameter" to that branch. The fractal emerges **proportional to actual use**, not just topology.

```rust
/// Flow-weighted Murray: children with more traffic get more "diameter".
fn rebalance_branch_flow_weighted(
    &self,
    incoming_edge: &Edge,
    outgoing_edges: &[&Edge],
    flow_stats: &FlowLedger,
) -> Option<MurrayDelta> {
    if outgoing_edges.len() < self.config.min_fanout {
        return None;
    }

    // Get flow rates for each child edge
    let flows: Vec<f64> = outgoing_edges.iter()
        .map(|e| flow_stats.flow_rate(e.id).max(1e-9))
        .collect();
    let mean_flow = flows.iter().sum::<f64>() / flows.len() as f64;

    // Flow-weighted cubic sum
    let child_cubic_sum: f64 = outgoing_edges.iter()
        .zip(flows.iter())
        .map(|(e, &f)| {
            let flow_weight = f / mean_flow.max(1e-9);
            (e.conductivity as f64).powi(3) * flow_weight
        })
        .sum();

    let murray_target = child_cubic_sum.cbrt() as f32;
    // ... same damping logic as above ...
}
```

### MurrayReport

```rust
/// Report from a Murray rebalancing pass.
#[derive(Debug)]
pub struct MurrayReport {
    /// Number of branch points evaluated.
    pub branches_evaluated: usize,
    /// Number of edges whose conductivity was adjusted.
    pub edges_adjusted: usize,
    /// Mean Murray deviation before adjustment (0 = perfect fractal).
    pub mean_deviation_before: f64,
    /// Mean Murray deviation after adjustment.
    pub mean_deviation_after: f64,
    /// Top-K largest adjustments (for debugging).
    pub top_adjustments: Vec<MurrayDelta>,
    /// Global Murray compliance score ∈ [0, 1] (1 = perfect fractal).
    pub compliance_score: f64,
}
```

### Murray Compliance Score

A single number that measures how "fractal" the conductivity network is:

$$\text{compliance} = 1 - \frac{1}{|B|} \sum_{b \in B} \left| \frac{\kappa_b^3 - \sum_i \kappa_{b_i}^3}{\kappa_b^3} \right|$$

where $B$ = set of branch points. Perfect compliance = 1.0 (all branch points satisfy Murray's Law).

This metric can be added to the `AgencyTickReport` for monitoring and to the dashboard.

---

## 6. Emergent Behavior: Bejan's Constructal Law

Adrian Bejan's Constructal Law (1996) states:

> *"For a finite-size system to persist in time (to live), it must evolve in such a way that it provides easier access to the imposed currents that flow through it."*

**We do not implement Bejan.** Bejan **emerges** from the interaction of FlowLedger + ConductivityTensor + MurrayRebalancer.

### How It Emerges

1. **FlowLedger** measures where the EVA's "queries" (current) flow most
2. **ConductivityTensor** widens those channels (Hebbian + flow reinforcement)
3. **MurrayRebalancer** ensures the widened channels form a **fractal tree** (not a random blob)
4. **Temporal Decay** narrows unused channels back to baseline

The result: the graph's conductivity network evolves toward a **dendritic fractal** — the same shape as rivers, lungs, blood vessels, and lightning — because that shape minimizes total flow resistance. This is Bejan's Constructal Law, emerging from first principles without being explicitly programmed.

### The Constructal Flow Energy

The total "metabolic cost" of the graph, which the system minimizes over time:

$$E_{flow} = \sum_{e \in \text{edges}} \frac{f_e^2}{\kappa_e}$$

where $f_e$ = flow rate, $\kappa_e$ = conductivity. This is the discrete analog of Bejan's flow resistance integral. When $E_{flow}$ decreases between sleep cycles, the graph is becoming more efficient. When it increases, something disrupted the flow network (new knowledge, trauma, etc.).

This metric is the **single most important number** for the Hydraulic Engine. It should be tracked in `AgencyTickReport` and plotted on the dashboard.

---

## 7. Integration: The Complete Breath (Updated)

The existing Agency tick gains a new sub-phase:

```
Agency Tick (updated):
  Phase 1-10:  Core L-System (unchanged)
  Phase 11:    Sensory degradation (unchanged)
  Phase 12:    ECAN attention bids (unchanged)
  Phase 12.5:  Hebbian LTP (EXTENDED → writes conductivity_delta)
  Phase 13:    Thermodynamics (EXTENDED → uses d_eff for heat flow)
  Phase 14:    Gravity (EXTENDED → uses d_eff for force calculation)
  Phase 15:    DirtySet (unchanged)
  Phase 16:    Shatter (unchanged)
  ★ Phase 17:  FLOW ANALYSIS (NEW)
               - Drain FlowLedger epoch
               - Compute flow-proportional conductivity deltas
               - Emit FlowConductivityUpdate intents
               - Compute E_flow metric
               - Emit FlowReport
  Phase 18-26: Training, Decay, Growth, Cognitive (unchanged)
```

### New AgencyIntents

```rust
// In reactor.rs — AgencyIntent enum

// ── Phase XVII: Hydraulic Flow ──────────────────────────────

/// Update edge conductivity based on flow analysis.
/// Produced when: FlowLedger shows edge flow significantly above/below mean.
/// The server executes: `edge.conductivity = new_conductivity.clamp(0.01, 10.0)`
FlowConductivityUpdate {
    edge_id: Uuid,
    old_conductivity: f32,
    new_conductivity: f32,
    flow_rate: f64,
    pressure: f64,
},

/// Propose a shortcut edge to reduce flow resistance.
/// Produced when: FlowLedger detects a multi-hop path with high total CPU cost.
/// The server creates a direct edge with high initial conductivity.
ProposeShortcut {
    from_id: Uuid,
    to_id: Uuid,
    hops_saved: usize,
    cpu_saved_ns: u64,
    initial_conductivity: f32,
},

// ── SleepCycle: Murray Rebalancing ──────────────────────────

/// Adjust edge conductivity per Murray's Law during sleep.
/// Produced when: MurrayRebalancer detects deviation from fractal equilibrium.
MurrayAdjust {
    edge_id: Uuid,
    old_conductivity: f32,
    new_conductivity: f32,
    murray_target: f32,
},
```

### New AgencyTickReport Fields

```rust
// In engine.rs — AgencyTickReport

/// Hydraulic flow analysis report (None if not yet time for analysis).
pub flow_report: Option<FlowReport>,

/// Murray rebalancing report (None if not during SleepCycle).
pub murray_report: Option<MurrayReport>,
```

### FlowReport

```rust
#[derive(Debug)]
pub struct FlowReport {
    /// Total traversals this epoch.
    pub total_traversals: u64,
    /// Mean CPU cost per traversal (nanoseconds).
    pub mean_cpu_ns: f64,
    /// Constructal flow energy E_flow (lower = more efficient).
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
```

---

## 8. Shortcut Erosion: The River Carves a Canyon

The most dramatic emergent behavior: when the FlowLedger detects a **multi-hop bottleneck** (high CPU cost spread across a chain of edges), the system proposes a **direct shortcut** — a new edge that bypasses the expensive chain.

### Detection

```rust
/// Detect multi-hop bottlenecks that could benefit from a shortcut.
///
/// A bottleneck chain is: A → B → C → ... → Z where:
/// - Total CPU cost > shortcut_threshold
/// - Chain length ≥ 3 hops
/// - Both A and Z are high-traffic nodes
///
/// The shortcut A → Z with high initial conductivity eliminates
/// the intermediate hops for future traversals.
fn detect_shortcut_candidates(
    ledger: &FlowLedger,
    adjacency: &AdjacencyIndex,
    config: &FlowConfig,
) -> Vec<ShortcutCandidate> {
    // BFS from top-K highway endpoints, tracking cumulative CPU cost
    // If path cost > threshold AND endpoints are both high-traffic:
    //   → emit ShortcutCandidate
}
```

### Shortcut Lifecycle

```
1. FlowLedger detects: A→B→C→D costs 800μs average
2. Phase 17 emits: ProposeShortcut { A→D, hops_saved: 3, cpu_saved: 600μs }
3. Server creates: Edge { A→D, weight: 0.5, conductivity: 3.0 }
4. Future queries use A→D (d_eff is much shorter)
5. FlowLedger confirms: A→D gets traffic, B→C flow drops
6. Temporal Decay: B→C conductivity decays toward 1.0
7. Murray Rebalancer: A→D conductivity stabilizes in fractal equilibrium
```

This is **exactly** how rivers erode shortcuts through soft terrain. The water (query flow) finds the path of least resistance, and the terrain (conductivity) reshapes itself to accommodate.

### Safety: Maximum Shortcuts Per Epoch

To prevent runaway edge creation:
- `AGENCY_FLOW_MAX_SHORTCUTS_PER_EPOCH` (default: 5)
- Shortcuts must save at least `AGENCY_FLOW_SHORTCUT_MIN_SAVINGS_NS` (default: 100_000 = 100μs)
- Shortcuts are created with `edge_type: EdgeType::Association` and `lsystem_rule: Some("hydraulic_shortcut")`
- The NiilistaGcDaemon can later merge or prune shortcuts that become redundant

---

## 9. Configuration

### Environment Variables

All prefixed with `AGENCY_FLOW_`:

| Variable | Default | Description |
|---|---|---|
| `AGENCY_FLOW_ENABLED` | `true` | Master switch for the Hydraulic Flow Engine |
| `AGENCY_FLOW_INTERVAL` | `5` | Agency ticks between flow analysis passes |
| `AGENCY_FLOW_SAMPLE_RATE` | `1` | Record 1-in-N traversals (1 = all, 10 = 10% sample) |
| `AGENCY_FLOW_EMA_ALPHA` | `0.1` | EMA smoothing for CPU cost tracking |
| `AGENCY_FLOW_CONDUCTIVITY_RATE` | `0.01` | Flow-proportional conductivity learning rate |
| `AGENCY_FLOW_CONDUCTIVITY_MIN` | `0.01` | Minimum conductivity (prevents division by zero) |
| `AGENCY_FLOW_CONDUCTIVITY_MAX` | `10.0` | Maximum conductivity (prevents metric collapse) |
| `AGENCY_FLOW_DECAY_LAMBDA` | `0.00001` | Conductivity decay rate toward baseline |
| `AGENCY_FLOW_MAX_SHORTCUTS_PER_EPOCH` | `5` | Maximum new shortcut edges per flow analysis |
| `AGENCY_FLOW_SHORTCUT_MIN_SAVINGS_NS` | `100000` | Minimum CPU savings to justify a shortcut (ns) |
| `AGENCY_FLOW_SHORTCUT_MIN_HOPS` | `3` | Minimum chain length for shortcut consideration |
| `AGENCY_FLOW_SHORTCUT_INITIAL_CONDUCTIVITY` | `3.0` | Initial conductivity for new shortcuts |
| `AGENCY_MURRAY_ENABLED` | `true` | Whether Murray rebalancing runs during sleep |
| `AGENCY_MURRAY_DAMPING` | `0.1` | Damping factor for Murray convergence |
| `AGENCY_MURRAY_MIN_FANOUT` | `2` | Minimum children to qualify as branch point |
| `AGENCY_MURRAY_MAX_DELTA` | `0.5` | Maximum conductivity change per sleep cycle |
| `AGENCY_MURRAY_BIDIRECTIONAL` | `false` | Also balance incoming edges |
| `AGENCY_MURRAY_MAX_NODES` | `5000` | Maximum branch points per sleep cycle |

### Hebbian Config Extensions

| Variable | Default | Description |
|---|---|---|
| `AGENCY_HEBBIAN_CONDUCTIVITY_LTP_RATE` | `0.005` | Conductivity increase per unit of LTP trace |
| `AGENCY_HEBBIAN_CONDUCTIVITY_MAX` | `10.0` | Cap on Hebbian-driven conductivity |

---

## 10. File Layout

```
crates/nietzsche-agency/src/
├── hydraulic/
│   ├── mod.rs                    // Module exports
│   ├── flow_ledger.rs            // FlowLedger (Layer 1)
│   ├── conductivity.rs           // ConductivityTensor logic (Layer 2)
│   ├── murray.rs                 // MurrayRebalancer (Layer 3)
│   ├── shortcut.rs               // Shortcut erosion detection
│   └── report.rs                 // FlowReport, MurrayReport
├── hydraulic.rs                  // Re-exports
```

```
crates/nietzsche-graph/src/
└── model.rs                      // Edge gains `conductivity: f32` field
```

```
crates/nietzsche-sleep/src/
└── cycle.rs                      // SleepCycle gains Murray step
```

---

## 11. Dependencies

### New Dependencies: None

The Hydraulic Flow Engine uses only existing dependencies:
- `dashmap` (already in workspace — for FlowLedger)
- `uuid` (already in workspace)
- `std::sync::atomic` (stdlib)
- `std::time::Instant` (stdlib)

No new crates needed. This is pure Rust arithmetic on existing data structures.

---

## 12. Metrics & Observability

### Dashboard Additions

| Metric | Type | Source | Meaning |
|---|---|---|---|
| `flow.constructal_energy` | Gauge | FlowReport | Total flow resistance (lower = healthier) |
| `flow.delta_energy` | Gauge | FlowReport | Change since last epoch (negative = improving) |
| `flow.total_traversals` | Counter | FlowLedger | Graph activity level |
| `flow.mean_cpu_ns` | Gauge | FlowLedger | Average traversal cost |
| `flow.conductivity_updates` | Counter | FlowReport | How actively the graph is reshaping |
| `flow.shortcuts_proposed` | Counter | FlowReport | River erosion activity |
| `murray.compliance_score` | Gauge | MurrayReport | How fractal the network is [0, 1] |
| `murray.edges_adjusted` | Counter | MurrayReport | Remodeling activity per sleep |
| `murray.mean_deviation` | Gauge | MurrayReport | Average Murray deviation |

### gRPC Stream Extension

The existing `QuantumEventStream` (from Quantum Kernel Spec) can carry hydraulic events:

```protobuf
// In quantum.proto (or new hydraulic.proto)
message FlowEvent {
    string edge_id = 1;
    double flow_rate = 2;
    double pressure = 3;
    float old_conductivity = 4;
    float new_conductivity = 5;
}

message ShortcutEvent {
    string from_id = 1;
    string to_id = 2;
    uint32 hops_saved = 3;
    uint64 cpu_saved_ns = 4;
}
```

---

## 13. Implementation Order

```
Step 1: FlowLedger (MEASURE)
  ├─ Add FlowLedger struct + FlowStats
  ├─ Hook record() into NQL executor (MATCH traversal)
  ├─ Hook record() into DIFFUSE walk
  ├─ Add drain_epoch() + FlowEpochReport
  ├─ Wire into AgencyTickReport
  └─ Tests: concurrent record, pressure calculation, epoch drain

Step 2: Edge.conductivity (FIELD)
  ├─ Add field to Edge struct with serde(default)
  ├─ Add deserialize_edge_compat() for V2→V3 migration
  ├─ Add effective_distance() utility function
  ├─ Wire into DIFFUSE walk distance calculation
  ├─ Wire into Gravity force calculation (d_eff)
  ├─ Wire into Thermodynamics heat flow (d_eff)
  └─ Tests: backward compat, effective_distance correctness

Step 3: Hebbian → Conductivity (CONNECT)
  ├─ Add conductivity_delta to HebbianLTP intent
  ├─ Extend server handler to apply conductivity delta
  ├─ Add AGENCY_HEBBIAN_CONDUCTIVITY_LTP_RATE config
  └─ Tests: Hebbian tick produces conductivity updates

Step 4: Flow Analysis Phase (ANALYZE)
  ├─ Add Phase 17 to agency tick
  ├─ Implement flow-proportional conductivity update
  ├─ Implement shortcut detection
  ├─ Add FlowConductivityUpdate + ProposeShortcut intents
  ├─ Add FlowReport to AgencyTickReport
  └─ Tests: flow analysis produces correct intents

Step 5: MurrayRebalancer (EQUILIBRATE)
  ├─ Implement MurrayRebalancer + MurrayConfig
  ├─ Add Murray step to SleepCycle
  ├─ Implement compliance score
  ├─ Add MurrayAdjust intent
  ├─ Add MurrayReport
  └─ Tests: Murray convergence, compliance scoring

Step 6: Temporal Decay → Conductivity (DECAY)
  ├─ Extend scan_temporal_decay to also decay conductivity
  ├─ Conductivity decays toward 1.0 (not 0)
  └─ Tests: unused edges return to baseline
```

---

## 14. Theoretical Foundation

### Constructal Law (Bejan, 1996)

Adrian Bejan, Professor of Mechanical Engineering at Duke University, proposed that all flow systems in nature — from river deltas to lungs to lightning — evolve toward tree-shaped architectures that maximize flow access. The law is:

> *"For a finite-size flow system to persist in time, it must evolve freely such that it provides progressively easier access to its currents."*

Our system doesn't implement Bejan's law directly. Instead, the combination of FlowLedger (measurement), ConductivityTensor (adaptation), and MurrayRebalancer (fractal equilibrium) creates the **conditions** for constructal patterns to emerge naturally.

### Murray's Law (Murray, 1926)

Cecil D. Murray's optimization principle for biological transport networks: the cube of the parent vessel radius equals the sum of cubes of child vessel radii. This minimizes the total work (pumping power + metabolic maintenance) of the vascular network.

Our adaptation replaces physical radius with computational conductivity, and physical blood flow with query traversal frequency. The mathematical structure is identical.

### Hagen-Poiseuille Flow (Analogy)

In fluid mechanics, flow through a cylindrical vessel follows:

$$Q = \frac{\pi r^4 \Delta P}{8 \mu L}$$

Our discrete analog:
- $Q$ → query flow rate (traversals/sec)
- $r$ → `edge.conductivity` (κ)
- $\Delta P$ → pressure gradient (energy difference / d_eff)
- $L$ → Poincaré distance
- $\mu$ → "viscosity" (CPU cost per traversal)

The fourth-power dependence of flow on radius ($r^4$) explains why small conductivity changes have dramatic effects on flow distribution — doubling conductivity increases flow capacity 16×.

### Why Not Use $r^4$ Instead of $r^3$?

Murray's Law uses $r^3$ (not $r^4$) because it optimizes **total cost** (flow resistance + metabolic maintenance), not just flow resistance. The metabolic cost of maintaining a vessel scales with $r^2$ (wall surface area), creating a trade-off. The $r^3$ exponent is the sweet spot.

In our system, the "metabolic cost" of maintaining a high-conductivity edge is the memory and computational overhead of tracking it. The $r^3$ balance remains appropriate.

---

## 15. References

1. Bejan, A. (1997). *Constructal-theory network of conducting paths for cooling a heat generating volume*. International Journal of Heat and Mass Transfer, 40(4), 799-816.
2. Bejan, A. & Lorente, S. (2008). *Design with Constructal Theory*. Wiley.
3. Murray, C.D. (1926). *The physiological principle of minimum work: I. The vascular system and the cost of blood volume*. Proceedings of the National Academy of Sciences, 12(3), 207-214.
4. Hess, W.R. (1917). *Über die periphere Regulierung der Blutzirkulation*. Pflügers Archiv, 168, 439-490.
5. West, G.B., Brown, J.H. & Enquist, B.J. (1997). *A general model for the origin of allometric scaling laws in biology*. Science, 276(5309), 122-126.
6. Katifori, E., Szöllősi, G.J. & Magnasco, M.O. (2010). *Damage and fluctuations induce loops in optimal transport networks*. Physical Review Letters, 104(4), 048704.
7. Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience, 11(2), 127-138.
