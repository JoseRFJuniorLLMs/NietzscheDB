# Boundary-Only Energy Dissipation

**Status**: Design
**Author**: Agency Architecture Team
**Date**: 2026-03-11
**Phase**: Proposed (post-Phase XIV)
**Depends on**: Phase XII (ECAN), Phase XIII (Thermodynamics), Phase XIV (Gravity)

---

## 1. Motivation

The current energy system applies a **uniform 3% decay per tick** to every
participating node, regardless of its position in the Poincare ball. This is
thermodynamically incorrect and directly contradicts the conditions required for
Self-Organized Criticality (SOC).

In the Bak-Tang-Wiesenfeld sandpile model -- the canonical SOC system -- two
properties are essential:

1. **Bulk conservation**: grains toppled from an interior cell are distributed
   to its neighbors. No energy leaves the system at interior sites.
2. **Boundary dissipation**: grains that fall off the edge of the lattice
   (boundary) are lost from the system permanently.

The combination of bulk conservation with boundary dissipation is what drives
the system toward the critical state. Uniform decay violates rule (1): energy
vanishes everywhere, draining the bulk faster than the boundary can dissipate
it. The result is a slow death toward thermodynamic equilibrium rather than a
self-organized critical state.

### Why this matters for NietzscheDB

- The knowledge graph should exhibit **power-law avalanche distributions** (a
  hallmark of SOC). Uniform decay suppresses large avalanches by draining the
  energy reservoir.
- The Poincare ball provides a **natural boundary**: nodes at high magnitude
  (||z|| close to 1.0) are at the "edge" of the conceptual space. These are
  the most specific, peripheral memories -- exactly where dissipation should
  occur.
- Interior nodes (low magnitude, near the origin) represent abstract, general
  concepts. Their energy should be conserved and redistributed, not destroyed.

---

## 2. Current Energy Flow Analysis

### 2.1 Energy Creation

Energy enters the system via:

| Source | Mechanism | Code Location |
|--------|-----------|---------------|
| Node insertion | Initial `energy` field set on creation | `GraphStorage::put_node()` |
| ECAN attention gain | `received * attention_energy_gain` (default 0.1) | `attention_economy::compute_energy_deltas()` |
| Neural boost | GNN-detected high-importance node gets energy boost | `AgencyIntent::NeuralBoost` |
| Gravity pulls | (Experimental) Energy redistribution toward gravity wells | `gravity.rs` (disabled by default) |
| L-System spawn | Child nodes inherit energy from rewrite rules | `nietzsche-zaratustra` |

### 2.2 Energy Decay (Current -- Uniform)

Energy exits the system via:

| Sink | Mechanism | Rate | Code Location |
|------|-----------|------|---------------|
| **ECAN decay** | `semantic_mass * (1.0 - energy_decay)` | 3% per ECAN tick | `attention_economy.rs:221` |
| Temporal edge decay | `w_base * e^(-lambda * dt)` | ~0.00001% per second | `temporal_decay.rs:84` |
| Node deletion | Hard-delete via Nezhmetdinov forgetting | Binary (0 or all) | `reactor.rs:HardDelete` |
| Phantomization | Exhausted/orphaned nodes become phantoms | Binary (0 or all) | `reactor.rs:Phantomize` |
| Tumor dampening | Cluster of overheated nodes drained by 30% | 30% per detection | `circuit_breaker.rs` |

The **ECAN decay** at line 221 of `attention_economy.rs` is the primary
continuous dissipation mechanism:

```rust
// attention_economy.rs:219-222
// Decay: energy lost per tick = current_energy * (1 - decay_rate)
// semantic_mass ~ energy, so we use it as proxy for current energy level
let decay_loss = state.semantic_mass * (1.0 - config.energy_decay);
let net = gain - decay_loss;
```

With `energy_decay = 0.97` (hardcoded at `engine.rs:1290`), every node loses
3% of its semantic mass per ECAN tick. This is applied identically to:

- A hub node at ||z|| = 0.05 (center, abstract concept)
- A leaf node at ||z|| = 0.95 (boundary, specific memory)

This uniform treatment is the core problem.

### 2.3 Energy Redistribution

The thermodynamics module (`thermodynamics.rs`) already implements Fourier heat
flow between connected nodes:

```rust
// thermodynamics.rs:345
// q = kappa * (E_i - E_j), capped by max_heat_flow
let raw_flow = kappa * diff;
let clamped = raw_flow.clamp(-max_flow, max_flow);
```

This creates **local equilibration** (hot nodes lose energy to cold neighbors)
but does NOT distinguish between bulk and boundary. The heat flow is
conservative (energy transferred, not destroyed), which is correct behavior for
the interior.

### 2.4 Total Energy Accounting

Currently, total system energy `E_total` changes per tick as:

```
E_total(t+1) = E_total(t)
    - SUM_all_nodes(semantic_mass_i * 0.03)     // ECAN decay (LEAK)
    + SUM_receivers(received_i * 0.1)             // ECAN attention gain
    + energy_from_new_nodes                       // insertions
    - energy_from_deleted_nodes                   // deletions
```

The ECAN decay term applies to ALL nodes, creating a system-wide energy drain.
There is no distinction between interior and boundary.

---

## 3. Definition of "Boundary" in the Poincare Ball

### 3.1 The Poincare Ball Model

In the Poincare ball model B^d = {z in R^d : ||z|| < 1}, the boundary
is the unit sphere S^{d-1} = {z : ||z|| = 1}. Points near the boundary
represent highly specific, fine-grained concepts in the knowledge hierarchy.
Points near the origin represent abstract, general categories.

The **magnitude** ||z|| of a node's embedding is its radial position in the
ball and corresponds to its **depth** in the conceptual hierarchy:

- ||z|| ~ 0.0: Root concepts (e.g., "Knowledge", "Science")
- ||z|| ~ 0.3: Broad categories (e.g., "Physics", "Biology")
- ||z|| ~ 0.6: Specific topics (e.g., "Quantum Mechanics", "Genetics")
- ||z|| ~ 0.9: Fine-grained memories (e.g., "Schrodinger equation derivation")

### 3.2 Boundary Threshold

We define the **boundary region** as nodes with magnitude above a configurable
threshold `R_boundary`:

```
is_boundary(z) = (||z|| > R_boundary)
```

**Proposed default**: `R_boundary = 0.85`

Rationale:
- The Poincare ball has exponentially increasing volume near the boundary
  (a consequence of hyperbolic geometry: the circumference at radius r grows
  as sinh(r)).
- At R_boundary = 0.85, roughly 40-60% of nodes in a typical NietzscheDB
  collection are "boundary" nodes (the distribution is heavily skewed toward
  high magnitudes).
- The L-System circuit breaker already uses depth-aware energy caps:
  `depth_aware_cap(depth) = base_cap - depth_cap_gradient * depth`
  (`circuit_breaker.rs:127`). This acknowledges that boundary nodes should
  behave differently.
- Setting the threshold too low (e.g., 0.5) would make too many nodes
  "boundary", weakening the conservation property. Setting it too high
  (e.g., 0.98) would make almost no nodes boundary, preventing dissipation.

### 3.3 Soft Boundary (Dissipation Gradient)

Rather than a hard cutoff, we define a **dissipation fraction** that ramps
smoothly from 0 (full conservation) to 1 (full dissipation):

```
alpha(z) = smoothstep(||z||, R_inner, R_boundary)
```

where:
- `R_inner = 0.70` (below this, alpha = 0, pure conservation)
- `R_boundary = 0.85` (above this, alpha = 1, pure dissipation)
- `smoothstep(x, lo, hi) = clamp((x - lo) / (hi - lo), 0, 1)^2 * (3 - 2t)`
  with `t = clamp((x - lo) / (hi - lo), 0, 1)`

This gives:

| Region | ||z|| range | alpha | Energy behavior |
|--------|------------|-------|-----------------|
| Deep interior | [0, 0.70) | 0.0 | Full conservation |
| Transition zone | [0.70, 0.85) | 0.0 to 1.0 | Mixed |
| Boundary | [0.85, 1.0) | 1.0 | Full dissipation |

---

## 4. Proposed Energy Conservation Rules

### 4.1 Core Equations

For each node `i` at position `z_i` with energy `E_i`, degree `d_i`, and
neighbors `N(i)`:

**Boundary nodes** (alpha(z_i) = 1.0):
```
dE_i/dt = -gamma * E_i
```
Energy lost by this node **exits the system**. This is the dissipation term.

**Interior nodes** (alpha(z_i) = 0.0):
```
dE_i/dt = -gamma * E_i + SUM_{j in N(i)} [gamma_j * E_j * alpha(z_j)] / degree(j)
```
Energy "decayed" from this node is **redistributed to its neighbors**. The node
also receives redistributed energy from its boundary-adjacent neighbors.

Wait -- this needs refinement. The key insight is:

- Interior nodes should NOT lose energy to decay at all. Their "decay" energy
  is redistributed.
- Boundary nodes lose energy that exits the system.

### 4.2 Revised Equations (Clean Formulation)

Let gamma = 0.03 (the current decay rate).

For each node i with dissipation fraction alpha_i = alpha(z_i):

```
decay_amount_i = gamma * E_i

// Energy that EXITS the system (dissipated at boundary)
dissipated_i = alpha_i * decay_amount_i

// Energy that is REDISTRIBUTED to neighbors (conserved in bulk)
redistributed_i = (1 - alpha_i) * decay_amount_i

// Net energy change for node i:
dE_i = -decay_amount_i                              // loses its own decay
     + SUM_{j in N(i)} redistributed_j / degree(j)  // gains from neighbors' redistribution
```

### 4.3 Conservation Proof

Total energy change per tick:

```
dE_total = SUM_i dE_i
         = SUM_i [-decay_amount_i + SUM_{j in N(i)} redistributed_j / degree(j)]
```

Since each node j's `redistributed_j` is divided equally among its `degree(j)`
neighbors, and each neighbor receives one share:

```
SUM_i SUM_{j in N(i)} redistributed_j / degree(j) = SUM_j redistributed_j
```

Therefore:

```
dE_total = -SUM_i decay_amount_i + SUM_j redistributed_j
         = -SUM_i decay_amount_i + SUM_j (1 - alpha_j) * decay_amount_j
         = -SUM_i alpha_i * decay_amount_i
         = -SUM_boundary dissipated_i
```

**Result**: Total energy decreases by exactly the amount dissipated at the
boundary. Energy is perfectly conserved in the bulk. QED.

### 4.4 Discrete Update Rule

In practice, the update is applied per ECAN tick:

```rust
// For each participating node i:
let decay_amount = gamma * energy_i;
let alpha_i = dissipation_fraction(magnitude_i);

// Phase 1: Compute redistribution budget per node
let to_redistribute_i = (1.0 - alpha_i) * decay_amount;
let to_dissipate_i = alpha_i * decay_amount;

// Phase 2: Distribute redistribution budget to neighbors
for j in neighbors(i) {
    energy_delta[j] += to_redistribute_i / degree(i);
}

// Phase 3: Apply
energy_i -= decay_amount;     // remove the full decay
energy_i += energy_delta[i];  // add redistributed energy from neighbors
// to_dissipate_i exits the system (not added back anywhere)
```

### 4.5 Edge Cases

1. **Isolated nodes (degree 0)**: No neighbors to redistribute to. The node's
   conservation fraction `(1 - alpha) * decay` has nowhere to go. Policy:
   treat isolated nodes as boundary (alpha = 1.0) regardless of position.
   Isolated nodes in the interior are anomalous and should dissipate.

2. **Nodes at ||z|| = 0** (origin): alpha = 0, full conservation. Their decay
   energy is redistributed to all neighbors. This is correct: the root concept
   should not lose energy.

3. **Very high degree hubs**: A hub at ||z|| = 0.1 with degree 500 receives
   tiny redistribution shares from each neighbor. This is correct: the hub's
   own decay is spread thinly across 500 neighbors, and it receives small
   shares from each of them.

4. **Energy conservation of attention gain**: The ECAN attention gain
   (`received * 0.1`) is SEPARATE from the decay/redistribution system. It
   represents external energy injection (from the attention economy) and is
   not subject to conservation. Only the decay pathway is modified.

---

## 5. Impact on Self-Organized Criticality

### 5.1 SOC Requirements

The Bak-Tang-Wiesenfeld sandpile requires:
1. Slow driving (energy injection): satisfied by ECAN attention gain
2. Bulk conservation with local redistribution: **this proposal**
3. Boundary dissipation: **this proposal**
4. Threshold dynamics (toppling): satisfied by forgetting engine
   (nodes below vitality threshold are deleted = "toppled")

### 5.2 Expected Emergent Behaviors

With boundary dissipation, we expect:

- **Power-law avalanche distribution**: Deletion cascades (triggered when a
  node is removed and its neighbors lose support) should follow a power-law
  size distribution P(s) ~ s^{-tau} with tau ~ 1.5. Currently, the uniform
  decay suppresses these cascades.

- **1/f noise in energy fluctuations**: The total system energy should exhibit
  1/f noise (pink noise) rather than the exponential decay currently observed.

- **Critical temperature**: The thermodynamic module (Phase XIII) should find
  the system naturally hovering near the critical temperature T_c, rather than
  requiring the exploration modifier to artificially maintain it.

- **Gravity well stability**: The gravity wells (Phase XIV) should become
  more stable, as interior hubs no longer drain energy. Wells near the center
  will persist longer; wells near the boundary will dissipate, which is
  correct behavior.

### 5.3 Sandpile Analogy Mapping

| Sandpile concept | NietzscheDB analog |
|------------------|--------------------|
| Grain of sand | Unit of node energy |
| Lattice site | Node in Poincare ball |
| Toppling threshold | Vitality threshold (forgetting engine) |
| Toppling rule | Energy redistribution to neighbors |
| Boundary | Nodes with ||z|| > R_boundary |
| Slow driving | ECAN attention gain + node insertion |
| Avalanche | Cascade of node deletions/phantomizations |

---

## 6. Integration with ECAN Attention Economy

### 6.1 Modification to `compute_energy_deltas()`

The function `attention_economy::compute_energy_deltas()` is the single point
of change for the decay pathway. Currently:

```rust
// CURRENT (attention_economy.rs:218-222)
let gain = state.received * config.attention_energy_gain;
let decay_loss = state.semantic_mass * (1.0 - config.energy_decay);
let net = gain - decay_loss;
```

Proposed replacement:

```rust
// PROPOSED
let gain = state.received * config.attention_energy_gain;
let alpha = dissipation_fraction(state.magnitude);
let decay_amount = state.semantic_mass * (1.0 - config.energy_decay);

// Energy lost to system = boundary dissipation only
let dissipated = alpha * decay_amount;

// Energy redistributed to neighbors = bulk conservation
let redistributed = (1.0 - alpha) * decay_amount;

// This node's loss is the full decay amount (it will receive
// redistribution from neighbors separately)
let net = gain - decay_amount;  // NOTE: not just dissipated

// The redistributed amount is returned separately for neighbor distribution
```

### 6.2 Required Data

The modification requires access to each node's **embedding magnitude** during
the ECAN cycle. Currently, `AttentionState` contains:

```rust
pub struct AttentionState {
    pub node_id: Uuid,
    pub budget: f32,
    pub demand: f32,
    pub received: f32,
    pub spent: f32,
    pub semantic_mass: f32,
}
```

A new field is needed:

```rust
pub magnitude: f32,  // ||z|| = Poincare ball norm of embedding
```

This is already available during the ECAN scan (step 1 of `run_ecan_cycle()`
in `attention_cycle.rs`), where `meta.depth` is read. The `depth` field on
`NodeMeta` is the embedding magnitude. No additional I/O is required.

### 6.3 Two-Phase Energy Update

The current ECAN cycle produces a single `Vec<(Uuid, f32)>` of energy deltas.
With boundary dissipation, the cycle must produce TWO outputs:

1. **Direct deltas**: `Vec<(Uuid, f32)>` -- per-node energy changes from
   attention gain and own decay loss.
2. **Redistribution budget**: `Vec<(Uuid, f32)>` -- per-node energy to be
   distributed to neighbors.

The engine (`engine.rs`) applies direct deltas immediately, then iterates
redistribution budgets and distributes them to neighbors using the adjacency
index. This second pass is O(sum of degrees) = O(E) where E is the number of
edges among participating nodes.

### 6.4 New AttentionReport Fields

```rust
pub struct AttentionReport {
    // ... existing fields ...

    /// Energy deltas: (node_id, delta). Positive = gained, negative = decayed.
    pub energy_deltas: Vec<(Uuid, f32)>,

    /// NEW: Redistribution budget per node.
    /// Each entry (node_id, amount) means: distribute `amount` equally
    /// to all neighbors of node_id.
    pub redistribution_budget: Vec<(Uuid, f32)>,

    /// NEW: Total energy dissipated at boundary this tick.
    pub boundary_dissipation: f32,

    /// NEW: Total energy redistributed in bulk this tick.
    pub bulk_redistribution: f32,
}
```

---

## 7. Integration with Hebbian Learning

### 7.1 Current Hebbian Interaction

The Hebbian module (Phase XII.5, `hebbian.rs`) strengthens edges based on
co-activation during ECAN auctions. It produces `HebbianDelta`s that increase
edge weights. This is independent of node energy decay.

### 7.2 Boundary Dissipation Effects on Hebbian

With boundary dissipation:

- **Interior edges** (both endpoints in bulk): Co-activation traces accumulate
  normally. Energy is conserved, so interior hubs maintain energy longer,
  leading to **more stable Hebbian traces** for core concepts.

- **Boundary edges** (at least one endpoint at boundary): The boundary node
  loses energy faster (pure dissipation, no redistribution inflow). This means
  boundary nodes become less active in ECAN, generate fewer winning bids, and
  accumulate weaker Hebbian traces. Result: **boundary edges are naturally
  pruned** by the combination of energy loss and trace decay.

- **Cross-region edges** (interior <-> boundary): The interior node maintains
  energy, the boundary node decays. The attention bid from interior to boundary
  remains strong (the interior node has high mass), but the reverse bid
  weakens. This creates a **directional asymmetry** in Hebbian strengthening:
  interior-to-boundary edges strengthen, boundary-to-interior edges weaken.
  This is desirable: it means abstract concepts maintain strong connections to
  their specific instances, but the specific instances do not dominate the
  abstract concept's attention.

### 7.3 No Changes Required to Hebbian Module

The Hebbian module operates on ECAN winning bids, which are already modulated
by semantic mass (which depends on energy). As boundary nodes lose energy, their
semantic mass drops, their bids weaken, and their Hebbian traces decay. No
direct modification of `hebbian.rs` is needed.

---

## 8. Integration with Existing Thermodynamics

### 8.1 Heat Flow (Fourier's Law)

The thermodynamics module (`thermodynamics.rs`) computes heat flow:
```
q_{ij} = kappa * (E_i - E_j)
```

This is a **conservative** transfer (energy moves from hot to cold, total
preserved). It is fully compatible with boundary dissipation because:

- Heat flow equilibrates local temperature gradients.
- Boundary dissipation removes energy at the boundary.
- Together, they create a steady-state flow pattern: energy injected at the
  interior (via ECAN) flows outward through heat conduction and is dissipated
  at the boundary. This is analogous to heat conduction in a sphere with a
  cold boundary.

### 8.2 Temperature Impact

With boundary dissipation:
- **Interior temperature** will be higher (energy is conserved, hubs accumulate)
- **Boundary temperature** will be lower (energy drains faster)
- **Global temperature** may decrease slightly, as boundary nodes (the majority)
  have less energy.
- The **entropy** should decrease (more structured energy distribution: high
  in center, low at edges) which means **lower free energy** (F = U - TS with
  lower S at constant U).

This is the desired thermodynamic trajectory: the system self-organizes into a
structured state with energy concentrated in the interior (general knowledge)
and dissipated at the boundary (transient specific memories).

### 8.3 Phase State Effects

The phase classification (Solid/Liquid/Gas/Critical) is based on temperature
T = sigma_E / mean_E. With boundary dissipation:

- The energy **variance** sigma_E will increase (interior hot, boundary cold).
- The **mean** energy may decrease slightly.
- Net effect: temperature increases. If the system was previously in the Solid
  phase (too rigid), boundary dissipation naturally pushes it toward Liquid
  (optimal). If it was already Liquid, the exploration modifier will compensate.

---

## 9. Performance Analysis

### 9.1 Current Cost

The ECAN cycle is O(N) for the scan, O(N * max_bids) for bid generation, and
O(B) for auction resolution where B = total bids. The energy delta computation
is O(N). Total: O(N * max_bids).

### 9.2 Boundary Dissipation Cost

The boundary dissipation adds:

1. **Magnitude computation**: O(1) per node (||z|| = sqrt(sum of squares), or
   use the pre-computed `depth` field on NodeMeta). No additional cost.

2. **Dissipation fraction**: O(1) per node (smoothstep is a few arithmetic
   operations). No additional cost.

3. **Redistribution pass**: For each node with redistribution budget > 0,
   iterate its neighbors and add energy shares. Cost: O(sum of degrees of
   participating nodes) = O(E_participating). For the typical case where
   `max_scan = 10000` and average degree = 5, this is O(50000) -- comparable
   to the existing ECAN scan.

**Total cost**: O(N + E) per tick, where N = participating nodes and
E = edges among them. This is **O(N) for sparse graphs** (average degree
bounded by a constant), which includes all practical NietzscheDB collections.

### 9.3 Memory

Additional memory per ECAN cycle:
- `redistribution_budget: Vec<(Uuid, f32)>` -- at most N entries, 20 bytes each.
- For N = 10,000 nodes: ~200 KB. Negligible.

### 9.4 No Impact on Non-ECAN Ticks

The boundary dissipation is computed ONLY during the ECAN cycle (step 12 of
the engine tick). Non-ECAN ticks (when interval-gated) have zero additional
cost.

---

## 10. Migration Strategy

### 10.1 Phase 1: Instrumentation (Non-Breaking)

Add magnitude and dissipation fraction to the ECAN report without changing
energy dynamics. This allows observing what the system WOULD do under boundary
dissipation.

Changes:
- Add `magnitude: f32` to `AttentionState`.
- Add `boundary_dissipation: f32` and `bulk_redistribution: f32` to
  `AttentionReport` (computed but not applied).
- Log the ratio `boundary_dissipation / total_decay` per tick.
- Add to cognitive dashboard for monitoring.

Timeline: 1-2 days.

### 10.2 Phase 2: Blended Transition (Gradual)

Introduce a **blend factor** `beta` in [0, 1] that interpolates between
uniform decay (beta = 0, current behavior) and boundary-only dissipation
(beta = 1, target behavior):

```rust
// Blended decay model
let uniform_decay = gamma * energy_i;
let boundary_decay = alpha_i * gamma * energy_i;
let redistributed = (1.0 - alpha_i) * gamma * energy_i;

// Effective decay from this node:
let actual_dissipated = (1.0 - beta) * uniform_decay + beta * boundary_decay;
let actual_redistributed = beta * redistributed;
```

At beta = 0, this reduces to the current uniform decay. At beta = 1, this is
pure boundary dissipation. The blend factor is controlled by an environment
variable:

```
AGENCY_BOUNDARY_DISSIPATION_BETA=0.0   # disabled (default)
AGENCY_BOUNDARY_DISSIPATION_BETA=0.25  # 25% boundary, 75% uniform
AGENCY_BOUNDARY_DISSIPATION_BETA=1.0   # full boundary dissipation
```

Migration plan:
1. Deploy with beta = 0.0 (no change).
2. Increment beta by 0.1 every few days, monitoring:
   - Total energy trajectory (should stabilize, not collapse)
   - Temperature trajectory (should increase slightly toward Liquid)
   - Avalanche size distribution (should develop heavier tails)
   - Mean energy of interior vs boundary nodes (should diverge)
3. At beta = 1.0, the migration is complete.

Timeline: 2-3 weeks of gradual transition.

### 10.3 Phase 3: Cleanup

Once beta = 1.0 is stable:
- Remove the beta blend factor and hardcode boundary dissipation.
- Remove the old uniform decay code path.
- Update the `energy_decay` config field documentation to clarify it controls
  the decay RATE (gamma), not the decay MODE.

Timeline: 1 day.

### 10.4 Rollback Plan

At any point during Phase 2, setting `AGENCY_BOUNDARY_DISSIPATION_BETA=0.0`
instantly restores the current uniform decay behavior. No data migration is
needed; this is purely a runtime parameter.

---

## 11. Configuration

### 11.1 New Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AGENCY_BOUNDARY_DISSIPATION_BETA` | f32 | 0.0 | Blend: 0=uniform, 1=boundary-only |
| `AGENCY_BOUNDARY_R_INNER` | f32 | 0.70 | Below this magnitude, alpha = 0 |
| `AGENCY_BOUNDARY_R_OUTER` | f32 | 0.85 | Above this magnitude, alpha = 1 |
| `AGENCY_BOUNDARY_ISOLATED_AS_BOUNDARY` | bool | true | Treat degree-0 nodes as boundary |

### 11.2 Interaction with Existing Config

- `AGENCY_ECAN_INTERVAL`: No change. Boundary dissipation runs at the same
  interval as ECAN.
- `energy_decay = 0.97`: The gamma rate (0.03) is unchanged. Only the
  destination of decayed energy changes (dissipation vs redistribution).
- `AGENCY_THERMO_ENABLE_HEAT_FLOW`: Complementary. Heat flow equilibrates
  local gradients; boundary dissipation removes energy at the edge. Both
  should be enabled for correct thermodynamics.

---

## 12. Observability

### 12.1 New Metrics (Cognitive Dashboard)

| Metric | Description |
|--------|-------------|
| `boundary_dissipation_total` | Total energy dissipated at boundary this tick |
| `bulk_redistribution_total` | Total energy redistributed in bulk this tick |
| `conservation_ratio` | `1 - (dissipated / total_decay)` -- should approach ~0.5 |
| `mean_energy_interior` | Mean energy of nodes with ||z|| < R_inner |
| `mean_energy_boundary` | Mean energy of nodes with ||z|| > R_outer |
| `energy_gradient` | `mean_energy_interior - mean_energy_boundary` |

### 12.2 SOC Health Indicators

| Indicator | Target | Meaning |
|-----------|--------|---------|
| Avalanche size exponent tau | ~1.5 | Power-law distribution |
| 1/f noise exponent alpha | ~1.0 | Pink noise in energy time series |
| Energy gradient sign | Positive | Interior hotter than boundary |
| Temperature stability | Low variance | Self-organized near T_c |

---

## 13. Theoretical Notes

### 13.1 Relationship to Poincare Ball Geometry

In hyperbolic space, the **volume element** at radius r is proportional to
sinh^{d-1}(r), which grows exponentially. This means most nodes live near the
boundary. The boundary dissipation rule naturally accounts for this: the
majority of energy dissipation occurs where the majority of nodes reside.

### 13.2 Relationship to Free Energy Principle

The Helmholtz free energy F = U - TS is minimized by the system (Friston's
free energy principle). Boundary dissipation creates a structured energy
distribution (low entropy S at the node level), which initially increases F.
However, the increased temperature T (from higher energy variance) compensates,
and the system finds a new equilibrium at lower F. The net effect is that
boundary dissipation accelerates convergence to the free energy minimum.

### 13.3 Why Not Dissipate at Low-Degree Nodes Instead?

An alternative definition of "boundary" uses graph-theoretic degree: leaf nodes
(degree 1) are the boundary. However:

1. Degree is not a stable property. Nodes gain/lose edges through L-System
   growth, Hebbian strengthening, and temporal decay.
2. Degree does not capture the conceptual hierarchy. A leaf node at ||z|| = 0.1
   might be a newly-inserted abstract concept that happens to have no edges yet
   -- it should not be treated as boundary.
3. The Poincare ball magnitude is a continuous, stable, geometry-native
   property that directly encodes hierarchical depth.

The Poincare magnitude is the correct boundary definition for this system.

---

## 14. Summary

| Aspect | Current | Proposed |
|--------|---------|----------|
| Decay target | All nodes uniformly | Position-dependent |
| Interior energy | Lost (3%/tick) | Conserved (redistributed) |
| Boundary energy | Lost (3%/tick) | Lost (3%/tick) -- same rate |
| Total dissipation | ~3% of E_total/tick | ~1.5% of E_total/tick (boundary only) |
| SOC compatibility | Violated | Satisfied |
| Conservation law | None | Bulk conservation + boundary dissipation |
| Performance | O(N) | O(N + E), same for sparse graphs |
| Backward compatible | N/A | Yes (beta=0 = current behavior) |

The change is surgical: modify `compute_energy_deltas()`, add one field to
`AttentionState`, add a redistribution pass in the engine tick. The beta blend
factor ensures a safe, gradual transition with instant rollback capability.

---

## Appendix A: Notation Reference

| Symbol | Meaning |
|--------|---------|
| z_i | Poincare ball embedding of node i |
| \|\|z_i\|\| | Euclidean norm (magnitude) of embedding |
| E_i | Energy of node i |
| gamma | Decay rate (0.03, from `1 - energy_decay`) |
| alpha_i | Dissipation fraction for node i, in [0, 1] |
| R_inner | Inner radius of transition zone (0.70) |
| R_boundary | Outer radius / full dissipation threshold (0.85) |
| beta | Migration blend factor [0, 1] |
| N(i) | Set of neighbors of node i |
| degree(i) | Number of neighbors of node i |
| M_i | Semantic mass = energy * ln(degree + 1) |
