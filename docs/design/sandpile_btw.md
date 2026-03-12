# BTW Sandpile: Self-Organized Criticality for NietzscheDB

**Status**: Design
**Phase**: 28 (proposed)
**Author**: Claude Opus 4.6
**Date**: 2026-03-11

---

## 1. Motivation

NietzscheDB already models energy as a scalar field over the knowledge graph, with
multiple systems driving, dissipating, and redistributing it:

| System | Role | Phase |
|--------|------|-------|
| ECAN (attention economy) | Drives energy via auction bids | XII |
| Hebbian LTP | Strengthens edges on co-activation | XII.5 |
| Cognitive Thermodynamics | Heat flow from hot to cold nodes | XIII |
| Semantic Gravity | Pulls energy toward gravity wells | XIV |
| Temporal Decay | Exponential edge weight decay | B1 |
| Criticality Detector | Classifies rigid/critical/turbulent regime | AGI |

What is **missing** is a mechanism that produces **power-law distributed avalanches**
--- the hallmark of Self-Organized Criticality (SOC). The Bak-Tang-Wiesenfeld (BTW)
sandpile is the canonical model: energy accumulates grain by grain until a threshold
is exceeded, triggering a topple that may cascade across the graph, producing
avalanches whose size distribution follows P(S) ~ S^{-tau}.

SOC is the *only* known mechanism that naturally drives a system to the edge of chaos
**without external tuning**. NietzscheDB's existing criticality detector can
*measure* the regime, but it cannot *steer* toward it. The BTW sandpile fills this
gap: it is the **self-tuning engine** that makes the criticality detector's "Critical"
regime an **attractor** rather than a coincidence.

---

## 2. BTW-to-NietzscheDB Mapping

### 2.1 Classical BTW Rules (on a lattice)

Given a lattice site i with height z(i) and critical threshold z_c:

```
if z(i) >= z_c:
    z(i) -= z_c
    for each neighbor j of i:
        z(j) += z_c / degree(i)
```

Boundary sites dissipate: energy that would flow off the lattice is lost.

### 2.2 Adaptation to the Hyperbolic Knowledge Graph

| BTW Concept | NietzscheDB Mapping | Notes |
|-------------|---------------------|-------|
| z(i) | `node.energy` (f32, [0, 1]) | Already exists in NodeMeta |
| z_c | `sandpile_critical_energy` | Configurable, default 0.95 |
| degree(i) | `adj.degree_out(id) + adj.degree_in(id)` | Total degree from AdjacencyIndex |
| neighbor j | All nodes connected by edges (in + out) | Undirected for toppling purposes |
| lattice boundary | Nodes with embedding norm > boundary_radius | Poincare ball boundary = dissipation zone |

### 2.3 Toppling Rule

When a node i has `energy >= z_c`:

```
delta = z_c / degree(i)
energy(i) -= z_c
for each neighbor j:
    if is_boundary(j):
        // energy dissipated (lost from system)
        dissipated += delta
    else:
        energy(j) += delta
```

Where `is_boundary(j)` is true when the L2 norm of j's embedding exceeds
`boundary_radius` (default 0.9). This is the **open boundary condition** that
ensures the system reaches a stationary critical state rather than accumulating
energy indefinitely.

### 2.4 Poincare Ball Considerations

In the Poincare ball model, the *magnitude* of the embedding encodes
*hierarchical depth*: nodes near the origin are abstract concepts (shallow), while
nodes near the boundary are concrete instances (deep). This creates a natural
interpretation for the dissipation boundary:

- **Boundary nodes** (||embedding|| > 0.9) are *leaves* --- concrete, peripheral
  knowledge. Energy that reaches them dissipates, modeling the idea that attention
  on maximally specific facts has diminishing returns.

- **Core nodes** (||embedding|| < 0.3) are *abstract hubs* --- they topple and
  propagate energy inward through the hierarchy, modeling how conceptual
  reorganization cascades through abstractions.

**Important**: The sandpile phase MUST NOT modify embeddings. It only modifies
`node.energy`. Poincare ball constraints (||x|| < 1) are never at risk because
energy is a scalar metadata field, not a geometric coordinate.

---

## 3. Driving Mechanism: How Energy Enters the System

The BTW sandpile requires a slow driving force (one grain at a time) separated from
fast relaxation (avalanches). In NietzscheDB, energy enters through multiple channels
that are already implemented:

### 3.1 Existing Energy Sources (unchanged)

| Source | Mechanism | Typical Rate |
|--------|-----------|-------------|
| Node insertion | New nodes receive initial energy (default ~0.5) | Per-insert |
| ECAN attention | Auction winners boost target energy | Every tick |
| Hebbian LTP | Co-activation trace strengthens + energy spill | Every tick |
| Gravity pulls | Gravity wells boost attracted nodes | Every 3 ticks |
| Neural boost | GNN-detected structural importance | Sporadic |

### 3.2 Explicit BTW Driving (new, optional)

For collections where natural driving is insufficient to sustain criticality,
an explicit slow-drive mode injects a small energy quantum into a randomly selected
node each tick:

```
if sandpile_enable_driving:
    node = random_node(below_threshold)
    node.energy += drive_amount   // default 0.01
```

This is analogous to dropping one grain of sand on a random lattice site.

### 3.3 Timescale Separation

The critical requirement for SOC is that the **driving timescale >> relaxation
timescale**. In NietzscheDB:

- **Driving**: Once per agency tick (default 60 seconds).
- **Relaxation**: All topples in an avalanche complete within a single tick
  (synchronous BFS propagation, see Section 5).

This separation is naturally satisfied because the agency tick interval is much
longer than the in-memory BFS traversal.

---

## 4. Avalanche Dynamics

### 4.1 Synchronous Toppling (Parallel Update)

Within a single sandpile phase execution, all nodes above z_c topple simultaneously
(parallel update), matching the classical BTW model:

```
REPEAT:
    topple_set = { i : energy(i) >= z_c }
    if topple_set is empty:
        BREAK
    for each i in topple_set:
        execute topple(i)
    propagation_steps += 1
    if propagation_steps >= max_cascade_depth:
        BREAK (safety valve)
RETURN (avalanche_size, propagation_steps)
```

Where `avalanche_size` = total number of individual topple events, and
`propagation_steps` = number of parallel update rounds (the temporal duration).

### 4.2 Why Synchronous, Not Asynchronous

The classical BTW model uses synchronous updates because:

1. It guarantees a unique critical state (abelian property: final state does not
   depend on toppling order).
2. It makes avalanche measurement well-defined (size and duration are deterministic).
3. Asynchronous toppling can produce different critical exponents and complicates
   analysis.

### 4.3 Cascade Depth Limit

To prevent pathological runaway avalanches (e.g., in highly connected subgraphs),
a configurable `max_cascade_depth` (default: 100) caps the number of propagation
steps. If the limit is hit, the avalanche is truncated and a warning event is emitted.
This is analogous to a finite-size effect and does not affect the scaling behavior
for avalanches much smaller than the limit.

### 4.4 Energy Conservation

Within the bulk (non-boundary) graph, energy is strictly conserved during toppling:
each topple removes z_c from node i and distributes z_c total across its neighbors.
Energy is only lost at boundary nodes. This open-boundary dissipation is what drives
the system to criticality.

**Total energy balance per avalanche**:

```
E_after = E_before - dissipated_at_boundary
```

---

## 5. Implementation: Agency Phase 28

### 5.1 Phase Placement

The sandpile phase runs **after** all energy-modifying systems (ECAN, Hebbian,
Thermodynamics, Gravity) have completed their updates for the current tick, and
**before** the flywheel and observer. This ensures the sandpile relaxes the energy
state that was just driven by other systems.

Proposed tick ordering:

```
... existing phases 1-27 ...
Phase 28: BTW Sandpile (sandpile_interval, default: 1 = every tick)
... flywheel, observer, etc ...
```

### 5.2 Configuration

All parameters are env-var configurable following the `AGENCY_` prefix convention:

| Parameter | Env Var | Default | Description |
|-----------|---------|---------|-------------|
| `sandpile_enabled` | `AGENCY_SANDPILE_ENABLED` | `false` | Master toggle (off by default, opt-in) |
| `sandpile_interval` | `AGENCY_SANDPILE_INTERVAL` | `1` | Ticks between sandpile executions |
| `sandpile_critical_energy` | `AGENCY_SANDPILE_ZC` | `0.95` | Critical threshold z_c |
| `sandpile_boundary_radius` | `AGENCY_SANDPILE_BOUNDARY_R` | `0.9` | Embedding norm for dissipation |
| `sandpile_max_cascade` | `AGENCY_SANDPILE_MAX_CASCADE` | `100` | Max propagation steps |
| `sandpile_max_scan` | `AGENCY_SANDPILE_MAX_SCAN` | `10000` | Max nodes to scan per tick |
| `sandpile_enable_driving` | `AGENCY_SANDPILE_DRIVE` | `false` | Explicit slow-drive mode |
| `sandpile_drive_amount` | `AGENCY_SANDPILE_DRIVE_AMOUNT` | `0.01` | Energy quantum per drive |

### 5.3 Data Structures

```rust
/// Configuration for the BTW Sandpile phase.
pub struct SandpileConfig {
    pub enabled: bool,
    pub interval: u64,
    pub critical_energy: f32,      // z_c
    pub boundary_radius: f64,      // ||embedding|| threshold
    pub max_cascade_depth: usize,
    pub max_scan: usize,
    pub enable_driving: bool,
    pub drive_amount: f32,
}

/// Report from one sandpile relaxation.
pub struct SandpileReport {
    /// Number of nodes scanned.
    pub nodes_scanned: usize,
    /// Number of nodes that were above z_c at scan start.
    pub initial_supercritical: usize,
    /// Total topple events across all cascades.
    pub total_topples: usize,
    /// Number of distinct avalanches triggered.
    pub avalanche_count: usize,
    /// Sizes of each avalanche (total topples per avalanche).
    pub avalanche_sizes: Vec<usize>,
    /// Durations of each avalanche (propagation steps).
    pub avalanche_durations: Vec<usize>,
    /// Total energy dissipated at boundaries.
    pub energy_dissipated: f32,
    /// Whether any avalanche was truncated by max_cascade_depth.
    pub truncated: bool,
    /// Energy deltas to apply: (node_id, delta).
    pub energy_deltas: Vec<(Uuid, f32)>,
}

/// Persistent state across ticks.
pub struct SandpileState {
    tick_count: u64,
    /// Rolling history of avalanche sizes for P(S) estimation.
    avalanche_history: VecDeque<usize>,
    /// Estimated power-law exponent tau (updated periodically).
    estimated_tau: Option<f64>,
}
```

### 5.4 New AgencyIntent Variant

```rust
AgencyIntent::SandpileTopple {
    /// Energy adjustments: (node_id, energy_delta).
    /// Positive = gained energy from neighbor topple.
    /// Negative = lost energy from own topple or dissipation.
    energy_deltas: Vec<(Uuid, f32)>,
    /// Total energy dissipated at boundaries (leaves the system).
    dissipated: f32,
    /// Avalanche size (total topples).
    avalanche_size: usize,
}
```

### 5.5 Algorithm (Pseudocode)

```
fn run_sandpile_tick(storage, adjacency, config, state) -> SandpileReport:
    state.tick_count += 1
    if config.interval > 0 and state.tick_count % config.interval != 0:
        return None

    // Step 1: Scan all nodes, collect (id, energy, degree, embedding_norm)
    let mut nodes = Vec::new()
    for meta in storage.iter_nodes_meta():
        nodes.push((meta.id, meta.energy, degree(meta.id), embedding_norm(meta.id)))
        if nodes.len() >= config.max_scan: break

    // Step 2: Optional driving — inject one grain
    if config.enable_driving:
        pick random node with energy < z_c
        energy_deltas[node] += config.drive_amount

    // Step 3: Apply deltas from driving, then run toppling
    let mut energy_map: HashMap<Uuid, f32> = nodes into map
    let mut total_topples = 0
    let mut avalanche_sizes = Vec::new()
    let mut avalanche_durations = Vec::new()
    let mut dissipated = 0.0

    // Find initial supercritical set
    let mut active: HashSet<Uuid> = energy_map.keys()
        .filter(|id| energy_map[id] >= z_c)
        .collect()

    while !active.is_empty():
        let mut next_active = HashSet::new()
        let mut step_topples = 0

        for id in active:
            let e = energy_map[id]
            if e < z_c: continue

            let deg = degree(id)
            if deg == 0:
                // Isolated node: dissipate entire z_c
                energy_map[id] -= z_c
                dissipated += z_c
                step_topples += 1
                continue

            let delta = z_c / deg as f32
            energy_map[id] -= z_c
            step_topples += 1

            for neighbor in neighbors(id):
                if is_boundary(neighbor):
                    dissipated += delta
                else:
                    energy_map[neighbor] += delta
                    if energy_map[neighbor] >= z_c:
                        next_active.insert(neighbor)

        total_topples += step_topples
        propagation_steps += 1
        active = next_active

        if propagation_steps >= config.max_cascade_depth:
            truncated = true
            break

    // Step 4: Compute final energy deltas
    for (id, final_energy) in energy_map:
        let original = original_energy(id)
        let delta = final_energy - original
        if delta.abs() > 1e-6:
            energy_deltas.push((id, delta))

    // Step 5: Record avalanche statistics
    state.avalanche_history.push_back(total_topples)
    if state.avalanche_history.len() > 10000:
        state.avalanche_history.pop_front()

    return SandpileReport { ... }
```

### 5.6 Embedding Norm Computation

The sandpile needs to check whether a node is at the boundary. Two options:

**Option A (preferred)**: Use `meta.depth` as a proxy. The depth field is already
stored in NodeMeta and approximates the radial position. Check:
`meta.depth > config.boundary_radius`. This requires no embedding loads.

**Option B**: Load the embedding and compute L2 norm. This is expensive (~12-24 KB
per embedding for dim=128-3072) and should be avoided in the hot path.

Recommendation: Use Option A (`meta.depth`) for the boundary check in the main loop.
The `depth` field is already populated by the L-System and hyperbolic training
phases, and is stored in the compact NodeMeta struct.

---

## 6. Relationship to Existing Energy Systems

### 6.1 ECAN (Phase XII)

ECAN drives energy into the system via attention auctions. Nodes that win bids
receive energy boosts. Over time, popular hub nodes accumulate energy approaching
z_c. The sandpile then redistributes this concentrated energy via toppling,
preventing energy monopolies.

**Interaction**: ECAN drives -> Sandpile relaxes. They are complementary: ECAN
creates energy gradients (rich-get-richer), sandpile caps them (forced
redistribution at z_c).

### 6.2 Thermodynamics (Phase XIII)

Heat flow (Fourier's law) is a **continuous, gradient-based** redistribution:
q = kappa * (E_i - E_j). The sandpile is a **threshold-based, discontinuous**
redistribution: nothing happens below z_c, then sudden toppling above it.

These two mechanisms operate at different scales:
- Heat flow smooths small gradients (linear response regime).
- Sandpile toppling handles extreme accumulation (nonlinear critical regime).

**No conflict**: Heat flow operates on every edge pair. The sandpile only fires when
energy exceeds z_c. Since z_c = 0.95 is near the energy ceiling and heat flow
transfers at most `max_heat_flow = 0.02` per edge per tick, heat flow alone cannot
prevent z_c from being reached by nodes that receive sustained attention.

### 6.3 Gravity (Phase XIV)

Gravity pulls are currently experimental (default: disabled). When enabled, they
transfer energy from lighter nodes to heavier gravity wells. This **concentrates**
energy and could *increase* the frequency of sandpile topples near gravity wells ---
which is physically intuitive: massive knowledge hubs undergo the most
reorganization.

### 6.4 Criticality Detector (AGI crate)

The criticality detector classifies the system as Rigid, Critical, or Turbulent
based on spectral and energy variance measurements. The sandpile is the **actuator**
that the detector is the **sensor** for:

- **Rigid regime**: Energy variance too low. The sandpile is inactive (no nodes
  above z_c) because energy is uniformly distributed. This is correct --- the system
  needs more driving (new knowledge injection) to reach criticality.

- **Critical regime**: Power-law avalanche distribution. The sandpile fires
  intermittently, producing avalanches of all sizes. Energy variance is at the
  critical value. This is the target state.

- **Turbulent regime**: Energy variance too high. Many large avalanches. The sandpile
  is working to redistribute energy but the driving rate is too high. The solution
  is to reduce driving (e.g., lower ECAN energy gain) or increase z_c to delay
  toppling.

### 6.5 Circuit Breaker

The existing `EnergyCircuitBreaker` (max actions per tick, global energy sum
threshold) provides a safety net. The sandpile respects the circuit breaker: if the
global energy sum exceeds `circuit_breaker_energy_sum_threshold`, the sandpile phase
is skipped for that tick. This prevents the sandpile from generating unbounded
intents during already-overheated ticks.

---

## 7. Critical Exponents and Expected Scaling

### 7.1 Classical BTW Exponents

On a 2D square lattice, the BTW model produces:

| Exponent | Symbol | Value (2D) | Meaning |
|----------|--------|-----------|---------|
| Avalanche size | tau | ~1.1 | P(S) ~ S^{-tau} |
| Avalanche duration | alpha | ~1.22 | P(T) ~ T^{-alpha} |
| Avalanche area | delta | ~1.02 | P(A) ~ A^{-delta} |

### 7.2 Mean-Field BTW Exponents

On complete graphs or random graphs (mean-field approximation):

| Exponent | Symbol | Value (MF) | Meaning |
|----------|--------|-----------|---------|
| Avalanche size | tau | 3/2 = 1.5 | P(S) ~ S^{-3/2} |
| Avalanche duration | alpha | 2.0 | P(T) ~ T^{-2} |

### 7.3 Expected Behavior on NietzscheDB's Hyperbolic Graph

NietzscheDB's knowledge graph lives in the Poincare ball, which has several
properties that affect critical scaling:

1. **Exponential volume growth**: The hyperbolic plane has exponentially more area
   at larger radii. This means the "effective dimension" for wave propagation is
   infinite (more akin to a tree than a lattice). Tree-like topologies produce
   mean-field exponents.

2. **Heterogeneous degree distribution**: Knowledge graphs are scale-free (hubs
   with degree >> mean). Scale-free networks in sandpile models produce modified
   exponents that depend on the degree distribution exponent gamma.

3. **Hierarchical structure**: Concept nodes at small ||embedding|| connect to many
   leaves at large ||embedding||. Avalanches that start at the core propagate
   outward through the hierarchy and dissipate at the boundary leaves.

**Expected regime**: tau between 1.2 and 1.5, interpolating between 2D lattice
and mean-field depending on graph connectivity. The boundary dissipation at
||embedding|| > 0.9 creates an effective "finite size" that steepens the exponent
slightly.

### 7.4 Tau Estimation

After accumulating sufficient avalanche history (>1000 events), the sandpile state
estimates tau using the maximum likelihood estimator for discrete power laws
(Clauset-Shalizi-Newman method):

```
tau_hat = 1 + n * [ sum_{i=1}^{n} ln(S_i / S_min) ]^{-1}
```

where S_min is the minimum avalanche size cutoff (default: 5 topples) to avoid
small-avalanche bias. This estimate is stored in `SandpileState.estimated_tau` and
reported in the cognitive dashboard.

---

## 8. Performance Analysis

### 8.1 Per-Tick Cost

The dominant cost is the BFS-based topple propagation:

- **Node scan**: O(N) to find supercritical nodes. Using `iter_nodes_meta()` which
  loads only the compact NodeMeta (no embeddings). For 10K nodes: ~1ms.

- **Topple propagation**: O(S) where S is the total avalanche size. Each topple
  requires a degree lookup (O(1) from AdjacencyIndex) and iteration over neighbors
  (O(deg)). For a typical avalanche of size 50 on nodes with mean degree 10: ~500
  neighbor lookups, <1ms.

- **Total**: For most ticks, the sandpile phase costs <5ms. Large avalanches
  (S > 1000) could take 10-50ms, still well within the 60s tick budget.

### 8.2 Memory

- `SandpileState.avalanche_history`: VecDeque of up to 10K usize values = 80 KB.
- Per-tick working set: HashMap of (Uuid, f32) for energy tracking = 24 bytes x N.
  For 10K nodes: 240 KB.
- Total persistent memory: <1 MB.

### 8.3 Worst Case

In the worst case (maximally connected subgraph where every node is above z_c),
a single topple can trigger N-1 neighbor updates, and the cascade can propagate
for max_cascade_depth steps. Cost: O(max_cascade_depth * sum_of_degrees).

With the default max_cascade_depth = 100 and max_scan = 10000, the absolute
worst case is 100 * 10000 * mean_degree iterations. For mean_degree = 10:
10M operations, taking ~100ms. The circuit breaker and cascade depth limit
prevent this from becoming problematic.

---

## 9. Measurement and Observability

### 9.1 Avalanche Telemetry

Each sandpile tick produces a `SandpileReport` that is stored in the
`AgencyTickReport` and exposed via the cognitive dashboard:

```
GET /api/agency/sandpile?collection=X
```

Response includes:
- Current tick's avalanche count, sizes, durations
- Cumulative dissipation
- Estimated tau (if sufficient history)
- Whether any cascade was truncated

### 9.2 Power-Law Verification

The dashboard endpoint also returns the avalanche size histogram (binned
logarithmically) and the estimated tau. Operators can verify SOC by checking:

1. The histogram is approximately linear on a log-log plot.
2. tau is in the range [1.0, 2.0].
3. Avalanche sizes span at least two orders of magnitude.

### 9.3 Integration with Existing Metrics

The sandpile report feeds into:

- **Thermodynamic state**: Total dissipated energy reduces the system's internal
  energy U, which affects temperature T = sigma_E / mean_E.

- **Flywheel**: Avalanche frequency is a subsystem health indicator. Too many
  truncated avalanches = system is supercritical (needs higher z_c or lower drive).

- **World Model**: Avalanche statistics are recorded as environmental observations,
  enabling anomaly detection on SOC disruption.

- **Observation frame**: Active toppling nodes can be visualized as "flashing" in
  the 3D Poincare ball visualization, showing avalanche propagation in real-time.

---

## 10. Configuration Tuning Guide

### 10.1 Starting Configuration (conservative)

```env
AGENCY_SANDPILE_ENABLED=true
AGENCY_SANDPILE_INTERVAL=1
AGENCY_SANDPILE_ZC=0.95
AGENCY_SANDPILE_BOUNDARY_R=0.9
AGENCY_SANDPILE_MAX_CASCADE=100
AGENCY_SANDPILE_MAX_SCAN=10000
AGENCY_SANDPILE_DRIVE=false
AGENCY_SANDPILE_DRIVE_AMOUNT=0.01
```

### 10.2 Tuning z_c

- **z_c too high** (e.g., 0.99): Sandpile rarely fires. System accumulates energy
  without redistribution. Risk of energy monopolies at hub nodes.

- **z_c too low** (e.g., 0.5): Sandpile fires too frequently. Massive avalanches,
  high dissipation, system constantly in relaxation. Nodes cannot accumulate
  meaningful energy differences.

- **Sweet spot** (0.85-0.95): Fires intermittently, produces power-law distributed
  avalanches, maintains energy heterogeneity.

### 10.3 Tuning boundary_radius

- **boundary_radius too low** (e.g., 0.5): Too many boundary nodes, too much
  dissipation. System drains energy rapidly, collapses to the rigid regime.

- **boundary_radius too high** (e.g., 0.99): Too few boundary nodes. Energy cannot
  leave the system, accumulates without limit. The sandpile churns energy internally
  without reaching a stationary state.

- **Sweet spot** (0.85-0.95): Matches the natural hierarchical structure where
  leaf nodes are at high embedding norms.

### 10.4 Interaction with Thermodynamics

If the thermodynamic phase detects a Gas (hot) state after the sandpile runs,
consider increasing z_c to reduce avalanche energy redistribution. If the state
is Solid (cold), consider decreasing z_c or enabling explicit driving. This
feedback loop can be automated in a future phase.

---

## 11. Safety and Invariants

### 11.1 Energy Bounds

- Node energy MUST remain in [0.0, 1.0] after toppling. Clamp after each delta
  application: `energy = (energy + delta).clamp(0.0, 1.0)`.

- Toppling cannot drive energy negative because z_c is subtracted only from nodes
  with energy >= z_c, and z_c <= 1.0.

- Neighbor energy gain is z_c / degree, which for degree >= 1 is at most z_c.
  Combined with the 1.0 clamp, no overflow.

### 11.2 Poincare Ball Invariant

The sandpile NEVER modifies embeddings. It only modifies `node.energy`. The
Poincare ball constraint (||embedding|| < 1) is never at risk.

### 11.3 No Negative Energy

If a node's energy after all deltas would be negative (due to floating-point
accumulation), clamp to 0.0. This should be extremely rare given that toppling
only removes z_c from nodes that had energy >= z_c.

### 11.4 Abelian Property

The synchronous update scheme preserves the abelian property of the BTW model:
the final stable state is independent of the order in which simultaneously
supercritical nodes are processed. This is guaranteed because each topple's
effect is purely additive and determined only by the node's degree.

### 11.5 Toggleability

The entire phase is gated behind `AGENCY_SANDPILE_ENABLED` (default: false).
When disabled, zero code paths are executed. This makes it a safe opt-in for
collections where SOC dynamics are desired.

---

## 12. Future Extensions

### 12.1 Adaptive z_c

Instead of a fixed threshold, z_c could be dynamically adjusted based on the
current criticality regime:

- Rigid -> lower z_c to promote toppling.
- Turbulent -> raise z_c to suppress avalanches.
- Critical -> hold z_c steady.

This creates a closed-loop SOC controller.

### 12.2 Directed Toppling

Instead of distributing z_c/degree equally to all neighbors, weight the
distribution by edge weight or Poincare distance:

```
delta_j = z_c * (w_ij / sum_j w_ij)
```

This preserves the total energy transfer while biasing redistribution toward
strongly-connected or semantically-close neighbors. It could produce different
universality classes.

### 12.3 Sandpile-Driven Edge Creation

When an avalanche of size S > S_threshold crosses a community boundary (detected
via Louvain partition), propose a new edge along the avalanche path. This turns
the sandpile into a **structural growth mechanism** in addition to its energy
redistribution role.

### 12.4 Multi-Scale Sandpiles

Run sandpiles at different z_c thresholds simultaneously, each operating on
different aspects of the energy landscape:

- z_c = 0.95 for raw energy (current proposal).
- z_c_w = max_weight for edge weights (topple = redistribute edge weight when
  it exceeds a threshold).
- z_c_degree = degree_threshold for structural toppling (shatter protocol
  integration).

### 12.5 Hyperbolic-Aware Dissipation

Instead of binary boundary/not-boundary, use a continuous dissipation rate
proportional to the conformal factor:

```
dissipation_rate(j) = 1 - (1 - ||embedding_j||^2)^2
```

This smoothly increases dissipation near the boundary of the Poincare ball,
matching the Riemannian volume form. Nodes deep in the hierarchy (high ||x||)
dissipate more energy, while core concepts conserve it.

---

## 13. References

1. Bak, P., Tang, C., Wiesenfeld, K. (1987). "Self-organized criticality: An
   explanation of 1/f noise." Physical Review Letters, 59(4), 381-384.

2. Dhar, D. (1990). "Self-organized critical state of sandpile automaton models."
   Physical Review Letters, 64(14), 1613-1616.

3. Goh, K.I., Lee, D.S., Kahng, B., Kim, D. (2003). "Sandpile on scale-free
   networks." Physical Review Letters, 91(14), 148701.

4. Clauset, A., Shalizi, C.R., Newman, M.E.J. (2009). "Power-law distributions
   in empirical data." SIAM Review, 51(4), 661-703.

5. Krioukov, D. et al. (2010). "Hyperbolic geometry of complex networks."
   Physical Review E, 82(3), 036106.

---

## Appendix A: Phase Numbering Context

| Phase | Name | Status |
|-------|------|--------|
| XII | ECAN (attention economy) | Implemented |
| XII.5 | Hebbian LTP | Implemented |
| XIII | Cognitive Thermodynamics | Implemented |
| XIV | Semantic Gravity | Implemented |
| XV | DirtySet (adaptive sampling) | Implemented |
| XVI | Shatter Protocol | Implemented |
| XVII | Hydraulic Flow | Implemented |
| XVIII | Hyperbolic Training | Implemented |
| XIX | Self-Healing | Implemented |
| XX | Graph Learning | Implemented |
| XXI | Knowledge Compression | Implemented |
| XXII | Hyperbolic Sharding | Implemented |
| XXIII | World Model | Implemented |
| XXIV | Cognitive Flywheel | Implemented |
| B1 | Temporal Edge Decay | Implemented |
| C | Autonomous Graph Growth | Implemented |
| E | Cognitive Layer | Implemented |
| 27 | Epistemic Evolution | Implemented |
| **28** | **BTW Sandpile (SOC)** | **This Design** |
