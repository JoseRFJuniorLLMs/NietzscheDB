# Glauber Dynamics for Valence Spin-Flip Mechanism

**Phase**: Design (pre-implementation)
**Date**: 2026-03-11
**Status**: Proposed
**Depends on**: Phase XIII (Cognitive Thermodynamics), Phase XII (ECAN), Poincare ball geometry

---

## 1. Motivation

Currently, node valence in NietzscheDB is set directly via `set_emotion()` (in
`crates/nietzsche-graph/src/valence.rs`) with no dynamics. Valence is a static
scalar that changes only on external writes or via `reinforce_emotion()`. This
means the emotional landscape of the knowledge graph lacks the spontaneous
collective phenomena that emerge from local interactions: there are no emotional
phase transitions, no spontaneous symmetry breaking, no critical fluctuations.

Glauber dynamics provides a principled stochastic mechanism for valence evolution
that couples each node's emotional polarity to its neighbors via the graph
topology. At low temperature the graph exhibits **emotional ordering** (coherent
sentiment clusters), while at high temperature valence becomes random noise. At
the critical temperature, the system exhibits **scale-free fluctuations** --
precisely the self-organized criticality (SOC) regime that the agency engine
seeks to maintain.

---

## 2. Mathematical Formulation

### 2.1 Classical Ising Model (Discrete Spins)

In the standard Ising model, each site i carries a spin sigma_i in {-1, +1}.
The Hamiltonian is:

```
H = -sum_{(i,j)} J_ij * sigma_i * sigma_j
```

Glauber dynamics flips spin i with probability:

```
P(sigma_i -> -sigma_i) = 1 / (1 + exp(2 * beta * sigma_i * h_i))
```

where `h_i = sum_j J_ij * sigma_j` is the local field at site i, and
`beta = 1/T` is the inverse temperature.

### 2.2 Continuous Extension for NietzscheDB Valence

NietzscheDB valence v_i lives in [-1.0, +1.0], not {-1, +1}. We generalize
Glauber dynamics to continuous spins using the **Langevin-Glauber** formulation:

```
dv_i/dt = -gamma * dH/dv_i + eta_i(t)
```

Discretized per agency tick:

```
v_i(t+1) = clamp( v_i(t) + delta_v_i , -1.0, +1.0 )
```

where:

```
delta_v_i = -gamma * (v_i - tanh(beta * h_i)) + sigma_noise * xi_i
```

The term `tanh(beta * h_i)` is the mean-field equilibrium valence: the value
v_i would relax to if it could freely align with its local field. The noise
term `xi_i` is drawn from a standard normal distribution, scaled by
`sigma_noise = sqrt(2 * gamma / beta)` to satisfy detailed balance at
temperature T.

**Key parameters:**

| Symbol         | Meaning                                  | Default     |
|----------------|------------------------------------------|-------------|
| `gamma`        | Relaxation rate (how fast v_i tracks h_i)| 0.1         |
| `beta`         | Inverse temperature (1/T from thermo)    | computed     |
| `sigma_noise`  | Thermal noise amplitude                  | sqrt(2*gamma/beta) |
| `J_ij`         | Coupling strength between nodes i,j      | f(d_H(i,j)) |

### 2.3 Discrete Spin-Flip Mode (Optional)

For backward compatibility or performance, a discrete mode can be offered where
valence is quantized to {-1, 0, +1} and the classic Glauber formula is used
directly. The `tanh(beta * h_i)` already maps to this regime when beta is large.

---

## 3. Computing the Local Field h_i from Hyperbolic Neighbors

### 3.1 Coupling Kernel J_ij

The coupling J_ij between nodes i and j must satisfy two properties:

1. **Proximity preference**: nearby nodes influence each other more.
2. **Hierarchy respect**: coupling should decay with Poincare distance.

We define:

```
J_ij = w_ij * exp(-alpha * d_H(i,j))
```

where:
- `d_H(i,j)` is the Poincare ball distance (`PoincareVector::distance()`)
- `w_ij` is the edge weight from the adjacency index
- `alpha` is the coupling decay rate (default: 1.0)

The exponential decay is natural for hyperbolic space: Poincare distance grows
logarithmically near the boundary, so the decay is mild for same-depth nodes
but steep across hierarchical levels. This means **sibling nodes** (same depth,
close angular position) strongly couple, while **ancestor-descendant** pairs
decouple -- matching the intuition that emotional polarity is more contagious
among peers than across abstraction layers.

### 3.2 Local Field Computation

For node i with neighbor set N(i) from the adjacency index:

```
h_i = sum_{j in N(i)} J_ij * v_j
```

where v_j is the valence of neighbor j.

**Practical details:**
- Use `AdjacencyIndex::neighbors_out(i)` plus `neighbors_in(i)` for the
  undirected neighborhood (valence coupling should be symmetric).
- Load `NodeMeta` only (not full `Node`) -- this avoids the 24KB embedding
  load per neighbor. Valence is in `NodeMeta`.
- For Poincare distance computation, we need embeddings of both endpoints.
  This is expensive. **Optimization**: use precomputed edge metadata or cache
  distances during the gravity/training phases. Alternatively, use the edge
  weight as a proxy for coupling strength when embeddings are not loaded:
  `J_ij = w_ij` (fast path) vs `J_ij = w_ij * exp(-alpha * d_H)` (full path).

### 3.3 Arousal as Coupling Amplifier

High-arousal nodes are emotionally "louder" -- they exert more influence on
their neighbors. This is consistent with the existing `arousal_modulated_bias()`
in valence.rs. We modulate:

```
J_ij_effective = J_ij * (1 + arousal_j)
```

A node with `arousal = 1.0` doubles its influence on neighbors' valence, while
a calm node (`arousal = 0.0`) contributes at base coupling.

---

## 4. Relationship to the Existing Temperature System

### 4.1 Temperature Source

The cognitive temperature T is computed by `thermodynamics.rs` as:

```
T = sigma_E / mean_E    (coefficient of variation of energy distribution)
```

This T is already tracked via `ThermodynamicState::ema_temperature()` with
exponential smoothing. Glauber dynamics uses `beta = 1/T` directly from this
EMA value.

**Edge case**: when T = 0 (dead graph), set `beta = beta_max` (a configurable
cap, default 100.0) to prevent division by zero. This corresponds to a "frozen"
graph with no valence flips.

### 4.2 Phase Coupling

The thermodynamic phase classification (`Solid`, `Liquid`, `Gas`, `Critical`)
directly predicts Glauber behavior:

| Phase    | Temperature | beta    | Glauber Behavior                           |
|----------|------------|---------|---------------------------------------------|
| Solid    | T < 0.15   | > 6.67  | Ordered: valence clusters frozen, no flips  |
| Liquid   | 0.15..0.85 | 1.18..6.67 | Dynamic: moderate flipping, local order  |
| Gas      | T > 0.85   | < 1.18  | Disordered: random valence, frequent flips  |
| Critical | T ~ T_c    | ~ beta_c | Scale-free fluctuations, power-law corr.  |

### 4.3 Free Energy Integration

The Glauber dynamics adds a **valence-dependent term** to the Helmholtz free
energy:

```
F_total = F_energy + F_valence
F_valence = -T * S_valence - sum_{(i,j)} J_ij * v_i * v_j
```

where `S_valence` is the entropy of the valence distribution. This can be
reported alongside the existing thermodynamic metrics but does NOT require
modifying the existing `helmholtz_free_energy()` function -- it is tracked
separately in the Glauber subsystem.

---

## 5. Phase Transition Predictions

### 5.1 Mean-Field Critical Temperature

For the Ising model on a graph with mean degree z and mean coupling J_mean:

```
T_c = z * J_mean / atanh(1/z)
```

For large z this simplifies to `T_c ~ z * J_mean`.

Typical NietzscheDB collections have:
- Mean degree z ~ 5-20 (from adjacency statistics)
- Mean coupling J_mean ~ 0.3-0.7 (depending on edge weights and distances)

This gives T_c estimates in the range **1.5 - 14.0**, which is above the
thermodynamic "hot" threshold (T_hot = 0.85). This means the valence phase
transition occurs in the **gas phase** of the energy distribution.

**Implication**: To observe ordered valence states, either:
1. The coupling must be strong enough (high J), or
2. The temperature definition must be adapted.

**Resolution**: Use an **effective temperature** for valence that decouples
from the energy temperature:

```
T_valence = alpha_v * T_energy + (1 - alpha_v) * T_intrinsic
```

where `T_intrinsic` is derived from valence variance itself (analogous to how
T_energy is the coefficient of variation of energy). The mixing parameter
`alpha_v` (default: 0.5) controls how much the energy landscape drives
emotional dynamics vs. valence having its own thermal scale.

### 5.2 SOC Criticality Preservation

The agency engine targets the critical regime (T ~ T_c) for maximal
information processing capacity. The Glauber subsystem should **not** fight
the thermodynamic controller. Rules:

1. If the thermodynamic phase is `Solid`, Glauber is effectively frozen.
   No valence flips occur (gamma -> 0).
2. If the thermodynamic phase is `Gas`, Glauber noise dominates. Valence
   is nearly random, consistent with high exploration.
3. If the thermodynamic phase is `Liquid` or `Critical`, Glauber produces
   meaningful emotional dynamics with local order and fluctuations.

To maintain SOC, the Glauber subsystem should report its own "magnetization"
(mean |v_i|) and "susceptibility" (variance of valence) to the thermodynamic
dashboard. Divergent susceptibility signals that the graph is near the valence
phase transition -- this is informative for the observer/reactor.

### 5.3 Observable Metrics

| Metric              | Formula                               | Interpretation                     |
|---------------------|---------------------------------------|-------------------------------------|
| Magnetization M     | (1/N) * sum_i v_i                     | Net emotional polarity              |
| Absolute M          | (1/N) * sum_i |v_i|                   | Degree of emotional ordering        |
| Susceptibility chi  | N * (mean(v^2) - mean(v)^2)           | Response to perturbation            |
| Valence entropy     | -sum p_k * ln(p_k) on histogram bins  | Diversity of emotional states       |
| Correlation length  | decay scale of v_i * v_j vs d_H(i,j) | Spatial extent of emotional order   |

---

## 6. Integration Points in the Agency Engine Tick

### 6.1 Tick Phase Assignment

Glauber dynamics should run as **Phase 28** in the agency engine tick, after
the thermodynamic cycle (Phase 13/14) provides the current temperature, and
after ECAN/Hebbian have updated energies and edge weights. The ordering:

```
Phase 13: Thermodynamics   -> provides T, phase state
Phase 14: Gravity          -> provides semantic mass context
...
Phase 28: Glauber Dynamics -> uses T, adjacency, valence, arousal
```

### 6.2 Integration Contract

**Input** (read-only from the graph):
- `ThermodynamicState::ema_temperature()` for beta = 1/T
- `AdjacencyIndex` for neighbor lookups
- `GraphStorage::get_node_meta()` for valence, arousal of each node
- `GraphStorage::get_embedding()` for Poincare distance (optional, slow path)
- Edge weights from `AdjEntry::weight`

**Output** (via `AgencyIntent`):
- `AgencyIntent::UpdateValence { node_id: Uuid, new_valence: f32, delta: f32 }`:
  a new intent variant for the server to apply under write lock.
- `GlauberReport` added to `AgencyTickReport::glauber_report: Option<GlauberReport>`

### 6.3 New Intent Variant

Add to `AgencyIntent` in `reactor.rs`:

```rust
/// Update a node's emotional valence via Glauber dynamics.
/// Produced when: valence spin-flip or relaxation changes valence by more
/// than the minimum delta threshold.
UpdateValence {
    node_id: Uuid,
    new_valence: f32,
    /// For diagnostics: how much the valence changed.
    delta: f32,
}
```

### 6.4 GlauberReport Structure

```rust
pub struct GlauberReport {
    /// Number of nodes evaluated.
    pub nodes_evaluated: usize,
    /// Number of nodes whose valence actually changed.
    pub flips: usize,
    /// Mean absolute valence delta across all evaluated nodes.
    pub mean_delta: f32,
    /// Magnetization: mean(v_i).
    pub magnetization: f64,
    /// Absolute magnetization: mean(|v_i|).
    pub abs_magnetization: f64,
    /// Susceptibility: N * var(v).
    pub susceptibility: f64,
    /// Effective beta used (1/T).
    pub beta: f64,
    /// Valence entropy.
    pub valence_entropy: f64,
}
```

### 6.5 Event Bus Integration

Emit `AgencyEvent::ValencePhaseTransition` when the absolute magnetization
crosses a threshold (e.g., 0.5 -> ordered, below 0.2 -> disordered). This
allows the reactor and observer to respond to collective emotional shifts.

### 6.6 Interaction with Existing Valence Systems

The `reinforce_emotion()` function in `valence.rs` applies external valence
shifts (e.g., from user input or emotional context). Glauber dynamics is the
*internal* dynamics that runs between external perturbations. They compose:

```
v_i(t+1) = clamp(v_i(t) + delta_external + delta_glauber, -1.0, +1.0)
```

External reinforcement takes priority (applied first in the tick), then
Glauber relaxes the valence distribution toward thermodynamic equilibrium.

The existing `valence_edge_modifier()` (which boosts heat conductivity between
same-polarity nodes) creates a feedback loop: Glauber ordering increases
same-polarity clustering, which increases heat conductivity, which increases
energy coherence, which lowers temperature, which increases Glauber ordering.
This positive feedback must be monitored for runaway effects (see Section 8).

---

## 7. Performance Considerations

### 7.1 Per-Node Computation Budget

For each node i, the Glauber update requires:
1. Load `NodeMeta` for i: ~100 bytes, O(1) RocksDB lookup.
2. Load neighbor list: `AdjacencyIndex::neighbors_out(i)` + `neighbors_in(i)`,
   O(degree) in-memory.
3. For each neighbor j, load `NodeMeta` for j: ~100 bytes, O(1) per neighbor.
4. Optionally load embeddings for Poincare distance: ~24KB each, expensive.
5. Compute h_i = sum of J_ij * v_j: O(degree) arithmetic.
6. Compute delta_v_i: O(1) arithmetic (tanh, exp, random).

**Cost per node** (fast path, no embedding loads):
- ~degree * 100 bytes of metadata reads
- ~degree multiply-adds
- For degree ~10: ~1KB reads + 10 FLOPs. Trivial.

**Cost per node** (full path, with Poincare distance):
- ~degree * 24KB embedding reads (expensive!)
- ~degree * dim multiply-adds for distance
- For degree ~10, dim ~128: ~240KB reads + ~1280 FLOPs.

### 7.2 Sampling Strategy

For collections with N > 10,000 nodes, evaluating every node per tick is
prohibitive. Use the **DirtySet** (Phase XV) to select candidates:

1. **Dirty nodes**: nodes whose energy, edges, or valence changed recently.
   Always evaluate these.
2. **Random sample**: from the clean set, evaluate a fixed-size sample
   (default: `glauber_max_scan = 2000`).
3. **Priority by arousal**: high-arousal nodes flip more frequently and
   should be sampled more often. Use arousal as sampling weight.

This gives amortized O(dirty + sample_size) per tick instead of O(N).

### 7.3 Embedding Distance Cache

To avoid repeated embedding loads for Poincare distance:
- Maintain a tick-local `HashMap<(Uuid, Uuid), f64>` of distances computed
  during the gravity phase. If gravity has already computed d_H(i,j), reuse it.
- If no cached distance is available, fall back to `J_ij = w_ij` (the edge
  weight proxy).

### 7.4 Interval Gating

Like all agency phases, Glauber is interval-gated:
- Default interval: **8 ticks** (runs every 8th agency tick).
- At 60-second tick interval, this means valence updates every ~8 minutes.
- This is appropriate because emotional dynamics should be slower than
  energy/attention dynamics (ECAN runs every tick).

### 7.5 Batch Updates

Collect all `UpdateValence` intents into a batch and apply them atomically
under the write lock. This prevents partial updates where some nodes see
new valence while others still have old values (consistency within a tick).

---

## 8. Risks and Mitigations

### 8.1 Runaway Ferromagnetic Ordering

**Risk**: The positive feedback loop between Glauber ordering and valence edge
modifier (Section 6.6) could drive the graph into a fully ordered state where
all valence = +1 or all valence = -1, destroying information.

**Mitigation**:
- **Gamma cap**: Limit `gamma` (relaxation rate) to prevent instantaneous
  alignment. Default gamma = 0.1 means each tick moves valence at most 10%
  toward the local field equilibrium.
- **Delta cap**: Clamp `|delta_v_i|` to a maximum per tick (default: 0.05).
  This ensures no single tick can flip valence dramatically.
- **Magnetization circuit breaker**: If `|M| > 0.9` (nearly complete ordering),
  temporarily disable Glauber dynamics or inject extra noise. Emit an
  `AgencyEvent::ValenceRunaway` so the observer can react.
- **Arousal decay**: The existing `decay_arousal()` naturally reduces coupling
  amplification over time, damping the feedback loop.

### 8.2 Noise Domination at High Temperature

**Risk**: In the gas phase (T > 0.85), the noise term dominates and valence
becomes pure random walk, erasing meaningful emotional state.

**Mitigation**:
- Gate Glauber off when `phase == Gas` or when `T > T_hot + margin`. In this
  regime, valence is already meaningless and the system should focus on cooling
  (reducing exploration, not randomizing emotions).
- Alternatively, reduce gamma in the gas phase:
  `gamma_eff = gamma * min(1.0, T_hot / T)`.

### 8.3 Breaking Poincare Ball Invariants

**Risk**: Glauber dynamics modifies valence only, not embeddings. But the
existing `valence_edge_modifier()` affects edge weights in the diffusion
Laplacian, which could indirectly affect embedding training (Phase XVIII).

**Mitigation**:
- Glauber updates valence in NodeMeta only. No embedding coordinates change.
- The hyperbolic training phase (Phase XVIII) uses edge weights that already
  include valence modifiers -- this is intentional and desirable (emotional
  affinity should guide embedding learning).
- Monitor Poincare ball health via `HyperbolicHealthMonitor` (Phase X). If
  Glauber-induced weight changes cause boundary crowding or norm violations,
  the health monitor will flag it.

### 8.4 Performance Pathology on High-Degree Nodes

**Risk**: Hub nodes with degree > 500 (pre-shatter) require loading hundreds
of neighbor metadata entries to compute h_i.

**Mitigation**:
- Skip shattered nodes (ghosts/avatars from Phase XVI). Shattered super-nodes
  have delegated their topology to avatars.
- For non-shattered high-degree nodes, sample a fixed maximum of neighbors
  (default: 50) and scale h_i by `degree / sampled_count` to get an unbiased
  estimate. This caps per-node cost regardless of degree.
- The `shatter_threshold` (default: 500) should be below the Glauber sampling
  cap to ensure this rarely triggers.

### 8.5 Stale Temperature Signal

**Risk**: Temperature is computed every `thermo_interval` ticks (default: 5),
but Glauber may run on a different interval. The beta used could be stale.

**Mitigation**:
- Use the EMA-smoothed temperature (`ThermodynamicState::ema_temperature()`)
  which is updated every thermo tick and smoothed across ticks. This is stable
  enough for Glauber's purposes.
- If the thermo subsystem is disabled (`thermo_interval = 0`), fall back to
  `beta = beta_default` (default: 2.0), corresponding to a lukewarm graph
  with moderate dynamics.

### 8.6 Interaction with External Valence Writes

**Risk**: The gRPC API allows direct valence writes via UpdateNode. If an
external system sets valence to +1.0 and Glauber immediately relaxes it back
toward the local field, the external write appears to have no effect.

**Mitigation**:
- When a node's valence is externally written, mark it as "pinned" for a
  configurable number of ticks (default: 5). Pinned nodes skip Glauber
  dynamics, giving the external write time to propagate influence to neighbors
  before the local field can pull it back.
- Implement pinning via a `glauber_pin_until: Option<u64>` tick counter in a
  side map (not in NodeMeta, to avoid storage format changes).

### 8.7 Detailed Balance Violation from Arousal Coupling

**Risk**: The arousal amplifier (Section 3.3) breaks the symmetry of the
coupling: `J_ij_eff != J_ji_eff` when arousal_i != arousal_j. This violates
detailed balance, meaning the system may not relax to a true Boltzmann
equilibrium.

**Mitigation**:
- This is intentional. NietzscheDB is a driven system (constant external input,
  agency ticks, ECAN auctions), not an equilibrium system. Exact detailed
  balance is not required.
- For monitoring: compute and report the asymmetry
  `mean(|J_ij - J_ji|) / mean(J_ij)`. If this ratio exceeds 0.5, log a
  warning -- it means arousal is dominating the coupling structure.
- Optionally, symmetrize: `J_ij_sym = (J_ij_eff + J_ji_eff) / 2`. This
  restores detailed balance at the cost of losing the directionality that
  arousal provides.

---

## 9. Configuration (Environment Variables)

All variables follow the existing `AGENCY_` prefix convention from
`config.rs::AgencyConfig::from_env()`.

| Variable                          | Type  | Default | Description                                           |
|-----------------------------------|-------|---------|-------------------------------------------------------|
| `AGENCY_GLAUBER_ENABLED`          | bool  | true    | Master switch for Glauber dynamics                     |
| `AGENCY_GLAUBER_INTERVAL`         | u64   | 8       | Tick interval (run every N agency ticks)               |
| `AGENCY_GLAUBER_GAMMA`            | f64   | 0.1     | Relaxation rate toward local field equilibrium         |
| `AGENCY_GLAUBER_ALPHA`            | f64   | 1.0     | Coupling decay rate for Poincare distance              |
| `AGENCY_GLAUBER_MAX_SCAN`         | usize | 2000    | Maximum nodes to evaluate per tick                     |
| `AGENCY_GLAUBER_MAX_DELTA`        | f32   | 0.05    | Maximum |delta_v| per node per tick                    |
| `AGENCY_GLAUBER_NEIGHBOR_SAMPLE`  | usize | 50      | Max neighbors sampled for high-degree nodes            |
| `AGENCY_GLAUBER_BETA_MAX`         | f64   | 100.0   | Cap on inverse temperature (prevents div-by-zero)      |
| `AGENCY_GLAUBER_BETA_DEFAULT`     | f64   | 2.0     | Fallback beta when thermodynamics is disabled          |
| `AGENCY_GLAUBER_MAGNETIZATION_CAP`| f64   | 0.9     | |M| threshold for circuit breaker                      |
| `AGENCY_GLAUBER_PIN_TICKS`        | u64   | 5       | Ticks to pin valence after external write              |
| `AGENCY_GLAUBER_USE_POINCARE`     | bool  | false   | Use Poincare distance for J_ij (true = expensive)      |
| `AGENCY_GLAUBER_MIN_DELTA`        | f32   | 0.001   | Minimum |delta_v| to emit an UpdateValence intent      |
| `AGENCY_GLAUBER_ALPHA_V`          | f64   | 0.5     | Mixing of energy T vs intrinsic valence T              |

---

## 10. Data Flow Summary

```
                         +-------------------+
                         |  ThermodynamicState |
                         |  ema_temperature()  |-----> beta = 1/T
                         +-------------------+
                                   |
                                   v
+----------------+        +------------------+        +------------------+
| AdjacencyIndex |------->| GlauberEngine    |------->| Vec<AgencyIntent>|
| (neighbors)    |        |                  |        | ::UpdateValence  |
+----------------+        | For each node i: |        +------------------+
                          |  1. Load N(i)    |                 |
+----------------+        |  2. Compute h_i  |                 v
| GraphStorage   |------->|  3. delta_v_i    |        +------------------+
| get_node_meta()|        |  4. clamp & emit |        | Server writes    |
| (valence,      |        +------------------+        | node.valence     |
|  arousal)      |                 |                   | under write lock |
+----------------+                 v                   +------------------+
                          +------------------+
                          | GlauberReport    |
                          | M, chi, flips    |
                          +------------------+
```

---

## 11. Pseudocode

```
fn run_glauber_tick(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    thermo: &ThermodynamicState,
    config: &GlauberConfig,
    dirty: &DirtySet,
    rng: &mut impl Rng,
) -> (GlauberReport, Vec<AgencyIntent>) {

    let T = thermo.ema_temperature().max(1e-9);
    let beta = (1.0 / T).min(config.beta_max);

    // Phase gate: skip if gas phase
    if T > config.t_hot + 0.1 {
        return (GlauberReport::empty(beta), vec![]);
    }

    let mut intents = Vec::new();
    let mut sum_v = 0.0;
    let mut sum_v2 = 0.0;
    let mut flips = 0;
    let mut total_delta = 0.0;
    let mut n_eval = 0;

    // Select nodes: dirty + random sample
    let node_ids = select_candidates(storage, dirty, config.max_scan, rng);

    for node_id in &node_ids {
        let meta = storage.get_node_meta(node_id)?;
        if meta.is_phantom { continue; }
        if is_pinned(node_id) { continue; }

        // Compute local field h_i
        let neighbors = adjacency.neighbors_all(node_id);
        let sample = if neighbors.len() > config.neighbor_sample {
            sample_neighbors(&neighbors, config.neighbor_sample, rng)
        } else {
            neighbors
        };
        let scale = neighbors.len() as f64 / sample.len() as f64;

        let mut h_i = 0.0;
        for adj in &sample {
            let n_meta = storage.get_node_meta(&adj.neighbor_id)?;
            let j_ij = if config.use_poincare {
                let dist = poincare_distance(meta, n_meta)?;
                adj.weight as f64 * (-config.alpha * dist).exp()
            } else {
                adj.weight as f64
            };
            let j_eff = j_ij * (1.0 + n_meta.arousal as f64);
            h_i += j_eff * n_meta.valence as f64;
        }
        h_i *= scale;

        // Compute delta_v
        let v_eq = (beta * h_i).tanh();
        let noise = rng.sample(StandardNormal) * (2.0 * config.gamma / beta).sqrt();
        let delta = -config.gamma * (meta.valence as f64 - v_eq) + noise;
        let delta_clamped = delta.clamp(-config.max_delta as f64, config.max_delta as f64);
        let new_v = (meta.valence as f64 + delta_clamped).clamp(-1.0, 1.0);

        // Emit intent if change is significant
        if (new_v - meta.valence as f64).abs() > config.min_delta as f64 {
            intents.push(AgencyIntent::UpdateValence {
                node_id: *node_id,
                new_valence: new_v as f32,
                delta: delta_clamped as f32,
            });
            flips += 1;
        }

        sum_v += new_v;
        sum_v2 += new_v * new_v;
        total_delta += delta_clamped.abs();
        n_eval += 1;
    }

    let n = n_eval as f64;
    let magnetization = if n > 0.0 { sum_v / n } else { 0.0 };
    let abs_mag = if n > 0.0 { sum_v.abs() / n } else { 0.0 };
    let susceptibility = if n > 1.0 {
        n * (sum_v2 / n - (sum_v / n).powi(2))
    } else {
        0.0
    };

    // Circuit breaker
    if abs_mag > config.magnetization_cap {
        tracing::warn!(M = abs_mag, "valence magnetization runaway detected");
        // inject extra noise or disable for this tick
    }

    let report = GlauberReport {
        nodes_evaluated: n_eval,
        flips,
        mean_delta: (total_delta / n.max(1.0)) as f32,
        magnetization,
        abs_magnetization: abs_mag,
        susceptibility,
        beta,
        valence_entropy: compute_valence_entropy(storage, config),
    };

    (report, intents)
}
```

---

## 12. Testing Strategy

### 12.1 Unit Tests (no storage)

- **Tanh equilibrium**: For a node with h_i = 1.0 and beta = 2.0, verify that
  repeated Glauber updates converge v_i toward tanh(2.0) ~ 0.964.
- **Zero field**: For an isolated node (no neighbors), verify that valence
  performs a bounded random walk toward 0.
- **Symmetry**: Two identical nodes connected by a strong edge should
  synchronize valence (both positive or both negative).
- **Delta clamping**: Verify that |delta_v| never exceeds max_delta.
- **Circuit breaker**: Verify that magnetization > 0.9 triggers the breaker.

### 12.2 Integration Tests (with mock storage)

- **Small graph ordering**: Create a complete graph of 10 nodes with strong
  coupling (J = 1.0) at low temperature (beta = 10). After 100 ticks, verify
  that |M| > 0.8 (ordered phase).
- **Small graph disordering**: Same graph at high temperature (beta = 0.1).
  After 100 ticks, verify that |M| < 0.2 (disordered phase).
- **Phase transition sweep**: Vary beta from 0.1 to 10.0 and verify that
  susceptibility peaks near the theoretical T_c for the graph.
- **Pinning**: Set a node's valence to +1.0, pin it, verify it stays at +1.0
  for `pin_ticks` and then begins relaxing.

---

## 13. Dashboard API Extension

Add the following endpoint to the HTTP dashboard:

```
GET /api/agency/glauber?collection=X
```

Returns:

```json
{
    "enabled": true,
    "last_tick": 1234,
    "beta": 2.35,
    "magnetization": 0.12,
    "abs_magnetization": 0.45,
    "susceptibility": 3.21,
    "valence_entropy": 1.82,
    "flips_last_tick": 42,
    "nodes_evaluated": 2000,
    "pinned_nodes": 3,
    "circuit_breaker_active": false
}
```

---

## 14. Implementation Roadmap

1. **Phase A**: Add `GlauberConfig` to `AgencyConfig` with all env vars.
   Add `UpdateValence` to `AgencyIntent`. No logic yet -- pure types.

2. **Phase B**: Implement `glauber.rs` with the core local field computation
   and delta_v formula. Unit tests for the math layer.

3. **Phase C**: Integrate into `engine.rs` as Phase 28. Wire up temperature,
   adjacency, storage reads. Emit `UpdateValence` intents.

4. **Phase D**: Server handler applies `UpdateValence` intents. Integration
   tests with mock collections.

5. **Phase E**: Dashboard endpoint. Metrics in the morning report.

6. **Phase F**: Pinning mechanism. External write detection.

7. **Phase G**: Full path with Poincare distance coupling (behind
   `AGENCY_GLAUBER_USE_POINCARE=true` flag).

---

## 15. Open Questions

1. **Should Glauber also update arousal?** Currently only valence is modeled
   as a spin. Arousal could be a second coupled field (XY model), but this
   doubles complexity. Recommendation: valence-only for v1, consider arousal
   coupling in v2 if the emotional dynamics are too one-dimensional.

2. **Should phantoms participate?** Phantom nodes retain topology but are
   excluded from KNN. They could still participate in valence coupling (their
   emotional trace lingers) or be fully excluded. Recommendation: exclude
   phantoms from evaluation but include their valence when computing neighbors'
   local field (ghost emotions still influence the living).

3. **Antiferromagnetic coupling?** The current model assumes ferromagnetic
   coupling (J > 0: neighbors prefer same polarity). Some edge types
   (e.g., "contradicts", "opposes") might warrant antiferromagnetic coupling
   (J < 0: neighbors prefer opposite polarity). This is straightforward to
   implement by allowing negative J for specific EdgeType variants.

4. **Coupling to embedding training?** Should the contrastive loss in
   hyperbolic training (Phase XVIII) include a valence similarity term? This
   would make same-valence nodes attract in embedding space. Attractive but
   risks conflating semantic similarity with emotional similarity.
