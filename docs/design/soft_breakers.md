# Design: Replacing Hard Circuit Breakers with Soft Breakers

**Status**: Proposal
**Date**: 2026-03-11
**Author**: Agency Engine Team
**Affects**: `nietzsche-agency`, `nietzsche-lsystem`

---

## 1. Motivation

Self-Organized Criticality (SOC) is the engine's foundational dynamics model.
SOC requires avalanches at **all** scales, producing a power-law distribution:

```
P(S) ~ S^{-tau}    (tau ~ 1.5 for sandpile universality class)
```

Hard circuit breakers impose a fixed cutoff S_max where `P(S > S_max) = 0`.
This replaces the power-law tail with exponential truncation:

```
P(S) ~ S^{-tau} * exp(-S / S_max)
```

The result is that large-scale reorganization events -- the "earthquakes" that
maintain criticality -- are surgically removed. Over time the system drifts
from the critical point into a subcritical regime (frozen, unable to self-heal)
or supercritical regime (oscillating between kill and burst).

Soft breakers replace the wall with a hill: activity is damped exponentially
as intensity grows, but never fully killed. The tail becomes:

```
P(S) ~ S^{-tau} * (1 + (S/S_0)^alpha)^{-1}      (soft cutoff)
```

This preserves the power-law slope for S << S_0 while providing finite-resource
protection for S >> S_0. The critical exponent tau is preserved.

---

## 2. Inventory of Hard Circuit Breakers

Eight hard breakers were identified across the codebase.

### CB-1: Agency Energy Circuit Breaker

- **File**: `crates/nietzsche-agency/src/circuit_breaker.rs`
- **Used in**: `engine.rs` step 11 (Code-as-Data reflexive actions)
- **Mechanism**: Two binary checks:
  1. `activated_count > max_active_reflexes` (default 20) --> return `false`
  2. `total_energy > energy_sum_threshold` (default 50.0) --> return `false`
- **Effect**: All reflexive NQL actions are blocked for the entire tick.
  `P(reflex fires | count > 20) = 0`.
- **SOC impact**: HIGH. Reflexive chains are the primary avalanche mechanism in
  Code-as-Data. Hard-killing at 20 means no avalanche can exceed 20 steps.
- **Config**: `AGENCY_CIRCUIT_BREAKER_MAX` (20), `AGENCY_CIRCUIT_BREAKER_ENERGY` (50.0)

### CB-2: L-System Gaussian Spawn Breaker

- **File**: `crates/nietzsche-lsystem/src/engine.rs` (step 4b)
- **Mechanism**: Computes `mu + k * sigma` over all node energies (default k=2.0).
  Nodes with `energy > threshold` are blocked from SpawnChild / SpawnSibling.
  The action is `break` -- the node gets zero spawns.
- **Effect**: Sharp binary cutoff at the 2-sigma point of the energy distribution.
  Assumes Gaussian, but energy distributions in scale-free graphs are
  heavy-tailed (log-normal or power-law), making mu+2*sigma far too
  aggressive. Nodes in the natural high-energy tail are treated as pathological.
- **SOC impact**: CRITICAL. The L-System is the primary growth mechanism.
  Blocking growth for the most energetic nodes prevents large-scale branching
  cascades. Directly truncates the size distribution of growth avalanches.
- **Config**: `LSystemEngine::circuit_breaker_sigma` (default 2.0; 0.0 = disabled)

### CB-3: L-System Tumor Detection and Dampening

- **File**: `crates/nietzsche-lsystem/src/circuit_breaker.rs`
- **Mechanism**: Three hard mechanisms:
  1. **Depth-aware energy cap**: `cap(d) = base_cap - gradient * d`. Energy above
     cap is clamped. (`base_cap=1.0`, `gradient=0.3`)
  2. **Tumor detection**: BFS finds connected components where all nodes have
     `energy > overheated_threshold` (0.92) and `size >= min_tumor_size` (3).
  3. **Dampening**: All tumor nodes get `energy *= dampening_factor` (0.7).
  4. **Rate limiting**: `max_energy_delta` (0.25) per node per cycle.
- **Effect**: Multi-layered hard capping. The depth cap is a hard wall.
  The tumor dampening is multiplicative (softer) but the threshold detection
  is still binary: a cluster of 2 at energy 0.93 is ignored; a cluster of 3
  gets 30% drained. The transition is discontinuous.
- **SOC impact**: MEDIUM. The dampening itself is reasonably soft (multiplicative
  decay). The hard part is the binary threshold for tumor detection and the
  hard depth cap. Rate limiting at 0.25 is also a hard wall.

### CB-4: Forgetting Engine Hard Bounds (Nezhmetdinov)

- **File**: `crates/nietzsche-agency/src/forgetting/bounds.rs`
- **Used in**: `reactor.rs` (ForgettingCondemned handler)
- **Mechanism**: Two hard limits enforced in the reactor:
  1. `min_universe_size` (100): If node count would drop below 100, **all**
     deletions are blocked. `P(delete | count <= 100) = 0`.
  2. `max_deletion_rate` (0.10): Max 10% of nodes deleted per cycle.
     Cap is `min(rate_cap, universe_cap)`.
- **Effect**: A condemned node at the boundary gets zero probability of deletion.
  The deletion avalanche is truncated: if a cascading failure condemns 200
  nodes in a 500-node graph, only 50 are deleted (rate cap), and the rest
  survive. The surviving condemned nodes may accumulate and cause a burst
  in the next cycle.
- **SOC impact**: LOW-MEDIUM. Forgetting is a secondary process (cleanup, not
  growth). The hard bounds serve a genuine safety purpose (preventing total
  graph death). However, the sharp 10% cap creates bursty deletion patterns.

### CB-5: Shatter Protocol Degree Threshold

- **File**: `crates/nietzsche-agency/src/shatter.rs`
- **Mechanism**: `degree >= degree_threshold` (default 500) triggers shattering.
  Below 500: nothing happens. At 500: the node is split into up to
  `max_avatars` (8) shards.
- **Effect**: Step function at degree 500. A node at 499 is untouched; a node
  at 500 is violently restructured. No gradual response to increasing degree.
- **SOC impact**: LOW. Shatter is a structural repair mechanism, not a dynamics
  limiter. However, the sharp threshold can cause oscillations: a node is
  shattered, its avatars grow back to 499, then one more edge triggers
  another shatter. A gradual response would prevent this chatter.

### CB-6: World Model Anomaly Detection (Gaussian z-score)

- **File**: `crates/nietzsche-agency/src/world_model.rs`
- **Mechanism**: Flags metrics where `|z_score| > anomaly_sensitivity` (default
  3.0 standard deviations from the rolling mean). Currently informational
  only (produces `Anomaly` structs in the WorldSnapshot), but designed as
  a trigger for future defensive actions.
- **Effect**: Binary anomaly flag at the 3-sigma boundary. Like CB-2, this
  assumes Gaussian distributions. Operational metrics (query rate, mutation
  rate) are likely heavy-tailed in practice.
- **SOC impact**: LOW (currently informational). But if future code uses this
  as a trigger to throttle or block operations, it would become a hard
  breaker with the same Gaussian-assumption problems as CB-2.

### CB-7: Hebbian Max Weight Cap

- **File**: `crates/nietzsche-agency/src/hebbian.rs`
- **Mechanism**: Edge weight is capped at `max_weight` (5.0). Total outgoing
  weight per node is capped at `max_total_weight` (50.0). These are hard
  `min()/clamp()` operations.
- **Effect**: Edges at 5.0 receive zero additional potentiation regardless of
  co-activation intensity. Strong associations are artificially flattened.
- **SOC impact**: LOW-MEDIUM. Weight is not energy, so this does not directly
  truncate avalanches. But it prevents the formation of "superhighways"
  between critical concepts, which in SOC terms prevents the formation of
  long-range correlations.

### CB-8: Self-Healing Norm Threshold

- **File**: `crates/nietzsche-agency/src/self_healing.rs`
- **Mechanism**: Nodes with `||embedding|| > norm_threshold` (0.995) are
  re-projected to `target_norm` (0.98). This is a hard wall: any node
  that drifts beyond 0.995 is snapped back to 0.98. Ghost cleanup triggers
  when `phantom_ratio > ghost_ratio_threshold` (0.15).
- **Effect**: Prevents boundary exploration. In the Poincare ball, the
  boundary region (||x|| -> 1) encodes highly specific memories. Hard
  re-projection destroys any natural tendency to explore near the boundary.
- **SOC impact**: LOW. This is geometric hygiene, not dynamics limiting.
  However, it interacts with CB-3's depth cap: the double constraint
  (energy cap + norm cap) at the boundary is aggressive.

---

## 3. Soft Replacement Proposals

### 3.1 CB-1: Agency Energy Circuit Breaker --> Exponential Throttle

**Current**: `if count > 20 { block_all(); }`

**Proposed**: Probabilistic throttle with exponential damping.

```
P(permit action i) = exp(-max(0, count - S_knee) / S_scale)
```

- `S_knee` (default: 15): Number of reflexes below which P=1 (no throttling).
- `S_scale` (default: 10): Exponential decay scale. At `count = S_knee + S_scale`,
  P ~ 0.37. At `count = S_knee + 3*S_scale`, P ~ 0.05.

For energy sum:

```
rate_factor = exp(-max(0, total_energy - E_knee) / E_scale)
```

Each action is independently decided: `rand() < rate_factor`. This means a
burst of 50 reflexes still permits some (probabilistically), preserving the
tail of the avalanche distribution.

**Implementation**: Replace `check_safety() -> bool` with
`throttle_probability() -> f32`. Callers iterate the activated list and
apply per-action Bernoulli sampling.

**Config vars**:
- `AGENCY_SOFT_BREAKER_REFLEX_KNEE` (default: 15)
- `AGENCY_SOFT_BREAKER_REFLEX_SCALE` (default: 10)
- `AGENCY_SOFT_BREAKER_ENERGY_KNEE` (default: 40.0)
- `AGENCY_SOFT_BREAKER_ENERGY_SCALE` (default: 20.0)
- `AGENCY_SOFT_BREAKER_ENABLED` (default: false -- opt-in migration)

### 3.2 CB-2: L-System Gaussian Spawn Breaker --> Heavy-Tail Aware Throttle

**Current**: `if energy > mu + k*sigma { block_spawn(); }`

**Proposed**: Logistic suppression with configurable steepness.

```
P(spawn) = 1 / (1 + (energy / E_median)^alpha)
```

- `E_median`: Running median of node energies (robust to outliers, unlike mean).
- `alpha` (default: 4): Controls steepness. At alpha=4, nodes at 2x the median
  have P(spawn) ~ 6%. At alpha=2, P ~ 20%.

Alternative (budget-based):

```
spawn_budget(node) = max_spawns * exp(-(energy - E_median)^2 / (2 * bandwidth^2))
```

Where `max_spawns` is the maximum children a single node can produce per tick
and `bandwidth` controls the Gaussian kernel width. This is still Gaussian-shaped
but uses it as a smooth envelope rather than a binary cutoff.

The key improvement: instead of Gaussian statistics (mu + k*sigma) which
assume thin tails, use a rank-based measure (median, or even the 75th
percentile) which is robust to the heavy-tailed energy distributions that
SOC produces.

**Config vars**:
- `LSYSTEM_SOFT_BREAKER_ALPHA` (default: 4)
- `LSYSTEM_SOFT_BREAKER_BANDWIDTH` (default: 0.5)
- `LSYSTEM_SOFT_BREAKER_ENABLED` (default: false)

### 3.3 CB-3: Tumor Detection --> Graduated Response

**Current**: Binary detection (cluster >= 3 hot nodes) then multiplicative dampen.

**Proposed**: Two changes:

**A. Smooth depth cap** (replace hard clamp):

```
effective_energy = energy * sigmoid(-(||x|| - r_soft) / r_width)
```

Where `r_soft` (default: 0.85) is the radius at which suppression begins
and `r_width` (default: 0.05) controls the transition width. At r=0.85,
suppression is 50%. At r=0.95, suppression is ~88%. No hard wall.

**B. Continuous tumor response** (replace binary threshold):

Instead of binary `overheated/not-overheated`, compute a continuous
"overheating score" per node:

```
overheat(n) = max(0, energy(n) - E_soft) / (E_range)
```

Cluster score = mean(overheat) over connected component. Dampening is
proportional:

```
dampen(cluster) = 1 - dampening_strength * cluster_score^beta
```

Where `beta` (default: 2) makes the response superlinear in severity.
Small hot spots get gentle dampening; large severe tumors get aggressive
dampening. No discontinuity.

**C. Smooth rate limiting** (replace hard max_energy_delta):

```
effective_delta = delta * tanh(delta / soft_max_delta)
```

Where `soft_max_delta` (default: 0.25). For small deltas, tanh(x)~x so
the delta passes unchanged. For large deltas, it saturates at `soft_max_delta`.

**Config vars**:
- `LSYSTEM_TUMOR_SOFT_RADIUS` (default: 0.85)
- `LSYSTEM_TUMOR_RADIUS_WIDTH` (default: 0.05)
- `LSYSTEM_TUMOR_SOFT_ENERGY` (default: 0.85)
- `LSYSTEM_TUMOR_ENERGY_RANGE` (default: 0.15)
- `LSYSTEM_TUMOR_DAMPEN_BETA` (default: 2)
- `LSYSTEM_TUMOR_SOFT_DELTA` (default: 0.25)
- `LSYSTEM_TUMOR_SOFT_ENABLED` (default: false)

### 3.4 CB-4: Forgetting Hard Bounds --> Asymptotic Slowdown

**Current**: `if count - deletions < min_universe_size { block_all(); }`

**Proposed**: Deletion probability decreases asymptotically as the graph
approaches the minimum size:

```
P(delete) = base_p * (1 - (min_size / current_size)^gamma)
```

Where:
- `gamma` (default: 3): Controls how sharply deletion slows near the limit.
  At gamma=3, when `current_size = 2 * min_size`, P ~ 87.5% of base.
  When `current_size = 1.1 * min_size`, P ~ 17.4% of base.
- The deletion rate is also softened:

```
effective_rate = max_rate * exp(-lambda / (current_size - min_size + epsilon))
```

This means deletion never fully stops (preserving the SOC tail), but becomes
exponentially unlikely as the graph shrinks toward the safety boundary.

**Safety guarantee**: The `min_universe_size` hard floor is preserved as an
absolute last resort (defense in depth). The soft curve handles 99% of cases;
the hard floor catches numeric edge cases.

**Config vars**:
- `NEZHMETDINOV_SOFT_GAMMA` (default: 3)
- `NEZHMETDINOV_SOFT_RATE_LAMBDA` (default: 10.0)
- `NEZHMETDINOV_SOFT_ENABLED` (default: false)
- `NEZHMETDINOV_HARD_FLOOR_ENABLED` (default: true -- keep as safety net)

### 3.5 CB-5: Shatter Degree Threshold --> Graduated Shatter

**Current**: Binary at degree 500.

**Proposed**: Probability of shattering increases sigmoidally with degree:

```
P(shatter) = sigmoid((degree - D_center) / D_width)
```

Where:
- `D_center` (default: 400): Degree at which P = 50%.
- `D_width` (default: 50): Controls transition steepness.

Additionally, the number of avatars scales with degree:

```
n_avatars = min(max_avatars, ceil((degree - D_min) / D_per_avatar))
```

Where `D_per_avatar` (default: 100) means a node at degree 600 gets 6
avatars, not the full 8. A node at degree 450 gets 2-3 avatars (light
restructuring) rather than the full shatter.

**Config vars**:
- `AGENCY_SHATTER_SOFT_CENTER` (default: 400)
- `AGENCY_SHATTER_SOFT_WIDTH` (default: 50)
- `AGENCY_SHATTER_D_PER_AVATAR` (default: 100)
- `AGENCY_SHATTER_SOFT_ENABLED` (default: false)

### 3.6 CB-6: World Model Anomaly --> Robust Anomaly Scoring

**Current**: Binary flag at z > 3.0 (Gaussian assumption).

**Proposed**: Replace z-score with a robust anomaly score based on
Median Absolute Deviation (MAD):

```
modified_z = 0.6745 * (x - median) / MAD
```

The constant 0.6745 normalizes MAD to match sigma for Gaussian data, but
MAD is robust to heavy tails. Anomaly is a continuous score, not a binary flag.

Additionally, provide a continuous "anomaly intensity" to downstream consumers:

```
intensity = sigmoid((modified_z - z_soft) / z_width)
```

Where `z_soft` (default: 3.0) and `z_width` (default: 1.0). This gives
downstream systems a smooth signal instead of a binary trigger.

**Config vars**:
- `AGENCY_WORLD_MODEL_ROBUST_ANOMALY` (default: true)
- `AGENCY_WORLD_MODEL_ANOMALY_Z_SOFT` (default: 3.0)
- `AGENCY_WORLD_MODEL_ANOMALY_Z_WIDTH` (default: 1.0)

### 3.7 CB-7: Hebbian Max Weight --> Logarithmic Saturation

**Current**: `edge.weight = min(edge.weight + delta, max_weight)`

**Proposed**: Logarithmic saturation curve:

```
effective_weight = W_max * (1 - exp(-raw_weight / W_tau))
```

Where:
- `W_max` (default: 5.0): Asymptotic maximum weight.
- `W_tau` (default: 2.5): Time constant. At raw_weight = W_tau, the
  effective weight is ~63% of W_max.

The key property: the derivative `dW_eff/dW_raw = exp(-W_raw / W_tau)`
is always positive, so additional co-activation always strengthens the
edge, but with diminishing returns. There is no hard wall where
additional evidence is ignored.

For homeostatic scaling, replace the hard cap on total outgoing weight
with a soft normalization:

```
scale = min(1.0, W_total_max / (sum_of_outgoing_weights + epsilon))
```

Applied only when the sum exceeds `W_total_max`, this smoothly redistributes
without a sharp cliff.

**Config vars**:
- `AGENCY_HEBBIAN_W_TAU` (default: 2.5)
- `AGENCY_HEBBIAN_SOFT_WEIGHT_ENABLED` (default: false)

### 3.8 CB-8: Self-Healing Norm Threshold --> Elastic Boundary

**Current**: Hard re-projection at ||x|| > 0.995.

**Proposed**: Elastic restoring force that increases with distance beyond a
soft boundary:

```
restoring_force = k * max(0, ||x|| - r_elastic)^2
new_norm = ||x|| - restoring_force * dt
```

Where:
- `r_elastic` (default: 0.98): Radius at which the restoring force begins.
- `k` (default: 10.0): Stiffness constant.
- `dt`: Tick duration factor (normalized to 1 for per-tick updates).

For ||x|| = 0.99 (just past elastic boundary), force = 10 * 0.01^2 = 0.001.
For ||x|| = 0.999, force = 10 * 0.019^2 = 0.0036. The node is gently
pushed inward, not teleported. Nodes near the boundary can still explore
the high-specificity region but are gradually reeled back.

**Safety**: Keep the hard wall at ||x|| > 0.9999 as absolute protection
against numeric instability (Poincare ball diverges at ||x||=1).

**Config vars**:
- `AGENCY_HEALING_ELASTIC_RADIUS` (default: 0.98)
- `AGENCY_HEALING_ELASTIC_K` (default: 10.0)
- `AGENCY_HEALING_HARD_WALL_RADIUS` (default: 0.9999)
- `AGENCY_HEALING_SOFT_ENABLED` (default: false)

---

## 4. Mathematical Analysis: Preserving Power-Law Tails

### 4.1 Hard Cutoff: Exponential Truncation

A hard breaker at S_max modifies the avalanche distribution:

```
P_hard(S) = { C * S^{-tau}     if S <= S_max
            { 0                 if S > S_max
```

The moments diverge differently from the true power law. In particular,
the variance is artificially bounded:

```
Var[S]_hard < S_max^{2-tau} / (2-tau)    (for tau < 2)
```

This means fluctuations are suppressed, which in SOC theory corresponds to
subcriticality. The system cannot self-tune back to the critical point
because the feedback mechanism (large avalanches dissipating excess drive)
is removed.

### 4.2 Exponential Damping: Preserved Slope

With an exponential soft breaker:

```
P_soft(S) = C * S^{-tau} * exp(-S / S_scale)
```

For S << S_scale, the exponential factor is approximately 1, so the power-law
regime is preserved. The crossover occurs at S ~ S_scale. The key properties:

1. **All moments exist**: The exponential ensures convergence of all moments,
   preventing numeric overflow.
2. **Slope preservation**: A log-log plot shows the same slope -tau for
   S < S_scale, with a gradual rolloff above. This is measurable and
   verifiable.
3. **Critical exponent preservation**: The exponent tau is unchanged; only the
   cutoff scale S_scale is introduced as a new parameter. In the limit
   S_scale -> infinity, we recover the pure power law.

### 4.3 Probabilistic Throttle: Hill Function

The logistic/Hill function throttle:

```
P(continue) = 1 / (1 + (S/S_0)^alpha)
```

modifies the avalanche distribution to:

```
P_throttled(S) ~ S^{-tau} * (1 + (S/S_0)^alpha)^{-1}
```

This is a "tempered" power law. For S << S_0, the distribution is pure power
law. For S >> S_0, P decays as S^{-(tau+alpha)}. The effective exponent
smoothly transitions from tau to tau+alpha.

The alpha parameter controls the "softness" of the transition:
- alpha = 1: Very gradual (Lorentzian-like)
- alpha = 2: Moderate
- alpha = 4: Relatively sharp but still smooth (no discontinuity)
- alpha -> infinity: Recovers the hard cutoff

For SOC preservation, alpha in [2, 4] is recommended.

### 4.4 Budget-Based: Token Bucket

The budget approach allocates a compute budget B per tick. Each action
consumes c tokens. When the budget is low, a proportional throttle engages:

```
P(permit) = min(1, B_remaining / (c * N_remaining))
```

This is effectively an adaptive S_scale: when the system is quiet (few
actions), the budget is generous and no throttling occurs. When the system
is active, the budget creates a natural soft ceiling that scales with the
total activity level.

The resulting avalanche distribution is:

```
P_budget(S) ~ S^{-tau} * f(S/B)
```

Where f is a smooth function with f(0)=1 and f -> 0 as S/B -> infinity.
The key property is that B scales with the system (larger graphs get more
budget), so S_scale grows with system size, preserving the power law over
the relevant range.

---

## 5. Implementation Priority

Ordered by SOC impact (most harmful hard breaker first):

| Priority | Breaker | SOC Impact | Effort | Risk |
|----------|---------|------------|--------|------|
| **P0** | CB-2: L-System Gaussian Spawn | CRITICAL | Medium | Low -- growth only |
| **P1** | CB-1: Agency Reflex Breaker | HIGH | Low | Low -- probabilistic sampling |
| **P2** | CB-3: Tumor Detection | MEDIUM | Medium | Medium -- multi-part |
| **P3** | CB-4: Forgetting Hard Bounds | LOW-MED | Low | Low -- keep hard floor |
| **P4** | CB-5: Shatter Threshold | LOW | Low | Low |
| **P5** | CB-7: Hebbian Max Weight | LOW-MED | Low | Low |
| **P6** | CB-8: Self-Healing Norm | LOW | Low | Very low |
| **P7** | CB-6: World Model Anomaly | LOW | Low | Very low (informational) |

### Rationale

**CB-2 is P0** because the L-System is the primary growth engine, running on
every tick. The Gaussian assumption is provably wrong for SOC systems (which
produce power-law energy distributions). Every tick, the high-energy tail of
the distribution is silenced. This is the single most impactful change.

**CB-1 is P1** because reflexive chains are the only mechanism for "thinking"
(Code-as-Data). A cap at 20 means no thought can involve more than 20 steps.
The fix is trivial (Bernoulli sampling instead of binary gate).

**CB-3 is P2** because the tumor mechanism is actually partially soft already
(multiplicative dampening). The hard parts are the binary threshold and the
depth cap, which are moderate effort to soften.

**CB-4 through CB-8** are lower priority because they affect secondary
processes (forgetting, shattering, weight capping, geometry repair) rather
than the core growth and activation dynamics.

---

## 6. Toggleability: Independent Soft/Hard Switching

Every soft breaker MUST be independently toggleable. The system must be able
to run with any combination of soft and hard breakers.

### Toggle Pattern

```rust
enum BreakerMode {
    Hard,       // Legacy behavior (default)
    Soft,       // New soft behavior
    Disabled,   // No limiting at all (testing/debugging only)
}
```

Each breaker reads its mode from an environment variable:

| Breaker | Env Var | Values |
|---------|---------|--------|
| CB-1 | `AGENCY_BREAKER_REFLEX_MODE` | `hard` (default), `soft`, `disabled` |
| CB-2 | `LSYSTEM_BREAKER_SPAWN_MODE` | `hard` (default), `soft`, `disabled` |
| CB-3 | `LSYSTEM_BREAKER_TUMOR_MODE` | `hard` (default), `soft`, `disabled` |
| CB-4 | `NEZHMETDINOV_BREAKER_DELETION_MODE` | `hard` (default), `soft`, `disabled` |
| CB-5 | `AGENCY_BREAKER_SHATTER_MODE` | `hard` (default), `soft`, `disabled` |
| CB-6 | `AGENCY_BREAKER_ANOMALY_MODE` | `hard` (default), `soft`, `disabled` |
| CB-7 | `AGENCY_BREAKER_HEBBIAN_MODE` | `hard` (default), `soft`, `disabled` |
| CB-8 | `AGENCY_BREAKER_HEALING_MODE` | `hard` (default), `soft`, `disabled` |

The `disabled` mode is for controlled experiments only (measure the true
power-law exponent without any breaker interference). It should log a
warning on startup.

---

## 7. Monitoring: Verifying SOC Improvement

### 7.1 Avalanche Size Distribution

The primary metric. Measure the distribution of avalanche sizes S (defined as
the number of intents produced in a single reactive chain).

**Method**: For each tick, count the total intents produced. Maintain a
histogram of intent counts across ticks. Fit a power law using Maximum
Likelihood Estimation (Clauset-Shalizi-Newman method).

**Metric**: Estimated exponent `tau_hat` and goodness-of-fit p-value.
- tau_hat in [1.2, 1.8] with p > 0.1: SOC confirmed.
- tau_hat exists but p < 0.1: Marginal criticality.
- No power-law fit: Not SOC (subcritical or supercritical).

### 7.2 Branching Ratio

SOC systems at criticality have branching ratio sigma ~ 1.

```
sigma = <S_{t+1}> / <S_t>
```

Where S_t is the number of active nodes at time step t during an avalanche.

**Method**: Track the number of newly activated nodes per sub-step within
a single tick. Compute the ratio of consecutive sub-steps averaged over
many avalanches.

**Metric**:
- sigma ~ 1.0 (+/- 0.05): Critical.
- sigma < 0.9: Subcritical (hard breakers too aggressive).
- sigma > 1.1: Supercritical (breakers too permissive, runaway risk).

### 7.3 Truncation Detection

Directly measure the truncation scale S_max by fitting:

```
P(S) = C * S^{-tau} * exp(-S / S_trunc)
```

and comparing S_trunc between hard and soft breaker modes.

**Success criterion**: S_trunc(soft) > 10 * S_trunc(hard), indicating
that the soft breaker has substantially extended the power-law regime.

### 7.4 1/f Noise in Energy Time Series

SOC systems produce 1/f (pink) noise in their time series.

**Method**: Compute the power spectral density (PSD) of the total graph
energy over time. Fit the spectral exponent beta in PSD ~ f^{-beta}.

**Metric**:
- beta in [0.8, 1.2]: SOC (1/f noise).
- beta ~ 0: White noise (subcritical, no correlations).
- beta ~ 2: Brownian noise (supercritical, random walk).

### 7.5 Dashboard Integration

Add the following to the existing `/api/agency/dashboard` endpoint:

```json
{
  "soc_metrics": {
    "tau_hat": 1.47,
    "tau_pvalue": 0.32,
    "branching_ratio": 0.98,
    "s_truncation": 450,
    "spectral_beta": 1.05,
    "breaker_modes": {
      "reflex": "soft",
      "spawn": "soft",
      "tumor": "hard",
      "deletion": "hard",
      "shatter": "hard",
      "anomaly": "soft",
      "hebbian": "hard",
      "healing": "hard"
    },
    "breaker_activations": {
      "reflex_throttled": 12,
      "spawn_dampened": 45,
      "tumor_dampened": 3,
      "deletions_capped": 0,
      "shatter_triggered": 1,
      "anomalies_flagged": 2,
      "hebbian_saturated": 89,
      "healing_reprojected": 7
    }
  }
}
```

### 7.6 A/B Testing Protocol

To validate the change:

1. Run two identical collections (e.g., duplicate `tech_galaxies` into
   `tech_galaxies_soft`).
2. Collection A: all hard breakers (status quo).
3. Collection B: all soft breakers enabled.
4. Run the agency engine for 1000+ ticks on both.
5. Compare:
   - Avalanche size distributions (log-log plots).
   - Branching ratio over time.
   - Graph health metrics (Hausdorff, mean energy, coherence).
   - System stability (no runaway growth, no graph death).

If collection B shows a cleaner power law without degraded health metrics,
the soft breakers are validated. If collection B shows runaway behavior,
tune S_scale / alpha / bandwidth parameters.

---

## 8. Migration Strategy

### Phase 1: Opt-in (Default: Hard)
- Implement all soft breakers behind `_MODE=soft` env vars.
- Default remains `hard` -- zero production impact.
- Run A/B experiments internally.

### Phase 2: Gradual Rollout
- Flip P0 (CB-2) to `soft` by default after 1000-tick validation.
- Monitor SOC metrics for 1 week.
- Flip P1 (CB-1) to `soft`.
- Continue down the priority list with 1-week validation gaps.

### Phase 3: Default Soft
- All breakers default to `soft`.
- Hard mode remains available for emergencies (`_MODE=hard`).
- `disabled` mode available for research.

---

## 9. Summary

| Breaker | Hard Mechanism | Soft Replacement | Key Parameter |
|---------|---------------|------------------|---------------|
| CB-1 | count > 20 -> block all | P = exp(-(count-knee)/scale) | S_scale=10 |
| CB-2 | energy > mu+2sigma -> block spawn | P = 1/(1+(E/E_med)^alpha) | alpha=4 |
| CB-3 | binary tumor detect + hard cap | continuous overheat score + sigmoid cap | beta=2 |
| CB-4 | count < min -> block all | P ~ (1-(min/count)^gamma) | gamma=3 |
| CB-5 | degree >= 500 -> shatter | P = sigmoid((deg-center)/width) | D_center=400 |
| CB-6 | z > 3sigma -> flag | MAD-based robust z + continuous intensity | z_width=1.0 |
| CB-7 | weight > max -> clamp | W_eff = W_max*(1-exp(-W/W_tau)) | W_tau=2.5 |
| CB-8 | norm > 0.995 -> reproject | elastic force k*(norm-r_elastic)^2 | k=10.0 |

The fundamental principle: **hills, not walls**. Every breaker must have
a smooth, differentiable transition. The system can push against the hill
(large avalanches become rarer but not impossible), preserving the power-law
tail that SOC requires.
