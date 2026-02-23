# AGENT 8: Neural Thresholds Blueprint (Code-as-Data Reflexes)

> **Domain**: Learned adaptive thresholds replacing hard-coded values
> **Status**: Deep analysis complete
> **Date**: 2026-02-23
> **Risk Level**: HIGHEST — this is the most safety-sensitive neural component

---

## 1. Current Hard-Coded Thresholds — Complete Inventory

### 1.1 Agency Engine Thresholds (`nietzsche-agency`)

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| A1 | `0.25` | Entropy variance threshold — triggers EntropySpike when local Hausdorff variance exceeds this | `config.rs:80` | heuristic | **LEARNABLE** |
| A2 | `0.70` | Coherence overlap threshold — Jaccard(t=0.1, t=10.0) above this means hierarchy lost discriminative power | `config.rs:82` | heuristic | **LEARNABLE** |
| A3 | `0.1` | Gap minimum density — fraction of expected uniform density below which a sector is declared a "gap" | `config.rs:85` | heuristic | **LEARNABLE** |
| A4 | `0.3` | Observer wake energy threshold — mean energy below this triggers WakeUp event | `config.rs:87` | performance-tuning | **LEARNABLE** |
| A5 | `0.5` | Observer Hausdorff low bound — global Hausdorff below this triggers WakeUp | `config.rs:88` | performance-tuning | **LEARNABLE** (with bounds) |
| A6 | `1.9` | Observer Hausdorff high bound — global Hausdorff above this triggers WakeUp | `config.rs:89` | performance-tuning | **LEARNABLE** (with bounds) |
| A7 | `0.6` | Desire dream threshold — minimum desire priority to trigger an automatic dream | `config.rs:92` | heuristic | **LEARNABLE** |
| A8 | `0.01` | Niilista distance threshold — Poincare distance below which nodes are redundant | `config.rs:94` | performance-tuning | **LEARNABLE** |
| A9 | `20` | Circuit breaker max actions — max reflexive actions per tick | `config.rs:98` | safety-critical | **SACRED** |
| A10 | `50.0` | Circuit breaker energy sum threshold — max global energy sum before blocking reflexes | `config.rs:99` | safety-critical | **SACRED** |
| A11 | `0.05` | LTD rate — weight reduction per caregiver correction (5%) | `ltd_daemon.rs:85` | safety-critical | **SACRED** |
| A12 | `1` | LTD minimum correction threshold — corrections needed before LTD triggers | `ltd_daemon.rs:37` | safety-critical | **SACRED** |
| A13 | `0.5` | Min energy for coherence probe — nodes below this are excluded from coherence analysis | `coherence.rs:27` | heuristic | **LEARNABLE** |
| A14 | `10` | Max coherence probes — maximum nodes sampled for coherence daemon | `coherence.rs:25` | performance-tuning | **LEARNABLE** |
| A15 | `0.8` | Default action activation threshold — energy above which Code-as-Data actions fire | `code_as_data.rs:88` | heuristic | **LEARNABLE** |
| A16 | `3` | Reactor cooldown ticks — minimum ticks between repeated sleep/lsystem intents | `config.rs:90` | performance-tuning | **LEARNABLE** |
| A17 | `5` | Evolution cooldown ticks — minimum ticks between evolution suggestions | `config.rs:97` | performance-tuning | **LEARNABLE** |

### 1.2 Dialectic Engine Thresholds (`nietzsche-agency/dialectic.rs`)

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| D1 | `0.8` | Proximity threshold — max Poincare distance for contradiction candidacy | `dialectic.rs:41` | heuristic | **LEARNABLE** |
| D2 | `0.3` | Min certainty — minimum certainty for dialectic participation | `dialectic.rs:45` | heuristic | **LEARNABLE** |
| D3 | `1.2` | Polarity gap threshold — min polarity difference to flag contradiction | `dialectic.rs:54` | heuristic | **LEARNABLE** |
| D4 | `0.5` | Default certainty — assigned to new nodes | `dialectic.rs:57` | heuristic | **LEARNABLE** |
| D5 | `0.7` | Certainty reduction factor — thesis/antithesis certainty multiplied by this after synthesis | `dialectic.rs:398` | heuristic | **LEARNABLE** |
| D6 | `0.8` | Synthesis pull factor — coordinates multiplied by 0.8 to pull toward center | `dialectic.rs:331` | experimental | **LEARNABLE** |

### 1.3 Evolution Strategy Thresholds (`nietzsche-agency/evolution.rs`)

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| E1 | `80.0` | Gap ratio denominator — `gap_count / 80.0` determines gap severity | `evolution.rs:87` | heuristic | **LEARNABLE** |
| E2 | `0.3`/`0.8` | Energy "ok" range — strategy depends on energy being in [0.3, 0.8] | `evolution.rs:88` | heuristic | **LEARNABLE** |
| E3 | `2` | Entropy spike count threshold — more than 2 spikes suggests pruning | `evolution.rs:90` | heuristic | **LEARNABLE** |
| E4 | `0.3` | Growth gap ratio — gaps > 30% of total triggers FavorGrowth | `evolution.rs:95` | heuristic | **LEARNABLE** |
| E5 | `0.3`-`0.7` | Evolved rule energy thresholds — per-strategy thresholds for growth/prune/lateral rules | `evolution.rs:112-181` | heuristic | **LEARNABLE** |
| E6 | `0.25/0.25/0.3/0.1/0.1` | Fitness function weights — energy, coherence, fractal, gap penalty, entropy penalty | `evolution.rs:193-195` | heuristic | **LEARNABLE** |

### 1.4 Reactor Zaratustra Modulation (`nietzsche-agency/reactor.rs`)

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| R1 | `0.10` | Base alpha — Zaratustra propagation coefficient | `reactor.rs:336` | performance-tuning | **LEARNABLE** |
| R2 | `0.02` | Base decay — Zaratustra temporal decay coefficient | `reactor.rs:337` | performance-tuning | **LEARNABLE** |
| R3 | `0.2`/`0.4`/`0.85` | Energy level breakpoints — determine alpha/decay modulation strategy | `reactor.rs:339-360` | heuristic | **LEARNABLE** |
| R4 | `0.25`/`0.005`/`0.08`/`0.06` | Alpha/decay min/max bounds — hard limits on modulated values | `reactor.rs:342-364` | safety-critical | **SACRED** (bounds only) |
| R5 | `3` | Entropy spike count threshold for decay boost | `reactor.rs:360` | heuristic | **LEARNABLE** |

### 1.5 Observer Identity (`nietzsche-agency/identity.rs`)

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| I1 | `0.01` | Observer node depth — placed near center of Poincare ball | `identity.rs:69` | experimental | **SACRED** (architectural constant) |
| I2 | `40.0` | Gap penalty denominator — `gap_count / 40.0` for observer health score | `identity.rs:139` | heuristic | **LEARNABLE** |
| I3 | `0.5` | Non-fractal score — health score when graph is not fractal | `identity.rs:138` | heuristic | **LEARNABLE** |

### 1.6 Zaratustra Engine (`nietzsche-zaratustra`)

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| Z1 | `0.10` | Alpha — energy propagation coefficient | `config.rs:65` | performance-tuning | **LEARNABLE** |
| Z2 | `0.02` | Decay — temporal decay coefficient | `config.rs:66` | performance-tuning | **LEARNABLE** |
| Z3 | `1.0` | Energy cap — absolute maximum node energy | `config.rs:67` | safety-critical | **SACRED** |
| Z4 | `0.70` | Echo threshold — min energy for Eternal Recurrence snapshot | `config.rs:68` | heuristic | **LEARNABLE** |
| Z5 | `0.10` | Ubermensch top fraction — top 10% considered elite tier | `config.rs:70` | heuristic | **LEARNABLE** |

### 1.7 L-System Engine (`nietzsche-lsystem`)

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| L1 | `0.5` | DEFAULT_HAUSDORFF_LO — auto-prune below this fractal dimension | `engine.rs:40` | performance-tuning | **LEARNABLE** (with bounds) |
| L2 | `1.9` | DEFAULT_HAUSDORFF_HI — auto-prune above this fractal dimension | `engine.rs:42` | performance-tuning | **LEARNABLE** (with bounds) |
| L3 | `2.0` | DEFAULT_CIRCUIT_BREAKER_SIGMA — block spawning above mean + k*sigma | `engine.rs:51` | safety-critical | **SACRED** |
| L4 | `12` | LOCAL_K — k-nearest neighbors for local Hausdorff computation | `hausdorff.rs:19` | performance-tuning | **LEARNABLE** |

### 1.8 Circuit Breaker (Anti-Tumor) (`nietzsche-lsystem/circuit_breaker.rs`)

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| CB1 | `0.92` | Overheated threshold — node energy above this is overheated | `circuit_breaker.rs:76` | safety-critical | **SACRED** |
| CB2 | `3` | Min tumor size — minimum cluster size to qualify as tumor | `circuit_breaker.rs:77` | safety-critical | **SACRED** |
| CB3 | `0.7` | Dampening factor — tumor energy multiplied by this (30% drain) | `circuit_breaker.rs:78` | safety-critical | **SACRED** |
| CB4 | `0.3` | Depth cap gradient — deeper nodes get lower energy cap | `circuit_breaker.rs:79` | heuristic | **LEARNABLE** (with bounds) |
| CB5 | `1.0` | Base energy cap (at depth=0) | `circuit_breaker.rs:80` | safety-critical | **SACRED** |
| CB6 | `0.25` | Max energy delta per cycle — rate limiting for energy changes | `circuit_breaker.rs:81` | safety-critical | **SACRED** |

### 1.9 Sleep Cycle (`nietzsche-sleep`)

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| S1 | `0.02` | Sleep noise — perturbation standard deviation | `cycle.rs:77` | performance-tuning | **LEARNABLE** |
| S2 | `5e-3` | Adam learning rate — Riemannian Adam optimizer step size | `cycle.rs:79` | performance-tuning | **LEARNABLE** |
| S3 | `0.15` | Hausdorff threshold — max allowed Hausdorff delta for commit | `cycle.rs:80` | safety-critical | **SACRED** |
| S4 | `0.5` | Fusion distance threshold — max Poincare distance for memory fusion | `fusion.rs:39` | heuristic | **LEARNABLE** |
| S5 | `0.3` | Fusion min energy — min energy for fusion eligibility | `fusion.rs:42` | heuristic | **LEARNABLE** |

### 1.10 Dream Engine (`nietzsche-dream`)

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| DR1 | `0.9` | Energy spike threshold — dream energy above this fires spike event | `engine.rs:32` | heuristic | **LEARNABLE** |
| DR2 | `0.5` | Curvature anomaly threshold — Hausdorff change threshold | `engine.rs:33` | heuristic | **LEARNABLE** |
| DR3 | `0.05` | Default noise — dream perturbation amplitude | `engine.rs:31` | performance-tuning | **LEARNABLE** |

### 1.11 Diffusion Engine (`nietzsche-pregel`)

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| DF1 | `10` | K_DEFAULT — Chebyshev polynomial terms | `chebyshev.rs:40` | performance-tuning | **LEARNABLE** |
| DF2 | `2.0` | LAMBDA_MAX — upper bound on largest Laplacian eigenvalue | `chebyshev.rs:44` | performance-tuning | SACRED (mathematical bound) |
| DF3 | `1e-6` | Min score — minimum activation score for inclusion | `diffusion.rs:60` | performance-tuning | **LEARNABLE** |

### 1.12 Infrastructure Thresholds

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| IN1 | `1000` | GPU_THRESHOLD — min vectors for GPU acceleration | `hnsw-gpu/lib.rs:60` | performance-tuning | **LEARNABLE** |
| IN2 | `1000` | TPU_THRESHOLD — min vectors for TPU acceleration | `tpu/lib.rs:76` | performance-tuning | **LEARNABLE** |
| IN3 | `0.10` | REBUILD_DELTA_RATIO — dirty ratio triggering index rebuild | `hnsw-gpu/lib.rs:64`, `tpu/lib.rs:80` | performance-tuning | **LEARNABLE** |
| IN4 | `0.20` | RECOMPILE_RATIO — dirty ratio triggering TPU recompile | `tpu/lib.rs:84` | performance-tuning | **LEARNABLE** |
| IN5 | `0.999` | MAX_NORM — Poincare ball boundary clamp | `hyp-ops/lib.rs:269` | safety-critical | **SACRED** (mathematical invariant) |
| IN6 | `500` | PLAIN_SCAN_THRESHOLD — hybrid search switches at this node count | `graph/db.rs:1233` | performance-tuning | **LEARNABLE** |
| IN7 | `60.0` | RRF_K — Reciprocal Rank Fusion constant | `graph/db.rs:1289` | heuristic | **LEARNABLE** |

### 1.13 API Safety Limits

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| API1 | `4096` | MAX_EMBEDDING_DIM | `api/validation.rs:27` | safety-critical | **SACRED** |
| API2 | `8192` | MAX_NQL_LEN | `api/validation.rs:30` | safety-critical | **SACRED** |
| API3 | `10000` | MAX_KNN_K | `api/validation.rs:33` | safety-critical | **SACRED** |
| API4 | `10000` | MAX_BATCH_SIZE | `api/server.rs:1367` | safety-critical | **SACRED** |
| API5 | `50000` | DEFAULT_GAS_LIMIT — NQL query gas budget | `query/executor.rs:217` | safety-critical | **SACRED** |

### 1.14 EVA Clinical Thresholds (Go)

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| EV1 | `0.75` | Speaker match threshold — voice recognition confidence | `cortex/voice/speaker/store.go:22` | safety-critical | **SACRED** |
| EV2 | `0.7` | Confidence gate threshold — executive attention confidence | `cortex/attention/executive.go:27` | safety-critical | **SACRED** |
| EV3 | `0.88` | Semantic dedup threshold — memory deduplication similarity | `cortex/self/semantic_deduplicator.go:35` | heuristic | **LEARNABLE** |
| EV4 | `0.85` | Memory consolidation clustering threshold | `cortex/self/semantic_deduplicator.go:292` | heuristic | **LEARNABLE** |
| EV5 | `0.6` | Veracity low confidence — below this, lie detector flags uncertainty | `cortex/veracity/response_strategy.go:33` | safety-critical | **SACRED** |
| EV6 | `0.5/0.3/0.2` | Confidence assessment weights (clarity, context, complexity) | `cortex/attention/confidence_gate.go:49-51` | heuristic | **LEARNABLE** |
| EV7 | `3` | Family tracker sessions threshold — min sessions for family analysis | `clinical/graph/family_tracker.go:105` | safety-critical | **SACRED** |

### 1.15 Desire Engine Priority Computation (`nietzsche-agency/desire.rs`)

| # | Value | Purpose | File:Line | Category | Classification |
|---|-------|---------|-----------|----------|---------------|
| DE1 | `0.4` / `0.6` | Priority weights — depth_weight * 0.4 + density_weight * 0.6 | `desire.rs:109` | heuristic | **LEARNABLE** |
| DE2 | `0.3` / `0.6` | Depth descriptor breakpoints — abstract/mid-level/specific classification | `desire.rs:155-160` | heuristic | **LEARNABLE** |
| DE3 | `50` | Max stored desires — prune old desires above this | `desire.rs:140` | performance-tuning | **LEARNABLE** |

---

## 2. ONNX Tensor Specification

### 2.1 Context Input Tensor

```
Shape: [batch, context_dim] where context_dim = 16
```

| Index | Feature | Range | Source |
|-------|---------|-------|--------|
| 0 | `mean_energy` | [0.0, 1.0] | HealthReport |
| 1 | `energy_std` | [0.0, 0.5] (normalized) | HealthReport |
| 2 | `global_hausdorff` | [0.0, 3.0] (normalized to [0,1]) | HealthReport |
| 3 | `coherence_score` | [0.0, 1.0] | HealthReport |
| 4 | `gap_fraction` | [0.0, 1.0] | gap_count / total_sectors |
| 5 | `entropy_spike_rate` | [0.0, 1.0] | spike_count / region_count |
| 6 | `total_nodes` (log-scaled) | [0.0, 1.0] | log10(nodes) / 7.0 |
| 7 | `total_edges` (log-scaled) | [0.0, 1.0] | log10(edges) / 7.0 |
| 8 | `is_fractal` | {0.0, 1.0} | boolean |
| 9 | `p10_energy` | [0.0, 1.0] | EnergyPercentiles |
| 10 | `p90_energy` | [0.0, 1.0] | EnergyPercentiles |
| 11 | `tick_number` (cyclic) | [0.0, 1.0] | sin(tick * 2pi / period) |
| 12 | `recent_sleep_cycles` | [0.0, 1.0] | count / max_expected |
| 13 | `recent_lsystem_ticks` | [0.0, 1.0] | count / max_expected |
| 14 | `action_nodes_active` | [0.0, 1.0] | active / total_actions |
| 15 | `desire_backlog` | [0.0, 1.0] | pending_desires / 50 |

### 2.2 Threshold Output Tensor

```
Shape: [batch, num_thresholds] where num_thresholds = 24
```

Each output neuron uses **sigmoid activation** to produce a value in [0, 1], which is then **affine-scaled** to the threshold's valid range:

```
threshold_value = min_safe + sigmoid(raw_output) * (max_safe - min_safe)
```

| Output Index | Maps to | Min Safe | Max Safe | Default |
|-------------|---------|----------|----------|---------|
| 0 | entropy_variance_threshold (A1) | 0.05 | 1.0 | 0.25 |
| 1 | coherence_overlap_threshold (A2) | 0.3 | 0.95 | 0.70 |
| 2 | gap_min_density (A3) | 0.01 | 0.5 | 0.10 |
| 3 | observer_wake_energy_threshold (A4) | 0.05 | 0.7 | 0.30 |
| 4 | observer_wake_hausdorff_lo (A5) | 0.1 | 1.0 | 0.50 |
| 5 | observer_wake_hausdorff_hi (A6) | 1.2 | 2.5 | 1.90 |
| 6 | desire_dream_threshold (A7) | 0.2 | 0.95 | 0.60 |
| 7 | niilista_distance_threshold (A8) | 0.001 | 0.1 | 0.01 |
| 8 | min_energy_for_probe (A13) | 0.1 | 0.9 | 0.50 |
| 9 | reactor_cooldown_ticks (A16) | 1 | 10 | 3 |
| 10 | evolution_cooldown_ticks (A17) | 1 | 20 | 5 |
| 11 | dialectic_proximity_threshold (D1) | 0.2 | 2.0 | 0.80 |
| 12 | dialectic_polarity_gap (D3) | 0.5 | 2.0 | 1.20 |
| 13 | zaratustra_alpha (Z1) | 0.01 | 0.5 | 0.10 |
| 14 | zaratustra_decay (Z2) | 0.001 | 0.1 | 0.02 |
| 15 | echo_threshold (Z4) | 0.2 | 0.95 | 0.70 |
| 16 | ubermensch_fraction (Z5) | 0.01 | 0.5 | 0.10 |
| 17 | hausdorff_lo (L1) | 0.1 | 1.0 | 0.50 |
| 18 | hausdorff_hi (L2) | 1.2 | 2.5 | 1.90 |
| 19 | sleep_noise (S1) | 0.001 | 0.1 | 0.02 |
| 20 | sleep_adam_lr (S2) | 1e-4 | 5e-2 | 5e-3 |
| 21 | fusion_distance_threshold (S4) | 0.1 | 1.5 | 0.50 |
| 22 | dream_energy_spike (DR1) | 0.5 | 0.99 | 0.90 |
| 23 | depth_cap_gradient (CB4) | 0.0 | 0.8 | 0.30 |

---

## 3. Neural Architecture

### 3.1 Model Design: Multi-Output MLP

```
Input Layer:    [batch, 16]   (context features)
    |
Hidden Layer 1: [batch, 64]   (ReLU activation)
    |
Hidden Layer 2: [batch, 32]   (ReLU activation)
    |
Output Layer:   [batch, 24]   (Sigmoid activation → affine scale)
```

**Total parameters**: 16*64 + 64 + 64*32 + 32 + 32*24 + 24 = **3,960 parameters**

This is intentionally tiny (~16 KB ONNX file). Inference cost: <10 microseconds on CPU.

### 3.2 Why One Multi-Output Model (Not Separate Models)

- **Correlation matters**: Hausdorff thresholds (lo/hi) must be jointly calibrated; alpha/decay must be inversely related; coherence and entropy thresholds co-depend.
- **Shared representation**: The hidden layers learn a compressed "graph health state" that all thresholds condition on.
- **Deployment simplicity**: One ONNX file, one inference call, one version to track.

### 3.3 Training Strategy

**Phase 1 — Imitation Learning (Warm Start)**
- Collect historical `(HealthReport, effective_thresholds)` pairs from production logs.
- The "effective threshold" for tick N is the hard-coded default (initially), augmented by the *actual* agency behavior observed.
- Train the MLP to replicate current behavior exactly (behavioral cloning).
- Loss: MSE between predicted thresholds and the defaults (weighted by threshold importance).

**Phase 2 — Reward-Driven Fine-Tuning**
- Define reward function from subsequent health reports:
  ```
  R = w_energy * mean_energy_stability
    + w_fractal * is_fractal_maintained
    + w_coherence * coherence_improvement
    - w_entropy * entropy_spike_count
    - w_gap * gap_fraction_increase
  ```
- Use simple evolutionary strategy (CMA-ES) or REINFORCE gradient:
  - Perturb thresholds slightly from MLP outputs.
  - Measure reward over the next N ticks.
  - Update MLP weights toward higher-reward threshold vectors.

**Phase 3 — Online Adaptation**
- After initial offline training, the model runs live with slow online updates.
- Exponential moving average of gradients (conservative update rate: `lr_online = 1e-5`).
- Each health report becomes a single-sample training signal.
- Bounded update: per-tick weight change capped at 0.1% of current weight.

### 3.4 Input Feature Computation

All features computed from `HealthReport` + `AgencyTickReport` at the *start* of each tick, before thresholds are consumed:

```rust
fn compute_context_features(
    report: &HealthReport,
    tick_report: &AgencyTickReport,
    pending_desires: usize,
) -> [f32; 16] {
    [
        report.mean_energy.clamp(0.0, 1.0),
        (report.energy_std / 0.5).clamp(0.0, 1.0),
        (report.global_hausdorff / 3.0).clamp(0.0, 1.0),
        report.coherence_score.clamp(0.0, 1.0) as f32,
        (report.gap_count as f32 / 80.0).clamp(0.0, 1.0),
        (report.entropy_spike_count as f32 / 32.0).clamp(0.0, 1.0),
        (report.total_nodes as f32).log10().max(0.0) / 7.0,
        (report.total_edges as f32).log10().max(0.0) / 7.0,
        if report.is_fractal { 1.0 } else { 0.0 },
        report.energy_percentiles.p10,
        report.energy_percentiles.p90,
        (report.tick_number as f32 * std::f32::consts::TAU / 100.0).sin(),
        // ... remaining features from tick_report
        0.0, // placeholder: recent sleep cycles
        0.0, // placeholder: recent lsystem ticks
        0.0, // placeholder: active action nodes
        (pending_desires as f32 / 50.0).clamp(0.0, 1.0),
    ]
}
```

---

## 4. Integration Point

### 4.1 Code-as-Data Pattern

**Current code** (scattered throughout codebase):
```rust
if score > 0.85 {
    // trigger action
}
```

**Neural threshold code** (unified):
```rust
use nietzsche_agency::neural::ThresholdProvider;

// In AgencyEngine::tick():
let thresholds = self.threshold_provider.get_thresholds(&context);

// In EntropyDaemon:
if variance > thresholds.entropy_variance_threshold {
    bus.publish(AgencyEvent::EntropySpike { .. });
}
```

### 4.2 ThresholdProvider Trait

```rust
/// Provides adaptive thresholds for the agency engine.
///
/// Two implementations:
/// - `StaticThresholds`: wraps AgencyConfig defaults (feature = "default")
/// - `NeuralThresholds`: ONNX inference (feature = "neural")
pub trait ThresholdProvider: Send + Sync {
    fn get_thresholds(&self, context: &ThresholdContext) -> AdaptiveThresholds;
}

/// All learnable thresholds in one struct.
#[derive(Debug, Clone)]
pub struct AdaptiveThresholds {
    // Agency
    pub entropy_variance_threshold: f32,
    pub coherence_overlap_threshold: f64,
    pub gap_min_density: f64,
    pub observer_wake_energy_threshold: f32,
    pub observer_wake_hausdorff_lo: f32,
    pub observer_wake_hausdorff_hi: f32,
    pub desire_dream_threshold: f32,
    pub niilista_distance_threshold: f32,
    pub min_energy_for_probe: f32,
    pub reactor_cooldown_ticks: u64,
    pub evolution_cooldown_ticks: u64,

    // Dialectic
    pub dialectic_proximity_threshold: f64,
    pub dialectic_polarity_gap: f32,

    // Zaratustra
    pub zaratustra_alpha: f32,
    pub zaratustra_decay: f32,
    pub echo_threshold: f32,
    pub ubermensch_fraction: f32,

    // L-System
    pub hausdorff_lo: f32,
    pub hausdorff_hi: f32,

    // Sleep
    pub sleep_noise: f64,
    pub sleep_adam_lr: f64,
    pub fusion_distance_threshold: f64,

    // Dream
    pub dream_energy_spike: f64,

    // Circuit Breaker (learnable subset)
    pub depth_cap_gradient: f32,
}
```

### 4.3 Feature Flag Integration

```toml
# Cargo.toml
[features]
default = []
neural = ["ort"]  # ONNX Runtime dependency
```

```rust
#[cfg(feature = "neural")]
pub struct NeuralThresholdProvider {
    session: ort::Session,
    config: NeuralThresholdConfig,
    fallback: AdaptiveThresholds,  // hard-coded defaults for fallback
}

#[cfg(not(feature = "neural"))]
pub struct StaticThresholdProvider {
    thresholds: AdaptiveThresholds,
}
```

### 4.4 Configuration

```rust
/// Per-threshold safety bounds.
pub struct NeuralThresholdConfig {
    /// Path to the ONNX model file.
    pub model_path: String,

    /// Per-threshold min/max bounds. The neural output is ALWAYS clamped
    /// to these bounds, regardless of what the model predicts.
    pub bounds: Vec<ThresholdBound>,

    /// If true, log every threshold value to the audit trail.
    pub audit_enabled: bool,

    /// Maximum allowed deviation from the default per tick.
    /// Prevents sudden wild swings. Default: 20% of range.
    pub max_deviation_per_tick: f32,
}

pub struct ThresholdBound {
    pub name: String,
    pub min_safe: f64,
    pub max_safe: f64,
    pub default: f64,
}
```

---

## 5. Safety & Fallback

### 5.1 SACRED Thresholds — NEVER Neural

These thresholds MUST remain hard-coded. They are safety-critical and/or mathematical invariants:

| ID | Threshold | Value | Reason |
|----|-----------|-------|--------|
| A9 | circuit_breaker_max_actions | 20 | Prevents runaway autonomous action storms |
| A10 | circuit_breaker_energy_sum | 50.0 | Global energy cap prevents pathological accumulation |
| A11 | LTD rate | 0.05 | Caregiver correction rate — clinical safety |
| A12 | LTD correction threshold | 1 | Must respond to every correction — clinical safety |
| Z3 | Energy cap | 1.0 | Mathematical: energy is in [0,1] by definition |
| L3 | Circuit breaker sigma | 2.0 | Statistical safety bound (2-sigma rule) |
| CB1 | Overheated threshold | 0.92 | Tumor detection — must be conservative |
| CB2 | Min tumor size | 3 | Prevents false positives on isolated hot nodes |
| CB3 | Dampening factor | 0.7 | Must aggressively drain tumors |
| CB5 | Base energy cap | 1.0 | Mathematical invariant |
| CB6 | Max energy delta | 0.25 | Rate limiting — prevents energy explosions |
| S3 | Hausdorff threshold (sleep) | 0.15 | Sleep safety gate — rollback if exceeded |
| IN5 | MAX_NORM | 0.999 | Mathematical: Poincare ball boundary |
| DF2 | LAMBDA_MAX | 2.0 | Mathematical: normalized Laplacian bound |
| API* | All API limits | Various | DoS prevention — must be hard-coded |
| EV1 | Speaker match threshold | 0.75 | Clinical: voice identification safety |
| EV2 | Confidence gate | 0.7 | Clinical: don't respond without confidence |
| EV5 | Veracity threshold | 0.6 | Clinical: inconsistency detection |
| EV7 | Family sessions threshold | 3 | Clinical: minimum data for family analysis |

### 5.2 Mandatory Fallback Chain

```
1. ONNX inference attempt
   |
   +--[SUCCESS]--> Clamp outputs to [min_safe, max_safe]
   |                   |
   |                   +--[Deviation check]--> If any threshold changed > max_deviation_per_tick
   |                   |                        from previous tick, clamp to max_deviation
   |                   |
   |                   +--[Audit log]--> Record all 24 threshold values + context features
   |                   |
   |                   +--[Return adaptive thresholds]
   |
   +--[FAILURE: model load error]--> Log error, return hard-coded defaults
   |
   +--[FAILURE: inference error]--> Log error, return previous tick's thresholds
   |                                 (or defaults if first tick)
   |
   +--[FAILURE: NaN/Inf in output]--> Log error, return hard-coded defaults
```

### 5.3 Clinical Threshold Extra Constraints

For any threshold that affects EVA's clinical behavior (directly or indirectly):

1. **Tighter bounds**: Clinical thresholds use 50% of the nominal safe range.
   - Example: `echo_threshold` range is [0.2, 0.95] nominal, but clinical mode restricts to [0.4, 0.85].

2. **Slower adaptation**: Online learning rate for clinical thresholds is 10x lower.
   - `lr_clinical = lr_online / 10.0`

3. **Mandatory human review**: If any clinical threshold drifts more than 10% from its default over 100 ticks, the system emits an alert and freezes the threshold at the default until a human operator acknowledges.

4. **LTD rate is ALWAYS hard-coded**: The caregiver correction mechanism (LTD daemon) uses a fixed 5% rate. This is a clinical safety guarantee that the system responds predictably to corrections. Neural adaptation of this value is **permanently rejected**.

### 5.4 Audit Trail

Every tick that uses neural thresholds MUST record:

```rust
#[derive(Debug, Serialize)]
pub struct ThresholdAuditEntry {
    pub tick_number: u64,
    pub timestamp_ms: u64,
    /// The 16-dimensional context vector that was fed to the model.
    pub context_features: Vec<f32>,
    /// The 24 threshold values produced (after clamping).
    pub threshold_values: Vec<f64>,
    /// Which thresholds were clamped by safety bounds.
    pub clamped_indices: Vec<usize>,
    /// Whether fallback was used (and why).
    pub fallback_reason: Option<String>,
    /// Outcomes: what happened in this tick as a result of these thresholds.
    pub daemon_events: usize,
    pub intents_produced: usize,
}
```

Audit entries are stored in `CF_META` with key prefix `agency:threshold_audit:` and pruned to keep the most recent 1000 entries.

### 5.5 Kill Switch

```rust
impl NeuralThresholdProvider {
    /// Permanently disable neural thresholds for this session.
    /// Called if audit detects anomalous behavior patterns.
    pub fn emergency_disable(&self) {
        self.disabled.store(true, Ordering::SeqCst);
        tracing::error!(
            "NEURAL THRESHOLDS EMERGENCY DISABLED — reverting to hard-coded defaults"
        );
    }
}
```

Trigger conditions for automatic kill switch:
- 3 consecutive ticks with NaN/Inf outputs
- Any threshold hitting its min/max bound for 10 consecutive ticks
- Circuit breaker tripping 3 times in 5 ticks (suggests thresholds are too permissive)
- Health report shows mean_energy < 0.05 or > 0.98 for 5 consecutive ticks

---

## 6. Implementation Roadmap

### Phase 0: Instrumentation (prerequisite)
- Add `ThresholdProvider` trait with `StaticThresholdProvider` implementation.
- Refactor all daemon/reactor code to read thresholds from the provider instead of directly from `AgencyConfig`.
- Add audit logging infrastructure to `CF_META`.
- **Zero behavioral change** — this is a pure refactor.

### Phase 1: Data Collection
- Deploy instrumented build to production.
- Collect 10,000+ tick reports with context features and outcomes.
- Build training dataset: `(context_features, default_thresholds, health_outcomes)`.

### Phase 2: Offline Training
- Train MLP to replicate current defaults (behavioral cloning).
- Validate: neural thresholds must produce identical behavior to defaults on historical data.
- Deploy as shadow model: runs alongside defaults, logs predictions, but does not control.

### Phase 3: Gradual Rollout
- Feature flag: `AGENCY_NEURAL_THRESHOLDS=true`
- Start with 5% of thresholds being neural (least critical ones: cooldowns, scan limits).
- Monitor for 1 week. Compare health metrics to baseline.
- Gradually increase to all LEARNABLE thresholds over 4 weeks.

### Phase 4: Online Adaptation
- Enable online learning with extremely conservative learning rate.
- Monitor drift metrics. If any threshold drifts > 15% from default, alert.
- Weekly human review of audit trail.

---

## 7. Summary Statistics

| Category | Total Thresholds | LEARNABLE | SACRED |
|----------|-----------------|-----------|--------|
| Agency Engine | 17 | 12 | 5 |
| Dialectic | 6 | 6 | 0 |
| Evolution | 6 | 6 | 0 |
| Reactor | 5 | 3 | 2 (bounds) |
| Observer Identity | 3 | 2 | 1 |
| Zaratustra | 5 | 4 | 1 |
| L-System | 4 | 2 | 2 |
| Circuit Breaker | 6 | 1 | 5 |
| Sleep | 5 | 4 | 1 |
| Dream | 3 | 3 | 0 |
| Diffusion | 3 | 2 | 1 |
| Infrastructure | 7 | 5 | 2 |
| API | 5 | 0 | 5 |
| EVA Clinical | 7 | 3 | 4 |
| Desire Engine | 3 | 3 | 0 |
| **TOTAL** | **85** | **56** | **29** |

**56 thresholds** (66%) can be replaced with neural inference.
**29 thresholds** (34%) must remain hard-coded for safety.
**24 thresholds** are selected for the initial ONNX model (the most impactful LEARNABLE ones).

---

*This blueprint is the most safety-sensitive neural component in the EVA v2.0 architecture. Every design decision prioritizes fail-safe behavior: hard bounds, mandatory fallback, audit trails, and clinical constraints. The system must be indistinguishable from the static defaults until proven safe through extensive shadow deployment.*
