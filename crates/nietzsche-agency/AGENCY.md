# nietzsche-agency — Autonomous Agency Engine

> "The graph no longer just remembers — it watches, desires, dreams, evolves, and acts."

## Overview

`nietzsche-agency` transforms NietzscheDB from a reactive knowledge graph into a **self-monitoring subconscious** with autonomous initiative. It implements the full agency stack identified in [agency.txt](../../EVA/docs/agency.txt):

1. **Daemons with Agency** — autonomous monitors that detect entropy, coherence, and knowledge gaps
2. **Motor de Desejo** — a desire engine that transforms gaps into structured knowledge queries
3. **Observer Identity** — a meta-node in the graph representing self-awareness
4. **Counterfactual Engine** — "what-if" simulation without mutating the real graph
5. **Reactor** — converts events into executable intents (sleep, L-System, dreams, Zaratustra, narrative)
6. **Open Evolution** — L-System rule mutation/selection driven by health metrics
7. **Neuromorphic/Quantum Bridge** — Poincare ball to Bloch sphere mapping

## Architecture

```
                    ┌─────────────┐
                    │ AgencyEngine │
                    └──────┬──────┘
                           │ tick()
    ┌─────────┬────────┬───┴───┬──────────┬──────────┬───────────┐
    │         │        │       │          │          │           │
    ▼         ▼        ▼       ▼          ▼          ▼           ▼
 Entropy  Coherence  Gap   Observer   Reactor    Desire     Counterfactual
 Daemon   Daemon    Daemon  Identity             Engine     Engine
    │         │        │       ▲          │
    └─────────┴────────┘       │          ▼
                 │ publish()   │   AgencyIntents
                 ▼             │   ┌──────────────────────────┐
           AgencyEventBus ─────┘   │ TriggerSleepCycle        │
                                   │ TriggerLSystemGrowth     │
    ┌──────────────────┐           │ TriggerDream             │
    │  Rule Evolution  │◄──────────│ ModulateZaratustra       │
    └──────────────────┘           │ GenerateNarrative        │
                                   │ EvolveLSystemRules       │
    ┌──────────────────┐           │ PersistHealthReport      │
    │  Quantum Bridge  │           │ SignalKnowledgeGap       │
    │  (Poincare→Bloch)│           └──────────────────────────┘
    └──────────────────┘
```

## Crate Structure

```
crates/nietzsche-agency/
├── Cargo.toml
├── AGENCY.md              # This file
└── src/
    ├── lib.rs             # Re-exports
    ├── error.rs           # AgencyError (thiserror)
    ├── config.rs          # AgencyConfig + from_env()
    ├── event_bus.rs       # Internal pub/sub (tokio::broadcast, capacity=256)
    ├── daemons/
    │   ├── mod.rs         # AgencyDaemon trait + DaemonReport
    │   ├── entropy.rs     # EntropyDaemon — Hausdorff variance detection
    │   ├── coherence.rs   # CoherenceDaemon — multi-scale diffusion overlap
    │   ├── gap.rs         # GapDaemon — sparse sector detection
    │   ├── ltd_daemon.rs  # LtdDaemon — link topology decay
    │   └── nezhmetdinov.rs # NezhmetdinovDaemon — Active Forgetting Engine
    ├── forgetting/        # *** NEZHMETDINOV ACTIVE FORGETTING SYSTEM ***
    │   ├── mod.rs         # Module root + re-exports (4 Camadas)
    │   ├── vitality.rs    # CAMADA 1: V(n) sigmoid vitality function
    │   ├── judgment.rs    # CAMADA 1: Verdict enum + MikhailThallReport
    │   ├── bounds.rs      # CAMADA 1: HardBounds + NezhmetdinovConfig
    │   ├── ricci.rs       # CAMADA 1: Ricci curvature shield (degree-variance proxy)
    │   ├── causal_immunity.rs # CAMADA 1: Minkowski causal classification
    │   ├── ledger.rs      # CAMADA 2: Merkle Tree deletion receipts
    │   ├── void_tracker.rs # CAMADA 3: Poincaré void coordinate seeds
    │   ├── tgc.rs         # CAMADA 4: Topological Generative Capacity
    │   ├── elite_drift.rs # CAMADA 4: Elite centroid drift tracking
    │   ├── anti_gaming.rs # CAMADA 4: Goodhart anti-gaming monitor
    │   ├── stability.rs   # CAMADA 4: Collapse detection (3 pathologies)
    │   ├── vitality_variance.rs # CAMADA 4: Cognitive health classification
    │   ├── friction.rs    # CAMADA 4: Forgetting friction (Ricci vetoes/cycle)
    │   ├── zaratustra_cycle.rs  # Master orchestrator (all 4 Camadas)
    │   └── telemetry.rs   # CSV telemetry writer
    ├── observer.rs        # MetaObserver (HealthReport + wake-ups)
    ├── reactor.rs         # AgencyReactor (events → intents)
    ├── desire.rs          # DesireEngine (Motor de Desejo)
    ├── identity.rs        # ObserverIdentity (meta-node in graph)
    ├── store.rs           # HealthReport persistence to CF_META
    ├── evolution.rs       # Open Evolution (L-System rule mutation/selection)
    ├── quantum.rs         # Neuromorphic/Quantum Bridge (Poincare → Bloch)
    ├── counterfactual/
    │   ├── mod.rs         # CounterfactualEngine public API
    │   ├── shadow.rs      # ShadowGraph (lightweight copy)
    │   └── simulator.rs   # BFS impact propagation
    ├── engine.rs          # AgencyEngine (top-level tick orchestrator)
    └── bin/
        └── simulate_forgetting.rs  # 5000-node × 500-cycle simulation binary
```

## Components

### Daemons (read-only graph scanners)

| Daemon | Detects | Complexity | Emits |
|--------|---------|------------|-------|
| **EntropyDaemon** | Hausdorff variance spikes per region | O(N) | `EntropySpike` |
| **GapDaemon** | Sparse sectors in the Poincare ball | O(N) | `KnowledgeGap` |
| **CoherenceDaemon** | Diffusion overlap across scales | O(probes * Laplacian) | `CoherenceDrop` |
| **LtdDaemon** | Link topology decay (edge staleness) | O(E) | `LinkDecay` |
| **NezhmetdinovDaemon** | Active forgetting — node vitality evaluation | O(N) | `ForgettingCondemned` |

All daemons use UUID hash for angular binning (avoids loading 12KB embeddings).

### Reactor (events -> intents)

The reactor converts agency events into declarative `AgencyIntent`s:

| Event | Intent | Action |
|-------|--------|--------|
| `EntropySpike` | `TriggerSleepCycle` | Reconsolidate embeddings via Riemannian Adam |
| `CoherenceDrop` | `TriggerLSystemGrowth` | Re-grow structure via L-System rules |
| `HausdorffOutOfRange` | `TriggerLSystemGrowth` | Force re-hierarchization |
| `MeanEnergyBelow` | `TriggerSleepCycle` | Emergency reconsolidation |
| `GapCountExceeded` | `TriggerLSystemGrowth` | Fill structural gaps |
| `HealthReport` | `PersistHealthReport` | Store to CF_META for historical tracking |
| `HealthReport` | `ModulateZaratustra` | Adjust alpha/decay based on energy |
| `HealthReport` | `EvolveLSystemRules` | Evolve production rules based on health |
| `HealthReport` | `GenerateNarrative` | Generate self-description of graph state |
| `ForgettingCondemned` | `HardDelete` | Delete condemned node from graph |
| `ForgettingCondemned` | `RecordDeletion` | Write Merkle receipt to deletion ledger |
| `ForgettingCycleComplete` | `PersistHealthReport` | Store forgetting metrics for auditing |
| High-priority desire | `TriggerDream` | Directed dream from seed node near gap |

Cooldown: configurable minimum ticks between repeated intents (default: 3).

### Zaratustra Modulation

The reactor computes adaptive Zaratustra parameters from HealthReport metrics:

| Condition | Alpha | Decay | Reason |
|-----------|-------|-------|--------|
| Energy < 0.2 (critical) | base * 2.0 (max 0.25) | base * 0.5 (min 0.005) | Boost energy injection |
| Energy < 0.4 (low) | base * 1.5 | base * 0.75 | Moderate energy boost |
| Energy > 0.85 (inflated) | base * 0.5 | base * 2.5 (max 0.08) | Drain excess energy |
| Entropy spikes > 3 | base (0.10) | base * 2.0 (max 0.06) | Force reconsolidation |
| Healthy range | 0.10 | 0.02 | Base parameters |

The server holds `ZaratustraConfig` behind `Arc<Mutex>` and updates it when `ModulateZaratustra` intents are executed.

### Desire -> Dream Loop

The engine closes the loop from desire signals to directed dreams:

1. Daemons detect knowledge gaps -> `KnowledgeGap` events
2. Motor de Desejo transforms gaps into `DesireSignal`s with priority/query
3. Engine checks desires above `desire_dream_threshold` (default 0.6)
4. `find_seed_near_sector()` scans NodeMeta to find the highest-energy node in/adjacent to the gap sector (UUID hash angular bin, same algorithm as daemons)
5. Emits `TriggerDream { seed_node_id, depth, reason }` intent
6. Server executes: `DreamEngine::dream_from(seed, depth, noise)` -> auto-applies if events found

Only the top desire is processed per tick to avoid intent storms.

### Motor de Desejo (Desire Engine)

Transforms `KnowledgeGap` events into structured **DesireSignal**s:
- Prioritized by depth (center = more critical) and density
- Persisted to CF_META for external consumption by EVA
- Query templates suggest what knowledge to seek
- Fulfillable via REST API when gap is filled

### Observer Identity (meta-node)

A special `NodeType::Concept` node at the center of the Poincare ball (depth ~ 0.01):
- Tagged with metadata `{"nietzsche_agency": "observer"}`
- Energy reflects normalized graph health (mean of energy, coherence, fractal indicators)
- Content stores the latest HealthReport as JSON
- UUID persisted in CF_META at key `agency:observer_id`

### Counterfactual Engine

Simulates "what-if" scenarios without mutating the real graph:
- `what_if_remove(node_id)` — predict impact of removing a node
- `what_if_add(meta, connect_to)` — predict impact of adding a node
- Uses lightweight ShadowGraph (~100 bytes/node, no embeddings)
- BFS energy propagation with 0.5x decay per hop

### Open Evolution (L-System Rule Mutation/Selection)

The evolution module observes graph health over time and adapts L-System production rules:

**Strategy selection** from HealthReport:

| Condition | Strategy | Rule Characteristics |
|-----------|----------|---------------------|
| Hausdorff not fractal | `Consolidate` | Energy boost + aggressive pruning |
| Many gaps (>30%) + healthy energy | `FavorGrowth` | Low thresholds, more child/lateral rules |
| High entropy or energy > 0.8 | `FavorPruning` | High thresholds, energy drain rules |
| Steady state | `Balanced` | Moderate growth + maintenance pruning |

**Evolved rule types**:
- `GrowthChild` — spawn child nodes
- `LateralAssociation` — create cross-links
- `PruneFading` — remove low-energy nodes
- `EnergyBoost { delta }` — adjust node energy (positive or negative)

**Fitness scoring**: `energy*0.25 + coherence*0.25 + fractal*0.3 - gap_penalty*0.1 - entropy_penalty*0.1`

**State persistence**: `EvolutionState` (generation, strategy, fitness_history) stored in CF_META at key `agency:evolution_state`.

### Neuromorphic/Quantum Bridge (Poincare -> Bloch)

Maps points in the hyperbolic Poincare ball to quantum states on the Bloch sphere, enabling future neuromorphic and quantum hardware integration.

**Mathematical foundation**:
- Radial `||p|| in [0,1)` -> polar `theta = 2*arctan(||p||)` (conformal map)
- Angular `atan2(p1, p0)` -> azimuthal `phi in [0, 2*pi)`
- Node energy `in [0,1]` -> state purity (mixed vs pure state)

**Key operations**:
- `poincare_to_bloch(embedding, energy)` — forward mapping
- `bloch_to_poincare(state)` — inverse mapping (round-trip preserving)
- `batch_poincare_to_bloch(nodes)` — efficient batch mapping
- `entanglement_proxy(group_a, group_b)` — average fidelity between groups (coupling measure)
- `BlochState::fidelity()` — quantum fidelity between two states
- `BlochState::trace_distance()` — `D(rho, sigma) = ||r - s|| / 2`
- `BlochState::apply_gate(gate)` — quantum gate operations (RotX, RotY, RotZ, Hadamard)

**Practical use**:
- Hardware mapping: Poincare embeddings -> qubit rotations for neuromorphic processing
- Fidelity-based similarity: quantum fidelity as alternative distance metric
- Entanglement proxy: cross-cluster fidelity as inter-group coupling measure

## Nezhmetdinov Active Forgetting Engine

> *"Named after Rashid Nezhmetdinov, the chess grandmaster famous for brilliant sacrifices.
> This engine sacrifices data persistence for topological clarity — proving that intelligent
> forgetting creates a more lucid mind."*

The Forgetting Engine is a **four-layer architecture** (4 Camadas) for intelligent data deletion,
implementing the Protocolo Niilista, TGC Integrada, Defesa de Identidade, and Auditabilidade.

### Architecture Overview

```
                    ┌──────────────────────────────────┐
                    │     ZARATUSTRA CYCLE              │
                    │  (Master Orchestrator per tick)   │
                    └─────────────┬────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
  CAMADA 1               CAMADA 2 + 3              CAMADA 4
  Julgamento Local       Registro + Metabolismo    Saude Global
  ┌──────────────┐       ┌────────────────┐       ┌───────────────┐
  │ Vitality V(n)│       │ Deletion Ledger│       │ TGC Calculator│
  │ Triple Cond. │       │ (Merkle Tree)  │       │ Elite Drift   │
  │ Ricci Shield │       │ Void Tracker   │       │ Anti-Gaming   │
  │ Causal Immun.│       │ (Poincare seed)│       │ Stability Mon.│
  │ Bounds Check │       │                │       │ Var(V) Health │
  └──────┬───────┘       └───────┬────────┘       └───────┬───────┘
         │                       │                         │
         └───────────── Events ──┴────── Telemetry ────────┘
                         │
                    ForgettingCondemned
                    ForgettingCycleComplete
```

### CAMADA 1 — O Julgamento Local

**Vitality Function**: `V(n) = sigma(w1*e + w2*H - w3*xi + w4*pi + w5*kappa - w6*tau)`

| Weight | Default | Signal |
|--------|---------|--------|
| `w1_energy` | 1.0 | Node energy `e in [0,1]` |
| `w2_hausdorff` | 0.8 | Local fractal dimension `H` |
| `w3_entropy` | 1.2 | Entropy delta `xi` (negative = protective) |
| `w4_elite_prox` | 1.5 | Proximity to elite centroid `pi` |
| `w5_causal` | **2.0** | Causal centrality `kappa` (strongest protector) |
| `w6_toxicity` | 1.0 | Emotional toxicity `tau = max(0,-valence)*arousal` |

**Triple Condition (Regra de Ouro)** — ALL three must hold for condemnation:
1. `V(n) < theta_vitality` (default 0.25)
2. `e(n) < theta_energy` (default 0.10)
3. `kappa(n) = 0` (zero timelike/lightlike Minkowski edges)

**Ricci Curvature Shield**: Degree-variance proxy `kappa_proxy(v) = 1 - Var(deg(N(v))) / E[deg(N(v))]^2`.
Removing a hub node with high local curvature (>0.5) and many edges (>5) triggers veto.

**Causal Immunity**: Nodes connected by Minkowski timelike/lightlike edges are immune to deletion.
Causal classification via `ds^2 = -c^2*dt^2 + ||dx||^2`.

### CAMADA 2 — Registro Historico

**Deletion Ledger**: Merkle Tree-based cryptographic receipts for tamper-proof auditing.
- Each deletion produces a `DeletionReceipt` with node_id, cycle, vitality, reason, structural_hash
- Merkle root computed via `DefaultHasher` double-hash over all receipts
- `InclusionProof` with sibling hashes for verification
- Full audit trail: `verify_receipt()` validates any receipt against the Merkle root

### CAMADA 3 — Metabolismo Generativo

**Void Tracker**: Captures Poincare coordinates of deleted nodes as generation seeds.
- `VoidCoordinate { coordinates, energy_at_death, depth, cycle_deleted }`
- Plausibility scoring: `score = 1.0 - energy * 0.5 - depth * 0.3`
- Seeds sorted by plausibility for directed dream generation
- Configurable capacity with LRU-style eviction

### CAMADA 4 — Saude Global (4 Vital Signs)

| Vital Sign | Module | Monitors | Alert Condition |
|------------|--------|----------|-----------------|
| **TGC** | `tgc.rs` | Topological Generative Capacity | TGC declining over window |
| **Var(V)** | `vitality_variance.rs` | Vitality distribution health | Monoculture (<0.05) or Chaotic (>0.25) |
| **Elite Drift** | `elite_drift.rs` | Identity preservation | Drift > threshold (identity lost) |
| **Anti-Gaming** | `anti_gaming.rs` | Goodhart violations | 5 violation types with 50% TGC penalty |

**TGC Formula**: `TGC(t) = (G_t / V_t) * Quality` with EMA smoothing (alpha=0.3).
Measures whether forgetting creates structural fertility or just destroys data.

**Cognitive Health Classes** (from Var(V)):
- `Diverse`: 0.05 <= Var(V) < 0.25 (healthy heterogeneity)
- `Monoculture`: Var(V) < 0.05 (elitist collapse risk)
- `Chaotic`: Var(V) >= 0.25 (no stable identity)
- `Exhausted`: mean(V) < 0.2 (minimalist collapse risk)

**Stability Monitor** — detects 3 pathological collapses:
1. `Elitist`: High elite%, low vitality variance (only elites survive)
2. `Minimalist`: Universe below min_size (too aggressive pruning)
3. `Stationary`: Zero deletions for N consecutive cycles (engine stalled)

**Anti-Gaming** — 5 Goodhart violation types:
1. `EnergyPumping`: Mean energy > 0.95 (artificial inflation)
2. `CausalSpam`: >50% nodes with causal edges (fake immunity)
3. `EliteMonoculture`: >80% nodes are elite (meaningless elites)
4. `ZeroForgetting`: 0 deletions in cycle (engine disabled)
5. `MassExtinction`: >30% nodes killed in single cycle

### Verdict Classification

| Verdict | Meaning | Action |
|---------|---------|--------|
| `Sacred` | V(n) > sacred_threshold OR kappa >= sacred_causal | Protected forever |
| `Dormant` | Doesn't meet any condition | Survives this cycle |
| `Condemned` | Triple condition met, no Ricci veto | Hard delete + ledger receipt |
| `Toxic` | High toxicity + no causal protection | Hard delete (fast-track) |
| `RicciShielded` | Triple condition met, but Ricci veto saved | Survives (topology critical) |

### HardBounds (Immutable Deployment Parameters)

These CANNOT be overridden by any Zaratustra cycle:

| Bound | Default | Purpose |
|-------|---------|---------|
| `min_universe_size` | 100 | Absolute floor — stops all deletion below this |
| `max_deletion_rate` | 0.10 | Max 10% of universe per cycle |
| `sacred_causal_threshold` | 3 | Nodes with >= 3 causal edges are auto-sacred |

### Simulation Binary

`cargo run --release --bin simulate_forgetting` runs a standalone simulation:
- 5,000 synthetic nodes (1,000 Signal + 4,000 Noise)
- 500 accelerated Zaratustra cycles
- CSV output: `forgetting_telemetry.csv` with all 4 vital signs
- Validates: zero false positives (no signal nodes killed), noise convergence to zero

## Configuration

All via environment variables:

| Env Var | Default | Purpose |
|---------|---------|---------|
| `AGENCY_TICK_SECS` | `60` | Background loop interval (0=disabled) |
| `AGENCY_ENTROPY_THRESHOLD` | `0.25` | Hausdorff variance threshold |
| `AGENCY_ENTROPY_REGIONS` | `8` | Angular regions for entropy scan |
| `AGENCY_COHERENCE_THRESHOLD` | `0.70` | Max Jaccard overlap before alert |
| `AGENCY_GAP_SECTORS` | `16` | Angular sector count |
| `AGENCY_GAP_DEPTH_BINS` | `5` | Radial depth bands |
| `AGENCY_GAP_MIN_DENSITY` | `0.1` | Min relative density per sector |
| `AGENCY_OBSERVER_INTERVAL` | `5` | Ticks between full health reports |
| `AGENCY_OBSERVER_ENERGY_WAKE` | `0.3` | Mean energy wake-up threshold |
| `AGENCY_OBSERVER_HAUSDORFF_LO` | `0.5` | Hausdorff lower bound for wake-up |
| `AGENCY_OBSERVER_HAUSDORFF_HI` | `1.9` | Hausdorff upper bound for wake-up |
| `AGENCY_REACTOR_COOLDOWN` | `3` | Ticks between repeated intents |
| `AGENCY_CF_MAX_HOPS` | `3` | Max BFS hops for counterfactual |
| `AGENCY_DESIRE_DREAM_THRESHOLD` | `0.6` | Min desire priority for auto-dream |
| `AGENCY_DESIRE_DREAM_DEPTH` | `5` | BFS depth for desire-triggered dreams |
| `AGENCY_EVOLUTION_COOLDOWN` | `5` | Min ticks between evolution suggestions |
| `NEZHMETDINOV_ENABLED` | `true` | Enable/disable the forgetting engine |
| `NEZHMETDINOV_VITALITY_THRESHOLD` | `0.25` | V(n) below this = candidate for deletion |
| `NEZHMETDINOV_ENERGY_THRESHOLD` | `0.10` | Energy below this = candidate for deletion |
| `NEZHMETDINOV_RICCI_EPSILON` | `0.15` | Ricci curvature veto sensitivity |
| `NEZHMETDINOV_TOXICITY_THRESHOLD` | `0.80` | Toxicity above this = fast-track deletion |
| `NEZHMETDINOV_SACRED_VITALITY` | `0.80` | V(n) above this = auto-sacred |
| `NEZHMETDINOV_MAX_SCAN` | `5000` | Max nodes scanned per tick |

## REST API Endpoints (Dashboard)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/agency/health` | All persisted HealthReports |
| GET | `/api/agency/health/latest` | Most recent HealthReport |
| GET | `/api/agency/counterfactual/remove/:id` | Simulate removing a node |
| POST | `/api/agency/counterfactual/add` | Simulate adding a node |
| GET | `/api/agency/desires` | List pending desire signals |
| POST | `/api/agency/desires/:id/fulfill` | Mark a desire as fulfilled |
| GET | `/api/agency/observer` | Observer Identity meta-node state |
| GET | `/api/agency/evolution` | Evolution state (generation, strategy, fitness history) |
| GET | `/api/agency/narrative` | Latest narrative self-description |
| POST | `/api/agency/quantum/map` | Map Poincare nodes to Bloch states |
| POST | `/api/agency/quantum/fidelity` | Compute entanglement proxy between groups |

## Server Integration

The agency engine runs as a background `tokio::spawn` task in `nietzsche-server`:

### Multi-Collection Support

The server maintains a `HashMap<String, AgencyEngine>` — one engine per collection. Each tick iterates all collections, creating engines lazily for new collections.

### Tick Protocol

1. **Read lock**: Daemons scan the graph, observer aggregates metrics, reactor produces intents
2. **Intent execution**: Server executes intents under write lock:

| Intent | Execution |
|--------|-----------|
| `TriggerSleepCycle` | Runs `SleepCycle::run()` |
| `TriggerLSystemGrowth` | Runs `LSystemEngine::tick()` with standard rules |
| `TriggerDream` | Runs `DreamEngine::dream_from(seed, depth, noise)`, auto-applies if events found |
| `ModulateZaratustra` | Updates shared `Arc<Mutex<ZaratustraConfig>>` alpha/decay |
| `GenerateNarrative` | Runs `NarrativeEngine::narrate()`, persists summary to CF_META at `agency:latest_narrative` |
| `EvolveLSystemRules` | Loads evolution state, generates evolved rules, converts to ProductionRules, runs L-System tick |
| `PersistHealthReport` | Writes to CF_META (atomic, safe under read lock) |
| `SignalKnowledgeGap` | Logged, desires persisted by Motor de Desejo |

3. **Observer Identity**: Created at startup per collection, updated with each HealthReport

### Dashboard UI

The HTML dashboard includes a dedicated **Agency** tab with real-time panels:

- **Health Report** — 8 metrics (nodes, edges, energy, std, hausdorff, coherence, gaps, entropy) with color-coded thresholds (good/warn/bad)
- **Observer Identity** — energy, node count, concept type
- **Motor de Desejo** — pending desire signals with priority and query
- **Rule Evolution** — current generation, strategy, fitness history
- **Narrative** — latest self-description text

## Tick Flow (Complete)

```
AgencyEngine::tick()
  │
  ├── 1. Tick daemons (Entropy, Gap, Coherence, Ltd, Nezhmetdinov + more)
  │      └── Emit: EntropySpike, KnowledgeGap, CoherenceDrop,
  │                ForgettingCondemned events
  │
  ├── 2. Tick observer (MetaObserver)
  │      └── Emit: HealthReport + DaemonWakeUp events (every N ticks)
  │
  ├── 3. Tick reactor (AgencyReactor)
  │      └── Produce: TriggerSleepCycle, TriggerLSystemGrowth,
  │                    ModulateZaratustra, EvolveLSystemRules,
  │                    GenerateNarrative, PersistHealthReport,
  │                    HardDelete, RecordDeletion,
  │                    SignalKnowledgeGap intents
  │
  ├── 4. Process gaps through Motor de Desejo
  │      └── Generate: DesireSignal list
  │
  ├── 5. Desire → Dream: find seed node near gap sector
  │      └── Produce: TriggerDream intent (if desire > threshold)
  │
  ├── 6. Update Observer Identity meta-node with latest health
  │
  └── Return: AgencyTickReport { daemon_reports, health_report, intents, desires }
```

## Test Coverage

155 unit tests covering all components (72 new forgetting tests):

**Core Agency (83 tests):**
- Event bus: publish/subscribe, multiple subscribers, no-op on empty
- EntropyDaemon: uniform (no spike), mixed (detects spike), empty graph
- GapDaemon: uniform (few gaps), sparse (many gaps), empty graph
- CoherenceDaemon: chain graph, too few nodes
- Observer: report intervals, wake-up triggers, energy percentiles
- Reactor: entropy->sleep, coherence->lsystem, cooldown, wake-up handling, zaratustra modulation, evolution trigger, narrative trigger, param calculation (healthy + inflated), forgetting condemned->hard delete
- Desire: generation, persistence, fulfillment, priority ordering
- Identity: create/retrieve, health update, is_observer check, low health
- Store: persist, retrieve, sort by tick, prune old entries
- Counterfactual: shadow snapshot, remove hub/leaf, add node
- Engine: full tick, health report interval, low energy intent, observer identity, empty graph, 8 daemons registered
- Evolution: strategy suggestions (balanced, growth, pruning, consolidate), rule thresholds, fitness scoring, state persistence
- Quantum: origin->north pole, boundary->equator, round-trip preservation, fidelity (same/orthogonal), Hadamard gate, RotZ, mixed states, entanglement proxy, trace distance, batch mapping

**Forgetting Engine (72 tests):**
- Vitality: sigmoid, weighted calculation, batch statistics, edge cases, input clamping
- Judgment: verdict classification, MikhailThallReport aggregation, death sentence detection
- Bounds: HardBounds immutability, NezhmetdinovConfig enforce_bounds, from_env parsing
- Ricci: star graph hub veto, leaf removal safe, isolated node safe, curvature proxy
- Causal Immunity: timelike/lightlike/spacelike classification, interval computation
- Ledger: receipt creation, Merkle root computation, inclusion proof verification, multi-receipt
- Void Tracker: coordinate capture, plausibility scoring, capacity limits, seed retrieval
- TGC: computation, EMA smoothing, declining detection, snapshot history
- Elite Drift: centroid tracking, drift computation, threshold alerting
- Anti-Gaming: 5 violation types, penalty calculation, report generation
- Stability: elitist/minimalist/stationary collapse detection, thermal perturbation
- Vitality Variance: cognitive health classification (diverse/monoculture/chaotic/exhausted)
- Friction: per-cycle friction scoring, Ricci veto rate tracking
- Zaratustra Cycle: full orchestration, ledger accumulation, multi-cycle execution
- NezhmetdinovDaemon: empty graph, high-energy sacred nodes, low-energy condemned nodes
