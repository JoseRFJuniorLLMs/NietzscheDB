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
    │   └── gap.rs         # GapDaemon — sparse sector detection
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
    └── engine.rs          # AgencyEngine (top-level tick orchestrator)
```

## Components

### Daemons (read-only graph scanners)

| Daemon | Detects | Complexity | Emits |
|--------|---------|------------|-------|
| **EntropyDaemon** | Hausdorff variance spikes per region | O(N) | `EntropySpike` |
| **GapDaemon** | Sparse sectors in the Poincare ball | O(N) | `KnowledgeGap` |
| **CoherenceDaemon** | Diffusion overlap across scales | O(probes * Laplacian) | `CoherenceDrop` |

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
  ├── 1. Tick daemons (EntropyDaemon, GapDaemon, CoherenceDaemon)
  │      └── Emit: EntropySpike, KnowledgeGap, CoherenceDrop events
  │
  ├── 2. Tick observer (MetaObserver)
  │      └── Emit: HealthReport + DaemonWakeUp events (every N ticks)
  │
  ├── 3. Tick reactor (AgencyReactor)
  │      └── Produce: TriggerSleepCycle, TriggerLSystemGrowth,
  │                    ModulateZaratustra, EvolveLSystemRules,
  │                    GenerateNarrative, PersistHealthReport,
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

68 unit tests covering all components:
- Event bus: publish/subscribe, multiple subscribers, no-op on empty
- EntropyDaemon: uniform (no spike), mixed (detects spike), empty graph
- GapDaemon: uniform (few gaps), sparse (many gaps), empty graph
- CoherenceDaemon: chain graph, too few nodes
- Observer: report intervals, wake-up triggers, energy percentiles
- Reactor: entropy->sleep, coherence->lsystem, cooldown, wake-up handling, zaratustra modulation, evolution trigger, narrative trigger, param calculation (healthy + inflated)
- Desire: generation, persistence, fulfillment, priority ordering
- Identity: create/retrieve, health update, is_observer check, low health
- Store: persist, retrieve, sort by tick, prune old entries
- Counterfactual: shadow snapshot, remove hub/leaf, add node
- Engine: full tick, health report interval, low energy intent, observer identity, empty graph
- Evolution: strategy suggestions (balanced, growth, pruning, consolidate), rule thresholds, fitness scoring, state persistence
- Quantum: origin->north pole, boundary->equator, round-trip preservation, fidelity (same/orthogonal), Hadamard gate, RotZ, mixed states, entanglement proxy, trace distance, batch mapping
