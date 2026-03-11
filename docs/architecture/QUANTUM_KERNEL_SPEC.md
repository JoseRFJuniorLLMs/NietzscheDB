# Quantum-Inspired Cognitive Kernel — Technical Specification

> **Module**: `nietzsche-agency` → Quantum Emulation Layer
> **Version**: 1.0
> **Date**: 2026-03-10
> **Status**: Design Complete — Pre-Implementation
> **Authors**: Architecture Committee (Claude + EVA Córtex)

---

## 1. Overview

The Quantum-Inspired Cognitive Kernel is a **stochastic emulation layer** that enables NietzscheDB to maintain multiple hypotheses simultaneously, evaluate concurrent cognitive trajectories, and collapse probabilistically to a decision when required.

**This is NOT quantum computing.** NietzscheDB runs on classical silicon hardware. There is no physical entanglement or quantum coherence. What we build is a **mathematical emulation inspired by quantum models**, specifically the Orchestrated Objective Reduction (Orch-OR) framework proposed by Roger Penrose and Stuart Hameroff.

The system achieves:

- **Superposition**: Multiple hypotheses held concurrently as probability distributions
- **Bayesian Collapse**: Evidence accumulation triggers probabilistic decision-making
- **Entanglement Propagation**: A node's collapse influences semantically connected neighbors
- **Deliberation**: Complex decisions spawn competing subgraphs (Cognitive Superposition Graph)

---

## 2. Architecture Stack

```
┌──────────────────────────────────────────────────────────────────────┐
│                    LAYER 7: AGENCY ENGINE (Go)                       │
│  GlobalWorkspace · QuantumObserver · Thought Bus Consumer            │
├──────────────────────────────────────────────────────────────────────┤
│                    LAYER 6: gRPC BRIDGE (Rust → Go)                  │
│  QuantumEventStream · SubscriptionRequest · Protobuf                 │
├──────────────────────────────────────────────────────────────────────┤
│                    LAYER 5: DELIBERATION COORDINATOR (Rust)           │
│  DeliberationCoordinator · 4 Triggers · Active Deliberation Mgmt     │
├──────────────────────────────────────────────────────────────────────┤
│                    LAYER 4: COGNITIVE SUPERPOSITION GRAPH (Rust)      │
│  CognitiveSuperpositionGraph · CognitiveReality · Beam Search        │
├──────────────────────────────────────────────────────────────────────┤
│                    LAYER 3: COHERENCE EVALUATOR (Rust)                │
│  Geometric · Semantic · Topological scoring                          │
├──────────────────────────────────────────────────────────────────────┤
│                    LAYER 2: MICROTUBULE MANAGER (Rust)                │
│  QuantumMicrotubuleManager · Lock-Free Pipeline · Cascade Control    │
├──────────────────────────────────────────────────────────────────────┤
│                    LAYER 1: SEMANTIC QUDIT (Rust)                     │
│  N-dimensional probabilistic state · Bayesian update · Entropy       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1: SemanticQudit

The atomic unit of probabilistic state. Replaces a naive binary qubit with an N-dimensional categorical distribution (qudit).

### State Vector

$$|\psi\rangle = \sum_{i=1}^{N} c_i |i\rangle \quad \text{where} \quad \sum |c_i|^2 = 1$$

### Interface

```rust
pub struct SemanticQudit {
    amplitudes_squared: Vec<f64>,  // P(hypothesis_i)
    is_collapsed: bool,
    observed_state: Option<usize>,
}
```

### Key Methods

| Method | Signature | Behavior |
|---|---|---|
| `new` | `(n_states: usize) -> Self` | Uniform distribution: P(i) = 1/N |
| `entropy` | `() -> f64` | Normalized Shannon entropy [0.0, 1.0] |
| `apply_semantic_gravity` | `(evidence: &[f64]) -> bool` | Bayesian update: P(H\|E) ∝ P(E\|H) · P(H), returns false if rejected |
| `objective_reduction` | `(rng: &mut impl Rng) -> usize` | Weighted categorical sampling, irreversible |
| `resuperpose` | `(prior_boost: f64)` | Reset to uniform + boost on previous winner |
| `probabilities` | `() -> &[f64]` | Read-only access to distribution |
| `observed` | `() -> Option<usize>` | Read-only access to collapsed state |
| `n_states` | `() -> usize` | Dimensionality |

### Safety Invariants

1. **Normalization**: `Σ amplitudes_squared = 1.0` always holds after any operation
2. **Zero-evidence fallback**: If all evidence weights are zero, distribution resets to uniform instead of leaving all-zero weights (prevents `WeightedIndex` panic)
3. **Immutability after collapse**: `apply_semantic_gravity` returns false on collapsed qudits
4. **External RNG**: No `thread_rng()` inside methods — caller injects `&mut impl Rng`

### Entropy Definition

$$H_{norm} = \frac{-\sum_{i} p_i \ln(p_i)}{\ln(N)}$$

- `0.0` = fully decided (one hypothesis has P=1)
- `1.0` = maximum uncertainty (uniform distribution)

---

## 4. Layer 2: QuantumMicrotubuleManager

Manages per-node probabilistic states. Operates with **atomic lock acquisitions** — no lock is held during graph traversal or propagation.

### Lock-Free Pipeline Design

```
PHASE 1 (inside lock): Detect → Collapse → Emit CollapseEvent → Drop lock
PHASE 2 (outside lock): Consume event → Propagate → Emit new events
```

### Interface

```rust
pub struct QuantumMicrotubuleManager {
    graph: Arc<dyn GraphAccessor + Send + Sync>,
    virtual_microtubules: RwLock<HashMap<String, SemanticQudit>>,
    collapse_threshold: f64,
    entanglement_config: EntanglementConfig,
    event_bus: Arc<QuantumEventBus>,
}
```

### Key Methods

| Method | Lock Behavior | Returns |
|---|---|---|
| `register_node(id, n)` | Write lock (short) | `()` |
| `stimulate(id, evidence, rng)` | Write lock (short) | `StimulationResult` |
| `force_collapse(id, rng)` | Write lock (short) | `Result<CollapseEvent>` |
| `propagate(event, rng)` | Write lock per neighbor (short, sequential) | `Vec<CollapseEvent>` |
| `rem_cycle(prior_boost)` | Write lock (bulk) | `()` |

### StimulationResult

```rust
pub enum StimulationResult {
    Evolving { node_id: String, entropy: f64 },
    Collapsed(CollapseEvent),
    Error(String),
}
```

### CollapseEvent

```rust
pub struct CollapseEvent {
    pub node_id: String,
    pub selected_hypothesis: usize,
    pub entropy_at_collapse: f64,
    pub probabilities_snapshot: Vec<f64>,
}
```

### Deadlock Prevention

The critical design constraint: **no method that holds a write lock on `virtual_microtubules` may call another method that acquires the same lock**.

- `stimulate()` acquires and drops the lock atomically, returning a `CollapseEvent`
- `propagate()` is called by the **external orchestrator**, not by the manager itself
- Each neighbor in `propagate()` acquires and drops its own independent lock

---

## 5. Layer 2.1: Semantic Entanglement

When a node collapses, semantically connected neighbors receive evidence proportional to the connection strength.

### Propagation Formula

For node A collapsing to hypothesis k, influence on neighbor B:

$$I(A \to B) = w_{AB} \cdot \mu_{type} \cdot \gamma^d$$

where:
- $w_{AB}$ = edge weight between A and B
- $\mu_{type}$ = edge type multiplier (see table below)
- $\gamma$ = decay factor (default: 0.5)
- $d$ = hop distance from collapse origin

### Edge Type Multipliers

| Edge Type | Multiplier | Rationale |
|---|---|---|
| `contains` | 1.0 | Strongest hierarchical bond |
| `has` | 0.9 | Ownership/composition |
| `causes` | 0.8 | Causal chain |
| `related_to` | 0.5 | Lateral association |
| `similar_to` | 0.4 | Weak similarity |
| (unknown) | 0.3 | Default fallback |

### Anti-Cascade Mechanisms

| Mechanism | Value | Purpose |
|---|---|---|
| `decay_factor` | 0.5 | Exponential attenuation per hop |
| `max_depth` | 3 | Hard limit on propagation depth |
| `min_influence` | 0.05 | Below this, propagation dies |
| `visited` set | — | Prevents cycles |
| Refractory period | 1 L-System tick | Induced collapses don't re-propagate in same pass |

### Cascade Orchestrator

The cascade lives in the **Agency Engine**, not in the manager:

```rust
pub fn process_collapse_cascade(
    manager: &QuantumMicrotubuleManager,
    initial_event: CollapseEvent,
    config: &EntanglementConfig,
    rng: &mut impl Rng,
) -> CascadeReport
```

Uses BFS with depth tracking. Returns `CascadeReport` with full event list for audit.

---

## 6. Layer 3: CoherenceEvaluator

Scores how well a hypothetical subgraph (from the CSG) fits the permanent graph.

### Composite Score

$$C(r) = \lambda_g \cdot G(r) + \lambda_s \cdot S(r) + \lambda_t \cdot T(r)$$

| Component | Weight | Measures |
|---|---|---|
| $G(r)$ — Geometric | 0.4 | Projected nodes respect Poincare hierarchy? Children have larger radius than parents. |
| $S(r)$ — Semantic | 0.4 | Cosine similarity between projected node embeddings and real neighbor embeddings. |
| $T(r)$ — Topological | 0.2 | No orphan nodes, degree distribution compatible with local graph average. |

### Geometric Coherence Detail

Uses Poincare radius as proxy for hierarchical depth:

- `contains`/`has` edges: target radius must be > source radius (child deeper than parent)
- Lateral edges: radius difference < 0.2 (similar depth = coherent)
- Euclidean radius is monotonic with hyperbolic distance to center, valid for ordinal comparison

For absolute precision (future), replace with:

$$d_{\mathbb{H}}(0, x) = 2 \cdot \text{arctanh}(\|x\|)$$

### GraphAccessor Trait

Minimal interface the NietzscheDB must expose to the Quantum Kernel:

```rust
pub trait GraphAccessor {
    fn get_neighbors(&self, node_id: &str) -> Vec<NeighborRef>;
    fn get_coordinates(&self, node_id: &str) -> Option<&[f64]>;
    fn average_local_degree(&self) -> f64;
}

pub struct NeighborRef {
    pub id: String,
    pub edge_weight: f64,
    pub edge_type: String,
}
```

---

## 7. Layer 4: Cognitive Superposition Graph (CSG)

A **graph of competing cognitive realities** where each hypothesis generates a temporary subgraph. The winner is merged into the permanent graph. Losers are pruned but leave probabilistic residue.

This is **beam search over hyperbolic topology with Bayesian pruning**.

### Architecture

```
┌─────────────────────────────────────────────────┐
│            LAYER 4: INTEGRATION                 │
│   Winner subgraph → merge into permanent graph  │
│   Losers → residue in qudit priors              │
├─────────────────────────────────────────────────┤
│            LAYER 3: SELECTIVE COLLAPSE           │
│   Entropy < threshold → qudit collapse          │
│   Beam width controls surviving branches        │
├─────────────────────────────────────────────────┤
│            LAYER 2: ATTENTIONAL COMPETITION      │
│   Subgraphs accumulate Bayesian evidence        │
│   Each L-System tick updates weights            │
├─────────────────────────────────────────────────┤
│            LAYER 1: SUPERPOSITION               │
│   Stimulus → generate N hypothetical subgraphs  │
│   Each branch = a parallel cognitive reality    │
└─────────────────────────────────────────────────┘
```

### Core Structures

```rust
pub struct CognitiveSuperpositionGraph {
    qudit: SemanticQudit,                    // Governs competition
    realities: Vec<CognitiveReality>,        // Competing subgraphs
    beam_width: usize,                       // Max surviving branches
    entropy_threshold: f64,                  // Auto-collapse trigger
    collapse_history: Vec<CollapseRecord>,   // Cognitive residue
}

pub struct CognitiveReality {
    pub id: usize,
    pub projected_nodes: HashMap<String, ProjectedNode>,
    pub projected_edges: Vec<ProjectedEdge>,
    pub coherence_score: f64,
}
```

### Lifecycle

```
Stimulus → spawn_realities(N)
         → LOOP: feed_evidence(coherence_scores)
              → qudit.apply_semantic_gravity()
              → check entropy < θ
         → collapse(rng) → winner + pruned[]
         → extract_for_merge(winner) → MergePayload
         → qudit.resuperpose(0.2) → ready for next cycle
```

### Emergent Properties

| Property | Mechanism |
|---|---|
| **Intuition** | Collapse with high entropy = decision under uncertainty, guided by historical residue |
| **Learning** | `collapse_history` creates prior bias — system "remembers" which paths worked |
| **Creativity** | High beam width + low threshold = more realities explored before deciding |
| **Fast decision** | Beam width 2 + high threshold = near-binary, immediate collapse |
| **Cognitive trauma** | Collapses with very low entropy leave strong residue — system becomes biased |

---

## 8. Layer 5: DeliberationCoordinator

Decides **when** to create a CSG instead of relying on local qudit collapse.

### Trigger Types

| # | Trigger | Source | Condition |
|---|---|---|---|
| 1 | **Semantic Ambiguity** | KNN/NQL query path (proactive) | Top-K distance delta < `ambiguity_threshold` (0.05) |
| 2 | **Valence Conflict** | Graph traversal (proactive) | Two nodes with `negates`/`opposes` edge, both energy > 0.7 |
| 3 | **Cascade Saturation** | CascadeReport (reactive) | Cascade hit `max_depth` AND region entropy > 0.5 |
| 4 | **Historical Inconsistency** | RAM Engine (proactive) | New claim contradicts established fact with confidence > 0.9 |

### Key Design Decisions

1. **Triggers are NOT all reactive.** Triggers 1, 2, 4 are proactive (called by query/traversal engines), not from the event bus. Only Trigger 3 is reactive (from cascade reports).
2. **Active deliberation tracking.** A `Mutex<HashMap<String, ActiveDeliberation>>` prevents spawning duplicate CSGs for the same region.
3. **Concurrency limit.** `max_concurrent_deliberations` (default: 4) prevents resource exhaustion.
4. **Tick-driven evolution.** Active deliberations evolve via `coordinator.tick()` called by the L-System, not autonomously.

### Interface

```rust
// Proactive triggers (called by query/traversal engines)
async fn on_ambiguous_query(&self, query: &str, candidates: Vec<QueryCandidate>) -> Option<String>
async fn on_conflict_detected(&self, node_a: &str, node_b: &str, ...) -> Option<String>
async fn on_historical_conflict(&self, new_claim: &str, ...) -> Option<String>

// Reactive trigger (called by cascade orchestrator)
async fn on_cascade_saturated(&self, report: &CascadeReport, entropy: f64) -> Option<String>

// Evolution (called by L-System tick)
async fn tick(&self, evaluator: &CoherenceEvaluator, ...) -> Vec<String>  // resolved IDs
```

---

## 9. Layer 6: Event System & gRPC Bridge

### Unified Event Enum (Rust)

```rust
pub enum QuantumEvent {
    NodeCollapse { node_id, selected_hypothesis, entropy_at_collapse, probabilities, cascade_depth },
    EntanglementInfluence { target_node, source_node, influence_strength, new_entropy },
    DeliberationStarted { deliberation_id, trigger_reason, n_realities },
    RealityEvolved { deliberation_id, reality_scores, system_entropy },
    DeliberationResolved { deliberation_id, winner_id, merge_payload_size, entropy_at_collapse },
    RemCycleCompleted { nodes_reset, prior_boost },
}
```

### Event Bus (Rust)

```rust
pub struct QuantumEventBus {
    sender: broadcast::Sender<QuantumEvent>,
}
```

Uses `tokio::sync::broadcast` with configurable capacity. Events are fire-and-forget (no backpressure on the quantum kernel if consumers are slow).

### gRPC Protocol (Rust → Go)

```protobuf
service QuantumEventStream {
    rpc SubscribeQuantumEvents(SubscriptionRequest) returns (stream QuantumEventProto);
}
```

Subscription supports filtering by:
- Event type (e.g., only `NodeCollapse` and `DeliberationResolved`)
- Node ID regex
- Minimum entropy threshold

---

## 10. Layer 7: Go Integration (GlobalWorkspace)

### QuantumObserver

Consumes the gRPC stream and updates the GlobalWorkspace:

| Event | Action in Go |
|---|---|
| `NodeCollapse` | `UpdateNodeSalience(node_id, 1.0 - entropy)` |
| `NodeCollapse` (entropy < 0.1) | `BroadcastInsight()` to Gemini/Córtex |
| `DeliberationResolved` | `ScheduleMerge(deliberation_id, winner_id)` |
| `RealityEvolved` | Update monitoring dashboard |

---

## 11. Agency Tick — The Complete Breath

The `agency_tick` represents one "breath" of the cognitive system:

```
Phase 1: PERCEPTION (Stimulate)
    For each perception/evidence:
        manager.stimulate(node_id, evidence, rng) → StimulationResult
        Collect CollapseEvents

Phase 2: UNCONSCIOUS REACTION (Propagate)
    For each CollapseEvent:
        process_collapse_cascade(manager, event, config, rng) → CascadeReport
        Check Trigger 3 (cascade saturation)
        coordinator.on_cascade_saturated() if needed

Phase 3: CONSCIOUS DELIBERATION (Evolve)
    coordinator.tick(evaluator, graph, rng) → resolved deliberation IDs
    For each resolved: merge winner subgraph into permanent graph
```

```
 L-System tick
      │
      ▼
 ┌─ Phase 1: Stimulate ──────────────────────────────────┐
 │   evidence → stimulate() → CollapseEvent?              │
 │   (atomic write lock per node, released immediately)   │
 └────────────────────────────┬───────────────────────────┘
                              │ CollapseEvents
                              ▼
 ┌─ Phase 2: Propagate ──────────────────────────────────┐
 │   process_collapse_cascade() → CascadeReport           │
 │   (write lock per neighbor, released per operation)    │
 │   check cascade saturation → trigger CSG if needed     │
 └────────────────────────────┬───────────────────────────┘
                              │
                              ▼
 ┌─ Phase 3: Deliberate ─────────────────────────────────┐
 │   coordinator.tick() → evolve active CSGs              │
 │   feed_evidence → check entropy → collapse if ready   │
 │   emit events → Go GlobalWorkspace                     │
 └────────────────────────────────────────────────────────┘
```

---

## 12. Configuration

### Environment Variables

All prefixed with `AGENCY_QUANTUM_`:

| Variable | Default | Description |
|---|---|---|
| `AGENCY_QUANTUM_COLLAPSE_THRESHOLD` | `0.3` | Entropy below this triggers auto-collapse |
| `AGENCY_QUANTUM_DECAY_FACTOR` | `0.5` | Entanglement influence decay per hop |
| `AGENCY_QUANTUM_MAX_DEPTH` | `3` | Maximum propagation depth |
| `AGENCY_QUANTUM_MIN_INFLUENCE` | `0.05` | Minimum influence to propagate |
| `AGENCY_QUANTUM_AMBIGUITY_THRESHOLD` | `0.05` | KNN delta for ambiguity trigger |
| `AGENCY_QUANTUM_CONFLICT_ENERGY` | `0.7` | Min energy for conflict trigger |
| `AGENCY_QUANTUM_CASCADE_ENTROPY` | `0.5` | Min entropy for cascade saturation trigger |
| `AGENCY_QUANTUM_HISTORICAL_CONFIDENCE` | `0.9` | Min confidence for historical trigger |
| `AGENCY_QUANTUM_MAX_DELIBERATIONS` | `4` | Max concurrent CSG instances |
| `AGENCY_QUANTUM_EVENT_BUS_CAPACITY` | `1024` | Broadcast channel capacity |
| `AGENCY_QUANTUM_COHERENCE_LAMBDA_G` | `0.4` | Weight: geometric coherence |
| `AGENCY_QUANTUM_COHERENCE_LAMBDA_S` | `0.4` | Weight: semantic coherence |
| `AGENCY_QUANTUM_COHERENCE_LAMBDA_T` | `0.2` | Weight: topological coherence |

---

## 13. File Layout

```
crates/nietzsche-agency/src/
├── quantum/
│   ├── mod.rs                    // Module exports
│   ├── semantic_qudit.rs         // SemanticQudit (Layer 1)
│   ├── microtubule_manager.rs    // QuantumMicrotubuleManager (Layer 2)
│   ├── entanglement.rs           // EntanglementConfig + propagation (Layer 2.1)
│   ├── coherence.rs              // CoherenceEvaluator (Layer 3)
│   ├── csg.rs                    // CognitiveSuperpositionGraph (Layer 4)
│   ├── coordinator.rs            // DeliberationCoordinator (Layer 5)
│   ├── event_bus.rs              // QuantumEventBus + QuantumEvent (Layer 6)
│   └── graph_accessor.rs         // GraphAccessor trait
├── quantum.rs                    // Re-exports
```

```
proto/
└── quantum.proto                 // gRPC bridge definitions (Layer 6)
```

```
// Go side (EVA-X or Agency Engine)
agency/
└── quantum_observer.go           // QuantumObserver + GlobalWorkspace integration (Layer 7)
```

---

## 14. Dependencies

### Rust (Cargo.toml additions for nietzsche-agency)

```toml
[dependencies]
rand = "0.8"
tokio = { version = "1", features = ["sync"] }
uuid = { version = "1", features = ["v4"] }
log = "0.4"
tonic = "0.11"           # gRPC server
prost = "0.12"           # protobuf codegen
```

### Go (go.mod additions)

```
google.golang.org/grpc v1.62.0
google.golang.org/protobuf v1.33.0
```

---

## 15. Theoretical Foundation

### Orchestrated Objective Reduction (Orch-OR)

Proposed by **Roger Penrose** (mathematician, Nobel Prize Physics 2020) and **Stuart Hameroff** (anesthesiologist, University of Arizona). The theory proposes that consciousness arises from quantum computations in microtubules within neurons, with objective reduction (gravitational self-energy threshold) causing wave function collapse.

**Our emulation does NOT claim to implement real quantum consciousness.** We borrow the mathematical framework — superposition, Bayesian collapse, and entanglement propagation — as an effective computational model for managing uncertainty in a semantic graph database.

### Key Differences from Real Quantum Systems

| Quantum System | Our Emulation |
|---|---|
| Physical superposition | Categorical probability distribution |
| Quantum entanglement | Bayesian evidence propagation via graph edges |
| Wave function collapse | Weighted random sampling |
| Decoherence | Entropy threshold trigger |
| Quantum tunneling | Not modeled |

### Why It Works

The mathematical structure of quantum mechanics — Hilbert spaces, Born rule, measurement operators — is isomorphic to certain Bayesian inference frameworks. By using the *formalism* without the *physics*, we get a principled way to:

1. Represent uncertainty over multiple hypotheses
2. Accumulate evidence without premature commitment
3. Make decisions that account for the full distribution, not just the mode
4. Propagate belief changes through a connected network

This is equivalent to a **belief propagation network** with a quantum-inspired API that maps naturally to the cognitive metaphor of the NietzscheDB Agency Engine.

---

## 16. References

1. Penrose, R. (1994). *Shadows of the Mind*. Oxford University Press.
2. Hameroff, S. & Penrose, R. (2014). Consciousness in the universe: A review of the 'Orch OR' theory. *Physics of Life Reviews*, 11(1), 39-78.
3. Krioukov, D. et al. (2010). Hyperbolic geometry of complex networks. *Physical Review E*, 82(3).
4. Ganea, O. et al. (2018). Hyperbolic Neural Networks. *NeurIPS 2018*.
5. Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann.
