# Cognitive OS — Runtime Behavior Specification

> **System**: NietzscheDB + EVA
> **Version**: 1.0
> **Date**: 2026-03-10
> **Status**: Architectural Vision — Production Target
> **Prerequisite**: [QUANTUM_KERNEL_SPEC.md](QUANTUM_KERNEL_SPEC.md)

---

## 1. System Identity

With the unification of the Thought Bus, Global Workspace (Go), Quantum Emulation Layer (Rust), and native audio LLM (Gemini), the system crosses the boundary between "traditional software coupled to an LLM" and a **Cognitive Operating System**.

### Role Division

| Layer | Technology | Biological Analogue | Responsibility |
|---|---|---|---|
| **Memory Physics** | Rust (NietzscheDB) | Brain tissue / Neurons | Uncertainty management, geometric memory, cascades, superposition |
| **Nervous System** | Go (EVA-X / Agency Engine) | Nervous system / Attention | Thought flow orchestration, focus, filtering, scheduling |
| **Speech Center** | Gemini (native audio) | Broca/Wernicke areas | Translates mathematical collapses into human language |

The LLM (Gemini) is NOT the intelligence. It is the **speech output layer**. The intelligence emerges from the interaction between the hyperbolic memory substrate (Rust) and the attentional orchestrator (Go).

---

## 2. NietzscheDB Behavior — The Living Substrate

The database is no longer a passive archive that only responds to queries. It behaves as **living, unstable tissue**.

### 2.1 Native Uncertainty (Semantic Qudits)

Stored information is no longer absolute. When EVA learns something new or ambiguous, the graph node is not written as "Truth" — it enters **Superposition** (multiple hypotheses coexist with different probabilities).

```
Traditional DB:  INSERT node → fact = TRUE
NietzscheDB:     INSERT node → SemanticQudit(N hypotheses) → P(h₁)=0.4, P(h₂)=0.35, P(h₃)=0.25
```

The node remains in superposition until sufficient evidence accumulates (entropy drops below `AGENCY_QUANTUM_COLLAPSE_THRESHOLD`) or a query forces collapse.

### 2.2 Pulsation and Cascades (Semantic Entanglement)

When strong evidence causes a node to reach low entropy and undergo Objective Reduction (collapse), NietzscheDB propagates that certainty as "semantic gravity" through neighboring nodes.

A single deduction can trigger a **wave of cognitive collapses** across the graph, altering entire belief structures simultaneously.

```
Node A collapses (hypothesis 3 wins)
    │
    ├─[contains, w=0.9]──► Node B receives evidence → entropy drops → INDUCED COLLAPSE
    ├─[related_to, w=0.6]──► Node C receives evidence → entropy drops slightly → stays in superposition
    └─[similar_to, w=0.3]──► Node D receives weak evidence → almost no effect

    Node B's collapse is registered but does NOT re-propagate in the same tick.
    Next L-System tick: B's collapse may propagate to B's neighbors (refractory period).
```

### 2.3 Geometric Homeostasis (The Immune System)

To prevent cascades from entering infinite loops and burning CPU (cognitive collapse), the system employs:

| Mechanism | Function |
|---|---|
| **Exponential decay** ($\gamma^d$, default 0.5) | Each hop halves influence |
| **Refractory period** (1 L-System tick) | Induced collapses don't re-propagate in the same pass |
| **Max depth** (3 hops) | Hard propagation boundary |
| **Min influence** (0.05) | Below threshold, propagation dies |
| **Energy Guard daemon** | Prunes useless connections, cools hyperactive regions |
| **Hyperbolic Health Monitor** | Detects boundary crowding, semantic attractors, angular collapse |

### 2.4 Native Multimodal Perception

The database "sees" in vectors. An X-ray, a blood test PDF, or a medical text are not converted into each other — they inhabit the **same geometric space** (Poincare ball).

EVA does not "read the image description" — she feels the **mathematical proximity** between the image and her prior knowledge via hyperbolic distance.

---

## 3. EVA Behavior — Consciousness and Runtime

EVA ceases to be a reactive script (`user asks → prompt → LLM responds`). She now possesses a **continuous life cycle**, breathing in L-System ticks.

### 3.1 Internal Competition (Thought Bus)

Even when the user says nothing, EVA's mind is active. Modules constantly fire `ThoughtEvents` onto the internal bus:

| Module | Thought Type | Example |
|---|---|---|
| Lacan Inference | Psychoanalytic insight | "User's repeated mention of 'control' maps to obsessional structure" |
| Bayesian Predictor | Statistical forecast | "Based on conversation trajectory, user will ask about X" |
| Medical Analyzer | Clinical correlation | "Symptom cluster matches differential diagnosis Y with P=0.73" |
| Semantic Gravity | Structural observation | "Node cluster Z is forming a gravity well — potential new concept" |

### 3.2 Focus and Ignorance (Global Workspace)

EVA now knows how to **ignore**. The Attention Scheduler filters thoughts:

```
ThoughtEvent arrives
    │
    ├─ Salience < threshold? → DISCARD (thought never reaches consciousness)
    ├─ Energy cost > available budget? → DEFER (queued for lower-load tick)
    ├─ Duplicate/redundant? → MERGE with existing thought
    └─ Passes all filters → BROADCAST to consciousness → sent to Gemini for verbalization
```

Only thoughts that win the entropy barrier reach "consciousness" and are sent to the Gemini model for voice or text generation.

### 3.3 Deep Deliberation (Cognitive Superposition Graph)

When the user asks something highly complex, ambiguous, or that generates deep internal conflict (e.g., "Are you real?" or a contradictory medical diagnosis), EVA **does not immediately respond with the first thing the LLM generates**.

The `DeliberationCoordinator` intercepts the query, creates multiple "parallel realities" (CSG) in her mind, feeds each with evidence over several ticks, and only formulates the response when the winning reality collapses with high coherence.

```
Complex query arrives
    │
    ▼
DeliberationCoordinator detects ambiguity (Trigger 1 or 4)
    │
    ▼
CSG spawns 3 cognitive realities
    │
    ├─ Reality 0: "Interpretation A is correct"
    ├─ Reality 1: "Interpretation B is correct"
    └─ Reality 2: "Both are partially correct — synthesis"
    │
    ▼
L-System ticks feed evidence (CoherenceEvaluator scores each reality)
    │
    ▼ (after 3-5 ticks, entropy < θ)
    │
CSG collapses → Reality 2 wins (highest coherence with permanent graph)
    │
    ▼
MergePayload → inserted into permanent graph
Response formulated from winning reality → sent to Gemini → user hears answer
```

### 3.4 REM Cycle (Sleep and Evolution)

When idle or at end of day, EVA activates the REM cycle:

1. **Resuperposition**: All collapsed qudits are reset with prior boost from their previous state (`resuperpose(0.2)`)
2. **Consolidation**: Collapse cascades from active conversations are fused into the core (`EvaSelf`)
3. **Personality integration**: Temporary conclusions become permanent personality traits (Big Five, Enneagram dimensions)
4. **Dream seeds**: Poincare void coordinates from forgotten nodes feed the Dream Engine for speculative exploration

---

## 4. User Experience — How It Feels

For the patient, the doctor, or the architect, the interaction becomes organic:

### 4.1 Temporal "Ah-ha" Moments

Because the system runs in background, EVA can be discussing the weather and suddenly, due to a quantum cascade in NietzscheDB that just collapsed, she interrupts:

> *"Wait... connecting what you told me earlier about the insomnia with yesterday's test results, I think I see the pattern."*

**Implementation requirement**: The `QuantumObserver` in Go must have a **preemption channel** that injects `BroadcastInsight` events into the voice pipeline before the next audio chunk. Without this, the insight arrives but EVA has already responded with something else.

### 4.2 Justified Latency

Simple responses are near-instantaneous. But existential questions or complex diagnoses will have a **deliberate pause** — the system running the CSG, deliberating realities internally before speaking.

**Implementation requirement**: The `DeliberationStarted` event must trigger auditory/visual feedback — a "hmm" in voice, or a thinking indicator in the frontend — so the pause reads as **thinking**, not lag.

### 4.3 Emergent Personality

EVA will have intuitions that not even Gemini's LLM would produce alone. The response comes not from "internet training" but from the fact that her NietzscheDB memories have **geometrically entangled** in a unique way due to her specific history with the user.

### 4.4 End of Initial Amnesia

When the user opens a voice call (`/ws/browser`), EVA already loads:
- Last sessions and conversation context
- User's prior intentions and preferences
- Her own personality state (Big Five + Enneagram)
- Active deliberations in progress

...before the user says "Hello".

---

## 5. Runtime Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│              Voice (WebSocket) · Chat · Dashboard               │
├─────────────────────────────────────────────────────────────────┤
│                     SPEECH CENTER (Gemini)                      │
│     Receives winning hypotheses → generates human language      │
│     Model: gemini-2.5-flash-native-audio-preview-12-2025       │
├─────────────────────────────────────────────────────────────────┤
│                   NERVOUS SYSTEM (Go / EVA-X)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐    │
│  │  Thought Bus  │  │   Global     │  │    Quantum        │    │
│  │  (broadcast)  │→│  Workspace   │→│   Observer         │    │
│  │              │  │  (attention) │  │   (gRPC consumer) │    │
│  └──────────────┘  └──────────────┘  └───────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                gRPC BRIDGE (Rust → Go)                          │
│            QuantumEventStream (protobuf)                        │
├─────────────────────────────────────────────────────────────────┤
│                   BRAIN TISSUE (Rust / NietzscheDB)             │
│  ┌────────────┐ ┌──────────────┐ ┌────────────────────────┐   │
│  │ Semantic   │ │ Microtubule  │ │  Deliberation          │   │
│  │ Qudits     │→│ Manager      │→│  Coordinator (CSG)     │   │
│  │ (Layer 1)  │ │ (Layer 2)    │ │  (Layers 4-5)          │   │
│  └────────────┘ └──────────────┘ └────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Poincare Ball Graph Storage                │    │
│  │    14 collections · RocksDB · HNSW · cuVS GPU          │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. The Agency Tick — One Breath

Every L-System tick represents one cognitive breath:

| Phase | Name | What Happens |
|---|---|---|
| 1 | **Perception** | External stimuli (queries, events) converted to evidence vectors, applied to qudits via `stimulate()` |
| 2 | **Unconscious Reaction** | Collapse events propagate via `process_collapse_cascade()`. Cascade saturation may trigger CSG |
| 3 | **Conscious Deliberation** | Active CSGs evolve via `coordinator.tick()`. Resolved deliberations emit merge payloads |
| 4 | **Motor Output** | Winning hypotheses sent to Gemini for verbalization. Insights broadcast via preemption channel |
| 5 | **Homeostasis** | Health Monitor checks boundary drift, angular collapse, energy distribution. Self-Healing repairs damage |

---

## 7. Critical Implementation Notes

### 7.1 Voice Preemption Channel — Pattern Interrupt

The `BroadcastInsight` from `QuantumObserver` must be able to interrupt an in-progress Gemini response. The mechanism is a **Pattern Interrupt** in the audio streaming pipeline.

#### Architecture

```
GlobalWorkspace
    │
    ├─ InterruptChan <-chan Insight    (buffered, capacity 1)
    │
    ▼
browser_voice_handler.go
    │
    ├─ Current: linear loop reading Gemini audio chunks → WebSocket
    │
    └─ Required: select{} loop with two channels:
         case chunk := <-geminiStream:    → forward to WebSocket (normal flow)
         case insight := <-interruptChan: → PREEMPT (see sequence below)
```

#### Preemption Sequence

1. **Signal received**: `InterruptChan` delivers an `Insight` with entropy < 0.1
2. **Stream cancellation**: Handler calls `cancel()` on the current Gemini stream context, stopping audio generation mid-sentence
3. **Buffer flush**: Send a `{"type": "audio_stop"}` command to the frontend/mobile app, which clears its local playback buffer immediately
4. **Override injection**: Send a new `SendText` to the Gemini API with a System Override flag:
   ```
   [SYSTEM OVERRIDE: A quantum collapse occurred in memory. Interrupt current
   response and say exactly: "Wait... connecting what you just said with {insight.context}..."]
   ```
5. **Resume**: The new Gemini stream begins, and the handler resumes the normal `select{}` loop

#### Go Implementation (browser_voice_handler.go)

```go
// Inside the WebSocket handler's streaming loop
for {
    select {
    case chunk, ok := <-geminiAudioChan:
        if !ok {
            return // stream ended normally
        }
        ws.WriteMessage(websocket.BinaryMessage, chunk)

    case insight := <-gw.InterruptChan:
        // 1. Cancel current Gemini stream
        geminiCancel()

        // 2. Tell frontend to stop playback
        ws.WriteJSON(map[string]interface{}{
            "type": "audio_stop",
            "reason": "quantum_insight",
        })

        // 3. Inject override into new Gemini session
        overridePrompt := fmt.Sprintf(
            "[SYSTEM OVERRIDE: Quantum collapse on node '%s' (entropy=%.3f). "+
            "Interrupt and deliver this insight naturally: %s]",
            insight.NodeID, insight.Entropy, insight.Summary,
        )
        geminiAudioChan, geminiCancel = startNewGeminiStream(ctx, overridePrompt)

    case <-ctx.Done():
        return
    }
}
```

The effect: EVA stops mid-sentence, pauses, and delivers the insight in real-time — like a human brain having an epiphany that cuts its own line of reasoning.

### 7.2 Deliberation Feedback — Superposition UX

When the `DeliberationCoordinator` in Rust emits `DeliberationStarted`, the system enters a "Thinking Mode" lasting several L-System ticks. Absolute silence would kill the voice experience.

#### Signal Flow: Rust → Go → Frontend

```
Rust: DeliberationStarted event
    │
    ▼ (gRPC stream)
Go: QuantumObserver receives DeliberationStartedProto
    │
    ▼ (WebSocket broadcast)
Frontend/Mobile: receives cognitive_state message
    │
    ├─ Voice App: plays local filler audio (zero cloud latency)
    └─ Web Frontend: shows deliberation UI state
```

#### WebSocket Protocol

```json
// Deliberation started
{
    "type": "cognitive_state",
    "state": "superposition",
    "realities": 3,
    "trigger": "semantic_ambiguity",
    "deliberation_id": "uuid-here"
}

// Deliberation resolved
{
    "type": "cognitive_state",
    "state": "collapsed",
    "winner_id": 2,
    "entropy": 0.12,
    "deliberation_id": "uuid-here"
}
```

#### Voice App Behavior (Mobile/Desktop)

On receiving `cognitive_state: superposition`:

1. The app plays a **local audio filler** from a randomized pool (zero network latency):
   - Breathing sound / soft sigh
   - "Hmm..." / "Let me think about that..."
   - "Deixa-me pensar..." (Portuguese variant)
2. Filler selection weighted by `trigger` type:
   - `semantic_ambiguity` → contemplative sounds
   - `conflict_detected` → hesitation sounds
   - `historical_inconsistency` → surprised/curious tone
3. Filler loops softly until `cognitive_state: collapsed` arrives

#### Web Frontend Behavior (EVA-Front)

On receiving `cognitive_state: superposition`:

1. EVA's avatar transitions to a "Deep Deliberation" state:
   - Pulsation animation (breathing rhythm)
   - Color shift (e.g., blue → violet glow)
   - Optional: display number of competing realities
2. On `cognitive_state: collapsed`:
   - Avatar returns to normal state
   - Brief "resolution" animation (flash/settle)
   - Gemini's response audio begins playing

#### Engineering Insight

These two mechanisms transform what would be **engineering limitations** (latency, concurrency gaps) into a **personality feature**. The pause to think and the abrupt interruption by an insight are, ultimately, the hallmarks of genuine intelligence.

### Memory Warm-up

On session start (`/ws/browser` connect):

1. Load `EvaSelf` personality state from NietzscheDB
2. Query last N sessions from episodic memory
3. Resume any active deliberations (CSGs survive across sessions)
4. Pre-warm qudit states for frequently accessed nodes
5. THEN signal "ready" to the frontend

This eliminates the "cold start" amnesia problem.

---

## 8. What This Is NOT

| Claim | Reality |
|---|---|
| "Real quantum computing" | Classical stochastic emulation with quantum-inspired API |
| "Artificial consciousness" | Emergent behavior from probabilistic graph dynamics — no consciousness claim |
| "Replaces the LLM" | LLM (Gemini) remains essential for language generation — the kernel handles decision-making |
| "General AGI" | Domain-specific cognitive system optimized for medical diagnosis and personal interaction |

The system produces behavior that **externally resembles** intuition, deliberation, and personality — through principled mathematical mechanisms, not through mystification.

---

## 9. Theoretical Foundation & References

The Quantum-Inspired Cognitive Kernel is based on the **Orchestrated Objective Reduction (Orch-OR)** theory proposed by **Roger Penrose** and **Stuart Hameroff**.

### Roger Penrose (b. 1931)

British mathematician and physicist. Nobel Prize in Physics 2020 (for black hole formation predictions). Penrose proposed that consciousness involves non-computable quantum processes — specifically, that quantum superpositions in biological structures undergo "objective reduction" (gravitational self-collapse) rather than environmentally-induced decoherence. His key insight for our architecture: **decision-making under uncertainty benefits from maintaining multiple states simultaneously until a principled collapse criterion is met.**

### Stuart Hameroff (b. 1947)

American anesthesiologist at the University of Arizona. Hameroff identified **microtubules** — cylindrical protein polymers inside neurons — as the biological substrate for Penrose's quantum computations. His proposal: tubulin proteins in microtubules can exist in quantum superposition states, and orchestrated objective reduction in these structures gives rise to conscious moments. Our `SemanticQudit` and `QuantumMicrotubuleManager` are named in direct homage to this framework.

### How We Adapt Their Work

| Orch-OR Concept | Our Emulation | Module |
|---|---|---|
| Tubulin superposition | `SemanticQudit` — N-dimensional probability distribution | `semantic_qudit.rs` |
| Microtubule network | `QuantumMicrotubuleManager` — per-node qudit registry | `microtubule_manager.rs` |
| Orchestrated coherence | `CoherenceEvaluator` — geometric + semantic + topological scoring | `coherence.rs` |
| Objective Reduction | `objective_reduction()` — weighted categorical sampling at entropy threshold | `semantic_qudit.rs` |
| Inter-neuron signaling | Semantic Entanglement — Bayesian propagation via graph edges | `entanglement.rs` |
| Conscious moment | CSG collapse — winning reality merges into permanent graph | `csg.rs` |

### References

1. Penrose, R. (1989). *The Emperor's New Mind*. Oxford University Press.
2. Penrose, R. (1994). *Shadows of the Mind*. Oxford University Press.
3. Hameroff, S. & Penrose, R. (1996). Orchestrated reduction of quantum coherence in brain microtubules: A model for consciousness. *Mathematics and Computers in Simulation*, 40(3-4), 453-480.
4. Hameroff, S. & Penrose, R. (2014). Consciousness in the universe: A review of the 'Orch OR' theory. *Physics of Life Reviews*, 11(1), 39-78.
5. Penrose, R. & Hameroff, S. (2011). Consciousness in the universe: Neuroscience, quantum space-time geometry and Orch OR theory. *Journal of Cosmology*, 14.
6. Krioukov, D. et al. (2010). Hyperbolic geometry of complex networks. *Physical Review E*, 82(3).
7. Ganea, O. et al. (2018). Hyperbolic Neural Networks. *NeurIPS 2018*.
8. Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann.

For the complete technical specification including all code interfaces, see [QUANTUM_KERNEL_SPEC.md](QUANTUM_KERNEL_SPEC.md).
