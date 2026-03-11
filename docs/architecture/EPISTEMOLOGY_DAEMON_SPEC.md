# EpistemologyDaemon — Autopoietic Research Engine

> **Module**: `nietzsche-agency` → `daemons/epistemology.rs`
> **Version**: 1.0
> **Date**: 2026-03-10
> **Status**: Daemon Implemented — LLM Bridge Pending
> **Phase**: XXV (Autopoietic Research)

---

## 1. Overview

The EpistemologyDaemon implements **Karpathy's autoresearch loop** mapped to NietzscheDB's hyperbolic geometry. It enables EVA to **rewrite her own cognitive rules while sleeping**, using a Darwinian selection process where mutations are accepted only if they improve the graph's structural health (TGC).

**This is NOT code generation.** The daemon doesn't write Rust. It mutates **NQL queries stored as graph nodes** (Code-as-Data), tests them in a sandboxed ShadowGraph, and accepts only improvements.

---

## 2. Karpathy Mapping

| autoresearch (Karpathy) | EpistemologyDaemon (NietzscheDB) |
|---|---|
| `train.py` | `ActionNode` (Code-as-Data NQL queries) |
| `val_bpb` (Bits Per Byte) | ΔTGC (Topological Generative Capacity) |
| 5-minute sandbox | `ShadowGraph` simulation |
| `git commit` (KEEP) | `AgencyIntent::EpistemologyMerge` |
| `git revert` (DISCARD) | `AgencyIntent::Phantomize` mutant node |
| LLM proposes code edits | LLM proposes NQL rewrites via MCP |

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    L-System Tick                             │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────┐        │
│  │  Phase A: CANDIDATE SELECTION                    │        │
│  │  scan CodeAsData → rank by FlowLedger friction  │        │
│  │  emit EpistemologyCandidate event               │        │
│  └──────────────────────┬──────────────────────────┘        │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────┐        │
│  │  EXTERNAL: LLM MUTATION BRIDGE (MCP/Python)     │        │
│  │  consume candidate → call LLM → write mutation  │        │
│  │  node tagged mutation_pending: true              │        │
│  └──────────────────────┬──────────────────────────┘        │
│                         │ (async, between ticks)            │
│  ┌──────────────────────▼──────────────────────────┐        │
│  │  Phase B: MUTATION EVALUATION                    │        │
│  │  find pending mutations → ShadowGraph sandbox   │        │
│  │  measure ΔTGC → emit EpistemologyVerdict        │        │
│  └──────────────────────┬──────────────────────────┘        │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────┐        │
│  │  REACTOR: Execute Verdict                        │        │
│  │  KEEP → EpistemologyMerge (swap NQL)            │        │
│  │  DISCARD → Phantomize mutant node               │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Implemented Components

### 4.1 Daemon (Rust)

**File**: `crates/nietzsche-agency/src/daemons/epistemology.rs`

```rust
pub struct EpistemologyDaemon {
    rejected_hashes: Mutex<HashSet<u64>>,    // avoid retrying failed mutations
    flow_ledger: Option<Arc<FlowLedger>>,    // friction-based ranking
}
```

Implements `AgencyDaemon` trait (read-only, never mutates graph).

### 4.2 Events

Added to `AgencyEvent` enum:

| Event | Direction | Purpose |
|---|---|---|
| `EpistemologyCandidate` | Daemon → MCP Bridge | "This NQL node needs mutation" |
| `EpistemologyVerdict` | Daemon → Reactor | "This mutation is KEEP/DISCARD" |

### 4.3 Intent

Added to `AgencyIntent` enum:

| Intent | Trigger | Server Action |
|---|---|---|
| `EpistemologyMerge { original_id, mutant_id }` | Verdict = KEEP | Copy mutant NQL → original, phantomize mutant |

### 4.4 Config

| Env Variable | Default | Description |
|---|---|---|
| `AGENCY_EPISTEMOLOGY_ENABLED` | `false` | Opt-in (disabled by default) |
| `AGENCY_EPISTEMOLOGY_MAX_SCAN` | `500` | Max CodeAsData nodes scanned per tick |
| `AGENCY_EPISTEMOLOGY_THRESHOLD` | `0.01` | Minimum ΔTGC to accept a mutation |

---

## 5. LLM Mutation Bridge (TODO — Next Phase)

The bridge is the **external process** that converts an `EpistemologyCandidate` event into a `mutation_pending` node. It runs outside the Rust daemon (MCP server, Python script, or Go service).

### 5.1 Mutation Node Format

The LLM bridge must write a node with this exact JSON structure:

```json
{
  "mutation_pending": true,
  "mutation_parent": "<UUID of original ActionNode>",
  "action": {
    "nql": "<mutated NQL query>",
    "activation_threshold": 0.5,
    "cooldown_ticks": 0,
    "max_firings": 0,
    "firings": 0,
    "cooldown_remaining": 0,
    "description": "LLM-mutated version of <original description>"
  }
}
```

### 5.2 LLM Prompt Template

```
You are a NietzscheDB NQL optimizer. You receive a query that runs inside
a hyperbolic semantic graph database. Your goal is to rewrite the query
to improve one or more of:

1. **Thermodynamic efficiency**: Reduce traversal cost (fewer hops, tighter filters)
2. **Structural health**: Improve TGC (prefer queries that create/strengthen
   edges at appropriate Poincaré depths rather than creating orphans)
3. **Precision**: Tighter WHERE clauses that avoid false positives

## Current Query
```nql
{current_nql}
```

## Context
- This action has fired {firings} times
- Current CPU cost per execution: {cpu_cost_ns} ns
- Graph has {total_nodes} active nodes, {total_edges} edges
- Average local degree: {avg_degree}
- Current TGC EMA: {tgc_ema}

## NQL Reference
- MATCH (n:Type) WHERE field op value
- POINCARE_DIST(a, b) < threshold
- DIFFUSE FROM seed RADIUS r TIME t [USE_CONDUCTIVITY]
- SET field = value
- Edge types: Association, Hierarchical, Causal, Temporal, Similarity

## Constraints
- Output ONLY the rewritten NQL query, no explanation
- The query must be syntactically valid NQL
- Do NOT change the semantic intent (what the query does)
- Optimize the HOW, not the WHAT
```

### 5.3 Implementation Options

**Option A: Python MCP Client (Recommended for MVP)**

```python
# epistemology_bridge.py
# Subscribes to gRPC QuantumEventStream for EpistemologyCandidate events
# Calls Gemini/Claude to generate mutation
# Writes mutation node via NietzscheDB gRPC InsertNode

async def on_candidate(event):
    prompt = MUTATION_PROMPT.format(
        current_nql=event.nql,
        firings=event.friction,
        ...
    )
    mutant_nql = await llm.generate(prompt)

    # Validate syntax before writing
    if not validate_nql(mutant_nql):
        return  # silent discard

    await nietzsche_client.insert_node(
        content={
            "mutation_pending": True,
            "mutation_parent": str(event.node_id),
            "action": {"nql": mutant_nql, ...}
        },
        node_type="Concept",
        embedding=zero_vector(128),  # placeholder
    )
```

**Option B: Go Integration (EVA-X)**

Add a `QuantumObserver` handler in EVA-X that consumes `EpistemologyCandidate` from the gRPC stream, calls Gemini, and writes back via the NietzscheDB gRPC API.

**Option C: MCP Tool (nietzsche-mcp)**

Add a `force_evolution` tool to `nietzsche-mcp/src/tools.rs` that accepts a node ID, generates the mutation locally (requires LLM client in Rust), and writes the mutation node.

---

## 6. SleepCycle Integration (TODO — Phase 4)

The daemon should run primarily during the `SleepCycle`:

1. Modify `nietzsche-sleep/src/cycle.rs` to allocate 20% of sleep time for epistemology
2. The `SleepConfig` gains an `epistemology_budget` field (number of mutations to attempt per sleep)
3. After sleep reconsolidation, the daemon produces a "Morning Report" summarizing overnight discoveries

### Morning Report Format

```
[EpistemologyDaemon] Sleep cycle completed:
  Candidates evaluated: 12
  Mutations accepted:   3
  Mutations rejected:   9
  Best improvement:     ΔTGC = +0.047 (action node 8f3a...)
    Original: MATCH (n) WHERE n.energy < 0.1 SET n.energy = 0.0
    Mutated:  MATCH (n) WHERE n.energy < 0.1 AND POINCARE_DIST(n, origin) > 0.8 SET n.energy = 0.0
    Rationale: Added depth filter — only drain dying nodes in periphery, preserving core
```

---

## 7. Safety Guarantees

| Mechanism | Guarantee |
|---|---|
| Read-only daemon | Never mutates graph directly |
| ShadowGraph sandbox | Simulation uses metadata-only copy |
| Rejected hash set | Same mutation never retried |
| `mutable: false` flag | ActionNodes can opt out of mutation |
| Improvement threshold | Only positive ΔTGC accepted (configurable) |
| `epistemology_enabled: false` | Disabled by default (opt-in) |
| Phantomize on reject | Failed mutations become ghosts (auto-cleaned) |

---

## 8. File Layout

```
crates/nietzsche-agency/src/
├── daemons/
│   ├── mod.rs                    // +pub mod epistemology; +pub use
│   └── epistemology.rs           // EpistemologyDaemon (Phase A + B)
├── event_bus.rs                  // +EpistemologyCandidate, +EpistemologyVerdict
├── reactor.rs                    // +EpistemologyMerge intent, +event handling
└── config.rs                     // +epistemology_enabled/max_scan/threshold
```

---

## 9. What's Next

| Priority | Task | Effort |
|---|---|---|
| **P0** | LLM Mutation Bridge (Python MCP client) | 2-3 hours |
| **P1** | NQL syntax validator (pre-simulation gate) | 1 hour |
| **P2** | Full TGC measurement in ShadowGraph (not just proxy) | 4 hours |
| **P3** | SleepCycle integration (20% budget allocation) | 2 hours |
| **P4** | Morning Report via NarrativeEngine | 1 hour |
| **P5** | FlowLedger per-action friction tracking | 3 hours |
