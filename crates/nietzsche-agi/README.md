# nietzsche-agi

**AGI inference layer for NietzscheDB** — explicit reasoning with verifiable trajectories over non-Euclidean geometry.

This crate sits on top of the multi-manifold geometry engine (`nietzsche-hyp-ops`) and the graph engine (`nietzsche-graph`) to add a **consciousness layer**: geodesic computation, GCS validation, formal inference, dialectical synthesis, metabolic knowledge gating, and autonomous evolution.

## Architecture (6 Layers)

```
Layer 6 — Metabolic Equilibrium (Phase VI)
│  discovery.rs      D(τ) = w_g·|∇E| + w_c·θ_cluster
│  innovation.rs     Φ(τ) = αS + βD - γR  →  Accept / Sandbox / Reject
│  sandbox.rs        Quarantine lifecycle with Δλ₂ promotion
│
Layer 5 — Stability Motor (Phase V)
│  stability.rs      E(τ) continuous energy function (4 geometry components)
│  certification.rs  4-tier epistemological seal (Stable → Rupture)
│  spectral.rs       λ₂ Fiedler eigenvalue + DriftTracker
│
Layer 4 — Dynamic Update
│  feedback_loop.rs  Off-graph simulation + re-insertion
│  homeostasis.rs    Radial repulsion/attraction field
│  relevance_decay.rs  Frequency-based weight adjustment
│  evolution.rs      Autonomous heartbeat orchestrator
│
Layer 3 — Explicit Inference
│  inference_engine.rs  Rule engine: trajectory → classification
│  synthesis.rs         Fréchet mean on Riemann sphere
│  dialectic.rs         Cross-cluster tension detection
│
Layer 2 — Verifiable Semantic Navigation
│  trajectory.rs     GeodesicTrajectory + GCS validation
│
Layer 1 — Representation & Storage
   representation.rs  SynthesisNode wrapper
   rationale.rs       Proof object + InferenceType + energy seal
```

## Key Concepts

### Geodesic Coherence Score (GCS)

Every hop in a trajectory is validated for geometric coherence:
- **Collinearity** (Klein model): are intermediate points on the geodesic?
- **Radial gradient** (Poincare ball): does depth increase monotonically?
- **GCS = H(collinearity, gradient)** — harmonic mean, range [0, 1]

### Energy Function E(τ)

Continuous quality metric for trajectories:
```
E(τ) = w₁·H_GCS + w₂·(1 - θ_klein/π) + w₃·causal_fraction + w₄·(1 - H(τ))
```
- **H_GCS**: harmonic mean of per-hop GCS scores
- **θ_klein**: angular deviation in Klein disk
- **causal_fraction**: ratio of timelike (Minkowski) hops
- **H(τ)**: trajectory entropy (Shannon)

### Acceptance Function Φ(τ)

Metabolic gate deciding the fate of every inference:
```
Φ(τ) = α·S + β·D - γ·R    (α=0.50, β=0.35, γ=0.60)
```
| Decision | Condition | Action |
|----------|-----------|--------|
| **Accept** | Φ ≥ 0.50 | Direct insertion with full weight |
| **Sandbox** | 0.20 ≤ Φ < 0.50 | Quarantine for spectral testing |
| **Reject** | Φ < 0.20 | Discarded — not worth the risk |

### Sandbox Quarantine

Sandboxed inferences enter a controlled lifecycle:
1. Insert with **0.3x weight** + **3x accelerated decay**
2. Monitor **Δλ₂** (Fiedler eigenvalue drift) for N cycles
3. **Promote** if stable (drift < threshold) or **Reject** if catastrophic

### Spectral Monitoring (λ₂)

Algebraic connectivity via the graph Laplacian's second-smallest eigenvalue:
- **Rigid** (λ₂ > 1.0): well-connected, resistant to fragmentation
- **Stable** (0.1 < λ₂ ≤ 1.0): healthy connectivity
- **Fragile** (0 < λ₂ ≤ 0.1): at risk of splitting
- **Disconnected** (λ₂ ≈ 0): graph has isolated components

The **DriftTracker** monitors λ₂ evolution over time for the Evolutionary Model (controlled drift rather than fixed λ₂).

## Design Principles

1. **Every inference carries a Rationale** — no black-box reasoning
2. **GCS validates every hop** — broken geodesics are rejected
3. **Synthesis preserves manifold structure** — Frechet mean, not Euclidean average
4. **Homeostasis prevents collapse** — radial repulsion near the origin
5. **All operations use f64 internally** — promoted from f32 storage
6. **Every rationale carries an energy seal** — E(τ) certifies quality
7. **Spectral health monitors structural integrity** — λ₂ detects fragmentation
8. **Innovation is metabolized, not suppressed** — Φ(τ) balances stability vs discovery
9. **Sandbox quarantine tests before committing** — no untested knowledge enters the manifold

## Stats

| Metric | Value |
|--------|-------|
| Source files | 17 |
| Lines of code | ~7,000 |
| Unit tests | 123 |
| Test coverage | 100% modules |
| Dependencies | nietzsche-graph, nietzsche-hyp-ops, serde, uuid, tracing |

## Module Map

| Module | Lines | Tests | Purpose |
|--------|-------|-------|---------|
| `trajectory.rs` | 404 | 5 | GCS validation pipeline |
| `rationale.rs` | 459 | 8 | Proof objects + energy seals |
| `representation.rs` | 230 | 2 | SynthesisNode ↔ graph bridge |
| `inference_engine.rs` | 325 | 6 | Rule-based trajectory classification |
| `synthesis.rs` | 329 | 3 | Frechet mean dialectical synthesis |
| `dialectic.rs` | 275 | 3 | Cross-cluster tension detection |
| `feedback_loop.rs` | 488 | 5 | Simulation + graph re-insertion |
| `homeostasis.rs` | 393 | 12 | Radial field + origin guard |
| `relevance_decay.rs` | 236 | 7 | Frequency-based decay + boost |
| `evolution.rs` | 262 | 3 | Autonomous scheduler |
| `stability.rs` | 551 | 8 | E(τ) energy function |
| `certification.rs` | 349 | 11 | 4-tier epistemological seal |
| `spectral.rs` | 817 | 16 | λ₂ monitoring + DriftTracker |
| `discovery.rs` | 413 | 9 | Discovery field D(τ) |
| `innovation.rs` | 389 | 12 | Acceptance function Φ(τ) |
| `sandbox.rs` | 550 | 11 | Quarantine lifecycle |
| `error.rs` | 50 | — | Error types |

## Example

See [`examples/pipeline_ignition.rs`](examples/pipeline_ignition.rs) for a full AGI inference pipeline demo:

```bash
cargo run -p nietzsche-agi --example pipeline_ignition
```

## License

Same as NietzscheDB — AGPL-3.0
