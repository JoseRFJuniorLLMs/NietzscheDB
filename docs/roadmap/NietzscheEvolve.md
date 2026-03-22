# NietzscheEvolve — AlphaEvolve-Inspired Self-Evolution for NietzscheDB

> Inspirado no AlphaEvolve da DeepMind, aplicado ao NietzscheDB e EVA.
> Em vez de evoluir codigo, evoluimos **conhecimento**, **parametros**, e **estrategias cognitivas**.
>
> **Status**: Implementado (2026-03-19).

---

## Conceito

O AlphaEvolve da DeepMind usa um loop evolutivo:
```
LLM gera candidatos -> avalia automaticamente -> seleciona melhores -> muta/recombina -> repete
```

O NietzscheEvolve aplica este padrao a 5 dimensoes do NietzscheDB:

| Fase | Nome | O que evolui | Ficheiro |
|------|------|-------------|----------|
| 0 | Phase 27 Integration | Epistemic mutations (connect/reclassify/boost) | `engine.rs`, `main.rs` |
| 1 | ParameterEvolve | HNSW params (ef_search, M, ef_construction) | `param_evolve.rs` |
| 2 | EnergyEvolve | Energy decay/boost functions | `energy_evolve.rs` |
| 3 | HypothesisEvolve | LLM system prompts para NietzscheLab | `prompt_evolve.py` |
| 4 | CogEvolve | Cognitive strategy params para EVA | `cog_evolve.rs` |
| 5 | EvoViz | Dashboard de evolucao | `dashboard.rs` |

---

## Fase 0: Phase 27 Integration (CRITICAL FIX)

Phase 27 existia mas **nao estava ligado ao engine.tick()**. Agora:

- `evolution_27_tick` counter no AgencyEngine
- Chamado a cada `evolution_27_interval` ticks (default: 40)
- Converte `EvolutionProposal` -> `AgencyIntent::EpistemicMutation`
- Server handler executa mutations reais:
  - `ProposeEdge` -> insere edge Association
  - `EnergyBoost` -> incrementa energia em +0.05
  - `Reclassify` -> ajusta profundidade do no
  - `PruneEdge` -> remove edge

**Ficheiros**:
- `crates/nietzsche-agency/src/engine.rs` — tick integration
- `crates/nietzsche-server/src/main.rs` — intent handler

---

## Fase 1: ParameterEvolve

Busca evolutiva de parametros HNSW otimos para espaco hiperbolico.

### Genoma
```rust
HnswGenome {
    ef_construction: usize,  // [16..512]
    ef_search: usize,        // [10..1000]
    m: usize,                // [8..64]
}
```

### Fitness
```
fitness = w_recall * recall@k + w_latency * (1 - p95/budget) + w_memory * (1 - mem/budget)
```

### Constraints hiperbolicos
- `ef_construction >= 50` (Poincare precisa de mais qualidade no build)
- `M >= 16` (hierarquias precisam de mais vizinhos na layer 0)
- Distance nao-linear: ef_search deve ser maior que em Euclidiano

### Uso
```rust
use nietzsche_core::param_evolve::{ParamEvolveConfig, run_param_evolution};

let report = run_param_evolution(ParamEvolveConfig::default(), |genome| {
    // benchmark com genome.ef_search, genome.m, etc.
    BenchmarkResult { recall: 0.95, p95_latency_us: 30_000, memory_ratio: 0.3 }
});
println!("Best: ef={} M={}", report.best.ef_search, report.best.m);
```

**Ficheiro**: `crates/nietzsche-core/src/param_evolve.rs`

---

## Fase 2: EnergyEvolve

Evolui as funcoes de decaimento e boost de energia dos nos.

### Genoma
```rust
EnergyGenome {
    decay_rate: f32,           // [0.001..0.1]
    boost_on_access: f32,      // [0.01..0.2]
    hub_bonus_factor: f32,     // [0.0..0.5]
    depth_scaling: f32,        // [0.5..2.0]
    coherence_bonus: f32,      // [0.0..0.3]
    thermal_conductivity: f32, // [0.01..0.2]
}
```

### Funcao de decaimento efetivo
```
effective_decay = decay_rate * depth_scaling^depth * (1 - hub_bonus * log2(degree+1)/5)
```

### Fitness
Otimo: fase liquida (T entre 0.15 e 0.85), ~80% nos vivos, hubs sobrevivem, energia moderada.

### Uso
```rust
use nietzsche_agency::energy_evolve::{EnergyEvolveConfig, SimNode, run_energy_evolution};

let nodes = vec![SimNode { energy: 0.5, depth: 0.3, degree: 5, coherence: 0.7, accessed: false }];
let report = run_energy_evolution(EnergyEvolveConfig::default(), &nodes);
println!("Best decay_rate: {}", report.best.decay_rate);
```

**Ficheiro**: `crates/nietzsche-agency/src/energy_evolve.rs`

---

## Fase 3: HypothesisEvolve

Evolui o **system prompt** do hypothesis_generator.py do NietzscheLab.

### Loop
```
Population de N prompts
For each generation:
  1. Para cada prompt: gerar hipoteses, aplicar, medir delta epistemico
  2. Fitness = acceptance_rate * mean_delta
  3. Top-K sobrevivem
  4. LLM muta prompts: "Rewrite to generate better hypotheses"
  5. Proxima geracao
```

### Auto-referencial
Os prompts evoluidos podem ser guardados como nos no NietzscheDB (collection: `meta_evolution`).
O sistema literalmente **evolui dentro do seu proprio substrato**.

### Uso
```bash
cd nietzsche-lab
python prompt_evolve.py \
  --collection tech_galaxies \
  --generations 5 \
  --population 4 \
  --experiments-per-prompt 3
```

**Ficheiro**: `nietzsche-lab/prompt_evolve.py`

---

## Fase 4: CogEvolve

Evolui os parametros cognitivos que controlam o comportamento da EVA.

### Genoma (10 genes)
```rust
CognitiveGenome {
    consolidation_threshold: f32,  // [0.1..0.9]
    attention_decay: f32,          // [0.01..0.2]
    hebbian_rate: f32,             // [0.01..0.3]
    novelty_weight: f32,           // [0.1..0.9]
    perception_threshold: f32,     // [0.3..0.9]
    dream_interval: u32,           // [10..200]
    thermal_conductivity: f32,     // [0.01..0.2]
    growth_distance: f32,          // [0.5..3.0]
    decay_lambda: f64,             // [0.0000001..0.001]
    cluster_radius: f32,           // [0.1..0.8]
}
```

### Fitness
```
fitness = 0.30 * epistemic_delta
        + 0.20 * utilization_score
        + 0.15 * entropy_score
        + 0.20 * stability_score
        + 0.15 * growth_score
```

### Output
Genoma otimo exportavel como env vars:
```bash
AGENCY_THERMAL_CONDUCTIVITY=0.07
AGENCY_GROWTH_DISTANCE_THRESHOLD=1.2
AGENCY_TEMPORAL_DECAY_LAMBDA=0.0000005
AGENCY_COGNITIVE_CLUSTER_RADIUS=0.25
```

**Ficheiro**: `crates/nietzsche-agency/src/cog_evolve.rs`

---

## Fase 5: EvoViz Dashboard

Endpoint: `GET /api/agency/nietzsche-evolve?collection=X`

Retorna:
```json
{
  "nietzsche_evolve": {
    "phase_27_active": true,
    "evolution_state": { "generation": 7, "fitness_history": [...] },
    "evolved_parameters": {
      "energy_genome": { "decay_rate": 0.02, "hub_bonus_factor": 0.15, ... },
      "cognitive_genome": { "novelty_weight": 0.45, ... },
      "hnsw_genome": { "ef_search": 200, "m": 24, ... }
    },
    "components": [...]
  }
}
```

**Ficheiro**: `crates/nietzsche-server/src/dashboard.rs`

---

## Diferenca face ao AlphaEvolve

| AlphaEvolve (DeepMind) | NietzscheEvolve |
|------------------------|-----------------|
| Evolui **codigo** (algoritmos) | Evolui **conhecimento** + **parametros** + **estrategias** |
| LLM gera mutacoes de codigo | LLM gera hipoteses sobre o grafo + mutacoes de prompts |
| Avalia via testes unitarios | Avalia via metricas epistemicas (5 dimensoes) |
| Optimiza chips/sorting | Optimiza HNSW hiperbolico + funcoes de energia + cognicao |
| Centralizado | Distribuido (Rust server + Python lab + Go EVA) |
| Evolui sobre substrato externo | **Auto-referencial**: evolui dentro do seu proprio grafo |

---

## Env Vars

```bash
# Phase 27 (epistemic evolution)
AGENCY_EVOLUTION_27_ENABLED=true
AGENCY_EVOLUTION_27_INTERVAL=40
AGENCY_EVOLUTION_27_MAX_EVAL=500
AGENCY_EVOLUTION_27_QUALITY_FLOOR=0.4
AGENCY_EVOLUTION_27_MAX_PROPOSALS=5
AGENCY_EVOLUTION_27_MIN_ENERGY=0.05
```

---

## Ficheiros Criados/Modificados

### Novos
| Ficheiro | LOC | Descricao |
|----------|-----|-----------|
| `crates/nietzsche-core/src/param_evolve.rs` | ~320 | ParameterEvolve (HNSW evolution) |
| `crates/nietzsche-agency/src/energy_evolve.rs` | ~340 | EnergyEvolve (energy function evolution) |
| `crates/nietzsche-agency/src/cog_evolve.rs` | ~340 | CogEvolve (cognitive strategy evolution) |
| `nietzsche-lab/prompt_evolve.py` | ~320 | HypothesisEvolve (prompt meta-evolution) |
| `docs/roadmap/NietzscheEvolve.md` | este ficheiro |

### Modificados
| Ficheiro | Mudanca |
|----------|---------|
| `crates/nietzsche-agency/src/engine.rs` | +Phase 27 tick integration (~35 lines) |
| `crates/nietzsche-agency/src/lib.rs` | +module declarations + exports |
| `crates/nietzsche-core/src/lib.rs` | +param_evolve module |
| `crates/nietzsche-server/src/main.rs` | +EpistemicMutation execution handler (~60 lines) |
| `crates/nietzsche-server/src/dashboard.rs` | +nietzsche-evolve endpoint (~50 lines) |
| `README.md` | +NietzscheEvolve features |
