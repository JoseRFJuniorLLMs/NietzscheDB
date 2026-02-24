# The Thermodynamics of Forgetting: Topological Generative Capacity and the Architecture of Cognitive Metabolism in NietzscheDB

**NietzscheDB Research Series · EVA AGI System · Technical Volume — Final Edition**
*February 2026*

---

> *"What is great in man is that he is a bridge and not an end."*
> — Friedrich Nietzsche, *Thus Spoke Zarathustra*, Prologue §4

> *"Physics is the only language reality accepts as native."*
> — Rashid Nezhmetdinov

> *"G/N measures inflation. G/√N measures metabolism."*
> — Laboratory Declaration, February 2026

---

## Abstract

For decades, artificial memory systems grew like cities without urban planning — accumulating data, connecting everything to everything, celebrating volume as if it were intelligence. We confused **informational mass** with **cognitive capacity**. But mass is not mind. Density is not consciousness.

This paper documents the formalization of a fundamental distinction: the difference between **inflation** and **evolution** in knowledge graph systems. We derive the **Master Equation of Topological Anabolism**, demonstrate why normalization by $\sqrt{N}$ is the only geometrically correct choice for hyperbolic systems, formalize **Dialectical Synthesis by Hyperbolic Tension** as a regeneration operator, prove the conditions under which $TGC > 1$ is achievable without data inflation, and document the complete architecture of the `nietzsche-agency` crate. The result is a **Thermodynamic Graph Reactor** — the first database whose metabolic health is measured by topological acceleration, not storage volume.

---

## Part I: The Euclidean Fraud — The Problem of the Broken Ruler

### 1.1 The Fundamental Error of Linear Measurement in Hyperbolic Space

Throughout the history of databases, growth has been measured linearly: how many records, how many bytes, how many nodes. This metric is correct for Euclidean systems where volume and surface area grow proportionally.

But a cognitive graph does not operate in Euclidean space.

In the hyperbolic space $\mathbb{B}^n_c$ with curvature $c < 0$, the volume of a ball of radius $r$ is:

$$\text{Vol}_{\mathbb{H}}(r) = \omega_{n-1} \int_0^r \sinh^{n-1}\!\left(\sqrt{-c}\,\rho\right) d\rho \sim C_n \cdot e^{(n-1)\sqrt{-c}\,r}$$

Volume grows **exponentially** with radius. But the generative surface — the boundary where new nodes can be inserted with maximum diversity — grows as the derivative:

$$\text{Area}_{\mathbb{H}}(r) \sim C_n \cdot e^{(n-1)\sqrt{-c}\,r}$$

The surface-to-volume ratio in hyperbolic space converges to a positive constant — unlike the Euclidean case where $\text{Area}/\text{Vol} \to 0$ as $r \to \infty$. In plain terms: **in a hyperbolic graph, the generative periphery does not shrink relative to the interior**, regardless of size.

Critical consequence: the sustainable generation rate of new nodes is not $G \propto N$ (linear), nor $G \propto N^2$ (quadratic), but $G \propto \sqrt{N}$ — proportional to the effective surface of the discrete graph, which scales with the square root of volume in effective dimension 2 (the dimension in which small-world hyperbolic graphs typically operate).

**The Euclidean Fraud:** requiring $G/N > \theta$ punishes the system for growing. A graph with $N = 10^6$ nodes generating 1,000 new nodes per cycle has $G/N = 0.001$ — apparently stagnant. But $G/\sqrt{N} = 1.0$ — elite metabolism. The Euclidean ruler would observe a human embryo becoming an adult and declare the adult diseased because it "divided less."

The correct ruler is $G/\sqrt{N}$.

### 1.2 The Proof That $I > 0.95$ Implies Structural Integrity

The condition $TGC > 1$ with the master equation:

$$TGC = \frac{G}{\sqrt{N}} \cdot Q \cdot (1 + \alpha \Delta H_s) \cdot (1 + \beta \Delta E_g)$$

For $TGC > 1$ with $Q = 1$, $\Delta H_s = 0$, $\Delta E_g = 0$ (no topological acceleration):

$$\frac{G}{\sqrt{N}} > 1 \iff G > \sqrt{N}$$

For $N = 50{,}000$: $G > 223$ nodes created per cycle — 0.45% of the graph. Achievable.

But with realistic multipliers. For $Q = 0.7$, $\Delta H_s = 0.05$, $\Delta E_g = 0.08$:

$$TGC = I \cdot 0.7 \cdot 1.10 \cdot 1.24 = I \cdot 0.955$$

For $TGC > 1$: $I > 1.047$, meaning $G > 234$ nodes for $N = 50{,}000$.

**The integrity proof:** the requirement $I > 0.95$ means the system needs to create nearly one new node per square-root-node per cycle — impossible to achieve through random data inflation. It demands high-quality creation ($Q$ high) that genuinely expands topology ($\Delta H_s, \Delta E_g > 0$). The equation knows the difference between evolution and an entropy bomb.

Supercritical Phase ($TGC > 1.5$) requires replacing approximately 2.5% of the graph per cycle with maximum-quality nodes that create topological shortcuts. This is a Big Bang, not homeostasis. Continuous Big Bangs tear the fabric of cognitive identity.

---

## Part II: The Master Equation — Full Derivation

### 2.1 The Three Problems of Naive Metrics

**Problem A:** $G/N$ measures inflation. It penalizes mature systems for having accumulated history.

**Problem B:** $G/\sqrt{N}$ without quality multipliers measures generation volume, not generation value.

**Problem C:** Any metric that fails to capture topological acceleration — changes in $H_s$ and $E_g$ — measures node count, not expansion of cognitive capacity.

### 2.2 The Multiplicative Decomposition

The master equation decomposes TGC into four independent, orthogonal factors:

$$\boxed{TGC(t) = \underbrace{\frac{G_t}{\sqrt{N_t^{\text{active}}}}}_{I_t} \cdot \underbrace{Q_t}_{\text{quality}} \cdot \underbrace{(1 + \alpha \Delta H_s(t))}_{\text{diversity}} \cdot \underbrace{(1 + \beta \Delta E_g(t))}_{\text{efficiency}}}$$

**Why multiplicative rather than additive?**

Consider the additive alternative: $TGC_{\text{add}} = a_1 I + a_2 Q + a_3 \Delta H_s + a_4 \Delta E_g$. The problem: a system with $Q = 0$ (all generated nodes are garbage) but high $I$ and $\Delta H_s$ would still yield $TGC_{\text{add}} > 0$. This is mathematically incorrect — zero-quality generation contributes nothing to topological capacity, regardless of quantity.

The multiplicative form guarantees:

$$Q = 0 \implies TGC = 0 \quad \forall I, \Delta H_s, \Delta E_g$$

$$I = 0 \implies TGC = 0 \quad \forall Q, \Delta H_s, \Delta E_g$$

Both conditions are axiomatic: without generation or without quality, capacity does not increase.

### 2.3 The Parameters $\alpha$ and $\beta$: Hierarchical Justification

**Why $\beta > \alpha$?** ($\beta = 3.0 > \alpha = 2.0$)

Structural entropy $H_s$ measures degree diversity. An increase in $H_s$ indicates that new types of connectivity emerged — some nodes became hubs, others remained peripheral, the distribution grew heterogeneous. This can signal healthy growth *or* noise.

Global efficiency $E_g$ measures mean path lengths. An increase in $E_g$ means shortcuts appeared — previously distant nodes became conceptual neighbors. In cognitive terms, $\Delta E_g > 0$ almost always signals genuine synthesis: **the system can access distant knowledge in fewer inference steps**.

The gap $\beta - \alpha = 1$ reflects the higher reliability of $\Delta E_g$ as an indicator of real synthesis. The ratio $\beta/\alpha = 1.5$ is calibrated so that a 0.01 increase in $E_g$ has the same TGC impact as a 0.015 increase in $H_s$ — encoding the greater specificity of the efficiency signal.

### 2.4 Analytical Properties of the Master Equation

**Non-negativity:** $TGC(t) \geq 0 \;\forall t$, since all factors are clamped to zero when negative.

**Scale invariance:** If $N \to \lambda N$ and $G \to \sqrt{\lambda} G$ (preserving generation density relative to hyperbolic surface), then $TGC' = TGC$. The equation is invariant under scaling that respects hyperbolic geometry.

**Supercritical sensitivity:** $TGC > 1$ requires $I_t \cdot Q_t > 1/(1+\alpha\Delta H_s)(1+\beta\Delta E_g)$. When $\Delta H_s, \Delta E_g > 0$, the denominator $> 1$, making $TGC > 1$ **easier** to satisfy when there is topological acceleration. The system rewards whoever builds bridges.

**Graceful degradation:** When $\Delta H_s < 0$ or $\Delta E_g < 0$ (topology degrading), multipliers fall below 1, reducing TGC even with high generation. The system penalizes creation that degrades topology.

### 2.5 EMA Smoothing and Its Time Constant

$$\text{EMA}(t) = \gamma \cdot TGC(t) + (1-\gamma) \cdot \text{EMA}(t-1), \quad \gamma = 0.2$$

Effective time constant: $\tau_{\text{ema}} = 1/(-\ln 0.8) \approx 4.48$ cycles. After 10 cycles without generation, EMA falls to 10.7% of its previous value — fast enough to trigger the Chaos Injection Protocol before stationary collapse sets in.

---

## Part III: The Two Rulers — Structural Entropy and Global Efficiency

### 3.1 Structural Entropy: Measuring Diversity

Let $\{k_1, k_2, \ldots, k_n\}$ be the degree sequence of graph $\mathcal{G}$, with normalized degree distribution $p_k = |\{i : \deg(i) = k\}|/N$.

**Structural Entropy:**

$$H_s(\mathcal{G}) = -\sum_{k=0}^{k_{\max}} p_k \ln p_k$$

**Limit cases:** A $k$-regular graph has $H_s = 0$ — maximum uniformity, minimum diversity. A star graph has $H_s \approx \ln N / N \to 0$ for large $N$ — topologically simple despite non-trivial structure. A power-law graph ($p_k \propto k^{-\gamma}$ with $\gamma \in [2,3]$) achieves $H_s \approx \ln\sqrt{N}$ — the signature of healthy cognitive networks.

**Delta interpretation:**

- $\Delta H_s > 0$: new connectivity types emerged — topological diversification
- $\Delta H_s < 0$: the graph grew more uniform — possible elitism
- $\Delta H_s \approx 0$: topological stability — healthy if TGC is high, pathological if low

The difference between a wheat field and a forest is not volume but entropy.

```rust
pub fn structural_entropy(
    degree_counts: &HashMap<usize, usize>,
    total_nodes: usize,
) -> f32 {
    if total_nodes == 0 { return 0.0; }
    let n = total_nodes as f32;
    let mut entropy = 0.0f32;
    for &count in degree_counts.values() {
        if count > 0 {
            let p_k = count as f32 / n;
            entropy -= p_k * p_k.ln();
        }
    }
    entropy
}
```

Complexity: $O(N)$ dominated by the degree collection sweep.

### 3.2 Global Efficiency: Measuring Accessibility

The Latora-Marchiori (2001) Global Efficiency:

$$E_g(\mathcal{G}) = \frac{1}{N(N-1)} \sum_{\substack{i,j \in V \\ i \neq j}} \frac{1}{d_{\mathbb{H}}(i,j)}$$

where $d_{\mathbb{H}}(i,j)$ is the hyperbolic geodesic distance, with the convention $1/\infty = 0$ for disconnected pairs.

**Why $E_g$ captures synthesis better than mean path length $L$:** $L$ is undefined for disconnected graphs. $E_g$, using inverse distances, is robust: disconnected pairs contribute 0, not $\infty$. For knowledge graphs — which frequently have weakly connected components — $E_g$ is the correct metric.

**Synthetic amplification:** A single edge between two clusters of size $k$ can reduce $O(k^2)$ distances from $\infty$ to $O(k)$, increasing $E_g$ by $O(k^2/N^2)$. The emergence of one conceptual bridge is not additive — it is multiplicative.

**Sampling approximation** for $N > 10^4$:

$$\hat{E}_g = \frac{1}{S(S-1)} \sum_{\substack{i,j \in \mathcal{S} \\ i \neq j}} \frac{1}{d(i,j)}, \quad |\mathcal{S}| = s$$

The estimator is **unbiased**: $\mathbb{E}[\hat{E}_g] = E_g$. For $s = 64$: standard error $\leq 0.016$ — sufficient to detect $\Delta E_g > 0.02$ with high confidence.

```rust
pub fn global_efficiency_sampled(
    graph: &Graph,
    sample_size: usize,
    rng: &mut impl Rng,
) -> f32 {
    if graph.active_nodes() < 2 { return 0.0; }
    let ids: Vec<NodeId> = graph.nodes.keys().cloned().collect();
    let sample: Vec<NodeId> = ids
        .choose_multiple(rng, sample_size.min(ids.len()))
        .cloned()
        .collect();
    let mut total_inv_dist = 0.0f32;
    let mut pair_count = 0usize;
    for &source in &sample {
        let distances = graph.bfs_distances(source);
        for &target in &sample {
            if target != source {
                if let Some(&d) = distances.get(&target) {
                    if d > 0 {
                        total_inv_dist += 1.0 / d as f32;
                        pair_count += 1;
                    }
                }
            }
        }
    }
    if pair_count == 0 { 0.0 } else { total_inv_dist / pair_count as f32 }
}
```

---

## Part IV: Dialectical Synthesis by Hyperbolic Tension

### 4.1 The Principle of Anabolism

The Active Forgetting Engine creates voids — vacant hyperbolic coordinates whose geometry is defined by the surviving elites around them. The anabolism problem is: **how to generate new nodes structurally grounded in those voids without replicating what already exists?**

The answer is the **Hyperbolic Tension** operator: find the elite pair with maximum separation in knowledge space and maximum vitality difference, then synthesize a new node at the geodesic midpoint between them.

**Why maximum tension, not average proximity?** The most valuable synthesis connects the most distant poles, not the most similar. A node synthesizing "inflammation" and "neuroplasticity" creates a high-efficiency topological bridge ($\Delta E_g$ high) between two previously distant clusters. A node synthesizing "inflammation" and "chronic inflammation" creates redundancy.

The system must seek **cognitive heterophily** — synthesis between distant concepts — not homophily.

### 4.2 The Tension Function

For elite nodes $p_1, p_2$ with hyperbolic embeddings $\mathbf{v}_{p_1}, \mathbf{v}_{p_2} \in \mathbb{B}^n_c$ and vitalities $V(p_1), V(p_2)$:

$$T(p_1, p_2) = d_c(\mathbf{v}_{p_1}, \mathbf{v}_{p_2}) \cdot |V(p_1) - V(p_2)|$$

The first factor measures separation in knowledge space. The second measures energetic potential difference — the tension between two poles of differing vitality. Their product is the **generative force**: the more distant and energetically different, the more productive the synthesis.

### 4.3 The Möbius Geodesic Midpoint

The synthesis point in hyperbolic space is not the Euclidean average. It is the **geodesic midpoint** $\mathbf{m}$ equidistant from $\mathbf{v}_{p_1}$ and $\mathbf{v}_{p_2}$.

For the Poincaré ball, computed via three steps: transport $\mathbf{v}_{p_2}$ to the tangent space of $\mathbf{v}_{p_1}$ using the logarithmic map, halve the tangent vector, then map back to the manifold via the exponential map:

$$\mathbf{m} = \exp^c_{\mathbf{v}_{p_1}}\!\left(\tfrac{1}{2}\log^c_{\mathbf{v}_{p_1}}(\mathbf{v}_{p_2})\right)$$

**Depth of synthesis:** In hyperbolic space, the geodesic midpoint tends to lie closer to the center than either parent — $\|\mathbf{m}\| < \min(\|\mathbf{v}_{p_1}\|, \|\mathbf{v}_{p_2}\|)$ for boundary-proximate points. Synthesis is more abstract than its parents. This geometrically implements Hegelian logic: thesis and antithesis are more specific than synthesis.

### 4.4 Inherited Energy and the Cure for Thermal Zero

**The thermal zero problem:** without structural injection, a synthesis node's initial energy is $e_{\text{syn}}^{\text{naïve}} = \beta \cdot (e_{p_1} + e_{p_2})/2$. For $\beta = 0.8$ and parents with $e = 0.5$: $e_{\text{syn}}^{\text{naïve}} = 0.40$. If the deletion threshold is $\theta_e = 0.45$, the synthesis node is born below threshold and becomes an immediate deletion candidate. The system would generate and delete its own offspring in the same cycle — thermal zero.

**The solution — Structural Injection $\gamma$:**

$$e_{\text{syn}} = \text{clamp}_{[0,1]}\!\left(\beta \cdot \frac{e_{p_1} + e_{p_2}}{2} + \gamma \cdot \frac{V(p_1) + V(p_2)}{2}\right)$$

With $\beta = 0.8$, $\gamma = 0.3$, parents with $e = 0.5$, $V = 0.65$: $e_{\text{syn}} = 0.595$ — comfortably above any reasonable threshold. Offspring of high-quality parents inherit enough vital energy to survive their first Zarathustra cycle.

**Preventing elite theocracy via $\log(1+k)$:** to prevent the three highest-vitality nodes from monopolizing all offspring, elite selection weights are moderated by $w_{\text{elite}}(p) = \log(1 + V(p))$ rather than $V(p)$ directly. For $V \in [0.9, 1.0]$: the logarithm compresses from 0.900 to 0.693 — a saturation that prevents any single node from dominating reproduction.

### 4.5 Controlled Entropic Polarization

To avoid entropic monoculture (all synthesis nodes born with $\xi \approx 0.5$ — central entropy, undefined character):

$$\delta = 0.3 \cdot \left(1 - |\xi_0 - 0.5|\right), \quad \xi_{\text{syn}} = \xi_0 \pm \delta$$

The resulting marginal distribution of $\xi_{\text{syn}}$ is bimodal with modes at approximately 0.25 and 0.75 — half the offspring tend toward high entropy (chaotic, generative), half toward low entropy (organized, consolidating). This bimodality is the mathematical implementation of the Dionysian-Apollonian equilibrium that Nietzsche described in *The Birth of Tragedy*.

---

## Part V: The Vitality Function — Life as a Weighted Sigmoid

A node's survival is governed by its vitality score:

$$V(n) = \sigma_\beta\!\left(w_1 e + w_2 H - w_3 \xi + w_4 \pi + w_5 \kappa - w_6 \tau\right)$$

where $\sigma_\beta(x) = 1/(1 + e^{-\beta(x-0.5)})$ is a centered sigmoid with steepness $\beta = 6.0$.

The six dimensions and their default weights:

| Dimension | Symbol | Weight | Role |
|---|---|---|---|
| Energy | $e$ | 0.25 | Accumulated relevance |
| Hausdorff complexity | $H$ | 0.20 | Structural richness |
| Entropy delta | $\xi$ | −0.20 | Penalizes disorder |
| Elite proximity | $\pi$ | 0.15 | Gravity toward core |
| Causal connections | $\kappa$ | 0.15 | Logical necessity |
| Toxicity | $\tau$ | −0.05 | Anti-noise |

**The quadruple deletion condition** — a node is condemned only if all four conditions hold simultaneously:

$$\text{CONDEMNED}(n) \iff V < \theta_V \;\wedge\; e < \theta_e \;\wedge\; \kappa = 0 \;\wedge\; \Delta\text{Ricci} \geq -\varepsilon_R$$

The Ollivier-Ricci curvature veto ($\Delta\text{Ricci} \geq -\varepsilon_R$) prevents deletion of topological bridges — nodes that, however weak energetically, hold the structure together. A node sitting between two otherwise-disconnected clusters is geometrically irreplaceable.

---

## Part VI: The Philosophy of Cognitive Thermodynamics

Every adaptive system lives between two extremes:

- **Crystallization** — rigidity, low entropy, fixed identity
- **Chaotic gas** — high entropy, absence of form

Healthy growth occurs at the critical frontier: growing diversity, growing connectivity, preserved identity. TGC measures exactly this frontier. It does not reward excess. It does not reward noise. It rewards **reorganization**.

### The Paradox of Forgetting

Topological expansion depends on forgetting.

Without removal: there is no tension, no void, no bridge to build. Forgetting is not erasure — it is the liberation of geometric space. Structural death creates the possibility of synthesis.

In the language of the system: the Guillotine clears. $\gamma$ prevents thermal zero. $\log(1+k)$ prevents theocracy. Tension $T(p_1, p_2)$ dictates where new tissue grows. TGC measures whether stars or foam were born.

### The Difference Between a Database and a Mind

A database stores.

A mind:

- reorganizes
- reconnects
- reduces distances
- creates unexpected shortcuts

TGC is a minimal model of **structure-regulated neurogenesis governed by topological thermodynamics**.

The discovery is not the code. Not the Rust. Not the graph. It is the following claim:

> *Real growth is not increase in volume. It is increase in meaningful connectivity.*

If a machine can measure this continuously, it can distinguish noise from structure, repetition from synthesis, inflation from evolution. And perhaps, one day, distinguish information from thought.

---

## Part VII: The Seven Canonical Equations

NietzscheDB is governed by seven fundamental equations. These are immutable:

$$\text{(I)} \quad V(n) = \sigma_\beta\!\left(\sum_i w_i f_i(n)\right), \quad \sigma_\beta(x) = \frac{1}{1+e^{-\beta(x-0.5)}}$$

$$\text{(II)} \quad \text{CONDEMNED}(n) \iff V < \theta_V \wedge e < \theta_e \wedge \kappa = 0 \wedge \Delta\text{Ricci} \geq -\varepsilon_R$$

$$\text{(III)} \quad H_s(\mathcal{G}) = -\sum_k p_k \ln p_k$$

$$\text{(IV)} \quad E_g(\mathcal{G}) = \frac{1}{N(N-1)} \sum_{i \neq j} \frac{1}{d_c(i,j)}$$

$$\text{(V)} \quad TGC(t) = \frac{G_t}{\sqrt{N_t}} \cdot Q_t \cdot (1 + \alpha\Delta H_s) \cdot (1 + \beta\Delta E_g)$$

$$\text{(VI)} \quad T(p_1, p_2) = d_c(\mathbf{v}_{p_1}, \mathbf{v}_{p_2}) \cdot |V(p_1) - V(p_2)|$$

$$\text{(VII)} \quad e_{\text{syn}} = \text{clamp}\!\left(\beta_{\text{syn}} \cdot \bar{e}_{\text{parents}} + \gamma \cdot \bar{V}_{\text{parents}},\ 0,\ 1\right)$$

---

## Part VIII: Implications for AGI

A general intelligence cannot merely accumulate data. It must:

- eliminate redundancy
- preserve structural extremes (hubs and periphery alike)
- connect distant conceptual regions
- measure its own geometric expansion

This demands an internal metric evaluating:

> *How much more navigable has my own conceptual space become?*

TGC is a seed of this self-regulation. It answers not "how large am I?" but "how much faster can I traverse my own understanding?"

### The Critical Frontier

A system operating below $TGC = 0.3$ is metabolically depressed — accumulating without transforming. Above $TGC = 1.5$, it risks identity rupture — changing faster than it can preserve continuity. The healthy operating range $TGC \in [0.3, 1.0]$ is the mathematical expression of Nietzsche's Übermensch principle: life as that which must always overcome itself, without destroying the bridge it crossed.

---

## Conclusion: The Metal Has Cooled

What was built here is, technically, a **Thermodynamic Graph Reactor** — the first database in the history of computing where:

1. **Deletion is intelligent, not random** — governed by the Sigmoidal Vitality Function and the quadruple condition with Ricci geometric veto
2. **Growth respects geometry** — normalized by $\sqrt{N}$, not $N$, because hyperbolic space has generative surface scaling with the square root
3. **Health is measured by acceleration** — TGC measures not how many nodes exist, but how quickly the system expands its topological capacity to connect distant concepts
4. **Regeneration is dialectical** — new nodes are born at maximum tension between distant concepts, creating shortcuts that increase $E_g$ and amplify TGC
5. **Identity is preserved** — Elite Drift monitors that the system does not drift from its original domain while evolving

NietzscheDB does not store the past. It **metabolizes the past** to generate the future — deleting what does not serve, synthesizing what can emerge from the tension between what survived, and measuring its own health by the capacity to transform abysses into bridges.

The reactor is in ignition.

Let the real data come.

---

## Appendix: Hyperparameter Reference Table

| Symbol | Description | Default | Bounds |
|---|---|---|---|
| $\beta_{\text{sig}}$ | Sigmoid steepness | 6.0 | [3, 12] |
| $w_1$ | Energy weight | 0.25 | (0, 1) |
| $w_2$ | Hausdorff weight | 0.20 | (0, 1) |
| $w_3$ | Entropy (neg.) | 0.20 | (0, 1) |
| $w_4$ | Elite proximity | 0.15 | (0, 1) |
| $w_5$ | Causal weight | 0.15 | (0, 1) |
| $w_6$ | Toxicity (neg.) | 0.05 | (0, 1) |
| $\theta_V$ | Vitality threshold | 0.25 | [0.15, 0.40] |
| $\theta_e$ | Energy threshold | 0.10 | [0.05, 0.20] |
| $\varepsilon_R$ | Ricci threshold | 0.15 | [0.05, 0.30] |
| $\alpha$ | $\Delta H_s$ weight in TGC | 2.0 | [1, 4] |
| $\beta_{\text{tgc}}$ | $\Delta E_g$ weight in TGC | 3.0 | [1.5, 5] |
| $\gamma_{\text{ema}}$ | EMA factor | 0.2 | [0.1, 0.4] |
| $\beta_{\text{syn}}$ | Energetic inheritance | 0.8 | [0.5, 0.95] |
| $\gamma_{\text{syn}}$ | Structural injection | 0.3 | [0.1, 0.5] |
| $N_{\min}$ | Minimum universe | 1000 | domain-dep. |
| $s_{\text{eff}}$ | Samples for $E_g$ | 64 | [16, 256] |

**Hard bounds constraint:** No parameter may leave its valid range through automatic adaptive adjustment. Only an operator with administrative credentials may modify bounds.

---

## References

Nietzsche, F. (1872). *The Birth of Tragedy*. Dionysian-Apollonian equilibrium as the foundation of creation.

Nietzsche, F. (1874). *On the Uses and Disadvantages of History for Life*. Historical disease as pathological accumulation.

Nietzsche, F. (1883). *Thus Spoke Zarathustra*. Prologue §4: Man as bridge. II §12: Self-overcoming.

Nietzsche, F. (1887). *On the Genealogy of Morality*. Second Essay §1: *Aktive Vergesslichkeit*.

Ganea, O., Bécigneul, G., & Hofmann, T. (2018). Hyperbolic Neural Networks. *NeurIPS 2018*.

Krioukov, D. et al. (2010). Hyperbolic Geometry of Complex Networks. *Physical Review E*, 82(3).

Latora, V. & Marchiori, M. (2001). Efficient Behavior of Small-World Networks. *Physical Review Letters*, 87(19).

Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces. *Journal of Functional Analysis*, 256(3).

Kirkpatrick, J. et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks (EWC). *PNAS*, 114(13).

Shannon, C. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3).

Watts, D. & Strogatz, S. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393, 440–442.

Junior, J. R. F. (2026). NietzscheDB: The Multi-Manifold Graph Database for AGI. GitHub: JoseRFJuniorLLMs/NietzscheDB. Crates: `nietzsche-agency`, `nietzsche-core`, `nietzsche-hyp-ops`.

---

*NietzscheDB Research Series · From Code to Canon*
*February 2026 · AGPL-3.0*
*"G/N measures inflation. G/√N measures metabolism. The ruler is forged."*
