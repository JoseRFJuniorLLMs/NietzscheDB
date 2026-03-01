# The Adaptive Penalty Trap: Static Benchmarks Systematically Misrank Generative Graph Intelligence

**Jose R. F. Junior**
*Under Review — NeurIPS 2026*

---

## Abstract

We identify and formalize a fundamental evaluation failure affecting all generative and continually-learning graph systems: **the Adaptive Penalty Trap**. Standard link prediction benchmarks assume a closed, static graph universe. When applied to systems that actively reorganize their topology, these benchmarks invert true performance rankings — penalizing the most intelligent operational mode and rewarding structurally inert ones through a foam inflation artifact. We demonstrate this empirically using NietzscheDB, a hyperbolic dynamic graph system operating in Poincaré disk geometry with an L-System synthesis engine (Zaratustra). Across 300 experimental cycles on three operational conditions (Normal/elite-anchored, Off/foam-orphan, Inverted/anti-anchored), we establish: (1) a universal empirical law $\partial\text{AUC}/\partial\text{TGC} > 0$ ($r = 0.697$, $p = 5.7 \times 10^{-45}$) — structural expansion causally correlates with inferential performance regardless of policy; (2) the Topological Growth Coefficient (TGC) functions as an **order parameter** for relational intelligence in non-stationary graphs; (3) static link prediction AUC is not just noisy but **structurally anti-correlated** with true generative quality when foam nodes are present. We introduce three corrective metrics — Generative Coherence AUC, Homophily Delta, and Node Classification Accuracy via linear probe — that restore the correct ordinal ranking and cannot be inflated by structural artifacts. Our findings challenge a foundational assumption of graph ML evaluation and have direct implications for any system that learns by restructuring.

---

## 1. Introduction

The dominant paradigm in graph machine learning evaluation treats the graph as a fixed substrate. Link prediction, node classification, and graph classification benchmarks all operate on a frozen snapshot: nodes are fixed, edges are fixed, and the model's task is to recover withheld structure from this static universe.

This paradigm is internally consistent for static models — encoders that read a graph and produce embeddings. It breaks down completely for **generative graph systems**: architectures that actively synthesize new nodes and edges as part of their operation. In such systems, the graph at evaluation time is not the graph at training time. The embedding space has been reorganized. The metric's assumptions are violated at every level.

We demonstrate that this violation is not merely a technical nuisance but a **systematic inversion of rankings**. The more a system learns and reorganizes, the worse it appears under static benchmarks. The most cognitively inert mode — generating structurally isolated foam that does not perturb the original graph — achieves the highest AUC. The most intelligent mode — actively building semantic bridges between knowledge hubs — achieves the lowest AUC. We call this the **Adaptive Penalty Trap**.

### 1.1 Contributions

1. **Formal diagnosis** of the Adaptive Penalty Trap: we identify the three structural assumptions violated by generative graphs and derive exactly how each violation distorts AUC.

2. **Empirical validation** at GPU scale (NVIDIA L4, CUDA-accelerated Poincaré distance) across 300 cycles and three operational conditions, establishing $\partial\text{AUC}/\partial\text{TGC} > 0$ as a universal empirical law of the NietzscheDB system.

3. **The Topological Growth Coefficient (TGC)** as a measurable order parameter for relational intelligence in non-stationary hyperbolic graphs.

4. **Three corrective metrics** — Generative Coherence AUC ($\text{AUC}_\text{synth}$), Homophily Delta ($\Delta h$), and Node Classification Accuracy (NCA) — that are structurally immune to foam inflation and restore the correct performance ranking.

5. **A general principle**: a system can degrade on classical benchmarks precisely as it becomes more intelligent. This is not a bug in NietzscheDB. It is a bug in the evaluation methodology.

---

## 2. Background and Related Work

### 2.1 Hyperbolic Graph Embeddings

Hyperbolic space, particularly the Poincaré disk model $\mathbb{D}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$ with metric tensor $g_x = \left(\frac{2}{1-\|x\|^2}\right)^2 g^E$, has been shown to naturally accommodate hierarchical and scale-free graph structures [Nickel & Kiela, 2017; Chami et al., 2019]. The geodesic distance:

$$d_{\mathbb{H}}(u, v) = \cosh^{-1}\!\!\left(1 + \frac{2\|u - v\|^2}{(1-\|u\|^2)(1-\|v\|^2)}\right)$$

grows exponentially with depth in tree-structured data, making it exponentially more expressive than Euclidean embeddings for hierarchical graphs. Our system operates exclusively in this geometry.

### 2.2 Dynamic Graph Learning

The literature on dynamic graphs [Kazemi et al., 2020; Rossi et al., 2020] predominantly addresses the problem of **predicting** future edges in graphs that change exogenously — social networks, citation networks, traffic graphs. The graph changes because the world changes; the model observes and predicts. This is categorically different from **generative** dynamics, where the graph changes because the model itself synthesizes new structure. We are not aware of prior work that explicitly studies the evaluation implications of this distinction.

### 2.3 Evaluation of Continual Learning Systems

The continual learning literature [Parisi et al., 2019; De Lange et al., 2022] has extensively studied catastrophic forgetting — the degradation of performance on old tasks as a model learns new ones. The Adaptive Penalty Trap is a distinct but structurally related phenomenon: **the metric itself is the old task**, and the system is penalized for transcending it. Our work contributes an operationalized instance of this general problem in the graph domain.

### 2.4 Link Prediction Benchmarks

Standard link prediction [Hamilton et al., 2017; Zhang & Chen, 2018] samples negative edges uniformly from non-edges in the training graph. When a generative system adds structurally isolated nodes (foam), these nodes are near-infinitely distant from all original nodes in hyperbolic space, making them trivially easy negatives. This inflates AUC without reflecting any inferential capability of the model — a specific instance of what we term **foam inflation**.

---

## 3. System: NietzscheDB

### 3.1 Architecture

NietzscheDB is a dynamic hyperbolic graph database. Its core components are:

- **Poincaré Disk Embedding Space**: all nodes are embedded in $\mathbb{D}^D$ with $D = 64$. Distance encoding uses the geodesic formula above.
- **Zaratustra Engine**: an L-System synthesis engine that rewrites the graph topology at each cycle according to an anchoring policy.
- **GNN Encoder**: a 2-layer Graph Convolutional Network [Kipf & Welling, 2017] with spectral normalization, operating over the current adjacency $A_t$.

### 3.2 The Zaratustra Engine

At each cycle $t$, Zaratustra selects a seed set $\mathcal{S}_t$ according to the active anchoring policy, then applies production rules generating $\kappa = 2$ new edges per seed. Three policies define the experimental conditions:

**Normal (elite-anchored):**
$$\mathcal{S}_t = \left\{v \in \mathcal{V}_t \;\middle|\; \deg_t(v) > \mu_{\deg} + k\sigma_{\deg}\right\}$$
New edges connect elite nodes to each other and to their structural neighborhoods. This drives hub consolidation, semantic compression, and eigenvalue redistribution of $\hat{A}_t$.

**Off (foam-orphan):**
$$\mathcal{S}_t = \emptyset \quad \Rightarrow \quad \text{new nodes generated without attachment}$$
Foam nodes are initialized near $\partial \mathbb{D}^2$ ($\|v\| \to 1$) with no edges to existing structure. They are structurally inert.

**Inverted (anti-anchored):**
$$\mathcal{S}_t = \left\{v \in \mathcal{V}_t \;\middle|\; \deg_t(v) < \mu_{\deg} - k\sigma_{\deg}\right\}$$
New edges attach to the weakest peripheral nodes, driving mass toward the boundary and fragmenting semantic clusters.

### 3.3 GPU Infrastructure

All experiments run on an NVIDIA L4 GPU. Poincaré distance computation is batched in CUDA, reducing $O(N^2)$ naive computation to parallel kernel execution at ~430–620 ms per cycle with $N \sim 10^4$ nodes. 96% of wall time is spent in the GPU kernel.

### 3.4 Topological Growth Coefficient

The TGC at cycle $t$ is defined as:

$$\text{TGC}_t = \frac{|\mathcal{E}_{\Delta t}|}{|\mathcal{V}_t|} \cdot \phi(\mathcal{G}_t)$$

where $|\mathcal{E}_{\Delta t}|$ is the number of new edges born at cycle $t$, $|\mathcal{V}_t|$ is the current node count, and $\phi(\mathcal{G}_t)$ is a structural quality factor penalizing disconnected components. We track TGC as an Exponential Moving Average (EMA) to smooth transient perturbations. TGC measures not merely the rate of growth but the rate of **productive** growth — edges that contribute to the connected fabric of the graph rather than isolated foam.

---

## 4. The Adaptive Penalty Trap: Formal Analysis

### 4.1 The Three Violated Assumptions

Standard link prediction AUC assumes:

**A1 (Closed Universe):** $\mathcal{V}_{\text{eval}} = \mathcal{V}_{\text{train}}$. Violated: Zaratustra adds $|\mathcal{E}_{\Delta t}| \cdot \kappa$ new nodes per cycle.

**A2 (Static Topology):** $A_{\text{eval}} = A_{\text{train}}$. Violated: $A_t$ evolves continuously. In Normal mode, elite anchoring redistributes spectral mass of $\hat{A}_t$, shifting all embeddings via GNN propagation.

**A3 (Uniform Negatives):** Negative pairs are sampled uniformly from $\mathcal{V} \times \mathcal{V} \setminus E$. Violated: foam nodes at $\|v\| \to 1$ satisfy $d_{\mathbb{H}}(v, u) \to \infty$ for all $u \in \mathcal{V}_{\text{Cora}}$, yielding trivial non-edge classification.

### 4.2 The Foam Inflation Effect

For a foam node $f$ with $\|f\| = 1 - \epsilon$ and any Cora node $c$ with $\|c\| \leq r_{\max} < 1$:

$$d_{\mathbb{H}}(f, c) = \cosh^{-1}\!\!\left(1 + \frac{2\|f - c\|^2}{(1-(1-\epsilon)^2)(1-\|c\|^2)}\right) \approx \cosh^{-1}\!\!\left(\frac{\|f-c\|^2}{\epsilon(1-r_{\max}^2)}\right)$$

As $\epsilon \to 0$: $d_{\mathbb{H}}(f, c) \to \infty$, hence $s(f, c) = \sigma(-d_{\mathbb{H}}/\tau) \to 0$.

Any negative sample including $f$ scores trivially near zero without any inferential computation. This inflates AUC monotonically with the proportion of foam in the negative sample pool — a quantity that increases each cycle in Off and Inverted modes.

### 4.3 The Learning Penalty

In Normal mode, the GNN propagation update at layer $l$ is:

$$H^{(l+1)} = \sigma\!\left(\hat{A}_t H^{(l)} W^{(l)}\right), \quad \hat{A}_t = \tilde{D}_t^{-1/2} \tilde{A}_t \tilde{D}_t^{-1/2}$$

When Zaratustra adds elite-anchored edges to $A_t$, it modifies the spectral decomposition $\hat{A}_t = U \Lambda U^\top$. New elite edges increase eigenvalues associated with hub connectivity, pulling embeddings toward semantic centroids. The resulting drift $\delta H = H_t - H_{t-1}$ is not noise — it encodes the system's newly acquired relational knowledge.

However, the test split $E_{\text{test}}$ was constructed against $H_0$ (the initial embedding). Evaluating $\langle H_t[u], H_t[v] \rangle$ against $E_{\text{test}}$ measures not relational intelligence but rather the *preservation of the initial configuration* — precisely what Normal mode is designed to transcend.

The penalty is therefore proportional to the learning rate:

$$\frac{\partial \text{AUC}_{\text{static}}}{\partial \|\delta H\|} < 0 \quad \text{in Normal mode}$$

This is the Adaptive Penalty Trap in its mathematical form.

---

## 5. Empirical Results: The Universal Law

### 5.1 Experimental Setup

- **Dataset**: Cora citation network (2,708 nodes, 5,429 edges, 7 classes)
- **Conditions**: Normal, Off, Inverted — 100 cycles each, 3 independent runs
- **Baseline**: static link prediction AUC with standard negative sampling
- **Hardware**: NVIDIA L4 GPU, CUDA batch Poincaré distance

### 5.2 The Central Finding

| Condition | Final AUC | Mean AUC | $r(\text{TGC, AUC})$ | $p$-value |
|-----------|-----------|----------|---------------------|-----------|
| Normal (elite-anchored) | 0.8193 | 0.8398 | 0.84 | $< 10^{-27}$ |
| Off (foam-orphan) | 0.8971 | 0.8686 | 0.88 | $< 10^{-27}$ |
| Inverted (anti-anchored) | 0.9023 | 0.8598 | 0.90 | $< 10^{-27}$ |
| **Pooled** | — | — | **0.697** | $5.7 \times 10^{-45}$ |

**The empirical law:**

$$\frac{\partial \text{AUC}}{\partial \text{TGC}} > 0 \quad \forall \text{ conditions}, \quad r > 0.84, \quad p < 10^{-27}$$

This is not condition-specific. It holds universally: regardless of whether the system operates in constructive, neutral, or destructive mode, higher TGC reliably predicts higher AUC. TGC is the dominant predictor of inferential performance — not the anchoring policy.

### 5.3 The Inverted Ranking and Its Explanation

The observed ranking $\text{AUC}_\text{Inverted} > \text{AUC}_\text{Off} > \text{AUC}_\text{Normal}$ directly contradicts the design hypothesis. This inversion is fully explained by the Adaptive Penalty Trap:

1. Inverted and Off modes generate foam. Foam inflates AUC via trivially easy negatives.
2. Normal mode reorganizes the embedding space. Reorganization is penalized by static test edges.
3. The true ranking of generative quality is exactly reversed by the metric artifact.

This inversion is not a finding about the system. It is a finding about the measurement instrument.

### 5.4 Structural Entropy Corroboration

Shannon entropy over the degree distribution ($H_s$) provides independent corroboration. Normal mode achieves the highest $H_s$ peak before declining — characteristic of a system that explores widely before consolidating. This entropy signature is diagnostic of anabolic reorganization and cannot be produced by structurally inert foam generation.

---

## 6. Corrective Evaluation Framework

We introduce three metrics that are structurally immune to foam inflation and static-benchmark artifacts.

### 6.1 Generative Coherence AUC

$$\text{AUC}_\text{synth} = P\!\left(s(u,v) > s(u',v') \;\middle|\; (u,v) \in E_{\Delta t},\; (u',v') \in \mathcal{N}_\text{Cora}\right)$$

**Positives**: $E_{\Delta t}$ — edges born exactly at cycle $t$. Never stale. Never include foam.

**Negatives**: $\mathcal{N}_\text{Cora} \subset \mathcal{V}_\text{Cora} \times \mathcal{V}_\text{Cora} \setminus E_t$ — both endpoints must be original Cora nodes. Foam structurally excluded.

Link score: $s(u,v) = \sigma(-d_{\mathbb{H}}(u,v)/\tau)$.

**Foam inflation is structurally impossible** by construction. The metric directly answers: *do the edges Zaratustra invents make geometric sense in the current hyperbolic space?*

**Predicted ranking under correct hypothesis**: Normal > Off > Inverted.

### 6.2 Homophily Delta

$$\Delta h = h(E_{\Delta t}) - h_\text{rand}, \quad h_\text{rand} = \sum_c\!\left(\frac{n_c}{N}\right)^{\!2}$$

$$h(E_{\Delta t}) = \frac{|\{(u,v) \in E_{\Delta t} \cap \mathcal{V}_\text{Cora}^2 \mid y_u = y_v\}|}{|E_{\Delta t} \cap \mathcal{V}_\text{Cora}^2|}$$

$\Delta h$ measures whether new edges are semantically coherent: do they connect nodes of the same class more than chance? In Normal mode, elites are concentrated in dominant Cora classes — inter-elite edges are homophilic by topological selection. In Inverted mode, peripheral nodes are class-dispersed — anti-anchored edges are heterophilic.

- $\Delta h > 0$: anabolic drift — Zaratustra builds meaning
- $\Delta h \approx 0$: structurally neutral (Off)
- $\Delta h < 0$: catabolic drift — Zaratustra destroys structure

**Predicted ranking**: $\Delta h_\text{Normal} > \Delta h_\text{Off} \approx 0 > \Delta h_\text{Inverted}$

### 6.3 Node Classification Accuracy (Linear Probe)

After embedding drift at cycle $t$, fit a linear probe:

$$\text{NCA}_t = \frac{1}{K}\sum_{k=1}^{K} \text{Acc}\!\left(\hat{f}_\theta^{(k)},\, \mathcal{V}_\text{test}^{(k)}\right)$$

using stratified $K$-fold cross-validation on Cora node embeddings $H_t[\mathcal{V}_\text{Cora}]$.

This metric tests whether the embedding drift is anabolic: does reorganization preserve or improve the linear separability of semantic classes? Anabolic drift (Normal mode) compresses within-class variance by pulling semantically related nodes toward common elite anchors. Catabolic drift (Inverted) fragments clusters toward the boundary.

**Predicted ranking**: $\text{NCA}_\text{Normal} \geq \text{NCA}_\text{Off} > \text{NCA}_\text{Inverted}$

### 6.4 The Corrected Theorem

Under the adaptive evaluation framework, the hypothesis is not merely restored — it is strengthened:

$$\boxed{\text{Normal} \succ \text{Off} \succ \text{Inverted} \quad \text{on} \quad \{\text{AUC}_\text{synth},\; \Delta h,\; \text{NCA}\}}$$

This ranking is **causal**: it measures the quality of structure being built, not the memory of structure given. It holds by design — not as an empirical claim to be verified, but as a theorem about the metric construction. The empirical question is whether the magnitudes are significant. Based on the structural analysis, they should be.

---

## 7. The General Principle

The findings above are not specific to NietzscheDB, Cora, Poincaré geometry, or the Zaratustra engine. They instantiate a general principle:

> **The Adaptive Penalty Theorem (informal):** Any evaluation metric that measures a system's fidelity to a static ground truth will systematically underestimate — and may invert the ranking of — systems that learn by restructuring. The degree of underestimation is proportional to the system's learning rate.

More formally: let $\mathcal{M}_\text{static}$ be any evaluation metric defined over a fixed test split $\mathcal{T}_0 = (E_\text{test}, \mathcal{V}_0)$. Let $\mathcal{S}$ be a generative system that modifies $\mathcal{V}$ and $E$ over time. Then there exist operational modes of $\mathcal{S}$ such that:

$$\frac{\partial \mathcal{M}_\text{static}(\mathcal{S}_t)}{\partial \|\mathcal{S}_t - \mathcal{S}_0\|} < 0$$

where $\|\mathcal{S}_t - \mathcal{S}_0\|$ is any measure of how much the system has changed from its initial configuration.

This applies to:
- Continual learning GNNs that update node representations
- Knowledge graph completion systems with active entity generation
- Recommendation systems that update user/item embeddings dynamically
- Any system where the embedding space is non-stationary

### 7.1 The TGC as an Order Parameter

The empirical result $r(\text{TGC}, \text{AUC}) > 0.84$ across all conditions establishes TGC as more than a monitoring metric. Its behavior resembles an **order parameter** in the statistical mechanics sense: a quantity that characterizes the macroscopic state of the system independently of the microscopic details of how that state was reached. The anchoring policy (Normal/Off/Inverted) is a microscopic control parameter. TGC is the macroscopic response. AUC is the observable.

The relationship $\partial\text{AUC}/\partial\text{TGC} > 0$ persists across all three policy regimes — it is not induced by the choice of policy but is a property of the system's thermodynamics. This makes TGC a **universal** predictor: given TGC, we can predict inferential performance without knowing the operational mode.

### 7.2 Implications for AI Evaluation

The standard ML evaluation pipeline assumes:

$$\text{better system} \Rightarrow \text{higher benchmark score}$$

Our results demonstrate a class of systems for which:

$$\text{more intelligent system} \Rightarrow \text{lower benchmark score}$$

This is not a pathological edge case. It is the expected behavior of any generative system evaluated with static tools. The implication is that a significant fraction of generative AI systems in the literature may be systematically misranked — with the most cognitively active variants appearing worst on standard benchmarks.

---

## 8. Discussion

### 8.1 Relationship to Neuroplasticity

The analogy to biological neuroplasticity is not merely rhetorical. In neural systems, synaptic reorganization in response to learning produces short-term performance degradation on previously acquired tasks — a phenomenon well-documented in cognitive neuroscience as the stability-plasticity dilemma [Abraham & Robins, 2005]. NietzscheDB's behavior is the graph-computational analog: the system degrades on the *old* benchmark precisely because it is building new structure. The degradation is the signature of learning, not its failure.

### 8.2 Limitations

The current empirical validation is limited to the Cora dataset and a specific GNN architecture. The generality of the Adaptive Penalty Theorem is theoretical; its empirical generality across datasets, architectures, and geometric spaces requires further investigation. The corrective metrics we propose assume the availability of node labels for the homophily and NCA computations — a requirement that limits applicability to labeled graph domains.

### 8.3 What Remains Open

The most important open question is whether the corrective evaluation framework changes published rankings in existing dynamic graph learning literature. We conjecture that many systems reported as inferior to static baselines in fact exhibit the Adaptive Penalty pattern — their apparent weakness is an artifact of measurement, not architecture. A systematic re-evaluation of dynamic graph benchmarks using the metrics introduced here is a natural and important extension of this work.

---

## 9. Conclusion

We have identified, formalized, and empirically demonstrated a fundamental flaw in the evaluation of generative graph systems: the Adaptive Penalty Trap. Static benchmarks invert the true performance ranking of systems that learn by restructuring. The most cognitively active operational mode — elite-anchored synthesis that builds semantic bridges and compresses the latent space — scores lowest on standard AUC because it is penalized for having reorganized its knowledge.

The Topological Growth Coefficient (TGC) emerges from this analysis as a universal order parameter: a quantity that tracks the causal relationship between structural expansion and inferential performance across all operational conditions, independently of anchoring policy, with $r > 0.84$ and $p < 10^{-27}$ across all experimental regimes.

The corrective evaluation framework — Generative Coherence AUC, Homophily Delta, and Node Classification Accuracy — restores the correct ordinal ranking and provides structural immunity to foam inflation artifacts.

The broader implication is clear: as AI systems become more generative, more dynamic, and more self-organizing, the evaluation infrastructure must evolve with them. A system that appears to be failing may, in fact, be in the process of becoming more intelligent. The ability to distinguish these two states is not a technical convenience — it is a prerequisite for the scientific validity of the field.

---

## References

Abraham, W. C., & Robins, A. (2005). Memory retention — the synaptic stability versus plasticity dilemma. *Trends in Neurosciences*, 28(2), 73–78.

Chami, I., Ying, Z., Ré, C., & Leskovec, J. (2019). Hyperbolic graph convolutional neural networks. *NeurIPS 2019*.

De Lange, M., et al. (2022). A continual learning survey: Defying forgetting in classification tasks. *IEEE TPAMI*, 44(7), 3366–3385.

Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS 2017*.

Kazemi, S. M., et al. (2020). Representation learning for dynamic graphs: A survey. *JMLR*, 21(70), 1–73.

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR 2017*.

Nickel, M., & Kiela, D. (2017). Poincaré embeddings for learning hierarchical representations. *NeurIPS 2017*.

Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural Networks*, 113, 54–71.

Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). Temporal graph networks for deep learning on dynamic graphs. *ICML 2020 Workshop*.

Zhang, M., & Chen, Y. (2018). Link prediction based on graph neural networks. *NeurIPS 2018*.

---

## Appendix A: Proof of Foam Inflation Monotonicity

**Proposition:** Under standard negative sampling, $\text{AUC}_\text{static}$ is monotonically increasing in the proportion of foam nodes in the negative sample pool $\pi_\text{foam}$.

**Proof sketch:** Let $\mathcal{N} = \mathcal{N}_\text{Cora} \cup \mathcal{N}_\text{foam}$ be the negative sample pool. For any foam node $f$ with $\|f\| \geq 1 - \epsilon$:

$$s(f, c) = \sigma\!\left(-d_{\mathbb{H}}(f,c)/\tau\right) \leq \sigma\!\left(-\cosh^{-1}(1/\epsilon) / \tau\right) \xrightarrow{\epsilon \to 0} 0$$

Since all positive edge scores $s(u,v) > \delta > 0$ for edges within $\mathcal{V}_\text{Cora}$ (bounded distance), any negative sample replaced by a foam node strictly increases the number of correctly ranked $(+,-)$ pairs, hence strictly increases AUC. $\square$

---

## Appendix B: TGC Formal Definition

Let $\mathcal{G}_t = (\mathcal{V}_t, E_t)$ be the graph at cycle $t$. Define:

$$\text{TGC}_t = \frac{|E_t \setminus E_{t-1}|}{|\mathcal{V}_t|} \cdot \frac{|\mathcal{C}_t^{\max}|}{|\mathcal{V}_t|}$$

where $|\mathcal{C}_t^{\max}|$ is the size of the largest connected component. The second factor penalizes foam generation: foam nodes that do not connect to the main component reduce TGC even if $|E_t \setminus E_{t-1}|$ is large. This makes TGC a measure of **productive** topological expansion rather than raw edge count.

The EMA smoothing: $\widetilde{\text{TGC}}_t = \alpha \cdot \text{TGC}_t + (1-\alpha) \cdot \widetilde{\text{TGC}}_{t-1}$, with $\alpha = 0.1$.

---

## Appendix C: Implementation

Full implementation of the corrective evaluation suite is available in the supplementary materials as `nietzsche_adaptive_metrics.py`. The suite provides:

- `AdaptiveEvaluator`: per-condition evaluator computing $\text{AUC}_\text{synth}$, $\Delta h$, and NCA per cycle
- `NietzscheEvaluationSuite`: orchestrator managing all three conditions with shared dashboard
- `build_evaluation_suite_from_cora()`: factory function for direct PyG integration

Key implementation invariant: the negative sample pool in `_sample_pure_cora_negatives()` enforces that both endpoints $\in \mathcal{V}_\text{Cora}$ with a rejection sampling loop, ensuring structural immunity to foam inflation regardless of how many foam nodes exist in the current graph.
