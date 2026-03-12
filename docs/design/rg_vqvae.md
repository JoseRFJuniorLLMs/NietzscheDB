# Multi-Level Renormalization Group (RG) via VQ-VAE

**Status**: Design Document (analysis/monitoring only -- no behavioral changes)
**Date**: 2026-03-11
**Scope**: Periodic analytical pipeline, not per-tick

---

## 1. Motivation

The Renormalization Group (RG) is a framework from statistical physics for studying
systems across multiple scales. The core idea: coarse-grain a system at progressively
larger scales and observe how the effective description (Hamiltonian) flows under
these transformations.

NietzscheDB already has the building blocks for a multi-scale RG analysis:

- **VQ-VAE** provides a natural coarse-graining: continuous embeddings are mapped
  to discrete codebook entries, reducing degrees of freedom
- **DSI hierarchical IDs** organize nodes into a tree of VQ codes at multiple levels
- **Cognitive Layer** discovers Louvain/radius-based clusters (a form of spatial RG)
- **Thermodynamics** already computes temperature, entropy, and free energy

An RG analysis would reveal whether the knowledge graph operates near a **critical
point** (self-organized criticality), which would explain the power-law distributions
and scale-free properties observed in mature collections.

---

## 2. RG Levels: VQ-VAE as Coarse-Graining Operator

### 2.1 Level Hierarchy

```
L0  (microscopic)    Individual nodes, full embeddings
    |                 3072D raw embeddings (or 128D Poincare coords)
    | VQ encode
    v
L1  (mesoscopic)     VQ codebook vectors
    |                 1024 codes (from VqVaeConfig.num_embeddings)
    |                 Each code represents a cluster of similar nodes
    | k-means on codebook
    v
L2  (macroscopic)    Super-codes (clusters of codebook vectors)
    |                 ~64-128 super-clusters
    | recursive clustering
    v
L3  (global)         Meta-clusters
                      ~8-16 top-level domains
```

### 2.2 Existing Infrastructure Mapping

| RG Level | NietzscheDB Component | Dimensionality | Count |
|----------|-----------------------|----------------|-------|
| L0 | `GraphStorage::get_embedding()` | 128D (Poincare coords, f32) | N nodes |
| L1 | `VqEncoder::encode()` -> codebook index | 512D latent -> 1 discrete code | 1024 codes (from `train_vqvae.py`: `num_embeddings=1024`) |
| L2 | `PQEncoder::encode()` -> M sub-codes | 128D -> M x u8 sub-codes | K^M possible, ~256 effective |
| L_cog | `CognitiveLayer::run_cognitive_scan()` | Poincare radius-based clusters | ~10-50 concept nodes |
| L_louv | `louvain()` | Modularity-optimized communities | variable |

### 2.3 VQ-VAE Architecture (Existing)

From `scripts/models/train_vqvae.py`:

```
Encoder: Linear(3072, 1024) -> ReLU -> Linear(1024, 512)
VQ:      512D latent space, 1024 codebook vectors
         Commitment cost: 0.25
Decoder: Linear(512, 1024) -> ReLU -> Linear(1024, 3072)
```

From `crates/nietzsche-vqvae/src/model.rs` (Rust runtime config):

```
embedding_dim:    128   (Poincare-projected)
num_embeddings:   512   (production codebook size)
```

The discrepancy (training=1024 codes at 512D, runtime=512 codes at 128D) reflects
that the ONNX-exported model operates on Poincare-projected coordinates rather than
raw 3072D embeddings. The RG pipeline should use the **runtime configuration**
(128D, 512 codes) since that matches the actual stored embeddings.

### 2.4 The RG Transformation

The RG transformation T: L_k -> L_{k+1} is defined as:

```
T(L0 -> L1):  node_embedding (128D f32) --[VqEncoder::encode]--> codebook_index (u16)
              Groups ~N/512 nodes per code on average

T(L1 -> L2):  codebook_vectors (512 x 128D) --[k-means, K=64]--> super-code_index
              Groups ~8 codes per super-code on average

T(L2 -> L3):  super-code_centroids (64 x 128D) --[k-means, K=8]--> meta-cluster
              Groups ~8 super-codes per meta-cluster
```

Each transformation reduces degrees of freedom by a factor of ~8x (the "decimation
ratio"). This is analogous to block-spin RG where a block of spins is replaced by
a single effective spin.

**Key property**: Because VQ-VAE was trained with a reconstruction loss, the
codebook vectors are learned representatives that preserve the essential structure
of the embedding space -- they are not arbitrary partitions. This makes VQ-VAE a
better RG operator than naive grid-based coarse-graining.

---

## 3. Effective Hamiltonian at Each Level

### 3.1 Definition

At each RG level L, we define an effective Hamiltonian H_eff(L) that captures the
essential physics of the coarse-grained system:

```
H_eff(L) = H_energy(L) + H_connectivity(L) + H_hyperbolic(L)
```

Where:

**Energy term** (from existing thermodynamics):
```
H_energy(L) = { mean_energy(L), energy_std(L), entropy(L), free_energy(L) }

At L0: computed directly from node energies (existing ThermodynamicReport)
At L1: aggregate energies of all nodes mapped to each code
       mean_energy_code_j = mean(energy_i for all nodes i where VQ(i) = j)
At L2: aggregate over super-codes
At L3: aggregate over meta-clusters
```

**Connectivity term** (from graph structure):
```
H_connectivity(L) = { mean_degree(L), clustering_coefficient(L), modularity(L) }

At L0: direct graph metrics (existing adjacency index)
At L1: construct "code-level graph" where codes are nodes, edges are
       inter-code connections (weighted by number of L0 edges between codes)
At L2: super-code graph from code-level graph
At L3: meta-cluster graph
```

**Hyperbolic term** (specific to Poincare ball):
```
H_hyperbolic(L) = { mean_depth(L), depth_variance(L), curvature_estimate(L) }

At L0: depth = ||embedding|| (magnitude = hierarchy depth in Poincare ball)
At L1: mean depth of code centroid = ||codebook_vector_j||
At L2: mean depth of super-code centroid
At L3: mean depth of meta-cluster centroid

NOTE: magnitude preservation is critical here -- this is why Binary Quantization
is rejected (it destroys magnitude). VQ-VAE and PQ both preserve it
(see test_magnitude_preservation in nietzsche-pq tests).
```

### 3.2 Concrete Computation

```python
# Pseudocode for H_eff computation at level L

def compute_H_eff(level: int, assignments: dict, graph_data: GraphData) -> H_eff:
    """
    assignments: maps each entity at level L to its parent at level L+1
    graph_data: node embeddings, energies, edges
    """
    # Group nodes by their level-L assignment
    groups = group_by(assignments)

    # Energy statistics per group
    group_energies = {
        g: [node.energy for node in groups[g]]
        for g in groups
    }
    mean_E = mean([mean(e) for e in group_energies.values()])
    std_E  = std([mean(e) for e in group_energies.values()])
    S      = shannon_entropy([mean(e) for e in group_energies.values()])
    T      = std_E / mean_E if mean_E > 0 else 0  # cognitive temperature
    F      = mean_E - T * S                         # Helmholtz free energy

    # Connectivity: build coarse graph
    coarse_edges = defaultdict(float)
    for (u, v, w) in graph_data.edges:
        gu, gv = assignments[u], assignments[v]
        if gu != gv:
            coarse_edges[(gu, gv)] += w
    mean_degree = mean([degree(g) for g in coarse_graph])
    clustering  = mean_clustering_coefficient(coarse_graph)
    modularity  = compute_modularity(coarse_graph)

    # Hyperbolic depth
    group_depths = {
        g: mean([||node.embedding|| for node in groups[g]])
        for g in groups
    }
    mean_depth    = mean(group_depths.values())
    depth_var     = variance(group_depths.values())

    return H_eff(
        temperature=T, entropy=S, free_energy=F,
        mean_energy=mean_E, energy_std=std_E,
        mean_degree=mean_degree,
        clustering_coefficient=clustering,
        modularity=modularity,
        mean_depth=mean_depth,
        depth_variance=depth_var,
        num_entities=len(groups),
    )
```

### 3.3 Connection to Existing Thermodynamics

The existing `ThermodynamicReport` from `crates/nietzsche-agency/src/thermodynamics.rs`
already computes `H_eff(L0)`:

```rust
ThermodynamicReport {
    temperature,      // T = sigma_E / mean_E
    entropy,          // S = -sum(p_i * ln(p_i))
    free_energy,      // F = U - T*S
    mean_energy,      // U
    energy_std,       // sigma_E
    phase,            // {Solid, Liquid, Gas, Critical}
    ...
}
```

The RG pipeline extends this to compute the same quantities at L1, L2, L3.

---

## 4. RG Flow and Fixed Point Detection

### 4.1 RG Flow

The RG flow describes how H_eff evolves across levels. Define the flow vector:

```
Delta_H(L) = H_eff(L+1) - H_eff(L)
```

For each component:

```
Delta_T(L) = T(L+1) - T(L)      # temperature flow
Delta_S(L) = S(L+1) - S(L)      # entropy flow
Delta_F(L) = F(L+1) - F(L)      # free energy flow
Delta_Q(L) = Q(L+1) - Q(L)      # modularity flow
Delta_d(L) = d(L+1) - d(L)      # mean depth flow
```

### 4.2 Fixed Point Criterion

A **fixed point** exists when the effective Hamiltonian is invariant under the
RG transformation:

```
||H_eff(L+1) - H_eff(L)|| < epsilon    for some threshold epsilon
```

In practice, we compute a normalized distance between successive levels:

```
RG_distance(L) = sqrt(
    w_T * (Delta_T / T_scale)^2 +
    w_S * (Delta_S / S_scale)^2 +
    w_F * (Delta_F / F_scale)^2 +
    w_Q * (Delta_Q / Q_scale)^2 +
    w_d * (Delta_d / d_scale)^2
)
```

Where `w_*` are weights and `*_scale` are normalization factors.

**Fixed point detection**: If RG_distance(L) < epsilon for consecutive levels,
the system is at or near a critical point.

### 4.3 Interpretation for NietzscheDB

| RG Flow Pattern | Interpretation |
|-----------------|----------------|
| H_eff converges to fixed point | System at SOC critical point -- scale-invariant, optimal for knowledge organization |
| T increases with level | System becomes more "chaotic" at large scales -- poor global coherence |
| T decreases with level | System becomes more "rigid" at large scales -- good hierarchical structure |
| S constant across levels | Self-similar entropy distribution -- hallmark of criticality |
| Modularity increases with level | Hierarchical community structure strengthens at coarser scales |
| Depth variance decreases with level | Hyperbolic hierarchy becoming flatter at coarse scales (expected) |

### 4.4 Connection to Self-Organized Criticality (SOC)

NietzscheDB's agency engine already drives the system toward self-organization:

- **Hebbian learning** strengthens useful connections (positive feedback)
- **Temporal decay** prunes weak connections (negative feedback)
- **Energy minimization** (Phase 21) seeks low-energy configurations
- **Cognitive temperature** modulates explore/exploit balance

If the system reaches SOC, the RG flow should show:
1. Fixed-point behavior in H_eff
2. Power-law distributions in code cluster sizes
3. Scale-invariant correlation functions
4. Critical exponents matching known universality classes

---

## 5. Critical Exponent Extraction

### 5.1 Correlation Length Exponent (nu)

Near criticality, the correlation length xi diverges as:

```
xi ~ |T - T_c|^(-nu)
```

Estimate nu from the RG flow:

```
nu = -ln(b) / ln(lambda_T)
```

Where:
- b = decimation ratio (~8 for our VQ-based RG)
- lambda_T = eigenvalue of the linearized RG flow near the fixed point
  for the temperature direction

### 5.2 Scaling Dimension of Energy (y_E)

```
mean_energy(L+1) / mean_energy(L) = b^(y_E)
y_E = ln(mean_energy(L+1) / mean_energy(L)) / ln(b)
```

### 5.3 Dynamic Exponent (z)

Relates spatial and temporal scaling. Measurable from the agency engine:

```
relaxation_time ~ xi^z
```

Where relaxation_time is the number of agency ticks needed for the thermodynamic
report to stabilize after a perturbation (e.g., bulk insertion of nodes).

### 5.4 Fractal Dimension (d_f)

```
N(L) ~ R(L)^(d_f)
```

Where N(L) is the number of entities at level L and R(L) is the effective
radius of the system at that level. In the Poincare ball:

```
R(L) = mean(||centroid_j|| for j in level L entities)
d_f = ln(N(L) / N(L+1)) / ln(R(L+1) / R(L))
```

### 5.5 Universality Class Identification

Compare extracted exponents to known universality classes:

| Class | nu | d_f | System |
|-------|----|-----|--------|
| Mean-field | 0.5 | 4 | Random graphs, Erdos-Renyi |
| 2D Ising | 1.0 | 2 | Planar systems |
| Percolation (2D) | 1.33 | 1.89 | Network connectivity |
| KPZ | 0.63 | -- | Growth processes |
| Directed percolation | 0.73 | -- | Non-equilibrium phase transitions |

If the NietzscheDB knowledge graph falls into a known universality class, it
would reveal deep structural properties about how knowledge self-organizes.

---

## 6. Connection to Existing Systems

### 6.1 Cognitive Layer (Phase E)

The cognitive layer in `crates/nietzsche-agency/src/cognitive_layer.rs` already
performs a form of spatial coarse-graining:

- Samples embeddings from the Poincare ball (max 2000 nodes)
- Greedy radius-based clustering (Poincare distance < `cluster_radius` = 0.3)
- Computes gyro-midpoint (Frechet mean) as cluster centroid
- Proposes ConceptNode at centroid with edges to members

**RG connection**: Cognitive layer clusters are an L1-equivalent in the spatial
domain. The RG pipeline should compare:

```
L1_vq (from VQ-VAE):      code assignments based on embedding similarity
L1_cog (from cognitive):   cluster assignments based on Poincare distance
L1_louv (from Louvain):    community assignments based on graph connectivity
```

If all three L1 representations yield similar H_eff values, the system has
**consistent multi-scale structure** -- a strong indicator of criticality.

### 6.2 DSI Indexing

The DSI indexer in `crates/nietzsche-dsi/src/indexer.rs` generates hierarchical
SemanticId codes:

```rust
SemanticId(Vec<u16>)  // e.g., [42, 17, 8] for a 3-level hierarchy
```

Each level of the SemanticId is a coarse-graining step. The DSI decoder
(`crates/nietzsche-dsi/src/decoder.rs`) uses a neural model that outputs:

```
[B, 4, 1024]  -- logits per hierarchy level, 1024-code vocabulary, 4 levels
```

This 4-level hierarchy maps directly to RG levels L0-L3:

```
DSI Level 0 (finest)   <-> RG L0 (individual nodes)
DSI Level 1            <-> RG L1 (VQ codes)
DSI Level 2            <-> RG L2 (super-codes)
DSI Level 3 (coarsest) <-> RG L3 (meta-clusters)
```

**RG connection**: The DSI confidence per level (softmax probability) can serve
as a proxy for the "order parameter" at each scale. High confidence at a level
means the assignment is unambiguous -- the system has clear structure at that
scale. Low confidence indicates a phase boundary where the assignment is
degenerate -- a signature of criticality.

### 6.3 Product Quantization (PQ)

The PQ codebook in `crates/nietzsche-pq/src/codebook.rs` provides an alternative
coarse-graining:

```
Codebook {
    config: PQConfig { m: 8, k: 256, ... },  // 8 sub-quantizers, 256 centroids each
    dim: 128,
    sub_dim: 16,          // 128 / 8
    centroids: [M][K][sub_dim],  // Shape: [8][256][16]
}
```

PQ decomposes the embedding into M independent sub-spaces, each quantized to K
centroids. This is a **factored RG transformation** -- each sub-space is
coarse-grained independently.

The RG pipeline can use PQ codes to compute a "sub-space Hamiltonian":

```
H_eff_subspace(m, L) = thermodynamics on the m-th sub-vector of all nodes
```

If all sub-space Hamiltonians converge to the same fixed point, the system has
**isotropic criticality** (same physics in all embedding directions). If they
diverge, there is anisotropy -- some directions are "more ordered" than others.

### 6.4 Thermodynamics (Phase XIII)

The existing `ThermodynamicReport` provides H_eff(L0) directly. The RG pipeline
needs to generalize this computation to higher levels. The key functions to
reuse from `crates/nietzsche-agency/src/thermodynamics.rs`:

```rust
cognitive_temperature(mean_energy, energy_std) -> f64  // T = sigma/mu
shannon_entropy(energies: &[f32]) -> f64               // S = -sum(p*ln(p))
helmholtz_free_energy(U, T, S) -> f64                  // F = U - T*S
classify_phase(T, config) -> PhaseState                // {Solid, Liquid, Gas, Critical}
```

These functions are level-agnostic -- they operate on energy arrays. The RG
pipeline simply passes aggregated energy arrays at each level.

### 6.5 Observation Bridge (Phase XV.1)

The `ObservationFrame` from `crates/nietzsche-agency/src/observation.rs` already
has the serialization infrastructure for dashboard visualization. The RG report
should be surfaced through the same pipeline:

```
GET /api/agency/rg-flow?collection=X
```

Returns the RG flow data for dashboard rendering.

---

## 7. Implementation Plan

### 7.1 Architecture

```
RG Pipeline (periodic, not per-tick)
    |
    +-- [1] Sample L0 nodes (max 5000, configurable)
    |       Read embeddings + energies from GraphStorage
    |
    +-- [2] Compute L1 assignments
    |       VqEncoder::encode() for each sampled node
    |       Group nodes by VQ code index
    |
    +-- [3] Compute L1 H_eff
    |       Aggregate energies per code group
    |       Build code-level graph (inter-code edges)
    |       Compute thermodynamics on aggregated energies
    |
    +-- [4] Compute L2 assignments
    |       k-means on L1 codebook vectors (K=64)
    |       Group codes by super-code index
    |
    +-- [5] Compute L2 H_eff
    |       Same pattern as L1 but on super-code groups
    |
    +-- [6] Compute L3 assignments
    |       k-means on L2 centroids (K=8)
    |
    +-- [7] Compute L3 H_eff
    |
    +-- [8] Compute RG flow: Delta_H(L) for L=0,1,2
    |       Fixed point distance
    |       Critical exponents
    |
    +-- [9] Persist RG report to CF_META
    |       Key: "agency:rg_flow:{collection}"
    |
    +-- [10] Emit to ObservationBridge for dashboard
```

### 7.2 Data Structures

```rust
/// RG level effective Hamiltonian.
pub struct EffectiveHamiltonian {
    /// RG level (0 = microscopic, 3 = macroscopic).
    pub level: u8,
    /// Number of entities at this level.
    pub num_entities: usize,
    /// Cognitive temperature at this level.
    pub temperature: f64,
    /// Shannon entropy of energy distribution.
    pub entropy: f64,
    /// Helmholtz free energy.
    pub free_energy: f64,
    /// Mean energy per entity.
    pub mean_energy: f64,
    /// Energy standard deviation.
    pub energy_std: f64,
    /// Mean degree in the coarse-grained graph.
    pub mean_degree: f64,
    /// Clustering coefficient.
    pub clustering_coefficient: f64,
    /// Modularity of the coarse-grained graph.
    pub modularity: f64,
    /// Mean Poincare depth (||embedding||).
    pub mean_depth: f64,
    /// Depth variance.
    pub depth_variance: f64,
    /// Phase classification at this level.
    pub phase: PhaseState,
}

/// RG flow between consecutive levels.
pub struct RgFlow {
    /// Source level.
    pub from_level: u8,
    /// Target level.
    pub to_level: u8,
    /// Temperature change.
    pub delta_temperature: f64,
    /// Entropy change.
    pub delta_entropy: f64,
    /// Free energy change.
    pub delta_free_energy: f64,
    /// Modularity change.
    pub delta_modularity: f64,
    /// Depth change.
    pub delta_depth: f64,
    /// Normalized RG distance (composite metric).
    pub rg_distance: f64,
}

/// Complete RG analysis report.
pub struct RgReport {
    /// Collection name.
    pub collection: String,
    /// Timestamp.
    pub timestamp_ms: u64,
    /// H_eff at each level.
    pub levels: Vec<EffectiveHamiltonian>,
    /// RG flow between consecutive levels.
    pub flows: Vec<RgFlow>,
    /// Whether a fixed point was detected.
    pub fixed_point_detected: bool,
    /// Fixed point distance (min RG distance across levels).
    pub fixed_point_distance: f64,
    /// Estimated critical exponents.
    pub critical_exponents: CriticalExponents,
    /// Code cluster size distribution at L1 (for power-law check).
    pub cluster_size_distribution: Vec<(usize, usize)>,
    /// DSI confidence per level (order parameter proxy).
    pub dsi_confidences: Vec<f64>,
}

/// Extracted critical exponents.
pub struct CriticalExponents {
    /// Correlation length exponent.
    pub nu: Option<f64>,
    /// Energy scaling dimension.
    pub y_energy: Option<f64>,
    /// Fractal dimension.
    pub d_fractal: Option<f64>,
    /// Power-law exponent of cluster size distribution.
    pub alpha_cluster: Option<f64>,
}
```

### 7.3 Configuration

```rust
pub struct RgConfig {
    /// Maximum nodes to sample at L0 (default: 5000).
    pub max_sample: usize,
    /// Number of super-codes at L2 (default: 64).
    pub l2_clusters: usize,
    /// Number of meta-clusters at L3 (default: 8).
    pub l3_clusters: usize,
    /// k-means max iterations for L2/L3 clustering (default: 50).
    pub kmeans_max_iter: usize,
    /// k-means convergence threshold (default: 1e-5).
    pub kmeans_threshold: f64,
    /// Fixed point detection epsilon (default: 0.05).
    pub fixed_point_epsilon: f64,
    /// RG flow component weights.
    pub weight_temperature: f64,  // default: 1.0
    pub weight_entropy: f64,      // default: 1.0
    pub weight_free_energy: f64,  // default: 0.5
    pub weight_modularity: f64,   // default: 0.5
    pub weight_depth: f64,        // default: 0.3
}
```

### 7.4 Environment Variables

```env
AGENCY_RG_ENABLED=true
AGENCY_RG_INTERVAL=100           # every 100 agency ticks (~periodic, not per-tick)
AGENCY_RG_MAX_SAMPLE=5000
AGENCY_RG_L2_CLUSTERS=64
AGENCY_RG_L3_CLUSTERS=8
AGENCY_RG_FIXED_POINT_EPSILON=0.05
```

### 7.5 Scheduling

The RG analysis is computationally expensive (O(N * K) for VQ encoding, O(K^2)
for k-means at each level). It should run:

- **Not per-tick**: Too expensive. A single RG analysis on 5000 nodes with
  512 codebook vectors would take O(5000 * 512) = ~2.5M distance computations
  at L1 alone.
- **Every 100 agency ticks**: Approximately every ~10 minutes with typical
  tick rates.
- **On-demand via API**: `GET /api/agency/rg-flow?collection=X` triggers
  a fresh computation (with caching for recent results).

This is consistent with Phase 27 (Epistemic Evolution) which runs every 40 ticks --
the RG analysis at 100 ticks is even less frequent.

---

## 8. Dashboard Visualization

### 8.1 RG Flow Diagram

The primary visualization is a **flow diagram** showing H_eff at each level
connected by arrows indicating the direction and magnitude of change:

```
  L0 (N nodes)           L1 (512 codes)         L2 (64 super-codes)    L3 (8 meta)
 +-----------+          +-----------+          +-----------+          +---------+
 | T = 0.42  |--dT=-0.03-->| T = 0.39  |--dT=-0.01-->| T = 0.38  |--dT=+0.01->| T = 0.39|
 | S = 5.21  |          | S = 4.89  |          | S = 3.52  |          | S = 1.93|
 | F = -1.70 |          | F = -1.52 |          | F = -0.95 |          | F = -0.36|
 | Q = 0.32  |          | Q = 0.45  |          | Q = 0.51  |          | Q = 0.48|
 | d = 0.35  |          | d = 0.33  |          | d = 0.28  |          | d = 0.22|
 +-----------+          +-----------+          +-----------+          +---------+
                         RG dist=0.12           RG dist=0.08           RG dist=0.03
                                                                       ^^^ FIXED POINT?
```

### 8.2 Phase Portrait

A 2D scatter plot with axes:
- X: Temperature (T)
- Y: Entropy (S)

Each level is a point. The trajectory L0 -> L1 -> L2 -> L3 shows the RG flow
in (T, S) space. A fixed point appears as convergence to a single point.

### 8.3 Cluster Size Distribution (L1)

A log-log histogram of code cluster sizes at L1. If the distribution follows
a power law P(s) ~ s^(-alpha), the system is near criticality.

### 8.4 Critical Exponent Dashboard

A table showing:
- Extracted exponents (nu, y_E, d_f, alpha)
- Closest known universality class
- Confidence score (how well the exponents match)

### 8.5 API Endpoint

```
GET /api/agency/rg-flow?collection=X

Response: {
    "collection": "tech_galaxies",
    "timestamp_ms": 1741700000000,
    "levels": [
        { "level": 0, "num_entities": 5000, "temperature": 0.42, ... },
        { "level": 1, "num_entities": 512,  "temperature": 0.39, ... },
        { "level": 2, "num_entities": 64,   "temperature": 0.38, ... },
        { "level": 3, "num_entities": 8,    "temperature": 0.39, ... }
    ],
    "flows": [
        { "from_level": 0, "to_level": 1, "delta_temperature": -0.03, "rg_distance": 0.12, ... },
        { "from_level": 1, "to_level": 2, "delta_temperature": -0.01, "rg_distance": 0.08, ... },
        { "from_level": 2, "to_level": 3, "delta_temperature": +0.01, "rg_distance": 0.03, ... }
    ],
    "fixed_point_detected": true,
    "fixed_point_distance": 0.03,
    "critical_exponents": {
        "nu": 0.87,
        "y_energy": 1.23,
        "d_fractal": 2.14,
        "alpha_cluster": 2.1
    },
    "cluster_size_distribution": [[1, 45], [2, 38], [3, 22], ...],
    "dsi_confidences": [0.92, 0.78, 0.61, 0.43]
}
```

---

## 9. Theoretical Notes

### 9.1 Why VQ-VAE is a Good RG Operator

Traditional block-spin RG uses a fixed coarse-graining rule (majority vote, spatial
averaging). VQ-VAE learns an optimal coarse-graining from data:

1. **Information-theoretic optimality**: The VQ-VAE codebook minimizes reconstruction
   error (rate-distortion theory). This means the codebook captures the maximum
   information about the original embedding space in the fewest discrete symbols.

2. **Adaptive partitioning**: Unlike grid-based RG, VQ-VAE partitions the space
   based on the actual data distribution. Dense regions get more codes; sparse
   regions get fewer. This matches the physical intuition that RG should
   preserve information where it matters most.

3. **Magnitude preservation**: The PQ tests prove that the learned centroids
   preserve the magnitude ordering (shallow < mid < deep). In the Poincare ball,
   this means the RG transformation preserves the hierarchical depth structure.

### 9.2 Hyperbolic Geometry and RG

The Poincare ball has a natural notion of scale: distance from the origin. Points
near the origin are "coarse" (general concepts), while points near the boundary
are "fine" (specific details). This is precisely the RG hierarchy:

```
Origin (||x|| ~ 0)   <->  L3 (macroscopic, general)
Boundary (||x|| ~ 1) <->  L0 (microscopic, specific)
```

The RG flow in the Poincare ball should show:
- L0 entities have high mean depth (near boundary)
- L3 entities have low mean depth (near origin)
- The flow is "inward" in the ball

If the flow reverses at some level (entities move outward), it indicates a
breakdown of the hierarchical structure at that scale.

### 9.3 Relationship to Kadanoff Block Spins

Kadanoff's original block-spin idea (1966):
1. Divide the lattice into blocks of b^d sites
2. Replace each block with a single "block spin"
3. Rescale so the new system looks like the original

Our VQ-based RG:
1. VQ-VAE groups embeddings into 512 clusters (b ~ 8, d ~ 128)
2. Each cluster is represented by its codebook vector
3. The codebook vectors form a new "lattice" for the next level

The key difference: our lattice is the Poincare ball (hyperbolic, non-Euclidean),
and our "spins" are high-dimensional embeddings. The universality class may be
novel -- not matching any known condensed matter system.

### 9.4 Limitations

1. **Finite-size effects**: With N ~ 5000-14000 nodes per collection, finite-size
   scaling corrections may be significant. The RG flow may not converge cleanly.

2. **Non-equilibrium**: NietzscheDB is a driven system (new nodes are constantly
   inserted, edges are pruned). The RG analysis captures a snapshot, not
   equilibrium behavior. The relevant framework may be **driven dissipative RG**
   rather than equilibrium statistical mechanics.

3. **VQ-VAE model quality**: The RG transformation quality depends on the VQ-VAE
   model. A poorly trained model produces poor coarse-graining. The pipeline
   uses the existing trained model and does not retrain.

4. **Decimation ratio**: The ratio between levels (~8x) is somewhat arbitrary.
   A systematic study should vary this ratio and check for consistency.

---

## 10. Future Extensions (Out of Scope)

These are noted for completeness but are not part of this design:

1. **Real-space RG**: Use the Poincare ball distance directly for coarse-graining
   (spatial blocking based on Poincare neighborhoods) instead of VQ codes.

2. **Functional RG (Wetterich equation)**: Track the effective action as a function
   of the cutoff scale, providing a continuous RG flow rather than discrete levels.

3. **RG-informed agency**: If the system is found to be away from criticality, the
   agency engine could adjust parameters (thermal conductivity, decay rate) to
   drive it toward the critical point. This would make the RG analysis active
   rather than passive -- explicitly out of scope for this design.

4. **Temporal RG**: Coarse-grain in time as well as space, analyzing how the system
   evolves at different temporal scales.

5. **Tensor Network RG (MERA)**: Use the multi-scale entanglement renormalization
   ansatz for a more principled RG on the graph, potentially connecting to the
   quantum-inspired features in `nietzsche-agency/src/orch_or/`.

---

## 11. File Reference

| Component | File | Relevant Functions/Structs |
|-----------|------|---------------------------|
| VQ-VAE (Python) | `scripts/models/train_vqvae.py` | `VqVae`, `VectorQuantizer` (3072D -> 512D latent, 1024 codes) |
| VQ-VAE (Rust) | `crates/nietzsche-vqvae/src/model.rs` | `VqVaeConfig` (128D, 512 codes) |
| VQ Encoder | `crates/nietzsche-vqvae/src/encoder.rs` | `VqEncoder::encode()` -> discrete code index |
| VQ Decoder | `crates/nietzsche-vqvae/src/decoder.rs` | `VqDecoder::decode()` -> continuous embedding |
| DSI Indexer | `crates/nietzsche-dsi/src/indexer.rs` | `DsiIndexer::index_node()` -> hierarchical SemanticId |
| DSI Decoder | `crates/nietzsche-dsi/src/decoder.rs` | `DsiDecoderNet::decode()` -> 4 levels, 1024 codes |
| Semantic ID | `crates/nietzsche-dsi/src/id.rs` | `SemanticId(Vec<u16>)`, parent/prefix operations |
| PQ Codebook | `crates/nietzsche-pq/src/codebook.rs` | `Codebook::train()`, k-means, M sub-quantizers |
| PQ Encoder | `crates/nietzsche-pq/src/encoder.rs` | `PQEncoder::encode/decode()`, magnitude-preserving |
| Cognitive Layer | `crates/nietzsche-agency/src/cognitive_layer.rs` | `run_cognitive_scan()`, gyro-midpoint, radius clustering |
| Louvain | `crates/nietzsche-algo/src/community.rs` | `louvain()`, modularity optimization |
| Thermodynamics | `crates/nietzsche-agency/src/thermodynamics.rs` | `cognitive_temperature()`, `shannon_entropy()`, `helmholtz_free_energy()` |
| Observation | `crates/nietzsche-agency/src/observation.rs` | `ObservationFrame`, dashboard visualization |
| Dashboard | `crates/nietzsche-agency/src/cognitive_dashboard.rs` | `CognitiveDashboard`, `ThermoSnapshot` |
| ONNX Models | `models/vqvae.onnx` | Exported VQ-VAE for inference |
| VQ-VAE Checkpoint | `checkpoints/vqvae.pt` | PyTorch weights |
