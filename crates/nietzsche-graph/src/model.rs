use std::collections::HashMap;
use uuid::Uuid;
use serde::{Serialize, Deserialize};

// ─────────────────────────────────────────────
// Bincode ↔ serde_json::Value bridge
// ─────────────────────────────────────────────
//
// Bincode is not a self-describing format: it cannot handle
// serde_json::Value (which uses `deserialize_any`).  The workaround is to
// round-trip the JSON fields through a plain String inside the bincode
// envelope — JSON IS self-describing and can always round-trip Value.
//
// Usage: annotate problematic fields with
//   #[serde(with = "as_json_string")]
mod as_json_string {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<T, S>(value: &T, ser: S) -> Result<S::Ok, S::Error>
    where
        T: Serialize,
        S: Serializer,
    {
        let s = serde_json::to_string(value).map_err(serde::ser::Error::custom)?;
        s.serialize(ser)
    }

    pub fn deserialize<'de, T, D>(de: D) -> Result<T, D::Error>
    where
        T: for<'a> Deserialize<'a>,
        D: Deserializer<'de>,
    {
        let s = String::deserialize(de)?;
        serde_json::from_str(&s).map_err(serde::de::Error::custom)
    }
}

// ─────────────────────────────────────────────
// PoincaréVector
// ─────────────────────────────────────────────

/// A point in the Poincaré ball model of hyperbolic space.
/// Invariant: ‖coords‖ < 1.0 must hold at all times.
///
/// ## ITEM C (Committee 2026-02-19)
/// Coordinates stored as `Vec<f32>` (4 bytes/coord) instead of `Vec<f64>` (8 bytes).
/// For 3072 dimensions: 12 KB instead of 24 KB — **50% memory reduction**.
/// The distance kernel promotes to f64 internally to preserve numerical precision
/// near the boundary where `(1−‖x‖²)` can underflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoincareVector {
    pub coords: Vec<f32>,
    pub dim: usize,
}

impl PoincareVector {
    /// Create from f32 coordinates (primary path).
    pub fn new(coords: Vec<f32>) -> Self {
        let dim = coords.len();
        Self { coords, dim }
    }

    /// Create from f64 coordinates (convenience for API boundaries / math results).
    /// Narrows each coordinate from f64 → f32.
    pub fn from_f64(coords: Vec<f64>) -> Self {
        let f32_coords: Vec<f32> = coords.iter().map(|&x| x as f32).collect();
        Self::new(f32_coords)
    }

    /// Return coordinates promoted to f64 (for high-precision math operations).
    #[inline]
    pub fn coords_f64(&self) -> Vec<f64> {
        self.coords.iter().map(|&x| x as f64).collect()
    }

    /// Euclidean norm of the coordinate vector (computed in f64 for precision).
    #[inline]
    pub fn norm(&self) -> f64 {
        self.coords.iter().map(|&x| { let xf = x as f64; xf * xf }).sum::<f64>().sqrt()
    }

    /// Returns true iff the point is strictly inside the unit ball.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.norm() < 1.0
    }

    /// Hyperbolic distance in the Poincaré ball model.
    ///
    /// d(u, v) = acosh(1 + 2‖u−v‖² / ((1−‖u‖²)(1−‖v‖²)))
    ///
    /// ## Performance
    /// Computes all three sums (diff_sq, norm_u_sq, norm_v_sq) in **one pass**
    /// over the coordinate arrays.  This improves cache utilization and allows
    /// the compiler to auto-vectorize with SIMD (AVX2 / SSE4.2) when the target
    /// has those features enabled (`RUSTFLAGS="-C target-cpu=native"`).
    ///
    /// ## Precision (ITEM C)
    /// Coordinates are `f32` but the kernel promotes to `f64` internally.
    /// This prevents catastrophic cancellation in `(1−‖x‖²)` for nodes
    /// near the Poincaré ball boundary (‖x‖ > 0.99).
    #[inline]
    pub fn distance(&self, other: &Self) -> f64 {
        debug_assert_eq!(self.dim, other.dim, "dimension mismatch");

        let (diff_sq, norm_u_sq, norm_v_sq) = poincare_sums(&self.coords, &other.coords);

        let denom = (1.0 - norm_u_sq) * (1.0 - norm_v_sq);

        // Clamp to avoid NaN from floating-point drift near the boundary
        let arg = (1.0 + 2.0 * diff_sq / denom).max(1.0);
        arg.acosh()
    }

    /// Squared Euclidean distance (fast path for pre-filtering before acosh).
    /// Cheaper than `distance()` — avoids the acosh and the norm computations.
    #[inline]
    pub fn sq_euclidean(&self, other: &Self) -> f64 {
        debug_assert_eq!(self.dim, other.dim, "dimension mismatch");
        self.coords.iter()
            .zip(other.coords.iter())
            .map(|(a, b)| { let d = (*a as f64) - (*b as f64); d * d })
            .sum()
    }

    /// Project a vector that has drifted outside or near the boundary of the ball.
    ///
    /// Two-stage clamp:
    /// - **Soft clamp**: if ‖x‖ > 0.999, scale to 0.999.
    ///   Prevents catastrophic denominator underflow in `(1−‖x‖²)` during long
    ///   training runs where episodic nodes accumulate near the boundary.
    /// - **Hard clamp**: if ‖x‖ ≥ 1.0 (outside the ball entirely), also caught
    ///   by the soft-clamp threshold above — no separate branch needed.
    ///
    /// Recommendation from Grok / Hyperbolic Invariant Analysis (2026-02-19).
    pub fn project_into_ball(mut self) -> Self {
        let n = self.norm();
        if n > 0.999 {
            let scale = (0.999 / (n + 1e-10)) as f32;
            for c in self.coords.iter_mut() {
                *c *= scale;
            }
        }
        self
    }

    /// Origin of the Poincaré ball in `dim` dimensions.
    pub fn origin(dim: usize) -> Self {
        Self { coords: vec![0.0f32; dim], dim }
    }

    /// Depth heuristic: how far from the center (0 = center, ~1 = boundary).
    /// Returns the raw norm — callers can scale to [0, 1).
    #[inline]
    pub fn depth(&self) -> f64 {
        self.norm()
    }
}

// ─────────────────────────────────────────────
// Poincaré distance inner kernel
// ─────────────────────────────────────────────

/// Compute (diff_sq, norm_u_sq, norm_v_sq) in **one pass** over u and v.
///
/// ## ITEM C (Committee 2026-02-19)
/// Input coordinates are `f32` but all accumulation is in `f64` to prevent
/// catastrophic cancellation near the Poincaré ball boundary. Each `f32`
/// is promoted to `f64` before the multiply-add, preserving the full
/// precision needed for `(1−‖x‖²)` when `‖x‖ > 0.99`.
///
/// The loop body has no data dependency between iterations, making it
/// trivially vectorizable.  When compiled with `-C target-cpu=native` the
/// compiler emits mixed `vmovss` (load f32) + `vcvtss2sd` (promote) +
/// `vfmadd` (f64 FMA) instructions.
#[inline(always)]
fn poincare_sums(u: &[f32], v: &[f32]) -> (f64, f64, f64) {
    let mut diff_sq   = 0.0f64;
    let mut norm_u_sq = 0.0f64;
    let mut norm_v_sq = 0.0f64;

    let n = u.len().min(v.len());
    let u = &u[..n];
    let v = &v[..n];

    for i in 0..n {
        let a = u[i] as f64;
        let b = v[i] as f64;
        let d = a - b;
        diff_sq   += d * d;
        norm_u_sq += a * a;
        norm_v_sq += b * b;
    }

    (diff_sq, norm_u_sq, norm_v_sq)
}

// ─────────────────────────────────────────────
// SparseVector (SPLADE / sparse embeddings)
// ─────────────────────────────────────────────

/// A sparse vector representation suitable for SPLADE-style embeddings.
///
/// Only non-zero dimensions are stored, using a pair of parallel arrays:
/// - `indices`: sorted, unique dimension indices (`u32` to save space).
/// - `values`:  the corresponding `f32` weights (same length as `indices`).
///
/// `dim` records the *logical* dimensionality of the full vector (e.g. 30522
/// for a BERT-vocabulary SPLADE model).  It is not required for arithmetic
/// but is useful for validation and display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    /// Logical dimensionality of the full (dense) vector space.
    pub dim: usize,
    /// Sorted, unique indices of non-zero dimensions.
    pub indices: Vec<u32>,
    /// Values corresponding 1-to-1 with `indices`.
    pub values: Vec<f32>,
}

impl SparseVector {
    /// Construct a new `SparseVector`.
    ///
    /// # Panics
    /// - If `indices.len() != values.len()`.
    /// - If `indices` contains duplicates or is not sorted ascending.
    /// - If any index is `>= dim`.
    pub fn new(dim: usize, indices: Vec<u32>, values: Vec<f32>) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "indices and values must have the same length"
        );
        // Verify sorted-unique invariant
        for w in indices.windows(2) {
            assert!(
                w[0] < w[1],
                "indices must be strictly ascending (found {} >= {})",
                w[0],
                w[1]
            );
        }
        if let Some(&last) = indices.last() {
            assert!(
                (last as usize) < dim,
                "index {} is out of bounds for dim {}",
                last,
                dim
            );
        }
        Self { dim, indices, values }
    }

    /// Number of stored (non-zero) entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Density = nnz / dim.  Returns 0.0 when `dim == 0`.
    #[inline]
    pub fn density(&self) -> f64 {
        if self.dim == 0 {
            return 0.0;
        }
        self.nnz() as f64 / self.dim as f64
    }

    /// Sparse dot product: sum of `a_i * b_i` over shared indices.
    ///
    /// Runs in O(nnz_a + nnz_b) using a merge-join on the sorted index arrays.
    pub fn dot(&self, other: &Self) -> f32 {
        let mut sum = 0.0f64; // accumulate in f64 then narrow
        let (mut i, mut j) = (0usize, 0usize);
        let (ai, av) = (&self.indices, &self.values);
        let (bi, bv) = (&other.indices, &other.values);

        while i < ai.len() && j < bi.len() {
            match ai[i].cmp(&bi[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    sum += (av[i] as f64) * (bv[j] as f64);
                    i += 1;
                    j += 1;
                }
            }
        }
        sum as f32
    }

    /// L2 norm: sqrt(sum of values^2).
    #[inline]
    pub fn norm(&self) -> f32 {
        let sq_sum: f64 = self.values.iter().map(|&v| (v as f64) * (v as f64)).sum();
        sq_sum.sqrt() as f32
    }

    /// Cosine similarity between two sparse vectors.
    ///
    /// Returns 0.0 if either vector has zero norm (to avoid NaN).
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let d = self.dot(other);
        let na = self.norm();
        let nb = other.norm();
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        d / (na * nb)
    }
}

// ─────────────────────────────────────────────
// NodeType / EdgeType
// ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    /// Episodic memory — specific event, lives near the boundary (high depth).
    Episodic,
    /// Semantic memory — abstract concept, lives near the center (low depth).
    Semantic,
    /// Pure concept node with no episodic grounding.
    Concept,
    /// Snapshot captured during a sleep/reconsolidation cycle.
    DreamSnapshot,
}

impl Default for NodeType {
    fn default() -> Self {
        Self::Semantic
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// Standard associative link.
    Association,
    /// Created by an L-System production rule.
    LSystemGenerated,
    /// Hierarchical parent → child relationship.
    Hierarchical,
    /// Archived (not deleted) — marked during pruning.
    Pruned,
}

impl Default for EdgeType {
    fn default() -> Self {
        Self::Association
    }
}

// ─────────────────────────────────────────────
// NodeMeta — lightweight metadata (no embedding)
// ─────────────────────────────────────────────

/// Lightweight node metadata — everything except the embedding vector.
///
/// Stored in the `CF_NODES` column family (~100 bytes per node).
/// BFS, energy updates, Hausdorff updates, and most NQL filters only need
/// this struct — they never touch the 24 KB embedding.
///
/// # BUG A fix (Committee 2026-02-19)
/// Previously the full `Node` (including the ~24 KB embedding) was stored in a
/// single CF and deserialized on every `get_node()` call.  Splitting into
/// `NodeMeta` + separate `CF_EMBEDDINGS` gives 10–25× speedup in traversal
/// and reduces RAM usage in the hot-tier cache by ~240× per node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMeta {
    /// Unique identifier (UUIDv4).
    pub id: Uuid,

    /// Depth proxy = ‖embedding‖ ∈ [0, 1).
    /// Low depth → abstract / semantic (near center).
    /// High depth → specific / episodic (near boundary).
    pub depth: f32,

    /// Arbitrary JSON content stored with the node.
    #[serde(with = "as_json_string")]
    pub content: serde_json::Value,

    /// Semantic category of this node.
    pub node_type: NodeType,

    /// Energy level ∈ [0.0, 1.0]. Decays over time; at 0.0 the node is prunable.
    pub energy: f32,

    /// Which L-System generation created this node (0 = manually inserted).
    pub lsystem_generation: u32,

    /// Local Hausdorff dimension of the neighborhood (computed lazily).
    /// Range [0, 2]. Nodes with D < 0.5 or D > 1.9 are candidates for pruning.
    pub hausdorff_local: f32,

    /// Unix timestamp (seconds) of creation.
    pub created_at: i64,

    /// Optional TTL: Unix timestamp (seconds) after which this node expires.
    /// `None` means the node never expires.
    /// Expired nodes are candidates for garbage collection by the JanitorTask.
    #[serde(default)]
    pub expires_at: Option<i64>,

    /// Arbitrary key→value metadata.
    #[serde(with = "as_json_string")]
    pub metadata: HashMap<String, serde_json::Value>,

    /// Emotional valence ∈ [-1.0, 1.0].
    /// Negative = punishing/traumatic, Positive = rewarding/pleasant,
    /// Zero = neutral. Affects how heat propagation (DIFFUSE) weights edges.
    #[serde(default)]
    pub valence: f32,

    /// Emotional arousal ∈ [0.0, 1.0].
    /// High = emotionally intense, Low = calm/neutral.
    /// Amplifies energy_bias in diffusion_walk: heat travels faster
    /// through emotionally charged memories than neutral facts.
    #[serde(default)]
    pub arousal: f32,

    /// Phantom node flag: structural "scar" left after pruning/TTL expiry.
    ///
    /// Phantom nodes retain their topological connections (edges, adjacency)
    /// so the hyperbolic geometry doesn't collapse, but are excluded from
    /// KNN search and active traversal. This mimics how the brain preserves
    /// structural memory traces that facilitate re-learning.
    #[serde(default)]
    pub is_phantom: bool,
}

impl NodeMeta {
    /// Returns true if this node should be considered for pruning.
    pub fn is_prunable(&self) -> bool {
        self.energy <= 0.0
            || self.hausdorff_local < 0.5
            || self.hausdorff_local > 1.9
    }

    /// Returns true if this node has expired based on the current time.
    pub fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(exp) => now_unix() >= exp,
            None => false,
        }
    }

    /// Returns true if this node is a phantom (structural scar).
    pub fn is_phantom(&self) -> bool {
        self.is_phantom
    }
}

// ─────────────────────────────────────────────
// Node — full record (metadata + embedding)
// ─────────────────────────────────────────────

/// A node in the NietzscheDB hyperbolic knowledge graph.
///
/// Composed of [`NodeMeta`] (lightweight, ~100 bytes) and a [`PoincareVector`]
/// embedding (~24 KB at 3072 dims).  Use `NodeMeta` directly when you don't
/// need the embedding (BFS, energy updates, NQL filters).
///
/// `Node` implements `Deref<Target = NodeMeta>` so all metadata fields are
/// accessible directly (e.g. `node.energy`, `node.id`) without breaking
/// existing code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Lightweight metadata (id, energy, depth, content, etc.).
    pub meta: NodeMeta,

    /// Position in the Poincaré ball. Invariant: ‖embedding‖ < 1.0.
    pub embedding: PoincareVector,
}

/// Allows `node.energy`, `node.id`, etc. to work without `.meta.` prefix.
impl std::ops::Deref for Node {
    type Target = NodeMeta;
    #[inline]
    fn deref(&self) -> &NodeMeta {
        &self.meta
    }
}

/// Allows `node.energy = 0.5` etc. to work without `.meta.` prefix.
impl std::ops::DerefMut for Node {
    #[inline]
    fn deref_mut(&mut self) -> &mut NodeMeta {
        &mut self.meta
    }
}

impl From<(NodeMeta, PoincareVector)> for Node {
    fn from((meta, embedding): (NodeMeta, PoincareVector)) -> Self {
        Self { meta, embedding }
    }
}

impl Node {
    /// Construct a new node with required fields; defaults the rest.
    pub fn new(id: Uuid, embedding: PoincareVector, content: serde_json::Value) -> Self {
        assert!(embedding.is_valid(), "embedding must satisfy ‖x‖ < 1.0");
        let depth = embedding.depth() as f32;
        Self {
            meta: NodeMeta {
                id,
                depth,
                content,
                node_type: NodeType::default(),
                energy: 1.0,
                lsystem_generation: 0,
                hausdorff_local: 1.0,
                created_at: now_unix(),
                expires_at: None,
                metadata: HashMap::new(),
                valence: 0.0,
                arousal: 0.0,
                is_phantom: false,
            },
            embedding,
        }
    }

    /// Split into metadata and embedding.
    pub fn into_parts(self) -> (NodeMeta, PoincareVector) {
        (self.meta, self.embedding)
    }

    /// Returns true if this node should be considered for pruning.
    pub fn is_prunable(&self) -> bool {
        self.meta.is_prunable()
    }
}

// ─────────────────────────────────────────────
// Edge
// ─────────────────────────────────────────────

/// A directed edge between two nodes in the hyperbolic graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Unique identifier (UUIDv4).
    pub id: Uuid,

    /// Source node.
    pub from: Uuid,

    /// Target node.
    pub to: Uuid,

    /// Semantic type of this edge.
    pub edge_type: EdgeType,

    /// Edge weight ∈ [0.0, 1.0]. Used in traversal cost functions.
    pub weight: f32,

    /// Name of the L-System rule that created this edge, if any.
    pub lsystem_rule: Option<String>,

    /// Unix timestamp (seconds) of creation.
    pub created_at: i64,

    /// Arbitrary key→value metadata.
    #[serde(with = "as_json_string")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Edge {
    pub fn new(from: Uuid, to: Uuid, edge_type: EdgeType, weight: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            from,
            to,
            edge_type,
            weight,
            lsystem_rule: None,
            created_at: now_unix(),
            metadata: HashMap::new(),
        }
    }

    pub fn association(from: Uuid, to: Uuid, weight: f32) -> Self {
        Self::new(from, to, EdgeType::Association, weight)
    }

    pub fn hierarchical(parent: Uuid, child: Uuid) -> Self {
        Self::new(parent, child, EdgeType::Hierarchical, 1.0)
    }
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

fn now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn vec2(x: f32, y: f32) -> PoincareVector {
        PoincareVector::new(vec![x, y])
    }

    #[test]
    fn poincare_origin_is_valid() {
        assert!(PoincareVector::origin(4).is_valid());
    }

    #[test]
    fn poincare_outside_ball_is_invalid() {
        let v = PoincareVector::new(vec![0.8, 0.8]);
        assert!(!v.is_valid()); // norm ≈ 1.13
    }

    #[test]
    fn poincare_distance_zero_to_self() {
        let v = vec2(0.3, 0.4);
        assert!(v.distance(&v) < 1e-6);
    }

    #[test]
    fn poincare_distance_is_symmetric() {
        let u = vec2(0.1, 0.2);
        let v = vec2(0.3, -0.1);
        let d_uv = u.distance(&v);
        let d_vu = v.distance(&u);
        assert!((d_uv - d_vu).abs() < 1e-10, "d(u,v)={d_uv} ≠ d(v,u)={d_vu}");
    }

    #[test]
    fn poincare_hierarchy_depth_ordering() {
        // A point closer to the boundary should have greater depth
        let center = vec2(0.05, 0.05);   // depth ≈ 0.07
        let middle = vec2(0.3, 0.3);    // depth ≈ 0.42
        let boundary = vec2(0.6, 0.6);  // depth ≈ 0.85

        assert!(center.depth() < middle.depth());
        assert!(middle.depth() < boundary.depth());
    }

    #[test]
    fn poincare_distance_respects_hierarchy() {
        // Hierarchical triple: abstract → mid → specific
        // d(abstract, mid) < d(abstract, specific)
        let abstract_node = vec2(0.05, 0.0);
        let mid_node      = vec2(0.3, 0.0);
        let specific_node = vec2(0.7, 0.0);

        let d_ab_mid = abstract_node.distance(&mid_node);
        let d_ab_spe = abstract_node.distance(&specific_node);

        assert!(
            d_ab_mid < d_ab_spe,
            "expected d(abstract,mid)={d_ab_mid:.4} < d(abstract,specific)={d_ab_spe:.4}"
        );
    }

    #[test]
    fn poincare_projection_brings_inside_ball() {
        let outside = PoincareVector::new(vec![0.9, 0.9]); // norm > 1
        assert!(!outside.is_valid());
        let projected = outside.project_into_ball();
        assert!(projected.is_valid());
    }

    #[test]
    fn poincare_from_f64_roundtrip() {
        let original = vec![0.3_f64, 0.4, -0.1];
        let pv = PoincareVector::from_f64(original.clone());
        assert_eq!(pv.dim, 3);
        assert!(pv.is_valid());
        // f64→f32 narrows, verify invariant preserved
        let back = pv.coords_f64();
        for (a, b) in original.iter().zip(back.iter()) {
            assert!((*a as f32 - *b as f32).abs() < 1e-7);
        }
    }

    #[test]
    fn poincare_f32_invariant_near_boundary() {
        // Verify ‖x‖ < 1.0 is preserved through f64→f32 round-trip
        let v = PoincareVector::from_f64(vec![0.7, 0.7]); // norm ~0.99
        assert!(v.is_valid(), "invariant must hold after f64→f32 narrow");
    }

    #[test]
    fn node_new_sets_depth_from_embedding() {
        let emb = vec2(0.3, 0.4); // norm = 0.5
        let node = Node::new(Uuid::new_v4(), emb.clone(), serde_json::json!({}));
        assert!((node.depth - emb.depth() as f32).abs() < 1e-6);
    }

    #[test]
    fn node_rejects_invalid_embedding() {
        // Should panic because ‖x‖ >= 1.0
        let result = std::panic::catch_unwind(|| {
            Node::new(
                Uuid::new_v4(),
                PoincareVector::new(vec![0.8, 0.8]),
                serde_json::json!({}),
            )
        });
        assert!(result.is_err(), "Node::new must panic on invalid embedding");
    }

    #[test]
    fn edge_new_has_unique_ids() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let e1 = Edge::association(a, b, 1.0);
        let e2 = Edge::association(a, b, 1.0);
        assert_ne!(e1.id, e2.id);
    }

    #[test]
    fn node_is_prunable_at_zero_energy() {
        let mut node = Node::new(Uuid::new_v4(), vec2(0.1, 0.1), serde_json::json!({}));
        assert!(!node.is_prunable());
        node.energy = 0.0;
        assert!(node.is_prunable());
    }

    #[test]
    fn serde_roundtrip_node() {
        let node = Node::new(Uuid::new_v4(), vec2(0.2, 0.3), serde_json::json!({"text": "hello"}));
        let encoded = bincode::serialize(&node).expect("serialize");
        let decoded: Node = bincode::deserialize(&encoded).expect("deserialize");
        assert_eq!(node.id, decoded.id);
        assert_eq!(node.embedding.coords, decoded.embedding.coords);
    }

    #[test]
    fn serde_roundtrip_edge() {
        let e = Edge::hierarchical(Uuid::new_v4(), Uuid::new_v4());
        let encoded = bincode::serialize(&e).expect("serialize");
        let decoded: Edge = bincode::deserialize(&encoded).expect("deserialize");
        assert_eq!(e.id, decoded.id);
        assert_eq!(e.edge_type, decoded.edge_type);
    }

    // ── SparseVector tests ──────────────────────

    #[test]
    fn sparse_new_basic() {
        let sv = SparseVector::new(100, vec![2, 10, 50], vec![1.0, 2.0, 3.0]);
        assert_eq!(sv.dim, 100);
        assert_eq!(sv.nnz(), 3);
    }

    #[test]
    fn sparse_density() {
        let sv = SparseVector::new(1000, vec![0, 999], vec![1.0, 1.0]);
        let d = sv.density();
        assert!((d - 0.002).abs() < 1e-9, "expected 0.002, got {d}");
    }

    #[test]
    fn sparse_density_empty_dim() {
        let sv = SparseVector::new(0, vec![], vec![]);
        assert_eq!(sv.density(), 0.0);
    }

    #[test]
    fn sparse_dot_disjoint() {
        let a = SparseVector::new(100, vec![0, 2, 4], vec![1.0, 2.0, 3.0]);
        let b = SparseVector::new(100, vec![1, 3, 5], vec![4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), 0.0);
    }

    #[test]
    fn sparse_dot_overlapping() {
        // shared indices: 2, 5
        let a = SparseVector::new(100, vec![2, 5, 8], vec![1.0, 3.0, 5.0]);
        let b = SparseVector::new(100, vec![2, 4, 5], vec![2.0, 4.0, 6.0]);
        // dot = 1*2 + 3*6 = 2 + 18 = 20
        let d = a.dot(&b);
        assert!((d - 20.0).abs() < 1e-6, "expected 20.0, got {d}");
    }

    #[test]
    fn sparse_dot_identical() {
        let a = SparseVector::new(10, vec![1, 3, 7], vec![2.0, 4.0, 6.0]);
        // dot with self = 4 + 16 + 36 = 56
        let d = a.dot(&a);
        assert!((d - 56.0).abs() < 1e-5, "expected 56.0, got {d}");
    }

    #[test]
    fn sparse_norm() {
        let sv = SparseVector::new(10, vec![0, 1], vec![3.0, 4.0]);
        let n = sv.norm();
        assert!((n - 5.0).abs() < 1e-6, "expected 5.0, got {n}");
    }

    #[test]
    fn sparse_norm_empty() {
        let sv = SparseVector::new(10, vec![], vec![]);
        assert_eq!(sv.norm(), 0.0);
    }

    #[test]
    fn sparse_cosine_identical() {
        let a = SparseVector::new(100, vec![0, 5, 99], vec![1.0, 2.0, 3.0]);
        let cs = a.cosine_similarity(&a);
        assert!((cs - 1.0).abs() < 1e-6, "expected 1.0, got {cs}");
    }

    #[test]
    fn sparse_cosine_orthogonal() {
        let a = SparseVector::new(100, vec![0, 1], vec![1.0, 0.0]);
        let b = SparseVector::new(100, vec![2, 3], vec![0.0, 1.0]);
        assert_eq!(a.cosine_similarity(&b), 0.0);
    }

    #[test]
    fn sparse_cosine_zero_norm() {
        let a = SparseVector::new(10, vec![], vec![]);
        let b = SparseVector::new(10, vec![0], vec![1.0]);
        assert_eq!(a.cosine_similarity(&b), 0.0);
    }

    #[test]
    fn sparse_cosine_known_value() {
        // a = [1, 2, 0], b = [2, 0, 3]  (only stored non-zero)
        let a = SparseVector::new(3, vec![0, 1], vec![1.0, 2.0]);
        let b = SparseVector::new(3, vec![0, 2], vec![2.0, 3.0]);
        // dot = 1*2 = 2
        // norm_a = sqrt(5), norm_b = sqrt(13)
        // cosine = 2 / sqrt(65) ≈ 0.24806946
        let cs = a.cosine_similarity(&b);
        let expected = 2.0_f32 / 65.0_f32.sqrt();
        assert!(
            (cs - expected).abs() < 1e-5,
            "expected {expected}, got {cs}"
        );
    }

    #[test]
    #[should_panic(expected = "indices and values must have the same length")]
    fn sparse_panics_length_mismatch() {
        SparseVector::new(10, vec![0, 1], vec![1.0]);
    }

    #[test]
    #[should_panic(expected = "strictly ascending")]
    fn sparse_panics_unsorted_indices() {
        SparseVector::new(10, vec![5, 2], vec![1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "strictly ascending")]
    fn sparse_panics_duplicate_indices() {
        SparseVector::new(10, vec![3, 3], vec![1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn sparse_panics_index_out_of_bounds() {
        SparseVector::new(10, vec![0, 10], vec![1.0, 2.0]);
    }

    #[test]
    fn sparse_serde_roundtrip() {
        let sv = SparseVector::new(30522, vec![100, 500, 10000, 30000], vec![0.5, 1.2, 0.8, 2.1]);
        let encoded = bincode::serialize(&sv).expect("serialize");
        let decoded: SparseVector = bincode::deserialize(&encoded).expect("deserialize");
        assert_eq!(sv.dim, decoded.dim);
        assert_eq!(sv.indices, decoded.indices);
        assert_eq!(sv.values, decoded.values);
    }
}
