// ─────────────────────────────────────────────
// NQL Abstract Syntax Tree  (v2)
// ─────────────────────────────────────────────

use serde::{Serialize, Deserialize};

/// Top-level NQL query.
#[derive(Debug, Clone)]
pub enum Query {
    Match(MatchQuery),
    Diffuse(DiffuseQuery),
    /// Phase 11: reconstruct sensory data from a node's latent vector.
    Reconstruct(ReconstructQuery),
    /// `EXPLAIN <inner query>` — return the execution plan without running it.
    Explain(Box<Query>),
    /// Phase C: `INVOKE ZARATUSTRA [IN "col"] [CYCLES n] [ALPHA f] [DECAY f]`
    InvokeZaratustra(InvokeZaratustraQuery),
    /// Phase F: `BEGIN` — start a new transaction.
    BeginTx,
    /// Phase F: `COMMIT` — commit the active transaction.
    CommitTx,
    /// Phase F: `ROLLBACK` — rollback the active transaction.
    RollbackTx,
    /// `MERGE (pattern) [ON CREATE SET …] [ON MATCH SET …] [RETURN …]`
    Merge(MergeQuery),
    /// `CREATE (n:Label {key: val, …}) [RETURN …]`
    Create(CreateQuery),
    /// `MATCH (n) [WHERE …] SET n.field = val [RETURN …]`
    MatchSet(MatchSetQuery),
    /// `MATCH (n) [WHERE …] DELETE n`
    MatchDelete(MatchDeleteQuery),
    /// `CREATE DAEMON name ON (n:Label) WHEN … THEN … EVERY … ENERGY …`
    CreateDaemon(CreateDaemonQuery),
    /// `DROP DAEMON name`
    DropDaemon(DropDaemonQuery),
    /// `SHOW DAEMONS`
    ShowDaemons,
    // ── Dream Queries (Phase 15.2) ──────────────────────
    /// `DREAM FROM $node [DEPTH n] [NOISE f]`
    DreamFrom(DreamFromQuery),
    /// `APPLY DREAM $dream_id`
    ApplyDream(ApplyDreamQuery),
    /// `REJECT DREAM $dream_id`
    RejectDream(RejectDreamQuery),
    /// `SHOW DREAMS`
    ShowDreams,
    // ── Synesthesia (Phase 15.3) ────────────────────────
    /// `TRANSLATE $node FROM modality TO modality [QUALITY q]`
    Translate(TranslateQuery),
    // ── Eternal Return (Phase 15.4) ─────────────────────
    /// `COUNTERFACTUAL SET … MATCH … WHERE … RETURN …`
    Counterfactual(CounterfactualQuery),
    // ── Collective Unconscious (Phase 15.6) ─────────────
    /// `SHOW ARCHETYPES`
    ShowArchetypes,
    /// `SHARE ARCHETYPE $node_id TO "collection"`
    ShareArchetype(ShareArchetypeQuery),
    // ── Narrative Engine (Phase 15.7) ───────────────────
    /// `NARRATE [IN "col"] [WINDOW n] [FORMAT fmt]`
    Narrate(NarrateQuery),
    // ── Psychoanalyze (Lineage) ──────────────────────────
    /// `PSYCHOANALYZE $node` — return the evolutionary lineage of a node.
    Psychoanalyze(PsychoanalyzeQuery),
}

// ── PSYCHOANALYZE ────────────────────────────────────────────

/// `PSYCHOANALYZE $node_id`
///
/// Returns the evolutionary lineage of a concept node: creation source,
/// L-System generation, energy trajectory, sleep/dream events, merges,
/// and structural role in the Poincaré ball.
#[derive(Debug, Clone)]
pub struct PsychoanalyzeQuery {
    /// Target node — `$param` or a bound alias.
    pub target: ReconstructTarget,
}

// ── CREATE / SET / DELETE ────────────────────────────────────

/// `CREATE (n:Label {key: value, …}) [RETURN …]`
#[derive(Debug, Clone)]
pub struct CreateQuery {
    pub alias:      String,
    pub label:      Option<String>,
    pub properties: Vec<(String, Expr)>,
    pub ret:        Option<ReturnClause>,
}

/// `MATCH (n) [WHERE …] SET n.x = val, … [RETURN …]`
#[derive(Debug, Clone)]
pub struct MatchSetQuery {
    pub pattern:     Pattern,
    pub conditions:  Vec<Condition>,
    pub assignments: Vec<SetAssignment>,
    pub ret:         Option<ReturnClause>,
}

/// `MATCH (n) [WHERE …] DELETE n, …` or `MATCH (n) [WHERE …] DETACH DELETE n`
#[derive(Debug, Clone)]
pub struct MatchDeleteQuery {
    pub pattern:    Pattern,
    pub conditions: Vec<Condition>,
    pub targets:    Vec<String>,
    /// `true` for `DETACH DELETE` — removes node + all incident edges.
    pub detach:     bool,
}

// ── DAEMON Agents ────────────────────────────────────────

/// Action a daemon executes when its WHEN condition fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonAction {
    /// `DELETE n`
    Delete { alias: String },
    /// `SET n.field = val, …`
    Set { assignments: Vec<SetAssignment> },
    /// `DIFFUSE FROM n WITH t=[…] MAX_HOPS k`
    Diffuse { alias: String, t_values: Vec<f64>, max_hops: usize },
}

/// `CREATE DAEMON name ON (n:Label) WHEN cond THEN action EVERY interval ENERGY e`
#[derive(Debug, Clone)]
pub struct CreateDaemonQuery {
    pub name:        String,
    pub on_pattern:  NodePattern,
    pub when_cond:   Condition,
    pub then_action: DaemonAction,
    /// Interval expression (e.g. `INTERVAL("1h")`) stored as an Expr.
    pub every:       Expr,
    /// Initial energy budget (0.0–1.0). Defaults to 1.0.
    pub energy:      Option<f64>,
}

/// `DROP DAEMON name`
#[derive(Debug, Clone)]
pub struct DropDaemonQuery {
    pub name: String,
}

// ── Dream Queries (Phase 15.2) ────────────────────────────

/// `DREAM FROM $node [DEPTH n] [NOISE f]`
#[derive(Debug, Clone)]
pub struct DreamFromQuery {
    pub seed:  DiffuseFrom,
    pub depth: Option<usize>,
    pub noise: Option<f64>,
}

/// `APPLY DREAM $dream_id`
#[derive(Debug, Clone)]
pub struct ApplyDreamQuery {
    pub dream_id: String,
}

/// `REJECT DREAM $dream_id`
#[derive(Debug, Clone)]
pub struct RejectDreamQuery {
    pub dream_id: String,
}

// ── Synesthesia (Phase 15.3) ──────────────────────────────

/// `TRANSLATE $node FROM modality TO modality [QUALITY q]`
#[derive(Debug, Clone)]
pub struct TranslateQuery {
    pub target:        ReconstructTarget,
    pub from_modality: String,
    pub to_modality:   String,
    pub quality:       Option<String>,
}

// ── Eternal Return — Counterfactual (Phase 15.4) ──────────

/// `COUNTERFACTUAL SET alias.field = val, … MATCH (n) WHERE … RETURN …`
#[derive(Debug, Clone)]
pub struct CounterfactualQuery {
    pub overlays: Vec<SetAssignment>,
    pub inner:    MatchQuery,
}

// ── Collective Unconscious (Phase 15.6) ───────────────────

/// `SHARE ARCHETYPE $node_id TO "collection"`
#[derive(Debug, Clone)]
pub struct ShareArchetypeQuery {
    pub node_param:        String,
    pub target_collection: String,
}

// ── Narrative Engine (Phase 15.7) ─────────────────────────

/// `NARRATE [IN "col"] [WINDOW hours] [FORMAT fmt]`
#[derive(Debug, Clone)]
pub struct NarrateQuery {
    pub collection:   Option<String>,
    pub window_hours: Option<u64>,
    pub format:       Option<String>,
}

// ── INVOKE ZARATUSTRA (Phase C) ───────────────────────────

/// `INVOKE ZARATUSTRA [IN "collection"] [CYCLES n] [ALPHA f] [DECAY f]`
#[derive(Debug, Clone)]
pub struct InvokeZaratustraQuery {
    /// Override the collection. `None` → use the server's active collection.
    pub collection: Option<String>,
    /// Number of engine cycles to run. `None` → default (1).
    pub cycles: Option<u32>,
    /// Override alpha (energy propagation coefficient). `None` → env/default.
    pub alpha: Option<f64>,
    /// Override decay coefficient. `None` → env/default.
    pub decay: Option<f64>,
}

// ── MERGE ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MergeQuery {
    pub pattern:   MergePattern,
    pub on_create: Vec<SetAssignment>,
    pub on_match:  Vec<SetAssignment>,
    pub ret:       Option<ReturnClause>,
}

#[derive(Debug, Clone)]
pub enum MergePattern {
    Node(MergeNodePattern),
    Edge(MergeEdgePattern),
}

#[derive(Debug, Clone)]
pub struct MergeNodePattern {
    pub alias:      String,
    pub label:      Option<String>,
    pub properties: Vec<(String, Expr)>,
}

#[derive(Debug, Clone)]
pub struct MergeEdgePattern {
    pub from:       MergeNodePattern,
    pub edge_label: Option<String>,
    pub direction:  Direction,
    pub to:         MergeNodePattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetAssignment {
    pub alias: String,
    pub field: String,
    pub value: Expr,
}

// ── MATCH ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MatchQuery {
    pub pattern:      Pattern,
    pub conditions:   Vec<Condition>,
    pub ret:          ReturnClause,
    /// `AS OF CYCLE n` — Eternal Return time-travel (Phase 15.4).
    pub as_of_cycle:  Option<u32>,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Node(NodePattern),
    Path(PathPattern),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePattern {
    pub alias: String,
    pub label: Option<String>,
    /// Phase 4: Optional semantic ID for generative retrieval.
    pub semantic_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct HopRange {
    pub min: usize,
    pub max: usize,
}

#[derive(Debug, Clone)]
pub struct PathPattern {
    pub from:       NodePattern,
    pub edge_label: Option<String>,
    pub edge_alias: Option<String>,
    pub direction:  Direction,
    pub to:         NodePattern,
    pub hop_range:  Option<HopRange>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Direction {
    /// (a)-->(b)  or  (a)-[:T]->(b)
    Out,
    /// (a)<--(b)  or  (a)<-[:T]-(b)
    In,
}

// ── Conditions ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Condition {
    Compare { left: Expr, op: CompOp, right: Expr },
    And(Box<Condition>, Box<Condition>),
    Or(Box<Condition>, Box<Condition>),
    Not(Box<Condition>),
    /// `expr IN (val1, val2, …)`
    In { expr: Expr, values: Vec<Expr> },
    /// `expr BETWEEN low AND high`  (inclusive)
    Between { expr: Expr, low: Expr, high: Expr },
    /// `expr CONTAINS|STARTS_WITH|ENDS_WITH expr`
    StringOp { left: Expr, op: StringCompOp, right: Expr },
}

/// Comparison operators.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompOp {
    Lt, Lte, Gt, Gte, Eq, Neq,
}

/// String match operators.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StringCompOp {
    Contains,
    StartsWith,
    EndsWith,
}

// ── Expressions ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    /// `alias.field`, e.g. `n.energy`
    Property { alias: String, field: String },
    /// `$name`
    Param(String),
    Float(f64),
    Int(i64),
    Str(String),
    Bool(bool),
    /// `HYPERBOLIC_DIST(alias.embedding, arg)`
    HyperbolicDist {
        alias: String,
        field: String,
        arg:   HDistArg,
    },
    /// Phase 11: `SENSORY_DIST(alias.field, $param)`
    SensoryDist {
        alias: String,
        field: String,
        arg:   String,  // param name (without $)
    },
    /// Mathematician-named function call with variable arity.
    MathFunc {
        func: MathFunc,
        args: Vec<MathFuncArg>,
    },
    /// Binary arithmetic: `n.count + 1`, `n.energy - 0.1`
    BinOp {
        left:  Box<Expr>,
        op:    ArithOp,
        right: Box<Expr>,
    },
}

/// Arithmetic operators for SET expressions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArithOp {
    Add,
    Sub,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HDistArg {
    Param(String),
    Vector(Vec<f64>),
}

// ── Mathematician-named functions ─────────────────────────

/// Built-in functions named after the mathematicians whose work
/// underpins NietzscheDB.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MathFunc {
    /// Henri Poincaré — distance in the Poincaré ball model
    PoincareDist,
    /// Felix Klein — distance in the Klein (Beltrami-Klein) model
    KleinDist,
    /// Hermann Minkowski — conformal factor / Lorentz norm
    MinkowskiNorm,
    /// Nikolai Lobachevsky — angle of parallelism
    LobachevskyAngle,
    /// Bernhard Riemann — discrete Ollivier-Ricci curvature estimate
    RiemannCurvature,
    /// Carl Friedrich Gauss — heat kernel exp(-d²/4t)
    GaussKernel,
    /// Pafnuty Chebyshev — Chebyshev polynomial T_k at node position
    ChebyshevCoeff,
    /// Srinivasa Ramanujan — local spectral expansion ratio
    RamanujanExpansion,
    /// Felix Hausdorff — local Hausdorff dimension
    HausdorffDim,
    /// Leonhard Euler — local Euler characteristic (V - E)
    EulerChar,
    /// Pierre-Simon Laplace — graph Laplacian diagonal score
    LaplacianScore,
    /// Joseph Fourier — graph Fourier coefficient cos(k·π·x)
    FourierCoeff,
    /// Peter Gustav Lejeune Dirichlet — local Dirichlet energy
    DirichletEnergy,
    /// NOW() — current Unix timestamp in seconds (f64)
    Now,
    /// EPOCH_MS() — current Unix epoch in milliseconds (f64)
    EpochMs,
    /// INTERVAL("1h") — parse duration string to seconds (f64)
    /// Supported units: s (seconds), m (minutes), h (hours), d (days), w (weeks)
    Interval,
}

/// Argument to a mathematician-named function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MathFuncArg {
    Property(String, String),
    Param(String),
    Float(f64),
    Int(i64),
    Str(String),
    Vector(Vec<f64>),
    Alias(String),
}

// ── RETURN ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ReturnClause {
    pub distinct: bool,
    pub items:    Vec<ReturnItem>,
    pub group_by: Vec<GroupByItem>,
    pub order_by: Option<OrderBy>,
    pub limit:    Option<usize>,
    pub skip:     Option<usize>,
}

#[derive(Debug, Clone)]
pub struct ReturnItem {
    pub expr:     ReturnExpr,
    pub as_alias: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ReturnExpr {
    /// Return an entire node/alias: `RETURN n`
    Alias(String),
    /// Return a specific property: `RETURN n.energy`
    Property(String, String),
    /// Return an aggregate: `RETURN COUNT(n.energy)`
    Aggregate { func: AggFunc, arg: AggArg },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggFunc { Count, Sum, Avg, Min, Max }

#[derive(Debug, Clone)]
pub enum AggArg {
    Star,
    Property(String, String),
    Alias(String),
}

#[derive(Debug, Clone)]
pub enum GroupByItem {
    Property(String, String),
    Alias(String),
}

#[derive(Debug, Clone)]
pub struct OrderBy {
    pub expr: OrderExpr,
    pub dir:  OrderDir,
}

#[derive(Debug, Clone)]
pub enum OrderExpr {
    Property(String, String),
    HyperbolicDist { alias: String, field: String, arg: HDistArg },
    /// Phase 11: ORDER BY SENSORY_DIST(alias.field, $param)
    SensoryDist { alias: String, field: String, arg: String },
    /// ORDER BY POINCARE_DIST(...) etc.
    MathFunc { func: MathFunc, args: Vec<MathFuncArg> },
    Alias(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderDir { Asc, Desc }

// ── DIFFUSE ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DiffuseQuery {
    /// Starting node — `$param` or a previously bound alias (not implemented here).
    pub from:     DiffuseFrom,
    /// Diffusion time scales.  Defaults to `[1.0]`.
    pub t_values: Vec<f64>,
    /// Maximum walk steps.  Defaults to 10.
    pub max_hops: usize,
    pub ret:      Option<ReturnClause>,
}

#[derive(Debug, Clone)]
pub enum DiffuseFrom {
    Param(String),
    Alias(String),
}

// ── RECONSTRUCT (Phase 11) ───────────────────────────────

/// `RECONSTRUCT $node_id MODALITY audio QUALITY high`
#[derive(Debug, Clone)]
pub struct ReconstructQuery {
    /// Target node — a `$param` or an alias.
    pub target:   ReconstructTarget,
    /// Optional modality filter (e.g. "audio", "text", "image", "fused").
    pub modality: Option<String>,
    /// Optional quality hint (e.g. "high", "medium", "low", "draft").
    pub quality:  Option<String>,
}

#[derive(Debug, Clone)]
pub enum ReconstructTarget {
    Param(String),
    Alias(String),
}
