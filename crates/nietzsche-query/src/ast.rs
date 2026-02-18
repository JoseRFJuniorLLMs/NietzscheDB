// ─────────────────────────────────────────────
// NQL Abstract Syntax Tree  (v2)
// ─────────────────────────────────────────────

/// Top-level NQL query.
#[derive(Debug, Clone)]
pub enum Query {
    Match(MatchQuery),
    Diffuse(DiffuseQuery),
    /// Phase 11: reconstruct sensory data from a node's latent vector.
    Reconstruct(ReconstructQuery),
    /// `EXPLAIN <inner query>` — return the execution plan without running it.
    Explain(Box<Query>),
}

// ── MATCH ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MatchQuery {
    pub pattern:    Pattern,
    pub conditions: Vec<Condition>,
    pub ret:        ReturnClause,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Node(NodePattern),
    Path(PathPattern),
}

#[derive(Debug, Clone)]
pub struct NodePattern {
    pub alias: String,
    pub label: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PathPattern {
    pub from:       NodePattern,
    pub edge_label: Option<String>,
    pub direction:  Direction,
    pub to:         NodePattern,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Direction {
    /// (a)-->(b)  or  (a)-[:T]->(b)
    Out,
    /// (a)<--(b)  or  (a)<-[:T]-(b)
    In,
}

// ── Conditions ────────────────────────────────────────────

#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompOp {
    Lt, Lte, Gt, Gte, Eq, Neq,
}

/// String match operators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StringCompOp {
    Contains,
    StartsWith,
    EndsWith,
}

// ── Expressions ───────────────────────────────────────────

#[derive(Debug, Clone)]
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
}

#[derive(Debug, Clone)]
pub enum HDistArg {
    Param(String),
    Vector(Vec<f64>),
}

// ── Mathematician-named functions ─────────────────────────

/// Built-in functions named after the mathematicians whose work
/// underpins NietzscheDB.
#[derive(Debug, Clone, PartialEq, Eq)]
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
}

/// Argument to a mathematician-named function.
#[derive(Debug, Clone)]
pub enum MathFuncArg {
    Property(String, String),
    Param(String),
    Float(f64),
    Int(i64),
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
