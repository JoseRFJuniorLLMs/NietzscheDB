// ─────────────────────────────────────────────
// NQL Abstract Syntax Tree
// ─────────────────────────────────────────────

/// Top-level NQL query.
#[derive(Debug, Clone)]
pub enum Query {
    Match(MatchQuery),
    Diffuse(DiffuseQuery),
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
}

/// Comparison operators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompOp {
    Lt, Lte, Gt, Gte, Eq, Neq,
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
}

#[derive(Debug, Clone)]
pub enum HDistArg {
    Param(String),
    Vector(Vec<f64>),
}

// ── RETURN ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ReturnClause {
    pub items:    Vec<ReturnItem>,
    pub order_by: Option<OrderBy>,
    pub limit:    Option<usize>,
}

#[derive(Debug, Clone)]
pub enum ReturnItem {
    /// Return an entire node/alias: `RETURN n`
    Alias(String),
    /// Return a specific property: `RETURN n.energy`
    Property(String, String),
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
