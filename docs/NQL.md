# NQL — Nietzsche Query Language

**Complete Language Reference**

NQL is a declarative query language designed for temporal hyperbolic graph databases. It provides first-class primitives for hyperbolic geometry, graph traversal, heat-kernel diffusion, and fractal-space operations that have no equivalent in SQL, Cypher, or SPARQL.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Query Types](#query-types)
3. [MATCH Query](#match-query)
   - [Patterns](#patterns)
   - [Node Properties](#node-properties)
   - [WHERE Conditions](#where-conditions)
   - [Built-in Functions](#built-in-functions)
   - [Mathematician Functions](#mathematician-functions)
   - [RETURN Clause](#return-clause)
   - [Aggregates](#aggregates)
4. [DIFFUSE Query](#diffuse-query)
5. [RECONSTRUCT Query](#reconstruct-query)
6. [EXPLAIN](#explain)
7. [Parameters](#parameters)
8. [Multi-Collection Routing](#multi-collection-routing)
9. [Complete Examples](#complete-examples)
10. [Error Reference](#error-reference)
11. [Grammar Summary](#grammar-summary)

---

## Quick Reference

```sql
-- Hyperbolic nearest-neighbor search
MATCH (n:Memory)
WHERE HYPERBOLIC_DIST(n.embedding, $q) < 0.5
  AND n.depth > 0.6
RETURN n
ORDER BY HYPERBOLIC_DIST(n.embedding, $q) ASC
LIMIT 10

-- Graph traversal with energy filter
MATCH (a:Concept)-[:Hierarchical]->(b)
WHERE a.energy > 0.7
RETURN b ORDER BY b.depth DESC LIMIT 20

-- Heat-kernel diffusion across multiple scales
DIFFUSE FROM $seed
  WITH t = [0.1, 1.0, 10.0]
  MAX_HOPS 6
RETURN path

-- Inspect execution plan
EXPLAIN MATCH (n) WHERE n.energy > 0.5 RETURN n LIMIT 5
```

---

## Query Types

| Type | Keyword | Purpose |
|---|---|---|
| Pattern match | `MATCH` | Filter nodes/edges by properties and geometry |
| Diffusion | `DIFFUSE` | Multi-scale heat-kernel activation propagation |
| Reconstruct | `RECONSTRUCT` | Decode sensory data from latent representation (Phase 11) |
| Explain | `EXPLAIN` | Return execution plan without running the query |

---

## MATCH Query

### Syntax

```
MATCH <pattern>
[WHERE <condition>]
RETURN <return_clause>
[ORDER BY <expr> [ASC | DESC]]
[LIMIT <n>]
[SKIP <n>]
```

### Patterns

#### Node Pattern

```sql
MATCH (n)                  -- any node, alias n
MATCH (m:Memory)           -- node with label Memory
MATCH (c:Concept)          -- node with label Concept
```

#### Path Pattern

```sql
-- Outgoing edge
MATCH (a)-[:Association]->(b)
MATCH (a:Concept)-[:Hierarchical]->(b:Memory)

-- Incoming edge
MATCH (a)<-[:Hierarchical]-(b)

-- Any edge type (omit label)
MATCH (a)-[]->(b)
```

**Edge types:**

| Label | Description |
|---|---|
| `Association` | Semantic association (default) |
| `Hierarchical` | Parent → child in hyperbolic hierarchy |
| `LSystemGenerated` | Created by the L-System growth engine |
| `Pruned` | Archived — low Hausdorff complexity region |

---

### Node Properties

| Property | Type | Range | Description |
|---|---|---|---|
| `n.energy` | float | [0.0, 1.0] | Activation energy of the node |
| `n.depth` | float | [0.0, 1.0) | Hyperbolic depth — 0 = center (abstract), →1 = boundary (specific) |
| `n.hausdorff` | float | ≥ 0 | Local Hausdorff dimension (fractal complexity) |
| `n.node_type` | string | — | `"Semantic"` \| `"Episodic"` \| `"Concept"` \| `"DreamSnapshot"` |
| `n.embedding` | vector | ‖x‖ < 1 | Poincaré ball vector (for distance functions) |
| `n.content` | JSON | — | Arbitrary payload stored with the node |
| `n.created_at` | int | Unix µs | Creation timestamp |

---

### WHERE Conditions

#### Comparison Operators

```sql
WHERE n.energy > 0.5
WHERE n.depth <= 0.8
WHERE n.energy >= 0.3 AND n.depth < 0.9
WHERE n.node_type = "Episodic"
WHERE n.energy != 0.0
```

Operators: `<`  `<=`  `>`  `>=`  `=`  `!=`

#### Logical Combinators

```sql
WHERE n.energy > 0.3 AND n.depth < 0.8
WHERE n.energy > 0.8 OR n.depth > 0.9
WHERE NOT n.node_type = "Pruned"
WHERE (n.energy > 0.5 OR n.depth > 0.7) AND NOT n.node_type = "DreamSnapshot"
```

#### IN — set membership

```sql
WHERE n.node_type IN ("Semantic", "Episodic")
WHERE n.energy IN (0.1, 0.5, 0.9)
```

#### BETWEEN — inclusive range

```sql
WHERE n.energy BETWEEN 0.3 AND 0.9
WHERE n.depth BETWEEN 0.1 AND 0.6
```

#### String Operators

```sql
WHERE n.node_type CONTAINS "mem"
WHERE n.node_type STARTS_WITH "Ep"
WHERE n.node_type ENDS_WITH "ic"
```

---

### Built-in Functions

#### HYPERBOLIC_DIST

Poincaré ball geodesic distance between a node's embedding and a query vector.

```sql
-- Filter by hyperbolic distance
WHERE HYPERBOLIC_DIST(n.embedding, $query_vec) < 0.4

-- Order by hyperbolic distance (nearest-neighbor)
ORDER BY HYPERBOLIC_DIST(n.embedding, $query_vec) ASC

-- Combine with depth filter
WHERE HYPERBOLIC_DIST(n.embedding, $q) < 0.5
  AND n.depth > 0.6
```

**Signature:** `HYPERBOLIC_DIST(alias.field, $param | [v0, v1, ...])`

The second argument can be a parameter reference (`$q`) or an inline vector literal (`[0.1, 0.2, 0.3]`).

#### SENSORY_DIST *(Phase 11)*

Distance in the sensory latent space.

```sql
WHERE SENSORY_DIST(n.latent, $audio_latent) < 0.3
ORDER BY SENSORY_DIST(n.latent, $audio_latent) ASC
```

---

### Mathematician Functions

NQL names its geometric functions after the mathematicians whose work underpins them.

| Function | Named after | What it computes |
|---|---|---|
| `POINCARE_DIST(n, $p)` | Henri Poincaré | Geodesic distance in the Poincaré ball model |
| `KLEIN_DIST(n, $p)` | Felix Klein | Distance in the Beltrami-Klein disk model |
| `MINKOWSKI_NORM(n)` | Hermann Minkowski | Conformal factor / Minkowski-Lorentz norm |
| `LOBACHEVSKY_ANGLE(n, $p)` | Nikolai Lobachevsky | Angle of parallelism at point n toward p |
| `RIEMANN_CURVATURE(n)` | Bernhard Riemann | Ollivier-Ricci curvature at node n |
| `GAUSS_KERNEL(n, t)` | Carl Friedrich Gauss | Heat kernel value `exp(-d²/4t)` |
| `CHEBYSHEV_COEFF(n, k)` | Pafnuty Chebyshev | Chebyshev polynomial `T_k` at node position |
| `RAMANUJAN_EXPANSION(n)` | Srinivasa Ramanujan | Local spectral expansion ratio |
| `HAUSDORFF_DIM(n)` | Felix Hausdorff | Local Hausdorff fractal dimension |
| `EULER_CHAR(n)` | Leonhard Euler | Local Euler characteristic `V − E` |
| `LAPLACIAN_SCORE(n)` | Pierre-Simon Laplace | Graph Laplacian diagonal score |
| `FOURIER_COEFF(n, k)` | Joseph Fourier | Graph Fourier coefficient `cos(k·π·x)` |
| `DIRICHLET_ENERGY(n)` | P.G.L. Dirichlet | Local Dirichlet energy (smoothness) |

**Usage examples:**

```sql
-- Nodes with high local curvature (structural bottlenecks)
MATCH (n)
WHERE RIEMANN_CURVATURE(n) > 0.4
RETURN n ORDER BY RIEMANN_CURVATURE(n) DESC LIMIT 20

-- Fractal complexity filter
MATCH (n:Memory)
WHERE HAUSDORFF_DIM(n) BETWEEN 1.2 AND 1.8
RETURN n LIMIT 50

-- Low Dirichlet energy = smooth neighborhood
MATCH (n)
WHERE DIRICHLET_ENERGY(n) < 0.1
  AND n.energy > 0.5
RETURN n ORDER BY n.depth ASC LIMIT 10

-- Heat kernel activation at scale t=2.0
MATCH (n)
WHERE GAUSS_KERNEL(n, 2.0) > 0.7
RETURN n LIMIT 30
```

---

### RETURN Clause

#### Return whole node

```sql
RETURN n
RETURN a, b          -- multiple aliases
RETURN DISTINCT n    -- deduplicate
```

#### Return specific property

```sql
RETURN n.energy
RETURN n.depth, n.node_type
RETURN n.energy AS e, n.depth AS d
```

#### ORDER BY

```sql
ORDER BY n.energy DESC
ORDER BY n.depth ASC
ORDER BY HYPERBOLIC_DIST(n.embedding, $q) ASC   -- nearest-first
ORDER BY POINCARE_DIST(n, $q) ASC
```

#### LIMIT and SKIP

```sql
RETURN n LIMIT 10
RETURN n SKIP 20 LIMIT 10    -- pagination (page 3 of 10)
```

---

### Aggregates

```sql
-- Count all matches
RETURN COUNT(*) AS total

-- Count with property
RETURN COUNT(n.energy) AS count

-- Numeric aggregates
RETURN SUM(n.energy) AS total_energy
RETURN AVG(n.depth)  AS mean_depth
RETURN MIN(n.energy) AS min_e
RETURN MAX(n.depth)  AS max_d
```

#### GROUP BY

```sql
-- Average energy by node type
MATCH (n)
RETURN n.node_type, AVG(n.energy) AS avg_e
GROUP BY n.node_type
ORDER BY avg_e DESC
```

---

## DIFFUSE Query

Propagates activation from a source node using the **Chebyshev heat-kernel approximation** of `e^(-tL)`, where `L` is the hyperbolic graph Laplacian.

### Syntax

```
DIFFUSE FROM <source>
  [WITH t = [t1, t2, ...]]
  [MAX_HOPS <n>]
  [RETURN <path | ...>]
```

### Source

```sql
DIFFUSE FROM $node_id          -- named parameter (UUID)
DIFFUSE FROM $seed             -- any $param name
```

### Diffusion scales (t)

The `t` parameter controls how far activation spreads:

| t value | Effect |
|---|---|
| `0.01 – 0.1` | Focused: only immediate neighbors activated |
| `1.0` | Local neighborhood — 2–3 hops |
| `10.0` | Broad: structurally connected but semantically distant |
| `100.0` | Global: near-uniform activation |

```sql
-- Focused recall (direct associations only)
DIFFUSE FROM $seed WITH t = [0.1] MAX_HOPS 3

-- Multi-scale: from focused to free association
DIFFUSE FROM $seed WITH t = [0.1, 1.0, 10.0] MAX_HOPS 6 RETURN path

-- Default (t=1.0, MAX_HOPS=10 if omitted)
DIFFUSE FROM $seed RETURN path
```

### Use cases

```sql
-- Activate EVA's memory from a sensory cue
DIFFUSE FROM $audio_latent_node
  WITH t = [0.1, 1.0]
  MAX_HOPS 4
RETURN path

-- Free association for creative reasoning
DIFFUSE FROM $concept_node
  WITH t = [5.0, 20.0]
  MAX_HOPS 8
RETURN path

-- Map influence radius for sleep cycle targeting
DIFFUSE FROM $high_curvature_node
  WITH t = [0.5]
  MAX_HOPS 3
```

---

## RECONSTRUCT Query

*(Phase 11 — Sensory Encoder/Decoder)*

Decodes sensory data (audio, image, text) from a node's latent vector.

### Syntax

```
RECONSTRUCT <target>
  [MODALITY <audio | text | image | fused>]
  [QUALITY <high | medium | low | draft>]
```

### Examples

```sql
-- Reconstruct audio memory
RECONSTRUCT $node_id MODALITY audio QUALITY high

-- Reconstruct as text (transcription)
RECONSTRUCT $memory_node MODALITY text QUALITY medium

-- Fused multi-modal reconstruction
RECONSTRUCT $episodic_node MODALITY fused QUALITY low

-- No modality hint (auto-detected from node type)
RECONSTRUCT $node_id
```

---

## EXPLAIN

Returns the execution plan without running the query. Useful for understanding which index or scan strategy NQL will use.

### Syntax

```
EXPLAIN <match_query | diffuse_query | reconstruct_query>
```

### Examples

```sql
EXPLAIN MATCH (n:Memory) WHERE n.energy > 0.5 RETURN n LIMIT 10

EXPLAIN MATCH (n)
WHERE HYPERBOLIC_DIST(n.embedding, $q) < 0.4
RETURN n ORDER BY HYPERBOLIC_DIST(n.embedding, $q) ASC LIMIT 5

EXPLAIN DIFFUSE FROM $seed WITH t = [1.0] MAX_HOPS 5 RETURN path
```

**Plan output (example):**
```json
{
  "plan": "HyperbolicKnn",
  "index": "HNSW",
  "filter": "energy > 0.5",
  "limit": 10,
  "estimated_cost": "O(log n)"
}
```

---

## Parameters

Named parameters prefixed with `$`. Passed at query time via the `Params` map.

```sql
-- Single parameter
MATCH (n) WHERE HYPERBOLIC_DIST(n.embedding, $q) < 0.5 RETURN n

-- Multiple parameters
MATCH (n)
WHERE HYPERBOLIC_DIST(n.embedding, $query_vec) < $threshold
  AND n.depth > $min_depth
RETURN n LIMIT $k
```

**Rust API:**
```rust
let mut params = Params::new();
params.insert("q", ParamValue::Vector(vec![0.1, 0.2, 0.3]));
params.insert("threshold", ParamValue::Float(0.5));
params.insert("min_depth", ParamValue::Float(0.6));
params.insert("k", ParamValue::Int(10));

let results = execute(&query, storage, adjacency, &params)?;
```

**Python SDK:**
```python
results = db.query(
    "MATCH (n) WHERE HYPERBOLIC_DIST(n.embedding, $q) < 0.5 RETURN n LIMIT 10",
    params={"q": [0.1, 0.2, 0.3]}
)
```

---

## Multi-Collection Routing

Every NQL query is executed against a specific **collection** (Fase B). The collection is specified at the gRPC level, not in NQL syntax.

Each collection has its own isolated:
- RocksDB storage
- HNSW index (dimension and metric configured per collection)
- WAL
- Graph adjacency

**gRPC (protobuf):**
```protobuf
message QueryRequest {
  string collection = 1;  // empty → "default"
  string nql        = 2;
}
```

**Python SDK:**
```python
# Query the "core" collection (EVA's personal memory)
results = db.query(
    "MATCH (n:Concept) WHERE n.energy > 0.7 RETURN n LIMIT 5",
    collection="core"
)

# Query patient data
results = db.query(
    "MATCH (n:Episodic) WHERE n.depth > 0.5 RETURN n",
    collection="patients"
)

# Default collection (omit or leave empty)
results = db.query("MATCH (n) RETURN n LIMIT 10")
```

**HTTP dashboard REST endpoint:**
```bash
curl -X POST http://localhost:8082/api/query \
  -H "Content-Type: application/json" \
  -d '{"nql": "MATCH (n) WHERE n.energy > 0.5 RETURN n LIMIT 5"}'
```

---

## Complete Examples

### 1. Semantic memory retrieval (EVA recall)

```sql
-- Retrieve top-10 memories closest to a query embedding,
-- filtering out pruned nodes and requiring minimum depth
MATCH (m:Memory)
WHERE HYPERBOLIC_DIST(m.embedding, $query_embedding) < 0.4
  AND m.depth > 0.5
  AND NOT m.node_type = "Pruned"
RETURN m
ORDER BY HYPERBOLIC_DIST(m.embedding, $query_embedding) ASC
LIMIT 10
```

### 2. Hierarchical concept expansion

```sql
-- Find all concepts under "consciousness" in the Poincaré hierarchy
MATCH (root:Concept)-[:Hierarchical]->(child)
WHERE root.energy > 0.8
  AND child.depth > root.depth
RETURN child
ORDER BY child.depth ASC
LIMIT 50
```

### 3. Energy-gated BFS for active memory

```sql
-- Active memories (high energy) up to 2 hops from seed
MATCH (seed)-[:Association]->(n)
WHERE seed.energy > 0.7
  AND n.energy > 0.4
RETURN n ORDER BY n.energy DESC LIMIT 30
```

### 4. Multi-scale free association

```sql
-- Use diffusion to find "resonant" nodes at different scales
-- t=0.1 → immediate neighbors; t=10 → distant but structurally connected
DIFFUSE FROM $concept_node
  WITH t = [0.1, 1.0, 5.0, 10.0]
  MAX_HOPS 8
RETURN path
```

### 5. Fractal health monitoring

```sql
-- Nodes drifting outside healthy fractal regime (for sleep cycle targeting)
MATCH (n)
WHERE HAUSDORFF_DIM(n) < 0.5
   OR HAUSDORFF_DIM(n) > 1.9
RETURN n.node_type, COUNT(*) AS count
GROUP BY n.node_type
ORDER BY count DESC
```

### 6. Sleep cycle candidate selection

```sql
-- High curvature + high energy = prime reconsolidation targets
MATCH (n)
WHERE RIEMANN_CURVATURE(n) > 0.3
  AND n.energy > 0.6
  AND n.node_type IN ("Episodic", "Concept")
RETURN n
ORDER BY RIEMANN_CURVATURE(n) DESC
LIMIT 20
```

### 7. Aggregation — knowledge topology report

```sql
-- Distribution of nodes by type and average depth
MATCH (n)
RETURN n.node_type AS type,
       COUNT(*) AS count,
       AVG(n.depth) AS avg_depth,
       AVG(n.energy) AS avg_energy
GROUP BY n.node_type
ORDER BY count DESC
```

### 8. Poincaré distance ordering (explicit model)

```sql
-- Use POINCARE_DIST (mathematician function) instead of HYPERBOLIC_DIST alias
MATCH (n:Memory)
WHERE POINCARE_DIST(n, $q) < 0.35
  AND n.energy > 0.3
RETURN n
ORDER BY POINCARE_DIST(n, $q) ASC
LIMIT 15
```

### 9. Path query — L-System lineage

```sql
-- Find nodes created by L-System growth from a parent concept
MATCH (parent:Concept)-[:LSystemGenerated]->(child)
WHERE parent.energy > 0.5
RETURN child
ORDER BY child.depth DESC
LIMIT 100
```

### 10. Pagination — browsing the knowledge graph

```sql
-- Page 3 of semantic memories (10 per page)
MATCH (n:Memory)
WHERE n.depth BETWEEN 0.3 AND 0.8
RETURN n
ORDER BY n.created_at DESC
SKIP 20
LIMIT 10
```

### 11. Sensory search + reconstruction (Phase 11)

```sql
-- Find nodes close to an audio latent vector, then reconstruct
MATCH (n)
WHERE SENSORY_DIST(n.latent, $audio_q) < 0.25
  AND n.node_type = "Episodic"
RETURN n
ORDER BY SENSORY_DIST(n.latent, $audio_q) ASC
LIMIT 3
```

```sql
-- Reconstruct the top match
RECONSTRUCT $top_node_id MODALITY audio QUALITY high
```

### 12. Dirichlet energy — smooth neighborhood detection

```sql
-- Smooth neighborhoods = stable memory regions (low gradient)
MATCH (n)
WHERE DIRICHLET_ENERGY(n) < 0.05
  AND n.energy > 0.4
RETURN n
ORDER BY n.energy DESC
LIMIT 25
```

---

## Error Reference

| Error | Cause | Fix |
|---|---|---|
| `Parse: unexpected token` | Syntax error in query | Check keyword spelling and pattern syntax |
| `Parse: missing RETURN clause` | MATCH without RETURN | Add `RETURN n` |
| `Parse: missing MATCH pattern` | MATCH with no pattern | Add `(n)` or `(a)-[:T]->(b)` |
| `Parse: EXPLAIN requires a query` | `EXPLAIN` with no inner query | Add a MATCH or DIFFUSE after EXPLAIN |
| `Execution: collection not found` | Named collection doesn't exist | Create it via `CreateCollection` RPC first |
| `Execution: param not found: $x` | Missing parameter in Params map | Add `$x` to the params before calling execute |
| `Validation: NQL exceeds 8192 bytes` | Query too long | Split into multiple queries |

---

## Grammar Summary

```peg
query         = { match_query | diffuse_query | reconstruct_query | explain_query }
explain_query = { "EXPLAIN" ~ (match_query | diffuse_query | reconstruct_query) }

match_query   = { match_clause ~ where_clause? ~ return_clause }
match_clause  = { "MATCH" ~ (path_pattern | node_pattern) }
node_pattern  = { "(" ~ ident ~ (":" ~ ident)? ~ ")" }
path_pattern  = { node_pattern ~ ("-[:" ~ ident ~ "]->" | "<-[:" ~ ident ~ "]-")  ~ node_pattern }

where_clause  = { "WHERE" ~ condition }
condition     = { atom_cond ~ (("AND" | "OR") ~ condition)* }
atom_cond     = { "NOT" ~ condition | "(" ~ condition ~ ")" | compare | in_cond | between_cond | string_op_cond }
compare       = { expr ~ comp_op ~ expr }
comp_op       = { "<=" | ">=" | "!=" | "<" | ">" | "=" }
in_cond       = { expr ~ "IN" ~ "(" ~ expr ~ ("," ~ expr)* ~ ")" }
between_cond  = { expr ~ "BETWEEN" ~ expr ~ "AND" ~ expr }
string_op_cond = { expr ~ ("CONTAINS" | "STARTS_WITH" | "ENDS_WITH") ~ expr }

expr          = { hyperbolic_dist | sensory_dist | math_func_call | property | param | literal }
hyperbolic_dist = { "HYPERBOLIC_DIST" ~ "(" ~ property ~ "," ~ (param | vector_lit) ~ ")" }
math_func_call = { math_func_name ~ "(" ~ math_func_args ~ ")" }

return_clause = { "RETURN" ~ "DISTINCT"? ~ return_items
                  ~ ("GROUP" ~ "BY" ~ group_by_items)?
                  ~ ("ORDER" ~ "BY" ~ order_by)?
                  ~ ("SKIP" ~ integer)?
                  ~ ("LIMIT" ~ integer)? }

diffuse_query = { "DIFFUSE" ~ "FROM" ~ (param | ident)
                  ~ ("WITH" ~ "t" ~ "=" ~ "[" ~ float_list ~ "]")?
                  ~ ("MAX_HOPS" ~ integer)?
                  ~ ("RETURN" ~ return_clause)? }

reconstruct_query = { "RECONSTRUCT" ~ (param | ident)
                      ~ ("MODALITY" ~ ident)?
                      ~ ("QUALITY" ~ ident)? }
```

*(Simplified — see `crates/nietzsche-query/src/nql.pest` for the authoritative PEG grammar.)*

---

*NQL is case-insensitive for keywords (`MATCH`, `match`, `Match` are equivalent). Identifiers and string literals are case-sensitive.*
