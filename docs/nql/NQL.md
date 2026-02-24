# NQL — Nietzsche Query Language

**Complete Language Reference**

NQL is a declarative query language designed for multi-manifold graph databases. It provides first-class primitives for non-Euclidean geometry (Poincaré, Klein, Riemann, Minkowski), graph traversal, heat-kernel diffusion, and fractal-space operations that have no equivalent in SQL, Cypher, or SPARQL.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Query Types](#query-types)
3. [MATCH Query](#match-query)
   - [Patterns](#patterns)
   - [Edge Alias & Properties](#edge-alias--properties)
   - [Node Properties](#node-properties)
   - [WHERE Conditions](#where-conditions)
   - [Built-in Functions](#built-in-functions)
   - [Mathematician Functions](#mathematician-functions)
   - [Time Functions](#time-functions)
   - [RETURN Clause](#return-clause)
   - [Aggregates](#aggregates)
4. [CREATE Query](#create-query)
5. [MATCH SET / DELETE / DETACH DELETE](#match-set--delete--detach-delete)
6. [MERGE Query](#merge-query)
7. [DIFFUSE Query](#diffuse-query)
8. [RECONSTRUCT Query](#reconstruct-query)
9. [EXPLAIN](#explain)
10. [DAEMON Queries](#daemon-queries)
11. [Dream Queries](#dream-queries)
12. [TRANSLATE (Synesthesia)](#translate-synesthesia)
13. [Time-Travel (AS OF CYCLE)](#time-travel-as-of-cycle)
14. [COUNTERFACTUAL](#counterfactual)
15. [Archetypes](#archetypes)
16. [NARRATE](#narrate)
17. [PSYCHOANALYZE](#psychoanalyze)
18. [Transactions](#transactions)
19. [Parameters](#parameters)
20. [Multi-Collection Routing](#multi-collection-routing)
21. [Complete Examples](#complete-examples)
22. [Error Reference](#error-reference)
23. [Grammar Summary](#grammar-summary)

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
| Create | `CREATE` | Insert new nodes with labels, properties, and optional TTL |
| Update | `MATCH ... SET` | Update matched nodes' properties (supports arithmetic) |
| Delete | `MATCH ... DELETE` | Delete matched nodes |
| Detach Delete | `MATCH ... DETACH DELETE` | Delete matched nodes and all incident edges |
| Upsert | `MERGE` | Upsert nodes/edges (ON CREATE SET / ON MATCH SET) |
| Diffusion | `DIFFUSE` | Multi-scale heat-kernel activation propagation |
| Reconstruct | `RECONSTRUCT` | Decode sensory data from latent representation |
| Explain | `EXPLAIN` | Return execution plan without running the query |
| Dream | `DREAM FROM` | Speculative graph exploration via hyperbolic diffusion |
| Apply Dream | `APPLY DREAM` | Accept dream simulation results |
| Reject Dream | `REJECT DREAM` | Discard dream simulation results |
| Show Dreams | `SHOW DREAMS` | List pending dream sessions |
| Translate | `TRANSLATE` | Cross-modal projection via Poincare ball log/exp map |
| Time-travel | `MATCH ... AS OF CYCLE` | Query on named snapshots |
| Counterfactual | `COUNTERFACTUAL` | What-if query with ephemeral property overlays |
| Create Daemon | `CREATE DAEMON` | Create autonomous daemon agent |
| Drop Daemon | `DROP DAEMON` | Remove daemon agent |
| Show Daemons | `SHOW DAEMONS` | List active daemon agents |
| Show Archetypes | `SHOW ARCHETYPES` | List shared cross-collection archetypes |
| Share Archetype | `SHARE ARCHETYPE` | Publish elite node for cross-collection discovery |
| Narrate | `NARRATE` | Generate human-readable narrative from graph evolution |
| Psychoanalyze | `PSYCHOANALYZE` | Return evolutionary lineage of a node |
| Transaction | `BEGIN` / `COMMIT` / `ROLLBACK` | Transaction control |

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

### DSI — Generative Retrieval

NietzscheDB supports **Generative Retrieval** via a DSI (Differentiable Search Index). You can query nodes by their semantic hierarchical codes using the `semantic_id` filter directly in the node pattern.

```sql
-- Find nodes matching a semantic prefix (e.g., all under concept 1.25)
MATCH (n {semantic_id: "1.25.*"})
RETURN n

-- Exact semantic ID match
MATCH (n {semantic_id: "1.25.4"})
RETURN n
```
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

-- Multi-hop path (BFS 2..4 hops)
MATCH (a)-[:Association*2..4]->(b)
WHERE a.energy > 0.5
RETURN a, b LIMIT 50
```

**Edge types:**

| Label | Description |
|---|---|
| `Association` | Semantic association (default) |
| `Hierarchical` | Parent → child in hyperbolic hierarchy |
| `LSystemGenerated` | Created by the L-System growth engine |
| `Pruned` | Archived — low Hausdorff complexity region |

---

### Edge Alias & Properties

Edge aliases allow accessing edge properties in WHERE and ORDER BY clauses.

#### Syntax

```sql
-- Named edge alias: r
MATCH (a)-[r:MENTIONED]->(b)

-- Access edge properties
WHERE r.weight > 0.5
ORDER BY r.weight DESC
```

#### Edge Properties

| Property | Type | Description |
|---|---|---|
| `r.id` | UUID | Edge unique identifier |
| `r.from` | UUID | Source node ID |
| `r.to` | UUID | Target node ID |
| `r.weight` | float | Edge weight |
| `r.edge_type` | string | `"Association"` \| `"Hierarchical"` \| `"LSystemGenerated"` \| `"Pruned"` |
| `r.created_at` | int | Creation timestamp (Unix µs) |
| `r.<field>` | any | Custom field from `edge.metadata` |

#### Examples

```sql
-- Filter by edge weight
MATCH (a:Person)-[r:MENTIONED]->(b:Topic)
WHERE r.weight > 0.5
RETURN a, b ORDER BY r.weight DESC LIMIT 10

-- Order by edge creation time
MATCH (a)-[r:Association]->(b)
WHERE a.energy > 0.3
RETURN a, b ORDER BY r.created_at DESC LIMIT 20

-- Access custom edge metadata
MATCH (a)-[r:MENTIONED]->(b)
WHERE r.count > 5
RETURN a, b, r.count
```

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
| `n.valence` | float | [-1.0, 1.0] | Emotional valence: negative = punishing, positive = rewarding, 0 = neutral |
| `n.arousal` | float | [0.0, 1.0] | Emotional intensity: high = emotionally charged, low = calm |
| `n.created_at` | int | Unix µs | Creation timestamp |
| `n.expires_at` | float | Unix s | Expiration timestamp (0 if no TTL) |
| `n.<field>` | any | — | Dynamic field: falls back to `node.content` then `node.metadata` |

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

### Time Functions

| Function | Args | Returns |
|---|---|---|
| `NOW()` | none | Current Unix timestamp in seconds (f64) |
| `EPOCH_MS()` | none | Current Unix epoch in milliseconds (f64) |
| `INTERVAL("duration")` | string | Duration converted to seconds (f64) |

**INTERVAL units:** `s` (seconds), `m` (minutes), `h` (hours), `d` (days), `w` (weeks). Decimal values supported (e.g. `"1.5h"` = 5400 seconds).

**Time function examples:**

```sql
-- Nodes created in the last 7 days
MATCH (n) WHERE n.created_at > NOW() - INTERVAL("7d") RETURN n

-- Nodes expiring within 1 hour
MATCH (n) WHERE n.expires_at < EPOCH_MS() + INTERVAL("1h") * 1000.0 RETURN n

-- Verify interval conversions
MATCH (n) WHERE INTERVAL("30m") = 1800.0 RETURN n
```

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

## CREATE Query

Insert new nodes into the graph. Supports optional **TTL** (time-to-live) for automatic expiration.

### Syntax

```
CREATE (<alias>:<Label> { <property>: <value>, ... })
[RETURN <alias>]
```

### TTL Support

Include a `ttl` property (in seconds) to auto-compute `expires_at`. The `ttl` key is extracted from properties and not stored as content — it sets the node's expiration timestamp.

### Examples

```sql
-- Create a new episodic memory
CREATE (n:Episodic {title: "first meeting", source: "manual"})
RETURN n

-- Create a concept node
CREATE (c:Concept {title: "quantum mechanics", domain: "physics"})
RETURN c

-- Create with TTL (auto-expires after 1 hour)
CREATE (n:EvaSession {id: "sess_1", turn_count: 0, ttl: 3600})
RETURN n

-- Create with TTL (30 minutes)
CREATE (n:TempCache {query: "latest results", ttl: 1800})
RETURN n

-- Create with embedding (passed via params)
CREATE (n:Semantic {title: "memory of the sea"})
RETURN n
```

---

## MATCH SET / DELETE / DETACH DELETE

Update or delete matched nodes.

### SET Syntax

```
MATCH <pattern>
[WHERE <condition>]
SET <alias>.<field> = <expr> [, <alias>.<field> = <expr>]*
[RETURN <alias>]
```

**Arithmetic expressions** are supported in SET values. Each matched node is evaluated independently, so `n.count = n.count + 1` reads the current value per node.

### DELETE Syntax

```
MATCH <pattern>
[WHERE <condition>]
DELETE <alias>
```

### DETACH DELETE Syntax

Deletes matched nodes **and all incident edges** (both incoming and outgoing).

```
MATCH <pattern>
[WHERE <condition>]
DETACH DELETE <alias>
```

### Examples

```sql
-- Update energy of low-energy semantic nodes
MATCH (n:Semantic) WHERE n.energy < 0.1 SET n.energy = 0.5 RETURN n

-- Arithmetic SET: increment counter per node
MATCH (n:EvaSession {id: "sess_1"})
SET n.turn_count = n.turn_count + 1, n.status = "active"
RETURN n

-- Arithmetic SET: decrement energy
MATCH (n) WHERE n.energy > 0.8
SET n.energy = n.energy - 0.1
RETURN n

-- Delete expired nodes
MATCH (n) WHERE n.energy = 0.0 DELETE n

-- DETACH DELETE: remove node and all edges
MATCH (n:EvaSession) WHERE n.status = "expired" DETACH DELETE n

-- Update multiple fields
MATCH (n) WHERE n.id = $id SET n.energy = 0.9, n.title = "updated" RETURN n

-- Dynamic property access (falls back to node.content / node.metadata)
MATCH (n:EvaSession) WHERE n.turn_count > 10 RETURN n
```

---

## MERGE Query

Upsert pattern — insert if not exists, update if exists.

### Syntax

```
MERGE (<alias>:<Label> { <match_key>: <value>, ... })
[ON CREATE SET <alias>.<field> = <value>, ...]
[ON MATCH SET <alias>.<field> = <value>, ...]
[RETURN <alias>]
```

### Examples

```sql
-- Upsert a concept: create if new, update energy if existing
MERGE (n:Concept {title: "AI Safety"})
ON CREATE SET n.energy = 0.5, n.source = "auto"
ON MATCH SET n.energy = 0.9
RETURN n

-- Upsert edge between nodes
MERGE (a)-[:Association]->(b)
ON CREATE SET a.energy = 0.6
RETURN a, b
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

## Gas Limits & Safety

To prevent infinite recursion in graph traversals and protect resources from malformed queries, NQL implements a **Gas Limit** system.

Each query starts with a fixed gas budget (default: **50,000 units**). As the query executes, it consumes gas for every operation:
- **Node scan/read**: 1 gas
- **Edge scan/read**: 2 gas
- **Condition evaluation**: 1 gas

If the budget is exhausted, the query is terminated with a `gas limit exceeded` error.

### Custom Gas Limits
Privileged callers can specify custom gas limits for complex agency queries.

```rust
// Execute with 5x default budget
let results = execute_with_gas_limit(&query, storage, adj, params, 250_000)?;
```

---

## DAEMON Queries

Autonomous agents that live inside the database, patrolling the graph and executing actions when conditions are met. Daemons have energy budgets and decay over time.

### CREATE DAEMON

```sql
CREATE DAEMON <name> ON (<alias>:<Label>)
  WHEN <condition>
  THEN <action>
  EVERY INTERVAL("<duration>")
  [ENERGY <float>]
```

**Actions:**

| Action | Syntax | Description |
|---|---|---|
| Delete | `DELETE <alias>` | Remove matched nodes |
| Set | `SET <alias>.<field> = <value>` | Update matched node fields |
| Diffuse | `DIFFUSE FROM <alias> [WITH t=[...]] [MAX_HOPS n]` | Run diffusion from matched nodes |

### DROP DAEMON / SHOW DAEMONS

```sql
DROP DAEMON <name>
SHOW DAEMONS
```

### Examples

```sql
-- Guardian: detect curvature anomalies and diffuse to stabilize
CREATE DAEMON guardian ON (n:Memory)
  WHEN n.energy > 0.8
  THEN DIFFUSE FROM n WITH t=[0.1, 1.0] MAX_HOPS 5
  EVERY INTERVAL("1h")
  ENERGY 0.8

-- Archivist: forget old low-energy memories
CREATE DAEMON archivist ON (n:Memory)
  WHEN n.energy < 0.05 AND NOW() - n.created_at > INTERVAL("30d")
  THEN DELETE n
  EVERY INTERVAL("24h")
  ENERGY 0.4

-- List and manage
SHOW DAEMONS
DROP DAEMON archivist
```

**Energy budget:** Daemons decay energy each tick. When energy drops below threshold, the daemon is automatically reaped (deleted). Default energy: 1.0.

---

## Dream Queries

Speculative graph exploration via hyperbolic diffusion with stochastic noise.

### Syntax

```sql
-- Start a dream session
DREAM FROM <$param | alias> [DEPTH <n>] [NOISE <float>]

-- List pending dream sessions
SHOW DREAMS

-- Accept a dream's discoveries
APPLY DREAM "<dream_id>"

-- Reject a dream
REJECT DREAM "<dream_id>"
```

### Examples

```sql
-- Explore speculatively from a seed node
DREAM FROM $seed DEPTH 5 NOISE 0.05

-- See what the database "dreamed"
SHOW DREAMS

-- Accept a useful discovery
APPLY DREAM "dream_abc123"

-- Reject an unhelpful dream
REJECT DREAM "dream_xyz789"
```

Dream sessions detect energy spikes, curvature anomalies, and latent connections. Applied dreams modify node energy; rejected dreams are discarded.

---

## TRANSLATE (Synesthesia)

Cross-modal projection via hyperbolic parallel transport on the Poincare ball.

### Syntax

```sql
TRANSLATE <$param | alias> FROM <modality> TO <modality>
```

### Examples

```sql
-- "How does this text SOUND?" — translate text embedding to audio latent space
TRANSLATE $node FROM text TO audio

-- Cross-modal: visual to text
TRANSLATE $image_node FROM visual TO text
```

The algorithm: `log_map(embedding) → modal rotation → exp_map(rotated)`. Preserves radius (hierarchical depth) while changing angle (modality).

---

## Time-Travel (AS OF CYCLE)

Query named snapshots for historical state.

### Syntax

```sql
MATCH <pattern> AS OF CYCLE <n>
[WHERE <condition>]
RETURN <clause>
```

### Examples

```sql
-- How was the graph 3 cycles ago?
MATCH (n:Memory) AS OF CYCLE 3
WHERE n.energy > 0.5
RETURN n

-- Compare current vs historical
MATCH (n) AS OF CYCLE 1
RETURN n.energy, n.depth
```

Uses the `SnapshotRegistry` to load embeddings from named snapshots. Returns a read-only view of the historical graph state.

---

## COUNTERFACTUAL

What-if queries with ephemeral property overlays. No side effects on the real graph.

### Syntax

```sql
COUNTERFACTUAL SET <alias>.<field> = <value> [, ...]
MATCH <pattern>
[WHERE <condition>]
RETURN <clause>
```

### Examples

```sql
-- "What if this node had high energy?"
COUNTERFACTUAL SET n.energy = 0.95
MATCH (n:Memory)
WHERE n.depth > 0.5
RETURN n

-- Test a hypothetical scenario
COUNTERFACTUAL SET n.energy = 0.0
MATCH (n)
WHERE n.node_type = "Concept"
RETURN COUNT(*) AS would_be_pruned
```

Creates an ephemeral overlay (Copy-on-Write) over the real graph. Discarded after the query — zero side effects.

---

## INVOKE ZARATUSTRA

Manual trigger for the Zaratustra AGI engine cycles. Allows for on-demand energy propagation and phase transitions.

### Syntax

```sql
INVOKE ZARATUSTRA [IN "collection"] [CYCLES <n>] [ALPHA <f>] [DECAY <f>]
```

### Examples

```sql
-- Run one standard cycle on the default collection
INVOKE ZARATUSTRA

-- Run 5 deep cycles with custom energy propagation (alpha)
INVOKE ZARATUSTRA CYCLES 5 ALPHA 0.15 DECAY 0.02
```

The algorithm follows the **Will to Power → Eternal Recurrence → Ubermensch** phase transition logic documented in the architecture.

---


## Archetypes

Cross-collection archetype sharing via gossip protocol.

### Syntax

```sql
-- List all shared archetypes
SHOW ARCHETYPES

-- Publish a node as archetype
SHARE ARCHETYPE <$param> TO "<collection>"
```

### Examples

```sql
-- See archetypes from all collections
SHOW ARCHETYPES

-- Share an elite node to another collection
SHARE ARCHETYPE $node_id TO "shared_knowledge"
```

---

## NARRATE

Generate human-readable narrative from graph evolution.

### Syntax

```sql
NARRATE IN "<collection>" WINDOW <hours> FORMAT <json | text>
```

### Examples

```sql
-- What happened in the last 24 hours?
NARRATE IN "memories" WINDOW 24 FORMAT json

-- Weekly summary in text format
NARRATE IN "default" WINDOW 168 FORMAT text
```

Returns narrative with energy statistics, elite emergence, decay events, and auto-generated summaries.

---

## PSYCHOANALYZE

Returns the evolutionary lineage of a node — its creation source, L-System generation, energy trajectory, connections, and metadata history. Named after Freud's technique of uncovering hidden origins.

### Syntax

```sql
PSYCHOANALYZE <$param | alias>
```

### Examples

```sql
-- Trace the evolutionary history of a concept node
PSYCHOANALYZE $node_id

-- Using an alias
PSYCHOANALYZE mynode
```

### Output

Returns a JSON lineage report including:

| Field | Description |
|---|---|
| `node_id` | UUID of the analyzed node |
| `node_type` | Semantic, Episodic, Concept, etc. |
| `depth` | Hyperbolic depth (position in Poincaré ball) |
| `energy` | Current activation energy |
| `lsystem_generation` | Which L-System generation created this node (0 = manual) |
| `hausdorff_local` | Local fractal dimension |
| `is_phantom` | Whether the node is a structural scar |
| `valence` | Emotional valence [-1, 1] |
| `arousal` | Emotional intensity [0, 1] |
| `connections.outgoing` | Number of outgoing edges |
| `connections.incoming` | Number of incoming edges |
| `connections.total` | Total edge count |
| `edge_types` | Distribution of edge types (Association, Hierarchical, etc.) |
| `content` | Node content payload |
| `metadata` | Node metadata key-value pairs |

### Use Cases

```sql
-- Understand why a concept emerged from L-System growth
PSYCHOANALYZE $concept_id
-- → Shows lsystem_generation=3, spawned by Hierarchical edge from parent

-- Debug a phantom node's history
PSYCHOANALYZE $phantom_id
-- → Shows is_phantom=true, original connections preserved

-- Audit emotional state of a memory
PSYCHOANALYZE $memory_id
-- → Shows valence=0.7 (rewarding), arousal=0.9 (intense)
```

---

## Transactions

NQL supports explicit transaction control.

```sql
BEGIN
-- ... multiple queries ...
COMMIT

-- Or rollback
BEGIN
-- ... queries ...
ROLLBACK
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

### 13. Edge alias — filter and sort by edge properties

```sql
-- Find strongly connected topics by edge weight
MATCH (a:Person)-[r:MENTIONED]->(b:Topic)
WHERE r.weight > 0.5 AND a.energy > 0.3
RETURN a, b
ORDER BY r.weight DESC
LIMIT 10
```

### 14. Arithmetic SET — session counters

```sql
-- Increment session turn counter (per-node evaluation)
MATCH (n:EvaSession {id: "sess_42"})
SET n.turn_count = n.turn_count + 1, n.last_active = "2026-02-22"
RETURN n
```

### 15. CREATE with TTL — ephemeral cache nodes

```sql
-- Create a temporary node that auto-expires in 1 hour
CREATE (n:TempResult {query: "weather São Paulo", result: "28°C", ttl: 3600})
RETURN n
```

### 16. DETACH DELETE — clean removal

```sql
-- Remove expired sessions and all their edges
MATCH (n:EvaSession)
WHERE n.status = "expired"
DETACH DELETE n
```

### 17. Dynamic property access — EVA-Mind fields

```sql
-- Access fields stored in node.content (automatic fallback)
MATCH (n:EvaSession)
WHERE n.turn_count > 10 AND n.status = "active"
RETURN n.turn_count, n.status, n.energy
ORDER BY n.turn_count DESC
LIMIT 5
```

### 18. Emotional memory retrieval — Valence/Arousal

```sql
-- Find emotionally intense memories (high arousal)
MATCH (n:Episodic)
WHERE n.arousal > 0.7
RETURN n ORDER BY n.arousal DESC LIMIT 20

-- Find rewarding memories (positive valence)
MATCH (n:Memory)
WHERE n.valence > 0.5 AND n.energy > 0.3
RETURN n ORDER BY n.valence DESC LIMIT 15

-- Find traumatic memories (negative valence, high arousal)
MATCH (n:Episodic)
WHERE n.valence < -0.5 AND n.arousal > 0.8
RETURN n LIMIT 10

-- Update emotional state after retrieval
MATCH (n) WHERE n.id = $id
SET n.valence = 0.6, n.arousal = 0.8
RETURN n
```

### 19. PSYCHOANALYZE — node lineage

```sql
-- Trace the evolutionary lineage of a concept
PSYCHOANALYZE $node_id
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
query         = { explain_query | create_daemon_query | drop_daemon_query | show_daemons_query
                | dream_from_query | apply_dream_query | reject_dream_query | show_dreams_query
                | translate_query | counterfactual_query | show_archetypes_query
                | share_archetype_query | narrate_query
                | create_query | merge_query | match_set_query | match_delete_query
                | match_query | diffuse_query | reconstruct_query
                | begin_tx | commit_tx | rollback_tx }

explain_query = { "EXPLAIN" ~ (match_query | diffuse_query | reconstruct_query) }

-- Pattern matching
match_query   = { match_clause ~ as_of_clause? ~ where_clause? ~ return_clause }
match_clause  = { "MATCH" ~ (path_pattern | node_pattern) }
node_pattern  = { "(" ~ ident ~ (":" ~ ident)? ~ ")" }
path_pattern  = { node_pattern ~ edge_pattern ~ node_pattern }
edge_pattern  = { "-[" ~ ident? ~ ":" ~ ident ~ ("*" ~ integer ~ ".." ~ integer)? ~ "]->"
               | "<-[" ~ ident? ~ ":" ~ ident ~ "]-" | "-[]->" }

-- CRUD
create_query      = { "CREATE" ~ node_with_props ~ return_clause? }
merge_query       = { "MERGE" ~ (path_pattern | node_with_props)
                      ~ ("ON" ~ "CREATE" ~ "SET" ~ set_assignments)?
                      ~ ("ON" ~ "MATCH" ~ "SET" ~ set_assignments)?
                      ~ return_clause? }
match_set_query   = { match_clause ~ where_clause? ~ "SET" ~ set_assignments ~ return_clause? }
match_delete_query = { match_clause ~ where_clause? ~ "DETACH"? ~ "DELETE" ~ ident }

-- SET expressions (supports arithmetic)
set_assignment    = { property ~ "=" ~ set_expr }
set_expr          = { atom ~ (("+" | "-") ~ atom)? }   -- n.count + 1, n.energy - 0.1

-- Conditions & expressions
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

-- Return
return_clause = { "RETURN" ~ "DISTINCT"? ~ return_items
                  ~ ("GROUP" ~ "BY" ~ group_by_items)?
                  ~ ("ORDER" ~ "BY" ~ order_by)?
                  ~ ("SKIP" ~ integer)?
                  ~ ("LIMIT" ~ integer)? }

-- Diffusion & Reconstruct
diffuse_query = { "DIFFUSE" ~ "FROM" ~ (param | ident)
                  ~ ("WITH" ~ "t" ~ "=" ~ "[" ~ float_list ~ "]")?
                  ~ ("MAX_HOPS" ~ integer)?
                  ~ ("RETURN" ~ return_clause)? }
reconstruct_query = { "RECONSTRUCT" ~ (param | ident)
                      ~ ("MODALITY" ~ ident)?
                      ~ ("QUALITY" ~ ident)? }

-- DAEMON Agents
create_daemon_query = { "CREATE" ~ "DAEMON" ~ daemon_name ~ "ON" ~ node_pattern
                        ~ daemon_when ~ daemon_then ~ daemon_every ~ daemon_energy? }
daemon_when   = { "WHEN" ~ condition }
daemon_then   = { "THEN" ~ daemon_action }
daemon_action = { "DELETE" ~ ident | "SET" ~ set_assignments | "DIFFUSE" ~ "FROM" ~ ident ~ diffuse_t? ~ diffuse_hops? }
daemon_every  = { "EVERY" ~ atom }
daemon_energy = { "ENERGY" ~ (float | integer) }
drop_daemon_query  = { "DROP" ~ "DAEMON" ~ daemon_name }
show_daemons_query = { "SHOW" ~ "DAEMONS" }

-- Dream Queries
dream_from_query    = { "DREAM" ~ "FROM" ~ (param | ident) ~ ("DEPTH" ~ integer)? ~ ("NOISE" ~ float)? }
apply_dream_query   = { "APPLY" ~ "DREAM" ~ string }
reject_dream_query  = { "REJECT" ~ "DREAM" ~ string }
show_dreams_query   = { "SHOW" ~ "DREAMS" }

-- Synesthesia
translate_query = { "TRANSLATE" ~ (param | ident) ~ "FROM" ~ ident ~ "TO" ~ ident }

-- Time-travel & Counterfactual
as_of_clause        = { "AS" ~ "OF" ~ "CYCLE" ~ integer }
counterfactual_query = { "COUNTERFACTUAL" ~ "SET" ~ set_assignments ~ match_clause ~ where_clause? ~ return_clause }

-- Archetypes
show_archetypes_query  = { "SHOW" ~ "ARCHETYPES" }
share_archetype_query  = { "SHARE" ~ "ARCHETYPE" ~ (param | ident) ~ "TO" ~ string }

-- Narrative
narrate_query = { "NARRATE" ~ "IN" ~ string ~ "WINDOW" ~ integer ~ "FORMAT" ~ ident }

-- Psychoanalyze (Lineage)
psychoanalyze_query = { "PSYCHOANALYZE" ~ (param | ident) }

-- Transactions
begin_tx    = { "BEGIN" }
commit_tx   = { "COMMIT" }
rollback_tx = { "ROLLBACK" }
```

*(Simplified — see `crates/nietzsche-query/src/nql.pest` for the authoritative PEG grammar.)*

---

*NQL is case-insensitive for keywords (`MATCH`, `match`, `Match` are equivalent). Identifiers and string literals are case-sensitive.*
