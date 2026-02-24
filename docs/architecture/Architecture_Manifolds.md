# NietzscheDB: Multi-Manifold Architecture

NietzscheDB is the first database to implement **Geometric Perspectivism**. It doesn't treat geometry as a fixed property of the data, but as a dynamic lens applied at query time.

## The Unified Poincaré Layer

All embeddings in NietzscheDB are stored in the **Poincaré Ball** model ($\|x\| < 1$). This is the "anatomical" storage layer because it is the most efficient at representing hierarchical structures.

However, the engine provides 4 specialized projections:

### 1. Poincaré Ball (Hierarchy)
- **Use case**: General similarity and taxonomic depth.
- **Metric**: Geodesic distance $d(u,v) = \text{arcosh}(1 + \frac{2\|u-v\|^2}{(1-\|u\|^2)(1-\|v\|^2)})$.
- **Logic**: Objects near the boundary are more specific; objects near the center are more abstract.

### 2. Klein Model (Rational Reasoning)
- **Use case**: Collinearity checks and logical chain verification.
- **Logic**: In Klein space, hyperbolic geodesics are straight lines. This transforms complex manifold navigation into O(1) linear checks.
- **Operation**: `PROJECT_TO_KLEIN(u) = \frac{u}{1+\|u\|^2}`.

### 3. Minkowski Spacetime (Causality)
- **Use case**: Causal auditing and temporal scrubbing.
- **Metric**: Lorentzian interval $ds^2 = -c^2\Delta t^2 + \|\Delta x\|^2$.
- **Operation**: The database treats the `created_at` timestamp as the 4th dimension ($t$) and the Poincaré embedding as the spatial vector ($x$).
- **Agential Role**: Ensure that a refined concept (effect) always falls within the "future light cone" of its foundational concept (cause).

### 4. Riemann Sphere (Synthesis)
- **Use case**: Combining conflicting ideas (Hegelian Dialectic).
- **Logic**: Two points on opposite sides of the Poincaré ball represent total antithesis. Projecting them onto the Riemann Sphere allows for a spherical midpoint that represents a "shallower" synthesis point.

## Geometric Transition Table

| From | To | Method | Stability |
|---|---|---|---|
| Poincaré | Klein | Conformal mapping | High |
| Poincaré | Riemann | Stereographic projection | Med |
| Poincaré | Minkowski | Conformal integration (t-axis) | High |

## Implementation
The `nietzsche-hyp-ops` crate handles these transitions using high-precision f64 arithmetic, ensuring that cascaded roundtrip errors remain below 1e-4.
