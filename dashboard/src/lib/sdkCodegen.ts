/**
 * SDK Code Generator — produces NQL, Python, and TypeScript code snippets
 * from structured query parameters. Pure client-side, no server dependency.
 */

export interface CodegenOutput {
    nql: string
    python: string
    typescript: string
}

// ── MATCH Query Builder ─────────────────────────────────────

export interface MatchParams {
    alias?: string
    label?: string
    conditions: { field: string; op: string; value: string }[]
    orderBy?: { field: string; dir: "ASC" | "DESC" }
    limit?: number
    skip?: number
    returnFields?: string[]
    collection?: string
}

export function generateMatchCode(p: MatchParams): CodegenOutput {
    const alias = p.alias || "n"
    const pattern = p.label ? `(${alias}:${p.label})` : `(${alias})`
    const where = p.conditions.length > 0
        ? " WHERE " + p.conditions.map(c => `${alias}.${c.field} ${c.op} ${formatValue(c.value)}`).join(" AND ")
        : ""
    const order = p.orderBy ? ` ORDER BY ${alias}.${p.orderBy.field} ${p.orderBy.dir}` : ""
    const limit = p.limit ? ` LIMIT ${p.limit}` : ""
    const skip = p.skip ? ` SKIP ${p.skip}` : ""
    const ret = p.returnFields?.length ? ` RETURN ${p.returnFields.join(", ")}` : ` RETURN ${alias}`
    const nql = `MATCH ${pattern}${where}${order}${limit}${skip}${ret}`
    const col = p.collection || "eva_core"

    return {
        nql,
        python: `from nietzschedb import NietzscheClient

client = NietzscheClient("localhost:50051")
result = client.query(
    """${nql}""",
    collection="${col}"
)
for node in result.nodes:
    print(node.id, node.energy)`,
        typescript: `import { NietzscheClient } from 'nietzschedb';

const client = new NietzscheClient('localhost:50051');
const result = await client.query(
    \`${nql}\`,
    { collection: '${col}' }
);
result.nodes.forEach(node => console.log(node.id, node.energy));`,
    }
}

// ── CREATE Query Builder ────────────────────────────────────

export interface CreateParams {
    label: string
    properties: { key: string; value: string }[]
    collection?: string
}

export function generateCreateCode(p: CreateParams): CodegenOutput {
    const props = p.properties.map(pr => `${pr.key}: ${formatValue(pr.value)}`).join(", ")
    const nql = `CREATE (n:${p.label} {${props}}) RETURN n`
    const col = p.collection || "eva_core"

    return {
        nql,
        python: `from nietzschedb import NietzscheClient

client = NietzscheClient("localhost:50051")
result = client.query(
    """${nql}""",
    collection="${col}"
)
print("Created:", result.nodes[0].id)`,
        typescript: `import { NietzscheClient } from 'nietzschedb';

const client = new NietzscheClient('localhost:50051');
const result = await client.query(
    \`${nql}\`,
    { collection: '${col}' }
);
console.log('Created:', result.nodes[0].id);`,
    }
}

// ── DIFFUSE Query Builder ───────────────────────────────────

export interface DiffuseParams {
    nodeId: string
    tValues?: number[]
    maxHops?: number
    collection?: string
}

export function generateDiffuseCode(p: DiffuseParams): CodegenOutput {
    const t = p.tValues?.length ? ` WITH t = [${p.tValues.join(", ")}]` : ""
    const hops = p.maxHops ? ` MAX_HOPS ${p.maxHops}` : ""
    const nql = `DIFFUSE FROM $node${t}${hops} RETURN path`
    const col = p.collection || "eva_core"

    return {
        nql,
        python: `from nietzschedb import NietzscheClient

client = NietzscheClient("localhost:50051")
result = client.diffuse(
    source_ids=["${p.nodeId}"],
    t_values=[${(p.tValues || [0.1, 1.0, 10.0]).join(", ")}],
    k_chebyshev=${p.maxHops || 5},
    collection="${col}"
)
for scale in result.scales:
    print(f"t={scale.t}: {len(scale.node_ids)} nodes reached")`,
        typescript: `import { NietzscheClient } from 'nietzschedb';

const client = new NietzscheClient('localhost:50051');
const result = await client.diffuse(
    ['${p.nodeId}'],
    [${(p.tValues || [0.1, 1.0, 10.0]).join(", ")}],
    ${p.maxHops || 5},
    { collection: '${col}' }
);
result.scales.forEach(s => console.log(\`t=\${s.t}: \${s.nodeIds.length} nodes\`));`,
    }
}

// ── DREAM Query Builder ─────────────────────────────────────

export interface DreamParams {
    nodeId: string
    depth?: number
    noise?: number
    collection?: string
}

export function generateDreamCode(p: DreamParams): CodegenOutput {
    const depth = p.depth ? ` DEPTH ${p.depth}` : ""
    const noise = p.noise ? ` NOISE ${p.noise}` : ""
    const nql = `DREAM FROM $node${depth}${noise}`
    const col = p.collection || "eva_core"

    return {
        nql,
        python: `from nietzschedb import NietzscheClient

client = NietzscheClient("localhost:50051")
# Generate dream snapshot
result = client.query(
    """${nql}""",
    collection="${col}",
    params={"node": "${p.nodeId}"}
)
print("Dream generated")

# Later: APPLY DREAM $dream_id  or  REJECT DREAM $dream_id`,
        typescript: `import { NietzscheClient } from 'nietzschedb';

const client = new NietzscheClient('localhost:50051');
const result = await client.query(
    \`${nql}\`,
    { collection: '${col}', params: { node: '${p.nodeId}' } }
);
console.log('Dream generated');`,
    }
}

// ── Synthesis Code ──────────────────────────────────────────

export interface SynthesisCodeParams {
    nodeIdA: string
    nodeIdB: string
    collection?: string
}

export function generateSynthesisCode(p: SynthesisCodeParams): CodegenOutput {
    const col = p.collection || "eva_core"
    return {
        nql: `-- Synthesis is a gRPC/REST operation, not NQL (yet)
-- Future NQL: SYNTHESIZE (a {id:"${p.nodeIdA}"}) WITH (b {id:"${p.nodeIdB}"}) YIELD (c)`,
        python: `from nietzschedb import NietzscheClient

client = NietzscheClient("localhost:50051")
result = client.synthesis(
    "${p.nodeIdA}",
    "${p.nodeIdB}",
    collection="${col}"
)
print(f"Synthesis point: {result.synthesis_coords}")
print(f"Nearest node: {result.nearest_node_id} (dist: {result.nearest_distance:.4f})")`,
        typescript: `import { NietzscheClient } from 'nietzschedb';

const client = new NietzscheClient('localhost:50051');
const result = await client.synthesis(
    '${p.nodeIdA}',
    '${p.nodeIdB}',
    { collection: '${col}' }
);
console.log('Synthesis point:', result.synthesisCoords);
console.log('Nearest node:', result.nearestNodeId, 'dist:', result.nearestDistance);`,
    }
}

// ── Causal Code ─────────────────────────────────────────────

export function generateCausalCode(nodeId: string, direction: string, collection?: string): CodegenOutput {
    const col = collection || "eva_core"
    return {
        nql: `-- Causal exploration is a gRPC/REST operation
-- Future NQL: CAUSAL NEIGHBORS $node DIRECTION "${direction}"`,
        python: `from nietzschedb import NietzscheClient

client = NietzscheClient("localhost:50051")
result = client.causal_neighbors(
    "${nodeId}",
    direction="${direction}",
    collection="${col}"
)
for edge in result.edges:
    print(f"{edge.from_node_id} -> {edge.to_node_id} [{edge.causal_type}] ds²={edge.minkowski_interval:.4f}")`,
        typescript: `import { NietzscheClient } from 'nietzschedb';

const client = new NietzscheClient('localhost:50051');
const result = await client.causalNeighbors(
    '${nodeId}',
    { direction: '${direction}', collection: '${col}' }
);
result.edges.forEach(e =>
    console.log(\`\${e.fromNodeId} -> \${e.toNodeId} [\${e.causalType}] ds²=\${e.minkowskiInterval}\`)
);`,
    }
}

// ── Klein Path Code ─────────────────────────────────────────

export function generateKleinCode(startId: string, goalId: string, collection?: string): CodegenOutput {
    const col = collection || "eva_core"
    return {
        nql: `-- Klein pathfinding is a gRPC/REST operation
-- Future NQL: KLEIN PATH FROM $start TO $goal`,
        python: `from nietzschedb import NietzscheClient

client = NietzscheClient("localhost:50051")
result = client.klein_path(
    "${startId}",
    "${goalId}",
    collection="${col}"
)
if result.found:
    print(f"Path ({result.hops} hops, cost={result.cost:.4f}):")
    print(" -> ".join(result.path))
else:
    print("No path found")`,
        typescript: `import { NietzscheClient } from 'nietzschedb';

const client = new NietzscheClient('localhost:50051');
const result = await client.kleinPath(
    '${startId}',
    '${goalId}',
    { collection: '${col}' }
);
if (result.found) {
    console.log(\`Path (\${result.hops} hops, cost=\${result.cost.toFixed(4)}):\`);
    console.log(result.path.join(' -> '));
}`,
    }
}

// ── Schema Code ─────────────────────────────────────────────

export interface SchemaCodeParams {
    nodeType: string
    requiredFields: string[]
    fieldTypes: { field_name: string; field_type: string }[]
}

export function generateSchemaCode(p: SchemaCodeParams): CodegenOutput {
    const pyTypeMap: Record<string, string> = {
        string: "str", number: "float", bool: "bool", array: "List[Any]", object: "dict",
    }
    const tsTypeMap: Record<string, string> = {
        string: "string", number: "number", bool: "boolean", array: "unknown[]", object: "Record<string, unknown>",
    }

    const pyFields = p.fieldTypes.map(f => {
        const t = pyTypeMap[f.field_type] || "Any"
        const opt = !p.requiredFields.includes(f.field_name)
        return `    ${f.field_name}: ${opt ? `Optional[${t}]` : t}${opt ? " = None" : ""}`
    }).join("\n")

    const tsFields = p.fieldTypes.map(f => {
        const t = tsTypeMap[f.field_type] || "unknown"
        const opt = !p.requiredFields.includes(f.field_name) ? "?" : ""
        return `    ${f.field_name}${opt}: ${t};`
    }).join("\n")

    const nqlProps = p.fieldTypes.map(f => {
        const ex = f.field_type === "string" ? '"example"' : f.field_type === "number" ? "0.0" : f.field_type === "bool" ? "true" : "null"
        return `${f.field_name}: ${ex}`
    }).join(", ")

    return {
        nql: `CREATE (n:${p.nodeType} {${nqlProps}}) RETURN n`,
        python: `from pydantic import BaseModel
from typing import Optional, List, Any
from nietzschedb.ogm import NietzscheNode

class ${p.nodeType}(NietzscheNode):
    """Auto-generated from NietzscheDB schema."""
${pyFields}

# Usage:
# obj = ${p.nodeType}(${p.requiredFields.map(f => `${f}=...`).join(", ")})
# obj.save(db)`,
        typescript: `// Auto-generated from NietzscheDB schema
interface ${p.nodeType} {
${tsFields}
}

// Usage:
// await client.query(\`CREATE (n:${p.nodeType} {...}) RETURN n\`);`,
    }
}

// ── MERGE Query Builder ─────────────────────────────────────

export interface MergeParams {
    label: string
    matchProps: { key: string; value: string }[]
    onCreateProps: { key: string; value: string }[]
    collection?: string
}

export function generateMergeCode(p: MergeParams): CodegenOutput {
    const matchStr = p.matchProps.map(pr => `${pr.key}: ${formatValue(pr.value)}`).join(", ")
    const onCreate = p.onCreateProps.length
        ? ` ON CREATE SET ${p.onCreateProps.map(pr => `n.${pr.key} = ${formatValue(pr.value)}`).join(", ")}`
        : ""
    const nql = `MERGE (n:${p.label} {${matchStr}})${onCreate} RETURN n`
    const col = p.collection || "eva_core"

    return {
        nql,
        python: `from nietzschedb import NietzscheClient

client = NietzscheClient("localhost:50051")
result = client.query(
    """${nql}""",
    collection="${col}"
)
print("Merged:", result.nodes[0].id)`,
        typescript: `import { NietzscheClient } from 'nietzschedb';

const client = new NietzscheClient('localhost:50051');
const result = await client.query(
    \`${nql}\`,
    { collection: '${col}' }
);
console.log('Merged:', result.nodes[0].id);`,
    }
}

// ── Helpers ─────────────────────────────────────────────────

function formatValue(v: string): string {
    const n = Number(v)
    if (!isNaN(n) && v.trim() !== "") return v
    if (v === "true" || v === "false") return v
    return `"${v.replace(/"/g, '\\"')}"`
}
