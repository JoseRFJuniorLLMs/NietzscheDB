import { useState, useEffect, useMemo } from "react"
import {
    Wand2, Play, Loader2, AlertCircle, Plus, Trash2, Copy, Check,
    Search, GitMerge, Waves, Moon, FileText, HelpCircle,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ScrollArea } from "@/components/ui/scroll-area"
import { api, DEFAULT_COLLECTION, executeNql } from "@/lib/api"
import {
    generateMatchCode, generateCreateCode, generateDiffuseCode,
    generateDreamCode, generateMergeCode,
    type MatchParams, type CreateParams, type DiffuseParams, type DreamParams, type MergeParams,
    type CodegenOutput,
} from "@/lib/sdkCodegen"

// ── Query Types ─────────────────────────────────────────────

interface QueryTypeDef {
    id: string; label: string; icon: React.ElementType; desc: string
}

const QUERY_TYPES: QueryTypeDef[] = [
    { id: "match",   label: "MATCH",   icon: Search,     desc: "Read nodes with filters" },
    { id: "create",  label: "CREATE",  icon: Plus,       desc: "Create new node" },
    { id: "merge",   label: "MERGE",   icon: GitMerge,   desc: "Upsert node" },
    { id: "diffuse", label: "DIFFUSE", icon: Waves,      desc: "Heat diffusion walk" },
    { id: "dream",   label: "DREAM",   icon: Moon,       desc: "Generate dream snapshot" },
    { id: "narrate", label: "NARRATE", icon: FileText,   desc: "Generate narrative" },
    { id: "explain", label: "EXPLAIN", icon: HelpCircle, desc: "Explain query plan" },
]

// ── Main Component ──────────────────────────────────────────

export default function QueryBuilderPage() {
    const [collection, setCollection] = useState(DEFAULT_COLLECTION)
    const [collections, setCollections] = useState<string[]>([])
    const [selectedType, setSelectedType] = useState<string>("match")
    const [results, setResults] = useState<any[] | null>(null)
    const [executing, setExecuting] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [copied, setCopied] = useState<string | null>(null)

    // MATCH state
    const [matchLabel, setMatchLabel] = useState("")
    const [conditions, setConditions] = useState<{ field: string; op: string; value: string }[]>([])
    const [orderField, setOrderField] = useState("")
    const [orderDir, setOrderDir] = useState<"ASC" | "DESC">("DESC")
    const [limit, setLimit] = useState(10)
    const [skip, setSkip] = useState(0)

    // CREATE state
    const [createLabel, setCreateLabel] = useState("Semantic")
    const [createProps, setCreateProps] = useState<{ key: string; value: string }[]>([{ key: "content", value: "" }])

    // MERGE state
    const [mergeLabel, setMergeLabel] = useState("Semantic")
    const [mergeMatchProps, setMergeMatchProps] = useState<{ key: string; value: string }[]>([{ key: "", value: "" }])
    const [mergeCreateProps, setMergeCreateProps] = useState<{ key: string; value: string }[]>([])

    // DIFFUSE state
    const [diffuseNodeId, setDiffuseNodeId] = useState("")
    const [diffuseTValues, setDiffuseTValues] = useState("0.1, 1.0, 10.0")
    const [diffuseMaxHops, setDiffuseMaxHops] = useState(5)

    // DREAM state
    const [dreamNodeId, setDreamNodeId] = useState("")
    const [dreamDepth, setDreamDepth] = useState(3)
    const [dreamNoise, setDreamNoise] = useState(0.1)

    // NARRATE state
    const [narrateWindow, setNarrateWindow] = useState(24)
    const [narrateFormat, setNarrateFormat] = useState("json")

    useEffect(() => {
        api.get("/collections").then(r => {
            setCollections((r.data as { name: string }[]).map(c => c.name))
        }).catch(() => {})
    }, [])

    // Generate code from current state
    const generated: CodegenOutput = useMemo(() => {
        switch (selectedType) {
            case "match":
                return generateMatchCode({
                    label: matchLabel || undefined,
                    conditions,
                    orderBy: orderField ? { field: orderField, dir: orderDir } : undefined,
                    limit: limit || undefined,
                    skip: skip || undefined,
                    collection,
                } as MatchParams)
            case "create":
                return generateCreateCode({
                    label: createLabel,
                    properties: createProps.filter(p => p.key),
                    collection,
                } as CreateParams)
            case "merge":
                return generateMergeCode({
                    label: mergeLabel,
                    matchProps: mergeMatchProps.filter(p => p.key),
                    onCreateProps: mergeCreateProps.filter(p => p.key),
                    collection,
                } as MergeParams)
            case "diffuse":
                return generateDiffuseCode({
                    nodeId: diffuseNodeId,
                    tValues: diffuseTValues.split(",").map(s => parseFloat(s.trim())).filter(n => !isNaN(n)),
                    maxHops: diffuseMaxHops,
                    collection,
                } as DiffuseParams)
            case "dream":
                return generateDreamCode({
                    nodeId: dreamNodeId,
                    depth: dreamDepth,
                    noise: dreamNoise,
                    collection,
                } as DreamParams)
            case "narrate":
                return {
                    nql: `NARRATE IN "${collection}" WINDOW ${narrateWindow} FORMAT ${narrateFormat}`,
                    python: `client.query('NARRATE IN "${collection}" WINDOW ${narrateWindow} FORMAT ${narrateFormat}')`,
                    typescript: `await client.query(\`NARRATE IN "${collection}" WINDOW ${narrateWindow} FORMAT ${narrateFormat}\`);`,
                }
            case "explain":
                return {
                    nql: `EXPLAIN MATCH (n${matchLabel ? `:${matchLabel}` : ""}) RETURN n`,
                    python: `client.query('EXPLAIN MATCH (n) RETURN n', collection="${collection}")`,
                    typescript: `await client.query('EXPLAIN MATCH (n) RETURN n', { collection: '${collection}' });`,
                }
            default:
                return { nql: "", python: "", typescript: "" }
        }
    }, [selectedType, matchLabel, conditions, orderField, orderDir, limit, skip,
        createLabel, createProps, mergeLabel, mergeMatchProps, mergeCreateProps,
        diffuseNodeId, diffuseTValues, diffuseMaxHops,
        dreamNodeId, dreamDepth, dreamNoise,
        narrateWindow, narrateFormat, collection])

    const execute = async () => {
        setExecuting(true); setError(null); setResults(null)
        try {
            const res = await executeNql(generated.nql, collection)
            if (res.error) { setError(res.error) }
            else { setResults(res.nodes as any[]) }
        } catch (e: any) {
            setError(e.response?.data?.error || e.message)
        } finally { setExecuting(false) }
    }

    const copyToClipboard = (text: string, lang: string) => {
        navigator.clipboard.writeText(text)
        setCopied(lang)
        setTimeout(() => setCopied(null), 2000)
    }

    return (
        <div className="space-y-6 fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold flex items-center gap-2">
                        <Wand2 className="h-6 w-6" /> Visual Query Builder
                    </h1>
                    <p className="text-sm text-muted-foreground mt-1">
                        Build NQL queries visually — generates NQL, Python SDK, and TypeScript SDK code
                    </p>
                </div>
                <div className="w-48">
                    <Select value={collection} onValueChange={setCollection}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                            {collections.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                        </SelectContent>
                    </Select>
                </div>
            </div>

            {/* Query Type Selector */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
                {QUERY_TYPES.map(qt => (
                    <Card key={qt.id}
                        className={`cursor-pointer transition-all hover:border-primary/50 ${selectedType === qt.id ? "border-primary ring-1 ring-primary/30" : ""}`}
                        onClick={() => setSelectedType(qt.id)}>
                        <CardContent className="p-3 text-center">
                            <qt.icon className="h-5 w-5 mx-auto mb-1" />
                            <p className="text-sm font-bold">{qt.label}</p>
                            <p className="text-[10px] text-muted-foreground">{qt.desc}</p>
                        </CardContent>
                    </Card>
                ))}
            </div>

            {/* Builder + Code Panel */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Builder Panel */}
                <Card>
                    <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                            <CardTitle className="text-base">Parameters</CardTitle>
                            <Button onClick={execute} disabled={executing || !generated.nql}>
                                {executing ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Running...</> : <><Play className="h-4 w-4 mr-2" /> Execute</>}
                            </Button>
                        </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        {selectedType === "match" && (
                            <MatchBuilder
                                label={matchLabel} setLabel={setMatchLabel}
                                conditions={conditions} setConditions={setConditions}
                                orderField={orderField} setOrderField={setOrderField}
                                orderDir={orderDir} setOrderDir={setOrderDir}
                                limit={limit} setLimit={setLimit}
                                skip={skip} setSkip={setSkip}
                            />
                        )}
                        {selectedType === "create" && (
                            <CreateBuilder
                                label={createLabel} setLabel={setCreateLabel}
                                props={createProps} setProps={setCreateProps}
                            />
                        )}
                        {selectedType === "merge" && (
                            <MergeBuilder
                                label={mergeLabel} setLabel={setMergeLabel}
                                matchProps={mergeMatchProps} setMatchProps={setMergeMatchProps}
                                createProps={mergeCreateProps} setCreateProps={setMergeCreateProps}
                            />
                        )}
                        {selectedType === "diffuse" && (
                            <DiffuseBuilder
                                nodeId={diffuseNodeId} setNodeId={setDiffuseNodeId}
                                tValues={diffuseTValues} setTValues={setDiffuseTValues}
                                maxHops={diffuseMaxHops} setMaxHops={setDiffuseMaxHops}
                            />
                        )}
                        {selectedType === "dream" && (
                            <DreamBuilder
                                nodeId={dreamNodeId} setNodeId={setDreamNodeId}
                                depth={dreamDepth} setDepth={setDreamDepth}
                                noise={dreamNoise} setNoise={setDreamNoise}
                            />
                        )}
                        {selectedType === "narrate" && (
                            <div className="space-y-3">
                                <div className="space-y-1">
                                    <Label>Window (hours)</Label>
                                    <Input type="number" value={narrateWindow} onChange={e => setNarrateWindow(Number(e.target.value))} />
                                </div>
                                <div className="space-y-1">
                                    <Label>Format</Label>
                                    <Select value={narrateFormat} onValueChange={setNarrateFormat}>
                                        <SelectTrigger><SelectValue /></SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="json">JSON</SelectItem>
                                            <SelectItem value="markdown">Markdown</SelectItem>
                                            <SelectItem value="text">Plain Text</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                            </div>
                        )}
                        {selectedType === "explain" && (
                            <p className="text-sm text-muted-foreground">
                                EXPLAIN wraps the current MATCH query to show the execution plan and cost estimate.
                                Configure the MATCH parameters to build the inner query.
                            </p>
                        )}

                        {error && (
                            <div className="flex items-center gap-2 text-destructive text-sm">
                                <AlertCircle className="h-4 w-4 shrink-0" />
                                <span className="font-mono text-xs">{error}</span>
                            </div>
                        )}
                    </CardContent>
                </Card>

                {/* Code Panel */}
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-base">Generated Code</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <Tabs defaultValue="nql">
                            <TabsList>
                                <TabsTrigger value="nql">NQL</TabsTrigger>
                                <TabsTrigger value="python">Python SDK</TabsTrigger>
                                <TabsTrigger value="typescript">TypeScript SDK</TabsTrigger>
                            </TabsList>
                            {(["nql", "python", "typescript"] as const).map(lang => (
                                <TabsContent key={lang} value={lang}>
                                    <div className="relative">
                                        <Button variant="ghost" size="icon" className="absolute top-2 right-2 h-7 w-7"
                                            onClick={() => copyToClipboard(generated[lang], lang)}>
                                            {copied === lang ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
                                        </Button>
                                        <ScrollArea className="max-h-80">
                                            <pre className="text-xs font-mono bg-muted/50 rounded-md p-3 whitespace-pre-wrap">{generated[lang]}</pre>
                                        </ScrollArea>
                                    </div>
                                </TabsContent>
                            ))}
                        </Tabs>
                    </CardContent>
                </Card>
            </div>

            {/* Results */}
            {results && (
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-base">Results ({results.length} nodes)</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <ScrollArea className="max-h-96">
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead>ID</TableHead>
                                        <TableHead>Type</TableHead>
                                        <TableHead>Energy</TableHead>
                                        <TableHead>Depth</TableHead>
                                        <TableHead>Content</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {results.map((n: any, i: number) => (
                                        <TableRow key={i}>
                                            <TableCell className="font-mono text-xs">{(n.id || "").substring(0, 12)}...</TableCell>
                                            <TableCell><Badge variant="outline">{n.node_type}</Badge></TableCell>
                                            <TableCell className="font-mono text-xs">{(n.energy ?? 0).toFixed(4)}</TableCell>
                                            <TableCell className="font-mono text-xs">{(n.depth ?? 0).toFixed(4)}</TableCell>
                                            <TableCell className="max-w-xs truncate text-xs">{JSON.stringify(n.content).substring(0, 80)}</TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </ScrollArea>
                    </CardContent>
                </Card>
            )}
        </div>
    )
}

// ── Builder Sub-Components ──────────────────────────────────

const NODE_TYPES = ["", "Semantic", "Episodic", "Concept", "DreamSnapshot", "Somatic", "Linguistic", "Composite"]
const OPERATORS = ["=", "!=", ">", "<", ">=", "<=", "CONTAINS", "STARTS_WITH", "ENDS_WITH"]
const COMMON_FIELDS = ["energy", "depth", "hausdorff", "node_type", "created_at", "arousal", "valence"]

function MatchBuilder({ label, setLabel, conditions, setConditions, orderField, setOrderField, orderDir, setOrderDir, limit, setLimit, skip, setSkip }: {
    label: string; setLabel: (v: string) => void
    conditions: { field: string; op: string; value: string }[]
    setConditions: (v: { field: string; op: string; value: string }[]) => void
    orderField: string; setOrderField: (v: string) => void
    orderDir: "ASC" | "DESC"; setOrderDir: (v: "ASC" | "DESC") => void
    limit: number; setLimit: (v: number) => void
    skip: number; setSkip: (v: number) => void
}) {
    return (
        <div className="space-y-4">
            {/* Label */}
            <div className="space-y-1">
                <Label>Node Type (optional)</Label>
                <Select value={label} onValueChange={setLabel}>
                    <SelectTrigger><SelectValue placeholder="Any type..." /></SelectTrigger>
                    <SelectContent>
                        {NODE_TYPES.map(t => <SelectItem key={t || "__any"} value={t}>{t || "(any)"}</SelectItem>)}
                    </SelectContent>
                </Select>
            </div>

            {/* WHERE conditions */}
            <div className="space-y-2">
                <Label>WHERE Conditions</Label>
                {conditions.map((c, i) => (
                    <div key={i} className="flex gap-2 items-center">
                        <Select value={c.field} onValueChange={v => { const next = [...conditions]; next[i] = { ...next[i], field: v }; setConditions(next) }}>
                            <SelectTrigger className="w-36"><SelectValue placeholder="field" /></SelectTrigger>
                            <SelectContent>
                                {COMMON_FIELDS.map(f => <SelectItem key={f} value={f}>{f}</SelectItem>)}
                            </SelectContent>
                        </Select>
                        <Select value={c.op} onValueChange={v => { const next = [...conditions]; next[i] = { ...next[i], op: v }; setConditions(next) }}>
                            <SelectTrigger className="w-28"><SelectValue /></SelectTrigger>
                            <SelectContent>
                                {OPERATORS.map(o => <SelectItem key={o} value={o}>{o}</SelectItem>)}
                            </SelectContent>
                        </Select>
                        <Input className="flex-1" placeholder="value" value={c.value}
                            onChange={e => { const next = [...conditions]; next[i] = { ...next[i], value: e.target.value }; setConditions(next) }} />
                        <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0" onClick={() => setConditions(conditions.filter((_, j) => j !== i))}>
                            <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                    </div>
                ))}
                <Button variant="outline" size="sm" onClick={() => setConditions([...conditions, { field: "energy", op: ">", value: "0.5" }])}>
                    <Plus className="h-3.5 w-3.5 mr-1" /> Add Filter
                </Button>
            </div>

            {/* ORDER BY / LIMIT / SKIP */}
            <div className="flex flex-wrap gap-4 items-end">
                <div className="space-y-1">
                    <Label>Order By</Label>
                    <Select value={orderField} onValueChange={setOrderField}>
                        <SelectTrigger className="w-32"><SelectValue placeholder="none" /></SelectTrigger>
                        <SelectContent>
                            <SelectItem value="">(none)</SelectItem>
                            {COMMON_FIELDS.map(f => <SelectItem key={f} value={f}>{f}</SelectItem>)}
                        </SelectContent>
                    </Select>
                </div>
                {orderField && (
                    <div className="space-y-1">
                        <Label>Direction</Label>
                        <Select value={orderDir} onValueChange={v => setOrderDir(v as "ASC" | "DESC")}>
                            <SelectTrigger className="w-24"><SelectValue /></SelectTrigger>
                            <SelectContent>
                                <SelectItem value="ASC">ASC</SelectItem>
                                <SelectItem value="DESC">DESC</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                )}
                <div className="space-y-1">
                    <Label>Limit</Label>
                    <Input type="number" value={limit} onChange={e => setLimit(Number(e.target.value))} className="w-20" min={1} max={1000} />
                </div>
                <div className="space-y-1">
                    <Label>Skip</Label>
                    <Input type="number" value={skip} onChange={e => setSkip(Number(e.target.value))} className="w-20" min={0} />
                </div>
            </div>
        </div>
    )
}

function CreateBuilder({ label, setLabel, props, setProps }: {
    label: string; setLabel: (v: string) => void
    props: { key: string; value: string }[]; setProps: (v: { key: string; value: string }[]) => void
}) {
    return (
        <div className="space-y-4">
            <div className="space-y-1">
                <Label>Node Type</Label>
                <Select value={label} onValueChange={setLabel}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                        {NODE_TYPES.filter(Boolean).map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}
                    </SelectContent>
                </Select>
            </div>
            <div className="space-y-2">
                <Label>Properties</Label>
                {props.map((p, i) => (
                    <div key={i} className="flex gap-2 items-center">
                        <Input className="w-36" placeholder="key" value={p.key}
                            onChange={e => { const next = [...props]; next[i] = { ...next[i], key: e.target.value }; setProps(next) }} />
                        <Input className="flex-1" placeholder="value" value={p.value}
                            onChange={e => { const next = [...props]; next[i] = { ...next[i], value: e.target.value }; setProps(next) }} />
                        <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0" onClick={() => setProps(props.filter((_, j) => j !== i))} disabled={props.length <= 1}>
                            <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                    </div>
                ))}
                <Button variant="outline" size="sm" onClick={() => setProps([...props, { key: "", value: "" }])}>
                    <Plus className="h-3.5 w-3.5 mr-1" /> Add Property
                </Button>
            </div>
        </div>
    )
}

function MergeBuilder({ label, setLabel, matchProps, setMatchProps, createProps, setCreateProps }: {
    label: string; setLabel: (v: string) => void
    matchProps: { key: string; value: string }[]; setMatchProps: (v: { key: string; value: string }[]) => void
    createProps: { key: string; value: string }[]; setCreateProps: (v: { key: string; value: string }[]) => void
}) {
    return (
        <div className="space-y-4">
            <div className="space-y-1">
                <Label>Node Type</Label>
                <Select value={label} onValueChange={setLabel}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                        {NODE_TYPES.filter(Boolean).map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}
                    </SelectContent>
                </Select>
            </div>
            <div className="space-y-2">
                <Label>Match Properties (identify existing node)</Label>
                <PropsEditor props={matchProps} setProps={setMatchProps} />
            </div>
            <div className="space-y-2">
                <Label>ON CREATE SET (properties for new node)</Label>
                <PropsEditor props={createProps} setProps={setCreateProps} />
            </div>
        </div>
    )
}

function DiffuseBuilder({ nodeId, setNodeId, tValues, setTValues, maxHops, setMaxHops }: {
    nodeId: string; setNodeId: (v: string) => void
    tValues: string; setTValues: (v: string) => void
    maxHops: number; setMaxHops: (v: number) => void
}) {
    return (
        <div className="space-y-3">
            <div className="space-y-1">
                <Label>Source Node ID</Label>
                <Input placeholder="UUID..." value={nodeId} onChange={e => setNodeId(e.target.value)} className="font-mono text-xs" />
            </div>
            <div className="space-y-1">
                <Label>t Values (comma-separated)</Label>
                <Input placeholder="0.1, 1.0, 10.0" value={tValues} onChange={e => setTValues(e.target.value)} />
                <p className="text-[10px] text-muted-foreground">Chebyshev multi-scale diffusion time constants</p>
            </div>
            <div className="space-y-1">
                <Label>Max Hops</Label>
                <Input type="number" value={maxHops} onChange={e => setMaxHops(Number(e.target.value))} className="w-24" min={1} max={20} />
            </div>
        </div>
    )
}

function DreamBuilder({ nodeId, setNodeId, depth, setDepth, noise, setNoise }: {
    nodeId: string; setNodeId: (v: string) => void
    depth: number; setDepth: (v: number) => void
    noise: number; setNoise: (v: number) => void
}) {
    return (
        <div className="space-y-3">
            <div className="space-y-1">
                <Label>Seed Node ID</Label>
                <Input placeholder="UUID..." value={nodeId} onChange={e => setNodeId(e.target.value)} className="font-mono text-xs" />
            </div>
            <div className="space-y-1">
                <Label>Depth</Label>
                <Input type="number" value={depth} onChange={e => setDepth(Number(e.target.value))} className="w-24" min={1} max={10} />
                <p className="text-[10px] text-muted-foreground">Dream generation depth (1-10)</p>
            </div>
            <div className="space-y-1">
                <Label>Noise</Label>
                <Input type="number" step="0.01" value={noise} onChange={e => setNoise(Number(e.target.value))} className="w-24" min={0} max={1} />
                <p className="text-[10px] text-muted-foreground">Perturbation noise (0.0-1.0)</p>
            </div>
        </div>
    )
}

function PropsEditor({ props, setProps }: {
    props: { key: string; value: string }[]
    setProps: (v: { key: string; value: string }[]) => void
}) {
    return (
        <>
            {props.map((p, i) => (
                <div key={i} className="flex gap-2 items-center">
                    <Input className="w-36" placeholder="key" value={p.key}
                        onChange={e => { const next = [...props]; next[i] = { ...next[i], key: e.target.value }; setProps(next) }} />
                    <Input className="flex-1" placeholder="value" value={p.value}
                        onChange={e => { const next = [...props]; next[i] = { ...next[i], value: e.target.value }; setProps(next) }} />
                    <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0" onClick={() => setProps(props.filter((_, j) => j !== i))}>
                        <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                </div>
            ))}
            <Button variant="outline" size="sm" onClick={() => setProps([...props, { key: "", value: "" }])}>
                <Plus className="h-3.5 w-3.5 mr-1" /> Add
            </Button>
        </>
    )
}
