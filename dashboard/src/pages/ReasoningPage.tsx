import { useState, useEffect } from "react"
import { Lightbulb, Play, Loader2, AlertCircle, ArrowRight, Copy, Check } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
    api, DEFAULT_COLLECTION,
    synthesisTwo, synthesisMulti,
    causalNeighbors, causalChain,
    kleinPath,
    type SynthesisResult, type CausalEdge, type KleinPathResult,
} from "@/lib/api"
import { generateSynthesisCode, generateCausalCode, generateKleinCode } from "@/lib/sdkCodegen"

export default function ReasoningPage() {
    const [collection, setCollection] = useState(DEFAULT_COLLECTION)
    const [collections, setCollections] = useState<string[]>([])

    useEffect(() => {
        api.get("/collections").then(r => {
            setCollections((r.data as { name: string }[]).map(c => c.name))
        }).catch(() => {})
    }, [])

    return (
        <div className="space-y-6 fade-in">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold flex items-center gap-2">
                        <Lightbulb className="h-6 w-6" /> Reasoning Tools
                    </h1>
                    <p className="text-sm text-muted-foreground mt-1">
                        Multi-manifold cognitive operations — Riemann Synthesis, Minkowski Causality, Klein Geodesics
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

            <Tabs defaultValue="synthesis" className="space-y-4">
                <TabsList className="flex-wrap h-auto gap-1">
                    <TabsTrigger value="synthesis">Synthesis (Riemann)</TabsTrigger>
                    <TabsTrigger value="causal">Causal Explorer (Minkowski)</TabsTrigger>
                    <TabsTrigger value="klein">Klein Pathfinder</TabsTrigger>
                </TabsList>

                <TabsContent value="synthesis">
                    <SynthesisTab collection={collection} />
                </TabsContent>
                <TabsContent value="causal">
                    <CausalTab collection={collection} />
                </TabsContent>
                <TabsContent value="klein">
                    <KleinTab collection={collection} />
                </TabsContent>
            </Tabs>
        </div>
    )
}

// ── Synthesis Tab ───────────────────────────────────────────

function SynthesisTab({ collection }: { collection: string }) {
    const [mode, setMode] = useState<"binary" | "multi">("binary")
    const [nodeIdA, setNodeIdA] = useState("")
    const [nodeIdB, setNodeIdB] = useState("")
    const [multiIds, setMultiIds] = useState("")
    const [result, setResult] = useState<SynthesisResult | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [showCode, setShowCode] = useState(false)
    const [copied, setCopied] = useState<string | null>(null)

    const execute = async () => {
        setLoading(true); setError(null); setResult(null)
        try {
            if (mode === "binary") {
                setResult(await synthesisTwo(nodeIdA.trim(), nodeIdB.trim(), collection))
            } else {
                const ids = multiIds.split(/[,\n]/).map(s => s.trim()).filter(Boolean)
                setResult(await synthesisMulti(ids, collection))
            }
        } catch (e: any) {
            setError(e.response?.data?.error || e.message)
        } finally { setLoading(false) }
    }

    const code = generateSynthesisCode({ nodeIdA, nodeIdB, collection })

    return (
        <div className="space-y-4">
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base">Dialectical Synthesis</CardTitle>
                    <CardDescription>
                        Compute the Hegelian synthesis (Thesis + Antithesis = Synthesis) via Riemann sphere projection.
                        The result is a new point in the Poincare ball that unifies the input concepts.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex gap-2">
                        <Button variant={mode === "binary" ? "default" : "outline"} size="sm" onClick={() => setMode("binary")}>Binary (2 nodes)</Button>
                        <Button variant={mode === "multi" ? "default" : "outline"} size="sm" onClick={() => setMode("multi")}>Multi (N nodes)</Button>
                    </div>

                    {mode === "binary" ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="space-y-1">
                                <Label>Node A (Thesis)</Label>
                                <Input placeholder="UUID..." value={nodeIdA} onChange={e => setNodeIdA(e.target.value)} className="font-mono text-xs" />
                            </div>
                            <div className="space-y-1">
                                <Label>Node B (Antithesis)</Label>
                                <Input placeholder="UUID..." value={nodeIdB} onChange={e => setNodeIdB(e.target.value)} className="font-mono text-xs" />
                            </div>
                        </div>
                    ) : (
                        <div className="space-y-1">
                            <Label>Node IDs (one per line or comma-separated)</Label>
                            <textarea className="w-full h-24 rounded-md border bg-background px-3 py-2 text-xs font-mono resize-none focus:outline-none focus:ring-1 focus:ring-ring"
                                placeholder="uuid-1&#10;uuid-2&#10;uuid-3"
                                value={multiIds} onChange={e => setMultiIds(e.target.value)} />
                        </div>
                    )}

                    <div className="flex gap-2">
                        <Button onClick={execute} disabled={loading}>
                            {loading ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Synthesizing...</> : <><Play className="h-4 w-4 mr-2" /> Synthesize</>}
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => setShowCode(!showCode)}>
                            {"</>"} SDK Code
                        </Button>
                    </div>

                    {error && <ErrorInline message={error} />}
                </CardContent>
            </Card>

            {result && (
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-base">Synthesis Result</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <MetricCard label="Nearest Node" value={result.nearest_node_id.substring(0, 12) + "..."} mono />
                            <MetricCard label="Distance to Nearest" value={result.nearest_distance.toFixed(6)} />
                            <MetricCard label="Dimensions" value={String(result.synthesis_coords.length)} />
                        </div>
                        <div className="mt-4">
                            <Label className="text-xs text-muted-foreground">Synthesis Coordinates</Label>
                            <pre className="text-xs font-mono bg-muted/50 rounded-md p-3 mt-1 overflow-x-auto">
                                [{result.synthesis_coords.map(c => c.toFixed(6)).join(", ")}]
                            </pre>
                        </div>
                    </CardContent>
                </Card>
            )}

            {showCode && <SDKCodePanel code={code} copied={copied} setCopied={setCopied} />}
        </div>
    )
}

// ── Causal Explorer Tab ─────────────────────────────────────

function CausalTab({ collection }: { collection: string }) {
    const [nodeId, setNodeId] = useState("")
    const [direction, setDirection] = useState("both")
    const [neighborsResult, setNeighborsResult] = useState<CausalEdge[] | null>(null)
    const [chainResult, setChainResult] = useState<{ chain_ids: string[]; edges: CausalEdge[] } | null>(null)
    const [chainDepth, setChainDepth] = useState(10)
    const [chainDir, setChainDir] = useState("past")
    const [loadingN, setLoadingN] = useState(false)
    const [loadingC, setLoadingC] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [showCode, setShowCode] = useState(false)
    const [copied, setCopied] = useState<string | null>(null)

    const exploreNeighbors = async () => {
        setLoadingN(true); setError(null); setNeighborsResult(null)
        try {
            const res = await causalNeighbors(nodeId.trim(), direction, collection)
            setNeighborsResult(res.edges)
        } catch (e: any) { setError(e.response?.data?.error || e.message) }
        finally { setLoadingN(false) }
    }

    const traceChain = async () => {
        setLoadingC(true); setError(null); setChainResult(null)
        try {
            setChainResult(await causalChain(nodeId.trim(), chainDepth, chainDir, collection))
        } catch (e: any) { setError(e.response?.data?.error || e.message) }
        finally { setLoadingC(false) }
    }

    const code = generateCausalCode(nodeId, direction, collection)

    const causalBadge = (type: string) => {
        const v = type === "Timelike" ? "default" : type === "Spacelike" ? "secondary" : "outline"
        const color = type === "Timelike" ? "bg-emerald-500/10 text-emerald-500 border-emerald-500/30"
            : type === "Spacelike" ? "bg-amber-500/10 text-amber-500 border-amber-500/30"
            : "bg-purple-500/10 text-purple-500 border-purple-500/30"
        return <Badge variant={v} className={`text-xs ${color}`}>{type}</Badge>
    }

    return (
        <div className="space-y-4">
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base">Causal Explorer</CardTitle>
                    <CardDescription>
                        Explore the Minkowski light cone of a node — discover what caused it (past) and what it caused (future).
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex flex-wrap gap-4 items-end">
                        <div className="space-y-1">
                            <Label>Node ID</Label>
                            <Input placeholder="UUID..." value={nodeId} onChange={e => setNodeId(e.target.value)} className="font-mono text-xs w-80" />
                        </div>
                        <div className="space-y-1">
                            <Label>Direction</Label>
                            <Select value={direction} onValueChange={setDirection}>
                                <SelectTrigger className="w-28"><SelectValue /></SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="past">Past</SelectItem>
                                    <SelectItem value="future">Future</SelectItem>
                                    <SelectItem value="both">Both</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                        <Button onClick={exploreNeighbors} disabled={loadingN || !nodeId.trim()}>
                            {loadingN ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Play className="h-4 w-4 mr-2" />}
                            Explore
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => setShowCode(!showCode)}>{"</>"} SDK</Button>
                    </div>
                    {error && <ErrorInline message={error} />}
                </CardContent>
            </Card>

            {/* Neighbors Results */}
            {neighborsResult && (
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-base">Causal Neighbors ({neighborsResult.length})</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <ScrollArea className="max-h-96">
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead>From</TableHead>
                                        <TableHead></TableHead>
                                        <TableHead>To</TableHead>
                                        <TableHead>ds²</TableHead>
                                        <TableHead>Causal Type</TableHead>
                                        <TableHead>Edge Type</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {neighborsResult.map((e, i) => (
                                        <TableRow key={i}>
                                            <TableCell className="font-mono text-xs">{e.from_node_id.substring(0, 12)}...</TableCell>
                                            <TableCell><ArrowRight className="h-3.5 w-3.5 text-muted-foreground" /></TableCell>
                                            <TableCell className="font-mono text-xs">{e.to_node_id.substring(0, 12)}...</TableCell>
                                            <TableCell className="font-mono text-xs text-emerald-400">{e.minkowski_interval.toFixed(4)}</TableCell>
                                            <TableCell>{causalBadge(e.causal_type)}</TableCell>
                                            <TableCell><Badge variant="outline" className="text-xs">{e.edge_type}</Badge></TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </ScrollArea>
                    </CardContent>
                </Card>
            )}

            {/* Causal Chain */}
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base">Causal Chain (Recursive BFS)</CardTitle>
                    <CardDescription>Trace the full causal chain following only timelike edges.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex flex-wrap gap-4 items-end">
                        <div className="space-y-1">
                            <Label>Max Depth</Label>
                            <Input type="number" value={chainDepth} onChange={e => setChainDepth(Number(e.target.value))} className="w-24" min={1} max={50} />
                        </div>
                        <div className="space-y-1">
                            <Label>Direction</Label>
                            <Select value={chainDir} onValueChange={setChainDir}>
                                <SelectTrigger className="w-28"><SelectValue /></SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="past">Past (WHY)</SelectItem>
                                    <SelectItem value="future">Future (WHAT)</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                        <Button onClick={traceChain} disabled={loadingC || !nodeId.trim()}>
                            {loadingC ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Play className="h-4 w-4 mr-2" />}
                            Trace Chain
                        </Button>
                    </div>

                    {chainResult && (
                        <div className="space-y-3">
                            <div className="flex gap-4">
                                <MetricCard label="Chain Length" value={String(chainResult.chain_ids.length)} />
                                <MetricCard label="Edges" value={String(chainResult.edges.length)} />
                            </div>
                            <div>
                                <Label className="text-xs text-muted-foreground">Chain Path</Label>
                                <div className="flex flex-wrap gap-1 mt-1">
                                    {chainResult.chain_ids.map((id, i) => (
                                        <span key={i} className="flex items-center gap-1">
                                            <Badge variant="outline" className="font-mono text-[10px]">{id.substring(0, 10)}</Badge>
                                            {i < chainResult.chain_ids.length - 1 && <ArrowRight className="h-3 w-3 text-muted-foreground" />}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
                </CardContent>
            </Card>

            {showCode && <SDKCodePanel code={code} copied={copied} setCopied={setCopied} />}
        </div>
    )
}

// ── Klein Pathfinder Tab ────────────────────────────────────

function KleinTab({ collection }: { collection: string }) {
    const [startId, setStartId] = useState("")
    const [goalId, setGoalId] = useState("")
    const [result, setResult] = useState<KleinPathResult | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [showCode, setShowCode] = useState(false)
    const [copied, setCopied] = useState<string | null>(null)

    const execute = async () => {
        setLoading(true); setError(null); setResult(null)
        try {
            setResult(await kleinPath(startId.trim(), goalId.trim(), collection))
        } catch (e: any) { setError(e.response?.data?.error || e.message) }
        finally { setLoading(false) }
    }

    const code = generateKleinCode(startId, goalId, collection)

    return (
        <div className="space-y-4">
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base">Klein Geodesic Pathfinder</CardTitle>
                    <CardDescription>
                        Find the shortest hyperbolic path between two nodes using Dijkstra in the Beltrami-Klein model,
                        where geodesics are straight lines.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex flex-wrap gap-4 items-end">
                        <div className="space-y-1">
                            <Label>Start Node</Label>
                            <Input placeholder="UUID..." value={startId} onChange={e => setStartId(e.target.value)} className="font-mono text-xs w-80" />
                        </div>
                        <div className="space-y-1">
                            <Label>Goal Node</Label>
                            <Input placeholder="UUID..." value={goalId} onChange={e => setGoalId(e.target.value)} className="font-mono text-xs w-80" />
                        </div>
                        <Button onClick={execute} disabled={loading || !startId.trim() || !goalId.trim()}>
                            {loading ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Finding path...</> : <><Play className="h-4 w-4 mr-2" /> Find Path</>}
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => setShowCode(!showCode)}>{"</>"} SDK</Button>
                    </div>
                    {error && <ErrorInline message={error} />}
                </CardContent>
            </Card>

            {result && (
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-base">
                            {result.found ? "Path Found" : "No Path Found"}
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        {result.found ? (
                            <div className="space-y-4">
                                <div className="grid grid-cols-3 gap-4">
                                    <MetricCard label="Hops" value={String(result.hops)} />
                                    <MetricCard label="Total Cost" value={result.cost.toFixed(6)} />
                                    <MetricCard label="Nodes in Path" value={String(result.path.length)} />
                                </div>
                                <div>
                                    <Label className="text-xs text-muted-foreground">Path</Label>
                                    <div className="flex flex-wrap gap-1 mt-1">
                                        {result.path.map((id, i) => (
                                            <span key={i} className="flex items-center gap-1">
                                                <Badge variant={i === 0 || i === result.path.length - 1 ? "default" : "outline"}
                                                    className="font-mono text-[10px]">
                                                    {id.substring(0, 12)}
                                                </Badge>
                                                {i < result.path.length - 1 && <ArrowRight className="h-3 w-3 text-muted-foreground" />}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <p className="text-muted-foreground text-sm">No path exists between these nodes in the directed graph (max 2000 nodes explored).</p>
                        )}
                    </CardContent>
                </Card>
            )}

            {showCode && <SDKCodePanel code={code} copied={copied} setCopied={setCopied} />}
        </div>
    )
}

// ── Shared Components ───────────────────────────────────────

function MetricCard({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
    return (
        <div className="rounded-lg border bg-muted/30 p-3">
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</p>
            <p className={`text-lg font-semibold ${mono ? "font-mono" : ""}`}>{value}</p>
        </div>
    )
}

function ErrorInline({ message }: { message: string }) {
    return (
        <div className="flex items-center gap-2 text-destructive text-sm">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span className="font-mono text-xs">{message}</span>
        </div>
    )
}

function SDKCodePanel({ code, copied, setCopied }: {
    code: { nql: string; python: string; typescript: string }
    copied: string | null; setCopied: (v: string | null) => void
}) {
    const copyToClipboard = (text: string, lang: string) => {
        navigator.clipboard.writeText(text)
        setCopied(lang)
        setTimeout(() => setCopied(null), 2000)
    }

    return (
        <Card>
            <CardHeader className="pb-2">
                <CardTitle className="text-sm">SDK Code</CardTitle>
            </CardHeader>
            <CardContent>
                <Tabs defaultValue="python">
                    <TabsList>
                        <TabsTrigger value="python">Python</TabsTrigger>
                        <TabsTrigger value="typescript">TypeScript</TabsTrigger>
                        <TabsTrigger value="nql">NQL</TabsTrigger>
                    </TabsList>
                    {(["python", "typescript", "nql"] as const).map(lang => (
                        <TabsContent key={lang} value={lang}>
                            <div className="relative">
                                <Button variant="ghost" size="icon" className="absolute top-2 right-2 h-7 w-7"
                                    onClick={() => copyToClipboard(code[lang], lang)}>
                                    {copied === lang ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
                                </Button>
                                <pre className="text-xs font-mono bg-muted/50 rounded-md p-3 whitespace-pre-wrap">{code[lang]}</pre>
                            </div>
                        </TabsContent>
                    ))}
                </Tabs>
            </CardContent>
        </Card>
    )
}
