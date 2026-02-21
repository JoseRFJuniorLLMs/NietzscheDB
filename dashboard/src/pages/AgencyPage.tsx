import { useState, useCallback } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
    Brain, Heart, Eye, TrendingUp, BookOpen, Flame, FlaskConical,
    Atom, CheckCircle2, Loader2, RefreshCw, AlertCircle,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
    Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import {
    getAgencyHealthLatest, getObserver, getEvolution, getNarrative,
    getDesires, fulfillDesire, counterfactualRemove, counterfactualAdd,
    quantumMap, quantumFidelity,
} from "@/lib/api"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

export default function AgencyPage() {
    return (
        <div className="space-y-6 fade-in">
            <div>
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Brain className="h-6 w-6" /> Autonomous Agency
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                    Daemons, MetaObserver, Evolution, Narrative, Desires, Counterfactual, Quantum
                </p>
            </div>

            <Tabs defaultValue="health" className="space-y-4">
                <TabsList className="flex-wrap h-auto gap-1">
                    <TabsTrigger value="health"><Heart className="h-3.5 w-3.5 mr-1" />Health</TabsTrigger>
                    <TabsTrigger value="observer"><Eye className="h-3.5 w-3.5 mr-1" />Observer</TabsTrigger>
                    <TabsTrigger value="evolution"><TrendingUp className="h-3.5 w-3.5 mr-1" />Evolution</TabsTrigger>
                    <TabsTrigger value="narrative"><BookOpen className="h-3.5 w-3.5 mr-1" />Narrative</TabsTrigger>
                    <TabsTrigger value="desires"><Flame className="h-3.5 w-3.5 mr-1" />Desires</TabsTrigger>
                    <TabsTrigger value="counterfactual"><FlaskConical className="h-3.5 w-3.5 mr-1" />Counterfactual</TabsTrigger>
                    <TabsTrigger value="quantum"><Atom className="h-3.5 w-3.5 mr-1" />Quantum</TabsTrigger>
                </TabsList>

                {/* ── Health Tab ─────────────────────────── */}
                <TabsContent value="health"><HealthTab /></TabsContent>
                <TabsContent value="observer"><ObserverTab /></TabsContent>
                <TabsContent value="evolution"><EvolutionTab /></TabsContent>
                <TabsContent value="narrative"><NarrativeTab /></TabsContent>
                <TabsContent value="desires"><DesiresTab /></TabsContent>
                <TabsContent value="counterfactual"><CounterfactualTab /></TabsContent>
                <TabsContent value="quantum"><QuantumTab /></TabsContent>
            </Tabs>
        </div>
    )
}

/* ── Health ────────────────────────────────────────────────── */
function HealthTab() {
    const { data, isLoading, error } = useQuery({
        queryKey: ["agency-health"],
        queryFn: getAgencyHealthLatest,
        refetchInterval: 10000,
    })

    if (isLoading) return <LoadingSkeleton />
    if (error) return <ErrorCard message="Could not fetch health report" />

    return (
        <Card>
            <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                    <Heart className="h-4 w-4" /> Latest Health Report
                </CardTitle>
            </CardHeader>
            <CardContent>
                {data ? (
                    <pre className="text-xs font-mono bg-muted p-4 rounded-lg overflow-auto max-h-96">
                        {JSON.stringify(data, null, 2)}
                    </pre>
                ) : (
                    <p className="text-sm text-muted-foreground">No health reports available yet.</p>
                )}
            </CardContent>
        </Card>
    )
}

/* ── Observer ─────────────────────────────────────────────── */
function ObserverTab() {
    const { data, isLoading, error } = useQuery({
        queryKey: ["agency-observer"],
        queryFn: getObserver,
        refetchInterval: 15000,
    })

    if (isLoading) return <LoadingSkeleton />
    if (error) return <ErrorCard message="Could not fetch Observer meta-node" />

    return (
        <Card>
            <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                    <Eye className="h-4 w-4" /> MetaObserver Identity
                </CardTitle>
                <CardDescription>The self-referential meta-node of the graph</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                {data ? (
                    <>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <MetricCard label="Observer ID" value={data.observer_id?.slice(0, 12) + "..."} />
                            <MetricCard label="Energy" value={data.energy?.toFixed(4)} />
                            <MetricCard label="Depth" value={data.depth?.toFixed(4)} />
                            <MetricCard label="Hausdorff" value={data.hausdorff_local?.toFixed(4)} />
                        </div>
                        <div className="flex items-center gap-2">
                            <Badge variant={data.is_observer ? "default" : "secondary"}>
                                {data.is_observer ? "Active Observer" : "Inactive"}
                            </Badge>
                        </div>
                        {data.content && (
                            <pre className="text-xs font-mono bg-muted p-3 rounded-lg overflow-auto max-h-48">
                                {JSON.stringify(data.content, null, 2)}
                            </pre>
                        )}
                    </>
                ) : (
                    <p className="text-sm text-muted-foreground">No Observer meta-node found.</p>
                )}
            </CardContent>
        </Card>
    )
}

/* ── Evolution ────────────────────────────────────────────── */
function EvolutionTab() {
    const { data, isLoading, error } = useQuery({
        queryKey: ["agency-evolution"],
        queryFn: getEvolution,
        refetchInterval: 10000,
    })

    if (isLoading) return <LoadingSkeleton />
    if (error) return <ErrorCard message="Could not fetch evolution state" />

    const chartData = data?.fitness_history?.map((f: number, i: number) => ({
        generation: i + 1,
        fitness: f,
    })) ?? []

    return (
        <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                    <CardContent className="pt-4">
                        <div className="text-sm text-muted-foreground">Generation</div>
                        <div className="text-3xl font-bold font-mono">{data?.generation ?? 0}</div>
                    </CardContent>
                </Card>
                <Card>
                    <CardContent className="pt-4">
                        <div className="text-sm text-muted-foreground">Last Strategy</div>
                        <div className="text-lg font-semibold">{data?.last_strategy ?? "—"}</div>
                    </CardContent>
                </Card>
            </div>

            {chartData.length > 0 && (
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-base">Fitness History</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="hsl(216 34% 17%)" />
                                <XAxis dataKey="generation" stroke="hsl(215.4 16.3% 56.9%)" fontSize={11} />
                                <YAxis stroke="hsl(215.4 16.3% 56.9%)" fontSize={11} />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: "hsl(224 71% 4%)",
                                        border: "1px solid hsl(216 34% 17%)",
                                        borderRadius: "8px",
                                        fontSize: "12px",
                                    }}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="fitness"
                                    stroke="hsl(263.4 70% 50.4%)"
                                    strokeWidth={2}
                                    dot={false}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            )}
        </div>
    )
}

/* ── Narrative ────────────────────────────────────────────── */
function NarrativeTab() {
    const { data, isLoading, error, refetch } = useQuery({
        queryKey: ["agency-narrative"],
        queryFn: getNarrative,
    })

    if (isLoading) return <LoadingSkeleton />
    if (error) return <ErrorCard message="Could not fetch narrative" />

    return (
        <Card>
            <CardHeader className="flex flex-row items-center justify-between">
                <div>
                    <CardTitle className="text-base flex items-center gap-2">
                        <BookOpen className="h-4 w-4" /> Graph Narrative
                    </CardTitle>
                    <CardDescription>Auto-generated narrative of graph evolution</CardDescription>
                </div>
                <Button size="sm" variant="outline" onClick={() => refetch()}>
                    <RefreshCw className="h-3.5 w-3.5 mr-1" /> Refresh
                </Button>
            </CardHeader>
            <CardContent>
                {data?.narrative ? (
                    <div className="prose prose-invert max-w-none text-sm leading-relaxed whitespace-pre-wrap">
                        {data.narrative}
                    </div>
                ) : (
                    <p className="text-sm text-muted-foreground">No narrative generated yet.</p>
                )}
            </CardContent>
        </Card>
    )
}

/* ── Desires ──────────────────────────────────────────────── */
function DesiresTab() {
    const qc = useQueryClient()
    const { data, isLoading, error } = useQuery({
        queryKey: ["agency-desires"],
        queryFn: getDesires,
        refetchInterval: 10000,
    })
    const [fulfilling, setFulfilling] = useState<string | null>(null)

    const handleFulfill = useCallback(async (id: string) => {
        setFulfilling(id)
        try {
            await fulfillDesire(id)
            qc.invalidateQueries({ queryKey: ["agency-desires"] })
        } finally {
            setFulfilling(null)
        }
    }, [qc])

    if (isLoading) return <LoadingSkeleton />
    if (error) return <ErrorCard message="Could not fetch desires" />

    const desires = data?.desires ?? []

    return (
        <Card>
            <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                    <Flame className="h-4 w-4" /> Pending Desires
                    <Badge variant="secondary">{data?.count ?? 0}</Badge>
                </CardTitle>
            </CardHeader>
            <CardContent>
                {desires.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No pending desires.</p>
                ) : (
                    <ScrollArea className="max-h-96">
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>ID</TableHead>
                                    <TableHead>Details</TableHead>
                                    <TableHead className="w-24">Action</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {desires.map((d: any, i: number) => (
                                    <TableRow key={d.id ?? i}>
                                        <TableCell className="font-mono text-xs">{(d.id ?? String(i)).slice(0, 12)}</TableCell>
                                        <TableCell className="text-xs max-w-[300px] truncate">
                                            {JSON.stringify(d).slice(0, 100)}
                                        </TableCell>
                                        <TableCell>
                                            <Button
                                                size="sm"
                                                variant="outline"
                                                className="h-7 text-xs"
                                                disabled={fulfilling === (d.id ?? String(i))}
                                                onClick={() => handleFulfill(d.id ?? String(i))}
                                            >
                                                {fulfilling === (d.id ?? String(i)) ? (
                                                    <Loader2 className="h-3 w-3 animate-spin" />
                                                ) : (
                                                    <CheckCircle2 className="h-3 w-3 mr-1" />
                                                )}
                                                Fulfill
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </ScrollArea>
                )}
            </CardContent>
        </Card>
    )
}

/* ── Counterfactual ───────────────────────────────────────── */
function CounterfactualTab() {
    const [removeId, setRemoveId] = useState("")
    const [removeResult, setRemoveResult] = useState<any>(null)
    const [addEnergy, setAddEnergy] = useState("0.5")
    const [addDepth, setAddDepth] = useState("0.3")
    const [addConnectTo, setAddConnectTo] = useState("")
    const [addResult, setAddResult] = useState<any>(null)
    const [loading, setLoading] = useState(false)

    const simulateRemove = useCallback(async () => {
        if (!removeId.trim()) return
        setLoading(true)
        try {
            const data = await counterfactualRemove(removeId.trim())
            setRemoveResult(data)
        } catch { setRemoveResult({ error: "Failed to simulate" }) }
        finally { setLoading(false) }
    }, [removeId])

    const simulateAdd = useCallback(async () => {
        setLoading(true)
        try {
            const data = await counterfactualAdd({
                energy: parseFloat(addEnergy) || 0.5,
                depth: parseFloat(addDepth) || 0.3,
                connect_to: addConnectTo.split(",").map((s) => s.trim()).filter(Boolean),
            })
            setAddResult(data)
        } catch { setAddResult({ error: "Failed to simulate" }) }
        finally { setLoading(false) }
    }, [addEnergy, addDepth, addConnectTo])

    return (
        <div className="space-y-4">
            {/* Remove simulation */}
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base">Simulate Node Removal</CardTitle>
                    <CardDescription>What happens if a node is removed?</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                    <div className="flex gap-2">
                        <Input value={removeId} onChange={(e) => setRemoveId(e.target.value)} placeholder="Node UUID" className="flex-1" />
                        <Button onClick={simulateRemove} disabled={loading || !removeId.trim()}>Simulate</Button>
                    </div>
                    {removeResult && <ResultJson data={removeResult} />}
                </CardContent>
            </Card>

            {/* Add simulation */}
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base">Simulate Node Addition</CardTitle>
                    <CardDescription>What happens if a new node is added?</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                    <div className="flex flex-wrap gap-3">
                        <div className="space-y-1">
                            <Label className="text-xs">Energy</Label>
                            <Input type="number" step="0.1" value={addEnergy} onChange={(e) => setAddEnergy(e.target.value)} className="w-24 h-8" />
                        </div>
                        <div className="space-y-1">
                            <Label className="text-xs">Depth</Label>
                            <Input type="number" step="0.1" value={addDepth} onChange={(e) => setAddDepth(e.target.value)} className="w-24 h-8" />
                        </div>
                        <div className="space-y-1 flex-1">
                            <Label className="text-xs">Connect to (UUIDs, comma-separated)</Label>
                            <Input value={addConnectTo} onChange={(e) => setAddConnectTo(e.target.value)} placeholder="uuid1, uuid2" className="h-8" />
                        </div>
                    </div>
                    <Button onClick={simulateAdd} disabled={loading}>Simulate Add</Button>
                    {addResult && <ResultJson data={addResult} />}
                </CardContent>
            </Card>
        </div>
    )
}

/* ── Quantum ──────────────────────────────────────────────── */
function QuantumTab() {
    const [mapInput, setMapInput] = useState('[\n  {"embedding": [0.1, 0.2, 0.3], "energy": 0.8}\n]')
    const [mapResult, setMapResult] = useState<any>(null)
    const [fidA, setFidA] = useState('[\n  {"embedding": [0.1, 0.2], "energy": 0.5}\n]')
    const [fidB, setFidB] = useState('[\n  {"embedding": [0.3, 0.4], "energy": 0.7}\n]')
    const [fidResult, setFidResult] = useState<any>(null)
    const [loading, setLoading] = useState(false)

    const runMap = useCallback(async () => {
        setLoading(true)
        try {
            const nodes = JSON.parse(mapInput)
            const data = await quantumMap(nodes)
            setMapResult(data)
        } catch { setMapResult({ error: "Invalid JSON or API error" }) }
        finally { setLoading(false) }
    }, [mapInput])

    const runFidelity = useCallback(async () => {
        setLoading(true)
        try {
            const a = JSON.parse(fidA)
            const b = JSON.parse(fidB)
            const data = await quantumFidelity(a, b)
            setFidResult(data)
        } catch { setFidResult({ error: "Invalid JSON or API error" }) }
        finally { setLoading(false) }
    }, [fidA, fidB])

    return (
        <div className="space-y-4">
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2">
                        <Atom className="h-4 w-4" /> Poincare to Bloch Map
                    </CardTitle>
                    <CardDescription>Map hyperbolic embeddings to quantum Bloch sphere states</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                    <textarea
                        value={mapInput}
                        onChange={(e) => setMapInput(e.target.value)}
                        className="w-full h-24 rounded-md border border-input bg-background px-3 py-2 text-xs font-mono resize-y"
                        spellCheck={false}
                    />
                    <Button onClick={runMap} disabled={loading}>Map to Bloch States</Button>
                    {mapResult && <ResultJson data={mapResult} />}
                </CardContent>
            </Card>

            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2">
                        <Atom className="h-4 w-4" /> Quantum Fidelity
                    </CardTitle>
                    <CardDescription>Compute entanglement proxy between two groups</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div className="space-y-1">
                            <Label className="text-xs">Group A</Label>
                            <textarea
                                value={fidA}
                                onChange={(e) => setFidA(e.target.value)}
                                className="w-full h-20 rounded-md border border-input bg-background px-3 py-2 text-xs font-mono resize-y"
                                spellCheck={false}
                            />
                        </div>
                        <div className="space-y-1">
                            <Label className="text-xs">Group B</Label>
                            <textarea
                                value={fidB}
                                onChange={(e) => setFidB(e.target.value)}
                                className="w-full h-20 rounded-md border border-input bg-background px-3 py-2 text-xs font-mono resize-y"
                                spellCheck={false}
                            />
                        </div>
                    </div>
                    <Button onClick={runFidelity} disabled={loading}>Compute Fidelity</Button>
                    {fidResult && <ResultJson data={fidResult} />}
                </CardContent>
            </Card>
        </div>
    )
}

/* ── Shared Components ────────────────────────────────────── */
function MetricCard({ label, value }: { label: string; value: string | undefined }) {
    return (
        <div className="rounded-lg border bg-muted/30 p-3">
            <div className="text-[11px] text-muted-foreground uppercase tracking-wide">{label}</div>
            <div className="text-lg font-mono font-semibold mt-0.5">{value ?? "—"}</div>
        </div>
    )
}

function ResultJson({ data }: { data: unknown }) {
    return (
        <pre className="text-xs font-mono bg-muted p-3 rounded-lg overflow-auto max-h-64">
            {JSON.stringify(data, null, 2)}
        </pre>
    )
}

function LoadingSkeleton() {
    return (
        <Card>
            <CardContent className="pt-4 space-y-3">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
                <Skeleton className="h-32 w-full" />
            </CardContent>
        </Card>
    )
}

function ErrorCard({ message }: { message: string }) {
    return (
        <Card className="border-destructive/30">
            <CardContent className="pt-4 flex items-center gap-2 text-sm text-muted-foreground">
                <AlertCircle className="h-4 w-4 text-destructive" /> {message}
            </CardContent>
        </Card>
    )
}
