import { useState, useCallback } from "react"
import { useQuery } from "@tanstack/react-query"
import {
    GitBranch, Play, Timer, Hash, Layers, ArrowUpDown,
    Network, Triangle, Users, Waypoints, Target, Loader2,
    BrainCircuit, TreePine, Package, Trash2, FolderOpen,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
    Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
    Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { runAlgorithm, api, DEFAULT_COLLECTION, type AlgoParams } from "@/lib/api"

interface AlgoDef {
    id: string
    name: string
    desc: string
    icon: React.ReactNode
    params: { key: string; label: string; default_: string; type: "number" | "select"; options?: string[] }[]
    resultType: "scores" | "communities" | "count" | "pairs"
}

const ALGORITHMS: AlgoDef[] = [
    {
        id: "pagerank", name: "PageRank", desc: "Node importance by link structure",
        icon: <Target className="h-5 w-5" />,
        params: [
            { key: "damping", label: "Damping", default_: "0.85", type: "number" },
            { key: "iterations", label: "Iterations", default_: "20", type: "number" },
        ],
        resultType: "scores",
    },
    {
        id: "louvain", name: "Louvain", desc: "Community detection by modularity optimization",
        icon: <Users className="h-5 w-5" />,
        params: [
            { key: "iterations", label: "Iterations", default_: "10", type: "number" },
            { key: "resolution", label: "Resolution", default_: "1.0", type: "number" },
        ],
        resultType: "communities",
    },
    {
        id: "labelprop", name: "Label Propagation", desc: "Community detection by label spreading",
        icon: <Layers className="h-5 w-5" />,
        params: [
            { key: "iterations", label: "Iterations", default_: "10", type: "number" },
        ],
        resultType: "communities",
    },
    {
        id: "betweenness", name: "Betweenness", desc: "Bridge nodes between communities",
        icon: <Waypoints className="h-5 w-5" />,
        params: [
            { key: "sample", label: "Sample size", default_: "", type: "number" },
        ],
        resultType: "scores",
    },
    {
        id: "closeness", name: "Closeness", desc: "How close a node is to all others",
        icon: <Target className="h-5 w-5" />,
        params: [],
        resultType: "scores",
    },
    {
        id: "degree", name: "Degree Centrality", desc: "Node connectivity count",
        icon: <ArrowUpDown className="h-5 w-5" />,
        params: [
            { key: "direction", label: "Direction", default_: "both", type: "select", options: ["in", "out", "both"] },
        ],
        resultType: "scores",
    },
    {
        id: "wcc", name: "WCC", desc: "Weakly connected components",
        icon: <Network className="h-5 w-5" />,
        params: [],
        resultType: "communities",
    },
    {
        id: "scc", name: "SCC", desc: "Strongly connected components",
        icon: <Network className="h-5 w-5" />,
        params: [],
        resultType: "communities",
    },
    {
        id: "triangles", name: "Triangle Count", desc: "Total triangles in graph",
        icon: <Triangle className="h-5 w-5" />,
        params: [],
        resultType: "count",
    },
    {
        id: "jaccard", name: "Jaccard Similarity", desc: "Node pair similarity by neighbors",
        icon: <Hash className="h-5 w-5" />,
        params: [
            { key: "top_k", label: "Top K", default_: "100", type: "number" },
            { key: "threshold", label: "Threshold", default_: "0.0", type: "number" },
        ],
        resultType: "pairs",
    },
]

export default function AlgorithmsPage() {
    const [selected, setSelected] = useState<AlgoDef | null>(null)
    const [paramValues, setParamValues] = useState<Record<string, string>>({})
    const [collection, setCollection] = useState(DEFAULT_COLLECTION)
    const [result, setResult] = useState<any>(null)
    const [executing, setExecuting] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // ── Neural section state ──
    const [gnnNodeId, setGnnNodeId] = useState("")
    const [gnnHopDepth, setGnnHopDepth] = useState("2")
    const [gnnRunning, setGnnRunning] = useState(false)
    const [gnnResult, setGnnResult] = useState<any>(null)
    const [gnnError, setGnnError] = useState<string | null>(null)

    const [mctsStartNode, setMctsStartNode] = useState("")
    const [mctsMaxSims, setMctsMaxSims] = useState("100")
    const [mctsRunning, setMctsRunning] = useState(false)
    const [mctsResult, setMctsResult] = useState<any>(null)
    const [mctsError, setMctsError] = useState<string | null>(null)

    const [modelPath, setModelPath] = useState("")
    const [modelLoading, setModelLoading] = useState(false)
    const [modelError, setModelError] = useState<string | null>(null)

    const { data: collections } = useQuery({
        queryKey: ["collections"],
        queryFn: () => api.get("/collections").then((r) => r.data),
    })

    const selectAlgo = useCallback((algo: AlgoDef) => {
        setSelected(algo)
        setResult(null)
        setError(null)
        const defaults: Record<string, string> = {}
        algo.params.forEach((p) => { defaults[p.key] = p.default_ })
        setParamValues(defaults)
    }, [])

    const execute = useCallback(async () => {
        if (!selected) return
        setExecuting(true)
        setError(null)
        setResult(null)
        try {
            const params: AlgoParams = { collection: collection || undefined }
            selected.params.forEach((p) => {
                const v = paramValues[p.key]
                if (v !== undefined && v !== "") {
                    if (p.type === "number") (params as any)[p.key] = parseFloat(v)
                    else (params as any)[p.key] = v
                }
            })
            const data = await runAlgorithm(selected.id, params)
            setResult(data)
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : "Algorithm execution failed")
        } finally {
            setExecuting(false)
        }
    }, [selected, paramValues, collection])

    // ── Neural handlers ──
    const runGnn = useCallback(async () => {
        if (!gnnNodeId.trim()) return
        setGnnRunning(true)
        setGnnError(null)
        setGnnResult(null)
        try {
            const depth = Math.min(3, Math.max(1, parseInt(gnnHopDepth) || 2))
            const res = await api.post("/query", {
                nql: `GNN INFER NODE "${gnnNodeId.trim()}" HOPS ${depth}`,
                collection: collection || DEFAULT_COLLECTION,
            })
            setGnnResult(res.data)
        } catch (e: unknown) {
            setGnnError(e instanceof Error ? e.message : "GNN inference failed")
        } finally {
            setGnnRunning(false)
        }
    }, [gnnNodeId, gnnHopDepth, collection])

    const runMcts = useCallback(async () => {
        if (!mctsStartNode.trim()) return
        setMctsRunning(true)
        setMctsError(null)
        setMctsResult(null)
        try {
            const sims = Math.min(1000, Math.max(10, parseInt(mctsMaxSims) || 100))
            const res = await api.post("/query", {
                nql: `MCTS SEARCH FROM "${mctsStartNode.trim()}" SIMULATIONS ${sims}`,
                collection: collection || DEFAULT_COLLECTION,
            })
            setMctsResult(res.data)
        } catch (e: unknown) {
            setMctsError(e instanceof Error ? e.message : "MCTS search failed")
        } finally {
            setMctsRunning(false)
        }
    }, [mctsStartNode, mctsMaxSims, collection])

    const { data: loadedModels, refetch: refetchModels } = useQuery({
        queryKey: ["neural-models", collection],
        queryFn: () => api.post("/query", {
            nql: "LIST MODELS",
            collection: collection || DEFAULT_COLLECTION,
        }).then((r) => r.data),
        refetchInterval: false,
        retry: false,
    })

    const loadModel = useCallback(async () => {
        if (!modelPath.trim()) return
        setModelLoading(true)
        setModelError(null)
        try {
            await api.post("/query", {
                nql: `LOAD MODEL "${modelPath.trim()}"`,
                collection: collection || DEFAULT_COLLECTION,
            })
            setModelPath("")
            refetchModels()
        } catch (e: unknown) {
            setModelError(e instanceof Error ? e.message : "Failed to load model")
        } finally {
            setModelLoading(false)
        }
    }, [modelPath, collection, refetchModels])

    const unloadModel = useCallback(async (modelId: string) => {
        try {
            await api.post("/query", {
                nql: `UNLOAD MODEL "${modelId}"`,
                collection: collection || DEFAULT_COLLECTION,
            })
            refetchModels()
        } catch (e: unknown) {
            setModelError(e instanceof Error ? e.message : "Failed to unload model")
        }
    }, [collection, refetchModels])

    return (
        <div className="space-y-6 fade-in">
            <div>
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    <GitBranch className="h-6 w-6" /> Graph Algorithms
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                    11 algorithms powered by nietzsche-algo (10 REST + A* via gRPC)
                </p>
            </div>

            {/* Algorithm Grid */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                {ALGORITHMS.map((algo) => (
                    <Card
                        key={algo.id}
                        className={`cursor-pointer transition-all hover:border-primary/50 ${
                            selected?.id === algo.id ? "border-primary ring-1 ring-primary/30" : ""
                        }`}
                        onClick={() => selectAlgo(algo)}
                    >
                        <CardContent className="pt-4 pb-3 px-3">
                            <div className="flex items-center gap-2 mb-1">
                                {algo.icon}
                                <span className="font-medium text-sm">{algo.name}</span>
                            </div>
                            <p className="text-[11px] text-muted-foreground leading-tight">{algo.desc}</p>
                        </CardContent>
                    </Card>
                ))}
            </div>

            {/* Execution Panel */}
            {selected && (
                <Card>
                    <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                            <div>
                                <CardTitle className="text-base flex items-center gap-2">
                                    {selected.icon} {selected.name}
                                </CardTitle>
                                <CardDescription>{selected.desc}</CardDescription>
                            </div>
                            <Button onClick={execute} disabled={executing}>
                                {executing ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Play className="h-4 w-4 mr-1" />}
                                {executing ? "Running..." : "Execute"}
                            </Button>
                        </div>
                    </CardHeader>
                    <CardContent>
                        <div className="flex flex-wrap gap-4 items-end">
                            {/* Collection selector */}
                            <div className="space-y-1">
                                <Label className="text-xs">Collection</Label>
                                <Select value={collection} onValueChange={setCollection}>
                                    <SelectTrigger className="w-40 h-8">
                                        <SelectValue placeholder="default" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="">default</SelectItem>
                                        {(collections ?? []).map((c: any) => (
                                            <SelectItem key={c.name} value={c.name}>{c.name}</SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>
                            {/* Dynamic params */}
                            {selected.params.map((p) => (
                                <div key={p.key} className="space-y-1">
                                    <Label className="text-xs">{p.label}</Label>
                                    {p.type === "select" ? (
                                        <Select value={paramValues[p.key] ?? p.default_} onValueChange={(v) => setParamValues((prev) => ({ ...prev, [p.key]: v }))}>
                                            <SelectTrigger className="w-28 h-8">
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                {p.options!.map((o) => (
                                                    <SelectItem key={o} value={o}>{o}</SelectItem>
                                                ))}
                                            </SelectContent>
                                        </Select>
                                    ) : (
                                        <Input
                                            type="number"
                                            step="any"
                                            value={paramValues[p.key] ?? ""}
                                            onChange={(e) => setParamValues((prev) => ({ ...prev, [p.key]: e.target.value }))}
                                            className="w-28 h-8"
                                            placeholder={p.default_}
                                        />
                                    )}
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Error */}
            {error && (
                <Card className="border-destructive/50">
                    <CardContent className="pt-4">
                        <p className="text-sm text-destructive font-mono">{error}</p>
                    </CardContent>
                </Card>
            )}

            {/* Loading */}
            {executing && (
                <Card>
                    <CardContent className="pt-4 space-y-2">
                        <Skeleton className="h-4 w-full" />
                        <Skeleton className="h-4 w-3/4" />
                    </CardContent>
                </Card>
            )}

            {/* Results */}
            {result && (
                <Card>
                    <CardHeader className="pb-3">
                        <div className="flex items-center gap-3 flex-wrap">
                            <CardTitle className="text-base">{result.algorithm ?? selected?.name}</CardTitle>
                            {result.duration_ms !== undefined && (
                                <Badge variant="outline"><Timer className="h-3 w-3 mr-1" />{result.duration_ms}ms</Badge>
                            )}
                            {result.converged !== undefined && (
                                <Badge variant={result.converged ? "default" : "secondary"}>
                                    {result.converged ? "Converged" : "Not converged"}
                                </Badge>
                            )}
                            {result.community_count !== undefined && (
                                <Badge variant="secondary">{result.community_count} communities</Badge>
                            )}
                            {result.largest_component_size !== undefined && (
                                <Badge variant="outline">Largest: {result.largest_component_size}</Badge>
                            )}
                            {result.modularity !== undefined && (
                                <Badge variant="outline">Modularity: {result.modularity.toFixed(4)}</Badge>
                            )}
                        </div>
                    </CardHeader>
                    <CardContent>
                        {/* Count result */}
                        {result.count !== undefined && (
                            <div className="text-center py-8">
                                <p className="text-4xl font-bold font-mono">{result.count.toLocaleString()}</p>
                                <p className="text-sm text-muted-foreground mt-1">triangles found</p>
                            </div>
                        )}

                        {/* Scores table */}
                        {result.scores && (
                            <ScrollArea className="max-h-96">
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead>#</TableHead>
                                            <TableHead>Node ID</TableHead>
                                            <TableHead>Score</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {result.scores.slice(0, 100).map((s: any, i: number) => (
                                            <TableRow key={s.node_id}>
                                                <TableCell className="text-muted-foreground">{i + 1}</TableCell>
                                                <TableCell className="font-mono text-xs">{s.node_id}</TableCell>
                                                <TableCell className="font-mono">{s.score.toFixed(6)}</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </ScrollArea>
                        )}

                        {/* Communities table */}
                        {(result.communities || result.labels || result.components) && (
                            <ScrollArea className="max-h-96">
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead>Node ID</TableHead>
                                            <TableHead>Community / Component</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {(result.communities || result.labels || result.components || []).slice(0, 100).map((c: any, i: number) => (
                                            <TableRow key={c.node_id ?? i}>
                                                <TableCell className="font-mono text-xs">{c.node_id}</TableCell>
                                                <TableCell>
                                                    <Badge variant="outline">{c.community_id ?? c.component_id ?? "—"}</Badge>
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </ScrollArea>
                        )}

                        {/* Pairs table */}
                        {result.pairs && (
                            <ScrollArea className="max-h-96">
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead>Node A</TableHead>
                                            <TableHead>Node B</TableHead>
                                            <TableHead>Score</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {result.pairs.slice(0, 100).map((p: any, i: number) => (
                                            <TableRow key={i}>
                                                <TableCell className="font-mono text-xs">{p.node_a}</TableCell>
                                                <TableCell className="font-mono text-xs">{p.node_b}</TableCell>
                                                <TableCell className="font-mono">{p.score.toFixed(6)}</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </ScrollArea>
                        )}
                    </CardContent>
                </Card>
            )}

            {/* ═══════════════════════════════════════════════════════════ */}
            {/*  NEURAL SECTION                                           */}
            {/* ═══════════════════════════════════════════════════════════ */}
            <div className="border-t border-border my-4" />
            <div>
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    <BrainCircuit className="h-6 w-6" /> Neural
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                    GNN inference, MCTS search, and model management
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* ── GNN Inference ── */}
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-base flex items-center gap-2">
                            <BrainCircuit className="h-5 w-5" /> GNN Inference
                        </CardTitle>
                        <CardDescription>
                            Run graph neural network inference on a local subgraph
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="flex flex-wrap gap-4 items-end">
                            <div className="space-y-1">
                                <Label className="text-xs">Node ID (subgraph center)</Label>
                                <Input
                                    value={gnnNodeId}
                                    onChange={(e) => setGnnNodeId(e.target.value)}
                                    placeholder="node-uuid"
                                    className="w-56 h-8 font-mono text-xs"
                                />
                            </div>
                            <div className="space-y-1">
                                <Label className="text-xs">Hop Depth</Label>
                                <Select value={gnnHopDepth} onValueChange={setGnnHopDepth}>
                                    <SelectTrigger className="w-20 h-8">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="1">1</SelectItem>
                                        <SelectItem value="2">2</SelectItem>
                                        <SelectItem value="3">3</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                            <Button onClick={runGnn} disabled={gnnRunning || !gnnNodeId.trim()}>
                                {gnnRunning ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Play className="h-4 w-4 mr-1" />}
                                {gnnRunning ? "Running..." : "RUN GNN"}
                            </Button>
                        </div>

                        {gnnError && (
                            <p className="text-sm text-destructive font-mono">{gnnError}</p>
                        )}

                        {gnnResult && (
                            <div className="space-y-2">
                                <div className="flex items-center gap-2 flex-wrap">
                                    <Badge variant="outline" className="font-mono">
                                        <Timer className="h-3 w-3 mr-1" />
                                        {gnnResult.duration_ms ?? "?"}ms
                                    </Badge>
                                    {gnnResult.model_name && (
                                        <Badge variant="secondary">{gnnResult.model_name}</Badge>
                                    )}
                                    {gnnResult.subgraph_nodes !== undefined && (
                                        <Badge variant="outline">{gnnResult.subgraph_nodes} nodes in subgraph</Badge>
                                    )}
                                </div>
                                {/* Prediction scores table */}
                                {(gnnResult.predictions || gnnResult.scores) && (
                                    <ScrollArea className="max-h-64">
                                        <Table>
                                            <TableHeader>
                                                <TableRow>
                                                    <TableHead>#</TableHead>
                                                    <TableHead>Node ID</TableHead>
                                                    <TableHead>Score</TableHead>
                                                    <TableHead>Label</TableHead>
                                                </TableRow>
                                            </TableHeader>
                                            <TableBody>
                                                {(gnnResult.predictions || gnnResult.scores || []).slice(0, 100).map((p: any, i: number) => (
                                                    <TableRow key={p.node_id ?? i}>
                                                        <TableCell className="text-muted-foreground">{i + 1}</TableCell>
                                                        <TableCell className="font-mono text-xs">{p.node_id}</TableCell>
                                                        <TableCell className="font-mono">{typeof p.score === "number" ? p.score.toFixed(6) : p.score ?? "—"}</TableCell>
                                                        <TableCell className="text-xs">{p.label ?? p.predicted_class ?? "—"}</TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </ScrollArea>
                                )}
                                {/* Raw JSON fallback */}
                                {!gnnResult.predictions && !gnnResult.scores && (
                                    <ScrollArea className="max-h-64">
                                        <pre className="text-xs font-mono whitespace-pre-wrap p-3 rounded bg-muted/50">
                                            {JSON.stringify(gnnResult, null, 2)}
                                        </pre>
                                    </ScrollArea>
                                )}
                            </div>
                        )}
                    </CardContent>
                </Card>

                {/* ── MCTS Search ── */}
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-base flex items-center gap-2">
                            <TreePine className="h-5 w-5" /> MCTS Search
                        </CardTitle>
                        <CardDescription>
                            Monte Carlo Tree Search for optimal graph traversal
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="flex flex-wrap gap-4 items-end">
                            <div className="space-y-1">
                                <Label className="text-xs">Start Node ID</Label>
                                <Input
                                    value={mctsStartNode}
                                    onChange={(e) => setMctsStartNode(e.target.value)}
                                    placeholder="node-uuid"
                                    className="w-56 h-8 font-mono text-xs"
                                />
                            </div>
                            <div className="space-y-1">
                                <Label className="text-xs">Max Simulations</Label>
                                <Input
                                    type="number"
                                    min={10}
                                    max={1000}
                                    value={mctsMaxSims}
                                    onChange={(e) => setMctsMaxSims(e.target.value)}
                                    className="w-28 h-8"
                                    placeholder="100"
                                />
                            </div>
                            <Button onClick={runMcts} disabled={mctsRunning || !mctsStartNode.trim()}>
                                {mctsRunning ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Play className="h-4 w-4 mr-1" />}
                                {mctsRunning ? "Running..." : "RUN MCTS"}
                            </Button>
                        </div>

                        {mctsError && (
                            <p className="text-sm text-destructive font-mono">{mctsError}</p>
                        )}

                        {mctsResult && (
                            <div className="space-y-2">
                                <div className="flex items-center gap-2 flex-wrap">
                                    <Badge variant="outline" className="font-mono">
                                        <Timer className="h-3 w-3 mr-1" />
                                        {mctsResult.duration_ms ?? "?"}ms
                                    </Badge>
                                    {mctsResult.total_simulations !== undefined && (
                                        <Badge variant="secondary">{mctsResult.total_simulations} simulations</Badge>
                                    )}
                                    {mctsResult.best_value !== undefined && (
                                        <Badge variant="outline">Best value: {typeof mctsResult.best_value === "number" ? mctsResult.best_value.toFixed(4) : mctsResult.best_value}</Badge>
                                    )}
                                </div>

                                {/* Best path */}
                                {mctsResult.best_path && (
                                    <div>
                                        <p className="text-xs font-medium text-muted-foreground mb-1">Best Path</p>
                                        <div className="flex flex-wrap items-center gap-1">
                                            {(Array.isArray(mctsResult.best_path) ? mctsResult.best_path : []).map((nodeId: string, i: number) => (
                                                <span key={i} className="inline-flex items-center">
                                                    <Badge variant="outline" className="font-mono text-xs">{nodeId}</Badge>
                                                    {i < mctsResult.best_path.length - 1 && (
                                                        <span className="text-muted-foreground mx-0.5">&rarr;</span>
                                                    )}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Visit counts / tree summary */}
                                {(mctsResult.visit_counts || mctsResult.tree_summary) && (
                                    <ScrollArea className="max-h-64">
                                        <Table>
                                            <TableHeader>
                                                <TableRow>
                                                    <TableHead>Node ID</TableHead>
                                                    <TableHead>Visits</TableHead>
                                                    <TableHead>Avg Value</TableHead>
                                                </TableRow>
                                            </TableHeader>
                                            <TableBody>
                                                {(mctsResult.visit_counts || mctsResult.tree_summary || []).slice(0, 100).map((v: any, i: number) => (
                                                    <TableRow key={v.node_id ?? i}>
                                                        <TableCell className="font-mono text-xs">{v.node_id}</TableCell>
                                                        <TableCell className="font-mono">{v.visits ?? v.visit_count ?? "—"}</TableCell>
                                                        <TableCell className="font-mono">{typeof v.avg_value === "number" ? v.avg_value.toFixed(4) : v.avg_value ?? "—"}</TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </ScrollArea>
                                )}

                                {/* Raw JSON fallback */}
                                {!mctsResult.best_path && !mctsResult.visit_counts && !mctsResult.tree_summary && (
                                    <ScrollArea className="max-h-64">
                                        <pre className="text-xs font-mono whitespace-pre-wrap p-3 rounded bg-muted/50">
                                            {JSON.stringify(mctsResult, null, 2)}
                                        </pre>
                                    </ScrollArea>
                                )}
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>

            {/* ── Model Management ── */}
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2">
                        <Package className="h-5 w-5" /> Model Management
                    </CardTitle>
                    <CardDescription>
                        Load, list, and unload neural models
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    {/* Load model */}
                    <div className="flex flex-wrap gap-3 items-end">
                        <div className="space-y-1 flex-1 min-w-[250px]">
                            <Label className="text-xs">Model Path</Label>
                            <Input
                                value={modelPath}
                                onChange={(e) => setModelPath(e.target.value)}
                                placeholder="/path/to/model.onnx"
                                className="h-8 font-mono text-xs"
                            />
                        </div>
                        <Button onClick={loadModel} disabled={modelLoading || !modelPath.trim()}>
                            {modelLoading ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <FolderOpen className="h-4 w-4 mr-1" />}
                            {modelLoading ? "Loading..." : "Load Model"}
                        </Button>
                    </div>

                    {modelError && (
                        <p className="text-sm text-destructive font-mono">{modelError}</p>
                    )}

                    {/* Loaded models list */}
                    {loadedModels?.models && Array.isArray(loadedModels.models) && loadedModels.models.length > 0 ? (
                        <ScrollArea className="max-h-64">
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead>Model ID</TableHead>
                                        <TableHead>Name</TableHead>
                                        <TableHead>Type</TableHead>
                                        <TableHead>Status</TableHead>
                                        <TableHead className="w-20"></TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {loadedModels.models.map((m: any) => (
                                        <TableRow key={m.id ?? m.name}>
                                            <TableCell className="font-mono text-xs">{m.id ?? "—"}</TableCell>
                                            <TableCell className="text-sm">{m.name ?? m.path ?? "—"}</TableCell>
                                            <TableCell>
                                                <Badge variant="outline">{m.model_type ?? m.type ?? "unknown"}</Badge>
                                            </TableCell>
                                            <TableCell>
                                                <Badge variant={m.status === "loaded" ? "default" : "secondary"}>
                                                    {m.status ?? "loaded"}
                                                </Badge>
                                            </TableCell>
                                            <TableCell>
                                                <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    className="h-7 text-destructive hover:text-destructive"
                                                    onClick={() => unloadModel(m.id ?? m.name)}
                                                >
                                                    <Trash2 className="h-3.5 w-3.5" />
                                                </Button>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </ScrollArea>
                    ) : (
                        <div className="text-center py-6 text-sm text-muted-foreground">
                            No models loaded. Use the input above to load a model.
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    )
}
