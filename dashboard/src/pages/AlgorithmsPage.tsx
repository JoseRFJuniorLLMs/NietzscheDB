import { useState, useCallback } from "react"
import { useQuery } from "@tanstack/react-query"
import {
    GitBranch, Play, Timer, Hash, Layers, ArrowUpDown,
    Network, Triangle, Users, Waypoints, Target, Loader2,
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
        </div>
    )
}
