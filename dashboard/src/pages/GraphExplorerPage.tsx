import { useState, useMemo, useCallback } from "react"
import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"
import {
    ChevronRight, RefreshCw, AlertCircle,
    Network, GitBranch, Download, Palette,
    BarChart3, Cpu, Route, Users, Zap,
} from "lucide-react"
import { PerspektiveView } from "@/components/PerspektiveView"
import type { StreamingMode } from "@/components/PerspektiveView"
import { BarChart, Bar, ResponsiveContainer } from "recharts"

// ── Types ─────────────────────────────────────────────────────────────────────

interface NodeRecord {
    id:         string
    node_type:  string
    energy:     number
    depth:      number
    hausdorff:  number
    created_at: number
    content:    Record<string, unknown>
    causal_chain?: string
    is_archived?: boolean
    [key: string]: unknown
}

interface EdgeRecord {
    id:        string
    from:      string
    to:        string
    edge_type: string
    weight:    number
    [key: string]: unknown
}

interface GraphResponse {
    nodes: NodeRecord[]
    edges: EdgeRecord[]
}

interface AlgoResult {
    name: string
    data: Map<string, number> | Array<{ id: string; score: number }>
    running: boolean
}

// ── Constants ──────────────────────────────────────────────────────────────────

const NODE_TYPE_COLORS: Record<string, string> = {
    Semantic:      "#00ff66",
    Episodic:      "#00f0ff",
    Concept:       "#f59e0b",
    DreamSnapshot: "#8b5cf6",
    Somatic:       "#22c55e",
    Linguistic:    "#f43f5e",
    Composite:     "#a78bfa",
}

const DEFAULT_COLLECTION = "eva_core"

const LAYOUT_OPTIONS = [
    { value: "default", label: "DEFAULT (Embedding)", icon: Network },
    { value: "force", label: "FORCE-DIRECTED", icon: Zap },
    { value: "radial", label: "RADIAL", icon: GitBranch },
    { value: "tree", label: "TREE", icon: GitBranch },
    { value: "grid", label: "GRID", icon: BarChart3 },
    { value: "concentric", label: "CONCENTRIC", icon: Network },
]

const ALGO_OPTIONS = [
    { value: "pagerank", label: "PAGERANK", icon: BarChart3, desc: "Importancia recursiva" },
    { value: "degree", label: "DEGREE", icon: Network, desc: "Centralidade por grau" },
    { value: "betweenness", label: "BETWEENNESS", icon: Route, desc: "Pontes entre clusters" },
    { value: "closeness", label: "CLOSENESS", icon: Zap, desc: "Proximidade media" },
    { value: "eigenvector", label: "EIGENVECTOR", icon: Cpu, desc: "Influencia de vizinhos" },
    { value: "louvain", label: "LOUVAIN", icon: Users, desc: "Deteccao de comunidades" },
    { value: "label-prop", label: "LABEL PROP", icon: Users, desc: "Propagacao de rotulos" },
]

const THEME_OPTIONS = [
    { value: "cyberpunk", label: "CYBERPUNK", color: "#00f0ff" },
    { value: "midnight", label: "MIDNIGHT", color: "#a0b4d0" },
    { value: "paper", label: "PAPER", color: "#2d2d2d" },
    { value: "matrix", label: "MATRIX", color: "#00ff00" },
    { value: "aurora", label: "AURORA", color: "#50c8ff" },
]

// ── GraphExplorerPage ─────────────────────────────────────────────────────────

export function GraphExplorerPage() {
    const collection = DEFAULT_COLLECTION
    const [limit, setLimit]                     = useState(2000)
    const [panelOpen, setPanelOpen]             = useState(true)
    const [panelTab, setPanelTab]               = useState<"ANALYTICS" | "ALGORITHMS" | "LAYOUT" | "EXPORT">("ANALYTICS")
    const [streamingMode, setStreamingMode]     = useState<StreamingMode>("sse")
    const [selectedLayout, setSelectedLayout]   = useState("default")
    const [selectedTheme, setSelectedTheme]     = useState("cyberpunk")
    const [algoResults, setAlgoResults]         = useState<AlgoResult[]>([])
    const [runningAlgo, setRunningAlgo]         = useState<string | null>(null)

    // ── Graph data fetch (for analytics sidebar) ──────────────────────────────
    const { data: graphData, isLoading, error, refetch } = useQuery<GraphResponse>({
        queryKey: ["graph", collection, limit],
        queryFn: () => api.get(`/graph?collection=${collection}&limit=${limit}`).then(r => r.data),
        staleTime: 30_000,
    })

    // ── Analytics data ────────────────────────────────────────────────────────
    const energyHistData = useMemo(() => {
        if (!graphData?.nodes) return []
        const buckets = Array.from({ length: 10 }, (_, i) => ({
            range: `${(i / 10).toFixed(1)}`,
            count: 0,
        }))
        for (const n of graphData.nodes) {
            const idx = Math.min(Math.floor(n.energy * 10), 9)
            buckets[idx].count++
        }
        return buckets
    }, [graphData])

    const depthHistData = useMemo(() => {
        if (!graphData?.nodes) return []
        const buckets = Array.from({ length: 10 }, (_, i) => ({
            range: `${(i / 10).toFixed(1)}`,
            count: 0,
        }))
        for (const n of graphData.nodes) {
            const idx = Math.min(Math.floor((n.depth ?? 0) * 10), 9)
            buckets[idx].count++
        }
        return buckets
    }, [graphData])

    const edgeTypeChartData = useMemo(() => {
        if (!graphData?.edges) return []
        const counts = new Map<string, number>()
        for (const e of graphData.edges) {
            counts.set(e.edge_type, (counts.get(e.edge_type) ?? 0) + 1)
        }
        return Array.from(counts, ([name, count]) => ({ name, count }))
            .sort((a, b) => b.count - a.count)
    }, [graphData])

    const nodeTypeStats = useMemo(() => {
        if (!graphData?.nodes) return []
        const counts = new Map<string, number>()
        for (const n of graphData.nodes) {
            counts.set(n.node_type, (counts.get(n.node_type) ?? 0) + 1)
        }
        return Array.from(counts, ([type, count]) => ({ type, count }))
            .sort((a, b) => b.count - a.count)
    }, [graphData])

    const graphStats = useMemo(() => {
        if (!graphData) return null
        const nodes = graphData.nodes
        const avgEnergy = nodes.reduce((s, n) => s + n.energy, 0) / (nodes.length || 1)
        const avgDepth = nodes.reduce((s, n) => s + (n.depth ?? 0), 0) / (nodes.length || 1)
        const ubermensch = nodes.filter(n => n.energy > 0.8).length
        const archived = nodes.filter(n => n.is_archived).length
        return { avgEnergy, avgDepth, ubermensch, archived }
    }, [graphData])

    // ── Run server-side algorithm ─────────────────────────────────────────────
    const runAlgorithm = useCallback(async (algoName: string) => {
        setRunningAlgo(algoName)
        try {
            const result = await api.get(`/algo/${algoName}`, {
                params: { collection, top_k: 20 },
            })
            setAlgoResults(prev => {
                const filtered = prev.filter(r => r.name !== algoName)
                return [...filtered, {
                    name: algoName,
                    data: result.data?.results || result.data,
                    running: false,
                }]
            })
        } catch {
            // Algorithm might not be available server-side
            setAlgoResults(prev => {
                const filtered = prev.filter(r => r.name !== algoName)
                return [...filtered, {
                    name: algoName,
                    data: [],
                    running: false,
                }]
            })
        } finally {
            setRunningAlgo(null)
        }
    }, [collection])

    // ── Export data ──────────────────────────────────────────────────────────
    const handleExportJSONL = useCallback(async () => {
        try {
            const blob = await api.get(`/export/nodes`, {
                params: { format: "jsonl", collection },
                responseType: "blob",
            }).then(r => r.data)
            const url = URL.createObjectURL(blob)
            const a = document.createElement("a")
            a.href = url
            a.download = `nietzsche-${collection}-nodes-${Date.now()}.jsonl`
            a.click()
            URL.revokeObjectURL(url)
        } catch { /* ignore */ }
    }, [collection])

    const handleExportCSV = useCallback(async () => {
        try {
            const blob = await api.get(`/export/nodes`, {
                params: { format: "csv", collection },
                responseType: "blob",
            }).then(r => r.data)
            const url = URL.createObjectURL(blob)
            const a = document.createElement("a")
            a.href = url
            a.download = `nietzsche-${collection}-nodes-${Date.now()}.csv`
            a.click()
            URL.revokeObjectURL(url)
        } catch { /* ignore */ }
    }, [collection])

    // ── Render ───────────────────────────────────────────────────────────────
    return (
        <div className="flex flex-col h-screen relative bg-[#020617] overflow-hidden">

            {/* ── Cockpit HUD Overlay (Top Left) ────────────────────────────────── */}
            <div className="absolute top-6 left-6 z-30 pointer-events-none">
                <div className="flex flex-col gap-1">
                    <h2 className="text-[#00f0ff] text-xl font-mono font-bold tracking-tighter" style={{ textShadow: "0 0 10px rgba(0, 240, 255, 0.5)" }}>
                        NIETZSCHEDB // PERSPEKTIVE ENGINE
                    </h2>
                    <div className="flex items-center gap-2 pointer-events-auto mt-2">
                        <button
                            onClick={() => refetch()}
                            disabled={isLoading}
                            className="h-9 w-9 flex items-center justify-center bg-black/60 border border-border/20 rounded-md text-muted-foreground hover:text-[#00f0ff] hover:border-[#00f0ff]/50 transition-all backdrop-blur-md"
                            title="Refetch Analytics"
                        >
                            <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
                        </button>
                        <button
                            onClick={() => setPanelOpen(v => !v)}
                            className={`h-9 w-9 flex items-center justify-center bg-black/60 border rounded-md transition-all backdrop-blur-md ${panelOpen ? "border-[#ff00ff]/50 text-[#ff00ff]" : "border-border/20 text-muted-foreground hover:text-foreground"}`}
                            title="Toggle Analysis Panel"
                        >
                            <ChevronRight className={`h-4 w-4 transition-transform ${panelOpen ? "rotate-180" : ""}`} />
                        </button>
                    </div>

                    <div className="flex items-center gap-2 mt-2 font-mono text-[10px] text-muted-foreground pointer-events-none">
                        <span className="flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-[#00ff66] animate-pulse" />
                            {streamingMode === "ws" ? "WS STREAMING" : "SSE STREAMING"}
                        </span>
                        <span>|</span>
                        <span>NODES: <span className="text-white font-bold">{graphData?.nodes.length ?? 0}</span></span>
                        <span>|</span>
                        <span>EDGES: <span className="text-white font-bold">{graphData?.edges.length ?? 0}</span></span>
                    </div>
                </div>
            </div>

            {/* ── Subtitle / Collection Info (Top Right) ────────────────────────── */}
            <div className="absolute top-6 right-6 z-20 text-right pointer-events-none">
                <div className="text-[10px] font-mono tracking-widest text-[#94a3b8] uppercase">
                    Cortex: <span className="text-[#00f0ff]">{collection}</span>
                </div>
                <div className="text-[9px] font-mono text-muted-foreground/60 mt-1">
                    LAYOUT: {selectedLayout.toUpperCase()} | THEME: {selectedTheme.toUpperCase()}
                </div>
            </div>

            {/* ── Main area ────────────────────────────────────────────────── */}
            <div className="flex flex-1 overflow-hidden relative">

                {/* ── Left analysis panel (Floating, translucent) ──────────────── */}
                {panelOpen && (
                    <div className="absolute top-24 left-6 bottom-24 w-80 flex flex-col z-20 bg-black/70 border border-border/20 rounded-lg backdrop-blur-xl overflow-hidden shadow-2xl">
                        {/* Tab switcher */}
                        <div className="flex border-b border-border/20 px-1 shrink-0 bg-white/5 overflow-x-auto">
                            {(["ANALYTICS", "ALGORITHMS", "LAYOUT", "EXPORT"] as const).map(tab => (
                                <button
                                    key={tab}
                                    onClick={() => setPanelTab(tab)}
                                    className={`px-2.5 py-2 text-[9px] font-bold tracking-widest transition-colors whitespace-nowrap ${panelTab === tab ? "border-b border-[#00f0ff] text-[#00f0ff]" : "text-muted-foreground hover:text-foreground"}`}
                                >
                                    {tab}
                                </button>
                            ))}
                        </div>

                        <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-5">
                            {/* ═══ ANALYTICS TAB ═══ */}
                            {panelTab === "ANALYTICS" && (
                                <>
                                    {/* Graph Stats Summary */}
                                    {graphStats && (
                                        <PanelSection label="graph summary">
                                            <div className="grid grid-cols-2 gap-2 mt-2">
                                                <StatCard label="AVG ENERGY" value={`${(graphStats.avgEnergy * 100).toFixed(1)}%`} color="#ff00ff" />
                                                <StatCard label="AVG DEPTH" value={graphStats.avgDepth.toFixed(3)} color="#00f0ff" />
                                                <StatCard label="UBERMENSCH" value={`${graphStats.ubermensch}`} color="#ffd700" />
                                                <StatCard label="ARCHIVED" value={`${graphStats.archived}`} color="#a5f3fc" />
                                            </div>
                                        </PanelSection>
                                    )}

                                    {/* Node Type Distribution */}
                                    <PanelSection label="distribuicao de tipos">
                                        <div className="space-y-1.5 mt-2">
                                            {nodeTypeStats.map(({ type, count }) => (
                                                <div key={type} className="flex items-center gap-2 text-[10px] px-2 py-1.5 rounded text-muted-foreground">
                                                    <div className="w-2 h-2 rounded-full" style={{ background: NODE_TYPE_COLORS[type] ?? "#00ff66" }} />
                                                    <span className="font-mono flex-1">{type.toUpperCase()}</span>
                                                    <span className="text-[#00d8ff] font-mono">{count}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </PanelSection>

                                    {/* Energy Histogram */}
                                    <PanelSection label="energia (ubermensch)">
                                        <ResponsiveContainer width="100%" height={60}>
                                            <BarChart data={energyHistData}>
                                                <Bar dataKey="count" fill="#ff00ff" radius={[2, 2, 0, 0]} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </PanelSection>

                                    {/* Depth Histogram */}
                                    <PanelSection label="profundidade (poincare)">
                                        <ResponsiveContainer width="100%" height={60}>
                                            <BarChart data={depthHistData}>
                                                <Bar dataKey="count" fill="#00f0ff" radius={[2, 2, 0, 0]} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </PanelSection>

                                    {/* Edge Types */}
                                    <PanelSection label="tipos de conexoes">
                                        <div className="space-y-1.5 mt-2">
                                            {edgeTypeChartData.map(edge => (
                                                <div key={edge.name} className="flex flex-col gap-1">
                                                    <div className="flex justify-between text-[9px] font-mono">
                                                        <span className="text-muted-foreground">{edge.name}</span>
                                                        <span className="text-[#00d8ff]">{edge.count}</span>
                                                    </div>
                                                    <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                                                        <div
                                                            className="h-full bg-[#00d8ff]/50"
                                                            style={{ width: `${(edge.count / Math.max(...edgeTypeChartData.map(e => e.count))) * 100}%` }}
                                                        />
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </PanelSection>

                                    {/* Config */}
                                    <div className="pt-4 border-t border-border/20">
                                        <div className="text-[9px] font-mono text-muted-foreground mb-2 uppercase">Configuracao de Carga</div>
                                        <div className="flex gap-2">
                                            <select
                                                value={limit}
                                                onChange={e => setLimit(Number(e.target.value))}
                                                className="flex-1 bg-black/40 border border-border/20 rounded px-2 py-1 text-[10px] font-mono text-foreground focus:outline-none focus:border-[#00f0ff]"
                                            >
                                                {[500, 1000, 2000, 5000, 10000].map(n => <option key={n} value={n}>{n} NODES</option>)}
                                            </select>
                                            <select
                                                value={streamingMode}
                                                onChange={e => setStreamingMode(e.target.value as StreamingMode)}
                                                className="bg-black/40 border border-border/20 rounded px-2 py-1 text-[10px] font-mono text-foreground focus:outline-none focus:border-[#00f0ff]"
                                            >
                                                <option value="sse">SSE</option>
                                                <option value="ws">WebSocket</option>
                                                <option value="none">Polling</option>
                                            </select>
                                        </div>
                                    </div>
                                </>
                            )}

                            {/* ═══ ALGORITHMS TAB ═══ */}
                            {panelTab === "ALGORITHMS" && (
                                <>
                                    <PanelSection label="graph algorithms" hint="Server-side execution via /algo API">
                                        <div className="space-y-2 mt-2">
                                            {ALGO_OPTIONS.map(algo => {
                                                const Icon = algo.icon
                                                const isRunning = runningAlgo === algo.value
                                                const hasResult = algoResults.some(r => r.name === algo.value)
                                                return (
                                                    <button
                                                        key={algo.value}
                                                        onClick={() => runAlgorithm(algo.value)}
                                                        disabled={isRunning}
                                                        className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-left transition-all ${
                                                            hasResult
                                                                ? "bg-[#00f0ff]/10 border border-[#00f0ff]/20"
                                                                : "bg-white/5 border border-border/10 hover:bg-white/10 hover:border-[#00f0ff]/30"
                                                        }`}
                                                    >
                                                        <Icon className={`h-3.5 w-3.5 ${isRunning ? "animate-spin text-[#ff00ff]" : hasResult ? "text-[#00f0ff]" : "text-muted-foreground"}`} />
                                                        <div className="flex-1 min-w-0">
                                                            <div className="text-[10px] font-mono font-bold text-foreground">{algo.label}</div>
                                                            <div className="text-[8px] text-muted-foreground/70">{algo.desc}</div>
                                                        </div>
                                                        {isRunning && <span className="text-[8px] text-[#ff00ff] animate-pulse">RUNNING</span>}
                                                        {hasResult && !isRunning && <span className="text-[8px] text-[#00ff66]">DONE</span>}
                                                    </button>
                                                )
                                            })}
                                        </div>
                                    </PanelSection>

                                    {/* Algorithm Results */}
                                    {algoResults.length > 0 && (
                                        <PanelSection label="results">
                                            <div className="space-y-3 mt-2">
                                                {algoResults.map(result => (
                                                    <div key={result.name} className="bg-black/40 rounded-md p-3 border border-border/10">
                                                        <div className="text-[9px] font-mono font-bold text-[#00f0ff] mb-2">
                                                            {result.name.toUpperCase()}
                                                        </div>
                                                        <div className="space-y-1 max-h-40 overflow-y-auto custom-scrollbar">
                                                            {Array.isArray(result.data) ? (
                                                                result.data.slice(0, 10).map((item: { id?: string; node_id?: string; score?: number; rank?: number }, i: number) => (
                                                                    <div key={i} className="flex justify-between text-[9px] font-mono">
                                                                        <span className="text-muted-foreground truncate mr-2">
                                                                            {(item.id || item.node_id || "").substring(0, 16)}...
                                                                        </span>
                                                                        <span className="text-[#ff00ff]">
                                                                            {(item.score ?? item.rank ?? 0).toFixed?.(4) ?? item.score}
                                                                        </span>
                                                                    </div>
                                                                ))
                                                            ) : (
                                                                <div className="text-[9px] text-muted-foreground">
                                                                    {JSON.stringify(result.data).substring(0, 200)}
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </PanelSection>
                                    )}

                                    <div className="text-[8px] text-muted-foreground/50 font-mono mt-4 px-1">
                                        Client-side algorithms (centrality, community, pathfinding, TDA)
                                        are computed automatically by the PerspektiveEngine.
                                        Betti numbers (B0, B1) are shown in the engine HUD.
                                    </div>
                                </>
                            )}

                            {/* ═══ LAYOUT TAB ═══ */}
                            {panelTab === "LAYOUT" && (
                                <>
                                    <PanelSection label="layout engine" hint="Perspective.js built-in layouts">
                                        <div className="space-y-2 mt-2">
                                            {LAYOUT_OPTIONS.map(layout => {
                                                const Icon = layout.icon
                                                const isActive = selectedLayout === layout.value
                                                return (
                                                    <button
                                                        key={layout.value}
                                                        onClick={() => setSelectedLayout(layout.value)}
                                                        className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-left transition-all ${
                                                            isActive
                                                                ? "bg-[#00f0ff]/10 border border-[#00f0ff]/30 text-[#00f0ff]"
                                                                : "bg-white/5 border border-border/10 hover:bg-white/10 text-muted-foreground"
                                                        }`}
                                                    >
                                                        <Icon className="h-3.5 w-3.5" />
                                                        <span className="text-[10px] font-mono font-bold">{layout.label}</span>
                                                        {isActive && <span className="ml-auto text-[8px] opacity-70">ACTIVE</span>}
                                                    </button>
                                                )
                                            })}
                                        </div>
                                    </PanelSection>

                                    <PanelSection label="theme" hint="Animated transitions between presets">
                                        <div className="space-y-2 mt-2">
                                            {THEME_OPTIONS.map(theme => {
                                                const isActive = selectedTheme === theme.value
                                                return (
                                                    <button
                                                        key={theme.value}
                                                        onClick={() => setSelectedTheme(theme.value)}
                                                        className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-left transition-all ${
                                                            isActive
                                                                ? "bg-white/10 border border-border/30"
                                                                : "bg-white/5 border border-border/10 hover:bg-white/10"
                                                        }`}
                                                    >
                                                        <div className="w-3 h-3 rounded-full" style={{ background: theme.color, boxShadow: isActive ? `0 0 8px ${theme.color}` : "none" }} />
                                                        <span className={`text-[10px] font-mono font-bold ${isActive ? "text-foreground" : "text-muted-foreground"}`}>
                                                            {theme.label}
                                                        </span>
                                                        {isActive && <Palette className="ml-auto h-3 w-3 text-muted-foreground" />}
                                                    </button>
                                                )
                                            })}
                                        </div>
                                    </PanelSection>

                                    <div className="text-[8px] text-muted-foreground/50 font-mono mt-4 px-1">
                                        The PerspektiveEngine includes built-in layout engines
                                        (Force, Radial, Tree, Grid, Concentric, WebGPU).
                                        Theme transitions are animated via lerp interpolation.
                                        Use the engine HUD (top-right) to toggle overlays:
                                        Heatmap, Energy Pulse, Kinetic Flow, Daemons, Emotion.
                                    </div>
                                </>
                            )}

                            {/* ═══ EXPORT TAB ═══ */}
                            {panelTab === "EXPORT" && (
                                <>
                                    <PanelSection label="graph export" hint="Engine-side (PNG/JSON/GraphML) in top-right HUD">
                                        <div className="space-y-2 mt-2">
                                            <div className="text-[9px] text-muted-foreground/70 font-mono">
                                                The PerspektiveEngine HUD (top-right corner) provides:
                                            </div>
                                            <div className="space-y-1 mt-1">
                                                {[
                                                    { label: "PNG Screenshot", desc: "Capture WebGL canvas as PNG image" },
                                                    { label: "JSON Export", desc: "Raw graph data with positions" },
                                                    { label: "GraphML Export", desc: "Interoperable XML graph format" },
                                                ].map(item => (
                                                    <div key={item.label} className="flex items-center gap-2 px-3 py-2 bg-white/5 rounded-md border border-border/10">
                                                        <Download className="h-3 w-3 text-[#00f0ff]" />
                                                        <div>
                                                            <div className="text-[10px] font-mono font-bold text-foreground">{item.label}</div>
                                                            <div className="text-[8px] text-muted-foreground/70">{item.desc}</div>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </PanelSection>

                                    <PanelSection label="server-side export" hint="Data export via NietzscheDB API">
                                        <div className="space-y-2 mt-2">
                                            <button
                                                onClick={handleExportJSONL}
                                                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-md bg-white/5 border border-border/10 hover:bg-white/10 hover:border-[#00f0ff]/30 transition-all text-left"
                                            >
                                                <Download className="h-3.5 w-3.5 text-[#00ff66]" />
                                                <div>
                                                    <div className="text-[10px] font-mono font-bold text-foreground">EXPORT JSONL</div>
                                                    <div className="text-[8px] text-muted-foreground/70">All nodes as JSON Lines</div>
                                                </div>
                                            </button>
                                            <button
                                                onClick={handleExportCSV}
                                                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-md bg-white/5 border border-border/10 hover:bg-white/10 hover:border-[#00f0ff]/30 transition-all text-left"
                                            >
                                                <Download className="h-3.5 w-3.5 text-[#f59e0b]" />
                                                <div>
                                                    <div className="text-[10px] font-mono font-bold text-foreground">EXPORT CSV</div>
                                                    <div className="text-[8px] text-muted-foreground/70">Tabular format for analysis</div>
                                                </div>
                                            </button>
                                        </div>
                                    </PanelSection>

                                    <PanelSection label="backup" hint="Create a snapshot backup">
                                        <button
                                            onClick={async () => {
                                                try {
                                                    await api.post("/backup", { label: `dashboard-${Date.now()}`, collection })
                                                } catch { /* ignore */ }
                                            }}
                                            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-md bg-[#ff00ff]/5 border border-[#ff00ff]/20 hover:bg-[#ff00ff]/10 transition-all text-left mt-2"
                                        >
                                            <Download className="h-3.5 w-3.5 text-[#ff00ff]" />
                                            <div>
                                                <div className="text-[10px] font-mono font-bold text-foreground">CREATE BACKUP</div>
                                                <div className="text-[8px] text-muted-foreground/70">Snapshot current state</div>
                                            </div>
                                        </button>
                                    </PanelSection>
                                </>
                            )}
                        </div>
                    </div>
                )}

                {/* ── Graph canvas (PerspektiveEngine) ──────────────────────── */}
                <div className="flex-1 relative overflow-hidden bg-[#000]">
                    {error && (
                        <div className="absolute top-32 left-1/2 -translate-x-1/2 z-40 flex items-center gap-2 bg-destructive/20 text-destructive border border-destructive/30 rounded-lg px-4 py-2 text-sm backdrop-blur-md">
                            <AlertCircle className="h-4 w-4" />
                            Connection Lost: NietzscheDB Unreachable
                        </div>
                    )}

                    {isLoading && !graphData && (
                        <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/20 backdrop-blur-[2px]">
                            <div className="flex flex-col items-center gap-4">
                                <RefreshCw className="h-10 w-10 animate-spin text-[#00f0ff]" />
                                <span className="text-[10px] font-mono tracking-[0.3em] text-[#00f0ff] animate-pulse">RECONSTRUCTING NEURAL GRAPH...</span>
                            </div>
                        </div>
                    )}

                    <PerspektiveView
                        collection={collection}
                        apiBase=""
                        limit={limit}
                        zoom={300}
                        bloomIntensity={1.5}
                        lerpRate={0.05}
                        streamingMode={streamingMode}
                        enableDrag={true}
                        enableBoxSelect={true}
                        enableContextMenu={true}
                        enableMobiusZoom={true}
                    />
                </div>
            </div>

            {/* ── Features Legend (bottom-right, above engine controls) ──────── */}
            <div className="absolute bottom-20 right-6 z-20 pointer-events-none">
                <div className="text-[8px] font-mono text-muted-foreground/40 space-y-0.5 text-right">
                    <div>[F] Fit View | [M] Cycle Manifolds</div>
                    <div>Scroll = Mobius Zoom (Poincare)</div>
                    <div>Drag Nodes | Box Select | Right-Click Menu</div>
                </div>
            </div>
        </div>
    )
}

// ── Helper Components ──────────────────────────────────────────────────────────

function PanelSection({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
    return (
        <div>
            <div className="flex items-center gap-2 mb-1.5">
                <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
                    {label}
                </div>
                {hint && (
                    <div className="text-[8px] text-muted-foreground/50 italic">
                        {hint}
                    </div>
                )}
            </div>
            {children}
        </div>
    )
}

function StatCard({ label, value, color }: { label: string; value: string; color: string }) {
    return (
        <div className="bg-black/40 rounded-md p-2.5 border border-border/10">
            <div className="text-[8px] font-mono text-muted-foreground/70 uppercase tracking-wider">{label}</div>
            <div className="text-sm font-mono font-bold mt-0.5" style={{ color }}>{value}</div>
        </div>
    )
}
