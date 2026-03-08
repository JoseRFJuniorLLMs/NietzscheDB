import { useState, useMemo, useCallback } from "react"
import { useQuery } from "@tanstack/react-query"
import { api, listCollections, invokeZaratustra, dreamFrom, applyDream, rejectDream, narrateArcs, listDaemons } from "@/lib/api"
import type { ZaratustraResult, DreamSession, NarrativeArc } from "@/lib/api"
import {
    ChevronRight, RefreshCw, AlertCircle,
    Network, GitBranch, Download, Palette,
    BarChart3, Cpu, Route, Users, Zap,
    Sparkles, Brain, BookOpen,
} from "lucide-react"
import { PerspektiveView } from "@/components/PerspektiveView"
import type { StreamingMode } from "@/components/PerspektiveView"
import { EventTimeline } from "@/components/EventTimeline"
import type { TimelineEvent } from "@/components/EventTimeline"
import { CDCEventPanel } from "@/components/CDCEventPanel"
import { NodeInspector } from "@/components/NodeInspector"
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
    const [collection, setCollection]           = useState(DEFAULT_COLLECTION)
    const [limit, setLimit]                     = useState(2000)
    const [panelOpen, setPanelOpen]             = useState(true)
    const [panelTab, setPanelTab]               = useState<"ANALYTICS" | "ALGORITHMS" | "LAYOUT" | "EXPORT" | "AGI">("ANALYTICS")
    const [streamingMode, setStreamingMode]     = useState<StreamingMode>("sse")
    const [selectedLayout, setSelectedLayout]   = useState("default")
    const [selectedTheme, setSelectedTheme]     = useState("cyberpunk")
    const [algoResults, setAlgoResults]         = useState<AlgoResult[]>([])
    const [runningAlgo, setRunningAlgo]         = useState<string | null>(null)

    // AGI features state
    const [zaratustraResult, setZaratustraResult] = useState<ZaratustraResult | null>(null)
    const [zaratustraRunning, setZaratustraRunning] = useState(false)
    const [dreamSession, setDreamSession]       = useState<DreamSession | null>(null)
    const [dreamSeedId, setDreamSeedId]         = useState("")
    const [dreamRunning, setDreamRunning]       = useState(false)
    const [narrativeArcs, setNarrativeArcs]     = useState<NarrativeArc[]>([])
    const [narrativeLoading, setNarrativeLoading] = useState(false)

    // Timeline & CDC state
    const [timelineEvents, setTimelineEvents]   = useState<TimelineEvent[]>([])
    const [showTimeline, setShowTimeline]       = useState(true)
    const [showCDCPanel, setShowCDCPanel]       = useState(false)
    const [inspectedNodeId, setInspectedNodeId] = useState<string | null>(null)

    // Collections list for dynamic selector
    const { data: collectionsData } = useQuery({
        queryKey: ["collections"],
        queryFn: () => listCollections(),
        staleTime: 60_000,
    })

    // Live daemons
    const { data: daemonsData } = useQuery({
        queryKey: ["daemons", collection],
        queryFn: () => listDaemons(collection),
        refetchInterval: 5_000,
    })

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

                    <div className="flex items-center gap-2 mt-2 font-mono text-[10px] text-muted-foreground">
                        <span className="flex items-center gap-1 pointer-events-none">
                            <span className="w-1.5 h-1.5 rounded-full bg-[#00ff66] animate-pulse" />
                            {streamingMode === "ws" ? "WS STREAMING" : "SSE STREAMING"}
                        </span>
                        <span className="pointer-events-none">|</span>
                        <span className="pointer-events-none">NODES: <span className="text-white font-bold">{graphData?.nodes.length ?? 0}</span></span>
                        <span className="pointer-events-none">|</span>
                        <span className="pointer-events-none">EDGES: <span className="text-white font-bold">{graphData?.edges.length ?? 0}</span></span>
                        <span className="pointer-events-none">|</span>
                        <button onClick={() => setShowTimeline(t => !t)} className={`px-1.5 py-0.5 rounded text-[9px] font-bold border transition-colors ${showTimeline ? "border-[#00f0ff] text-[#00f0ff] bg-[#00f0ff]/10" : "border-border/30 text-muted-foreground hover:text-white"}`}>
                            TIMELINE
                        </button>
                        <button onClick={() => setShowCDCPanel(c => !c)} className={`px-1.5 py-0.5 rounded text-[9px] font-bold border transition-colors ${showCDCPanel ? "border-[#00ff66] text-[#00ff66] bg-[#00ff66]/10" : "border-border/30 text-muted-foreground hover:text-white"}`}>
                            CDC
                        </button>
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
                            {(["ANALYTICS", "ALGORITHMS", "LAYOUT", "EXPORT", "AGI"] as const).map(tab => (
                                <button
                                    key={tab}
                                    onClick={() => setPanelTab(tab)}
                                    className={`px-2.5 py-2 text-[9px] font-bold tracking-widest transition-colors whitespace-nowrap ${panelTab === tab ? (tab === "AGI" ? "border-b border-[#ffd700] text-[#ffd700]" : "border-b border-[#00f0ff] text-[#00f0ff]") : "text-muted-foreground hover:text-foreground"}`}
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
                                        <div className="flex gap-2 mb-2">
                                            <select
                                                value={collection}
                                                onChange={e => setCollection(e.target.value)}
                                                className="flex-1 bg-black/40 border border-border/20 rounded px-2 py-1 text-[10px] font-mono text-[#00f0ff] focus:outline-none focus:border-[#00f0ff]"
                                            >
                                                {collectionsData?.collections?.map(c => (
                                                    <option key={c.name} value={c.name}>{c.name} ({c.vector_count})</option>
                                                )) ?? <option value={collection}>{collection}</option>}
                                            </select>
                                        </div>
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

                            {/* ═══ AGI TAB (Zaratustra, Dream, Narrative) ═══ */}
                            {panelTab === "AGI" && (
                                <>
                                    {/* Zaratustra */}
                                    <PanelSection label="zaratustra" hint="Will-to-Power + Eternal Recurrence + Ubermensch">
                                        <button
                                            onClick={async () => {
                                                setZaratustraRunning(true)
                                                try {
                                                    const result = await invokeZaratustra(collection)
                                                    setZaratustraResult(result)
                                                } catch { /* ignore */ }
                                                finally { setZaratustraRunning(false) }
                                            }}
                                            disabled={zaratustraRunning}
                                            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-md bg-[#ffd700]/5 border border-[#ffd700]/20 hover:bg-[#ffd700]/10 transition-all text-left mt-2"
                                        >
                                            <Sparkles className={`h-4 w-4 text-[#ffd700] ${zaratustraRunning ? "animate-spin" : ""}`} />
                                            <div>
                                                <div className="text-[10px] font-mono font-bold text-foreground">INVOKE ZARATUSTRA</div>
                                                <div className="text-[8px] text-muted-foreground/70">3-phase evolution cycle</div>
                                            </div>
                                            {zaratustraRunning && <span className="text-[8px] text-[#ffd700] animate-pulse ml-auto">RUNNING</span>}
                                        </button>
                                        {zaratustraResult && (
                                            <div className="mt-2 bg-black/40 rounded-md p-3 border border-[#ffd700]/20 space-y-1">
                                                <div className="grid grid-cols-2 gap-2">
                                                    <div className="text-[9px] font-mono"><span className="text-muted-foreground">Updated:</span> <span className="text-[#ffd700]">{zaratustraResult.nodes_updated}</span></div>
                                                    <div className="text-[9px] font-mono"><span className="text-muted-foreground">Delta:</span> <span className="text-[#ff00ff]">{zaratustraResult.energy_delta?.toFixed(4)}</span></div>
                                                    <div className="text-[9px] font-mono"><span className="text-muted-foreground">Echoes:</span> <span className="text-[#8b5cf6]">{zaratustraResult.echoes_created}</span></div>
                                                    <div className="text-[9px] font-mono"><span className="text-muted-foreground">Elite:</span> <span className="text-[#ffd700]">{zaratustraResult.elite_count}</span></div>
                                                </div>
                                            </div>
                                        )}
                                    </PanelSection>

                                    {/* Dream Engine */}
                                    <PanelSection label="dream engine" hint="Speculative graph evolution">
                                        <div className="flex gap-2 mt-2">
                                            <input
                                                value={dreamSeedId}
                                                onChange={e => setDreamSeedId(e.target.value)}
                                                placeholder="Seed Node ID..."
                                                className="flex-1 bg-black/40 border border-border/20 rounded px-2 py-1.5 text-[10px] font-mono text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-[#ffd700]"
                                            />
                                            <button
                                                onClick={async () => {
                                                    if (!dreamSeedId) return
                                                    setDreamRunning(true)
                                                    try {
                                                        const result = await dreamFrom(dreamSeedId, 3, 0.1, collection)
                                                        setDreamSession(result)
                                                    } catch { /* ignore */ }
                                                    finally { setDreamRunning(false) }
                                                }}
                                                disabled={dreamRunning || !dreamSeedId}
                                                className="px-3 py-1.5 bg-[#ffd700]/10 border border-[#ffd700]/30 rounded text-[9px] font-mono font-bold text-[#ffd700] hover:bg-[#ffd700]/20 transition-all disabled:opacity-30"
                                            >
                                                {dreamRunning ? "..." : "DREAM"}
                                            </button>
                                        </div>
                                        {dreamSession && (
                                            <div className="mt-2 bg-black/40 rounded-md p-3 border border-[#ffd700]/20">
                                                <div className="flex justify-between items-center mb-2">
                                                    <span className="text-[9px] font-mono text-[#ffd700] font-bold">SESSION {dreamSession.id?.substring(0, 8)}</span>
                                                    <span className={`text-[8px] font-mono font-bold ${dreamSession.status === "pending" ? "text-[#ffd700]" : dreamSession.status === "applied" ? "text-[#00ff66]" : "text-[#ff4444]"}`}>
                                                        {dreamSession.status?.toUpperCase()}
                                                    </span>
                                                </div>
                                                <div className="text-[9px] font-mono text-muted-foreground">
                                                    +{dreamSession.nodes?.length ?? 0} nodes | +{dreamSession.edges?.length ?? 0} edges
                                                </div>
                                                {dreamSession.status === "pending" && (
                                                    <div className="flex gap-2 mt-2">
                                                        <button
                                                            onClick={async () => { try { await applyDream(dreamSession.id, collection); setDreamSession(s => s ? { ...s, status: "applied" } : null) } catch {} }}
                                                            className="flex-1 px-2 py-1 bg-[#00ff66]/10 border border-[#00ff66]/30 rounded text-[9px] font-mono font-bold text-[#00ff66]"
                                                        >APPLY</button>
                                                        <button
                                                            onClick={async () => { try { await rejectDream(dreamSession.id, collection); setDreamSession(s => s ? { ...s, status: "rejected" } : null) } catch {} }}
                                                            className="flex-1 px-2 py-1 bg-[#ff4444]/10 border border-[#ff4444]/30 rounded text-[9px] font-mono font-bold text-[#ff4444]"
                                                        >REJECT</button>
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </PanelSection>

                                    {/* Narrative */}
                                    <PanelSection label="narrative arcs" hint="Story arcs from graph evolution">
                                        <button
                                            onClick={async () => {
                                                setNarrativeLoading(true)
                                                try {
                                                    const result = await narrateArcs(24, collection)
                                                    setNarrativeArcs(result.arcs ?? [])
                                                } catch { /* ignore */ }
                                                finally { setNarrativeLoading(false) }
                                            }}
                                            disabled={narrativeLoading}
                                            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-md bg-white/5 border border-border/10 hover:bg-white/10 hover:border-[#8b5cf6]/30 transition-all text-left mt-2"
                                        >
                                            <BookOpen className={`h-3.5 w-3.5 text-[#8b5cf6] ${narrativeLoading ? "animate-pulse" : ""}`} />
                                            <div>
                                                <div className="text-[10px] font-mono font-bold text-foreground">NARRATE (24h)</div>
                                                <div className="text-[8px] text-muted-foreground/70">Detect narrative arcs</div>
                                            </div>
                                        </button>
                                        {narrativeArcs.length > 0 && (
                                            <div className="mt-2 space-y-1.5 max-h-48 overflow-y-auto custom-scrollbar">
                                                {narrativeArcs.map(arc => {
                                                    const arcColors: Record<string, string> = { emergence: "#00ff66", conflict: "#ff4444", decay: "#666", recurrence: "#8b5cf6", synthesis: "#00f0ff" }
                                                    return (
                                                        <div key={arc.id} className="flex items-center gap-2 px-2 py-1.5 bg-black/30 rounded text-[9px] font-mono">
                                                            <div className="w-2 h-2 rounded-full shrink-0" style={{ background: arcColors[arc.type] ?? "#fff" }} />
                                                            <span className="text-muted-foreground flex-1 truncate">{arc.description}</span>
                                                            <span style={{ color: arcColors[arc.type] }}>{(arc.intensity * 100).toFixed(0)}%</span>
                                                        </div>
                                                    )
                                                })}
                                            </div>
                                        )}
                                    </PanelSection>

                                    <div className="text-[8px] text-muted-foreground/50 font-mono mt-4 px-1">
                                        Zaratustra runs 3 phases: Will-to-Power (energy propagation),
                                        Eternal Recurrence (temporal echo ring), Ubermensch (elite selection).
                                        Dream generates speculative evolution. Narrative detects story arcs.
                                    </div>
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
                        dreamSession={dreamSession ? {
                            dreamId: dreamSession.id,
                            ghostNodes: (dreamSession.nodes ?? []).map((n: any) => ({
                                id: n.id ?? `dream-${Math.random().toString(36).slice(2)}`,
                                x: n.x ?? 0, y: n.y ?? 0, z: n.z ?? 0.01,
                                label: n.label ?? n.content?.title ?? 'dream node',
                                color: '#ffd700',
                            })),
                            ghostEdges: (dreamSession.edges ?? []).map((e: any) => ({
                                source: e.source ?? e.from,
                                target: e.target ?? e.to,
                            })),
                        } : null}
                        onDreamAction={async (action, dreamId) => {
                            try {
                                if (action === 'apply') {
                                    await applyDream(dreamId, collection);
                                    setDreamSession(s => s ? { ...s, status: 'applied' } : null);
                                } else {
                                    await rejectDream(dreamId, collection);
                                    setDreamSession(s => s ? { ...s, status: 'rejected' } : null);
                                }
                            } catch {}
                        }}
                        zaratustraResult={zaratustraResult ? {
                            phase: 'ubermensch',
                            nodesUpdated: zaratustraResult.nodes_updated ?? 0,
                            energyDelta: zaratustraResult.energy_delta ?? 0,
                            eliteNodeIds: zaratustraResult.elite_ids ?? [],
                            echoNodeIds: zaratustraResult.echo_ids ?? [],
                        } : null}
                        narrativeArcs={narrativeArcs}
                        activeDaemons={daemonsData?.data?.map((d: any) => ({
                            id: d.id ?? d.daemon_id,
                            x: d.x ?? d.position?.x ?? 0,
                            y: d.y ?? d.position?.y ?? 0,
                            type: d.daemon_type === 'entropy_decay' ? 'entropy' as const
                                : d.daemon_type === 'evolution' ? 'evolution' as const
                                : 'patrol' as const,
                            energy: d.energy ?? d.health ?? 1.0,
                        }))}
                    />
                </div>
            </div>

            {/* ── Event Timeline Bar (bottom) ──────────────────────────────── */}
            {showTimeline && (
                <div className="border-t border-border/30 bg-black/80 backdrop-blur-sm">
                    <EventTimeline
                        events={timelineEvents}
                        onEventClick={(evt) => {
                            // Future: highlight node in graph
                            console.log("Timeline event clicked:", evt)
                        }}
                    />
                </div>
            )}

            {/* ── CDC Event Stream Panel (right side overlay) ─────────────── */}
            {showCDCPanel && (
                <div className="absolute top-16 right-6 z-30 w-72">
                    <CDCEventPanel
                        events={timelineEvents.filter(e =>
                            ["InsertNode", "UpdateNode", "DeleteNode", "InsertEdge", "DeleteEdge", "SleepCycle", "Zaratustra"].includes(e.type)
                        ).map(e => ({
                            id: e.id,
                            timestamp: e.timestamp,
                            event_type: e.type as any,
                            node_id: e.nodeId ?? null,
                        }))}
                    />
                </div>
            )}

            {/* ── Features Legend (bottom-right, above engine controls) ──────── */}
            <div className="absolute bottom-20 right-6 z-20 pointer-events-none">
                <div className="text-[8px] font-mono text-muted-foreground/40 space-y-0.5 text-right">
                    <div>[F] Fit View | [M] Cycle Manifolds | [T] Timeline | [C] CDC</div>
                    <div>Scroll = Mobius Zoom (Poincare)</div>
                    <div>Drag Nodes | Box Select | Right-Click Menu</div>
                </div>
            </div>

            {/* ── Node Inspector Drawer ──────────────────────────────────── */}
            <NodeInspector
                nodeId={inspectedNodeId}
                onClose={() => setInspectedNodeId(null)}
                collection={collection}
            />
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
