import { useState, useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { ChevronRight, RefreshCw, AlertCircle, Search, X } from "lucide-react"
import { PerspektiveView } from "@/components/PerspektiveView"
import type { ViewNodeData, ViewEdgeData, ManifoldType } from "@/components/PerspektiveView"
import { BarChart, Bar, ResponsiveContainer } from "recharts"

// ─── Types ─────────────────────────────────────────────────────────────────────

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

// ─── Constants ──────────────────────────────────────────────────────────────────

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

// ─── Data Projection ────────────────────────────────────────────────────────────

function hashToAngle(id: string): number {
    let hash = 0
    for (let i = 0; i < id.length; i++) {
        hash = ((hash << 5) - hash) + id.charCodeAt(i)
        hash |= 0
    }
    return (hash / 2147483647) * Math.PI * 2
}

function projectNode(node: NodeRecord, manifold: ManifoldType): { x: number; y: number; z: number } {
    const angle = hashToAngle(node.id)
    const radius = Math.min(node.depth ?? 0.5, 0.99)

    if (manifold === "POINCARE") {
        return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius, z: 0.01 }
    }
    if (manifold === "RIEMANN") {
        const px = Math.cos(angle) * radius * 3
        const py = Math.sin(angle) * radius * 3
        const denom = 1 + px * px + py * py
        return { x: (2 * px) / denom, y: (2 * py) / denom, z: (px * px + py * py - 1) / denom }
    }
    if (manifold === "MINKOWSKI") {
        return { x: Math.cos(angle) * radius * 2, y: (node.energy * 5) - 2.5, z: Math.sin(angle) * radius * 2 }
    }
    return { x: (node.energy * 2) - 1, y: (node.depth * 2) - 1, z: 0.01 }
}

/** Extract a human label from a node's content, falling back to truncated ID */
function extractLabel(n: NodeRecord): string {
    if (typeof n.content?.label === "string") return n.content.label
    if (typeof n.content?.title === "string") return n.content.title
    if (typeof n.content?.name === "string") return n.content.name
    return n.id.slice(0, 12)
}

// ─── GraphExplorerPage ─────────────────────────────────────────────────────────

export function GraphExplorerPage() {
    const collection = DEFAULT_COLLECTION
    const [limit, setLimit]             = useState(1000)
    const [panelOpen, setPanelOpen]     = useState(true)
    const [panelTab, setPanelTab]       = useState<"POINTS" | "LINKS">("POINTS")
    const [manifold, setManifold]       = useState<ManifoldType>("POINCARE")
    const [searchTerm, setSearchTerm]   = useState("")
    const [activeFilter, setActiveFilter] = useState<string | null>(null)

    // ── Graph data fetch ──────────────────────────────────────────────────────
    const { data: graphData, isLoading, error, refetch } = useQuery<GraphResponse>({
        queryKey: ["graph", collection, limit],
        queryFn: () => api.get(`/graph?collection=${collection}&limit=${limit}`).then(r => r.data),
        staleTime: 30_000,
    })

    // ── Map API data → PerspektiveView format ─────────────────────────────────
    const viewNodes = useMemo<ViewNodeData[]>(() => {
        if (!graphData?.nodes) return []
        return graphData.nodes.map(n => ({
            id: n.id,
            node_type: n.node_type,
            energy: n.energy,
            depth: n.depth,
            label: extractLabel(n),
            hausdorff: n.hausdorff ?? 0,
            causal_chain: n.causal_chain,
            is_archived: n.is_archived,
            ...projectNode(n, manifold),
        }))
    }, [graphData, manifold])

    const viewEdges = useMemo<ViewEdgeData[]>(() => {
        if (!graphData?.edges) return []
        return graphData.edges.map(e => ({
            source: e.from,
            target: e.to,
            weight: e.weight ?? 0.5,
        }))
    }, [graphData])

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

    const hasActiveFilters = searchTerm !== "" || activeFilter !== null

    const resetFilters = () => {
        setSearchTerm("")
        setActiveFilter(null)
    }

    // ─── Render ───────────────────────────────────────────────────────────────
    return (
        <div className="flex flex-col h-screen relative bg-[#020617] overflow-hidden">
            
            {/* ── Cockpit HUD Overlay (Top Left) ────────────────────────────────── */}
            <div className="absolute top-6 left-6 z-30 pointer-events-none">
                <div className="flex flex-col gap-1">
                    <h2 className="text-[#00f0ff] text-xl font-mono font-bold tracking-tighter" style={{ textShadow: "0 0 10px rgba(0, 240, 255, 0.5)" }}>
                        NIETZSCHEDB // PERSPEKTIVE
                    </h2>
                    <div className="flex items-center gap-3 pointer-events-auto mt-2">
                        <div className="relative">
                            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                            <input
                                type="text"
                                placeholder="BUSCAR MENTE DA EVA..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className="h-9 w-72 bg-black/60 border border-[#00f0ff]/30 rounded-md pl-9 pr-8 text-xs font-mono text-white placeholder:text-muted-foreground/50 focus:outline-none focus:border-[#00f0ff] focus:ring-1 focus:ring-[#00f0ff] transition-all backdrop-blur-md"
                                style={{ boxShadow: searchTerm ? "0 0 15px rgba(0, 240, 255, 0.2)" : "none" }}
                            />
                            {searchTerm && (
                                <button
                                    onClick={() => setSearchTerm("")}
                                    className="absolute right-2 top-1/2 -translate-y-1/2 h-5 w-5 rounded hover:bg-white/10 flex items-center justify-center text-muted-foreground transition-colors"
                                >
                                    <X className="h-3 w-3" />
                                </button>
                            )}
                        </div>

                        {/* Reload & Config Buttons */}
                        <div className="flex items-center gap-1">
                            <button
                                onClick={() => refetch()}
                                disabled={isLoading}
                                className="h-9 w-9 flex items-center justify-center bg-black/60 border border-border/20 rounded-md text-muted-foreground hover:text-[#00f0ff] hover:border-[#00f0ff]/50 transition-all backdrop-blur-md"
                                title="Refetch Graph"
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
                    </div>

                    <div className="flex items-center gap-2 mt-2 font-mono text-[10px] text-muted-foreground">
                        <span className="flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-[#00ff66] animate-pulse" />
                            SSE STREAMING
                        </span>
                        <span>|</span>
                        <span>VISÍVEIS: <span className="text-white font-bold">{viewNodes.length}</span> / {graphData?.nodes.length ?? 0}</span>
                        {hasActiveFilters && (
                            <>
                                <span>|</span>
                                <button onClick={resetFilters} className="text-[#f87171] hover:underline uppercase tracking-widest">
                                    [ RESETAR LENTES ]
                                </button>
                            </>
                        )}
                    </div>
                </div>
            </div>

            {/* ── Subtitle / Collection Info (Top Right) ────────────────────────── */}
            <div className="absolute top-6 right-6 z-20 text-right pointer-events-none">
                <div className="text-[10px] font-mono tracking-widest text-[#94a3b8] uppercase">
                    Córtex: <span className="text-[#00f0ff]">{collection}</span>
                </div>
                <div className="text-[9px] font-mono text-muted-foreground/60 mt-1">
                    MODO: {manifold}
                </div>
            </div>

            {/* ── Main area ────────────────────────────────────────────────── */}
            <div className="flex flex-1 overflow-hidden relative">

                {/* ── Left analysis panel (Now floating and translucent) ────── */}
                {panelOpen && (
                    <div className="absolute top-24 left-6 bottom-24 w-80 flex flex-col z-20 bg-black/70 border border-border/20 rounded-lg backdrop-blur-xl overflow-hidden shadow-2xl">
                        {/* Tab switcher */}
                        <div className="flex border-b border-border/20 px-3 shrink-0 bg-white/5">
                            {(["POINTS", "LINKS"] as const).map(tab => (
                                <button
                                    key={tab}
                                    onClick={() => setPanelTab(tab)}
                                    className={`px-3 py-2 text-[10px] font-bold tracking-widest transition-colors ${panelTab === tab ? "border-b border-[#00f0ff] text-[#00f0ff]" : "text-muted-foreground hover:text-foreground"}`}
                                >
                                    {tab}
                                </button>
                            ))}
                        </div>

                        <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-6">
                            {panelTab === "POINTS" && (
                                <>
                                    <FilterSection label="distribuição de tipos">
                                        <div className="space-y-1.5 mt-2">
                                            {Object.entries(NODE_TYPE_COLORS).map(([type, color]) => (
                                                <button
                                                    key={type}
                                                    onClick={() => setActiveFilter(prev => prev === type ? null : type)}
                                                    className={`flex items-center gap-2 text-[10px] w-full px-2 py-1.5 rounded transition-all ${
                                                        activeFilter === type
                                                            ? "bg-[#00f0ff]/10 text-[#00f0ff] border border-[#00f0ff]/20"
                                                            : "text-muted-foreground hover:bg-white/5"
                                                    }`}
                                                >
                                                    <div className="w-2 h-2 rounded-full" style={{ background: color, opacity: activeFilter === null || activeFilter === type ? 1 : 0.3 }} />
                                                    <span className="font-mono">{type.toUpperCase()}</span>
                                                    {activeFilter === type && <span className="ml-auto text-[8px] opacity-70">ACTIVE</span>}
                                                </button>
                                            ))}
                                        </div>
                                    </FilterSection>

                                    <FilterSection label="energia (ubermensch)">
                                        <ResponsiveContainer width="100%" height={60}>
                                            <BarChart data={energyHistData}>
                                                <Bar dataKey="count" fill="#ff00ff" radius={[2, 2, 0, 0]} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </FilterSection>

                                    <FilterSection label="profundidade (poincaré)">
                                        <ResponsiveContainer width="100%" height={60}>
                                            <BarChart data={depthHistData}>
                                                <Bar dataKey="count" fill="#00f0ff" radius={[2, 2, 0, 0]} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </FilterSection>

                                    <div className="pt-4 border-t border-border/20">
                                        <div className="text-[9px] font-mono text-muted-foreground mb-2 uppercase">Configuração de Carga</div>
                                        <div className="flex gap-2">
                                            <select
                                                value={limit}
                                                onChange={e => setLimit(Number(e.target.value))}
                                                className="flex-1 bg-black/40 border border-border/20 rounded px-2 py-1 text-[10px] font-mono text-foreground focus:outline-none focus:border-[#00f0ff]"
                                            >
                                                {[500, 1000, 2000, 5000].map(n => <option key={n} value={n}>{n} NODES</option>)}
                                            </select>
                                        </div>
                                    </div>
                                </>
                            )}

                            {panelTab === "LINKS" && (
                                <FilterSection label="tipos de conexões">
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
                                </FilterSection>
                            )}
                        </div>
                    </div>
                )}

                {/* ── Graph canvas ──────────────────────────────────────── */}
                <div className="flex-1 relative overflow-hidden bg-[#000]">
                    {error && (
                        <div className="absolute top-32 left-1/2 -translate-x-1/2 z-40 flex items-center gap-2 bg-destructive/20 text-destructive border border-destructive/30 rounded-lg px-4 py-2 text-sm backdrop-blur-md">
                            <AlertCircle className="h-4 w-4" />
                            Connection Lost: NietzscheDB Unreachable
                        </div>
                    )}

                    {isLoading && (
                        <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/20 backdrop-blur-[2px]">
                            <div className="flex flex-col items-center gap-4">
                                <RefreshCw className="h-10 w-10 animate-spin text-[#00f0ff]" />
                                <span className="text-[10px] font-mono tracking-[0.3em] text-[#00f0ff] animate-pulse">RECONSTRUCTING NEURAL GRAPH...</span>
                            </div>
                        </div>
                    )}

                    <PerspektiveView
                        nodes={viewNodes}
                        edges={viewEdges}
                        manifold={manifold}
                        onManifoldChange={setManifold}
                        searchTerm={searchTerm}
                        activeFilter={activeFilter}
                    />
                </div>
            </div>
        </div>
    )
}

// ─── FilterSection helper ──────────────────────────────────────────────────────

function FilterSection({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
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
