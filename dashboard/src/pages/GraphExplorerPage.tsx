import { useState, useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { ChevronLeft, ChevronRight, RefreshCw, AlertCircle, Search, X } from "lucide-react"
import { PerspektiveView } from "@/components/PerspektiveView"
import type { ViewNodeData, ViewEdgeData, ManifoldType } from "@/components/PerspektiveView"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts"

// ─── Types ─────────────────────────────────────────────────────────────────────

interface NodeRecord {
    id:         string
    node_type:  string
    energy:     number
    depth:      number
    hausdorff:  number
    created_at: number
    content:    Record<string, unknown>
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
    const [collection, setCollection]   = useState(DEFAULT_COLLECTION)
    const [limit, setLimit]             = useState(1000)
    const [panelOpen, setPanelOpen]     = useState(true)
    const [panelTab, setPanelTab]       = useState<"POINTS" | "LINKS">("POINTS")
    const [manifold, setManifold]       = useState<ManifoldType>("POINCARE")
    const [searchTerm, setSearchTerm]   = useState("")
    const [activeFilter, setActiveFilter] = useState<string | null>(null)

    // ── Collections list ──────────────────────────────────────────────────────
    const { data: collections } = useQuery({
        queryKey: ["collections"],
        queryFn: () => api.get("/collections").then(r => r.data as { name: string; node_count: number }[]),
    })

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

    // ── Chart data ────────────────────────────────────────────────────────────
    const nodeTypeChartData = useMemo(() => {
        if (!graphData?.nodes) return []
        const counts = new Map<string, number>()
        for (const n of graphData.nodes) {
            counts.set(n.node_type, (counts.get(n.node_type) ?? 0) + 1)
        }
        return Array.from(counts, ([name, count]) => ({ name, count }))
            .sort((a, b) => b.count - a.count)
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

    // ── Handle bar chart click → toggle node_type filter ──────────────────────
    const handleTypeBarClick = (data: { name: string }) => {
        setActiveFilter(prev => prev === data.name ? null : data.name)
    }

    // ─── Render ───────────────────────────────────────────────────────────────
    return (
        <div className="flex flex-col h-[calc(100vh-64px)] relative bg-[#0b0f1e] overflow-hidden rounded-lg border border-border/20">

            {/* ── Top toolbar ──────────────────────────────────────────────── */}
            <div className="flex items-center gap-2 px-3 py-2 border-b border-border/30 bg-card/60 backdrop-blur z-20 flex-shrink-0">
                {/* Collection selector */}
                <select
                    value={collection}
                    onChange={e => setCollection(e.target.value)}
                    className="h-7 text-xs rounded bg-muted border border-border/40 px-2 text-foreground"
                >
                    {(collections ?? [{ name: DEFAULT_COLLECTION, node_count: 0 }])
                        .filter((c: { name: string; node_count: number }) => c.node_count > 0 || c.name === collection)
                        .map((c: { name: string; node_count: number }) => (
                            <option key={c.name} value={c.name}>
                                {c.name} ({c.node_count ?? 0})
                            </option>
                        ))}
                </select>

                {/* Limit selector */}
                <select
                    value={limit}
                    onChange={e => setLimit(Number(e.target.value))}
                    className="h-7 text-xs rounded bg-muted border border-border/40 px-2 text-foreground"
                >
                    {[200, 500, 1000, 2000, 5000].map(n => (
                        <option key={n} value={n}>{n} nodes</option>
                    ))}
                </select>

                {/* Reload */}
                <button
                    onClick={() => refetch()}
                    disabled={isLoading}
                    className="h-7 px-2 rounded bg-muted hover:bg-muted/80 border border-border/40 text-xs text-muted-foreground flex items-center gap-1"
                >
                    <RefreshCw className={`h-3 w-3 ${isLoading ? "animate-spin" : ""}`} />
                    Reload
                </button>

                {/* Search input */}
                <div className="relative flex-1 max-w-xs">
                    <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
                    <input
                        type="text"
                        placeholder="Search nodes..."
                        value={searchTerm}
                        onChange={e => setSearchTerm(e.target.value)}
                        className="h-7 w-full text-xs rounded bg-muted border border-border/40 pl-7 pr-7 text-foreground placeholder:text-muted-foreground/60 font-mono focus:outline-none focus:border-[#00f0ff] transition-colors"
                    />
                    {searchTerm && (
                        <button
                            onClick={() => setSearchTerm("")}
                            className="absolute right-1.5 top-1/2 -translate-y-1/2 h-4 w-4 rounded-sm hover:bg-muted-foreground/20 flex items-center justify-center text-muted-foreground"
                        >
                            <X className="h-3 w-3" />
                        </button>
                    )}
                </div>

                <div className="flex-1" />

                {/* Stats */}
                <span className="text-xs text-muted-foreground font-mono">
                    {viewNodes.length.toLocaleString()} nodes · {viewEdges.length.toLocaleString()} links
                </span>

                {/* Reset filters */}
                {hasActiveFilters && (
                    <button
                        onClick={resetFilters}
                        className="h-7 px-2 rounded bg-destructive/20 hover:bg-destructive/30 border border-destructive/30 text-xs text-destructive flex items-center gap-1 transition-colors"
                    >
                        <X className="h-3 w-3" />
                        Reset
                    </button>
                )}

                {/* Active filter badge */}
                {activeFilter && (
                    <span className="h-6 px-2 rounded-full text-[10px] font-mono font-bold flex items-center gap-1"
                        style={{
                            background: NODE_TYPE_COLORS[activeFilter] ?? "#64748b",
                            color: "#000",
                        }}
                    >
                        {activeFilter}
                    </span>
                )}

                {/* Panel toggle */}
                <button
                    onClick={() => setPanelOpen(v => !v)}
                    className="h-7 w-7 rounded bg-muted hover:bg-muted/80 border border-border/40 flex items-center justify-center"
                >
                    {panelOpen ? <ChevronLeft className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                </button>
            </div>

            {/* ── Main area ────────────────────────────────────────────────── */}
            <div className="flex flex-1 overflow-hidden relative">

                {/* ── Left filter panel ─────────────────────────────────── */}
                {panelOpen && (
                    <div className="w-72 flex-shrink-0 border-r border-border/30 bg-card/80 backdrop-blur overflow-y-auto z-10 flex flex-col">

                        {/* Tab switcher */}
                        <div className="flex border-b border-border/30 px-3">
                            {(["POINTS", "LINKS"] as const).map(tab => (
                                <button
                                    key={tab}
                                    onClick={() => setPanelTab(tab)}
                                    className={`px-3 py-1.5 text-[11px] font-semibold tracking-wider transition-colors ${panelTab === tab ? "border-b-2 border-primary text-primary" : "text-muted-foreground hover:text-foreground"}`}
                                >
                                    {tab}
                                </button>
                            ))}
                            <div className="flex-1" />
                            {hasActiveFilters && (
                                <button
                                    onClick={resetFilters}
                                    className="text-[10px] text-destructive hover:text-destructive/80 transition-colors py-1.5 px-1"
                                >
                                    reset
                                </button>
                            )}
                        </div>

                        {panelTab === "POINTS" && (
                            <div className="p-3 space-y-4">
                                {/* Node type distribution — CLICKABLE for filtering */}
                                <FilterSection label="node_type" hint="click to filter">
                                    <ResponsiveContainer width="100%" height={Math.max(nodeTypeChartData.length * 18, 60)}>
                                        <BarChart data={nodeTypeChartData} layout="vertical">
                                            <XAxis type="number" hide />
                                            <YAxis type="category" dataKey="name" width={70} tick={{ fontSize: 10, fill: "#94a3b8" }} />
                                            <Tooltip
                                                contentStyle={{ background: "#0b0f1e", border: "1px solid #334155", fontSize: 11 }}
                                                labelStyle={{ color: "#00f0ff" }}
                                            />
                                            <Bar
                                                dataKey="count"
                                                radius={[0, 2, 2, 0]}
                                                cursor="pointer"
                                                onClick={(_data: unknown, index: number) => {
                                                    const entry = nodeTypeChartData[index]
                                                    if (entry) handleTypeBarClick(entry)
                                                }}
                                            >
                                                {nodeTypeChartData.map(entry => (
                                                    <Cell
                                                        key={entry.name}
                                                        fill={NODE_TYPE_COLORS[entry.name] ?? "#64748b"}
                                                        opacity={activeFilter === null || activeFilter === entry.name ? 1 : 0.2}
                                                    />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </FilterSection>

                                {/* Energy histogram */}
                                <FilterSection label="energy">
                                    <ResponsiveContainer width="100%" height={60}>
                                        <BarChart data={energyHistData}>
                                            <XAxis dataKey="range" tick={{ fontSize: 9, fill: "#64748b" }} />
                                            <YAxis hide />
                                            <Tooltip
                                                contentStyle={{ background: "#0b0f1e", border: "1px solid #334155", fontSize: 11 }}
                                                labelStyle={{ color: "#00f0ff" }}
                                            />
                                            <Bar dataKey="count" fill="#ff00ff" radius={[2, 2, 0, 0]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </FilterSection>

                                {/* Depth histogram */}
                                <FilterSection label="depth (poincaré radius)">
                                    <ResponsiveContainer width="100%" height={60}>
                                        <BarChart data={depthHistData}>
                                            <XAxis dataKey="range" tick={{ fontSize: 9, fill: "#64748b" }} />
                                            <YAxis hide />
                                            <Tooltip
                                                contentStyle={{ background: "#0b0f1e", border: "1px solid #334155", fontSize: 11 }}
                                                labelStyle={{ color: "#00f0ff" }}
                                            />
                                            <Bar dataKey="count" fill="#00f0ff" radius={[2, 2, 0, 0]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </FilterSection>

                                {/* Color legend — clickable */}
                                <FilterSection label="color legend" hint="click to filter">
                                    <div className="space-y-1">
                                        {Object.entries(NODE_TYPE_COLORS).map(([type, color]) => (
                                            <button
                                                key={type}
                                                onClick={() => setActiveFilter(prev => prev === type ? null : type)}
                                                className={`flex items-center gap-2 text-[10px] w-full px-1 py-0.5 rounded transition-colors ${
                                                    activeFilter === type
                                                        ? "bg-muted text-foreground"
                                                        : "text-muted-foreground hover:text-foreground"
                                                }`}
                                            >
                                                <div
                                                    className="w-3 h-3 rounded-sm flex-shrink-0"
                                                    style={{
                                                        background: color,
                                                        opacity: activeFilter === null || activeFilter === type ? 1 : 0.2,
                                                    }}
                                                />
                                                {type}
                                                {activeFilter === type && <span className="ml-auto text-[8px] text-primary">ACTIVE</span>}
                                            </button>
                                        ))}
                                    </div>
                                </FilterSection>
                            </div>
                        )}

                        {panelTab === "LINKS" && (
                            <div className="p-3 space-y-4">
                                <FilterSection label="edge_type">
                                    <ResponsiveContainer width="100%" height={Math.max(edgeTypeChartData.length * 18, 60)}>
                                        <BarChart data={edgeTypeChartData} layout="vertical">
                                            <XAxis type="number" hide />
                                            <YAxis type="category" dataKey="name" width={80} tick={{ fontSize: 10, fill: "#94a3b8" }} />
                                            <Tooltip
                                                contentStyle={{ background: "#0b0f1e", border: "1px solid #334155", fontSize: 11 }}
                                                labelStyle={{ color: "#00f0ff" }}
                                            />
                                            <Bar dataKey="count" fill="#00d8ff" radius={[0, 2, 2, 0]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </FilterSection>
                            </div>
                        )}
                    </div>
                )}

                {/* ── Graph canvas ──────────────────────────────────────── */}
                <div className="flex-1 relative overflow-hidden">
                    {error && (
                        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2 bg-destructive/20 text-destructive border border-destructive/30 rounded-lg px-4 py-2 text-sm">
                            <AlertCircle className="h-4 w-4 flex-shrink-0" />
                            Failed to load graph data
                        </div>
                    )}

                    {isLoading && (
                        <div className="absolute inset-0 z-10 flex items-center justify-center bg-background/40">
                            <div className="flex flex-col items-center gap-3 text-muted-foreground">
                                <RefreshCw className="h-8 w-8 animate-spin text-primary" />
                                <span className="text-sm">Loading graph…</span>
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
