import { useEffect, useRef, useState, useCallback } from "react"
import {
    Cosmograph,
    CosmographTimeline,
    CosmographHistogram,
    CosmographBars,
    CosmographSearch,
    CosmographTypeColorLegend,
    CosmographButtonFitView,
    CosmographButtonPlayPause,
    CosmographButtonZoomInOut,
    CosmographButtonRectangularSelection,
    CosmographButtonPolygonalSelection,
    CosmographPointSizeStrategy,
    CosmographLinkWidthStrategy,
} from "@cosmograph/cosmograph"
import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { ChevronLeft, ChevronRight, RefreshCw, AlertCircle, Filter } from "lucide-react"

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
    Semantic:      "#6366f1",
    Episodic:      "#06b6d4",
    Concept:       "#f59e0b",
    DreamSnapshot: "#8b5cf6",
    Somatic:       "#22c55e",
    Linguistic:    "#f43f5e",
    Composite:     "#a78bfa",
}

const DEFAULT_COLLECTION = "eva_core"

// ─── GraphExplorerPage ─────────────────────────────────────────────────────────

export function GraphExplorerPage() {
    // ── Container refs ────────────────────────────────────────────────────────
    const graphContainerRef   = useRef<HTMLDivElement>(null)
    const timelineContainerRef = useRef<HTMLDivElement>(null)
    const histEnergyRef       = useRef<HTMLDivElement>(null)
    const histDepthRef        = useRef<HTMLDivElement>(null)
    const histHausdorffRef    = useRef<HTMLDivElement>(null)
    const barsTypeRef         = useRef<HTMLDivElement>(null)
    const barsEdgeRef         = useRef<HTMLDivElement>(null)
    const searchRef           = useRef<HTMLDivElement>(null)
    const legendColorRef      = useRef<HTMLDivElement>(null)
    const btnFitRef           = useRef<HTMLDivElement>(null)
    const btnPlayRef          = useRef<HTMLDivElement>(null)
    const btnZoomRef          = useRef<HTMLDivElement>(null)
    const btnRectRef          = useRef<HTMLDivElement>(null)
    const btnPolyRef          = useRef<HTMLDivElement>(null)

    // ── Cosmograph instance refs ──────────────────────────────────────────────
    const cosmoRef        = useRef<Cosmograph | null>(null)
    const timelineRef     = useRef<CosmographTimeline | null>(null)
    const histEnergyInst  = useRef<CosmographHistogram | null>(null)
    const histDepthInst   = useRef<CosmographHistogram | null>(null)
    const histHausInst    = useRef<CosmographHistogram | null>(null)
    const barsTypeInst    = useRef<CosmographBars | null>(null)
    const barsEdgeInst    = useRef<CosmographBars | null>(null)
    const searchInst      = useRef<CosmographSearch | null>(null)
    const legendColorInst = useRef<CosmographTypeColorLegend | null>(null)
    const btnFitInst      = useRef<CosmographButtonFitView | null>(null)
    const btnPlayInst     = useRef<CosmographButtonPlayPause | null>(null)
    const btnZoomInst     = useRef<CosmographButtonZoomInOut | null>(null)
    const btnRectInst     = useRef<CosmographButtonRectangularSelection | null>(null)
    const btnPolyInst     = useRef<CosmographButtonPolygonalSelection | null>(null)

    // ── State ─────────────────────────────────────────────────────────────────
    const [collection, setCollection]   = useState(DEFAULT_COLLECTION)
    const [limit, setLimit]             = useState(1000)
    const [panelOpen, setPanelOpen]     = useState(true)
    const [panelTab, setPanelTab]       = useState<"POINTS" | "LINKS">("POINTS")
    const [selectedCount, setSelectedCount] = useState(0)
    const [stats, setStats]             = useState<{ points: number; links: number } | null>(null)
    const [initialized, setInitialized] = useState(false)

    // ── Collections list ──────────────────────────────────────────────────────
    const { data: collections } = useQuery({
        queryKey: ['collections'],
        queryFn: () => api.get("/collections").then(r => r.data as { name: string; node_count: number }[]),
    })

    // ── Graph data fetch ──────────────────────────────────────────────────────
    const { data: graphData, isLoading, error, refetch } = useQuery<GraphResponse>({
        queryKey: ['graph', collection, limit],
        queryFn: () => api.get(`/graph?collection=${collection}&limit=${limit}`).then(r => r.data),
        staleTime: 30_000,
    })

    // ── Build points / links arrays for Cosmograph ────────────────────────────
    const points = graphData?.nodes?.map((n, i) => ({
        _idx:       i,
        id:         n.id,
        node_type:  n.node_type,
        energy:     n.energy,
        depth:      n.depth,
        hausdorff:  n.hausdorff,
        created_at: n.created_at,
        label:      typeof n.content?.label === "string"
                        ? n.content.label
                        : typeof n.content?.title === "string"
                            ? n.content.title
                            : n.id.slice(0, 12),
        color: NODE_TYPE_COLORS[n.node_type] ?? "#64748b",
    })) ?? []

    const links = graphData?.edges?.map(e => ({
        source:    e.from,
        target:    e.to,
        edge_type: e.edge_type,
        weight:    e.weight,
    })) ?? []

    // ── Initialize Cosmograph ─────────────────────────────────────────────────
    useEffect(() => {
        if (!graphContainerRef.current) return
        if (cosmoRef.current) { cosmoRef.current.destroy(); cosmoRef.current = null }

        const cosmo = new Cosmograph(graphContainerRef.current, {
            // ── Data mapping ────────────────────────────────────────────────
            pointIdBy:       "id",
            pointIndexBy:    "_idx",
            pointColorBy:    "color",
            pointSizeBy:     "energy",
            pointSizeRange:  [2, 14],
            pointSizeStrategy: CosmographPointSizeStrategy.Auto,
            pointLabelBy:    "label",
            pointIncludeColumns: ["*"],

            linkSourceBy:         "source",
            linkSourceIndexBy:    "source",
            linkTargetBy:         "target",
            linkTargetIndexBy:    "target",
            linkWidthBy:     "weight",
            linkWidthRange:  [0.3, 3],
            linkWidthStrategy: CosmographLinkWidthStrategy.Sum,
            linkIncludeColumns: ["*"],

            // ── Appearance ──────────────────────────────────────────────────
            backgroundColor: "#0b0f1e",
            enableSimulation: true,
            simulationGravity:    0.25,
            simulationRepulsion:  1.2,
            simulationLinkSpring: 1.0,
            simulationFriction:   0.85,
            simulationDecay:      5000,

            // ── Labels ──────────────────────────────────────────────────────
            showLabels:           true,
            showTopLabels:        true,
            showTopLabelsLimit:   20,
            showDynamicLabels:    true,
            showDynamicLabelsLimit: 30,
            showHoveredPointLabel: true,
            showFocusedPointLabel: true,
            pointLabelFontSize:   12,

            // ── Interaction ──────────────────────────────────────────────────
            selectPointOnClick:   true,
            focusPointOnClick:    true,
            resetSelectionOnEmptyCanvasClick: true,

            // ── Callbacks ────────────────────────────────────────────────────
            onGraphRebuilt: (s) => setStats({ points: s.pointsCount, links: s.linksCount }),
            onPointsFiltered: (_fp, selPts) => setSelectedCount(selPts?.length ?? 0),
        })

        cosmoRef.current = cosmo
        setInitialized(true)

        return () => {
            cosmo.destroy()
            cosmoRef.current = null
            setInitialized(false)
        }
    }, [])

    // ── Feed data into Cosmograph ─────────────────────────────────────────────
    useEffect(() => {
        if (!cosmoRef.current || !initialized) return
        cosmoRef.current.setConfig({
            pointIdBy:       "id",
            pointIndexBy:    "_idx",
            pointColorBy:    "color",
            pointSizeBy:     "energy",
            pointLabelBy:    "label",
            linkSourceBy:         "source",
            linkSourceIndexBy:    "source",
            linkTargetBy:         "target",
            linkTargetIndexBy:    "target",
            linkWidthBy:     "weight",
            points: points as any,
            links: links as any,
        })
    }, [points, links, initialized])

    // ── Initialize Components after Cosmograph is ready ───────────────────────
    useEffect(() => {
        if (!cosmoRef.current || !initialized) return
        const cosmo = cosmoRef.current

        // ── Timeline ────────────────────────────────────────────────────────
        if (timelineContainerRef.current && !timelineRef.current) {
            timelineRef.current = new CosmographTimeline(
                cosmo,
                timelineContainerRef.current,
                { accessor: "created_at", useQuantiles: true, highlightSelectedData: true }
            )
        }

        // ── Histograms (Points) ──────────────────────────────────────────────
        if (histEnergyRef.current && !histEnergyInst.current) {
            histEnergyInst.current = new CosmographHistogram(
                cosmo, histEnergyRef.current, { accessor: "energy", highlightSelectedData: true }
            )
        }
        if (histDepthRef.current && !histDepthInst.current) {
            histDepthInst.current = new CosmographHistogram(
                cosmo, histDepthRef.current, { accessor: "depth", highlightSelectedData: true }
            )
        }
        if (histHausdorffRef.current && !histHausInst.current) {
            histHausInst.current = new CosmographHistogram(
                cosmo, histHausdorffRef.current, { accessor: "hausdorff", highlightSelectedData: true }
            )
        }

        // ── Bars (categorical) ───────────────────────────────────────────────
        if (barsTypeRef.current && !barsTypeInst.current) {
            barsTypeInst.current = new CosmographBars(
                cosmo, barsTypeRef.current, { accessor: "node_type", highlightSelectedData: true, selectOnClick: true }
            )
        }
        if (barsEdgeRef.current && !barsEdgeInst.current) {
            barsEdgeInst.current = new CosmographBars(
                cosmo, barsEdgeRef.current, { accessor: "edge_type", useLinksData: true, selectOnClick: true }
            )
        }

        // ── Search ───────────────────────────────────────────────────────────
        if (searchRef.current && !searchInst.current) {
            searchInst.current = new CosmographSearch(
                cosmo, searchRef.current, { accessor: "label" }
            )
        }

        // ── Color Legend ─────────────────────────────────────────────────────
        if (legendColorRef.current && !legendColorInst.current) {
            legendColorInst.current = new CosmographTypeColorLegend(
                cosmo, legendColorRef.current, {}
            )
        }

        // ── Controls ─────────────────────────────────────────────────────────
        if (btnFitRef.current && !btnFitInst.current) {
            btnFitInst.current = new CosmographButtonFitView(cosmo, btnFitRef.current, { duration: 500 })
        }
        if (btnPlayRef.current && !btnPlayInst.current) {
            btnPlayInst.current = new CosmographButtonPlayPause(cosmo, btnPlayRef.current)
        }
        if (btnZoomRef.current && !btnZoomInst.current) {
            btnZoomInst.current = new CosmographButtonZoomInOut(cosmo, btnZoomRef.current, { zoomIncrement: 1.5 })
        }
        if (btnRectRef.current && !btnRectInst.current) {
            btnRectInst.current = new CosmographButtonRectangularSelection(cosmo, btnRectRef.current)
        }
        if (btnPolyRef.current && !btnPolyInst.current) {
            btnPolyInst.current = new CosmographButtonPolygonalSelection(cosmo, btnPolyRef.current)
        }
    }, [initialized])

    // ── Reset all filters ──────────────────────────────────────────────────────
    const resetFilters = useCallback(() => {
        timelineRef.current?.setSelection(undefined)
        histEnergyInst.current?.setSelection(undefined)
        histDepthInst.current?.setSelection(undefined)
        histHausInst.current?.setSelection(undefined)
        barsTypeInst.current?.setSelectedItem(undefined)
        barsEdgeInst.current?.setSelectedItem(undefined)
        cosmoRef.current?.unselectAllPoints()
    }, [])

    // ─────────────────────────────────────────────────────────────────────────
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
                        .filter((c: any) => c.node_count > 0 || c.name === collection)
                        .map((c: any) => (
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

                <div className="flex-1" />

                {/* Stats */}
                {stats && (
                    <span className="text-xs text-muted-foreground font-mono">
                        {stats.points.toLocaleString()} nodes · {stats.links.toLocaleString()} links
                        {selectedCount > 0 && <span className="text-primary ml-1">· {selectedCount} selected</span>}
                    </span>
                )}

                {/* Cosmograph controls */}
                <div className="flex items-center gap-1 border-l border-border/30 pl-2">
                    <div ref={btnFitRef} className="cosmo-btn" />
                    <div ref={btnPlayRef} className="cosmo-btn" />
                    <div ref={btnZoomRef} className="cosmo-btn" />
                    <div ref={btnRectRef} className="cosmo-btn" />
                    <div ref={btnPolyRef} className="cosmo-btn" />
                </div>

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

                        {/* Search */}
                        <div className="px-3 pt-3 pb-2">
                            <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-1.5">Search</div>
                            <div ref={searchRef} className="cosmo-search" />
                        </div>

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
                            <button
                                onClick={resetFilters}
                                className="text-[10px] text-muted-foreground hover:text-primary transition-colors py-1.5 px-1"
                            >
                                reset
                            </button>
                        </div>

                        {panelTab === "POINTS" && (
                            <div className="p-3 space-y-4">
                                {/* Node type bars */}
                                <FilterSection label="node_type">
                                    <div ref={barsTypeRef} className="cosmo-bars" />
                                </FilterSection>

                                {/* Energy histogram */}
                                <FilterSection label="energy">
                                    <div ref={histEnergyRef} className="cosmo-hist" />
                                </FilterSection>

                                {/* Depth histogram */}
                                <FilterSection label="depth">
                                    <div ref={histDepthRef} className="cosmo-hist" />
                                </FilterSection>

                                {/* Hausdorff histogram */}
                                <FilterSection label="hausdorff">
                                    <div ref={histHausdorffRef} className="cosmo-hist" />
                                </FilterSection>

                                {/* Color legend */}
                                <FilterSection label="color legend">
                                    <div ref={legendColorRef} className="cosmo-legend" />
                                </FilterSection>
                            </div>
                        )}

                        {panelTab === "LINKS" && (
                            <div className="p-3 space-y-4">
                                {/* Edge type bars */}
                                <FilterSection label="edge_type">
                                    <div ref={barsEdgeRef} className="cosmo-bars" />
                                </FilterSection>
                            </div>
                        )}
                    </div>
                )}

                {/* ── Graph canvas ──────────────────────────────────────── */}
                <div className="flex-1 relative overflow-hidden">
                    {/* Error */}
                    {error && (
                        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2 bg-destructive/20 text-destructive border border-destructive/30 rounded-lg px-4 py-2 text-sm">
                            <AlertCircle className="h-4 w-4 flex-shrink-0" />
                            Failed to load graph data
                        </div>
                    )}

                    {/* Loading overlay */}
                    {isLoading && (
                        <div className="absolute inset-0 z-10 flex items-center justify-center bg-background/40">
                            <div className="flex flex-col items-center gap-3 text-muted-foreground">
                                <RefreshCw className="h-8 w-8 animate-spin text-primary" />
                                <span className="text-sm">Loading graph…</span>
                            </div>
                        </div>
                    )}

                    {/* Cosmograph container */}
                    <div ref={graphContainerRef} className="w-full h-full" />
                </div>
            </div>

            {/* ── Timeline (bottom) ─────────────────────────────────────────── */}
            <div className="flex-shrink-0 border-t border-border/30 bg-card/60 backdrop-blur px-3 py-1 z-20">
                <div className="text-[10px] text-muted-foreground mb-1 flex items-center gap-2">
                    <Filter className="h-3 w-3" />
                    <span>Timeline · created_at</span>
                </div>
                <div ref={timelineContainerRef} className="cosmo-timeline" />
            </div>
        </div>
    )
}

// ─── FilterSection helper ──────────────────────────────────────────────────────

function FilterSection({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <div>
            <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-1.5">
                {label}
            </div>
            {children}
        </div>
    )
}
