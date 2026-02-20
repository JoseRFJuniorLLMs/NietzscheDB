import { useEffect, useRef, useState, useCallback } from "react"
import { Cosmograph, prepareCosmographData } from "@cosmograph/cosmograph"
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceArea,
} from "recharts"
import {
    RefreshCw, AlertCircle, Info, X,
    MousePointer2, Square, ZoomIn, ZoomOut, Maximize2, Play, Pause,
    ChevronDown, Network, SlidersHorizontal, ChevronLeft, ChevronRight,
} from "lucide-react"
import { Button } from "@/components/ui/button"

// ─── Types ────────────────────────────────────────────────────────────────────

interface CosmoNode {
    id:         string
    label:      string
    energy:     number
    node_type:  string
    color:      string
    x:          number
    y:          number
    created_at: number
    [key: string]: unknown
}

interface CosmoLink {
    source: string
    target: string
    [key: string]: unknown
}

interface GraphData {
    nodes:     CosmoNode[]
    links:     CosmoLink[]
    reachable: boolean
    error?:    string
}

type ToolMode    = "cursor" | "rect" | "lasso"
type LayoutName  = "force" | "forceatlas2" | "noverlap" | "circular" | "random"
type ConfigTab   = "POINTS" | "LINKS" | "SIMULATION"

// ─── Constants ────────────────────────────────────────────────────────────────

const NODE_TYPE_COLORS: Record<string, string> = {
    Semantic:      "#6366f1",
    Episodic:      "#06b6d4",
    Concept:       "#f59e0b",
    DreamSnapshot: "#8b5cf6",
    Pattern:       "#10b981",
    Fact:          "#f43f5e",
    Skill:         "#3b82f6",
    Emotion:       "#ec4899",
}

const DEFAULT_COLORS = ["#6366f1","#06b6d4","#f59e0b","#8b5cf6","#64748b","#10b981","#f43f5e","#3b82f6"]

const LAYOUT_LABELS: Record<LayoutName, string> = {
    force:       "Force Directed",
    forceatlas2: "ForceAtlas2",
    noverlap:    "Noverlap",
    circular:    "Circular",
    random:      "Random",
}

const LAYOUT_PARAMS: Record<string, Record<string, number>> = {
    force: {
        simulationGravity:    0.25,
        simulationRepulsion:  1.0,
        simulationLinkSpring: 1.0,
        simulationDecay:      5000,
        simulationCenter:     0,
    },
    forceatlas2: {
        simulationGravity:    0.5,
        simulationRepulsion:  2.0,
        simulationLinkSpring: 2.0,
        simulationDecay:      8000,
        simulationCenter:     0.1,
    },
    noverlap: {
        simulationGravity:    0.05,
        simulationRepulsion:  4.0,
        simulationLinkSpring: 0.3,
        simulationDecay:      3000,
        simulationCenter:     0,
    },
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function buildTimeline(nodes: CosmoNode[]) {
    const withTime = nodes.filter(n => n.created_at > 0)
    if (!withTime.length) return []
    const years   = withTime.map(n => new Date(n.created_at * 1000).getFullYear())
    const minYear = Math.min(...years)
    const maxYear = Math.max(...years)
    const span    = maxYear - minYear || 1
    const step    = span <= 30 ? 1 : span <= 100 ? 10 : Math.ceil(span / 20) * 5
    const buckets = new Map<number, number>()
    for (let y = Math.floor(minYear / step) * step; y <= maxYear + step; y += step) buckets.set(y, 0)
    for (const year of years) { const k = Math.floor(year / step) * step; buckets.set(k, (buckets.get(k) ?? 0) + 1) }
    return [...buckets.entries()].sort(([a],[b]) => a-b).map(([year,count]) => ({
        year, label: step > 1 ? `${year}s` : `${year}`, count,
    }))
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function ToolButton({ icon, active, title, onClick }: {
    icon: React.ReactNode; active?: boolean; title: string; onClick: () => void
}) {
    return (
        <button onClick={onClick} title={title}
            className={`w-8 h-8 flex items-center justify-center rounded-lg transition-colors ${
                active
                    ? "bg-primary/20 text-primary border border-primary/30"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent/30"
            }`}
        >{icon}</button>
    )
}

const LassoIcon = () => (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none"
        stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <path d="M7 10C7 7.79 9.24 6 12 6s5 1.79 5 4-2.24 4-5 4c-1.08 0-2.08-.27-2.9-.74"/>
        <path d="M9.1 13.26C8.4 14.01 8 14.97 8 16c0 2.21 1.79 4 4 4s4-1.79 4-4-1.79-4-4-4"/>
        <circle cx="12" cy="21" r="1" fill="currentColor" stroke="none"/>
    </svg>
)

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
    return (
        <button
            onClick={() => onChange(!checked)}
            className={`relative w-9 h-5 rounded-full transition-colors ${checked ? "bg-primary" : "bg-muted"}`}
        >
            <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${checked ? "translate-x-4" : ""}`} />
        </button>
    )
}

function SliderRow({ label, value, min, max, step, onChange, format }: {
    label: string; value: number; min: number; max: number; step: number
    onChange: (v: number) => void; format?: (v: number) => string
}) {
    return (
        <div className="space-y-1">
            <div className="flex justify-between text-[11px]">
                <span className="text-muted-foreground">{label}</span>
                <span className="text-foreground/70 tabular-nums font-mono">{format ? format(value) : value}</span>
            </div>
            <input type="range" min={min} max={max} step={step} value={value}
                onChange={e => onChange(Number(e.target.value))}
                className="w-full accent-primary h-1"
            />
            <div className="flex justify-between text-[9px] text-muted-foreground/30">
                <span>{min}</span><span>{max}</span>
            </div>
        </div>
    )
}

function SectionTitle({ children }: { children: React.ReactNode }) {
    return <div className="text-[9px] uppercase tracking-widest text-muted-foreground/40 font-semibold pt-2 pb-1">{children}</div>
}

function Row({ label, value }: { label: string; value: string }) {
    return (
        <div className="flex justify-between gap-2">
            <span className="opacity-60">{label}</span>
            <span className="font-mono text-right break-all">{value}</span>
        </div>
    )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function GraphExplorerPage() {
    const containerRef  = useRef<HTMLDivElement>(null)
    const cosmographRef = useRef<Cosmograph | null>(null)

    // Data
    const [loading,      setLoading]      = useState(false)
    const [data,         setData]         = useState<GraphData | null>(null)
    const [rawNodes,     setRawNodes]     = useState<CosmoNode[]>([])
    const [rawLinks,     setRawLinks]     = useState<CosmoLink[]>([])
    const [timeline,     setTimeline]     = useState<ReturnType<typeof buildTimeline>>([])
    const [selectedNode, setSelectedNode] = useState<CosmoNode | null>(null)

    // Query
    const [collection, setCollection] = useState("")
    const [nodeLimit,  setNodeLimit]  = useState(500)

    // Toolbar
    const [toolMode,     setToolMode]     = useState<ToolMode>("cursor")
    const [isSimRunning, setIsSimRunning] = useState(false)

    // Layout
    const [layout,      setLayout]      = useState<LayoutName>("force")
    const [showLayouts, setShowLayouts] = useState(false)

    // Config panel
    const [configOpen,  setConfigOpen]  = useState(true)
    const [configTab,   setConfigTab]   = useState<ConfigTab>("POINTS")

    // ── Points config ──────────────────────────────────────────────────────────
    const [pointSize,           setPointSize]           = useState(3)
    const [pointSizeScale,      setPointSizeScale]      = useState(1)
    const [colorByEnergy,       setColorByEnergy]       = useState(false)
    const [showColorLegend,     setShowColorLegend]     = useState(true)
    const [showLabels,          setShowLabels]          = useState(false)
    const [labelThreshold,      setLabelThreshold]      = useState(3)

    // ── Links config ───────────────────────────────────────────────────────────
    const [linkWidth,           setLinkWidth]           = useState(0.6)
    const [linkWidthScale,      setLinkWidthScale]      = useState(1)
    const [linkArrows,          setLinkArrows]          = useState(false)
    const [scaleLinksOnZoom,    setScaleLinksOnZoom]    = useState(false)
    const [showTimeline,        setShowTimeline]        = useState(true)

    // ── Simulation config ──────────────────────────────────────────────────────
    const [simGravity,    setSimGravity]    = useState(0.25)
    const [simRepulsion,  setSimRepulsion]  = useState(1.0)
    const [simLinkSpring, setSimLinkSpring] = useState(1.0)
    const [simDecay,      setSimDecay]      = useState(5000)
    const [simCenter,     setSimCenter]     = useState(0)
    const [simFriction,   setSimFriction]   = useState(0.85)

    // Timeline brush
    const [brushStart,  setBrushStart]  = useState<number | null>(null)
    const [brushEnd,    setBrushEnd]    = useState<number | null>(null)
    const [brushActive, setBrushActive] = useState(false)

    // ── Apply config to Cosmograph ────────────────────────────────────────────

    const applyPointsConfig = useCallback(() => {
        const cg = cosmographRef.current
        if (!cg) return
        cg.setConfig({
            pointSize,
            pointSizeScale,
        } as any)
    }, [pointSize, pointSizeScale])

    const applyLinksConfig = useCallback(() => {
        const cg = cosmographRef.current
        if (!cg) return
        cg.setConfig({
            linkWidth: linkWidth * linkWidthScale,
            linkArrows,
        } as any)
    }, [linkWidth, linkWidthScale, linkArrows])

    const applySimConfig = useCallback(() => {
        const cg = cosmographRef.current
        if (!cg) return
        cg.setConfig({
            simulationGravity:    simGravity,
            simulationRepulsion:  simRepulsion,
            simulationLinkSpring: simLinkSpring,
            simulationDecay:      simDecay,
            simulationCenter:     simCenter,
            simulationFriction:   simFriction,
        } as any)
    }, [simGravity, simRepulsion, simLinkSpring, simDecay, simCenter, simFriction])

    useEffect(() => { applyPointsConfig() }, [applyPointsConfig])
    useEffect(() => { applyLinksConfig()  }, [applyLinksConfig])
    useEffect(() => { applySimConfig()    }, [applySimConfig])

    // ── Toolbar handlers ──────────────────────────────────────────────────────

    const setTool = useCallback((mode: ToolMode) => {
        const cg = cosmographRef.current
        if (!cg) return
        cg.deactivateRectSelection()
        cg.deactivatePolygonalSelection()
        if (mode !== "cursor") cg.unselectAllPoints()
        if (mode === "rect")   cg.activateRectSelection()
        if (mode === "lasso")  cg.activatePolygonalSelection()
        setToolMode(mode)
    }, [])

    const handleZoomIn  = useCallback(() => {
        const cg = cosmographRef.current; if (!cg) return
        cg.setZoomLevel((cg.getZoomLevel() ?? 1) * 1.5, 300)
    }, [])

    const handleZoomOut = useCallback(() => {
        const cg = cosmographRef.current; if (!cg) return
        cg.setZoomLevel((cg.getZoomLevel() ?? 1) / 1.5, 300)
    }, [])

    const handleFitView = useCallback(() => { cosmographRef.current?.fitView(600) }, [])

    const toggleSim = useCallback(() => {
        const cg = cosmographRef.current; if (!cg) return
        if (cg.isSimulationRunning) { cg.pause();   setIsSimRunning(false) }
        else                        { cg.unpause(); setIsSimRunning(true)  }
    }, [])

    // ── Layout ────────────────────────────────────────────────────────────────

    const applyLayout = useCallback(async (name: LayoutName) => {
        const cg = cosmographRef.current; if (!cg) return
        setLayout(name); setShowLayouts(false)

        if (name === "circular" || name === "random") {
            if (!rawNodes.length) return
            const n = rawNodes.length
            const modified = rawNodes.map((node, i) => ({
                ...node,
                _lx: name === "circular" ? Math.cos((2 * Math.PI * i) / n) : (Math.random() - 0.5) * 2,
                _ly: name === "circular" ? Math.sin((2 * Math.PI * i) / n) : (Math.random() - 0.5) * 2,
            }))
            const result = await prepareCosmographData(
                { points: { pointIdBy: "id", pointXBy: "_lx", pointYBy: "_ly" }, links: { linkSourceBy: "source", linkTargetsBy: ["target"] } },
                modified, rawLinks,
            )
            if (!result) return
            const { points, links, cosmographConfig } = result
            await cg.setConfig({ points, links, ...cosmographConfig, backgroundColor: "#0b0d12", pointSize, linkWidth, linkArrows, enableSimulation: false } as any)
            cg.fitView(600); setIsSimRunning(false)
        } else {
            const params = LAYOUT_PARAMS[name]
            await cg.setConfig({ ...params, enableSimulation: true } as any)
            cg.start(1); setIsSimRunning(true)
        }
    }, [rawNodes, rawLinks, pointSize, linkWidth, linkArrows])

    // ── Load graph ────────────────────────────────────────────────────────────

    const loadGraph = useCallback(async () => {
        if (!containerRef.current) return
        setLoading(true); setSelectedNode(null)
        try {
            const qs = new URLSearchParams({ collection, node_limit: String(nodeLimit), edge_limit: "5000" })
            const resp = await fetch(`/api/nietzsche/graph?${qs}`)
            const json: GraphData = await resp.json()
            setData(json)
            const nodes = json.nodes ?? []; const links = json.links ?? []
            setRawNodes(nodes); setRawLinks(links); setTimeline(buildTimeline(nodes))
            setBrushStart(null); setBrushEnd(null)
            if (!json.reachable || !containerRef.current) return

            const result = await prepareCosmographData(
                { points: { pointIdBy: "id" }, links: { linkSourceBy: "source", linkTargetsBy: ["target"] } },
                nodes, links,
            )
            if (!result || !containerRef.current) return
            const { points, links: prepLinks, cosmographConfig } = result
            const simParams = LAYOUT_PARAMS[layout] ?? LAYOUT_PARAMS.force

            if (cosmographRef.current) {
                await cosmographRef.current.setConfig({
                    points, links: prepLinks, ...cosmographConfig, ...simParams,
                    backgroundColor: "#0b0d12", pointSize, linkWidth, linkArrows,
                } as any)
            } else {
                cosmographRef.current = new Cosmograph(containerRef.current, {
                    points, links: prepLinks, ...cosmographConfig, ...simParams,
                    backgroundColor: "#0b0d12", pointSize, linkWidth, linkArrows,
                    enableSimulation: true,
                    onSimulationStart:  () => setIsSimRunning(true),
                    onSimulationEnd:    () => setIsSimRunning(false),
                    onSimulationPause:  () => setIsSimRunning(false),
                    onPointClick: (point: any) => {
                        const node = nodes.find(n => n.id === point?.id) ?? null
                        setSelectedNode(node)
                    },
                } as any)
                setIsSimRunning(true)
            }
        } catch (err) {
            console.error("Graph load failed:", err)
            setData({ nodes: [], links: [], reachable: false, error: String(err) })
        } finally { setLoading(false) }
    }, [collection, nodeLimit, layout, pointSize, linkWidth, linkArrows])

    useEffect(() => {
        loadGraph()
        return () => { cosmographRef.current?.destroy(); cosmographRef.current = null }
    }, []) // eslint-disable-line react-hooks/exhaustive-deps

    // ── Timeline brush ────────────────────────────────────────────────────────

    const onChartMouseDown = useCallback((e: any) => {
        const year = e?.activePayload?.[0]?.payload?.year
        if (year != null) { setBrushStart(year); setBrushEnd(year); setBrushActive(true) }
    }, [])
    const onChartMouseMove = useCallback((e: any) => {
        if (!brushActive) return
        const year = e?.activePayload?.[0]?.payload?.year
        if (year != null) setBrushEnd(year)
    }, [brushActive])
    const onChartMouseUp   = useCallback(() => { setBrushActive(false) }, [])
    const clearBrush       = useCallback(() => { setBrushStart(null); setBrushEnd(null); setBrushActive(false) }, [])

    // ── Derived ───────────────────────────────────────────────────────────────

    const nodeTypes   = data ? [...new Set(data.nodes.map(n => n.node_type))].sort() : []
    const brushMin    = brushStart != null && brushEnd != null ? Math.min(brushStart, brushEnd) : null
    const brushMax    = brushStart != null && brushEnd != null ? Math.max(brushStart, brushEnd) : null
    const brushMinLbl = timeline.find(t => t.year === brushMin)?.label
    const brushMaxLbl = timeline.find(t => t.year === brushMax)?.label

    // ── Render ────────────────────────────────────────────────────────────────

    return (
        <div className="flex flex-col gap-0 -mx-6 -my-6 md:-mx-8 md:-my-8 h-[calc(100vh-0px)]">

            {/* ── Top bar ─────────────────────────────────────────────────────── */}
            <div className="flex flex-wrap items-center gap-3 px-5 py-3 border-b bg-card/80 backdrop-blur shrink-0">
                <span className="font-semibold text-sm text-foreground">Graph Explorer</span>

                <input
                    value={collection} onChange={e => setCollection(e.target.value)}
                    placeholder="collection (default)"
                    className="h-7 text-xs rounded border border-border bg-background px-2 w-36 focus:outline-none focus:ring-1 focus:ring-primary"
                />

                <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">nodes</span>
                    <input type="range" min={50} max={2000} step={50} value={nodeLimit}
                        onChange={e => setNodeLimit(Number(e.target.value))}
                        className="w-24 accent-primary"
                    />
                    <span className="text-xs tabular-nums w-10">{nodeLimit}</span>
                </div>

                <Button size="sm" variant="outline" onClick={loadGraph} disabled={loading} className="h-7 text-xs gap-1">
                    <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
                    {loading ? "Loading…" : "Refresh"}
                </Button>

                {/* Config toggle */}
                <button
                    onClick={() => setConfigOpen(v => !v)}
                    className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs border transition-colors ${
                        configOpen
                            ? "bg-primary/10 text-primary border-primary/30"
                            : "text-muted-foreground border-border/50 hover:bg-accent/20"
                    }`}
                >
                    <SlidersHorizontal className="h-3 w-3" />
                    Configure
                </button>

                {data && (
                    <div className="flex items-center gap-3 ml-auto text-xs text-muted-foreground">
                        <span>{data.nodes.length.toLocaleString()} nodes</span>
                        <span>{data.links.length.toLocaleString()} links</span>
                        {showColorLegend && nodeTypes.map((t, i) => (
                            <span key={t} className="flex items-center gap-1">
                                <span className="inline-block w-2 h-2 rounded-full"
                                    style={{ background: NODE_TYPE_COLORS[t] ?? DEFAULT_COLORS[i % 8] }} />
                                {t}
                            </span>
                        ))}
                    </div>
                )}
            </div>

            {/* ── Error banner ────────────────────────────────────────────────── */}
            {data && !data.reachable && (
                <div className="flex items-center gap-2 px-5 py-2.5 bg-destructive/10 text-destructive text-sm border-b border-destructive/20 shrink-0">
                    <AlertCircle className="h-4 w-4 shrink-0" />
                    <span>
                        NietzscheDB not reachable.
                        {data.error && <span className="ml-1 opacity-70 font-mono text-xs">{data.error}</span>}
                        {" "}Set <code className="bg-destructive/10 px-1 rounded">NIETZSCHE_ADDR</code> env var.
                    </span>
                </div>
            )}

            {/* ── Main area ───────────────────────────────────────────────────── */}
            <div className="flex flex-1 min-h-0 overflow-hidden relative">

                {/* ══ LEFT CONFIG PANEL (Cosmograph 2.0 style) ══════════════════ */}
                {configOpen && (
                    <aside
                        className="w-64 border-r flex flex-col overflow-hidden shrink-0 z-20"
                        style={{ background: "rgba(11,13,18,0.97)" }}
                    >
                        {/* Panel header */}
                        <div className="flex items-center justify-between px-4 py-3 border-b border-border/30">
                            <span className="text-[10px] uppercase tracking-widest text-muted-foreground/60 font-semibold">
                                Graph Configuration
                            </span>
                            <button onClick={() => setConfigOpen(false)} className="text-muted-foreground/40 hover:text-muted-foreground">
                                <ChevronLeft className="h-4 w-4" />
                            </button>
                        </div>

                        {/* Tabs */}
                        <div className="flex border-b border-border/30">
                            {(["POINTS","LINKS","SIMULATION"] as ConfigTab[]).map(tab => (
                                <button
                                    key={tab}
                                    onClick={() => setConfigTab(tab)}
                                    className={`flex-1 py-2 text-[10px] font-semibold tracking-wider transition-colors ${
                                        configTab === tab
                                            ? "text-foreground border-b-2 border-primary"
                                            : "text-muted-foreground/40 hover:text-muted-foreground/70"
                                    }`}
                                >
                                    {tab}
                                </button>
                            ))}
                        </div>

                        {/* Tab content */}
                        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">

                            {/* ── POINTS tab ─────────────────────────────────── */}
                            {configTab === "POINTS" && (
                                <>
                                    <SectionTitle>Display</SectionTitle>

                                    <div className="flex items-center justify-between text-[11px]">
                                        <span className="text-muted-foreground">show color legend</span>
                                        <Toggle checked={showColorLegend} onChange={setShowColorLegend} />
                                    </div>

                                    <div className="flex items-center justify-between text-[11px]">
                                        <span className="text-muted-foreground">show labels</span>
                                        <Toggle checked={showLabels} onChange={v => {
                                            setShowLabels(v)
                                            cosmographRef.current?.setConfig({ showLabels: v } as any)
                                        }} />
                                    </div>

                                    {showLabels && (
                                        <SliderRow label="label threshold" value={labelThreshold} min={1} max={10} step={0.5}
                                            onChange={v => {
                                                setLabelThreshold(v)
                                                cosmographRef.current?.setConfig({ pointLabelWeight: v } as any)
                                            }}
                                            format={v => v.toFixed(1)}
                                        />
                                    )}

                                    <SectionTitle>Size</SectionTitle>

                                    <SliderRow label="point size" value={pointSize} min={0.5} max={20} step={0.5}
                                        onChange={v => {
                                            setPointSize(v)
                                            cosmographRef.current?.setConfig({ pointSize: v } as any)
                                        }}
                                        format={v => v.toFixed(1)}
                                    />

                                    <SliderRow label="size scale" value={pointSizeScale} min={0.1} max={5} step={0.1}
                                        onChange={v => {
                                            setPointSizeScale(v)
                                            cosmographRef.current?.setConfig({ pointSizeScale: v } as any)
                                        }}
                                        format={v => v.toFixed(2)}
                                    />

                                    <SectionTitle>Color</SectionTitle>

                                    <div className="flex items-center justify-between text-[11px]">
                                        <span className="text-muted-foreground">color by energy</span>
                                        <Toggle checked={colorByEnergy} onChange={setColorByEnergy} />
                                    </div>

                                    {showColorLegend && nodeTypes.length > 0 && (
                                        <div className="space-y-1.5 pt-1">
                                            {nodeTypes.map((t, i) => (
                                                <div key={t} className="flex items-center gap-2 text-[11px]">
                                                    <span className="w-2.5 h-2.5 rounded-full shrink-0"
                                                        style={{ background: NODE_TYPE_COLORS[t] ?? DEFAULT_COLORS[i % 8] }} />
                                                    <span className="text-muted-foreground">{t}</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </>
                            )}

                            {/* ── LINKS tab ──────────────────────────────────── */}
                            {configTab === "LINKS" && (
                                <>
                                    <SectionTitle>Width</SectionTitle>

                                    <SliderRow label="link width" value={linkWidth} min={0.1} max={5} step={0.1}
                                        onChange={v => {
                                            setLinkWidth(v)
                                            cosmographRef.current?.setConfig({ linkWidth: v * linkWidthScale } as any)
                                        }}
                                        format={v => v.toFixed(2)}
                                    />

                                    <SliderRow label="width scale" value={linkWidthScale} min={0.1} max={5} step={0.1}
                                        onChange={v => {
                                            setLinkWidthScale(v)
                                            cosmographRef.current?.setConfig({ linkWidth: linkWidth * v } as any)
                                        }}
                                        format={v => v.toFixed(2)}
                                    />

                                    <div className="flex items-center justify-between text-[11px]">
                                        <span className="text-muted-foreground">scale links on zoom</span>
                                        <Toggle checked={scaleLinksOnZoom} onChange={v => {
                                            setScaleLinksOnZoom(v)
                                            cosmographRef.current?.setConfig({ scaleNodesOnZoom: v } as any)
                                        }} />
                                    </div>

                                    <SectionTitle>Arrows</SectionTitle>

                                    <div className="flex items-center justify-between text-[11px]">
                                        <span className="text-muted-foreground">show link arrows</span>
                                        <Toggle checked={linkArrows} onChange={v => {
                                            setLinkArrows(v)
                                            cosmographRef.current?.setConfig({ linkArrows: v } as any)
                                        }} />
                                    </div>

                                    <SectionTitle>Timeline</SectionTitle>

                                    <div className="flex items-center justify-between text-[11px]">
                                        <span className="text-muted-foreground">show links timeline</span>
                                        <Toggle checked={showTimeline} onChange={setShowTimeline} />
                                    </div>

                                    <div className="text-[10px] text-muted-foreground/40 leading-relaxed">
                                        time data column: <span className="text-muted-foreground/60 font-mono">created_at</span>
                                    </div>
                                </>
                            )}

                            {/* ── SIMULATION tab ─────────────────────────────── */}
                            {configTab === "SIMULATION" && (
                                <>
                                    <SectionTitle>Force</SectionTitle>

                                    <SliderRow label="gravity" value={simGravity} min={0} max={2} step={0.01}
                                        onChange={v => { setSimGravity(v); cosmographRef.current?.setConfig({ simulationGravity: v } as any) }}
                                        format={v => v.toFixed(2)}
                                    />

                                    <SliderRow label="repulsion" value={simRepulsion} min={0} max={5} step={0.05}
                                        onChange={v => { setSimRepulsion(v); cosmographRef.current?.setConfig({ simulationRepulsion: v } as any) }}
                                        format={v => v.toFixed(2)}
                                    />

                                    <SliderRow label="link spring" value={simLinkSpring} min={0} max={5} step={0.05}
                                        onChange={v => { setSimLinkSpring(v); cosmographRef.current?.setConfig({ simulationLinkSpring: v } as any) }}
                                        format={v => v.toFixed(2)}
                                    />

                                    <SectionTitle>Decay</SectionTitle>

                                    <SliderRow label="decay" value={simDecay} min={100} max={20000} step={100}
                                        onChange={v => { setSimDecay(v); cosmographRef.current?.setConfig({ simulationDecay: v } as any) }}
                                        format={v => v.toLocaleString()}
                                    />

                                    <SliderRow label="friction" value={simFriction} min={0.1} max={0.99} step={0.01}
                                        onChange={v => { setSimFriction(v); cosmographRef.current?.setConfig({ simulationFriction: v } as any) }}
                                        format={v => v.toFixed(2)}
                                    />

                                    <SliderRow label="center" value={simCenter} min={0} max={1} step={0.01}
                                        onChange={v => { setSimCenter(v); cosmographRef.current?.setConfig({ simulationCenter: v } as any) }}
                                        format={v => v.toFixed(2)}
                                    />

                                    <SectionTitle>Layout</SectionTitle>

                                    <div className="space-y-1">
                                        {(Object.entries(LAYOUT_LABELS) as [LayoutName, string][]).map(([name, label]) => (
                                            <button key={name} onClick={() => applyLayout(name)}
                                                className={`w-full text-left px-2.5 py-1.5 rounded text-[11px] transition-colors ${
                                                    layout === name
                                                        ? "bg-primary/15 text-primary"
                                                        : "text-muted-foreground hover:bg-accent/20 hover:text-foreground"
                                                }`}
                                            >
                                                {label}
                                            </button>
                                        ))}
                                    </div>

                                    <div className="pt-2 flex gap-2">
                                        <Button size="sm" variant="outline" className="flex-1 h-7 text-xs"
                                            onClick={() => { cosmographRef.current?.start(1); setIsSimRunning(true) }}>
                                            <Play className="h-3 w-3 mr-1" /> Restart
                                        </Button>
                                        <Button size="sm" variant="outline" className="flex-1 h-7 text-xs"
                                            onClick={() => { cosmographRef.current?.pause(); setIsSimRunning(false) }}>
                                            <Pause className="h-3 w-3 mr-1" /> Pause
                                        </Button>
                                    </div>
                                </>
                            )}
                        </div>
                    </aside>
                )}

                {/* Config collapsed toggle */}
                {!configOpen && (
                    <button
                        onClick={() => setConfigOpen(true)}
                        className="absolute left-0 top-1/2 -translate-y-1/2 z-20 flex items-center justify-center w-5 h-16 rounded-r-lg bg-card/90 border border-l-0 border-border/50 text-muted-foreground hover:text-foreground transition-colors"
                    >
                        <ChevronRight className="h-3 w-3" />
                    </button>
                )}

                {/* ── Left Toolbar ──────────────────────────────────────────── */}
                <div className={`absolute ${configOpen ? "left-[268px]" : "left-3"} top-1/2 -translate-y-1/2 z-10 flex flex-col gap-1 p-1.5 rounded-xl bg-card/90 backdrop-blur border border-border/50 shadow-lg transition-all`}>
                    <ToolButton icon={<MousePointer2 className="h-4 w-4" />} active={toolMode === "cursor"} title="Cursor" onClick={() => setTool("cursor")} />
                    <ToolButton icon={<Square className="h-4 w-4" />}        active={toolMode === "rect"}   title="Rectangle select" onClick={() => setTool("rect")} />
                    <ToolButton icon={<LassoIcon />}                          active={toolMode === "lasso"}  title="Lasso select"     onClick={() => setTool("lasso")} />
                    <div className="w-full h-px bg-border/40 my-0.5" />
                    <ToolButton icon={<ZoomIn  className="h-4 w-4" />} title="Zoom in"        onClick={handleZoomIn} />
                    <ToolButton icon={<ZoomOut className="h-4 w-4" />} title="Zoom out"       onClick={handleZoomOut} />
                    <div className="w-full h-px bg-border/40 my-0.5" />
                    <ToolButton icon={<Maximize2 className="h-4 w-4" />} title="Fit to screen"  onClick={handleFitView} />
                    <ToolButton
                        icon={isSimRunning ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                        active={isSimRunning} title={isSimRunning ? "Pause" : "Resume"}
                        onClick={toggleSim}
                    />
                </div>

                {/* ── Layout panel (top-right) ──────────────────────────────── */}
                <div className="absolute right-3 top-3 z-10">
                    <div className="relative">
                        <button
                            onClick={() => setShowLayouts(v => !v)}
                            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-card/90 backdrop-blur border border-border/50 shadow text-xs text-foreground hover:bg-accent/20 transition-colors"
                        >
                            <Network className="h-3.5 w-3.5 text-primary" />
                            <span className="font-medium">{LAYOUT_LABELS[layout]}</span>
                            <ChevronDown className={`h-3 w-3 text-muted-foreground transition-transform ${showLayouts ? "rotate-180" : ""}`} />
                        </button>
                        {showLayouts && (
                            <div className="absolute right-0 mt-1.5 w-48 rounded-xl bg-card/95 backdrop-blur border border-border/60 shadow-xl py-1.5 overflow-hidden">
                                <div className="px-3 py-1 text-[10px] uppercase tracking-widest text-muted-foreground/50 font-medium">Layout</div>
                                {(Object.entries(LAYOUT_LABELS) as [LayoutName, string][]).map(([name, label]) => (
                                    <button key={name} onClick={() => applyLayout(name)}
                                        className={`w-full text-left px-3 py-1.5 text-xs transition-colors hover:bg-accent/20 ${
                                            layout === name ? "text-primary font-medium bg-primary/10" : "text-foreground"
                                        }`}
                                    >{label}</button>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                {/* ── Color legend (bottom-left floating) ──────────────────── */}
                {showColorLegend && nodeTypes.length > 0 && (
                    <div
                        className={`absolute bottom-4 ${configOpen ? "left-[272px]" : "left-14"} z-10 px-3 py-2 rounded-xl bg-card/90 backdrop-blur border border-border/50 shadow-lg`}
                    >
                        <div className="space-y-1">
                            {nodeTypes.map((t, i) => (
                                <div key={t} className="flex items-center gap-2 text-[10px]">
                                    <span className="w-2 h-2 rounded-full shrink-0"
                                        style={{ background: NODE_TYPE_COLORS[t] ?? DEFAULT_COLORS[i % 8] }} />
                                    <span className="text-muted-foreground">{t}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* ── Cosmograph canvas ─────────────────────────────────────── */}
                <div
                    ref={containerRef}
                    className="flex-1 min-w-0"
                    style={{ background: "#0b0d12" }}
                    onClick={() => { if (showLayouts) setShowLayouts(false) }}
                />

                {/* ── Node detail panel ─────────────────────────────────────── */}
                {selectedNode && (
                    <aside className="w-64 border-l bg-card flex flex-col overflow-hidden shrink-0">
                        <div className="flex items-center justify-between px-4 py-3 border-b">
                            <div className="flex items-center gap-2">
                                <Info className="h-4 w-4 text-primary" />
                                <span className="text-sm font-semibold">Node</span>
                            </div>
                            <button onClick={() => setSelectedNode(null)} className="text-muted-foreground hover:text-foreground">
                                <X className="h-4 w-4" />
                            </button>
                        </div>
                        <div className="flex-1 overflow-auto p-4 space-y-3 text-sm">
                            <div className="flex items-start gap-2">
                                <span className="mt-1 w-3 h-3 rounded-full shrink-0" style={{ background: selectedNode.color }} />
                                <span className="font-medium break-words">{selectedNode.label}</span>
                            </div>
                            <div className="space-y-1 text-xs text-muted-foreground">
                                <Row label="Type"    value={selectedNode.node_type} />
                                <Row label="Energy"  value={selectedNode.energy.toFixed(4)} />
                                <Row label="X"       value={selectedNode.x.toFixed(6)} />
                                <Row label="Y"       value={selectedNode.y.toFixed(6)} />
                                <Row label="Created" value={
                                    selectedNode.created_at
                                        ? new Date(selectedNode.created_at * 1000).toLocaleString()
                                        : "—"
                                } />
                            </div>
                            <div className="pt-2 border-t">
                                <p className="text-xs text-muted-foreground font-mono break-all opacity-60">{selectedNode.id}</p>
                            </div>
                        </div>
                    </aside>
                )}
            </div>

            {/* ── Timeline spectrogram ──────────────────────────────────────── */}
            {showTimeline && (
                <div className="shrink-0 border-t" style={{ height: 108, background: "rgba(11,13,18,0.97)" }}>
                    <div className="flex items-center justify-between px-4 pt-2 pb-0">
                        <div className="flex items-center gap-3">
                            <span className="text-[10px] uppercase tracking-widest text-muted-foreground/50 font-medium">Timeline</span>
                            {data && (
                                <span className="text-[10px] tabular-nums text-muted-foreground/35">
                                    {data.nodes.length.toLocaleString()} nodes · {data.links.length.toLocaleString()} links
                                </span>
                            )}
                            {timeline.length > 0 && (
                                <span className="text-[10px] tabular-nums text-muted-foreground/25">
                                    {timeline[0].label} — {timeline[timeline.length - 1].label}
                                </span>
                            )}
                        </div>
                        <div className="flex items-center gap-2">
                            {brushMin != null ? (
                                <>
                                    <span className="text-[10px] text-primary/60 tabular-nums">{brushMinLbl} – {brushMaxLbl}</span>
                                    <button onClick={clearBrush} className="text-[10px] text-muted-foreground/40 hover:text-muted-foreground px-1 py-0.5 rounded hover:bg-accent/20 transition-colors">✕ clear</button>
                                </>
                            ) : (
                                <span className="text-[10px] text-muted-foreground/25">drag to filter</span>
                            )}
                        </div>
                    </div>

                    {timeline.length > 0 ? (
                        <ResponsiveContainer width="100%" height={82}>
                            <BarChart data={timeline} barGap={1} barCategoryGap={2}
                                onMouseDown={onChartMouseDown} onMouseMove={onChartMouseMove} onMouseUp={onChartMouseUp}
                                style={{ cursor: "crosshair" }}>
                                <XAxis dataKey="label" tick={{ fontSize: 9, fill: "#4b5563" }}
                                    interval="preserveStartEnd" tickLine={false} axisLine={false} height={16} />
                                <YAxis hide />
                                <Tooltip cursor={false}
                                    contentStyle={{ background: "#1a1d26", border: "1px solid #2a2d3a", borderRadius: 6, fontSize: 11 }}
                                    formatter={(v: any) => [`${v} nodes`, ""]}
                                />
                                {brushMinLbl && brushMaxLbl && (
                                    <ReferenceArea x1={brushMinLbl} x2={brushMaxLbl}
                                        fill="#6366f1" fillOpacity={0.15} stroke="#6366f1" strokeOpacity={0.4} />
                                )}
                                <Bar dataKey="count" radius={[2,2,0,0]} maxBarSize={40}>
                                    {timeline.map((entry, i) => {
                                        const inBrush = brushMin != null && brushMax != null && entry.year >= brushMin && entry.year <= brushMax
                                        return <Cell key={i} fill={inBrush ? "#818cf8" : "#1e2535"} opacity={inBrush ? 1 : 0.85} />
                                    })}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="flex items-center justify-center" style={{ height: 82 }}>
                            <span className="text-xs text-muted-foreground/30">
                                {loading ? "loading…" : "no timestamp data — populate NietzscheDB first"}
                            </span>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
