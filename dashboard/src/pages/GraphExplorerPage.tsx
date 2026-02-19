import { useEffect, useRef, useState, useCallback } from "react"
import { Cosmograph, prepareCosmographData } from "@cosmograph/cosmograph"
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from "recharts"
import { RefreshCw, AlertCircle, Info, X } from "lucide-react"
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
}

interface CosmoLink {
    source: string
    target: string
}

interface GraphData {
    nodes:     CosmoNode[]
    links:     CosmoLink[]
    reachable: boolean
    error?:    string
}

// ─── Constants ────────────────────────────────────────────────────────────────

const NODE_TYPE_COLORS: Record<string, string> = {
    Semantic:      "#6366f1",
    Episodic:      "#06b6d4",
    Concept:       "#f59e0b",
    DreamSnapshot: "#8b5cf6",
}

const DEFAULT_COLORS = ["#6366f1", "#06b6d4", "#f59e0b", "#8b5cf6", "#64748b"]

// Build a spectrogram-style histogram of energy values (0→1 in 20 buckets)
function buildEnergyHistogram(nodes: CosmoNode[]) {
    const BUCKETS = 20
    const counts = Array(BUCKETS).fill(0)
    for (const n of nodes) {
        const idx = Math.min(Math.floor(n.energy * BUCKETS), BUCKETS - 1)
        counts[idx]++
    }
    return counts.map((count, i) => ({
        energy: ((i + 0.5) / BUCKETS).toFixed(2),
        count,
        color: `hsl(${260 + i * 5}, 80%, ${40 + count * 3}%)`,
    }))
}

// ─── Component ────────────────────────────────────────────────────────────────

export function GraphExplorerPage() {
    const containerRef   = useRef<HTMLDivElement>(null)
    const cosmographRef  = useRef<Cosmograph | null>(null)

    const [loading,      setLoading]      = useState(false)
    const [data,         setData]         = useState<GraphData | null>(null)
    const [selectedNode, setSelectedNode] = useState<CosmoNode | null>(null)
    const [collection,   setCollection]   = useState("")
    const [nodeLimit,    setNodeLimit]    = useState(500)
    const [histo,        setHisto]        = useState<ReturnType<typeof buildEnergyHistogram>>([])

    // ── Load graph data ───────────────────────────────────────────────────────

    const loadGraph = useCallback(async () => {
        if (!containerRef.current) return
        setLoading(true)
        setSelectedNode(null)

        try {
            const qs = new URLSearchParams({
                collection: collection,
                node_limit: String(nodeLimit),
                edge_limit: "5000",
            })
            const resp = await fetch(`/api/nietzsche/graph?${qs}`)
            const json: GraphData = await resp.json()
            setData(json)
            setHisto(buildEnergyHistogram(json.nodes ?? []))

            if (!json.reachable || !containerRef.current) return

            const rawPoints = json.nodes
            const rawLinks  = json.links

            const dataConfig = {
                points: { pointIdBy: "id" },
                links:  { linkSourceBy: "source", linkTargetsBy: ["target"] },
            }

            const result = await prepareCosmographData(dataConfig, rawPoints, rawLinks)
            if (!result || !containerRef.current) return

            const { points, links, cosmographConfig } = result

            if (cosmographRef.current) {
                cosmographRef.current.setConfig({ points, links, ...cosmographConfig })
            } else {
                cosmographRef.current = new Cosmograph(containerRef.current, {
                    points,
                    links,
                    ...cosmographConfig,
                    backgroundColor: "#0b0d12",
                    // Size nodes by energy field (0.5–5 px)
                    pointSize: 3,
                    linkWidth: 0.6,
                    linkArrows: false,
                    onPointClick: (point: any) => {
                        const node = json.nodes.find(n => n.id === point?.id) ?? null
                        setSelectedNode(node)
                    },
                })
            }
        } catch (err) {
            console.error("Graph load failed:", err)
            setData({ nodes: [], links: [], reachable: false, error: String(err) })
        } finally {
            setLoading(false)
        }
    }, [collection, nodeLimit])

    // Initial load
    useEffect(() => {
        loadGraph()
        return () => {
            cosmographRef.current?.destroy()
            cosmographRef.current = null
        }
    }, []) // eslint-disable-line react-hooks/exhaustive-deps

    const nodeTypes = data
        ? [...new Set(data.nodes.map(n => n.node_type))].sort()
        : []

    // ── Render ────────────────────────────────────────────────────────────────

    return (
        <div className="flex flex-col gap-0 -mx-6 -my-6 md:-mx-8 md:-my-8 h-[calc(100vh-0px)]">

            {/* ── Controls bar ──────────────────────────────────────────────── */}
            <div className="flex flex-wrap items-center gap-3 px-5 py-3 border-b bg-card/80 backdrop-blur shrink-0">
                <span className="font-semibold text-sm text-foreground">Graph Explorer</span>

                <input
                    value={collection}
                    onChange={e => setCollection(e.target.value)}
                    placeholder="collection (default)"
                    className="h-7 text-xs rounded border border-border bg-background px-2 w-36 focus:outline-none focus:ring-1 focus:ring-primary"
                />

                <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">nodes</span>
                    <input
                        type="range"
                        min={50} max={2000} step={50}
                        value={nodeLimit}
                        onChange={e => setNodeLimit(Number(e.target.value))}
                        className="w-24 accent-primary"
                    />
                    <span className="text-xs tabular-nums w-10">{nodeLimit}</span>
                </div>

                <Button
                    size="sm" variant="outline"
                    onClick={loadGraph}
                    disabled={loading}
                    className="h-7 text-xs gap-1"
                >
                    <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
                    {loading ? "Loading…" : "Refresh"}
                </Button>

                {/* Stats */}
                {data && (
                    <div className="flex items-center gap-3 ml-auto text-xs text-muted-foreground">
                        <span>{data.nodes.length.toLocaleString()} nodes</span>
                        <span>{data.links.length.toLocaleString()} links</span>
                        {/* Legend chips */}
                        {nodeTypes.map((t, i) => (
                            <span key={t} className="flex items-center gap-1">
                                <span
                                    className="inline-block w-2 h-2 rounded-full"
                                    style={{ background: NODE_TYPE_COLORS[t] ?? DEFAULT_COLORS[i % 5] }}
                                />
                                {t}
                            </span>
                        ))}
                    </div>
                )}
            </div>

            {/* ── Error / not reachable banner ──────────────────────────────── */}
            {data && !data.reachable && (
                <div className="flex items-center gap-2 px-5 py-2.5 bg-destructive/10 text-destructive text-sm border-b border-destructive/20 shrink-0">
                    <AlertCircle className="h-4 w-4 shrink-0" />
                    <span>
                        NietzscheDB not reachable.
                        {data.error && <span className="ml-1 opacity-70 font-mono text-xs">{data.error}</span>}
                        {" "}Set <code className="bg-destructive/10 px-1 rounded">NIETZSCHE_ADDR</code> env var on the server.
                    </span>
                </div>
            )}

            {/* ── Main canvas + detail panel ────────────────────────────────── */}
            <div className="flex flex-1 min-h-0 overflow-hidden">
                {/* Cosmograph canvas */}
                <div
                    ref={containerRef}
                    className="flex-1 min-w-0"
                    style={{ background: "#0b0d12" }}
                />

                {/* Node detail side panel */}
                {selectedNode && (
                    <aside className="w-64 border-l bg-card flex flex-col overflow-hidden shrink-0">
                        <div className="flex items-center justify-between px-4 py-3 border-b">
                            <div className="flex items-center gap-2">
                                <Info className="h-4 w-4 text-primary" />
                                <span className="text-sm font-semibold">Node</span>
                            </div>
                            <button
                                onClick={() => setSelectedNode(null)}
                                className="text-muted-foreground hover:text-foreground"
                            >
                                <X className="h-4 w-4" />
                            </button>
                        </div>

                        <div className="flex-1 overflow-auto p-4 space-y-3 text-sm">
                            {/* Color swatch + label */}
                            <div className="flex items-start gap-2">
                                <span
                                    className="mt-1 w-3 h-3 rounded-full shrink-0"
                                    style={{ background: selectedNode.color }}
                                />
                                <span className="font-medium break-words">{selectedNode.label}</span>
                            </div>

                            <div className="space-y-1 text-xs text-muted-foreground">
                                <Row label="Type"       value={selectedNode.node_type} />
                                <Row label="Energy"     value={selectedNode.energy.toFixed(4)} />
                                <Row label="X"          value={selectedNode.x.toFixed(6)} />
                                <Row label="Y"          value={selectedNode.y.toFixed(6)} />
                                <Row label="Created"    value={
                                    selectedNode.created_at
                                        ? new Date(selectedNode.created_at * 1000).toLocaleString()
                                        : "—"
                                } />
                            </div>

                            <div className="pt-2 border-t">
                                <p className="text-xs text-muted-foreground font-mono break-all opacity-60">
                                    {selectedNode.id}
                                </p>
                            </div>
                        </div>
                    </aside>
                )}
            </div>

            {/* ── Energy spectrogram ────────────────────────────────────────── */}
            <div className="shrink-0 border-t bg-card/60" style={{ height: 96 }}>
                <div className="flex items-center justify-between px-4 pt-1.5 pb-0">
                    <span className="text-[10px] uppercase tracking-widest text-muted-foreground/60">
                        Energy spectrum  ·  {data?.nodes.length ?? 0} nodes
                    </span>
                    <span className="text-[10px] text-muted-foreground/40">0 ──────────── energy ──────────── 1</span>
                </div>
                {histo.length > 0 ? (
                    <ResponsiveContainer width="100%" height={72}>
                        <BarChart data={histo} barGap={1} barCategoryGap={1}>
                            <XAxis dataKey="energy" hide />
                            <YAxis hide />
                            <Tooltip
                                cursor={false}
                                contentStyle={{
                                    background: "#1a1d26",
                                    border: "1px solid #2a2d3a",
                                    borderRadius: 6,
                                    fontSize: 11,
                                }}
                                formatter={(v: any) => [`${v} nodes`, "count"]}
                                labelFormatter={(l: any) => `energy ≈ ${l}`}
                            />
                            <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                                {histo.map((entry, i) => (
                                    <Cell key={i} fill={entry.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="flex items-center justify-center h-16 text-xs text-muted-foreground/40">
                        {loading ? "loading graph…" : "no data"}
                    </div>
                )}
            </div>
        </div>
    )
}

function Row({ label, value }: { label: string; value: string }) {
    return (
        <div className="flex justify-between gap-2">
            <span className="opacity-60">{label}</span>
            <span className="font-mono text-right break-all">{value}</span>
        </div>
    )
}
