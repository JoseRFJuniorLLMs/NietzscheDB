import { useState, useEffect, useRef, useCallback } from "react"
import {
    Rewind, FastForward, Play, Pause, SkipBack, SkipForward,
    Loader2, AlertCircle, Clock, Scan, Eye, ChevronDown, ChevronUp,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
    api, DEFAULT_COLLECTION, getNode,
    causalChain, type CausalEdge,
} from "@/lib/api"

// ── Types ────────────────────────────────────────────────────

interface ChainNode {
    id: string
    content: Record<string, unknown>
    node_type: string
    energy: number
    depth: number
    created_at?: string
}

interface ScrubberState {
    chain_ids: string[]
    edges: CausalEdge[]
    nodes: Map<string, ChainNode>
    currentIndex: number
}

// ── Main Page ────────────────────────────────────────────────

export default function CausalScrubberPage() {
    const [collection, setCollection] = useState(DEFAULT_COLLECTION)
    const [collections, setCollections] = useState<string[]>([])
    const [seedNodeId, setSeedNodeId] = useState("")
    const [maxDepth, setMaxDepth] = useState(20)
    const [direction, setDirection] = useState<"past" | "future">("past")
    const [scrubber, setScrubber] = useState<ScrubberState | null>(null)
    const [loading, setLoading] = useState(false)
    const [loadingNodes, setLoadingNodes] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [playing, setPlaying] = useState(false)
    const [speed, setSpeed] = useState(1000) // ms per step
    const playRef = useRef<ReturnType<typeof setInterval> | null>(null)

    useEffect(() => {
        api.get("/collections").then(r => {
            const d = r.data
            const arr = Array.isArray(d) ? d : (d?.collections ?? [])
            setCollections(arr.map((c: unknown) => typeof c === "string" ? c : (c as { name: string }).name).filter(Boolean))
        }).catch(() => {})
    }, [])

    // Fetch causal chain
    const loadChain = async () => {
        if (!seedNodeId.trim()) return
        setLoading(true)
        setError(null)
        setScrubber(null)
        setPlaying(false)
        try {
            const res = await causalChain(seedNodeId.trim(), maxDepth, direction, collection)
            if (!res.chain_ids.length) {
                setError("No causal chain found from this node. Try a different direction or node.")
                return
            }
            const state: ScrubberState = {
                chain_ids: res.chain_ids,
                edges: res.edges,
                nodes: new Map(),
                currentIndex: direction === "past" ? res.chain_ids.length - 1 : 0,
            }
            setScrubber(state)
            // Load node details in background
            loadNodeDetails(state, collection)
        } catch (e: unknown) {
            const err = e as { response?: { data?: { error?: string } }; message: string }
            setError(err.response?.data?.error || err.message)
        } finally {
            setLoading(false)
        }
    }

    // Fetch node metadata for each chain node
    const loadNodeDetails = async (state: ScrubberState, col: string) => {
        setLoadingNodes(true)
        const nodes = new Map(state.nodes)
        for (const id of state.chain_ids) {
            try {
                const data = await getNode(id, col)
                nodes.set(id, {
                    id,
                    content: data.content || {},
                    node_type: data.node_type || "Unknown",
                    energy: data.energy ?? 0,
                    depth: data.depth ?? 0,
                    created_at: data.created_at,
                })
            } catch {
                nodes.set(id, {
                    id,
                    content: {},
                    node_type: "Unknown",
                    energy: 0,
                    depth: 0,
                })
            }
        }
        setScrubber(prev => prev ? { ...prev, nodes } : null)
        setLoadingNodes(false)
    }

    // Playback controls
    const stepForward = useCallback(() => {
        setScrubber(prev => {
            if (!prev || prev.currentIndex >= prev.chain_ids.length - 1) return prev
            return { ...prev, currentIndex: prev.currentIndex + 1 }
        })
    }, [])

    const stepBack = useCallback(() => {
        setScrubber(prev => {
            if (!prev || prev.currentIndex <= 0) return prev
            return { ...prev, currentIndex: prev.currentIndex - 1 }
        })
    }, [])

    const goToStart = () => setScrubber(prev => prev ? { ...prev, currentIndex: 0 } : null)
    const goToEnd = () => setScrubber(prev => prev ? { ...prev, currentIndex: prev.chain_ids.length - 1 } : null)

    const togglePlay = () => {
        if (playing) {
            setPlaying(false)
        } else {
            setPlaying(true)
        }
    }

    // Auto-play interval
    useEffect(() => {
        if (playing && scrubber) {
            playRef.current = setInterval(() => {
                setScrubber(prev => {
                    if (!prev || prev.currentIndex >= prev.chain_ids.length - 1) {
                        setPlaying(false)
                        return prev
                    }
                    return { ...prev, currentIndex: prev.currentIndex + 1 }
                })
            }, speed)
        }
        return () => {
            if (playRef.current) clearInterval(playRef.current)
        }
    }, [playing, speed, scrubber])

    // Keyboard shortcuts
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (!scrubber) return
            if (e.key === "ArrowRight" || e.key === "l") stepForward()
            if (e.key === "ArrowLeft" || e.key === "h") stepBack()
            if (e.key === " ") { e.preventDefault(); togglePlay() }
            if (e.key === "Home") goToStart()
            if (e.key === "End") goToEnd()
        }
        window.addEventListener("keydown", handler)
        return () => window.removeEventListener("keydown", handler)
    }, [scrubber, stepForward, stepBack, playing])

    const currentNodeId = scrubber?.chain_ids[scrubber.currentIndex]
    const currentNode = currentNodeId ? scrubber?.nodes.get(currentNodeId) : null
    const currentEdge = scrubber && scrubber.currentIndex > 0
        ? scrubber.edges.find(e =>
            (e.from_node_id === scrubber.chain_ids[scrubber.currentIndex - 1] && e.to_node_id === currentNodeId) ||
            (e.to_node_id === scrubber.chain_ids[scrubber.currentIndex - 1] && e.from_node_id === currentNodeId))
        : null

    return (
        <div className="space-y-6 fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold flex items-center gap-2">
                        <Scan className="h-6 w-6" /> Causal Scrubber
                    </h1>
                    <p className="text-sm text-muted-foreground mt-1">
                        Minkowski Time Machine — scrub through causal chains, watch reasoning dissolve and reform
                    </p>
                </div>
                <div className="w-48">
                    <Select value={collection} onValueChange={setCollection}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                            {collections.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                        </SelectContent>
                    </Select>
                </div>
            </div>

            {/* Seed Node Input */}
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base">Trace Causal Chain</CardTitle>
                    <CardDescription>
                        Enter a seed node and direction to trace its causal history (past = WHY) or consequences (future = WHAT).
                        Then scrub through the chain to see each node dissolve and reform.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex flex-wrap gap-4 items-end">
                        <div className="space-y-1 flex-1 min-w-[300px]">
                            <Label>Seed Node ID</Label>
                            <Input
                                placeholder="UUID of the starting node..."
                                value={seedNodeId}
                                onChange={e => setSeedNodeId(e.target.value)}
                                className="font-mono text-xs"
                                onKeyDown={e => e.key === "Enter" && loadChain()}
                            />
                        </div>
                        <div className="space-y-1">
                            <Label>Direction</Label>
                            <Select value={direction} onValueChange={v => setDirection(v as "past" | "future")}>
                                <SelectTrigger className="w-32"><SelectValue /></SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="past">Past (WHY)</SelectItem>
                                    <SelectItem value="future">Future (WHAT)</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                        <div className="space-y-1">
                            <Label>Max Depth</Label>
                            <Input
                                type="number" value={maxDepth}
                                onChange={e => setMaxDepth(Number(e.target.value))}
                                className="w-20" min={1} max={100}
                            />
                        </div>
                        <Button onClick={loadChain} disabled={loading || !seedNodeId.trim()}>
                            {loading
                                ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Tracing...</>
                                : <><Scan className="h-4 w-4 mr-2" /> Trace</>}
                        </Button>
                    </div>
                    {error && <ErrorInline message={error} />}
                </CardContent>
            </Card>

            {/* Scrubber UI */}
            {scrubber && (
                <>
                    {/* Timeline Visualization */}
                    <Card className="overflow-hidden">
                        <CardContent className="p-0">
                            <CausalTimeline
                                scrubber={scrubber}
                                onScrub={idx => setScrubber(prev => prev ? { ...prev, currentIndex: idx } : null)}
                                direction={direction}
                            />
                        </CardContent>
                    </Card>

                    {/* Transport Controls */}
                    <div className="flex items-center justify-center gap-2">
                        <Button variant="outline" size="icon" onClick={goToStart} title="Go to start (Home)">
                            <SkipBack className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="icon" onClick={stepBack} title="Step back (Left arrow)">
                            <Rewind className="h-4 w-4" />
                        </Button>
                        <Button size="icon" onClick={togglePlay} title="Play/Pause (Space)" className="h-12 w-12">
                            {playing ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
                        </Button>
                        <Button variant="outline" size="icon" onClick={stepForward} title="Step forward (Right arrow)">
                            <FastForward className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="icon" onClick={goToEnd} title="Go to end (End)">
                            <SkipForward className="h-4 w-4" />
                        </Button>

                        <div className="ml-4 flex items-center gap-2">
                            <Label className="text-xs text-muted-foreground whitespace-nowrap">Speed</Label>
                            <Select value={String(speed)} onValueChange={v => setSpeed(Number(v))}>
                                <SelectTrigger className="w-24 h-8 text-xs"><SelectValue /></SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="250">0.25s</SelectItem>
                                    <SelectItem value="500">0.5s</SelectItem>
                                    <SelectItem value="1000">1s</SelectItem>
                                    <SelectItem value="2000">2s</SelectItem>
                                    <SelectItem value="3000">3s</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="ml-4 text-sm font-mono text-muted-foreground">
                            {scrubber.currentIndex + 1} / {scrubber.chain_ids.length}
                        </div>
                    </div>

                    {/* Current Node Detail */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                        {/* Node Inspector */}
                        <Card className="lg:col-span-2">
                            <CardHeader className="pb-3">
                                <CardTitle className="text-base flex items-center gap-2">
                                    <Eye className="h-4 w-4" />
                                    {direction === "past" ? "Cause" : "Effect"} #{scrubber.currentIndex + 1}
                                    {loadingNodes && <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />}
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                {currentNode ? (
                                    <NodeDetail node={currentNode} isSeed={currentNodeId === seedNodeId.trim()} />
                                ) : (
                                    <div className="space-y-2">
                                        <p className="font-mono text-xs text-muted-foreground">{currentNodeId}</p>
                                        {loadingNodes && <p className="text-xs text-muted-foreground">Loading node details...</p>}
                                    </div>
                                )}
                            </CardContent>
                        </Card>

                        {/* Edge / Interval Info */}
                        <Card>
                            <CardHeader className="pb-3">
                                <CardTitle className="text-base flex items-center gap-2">
                                    <Clock className="h-4 w-4" /> Minkowski Interval
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                {currentEdge ? (
                                    <EdgeDetail edge={currentEdge} />
                                ) : (
                                    <p className="text-xs text-muted-foreground">
                                        {scrubber.currentIndex === 0 ? "Origin — no incoming edge" : "No edge data"}
                                    </p>
                                )}

                                {/* Chain Stats */}
                                <div className="mt-4 pt-4 border-t space-y-2">
                                    <ChainStats scrubber={scrubber} />
                                </div>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Full Chain List */}
                    <ChainListView scrubber={scrubber} onSelect={idx => setScrubber(prev => prev ? { ...prev, currentIndex: idx } : null)} />
                </>
            )}

            {/* Keyboard shortcuts hint */}
            {scrubber && (
                <p className="text-xs text-muted-foreground text-center">
                    Keyboard: <kbd className="px-1 py-0.5 rounded bg-muted text-[10px] font-mono">Space</kbd> play/pause
                    {" "}<kbd className="px-1 py-0.5 rounded bg-muted text-[10px] font-mono">&larr; &rarr;</kbd> step
                    {" "}<kbd className="px-1 py-0.5 rounded bg-muted text-[10px] font-mono">Home</kbd>/<kbd className="px-1 py-0.5 rounded bg-muted text-[10px] font-mono">End</kbd> jump
                </p>
            )}
        </div>
    )
}

// ── Causal Timeline (SVG Scrubber) ───────────────────────────

function CausalTimeline({ scrubber, onScrub, direction }: {
    scrubber: ScrubberState
    onScrub: (idx: number) => void
    direction: "past" | "future"
}) {
    const svgRef = useRef<SVGSVGElement>(null)
    const { chain_ids, edges, nodes, currentIndex } = scrubber
    const count = chain_ids.length
    const padding = 40
    const height = 120
    const nodeRadius = 8
    const activeRadius = 14

    const getX = (i: number, width: number) => {
        if (count <= 1) return width / 2
        return padding + (i / (count - 1)) * (width - padding * 2)
    }

    const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
        const svg = svgRef.current
        if (!svg) return
        const rect = svg.getBoundingClientRect()
        const x = e.clientX - rect.left
        const width = rect.width
        const ratio = Math.max(0, Math.min(1, (x - padding) / (width - padding * 2)))
        const idx = Math.round(ratio * (count - 1))
        onScrub(idx)
    }

    // Get causal type for edge between i and i+1
    const getEdgeType = (i: number): string => {
        const fromId = chain_ids[i]
        const toId = chain_ids[i + 1]
        const edge = edges.find(e =>
            (e.from_node_id === fromId && e.to_node_id === toId) ||
            (e.to_node_id === fromId && e.from_node_id === toId))
        return edge?.causal_type || "Unknown"
    }

    const getNodeColor = (i: number) => {
        const node = nodes.get(chain_ids[i])
        if (!node) return "#6b7280"
        switch (node.node_type) {
            case "Semantic": return "#00ff66"
            case "Episodic": return "#00f0ff"
            case "Concept": return "#f59e0b"
            case "DreamSnapshot": return "#8b5cf6"
            default: return "#6b7280"
        }
    }

    const edgeColor = (type: string) => {
        switch (type) {
            case "Timelike": return "#10b981"
            case "Lightlike": return "#a855f7"
            case "Spacelike": return "#f59e0b"
            default: return "#374151"
        }
    }

    return (
        <div className="relative">
            {/* Direction label */}
            <div className="absolute top-2 left-4 flex items-center gap-2 z-10">
                <Badge variant="outline" className="text-[10px] bg-background/80 backdrop-blur">
                    {direction === "past" ? "EFFECT \u2192 CAUSE (rewinding)" : "CAUSE \u2192 EFFECT (forward)"}
                </Badge>
            </div>

            <svg
                ref={svgRef}
                className="w-full cursor-pointer select-none"
                height={height}
                onClick={handleClick}
                style={{ minWidth: Math.max(600, count * 30) }}
            >
                {/* Background gradient */}
                <defs>
                    <linearGradient id="timeline-bg" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor="rgba(16,185,129,0.05)" />
                        <stop offset="50%" stopColor="rgba(168,85,247,0.05)" />
                        <stop offset="100%" stopColor="rgba(245,158,11,0.05)" />
                    </linearGradient>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="3" result="blur" />
                        <feMerge>
                            <feMergeNode in="blur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>
                <rect x="0" y="0" width="100%" height={height} fill="url(#timeline-bg)" />

                {/* Center line */}
                <line
                    x1={padding} y1={height / 2}
                    x2={`calc(100% - ${padding}px)`} y2={height / 2}
                    stroke="#374151" strokeWidth="1" strokeDasharray="4,4"
                />

                {/* Scrubbed portion highlight */}
                {count > 1 && (
                    <line
                        x1={padding} y1={height / 2}
                        x2={getX(currentIndex, (svgRef.current?.getBoundingClientRect().width || 800))}
                        y2={height / 2}
                        stroke="#10b981" strokeWidth="2" opacity="0.6"
                    />
                )}

                {/* Edges */}
                {chain_ids.slice(0, -1).map((_, i) => {
                    const w = svgRef.current?.getBoundingClientRect().width || 800
                    const x1 = getX(i, w)
                    const x2 = getX(i + 1, w)
                    const type = getEdgeType(i)
                    const isPast = i < currentIndex
                    return (
                        <line
                            key={`edge-${i}`}
                            x1={x1} y1={height / 2}
                            x2={x2} y2={height / 2}
                            stroke={edgeColor(type)}
                            strokeWidth={isPast ? 2.5 : 1.5}
                            opacity={isPast ? 0.9 : 0.3}
                        />
                    )
                })}

                {/* Nodes */}
                {chain_ids.map((id, i) => {
                    const w = svgRef.current?.getBoundingClientRect().width || 800
                    const x = getX(i, w)
                    const isActive = i === currentIndex
                    const isPast = i <= currentIndex
                    const isSeed = id === chain_ids[direction === "past" ? chain_ids.length - 1 : 0]
                    const r = isActive ? activeRadius : nodeRadius
                    const color = getNodeColor(i)

                    return (
                        <g key={id} onClick={e => { e.stopPropagation(); onScrub(i) }}
                            className="cursor-pointer">
                            {/* Active glow */}
                            {isActive && (
                                <circle cx={x} cy={height / 2} r={r + 4}
                                    fill="none" stroke={color} strokeWidth="2"
                                    opacity="0.4" filter="url(#glow)">
                                    <animate attributeName="r" values={`${r + 2};${r + 6};${r + 2}`} dur="2s" repeatCount="indefinite" />
                                    <animate attributeName="opacity" values="0.4;0.1;0.4" dur="2s" repeatCount="indefinite" />
                                </circle>
                            )}

                            {/* Node circle */}
                            <circle
                                cx={x} cy={height / 2} r={r}
                                fill={isPast ? color : "#1f2937"}
                                stroke={color}
                                strokeWidth={isActive ? 3 : 1.5}
                                opacity={isPast ? 1 : 0.4}
                            />

                            {/* Seed marker */}
                            {isSeed && (
                                <circle cx={x} cy={height / 2} r={3}
                                    fill="#fff" opacity="0.8" />
                            )}

                            {/* Index label */}
                            <text
                                x={x} y={height / 2 + r + 14}
                                textAnchor="middle"
                                className="text-[9px] fill-muted-foreground font-mono"
                                opacity={isActive ? 1 : 0.5}
                            >
                                {i + 1}
                            </text>

                            {/* ds² label on active */}
                            {isActive && i > 0 && (() => {
                                const edge = edges.find(e =>
                                    (e.from_node_id === chain_ids[i - 1] && e.to_node_id === id) ||
                                    (e.to_node_id === chain_ids[i - 1] && e.from_node_id === id))
                                return edge ? (
                                    <text
                                        x={x} y={height / 2 - r - 8}
                                        textAnchor="middle"
                                        className="text-[10px] fill-emerald-400 font-mono font-bold"
                                    >
                                        ds²={edge.minkowski_interval.toFixed(3)}
                                    </text>
                                ) : null
                            })()}
                        </g>
                    )
                })}
            </svg>

            {/* Scrub slider (below SVG, for precise control) */}
            <div className="px-10 pb-3">
                <input
                    type="range"
                    min={0}
                    max={count - 1}
                    value={currentIndex}
                    onChange={e => onScrub(Number(e.target.value))}
                    className="w-full h-1 accent-emerald-500 cursor-pointer"
                />
            </div>
        </div>
    )
}

// ── Node Detail Card ─────────────────────────────────────────

function NodeDetail({ node, isSeed }: { node: ChainNode; isSeed: boolean }) {
    const [expanded, setExpanded] = useState(false)
    const contentStr = JSON.stringify(node.content, null, 2)
    const isLong = contentStr.length > 200

    return (
        <div className="space-y-3">
            <div className="flex items-center gap-2 flex-wrap">
                <Badge variant="outline" className="font-mono text-[10px]">{node.id.substring(0, 16)}...</Badge>
                <NodeTypeBadge type={node.node_type} />
                {isSeed && <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30 text-[10px]">SEED</Badge>}
            </div>

            <div className="grid grid-cols-3 gap-3">
                <MetricCard label="Energy" value={node.energy.toFixed(4)} />
                <MetricCard label="Depth (Poincare)" value={node.depth.toFixed(4)} />
                <MetricCard label="Type" value={node.node_type} />
            </div>

            {node.created_at && (
                <p className="text-[10px] text-muted-foreground">
                    Created: {new Date(node.created_at).toLocaleString()}
                </p>
            )}

            {/* Content */}
            <div>
                <div className="flex items-center gap-2">
                    <Label className="text-xs text-muted-foreground">Content</Label>
                    {isLong && (
                        <Button variant="ghost" size="sm" className="h-5 px-1" onClick={() => setExpanded(!expanded)}>
                            {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                        </Button>
                    )}
                </div>
                <pre className={`text-xs font-mono bg-muted/50 rounded-md p-3 mt-1 overflow-x-auto whitespace-pre-wrap ${!expanded && isLong ? "max-h-24 overflow-y-hidden" : ""}`}>
                    {contentStr}
                </pre>
                {!expanded && isLong && (
                    <div className="h-6 -mt-6 relative bg-gradient-to-t from-card to-transparent" />
                )}
            </div>
        </div>
    )
}

// ── Edge Detail ──────────────────────────────────────────────

function EdgeDetail({ edge }: { edge: CausalEdge }) {
    const ds2 = edge.minkowski_interval
    const classification = ds2 < -0.01 ? "Timelike" : Math.abs(ds2) <= 0.01 ? "Lightlike" : "Spacelike"

    return (
        <div className="space-y-3">
            <div className="space-y-1">
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground">ds² (Minkowski Interval)</p>
                <p className="text-2xl font-bold font-mono text-emerald-400">{ds2.toFixed(6)}</p>
            </div>

            <div className="space-y-1">
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Classification</p>
                <CausalTypeBadge type={classification} />
            </div>

            <div className="space-y-1">
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Edge Type</p>
                <Badge variant="outline" className="text-xs">{edge.edge_type}</Badge>
            </div>

            <div className="text-[10px] text-muted-foreground space-y-0.5 pt-2 border-t">
                <p>ds² = -c²Δt² + ||Δx||²</p>
                <p className="mt-1">
                    {classification === "Timelike" && "Causal connection: source influenced target."}
                    {classification === "Lightlike" && "Light cone boundary: marginal causal contact."}
                    {classification === "Spacelike" && "Causally independent: no influence possible."}
                </p>
            </div>
        </div>
    )
}

// ── Chain Stats ──────────────────────────────────────────────

function ChainStats({ scrubber }: { scrubber: ScrubberState }) {
    const { edges } = scrubber
    const timelike = edges.filter(e => e.causal_type === "Timelike").length
    const lightlike = edges.filter(e => e.causal_type === "Lightlike").length
    const spacelike = edges.filter(e => e.causal_type === "Spacelike").length

    return (
        <>
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold">Chain Composition</p>
            <div className="flex gap-2 flex-wrap">
                <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/30 text-[10px]">
                    Timelike: {timelike}
                </Badge>
                <Badge className="bg-purple-500/10 text-purple-400 border-purple-500/30 text-[10px]">
                    Lightlike: {lightlike}
                </Badge>
                <Badge className="bg-amber-500/10 text-amber-400 border-amber-500/30 text-[10px]">
                    Spacelike: {spacelike}
                </Badge>
            </div>

            {/* Causal purity */}
            {edges.length > 0 && (
                <div className="mt-1">
                    <p className="text-[10px] text-muted-foreground">
                        Causal Purity: {((timelike / edges.length) * 100).toFixed(0)}%
                        {timelike === edges.length && " (pure timelike chain)"}
                    </p>
                </div>
            )}
        </>
    )
}

// ── Full Chain List ──────────────────────────────────────────

function ChainListView({ scrubber, onSelect }: { scrubber: ScrubberState; onSelect: (idx: number) => void }) {
    const [collapsed, setCollapsed] = useState(true)

    return (
        <Card>
            <CardHeader className="pb-3 cursor-pointer" onClick={() => setCollapsed(!collapsed)}>
                <CardTitle className="text-base flex items-center gap-2">
                    Full Chain ({scrubber.chain_ids.length} nodes)
                    {collapsed ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
                </CardTitle>
            </CardHeader>
            {!collapsed && (
                <CardContent>
                    <ScrollArea className="max-h-80">
                        <div className="space-y-1">
                            {scrubber.chain_ids.map((id, i) => {
                                const node = scrubber.nodes.get(id)
                                const isActive = i === scrubber.currentIndex
                                const edge = i > 0 ? scrubber.edges.find(e =>
                                    (e.from_node_id === scrubber.chain_ids[i - 1] && e.to_node_id === id) ||
                                    (e.to_node_id === scrubber.chain_ids[i - 1] && e.from_node_id === id)
                                ) : null

                                return (
                                    <div
                                        key={id}
                                        className={`flex items-center gap-2 px-3 py-1.5 rounded cursor-pointer transition-colors ${isActive ? "bg-emerald-500/10 border border-emerald-500/30" : "hover:bg-muted/50"}`}
                                        onClick={() => onSelect(i)}
                                    >
                                        <span className="text-[10px] font-mono text-muted-foreground w-6 text-right">{i + 1}</span>
                                        <span className="font-mono text-xs">{id.substring(0, 16)}...</span>
                                        {node && <NodeTypeBadge type={node.node_type} />}
                                        {node && <span className="text-[10px] text-muted-foreground ml-auto">E={node.energy.toFixed(2)}</span>}
                                        {edge && (
                                            <span className="text-[10px] font-mono text-emerald-400">
                                                ds²={edge.minkowski_interval.toFixed(3)}
                                            </span>
                                        )}
                                    </div>
                                )
                            })}
                        </div>
                    </ScrollArea>
                </CardContent>
            )}
        </Card>
    )
}

// ── Shared UI Components ─────────────────────────────────────

function MetricCard({ label, value }: { label: string; value: string }) {
    return (
        <div className="rounded-lg border bg-muted/30 p-3">
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</p>
            <p className="text-lg font-semibold font-mono">{value}</p>
        </div>
    )
}

function NodeTypeBadge({ type }: { type: string }) {
    const color = type === "Semantic" ? "bg-green-500/10 text-green-400 border-green-500/30"
        : type === "Episodic" ? "bg-cyan-500/10 text-cyan-400 border-cyan-500/30"
        : type === "Concept" ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
        : type === "DreamSnapshot" ? "bg-purple-500/10 text-purple-400 border-purple-500/30"
        : "bg-gray-500/10 text-gray-400 border-gray-500/30"
    return <Badge variant="outline" className={`text-[10px] ${color}`}>{type}</Badge>
}

function CausalTypeBadge({ type }: { type: string }) {
    const color = type === "Timelike" ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
        : type === "Lightlike" ? "bg-purple-500/20 text-purple-400 border-purple-500/30"
        : "bg-amber-500/20 text-amber-400 border-amber-500/30"
    return <Badge variant="outline" className={`text-sm font-semibold ${color}`}>{type}</Badge>
}

function ErrorInline({ message }: { message: string }) {
    return (
        <div className="flex items-center gap-2 text-destructive text-sm">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span className="font-mono text-xs">{message}</span>
        </div>
    )
}
