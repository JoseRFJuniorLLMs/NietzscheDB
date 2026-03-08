// EmbeddingProjector.tsx — 2D embedding visualization for NietzscheDB Dashboard
// Renders pre-projected points on an SVG canvas with zoom, pan, tooltips, and legend
// Actual t-SNE/UMAP reduction should be done server-side

import { useCallback, useMemo, useRef, useState } from "react"
import { Maximize2, Orbit } from "lucide-react"

import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ProjectedPoint {
    id: string
    x: number
    y: number
    label?: string
    type?: string
    energy?: number
}

export interface EmbeddingProjectorProps {
    points: ProjectedPoint[]
    onPointClick?: (id: string) => void
    className?: string
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SVG_WIDTH = 600
const SVG_HEIGHT = 400

const TYPE_COLORS: Record<string, string> = {
    Semantic: "#00ff66",
    Episodic: "#00f0ff",
    Concept: "#ff00ff",
    DreamSnapshot: "#ffd700",
}

const DEFAULT_COLOR = "#94a3b8"

function getColor(type?: string): string {
    if (!type) return DEFAULT_COLOR
    return TYPE_COLORS[type] ?? DEFAULT_COLOR
}

function getRadius(energy?: number): number {
    const e = energy ?? 0.5
    return 3 + e * 5 // 3-8px
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function EmbeddingProjector({ points, onPointClick, className }: EmbeddingProjectorProps) {
    const svgRef = useRef<SVGSVGElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)

    // View state
    const [viewBox, setViewBox] = useState({ x: 0, y: 0, w: SVG_WIDTH, h: SVG_HEIGHT })
    const [tooltip, setTooltip] = useState<{ px: number; py: number; point: ProjectedPoint } | null>(null)
    const [selectedId, setSelectedId] = useState<string | null>(null)
    const [colorBy, setColorBy] = useState<"type" | "energy">("type")

    // Pan state
    const dragRef = useRef<{ startX: number; startY: number; startVB: typeof viewBox } | null>(null)

    // Normalize points to SVG coordinate space (with 40px padding)
    const normalizedPoints = useMemo(() => {
        if (points.length === 0) return []
        const pad = 40
        const xs = points.map((p) => p.x)
        const ys = points.map((p) => p.y)
        const minX = Math.min(...xs)
        const maxX = Math.max(...xs)
        const minY = Math.min(...ys)
        const maxY = Math.max(...ys)
        const rangeX = maxX - minX || 1
        const rangeY = maxY - minY || 1
        return points.map((p) => ({
            ...p,
            nx: pad + ((p.x - minX) / rangeX) * (SVG_WIDTH - 2 * pad),
            ny: pad + ((p.y - minY) / rangeY) * (SVG_HEIGHT - 2 * pad),
        }))
    }, [points])

    // Unique types for legend
    const uniqueTypes = useMemo(() => {
        const types = new Set<string>()
        for (const p of points) {
            if (p.type) types.add(p.type)
        }
        return Array.from(types).sort()
    }, [points])

    // Mouse wheel zoom
    const onWheel = useCallback(
        (e: React.WheelEvent) => {
            e.preventDefault()
            const factor = e.deltaY > 0 ? 1.15 : 0.87
            setViewBox((vb) => {
                const cx = vb.x + vb.w / 2
                const cy = vb.y + vb.h / 2
                const nw = vb.w * factor
                const nh = vb.h * factor
                return { x: cx - nw / 2, y: cy - nh / 2, w: nw, h: nh }
            })
        },
        [],
    )

    // Pan handlers
    const onPointerDown = useCallback(
        (e: React.PointerEvent) => {
            dragRef.current = { startX: e.clientX, startY: e.clientY, startVB: { ...viewBox } }
            ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
        },
        [viewBox],
    )

    const onPointerMove = useCallback(
        (e: React.PointerEvent) => {
            if (!dragRef.current || !containerRef.current) return
            const rect = containerRef.current.getBoundingClientRect()
            const scaleX = viewBox.w / rect.width
            const scaleY = viewBox.h / rect.height
            const dx = (dragRef.current.startX - e.clientX) * scaleX
            const dy = (dragRef.current.startY - e.clientY) * scaleY
            setViewBox({
                x: dragRef.current.startVB.x + dx,
                y: dragRef.current.startVB.y + dy,
                w: dragRef.current.startVB.w,
                h: dragRef.current.startVB.h,
            })
        },
        [viewBox.w, viewBox.h],
    )

    const onPointerUp = useCallback(() => {
        dragRef.current = null
    }, [])

    // Reset view
    const resetView = useCallback(() => {
        setViewBox({ x: 0, y: 0, w: SVG_WIDTH, h: SVG_HEIGHT })
        setSelectedId(null)
        setTooltip(null)
    }, [])

    // Energy-based color (green → yellow → red gradient)
    function energyColor(energy: number): string {
        if (energy < 0.5) {
            const t = energy * 2
            const r = Math.round(255 * t)
            const g = 255
            return `rgb(${r}, ${g}, 50)`
        }
        const t = (energy - 0.5) * 2
        const r = 255
        const g = Math.round(255 * (1 - t))
        return `rgb(${r}, ${g}, 50)`
    }

    function pointColor(p: ProjectedPoint): string {
        if (colorBy === "energy") return energyColor(p.energy ?? 0.5)
        return getColor(p.type)
    }

    return (
        <Card className={cn("border-purple-900/40 bg-background p-4", className)}>
            {/* Header */}
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                    <Orbit className="h-4 w-4 text-purple-400" />
                    <span className="text-sm font-semibold text-slate-200">
                        Embedding Space (2D Projection)
                    </span>
                    <Badge
                        variant="outline"
                        className="border-purple-800/50 px-1.5 py-0 text-[10px] font-mono text-purple-400"
                    >
                        {points.length} points
                    </Badge>
                </div>

                <div className="flex items-center gap-2">
                    <Select value={colorBy} onValueChange={(v) => setColorBy(v as "type" | "energy")}>
                        <SelectTrigger className="h-7 w-[110px] border-slate-700 bg-slate-900 text-xs">
                            <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="type">Color: Type</SelectItem>
                            <SelectItem value="energy">Color: Energy</SelectItem>
                        </SelectContent>
                    </Select>

                    <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 gap-1 px-2 text-xs text-slate-400 hover:text-slate-200"
                        onClick={resetView}
                    >
                        <Maximize2 className="h-3 w-3" />
                        Reset View
                    </Button>
                </div>
            </div>

            {/* SVG Canvas */}
            <div
                ref={containerRef}
                className="relative select-none overflow-hidden rounded-md border border-slate-800 bg-slate-950"
                onWheel={onWheel}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                style={{ cursor: dragRef.current ? "grabbing" : "grab" }}
            >
                <svg
                    ref={svgRef}
                    viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`}
                    className="w-full"
                    style={{ aspectRatio: `${SVG_WIDTH}/${SVG_HEIGHT}` }}
                >
                    {/* Grid lines */}
                    {Array.from({ length: 7 }, (_, i) => {
                        const x = (i * SVG_WIDTH) / 6
                        return (
                            <line key={`vg-${i}`} x1={x} y1={0} x2={x} y2={SVG_HEIGHT} stroke="#1e293b" strokeWidth={0.5} />
                        )
                    })}
                    {Array.from({ length: 5 }, (_, i) => {
                        const y = (i * SVG_HEIGHT) / 4
                        return (
                            <line key={`hg-${i}`} x1={0} y1={y} x2={SVG_WIDTH} y2={y} stroke="#1e293b" strokeWidth={0.5} />
                        )
                    })}

                    {/* Points */}
                    {normalizedPoints.map((p) => {
                        const isSelected = selectedId === p.id
                        const r = getRadius(p.energy)
                        const color = pointColor(p)
                        return (
                            <g key={p.id}>
                                {/* Glow */}
                                <circle
                                    cx={p.nx}
                                    cy={p.ny}
                                    r={r * 2.5}
                                    fill={color}
                                    fillOpacity={isSelected ? 0.15 : 0.06}
                                />
                                {/* Selection ring */}
                                {isSelected && (
                                    <circle
                                        cx={p.nx}
                                        cy={p.ny}
                                        r={r + 4}
                                        fill="none"
                                        stroke="#7c3aed"
                                        strokeWidth={1.5}
                                        opacity={0.8}
                                    />
                                )}
                                {/* Point */}
                                <circle
                                    cx={p.nx}
                                    cy={p.ny}
                                    r={isSelected ? r + 1 : r}
                                    fill={color}
                                    fillOpacity={isSelected ? 1 : 0.85}
                                    stroke={isSelected ? "#fff" : "none"}
                                    strokeWidth={1}
                                    className="cursor-pointer transition-all duration-100"
                                    onMouseEnter={(e) => {
                                        const rect = (e.target as SVGCircleElement).getBoundingClientRect()
                                        setTooltip({
                                            px: rect.left + rect.width / 2,
                                            py: rect.top,
                                            point: p,
                                        })
                                    }}
                                    onMouseLeave={() => setTooltip(null)}
                                    onClick={() => {
                                        setSelectedId(p.id === selectedId ? null : p.id)
                                        onPointClick?.(p.id)
                                    }}
                                />
                            </g>
                        )
                    })}
                </svg>

                {/* Tooltip */}
                {tooltip && (
                    <div
                        className="pointer-events-none fixed z-50 rounded border border-purple-800/50 bg-slate-900 px-2.5 py-1.5 text-[11px] shadow-lg"
                        style={{ left: tooltip.px, top: tooltip.py - 60, transform: "translateX(-50%)" }}
                    >
                        <div className="flex items-center gap-1.5">
                            <span
                                className="inline-block h-2 w-2 rounded-full"
                                style={{ backgroundColor: pointColor(tooltip.point) }}
                            />
                            <span className="font-mono font-semibold text-slate-100">
                                {tooltip.point.label ?? tooltip.point.id}
                            </span>
                        </div>
                        {tooltip.point.type && (
                            <div className="text-slate-400">
                                Type: <span className="text-slate-300">{tooltip.point.type}</span>
                            </div>
                        )}
                        {tooltip.point.energy !== undefined && (
                            <div className="text-slate-400">
                                Energy: <span className="font-mono text-slate-300">{tooltip.point.energy.toFixed(3)}</span>
                            </div>
                        )}
                        <div className="font-mono text-slate-500 text-[9px]">{tooltip.point.id}</div>
                    </div>
                )}
            </div>

            {/* Legend */}
            <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-slate-800 pt-2">
                {colorBy === "type" ? (
                    <>
                        {uniqueTypes.map((type) => (
                            <div key={type} className="flex items-center gap-1.5">
                                <span
                                    className="inline-block h-2.5 w-2.5 rounded-full"
                                    style={{ backgroundColor: getColor(type) }}
                                />
                                <span className="text-[11px] text-slate-400">{type}</span>
                            </div>
                        ))}
                        {uniqueTypes.length === 0 && (
                            <span className="text-[11px] text-slate-500">No points to display</span>
                        )}
                    </>
                ) : (
                    <div className="flex items-center gap-2">
                        <span className="text-[11px] text-slate-400">Energy:</span>
                        <div
                            className="h-2.5 w-24 rounded-full"
                            style={{
                                background: "linear-gradient(to right, rgb(0,255,50), rgb(255,255,50), rgb(255,0,50))",
                            }}
                        />
                        <span className="text-[10px] font-mono text-slate-500">0.0</span>
                        <span className="text-[10px] font-mono text-slate-500">1.0</span>
                    </div>
                )}
            </div>
        </Card>
    )
}

export default EmbeddingProjector
