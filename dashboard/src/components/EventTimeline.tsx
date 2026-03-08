// EventTimeline.tsx — Interactive event timeline bar for NietzscheDB Dashboard
// Shows graph events (inserts, updates, deletes, sleep, zaratustra) on a time axis
// Supports zoom levels: 1min, 5min, 1h, 6h, 24h, 7d
// Click on event to highlight, hover for tooltip

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { format } from "date-fns"
import { Clock, ZoomIn, ZoomOut } from "lucide-react"

import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TimelineEvent {
  id: string
  timestamp: number // epoch ms
  type:
    | "InsertNode"
    | "UpdateNode"
    | "DeleteNode"
    | "InsertEdge"
    | "DeleteEdge"
    | "SleepCycle"
    | "Zaratustra"
    | "Dream"
    | "Query"
    | "Backup"
  nodeId?: string
  description?: string
}

export interface EventTimelineProps {
  events: TimelineEvent[]
  onEventClick?: (event: TimelineEvent) => void
  className?: string
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EVENT_COLORS: Record<TimelineEvent["type"], string> = {
  InsertNode: "#00ff66",
  UpdateNode: "#00f0ff",
  DeleteNode: "#ff4444",
  InsertEdge: "#22c55e",
  DeleteEdge: "#f59e0b",
  SleepCycle: "#8b5cf6",
  Zaratustra: "#ffd700",
  Dream: "#ff00ff",
  Query: "#94a3b8",
  Backup: "#3b82f6",
}

type ZoomLevel = "1m" | "5m" | "1h" | "6h" | "24h" | "7d"

const ZOOM_MS: Record<ZoomLevel, number> = {
  "1m": 60_000,
  "5m": 300_000,
  "1h": 3_600_000,
  "6h": 21_600_000,
  "24h": 86_400_000,
  "7d": 604_800_000,
}

const ZOOM_ORDER: ZoomLevel[] = ["1m", "5m", "1h", "6h", "24h", "7d"]

const TICK_FORMATS: Record<ZoomLevel, string> = {
  "1m": "HH:mm:ss",
  "5m": "HH:mm:ss",
  "1h": "HH:mm",
  "6h": "HH:mm",
  "24h": "HH:mm",
  "7d": "MMM dd HH:mm",
}

const SVG_HEIGHT = 100
const AXIS_Y = 80
const DOT_RADIUS = 5
const BIN_THRESHOLD = 200

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface Bin {
  start: number
  end: number
  events: TimelineEvent[]
}

function binEvents(events: TimelineEvent[], start: number, end: number, binCount: number): Bin[] {
  const span = end - start
  const binWidth = span / binCount
  const bins: Bin[] = Array.from({ length: binCount }, (_, i) => ({
    start: start + i * binWidth,
    end: start + (i + 1) * binWidth,
    events: [],
  }))
  for (const ev of events) {
    const idx = Math.min(Math.floor(((ev.timestamp - start) / span) * binCount), binCount - 1)
    if (idx >= 0 && idx < binCount) bins[idx].events.push(ev)
  }
  return bins
}

function dominantType(events: TimelineEvent[]): TimelineEvent["type"] {
  const counts = new Map<TimelineEvent["type"], number>()
  for (const e of events) counts.set(e.type, (counts.get(e.type) ?? 0) + 1)
  let best: TimelineEvent["type"] = events[0].type
  let max = 0
  for (const [t, c] of counts) {
    if (c > max) { max = c; best = t }
  }
  return best
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function EventTimeline({ events, onEventClick, className }: EventTimelineProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [zoom, setZoom] = useState<ZoomLevel>("1h")
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [tooltip, setTooltip] = useState<{ x: number; y: number; event: TimelineEvent } | null>(null)
  const [panOffset, setPanOffset] = useState(0) // px offset from right edge (0 = now is rightmost)
  const dragRef = useRef<{ startX: number; startOffset: number } | null>(null)

  const svgWidth = 1200

  const now = Date.now()
  const windowMs = ZOOM_MS[zoom]
  const pxPerMs = svgWidth / windowMs
  const panMs = panOffset / pxPerMs
  const windowEnd = now - panMs
  const windowStart = windowEnd - windowMs

  // Visible events
  const visible = useMemo(
    () => events.filter((e) => e.timestamp >= windowStart && e.timestamp <= windowEnd),
    [events, windowStart, windowEnd],
  )

  const useBins = visible.length > BIN_THRESHOLD
  const bins = useMemo(
    () => (useBins ? binEvents(visible, windowStart, windowEnd, 80) : []),
    [useBins, visible, windowStart, windowEnd],
  )

  // Counts per type
  const counts = useMemo(() => {
    const m = new Map<TimelineEvent["type"], number>()
    for (const e of visible) m.set(e.type, (m.get(e.type) ?? 0) + 1)
    return m
  }, [visible])

  // Ticks
  const ticks = useMemo(() => {
    const count = 8
    const step = windowMs / count
    return Array.from({ length: count + 1 }, (_, i) => windowStart + i * step)
  }, [windowStart, windowMs])

  const toX = useCallback((ts: number) => ((ts - windowStart) / windowMs) * svgWidth, [windowStart, windowMs])

  // Pan handlers
  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      dragRef.current = { startX: e.clientX, startOffset: panOffset }
      ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    },
    [panOffset],
  )

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragRef.current) return
      const dx = dragRef.current.startX - e.clientX
      setPanOffset(Math.max(0, dragRef.current.startOffset + dx))
    },
    [],
  )

  const onPointerUp = useCallback(() => {
    dragRef.current = null
  }, [])

  // Zoom in/out helpers
  const zoomIn = () => {
    const idx = ZOOM_ORDER.indexOf(zoom)
    if (idx > 0) setZoom(ZOOM_ORDER[idx - 1])
  }
  const zoomOut = () => {
    const idx = ZOOM_ORDER.indexOf(zoom)
    if (idx < ZOOM_ORDER.length - 1) setZoom(ZOOM_ORDER[idx + 1])
  }

  // Reset pan on zoom change
  useEffect(() => setPanOffset(0), [zoom])

  // Now marker x
  const nowX = toX(now)

  return (
    <Card className={cn("border-purple-900/40 bg-background p-3", className)}>
      {/* Header: counts + zoom controls */}
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          <Clock className="h-4 w-4 text-purple-400" />
          <span className="text-xs font-medium text-slate-300">Events</span>
          {(Object.keys(EVENT_COLORS) as TimelineEvent["type"][]).map((t) => {
            const c = counts.get(t)
            if (!c) return null
            return (
              <Badge
                key={t}
                variant="outline"
                className="border-none px-1.5 py-0 text-[10px] font-mono"
                style={{ backgroundColor: EVENT_COLORS[t] + "22", color: EVENT_COLORS[t] }}
              >
                {t} {c}
              </Badge>
            )
          })}
        </div>

        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon" className="h-6 w-6 text-slate-400" onClick={zoomIn}>
            <ZoomIn className="h-3.5 w-3.5" />
          </Button>
          {ZOOM_ORDER.map((z) => (
            <Button
              key={z}
              variant={zoom === z ? "default" : "ghost"}
              size="sm"
              className={cn(
                "h-6 px-1.5 text-[10px] font-mono",
                zoom === z ? "bg-purple-700 text-white hover:bg-purple-600" : "text-slate-400",
              )}
              onClick={() => setZoom(z)}
            >
              {z}
            </Button>
          ))}
          <Button variant="ghost" size="icon" className="h-6 w-6 text-slate-400" onClick={zoomOut}>
            <ZoomOut className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Timeline SVG */}
      <ScrollArea className="w-full">
        <div
          ref={containerRef}
          className="relative select-none"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          style={{ cursor: dragRef.current ? "grabbing" : "grab" }}
        >
          <svg
            ref={svgRef}
            viewBox={`0 0 ${svgWidth} ${SVG_HEIGHT}`}
            className="w-full"
            style={{ minWidth: 600 }}
          >
            {/* Axis line */}
            <line x1={0} y1={AXIS_Y} x2={svgWidth} y2={AXIS_Y} stroke="#334155" strokeWidth={1} />

            {/* Tick marks + labels */}
            {ticks.map((ts, i) => {
              const x = toX(ts)
              return (
                <g key={i}>
                  <line x1={x} y1={AXIS_Y - 4} x2={x} y2={AXIS_Y + 4} stroke="#475569" strokeWidth={1} />
                  <text x={x} y={AXIS_Y + 16} textAnchor="middle" fill="#64748b" fontSize={9} fontFamily="monospace">
                    {format(new Date(ts), TICK_FORMATS[zoom])}
                  </text>
                </g>
              )
            })}

            {/* Now marker */}
            {nowX >= 0 && nowX <= svgWidth && (
              <g>
                <line x1={nowX} y1={10} x2={nowX} y2={AXIS_Y} stroke="#7c3aed" strokeWidth={1.5} strokeDasharray="4 2" />
                <text x={nowX} y={8} textAnchor="middle" fill="#7c3aed" fontSize={8} fontFamily="monospace">
                  now
                </text>
              </g>
            )}

            {/* Binned heatmap mode */}
            {useBins &&
              bins.map((bin, i) => {
                if (bin.events.length === 0) return null
                const x = toX(bin.start)
                const w = Math.max(toX(bin.end) - x, 2)
                const maxCount = Math.max(...bins.map((b) => b.events.length), 1)
                const height = Math.max(4, (bin.events.length / maxCount) * (AXIS_Y - 16))
                const color = EVENT_COLORS[dominantType(bin.events)]
                return (
                  <rect
                    key={i}
                    x={x}
                    y={AXIS_Y - height}
                    width={w}
                    height={height}
                    rx={1}
                    fill={color}
                    fillOpacity={0.25 + 0.55 * (bin.events.length / maxCount)}
                    className="cursor-pointer"
                    onMouseEnter={(e) => {
                      const rect = (e.target as SVGRectElement).getBoundingClientRect()
                      setTooltip({
                        x: rect.left + rect.width / 2,
                        y: rect.top,
                        event: {
                          id: `bin-${i}`,
                          timestamp: bin.start,
                          type: dominantType(bin.events),
                          description: `${bin.events.length} events`,
                        },
                      })
                    }}
                    onMouseLeave={() => setTooltip(null)}
                  />
                )
              })}

            {/* Individual event dots */}
            {!useBins &&
              visible.map((ev) => {
                const x = toX(ev.timestamp)
                const isSelected = selectedId === ev.id
                return (
                  <g key={ev.id}>
                    {isSelected && (
                      <circle cx={x} cy={AXIS_Y - 24} r={DOT_RADIUS + 4} fill="none" stroke="#7c3aed" strokeWidth={1.5} opacity={0.7} />
                    )}
                    <circle
                      cx={x}
                      cy={AXIS_Y - 24}
                      r={isSelected ? DOT_RADIUS + 1 : DOT_RADIUS}
                      fill={EVENT_COLORS[ev.type]}
                      fillOpacity={isSelected ? 1 : 0.85}
                      stroke={isSelected ? "#fff" : "none"}
                      strokeWidth={1}
                      className="cursor-pointer transition-all duration-150"
                      onMouseEnter={(e) => {
                        const rect = (e.target as SVGCircleElement).getBoundingClientRect()
                        setTooltip({ x: rect.left + rect.width / 2, y: rect.top, event: ev })
                      }}
                      onMouseLeave={() => setTooltip(null)}
                      onClick={() => {
                        setSelectedId(ev.id === selectedId ? null : ev.id)
                        onEventClick?.(ev)
                      }}
                    />
                    {/* Stem line from dot to axis */}
                    <line
                      x1={x}
                      y1={AXIS_Y - 24 + DOT_RADIUS}
                      x2={x}
                      y2={AXIS_Y}
                      stroke={EVENT_COLORS[ev.type]}
                      strokeWidth={0.7}
                      opacity={0.35}
                    />
                  </g>
                )
              })}
          </svg>

          {/* Tooltip overlay */}
          {tooltip && containerRef.current && (
            <div
              className="pointer-events-none fixed z-50 rounded border border-purple-800/50 bg-slate-900 px-2.5 py-1.5 text-[11px] shadow-lg"
              style={{ left: tooltip.x, top: tooltip.y - 48, transform: "translateX(-50%)" }}
            >
              <div className="flex items-center gap-1.5">
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ backgroundColor: EVENT_COLORS[tooltip.event.type] }}
                />
                <span className="font-mono font-semibold text-slate-100">{tooltip.event.type}</span>
              </div>
              <div className="text-slate-400">{format(new Date(tooltip.event.timestamp), "yyyy-MM-dd HH:mm:ss")}</div>
              {tooltip.event.nodeId && <div className="font-mono text-slate-500">node: {tooltip.event.nodeId}</div>}
              {tooltip.event.description && <div className="text-slate-400">{tooltip.event.description}</div>}
            </div>
          )}
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>

      {/* Legend */}
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-slate-800 pt-2">
        {(Object.entries(EVENT_COLORS) as [TimelineEvent["type"], string][]).map(([type, color]) => (
          <div key={type} className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-[10px] text-slate-500">{type}</span>
          </div>
        ))}
      </div>
    </Card>
  )
}

export default EventTimeline
