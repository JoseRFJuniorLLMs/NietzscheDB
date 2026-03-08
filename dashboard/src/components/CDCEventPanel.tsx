import { useCallback, useEffect, useRef, useState } from "react"
import { Activity, ChevronDown, ChevronUp, Filter, Pause, Play } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

export interface CDCEvent {
    id: string
    timestamp: number
    event_type:
        | "InsertNode"
        | "UpdateNode"
        | "DeleteNode"
        | "InsertEdge"
        | "DeleteEdge"
        | "SleepCycle"
        | "Zaratustra"
        | "Defibrillate"
    node_id: string | null
    data?: Record<string, unknown>
}

export interface CDCEventPanelProps {
    events: CDCEvent[]
    maxVisible?: number
    onEventClick?: (event: CDCEvent) => void
    className?: string
}

const EVENT_COLORS: Record<CDCEvent["event_type"], string> = {
    InsertNode: "#00ff66",
    UpdateNode: "#00f0ff",
    DeleteNode: "#ff4444",
    InsertEdge: "#a855f7",
    DeleteEdge: "#f97316",
    SleepCycle: "#facc15",
    Zaratustra: "#ec4899",
    Defibrillate: "#6366f1",
}

function relativeTime(ts: number): string {
    const delta = Math.floor((Date.now() - ts) / 1000)
    if (delta < 5) return "just now"
    if (delta < 60) return `${delta}s ago`
    if (delta < 3600) return `${Math.floor(delta / 60)}m ago`
    return `${Math.floor(delta / 3600)}h ago`
}

export function CDCEventPanel({
    events,
    maxVisible = 50,
    onEventClick,
    className,
}: CDCEventPanelProps) {
    const [paused, setPaused] = useState(false)
    const [compact, setCompact] = useState(false)
    const [activeFilters, setActiveFilters] = useState<Set<CDCEvent["event_type"]>>(
        new Set(Object.keys(EVENT_COLORS) as CDCEvent["event_type"][])
    )
    const [showFilters, setShowFilters] = useState(false)
    const scrollRef = useRef<HTMLDivElement>(null)
    const prevCountRef = useRef(events.length)

    // Auto-scroll to top when new events arrive (unless paused)
    useEffect(() => {
        if (!paused && events.length > prevCountRef.current && scrollRef.current) {
            scrollRef.current.scrollTop = 0
        }
        prevCountRef.current = events.length
    }, [events.length, paused])

    const toggleFilter = useCallback((type: CDCEvent["event_type"]) => {
        setActiveFilters((prev) => {
            const next = new Set(prev)
            if (next.has(type)) next.delete(type)
            else next.add(type)
            return next
        })
    }, [])

    const filtered = events
        .filter((e) => activeFilters.has(e.event_type))
        .slice(0, maxVisible)

    const isReceiving = events.length > 0 && Date.now() - events[0]?.timestamp < 5000

    return (
        <div
            className={cn(
                "flex flex-col rounded-lg border border-white/10 bg-black/60 backdrop-blur-sm",
                className
            )}
        >
            {/* Header */}
            <div className="flex items-center justify-between gap-2 border-b border-white/10 px-3 py-2">
                <div className="flex items-center gap-2">
                    <span
                        className={cn(
                            "h-2 w-2 rounded-full",
                            isReceiving
                                ? "animate-pulse bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.7)]"
                                : "bg-zinc-500"
                        )}
                    />
                    <Badge variant="outline" className="border-white/20 text-xs text-zinc-300">
                        {events.length} events
                    </Badge>
                </div>
                <div className="flex items-center gap-1">
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-zinc-400 hover:text-white"
                        onClick={() => setShowFilters((v) => !v)}
                        title="Filter events"
                    >
                        <Filter className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-zinc-400 hover:text-white"
                        onClick={() => setCompact((v) => !v)}
                        title={compact ? "Expand" : "Compact"}
                    >
                        {compact ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronUp className="h-3.5 w-3.5" />}
                    </Button>
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-zinc-400 hover:text-white"
                        onClick={() => setPaused((v) => !v)}
                        title={paused ? "Resume auto-scroll" : "Pause auto-scroll"}
                    >
                        {paused ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
                    </Button>
                </div>
            </div>

            {/* Filter toggles */}
            {showFilters && (
                <div className="flex flex-wrap gap-1.5 border-b border-white/10 px-3 py-2">
                    {(Object.keys(EVENT_COLORS) as CDCEvent["event_type"][]).map((type) => (
                        <button
                            key={type}
                            onClick={() => toggleFilter(type)}
                            className={cn(
                                "rounded-full border px-2 py-0.5 text-[10px] font-medium transition-all",
                                activeFilters.has(type)
                                    ? "border-white/30 text-white"
                                    : "border-white/5 text-zinc-600"
                            )}
                            style={{
                                backgroundColor: activeFilters.has(type)
                                    ? `${EVENT_COLORS[type]}18`
                                    : undefined,
                            }}
                        >
                            {type}
                        </button>
                    ))}
                </div>
            )}

            {/* Event list */}
            <ScrollArea className="h-[320px]">
                <div ref={scrollRef} className="flex flex-col">
                    {filtered.map((event) => (
                        <button
                            key={event.id}
                            onClick={() => onEventClick?.(event)}
                            className="flex items-center gap-2 border-b border-white/5 px-3 py-1.5 text-left transition-colors hover:bg-white/5"
                        >
                            <span
                                className="h-2 w-2 shrink-0 rounded-full"
                                style={{
                                    backgroundColor: EVENT_COLORS[event.event_type],
                                    boxShadow: `0 0 6px ${EVENT_COLORS[event.event_type]}66`,
                                }}
                            />
                            <Badge
                                variant="outline"
                                className="shrink-0 border-white/10 text-[10px] text-zinc-300"
                            >
                                {event.event_type}
                            </Badge>
                            {event.node_id && (
                                <span className="min-w-0 truncate font-mono text-[11px] text-zinc-500">
                                    {event.node_id.length > 12
                                        ? `${event.node_id.slice(0, 6)}...${event.node_id.slice(-4)}`
                                        : event.node_id}
                                </span>
                            )}
                            <span className="ml-auto shrink-0 text-[10px] text-zinc-600">
                                {relativeTime(event.timestamp)}
                            </span>
                        </button>
                    ))}
                    {filtered.length === 0 && (
                        <div className="flex flex-col items-center gap-2 py-10 text-zinc-600">
                            <Activity className="h-5 w-5" />
                            <span className="text-xs">No events</span>
                        </div>
                    )}
                </div>
            </ScrollArea>
        </div>
    )
}
