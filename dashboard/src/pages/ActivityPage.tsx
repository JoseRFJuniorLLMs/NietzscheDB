import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { create } from "zustand"
import { persist } from "zustand/middleware"
import {
    Archive, GitBranch, Moon, Pencil, Plus, Route, Search,
    Sparkles, Terminal, Trash2, Upload, Zap, X, Clock,
    CheckCircle2, XCircle, AlertCircle, BarChart3,
} from "lucide-react"
import { formatDistanceToNow, isToday, isYesterday, isThisWeek } from "date-fns"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"
import { api } from "@/lib/api"

/* ── Types ──────────────────────────────────────────────────── */

type ActivityType =
    | "query" | "insert" | "update" | "delete"
    | "sleep" | "dream" | "zaratustra"
    | "backup" | "import" | "algorithm" | "navigate"

interface ActivityEntry {
    id: string
    timestamp: number
    type: ActivityType
    description: string
    collection?: string
    nodeId?: string
    duration_ms?: number
    status: "success" | "error" | "pending"
    details?: Record<string, unknown>
}

/* ── Constants ──────────────────────────────────────────────── */

const MAX_ENTRIES = 500
const POLL_INTERVAL = 5_000
const STORAGE_KEY = "nietzsche_activity_log"

const ACTIVITY_TYPES: ActivityType[] = [
    "query", "insert", "update", "delete", "sleep", "dream",
    "zaratustra", "backup", "import", "algorithm", "navigate",
]

const TYPE_ICONS: Record<ActivityType, typeof Terminal> = {
    query: Terminal,
    insert: Plus,
    update: Pencil,
    delete: Trash2,
    sleep: Moon,
    dream: Sparkles,
    zaratustra: Zap,
    backup: Archive,
    import: Upload,
    algorithm: GitBranch,
    navigate: Route,
}

const TYPE_COLORS: Record<ActivityType, string> = {
    query:      "bg-blue-500/20 text-blue-400 border-blue-500/30",
    insert:     "bg-green-500/20 text-green-400 border-green-500/30",
    update:     "bg-amber-500/20 text-amber-400 border-amber-500/30",
    delete:     "bg-red-500/20 text-red-400 border-red-500/30",
    sleep:      "bg-indigo-500/20 text-indigo-400 border-indigo-500/30",
    dream:      "bg-pink-500/20 text-pink-400 border-pink-500/30",
    zaratustra: "bg-purple-500/20 text-purple-400 border-purple-500/30",
    backup:     "bg-teal-500/20 text-teal-400 border-teal-500/30",
    import:     "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
    algorithm:  "bg-orange-500/20 text-orange-400 border-orange-500/30",
    navigate:   "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
}

const STATUS_CONFIG = {
    success: { color: "bg-green-500", icon: CheckCircle2 },
    error:   { color: "bg-red-500",   icon: XCircle },
    pending: { color: "bg-amber-500", icon: AlertCircle },
} as const

/* ── Zustand Store ──────────────────────────────────────────── */

interface ActivityState {
    entries: ActivityEntry[]
    push: (entry: ActivityEntry) => void
    pushMany: (entries: ActivityEntry[]) => void
    clear: () => void
}

export const useActivityStore = create<ActivityState>()(
    persist(
        (set) => ({
            entries: [],
            push: (entry) =>
                set((state) => ({
                    entries: [entry, ...state.entries].slice(0, MAX_ENTRIES),
                })),
            pushMany: (newEntries) =>
                set((state) => {
                    const existingIds = new Set(state.entries.map((e) => e.id))
                    const unique = newEntries.filter((e) => !existingIds.has(e.id))
                    if (unique.length === 0) return state
                    return {
                        entries: [...unique, ...state.entries]
                            .sort((a, b) => b.timestamp - a.timestamp)
                            .slice(0, MAX_ENTRIES),
                    }
                }),
            clear: () => set({ entries: [] }),
        }),
        { name: STORAGE_KEY },
    ),
)

/* ── Log Parser ─────────────────────────────────────────────── */

function parseLogLine(line: string): ActivityEntry | null {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith("#")) return null

    const ts = Date.now()
    const id = `log-${ts}-${Math.random().toString(36).slice(2, 9)}`

    const lower = trimmed.toLowerCase()

    let type: ActivityType = "query"
    if (lower.includes("insert") || lower.includes("created node")) type = "insert"
    else if (lower.includes("update") || lower.includes("merge")) type = "update"
    else if (lower.includes("delete") || lower.includes("removed")) type = "delete"
    else if (lower.includes("sleep")) type = "sleep"
    else if (lower.includes("dream")) type = "dream"
    else if (lower.includes("zaratustra")) type = "zaratustra"
    else if (lower.includes("backup") || lower.includes("snapshot")) type = "backup"
    else if (lower.includes("import") || lower.includes("batch")) type = "import"
    else if (lower.includes("algo") || lower.includes("pagerank") || lower.includes("louvain")) type = "algorithm"
    else if (lower.includes("navigate") || lower.includes("geodesic") || lower.includes("path")) type = "navigate"

    const status: ActivityEntry["status"] = lower.includes("error") || lower.includes("fail")
        ? "error"
        : lower.includes("pending") || lower.includes("running")
            ? "pending"
            : "success"

    const collMatch = trimmed.match(/collection[=: ]+["']?(\w+)["']?/i)
    const durMatch = trimmed.match(/(\d+(?:\.\d+)?)\s*ms/i)

    return {
        id,
        timestamp: ts,
        type,
        description: trimmed.length > 200 ? trimmed.slice(0, 197) + "..." : trimmed,
        collection: collMatch?.[1],
        duration_ms: durMatch ? parseFloat(durMatch[1]) : undefined,
        status,
    }
}

/* ── Temporal Grouping ──────────────────────────────────────── */

type TemporalGroup = "Today" | "Yesterday" | "This Week" | "Older"

function getTemporalGroup(timestamp: number): TemporalGroup {
    const date = new Date(timestamp)
    if (isToday(date)) return "Today"
    if (isYesterday(date)) return "Yesterday"
    if (isThisWeek(date)) return "This Week"
    return "Older"
}

function groupByTime(entries: ActivityEntry[]): Map<TemporalGroup, ActivityEntry[]> {
    const groups = new Map<TemporalGroup, ActivityEntry[]>([
        ["Today", []],
        ["Yesterday", []],
        ["This Week", []],
        ["Older", []],
    ])
    for (const entry of entries) {
        const group = getTemporalGroup(entry.timestamp)
        groups.get(group)!.push(entry)
    }
    return groups
}

/* ── Components ─────────────────────────────────────────────── */

function StatsSummary({ entries }: { entries: ActivityEntry[] }) {
    const stats = useMemo(() => {
        const total = entries.length
        const success = entries.filter((e) => e.status === "success").length
        const rate = total > 0 ? ((success / total) * 100).toFixed(1) : "0.0"

        const collectionCounts = new Map<string, number>()
        for (const e of entries) {
            if (e.collection) {
                collectionCounts.set(e.collection, (collectionCounts.get(e.collection) ?? 0) + 1)
            }
        }
        let mostActive = "-"
        let maxCount = 0
        for (const [name, count] of collectionCounts) {
            if (count > maxCount) { mostActive = name; maxCount = count }
        }

        return { total, rate, mostActive }
    }, [entries])

    return (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <Card className="border-border/50 bg-card/50">
                <CardContent className="pt-4 pb-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-purple-500/20">
                            <BarChart3 className="h-5 w-5 text-purple-400" />
                        </div>
                        <div>
                            <p className="text-xs text-muted-foreground font-mono uppercase tracking-wider">Total Ops</p>
                            <p className="text-2xl font-bold font-mono text-foreground">{stats.total.toLocaleString()}</p>
                        </div>
                    </div>
                </CardContent>
            </Card>
            <Card className="border-border/50 bg-card/50">
                <CardContent className="pt-4 pb-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-green-500/20">
                            <CheckCircle2 className="h-5 w-5 text-green-400" />
                        </div>
                        <div>
                            <p className="text-xs text-muted-foreground font-mono uppercase tracking-wider">Success Rate</p>
                            <p className="text-2xl font-bold font-mono text-foreground">{stats.rate}%</p>
                        </div>
                    </div>
                </CardContent>
            </Card>
            <Card className="border-border/50 bg-card/50">
                <CardContent className="pt-4 pb-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-cyan-500/20">
                            <Terminal className="h-5 w-5 text-cyan-400" />
                        </div>
                        <div>
                            <p className="text-xs text-muted-foreground font-mono uppercase tracking-wider">Most Active</p>
                            <p className="text-lg font-bold font-mono text-foreground truncate max-w-[140px]">{stats.mostActive}</p>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}

function ActivityCard({ entry }: { entry: ActivityEntry }) {
    const Icon = TYPE_ICONS[entry.type]
    const StatusIcon = STATUS_CONFIG[entry.status].icon

    return (
        <div className="group flex items-start gap-3 p-3 rounded-lg border border-border/30 bg-card/30 hover:bg-card/60 hover:border-border/50 transition-all duration-200">
            {/* Icon */}
            <div className={cn(
                "mt-0.5 flex-shrink-0 p-2 rounded-md border",
                TYPE_COLORS[entry.type],
            )}>
                <Icon className="h-4 w-4" />
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0 space-y-1.5">
                <div className="flex items-center gap-2 flex-wrap">
                    <Badge variant="outline" className={cn("text-[10px] font-mono uppercase", TYPE_COLORS[entry.type])}>
                        {entry.type}
                    </Badge>
                    {entry.collection && (
                        <Badge variant="outline" className="text-[10px] font-mono text-muted-foreground border-border/50">
                            {entry.collection}
                        </Badge>
                    )}
                    {entry.duration_ms != null && (
                        <Badge variant="outline" className="text-[10px] font-mono text-muted-foreground border-border/50">
                            <Clock className="h-3 w-3 mr-1" />
                            {entry.duration_ms < 1000
                                ? `${entry.duration_ms.toFixed(0)}ms`
                                : `${(entry.duration_ms / 1000).toFixed(2)}s`}
                        </Badge>
                    )}
                </div>
                <p className="text-sm text-foreground/90 font-mono leading-relaxed break-words">
                    {entry.description}
                </p>
                {entry.nodeId && (
                    <p className="text-[11px] text-muted-foreground font-mono truncate">
                        node: {entry.nodeId}
                    </p>
                )}
            </div>

            {/* Right: status + time */}
            <div className="flex-shrink-0 flex flex-col items-end gap-1.5">
                <div className="flex items-center gap-1.5">
                    <StatusIcon className={cn(
                        "h-3.5 w-3.5",
                        entry.status === "success" && "text-green-500",
                        entry.status === "error" && "text-red-500",
                        entry.status === "pending" && "text-amber-500",
                    )} />
                    <span className={cn(
                        "h-2 w-2 rounded-full",
                        STATUS_CONFIG[entry.status].color,
                    )} />
                </div>
                <span className="text-[11px] text-muted-foreground font-mono whitespace-nowrap">
                    {formatDistanceToNow(new Date(entry.timestamp), { addSuffix: true })}
                </span>
            </div>
        </div>
    )
}

function TemporalGroupSection({ label, entries }: { label: string; entries: ActivityEntry[] }) {
    if (entries.length === 0) return null
    return (
        <div className="space-y-2">
            <div className="sticky top-0 z-10 bg-background/80 backdrop-blur-sm py-2 border-b border-border/20">
                <h3 className="text-xs font-mono uppercase tracking-widest text-muted-foreground">
                    {label}
                    <span className="ml-2 text-foreground/60">{entries.length}</span>
                </h3>
            </div>
            <div className="space-y-1.5">
                {entries.map((entry) => (
                    <ActivityCard key={entry.id} entry={entry} />
                ))}
            </div>
        </div>
    )
}

function LoadingSkeleton() {
    return (
        <div className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="flex items-start gap-3 p-3 rounded-lg border border-border/30">
                    <Skeleton className="h-8 w-8 rounded-md" />
                    <div className="flex-1 space-y-2">
                        <div className="flex gap-2">
                            <Skeleton className="h-4 w-16" />
                            <Skeleton className="h-4 w-20" />
                        </div>
                        <Skeleton className="h-4 w-3/4" />
                    </div>
                    <Skeleton className="h-4 w-20" />
                </div>
            ))}
        </div>
    )
}

/* ── Main Page ──────────────────────────────────────────────── */

export default function ActivityPage() {
    const { entries, pushMany, clear } = useActivityStore()
    const [searchQuery, setSearchQuery] = useState("")
    const [activeTypes, setActiveTypes] = useState<Set<ActivityType>>(new Set(ACTIVITY_TYPES))
    const [activeTab, setActiveTab] = useState("all")
    const scrollRef = useRef<HTMLDivElement>(null)
    const prevCountRef = useRef(entries.length)

    /* Poll server logs */
    const { isLoading } = useQuery({
        queryKey: ["activity-logs"],
        queryFn: async () => {
            try {
                const res = await api.get("/logs")
                const data = res.data
                const lines: string[] = typeof data === "string"
                    ? data.split("\n")
                    : Array.isArray(data)
                        ? data.map((l: unknown) => (typeof l === "string" ? l : JSON.stringify(l)))
                        : []

                const parsed = lines
                    .map(parseLogLine)
                    .filter((e): e is ActivityEntry => e !== null)

                if (parsed.length > 0) pushMany(parsed)
                return parsed
            } catch {
                return []
            }
        },
        refetchInterval: POLL_INTERVAL,
        refetchIntervalInBackground: false,
    })

    /* Auto-scroll on new entries */
    useEffect(() => {
        if (entries.length > prevCountRef.current && scrollRef.current) {
            scrollRef.current.scrollTo({ top: 0, behavior: "smooth" })
        }
        prevCountRef.current = entries.length
    }, [entries.length])

    /* Toggle type filter */
    const toggleType = useCallback((type: ActivityType) => {
        setActiveTypes((prev) => {
            const next = new Set(prev)
            if (next.has(type)) {
                next.delete(type)
            } else {
                next.add(type)
            }
            return next
        })
    }, [])

    /* Filter entries */
    const filtered = useMemo(() => {
        let result = entries.filter((e) => activeTypes.has(e.type))
        if (searchQuery.trim()) {
            const q = searchQuery.toLowerCase()
            result = result.filter(
                (e) =>
                    e.description.toLowerCase().includes(q) ||
                    e.collection?.toLowerCase().includes(q) ||
                    e.type.includes(q) ||
                    e.nodeId?.toLowerCase().includes(q),
            )
        }
        return result
    }, [entries, activeTypes, searchQuery])

    /* Group for the "grouped" tab */
    const grouped = useMemo(() => groupByTime(filtered), [filtered])

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold font-mono tracking-tight">Activity Feed</h1>
                    <p className="text-sm text-muted-foreground font-mono mt-1">
                        Operation history and real-time event log
                    </p>
                </div>
                <Button
                    variant="outline"
                    size="sm"
                    onClick={clear}
                    className="text-red-400 border-red-500/30 hover:bg-red-500/10 hover:text-red-300 font-mono"
                    disabled={entries.length === 0}
                >
                    <Trash2 className="h-4 w-4 mr-1.5" />
                    Clear History
                </Button>
            </div>

            {/* Stats Summary */}
            <StatsSummary entries={entries} />

            {/* Search + Type Filters */}
            <Card className="border-border/50 bg-card/50">
                <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-mono uppercase tracking-wider text-muted-foreground">
                        Filters
                    </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                    {/* Search */}
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                        <Input
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Search activities..."
                            className="pl-10 font-mono text-sm bg-background/50 border-border/50"
                        />
                        {searchQuery && (
                            <button
                                onClick={() => setSearchQuery("")}
                                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                            >
                                <X className="h-4 w-4" />
                            </button>
                        )}
                    </div>

                    {/* Type toggles */}
                    <div className="flex flex-wrap gap-1.5">
                        {ACTIVITY_TYPES.map((type) => {
                            const Icon = TYPE_ICONS[type]
                            const active = activeTypes.has(type)
                            return (
                                <Button
                                    key={type}
                                    variant="outline"
                                    size="sm"
                                    onClick={() => toggleType(type)}
                                    className={cn(
                                        "h-7 px-2.5 text-[11px] font-mono uppercase transition-all",
                                        active
                                            ? TYPE_COLORS[type]
                                            : "text-muted-foreground/50 border-border/20 opacity-50",
                                    )}
                                >
                                    <Icon className="h-3 w-3 mr-1" />
                                    {type}
                                </Button>
                            )
                        })}
                    </div>
                </CardContent>
            </Card>

            {/* Activity Feed */}
            <Card className="border-border/50 bg-card/50">
                <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                        <CardTitle className="text-sm font-mono uppercase tracking-wider text-muted-foreground">
                            Events
                            <span className="ml-2 text-foreground/60">
                                {filtered.length}/{entries.length}
                            </span>
                        </CardTitle>
                    </div>
                </CardHeader>
                <CardContent>
                    <Tabs value={activeTab} onValueChange={setActiveTab}>
                        <TabsList className="mb-4 bg-background/50">
                            <TabsTrigger value="all" className="font-mono text-xs">
                                All
                            </TabsTrigger>
                            <TabsTrigger value="grouped" className="font-mono text-xs">
                                Grouped
                            </TabsTrigger>
                        </TabsList>

                        <TabsContent value="all">
                            <ScrollArea className="h-[600px] pr-3" ref={scrollRef}>
                                {isLoading && entries.length === 0 ? (
                                    <LoadingSkeleton />
                                ) : filtered.length === 0 ? (
                                    <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                                        <Terminal className="h-12 w-12 mb-4 opacity-30" />
                                        <p className="font-mono text-sm">No activity recorded yet</p>
                                        <p className="font-mono text-xs mt-1 opacity-60">
                                            Operations will appear here in real time
                                        </p>
                                    </div>
                                ) : (
                                    <div className="space-y-1.5">
                                        {filtered.map((entry) => (
                                            <ActivityCard key={entry.id} entry={entry} />
                                        ))}
                                    </div>
                                )}
                            </ScrollArea>
                        </TabsContent>

                        <TabsContent value="grouped">
                            <ScrollArea className="h-[600px] pr-3" ref={scrollRef}>
                                {isLoading && entries.length === 0 ? (
                                    <LoadingSkeleton />
                                ) : filtered.length === 0 ? (
                                    <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                                        <Terminal className="h-12 w-12 mb-4 opacity-30" />
                                        <p className="font-mono text-sm">No activity recorded yet</p>
                                        <p className="font-mono text-xs mt-1 opacity-60">
                                            Operations will appear here in real time
                                        </p>
                                    </div>
                                ) : (
                                    <div className="space-y-6">
                                        {(["Today", "Yesterday", "This Week", "Older"] as const).map((group) => (
                                            <TemporalGroupSection
                                                key={group}
                                                label={group}
                                                entries={grouped.get(group) ?? []}
                                            />
                                        ))}
                                    </div>
                                )}
                            </ScrollArea>
                        </TabsContent>
                    </Tabs>
                </CardContent>
            </Card>
        </div>
    )
}
