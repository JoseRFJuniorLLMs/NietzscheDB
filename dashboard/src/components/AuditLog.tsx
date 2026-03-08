// AuditLog.tsx — Audit log viewer for NietzscheDB Dashboard
// Shows who did what and when, with search, filtering, pagination, and CSV export

import { useCallback, useMemo, useState } from "react"
import { format, formatDistanceToNow } from "date-fns"
import {
    ChevronLeft,
    ChevronRight,
    Download,
    Filter,
    RefreshCw,
    Search,
} from "lucide-react"

import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AuditEntry {
    id: string
    timestamp: number
    action: string
    actor: string
    target?: string
    details?: string
    status: "success" | "error"
}

export interface AuditLogProps {
    entries: AuditEntry[]
    onRefresh?: () => void
    className?: string
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PAGE_SIZE = 20

const ACTION_COLORS: Record<string, string> = {
    CREATE_NODE: "#00ff66",
    DELETE_NODE: "#ff4444",
    UPDATE_NODE: "#00f0ff",
    INSERT_EDGE: "#22c55e",
    DELETE_EDGE: "#f59e0b",
    RUN_ALGORITHM: "#ff00ff",
    SLEEP_CYCLE: "#8b5cf6",
    DREAM: "#a855f7",
    BACKUP: "#3b82f6",
    RESTORE: "#06b6d4",
    QUERY: "#94a3b8",
    ZARATUSTRA: "#ffd700",
}

const ACTOR_COLORS: Record<string, string> = {
    system: "#8b5cf6",
    api: "#00f0ff",
    dashboard: "#00ff66",
    daemon: "#ffd700",
}

function getActionColor(action: string): string {
    return ACTION_COLORS[action] ?? "#64748b"
}

function getActorColor(actor: string): string {
    return ACTOR_COLORS[actor] ?? "#64748b"
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function AuditLog({ entries, onRefresh, className }: AuditLogProps) {
    const [search, setSearch] = useState("")
    const [filterActor, setFilterActor] = useState<string | null>(null)
    const [filterAction, setFilterAction] = useState<string | null>(null)
    const [filterStatus, setFilterStatus] = useState<"success" | "error" | null>(null)
    const [page, setPage] = useState(0)
    const [showFilters, setShowFilters] = useState(false)

    // Unique actors and actions for filter chips
    const uniqueActors = useMemo(() => Array.from(new Set(entries.map((e) => e.actor))).sort(), [entries])
    const uniqueActions = useMemo(() => Array.from(new Set(entries.map((e) => e.action))).sort(), [entries])

    // Filtered entries
    const filtered = useMemo(() => {
        const q = search.toLowerCase().trim()
        return entries.filter((e) => {
            if (filterActor && e.actor !== filterActor) return false
            if (filterAction && e.action !== filterAction) return false
            if (filterStatus && e.status !== filterStatus) return false
            if (q) {
                const hay = `${e.action} ${e.actor} ${e.target ?? ""} ${e.details ?? ""} ${e.id}`.toLowerCase()
                if (!hay.includes(q)) return false
            }
            return true
        })
    }, [entries, search, filterActor, filterAction, filterStatus])

    // Pagination
    const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
    const safePageNum = Math.min(page, totalPages - 1)
    const pageEntries = filtered.slice(safePageNum * PAGE_SIZE, (safePageNum + 1) * PAGE_SIZE)

    // Reset page on filter change
    const setSearchAndReset = (v: string) => { setSearch(v); setPage(0) }
    const toggleActor = (a: string) => { setFilterActor((prev) => (prev === a ? null : a)); setPage(0) }
    const toggleAction = (a: string) => { setFilterAction((prev) => (prev === a ? null : a)); setPage(0) }
    const toggleStatus = (s: "success" | "error") => { setFilterStatus((prev) => (prev === s ? null : s)); setPage(0) }

    const clearFilters = () => {
        setSearch("")
        setFilterActor(null)
        setFilterAction(null)
        setFilterStatus(null)
        setPage(0)
    }

    // CSV export
    const exportCSV = useCallback(() => {
        const header = "ID,Timestamp,Action,Actor,Target,Status,Details"
        const rows = filtered.map((e) => {
            const ts = format(new Date(e.timestamp), "yyyy-MM-dd HH:mm:ss")
            const details = (e.details ?? "").replace(/"/g, '""')
            return `"${e.id}","${ts}","${e.action}","${e.actor}","${e.target ?? ""}","${e.status}","${details}"`
        })
        const csv = [header, ...rows].join("\n")
        const blob = new Blob([csv], { type: "text/csv" })
        const url = URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = `nietzsche-audit-${format(new Date(), "yyyyMMdd-HHmmss")}.csv`
        a.click()
        URL.revokeObjectURL(url)
    }, [filtered])

    const hasActiveFilters = filterActor || filterAction || filterStatus || search

    return (
        <Card className={cn("border-purple-900/40 bg-background p-4", className)}>
            {/* Header */}
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-slate-200">Audit Log</span>
                    <Badge
                        variant="outline"
                        className="border-purple-800/50 px-1.5 py-0 text-[10px] font-mono text-purple-400"
                    >
                        {filtered.length} / {entries.length}
                    </Badge>
                </div>

                <div className="flex items-center gap-1.5">
                    <Button
                        variant="ghost"
                        size="sm"
                        className={cn(
                            "h-7 gap-1 px-2 text-xs",
                            showFilters ? "text-purple-400" : "text-slate-400",
                        )}
                        onClick={() => setShowFilters(!showFilters)}
                    >
                        <Filter className="h-3 w-3" />
                        Filters
                    </Button>
                    <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 gap-1 px-2 text-xs text-slate-400 hover:text-slate-200"
                        onClick={exportCSV}
                    >
                        <Download className="h-3 w-3" />
                        CSV
                    </Button>
                    {onRefresh && (
                        <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 gap-1 px-2 text-xs text-slate-400 hover:text-slate-200"
                            onClick={onRefresh}
                        >
                            <RefreshCw className="h-3 w-3" />
                        </Button>
                    )}
                </div>
            </div>

            {/* Search bar */}
            <div className="relative mb-3">
                <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-500" />
                <Input
                    placeholder="Search actions, actors, targets..."
                    value={search}
                    onChange={(e) => setSearchAndReset(e.target.value)}
                    className="h-8 border-slate-700 bg-slate-900 pl-8 text-xs placeholder:text-slate-600"
                />
            </div>

            {/* Filter chips */}
            {showFilters && (
                <div className="mb-3 space-y-2 rounded-md border border-slate-800 bg-slate-950 p-3">
                    {/* Actor filter */}
                    <div>
                        <span className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                            Actor
                        </span>
                        <div className="flex flex-wrap gap-1">
                            {uniqueActors.map((actor) => (
                                <button
                                    key={actor}
                                    onClick={() => toggleActor(actor)}
                                    className={cn(
                                        "rounded-full px-2 py-0.5 text-[10px] font-mono font-medium transition-colors",
                                        filterActor === actor
                                            ? "bg-purple-800/60 text-purple-200"
                                            : "bg-slate-800 text-slate-400 hover:bg-slate-700",
                                    )}
                                >
                                    {actor}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Action filter */}
                    <div>
                        <span className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                            Action
                        </span>
                        <div className="flex flex-wrap gap-1">
                            {uniqueActions.map((action) => (
                                <button
                                    key={action}
                                    onClick={() => toggleAction(action)}
                                    className={cn(
                                        "rounded-full px-2 py-0.5 text-[10px] font-mono font-medium transition-colors",
                                        filterAction === action
                                            ? "bg-purple-800/60 text-purple-200"
                                            : "bg-slate-800 text-slate-400 hover:bg-slate-700",
                                    )}
                                >
                                    {action}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Status filter */}
                    <div>
                        <span className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                            Status
                        </span>
                        <div className="flex gap-1">
                            <button
                                onClick={() => toggleStatus("success")}
                                className={cn(
                                    "rounded-full px-2 py-0.5 text-[10px] font-medium transition-colors",
                                    filterStatus === "success"
                                        ? "bg-emerald-800/60 text-emerald-200"
                                        : "bg-slate-800 text-slate-400 hover:bg-slate-700",
                                )}
                            >
                                success
                            </button>
                            <button
                                onClick={() => toggleStatus("error")}
                                className={cn(
                                    "rounded-full px-2 py-0.5 text-[10px] font-medium transition-colors",
                                    filterStatus === "error"
                                        ? "bg-red-800/60 text-red-200"
                                        : "bg-slate-800 text-slate-400 hover:bg-slate-700",
                                )}
                            >
                                error
                            </button>
                        </div>
                    </div>

                    {hasActiveFilters && (
                        <button
                            onClick={clearFilters}
                            className="text-[10px] text-purple-400 underline underline-offset-2 hover:text-purple-300"
                        >
                            Clear all filters
                        </button>
                    )}
                </div>
            )}

            {/* Table */}
            <ScrollArea className="w-full">
                <Table>
                    <TableHeader>
                        <TableRow className="border-slate-800 hover:bg-transparent">
                            <TableHead className="w-[140px] text-[10px] font-semibold uppercase text-slate-500">Time</TableHead>
                            <TableHead className="text-[10px] font-semibold uppercase text-slate-500">Action</TableHead>
                            <TableHead className="text-[10px] font-semibold uppercase text-slate-500">Actor</TableHead>
                            <TableHead className="text-[10px] font-semibold uppercase text-slate-500">Target</TableHead>
                            <TableHead className="w-[60px] text-[10px] font-semibold uppercase text-slate-500">Status</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {pageEntries.length === 0 && (
                            <TableRow>
                                <TableCell colSpan={5} className="py-8 text-center text-xs text-slate-500">
                                    {entries.length === 0 ? "No audit entries" : "No entries match your filters"}
                                </TableCell>
                            </TableRow>
                        )}
                        {pageEntries.map((entry) => (
                            <TableRow key={entry.id} className="border-slate-800/50 hover:bg-slate-900/50">
                                {/* Time */}
                                <TableCell className="py-2" title={format(new Date(entry.timestamp), "yyyy-MM-dd HH:mm:ss.SSS")}>
                                    <span className="text-[11px] text-slate-400">
                                        {formatDistanceToNow(new Date(entry.timestamp), { addSuffix: true })}
                                    </span>
                                </TableCell>
                                {/* Action */}
                                <TableCell className="py-2">
                                    <Badge
                                        variant="outline"
                                        className="border-none px-1.5 py-0 text-[10px] font-mono"
                                        style={{
                                            backgroundColor: getActionColor(entry.action) + "18",
                                            color: getActionColor(entry.action),
                                        }}
                                    >
                                        {entry.action}
                                    </Badge>
                                </TableCell>
                                {/* Actor */}
                                <TableCell className="py-2">
                                    <Badge
                                        variant="outline"
                                        className="border-none px-1.5 py-0 text-[10px] font-mono"
                                        style={{
                                            backgroundColor: getActorColor(entry.actor) + "18",
                                            color: getActorColor(entry.actor),
                                        }}
                                    >
                                        {entry.actor}
                                    </Badge>
                                </TableCell>
                                {/* Target */}
                                <TableCell className="py-2">
                                    {entry.target ? (
                                        <span className="font-mono text-[11px] text-slate-400">{entry.target}</span>
                                    ) : (
                                        <span className="text-[11px] text-slate-600">--</span>
                                    )}
                                </TableCell>
                                {/* Status */}
                                <TableCell className="py-2">
                                    <div className="flex items-center gap-1.5">
                                        <span
                                            className={cn(
                                                "inline-block h-2 w-2 rounded-full",
                                                entry.status === "success" ? "bg-emerald-500" : "bg-red-500",
                                            )}
                                        />
                                        <span className="sr-only">{entry.status}</span>
                                    </div>
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </ScrollArea>

            {/* Pagination */}
            <div className="mt-3 flex items-center justify-between border-t border-slate-800 pt-3">
                <span className="text-[11px] text-slate-500">
                    Page {safePageNum + 1} of {totalPages}
                </span>
                <div className="flex items-center gap-1">
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-slate-400"
                        disabled={safePageNum === 0}
                        onClick={() => setPage((p) => Math.max(0, p - 1))}
                    >
                        <ChevronLeft className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-slate-400"
                        disabled={safePageNum >= totalPages - 1}
                        onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                    >
                        <ChevronRight className="h-3.5 w-3.5" />
                    </Button>
                </div>
            </div>
        </Card>
    )
}

export default AuditLog
