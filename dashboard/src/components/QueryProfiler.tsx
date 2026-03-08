// QueryProfiler.tsx — Query execution profiler with stacked bar timing breakdown
// Shows phase-level performance analysis for NQL queries

import { useMemo } from "react"
import { Activity, Clock, Database, Hash } from "lucide-react"

import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface QueryProfile {
  query: string
  totalMs: number
  phases: Array<{
    name: string
    durationMs: number
    percentage: number
  }>
  timestamp: number
  collection?: string
  resultCount?: number
}

export interface QueryProfilerProps {
  profiles: QueryProfile[]
  className?: string
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PHASE_COLORS: Record<string, string> = {
  Parse: "#8b5cf6",
  Plan: "#3b82f6",
  Execute: "#00ff66",
  Serialize: "#f59e0b",
  Network: "#94a3b8",
}

const PHASE_ORDER = ["Parse", "Plan", "Execute", "Serialize", "Network"]

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function relativeTimestamp(ts: number): string {
  const diff = Date.now() - ts
  const secs = Math.floor(diff / 1000)
  if (secs < 5) return "just now"
  if (secs < 60) return `${secs}s ago`
  const mins = Math.floor(secs / 60)
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  return `${hrs}h ago`
}

function durationBadgeColor(ms: number): string {
  if (ms < 10) return "text-green-400 border-green-800/50"
  if (ms < 50) return "text-cyan-400 border-cyan-800/50"
  if (ms < 200) return "text-amber-400 border-amber-800/50"
  return "text-red-400 border-red-800/50"
}

function truncateQuery(q: string, maxLen = 60): string {
  const cleaned = q.replace(/\s+/g, " ").trim()
  return cleaned.length > maxLen ? cleaned.slice(0, maxLen) + "\u2026" : cleaned
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function QueryProfiler({ profiles, className }: QueryProfilerProps) {
  const recent = profiles.slice(-10).reverse()

  const avgMs = useMemo(() => {
    if (recent.length === 0) return 0
    return recent.reduce((sum, p) => sum + p.totalMs, 0) / recent.length
  }, [recent])

  const phaseAvg = useMemo(() => {
    if (recent.length === 0) return []
    const sums: Record<string, number> = {}
    for (const p of recent) {
      for (const phase of p.phases) {
        sums[phase.name] = (sums[phase.name] || 0) + phase.durationMs
      }
    }
    return PHASE_ORDER.filter((name) => sums[name]).map((name) => ({
      name,
      avgMs: sums[name] / recent.length,
    }))
  }, [recent])

  if (recent.length === 0) {
    return (
      <Card className={cn("border-slate-800 bg-slate-950/80 p-4", className)}>
        <div className="flex items-center gap-2 text-slate-500 text-xs">
          <Activity className="h-4 w-4" />
          <span>No queries profiled yet</span>
        </div>
      </Card>
    )
  }

  return (
    <Card className={cn("border-purple-900/40 bg-slate-950/80 p-4", className)}>
      {/* Summary header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-purple-400" />
          <span className="text-xs font-medium text-slate-300">Query Profiler</span>
          <Badge variant="outline" className="border-slate-700 text-slate-500 text-[10px] font-mono px-1.5 py-0">
            {recent.length} queries
          </Badge>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right">
            <div className="text-[10px] text-slate-600 uppercase tracking-wider">Avg Response</div>
            <div className={cn("text-sm font-mono font-semibold", avgMs < 50 ? "text-green-400" : avgMs < 200 ? "text-amber-400" : "text-red-400")}>
              {avgMs.toFixed(1)}ms
            </div>
          </div>
        </div>
      </div>

      {/* Phase legend */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mb-3 pb-3 border-b border-slate-800">
        {phaseAvg.map(({ name, avgMs: ms }) => (
          <div key={name} className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-sm" style={{ backgroundColor: PHASE_COLORS[name] || "#666" }} />
            <span className="text-[10px] text-slate-500">{name}</span>
            <span className="text-[10px] font-mono text-slate-600">{ms.toFixed(1)}ms</span>
          </div>
        ))}
      </div>

      {/* Query list */}
      <div className="space-y-2">
        {recent.map((profile, idx) => (
          <div key={`${profile.timestamp}-${idx}`} className="group rounded border border-slate-800/80 bg-slate-900/40 p-2.5 hover:border-purple-800/40 transition-colors">
            {/* Query text + badges */}
            <div className="flex items-start justify-between gap-2 mb-2">
              <code className="text-[11px] font-mono text-slate-400 leading-tight break-all">
                {truncateQuery(profile.query)}
              </code>
              <div className="flex items-center gap-1.5 shrink-0">
                {profile.resultCount !== undefined && (
                  <Badge variant="outline" className="border-slate-700 text-slate-500 text-[10px] font-mono px-1 py-0 gap-0.5">
                    <Hash className="h-2.5 w-2.5" />
                    {profile.resultCount}
                  </Badge>
                )}
                <Badge
                  variant="outline"
                  className={cn("text-[10px] font-mono px-1.5 py-0", durationBadgeColor(profile.totalMs))}
                >
                  {profile.totalMs.toFixed(1)}ms
                </Badge>
              </div>
            </div>

            {/* Stacked bar */}
            <div className="h-3 w-full rounded-sm bg-slate-900 overflow-hidden flex">
              {profile.phases
                .filter((p) => p.percentage > 0)
                .sort((a, b) => PHASE_ORDER.indexOf(a.name) - PHASE_ORDER.indexOf(b.name))
                .map((phase) => (
                  <div
                    key={phase.name}
                    className="h-full relative group/phase"
                    style={{
                      width: `${Math.max(phase.percentage, 1)}%`,
                      backgroundColor: PHASE_COLORS[phase.name] || "#666",
                      opacity: 0.8,
                    }}
                    title={`${phase.name}: ${phase.durationMs.toFixed(1)}ms (${phase.percentage.toFixed(0)}%)`}
                  />
                ))}
            </div>

            {/* Phase breakdown row */}
            <div className="flex items-center justify-between mt-1.5">
              <div className="flex items-center gap-2">
                {profile.phases
                  .filter((p) => p.durationMs > 0)
                  .sort((a, b) => PHASE_ORDER.indexOf(a.name) - PHASE_ORDER.indexOf(b.name))
                  .map((phase) => (
                    <span key={phase.name} className="text-[9px] font-mono" style={{ color: PHASE_COLORS[phase.name] || "#666" }}>
                      {phase.name[0]}:{phase.durationMs.toFixed(0)}
                    </span>
                  ))}
              </div>
              <div className="flex items-center gap-1.5 text-[10px] text-slate-600">
                {profile.collection && (
                  <span className="flex items-center gap-0.5">
                    <Database className="h-2.5 w-2.5" />
                    {profile.collection}
                  </span>
                )}
                <span className="flex items-center gap-0.5">
                  <Clock className="h-2.5 w-2.5" />
                  {relativeTimestamp(profile.timestamp)}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}

export default QueryProfiler
