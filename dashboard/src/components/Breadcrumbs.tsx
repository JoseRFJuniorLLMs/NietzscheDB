// Breadcrumbs.tsx — Route-aware breadcrumb navigation for NietzscheDB Dashboard
// Auto-detects current path and renders clickable breadcrumb trail

import { NavLink, useLocation, useSearchParams } from "react-router-dom"
import { ChevronRight, Home } from "lucide-react"

// ---------------------------------------------------------------------------
// Route Map
// ---------------------------------------------------------------------------

const ROUTE_NAMES: Record<string, string> = {
  "/": "Overview",
  "/collections": "Collections",
  "/nodes": "Cluster Nodes",
  "/explorer": "Data Explorer",
  "/graph": "Graph Explorer",
  "/settings": "Settings",
  "/query": "NQL Console",
  "/builder": "Query Builder",
  "/algorithms": "Algorithms",
  "/agency": "Agency",
  "/reasoning": "Reasoning",
  "/sleep": "Sleep & Dream",
  "/backup": "Backup & Export",
  "/monitoring": "Monitoring",
  "/schemas": "Schema Manager",
  "/sensory": "Sensory Layer",
  "/activity": "Activity Feed",
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function Breadcrumbs() {
  const { pathname } = useLocation()
  const [searchParams] = useSearchParams()

  const pageName = ROUTE_NAMES[pathname] || pathname.replace("/", "").replace(/-/g, " ")
  const nodeParam = searchParams.get("node")
  const isHome = pathname === "/"

  return (
    <nav className="flex items-center gap-1 text-xs text-slate-500" aria-label="Breadcrumb">
      <NavLink
        to="/"
        className="flex items-center gap-1 text-slate-500 hover:text-purple-400 transition-colors"
      >
        <Home className="h-3 w-3" />
        <span className="hidden sm:inline">Home</span>
      </NavLink>

      {!isHome && (
        <>
          <ChevronRight className="h-3 w-3 text-slate-700 shrink-0" />
          <NavLink
            to={pathname}
            className={({ isActive }) =>
              isActive && !nodeParam
                ? "font-medium text-slate-300"
                : "text-slate-500 hover:text-purple-400 transition-colors"
            }
          >
            {pageName}
          </NavLink>
        </>
      )}

      {nodeParam && (
        <>
          <ChevronRight className="h-3 w-3 text-slate-700 shrink-0" />
          <span className="font-mono text-slate-400 truncate max-w-[120px]">
            {nodeParam.length > 16 ? nodeParam.slice(0, 16) + "\u2026" : nodeParam}
          </span>
        </>
      )}
    </nav>
  )
}

export default Breadcrumbs
