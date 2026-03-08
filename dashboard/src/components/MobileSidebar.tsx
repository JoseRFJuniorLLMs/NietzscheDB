// MobileSidebar.tsx — Mobile-responsive sidebar with hamburger menu
// Slides in from left on mobile, same nav structure as DashboardLayout

import { useCallback, useEffect, useState } from "react"
import { NavLink } from "react-router-dom"
import {
    Activity,
    Archive,
    AudioLines,
    Brain,
    Database,
    FileJson,
    GitBranch,
    LayoutDashboard,
    Lightbulb,
    Menu,
    Moon,
    Network,
    Search,
    Settings,
    Terminal,
    Wand2,
    X,
} from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

// ---------------------------------------------------------------------------
// Nav structure (mirrors DashboardLayout)
// ---------------------------------------------------------------------------

interface NavEntry {
    to: string
    icon: React.ElementType
    label: string
    badge?: string
}

const mainNav: NavEntry[] = [
    { to: "/", icon: LayoutDashboard, label: "Overview" },
    { to: "/collections", icon: Database, label: "Collections" },
    { to: "/nodes", icon: Network, label: "Cluster Nodes" },
    { to: "/explorer", icon: Search, label: "Data Explorer" },
    { to: "/graph", icon: Network, label: "Graph Explorer", badge: "New" },
    { to: "/settings", icon: Settings, label: "Settings" },
]

const advancedNav: NavEntry[] = [
    { to: "/query", icon: Terminal, label: "NQL Console", badge: "AI" },
    { to: "/builder", icon: Wand2, label: "Query Builder", badge: "New" },
    { to: "/algorithms", icon: GitBranch, label: "Algorithms" },
    { to: "/agency", icon: Brain, label: "Agency" },
    { to: "/reasoning", icon: Lightbulb, label: "Reasoning", badge: "New" },
    { to: "/sleep", icon: Moon, label: "Sleep & Dream" },
    { to: "/backup", icon: Archive, label: "Backup & Export" },
    { to: "/monitoring", icon: Activity, label: "Monitoring" },
]

const sdkNav: NavEntry[] = [
    { to: "/schemas", icon: FileJson, label: "Schema Manager" },
    { to: "/sensory", icon: AudioLines, label: "Sensory Layer", badge: "New" },
]

// ---------------------------------------------------------------------------
// NavItem
// ---------------------------------------------------------------------------

function MobileNavItem({ to, icon: Icon, label, badge, onClick }: NavEntry & { onClick: () => void }) {
    return (
        <NavLink
            to={to}
            onClick={onClick}
            className={({ isActive }) =>
                cn(
                    "flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground",
                    isActive ? "bg-accent text-accent-foreground" : "text-muted-foreground",
                )
            }
        >
            <Icon className="h-4 w-4" />
            <span>{label}</span>
            {badge && (
                <span className="ml-auto text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded font-semibold">
                    {badge}
                </span>
            )}
        </NavLink>
    )
}

// ---------------------------------------------------------------------------
// MobileSidebar
// ---------------------------------------------------------------------------

export function MobileSidebar() {
    const [open, setOpen] = useState(false)

    const close = useCallback(() => setOpen(false), [])

    // Close on Escape
    useEffect(() => {
        if (!open) return
        const handler = (e: KeyboardEvent) => {
            if (e.key === "Escape") close()
        }
        window.addEventListener("keydown", handler)
        return () => window.removeEventListener("keydown", handler)
    }, [open, close])

    // Lock body scroll when open
    useEffect(() => {
        if (open) {
            document.body.style.overflow = "hidden"
        } else {
            document.body.style.overflow = ""
        }
        return () => { document.body.style.overflow = "" }
    }, [open])

    return (
        <>
            {/* Hamburger button — visible only on mobile */}
            <Button
                variant="ghost"
                size="icon"
                className="md:hidden fixed top-4 left-4 z-40 h-9 w-9 text-slate-300 hover:text-white bg-slate-900/80 backdrop-blur border border-slate-700/50"
                onClick={() => setOpen(true)}
                aria-label="Open navigation menu"
            >
                <Menu className="h-5 w-5" />
            </Button>

            {/* Backdrop */}
            <div
                className={cn(
                    "fixed inset-0 z-40 bg-black/60 backdrop-blur-sm transition-opacity duration-300 md:hidden",
                    open ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none",
                )}
                onClick={close}
                aria-hidden="true"
            />

            {/* Sidebar panel */}
            <aside
                className={cn(
                    "fixed inset-y-0 left-0 z-50 w-72 bg-card border-r border-border flex flex-col transition-transform duration-300 ease-in-out md:hidden",
                    open ? "translate-x-0" : "-translate-x-full",
                )}
            >
                {/* Header */}
                <div className="flex items-center justify-between p-5 border-b border-border/50">
                    <div className="flex items-center gap-3">
                        <div className="h-8 w-8 flex items-center justify-center font-bold text-primary text-lg">
                            [N]
                        </div>
                        <span className="text-lg font-bold tracking-tight">NietzscheDB</span>
                    </div>
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-slate-400 hover:text-white"
                        onClick={close}
                        aria-label="Close navigation menu"
                    >
                        <X className="h-4 w-4" />
                    </Button>
                </div>

                {/* Nav */}
                <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
                    {mainNav.map((item) => (
                        <MobileNavItem key={item.to} {...item} onClick={close} />
                    ))}

                    <div className="my-3 border-t border-border/30" />
                    <div className="px-3 mb-1">
                        <span className="text-[10px] uppercase tracking-widest text-muted-foreground/50 font-semibold">
                            Advanced
                        </span>
                    </div>
                    {advancedNav.map((item) => (
                        <MobileNavItem key={item.to} {...item} onClick={close} />
                    ))}

                    <div className="my-3 border-t border-border/30" />
                    <div className="px-3 mb-1">
                        <span className="text-[10px] uppercase tracking-widest text-muted-foreground/50 font-semibold">
                            SDK & Data
                        </span>
                    </div>
                    {sdkNav.map((item) => (
                        <MobileNavItem key={item.to} {...item} onClick={close} />
                    ))}
                </nav>

                {/* Footer */}
                <div className="p-4 border-t border-border/50">
                    <div className="text-xs text-muted-foreground">
                        <p>Version 2.1.0</p>
                        <p className="opacity-50">38 crates &middot; Hyperbolic Engine</p>
                    </div>
                </div>
            </aside>
        </>
    )
}

export default MobileSidebar
