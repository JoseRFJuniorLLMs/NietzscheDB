import { useQuery } from "@tanstack/react-query"
import { useMemo } from "react"
import { api, fetchStatus, getMetrics } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
    Database, Server, Zap, GitBranch, Clock, Activity,
    Terminal, Brain, Moon, Archive, Search, Cpu,
    Layers, Radio, Shield, Box, Workflow, Network,
} from "lucide-react"
import { Skeleton } from "@/components/ui/skeleton"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useState } from "react"
import { Link } from "react-router-dom"

function parsePrometheusValue(raw: string, metricName: string): number | null {
    if (!raw) return null
    const lines = raw.split("\n")
    for (const line of lines) {
        if (line.startsWith(metricName + " ")) {
            return parseFloat(line.split(" ")[1])
        }
    }
    return null
}

function formatUptime(seconds: number): string {
    const d = Math.floor(seconds / 86400)
    const h = Math.floor((seconds % 86400) / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    if (d > 0) return `${d}d ${h}h ${m}m`
    if (h > 0) return `${h}h ${m}m`
    return `${m}m`
}

/* ── Platform Capabilities Data ─────────────────────────────── */
const PLATFORM_STATS = [
    { label: "Rust Crates", value: "38", icon: Box, color: "text-orange-400" },
    { label: "NQL Query Types", value: "17+", icon: Terminal, color: "text-emerald-400" },
    { label: "gRPC RPCs", value: "65+", icon: Radio, color: "text-blue-400" },
    { label: "Graph Algorithms", value: "11", icon: GitBranch, color: "text-purple-400" },
    { label: "Math Functions", value: "13", icon: Workflow, color: "text-amber-400" },
    { label: "Node Types", value: "7", icon: Database, color: "text-cyan-400" },
]

interface FeatureCategory {
    title: string
    icon: React.ReactNode
    badge?: string
    items: string[]
    link?: string
}

const FEATURE_CATEGORIES: FeatureCategory[] = [
    {
        title: "Query Engine (NQL)",
        icon: <Terminal className="h-4 w-4" />,
        badge: "17+ types",
        link: "/query",
        items: [
            "MATCH, CREATE, MERGE, DELETE, SET",
            "DIFFUSE (heat-kernel propagation)",
            "DREAM (speculative exploration)",
            "RECONSTRUCT (sensory decode)",
            "NARRATE (auto-narrative)",
            "COUNTERFACTUAL (what-if)",
            "TRANSLATE (cross-modal)",
            "DAEMON (autonomous agents)",
            "EXPLAIN, TIME-TRAVEL, Transactions",
            "13 named math functions (Poincare, Riemann, Hausdorff...)",
        ],
    },
    {
        title: "Graph Algorithms",
        icon: <GitBranch className="h-4 w-4" />,
        badge: "11 algos",
        link: "/algorithms",
        items: [
            "PageRank, Louvain, Label Propagation",
            "Betweenness & Closeness Centrality",
            "Degree Centrality, WCC, SCC",
            "Triangle Count, Jaccard Similarity",
            "A* Pathfinding (gRPC)",
        ],
    },
    {
        title: "Autonomous Agency",
        icon: <Brain className="h-4 w-4" />,
        badge: "4 crates",
        link: "/agency",
        items: [
            "MetaObserver (self-referential identity)",
            "Zarathustra (evolutionary optimization)",
            "Narrative Engine (auto-generated stories)",
            "Desire System (goal-driven behavior)",
            "Counterfactual & Quantum Mapping",
        ],
    },
    {
        title: "Sleep & Reconsolidation",
        icon: <Moon className="h-4 w-4" />,
        link: "/sleep",
        items: [
            "RiemannianAdam optimization",
            "Hausdorff dimension monitoring",
            "Dream snapshots & apply/reject",
            "Sensory encoder/decoder",
        ],
    },
    {
        title: "Storage & Infrastructure",
        icon: <Layers className="h-4 w-4" />,
        badge: "Production",
        items: [
            "RocksDB with 7 column families",
            "HNSW index (Poincare ball metric)",
            "WAL with CRC32 + AES-256 encryption",
            "Redis-compatible cache layer",
            "TTL Janitor (auto-expiration)",
            "Multi-collection isolation",
        ],
    },
    {
        title: "Hardware Acceleration",
        icon: <Cpu className="h-4 w-4" />,
        link: "/monitoring",
        items: [
            "NVIDIA cuVS CAGRA (GPU HNSW)",
            "Google PJRT (TPU v5e/v6e/v7)",
            "cuGraph (GPU graph traversal)",
            "Kafka CDC (change data capture)",
            "SQLite table layer",
            "OpenDAL media storage (S3/GCS)",
        ],
    },
    {
        title: "Data Management",
        icon: <Archive className="h-4 w-4" />,
        link: "/backup",
        items: [
            "Snapshot backups with labels",
            "JSONL & CSV export",
            "Node & Edge CRUD (REST + gRPC)",
            "Batch import (nodes + edges)",
            "Full-text search (BM25)",
        ],
    },
    {
        title: "Cluster & Observability",
        icon: <Network className="h-4 w-4" />,
        link: "/monitoring",
        items: [
            "Consistent hash ring clustering",
            "Prometheus metrics export",
            "gRPC + REST dual API",
            "Gossip-based archetype sharing",
        ],
    },
]

export function OverviewPage() {
    const { data: status, isLoading: sLoading } = useQuery({
        queryKey: ['status'],
        queryFn: fetchStatus,
        refetchInterval: 5000
    })
    const { data: rawStats } = useQuery({
        queryKey: ['metrics'],
        queryFn: () => api.get("/stats").then(r => r.data),
        refetchInterval: 5000
    })
    const { data: rawProm } = useQuery({
        queryKey: ['prometheus'],
        queryFn: getMetrics,
        refetchInterval: 10000
    })

    const promMetrics = useMemo(() => {
        if (!rawProm || typeof rawProm !== "string") return null
        return {
            uptime: parsePrometheusValue(rawProm, "nietzsche_uptime_seconds"),
            insertNodeTotal: parsePrometheusValue(rawProm, "nietzsche_insert_node_total"),
            queryTotal: parsePrometheusValue(rawProm, "nietzsche_query_total"),
            knnTotal: parsePrometheusValue(rawProm, "nietzsche_knn_total"),
            sleepTotal: parsePrometheusValue(rawProm, "nietzsche_sleep_total"),
            zaratustraTotal: parsePrometheusValue(rawProm, "nietzsche_zaratustra_total"),
            errorTotal: parsePrometheusValue(rawProm, "nietzsche_error_total"),
        }
    }, [rawProm])

    const nodeCount = rawStats?.node_count ?? 0
    const edgeCount = rawStats?.edge_count ?? 0
    const collectionCount = rawStats?.collections ?? 0

    if (sLoading && !status) return <OverviewSkeleton />

    return (
        <div className="space-y-6 fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">System Overview</h1>
                    <p className="text-sm text-muted-foreground mt-1">
                        Temporal Hyperbolic Graph Database — 38 Rust crates, Poincare ball geometry
                    </p>
                </div>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className={`h-2 w-2 rounded-full ${status?.status === 'ONLINE' ? 'bg-green-500 shadow-[0_0_8px_#22c55e]' : 'bg-red-500'}`}></div>
                    {status?.status || "Connecting..."}
                </div>
            </div>

            {/* Live Stats */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
                <StatCard title="Total Nodes" value={nodeCount.toLocaleString()} icon={Database} desc="Across all collections" />
                <StatCard title="Total Edges" value={edgeCount.toLocaleString()} icon={GitBranch} desc="Graph relationships" />
                <StatCard title="Collections" value={collectionCount} icon={Server} desc="Active indices" />
                <StatCard title="Uptime" value={promMetrics?.uptime ? formatUptime(promMetrics.uptime) : "—"} icon={Clock} desc="Server uptime" />
                <StatCard title="GPU" value="NVIDIA L4" icon={Zap} desc="24 GB VRAM · CUDA" highlight />
            </div>

            {/* Platform Capabilities Banner */}
            <Card className="border-primary/20 bg-gradient-to-r from-primary/5 to-transparent">
                <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                        <div>
                            <CardTitle className="text-lg flex items-center gap-2">
                                <Shield className="h-5 w-5 text-primary" /> Platform Capabilities
                            </CardTitle>
                            <CardDescription>NietzscheDB v2.1 — Rust nightly, hyperbolic geometry engine</CardDescription>
                        </div>
                    </div>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                        {PLATFORM_STATS.map((s) => (
                            <div key={s.label} className="flex items-center gap-2.5 rounded-lg border border-border/40 bg-background/50 px-3 py-2.5">
                                <s.icon className={`h-4 w-4 ${s.color}`} />
                                <div>
                                    <div className="text-lg font-bold font-mono leading-none">{s.value}</div>
                                    <div className="text-[10px] text-muted-foreground uppercase tracking-wide">{s.label}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>

            {/* Feature Categories Grid */}
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                {FEATURE_CATEGORIES.map((cat) => (
                    <Card key={cat.title} className="hover:border-primary/30 transition-colors">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm flex items-center gap-2">
                                {cat.icon}
                                <span>{cat.title}</span>
                                {cat.badge && <Badge variant="secondary" className="text-[10px] ml-auto">{cat.badge}</Badge>}
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <ul className="space-y-1">
                                {cat.items.map((item, i) => (
                                    <li key={i} className="text-[11px] text-muted-foreground leading-snug flex items-start gap-1.5">
                                        <span className="text-primary/60 mt-0.5">·</span>
                                        <span>{item}</span>
                                    </li>
                                ))}
                            </ul>
                            {cat.link && (
                                <Link to={cat.link} className="text-[11px] text-primary hover:underline mt-2 inline-block">
                                    Open →
                                </Link>
                            )}
                        </CardContent>
                    </Card>
                ))}
            </div>

            {/* Configuration + Operations row */}
            <div className="grid gap-4 md:grid-cols-2">
                <Card>
                    <CardHeader><CardTitle>Configuration</CardTitle><CardDescription>Runtime parameters</CardDescription></CardHeader>
                    <CardContent>
                        <div className="space-y-4">
                            <ConfigRow label="Version" value={status?.version} />
                            <ConfigRow label="Global Dimension" value={status?.config?.dimension} />
                            <ConfigRow label="Metric Space" value={
                                status?.config?.metric === 'cosine' ? 'Cosine Similarity' :
                                    status?.config?.metric === 'l2' || status?.config?.metric === 'euclidean' ? 'Euclidean (L2)' :
                                        status?.config?.metric === 'poincare' ? 'Hyperbolic (Poincare)' :
                                            status?.config?.metric || 'Unknown'
                            } />
                            <ConfigRow label="Quantization" value={(status as any)?.config?.quantization || "Scalar I8"} />
                            <ConfigRow label="Uptime" value={status?.uptime} />
                            <ConfigRow label="Embedding" value={status?.embedding?.enabled ? "Enabled" : "Disabled"} />
                            {status?.embedding?.enabled && (
                                <>
                                    <ConfigRow label="Provider" value={(status as any)?.embedding?.provider} />
                                    <ConfigRow label="Model" value={(status as any)?.embedding?.model} />
                                </>
                            )}
                        </div>
                    </CardContent>
                </Card>

                <IngestionStatusCard promMetrics={promMetrics} />

                <Card>
                    <CardHeader>
                        <CardTitle>Maintenance</CardTitle>
                        <CardDescription>System-level operations</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-medium">Memory Management</span>
                                <Button variant="outline" size="sm" onClick={() => {
                                    if (confirm("Trigger manual memory vacuum? This may cause temporary latency.")) {
                                        api.post("/admin/vacuum")
                                            .then(() => alert("Memory cleanup triggered!"))
                                            .catch(e => alert("Failed: " + e.message))
                                    }
                                }}>
                                    Reset Memory
                                </Button>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Quick Navigation */}
                <Card>
                    <CardHeader>
                        <CardTitle>Quick Navigation</CardTitle>
                        <CardDescription>Jump to any dashboard section</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-2 gap-2">
                            <QuickLink to="/query" icon={Terminal} label="NQL Console" />
                            <QuickLink to="/algorithms" icon={GitBranch} label="Algorithms" />
                            <QuickLink to="/graph" icon={Search} label="Graph Explorer" />
                            <QuickLink to="/agency" icon={Brain} label="Agency" />
                            <QuickLink to="/sleep" icon={Moon} label="Sleep & Dream" />
                            <QuickLink to="/backup" icon={Archive} label="Backup" />
                            <QuickLink to="/monitoring" icon={Activity} label="Monitoring" />
                            <QuickLink to="/collections" icon={Database} label="Collections" />
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}

function ConfigRow({ label, value }: any) {
    return (
        <div className="flex items-center justify-between py-1 border-b border-border/40 last:border-0">
            <span className="text-sm font-medium text-muted-foreground">{label}</span>
            <span className="font-mono text-sm">{value || "-"}</span>
        </div>
    )
}

function IngestionStatusCard({ promMetrics }: { promMetrics: any }) {
    const [refreshInterval, setRefreshInterval] = useState("10")

    const { data: liveRaw } = useQuery({
        queryKey: ['live-metrics'],
        queryFn: () => api.get("/stats").then(r => r.data),
        refetchInterval: parseInt(refreshInterval) * 1000
    })

    return (
        <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <div>
                    <CardTitle>Operations</CardTitle>
                    <CardDescription>Prometheus counters (live)</CardDescription>
                </div>
                <Select value={refreshInterval} onValueChange={setRefreshInterval}>
                    <SelectTrigger className="w-[110px]">
                        <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="5">5 sec</SelectItem>
                        <SelectItem value="10">10 sec</SelectItem>
                        <SelectItem value="30">30 sec</SelectItem>
                        <SelectItem value="60">60 sec</SelectItem>
                    </SelectContent>
                </Select>
            </CardHeader>
            <CardContent>
                <div className="space-y-4">
                    <div className="flex items-center justify-between py-2 border-b">
                        <span className="text-sm text-muted-foreground">Total Nodes</span>
                        <span className="font-mono font-bold text-lg">{(liveRaw?.node_count ?? 0).toLocaleString()}</span>
                    </div>
                    <div className="flex items-center justify-between py-2 border-b">
                        <span className="text-sm text-muted-foreground">Inserts</span>
                        <span className="font-mono font-bold text-lg">{promMetrics?.insertNodeTotal?.toLocaleString() ?? "—"}</span>
                    </div>
                    <div className="flex items-center justify-between py-2 border-b">
                        <span className="text-sm text-muted-foreground">Queries</span>
                        <span className="font-mono font-bold text-lg">{promMetrics?.queryTotal?.toLocaleString() ?? "—"}</span>
                    </div>
                    <div className="flex items-center justify-between py-2 border-b">
                        <span className="text-sm text-muted-foreground">KNN Searches</span>
                        <span className="font-mono font-bold text-lg">{promMetrics?.knnTotal?.toLocaleString() ?? "—"}</span>
                    </div>
                    <div className="flex items-center justify-between py-2 border-b">
                        <span className="text-sm text-muted-foreground">Sleep Cycles</span>
                        <span className="font-mono font-bold text-lg">{promMetrics?.sleepTotal?.toLocaleString() ?? "—"}</span>
                    </div>
                    <div className="flex items-center justify-between py-2">
                        <span className="text-sm text-muted-foreground">Errors</span>
                        <span className={`font-mono font-bold text-lg ${(promMetrics?.errorTotal ?? 0) > 0 ? 'text-red-500' : ''}`}>
                            {promMetrics?.errorTotal?.toLocaleString() ?? "0"}
                        </span>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}

function StatCard({ title, value, icon: Icon, desc, highlight }: any) {
    return (
        <Card className={highlight ? "border-primary/30" : ""}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{title}</CardTitle>
                <Icon className={`h-4 w-4 ${highlight ? "text-primary" : "text-muted-foreground"}`} />
            </CardHeader>
            <CardContent>
                <div className="text-2xl font-bold">{value}</div>
                <p className="text-xs text-muted-foreground">{desc}</p>
            </CardContent>
        </Card>
    )
}

function QuickLink({ to, icon: Icon, label }: { to: string; icon: any; label: string }) {
    return (
        <Link
            to={to}
            className="flex items-center gap-2 rounded-md border border-border/40 px-3 py-2 text-sm hover:bg-accent hover:text-accent-foreground transition-colors"
        >
            <Icon className="h-3.5 w-3.5 text-muted-foreground" />
            <span>{label}</span>
        </Link>
    )
}

function OverviewSkeleton() {
    return <div className="space-y-6"><Skeleton className="h-10 w-48" /><div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5"><Skeleton className="h-32" /><Skeleton className="h-32" /><Skeleton className="h-32" /><Skeleton className="h-32" /><Skeleton className="h-32" /></div></div>
}
