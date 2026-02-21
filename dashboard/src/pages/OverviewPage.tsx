import { useQuery } from "@tanstack/react-query"
import { useMemo } from "react"
import { api, fetchStatus, getMetrics } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Database, Server, Zap, GitBranch, Clock } from "lucide-react"
import { Skeleton } from "@/components/ui/skeleton"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useState } from "react"

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
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold tracking-tight">System Overview</h1>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className={`h-2 w-2 rounded-full ${status?.status === 'ONLINE' ? 'bg-green-500 shadow-[0_0_8px_#22c55e]' : 'bg-red-500'}`}></div>
                    {status?.status || "Connecting..."}
                </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
                <StatCard title="Total Nodes" value={nodeCount.toLocaleString()} icon={Database} desc="Across all collections" />
                <StatCard title="Total Edges" value={edgeCount.toLocaleString()} icon={GitBranch} desc="Graph relationships" />
                <StatCard title="Collections" value={collectionCount} icon={Server} desc="Active indices" />
                <StatCard title="Uptime" value={promMetrics?.uptime ? formatUptime(promMetrics.uptime) : "—"} icon={Clock} desc="Server uptime" />
                <StatCard title="GPU" value="NVIDIA L4" icon={Zap} desc="24 GB VRAM · CUDA" highlight />
            </div>

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
                                        status?.config?.metric === 'poincare' ? 'Hyperbolic (Poincaré)' :
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

function StatCard({ title, value, icon: Icon, desc }: any) {
    return (
        <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{title}</CardTitle>
                <Icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
                <div className="text-2xl font-bold">{value}</div>
                <p className="text-xs text-muted-foreground">{desc}</p>
            </CardContent>
        </Card>
    )
}

function OverviewSkeleton() {
    return <div className="space-y-6"><Skeleton className="h-10 w-48" /><div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5"><Skeleton className="h-32" /><Skeleton className="h-32" /><Skeleton className="h-32" /><Skeleton className="h-32" /><Skeleton className="h-32" /></div></div>
}
