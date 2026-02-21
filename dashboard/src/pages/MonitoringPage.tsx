import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import {
    Activity, Cpu, Gauge, HardDrive, Radio, Server,
    Wifi, WifiOff, Zap,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
    Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { getMetrics, getClusterRing } from "@/lib/api"

/* ── Parse Prometheus text format ─────────────────────────── */
interface ParsedMetric {
    name: string
    labels: string
    value: string
    help?: string
    type?: string
}

function parsePrometheus(raw: string): ParsedMetric[] {
    const lines = raw.split("\n")
    const metrics: ParsedMetric[] = []
    let currentHelp = ""
    let currentType = ""

    for (const line of lines) {
        if (line.startsWith("# HELP ")) {
            currentHelp = line.slice(7)
        } else if (line.startsWith("# TYPE ")) {
            currentType = line.split(" ").pop() ?? ""
        } else if (line && !line.startsWith("#")) {
            const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)((?:\{[^}]*\})?)?\s+(.+)$/)
            if (match) {
                metrics.push({
                    name: match[1],
                    labels: match[2] || "",
                    value: match[3],
                    help: currentHelp,
                    type: currentType,
                })
            }
        }
    }
    return metrics
}

/* ── Extract key metrics for cards ────────────────────────── */
function extractKeyMetrics(metrics: ParsedMetric[]) {
    const find = (prefix: string) => metrics.filter((m) => m.name.startsWith(prefix))
    const findOne = (name: string) => metrics.find((m) => m.name === name)

    const httpTotal = find("http_requests_total").reduce((s, m) => s + parseFloat(m.value || "0"), 0)
    const grpcTotal = find("grpc_requests_total").reduce((s, m) => s + parseFloat(m.value || "0"), 0)
    const memBytes = findOne("process_resident_memory_bytes")
    const cpuSec = findOne("process_cpu_seconds_total")
    const nodeCount = findOne("nietzsche_nodes_total") ?? findOne("graph_node_count")
    const edgeCount = findOne("nietzsche_edges_total") ?? findOne("graph_edge_count")

    return {
        httpRequests: Math.round(httpTotal),
        grpcRequests: Math.round(grpcTotal),
        memoryMb: memBytes ? (parseFloat(memBytes.value) / 1048576).toFixed(1) : null,
        cpuSeconds: cpuSec ? parseFloat(cpuSec.value).toFixed(1) : null,
        nodeCount: nodeCount ? parseInt(nodeCount.value) : null,
        edgeCount: edgeCount ? parseInt(edgeCount.value) : null,
    }
}

export default function MonitoringPage() {
    const { data: rawMetrics, isLoading: metricsLoading } = useQuery({
        queryKey: ["metrics"],
        queryFn: getMetrics,
        refetchInterval: 10000,
    })

    const { data: ringData, isLoading: ringLoading } = useQuery({
        queryKey: ["cluster-ring"],
        queryFn: getClusterRing,
        refetchInterval: 15000,
    })

    const parsedMetrics = useMemo(() => {
        if (!rawMetrics || typeof rawMetrics !== "string") return []
        return parsePrometheus(rawMetrics)
    }, [rawMetrics])

    const keyMetrics = useMemo(() => extractKeyMetrics(parsedMetrics), [parsedMetrics])

    return (
        <div className="space-y-6 fade-in">
            <div>
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Activity className="h-6 w-6" /> System Monitoring
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                    Prometheus metrics, cluster hash ring, hardware acceleration status
                </p>
            </div>

            {/* Key Metric Cards */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                <StatCard
                    icon={<Gauge className="h-4 w-4" />}
                    label="HTTP Requests"
                    value={keyMetrics.httpRequests > 0 ? keyMetrics.httpRequests.toLocaleString() : "—"}
                    loading={metricsLoading}
                />
                <StatCard
                    icon={<Radio className="h-4 w-4" />}
                    label="gRPC Requests"
                    value={keyMetrics.grpcRequests > 0 ? keyMetrics.grpcRequests.toLocaleString() : "—"}
                    loading={metricsLoading}
                />
                <StatCard
                    icon={<HardDrive className="h-4 w-4" />}
                    label="Memory"
                    value={keyMetrics.memoryMb ? `${keyMetrics.memoryMb} MB` : "—"}
                    loading={metricsLoading}
                />
                <StatCard
                    icon={<Cpu className="h-4 w-4" />}
                    label="CPU Time"
                    value={keyMetrics.cpuSeconds ? `${keyMetrics.cpuSeconds}s` : "—"}
                    loading={metricsLoading}
                />
                <StatCard
                    icon={<Server className="h-4 w-4" />}
                    label="Nodes"
                    value={keyMetrics.nodeCount !== null ? keyMetrics.nodeCount.toLocaleString() : "—"}
                    loading={metricsLoading}
                />
                <StatCard
                    icon={<Zap className="h-4 w-4" />}
                    label="Edges"
                    value={keyMetrics.edgeCount !== null ? keyMetrics.edgeCount.toLocaleString() : "—"}
                    loading={metricsLoading}
                />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Cluster Hash Ring */}
                <Card>
                    <CardHeader>
                        <CardTitle className="text-base flex items-center gap-2">
                            <Radio className="h-4 w-4" /> Cluster Hash Ring
                        </CardTitle>
                        <CardDescription>Consistent hash ring for data distribution</CardDescription>
                    </CardHeader>
                    <CardContent>
                        {ringLoading ? (
                            <div className="space-y-2"><Skeleton className="h-8 w-full" /><Skeleton className="h-8 w-full" /></div>
                        ) : ringData ? (
                            <>
                                <div className="flex items-center gap-2 mb-3">
                                    <Badge variant={ringData.enabled ? "default" : "secondary"}>
                                        {ringData.enabled ? <><Wifi className="h-3 w-3 mr-1" />Enabled</> : <><WifiOff className="h-3 w-3 mr-1" />Disabled</>}
                                    </Badge>
                                    <Badge variant="outline">{ringData.ring?.length ?? 0} entries</Badge>
                                </div>
                                {ringData.ring && ringData.ring.length > 0 ? (
                                    <ScrollArea className="max-h-64">
                                        <Table>
                                            <TableHeader>
                                                <TableRow>
                                                    <TableHead>Token</TableHead>
                                                    <TableHead>Name</TableHead>
                                                    <TableHead>Address</TableHead>
                                                    <TableHead>Health</TableHead>
                                                </TableRow>
                                            </TableHeader>
                                            <TableBody>
                                                {ringData.ring.map((r, i) => (
                                                    <TableRow key={i}>
                                                        <TableCell className="font-mono text-xs">{r.token}</TableCell>
                                                        <TableCell className="text-xs">{r.name}</TableCell>
                                                        <TableCell className="font-mono text-xs">{r.addr}</TableCell>
                                                        <TableCell>
                                                            <Badge variant={r.health === "healthy" ? "default" : "destructive"} className="text-[10px]">
                                                                {r.health}
                                                            </Badge>
                                                        </TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </ScrollArea>
                                ) : (
                                    <p className="text-sm text-muted-foreground">Single-node mode — no ring entries.</p>
                                )}
                            </>
                        ) : (
                            <p className="text-sm text-muted-foreground">Could not fetch cluster ring.</p>
                        )}
                    </CardContent>
                </Card>

                {/* Hardware Acceleration */}
                <Card>
                    <CardHeader>
                        <CardTitle className="text-base flex items-center gap-2">
                            <Zap className="h-4 w-4" /> Hardware Acceleration
                        </CardTitle>
                        <CardDescription>GPU, TPU, and streaming status</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        <HardwareRow
                            icon={<Cpu className="h-4 w-4" />}
                            name="NVIDIA cuVS CAGRA"
                            desc="HNSW acceleration via GPU (nietzsche-hnsw-gpu)"
                            crate="nietzsche-hnsw-gpu"
                        />
                        <HardwareRow
                            icon={<Server className="h-4 w-4" />}
                            name="Google PJRT (TPU)"
                            desc="TPU v5e/v6e/v7 compute (nietzsche-tpu)"
                            crate="nietzsche-tpu"
                        />
                        <HardwareRow
                            icon={<Zap className="h-4 w-4" />}
                            name="cuGraph"
                            desc="GPU graph traversal (nietzsche-cugraph)"
                            crate="nietzsche-cugraph"
                        />
                        <HardwareRow
                            icon={<Radio className="h-4 w-4" />}
                            name="Kafka CDC"
                            desc="Change Data Capture streaming (nietzsche-kafka)"
                            crate="nietzsche-kafka"
                        />
                        <HardwareRow
                            icon={<HardDrive className="h-4 w-4" />}
                            name="SQLite Table Layer"
                            desc="Relational queries via SQLite (nietzsche-table)"
                            crate="nietzsche-table"
                        />
                        <HardwareRow
                            icon={<HardDrive className="h-4 w-4" />}
                            name="Media Storage"
                            desc="S3/GCS/local via OpenDAL (nietzsche-media)"
                            crate="nietzsche-media"
                        />
                    </CardContent>
                </Card>
            </div>

            {/* Raw Prometheus Metrics */}
            <Card>
                <CardHeader>
                    <CardTitle className="text-base">Raw Prometheus Metrics</CardTitle>
                    <CardDescription>
                        {parsedMetrics.length > 0
                            ? `${parsedMetrics.length} metric entries · auto-refresh 10s`
                            : "Fetching /metrics endpoint..."}
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    {metricsLoading ? (
                        <div className="space-y-2"><Skeleton className="h-4 w-full" /><Skeleton className="h-4 w-3/4" /></div>
                    ) : rawMetrics ? (
                        <ScrollArea className="max-h-80">
                            <pre className="text-[11px] font-mono bg-muted p-4 rounded-lg whitespace-pre-wrap break-all">
                                {typeof rawMetrics === "string" ? rawMetrics : JSON.stringify(rawMetrics, null, 2)}
                            </pre>
                        </ScrollArea>
                    ) : (
                        <p className="text-sm text-muted-foreground">
                            Could not fetch /metrics endpoint. The server may not expose Prometheus metrics.
                        </p>
                    )}
                </CardContent>
            </Card>
        </div>
    )
}

/* ── Shared Components ────────────────────────────────────── */
function StatCard({
    icon, label, value, loading,
}: {
    icon: React.ReactNode; label: string; value: string; loading: boolean
}) {
    return (
        <Card>
            <CardContent className="pt-4 pb-3">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                    {icon}
                    <span className="text-[11px] uppercase tracking-wide">{label}</span>
                </div>
                {loading ? (
                    <Skeleton className="h-6 w-16" />
                ) : (
                    <p className="text-lg font-mono font-semibold">{value}</p>
                )}
            </CardContent>
        </Card>
    )
}

function HardwareRow({
    icon, name, desc, crate,
}: {
    icon: React.ReactNode; name: string; desc: string; crate: string
}) {
    return (
        <div className="flex items-center gap-3 p-2 rounded-lg border border-border/30 bg-muted/20">
            <div className="text-muted-foreground">{icon}</div>
            <div className="flex-1 min-w-0">
                <div className="text-sm font-medium">{name}</div>
                <div className="text-[11px] text-muted-foreground">{desc}</div>
            </div>
            <Badge variant="outline" className="text-[10px] shrink-0">{crate}</Badge>
        </div>
    )
}
