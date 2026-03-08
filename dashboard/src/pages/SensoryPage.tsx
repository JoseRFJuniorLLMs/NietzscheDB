import { useState, useCallback } from "react"
import { useQuery } from "@tanstack/react-query"
import {
    Waves, Search, RotateCcw, TrendingDown, Loader2,
    AlertCircle, Zap, RefreshCw,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
    Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
    Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { api, executeNql } from "@/lib/api"

export default function SensoryPage() {
    return (
        <div className="space-y-6 fade-in">
            <div>
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Waves className="h-6 w-6" /> Sensory Compression
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                    Phase 11 — Energy-based quantization, reconstruction, and degradation control
                </p>
            </div>

            <Tabs defaultValue="explore" className="space-y-4">
                <TabsList className="flex-wrap h-auto gap-1">
                    <TabsTrigger value="explore"><Search className="h-3.5 w-3.5 mr-1" />Explore</TabsTrigger>
                    <TabsTrigger value="reconstruct"><RotateCcw className="h-3.5 w-3.5 mr-1" />Reconstruct</TabsTrigger>
                    <TabsTrigger value="degradation"><TrendingDown className="h-3.5 w-3.5 mr-1" />Degradation</TabsTrigger>
                </TabsList>

                <TabsContent value="explore"><ExploreTab /></TabsContent>
                <TabsContent value="reconstruct"><ReconstructTab /></TabsContent>
                <TabsContent value="degradation"><DegradationTab /></TabsContent>
            </Tabs>
        </div>
    )
}

/* ── Helpers ──────────────────────────────────────────────── */

const MODALITY_ICON: Record<string, string> = {
    audio: "\uD83C\uDFB5",
    image: "\uD83D\uDCF7",
    text: "\uD83D\uDCDD",
}

function modalityIcon(modality?: string): string {
    if (!modality) return "\u2753"
    return MODALITY_ICON[modality.toLowerCase()] ?? "\u2753"
}

function energyColor(energy: number): string {
    if (energy >= 0.7) return "text-emerald-400"
    if (energy >= 0.5) return "text-cyan-400"
    if (energy >= 0.3) return "text-amber-400"
    if (energy >= 0.1) return "text-orange-400"
    return "text-red-400"
}

function energyBadgeVariant(energy: number): "default" | "secondary" | "destructive" | "outline" {
    if (energy >= 0.7) return "default"
    if (energy >= 0.3) return "secondary"
    return "destructive"
}

function quantizationLabel(energy: number): string {
    if (energy >= 0.7) return "f32"
    if (energy >= 0.5) return "f16"
    if (energy >= 0.3) return "int8"
    if (energy >= 0.1) return "PQ 64B"
    return "None"
}

/* ── Explore Tab ──────────────────────────────────────────── */
function ExploreTab() {
    const { data, isLoading, error, refetch } = useQuery({
        queryKey: ["sensory-explore"],
        queryFn: () => executeNql('MATCH (n) WHERE n.modality IS NOT NULL RETURN n LIMIT 50'),
        refetchInterval: 15000,
    })

    if (isLoading) return <LoadingSkeleton />
    if (error) return <ErrorCard message="Could not fetch sensory nodes" />

    const nodes: any[] = data?.nodes ?? []

    return (
        <Card>
            <CardHeader className="flex flex-row items-center justify-between">
                <div>
                    <CardTitle className="text-base flex items-center gap-2">
                        <Search className="h-4 w-4" /> Sensory Nodes
                        <Badge variant="secondary">{nodes.length}</Badge>
                    </CardTitle>
                    <CardDescription>Nodes with modality metadata (audio, image, text)</CardDescription>
                </div>
                <Button size="sm" variant="outline" onClick={() => refetch()}>
                    <RefreshCw className="h-3.5 w-3.5 mr-1" /> Refresh
                </Button>
            </CardHeader>
            <CardContent>
                {nodes.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No sensory nodes found. Nodes must have a <code className="text-xs font-mono bg-muted px-1 rounded">modality</code> field in content.</p>
                ) : (
                    <ScrollArea className="max-h-[500px]">
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead className="w-12">Type</TableHead>
                                    <TableHead>Node ID</TableHead>
                                    <TableHead>Modality</TableHead>
                                    <TableHead>Energy</TableHead>
                                    <TableHead>Quantization</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {nodes.map((n: any, i: number) => {
                                    const id = n.id ?? n.node_id ?? `node-${i}`
                                    const modality = n.content?.modality ?? n.modality ?? "unknown"
                                    const energy = n.energy ?? 0
                                    return (
                                        <TableRow key={id}>
                                            <TableCell className="text-lg text-center">
                                                {modalityIcon(modality)}
                                            </TableCell>
                                            <TableCell className="font-mono text-xs">
                                                {id.length > 24 ? id.slice(0, 12) + "..." + id.slice(-8) : id}
                                            </TableCell>
                                            <TableCell>
                                                <Badge variant="outline" className="text-xs uppercase">
                                                    {modality}
                                                </Badge>
                                            </TableCell>
                                            <TableCell>
                                                <span className={`font-mono font-semibold ${energyColor(energy)}`}>
                                                    {energy.toFixed(4)}
                                                </span>
                                            </TableCell>
                                            <TableCell>
                                                <Badge variant={energyBadgeVariant(energy)} className="font-mono text-xs">
                                                    {quantizationLabel(energy)}
                                                </Badge>
                                            </TableCell>
                                        </TableRow>
                                    )
                                })}
                            </TableBody>
                        </Table>
                    </ScrollArea>
                )}
            </CardContent>
        </Card>
    )
}

/* ── Reconstruct Tab ──────────────────────────────────────── */
function ReconstructTab() {
    const [nodeId, setNodeId] = useState("")
    const [quality, setQuality] = useState("high")
    const [result, setResult] = useState<any>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const handleReconstruct = useCallback(async () => {
        if (!nodeId.trim()) return
        setLoading(true)
        setError(null)
        setResult(null)
        try {
            const data = await executeNql(
                `RECONSTRUCT "${nodeId.trim()}" MODALITY auto QUALITY ${quality}`
            )
            setResult(data)
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : "Reconstruction failed")
        } finally {
            setLoading(false)
        }
    }, [nodeId, quality])

    return (
        <div className="space-y-4">
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2">
                        <RotateCcw className="h-4 w-4" /> Sensory Reconstruction
                    </CardTitle>
                    <CardDescription>
                        Reconstruct compressed sensory data from a node at the desired quality level
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex flex-wrap gap-4 items-end">
                        <div className="space-y-1 flex-1 min-w-[200px]">
                            <Label className="text-xs">Node ID</Label>
                            <Input
                                value={nodeId}
                                onChange={(e) => setNodeId(e.target.value)}
                                placeholder="Enter node UUID"
                                className="font-mono"
                            />
                        </div>
                        <div className="space-y-1">
                            <Label className="text-xs">Quality</Label>
                            <Select value={quality} onValueChange={setQuality}>
                                <SelectTrigger className="w-32 h-9">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="high">high</SelectItem>
                                    <SelectItem value="med">med</SelectItem>
                                    <SelectItem value="low">low</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                        <Button onClick={handleReconstruct} disabled={loading || !nodeId.trim()}>
                            {loading ? (
                                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                            ) : (
                                <RotateCcw className="h-4 w-4 mr-1" />
                            )}
                            {loading ? "Reconstructing..." : "RECONSTRUCT"}
                        </Button>
                    </div>
                </CardContent>
            </Card>

            {error && (
                <Card className="border-destructive/50">
                    <CardContent className="pt-4 flex items-center gap-2 text-sm">
                        <AlertCircle className="h-4 w-4 text-destructive" />
                        <span className="text-destructive font-mono">{error}</span>
                    </CardContent>
                </Card>
            )}

            {result && (
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-base">Reconstruction Result</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <pre className="text-xs font-mono bg-muted p-4 rounded-lg overflow-auto max-h-96">
                            {JSON.stringify(result, null, 2)}
                        </pre>
                    </CardContent>
                </Card>
            )}
        </div>
    )
}

/* ── Degradation Tab ──────────────────────────────────────── */

const DEGRADATION_TABLE = [
    { threshold: "\u2265 0.7", precision: "f32", compression: "1x", color: "text-emerald-400", bg: "bg-emerald-400/10" },
    { threshold: "\u2265 0.5", precision: "f16", compression: "2x", color: "text-cyan-400", bg: "bg-cyan-400/10" },
    { threshold: "\u2265 0.3", precision: "int8", compression: "4x", color: "text-amber-400", bg: "bg-amber-400/10" },
    { threshold: "\u2265 0.1", precision: "PQ 64B", compression: "16x", color: "text-orange-400", bg: "bg-orange-400/10" },
    { threshold: "< 0.1", precision: "None", compression: "\u221E", color: "text-red-400", bg: "bg-red-400/10" },
]

function DegradationTab() {
    const [nodeId, setNodeId] = useState("")
    const [targetEnergy, setTargetEnergy] = useState("0.3")
    const [result, setResult] = useState<any>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const handleDegrade = useCallback(async () => {
        if (!nodeId.trim()) return
        setLoading(true)
        setError(null)
        setResult(null)
        try {
            const data = await api.post("/query", {
                nql: `DEGRADE "${nodeId.trim()}" ENERGY ${parseFloat(targetEnergy) || 0.3}`,
            }).then((r) => r.data)
            setResult(data)
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : "Degradation failed")
        } finally {
            setLoading(false)
        }
    }, [nodeId, targetEnergy])

    return (
        <div className="space-y-4">
            {/* Degradation Reference Table */}
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2">
                        <Zap className="h-4 w-4" /> Energy-Precision Mapping
                    </CardTitle>
                    <CardDescription>
                        Sensory data precision degrades as node energy decays over time
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Energy Threshold</TableHead>
                                <TableHead>Precision</TableHead>
                                <TableHead>Compression Ratio</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {DEGRADATION_TABLE.map((row) => (
                                <TableRow key={row.precision}>
                                    <TableCell>
                                        <span className={`font-mono font-semibold ${row.color}`}>
                                            {row.threshold}
                                        </span>
                                    </TableCell>
                                    <TableCell>
                                        <Badge variant="outline" className={`font-mono ${row.bg} ${row.color} border-0`}>
                                            {row.precision}
                                        </Badge>
                                    </TableCell>
                                    <TableCell className="font-mono text-muted-foreground">
                                        {row.compression}
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>

            {/* Manual Degradation */}
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2">
                        <TrendingDown className="h-4 w-4" /> Manual Degradation
                    </CardTitle>
                    <CardDescription>
                        Force a sensory node to degrade to a target energy level
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex flex-wrap gap-4 items-end">
                        <div className="space-y-1 flex-1 min-w-[200px]">
                            <Label className="text-xs">Node ID</Label>
                            <Input
                                value={nodeId}
                                onChange={(e) => setNodeId(e.target.value)}
                                placeholder="Enter node UUID"
                                className="font-mono"
                            />
                        </div>
                        <div className="space-y-1">
                            <Label className="text-xs">Target Energy</Label>
                            <Input
                                type="number"
                                step="0.1"
                                min="0"
                                max="1"
                                value={targetEnergy}
                                onChange={(e) => setTargetEnergy(e.target.value)}
                                className="w-28 h-9 font-mono"
                            />
                        </div>
                        <Button
                            onClick={handleDegrade}
                            disabled={loading || !nodeId.trim()}
                            variant="destructive"
                        >
                            {loading ? (
                                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                            ) : (
                                <TrendingDown className="h-4 w-4 mr-1" />
                            )}
                            {loading ? "Degrading..." : "DEGRADE"}
                        </Button>
                    </div>

                    {/* Preview what precision the target energy maps to */}
                    {targetEnergy && (
                        <div className="rounded-lg border bg-muted/30 p-3 inline-flex items-center gap-3">
                            <span className="text-xs text-muted-foreground">Target precision:</span>
                            <span className={`font-mono font-semibold ${energyColor(parseFloat(targetEnergy) || 0)}`}>
                                {quantizationLabel(parseFloat(targetEnergy) || 0)}
                            </span>
                            <span className="text-xs text-muted-foreground">
                                ({(parseFloat(targetEnergy) || 0).toFixed(2)} energy)
                            </span>
                        </div>
                    )}
                </CardContent>
            </Card>

            {error && (
                <Card className="border-destructive/50">
                    <CardContent className="pt-4 flex items-center gap-2 text-sm">
                        <AlertCircle className="h-4 w-4 text-destructive" />
                        <span className="text-destructive font-mono">{error}</span>
                    </CardContent>
                </Card>
            )}

            {result && (
                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-base">Degradation Result</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <pre className="text-xs font-mono bg-muted p-4 rounded-lg overflow-auto max-h-64">
                            {JSON.stringify(result, null, 2)}
                        </pre>
                    </CardContent>
                </Card>
            )}
        </div>
    )
}

/* ── Shared Components ────────────────────────────────────── */
function LoadingSkeleton() {
    return (
        <Card>
            <CardContent className="pt-4 space-y-3">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
                <Skeleton className="h-32 w-full" />
            </CardContent>
        </Card>
    )
}

function ErrorCard({ message }: { message: string }) {
    return (
        <Card className="border-destructive/30">
            <CardContent className="pt-4 flex items-center gap-2 text-sm text-muted-foreground">
                <AlertCircle className="h-4 w-4 text-destructive" /> {message}
            </CardContent>
        </Card>
    )
}
