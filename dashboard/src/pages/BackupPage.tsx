import { useState, useCallback } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
    Archive, Download, Upload, Plus, Trash2, Search, Loader2,
    Database, GitBranch as EdgeIcon, Layers,
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
import {
    Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import {
    createBackup, listBackups, exportData,
    getNode, insertNode, deleteNode,
    insertEdge, deleteEdge,
    batchInsertNodes, batchInsertEdges,
} from "@/lib/api"

function formatBytes(bytes: number) {
    if (bytes < 1024) return bytes + " B"
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB"
    return (bytes / 1048576).toFixed(2) + " MB"
}

export default function BackupPage() {
    return (
        <div className="space-y-6 fade-in">
            <div>
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Archive className="h-6 w-6" /> Backup & Data Management
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                    Backup, export, CRUD, and batch operations
                </p>
            </div>

            <Tabs defaultValue="backups" className="space-y-4">
                <TabsList className="flex-wrap h-auto gap-1">
                    <TabsTrigger value="backups"><Archive className="h-3.5 w-3.5 mr-1" />Backups</TabsTrigger>
                    <TabsTrigger value="export"><Download className="h-3.5 w-3.5 mr-1" />Export</TabsTrigger>
                    <TabsTrigger value="nodes"><Database className="h-3.5 w-3.5 mr-1" />Node CRUD</TabsTrigger>
                    <TabsTrigger value="edges"><EdgeIcon className="h-3.5 w-3.5 mr-1" />Edge CRUD</TabsTrigger>
                    <TabsTrigger value="batch"><Layers className="h-3.5 w-3.5 mr-1" />Batch Import</TabsTrigger>
                </TabsList>

                <TabsContent value="backups"><BackupsTab /></TabsContent>
                <TabsContent value="export"><ExportTab /></TabsContent>
                <TabsContent value="nodes"><NodeCrudTab /></TabsContent>
                <TabsContent value="edges"><EdgeCrudTab /></TabsContent>
                <TabsContent value="batch"><BatchTab /></TabsContent>
            </Tabs>
        </div>
    )
}

/* ── Backups ──────────────────────────────────────────────── */
function BackupsTab() {
    const qc = useQueryClient()
    const [label, setLabel] = useState("")
    const [creating, setCreating] = useState(false)

    const { data, isLoading } = useQuery({
        queryKey: ["backups"],
        queryFn: listBackups,
    })

    const handleCreate = useCallback(async () => {
        setCreating(true)
        try {
            await createBackup(label || undefined)
            setLabel("")
            qc.invalidateQueries({ queryKey: ["backups"] })
        } finally {
            setCreating(false)
        }
    }, [label, qc])

    return (
        <Card>
            <CardHeader>
                <CardTitle className="text-base">Manage Backups</CardTitle>
                <CardDescription>Create and list backup snapshots</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="flex gap-2">
                    <Input
                        value={label}
                        onChange={(e) => setLabel(e.target.value)}
                        placeholder="Backup label (optional)"
                        className="max-w-xs"
                    />
                    <Button onClick={handleCreate} disabled={creating}>
                        {creating ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Plus className="h-4 w-4 mr-1" />}
                        Create Backup
                    </Button>
                </div>

                {isLoading ? (
                    <div className="space-y-2"><Skeleton className="h-8 w-full" /><Skeleton className="h-8 w-full" /></div>
                ) : (
                    <ScrollArea className="max-h-80">
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Label</TableHead>
                                    <TableHead>Path</TableHead>
                                    <TableHead>Created</TableHead>
                                    <TableHead>Size</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {(data?.backups ?? []).length === 0 ? (
                                    <TableRow><TableCell colSpan={4} className="text-center text-muted-foreground">No backups yet</TableCell></TableRow>
                                ) : (
                                    (data?.backups ?? []).map((b, i) => (
                                        <TableRow key={i}>
                                            <TableCell><Badge variant="outline">{b.label}</Badge></TableCell>
                                            <TableCell className="font-mono text-xs max-w-[200px] truncate">{b.path}</TableCell>
                                            <TableCell className="text-xs">{b.created_at}</TableCell>
                                            <TableCell className="text-xs">{formatBytes(b.size_bytes)}</TableCell>
                                        </TableRow>
                                    ))
                                )}
                            </TableBody>
                        </Table>
                    </ScrollArea>
                )}
            </CardContent>
        </Card>
    )
}

/* ── Export ────────────────────────────────────────────────── */
function ExportTab() {
    const [type, setType] = useState<"nodes" | "edges">("nodes")
    const [format, setFormat] = useState<"jsonl" | "csv">("jsonl")
    const [downloading, setDownloading] = useState(false)

    const handleDownload = useCallback(async () => {
        setDownloading(true)
        try {
            const blob = await exportData(type, format)
            const url = URL.createObjectURL(blob)
            const a = document.createElement("a")
            a.href = url
            a.download = `nietzsche_${type}.${format}`
            a.click()
            URL.revokeObjectURL(url)
        } finally {
            setDownloading(false)
        }
    }, [type, format])

    return (
        <Card>
            <CardHeader>
                <CardTitle className="text-base">Export Data</CardTitle>
                <CardDescription>Download nodes or edges in JSONL or CSV format</CardDescription>
            </CardHeader>
            <CardContent>
                <div className="flex flex-wrap gap-4 items-end">
                    <div className="space-y-1">
                        <Label className="text-xs">Data Type</Label>
                        <Select value={type} onValueChange={(v) => setType(v as "nodes" | "edges")}>
                            <SelectTrigger className="w-32 h-8"><SelectValue /></SelectTrigger>
                            <SelectContent>
                                <SelectItem value="nodes">Nodes</SelectItem>
                                <SelectItem value="edges">Edges</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    <div className="space-y-1">
                        <Label className="text-xs">Format</Label>
                        <Select value={format} onValueChange={(v) => setFormat(v as "jsonl" | "csv")}>
                            <SelectTrigger className="w-32 h-8"><SelectValue /></SelectTrigger>
                            <SelectContent>
                                <SelectItem value="jsonl">JSONL</SelectItem>
                                <SelectItem value="csv">CSV</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    <Button onClick={handleDownload} disabled={downloading}>
                        {downloading ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Download className="h-4 w-4 mr-1" />}
                        Download
                    </Button>
                </div>
            </CardContent>
        </Card>
    )
}

/* ── Node CRUD ────────────────────────────────────────────── */
function NodeCrudTab() {
    // Insert
    const [nodeType, setNodeType] = useState("Semantic")
    const [nodeEnergy, setNodeEnergy] = useState("0.5")
    const [nodeContent, setNodeContent] = useState('{"label": "new node"}')
    const [insertResult, setInsertResult] = useState<string | null>(null)
    const [insertLoading, setInsertLoading] = useState(false)

    // Get
    const [getNodeId, setGetNodeId] = useState("")
    const [getResult, setGetResult] = useState<any>(null)
    const [getLoading, setGetLoading] = useState(false)

    // Delete
    const [delNodeId, setDelNodeId] = useState("")
    const [delResult, setDelResult] = useState<string | null>(null)

    const handleInsert = useCallback(async () => {
        setInsertLoading(true)
        try {
            const content = JSON.parse(nodeContent)
            const res = await insertNode({ node_type: nodeType, energy: parseFloat(nodeEnergy), content })
            setInsertResult(`Created: ${res.id}`)
        } catch (e: any) {
            setInsertResult(`Error: ${e.message}`)
        } finally {
            setInsertLoading(false)
        }
    }, [nodeType, nodeEnergy, nodeContent])

    const handleGet = useCallback(async () => {
        setGetLoading(true)
        try {
            const data = await getNode(getNodeId.trim())
            setGetResult(data)
        } catch { setGetResult({ error: "Node not found" }) }
        finally { setGetLoading(false) }
    }, [getNodeId])

    const handleDelete = useCallback(async () => {
        try {
            const res = await deleteNode(delNodeId.trim())
            setDelResult(`Deleted: ${res.deleted}`)
        } catch (e: any) {
            setDelResult(`Error: ${e.message}`)
        }
    }, [delNodeId])

    return (
        <div className="space-y-4">
            {/* Insert */}
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base">Insert Node</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                    <div className="flex flex-wrap gap-3 items-end">
                        <div className="space-y-1">
                            <Label className="text-xs">Type</Label>
                            <Select value={nodeType} onValueChange={setNodeType}>
                                <SelectTrigger className="w-36 h-8"><SelectValue /></SelectTrigger>
                                <SelectContent>
                                    {["Semantic", "Episodic", "Concept", "DreamSnapshot", "Somatic", "Linguistic", "Composite"].map((t) => (
                                        <SelectItem key={t} value={t}>{t}</SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>
                        <div className="space-y-1">
                            <Label className="text-xs">Energy</Label>
                            <Input type="number" step="0.1" value={nodeEnergy} onChange={(e) => setNodeEnergy(e.target.value)} className="w-24 h-8" />
                        </div>
                    </div>
                    <div className="space-y-1">
                        <Label className="text-xs">Content (JSON)</Label>
                        <textarea
                            value={nodeContent}
                            onChange={(e) => setNodeContent(e.target.value)}
                            className="w-full h-20 rounded-md border border-input bg-background px-3 py-2 text-xs font-mono resize-y"
                            spellCheck={false}
                        />
                    </div>
                    <div className="flex items-center gap-3">
                        <Button onClick={handleInsert} disabled={insertLoading} size="sm">
                            {insertLoading ? <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" /> : <Plus className="h-3.5 w-3.5 mr-1" />}
                            Insert
                        </Button>
                        {insertResult && <span className="text-xs font-mono text-muted-foreground">{insertResult}</span>}
                    </div>
                </CardContent>
            </Card>

            {/* Get */}
            <Card>
                <CardHeader className="pb-3"><CardTitle className="text-base">Get Node</CardTitle></CardHeader>
                <CardContent className="space-y-3">
                    <div className="flex gap-2">
                        <Input value={getNodeId} onChange={(e) => setGetNodeId(e.target.value)} placeholder="Node UUID" className="flex-1" />
                        <Button onClick={handleGet} disabled={getLoading || !getNodeId.trim()} size="sm">
                            <Search className="h-3.5 w-3.5 mr-1" /> Get
                        </Button>
                    </div>
                    {getResult && (
                        <pre className="text-xs font-mono bg-muted p-3 rounded-lg overflow-auto max-h-48">
                            {JSON.stringify(getResult, null, 2)}
                        </pre>
                    )}
                </CardContent>
            </Card>

            {/* Delete */}
            <Card>
                <CardHeader className="pb-3"><CardTitle className="text-base">Delete Node</CardTitle></CardHeader>
                <CardContent className="space-y-3">
                    <div className="flex gap-2">
                        <Input value={delNodeId} onChange={(e) => setDelNodeId(e.target.value)} placeholder="Node UUID" className="flex-1" />
                        <Button onClick={handleDelete} variant="destructive" disabled={!delNodeId.trim()} size="sm">
                            <Trash2 className="h-3.5 w-3.5 mr-1" /> Delete
                        </Button>
                    </div>
                    {delResult && <span className="text-xs font-mono text-muted-foreground">{delResult}</span>}
                </CardContent>
            </Card>
        </div>
    )
}

/* ── Edge CRUD ────────────────────────────────────────────── */
function EdgeCrudTab() {
    const [from, setFrom] = useState("")
    const [to, setTo] = useState("")
    const [edgeType, setEdgeType] = useState("Association")
    const [weight, setWeight] = useState("1.0")
    const [insertResult, setInsertResult] = useState<string | null>(null)
    const [insertLoading, setInsertLoading] = useState(false)
    const [delEdgeId, setDelEdgeId] = useState("")
    const [delResult, setDelResult] = useState<string | null>(null)

    const handleInsert = useCallback(async () => {
        setInsertLoading(true)
        try {
            const res = await insertEdge({ from: from.trim(), to: to.trim(), edge_type: edgeType, weight: parseFloat(weight) })
            setInsertResult(`Created: ${res.id}`)
        } catch (e: any) {
            setInsertResult(`Error: ${e.message}`)
        } finally {
            setInsertLoading(false)
        }
    }, [from, to, edgeType, weight])

    const handleDelete = useCallback(async () => {
        try {
            const res = await deleteEdge(delEdgeId.trim())
            setDelResult(`Deleted: ${res.deleted}`)
        } catch (e: any) {
            setDelResult(`Error: ${e.message}`)
        }
    }, [delEdgeId])

    return (
        <div className="space-y-4">
            <Card>
                <CardHeader className="pb-3"><CardTitle className="text-base">Insert Edge</CardTitle></CardHeader>
                <CardContent className="space-y-3">
                    <div className="flex flex-wrap gap-3 items-end">
                        <div className="space-y-1 flex-1">
                            <Label className="text-xs">From (UUID)</Label>
                            <Input value={from} onChange={(e) => setFrom(e.target.value)} placeholder="source node UUID" className="h-8" />
                        </div>
                        <div className="space-y-1 flex-1">
                            <Label className="text-xs">To (UUID)</Label>
                            <Input value={to} onChange={(e) => setTo(e.target.value)} placeholder="target node UUID" className="h-8" />
                        </div>
                        <div className="space-y-1">
                            <Label className="text-xs">Type</Label>
                            <Select value={edgeType} onValueChange={setEdgeType}>
                                <SelectTrigger className="w-40 h-8"><SelectValue /></SelectTrigger>
                                <SelectContent>
                                    {["Association", "Hierarchical", "LSystemGenerated", "Pruned"].map((t) => (
                                        <SelectItem key={t} value={t}>{t}</SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>
                        <div className="space-y-1">
                            <Label className="text-xs">Weight</Label>
                            <Input type="number" step="0.1" value={weight} onChange={(e) => setWeight(e.target.value)} className="w-20 h-8" />
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        <Button onClick={handleInsert} disabled={insertLoading || !from.trim() || !to.trim()} size="sm">
                            {insertLoading ? <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" /> : <Plus className="h-3.5 w-3.5 mr-1" />}
                            Insert Edge
                        </Button>
                        {insertResult && <span className="text-xs font-mono text-muted-foreground">{insertResult}</span>}
                    </div>
                </CardContent>
            </Card>

            <Card>
                <CardHeader className="pb-3"><CardTitle className="text-base">Delete Edge</CardTitle></CardHeader>
                <CardContent className="space-y-3">
                    <div className="flex gap-2">
                        <Input value={delEdgeId} onChange={(e) => setDelEdgeId(e.target.value)} placeholder="Edge UUID" className="flex-1" />
                        <Button onClick={handleDelete} variant="destructive" disabled={!delEdgeId.trim()} size="sm">
                            <Trash2 className="h-3.5 w-3.5 mr-1" /> Delete
                        </Button>
                    </div>
                    {delResult && <span className="text-xs font-mono text-muted-foreground">{delResult}</span>}
                </CardContent>
            </Card>
        </div>
    )
}

/* ── Batch Import ─────────────────────────────────────────── */
function BatchTab() {
    const [nodesJson, setNodesJson] = useState('[\n  {"node_type": "Concept", "energy": 0.8, "content": {"label": "batch1"}}\n]')
    const [edgesJson, setEdgesJson] = useState('[\n  {"from": "uuid1", "to": "uuid2", "edge_type": "Association", "weight": 1.0}\n]')
    const [nodesResult, setNodesResult] = useState<string | null>(null)
    const [edgesResult, setEdgesResult] = useState<string | null>(null)
    const [loading, setLoading] = useState(false)

    const handleBatchNodes = useCallback(async () => {
        setLoading(true)
        try {
            const nodes = JSON.parse(nodesJson)
            const res = await batchInsertNodes(nodes)
            setNodesResult(`Inserted ${res.inserted} nodes: ${res.node_ids.slice(0, 3).join(", ")}${res.node_ids.length > 3 ? "..." : ""}`)
        } catch (e: any) {
            setNodesResult(`Error: ${e.message}`)
        } finally {
            setLoading(false)
        }
    }, [nodesJson])

    const handleBatchEdges = useCallback(async () => {
        setLoading(true)
        try {
            const edges = JSON.parse(edgesJson)
            const res = await batchInsertEdges(edges)
            setEdgesResult(`Inserted ${res.inserted} edges: ${res.edge_ids.slice(0, 3).join(", ")}${res.edge_ids.length > 3 ? "..." : ""}`)
        } catch (e: any) {
            setEdgesResult(`Error: ${e.message}`)
        } finally {
            setLoading(false)
        }
    }, [edgesJson])

    return (
        <div className="space-y-4">
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2">
                        <Upload className="h-4 w-4" /> Batch Insert Nodes
                    </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                    <textarea
                        value={nodesJson}
                        onChange={(e) => setNodesJson(e.target.value)}
                        className="w-full h-28 rounded-md border border-input bg-background px-3 py-2 text-xs font-mono resize-y"
                        spellCheck={false}
                    />
                    <div className="flex items-center gap-3">
                        <Button onClick={handleBatchNodes} disabled={loading} size="sm">
                            <Upload className="h-3.5 w-3.5 mr-1" /> Import Nodes
                        </Button>
                        {nodesResult && <span className="text-xs font-mono text-muted-foreground">{nodesResult}</span>}
                    </div>
                </CardContent>
            </Card>

            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2">
                        <Upload className="h-4 w-4" /> Batch Insert Edges
                    </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                    <textarea
                        value={edgesJson}
                        onChange={(e) => setEdgesJson(e.target.value)}
                        className="w-full h-28 rounded-md border border-input bg-background px-3 py-2 text-xs font-mono resize-y"
                        spellCheck={false}
                    />
                    <div className="flex items-center gap-3">
                        <Button onClick={handleBatchEdges} disabled={loading} size="sm">
                            <Upload className="h-3.5 w-3.5 mr-1" /> Import Edges
                        </Button>
                        {edgesResult && <span className="text-xs font-mono text-muted-foreground">{edgesResult}</span>}
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}
