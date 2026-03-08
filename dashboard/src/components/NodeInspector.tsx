// NodeInspector.tsx — Slide-out drawer for detailed graph node inspection
// Shows node properties, neighbors, embedding preview, and actions

import { useCallback, useEffect, useRef, useState } from "react"
import {
  Copy,
  Crosshair,
  Loader2,
  Trash2,
  X,
  Zap,
} from "lucide-react"

import { cn } from "@/lib/utils"
import { api } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface NodeData {
  id: string
  node_type?: string
  energy?: number
  depth?: number
  hausdorff?: number
  created_at?: string
  lsystem_generation?: number
  content?: Record<string, unknown>
  embedding?: number[]
}

interface NeighborNode {
  id: string
  node_type?: string
  energy?: number
}

export interface NodeInspectorProps {
  nodeId: string | null
  onClose: () => void
  collection?: string
  onFocusNode?: (nodeId: string) => void
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const NODE_TYPE_COLORS: Record<string, string> = {
  Episodic: "#00ff66",
  Semantic: "#00f0ff",
  Concept: "#8b5cf6",
  DreamSnapshot: "#ff00ff",
}

function truncateId(id: string, len = 16): string {
  return id.length > len ? id.slice(0, len) + "\u2026" : id
}

function energyColor(energy: number): string {
  if (energy >= 0.8) return "#00ff66"
  if (energy >= 0.5) return "#f59e0b"
  if (energy >= 0.2) return "#fb923c"
  return "#ff4444"
}

function relativeTime(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime()
  const mins = Math.floor(diff / 60_000)
  if (mins < 1) return "just now"
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  return `${days}d ago`
}

// ---------------------------------------------------------------------------
// JSON Tree Renderer
// ---------------------------------------------------------------------------

function JsonTree({ data, depth = 0 }: { data: unknown; depth?: number }) {
  if (data === null || data === undefined) {
    return <span className="text-slate-500 italic">null</span>
  }
  if (typeof data === "boolean") {
    return <span className="text-amber-400">{String(data)}</span>
  }
  if (typeof data === "number") {
    return <span className="text-cyan-400">{data}</span>
  }
  if (typeof data === "string") {
    return <span className="text-green-400">"{data.length > 120 ? data.slice(0, 120) + "\u2026" : data}"</span>
  }
  if (Array.isArray(data)) {
    if (data.length === 0) return <span className="text-slate-500">[]</span>
    return (
      <div className="pl-3 border-l border-slate-700/50">
        {data.map((item, i) => (
          <div key={i} className="flex items-start gap-1">
            <span className="text-slate-600 text-[10px] font-mono select-none w-4 shrink-0 text-right">{i}</span>
            <JsonTree data={item} depth={depth + 1} />
          </div>
        ))}
      </div>
    )
  }
  if (typeof data === "object") {
    const entries = Object.entries(data as Record<string, unknown>)
    if (entries.length === 0) return <span className="text-slate-500">{"{}"}</span>
    return (
      <div className={cn(depth > 0 && "pl-3 border-l border-slate-700/50")}>
        {entries.map(([key, val]) => (
          <div key={key} className="flex items-start gap-1.5 py-0.5">
            <span className="text-purple-400 font-mono text-xs shrink-0">{key}:</span>
            <JsonTree data={val} depth={depth + 1} />
          </div>
        ))}
      </div>
    )
  }
  return <span className="text-slate-400">{String(data)}</span>
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function NodeInspector({ nodeId, onClose, collection, onFocusNode }: NodeInspectorProps) {
  const [node, setNode] = useState<NodeData | null>(null)
  const [neighbors, setNeighbors] = useState<NeighborNode[]>([])
  const [loading, setLoading] = useState(false)
  const [neighborsLoading, setNeighborsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [copied, setCopied] = useState(false)
  const [tab, setTab] = useState("properties")
  const backdropRef = useRef<HTMLDivElement>(null)

  const isOpen = nodeId !== null

  // Fetch node data
  useEffect(() => {
    if (!nodeId) { setNode(null); setError(null); return }
    setLoading(true)
    setError(null)
    setConfirmDelete(false)
    setTab("properties")
    api.get(`/node/${nodeId}`, { params: { collection: collection || "eva_core" } })
      .then((r) => setNode(r.data))
      .catch((e) => setError(e.response?.data?.error || "Failed to fetch node"))
      .finally(() => setLoading(false))
  }, [nodeId, collection])

  // Fetch neighbors when tab switches
  useEffect(() => {
    if (tab !== "neighbors" || !nodeId) return
    setNeighborsLoading(true)
    api.get("/query", {
      params: {
        nql: `MATCH (n)-[e]->(m) WHERE n.id = "${nodeId}" RETURN m LIMIT 20`,
        collection: collection || "eva_core",
      },
    })
      .then((r) => {
        const nodes = r.data?.nodes || r.data?.results || []
        setNeighbors(nodes)
      })
      .catch(() => setNeighbors([]))
      .finally(() => setNeighborsLoading(false))
  }, [tab, nodeId, collection])

  const handleCopyId = useCallback(() => {
    if (!nodeId) return
    navigator.clipboard.writeText(nodeId)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }, [nodeId])

  const handleDelete = useCallback(() => {
    if (!nodeId) return
    api.delete(`/node/${nodeId}`, { params: { collection: collection || "eva_core" } })
      .then(() => onClose())
      .catch(() => setError("Delete failed"))
  }, [nodeId, collection, onClose])

  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === backdropRef.current) onClose()
    },
    [onClose],
  )

  // Keyboard escape
  useEffect(() => {
    if (!isOpen) return
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose() }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [isOpen, onClose])

  const energy = node?.energy ?? 0
  const typeColor = NODE_TYPE_COLORS[node?.node_type || ""] || "#94a3b8"
  const embedding = node?.embedding || []

  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div
          ref={backdropRef}
          className="fixed inset-0 z-40 bg-black/40 backdrop-blur-[2px] transition-opacity duration-200"
          onClick={handleBackdropClick}
        />
      )}

      {/* Drawer */}
      <div
        className={cn(
          "fixed top-0 right-0 z-50 h-full w-[420px] border-l border-purple-900/50",
          "bg-slate-950/95 shadow-2xl shadow-purple-900/20",
          "transition-transform duration-300 ease-out",
          isOpen ? "translate-x-0" : "translate-x-full",
        )}
      >
        {isOpen && (
          <div className="flex h-full flex-col">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
              <div className="flex items-center gap-2 min-w-0">
                <Zap className="h-4 w-4 text-purple-400 shrink-0" />
                <span className="font-mono text-sm text-slate-200 truncate">
                  {nodeId ? truncateId(nodeId) : ""}
                </span>
                {node?.node_type && (
                  <Badge
                    variant="outline"
                    className="shrink-0 border-none px-1.5 py-0 text-[10px] font-mono"
                    style={{ backgroundColor: typeColor + "22", color: typeColor }}
                  >
                    {node.node_type}
                  </Badge>
                )}
              </div>
              <Button variant="ghost" size="icon" className="h-7 w-7 text-slate-400 hover:text-white" onClick={onClose}>
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Energy bar */}
            {node && (
              <div className="px-4 py-2 border-b border-slate-800/50">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">Energy</span>
                  <span className="text-xs font-mono" style={{ color: energyColor(energy) }}>
                    {energy.toFixed(4)}
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.min(energy * 100, 100)}%`,
                      backgroundColor: energyColor(energy),
                      boxShadow: `0 0 6px ${energyColor(energy)}60`,
                    }}
                  />
                </div>
              </div>
            )}

            {/* Loading / Error */}
            {loading && (
              <div className="flex items-center justify-center py-16">
                <Loader2 className="h-6 w-6 animate-spin text-purple-400" />
              </div>
            )}
            {error && (
              <div className="px-4 py-6 text-center text-sm text-red-400">{error}</div>
            )}

            {/* Tabs */}
            {node && !loading && (
              <Tabs value={tab} onValueChange={setTab} className="flex-1 flex flex-col min-h-0">
                <TabsList className="mx-4 mt-2 bg-slate-900 border border-slate-800">
                  <TabsTrigger value="properties" className="text-xs data-[state=active]:bg-purple-900/40">
                    Properties
                  </TabsTrigger>
                  <TabsTrigger value="neighbors" className="text-xs data-[state=active]:bg-purple-900/40">
                    Neighbors
                  </TabsTrigger>
                  <TabsTrigger value="embedding" className="text-xs data-[state=active]:bg-purple-900/40">
                    Embedding
                  </TabsTrigger>
                  <TabsTrigger value="history" className="text-xs data-[state=active]:bg-purple-900/40">
                    History
                  </TabsTrigger>
                </TabsList>

                <ScrollArea className="flex-1 min-h-0">
                  {/* Properties tab */}
                  <TabsContent value="properties" className="px-4 pb-4 mt-0">
                    <Card className="border-slate-800 bg-slate-900/50 mt-3">
                      <CardHeader className="pb-2 pt-3 px-3">
                        <CardTitle className="text-xs font-mono text-slate-400">Node Metadata</CardTitle>
                      </CardHeader>
                      <CardContent className="px-3 pb-3 space-y-2 text-xs">
                        <div className="flex justify-between">
                          <span className="text-slate-500">ID</span>
                          <span className="font-mono text-slate-300 truncate ml-2 max-w-[220px]">{node.id}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Type</span>
                          <span className="font-mono" style={{ color: typeColor }}>{node.node_type || "unknown"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Energy</span>
                          <span className="font-mono" style={{ color: energyColor(energy) }}>{energy.toFixed(6)}</span>
                        </div>
                        {node.depth !== undefined && (
                          <div className="flex justify-between">
                            <span className="text-slate-500">Depth</span>
                            <span className="font-mono text-cyan-400">{node.depth.toFixed(4)}</span>
                          </div>
                        )}
                        {node.hausdorff !== undefined && (
                          <div className="flex justify-between">
                            <span className="text-slate-500">Hausdorff</span>
                            <span className="font-mono text-amber-400">{node.hausdorff.toFixed(6)}</span>
                          </div>
                        )}
                        {node.created_at && (
                          <div className="flex justify-between">
                            <span className="text-slate-500">Created</span>
                            <span className="font-mono text-slate-400">{relativeTime(node.created_at)}</span>
                          </div>
                        )}
                        {node.lsystem_generation !== undefined && (
                          <div className="flex justify-between">
                            <span className="text-slate-500">L-System Gen</span>
                            <span className="font-mono text-purple-400">{node.lsystem_generation}</span>
                          </div>
                        )}
                      </CardContent>
                    </Card>

                    {node.content && Object.keys(node.content).length > 0 && (
                      <Card className="border-slate-800 bg-slate-900/50 mt-3">
                        <CardHeader className="pb-2 pt-3 px-3">
                          <CardTitle className="text-xs font-mono text-slate-400">Content</CardTitle>
                        </CardHeader>
                        <CardContent className="px-3 pb-3 text-xs">
                          <JsonTree data={node.content} />
                        </CardContent>
                      </Card>
                    )}
                  </TabsContent>

                  {/* Neighbors tab */}
                  <TabsContent value="neighbors" className="px-4 pb-4 mt-0">
                    {neighborsLoading ? (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 className="h-5 w-5 animate-spin text-purple-400" />
                      </div>
                    ) : neighbors.length === 0 ? (
                      <div className="py-8 text-center text-xs text-slate-500">No neighbors found</div>
                    ) : (
                      <div className="space-y-1.5 mt-3">
                        {neighbors.map((n, i) => (
                          <div
                            key={n.id || i}
                            className="flex items-center justify-between rounded border border-slate-800 bg-slate-900/50 px-3 py-2 hover:border-purple-800/50 transition-colors cursor-pointer"
                            onClick={() => onFocusNode?.(n.id)}
                          >
                            <div className="flex items-center gap-2 min-w-0">
                              <div
                                className="h-2 w-2 rounded-full shrink-0"
                                style={{ backgroundColor: NODE_TYPE_COLORS[n.node_type || ""] || "#94a3b8" }}
                              />
                              <span className="font-mono text-xs text-slate-300 truncate">{truncateId(n.id, 24)}</span>
                            </div>
                            {n.energy !== undefined && (
                              <span className="text-[10px] font-mono" style={{ color: energyColor(n.energy) }}>
                                {n.energy.toFixed(3)}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </TabsContent>

                  {/* Embedding tab */}
                  <TabsContent value="embedding" className="px-4 pb-4 mt-0">
                    <div className="mt-3 flex items-center gap-2 mb-3">
                      <span className="text-xs text-slate-500">Dimensions:</span>
                      <Badge variant="outline" className="border-cyan-800/50 text-cyan-400 text-[10px] font-mono">
                        {embedding.length}
                      </Badge>
                    </div>
                    {embedding.length === 0 ? (
                      <div className="py-8 text-center text-xs text-slate-500">No embedding vector</div>
                    ) : (
                      <div className="space-y-1">
                        {embedding.slice(0, 16).map((val, i) => {
                          const absMax = Math.max(...embedding.slice(0, 16).map(Math.abs), 0.001)
                          const normalized = val / absMax
                          const width = Math.abs(normalized) * 100
                          const isPositive = val >= 0
                          return (
                            <div key={i} className="flex items-center gap-2">
                              <span className="text-[10px] font-mono text-slate-600 w-5 text-right shrink-0">{i}</span>
                              <div className="flex-1 h-3 bg-slate-900 rounded-sm overflow-hidden relative">
                                <div className="absolute inset-0 flex items-center">
                                  <div className="w-1/2 flex justify-end">
                                    {!isPositive && (
                                      <div
                                        className="h-2 rounded-l-sm"
                                        style={{
                                          width: `${width}%`,
                                          backgroundColor: "#f87171",
                                          boxShadow: "0 0 4px #f8717140",
                                        }}
                                      />
                                    )}
                                  </div>
                                  <div className="w-px h-full bg-slate-700" />
                                  <div className="w-1/2">
                                    {isPositive && (
                                      <div
                                        className="h-2 rounded-r-sm"
                                        style={{
                                          width: `${width}%`,
                                          backgroundColor: "#00ff66",
                                          boxShadow: "0 0 4px #00ff6640",
                                        }}
                                      />
                                    )}
                                  </div>
                                </div>
                              </div>
                              <span className="text-[10px] font-mono text-slate-500 w-16 text-right shrink-0">
                                {val.toFixed(4)}
                              </span>
                            </div>
                          )
                        })}
                        {embedding.length > 16 && (
                          <div className="text-center text-[10px] text-slate-600 pt-1">
                            +{embedding.length - 16} more dimensions
                          </div>
                        )}
                      </div>
                    )}
                  </TabsContent>

                  {/* History tab */}
                  <TabsContent value="history" className="px-4 pb-4 mt-0">
                    <Card className="border-slate-800 bg-slate-900/50 mt-3">
                      <CardContent className="py-8 text-center">
                        <div className="text-slate-600 text-xs font-mono">Coming soon</div>
                        <div className="text-slate-700 text-[10px] mt-1">
                          Energy changes over time will be tracked here
                        </div>
                      </CardContent>
                    </Card>
                  </TabsContent>
                </ScrollArea>
              </Tabs>
            )}

            {/* Actions footer */}
            {node && !loading && (
              <div className="border-t border-slate-800 px-4 py-3 flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs text-slate-400 hover:text-white gap-1.5"
                  onClick={handleCopyId}
                >
                  <Copy className="h-3 w-3" />
                  {copied ? "Copied!" : "Copy ID"}
                </Button>
                {onFocusNode && nodeId && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 text-xs text-slate-400 hover:text-cyan-400 gap-1.5"
                    onClick={() => onFocusNode(nodeId)}
                  >
                    <Crosshair className="h-3 w-3" />
                    Focus
                  </Button>
                )}
                <div className="flex-1" />
                {confirmDelete ? (
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] text-red-400">Sure?</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 text-xs text-red-400 hover:bg-red-950 hover:text-red-300"
                      onClick={handleDelete}
                    >
                      Confirm
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 text-xs text-slate-500"
                      onClick={() => setConfirmDelete(false)}
                    >
                      Cancel
                    </Button>
                  </div>
                ) : (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 text-xs text-red-500/70 hover:text-red-400 hover:bg-red-950/50 gap-1.5"
                    onClick={() => setConfirmDelete(true)}
                  >
                    <Trash2 className="h-3 w-3" />
                    Delete
                  </Button>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </>
  )
}

export default NodeInspector
