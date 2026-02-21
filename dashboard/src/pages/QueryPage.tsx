import { useState, useRef, useEffect, useCallback } from "react"
import { useQuery } from "@tanstack/react-query"
import {
    Play, Trash2, Send, BookOpen, ChevronDown, ChevronRight,
    Search, Sparkles, Terminal, ArrowDownToLine,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
    Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import { Skeleton } from "@/components/ui/skeleton"
import { executeNql, fullTextSearch, api } from "@/lib/api"
import {
    generateNqlFromPrompt, extractNqlFromResponse,
    isAiConfigured, type Message,
} from "@/services/nqlAssistant"

/* ── NQL keyword lists for highlighting ───────────────────── */
const NQL_KEYWORDS = [
    "MATCH", "CREATE", "MERGE", "DELETE", "SET", "RETURN", "WHERE",
    "AND", "OR", "NOT", "ORDER BY", "ASC", "DESC", "LIMIT", "SKIP",
    "DISTINCT", "AS", "IN", "BETWEEN", "CONTAINS", "STARTS_WITH",
    "ENDS_WITH", "DREAM", "DIFFUSE", "NARRATE", "RECONSTRUCT",
    "EXPLAIN", "COUNTERFACTUAL", "TRANSLATE", "BEGIN", "COMMIT",
    "ROLLBACK", "ON CREATE SET", "ON MATCH SET", "FROM", "WITH",
    "DEPTH", "NOISE", "MAX_HOPS", "WINDOW", "FORMAT", "MODALITY",
    "QUALITY", "EVERY", "INTERVAL", "WHEN", "THEN", "DAEMON",
    "GROUP BY", "COUNT", "SUM", "AVG", "MIN", "MAX",
]

const NQL_EXAMPLES = [
    { label: "All nodes (limit 20)", nql: 'MATCH (n) LIMIT 20 RETURN n' },
    { label: "High energy nodes", nql: 'MATCH (n) WHERE n.energy > 0.8 ORDER BY n.energy DESC LIMIT 10 RETURN n' },
    { label: "Concepts only", nql: 'MATCH (n:Concept) RETURN n' },
    { label: "Path traversal", nql: 'MATCH (a)-[:Association]->(b) LIMIT 20 RETURN a, b' },
    { label: "Hierarchical edges", nql: 'MATCH (a)-[:Hierarchical]->(b) WHERE a.depth < 0.3 RETURN a, b' },
    { label: "Dream exploration", nql: 'DREAM FROM $embedding DEPTH 3 NOISE 0.1' },
    { label: "Heat diffusion", nql: 'DIFFUSE FROM $embedding WITH t = [0.1, 1.0, 10.0] MAX_HOPS 5' },
    { label: "Narrate last 24h", nql: 'NARRATE IN "default" WINDOW 24 FORMAT json' },
    { label: "Deep nodes", nql: 'MATCH (n) WHERE n.depth > 0.7 ORDER BY n.depth DESC LIMIT 10 RETURN n' },
    { label: "Low energy cleanup", nql: 'MATCH (n) WHERE n.energy < 0.05 RETURN n.id, n.energy, n.node_type' },
]

const NQL_REFERENCE = [
    { cmd: "MATCH", desc: "Pattern matching — MATCH (n:Type) WHERE ... RETURN ..." },
    { cmd: "CREATE", desc: "Insert node — CREATE (n:Concept {energy: 0.8, content: {...}})" },
    { cmd: "MERGE", desc: "Upsert — MERGE (n:Type {...}) ON CREATE SET ... ON MATCH SET ..." },
    { cmd: "DELETE", desc: "Remove matched — MATCH (n) WHERE ... DELETE n" },
    { cmd: "SET", desc: "Update field — MATCH (n) WHERE ... SET n.energy = 0.9" },
    { cmd: "DREAM", desc: "Speculative query — DREAM FROM $vec DEPTH 3 NOISE 0.1" },
    { cmd: "DIFFUSE", desc: "Heat kernel — DIFFUSE FROM $vec WITH t = [...] MAX_HOPS 5" },
    { cmd: "NARRATE", desc: "Graph narrative — NARRATE IN \"col\" WINDOW 24 FORMAT json" },
    { cmd: "RECONSTRUCT", desc: "Sensory rebuild — RECONSTRUCT $id MODALITY text QUALITY high" },
    { cmd: "EXPLAIN", desc: "Show plan — EXPLAIN MATCH (n) RETURN n" },
    { cmd: "COUNTERFACTUAL", desc: "What-if — COUNTERFACTUAL SET n.energy = 0 MATCH ..." },
    { cmd: "TRANSLATE", desc: "Cross-modal — TRANSLATE $id FROM audio TO text" },
    { cmd: "CREATE DAEMON", desc: "Agent — CREATE DAEMON name ON (n:T) WHEN ... THEN ... EVERY INTERVAL(\"1h\")" },
    { cmd: "Math funcs (13)", desc: "POINCARE_DIST, KLEIN_DIST, MINKOWSKI_NORM, GAUSS_KERNEL, HAUSDORFF_DIM, RIEMANN_CURVATURE, ..." },
]

interface ChatMsg {
    role: "user" | "assistant"
    content: string
}

export default function QueryPage() {
    const [nql, setNql] = useState("")
    const [results, setResults] = useState<unknown[] | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [running, setRunning] = useState(false)
    const [refOpen, setRefOpen] = useState(false)

    // Full-text search
    const [searchQ, setSearchQ] = useState("")
    const [searchResults, setSearchResults] = useState<{ node_id: string; score: number }[] | null>(null)
    const [searching, setSearching] = useState(false)

    // AI Chat
    const [chatInput, setChatInput] = useState("")
    const [chatMessages, setChatMessages] = useState<ChatMsg[]>([])
    const [chatLoading, setChatLoading] = useState(false)
    const chatEndRef = useRef<HTMLDivElement>(null)

    // Collections for context
    const { data: collections } = useQuery({
        queryKey: ["collections"],
        queryFn: () => api.get("/collections").then((r) => r.data),
    })

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [chatMessages])

    const runNql = useCallback(async () => {
        if (!nql.trim()) return
        setRunning(true)
        setError(null)
        setResults(null)
        try {
            const data = await executeNql(nql)
            if (data.error) setError(data.error)
            else setResults(data.nodes ?? [])
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : "Query failed")
        } finally {
            setRunning(false)
        }
    }, [nql])

    const runSearch = useCallback(async () => {
        if (!searchQ.trim()) return
        setSearching(true)
        try {
            const data = await fullTextSearch(searchQ, 20)
            setSearchResults(data.results ?? [])
        } catch {
            setSearchResults([])
        } finally {
            setSearching(false)
        }
    }, [searchQ])

    const sendChat = useCallback(async () => {
        if (!chatInput.trim()) return
        const userMsg: ChatMsg = { role: "user", content: chatInput }
        setChatMessages((prev) => [...prev, userMsg])
        setChatInput("")
        setChatLoading(true)
        try {
            const history: Message[] = chatMessages.map((m) => ({
                role: m.role,
                content: m.content,
            }))
            const response = await generateNqlFromPrompt(chatInput, history)
            setChatMessages((prev) => [...prev, { role: "assistant", content: response }])
        } catch {
            setChatMessages((prev) => [
                ...prev,
                { role: "assistant", content: "Error calling AI. Check your API configuration." },
            ])
        } finally {
            setChatLoading(false)
        }
    }, [chatInput, chatMessages])

    const insertNqlFromChat = useCallback(
        (content: string) => {
            const extracted = extractNqlFromResponse(content)
            if (extracted) setNql(extracted)
        },
        [],
    )

    const highlightNql = (text: string) => {
        let html = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        NQL_KEYWORDS.forEach((kw) => {
            const re = new RegExp(`\\b${kw}\\b`, "gi")
            html = html.replace(re, `<span class="text-primary font-bold">${kw}</span>`)
        })
        html = html.replace(/(["'])(?:(?=(\\?))\2.)*?\1/g, '<span class="text-emerald-400">$&</span>')
        html = html.replace(/\b\d+\.?\d*\b/g, '<span class="text-amber-400">$&</span>')
        return html
    }

    return (
        <div className="space-y-4 fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold flex items-center gap-2">
                        <Terminal className="h-6 w-6" /> NQL Console
                    </h1>
                    <p className="text-sm text-muted-foreground mt-1">
                        Nietzsche Query Language — {collections?.length ?? 0} collections available
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <Badge variant={isAiConfigured() ? "default" : "outline"}>
                        <Sparkles className="h-3 w-3 mr-1" />
                        Gemini {isAiConfigured() ? "ON" : "OFF"}
                    </Badge>
                </div>
            </div>

            <Tabs defaultValue="nql" className="space-y-4">
                <TabsList>
                    <TabsTrigger value="nql">NQL Editor</TabsTrigger>
                    <TabsTrigger value="search">Full-Text Search</TabsTrigger>
                </TabsList>

                {/* ── NQL Tab ─────────────────────────────────── */}
                <TabsContent value="nql">
                    <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
                        {/* Left: Editor + Results (3/5) */}
                        <div className="lg:col-span-3 space-y-4">
                            <Card>
                                <CardHeader className="pb-3">
                                    <div className="flex items-center justify-between">
                                        <CardTitle className="text-base">Query Editor</CardTitle>
                                        <div className="flex items-center gap-2">
                                            {/* Example selector */}
                                            <select
                                                className="h-8 rounded border border-input bg-background px-2 text-xs"
                                                onChange={(e) => {
                                                    if (e.target.value) setNql(e.target.value)
                                                }}
                                                defaultValue=""
                                            >
                                                <option value="" disabled>Examples...</option>
                                                {NQL_EXAMPLES.map((ex) => (
                                                    <option key={ex.label} value={ex.nql}>{ex.label}</option>
                                                ))}
                                            </select>
                                            <Button size="sm" variant="ghost" onClick={() => setNql("")}>
                                                <Trash2 className="h-3.5 w-3.5" />
                                            </Button>
                                            <Button size="sm" onClick={runNql} disabled={running || !nql.trim()}>
                                                <Play className="h-3.5 w-3.5 mr-1" />
                                                {running ? "Running..." : "Run"}
                                            </Button>
                                        </div>
                                    </div>
                                </CardHeader>
                                <CardContent>
                                    <div className="relative">
                                        <textarea
                                            value={nql}
                                            onChange={(e) => setNql(e.target.value)}
                                            onKeyDown={(e) => {
                                                if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                                                    e.preventDefault()
                                                    runNql()
                                                }
                                            }}
                                            placeholder="MATCH (n) WHERE n.energy > 0.5 LIMIT 10 RETURN n"
                                            className="w-full h-32 rounded-md border border-input bg-background px-3 py-2 text-sm font-mono resize-y placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                                            spellCheck={false}
                                        />
                                        {/* Syntax-highlighted preview */}
                                        {nql && (
                                            <div className="mt-2 p-2 rounded bg-muted/50 text-xs font-mono overflow-x-auto">
                                                <div dangerouslySetInnerHTML={{ __html: highlightNql(nql) }} />
                                            </div>
                                        )}
                                    </div>
                                    <p className="text-[11px] text-muted-foreground mt-2">Ctrl+Enter to run</p>
                                </CardContent>
                            </Card>

                            {/* Results */}
                            {error && (
                                <Card className="border-destructive/50">
                                    <CardContent className="pt-4">
                                        <p className="text-sm text-destructive font-mono">{error}</p>
                                    </CardContent>
                                </Card>
                            )}

                            {results !== null && (
                                <Card>
                                    <CardHeader className="pb-3">
                                        <CardTitle className="text-base flex items-center gap-2">
                                            Results <Badge variant="secondary">{results.length} nodes</Badge>
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        {results.length === 0 ? (
                                            <p className="text-sm text-muted-foreground">No results returned.</p>
                                        ) : (
                                            <div className="max-h-96 overflow-auto">
                                                <Table>
                                                    <TableHeader>
                                                        <TableRow>
                                                            <TableHead className="w-[200px]">ID</TableHead>
                                                            <TableHead>Type</TableHead>
                                                            <TableHead>Energy</TableHead>
                                                            <TableHead>Depth</TableHead>
                                                            <TableHead>Content</TableHead>
                                                        </TableRow>
                                                    </TableHeader>
                                                    <TableBody>
                                                        {results.map((node: any, i: number) => (
                                                            <TableRow key={node.id ?? i}>
                                                                <TableCell className="font-mono text-xs">
                                                                    {typeof node.id === "string" ? node.id.slice(0, 12) + "..." : String(i)}
                                                                </TableCell>
                                                                <TableCell>
                                                                    <Badge variant="outline">{node.node_type ?? "—"}</Badge>
                                                                </TableCell>
                                                                <TableCell className="font-mono">
                                                                    {typeof node.energy === "number" ? node.energy.toFixed(4) : "—"}
                                                                </TableCell>
                                                                <TableCell className="font-mono">
                                                                    {typeof node.depth === "number" ? node.depth.toFixed(4) : "—"}
                                                                </TableCell>
                                                                <TableCell className="max-w-[200px] truncate text-xs">
                                                                    {node.content ? JSON.stringify(node.content).slice(0, 80) : "—"}
                                                                </TableCell>
                                                            </TableRow>
                                                        ))}
                                                    </TableBody>
                                                </Table>
                                            </div>
                                        )}
                                    </CardContent>
                                </Card>
                            )}

                            {running && (
                                <Card>
                                    <CardContent className="pt-4 space-y-2">
                                        <Skeleton className="h-4 w-full" />
                                        <Skeleton className="h-4 w-3/4" />
                                        <Skeleton className="h-4 w-1/2" />
                                    </CardContent>
                                </Card>
                            )}

                            {/* Reference panel */}
                            <Card>
                                <CardHeader className="pb-2 cursor-pointer" onClick={() => setRefOpen(!refOpen)}>
                                    <CardTitle className="text-sm flex items-center gap-2">
                                        <BookOpen className="h-4 w-4" />
                                        NQL Quick Reference
                                        {refOpen ? <ChevronDown className="h-4 w-4 ml-auto" /> : <ChevronRight className="h-4 w-4 ml-auto" />}
                                    </CardTitle>
                                </CardHeader>
                                {refOpen && (
                                    <CardContent>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                                            {NQL_REFERENCE.map((r) => (
                                                <div key={r.cmd} className="text-xs border-b border-border/30 pb-1">
                                                    <span className="font-mono font-bold text-primary">{r.cmd}</span>
                                                    <span className="text-muted-foreground ml-2">{r.desc}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </CardContent>
                                )}
                            </Card>
                        </div>

                        {/* Right: AI Chat (2/5) */}
                        <div className="lg:col-span-2">
                            <Card className="h-[calc(100vh-220px)] flex flex-col">
                                <CardHeader className="pb-3 flex-shrink-0">
                                    <CardTitle className="text-base flex items-center gap-2">
                                        <Sparkles className="h-4 w-4" /> NQL AI Assistant
                                    </CardTitle>
                                    <p className="text-xs text-muted-foreground">
                                        Describe what you want in natural language
                                    </p>
                                </CardHeader>
                                <CardContent className="flex-1 flex flex-col overflow-hidden p-0">
                                    {/* Messages */}
                                    <ScrollArea className="flex-1 px-4">
                                        <div className="space-y-3 py-2">
                                            {chatMessages.length === 0 && (
                                                <div className="text-center text-xs text-muted-foreground py-8">
                                                    {isAiConfigured() ? (
                                                        <>
                                                            <Sparkles className="h-8 w-8 mx-auto mb-2 opacity-30" />
                                                            <p>Gemini-powered NQL assistant</p>
                                                            <p className="mt-1">e.g. &quot;Find all concepts with high energy&quot;</p>
                                                        </>
                                                    ) : (
                                                        <>
                                                            <Sparkles className="h-8 w-8 mx-auto mb-2 opacity-20" />
                                                            <p>AI not configured</p>
                                                            <p className="mt-1">Set VITE_GOOGLE_API_KEY for Gemini</p>
                                                            <p className="mt-1">Fallback: VITE_AI_API_KEY (OpenAI/Anthropic/Ollama)</p>
                                                        </>
                                                    )}
                                                </div>
                                            )}
                                            {chatMessages.map((msg, i) => (
                                                <div
                                                    key={i}
                                                    className={`text-sm rounded-lg px-3 py-2 ${
                                                        msg.role === "user"
                                                            ? "bg-primary/10 ml-8"
                                                            : "bg-muted mr-4"
                                                    }`}
                                                >
                                                    <div className="text-[10px] font-bold mb-1 text-muted-foreground uppercase">
                                                        {msg.role === "user" ? "You" : "AI"}
                                                    </div>
                                                    <div className="whitespace-pre-wrap font-mono text-xs leading-relaxed">
                                                        {msg.content}
                                                    </div>
                                                    {msg.role === "assistant" && extractNqlFromResponse(msg.content) && (
                                                        <Button
                                                            size="sm"
                                                            variant="outline"
                                                            className="mt-2 h-6 text-[10px]"
                                                            onClick={() => insertNqlFromChat(msg.content)}
                                                        >
                                                            <ArrowDownToLine className="h-3 w-3 mr-1" />
                                                            Insert to Editor
                                                        </Button>
                                                    )}
                                                </div>
                                            ))}
                                            {chatLoading && (
                                                <div className="bg-muted rounded-lg px-3 py-2 mr-4">
                                                    <Skeleton className="h-3 w-24 mb-2" />
                                                    <Skeleton className="h-3 w-full" />
                                                    <Skeleton className="h-3 w-3/4 mt-1" />
                                                </div>
                                            )}
                                            <div ref={chatEndRef} />
                                        </div>
                                    </ScrollArea>

                                    {/* Input */}
                                    <div className="flex-shrink-0 border-t p-3">
                                        <div className="flex gap-2">
                                            <Input
                                                value={chatInput}
                                                onChange={(e) => setChatInput(e.target.value)}
                                                onKeyDown={(e) => {
                                                    if (e.key === "Enter" && !e.shiftKey) {
                                                        e.preventDefault()
                                                        sendChat()
                                                    }
                                                }}
                                                placeholder="Find nodes with high energy..."
                                                className="text-sm"
                                                disabled={chatLoading}
                                            />
                                            <Button
                                                size="icon"
                                                onClick={sendChat}
                                                disabled={chatLoading || !chatInput.trim()}
                                            >
                                                <Send className="h-4 w-4" />
                                            </Button>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        </div>
                    </div>
                </TabsContent>

                {/* ── Full-Text Search Tab ────────────────────── */}
                <TabsContent value="search">
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-base flex items-center gap-2">
                                <Search className="h-4 w-4" /> Full-Text Search
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="flex gap-2">
                                <Input
                                    value={searchQ}
                                    onChange={(e) => setSearchQ(e.target.value)}
                                    onKeyDown={(e) => e.key === "Enter" && runSearch()}
                                    placeholder="Search for content..."
                                    className="flex-1"
                                />
                                <Button onClick={runSearch} disabled={searching || !searchQ.trim()}>
                                    <Search className="h-4 w-4 mr-1" />
                                    {searching ? "Searching..." : "Search"}
                                </Button>
                            </div>

                            {searchResults !== null && (
                                searchResults.length === 0 ? (
                                    <p className="text-sm text-muted-foreground">No results found.</p>
                                ) : (
                                    <Table>
                                        <TableHeader>
                                            <TableRow>
                                                <TableHead>Node ID</TableHead>
                                                <TableHead>Score</TableHead>
                                            </TableRow>
                                        </TableHeader>
                                        <TableBody>
                                            {searchResults.map((r) => (
                                                <TableRow key={r.node_id}>
                                                    <TableCell className="font-mono text-xs">{r.node_id}</TableCell>
                                                    <TableCell className="font-mono">{r.score.toFixed(6)}</TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                )
                            )}
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    )
}
