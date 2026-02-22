import { useState, useEffect } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { FileJson, Plus, Trash2, Loader2, Copy, Check, AlertCircle } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { api, DEFAULT_COLLECTION, listSchemas, setSchema, deleteSchema, type SchemaConstraint } from "@/lib/api"
import { generateSchemaCode } from "@/lib/sdkCodegen"

export default function SchemaManagerPage() {
    const [collection, setCollection] = useState(DEFAULT_COLLECTION)
    const [collections, setCollections] = useState<string[]>([])

    useEffect(() => {
        api.get("/collections").then(r => {
            const names = (r.data as { name: string }[]).map(c => c.name)
            setCollections(names)
        }).catch(() => {})
    }, [])

    return (
        <div className="space-y-6 fade-in">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold flex items-center gap-2">
                        <FileJson className="h-6 w-6" /> Schema Manager
                    </h1>
                    <p className="text-sm text-muted-foreground mt-1">
                        Define, manage and export node type schemas — generates Pydantic OGM and TypeScript code
                    </p>
                </div>
                <div className="w-48">
                    <Select value={collection} onValueChange={setCollection}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                            {collections.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                        </SelectContent>
                    </Select>
                </div>
            </div>

            <Tabs defaultValue="list" className="space-y-4">
                <TabsList className="flex-wrap h-auto gap-1">
                    <TabsTrigger value="list"><FileJson className="h-3.5 w-3.5 mr-1" />Schemas</TabsTrigger>
                    <TabsTrigger value="create"><Plus className="h-3.5 w-3.5 mr-1" />Create Schema</TabsTrigger>
                    <TabsTrigger value="code">{"</>"} Code Generator</TabsTrigger>
                </TabsList>

                <TabsContent value="list">
                    <SchemaListTab collection={collection} />
                </TabsContent>
                <TabsContent value="create">
                    <CreateSchemaTab collection={collection} />
                </TabsContent>
                <TabsContent value="code">
                    <CodeGeneratorTab collection={collection} />
                </TabsContent>
            </Tabs>
        </div>
    )
}

// ── Schema List Tab ─────────────────────────────────────────

function SchemaListTab({ collection }: { collection: string }) {
    const queryClient = useQueryClient()
    const { data, isLoading, error } = useQuery({
        queryKey: ["schemas", collection],
        queryFn: () => listSchemas(collection),
    })

    const deleteMut = useMutation({
        mutationFn: (nodeType: string) => deleteSchema(nodeType, collection),
        onSuccess: () => queryClient.invalidateQueries({ queryKey: ["schemas", collection] }),
    })

    if (isLoading) return <LoadingSkeleton />
    if (error) return <ErrorCard message={(error as Error).message} />

    const schemas = data?.schemas || []

    if (schemas.length === 0) {
        return (
            <Card>
                <CardContent className="py-12 text-center text-muted-foreground">
                    No schemas defined yet. Create one in the "Create Schema" tab.
                </CardContent>
            </Card>
        )
    }

    return (
        <Card>
            <CardHeader className="pb-3">
                <CardTitle className="text-base">Schemas in "{collection}"</CardTitle>
                <CardDescription>{schemas.length} schema(s) defined</CardDescription>
            </CardHeader>
            <CardContent>
                <ScrollArea className="max-h-[500px]">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Node Type</TableHead>
                                <TableHead>Required Fields</TableHead>
                                <TableHead>Field Types</TableHead>
                                <TableHead className="w-24">Actions</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {schemas.map(s => (
                                <TableRow key={s.node_type}>
                                    <TableCell className="font-mono font-medium">{s.node_type}</TableCell>
                                    <TableCell>
                                        <div className="flex flex-wrap gap-1">
                                            {s.required_fields.map(f => <Badge key={f} variant="secondary" className="text-xs">{f}</Badge>)}
                                        </div>
                                    </TableCell>
                                    <TableCell>
                                        <div className="flex flex-wrap gap-1">
                                            {s.field_types.map(f => (
                                                <Badge key={f.field_name} variant="outline" className="text-xs">
                                                    {f.field_name}: {f.field_type}
                                                </Badge>
                                            ))}
                                        </div>
                                    </TableCell>
                                    <TableCell>
                                        <Button
                                            variant="ghost" size="icon"
                                            className="h-8 w-8 text-destructive hover:text-destructive"
                                            onClick={() => deleteMut.mutate(s.node_type)}
                                            disabled={deleteMut.isPending}
                                        >
                                            <Trash2 className="h-4 w-4" />
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </ScrollArea>
            </CardContent>
        </Card>
    )
}

// ── Create Schema Tab ───────────────────────────────────────

function CreateSchemaTab({ collection }: { collection: string }) {
    const queryClient = useQueryClient()
    const [nodeType, setNodeType] = useState("")
    const [requiredFields, setRequiredFields] = useState<string[]>([])
    const [fieldTypes, setFieldTypes] = useState<{ field_name: string; field_type: string }[]>([
        { field_name: "", field_type: "string" },
    ])
    const [newRequired, setNewRequired] = useState("")

    const saveMut = useMutation({
        mutationFn: () => setSchema({ node_type: nodeType, required_fields: requiredFields, field_types: fieldTypes.filter(f => f.field_name) }, collection),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["schemas", collection] })
            setNodeType(""); setRequiredFields([]); setFieldTypes([{ field_name: "", field_type: "string" }])
        },
    })

    return (
        <Card>
            <CardHeader className="pb-3">
                <CardTitle className="text-base">Create Node Type Schema</CardTitle>
                <CardDescription>Define the structure for a node type. The schema validates nodes on insert.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Node Type */}
                <div className="space-y-1">
                    <Label>Node Type</Label>
                    <Select value={nodeType} onValueChange={setNodeType}>
                        <SelectTrigger><SelectValue placeholder="Select or type custom..." /></SelectTrigger>
                        <SelectContent>
                            {["Semantic", "Episodic", "Concept", "DreamSnapshot", "Somatic", "Linguistic", "Composite"].map(t =>
                                <SelectItem key={t} value={t}>{t}</SelectItem>
                            )}
                        </SelectContent>
                    </Select>
                    <Input className="mt-2" placeholder="Or type a custom node type..." value={nodeType} onChange={e => setNodeType(e.target.value)} />
                </div>

                {/* Required Fields */}
                <div className="space-y-2">
                    <Label>Required Fields</Label>
                    <div className="flex flex-wrap gap-1 min-h-[28px]">
                        {requiredFields.map(f => (
                            <Badge key={f} variant="secondary" className="gap-1">
                                {f}
                                <button className="ml-1 hover:text-destructive" onClick={() => setRequiredFields(rf => rf.filter(x => x !== f))}>×</button>
                            </Badge>
                        ))}
                    </div>
                    <div className="flex gap-2">
                        <Input placeholder="Field name..." value={newRequired} onChange={e => setNewRequired(e.target.value)}
                            onKeyDown={e => { if (e.key === "Enter" && newRequired.trim()) { setRequiredFields(r => [...r, newRequired.trim()]); setNewRequired("") } }} />
                        <Button variant="outline" size="sm" onClick={() => { if (newRequired.trim()) { setRequiredFields(r => [...r, newRequired.trim()]); setNewRequired("") } }}>Add</Button>
                    </div>
                </div>

                {/* Field Types */}
                <div className="space-y-2">
                    <Label>Field Types</Label>
                    {fieldTypes.map((ft, i) => (
                        <div key={i} className="flex gap-2 items-center">
                            <Input placeholder="field_name" value={ft.field_name}
                                onChange={e => { const next = [...fieldTypes]; next[i] = { ...next[i], field_name: e.target.value }; setFieldTypes(next) }} />
                            <Select value={ft.field_type} onValueChange={v => { const next = [...fieldTypes]; next[i] = { ...next[i], field_type: v }; setFieldTypes(next) }}>
                                <SelectTrigger className="w-36"><SelectValue /></SelectTrigger>
                                <SelectContent>
                                    {["string", "number", "bool", "array", "object"].map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}
                                </SelectContent>
                            </Select>
                            <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0" onClick={() => setFieldTypes(f => f.filter((_, j) => j !== i))} disabled={fieldTypes.length <= 1}>
                                <Trash2 className="h-3.5 w-3.5" />
                            </Button>
                        </div>
                    ))}
                    <Button variant="outline" size="sm" onClick={() => setFieldTypes(f => [...f, { field_name: "", field_type: "string" }])}>
                        <Plus className="h-3.5 w-3.5 mr-1" /> Add Field
                    </Button>
                </div>

                {/* Save */}
                <Button onClick={() => saveMut.mutate()} disabled={!nodeType || saveMut.isPending} className="w-full">
                    {saveMut.isPending ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Saving...</> : "Save Schema"}
                </Button>

                {saveMut.isError && <p className="text-sm text-destructive">{(saveMut.error as Error).message}</p>}
                {saveMut.isSuccess && <p className="text-sm text-emerald-500">Schema saved successfully!</p>}
            </CardContent>
        </Card>
    )
}

// ── Code Generator Tab ──────────────────────────────────────

function CodeGeneratorTab({ collection }: { collection: string }) {
    const { data } = useQuery({
        queryKey: ["schemas", collection],
        queryFn: () => listSchemas(collection),
    })
    const schemas = data?.schemas || []
    const [selected, setSelected] = useState<SchemaConstraint | null>(null)
    const [copied, setCopied] = useState<string | null>(null)

    useEffect(() => {
        if (schemas.length > 0 && !selected) setSelected(schemas[0])
    }, [schemas, selected])

    const code = selected ? generateSchemaCode({
        nodeType: selected.node_type,
        requiredFields: selected.required_fields,
        fieldTypes: selected.field_types,
    }) : null

    const copyToClipboard = (text: string, lang: string) => {
        navigator.clipboard.writeText(text)
        setCopied(lang)
        setTimeout(() => setCopied(null), 2000)
    }

    if (schemas.length === 0) {
        return (
            <Card>
                <CardContent className="py-12 text-center text-muted-foreground">
                    No schemas to generate code from. Create one first.
                </CardContent>
            </Card>
        )
    }

    return (
        <div className="space-y-4">
            {/* Schema Selector */}
            <Card>
                <CardContent className="pt-4">
                    <Label>Select Schema</Label>
                    <Select value={selected?.node_type || ""} onValueChange={v => setSelected(schemas.find(s => s.node_type === v) || null)}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                            {schemas.map(s => <SelectItem key={s.node_type} value={s.node_type}>{s.node_type}</SelectItem>)}
                        </SelectContent>
                    </Select>
                </CardContent>
            </Card>

            {code && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    <CodeBlock title="Pydantic OGM (Python)" lang="python" code={code.python} copied={copied} onCopy={copyToClipboard} />
                    <CodeBlock title="TypeScript Interface" lang="typescript" code={code.typescript} copied={copied} onCopy={copyToClipboard} />
                    <CodeBlock title="NQL CREATE Example" lang="nql" code={code.nql} copied={copied} onCopy={copyToClipboard} />
                </div>
            )}
        </div>
    )
}

// ── Shared Components ───────────────────────────────────────

function CodeBlock({ title, lang, code, copied, onCopy }: {
    title: string; lang: string; code: string; copied: string | null
    onCopy: (text: string, lang: string) => void
}) {
    return (
        <Card>
            <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                    <CardTitle className="text-sm">{title}</CardTitle>
                    <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => onCopy(code, lang)}>
                        {copied === lang ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
                    </Button>
                </div>
            </CardHeader>
            <CardContent>
                <ScrollArea className="max-h-80">
                    <pre className="text-xs font-mono bg-muted/50 rounded-md p-3 whitespace-pre-wrap">{code}</pre>
                </ScrollArea>
            </CardContent>
        </Card>
    )
}

function LoadingSkeleton() {
    return (
        <Card>
            <CardContent className="space-y-3 pt-6">
                <Skeleton className="h-4 w-1/3" />
                <Skeleton className="h-20 w-full" />
                <Skeleton className="h-4 w-2/3" />
            </CardContent>
        </Card>
    )
}

function ErrorCard({ message }: { message: string }) {
    return (
        <Card className="border-destructive/30">
            <CardContent className="pt-6 flex items-center gap-2 text-destructive">
                <AlertCircle className="h-4 w-4" />
                <span className="text-sm font-mono">{message}</span>
            </CardContent>
        </Card>
    )
}
