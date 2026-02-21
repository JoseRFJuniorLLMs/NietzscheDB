import { useState, useCallback, useEffect } from "react"
import {
    Moon, Play, Loader2, ArrowRight, CheckCircle2, XCircle,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
    Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import { ScrollArea } from "@/components/ui/scroll-area"
import { triggerSleep } from "@/lib/api"
import {
    BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Cell,
} from "recharts"

interface SleepResult {
    hausdorff_before: number
    hausdorff_after: number
    hausdorff_delta: number
    committed: boolean
    nodes_perturbed: number
    timestamp: string
    noise: number
    adam_steps: number
    hausdorff_threshold: number
}

export default function SleepPage() {
    const [noise, setNoise] = useState("0.02")
    const [adamSteps, setAdamSteps] = useState("10")
    const [threshold, setThreshold] = useState("0.15")
    const [running, setRunning] = useState(false)
    const [lastResult, setLastResult] = useState<SleepResult | null>(null)
    const [history, setHistory] = useState<SleepResult[]>(() => {
        try {
            const saved = localStorage.getItem("nietzsche_sleep_history")
            return saved ? JSON.parse(saved) : []
        } catch { return [] }
    })
    const [error, setError] = useState<string | null>(null)

    // Persist history to localStorage
    useEffect(() => {
        localStorage.setItem("nietzsche_sleep_history", JSON.stringify(history))
    }, [history])

    const execute = useCallback(async () => {
        setRunning(true)
        setError(null)
        try {
            const params = {
                noise: parseFloat(noise) || 0.02,
                adam_steps: parseInt(adamSteps) || 10,
                hausdorff_threshold: parseFloat(threshold) || 0.15,
            }
            const data = await triggerSleep(params)
            const result: SleepResult = {
                ...data,
                timestamp: new Date().toLocaleTimeString(),
                noise: params.noise,
                adam_steps: params.adam_steps,
                hausdorff_threshold: params.hausdorff_threshold,
            }
            setLastResult(result)
            setHistory((prev) => [...prev, result])
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : "Sleep cycle failed")
        } finally {
            setRunning(false)
        }
    }, [noise, adamSteps, threshold])

    const barData = lastResult
        ? [
              { name: "Before", value: lastResult.hausdorff_before, fill: "hsl(215.4 16.3% 56.9%)" },
              { name: "After", value: lastResult.hausdorff_after, fill: "hsl(263.4 70% 50.4%)" },
          ]
        : []

    const lineData = history.map((h, i) => ({
        cycle: i + 1,
        delta: h.hausdorff_delta,
        before: h.hausdorff_before,
        after: h.hausdorff_after,
    }))

    return (
        <div className="space-y-6 fade-in">
            <div>
                <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Moon className="h-6 w-6" /> Sleep & Reconsolidation
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                    RiemannianAdam reconsolidation cycle — nietzsche-sleep + nietzsche-sensory + nietzsche-dream
                </p>
            </div>

            {/* Parameters */}
            <Card>
                <CardHeader className="pb-3">
                    <CardTitle className="text-base">Sleep Parameters</CardTitle>
                    <CardDescription>Configure the reconsolidation cycle</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="flex flex-wrap gap-6 items-end">
                        <div className="space-y-1.5">
                            <Label className="text-xs">Noise (perturbation)</Label>
                            <Input
                                type="number"
                                step="0.005"
                                min="0"
                                max="0.5"
                                value={noise}
                                onChange={(e) => setNoise(e.target.value)}
                                className="w-32 h-9"
                            />
                            <p className="text-[10px] text-muted-foreground">Range: 0 — 0.5</p>
                        </div>
                        <div className="space-y-1.5">
                            <Label className="text-xs">Adam Steps</Label>
                            <Input
                                type="number"
                                step="1"
                                min="1"
                                max="200"
                                value={adamSteps}
                                onChange={(e) => setAdamSteps(e.target.value)}
                                className="w-32 h-9"
                            />
                            <p className="text-[10px] text-muted-foreground">Range: 1 — 200</p>
                        </div>
                        <div className="space-y-1.5">
                            <Label className="text-xs">Hausdorff Threshold</Label>
                            <Input
                                type="number"
                                step="0.01"
                                min="0"
                                max="1.0"
                                value={threshold}
                                onChange={(e) => setThreshold(e.target.value)}
                                className="w-32 h-9"
                            />
                            <p className="text-[10px] text-muted-foreground">Range: 0 — 1.0</p>
                        </div>
                        <Button onClick={execute} disabled={running} className="h-9">
                            {running ? (
                                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                            ) : (
                                <Play className="h-4 w-4 mr-1" />
                            )}
                            {running ? "Sleeping..." : "Trigger Sleep Cycle"}
                        </Button>
                    </div>
                </CardContent>
            </Card>

            {/* Error */}
            {error && (
                <Card className="border-destructive/50">
                    <CardContent className="pt-4">
                        <p className="text-sm text-destructive font-mono">{error}</p>
                    </CardContent>
                </Card>
            )}

            {/* Last Result */}
            {lastResult && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <Card>
                        <CardHeader className="pb-3">
                            <CardTitle className="text-base">Last Cycle Result</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="flex items-center justify-center gap-4 py-4">
                                <div className="text-center">
                                    <p className="text-xs text-muted-foreground mb-1">Before</p>
                                    <p className="text-2xl font-mono font-bold">{lastResult.hausdorff_before.toFixed(6)}</p>
                                </div>
                                <ArrowRight className="h-6 w-6 text-muted-foreground" />
                                <div className="text-center">
                                    <p className="text-xs text-muted-foreground mb-1">After</p>
                                    <p className="text-2xl font-mono font-bold">{lastResult.hausdorff_after.toFixed(6)}</p>
                                </div>
                            </div>
                            <div className="flex items-center justify-center gap-3 mt-2">
                                <Badge variant={lastResult.hausdorff_delta <= 0 ? "default" : "destructive"}>
                                    Delta: {lastResult.hausdorff_delta >= 0 ? "+" : ""}{lastResult.hausdorff_delta.toFixed(6)}
                                </Badge>
                                <Badge variant={lastResult.committed ? "default" : "secondary"}>
                                    {lastResult.committed ? (
                                        <><CheckCircle2 className="h-3 w-3 mr-1" /> Committed</>
                                    ) : (
                                        <><XCircle className="h-3 w-3 mr-1" /> Rejected</>
                                    )}
                                </Badge>
                                <Badge variant="outline">{lastResult.nodes_perturbed} nodes perturbed</Badge>
                            </div>
                        </CardContent>
                    </Card>

                    {/* Bar chart */}
                    <Card>
                        <CardHeader className="pb-2">
                            <CardTitle className="text-base">Hausdorff Comparison</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <ResponsiveContainer width="100%" height={200}>
                                <BarChart data={barData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(216 34% 17%)" />
                                    <XAxis dataKey="name" stroke="hsl(215.4 16.3% 56.9%)" fontSize={11} />
                                    <YAxis stroke="hsl(215.4 16.3% 56.9%)" fontSize={11} />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: "hsl(224 71% 4%)",
                                            border: "1px solid hsl(216 34% 17%)",
                                            borderRadius: "8px",
                                            fontSize: "12px",
                                        }}
                                    />
                                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                                        {barData.map((entry, index) => (
                                            <Cell key={index} fill={entry.fill} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </div>
            )}

            {/* History */}
            {history.length > 0 && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <Card>
                        <CardHeader className="pb-3">
                            <CardTitle className="text-base">
                                Sleep History <Badge variant="secondary">{history.length} cycles</Badge>
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <ScrollArea className="max-h-64">
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead>#</TableHead>
                                            <TableHead>Time</TableHead>
                                            <TableHead>Noise</TableHead>
                                            <TableHead>Steps</TableHead>
                                            <TableHead>Delta</TableHead>
                                            <TableHead>Status</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {history.map((h, i) => (
                                            <TableRow key={i}>
                                                <TableCell className="text-muted-foreground">{i + 1}</TableCell>
                                                <TableCell className="text-xs">{h.timestamp}</TableCell>
                                                <TableCell className="font-mono text-xs">{h.noise}</TableCell>
                                                <TableCell className="font-mono text-xs">{h.adam_steps}</TableCell>
                                                <TableCell className="font-mono text-xs">
                                                    <span className={h.hausdorff_delta <= 0 ? "text-emerald-400" : "text-red-400"}>
                                                        {h.hausdorff_delta >= 0 ? "+" : ""}{h.hausdorff_delta.toFixed(6)}
                                                    </span>
                                                </TableCell>
                                                <TableCell>
                                                    <Badge variant={h.committed ? "default" : "secondary"} className="text-[10px]">
                                                        {h.committed ? "OK" : "REJ"}
                                                    </Badge>
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </ScrollArea>
                        </CardContent>
                    </Card>

                    {/* Delta evolution chart */}
                    {lineData.length > 1 && (
                        <Card>
                            <CardHeader className="pb-2">
                                <CardTitle className="text-base">Delta Evolution</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <ResponsiveContainer width="100%" height={200}>
                                    <LineChart data={lineData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(216 34% 17%)" />
                                        <XAxis dataKey="cycle" stroke="hsl(215.4 16.3% 56.9%)" fontSize={11} />
                                        <YAxis stroke="hsl(215.4 16.3% 56.9%)" fontSize={11} />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: "hsl(224 71% 4%)",
                                                border: "1px solid hsl(216 34% 17%)",
                                                borderRadius: "8px",
                                                fontSize: "12px",
                                            }}
                                        />
                                        <Line type="monotone" dataKey="delta" stroke="hsl(263.4 70% 50.4%)" strokeWidth={2} dot />
                                    </LineChart>
                                </ResponsiveContainer>
                            </CardContent>
                        </Card>
                    )}
                </div>
            )}
        </div>
    )
}
