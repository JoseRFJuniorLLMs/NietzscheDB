import axios from "axios"

export const api = axios.create({
    baseURL: "/api",
})

export const setAuthToken = (token: string) => {
    api.defaults.headers.common["x-api-key"] = token
}

api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            // Only clear if we're not on login page already
            if (localStorage.getItem("nietzsche_api_key") && !window.location.pathname.includes("/login")) {
                localStorage.removeItem("nietzsche_api_key")
                window.location.href = "/login"
            }
        }
        return Promise.reject(error)
    }
)

const token = localStorage.getItem("nietzsche_api_key")
if (token) setAuthToken(token)

export const fetchStatus = async () => {
    const [health, stats] = await Promise.allSettled([
        api.get("/health"),
        api.get("/stats"),
    ])
    const s = stats.status === "fulfilled" ? stats.value.data : {}
    const h = health.status === "fulfilled" ? health.value.data : {}
    return {
        status: h.status === "ok" ? "ONLINE" : "OFFLINE",
        version: s.version ?? "0.1.0",
        config: { dimension: 3072, metric: "cosine" },
        uptime: s.uptime ?? null,
        embedding: { enabled: !!s.embedding_enabled },
        _raw: { ...s, ...h },
    }
}

/* ── NQL & Search ─────────────────────────────────────────── */
export const executeNql = (nql: string) =>
    api.post("/query", { nql }).then((r) => r.data as { nodes: unknown[]; error: string | null })

export const fullTextSearch = (q: string, limit = 10) =>
    api.get("/search", { params: { q, limit } }).then((r) => r.data as { query: string; results: { node_id: string; score: number }[] })

/* ── Node CRUD ────────────────────────────────────────────── */
export const getNode = (id: string) =>
    api.get(`/node/${id}`).then((r) => r.data)

export const insertNode = (data: { id?: string; node_type?: string; energy?: number; content?: object; embedding?: number[] }) =>
    api.post("/node", data).then((r) => r.data as { id: string })

export const deleteNode = (id: string) =>
    api.delete(`/node/${id}`).then((r) => r.data as { deleted: string })

/* ── Edge CRUD ────────────────────────────────────────────── */
export const insertEdge = (data: { from: string; to: string; edge_type?: string; weight?: number }) =>
    api.post("/edge", data).then((r) => r.data as { id: string })

export const deleteEdge = (id: string) =>
    api.delete(`/edge/${id}`).then((r) => r.data as { deleted: string })

/* ── Batch ────────────────────────────────────────────────── */
export const batchInsertNodes = (nodes: object[]) =>
    api.post("/batch/nodes", { nodes }).then((r) => r.data as { inserted: number; node_ids: string[] })

export const batchInsertEdges = (edges: object[]) =>
    api.post("/batch/edges", { edges }).then((r) => r.data as { inserted: number; edge_ids: string[] })

/* ── Graph Algorithms ─────────────────────────────────────── */
export interface AlgoParams {
    collection?: string
    damping?: number
    iterations?: number
    resolution?: number
    sample?: number
    direction?: string
    top_k?: number
    threshold?: number
}

export const runAlgorithm = (name: string, params: AlgoParams = {}) =>
    api.get(`/algo/${name}`, { params }).then((r) => r.data)

/* ── Sleep ────────────────────────────────────────────────── */
export const triggerSleep = (params: { noise?: number; adam_steps?: number; hausdorff_threshold?: number } = {}) =>
    api.post("/sleep", params).then((r) => r.data as {
        hausdorff_before: number; hausdorff_after: number; hausdorff_delta: number
        committed: boolean; nodes_perturbed: number
    })

/* ── Backup & Export ──────────────────────────────────────── */
export const createBackup = (label?: string) =>
    api.post("/backup", { label: label || "manual" }).then((r) => r.data as { label: string; path: string; created_at: string; size_bytes: number })

export const listBackups = () =>
    api.get("/backup").then((r) => r.data as { backups: { label: string; path: string; created_at: string; size_bytes: number }[] })

export const exportData = (type: "nodes" | "edges", format: "jsonl" | "csv" = "jsonl") =>
    api.get(`/export/${type}`, { params: { format }, responseType: "blob" }).then((r) => r.data as Blob)

/* ── Agency ───────────────────────────────────────────────── */
export const getAgencyHealth = () =>
    api.get("/agency/health").then((r) => r.data)

export const getAgencyHealthLatest = () =>
    api.get("/agency/health/latest").then((r) => r.data)

export const getObserver = () =>
    api.get("/agency/observer").then((r) => r.data as { observer_id: string; energy: number; depth: number; hausdorff_local: number; is_observer: boolean; content: object })

export const getEvolution = () =>
    api.get("/agency/evolution").then((r) => r.data as { generation: number; last_strategy: string; fitness_history: number[] })

export const getNarrative = () =>
    api.get("/agency/narrative").then((r) => r.data as { narrative: string })

export const getDesires = () =>
    api.get("/agency/desires").then((r) => r.data as { count: number; desires: unknown[] })

export const fulfillDesire = (id: string) =>
    api.post(`/agency/desires/${id}/fulfill`).then((r) => r.data)

export const counterfactualRemove = (id: string) =>
    api.get(`/agency/counterfactual/remove/${id}`).then((r) => r.data)

export const counterfactualAdd = (data: { energy?: number; depth?: number; connect_to?: string[] }) =>
    api.post("/agency/counterfactual/add", data).then((r) => r.data)

export const quantumMap = (nodes: { embedding: number[]; energy: number }[]) =>
    api.post("/agency/quantum/map", { nodes }).then((r) => r.data)

export const quantumFidelity = (group_a: object[], group_b: object[]) =>
    api.post("/agency/quantum/fidelity", { group_a, group_b }).then((r) => r.data)

/* ── Cluster ──────────────────────────────────────────────── */
export const getClusterRing = () =>
    api.get("/cluster/ring").then((r) => r.data as { enabled: boolean; ring: { token: number; name: string; addr: string; health: string }[] })

/* ── Metrics ──────────────────────────────────────────────── */
export const getMetrics = () =>
    axios.get("/metrics").then((r) => r.data as string)

