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

/** Default collection used across the dashboard when none is selected. */
export const DEFAULT_COLLECTION = "eva_core"

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
export const executeNql = (nql: string, collection?: string) =>
    api.post("/query", { nql, collection: collection || DEFAULT_COLLECTION }).then((r) => r.data as { nodes: unknown[]; error: string | null })

export const fullTextSearch = (q: string, limit = 10, collection?: string) =>
    api.get("/search", { params: { q, limit, collection: collection || DEFAULT_COLLECTION } }).then((r) => r.data as { query: string; results: { node_id: string; score: number }[] })

/* ── Node CRUD ────────────────────────────────────────────── */
export const getNode = (id: string, collection?: string) =>
    api.get(`/node/${id}`, { params: { collection: collection || DEFAULT_COLLECTION } }).then((r) => r.data)

export const insertNode = (data: { id?: string; node_type?: string; energy?: number; content?: object; embedding?: number[]; collection?: string }) =>
    api.post("/node", { ...data, collection: data.collection || DEFAULT_COLLECTION }).then((r) => r.data as { id: string })

export const deleteNode = (id: string, collection?: string) =>
    api.delete(`/node/${id}`, { params: { collection: collection || DEFAULT_COLLECTION } }).then((r) => r.data as { deleted: string })

/* ── Edge CRUD ────────────────────────────────────────────── */
export const insertEdge = (data: { from: string; to: string; edge_type?: string; weight?: number; collection?: string }) =>
    api.post("/edge", { ...data, collection: data.collection || DEFAULT_COLLECTION }).then((r) => r.data as { id: string })

export const deleteEdge = (id: string, collection?: string) =>
    api.delete(`/edge/${id}`, { params: { collection: collection || DEFAULT_COLLECTION } }).then((r) => r.data as { deleted: string })

/* ── Batch ────────────────────────────────────────────────── */
export const batchInsertNodes = (nodes: object[], collection?: string) =>
    api.post("/batch/nodes", { nodes, collection: collection || DEFAULT_COLLECTION }).then((r) => r.data as { inserted: number; node_ids: string[] })

export const batchInsertEdges = (edges: object[], collection?: string) =>
    api.post("/batch/edges", { edges, collection: collection || DEFAULT_COLLECTION }).then((r) => r.data as { inserted: number; edge_ids: string[] })

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
    api.get(`/algo/${name}`, { params: { ...params, collection: params.collection || DEFAULT_COLLECTION } }).then((r) => r.data)

/* ── Sleep ────────────────────────────────────────────────── */
export const triggerSleep = (params: { noise?: number; adam_steps?: number; hausdorff_threshold?: number; collection?: string } = {}) =>
    api.post("/sleep", { ...params, collection: params.collection || DEFAULT_COLLECTION }).then((r) => r.data as {
        hausdorff_before: number; hausdorff_after: number; hausdorff_delta: number
        committed: boolean; nodes_perturbed: number
    })

/* ── Backup & Export ──────────────────────────────────────── */
export const createBackup = (label?: string, collection?: string) =>
    api.post("/backup", { label: label || "manual", collection: collection || DEFAULT_COLLECTION }).then((r) => r.data as { label: string; path: string; created_at: string; size_bytes: number })

export const listBackups = () =>
    api.get("/backup").then((r) => r.data as { backups: { label: string; path: string; created_at: string; size_bytes: number }[] })

export const exportData = (type: "nodes" | "edges", format: "jsonl" | "csv" = "jsonl", collection?: string) =>
    api.get(`/export/${type}`, { params: { format, collection: collection || DEFAULT_COLLECTION }, responseType: "blob" }).then((r) => r.data as Blob)

/* ── Agency ───────────────────────────────────────────────── */
const agencyParams = (collection?: string) => ({ params: { collection: collection || DEFAULT_COLLECTION } })

export const getAgencyHealth = (collection?: string) =>
    api.get("/agency/health", agencyParams(collection)).then((r) => r.data)

export const getAgencyHealthLatest = (collection?: string) =>
    api.get("/agency/health/latest", agencyParams(collection)).then((r) => r.data)

export const getObserver = (collection?: string) =>
    api.get("/agency/observer", agencyParams(collection)).then((r) => r.data as { observer_id: string; energy: number; depth: number; hausdorff_local: number; is_observer: boolean; content: object })

export const getEvolution = (collection?: string) =>
    api.get("/agency/evolution", agencyParams(collection)).then((r) => r.data as { generation: number; last_strategy: string; fitness_history: number[] })

export const getNarrative = (collection?: string) =>
    api.get("/agency/narrative", agencyParams(collection)).then((r) => r.data as { narrative: string })

export const getDesires = (collection?: string) =>
    api.get("/agency/desires", agencyParams(collection)).then((r) => r.data as { count: number; desires: unknown[] })

export const fulfillDesire = (id: string, collection?: string) =>
    api.post(`/agency/desires/${id}/fulfill`, null, agencyParams(collection)).then((r) => r.data)

export const counterfactualRemove = (id: string, collection?: string) =>
    api.get(`/agency/counterfactual/remove/${id}`, agencyParams(collection)).then((r) => r.data)

export const counterfactualAdd = (data: { energy?: number; depth?: number; connect_to?: string[]; collection?: string }) =>
    api.post("/agency/counterfactual/add", { ...data, collection: data.collection || DEFAULT_COLLECTION }).then((r) => r.data)

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
