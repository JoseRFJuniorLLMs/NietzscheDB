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

