/**
 * NQL AI Assistant — Gemini-powered NQL generator for NietzscheDB
 * Uses Google Gemini API (gemini-2.5-flash) as primary provider,
 * with fallback to OpenAI/Anthropic/Ollama.
 */

export interface Message {
    role: "user" | "assistant" | "system" | "model"
    content: string
}

type Provider = "gemini" | "openai" | "anthropic" | "ollama"

interface AiConfig {
    provider: Provider
    googleApiKey: string
    geminiModel: string
    fallbackApiKey: string
    fallbackBaseUrl: string
    fallbackModel: string
    fallbackProvider: Provider
}

const AI_CONFIG: AiConfig = {
    // Gemini (primary) — uses VITE_GOOGLE_API_KEY
    provider: (import.meta.env.VITE_AI_PROVIDER as Provider) || "gemini",
    googleApiKey: import.meta.env.VITE_GOOGLE_API_KEY || "",
    geminiModel: import.meta.env.VITE_GEMINI_MODEL || "gemini-2.5-flash",

    // Fallback (OpenAI/Anthropic/Ollama)
    fallbackApiKey: import.meta.env.VITE_AI_API_KEY || "",
    fallbackBaseUrl: import.meta.env.VITE_AI_BASE_URL || "https://api.openai.com/v1",
    fallbackModel: import.meta.env.VITE_AI_MODEL || "gpt-4o-mini",
    fallbackProvider: (import.meta.env.VITE_AI_FALLBACK_PROVIDER as Provider) || "openai",
}

export const isAiConfigured = () =>
    !!(AI_CONFIG.googleApiKey || AI_CONFIG.fallbackApiKey)

const NQL_SYSTEM_PROMPT = `You are an expert NQL (Nietzsche Query Language) assistant for NietzscheDB — a Temporal Hyperbolic Graph Database.

Your job is to convert natural language requests into valid NQL queries. Always return the NQL query inside a \`\`\`nql code block.

## NQL Reference

### Core Statements
- MATCH (n) WHERE n.energy > 0.5 RETURN n — Pattern matching with filters
- MATCH (a)-[:Association]->(b) RETURN a, b — Path patterns
- MATCH (m:Memory) WHERE m.depth < 0.3 ORDER BY m.energy DESC LIMIT 10 RETURN m
- CREATE (n:Concept {energy: 0.8, content: {label: "idea"}}) RETURN n
- MERGE (n:Memory {content: {label: "test"}}) ON CREATE SET n.energy = 0.5 ON MATCH SET n.energy = n.energy + 0.1
- MATCH (n) WHERE n.id = "uuid" SET n.energy = 0.9 RETURN n
- MATCH (n) WHERE n.id = "uuid" DELETE n
- DREAM FROM $embedding DEPTH 3 NOISE 0.1 — Speculative exploration
- DIFFUSE FROM $embedding WITH t = [0.1, 1.0, 10.0] MAX_HOPS 5
- NARRATE IN "collection" WINDOW 24 FORMAT json
- RECONSTRUCT $id MODALITY text QUALITY high
- EXPLAIN MATCH (n) RETURN n — Show execution plan
- COUNTERFACTUAL SET n.energy = 0 MATCH (n)-[]->(m) RETURN AVG(m.energy)

### Node Types
Semantic, Episodic, Concept, DreamSnapshot, Somatic, Linguistic, Composite

### Edge Types
Association, Hierarchical, LSystemGenerated, Pruned

### Properties
n.id, n.energy (0.0-1.0), n.depth, n.hausdorff_local, n.node_type, n.created_at, n.lsystem_generation

### 13 Mathematical Functions
- POINCARE_DIST(a.embedding, b.embedding) — Geodesic distance in Poincaré ball
- KLEIN_DIST(a.embedding, b.embedding) — Beltrami-Klein disk distance
- MINKOWSKI_NORM(n.embedding) — Conformal factor λ
- LOBACHEVSKY_ANGLE(n.embedding) — Angle of parallelism
- RIEMANN_CURVATURE(n) — Ollivier-Ricci curvature
- GAUSS_KERNEL(a.embedding, b.embedding, t) — Heat kernel exp(-d²/4t)
- CHEBYSHEV_COEFF(n, k) — Polynomial T_k at node position
- RAMANUJAN_EXPANSION(n) — Local spectral expansion
- HAUSDORFF_DIM(n) — Local fractal dimension
- EULER_CHAR(n) — Local Euler characteristic (V-E)
- LAPLACIAN_SCORE(n) — Graph Laplacian diagonal
- FOURIER_COEFF(n, k) — Graph Fourier coefficient
- DIRICHLET_ENERGY(n) — Local smoothness energy

### Clauses & Operators
WHERE, AND, OR, NOT, ORDER BY ASC|DESC, LIMIT, SKIP, RETURN, DISTINCT, AS
IN (...), BETWEEN ... AND ..., CONTAINS, STARTS_WITH, ENDS_WITH
COUNT(*), SUM(), AVG(), MIN(), MAX(), GROUP BY

### Time Functions
NOW(), EPOCH_MS(), INTERVAL("7d"), INTERVAL("1.5h"), INTERVAL("30m")

### Daemon Creation
CREATE DAEMON entropy_watcher ON (n:Memory) WHEN n.energy < 0.1 THEN DELETE n EVERY INTERVAL("1h")

### Transaction Control
BEGIN, COMMIT, ROLLBACK

## Rules
1. Always wrap NQL in \`\`\`nql code blocks
2. Use correct node/edge type names
3. Use $param for embedding vectors the user would need to provide
4. Explain what the query does in 1-2 sentences before the code block
5. If the request is ambiguous, ask for clarification
6. Respond in the same language the user writes in`

/* ── Gemini REST API (primary) ────────────────────────────── */
async function callGemini(messages: Message[]): Promise<string> {
    const systemMsg = messages.find((m) => m.role === "system")?.content ?? ""
    const chatMessages = messages.filter((m) => m.role !== "system")

    // Convert to Gemini format: user/model roles, parts array
    const contents = chatMessages.map((m) => ({
        role: m.role === "assistant" ? "model" : "user",
        parts: [{ text: m.content }],
    }))

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${AI_CONFIG.geminiModel}:generateContent?key=${AI_CONFIG.googleApiKey}`

    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            system_instruction: {
                parts: [{ text: systemMsg }],
            },
            contents,
            generationConfig: {
                temperature: 0.2,
                maxOutputTokens: 2048,
            },
        }),
    })

    if (!res.ok) {
        const errBody = await res.text().catch(() => "")
        throw new Error(`Gemini API error ${res.status}: ${errBody.slice(0, 200)}`)
    }

    const data = await res.json()
    return data.candidates?.[0]?.content?.parts?.[0]?.text ?? ""
}

/* ── Fallback providers ───────────────────────────────────── */
async function callOpenAI(messages: Message[]): Promise<string> {
    const res = await fetch(`${AI_CONFIG.fallbackBaseUrl}/chat/completions`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${AI_CONFIG.fallbackApiKey}`,
        },
        body: JSON.stringify({
            model: AI_CONFIG.fallbackModel,
            messages,
            temperature: 0.3,
            max_tokens: 2048,
        }),
    })
    if (!res.ok) throw new Error(`OpenAI API error: ${res.status}`)
    const data = await res.json()
    return data.choices?.[0]?.message?.content ?? ""
}

async function callAnthropic(messages: Message[]): Promise<string> {
    const systemMsg = messages.find((m) => m.role === "system")?.content ?? ""
    const nonSystem = messages.filter((m) => m.role !== "system")
    const res = await fetch(`${AI_CONFIG.fallbackBaseUrl}/messages`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "x-api-key": AI_CONFIG.fallbackApiKey,
            "anthropic-version": "2023-06-01",
            "anthropic-dangerous-direct-browser-access": "true",
        },
        body: JSON.stringify({
            model: AI_CONFIG.fallbackModel,
            system: systemMsg,
            messages: nonSystem.map((m) => ({ role: m.role, content: m.content })),
            max_tokens: 2048,
            temperature: 0.3,
        }),
    })
    if (!res.ok) throw new Error(`Anthropic API error: ${res.status}`)
    const data = await res.json()
    return data.content?.[0]?.text ?? ""
}

async function callOllama(messages: Message[]): Promise<string> {
    const res = await fetch(`${AI_CONFIG.fallbackBaseUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            model: AI_CONFIG.fallbackModel,
            messages,
            stream: false,
        }),
    })
    if (!res.ok) throw new Error(`Ollama API error: ${res.status}`)
    const data = await res.json()
    return data.message?.content ?? ""
}

/* ── Provider dispatch ────────────────────────────────────── */
async function callLLM(messages: Message[]): Promise<string> {
    // Prefer Gemini if Google API key is available
    if (AI_CONFIG.googleApiKey && (AI_CONFIG.provider === "gemini" || !AI_CONFIG.fallbackApiKey)) {
        return callGemini(messages)
    }

    // Fallback providers
    const provider = AI_CONFIG.fallbackApiKey ? AI_CONFIG.fallbackProvider : AI_CONFIG.provider
    switch (provider) {
        case "anthropic":
            return callAnthropic(messages)
        case "ollama":
            return callOllama(messages)
        case "openai":
            return callOpenAI(messages)
        default:
            // If explicit provider set and it's gemini, use it
            if (AI_CONFIG.googleApiKey) return callGemini(messages)
            throw new Error("No AI provider configured")
    }
}

/* ── Public API ───────────────────────────────────────────── */
export async function generateNqlFromPrompt(
    userMessage: string,
    history: Message[] = [],
): Promise<string> {
    if (!isAiConfigured()) {
        return "AI not configured. Set VITE_GOOGLE_API_KEY in your .env for Gemini.\n\nAlternatively: VITE_AI_PROVIDER + VITE_AI_API_KEY for OpenAI/Anthropic/Ollama.\n\nYou can still write NQL manually using the reference panel."
    }

    const messages: Message[] = [
        { role: "system", content: NQL_SYSTEM_PROMPT },
        ...history.slice(-10),
        { role: "user", content: userMessage },
    ]

    return callLLM(messages)
}

export function extractNqlFromResponse(response: string): string | null {
    const match = response.match(/```nql\s*\n([\s\S]*?)```/)
    return match ? match[1].trim() : null
}
