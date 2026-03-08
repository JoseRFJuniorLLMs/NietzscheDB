/**
 * GlobalAIAssistant.tsx — Floating AI chat assistant for NietzscheDB dashboard
 *
 * Provides a persistent chat interface powered by the NQL AI assistant service.
 * Features session management, NQL code block detection, and localStorage persistence.
 */

import { useState, useRef, useEffect, useCallback, useMemo } from "react"
import { create } from "zustand"
import {
    Brain,
    X,
    Send,
    Plus,
    Trash2,
    Search,
    Copy,
    Check,
    ChevronLeft,
    ChevronRight,
    MessageSquare,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import {
    generateNqlFromPrompt,
    isAiConfigured,
    type Message,
} from "@/services/nqlAssistant"

// ── Types ────────────────────────────────────────────────────

export interface ChatMessage {
    role: "user" | "assistant"
    content: string
    timestamp: number
}

export interface ChatSession {
    id: string
    title: string
    messages: ChatMessage[]
    createdAt: number
    updatedAt: number
}

export interface ChatStore {
    sessions: ChatSession[]
    activeSessionId: string | null
    isOpen: boolean
    toggle: () => void
    close: () => void
    newSession: () => void
    setActiveSession: (id: string) => void
    addMessage: (role: "user" | "assistant", content: string) => void
    deleteSession: (id: string) => void
    clearAll: () => void
}

// ── Constants ────────────────────────────────────────────────

const STORAGE_KEY = "nietzsche_chat_sessions"
const MAX_TITLE_LENGTH = 40

function generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

function loadSessions(): ChatSession[] {
    try {
        const raw = localStorage.getItem(STORAGE_KEY)
        if (!raw) return []
        return JSON.parse(raw) as ChatSession[]
    } catch {
        return []
    }
}

function persistSessions(sessions: ChatSession[]) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions))
    } catch {
        // Storage full — silently ignore
    }
}

// ── Zustand Store ────────────────────────────────────────────

export const useChatStore = create<ChatStore>((set, get) => ({
    sessions: loadSessions(),
    activeSessionId: null,
    isOpen: false,

    toggle: () => set((s) => ({ isOpen: !s.isOpen })),
    close: () => set({ isOpen: false }),

    newSession: () => {
        const session: ChatSession = {
            id: generateId(),
            title: "New Chat",
            messages: [],
            createdAt: Date.now(),
            updatedAt: Date.now(),
        }
        set((s) => ({
            sessions: [session, ...s.sessions],
            activeSessionId: session.id,
        }))
    },

    setActiveSession: (id: string) => set({ activeSessionId: id }),

    addMessage: (role, content) => {
        const state = get()
        let { activeSessionId, sessions } = state

        // Auto-create session if none active
        if (!activeSessionId) {
            const session: ChatSession = {
                id: generateId(),
                title: "New Chat",
                messages: [],
                createdAt: Date.now(),
                updatedAt: Date.now(),
            }
            sessions = [session, ...sessions]
            activeSessionId = session.id
        }

        const updated = sessions.map((s) => {
            if (s.id !== activeSessionId) return s
            const messages = [...s.messages, { role, content, timestamp: Date.now() }]
            // Set title from first user message
            const title =
                s.title === "New Chat" && role === "user"
                    ? content.slice(0, MAX_TITLE_LENGTH) + (content.length > MAX_TITLE_LENGTH ? "..." : "")
                    : s.title
            return { ...s, messages, title, updatedAt: Date.now() }
        })

        set({ sessions: updated, activeSessionId })
    },

    deleteSession: (id: string) =>
        set((s) => ({
            sessions: s.sessions.filter((sess) => sess.id !== id),
            activeSessionId: s.activeSessionId === id ? null : s.activeSessionId,
        })),

    clearAll: () => set({ sessions: [], activeSessionId: null }),
}))

// Persist on every change
useChatStore.subscribe((state) => {
    persistSessions(state.sessions)
})

// ── Helpers ──────────────────────────────────────────────────

function extractNqlBlocks(text: string): string[] {
    const matches = text.matchAll(/```nql\s*\n([\s\S]*?)```/g)
    return Array.from(matches, (m) => m[1].trim())
}

function formatDateGroup(ts: number): string {
    const now = new Date()
    const date = new Date(ts)
    const diffDays = Math.floor((now.getTime() - date.getTime()) / 86400000)
    if (diffDays === 0) return "Today"
    if (diffDays === 1) return "Yesterday"
    return "Previous"
}

function renderMessageContent(content: string): React.ReactNode {
    // Split by ```nql blocks and render them with special styling
    const parts = content.split(/(```nql\s*\n[\s\S]*?```)/g)
    return parts.map((part, i) => {
        const nqlMatch = part.match(/```nql\s*\n([\s\S]*?)```/)
        if (nqlMatch) {
            return (
                <pre
                    key={i}
                    className="my-2 rounded-md bg-zinc-900 border border-zinc-700 p-3 text-xs font-mono text-emerald-400 overflow-x-auto"
                >
                    <code>{nqlMatch[1].trim()}</code>
                </pre>
            )
        }
        // Regular text — preserve line breaks
        if (!part.trim()) return null
        return (
            <span key={i} className="whitespace-pre-wrap">
                {part}
            </span>
        )
    })
}

// ── Component ────────────────────────────────────────────────

export function GlobalAIAssistant() {
    const {
        sessions,
        activeSessionId,
        isOpen,
        toggle,
        close,
        newSession,
        setActiveSession,
        addMessage,
        deleteSession,
    } = useChatStore()

    const [input, setInput] = useState("")
    const [isLoading, setIsLoading] = useState(false)
    const [showSidebar, setShowSidebar] = useState(false)
    const [searchQuery, setSearchQuery] = useState("")
    const [copiedBlock, setCopiedBlock] = useState<string | null>(null)

    const messagesEndRef = useRef<HTMLDivElement>(null)
    const inputRef = useRef<HTMLInputElement>(null)

    const aiConfigured = useMemo(() => isAiConfigured(), [])
    const activeSession = sessions.find((s) => s.id === activeSessionId) ?? null

    // Auto-scroll to bottom on new messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [activeSession?.messages.length])

    // Focus input when panel opens
    useEffect(() => {
        if (isOpen) {
            setTimeout(() => inputRef.current?.focus(), 100)
        }
    }, [isOpen])

    const handleSend = useCallback(async () => {
        const trimmed = input.trim()
        if (!trimmed || isLoading) return

        setInput("")
        addMessage("user", trimmed)

        setIsLoading(true)
        try {
            // Build history from current session for context
            const currentSession = useChatStore.getState().sessions.find(
                (s) => s.id === useChatStore.getState().activeSessionId
            )
            const history: Message[] =
                currentSession?.messages.slice(-10).map((m) => ({
                    role: m.role,
                    content: m.content,
                })) ?? []

            const response = await generateNqlFromPrompt(trimmed, history)
            addMessage("assistant", response)
        } catch (err) {
            const msg = err instanceof Error ? err.message : "Unknown error"
            addMessage("assistant", `Error: ${msg}`)
        } finally {
            setIsLoading(false)
        }
    }, [input, isLoading, addMessage])

    const handleKeyDown = useCallback(
        (e: React.KeyboardEvent) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault()
                handleSend()
            }
        },
        [handleSend]
    )

    const handleCopyNql = useCallback((nql: string) => {
        navigator.clipboard.writeText(nql).then(() => {
            setCopiedBlock(nql)
            setTimeout(() => setCopiedBlock(null), 2000)
        })
    }, [])

    // Group sessions by date
    const groupedSessions = useMemo(() => {
        const filtered = searchQuery
            ? sessions.filter(
                  (s) =>
                      s.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                      s.messages.some((m) =>
                          m.content.toLowerCase().includes(searchQuery.toLowerCase())
                      )
              )
            : sessions

        const groups: Record<string, ChatSession[]> = {}
        for (const session of filtered) {
            const group = formatDateGroup(session.updatedAt)
            if (!groups[group]) groups[group] = []
            groups[group].push(session)
        }
        return groups
    }, [sessions, searchQuery])

    return (
        <>
            {/* Floating Button */}
            <button
                onClick={toggle}
                className={cn(
                    "fixed bottom-6 right-6 z-50 flex h-14 w-14 items-center justify-center rounded-full shadow-lg transition-all duration-300 hover:scale-110 focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-zinc-900",
                    "bg-[#7c3aed] text-white",
                    aiConfigured && "animate-pulse"
                )}
                aria-label="Toggle AI Assistant"
            >
                <Brain className="h-6 w-6" />
            </button>

            {/* Chat Panel */}
            {isOpen && (
                <div
                    className={cn(
                        "fixed bottom-24 right-6 z-50 flex overflow-hidden rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl",
                        "transition-all duration-300 ease-out",
                        showSidebar ? "w-[600px]" : "w-[400px]"
                    )}
                    style={{ height: 500 }}
                >
                    {/* Session Sidebar */}
                    {showSidebar && (
                        <div className="flex w-[200px] flex-col border-r border-zinc-700 bg-zinc-950">
                            <div className="flex items-center justify-between border-b border-zinc-800 p-2">
                                <span className="text-xs font-medium text-zinc-400">
                                    History
                                </span>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-6 w-6 text-zinc-400 hover:text-white"
                                    onClick={() => setShowSidebar(false)}
                                >
                                    <ChevronLeft className="h-3.5 w-3.5" />
                                </Button>
                            </div>
                            <div className="px-2 py-1.5">
                                <Input
                                    placeholder="Search..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="h-7 border-zinc-700 bg-zinc-900 text-xs text-zinc-300 placeholder:text-zinc-600"
                                />
                            </div>
                            <ScrollArea className="flex-1">
                                <div className="p-1.5 space-y-2">
                                    {Object.entries(groupedSessions).map(
                                        ([group, items]) => (
                                            <div key={group}>
                                                <span className="px-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-600">
                                                    {group}
                                                </span>
                                                <div className="mt-0.5 space-y-0.5">
                                                    {items.map((session) => (
                                                        <button
                                                            key={session.id}
                                                            onClick={() =>
                                                                setActiveSession(
                                                                    session.id
                                                                )
                                                            }
                                                            className={cn(
                                                                "group flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-left text-xs transition-colors",
                                                                session.id ===
                                                                    activeSessionId
                                                                    ? "bg-purple-900/40 text-purple-300"
                                                                    : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                                                            )}
                                                        >
                                                            <MessageSquare className="h-3 w-3 shrink-0" />
                                                            <span className="flex-1 truncate">
                                                                {session.title}
                                                            </span>
                                                            <button
                                                                onClick={(e) => {
                                                                    e.stopPropagation()
                                                                    deleteSession(
                                                                        session.id
                                                                    )
                                                                }}
                                                                className="hidden shrink-0 text-zinc-600 hover:text-red-400 group-hover:block"
                                                            >
                                                                <Trash2 className="h-3 w-3" />
                                                            </button>
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>
                                        )
                                    )}
                                    {Object.keys(groupedSessions).length === 0 && (
                                        <p className="px-2 py-4 text-center text-[10px] text-zinc-600">
                                            {searchQuery
                                                ? "No results"
                                                : "No sessions yet"}
                                        </p>
                                    )}
                                </div>
                            </ScrollArea>
                        </div>
                    )}

                    {/* Main Chat Area */}
                    <div className="flex flex-1 flex-col">
                        {/* Header */}
                        <div className="flex items-center justify-between border-b border-zinc-700 px-3 py-2">
                            <div className="flex items-center gap-2">
                                {!showSidebar && (
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-7 w-7 text-zinc-400 hover:text-white"
                                        onClick={() => setShowSidebar(true)}
                                    >
                                        <ChevronRight className="h-4 w-4" />
                                    </Button>
                                )}
                                <Brain className="h-4 w-4 text-purple-400" />
                                <span className="text-sm font-medium text-zinc-200">
                                    NQL Assistant
                                </span>
                                {aiConfigured && (
                                    <Badge
                                        variant="outline"
                                        className="border-emerald-700 text-emerald-400 text-[10px] px-1.5 py-0"
                                    >
                                        AI Ready
                                    </Badge>
                                )}
                            </div>
                            <div className="flex items-center gap-1">
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7 text-zinc-400 hover:text-white"
                                    onClick={newSession}
                                    title="New chat"
                                >
                                    <Plus className="h-4 w-4" />
                                </Button>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7 text-zinc-400 hover:text-white"
                                    onClick={() => setShowSidebar((p) => !p)}
                                    title="Search sessions"
                                >
                                    <Search className="h-4 w-4" />
                                </Button>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7 text-zinc-400 hover:text-white"
                                    onClick={close}
                                    title="Close"
                                >
                                    <X className="h-4 w-4" />
                                </Button>
                            </div>
                        </div>

                        {/* Messages */}
                        <ScrollArea className="flex-1 px-3 py-2">
                            <div className="space-y-3">
                                {!activeSession || activeSession.messages.length === 0 ? (
                                    <div className="flex flex-col items-center justify-center py-12 text-center">
                                        <Brain className="mb-3 h-10 w-10 text-purple-500/40" />
                                        <p className="text-sm text-zinc-400">
                                            Ask me anything about NQL
                                        </p>
                                        <p className="mt-1 text-xs text-zinc-600">
                                            I can generate queries, explain syntax,
                                            and help explore your graph.
                                        </p>
                                        {!aiConfigured && (
                                            <p className="mt-3 rounded-md border border-amber-800/50 bg-amber-950/30 px-3 py-2 text-xs text-amber-400">
                                                Set VITE_GOOGLE_API_KEY in .env to
                                                enable AI responses.
                                            </p>
                                        )}
                                    </div>
                                ) : (
                                    activeSession.messages.map((msg, idx) => {
                                        const nqlBlocks = extractNqlBlocks(msg.content)
                                        return (
                                            <div
                                                key={idx}
                                                className={cn(
                                                    "flex",
                                                    msg.role === "user"
                                                        ? "justify-end"
                                                        : "justify-start"
                                                )}
                                            >
                                                <div
                                                    className={cn(
                                                        "max-w-[85%] rounded-lg px-3 py-2 text-sm",
                                                        msg.role === "user"
                                                            ? "bg-purple-700/60 text-purple-50"
                                                            : "bg-zinc-800 text-zinc-200"
                                                    )}
                                                >
                                                    {renderMessageContent(msg.content)}
                                                    {msg.role === "assistant" &&
                                                        nqlBlocks.length > 0 && (
                                                            <div className="mt-2 flex flex-wrap gap-1">
                                                                {nqlBlocks.map(
                                                                    (nql, bi) => (
                                                                        <button
                                                                            key={bi}
                                                                            onClick={() =>
                                                                                handleCopyNql(
                                                                                    nql
                                                                                )
                                                                            }
                                                                            className="inline-flex items-center gap-1 rounded border border-zinc-600 bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-400 transition-colors hover:border-purple-500 hover:text-purple-300"
                                                                        >
                                                                            {copiedBlock ===
                                                                            nql ? (
                                                                                <>
                                                                                    <Check className="h-2.5 w-2.5" />
                                                                                    Copied
                                                                                </>
                                                                            ) : (
                                                                                <>
                                                                                    <Copy className="h-2.5 w-2.5" />
                                                                                    Copy
                                                                                    NQL
                                                                                </>
                                                                            )}
                                                                        </button>
                                                                    )
                                                                )}
                                                            </div>
                                                        )}
                                                </div>
                                            </div>
                                        )
                                    })
                                )}

                                {isLoading && (
                                    <div className="flex justify-start">
                                        <div className="flex items-center gap-1.5 rounded-lg bg-zinc-800 px-3 py-2 text-sm text-zinc-400">
                                            <span className="inline-block h-1.5 w-1.5 animate-bounce rounded-full bg-purple-400 [animation-delay:0ms]" />
                                            <span className="inline-block h-1.5 w-1.5 animate-bounce rounded-full bg-purple-400 [animation-delay:150ms]" />
                                            <span className="inline-block h-1.5 w-1.5 animate-bounce rounded-full bg-purple-400 [animation-delay:300ms]" />
                                        </div>
                                    </div>
                                )}

                                <div ref={messagesEndRef} />
                            </div>
                        </ScrollArea>

                        {/* Input Area */}
                        <div className="border-t border-zinc-700 p-2">
                            <div className="flex items-center gap-2">
                                <Input
                                    ref={inputRef}
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={handleKeyDown}
                                    placeholder="Ask about NQL..."
                                    disabled={isLoading}
                                    className="flex-1 border-zinc-700 bg-zinc-800 text-sm text-zinc-200 placeholder:text-zinc-500"
                                />
                                <Button
                                    size="icon"
                                    onClick={handleSend}
                                    disabled={!input.trim() || isLoading}
                                    className="h-10 w-10 shrink-0 bg-purple-600 text-white hover:bg-purple-500 disabled:bg-zinc-700 disabled:text-zinc-500"
                                >
                                    <Send className="h-4 w-4" />
                                </Button>
                            </div>
                            <p className="mt-1 text-[10px] text-zinc-600">
                                Ctrl+Enter to send
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </>
    )
}

export default GlobalAIAssistant
