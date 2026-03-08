import { useCallback, useEffect, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import { Keyboard } from "lucide-react"
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
} from "@/components/ui/dialog"
import { cn } from "@/lib/utils"

interface Shortcut {
    keys: string[]
    description: string
}

const SHORTCUTS: Shortcut[] = [
    { keys: ["?"], description: "Show this panel" },
    { keys: ["g", "o"], description: "Go to Overview" },
    { keys: ["g", "g"], description: "Go to Graph Explorer" },
    { keys: ["g", "q"], description: "Go to NQL Console" },
    { keys: ["g", "a"], description: "Go to Algorithms" },
    { keys: ["g", "s"], description: "Go to Settings" },
    { keys: ["/"], description: "Focus search" },
    { keys: ["Escape"], description: "Close panel / dialog" },
    { keys: ["t"], description: "Toggle theme" },
    { keys: ["n"], description: "New chat session" },
]

const NAV_MAP: Record<string, string> = {
    o: "/",
    g: "/graph",
    q: "/nql",
    a: "/algorithms",
    s: "/settings",
}

function Kbd({ children, className }: { children: React.ReactNode; className?: string }) {
    return (
        <kbd
            className={cn(
                "inline-flex h-6 min-w-[24px] items-center justify-center rounded border border-white/20 bg-white/5 px-1.5 font-mono text-[11px] font-medium text-zinc-300 shadow-[0_1px_0_rgba(255,255,255,0.1)]",
                className
            )}
        >
            {children}
        </kbd>
    )
}

export function KeyboardShortcuts() {
    const [open, setOpen] = useState(false)
    const navigate = useNavigate()
    const pendingRef = useRef<string | null>(null)
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

    const clearPending = useCallback(() => {
        pendingRef.current = null
        if (timerRef.current) {
            clearTimeout(timerRef.current)
            timerRef.current = null
        }
    }, [])

    useEffect(() => {
        function isInputFocused() {
            const el = document.activeElement
            if (!el) return false
            const tag = el.tagName.toLowerCase()
            return (
                tag === "input" ||
                tag === "textarea" ||
                tag === "select" ||
                (el as HTMLElement).isContentEditable
            )
        }

        function handleKeydown(e: KeyboardEvent) {
            // Don't capture when typing in inputs (except Escape)
            if (e.key !== "Escape" && isInputFocused()) return
            // Ignore if modifier keys are held (except shift)
            if (e.ctrlKey || e.metaKey || e.altKey) return

            // Handle pending "g then X" combo
            if (pendingRef.current === "g") {
                clearPending()
                const route = NAV_MAP[e.key]
                if (route) {
                    e.preventDefault()
                    navigate(route)
                }
                return
            }

            switch (e.key) {
                case "?":
                    e.preventDefault()
                    setOpen((v) => !v)
                    break
                case "Escape":
                    if (open) {
                        e.preventDefault()
                        setOpen(false)
                    }
                    break
                case "g":
                    e.preventDefault()
                    pendingRef.current = "g"
                    timerRef.current = setTimeout(clearPending, 500)
                    break
                case "/": {
                    e.preventDefault()
                    const searchInput = document.querySelector<HTMLInputElement>(
                        '[data-search-input], input[type="search"], input[placeholder*="Search"]'
                    )
                    searchInput?.focus()
                    break
                }
                case "t":
                    e.preventDefault()
                    document
                        .querySelector<HTMLButtonElement>('[data-theme-toggle]')
                        ?.click()
                    break
                case "n":
                    e.preventDefault()
                    document
                        .querySelector<HTMLButtonElement>('[data-new-chat]')
                        ?.click()
                    break
            }
        }

        window.addEventListener("keydown", handleKeydown)
        return () => {
            window.removeEventListener("keydown", handleKeydown)
            clearPending()
        }
    }, [open, navigate, clearPending])

    return (
        <Dialog open={open} onOpenChange={setOpen}>
            <DialogContent className="border-white/10 bg-zinc-950/95 backdrop-blur-md sm:max-w-md">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2 text-zinc-100">
                        <Keyboard className="h-5 w-5 text-zinc-400" />
                        Keyboard Shortcuts
                    </DialogTitle>
                    <DialogDescription className="text-zinc-500">
                        Navigate the dashboard quickly with these shortcuts.
                    </DialogDescription>
                </DialogHeader>

                <div className="grid gap-1 py-2">
                    {SHORTCUTS.map((shortcut) => (
                        <div
                            key={shortcut.description}
                            className="flex items-center justify-between rounded-md px-3 py-2 transition-colors hover:bg-white/5"
                        >
                            <span className="text-sm text-zinc-300">
                                {shortcut.description}
                            </span>
                            <div className="flex items-center gap-1">
                                {shortcut.keys.map((key, i) => (
                                    <span key={i} className="flex items-center gap-1">
                                        {i > 0 && (
                                            <span className="text-[10px] text-zinc-600">
                                                then
                                            </span>
                                        )}
                                        <Kbd>{key}</Kbd>
                                    </span>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>

                <div className="border-t border-white/10 pt-3 text-center text-[11px] text-zinc-600">
                    Press <Kbd>?</Kbd> to toggle this panel
                </div>
            </DialogContent>
        </Dialog>
    )
}
