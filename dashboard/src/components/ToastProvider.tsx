/**
 * ToastProvider.tsx — Minimal toast notification system for NietzscheDB dashboard
 *
 * Built with React context + portals. No external dependencies required.
 * Renders toasts in the bottom-right corner, stacked above the AI assistant button.
 */

import {
    createContext,
    useContext,
    useCallback,
    useState,
    useEffect,
    useRef,
    type ReactNode,
} from "react"
import { createPortal } from "react-dom"
import { CheckCircle2, XCircle, Info, AlertTriangle, X } from "lucide-react"
import { cn } from "@/lib/utils"

// ── Types ────────────────────────────────────────────────────

export interface Toast {
    id: string
    type: "success" | "error" | "info" | "warning"
    title: string
    description?: string
    duration?: number
}

type ToastInput = Omit<Toast, "id">

interface ToastContextValue {
    toast: (opts: ToastInput) => string
    dismiss: (id: string) => void
}

// ── Constants ────────────────────────────────────────────────

const DEFAULT_DURATION = 4000
const MAX_VISIBLE = 5
const ANIMATION_DURATION = 300

const TOAST_ICONS: Record<Toast["type"], typeof CheckCircle2> = {
    success: CheckCircle2,
    error: XCircle,
    info: Info,
    warning: AlertTriangle,
}

const TOAST_STYLES: Record<Toast["type"], string> = {
    success: "border-l-emerald-500 bg-zinc-900",
    error: "border-l-red-500 bg-zinc-900",
    info: "border-l-blue-500 bg-zinc-900",
    warning: "border-l-amber-500 bg-zinc-900",
}

const ICON_STYLES: Record<Toast["type"], string> = {
    success: "text-emerald-400",
    error: "text-red-400",
    info: "text-blue-400",
    warning: "text-amber-400",
}

// ── Context ──────────────────────────────────────────────────

const ToastContext = createContext<ToastContextValue | null>(null)

// ── Hook ─────────────────────────────────────────────────────

export function useToast(): ToastContextValue {
    const ctx = useContext(ToastContext)
    if (!ctx) {
        throw new Error("useToast must be used within a ToastProvider")
    }
    return ctx
}

// ── Toast Item Component ─────────────────────────────────────

interface ToastItemProps {
    toast: Toast
    onDismiss: (id: string) => void
}

function ToastItem({ toast, onDismiss }: ToastItemProps) {
    const [state, setState] = useState<"entering" | "visible" | "exiting">("entering")
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

    useEffect(() => {
        // Trigger enter animation
        const enterTimer = setTimeout(() => setState("visible"), 10)

        // Schedule auto-dismiss
        const duration = toast.duration ?? DEFAULT_DURATION
        if (duration > 0) {
            timerRef.current = setTimeout(() => {
                setState("exiting")
                setTimeout(() => onDismiss(toast.id), ANIMATION_DURATION)
            }, duration)
        }

        return () => {
            clearTimeout(enterTimer)
            if (timerRef.current) clearTimeout(timerRef.current)
        }
    }, [toast.id, toast.duration, onDismiss])

    const handleClose = useCallback(() => {
        if (timerRef.current) clearTimeout(timerRef.current)
        setState("exiting")
        setTimeout(() => onDismiss(toast.id), ANIMATION_DURATION)
    }, [toast.id, onDismiss])

    const Icon = TOAST_ICONS[toast.type]

    return (
        <div
            role="alert"
            aria-live="assertive"
            className={cn(
                "pointer-events-auto flex w-[360px] items-start gap-3 rounded-lg border border-zinc-700 border-l-4 px-4 py-3 shadow-xl transition-all",
                TOAST_STYLES[toast.type],
                state === "entering" && "translate-x-full opacity-0",
                state === "visible" && "translate-x-0 opacity-100",
                state === "exiting" && "translate-x-full opacity-0"
            )}
            style={{ transitionDuration: `${ANIMATION_DURATION}ms` }}
        >
            <Icon
                className={cn("mt-0.5 h-5 w-5 shrink-0", ICON_STYLES[toast.type])}
            />
            <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-zinc-100">{toast.title}</p>
                {toast.description && (
                    <p className="mt-0.5 text-xs text-zinc-400 leading-relaxed">
                        {toast.description}
                    </p>
                )}
            </div>
            <button
                onClick={handleClose}
                className="shrink-0 rounded-md p-0.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300"
                aria-label="Dismiss notification"
            >
                <X className="h-4 w-4" />
            </button>
        </div>
    )
}

// ── Toast Container ──────────────────────────────────────────

interface ToastContainerProps {
    toasts: Toast[]
    onDismiss: (id: string) => void
}

function ToastContainer({ toasts, onDismiss }: ToastContainerProps) {
    // Only show the most recent MAX_VISIBLE toasts
    const visible = toasts.slice(-MAX_VISIBLE)

    return createPortal(
        <div
            className="fixed bottom-24 right-6 z-[60] flex flex-col-reverse gap-2 pointer-events-none"
            aria-label="Notifications"
        >
            {visible.map((t) => (
                <ToastItem key={t.id} toast={t} onDismiss={onDismiss} />
            ))}
        </div>,
        document.body
    )
}

// ── Provider ─────────────────────────────────────────────────

interface ToastProviderProps {
    children: ReactNode
}

export function ToastProvider({ children }: ToastProviderProps) {
    const [toasts, setToasts] = useState<Toast[]>([])
    const counterRef = useRef(0)

    const toast = useCallback((opts: ToastInput): string => {
        const id = `toast-${Date.now()}-${++counterRef.current}`
        const newToast: Toast = { ...opts, id }
        setToasts((prev) => [...prev, newToast])
        return id
    }, [])

    const dismiss = useCallback((id: string) => {
        setToasts((prev) => prev.filter((t) => t.id !== id))
    }, [])

    const contextValue: ToastContextValue = { toast, dismiss }

    return (
        <ToastContext.Provider value={contextValue}>
            {children}
            <ToastContainer toasts={toasts} onDismiss={dismiss} />
        </ToastContext.Provider>
    )
}

export default ToastProvider
