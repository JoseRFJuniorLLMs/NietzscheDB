// OnboardingTour.tsx — First-use onboarding tour for NietzscheDB Dashboard
// Shows step-by-step overlay with spotlight on target nav elements

import { useCallback, useEffect, useState } from "react"
import {
    Activity,
    ChevronRight,
    GitBranch,
    Network,
    Sparkles,
    Terminal,
    X,
} from "lucide-react"

import { cn } from "@/lib/utils"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

const STORAGE_KEY = "nietzsche_onboarding_complete"

interface TourStep {
    title: string
    description: string
    icon: React.ElementType
    /** CSS selector for the target element to highlight. null = center overlay */
    target: string | null
}

const STEPS: TourStep[] = [
    {
        title: "Welcome to NietzscheDB Dashboard",
        description:
            "Your hyperbolic graph database control center. Let us show you around the key features.",
        icon: Sparkles,
        target: null,
    },
    {
        title: "Explore your graph",
        description:
            "Visualize nodes and edges in the hyperbolic space. Drag, zoom, and interact with your graph data in real time.",
        icon: Network,
        target: 'a[href="/graph"]',
    },
    {
        title: "Query with AI",
        description:
            "Write NQL queries with AI assistance. The console supports syntax highlighting, auto-complete, and result previews.",
        icon: Terminal,
        target: 'a[href="/query"]',
    },
    {
        title: "Run algorithms",
        description:
            "Execute graph algorithms like PageRank, Louvain community detection, and betweenness centrality on your data.",
        icon: GitBranch,
        target: 'a[href="/algorithms"]',
    },
    {
        title: "Monitor health",
        description:
            "Track collection sizes, memory usage, gRPC latency, and daemon activity from a single dashboard.",
        icon: Activity,
        target: 'a[href="/monitoring"]',
    },
    {
        title: "You're ready!",
        description:
            "That covers the essentials. Dive in and explore the rest of the features on your own. Happy graphing!",
        icon: Sparkles,
        target: null,
    },
]

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function OnboardingTour() {
    const [active, setActive] = useState(false)
    const [step, setStep] = useState(0)
    const [targetRect, setTargetRect] = useState<DOMRect | null>(null)

    // Check localStorage on mount
    useEffect(() => {
        try {
            if (!localStorage.getItem(STORAGE_KEY)) {
                setActive(true)
            }
        } catch {
            // localStorage not available — skip tour
        }
    }, [])

    // Measure target element position
    useEffect(() => {
        if (!active) return
        const current = STEPS[step]
        if (!current.target) {
            setTargetRect(null)
            return
        }
        const el = document.querySelector(current.target)
        if (el) {
            setTargetRect(el.getBoundingClientRect())
        } else {
            setTargetRect(null)
        }
    }, [active, step])

    // Re-measure on resize
    useEffect(() => {
        if (!active) return
        const handler = () => {
            const current = STEPS[step]
            if (!current.target) return
            const el = document.querySelector(current.target)
            if (el) setTargetRect(el.getBoundingClientRect())
        }
        window.addEventListener("resize", handler)
        return () => window.removeEventListener("resize", handler)
    }, [active, step])

    const complete = useCallback(() => {
        setActive(false)
        try {
            localStorage.setItem(STORAGE_KEY, "true")
        } catch {
            // ignore
        }
    }, [])

    const next = useCallback(() => {
        if (step >= STEPS.length - 1) {
            complete()
        } else {
            setStep((s) => s + 1)
        }
    }, [step, complete])

    const skip = useCallback(() => {
        complete()
    }, [complete])

    if (!active) return null

    const current = STEPS[step]
    const Icon = current.icon
    const isCenter = current.target === null
    const isLast = step === STEPS.length - 1

    // Tooltip positioning near target
    let tooltipStyle: React.CSSProperties
    if (isCenter || !targetRect) {
        tooltipStyle = {
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
        }
    } else {
        // Position to the right of the target element, or below on small screens
        const spaceRight = window.innerWidth - targetRect.right
        if (spaceRight > 340) {
            tooltipStyle = {
                position: "fixed",
                top: Math.max(16, targetRect.top - 10),
                left: targetRect.right + 16,
            }
        } else {
            tooltipStyle = {
                position: "fixed",
                top: targetRect.bottom + 12,
                left: Math.max(16, targetRect.left - 40),
            }
        }
    }

    return (
        <div className="fixed inset-0 z-[100]">
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />

            {/* Spotlight cutout on target */}
            {targetRect && !isCenter && (
                <div
                    className="absolute rounded-md ring-2 ring-purple-500 ring-offset-2 ring-offset-transparent"
                    style={{
                        top: targetRect.top - 4,
                        left: targetRect.left - 4,
                        width: targetRect.width + 8,
                        height: targetRect.height + 8,
                        backgroundColor: "rgba(124, 58, 237, 0.08)",
                        boxShadow: "0 0 0 9999px rgba(0,0,0,0.7)",
                        zIndex: 101,
                    }}
                />
            )}

            {/* Tooltip card */}
            <div
                style={tooltipStyle}
                className={cn(
                    "z-[102] w-[320px] rounded-lg border border-purple-800/60 bg-slate-900 p-5 shadow-2xl shadow-purple-950/40",
                    isCenter && "w-[380px]",
                )}
            >
                {/* Close / Skip */}
                <button
                    onClick={skip}
                    className="absolute top-3 right-3 text-slate-500 hover:text-slate-300 transition-colors"
                    aria-label="Skip tour"
                >
                    <X className="h-4 w-4" />
                </button>

                {/* Icon */}
                <div
                    className={cn(
                        "mb-3 flex h-10 w-10 items-center justify-center rounded-lg",
                        isCenter
                            ? "bg-purple-600/20 text-purple-400"
                            : "bg-purple-600/15 text-purple-400",
                    )}
                >
                    <Icon className="h-5 w-5" />
                </div>

                {/* Title */}
                <h3 className="mb-1.5 text-base font-semibold text-slate-100">
                    {current.title}
                </h3>

                {/* Description */}
                <p className="mb-4 text-sm leading-relaxed text-slate-400">
                    {current.description}
                </p>

                {/* Footer: step counter + buttons */}
                <div className="flex items-center justify-between">
                    {/* Step dots */}
                    <div className="flex items-center gap-1.5">
                        {STEPS.map((_, i) => (
                            <div
                                key={i}
                                className={cn(
                                    "h-1.5 rounded-full transition-all duration-200",
                                    i === step
                                        ? "w-4 bg-purple-500"
                                        : i < step
                                            ? "w-1.5 bg-purple-700"
                                            : "w-1.5 bg-slate-700",
                                )}
                            />
                        ))}
                        <span className="ml-2 text-[10px] font-mono text-slate-600">
                            {step + 1}/{STEPS.length}
                        </span>
                    </div>

                    {/* Buttons */}
                    <div className="flex items-center gap-2">
                        {!isLast && (
                            <button
                                onClick={skip}
                                className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
                            >
                                Skip Tour
                            </button>
                        )}
                        <button
                            onClick={next}
                            className={cn(
                                "flex items-center gap-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                                "bg-purple-600 text-white hover:bg-purple-500",
                            )}
                        >
                            {isLast ? "Get Started" : "Next"}
                            {!isLast && <ChevronRight className="h-3 w-3" />}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default OnboardingTour
