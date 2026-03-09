/**
 * PerspektiveView.tsx — Full integration wrapper around perspektive.js PerspektiveEngine
 *
 * This replaces the old custom implementation with the complete perspektive.js engine:
 *  - PerspektiveEngine (4 manifolds, Lerp, Bloom, Schrodinger edges)
 *  - Drag, Box-Select, Context Menu, Mobius Zoom
 *  - SSE + WebSocket streaming with GraphStore
 *  - SearchBar + FilterPanel (node types, energy slider)
 *  - ExportToolbar (PNG, JSON, GraphML)
 *  - DiffusionHeatmap + EnergyPulse overlays
 *  - Theme system with presets (cyberpunk, midnight, paper, matrix, aurora)
 *  - Topological Analysis (Betti numbers)
 *  - GPU-accelerated force layout
 *  - Reasoning trace visualization
 *  - Accessibility (ARIA descriptions)
 */

import { PerspektiveEngine } from "@nietzsche/perspektive"
import type { PerspektiveEngineProps, ManifoldType, StreamingMode } from "@nietzsche/perspektive"
import type { InteractionCallbacks } from "@nietzsche/perspektive"
import type { DreamSession, CausalEdge, CausalChainResult, ZaratustraResult, NarrativeArc } from "@nietzsche/perspektive"

// Re-export types that other parts of the dashboard may need
export type { ManifoldType, StreamingMode }
export type { PerspektiveEngineProps }

// Re-export engine types for graph data
export type { NodeData, EdgeData } from "@nietzsche/perspektive"

export interface AlgorithmOverlay {
    type: "pagerank" | "louvain" | "betweenness" | "degree" | "closeness"
    /** Map of node ID → score (0-1 normalized) */
    scores: Map<string, number>
    /** For community detection: Map of node ID → community ID */
    communities?: Map<string, number>
}

export interface PerspektiveViewProps {
    collection?: string
    apiBase?: string
    limit?: number
    zoom?: number
    bloomIntensity?: number
    lerpRate?: number
    streamingMode?: StreamingMode
    wsUrl?: string
    enableDrag?: boolean
    enableBoxSelect?: boolean
    enableContextMenu?: boolean
    enableMobiusZoom?: boolean
    /** Interaction callbacks (onFocus, onDragEnd, etc.) */
    callbacks?: InteractionCallbacks
    /** Algorithm visualization overlay */
    algorithmOverlay?: AlgorithmOverlay | null
    /** Dream session overlay */
    dreamSession?: DreamSession | null
    onDreamAction?: (action: 'apply' | 'reject', dreamId: string) => void
    /** Causal overlay */
    causalEdges?: CausalEdge[]
    causalChain?: CausalChainResult | null
    /** Zaratustra visualization */
    zaratustraResult?: ZaratustraResult | null
    /** Narrative arcs */
    narrativeArcs?: NarrativeArc[]
    /** Live daemon data */
    activeDaemons?: Array<{ id: string; x: number; y: number; type: 'entropy' | 'evolution' | 'patrol'; energy: number }>
}

/**
 * Drop-in replacement wrapper that renders the full PerspektiveEngine.
 * All internal features (streaming, manifold switching, search, filter,
 * export, overlays, themes, algorithms) are handled by the engine.
 */
export function PerspektiveView({
    collection = "eva_core",
    apiBase = "",
    limit = 2000,
    zoom = 300,
    bloomIntensity = 1.5,
    lerpRate = 0.05,
    streamingMode = "sse",
    wsUrl,
    enableDrag = true,
    enableBoxSelect = true,
    enableContextMenu = true,
    enableMobiusZoom = true,
    callbacks,
    algorithmOverlay: _algorithmOverlay,
    dreamSession,
    onDreamAction,
    causalEdges,
    causalChain,
    zaratustraResult,
    narrativeArcs,
    activeDaemons,
}: PerspektiveViewProps) {
    return (
        <PerspektiveEngine
            collection={collection}
            apiBase={apiBase}
            limit={limit}
            zoom={zoom}
            bloomIntensity={bloomIntensity}
            lerpRate={lerpRate}
            streamingMode={streamingMode}
            wsUrl={wsUrl}
            enableDrag={enableDrag}
            enableBoxSelect={enableBoxSelect}
            enableContextMenu={enableContextMenu}
            enableMobiusZoom={enableMobiusZoom}
            callbacks={callbacks}
            dreamSession={dreamSession}
            onDreamAction={onDreamAction}
            causalEdges={causalEdges}
            causalChain={causalChain}
            zaratustraResult={zaratustraResult}
            narrativeArcs={narrativeArcs}
            activeDaemons={activeDaemons}
        />
    )
}
