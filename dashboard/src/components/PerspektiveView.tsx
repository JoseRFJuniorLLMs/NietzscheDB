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

// Re-export types that other parts of the dashboard may need
export type { ManifoldType, StreamingMode }
export type { PerspektiveEngineProps }

// Re-export engine types for graph data
export type { NodeData, EdgeData } from "@nietzsche/perspektive"

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
        />
    )
}
