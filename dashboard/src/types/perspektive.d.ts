// Temporary type bridge for perspektive@0.1.3.
// The published package declares dist/index.d.ts but that file is absent from
// the npm artifact. Remove this shim when the upstream package ships types.

declare module "@nietzsche/perspektive" {
  import type { ComponentType } from "react"

  export type ManifoldType = string
  export type StreamingMode = string

  export type InteractionCallbacks = Record<string, unknown>
  export type DreamSession = any
  export type CausalEdge = any
  export type CausalChainResult = any
  export type ZaratustraResult = any
  export type NarrativeArc = any
  export type NodeData = any
  export type EdgeData = any

  export interface PerspektiveEngineProps {
    [key: string]: any
  }

  export const PerspektiveEngine: ComponentType<PerspektiveEngineProps>
}
