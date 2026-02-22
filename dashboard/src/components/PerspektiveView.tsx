/**
 * PerspektiveView.tsx — Hyperbolic graph renderer for NietzscheDB Dashboard
 *
 * Replaces Cosmograph with a real Poincaré disk engine.
 * Supports 4 manifold lenses: Poincaré, Riemann, Minkowski, Emotion.
 * InstancedMesh for 100k+ nodes, geodesic arcs, Bloom HDR, Lerp animation.
 *
 * Features:
 *  - Search dimming (non-matching nodes fade to darkness)
 *  - Interactive type filter (toggle node_type isolation)
 *  - SDF labels for top 15 elite nodes (troika-three-text)
 *  - Fit View / Zoom +/- camera controls
 *  - Raycaster hover + cyberpunk tooltip
 */

import { useState, useMemo, useRef, useEffect, useCallback } from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { OrthographicCamera, OrbitControls, Line, Html, Text } from "@react-three/drei"
import { EffectComposer, Bloom } from "@react-three/postprocessing"
import * as THREE from "three"
import type { ThreeEvent } from "@react-three/fiber"

// ─── Types ──────────────────────────────────────────────────────────────────────

interface Point2D {
    x: number
    y: number
}

export interface ViewNodeData extends Point2D {
    id: string
    energy: number
    node_type: string
    depth: number
    label: string
    z: number
}

export interface ViewEdgeData {
    source: string
    target: string
    weight: number
}

export type ManifoldType = "POINCARE" | "RIEMANN" | "MINKOWSKI" | "EMOTION"

export interface PerspektiveViewProps {
    nodes: ViewNodeData[]
    edges: ViewEdgeData[]
    manifold: ManifoldType
    onManifoldChange: (m: ManifoldType) => void
    searchTerm: string
    activeFilter: string | null
}

// ─── Poincaré Math ──────────────────────────────────────────────────────────────

const EPSILON = 1e-6

interface ArcGeodesic {
    type: "arc"
    center: Point2D
    radius: number
    startAngle: number
    endAngle: number
    ccw: boolean
}
interface LineGeodesic {
    type: "line"
    p1: Point2D
    p2: Point2D
}
type Geodesic = ArcGeodesic | LineGeodesic

function calculateGeodesic(p1: Point2D, p2: Point2D): Geodesic {
    const d1 = p1.x * p1.x + p1.y * p1.y
    const d2 = p2.x * p2.x + p2.y * p2.y
    const denominator = 2 * (p1.x * p2.y - p2.x * p1.y)

    if (Math.abs(denominator) < EPSILON) {
        return { type: "line", p1, p2 }
    }

    const cx = ((1 + d1) * p2.y - (1 + d2) * p1.y) / denominator
    const cy = (p1.x * (1 + d2) - p2.x * (1 + d1)) / denominator
    const radius = Math.sqrt(cx * cx + cy * cy - 1)

    const startAngle = Math.atan2(p1.y - cy, p1.x - cx)
    const endAngle = Math.atan2(p2.y - cy, p2.x - cx)

    let diff = endAngle - startAngle
    while (diff > Math.PI) diff -= 2 * Math.PI
    while (diff < -Math.PI) diff += 2 * Math.PI

    return {
        type: "arc",
        center: { x: cx, y: cy },
        radius,
        startAngle,
        endAngle,
        ccw: diff > 0,
    }
}

function getVisualRadius(p: Point2D, baseEnergy: number): number {
    const norm = Math.sqrt(p.x * p.x + p.y * p.y)
    const conformalFactor = 2 / (1 - norm * norm)
    return baseEnergy / conformalFactor
}

// ─── Highlight check ────────────────────────────────────────────────────────────

function isNodeHighlighted(node: ViewNodeData, searchTerm: string, activeFilter: string | null): boolean {
    const term = searchTerm.toLowerCase()
    const matchesSearch = term === "" ||
        node.id.toLowerCase().includes(term) ||
        node.node_type.toLowerCase().includes(term) ||
        node.label.toLowerCase().includes(term)
    const matchesFilter = activeFilter === null || node.node_type === activeFilter
    return matchesSearch && matchesFilter
}

// ─── Geodesic Points ────────────────────────────────────────────────────────────

function geodesicPoints(p1: Point2D, p2: Point2D, segments = 40): [number, number, number][] {
    const geo = calculateGeodesic(p1, p2)
    if (geo.type === "line") {
        return [[p1.x, p1.y, 0], [p2.x, p2.y, 0]]
    }
    const curve = new THREE.EllipseCurve(
        geo.center.x, geo.center.y,
        geo.radius, geo.radius,
        -geo.startAngle, -geo.endAngle,
        !geo.ccw, 0,
    )
    return curve.getPoints(segments).map(v => [v.x, v.y, 0])
}

// ─── Camera API (bridge between Canvas internals and HTML overlay buttons) ──────

interface CameraAPIHandle {
    zoomIn: () => void
    zoomOut: () => void
    fitView: () => void
}

const CameraAPI = ({ apiRef }: { apiRef: React.MutableRefObject<CameraAPIHandle> }) => {
    const { camera } = useThree()

    useEffect(() => {
        apiRef.current = {
            zoomIn: () => {
                ;(camera as THREE.OrthographicCamera).zoom *= 1.3
                camera.updateProjectionMatrix()
            },
            zoomOut: () => {
                ;(camera as THREE.OrthographicCamera).zoom /= 1.3
                camera.updateProjectionMatrix()
            },
            fitView: () => {
                camera.position.set(0, 0, 5)
                ;(camera as THREE.OrthographicCamera).zoom = 300
                camera.updateProjectionMatrix()
            },
        }
    }, [camera, apiRef])

    return null
}

// ─── Graph Edges ────────────────────────────────────────────────────────────────

const GraphEdges = ({ nodes, edges, manifold, searchTerm, activeFilter }: {
    nodes: ViewNodeData[]
    edges: ViewEdgeData[]
    manifold: ManifoldType
    searchTerm: string
    activeFilter: string | null
}) => {
    const nodeMap = useMemo(() => new Map(nodes.map(n => [n.id, n])), [nodes])
    const hasFilter = searchTerm !== "" || activeFilter !== null

    const lines = useMemo(() => {
        return edges.map(edge => {
            const p1 = nodeMap.get(edge.source)
            const p2 = nodeMap.get(edge.target)
            if (!p1 || !p2) return null

            // Dim edges where neither endpoint is highlighted
            const eitherHighlighted = !hasFilter ||
                isNodeHighlighted(p1, searchTerm, activeFilter) ||
                isNodeHighlighted(p2, searchTerm, activeFilter)

            const baseOpacity = 0.1 + (edge.weight * 0.4)
            const opacity = eitherHighlighted ? baseOpacity : baseOpacity * 0.08
            const hdrColor = new THREE.Color(0x00d8ff).multiplyScalar(eitherHighlighted ? 1.5 : 0.2)

            const points: [number, number, number][] = manifold === "POINCARE"
                ? geodesicPoints(p1, p2, 40)
                : [[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]]

            return (
                <Line
                    key={`${edge.source}-${edge.target}`}
                    points={points}
                    color={hdrColor}
                    lineWidth={1}
                    transparent
                    opacity={opacity}
                    blending={THREE.AdditiveBlending}
                />
            )
        }).filter(Boolean)
    }, [edges, nodeMap, manifold, hasFilter, searchTerm, activeFilter])

    return <group>{lines}</group>
}

// ─── Graph Nodes (InstancedMesh + Lerp + Hover + Search Dimming) ────────────────

const NODE_COLORS: Record<string, number> = {
    Pruned:        0x1e293b,
    Episodic:      0x00f0ff,
    Concept:       0xf59e0b,
    DreamSnapshot: 0x8b5cf6,
    Somatic:       0x22c55e,
    Linguistic:    0xf43f5e,
    Composite:     0xa78bfa,
}

const GraphNodes = ({ nodes, manifold, searchTerm, activeFilter }: {
    nodes: ViewNodeData[]
    manifold: ManifoldType
    searchTerm: string
    activeFilter: string | null
}) => {
    const meshRef = useRef<THREE.InstancedMesh>(null)
    const [hoveredNode, setHoveredNode] = useState<ViewNodeData | null>(null)

    const dummy = useMemo(() => new THREE.Object3D(), [])
    const color = useMemo(() => new THREE.Color(), [])
    const targetPos = useMemo(() => new THREE.Vector3(), [])
    const currentPos = useMemo(() => new THREE.Vector3(), [])

    const hasFilter = searchTerm !== "" || activeFilter !== null

    // Lerp + color update every frame
    useFrame(() => {
        if (!meshRef.current || nodes.length === 0) return
        let matrixDirty = false
        let colorDirty = false

        nodes.forEach((node, i) => {
            meshRef.current!.getMatrixAt(i, dummy.matrix)
            currentPos.setFromMatrixPosition(dummy.matrix)
            targetPos.set(node.x, node.y, node.z)

            const highlighted = !hasFilter || isNodeHighlighted(node, searchTerm, activeFilter)

            // Size: highlighted = full, dimmed = tiny
            const baseRadius = highlighted ? (0.01 + node.energy * 0.03) : 0.004
            const radius = manifold === "POINCARE" ? getVisualRadius(node, baseRadius) : baseRadius

            // Lerp position
            if (currentPos.distanceTo(targetPos) > 0.0005) {
                currentPos.lerp(targetPos, 0.05)
                matrixDirty = true
            }

            dummy.position.copy(currentPos)
            dummy.scale.set(radius, radius, 1)
            if (manifold === "RIEMANN") dummy.lookAt(0, 0, 0)
            dummy.updateMatrix()
            meshRef.current!.setMatrixAt(i, dummy.matrix)

            // Color
            if (!highlighted) {
                color.setHex(0x0a0a1a) // Near-black (dimmed into the abyss)
            } else {
                color.setHex(NODE_COLORS[node.node_type] ?? 0x00ff66)
                if (node.energy > 0.8) {
                    color.setHex(0xff00ff)
                    color.multiplyScalar(4.0)
                } else if (node.energy > 0.5) {
                    color.multiplyScalar(2.0)
                }
            }
            meshRef.current!.setColorAt(i, color)
            colorDirty = true
        })

        if (matrixDirty) meshRef.current.instanceMatrix.needsUpdate = true
        if (colorDirty && meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true
    })

    const handlePointerMove = useCallback((e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation()
        if (e.instanceId !== undefined && e.instanceId < nodes.length) {
            document.body.style.cursor = "crosshair"
            setHoveredNode(nodes[e.instanceId])
        }
    }, [nodes])

    const handlePointerOut = useCallback(() => {
        document.body.style.cursor = "auto"
        setHoveredNode(null)
    }, [])

    // Top 15 elite nodes for SDF labels
    const eliteNodes = useMemo(() => {
        const candidates = hasFilter
            ? nodes.filter(n => isNodeHighlighted(n, searchTerm, activeFilter))
            : nodes
        return [...candidates].sort((a, b) => b.energy - a.energy).slice(0, 15)
    }, [nodes, hasFilter, searchTerm, activeFilter])

    return (
        <group>
            <instancedMesh
                ref={meshRef}
                args={[undefined, undefined, Math.max(nodes.length, 1)]}
                onPointerMove={handlePointerMove}
                onPointerOut={handlePointerOut}
            >
                <circleGeometry args={[1, 32]} />
                <meshBasicMaterial toneMapped={false} />
            </instancedMesh>

            {/* SDF Labels — only top 15 elite nodes */}
            {eliteNodes.map(node => (
                <Text
                    key={`lbl-${node.id}`}
                    position={[node.x, node.y - 0.025, node.z + 0.01]}
                    fontSize={0.015}
                    color="#ffffff"
                    anchorX="center"
                    anchorY="top"
                    outlineWidth={0.002}
                    outlineColor="#000000"
                    maxWidth={0.15}
                >
                    {node.label}
                </Text>
            ))}

            {/* Tooltip */}
            {hoveredNode && (
                <Html position={[hoveredNode.x, hoveredNode.y + 0.05, hoveredNode.z]} center style={{ pointerEvents: "none" }}>
                    <div style={{
                        background: "rgba(2, 6, 23, 0.9)",
                        border: "1px solid #00f0ff",
                        color: "#fff",
                        padding: "10px",
                        borderRadius: "4px",
                        fontFamily: "monospace",
                        width: "220px",
                        backdropFilter: "blur(4px)",
                        boxShadow: "0 0 10px rgba(0, 240, 255, 0.5)",
                    }}>
                        <div style={{ color: "#00f0ff", fontWeight: "bold", marginBottom: 4 }}>
                            {hoveredNode.node_type}
                        </div>
                        <div style={{ fontSize: 11, color: "#e2e8f0" }}>
                            {hoveredNode.label}
                        </div>
                        <div style={{ fontSize: 11, color: "#94a3b8" }}>
                            ID: {hoveredNode.id.substring(0, 12)}...
                        </div>
                        <div style={{ fontSize: 11, color: "#ff00ff" }}>
                            Energy: {(hoveredNode.energy * 100).toFixed(1)}%
                        </div>
                        <div style={{ fontSize: 11, color: "#00ff66" }}>
                            Depth: {hoveredNode.depth.toFixed(3)}
                        </div>
                    </div>
                </Html>
            )}
        </group>
    )
}

// ─── Disk Border (Poincaré only) ────────────────────────────────────────────────

const DiskBorder = () => {
    const points = useMemo(() => {
        const pts: [number, number, number][] = []
        for (let i = 0; i <= 64; i++) {
            const angle = (i / 64) * Math.PI * 2
            pts.push([Math.cos(angle) * 1.005, Math.sin(angle) * 1.005, 0])
        }
        return pts
    }, [])
    return <Line points={points} color="#1e293b" transparent opacity={0.3} lineWidth={1} />
}

// ─── Riemann Wireframe ──────────────────────────────────────────────────────────

const RiemannWireframe = ({ manifold }: { manifold: ManifoldType }) => {
    if (manifold !== "RIEMANN") return null
    return (
        <mesh>
            <sphereGeometry args={[1, 32, 32]} />
            <meshBasicMaterial color="#1e293b" wireframe transparent opacity={0.15} />
        </mesh>
    )
}

// ─── Minkowski Light Cones ──────────────────────────────────────────────────────

const MinkowskiLightCones = ({ nodes, manifold }: { nodes: ViewNodeData[]; manifold: ManifoldType }) => {
    if (manifold !== "MINKOWSKI") return null

    const eliteNodes = nodes.filter(n => n.energy > 0.8)

    return (
        <group>
            {eliteNodes.map(node => (
                <group key={`cone-${node.id}`} position={[node.x, node.y, node.z]}>
                    <mesh position={[0, 1, 0]}>
                        <coneGeometry args={[1, 2, 32, 1, true]} />
                        <meshBasicMaterial
                            color="#00f0ff" transparent opacity={0.08}
                            side={THREE.DoubleSide} depthWrite={false}
                            blending={THREE.AdditiveBlending}
                        />
                    </mesh>
                    <mesh position={[0, -1, 0]} rotation={[Math.PI, 0, 0]}>
                        <coneGeometry args={[1, 2, 32, 1, true]} />
                        <meshBasicMaterial
                            color="#ff00ff" transparent opacity={0.08}
                            side={THREE.DoubleSide} depthWrite={false}
                            blending={THREE.AdditiveBlending}
                        />
                    </mesh>
                </group>
            ))}
            <gridHelper args={[10, 20, 0x334155, 0x1e293b]} position={[0, -2.5, 0]} />
        </group>
    )
}

// ─── Main Component ─────────────────────────────────────────────────────────────

export function PerspektiveView({
    nodes, edges, manifold, onManifoldChange, searchTerm, activeFilter,
}: PerspektiveViewProps) {
    const is2D = manifold === "POINCARE" || manifold === "EMOTION"
    const cameraAPI = useRef<CameraAPIHandle>({ zoomIn() {}, zoomOut() {}, fitView() {} })

    // Count visible nodes
    const visibleCount = useMemo(() => {
        if (searchTerm === "" && activeFilter === null) return nodes.length
        return nodes.filter(n => isNodeHighlighted(n, searchTerm, activeFilter)).length
    }, [nodes, searchTerm, activeFilter])

    return (
        <div className="relative w-full h-full" style={{ background: "#000" }}>
            <Canvas>
                {is2D && <OrthographicCamera makeDefault position={[0, 0, 5]} zoom={300} />}

                <OrbitControls
                    enableRotate={!is2D}
                    enablePan={true}
                    enableZoom={true}
                />

                <CameraAPI apiRef={cameraAPI} />

                {/* Poincaré disk + border */}
                {manifold === "POINCARE" && (
                    <>
                        <mesh position={[0, 0, -0.1]}>
                            <circleGeometry args={[1, 64]} />
                            <meshBasicMaterial color="#050510" />
                        </mesh>
                        <DiskBorder />
                    </>
                )}

                {/* Emotion circumplex axes */}
                {manifold === "EMOTION" && (
                    <group>
                        <Line points={[[-1, 0, 0], [1, 0, 0]]} color="#334155" transparent opacity={0.3} lineWidth={1} />
                        <Line points={[[0, -1, 0], [0, 1, 0]]} color="#334155" transparent opacity={0.3} lineWidth={1} />
                    </group>
                )}

                <RiemannWireframe manifold={manifold} />
                <MinkowskiLightCones nodes={nodes} manifold={manifold} />

                <GraphEdges nodes={nodes} edges={edges} manifold={manifold} searchTerm={searchTerm} activeFilter={activeFilter} />
                <GraphNodes nodes={nodes} manifold={manifold} searchTerm={searchTerm} activeFilter={activeFilter} />

                <EffectComposer enableNormalPass={false}>
                    <Bloom luminanceThreshold={0.2} mipmapBlur intensity={1.5} radius={0.8} />
                </EffectComposer>
            </Canvas>

            {/* Camera Controls (top-right) */}
            <div className="absolute top-3 right-3 flex gap-1 z-10">
                <button
                    onClick={() => cameraAPI.current.zoomIn()}
                    className="h-7 w-7 rounded bg-black/80 border border-border/40 text-xs text-muted-foreground hover:text-foreground hover:border-[#00f0ff] font-mono flex items-center justify-center transition-colors"
                    title="Zoom In"
                >+</button>
                <button
                    onClick={() => cameraAPI.current.zoomOut()}
                    className="h-7 w-7 rounded bg-black/80 border border-border/40 text-xs text-muted-foreground hover:text-foreground hover:border-[#00f0ff] font-mono flex items-center justify-center transition-colors"
                    title="Zoom Out"
                >-</button>
                <button
                    onClick={() => cameraAPI.current.fitView()}
                    className="h-7 px-2 rounded bg-black/80 border border-border/40 text-[10px] text-muted-foreground hover:text-foreground hover:border-[#00f0ff] font-mono flex items-center justify-center transition-colors"
                    title="Fit View"
                >FIT</button>
            </div>

            {/* Manifold Switcher (bottom-center) */}
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2 bg-black/80 px-4 py-2 rounded-lg border border-border/30 backdrop-blur z-10">
                {(["POINCARE", "RIEMANN", "MINKOWSKI", "EMOTION"] as ManifoldType[]).map(m => (
                    <button
                        key={m}
                        onClick={() => onManifoldChange(m)}
                        className={`px-3 py-1.5 rounded text-xs font-mono font-bold transition-colors ${
                            manifold === m
                                ? "bg-[#00f0ff] text-black"
                                : "bg-transparent text-muted-foreground border border-border/40 hover:text-foreground"
                        }`}
                    >
                        {m}
                    </button>
                ))}
            </div>

            {/* HUD Overlay (bottom-left) */}
            <div className="absolute bottom-4 left-3 pointer-events-none font-mono text-[10px] z-10" style={{ color: "#00f0ff", textShadow: "0 0 5px #00f0ff" }}>
                <div>
                    {visibleCount === nodes.length
                        ? `${nodes.length} NODES`
                        : <><span style={{ color: "#fff" }}>{visibleCount}</span> / {nodes.length} NODES</>
                    } | {edges.length} EDGES
                </div>
                <div>MANIFOLD: {manifold} {manifold === "POINCARE" ? "K < 0" : manifold === "RIEMANN" ? "K > 0" : manifold === "MINKOWSKI" ? "ds\u00B2" : "V\u00D7A"}</div>
            </div>
        </div>
    )
}
