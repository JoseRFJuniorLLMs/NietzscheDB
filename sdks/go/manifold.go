// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"

	pb "nietzsche-sdk/pb"
)

// ── Multi-Manifold Types ──────────────────────────────────────────────────

// SynthesisResult contains the result of a Riemann synthesis operation.
// The synthesis point is the "parent concept" that unifies two or more
// concepts via the spherical midpoint in Riemann space.
type SynthesisResult struct {
	// Poincaré ball coordinates of the synthesized point.
	SynthesisCoords []float64
	// UUID of the nearest existing node to the synthesis point.
	NearestNodeID string
	// Hyperbolic distance from synthesis point to nearest node.
	NearestDistance float64
}

// CausalEdge represents a single edge with Minkowski causality metadata.
type CausalEdge struct {
	EdgeID            string
	FromNodeID        string
	ToNodeID          string
	MinkowskiInterval float64 // ds² value: negative = timelike (causal)
	CausalType        string  // "Timelike" | "Spacelike" | "Lightlike" | "Unknown"
	EdgeType          string
}

// CausalChainResult contains the result of a causal chain traversal.
type CausalChainResult struct {
	// Ordered node UUIDs in the causal chain.
	ChainIDs []string
	// Edges in the chain with full causality metadata.
	Edges []CausalEdge
}

// KleinPathResult contains the result of a Klein-model pathfinding operation.
type KleinPathResult struct {
	Found bool
	Path  []string // node UUIDs from start to goal
	Cost  float64  // total hyperbolic distance
}

// ShortestPathCheckResult tells whether a point lies on the geodesic between two others.
type ShortestPathCheckResult struct {
	OnPath   bool    // true if C is on the Klein geodesic A→B
	Distance float64 // Klein distance A→B
}

// ── Riemann Synthesis ─────────────────────────────────────────────────────

// Synthesis computes the dialectical synthesis of two concepts via Riemann space.
//
// Given two nodes (Thesis and Antithesis), projects them onto the unit sphere,
// computes the spherical midpoint, and projects back to the Poincaré ball at
// a shallower depth (more abstract concept).
//
// Returns the synthesis point coordinates and the nearest existing node.
func (c *NietzscheClient) Synthesis(ctx context.Context, nodeIDA, nodeIDB, collection string) (*SynthesisResult, error) {
	resp, err := c.stub.Synthesis(ctx, &pb.SynthesisRequest{
		NodeIdA:    nodeIDA,
		NodeIdB:    nodeIDB,
		Collection: collection,
	})
	if err != nil {
		return nil, err
	}
	return &SynthesisResult{
		SynthesisCoords: resp.GetSynthesisCoords(),
		NearestNodeID:   resp.GetNearestNodeId(),
		NearestDistance:  resp.GetNearestDistance(),
	}, nil
}

// SynthesisMulti computes the N-ary synthesis of multiple concepts.
//
// Generalizes binary synthesis: finds the concept that best unifies
// all input concepts via Fréchet mean on the sphere.
func (c *NietzscheClient) SynthesisMulti(ctx context.Context, nodeIDs []string, collection string) (*SynthesisResult, error) {
	resp, err := c.stub.SynthesisMulti(ctx, &pb.SynthesisMultiRequest{
		NodeIds:    nodeIDs,
		Collection: collection,
	})
	if err != nil {
		return nil, err
	}
	return &SynthesisResult{
		SynthesisCoords: resp.GetSynthesisCoords(),
		NearestNodeID:   resp.GetNearestNodeId(),
		NearestDistance:  resp.GetNearestDistance(),
	}, nil
}

// ── Minkowski Causal Operations ───────────────────────────────────────────

// CausalNeighbors returns only the causally connected neighbors of a node.
//
// Uses the Minkowski interval (ds² < 0) to filter edges down to provably
// causal relationships. Direction controls which half of the light cone:
//   - "future": events caused BY this node
//   - "past": events that CAUSED this node (WHY query)
//   - "both": full light cone (default)
func (c *NietzscheClient) CausalNeighbors(ctx context.Context, nodeID, direction, collection string) ([]CausalEdge, error) {
	resp, err := c.stub.CausalNeighbors(ctx, &pb.CausalNeighborsRequest{
		NodeId:     nodeID,
		Direction:  direction,
		Collection: collection,
	})
	if err != nil {
		return nil, err
	}
	return convertCausalEdges(resp.GetEdges()), nil
}

// CausalChain traverses the causal graph recursively, following only
// timelike (ds² < 0) edges up to maxDepth hops.
//
// This answers "WHY did this node come to exist?" by returning the
// unbroken chain of causal events.
//
// Direction:
//   - "past": follow backward causality (default, for WHY queries)
//   - "future": follow forward causality (for WHAT-IF queries)
func (c *NietzscheClient) CausalChain(ctx context.Context, nodeID string, maxDepth uint32, direction, collection string) (*CausalChainResult, error) {
	resp, err := c.stub.CausalChain(ctx, &pb.CausalChainRequest{
		NodeId:     nodeID,
		MaxDepth:   maxDepth,
		Direction:  direction,
		Collection: collection,
	})
	if err != nil {
		return nil, err
	}
	return &CausalChainResult{
		ChainIDs: resp.GetChainIds(),
		Edges:    convertCausalEdges(resp.GetEdges()),
	}, nil
}

// ── Klein Pathfinding ─────────────────────────────────────────────────────

// KleinPath finds the shortest path between two nodes using Klein-model
// pathfinding where geodesics are straight lines.
//
// This is significantly faster than Poincaré pathfinding for deep graphs
// because colinearity checks are O(1) determinant computations instead
// of expensive trigonometric operations.
func (c *NietzscheClient) KleinPath(ctx context.Context, startNodeID, goalNodeID, collection string) (*KleinPathResult, error) {
	resp, err := c.stub.KleinPath(ctx, &pb.KleinPathRequest{
		StartNodeId: startNodeID,
		GoalNodeId:  goalNodeID,
		Collection:  collection,
	})
	if err != nil {
		return nil, err
	}
	return &KleinPathResult{
		Found: resp.GetFound(),
		Path:  resp.GetPath(),
		Cost:  resp.GetCost(),
	}, nil
}

// IsOnShortestPath checks if node C lies on the Klein geodesic between A and B.
//
// In the Klein model, this is a simple colinearity + betweenness check
// (linear algebra, O(1)) instead of the expensive Poincaré arc computation.
func (c *NietzscheClient) IsOnShortestPath(ctx context.Context, nodeIDA, nodeIDB, nodeIDC, collection string) (*ShortestPathCheckResult, error) {
	resp, err := c.stub.IsOnShortestPath(ctx, &pb.ShortestPathCheckRequest{
		NodeIdA:    nodeIDA,
		NodeIdB:    nodeIDB,
		NodeIdC:    nodeIDC,
		Collection: collection,
	})
	if err != nil {
		return nil, err
	}
	return &ShortestPathCheckResult{
		OnPath:   resp.GetOnPath(),
		Distance: resp.GetDistance(),
	}, nil
}

// ── Helpers ───────────────────────────────────────────────────────────────

func convertCausalEdges(pbEdges []*pb.CausalEdge) []CausalEdge {
	edges := make([]CausalEdge, 0, len(pbEdges))
	for _, e := range pbEdges {
		edges = append(edges, CausalEdge{
			EdgeID:            e.GetEdgeId(),
			FromNodeID:        e.GetFromNodeId(),
			ToNodeID:          e.GetToNodeId(),
			MinkowskiInterval: e.GetMinkowskiInterval(),
			CausalType:        e.GetCausalType(),
			EdgeType:          e.GetEdgeType(),
		})
	}
	return edges
}
