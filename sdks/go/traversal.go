// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// Bfs performs a breadth-first search from the given start node.
// Returns visited node IDs in BFS order.
func (c *NietzscheClient) Bfs(ctx context.Context, startID string, opts TraversalOpts, collection string) ([]string, error) {
	resp, err := c.stub.Bfs(ctx, &pb.TraversalRequest{
		StartNodeId: startID,
		MaxDepth:    opts.MaxDepth,
		MaxNodes:    opts.MaxNodes,
		EnergyMin:   opts.EnergyMin,
		Collection:  collection,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche Bfs: %w", err)
	}

	return resp.VisitedIds, nil
}

// Dijkstra performs a shortest-path traversal from the given start node.
// Returns visited node IDs and their costs (parallel arrays).
func (c *NietzscheClient) Dijkstra(ctx context.Context, startID string, opts TraversalOpts, collection string) ([]string, []float64, error) {
	resp, err := c.stub.Dijkstra(ctx, &pb.TraversalRequest{
		StartNodeId: startID,
		MaxDepth:    opts.MaxDepth,
		MaxCost:     opts.MaxCost,
		MaxNodes:    opts.MaxNodes,
		EnergyMin:   opts.EnergyMin,
		Collection:  collection,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("nietzsche Dijkstra: %w", err)
	}

	return resp.VisitedIds, resp.Costs, nil
}

// Diffuse runs heat-kernel diffusion from the specified source nodes.
// Returns results grouped by diffusion time scale.
func (c *NietzscheClient) Diffuse(ctx context.Context, sourceIDs []string, opts DiffuseOpts) ([]DiffusionScale, error) {
	resp, err := c.stub.Diffuse(ctx, &pb.DiffusionRequest{
		SourceIds:  sourceIDs,
		TValues:    opts.TValues,
		KChebyshev: opts.KChebyshev,
		Collection: opts.Collection,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche Diffuse: %w", err)
	}

	scales := make([]DiffusionScale, len(resp.Scales))
	for i, s := range resp.Scales {
		scales[i] = DiffusionScale{
			T:       s.T,
			NodeIDs: s.NodeIds,
			Scores:  s.Scores,
		}
	}

	return scales, nil
}
