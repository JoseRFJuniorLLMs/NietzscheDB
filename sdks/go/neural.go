// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// GnnInfer performs a GNN inference on a list of nodes.
func (c *NietzscheClient) GnnInfer(ctx context.Context, opts GnnInferOpts) (GnnInferResult, error) {
	req := &pb.GnnInferRequest{
		ModelName:  opts.ModelName,
		Collection: opts.Collection,
		NodeIds:    opts.NodeIDs,
	}

	resp, err := c.stub.GnnInfer(ctx, req)
	if err != nil {
		return GnnInferResult{}, fmt.Errorf("nietzsche: GnnInfer failed: %w", err)
	}

	embeddings := make([][]float64, len(resp.Embeddings))
	for i, v := range resp.Embeddings {
		embeddings[i] = v.Coords
	}

	return GnnInferResult{Embeddings: embeddings}, nil
}

// MctsSearch performs a Monte Carlo Tree Search for the best action starting from a node.
func (c *NietzscheClient) MctsSearch(ctx context.Context, opts MctsOpts) (MctsResult, error) {
	req := &pb.MctsRequest{
		ModelName:   opts.ModelName,
		StartNodeId: opts.StartNodeID,
		Simulations: opts.Simulations,
		Collection:  opts.Collection,
	}

	resp, err := c.stub.MctsSearch(ctx, req)
	if err != nil {
		return MctsResult{}, fmt.Errorf("nietzsche: MctsSearch failed: %w", err)
	}

	return MctsResult{
		BestActionID: resp.BestActionId,
		Value:        resp.Value,
	}, nil
}
