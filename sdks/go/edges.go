// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// InsertEdge creates a new edge between two nodes.
// Returns the edge ID (server-generated if opts.ID is empty).
func (c *NietzscheClient) InsertEdge(ctx context.Context, opts InsertEdgeOpts) (string, error) {
	resp, err := c.stub.InsertEdge(ctx, &pb.InsertEdgeRequest{
		Id:         opts.ID,
		From:       opts.From,
		To:         opts.To,
		EdgeType:   opts.EdgeType,
		Weight:     opts.Weight,
		Collection: opts.Collection,
	})
	if err != nil {
		return "", fmt.Errorf("nietzsche InsertEdge: %w", err)
	}

	if !resp.Success {
		return "", fmt.Errorf("nietzsche InsertEdge: server returned success=false")
	}

	return resp.Id, nil
}

// DeleteEdge removes an edge by ID from the specified collection.
func (c *NietzscheClient) DeleteEdge(ctx context.Context, id, collection string) error {
	resp, err := c.stub.DeleteEdge(ctx, &pb.EdgeIdRequest{
		Id:         id,
		Collection: collection,
	})
	if err != nil {
		return fmt.Errorf("nietzsche DeleteEdge: %w", err)
	}

	if resp.Status == "error" {
		return fmt.Errorf("nietzsche DeleteEdge: %s", resp.Error)
	}

	return nil
}
