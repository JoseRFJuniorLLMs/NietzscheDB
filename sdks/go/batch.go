// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"encoding/json"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// BatchInsertNodes inserts multiple nodes in a single RPC call.
// Returns the server-assigned UUIDs in the same order as the input slice.
func (c *NietzscheClient) BatchInsertNodes(ctx context.Context, nodes []InsertNodeOpts, collection string) ([]string, error) {
	pbNodes := make([]*pb.InsertNodeRequest, len(nodes))
	for i, opts := range nodes {
		var contentBytes []byte
		if opts.Content != nil {
			var err error
			contentBytes, err = json.Marshal(opts.Content)
			if err != nil {
				return nil, fmt.Errorf("nietzsche BatchInsertNodes: failed to marshal content for node %d: %w", i, err)
			}
		}

		req := &pb.InsertNodeRequest{
			Id:         opts.ID,
			Content:    contentBytes,
			NodeType:   opts.NodeType,
			Energy:     opts.Energy,
			Collection: opts.Collection,
		}

		if len(opts.Coords) > 0 {
			req.Embedding = &pb.PoincareVector{
				Coords: opts.Coords,
				Dim:    uint32(len(opts.Coords)),
			}
		}

		pbNodes[i] = req
	}

	resp, err := c.stub.BatchInsertNodes(ctx, &pb.BatchInsertNodesRequest{
		Nodes:      pbNodes,
		Collection: collection,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche BatchInsertNodes: %w", err)
	}

	return resp.NodeIds, nil
}

// BatchInsertEdges inserts multiple edges in a single RPC call.
// Returns the server-assigned UUIDs in the same order as the input slice.
func (c *NietzscheClient) BatchInsertEdges(ctx context.Context, edges []InsertEdgeOpts, collection string) ([]string, error) {
	pbEdges := make([]*pb.InsertEdgeRequest, len(edges))
	for i, opts := range edges {
		pbEdges[i] = &pb.InsertEdgeRequest{
			Id:         opts.ID,
			From:       opts.From,
			To:         opts.To,
			EdgeType:   opts.EdgeType,
			Weight:     opts.Weight,
			Collection: opts.Collection,
		}
	}

	resp, err := c.stub.BatchInsertEdges(ctx, &pb.BatchInsertEdgesRequest{
		Edges:      pbEdges,
		Collection: collection,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche BatchInsertEdges: %w", err)
	}

	return resp.EdgeIds, nil
}
