// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"encoding/json"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// InsertNode creates a new node in NietzscheDB.
// Content is automatically JSON-marshalled to bytes.
func (c *NietzscheClient) InsertNode(ctx context.Context, opts InsertNodeOpts) (NodeResult, error) {
	var contentBytes []byte
	if opts.Content != nil {
		var err error
		contentBytes, err = json.Marshal(opts.Content)
		if err != nil {
			return NodeResult{}, fmt.Errorf("nietzsche InsertNode: failed to marshal content: %w", err)
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

	resp, err := c.stub.InsertNode(ctx, req)
	if err != nil {
		return NodeResult{}, fmt.Errorf("nietzsche InsertNode: %w", err)
	}

	return nodeResponseToResult(resp), nil
}

// GetNode retrieves a node by ID from the specified collection.
func (c *NietzscheClient) GetNode(ctx context.Context, id, collection string) (NodeResult, error) {
	resp, err := c.stub.GetNode(ctx, &pb.NodeIdRequest{
		Id:         id,
		Collection: collection,
	})
	if err != nil {
		return NodeResult{}, fmt.Errorf("nietzsche GetNode: %w", err)
	}

	return nodeResponseToResult(resp), nil
}

// DeleteNode removes a node by ID from the specified collection.
func (c *NietzscheClient) DeleteNode(ctx context.Context, id, collection string) error {
	resp, err := c.stub.DeleteNode(ctx, &pb.NodeIdRequest{
		Id:         id,
		Collection: collection,
	})
	if err != nil {
		return fmt.Errorf("nietzsche DeleteNode: %w", err)
	}

	if resp.Status == "error" {
		return fmt.Errorf("nietzsche DeleteNode: %s", resp.Error)
	}

	return nil
}

// UpdateEnergy modifies a node's energy level.
func (c *NietzscheClient) UpdateEnergy(ctx context.Context, nodeID string, energy float32, collection string) error {
	resp, err := c.stub.UpdateEnergy(ctx, &pb.UpdateEnergyRequest{
		NodeId:     nodeID,
		Energy:     energy,
		Collection: collection,
	})
	if err != nil {
		return fmt.Errorf("nietzsche UpdateEnergy: %w", err)
	}

	if resp.Status == "error" {
		return fmt.Errorf("nietzsche UpdateEnergy: %s", resp.Error)
	}

	return nil
}

// nodeResponseToResult converts a protobuf NodeResponse to an SDK NodeResult.
func nodeResponseToResult(resp *pb.NodeResponse) NodeResult {
	result := NodeResult{
		Found:          resp.Found,
		ID:             resp.Id,
		Energy:         resp.Energy,
		Depth:          resp.Depth,
		HausdorffLocal: resp.HausdorffLocal,
		CreatedAt:      resp.CreatedAt,
		NodeType:       resp.NodeType,
	}

	if resp.Embedding != nil {
		result.Embedding = resp.Embedding.Coords
	}

	if len(resp.Content) > 0 {
		var content map[string]interface{}
		if err := json.Unmarshal(resp.Content, &content); err == nil {
			result.Content = content
		}
	}

	return result
}
