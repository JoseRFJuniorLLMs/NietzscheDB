// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"encoding/json"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// MergeNode performs an upsert: finds a node by node_type + match_keys, or creates one.
//
// This is the NietzscheDB equivalent of NietzscheDB's MERGE statement:
//
//	MERGE (t:Topic {name: $name})
//	ON CREATE SET t.created = datetime()
//	ON MATCH SET t.count = t.count + 1
//
// Translates to:
//
//	sdk.MergeNode(ctx, MergeNodeOpts{
//	    Collection: "patient_graph",
//	    NodeType:   "Topic",
//	    MatchKeys:  map[string]interface{}{"name": name},
//	    OnCreateSet: map[string]interface{}{"created": time.Now().Unix()},
//	    OnMatchSet:  map[string]interface{}{"count_increment": 1},
//	})
func (c *NietzscheClient) MergeNode(ctx context.Context, opts MergeNodeOpts) (*MergeNodeResult, error) {
	matchKeysJSON, err := json.Marshal(opts.MatchKeys)
	if err != nil {
		return nil, fmt.Errorf("nietzsche MergeNode: marshal match_keys: %w", err)
	}

	var onCreateJSON []byte
	if len(opts.OnCreateSet) > 0 {
		onCreateJSON, err = json.Marshal(opts.OnCreateSet)
		if err != nil {
			return nil, fmt.Errorf("nietzsche MergeNode: marshal on_create_set: %w", err)
		}
	}

	var onMatchJSON []byte
	if len(opts.OnMatchSet) > 0 {
		onMatchJSON, err = json.Marshal(opts.OnMatchSet)
		if err != nil {
			return nil, fmt.Errorf("nietzsche MergeNode: marshal on_match_set: %w", err)
		}
	}

	req := &pb.MergeNodeRequest{
		Collection:  opts.Collection,
		NodeType:    opts.NodeType,
		MatchKeys:   matchKeysJSON,
		OnCreateSet: onCreateJSON,
		OnMatchSet:  onMatchJSON,
		Energy:      opts.Energy,
	}

	// Set embedding if provided
	if len(opts.Coords) > 0 {
		req.Embedding = &pb.PoincareVector{
			Coords: opts.Coords,
			Dim:    uint32(len(opts.Coords)),
		}
	}

	resp, err := c.stub.MergeNode(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("nietzsche MergeNode: %w", err)
	}

	result := &MergeNodeResult{
		Created: resp.Created,
		NodeID:  resp.NodeId,
	}

	if resp.Node != nil {
		result.Node = nodeResponseToResult(resp.Node)
	}

	return result, nil
}

// MergeEdge performs an upsert: finds an edge by (from, to, type), or creates one.
//
// This is the NietzscheDB equivalent of NietzscheDB's MERGE on relationships:
//
//	MERGE (a)-[:KNOWS]->(b)
func (c *NietzscheClient) MergeEdge(ctx context.Context, opts MergeEdgeOpts) (*MergeEdgeResult, error) {
	var onCreateJSON []byte
	var err error
	if len(opts.OnCreateSet) > 0 {
		onCreateJSON, err = json.Marshal(opts.OnCreateSet)
		if err != nil {
			return nil, fmt.Errorf("nietzsche MergeEdge: marshal on_create_set: %w", err)
		}
	}

	var onMatchJSON []byte
	if len(opts.OnMatchSet) > 0 {
		onMatchJSON, err = json.Marshal(opts.OnMatchSet)
		if err != nil {
			return nil, fmt.Errorf("nietzsche MergeEdge: marshal on_match_set: %w", err)
		}
	}

	resp, err := c.stub.MergeEdge(ctx, &pb.MergeEdgeRequest{
		Collection:  opts.Collection,
		FromNodeId:  opts.FromNodeID,
		ToNodeId:    opts.ToNodeID,
		EdgeType:    opts.EdgeType,
		OnCreateSet: onCreateJSON,
		OnMatchSet:  onMatchJSON,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche MergeEdge: %w", err)
	}

	return &MergeEdgeResult{
		Created: resp.Created,
		EdgeID:  resp.EdgeId,
	}, nil
}
