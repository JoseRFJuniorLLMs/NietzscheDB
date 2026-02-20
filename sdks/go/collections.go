// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// CreateCollection creates a new collection in NietzscheDB.
// Returns true if the collection was newly created, false if it already existed (idempotent).
func (c *NietzscheClient) CreateCollection(ctx context.Context, cfg CollectionConfig) (bool, error) {
	resp, err := c.stub.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Collection: cfg.Name,
		Dim:        cfg.Dim,
		Metric:     cfg.Metric,
	})
	if err != nil {
		return false, fmt.Errorf("nietzsche CreateCollection: %w", err)
	}

	return resp.Created, nil
}

// DropCollection permanently removes a collection and all its data.
// Cannot drop the "default" collection.
func (c *NietzscheClient) DropCollection(ctx context.Context, name string) error {
	resp, err := c.stub.DropCollection(ctx, &pb.DropCollectionRequest{
		Collection: name,
	})
	if err != nil {
		return fmt.Errorf("nietzsche DropCollection: %w", err)
	}

	if resp.Status == "error" {
		return fmt.Errorf("nietzsche DropCollection: %s", resp.Error)
	}

	return nil
}

// ListCollections returns metadata about all collections.
func (c *NietzscheClient) ListCollections(ctx context.Context) ([]CollectionInfo, error) {
	resp, err := c.stub.ListCollections(ctx, &pb.Empty{})
	if err != nil {
		return nil, fmt.Errorf("nietzsche ListCollections: %w", err)
	}

	infos := make([]CollectionInfo, len(resp.Collections))
	for i, col := range resp.Collections {
		infos[i] = CollectionInfo{
			Name:      col.Collection,
			Dim:       col.Dim,
			Metric:    col.Metric,
			NodeCount: col.NodeCount,
			EdgeCount: col.EdgeCount,
		}
	}

	return infos, nil
}
