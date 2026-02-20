// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// InsertSensory attaches compressed sensory data to an existing node.
func (c *NietzscheClient) InsertSensory(ctx context.Context, opts InsertSensoryOpts) error {
	resp, err := c.stub.InsertSensory(ctx, &pb.InsertSensoryRequest{
		NodeId:         opts.NodeID,
		Modality:       opts.Modality,
		Latent:         opts.Latent,
		OriginalShape:  opts.OriginalShape,
		OriginalBytes:  opts.OriginalBytes,
		EncoderVersion: opts.EncoderVersion,
		ModalityMeta:   opts.ModalityMeta,
		Collection:     opts.Collection,
	})
	if err != nil {
		return fmt.Errorf("nietzsche InsertSensory: %w", err)
	}

	if resp.Status == "error" {
		return fmt.Errorf("nietzsche InsertSensory: %s", resp.Error)
	}

	return nil
}

// GetSensory retrieves sensory metadata for a node (without the full latent vector).
func (c *NietzscheClient) GetSensory(ctx context.Context, nodeID, collection string) (*SensoryResult, error) {
	resp, err := c.stub.GetSensory(ctx, &pb.NodeIdRequest{
		Id:         nodeID,
		Collection: collection,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche GetSensory: %w", err)
	}

	return &SensoryResult{
		Found:                 resp.Found,
		NodeID:                resp.NodeId,
		Modality:              resp.Modality,
		Dim:                   resp.Dim,
		QuantLevel:            resp.QuantLevel,
		ReconstructionQuality: resp.ReconstructionQuality,
		CompressionRatio:      resp.CompressionRatio,
		EncoderVersion:        resp.EncoderVersion,
		ByteSize:              resp.ByteSize,
	}, nil
}

// Reconstruct retrieves the sensory latent vector for decoder input.
// Quality can be "full", "degraded", or "best_available".
func (c *NietzscheClient) Reconstruct(ctx context.Context, nodeID, quality string) (*ReconstructResult, error) {
	resp, err := c.stub.Reconstruct(ctx, &pb.ReconstructRequest{
		NodeId:  nodeID,
		Quality: quality,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche Reconstruct: %w", err)
	}

	return &ReconstructResult{
		Found:         resp.Found,
		NodeID:        resp.NodeId,
		Latent:        resp.Latent,
		Modality:      resp.Modality,
		Quality:       resp.Quality,
		OriginalShape: resp.OriginalShape,
	}, nil
}

// DegradeSensory triggers progressive quantisation degradation on a node's sensory data.
func (c *NietzscheClient) DegradeSensory(ctx context.Context, nodeID, collection string) error {
	resp, err := c.stub.DegradeSensory(ctx, &pb.NodeIdRequest{
		Id:         nodeID,
		Collection: collection,
	})
	if err != nil {
		return fmt.Errorf("nietzsche DegradeSensory: %w", err)
	}

	if resp.Status == "error" {
		return fmt.Errorf("nietzsche DegradeSensory: %s", resp.Error)
	}

	return nil
}
