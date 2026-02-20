// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// GetStats returns global database statistics.
func (c *NietzscheClient) GetStats(ctx context.Context) (Stats, error) {
	resp, err := c.stub.GetStats(ctx, &pb.Empty{})
	if err != nil {
		return Stats{}, fmt.Errorf("nietzsche GetStats: %w", err)
	}

	return Stats{
		NodeCount:    resp.NodeCount,
		EdgeCount:    resp.EdgeCount,
		Version:      resp.Version,
		SensoryCount: resp.SensoryCount,
	}, nil
}

// HealthCheck verifies that NietzscheDB is reachable and healthy.
func (c *NietzscheClient) HealthCheck(ctx context.Context) error {
	resp, err := c.stub.HealthCheck(ctx, &pb.Empty{})
	if err != nil {
		return fmt.Errorf("nietzsche HealthCheck: %w", err)
	}

	if resp.Status == "error" {
		return fmt.Errorf("nietzsche HealthCheck: %s", resp.Error)
	}

	return nil
}
