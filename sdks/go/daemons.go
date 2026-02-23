// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// CreateDaemon registers a new Wiederkehr Daemon to execute NQL periodically.
func (c *NietzscheClient) CreateDaemon(ctx context.Context, opts CreateDaemonOpts) error {
	_, err := c.stub.CreateDaemon(ctx, &pb.CreateDaemonRequest{
		Collection:   opts.Collection,
		Label:        opts.Label,
		Nql:          opts.NQL,
		IntervalSecs: opts.IntervalSecs,
	})
	if err != nil {
		return fmt.Errorf("nietzsche CreateDaemon: %w", err)
	}
	return nil
}

// ListDaemons returns all daemons registered in a collection.
func (c *NietzscheClient) ListDaemons(ctx context.Context, collection string) ([]DaemonInfo, error) {
	resp, err := c.stub.ListDaemons(ctx, &pb.ListDaemonsRequest{
		Collection: collection,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche ListDaemons: %w", err)
	}

	var daemons []DaemonInfo
	for _, d := range resp.Daemons {
		daemons = append(daemons, DaemonInfo{
			Label:        d.Label,
			NQL:          d.Nql,
			IntervalSecs: d.IntervalSecs,
			LastRunAt:    d.LastRunAt,
			RunCount:     d.RunCount,
		})
	}
	return daemons, nil
}

// DropDaemon removes a registered daemon.
func (c *NietzscheClient) DropDaemon(ctx context.Context, collection, label string) error {
	_, err := c.stub.DropDaemon(ctx, &pb.DropDaemonRequest{
		Collection: collection,
		Label:      label,
	})
	if err != nil {
		return fmt.Errorf("nietzsche DropDaemon: %w", err)
	}
	return nil
}
