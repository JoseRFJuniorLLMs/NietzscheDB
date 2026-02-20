// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"
	"io"
	"time"

	"google.golang.org/grpc"

	pb "nietzsche-sdk/pb"
)

// CDCEvent represents a single Change Data Capture event from NietzscheDB.
type CDCEvent struct {
	LSN        uint64    // logical sequence number (monotonically increasing)
	EventType  string    // INSERT_NODE | UPDATE_NODE | DELETE_NODE | INSERT_EDGE | DELETE_EDGE | BATCH_INSERT_NODES | BATCH_INSERT_EDGES | SLEEP_CYCLE | ZARATUSTRA
	Timestamp  time.Time // wall-clock time when the mutation occurred
	EntityID   string    // UUID of the mutated node or edge
	Collection string    // collection where the mutation occurred
	BatchCount uint32    // for BATCH_INSERT_* events: number of items
}

// CDCSubscription is a live server-streaming connection for CDC events.
type CDCSubscription struct {
	stream grpc.ServerStreamingClient[pb.CdcEvent]
	cancel context.CancelFunc
}

// SubscribeCDC opens a server-streaming CDC subscription.
// fromLSN: start from this LSN (0 = from current moment).
// collection: filter to collection; "" = all.
//
// Cancel the provided context to stop the stream.
//
// Usage:
//
//	ctx, cancel := context.WithCancel(context.Background())
//	defer cancel()
//	sub, err := client.SubscribeCDC(ctx, "", 0)
//	for {
//	    event, err := sub.Recv()
//	    if err == io.EOF { break }
//	    // handle event
//	}
func (c *NietzscheClient) SubscribeCDC(ctx context.Context, collection string, fromLSN uint64) (*CDCSubscription, error) {
	ctx, cancel := context.WithCancel(ctx)
	stream, err := c.stub.SubscribeCDC(ctx, &pb.CdcRequest{
		Collection: collection,
		FromLsn:    fromLSN,
	})
	if err != nil {
		cancel()
		return nil, fmt.Errorf("nietzsche SubscribeCDC: %w", err)
	}
	return &CDCSubscription{stream: stream, cancel: cancel}, nil
}

// Recv blocks until the next CDC event arrives.
// Returns (event, nil) on success, (CDCEvent{}, io.EOF) when the stream ends.
func (s *CDCSubscription) Recv() (CDCEvent, error) {
	msg, err := s.stream.Recv()
	if err != nil {
		if err == io.EOF {
			return CDCEvent{}, io.EOF
		}
		return CDCEvent{}, fmt.Errorf("nietzsche CDC recv: %w", err)
	}
	return CDCEvent{
		LSN:        msg.Lsn,
		EventType:  msg.EventType,
		Timestamp:  time.UnixMilli(int64(msg.TimestampMs)),
		EntityID:   msg.EntityId,
		Collection: msg.Collection,
		BatchCount: msg.BatchCount,
	}, nil
}

// Close cancels the CDC subscription.
func (s *CDCSubscription) Close() {
	s.cancel()
}
