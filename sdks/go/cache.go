// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// ── Cache methods ───────────────────────────────────────────────────────────
// Key-value cache scoped to a collection. Values are opaque bytes with optional TTL.

// CacheSet stores a value under key in the given collection.
// ttlSecs: time-to-live in seconds (0 = no expiry).
func (c *NietzscheClient) CacheSet(ctx context.Context, collection, key string, value []byte, ttlSecs uint64) error {
	resp, err := c.stub.CacheSet(ctx, &pb.CacheSetRequest{
		Collection: collection,
		Key:        key,
		Value:      value,
		TtlSecs:    ttlSecs,
	})
	if err != nil {
		return fmt.Errorf("nietzsche CacheSet: %w", err)
	}
	if resp.Status == "error" {
		return fmt.Errorf("nietzsche CacheSet: %s", resp.Error)
	}
	return nil
}

// CacheGet retrieves a cached value. Returns (value, found, err).
func (c *NietzscheClient) CacheGet(ctx context.Context, collection, key string) ([]byte, bool, error) {
	resp, err := c.stub.CacheGet(ctx, &pb.CacheGetRequest{
		Collection: collection,
		Key:        key,
	})
	if err != nil {
		return nil, false, fmt.Errorf("nietzsche CacheGet: %w", err)
	}
	return resp.Value, resp.Found, nil
}

// CacheDel removes a cached value by key.
func (c *NietzscheClient) CacheDel(ctx context.Context, collection, key string) error {
	resp, err := c.stub.CacheDel(ctx, &pb.CacheDelRequest{
		Collection: collection,
		Key:        key,
	})
	if err != nil {
		return fmt.Errorf("nietzsche CacheDel: %w", err)
	}
	if resp.Status == "error" {
		return fmt.Errorf("nietzsche CacheDel: %s", resp.Error)
	}
	return nil
}

// ── List methods ────────────────────────────────────────────────────────────
// Node-scoped ordered lists. Each list is identified by (nodeID, listName) within a collection.

// ListRPush appends a value to the right end of a node's named list.
// Returns the new length of the list after the push.
func (c *NietzscheClient) ListRPush(ctx context.Context, nodeID, listName string, value []byte, collection string) (uint64, error) {
	resp, err := c.stub.ListRPush(ctx, &pb.ListPushRequest{
		NodeId:     nodeID,
		ListName:   listName,
		Value:      value,
		Collection: collection,
	})
	if err != nil {
		return 0, fmt.Errorf("nietzsche ListRPush: %w", err)
	}
	return resp.NewLength, nil
}

// ListLRange returns a slice of values from a node's named list.
// start: 0-based start index.
// stop: 0-based inclusive stop index (-1 = to end).
func (c *NietzscheClient) ListLRange(ctx context.Context, nodeID, listName string, start uint64, stop int64, collection string) ([][]byte, error) {
	resp, err := c.stub.ListLRange(ctx, &pb.ListRangeRequest{
		NodeId:     nodeID,
		ListName:   listName,
		Start:      start,
		Stop:       stop,
		Collection: collection,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche ListLRange: %w", err)
	}
	return resp.Values, nil
}

// ListLen returns the number of elements in a node's named list.
func (c *NietzscheClient) ListLen(ctx context.Context, nodeID, listName, collection string) (uint64, error) {
	resp, err := c.stub.ListLen(ctx, &pb.ListLenRequest{
		NodeId:     nodeID,
		ListName:   listName,
		Collection: collection,
	})
	if err != nil {
		return 0, fmt.Errorf("nietzsche ListLen: %w", err)
	}
	return resp.Length, nil
}
