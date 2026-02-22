// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// FtsResult is a single full-text search hit.
type FtsResult struct {
	NodeID string
	Score  float64
}

// FullTextSearch performs a BM25 inverted-index search over node content.
// limit: max results to return (0 → server default of 10).
func (c *NietzscheClient) FullTextSearch(ctx context.Context, query, collection string, limit uint32) ([]FtsResult, error) {
	resp, err := c.stub.FullTextSearch(ctx, &pb.FullTextSearchRequest{
		Query:      query,
		Collection: collection,
		Limit:      limit,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche FullTextSearch: %w", err)
	}
	results := make([]FtsResult, len(resp.Results))
	for i, r := range resp.Results {
		results[i] = FtsResult{NodeID: r.NodeId, Score: r.Score}
	}
	return results, nil
}

// HybridSearch combines full-text BM25 and vector KNN search with configurable weights.
// textWeight + vectorWeight should sum to 1.0 for normalized scoring.
func (c *NietzscheClient) HybridSearch(ctx context.Context, textQuery string, queryCoords []float64, k uint32, textWeight, vectorWeight float64, collection string) ([]KnnResult, error) {
	resp, err := c.stub.HybridSearch(ctx, &pb.HybridSearchRequest{
		TextQuery:    textQuery,
		QueryCoords:  queryCoords,
		K:            k,
		TextWeight:   textWeight,
		VectorWeight: vectorWeight,
		Collection:   collection,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche HybridSearch: %w", err)
	}
	results := make([]KnnResult, len(resp.Results))
	for i, r := range resp.Results {
		results[i] = KnnResult{ID: r.Id, Distance: r.Distance}
	}
	return results, nil
}
