// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// ── Graph Algorithm result types ─────────────────────────────────────────────

// NodeScore pairs a node UUID with a floating-point score.
type NodeScore struct {
	NodeID string
	Score  float64
}

// NodeCommunity pairs a node UUID with a community/component integer ID.
type NodeCommunity struct {
	NodeID      string
	CommunityID uint64
}

// AlgoScoreResult is returned by score-based algorithms (PageRank, Betweenness, etc.).
type AlgoScoreResult struct {
	Scores     []NodeScore
	DurationMs uint64
	Iterations uint32
}

// AlgoCommunityResult is returned by community/component algorithms (Louvain, WCC, SCC).
type AlgoCommunityResult struct {
	Assignments    []NodeCommunity
	CommunityCount uint64
	LargestSize    uint64
	Modularity     float64 // Louvain only
	DurationMs     uint64
	Iterations     uint32
}

// AStarResult is returned by the A* shortest-path algorithm.
type AStarResult struct {
	Found bool
	Path  []string
	Cost  float64
}

// TriangleResult is returned by the triangle count algorithm.
type TriangleResult struct {
	Count uint64
}

// SimilarityPair represents a pair of nodes with their similarity score.
type SimilarityPair struct {
	NodeA string
	NodeB string
	Score float64
}

// SimilarityResult is returned by Jaccard/Overlap similarity algorithms.
type SimilarityResult struct {
	Pairs []SimilarityPair
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func toScoreResult(resp *pb.AlgorithmScoreResponse) AlgoScoreResult {
	scores := make([]NodeScore, len(resp.Scores))
	for i, s := range resp.Scores {
		scores[i] = NodeScore{NodeID: s.NodeId, Score: s.Score}
	}
	return AlgoScoreResult{
		Scores:     scores,
		DurationMs: resp.DurationMs,
		Iterations: resp.Iterations,
	}
}

func toCommunityResult(resp *pb.AlgorithmCommunityResponse) AlgoCommunityResult {
	assignments := make([]NodeCommunity, len(resp.Assignments))
	for i, c := range resp.Assignments {
		assignments[i] = NodeCommunity{NodeID: c.NodeId, CommunityID: c.CommunityId}
	}
	return AlgoCommunityResult{
		Assignments:    assignments,
		CommunityCount: resp.CommunityCount,
		LargestSize:    resp.LargestSize,
		Modularity:     resp.Modularity,
		DurationMs:     resp.DurationMs,
		Iterations:     resp.Iterations,
	}
}

// ── Algorithm methods ─────────────────────────────────────────────────────────

// RunPageRank executes the PageRank algorithm (power iteration).
// damping: typically 0.85 | maxIterations: typically 20.
func (c *NietzscheClient) RunPageRank(ctx context.Context, collection string, damping float64, maxIterations uint32) (AlgoScoreResult, error) {
	if damping == 0 {
		damping = 0.85
	}
	if maxIterations == 0 {
		maxIterations = 20
	}
	resp, err := c.stub.RunPageRank(ctx, &pb.PageRankRequest{
		Collection:    collection,
		DampingFactor: damping,
		MaxIterations: maxIterations,
	})
	if err != nil {
		return AlgoScoreResult{}, fmt.Errorf("nietzsche RunPageRank: %w", err)
	}
	return toScoreResult(resp), nil
}

// RunLouvain executes the Louvain community detection algorithm.
func (c *NietzscheClient) RunLouvain(ctx context.Context, collection string, maxIterations uint32, resolution float64) (AlgoCommunityResult, error) {
	if maxIterations == 0 {
		maxIterations = 50
	}
	if resolution == 0 {
		resolution = 1.0
	}
	resp, err := c.stub.RunLouvain(ctx, &pb.LouvainRequest{
		Collection:    collection,
		MaxIterations: maxIterations,
		Resolution:    resolution,
	})
	if err != nil {
		return AlgoCommunityResult{}, fmt.Errorf("nietzsche RunLouvain: %w", err)
	}
	return toCommunityResult(resp), nil
}

// RunLabelProp executes the Label Propagation community detection algorithm.
func (c *NietzscheClient) RunLabelProp(ctx context.Context, collection string, maxIterations uint32) (AlgoCommunityResult, error) {
	if maxIterations == 0 {
		maxIterations = 10
	}
	resp, err := c.stub.RunLabelProp(ctx, &pb.LabelPropRequest{
		Collection:    collection,
		MaxIterations: maxIterations,
	})
	if err != nil {
		return AlgoCommunityResult{}, fmt.Errorf("nietzsche RunLabelProp: %w", err)
	}
	return toCommunityResult(resp), nil
}

// RunBetweenness executes Brandes betweenness centrality.
// sampleSize: 0 = exact (all sources), >0 = approximate with k random sources.
func (c *NietzscheClient) RunBetweenness(ctx context.Context, collection string, sampleSize uint32) (AlgoScoreResult, error) {
	resp, err := c.stub.RunBetweenness(ctx, &pb.BetweennessRequest{
		Collection: collection,
		SampleSize: sampleSize,
	})
	if err != nil {
		return AlgoScoreResult{}, fmt.Errorf("nietzsche RunBetweenness: %w", err)
	}
	return toScoreResult(resp), nil
}

// RunCloseness executes closeness centrality (BFS from each node).
func (c *NietzscheClient) RunCloseness(ctx context.Context, collection string) (AlgoScoreResult, error) {
	resp, err := c.stub.RunCloseness(ctx, &pb.ClosenessRequest{Collection: collection})
	if err != nil {
		return AlgoScoreResult{}, fmt.Errorf("nietzsche RunCloseness: %w", err)
	}
	return toScoreResult(resp), nil
}

// RunDegreeCentrality computes degree centrality.
// direction: "in" | "out" | "both" (default "out").
func (c *NietzscheClient) RunDegreeCentrality(ctx context.Context, collection, direction string) (AlgoScoreResult, error) {
	if direction == "" {
		direction = "out"
	}
	resp, err := c.stub.RunDegreeCentrality(ctx, &pb.DegreeCentralityRequest{
		Collection: collection,
		Direction:  direction,
	})
	if err != nil {
		return AlgoScoreResult{}, fmt.Errorf("nietzsche RunDegreeCentrality: %w", err)
	}
	return toScoreResult(resp), nil
}

// RunWCC finds Weakly Connected Components using Union-Find.
func (c *NietzscheClient) RunWCC(ctx context.Context, collection string) (AlgoCommunityResult, error) {
	resp, err := c.stub.RunWCC(ctx, &pb.WccRequest{Collection: collection})
	if err != nil {
		return AlgoCommunityResult{}, fmt.Errorf("nietzsche RunWCC: %w", err)
	}
	return toCommunityResult(resp), nil
}

// RunSCC finds Strongly Connected Components using Tarjan's algorithm.
func (c *NietzscheClient) RunSCC(ctx context.Context, collection string) (AlgoCommunityResult, error) {
	resp, err := c.stub.RunSCC(ctx, &pb.SccRequest{Collection: collection})
	if err != nil {
		return AlgoCommunityResult{}, fmt.Errorf("nietzsche RunSCC: %w", err)
	}
	return toCommunityResult(resp), nil
}

// RunAStar finds the shortest hyperbolic path between two nodes.
func (c *NietzscheClient) RunAStar(ctx context.Context, collection, startID, goalID string) (AStarResult, error) {
	resp, err := c.stub.RunAStar(ctx, &pb.AStarRequest{
		Collection:  collection,
		StartNodeId: startID,
		GoalNodeId:  goalID,
	})
	if err != nil {
		return AStarResult{}, fmt.Errorf("nietzsche RunAStar: %w", err)
	}
	return AStarResult{
		Found: resp.Found,
		Path:  resp.Path,
		Cost:  resp.Cost,
	}, nil
}

// RunTriangleCount counts triangles in the graph.
func (c *NietzscheClient) RunTriangleCount(ctx context.Context, collection string) (TriangleResult, error) {
	resp, err := c.stub.RunTriangleCount(ctx, &pb.TriangleCountRequest{Collection: collection})
	if err != nil {
		return TriangleResult{}, fmt.Errorf("nietzsche RunTriangleCount: %w", err)
	}
	return TriangleResult{Count: resp.Count}, nil
}

// RunJaccardSimilarity finds top-k most similar node pairs by Jaccard coefficient.
func (c *NietzscheClient) RunJaccardSimilarity(ctx context.Context, collection string, topK uint32, threshold float64) (SimilarityResult, error) {
	if topK == 0 {
		topK = 10
	}
	resp, err := c.stub.RunJaccardSimilarity(ctx, &pb.JaccardRequest{
		Collection: collection,
		TopK:       topK,
		Threshold:  threshold,
	})
	if err != nil {
		return SimilarityResult{}, fmt.Errorf("nietzsche RunJaccardSimilarity: %w", err)
	}
	pairs := make([]SimilarityPair, len(resp.Pairs))
	for i, p := range resp.Pairs {
		pairs[i] = SimilarityPair{NodeA: p.NodeA, NodeB: p.NodeB, Score: p.Score}
	}
	return SimilarityResult{Pairs: pairs}, nil
}
