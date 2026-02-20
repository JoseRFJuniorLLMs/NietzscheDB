// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// Query executes an NQL (Nietzsche Query Language) query.
// Params are automatically type-detected: string, float64, int/int64, []float64.
func (c *NietzscheClient) Query(ctx context.Context, nql string, params map[string]interface{}, collection string) (*QueryResult, error) {
	pbParams, err := convertParams(params)
	if err != nil {
		return nil, fmt.Errorf("nietzsche Query: %w", err)
	}

	resp, err := c.stub.Query(ctx, &pb.QueryRequest{
		Nql:        nql,
		Params:     pbParams,
		Collection: collection,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche Query: %w", err)
	}

	result := &QueryResult{
		PathIDs: resp.PathIds,
		Explain: resp.Explain,
		Error:   resp.Error,
	}

	for _, n := range resp.Nodes {
		result.Nodes = append(result.Nodes, nodeResponseToResult(n))
	}

	for _, pair := range resp.NodePairs {
		np := NodePairResult{}
		if pair.From != nil {
			np.From = nodeResponseToResult(pair.From)
		}
		if pair.To != nil {
			np.To = nodeResponseToResult(pair.To)
		}
		result.NodePairs = append(result.NodePairs, np)
	}

	for _, row := range resp.ScalarRows {
		m := make(map[string]interface{})
		for _, entry := range row.Entries {
			if entry.IsNull {
				m[entry.Column] = nil
				continue
			}
			switch v := entry.Value.(type) {
			case *pb.ScalarEntry_FloatVal:
				m[entry.Column] = v.FloatVal
			case *pb.ScalarEntry_IntVal:
				m[entry.Column] = v.IntVal
			case *pb.ScalarEntry_StringVal:
				m[entry.Column] = v.StringVal
			case *pb.ScalarEntry_BoolVal:
				m[entry.Column] = v.BoolVal
			}
		}
		result.ScalarRows = append(result.ScalarRows, m)
	}

	return result, nil
}

// KnnSearch performs a hyperbolic nearest-neighbour search.
func (c *NietzscheClient) KnnSearch(ctx context.Context, queryCoords []float64, k uint32, collection string) ([]KnnResult, error) {
	resp, err := c.stub.KnnSearch(ctx, &pb.KnnRequest{
		QueryCoords: queryCoords,
		K:           k,
		Collection:  collection,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche KnnSearch: %w", err)
	}

	results := make([]KnnResult, len(resp.Results))
	for i, r := range resp.Results {
		results[i] = KnnResult{
			ID:       r.Id,
			Distance: r.Distance,
		}
	}

	return results, nil
}

// convertParams transforms Go map values into protobuf QueryParamValue messages.
func convertParams(params map[string]interface{}) (map[string]*pb.QueryParamValue, error) {
	if len(params) == 0 {
		return nil, nil
	}

	result := make(map[string]*pb.QueryParamValue, len(params))

	for key, val := range params {
		pv := &pb.QueryParamValue{}

		switch v := val.(type) {
		case string:
			pv.Value = &pb.QueryParamValue_StringVal{StringVal: v}
		case float64:
			pv.Value = &pb.QueryParamValue_FloatVal{FloatVal: v}
		case float32:
			pv.Value = &pb.QueryParamValue_FloatVal{FloatVal: float64(v)}
		case int:
			pv.Value = &pb.QueryParamValue_IntVal{IntVal: int64(v)}
		case int64:
			pv.Value = &pb.QueryParamValue_IntVal{IntVal: v}
		case int32:
			pv.Value = &pb.QueryParamValue_IntVal{IntVal: int64(v)}
		case []float64:
			pv.Value = &pb.QueryParamValue_VecVal{VecVal: &pb.VectorParam{Coords: v}}
		default:
			return nil, fmt.Errorf("unsupported param type for key %q: %T", key, val)
		}

		result[key] = pv
	}

	return result, nil
}
