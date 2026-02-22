// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// ── Schema types ────────────────────────────────────────────────────────────

// SchemaFieldType defines a field name and its expected type.
type SchemaFieldType struct {
	FieldName string // metadata field name
	FieldType string // "string" | "number" | "bool" | "array" | "object"
}

// SchemaResult holds a node-type schema definition retrieved from the server.
type SchemaResult struct {
	Found          bool
	NodeType       string
	RequiredFields []string
	FieldTypes     []SchemaFieldType
}

// ── Schema methods ──────────────────────────────────────────────────────────

// SetSchema registers a validation schema for a given node type in a collection.
// Subsequent InsertNode/MergeNode calls with this node_type will be validated.
func (c *NietzscheClient) SetSchema(ctx context.Context, nodeType string, requiredFields []string, fieldTypes []SchemaFieldType, collection string) error {
	pbFields := make([]*pb.SchemaFieldType, len(fieldTypes))
	for i, ft := range fieldTypes {
		pbFields[i] = &pb.SchemaFieldType{
			FieldName: ft.FieldName,
			FieldType: ft.FieldType,
		}
	}

	resp, err := c.stub.SetSchema(ctx, &pb.SetSchemaRequest{
		NodeType:       nodeType,
		RequiredFields: requiredFields,
		FieldTypes:     pbFields,
		Collection:     collection,
	})
	if err != nil {
		return fmt.Errorf("nietzsche SetSchema: %w", err)
	}
	if resp.Status == "error" {
		return fmt.Errorf("nietzsche SetSchema: %s", resp.Error)
	}
	return nil
}

// GetSchema retrieves the validation schema for a node type.
func (c *NietzscheClient) GetSchema(ctx context.Context, nodeType, collection string) (*SchemaResult, error) {
	resp, err := c.stub.GetSchema(ctx, &pb.GetSchemaRequest{
		NodeType:   nodeType,
		Collection: collection,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche GetSchema: %w", err)
	}
	fts := make([]SchemaFieldType, len(resp.FieldTypes))
	for i, ft := range resp.FieldTypes {
		fts[i] = SchemaFieldType{FieldName: ft.FieldName, FieldType: ft.FieldType}
	}
	return &SchemaResult{
		Found:          resp.Found,
		NodeType:       resp.NodeType,
		RequiredFields: resp.RequiredFields,
		FieldTypes:     fts,
	}, nil
}

// ── Index methods ───────────────────────────────────────────────────────────

// CreateIndex creates a secondary index on a metadata field within a collection.
// This accelerates NQL queries that filter on the indexed field.
func (c *NietzscheClient) CreateIndex(ctx context.Context, collection, field string) error {
	resp, err := c.stub.CreateIndex(ctx, &pb.CreateIndexRequest{
		Collection: collection,
		Field:      field,
	})
	if err != nil {
		return fmt.Errorf("nietzsche CreateIndex: %w", err)
	}
	if resp.Status == "error" {
		return fmt.Errorf("nietzsche CreateIndex: %s", resp.Error)
	}
	return nil
}

// DropIndex removes a secondary index on a metadata field.
func (c *NietzscheClient) DropIndex(ctx context.Context, collection, field string) error {
	resp, err := c.stub.DropIndex(ctx, &pb.DropIndexRequest{
		Collection: collection,
		Field:      field,
	})
	if err != nil {
		return fmt.Errorf("nietzsche DropIndex: %w", err)
	}
	if resp.Status == "error" {
		return fmt.Errorf("nietzsche DropIndex: %s", resp.Error)
	}
	return nil
}

// ListIndexes returns the names of all indexed fields in a collection.
func (c *NietzscheClient) ListIndexes(ctx context.Context, collection string) ([]string, error) {
	resp, err := c.stub.ListIndexes(ctx, &pb.ListIndexesRequest{
		Collection: collection,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche ListIndexes: %w", err)
	}
	return resp.Fields, nil
}
