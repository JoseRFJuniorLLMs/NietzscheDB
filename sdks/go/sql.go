// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"encoding/json"
	"fmt"

	pb "nietzsche-sdk/pb"
)

// ── Swartz SQL Layer result types ────────────────────────────────────────────

// SqlResultSet contains the result of a SQL query.
type SqlResultSet struct {
	Columns      []SqlColumn              // column definitions
	Rows         []map[string]interface{} // result rows as column→value maps
	AffectedRows uint64                   // number of affected rows (for non-SELECT)
}

// SqlColumn describes a column in a SQL result set.
type SqlColumn struct {
	Name string // column name
	Type string // column type ("TEXT", "INTEGER", "FLOAT", "BOOLEAN", etc.)
}

// SqlExecResult contains the result of a SQL exec (DDL/DML) statement.
type SqlExecResult struct {
	AffectedRows uint64 // number of rows affected
	Success      bool   // true if statement succeeded
	Message      string // optional message
}

// ── Swartz SQL Layer methods ─────────────────────────────────────────────────

// SqlQuery executes a SQL query (SELECT) and returns rows.
//
// Each row is returned as a map[string]interface{} keyed by column name.
// Values are JSON-decoded from the wire format.
//
// Example:
//
//	rows, err := client.SqlQuery(ctx, "SELECT * FROM moods WHERE score > 0.5", "session_data")
func (c *NietzscheClient) SqlQuery(ctx context.Context, sql, collection string) (*SqlResultSet, error) {
	resp, err := c.stub.SqlQuery(ctx, &pb.SqlRequest{
		Sql:        sql,
		Collection: collection,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche SqlQuery: %w", err)
	}

	result := &SqlResultSet{
		AffectedRows: resp.AffectedRows,
	}

	// Convert column definitions
	result.Columns = make([]SqlColumn, len(resp.Columns))
	for i, col := range resp.Columns {
		result.Columns[i] = SqlColumn{
			Name: col.Name,
			Type: col.Type,
		}
	}

	// Convert rows: each SqlRow.Values is a slice of JSON-encoded bytes
	result.Rows = make([]map[string]interface{}, 0, len(resp.Rows))
	for _, row := range resp.Rows {
		m := make(map[string]interface{}, len(result.Columns))
		for i, valBytes := range row.Values {
			if i < len(result.Columns) {
				var v interface{}
				if err := json.Unmarshal(valBytes, &v); err != nil {
					// If JSON decode fails, store as raw string
					m[result.Columns[i].Name] = string(valBytes)
				} else {
					m[result.Columns[i].Name] = v
				}
			}
		}
		result.Rows = append(result.Rows, m)
	}

	return result, nil
}

// SqlExec executes a SQL DDL/DML statement (CREATE TABLE, INSERT, UPDATE, DELETE, DROP TABLE).
//
// Returns the number of affected rows and success status.
//
// Example:
//
//	result, err := client.SqlExec(ctx, "INSERT INTO moods VALUES (1, 'calm', 0.85)", "session_data")
//	fmt.Printf("affected: %d\n", result.AffectedRows)
func (c *NietzscheClient) SqlExec(ctx context.Context, sql, collection string) (*SqlExecResult, error) {
	resp, err := c.stub.SqlExec(ctx, &pb.SqlRequest{
		Sql:        sql,
		Collection: collection,
	})
	if err != nil {
		return nil, fmt.Errorf("nietzsche SqlExec: %w", err)
	}

	return &SqlExecResult{
		AffectedRows: resp.AffectedRows,
		Success:      resp.Success,
		Message:      resp.Message,
	}, nil
}
