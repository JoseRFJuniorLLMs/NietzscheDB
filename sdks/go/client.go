// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"fmt"

	pb "nietzsche-sdk/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// NietzscheClient wraps a gRPC connection to NietzscheDB.
type NietzscheClient struct {
	conn *grpc.ClientConn
	stub pb.NietzscheDBClient
}

// Connect establishes a gRPC connection to NietzscheDB at the given address.
// Custom dial options (TLS, interceptors, etc.) can be provided.
func Connect(addr string, opts ...grpc.DialOption) (*NietzscheClient, error) {
	conn, err := grpc.NewClient(addr, opts...)
	if err != nil {
		return nil, fmt.Errorf("nietzsche: failed to connect to %s: %w", addr, err)
	}
	return &NietzscheClient{
		conn: conn,
		stub: pb.NewNietzscheDBClient(conn),
	}, nil
}

// ConnectInsecure establishes an insecure gRPC connection (no TLS).
// Shorthand for development / docker-compose environments.
func ConnectInsecure(addr string) (*NietzscheClient, error) {
	return Connect(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
}

// Close releases the underlying gRPC connection.
func (c *NietzscheClient) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}
