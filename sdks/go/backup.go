// Copyright (C) 2025-2026 Jose R F Junior <petweofc@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

package nietzsche

import (
	"context"
	"fmt"
	"time"

	pb "nietzsche-sdk/pb"
)

// ── Backup result types ───────────────────────────────────────────────────────

// BackupInfo describes a single database backup.
type BackupInfo struct {
	Label     string
	Path      string    // server-side filesystem path to the backup directory
	CreatedAt time.Time // wall-clock time when backup was created
	SizeBytes uint64
}

// ── Backup methods ────────────────────────────────────────────────────────────

// CreateBackup creates a RocksDB checkpoint backup with an optional label.
func (c *NietzscheClient) CreateBackup(ctx context.Context, label string) (BackupInfo, error) {
	resp, err := c.stub.CreateBackup(ctx, &pb.CreateBackupRequest{Label: label})
	if err != nil {
		return BackupInfo{}, fmt.Errorf("nietzsche CreateBackup: %w", err)
	}
	return BackupInfo{
		Label:     resp.Label,
		Path:      resp.Path,
		CreatedAt: time.Unix(int64(resp.CreatedAt), 0),
		SizeBytes: resp.SizeBytes,
	}, nil
}

// ListBackups returns all available backups on the server.
func (c *NietzscheClient) ListBackups(ctx context.Context) ([]BackupInfo, error) {
	resp, err := c.stub.ListBackups(ctx, &pb.Empty{})
	if err != nil {
		return nil, fmt.Errorf("nietzsche ListBackups: %w", err)
	}
	result := make([]BackupInfo, len(resp.Backups))
	for i, b := range resp.Backups {
		result[i] = BackupInfo{
			Label:     b.Label,
			Path:      b.Path,
			CreatedAt: time.Unix(int64(b.CreatedAt), 0),
			SizeBytes: b.SizeBytes,
		}
	}
	return result, nil
}

// RestoreBackup restores from backupPath to targetPath.
// backupPath: server-side path of the backup (from BackupInfo.Path).
// targetPath: server-side destination directory.
func (c *NietzscheClient) RestoreBackup(ctx context.Context, backupPath, targetPath string) error {
	resp, err := c.stub.RestoreBackup(ctx, &pb.RestoreBackupRequest{
		BackupPath: backupPath,
		TargetPath: targetPath,
	})
	if err != nil {
		return fmt.Errorf("nietzsche RestoreBackup: %w", err)
	}
	if resp.Status == "error" {
		return fmt.Errorf("nietzsche RestoreBackup: %s", resp.Error)
	}
	return nil
}
