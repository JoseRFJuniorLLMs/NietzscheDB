//! Backup and restore using RocksDB checkpoints.
//!
//! A checkpoint is a hard-link-based snapshot — creating one is O(1) and
//! does not block writes.  Restoring copies the checkpoint into a fresh path.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use rocksdb::checkpoint::Checkpoint;

use crate::error::GraphError;
use crate::storage::GraphStorage;

/// Metadata about a backup.
#[derive(Debug, Clone)]
pub struct BackupInfo {
    pub label: String,
    pub path: PathBuf,
    pub created_at: u64, // Unix seconds
    pub size_bytes: u64,
}

/// Manages backup/restore lifecycle for a single database instance.
pub struct BackupManager {
    backup_dir: PathBuf,
}

impl BackupManager {
    /// Create a new manager that stores backups under `backup_dir`.
    /// Creates the directory if it doesn't exist.
    pub fn new(backup_dir: impl Into<PathBuf>) -> Result<Self, GraphError> {
        let dir = backup_dir.into();
        fs::create_dir_all(&dir)
            .map_err(|e| GraphError::Storage(format!("cannot create backup dir: {e}")))?;
        Ok(Self { backup_dir: dir })
    }

    /// Create a checkpoint snapshot of the database.
    ///
    /// `label` is a user-chosen name (e.g. "pre-migration", "daily-2026-02-19").
    /// Returns the [`BackupInfo`] of the created backup.
    pub fn create_backup(
        &self,
        storage: &GraphStorage,
        label: &str,
    ) -> Result<BackupInfo, GraphError> {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let safe_label = label
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect::<String>();

        let dir_name = format!("{}-{}", timestamp, safe_label);
        let backup_path = self.backup_dir.join(&dir_name);

        let cp = Checkpoint::new(storage.db_handle())
            .map_err(|e| GraphError::Storage(format!("checkpoint create error: {e}")))?;

        cp.create_checkpoint(&backup_path)
            .map_err(|e| GraphError::Storage(format!("checkpoint write error: {e}")))?;

        let size = dir_size(&backup_path);

        Ok(BackupInfo {
            label: safe_label,
            path: backup_path,
            created_at: timestamp,
            size_bytes: size,
        })
    }

    /// List all available backups, sorted newest-first.
    pub fn list_backups(&self) -> Result<Vec<BackupInfo>, GraphError> {
        let mut backups = Vec::new();

        let entries = fs::read_dir(&self.backup_dir)
            .map_err(|e| GraphError::Storage(format!("cannot read backup dir: {e}")))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| GraphError::Storage(format!("dir entry error: {e}")))?;
            let path = entry.path();
            if !path.is_dir() { continue; }

            let name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            // Parse timestamp from directory name: "1708300000-label"
            let (ts_str, label) = name.split_once('-').unwrap_or(("0", &name));
            let created_at = ts_str.parse::<u64>().unwrap_or(0);
            let size = dir_size(&path);

            backups.push(BackupInfo {
                label: label.to_string(),
                path,
                created_at,
                size_bytes: size,
            });
        }

        // Sort newest-first
        backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(backups)
    }

    /// Restore a backup into `target_path` by copying the checkpoint directory.
    ///
    /// Returns the path that was restored to (same as `target_path`).
    /// The caller should open a new `GraphStorage` pointing at `target_path`.
    pub fn restore_backup(
        &self,
        backup_path: &Path,
        target_path: &Path,
    ) -> Result<PathBuf, GraphError> {
        if !backup_path.is_dir() {
            return Err(GraphError::Storage(format!(
                "backup not found: {}",
                backup_path.display()
            )));
        }

        // Copy the entire backup directory to the target
        copy_dir_all(backup_path, target_path)
            .map_err(|e| GraphError::Storage(format!("restore copy error: {e}")))?;

        Ok(target_path.to_path_buf())
    }

    /// Delete a backup directory.
    pub fn delete_backup(&self, backup_path: &Path) -> Result<(), GraphError> {
        if !backup_path.starts_with(&self.backup_dir) {
            return Err(GraphError::Storage("backup path outside backup directory".into()));
        }
        fs::remove_dir_all(backup_path)
            .map_err(|e| GraphError::Storage(format!("delete backup error: {e}")))?;
        Ok(())
    }
}

/// Recursively compute directory size in bytes.
fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            } else if p.is_dir() {
                total += dir_size(&p);
            }
        }
    }
    total
}

/// Recursively copy a directory.
fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}
