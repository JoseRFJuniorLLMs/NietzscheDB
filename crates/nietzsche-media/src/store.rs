use opendal::services::{Fs, Memory};
use opendal::Operator;
use uuid::Uuid;

use crate::error::MediaError;
use crate::model::MediaMeta;

/// Backend-agnostic media store powered by Apache OpenDAL.
pub struct MediaStore {
    op: Operator,
}

impl MediaStore {
    /// Create a MediaStore backed by the local filesystem.
    pub fn new_fs(root_path: &str) -> Result<Self, MediaError> {
        let mut builder = Fs::default();
        builder.root(root_path);
        let op = Operator::new(builder)?.finish();
        Ok(Self { op })
    }

    /// Create a MediaStore backed by in-memory storage (for tests).
    pub fn new_memory() -> Result<Self, MediaError> {
        let op = Operator::new(Memory::default())?.finish();
        Ok(Self { op })
    }

    /// Create a MediaStore from any OpenDAL Operator.
    pub fn from_operator(op: Operator) -> Self {
        Self { op }
    }

    /// Store a blob and its metadata.
    pub async fn put(&self, meta: &MediaMeta, data: &[u8]) -> Result<(), MediaError> {
        // Write the blob data
        self.op.write(&meta.storage_key(), data.to_vec()).await?;
        // Write the metadata as JSON
        let meta_json = serde_json::to_vec(meta)
            .map_err(|e| MediaError::Metadata(e.to_string()))?;
        self.op.write(&meta.meta_key(), meta_json).await?;
        Ok(())
    }

    /// Retrieve a blob's data.
    pub async fn get(&self, meta: &MediaMeta) -> Result<Vec<u8>, MediaError> {
        let data = self.op.read(&meta.storage_key()).await?;
        Ok(data.to_vec())
    }

    /// Retrieve metadata for a media item.
    pub async fn get_meta(
        &self,
        node_id: &Uuid,
        media_id: &Uuid,
        media_type: crate::model::MediaType,
    ) -> Result<Option<MediaMeta>, MediaError> {
        let tmp = MediaMeta {
            id: *media_id,
            node_id: *node_id,
            filename: String::new(),
            media_type,
            content_type: String::new(),
            size_bytes: 0,
            created_at: 0,
        };
        let key = tmp.meta_key();
        match self.op.read(&key).await {
            Ok(data) => {
                let meta: MediaMeta = serde_json::from_slice(&data.to_vec())
                    .map_err(|e| MediaError::Metadata(e.to_string()))?;
                Ok(Some(meta))
            }
            Err(e) if e.kind() == opendal::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Delete a blob and its metadata.
    pub async fn delete(&self, meta: &MediaMeta) -> Result<(), MediaError> {
        self.op.delete(&meta.storage_key()).await?;
        self.op.delete(&meta.meta_key()).await?;
        Ok(())
    }

    /// List all media for a given node.
    pub async fn list_for_node(&self, node_id: &Uuid) -> Result<Vec<MediaMeta>, MediaError> {
        let prefix = format!("{}/", node_id);
        let entries = self.op.list(&prefix).await?;
        let mut result = Vec::new();

        for entry in entries {
            let path = entry.path();
            if path.ends_with(".meta") {
                match self.op.read(path).await {
                    Ok(data) => {
                        if let Ok(meta) = serde_json::from_slice::<MediaMeta>(&data.to_vec()) {
                            result.push(meta);
                        }
                    }
                    Err(_) => continue,
                }
            }
        }
        Ok(result)
    }

    /// Check if a blob exists.
    pub async fn exists(&self, meta: &MediaMeta) -> Result<bool, MediaError> {
        Ok(self.op.is_exist(&meta.storage_key()).await?)
    }
}
