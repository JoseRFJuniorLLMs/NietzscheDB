use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Metadata for a media blob associated with a graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaMeta {
    pub id: Uuid,
    pub node_id: Uuid,
    pub filename: String,
    pub media_type: MediaType,
    pub content_type: String,
    pub size_bytes: u64,
    pub created_at: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum MediaType {
    Image,
    Audio,
    Video,
    Document,
    Binary,
}

impl MediaMeta {
    pub fn storage_key(&self) -> String {
        format!("{}/{}", self.node_id, self.id)
    }

    pub fn meta_key(&self) -> String {
        format!("{}/{}.meta", self.node_id, self.id)
    }
}
