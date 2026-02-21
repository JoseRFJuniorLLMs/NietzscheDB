//! Media/blob store for NietzscheDB.
//!
//! Stores media files (images, audio, documents, etc.) associated
//! with graph nodes, using Apache OpenDAL for backend-agnostic storage.
//!
//! Supported backends: local filesystem, S3, GCS, Azure, and any
//! other storage service supported by OpenDAL.

pub mod error;
pub mod model;
pub mod store;

pub use error::MediaError;
pub use model::{MediaMeta, MediaType};
pub use store::MediaStore;

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_meta(node_id: Uuid, media_type: MediaType) -> MediaMeta {
        MediaMeta {
            id: Uuid::new_v4(),
            node_id,
            filename: "test.png".into(),
            media_type,
            content_type: "image/png".into(),
            size_bytes: 0,
            created_at: 1234567890,
        }
    }

    #[tokio::test]
    async fn test_put_and_get() {
        let store = MediaStore::new_memory().unwrap();
        let node_id = Uuid::new_v4();
        let meta = make_meta(node_id, MediaType::Image);
        let data = b"hello world blob data";

        store.put(&meta, data).await.unwrap();
        let retrieved = store.get(&meta).await.unwrap();
        assert_eq!(retrieved, data);
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let store = MediaStore::new_memory().unwrap();
        let meta = make_meta(Uuid::new_v4(), MediaType::Audio);
        let result = store.get(&meta).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delete() {
        let store = MediaStore::new_memory().unwrap();
        let meta = make_meta(Uuid::new_v4(), MediaType::Document);
        store.put(&meta, b"data").await.unwrap();
        assert!(store.exists(&meta).await.unwrap());

        store.delete(&meta).await.unwrap();
        assert!(!store.exists(&meta).await.unwrap());
    }

    #[tokio::test]
    async fn test_exists() {
        let store = MediaStore::new_memory().unwrap();
        let meta = make_meta(Uuid::new_v4(), MediaType::Binary);
        assert!(!store.exists(&meta).await.unwrap());

        store.put(&meta, b"data").await.unwrap();
        assert!(store.exists(&meta).await.unwrap());
    }

    #[tokio::test]
    async fn test_list_for_node() {
        let store = MediaStore::new_memory().unwrap();
        let node_id = Uuid::new_v4();

        let m1 = make_meta(node_id, MediaType::Image);
        let m2 = make_meta(node_id, MediaType::Audio);
        let m3 = make_meta(node_id, MediaType::Video);

        store.put(&m1, b"img").await.unwrap();
        store.put(&m2, b"aud").await.unwrap();
        store.put(&m3, b"vid").await.unwrap();

        let list = store.list_for_node(&node_id).await.unwrap();
        assert_eq!(list.len(), 3);
    }

    #[tokio::test]
    async fn test_different_nodes() {
        let store = MediaStore::new_memory().unwrap();
        let n1 = Uuid::new_v4();
        let n2 = Uuid::new_v4();

        let m1 = make_meta(n1, MediaType::Image);
        let m2 = make_meta(n2, MediaType::Image);

        store.put(&m1, b"node1").await.unwrap();
        store.put(&m2, b"node2").await.unwrap();

        let list1 = store.list_for_node(&n1).await.unwrap();
        let list2 = store.list_for_node(&n2).await.unwrap();
        assert_eq!(list1.len(), 1);
        assert_eq!(list2.len(), 1);
        assert_eq!(list1[0].id, m1.id);
        assert_eq!(list2[0].id, m2.id);
    }

    #[tokio::test]
    async fn test_metadata_roundtrip() {
        let store = MediaStore::new_memory().unwrap();
        let node_id = Uuid::new_v4();
        let mut meta = make_meta(node_id, MediaType::Document);
        meta.filename = "report.pdf".into();
        meta.content_type = "application/pdf".into();
        meta.size_bytes = 42;

        store.put(&meta, b"pdf data").await.unwrap();
        let retrieved = store.get_meta(&node_id, &meta.id, MediaType::Document).await.unwrap().unwrap();

        assert_eq!(retrieved.id, meta.id);
        assert_eq!(retrieved.node_id, node_id);
        assert_eq!(retrieved.filename, "report.pdf");
        assert_eq!(retrieved.content_type, "application/pdf");
        assert_eq!(retrieved.size_bytes, 42);
        assert_eq!(retrieved.media_type, MediaType::Document);
    }

    #[tokio::test]
    async fn test_memory_backend() {
        let store = MediaStore::new_memory().unwrap();
        let node_id = Uuid::new_v4();
        let meta = make_meta(node_id, MediaType::Image);

        // Full lifecycle
        assert!(!store.exists(&meta).await.unwrap());
        store.put(&meta, b"test").await.unwrap();
        assert!(store.exists(&meta).await.unwrap());
        assert_eq!(store.get(&meta).await.unwrap(), b"test");
        store.delete(&meta).await.unwrap();
        assert!(!store.exists(&meta).await.unwrap());
    }
}
