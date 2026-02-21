use uuid::Uuid;

use nietzsche_graph::GraphStorage;

use crate::error::NamedVectorError;
use crate::model::NamedVector;

/// Key prefix used in CF_META for named vectors.
const KEY_PREFIX: &str = "nvec:";

/// Persistence layer for named vectors.
///
/// Uses `GraphStorage`'s CF_META column family with key format:
///
/// ```text
/// nvec:{node_id}:{name}
/// ```
///
/// Values are bincode-serialized [`NamedVector`] structs.
pub struct NamedVectorStore;

impl NamedVectorStore {
    /// Build the CF_META key for a named vector.
    #[inline]
    fn meta_key(node_id: &Uuid, name: &str) -> String {
        format!("{KEY_PREFIX}{}:{}", node_id, name)
    }

    /// Build the prefix that matches all named vectors for a given node.
    #[inline]
    fn node_prefix(node_id: &Uuid) -> String {
        format!("{KEY_PREFIX}{}:", node_id)
    }

    /// Store (insert or overwrite) a named vector.
    pub fn put(storage: &GraphStorage, vec: &NamedVector) -> Result<(), NamedVectorError> {
        let key = Self::meta_key(&vec.node_id, &vec.name);
        let value = bincode::serialize(vec)
            .map_err(|e| NamedVectorError::Serialization(e.to_string()))?;
        storage.put_meta(&key, &value)?;
        Ok(())
    }

    /// Retrieve a named vector by node ID and name.
    ///
    /// Returns `None` if no vector with that name exists for the node.
    pub fn get(
        storage: &GraphStorage,
        node_id: &Uuid,
        name: &str,
    ) -> Result<Option<NamedVector>, NamedVectorError> {
        let key = Self::meta_key(node_id, name);
        match storage.get_meta(&key)? {
            Some(bytes) => {
                let nv: NamedVector = bincode::deserialize(&bytes)
                    .map_err(|e| NamedVectorError::Serialization(e.to_string()))?;
                Ok(Some(nv))
            }
            None => Ok(None),
        }
    }

    /// List all named vectors attached to a node.
    ///
    /// Performs a prefix scan on `nvec:{node_id}:` in CF_META.
    pub fn list(
        storage: &GraphStorage,
        node_id: &Uuid,
    ) -> Result<Vec<NamedVector>, NamedVectorError> {
        let prefix = Self::node_prefix(node_id);
        let entries = storage.scan_meta_prefix(prefix.as_bytes())?;
        let mut vectors = Vec::with_capacity(entries.len());
        for (_key, value) in entries {
            let nv: NamedVector = bincode::deserialize(&value)
                .map_err(|e| NamedVectorError::Serialization(e.to_string()))?;
            vectors.push(nv);
        }
        Ok(vectors)
    }

    /// Delete a single named vector.
    pub fn delete(
        storage: &GraphStorage,
        node_id: &Uuid,
        name: &str,
    ) -> Result<(), NamedVectorError> {
        let key = Self::meta_key(node_id, name);
        storage.delete_meta(&key)?;
        Ok(())
    }

    /// Delete all named vectors for a node.
    ///
    /// Returns the number of vectors deleted.
    pub fn delete_all(
        storage: &GraphStorage,
        node_id: &Uuid,
    ) -> Result<usize, NamedVectorError> {
        let prefix = Self::node_prefix(node_id);
        let entries = storage.scan_meta_prefix(prefix.as_bytes())?;
        let count = entries.len();
        for (key_bytes, _) in &entries {
            // Convert key bytes back to a string for delete_meta
            let key_str = std::str::from_utf8(key_bytes)
                .map_err(|e| NamedVectorError::Serialization(e.to_string()))?;
            storage.delete_meta(key_str)?;
        }
        Ok(count)
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::VectorMetric;
    use tempfile::TempDir;

    fn open_temp_storage() -> (GraphStorage, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();
        (storage, dir)
    }

    fn make_named_vec(node_id: Uuid, name: &str, metric: VectorMetric) -> NamedVector {
        NamedVector::new(node_id, name, vec![0.1, 0.2, 0.3], metric)
    }

    #[test]
    fn test_put_and_get() {
        let (storage, _dir) = open_temp_storage();
        let node_id = Uuid::new_v4();
        let nv = make_named_vec(node_id, "text", VectorMetric::Cosine);

        NamedVectorStore::put(&storage, &nv).unwrap();
        let retrieved = NamedVectorStore::get(&storage, &node_id, "text").unwrap().unwrap();

        assert_eq!(retrieved.node_id, node_id);
        assert_eq!(retrieved.name, "text");
        assert_eq!(retrieved.coordinates, vec![0.1, 0.2, 0.3]);
        assert_eq!(retrieved.metric, VectorMetric::Cosine);
    }

    #[test]
    fn test_get_nonexistent() {
        let (storage, _dir) = open_temp_storage();
        let node_id = Uuid::new_v4();

        let result = NamedVectorStore::get(&storage, &node_id, "nope").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_list_vectors() {
        let (storage, _dir) = open_temp_storage();
        let node_id = Uuid::new_v4();

        let v_text = make_named_vec(node_id, "text", VectorMetric::Cosine);
        let v_image = make_named_vec(node_id, "image", VectorMetric::Euclidean);
        let v_audio = make_named_vec(node_id, "audio", VectorMetric::Poincare);

        NamedVectorStore::put(&storage, &v_text).unwrap();
        NamedVectorStore::put(&storage, &v_image).unwrap();
        NamedVectorStore::put(&storage, &v_audio).unwrap();

        let listed = NamedVectorStore::list(&storage, &node_id).unwrap();
        assert_eq!(listed.len(), 3);

        let names: Vec<&str> = listed.iter().map(|v| v.name.as_str()).collect();
        assert!(names.contains(&"text"));
        assert!(names.contains(&"image"));
        assert!(names.contains(&"audio"));
    }

    #[test]
    fn test_delete() {
        let (storage, _dir) = open_temp_storage();
        let node_id = Uuid::new_v4();
        let nv = make_named_vec(node_id, "text", VectorMetric::Cosine);

        NamedVectorStore::put(&storage, &nv).unwrap();
        assert!(NamedVectorStore::get(&storage, &node_id, "text").unwrap().is_some());

        NamedVectorStore::delete(&storage, &node_id, "text").unwrap();
        assert!(NamedVectorStore::get(&storage, &node_id, "text").unwrap().is_none());
    }

    #[test]
    fn test_delete_all() {
        let (storage, _dir) = open_temp_storage();
        let node_id = Uuid::new_v4();

        NamedVectorStore::put(&storage, &make_named_vec(node_id, "text", VectorMetric::Cosine)).unwrap();
        NamedVectorStore::put(&storage, &make_named_vec(node_id, "image", VectorMetric::Euclidean)).unwrap();
        NamedVectorStore::put(&storage, &make_named_vec(node_id, "audio", VectorMetric::Poincare)).unwrap();

        let count = NamedVectorStore::delete_all(&storage, &node_id).unwrap();
        assert_eq!(count, 3);

        let remaining = NamedVectorStore::list(&storage, &node_id).unwrap();
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_overwrite() {
        let (storage, _dir) = open_temp_storage();
        let node_id = Uuid::new_v4();

        // First write
        let v1 = NamedVector::new(node_id, "text", vec![1.0, 2.0, 3.0], VectorMetric::Cosine);
        NamedVectorStore::put(&storage, &v1).unwrap();

        // Overwrite with different coordinates and metric
        let v2 = NamedVector::new(node_id, "text", vec![4.0, 5.0, 6.0], VectorMetric::Euclidean);
        NamedVectorStore::put(&storage, &v2).unwrap();

        let retrieved = NamedVectorStore::get(&storage, &node_id, "text").unwrap().unwrap();
        assert_eq!(retrieved.coordinates, vec![4.0, 5.0, 6.0]);
        assert_eq!(retrieved.metric, VectorMetric::Euclidean);
    }

    #[test]
    fn test_different_nodes() {
        let (storage, _dir) = open_temp_storage();
        let node_a = Uuid::new_v4();
        let node_b = Uuid::new_v4();

        let va = NamedVector::new(node_a, "text", vec![1.0, 0.0], VectorMetric::Cosine);
        let vb = NamedVector::new(node_b, "text", vec![0.0, 1.0], VectorMetric::Euclidean);

        NamedVectorStore::put(&storage, &va).unwrap();
        NamedVectorStore::put(&storage, &vb).unwrap();

        // Each node sees only its own vector
        let list_a = NamedVectorStore::list(&storage, &node_a).unwrap();
        assert_eq!(list_a.len(), 1);
        assert_eq!(list_a[0].coordinates, vec![1.0, 0.0]);

        let list_b = NamedVectorStore::list(&storage, &node_b).unwrap();
        assert_eq!(list_b.len(), 1);
        assert_eq!(list_b[0].coordinates, vec![0.0, 1.0]);

        // Deleting all for node_a does not affect node_b
        NamedVectorStore::delete_all(&storage, &node_a).unwrap();
        assert!(NamedVectorStore::list(&storage, &node_a).unwrap().is_empty());
        assert_eq!(NamedVectorStore::list(&storage, &node_b).unwrap().len(), 1);
    }

    #[test]
    fn test_vector_metrics() {
        let (storage, _dir) = open_temp_storage();
        let node_id = Uuid::new_v4();

        let v_poincare = NamedVector::new(node_id, "default", vec![0.1, 0.2], VectorMetric::Poincare);
        let v_cosine = NamedVector::new(node_id, "text", vec![0.3, 0.4], VectorMetric::Cosine);
        let v_euclid = NamedVector::new(node_id, "image", vec![0.5, 0.6], VectorMetric::Euclidean);

        NamedVectorStore::put(&storage, &v_poincare).unwrap();
        NamedVectorStore::put(&storage, &v_cosine).unwrap();
        NamedVectorStore::put(&storage, &v_euclid).unwrap();

        let got_p = NamedVectorStore::get(&storage, &node_id, "default").unwrap().unwrap();
        assert_eq!(got_p.metric, VectorMetric::Poincare);

        let got_c = NamedVectorStore::get(&storage, &node_id, "text").unwrap().unwrap();
        assert_eq!(got_c.metric, VectorMetric::Cosine);

        let got_e = NamedVectorStore::get(&storage, &node_id, "image").unwrap().unwrap();
        assert_eq!(got_e.metric, VectorMetric::Euclidean);
    }
}
