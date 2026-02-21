use nietzsche_graph::GraphStorage;

use crate::error::DreamError;
use crate::model::{DreamSession, DreamStatus};

const DREAM_PREFIX: &str = "dream:";

/// Persist a dream session to CF_META.
pub fn put_dream(storage: &GraphStorage, session: &DreamSession) -> Result<(), DreamError> {
    let key = format!("{}{}", DREAM_PREFIX, session.id);
    let json = serde_json::to_vec(session)?;
    storage.put_meta(&key, &json)
        .map_err(|e| DreamError::Storage(e.to_string()))
}

/// Retrieve a dream session by ID.
pub fn get_dream(storage: &GraphStorage, id: &str) -> Result<Option<DreamSession>, DreamError> {
    let key = format!("{}{}", DREAM_PREFIX, id);
    match storage.get_meta(&key).map_err(|e| DreamError::Storage(e.to_string()))? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// Delete a dream session.
pub fn delete_dream(storage: &GraphStorage, id: &str) -> Result<(), DreamError> {
    let key = format!("{}{}", DREAM_PREFIX, id);
    storage.delete_meta(&key)
        .map_err(|e| DreamError::Storage(e.to_string()))
}

/// List all dream sessions (any status).
pub fn list_dreams(storage: &GraphStorage) -> Result<Vec<DreamSession>, DreamError> {
    let entries = storage.scan_meta_prefix(DREAM_PREFIX.as_bytes())
        .map_err(|e| DreamError::Storage(e.to_string()))?;
    let mut dreams = Vec::with_capacity(entries.len());
    for (_key, value) in entries {
        let session: DreamSession = serde_json::from_slice(&value)?;
        dreams.push(session);
    }
    Ok(dreams)
}

/// List only pending dream sessions.
pub fn list_pending_dreams(storage: &GraphStorage) -> Result<Vec<DreamSession>, DreamError> {
    Ok(list_dreams(storage)?
        .into_iter()
        .filter(|d| d.status == DreamStatus::Pending)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::*;
    use uuid::Uuid;

    fn temp_storage() -> (tempfile::TempDir, GraphStorage) {
        let dir = tempfile::tempdir().unwrap();
        let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();
        (dir, storage)
    }

    fn sample_dream() -> DreamSession {
        DreamSession {
            id:          "dream_001".to_string(),
            seed_node:   Uuid::new_v4(),
            depth:       3,
            noise:       0.1,
            events:      vec![DreamEvent {
                event_type: DreamEventType::EnergySpike,
                node_id:    Uuid::new_v4(),
                energy:     0.95,
                depth:      0.3,
                description: "energy spike detected".to_string(),
            }],
            created_at:  1000,
            status:      DreamStatus::Pending,
            dream_nodes: vec![],
        }
    }

    #[test]
    fn put_and_get() {
        let (_dir, storage) = temp_storage();
        let session = sample_dream();
        put_dream(&storage, &session).unwrap();
        let loaded = get_dream(&storage, "dream_001").unwrap().unwrap();
        assert_eq!(loaded.id, "dream_001");
        assert_eq!(loaded.status, DreamStatus::Pending);
    }

    #[test]
    fn list_dreams_returns_all() {
        let (_dir, storage) = temp_storage();
        put_dream(&storage, &sample_dream()).unwrap();
        let mut d2 = sample_dream();
        d2.id = "dream_002".to_string();
        put_dream(&storage, &d2).unwrap();
        let all = list_dreams(&storage).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn delete_removes_dream() {
        let (_dir, storage) = temp_storage();
        put_dream(&storage, &sample_dream()).unwrap();
        delete_dream(&storage, "dream_001").unwrap();
        assert!(get_dream(&storage, "dream_001").unwrap().is_none());
    }

    #[test]
    fn list_pending_filters() {
        let (_dir, storage) = temp_storage();
        put_dream(&storage, &sample_dream()).unwrap();
        let mut d2 = sample_dream();
        d2.id = "dream_applied".to_string();
        d2.status = DreamStatus::Applied;
        put_dream(&storage, &d2).unwrap();
        let pending = list_pending_dreams(&storage).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, "dream_001");
    }
}
