use nietzsche_graph::GraphStorage;

use crate::error::DaemonError;
use crate::model::DaemonDef;

const DAEMON_PREFIX: &str = "daemon:";

/// Persist a daemon definition to CF_META.
pub fn put_daemon(storage: &GraphStorage, def: &DaemonDef) -> Result<(), DaemonError> {
    let key = format!("{}{}", DAEMON_PREFIX, def.name);
    let json = serde_json::to_vec(def)?;
    storage.put_meta(&key, &json)
        .map_err(|e| DaemonError::Storage(e.to_string()))
}

/// Retrieve a daemon definition by name.
pub fn get_daemon(storage: &GraphStorage, name: &str) -> Result<Option<DaemonDef>, DaemonError> {
    let key = format!("{}{}", DAEMON_PREFIX, name);
    match storage.get_meta(&key).map_err(|e| DaemonError::Storage(e.to_string()))? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// Delete a daemon definition by name.
pub fn delete_daemon(storage: &GraphStorage, name: &str) -> Result<(), DaemonError> {
    let key = format!("{}{}", DAEMON_PREFIX, name);
    storage.delete_meta(&key)
        .map_err(|e| DaemonError::Storage(e.to_string()))
}

/// List all daemon definitions.
pub fn list_daemons(storage: &GraphStorage) -> Result<Vec<DaemonDef>, DaemonError> {
    let entries = storage.scan_meta_prefix(DAEMON_PREFIX.as_bytes())
        .map_err(|e| DaemonError::Storage(e.to_string()))?;
    let mut daemons = Vec::with_capacity(entries.len());
    for (_key, value) in entries {
        let def: DaemonDef = serde_json::from_slice(&value)?;
        daemons.push(def);
    }
    Ok(daemons)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_storage() -> (tempfile::TempDir, GraphStorage) {
        let dir = tempfile::tempdir().unwrap();
        let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();
        (dir, storage)
    }

    fn sample_daemon(name: &str) -> DaemonDef {
        use nietzsche_query::ast::*;
        DaemonDef {
            name: name.to_string(),
            on_pattern: NodePattern { alias: "n".into(), label: Some("Memory".into()) },
            when_cond: Condition::Compare {
                left:  Expr::Property { alias: "n".into(), field: "energy".into() },
                op:    CompOp::Gt,
                right: Expr::Float(0.8),
            },
            then_action: DaemonAction::Delete { alias: "n".into() },
            every: Expr::MathFunc {
                func: MathFunc::Interval,
                args: vec![MathFuncArg::Str("1h".into())],
            },
            energy: 1.0,
            last_run: 0.0,
            interval_secs: 3600.0,
        }
    }

    #[test]
    fn put_and_get() {
        let (_dir, storage) = temp_storage();
        let def = sample_daemon("guardian");
        put_daemon(&storage, &def).unwrap();
        let loaded = get_daemon(&storage, "guardian").unwrap().unwrap();
        assert_eq!(loaded.name, "guardian");
        assert!((loaded.energy - 1.0).abs() < 1e-10);
    }

    #[test]
    fn list_daemons_returns_all() {
        let (_dir, storage) = temp_storage();
        put_daemon(&storage, &sample_daemon("alpha")).unwrap();
        put_daemon(&storage, &sample_daemon("beta")).unwrap();
        let all = list_daemons(&storage).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn delete_removes_daemon() {
        let (_dir, storage) = temp_storage();
        put_daemon(&storage, &sample_daemon("reaper")).unwrap();
        delete_daemon(&storage, "reaper").unwrap();
        assert!(get_daemon(&storage, "reaper").unwrap().is_none());
    }

    #[test]
    fn overwrite_updates_daemon() {
        let (_dir, storage) = temp_storage();
        let mut def = sample_daemon("guardian");
        put_daemon(&storage, &def).unwrap();
        def.energy = 0.5;
        put_daemon(&storage, &def).unwrap();
        let loaded = get_daemon(&storage, "guardian").unwrap().unwrap();
        assert!((loaded.energy - 0.5).abs() < 1e-10);
    }
}
