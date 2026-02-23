//! Server configuration loaded from environment variables.
//!
//! All settings have production-safe defaults. Override any variable at
//! container / process startup — no config file required.
//!
//! | Variable                        | Default           | Description                               |
//! |---------------------------------|-------------------|-------------------------------------------|
//! | `NIETZSCHE_DATA_DIR`            | `/data/nietzsche` | RocksDB + WAL root directory              |
//! | `NIETZSCHE_PORT`                | `50051`           | gRPC listen port                          |
//! | `NIETZSCHE_LOG_LEVEL`           | `info`            | tracing level (trace/debug/info/warn/error) |
//! | `NIETZSCHE_SLEEP_INTERVAL_SECS` | `0`               | Auto sleep cycle interval in s (0 = off) |
//! | `NIETZSCHE_SLEEP_NOISE`         | `0.02`            | Sleep perturbation magnitude              |
//! | `NIETZSCHE_SLEEP_ADAM_STEPS`    | `10`              | RiemannianAdam steps per node             |
//! | `NIETZSCHE_HAUSDORFF_THRESHOLD` | `0.15`            | Max \|ΔH\| to commit a sleep cycle       |
//! | `NIETZSCHE_MAX_CONNECTIONS`     | `1024`            | Max concurrent gRPC connections           |

/// Runtime configuration for the NietzscheDB server process.
#[derive(Debug)]
pub struct Config {
    /// Root directory for RocksDB and the WAL.
    pub data_dir: String,

    /// gRPC listen port.
    pub port: u16,

    /// Tracing filter string, e.g. `"nietzsche_api=debug,info"`.
    pub log_level: String,

    /// Seconds between automatic sleep cycles (0 = disabled).
    pub sleep_interval_secs: u64,

    /// Perturbation noise for the sleep cycle.
    pub sleep_noise: f64,

    /// RiemannianAdam steps per node during the sleep cycle.
    pub sleep_adam_steps: usize,

    /// Maximum |ΔH| to commit the sleep cycle result.
    pub hausdorff_threshold: f32,

    /// Maximum concurrent gRPC connections (passed to tonic concurrency limit).
    pub max_connections: usize,

    /// HTTP dashboard listen port (0 = disabled).
    pub dashboard_port: u16,

    // ── Cluster ──────────────────────────────────────────────────────────────
    /// Enable cluster mode (false = standalone).
    pub cluster_enabled: bool,

    /// Human-readable name for this node (e.g. "shard-0").
    pub cluster_node_name: String,

    /// Role: "primary" | "replica" | "coordinator". Default "primary".
    pub cluster_role: String,

    /// Comma-separated seed peers: "name@host:port,name@host:port".
    pub cluster_seeds: String,

    // ── TTL ──────────────────────────────────────────────────────────────────
    /// Interval in seconds between TTL reaper runs (0 = disabled). Default: 60.
    pub ttl_reaper_interval_secs: u64,

    // ── Backup ───────────────────────────────────────────────────────────────
    /// Interval in seconds between automatic backups (0 = disabled). Default: 0.
    pub backup_interval_secs: u64,

    /// Number of most recent backups to keep (older ones pruned). Default: 5.
    pub backup_retention_count: usize,

    // ── Metadata Indexes ──────────────────────────────────────────────────
    /// Comma-separated list of metadata fields to maintain secondary indexes on.
    /// e.g. `"created_at,node_type,category"`. Empty = no metadata indexing.
    pub indexed_fields: Vec<String>,

    /// Directory containing .onnx models for the neural foundation.
    pub model_dir: String,
}

impl Config {
    /// Load configuration from environment variables, applying defaults where
    /// a variable is absent or unparseable.
    pub fn from_env() -> Self {
        Self {
            data_dir:            env_str("NIETZSCHE_DATA_DIR", "/data/nietzsche"),
            port:                env_parse("NIETZSCHE_PORT", 50051),
            log_level:           env_str("NIETZSCHE_LOG_LEVEL", "info"),
            sleep_interval_secs: env_parse("NIETZSCHE_SLEEP_INTERVAL_SECS", 0),
            sleep_noise:         env_parse("NIETZSCHE_SLEEP_NOISE", 0.02),
            sleep_adam_steps:    env_parse("NIETZSCHE_SLEEP_ADAM_STEPS", 10),
            hausdorff_threshold: env_parse("NIETZSCHE_HAUSDORFF_THRESHOLD", 0.15_f32),
            max_connections:     env_parse("NIETZSCHE_MAX_CONNECTIONS", 1024),
            dashboard_port:      env_parse("NIETZSCHE_DASHBOARD_PORT", 8080_u16),
            cluster_enabled:     env_bool("NIETZSCHE_CLUSTER_ENABLED"),
            cluster_node_name:   env_str("NIETZSCHE_CLUSTER_NODE_NAME", "nietzsche-0"),
            cluster_role:        env_str("NIETZSCHE_CLUSTER_ROLE", "primary"),
            cluster_seeds:       env_str("NIETZSCHE_CLUSTER_SEEDS", ""),
            ttl_reaper_interval_secs: env_parse("NIETZSCHE_TTL_REAPER_INTERVAL_SECS", 60),
            backup_interval_secs:    env_parse("NIETZSCHE_BACKUP_INTERVAL_SECS", 0),
            backup_retention_count:  env_parse("NIETZSCHE_BACKUP_RETENTION_COUNT", 5),
            indexed_fields:          env_csv("NIETZSCHE_INDEXED_FIELDS"),
            model_dir:               env_str("NIETZSCHE_MODEL_DIR", "/data/nietzsche/models"),
        }
    }
}

fn env_bool(key: &str) -> bool {
    std::env::var(key).map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false)
}

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    match std::env::var(key) {
        Ok(v) => match v.parse() {
            Ok(parsed) => parsed,
            Err(_) => {
                eprintln!("WARNING: env var {key}={v:?} is not valid; using default");
                default
            }
        },
        Err(_) => default,
    }
}

fn env_csv(key: &str) -> Vec<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let cfg = Config::from_env();
        assert!(!cfg.data_dir.is_empty());
        assert!(cfg.port > 0);
        assert!(cfg.sleep_noise > 0.0);
        assert!(cfg.hausdorff_threshold > 0.0);
        assert!(cfg.max_connections > 0);
    }

    #[test]
    fn env_override_applied() {
        std::env::set_var("NIETZSCHE_PORT", "9090");
        let cfg = Config::from_env();
        assert_eq!(cfg.port, 9090);
        std::env::remove_var("NIETZSCHE_PORT");
    }
}
