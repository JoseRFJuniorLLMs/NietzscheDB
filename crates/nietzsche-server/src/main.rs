//! NietzscheDB production gRPC server.
//!
//! Reads configuration from environment variables (see [`config::Config`]),
//! opens (or creates) a [`CollectionManager`] at the data directory,
//! optionally starts a background sleep-cycle scheduler against the `"default"`
//! collection, then serves the gRPC API + HTTP dashboard until SIGINT.
//!
//! ## Quick start
//!
//! ```bash
//! # Development (local data dir, port 50051, info log)
//! cargo run --bin nietzsche-server --release
//!
//! # Docker
//! docker run -p 50051:50051 -v /mnt/data:/data/nietzsche nietzsche-server
//!
//! # Custom config
//! NIETZSCHE_PORT=9090 \
//! NIETZSCHE_DATA_DIR=/mnt/data \
//! NIETZSCHE_SLEEP_INTERVAL_SECS=300 \
//! NIETZSCHE_LOG_LEVEL=debug \
//! NIETZSCHE_VECTOR_BACKEND=embedded \
//!   cargo run --bin nietzsche-server --release
//! ```

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tracing::{error, info, warn};
use tracing_subscriber::{fmt, EnvFilter};

use nietzsche_api::{CdcBroadcaster, NietzscheServer};
use nietzsche_graph::{CollectionManager, embedded_vector_store::AnyVectorStore};
use nietzsche_sleep::{SleepConfig, SleepCycle};
use nietzsche_zaratustra::{ZaratustraEngine, ZaratustraConfig};
#[allow(unused_imports)]
use nietzsche_lsystem::LSystemEngine;
use nietzsche_dream::{DreamConfig, DreamEngine};
use nietzsche_narrative::{NarrativeConfig, NarrativeEngine};
use nietzsche_dsi::DsiIndexer;
use nietzsche_vqvae::{VqEncoder, VqVaeConfig};

#[cfg(feature = "gpu")]
use nietzsche_hnsw_gpu::GpuVectorStore;

#[cfg(feature = "tpu")]
use nietzsche_tpu::TpuVectorStore;

mod auth;
mod cluster_service;
mod config;
mod dashboard;
mod html;
mod metrics;
use config::Config;
use metrics::OperationMetrics;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── Tracing ───────────────────────────────────────────────────────────────
    let config = Config::from_env();

    let filter = EnvFilter::try_new(&config.log_level)
        .unwrap_or_else(|_| EnvFilter::new("info"));

    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(false)
        .compact()
        .init();

    info!(
        version  = env!("CARGO_PKG_VERSION"),
        data_dir = %config.data_dir,
        port     = config.port,
        "NietzscheDB starting"
    );

    // ── Open CollectionManager ────────────────────────────────────────────────
    let db_path = PathBuf::from(&config.data_dir);
    std::fs::create_dir_all(&db_path)?;

    let cm = CollectionManager::open(&db_path)
        .map_err(|e| anyhow::anyhow!("failed to open CollectionManager at {}: {e}", config.data_dir))?;

    // Log collection inventory & configure indexed_fields
    let indexed_fields: std::collections::HashSet<String> =
        config.indexed_fields.iter().cloned().collect();
    for col in cm.list() {
        info!(
            collection = %col.name,
            dim        = col.dim,
            metric     = %col.metric,
            nodes      = col.node_count,
            edges      = col.edge_count,
            "collection loaded"
        );
        if !indexed_fields.is_empty() {
            if let Some(shared) = cm.get(&col.name) {
                let mut db = shared.blocking_write();
                db.set_indexed_fields(indexed_fields.clone());
            }
        }
    }
    if !indexed_fields.is_empty() {
        info!(fields = ?config.indexed_fields, "metadata indexing enabled");
    }

    // ── Neural Model Registry — Background Scanner ────────────────────────────
    // Periodically scans NIETZSCHE_MODEL_DIR for new .onnx files and loads them.
    {
        let model_dir = config.model_dir.clone();
        tokio::spawn(async move {
            let interval = Duration::from_secs(60);
            info!(path = %model_dir, "background neural model scanner started");
            loop {
                let model_path = PathBuf::from(&model_dir);
                if model_path.exists() && model_path.is_dir() {
                    if let Ok(entries) = std::fs::read_dir(&model_path) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                                let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
                                
                                // Check if already loaded (REGISTRY.list_models() or similar)
                                // For now, load_model will just overwrite in DashMap which is safe.
                                let meta = nietzsche_neural::ModelMetadata {
                                    name: name.to_string(),
                                    path: path.clone(),
                                    version: "1.0".to_string(),
                                    input_shape: vec![],
                                    output_shape: vec![],
                                };
                                
                                match nietzsche_neural::REGISTRY.load_model(meta) {
                                    Ok(_) => {} // Silent on repeat loads
                                    Err(e) => warn!(model = %name, error = %e, "failed to load neural model in background"),
                                }
                            }
                        }
                    }
                }
                tokio::time::sleep(interval).await;
            }
        });
    }

    // ── DSI Background Indexer — Periodically index nodes for generative retrieval ──
    {
        let cm_dsi = Arc::clone(&cm);
        tokio::spawn(async move {
            info!("background DSI indexer started");
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            // Wait for models to be loaded by the other scanner
            tokio::time::sleep(Duration::from_secs(10)).await;

            let config = VqVaeConfig::default();
            let indexer = DsiIndexer::new(VqEncoder::new(config), 3); // 3 levels: e.g. 1.2.3

            loop {
                interval.tick().await;

                for col_info in cm_dsi.list() {
                    let Some(shared) = cm_dsi.get(&col_info.name) else { continue; };
                    // We need a read lock to scan, but index_node might need its own locks or internal access.
                    // Actually DsiIndexer::index_node takes &GraphStorage, which we can get from the db handle.
                    let db = shared.read().await;
                    let storage = db.storage();

                    let nodes = match storage.scan_nodes() {
                        Ok(n) => n,
                        Err(e) => {
                            warn!(collection = %col_info.name, error = %e, "DSI indexer: scan failed");
                            continue;
                        }
                    };

                    for node in nodes {
                        // Check if node already has a DSI ID
                        match storage.get_dsi_id(&node.id) {
                            Ok(Some(_)) => continue, // already indexed
                            Ok(None) => {
                                // Attempt to index the node
                                // Note: this requires the "vqvae_graph_v1" model to be loaded in REGISTRY.
                                if let Err(e) = indexer.index_node(storage, &node.id).await {
                                    // Silent on most errors as it might just be the model not loaded yet
                                    tracing::trace!(node_id = %node.id, error = %e, "DSI indexer: index failed");
                                } else {
                                    info!(node_id = %node.id, collection = %col_info.name, "DSI indexer: node indexed");
                                }
                            }
                            Err(e) => warn!(node_id = %node.id, error = %e, "DSI indexer: storage error"),
                        }
                    }
                }
            }
        });
    }

    // ── GPU vector backend injection ──────────────────────────────────────────
    // Activated when compiled with `--features gpu` AND
    // `NIETZSCHE_VECTOR_BACKEND=gpu` is set at runtime.
    // Replaces each collection's CPU HNSW with a GPU CAGRA store before
    // the gRPC server starts accepting requests.
    #[cfg(feature = "gpu")]
    {
        let backend = std::env::var("NIETZSCHE_VECTOR_BACKEND")
            .unwrap_or_default()
            .to_lowercase();

        if backend == "gpu" {
            info!("GPU vector backend requested — initialising CAGRA for all collections");

            for col in cm.list() {
                match GpuVectorStore::new(col.dim) {
                    Ok(gpu_store) => {
                        if let Some(shared) = cm.get(&col.name) {
                            let mut db = shared.write().await;
                            db.set_vector_store(AnyVectorStore::gpu(Box::new(gpu_store)));
                            info!(
                                collection = %col.name,
                                dim        = col.dim,
                                backend    = "GpuVectorStore(CAGRA/cuVS)",
                                "GPU backend active"
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            collection = %col.name,
                            error      = %e,
                            "GPU init failed — keeping CPU HNSW for this collection"
                        );
                    }
                }
            }
        } else {
            info!("GPU feature compiled in but NIETZSCHE_VECTOR_BACKEND != gpu — using CPU HNSW");
        }
    }

    // ── TPU vector backend injection ──────────────────────────────────────────
    // Activated when compiled with `--features tpu` AND
    // `NIETZSCHE_VECTOR_BACKEND=tpu` is set at runtime.
    // Requires `PJRT_PLUGIN_PATH` pointing to libtpu.so on a Cloud TPU VM.
    // Falls back to CPU scan per collection if PJRT init fails.
    #[cfg(feature = "tpu")]
    {
        let backend = std::env::var("NIETZSCHE_VECTOR_BACKEND")
            .unwrap_or_default()
            .to_lowercase();

        if backend == "tpu" {
            info!(
                plugin = %std::env::var("PJRT_PLUGIN_PATH").unwrap_or_else(|_| "(not set)".into()),
                "TPU vector backend requested — initialising PJRT for all collections"
            );

            for col in cm.list() {
                match TpuVectorStore::new(col.dim) {
                    Ok(tpu_store) => {
                        if let Some(shared) = cm.get(&col.name) {
                            let mut db = shared.write().await;
                            db.set_vector_store(AnyVectorStore::tpu(Box::new(tpu_store)));
                            info!(
                                collection = %col.name,
                                dim        = col.dim,
                                backend    = "TpuVectorStore(PJRT/MHLO)",
                                "TPU backend active"
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            collection = %col.name,
                            error      = %e,
                            "TPU init failed — keeping CPU HNSW for this collection"
                        );
                    }
                }
            }
        } else {
            info!("TPU feature compiled in but NIETZSCHE_VECTOR_BACKEND != tpu — using CPU HNSW");
        }
    }

    // ── Sleep scheduler — runs against the "default" collection ───────────────
    if config.sleep_interval_secs > 0 {
        let interval  = Duration::from_secs(config.sleep_interval_secs);
        let sleep_cfg = SleepConfig {
            noise:               config.sleep_noise,
            adam_steps:          config.sleep_adam_steps,
            adam_lr:             5e-3,
            hausdorff_threshold: config.hausdorff_threshold,
            ..Default::default()
        };
        let cm_sleep = Arc::clone(&cm);

        tokio::spawn(async move {
            info!(interval_secs = config.sleep_interval_secs, "sleep scheduler started (default collection)");
            loop {
                tokio::time::sleep(interval).await;

                let Some(shared) = cm_sleep.get_or_default("default") else {
                    warn!("sleep scheduler: 'default' collection not found — skipping");
                    continue;
                };

                let mut db    = shared.write().await;
                let seed: u64 = rand::random();
                let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);

                match SleepCycle::run(&sleep_cfg, &mut *db, &mut rng) {
                    Ok(r) => info!(
                        hausdorff_before    = r.hausdorff_before,
                        hausdorff_after     = r.hausdorff_after,
                        hausdorff_delta     = r.hausdorff_delta,
                        semantic_drift_avg  = %format!("{:.6}", r.semantic_drift_avg),
                        semantic_drift_max  = %format!("{:.6}", r.semantic_drift_max),
                        committed           = r.committed,
                        nodes_perturbed     = r.nodes_perturbed,
                        "sleep cycle complete"
                    ),
                    Err(e) => warn!(error = %e, "sleep cycle failed"),
                }
            }
        });
    } else {
        info!("sleep scheduler disabled (NIETZSCHE_SLEEP_INTERVAL_SECS=0)");
    }

    // ── Zaratustra scheduler ──────────────────────────────────────────────────
    // Reads `ZARATUSTRA_INTERVAL_SECS` (default 600 = 10 min, 0 = disabled).
    // Runs the three-phase Zaratustra cycle against the "default" collection.
    let zaratustra_interval: u64 = std::env::var("ZARATUSTRA_INTERVAL_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);

    if zaratustra_interval > 0 {
        let engine     = ZaratustraEngine::from_env();
        let cm_zara    = Arc::clone(&cm);
        let interval_z = Duration::from_secs(zaratustra_interval);

        tokio::spawn(async move {
            info!(
                interval_secs = zaratustra_interval,
                alpha  = engine.config.alpha,
                decay  = engine.config.decay,
                "Zaratustra scheduler started"
            );
            loop {
                tokio::time::sleep(interval_z).await;

                let Some(shared) = cm_zara.get_or_default("default") else {
                    warn!("zaratustra: 'default' collection not found — skipping");
                    continue;
                };

                // Hold a read lock for Zaratustra — it only reads storage/adjacency.
                let db        = shared.read().await;
                let storage   = db.storage();
                let adjacency = db.adjacency();

                match engine.run_cycle(storage, adjacency) {
                    Ok(r) => info!(
                        nodes_updated    = r.will_to_power.nodes_updated,
                        energy_before    = r.will_to_power.mean_energy_before,
                        energy_after     = r.will_to_power.mean_energy_after,
                        echoes_created   = r.eternal_recurrence.echoes_created,
                        elite_count      = r.ubermensch.elite_count,
                        elite_threshold  = r.ubermensch.energy_threshold,
                        duration_ms      = r.duration_ms,
                        "Zaratustra cycle complete"
                    ),
                    Err(e) => warn!(error = %e, "Zaratustra cycle failed"),
                }
            }
        });
    } else {
        info!("Zaratustra scheduler disabled (ZARATUSTRA_INTERVAL_SECS=0)");
    }

    // ── TTL reaper — periodically deletes expired nodes ──────────────────────
    if config.ttl_reaper_interval_secs > 0 {
        let cm_ttl   = Arc::clone(&cm);
        let interval = Duration::from_secs(config.ttl_reaper_interval_secs);

        tokio::spawn(async move {
            info!(interval_secs = config.ttl_reaper_interval_secs, "TTL reaper started");
            loop {
                tokio::time::sleep(interval).await;
                for col_info in cm_ttl.list() {
                    if let Some(shared) = cm_ttl.get(&col_info.name) {
                        let mut db = shared.write().await;
                        match db.reap_expired() {
                            Ok(0) => {}
                            Ok(n) => info!(
                                collection = %col_info.name,
                                reaped     = n,
                                "TTL reaper: expired nodes removed"
                            ),
                            Err(e) => warn!(
                                collection = %col_info.name,
                                error      = %e,
                                "TTL reaper error"
                            ),
                        }
                    }
                }
            }
        });
    } else {
        info!("TTL reaper disabled (NIETZSCHE_TTL_REAPER_INTERVAL_SECS=0)");
    }

    // ── Scheduled backup with pruning ────────────────────────────────────────
    if config.backup_interval_secs > 0 {
        let cm_bak     = Arc::clone(&cm);
        let interval   = Duration::from_secs(config.backup_interval_secs);
        let retention  = config.backup_retention_count;
        let data_dir   = config.data_dir.clone();

        tokio::spawn(async move {
            info!(
                interval_secs = config.backup_interval_secs,
                retention     = retention,
                "scheduled backup started"
            );
            loop {
                tokio::time::sleep(interval).await;

                // Backup each collection
                for col_info in cm_bak.list() {
                    if let Some(shared) = cm_bak.get(&col_info.name) {
                        let db = shared.read().await;
                        let backup_dir = PathBuf::from(&data_dir)
                            .join("backups")
                            .join(&col_info.name);

                        match nietzsche_graph::BackupManager::new(&backup_dir) {
                            Ok(mgr) => {
                                match mgr.create_backup(db.storage(), "scheduled") {
                                    Ok(bi) => {
                                        info!(
                                            collection = %col_info.name,
                                            label      = %bi.label,
                                            size_bytes = bi.size_bytes,
                                            "scheduled backup created"
                                        );
                                        // Prune old backups
                                        match mgr.prune_backups(retention) {
                                            Ok(0) => {}
                                            Ok(n) => info!(
                                                collection = %col_info.name,
                                                pruned     = n,
                                                "old backups pruned"
                                            ),
                                            Err(e) => warn!(
                                                collection = %col_info.name,
                                                error      = %e,
                                                "backup prune error"
                                            ),
                                        }
                                    }
                                    Err(e) => warn!(
                                        collection = %col_info.name,
                                        error      = %e,
                                        "scheduled backup failed"
                                    ),
                                }
                            }
                            Err(e) => warn!(
                                collection = %col_info.name,
                                error      = %e,
                                "backup manager init failed"
                            ),
                        }
                    }
                }
            }
        });
    } else {
        info!("scheduled backup disabled (NIETZSCHE_BACKUP_INTERVAL_SECS=0)");
    }

    // ── Wiederkehr daemon engine ─────────────────────────────────────────────
    // Background loop that ticks daemon agents. Reads `DAEMON_TICK_SECS` (default 30, 0 = disabled).
    {
        let daemon_cfg = nietzsche_wiederkehr::DaemonEngineConfig::from_env();
        if daemon_cfg.tick_secs > 0 {
            let cm_daemon  = Arc::clone(&cm);
            let tick_dur   = Duration::from_secs(daemon_cfg.tick_secs);
            let engine     = nietzsche_wiederkehr::DaemonEngine::new(daemon_cfg);

            tokio::spawn(async move {
                info!(tick_secs = tick_dur.as_secs(), "Wiederkehr daemon engine started");
                loop {
                    tokio::time::sleep(tick_dur).await;

                    let Some(shared) = cm_daemon.get_or_default("default") else {
                        warn!("wiederkehr: 'default' collection not found — skipping");
                        continue;
                    };

                    // Tick under read lock (put_daemon internally writes to CF_META
                    // which is safe even under read lock since CF_META updates are
                    // atomic at the RocksDB level).
                    let db = shared.read().await;
                    match engine.tick(db.storage()) {
                        Ok(result) => {
                            if !result.intents.is_empty() || !result.reaped.is_empty() {
                                info!(
                                    intents = result.intents.len(),
                                    reaped  = result.reaped.len(),
                                    "Wiederkehr tick complete"
                                );
                            }
                            // Execute intents that require mutation
                            for intent in result.intents {
                                match intent {
                                    nietzsche_wiederkehr::DaemonIntent::DeleteNode(nid) => {
                                        drop(db);
                                        let mut db_w = shared.write().await;
                                        if let Err(e) = db_w.delete_node(nid) {
                                            warn!(node_id = %nid, error = %e, "daemon delete failed");
                                        }
                                        drop(db_w);
                                        break; // re-tick next cycle
                                    }
                                    nietzsche_wiederkehr::DaemonIntent::SetNodeFields { node_id, fields } => {
                                        if let Ok(Some(mut meta)) = db.storage().get_node_meta(&node_id) {
                                            if let serde_json::Value::Object(map) = fields {
                                                for (k, v) in map {
                                                    meta.content[&k] = v;
                                                }
                                            }
                                            if let Err(e) = db.storage().put_node_meta(&meta) {
                                                warn!(node_id = %node_id, error = %e, "daemon set failed");
                                            }
                                        }
                                    }
                                    nietzsche_wiederkehr::DaemonIntent::DiffuseFromNode { node_id, t_values, max_hops } => {
                                        // Diffuse intents are logged; actual diffusion
                                        // can be triggered via the standard NQL pipeline.
                                        info!(
                                            node_id  = %node_id,
                                            t_values = ?t_values,
                                            max_hops = max_hops,
                                            "daemon diffuse intent (logged)"
                                        );
                                    }
                                }
                            }
                        }
                        Err(e) => warn!(error = %e, "Wiederkehr tick failed"),
                    }
                }
            });
        } else {
            info!("Wiederkehr daemon engine disabled (DAEMON_TICK_SECS=0)");
        }
    }

    // ── Agency engine ─────────────────────────────────────────────────────────
    // Background loop for autonomous daemons (entropy, coherence, gap detection),
    // meta-observer health reports, reactor (event → intent), desire engine,
    // evolution, dream triggers, narrative generation, and Zaratustra modulation.
    // Reads `AGENCY_TICK_SECS` (default 60, 0 = disabled).
    // Multi-collection: iterates all collections, one engine per collection.
    {
        let agency_cfg = nietzsche_agency::AgencyConfig::from_env();
        if agency_cfg.tick_secs > 0 {
            let cm_agency  = Arc::clone(&cm);
            let tick_dur   = Duration::from_secs(agency_cfg.tick_secs);

            // Build collection → engine map for multi-collection support
            let mut engines: std::collections::HashMap<String, nietzsche_agency::AgencyEngine> =
                std::collections::HashMap::new();
            for col_info in cm_agency.list() {
                let engine = nietzsche_agency::AgencyEngine::new(agency_cfg.clone());
                engines.insert(col_info.name.clone(), engine);
            }
            // Ensure at least the "default" collection has an engine
            engines
                .entry("default".to_string())
                .or_insert_with(|| nietzsche_agency::AgencyEngine::new(agency_cfg.clone()));

            // Initialize Observer Identity for each collection
            for (col_name, eng) in &engines {
                if let Some(shared) = cm_agency.get_or_default(col_name) {
                    let db = shared.read().await;
                    match eng.ensure_observer_identity(db.storage()) {
                        Ok(id) => info!(collection = %col_name, observer_id = %id, "Observer Identity ready"),
                        Err(e) => warn!(collection = %col_name, error = %e, "failed to create Observer Identity"),
                    }
                }
            }

            // Shared Zaratustra config for modulation (wrapped in Arc<Mutex>)
            let zara_config = Arc::new(tokio::sync::Mutex::new(ZaratustraConfig::from_env()));

            // Shared dream + narrative engines
            let dream_engine = DreamEngine::new(DreamConfig::from_env());
            let narrative_engine = NarrativeEngine::new(NarrativeConfig::default());

            tokio::spawn(async move {
                info!(
                    tick_secs   = tick_dur.as_secs(),
                    collections = engines.len(),
                    "Agency engine started (multi-collection)"
                );
                loop {
                    tokio::time::sleep(tick_dur).await;

                    for (col_name, engine) in &mut engines {
                        let Some(shared) = cm_agency.get_or_default(col_name) else {
                            continue;
                        };

                        let mut db = shared.read().await;
                        match engine.tick(db.storage(), db.adjacency()) {
                            Ok(report) => {
                                if let Some(ref health) = report.health_report {
                                    info!(
                                        collection       = %col_name,
                                        global_hausdorff = health.global_hausdorff,
                                        mean_energy      = health.mean_energy,
                                        coherence        = health.coherence_score,
                                        gap_count        = health.gap_count,
                                        entropy_spikes   = health.entropy_spike_count,
                                        duration_ms      = report.duration_ms,
                                        "Agency health report"
                                    );
                                }

                                // Execute intents produced by the reactor
                                for intent in report.intents {
                                    match intent {
                                        nietzsche_agency::AgencyIntent::PersistHealthReport { report: hr } => {
                                            if let Err(e) = nietzsche_agency::put_health_report(db.storage(), &hr) {
                                                warn!(error = %e, "agency: failed to persist health report");
                                            } else {
                                                let _ = nietzsche_agency::prune_health_reports(db.storage());
                                            }
                                        }
                                        nietzsche_agency::AgencyIntent::TriggerSleepCycle { reason } => {
                                            info!(collection = %col_name, reason = %reason, "agency: triggering sleep cycle");
                                            drop(db);
                                            let mut db_w = shared.write().await;
                                            let seed: u64 = rand::random();
                                            let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);
                                            let sleep_cfg = SleepConfig {
                                                noise: 0.02, adam_steps: 10, adam_lr: 5e-3,
                                                hausdorff_threshold: 0.15,
                                                ..Default::default()
                                            };
                                            match SleepCycle::run(&sleep_cfg, &mut *db_w, &mut rng) {
                                                Ok(r) => info!(hausdorff_delta = r.hausdorff_delta, semantic_drift_avg = %format!("{:.6}", r.semantic_drift_avg), committed = r.committed, "agency sleep cycle complete"),
                                                Err(e) => warn!(error = %e, "agency sleep cycle failed"),
                                            }
                                            break;
                                        }
                                        nietzsche_agency::AgencyIntent::TriggerLSystemGrowth { reason } => {
                                            info!(collection = %col_name, reason = %reason, "agency: triggering L-System growth");
                                            drop(db);
                                            let mut db_w = shared.write().await;
                                            let lsystem = nietzsche_lsystem::LSystemEngine::new(vec![
                                                nietzsche_lsystem::ProductionRule::growth_child("agency-grow", 3),
                                                nietzsche_lsystem::ProductionRule::lateral_association("agency-assoc", 3),
                                                nietzsche_lsystem::ProductionRule::prune_fading("agency-prune", 0.1),
                                            ]);
                                            match lsystem.tick(&mut *db_w) {
                                                Ok(r) => info!(spawned = r.nodes_spawned, pruned = r.nodes_pruned, "agency L-System tick complete"),
                                                Err(e) => warn!(error = %e, "agency L-System tick failed"),
                                            }
                                            break;
                                        }
                                        nietzsche_agency::AgencyIntent::SignalKnowledgeGap { sectors, suggested_depth_range } => {
                                            info!(
                                                collection = %col_name,
                                                gap_sectors = sectors.len(),
                                                depth_range = ?(suggested_depth_range.0, suggested_depth_range.1),
                                                "agency: knowledge gap → desires persisted"
                                            );
                                        }
                                        nietzsche_agency::AgencyIntent::TriggerDream { seed_node_id, depth, reason } => {
                                            info!(collection = %col_name, seed = %seed_node_id, depth = depth, reason = %reason, "agency: triggering dream");
                                            match dream_engine.dream_from(db.storage(), seed_node_id, Some(depth), None) {
                                                Ok(session) => {
                                                    let event_count = session.events.len();
                                                    // Auto-apply dreams with significant events
                                                    if event_count > 0 {
                                                        match dream_engine.apply_dream(db.storage(), &session.id) {
                                                            Ok(_) => info!(dream_id = %session.id, events = event_count, "agency dream applied"),
                                                            Err(e) => warn!(error = %e, "agency dream apply failed"),
                                                        }
                                                    } else {
                                                        info!(dream_id = %session.id, "agency dream: no events, skipping");
                                                    }
                                                }
                                                Err(e) => warn!(error = %e, "agency dream_from failed"),
                                            }
                                        }
                                        nietzsche_agency::AgencyIntent::ModulateZaratustra { alpha, decay, reason } => {
                                            info!(alpha = alpha, decay = decay, reason = %reason, "agency: modulating Zaratustra");
                                            let mut zc = zara_config.lock().await;
                                            zc.alpha = alpha;
                                            zc.decay = decay;
                                        }
                                        nietzsche_agency::AgencyIntent::GenerateNarrative { collection, reason } => {
                                            match narrative_engine.narrate(db.storage(), &collection, None) {
                                                Ok(report) => {
                                                    info!(
                                                        collection = %collection,
                                                        events     = report.events.len(),
                                                        reason     = %reason,
                                                        summary    = %report.summary,
                                                        "agency narrative generated"
                                                    );
                                                    // Persist narrative to CF_META for dashboard access
                                                    if let Ok(json) = serde_json::to_vec(&report.summary) {
                                                        let _ = db.storage().put_meta("agency:latest_narrative", &json);
                                                    }
                                                }
                                                Err(e) => warn!(error = %e, "agency narrative failed"),
                                            }
                                        }
                                        nietzsche_agency::AgencyIntent::TriggerSemanticGc { archetype_id, redundant_ids } => {
                                            info!(archetype = %archetype_id, redundant = redundant_ids.len(), "agency: semantic GC triggered");
                                        }
                                        nietzsche_agency::AgencyIntent::EvolveLSystemRules { strategy, reason } => {
                                            info!(strategy = ?strategy, reason = %reason, "agency: evolving L-System rules");
                                            // Load evolution state, advance generation, persist
                                            if let Ok(mut state) = nietzsche_agency::RuleEvolution::load_state(db.storage()) {
                                                state.generation += 1;
                                                state.last_strategy = format!("{:?}", strategy);
                                                let _ = nietzsche_agency::RuleEvolution::save_state(db.storage(), &state);
                                            }
                                            // Apply evolved rules via L-System tick
                                            let evolved = nietzsche_agency::RuleEvolution::evolve_rules(&strategy, 0);
                                            let rules: Vec<_> = evolved.iter().map(|er| {
                                                match er.rule_type {
                                                    nietzsche_agency::EvolvedRuleType::GrowthChild =>
                                                        nietzsche_lsystem::ProductionRule::growth_child(&er.name, er.max_generation),
                                                    nietzsche_agency::EvolvedRuleType::LateralAssociation =>
                                                        nietzsche_lsystem::ProductionRule::lateral_association(&er.name, er.max_generation),
                                                    nietzsche_agency::EvolvedRuleType::PruneFading =>
                                                        nietzsche_lsystem::ProductionRule::prune_fading(&er.name, er.energy_threshold),
                                                    nietzsche_agency::EvolvedRuleType::EnergyBoost { .. } =>
                                                        nietzsche_lsystem::ProductionRule::prune_fading(&er.name, er.energy_threshold),
                                                }
                                            }).collect();
                                            if !rules.is_empty() {
                                                drop(db);
                                                let mut db_w = shared.write().await;
                                                let lsystem = nietzsche_lsystem::LSystemEngine::new(rules);
                                                match lsystem.tick(&mut *db_w) {
                                                    Ok(r) => info!(spawned = r.nodes_spawned, pruned = r.nodes_pruned, "evolved L-System tick complete"),
                                                    Err(e) => warn!(error = %e, "evolved L-System tick failed"),
                                                }
                                                break;
                                            }
                                        }
                                        nietzsche_agency::AgencyIntent::ExecuteNQL { node_id, nql, description } => {
                                            info!(
                                                collection = %col_name,
                                                node_id    = %node_id,
                                                desc       = %description,
                                                "agency: executing reflexive action (Code-as-Data)"
                                            );
                                            match nietzsche_query::parse(&nql) {
                                                Ok(query) => {
                                                    match nietzsche_query::execute(&query, db.storage(), db.adjacency(), &Default::default()) {
                                                        Ok(_) => {
                                                            // Record firing to trigger cooldown (write-back to node content)
                                                            if let Err(e) = nietzsche_agency::code_as_data::record_firing(db.storage(), node_id) {
                                                                warn!(node_id = %node_id, error = %e, "agency: failed to record firing");
                                                            }
                                                        }
                                                        Err(e) => warn!(node_id = %node_id, error = %e, "agency: reflexive NQL failed"),
                                                    }
                                                }
                                                Err(e) => warn!(node_id = %node_id, error = %e, "agency: reflexive NQL parse failed"),
                                            }
                                        }
                                        nietzsche_agency::AgencyIntent::ApplyLtd { from_id, to_id, weight_delta, correction_count } => {
                                            info!(
                                                weight_delta = weight_delta,
                                                count = correction_count,
                                                "agency: ApplyLtd intent executed"
                                            );
                                            
                                            // Find the specific edge ID between from_id and to_id
                                            let edge_id = db.adjacency().entries_out(&from_id)
                                                .into_iter()
                                                .find(|e| e.neighbor_id == to_id)
                                                .map(|e| e.edge_id);

                                            if let Some(eid) = edge_id {
                                                drop(db);
                                                {
                                                    let mut db_w = shared.write().await;
                                                    if let Ok(Some(mut edge)) = db_w.storage().get_edge(&eid) {
                                                        edge.weight -= weight_delta;
                                                        if edge.weight <= 1e-4 {
                                                            // Pruning: weight reached zero
                                                            if let Err(e) = db_w.delete_edge(eid) {
                                                                warn!(edge_id = %eid, error = %e, "agency: LTD pruning failed");
                                                            } else {
                                                                info!(edge_id = %eid, "agency: LTD pruning complete (weight=0)");
                                                            }
                                                        } else if let Err(e) = db_w.storage().put_edge(&edge) {
                                                            warn!(from = %from_id, to = %to_id, error = %e, "agency: ApplyLtd update failed");
                                                        }
                                                    }
                                                }
                                                db = shared.read().await;
                                            }
                                        }
                                        nietzsche_agency::AgencyIntent::NeuralBoost { node_id, importance, reason } => {
                                            info!(
                                                collection = %col_name,
                                                node_id    = %node_id,
                                                importance = importance,
                                                reason     = %reason,
                                                "agency: NeuralBoost intent executed"
                                            );
                                            if let Ok(Some(mut meta)) = db.storage().get_node_meta(&node_id) {
                                                meta.energy = (meta.energy + 0.1).min(1.0);
                                                if let Err(e) = db.storage().put_node_meta(&meta) {
                                                    warn!(node_id = %node_id, error = %e, "agency: NeuralBoost update failed");
                                                }
                                            }
                                        }
                                    }
                                }

                                if !report.desires.is_empty() {
                                    info!(
                                        collection   = %col_name,
                                        desires      = report.desires.len(),
                                        top_priority = report.desires.first().map(|d| d.priority).unwrap_or(0.0),
                                        "Motor de Desejo: desires available"
                                    );
                                }
                            }
                            Err(e) => warn!(collection = %col_name, error = %e, "Agency tick failed"),
                        }
                    }
                }
            });
        } else {
            info!("Agency engine disabled (AGENCY_TICK_SECS=0)");
        }
    }

    // ── CDC broadcaster ───────────────────────────────────────────────────────
    let cdc = Arc::new(CdcBroadcaster::new(4096));

    // ── Prometheus metrics ─────────────────────────────────────────────────────
    let ops = Arc::new(OperationMetrics::new());

    // ── Cluster ───────────────────────────────────────────────────────────────
    let cluster = if config.cluster_enabled {
        let role = cluster_service::parse_role(&config.cluster_role);
        let local_addr = format!("[::]:{}", config.port);
        let (registry, local_id) = cluster_service::build_registry(
            &config.cluster_node_name,
            &local_addr,
            role,
            &config.cluster_seeds,
        );
        info!(
            node_name  = %config.cluster_node_name,
            role       = %config.cluster_role,
            seeds      = %config.cluster_seeds,
            peer_count = registry.len(),
            "cluster mode enabled"
        );
        cluster_service::start_heartbeat(registry.clone(), local_id, 30);
        cluster_service::start_gossip_loop(registry.clone(), local_id, 15);
        Some(registry)
    } else {
        info!("cluster mode disabled (NIETZSCHE_CLUSTER_ENABLED=false)");
        None
    };

    // ── HTTP dashboard ────────────────────────────────────────────────────────
    if config.dashboard_port > 0 {
        let cm_dash      = Arc::clone(&cm);
        let ops_dash     = Arc::clone(&ops);
        let port         = config.dashboard_port;
        let cluster_dash = cluster.clone();
        tokio::spawn(async move { dashboard::serve(cm_dash, ops_dash, port, cluster_dash).await });
    } else {
        info!("dashboard disabled (NIETZSCHE_DASHBOARD_PORT=0)");
    }

    // ── gRPC RBAC authentication ─────────────────────────────────────────────
    let interceptor = auth::AuthInterceptor::from_env();

    // ── gRPC server ───────────────────────────────────────────────────────────
    let addr: SocketAddr = format!("[::]:{}", config.port).parse()?;
    let mut server = NietzscheServer::new(Arc::clone(&cm), Arc::clone(&cdc));
    if let Some(ref registry) = cluster {
        server = server.with_cluster_registry(registry.clone());
    }

    // NOTE: gRPC reflection is enabled for tooling compatibility.
    // In production, restrict access via network policies / firewall rules.
    let reflection = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(nietzsche_api::NIETZSCHE_DESCRIPTOR)
        .build()
        .expect("failed to build gRPC reflection service");

    info!(%addr, "gRPC server listening");

    let db_service = server.into_service();
    let authed_service = tonic::service::interceptor::InterceptedService::new(
        db_service,
        interceptor,
    );

    tokio::select! {
        result = tonic::transport::Server::builder()
            .add_service(reflection)
            .add_service(authed_service)
            .serve(addr) =>
        {
            if let Err(e) = result {
                error!(error = %e, "gRPC server error");
                return Err(anyhow::anyhow!("{}", e));
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("received SIGINT — shutting down gracefully");
        }
    }

    info!("NietzscheDB shutdown complete");
    Ok(())
}
