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
mod bootstrap;
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
    let collections_path = db_path.join("collections");
    info!("[BOOT] NIETZSCHE_DATA_DIR={}", db_path.display());
    info!("[BOOT] collections_path={}", collections_path.display());
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

    // ── Cortex Bootstrap (Phase VI.1) ───────────────────────────────────────
    match bootstrap::run_bootstrap(&cm).await {
        Ok(true)  => info!("[BOOT] cortex bootstrap complete: 5 nodes, 4 edges, 1 daemon"),
        Ok(false) => info!("[BOOT] cortex already bootstrapped, skipping"),
        Err(e)    => warn!("[BOOT] cortex bootstrap failed: {e}"),
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
                        let mut db = shared.read().await;
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
                        // FIX #3: Skip cache-only and ephemeral collections
                        const AGENCY_SKIP: &[&str] = &[
                            "eva_cache", "eva_perceptions", "speaker_embeddings",
                            "eva_sensory", "lobby_", "test_",
                        ];
                        if AGENCY_SKIP.iter().any(|p| col_name.starts_with(p)) {
                            continue;
                        }

                        let Some(shared) = cm_agency.get_or_default(col_name) else {
                            continue;
                        };

                        let mut db = shared.read().await;
                        // Skip tiny collections (< 10 nodes) UNLESS they are EVA-critical
                        const AGENCY_ALWAYS: &[&str] = &[
                            "memories", "signifier_chains", "eva_mind",
                            "eva_core", "patient_graph",
                        ];
                        let is_critical = AGENCY_ALWAYS.iter().any(|c| col_name == *c);
                        if !is_critical && db.node_count().unwrap_or(0) < 10 {
                            continue;
                        }

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

                                // Persist cognitive dashboard + observation frame (before intents consume the report)
                                {
                                    let dashboard = nietzsche_agency::CognitiveDashboard::from_tick_report(&report, col_name);
                                    if let Ok(json) = serde_json::to_vec(&dashboard) {
                                        let _ = db.storage().put_meta(nietzsche_agency::CognitiveDashboard::meta_key(), &json);
                                    }
                                    // Observation frame for perspektive.js visualization
                                    let obs_frame = nietzsche_agency::ObservationFrame::from_dashboard(&dashboard, &[]);
                                    if let Ok(json) = serde_json::to_vec(&obs_frame) {
                                        let _ = db.storage().put_meta(nietzsche_agency::ObservationFrame::meta_key(), &json);
                                    }
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
                                            // Guard: skip L-System for huge collections.
                                            // With CUDA feature, GPU batch Hausdorff handles large collections efficiently.
                                            // Without CUDA, CPU Hausdorff is O(n²) — keep conservative limit.
                                            #[cfg(feature = "gpu")]
                                            const LSYSTEM_MAX_NODES: usize = 200_000;
                                            #[cfg(not(feature = "gpu"))]
                                            const LSYSTEM_MAX_NODES: usize = 50_000;
                                            let nc = db.node_count().unwrap_or(0);
                                            if nc > LSYSTEM_MAX_NODES {
                                                warn!(collection = %col_name, node_count = nc, limit = LSYSTEM_MAX_NODES,
                                                    "L-System SKIPPED: collection too large for Hausdorff computation");
                                                break;
                                            }
                                            info!(collection = %col_name, reason = %reason, node_count = nc, "agency: triggering L-System growth");
                                            drop(db);
                                            let mut db_w = shared.write().await;
                                            let mut lsystem = nietzsche_lsystem::LSystemEngine::new(vec![
                                                nietzsche_lsystem::ProductionRule::growth_child("agency-grow", 3),
                                                nietzsche_lsystem::ProductionRule::lateral_association("agency-assoc", 3),
                                                // prune_fading REMOVED — cortical disconnection syndrome fix
                                            ]);
                                            // Disable Hausdorff auto-prune: nodes with D=0.0 must NOT be killed
                                            lsystem.hausdorff_lo = 0.0;
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
                                                // Guard: skip L-System for huge collections.
                                                // GPU batch Hausdorff via CUDA supports much larger collections.
                                                #[cfg(feature = "gpu")]
                                                const LSYSTEM_MAX_NODES: usize = 200_000;
                                                #[cfg(not(feature = "gpu"))]
                                                const LSYSTEM_MAX_NODES: usize = 50_000;
                                                let nc = db.node_count().unwrap_or(0);
                                                if nc > LSYSTEM_MAX_NODES {
                                                    warn!(collection = %col_name, node_count = nc, limit = LSYSTEM_MAX_NODES,
                                                        "evolved L-System SKIPPED: collection too large for Hausdorff computation");
                                                    break;
                                                }
                                                drop(db);
                                                let mut db_w = shared.write().await;
                                                let mut lsystem = nietzsche_lsystem::LSystemEngine::new(rules);
                                                // Disable Hausdorff auto-prune — cortical disconnection fix
                                                lsystem.hausdorff_lo = 0.0;
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
                                        nietzsche_agency::AgencyIntent::HardDelete { node_id, vitality, reason } => {
                                            info!(
                                                collection = %col_name,
                                                node_id    = %node_id,
                                                vitality   = vitality,
                                                reason     = %reason,
                                                "agency: HardDelete — Nezhmetdinov condemned node"
                                            );
                                            drop(db);
                                            let mut db_w = shared.write().await;
                                            if let Err(e) = db_w.delete_node(node_id) {
                                                warn!(node_id = %node_id, error = %e, "agency: HardDelete failed");
                                            } else {
                                                info!(node_id = %node_id, "agency: HardDelete completed");
                                            }
                                            break;
                                        }
                                        nietzsche_agency::AgencyIntent::RecordDeletion { node_id, cycle, structural_hash } => {
                                            info!(
                                                collection       = %col_name,
                                                node_id          = %node_id,
                                                cycle            = cycle,
                                                structural_hash  = %structural_hash,
                                                "agency: RecordDeletion — audit ledger entry"
                                            );
                                        }
                                        nietzsche_agency::AgencyIntent::HebbianLTP { from_id, to_id, weight_delta, trace } => {
                                            // Strengthen edge via Hebbian LTP (co-activation)
                                            let edge_id = db.adjacency().entries_out(&from_id)
                                                .into_iter()
                                                .find(|e| e.neighbor_id == to_id)
                                                .map(|e| e.edge_id);

                                            if let Some(eid) = edge_id {
                                                if let Ok(Some(mut edge)) = db.storage().get_edge(&eid) {
                                                    edge.weight = (edge.weight + weight_delta).min(5.0);
                                                    if let Err(e) = db.storage().put_edge(&edge) {
                                                        warn!(from = %from_id, to = %to_id, error = %e, "agency: HebbianLTP update failed");
                                                    }
                                                }
                                            }
                                            let _ = (trace,); // logged at trace level
                                        }
                                        nietzsche_agency::AgencyIntent::HeatFlow { from_id, to_id, amount } => {
                                            // Fourier heat flow: transfer energy from hot → cold
                                            if let Ok(Some(mut from_meta)) = db.storage().get_node_meta(&from_id) {
                                                if let Ok(Some(mut to_meta)) = db.storage().get_node_meta(&to_id) {
                                                    let old_from = from_meta.energy;
                                                    let old_to = to_meta.energy;
                                                    from_meta.energy = (from_meta.energy - amount).max(0.0);
                                                    to_meta.energy = (to_meta.energy + amount).min(1.0);
                                                    let _ = db.storage().put_node_meta_update_energy(&from_meta, old_from);
                                                    let _ = db.storage().put_node_meta_update_energy(&to_meta, old_to);
                                                }
                                            }
                                        }
                                        nietzsche_agency::AgencyIntent::GravityPull { well_id: _, node_id, amount } => {
                                            // Gravity pull: boost attracted node's energy
                                            if let Ok(Some(mut meta)) = db.storage().get_node_meta(&node_id) {
                                                let old = meta.energy;
                                                meta.energy = (meta.energy + amount).min(1.0);
                                                let _ = db.storage().put_node_meta_update_energy(&meta, old);
                                            }
                                        }
                                        // ── Phase XIX: Self-Healing intents ──────────
                                        nietzsche_agency::AgencyIntent::ReProjectNode { node_id } => {
                                            // Re-project embedding back into Poincaré ball
                                            if let Ok(Some(node)) = db.get_node(node_id) {
                                                let mut coords = node.embedding.coords.clone();
                                                let norm: f64 = coords.iter().map(|c| (*c as f64).powi(2)).sum::<f64>().sqrt();
                                                if norm > 0.99 {
                                                    let target = 0.98;
                                                    let scale = (target / norm) as f32;
                                                    for c in &mut coords {
                                                        *c *= scale;
                                                    }
                                                    let new_emb = nietzsche_graph::PoincareVector::new(coords);
                                                    drop(db);
                                                    {
                                                        let mut db_w = shared.write().await;
                                                        if let Err(e) = db_w.update_embedding(node_id, new_emb) {
                                                            warn!(node = %node_id, error = %e, "healing: re-project failed");
                                                        } else {
                                                            info!(node = %node_id, old_norm = %norm, "healing: re-projected embedding");
                                                        }
                                                    }
                                                    break;
                                                }
                                            }
                                        }
                                        nietzsche_agency::AgencyIntent::HardDeleteEdge { edge_id } => {
                                            drop(db);
                                            {
                                                let mut db_w = shared.write().await;
                                                if let Err(e) = db_w.delete_edge(edge_id) {
                                                    warn!(edge = %edge_id, error = %e, "healing: dead edge delete failed");
                                                }
                                            }
                                            break;
                                        }
                                        nietzsche_agency::AgencyIntent::Phantomize { node_id } => {
                                            drop(db);
                                            {
                                                let mut db_w = shared.write().await;
                                                if let Err(e) = db_w.phantomize_node(node_id) {
                                                    warn!(node = %node_id, error = %e, "healing: phantomize failed");
                                                }
                                            }
                                            break;
                                        }
                                        // ── Phase XVIII: Hyperbolic Contrastive Training ──────
                                        nietzsche_agency::AgencyIntent::UpdateEmbeddingBatch { updates, final_loss, epochs_run } => {
                                            if !updates.is_empty() {
                                                info!(
                                                    collection = %col_name,
                                                    updates    = updates.len(),
                                                    final_loss = final_loss,
                                                    epochs     = epochs_run,
                                                    "agency: Phase XVIII — embedding batch update"
                                                );
                                                drop(db);
                                                {
                                                    let mut db_w = shared.write().await;
                                                    let mut ok = 0usize;
                                                    let mut fail = 0usize;
                                                    for (node_id, coords_f64) in &updates {
                                                        let coords_f32: Vec<f32> = coords_f64.iter().map(|&c| c as f32).collect();
                                                        let emb = nietzsche_graph::PoincareVector::new(coords_f32);
                                                        match db_w.update_embedding(*node_id, emb) {
                                                            Ok(_) => ok += 1,
                                                            Err(e) => {
                                                                fail += 1;
                                                                if fail <= 3 {
                                                                    warn!(node = %node_id, error = %e, "Phase XVIII: embedding update failed");
                                                                }
                                                            }
                                                        }
                                                    }
                                                    info!(ok = ok, fail = fail, "Phase XVIII: embedding batch applied");
                                                }
                                                break; // db was dropped, exit intent loop
                                            }
                                        }
                                        // ── Phase B1: Temporal Edge Decay ─────
                                        nietzsche_agency::AgencyIntent::ApplyTemporalDecay {
                                            edge_id, from_id: _, to_id: _, old_weight: _, new_weight
                                        } => {
                                            drop(db);
                                            {
                                                let db_w = shared.write().await;
                                                if let Ok(Some(mut edge)) = db_w.storage().get_edge(&edge_id) {
                                                    edge.weight = new_weight;
                                                    if let Err(e) = db_w.storage().put_edge(&edge) {
                                                        warn!(edge = %edge_id, error = %e, "temporal decay: update failed");
                                                    }
                                                }
                                            }
                                            db = shared.read().await;
                                        }
                                        nietzsche_agency::AgencyIntent::PruneDecayedEdge { edge_id, effective_weight: _ } => {
                                            drop(db);
                                            {
                                                let mut db_w = shared.write().await;
                                                if let Err(e) = db_w.delete_edge(edge_id) {
                                                    warn!(edge = %edge_id, error = %e, "temporal decay: prune failed");
                                                } else {
                                                    info!(edge = %edge_id, "temporal decay: pruned decayed edge");
                                                }
                                            }
                                            db = shared.read().await;
                                        }
                                        // ── Phase C: Autonomous Graph Growth ──
                                        nietzsche_agency::AgencyIntent::ProposeEdge {
                                            from_id, to_id, distance: _, weight
                                        } => {
                                            drop(db);
                                            {
                                                let mut db_w = shared.write().await;
                                                let edge = nietzsche_graph::Edge::new(
                                                    from_id, to_id,
                                                    nietzsche_graph::EdgeType::Association,
                                                    weight,
                                                );
                                                if let Err(e) = db_w.insert_edge(edge) {
                                                    warn!(%from_id, %to_id, error = %e, "graph growth: edge insert failed");
                                                }
                                            }
                                            db = shared.read().await;
                                        }
                                        // ── Phase E: Cognitive Layer ──────────
                                        nietzsche_agency::AgencyIntent::ProposeConcept {
                                            centroid, member_ids, label, avg_distance: _
                                        } => {
                                            drop(db);
                                            {
                                                let mut db_w = shared.write().await;
                                                // Create concept node
                                                let concept_id = uuid::Uuid::new_v4();
                                                let mut content = serde_json::Map::new();
                                                content.insert("_concept".into(), serde_json::json!(true));
                                                content.insert("label".into(), serde_json::json!(label));
                                                content.insert("member_count".into(), serde_json::json!(member_ids.len()));
                                                let concept_emb = nietzsche_graph::PoincareVector::new(
                                                    centroid.iter().map(|&c| c as f32).collect()
                                                );
                                                let concept_meta = nietzsche_graph::NodeMeta {
                                                    id: concept_id,
                                                    depth: concept_emb.coords.iter().map(|c| c * c).sum::<f32>().sqrt(),
                                                    content: serde_json::Value::Object(content),
                                                    node_type: nietzsche_graph::NodeType::Concept,
                                                    energy: 0.5,
                                                    lsystem_generation: 0,
                                                    hausdorff_local: 0.0,
                                                    created_at: std::time::SystemTime::now()
                                                        .duration_since(std::time::UNIX_EPOCH)
                                                        .unwrap_or_default()
                                                        .as_secs() as i64,
                                                    expires_at: None,
                                                    metadata: std::collections::HashMap::new(),
                                                    valence: 0.0,
                                                    arousal: 0.0,
                                                    is_phantom: false,
                                                };
                                                let concept_node = nietzsche_graph::Node {
                                                    meta: concept_meta,
                                                    embedding: concept_emb,
                                                };
                                                if let Err(e) = db_w.insert_node(concept_node) {
                                                    warn!(%concept_id, error = %e, "cognitive layer: concept insert failed");
                                                } else {
                                                    // Create edges from members to concept
                                                    let mut linked = 0usize;
                                                    for member_id in &member_ids {
                                                        let edge = nietzsche_graph::Edge::new(
                                                            *member_id, concept_id,
                                                            nietzsche_graph::EdgeType::Hierarchical,
                                                            1.0,
                                                        );
                                                        if db_w.insert_edge(edge).is_ok() {
                                                            linked += 1;
                                                        }
                                                    }
                                                    info!(
                                                        %concept_id,
                                                        %label,
                                                        members = member_ids.len(),
                                                        linked,
                                                        "Cognitive Layer: concept node created"
                                                    );
                                                }
                                            }
                                            db = shared.read().await;
                                        }
                                        nietzsche_agency::AgencyIntent::ShatterNode { node_id, avatars } => {
                                            // Shatter Protocol: split super-node into context avatars
                                            // Read data under read lock first
                                            let original_node = db.get_node(node_id).ok().flatten();
                                            let original_embedding = db.storage().get_embedding(&node_id).ok().flatten();

                                            // Collect neighbor embeddings for gyromidpoint computation
                                            let mut avatar_data: Vec<(usize, nietzsche_graph::PoincareVector)> = Vec::new();
                                            if let Some(ref original) = original_node {
                                                let base_emb = original_embedding.clone().unwrap_or_else(|| {
                                                    nietzsche_graph::PoincareVector::new(vec![0.0; 128])
                                                });
                                                for (idx, avatar_plan) in avatars.iter().enumerate() {
                                                    let mut neighbor_embs: Vec<Vec<f64>> = Vec::new();
                                                    for nid in avatar_plan.neighbor_ids.iter().take(20) {
                                                        if let Ok(Some(emb)) = db.storage().get_embedding(nid) {
                                                            if !emb.coords.is_empty() {
                                                                neighbor_embs.push(emb.coords.iter().map(|&c| c as f64).collect());
                                                            }
                                                        }
                                                    }
                                                    let avatar_emb = if neighbor_embs.len() >= 2 {
                                                        let refs: Vec<&[f64]> = neighbor_embs.iter().map(|v| v.as_slice()).collect();
                                                        nietzsche_hyp_ops::gyromidpoint(&refs)
                                                            .map(|mid| nietzsche_graph::PoincareVector::new(mid.iter().map(|&c| c as f32).collect()))
                                                            .unwrap_or_else(|_| base_emb.clone())
                                                    } else {
                                                        let mut coords = base_emb.coords.clone();
                                                        let dim = coords.len().max(1);
                                                        if let Some(c) = coords.get_mut(idx % dim) {
                                                            *c = (*c + 0.01 * (idx as f32 + 1.0)).min(0.98);
                                                        }
                                                        nietzsche_graph::PoincareVector::new(coords)
                                                    };
                                                    avatar_data.push((idx, avatar_emb));
                                                }
                                            }

                                            if original_node.is_some() && !avatars.is_empty() {
                                                let original = original_node.unwrap();
                                                drop(db);
                                                {
                                                    let mut db_w = shared.write().await;
                                                    for ((idx, avatar_emb), avatar_plan) in avatar_data.into_iter().zip(avatars.iter()) {
                                                        let mut avatar_content = original.meta.content.clone();
                                                        if let serde_json::Value::Object(ref mut map) = avatar_content {
                                                            map.insert("_parent_ghost_id".into(), serde_json::json!(node_id.to_string()));
                                                            map.insert("_context_tag".into(), serde_json::json!(avatar_plan.context_tag));
                                                            map.insert("_is_avatar".into(), serde_json::json!(true));
                                                        }

                                                        let avatar_meta = nietzsche_graph::NodeMeta {
                                                            id: avatar_plan.avatar_id,
                                                            depth: avatar_emb.coords.iter().map(|c| c * c).sum::<f32>().sqrt(),
                                                            content: avatar_content,
                                                            node_type: original.meta.node_type.clone(),
                                                            energy: original.meta.energy,
                                                            lsystem_generation: original.meta.lsystem_generation,
                                                            hausdorff_local: original.meta.hausdorff_local,
                                                            created_at: std::time::SystemTime::now()
                                                                .duration_since(std::time::UNIX_EPOCH)
                                                                .unwrap_or_default()
                                                                .as_secs() as i64,
                                                            expires_at: original.meta.expires_at,
                                                            metadata: original.meta.metadata.clone(),
                                                            valence: original.meta.valence,
                                                            arousal: original.meta.arousal,
                                                            is_phantom: false,
                                                        };

                                                        let avatar_node = nietzsche_graph::Node {
                                                            meta: avatar_meta,
                                                            embedding: avatar_emb,
                                                        };

                                                        if let Err(e) = db_w.insert_node(avatar_node) {
                                                            warn!(avatar = %avatar_plan.avatar_id, error = %e, "shatter: failed to insert avatar");
                                                            continue;
                                                        }

                                                        // Create Hierarchical edge: ghost → avatar
                                                        let hier_edge = nietzsche_graph::Edge::hierarchical(node_id, avatar_plan.avatar_id);
                                                        let _ = db_w.insert_edge(hier_edge);

                                                        // Reassign edges from original to this avatar
                                                        for &edge_id in &avatar_plan.edge_ids {
                                                            if let Ok(Some(old_edge)) = db_w.get_edge(edge_id) {
                                                                let new_edge = if old_edge.from == node_id {
                                                                    nietzsche_graph::Edge::new(
                                                                        avatar_plan.avatar_id, old_edge.to,
                                                                        old_edge.edge_type.clone(), old_edge.weight,
                                                                    )
                                                                } else {
                                                                    nietzsche_graph::Edge::new(
                                                                        old_edge.from, avatar_plan.avatar_id,
                                                                        old_edge.edge_type.clone(), old_edge.weight,
                                                                    )
                                                                };
                                                                let _ = db_w.insert_edge(new_edge);
                                                                let _ = db_w.delete_edge(edge_id);
                                                            }
                                                        }
                                                    }

                                                    // Phantomize the original super-node
                                                    if let Err(e) = db_w.phantomize_node(node_id) {
                                                        warn!(node = %node_id, error = %e, "shatter: failed to phantomize");
                                                    }
                                                }

                                                info!(
                                                    node = %node_id,
                                                    avatars = avatars.len(),
                                                    "Shatter Protocol: super-node split into avatars"
                                                );
                                                break; // db was dropped, exit intent loop
                                            }
                                        }
                                        // ── Phase 27: Epistemic Mutation ──────
                                        nietzsche_agency::AgencyIntent::EpistemicMutation {
                                            mutation_type, node_ids, reason, estimated_delta
                                        } => {
                                            info!(
                                                collection = %col_name,
                                                mutation   = %mutation_type,
                                                nodes      = node_ids.len(),
                                                delta      = estimated_delta,
                                                reason     = %reason,
                                                "Phase 27: epistemic mutation proposed (logged only)"
                                            );
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
