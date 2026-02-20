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
use nietzsche_zaratustra::ZaratustraEngine;

#[cfg(feature = "gpu")]
use nietzsche_hnsw_gpu::GpuVectorStore;

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

    // Log collection inventory
    for col in cm.list() {
        info!(
            collection = %col.name,
            dim        = col.dim,
            metric     = %col.metric,
            nodes      = col.node_count,
            edges      = col.edge_count,
            "collection loaded"
        );
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

    // ── Sleep scheduler — runs against the "default" collection ───────────────
    if config.sleep_interval_secs > 0 {
        let interval  = Duration::from_secs(config.sleep_interval_secs);
        let sleep_cfg = SleepConfig {
            noise:               config.sleep_noise,
            adam_steps:          config.sleep_adam_steps,
            adam_lr:             5e-3,
            hausdorff_threshold: config.hausdorff_threshold,
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
                        hausdorff_before = r.hausdorff_before,
                        hausdorff_after  = r.hausdorff_after,
                        hausdorff_delta  = r.hausdorff_delta,
                        committed        = r.committed,
                        nodes_perturbed  = r.nodes_perturbed,
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

    // ── gRPC authentication ──────────────────────────────────────────────────
    let api_key = std::env::var("NIETZSCHE_API_KEY").ok();
    let interceptor = auth::AuthInterceptor::new(api_key);

    // ── gRPC server ───────────────────────────────────────────────────────────
    let addr: SocketAddr = format!("[::]:{}", config.port).parse()?;
    let server = NietzscheServer::new(Arc::clone(&cm), Arc::clone(&cdc));

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
