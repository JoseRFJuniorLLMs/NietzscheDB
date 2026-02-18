//! NietzscheDB production gRPC server.
//!
//! Reads configuration from environment variables (see [`config::Config`]),
//! opens (or creates) the RocksDB + WAL data directory, starts an optional
//! background sleep-cycle scheduler, then serves the gRPC API until SIGINT.
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
//!   cargo run --bin nietzsche-server --release
//! ```

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, EnvFilter};

use nietzsche_api::NietzscheServer;
use nietzsche_graph::{MockVectorStore, NietzscheDB};
use nietzsche_sleep::{SleepConfig, SleepCycle};

mod config;
mod dashboard;
mod html;
use config::Config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── Tracing ──────────────────────────────────────────────────────────────
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
        version   = env!("CARGO_PKG_VERSION"),
        data_dir  = %config.data_dir,
        port      = config.port,
        "NietzscheDB starting"
    );

    // ── Open DB ───────────────────────────────────────────────────────────────
    let db_path = PathBuf::from(&config.data_dir);
    std::fs::create_dir_all(&db_path)?;

    let db = NietzscheDB::open(&db_path, MockVectorStore::default())
        .map_err(|e| anyhow::anyhow!("failed to open DB at {}: {e}", config.data_dir))?;

    info!(
        nodes = db.node_count().unwrap_or(0),
        edges = db.edge_count().unwrap_or(0),
        "database opened"
    );

    let shared_db = Arc::new(Mutex::new(db));

    // ── Sleep scheduler (optional background task) ─────────────────────────
    if config.sleep_interval_secs > 0 {
        let interval  = Duration::from_secs(config.sleep_interval_secs);
        let sleep_cfg = SleepConfig {
            noise:               config.sleep_noise,
            adam_steps:          config.sleep_adam_steps,
            adam_lr:             5e-3,
            hausdorff_threshold: config.hausdorff_threshold,
        };
        let db_clone = Arc::clone(&shared_db);

        tokio::spawn(async move {
            info!(interval_secs = config.sleep_interval_secs, "sleep scheduler started");
            loop {
                tokio::time::sleep(interval).await;

                let mut db  = db_clone.lock().await;
                let seed: u64 = rand::random();
                let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);

                match SleepCycle::run(&sleep_cfg, &mut *db, &mut rng) {
                    Ok(r) => info!(
                        hausdorff_before  = r.hausdorff_before,
                        hausdorff_after   = r.hausdorff_after,
                        hausdorff_delta   = r.hausdorff_delta,
                        committed         = r.committed,
                        nodes_perturbed   = r.nodes_perturbed,
                        "sleep cycle complete"
                    ),
                    Err(e) => warn!(error = %e, "sleep cycle failed"),
                }
            }
        });
    } else {
        info!("sleep scheduler disabled (NIETZSCHE_SLEEP_INTERVAL_SECS=0)");
    }

    // ── HTTP dashboard ────────────────────────────────────────────────────────
    if config.dashboard_port > 0 {
        let db_dash = Arc::clone(&shared_db);
        let port    = config.dashboard_port;
        tokio::spawn(async move { dashboard::serve(db_dash, port).await });
    } else {
        info!("dashboard disabled (NIETZSCHE_DASHBOARD_PORT=0)");
    }

    // ── gRPC server ───────────────────────────────────────────────────────────
    let addr: SocketAddr = format!("[::]:{}", config.port).parse()?;
    let server           = NietzscheServer::new(Arc::clone(&shared_db));

    let reflection = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(nietzsche_api::NIETZSCHE_DESCRIPTOR)
        .build()
        .expect("failed to build gRPC reflection service");

    info!(%addr, "gRPC server listening");

    tokio::select! {
        result = tonic::transport::Server::builder()
            .add_service(reflection)
            .add_service(server.into_service())
            .serve(addr) =>
        {
            if let Err(e) = result {
                error!(error = %e, "gRPC server error");
                return Err(anyhow::anyhow!(e));
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("received SIGINT — shutting down gracefully");
        }
    }

    info!("NietzscheDB shutdown complete");
    Ok(())
}
