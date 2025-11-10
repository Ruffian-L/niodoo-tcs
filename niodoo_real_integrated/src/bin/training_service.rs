//! Training Service Binary
//!
//! Standalone service that starts the HTTP server, initializes workers,
//! and handles graceful shutdown.

#[cfg(feature = "svc")]
use anyhow::{Context, Result};
#[cfg(feature = "svc")]
use clap::Parser;
#[cfg(feature = "svc")]
use niodoo_real_integrated::training_service::{
    AdapterStorage, JobQueue, TrainingServiceServer, TrainingWorker,
};
#[cfg(feature = "svc")]
use std::sync::Arc;
#[cfg(feature = "svc")]
use std::path::PathBuf;
#[cfg(feature = "svc")]
use tokio::signal;
#[cfg(feature = "svc")]
use tracing::{info, warn};

#[cfg(feature = "svc")]
#[derive(Parser, Debug)]
#[command(author, version, about = "Niodoo Training Service")]
struct Args {
    /// Port to listen on
    #[arg(long, default_value_t = 8001)]
    port: u16,

    /// Job queue directory
    #[arg(long, default_value = "data/training_queue")]
    queue_dir: PathBuf,

    /// Adapter storage directory
    #[arg(long, default_value = "models/system2_adapters")]
    storage_dir: PathBuf,

    /// Number of worker threads
    #[arg(long, default_value_t = 1)]
    workers: usize,
}

#[cfg(feature = "svc")]
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Starting Niodoo Training Service");
    info!(port = args.port, "Configuration");
    info!(queue_dir = %args.queue_dir.display(), "Job queue directory");
    info!(storage_dir = %args.storage_dir.display(), "Adapter storage directory");
    info!(workers = args.workers, "Worker threads");

    // Initialize job queue
    let job_queue = Arc::new(
        JobQueue::new(&args.queue_dir)
            .context("Failed to create job queue")?,
    );

    // Initialize adapter storage
    let adapter_storage = Arc::new(
        AdapterStorage::new(&args.storage_dir)
            .context("Failed to create adapter storage")?,
    );

    // Start worker threads
    let mut worker_handles = Vec::new();
    for i in 0..args.workers {
        let worker = TrainingWorker::new(
            Arc::clone(&job_queue),
            Arc::clone(&adapter_storage),
        );
        let handle = tokio::spawn(async move {
            info!(worker_id = i, "Starting worker");
            worker.run().await;
            info!(worker_id = i, "Worker stopped");
        });
        worker_handles.push(handle);
    }

    // Start HTTP server
    let server = TrainingServiceServer::new(
        Arc::clone(&job_queue),
        Arc::clone(&adapter_storage),
        args.port,
    );

    // Handle shutdown signals
    let shutdown = async {
        let ctrl_c = async {
            signal::ctrl_c()
                .await
                .expect("Failed to install Ctrl+C handler");
            info!("Received Ctrl+C signal");
        };

        #[cfg(unix)]
        let terminate = async {
            signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("Failed to install signal handler")
                .recv()
                .await;
            info!("Received SIGTERM signal");
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }
    };

    // Run server until shutdown
    tokio::select! {
        result = server.start() => {
            if let Err(e) = result {
                error!(error = %e, "Server error");
            }
        }
        _ = shutdown => {
            info!("Shutdown signal received, stopping server");
        }
    }

    // Wait for workers to finish
    info!("Waiting for workers to finish...");
    for handle in worker_handles {
        let _ = handle.await;
    }

    info!("Training service stopped");
    Ok(())
}

#[cfg(not(feature = "svc"))]
fn main() {
    eprintln!("Training service requires the 'svc' feature to be enabled");
    eprintln!("Build with: cargo build --features svc --bin training_service");
    std::process::exit(1);
}

