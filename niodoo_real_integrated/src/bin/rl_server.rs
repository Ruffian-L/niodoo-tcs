//! RL Execution Harness HTTP Server
//!
//! Starts the HTTP API server for the RL training harness.
//! Python RL trainer calls POST /rl/evaluate to evaluate generated code.

use anyhow::{Context, Result};
use axum::{
    http::StatusCode,
    routing::get,
    Router,
};
use niodoo_real_integrated::sandbox::{manager::SandboxManager, SecurityPolicy};
use std::sync::Arc;
use tracing::info;

#[cfg(feature = "svc")]
use niodoo_real_integrated::rl_harness::server::{create_harness, create_rl_harness_router};

async fn health_check() -> (StatusCode, &'static str) {
    (StatusCode::OK, "RL Harness Server is healthy")
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!("🚀 Starting RL Execution Harness Server...");

    #[cfg(not(feature = "svc"))]
    {
        eprintln!("ERROR: rl_server requires --features svc");
        eprintln!("Run with: cargo run --bin rl_server --features svc");
        std::process::exit(1);
    }

    #[cfg(feature = "svc")]
    {
        // Create sandbox manager with default security policy
        let security_policy = SecurityPolicy::default();
        let sandbox_manager = Arc::new(
            SandboxManager::new(security_policy)
                .context("Failed to create sandbox manager")?
        );
        info!("✅ SandboxManager initialized");

        // Create execution harness (no TCS analyzer for now - can be added later)
        let harness = create_harness(sandbox_manager, None)
            .await
            .context("Failed to create execution harness")?;
        info!("✅ ExecutionHarness initialized");

        // Create HTTP router with health check
        let app = Router::new()
            .route("/health", get(health_check))
            .merge(create_rl_harness_router(harness));
        info!("✅ HTTP router created");

        // Bind server to port 8080
        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], 8080));
        info!("🌐 Binding to {}", addr);

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .context("Failed to bind to port 8080")?;

        info!("✅ RL Harness Server running on http://{}", addr);
        info!("📡 Endpoints:");
        info!("   GET  /health");
        info!("   POST /rl/evaluate");
        info!("Ready to receive code evaluation requests from Python RL trainer");

        // Start server
        axum::serve(listener, app)
            .await
            .context("Server error")?;
    }

    Ok(())
}
