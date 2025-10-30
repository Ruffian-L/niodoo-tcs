//! Standalone Prometheus metrics server for the Niodoo pipeline
//! Exposes /metrics endpoint on port 9091 for Prometheus scraping

use anyhow::Result;
use axum::{Router, http::StatusCode, response::IntoResponse, routing::get};
use tokio::signal;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};

use niodoo_real_integrated::metrics::metrics;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Initialize TCS metrics
    tcs_core::metrics::init_metrics();

    info!("🚀 Starting Niodoo Metrics Server");
    info!("📊 Prometheus endpoint: http://localhost:9091/metrics");
    info!("🏥 Health check: http://localhost:9091/health");

    // Build router
    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .route("/", get(root_handler))
        .layer(
            ServiceBuilder::new()
                .layer(CorsLayer::permissive())
                .into_inner(),
        );

    // Start server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:9091").await?;
    info!("✅ Metrics server listening on 0.0.0.0:9091");

    // Run server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn metrics_handler() -> impl IntoResponse {
    match metrics().gather() {
        Ok(metrics_text) => (StatusCode::OK, metrics_text),
        Err(e) => {
            warn!("Failed to gather metrics: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Error gathering metrics: {}", e),
            )
        }
    }
}

async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

async fn root_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        r#"
<h1>🚀 Niodoo TCS Metrics Server</h1>
<p>Real-time metrics for self-learning AI consciousness system</p>
<ul>
  <li><a href="/metrics">Prometheus Metrics</a></li>
  <li><a href="/health">Health Check</a></li>
</ul>
<p><strong>Technologies:</strong> Qwen2.5-7B-Instruct-AWQ (Chinese model), Topological Data Analysis, Persistent Homology, Knot Theory</p>
<p><strong>Learning:</strong> LoRA fine-tuning with proper backpropagation, SGD with momentum, gradient clipping</p>
"#,
    )
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Ctrl+C received, shutting down gracefully");
        },
        _ = terminate => {
            info!("SIGTERM received, shutting down gracefully");
        },
    }

    info!("🛑 Metrics server shutting down gracefully");
}
