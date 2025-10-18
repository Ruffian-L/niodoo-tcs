//! Silicon Synapse monitoring system demo
use tracing::{error, info, warn};
// This example demonstrates:
// - Starting the monitoring system
// - Emitting telemetry events from a simulated inference pipeline
// - Viewing metrics at http://localhost:9090/metrics
// - Detecting anomalies in real-time

use std::time::{Duration, Instant};
use tokio::time::sleep;
use uuid::Uuid;

// Simple HTTP server for metrics
use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::get, Router};
use std::sync::Arc;
use tokio::sync::RwLock;

// Simple metrics storage
#[derive(Default)]
struct Metrics {
    inference_requests: u64,
    tokens_generated: u64,
    inference_errors: u64,
}

type SharedMetrics = Arc<RwLock<Metrics>>;

async fn metrics_handler(State(metrics): State<SharedMetrics>) -> impl IntoResponse {
    let metrics = metrics.read().await;
    format!(
        "# HELP silicon_synapse_inference_requests_total Total number of inference requests\n\
         # TYPE silicon_synapse_inference_requests_total counter\n\
         silicon_synapse_inference_requests_total {}\n\
         # HELP silicon_synapse_tokens_generated_total Total number of tokens generated\n\
         # TYPE silicon_synapse_tokens_generated_total counter\n\
         silicon_synapse_tokens_generated_total {}\n\
         # HELP silicon_synapse_inference_errors_total Total number of inference errors\n\
         # TYPE silicon_synapse_inference_errors_total counter\n\
         silicon_synapse_inference_errors_total {}\n",
        metrics.inference_requests, metrics.tokens_generated, metrics.inference_errors
    )
}

async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    tracing::info!("🧠💖 Silicon Synapse Demo - Simplified Version");
    tracing::info!("=============================================");

    // Create shared metrics
    let metrics = Arc::new(RwLock::new(Metrics::default()));

    // Start HTTP server
    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .with_state(metrics.clone());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:9090").await?;

    // Start server in background
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tracing::info!("✅ HTTP server started on http://localhost:9090");
    tracing::info!("📊 Metrics endpoint: http://localhost:9090/metrics");
    tracing::info!("🏥 Health endpoint: http://localhost:9090/health");

    // Simulate inference workload
    tracing::info!("\n🎭 Simulating inference workload...");

    for i in 0..10 {
        let request_id = Uuid::new_v4();
        tracing::info!("📤 Starting inference request {} ({})", i + 1, request_id);

        // Increment request counter
        {
            let mut m = metrics.write().await;
            m.inference_requests += 1;
        }

        // Simulate token generation
        let num_tokens = 20;
        for j in 0..num_tokens {
            sleep(Duration::from_millis(50)).await;

            // Increment token counter
            {
                let mut m = metrics.write().await;
                m.tokens_generated += 1;
            }
        }

        // Simulate occasional error
        if i == 7 {
            let mut m = metrics.write().await;
            m.inference_errors += 1;
            tracing::info!("⚠️  Simulated error for request {}", i + 1);
        }

        tracing::info!("✅ Completed request {}/10 ({} tokens)", i + 1, num_tokens);

        // Small delay between requests
        sleep(Duration::from_millis(100)).await;
    }

    tracing::info!("\n🎯 Demo complete! Metrics are available at:");
    tracing::info!("   http://localhost:9090/metrics");
    tracing::info!("   http://localhost:9090/health");
    tracing::info!("\n📈 Example queries:");
    tracing::info!("   curl http://localhost:9090/metrics");
    tracing::info!("   curl http://localhost:9090/health");
    tracing::info!("\nPress Ctrl+C to exit...");

    // Keep running
    tokio::signal::ctrl_c().await?;

    tracing::info!("👋 Shutting down...");
    Ok(())
}
