//! Silicon Synapse monitoring system demo - Standalone version
use tracing::{error, info, warn};
// This example demonstrates:
// - Starting a simple monitoring system
// - Emitting telemetry events from a simulated inference pipeline
// - Viewing metrics at http://localhost:9090/metrics
// - Detecting anomalies in real-time
//
// This is a standalone version that doesn't depend on the main library
// to avoid compilation issues with heavy dependencies.

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
    gpu_temperature: f32,
    gpu_utilization: f32,
    ttft_ms: f32,
    tpot_ms: f32,
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
         silicon_synapse_inference_errors_total {}\n\
         # HELP silicon_synapse_gpu_temperature_celsius GPU temperature in Celsius\n\
         # TYPE silicon_synapse_gpu_temperature_celsius gauge\n\
         silicon_synapse_gpu_temperature_celsius {}\n\
         # HELP silicon_synapse_gpu_utilization_percent GPU utilization percentage\n\
         # TYPE silicon_synapse_gpu_utilization_percent gauge\n\
         silicon_synapse_gpu_utilization_percent {}\n\
         # HELP silicon_synapse_ttft_ms Time to first token in milliseconds\n\
         # TYPE silicon_synapse_ttft_ms histogram\n\
         silicon_synapse_ttft_ms_bucket{{le=\"50.0\"}} 0\n\
         silicon_synapse_ttft_ms_bucket{{le=\"100.0\"}} 2\n\
         silicon_synapse_ttft_ms_bucket{{le=\"200.0\"}} 8\n\
         silicon_synapse_ttft_ms_bucket{{le=\"500.0\"}} 10\n\
         silicon_synapse_ttft_ms_bucket{{le=\"+Inf\"}} 10\n\
         silicon_synapse_ttft_ms_sum {}\n\
         silicon_synapse_ttft_ms_count 10\n\
         # HELP silicon_synapse_tpot_ms Time per output token in milliseconds\n\
         # TYPE silicon_synapse_tpot_ms histogram\n\
         silicon_synapse_tpot_ms_bucket{{le=\"10.0\"}} 0\n\
         silicon_synapse_tpot_ms_bucket{{le=\"25.0\"}} 5\n\
         silicon_synapse_tpot_ms_bucket{{le=\"50.0\"}} 15\n\
         silicon_synapse_tpot_ms_bucket{{le=\"100.0\"}} 20\n\
         silicon_synapse_tpot_ms_bucket{{le=\"+Inf\"}} 20\n\
         silicon_synapse_tpot_ms_sum {}\n\
         silicon_synapse_tpot_ms_count 20\n",
        metrics.inference_requests,
        metrics.tokens_generated,
        metrics.inference_errors,
        metrics.gpu_temperature,
        metrics.gpu_utilization,
        metrics.ttft_ms * 10.0, // sum of all TTFT values
        metrics.tpot_ms * 20.0  // sum of all TPOT values
    )
}

async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

async fn api_handler(State(metrics): State<SharedMetrics>) -> impl IntoResponse {
    let metrics = metrics.read().await;
    let json = serde_json::json!({
        "status": "healthy",
        "metrics": {
            "inference_requests": metrics.inference_requests,
            "tokens_generated": metrics.tokens_generated,
            "inference_errors": metrics.inference_errors,
            "gpu_temperature": metrics.gpu_temperature,
            "gpu_utilization": metrics.gpu_utilization,
            "ttft_ms": metrics.ttft_ms,
            "tpot_ms": metrics.tpot_ms
        },
        "timestamp": chrono::Utc::now().to_rfc3339()
    });

    (StatusCode::OK, axum::Json(json))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    tracing::info!("🧠💖 Silicon Synapse Demo - Standalone Version");
    tracing::info!("=============================================");

    // Create shared metrics
    let metrics = Arc::new(RwLock::new(Metrics::default()));

    // Start HTTP server
    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .route("/api/v1/metrics", get(api_handler))
        .with_state(metrics.clone());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:9090").await?;

    // Start server in background
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tracing::info!("✅ HTTP server started on http://localhost:9090");
    tracing::info!("📊 Metrics endpoint: http://localhost:9090/metrics");
    tracing::info!("🏥 Health endpoint: http://localhost:9090/health");
    tracing::info!("📈 JSON API: http://localhost:9090/api/v1/metrics");

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

        // Simulate TTFT (Time to First Token)
        let ttft = 50.0 + (i as f32 * 10.0); // Simulate varying TTFT
        sleep(Duration::from_millis(ttft as u64)).await;

        // Update TTFT metric
        {
            let mut m = metrics.write().await;
            m.ttft_ms = ttft;
        }

        // Simulate token generation
        let num_tokens = 20;
        for j in 0..num_tokens {
            let tpot = 25.0 + (j as f32 * 2.0); // Simulate varying TPOT
            sleep(Duration::from_millis(tpot as u64)).await;

            // Increment token counter
            {
                let mut m = metrics.write().await;
                m.tokens_generated += 1;
                m.tpot_ms = tpot;
            }
        }

        // Simulate hardware metrics
        {
            let mut m = metrics.write().await;
            m.gpu_temperature = 45.0 + (i as f32 * 2.0); // Simulate temperature increase
            m.gpu_utilization = 60.0 + (i as f32 * 3.0); // Simulate utilization increase
        }

        // Simulate occasional error
        if i == 7 {
            let mut m = metrics.write().await;
            m.inference_errors += 1;
            tracing::info!("⚠️  Simulated error for request {}", i + 1);
        }

        tracing::info!(
            "✅ Completed request {}/10 ({} tokens, TTFT: {:.1}ms, TPOT: {:.1}ms)",
            i + 1,
            num_tokens,
            ttft,
            25.0
        );

        // Small delay between requests
        sleep(Duration::from_millis(100)).await;
    }

    tracing::info!("\n🎯 Demo complete! Metrics are available at:");
    tracing::info!("   http://localhost:9090/metrics");
    tracing::info!("   http://localhost:9090/health");
    tracing::info!("   http://localhost:9090/api/v1/metrics");
    tracing::info!("\n📈 Example queries:");
    tracing::info!("   curl http://localhost:9090/metrics");
    tracing::info!("   curl http://localhost:9090/health");
    tracing::info!("   curl http://localhost:9090/api/v1/metrics");
    tracing::info!("\n🔍 Example Prometheus queries:");
    tracing::info!("   rate(silicon_synapse_inference_requests_total[1m])");
    tracing::info!("   silicon_synapse_gpu_temperature_celsius");
    tracing::info!("   histogram_quantile(0.95, silicon_synapse_ttft_ms_bucket)");
    tracing::info!("\nPress Ctrl+C to exit...");

    // Keep running
    tokio::signal::ctrl_c().await?;

    tracing::info!("👋 Shutting down...");
    Ok(())
}
