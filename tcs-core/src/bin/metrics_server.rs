//! Metrics server for TCS monitoring
//! Exposes Prometheus metrics on port 9091

use axum::{routing::get, Router};
use prometheus::{Encoder, TextEncoder};
use std::net::SocketAddr;
use tokio;
use tcs_core::metrics::{get_registry, init_metrics};

#[tokio::main]
async fn main() {
    // Initialize metrics
    init_metrics();
    
    println!("🚀 Starting TCS metrics server on :9091");
    
    // Create router
    let app = Router::new().route("/metrics", get(handler));
    
    // Bind to port 9091
    let addr = SocketAddr::from(([0, 0, 0, 0], 9091));
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("Failed to bind to :9091");
    
    println!("✅ Metrics endpoint available at http://localhost:9091/metrics");
    
    axum::serve(listener, app).await.expect("Server failed to start");
}

async fn handler() -> String {
    let encoder = TextEncoder::new();
    let registry = get_registry();
    let metric_families = registry.gather();
    
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

