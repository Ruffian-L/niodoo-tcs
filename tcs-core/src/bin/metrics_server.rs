// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Metrics server for TCS monitoring
//! Exposes Prometheus metrics on a configurable port (default 9091, fallback 9092)

use axum::{Router, routing::get};
use prometheus::{Encoder, TextEncoder};
use std::{env, net::SocketAddr};
use tcs_core::metrics::{get_registry, init_metrics};
use tokio;

#[tokio::main]
async fn main() {
    // Initialize metrics
    init_metrics();

    let requested_port = env::var("TCS_METRICS_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(9091);

    // Create router
    let app = Router::new().route("/metrics", get(handler));

    // Attempt to bind to the requested port, falling back to 9092 when in use.
    let (listener, bound_port) = match bind_listener(requested_port).await {
        Ok(pair) => pair,
        Err(error) => {
            eprintln!(
                "⚠️  Failed to bind metrics server to :{requested_port} ({error}). Attempting fallback :9092"
            );
            bind_listener(9092)
                .await
                .expect("Failed to bind metrics server to fallback port :9092")
        }
    };

    println!("✅ Metrics endpoint available at http://localhost:{bound_port}/metrics");

    axum::serve(listener, app)
        .await
        .expect("Server failed to start");
}

async fn bind_listener(port: u16) -> Result<(tokio::net::TcpListener, u16), std::io::Error> {
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let bound_port = listener.local_addr()?.port();
    println!("🚀 Starting TCS metrics server on :{bound_port}");
    Ok((listener, bound_port))
}

async fn handler() -> String {
    let encoder = TextEncoder::new();
    let registry = get_registry();
    let metric_families = registry.gather();

    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}
