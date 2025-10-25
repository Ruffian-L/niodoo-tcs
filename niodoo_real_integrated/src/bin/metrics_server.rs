use axum::{routing::get, Router};
use std::net::SocketAddr;
use tracing::info;
use niodoo_real_integrated::metrics::metrics;

async fn metrics_handler() -> String {
    // Gather current Prometheus metrics in text exposition format
    match metrics().gather() {
        Ok(text) => text,
        Err(_) => "".to_string(),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Bind address can be overridden via METRICS_ADDR env, defaults to 0.0.0.0:9095
    let bind_addr = std::env::var("METRICS_ADDR").unwrap_or_else(|_| "0.0.0.0:9095".to_string());
    let addr: SocketAddr = bind_addr.parse()?;

    let app = Router::new().route("/metrics", get(metrics_handler));

    info!("📊 metrics server listening on {}", addr);
    axum::serve(tokio::net::TcpListener::bind(&addr).await?, app).await?;
    Ok(())
}
