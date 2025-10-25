use std::net::SocketAddr;
use std::time::Duration;

use axum::{routing::get, Router};
use clap::Parser;
use niodoo_real_integrated::config::CliArgs;
use niodoo_real_integrated::metrics::metrics;
use niodoo_real_integrated::pipeline::Pipeline;
use tokio::time::sleep;
use tracing::{info, warn};
use tracing_subscriber::{fmt, EnvFilter};

async fn metrics_handler() -> String {
    match metrics().gather() {
        Ok(text) => text,
        Err(_) => String::new(),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Init logging
    let env_filter = EnvFilter::try_new(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = fmt().with_env_filter(env_filter).try_init();

    // Start metrics server
    let bind_addr: SocketAddr = std::env::var("METRICS_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:9095".to_string())
        .parse()?;
    let app = Router::new().route("/metrics", get(metrics_handler));
    tokio::spawn(async move {
        info!("📊 metrics server listening on {}", bind_addr);
        if let Err(err) = axum::serve(tokio::net::TcpListener::bind(&bind_addr).await.unwrap(), app).await {
            warn!(?err, "metrics server exited");
        }
    });

    // Build pipeline
    let args = CliArgs::parse();
    let mut pipeline = Pipeline::initialise(args.clone()).await?;

    // Run forever over prompts
    loop {
        let prompts = pipeline.rut_prompts();
        for prompt in prompts {
            let _ = pipeline.process_prompt(&prompt.text).await;
            // throttle between prompts a bit
            sleep(Duration::from_millis(100)).await;
        }
        // short pause between passes
        sleep(Duration::from_secs(2)).await;
    }
}
