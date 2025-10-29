//! Federated Runtime Integration Test
//!
//! This binary demonstrates integration of ShardClient with the pipeline runtime,
//! fetching topology signatures from telemetry endpoints and surfacing FluxMetrics.

use anyhow::Result;
use niodoo_real_integrated::config::CliArgs;
use niodoo_real_integrated::federated::{
    FederatedResilienceOrchestrator, FluxMetrics, NodalDiagnostics, ShardClient,
};
use niodoo_real_integrated::pipeline::Pipeline;
use std::time::Duration;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting Federated Runtime Integration Test");

    // Initialize pipeline
    let args = CliArgs {
        hardware: niodoo_real_integrated::config::HardwareProfile::H200,
        prompt: None,
        prompt_file: None,
        swarm: 1,
        iterations: 1,
        output: niodoo_real_integrated::config::OutputFormat::Csv,
        config: None,
        rng_seed_override: None,
    };

    let mut pipeline = Pipeline::initialise(args).await?;
    info!("Pipeline initialized");

    // Initialize telemetry client
    let telemetry_endpoint = std::env::var("TELEMETRY_ENDPOINT")
        .unwrap_or_else(|_| "http://localhost:50051".to_string());

    info!(endpoint = %telemetry_endpoint, "Connecting to telemetry server");

    match ShardClient::connect_with_timeout(&telemetry_endpoint, Duration::from_secs(5)).await {
        Ok(client) => {
            info!("Successfully connected to telemetry server");

            // Fetch shard signatures
            let endpoints = vec![
                "shard-1".to_string(),
                "shard-2".to_string(),
                "shard-3".to_string(),
            ];

            match client.fetch_shard_signatures(&endpoints).await {
                Ok(signatures) => {
                    info!(count = signatures.len(), "Retrieved shard signatures");

                    // Aggregate topology
                    let flux_coeff = pipeline.aggregate_topology(&signatures);
                    info!(flux_coeff, "Aggregated topology flux coefficient");

                    // Generate synthetic telemetry metrics
                    let mut diagnostics = NodalDiagnostics::new();
                    let sample_gaps: Vec<f64> =
                        signatures.iter().map(|sig| sig.spectral_gap).collect();

                    // Create mock telemetry batch
                    use niodoo_real_integrated::federated::proto::{FluxTrace, FluxTraceBatch};
                    use prost::Message;

                    let batch = FluxTraceBatch {
                        shard_id: "test-shard".to_string(),
                        traces: vec![
                            FluxTrace {
                                value: 0.75,
                                timestamp_ms: chrono::Utc::now().timestamp_millis(),
                            },
                            FluxTrace {
                                value: 0.82,
                                timestamp_ms: chrono::Utc::now().timestamp_millis() + 100,
                            },
                        ],
                    };

                    let proto_bytes = batch.encode_to_vec();
                    let metrics = diagnostics.merge_shard_metrics(&proto_bytes, &sample_gaps)?;

                    print_flux_metrics(&metrics);

                    // Record metrics in monitoring stack
                    record_metrics(&metrics);
                }
                Err(e) => {
                    warn!(error = %e, "Failed to fetch shard signatures");
                }
            }
        }
        Err(e) => {
            warn!(error = %e, "Failed to connect to telemetry server");
            info!("Continuing without telemetry integration");
        }
    }

    // Process a test prompt
    let prompt = "How do I manage stress during deadlines?";
    info!(prompt, "Processing test prompt");

    let cycle = pipeline.process_prompt(prompt).await?;

    info!("Cycle completed");
    info!("Baseline: {}", cycle.baseline_response);
    info!("Hybrid: {}", cycle.hybrid_response);
    info!("Latency: {}ms", cycle.latency_ms);
    info!("Entropy: {}", cycle.entropy);
    info!("ROUGE: {}", cycle.rouge);

    Ok(())
}

fn print_flux_metrics(metrics: &FluxMetrics) {
    info!("=== Flux Metrics ===");
    info!("Fused Flux: {:.4}", metrics.fused_flux);
    info!("Shard Count: {}", metrics.shard_count);
    info!("Interquartile Range: {:.4}", metrics.interquartile_range);
    info!("Median Gap: {:.4}", metrics.median_gap);
    if let Some(ref shard_id) = metrics.shard_id {
        info!("Shard ID: {}", shard_id);
    }
    if let Some(timestamp) = metrics.latest_timestamp {
        info!(
            "Latest Timestamp: {}",
            timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        );
    }
}

fn record_metrics(metrics: &FluxMetrics) {
    use prometheus::{Gauge, IntGauge};

    lazy_static::lazy_static! {
        static ref FUSED_FLUX: Gauge = Gauge::new("shard_fused_flux", "Fused flux metric").unwrap();
        static ref SHARD_COUNT: IntGauge = IntGauge::new("shard_count", "Number of shards").unwrap();
        static ref IQR: Gauge = Gauge::new("shard_iqr", "Interquartile range").unwrap();
        static ref MEDIAN_GAP: Gauge = Gauge::new("shard_median_gap", "Median spectral gap").unwrap();
    }

    FUSED_FLUX.set(metrics.fused_flux);
    SHARD_COUNT.set(metrics.shard_count as i64);
    IQR.set(metrics.interquartile_range);
    MEDIAN_GAP.set(metrics.median_gap);

    info!("Metrics recorded to Prometheus registry");
}
