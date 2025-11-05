use anyhow::Result;
use clap::Parser;
use niodoo_integrated::{NiodooIntegrated, PipelineResult, types::Args};
use prometheus::{Encoder, TextEncoder};
use tracing::info;
use tracing_subscriber::fmt;

#[tokio::main]
async fn main() -> Result<()> {
    fmt::init();

    let args = Args::parse();

    info!("Initializing NIODOO Integrated Pipeline...");
    let mut pipeline = NiodooIntegrated::new().await?;

    info!("Processing prompt: {}", args.prompt);
    let result = pipeline.process_pipeline(&args.prompt).await?;

    // Output results
    match args.output.as_str() {
        "csv" => {
            println!("response,entropy,is_threat,is_healing,latency,learning_events");
            println!("\"{}\",{},{},{},{},\"{:?}\"",
                result.response.replace("\"", "\"\""),
                result.entropy,
                result.is_threat,
                result.is_healing,
                result.latency,
                result.learning_events
            );
        }
        "json" => {
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        _ => {
            println!("{:?}", result);
        }
    }

    // Prometheus metrics endpoint (simple print for now)
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;
    println!("\nMetrics:\n{}", String::from_utf8(buffer)?);

    Ok(())
}
