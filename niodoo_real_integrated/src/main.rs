use anyhow::Result;
use niodoo_real_integrated::config::{CliArgs, HardwareProfile, OutputFormat};
use niodoo_real_integrated::pipeline::Pipeline;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting NIODOO pipeline smoke test");

    // Initialize pipeline with default config
    let args = CliArgs {
        hardware: HardwareProfile::H200,
        prompt: None,
        prompt_file: None,
        swarm: 1,
        iterations: 1,
        output: OutputFormat::Csv,
        config: None,
        rng_seed_override: None,
    };

    let mut pipeline = Pipeline::initialise_with_mode(args, niodoo_real_integrated::config::TopologyMode::Hybrid).await?;

    // Process sample
    let prompt = "test query";
    let cycle = pipeline.process_prompt(prompt).await?;
    info!("Smoke complete: Hybrid response: {}", cycle.hybrid_response);

    // Print brief result
    info!("✅ Smoke complete. Rouge {:.3}, Quadrant {:?}", cycle.rouge, cycle.compass.quadrant);
    Ok(())
}
