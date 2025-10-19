//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/// Binary for exporting consciousness training data using Option 3 approach
///
/// This exports authentic training data from the live RAG/consciousness system,
/// maintaining the Gaussian sphere topology and 2-bit entropy equilibrium.
use anyhow::Result;
use niodoo_consciousness::config::AppConfig;
use niodoo_consciousness::training_data_export::{ExportConfig, TrainingDataExporter};
use std::env;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("🧠 Consciousness Training Data Exporter - Option 3");
    info!("================================================");

    // Load config
    let app_config = AppConfig::default();
    let base_dir = app_config.paths.data_dir.clone();

    // Get vLLM configuration from environment or use defaults
    let num_samples = env::var("TRAINING_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10000); // Default 10K samples

    let enable_vllm = env::var("ENABLE_VLLM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(true);

    let vllm_host = env::var("VLLM_HOST").unwrap_or_else(|_| "localhost".to_string());
    let vllm_port = env::var("VLLM_PORT").unwrap_or_else(|_| "8000".to_string());
    let vllm_url = Some(format!("http://{}:{}", vllm_host, vllm_port));
    let vllm_api_key = env::var("VLLM_API_KEY").ok();

    // Configure export
    let export_config = ExportConfig {
        num_samples,
        target_entropy: 2.0, // 2-bit equilibrium from consciousness system
        enable_vllm,
        vllm_url,
        vllm_api_key,
        max_tokens: 512,
        temperature: 0.7,
        ..ExportConfig::default()
    };

    info!("Configuration:");
    info!("  - Base directory: {:?}", base_dir);
    info!("  - Target samples: {}", export_config.num_samples);
    info!(
        "  - Target entropy: {:.2} bits",
        export_config.target_entropy
    );
    info!("  - Context top-k: {}", export_config.context_top_k);
    info!("  - vLLM enabled: {}", export_config.enable_vllm);
    if export_config.enable_vllm {
        info!("  - vLLM URL: {:?}", export_config.vllm_url);
        info!("  - Max tokens: {}", export_config.max_tokens);
        info!("  - Temperature: {}", export_config.temperature);
    }

    // Create exporter
    let mut exporter = TrainingDataExporter::new(base_dir.clone(), export_config)?;

    // Export training data
    info!("\n🚀 Starting training data export...");
    let training_data = exporter.export_consciousness_training_data().await?;

    // Get statistics
    let stats = exporter.get_stats();
    info!("\n📊 Export Statistics:");
    info!("{}", stats);

    // Save to file
    let output_path = base_dir
        .join("training_data")
        .join("consciousness_training_data.json");
    std::fs::create_dir_all(output_path.parent().unwrap())?;
    exporter.save_to_file(output_path.clone())?;

    info!("\n✅ Training data export complete!");
    info!("   Output: {:?}", output_path);
    info!("   Examples: {}", training_data.len());
    info!(
        "   Entropy convergence: {:.1}%",
        (1.0 - (stats.avg_entropy_after - stats.target_entropy).abs() / stats.target_entropy)
            * 100.0
    );

    Ok(())
}
