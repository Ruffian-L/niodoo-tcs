use std::env;

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::EnvFilter;

use niodoo_real_integrated::config::CliArgs;
use niodoo_real_integrated::config::RuntimeConfig;
use niodoo_real_integrated::embedding::QwenStatefulEmbedder;

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,niodoo_real_integrated=debug,tcs-ml=debug"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .init();
}

fn main() -> Result<()> {
    init_tracing();

    tracing::info!("Starting embed_debug binary");

    // Load same runtime config as full pipeline so we reuse paths
    let args = CliArgs::parse();
    let config = RuntimeConfig::load(&args).context("failed to load RuntimeConfig")?;
    tracing::info!(
        model = %config.embedding_model_name,
        "Loaded RuntimeConfig for embedding test"
    );

    // Resolve model path from env (fallback to RuntimeConfig if needed)
    let model_path = env::var("QWEN_MODEL_PATH")
        .or_else(|_| env::var("ONNX_EMBED_MODEL_PATH"))
        .unwrap_or_else(|_| config.embedding_model_name.clone());

    tracing::info!(
        model_path = %model_path,
        "Using ONNX embedding model path for debug run"
    );

    let expected_dim: usize = env::var("NIODOO_EMBED_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2560);

    tracing::info!(
        expected_dim,
        "Initializing QwenStatefulEmbedder for debug run"
    );

    let embedder = QwenStatefulEmbedder::new(&model_path, expected_dim)
        .context("failed to initialize QwenStatefulEmbedder")?;

    let prompt = env::args()
        .nth(1)
        .unwrap_or_else(|| "test embedding".to_string());
    tracing::info!(prompt = %prompt, "Running single embedding call");

    let rt = tokio::runtime::Runtime::new().context("failed to create Tokio runtime")?;
    let embedding = rt
        .block_on(async { embedder.embed(&prompt).await })
        .context("embedding call failed")?;

    tracing::info!(dim = embedding.len(), "Embedding call succeeded");

    println!("First 8 dims: {:?}", &embedding[..embedding.len().min(8)]);

    Ok(())
}
