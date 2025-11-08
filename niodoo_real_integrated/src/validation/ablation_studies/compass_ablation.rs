//! Compass Ablation Study
//!
//! Tests the impact of disabling consciousness compass on decision-making.

use super::{AblationConfig, AblationMetrics, AblationResult};
use crate::config::CliArgs;
use crate::pipeline::Pipeline;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;
use tracing::info;

/// Run compass ablation study
pub async fn run_compass_ablation(
    config: AblationConfig,
    test_prompts: Vec<String>,
    output_dir: PathBuf,
) -> anyhow::Result<AblationResult> {
    info!("Running compass ablation study with {} prompts", test_prompts.len());

    let mut cli_args = CliArgs::default();
    cli_args.no_compass = true;

    let pipeline = Arc::new(AsyncMutex::new(Pipeline::initialise(cli_args).await?));

    let mut metrics = AblationMetrics {
        rouge_scores: Vec::new(),
        latency_ms: Vec::new(),
        memory_retrieval_accuracy: None,
        learning_rate: None,
        response_quality: Vec::new(),
    };

    let mut pipeline_guard = pipeline.lock().await;
    for prompt in test_prompts {
        let start = std::time::Instant::now();
        match pipeline_guard.process_prompt(&prompt).await {
            Ok(cycle) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                metrics.latency_ms.push(latency);

                metrics.rouge_scores.push(cycle.generation.rouge_score);

                if let Some(quality) = cycle.generation.curator_quality {
                    metrics.response_quality.push(quality);
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to process prompt in compass ablation");
            }
        }
    }

    Ok(AblationResult {
        config,
        metrics,
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

