//! Load Generator
//!
//! Generates diverse prompts for scale testing.

use super::{ScaleMetrics, ScaleTestResult};
use crate::config::CliArgs;
use crate::pipeline::Pipeline;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

/// Generate load and collect metrics
pub async fn generate_load(
    target_interactions: usize,
    prompt_pool: Vec<String>,
) -> anyhow::Result<ScaleTestResult> {
    info!("Generating load: {} interactions", target_interactions);

    let cli_args = CliArgs::default();
    let pipeline = Arc::new(Mutex::new(Pipeline::initialise(cli_args).await?));

    let mut rouge_scores = Vec::new();
    let mut latencies = Vec::new();

    for i in 0..target_interactions {
        let prompt = &prompt_pool[i % prompt_pool.len()];
        let start = std::time::Instant::now();
        
        let mut pipeline_guard = pipeline.lock().await;
        match pipeline_guard.process_prompt(prompt).await {
            Ok(cycle) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                latencies.push(latency);
                rouge_scores.push(cycle.generation.rouge_score);
            }
            Err(e) => {
                tracing::warn!(iteration = i, error = %e, "Failed to process prompt");
            }
        }

        if i % 100 == 0 {
            info!("Progress: {}/{} interactions", i, target_interactions);
        }
    }

    // Calculate improvement rate (trend in ROUGE scores)
    let improvement_rate = if rouge_scores.len() > 100 {
        let early_avg: f64 = rouge_scores[..100].iter().sum::<f64>() / 100.0;
        let late_avg: f64 = rouge_scores[rouge_scores.len().saturating_sub(100)..]
            .iter().sum::<f64>() / 100.0;
        late_avg - early_avg
    } else {
        0.0
    };

    // Calculate stability (inverse of variance)
    let mean_rouge = rouge_scores.iter().sum::<f64>() / rouge_scores.len() as f64;
    let variance: f64 = rouge_scores.iter()
        .map(|r| (r - mean_rouge).powi(2))
        .sum::<f64>() / rouge_scores.len() as f64;
    let stability_score = 1.0 / (1.0 + variance);

    let metrics = ScaleMetrics {
        rouge_scores,
        latency_ms: latencies,
        memory_usage_mb: 0.0, // Would measure actual memory
        improvement_rate,
        stability_score,
    };

    Ok(ScaleTestResult {
        interaction_count: target_interactions,
        metrics,
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

