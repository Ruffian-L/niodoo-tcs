//! Standard RAG Baseline
//!
//! Implements standard RAG (Qdrant + embeddings only, no topology) for comparison.

use super::{BenchmarkMetrics, BenchmarkResult};
use crate::config::CliArgs;
use crate::pipeline::Pipeline;
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;
use tracing::info;

/// Run standard RAG baseline
pub async fn run_rag_baseline(
    test_prompts: Vec<String>,
) -> anyhow::Result<BenchmarkResult> {
    info!("Running standard RAG baseline with {} prompts", test_prompts.len());

    let mut cli_args = CliArgs::default();
    cli_args.no_topology = true;
    cli_args.no_compass = true;
    cli_args.no_learning = true;

    let pipeline = Arc::new(AsyncMutex::new(Pipeline::initialise(cli_args).await?));

    let mut rouge_scores = Vec::new();
    let mut latencies = Vec::new();

    let mut pipeline_guard = pipeline.lock().await;
    for prompt in test_prompts {
        let start = std::time::Instant::now();
        match pipeline_guard.process_prompt(&prompt).await {
            Ok(cycle) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                latencies.push(latency);
                rouge_scores.push(cycle.generation.rouge_score);
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to process prompt in RAG baseline");
            }
        }
    }

    let metrics = BenchmarkMetrics {
        accuracy: 0.0, // Would calculate from task-specific accuracy
        latency_ms: latencies.iter().sum::<f64>() / latencies.len() as f64,
        rouge_score: rouge_scores.iter().sum::<f64>() / rouge_scores.len() as f64,
        memory_usage_mb: 0.0, // Would measure actual memory
    };

    Ok(BenchmarkResult {
        system_name: "standard_rag".to_string(),
        test_suite: "general".to_string(),
        metrics,
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

