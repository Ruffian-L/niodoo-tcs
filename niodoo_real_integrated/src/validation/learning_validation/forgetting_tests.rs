//! Catastrophic Forgetting Tests
//!
//! Measures retention on previous tasks after learning new ones.

use super::LearningValidationResult;
use crate::config::CliArgs;
use crate::pipeline::Pipeline;
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;
use tracing::info;

/// Test catastrophic forgetting
pub async fn test_forgetting(
    initial_tasks: Vec<(String, f64)>, // (task, initial_accuracy)
    new_tasks: Vec<String>,
) -> anyhow::Result<LearningValidationResult> {
    info!("Testing catastrophic forgetting with {} initial tasks, {} new tasks", 
        initial_tasks.len(), new_tasks.len());

    let cli_args = CliArgs::default();
    let pipeline = Arc::new(AsyncMutex::new(Pipeline::initialise(cli_args).await?));

    // Measure initial accuracy on tasks
    let mut initial_accuracies = Vec::new();
    let mut pipeline_guard = pipeline.lock().await;
    for (task, _expected_accuracy) in &initial_tasks {
        match pipeline_guard.process_prompt(task).await {
            Ok(cycle) => {
                initial_accuracies.push(cycle.generation.rouge_score);
            }
            Err(_) => {}
        }
    }

    let initial_avg = initial_accuracies.iter().sum::<f64>() / initial_accuracies.len() as f64;

    // Learn new tasks
    for task in new_tasks {
        let _ = pipeline_guard.process_prompt(&task).await;
    }

    // Measure accuracy on original tasks again
    let mut final_accuracies = Vec::new();
    for (task, _) in &initial_tasks {
        match pipeline_guard.process_prompt(task).await {
            Ok(cycle) => {
                final_accuracies.push(cycle.generation.rouge_score);
            }
            Err(_) => {}
        }
    }

    let final_avg = final_accuracies.iter().sum::<f64>() / final_accuracies.len() as f64;

    let forgetting_rate = if initial_avg > 0.0 {
        (initial_avg - final_avg) / initial_avg
    } else {
        0.0
    };

    Ok(LearningValidationResult {
        test_name: "forgetting_test".to_string(),
        forgetting_rate,
        improvement_rate: 0.0,
        breakthrough_precision: None,
        safety_score_delta: None,
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

