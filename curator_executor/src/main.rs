use anyhow::Result;
use curator_executor::{
    curator::{Curator, CuratorConfig},
    executor::{Executor, ExecutorConfig},
    memory_core::{MemoryCore, MemoryConfig},
    learning::{LearningLoop, LearningConfig},
};
use tokio;
use tracing::{info, warn, error};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🚀 STARTING CURATOR-EXECUTOR SYSTEM");
    info!("QDRANT_URL: {}", std::env::var("QDRANT_URL").unwrap_or_else(|_| "not set".to_string()));
    info!("VLLM_ENDPOINT: {}", std::env::var("VLLM_ENDPOINT").unwrap_or_else(|_| "not set".to_string()));
    info!("Connecting to services...");
    
    // Initialize memory core with Qdrant
    info!("📡 Connecting to Qdrant...");
    let memory = Arc::new(MemoryCore::new(Default::default()).await?);
    info!("✅ Qdrant connection established!");

    // Initialize Curator (0.5B model for efficiency)
    info!("🧠 Initializing Curator...");
    let curator_config = CuratorConfig::default();
    let curator = Arc::new(tokio::sync::Mutex::new(Curator::new(curator_config)?));
    info!("✅ Curator initialized!");

    // Initialize Executor (7B model for power)
    info!("⚡ Initializing Executor...");
    let executor_config = ExecutorConfig::default();
    let executor = Arc::new(Executor::new(executor_config)?);
    info!("✅ Executor initialized!");

    // Create learning loop
    let learning_config = LearningConfig::default();
    let learning_loop = LearningLoop::new(learning_config);
    info!("✅ Learning loop initialized!");

    info!("🔄 SYSTEM IS LIVE - BEGINNING CONTINUOUS LEARNING");
    info!("Press Ctrl+C to stop");
    info!("{}", "=".repeat(60));

    // Task list for continuous processing
    let tasks = vec![
        "Write a Python function to calculate fibonacci numbers",
        "Explain quantum computing in simple terms",
        "Create a Rust web server with axum",
        "Implement a binary search tree in C++",
        "Write a neural network from scratch in Python",
        "Explain the CAP theorem with examples",
        "Create a distributed cache system design",
        "Implement RAFT consensus algorithm",
        "Write a compiler for a simple language",
        "Design a real-time chat system",
    ];

    let mut task_index = 0;

    // Main processing loop
    loop {
        let task = &tasks[task_index % tasks.len()];
        task_index += 1;

        info!("📝 Task #{}: {}", task_index, task);

        // Execute task with memory context
        let curator_data;
        {
            let curator_guard = curator.lock().await;
            curator_data = (*curator_guard).clone(); // or extract needed fields
        }
        match executor.execute_task(task, "general", Some(&curator_data), Some(&memory)).await {
            Ok(result) => {
                info!("✅ Task completed successfully!");
                info!("   Success score: {:.2}%", result.success_score * 100.0);
                info!("   Execution time: {}ms", result.execution_time_ms);
                
                let preview = if result.output.len() > 200 {
                    format!("{}...", &result.output[..200])
                } else {
                    result.output.clone()
                };
                info!("   Output preview: {}", preview);

                // Store experience via curator
                let experience = curator_executor::memory_core::Experience::new(
                    task.to_string(),
                    result.output.clone(),
                    format!("Task execution #{}", task_index),
                    "general".to_string(),
                    result.success_score,
                );

                if let Err(e) = curator_guard.process_experience(experience, &memory).await {
                    warn!("Failed to process experience: {}", e);
                } else {
                    info!("   📚 Experience processed and stored");
                }

                // Log experience via executor
                if let Err(e) = executor.log_experience(
                    task,
                    &result,
                    "general_task",
                    "",
                    &mut *curator_guard,
                    &memory,
                ).await {
                    warn!("Failed to log experience via executor: {}", e);
                }
            }
            Err(e) => {
                error!("❌ Task execution failed: {}", e);
            }
        }

        // Periodically distill knowledge
        if task_index % 5 == 0 {
            info!("{}", "=".repeat(60));
            info!("📚 KNOWLEDGE DISTILLATION PHASE");

            let mut curator_guard = curator.lock().await;
            match curator_guard.distill_knowledge(&memory, 3).await {
                Ok(distilled) => {
                    if distilled.is_empty() {
                        info!("   No new knowledge to distill yet");
                    } else {
                        info!("   ✅ Distilled {} examples from experience clusters", distilled.len());

                        // Check if we should trigger fine-tuning
                        if let Ok(should_fine_tune) = learning_loop.record_experience().await {
                            if should_fine_tune && !distilled.is_empty() {
                                info!("   🎯 Triggering fine-tuning!");

                                match learning_loop.trigger_fine_tuning(
                                    distilled,
                                    "models/curator"
                                ).await {
                                    Ok(job_id) => {
                                        info!("   📊 Fine-tuning job started: {}", job_id);
                                    }
                                    Err(e) => {
                                        warn!("   Fine-tuning failed to start: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("   Knowledge distillation failed: {}", e);
                }
            }
            drop(curator_guard); // Release lock
            info!("{}", "=".repeat(60));
        }

        // Check memory statistics every 3 tasks
        if task_index % 3 == 0 {
            match memory.get_stats().await {
                Ok(stats) => {
                    let total: u64 = stats.values().sum();
                    info!("📊 Memory Statistics:");
                    info!("   Total vectors stored: {}", total);
                    for (key, count) in stats.iter() {
                        info!("   - {}: {} vectors", key, count);
                    }
                }
                Err(e) => {
                    warn!("Could not get memory stats: {}", e);
                }
            }
        }

        // Wait before next task
        info!("⏳ Waiting 15 seconds before next task...");
        tokio::time::sleep(tokio::time::Duration::from_secs(15)).await;
        info!("");
    }
}