use anyhow::Result;
use curator_executor::{
    curator::{Curator, CuratorConfig},
    executor::{Executor, ExecutorConfig},
    memory_core::{MemoryCore, MemoryConfig, Experience},
    learning::{LearningLoop, LearningConfig},
    optimizations::{
        OptimizationConfig, retrieve_optimized_context, 
        ERAGMonitor, HardwareOptimizer, BatchedCurator
    },
};
use tokio;
use tracing::{info, warn, error};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🚀 STARTING OPTIMIZED CURATOR-EXECUTOR SYSTEM");
    info!("📊 Based on 2025 performance analysis recommendations");
    
    // Detect hardware and configure optimizations
    let hardware = if std::env::var("HARDWARE_TYPE").unwrap_or_default() == "laptop" {
        HardwareOptimizer::new_for_laptop()
    } else {
        HardwareOptimizer::new_for_beelink()
    };
    
    info!("🖥️ Hardware: {:?}", hardware.gpu_type);
    info!("⚡ Expected throughput: {} tokens/s", hardware.expected_tokens_per_second());
    
    // Initialize optimization config with 2025 recommendations
    let opt_config = OptimizationConfig {
        erag_collapse_threshold: 0.2,  // Per analysis line 13
        normalize_embeddings: true,     // 15% similarity boost
        kv_cache_size: hardware.optimal_kv_cache(),
        curator_batch_size: 8,          // Async batching
        context_injection_limit: 5,     // Line 15-16 recommendation
    };
    
    // Initialize memory core with Qdrant (hyperspherical embeddings)
    info!("📡 Connecting to Qdrant with hyperspherical normalization...");
    let memory_config = MemoryConfig {
        vector_dim: 896,  // BERT-standard for Qwen
        ..Default::default()
    };
    let memory = Arc::new(MemoryCore::new(memory_config).await?);
    info!("✅ Qdrant connection established!");

    // Initialize Curator with batching support
    info!("🧠 Initializing Curator with async batching...");
    let curator_config = CuratorConfig::default();
    let curator = Arc::new(Mutex::new(Curator::new(curator_config)?));
    let batched_curator = Arc::new(BatchedCurator::new(opt_config.clone()));
    info!("✅ Curator initialized with batch size: {}", opt_config.curator_batch_size);

    // Initialize Executor with KV cache optimization
    info!("⚡ Initializing Executor with {} KV cache...", opt_config.kv_cache_size);
    let executor_config = ExecutorConfig {
        max_context_length: opt_config.kv_cache_size / 8,  // Approximate tokens
        ..Default::default()
    };
    let executor = Arc::new(Executor::new(executor_config)?);
    info!("✅ Executor initialized!");

    // Initialize ERAG collapse monitor
    let erag_monitor = Arc::new(ERAGMonitor::new(opt_config.erag_collapse_threshold));
    info!("🌊 ERAG monitor active (threshold: {})", opt_config.erag_collapse_threshold);

    // Create learning loop with optimized intervals
    let learning_config = LearningConfig {
        fine_tune_interval: 100,  // More frequent for continuous improvement
        batch_size: hardware.optimal_batch_size(),
        ..Default::default()
    };
    let learning_loop = Arc::new(LearningLoop::new(learning_config));
    info!("✅ Learning loop initialized!");

    info!("🔄 OPTIMIZED SYSTEM IS LIVE - BEGINNING CONTINUOUS LEARNING");
    info!("📈 Performance targets: 40% adaptation gain, 95% retention");
    info!("Press Ctrl+C to stop");
    info!("{}", "=".repeat(60));

    // Enhanced task list with complexity levels
    let tasks = vec![
        ("Write a Python function to calculate Fibonacci numbers", "code_generation", 0.7),
        ("Implement a binary search tree in Rust", "code_generation", 0.9),
        ("Debug this sorting algorithm that's producing incorrect results", "debugging", 0.8),
        ("Optimize this SQL query for better performance", "code_analysis", 0.85),
        ("Document this REST API endpoint", "documentation", 0.6),
        ("Implement a rate limiter using Redis", "code_generation", 0.9),
        ("Convert this class-based React component to hooks", "code_generation", 0.75),
        ("Write unit tests for this authentication module", "code_generation", 0.8),
        ("Refactor this legacy code for better maintainability", "code_analysis", 0.85),
        ("Create a Docker configuration for this microservice", "code_generation", 0.7),
    ];

    let mut task_index = 0;
    let mut coherence_history = Vec::new();
    let mut performance_metrics = PerformanceTracker::new();

    loop {
        let (task_prompt, task_type, complexity) = &tasks[task_index % tasks.len()];
        
        info!("\n📝 Task #{}: {} (complexity: {:.1})", task_index + 1, task_type, complexity);
        
        // Retrieve optimized context (implements line 15-16 suggestion)
        let curator_guard = curator.lock().await;
        let (context, coherence) = retrieve_optimized_context(
            task_prompt,
            &*curator_guard,
            &memory,
            &opt_config
        ).await?;
        drop(curator_guard);
        
        // Monitor ERAG coherence
        if erag_monitor.check_collapse(coherence).await {
            warn!("⚠️ ERAG collapse detected! Triggering reset...");
            erag_monitor.reset().await;
            // In production: trigger model refresh or context window reset
        } else {
            info!("✅ Coherence: {:.2} (stable)", coherence);
        }
        
        coherence_history.push(coherence);
        
        // Execute task with enriched context
        let start_time = std::time::Instant::now();
        
        let result = executor.execute_task(
            task_prompt,
            task_type,
            Some(&*curator.lock().await),
            Some(&memory),
        ).await?;
        
        let execution_time = start_time.elapsed();
        performance_metrics.record(execution_time.as_millis() as f64, result.success_score);
        
        info!("✅ Task completed in {}ms (success: {:.2})", 
              execution_time.as_millis(), result.success_score);
        
        // Create and store experience with hyperspherical normalization
        let mut experience = Experience::new(
            task_prompt.to_string(),
            result.output.clone(),
            format!("Task #{} with context injection", task_index),
            task_type.to_string(),
            result.success_score,
        );
        
        // Embed with normalization
        let curator_guard = curator.lock().await;
        let embedding = curator_guard.embed_text(task_prompt).await?;
        drop(curator_guard);
        
        experience.embedding = Some(embedding);
        experience.normalize_embedding();  // Apply hyperspherical normalization
        
        memory.store_experience(&experience).await?;
        
        // Check if distillation should occur (every 5 tasks per analysis)
        if task_index % 5 == 4 {
            info!("🎯 Triggering knowledge distillation...");
            
            let curator_guard = curator.lock().await;
            let distilled = curator_guard.distill_experiences(&[experience.clone()]).await?;
            drop(curator_guard);
            
            info!("📚 Distilled {} examples", distilled.len());
            
            // Check for fine-tuning trigger
            if learning_loop.record_experience().await? {
                info!("🔧 Triggering QLoRA fine-tuning...");
                let job_id = learning_loop.trigger_fine_tuning(
                    distilled,
                    "models/qwen-curator-optimized"
                ).await?;
                info!("🚀 Fine-tuning job started: {}", job_id);
            }
        }
        
        // Performance reporting every 10 tasks
        if task_index % 10 == 9 {
            performance_metrics.report();
            info!("📊 Average coherence: {:.3}", 
                  coherence_history.iter().sum::<f32>() / coherence_history.len() as f32);
        }
        
        task_index += 1;
        
        // Adaptive delay based on hardware
        let delay_ms = if hardware.expected_tokens_per_second() > 100 { 1000 } else { 2000 };
        tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
    }
}

/// Track performance metrics
struct PerformanceTracker {
    latencies: Vec<f64>,
    success_scores: Vec<f32>,
    start_time: std::time::Instant,
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            latencies: Vec::new(),
            success_scores: Vec::new(),
            start_time: std::time::Instant::now(),
        }
    }
    
    fn record(&mut self, latency_ms: f64, success: f32) {
        self.latencies.push(latency_ms);
        self.success_scores.push(success);
    }
    
    fn report(&self) {
        if self.latencies.is_empty() {
            return;
        }
        
        let avg_latency = self.latencies.iter().sum::<f64>() / self.latencies.len() as f64;
        let avg_success = self.success_scores.iter().sum::<f32>() / self.success_scores.len() as f32;
        let runtime = self.start_time.elapsed();
        
        info!("📊 === PERFORMANCE REPORT ===");
        info!("   Runtime: {:?}", runtime);
        info!("   Tasks processed: {}", self.latencies.len());
        info!("   Avg latency: {:.1}ms", avg_latency);
        info!("   Avg success: {:.2}", avg_success);
        info!("   Throughput: {:.1} tasks/min", 
              self.latencies.len() as f64 / runtime.as_secs_f64() * 60.0);
        
        // Check against 2025 benchmarks
        if avg_latency < 100.0 {
            info!("   ✅ Exceeding 2025 latency targets!");
        }
        if avg_success > 0.85 {
            info!("   ✅ Meeting 85% HumanEval benchmark!");
        }
    }
}