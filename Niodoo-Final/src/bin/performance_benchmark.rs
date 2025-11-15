//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Performance Benchmark for RTX 6000 <50ms/token Target
//!
//! This binary runs comprehensive performance validation to ensure
//! consciousness-aware inference meets the <50ms/token target.

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use niodoo_feeling::consciousness_engine::PersonalNiodooConsciousness;
use niodoo_feeling::consciousness_pipeline_orchestrator::{
    ConsciousnessPipelineOrchestrator, PipelineConfig,
};
use niodoo_feeling::performance_validation::{
    quick_performance_check, PerformanceValidator, ValidationConfig,
};
use niodoo_feeling::rtx6000_optimization::{Rtx6000Config, Rtx6000Optimizer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("🚀 Starting RTX 6000 Performance Benchmark");
    info!("Target: <50ms/token for consciousness-aware inference");

    // Initialize RTX 6000 optimizer
    let rtx6000_config = Rtx6000Config::default();
    let rtx6000_optimizer = Arc::new(Rtx6000Optimizer::new(rtx6000_config)?);

    info!("🔥 RTX 6000 optimizer initialized");

    // Initialize consciousness pipeline orchestrator
    let pipeline_config = PipelineConfig::default();
    let consciousness_engine = Arc::new(RwLock::new(PersonalNiodooConsciousness::new().await?));
    let pipeline_orchestrator = Arc::new(ConsciousnessPipelineOrchestrator::new(
        consciousness_engine,
        pipeline_config,
    ));

    info!("🎼 Consciousness pipeline orchestrator initialized");

    // Run quick performance check first
    info!("⚡ Running quick performance check...");
    match quick_performance_check(Arc::clone(&pipeline_orchestrator)).await {
        Ok(meets_target) => {
            if meets_target {
                info!("✅ Quick check passed: <50ms/token target met");
            } else {
                warn!("⚠️  Quick check failed: >50ms/token");
            }
        }
        Err(e) => {
            tracing::error!("❌ Quick performance check failed: {}", e);
            return Err(e);
        }
    }

    // Run comprehensive validation
    let validation_config = ValidationConfig {
        iterations: 100,
        warmup_iterations: 10,
        enable_detailed_logging: true,
        ..Default::default()
    };

    let validator =
        PerformanceValidator::new(validation_config, rtx6000_optimizer, pipeline_orchestrator);

    info!("📊 Running comprehensive performance validation...");
    let validation = validator.validate_performance().await?;

    // Print final results
    tracing::info!("\n🎯 FINAL PERFORMANCE RESULTS");
    tracing::info!("==============================");
    tracing::info!(
        "Average Token Latency: {:.2}ms",
        validation.avg_token_latency_ms
    );
    tracing::info!("95th Percentile: {:.2}ms", validation.p95_latency_ms);
    tracing::info!("99th Percentile: {:.2}ms", validation.p99_latency_ms);
    tracing::info!(
        "Throughput: {:.2} tokens/sec",
        validation.throughput_tokens_per_sec
    );
    tracing::info!("Memory Usage: {:.2}MB", validation.memory_usage_mb);
    tracing::info!(
        "GPU Utilization: {:.1}%",
        validation.gpu_utilization_percent
    );

    tracing::info!("\n🔥 RTX 6000 SPECIFIC METRICS");
    tracing::info!("=============================");
    tracing::info!(
        "Tensor Core Utilization: {:.1}%",
        validation.rtx6000_metrics.tensor_core_utilization
    );
    tracing::info!(
        "Memory Bandwidth Utilization: {:.1}%",
        validation.rtx6000_metrics.memory_bandwidth_utilization
    );
    tracing::info!(
        "Mixed Precision Speedup: {:.1}x",
        validation.rtx6000_metrics.mixed_precision_speedup
    );
    tracing::info!(
        "Memory Coalescing Efficiency: {:.1}%",
        validation.rtx6000_metrics.memory_coalescing_efficiency * 100.0
    );

    if validation.meets_target {
        tracing::info!("\n✅ SUCCESS: <50ms/token target achieved!");
        tracing::info!("🚀 RTX 6000 optimization successful - ready for production");

        // Get optimization recommendations
        let recommendations = validator.get_optimization_recommendations().await;
        if !recommendations.is_empty() {
            tracing::info!("\n💡 OPTIMIZATION RECOMMENDATIONS:");
            for (i, rec) in recommendations.iter().enumerate() {
                tracing::info!("  {}. {}", i + 1, rec);
            }
        }

        Ok(())
    } else {
        tracing::info!("\n⚠️  PERFORMANCE TARGET MISSED");
        tracing::info!(
            "Current: {:.2}ms > Target: 50ms",
            validation.avg_token_latency_ms
        );

        // Get optimization recommendations
        let recommendations = validator.get_optimization_recommendations().await;
        if !recommendations.is_empty() {
            tracing::info!("\n💡 OPTIMIZATION RECOMMENDATIONS:");
            for (i, rec) in recommendations.iter().enumerate() {
                tracing::info!("  {}. {}", i + 1, rec);
            }
        }

        Err("Performance target not met".into())
    }
}
