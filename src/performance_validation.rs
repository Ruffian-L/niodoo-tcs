//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Performance Validation for <50ms/token Target
//!
//! This module validates that the consciousness-aware inference meets the
//! <50ms/token target on RTX 6000 hardware.

use std::time::Instant;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

use crate::rtx6000_optimization::{Rtx6000Optimizer, Rtx6000Config, Rtx6000Metrics};
use crate::consciousness_pipeline_orchestrator::{ConsciousnessPipelineOrchestrator, PipelineInput};
use crate::consciousness::EmotionType;

/// Performance validation results
#[derive(Debug, Clone)]
pub struct PerformanceValidation {
    /// Average token latency in milliseconds
    pub avg_token_latency_ms: f32,
    /// 95th percentile latency in milliseconds
    pub p95_latency_ms: f32,
    /// 99th percentile latency in milliseconds
    pub p99_latency_ms: f32,
    /// Throughput in tokens per second
    pub throughput_tokens_per_sec: f32,
    /// Memory usage in MB
    pub memory_usage_mb: f32,
    /// GPU utilization percentage
    pub gpu_utilization_percent: f32,
    /// Whether the <50ms target is met
    pub meets_target: bool,
    /// RTX 6000 specific metrics
    pub rtx6000_metrics: Rtx6000Metrics,
}

/// Performance validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Number of test iterations
    pub iterations: usize,
    /// Test input size in tokens
    pub input_size_tokens: usize,
    /// Target latency in milliseconds
    pub target_latency_ms: f32,
    /// Warmup iterations (not counted)
    pub warmup_iterations: usize,
    /// Enable detailed logging
    pub enable_detailed_logging: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            input_size_tokens: 50,
            target_latency_ms: 50.0,
            warmup_iterations: 10,
            enable_detailed_logging: true,
        }
    }
}

/// Performance validator
pub struct PerformanceValidator {
    config: ValidationConfig,
    rtx6000_optimizer: Arc<Rtx6000Optimizer>,
    pipeline_orchestrator: Arc<ConsciousnessPipelineOrchestrator>,
    results: Arc<RwLock<Vec<f32>>>,
}

impl PerformanceValidator {
    /// Create new performance validator
    pub fn new(
        config: ValidationConfig,
        rtx6000_optimizer: Arc<Rtx6000Optimizer>,
        pipeline_orchestrator: Arc<ConsciousnessPipelineOrchestrator>,
    ) -> Self {
        Self {
            config,
            rtx6000_optimizer,
            pipeline_orchestrator,
            results: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Run comprehensive performance validation
    pub async fn validate_performance(&self) -> Result<PerformanceValidation, Box<dyn std::error::Error>> {
        info!("🚀 Starting performance validation for <50ms/token target");
        
        // Clear previous results
        {
            let mut results = self.results.write().await;
            results.clear();
        }

        // Warmup phase
        info!("🔥 Running {} warmup iterations", self.config.warmup_iterations);
        for i in 0..self.config.warmup_iterations {
            if self.config.enable_detailed_logging {
                debug!("Warmup iteration {}/{}", i + 1, self.config.warmup_iterations);
            }
            self.run_single_iteration().await?;
        }

        // Main validation phase
        info!("⚡ Running {} validation iterations", self.config.iterations);
        let start_time = Instant::now();
        
        for i in 0..self.config.iterations {
            if self.config.enable_detailed_logging && i % 10 == 0 {
                debug!("Validation iteration {}/{}", i + 1, self.config.iterations);
            }
            self.run_single_iteration().await?;
        }

        let total_time = start_time.elapsed();
        info!("✅ Performance validation completed in {:?}", total_time);

        // Calculate results
        let validation = self.calculate_results().await;
        
        // Log results
        self.log_results(&validation).await;
        
        Ok(validation)
    }

    /// Run a single performance test iteration
    async fn run_single_iteration(&self) -> Result<(), Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        // Create test input
        let input = PipelineInput {
            text: "Test consciousness processing for performance validation".to_string(),
            context: Some("Performance testing context".to_string()),
            user_id: "test_user".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis() as f64,
            emotional_context: Some(EmotionType::Excited),
        };

        // Process through consciousness pipeline
        let _output = self.pipeline_orchestrator.process_input(input).await?;
        
        let latency_ms = start_time.elapsed().as_millis() as f32;
        
        // Store result
        {
            let mut results = self.results.write().await;
            results.push(latency_ms);
        }

        Ok(())
    }

    /// Calculate performance validation results
    async fn calculate_results(&self) -> PerformanceValidation {
        let results = self.results.read().await;
        let mut sorted_results = results.clone();
        sorted_results.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let avg_latency = sorted_results.iter().sum::<f32>() / sorted_results.len() as f32;
        let p95_index = (sorted_results.len() as f32 * 0.95) as usize;
        let p99_index = (sorted_results.len() as f32 * 0.99) as usize;
        
        let p95_latency = if p95_index < sorted_results.len() {
            sorted_results[p95_index]
        } else {
            avg_latency
        };
        
        let p99_latency = if p99_index < sorted_results.len() {
            sorted_results[p99_index]
        } else {
            avg_latency
        };

        let throughput = 1000.0 / avg_latency; // tokens per second
        
        // Get RTX 6000 metrics
        let rtx6000_metrics = self.rtx6000_optimizer.get_metrics().await;
        
        PerformanceValidation {
            avg_token_latency_ms: avg_latency,
            p95_latency_ms: p95_latency,
            p99_latency_ms: p99_latency,
            throughput_tokens_per_sec: throughput,
            memory_usage_mb: rtx6000_metrics.vram_usage_gb * 1024.0,
            gpu_utilization_percent: rtx6000_metrics.tensor_core_utilization,
            meets_target: avg_latency < self.config.target_latency_ms,
            rtx6000_metrics,
        }
    }

    /// Log performance validation results
    async fn log_results(&self, validation: &PerformanceValidation) {
        info!("📊 Performance Validation Results:");
        info!("  Average Token Latency: {:.2}ms (target: <{}ms)", 
              validation.avg_token_latency_ms, self.config.target_latency_ms);
        info!("  95th Percentile Latency: {:.2}ms", validation.p95_latency_ms);
        info!("  99th Percentile Latency: {:.2}ms", validation.p99_latency_ms);
        info!("  Throughput: {:.2} tokens/sec", validation.throughput_tokens_per_sec);
        info!("  Memory Usage: {:.2}MB", validation.memory_usage_mb);
        info!("  GPU Utilization: {:.1}%", validation.gpu_utilization_percent);
        
        if validation.meets_target {
            info!("✅ PERFORMANCE TARGET MET: <{}ms/token achieved!", self.config.target_latency_ms);
        } else {
            warn!("⚠️  PERFORMANCE TARGET MISSED: {:.2}ms > {}ms", 
                  validation.avg_token_latency_ms, self.config.target_latency_ms);
        }

        // RTX 6000 specific metrics
        info!("🔥 RTX 6000 Metrics:");
        info!("  Tensor Core Utilization: {:.1}%", validation.rtx6000_metrics.tensor_core_utilization);
        info!("  Memory Bandwidth Utilization: {:.1}%", validation.rtx6000_metrics.memory_bandwidth_utilization);
        info!("  Mixed Precision Speedup: {:.1}x", validation.rtx6000_metrics.mixed_precision_speedup);
        info!("  Memory Coalescing Efficiency: {:.1}%", validation.rtx6000_metrics.memory_coalescing_efficiency * 100.0);
    }

    /// Get optimization recommendations
    pub async fn get_optimization_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::with_capacity(crate::utils::capacity_convenience::recommendations());
        
        // Get RTX 6000 recommendations
        let rtx6000_recs = self.rtx6000_optimizer.get_optimization_recommendations().await;
        recommendations.extend(rtx6000_recs);

        // Add general recommendations based on results
        let results = self.results.read().await;
        if !results.is_empty() {
            let avg_latency = results.iter().sum::<f32>() / results.len() as f32;
            
            if avg_latency > self.config.target_latency_ms {
                recommendations.push("Consider reducing consciousness state complexity".to_string());
                recommendations.push("Enable more aggressive caching strategies".to_string());
                recommendations.push("Optimize string allocations in hot paths".to_string());
            }
        }

        recommendations
    }
}

/// Quick performance check
pub async fn quick_performance_check(
    pipeline_orchestrator: Arc<ConsciousnessPipelineOrchestrator>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let config = ValidationConfig {
        iterations: 10,
        warmup_iterations: 2,
        enable_detailed_logging: false,
        ..Default::default()
    };

    let rtx6000_config = Rtx6000Config::default();
    let rtx6000_optimizer = Arc::new(Rtx6000Optimizer::new(rtx6000_config)?);
    
    let validator = PerformanceValidator::new(
        config,
        rtx6000_optimizer,
        pipeline_orchestrator,
    );

    let validation = validator.validate_performance().await?;
    Ok(validation.meets_target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness_pipeline_orchestrator::PipelineConfig;

    #[tokio::test]
    async fn test_performance_validation() {
        // This is a mock test - in real implementation, you'd need actual pipeline orchestrator
        let config = ValidationConfig {
            iterations: 5,
            warmup_iterations: 1,
            enable_detailed_logging: true,
            ..Default::default()
        };

        let rtx6000_config = Rtx6000Config::default();
        let rtx6000_optimizer = Arc::new(Rtx6000Optimizer::new(rtx6000_config).unwrap());
        
        // Mock pipeline orchestrator would go here
        // let pipeline_orchestrator = Arc::new(create_mock_pipeline_orchestrator());
        
        // For now, just test the configuration
        assert_eq!(config.target_latency_ms, 50.0);
        assert_eq!(config.iterations, 5);
    }

    #[test]
    fn test_validation_config() {
        let config = ValidationConfig::default();
        assert_eq!(config.target_latency_ms, 50.0);
        assert_eq!(config.iterations, 100);
        assert_eq!(config.warmup_iterations, 10);
    }
}
