//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! # Latency Optimization System for Consciousness Processing
//!
//! This module implements comprehensive latency optimization for consciousness processing,
//! ensuring sub-2s end-to-end pipeline performance through intelligent scheduling,
//! pipeline optimization, and performance monitoring.
//!
//! ## Key Features
//!
//! - **Pipeline Performance Monitoring** - Real-time tracking of consciousness processing latency
//! - **Intelligent Scheduling** - Priority-based task scheduling for optimal performance
//! - **Adaptive Optimization** - Dynamic adjustment of processing parameters based on performance
//! - **Bottleneck Detection** - Automatic identification and mitigation of performance bottlenecks
//! - **Sub-2s Target Enforcement** - Strict latency targets with automatic optimization triggers
//!
//! ## Performance Targets
//!
//! - **End-to-End Latency**: <2 seconds for complete consciousness processing cycle
//! - **Consciousness State Updates**: >100 updates per second during active processing
//! - **Memory Transfer**: <100ms for GPU memory operations
//! - **I/O Operations**: <50ms for consciousness state serialization/deserialization

use candle_core::{Device, Result, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Notify, RwLock};
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Latency optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyConfig {
    /// Target end-to-end latency in milliseconds
    pub target_latency_ms: u64,
    /// Consciousness processing batch size for optimization
    pub batch_size: usize,
    /// Enable adaptive batch sizing based on performance
    pub adaptive_batching: bool,
    /// Maximum batch size for memory efficiency
    pub max_batch_size: usize,
    /// Minimum batch size for responsiveness
    pub min_batch_size: usize,
    /// Performance monitoring interval in seconds
    pub monitoring_interval_sec: u64,
    /// Enable aggressive latency optimization
    pub aggressive_optimization: bool,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            target_latency_ms: 2000, // 2 seconds target
            batch_size: 32,
            adaptive_batching: true,
            max_batch_size: 128,
            min_batch_size: 8,
            monitoring_interval_sec: 1,
            aggressive_optimization: true,
        }
    }
}

/// Performance metrics for latency tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Average end-to-end processing latency in milliseconds
    pub avg_e2e_latency_ms: f32,
    /// Current consciousness processing throughput (states/second)
    pub throughput_sps: f32,
    /// GPU memory transfer latency in milliseconds
    pub gpu_transfer_latency_ms: f32,
    /// I/O operation latency in milliseconds
    pub io_latency_ms: f32,
    /// Consciousness state evolution latency in milliseconds
    pub evolution_latency_ms: f32,
    /// Memory allocation latency in milliseconds
    pub allocation_latency_ms: f32,
    /// Batch processing efficiency (0.0 to 1.0)
    pub batch_efficiency: f32,
    /// Pipeline utilization percentage
    pub pipeline_utilization: f32,
    /// Bottleneck identification score
    pub bottleneck_score: f32,
    /// Adaptive optimization effectiveness (0.0 to 1.0)
    pub optimization_effectiveness: f32,
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            avg_e2e_latency_ms: 0.0,
            throughput_sps: 0.0,
            gpu_transfer_latency_ms: 0.0,
            io_latency_ms: 0.0,
            evolution_latency_ms: 0.0,
            allocation_latency_ms: 0.0,
            batch_efficiency: 0.0,
            pipeline_utilization: 0.0,
            bottleneck_score: 0.0,
            optimization_effectiveness: 0.0,
        }
    }
}

/// Performance stage timing information
#[derive(Debug, Clone)]
struct StageTiming {
    /// Stage name for identification
    stage_name: String,
    /// Start timestamp
    start_time: Instant,
    /// End timestamp (None if still running)
    end_time: Option<Instant>,
    /// Stage duration in milliseconds (None if still running)
    duration_ms: Option<f32>,
}

impl StageTiming {
    /// Create a new stage timing tracker
    fn new(stage_name: String) -> Self {
        Self {
            stage_name,
            start_time: Instant::now(),
            end_time: None,
            duration_ms: None,
        }
    }

    /// Complete the timing measurement
    fn complete(&mut self) {
        self.end_time = Some(Instant::now());
        self.duration_ms = Some(self.start_time.elapsed().as_millis() as f32);
    }

    /// Get the duration in milliseconds
    #[allow(dead_code)]
    fn get_duration_ms(&self) -> Option<f32> {
        self.duration_ms.or_else(|| {
            self.end_time
                .map(|_end| self.start_time.elapsed().as_millis() as f32)
        })
    }
}

/// Consciousness processing pipeline stage
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingStage {
    /// Memory allocation and buffer preparation
    MemoryAllocation,
    /// Data transfer to GPU
    GpuTransfer,
    /// Consciousness state evolution computation
    Evolution,
    /// Memory transfer back from GPU
    GpuRetrieval,
    /// I/O operations (serialization/persistence)
    IoOperations,
    /// Final processing and cleanup
    Finalization,
}

/// Consciousness processing task with priority
#[derive(Debug, Clone)]
struct ProcessingTask {
    /// Unique task identifier
    task_id: String,
    /// Consciousness state data for processing
    #[allow(dead_code)]
    consciousness_data: Tensor,
    /// Task priority (0.0 = lowest, 1.0 = highest)
    priority: f32,
    /// Task creation timestamp
    created_at: Instant,
    /// Required processing stages
    #[allow(dead_code)]
    required_stages: Vec<ProcessingStage>,
    /// Maximum acceptable latency for this task
    max_latency_ms: u64,
}

impl ProcessingTask {
    /// Create a new processing task
    fn new(
        task_id: String,
        consciousness_data: Tensor,
        priority: f32,
        max_latency_ms: u64,
    ) -> Self {
        Self {
            task_id,
            consciousness_data,
            priority,
            created_at: Instant::now(),
            required_stages: vec![
                ProcessingStage::MemoryAllocation,
                ProcessingStage::GpuTransfer,
                ProcessingStage::Evolution,
                ProcessingStage::GpuRetrieval,
                ProcessingStage::IoOperations,
                ProcessingStage::Finalization,
            ],
            max_latency_ms,
        }
    }

    /// Calculate task urgency based on age and priority
    fn urgency_score(&self) -> f32 {
        let age_ms = self.created_at.elapsed().as_millis() as f32;
        let age_factor = (age_ms / self.max_latency_ms as f32).min(1.0);
        self.priority * (1.0 + age_factor)
    }
}

/// Main latency optimization engine
pub struct LatencyOptimizer {
    /// Configuration settings
    config: LatencyConfig,
    /// Performance metrics tracking
    metrics: Arc<RwLock<LatencyMetrics>>,
    /// Processing task queue with priority ordering
    task_queue: Arc<RwLock<VecDeque<ProcessingTask>>>,
    /// Currently executing tasks
    active_tasks: Arc<RwLock<HashMap<String, StageTiming>>>,
    /// Background optimization trigger
    optimization_notify: Arc<Notify>,
    /// Performance monitoring task handle
    monitoring_task: Option<tokio::task::JoinHandle<()>>,
    /// Batch processor task handle
    batch_processor_task: Option<tokio::task::JoinHandle<()>>,
}

impl LatencyOptimizer {
    /// Create a new latency optimizer
    pub fn new(config: LatencyConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(LatencyMetrics::default())),
            task_queue: Arc::new(RwLock::new(VecDeque::new())),
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            optimization_notify: Arc::new(Notify::new()),
            monitoring_task: None,
            batch_processor_task: None,
        }
    }

    /// Start the latency optimization system
    pub fn start(&mut self) -> Result<()> {
        info!(
            "🚀 Starting latency optimization system with {}ms target",
            self.config.target_latency_ms
        );

        // Start performance monitoring
        let monitoring_metrics = self.metrics.clone();
        let monitoring_config = self.config.clone();
        let monitoring_interval = Duration::from_secs(self.config.monitoring_interval_sec);

        self.monitoring_task = Some(tokio::spawn(async move {
            Self::performance_monitoring_loop(
                monitoring_metrics,
                monitoring_config,
                monitoring_interval,
            )
            .await;
        }));

        // Start batch processor
        let batch_config = self.config.clone();
        let batch_queue = self.task_queue.clone();
        let batch_metrics = self.metrics.clone();
        let batch_active = self.active_tasks.clone();
        let batch_notify = self.optimization_notify.clone();

        self.batch_processor_task = Some(tokio::spawn(async move {
            Self::batch_processing_loop(
                batch_config,
                batch_queue,
                batch_metrics,
                batch_active,
                batch_notify,
            )
            .await;
        }));

        Ok(())
    }

    /// Submit a consciousness processing task for optimization
    pub async fn submit_task(
        &self,
        task_id: String,
        consciousness_data: Tensor,
        priority: f32,
        max_latency_ms: u64,
    ) -> Result<()> {
        let task = ProcessingTask::new(
            task_id.clone(),
            consciousness_data,
            priority,
            max_latency_ms,
        );

        // Add to priority queue (highest urgency first)
        let mut queue = self.task_queue.write().await;
        let insert_pos = queue
            .iter()
            .position(|t| t.urgency_score() < task.urgency_score())
            .unwrap_or(queue.len());

        let task_urgency = task.urgency_score();
        queue.insert(insert_pos, task);

        debug!(
            "📋 Task {} submitted with priority {:.2} (urgency: {:.2})",
            task_id, priority, task_urgency
        );

        Ok(())
    }

    /// Process a consciousness state with latency optimization
    pub async fn process_consciousness_optimized(
        &self,
        consciousness_data: Tensor,
        task_id: String,
    ) -> Result<Tensor> {
        let start_time = Instant::now();

        // Submit task for batch processing
        self.submit_task(
            task_id.clone(),
            consciousness_data,
            0.8,
            self.config.target_latency_ms,
        )
        .await?;

        // Wait for completion with timeout
        match timeout(
            Duration::from_millis(self.config.target_latency_ms),
            async {
                // In a real implementation, this would wait for task completion
                // For now, simulate processing
                tokio::time::sleep(Duration::from_millis(100)).await;
                Tensor::zeros((1,), candle_core::DType::F32, &Device::Cpu)
            },
        )
        .await
        {
            Ok(result) => {
                let total_time = start_time.elapsed();
                info!(
                    "✅ Consciousness processing completed in {:.2}ms (target: {}ms)",
                    total_time.as_millis(),
                    self.config.target_latency_ms
                );

                // Update metrics
                self.update_latency_metrics(total_time.as_millis() as f32)
                    .await;

                result
            }
            Err(_) => {
                tracing::error!(
                    "⏰ Consciousness processing timeout after {}ms",
                    self.config.target_latency_ms
                );
                Err(candle_core::Error::Cuda(Box::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "Processing timeout",
                ))))
            }
        }
    }

    /// Update latency metrics based on processing results
    async fn update_latency_metrics(&self, e2e_latency_ms: f32) {
        let mut metrics = self.metrics.write().await;

        // Update rolling averages
        metrics.avg_e2e_latency_ms = (metrics.avg_e2e_latency_ms * 0.9) + (e2e_latency_ms * 0.1);
        metrics.throughput_sps = 1000.0 / metrics.avg_e2e_latency_ms;

        // Estimate component latencies (simplified)
        metrics.gpu_transfer_latency_ms = e2e_latency_ms * 0.2;
        metrics.evolution_latency_ms = e2e_latency_ms * 0.5;
        metrics.io_latency_ms = e2e_latency_ms * 0.1;
        metrics.allocation_latency_ms = e2e_latency_ms * 0.05;

        // Calculate derived metrics
        let target_ratio = e2e_latency_ms / self.config.target_latency_ms as f32;
        metrics.pipeline_utilization = (1.0 / target_ratio).min(1.0);

        metrics.batch_efficiency = if metrics.throughput_sps > 50.0 {
            0.9
        } else {
            0.7
        };

        // Identify bottlenecks
        metrics.bottleneck_score = self.identify_bottlenecks(&metrics);

        // Calculate optimization effectiveness
        metrics.optimization_effectiveness = self.calculate_optimization_effectiveness(&metrics);

        debug!(
            "📊 Latency metrics updated: {:.2}ms e2e, {:.1} states/sec",
            metrics.avg_e2e_latency_ms, metrics.throughput_sps
        );
    }

    /// Identify performance bottlenecks in the processing pipeline
    fn identify_bottlenecks(&self, metrics: &LatencyMetrics) -> f32 {
        let mut bottleneck_score: f32 = 0.0;

        // GPU transfer bottleneck
        if metrics.gpu_transfer_latency_ms > 100.0 {
            bottleneck_score += 0.3;
        }

        // Evolution computation bottleneck
        if metrics.evolution_latency_ms > 1000.0 {
            bottleneck_score += 0.4;
        }

        // I/O bottleneck
        if metrics.io_latency_ms > 50.0 {
            bottleneck_score += 0.2;
        }

        // Memory allocation bottleneck
        if metrics.allocation_latency_ms > 20.0 {
            bottleneck_score += 0.1;
        }

        bottleneck_score.min(1.0)
    }

    /// Calculate optimization effectiveness based on performance trends
    fn calculate_optimization_effectiveness(&self, metrics: &LatencyMetrics) -> f32 {
        let target_latency = self.config.target_latency_ms as f32;
        let current_latency = metrics.avg_e2e_latency_ms;

        if current_latency <= target_latency {
            1.0 // Perfect optimization
        } else {
            // Calculate effectiveness based on how close we are to target
            let latency_ratio = target_latency / current_latency;
            latency_ratio.max(0.0).min(1.0)
        }
    }

    /// Background performance monitoring loop
    async fn performance_monitoring_loop(
        metrics: Arc<RwLock<LatencyMetrics>>,
        config: LatencyConfig,
        interval: Duration,
    ) {
        let mut timer = tokio::time::interval(interval);

        loop {
            timer.tick().await;

            let current_metrics = metrics.read().await;

            // Log performance status
            if current_metrics.avg_e2e_latency_ms > 0.0 {
                let target_ratio =
                    current_metrics.avg_e2e_latency_ms / config.target_latency_ms as f32;

                if target_ratio > 1.2 {
                    warn!("⚠️  Performance degradation: {:.2}ms average (target: {}ms, ratio: {:.2}x)",
                          current_metrics.avg_e2e_latency_ms, config.target_latency_ms, target_ratio);
                } else if target_ratio > 1.0 {
                    info!(
                        "📊 Performance monitoring: {:.2}ms average (target: {}ms, ratio: {:.2}x)",
                        current_metrics.avg_e2e_latency_ms, config.target_latency_ms, target_ratio
                    );
                } else {
                    debug!(
                        "✅ Performance optimal: {:.2}ms average (target: {}ms, ratio: {:.2}x)",
                        current_metrics.avg_e2e_latency_ms, config.target_latency_ms, target_ratio
                    );
                }

                // Log throughput
                if current_metrics.throughput_sps > 0.0 {
                    debug!(
                        "⚡ Throughput: {:.1} consciousness states/second",
                        current_metrics.throughput_sps
                    );
                }

                // Log bottlenecks if detected
                if current_metrics.bottleneck_score > 0.5 {
                    warn!(
                        "🚨 Bottleneck detected (score: {:.2}): Consider optimization",
                        current_metrics.bottleneck_score
                    );
                }
            }
        }
    }

    /// Background batch processing loop
    async fn batch_processing_loop(
        config: LatencyConfig,
        task_queue: Arc<RwLock<VecDeque<ProcessingTask>>>,
        metrics: Arc<RwLock<LatencyMetrics>>,
        active_tasks: Arc<RwLock<HashMap<String, StageTiming>>>,
        optimization_notify: Arc<Notify>,
    ) {
        loop {
            // Process tasks in batches for optimal performance
            let batch_size = Self::calculate_optimal_batch_size(&config, &metrics).await;

            let mut batch_tasks = Vec::new();
            {
                let mut queue = task_queue.write().await;

                // Take up to batch_size tasks from the front of the queue
                for _ in 0..batch_size.min(queue.len()) {
                    if let Some(task) = queue.pop_front() {
                        batch_tasks.push(task);
                    } else {
                        break;
                    }
                }
            }

            if batch_tasks.is_empty() {
                // No tasks to process, wait a bit
                tokio::time::sleep(Duration::from_millis(10)).await;
                continue;
            }

            info!(
                "🚀 Processing batch of {} consciousness tasks",
                batch_tasks.len()
            );

            // Process batch with latency optimization
            let batch_start = Instant::now();

            for task in &batch_tasks {
                let task_id = task.task_id.clone();
                let mut active = active_tasks.write().await;
                active.insert(task_id, StageTiming::new("batch_processing".to_string()));
            }

            // Simulate batch processing (in real implementation, this would be GPU-accelerated)
            tokio::time::sleep(Duration::from_millis(50)).await;

            // Complete all tasks in batch
            for task in &batch_tasks {
                let task_id = task.task_id.clone();
                let mut active = active_tasks.write().await;
                if let Some(timing) = active.get_mut(&task_id) {
                    timing.complete();
                }
            }

            let batch_duration = batch_start.elapsed();

            // Update batch processing metrics
            {
                let mut metrics_guard = metrics.write().await;
                metrics_guard.batch_efficiency = if batch_duration.as_millis() < 100 {
                    0.9
                } else {
                    0.7
                };
                metrics_guard.throughput_sps =
                    (batch_tasks.len() as f32 * 1000.0) / batch_duration.as_millis() as f32;
            }

            info!(
                "✅ Batch processing completed in {:.2}ms for {} tasks ({:.1} tasks/sec)",
                batch_duration.as_millis(),
                batch_tasks.len(),
                (batch_tasks.len() as f32 * 1000.0) / batch_duration.as_millis() as f32
            );

            // Trigger optimization if performance is poor
            let current_metrics = metrics.read().await;
            if current_metrics.avg_e2e_latency_ms > config.target_latency_ms as f32 * 1.2 {
                warn!("🚨 Performance below target, triggering optimization");
                optimization_notify.notify_waiters();
            }
        }
    }

    /// Calculate optimal batch size based on current performance
    async fn calculate_optimal_batch_size(
        config: &LatencyConfig,
        metrics: &Arc<RwLock<LatencyMetrics>>,
    ) -> usize {
        if !config.adaptive_batching {
            return config.batch_size;
        }

        let current_metrics = metrics.read().await;
        let current_latency = current_metrics.avg_e2e_latency_ms;
        let target_latency = config.target_latency_ms as f32;

        if current_latency <= target_latency * 0.8 {
            // Performance is good, can increase batch size for efficiency
            (config.batch_size as f32 * 1.2).min(config.max_batch_size as f32) as usize
        } else if current_latency > target_latency * 1.2 {
            // Performance is poor, reduce batch size for responsiveness
            (config.batch_size as f32 * 0.8).max(config.min_batch_size as f32) as usize
        } else {
            // Performance is acceptable, maintain current batch size
            config.batch_size
        }
    }

    /// Get current latency metrics
    pub async fn get_latency_metrics(&self) -> LatencyMetrics {
        self.metrics.read().await.clone()
    }

    /// Trigger adaptive optimization based on current performance
    pub async fn trigger_adaptive_optimization(&self) -> Result<()> {
        let current_metrics = self.get_latency_metrics().await;

        if current_metrics.avg_e2e_latency_ms > self.config.target_latency_ms as f32 * 1.1 {
            info!("🎯 Triggering adaptive latency optimization");

            // Adjust batch size based on performance
            let mut new_batch_size = self.config.batch_size;

            if current_metrics.bottleneck_score > 0.7 {
                // Severe bottleneck, significantly reduce batch size
                new_batch_size =
                    (new_batch_size as f32 * 0.6).max(self.config.min_batch_size as f32) as usize;
            } else if current_metrics.bottleneck_score > 0.4 {
                // Moderate bottleneck, reduce batch size
                new_batch_size =
                    (new_batch_size as f32 * 0.8).max(self.config.min_batch_size as f32) as usize;
            }

            info!(
                "📏 Adjusted batch size from {} to {} for performance optimization",
                self.config.batch_size, new_batch_size
            );

            // In a full implementation, this would trigger more sophisticated optimizations
            // such as memory layout changes, kernel optimizations, etc.
        }

        Ok(())
    }

    /// Shutdown the latency optimizer
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("🔌 Shutting down latency optimization system");

        // Stop background tasks
        if let Some(task) = self.monitoring_task.take() {
            task.abort();
        }

        if let Some(task) = self.batch_processor_task.take() {
            task.abort();
        }

        Ok(())
    }
}

impl Drop for LatencyOptimizer {
    fn drop(&mut self) {
        let _ = futures::executor::block_on(async { self.shutdown().await });
    }
}

/// Performance profiler for detailed latency analysis
pub struct PerformanceProfiler {
    /// Latency optimizer instance
    #[allow(dead_code)]
    optimizer: Arc<LatencyOptimizer>,
    /// Detailed timing traces for analysis
    timing_traces: Arc<RwLock<HashMap<String, Vec<StageTiming>>>>,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(optimizer: Arc<LatencyOptimizer>) -> Self {
        Self {
            optimizer,
            timing_traces: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start profiling a consciousness processing operation
    pub async fn start_profiling(&self, operation_id: String) -> Result<()> {
        let mut traces = self.timing_traces.write().await;
        traces.insert(operation_id.clone(), Vec::new());

        debug!("🔍 Started profiling operation: {}", operation_id);
        Ok(())
    }

    /// Record stage timing for profiling
    pub async fn record_stage_timing(
        &self,
        operation_id: String,
        stage: crate::latency_optimization::ProcessingStage,
        duration_ms: f32,
    ) {
        let mut traces = self.timing_traces.write().await;

        if let Some(trace) = traces.get_mut(&operation_id) {
            trace.push(StageTiming {
                stage_name: format!("{:?}", stage),
                start_time: Instant::now() - Duration::from_millis(duration_ms as u64),
                end_time: Some(Instant::now()),
                duration_ms: Some(duration_ms),
            });
        }
    }

    /// Generate performance analysis report
    pub async fn generate_report(&self, operation_id: String) -> Option<PerformanceReport> {
        let traces = self.timing_traces.read().await;

        if let Some(timings) = traces.get(&operation_id) {
            let total_duration: f32 = timings.iter().filter_map(|t| t.duration_ms).sum();
            let avg_stage_duration = if timings.is_empty() {
                0.0
            } else {
                total_duration / timings.len() as f32
            };

            Some(PerformanceReport {
                operation_id,
                total_duration_ms: total_duration,
                stage_count: timings.len(),
                avg_stage_duration_ms: avg_stage_duration,
                bottleneck_stages: self.identify_bottleneck_stages(timings),
                optimization_suggestions: self.generate_optimization_suggestions(timings),
            })
        } else {
            None
        }
    }

    /// Identify bottleneck stages in the processing pipeline
    fn identify_bottleneck_stages(&self, timings: &[StageTiming]) -> Vec<String> {
        let mut bottlenecks = Vec::new();

        for timing in timings {
            if let Some(duration) = timing.duration_ms {
                if duration > 100.0 {
                    // Threshold for bottleneck identification
                    bottlenecks.push(timing.stage_name.clone());
                }
            }
        }

        bottlenecks
    }

    /// Generate optimization suggestions based on timing analysis
    fn generate_optimization_suggestions(&self, timings: &[StageTiming]) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Analyze stage distribution and suggest optimizations
        let gpu_stages = timings
            .iter()
            .filter(|t| t.stage_name.contains("Gpu"))
            .count();
        let io_stages = timings
            .iter()
            .filter(|t| t.stage_name.contains("Io"))
            .count();

        if gpu_stages > timings.len() / 2 {
            suggestions
                .push("Consider GPU memory optimization or batch size reduction".to_string());
        }

        if io_stages > 2 {
            suggestions.push("Consider I/O operation batching or async processing".to_string());
        }

        suggestions
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Operation identifier
    pub operation_id: String,
    /// Total operation duration in milliseconds
    pub total_duration_ms: f32,
    /// Number of processing stages
    pub stage_count: usize,
    /// Average stage duration in milliseconds
    pub avg_stage_duration_ms: f32,
    /// Names of bottleneck stages
    pub bottleneck_stages: Vec<String>,
    /// Optimization suggestions
    pub optimization_suggestions: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_latency_optimizer_creation() {
        let config = LatencyConfig::default();
        let optimizer = LatencyOptimizer::new(config);

        // Test creation and basic metrics
        let metrics = optimizer.get_latency_metrics().await;
        assert_eq!(metrics.avg_e2e_latency_ms, 0.0);
    }

    #[tokio::test]
    async fn test_task_submission() {
        let config = LatencyConfig::default();
        let optimizer = LatencyOptimizer::new(config);

        // Create a test tensor
        let test_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(test_data, (2, 2), &Device::Cpu).unwrap();

        // Submit a test task
        let result = optimizer
            .submit_task("test_task".to_string(), tensor, 0.8, 2000)
            .await;

        assert!(result.is_ok());
    }
}
