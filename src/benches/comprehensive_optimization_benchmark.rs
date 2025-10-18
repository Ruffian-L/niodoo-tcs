//! Comprehensive Optimization Benchmark Suite for Phase 2
use tracing::{error, info, warn};
// This benchmark suite tests all Phase 2 optimizations including:
// - GPU acceleration performance
// - Advanced caching efficiency
// - Async pattern optimization
// - Memory management improvements
// - ML optimization techniques
// - Monitoring system overhead

// use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use tokio::runtime::Runtime;

use niodoo_consciousness::{
    advanced_async_patterns::{ConsciousnessTask, TaskPriority, WorkStealingScheduler},
    advanced_caching::AdvancedCacheManager,
    enhanced_memory_management::{EnhancedMemoryManager, MemoryManagerConfig},
    gpu_acceleration::{ConsciousnessKernelConfig, CudaAccelerationManager, CudaConsciousnessOp},
    ml_optimization::{MLOptimizationConfig, MLOptimizationManager},
    performance_monitoring_system::{
        MonitoringConfig, PerformanceMetric, PerformanceMonitoringSystem,
    },
};

/// Comprehensive benchmark configuration
pub struct ComprehensiveBenchmarkConfig {
    /// Number of benchmark iterations
    pub iterations: usize,
    /// Timeout for each operation
    pub timeout_ms: u64,
    /// Sample size for statistical analysis
    pub sample_size: usize,
    /// Enable GPU benchmarks (if CUDA available)
    pub enable_gpu_benchmarks: bool,
    /// Enable distributed benchmarks
    pub enable_distributed_benchmarks: bool,
}

impl Default for ComprehensiveBenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            timeout_ms: 2000,
            sample_size: 50,
            enable_gpu_benchmarks: true,
            enable_distributed_benchmarks: false,
        }
    }
}

/// Comprehensive optimization benchmark suite
pub struct ComprehensiveOptimizationBenchmark {
    runtime: Runtime,
    config: ComprehensiveBenchmarkConfig,
    // Phase 2 optimization components
    cache_manager: Option<AdvancedCacheManager>,
    memory_manager: Option<EnhancedMemoryManager>,
    ml_optimizer: Option<MLOptimizationManager>,
    monitoring_system: Option<PerformanceMonitoringSystem>,
    async_scheduler: Option<WorkStealingScheduler>,
    gpu_manager: Option<CudaAccelerationManager>,
}

impl ComprehensiveOptimizationBenchmark {
    /// Create a new comprehensive benchmark
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let runtime = Runtime::new()?;
        let config = ComprehensiveBenchmarkConfig::default();

        // Initialize Phase 2 optimization components
        let cache_manager = runtime.block_on(async {
            match AdvancedCacheManager::new().await {
                Ok(manager) => Some(manager),
                Err(e) => {
                    tracing::info!("Warning: Advanced cache manager unavailable: {}", e);
                    None
                }
            }
        });

        let memory_manager = runtime.block_on(async {
            match EnhancedMemoryManager::new(MemoryManagerConfig::default()).await {
                Ok(manager) => Some(manager),
                Err(e) => {
                    tracing::info!("Warning: Enhanced memory manager unavailable: {}", e);
                    None
                }
            }
        });

        let ml_optimizer = runtime.block_on(async {
            match MLOptimizationManager::new(MLOptimizationConfig::default()).await {
                Ok(optimizer) => Some(optimizer),
                Err(e) => {
                    tracing::info!("Warning: ML optimizer unavailable: {}", e);
                    None
                }
            }
        });

        let monitoring_system = runtime.block_on(async {
            match PerformanceMonitoringSystem::new(MonitoringConfig::default()).await {
                Ok(system) => Some(system),
                Err(e) => {
                    tracing::info!("Warning: Monitoring system unavailable: {}", e);
                    None
                }
            }
        });

        let async_scheduler = runtime.block_on(async {
            match WorkStealingScheduler::new(4).await {
                Ok(scheduler) => Some(scheduler),
                Err(e) => {
                    tracing::info!("Warning: Async scheduler unavailable: {}", e);
                    None
                }
            }
        });

        let gpu_manager = if config.enable_gpu_benchmarks {
            #[cfg(feature = "cuda")]
            {
                runtime.block_on(async {
                    match CudaAccelerationManager::new() {
                        Ok(manager) => Some(manager),
                        Err(e) => {
                            tracing::info!("Warning: CUDA acceleration unavailable: {}", e);
                            None
                        }
                    }
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::info!("Warning: CUDA feature not enabled - skipping GPU benchmarks");
                None
            }
        } else {
            None
        };

        Ok(Self {
            runtime,
            config,
            cache_manager,
            memory_manager,
            ml_optimizer,
            monitoring_system,
            async_scheduler,
            gpu_manager,
        })
    }

    /// Benchmark advanced caching performance
    fn benchmark_advanced_caching(&self, c: &mut Criterion) {
        if self.cache_manager.is_none() {
            return;
        }

        let test_inputs = vec![
            "What is the meaning of consciousness?",
            "How do emotions affect decision making?",
            "Explain quantum consciousness theory",
            "What makes humans different from AI?",
            "How does memory consolidation work?",
        ];

        let mut group = c.benchmark_group("advanced_caching");
        group.sample_size(self.config.sample_size);
        group.measurement_time(Duration::from_secs(10));

        for (i, input) in test_inputs.iter().enumerate() {
            group.bench_with_input(format!("cache_input_{}", i), input, |b, input| {
                b.iter(|| {
                    if let Some(cache_manager) = &self.cache_manager {
                        let timeout_duration = Duration::from_millis(self.config.timeout_ms);
                        self.runtime.block_on(async {
                            // First call (cache miss)
                            black_box(
                                cache_manager
                                    .cache_result(black_box(input), black_box("cached_response"))
                                    .await,
                            )
                            .unwrap();

                            // Second call (cache hit)
                            black_box(cache_manager.get_cached_result(black_box(input)).await)
                                .unwrap();
                        });
                    }
                });
            });
        }

        group.finish();
    }

    /// Benchmark enhanced memory management
    fn benchmark_memory_management(&self, c: &mut Criterion) {
        if self.memory_manager.is_none() {
            return;
        }

        let memory_sizes = vec![1024, 4096, 16384, 65536]; // Different memory allocation sizes

        let mut group = c.benchmark_group("memory_management");
        group.sample_size(self.config.sample_size);
        group.measurement_time(Duration::from_secs(8));

        for (i, &size) in memory_sizes.iter().enumerate() {
            group.bench_with_input(format!("memory_alloc_{}", i), &size, |b, &size| {
                b.iter(|| {
                    if let Some(memory_manager) = &self.memory_manager {
                        self.runtime.block_on(async {
                            // Allocate optimized memory
                            let memory = black_box(
                                memory_manager
                                    .allocate_optimized(black_box(size), black_box("benchmark"))
                                    .await,
                            )
                            .unwrap();

                            // Store string with deduplication
                            let handle = black_box(
                                memory_manager
                                    .store_string_optimized(black_box(format!(
                                        "Test string of size {}",
                                        size
                                    )))
                                    .await,
                            )
                            .unwrap();

                            // Retrieve string
                            black_box(memory_manager.retrieve_string(&handle).await).unwrap();
                        });
                    }
                });
            });
        }

        group.finish();
    }

    /// Benchmark ML optimization techniques
    fn benchmark_ml_optimization(&self, c: &mut Criterion) {
        if self.ml_optimizer.is_none() {
            return;
        }

        // Create test tensors for ML optimization
        let test_tensors = vec![
            ("small_model", vec![1.0f32; 1000]),
            ("medium_model", vec![1.0f32; 10000]),
            ("large_model", vec![1.0f32; 100000]),
        ];

        let mut group = c.benchmark_group("ml_optimization");
        group.sample_size(self.config.sample_size);
        group.measurement_time(Duration::from_secs(15));

        for (model_name, weights_data) in test_tensors {
            group.bench_with_input(model_name, &weights_data, |b, weights_data| {
                b.iter(|| {
                    if let Some(ml_optimizer) = &self.ml_optimizer {
                        use candle_core::{DType, Device, Tensor};
                        self.runtime.block_on(async {
                            // Create test tensor
                            let tensor = black_box(Tensor::from_vec(
                                black_box(weights_data.clone()),
                                (weights_data.len(),),
                                &Device::Cpu,
                            ))
                            .unwrap();

                            // Test ML optimization (may fail if actual ML libraries not available)
                            let _result = black_box(
                                ml_optimizer
                                    .optimize_model(
                                        black_box(model_name),
                                        black_box(&tensor),
                                        black_box(&Device::Cpu),
                                    )
                                    .await,
                            );
                        });
                    }
                });
            });
        }

        group.finish();
    }

    /// Benchmark async pattern optimization
    fn benchmark_async_patterns(&self, c: &mut Criterion) {
        if self.async_scheduler.is_none() {
            return;
        }

        let mut group = c.benchmark_group("async_patterns");
        group.sample_size(self.config.sample_size);
        group.measurement_time(Duration::from_secs(12));

        // Test different priority levels
        let priorities = vec![
            TaskPriority::Critical,
            TaskPriority::High,
            TaskPriority::Normal,
            TaskPriority::Low,
        ];

        for (i, &priority) in priorities.iter().enumerate() {
            group.bench_with_input(
                format!("priority_{:?}", priority),
                &priority,
                |b, &priority| {
                    b.iter(|| {
                        if let Some(scheduler) = &self.async_scheduler {
                            self.runtime.block_on(async {
                                // Create test task
                                let task = black_box(ConsciousnessTask::new(
                                    format!("task_{}", i),
                                    black_box(priority),
                                    black_box(vec![1, 2, 3, 4]),
                                    None,
                                ));

                                // Submit to scheduler
                                let _handle =
                                    black_box(scheduler.submit_task(black_box(task), None).await);
                            });
                        }
                    });
                },
            );
        }

        group.finish();
    }

    /// Benchmark GPU acceleration (if available)
    fn benchmark_gpu_acceleration(&self, c: &mut Criterion) {
        if self.gpu_manager.is_none() || !self.config.enable_gpu_benchmarks {
            return;
        }

        let test_data_sizes = vec![1000, 5000, 10000];

        let mut group = c.benchmark_group("gpu_acceleration");
        group.sample_size(self.config.sample_size);
        group.measurement_time(Duration::from_secs(20));

        for (i, &data_size) in test_data_sizes.iter().enumerate() {
            group.bench_with_input(format!("gpu_compute_{}", i), &data_size, |b, &data_size| {
                b.iter(|| {
                    if let Some(gpu_manager) = &self.gpu_manager {
                        self.runtime.block_on(async {
                            let mut manager = gpu_manager; // Need mutable reference

                            // Create test data
                            let input_data = black_box(vec![1.0f32; data_size]);
                            let config = black_box(ConsciousnessKernelConfig::default());

                            let op = black_box(CudaConsciousnessOp::new(input_data, config));

                            // Execute GPU operation (may fail if CUDA not properly set up)
                            let _result = black_box(op.execute_cuda(&mut manager));
                        });
                    }
                });
            });
        }

        group.finish();
    }

    /// Benchmark monitoring system overhead
    fn benchmark_monitoring_overhead(&self, c: &mut Criterion) {
        if self.monitoring_system.is_none() {
            return;
        }

        let mut group = c.benchmark_group("monitoring_overhead");
        group.sample_size(self.config.sample_size);
        group.measurement_time(Duration::from_secs(10));

        // Test metric recording overhead
        group.bench_function("metric_recording", |b| {
            b.iter(|| {
                if let Some(monitoring) = &self.monitoring_system {
                    self.runtime.block_on(async {
                        // Record various metrics
                        black_box(
                            monitoring
                                .record_metric(black_box(PerformanceMetric::CpuUsage(50.0)))
                                .await,
                        )
                        .unwrap();

                        black_box(
                            monitoring
                                .record_metric(black_box(PerformanceMetric::MemoryUsage(60.0)))
                                .await,
                        )
                        .unwrap();

                        black_box(
                            monitoring
                                .record_metric(black_box(PerformanceMetric::ResponseTime(
                                    Duration::from_millis(150),
                                )))
                                .await,
                        )
                        .unwrap();
                    });
                }
            });
        });

        // Test health check overhead
        group.bench_function("health_check", |b| {
            b.iter(|| {
                if let Some(monitoring) = &self.monitoring_system {
                    self.runtime.block_on(async {
                        black_box(monitoring.perform_health_check().await).unwrap();
                    });
                }
            });
        });

        group.finish();
    }

    /// Benchmark comprehensive optimization pipeline
    fn benchmark_comprehensive_pipeline(&self, c: &mut Criterion) {
        let test_inputs = vec![
            "Complex consciousness processing task with emotional analysis",
            "Multi-brain coordination scenario requiring memory management",
            "Learning optimization task with predictive analytics",
        ];

        let mut group = c.benchmark_group("comprehensive_pipeline");
        group.sample_size(self.config.sample_size);
        group.measurement_time(Duration::from_secs(25));

        for (i, input) in test_inputs.iter().enumerate() {
            group.bench_with_input(format!("pipeline_{}", i), input, |b, input| {
                b.iter(|| {
                    self.runtime.block_on(async {
                        // Comprehensive optimization pipeline (if all components available)
                        if let (
                            Some(cache),
                            Some(memory),
                            Some(ml),
                            Some(monitoring),
                            Some(scheduler),
                        ) = (
                            &self.cache_manager,
                            &self.memory_manager,
                            &self.ml_optimizer,
                            &self.monitoring_system,
                            &self.async_scheduler,
                        ) {
                            // 1. Cache check
                            black_box(cache.get_cached_result(black_box(input)).await).unwrap();

                            // 2. Memory allocation
                            black_box(
                                memory
                                    .allocate_optimized(black_box(1024), black_box("pipeline_test"))
                                    .await,
                            )
                            .unwrap();

                            // 3. Monitoring metric
                            black_box(
                                monitoring
                                    .record_metric(black_box(PerformanceMetric::ResponseTime(
                                        Duration::from_millis(200),
                                    )))
                                    .await,
                            )
                            .unwrap();

                            // 4. Async task submission
                            let task = black_box(ConsciousnessTask::new(
                                format!("pipeline_task_{}", i),
                                black_box(TaskPriority::Normal),
                                black_box(input.as_bytes().to_vec()),
                                None,
                            ));
                            black_box(scheduler.submit_task(black_box(task), None).await).unwrap();

                            // 5. Health check
                            black_box(monitoring.perform_health_check().await).unwrap();
                        }
                    });
                });
            });
        }

        group.finish();
    }

    /// Run all comprehensive benchmarks
    pub fn run_all_benchmarks(&self, c: &mut Criterion) {
        tracing::info!("🚀 Running Comprehensive Phase 2 Optimization Benchmark Suite");

        let mut available_components = 0;
        if self.cache_manager.is_some() {
            available_components += 1;
            tracing::info!("✅ Advanced caching available");
        }
        if self.memory_manager.is_some() {
            available_components += 1;
            tracing::info!("✅ Enhanced memory management available");
        }
        if self.ml_optimizer.is_some() {
            available_components += 1;
            tracing::info!("✅ ML optimization available");
        }
        if self.monitoring_system.is_some() {
            available_components += 1;
            tracing::info!("✅ Performance monitoring available");
        }
        if self.async_scheduler.is_some() {
            available_components += 1;
            tracing::info!("✅ Async patterns available");
        }
        if self.gpu_manager.is_some() && self.config.enable_gpu_benchmarks {
            available_components += 1;
            tracing::info!("✅ GPU acceleration available");
        }

        tracing::info!(
            "📊 Available optimization components: {}/6",
            available_components
        );

        self.benchmark_advanced_caching(c);
        self.benchmark_memory_management(c);
        self.benchmark_ml_optimization(c);
        self.benchmark_async_patterns(c);
        self.benchmark_monitoring_overhead(c);

        if self.config.enable_gpu_benchmarks {
            self.benchmark_gpu_acceleration(c);
        }

        self.benchmark_comprehensive_pipeline(c);

        tracing::info!("✅ Comprehensive optimization benchmarking complete");
    }
}

/// Criterion benchmark functions
fn comprehensive_optimization_benchmarks(c: &mut Criterion) {
    let benchmark = ComprehensiveOptimizationBenchmark::new().unwrap();
    benchmark.run_all_benchmarks(c);
}

/// Criterion group configuration
criterion_group!(
    name = comprehensive_benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(2));
    targets = comprehensive_optimization_benchmarks
);

/// Main function for running benchmarks
criterion_main!(comprehensive_benches);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_benchmark_creation() {
        let benchmark = ComprehensiveOptimizationBenchmark::new();
        assert!(benchmark.is_ok());
    }

    #[tokio::test]
    async fn test_component_initialization() {
        let benchmark = ComprehensiveOptimizationBenchmark::new().unwrap();

        // Test that at least some components are available
        assert!(
            benchmark.cache_manager.is_some()
                || benchmark.memory_manager.is_some()
                || benchmark.ml_optimizer.is_some()
                || benchmark.monitoring_system.is_some()
                || benchmark.async_scheduler.is_some()
        );
    }

    #[test]
    fn test_benchmark_config() {
        let config = ComprehensiveBenchmarkConfig::default();
        assert_eq!(config.iterations, 100);
        assert_eq!(config.timeout_ms, 2000);
        assert_eq!(config.sample_size, 50);
        assert!(config.enable_gpu_benchmarks);
    }
}
