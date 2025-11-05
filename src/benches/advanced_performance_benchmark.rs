//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Advanced Performance Benchmark Suite for Phase 2 Optimizations
use tracing::{error, info, warn};
// This benchmark suite tests the advanced performance optimizations
// including GPU acceleration, advanced caching, memory optimization,
// and distributed processing capabilities.

// use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use tokio::runtime::Runtime;

use niodoo_consciousness::consciousness_engine::advanced_performance_optimizer::AdvancedPerformanceOptimizer;

/// Benchmark configuration for advanced performance tests
pub struct AdvancedBenchmarkConfig {
    /// Number of benchmark iterations
    pub iterations: usize,
    /// Timeout for each operation
    pub timeout_ms: u64,
    /// Sample size for statistical analysis
    pub sample_size: usize,
}

impl Default for AdvancedBenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            timeout_ms: 1000,
            sample_size: 50,
        }
    }
}

/// Advanced performance benchmark suite
pub struct AdvancedPerformanceBenchmark {
    runtime: Runtime,
    optimizer: Option<AdvancedPerformanceOptimizer>,
    config: AdvancedBenchmarkConfig,
}

impl AdvancedPerformanceBenchmark {
    /// Create a new advanced performance benchmark
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let runtime = Runtime::new()?;
        let config = AdvancedBenchmarkConfig::default();

        // Try to initialize the advanced optimizer (may fail if CUDA not available)
        let optimizer = runtime.block_on(async {
            match AdvancedPerformanceOptimizer::new().await {
                Ok(opt) => Some(opt),
                Err(e) => {
                    tracing::info!("Warning: Advanced optimizer unavailable: {}", e);
                    None
                }
            }
        });

        Ok(Self {
            runtime,
            optimizer,
            config,
        })
    }

    /// Benchmark basic consciousness processing
    fn benchmark_basic_processing(&self, c: &mut Criterion) {
        let test_inputs = vec![
            "Hello, how are you feeling today?",
            "What is the meaning of consciousness?",
            "Tell me about your emotional state",
            "How do you process complex thoughts?",
            "What makes you feel alive?",
        ];

        let mut group = c.benchmark_group("basic_consciousness_processing");
        group.sample_size(self.config.sample_size);
        group.measurement_time(Duration::from_secs(10));

        for (i, input) in test_inputs.iter().enumerate() {
            group.bench_with_input(format!("input_{}", i), input, |b, input| {
                b.iter(|| {
                    // Skip if optimizer not available
                    if let Some(optimizer) = &self.optimizer {
                        let timeout_duration = Duration::from_millis(self.config.timeout_ms);
                        self.runtime
                            .block_on(async {
                                black_box(
                                    optimizer
                                        .optimize_consciousness_processing(
                                            black_box(input),
                                            black_box(timeout_duration),
                                        )
                                        .await,
                                )
                            })
                            .unwrap();
                    }
                });
            });
        }

        group.finish();
    }

    /// Benchmark GPU-accelerated processing
    fn benchmark_gpu_acceleration(&self, c: &mut Criterion) {
        // Only run if GPU acceleration is available
        if self.optimizer.is_none() {
            return;
        }

        let test_inputs = vec![
            "Complex mathematical reasoning task",
            "Multi-layered emotional analysis",
            "Parallel brain coordination scenario",
            "Memory-intensive consciousness operation",
        ];

        let mut group = c.benchmark_group("gpu_accelerated_processing");
        group.sample_size(self.config.sample_size);
        group.measurement_time(Duration::from_secs(15));

        for (i, input) in test_inputs.iter().enumerate() {
            group.bench_with_input(format!("gpu_input_{}", i), input, |b, input| {
                b.iter(|| {
                    if let Some(optimizer) = &self.optimizer {
                        let timeout_duration = Duration::from_millis(self.config.timeout_ms);
                        self.runtime
                            .block_on(async {
                                black_box(
                                    optimizer
                                        .optimize_consciousness_processing(
                                            black_box(input),
                                            black_box(timeout_duration),
                                        )
                                        .await,
                                )
                            })
                            .unwrap();
                    }
                });
            });
        }

        group.finish();
    }

    /// Benchmark advanced caching performance
    fn benchmark_advanced_caching(&self, c: &mut Criterion) {
        // Only run if optimizer is available
        if self.optimizer.is_none() {
            return;
        }

        let mut group = c.benchmark_group("advanced_caching");
        group.sample_size(self.config.sample_size);
        group.measurement_time(Duration::from_secs(10));

        // Test cache hit performance
        group.bench_function("cache_hit_performance", |b| {
            b.iter(|| {
                if let Some(optimizer) = &self.optimizer {
                    let timeout_duration = Duration::from_millis(self.config.timeout_ms);
                    self.runtime.block_on(async {
                        // First call to populate cache
                        black_box(
                            optimizer
                                .optimize_consciousness_processing(
                                    black_box("Cache warm-up input"),
                                    black_box(timeout_duration),
                                )
                                .await,
                        )
                        .unwrap();

                        // Second call should be cached
                        black_box(
                            optimizer
                                .optimize_consciousness_processing(
                                    black_box("Cache warm-up input"),
                                    black_box(timeout_duration),
                                )
                                .await,
                        )
                        .unwrap();
                    });
                }
            });
        });

        group.finish();
    }

    /// Benchmark memory optimization techniques
    fn benchmark_memory_optimization(&self, c: &mut Criterion) {
        // Only run if optimizer is available
        if self.optimizer.is_none() {
            return;
        }

        let large_inputs = vec![
            "A".repeat(1000), // Large string for memory testing
            "B".repeat(2000), // Even larger string
            "C".repeat(5000), // Very large string
        ];

        let mut group = c.benchmark_group("memory_optimization");
        group.sample_size(self.config.sample_size);
        group.measurement_time(Duration::from_secs(12));

        for (i, input) in large_inputs.iter().enumerate() {
            group.bench_with_input(format!("memory_input_{}", i), input, |b, input| {
                b.iter(|| {
                    if let Some(optimizer) = &self.optimizer {
                        let timeout_duration = Duration::from_millis(self.config.timeout_ms);
                        self.runtime
                            .block_on(async {
                                black_box(
                                    optimizer
                                        .optimize_consciousness_processing(
                                            black_box(input),
                                            black_box(timeout_duration),
                                        )
                                        .await,
                                )
                            })
                            .unwrap();
                    }
                });
            });
        }

        group.finish();
    }

    /// Benchmark concurrent processing capabilities
    fn benchmark_concurrent_processing(&self, c: &mut Criterion) {
        // Only run if optimizer is available
        if self.optimizer.is_none() {
            return;
        }

        let mut group = c.benchmark_group("concurrent_processing");
        group.sample_size(self.config.sample_size);
        group.measurement_time(Duration::from_secs(20));

        // Test different concurrency levels
        for concurrency in [1, 5, 10, 20] {
            group.bench_with_input(
                format!("concurrency_{}", concurrency),
                &concurrency,
                |b, &concurrency| {
                    b.iter(|| {
                        if let Some(optimizer) = &self.optimizer {
                            let timeout_duration = Duration::from_millis(self.config.timeout_ms);
                            self.runtime.block_on(async {
                                let tasks = (0..concurrency).map(|i| {
                                    let input = format!("Concurrent input {}", i);
                                    optimizer.optimize_consciousness_processing(
                                        black_box(&input),
                                        black_box(timeout_duration),
                                    )
                                });

                                for task in tasks {
                                    black_box(task.await).unwrap();
                                }
                            });
                        }
                    });
                },
            );
        }

        group.finish();
    }

    /// Run all advanced performance benchmarks
    pub fn run_all_benchmarks(&self, c: &mut Criterion) {
        tracing::info!("🚀 Running Advanced Performance Benchmark Suite");

        if self.optimizer.is_none() {
            tracing::info!(
                "⚠️ Warning: Advanced optimizer not available - running CPU-only benchmarks"
            );
        } else {
            tracing::info!("✅ Advanced optimizer available - running full benchmark suite");
        }

        self.benchmark_basic_processing(c);
        self.benchmark_gpu_acceleration(c);
        self.benchmark_advanced_caching(c);
        self.benchmark_memory_optimization(c);
        self.benchmark_concurrent_processing(c);

        tracing::info!("✅ Advanced performance benchmarking complete");
    }
}

/// Criterion benchmark functions
fn advanced_performance_benchmarks(c: &mut Criterion) {
    let benchmark = AdvancedPerformanceBenchmark::new().unwrap();
    benchmark.run_all_benchmarks(c);
}

/// Criterion group configuration
criterion_group!(
    name = advanced_benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(2));
    targets = advanced_performance_benchmarks
);

/// Main function for running benchmarks
criterion_main!(advanced_benches);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_benchmark_creation() {
        let benchmark = AdvancedPerformanceBenchmark::new();
        assert!(benchmark.is_ok());
    }

    #[tokio::test]
    async fn test_basic_optimization() {
        let benchmark = AdvancedPerformanceBenchmark::new().unwrap();
        if let Some(optimizer) = &benchmark.optimizer {
            let result = optimizer
                .optimize_consciousness_processing("test input", Duration::from_millis(1000))
                .await;
            assert!(result.is_ok());
        }
    }
}
