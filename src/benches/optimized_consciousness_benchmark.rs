//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

// use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use criterion::black_box;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use tokio::sync::RwLock;

// Import optimized consciousness engine modules
use niodoo_consciousness::consciousness_engine::performance_optimizer::{
    PerformanceConfig, PerformanceOptimizationEngine, PerformanceReport,
};

/// Benchmark optimized consciousness engine
fn bench_optimized_consciousness_engine(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("optimized_consciousness_engine");
    group.measurement_time(Duration::from_secs(10));

    // Test different configurations
    for max_concurrent in [1, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("max_concurrent", max_concurrent),
            max_concurrent,
            |b, &max_concurrent| {
                b.to_async(&rt).iter(|| async {
                    let config = PerformanceConfig {
                        max_concurrent_operations: max_concurrent,
                        cache_size_limit: 1000,
                        pool_size_limit: 100,
                        optimization_threshold_ms: 100,
                        enable_hot_path_optimization: true,
                        enable_memory_pooling: true,
                        enable_caching: true,
                    };

                    let engine = PerformanceOptimizationEngine::new(config);
                    let result = engine
                        .optimize_consciousness_engine("Test input for optimization")
                        .await
                        .unwrap();
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory pooling performance
fn bench_memory_pooling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_pooling");
    group.measurement_time(Duration::from_secs(10));

    // Test different pool sizes
    for pool_size in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("pool_size", pool_size),
            pool_size,
            |b, &pool_size| {
                b.to_async(&rt).iter(|| async {
                    let config = PerformanceConfig {
                        max_concurrent_operations: 10,
                        cache_size_limit: 1000,
                        pool_size_limit: pool_size,
                        optimization_threshold_ms: 100,
                        enable_hot_path_optimization: true,
                        enable_memory_pooling: true,
                        enable_caching: true,
                    };

                    let engine = PerformanceOptimizationEngine::new(config);

                    // Simulate memory pooling operations
                    for i in 0..*pool_size {
                        let result = engine
                            .optimize_consciousness_engine(&format!("Pool test {}", i))
                            .await
                            .unwrap();
                        black_box(result);
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark caching performance
fn bench_caching_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("caching_performance");
    group.measurement_time(Duration::from_secs(10));

    // Test different cache sizes
    for cache_size in [100, 500, 1000, 2000].iter() {
        group.bench_with_input(
            BenchmarkId::new("cache_size", cache_size),
            cache_size,
            |b, &cache_size| {
                b.to_async(&rt).iter(|| async {
                    let config = PerformanceConfig {
                        max_concurrent_operations: 10,
                        cache_size_limit: cache_size,
                        pool_size_limit: 100,
                        optimization_threshold_ms: 100,
                        enable_hot_path_optimization: true,
                        enable_memory_pooling: true,
                        enable_caching: true,
                    };

                    let engine = PerformanceOptimizationEngine::new(config);

                    // Simulate cache operations
                    for i in 0..*cache_size {
                        let result = engine
                            .optimize_consciousness_engine(&format!("Cache test {}", i))
                            .await
                            .unwrap();
                        black_box(result);
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark hot path optimization
fn bench_hot_path_optimization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("hot_path_optimization");
    group.measurement_time(Duration::from_secs(10));

    // Test with and without hot path optimization
    for enable_optimization in [false, true].iter() {
        group.bench_with_input(
            BenchmarkId::new("hot_path_enabled", enable_optimization),
            enable_optimization,
            |b, &enable_optimization| {
                b.to_async(&rt).iter(|| async {
                    let config = PerformanceConfig {
                        max_concurrent_operations: 10,
                        cache_size_limit: 1000,
                        pool_size_limit: 100,
                        optimization_threshold_ms: 100,
                        enable_hot_path_optimization: enable_optimization,
                        enable_memory_pooling: true,
                        enable_caching: true,
                    };

                    let engine = PerformanceOptimizationEngine::new(config);

                    // Simulate hot path operations
                    for i in 0..100 {
                        let result = engine
                            .optimize_consciousness_engine(&format!("Hot path test {}", i))
                            .await
                            .unwrap();
                        black_box(result);
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark async vs sync performance
fn bench_async_vs_sync(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("async_vs_sync");
    group.measurement_time(Duration::from_secs(10));

    // Test async operations
    group.bench_function("async_operations", |b| {
        b.to_async(&rt).iter(|| async {
            let config = PerformanceConfig::default();
            let engine = PerformanceOptimizationEngine::new(config);

            // Simulate async operations
            let tasks: Vec<_> = (0..10)
                .map(|i| {
                    let engine = &engine;
                    tokio::spawn(async move {
                        engine
                            .optimize_consciousness_engine(&format!("Async test {}", i))
                            .await
                    })
                })
                .collect();

            // Wait for all tasks
            for task in tasks {
                let result = task.await.unwrap().unwrap();
                black_box(result);
            }
        })
    });

    // Test sync operations
    group.bench_function("sync_operations", |b| {
        b.iter(|| {
            let config = PerformanceConfig::default();
            let engine = PerformanceOptimizationEngine::new(config);

            // Simulate sync operations
            for i in 0..10 {
                let result = format!("Sync test {}", i);
                black_box(result);
            }
        })
    });

    group.finish();
}

/// Benchmark performance report generation
fn bench_performance_report(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("performance_report_generation", |b| {
        b.to_async(&rt).iter(|| async {
            let config = PerformanceConfig::default();
            let engine = PerformanceOptimizationEngine::new(config);

            // Generate some performance data
            for i in 0..100 {
                let result = engine
                    .optimize_consciousness_engine(&format!("Report test {}", i))
                    .await
                    .unwrap();
                black_box(result);
            }

            // Generate performance report
            let report = engine.get_performance_report();
            black_box(report)
        })
    });
}

/// Benchmark memory allocation patterns
fn bench_memory_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation_patterns");
    group.measurement_time(Duration::from_secs(5));

    // Test different allocation sizes
    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("allocation_size", size),
            size,
            |b, &size| {
                b.iter(|| {
                    // Simulate consciousness engine memory allocation patterns
                    let mut data: Vec<String> = Vec::with_capacity(*size);
                    for i in 0..*size {
                        data.push(format!("Consciousness data {}", i));
                    }
                    black_box(data)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark string processing performance
fn bench_string_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_processing");
    group.measurement_time(Duration::from_secs(5));

    // Test different string sizes
    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("string_size", size), size, |b, &size| {
            b.iter(|| {
                let input = "x".repeat(*size);
                let processed = input.to_lowercase();
                let words: Vec<&str> = processed.split_whitespace().collect();
                black_box(words)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_optimized_consciousness_engine,
    bench_memory_pooling,
    bench_caching_performance,
    bench_hot_path_optimization,
    bench_async_vs_sync,
    bench_performance_report,
    bench_memory_allocation_patterns,
    bench_string_processing
);

criterion_main!(benches);
