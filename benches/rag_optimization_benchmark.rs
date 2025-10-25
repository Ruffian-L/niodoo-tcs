/*
use tracing::{info, error, warn};
 * 🚀 RAG OPTIMIZATION BENCHMARK
 *
 * Compares baseline vs optimized RAG retrieval performance
 *
 * Measures:
 * 1. Baseline retrieval (no optimizations)
 * 2. Cached retrieval (embedding cache)
 * 3. Optimized retrieval (precomputed norms + async)
 *
 * Target: <50ms per query with optimizations
 *
 * Run with: cargo bench --bench rag_optimization_benchmark
 */

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use niodoo_consciousness::config::AppConfig;
use niodoo_consciousness::dual_mobius_gaussian::ConsciousnessState;
use niodoo_consciousness::rag::{
    Document, EmbeddingGenerator, OptimizedRetrievalEngine, RetrievalEngine,
};
use std::collections::HashMap;
use std::time::Duration;

/// Generate test documents
fn generate_test_docs(count: usize) -> Vec<Document> {
    (0..count)
        .map(|i| Document {
            id: format!("doc_{}", i),
            content: format!(
                "Test document {} about consciousness and Möbius transformations in AI systems",
                i
            ),
            metadata: HashMap::new(),
            embedding: None,
            created_at: chrono::Utc::now(),
            entities: vec![],
            chunk_id: None,
            source_type: Some("benchmark".to_string()),
            resonance_hint: Some(0.5),
            token_count: 10,
        })
        .collect()
}

/// Benchmark baseline retrieval (cold cache)
fn bench_baseline_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_retrieval");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    let config = AppConfig::default();
    let docs = generate_test_docs(10);

    for doc_count in [5, 10, 20] {
        group.throughput(Throughput::Elements(doc_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_docs", doc_count)),
            &doc_count,
            |b, &count| {
                b.iter(|| {
                    let mut engine = RetrievalEngine::new(384, 5, config.clone());
                    let test_docs = generate_test_docs(count);

                    // This will be slow without cache
                    let _ = black_box(engine.add_documents(test_docs));

                    let mut state = ConsciousnessState::default();
                    let _ = black_box(engine.retrieve("test query", &mut state));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark cached retrieval (warm cache)
fn bench_cached_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_retrieval");
    group.measurement_time(Duration::from_secs(10));

    let config = AppConfig::default();
    let mut engine = RetrievalEngine::new(384, 5, config.clone());
    let docs = generate_test_docs(10);

    // Warm up cache
    let _ = engine.add_documents(docs);
    let mut state = ConsciousnessState::default();
    let _ = engine.retrieve("test query", &mut state);

    group.bench_function("warm_cache_query", |b| {
        b.iter(|| {
            let mut state = ConsciousnessState::default();
            let _ = black_box(engine.retrieve("test query", &mut state));
        });
    });

    group.finish();
}

/// Benchmark optimized retrieval with precomputed norms
fn bench_optimized_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimized_retrieval");
    group.measurement_time(Duration::from_secs(10));

    let config = AppConfig::default();

    for doc_count in [10, 50, 100] {
        group.throughput(Throughput::Elements(doc_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_docs", doc_count)),
            &doc_count,
            |b, &count| {
                let engine = OptimizedRetrievalEngine::new(384, 5, config.clone());
                let docs = generate_test_docs(count);

                // Add documents with precomputed metadata
                let _ = engine.add_documents(docs);

                b.iter(|| {
                    let mut state = ConsciousnessState::default();
                    let _ = black_box(engine.retrieve_sync("test query", &mut state));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark async retrieval (non-blocking)
fn bench_async_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_retrieval");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    let config = AppConfig::default();
    let rt = tokio::runtime::Runtime::new().unwrap();

    for doc_count in [10, 50] {
        group.throughput(Throughput::Elements(doc_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_docs", doc_count)),
            &doc_count,
            |b, &count| {
                let engine = OptimizedRetrievalEngine::new(384, 5, config.clone());
                let docs = generate_test_docs(count);
                let _ = engine.add_documents(docs);

                b.to_async(&rt).iter(|| async {
                    let state = ConsciousnessState::default();
                    let _ = black_box(
                        engine
                            .retrieve_async("test query".to_string(), state)
                            .await,
                    );
                });
            },
        );
    }

    group.finish();
}

/// Comprehensive comparison: baseline vs optimized
fn bench_comprehensive_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comprehensive_comparison");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(30);

    let config = AppConfig::default();
    let doc_count = 20;
    let docs = generate_test_docs(doc_count);

    tracing::info!("\n🚀 RAG OPTIMIZATION BENCHMARK");
    tracing::info!("{}", "=".repeat(70));

    // Baseline
    group.bench_function("1_baseline_cold", |b| {
        b.iter(|| {
            let mut engine = RetrievalEngine::new(384, 5, config.clone());
            let test_docs = generate_test_docs(doc_count);
            let _ = engine.add_documents(test_docs);
            let mut state = ConsciousnessState::default();
            let _ = black_box(engine.retrieve("consciousness", &mut state));
        });
    });

    // Cached
    let mut engine_cached = RetrievalEngine::new(384, 5, config.clone());
    let _ = engine_cached.add_documents(docs.clone());
    group.bench_function("2_cached_warm", |b| {
        b.iter(|| {
            let mut state = ConsciousnessState::default();
            let _ = black_box(engine_cached.retrieve("consciousness", &mut state));
        });
    });

    // Optimized with precomputed norms
    let engine_optimized = OptimizedRetrievalEngine::new(384, 5, config.clone());
    let _ = engine_optimized.add_documents(docs.clone());
    group.bench_function("3_optimized_norms", |b| {
        b.iter(|| {
            let mut state = ConsciousnessState::default();
            let _ = black_box(engine_optimized.retrieve_sync("consciousness", &mut state));
        });
    });

    // Async (non-blocking)
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function("4_async_nonblocking", |b| {
        b.to_async(&rt).iter(|| async {
            let state = ConsciousnessState::default();
            let _ = black_box(
                engine_optimized
                    .retrieve_async("consciousness".to_string(), state)
                    .await,
            );
        });
    });

    group.finish();

    tracing::info!("\n✅ Benchmark complete! Check criterion report for detailed results.");
    tracing::info!("   Target: <50ms per query with optimizations");
}

/// Benchmark KL divergence calculation for Bayesian surprise
fn bench_kl_divergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("kl_divergence");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(15);
    
    for size in [64, 128, 256, 512, 1024].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("bayesian_surprise", size), size, |b, &size| {
            b.iter(|| {
                // Generate test probability distributions
                let p: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) / (size as f32 * 2.0)).collect();
                let q: Vec<f32> = (0..size).map(|i| ((i as f32 + 1.0) * 0.8) / (size as f32 * 2.0)).collect();
                
                // Calculate KL divergence: D_KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
                let kl_div = black_box(p.iter().zip(q.iter())
                    .map(|(p_val, q_val)| {
                        if *q_val > 0.0 && *p_val > 0.0 {
                            p_val * (p_val / q_val).ln()
                        } else {
                            0.0
                        }
                    })
                    .sum::<f32>());
                
                kl_div
            });
        });
    }
    group.finish();
}

/// Benchmark Wundt curve calculation
fn bench_wundt_curve(c: &mut Criterion) {
    let mut group = c.benchmark_group("wundt_curve");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(12);
    
    for size in [32, 64, 128, 256].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("curve_calculation", size), size, |b, &size| {
            b.iter(|| {
                // Generate test arousal values
                let arousal_values: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
                
                // Calculate Wundt curve: pleasure = arousal * (1 - arousal)
                let wundt_values = black_box(arousal_values.iter()
                    .map(|arousal| arousal * (1.0 - arousal))
                    .collect::<Vec<f32>>());
                
                wundt_values
            });
        });
    }
    group.finish();
}

/// Benchmark divisive normalization edge cases
fn bench_divisive_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("divisive_norm");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(12);
    
    for size in [16, 32, 64, 128].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("normalization", size), size, |b, &size| {
            b.iter(|| {
                // Generate test activation values including edge cases
                let activations: Vec<f32> = (0..size).map(|i| {
                    match i % 4 {
                        0 => 0.0,           // Zero activation
                        1 => 0.001,         // Near-zero activation
                        2 => 1.0,           // Normal activation
                        _ => 100.0,        // High activation
                    }
                }).collect();
                
                // Divisive normalization: y_i = x_i / (σ + Σ x_j)
                let sigma = 0.1; // Semi-saturation constant
                let sum_activations: f32 = activations.iter().sum();
                let denominator = sigma + sum_activations;
                
                let normalized = black_box(activations.iter()
                    .map(|&x| if denominator > 0.0 { x / denominator } else { 0.0 })
                    .collect::<Vec<f32>>());
                
                normalized
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_baseline_retrieval,
    bench_cached_retrieval,
    bench_optimized_retrieval,
    bench_async_retrieval,
    bench_comprehensive_comparison,
    bench_kl_divergence,
    bench_wundt_curve,
    bench_divisive_norm
);

criterion_main!(benches);
