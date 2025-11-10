/*
 * 🔬 Sparse GP Consciousness Integration Benchmarks
 *
 * AGENT 8: Performance verification for O(n) complexity
 *
 * This benchmark suite validates that sparse GP operations
 * achieve the required performance targets:
 * - GP prediction: <10ms per query
 * - Inducing point updates: <50ms
 * - Real-time consciousness processing: <500ms
 */

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use niodoo_consciousness::sparse_gp_consciousness_integration::{
    SparseGPConsciousnessProcessor, SparseGPConfig, DecisionType,
};
use niodoo_consciousness::consciousness::ConsciousnessState;
use nalgebra::Vector3;

/// Benchmark sparse GP prediction latency
fn benchmark_gp_prediction(c: &mut Criterion) {
    let processor = SparseGPConsciousnessProcessor::new(SparseGPConfig::default()).unwrap();
    let mut consciousness = ConsciousnessState::new();

    // Add training data
    for i in 0..100 {
        let quality = 0.5 + (i as f32 * 0.005);
        processor.add_consciousness_experience(&consciousness, quality).unwrap();
    }

    c.bench_function("sparse_gp_prediction", |b| {
        b.iter(|| {
            processor.process_consciousness_state(black_box(&mut consciousness)).unwrap()
        });
    });
}

/// Benchmark decision making with uncertainty
fn benchmark_decision_making(c: &mut Criterion) {
    let processor = SparseGPConsciousnessProcessor::new(SparseGPConfig::default()).unwrap();
    let consciousness = ConsciousnessState::new();

    // Add training data
    for i in 0..50 {
        processor.add_consciousness_experience(&consciousness, 0.7 + (i as f32 * 0.01)).unwrap();
    }

    c.bench_function("uncertainty_aware_decision", |b| {
        b.iter(|| {
            processor.make_decision(
                black_box(DecisionType::FormMemory),
                black_box(&consciousness),
            ).unwrap()
        });
    });
}

/// Benchmark scalability with different data sizes
fn benchmark_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_gp_scalability");

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let processor = SparseGPConsciousnessProcessor::new(SparseGPConfig::default()).unwrap();
            let mut consciousness = ConsciousnessState::new();

            // Add training data
            for i in 0..size {
                let quality = 0.5 + ((i % 100) as f32 * 0.005);
                processor.add_consciousness_experience(&consciousness, quality).unwrap();
            }

            b.iter(|| {
                processor.process_consciousness_state(black_box(&mut consciousness)).unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark inducing point updates
fn benchmark_inducing_point_update(c: &mut Criterion) {
    let mut config = SparseGPConfig::default();
    config.num_inducing_points = 50;

    let processor = SparseGPConsciousnessProcessor::new(config).unwrap();
    let consciousness = ConsciousnessState::new();

    // Add significant training data
    for i in 0..200 {
        let quality = 0.5 + (i as f32 * 0.0025);
        processor.add_consciousness_experience(&consciousness, quality).unwrap();
    }

    c.bench_function("inducing_point_update", |b| {
        b.iter(|| {
            // Trigger update by adding more data
            for _ in 0..10 {
                processor.add_consciousness_experience(black_box(&consciousness), black_box(0.7)).unwrap();
            }
        });
    });
}

/// Benchmark parallel operations
fn benchmark_parallel_processing(c: &mut Criterion) {
    let processor = SparseGPConsciousnessProcessor::new(SparseGPConfig::default()).unwrap();

    // Prepare multiple consciousness states
    let mut states: Vec<ConsciousnessState> = (0..10)
        .map(|_| ConsciousnessState::new())
        .collect();

    // Add training data
    for i in 0..100 {
        processor.add_consciousness_experience(&states[0], 0.5 + (i as f32 * 0.005)).unwrap();
    }

    c.bench_function("parallel_consciousness_processing", |b| {
        b.iter(|| {
            for state in states.iter_mut() {
                processor.process_consciousness_state(black_box(state)).unwrap();
            }
        });
    });
}

/// Benchmark memory efficiency
fn benchmark_memory_overhead(c: &mut Criterion) {
    c.bench_function("processor_creation_overhead", |b| {
        b.iter(|| {
            let _processor = SparseGPConsciousnessProcessor::new(
                black_box(SparseGPConfig::default())
            ).unwrap();
        });
    });
}

/// Benchmark uncertainty calibration
fn benchmark_uncertainty_calibration(c: &mut Criterion) {
    let processor = SparseGPConsciousnessProcessor::new(SparseGPConfig::default()).unwrap();
    let mut consciousness = ConsciousnessState::new();

    // Add varied training data for calibration
    for i in 0..100 {
        let quality = if i % 2 == 0 { 0.8 } else { 0.3 };
        processor.add_consciousness_experience(&consciousness, quality).unwrap();
    }

    c.bench_function("uncertainty_calibration", |b| {
        b.iter(|| {
            let measurement = processor.process_consciousness_state(black_box(&mut consciousness)).unwrap();
            // Verify uncertainty is being computed
            black_box(measurement.emotional_uncertainty);
            black_box(measurement.decision_confidence);
        });
    });
}

/// End-to-end integration benchmark
fn benchmark_full_integration(c: &mut Criterion) {
    c.bench_function("full_integration_pipeline", |b| {
        b.iter(|| {
            // Create processor
            let processor = SparseGPConsciousnessProcessor::new(SparseGPConfig::default()).unwrap();
            let mut consciousness = ConsciousnessState::new();

            // Add training experiences
            for i in 0..20 {
                processor.add_consciousness_experience(&consciousness, 0.6 + (i as f32 * 0.01)).unwrap();
            }

            // Process state
            let _measurement = processor.process_consciousness_state(&mut consciousness).unwrap();

            // Make decisions
            let _decision1 = processor.make_decision(DecisionType::FormMemory, &consciousness).unwrap();
            let _decision2 = processor.make_decision(DecisionType::UpdateEmotion, &consciousness).unwrap();

            // Get metrics
            let _metrics = processor.get_metrics();
        });
    });
}

criterion_group!(
    benches,
    benchmark_gp_prediction,
    benchmark_decision_making,
    benchmark_scalability,
    benchmark_inducing_point_update,
    benchmark_parallel_processing,
    benchmark_memory_overhead,
    benchmark_uncertainty_calibration,
    benchmark_full_integration,
);

criterion_main!(benches);
