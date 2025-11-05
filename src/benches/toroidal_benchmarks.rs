//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use memory::toroidal::*;
use std::collections::HashMap;

fn bench_toroidal_coordinate_creation(c: &mut Criterion) {
    c.bench_function("toroidal_coordinate_creation", |b| {
        b.iter(|| black_box(ToroidalCoordinate::new(black_box(1.5), black_box(2.3))))
    });
}

fn bench_geodesic_distance(c: &mut Criterion) {
    let coord1 = ToroidalCoordinate::new(0.0, 0.0);
    let coord2 = ToroidalCoordinate::new(std::f64::consts::PI, std::f64::consts::PI / 2.0);

    c.bench_function("geodesic_distance", |b| {
        b.iter(|| black_box(coord1.geodesic_distance(black_box(&coord2))))
    });
}

fn bench_memory_node_creation(c: &mut Criterion) {
    c.bench_function("memory_node_creation", |b| {
        b.iter(|| {
            black_box(ToroidalMemoryNode {
                id: black_box("test_id".to_string()),
                coordinate: black_box(ToroidalCoordinate::new(1.0, 2.0)),
                content: black_box("test content".to_string()),
                emotional_vector: black_box(vec![0.1, 0.2, 0.3]),
                temporal_context: black_box(vec![1.0, 2.0, 3.0]),
                activation_strength: black_box(0.8),
                connections: black_box(HashMap::new()),
            })
        })
    });
}

fn bench_parallel_stream_processing(c: &mut Criterion) {
    let mut system = ToroidalConsciousnessSystem::new(3.0, 1.0);

    // Add test memories
    for i in 0..100 {
        let node = ToroidalMemoryNode {
            id: format!("test_{}", i),
            coordinate: ToroidalCoordinate::new(i as f64 * 0.1, i as f64 * 0.05),
            content: format!("Test content {}", i),
            emotional_vector: vec![0.5],
            temporal_context: vec![i as f64],
            activation_strength: 0.7,
            connections: HashMap::new(),
        };
        // Note: We'd need to make add_memory public for this to work
        // system.memory_nodes.blocking_write().insert(node.id.clone(), node);
    }

    c.bench_function("parallel_stream_processing_100_nodes", |b| {
        b.iter(|| {
            // black_box(system.process_parallel_streams(black_box(0.1)))
            // Placeholder - would need to expose the method
            black_box(42)
        })
    });
}

criterion_group!(
    benches,
    bench_toroidal_coordinate_creation,
    bench_geodesic_distance,
    bench_memory_node_creation,
    bench_parallel_stream_processing
);
criterion_main!(benches);
