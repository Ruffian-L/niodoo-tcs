//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use crate::memory::mobius::*;
use crate::memory::toroidal::*;
use crate::memory::*;
// use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_mobius_memory_creation(c: &mut Criterion) {
    c.bench_function("mobius_memory_creation", |b| {
        b.iter(|| {
            black_box(MemoryFragment {
                content: black_box("test memory content".to_string()),
                layer: black_box(MemoryLayer::Semantic),
                relevance: black_box(0.8),
                timestamp: black_box(1.0),
            })
        })
    });
}

fn bench_mobius_traversal(c: &mut Criterion) {
    let mut system = MobiusMemorySystem::new();

    // Add test memories
    for i in 0..100 {
        let fragment = MemoryFragment {
            content: format!("Memory {}", i),
            layer: MemoryLayer::Working,
            relevance: 0.5,
            timestamp: i as f64,
        };
        // Note: We'd need to expose add_memory or similar method
        // system.add_memory(fragment);
    }

    c.bench_function("mobius_traversal_100_memories", |b| {
        b.iter(|| {
            // black_box(system.bi_directional_traverse(black_box("test query"), black_box("neutral")))
            // Placeholder - would need to expose the method
            black_box(42)
        })
    });
}

fn bench_toroidal_memory_migration(c: &mut Criterion) {
    let mobius_memories = vec![
        MemoryFragment {
            content: "Test memory 1".to_string(),
            layer: MemoryLayer::Semantic,
            relevance: 0.8,
            timestamp: 1.0,
        },
        MemoryFragment {
            content: "Test memory 2".to_string(),
            layer: MemoryLayer::Episodic,
            relevance: 0.6,
            timestamp: 2.0,
        },
    ];

    c.bench_function("toroidal_memory_migration", |b| {
        b.iter(|| {
            // Note: We'd need to expose migrate_mobius_to_torus function
            // black_box(migrate_mobius_to_torus(black_box(mobius_memories.clone())))
            // Placeholder - would need to expose the function
            black_box(42)
        })
    });
}

fn bench_memory_bounds_checking(c: &mut Criterion) {
    let mut system = MobiusMemorySystem::new();

    c.bench_function("memory_bounds_checking_1000_additions", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let fragment = MemoryFragment {
                    content: format!("Memory {}", i),
                    layer: MemoryLayer::Working,
                    relevance: 0.5,
                    timestamp: i as f64,
                };
                // Note: We'd need to expose add_memory method
                // system.add_memory(fragment);
                // assert!(system.persistent_memories.len() <= 10000);
            }
            black_box(42)
        })
    });
}

criterion_group!(
    benches,
    bench_mobius_memory_creation,
    bench_mobius_traversal,
    bench_toroidal_memory_migration,
    bench_memory_bounds_checking
);
criterion_main!(benches);
