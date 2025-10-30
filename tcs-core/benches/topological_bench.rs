// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::Rng;
use std::time::Instant;
use tcs_core::TopologicalEngine;

fn bench_potential_computation(c: &mut Criterion) {
    let engine = TopologicalEngine::new(512);
    let state: Vec<f32> = (0..512 * 64).map(|_| rand::random::<f32>()).collect();

    c.bench_function("potential_computation", |b| {
        b.iter(|| {
            let _phi = engine.potential(black_box(&state));
        });
    });
}

fn bench_persistence_homology(c: &mut Criterion) {
    let engine = TopologicalEngine::new(512);
    let state: Vec<f32> = (0..512 * 64).map(|_| rand::random::<f32>()).collect();

    c.bench_function("persistence_homology", |b| {
        b.iter(|| {
            let diagram = engine.compute_persistence(black_box(&state));
            let _pe = diagram.entropy();
            let betti = diagram.betti_numbers();
            black_box(betti);
        });
    });
}

fn bench_lora_prediction(c: &mut Criterion) {
    let engine = TopologicalEngine::new(512);
    let state: Vec<f32> = (0..512 * 64).map(|_| rand::random::<f32>()).collect();

    c.bench_function("lora_prediction", |b| {
        b.iter(|| {
            let _reward = engine.predict_reward(black_box(&state)).unwrap();
        });
    });
}

fn bench_evolution_step(c: &mut Criterion) {
    let engine = TopologicalEngine::new(128);
    let population: Vec<Vec<f32>> = (0..32)
        .map(|_| (0..128).map(|_| rand::random::<f32>()).collect())
        .collect();

    c.bench_function("evolution_one_generation", |b| {
        b.iter(|| {
            let pop_clone = population.clone();
            let _best = engine.evolve(black_box(pop_clone), 1).unwrap();
        });
    });
}

fn bench_large_tda(c: &mut Criterion) {
    let engine = TopologicalEngine::new(1024);
    let state: Vec<f32> = (0..1024 * 1000).map(|_| rand::random::<f32>()).collect();

    c.bench_function("large_tda_phlite", |b| {
        b.iter(|| engine.compute_persistence(black_box(&state)));
    });
}

fn bench_scale_nsga(c: &mut Criterion) {
    let engine = TopologicalEngine::new(256);
    let pop: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..256).map(|_| rand::random::<f32>()).collect())
        .collect();

    c.bench_function("nsga_pop100_gen10", |b| {
        b.iter(|| engine.evolve(black_box(pop.clone()), 10).unwrap());
    });
}

criterion_group!(
    benches,
    bench_potential_computation,
    bench_persistence_homology,
    bench_lora_prediction,
    bench_evolution_step,
    bench_large_tda,
    bench_scale_nsga
);
criterion_main!(benches);
