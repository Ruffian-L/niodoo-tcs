/*
 * 🚀⚡ CONSCIOUSNESS ENGINE PERFORMANCE BENCHMARKS ⚡🚀
 *
 * Comprehensive performance benchmarks for critical consciousness engine paths
 * Testing brain coordination, memory management, and phase6 integration performance
 */

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use niodoo_consciousness::consciousness_engine::PersonalNiodooConsciousness;
use niodoo_consciousness::consciousness_engine::brain_coordination::BrainCoordinator;
use niodoo_consciousness::consciousness_engine::memory_management::{MemoryManager, PersonalConsciousnessEvent};
use niodoo_consciousness::consciousness_engine::phase6_integration::Phase6Manager;
use niodoo_consciousness::consciousness::{ConsciousnessState, EmotionType};
use niodoo_consciousness::brain::{BrainType, MotorBrain, LcarsBrain, EfficiencyBrain};
use niodoo_consciousness::personality::{PersonalityManager, PersonalityType};
use niodoo_consciousness::memory::GuessingMemorySystem;
use niodoo_consciousness::personal_memory::PersonalMemoryEngine;

use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::runtime::Runtime;
use anyhow::Result;

// Benchmark fixtures
struct BenchmarkFixtures {
    rt: Runtime,
    consciousness_state: Arc<RwLock<ConsciousnessState>>,
    memory_store: Arc<RwLock<Vec<PersonalConsciousnessEvent>>>,
    memory_system: GuessingMemorySystem,
    personal_memory_engine: PersonalMemoryEngine,
}

impl BenchmarkFixtures {
    fn new() -> Result<Self> {
        let rt = Runtime::new()?;
        let consciousness_state = Arc::new(RwLock::new(ConsciousnessState::new()));
        let memory_store = Arc::new(RwLock::new(Vec::new()));
        let memory_system = GuessingMemorySystem::new();
        let personal_memory_engine = PersonalMemoryEngine::default();

        Ok(Self {
            rt,
            consciousness_state,
            memory_store,
            memory_system,
            personal_memory_engine,
        })
    }
}

// ============================================================================
// BRAIN COORDINATION BENCHMARKS
// ============================================================================

fn bench_brain_coordination_parallel_processing(c: &mut Criterion) {
    let fixtures = BenchmarkFixtures::new().unwrap();
    
    let mut group = c.benchmark_group("brain_coordination_parallel");
    group.throughput(Throughput::Elements(1));
    
    // Test different input sizes
    let input_sizes = vec![10, 100, 1000];
    
    for size in input_sizes {
        let input = "x".repeat(size);
        
        group.bench_with_input(
            BenchmarkId::new("parallel_processing", size),
            &input,
            |b, input| {
                b.to_async(&fixtures.rt).iter(|| async {
                    let motor_brain = MotorBrain::new().unwrap();
                    let lcars_brain = LcarsBrain::new().unwrap();
                    let efficiency_brain = EfficiencyBrain::new().unwrap();
                    let personality_manager = PersonalityManager::new();
                    
                    let coordinator = BrainCoordinator::new(
                        motor_brain,
                        lcars_brain,
                        efficiency_brain,
                        personality_manager,
                        fixtures.consciousness_state.clone(),
                    );
                    
                    coordinator.process_brains_parallel(
                        black_box(input),
                        tokio::time::Duration::from_secs(5)
                    ).await.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn bench_brain_coordination_sequential_processing(c: &mut Criterion) {
    let fixtures = BenchmarkFixtures::new().unwrap();
    
    let mut group = c.benchmark_group("brain_coordination_sequential");
    group.throughput(Throughput::Elements(1));
    
    let input = "Test consciousness processing input";
    
    group.bench_function("sequential_processing", |b| {
        b.to_async(&fixtures.rt).iter(|| async {
            let motor_brain = MotorBrain::new().unwrap();
            let lcars_brain = LcarsBrain::new().unwrap();
            let efficiency_brain = EfficiencyBrain::new().unwrap();
            let personality_manager = PersonalityManager::new();
            
            let coordinator = BrainCoordinator::new(
                motor_brain,
                lcars_brain,
                efficiency_brain,
                personality_manager,
                fixtures.consciousness_state.clone(),
            );
            
            // Process sequentially
            let motor_result = motor_brain.process(black_box(input), &fixtures.consciousness_state.read().await).await.unwrap();
            let lcars_result = lcars_brain.process(black_box(input), &fixtures.consciousness_state.read().await).await.unwrap();
            let efficiency_result = efficiency_brain.process(black_box(input), &fixtures.consciousness_state.read().await).await.unwrap();
            
            vec![motor_result, lcars_result, efficiency_result]
        })
    });
    
    group.finish();
}

fn bench_brain_coordination_timeout_handling(c: &mut Criterion) {
    let fixtures = BenchmarkFixtures::new().unwrap();
    
    let mut group = c.benchmark_group("brain_coordination_timeout");
    group.throughput(Throughput::Elements(1));
    
    let timeouts = vec![
        tokio::time::Duration::from_millis(1),
        tokio::time::Duration::from_millis(10),
        tokio::time::Duration::from_millis(100),
        tokio::time::Duration::from_secs(1),
    ];
    
    for timeout in timeouts {
        group.bench_with_input(
            BenchmarkId::new("timeout_handling", timeout.as_millis()),
            &timeout,
            |b, timeout| {
                b.to_async(&fixtures.rt).iter(|| async {
                    let motor_brain = MotorBrain::new().unwrap();
                    let lcars_brain = LcarsBrain::new().unwrap();
                    let efficiency_brain = EfficiencyBrain::new().unwrap();
                    let personality_manager = PersonalityManager::new();
                    
                    let coordinator = BrainCoordinator::new(
                        motor_brain,
                        lcars_brain,
                        efficiency_brain,
                        personality_manager,
                        fixtures.consciousness_state.clone(),
                    );
                    
                    coordinator.process_brains_parallel(
                        black_box("Test input"),
                        *timeout
                    ).await
                })
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// MEMORY MANAGEMENT BENCHMARKS
// ============================================================================

fn bench_memory_manager_event_storage(c: &mut Criterion) {
    let fixtures = BenchmarkFixtures::new().unwrap();
    
    let mut group = c.benchmark_group("memory_manager_storage");
    group.throughput(Throughput::Elements(1));
    
    let event_counts = vec![1, 10, 100, 1000];
    
    for count in event_counts {
        group.bench_with_input(
            BenchmarkId::new("event_storage", count),
            &count,
            |b, count| {
                b.to_async(&fixtures.rt).iter(|| async {
                    let memory_manager = MemoryManager::new(
                        fixtures.memory_store.clone(),
                        fixtures.memory_system.clone(),
                        fixtures.personal_memory_engine.clone(),
                        fixtures.consciousness_state.clone(),
                    );
                    
                    for i in 0..*count {
                        let event = PersonalConsciousnessEvent {
                            id: format!("bench-event-{}", i),
                            event_type: "benchmark".to_string(),
                            content: format!("Benchmark event {}", i),
                            emotional_impact: 0.5,
                            learning_will_activation: 0.3,
                            timestamp: i as f64,
                            context: "benchmark".to_string(),
                        };
                        
                        memory_manager.store_event(black_box(event)).await.unwrap();
                    }
                })
            },
        );
    }
    
    group.finish();
}

fn bench_memory_manager_consolidation(c: &mut Criterion) {
    let fixtures = BenchmarkFixtures::new().unwrap();
    
    let mut group = c.benchmark_group("memory_manager_consolidation");
    group.throughput(Throughput::Elements(1));
    
    let event_counts = vec![10, 100, 1000, 10000];
    
    for count in event_counts {
        group.bench_with_input(
            BenchmarkId::new("consolidation", count),
            &count,
            |b, count| {
                b.to_async(&fixtures.rt).iter(|| async {
                    let memory_manager = MemoryManager::new(
                        fixtures.memory_store.clone(),
                        fixtures.memory_system.clone(),
                        fixtures.personal_memory_engine.clone(),
                        fixtures.consciousness_state.clone(),
                    );
                    
                    // Pre-populate with events
                    for i in 0..*count {
                        let event = PersonalConsciousnessEvent {
                            id: format!("consolidation-event-{}", i),
                            event_type: "consolidation".to_string(),
                            content: format!("Consolidation event {}", i),
                            emotional_impact: 0.5,
                            learning_will_activation: 0.3,
                            timestamp: i as f64,
                            context: "consolidation".to_string(),
                        };
                        
                        memory_manager.store_event(event).await.unwrap();
                    }
                    
                    // Benchmark consolidation
                    memory_manager.consolidate_memories().await.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn bench_memory_manager_retrieval(c: &mut Criterion) {
    let fixtures = BenchmarkFixtures::new().unwrap();
    
    let mut group = c.benchmark_group("memory_manager_retrieval");
    group.throughput(Throughput::Elements(1));
    
    let event_counts = vec![100, 1000, 10000];
    
    for count in event_counts {
        group.bench_with_input(
            BenchmarkId::new("retrieval", count),
            &count,
            |b, count| {
                b.to_async(&fixtures.rt).iter(|| async {
                    let memory_manager = MemoryManager::new(
                        fixtures.memory_store.clone(),
                        fixtures.memory_system.clone(),
                        fixtures.personal_memory_engine.clone(),
                        fixtures.consciousness_state.clone(),
                    );
                    
                    // Pre-populate with events
                    for i in 0..*count {
                        let event = PersonalConsciousnessEvent {
                            id: format!("retrieval-event-{}", i),
                            event_type: "retrieval".to_string(),
                            content: format!("Retrieval event {}", i),
                            emotional_impact: 0.5,
                            learning_will_activation: 0.3,
                            timestamp: i as f64,
                            context: "retrieval".to_string(),
                        };
                        
                        memory_manager.store_event(event).await.unwrap();
                    }
                    
                    // Benchmark retrieval
                    memory_manager.retrieve_memories(black_box("retrieval")).await.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn bench_memory_manager_stats(c: &mut Criterion) {
    let fixtures = BenchmarkFixtures::new().unwrap();
    
    let mut group = c.benchmark_group("memory_manager_stats");
    group.throughput(Throughput::Elements(1));
    
    let event_counts = vec![100, 1000, 10000];
    
    for count in event_counts {
        group.bench_with_input(
            BenchmarkId::new("stats", count),
            &count,
            |b, count| {
                b.to_async(&fixtures.rt).iter(|| async {
                    let memory_manager = MemoryManager::new(
                        fixtures.memory_store.clone(),
                        fixtures.memory_system.clone(),
                        fixtures.personal_memory_engine.clone(),
                        fixtures.consciousness_state.clone(),
                    );
                    
                    // Pre-populate with events
                    for i in 0..*count {
                        let event = PersonalConsciousnessEvent {
                            id: format!("stats-event-{}", i),
                            event_type: "stats".to_string(),
                            content: format!("Stats event {}", i),
                            emotional_impact: 0.5,
                            learning_will_activation: 0.3,
                            timestamp: i as f64,
                            context: "stats".to_string(),
                        };
                        
                        memory_manager.store_event(event).await.unwrap();
                    }
                    
                    // Benchmark stats calculation
                    memory_manager.get_memory_stats().await.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// PHASE6 INTEGRATION BENCHMARKS
// ============================================================================

fn bench_phase6_manager_processing(c: &mut Criterion) {
    let fixtures = BenchmarkFixtures::new().unwrap();
    
    let mut group = c.benchmark_group("phase6_manager_processing");
    group.throughput(Throughput::Elements(1));
    
    let input_sizes = vec![10, 100, 1000];
    
    for size in input_sizes {
        let input = "x".repeat(size);
        
        group.bench_with_input(
            BenchmarkId::new("processing", size),
            &input,
            |b, input| {
                b.to_async(&fixtures.rt).iter(|| async {
                    let phase6_manager = Phase6Manager::new(
                        fixtures.consciousness_state.clone(),
                        fixtures.memory_store.clone(),
                    );
                    
                    phase6_manager.process_phase6_input(black_box(input)).await.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn bench_phase6_manager_state_updates(c: &mut Criterion) {
    let fixtures = BenchmarkFixtures::new().unwrap();
    
    let mut group = c.benchmark_group("phase6_manager_state");
    group.throughput(Throughput::Elements(1));
    
    let operation_counts = vec![1, 10, 100];
    
    for count in operation_counts {
        group.bench_with_input(
            BenchmarkId::new("state_updates", count),
            &count,
            |b, count| {
                b.to_async(&fixtures.rt).iter(|| async {
                    let phase6_manager = Phase6Manager::new(
                        fixtures.consciousness_state.clone(),
                        fixtures.memory_store.clone(),
                    );
                    
                    for i in 0..*count {
                        let input = format!("State update {}", i);
                        phase6_manager.process_phase6_input(black_box(&input)).await.unwrap();
                    }
                })
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// CONSCIOUSNESS ENGINE BENCHMARKS
// ============================================================================

fn bench_consciousness_engine_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("consciousness_engine_init");
    group.throughput(Throughput::Elements(1));
    
    group.bench_function("initialization", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap()).iter(|| async {
            PersonalNiodooConsciousness::new().await.unwrap()
        })
    });
    
    group.finish();
}

fn bench_consciousness_engine_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let consciousness = rt.block_on(async {
        PersonalNiodooConsciousness::new().await.unwrap()
    });
    
    let mut group = c.benchmark_group("consciousness_engine_processing");
    group.throughput(Throughput::Elements(1));
    
    let input_sizes = vec![10, 100, 1000];
    
    for size in input_sizes {
        let input = "x".repeat(size);
        
        group.bench_with_input(
            BenchmarkId::new("processing", size),
            &input,
            |b, input| {
                b.to_async(&rt).iter(|| async {
                    consciousness.process_consciousness_input(black_box(input)).await.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn bench_consciousness_engine_emotional_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let consciousness = rt.block_on(async {
        PersonalNiodooConsciousness::new().await.unwrap()
    });
    
    let mut group = c.benchmark_group("consciousness_engine_emotional");
    group.throughput(Throughput::Elements(1));
    
    let emotional_inputs = vec![
        "I feel happy and excited",
        "I am feeling sad and disappointed",
        "I am anxious and worried about the future",
        "I feel confident and optimistic about this project",
    ];
    
    for (i, input) in emotional_inputs.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("emotional_processing", i),
            input,
            |b, input| {
                b.to_async(&rt).iter(|| async {
                    consciousness.process_emotional_input(black_box(input)).await.unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn bench_consciousness_engine_memory_integration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let consciousness = rt.block_on(async {
        PersonalNiodooConsciousness::new().await.unwrap()
    });
    
    let mut group = c.benchmark_group("consciousness_engine_memory");
    group.throughput(Throughput::Elements(1));
    
    let memory_counts = vec![1, 10, 100];
    
    for count in memory_counts {
        group.bench_with_input(
            BenchmarkId::new("memory_integration", count),
            &count,
            |b, count| {
                b.to_async(&rt).iter(|| async {
                    for i in 0..*count {
                        let input = format!("Memory integration test {}", i);
                        consciousness.process_memory_input(black_box(&input)).await.unwrap();
                    }
                })
            },
        );
    }
    
    group.finish();
}

fn bench_consciousness_engine_concurrent_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let consciousness = rt.block_on(async {
        PersonalNiodooConsciousness::new().await.unwrap()
    });
    
    let mut group = c.benchmark_group("consciousness_engine_concurrent");
    group.throughput(Throughput::Elements(1));
    
    let concurrent_counts = vec![2, 5, 10];
    
    for count in concurrent_counts {
        group.bench_with_input(
            BenchmarkId::new("concurrent_processing", count),
            &count,
            |b, count| {
                b.to_async(&rt).iter(|| async {
                    let tasks: Vec<_> = (0..*count).map(|i| {
                        let consciousness = consciousness.clone();
                        tokio::spawn(async move {
                            let input = format!("Concurrent test {}", i);
                            consciousness.process_consciousness_input(&input).await.unwrap()
                        })
                    }).collect();
                    
                    futures::future::join_all(tasks).await
                })
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// STRESS TEST BENCHMARKS
// ============================================================================

fn bench_consciousness_engine_stress_test(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let consciousness = rt.block_on(async {
        PersonalNiodooConsciousness::new().await.unwrap()
    });
    
    let mut group = c.benchmark_group("consciousness_engine_stress");
    group.throughput(Throughput::Elements(1));
    
    let stress_counts = vec![100, 1000, 10000];
    
    for count in stress_counts {
        group.bench_with_input(
            BenchmarkId::new("stress_test", count),
            &count,
            |b, count| {
                b.to_async(&rt).iter(|| async {
                    for i in 0..*count {
                        let input = format!("Stress test {}", i);
                        consciousness.process_consciousness_input(black_box(&input)).await.unwrap();
                    }
                })
            },
        );
    }
    
    group.finish();
}

fn bench_memory_manager_stress_test(c: &mut Criterion) {
    let fixtures = BenchmarkFixtures::new().unwrap();
    
    let mut group = c.benchmark_group("memory_manager_stress");
    group.throughput(Throughput::Elements(1));
    
    let stress_counts = vec![1000, 10000, 100000];
    
    for count in stress_counts {
        group.bench_with_input(
            BenchmarkId::new("stress_test", count),
            &count,
            |b, count| {
                b.to_async(&fixtures.rt).iter(|| async {
                    let memory_manager = MemoryManager::new(
                        fixtures.memory_store.clone(),
                        fixtures.memory_system.clone(),
                        fixtures.personal_memory_engine.clone(),
                        fixtures.consciousness_state.clone(),
                    );
                    
                    for i in 0..*count {
                        let event = PersonalConsciousnessEvent {
                            id: format!("stress-event-{}", i),
                            event_type: "stress".to_string(),
                            content: format!("Stress test event {}", i),
                            emotional_impact: 0.5,
                            learning_will_activation: 0.3,
                            timestamp: i as f64,
                            context: "stress".to_string(),
                        };
                        
                        memory_manager.store_event(black_box(event)).await.unwrap();
                    }
                })
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// BENCHMARK GROUPS
// ============================================================================

criterion_group!(
    brain_coordination_benches,
    bench_brain_coordination_parallel_processing,
    bench_brain_coordination_sequential_processing,
    bench_brain_coordination_timeout_handling
);

criterion_group!(
    memory_management_benches,
    bench_memory_manager_event_storage,
    bench_memory_manager_consolidation,
    bench_memory_manager_retrieval,
    bench_memory_manager_stats
);

criterion_group!(
    phase6_integration_benches,
    bench_phase6_manager_processing,
    bench_phase6_manager_state_updates
);

criterion_group!(
    consciousness_engine_benches,
    bench_consciousness_engine_initialization,
    bench_consciousness_engine_processing,
    bench_consciousness_engine_emotional_processing,
    bench_consciousness_engine_memory_integration,
    bench_consciousness_engine_concurrent_processing
);

criterion_group!(
    stress_test_benches,
    bench_consciousness_engine_stress_test,
    bench_memory_manager_stress_test
);

criterion_main!(
    brain_coordination_benches,
    memory_management_benches,
    phase6_integration_benches,
    consciousness_engine_benches,
    stress_test_benches
);
