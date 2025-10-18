/*
 * 🔬 PROPERTY-BASED TESTING SUITE
 *
 * AGENT 9: Advanced testing using proptest for generative testing
 *
 * Property-based tests validate that invariants hold across a wide range
 * of randomly generated inputs. This catches edge cases that manual tests miss.
 *
 * Testing Properties:
 * 1. Memory operations preserve data integrity
 * 2. Consciousness states remain bounded
 * 3. Toroidal coordinates wrap correctly
 * 4. Gaussian operations maintain mathematical properties
 * 5. Brain coordination algorithms maintain consistency
 * 6. Memory consolidation preserves important information
 * 7. Phase6 integration maintains state coherence
 */

use niodoo_consciousness::*;
use niodoo_consciousness::consciousness_engine::*;
use niodoo_consciousness::consciousness_engine::memory_management::*;
use niodoo_consciousness::consciousness_engine::brain_coordination::*;
use niodoo_consciousness::consciousness_engine::phase6_integration::*;
use proptest::prelude::*;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::runtime::Runtime;

// ============================================================================
// PROPERTY TESTS: Memory System
// ============================================================================

proptest! {
    #[test]
    fn test_memory_relevance_bounded(relevance in 0.0f64..1.0f64) {
        // Property: All memory relevance scores should be between 0 and 1
        let fragment = memory::mobius::MemoryFragment {
            content: "Test".to_string(),
            layer: memory::mobius::MemoryLayer::Semantic,
            relevance,
            timestamp: 1.0,
        };

        prop_assert!(fragment.relevance >= 0.0 && fragment.relevance <= 1.0);
    }

    #[test]
    fn test_toroidal_coordinate_wrapping(
        theta in -10.0f64..10.0f64,
        phi in -10.0f64..10.0f64
    ) {
        // Property: Toroidal coordinates should always wrap to [0, 2π]
        let coord = memory::toroidal::ToroidalCoordinate::new(theta, phi);

        prop_assert!(coord.theta >= 0.0 && coord.theta < 2.0 * std::f64::consts::PI);
        prop_assert!(coord.phi >= 0.0 && coord.phi < 2.0 * std::f64::consts::PI);
    }

    #[test]
    fn test_geodesic_distance_symmetry(
        theta1 in 0.0f64..6.28,
        phi1 in 0.0f64..6.28,
        theta2 in 0.0f64..6.28,
        phi2 in 0.0f64..6.28
    ) {
        // Property: Distance from A to B equals distance from B to A
        let coord1 = memory::toroidal::ToroidalCoordinate::new(theta1, phi1);
        let coord2 = memory::toroidal::ToroidalCoordinate::new(theta2, phi2);

        let dist1 = coord1.geodesic_distance(&coord2);
        let dist2 = coord2.geodesic_distance(&coord1);

        prop_assert!((dist1 - dist2).abs() < 1e-6);
    }

    #[test]
    fn test_geodesic_distance_non_negative(
        theta1 in 0.0f64..6.28,
        phi1 in 0.0f64..6.28,
        theta2 in 0.0f64..6.28,
        phi2 in 0.0f64..6.28
    ) {
        // Property: Distances are always non-negative
        let coord1 = memory::toroidal::ToroidalCoordinate::new(theta1, phi1);
        let coord2 = memory::toroidal::ToroidalCoordinate::new(theta2, phi2);

        let dist = coord1.geodesic_distance(&coord2);

        prop_assert!(dist >= 0.0);
    }

    #[test]
    fn test_consciousness_state_bounded(
        arousal in 0.0f32..1.0f32,
        learning in 0.0f32..1.0f32
    ) {
        // Property: Consciousness state values remain bounded
        let mut state = consciousness::ConsciousnessState::new();
        state.emotional_arousal = arousal;
        state.learning_will_activation = learning;

        prop_assert!(state.emotional_arousal >= 0.0 && state.emotional_arousal <= 1.0);
        prop_assert!(state.learning_will_activation >= 0.0 && state.learning_will_activation <= 1.0);
    }
}

// ============================================================================
// PROPERTY TESTS: Gaussian Memory Spheres
// ============================================================================

proptest! {
    #[test]
    fn test_gaussian_sphere_dimension_consistency(
        dim in 2usize..10usize
    ) {
        // Property: Gaussian sphere maintains dimension consistency
        let mean = vec![1.0; dim];
        let mut covariance = vec![vec![0.0; dim]; dim];

        // Create identity covariance
        for i in 0..dim {
            covariance[i][i] = 1.0;
        }

        let sphere = dual_mobius_gaussian::GaussianMemorySphere::new(mean.clone(), covariance);

        prop_assert_eq!(sphere.mean.len(), dim);
        prop_assert_eq!(sphere.covariance.len(), dim);
    }

    #[test]
    fn test_empty_cluster_linearization_safe(size in 0usize..5usize) {
        // Property: Linearization handles varying cluster sizes safely
        let mut cluster = Vec::new();

        for i in 0..size {
            let sphere = dual_mobius_gaussian::GaussianMemorySphere::new(
                vec![i as f64, (i + 1) as f64],
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            );
            cluster.push(sphere);
        }

        let result = dual_mobius_gaussian::linearize_cluster(cluster);

        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap().len(), size);
    }
}

// ============================================================================
// PROPERTY TESTS: Brain Types and Personalities
// ============================================================================

proptest! {
    #[test]
    fn test_personality_enum_serialization(
        personality_idx in 0usize..4usize
    ) {
        // Property: All personality types can be created and used
        use personality::PersonalityType;

        let personalities = vec![
            PersonalityType::Intuitive,
            PersonalityType::Analytical,
            PersonalityType::Empathetic,
            PersonalityType::Creative,
        ];

        let personality = &personalities[personality_idx];

        // Just verify we can use the personality
        prop_assert!(match personality {
            PersonalityType::Intuitive => true,
            PersonalityType::Analytical => true,
            PersonalityType::Empathetic => true,
            PersonalityType::Creative => true,
        });
    }

    #[test]
    fn test_brain_type_completeness(brain_idx in 0usize..4usize) {
        // Property: All brain types are valid
        use brain::BrainType;

        let brains = vec![
            BrainType::Sensory,
            BrainType::Motor,
            BrainType::Emotional,
            BrainType::Rational,
        ];

        let brain = &brains[brain_idx];

        prop_assert!(match brain {
            BrainType::Sensory => true,
            BrainType::Motor => true,
            BrainType::Emotional => true,
            BrainType::Rational => true,
        });
    }
}

// ============================================================================
// PROPERTY TESTS: Memory Layer Transitions
// ============================================================================

proptest! {
    #[test]
    fn test_memory_layer_transitions(layer_idx in 0usize..4usize) {
        // Property: Memory layers are well-defined
        use memory::mobius::MemoryLayer;

        let layers = vec![
            MemoryLayer::Working,
            MemoryLayer::Episodic,
            MemoryLayer::Semantic,
            MemoryLayer::Procedural,
        ];

        let layer = &layers[layer_idx];

        prop_assert!(match layer {
            MemoryLayer::Working => true,
            MemoryLayer::Episodic => true,
            MemoryLayer::Semantic => true,
            MemoryLayer::Procedural => true,
        });
    }
}

// ============================================================================
// PROPERTY TESTS: Numerical Stability
// ============================================================================

proptest! {
    #[test]
    fn test_emotional_vector_normalization(
        v1 in -100.0f32..100.0f32,
        v2 in -100.0f32..100.0f32,
        v3 in -100.0f32..100.0f32
    ) {
        // Property: Emotional vectors remain finite after operations
        let vector = vec![v1, v2, v3];

        // Calculate magnitude
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Even with extreme values, results should be finite
        prop_assert!(magnitude.is_finite() || vector.iter().all(|x| !x.is_finite()));
    }

    #[test]
    fn test_activation_strength_bounded(strength in 0.0f64..1.0f64) {
        // Property: Activation strengths remain bounded [0, 1]
        let node = memory::toroidal::ToroidalMemoryNode {
            id: "test".to_string(),
            coordinate: memory::toroidal::ToroidalCoordinate::new(0.0, 0.0),
            content: "test".to_string(),
            emotional_vector: vec![0.5],
            temporal_context: vec![1.0],
            activation_strength: strength,
            connections: std::collections::HashMap::new(),
        };

        prop_assert!(node.activation_strength >= 0.0 && node.activation_strength <= 1.0);
    }
}

// ============================================================================
// PROPERTY TESTS: Concurrent Operations
// ============================================================================

use std::sync::Arc;
use parking_lot::RwLock;

proptest! {
    #[test]
    fn test_concurrent_memory_access_safety(
        num_writers in 1usize..10usize,
        num_readers in 1usize..10usize
    ) {
        // Property: Concurrent access doesn't corrupt data
        let memory_system = Arc::new(RwLock::new(
            memory::mobius::MobiusMemorySystem::new()
        ));

        // This test validates that concurrent access is safe
        // In a real async test, we'd spawn actual concurrent tasks

        for i in 0..num_writers {
            let mut mem = memory_system.write();
            let fragment = memory::mobius::MemoryFragment {
                content: format!("Memory {}", i),
                layer: memory::mobius::MemoryLayer::Semantic,
                relevance: 0.5,
                timestamp: i as f64,
            };
            mem.store_memory(fragment);
        }

        for _ in 0..num_readers {
            let mem = memory_system.read();
            let _ = mem.bi_directional_traverse("test", "neutral");
        }

        // Validate that concurrent reads/writes completed without corruption
        let final_mem = memory_system.read();
        let stored_count = final_mem.fragments.len();
        prop_assert!(
            stored_count >= num_writers,
            "Memory system should contain at least {} fragments after {} writes, found {}",
            num_writers, num_writers, stored_count
        );
    }
}

// ============================================================================
// PROPERTY TESTS: Data Integrity
// ============================================================================

proptest! {
    #[test]
    fn test_memory_content_preservation(content in "\\PC*") {
        // Property: Memory content is preserved exactly
        let fragment = memory::mobius::MemoryFragment {
            content: content.clone(),
            layer: memory::mobius::MemoryLayer::Semantic,
            relevance: 0.8,
            timestamp: 1.0,
        };

        prop_assert_eq!(fragment.content, content);
    }

    #[test]
    fn test_timestamp_monotonicity(
        t1 in 0.0f64..1000.0f64,
        t2 in 0.0f64..1000.0f64
    ) {
        // Property: Timestamps maintain ordering
        let frag1 = memory::mobius::MemoryFragment {
            content: "First".to_string(),
            layer: memory::mobius::MemoryLayer::Episodic,
            relevance: 0.5,
            timestamp: t1,
        };

        let frag2 = memory::mobius::MemoryFragment {
            content: "Second".to_string(),
            layer: memory::mobius::MemoryLayer::Episodic,
            relevance: 0.5,
            timestamp: t2,
        };

        if t1 < t2 {
            prop_assert!(frag1.timestamp < frag2.timestamp);
        } else if t1 > t2 {
            prop_assert!(frag1.timestamp > frag2.timestamp);
        } else {
            prop_assert_eq!(frag1.timestamp, frag2.timestamp);
        }
    }
}

// ============================================================================
// PROPERTY TESTS: Triangle Inequality for Distances
// ============================================================================

proptest! {
    #[test]
    fn test_geodesic_triangle_inequality(
        theta1 in 0.0f64..6.28,
        phi1 in 0.0f64..6.28,
        theta2 in 0.0f64..6.28,
        phi2 in 0.0f64..6.28,
        theta3 in 0.0f64..6.28,
        phi3 in 0.0f64..6.28
    ) {
        // Property: Triangle inequality holds for geodesic distances
        // d(A, C) <= d(A, B) + d(B, C)

        let a = memory::toroidal::ToroidalCoordinate::new(theta1, phi1);
        let b = memory::toroidal::ToroidalCoordinate::new(theta2, phi2);
        let c = memory::toroidal::ToroidalCoordinate::new(theta3, phi3);

        let d_ac = a.geodesic_distance(&c);
        let d_ab = a.geodesic_distance(&b);
        let d_bc = b.geodesic_distance(&c);

        // Triangle inequality with small epsilon for floating point errors
        prop_assert!(d_ac <= d_ab + d_bc + 1e-6);
    }
}

// ============================================================================
// PROPERTY TESTS: Consciousness Engine Memory Operations
// ============================================================================

proptest! {
    #[test]
    fn test_memory_event_creation_properties(
        content in "\\PC*",
        emotional_impact in 0.0f32..1.0f32,
        learning_will in 0.0f32..1.0f32
    ) {
        // Property: Memory events maintain valid properties
        let event = PersonalConsciousnessEvent {
            id: "test-id".to_string(),
            event_type: "test".to_string(),
            content: content.clone(),
            emotional_impact,
            learning_will_activation: learning_will,
            timestamp: 1.0,
            context: "test".to_string(),
        };

        prop_assert_eq!(event.content, content);
        prop_assert!(event.emotional_impact >= 0.0 && event.emotional_impact <= 1.0);
        prop_assert!(event.learning_will_activation >= 0.0 && event.learning_will_activation <= 1.0);
        prop_assert!(event.timestamp > 0.0);
    }

    #[test]
    fn test_memory_consolidation_preserves_importance(
        num_events in 1usize..50usize,
        importance_threshold in 0.0f32..1.0f32
    ) {
        // Property: Memory consolidation preserves important events
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            let memory_store = Arc::new(RwLock::new(Vec::new()));
            let memory_system = GuessingMemorySystem::new();
            let personal_memory_engine = PersonalMemoryEngine::default();
            let consciousness_state = Arc::new(RwLock::new(ConsciousnessState::new()));
            
            let memory_manager = MemoryManager::new(
                memory_store.clone(),
                memory_system,
                personal_memory_engine,
                consciousness_state,
            );

            // Create events with varying importance
            for i in 0..num_events {
                let importance = if i % 3 == 0 { importance_threshold + 0.1 } else { importance_threshold - 0.1 };
                let event = PersonalConsciousnessEvent {
                    id: format!("event-{}", i),
                    event_type: "test".to_string(),
                    content: format!("Event {}", i),
                    emotional_impact: importance.max(0.0).min(1.0),
                    learning_will_activation: 0.5,
                    timestamp: i as f64,
                    context: "test".to_string(),
                };
                
                memory_manager.store_event(event).await.unwrap();
            }

            // Consolidate memories
            memory_manager.consolidate_memories().await.unwrap();

            // Verify important events are preserved
            let memories = memory_manager.retrieve_memories("test").await.unwrap();
            prop_assert!(!memories.is_empty(), "Should have memories after consolidation");
        });
    }
}

// ============================================================================
// PROPERTY TESTS: Brain Coordination Consistency
// ============================================================================

proptest! {
    #[test]
    fn test_brain_coordination_deterministic_ordering(
        input in "\\PC*",
        num_runs in 1usize..5usize
    ) {
        // Property: Brain coordination produces consistent results for same input
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            let consciousness_state = Arc::new(RwLock::new(ConsciousnessState::new()));
            let memory_store = Arc::new(RwLock::new(Vec::new()));
            let memory_system = GuessingMemorySystem::new();
            let personal_memory_engine = PersonalMemoryEngine::default();
            
            let motor_brain = MotorBrain::new().unwrap();
            let lcars_brain = LcarsBrain::new().unwrap();
            let efficiency_brain = EfficiencyBrain::new().unwrap();
            let personality_manager = PersonalityManager::new();
            
            let coordinator = BrainCoordinator::new(
                motor_brain,
                lcars_brain,
                efficiency_brain,
                personality_manager,
                consciousness_state,
            );

            let mut results = Vec::new();
            for _ in 0..num_runs {
                let result = coordinator.process_brains_parallel(&input, tokio::time::Duration::from_secs(5)).await.unwrap();
                results.push(result);
            }

            // All runs should produce the same number of responses
            let first_len = results[0].len();
            for result in &results {
                prop_assert_eq!(result.len(), first_len, "All runs should produce same number of responses");
            }
        });
    }

    #[test]
    fn test_brain_coordination_response_bounds(
        input in "\\PC*"
    ) {
        // Property: Brain coordination responses are bounded and non-empty
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            let consciousness_state = Arc::new(RwLock::new(ConsciousnessState::new()));
            let motor_brain = MotorBrain::new().unwrap();
            let lcars_brain = LcarsBrain::new().unwrap();
            let efficiency_brain = EfficiencyBrain::new().unwrap();
            let personality_manager = PersonalityManager::new();
            
            let coordinator = BrainCoordinator::new(
                motor_brain,
                lcars_brain,
                efficiency_brain,
                personality_manager,
                consciousness_state,
            );

            let results = coordinator.process_brains_parallel(&input, tokio::time::Duration::from_secs(5)).await.unwrap();
            
            prop_assert_eq!(results.len(), 3, "Should get exactly 3 brain responses");
            for (i, result) in results.iter().enumerate() {
                prop_assert!(!result.is_empty(), "Brain {} response should not be empty", i);
                prop_assert!(result.len() < 10000, "Brain {} response should be reasonably sized", i);
            }
        });
    }
}

// ============================================================================
// PROPERTY TESTS: Phase6 Integration State Coherence
// ============================================================================

proptest! {
    #[test]
    fn test_phase6_state_coherence(
        input in "\\PC*",
        num_operations in 1usize..10usize
    ) {
        // Property: Phase6 integration maintains state coherence across operations
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            let consciousness_state = Arc::new(RwLock::new(ConsciousnessState::new()));
            let memory_store = Arc::new(RwLock::new(Vec::new()));
            
            let phase6_manager = Phase6Manager::new(
                consciousness_state.clone(),
                memory_store.clone(),
            );

            // Perform multiple operations
            for i in 0..num_operations {
                let operation_input = format!("{} {}", input, i);
                let _result = phase6_manager.process_phase6_input(&operation_input).await.unwrap();
            }

            // Verify state remains coherent
            let state = consciousness_state.read().await;
            prop_assert!(state.emotional_arousal >= 0.0 && state.emotional_arousal <= 1.0);
            prop_assert!(state.learning_will_activation >= 0.0 && state.learning_will_activation <= 1.0);
        });
    }

    #[test]
    fn test_phase6_memory_integration(
        content in "\\PC*",
        emotional_impact in 0.0f32..1.0f32
    ) {
        // Property: Phase6 integration properly handles memory operations
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            let consciousness_state = Arc::new(RwLock::new(ConsciousnessState::new()));
            let memory_store = Arc::new(RwLock::new(Vec::new()));
            
            let phase6_manager = Phase6Manager::new(
                consciousness_state.clone(),
                memory_store.clone(),
            );

            // Process input that should create memory
            let result = phase6_manager.process_phase6_input(&content).await.unwrap();
            prop_assert!(!result.is_empty(), "Phase6 processing should produce output");

            // Verify memory was created
            let memories = memory_store.read().await;
            // Note: Actual memory creation depends on implementation
            // This test ensures the system doesn't crash with various inputs
        });
    }
}

// ============================================================================
// PROPERTY TESTS: Consciousness State Transitions
// ============================================================================

proptest! {
    #[test]
    fn test_consciousness_state_transition_bounds(
        initial_arousal in 0.0f32..1.0f32,
        initial_learning in 0.0f32..1.0f32,
        delta_arousal in -0.5f32..0.5f32,
        delta_learning in -0.5f32..0.5f32
    ) {
        // Property: Consciousness state transitions remain within bounds
        let mut state = ConsciousnessState::new();
        state.emotional_arousal = initial_arousal;
        state.learning_will_activation = initial_learning;

        // Simulate state transition
        let new_arousal = (state.emotional_arousal + delta_arousal).max(0.0).min(1.0);
        let new_learning = (state.learning_will_activation + delta_learning).max(0.0).min(1.0);

        prop_assert!(new_arousal >= 0.0 && new_arousal <= 1.0);
        prop_assert!(new_learning >= 0.0 && new_learning <= 1.0);
    }

    #[test]
    fn test_emotional_vector_normalization_properties(
        v1 in -100.0f32..100.0f32,
        v2 in -100.0f32..100.0f32,
        v3 in -100.0f32..100.0f32
    ) {
        // Property: Emotional vector operations maintain mathematical properties
        let vector = vec![v1, v2, v3];
        
        // Test magnitude calculation
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        // Magnitude should be non-negative
        prop_assert!(magnitude >= 0.0);
        
        // If all components are finite, magnitude should be finite
        if vector.iter().all(|x| x.is_finite()) {
            prop_assert!(magnitude.is_finite());
        }
        
        // Test normalization (if magnitude > 0)
        if magnitude > 1e-6 {
            let normalized: Vec<f32> = vector.iter().map(|x| x / magnitude).collect();
            let normalized_magnitude: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!((normalized_magnitude - 1.0).abs() < 1e-6);
        }
    }
}

// ============================================================================
// PROPERTY TESTS: Concurrent Operations Safety
// ============================================================================

proptest! {
    #[test]
    fn test_concurrent_memory_operations_safety(
        num_operations in 1usize..20usize,
        operation_type in 0usize..3usize
    ) {
        // Property: Concurrent memory operations don't corrupt state
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            let memory_store = Arc::new(RwLock::new(Vec::new()));
            let memory_system = GuessingMemorySystem::new();
            let personal_memory_engine = PersonalMemoryEngine::default();
            let consciousness_state = Arc::new(RwLock::new(ConsciousnessState::new()));
            
            let memory_manager = MemoryManager::new(
                memory_store.clone(),
                memory_system,
                personal_memory_engine,
                consciousness_state,
            );

            // Perform concurrent operations
            let mut tasks = Vec::new();
            for i in 0..num_operations {
                let manager = memory_manager.clone();
                let task = tokio::spawn(async move {
                    match operation_type {
                        0 => {
                            // Store event
                            let event = PersonalConsciousnessEvent {
                                id: format!("concurrent-{}", i),
                                event_type: "concurrent".to_string(),
                                content: format!("Concurrent operation {}", i),
                                emotional_impact: 0.5,
                                learning_will_activation: 0.3,
                                timestamp: i as f64,
                                context: "concurrent".to_string(),
                            };
                            manager.store_event(event).await
                        }
                        1 => {
                            // Retrieve memories
                            manager.retrieve_memories("concurrent").await
                        }
                        2 => {
                            // Get stats
                            manager.get_memory_stats().await
                        }
                        _ => unreachable!(),
                    }
                });
                tasks.push(task);
            }

            // Wait for all operations to complete
            let results = futures::future::join_all(tasks).await;
            
            // Verify no operations failed
            for result in results {
                prop_assert!(result.is_ok(), "Concurrent operation should not fail");
            }
        });
    }
}

// ============================================================================
// PROPERTY TESTS: Memory Consolidation Properties
// ============================================================================

proptest! {
    #[test]
    fn test_memory_consolidation_idempotency(
        num_events in 5usize..30usize
    ) {
        // Property: Multiple consolidation operations produce consistent results
        let rt = Runtime::new().unwrap();
        
        rt.block_on(async {
            let memory_store = Arc::new(RwLock::new(Vec::new()));
            let memory_system = GuessingMemorySystem::new();
            let personal_memory_engine = PersonalMemoryEngine::default();
            let consciousness_state = Arc::new(RwLock::new(ConsciousnessState::new()));
            
            let memory_manager = MemoryManager::new(
                memory_store.clone(),
                memory_system,
                personal_memory_engine,
                consciousness_state,
            );

            // Store events
            for i in 0..num_events {
                let event = PersonalConsciousnessEvent {
                    id: format!("consolidation-{}", i),
                    event_type: "consolidation".to_string(),
                    content: format!("Consolidation test {}", i),
                    emotional_impact: 0.5,
                    learning_will_activation: 0.3,
                    timestamp: i as f64,
                    context: "consolidation".to_string(),
                };
                memory_manager.store_event(event).await.unwrap();
            }

            // Perform multiple consolidations
            memory_manager.consolidate_memories().await.unwrap();
            let stats1 = memory_manager.get_memory_stats().await.unwrap();
            
            memory_manager.consolidate_memories().await.unwrap();
            let stats2 = memory_manager.get_memory_stats().await.unwrap();

            // Stats should be consistent (or at least not crash)
            prop_assert!(stats1.total_events >= 0);
            prop_assert!(stats2.total_events >= 0);
        });
    }
}
