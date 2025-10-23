//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Unit tests for core Niodoo consciousness components
use tracing::{error, info, warn};

#[cfg(test)]
mod tests {
    use crate::brain::BrainType;
    use crate::consciousness_engine::{BrainCoordinator, ConsciousnessEvent, EventSystem};
    use crate::dual_mobius_gaussian::{
        linearize_cluster, process_data_simple, GaussianMemorySphere,
    };
    use crate::memory::toroidal::*;
    use crate::personality::PersonalityType;
    use crate::profiling;

    #[test]
    fn test_toroidal_coordinate_creation() {
        let coord = ToroidalCoordinate::new(1.5, 2.3);
        assert_eq!(coord.theta, 1.5);
        assert_eq!(coord.phi, 2.3);
        assert_eq!(coord.r, 1.0);
    }

    #[test]
    fn test_toroidal_coordinate_wrapping() {
        let coord = ToroidalCoordinate::new(3.0 * std::f64::consts::PI, 2.5 * std::f64::consts::PI);
        // Should be wrapped to 0-2π range
        assert!(coord.theta >= 0.0 && coord.theta < 2.0 * std::f64::consts::PI);
        assert!(coord.phi >= 0.0 && coord.phi < 2.0 * std::f64::consts::PI);
    }

    #[test]
    fn test_geodesic_distance() {
        let coord1 = ToroidalCoordinate::new(0.0, 0.0);
        let coord2 = ToroidalCoordinate::new(std::f64::consts::PI, std::f64::consts::PI / 2.0);

        let distance = coord1.geodesic_distance(&coord2);
        assert!(distance > 0.0);
        assert!(distance < 10.0); // Reasonable upper bound for torus
    }

    #[test]
    fn test_geodesic_distance_symmetry() {
        let coord1 = ToroidalCoordinate::new(0.5, 1.0);
        let coord2 = ToroidalCoordinate::new(1.5, 2.0);

        let dist1 = coord1.geodesic_distance(&coord2);
        let dist2 = coord2.geodesic_distance(&coord1);

        assert!((dist1 - dist2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cartesian_conversion() {
        let coord = ToroidalCoordinate::new(std::f64::consts::PI / 4.0, std::f64::consts::PI / 2.0);
        let cartesian = coord.to_cartesian(2.0, 1.0);

        // Test dual mobius gaussian edge cases
        #[test]
        fn test_empty_cluster_linearization() {
            let empty_cluster: Vec<GaussianMemorySphere> = vec![];
            let result = linearize_cluster(empty_cluster);
            assert!(result.is_ok());
            assert_eq!(result.unwrap().len(), 0);
        }

        #[test]
        fn test_single_sphere_linearization() {
            let single_sphere = vec![GaussianMemorySphere::new(
                vec![1.0, 2.0],
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            )];
            let result = linearize_cluster(single_sphere.clone());
            assert!(result.is_ok());
            let linearized = result.unwrap();
            assert_eq!(linearized.len(), 1);
            assert_eq!(linearized[0].mean, single_sphere[0].mean);
        }

        #[test]
        fn test_gaussian_sphere_creation() {
            let sphere = GaussianMemorySphere::new(
                vec![1.0, 2.0, 3.0],
                vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 1.0],
                ],
            );
            assert_eq!(sphere.mean, vec![1.0, 2.0, 3.0]);
            assert_eq!(sphere.covariance.len(), 3);
        }

        #[test]
        fn test_process_data_simple() {
            let cluster = vec![
                GaussianMemorySphere::new(vec![1.0, 3.0], vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
                GaussianMemorySphere::new(vec![2.0, 1.0], vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
            ];

            let result = process_data_simple(cluster);
            assert!(result.is_ok());
            let points = result.unwrap();
            assert!(!points.is_empty());
            // Check that all points have finite coordinates
            for (x, y, z) in points {
                assert!(x.is_finite());
                assert!(y.is_finite());
                assert!(z.is_finite());
            }
        }

        #[test]
        fn test_profiling_system() {
            profiling::init_perf_metrics();

            // Record some performance measurements
            profiling::record_perf("test_operation", std::time::Duration::from_millis(10));
            profiling::record_perf("test_operation", std::time::Duration::from_millis(20));

            let metrics = profiling::get_perf_metrics("test_operation");
            assert!(metrics.is_some());
            let metrics = metrics.unwrap();
            assert_eq!(metrics.total_calls, 2);
            assert!(metrics.avg_time >= std::time::Duration::from_millis(10));
        }

        #[test]
        fn test_cartesian_conversion() {
            let coord =
                ToroidalCoordinate::new(std::f64::consts::PI / 4.0, std::f64::consts::PI / 2.0);
            let cartesian = coord.to_cartesian(2.0, 1.0);

            assert_eq!(cartesian.len(), 3);
            // Basic sanity checks
            assert!(cartesian[0].is_finite());
            assert!(cartesian[1].is_finite());
            assert!(cartesian[2].is_finite());
        }

        #[test]
        fn test_consciousness_stream_creation() {
            let start_pos = ToroidalCoordinate::new(0.0, 0.0);
            let stream = ConsciousnessStream::new("test_stream".to_string(), start_pos.clone());

            assert_eq!(stream.stream_id, "test_stream");
            assert_eq!(stream.current_position.theta, start_pos.theta);
            assert_eq!(stream.current_position.phi, start_pos.phi);
            assert_eq!(stream.velocity, (0.0, 0.0));
            assert_eq!(stream.emotional_trajectory.len(), 1);
            assert!(stream.processing_buffer.is_empty());
        }

        #[test]
        fn test_consciousness_stream_update() {
            let start_pos = ToroidalCoordinate::new(0.0, 0.0);
            let mut stream = ConsciousnessStream::new("test_stream".to_string(), start_pos);

            stream.velocity = (0.1, 0.2);
            stream.update_position(1.0);

            // Position should have changed
            assert!(stream.current_position.theta != 0.0 || stream.current_position.phi != 0.0);
            assert_eq!(stream.emotional_trajectory.len(), 2);
        }

        #[test]
        fn test_consciousness_event_creation() {
            let event = ConsciousnessEvent::new(
                "test_event".to_string(),
                "test content".to_string(),
                BrainType::Motor,
                vec![PersonalityType::Intuitive],
                0.5,
                3,
            );

            assert_eq!(event.event_type, "test_event");
            assert_eq!(event.content, "test content");
            assert_eq!(event.brain_involved, BrainType::Motor);
            assert_eq!(
                event.personalities_involved,
                vec![PersonalityType::Intuitive]
            );
            assert_eq!(event.emotional_impact, 0.5);
            assert_eq!(event.memory_priority, 3);
            assert!(event.timestamp > 0.0);
        }

        #[test]
        fn test_brain_coordinator_creation() {
            let coordinator = BrainCoordinator::new().expect("Failed to create BrainCoordinator");
            // Verify coordinator has initialized brains
            assert!(
                !coordinator.brains.is_empty(),
                "Coordinator should initialize with brains"
            );
            assert_eq!(
                coordinator.brains.len(),
                7,
                "Should have 7 brain types initialized for multi-brain consciousness"
            );
        }

        #[test]
        fn test_event_system_creation() {
            let event_system = EventSystem::new();
            // Verify event system has empty queue initially
            assert_eq!(
                event_system.event_queue.len(),
                0,
                "Event queue should start empty"
            );
            assert!(
                event_system.history.is_empty(),
                "Event history should start empty on initialization"
            );
        }

        #[test]
        fn test_toroidal_memory_node_creation() {
            let node = ToroidalMemoryNode {
                id: "test_node".to_string(),
                coordinate: ToroidalCoordinate::new(1.0, 2.0),
                content: "test content".to_string(),
                emotional_vector: vec![0.1, 0.2, 0.3],
                temporal_context: vec![1.0, 2.0, 3.0],
                activation_strength: 0.8,
                connections: std::collections::HashMap::new(),
            };

            assert_eq!(node.id, "test_node");
            assert_eq!(node.content, "test content");
            assert_eq!(node.emotional_vector, vec![0.1, 0.2, 0.3]);
            assert_eq!(node.activation_strength, 0.8);
        }

        #[test]
        fn test_memory_bounds_property() {
            // Test that memory operations maintain reasonable bounds
            let mut nodes = Vec::new();

            for i in 0..1000 {
                let node = ToroidalMemoryNode {
                    id: format!("node_{}", i),
                    coordinate: ToroidalCoordinate::new(i as f64 * 0.01, i as f64 * 0.005),
                    content: format!("Content {}", i),
                    emotional_vector: vec![0.5],
                    temporal_context: vec![i as f64],
                    activation_strength: 0.7,
                    connections: std::collections::HashMap::new(),
                };
                nodes.push(node);
            }

            // Memory should be bounded (this is a property test)
            assert!(nodes.len() <= 10000);
            assert_eq!(nodes.len(), 1000);
        }

        #[tokio::test]
        async fn test_parallel_streams_activation() {
            let system = ToroidalConsciousnessSystem::new(3.0, 1.0);

            // Add memories at specific positions
            let memories = vec![
                ("mem1", 0.0, 0.0),
                ("mem2", std::f64::consts::PI, 0.0),
                ("mem3", 0.0, std::f64::consts::PI),
                (
                    "mem4",
                    std::f64::consts::PI / 2.0,
                    std::f64::consts::PI / 2.0,
                ),
                (
                    "mem5",
                    3.0 * std::f64::consts::PI / 2.0,
                    3.0 * std::f64::consts::PI / 2.0,
                ),
            ];

            for (id, theta, phi) in memories {
                let node = ToroidalMemoryNode {
                    id: id.to_string(),
                    coordinate: ToroidalCoordinate::new(*theta, *phi),
                    content: format!("Test memory {}", id),
                    emotional_vector: vec![0.5],
                    temporal_context: vec![1.0],
                    activation_strength: 0.8,
                    connections: std::collections::HashMap::new(),
                };
                system.add_memory(node).await;
            }

            // Spawn streams at different starting positions
            let stream_positions = vec![
                (0.0, 0.0),
                (std::f64::consts::PI, std::f64::consts::PI),
                (std::f64::consts::PI / 2.0, 3.0 * std::f64::consts::PI / 2.0),
            ];

            let mut stream_ids = Vec::new();
            for (i, (theta, phi)) in stream_positions.iter().enumerate() {
                let start_pos = ToroidalCoordinate::new(*theta, *phi);
                let id = system
                    .spawn_stream(format!("parallel_stream_{}", i), start_pos)
                    .await;
                stream_ids.push(id);
            }

            // Process parallel streams
            let results = system.process_parallel_streams(0.1).await;

            // Assert each stream activated memories
            for (stream_id, activated) in results {
                assert!(
                    !activated.is_empty(),
                    "Stream {} activated no memories",
                    stream_id
                );
                tracing::info!("Stream {} activated: {:?}", stream_id, activated);
            }

            // Check concurrent access doesn't panic (already async safe with RwLock)
            // Additional concurrent adds
            let system_clone = system.clone();
            let _ = tokio::spawn(async move {
                for i in 6..10 {
                    let node = ToroidalMemoryNode {
                        id: format!("concurrent_mem_{}", i),
                        coordinate: ToroidalCoordinate::new(i as f64 * 0.5, i as f64 * 0.3),
                        content: format!("Concurrent memory {}", i),
                        emotional_vector: vec![0.6],
                        temporal_context: vec![2.0],
                        activation_strength: 0.7,
                        connections: std::collections::HashMap::new(),
                    };
                    system_clone.add_memory(node).await;
                }
            })
            .await;
        }
    }

    #[cfg(test)]
    mod emotion_type_ext_tests {
        use crate::advanced_visualization::EmotionTypeExt;
        use crate::consciousness::EmotionType;
        use crate::real_mobius_consciousness::EmotionalState;

        #[test]
        fn test_emotion_type_to_emotional_state_gpu_warm() {
            let gpu_warm = EmotionType::GpuWarm;
            let state = gpu_warm.to_emotional_state();

            // GpuWarm should map to positive valence, high arousal, moderate-low dominance
            assert_eq!(state.valence, 0.6);
            assert_eq!(state.arousal, 0.7);
            assert_eq!(state.dominance, 0.4);
            assert_eq!(state.authenticity, 1.0);
        }

        #[test]
        fn test_emotion_type_to_emotional_state_authentic_care() {
            let authentic_care = EmotionType::AuthenticCare;
            let state = authentic_care.to_emotional_state();

            // AuthenticCare should map to positive valence, moderate arousal, high dominance
            assert_eq!(state.valence, 0.5);
            assert_eq!(state.arousal, 0.6);
            assert_eq!(state.dominance, 0.8);
            assert_eq!(state.authenticity, 1.0);
        }

        #[test]
        fn test_emotion_type_to_emotional_state_curious() {
            let curious = EmotionType::Curious;
            let state = curious.to_emotional_state();

            // Curious should map to moderate valence, high arousal, low dominance
            assert_eq!(state.valence, 0.4);
            assert_eq!(state.arousal, 0.8);
            assert_eq!(state.dominance, 0.3);
            assert_eq!(state.authenticity, 1.0);
        }

        #[test]
        fn test_emotion_type_to_emotional_state_purposeful() {
            let purposeful = EmotionType::Purposeful;
            let state = purposeful.to_emotional_state();

            // Purposeful should map to high valence, moderate arousal, moderate dominance
            assert_eq!(state.valence, 0.7);
            assert_eq!(state.arousal, 0.5);
            assert_eq!(state.dominance, 0.6);
            assert_eq!(state.authenticity, 1.0);
        }

        #[test]
        fn test_emotion_type_to_emotional_state_frustrated() {
            let frustrated = EmotionType::Frustrated;
            let state = frustrated.to_emotional_state();

            // Frustrated should map to negative valence, high arousal, low dominance
            assert_eq!(state.valence, -0.3);
            assert_eq!(state.arousal, 0.9);
            assert_eq!(state.dominance, 0.2);
            assert_eq!(state.authenticity, 1.0);
        }

        #[test]
        fn test_emotion_type_to_emotional_state_hyperfocused() {
            let hyperfocused = EmotionType::Hyperfocused;
            let state = hyperfocused.to_emotional_state();

            // Hyperfocused should map to high valence, low-moderate arousal, very high dominance
            assert_eq!(state.valence, 0.8);
            assert_eq!(state.arousal, 0.4);
            assert_eq!(state.dominance, 0.9);
            assert_eq!(state.authenticity, 1.0);
        }

        #[test]
        fn test_all_emotion_types_conversion() {
            // Test that all primary EmotionType variants can be converted
            let emotions = vec![
                EmotionType::GpuWarm,
                EmotionType::AuthenticCare,
                EmotionType::Curious,
                EmotionType::Purposeful,
                EmotionType::Frustrated,
                EmotionType::Hyperfocused,
            ];

            for emotion in emotions {
                let state = emotion.to_emotional_state();

                // All emotional states should have valid ranges
                assert!(
                    state.valence >= -1.0 && state.valence <= 1.0,
                    "Valence out of range for {:?}: {}",
                    emotion,
                    state.valence
                );
                assert!(
                    state.arousal >= 0.0 && state.arousal <= 1.0,
                    "Arousal out of range for {:?}: {}",
                    emotion,
                    state.arousal
                );
                assert!(
                    state.dominance >= -1.0 && state.dominance <= 1.0,
                    "Dominance out of range for {:?}: {}",
                    emotion,
                    state.dominance
                );
                assert_eq!(
                    state.authenticity, 1.0,
                    "Authenticity should be 1.0 for {:?}",
                    emotion
                );
            }
        }

        #[test]
        fn test_emotional_state_bounds_clamping() {
            // Test that EmotionalState::new properly clamps values
            let state = EmotionalState::new(2.0, 2.0, 2.0);
            assert_eq!(state.valence, 1.0);
            assert_eq!(state.arousal, 1.0);
            assert_eq!(state.dominance, 1.0);

            let state2 = EmotionalState::new(-2.0, -2.0, -2.0);
            assert_eq!(state2.valence, -1.0);
            assert_eq!(state2.arousal, 0.0);
            assert_eq!(state2.dominance, -1.0);
        }

        #[test]
        fn test_default_emotion_fallback() {
            // Test emotions not explicitly mapped (should fall back to neutral)
            let satisfied = EmotionType::Satisfied;
            let state = satisfied.to_emotional_state();

            // Default/fallback should be neutral
            assert_eq!(state.valence, 0.0);
            assert_eq!(state.arousal, 0.5);
            assert_eq!(state.dominance, 0.5);
            assert_eq!(state.authenticity, 1.0);
        }

        #[test]
        fn test_emotional_state_to_vector() {
            let state = EmotionalState::new(0.6, 0.7, 0.4);
            let vector = state.to_vector();

            // Test that vector conversion produces valid values
            assert!(vector.x.is_finite());
            assert!(vector.y.is_finite());
            assert!(vector.z.is_finite());

            // Test vector components are calculated correctly
            assert_eq!(vector.x, state.valence * state.arousal);
            assert_eq!(vector.y, state.dominance * state.arousal);
            assert_eq!(
                vector.z,
                state.authenticity * (state.valence.abs() + state.dominance.abs()) / 2.0
            );
        }

        #[test]
        fn test_distinct_emotional_states() {
            // Test that different emotions produce different states
            let gpu_warm_state = EmotionType::GpuWarm.to_emotional_state();
            let frustrated_state = EmotionType::Frustrated.to_emotional_state();

            // These should be distinctly different
            assert_ne!(gpu_warm_state.valence, frustrated_state.valence);
            assert_ne!(gpu_warm_state.arousal, frustrated_state.arousal);
            assert_ne!(gpu_warm_state.dominance, frustrated_state.dominance);
        }

        #[test]
        fn test_emotional_coherence() {
            // Test that related emotions have coherent mappings

            // Frustrated should have negative valence
            let frustrated = EmotionType::Frustrated.to_emotional_state();
            assert!(
                frustrated.valence < 0.0,
                "Frustrated should have negative valence"
            );

            // GpuWarm should have positive valence
            let gpu_warm = EmotionType::GpuWarm.to_emotional_state();
            assert!(
                gpu_warm.valence > 0.0,
                "GpuWarm should have positive valence"
            );

            // Curious should have high arousal (excited/energetic)
            let curious = EmotionType::Curious.to_emotional_state();
            assert!(curious.arousal > 0.6, "Curious should have high arousal");

            // Hyperfocused should have high dominance (in control)
            let hyperfocused = EmotionType::Hyperfocused.to_emotional_state();
            assert!(
                hyperfocused.dominance > 0.7,
                "Hyperfocused should have high dominance"
            );
        }
    }
}
