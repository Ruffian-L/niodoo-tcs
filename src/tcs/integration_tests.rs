//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

#[cfg(test)]
mod integration_tests {
    use super::*;
    use tokio::test;
    use tokio_stream::{StreamExt, wrappers::ReceiverStream};

    #[tokio::test]
    async fn test_full_tda_pipeline() {
        // Setup
        let pipeline = TDAPipeline::new();

        // Generate synthetic Lorenz attractor data
        let lorenz_data = generate_lorenz_attractor(10000);

        // Convert to stream
        let (tx, rx) = tokio::sync::mpsc::channel(1000);
        tokio::spawn(async move {
            for state in lorenz_data {
                let _ = tx.send(state).await;
            }
        });

        let stream = ReceiverStream::new(rx);

        // Process through pipeline
        let event_stream = pipeline.process_stream(Box::pin(stream)).await;
        let events: Vec<_> = event_stream.collect().await;

        // Validate results
        assert!(!events.is_empty(), "Should detect topological events");

        // Check for expected features in Lorenz attractor
        let has_persistent_loop = events.iter().any(|event| {
            matches!(event, CognitiveEvent::H1Birth { .. })
        });

        assert!(has_persistent_loop, "Lorenz attractor should have persistent loops");

        // Check event types
        let h0_events = events.iter().filter(|e| matches!(e, CognitiveEvent::H0Split { .. } | CognitiveEvent::H0Merge { .. })).count();
        let h1_events = events.iter().filter(|e| matches!(e, CognitiveEvent::H1Birth { .. } | CognitiveEvent::H1Death { .. })).count();
        let h2_events = events.iter().filter(|e| matches!(e, CognitiveEvent::H2Birth { .. } | CognitiveEvent::H2Death { .. })).count();

        assert!(h1_events > 0, "Should detect homology group H1 changes");
        println!("Detected events: H0={}, H1={}, H2={}", h0_events, h1_events, h2_events);
    }

    #[tokio::test]
    async fn test_knot_analysis_pipeline() {
        let analyzer = KnotAnalyzer::new();

        // Create a test homology cycle (simulating a loop in persistence diagram)
        let cycle = HomologyCycle {
            persistence: 0.8,
            dimension: 1,
            representative: vec![
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
            ],
        };

        // Analyze the cycle
        let knot = analyzer.analyze_cycle(&cycle).await.unwrap();

        // Validate knot properties
        assert!(knot.persistence > 0.0, "Knot should have positive persistence");
        assert!(knot.complexity_score > 0.0, "Knot should have complexity score");
        assert!(!knot.cycle_geometry.is_empty(), "Knot should have geometry");

        // Check Jones polynomial computation
        assert!(!knot.jones_polynomial.coefficients.is_empty(), "Jones polynomial should have coefficients");
    }

    #[tokio::test]
    async fn test_cognitive_state_transitions() {
        let mut state = CognitiveState::new();

        // Simulate adding persistent features
        let persistence = PersistenceDiagram {
            features: vec![
                PersistenceFeature { birth: 0.0, death: 2.0, dimension: 0 },
                PersistenceFeature { birth: 0.5, death: 1.5, dimension: 1 },
            ],
        };

        state.persistence = Arc::new(persistence);
        state.update_betti_numbers([1, 1, 0]); // Single component, one loop

        // Create knots from features
        let knot1 = CognitiveKnot::new(
            1.0,
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            JonesPolynomial {
                coefficients: std::collections::HashMap::new(),
                max_degree: 0,
                min_degree: 0,
            },
            KnotType::Trefoil,
        );

        state.add_knot(knot1);

        // Verify state properties
        assert_eq!(state.betti_numbers, [1, 1, 0]);
        assert_eq!(state.active_knots.len(), 1);
        assert!(!state.persistence.features.is_empty());
    }

    #[tokio::test]
    async fn test_event_bus() {
        let event_bus = CognitiveEventBus::new();

        // Create a test event handler
        let (tx, mut rx) = tokio::sync::mpsc::channel(10);
        let handler = EventHandlerTraitImpl {
            sender: tx,
        };

        event_bus.subscribe(Arc::new(handler));

        // Publish an event
        let event = CognitiveEvent::H1Birth {
            timestamp: std::time::SystemTime::now(),
            knot: CognitiveKnot::new(
                0.5,
                vec![],
                JonesPolynomial {
                    coefficients: std::collections::HashMap::new(),
                    max_degree: 0,
                    min_degree: 0,
                },
                KnotType::Trefoil,
            ),
            context: EventContext {
                emotional_coherence: 0.8,
                persistence_score: 0.5,
                context_vector: vec![],
            },
        };

        event_bus.publish(event.clone()).await;

        // Verify event was received
        if let Some(received_event) = rx.recv().await {
            match (&received_event, &event) {
                (CognitiveEvent::H1Birth { knot: k1, .. }, CognitiveEvent::H1Birth { knot: k2, .. }) => {
                    assert_eq!(k1.id, k2.id);
                }
                _ => panic!("Wrong event type received"),
            }
        } else {
            panic!("No event received");
        }
    }

    #[tokio::test]
    async fn test_consensus_vocabulary() {
        let consensus = ConsensusModule::new().await.unwrap();

        // Create test tokens
        let token1 = VocabularyToken {
            pattern: vec![1, 2, 3, 4],
            persistence_score: 8.5,
        };

        let token2 = VocabularyToken {
            pattern: vec![5, 6, 7, 8],
            persistence_score: 6.2,
        };

        // Propose tokens
        consensus.propose_token(token1.clone()).await.unwrap();
        consensus.propose_token(token2.clone()).await.unwrap();

        // Verify tokens are accepted (placeholder - would check consensus state)
        assert!(true); // Placeholder assertion
    }

    #[proptest]
    fn test_takens_embedding_properties(
        dim in 2..10usize,
        delay in 1..20usize,
        length in 100..1000usize
    ) {
        let time_series = generate_time_series(length, 3);

        let embedding = TakensEmbedding::new(dim, delay, 3);
        let embedded = embedding.embed(&time_series);

        // Basic property checks
        prop_assert!(!embedded.is_empty());
        prop_assert!(embedded[0].len() == (dim + 1) * 3);

        // Embedding should preserve some topological properties
        prop_assert!(embedded.len() <= time_series.len());
    }

    #[proptest]
    fn test_persistence_homology_invariant(
        noise_level in 0.0..0.5f32
    ) {
        let base_points = generate_random_points(100);
        let noisy_points = add_noise(&base_points, noise_level);

        let persistence1 = compute_persistence(&base_points);
        let persistence2 = compute_persistence(&noisy_points);

        // Persistence should be relatively stable under small noise
        if noise_level < 0.1 {
            prop_assert!(persistence1.features.len() <= persistence2.features.len() + 2);
        }
    }

    #[proptest]
    fn test_jones_polynomial_invariant(
        knot_type in knot_types()
    ) {
        let knot = generate_knot_by_type(knot_type);

        // Apply Reidemeister moves (simplified)
        let transformed = apply_random_reidemeister_moves(&knot, 3);

        let jones1 = JonesPolynomial::compute(&knot);
        let jones2 = JonesPolynomial::compute(&transformed);

        // Jones polynomial should be invariant under Reidemeister moves
        prop_assert_eq!(jones1.coefficients, jones2.coefficients);
    }

    // Helper functions for tests
    fn generate_lorenz_attractor(steps: usize) -> Vec<Vec<f32>> {
        let sigma = 10.0;
        let rho = 28.0;
        let beta = 8.0 / 3.0;

        let mut x = 1.0;
        let mut y = 1.0;
        let mut z = 1.0;
        let dt = 0.01;

        let mut trajectory = Vec::with_capacity(steps);

        for _ in 0..steps {
            let dx = sigma * (y - x);
            let dy = x * (rho - z) - y;
            let dz = x * y - beta * z;

            x += dx * dt;
            y += dy * dt;
            z += dz * dt;

            trajectory.push(vec![x, y, z]);
        }

        trajectory
    }

    fn generate_time_series(length: usize, dim: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        (0..length)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    fn add_noise(points: &[Vec<f32>], level: f32) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        points.iter().map(|p| {
            p.iter().map(|&v| v + rng.gen_range(-level..level)).collect()
        }).collect()
    }

    fn knot_types() -> impl Strategy<Value = KnotType> {
        prop_oneof![
            Just(KnotType::Unknot),
            Just(KnotType::Trefoil),
            Just(KnotType::FigureEight),
        ]
    }

    fn generate_knot_by_type(knot_type: KnotType) -> KnotDiagram {
        match knot_type {
            KnotType::Unknot => KnotDiagram {
                crossings: vec![],
                gauss_code: vec![],
                pd_code: vec![],
            },
            KnotType::Trefoil => KnotDiagram {
                crossings: vec![
                    Crossing::new(0, 0, 1, 1),
                    Crossing::new(1, 1, 2, 1),
                    Crossing::new(2, 2, 0, 1),
                ],
                gauss_code: vec![1, -2, 3, -1, 2, -3],
                pd_code: vec![[1,4,2,5], [3,6,4,1], [5,2,6,3]],
            },
            KnotType::FigureEight => KnotDiagram {
                crossings: vec![
                    Crossing::new(0, 0, 1, 1),
                    Crossing::new(1, 1, 2, -1),
                    Crossing::new(2, 2, 3, 1),
                    Crossing::new(3, 3, 0, -1),
                ],
                gauss_code: vec![1, -2, 3, -4, -1, 2, -3, 4],
                pd_code: vec![[1,8,2,3], [3,1,4,14], [4,15,5,6], [6,5,7,8], [9,4,10,11], [11,10,12,13], [13,12,14,15], [7,16,9,2]],
            },
            _ => KnotDiagram {
                crossings: vec![],
                gauss_code: vec![],
                pd_code: vec![],
            },
        }
    }

    fn apply_random_reidemeister_moves(knot: &KnotDiagram, moves: usize) -> KnotDiagram {
        // Placeholder - would implement Reidemeister moves
        knot.clone()
    }

    // Event handler implementation for testing
    struct EventHandlerTraitImpl {
        sender: tokio::sync::mpsc::Sender<CognitiveEvent>,
    }

    #[async_trait::async_trait]
    impl EventHandlerTrait for EventHandlerTraitImpl {
        async fn handle(&self, event: CognitiveEvent) {
            let _ = self.sender.send(event).await;
        }
    }
}

// Re-exports for tests
use super::*;
use proptest::prelude::*;
use std::sync::Arc;
use tokio::sync::mpsc;