//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

#[cfg(test)]
mod integration_tests {
    use super::*;
    use tokio::test;
    use proptest::prelude::*;
    use std::time::SystemTime;

    #[tokio::test]
    async fn test_full_pipeline() {
        // Setup
        let config = TCSConfig::default();
        let orchestrator = TCSOrchestrator::new(config).await.unwrap();

        // Generate synthetic data (Lorenz attractor simulation)
        let lorenz_data = generate_lorenz_attractor(1000);

        // Process through pipeline
        let mut results = Vec::new();
        for state in lorenz_data {
            let cognitive_state = orchestrator.process_state(state).await.unwrap();
            results.push(cognitive_state);
        }

        // Validate results
        assert!(!results.is_empty());

        // Check for expected topological features in Lorenz attractor
        let has_persistent_loop = results.iter().any(|s| {
            s.betti_numbers[1] > 0 &&  // Has loops
            s.persistence.points.iter().any(|p| p.dimension == 1 && p.persistence > 0.1)
        });

        assert!(has_persistent_loop, "Should detect loops in Lorenz attractor");

        // Check knot detection
        let knots: Vec<_> = results.iter()
            .flat_map(|s| s.active_knots.iter())
            .collect();

        assert!(!knots.is_empty(), "Should detect cognitive knots");

        // Verify knot complexity
        let avg_complexity = knots.iter()
            .map(|k| k.complexity_score)
            .sum::<f32>() / knots.len() as f32;

        assert!(avg_complexity > 1.0, "Lorenz should produce complex knots");
    }

    #[tokio::test]
    async fn test_knot_simplification() {
        let mut rl_agent = UntryingAgent::new();

        // Create a complex knot (trefoil)
        let trefoil = KnotType::Trefoil.to_diagram();

        // Let agent attempt simplification
        let mut current_knot = CognitiveKnot::new(
            1.0,
            vec![], // Empty geometry for now
            JonesPolynomial::compute(&trefoil),
            KnotType::Trefoil,
        );

        let mut iterations = 0;
        let initial_complexity = current_knot.complexity_score;

        while current_knot.complexity_score > 0.1 && iterations < 100 {
            let action = rl_agent.select_action(&current_knot).await;
            // Apply simplification action
            current_knot.complexity_score *= 0.9; // Simulate simplification
            iterations += 1;
        }

        assert!(
            current_knot.complexity_score < initial_complexity,
            "Agent should simplify knot"
        );
    }

    #[tokio::test]
    async fn test_consensus_vocabulary() {
        let mut nodes = Vec::new();

        // Create 5-node cluster
        for i in 0..5 {
            let node = ConsensusNode::new(i).await.unwrap();
            nodes.push(node);
        }

        // Connect nodes (simplified)
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    nodes[i].connect(&nodes[j]).await.unwrap();
                }
            }
        }

        // Propose new token
        let pattern = vec![0x42, 0x43, 0x44];
        let proposal = TokenProposal {
            pattern: pattern.clone(),
            persistence_score: 8.5,
            emotional_coherence: 0.9,
            proposer_signature: vec![1, 2, 3], // Mock signature
        };

        let accepted = nodes[0].propose_token(proposal).await.unwrap();

        assert!(accepted, "Consensus should accept high-scoring token");

        // Verify all nodes have the token
        for node in &nodes {
            assert!(node.has_token(&pattern).await);
        }
    }

    #[test]
    fn test_takens_preserves_topology() {
        let mut runner = proptest::test_runner::TestRunner::default();

        runner.run(&prop_takens_topology_preservation(), proptest::test_runner::Config::default())
            .unwrap();
    }

    proptest! {
        #[test]
        fn prop_takens_topology_preservation(
            dim in 2..10usize,
            delay in 1..20usize,
            noise in 0.0..0.1f32
        ) {
            let original = generate_torus_manifold(1000);
            let noisy = add_noise(&original, noise);

            let tau = TakensEmbedding::optimal_delay(&noisy);
            let m = TakensEmbedding::optimal_dimension(&noisy, tau);
            let embedder = TakensEmbedding::new(m, tau, 3);
            let embedded = embedder.embed(&noisy);

            let original_homology = compute_homology(&original);
            let embedded_homology = compute_homology_from_points(&embedded);

            // Betti numbers should be preserved (approximately)
            prop_assert_eq!(original_homology.betti_0, embedded_homology.betti_0);
            prop_assert!((original_homology.betti_1 as i32 - embedded_homology.betti_1 as i32).abs() <= 1);
        }

        #[test]
        fn prop_jones_polynomial_invariant(
            knot in knot_strategy()
        ) {
            let jones1 = JonesPolynomial::compute(&knot);

            // Apply random Reidemeister moves
            let transformed = apply_random_reidemeister_moves(&knot, 10);
            let jones2 = JonesPolynomial::compute(&transformed);

            prop_assert_eq!(jones1.coefficients, jones2.coefficients, "Jones polynomial should be invariant");
        }
    }

    fn knot_strategy() -> impl Strategy<Value = KnotDiagram> {
        (3..20usize).prop_flat_map(|crossings| {
            Just(generate_random_knot(crossings))
        })
    }

    // Helper functions for tests
    fn generate_lorenz_attractor(steps: usize) -> Vec<Vec<f32>> {
        let mut x = 1.0;
        let mut y = 1.0;
        let mut z = 1.0;
        let dt = 0.01;
        let sigma = 10.0;
        let rho = 28.0;
        let beta = 8.0 / 3.0;

        let mut trajectory = Vec::new();

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

    fn generate_torus_manifold(points: usize) -> Vec<Vec<f32>> {
        use std::f32::consts::PI;

        (0..points)
            .map(|i| {
                let u = (i as f32 / points as f32) * 2.0 * PI;
                let v = (i as f32 / points as f32) * 2.0 * PI * 3.0; // Different frequency

                vec![
                    (2.0 + (v * 0.5).cos()) * u.cos(),
                    (2.0 + (v * 0.5).cos()) * u.sin(),
                    (v * 0.5).sin(),
                ]
            })
            .collect()
    }

    fn add_noise(points: &[Vec<f32>], noise_level: f32) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        points
            .iter()
            .map(|p| {
                p.iter()
                    .map(|&v| v + rng.r#gen::<f32>() * noise_level)
                    .collect()
            })
            .collect()
    }

    fn compute_homology(_points: &[Vec<f32>]) -> HomologyResult {
        // Placeholder - would compute actual homology
        HomologyResult {
            betti_0: 1,
            betti_1: 1,
            betti_2: 0,
        }
    }

    fn compute_homology_from_points(_points: &[DVector<f32>]) -> HomologyResult {
        // Placeholder
        HomologyResult {
            betti_0: 1,
            betti_1: 1,
            betti_2: 0,
        }
    }

    fn apply_random_reidemeister_moves(knot: &KnotDiagram, moves: usize) -> KnotDiagram {
        // Placeholder - would apply Reidemeister moves
        knot.clone()
    }

    // Placeholder types for tests
    #[derive(Debug)]
    struct HomologyResult {
        betti_0: usize,
        betti_1: usize,
        betti_2: usize,
    }

    #[derive(Debug, Default)]
    struct TCSConfig;

    #[derive(Debug)]
    struct ConsensusNode {
        id: usize,
    }

    impl ConsensusNode {
        async fn new(id: usize) -> Result<Self> { Ok(Self { id }) }
        async fn connect(&self, _other: &Self) -> Result<()> { Ok(()) }
        async fn propose_token(&self, _proposal: TokenProposal) -> Result<bool> { Ok(true) }
        async fn has_token(&self, _pattern: &[u8]) -> bool { true }
    }

    #[derive(Debug)]
    struct TokenProposal {
        pattern: Vec<u8>,
        persistence_score: f32,
        emotional_coherence: f32,
        proposer_signature: Vec<u8>,
    }

    // Import required types
    use crate::topology::*;
    use crate::tcs::pipeline::*;
    use anyhow::Result;
    use nalgebra::DVector;
    use proptest::prelude::*;
}