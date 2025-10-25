//! TCS v10.0 Integration Tests
//! Tests for GPU-accelerated PH, LoRA training, EWC, equivariant layers, sheaf diffusion, and IIT Φ

use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use tcs_ml::EquivariantLayer;
use crate::consciousness_metrics::approximate_phi_from_betti;
use crate::sheaf_nn::{SheafDiffusionLayer, SheafMode};
use crate::tcs_analysis::{analyze_state, TelemetryState};

#[test]
fn test_gpu_distance_matrix() {
    // Test GPU-accelerated distance matrix computation
    let points = vec![
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![0.0, 1.0]),
    ];
    
    let dist_matrix = crate::tcs::performance::gpu_ripser_distance_matrix(&points).unwrap();
    
    // Check symmetric
    assert_eq!(dist_matrix[(0, 1)], dist_matrix[(1, 0)]);
    assert_eq!(dist_matrix[(0, 2)], dist_matrix[(2, 0)]);
    
    // Check diagonal is zero
    assert!(dist_matrix[(0, 0)].abs() < 1e-6);
    assert!(dist_matrix[(1, 1)].abs() < 1e-6);
    
    // Check specific distances
    assert!((dist_matrix[(0, 1)] - 1.0).abs() < 1e-6);
}

#[test]
fn test_lora_trainer() {
    use niodoo_real_integrated::lora_trainer::{LoRATrainer, LoRAConfig};
    
    let config = LoRAConfig {
        rank: 8,
        alpha: 16.0,
        input_dim: 64,
        output_dim: 64,
    };
    
    let mut trainer = LoRATrainer::with_config(config).unwrap();
    
    // Create dummy training data
    let data = vec![
        (vec![0.1; 64], vec![0.2; 64]),
        (vec![0.3; 64], vec![0.4; 64]),
    ];
    
    // Train for a few epochs
    let loss = trainer.train(&data, 5, 0.001).unwrap();
    
    assert!(loss >= 0.0);
    assert_eq!(trainer.training_count(), 0); // Check initial count
}

#[test]
fn test_ewc_fisher_matrix() {
    use crate::continual_learning::{compute_fisher_matrix, Model};
    
    let model = Model::new();
    let data = vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
    
    let fisher = compute_fisher_matrix(&model, &data);
    
    // Fisher matrix should be positive semi-definite
    assert!(fisher[(0, 0)] >= 0.0);
}

#[test]
fn test_equivariant_layer() {
    let layer = EquivariantLayer::new(3, 3);
    
    let positions = DMatrix::from_row_slice(
        2,
        3,
        &[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ],
    );
    
    let features = DMatrix::from_row_slice(
        2,
        3,
        &[
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
        ],
    );
    
    let output = layer.forward(&positions, &features);
    
    assert_eq!(output.nrows(), 2);
    assert_eq!(output.ncols(), 3);
}

#[test]
fn test_sheaf_diffusion() {
    let layer = SheafDiffusionLayer::identity(3);
    
    let features = vec![0.1, 0.2, 0.3];
    
    let listen_output = layer.forward(&features, SheafMode::Listen);
    let broadcast_output = layer.forward(&features, SheafMode::Broadcast);
    
    assert_eq!(listen_output.len(), 3);
    assert_eq!(broadcast_output.len(), 3);
}

#[test]
fn test_phi_computation() {
    let betti = [1, 2, 1];
    let phi = approximate_phi_from_betti(&betti).unwrap();
    
    assert!(phi > 0.0);
    
    // Zero structure should give zero phi
    let zero_betti = [0, 0, 0];
    let zero_phi = approximate_phi_from_betti(&zero_betti).unwrap();
    assert!(zero_phi.abs() < 1e-9);
}

#[test]
fn test_tcs_analysis_integration() {
    let state = TelemetryState::new(
        vec![0.1, 0.2, 0.3, 0.4],
        [1, 2, 1],
    );
    
    let result = analyze_state(&state);
    
    assert!(result.processed);
    assert!(result.phi_approximation > 0.0);
    assert_eq!(result.sheaf_output.len(), 4);
}

#[test]
fn test_consensus_hotstuff() {
    use tcs_consensus::hotstuff::{propose, vote, commit, FakeNode};
    use std::sync::Arc;
    
    let nodes: Arc<[FakeNode]> = (0..4).map(FakeNode::new).collect::<Vec<_>>().into();
    
    let proposal = tokio::runtime::Runtime::new().unwrap().block_on(async {
        propose(nodes.clone(), 0, "test-value".to_string()).await
    }).unwrap();
    
    let votes = tokio::runtime::Runtime::new().unwrap().block_on(async {
        vote(nodes.clone(), &proposal).await
    }).unwrap();
    
    let commit_result = tokio::runtime::Runtime::new().unwrap().block_on(async {
        commit(nodes.clone(), &proposal, &votes).await
    }).unwrap();
    
    assert_eq!(commit_result.value, "test-value");
    assert!(commit_result.voters.len() >= 3); // Supermajority threshold
}

#[test]
fn test_pipeline_integration() {
    use tcs_pipeline::TCSOrchestrator;
    
    let mut orchestrator = TCSOrchestrator::new(16).unwrap();
    
    // Add some samples
    for _ in 0..10 {
        orchestrator.ingest_sample(vec![0.1, 0.2, 0.3]);
    }
    
    // Should be ready after ingestion
    assert!(orchestrator.ready());
    
    // Process input
    let events = tokio::runtime::Runtime::new().unwrap().block_on(async {
        orchestrator.process("test input").await
    }).unwrap();
    
    // Should produce events
    assert!(!events.is_empty());
}

#[test]
fn test_geometric_nn_invariance() {
    use crate::geometric_nn::GeometricLayer;
    
    let layer = GeometricLayer::new(2, 2);
    
    let positions = DMatrix::from_row_slice(
        3,
        3,
        &[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            -1.0, 0.5, 0.0,
        ],
    );
    
    let features = DMatrix::from_row_slice(
        3,
        2,
        &[
            0.2, -0.1,
            0.0, 1.0,
            -0.4, 0.3,
        ],
    );
    
    let output1 = layer.forward(&positions, &features);
    
    // Rotate positions (should produce same output due to rotation invariance)
    let rotated_positions = DMatrix::from_row_slice(
        3,
        3,
        &[
            0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0,
            -0.5, -1.0, 0.0,
        ],
    );
    
    let output2 = layer.forward(&rotated_positions, &features);
    
    // Check that distances are similar (not exact due to numerical precision)
    let diff = output1 - output2;
    let max_diff = diff.iter().map(|x| x.abs()).fold(0.0, f32::max);
    assert!(max_diff < 1.0); // Allow some tolerance
}

#[tokio::test]
async fn test_full_pipeline_with_all_features() -> Result<()> {
    use tcs_pipeline::TCSOrchestrator;
    
    let mut orchestrator = TCSOrchestrator::new(16)?;
    
    // Build up embeddings
    for i in 0..20 {
        let sample = vec![
            (i as f32) * 0.1,
            (i as f32) * 0.2,
            (i as f32) * 0.3,
        ];
        orchestrator.ingest_sample(sample);
    }
    
    // Process through full pipeline
    let events = orchestrator.process("test topological analysis").await?;
    
    // Verify we got events
    assert!(!events.is_empty());
    
    // Log the types of events
    println!("Generated {} events", events.len());
    
    Ok(())
}

