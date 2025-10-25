use std::path::Path;

use chrono::Utc;

use niodoo_real_integrated::compass::{CompassOutcome, CompassQuadrant, MctsBranch};
use niodoo_real_integrated::data::{compute_dataset_stats, load_emotional_dataset};
use niodoo_real_integrated::erag::{CollapseResult, EmotionalVector, EragMemory};
use niodoo_real_integrated::generation::{GenerationResult, LensEcho};
use niodoo_real_integrated::learning::LearningLoop;
use niodoo_real_integrated::tokenizer::TokenizerEngine;
use niodoo_real_integrated::torus::{PadGhostState, TorusPadMapper};

#[test]
fn dataset_statistics_are_computed() {
    let path = Path::new("../data/training_data/emotion_training_data.json");
    let samples = load_emotional_dataset(path.to_str().unwrap(), Some(256)).expect("load samples");
    assert!(!samples.is_empty());
    let stats = compute_dataset_stats(&samples);
    assert!(stats.entropy_mean >= 0.0);
    assert!(stats.sample_count == samples.len());
}

#[test]
fn torus_projection_outputs_seven_dimensions() {
    let mut mapper = TorusPadMapper::new(42);
    let embedding: Vec<f32> = (0..896).map(|i| (i as f32 * 0.01).cos()).collect();
    let state = mapper.project(&embedding).expect("project");
    assert_eq!(state.pad.len(), 7);
    assert!(state.entropy >= 0.0);
}

#[test]
fn tokenizer_promotes_tokens_and_emits_ids() {
    let tokenizer_path = "../tokenizer.json";
    if !Path::new(tokenizer_path).exists() {
        eprintln!(
            "skipping tokenizer test; tokenizer spec not found at {}",
            tokenizer_path
        );
        return;
    }
    let mut engine = TokenizerEngine::new(tokenizer_path, 0.05).expect("tokenizer");

    let pad_state = PadGhostState {
        pad: [0.2; 7],
        entropy: 0.8,
        mu: [0.0; 7],
        sigma: [0.1; 7],
    };

    let collapse = CollapseResult {
        top_hits: vec![EragMemory {
            input: "test input".to_string(),
            output: "test output".to_string(),
            emotional_vector: EmotionalVector {
                joy: 0.1,
                sadness: 0.0,
                anger: 0.0,
                fear: 0.0,
                surprise: 0.0,
            },
            erag_context: vec!["context".to_string()],
            entropy_before: 0.4,
            entropy_after: 0.6,
            timestamp: Utc::now().to_rfc3339(),
            compass_state: Some("Discover".to_string()),
            quality_score: None,
            topology_betti: None,
            topology_knot_complexity: None,
        }],
        aggregated_context: "memory context".to_string(),
        average_similarity: 0.5,
    };

    let output = engine
        .process(
            "Rut prompt with emerging structure",
            &collapse,
            &pad_state,
            1.2,
        )
        .expect("process");

    assert!(!output.tokens.is_empty());
    assert!(output.vocab_size > 0);
}

#[test]
fn learning_loop_tracks_entropy_delta() {
    let mut loop_engine = LearningLoop::new(16, 0.2);
    let baseline_pad_state = PadGhostState {
        pad: [0.3; 7],
        entropy: 1.1,
        mu: [0.0; 7],
        sigma: [0.1; 7],
    };

    let pad_state = PadGhostState {
        pad: [0.4; 7],
        entropy: 1.5,
        mu: [0.1; 7],
        sigma: [0.2; 7],
    };

    let compass = CompassOutcome {
        quadrant: CompassQuadrant::Discover,
        is_threat: false,
        is_healing: true,
        mcts_branches: vec![MctsBranch {
            label: "branch".to_string(),
            ucb_score: 1.0,
            entropy_projection: 1.4,
        }],
        intrinsic_reward: 2.0,
    };

    let collapse = CollapseResult {
        top_hits: vec![],
        aggregated_context: String::new(),
        average_similarity: 0.0,
    };

    let generation = GenerationResult {
        baseline_response: "baseline".to_string(),
        hybrid_response: "hybrid".to_string(),
        echoes: vec![LensEcho {
            lens: "Claude".to_string(),
            response: "echo".to_string(),
        }],
        rouge_to_baseline: 0.7,
        latency_ms: 120.0,
        source: "test".to_string(),
    };

    loop_engine
        .update(&baseline_pad_state, &compass, &collapse, &generation)
        .expect("baseline update");

    let outcome = loop_engine
        .update(&pad_state, &compass, &collapse, &generation)
        .expect("update");

    assert!(outcome.events.iter().any(|e| e.contains("Entropy shift")));
    assert!(outcome.entropy_delta.abs() > 0.0);
}
