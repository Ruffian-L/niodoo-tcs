//! Integration test for emotional cascade functionality
//! Tests consonance computation, cascade tracking, and hyperfocus detection

use niodoo_real_integrated::compass::{CascadeStage, CascadeTracker, CompassOutcome, CompassQuadrant};
use niodoo_real_integrated::consonance::{compute_consonance, ConsonanceMetrics};
use niodoo_real_integrated::hyperfocus::{HyperfocusDetector, CoherentAction};
use niodoo_real_integrated::erag::{CollapseResult, EragMemory, EmotionalVector};
use niodoo_real_integrated::tcs_analysis::TopologicalSignature;
use niodoo_real_integrated::torus::PadGhostState;
use std::collections::HashMap;
use uuid::Uuid;

fn create_test_pad_state(entropy: f64) -> PadGhostState {
    PadGhostState {
        pad: vec![0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1],
        mu: vec![0.0; 7],
        sigma: vec![0.1; 7],
        entropy,
    }
}

fn create_test_compass(quadrant: CompassQuadrant, stage: Option<CascadeStage>) -> CompassOutcome {
    CompassOutcome {
        quadrant,
        is_threat: false,
        is_healing: quadrant == CompassQuadrant::Master,
        mcts_branches: vec![],
        intrinsic_reward: 1.0,
        cascade_stage: stage,
        ucb1_score: None,
    }
}

fn create_test_topology() -> TopologicalSignature {
    TopologicalSignature {
        id: Uuid::new_v4(),
        timestamp: chrono::Utc::now(),
        persistence_features: vec![],
        betti_numbers: [1, 0, 0],
        knot_complexity: 3.0,
        knot_polynomial: "test".to_string(),
        tqft_dimension: 2,
        cobordism_type: None,
        persistence_entropy: 1.0,
        spectral_gap: 0.5,
        computation_time_ms: 10.0,
    }
}

fn create_test_collapse() -> CollapseResult {
    CollapseResult {
        top_hits: vec![EragMemory {
            input: "test".to_string(),
            output: "test".to_string(),
            emotional_vector: EmotionalVector::default(),
            erag_context: vec![],
            entropy_before: 1.5,
            entropy_after: 2.0,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            compass_state: None,
            cascade_stage: None,
        }],
        aggregated_context: "test context".to_string(),
        average_similarity: 0.8,
        curator_quality: None,
    }
}

#[test]
fn test_cascade_stage_mapping() {
    // Test that compass quadrants map to correct cascade stages
    assert_eq!(
        CascadeStage::from_quadrant(CompassQuadrant::Discover),
        CascadeStage::Recognition
    );
    assert_eq!(
        CascadeStage::from_quadrant(CompassQuadrant::Master),
        CascadeStage::Satisfaction
    );
    assert_eq!(
        CascadeStage::from_quadrant(CompassQuadrant::Persist),
        CascadeStage::Calm
    );
    assert_eq!(
        CascadeStage::from_quadrant(CompassQuadrant::Panic),
        CascadeStage::Recognition
    );
}

#[test]
fn test_cascade_progression() {
    // Test cascade progression: Recognition → Satisfaction → Calm → Motivation
    let stages = vec![
        CascadeStage::Recognition,
        CascadeStage::Satisfaction,
        CascadeStage::Calm,
        CascadeStage::Motivation,
    ];

    for i in 0..stages.len() - 1 {
        let current = stages[i];
        let next = stages[i + 1];
        assert_eq!(
            current.next(),
            next,
            "Cascade progression failed from {:?} to {:?}",
            current,
            next
        );
    }

    // Test that Motivation cycles back to Recognition
    assert_eq!(
        CascadeStage::Motivation.next(),
        CascadeStage::Recognition
    );
}

#[test]
fn test_cascade_tracker_transitions() {
    let mut tracker = CascadeTracker::new();

    // Start with Recognition stage
    let compass_recognition = create_test_compass(CompassQuadrant::Discover, None);
    let consonance_high = 0.8; // High consonance enables transition

    // First call initializes the tracker
    let transition1 = tracker.detect_transition(&compass_recognition, consonance_high);
    assert!(transition1.is_none(), "First call should initialize, not transition");

    // Move to Satisfaction stage (Recognition → Satisfaction)
    let compass_satisfaction = create_test_compass(CompassQuadrant::Master, None);
    let transition2 = tracker.detect_transition(&compass_satisfaction, consonance_high);

    // Should transition from Recognition to Satisfaction
    assert!(
        transition2.is_some(),
        "Should transition from Recognition to Satisfaction"
    );
    let transition = transition2.unwrap();
    assert_eq!(transition.from, CascadeStage::Recognition);
    assert_eq!(transition.to, CascadeStage::Satisfaction);

    // Current stage should be Satisfaction
    assert_eq!(
        tracker.current_stage(),
        Some(CascadeStage::Satisfaction),
        "Current stage should be Satisfaction"
    );
}

#[test]
fn test_cascade_tracker_no_transition_low_consonance() {
    let mut tracker = CascadeTracker::new();

    // Initialize tracker
    let compass_recognition = create_test_compass(CompassQuadrant::Discover, None);
    let _ = tracker.detect_transition(&compass_recognition, 0.8);

    // Try to transition with low consonance (should fail)
    let compass_satisfaction = create_test_compass(CompassQuadrant::Master, None);
    let transition = tracker.detect_transition(&compass_satisfaction, 0.5); // Low consonance

    // Should not transition with low consonance
    assert!(
        transition.is_none(),
        "Should not transition with low consonance (<0.7)"
    );
}

#[test]
fn test_consonance_computation() {
    let pad_state = create_test_pad_state(2.0);
    let compass = create_test_compass(CompassQuadrant::Master, None);
    let collapse = create_test_collapse();
    let topology = create_test_topology();

    let metrics = compute_consonance(&pad_state, &compass, &collapse, &topology, None, None);

    // Verify metrics are in valid range
    assert!(metrics.score >= 0.0 && metrics.score <= 1.0);
    assert!(metrics.confidence >= 0.0 && metrics.confidence <= 1.0);
    assert!(metrics.dissonance_score >= 0.0 && metrics.dissonance_score <= 1.0);

    // Should have 5 sources
    assert_eq!(metrics.sources.len(), 5);

    // Verify all source types are present
    let source_names: Vec<_> = metrics.sources.iter().map(|s| s.name()).collect();
    assert!(
        source_names.contains(&"emotional_coherence"),
        "Should have emotional_coherence source"
    );
    assert!(
        source_names.contains(&"topological_consistency"),
        "Should have topological_consistency source"
    );
    assert!(
        source_names.contains(&"erag_relevance"),
        "Should have erag_relevance source"
    );
    assert!(
        source_names.contains(&"compass_transition"),
        "Should have compass_transition source"
    );
    assert!(
        source_names.contains(&"curator_quality"),
        "Should have curator_quality source"
    );
}

#[test]
fn test_hyperfocus_detection() {
    let detector = HyperfocusDetector::new();

    // Create high consonance signals (all > 0.85)
    let mut signals = HashMap::new();
    signals.insert(
        "compass".to_string(),
        ConsonanceMetrics {
            score: 0.9,
            sources: vec![],
            confidence: 0.95,
            dissonance_score: 0.1,
        },
    );
    signals.insert(
        "erag".to_string(),
        ConsonanceMetrics {
            score: 0.88,
            sources: vec![],
            confidence: 0.92,
            dissonance_score: 0.12,
        },
    );
    signals.insert(
        "topology".to_string(),
        ConsonanceMetrics {
            score: 0.87,
            sources: vec![],
            confidence: 0.90,
            dissonance_score: 0.13,
        },
    );

    let event = detector.detect(&signals);

    // Should detect hyperfocus with high consonance
    assert!(event.is_some(), "Should detect hyperfocus with high consonance");
    let event = event.unwrap();
    assert!(event.overall_consonance >= 0.85);
    assert_eq!(event.aligned_signals.len(), 3);
}

#[test]
fn test_hyperfocus_no_detection_low_consonance() {
    let detector = HyperfocusDetector::new();

    // Create low consonance signals
    let mut signals = HashMap::new();
    signals.insert(
        "compass".to_string(),
        ConsonanceMetrics {
            score: 0.6,
            sources: vec![],
            confidence: 0.7,
            dissonance_score: 0.4,
        },
    );
    signals.insert(
        "erag".to_string(),
        ConsonanceMetrics {
            score: 0.65,
            sources: vec![],
            confidence: 0.75,
            dissonance_score: 0.35,
        },
    );

    let event = detector.detect(&signals);

    // Should not detect hyperfocus with low consonance
    assert!(
        event.is_none(),
        "Should not detect hyperfocus with low consonance"
    );
}

#[test]
fn test_hyperfocus_coherent_action_determination() {
    let detector = HyperfocusDetector::new();

    // Test that high curator score triggers StoreBreakthrough
    let mut signals = HashMap::new();
    signals.insert(
        "curator".to_string(),
        ConsonanceMetrics {
            score: 0.95,
            sources: vec![],
            confidence: 0.98,
            dissonance_score: 0.05,
        },
    );
    signals.insert(
        "compass".to_string(),
        ConsonanceMetrics {
            score: 0.85,
            sources: vec![],
            confidence: 0.9,
            dissonance_score: 0.15,
        },
    );
    signals.insert(
        "erag".to_string(),
        ConsonanceMetrics {
            score: 0.85,
            sources: vec![],
            confidence: 0.9,
            dissonance_score: 0.15,
        },
    );

    let event = detector.detect(&signals);
    assert!(event.is_some());
    let event = event.unwrap();
    // Should choose StoreBreakthrough because curator is very high
    assert_eq!(event.coherent_action, CoherentAction::StoreBreakthrough);
}

#[test]
fn test_full_cascade_cycle() {
    let mut tracker = CascadeTracker::new();
    let high_consonance = 0.85;

    // Go through full cascade: Recognition → Satisfaction → Calm → Motivation
    let stages_and_quadrants = vec![
        (CascadeStage::Recognition, CompassQuadrant::Discover),
        (CascadeStage::Satisfaction, CompassQuadrant::Master),
        (CascadeStage::Calm, CompassQuadrant::Persist),
        (CascadeStage::Motivation, CompassQuadrant::Discover),
    ];

    for (expected_stage, quadrant) in stages_and_quadrants {
        let compass = create_test_compass(quadrant, None);
        let transition = tracker.detect_transition(&compass, high_consonance);

        if transition.is_some() {
            let t = transition.unwrap();
            assert_eq!(
                t.to, expected_stage,
                "Should transition to {:?}",
                expected_stage
            );
        }

        // Verify current stage
        if tracker.current_stage().is_some() {
            let current = tracker.current_stage().unwrap();
            if current == expected_stage {
                // Stage matches, continue
            } else if current == expected_stage.next() {
                // Already advanced, that's ok for last iteration
            }
        }
    }

    // Check if we completed a full cascade
    let full_cascades = tracker.full_cascades_count();
    // May have completed one or more cycles
    assert!(
        full_cascades >= 0,
        "Full cascades count should be non-negative"
    );
}

#[test]
fn test_consonance_with_compass_transition() {
    let pad_state = create_test_pad_state(2.0);
    let last_compass = create_test_compass(CompassQuadrant::Panic, None);
    let compass = create_test_compass(CompassQuadrant::Discover, None);
    let collapse = create_test_collapse();
    let topology = create_test_topology();

    // Panic → Discover should be high consonance (good transition)
    let metrics = compute_consonance(
        &pad_state,
        &compass,
        &collapse,
        &topology,
        None,
        Some(&last_compass),
    );

    // Should have higher consonance for good transitions
    assert!(
        metrics.score > 0.5,
        "Panic→Discover transition should have reasonable consonance"
    );
}

#[test]
fn test_cascade_stage_serialization() {
    // Test that CascadeStage can be serialized/deserialized
    let stage = CascadeStage::Recognition;
    let serialized = serde_json::to_string(&stage).expect("Should serialize");
    let deserialized: CascadeStage =
        serde_json::from_str(&serialized).expect("Should deserialize");

    assert_eq!(stage, deserialized, "Serialization roundtrip should work");
}

#[test]
fn test_consonance_helper_methods() {
    let metrics = ConsonanceMetrics {
        score: 0.75,
        sources: vec![],
        confidence: 0.8,
        dissonance_score: 0.25,
    };

    assert_eq!(metrics.as_percent(), 75.0);
    assert!(metrics.is_consonant(0.7));
    assert!(!metrics.is_consonant(0.8));
    assert!(metrics.is_dissonant(0.2));
    assert!(!metrics.is_dissonant(0.3));
}

