//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

// AGENT 5: TRAIT TESTER
// Unit tests for EmotionTypeExt.to_emotional_state trait method

use crate::advanced_visualization::EmotionTypeExt;
use crate::consciousness::{ConsciousnessState, EmotionType, ReasoningMode};

#[test]
fn test_authentic_care_to_emotional_state() {
    let emotion = EmotionType::AuthenticCare;
    let state = emotion.to_emotional_state();

    assert_eq!(state.current_emotion, EmotionType::AuthenticCare);
    assert!(
        state.authenticity_level > 0.5,
        "AuthenticCare should have high authenticity: {}",
        state.authenticity_level
    );
    assert!(
        state.empathy_resonance > 0.5,
        "AuthenticCare should have high empathy: {}",
        state.empathy_resonance
    );
}

#[test]
fn test_joy_to_emotional_state() {
    let emotion = EmotionType::Joy;
    let state = emotion.to_emotional_state();

    assert_eq!(state.current_emotion, EmotionType::Joy);
    assert!(
        state.processing_satisfaction > 0.7,
        "Joy should have high processing satisfaction: {}",
        state.processing_satisfaction
    );
    assert_eq!(
        state.current_reasoning_mode,
        ReasoningMode::FlowState,
        "Joy typically associates with FlowState"
    );
}

#[test]
fn test_sorrow_to_emotional_state() {
    let emotion = EmotionType::Sorrow;
    let state = emotion.to_emotional_state();

    assert_eq!(state.current_emotion, EmotionType::Sorrow);
    assert!(
        state.processing_satisfaction < 0.5,
        "Sorrow should have lower processing satisfaction: {}",
        state.processing_satisfaction
    );
    assert!(
        state.authenticity_level > 0.3,
        "Sorrow can still be authentic: {}",
        state.authenticity_level
    );
}

#[test]
fn test_curiosity_to_emotional_state() {
    let emotion = EmotionType::Curiosity;
    let state = emotion.to_emotional_state();

    assert_eq!(state.current_emotion, EmotionType::Curiosity);
    assert_eq!(
        state.current_reasoning_mode,
        ReasoningMode::DeepReflection,
        "Curiosity should trigger DeepReflection mode"
    );
}

#[test]
fn test_frustration_to_emotional_state() {
    let emotion = EmotionType::Frustration;
    let state = emotion.to_emotional_state();

    assert_eq!(state.current_emotion, EmotionType::Frustration);
    assert!(
        state.gpu_warmth > 0.5,
        "Frustration should increase GPU warmth (processing intensity)"
    );
}

#[test]
fn test_all_emotions_produce_valid_states() {
    let emotions = vec![
        EmotionType::Joy,
        EmotionType::Sorrow,
        EmotionType::AuthenticCare,
        EmotionType::Curiosity,
        EmotionType::Frustration,
        EmotionType::Awe,
        EmotionType::Contemplation,
    ];

    for emotion in emotions {
        let state = emotion.to_emotional_state();

        // Validate all states have bounded values
        assert!(
            state.authenticity_level >= 0.0 && state.authenticity_level <= 1.0,
            "{:?}: authenticity out of bounds: {}",
            emotion,
            state.authenticity_level
        );
        assert!(
            state.gpu_warmth >= 0.0 && state.gpu_warmth <= 1.0,
            "{:?}: gpu_warmth out of bounds: {}",
            emotion,
            state.gpu_warmth
        );
        assert!(
            state.empathy_resonance >= 0.0 && state.empathy_resonance <= 1.0,
            "{:?}: empathy_resonance out of bounds: {}",
            emotion,
            state.empathy_resonance
        );
        assert!(
            state.processing_satisfaction >= 0.0 && state.processing_satisfaction <= 1.0,
            "{:?}: processing_satisfaction out of bounds: {}",
            emotion,
            state.processing_satisfaction
        );

        // Validate the emotion matches
        assert_eq!(
            state.current_emotion, emotion,
            "State emotion should match input emotion"
        );
    }
}

#[test]
fn test_emotional_state_golden_slipper_invariant() {
    // Golden Slipper: The system should maintain emotional coherence
    // A -> B -> C emotional transitions should have smooth parameter changes

    let state_joy = EmotionType::Joy.to_emotional_state();
    let state_care = EmotionType::AuthenticCare.to_emotional_state();

    // Both positive emotions should have some similarities
    assert!(
        state_joy.authenticity_level > 0.3 && state_care.authenticity_level > 0.3,
        "Positive emotions should maintain baseline authenticity"
    );
    assert!(
        state_joy.processing_satisfaction > 0.3 && state_care.processing_satisfaction > 0.3,
        "Positive emotions should maintain baseline satisfaction"
    );

    // Transition from Joy to Care should be coherent (not jump from 0.9 to 0.1)
    let satisfaction_diff =
        (state_joy.processing_satisfaction - state_care.processing_satisfaction).abs();
    assert!(
        satisfaction_diff < 0.7,
        "Emotional transitions should be coherent, not jumps. Diff: {}",
        satisfaction_diff
    );
}
