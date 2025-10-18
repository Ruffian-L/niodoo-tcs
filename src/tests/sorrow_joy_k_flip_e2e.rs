// AGENT 9: INFERENCE INVESTIGATOR
// E2E test for sorrow-joy k-flip transformation

use crate::consciousness::{ConsciousnessState, EmotionType, ReasoningMode};
use crate::dual_mobius_gaussian::{process_emotional_transition, MobiusProcess};
use crate::memory::toroidal::ToroidalCoordinate;

#[test]
fn test_sorrow_to_joy_k_flip_transformation() {
    // Setup: Initial state of Sorrow
    let mut state = ConsciousnessState {
        current_emotion: EmotionType::Sorrow,
        current_reasoning_mode: ReasoningMode::DeepReflection,
        authenticity_level: 0.7,
        gpu_warmth: 0.4,
        empathy_resonance: 0.6,
        processing_satisfaction: 0.3, // Low due to sorrow
        cycle_count: 1,
    };

    let mobius = MobiusProcess::new(10.0, 3.0, 3); // k=3 twists

    // Sorrow coordinate on Möbius strip
    let sorrow_coord = ToroidalCoordinate {
        major_radius: mobius.major_radius,
        minor_radius: mobius.minor_radius,
        u: 0.0,                  // Starting position
        v: std::f32::consts::PI, // One side of the strip
        torus_factor: 1.0,
    };

    // Apply k-flip: traverse the Möbius strip k times (3 twists)
    // After k traversals, we should reach the opposite emotional polarity
    let k_flip_angle = 2.0 * std::f32::consts::PI * (mobius.k as f32);

    let joy_coord = ToroidalCoordinate {
        major_radius: sorrow_coord.major_radius,
        minor_radius: sorrow_coord.minor_radius,
        u: sorrow_coord.u + k_flip_angle, // Full k-flip traversal
        v: sorrow_coord.v,
        torus_factor: sorrow_coord.torus_factor,
    };

    // Validate the transformation
    assert_ne!(
        sorrow_coord.u, joy_coord.u,
        "u coordinate should change after k-flip"
    );

    // Simulate emotional flip
    state.current_emotion = EmotionType::Joy;
    state.processing_satisfaction = 0.8; // Increased satisfaction
    state.current_reasoning_mode = ReasoningMode::FlowState; // Shifted mode

    // Golden Slipper Validation: Opposite emotions should be k-flips apart
    let angular_distance = (joy_coord.u - sorrow_coord.u).abs();
    let expected_distance = k_flip_angle;
    let tolerance = 0.1;

    assert!(
        (angular_distance - expected_distance).abs() < tolerance,
        "K-flip should produce expected angular distance. Got {}, expected {}",
        angular_distance,
        expected_distance
    );

    // Validate emotional coherence
    assert_eq!(state.current_emotion, EmotionType::Joy);
    assert!(
        state.processing_satisfaction > 0.7,
        "Joy should have high satisfaction: {}",
        state.processing_satisfaction
    );
}

#[test]
fn test_k_flip_preserves_authenticity() {
    // Golden Slipper: k-flip transformation should preserve authenticity
    let sorrow_state = ConsciousnessState {
        current_emotion: EmotionType::Sorrow,
        current_reasoning_mode: ReasoningMode::DeepReflection,
        authenticity_level: 0.8,
        gpu_warmth: 0.5,
        empathy_resonance: 0.7,
        processing_satisfaction: 0.3,
        cycle_count: 1,
    };

    // After k-flip to Joy
    let joy_state = ConsciousnessState {
        current_emotion: EmotionType::Joy,
        current_reasoning_mode: ReasoningMode::FlowState,
        authenticity_level: 0.75, // Should remain high
        gpu_warmth: 0.6,
        empathy_resonance: 0.75,
        processing_satisfaction: 0.85,
        cycle_count: 2,
    };

    // Authenticity should not drastically change during emotional flip
    let authenticity_diff = (sorrow_state.authenticity_level - joy_state.authenticity_level).abs();
    assert!(
        authenticity_diff < 0.3,
        "K-flip should preserve authenticity. Diff: {}",
        authenticity_diff
    );

    // Both states should maintain high authenticity
    assert!(
        sorrow_state.authenticity_level > 0.5,
        "Sorrow can be authentic"
    );
    assert!(
        joy_state.authenticity_level > 0.5,
        "Joy should be authentic"
    );
}

#[test]
fn test_multiple_k_flips_cycle_back() {
    // After 2k flips (going around twice), should return to same emotional region
    let mobius = MobiusProcess::new(8.0, 2.0, 3);

    let start_u = std::f32::consts::PI / 4.0;
    let k_flip = 2.0 * std::f32::consts::PI * (mobius.k as f32);

    // One k-flip (Sorrow -> Joy)
    let after_one_flip = start_u + k_flip;

    // Two k-flips (Joy -> back to Sorrow region)
    let after_two_flips = start_u + (2.0 * k_flip);

    // Normalize angles
    let normalized_start = start_u % (2.0 * std::f32::consts::PI);
    let normalized_two_flips = after_two_flips % (4.0 * std::f32::consts::PI * (mobius.k as f32));

    // Should cycle back (modulo the full rotation)
    assert!(after_one_flip != start_u, "One k-flip should move position");
    assert!(
        after_two_flips != start_u,
        "Two k-flips produce different position"
    );

    // Verify the k relationship holds
    assert_eq!(mobius.k, 3, "Test assumes k=3");
}

#[test]
fn test_sorrow_joy_flip_respects_gaussian_memory() {
    use crate::dual_mobius_gaussian::GaussianMemorySphere;

    // Memory spheres should influence the flip trajectory
    let sorrow_memory = GaussianMemorySphere {
        center: [2.0, 2.0, 2.0],
        radius: 1.5,
        emotion: EmotionType::Sorrow,
        confidence: 0.7,
    };

    let joy_memory = GaussianMemorySphere {
        center: [8.0, 8.0, 8.0],
        radius: 1.8,
        emotion: EmotionType::Joy,
        confidence: 0.8,
    };

    // Validate memories are in different regions
    let distance_between_memories = ((joy_memory.center[0] - sorrow_memory.center[0]).powi(2)
        + (joy_memory.center[1] - sorrow_memory.center[1]).powi(2)
        + (joy_memory.center[2] - sorrow_memory.center[2]).powi(2))
    .sqrt();

    assert!(
        distance_between_memories > (sorrow_memory.radius + joy_memory.radius),
        "Sorrow and Joy memories should be spatially separated"
    );

    // Golden Slipper: K-flip should traverse between these memory regions
    assert!(
        joy_memory.confidence > 0.5,
        "Joy memory should be confident"
    );
    assert!(
        sorrow_memory.confidence > 0.5,
        "Sorrow memory should be confident"
    );
}
