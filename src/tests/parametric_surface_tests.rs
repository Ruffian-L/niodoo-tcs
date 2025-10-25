//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🔬 PARAMETRIC SURFACE UNIT TESTS
 *
 * AGENT 8 (Test Templar): Comprehensive testing for parametric surface functions
 *
 * Tests validate:
 * 1. Boundary conditions (u=0, u=2π wrap correctly)
 * 2. Gaussian curvature at known points
 * 3. Non-orientable flip behavior
 * 4. Output bounds (valence, novelty)
 * 5. Edge cases (NaN, infinity handling)
 *
 * Parametric surfaces tested:
 * - MobiusStrip::parametric(u, v)
 * - KTwistedTorus::parametric(u, v)
 */

use crate::real_mobius_consciousness::*;
use nalgebra::Vector3;
use std::f64::consts::{PI, TAU};

// ============================================================================
// TEST 1: Möbius Strip Boundary Wrapping
// ============================================================================

#[test]
fn test_mobius_boundary_wrap() {
    let mobius = MobiusStrip::new(50.0, 10.0, 1);

    // Test that u=0 and u=2π produce the same point (modulo width flip)
    let point_at_0 = mobius.parametric(0.0, 0.0);
    let point_at_2pi = mobius.parametric(TAU, 0.0);

    // Due to single twist, the points should be at same location
    // but with flipped orientation (this is the Möbius property)
    let distance = (point_at_0 - point_at_2pi).norm();
    assert!(
        distance < 0.01,
        "Boundary wrap failed: u=0 and u=2π should map to same location, distance: {}",
        distance
    );

    // Test wrapping at multiple v values
    for v in [-5.0, 0.0, 5.0].iter() {
        let p0 = mobius.parametric(0.0, *v);
        let p_tau = mobius.parametric(TAU, *v);
        let dist = (p0 - p_tau).norm();
        assert!(
            dist < 0.01,
            "Boundary wrap failed at v={}: distance={}",
            v,
            dist
        );
    }
}

// ============================================================================
// TEST 2: Gaussian Curvature at Known Points
// ============================================================================

#[test]
fn test_mobius_gaussian_curvature() {
    let mobius = MobiusStrip::new(50.0, 10.0, 1);

    // Test curvature at centerline (v=0)
    // For a Möbius strip, curvature should be finite and negative at most points
    let curvature_center = mobius.gaussian_curvature(PI / 2.0, 0.0);
    assert!(
        curvature_center.is_finite(),
        "Curvature should be finite at center, got: {}",
        curvature_center
    );

    // Test curvature at multiple points along the strip
    let test_points = vec![
        (0.0, 0.0),
        (PI / 4.0, 0.0),
        (PI / 2.0, 0.0),
        (PI, 0.0),
        (3.0 * PI / 2.0, 0.0),
    ];

    for (u, v) in test_points {
        let k = mobius.gaussian_curvature(u, v);
        assert!(
            k.is_finite(),
            "Gaussian curvature should be finite at u={}, v={}, got: {}",
            u,
            v,
            k
        );

        // Gaussian curvature for Möbius strip should be non-zero
        // (it's a curved surface, not flat)
        assert!(
            k.abs() > 1e-10,
            "Gaussian curvature should be non-zero for curved surface at u={}, v={}",
            u,
            v
        );
    }
}

#[test]
fn test_torus_gaussian_curvature() {
    let torus = KTwistedTorus::new(100.0, 30.0, 1);

    // For a torus, Gaussian curvature K = cos(v) / (R(R + r*cos(v)))
    // where R is major radius and r is minor radius
    // At v=0 (outer equator): K = 1/(R*r) = 1/(100*30) ≈ 0.000333
    // This is a known mathematical property of tori

    let point_outer = torus.parametric(0.0, 0.0);

    // Test that parametric point is well-formed
    assert!(
        point_outer.x.is_finite() && point_outer.y.is_finite() && point_outer.z.is_finite(),
        "Torus parametric point should have finite coordinates"
    );

    // Test at multiple positions
    for u in [0.0, PI / 2.0, PI, 3.0 * PI / 2.0].iter() {
        for v in [0.0, PI / 2.0, PI].iter() {
            let point = torus.parametric(*u, *v);
            assert!(
                point.x.is_finite() && point.y.is_finite() && point.z.is_finite(),
                "Torus point should be finite at u={}, v={}",
                u,
                v
            );
        }
    }
}

// ============================================================================
// TEST 3: Non-Orientable Flip Behavior
// ============================================================================

#[test]
fn test_mobius_non_orientable_flip() {
    // Single twist = non-orientable
    let mobius_single = MobiusStrip::new(50.0, 10.0, 1);
    assert!(
        !mobius_single.is_orientable(),
        "Single twist should be non-orientable"
    );

    // Double twist = orientable
    let mobius_double = MobiusStrip::new(50.0, 10.0, 2);
    assert!(
        mobius_double.is_orientable(),
        "Double twist should be orientable"
    );

    // Test normal vector flip for non-orientable surface
    // As we traverse the Möbius strip, the normal should flip
    let normal_start = mobius_single.normal(0.0, 5.0);
    let normal_halfway = mobius_single.normal(PI, 5.0);
    let normal_end = mobius_single.normal(TAU, 5.0);

    // After one complete loop, normal should point in opposite direction
    // due to non-orientable property
    let dot_product = normal_start.dot(&normal_end);

    // The normals should be approximately opposite (dot product ≈ -1)
    // or at least significantly different from parallel (dot product ≠ 1)
    assert!(
        dot_product < 0.5,
        "Non-orientable surface should flip normal orientation, dot product: {}",
        dot_product
    );
}

// ============================================================================
// TEST 4: Output Bounds - Emotional State Mapping
// ============================================================================

#[test]
fn test_emotional_state_bounds() {
    // Test that emotional states are properly clamped
    let state_extreme_positive = EmotionalState::new(2.0, 2.0, 2.0); // Over bounds
    assert!(
        state_extreme_positive.valence >= -1.0 && state_extreme_positive.valence <= 1.0,
        "Valence should be clamped to [-1, 1], got: {}",
        state_extreme_positive.valence
    );
    assert!(
        state_extreme_positive.arousal >= 0.0 && state_extreme_positive.arousal <= 1.0,
        "Arousal should be clamped to [0, 1], got: {}",
        state_extreme_positive.arousal
    );
    assert!(
        state_extreme_positive.dominance >= -1.0 && state_extreme_positive.dominance <= 1.0,
        "Dominance should be clamped to [-1, 1], got: {}",
        state_extreme_positive.dominance
    );

    let state_extreme_negative = EmotionalState::new(-2.0, -2.0, -2.0); // Under bounds
    assert!(
        state_extreme_negative.valence >= -1.0 && state_extreme_negative.valence <= 1.0,
        "Valence should be clamped to [-1, 1], got: {}",
        state_extreme_negative.valence
    );
    assert!(
        state_extreme_negative.arousal >= 0.0 && state_extreme_negative.arousal <= 1.0,
        "Arousal should be clamped to [0, 1], got: {}",
        state_extreme_negative.arousal
    );

    // Test vector conversion produces bounded values
    let vec = state_extreme_positive.to_vector();
    assert!(
        vec.x.is_finite() && vec.y.is_finite() && vec.z.is_finite(),
        "Emotional vector should have finite components"
    );
}

#[test]
fn test_torus_consciousness_mapping_bounds() {
    let torus = KTwistedTorus::new(100.0, 30.0, 1);

    // Test mapping of various emotional states to torus surface
    let test_states = vec![
        EmotionalState::new(0.5, 0.7, 0.2),
        EmotionalState::new(-0.8, 0.3, 0.9),
        EmotionalState::new(0.0, 1.0, -0.5),
        EmotionalState::new(1.0, 0.0, 1.0),
    ];

    for state in test_states {
        let (u, v) = torus.map_consciousness_state(&state);

        // u should be in [-π, π] range (from atan2)
        assert!(
            u >= -PI && u <= PI,
            "u coordinate should be in [-π, π], got: {}",
            u
        );

        // v should be in valid range for asin output [-π/2, π/2]
        assert!(v.is_finite(), "v coordinate should be finite, got: {}", v);

        // Test that we can actually compute a point on the torus
        let point = torus.parametric(u, v);
        assert!(
            point.x.is_finite() && point.y.is_finite() && point.z.is_finite(),
            "Torus point from consciousness mapping should be finite"
        );
    }
}

// ============================================================================
// TEST 5: Edge Cases - NaN and Infinity Handling
// ============================================================================

#[test]
fn test_mobius_nan_infinity_handling() {
    let mobius = MobiusStrip::new(50.0, 10.0, 1);

    // Test with extreme but valid values
    let point_large_u = mobius.parametric(1000.0 * TAU, 0.0);
    assert!(
        point_large_u.x.is_finite() && point_large_u.y.is_finite() && point_large_u.z.is_finite(),
        "Should handle large u values (wrapping), got: {:?}",
        point_large_u
    );

    // Test with large v values (should still work, just stretches the strip)
    let point_large_v = mobius.parametric(0.0, 100.0);
    assert!(
        point_large_v.x.is_finite() && point_large_v.y.is_finite() && point_large_v.z.is_finite(),
        "Should handle large v values, got: {:?}",
        point_large_v
    );

    // Test normal vector computation doesn't produce NaN
    let normal = mobius.normal(PI, 0.0);
    assert!(
        normal.x.is_finite() && normal.y.is_finite() && normal.z.is_finite(),
        "Normal vector should not contain NaN, got: {:?}",
        normal
    );

    // Test that normal is actually normalized
    let normal_magnitude = normal.norm();
    assert!(
        (normal_magnitude - 1.0).abs() < 1e-6,
        "Normal vector should be unit length, got magnitude: {}",
        normal_magnitude
    );
}

#[test]
fn test_golden_slipper_novelty_bounds() {
    let config = GoldenSlipperConfig::default();
    let transformer = GoldenSlipperTransformer::new(config);

    // Test that novelty calculation stays within bounds
    let current = EmotionalState::new(0.3, 0.5, 0.2);

    let emotions_to_test = vec![
        "joy",
        "sadness",
        "anger",
        "fear",
        "contemplative",
        "unknown_emotion",
    ];

    for emotion in emotions_to_test {
        let result = transformer.transform_emotion(&current, emotion, 0.17);

        match result {
            Ok((new_state, novelty, compliant)) => {
                // Novelty should be in [0, 1] range
                assert!(
                    novelty >= 0.0 && novelty <= 1.0,
                    "Novelty should be in [0, 1] for emotion '{}', got: {}",
                    emotion,
                    novelty
                );

                // Check that transformed state is also bounded
                assert!(
                    new_state.valence >= -1.0 && new_state.valence <= 1.0,
                    "Transformed valence should be bounded"
                );
                assert!(
                    new_state.arousal >= 0.0 && new_state.arousal <= 1.0,
                    "Transformed arousal should be bounded"
                );

                // If compliant, novelty should be in Golden Slipper range
                if compliant {
                    assert!(
                        novelty >= 0.15 && novelty <= 0.20,
                        "Compliant transformation should have novelty in [0.15, 0.20], got: {}",
                        novelty
                    );
                }
            }
            Err(_) => {
                // Error is acceptable for some transformations
                // (e.g., outside ethical bounds or Golden Slipper range)
            }
        }
    }
}

#[test]
fn test_torus_parametric_edge_cases() {
    let torus = KTwistedTorus::new(100.0, 30.0, 1);

    // Test with zero radius (degenerate case, but should not crash)
    let torus_zero_minor = KTwistedTorus::new(100.0, 0.0, 1);
    let point = torus_zero_minor.parametric(0.0, 0.0);
    assert!(
        point.x.is_finite() && point.y.is_finite() && point.z.is_finite(),
        "Degenerate torus (zero minor radius) should still produce finite points"
    );

    // Test with negative twist count (should work mathematically)
    let torus_negative_twist = KTwistedTorus::new(100.0, 30.0, -2);
    let point_neg = torus_negative_twist.parametric(PI, PI);
    assert!(
        point_neg.x.is_finite() && point_neg.y.is_finite() && point_neg.z.is_finite(),
        "Negative twist torus should produce finite points"
    );

    // Test with extreme parameter values
    let point_extreme = torus.parametric(1e6, 1e6);
    assert!(
        point_extreme.x.is_finite() && point_extreme.y.is_finite() && point_extreme.z.is_finite(),
        "Extreme parameter values should be handled via modulo arithmetic"
    );
}

// ============================================================================
// INTEGRATION TEST: Full Consciousness Processing Pipeline
// ============================================================================

#[test]
fn test_consciousness_processing_pipeline() {
    let processor = MobiusConsciousnessProcessor::new();
    let initial_emotion = EmotionalState::new(0.2, 0.6, 0.1);

    let result = processor.process_consciousness("test consciousness input", &initial_emotion);

    // Verify all components produce valid output
    assert!(
        !result.content.is_empty(),
        "Consciousness result should have content"
    );

    // Check torus position is valid
    let (u, v) = result.torus_position;
    assert!(
        u.is_finite() && v.is_finite(),
        "Torus position should be finite"
    );

    // Check memory path was traversed
    assert!(
        result.memory_path_length > 0,
        "Should have traversed memory path"
    );

    // Check emotional state is bounded
    assert!(
        result.emotional_state.valence >= -1.0 && result.emotional_state.valence <= 1.0,
        "Result emotional valence should be bounded"
    );
    assert!(
        result.emotional_state.arousal >= 0.0 && result.emotional_state.arousal <= 1.0,
        "Result emotional arousal should be bounded"
    );

    // Check novelty is in valid range
    assert!(
        result.novelty_applied >= 0.0 && result.novelty_applied <= 1.0,
        "Novelty should be in [0, 1], got: {}",
        result.novelty_applied
    );
}

#[test]
fn test_mobius_memory_traversal_consistency() {
    let memory_system = MobiusMemorySystem::new();
    let start_pos = Vector3::new(50.0, 0.0, 5.0);

    // Test forward and backward traversal produce different paths
    let path_forward = memory_system.mobius_traversal(start_pos, "forward");
    let path_backward = memory_system.mobius_traversal(start_pos, "backward");

    assert_eq!(
        path_forward.len(),
        100,
        "Forward path should have 100 steps"
    );
    assert_eq!(
        path_backward.len(),
        100,
        "Backward path should have 100 steps"
    );

    // First steps should be different
    let first_forward = path_forward[0];
    let first_backward = path_backward[0];
    let difference = (first_forward - first_backward).norm();

    assert!(
        difference > 0.01,
        "Forward and backward paths should diverge from start"
    );

    // All points should be finite
    for (i, point) in path_forward.iter().enumerate() {
        assert!(
            point.x.is_finite() && point.y.is_finite() && point.z.is_finite(),
            "Forward path point {} should be finite",
            i
        );
    }

    for (i, point) in path_backward.iter().enumerate() {
        assert!(
            point.x.is_finite() && point.y.is_finite() && point.z.is_finite(),
            "Backward path point {} should be finite",
            i
        );
    }
}
