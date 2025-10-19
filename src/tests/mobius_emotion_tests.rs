//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use candle_core::{Device, Tensor};
use ndarray::{Array1, Array2};
use ndarray_linalg::Norm; // For cosine distance calculation

use crate::dual_mobius_gaussian::MobiusProcess;
use crate::emotional_lora::{EmotionalContext, PersonalityType};

/// Compute cosine distance between two vectors
fn cosine_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot_product = a.dot(b);
    let norm_a = a.norm();
    let norm_b = b.norm();

    // Prevent division by zero
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance
    }

    1.0 - (dot_product / (norm_a * norm_b))
}

#[cfg(test)]
mod mobius_emotion_tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Test Möbius emotional state transition without jitter
    #[test]
    fn test_sad_to_joy_baseline_transition() {
        let mobius = MobiusProcess::new();

        // Initial sad state vector
        let sad_state = Array1::from(vec![0.8, 0.2, 0.0]);

        // Apply Möbius transformation
        let (x, y, z) = mobius.transform(sad_state[0], sad_state[1], sad_state[2]);
        let transformed_state = Array1::from(vec![x, y, z]);

        // Baseline novelty calculation
        let baseline_novelty = cosine_distance(&sad_state, &transformed_state);

        // Assertions
        assert!(baseline_novelty >= 0.0, "Novelty should be non-negative");
        assert!(
            baseline_novelty <= 0.15,
            "Baseline transition should have minimal novelty"
        );
    }

    /// Test Möbius emotional state transition WITH jitter (nurture_jitter_enabled)
    #[test]
    fn test_sad_to_joy_jitter_transition() {
        let mobius = MobiusProcess::new();

        // Initial sad state vector
        let sad_state = Array1::from(vec![0.8, 0.2, 0.0]);

        // Apply Möbius transformation baseline
        let (base_x, base_y, base_z) = mobius.transform(sad_state[0], sad_state[1], sad_state[2]);
        let baseline_state = Array1::from(vec![base_x, base_y, base_z]);

        // Jittered transformation (using emotional weight to simulate nurture_jitter)
        let (jitter_x, jitter_y, jitter_z) = mobius.emotional_flip(
            sad_state[0],
            sad_state[1],
            0.5, // emotion_weight to simulate jitter
        );
        let jittered_state = Array1::from(vec![jitter_x, jitter_y, jitter_z]);

        // Novelty calculations
        let baseline_novelty = cosine_distance(&sad_state, &baseline_state);
        let jitter_novelty = cosine_distance(&sad_state, &jittered_state);

        // Compute novelty ratio
        let novelty_ratio = (jitter_novelty - baseline_novelty) / baseline_novelty.max(1e-10);

        // Assertions
        assert!(
            baseline_novelty >= 0.0,
            "Baseline novelty should be non-negative"
        );
        assert!(
            jitter_novelty > baseline_novelty,
            "Jittered transition should increase novelty"
        );
        assert!(
            novelty_ratio >= 0.15 && novelty_ratio <= 0.20,
            format!(
                "Novelty ratio {} should be between 0.15 and 0.20. Baseline: {}, Jittered: {}",
                novelty_ratio, baseline_novelty, jitter_novelty
            )
        );
    }

    /// Test edge cases for emotional transformation
    #[test]
    fn test_emotional_transformation_edge_cases() {
        let mobius = MobiusProcess::new();

        // Test 1: Extreme sadness
        let extreme_sad = Array1::from(vec![0.99, 0.01, 0.0]);
        let (x1, y1, z1) = mobius.transform(extreme_sad[0], extreme_sad[1], extreme_sad[2]);

        // Test 2: Ambiguous state
        let ambiguous_state = Array1::from(vec![0.5, 0.5, 0.0]);
        let (x2, y2, z2) =
            mobius.transform(ambiguous_state[0], ambiguous_state[1], ambiguous_state[2]);

        // Test 3: Pure joy
        let pure_joy = Array1::from(vec![0.0, 1.0, 0.0]);
        let (x3, y3, z3) = mobius.transform(pure_joy[0], pure_joy[1], pure_joy[2]);

        // Assertions to ensure no NaN outputs
        assert!(x1.is_finite(), "Extreme sadness transformation x");
        assert!(y1.is_finite(), "Extreme sadness transformation y");
        assert!(z1.is_finite(), "Extreme sadness transformation z");

        assert!(x2.is_finite(), "Ambiguous state transformation x");
        assert!(y2.is_finite(), "Ambiguous state transformation y");
        assert!(z2.is_finite(), "Ambiguous state transformation z");

        assert!(x3.is_finite(), "Pure joy transformation x");
        assert!(y3.is_finite(), "Pure joy transformation y");
        assert!(z3.is_finite(), "Pure joy transformation z");
    }
}
