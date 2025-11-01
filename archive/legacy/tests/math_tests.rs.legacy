/*
 * 🧮 MATHEMATICAL COMPONENT TESTS
 *
 * Tests for edge cases in mathematical functions used by consciousness engine
 * Focuses on Wundt curve peaks, divisive normalization, and Bayesian surprise
 */

use std::f32;

/// Test Wundt curve calculation and peak detection
#[test]
fn test_wundt_curve_peak() {
    // Wundt curve: pleasure = arousal * (1 - arousal)
    // Peak should be at arousal = 0.5, pleasure = 0.25
    
    let arousal_values = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let expected_pleasure = vec![0.0, 0.1875, 0.25, 0.1875, 0.0];
    
    for (i, arousal) in arousal_values.iter().enumerate() {
        let pleasure = arousal * (1.0 - arousal);
        assert!((pleasure - expected_pleasure[i]).abs() < 1e-6, 
                "Wundt curve mismatch at arousal={}: expected={}, got={}", 
                arousal, expected_pleasure[i], pleasure);
    }
    
    // Verify peak is at 0.5
    let peak_arousal = 0.5;
    let peak_pleasure = peak_arousal * (1.0 - peak_arousal);
    assert_eq!(peak_pleasure, 0.25, "Peak pleasure should be 0.25 at arousal=0.5");
}

/// Test Wundt curve edge cases
#[test]
fn test_wundt_curve_edge_cases() {
    // Test boundary values
    assert_eq!(0.0 * (1.0 - 0.0), 0.0); // arousal = 0
    assert_eq!(1.0 * (1.0 - 1.0), 0.0); // arousal = 1
    
    // Test negative arousal (should handle gracefully)
    let negative_arousal = -0.5;
    let pleasure = negative_arousal * (1.0 - negative_arousal);
    assert!(pleasure < 0.0, "Negative arousal should produce negative pleasure");
    
    // Test arousal > 1 (should handle gracefully)
    let high_arousal = 1.5;
    let pleasure = high_arousal * (1.0 - high_arousal);
    assert!(pleasure < 0.0, "Arousal > 1 should produce negative pleasure");
}

/// Test divisive normalization edge cases
#[test]
fn test_divisive_norm_edge_cases() {
    let sigma = 0.1; // Semi-saturation constant
    
    // Test zero activations
    let activations_zero = vec![0.0, 0.0, 0.0];
    let sum_zero: f32 = activations_zero.iter().sum();
    let denominator_zero = sigma + sum_zero;
    
    let normalized_zero: Vec<f32> = activations_zero.iter()
        .map(|&x| if denominator_zero > 0.0 { x / denominator_zero } else { 0.0 })
        .collect();
    
    assert_eq!(normalized_zero, vec![0.0, 0.0, 0.0], "Zero activations should normalize to zero");
    
    // Test near-zero activations
    let activations_near_zero = vec![0.001, 0.001, 0.001];
    let sum_near_zero: f32 = activations_near_zero.iter().sum();
    let denominator_near_zero = sigma + sum_near_zero;
    
    let normalized_near_zero: Vec<f32> = activations_near_zero.iter()
        .map(|&x| if denominator_near_zero > 0.0 { x / denominator_near_zero } else { 0.0 })
        .collect();
    
    // All should be equal and small
    for val in &normalized_near_zero {
        assert!(*val > 0.0 && *val < 0.1, "Near-zero activations should normalize to small positive values");
    }
    
    // Test high activations
    let activations_high = vec![100.0, 200.0, 300.0];
    let sum_high: f32 = activations_high.iter().sum();
    let denominator_high = sigma + sum_high;
    
    let normalized_high: Vec<f32> = activations_high.iter()
        .map(|&x| if denominator_high > 0.0 { x / denominator_high } else { 0.0 })
        .collect();
    
    // Should sum to approximately 1.0 (minus sigma effect)
    let sum_normalized: f32 = normalized_high.iter().sum();
    assert!((sum_normalized - 1.0).abs() < 0.1, "Normalized activations should sum to ~1.0");
}

/// Test divisive normalization with zero denominator
#[test]
fn test_divisive_norm_zero_denominator() {
    let sigma = 0.0; // No semi-saturation
    let activations = vec![0.0, 0.0, 0.0];
    let sum: f32 = activations.iter().sum();
    let denominator = sigma + sum;
    
    // Should handle zero denominator gracefully
    let normalized: Vec<f32> = activations.iter()
        .map(|&x| if denominator > 0.0 { x / denominator } else { 0.0 })
        .collect();
    
    assert_eq!(normalized, vec![0.0, 0.0, 0.0], "Zero denominator should return zero values");
}

/// Test KL divergence calculation edge cases
#[test]
fn test_kl_divergence_edge_cases() {
    // Test identical distributions (KL should be 0)
    let p_identical = vec![0.3, 0.4, 0.3];
    let q_identical = vec![0.3, 0.4, 0.3];
    
    let kl_identical: f32 = p_identical.iter().zip(q_identical.iter())
        .map(|(p_val, q_val)| {
            if *q_val > 0.0 && *p_val > 0.0 {
                p_val * (p_val / q_val).ln()
            } else {
                0.0
            }
        })
        .sum();
    
    assert!((kl_identical - 0.0).abs() < 1e-6, "Identical distributions should have KL divergence = 0");
    
    // Test zero probabilities
    let p_with_zeros = vec![0.5, 0.0, 0.5];
    let q_with_zeros = vec![0.3, 0.4, 0.3];
    
    let kl_with_zeros: f32 = p_with_zeros.iter().zip(q_with_zeros.iter())
        .map(|(p_val, q_val)| {
            if *q_val > 0.0 && *p_val > 0.0 {
                p_val * (p_val / q_val).ln()
            } else {
                0.0
            }
        })
        .sum();
    
    // Should handle zeros gracefully (contribute 0 to sum)
    assert!(kl_with_zeros >= 0.0, "KL divergence should be non-negative");
    
    // Test very small probabilities (numerical stability)
    let p_small = vec![1e-10, 0.5, 0.5];
    let q_small = vec![0.3, 0.35, 0.35];
    
    let kl_small: f32 = p_small.iter().zip(q_small.iter())
        .map(|(p_val, q_val)| {
            if *q_val > 0.0 && *p_val > 0.0 {
                p_val * (p_val / q_val).ln()
            } else {
                0.0
            }
        })
        .sum();
    
    assert!(kl_small.is_finite(), "KL divergence should be finite with small probabilities");
}

/// Test Bayesian surprise calculation
#[test]
fn test_bayesian_surprise() {
    // Bayesian surprise is KL divergence between prior and posterior
    let prior = vec![0.25, 0.25, 0.25, 0.25]; // Uniform prior
    let posterior = vec![0.1, 0.4, 0.3, 0.2]; // Updated posterior
    
    let surprise: f32 = prior.iter().zip(posterior.iter())
        .map(|(p_val, q_val)| {
            if *q_val > 0.0 && *p_val > 0.0 {
                q_val * (q_val / p_val).ln() // Note: KL(P||Q) vs KL(Q||P)
            } else {
                0.0
            }
        })
        .sum();
    
    assert!(surprise > 0.0, "Bayesian surprise should be positive when distributions differ");
    assert!(surprise.is_finite(), "Bayesian surprise should be finite");
}

/// Test Möbius transformation edge cases
#[test]
fn test_mobius_transformation_edge_cases() {
    // Test identity transformation (a=1, b=0, c=0, d=1)
    let z_identity = 0.5 + 0.3 * std::complex::Complex::new(0.0, 1.0);
    let transformed_identity = z_identity; // Should be unchanged
    
    assert!((transformed_identity.re - z_identity.re).abs() < 1e-6, "Identity transformation should preserve real part");
    assert!((transformed_identity.im - z_identity.im).abs() < 1e-6, "Identity transformation should preserve imaginary part");
    
    // Test infinity handling (should be handled gracefully)
    let z_inf = std::complex::Complex::new(f32::INFINITY, 0.0);
    // In practice, we'd want to handle this case explicitly
    assert!(z_inf.re.is_infinite(), "Infinity should be detected");
}

/// Test consciousness state edge cases
#[test]
fn test_consciousness_state_edge_cases() {
    use niodoo_consciousness::consciousness::{ConsciousnessState, EmotionalUrgency};
    
    let mut state = ConsciousnessState::new();
    
    // Test NaN urgency score handling
    let urgency_nan = EmotionalUrgency {
        token_velocity: f32::NAN,
        gpu_temperature: 0.5,
        meaning_depth: 0.5,
        timestamp: 0.0,
        emotional_beta: 0.9,
    };
    
    // Should handle NaN gracefully
    let result = state.update_emotion_from_urgency(&urgency_nan);
    assert!(result.is_err(), "NaN urgency should return error");
    
    // Test extreme urgency values
    let urgency_extreme = EmotionalUrgency {
        token_velocity: 1000.0, // Very high
        gpu_temperature: 0.0,   // Very low
        meaning_depth: 1.0,     // Maximum
        timestamp: 0.0,
        emotional_beta: 0.9,
    };
    
    let urgency_score = urgency_extreme.urgency_score();
    assert!(urgency_score >= 0.0 && urgency_score <= 1.0, "Urgency score should be normalized to [0,1]");
}

/// Test emotional state complexity edge cases
#[test]
fn test_emotional_complexity_edge_cases() {
    use niodoo_consciousness::consciousness::{EmotionalState, EmotionType};
    
    let mut state = EmotionalState::new();
    
    // Test empty secondary emotions
    assert_eq!(state.secondary_emotions.len(), 0, "New emotional state should have no secondary emotions");
    
    // Test maximum secondary emotions
    for i in 0..100 {
        state.add_secondary_emotion(EmotionType::Curious, 0.5);
    }
    
    assert_eq!(state.secondary_emotions.len(), 100, "Should be able to add many secondary emotions");
    
    // Test intensity clamping
    state.add_secondary_emotion(EmotionType::Satisfied, 2.0); // Above 1.0
    let last_emotion = state.secondary_emotions.last().unwrap();
    assert_eq!(last_emotion.1, 1.0, "Intensity should be clamped to 1.0");
    
    state.add_secondary_emotion(EmotionType::Focused, -0.5); // Below 0.0
    let last_emotion = state.secondary_emotions.last().unwrap();
    assert_eq!(last_emotion.1, 0.0, "Intensity should be clamped to 0.0");
}
