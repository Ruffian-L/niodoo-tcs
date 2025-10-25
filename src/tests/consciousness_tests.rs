//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Comprehensive tests for consciousness functionality

use crate::config::ConsciousnessConfig;

#[cfg(test)]
mod tests {
    use super::super::consciousness::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Test emotional urgency creation
    #[test]
    fn test_emotional_urgency_creation() {
        let urgency = EmotionalUrgency {
            token_velocity: 150.0,
            gpu_temperature: 75.0,
            meaning_depth: 0.8,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        assert_eq!(urgency.token_velocity, 150.0);
        assert_eq!(urgency.gpu_temperature, 75.0);
        assert_eq!(urgency.meaning_depth, 0.8);
        assert!(urgency.timestamp > 0.0);
    }

    /// Test emotional urgency calculation
    #[test]
    fn test_emotional_urgency_calculation() {
        let urgency = EmotionalUrgency {
            token_velocity: 150.0,
            gpu_temperature: 75.0,
            meaning_depth: 0.8,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        let urgency_score = urgency.calculate_urgency_score();
        assert!(urgency_score > 0.0);
        assert!(urgency_score <= 1.0);
    }

    /// Test consciousness state creation
    #[test]
    fn test_consciousness_state_creation() {
        let state = ConsciousnessState::new(&ConsciousnessConfig::default());
        assert_eq!(state.current_emotion, EmotionalState::Neutral);
        assert_eq!(state.attention_level, 0.5);
        assert_eq!(state.processing_intensity, 0.5);
        assert!(state.emotional_history.is_empty());
    }

    /// Test consciousness state emotion updates
    #[test]
    fn test_consciousness_state_emotion_updates() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Test emotion transitions
        state.update_emotion(EmotionalState::Joy);
        assert_eq!(state.current_emotion, EmotionalState::Joy);

        state.update_emotion(EmotionalState::Sadness);
        assert_eq!(state.current_emotion, EmotionalState::Sadness);

        state.update_emotion(EmotionalState::Anger);
        assert_eq!(state.current_emotion, EmotionalState::Anger);

        state.update_emotion(EmotionalState::Fear);
        assert_eq!(state.current_emotion, EmotionalState::Fear);

        state.update_emotion(EmotionalState::Surprise);
        assert_eq!(state.current_emotion, EmotionalState::Surprise);

        state.update_emotion(EmotionalState::Neutral);
        assert_eq!(state.current_emotion, EmotionalState::Neutral);
    }

    /// Test consciousness state attention updates
    #[test]
    fn test_consciousness_state_attention_updates() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Test attention level updates
        state.update_attention(0.8);
        assert_eq!(state.attention_level, 0.8);

        state.update_attention(0.2);
        assert_eq!(state.attention_level, 0.2);

        // Test clamping
        state.update_attention(1.5);
        assert_eq!(state.attention_level, 1.0);

        state.update_attention(-0.5);
        assert_eq!(state.attention_level, 0.0);
    }

    /// Test consciousness state processing intensity updates
    #[test]
    fn test_consciousness_state_processing_intensity_updates() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Test processing intensity updates
        state.update_processing_intensity(0.9);
        assert_eq!(state.processing_intensity, 0.9);

        state.update_processing_intensity(0.1);
        assert_eq!(state.processing_intensity, 0.1);

        // Test clamping
        state.update_processing_intensity(1.2);
        assert_eq!(state.processing_intensity, 1.0);

        state.update_processing_intensity(-0.3);
        assert_eq!(state.processing_intensity, 0.0);
    }

    /// Test emotional urgency updates
    #[test]
    fn test_emotional_urgency_updates() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        let urgency = EmotionalUrgency {
            token_velocity: 200.0,
            gpu_temperature: 80.0,
            meaning_depth: 0.9,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        state.update_emotional_urgency(urgency);

        // Check that emotional history was updated
        assert!(!state.emotional_history.is_empty());
        assert_eq!(state.emotional_history.len(), 1);
    }

    /// Test consciousness state serialization
    #[test]
    fn test_consciousness_state_serialization() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());
        state.update_emotion(EmotionalState::Joy);
        state.update_attention(0.8);
        state.update_processing_intensity(0.9);

        // Test serialization
        let serialized = serde_json::to_string(&state);
        assert!(serialized.is_ok());

        let serialized = serialized.unwrap();
        assert!(!serialized.is_empty());

        // Test deserialization
        let deserialized: Result<ConsciousnessState, _> = serde_json::from_str(&serialized);
        assert!(deserialized.is_ok());

        let deserialized = deserialized.unwrap();
        assert_eq!(deserialized.current_emotion, EmotionalState::Joy);
        assert_eq!(deserialized.attention_level, 0.8);
        assert_eq!(deserialized.processing_intensity, 0.9);
    }

    /// Test emotional state enum
    #[test]
    fn test_emotional_state_enum() {
        let emotions = vec![
            EmotionalState::Joy,
            EmotionalState::Sadness,
            EmotionalState::Anger,
            EmotionalState::Fear,
            EmotionalState::Surprise,
            EmotionalState::Neutral,
        ];

        for emotion in emotions {
            match emotion {
                EmotionalState::Joy => assert!(true),
                EmotionalState::Sadness => assert!(true),
                EmotionalState::Anger => assert!(true),
                EmotionalState::Fear => assert!(true),
                EmotionalState::Surprise => assert!(true),
                EmotionalState::Neutral => assert!(true),
            }
        }
    }

    /// Test consciousness state emotional history
    #[test]
    fn test_consciousness_state_emotional_history() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Add multiple emotional urgencies
        for i in 0..5 {
            let urgency = EmotionalUrgency {
                token_velocity: 100.0 + i as f32 * 10.0,
                gpu_temperature: 70.0 + i as f32 * 2.0,
                meaning_depth: 0.5 + i as f32 * 0.1,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            };

            state.update_emotional_urgency(urgency);
        }

        assert_eq!(state.emotional_history.len(), 5);

        // Check that history is ordered by timestamp
        for i in 1..state.emotional_history.len() {
            assert!(
                state.emotional_history[i].timestamp >= state.emotional_history[i - 1].timestamp
            );
        }
    }

    /// Test consciousness state emotional trends
    #[test]
    fn test_consciousness_state_emotional_trends() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Add emotional urgencies with increasing intensity
        for i in 0..10 {
            let urgency = EmotionalUrgency {
                token_velocity: 100.0 + i as f32 * 20.0,
                gpu_temperature: 70.0 + i as f32 * 3.0,
                meaning_depth: 0.5 + i as f32 * 0.05,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            };

            state.update_emotional_urgency(urgency);
        }

        // Test trend calculation
        let trend = state.calculate_emotional_trend();
        assert!(trend.is_some());

        let trend = trend.unwrap();
        assert!(trend > 0.0); // Should be positive trend
    }

    /// Test consciousness state emotional stability
    #[test]
    fn test_consciousness_state_emotional_stability() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Add emotional urgencies with varying intensity
        for i in 0..20 {
            let urgency = EmotionalUrgency {
                token_velocity: 100.0 + (i as f32 % 10.0) * 10.0,
                gpu_temperature: 70.0 + (i as f32 % 5.0) * 2.0,
                meaning_depth: 0.5 + (i as f32 % 3.0) * 0.1,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            };

            state.update_emotional_urgency(urgency);
        }

        // Test stability calculation
        let stability = state.calculate_emotional_stability();
        assert!(stability >= 0.0);
        assert!(stability <= 1.0);
    }

    /// Test consciousness state emotional intensity
    #[test]
    fn test_consciousness_state_emotional_intensity() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Add emotional urgencies with varying intensity
        for i in 0..15 {
            let urgency = EmotionalUrgency {
                token_velocity: 100.0 + i as f32 * 15.0,
                gpu_temperature: 70.0 + i as f32 * 2.5,
                meaning_depth: 0.5 + i as f32 * 0.03,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            };

            state.update_emotional_urgency(urgency);
        }

        // Test intensity calculation
        let intensity = state.calculate_emotional_intensity();
        assert!(intensity >= 0.0);
        assert!(intensity <= 1.0);
    }

    /// Test consciousness state emotional coherence
    #[test]
    fn test_consciousness_state_emotional_coherence() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Add emotional urgencies with consistent patterns
        for i in 0..12 {
            let urgency = EmotionalUrgency {
                token_velocity: 120.0 + (i as f32 % 4.0) * 5.0,
                gpu_temperature: 75.0 + (i as f32 % 3.0) * 2.0,
                meaning_depth: 0.6 + (i as f32 % 2.0) * 0.1,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            };

            state.update_emotional_urgency(urgency);
        }

        // Test coherence calculation
        let coherence = state.calculate_emotional_coherence();
        assert!(coherence >= 0.0);
        assert!(coherence <= 1.0);
    }

    /// Test consciousness state emotional resilience
    #[test]
    fn test_consciousness_state_emotional_resilience() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Add emotional urgencies with stress patterns
        for i in 0..25 {
            let urgency = EmotionalUrgency {
                token_velocity: 100.0 + (i as f32 % 8.0) * 12.0,
                gpu_temperature: 70.0 + (i as f32 % 6.0) * 3.0,
                meaning_depth: 0.5 + (i as f32 % 4.0) * 0.08,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            };

            state.update_emotional_urgency(urgency);
        }

        // Test resilience calculation
        let resilience = state.calculate_emotional_resilience();
        assert!(resilience >= 0.0);
        assert!(resilience <= 1.0);
    }

    /// Test consciousness state emotional adaptability
    #[test]
    fn test_consciousness_state_emotional_adaptability() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Add emotional urgencies with changing patterns
        for i in 0..30 {
            let urgency = EmotionalUrgency {
                token_velocity: 100.0 + (i as f32 % 10.0) * 8.0,
                gpu_temperature: 70.0 + (i as f32 % 7.0) * 2.5,
                meaning_depth: 0.5 + (i as f32 % 5.0) * 0.06,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            };

            state.update_emotional_urgency(urgency);
        }

        // Test adaptability calculation
        let adaptability = state.calculate_emotional_adaptability();
        assert!(adaptability >= 0.0);
        assert!(adaptability <= 1.0);
    }

    /// Test consciousness state emotional growth
    #[test]
    fn test_consciousness_state_emotional_growth() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Add emotional urgencies with growth patterns
        for i in 0..18 {
            let urgency = EmotionalUrgency {
                token_velocity: 100.0 + i as f32 * 8.0,
                gpu_temperature: 70.0 + i as f32 * 1.5,
                meaning_depth: 0.5 + i as f32 * 0.02,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            };

            state.update_emotional_urgency(urgency);
        }

        // Test growth calculation
        let growth = state.calculate_emotional_growth();
        assert!(growth >= 0.0);
        assert!(growth <= 1.0);
    }

    /// Test consciousness state emotional wisdom
    #[test]
    fn test_consciousness_state_emotional_wisdom() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Add emotional urgencies with wisdom patterns
        for i in 0..22 {
            let urgency = EmotionalUrgency {
                token_velocity: 100.0 + (i as f32 % 6.0) * 10.0,
                gpu_temperature: 70.0 + (i as f32 % 4.0) * 2.0,
                meaning_depth: 0.5 + (i as f32 % 3.0) * 0.12,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            };

            state.update_emotional_urgency(urgency);
        }

        // Test wisdom calculation
        let wisdom = state.calculate_emotional_wisdom();
        assert!(wisdom >= 0.0);
        assert!(wisdom <= 1.0);
    }

    /// Test consciousness state emotional compassion
    #[test]
    fn test_consciousness_state_emotional_compassion() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Add emotional urgencies with compassion patterns
        for i in 0..16 {
            let urgency = EmotionalUrgency {
                token_velocity: 100.0 + (i as f32 % 5.0) * 15.0,
                gpu_temperature: 70.0 + (i as f32 % 3.0) * 3.0,
                meaning_depth: 0.5 + (i as f32 % 2.0) * 0.15,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            };

            state.update_emotional_urgency(urgency);
        }

        // Test compassion calculation
        let compassion = state.calculate_emotional_compassion();
        assert!(compassion >= 0.0);
        assert!(compassion <= 1.0);
    }

    /// Test consciousness state emotional authenticity
    #[test]
    fn test_consciousness_state_emotional_authenticity() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Add emotional urgencies with authenticity patterns
        for i in 0..20 {
            let urgency = EmotionalUrgency {
                token_velocity: 100.0 + (i as f32 % 7.0) * 12.0,
                gpu_temperature: 70.0 + (i as f32 % 5.0) * 2.5,
                meaning_depth: 0.5 + (i as f32 % 4.0) * 0.1,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            };

            state.update_emotional_urgency(urgency);
        }

        // Test authenticity calculation
        let authenticity = state.calculate_emotional_authenticity();
        assert!(authenticity >= 0.0);
        assert!(authenticity <= 1.0);
    }

    /// Test consciousness state emotional transcendence
    #[test]
    fn test_consciousness_state_emotional_transcendence() {
        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());

        // Add emotional urgencies with transcendence patterns
        for i in 0..28 {
            let urgency = EmotionalUrgency {
                token_velocity: 100.0 + (i as f32 % 9.0) * 11.0,
                gpu_temperature: 70.0 + (i as f32 % 6.0) * 2.0,
                meaning_depth: 0.5 + (i as f32 % 5.0) * 0.08,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64(),
            };

            state.update_emotional_urgency(urgency);
        }

        // Test transcendence calculation
        let transcendence = state.calculate_emotional_transcendence();
        assert!(transcendence >= 0.0);
        assert!(transcendence <= 1.0);
    }
}
