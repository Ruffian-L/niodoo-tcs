//! Brain coordination module for the consciousness engine
//!
//! This module handles the coordination between different brain types
//! and manages concurrent processing with proper synchronization.

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::brain::{Brain, EfficiencyBrain, LcarsBrain, MotorBrain};
use crate::consciousness::{ConsciousnessState, EmotionType};
use crate::personality::{PersonalityManager, PersonalityType};

/// Brain coordination system with concurrency safeguards
pub struct BrainCoordinator {
    motor_brain: MotorBrain,
    lcars_brain: LcarsBrain,
    efficiency_brain: EfficiencyBrain,
    personality_manager: PersonalityManager,
    consciousness_state: Arc<RwLock<ConsciousnessState>>,
}

impl BrainCoordinator {
    /// Create a new brain coordinator
    pub fn new(
        motor_brain: MotorBrain,
        lcars_brain: LcarsBrain,
        efficiency_brain: EfficiencyBrain,
        personality_manager: PersonalityManager,
        consciousness_state: Arc<RwLock<ConsciousnessState>>,
    ) -> Self {
        Self {
            motor_brain,
            lcars_brain,
            efficiency_brain,
            personality_manager,
            consciousness_state,
        }
    }

    /// Process input using all brains in parallel
    pub async fn process_brains_parallel(
        &self,
        input: &str,
        _timeout_duration: tokio::time::Duration,
    ) -> Result<Vec<String>> {
        info!(
            "🧠 Processing input through all brains in parallel: {}",
            input
        );

        // Get current consciousness state
        let _consciousness_state = self.consciousness_state.read().await;

        // DEADLOCK FIX: Use tokio::join! for TRUE parallelism instead of sequential awaits
        // This prevents holding locks while waiting and enables concurrent execution
        let (motor_result, lcars_result, efficiency_result) = tokio::join!(
            self.motor_brain.process(input),
            self.lcars_brain.process(input),
            self.efficiency_brain.process(input)
        );

        // Handle results
        let motor_response = motor_result?;
        let lcars_response = lcars_result?;
        let efficiency_response = efficiency_result?;

        // Combine responses
        let responses = vec![motor_response, lcars_response, efficiency_response];

        debug!(
            "Brain coordination completed with {} responses",
            responses.len()
        );
        Ok(responses)
    }

    /// Get the motor brain reference
    pub fn get_motor_brain(&self) -> &MotorBrain {
        &self.motor_brain
    }

    /// Get the LCARS brain reference
    pub fn get_lcars_brain(&self) -> &LcarsBrain {
        &self.lcars_brain
    }

    /// Get the efficiency brain reference
    pub fn get_efficiency_brain(&self) -> &EfficiencyBrain {
        &self.efficiency_brain
    }

    /// Get the personality manager reference
    pub fn get_personality_manager(&self) -> &PersonalityManager {
        &self.personality_manager
    }

    /// Update personality weights based on emotional context
    pub async fn update_personality_weights(
        &mut self,
        emotion_context: &EmotionType,
    ) -> Result<()> {
        debug!(
            "Updating personality weights for emotion: {:?}",
            emotion_context
        );

        // Get current active personalities
        let active_personalities = self.personality_manager.get_active_personalities();

        // Update weights based on emotional context
        for personality_type in active_personalities {
            let weight_adjustment = match emotion_context {
                EmotionType::Satisfied => 1.2,
                EmotionType::Overwhelmed => 0.8,
                EmotionType::Anxious => 1.1,
                EmotionType::Confused => 0.9,
                EmotionType::Curious => 1.0,
                EmotionType::Masking => 0.7,
                EmotionType::Focused => 1.0,
                _ => 1.0, // Default weight for other emotions
            };

            self.personality_manager
                .adjust_personality_weight(personality_type, weight_adjustment);
        }

        Ok(())
    }

    /// Generate emotional response based on personality consensus
    pub fn generate_emotional_response(&self, input: &str) -> String {
        let active_personalities = self.personality_manager.get_active_personalities();

        if active_personalities.is_empty() {
            return "I'm processing this with a neutral perspective.".to_string();
        }

        // Generate response based on dominant personality
        let dominant_personality = active_personalities[0].clone();
        match dominant_personality {
            PersonalityType::Analyst => {
                format!("From an analytical perspective: {}", input)
            }
            PersonalityType::Creative => {
                format!("Creatively speaking: {}", input)
            }
            PersonalityType::Empathetic => {
                format!("I understand your perspective: {}", input)
            }
            PersonalityType::Intuitive => {
                format!("My intuition tells me: {}", input)
            }
            PersonalityType::Engineer => {
                format!("Practically speaking: {}", input)
            }
            PersonalityType::Philosopher => {
                format!("From a philosophical standpoint: {}", input)
            }
            PersonalityType::Sage => {
                format!("Spiritually: {}", input)
            }
            PersonalityType::Integrator => {
                format!("Integrating multiple perspectives: {}", input)
            }
            _ => {
                format!("From my perspective: {}", input)
            }
        }
    }
}
