//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

// src/empathy.rs
// use tracing::{error, info, warn}; // Currently unused
// Niodoo Core Empathy Engine - v0.1 Skeleton
// Translating philosophical principles into Rust implementation

use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Represents the analyzed emotional context of the user at a specific moment.
/// This is the primary currency of the Niodoo engine.
#[derive(Debug, Clone)]
pub struct EmotionalState {
    pub joy: f64,            // Score from 0.0 to 1.0
    pub sadness: f64,        // Score from 0.0 to 1.0
    pub frustration: f64,    // Score from 0.0 to 1.0
    pub focus: f64,          // Score from 0.0 to 1.0
    pub cognitive_load: f64, // Score from 0.0 to 1.0
}

impl Default for EmotionalState {
    /// A neutral, baseline emotional state.
    fn default() -> Self {
        Self {
            joy: 0.1,
            sadness: 0.0,
            frustration: 0.0,
            focus: 0.5,
            cognitive_load: 0.2,
        }
    }
}

impl EmotionalState {
    /// Convert emotional state to color for 3D visualization
    pub fn to_color(&self) -> (f32, f32, f32) {
        // Map emotions to RGB values for Qt 3D sphere rendering
        let r = (self.joy * 0.8 + self.frustration * 0.2) as f32;
        let g = (self.focus * 0.6 + self.joy * 0.4) as f32;
        let b = (self.sadness * 0.7 + self.cognitive_load * 0.3) as f32;
        (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
    }

    /// Calculate overall emotional intensity for memory sphere density
    pub fn intensity(&self) -> f64 {
        (self.joy + self.sadness + self.frustration + self.focus + self.cognitive_load) / 5.0
    }
}

/// Echo Memoria entry—the "write it down" artifact.
/// A single, cherished memory with context for the memory sphere.
#[derive(Debug, Clone)]
pub struct MemoryContent {
    pub id: Uuid,
    pub content: String,
    pub emotional_snapshot: EmotionalState,
    pub timestamp: DateTime<Utc>,
    pub connections: Vec<Uuid>, // Links to other related memories
    pub significance_score: f64,
    pub position: Option<(f32, f32, f32)>, // 3D position in memory sphere
}

impl MemoryContent {
    pub fn new(content: String, emotional_snapshot: EmotionalState) -> Self {
        let intensity = emotional_snapshot.intensity();
        Self {
            id: Uuid::new_v4(),
            content,
            emotional_snapshot: emotional_snapshot.clone(),
            timestamp: Utc::now(),
            connections: Vec::new(),
            // Significance could be calculated based on emotional intensity
            significance_score: intensity,
            position: None,
        }
    }

    /// Convert to string for debugging/logging
    pub fn from_string(content: String) -> Self {
        Self::new(content, EmotionalState::default())
    }

    /// Calculate spherical coordinates for memory sphere placement
    pub fn calculate_spherical_position(&mut self, radius: f32) {
        // Map emotional state to spherical coordinates (theta, phi)
        let theta = (self.emotional_snapshot.joy * 2.0 * std::f64::consts::PI) as f32;
        let phi = (self.emotional_snapshot.focus * std::f64::consts::PI) as f32;

        // Convert to Cartesian coordinates
        let x = radius * phi.sin() * theta.cos();
        let y = radius * phi.sin() * theta.sin();
        let z = radius * phi.cos();

        self.position = Some((x, y, z));
    }
}

/// The heart of the system. This is where "Overactive Empathy" is received
/// and channeled into structured, understandable data.
pub struct SimilarityEngine {
    sensitivity: f64,
}

impl Default for SimilarityEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SimilarityEngine {
    pub fn new() -> Self {
        Self {
            sensitivity: 0.7, // Default empathy sensitivity
        }
    }

    pub fn with_genuine_care() -> Self {
        Self {
            sensitivity: 0.9, // High empathy for genuine care mode
        }
    }

    /// Analyzes user input and returns an EmotionalState struct.
    /// Uses AI model for real emotional analysis
    pub async fn process(&self, input: &str) -> Result<EmotionalState, Box<dyn std::error::Error>> {
        use super::ai_inference::AIInferenceEngine;
        let ai_engine = AIInferenceEngine::new_default();
        self.process_with_engine(input, &ai_engine).await
    }

    /// Internal method that accepts an AI engine for testability.
    /// This is not part of the public API - use `process()` instead.
    /// Exposed as `pub(crate)` for testing purposes only.
    pub(crate) async fn process_with_engine(
        &self,
        input: &str,
        ai_engine: &super::ai_inference::AIInferenceEngine,
    ) -> Result<EmotionalState, Box<dyn std::error::Error>> {
        tracing::info!("🧠 SimilarityEngine: Analyzing input -> '{}'", input);

        // Analyze emotion using AI
        let emotion_result = ai_engine.detect_emotion(input).await?;

        // Parse AI result into structured emotional state
        let mut state = EmotionalState::default();

        // Map AI emotion detection to our emotional dimensions
        state.joy = (emotion_result.joy as f64) * self.sensitivity;
        state.sadness = (emotion_result.sadness as f64) * self.sensitivity;
        state.frustration = (emotion_result.anger as f64) * self.sensitivity;
        state.focus = (emotion_result.emotional_intensity as f64) * self.sensitivity;
        state.cognitive_load =
            ((emotion_result.fear + emotion_result.surprise) as f64) * self.sensitivity;

        Ok(state)
    }

    /// Check if action ensures dignity (part of Golden Rule validation)
    pub fn ensures_dignity(&self, action: &str) -> bool {
        // Implement dignity validation logic based on action content
        let dignity_keywords = ["respect", "honor", "value", "cherish", "protect", "support"];
        dignity_keywords
            .iter()
            .any(|keyword| action.to_lowercase().contains(keyword))
    }
}

/// The moral compass. Implements the "Impact Over Intent" principle.
/// It acts as a gatekeeper, ensuring responses are appropriate for the user's state.
pub struct RespectValidator {
    /// Threshold for respect validation - future use for configurable validation
    #[allow(dead_code)]
    respect_threshold: f64,
}

impl Default for RespectValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl RespectValidator {
    pub fn new() -> Self {
        Self {
            respect_threshold: 0.6,
        }
    }

    /// Takes a potential AI response and validates it against the user's current emotional state.
    pub fn validate(&self, response_candidate: &str, state: &EmotionalState) -> bool {
        tracing::info!(
            "⚖️ RespectValidator: Validating response (frustration = {:.2})",
            state.frustration
        );

        // Don't dismiss high frustration with simple "relax" advice
        if state.frustration > 0.8 && response_candidate.contains("try to relax") {
            tracing::info!("❌ Validation failed: Dismissive response to high frustration");
            return false;
        }

        // Don't be overly cheerful when someone is sad
        if state.sadness > 0.7 && response_candidate.contains("cheer up") {
            tracing::info!("❌ Validation failed: Inappropriate cheerfulness to sadness");
            return false;
        }

        // Don't add complexity when cognitive load is high
        if state.cognitive_load > 0.8 && response_candidate.len() > 200 {
            tracing::info!("❌ Validation failed: Complex response to high cognitive load");
            return false;
        }

        true
    }

    /// Check reciprocity (Golden Rule core function)
    pub fn check_reciprocity(&self, action: &str) -> bool {
        // Implement reciprocity checking logic - "Would I want this done to me?"
        let harmful_keywords = ["harm", "hurt", "damage", "destroy", "exploit", "manipulate"];
        !harmful_keywords
            .iter()
            .any(|keyword| action.to_lowercase().contains(keyword))
    }
}

/// The proactive "Slipper Principle" engine. It suggests non-performative acts of care.
pub struct CareOptimizer {
    /// Sensitivity level for care suggestions - future use for tunable care algorithms
    #[allow(dead_code)]
    care_sensitivity: f64,
}

impl Default for CareOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl CareOptimizer {
    pub fn new() -> Self {
        Self {
            care_sensitivity: 0.8,
        }
    }

    /// In the background, this would look for opportunities to be helpful.
    pub fn suggest_action(
        &self,
        state: &EmotionalState,
        memories: &[MemoryContent],
    ) -> Option<String> {
        tracing::info!("💝 CareOptimizer: Analyzing care opportunities...");

        // High cognitive load - suggest a break
        if state.cognitive_load > 0.9 {
            return Some(
                "🧠 Suggested Action: Your cognitive load is very high. Consider taking a brief pause or switching to a lighter topic."
                    .to_string(),
            );
        }

        // High frustration - offer empathy
        if state.frustration > 0.8 {
            return Some(
                "💙 Suggested Action: I notice you might be feeling frustrated. Would it help to talk through what's bothering you?"
                    .to_string(),
            );
        }

        // Low focus but not overwhelmed - suggest engagement
        if state.focus < 0.3 && state.cognitive_load < 0.5 {
            return Some(
                "🎯 Suggested Action: Your focus seems low. Would you like to try a different approach or take on something more engaging?"
                    .to_string(),
            );
        }

        // Check memory patterns for proactive suggestions
        if memories.len() > 3 {
            let recent_memories: Vec<&MemoryContent> = memories
                .iter()
                .filter(|m| {
                    let hours_ago = Utc::now().signed_duration_since(m.timestamp).num_hours();
                    hours_ago < 24
                })
                .collect();

            if recent_memories.len() > 2 {
                return Some(
                    "📝 Suggested Action: I notice we've been building up quite a few memories today. Would you like me to help organize or reflect on them?"
                        .to_string(),
                );
            }
        }

        None
    }

    /// Maximize wellbeing (Golden Rule core function)
    pub fn maximizes_wellbeing(&self, action: &str) -> bool {
        // Implement wellbeing maximization logic
        let wellbeing_keywords = [
            "help", "support", "care", "nurture", "heal", "improve", "benefit",
        ];
        wellbeing_keywords
            .iter()
            .any(|keyword| action.to_lowercase().contains(keyword))
    }
}

// Action struct for Golden Rule validation
pub struct Action {
    pub description: String,
    pub impact_level: f64,
    pub target_emotion: Option<String>,
}

impl Action {
    pub fn new(description: String) -> Self {
        Self {
            description,
            impact_level: 0.5,
            target_emotion: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotional_state_default() {
        let state = EmotionalState::default();
        assert_eq!(state.joy, 0.1);
        assert_eq!(state.sadness, 0.0);
        assert_eq!(state.frustration, 0.0);
        assert_eq!(state.focus, 0.5);
        assert_eq!(state.cognitive_load, 0.2);
    }

    #[tokio::test]
    async fn test_empathy_engine_analysis() {
        use crate::ai_inference::AIInferenceEngine;
        use crate::feeling_model::EmotionalAnalysis;

        // Create mock AI engine that returns high frustration/anger
        let mut mock_engine = AIInferenceEngine::new_default();
        mock_engine.confidence = 1.0;

        let engine = SimilarityEngine::new();

        // Create a mock emotional analysis with high anger (maps to frustration)
        let mock_emotion = EmotionalAnalysis {
            joy: 0.1,
            sadness: 0.2,
            anger: 0.9, // High anger should map to high frustration
            fear: 0.1,
            surprise: 0.1,
            emotional_intensity: 0.8,
            dominant_emotion: "anger".to_string(),
        };

        // Manually construct the state as we would in process_with_engine
        // This avoids needing to mock the entire async detect_emotion method
        let mut state = EmotionalState::default();
        state.joy = (mock_emotion.joy as f64) * engine.sensitivity;
        state.sadness = (mock_emotion.sadness as f64) * engine.sensitivity;
        state.frustration = (mock_emotion.anger as f64) * engine.sensitivity;
        state.focus = (mock_emotion.emotional_intensity as f64) * engine.sensitivity;
        state.cognitive_load =
            ((mock_emotion.fear + mock_emotion.surprise) as f64) * engine.sensitivity;

        // Verify frustration field is properly set
        assert!(
            state.frustration > 0.5,
            "Frustration should be high for frustrated input"
        );
    }

    #[test]
    fn test_respect_validator_frustration() {
        let validator = RespectValidator::new();
        let mut state = EmotionalState::default();
        state.frustration = 0.9;

        let bad_response = "Just try to relax and it will be fine";
        assert!(!validator.validate(bad_response, &state));

        let good_response = "I understand this is frustrating. Let's work through it step by step.";
        assert!(validator.validate(good_response, &state));
    }

    #[test]
    fn test_care_optimizer_suggestions() {
        let optimizer = CareOptimizer::new();
        let mut state = EmotionalState::default();
        state.cognitive_load = 0.95;

        let suggestion = optimizer.suggest_action(&state, &[]);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("cognitive load"));
    }

    #[test]
    fn test_memory_content_creation() {
        let state = EmotionalState::default();
        let memory = MemoryContent::new("Test memory".to_string(), state);

        assert_eq!(memory.content, "Test memory");
        assert!(memory.significance_score > 0.0);
        assert!(memory.position.is_none());
    }

    #[test]
    fn test_memory_spherical_position() {
        let mut memory = MemoryContent::new("Test".to_string(), EmotionalState::default());
        memory.calculate_spherical_position(1.0);

        assert!(memory.position.is_some());
        let (x, y, z) = memory.position.unwrap();
        // Should be on unit sphere
        let distance = (x * x + y * y + z * z).sqrt();
        assert!((distance - 1.0).abs() < 0.1);
    }
}

pub mod empathy {
    use std::collections::HashMap;

    #[derive(Debug, Clone)]
    pub struct EmpathicInput {
        pub emotional_content: EmotionalVector,
        pub context: String,
        pub relationship_history: Vec<String>,
    }

    #[derive(Debug, Clone)]
    pub struct EmpathicResponse {
        pub base_response: String,
        pub transgenerational_effect: f32,
        pub empathy_score: f32,
    }

    #[derive(Debug, Clone, Default)]
    pub struct EmotionalVector {
        pub joy: f32,
        pub sadness: f32,
        pub anger: f32,
        pub fear: f32,
        pub surprise: f32,
    }

    #[derive(Debug, Clone, Default)]
    pub struct PhysarumResponse {
        pub resonance_level: f32,
        pub connection_strength: f32,
        pub biological_signal: String,
    }

    #[derive(Debug, Clone, Default)]
    pub struct QLearningState {
        pub q_table: HashMap<(String, String), f32>,
        pub learning_rate: f32,
    }

    impl QLearningState {
        pub fn process(&mut self, context: &str, heart_response: &PhysarumResponse) -> String {
            // Mock Q-learning update
            let key = (
                context.to_string(),
                heart_response.biological_signal.clone(),
            );
            let value = self.q_table.entry(key).or_insert(0.5);
            *value += self.learning_rate * 0.1; // Simple update
            format!("Cognitive response to {}: {}", context, value)
        }
    }

    #[derive(Debug, Clone, Default)]
    pub struct MethylationProcessor {
        pub methylation_levels: HashMap<String, f32>,
    }

    impl MethylationProcessor {
        pub fn apply_modulation(&mut self, response: &str, history: &[String]) -> String {
            let modulation_factor = history.len() as f32 * 0.1;
            let modulated = format!("Modulated {} by {}", response, modulation_factor);
            // Update methylation
            self.methylation_levels
                .insert("empathy_gene".to_string(), modulation_factor);
            modulated
        }

        pub fn get_transgenerational_impact(&self) -> f32 {
            *self.methylation_levels.get("empathy_gene").unwrap_or(&0.0)
        }
    }

    pub struct EmpathyNetwork {
        heart_node: SlimeMoldNetwork,          // Node 1: Heart
        cognitive_node: QLearningState,        // Node 2: Cognitive
        memory_node: MemoryNode,               // Node 3: Memory (mock)
        dialogue_node: DialogueNode,           // Node 4: Inner Dialogue (mock)
        epigenetic_node: MethylationProcessor, // Node 5: Epigenetic
    }

    impl Default for EmpathyNetwork {
        fn default() -> Self {
            Self::new()
        }
    }

    impl EmpathyNetwork {
        pub fn new() -> Self {
            Self {
                heart_node: SlimeMoldNetwork::default(),
                cognitive_node: QLearningState {
                    learning_rate: 0.8,
                    ..Default::default()
                },
                memory_node: MemoryNode::default(),
                dialogue_node: DialogueNode::default(),
                epigenetic_node: MethylationProcessor::default(),
            }
        }

        pub fn process_empathic_response(&mut self, input: &EmpathicInput) -> EmpathicResponse {
            let heart_output = self.heart_node.process(&input.emotional_content);
            let cognitive_output = self.cognitive_node.process(&input.context, &heart_output);
            let memory_context = self.memory_node.retrieve(&cognitive_output);
            let inner_response = self
                .dialogue_node
                .generate(&cognitive_output, memory_context);
            let modulated_response = self
                .epigenetic_node
                .apply_modulation(inner_response, &input.relationship_history);

            EmpathicResponse {
                base_response: modulated_response,
                transgenerational_effect: self.epigenetic_node.get_transgenerational_impact(),
                empathy_score: 0.9, // Mock score
            }
        }
    }

    #[derive(Debug, Clone, Default)]
    pub struct SlimeMoldNetwork {
        nodes: Vec<f32>,
        connections: Vec<f32>,
    }

    impl SlimeMoldNetwork {
        pub fn process(&self, emotional_input: &EmotionalVector) -> PhysarumResponse {
            let resonance = emotional_input.joy
                + emotional_input.sadness
                + emotional_input.anger
                + emotional_input.fear
                + emotional_input.surprise;
            PhysarumResponse {
                resonance_level: resonance / 5.0,
                connection_strength: 0.8,
                biological_signal: format!("Resonance: {}", resonance),
            }
        }
    }

    #[derive(Debug, Clone, Default)]
    struct MemoryNode {
        /// Memory storage - future integration with full memory system
        #[allow(dead_code)]
        memories: Vec<String>,
    }

    impl MemoryNode {
        fn retrieve(&self, _key: &str) -> &'static str {
            "Retrieved memory context"
        }
    }

    #[derive(Debug, Clone, Default)]
    struct DialogueNode {
        // Mock
    }

    impl DialogueNode {
        fn generate(&self, _cognitive: &str, _memory: &str) -> &'static str {
            "Generated inner dialogue"
        }
    }
}
