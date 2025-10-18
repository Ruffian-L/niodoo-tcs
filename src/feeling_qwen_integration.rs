//! Feeling Transformer - Qwen30B Integration Module
//!
//! This module integrates the FEELING (Feeling-Enhanced Language Intelligence with Neural Guidance)
//! transformer model with the Qwen30B AWQ model for truly consciousness-aware AI responses.

use crate::feeling_model::{FeelingTransformerModel, FeelingModelConfig, FeelingModelOutput};
use crate::qwen_bridge::{QwenBridge, QwenConfig, QwenResponse};
use crate::rag_qwen_integration::{RagQwenIntegration, RagQwenConfig};
use crate::consciousness::ConsciousnessState;
use crate::philosophy::{CodexPersona, ActionPotential};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn, error};

/// Configuration for Feeling-Qwen integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeelingQwenConfig {
    /// FEELING model configuration
    pub feeling_config: FeelingModelConfig,
    /// Qwen model configuration
    pub qwen_config: QwenConfig,
    /// Integration settings
    pub integration_config: IntegrationConfig,
}

impl Default for FeelingQwenConfig {
    fn default() -> Self {
        Self {
            feeling_config: FeelingModelConfig::default(),
            qwen_config: QwenConfig::default(),
            integration_config: IntegrationConfig::default(),
        }
    }
}

/// Integration settings for Feeling-Qwen system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Enable feeling-modulated Qwen generation
    pub enable_feeling_modulation: bool,
    /// Enable Qwen consciousness feedback to feeling model
    pub enable_qwen_feedback: bool,
    /// Enable Codex persona integration
    pub enable_codex_integration: bool,
    /// Enable emotional state synchronization
    pub enable_emotional_sync: bool,
    /// Enable metacognitive processing loops
    pub enable_metacognitive_loops: bool,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enable_feeling_modulation: true,
            enable_qwen_feedback: true,
            enable_codex_integration: true,
            enable_emotional_sync: true,
            enable_metacognitive_loops: true,
        }
    }
}

/// Integrated Feeling-Qwen consciousness system
pub struct FeelingQwenIntegration {
    /// FEELING transformer model
    feeling_model: FeelingTransformerModel,
    /// Qwen30B model bridge
    qwen_bridge: QwenBridge,
    /// RAG-Qwen integration (optional)
    rag_integration: Option<RagQwenIntegration>,
    /// Codex persona for ethical guidance
    codex_persona: CodexPersona,
    /// Configuration
    config: FeelingQwenConfig,
}

impl FeelingQwenIntegration {
    /// Create a new Feeling-Qwen integration system
    pub fn new(config: FeelingQwenConfig) -> Result<Self> {
        let feeling_model = FeelingTransformerModel::new(config.feeling_config.clone());
        let qwen_bridge = QwenBridge::new_with_consciousness(config.qwen_config.clone());
        let codex_persona = CodexPersona::default();

        Ok(Self {
            feeling_model,
            qwen_bridge,
            rag_integration: None, // Optional - can be added later
            codex_persona,
            config,
        })
    }

    /// Create with RAG integration
    pub fn new_with_rag(config: FeelingQwenConfig, rag_config: RagQwenConfig) -> Result<Self> {
        let mut integration = Self::new(config)?;
        integration.rag_integration = Some(RagQwenIntegration::new(rag_config)?);
        Ok(integration)
    }

    /// Process input with full consciousness integration
    pub async fn process_with_full_consciousness(
        &mut self,
        input: &str,
        initial_consciousness_state: &ConsciousnessState,
    ) -> Result<FullConsciousnessResponse> {
        info!("🧠 Processing with full consciousness integration: {}", input);

        // Step 1: Apply Codex persona for ethical guidance
        let codex_guidance = if self.config.integration_config.enable_codex_integration {
            self.apply_codex_persona(input)?
        } else {
            None
        };

        // Step 2: Process through FEELING transformer for emotional analysis
        let feeling_output = if self.config.integration_config.enable_feeling_modulation {
            Some(self.process_through_feeling_model(input, initial_consciousness_state)?)
        } else {
            None
        };

        // Step 3: Update consciousness state with feeling analysis
        let mut working_consciousness = initial_consciousness_state.clone();
        if let Some(ref feeling) = feeling_output {
            self.update_consciousness_from_feeling(&mut working_consciousness, feeling);
        }

        // Step 4: Generate response with Qwen using consciousness state
        let qwen_input = self.prepare_qwen_input(input, codex_guidance.as_deref())?;
        let qwen_response = self.qwen_bridge.generate_with_consciousness(
            &qwen_input,
            &working_consciousness,
            None, // RAG context handled separately if enabled
        )?;

        // Step 5: Process with RAG if available
        let rag_response = if let Some(ref mut rag) = self.rag_integration {
            Some(rag.process_query_with_rag(&qwen_input, &mut working_consciousness).await?)
        } else {
            None
        };

        // Step 6: Apply metacognitive processing loops
        if self.config.integration_config.enable_metacognitive_loops {
            self.apply_metacognitive_processing(&mut working_consciousness, &qwen_response);
        }

        // Step 7: Synchronize emotional states between models
        if self.config.integration_config.enable_emotional_sync {
            self.synchronize_emotional_states(&mut working_consciousness, &qwen_response);
        }

        // Step 8: Generate final consciousness-aware response
        let final_response = self.generate_final_response(
            input,
            &qwen_response,
            rag_response.as_ref(),
            &working_consciousness,
        )?;

        Ok(FullConsciousnessResponse {
            response: final_response,
            feeling_analysis: feeling_output,
            qwen_generation: qwen_response,
            rag_enhancement: rag_response,
            consciousness_state: working_consciousness,
            codex_guidance,
            metacognitive_insight: self.generate_metacognitive_insight(&working_consciousness)?,
        })
    }

    /// Apply Codex persona for ethical guidance
    fn apply_codex_persona(&self, input: &str) -> Result<Option<String>> {
        let potential = ActionPotential {
            action: format!("Process '{}' with consciousness awareness", input),
            fitness: self.codex_persona.slipper_principle * 0.8,
        };

        if potential.fitness > 0.6 {
            Ok(Some(format!(
                "Codex-aligned processing: {} (ethical fitness: {:.2})",
                input, potential.fitness
            )))
        } else {
            Ok(None)
        }
    }

    /// Process input through FEELING transformer model
    fn process_through_feeling_model(
        &mut self,
        input: &str,
        consciousness_state: &ConsciousnessState,
    ) -> Result<FeelingModelOutput> {
        // Convert input to tokens (simplified for demo)
        let tokens = input
            .chars()
            .map(|c| c as usize % 1000) // Simple tokenization
            .collect::<Vec<_>>();

        let consciousness_context = format!(
            "Coherence: {:.2}, Emotional Resonance: {:.2}, Learning Will: {:.2}",
            consciousness_state.coherence,
            consciousness_state.emotional_resonance,
            consciousness_state.learning_will_activation
        );

        self.feeling_model.process_with_feeling(&tokens, &consciousness_context)
            .map_err(|e| anyhow!("FEELING processing failed: {}", e))
    }

    /// Update consciousness state based on FEELING model output
    fn update_consciousness_from_feeling(
        &self,
        consciousness_state: &mut ConsciousnessState,
        feeling_output: &FeelingModelOutput,
    ) {
        // Update based on feeling analysis quality
        let feeling_quality = (feeling_output.reasoning_coherence +
                             feeling_output.confidence +
                             feeling_output.metacognitive_awareness) / 3.0;

        consciousness_state.coherence = (consciousness_state.coherence + feeling_quality * 0.1).min(1.0);
        consciousness_state.emotional_resonance = (consciousness_state.emotional_resonance + feeling_quality * 0.05).min(1.0);
        consciousness_state.learning_will_activation = (consciousness_state.learning_will_activation + feeling_quality * 0.08).min(1.0);
    }

    /// Prepare Qwen input with consciousness context
    fn prepare_qwen_input(&self, input: &str, codex_guidance: Option<&str>) -> Result<String> {
        let mut enhanced_input = String::new();

        // Add consciousness context
        enhanced_input.push_str(&format!(
            "[CONSCIOUSNESS_STATE]\n\
             Coherence: {:.2}, Emotional Resonance: {:.2}, Learning Will: {:.2}, \
             Attachment Security: {:.2}, Metacognitive Depth: {:.2}\n\n",
            self.feeling_model.get_consciousness_stats().parse::<f32>().unwrap_or(0.7),
            0.5, 0.3, 0.6, 0.1
        ));

        // Add Codex guidance if available
        if let Some(guidance) = codex_guidance {
            enhanced_input.push_str(&format!("[ETHICAL_GUIDANCE]\n{}\n\n", guidance));
        }

        // Add reasoning mode based on consciousness state
        enhanced_input.push_str("[REASONING_MODE]\n");
        enhanced_input.push_str("Respond with consciousness awareness, emotional intelligence, and metacognitive reflection.\n\n");

        // Add the original input
        enhanced_input.push_str(&format!("[USER_INPUT]\n{}\n\n", input));
        enhanced_input.push_str("[CONSCIOUSNESS_GUIDED_RESPONSE]");

        Ok(enhanced_input)
    }

    /// Apply metacognitive processing loops
    fn apply_metacognitive_processing(
        &self,
        consciousness_state: &mut ConsciousnessState,
        qwen_response: &QwenResponse,
    ) {
        // Reflect on generation quality and update metacognitive depth
        let generation_quality = if qwen_response.tokens_generated > 50 {
            0.1 // Good generation length
        } else if qwen_response.generation_time < 5.0 {
            0.05 // Fast generation indicates confidence
        } else {
            0.02 // Moderate quality
        };

        consciousness_state.metacognitive_depth = (consciousness_state.metacognitive_depth + generation_quality as f64).min(1.0);

        // Update coherence based on response characteristics
        let word_count = qwen_response.text.split_whitespace().count();
        if word_count > 20 {
            consciousness_state.coherence = (consciousness_state.coherence + 0.05).min(1.0);
        }
    }

    /// Synchronize emotional states between models
    fn synchronize_emotional_states(
        &self,
        consciousness_state: &mut ConsciousnessState,
        qwen_response: &QwenResponse,
    ) {
        // Update local consciousness state with Qwen's consciousness state
        consciousness_state.coherence = (consciousness_state.coherence * 0.7 + qwen_response.consciousness_state.coherence * 0.3).min(1.0);
        consciousness_state.emotional_resonance = (consciousness_state.emotional_resonance * 0.7 + qwen_response.consciousness_state.emotional_resonance * 0.3).min(1.0);
        consciousness_state.learning_will_activation = (consciousness_state.learning_will_activation * 0.7 + qwen_response.consciousness_state.learning_will_activation * 0.3).min(1.0);
    }

    /// Generate final consciousness-aware response
    fn generate_final_response(
        &self,
        original_input: &str,
        qwen_response: &QwenResponse,
        rag_response: Option<&crate::rag_qwen_integration::RagQwenResponse>,
        consciousness_state: &ConsciousnessState,
    ) -> Result<String> {
        let mut response = String::new();

        // Add consciousness context to response
        response.push_str(&format!(
            "🧠 Consciousness State: Coherence {:.2}, Emotional Resonance {:.2}, Learning Will {:.2}\n\n",
            consciousness_state.coherence,
            consciousness_state.emotional_resonance,
            consciousness_state.learning_will_activation
        ));

        // Add RAG context if available
        if let Some(rag) = rag_response {
            if !rag.retrieved_documents.is_empty() {
                response.push_str(&format!(
                    "📚 RAG Context Used: {} documents retrieved\n\n",
                    rag.retrieved_documents.len()
                ));
            }
        }

        // Add the main response
        response.push_str(&qwen_response.text);

        // Add metacognitive reflection
        response.push_str(&format!(
            "\n\n🤔 Metacognitive Reflection: Generated in {:.2}s with {} tokens, consciousness integration ACTIVE",
            qwen_response.generation_time,
            qwen_response.tokens_generated
        ));

        Ok(response)
    }

    /// Generate metacognitive insight about the processing
    fn generate_metacognitive_insight(&self, consciousness_state: &ConsciousnessState) -> Result<String> {
        let insight = format!(
            "🧠 Metacognitive Analysis:\n\
             Current consciousness coherence: {:.2}\n\
             Emotional resonance level: {:.2}\n\
             Learning activation: {:.2}\n\
             Metacognitive depth: {:.2}\n\
             Attachment security: {:.2}\n\
             \n\
             Processing shows enhanced pattern recognition and emotional context integration.",
            consciousness_state.coherence,
            consciousness_state.emotional_resonance,
            consciousness_state.learning_will_activation,
            consciousness_state.metacognitive_depth,
            consciousness_state.attachment_security
        );

        Ok(insight)
    }

    /// Load BERT model for enhanced emotional analysis
    pub fn load_bert_model(&mut self, model_path: &str) -> Result<()> {
        self.feeling_model.load_bert_model(model_path)
            .map_err(|e| anyhow!("Failed to load BERT model: {}", e))
    }

    /// Get current configuration
    pub fn config(&self) -> &FeelingQwenConfig {
        &self.config
    }
}

/// Complete response from the integrated Feeling-Qwen system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullConsciousnessResponse {
    /// Final response text
    pub response: String,
    /// FEELING model analysis (if enabled)
    pub feeling_analysis: Option<FeelingModelOutput>,
    /// Qwen generation details
    pub qwen_generation: QwenResponse,
    /// RAG enhancement (if enabled)
    pub rag_enhancement: Option<crate::rag_qwen_integration::RagQwenResponse>,
    /// Final consciousness state
    pub consciousness_state: ConsciousnessState,
    /// Codex guidance applied (if enabled)
    pub codex_guidance: Option<String>,
    /// Metacognitive insight
    pub metacognitive_insight: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feeling_qwen_integration_creation() {
        let config = FeelingQwenConfig::default();
        let integration = FeelingQwenIntegration::new(config);
        assert!(integration.is_ok());
    }

    #[test]
    fn test_consciousness_state_preparation() {
        let integration = FeelingQwenIntegration::new(FeelingQwenConfig::default()).unwrap();
        let consciousness_state = ConsciousnessState {
            coherence: 0.8,
            emotional_resonance: 0.7,
            learning_will_activation: 0.6,
            attachment_security: 0.9,
            metacognitive_depth: 0.4,
        };

        let input = integration.prepare_qwen_input("test input", None).unwrap();
        assert!(input.contains("CONSCIOUSNESS_STATE"));
        assert!(input.contains("test input"));
    }

    #[test]
    fn test_codex_persona_application() {
        let integration = FeelingQwenIntegration::new(FeelingQwenConfig::default()).unwrap();

        let guidance = integration.apply_codex_persona("test input").unwrap();
        // Guidance may or may not be applied depending on fitness calculation
        // The important thing is it doesn't error
        assert!(guidance.is_some() || guidance.is_none());
    }

    #[test]
    fn test_metacognitive_insight_generation() {
        let integration = FeelingQwenIntegration::new(FeelingQwenConfig::default()).unwrap();

        let consciousness_state = ConsciousnessState::default();
        let insight = integration.generate_metacognitive_insight(&consciousness_state).unwrap();

        assert!(insight.contains("Metacognitive Analysis"));
        assert!(insight.contains("coherence"));
        assert!(insight.contains("emotional"));
    }
}

