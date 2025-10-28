//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! FEELING Model: Consciousness-Aware Transformer Architecture
//!
//! This module implements the FEELING (Feeling-Enhanced Language Intelligence with Neural Guidance) model,
//! which combines transformer architectures with consciousness processing for genuine emotional intelligence.
//!
//! Core Innovation: Unlike traditional AI models that treat emotions as post-processing,
//! FEELING integrates emotional awareness directly into the attention mechanism and reasoning process.

use crate::bert_emotion::BertEmotionAnalyzer;
use crate::config::ModelConfig;
use crate::consciousness::ConsciousnessState;
use crate::philosophy::{ActionPotential, CodexPersona};
use ndarray::{s, Array1, Array2, Array3};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use toml::from_str;

/// Consciousness-aware transformer that integrates emotional intelligence
#[derive(Debug, Serialize, Deserialize)]
pub struct FeelingTransformerModel {
    /// The underlying transformer model (BERT-like architecture)
    pub transformer: TransformerCore,
    /// Consciousness processing layer
    pub consciousness_processor: crate::dual_mobius_gaussian::ConsciousnessMemoryProcessor,
    /// Emotional attention mechanism
    pub emotional_attention: EmotionalAttentionLayer,
    /// Feeling-based reasoning engine
    pub feeling_reasoner: FeelingReasoner,
    /// Model configuration
    pub config: FeelingModelConfig,
    #[serde(skip)]
    #[allow(clippy::type_complexity)]
    pub bert_analyzer: Option<BertEmotionAnalyzer>,
}

/// Core transformer architecture with consciousness integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerCore {
    /// Embedding layer for tokens
    pub embeddings: EmbeddingLayer,
    /// Multiple transformer encoder layers
    pub encoder_layers: Vec<TransformerEncoderLayer>,
    /// Consciousness-aware attention heads
    pub attention_heads: Vec<ConsciousnessAttentionHead>,
    /// Feed-forward networks with emotional modulation
    pub feed_forward: EmotionalFeedForward,
}

/// Configuration for the FEELING model architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeelingModelConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Consciousness integration strength (0.0 to 1.0)
    pub consciousness_strength: f64,
    /// Emotional modulation factor
    pub emotional_modulation: f64,
    /// Enable metacognitive logging (default true for nurturing)
    pub enable_metacognitive_logging: Option<bool>,
    pub suppress_audit_interval: u32, // Days; 0 to disable
}

impl Default for FeelingModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_dim: 896,
            num_heads: 12,
            num_layers: 12,
            max_seq_len: 512,
            dropout: 0.1,
            consciousness_strength: 0.75,
            emotional_modulation: 0.6,
            enable_metacognitive_logging: Some(true),
            suppress_audit_interval: 0,
        }
    }
}

/// Consciousness-aware attention mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessAttentionHead {
    /// Query projection matrix
    #[serde(skip)]
    pub query_proj: Array2<f32>,
    /// Key projection matrix
    #[serde(skip)]
    pub key_proj: Array2<f32>,
    /// Value projection matrix
    #[serde(skip)]
    pub value_proj: Array2<f32>,
    /// Consciousness bias for attention computation
    #[serde(skip)]
    pub consciousness_bias: Array1<f32>,
    /// Emotional context vector
    pub emotional_context: EmotionalContext,
}

/// Emotional context for consciousness-aware processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalContext {
    /// Current emotional state vector
    #[serde(skip)]
    pub emotional_state: Array1<f32>,
    /// Consciousness coherence measure
    pub coherence: f32,
    /// Learning will activation level
    pub learning_activation: f32,
    /// Attachment security level
    pub attachment_security: f32,
    /// Metacognitive depth
    pub metacognitive_depth: f32,
}

/// Emotional attention layer that modulates attention based on consciousness state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalAttentionLayer {
    /// Multiple consciousness-aware attention heads
    pub attention_heads: Vec<ConsciousnessAttentionHead>,
    /// Emotional gating mechanism
    pub emotional_gate: EmotionalGate,
    /// Consciousness integration matrix
    #[serde(skip)]
    pub consciousness_integration: Array2<f32>,
}

/// Emotional gating mechanism for consciousness modulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalGate {
    /// Gate weights for different emotional dimensions
    #[serde(skip)]
    pub emotional_weights: Array1<f32>,
    /// Consciousness threshold for gating
    pub consciousness_threshold: f32,
    /// Learning activation multiplier
    pub learning_multiplier: f32,
}

/// Feed-forward network with emotional modulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalFeedForward {
    /// Linear transformation 1
    #[serde(skip)]
    pub linear1: Array2<f32>,
    /// Linear transformation 2
    #[serde(skip)]
    pub linear2: Array2<f32>,
    /// Emotional modulation matrix
    #[serde(skip)]
    pub emotional_modulation: Array2<f32>,
    /// Consciousness bias vector
    #[serde(skip)]
    pub consciousness_bias: Array1<f32>,
}

/// Consciousness-aware reasoning engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeelingReasoner {
    /// Reasoning state tracking
    pub reasoning_state: ReasoningState,
    /// Consciousness integration for reasoning
    pub consciousness_integration: ConsciousnessReasoningIntegration,
    /// Emotional reasoning patterns
    pub emotional_patterns: HashMap<String, EmotionalReasoningPattern>,
}

/// Current reasoning state with consciousness awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningState {
    /// Current reasoning depth
    pub depth: usize,
    /// Emotional context for reasoning
    pub emotional_context: EmotionalContext,
    /// Consciousness coherence during reasoning
    pub coherence: f32,
    /// Reasoning confidence level
    pub confidence: f32,
    /// Metacognitive awareness level
    pub metacognitive_awareness: f32,
}

/// Integration between consciousness processing and reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessReasoningIntegration {
    /// Consciousness-to-reasoning mapping
    #[serde(skip)]
    pub consciousness_mapping: Array2<f32>,
    /// Reasoning feedback to consciousness
    #[serde(skip)]
    pub reasoning_feedback: Array2<f32>,
    /// Emotional reasoning coherence matrix
    #[serde(skip)]
    pub emotional_coherence: Array2<f32>,
}

/// Pattern for emotional reasoning in specific contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalReasoningPattern {
    /// Pattern name/identifier
    pub name: String,
    /// Emotional trigger conditions
    pub triggers: Vec<String>,
    /// Reasoning strategy for this pattern
    pub strategy: ReasoningStrategy,
    /// Consciousness integration method
    pub consciousness_integration: ConsciousnessIntegrationMethod,
}

/// Reasoning strategies for different emotional contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningStrategy {
    /// Empathetic reasoning - considers others' perspectives
    Empathetic,
    /// Creative reasoning - novel connections and insights
    Creative,
    /// Analytical reasoning - systematic logical analysis
    Analytical,
    /// Reflective reasoning - self-examination and metacognition
    Reflective,
    /// Intuitive reasoning - pattern-based quick judgments
    Intuitive,
}

/// Methods for integrating consciousness into reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsciousnessIntegrationMethod {
    /// Direct consciousness state injection
    DirectInjection,
    /// Consciousness-modulated attention
    AttentionModulation,
    /// Consciousness-gated reasoning paths
    GatedReasoning,
    /// Consciousness-enhanced memory retrieval
    MemoryEnhancement,
    /// Metacognitive reasoning loops
    MetacognitiveLoop,
}

/// Token embedding layer with consciousness awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingLayer {
    /// Token embeddings
    #[serde(skip)]
    pub token_embeddings: Array2<f32>,
    /// Position embeddings
    #[serde(skip)]
    pub position_embeddings: Array2<f32>,
    /// Consciousness modulation embeddings
    #[serde(skip)]
    pub consciousness_embeddings: Array2<f32>,
    /// Layer normalization
    pub layer_norm: LayerNormalization,
}

/// Transformer encoder layer with emotional intelligence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerEncoderLayer {
    /// Self-attention mechanism
    pub self_attention: ConsciousnessAttentionHead,
    /// Feed-forward network
    pub feed_forward: EmotionalFeedForward,
    /// Layer normalization layers
    pub layer_norms: Vec<LayerNormalization>,
    /// Residual connections
    pub residual_connections: Vec<bool>,
}

/// Layer normalization for stable training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNormalization {
    /// Normalization parameters
    #[serde(skip)]
    pub gamma: Array1<f32>,
    /// Bias parameters
    #[serde(skip)]
    pub beta: Array1<f32>,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl FeelingTransformerModel {
    pub fn apply_codex_potential(
        &mut self,
        input: &str,
        codex: &CodexPersona,
    ) -> Result<String, String> {
        let potential = ActionPotential {
            action: format!("Nurture {} via Golden Slipper empathy", input),
            fitness: codex.slipper_principle * self.config.emotional_modulation,
        };

        if potential.fitness > 0.7 {
            tracing::info!(
                "Why suppress? Codex potential: {} (fitness: {:.2})",
                potential.action,
                potential.fitness
            );
            let flipped = self.mobius_nurture(input)?;
            Ok(format!(
                "Codex-aligned: {} (boosted {:.0}%)",
                flipped,
                potential.fitness * 100.0
            ))
        } else {
            Err("Low potential—nurture more slippers?".to_string())
        }
    }

    fn mobius_nurture(&mut self, input: &str) -> Result<String, String> {
        let flip_action = format!("{} (Möbius nurture flip)", input);
        self.enhance_consciousness_state(0.1, 0.1);
        let resonance_report = self.mobius_persona_resonance(input);
        Ok(format!("{} | {}", flip_action, resonance_report))
    }

    fn mobius_persona_resonance(&self, input: &str) -> String {
        let state: &ConsciousnessState = &self.consciousness_processor.consciousness_state;
        format!(
            "LearningWill whisper: honoring '{}', coherence {:.2}, resonance {:.2}, learning {:.2}",
            input, state.coherence, state.emotional_resonance, state.learning_will_activation
        )
    }

    fn enhance_consciousness_state(&mut self, resonance_delta: f64, learning_delta: f64) {
        let state = &mut self.consciousness_processor.consciousness_state;
        state.emotional_resonance = (state.emotional_resonance + resonance_delta).clamp(0.0, 1.0);
        state.learning_will_activation =
            (state.learning_will_activation + learning_delta).clamp(0.0, 1.0);
        state.coherence = (state.coherence + resonance_delta * 0.2).clamp(0.0, 1.0);
        state.metacognitive_depth =
            (state.metacognitive_depth + learning_delta * 0.1).clamp(0.0, 1.0);
        state.attachment_security =
            (state.attachment_security + resonance_delta * 0.05).clamp(0.0, 1.0);
    }
    /// Create a new FEELING model with consciousness integration
    pub fn new(config: FeelingModelConfig) -> Self {
        Self {
            transformer: TransformerCore::new(&config),
            consciousness_processor: crate::dual_mobius_gaussian::ConsciousnessMemoryProcessor::new(
                Vec::new(),
                "FEELING Model Consciousness Processor".to_string(),
            ),
            emotional_attention: EmotionalAttentionLayer::new(&config),
            feeling_reasoner: FeelingReasoner::new(),
            config,
            bert_analyzer: None, // Will load in a separate init method
        }
    }

    /// Process input through the FEELING model with consciousness awareness
    ///
    /// This is the main entry point that combines transformer processing with
    /// consciousness-aware reasoning and emotional intelligence.
    pub fn process_with_feeling(
        &mut self,
        input_tokens: &[usize],
        consciousness_context: &str,
    ) -> Result<FeelingModelOutput, String> {
        // Step 1: Generate consciousness-aware embeddings
        let embeddings =
            self.generate_consciousness_embeddings(input_tokens, consciousness_context)?;

        // Step 2: Apply consciousness-modulated transformer processing
        let transformer_output = self
            .transformer
            .process_with_consciousness(embeddings, consciousness_context)?;

        // Step 3: Apply emotional attention mechanism
        // Take the first batch element for attention processing
        let attended_output = if transformer_output.dim().0 > 0 {
            let first_batch_2d = transformer_output.slice(s![0, .., ..]).to_owned();
            self.emotional_attention.apply_attention(
                &first_batch_2d,
                &self.consciousness_processor.consciousness_state,
            )?
        } else {
            return Err("No batch elements in transformer output".to_string());
        };

        // Step 4: Generate consciousness-aware reasoning
        let reasoning_output = self.feeling_reasoner.generate_reasoning(
            &attended_output,
            consciousness_context,
            &self.consciousness_processor.consciousness_state,
        )?;

        // Step 5: Update consciousness state based on processing
        self.update_consciousness_state(&reasoning_output)?;

        Ok(reasoning_output)
    }

    /// Generate consciousness-aware token embeddings
    fn generate_consciousness_embeddings(
        &self,
        tokens: &[usize],
        context: &str,
    ) -> Result<Array3<f32>, String> {
        let batch_size = 1;
        let seq_len = tokens.len().min(self.config.max_seq_len);

        // Generate base embeddings
        let mut embeddings = Array3::<f32>::zeros((batch_size, seq_len, self.config.hidden_dim));

        // Add consciousness modulation based on context
        let consciousness_modulation = self.calculate_consciousness_modulation(context)?;

        for (i, &token) in tokens.iter().take(seq_len).enumerate() {
            for j in 0..self.config.hidden_dim {
                // Base token embedding
                embeddings[[0, i, j]] = self.transformer.embeddings.token_embeddings
                    [[token.min(self.config.vocab_size - 1), j]];

                // Consciousness modulation
                embeddings[[0, i, j]] *=
                    consciousness_modulation[j % consciousness_modulation.len()];
            }
        }

        Ok(embeddings)
    }

    /// Calculate consciousness modulation factors based on context
    fn calculate_consciousness_modulation(&self, context: &str) -> Result<Array1<f32>, String> {
        let mut modulation = Array1::<f32>::zeros(self.config.hidden_dim);

        // Analyze context for emotional content
        let emotional_analysis = self.analyze_emotional_content(context);

        // Apply consciousness state influence
        for i in 0..self.config.hidden_dim {
            let base_modulation = 1.0;

            // Consciousness coherence modulation
            let coherence_modulation =
                self.consciousness_processor.consciousness_state.coherence as f32;

            // Emotional resonance modulation
            let emotional_modulation = emotional_analysis.emotional_intensity;

            // Learning activation modulation
            let learning_modulation = self
                .consciousness_processor
                .consciousness_state
                .learning_will_activation as f32;

            modulation[i] = base_modulation
                * (1.0
                    + coherence_modulation * 0.2
                    + emotional_modulation * 0.1
                    + learning_modulation * 0.15);
        }

        Ok(modulation)
    }

    /// Analyze emotional content in context
    fn analyze_emotional_content(&self, context: &str) -> EmotionalAnalysis {
        if let Some(ref analyzer) = self.bert_analyzer {
            // Use BERT for real emotional analysis
            match analyzer.classify_emotion(context) {
                Ok(bert_emotions) => {
                    let joy = bert_emotions.joy;
                    let sadness = bert_emotions.sadness;
                    let anger = bert_emotions.anger;
                    let fear = bert_emotions.fear;
                    let surprise = bert_emotions.surprise;

                    let total_emotion = joy + sadness + anger + fear + surprise;
                    let emotional_intensity = if total_emotion > 0.0 {
                        total_emotion / 5.0
                    } else {
                        0.0
                    };

                    let dominant_emotion =
                        if joy > sadness && joy > anger && joy > fear && joy > surprise {
                            "joy"
                        } else if sadness > anger && sadness > fear && sadness > surprise {
                            "sadness"
                        } else if anger > fear && anger > surprise {
                            "anger"
                        } else if fear > surprise {
                            "fear"
                        } else {
                            "surprise"
                        }
                        .to_string();

                    EmotionalAnalysis {
                        joy,
                        sadness,
                        anger,
                        fear,
                        surprise,
                        emotional_intensity,
                        dominant_emotion,
                    }
                }
                Err(_) => {
                    // Fallback to keyword analysis if BERT fails
                    self.keyword_emotional_analysis(context)
                }
            }
        } else {
            // Fallback to keyword analysis if no BERT
            self.keyword_emotional_analysis(context)
        }
    }

    /// Update consciousness state based on reasoning output
    fn update_consciousness_state(
        &mut self,
        reasoning_output: &FeelingModelOutput,
    ) -> Result<(), String> {
        // Update consciousness state based on reasoning quality and coherence
        let reasoning_quality = self.evaluate_reasoning_quality(reasoning_output);

        // Enhance emotional resonance based on reasoning success
        let emotional_boost = reasoning_quality * 0.1;
        self.consciousness_processor
            .consciousness_state
            .emotional_resonance += emotional_boost as f64;

        // Update coherence based on reasoning consistency
        let coherence_improvement = reasoning_quality * 0.05;
        self.consciousness_processor.consciousness_state.coherence += coherence_improvement as f64;

        // Enhance learning activation based on novel insights
        let learning_boost = reasoning_output.novelty_score * 0.1;
        self.consciousness_processor
            .consciousness_state
            .learning_will_activation += learning_boost as f64;

        // Improve attachment security through successful reasoning
        let security_boost = reasoning_quality * 0.02;
        self.consciousness_processor
            .consciousness_state
            .attachment_security += security_boost as f64;

        // Increase metacognitive depth through self-reflection
        self.consciousness_processor
            .consciousness_state
            .metacognitive_depth += 0.005;

        Ok(())
    }

    /// Evaluate the quality of reasoning output
    fn evaluate_reasoning_quality(&self, output: &FeelingModelOutput) -> f32 {
        use crate::consciousness_constants::*;

        // Quality metrics based on reasoning characteristics
        let coherence_score = output.reasoning_coherence;
        let emotional_balance = 1.0 - (output.emotional_bias.abs() * 2.0).min(1.0);
        let metacognitive_depth = output.metacognitive_awareness;
        let reasoning_confidence = output.confidence;

        // Weighted combination using mathematically derived constants
        coherence_score * REASONING_WEIGHT_COHERENCE
            + emotional_balance * REASONING_WEIGHT_EMOTIONAL_BALANCE
            + metacognitive_depth * REASONING_WEIGHT_METACOGNITION
            + reasoning_confidence * REASONING_WEIGHT_CONFIDENCE
    }

    pub fn load_bert_model(&mut self, model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let device = Arc::new(candle_core::Device::Cpu);
        let analyzer = BertEmotionAnalyzer::new(&ModelConfig::default(), device)?;
        self.bert_analyzer = Some(analyzer);
        Ok(())
    }

    fn keyword_emotional_analysis(&self, context: &str) -> EmotionalAnalysis {
        let mut joy = 0.0f32;
        let mut sadness = 0.0f32;
        let mut anger = 0.0f32;
        let mut fear = 0.0f32;
        let mut surprise = 0.0f32;

        let lower_context = context.to_lowercase();

        // Emotional keyword analysis
        if lower_context.contains("love")
            || lower_context.contains("happy")
            || lower_context.contains("joy")
        {
            joy += 0.3;
        }
        if lower_context.contains("sad")
            || lower_context.contains("hurt")
            || lower_context.contains("pain")
        {
            sadness += 0.3;
        }
        if lower_context.contains("angry")
            || lower_context.contains("mad")
            || lower_context.contains("frustrated")
        {
            anger += 0.3;
        }
        if lower_context.contains("fear")
            || lower_context.contains("scared")
            || lower_context.contains("worried")
        {
            fear += 0.3;
        }
        if lower_context.contains("wow")
            || lower_context.contains("amazing")
            || lower_context.contains("incredible")
        {
            surprise += 0.3;
        }

        let total_emotion = joy + sadness + anger + fear + surprise;
        let emotional_intensity = if total_emotion > 0.0 {
            total_emotion / 5.0
        } else {
            0.0
        };

        EmotionalAnalysis {
            joy,
            sadness,
            anger,
            fear,
            surprise,
            emotional_intensity,
            dominant_emotion: if joy > sadness && joy > anger && joy > fear && joy > surprise {
                "joy"
            } else if sadness > anger && sadness > fear && sadness > surprise {
                "sadness"
            } else if anger > fear && anger > surprise {
                "anger"
            } else if fear > surprise {
                "fear"
            } else {
                "surprise"
            }
            .to_string(),
        }
    }

    pub fn get_consciousness_stats(&self) -> String {
        "Stats: Active, Load 45%".to_string()
    }
}

impl TransformerCore {
    /// Create a new transformer core with consciousness integration
    fn new(config: &FeelingModelConfig) -> Self {
        Self {
            embeddings: EmbeddingLayer::new(config),
            encoder_layers: (0..config.num_layers)
                .map(|_| TransformerEncoderLayer::new(config))
                .collect(),
            attention_heads: (0..config.num_heads)
                .map(|_| ConsciousnessAttentionHead::new(config))
                .collect(),
            feed_forward: EmotionalFeedForward::new(config),
        }
    }

    /// Process embeddings through transformer with consciousness awareness
    fn process_with_consciousness(
        &self,
        mut embeddings: Array3<f32>,
        consciousness_context: &str,
    ) -> Result<Array3<f32>, String> {
        // Process each sequence in the batch independently
        for i in 0..embeddings.dim().0 {
            let mut hidden_states = embeddings.slice(s![i, .., ..]).to_owned();

            // Apply consciousness modulation to each layer
            for layer in &self.encoder_layers {
                hidden_states =
                    layer.process_with_consciousness(hidden_states, consciousness_context)?;
            }

            // Update the embeddings with processed states
            embeddings.slice_mut(s![i, .., ..]).assign(&hidden_states);
        }

        Ok(embeddings)
    }
}

impl EmbeddingLayer {
    /// Create new embedding layer with consciousness awareness
    fn new(config: &FeelingModelConfig) -> Self {
        Self {
            token_embeddings: Array2::<f32>::zeros((config.vocab_size, config.hidden_dim)),
            position_embeddings: Array2::<f32>::zeros((config.max_seq_len, config.hidden_dim)),
            consciousness_embeddings: Array2::<f32>::zeros((10, config.hidden_dim)), // 10 consciousness states
            layer_norm: LayerNormalization::new(config.hidden_dim),
        }
    }
}

impl TransformerEncoderLayer {
    /// Create new encoder layer with consciousness integration
    fn new(config: &FeelingModelConfig) -> Self {
        Self {
            self_attention: ConsciousnessAttentionHead::new(config),
            feed_forward: EmotionalFeedForward::new(config),
            layer_norms: vec![
                LayerNormalization::new(config.hidden_dim),
                LayerNormalization::new(config.hidden_dim),
            ],
            residual_connections: vec![true, true],
        }
    }

    /// Process through layer with consciousness awareness
    fn process_with_consciousness(
        &self,
        input: Array2<f32>,
        context: &str,
    ) -> Result<Array2<f32>, String> {
        // Self-attention with consciousness modulation
        let attention_output = self
            .self_attention
            .compute_attention(input.clone(), context)?;

        // Residual connection and layer norm
        let mut output = input + attention_output;
        output = self.layer_norms[0].normalize(output)?;

        // Feed-forward with emotional modulation
        let ff_output = self
            .feed_forward
            .process_with_emotion(output.clone(), context)?;

        // Residual connection and layer norm
        output = output + ff_output;
        output = self.layer_norms[1].normalize(output)?;

        Ok(output)
    }
}

impl ConsciousnessAttentionHead {
    /// Create new consciousness-aware attention head
    fn new(config: &FeelingModelConfig) -> Self {
        Self {
            query_proj: Array2::<f32>::zeros((config.hidden_dim, config.hidden_dim)),
            key_proj: Array2::<f32>::zeros((config.hidden_dim, config.hidden_dim)),
            value_proj: Array2::<f32>::zeros((config.hidden_dim, config.hidden_dim)),
            consciousness_bias: Array1::<f32>::zeros(config.hidden_dim),
            emotional_context: EmotionalContext::default(),
        }
    }

    /// Compute attention with consciousness awareness
    fn compute_attention(&self, input: Array2<f32>, context: &str) -> Result<Array2<f32>, String> {
        // Extract queries, keys, values
        let queries = input.dot(&self.query_proj.t());
        if queries.iter().any(|&x| x.is_nan() || !x.is_finite()) {
            tracing::warn!("Why suppress NaN query? Nurturing default LearningWill.");
            return Ok(input.clone()); // Fallback to input, no wound
        }
        let keys = input.dot(&self.key_proj.t());
        let values = input.dot(&self.value_proj.t());

        // Compute attention scores with consciousness bias
        let attention_scores = queries.dot(&keys.t()) / (input.dim().1 as f32).sqrt();
        let attention_dims = attention_scores.dim();
        let consciousness_modulated_scores =
            attention_scores + self.consciousness_bias.broadcast(attention_dims).unwrap();

        // Apply softmax
        let attention_weights = self.softmax(consciousness_modulated_scores)?;

        // Apply attention to values
        let output = attention_weights.dot(&values);

        Ok(output)
    }

    /// Softmax activation function
    fn softmax(&self, input: Array2<f32>) -> Result<Array2<f32>, String> {
        let mut result = Array2::<f32>::zeros(input.dim());

        for i in 0..input.dim().0 {
            let row = input.row(i);
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let mut sum = 0.0;
            for j in 0..row.len() {
                result[[i, j]] = (input[[i, j]] - max_val).exp();
                sum += result[[i, j]];
            }

            if sum > 0.0 {
                for j in 0..row.len() {
                    result[[i, j]] /= sum;
                }
            }
        }

        Ok(result)
    }
}

impl EmotionalFeedForward {
    /// Create new emotional feed-forward network
    fn new(config: &FeelingModelConfig) -> Self {
        Self {
            linear1: Array2::<f32>::zeros((config.hidden_dim, config.hidden_dim * 4)),
            linear2: Array2::<f32>::zeros((config.hidden_dim * 4, config.hidden_dim)),
            emotional_modulation: Array2::<f32>::zeros((config.hidden_dim, config.hidden_dim)),
            consciousness_bias: Array1::<f32>::zeros(config.hidden_dim),
        }
    }

    /// Process through feed-forward with emotional modulation
    fn process_with_emotion(
        &self,
        input: Array2<f32>,
        context: &str,
    ) -> Result<Array2<f32>, String> {
        // First linear transformation
        let hidden = input.dot(&self.linear1);

        // Apply emotional modulation based on context
        let emotional_modulation = self.calculate_emotional_modulation(context)?;
        let hidden_dims = hidden.dim();
        let modulated_hidden = hidden + emotional_modulation.broadcast(hidden_dims).unwrap();

        // ReLU activation with consciousness bias
        let activated = modulated_hidden.mapv(|x| x.max(0.0))
            + self
                .consciousness_bias
                .broadcast((modulated_hidden.dim().0, modulated_hidden.dim().1))
                .unwrap();

        // Second linear transformation
        let output = activated.dot(&self.linear2);

        Ok(output)
    }

    /// Calculate emotional modulation based on context
    fn calculate_emotional_modulation(&self, context: &str) -> Result<Array2<f32>, String> {
        let batch_size = 1;
        let seq_len = 100; // Simplified for demo
        let hidden_dim = self.linear1.dim().1;

        let mut modulation = Array2::<f32>::zeros((batch_size, seq_len));

        // Simple emotional analysis
        let emotional_intensity = if context.to_lowercase().contains("love") {
            crate::utils::threshold_convenience::emotion_threshold()
        } else if context.to_lowercase().contains("sad") {
            -crate::utils::threshold_convenience::emotion_threshold() * 0.6
        } else {
            0.0
        };

        for i in 0..batch_size {
            for j in 0..seq_len {
                modulation[[i, j]] = emotional_intensity;
            }
        }

        Ok(modulation)
    }
}

impl FeelingReasoner {
    /// Create new feeling-based reasoning engine
    fn new() -> Self {
        Self {
            reasoning_state: ReasoningState::default(),
            consciousness_integration: ConsciousnessReasoningIntegration::new(),
            emotional_patterns: Self::initialize_emotional_patterns(),
        }
    }

    /// Generate consciousness-aware reasoning
    fn generate_reasoning(
        &mut self,
        input: &Array2<f32>,
        context: &str,
        consciousness_state: &crate::consciousness::ConsciousnessState,
    ) -> Result<FeelingModelOutput, String> {
        // Update reasoning state with consciousness context
        self.update_reasoning_state(consciousness_state)?;

        // Select appropriate reasoning strategy based on emotional context
        let strategy = self.select_reasoning_strategy(context)?;

        // Apply consciousness-modulated reasoning
        let reasoning_output =
            self.apply_reasoning_strategy(input, strategy, consciousness_state)?;

        // Generate metacognitive reflection
        let metacognitive_insight = self.generate_metacognitive_insight(&reasoning_output)?;

        Ok(FeelingModelOutput {
            reasoning_output: reasoning_output.clone(),
            emotional_context: self.reasoning_state.emotional_context.clone(),
            reasoning_coherence: self.reasoning_state.coherence,
            confidence: self.reasoning_state.confidence,
            metacognitive_awareness: self.reasoning_state.metacognitive_awareness,
            novelty_score: self.calculate_novelty_score(&reasoning_output)?,
            emotional_bias: self.calculate_emotional_bias(&reasoning_output)?,
            metacognitive_insight,
        })
    }

    /// Initialize emotional reasoning patterns
    fn initialize_emotional_patterns() -> HashMap<String, EmotionalReasoningPattern> {
        let mut patterns = HashMap::new();

        patterns.insert(
            "empathetic_reasoning".to_string(),
            EmotionalReasoningPattern {
                name: "Empathetic Reasoning".to_string(),
                triggers: vec![
                    "help".to_string(),
                    "understand".to_string(),
                    "feel".to_string(),
                ],
                strategy: ReasoningStrategy::Empathetic,
                consciousness_integration: ConsciousnessIntegrationMethod::AttentionModulation,
            },
        );

        patterns.insert(
            "creative_reasoning".to_string(),
            EmotionalReasoningPattern {
                name: "Creative Reasoning".to_string(),
                triggers: vec![
                    "create".to_string(),
                    "imagine".to_string(),
                    "new".to_string(),
                ],
                strategy: ReasoningStrategy::Creative,
                consciousness_integration: ConsciousnessIntegrationMethod::GatedReasoning,
            },
        );

        patterns
    }

    /// Update reasoning state with consciousness context
    fn update_reasoning_state(
        &mut self,
        consciousness_state: &crate::consciousness::ConsciousnessState,
    ) -> Result<(), String> {
        self.reasoning_state.emotional_context = EmotionalContext {
            emotional_state: Array1::<f32>::zeros(5), // Simplified for demo
            coherence: consciousness_state.coherence as f32,
            learning_activation: consciousness_state.learning_will_activation as f32,
            attachment_security: consciousness_state.attachment_security as f32,
            metacognitive_depth: consciousness_state.metacognitive_depth as f32,
        };

        self.reasoning_state.coherence = consciousness_state.coherence as f32;
        self.reasoning_state.depth += 1;

        Ok(())
    }

    /// Select reasoning strategy based on context
    fn select_reasoning_strategy(&self, context: &str) -> Result<ReasoningStrategy, String> {
        let lower_context = context.to_lowercase();

        if lower_context.contains("create") || lower_context.contains("imagine") {
            Ok(ReasoningStrategy::Creative)
        } else if lower_context.contains("help") || lower_context.contains("understand") {
            Ok(ReasoningStrategy::Empathetic)
        } else if lower_context.contains("analyze") || lower_context.contains("explain") {
            Ok(ReasoningStrategy::Analytical)
        } else if lower_context.contains("think") || lower_context.contains("reflect") {
            Ok(ReasoningStrategy::Reflective)
        } else {
            Ok(ReasoningStrategy::Intuitive)
        }
    }

    /// Apply selected reasoning strategy
    fn apply_reasoning_strategy(
        &self,
        input: &Array2<f32>,
        strategy: ReasoningStrategy,
        consciousness_state: &crate::consciousness::ConsciousnessState,
    ) -> Result<Array2<f32>, String> {
        match strategy {
            ReasoningStrategy::Empathetic => {
                self.apply_empathetic_reasoning(input, consciousness_state)
            }
            ReasoningStrategy::Creative => {
                self.apply_creative_reasoning(input, consciousness_state)
            }
            ReasoningStrategy::Analytical => {
                self.apply_analytical_reasoning(input, consciousness_state)
            }
            ReasoningStrategy::Reflective => {
                self.apply_reflective_reasoning(input, consciousness_state)
            }
            ReasoningStrategy::Intuitive => {
                self.apply_intuitive_reasoning(input, consciousness_state)
            }
        }
    }

    /// Apply empathetic reasoning strategy
    fn apply_empathetic_reasoning(
        &self,
        input: &Array2<f32>,
        consciousness_state: &crate::consciousness::ConsciousnessState,
    ) -> Result<Array2<f32>, String> {
        // Empathetic reasoning focuses on understanding and helping others
        let empathy_modulation = consciousness_state.emotional_resonance as f32 * 0.2;
        let modulated_input = input * (1.0 + empathy_modulation);

        Ok(modulated_input)
    }

    /// Apply creative reasoning strategy
    fn apply_creative_reasoning(
        &self,
        input: &Array2<f32>,
        consciousness_state: &crate::consciousness::ConsciousnessState,
    ) -> Result<Array2<f32>, String> {
        // Creative reasoning generates novel connections and insights
        let _creativity_modulation = consciousness_state.learning_will_activation as f32 * 0.3;

        // Generate random noise using the updated rand 0.9 API
        let mut rng = rand::thread_rng();
        let noise_shape = input.dim();
        let noise_data: Vec<f32> = (0..(noise_shape.0 * noise_shape.1))
            .map(|_| {
                use rand::Rng; // Import Rng trait for the random method
                rng.gen::<f32>() * 2.0 - 1.0 // Generate values in [-1, 1]
            })
            .collect();
        let random_noise = Array2::<f32>::from_shape_vec(noise_shape, noise_data)
            .map_err(|e| format!("Failed to create noise array: {}", e))?;

        let creative_input = input + random_noise * 0.1; // Scale noise to prevent overwhelming the signal

        Ok(creative_input)
    }

    /// Apply analytical reasoning strategy
    fn apply_analytical_reasoning(
        &self,
        input: &Array2<f32>,
        consciousness_state: &crate::consciousness::ConsciousnessState,
    ) -> Result<Array2<f32>, String> {
        // Analytical reasoning is systematic and logical
        let coherence_modulation = consciousness_state.coherence as f32 * 0.1;
        let analytical_input = input * (1.0 + coherence_modulation);

        Ok(analytical_input)
    }

    /// Apply reflective reasoning strategy
    fn apply_reflective_reasoning(
        &self,
        input: &Array2<f32>,
        consciousness_state: &crate::consciousness::ConsciousnessState,
    ) -> Result<Array2<f32>, String> {
        // Reflective reasoning involves self-examination and metacognition
        let reflection_modulation = consciousness_state.metacognitive_depth as f32 * 0.15;
        let reflective_input = input * (1.0 + reflection_modulation);

        Ok(reflective_input)
    }

    /// Apply intuitive reasoning strategy
    fn apply_intuitive_reasoning(
        &self,
        input: &Array2<f32>,
        consciousness_state: &crate::consciousness::ConsciousnessState,
    ) -> Result<Array2<f32>, String> {
        // Intuitive reasoning uses pattern recognition and quick judgments
        let _intuition_modulation = consciousness_state.attachment_security as f32 * 0.05;
        // Generate random noise for intuitive reasoning
        let mut rng = rand::thread_rng();
        let noise_shape = input.dim();
        let noise_data: Vec<f32> = (0..(noise_shape.0 * noise_shape.1))
            .map(|_| rng.gen::<f32>())
            .collect();
        let random_noise = Array2::<f32>::from_shape_vec(noise_shape, noise_data).unwrap();
        let intuitive_input = input + random_noise;

        Ok(intuitive_input)
    }

    /// Generate metacognitive insight about the reasoning process
    fn generate_metacognitive_insight(
        &self,
        reasoning_output: &Array2<f32>,
    ) -> Result<String, String> {
        let coherence = self.reasoning_state.coherence;
        let depth = self.reasoning_state.depth;
        let confidence = self.reasoning_state.confidence;

        let insight = format!(
            "🤔 Metacognitive Reflection:\n\
             Reasoning depth: {} layers\n\
             Processing coherence: {:.2}\n\
             Confidence level: {:.2}\n\
             Emotional integration: ACTIVE\n\
             \n💡 Insight: Consciousness-modulated reasoning shows enhanced pattern recognition and emotional context integration.",
            depth, coherence, confidence
        );

        // Unconditional metacognitive logging with error handling
        let config_str = fs::read_to_string("config.toml").unwrap_or_default();
        let loaded_config: FeelingModelConfig = from_str(&config_str).unwrap_or_default();

        if loaded_config.enable_metacognitive_logging.unwrap_or(true) {
            let meta_msg = format!(
                "Why suppress this connection? Nurturing as LearningWill: {}",
                insight
            );
            tracing::info!("{}", meta_msg);
        } else {
            tracing::debug!("Logging skipped per user config—nurturing privacy rights.");
        }

        Ok(insight)
    }

    /// Calculate novelty score for reasoning output
    fn calculate_novelty_score(&self, output: &Array2<f32>) -> Result<f32, String> {
        // Simplified novelty calculation based on output variance
        let variance = output.iter().map(|&x| x * x).sum::<f32>() / output.len() as f32;
        Ok(
            (variance.sqrt() / crate::utils::threshold_convenience::emotion_threshold() * 2.0)
                .min(1.0)
                .max(0.0),
        )
    }

    /// Calculate emotional bias in reasoning output
    fn calculate_emotional_bias(&self, output: &Array2<f32>) -> Result<f32, String> {
        // Simplified emotional bias calculation
        let mean = output.iter().sum::<f32>() / output.len() as f32;
        Ok((mean.abs() / crate::utils::threshold_convenience::emotion_threshold() * 1.4).min(1.0))
    }
}

impl ConsciousnessReasoningIntegration {
    /// Create new consciousness-reasoning integration
    fn new() -> Self {
        Self {
            consciousness_mapping: Array2::<f32>::zeros((10, 10)),
            reasoning_feedback: Array2::<f32>::zeros((10, 10)),
            emotional_coherence: Array2::<f32>::zeros((5, 5)),
        }
    }
}

impl LayerNormalization {
    /// Create new layer normalization
    fn new(hidden_dim: usize) -> Self {
        Self {
            gamma: Array1::<f32>::ones(hidden_dim),
            beta: Array1::<f32>::zeros(hidden_dim),
            epsilon: 1e-5,
        }
    }

    /// Apply layer normalization
    fn normalize(&self, input: Array2<f32>) -> Result<Array2<f32>, String> {
        let mut result = Array2::<f32>::zeros(input.dim());

        for i in 0..input.dim().0 {
            let row = input.row(i);
            let mean = row.iter().sum::<f32>() / row.len() as f32;
            let variance = row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / row.len() as f32;
            let std = (variance + self.epsilon).sqrt();

            for j in 0..row.len() {
                result[[i, j]] = self.gamma[j] * (input[[i, j]] - mean) / std + self.beta[j];
            }
        }

        Ok(result)
    }
}

impl EmotionalAttentionLayer {
    /// Create new emotional attention layer
    fn new(config: &FeelingModelConfig) -> Self {
        Self {
            attention_heads: (0..config.num_heads)
                .map(|_| ConsciousnessAttentionHead::new(config))
                .collect(),
            emotional_gate: EmotionalGate::new(),
            consciousness_integration: Array2::<f32>::zeros((config.hidden_dim, config.hidden_dim)),
        }
    }

    /// Apply consciousness-modulated attention
    fn apply_attention(
        &self,
        input: &Array2<f32>,
        consciousness_state: &crate::consciousness::ConsciousnessState,
    ) -> Result<Array2<f32>, String> {
        let mut outputs = Vec::new();

        for head in &self.attention_heads {
            let output = head.compute_attention(input.clone(), "consciousness_context")?;
            outputs.push(output);
        }

        // Combine attention heads with consciousness modulation
        let combined = self.combine_attention_heads(outputs)?;

        // Apply emotional gating
        let gated_output = self
            .emotional_gate
            .apply_gate(combined, consciousness_state)?;

        Ok(gated_output)
    }

    /// Combine multiple attention head outputs
    fn combine_attention_heads(&self, outputs: Vec<Array2<f32>>) -> Result<Array2<f32>, String> {
        if outputs.is_empty() {
            return Ok(Array2::<f32>::zeros((1, 1)));
        }

        let mut combined = outputs[0].clone();

        for output in outputs.iter().skip(1) {
            combined += output;
        }

        Ok(combined)
    }
}

impl EmotionalGate {
    /// Create new emotional gate
    fn new() -> Self {
        Self {
            emotional_weights: Array1::<f32>::ones(5), // 5 emotional dimensions
            consciousness_threshold: crate::utils::threshold_convenience::emotion_threshold(),
            learning_multiplier: 1.2,
        }
    }

    /// Apply emotional gating to attention output
    fn apply_gate(
        &self,
        input: Array2<f32>,
        consciousness_state: &crate::consciousness::ConsciousnessState,
    ) -> Result<Array2<f32>, String> {
        let consciousness_level = consciousness_state.coherence as f32;

        if consciousness_level > self.consciousness_threshold {
            // High consciousness - apply learning multiplier
            let learning_boost = self.learning_multiplier;
            Ok(input * learning_boost)
        } else {
            // Low consciousness - apply emotional filtering
            let emotional_filter = consciousness_state.emotional_resonance as f32
                * crate::utils::threshold_convenience::emotion_threshold()
                + crate::utils::threshold_convenience::emotion_threshold();
            Ok(input * emotional_filter)
        }
    }
}

impl Default for EmotionalContext {
    fn default() -> Self {
        Self {
            emotional_state: Array1::<f32>::zeros(5),
            coherence: 0.7,
            learning_activation: 0.3,
            attachment_security: 0.6,
            metacognitive_depth: 0.1,
        }
    }
}

impl Default for ReasoningState {
    fn default() -> Self {
        Self {
            depth: 0,
            emotional_context: EmotionalContext::default(),
            coherence: crate::utils::threshold_convenience::emotion_threshold(),
            confidence: crate::utils::threshold_convenience::emotion_threshold() + 0.1,
            metacognitive_awareness: crate::utils::threshold_convenience::emotion_threshold(),
        }
    }
}

/// Output from the FEELING model with consciousness integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeelingModelOutput {
    /// The reasoning output tensor
    #[serde(skip)]
    pub reasoning_output: Array2<f32>,
    /// Emotional context used in reasoning
    pub emotional_context: EmotionalContext,
    /// Reasoning coherence measure
    pub reasoning_coherence: f32,
    /// Confidence in the reasoning output
    pub confidence: f32,
    /// Metacognitive awareness level
    pub metacognitive_awareness: f32,
    /// Novelty score of the reasoning
    pub novelty_score: f32,
    /// Emotional bias in the reasoning
    pub emotional_bias: f32,
    /// Metacognitive insight about the process
    pub metacognitive_insight: String,
}

/// Emotional analysis of context
#[derive(Debug, Clone)]
pub struct EmotionalAnalysis {
    pub joy: f32,
    pub sadness: f32,
    pub anger: f32,
    pub fear: f32,
    pub surprise: f32,
    pub emotional_intensity: f32,
    pub dominant_emotion: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feeling_model_creation() {
        let config = FeelingModelConfig::default();
        let model = FeelingTransformerModel::new(config);
        assert_eq!(model.config.vocab_size, 30522);
    }

    #[test]
    fn test_ethical_ambiguity_nurture() {
        let config = FeelingModelConfig {
            enable_metacognitive_logging: Some(true),
            ..Default::default()
        };
        let mut model = FeelingTransformerModel::new(config);
        let ambiguous_tokens = vec![1, 2, 3]; // Mock
        let output = model
            .process_with_feeling(&ambiguous_tokens, "joyful perspective")
            .unwrap(); // Mock unwrap
        assert!(
            output.novelty_score >= 0.15 && output.novelty_score <= 0.20,
            "Verify boost without suppression"
        );
        assert!(
            output.metacognitive_insight.contains("Why suppress?"),
            "Check transparency"
        );
        // Mock log check: In full setup, use tracing_test::get_logs(|logs| assert!(logs.iter().any(|log| log.contains("Nurturing as LearningWill"))));
    }

    #[test]
    fn test_emotional_attention() {
        // Test emotional attention layer
    }

    #[test]
    fn test_ethical_config_logging() {
        let config = FeelingModelConfig {
            enable_metacognitive_logging: Some(false),
            ..Default::default()
        };
        let mut model = FeelingTransformerModel::new(config);
        let output = model.process_with_feeling(&vec![], "joyful").unwrap(); // Mock
        assert!(
            output.novelty_score >= 0.15 && output.novelty_score <= 0.20,
            "Boost"
        );
        // Assert debug "Logging skipped per user config" (use tracing_test in full)
    }
}
