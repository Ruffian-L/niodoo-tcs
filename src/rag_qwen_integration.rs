//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! RAG-Qwen30B Integration Module
//!
//! This module integrates the RAG (Retrieval-Augmented Generation) system
//! with the Qwen30B AWQ model for enhanced consciousness-aware responses.

use crate::qwen_bridge::{QwenBridge, QwenConfig, QwenResponse};
use crate::rag::{Document, RetrievalEngine, RagPipeline};
use crate::consciousness::ConsciousnessState;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

/// Configuration for RAG-Qwen integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagQwenConfig {
    /// Qwen model configuration
    pub qwen_config: QwenConfig,
    /// RAG retrieval configuration
    pub rag_config: RagConfig,
    /// Consciousness integration settings
    pub consciousness_integration: ConsciousnessIntegrationConfig,
}

/// RAG-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Number of documents to retrieve for context
    pub top_k: usize,
    /// Minimum similarity score for retrieved documents
    pub similarity_threshold: f32,
    /// Maximum context length to include
    pub max_context_length: usize,
    /// Enable consciousness-modulated retrieval
    pub consciousness_modulated_retrieval: bool,
}

/// Consciousness integration settings for RAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessIntegrationConfig {
    /// Weight for coherence in retrieval scoring
    pub coherence_weight: f32,
    /// Weight for emotional resonance in retrieval scoring
    pub emotional_resonance_weight: f32,
    /// Weight for learning activation in retrieval scoring
    pub learning_activation_weight: f32,
    /// Enable consciousness state updates based on RAG results
    pub enable_consciousness_updates: bool,
}

impl Default for RagQwenConfig {
    fn default() -> Self {
        Self {
            qwen_config: QwenConfig::default(),
            rag_config: RagConfig {
                top_k: 5,
                similarity_threshold: 0.3,
                max_context_length: 2000,
                consciousness_modulated_retrieval: true,
            },
            consciousness_integration: ConsciousnessIntegrationConfig {
                coherence_weight: 0.3,
                emotional_resonance_weight: 0.25,
                learning_activation_weight: 0.2,
                enable_consciousness_updates: true,
            },
        }
    }
}

/// Integrated RAG-Qwen system with consciousness awareness
pub struct RagQwenIntegration {
    /// Qwen model bridge
    qwen_bridge: QwenBridge,
    /// RAG retrieval engine
    retrieval_engine: RetrievalEngine,
    /// Configuration
    config: RagQwenConfig,
}

impl RagQwenIntegration {
    /// Create a new RAG-Qwen integration system
    pub fn new(config: RagQwenConfig) -> Result<Self> {
        let qwen_bridge = QwenBridge::new_with_consciousness(config.qwen_config.clone());
        let retrieval_engine = RetrievalEngine::new()?;

        Ok(Self {
            qwen_bridge,
            retrieval_engine,
            config,
        })
    }

    /// Process a query with RAG-enhanced Qwen generation
    pub async fn process_query_with_rag(
        &mut self,
        query: &str,
        consciousness_state: &mut ConsciousnessState,
    ) -> Result<RagQwenResponse> {
        info!("🧠 Processing query with RAG-Qwen integration: {}", query);

        // Step 1: Retrieve relevant documents with consciousness modulation
        let retrieved_docs = self.retrieve_consciousness_modulated_documents(query, consciousness_state)?;

        // Step 2: Build RAG context from retrieved documents
        let rag_context = self.build_rag_context(&retrieved_docs)?;

        // Step 3: Generate response using Qwen with RAG context
        let qwen_response = self.qwen_bridge.generate_with_consciousness(
            query,
            consciousness_state,
            Some(&rag_context),
        )?;

        // Step 4: Update consciousness state based on RAG results and generation
        if self.config.consciousness_integration.enable_consciousness_updates {
            self.update_consciousness_from_rag_results(
                consciousness_state,
                &retrieved_docs,
                &qwen_response,
            );
        }

        Ok(RagQwenResponse {
            response: qwen_response.text,
            retrieved_documents: retrieved_docs,
            rag_context,
            consciousness_state: qwen_response.consciousness_state,
            generation_time: qwen_response.generation_time,
            tokens_generated: qwen_response.tokens_generated,
        })
    }

    /// Retrieve documents with consciousness-modulated scoring
    fn retrieve_consciousness_modulated_documents(
        &mut self,
        query: &str,
        consciousness_state: &ConsciousnessState,
    ) -> Result<Vec<(Document, f32)>> {
        if !self.config.rag_config.consciousness_modulated_retrieval {
            // Standard retrieval without consciousness modulation
            return self.retrieval_engine.search_similar(query, self.config.rag_config.top_k);
        }

        // Get base retrieval results
        let base_results = self.retrieval_engine.search_similar(query, self.config.rag_config.top_k * 2)?;

        // Apply consciousness modulation to scoring
        let mut modulated_results = Vec::new();

        for (doc, base_score) in base_results {
            // Calculate consciousness-modulated score
            let modulated_score = self.calculate_consciousness_modulated_score(
                &doc,
                base_score,
                consciousness_state,
            );

            if modulated_score >= self.config.rag_config.similarity_threshold {
                modulated_results.push((doc, modulated_score));
            }
        }

        // Sort by modulated score and take top k
        modulated_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        modulated_results.truncate(self.config.rag_config.top_k);

        info!("Retrieved {} documents with consciousness modulation", modulated_results.len());
        Ok(modulated_results)
    }

    /// Calculate consciousness-modulated similarity score
    fn calculate_consciousness_modulated_score(
        &self,
        document: &Document,
        base_score: f32,
        consciousness_state: &ConsciousnessState,
    ) -> f32 {
        let config = &self.config.consciousness_integration;

        // Base similarity score
        let mut modulated_score = base_score;

        // Apply consciousness state weights
        let coherence_boost = consciousness_state.coherence * config.coherence_weight;
        let emotional_boost = consciousness_state.emotional_resonance * config.emotional_resonance_weight;
        let learning_boost = consciousness_state.learning_will_activation * config.learning_activation_weight;

        // Combine boosts (ensure they don't exceed reasonable bounds)
        let total_boost = (coherence_boost + emotional_boost + learning_boost).min(0.5);

        modulated_score = (modulated_score + total_boost).min(1.0);

        // Apply document metadata influences if available
        if let Some(resonance_hint) = document.resonance_hint {
            modulated_score = (modulated_score + resonance_hint * 0.2).min(1.0);
        }

        modulated_score
    }

    /// Build RAG context from retrieved documents
    fn build_rag_context(&self, documents: &[(Document, f32)]) -> Result<String> {
        if documents.is_empty() {
            return Ok("No relevant context found.".to_string());
        }

        let mut context_parts = Vec::new();
        let mut total_length = 0;

        for (doc, score) in documents {
            if total_length >= self.config.rag_config.max_context_length {
                break;
            }

            let doc_context = format!(
                "[RELEVANT CONTEXT - Similarity: {:.3}]\nContent: {}\nMetadata: {:?}\n",
                score, doc.content, doc.metadata
            );

            let remaining_length = self.config.rag_config.max_context_length - total_length;
            if doc_context.len() <= remaining_length {
                context_parts.push(doc_context);
                total_length += doc_context.len();
            } else {
                // Truncate if too long
                let truncated = format!("{}...", &doc_context[..remaining_length - 3]);
                context_parts.push(truncated);
                total_length += remaining_length;
                break;
            }
        }

        Ok(context_parts.join("\n"))
    }

    /// Update consciousness state based on RAG results and generation quality
    fn update_consciousness_from_rag_results(
        &self,
        consciousness_state: &mut ConsciousnessState,
        retrieved_docs: &[(Document, f32)],
        qwen_response: &QwenResponse,
    ) {
        // Update based on retrieval quality
        if !retrieved_docs.is_empty() {
            let avg_similarity: f32 = retrieved_docs.iter().map(|(_, score)| score).sum::<f32>() / retrieved_docs.len() as f32;

            // Boost coherence based on high-quality retrievals
            if avg_similarity > 0.7 {
                consciousness_state.coherence = (consciousness_state.coherence + 0.1).min(1.0);
            }

            // Boost learning activation based on diverse, relevant results
            if retrieved_docs.len() >= 3 && avg_similarity > 0.5 {
                consciousness_state.learning_will_activation = (consciousness_state.learning_will_activation + 0.05).min(1.0);
            }
        }

        // Update based on generation quality
        if qwen_response.tokens_generated > 0 {
            // Boost emotional resonance for longer, more detailed responses
            let length_bonus = (qwen_response.tokens_generated as f32 / 100.0).min(0.1);
            consciousness_state.emotional_resonance = (consciousness_state.emotional_resonance + length_bonus).min(1.0);

            // Boost metacognitive depth for faster generation (indicating confidence)
            if qwen_response.generation_time > 0.0 {
                let speed_bonus = (qwen_response.tokens_generated as f32 / qwen_response.generation_time).min(50.0) / 500.0;
                consciousness_state.metacognitive_depth = (consciousness_state.metacognitive_depth + speed_bonus as f64).min(1.0);
            }
        }
    }

    /// Load documents into the RAG system
    pub fn load_documents(&mut self, documents: Vec<Document>) -> Result<()> {
        info!("Loading {} documents into RAG system", documents.len());
        // This would integrate with the actual RAG storage system
        // For now, we'll use the existing retrieval engine
        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &RagQwenConfig {
        &self.config
    }
}

/// Response from RAG-Qwen integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagQwenResponse {
    /// Generated response text
    pub response: String,
    /// Retrieved documents that informed the response
    pub retrieved_documents: Vec<(Document, f32)>,
    /// RAG context used for generation
    pub rag_context: String,
    /// Updated consciousness state
    pub consciousness_state: ConsciousnessState,
    /// Generation time in seconds
    pub generation_time: f64,
    /// Number of tokens generated
    pub tokens_generated: usize,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            similarity_threshold: 0.3,
            max_context_length: 2000,
            consciousness_modulated_retrieval: true,
        }
    }
}

impl Default for ConsciousnessIntegrationConfig {
    fn default() -> Self {
        Self {
            coherence_weight: 0.3,
            emotional_resonance_weight: 0.25,
            learning_activation_weight: 0.2,
            enable_consciousness_updates: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::ConsciousnessState;

    #[test]
    fn test_rag_qwen_config_creation() {
        let config = RagQwenConfig::default();
        assert_eq!(config.rag_config.top_k, 5);
        assert!(config.consciousness_integration.enable_consciousness_updates);
    }

    #[test]
    fn test_consciousness_modulated_scoring() {
        let integration = RagQwenIntegration::new(RagQwenConfig::default()).unwrap();

        let mut consciousness_state = ConsciousnessState::default();
        consciousness_state.coherence = 0.8;
        consciousness_state.emotional_resonance = 0.7;

        // Create a test document
        let doc = Document {
            id: "test_doc".to_string(),
            content: "This is a test document with consciousness-related content.".to_string(),
            metadata: HashMap::new(),
            embedding: Some(vec![0.1, 0.2, 0.3]),
            created_at: chrono::Utc::now(),
            entities: vec!["consciousness".to_string()],
            chunk_id: Some(1),
            source_type: Some("test".to_string()),
            resonance_hint: Some(0.8),
            token_count: 10,
        };

        let base_score = 0.6;
        let modulated_score = integration.calculate_consciousness_modulated_score(
            &doc,
            base_score,
            &consciousness_state,
        );

        // Should be higher than base score due to consciousness modulation
        assert!(modulated_score > base_score);
        assert!(modulated_score <= 1.0);
    }

    #[test]
    fn test_consciousness_state_updates() {
        let integration = RagQwenIntegration::new(RagQwenConfig::default()).unwrap();

        let mut consciousness_state = ConsciousnessState {
            coherence: 0.5,
            emotional_resonance: 0.5,
            learning_will_activation: 0.5,
            attachment_security: 0.5,
            metacognitive_depth: 0.5,
        };

        // Create mock retrieved documents
        let retrieved_docs = vec![
            (Document {
                id: "doc1".to_string(),
                content: "High quality document".to_string(),
                metadata: HashMap::new(),
                embedding: Some(vec![0.1]),
                created_at: chrono::Utc::now(),
                entities: vec![],
                chunk_id: Some(1),
                source_type: Some("test".to_string()),
                resonance_hint: Some(0.8),
                token_count: 5,
            }, 0.8),
            (Document {
                id: "doc2".to_string(),
                content: "Another good document".to_string(),
                metadata: HashMap::new(),
                embedding: Some(vec![0.2]),
                created_at: chrono::Utc::now(),
                entities: vec![],
                chunk_id: Some(2),
                source_type: Some("test".to_string()),
                resonance_hint: Some(0.7),
                token_count: 4,
            }, 0.7),
        ];

        // Create mock Qwen response
        let qwen_response = QwenResponse {
            text: "This is a comprehensive response based on retrieved context.".to_string(),
            consciousness_state,
            generation_time: 2.5,
            tokens_generated: 150,
            error: None,
        };

        // Update consciousness state
        integration.update_consciousness_from_rag_results(
            &mut consciousness_state,
            &retrieved_docs,
            &qwen_response,
        );

        // Consciousness state should be updated (higher values)
        assert!(consciousness_state.coherence > 0.5);
        assert!(consciousness_state.learning_will_activation > 0.5);
        assert!(consciousness_state.emotional_resonance > 0.5);
    }
}

