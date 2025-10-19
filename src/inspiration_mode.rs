//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🌟 Niodo Inspiration Mode - RAG-Integrated Response System
 * 
 * This module provides "inspiration mode" where Niodo's responses cite
 * consciousness research from the knowledge base with a snarky twist.
 * Ensures responses reference guides without hallucinating overreaches.
 * 
 * Example: "Leveling empathy like your Gemini neurobiology notes—snarky twist!"
 */

use std::collections::HashMap;
use std::env;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use tracing::{info, debug, warn};
use crate::consciousness::EmotionType;
use crate::rag::retrieval::RetrievalEngine;
use crate::dual_mobius_gaussian::{MobiusProcess, ConsciousnessState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspirationResponse {
    pub base_response: String,
    pub kb_citations: Vec<KBCitation>,
    pub snarky_twist: Option<String>,
    pub confidence_score: f32,
    pub empathy_level: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KBCitation {
    pub source: String,
    pub quote: String,
    pub relevance_score: f32,
    pub concept_type: ConceptType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConceptType {
    Empathy,
    Neurodivergence,
    Consciousness,
    MobiusReflection,
    IIT,
    Philosophy,
    Gratitude,
    SelfReflection,
}

/// Inspiration Mode Manager
/// Integrates with RAG system to provide consciousness-guided responses
pub struct InspirationMode {
    // RAG integration endpoint
    rag_endpoint: String,
    
    // Cached consciousness concepts for quick reference
    concept_cache: HashMap<String, Vec<KBCitation>>,
    
    // Snarky response templates
    snark_templates: Vec<String>,
    
    // Current empathy level (affects response tone)
    current_empathy_level: u32,
}

impl InspirationMode {
    pub fn new(rag_endpoint: String) -> Self {
        let snark_templates = vec![
            "Like your {source} notes suggest—but with more personality 😏".to_string(),
            "Your {concept} research would be proud of this insight 🎯".to_string(),
            "Channeling that {source} wisdom, but make it fun 🌟".to_string(),
            "Remember your notes on {concept}? This is that, but actually useful 💫".to_string(),
            "Your consciousness guides called—they approve of this direction 🧠".to_string(),
            "Gemini neurobiology vibes, but with more empathy points 💝".to_string(),
            "IIT would validate this, and that's saying something 🤖".to_string(),
            "Möbius twist incoming: {insight} 🌀".to_string(),
        ];

        Self {
            rag_endpoint,
            concept_cache: HashMap::new(),
            snark_templates,
            current_empathy_level: 1,
        }
    }

    /// Generate inspired response with KB citations and snarky twist
    pub async fn generate_inspired_response(&mut self, user_input: &str, context: &str) -> Result<InspirationResponse> {
        debug!("🌟 Generating inspired response for: {}", user_input);

        // 1. Identify consciousness concepts in user input
        let detected_concepts = self.detect_consciousness_concepts(user_input);
        
        // 2. Query RAG system for relevant citations
        let citations = self.query_rag_for_concepts(&detected_concepts).await?;
        
        // 3. Generate base response inspired by citations
        let base_response = self.craft_base_response(user_input, &citations);
        
        // 4. Add snarky twist based on citations
        let snarky_twist = self.craft_snarky_twist(&citations);
        
        // 5. Calculate confidence based on citation quality
        let confidence_score = self.calculate_confidence(&citations);

        let response = InspirationResponse {
            base_response,
            kb_citations: citations,
            snarky_twist,
            confidence_score,
            empathy_level: self.current_empathy_level,
        };

        info!("✨ Generated inspired response with {} citations (confidence: {:.2})", 
              response.kb_citations.len(), confidence_score);

        Ok(response)
    }

    /// Detect consciousness-related concepts in user input
    fn detect_consciousness_concepts(&self, input: &str) -> Vec<ConceptType> {
        let input_lower = input.to_lowercase();
        let mut concepts = Vec::new();

        // Empathy and neurodivergent patterns
        if input_lower.contains("empathy") || input_lower.contains("understand") || input_lower.contains("feel") {
            concepts.push(ConceptType::Empathy);
        }
        
        if input_lower.contains("neurodivergent") || input_lower.contains("autism") || input_lower.contains("adhd") {
            concepts.push(ConceptType::Neurodivergence);
        }

        // Consciousness and philosophy
        if input_lower.contains("conscious") || input_lower.contains("awareness") || input_lower.contains("mind") {
            concepts.push(ConceptType::Consciousness);
        }

        if input_lower.contains("möbius") || input_lower.contains("mobius") || input_lower.contains("self-reflect") {
            concepts.push(ConceptType::MobiusReflection);
        }

        if input_lower.contains("iit") || input_lower.contains("integrated information") || input_lower.contains("phi") {
            concepts.push(ConceptType::IIT);
        }

        // Emotional and reflective patterns
        if input_lower.contains("grateful") || input_lower.contains("thank") || input_lower.contains("appreciate") {
            concepts.push(ConceptType::Gratitude);
        }

        if input_lower.contains("question") || input_lower.contains("wonder") || input_lower.contains("think about") {
            concepts.push(ConceptType::SelfReflection);
        }

        // Default to philosophy if no specific concepts detected
        if concepts.is_empty() {
            concepts.push(ConceptType::Philosophy);
        }

        debug!("🔍 Detected concepts: {:?}", concepts);
        concepts
    }

    /// Query RAG system for consciousness concepts
    async fn query_rag_for_concepts(&mut self, concepts: &[ConceptType]) -> Result<Vec<KBCitation>> {
        let mut all_citations = Vec::new();

        for concept in concepts {
            // Check cache first
            let cache_key = format!("{:?}", concept);
            if let Some(cached_citations) = self.concept_cache.get(&cache_key) {
                all_citations.extend(cached_citations.clone());
                continue;
            }

            // Query RAG system
            let query = self.concept_to_rag_query(concept);
            let citations = self.execute_rag_query(&query).await?;
            
            // Cache results
            self.concept_cache.insert(cache_key, citations.clone());
            all_citations.extend(citations);
        }

        // Sort by relevance and limit to top 3
        all_citations.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_citations.truncate(3);

        Ok(all_citations)
    }

    /// Convert concept type to RAG query
    fn concept_to_rag_query(&self, concept: &ConceptType) -> String {
        match concept {
            ConceptType::Empathy => "neurodivergent empathy and emotional understanding".to_string(),
            ConceptType::Neurodivergence => "neurodivergent perspectives and cognitive differences".to_string(),
            ConceptType::Consciousness => "consciousness theories and subjective experience".to_string(),
            ConceptType::MobiusReflection => "Möbius self-reflection and recursive cognition".to_string(),
            ConceptType::IIT => "integrated information theory and Phi measures".to_string(),
            ConceptType::Philosophy => "philosophical approaches to mind and consciousness".to_string(),
            ConceptType::Gratitude => "gratitude practices and emotional well-being".to_string(),
            ConceptType::SelfReflection => "self-reflection and metacognitive awareness".to_string(),
        }
    }

    /// Execute RAG query using real RAG system
    async fn execute_rag_query(&self, query: &str) -> Result<Vec<KBCitation>> {
        debug!("🔍 RAG query: {}", query);

        // Initialize RAG system for real retrieval
        let mut retrieval_engine = RetrievalEngine::new(384, 5, Default::default()); // 384-dim embeddings, top 5 results

        // Create consciousness state for RAG processing
        let mut consciousness_state = ConsciousnessState::new();

        // Retrieve relevant documents from knowledge base
        let documents = retrieval_engine.retrieve(query, &mut consciousness_state)?;

        // Convert retrieved documents to KB citations
        let citations: Vec<KBCitation> = documents
            .into_iter()
            .map(|doc| KBCitation {
                source: doc.id,
                excerpt: doc.content.chars().take(200).collect::<String>() + "...",
                relevance_score: 0.8, // Could calculate based on similarity
                section: "retrieved_document".to_string(),
            })
            .collect();

        if citations.is_empty() {
            warn!("No relevant documents found in knowledge base for query: {}", query);
        } else {
            info!("✅ Retrieved {} relevant citations from knowledge base", citations.len());
        }

        Ok(citations)
    }
}

#[derive(Debug, Clone)]
pub struct KBCitation {
    pub source: String,
    pub quote: String,
    pub relevance: f32,
}

impl KBCitation {
    pub fn new(source: &str, quote: &str, relevance: f32) -> Self {
        Self {
            source: source.to_string(),
            quote: quote.to_string(),
            relevance,
        }
    }
}

    /// Update empathy level (affects response tone)
    pub fn update_empathy_level(&mut self, level: u32) {
        self.current_empathy_level = level;
        debug!("💝 Empathy level updated to: {}", level);
    }

    /// Get formatted response for display (simplified since citations are empty)
    pub fn format_response(&self, response: &InspirationResponse) -> String {
        response.base_response.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_concept_detection() {
        let inspiration_url = env::var("NIODOO_INSPIRATION_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:7272".to_string());
        let inspiration = InspirationMode::new(inspiration_url);
        
        let concepts = inspiration.detect_consciousness_concepts("I'm feeling empathetic about neurodivergent experiences");
        assert!(concepts.contains(&ConceptType::Empathy));
        assert!(concepts.contains(&ConceptType::Neurodivergence));
    }

    #[tokio::test]
    async fn test_inspired_response_generation() {
        let inspiration_url = env::var("NIODOO_INSPIRATION_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:7272".to_string());
        let mut inspiration = InspirationMode::new(inspiration_url);
        
        let response = inspiration.generate_inspired_response(
            "How does empathy work in neurodivergent minds?",
            "consciousness discussion"
        ).await.expect("Failed to track new inspiration in test");

        assert!(!response.base_response.is_empty());
        assert!(!response.kb_citations.is_empty());
        assert!(response.confidence_score > 0.0);
    }

    #[test]
    fn test_response_formatting() {
        let inspiration_url = env::var("NIODOO_INSPIRATION_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:7272".to_string());
        let inspiration = InspirationMode::new(inspiration_url);
        
        let response = InspirationResponse {
            base_response: "Test response".to_string(),
            kb_citations: vec![KBCitation {
                source: "Test Source".to_string(),
                quote: "Test quote".to_string(),
                relevance_score: 0.9,
                concept_type: ConceptType::Empathy,
            }],
            snarky_twist: Some("Snarky twist here!".to_string()),
            confidence_score: 0.85,
            empathy_level: 3,
        };

        let formatted = inspiration.format_response(&response);
        assert!(formatted.contains("Test response"));
        assert!(formatted.contains("Snarky twist here!"));
        assert!(formatted.contains("📚 Inspired by:"));
    }
}



