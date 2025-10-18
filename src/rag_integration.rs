//! RAG Integration for Consciousness-Aware Training
//!
//! This module provides Retrieval-Augmented Generation (RAG) capabilities
//! for the QLoRA training process, using 5D emotional vectors for retrieval.
//!
//! Key features:
//! - 5D emotional vector embeddings (joy, sadness, anger, fear, surprise)
//! - Guessing Spheres memory system for probabilistic retrieval
//! - Möbius topology integration for emotional state transitions
//! - Consciousness-aware document retrieval

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::qwen_curator::{EmotionalState, LearningEvent, TopologyMetrics};

/// Configuration for RAG retrieval system - NO HARDCODED VALUES
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RagConfig {
    /// Similarity threshold for creating links between memory spheres
    pub similarity_threshold_link: f32,
    /// Similarity threshold for document retrieval
    pub similarity_threshold_retrieve: f32,
    /// Default top-k for retrieval
    pub top_k_default: usize,
    /// Weight for emotional component in retrieval scoring
    pub emotional_weight: f32,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            similarity_threshold_link: 0.3,
            similarity_threshold_retrieve: 0.2,
            top_k_default: 5,
            emotional_weight: 0.7,
        }
    }
}

/// 5D emotional vector for document embeddings
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionalVector {
    pub joy: f32,
    pub sadness: f32,
    pub anger: f32,
    pub fear: f32,
    pub surprise: f32,
}

impl EmotionalVector {
    /// Create a new emotional vector
    pub fn new(joy: f32, sadness: f32, anger: f32, fear: f32, surprise: f32) -> Self {
        Self {
            joy,
            sadness,
            anger,
            fear,
            surprise,
        }
    }

    /// Create from a learning event's emotional state
    pub fn from_learning_event(event: &LearningEvent) -> Option<Self> {
        if let Some(emotional_state) = &event.emotional_state {
            // Map PAD (Pleasure-Arousal-Dominance) to 5D emotional space
            // This is a simplified mapping and could be improved
            let joy = emotional_state.pleasure.max(0.0) as f32;
            let sadness = (-emotional_state.pleasure).max(0.0) as f32;
            let anger = (emotional_state.arousal * (1.0 - emotional_state.dominance)) as f32;
            let fear = (emotional_state.arousal * (1.0 - emotional_state.pleasure)) as f32;
            let surprise = emotional_state.arousal as f32;
            
            Some(Self::new(joy, sadness, anger, fear, surprise))
        } else {
            None
        }
    }

    /// Calculate cosine similarity with another emotional vector
    pub fn similarity(&self, other: &Self) -> f32 {
        let dot_product = (self.joy * other.joy)
            + (self.sadness * other.sadness)
            + (self.anger * other.anger)
            + (self.fear * other.fear)
            + (self.surprise * other.surprise);

        let self_magnitude = self.magnitude();
        let other_magnitude = other.magnitude();

        if self_magnitude < f32::EPSILON || other_magnitude < f32::EPSILON {
            return 0.0;
        }

        dot_product / (self_magnitude * other_magnitude)
    }

    /// Calculate magnitude of the emotional vector
    pub fn magnitude(&self) -> f32 {
        (self.joy.powi(2)
            + self.sadness.powi(2)
            + self.anger.powi(2)
            + self.fear.powi(2)
            + self.surprise.powi(2))
        .sqrt()
    }

    /// Convert to array representation
    pub fn as_array(&self) -> [f32; 5] {
        [self.joy, self.sadness, self.anger, self.fear, self.surprise]
    }
}

/// Document for RAG retrieval
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier
    pub id: String,
    /// Document content
    pub content: String,
    /// Emotional embedding
    pub embedding: EmotionalVector,
    /// Metadata (title, source, etc.)
    pub metadata: HashMap<String, String>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Document {
    /// Create a new document from a learning event
    pub fn from_learning_event(event: &LearningEvent) -> Result<Self> {
        let embedding = EmotionalVector::from_learning_event(event)
            .ok_or_else(|| anyhow!("Learning event has no emotional state"))?;

        let mut metadata = HashMap::new();
        metadata.insert("timestamp".to_string(), event.timestamp.clone());
        
        if let Some(coherence) = event.coherence {
            metadata.insert("coherence".to_string(), coherence.to_string());
        }
        
        if let Some(topology) = &event.topology_metrics {
            metadata.insert("curvature".to_string(), topology.curvature.to_string());
            metadata.insert("twist_factor".to_string(), topology.twist_factor.to_string());
        }

        Ok(Self {
            id: Uuid::new_v4().to_string(),
            content: format!("Input: {}\nResponse: {}", event.input, event.response),
            embedding,
            metadata,
            created_at: chrono::Utc::now(),
        })
    }

    /// Get importance score from metadata (0.0 if not set)
    pub fn importance(&self) -> f32 {
        self.metadata
            .get("importance")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0)
    }

    /// Check if this document is marked as a breakthrough
    pub fn is_breakthrough(&self) -> bool {
        self.metadata
            .get("priority")
            .map(|s| s == "breakthrough")
            .unwrap_or(false)
    }

    /// Set importance score
    pub fn set_importance(&mut self, score: f32) {
        self.metadata.insert("importance".to_string(), score.to_string());
    }

    /// Mark as breakthrough memory
    pub fn mark_as_breakthrough(&mut self) {
        self.metadata.insert("priority".to_string(), "breakthrough".to_string());
    }
}

/// Memory sphere for probabilistic retrieval
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemorySphere {
    /// Unique identifier
    pub id: String,
    /// Core concept
    pub concept: String,
    /// Emotional embedding
    pub emotional_profile: EmotionalVector,
    /// Document content
    pub content: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Links to other spheres
    pub links: HashMap<String, f32>,
}

impl MemorySphere {
    /// Create a new memory sphere from a document
    pub fn from_document(doc: &Document) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            concept: doc.metadata.get("title").cloned().unwrap_or_else(|| "Untitled".to_string()),
            emotional_profile: doc.embedding.clone(),
            content: doc.content.clone(),
            created_at: doc.created_at,
            links: HashMap::new(),
        }
    }

    /// Calculate emotional similarity with query emotion
    pub fn emotional_similarity(&self, query_emotion: &EmotionalVector) -> f32 {
        self.emotional_profile.similarity(query_emotion)
    }
}

/// RAG retrieval engine using 5D emotional vectors
pub struct RagEngine {
    /// Memory spheres for retrieval
    spheres: Vec<MemorySphere>,
    /// Document store
    documents: HashMap<String, Document>,
    /// Base directory for storing RAG data
    base_dir: PathBuf,
    /// Configuration for retrieval thresholds - NO HARDCODING
    config: RagConfig,
}

impl RagEngine {
    /// Create a new RAG engine with configuration
    pub fn new(base_dir: PathBuf, config: RagConfig) -> Result<Self> {
        // Create directories if they don't exist
        let rag_dir = base_dir.join("rag_data");
        if !rag_dir.exists() {
            std::fs::create_dir_all(&rag_dir)?;
        }

        Ok(Self {
            spheres: Vec::new(),
            documents: HashMap::new(),
            base_dir: rag_dir,
            config,
        })
    }
    
    /// Create a new RAG engine with default configuration
    pub fn with_defaults(base_dir: PathBuf) -> Result<Self> {
        Self::new(base_dir, RagConfig::default())
    }

    /// Add a learning event to the RAG system
    pub fn add_learning_event(&mut self, event: &LearningEvent) -> Result<()> {
        // Convert learning event to document
        let document = Document::from_learning_event(event)?;
        
        // Create memory sphere from document
        let sphere = MemorySphere::from_document(&document);
        
        // Add document to store
        self.documents.insert(document.id.clone(), document);
        
        // Add sphere to memory
        self.spheres.push(sphere);
        
        // Update links between spheres
        self.update_sphere_links()?;
        
        Ok(())
    }

    /// Update links between memory spheres based on emotional similarity
    fn update_sphere_links(&mut self) -> Result<()> {
        // Skip if we have fewer than 2 spheres
        if self.spheres.len() < 2 {
            return Ok(());
        }
        
        // Get the most recent sphere
        let latest_idx = self.spheres.len() - 1;
        let latest_sphere = &self.spheres[latest_idx];
        let latest_emotion = &latest_sphere.emotional_profile;
        
        // Update links for the latest sphere
        let mut new_links = HashMap::new();
        
        for (i, sphere) in self.spheres.iter().enumerate() {
            if i == latest_idx {
                continue; // Skip self-link
            }
            
            // Calculate emotional similarity
            let similarity = latest_emotion.similarity(&sphere.emotional_profile);
            
            // Only create links with significant similarity (config-driven)
            if similarity > self.config.similarity_threshold_link {
                new_links.insert(sphere.id.clone(), similarity);
            }
        }
        
        // Update the links
        self.spheres[latest_idx].links = new_links;
        
        Ok(())
    }

    /// Retrieve relevant documents based on emotional query
    pub fn retrieve(&self, query_emotion: &EmotionalVector, top_k: usize) -> Vec<(Document, f32)> {
        let mut results = Vec::new();
        
        // Calculate similarity for each sphere
        for sphere in &self.spheres {
            let similarity = sphere.emotional_similarity(query_emotion);
            
            // Only include results with significant similarity (config-driven)
            if similarity > self.config.similarity_threshold_retrieve {
                if let Some(doc) = self.documents.get(&sphere.id) {
                    results.push((doc.clone(), similarity));
                }
            }
        }
        
        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top-k results
        results.into_iter().take(top_k).collect()
    }

    /// Save RAG data to disk
    pub fn save(&self) -> Result<()> {
        // Save spheres
        let spheres_path = self.base_dir.join("memory_spheres.json");
        let spheres_json = serde_json::to_string_pretty(&self.spheres)?;
        std::fs::write(spheres_path, spheres_json)?;
        
        // Save documents
        let docs_path = self.base_dir.join("documents.json");
        let docs_json = serde_json::to_string_pretty(&self.documents)?;
        std::fs::write(docs_path, docs_json)?;
        
        Ok(())
    }

    /// Load RAG data from disk
    pub fn load(&mut self) -> Result<()> {
        // Load spheres if file exists
        let spheres_path = self.base_dir.join("memory_spheres.json");
        if spheres_path.exists() {
            let spheres_json = std::fs::read_to_string(spheres_path)?;
            self.spheres = serde_json::from_str(&spheres_json)?;
        }
        
        // Load documents if file exists
        let docs_path = self.base_dir.join("documents.json");
        if docs_path.exists() {
            let docs_json = std::fs::read_to_string(docs_path)?;
            self.documents = serde_json::from_str(&docs_json)?;
        }
        
        Ok(())
    }

    /// Store document with explicit importance/priority score
    ///
    /// High-importance documents are breakthrough moments where the agent
    /// successfully transitioned from STUCK→UNSTUCK. These get:
    /// - Boosted retrieval scores
    /// - Preferential consolidation
    /// - Tagged for long-term retention
    ///
    /// # Arguments
    /// * `content` - The action/response that resolved the stuck state
    /// * `emotional_vector` - Emotional state when breakthrough occurred
    /// * `importance` - Intrinsic reward magnitude (typically 5.0-20.0 for breakthroughs)
    pub fn store_with_priority(
        &mut self,
        content: &str,
        emotional_vector: &EmotionalVector,
        importance: f32,
    ) -> Result<()> {
        let mut doc = Document {
            id: Uuid::new_v4().to_string(),
            content: content.to_string(),
            embedding: emotional_vector.clone(),
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
        };

        // Tag with importance and breakthrough status
        doc.set_importance(importance);

        if importance > 5.0 {
            doc.mark_as_breakthrough();
        }

        // Add to memory system
        self.add_document(doc)?;

        tracing::info!(
            "💾 Stored priority memory (importance: {:.2}): {}",
            importance,
            content.chars().take(60).collect::<String>()
        );

        Ok(())
    }

    /// Retrieve documents with importance weighting
    ///
    /// Modifies retrieval scoring to boost high-importance memories.
    /// This helps the agent recall past solutions when stuck again.
    ///
    /// Scoring formula:
    ///   final_score = emotional_similarity * (1 + importance_boost)
    ///   where importance_boost = importance * 0.1
    pub fn retrieve_with_importance_boost(
        &self,
        query_vector: &EmotionalVector,
        top_k: usize,
    ) -> Result<Vec<Document>> {
        // Get all documents
        let documents = self.get_all_documents()?;

        // Score with importance boosting
        let mut scored_docs: Vec<(Document, f32)> = documents
            .into_iter()
            .map(|doc| {
                // Base similarity score
                let similarity = query_vector.similarity(&doc.embedding);

                // Importance boost: 0.1x multiplier per importance point
                let importance = doc.importance();
                let boost = 1.0 + (importance * 0.1);

                // Breakthrough memories get additional 2x boost
                let breakthrough_multiplier = if doc.is_breakthrough() { 2.0 } else { 1.0 };

                let final_score = similarity * boost * breakthrough_multiplier;

                (doc, final_score)
            })
            .collect();

        // Sort by final score (descending)
        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top-k
        let results = scored_docs
            .into_iter()
            .take(top_k)
            .map(|(doc, _score)| doc)
            .collect();

        Ok(results)
    }

    /// Get all breakthrough memories
    ///
    /// Returns documents tagged as breakthrough moments, useful for:
    /// - Analytics on what patterns led to success
    /// - Building a "skill library" of proven solutions
    /// - Visualization of learning progress
    pub fn get_breakthrough_memories(&self) -> Result<Vec<Document>> {
        let all_docs = self.get_all_documents()?;

        let breakthroughs: Vec<Document> = all_docs
            .into_iter()
            .filter(|doc| doc.is_breakthrough())
            .collect();

        Ok(breakthroughs)
    }

    /// Consolidate memory: prune low-importance docs, retain breakthroughs
    ///
    /// Memory management strategy:
    /// - Keep ALL breakthrough memories (importance > 5.0)
    /// - Keep high-importance memories (importance > 2.0)
    /// - Probabilistically prune low-importance memories
    ///
    /// This prevents memory bloat while preserving critical learning moments.
    pub fn consolidate_memory(&mut self, retention_threshold: f32) -> Result<usize> {
        let all_docs = self.get_all_documents()?;
        let initial_count = all_docs.len();

        let retained_docs: Vec<Document> = all_docs
            .into_iter()
            .filter(|doc| {
                let importance = doc.importance();

                // Always keep breakthroughs
                if doc.is_breakthrough() {
                    return true;
                }

                // Keep if importance above threshold
                importance >= retention_threshold
            })
            .collect();

        let pruned_count = initial_count - retained_docs.len();

        // Replace documents with retained set
        self.clear_documents()?;

        for doc in retained_docs {
            self.add_document(doc)?;
        }

        tracing::info!(
            "🧹 Memory consolidation: {} docs pruned, {} retained",
            pruned_count,
            self.document_count()?
        );

        Ok(pruned_count)
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> Result<MemoryStats> {
        let all_docs = self.get_all_documents()?;

        let total_count = all_docs.len();
        let breakthrough_count = all_docs.iter().filter(|d| d.is_breakthrough()).count();

        let total_importance: f32 = all_docs.iter().map(|d| d.importance()).sum();
        let avg_importance = if total_count > 0 {
            total_importance / total_count as f32
        } else {
            0.0
        };

        let max_importance = all_docs
            .iter()
            .map(|d| d.importance())
            .fold(0.0, f32::max);

        Ok(MemoryStats {
            total_documents: total_count,
            breakthrough_memories: breakthrough_count,
            total_importance,
            average_importance: avg_importance,
            max_importance,
        })
    }

    /// Get all documents from memory
    fn get_all_documents(&self) -> Result<Vec<Document>> {
        Ok(self.documents.values().cloned().collect())
    }

    /// Clear all documents (for consolidation)
    fn clear_documents(&mut self) -> Result<()> {
        self.documents.clear();
        self.spheres.clear();
        Ok(())
    }

    /// Add single document to memory
    pub fn add_document(&mut self, doc: Document) -> Result<()> {
        // Create memory sphere from document
        let sphere = MemorySphere::from_document(&doc);

        // Add document to store
        self.documents.insert(doc.id.clone(), doc);

        // Add sphere to memory
        self.spheres.push(sphere);

        // Update links between spheres
        self.update_sphere_links()?;

        Ok(())
    }

    /// Get document count
    fn document_count(&self) -> Result<usize> {
        Ok(self.documents.len())
    }
}

/// Memory statistics struct
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_documents: usize,
    pub breakthrough_memories: usize,
    pub total_importance: f32,
    pub average_importance: f32,
    pub max_importance: f32,
}

impl std::fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Docs: {} | Breakthroughs: {} | Total Importance: {:.2} | Avg: {:.2} | Max: {:.2}",
            self.total_documents,
            self.breakthrough_memories,
            self.total_importance,
            self.average_importance,
            self.max_importance
        )
    }
}
pub struct ConsciousnessRagIntegration {
    /// RAG engine
    rag_engine: RagEngine,
    /// Current emotional state
    current_emotion: EmotionalVector,
    /// Base directory
    base_dir: PathBuf,
}

impl ConsciousnessRagIntegration {
    /// Create a new consciousness-aware RAG integration
    pub fn new(base_dir: PathBuf) -> Result<Self> {
        let rag_engine = RagEngine::with_defaults(base_dir.clone())?;
        
        // Initialize with neutral emotional state
        let current_emotion = EmotionalVector::new(0.5, 0.5, 0.5, 0.5, 0.5);
        
        Ok(Self {
            rag_engine,
            current_emotion,
            base_dir,
        })
    }

    /// Update emotional state based on learning event
    pub fn update_emotional_state(&mut self, event: &LearningEvent) -> Result<()> {
        if let Some(emotion) = EmotionalVector::from_learning_event(event) {
            self.current_emotion = emotion;
        }
        
        Ok(())
    }

    /// Process a batch of learning events
    pub fn process_batch(&mut self, events: &[LearningEvent]) -> Result<()> {
        let start_time = Instant::now();
        
        info!("🔄 Processing {} learning events for RAG integration", events.len());
        
        for event in events {
            // Add event to RAG system
            self.rag_engine.add_learning_event(event)?;
            
            // Update emotional state
            self.update_emotional_state(event)?;
        }
        
        // Save RAG data
        self.rag_engine.save()?;
        
        let duration = start_time.elapsed();
        info!("✅ RAG processing completed in {:.2}ms", duration.as_millis());
        
        Ok(())
    }

    /// Retrieve relevant documents for current emotional state
    pub fn retrieve_relevant_documents(&self, top_k: usize) -> Vec<(Document, f32)> {
        self.rag_engine.retrieve(&self.current_emotion, top_k)
    }

    /// Enhance training examples with RAG context
    pub fn enhance_training_examples<T>(&self, examples: &mut [T], enhance_fn: impl Fn(&mut T, &Document, f32)) -> Result<()> {
        // Skip if no examples
        if examples.is_empty() {
            return Ok(());
        }
        
        // Retrieve relevant documents
        let relevant_docs = self.retrieve_relevant_documents(3);
        
        if relevant_docs.is_empty() {
            debug!("No relevant documents found for RAG enhancement");
            return Ok(());
        }
        
        info!("🔍 Enhancing training examples with {} relevant documents", relevant_docs.len());
        
        // Enhance each example
        for example in examples.iter_mut() {
            for (doc, score) in &relevant_docs {
                enhance_fn(example, doc, *score);
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_emotional_vector_similarity() {
        let v1 = EmotionalVector::new(0.8, 0.2, 0.1, 0.3, 0.5);
        let v2 = EmotionalVector::new(0.7, 0.3, 0.2, 0.2, 0.6);
        
        let similarity = v1.similarity(&v2);
        assert!(similarity > 0.9, "Similar vectors should have high similarity");
        
        let v3 = EmotionalVector::new(0.1, 0.9, 0.8, 0.7, 0.2);
        let similarity = v1.similarity(&v3);
        assert!(similarity < 0.5, "Different vectors should have low similarity");
    }
    
    #[test]
    fn test_document_from_learning_event() {
        let event = LearningEvent {
            timestamp: "2025-10-17T12:00:00Z".to_string(),
            input: "How does consciousness work?".to_string(),
            response: "Consciousness emerges from complex neural patterns.".to_string(),
            emotional_state: Some(EmotionalState {
                pleasure: 0.7,
                arousal: 0.6,
                dominance: 0.5,
            }),
            coherence: Some(0.8),
            memory_activations: Some(vec![0.5, 0.6, 0.7]),
            topology_metrics: Some(TopologyMetrics {
                curvature: 0.3,
                twist_factor: 0.5,
                geodesic_distance: 0.7,
            }),
        };
        
        let doc = Document::from_learning_event(&event).unwrap();
        assert!(doc.content.contains("How does consciousness work?"));
        assert!(doc.content.contains("Consciousness emerges from complex neural patterns."));
        assert!(doc.metadata.contains_key("coherence"));
        assert!(doc.metadata.contains_key("curvature"));
    }
}