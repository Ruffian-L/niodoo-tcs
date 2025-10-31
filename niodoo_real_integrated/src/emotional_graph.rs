//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham
//!
//! Emotional Graph Builder - Phase 2 Integration Module
//!
//! This module builds emotional graphs from conversations using GuessingMemorySystem.
//! It converts conversation logs into emotional graph spheres and creates connections
//! based on emotional and semantic similarity.

use anyhow::{Context, Result};
use niodoo_core::memory::{
    EmotionalVector, GuessingMemorySystem, SphereId, TraversalDirection,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::conversation_log::{ConversationEntry, ConversationLogStore};

/// Configuration for emotional graph building
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalGraphConfig {
    /// Minimum similarity threshold for creating links between spheres
    pub min_link_similarity: f32,
    /// Weight for emotional similarity in link creation (0.0-1.0)
    pub emotional_weight: f32,
    /// Weight for semantic similarity in link creation (0.0-1.0)
    pub semantic_weight: f32,
    /// Maximum number of links per sphere
    pub max_links_per_sphere: usize,
}

impl Default for EmotionalGraphConfig {
    fn default() -> Self {
        Self {
            min_link_similarity: 0.3,
            emotional_weight: 0.6,
            semantic_weight: 0.4,
            max_links_per_sphere: 10,
        }
    }
}

/// Emotional graph builder that wraps GuessingMemorySystem
pub struct EmotionalGraphBuilder {
    graph: GuessingMemorySystem,
    config: EmotionalGraphConfig,
    /// Mapping from conversation entry IDs to sphere IDs
    entry_to_sphere: HashMap<String, SphereId>,
}

impl EmotionalGraphBuilder {
    /// Create a new emotional graph builder
    pub fn new(config: EmotionalGraphConfig) -> Self {
        Self {
            graph: GuessingMemorySystem::new(),
            config,
            entry_to_sphere: HashMap::new(),
        }
    }

    /// Create default builder with default config
    pub fn default() -> Self {
        Self::new(EmotionalGraphConfig::default())
    }

    /// Build graph from conversation log store
    pub fn build_from_conversations(
        &mut self,
        conversation_store: &ConversationLogStore,
    ) -> Result<()> {
        info!("Building emotional graph from {} conversations", conversation_store.count());
        
        // First pass: create spheres from all conversations
        for entry in conversation_store.all_entries() {
            self.add_conversation_entry(entry)?;
        }
        
        // Second pass: create links based on similarity
        self.create_links_from_conversations(conversation_store)?;
        
        info!("Emotional graph built: {} spheres", self.graph.sphere_count());
        Ok(())
    }

    /// Add a single conversation entry as a sphere
    fn add_conversation_entry(&mut self, entry: &ConversationEntry) -> Result<()> {
        let sphere_id = SphereId(format!("sphere_{}", entry.id));
        
        // Calculate 3D position from emotional vector and timestamp
        let position = self.calculate_position(&entry.emotional_vector, &entry.timestamp);
        
        // Create concept from conversation content
        let concept = format!("{} → {}", entry.user_input, entry.ai_response);
        
        // Store memory fragment
        let fragment = format!(
            "{} | {} | {}",
            entry.user_input, entry.ai_response, entry.emotion_state
        );
        
        // Store the sphere
        self.graph.store_memory(
            sphere_id.clone(),
            concept,
            position,
            entry.emotional_vector.clone(),
            fragment,
        );
        
        // Track mapping
        self.entry_to_sphere.insert(entry.id.clone(), sphere_id);
        
        debug!("Added sphere for conversation entry: {}", entry.id);
        Ok(())
    }

    /// Create links between spheres based on similarity
    fn create_links_from_conversations(
        &mut self,
        conversation_store: &ConversationLogStore,
    ) -> Result<()> {
        let entries: Vec<_> = conversation_store.all_entries().to_vec();
        let mut link_count = 0;
        
        for (i, entry_a) in entries.iter().enumerate() {
            let sphere_id_a = self
                .entry_to_sphere
                .get(&entry_a.id)
                .context("Missing sphere for entry")?;
            
            // Find similar entries
            let mut similarities: Vec<(SphereId, f32)> = Vec::new();
            
            for entry_b in entries.iter().skip(i + 1) {
                let sphere_id_b = self
                    .entry_to_sphere
                    .get(&entry_b.id)
                    .context("Missing sphere for entry")?;
                
                // Calculate combined similarity
                let emotional_sim = self.calculate_emotional_similarity(
                    &entry_a.emotional_vector,
                    &entry_b.emotional_vector,
                );
                
                let semantic_sim = entry_a.content_similarity(entry_b);
                
                let combined_similarity = (emotional_sim * self.config.emotional_weight)
                    + (semantic_sim * self.config.semantic_weight);
                
                if combined_similarity >= self.config.min_link_similarity {
                    similarities.push((sphere_id_b.clone(), combined_similarity));
                }
            }
            
            // Sort by similarity and take top N
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            similarities.truncate(self.config.max_links_per_sphere);
            
            // Create links - need to store entry_b references for later use
            let entry_emotions: Vec<(SphereId, f32, EmotionalVector)> = similarities
                .into_iter()
                .filter_map(|(sphere_id_b, similarity)| {
                    // Find the entry_b that corresponds to this sphere
                    let entry_b_id = self
                        .entry_to_sphere
                        .iter()
                        .find(|(_, sid)| **sid == sphere_id_b)
                        .map(|(id, _)| id.clone())?;
                    
                    let entry_b = entries.iter().find(|e| e.id == entry_b_id)?;
                    
                    // Calculate link emotion before mutable borrow
                    let link_emotion = self.calculate_link_emotion(
                        &entry_a.emotional_vector,
                        &entry_b.emotional_vector,
                    );
                    
                    Some((sphere_id_b, similarity, link_emotion))
                })
                .collect();
            
            // Create links
            if let Some(sphere_a) = self.graph.spheres_mut().find(|s| &s.id == sphere_id_a) {
                for (sphere_id_b, similarity, link_emotion) in entry_emotions {
                    sphere_a.add_link(sphere_id_b, similarity, link_emotion);
                    link_count += 1;
                }
            }
        }
        
        info!("Created {} links between spheres", link_count);
        Ok(())
    }

    /// Calculate 3D position from emotional vector and timestamp
    fn calculate_position(
        &self,
        emotion: &EmotionalVector,
        timestamp: &chrono::DateTime<chrono::Utc>,
    ) -> [f32; 3] {
        // Use emotional vector components for x, y
        // Use timestamp as z component (normalized)
        let timestamp_factor = timestamp.timestamp() as f32 / 1_000_000_000.0; // Normalize to reasonable range
        
        [
            emotion.joy * 10.0 - emotion.sadness * 10.0, // x: joy-sadness axis
            emotion.anger * 10.0 - emotion.fear * 10.0,  // y: anger-fear axis
            timestamp_factor % 100.0,                    // z: temporal dimension
        ]
    }

    /// Calculate emotional similarity between two vectors
    fn calculate_emotional_similarity(&self, a: &EmotionalVector, b: &EmotionalVector) -> f32 {
        let dot_product = a.joy * b.joy
            + a.sadness * b.sadness
            + a.anger * b.anger
            + a.fear * b.fear
            + a.surprise * b.surprise;
        
        let mag_a = a.magnitude();
        let mag_b = b.magnitude();
        
        if mag_a > 0.0 && mag_b > 0.0 {
            (dot_product / (mag_a * mag_b)).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Calculate link emotion as average of two emotional vectors
    fn calculate_link_emotion(&self, a: &EmotionalVector, b: &EmotionalVector) -> EmotionalVector {
        EmotionalVector::new(
            (a.joy + b.joy) / 2.0,
            (a.sadness + b.sadness) / 2.0,
            (a.anger + b.anger) / 2.0,
            (a.fear + b.fear) / 2.0,
            (a.surprise + b.surprise) / 2.0,
        )
    }

    /// Get a reference to the underlying graph
    pub fn graph(&self) -> &GuessingMemorySystem {
        &self.graph
    }

    /// Get a mutable reference to the underlying graph
    pub fn graph_mut(&mut self) -> &mut GuessingMemorySystem {
        &mut self.graph
    }

    /// Take ownership of the graph
    pub fn into_graph(self) -> GuessingMemorySystem {
        self.graph
    }

    /// Traverse the graph using Möbius traversal
    pub fn traverse(
        &self,
        start_sphere_id: &SphereId,
        direction: TraversalDirection,
        depth: usize,
    ) -> Vec<(SphereId, String)> {
        self.graph.mobius_traverse(start_sphere_id, direction, depth)
    }

    /// Find spheres by emotional similarity
    pub fn find_by_emotion(
        &self,
        query_emotion: &EmotionalVector,
        threshold: f32,
    ) -> Vec<(SphereId, String, f32)> {
        self.graph
            .spheres()
            .filter_map(|sphere| {
                let similarity = sphere.emotional_similarity(query_emotion);
                if similarity >= threshold {
                    Some((sphere.id.clone(), sphere.core_concept.clone(), similarity))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get sphere count
    pub fn sphere_count(&self) -> usize {
        self.graph.sphere_count()
    }

    /// Get link count across all spheres
    pub fn link_count(&self) -> usize {
        self.graph.spheres().map(|s| s.links.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation_log::ConversationLogStore;

    #[test]
    fn test_emotional_graph_builder() {
        let mut builder = EmotionalGraphBuilder::default();
        assert_eq!(builder.sphere_count(), 0);
    }

    #[test]
    fn test_build_from_conversations() {
        let temp_dir = std::env::temp_dir();
        let store_path = temp_dir.join("test_conv.json");
        let mut store = ConversationLogStore::new(&store_path);
        
        let entry1 = ConversationEntry::new(
            "Hello".to_string(),
            "Hi!".to_string(),
            EmotionalVector::new(0.8, 0.1, 0.0, 0.0, 0.1),
            "joyful".to_string(),
        );
        let entry2 = ConversationEntry::new(
            "How are you?".to_string(),
            "I'm good!".to_string(),
            EmotionalVector::new(0.7, 0.2, 0.0, 0.0, 0.1),
            "joyful".to_string(),
        );
        
        store.store(entry1).unwrap();
        store.store(entry2).unwrap();
        
        let mut builder = EmotionalGraphBuilder::default();
        assert!(builder.build_from_conversations(&store).is_ok());
        assert!(builder.sphere_count() >= 2);
    }
}

