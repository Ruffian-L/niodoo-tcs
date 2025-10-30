// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use crate::dual_mobius_gaussian::ConsciousnessState;
/// RAG-Memory Integration Adapter
///
/// This module provides transformation layers between RAG system types
/// and non-orientable memory system types (Möbius, Toroidal, Gaussian Spheres).
///
/// CRITICAL INTEGRATION POINTS:
/// 1. Document -> MemoryFragment transformation
/// 2. DocumentRecord -> ToroidalMemoryNode transformation
/// 3. Memory -> GaussianMemorySphere transformation
/// 4. Embedding synchronization across systems
/// 5. Persistence coordination
use crate::memory::{
    guessing_spheres::{EmotionalVector, GuessingMemorySystem, GuessingSphere, SphereId},
    mobius::{MemoryFragment, MemoryLayer},
    toroidal::{ToroidalCoordinate, ToroidalMemoryNode},
};
use crate::rag::{storage::DocumentRecord, Document};
use anyhow::{anyhow, Result};
use chrono::Utc;
use serde_json::Value;
use std::collections::HashMap;

/// Adapter for RAG-Memory integration
pub struct RagMemoryAdapter {
    /// Document ID to Sphere ID mapping
    doc_to_sphere: HashMap<String, SphereId>,
    /// Document ID to Toroidal coordinate mapping
    doc_to_toroidal: HashMap<String, ToroidalCoordinate>,
    /// Document ID to Memory layer mapping
    doc_to_layer: HashMap<String, MemoryLayer>,
}

impl RagMemoryAdapter {
    pub fn new() -> Self {
        Self {
            doc_to_sphere: HashMap::new(),
            doc_to_toroidal: HashMap::new(),
            doc_to_layer: HashMap::new(),
        }
    }

    /// Transform RAG Document to Möbius MemoryFragment
    ///
    /// TYPE MAPPING:
    /// - Document.content -> MemoryFragment.content
    /// - Document.embedding -> Used for layer classification
    /// - Document.resonance_hint -> MemoryFragment.relevance
    /// - Document.created_at -> MemoryFragment.timestamp
    pub fn document_to_memory_fragment(&self, doc: &Document) -> Result<MemoryFragment> {
        // Determine memory layer based on document metadata
        let layer = self.classify_memory_layer(doc);

        // Convert timestamp
        let timestamp = doc.created_at.timestamp() as f64;

        // Extract relevance from resonance hint or default
        let relevance = doc.resonance_hint.unwrap_or(0.5);

        Ok(MemoryFragment {
            content: doc.content.clone(),
            layer,
            relevance,
            timestamp,
        })
    }

    /// Transform RAG DocumentRecord to Toroidal MemoryNode
    ///
    /// TYPE MAPPING:
    /// - DocumentRecord.content_hash -> ToroidalMemoryNode.id
    /// - DocumentRecord.embedding -> ToroidalMemoryNode.emotional_vector
    /// - DocumentRecord.document.content -> ToroidalMemoryNode.content
    /// - Computed from embedding -> ToroidalMemoryNode.coordinate
    pub fn document_record_to_toroidal_node(
        &mut self,
        record: &DocumentRecord,
    ) -> Result<ToroidalMemoryNode> {
        // Compute toroidal coordinate from embedding
        let coordinate = self.embedding_to_toroidal_coordinate(&record.embedding)?;

        // Store mapping for future lookups
        self.doc_to_toroidal
            .insert(record.content_hash.clone(), coordinate.clone());

        // Convert embedding to emotional vector
        let emotional_vector = self.embedding_to_emotional_vector(&record.embedding);

        // Extract temporal context from metadata
        let temporal_context = vec![
            record.created_at.timestamp() as f64,
            record.token_count as f64,
            record.resonance_hint.unwrap_or(0.5) as f64,
        ];

        // Calculate activation strength from resonance hint
        let activation_strength = record.resonance_hint.unwrap_or(0.5) as f64;

        Ok(ToroidalMemoryNode {
            id: record.content_hash.clone(),
            coordinate,
            content: record.document.content.clone(),
            emotional_vector,
            temporal_context,
            activation_strength,
            connections: HashMap::new(), // Connections will be built by toroidal system
        })
    }

    /// Transform RAG Memory to Gaussian Sphere
    ///
    /// TYPE MAPPING:
    /// - Memory.content -> GuessingSphere concept
    /// - Memory.embedding -> GuessingSphere position (3D projection)
    /// - Memory.emotional_valence -> EmotionalVector
    /// - Memory.mobius_u, mobius_v -> Möbius coordinates
    pub fn rag_memory_to_gaussian_sphere(
        &mut self,
        memory: &crate::rag::real_storage::Memory,
        gaussian_system: &mut GuessingMemorySystem,
    ) -> Result<SphereId> {
        // Generate unique sphere ID
        let sphere_id = SphereId(format!("sphere_{}", memory.id));

        // Convert embedding to 3D position for Gaussian sphere
        let position = self.embedding_to_3d_position(&memory.embedding)?;

        // Convert emotional valence to EmotionalVector
        let emotional_vector = self.valence_to_emotional_vector(memory.emotional_valence);

        // Store the memory in Gaussian system
        gaussian_system.store_memory(
            sphere_id.clone(),
            memory.content.clone(),
            position,
            emotional_vector,
            format!(
                "Source: {} | Confidence: {:.2}",
                memory.source, memory.confidence
            ),
        );

        // Store mapping
        self.doc_to_sphere
            .insert(memory.id.to_string(), sphere_id.clone());

        Ok(sphere_id)
    }

    /// Bidirectional lookup: Document ID -> Sphere ID
    pub fn get_sphere_for_document(&self, doc_id: &str) -> Option<&SphereId> {
        self.doc_to_sphere.get(doc_id)
    }

    /// Bidirectional lookup: Document ID -> Toroidal Coordinate
    pub fn get_toroidal_coordinate(&self, doc_id: &str) -> Option<&ToroidalCoordinate> {
        self.doc_to_toroidal.get(doc_id)
    }

    /// Classify document into appropriate memory layer
    fn classify_memory_layer(&self, doc: &Document) -> MemoryLayer {
        // Classification logic based on document metadata
        if let Some(source_type) = &doc.source_type {
            match source_type.as_str() {
                "core" | "system" => MemoryLayer::CoreBurned,
                "procedure" | "howto" => MemoryLayer::Procedural,
                "event" | "experience" => MemoryLayer::Episodic,
                "fact" | "knowledge" => MemoryLayer::Semantic,
                "sensation" | "feeling" => MemoryLayer::Somatic,
                _ => MemoryLayer::Working,
            }
        } else {
            // Default to semantic for general knowledge
            MemoryLayer::Semantic
        }
    }

    /// Convert embedding vector to toroidal coordinate
    fn embedding_to_toroidal_coordinate(&self, embedding: &[f32]) -> Result<ToroidalCoordinate> {
        if embedding.len() < 2 {
            return Err(anyhow!("Embedding too short for coordinate computation"));
        }

        // Use first dimensions for theta and phi
        // Normalize to [0, 2π] range
        let sum: f32 = embedding.iter().take(64).sum();
        let theta = (embedding[0] as f64 * 6.28318).abs() % 6.28318;
        let phi = (embedding[1] as f64 * 6.28318).abs() % 6.28318;

        Ok(ToroidalCoordinate::new(theta, phi))
    }

    /// Convert embedding to emotional vector (5 dimensions)
    fn embedding_to_emotional_vector(&self, embedding: &[f32]) -> Vec<f64> {
        // Project high-dimensional embedding to 5-dimensional emotional space
        if embedding.len() < 5 {
            return vec![0.5; 5]; // Default neutral emotions
        }

        // Use first 5 dimensions and normalize
        let emotions: Vec<f64> = embedding
            .iter()
            .take(5)
            .map(|&x| (x as f64).abs().min(1.0))
            .collect();

        emotions
    }

    /// Convert embedding to 3D position for Gaussian sphere
    fn embedding_to_3d_position(&self, embedding: &[f32]) -> Result<[f32; 3]> {
        if embedding.len() < 3 {
            return Err(anyhow!("Embedding too short for 3D position"));
        }

        // Use PCA-like projection (simplified)
        Ok([embedding[0], embedding[1], embedding[2]])
    }

    /// Convert emotional valence [-1, 1] to EmotionalVector
    fn valence_to_emotional_vector(&self, valence: f32) -> EmotionalVector {
        if valence > 0.5 {
            // High positive valence -> Joy
            EmotionalVector {
                joy: valence,
                sadness: 0.0_f32,
                anger: 0.0_f32,
                fear: 0.0_f32,
                surprise: (1.0 - valence),
            }
        } else if valence < -0.5 {
            // High negative valence -> Sadness/Fear
            EmotionalVector {
                joy: 0.0_f32,
                sadness: (-valence) * 0.6_f32,
                anger: (-valence) * 0.2_f32,
                fear: (-valence) * 0.2_f32,
                surprise: 0.0_f32,
            }
        } else {
            // Neutral
            EmotionalVector {
                joy: 0.4_f32,
                sadness: 0.3_f32,
                anger: 0.1_f32,
                fear: 0.1_f32,
                surprise: 0.1_f32,
            }
        }
    }

    /// Persist all mappings to disk
    pub fn save_mappings(&self, path: &str) -> Result<()> {
        use std::fs;
        use std::path::Path;

        // Create directory if needed
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }

        // Serialize mappings
        let mappings = serde_json::json!({
            "doc_to_sphere": self.doc_to_sphere.iter()
                .map(|(k, v)| (k.clone(), v.0.clone()))
                .collect::<HashMap<String, String>>(),
            "doc_to_toroidal": self.doc_to_toroidal.iter()
                .map(|(k, v)| (k.clone(), serde_json::json!({
                    "theta": v.theta,
                    "phi": v.phi,
                    "r": v.r
                })))
                .collect::<HashMap<String, serde_json::Value>>(),
            "doc_to_layer": self.doc_to_layer.iter()
                .map(|(k, v)| (k.clone(), format!("{:?}", v)))
                .collect::<HashMap<String, String>>(),
        });

        fs::write(path, serde_json::to_string_pretty(&mappings)?)?;
        tracing::info!("💾 Saved RAG-Memory mappings to {}", path);

        Ok(())
    }

    /// Load mappings from disk
    pub fn load_mappings(&mut self, path: &str) -> Result<()> {
        use std::fs;

        let content = fs::read_to_string(path)?;
        let mappings: serde_json::Value = serde_json::from_str(&content)?;

        // Load doc_to_sphere mappings
        if let Some(sphere_map) = mappings.get("doc_to_sphere").and_then(|v| v.as_object()) {
            for (doc_id, sphere_id) in sphere_map {
                if let Some(sid) = sphere_id.as_str() {
                    self.doc_to_sphere
                        .insert(doc_id.clone(), SphereId(sid.to_string()));
                }
            }
        }

        // Load doc_to_toroidal mappings
        if let Some(toroidal_map) = mappings.get("doc_to_toroidal").and_then(|v| v.as_object()) {
            for (doc_id, coord) in toroidal_map {
                if let Some(coord_obj) = coord.as_object() {
                    let theta = coord_obj
                        .get("theta")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let phi = coord_obj.get("phi").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let r = coord_obj.get("r").and_then(|v| v.as_f64()).unwrap_or(1.0);

                    self.doc_to_toroidal
                        .insert(doc_id.clone(), ToroidalCoordinate { theta, phi, r });
                }
            }
        }

        tracing::info!("✅ Loaded RAG-Memory mappings from {}", path);

        Ok(())
    }
}

impl Default for RagMemoryAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_to_memory_fragment() {
        let adapter = RagMemoryAdapter::new();

        let doc = Document {
            id: "test-1".to_string(),
            content: "Test memory content".to_string(),
            metadata: HashMap::new(),
            embedding: Some(vec![0.1, 0.2, 0.3]),
            created_at: Utc::now(),
            entities: vec![],
            chunk_id: None,
            source_type: Some("knowledge".to_string()),
            resonance_hint: Some(0.8),
            token_count: 3,
        };

        let fragment = adapter.document_to_memory_fragment(&doc).unwrap();

        assert_eq!(fragment.content, "Test memory content");
        assert_eq!(fragment.relevance, 0.8);
        assert_eq!(fragment.layer, MemoryLayer::Semantic);
    }

    #[test]
    fn test_embedding_to_toroidal_coordinate() {
        let adapter = RagMemoryAdapter::new();

        let embedding = vec![0.5, 0.7, 0.3, 0.9];
        let coord = adapter
            .embedding_to_toroidal_coordinate(&embedding)
            .unwrap();

        assert!(coord.theta >= 0.0 && coord.theta < 6.28318);
        assert!(coord.phi >= 0.0 && coord.phi < 6.28318);
    }

    #[test]
    fn test_valence_to_emotional_vector() {
        let adapter = RagMemoryAdapter::new();

        // Test positive valence
        let positive_emotion = adapter.valence_to_emotional_vector(0.8);
        assert!(positive_emotion.joy > 0.5);

        // Test negative valence
        let negative_emotion = adapter.valence_to_emotional_vector(-0.8);
        assert!(negative_emotion.sadness > 0.3);

        // Test neutral
        let neutral_emotion = adapter.valence_to_emotional_vector(0.0);
        assert!(neutral_emotion.joy > 0.0);
    }

    #[test]
    fn test_embedding_to_3d_position() {
        let adapter = RagMemoryAdapter::new();

        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let position = adapter.embedding_to_3d_position(&embedding).unwrap();

        assert_eq!(position[0], 0.1);
        assert_eq!(position[1], 0.2);
        assert_eq!(position[2], 0.3);
    }

    #[test]
    fn test_save_load_mappings() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let mapping_path = dir.path().join("test_mappings.json");

        let mut adapter = RagMemoryAdapter::new();
        adapter
            .doc_to_sphere
            .insert("doc1".to_string(), SphereId("sphere1".to_string()));
        adapter
            .doc_to_toroidal
            .insert("doc1".to_string(), ToroidalCoordinate::new(1.0, 2.0));

        // Save
        adapter
            .save_mappings(mapping_path.to_str().unwrap())
            .unwrap();

        // Load into new adapter
        let mut new_adapter = RagMemoryAdapter::new();
        new_adapter
            .load_mappings(mapping_path.to_str().unwrap())
            .unwrap();

        // Verify
        assert!(new_adapter.doc_to_sphere.contains_key("doc1"));
        assert!(new_adapter.doc_to_toroidal.contains_key("doc1"));
    }
}
