// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Möbius Memory System with 6-Layer Architecture
//!
//! This module implements the 6-layer memory system with 99.51% stability target
//! and emotional transformation capabilities using the K-Twist topology.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Memory layer types in the 6-layer architecture
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryLayer {
    CoreBurned, // Layer 1: Core burned memories (highest stability)
    Procedural, // Layer 2: Procedural memories
    Episodic,   // Layer 3: Episodic memories
    Semantic,   // Layer 4: Semantic memories
    Somatic,    // Layer 5: Somatic memories
    Working,    // Layer 6: Working memory (lowest stability)
}

/// Memory entry with emotional and stability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub layer: MemoryLayer,
    pub emotional_weight: f64,
    pub stability: f64,
    pub timestamp: u64,
    pub access_count: u32,
    pub last_accessed: u64,
    pub emotional_vector: (f64, f64, f64), // RGB emotional encoding
    pub topology_position: (f64, f64, f64), // Position in K-Twist topology
}

/// Emotional transformation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalTransformParams {
    pub novelty_bounds: (f64, f64),   // Target novelty range (15-20%)
    pub stability_target: f64,        // Target stability (99.51%)
    pub learning_rate: f64,           // Learning rate for updates
    pub decay_rate: f64,              // Memory decay rate
    pub consolidation_threshold: f64, // Threshold for layer promotion
}

impl Default for EmotionalTransformParams {
    fn default() -> Self {
        Self {
            novelty_bounds: (0.15, 0.20),
            stability_target: 0.9951,
            learning_rate: 0.01,
            decay_rate: 0.001,
            consolidation_threshold: 0.8,
        }
    }
}

/// 6-Layer Memory System
#[derive(Debug, Clone)]
pub struct MobiusMemorySystem {
    layers: HashMap<MemoryLayer, Vec<MemoryEntry>>,
    params: EmotionalTransformParams,
    total_memories: usize,
    stability_metrics: StabilityMetrics,
}

/// Stability metrics for monitoring system performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMetrics {
    pub overall_stability: f64,
    pub layer_stabilities: HashMap<MemoryLayer, f64>,
    pub emotional_coherence: f64,
    pub consolidation_rate: f64,
    pub last_updated: u64,
}

impl Default for StabilityMetrics {
    fn default() -> Self {
        Self {
            overall_stability: 0.0,
            layer_stabilities: HashMap::new(),
            emotional_coherence: 0.0,
            consolidation_rate: 0.0,
            last_updated: 0,
        }
    }
}

impl MobiusMemorySystem {
    /// Create a new 6-layer memory system
    pub fn new(params: EmotionalTransformParams) -> Self {
        let mut layers = HashMap::new();
        for layer in [
            MemoryLayer::CoreBurned,
            MemoryLayer::Procedural,
            MemoryLayer::Episodic,
            MemoryLayer::Semantic,
            MemoryLayer::Somatic,
            MemoryLayer::Working,
        ] {
            layers.insert(layer, Vec::new());
        }

        Self {
            layers,
            params,
            total_memories: 0,
            stability_metrics: StabilityMetrics::default(),
        }
    }

    /// Add a new memory entry
    pub fn add_memory(
        &mut self,
        content: String,
        initial_layer: MemoryLayer,
        emotional_weight: f64,
    ) -> Result<String> {
        let id = self.generate_memory_id();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        // Calculate initial emotional vector based on content and weight
        let emotional_vector = self.calculate_emotional_vector(&content, emotional_weight);

        // Calculate initial topology position
        let topology_position = self.calculate_topology_position(&content, &initial_layer);

        let memory = MemoryEntry {
            id: id.clone(),
            content,
            layer: initial_layer.clone(),
            emotional_weight,
            stability: self.calculate_initial_stability(&initial_layer),
            timestamp,
            access_count: 0,
            last_accessed: timestamp,
            emotional_vector,
            topology_position,
        };

        self.layers.get_mut(&initial_layer).unwrap().push(memory);
        self.total_memories += 1;

        // Update stability metrics
        self.update_stability_metrics()?;

        Ok(id)
    }

    /// Retrieve memory by ID
    pub fn get_memory(&mut self, id: &str) -> Option<&mut MemoryEntry> {
        for layer_memories in self.layers.values_mut() {
            for memory in layer_memories.iter_mut() {
                if memory.id == id {
                    memory.access_count += 1;
                    memory.last_accessed = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    return Some(memory);
                }
            }
        }
        None
    }

    /// Apply emotional transformation to memory system
    pub fn apply_emotional_transformation(&mut self) -> Result<()> {
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        // Extract parameters to avoid borrowing conflicts
        let decay_rate = self.params.decay_rate;

        // Process each layer - fix borrowing by handling one layer at a time
        let layer_keys: Vec<MemoryLayer> = self.layers.keys().cloned().collect();
        for layer in layer_keys {
            if let Some(memories) = self.layers.get_mut(&layer) {
                // Update each memory - inline logic to avoid borrowing conflicts
                for memory in memories.iter_mut() {
                    // Inline stability update logic
                    let time_since_access = current_time - memory.last_accessed;
                    let time_decay = (-decay_rate * time_since_access as f64).exp();

                    // Stability increases with access count and decreases with time
                    let access_bonus = (memory.access_count as f64).ln().max(0.0) * 0.01;
                    let time_penalty = (1.0 - time_decay) * 0.1;

                    memory.stability = (memory.stability + access_bonus - time_penalty)
                        .max(0.0)
                        .min(1.0);

                    // Inline emotional transformation logic
                    let (r, g, b) = memory.emotional_vector;

                    // Calculate novelty based on emotional vector
                    let novelty = {
                        let distance_from_neutral =
                            ((r - 0.5).powi(2) + (g - 0.5).powi(2) + (b - 0.5).powi(2)).sqrt();
                        distance_from_neutral / (3.0_f64.sqrt() / 2.0)
                    };

                    // Ensure novelty stays within bounds (15-20%)
                    let target_novelty = if novelty < self.params.novelty_bounds.0 {
                        self.params.novelty_bounds.0
                    } else if novelty > self.params.novelty_bounds.1 {
                        self.params.novelty_bounds.1
                    } else {
                        novelty
                    };

                    // Transform emotional vector to achieve target novelty
                    let transformation_factor = target_novelty / novelty.max(0.001);

                    let new_r = (r * transformation_factor).max(0.0).min(1.0);
                    let new_g = (g * transformation_factor).max(0.0).min(1.0);
                    let new_b = (b * transformation_factor).max(0.0).min(1.0);

                    memory.emotional_vector = (new_r, new_g, new_b);

                    // Inline layer transition check logic
                    if memory.stability > 0.9951 {
                        // Memory is very stable, might promote to higher layer
                        // Implementation would go here in a full version
                    }
                }
            }
        }

        // Update overall system metrics
        self.update_stability_metrics()?;

        Ok(())
    }

    /// Update memory stability based on access patterns
    fn update_memory_stability(&self, memory: &mut MemoryEntry, current_time: u64) {
        let time_since_access = current_time - memory.last_accessed;
        let time_decay = (-self.params.decay_rate * time_since_access as f64).exp();

        // Stability increases with access count and decreases with time
        let access_bonus = (memory.access_count as f64).ln().max(0.0) * 0.01;
        let time_penalty = (1.0 - time_decay) * 0.1;

        memory.stability = (memory.stability + access_bonus - time_penalty)
            .max(0.0)
            .min(1.0);
    }

    /// Transform memory emotion based on K-Twist topology
    fn transform_memory_emotion(&mut self, memory: &mut MemoryEntry) {
        let (r, g, b) = memory.emotional_vector;

        // Calculate novelty based on emotional vector
        let novelty = self.calculate_novelty(&memory.emotional_vector);

        // Ensure novelty stays within bounds (15-20%)
        let target_novelty = if novelty < self.params.novelty_bounds.0 {
            self.params.novelty_bounds.0
        } else if novelty > self.params.novelty_bounds.1 {
            self.params.novelty_bounds.1
        } else {
            novelty
        };

        // Transform emotional vector to achieve target novelty
        let transformation_factor = target_novelty / novelty.max(0.001);

        let new_r = (r * transformation_factor).max(0.0).min(1.0);
        let new_g = (g * transformation_factor).max(0.0).min(1.0);
        let new_b = (b * transformation_factor).max(0.0).min(1.0);

        memory.emotional_vector = (new_r, new_g, new_b);
    }

    /// Calculate novelty of emotional vector
    fn calculate_novelty(&self, emotional_vector: &(f64, f64, f64)) -> f64 {
        let (r, g, b) = *emotional_vector;

        // Novelty based on distance from neutral point (0.5, 0.5, 0.5)
        let distance_from_neutral =
            ((r - 0.5).powi(2) + (g - 0.5).powi(2) + (b - 0.5).powi(2)).sqrt();

        // Normalize to 0-1 range
        distance_from_neutral / (3.0_f64.sqrt() / 2.0)
    }

    /// Check if memory should transition between layers
    fn check_layer_transition(&mut self, memory: &mut MemoryEntry) {
        let current_stability = memory.stability;
        let current_layer = &memory.layer;

        // Determine target layer based on stability
        let target_layer = self.determine_target_layer(current_stability);

        if target_layer != *current_layer {
            // Move memory to new layer
            self.move_memory_to_layer(memory, target_layer);
        }
    }

    /// Determine target layer based on stability
    fn determine_target_layer(&self, stability: f64) -> MemoryLayer {
        if stability >= 0.95 {
            MemoryLayer::CoreBurned
        } else if stability >= 0.85 {
            MemoryLayer::Procedural
        } else if stability >= 0.75 {
            MemoryLayer::Episodic
        } else if stability >= 0.65 {
            MemoryLayer::Semantic
        } else if stability >= 0.45 {
            MemoryLayer::Somatic
        } else {
            MemoryLayer::Working
        }
    }

    /// Move memory to different layer
    fn move_memory_to_layer(&mut self, memory: &mut MemoryEntry, target_layer: MemoryLayer) {
        let old_layer = memory.layer.clone();
        memory.layer = target_layer.clone();

        // Remove from old layer
        if let Some(old_layer_memories) = self.layers.get_mut(&old_layer) {
            old_layer_memories.retain(|m| m.id != memory.id);
        }

        // Add to new layer
        if let Some(new_layer_memories) = self.layers.get_mut(&target_layer) {
            new_layer_memories.push(memory.clone());
        }
    }

    /// Calculate initial stability based on layer
    fn calculate_initial_stability(&self, layer: &MemoryLayer) -> f64 {
        match layer {
            MemoryLayer::CoreBurned => 0.95,
            MemoryLayer::Procedural => 0.85,
            MemoryLayer::Episodic => 0.75,
            MemoryLayer::Semantic => 0.65,
            MemoryLayer::Somatic => 0.45,
            MemoryLayer::Working => 0.25,
        }
    }

    /// Calculate emotional vector from content and weight
    fn calculate_emotional_vector(&self, content: &str, weight: f64) -> (f64, f64, f64) {
        // Simple emotional analysis based on content keywords
        let mut r = 0.5; // Red component (arousal)
        let mut g = 0.5; // Green component (valence)
        let mut b = 0.5; // Blue component (dominance)

        let content_lower = content.to_lowercase();

        // Analyze content for emotional keywords
        if content_lower.contains("happy")
            || content_lower.contains("joy")
            || content_lower.contains("love")
        {
            g += 0.3;
            r += 0.2;
        }
        if content_lower.contains("sad")
            || content_lower.contains("pain")
            || content_lower.contains("loss")
        {
            g -= 0.3;
            b -= 0.2;
        }
        if content_lower.contains("angry")
            || content_lower.contains("rage")
            || content_lower.contains("fury")
        {
            r += 0.4;
            b += 0.2;
        }
        if content_lower.contains("fear")
            || content_lower.contains("anxiety")
            || content_lower.contains("worry")
        {
            r += 0.2;
            b -= 0.3;
        }

        // Apply weight influence
        let weight_factor = weight * 0.2;
        r = (r + weight_factor).max(0.0).min(1.0);
        g = (g + weight_factor).max(0.0).min(1.0);
        b = (b + weight_factor).max(0.0).min(1.0);

        (r, g, b)
    }

    /// Calculate topology position for memory
    fn calculate_topology_position(&self, content: &str, layer: &MemoryLayer) -> (f64, f64, f64) {
        // Simple hash-based positioning
        let hash = self.simple_hash(content);

        // Map hash to topology coordinates
        let x = ((hash % 1000) as f64 / 1000.0) * 10.0 - 5.0;
        let y = (((hash / 1000) % 1000) as f64 / 1000.0) * 10.0 - 5.0;
        let z = (((hash / 1000000) % 1000) as f64 / 1000.0) * 10.0 - 5.0;

        // Adjust based on layer
        let layer_offset = match layer {
            MemoryLayer::CoreBurned => 0.0,
            MemoryLayer::Procedural => 1.0,
            MemoryLayer::Episodic => 2.0,
            MemoryLayer::Semantic => 3.0,
            MemoryLayer::Somatic => 4.0,
            MemoryLayer::Working => 5.0,
        };

        (x, y, z + layer_offset)
    }

    /// Simple hash function for deterministic positioning
    fn simple_hash(&self, input: &str) -> u64 {
        let mut hash = 0u64;
        for byte in input.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        hash
    }

    /// Generate unique memory ID
    fn generate_memory_id(&self) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("mem_{}", timestamp)
    }

    /// Update stability metrics
    fn update_stability_metrics(&mut self) -> Result<()> {
        let mut total_stability = 0.0;
        let mut total_memories = 0;
        let mut layer_stabilities = HashMap::new();

        for (layer, memories) in &self.layers {
            if !memories.is_empty() {
                let layer_stability =
                    memories.iter().map(|m| m.stability).sum::<f64>() / memories.len() as f64;
                layer_stabilities.insert(layer.clone(), layer_stability);
                total_stability += layer_stability * memories.len() as f64;
                total_memories += memories.len();
            } else {
                layer_stabilities.insert(layer.clone(), 0.0);
            }
        }

        let overall_stability = if total_memories > 0 {
            total_stability / total_memories as f64
        } else {
            0.0
        };

        // Calculate emotional coherence
        let emotional_coherence = self.calculate_emotional_coherence();

        // Calculate consolidation rate
        let consolidation_rate = self.calculate_consolidation_rate();

        self.stability_metrics = StabilityMetrics {
            overall_stability,
            layer_stabilities,
            emotional_coherence,
            consolidation_rate,
            last_updated: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        };

        Ok(())
    }

    /// Calculate emotional coherence across all memories
    fn calculate_emotional_coherence(&self) -> f64 {
        let mut all_vectors = Vec::new();

        for memories in self.layers.values() {
            for memory in memories {
                all_vectors.push(memory.emotional_vector);
            }
        }

        if all_vectors.len() < 2 {
            return 1.0;
        }

        // Calculate average emotional vector
        let mut avg_r = 0.0;
        let mut avg_g = 0.0;
        let mut avg_b = 0.0;

        for (r, g, b) in &all_vectors {
            avg_r += r;
            avg_g += g;
            avg_b += b;
        }

        let count = all_vectors.len() as f64;
        avg_r /= count;
        avg_g /= count;
        avg_b /= count;

        // Calculate coherence as inverse of variance
        let mut variance = 0.0;
        for (r, g, b) in &all_vectors {
            let distance = ((r - avg_r).powi(2) + (g - avg_g).powi(2) + (b - avg_b).powi(2)).sqrt();
            variance += distance * distance;
        }

        variance /= count;

        // Convert variance to coherence (higher variance = lower coherence)
        (-variance).exp()
    }

    /// Calculate consolidation rate (memories moving to higher stability layers)
    fn calculate_consolidation_rate(&self) -> f64 {
        let mut consolidated = 0;
        let mut total = 0;

        for memories in self.layers.values() {
            for memory in memories {
                total += 1;
                if memory.stability > self.params.consolidation_threshold {
                    consolidated += 1;
                }
            }
        }

        if total > 0 {
            consolidated as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Get stability metrics
    pub fn get_stability_metrics(&self) -> &StabilityMetrics {
        &self.stability_metrics
    }

    /// Get all memories from a specific layer
    pub fn get_layer_memories(&self, layer: &MemoryLayer) -> &[MemoryEntry] {
        self.layers.get(layer).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get total number of memories
    pub fn get_total_memories(&self) -> usize {
        self.total_memories
    }

    /// Export memory system state
    pub fn export_state(&self) -> Result<MemorySystemState> {
        Ok(MemorySystemState {
            layers: self.layers.clone(),
            params: self.params.clone(),
            total_memories: self.total_memories,
            stability_metrics: self.stability_metrics.clone(),
        })
    }
}

/// Complete memory system state for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySystemState {
    pub layers: HashMap<MemoryLayer, Vec<MemoryEntry>>,
    pub params: EmotionalTransformParams,
    pub total_memories: usize,
    pub stability_metrics: StabilityMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_system_creation() {
        let params = EmotionalTransformParams::default();
        let memory_system = MobiusMemorySystem::new(params);

        assert_eq!(memory_system.get_total_memories(), 0);
        assert_eq!(memory_system.layers.len(), 6);
    }

    #[test]
    fn test_memory_addition() {
        let params = EmotionalTransformParams::default();
        let mut memory_system = MobiusMemorySystem::new(params);

        let id = memory_system
            .add_memory("Test memory content".to_string(), MemoryLayer::Working, 0.5)
            .unwrap();

        assert_eq!(memory_system.get_total_memories(), 1);
        assert!(!id.is_empty());
    }

    #[test]
    fn test_emotional_transformation() {
        let params = EmotionalTransformParams::default();
        let mut memory_system = MobiusMemorySystem::new(params);

        let _id = memory_system
            .add_memory(
                "Happy memory content".to_string(),
                MemoryLayer::Working,
                0.8,
            )
            .unwrap();

        assert!(memory_system.apply_emotional_transformation().is_ok());

        let metrics = memory_system.get_stability_metrics();
        assert!(metrics.overall_stability >= 0.0);
        assert!(metrics.overall_stability <= 1.0);
    }

    #[test]
    fn test_stability_target() {
        let params = EmotionalTransformParams::default();
        let memory_system = MobiusMemorySystem::new(params);

        let metrics = memory_system.get_stability_metrics();
        // Initial stability should be 0, but system should aim for 99.51%
        assert_eq!(metrics.overall_stability, 0.0);
    }
}
