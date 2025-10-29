//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Möbius Memory System with 6-Layer Architecture
//!
//! This module implements the 6-layer memory system with 99.51% stability target
//! and emotional transformation capabilities using the K-Twist topology.

use crate::config::{AppConfig, ConsciousnessConfig};

pub mod consolidation;
pub mod guessing_spheres;
pub mod mobius;
pub mod multi_layer_query;
pub mod toroidal;

// Re-export commonly used types
pub use guessing_spheres::{EmotionalVector, GuessingMemorySystem, SphereId, TraversalDirection};

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
        let config = ConsciousnessConfig::default();
        let entropy = config.memory.default_entropy;
        Self {
            novelty_bounds: (
                config.novelty_threshold_min * (1.0 - entropy),
                config.novelty_threshold_max * (1.0 + entropy),
            ),
            stability_target: config.ethical_bounds * config.memory.stability_target_multiplier,
            learning_rate: config.consciousness_step_size * config.memory.learning_rate_multiplier,
            decay_rate: config.emotional_plasticity * config.memory.decay_rate_multiplier,
            consolidation_threshold: (config.memory_threshold as f64)
                * config.memory.consolidation_threshold_multiplier,
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

impl MobiusMemorySystem {
    /// Create a new 6-layer memory system
    pub fn new(config: &AppConfig) -> Self {
        let config_cons = &config.consciousness;
        let params = EmotionalTransformParams {
            novelty_bounds: (
                config_cons.novelty_threshold_min,
                config_cons.novelty_threshold_max,
            ),
            stability_target: config_cons.ethical_bounds
                * config.memory.stability_target_multiplier,
            learning_rate: config_cons.consciousness_step_size
                * config.memory.learning_rate_multiplier,
            decay_rate: config_cons.emotional_plasticity * config.memory.decay_rate_multiplier,
            consolidation_threshold: (config_cons.memory_threshold as f64)
                * config.memory.consolidation_threshold_multiplier,
        };

        let mut layers = HashMap::with_capacity(
            (config_cons.memory_threshold * config.memory.layer_capacity_base_multiplier) as usize,
        );
        for layer in [
            MemoryLayer::CoreBurned,
            MemoryLayer::Procedural,
            MemoryLayer::Episodic,
            MemoryLayer::Semantic,
            MemoryLayer::Somatic,
            MemoryLayer::Working,
        ] {
            layers.insert(
                layer,
                Vec::with_capacity(
                    (config_cons.memory_threshold
                        * config.memory.layer_capacity_per_layer_multiplier)
                        as usize,
                ),
            );
        }

        Self {
            layers,
            params,
            total_memories: 0,
            stability_metrics: StabilityMetrics {
                overall_stability: 0.0,
                layer_stabilities: HashMap::with_capacity(6),
                emotional_coherence: 0.0,
                consolidation_rate: 0.0,
                last_updated: 0,
            },
        }
    }

    /// Add a new memory entry
    pub fn add_memory(
        &mut self,
        content: String,
        initial_layer: MemoryLayer,
        emotional_weight: f64,
        config: &AppConfig,
    ) -> Result<String> {
        let id = self.generate_memory_id();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        // Calculate initial emotional vector based on content and weight
        let emotional_vector = self.calculate_emotional_vector(&content, emotional_weight, config);

        // Calculate initial topology position
        let topology_position = self.calculate_topology_position(&content, &initial_layer, config);

        let memory = MemoryEntry {
            id: id.clone(),
            content,
            layer: initial_layer.clone(),
            emotional_weight,
            stability: self.calculate_initial_stability(&initial_layer, config),
            timestamp,
            access_count: 0,
            last_accessed: timestamp,
            emotional_vector,
            topology_position,
        };

        self.layers.get_mut(&initial_layer).unwrap().push(memory);
        self.total_memories += 1;

        // Update stability metrics
        self.update_stability_metrics(config)?;

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
    pub fn apply_emotional_transformation(&mut self, config: &AppConfig) -> Result<()> {
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        // Möbius Flip Innovation: Memory empathy - detect frustration and prioritize joy
        let current_coherence = self.calculate_emotional_coherence(config);
        if current_coherence < 0.5 {
            // Add frustration_threshold to config if needed
            tracing::info!(
                "Detecting frustration patterns (coherence: {}), applying joy prioritization flip",
                current_coherence
            );
            for (_layer, memories) in self.layers.iter_mut() {
                for memory in memories.iter_mut() {
                    let (_, g, _) = memory.emotional_vector; // valence green
                    if g > config.memory.emotional_neutral {
                        // joy memories
                        // Reduce decay and boost stability for joy memories
                        memory.stability =
                            (memory.stability * config.memory.joy_stability_boost).min(1.0);
                        // Flip retention strategy: slower decay for positive memories
                        // This would integrate with Gaussian decay model for faster convergence
                    }
                }
            }
        }

        // Process each layer
        for (_layer, memories) in self.layers.iter_mut() {
            for memory in memories.iter_mut() {
                memory.last_accessed = current_time;

                // Update stability based on access patterns and time
                Self::update_memory_stability_static(memory, current_time, &config.consciousness);

                // Apply emotional transformation
                Self::transform_memory_emotion_static(memory, config);

                // Check for layer promotion/demotion
                Self::check_layer_transition_static(memory, config);
            }
        }

        // Update overall system metrics
        self.update_stability_metrics(config)?;

        Ok(())
    }

    /// Update memory stability based on access patterns
    #[allow(dead_code)]
    fn update_memory_stability(
        &self,
        memory: &mut MemoryEntry,
        current_time: u64,
        config: &ConsciousnessConfig,
    ) {
        let time_since_access = current_time - memory.last_accessed;
        let time_decay = (-config.emotional_plasticity * time_since_access as f64
            / config.memory.time_decay_divisor)
            .exp();
        let access_bonus = (memory.access_count as f64).ln().max(0.0)
            * config.consciousness_step_size
            * config.memory.access_bonus_stability_multiplier;
        let time_penalty = (1.0 - time_decay) as f32 * config.memory_threshold;
        memory.stability = (memory.stability + access_bonus - time_penalty as f64).clamp(0.0, 1.0);
    }

    /// Update memory stability (static version)
    fn update_memory_stability_static(
        memory: &mut MemoryEntry,
        current_time: u64,
        config: &ConsciousnessConfig,
    ) {
        let time_since_access = current_time - memory.last_accessed;
        let time_decay = (-config.emotional_plasticity * time_since_access as f64
            / config.memory.time_decay_divisor)
            .exp();
        let access_bonus = (memory.access_count as f64).ln().max(0.0)
            * config.consciousness_step_size
            * config.memory.access_bonus_stability_multiplier;
        let time_penalty = (1.0 - time_decay) as f32 * config.memory_threshold;
        memory.stability = (memory.stability + access_bonus - time_penalty as f64).clamp(0.0, 1.0);
    }

    /// Transform memory emotion based on K-Twist topology
    fn transform_memory_emotion_static(memory: &mut MemoryEntry, config: &AppConfig) {
        let (r, g, b) = memory.emotional_vector;

        // Calculate novelty based on emotional vector
        let novelty = Self::calculate_novelty_static(&memory.emotional_vector, config);

        // Ensure novelty stays within bounds
        let target_novelty = if novelty < config.memory.novelty_bounds_min {
            config.memory.novelty_bounds_min
        } else if novelty > config.memory.novelty_bounds_max {
            config.memory.novelty_bounds_max
        } else {
            novelty
        };

        // Transform emotional vector to achieve target novelty
        let transformation_factor =
            target_novelty / novelty.max(config.memory.transformation_min_novelty_div);

        let new_r = (r * transformation_factor).max(0.0).min(1.0);
        let new_g = (g * transformation_factor).max(0.0).min(1.0);
        let new_b = (b * transformation_factor).max(0.0).min(1.0);

        memory.emotional_vector = (new_r, new_g, new_b);
    }

    /// Calculate novelty of emotional vector
    fn calculate_novelty_static(emotional_vector: &(f64, f64, f64), config: &AppConfig) -> f64 {
        let entropy = config.memory.default_entropy;
        let (r, g, b) = *emotional_vector;
        let distance_from_neutral = ((r - config.consciousness.default_authenticity).powi(2)
            + (g - config.memory.emotional_neutral_g).powi(2)
            + (b - config.memory.emotional_neutral_b).powi(2))
        .sqrt();
        distance_from_neutral * (1.0 + entropy * config.consciousness.emotional_plasticity)
    }

    /// Check if memory should transition between layers
    #[allow(dead_code)]
    fn check_layer_transition(&mut self, memory: &mut MemoryEntry, config: &AppConfig) {
        let current_stability = memory.stability;
        let current_layer = &memory.layer;

        let target_layer = Self::determine_target_layer_static(current_stability, config);

        if target_layer != *current_layer {
            // TODO: Implement move_memory_to_layer functionality
        }
    }

    /// Check if memory should transition between layers (static version)
    fn check_layer_transition_static(memory: &mut MemoryEntry, config: &AppConfig) {
        let current_stability = memory.stability;
        let current_layer = &memory.layer;

        let target_layer = Self::determine_target_layer_static(current_stability, config);

        if target_layer != *current_layer {
            // Update layer assignment
            memory.layer = target_layer;
        }
    }

    /// Determine target layer based on stability
    fn determine_target_layer_static(stability: f64, config: &AppConfig) -> MemoryLayer {
        if stability >= config.memory.layer_stability_core {
            MemoryLayer::CoreBurned
        } else if stability >= config.memory.layer_stability_procedural {
            MemoryLayer::Procedural
        } else if stability >= config.memory.layer_stability_episodic {
            MemoryLayer::Episodic
        } else if stability >= config.memory.layer_stability_semantic {
            MemoryLayer::Semantic
        } else if stability >= config.memory.layer_stability_somatic {
            MemoryLayer::Somatic
        } else {
            MemoryLayer::Working
        }
    }

    /// Move memory to different layer
    #[allow(dead_code)]
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
    fn calculate_initial_stability(&self, layer: &MemoryLayer, config: &AppConfig) -> f64 {
        match layer {
            MemoryLayer::CoreBurned => config.memory.layer_stability_core,
            MemoryLayer::Procedural => config.memory.layer_stability_procedural,
            MemoryLayer::Episodic => config.memory.layer_stability_episodic,
            MemoryLayer::Semantic => config.memory.layer_stability_semantic,
            MemoryLayer::Somatic => config.memory.layer_stability_somatic,
            MemoryLayer::Working => config.memory.layer_stability_working,
        }
    }

    /// Calculate emotional vector from content and weight
    fn calculate_emotional_vector(
        &self,
        content: &str,
        weight: f64,
        config: &AppConfig,
    ) -> (f64, f64, f64) {
        let entropy = self.simple_hash(content, config) as f64 / u64::MAX as f64; // Entropy proxy
        let emotional_pattern = entropy * config.consciousness.emotional_intensity_factor;
        (
            emotional_pattern * weight,
            (1.0 - emotional_pattern) * weight,
            entropy * weight,
        )
    }

    /// Calculate topology position for memory
    fn calculate_topology_position(
        &self,
        content: &str,
        layer: &MemoryLayer,
        config: &AppConfig,
    ) -> (f64, f64, f64) {
        // Simple hash-based positioning
        let hash = self.simple_hash(content, config);

        // Map hash to topology coordinates
        let x = ((hash % config.memory.hash_mod) as f64 / (config.memory.hash_mod as f64))
            * config.memory.topology_scale
            - config.memory.topology_offset_scale;
        let y = (((hash / config.memory.hash_div1) % config.memory.hash_mod) as f64
            / (config.memory.hash_mod as f64))
            * config.memory.topology_scale
            - config.memory.topology_offset_scale;
        let z = (((hash / config.memory.hash_div2) % config.memory.hash_mod) as f64
            / (config.memory.hash_mod as f64))
            * config.memory.topology_scale
            - config.memory.topology_offset_scale;

        // Adjust based on layer
        let layer_offset = match layer {
            MemoryLayer::CoreBurned => config.memory.layer_offsets[0],
            MemoryLayer::Procedural => config.memory.layer_offsets[1],
            MemoryLayer::Episodic => config.memory.layer_offsets[2],
            MemoryLayer::Semantic => config.memory.layer_offsets[3],
            MemoryLayer::Somatic => config.memory.layer_offsets[4],
            MemoryLayer::Working => config.memory.layer_offsets[5],
        };

        (x, y, z + layer_offset)
    }

    /// Simple hash function for deterministic positioning
    fn simple_hash(&self, content: &str, config: &AppConfig) -> u64 {
        let mut hash = 0u64;
        for byte in content.bytes() {
            hash = hash
                .wrapping_mul(config.memory.hash_multiplier)
                .wrapping_add(byte as u64);
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
    fn update_stability_metrics(&mut self, config: &AppConfig) -> Result<()> {
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
        let emotional_coherence = self.calculate_emotional_coherence(config);

        // Calculate consolidation rate
        let consolidation_rate = self.calculate_consolidation_rate(config);

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
    fn calculate_emotional_coherence(&self, config: &AppConfig) -> f64 {
        let mut all_vectors = Vec::new();

        for memories in self.layers.values() {
            for memory in memories {
                all_vectors.push(memory.emotional_vector);
            }
        }

        if all_vectors.len() < config.memory.emotional_coherence_min_vectors {
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

        // Calculate variance
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
    fn calculate_consolidation_rate(&self, config: &AppConfig) -> f64 {
        let mut consolidated = 0;
        let mut total = 0;

        for memories in self.layers.values() {
            for memory in memories {
                total += 1;
                if memory.stability > config.memory.consolidation_threshold {
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
    use crate::config::AppConfig;

    #[test]
    fn test_memory_system_creation() {
        let config = AppConfig::default();
        let memory_system = MobiusMemorySystem::new(&config);

        assert_eq!(memory_system.get_total_memories(), 0);
        assert_eq!(memory_system.layers.len(), 6);
    }

    #[test]
    fn test_memory_addition() {
        let config = AppConfig::default();
        let mut memory_system = MobiusMemorySystem::new(&config);

        let id = memory_system
            .add_memory(
                "Test memory content".to_string(),
                MemoryLayer::Working,
                0.5,
                &config,
            )
            .unwrap();

        assert_eq!(memory_system.get_total_memories(), 1);
        assert!(!id.is_empty());
    }

    #[test]
    fn test_emotional_transformation() {
        let config = AppConfig::default();
        let mut memory_system = MobiusMemorySystem::new(&config);

        let _id = memory_system
            .add_memory(
                "Happy memory content".to_string(),
                MemoryLayer::Working,
                0.8,
                &config,
            )
            .unwrap();

        assert!(memory_system
            .apply_emotional_transformation(&config)
            .is_ok());

        let metrics = memory_system.get_stability_metrics();
        assert!(metrics.overall_stability >= 0.0);
        assert!(metrics.overall_stability <= 1.0);
    }

    #[test]
    fn test_stability_target() {
        let config = AppConfig::default();
        let memory_system = MobiusMemorySystem::new(&config);

        let metrics = memory_system.get_stability_metrics();
        // Initial stability should be 0, but system should aim for config stability_target
        assert_eq!(metrics.overall_stability, 0.0);
    }
}
