// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
// use tracing::{error, info, warn}; // Currently unused
// use crate::consciousness::EmotionType; // If used, keep; else remove if not needed
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Forward,
    Backward,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFragment {
    pub content: String,
    pub layer: MemoryLayer,
    pub relevance: f32,
    pub timestamp: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryLayer {
    CoreBurned,
    Procedural,
    Episodic,
    Semantic,
    Somatic,
    Working,
}

impl MemoryLayer {
    pub fn new(_name: &str) -> Self {
        // Mock, return a default layer for simplicity
        MemoryLayer::Semantic
    }

    pub fn search(&self, _query: &str) -> Vec<MemoryFragment> {
        // Mock search: return sample fragments
        vec![MemoryFragment {
            content: format!("Mock memory from {:?}", self),
            layer: *self,
            relevance: 0.5,
            timestamp: 0.0,
        }]
    }
}

pub struct MobiusMemorySystem {
    layers: Vec<MemoryLayer>,              // Array of 6 layers
    emotion_weights: HashMap<String, f32>, // Emotion string to weight
    temporal_connector: TemporalBridge,
    persistent_memories: Vec<MemoryFragment>, // Added persistent store
    max_memories: usize,                      // Maximum memory limit
}

impl Default for MobiusMemorySystem {
    fn default() -> Self {
        Self::new()
    }
}

impl MobiusMemorySystem {
    pub fn new() -> Self {
        let mut system = Self {
            layers: vec![
                MemoryLayer::CoreBurned,
                MemoryLayer::Procedural,
                MemoryLayer::Episodic,
                MemoryLayer::Semantic,
                MemoryLayer::Somatic,
                MemoryLayer::Working,
            ],
            emotion_weights: HashMap::from([
                ("joy".to_string(), 1.2),
                ("sadness".to_string(), 1.5),
                ("anger".to_string(), 1.7),
                ("fear".to_string(), 1.6),
                ("surprise".to_string(), 1.3),
                ("neutral".to_string(), 1.0),
            ]),
            temporal_connector: TemporalBridge::new(),
            persistent_memories: Vec::new(), // Added persistent store
            max_memories: 10000,             // Set maximum memory limit
        };
        system.load_persistent_memories();
        system
    }

    pub fn bi_directional_traverse(
        &mut self,
        query: &str,
        emotion_str: &str,
    ) -> Vec<MemoryFragment> {
        let forward_path = self.emotion_driven_traversal(query, emotion_str, Direction::Forward);
        let backward_path = self.emotion_driven_traversal(query, emotion_str, Direction::Backward);
        let connected = self
            .temporal_connector
            .connect_paths(forward_path, backward_path);

        // Bounded memory growth with LRU eviction
        if self.persistent_memories.len() + connected.len() > self.max_memories {
            let overflow = (self.persistent_memories.len() + connected.len()) - self.max_memories;
            self.persistent_memories.drain(0..overflow);
        }

        self.persistent_memories.extend(connected.clone());
        self.save_persistent_memories();
        connected
    }

    fn emotion_driven_traversal(
        &self,
        query: &str,
        emotion: &str,
        direction: Direction,
    ) -> Vec<MemoryFragment> {
        let mut path = Vec::new();
        let mut current_layer_idx = 3; // Start at semantic layer (index 3, present)
        let depth_weight = self.emotion_weights.get(emotion).copied().unwrap_or(1.0);

        for _ in 0..5 {
            // Traverse 5 layers
            if current_layer_idx < self.layers.len() {
                let layer = &self.layers[current_layer_idx];
                let mut results = layer.search(query);

                // Apply emotion-based weighting
                let layer_weight = 1.0 + (current_layer_idx as f32 * 0.2) * depth_weight;
                for memory in &mut results {
                    memory.relevance *= layer_weight;
                }

                path.extend(results);
            }

            // Move to next layer based on direction (Möbius loop: 0-5)
            current_layer_idx = match direction {
                Direction::Forward => (current_layer_idx + 1) % 6,
                Direction::Backward => {
                    if current_layer_idx == 0 {
                        5
                    } else {
                        current_layer_idx - 1
                    }
                }
            };
        }

        path
    }

    fn load_persistent_memories(&mut self) {
        let path = "./data/mobius_memory.json";
        if Path::new(path).exists() {
            if let Ok(file) = File::open(path) {
                let reader = BufReader::new(file);
                if let Ok(memories) = serde_json::from_reader::<_, Vec<MemoryFragment>>(reader) {
                    self.persistent_memories = memories;
                }
            }
        }
    }

    fn save_persistent_memories(&self) {
        // Create directory first if needed
        if let Err(e) = std::fs::create_dir_all("./data") {
            tracing::info!("Failed to create data directory: {}", e);
            return;
        }

        let path = "./data/mobius_memory.json";
        let temp_path = format!("{}.tmp", path);

        // Write to temp file first for atomic operation
        match OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)
        {
            Ok(file) => {
                let writer = BufWriter::new(file);
                if let Err(e) = serde_json::to_writer(writer, &self.persistent_memories) {
                    tracing::info!("Failed to save memories: {}", e);
                    tracing::error!("Failed to save memories: {}", e);
                    return;
                }

                // Atomic rename
                if let Err(e) = std::fs::rename(&temp_path, path) {
                    tracing::info!("Failed to finalize save: {}", e);
                    tracing::error!("Failed to finalize save: {}", e);
                }
            }
            Err(e) => {
                tracing::info!("Failed to open temp file for saving: {}", e);
                tracing::error!("Failed to open temp file for saving: {}", e);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TemporalBridge {
    // Mock quantum entangler
    entanglement_strength: f32,
}

impl Default for TemporalBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalBridge {
    pub fn new() -> Self {
        Self {
            entanglement_strength: 0.9,
        }
    }

    pub fn connect_paths(
        &self,
        forward: Vec<MemoryFragment>,
        backward: Vec<MemoryFragment>,
    ) -> Vec<MemoryFragment> {
        let mut connected = forward;
        // Mock connection: add backward with entanglement adjustment
        for mut mem in backward {
            mem.relevance *= self.entanglement_strength;
            mem.content = format!("Entangled: {}", mem.content);
            connected.push(mem);
        }
        connected
    }
}
