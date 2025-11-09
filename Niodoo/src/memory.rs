use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

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
pub struct Memory {
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

impl Memory {
    pub fn new(content: String, layer: MemoryLayer) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: Uuid::new_v4().to_string(),
            content,
            layer,
            emotional_weight: 0.5,
            stability: match layer {
                MemoryLayer::CoreBurned => 0.99,
                MemoryLayer::Procedural => 0.85,
                MemoryLayer::Episodic => 0.70,
                MemoryLayer::Semantic => 0.60,
                MemoryLayer::Somatic => 0.45,
                MemoryLayer::Working => 0.30,
            },
            timestamp,
            access_count: 0,
            last_accessed: timestamp,
            emotional_vector: (0.0, 0.0, 0.0),
            topology_position: (0.0, 0.0, 0.0),
        }
    }

    pub fn access(&mut self) {
        self.access_count += 1;
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
}

/// Memory system for storing and retrieving memories
pub struct MemorySystem {
    memories: HashMap<String, Memory>,
}

impl MemorySystem {
    pub fn new() -> Self {
        Self {
            memories: HashMap::new(),
        }
    }

    pub fn store(&mut self, memory: Memory) {
        self.memories.insert(memory.id.clone(), memory);
    }

    pub fn retrieve(&mut self, id: &str) -> Option<&mut Memory> {
        self.memories.get_mut(id).map(|m| {
            m.access();
            m
        })
    }

    pub fn search_by_layer(&self, layer: &MemoryLayer) -> Vec<&Memory> {
        self.memories
            .values()
            .filter(|m| &m.layer == layer)
            .collect()
    }
}

impl Default for MemorySystem {
    fn default() -> Self {
        Self::new()
    }
}

