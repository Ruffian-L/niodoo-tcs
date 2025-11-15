//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Events module placeholder
//!
//! This module provides consciousness event handling

use crate::memory::guessing_spheres::EmotionalVector;
use serde::{Deserialize, Serialize};

/// Consciousness event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsciousnessEvent {
    StateChange {
        timestamp: u64,
        content: String,
        emotional_vector: EmotionalVector,
    },
    MemoryUpdate {
        timestamp: u64,
        content: String,
        emotional_vector: EmotionalVector,
    },
    EmotionalShift {
        timestamp: u64,
        content: String,
        emotional_vector: EmotionalVector,
    },
}

impl ConsciousnessEvent {
    pub fn should_store_memory(&self) -> bool {
        match self {
            ConsciousnessEvent::MemoryUpdate { .. } => true,
            ConsciousnessEvent::StateChange { .. } => true,
            ConsciousnessEvent::EmotionalShift { .. } => false,
        }
    }

    pub fn get_emotional_vector(&self) -> EmotionalVector {
        match self {
            ConsciousnessEvent::StateChange {
                emotional_vector, ..
            } => emotional_vector.clone(),
            ConsciousnessEvent::MemoryUpdate {
                emotional_vector, ..
            } => emotional_vector.clone(),
            ConsciousnessEvent::EmotionalShift {
                emotional_vector, ..
            } => emotional_vector.clone(),
        }
    }

    pub fn timestamp(&self) -> u64 {
        match self {
            ConsciousnessEvent::StateChange { timestamp, .. } => *timestamp,
            ConsciousnessEvent::MemoryUpdate { timestamp, .. } => *timestamp,
            ConsciousnessEvent::EmotionalShift { timestamp, .. } => *timestamp,
        }
    }

    pub fn content(&self) -> &str {
        match self {
            ConsciousnessEvent::StateChange { content, .. } => content,
            ConsciousnessEvent::MemoryUpdate { content, .. } => content,
            ConsciousnessEvent::EmotionalShift { content, .. } => content,
        }
    }

    pub fn event_type(&self) -> String {
        match self {
            ConsciousnessEvent::StateChange { .. } => "state_change".to_string(),
            ConsciousnessEvent::MemoryUpdate { .. } => "memory_update".to_string(),
            ConsciousnessEvent::EmotionalShift { .. } => "emotional_shift".to_string(),
        }
    }

    pub fn emotional_impact(&self) -> f32 {
        let vector = self.get_emotional_vector();
        vector.magnitude()
    }

    pub fn memory_priority(&self) -> u8 {
        match self {
            ConsciousnessEvent::StateChange { .. } => 8,
            ConsciousnessEvent::MemoryUpdate { .. } => 9,
            ConsciousnessEvent::EmotionalShift { .. } => 5,
        }
    }
}

impl Default for ConsciousnessEvent {
    fn default() -> Self {
        ConsciousnessEvent::StateChange {
            timestamp: 0,
            content: "default".to_string(),
            emotional_vector: EmotionalVector::new(0.0, 0.0, 0.0, 0.0, 0.0),
        }
    }
}
