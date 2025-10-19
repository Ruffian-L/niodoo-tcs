//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Memory management module for the consciousness engine
//!
//! This module handles memory consolidation, retrieval, and personal memory integration.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::brain::BrainType;
use crate::consciousness::{ConsciousnessState, EmotionType};
use crate::memory::GuessingMemorySystem;
use crate::personal_memory::{PersonalInsight, PersonalMemory, PersonalMemoryEngine};
use crate::personality::PersonalityType;
use chrono;

/// Personal consciousness event - deeply tied to your journey
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalConsciousnessEvent {
    pub timestamp: f64,
    pub event_type: String,
    pub content: String,
    pub brain_involved: BrainType,
    pub personalities_involved: Vec<PersonalityType>,
    pub emotional_impact: f32,
    pub learning_will_activation: f32,
    pub memory_consolidation_strength: f32,
    pub personal_significance: f32,
}

impl PersonalConsciousnessEvent {
    /// Create a new personal consciousness event
    pub fn new_personal(
        event_type: String,
        content: String,
        brain_involved: BrainType,
        personalities_involved: Vec<PersonalityType>,
        emotional_impact: f32,
        learning_will_activation: f32,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        Self {
            timestamp,
            event_type,
            content,
            brain_involved,
            personalities_involved,
            emotional_impact,
            learning_will_activation,
            memory_consolidation_strength: emotional_impact * learning_will_activation,
            personal_significance: (emotional_impact + learning_will_activation) / 2.0,
        }
    }
}

/// Memory management system for consciousness events
pub struct MemoryManager {
    memory_store: Arc<RwLock<Vec<PersonalConsciousnessEvent>>>,
    memory_system: GuessingMemorySystem,
    personal_memory_engine: PersonalMemoryEngine,
    consciousness_state: Arc<RwLock<ConsciousnessState>>,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new(
        memory_store: Arc<RwLock<Vec<PersonalConsciousnessEvent>>>,
        memory_system: GuessingMemorySystem,
        personal_memory_engine: PersonalMemoryEngine,
        consciousness_state: Arc<RwLock<ConsciousnessState>>,
    ) -> Self {
        Self {
            memory_store,
            memory_system,
            personal_memory_engine,
            consciousness_state,
        }
    }

    /// Store a consciousness event in memory
    pub async fn store_event(&mut self, event: PersonalConsciousnessEvent) -> Result<()> {
        debug!("Storing consciousness event: {}", event.event_type);

        {
            let mut memory_store = self.memory_store.write().await;
            memory_store.push(event);
        }

        // Trigger memory consolidation if we have enough events
        let should_consolidate = {
            let memory_store = self.memory_store.read().await;
            memory_store.len() % 10 == 0
        };

        if should_consolidate {
            self.consolidate_memories().await?;
        }

        Ok(())
    }

    /// Consolidate memories using the guessing memory system
    pub async fn consolidate_memories(&mut self) -> Result<()> {
        info!("Consolidating memories...");

        let memory_store = self.memory_store.read().await;
        let events: Vec<_> = memory_store.iter().cloned().collect();

        // Use the guessing memory system for consolidation
        for event in events {
            self.memory_system.store_memory(
                crate::memory::guessing_spheres::SphereId(format!(
                    "event_{}",
                    chrono::Utc::now().timestamp()
                )),
                event.content,
                [0.0, 0.0, 0.0], // Default position
                crate::memory::guessing_spheres::EmotionalVector::new(0.0, 0.0, 0.0, 0.0, 0.0),
                format!("Learning will: {}", event.learning_will_activation),
            );
        }

        debug!("Memory consolidation completed");
        Ok(())
    }

    /// Retrieve memories based on query
    pub async fn retrieve_memories(&self, query: &str) -> Result<Vec<PersonalConsciousnessEvent>> {
        debug!("Retrieving memories for query: {}", query);

        let memory_store = self.memory_store.read().await;
        let mut relevant_memories = Vec::new();

        // Simple keyword matching for now
        for event in memory_store.iter() {
            if event.content.to_lowercase().contains(&query.to_lowercase())
                || event
                    .event_type
                    .to_lowercase()
                    .contains(&query.to_lowercase())
            {
                relevant_memories.push(event.clone());
            }
        }

        // Sort by personal significance
        relevant_memories.sort_by(|a, b| {
            b.personal_significance
                .partial_cmp(&a.personal_significance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(relevant_memories)
    }

    /// Create a memory from conversation
    pub async fn create_memory_from_conversation(
        &mut self,
        content: String,
        emotional_significance: f32,
    ) -> Result<PersonalMemory> {
        info!("Creating personal memory from conversation");

        let memory = self
            .personal_memory_engine
            .create_memory_from_conversation(content, emotional_significance as f64)?;
        Ok(memory)
    }

    /// Get emotional memories
    pub fn get_emotional_memories(
        &self,
        emotion: &EmotionType,
        limit: usize,
    ) -> Vec<PersonalMemory> {
        self.personal_memory_engine
            .get_emotional_memories(&format!("{:?}", emotion), limit)
    }

    /// Get personal insights
    pub fn get_personal_insights(&self) -> Vec<PersonalInsight> {
        self.personal_memory_engine.get_personal_insights()
    }

    /// Export personal consciousness data
    pub async fn export_personal_consciousness(&self, _path: &std::path::Path) -> Result<()> {
        info!("Exporting personal consciousness data");

        // TODO: Implement actual export functionality
        // This would serialize the memory store and personal memories to the specified path

        Ok(())
    }

    /// Get memory statistics
    pub async fn get_memory_stats(&self) -> MemoryStats {
        let memory_store = self.memory_store.read().await;

        let total_events = memory_store.len();
        let total_emotional_impact: f32 = memory_store.iter().map(|e| e.emotional_impact).sum();
        let total_learning_will: f32 = memory_store
            .iter()
            .map(|e| e.learning_will_activation)
            .sum();
        let avg_personal_significance: f32 = if total_events > 0 {
            memory_store
                .iter()
                .map(|e| e.personal_significance)
                .sum::<f32>()
                / total_events as f32
        } else {
            0.0
        };

        MemoryStats {
            total_events,
            total_emotional_impact,
            total_learning_will,
            avg_personal_significance,
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_events: usize,
    pub total_emotional_impact: f32,
    pub total_learning_will: f32,
    pub avg_personal_significance: f32,
}
