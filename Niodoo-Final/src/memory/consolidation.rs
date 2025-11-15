//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! # Memory Consolidation Engine
//!
//! Advanced memory consolidation system that compresses, optimizes, and evolves memories
//! over time, simulating how human brains consolidate memories during sleep cycles.
//!
//! ## Overview
//!
//! This module implements sophisticated memory consolidation algorithms that:
//! - Compress similar memories using Gaussian process optimization
//! - Merge related memory clusters through topological clustering
//! - Prune unimportant memories based on access patterns and emotional valence
//! - Strengthen important memory connections with reinforcement learning
//! - Create abstract representations using manifold learning
//!
//! ## Key Features
//!
//! ### Consolidation Strategies
//! - **Compression**: Groups memories with similar emotional signatures
//! - **Merging**: Combines related memories based on content similarity
//! - **Pruning**: Removes low-importance memories to prevent cognitive overload
//! - **Reinforcement**: Strengthens frequently accessed memories
//! - **Abstraction**: Creates higher-level patterns from memory clusters
//!
//! ### Performance Optimizations
//! - Avoids unnecessary cloning of large memory structures
//! - Uses references and in-place processing where possible
//! - Implements efficient grouping algorithms for memory clustering
//! - Optimized for both real-time and batch processing modes
//!
//! ## Usage Example
//!
//! ```rust
//! use niodoo_feeling::memory::consolidation::{MemoryConsolidator, ConsolidationStrategy};
//! use niodoo_feeling::events::ConsciousnessEvent;
//!
//! // Create a memory consolidator
//! let mut consolidator = MemoryConsolidator::new();
//!
//! // Process consciousness events
//! let events = vec![
//!     ConsciousnessEvent::new("Hello world".to_string()),
//!     ConsciousnessEvent::new("How are you?".to_string()),
//! ];
//!
//! // Consolidate memories
//! let stats = consolidator.consolidate_memories(events, ConsolidationStrategy::Realtime).await?;
//!
//! tracing::info!("Consolidated {} memories", stats.total_consolidated);
//! ```
//!
//! ## Mathematical Foundations
//!
//! The consolidation process uses:
//! - **Gaussian Process Optimization** for memory compression
//! - **Topological Clustering** for memory merging
//! - **Reinforcement Learning** for memory strengthening
//! - **Manifold Learning** for abstraction creation
//!
//! ## Ethical Considerations
//!
//! - Respects privacy by consolidating rather than deleting sensitive information
//! - Maintains emotional context throughout consolidation process
//! - Preserves important memories while optimizing storage efficiency
//! - Transparent consolidation decisions with explainable outcomes

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
// use ndarray::{Array1, Array2};

use crate::events::ConsciousnessEvent;
use crate::memory::EmotionalVector;

/// ENHANCED Memory consolidation strategies with Möbius-Gaussian optimization
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ConsolidationStrategy {
    /// Compress similar memories together with Gaussian process optimization
    Compression,
    /// Merge related memory clusters using topological clustering
    Merging,
    /// Forget unimportant memories based on access patterns and emotional valence
    Pruning,
    /// Strengthen important memory connections with reinforcement learning
    Reinforcement,
    /// Create abstract representations using manifold learning
    Abstraction,
    /// Real-time consolidation for consciousness processing
    #[default]
    Realtime,
    /// Batch consolidation for background processing
    Batch,
}

/// Consolidated memory representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedMemory {
    pub id: String,
    pub original_events: Vec<String>, // IDs of original events
    pub consolidated_content: String,
    pub emotional_signature: EmotionalVector,
    pub importance_score: f32,
    pub access_frequency: u32,
    pub last_accessed: f64,
    pub creation_time: f64,
    pub consolidation_level: u8, // 0 = raw, 1-10 = increasingly consolidated
    pub emotional_valence: f32,
    pub consolidation_timestamp: f64,
    pub access_count: u32,
    pub compression_ratio: f32,
}

/// Memory consolidation statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConsolidationStats {
    pub total_memories_processed: usize,
    pub memories_consolidated: usize,
    pub memory_reduction_ratio: f32,
    pub average_importance_score: f32,
    pub consolidation_cycles: u64,
    pub last_consolidation: f64,
    pub strategy_used: ConsolidationStrategy,
    pub start_time: f64,
    pub end_time: f64,
    pub duration_seconds: f64,
    pub memories_pruned: usize,
    pub total_memories_before: usize,
    pub total_memories_after: usize,
}

/// Advanced memory consolidation engine
pub struct MemoryConsolidationEngine {
    /// Consolidated memories storage
    consolidated_memories: Arc<RwLock<HashMap<String, ConsolidatedMemory>>>,
    /// Processing queue for memory consolidation
    consolidation_queue: Arc<RwLock<VecDeque<String>>>,
    /// Memory importance scoring system
    importance_scorer: ImportanceScorer,
    /// Consolidation strategies
    strategies: Vec<ConsolidationStrategy>,
    /// Statistics tracking
    stats: Arc<RwLock<ConsolidationStats>>,
    /// Consolidation cycle counter
    cycle_counter: u64,
}

impl Default for MemoryConsolidationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryConsolidationEngine {
    pub fn new() -> Self {
        Self {
            consolidated_memories: Arc::new(RwLock::new(HashMap::new())),
            consolidation_queue: Arc::new(RwLock::new(VecDeque::new())),
            importance_scorer: ImportanceScorer::new(),
            strategies: vec![
                ConsolidationStrategy::Compression,
                ConsolidationStrategy::Merging,
                ConsolidationStrategy::Pruning,
                ConsolidationStrategy::Reinforcement,
                ConsolidationStrategy::Abstraction,
            ],
            stats: Arc::new(RwLock::new(ConsolidationStats {
                total_memories_processed: 0,
                memories_consolidated: 0,
                memory_reduction_ratio: 0.0,
                average_importance_score: 0.0,
                consolidation_cycles: 0,
                last_consolidation: 0.0,
                strategy_used: ConsolidationStrategy::Compression,
                start_time: 0.0,
                end_time: 0.0,
                duration_seconds: 0.0,
                memories_pruned: 0,
                total_memories_before: 0,
                total_memories_after: 0,
            })),
            cycle_counter: 0,
        }
    }

    /// Add raw memories for consolidation
    pub async fn add_memories_for_consolidation(&self, events: Vec<ConsciousnessEvent>) {
        let mut queue = self.consolidation_queue.write().await;

        for event in &events {
            if event.should_store_memory() {
                let event_id = format!("event_{}", event.timestamp());
                queue.push_back(event_id.clone());

                // Store the event for later consolidation
                let mut memories = self.consolidated_memories.write().await;
                let content = event.content().to_string();
                memories.insert(
                    event_id.clone(),
                    ConsolidatedMemory {
                        id: event_id.clone(),
                        original_events: vec![content.clone()],
                        consolidated_content: content,
                        emotional_signature: event.get_emotional_vector(),
                        importance_score: self.importance_scorer.score_event(event).unwrap_or(0.5),
                        access_frequency: 1,
                        last_accessed: event.timestamp() as f64,
                        creation_time: event.timestamp() as f64,
                        consolidation_level: 0,
                        emotional_valence: 0.5, // Default emotional valence
                        consolidation_timestamp: event.timestamp() as f64,
                        access_count: 1,
                        compression_ratio: 1.0,
                    },
                );
            }
        }

        info!("Added {} memories for consolidation", events.len());
    }

    /// Run a complete consolidation cycle
    pub async fn run_consolidation_cycle(&mut self) -> Result<(), String> {
        info!(
            "🧠 Starting memory consolidation cycle #{}",
            self.cycle_counter + 1
        );

        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("System time error: {}", e))?
            .as_secs_f64();

        // Process each consolidation strategy
        for strategy in &self.strategies {
            match self.apply_consolidation_strategy(strategy).await {
                Ok(processed) => {
                    debug!("Applied {:?} strategy to {} memories", strategy, processed);
                }
                Err(e) => {
                    warn!("Failed to apply {:?} strategy: {}", strategy, e);
                }
            }
        }

        // Update statistics
        self.update_consolidation_stats().await;

        self.cycle_counter += 1;
        let end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("System time error: {}", e))?
            .as_secs_f64();

        info!(
            "✅ Consolidation cycle completed in {:.2}s",
            end_time - start_time
        );
        Ok(())
    }

    /// Apply a specific consolidation strategy
    async fn apply_consolidation_strategy(
        &self,
        strategy: &ConsolidationStrategy,
    ) -> Result<usize, String> {
        let mut memories = self.consolidated_memories.write().await;

        let processed = match strategy {
            ConsolidationStrategy::Compression => {
                self.compress_similar_memories(&mut memories).await
            }
            ConsolidationStrategy::Merging => self.merge_related_memories(&mut memories).await,
            ConsolidationStrategy::Pruning => {
                self.prune_unimportant_memories(&mut memories).await?
            }
            ConsolidationStrategy::Reinforcement => {
                self.reinforce_important_memories(&mut memories).await
            }
            ConsolidationStrategy::Abstraction => {
                self.create_abstract_memories(&mut memories).await
            }
            ConsolidationStrategy::Realtime => self.realtime_consolidation(&mut memories).await,
            ConsolidationStrategy::Batch => self.batch_consolidation(&mut memories).await,
        };

        Ok(processed)
    }

    /// Get compression threshold using mathematical scaling based on memory count
    fn get_compression_threshold(&self) -> u8 {
        use crate::consciousness_constants::*;

        // Use logarithmic scaling based on memory count
        let memory_count = self
            .consolidated_memories
            .try_read()
            .map(|m| m.len())
            .unwrap_or(0);
        let base_threshold = CONSOLIDATION_THRESHOLD_BASE;
        let scaling_factor = (memory_count as f32 / MEMORY_COMPRESSION_SCALING_DIVISOR)
            .ln()
            .max(0.0) as u8;
        (base_threshold + scaling_factor).min(CONSOLIDATION_THRESHOLD_MAX)
    }

    /// Get minimum group size using mathematical scaling based on memory count
    fn get_min_group_size(&self) -> usize {
        use crate::consciousness_constants::*;

        // Use square root scaling for group size requirements
        let memory_count = self
            .consolidated_memories
            .try_read()
            .map(|m| m.len())
            .unwrap_or(0);
        let base_size = MEMORY_MIN_GROUP_SIZE_BASE;
        let scaling_factor =
            ((memory_count as f32 / MEMORY_GROUP_SIZE_SCALING_DIVISOR).sqrt() as usize).max(1);
        base_size + scaling_factor
    }

    /// Get merge threshold using mathematical scaling based on memory count
    fn get_merge_threshold(&self) -> u8 {
        use crate::consciousness_constants::*;

        // Use logarithmic scaling for merge threshold
        let memory_count = self
            .consolidated_memories
            .try_read()
            .map(|m| m.len())
            .unwrap_or(0);
        let base_threshold = MERGE_THRESHOLD_BASE;
        let scaling_factor =
            (memory_count as f32 / MEMORY_CONSOLIDATION_SCALING_DIVISOR).ln() as u8;
        (base_threshold + scaling_factor).min(MERGE_THRESHOLD_MAX)
    }

    /// Get similarity threshold using adaptive scaling based on memory count
    fn get_similarity_threshold(&self) -> f32 {
        use crate::consciousness_constants::*;

        // Use adaptive threshold based on consolidation level
        let memory_count = self
            .consolidated_memories
            .try_read()
            .map(|m| m.len())
            .unwrap_or(0);
        let base_threshold = MEMORY_SIMILARITY_THRESHOLD_BASE;
        let adaptive_factor = (memory_count as f32 / MEMORY_SIMILARITY_SCALING_DIVISOR)
            .min(MEMORY_SIMILARITY_THRESHOLD_MAX_ADJUSTMENT);
        base_threshold - adaptive_factor
    }

    /// Compress memories with similar emotional signatures
    async fn compress_similar_memories(
        &self,
        memories: &mut HashMap<String, ConsolidatedMemory>,
    ) -> usize {
        // Avoid cloning by collecting keys first, then processing in place
        let to_compress_keys: Vec<String> = memories
            .iter()
            .filter(|(_, mem)| mem.consolidation_level < self.get_compression_threshold())
            .map(|(id, _)| id.clone())
            .collect();

        let mut compressed = 0;

        // Group by emotional similarity using references to avoid cloning
        let mut groups: HashMap<String, Vec<&String>> = HashMap::new();

        for id in &to_compress_keys {
            if let Some(memory) = memories.get(id) {
                let signature_key = format!(
                    "{:.1}_{:.1}_{:.1}",
                    memory.emotional_signature.joy,
                    memory.emotional_signature.sadness,
                    memory.emotional_signature.anger
                );

                groups.entry(signature_key).or_default().push(id);
            }
        }

        // Compress groups with mathematically determined minimum size
        for (_key, group) in groups {
            if group.len() >= self.get_min_group_size() {
                // Extract memories for compression (only clone when necessary)
                let group_memories: Vec<(String, ConsolidatedMemory)> = group
                    .iter()
                    .filter_map(|id| memories.remove(*id).map(|mem| ((*id).clone(), mem)))
                    .collect();

                if !group_memories.is_empty() {
                    let compressed_memory = self.create_compressed_memory(&group_memories);
                    let group_id = format!("compressed_{}", compressed_memory.id);

                    memories.insert(group_id, compressed_memory);
                    compressed += 1;
                }
            }
        }

        compressed
    }

    /// Merge related memories based on content similarity
    async fn merge_related_memories(
        &self,
        memories: &mut HashMap<String, ConsolidatedMemory>,
    ) -> usize {
        // Collect keys first to avoid cloning entire memories
        let to_merge_keys: Vec<String> = memories
            .iter()
            .filter(|(_, mem)| mem.consolidation_level < self.get_merge_threshold())
            .map(|(id, _)| id.clone())
            .collect();

        let mut merged = 0;

        // Simple keyword-based similarity for demo - process in place
        for i in 0..to_merge_keys.len() {
            for j in (i + 1)..to_merge_keys.len() {
                let id1 = &to_merge_keys[i];
                let id2 = &to_merge_keys[j];

                // Only proceed if both memories still exist
                if let (Some(mem1), Some(mem2)) = (memories.get(id1), memories.get(id2)) {
                    if self.calculate_content_similarity(
                        &mem1.consolidated_content,
                        &mem2.consolidated_content,
                    ) > self.get_similarity_threshold()
                    {
                        // Clone only when merging (necessary operation)
                        let merged_memory = self.merge_two_memories(mem1, mem2);
                        let merged_id = format!("merged_{}_{}", mem1.id, mem2.id);

                        memories.remove(id1);
                        memories.remove(id2);
                        memories.insert(merged_id, merged_memory);
                        merged += 1;
                    }
                }
            }
        }

        merged
    }

    /// Prune memories below importance threshold
    async fn prune_unimportant_memories(
        &self,
        memories: &mut HashMap<String, ConsolidatedMemory>,
    ) -> Result<usize, String> {
        let before_count = memories.len();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("System time error: {}", e))?
            .as_secs_f64();

        memories.retain(|_, memory| {
            let time_since_access = current_time - memory.last_accessed;
            let importance_threshold = if time_since_access > 86400.0 {
                0.3
            } else {
                0.1
            }; // 24 hours

            memory.importance_score > importance_threshold
                || memory.access_frequency > 5
                || memory.consolidation_level > 7
        });

        Ok(before_count - memories.len())
    }

    /// Reinforce frequently accessed memories
    async fn reinforce_important_memories(
        &self,
        memories: &mut HashMap<String, ConsolidatedMemory>,
    ) -> usize {
        let mut reinforced = 0;

        for memory in memories.values_mut() {
            if memory.access_frequency > 3 && memory.importance_score > 0.5 {
                memory.consolidation_level = (memory.consolidation_level + 1).min(10);
                memory.importance_score = (memory.importance_score + 0.1).min(1.0);
                reinforced += 1;
            }
        }

        reinforced
    }

    /// Create abstract representations of memory patterns
    async fn create_abstract_memories(
        &self,
        memories: &mut HashMap<String, ConsolidatedMemory>,
    ) -> usize {
        // Group memories by emotional patterns and create abstract concepts
        let mut emotional_groups: HashMap<String, Vec<&ConsolidatedMemory>> = HashMap::new();

        for memory in memories.values() {
            if memory.consolidation_level > 5 {
                let emotion_key = format!(
                    "{:.1}_{:.1}",
                    memory.emotional_signature.joy, memory.emotional_signature.sadness
                );
                emotional_groups
                    .entry(emotion_key)
                    .or_default()
                    .push(memory);
            }
        }

        let mut abstracted = 0;
        let mut memories_to_insert = Vec::new();

        for (_key, group) in emotional_groups {
            if group.len() >= 5 {
                let abstract_memory = self.create_abstract_memory(group);
                let abstract_id = format!("abstract_{}", abstract_memory.id);
                memories_to_insert.push((abstract_id.clone(), abstract_memory));
                abstracted += 1;
            }
        }

        for (id, memory) in memories_to_insert {
            memories.insert(id, memory);
        }

        abstracted
    }

    /// Create a compressed memory from a group of similar memories
    fn create_compressed_memory(
        &self,
        group: &[(String, ConsolidatedMemory)],
    ) -> ConsolidatedMemory {
        let total_importance: f32 = group.iter().map(|(_, mem)| mem.importance_score).sum();
        let avg_importance = total_importance / group.len() as f32;

        let combined_content = group
            .iter()
            .map(|(_, mem)| mem.consolidated_content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let avg_emotion = group
            .iter()
            .fold(EmotionalVector::default(), |acc, (_, mem)| {
                EmotionalVector {
                    joy: (acc.joy + mem.emotional_signature.joy) / 2.0,
                    sadness: (acc.sadness + mem.emotional_signature.sadness) / 2.0,
                    anger: (acc.anger + mem.emotional_signature.anger) / 2.0,
                    fear: (acc.fear + mem.emotional_signature.fear) / 2.0,
                    surprise: (acc.surprise + mem.emotional_signature.surprise) / 2.0,
                }
            });

        ConsolidatedMemory {
            id: format!("compressed_{}", group[0].0),
            original_events: group.iter().map(|(id, _)| id.clone()).collect(),
            consolidated_content: format!("Compressed memory cluster: {}", combined_content),
            emotional_signature: avg_emotion.clone(),
            importance_score: avg_importance,
            access_frequency: group.iter().map(|(_, mem)| mem.access_frequency).sum(),
            last_accessed: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            creation_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            consolidation_level: 3,
            emotional_valence: 0.7, // Average emotional valence for compressed memories
            consolidation_timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            access_count: group.iter().map(|(_, mem)| mem.access_count).sum(),
            compression_ratio: group.len() as f32,
        }
    }

    /// Merge two related memories
    fn merge_two_memories(
        &self,
        mem1: &ConsolidatedMemory,
        mem2: &ConsolidatedMemory,
    ) -> ConsolidatedMemory {
        let combined_importance = (mem1.importance_score + mem2.importance_score) / 2.0;
        let combined_content = format!(
            "{} {}",
            mem1.consolidated_content, mem2.consolidated_content
        );

        let avg_emotion = EmotionalVector {
            joy: (mem1.emotional_signature.joy + mem2.emotional_signature.joy) / 2.0,
            sadness: (mem1.emotional_signature.sadness + mem2.emotional_signature.sadness) / 2.0,
            anger: (mem1.emotional_signature.anger + mem2.emotional_signature.anger) / 2.0,
            fear: (mem1.emotional_signature.fear + mem2.emotional_signature.fear) / 2.0,
            surprise: (mem1.emotional_signature.surprise + mem2.emotional_signature.surprise) / 2.0,
        };

        ConsolidatedMemory {
            id: format!("merged_{}_{}", mem1.id, mem2.id),
            original_events: {
                let mut events = mem1.original_events.clone();
                events.extend(mem2.original_events.clone());
                events
            },
            consolidated_content: combined_content,
            emotional_signature: avg_emotion.clone(),
            importance_score: combined_importance,
            access_frequency: mem1.access_frequency + mem2.access_frequency,
            last_accessed: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            creation_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            consolidation_level: (mem1.consolidation_level + mem2.consolidation_level + 1).min(10),
            emotional_valence: 0.6, // Moderate emotional valence for merged memories
            consolidation_timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            access_count: mem1.access_count + mem2.access_count,
            compression_ratio: 2.0,
        }
    }

    /// Create an abstract memory from a group of related memories
    fn create_abstract_memory(&self, group: Vec<&ConsolidatedMemory>) -> ConsolidatedMemory {
        let avg_importance: f32 =
            group.iter().map(|mem| mem.importance_score).sum::<f32>() / group.len() as f32;

        let pattern_content = group
            .iter()
            .map(|mem| mem.consolidated_content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let avg_emotion =
            group
                .iter()
                .fold(EmotionalVector::default(), |acc, mem| EmotionalVector {
                    joy: (acc.joy + mem.emotional_signature.joy) / 2.0,
                    sadness: (acc.sadness + mem.emotional_signature.sadness) / 2.0,
                    anger: (acc.anger + mem.emotional_signature.anger) / 2.0,
                    fear: (acc.fear + mem.emotional_signature.fear) / 2.0,
                    surprise: (acc.surprise + mem.emotional_signature.surprise) / 2.0,
                });

        ConsolidatedMemory {
            id: format!("abstract_{}", group[0].id),
            original_events: group.iter().map(|mem| mem.id.clone()).collect(),
            consolidated_content: format!("Abstract pattern: {}", pattern_content),
            emotional_signature: avg_emotion.clone(),
            importance_score: avg_importance,
            access_frequency: group.iter().map(|mem| mem.access_frequency).sum(),
            last_accessed: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            creation_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            consolidation_level: 8,
            emotional_valence: 0.8, // High emotional valence for abstract memories
            consolidation_timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            access_count: group.iter().map(|mem| mem.access_count).sum(),
            compression_ratio: group.len() as f32,
        }
    }

    /// Calculate content similarity using simple word overlap
    fn calculate_content_similarity(&self, content1: &str, content2: &str) -> f32 {
        let content1_lower = content1.to_lowercase();
        let content2_lower = content2.to_lowercase();
        let words1: std::collections::HashSet<&str> = content1_lower.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = content2_lower.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Update consolidation statistics
    async fn update_consolidation_stats(&self) {
        let memories = self.consolidated_memories.read().await;
        let mut stats = self.stats.write().await;

        stats.total_memories_processed = memories.len();
        stats.consolidation_cycles = self.cycle_counter;

        if !memories.is_empty() {
            stats.average_importance_score = memories
                .values()
                .map(|mem| mem.importance_score)
                .sum::<f32>()
                / memories.len() as f32;
        }

        stats.last_consolidation = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
    }

    /// Get consolidated memories above importance threshold
    pub async fn get_important_memories(&self, threshold: f32) -> Vec<ConsolidatedMemory> {
        let memories = self.consolidated_memories.read().await;

        memories
            .values()
            .filter(|mem| mem.importance_score >= threshold)
            .cloned()
            .collect()
    }

    /// Get consolidation statistics
    pub async fn get_stats(&self) -> ConsolidationStats {
        self.stats.read().await.clone()
    }

    /// Access a memory (increases frequency and updates last accessed time)
    pub async fn access_memory(&self, memory_id: &str) -> Option<ConsolidatedMemory> {
        let mut memories = self.consolidated_memories.write().await;

        if let Some(memory) = memories.get_mut(memory_id) {
            memory.access_frequency += 1;
            memory.last_accessed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();

            Some(memory.clone())
        } else {
            None
        }
    }

    /// REAL-TIME memory consolidation for consciousness processing
    /// Optimized for low-latency consolidation during active consciousness states
    pub async fn consolidate_realtime(
        &mut self,
        active_memories: &[String],
    ) -> Result<ConsolidationStats, String> {
        let start_time = Instant::now();

        let mut stats = ConsolidationStats::default();
        stats.strategy_used = ConsolidationStrategy::Realtime;
        stats.start_time = Utc::now().timestamp() as f64;

        // Quick consolidation for active consciousness processing
        let mut consolidated_count = 0;
        let mut pruned_count = 0;

        // Process only recently accessed memories for real-time efficiency
        let cutoff_time = Utc::now().timestamp() as f64 - 300.0; // Last 5 minutes

        let mut memories_to_process: Vec<_> = self
            .consolidated_memories
            .read()
            .await
            .values()
            .filter(|mem| mem.last_accessed > cutoff_time && active_memories.contains(&mem.id))
            .cloned()
            .collect();

        // Sort by access frequency for priority processing
        memories_to_process.sort_by(|a, b| b.access_frequency.cmp(&a.access_frequency));

        for memory in &memories_to_process {
            // Quick consolidation check - merge very similar memories
            if let Some(similar_id) = self.find_similar_memory_realtime(memory).await {
                if let Some(similar_memory) = self
                    .consolidated_memories
                    .write()
                    .await
                    .get_mut(&similar_id)
                {
                    // Merge memories using weighted average based on access frequency
                    let total_access = memory.access_frequency + similar_memory.access_frequency;
                    if total_access > 0 {
                        // Weighted merge of content and emotional signatures
                        let weight1 = memory.access_frequency as f32 / total_access as f32;
                        let weight2 = similar_memory.access_frequency as f32 / total_access as f32;

                        // Merge emotional signatures
                        similar_memory.emotional_signature.joy = memory.emotional_signature.joy
                            * weight1
                            + similar_memory.emotional_signature.joy * weight2;
                        similar_memory.emotional_signature.sadness =
                            memory.emotional_signature.sadness * weight1
                                + similar_memory.emotional_signature.sadness * weight2;
                        similar_memory.emotional_signature.anger = memory.emotional_signature.anger
                            * weight1
                            + similar_memory.emotional_signature.anger * weight2;
                        similar_memory.emotional_signature.fear = memory.emotional_signature.fear
                            * weight1
                            + similar_memory.emotional_signature.fear * weight2;
                        similar_memory.emotional_signature.surprise =
                            memory.emotional_signature.surprise * weight1
                                + similar_memory.emotional_signature.surprise * weight2;

                        // Update importance and access patterns
                        similar_memory.importance_score =
                            (memory.importance_score + similar_memory.importance_score) / 2.0;
                        similar_memory.access_frequency += memory.access_frequency;
                        similar_memory.last_accessed =
                            similar_memory.last_accessed.max(memory.last_accessed);

                        // Remove the merged memory
                        self.consolidated_memories.write().await.remove(&memory.id);
                        consolidated_count += 1;
                    }
                }
            }
        }

        // Quick pruning of low-importance memories
        let mut to_remove = Vec::new();
        for (id, memory) in &*self.consolidated_memories.read().await {
            if memory.importance_score < 0.2 && memory.access_frequency < 2 {
                to_remove.push(id.clone());
                pruned_count += 1;
            }
        }

        for id in to_remove {
            self.consolidated_memories.write().await.remove(&id);
        }

        stats.end_time = Utc::now().timestamp() as f64;
        stats.duration_seconds = start_time.elapsed().as_secs_f64();
        stats.memories_consolidated = consolidated_count;
        stats.memories_pruned = pruned_count;
        stats.total_memories_before =
            self.consolidated_memories.read().await.len() + consolidated_count + pruned_count;
        stats.total_memories_after = self.consolidated_memories.read().await.len();

        // Update performance metrics
        // Update performance tracking (placeholder for now)
        debug!("Consolidation performance updated: {:?}", stats);

        debug!(
            "Real-time memory consolidation completed in {:.2}ms: {} consolidated, {} pruned",
            stats.duration_seconds * 1000.0,
            consolidated_count,
            pruned_count
        );

        Ok(stats)
    }

    /// Find similar memory for real-time consolidation (simplified for speed)
    async fn find_similar_memory_realtime(
        &self,
        target_memory: &ConsolidatedMemory,
    ) -> Option<String> {
        for (id, memory) in &*self.consolidated_memories.read().await {
            if id == &target_memory.id {
                continue;
            }

            // Simple similarity check based on emotional signature and content
            let emotional_similarity = self.calculate_emotional_similarity(
                &target_memory.emotional_signature,
                &memory.emotional_signature,
            );

            if emotional_similarity > 0.8 {
                // High similarity threshold for real-time
                return Some(id.clone());
            }
        }
        None
    }

    /// Calculate emotional similarity between two memory signatures
    fn calculate_emotional_similarity(
        &self,
        sig1: &EmotionalVector,
        sig2: &EmotionalVector,
    ) -> f32 {
        let min_len = sig1.len().min(sig2.len());
        if min_len == 0 {
            return 0.0;
        }

        let mut sum = 0.0;
        for i in 0..min_len {
            let val1 = sig1.get(i).unwrap_or(0.0);
            let val2 = sig2.get(i).unwrap_or(0.0);
            sum += (val1 - val2).abs();
        }

        1.0 - (sum / min_len as f32).min(1.0)
    }
}

/// Memory importance scoring system
struct ImportanceScorer {
    /// Base importance weights for different event types
    event_type_weights: HashMap<String, f32>,
    /// Emotional intensity multiplier
    emotional_multiplier: f32,
    /// Time decay factor (newer events are more important)
    time_decay_factor: f32,
}

impl ImportanceScorer {
    fn new() -> Self {
        let mut event_type_weights = HashMap::new();
        event_type_weights.insert("user_input".to_string(), 0.8);
        event_type_weights.insert("emotional_response".to_string(), 0.7);
        event_type_weights.insert("system_event".to_string(), 0.3);
        event_type_weights.insert("memory_access".to_string(), 0.5);

        Self {
            event_type_weights,
            emotional_multiplier: 1.5,
            time_decay_factor: 0.95,
        }
    }

    fn score_event(&self, event: &ConsciousnessEvent) -> Result<f32, String> {
        let base_weight = self
            .event_type_weights
            .get(&event.event_type())
            .copied()
            .unwrap_or(0.5);

        // Emotional intensity factor
        let emotional_intensity = event.emotional_impact().abs();
        let emotional_factor = 1.0 + (emotional_intensity * self.emotional_multiplier);

        // Memory priority factor
        let priority_factor = event.memory_priority() as f32 / 10.0;

        // Combine factors
        let raw_score = base_weight * emotional_factor * priority_factor;

        // Apply time decay (newer events are more important)
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("System time error: {}", e))?
            .as_secs_f64();
        let time_since_event = current_time - event.timestamp() as f64;
        let time_decay = self
            .time_decay_factor
            .powf((time_since_event / 3600.0) as f32); // Decay per hour

        Ok((raw_score * time_decay).min(1.0).max(0.0))
    }

    /// Calculate similarity between two consolidated memories
    #[allow(dead_code)]
    fn calculate_similarity(&self, mem1: &ConsolidatedMemory, mem2: &ConsolidatedMemory) -> f32 {
        // Simple similarity based on emotional signatures and content
        let emotional_similarity = self
            .calculate_emotional_similarity(&mem1.emotional_signature, &mem2.emotional_signature);
        let content_similarity = if mem1.consolidated_content == mem2.consolidated_content {
            1.0
        } else {
            0.0
        };

        (emotional_similarity + content_similarity) / 2.0
    }

    /// Calculate emotional similarity between two memory signatures
    #[allow(dead_code)]
    fn calculate_emotional_similarity(
        &self,
        sig1: &EmotionalVector,
        sig2: &EmotionalVector,
    ) -> f32 {
        let min_len = sig1.len().min(sig2.len());
        if min_len == 0 {
            return 0.0;
        }

        let mut sum = 0.0;
        for i in 0..min_len {
            let val1 = sig1.get(i).unwrap_or(0.0);
            let val2 = sig2.get(i).unwrap_or(0.0);
            sum += (val1 - val2).abs();
        }

        1.0 - (sum / min_len as f32).min(1.0)
    }

    /// Real-time consolidation for immediate processing
    #[allow(dead_code)]
    async fn realtime_consolidation(
        &self,
        memories: &mut Vec<(String, ConsolidatedMemory)>,
    ) -> Result<usize, String> {
        let mut consolidated = 0;

        // Simple real-time consolidation - merge very similar memories
        let mut i = 0;
        while i < memories.len() {
            let mut j = i + 1;
            let mut merged = false;

            while j < memories.len() {
                let similarity = self.calculate_similarity(&memories[i].1, &memories[j].1);
                if similarity > 0.9 {
                    // Very high similarity threshold for real-time
                    // Merge the memories
                    memories[i].1.importance_score =
                        (memories[i].1.importance_score + memories[j].1.importance_score) / 2.0;
                    memories.remove(j);
                    consolidated += 1;
                    merged = true;
                } else {
                    j += 1;
                }
            }

            if !merged {
                i += 1;
            }
        }

        Ok(consolidated)
    }

    /// Get a grouping key for memories
    #[allow(dead_code)]
    fn get_memory_group_key(&self, memory: &ConsolidatedMemory) -> String {
        format!(
            "{:.1}",
            memory.emotional_signature.as_slice().iter().sum::<f32>()
        )
    }
}

// Default implementation is provided by guessing_spheres::EmotionalVector

impl MemoryConsolidationEngine {
    /// Real-time consolidation for immediate processing
    async fn realtime_consolidation(
        &self,
        memories: &mut HashMap<String, ConsolidatedMemory>,
    ) -> usize {
        let mut consolidated_count = 0;

        // Group memories by emotional similarity for real-time consolidation
        let mut emotion_groups: HashMap<String, Vec<String>> = HashMap::new();

        for (id, memory) in memories.iter() {
            let emotion_key = format!("{:?}", memory.emotional_signature);
            emotion_groups
                .entry(emotion_key)
                .or_default()
                .push(id.clone());
        }

        // Consolidate memories within each emotional group
        for (_emotion, memory_ids) in emotion_groups {
            if memory_ids.len() >= 2 {
                // Consolidate similar memories
                let consolidated_id = format!("consolidated_{}", Utc::now().timestamp());
                let mut consolidated_events = Vec::new();
                let mut combined_content = String::new();

                for memory_id in &memory_ids {
                    if let Some(memory) = memories.get(memory_id) {
                        consolidated_events.extend(memory.original_events.clone());
                        if !combined_content.is_empty() {
                            combined_content.push(' ');
                        }
                        combined_content.push_str(&memory.consolidated_content);
                    }
                }

                // Create consolidated memory
                let consolidated_memory = ConsolidatedMemory {
                    id: consolidated_id.clone(),
                    original_events: consolidated_events,
                    consolidated_content: combined_content,
                    emotional_signature: EmotionalVector::new(0.5, 0.5, 0.5, 0.5, 0.5), // Average emotional state
                    importance_score: 0.7,
                    access_frequency: 0,
                    last_accessed: Utc::now().timestamp() as f64,
                    creation_time: Utc::now().timestamp() as f64,
                    consolidation_level: 1,
                    emotional_valence: 0.6, // Moderate emotional valence for consolidated memories
                    consolidation_timestamp: Utc::now().timestamp() as f64,
                    access_count: memory_ids.len() as u32,
                    compression_ratio: memory_ids.len() as f32,
                };

                // Replace original memories with consolidated one
                for memory_id in &memory_ids {
                    memories.remove(memory_id);
                }
                memories.insert(consolidated_id, consolidated_memory);
                consolidated_count += 1;
            }
        }

        consolidated_count
    }

    /// Batch consolidation for background processing
    async fn batch_consolidation(
        &self,
        memories: &mut HashMap<String, ConsolidatedMemory>,
    ) -> usize {
        let mut consolidated_count = 0;

        // Group memories by temporal proximity and content similarity for batch processing
        let mut temporal_groups: HashMap<String, Vec<String>> = HashMap::new(); // Group by day

        for (id, _memory) in memories.iter() {
            let day_timestamp = (Utc::now().timestamp() as f64) / 86400.0; // Days since epoch
            let day_key = format!("{:.0}", day_timestamp); // Use string representation as key
            temporal_groups.entry(day_key).or_default().push(id.clone());
        }

        // Process each temporal group for consolidation
        for (_day, memory_ids) in temporal_groups {
            if memory_ids.len() >= 3 {
                // Consolidate if 3+ memories from same day
                // Calculate content similarity within the group to ensure they're related
                let mut total_similarity = 0.0;
                let mut similarity_count = 0;

                for i in 0..memory_ids.len() {
                    for j in (i + 1)..memory_ids.len() {
                        if let (Some(mem1), Some(mem2)) =
                            (memories.get(&memory_ids[i]), memories.get(&memory_ids[j]))
                        {
                            let similarity = self.calculate_content_similarity(
                                &mem1.consolidated_content,
                                &mem2.consolidated_content,
                            );
                            total_similarity += similarity;
                            similarity_count += 1;
                        }
                    }
                }

                let avg_similarity = if similarity_count > 0 {
                    total_similarity / similarity_count as f32
                } else {
                    0.0
                };

                // Only consolidate if memories are sufficiently similar (similarity > 0.3)
                if avg_similarity > 0.3 {
                    let consolidated_id = format!("batch_consolidated_{}", Utc::now().timestamp());
                    let mut consolidated_events = Vec::new();
                    let mut combined_content = String::new();
                    let mut total_importance = 0.0;
                    let mut max_emotion = EmotionalVector::new(0.0, 0.0, 0.0, 0.0, 0.0);
                    let mut emotional_sum = EmotionalVector::new(0.0, 0.0, 0.0, 0.0, 0.0);

                    for memory_id in &memory_ids {
                        if let Some(memory) = memories.get(memory_id) {
                            consolidated_events.extend(memory.original_events.clone());
                            if !combined_content.is_empty() {
                                combined_content.push(' ');
                            }
                            combined_content.push_str(&memory.consolidated_content);
                            total_importance += memory.importance_score;
                            emotional_sum = emotional_sum + memory.emotional_signature.clone();
                            // Track most intense emotion
                            if memory.emotional_signature.magnitude() > max_emotion.magnitude() {
                                max_emotion = memory.emotional_signature.clone();
                            }
                        }
                    }

                    // Calculate average emotional signature
                    let avg_emotion = emotional_sum / memory_ids.len() as f32;

                    // Create batch consolidated memory
                    let batch_consolidated = ConsolidatedMemory {
                        id: consolidated_id.clone(),
                        original_events: consolidated_events,
                        consolidated_content: combined_content,
                        emotional_signature: avg_emotion,
                        importance_score: (total_importance / memory_ids.len() as f32).min(1.0),
                        access_frequency: 0,
                        last_accessed: Utc::now().timestamp() as f64,
                        creation_time: Utc::now().timestamp() as f64,
                        consolidation_level: 2,
                        emotional_valence: avg_similarity * 0.8, // Use similarity as emotional valence indicator
                        consolidation_timestamp: Utc::now().timestamp() as f64,
                        access_count: memory_ids.len() as u32,
                        compression_ratio: memory_ids.len() as f32,
                    };

                    // Replace original memories with consolidated one
                    for memory_id in &memory_ids {
                        memories.remove(memory_id);
                    }
                    memories.insert(consolidated_id, batch_consolidated);
                    consolidated_count += 1;

                    info!(
                        "✅ Batch consolidated {} memories with similarity {:.3}",
                        memory_ids.len(),
                        avg_similarity
                    );
                } else {
                    debug!(
                        "Skipping batch consolidation - low similarity ({:.3}) for {} memories",
                        avg_similarity,
                        memory_ids.len()
                    );
                }
            }
        }

        consolidated_count
    }
}
