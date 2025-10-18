//! Optimized Memory Management Module
//!
//! This module provides high-performance memory management with pooling,
//! efficient data structures, and optimized async patterns.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use std::collections::{VecDeque, HashMap};
use parking_lot::Mutex;
use chrono;

use crate::consciousness::{ConsciousnessState, EmotionType};
use crate::personality::PersonalityType;
use crate::brain::BrainType;
use crate::memory::GuessingMemorySystem;
use crate::personal_memory::{PersonalMemoryEngine, PersonalMemory, PersonalInsight};

/// Optimized personal consciousness event with memory pooling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedPersonalConsciousnessEvent {
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

impl OptimizedPersonalConsciousnessEvent {
    /// Create a new optimized personal consciousness event
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

/// Memory pool for consciousness events
pub struct EventMemoryPool {
    pool: Mutex<VecDeque<OptimizedPersonalConsciousnessEvent>>,
}

impl EventMemoryPool {
    pub fn new() -> Self {
        Self {
            pool: Mutex::new(VecDeque::new()),
        }
    }

    pub fn get(&self) -> OptimizedPersonalConsciousnessEvent {
        let mut pool = self.pool.lock();
        if let Some(mut event) = pool.pop_front() {
            // Reset event fields
            event.timestamp = 0.0;
            event.event_type.clear();
            event.content.clear();
            event.personalities_involved.clear();
            event.emotional_impact = 0.0;
            event.learning_will_activation = 0.0;
            event.memory_consolidation_strength = 0.0;
            event.personal_significance = 0.0;
            event
        } else {
            OptimizedPersonalConsciousnessEvent {
                timestamp: 0.0,
                event_type: String::with_capacity(64),
                content: String::with_capacity(512),
                brain_involved: BrainType::Motor,
                personalities_involved: Vec::with_capacity(8),
                emotional_impact: 0.0,
                learning_will_activation: 0.0,
                memory_consolidation_strength: 0.0,
                personal_significance: 0.0,
            }
        }
    }

    pub fn return_event(&self, mut event: OptimizedPersonalConsciousnessEvent) {
        let mut pool = self.pool.lock();
        if pool.len() < 50 { // Limit pool size
            pool.push_back(event);
        }
    }
}

/// Optimized memory management system with performance improvements
pub struct OptimizedMemoryManager {
    memory_store: Arc<RwLock<VecDeque<OptimizedPersonalConsciousnessEvent>>>,
    memory_system: GuessingMemorySystem,
    personal_memory_engine: PersonalMemoryEngine,
    consciousness_state: Arc<RwLock<ConsciousnessState>>,

    // Performance optimizations
    event_pool: EventMemoryPool,
    query_cache: Arc<Mutex<HashMap<String, Vec<OptimizedPersonalConsciousnessEvent>>>>,
    consolidation_threshold: usize,

    // Performance tracking
    cache_hits: Arc<std::sync::atomic::AtomicUsize>,
    cache_misses: Arc<std::sync::atomic::AtomicUsize>,
    pool_hits: Arc<std::sync::atomic::AtomicUsize>,
    pool_misses: Arc<std::sync::atomic::AtomicUsize>,
}

impl OptimizedMemoryManager {
    /// Create a new optimized memory manager
    pub fn new(
        memory_store: Arc<RwLock<VecDeque<OptimizedPersonalConsciousnessEvent>>>,
        memory_system: GuessingMemorySystem,
        personal_memory_engine: PersonalMemoryEngine,
        consciousness_state: Arc<RwLock<ConsciousnessState>>,
    ) -> Self {
        Self {
            memory_store,
            memory_system,
            personal_memory_engine,
            consciousness_state,
            event_pool: EventMemoryPool::new(),
            query_cache: Arc::new(Mutex::new(HashMap::new())),
            consolidation_threshold: 10,
            cache_hits: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            cache_misses: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            pool_hits: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            pool_misses: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// Store a consciousness event with optimized memory management
    pub async fn store_event(&self, mut event: OptimizedPersonalConsciousnessEvent) -> Result<()> {
        debug!("Storing consciousness event: {}", event.event_type);
        
        let mut memory_store = self.memory_store.write().await;
        memory_store.push_back(event);
        
        // Trigger memory consolidation if we have enough events
        if memory_store.len() % self.consolidation_threshold == 0 {
            drop(memory_store); // Release lock before async operation
            self.consolidate_memories().await?;
        }
        
        Ok(())
    }

    /// Store event using memory pool
    pub async fn store_event_pooled(&self, event_type: String, content: String, brain_involved: BrainType, personalities_involved: Vec<PersonalityType>, emotional_impact: f32, learning_will_activation: f32) -> Result<()> {
        let mut event = self.event_pool.get();
        event.timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        event.event_type = event_type;
        event.content = content;
        event.brain_involved = brain_involved;
        event.personalities_involved = personalities_involved;
        event.emotional_impact = emotional_impact;
        event.learning_will_activation = learning_will_activation;
        event.memory_consolidation_strength = emotional_impact * learning_will_activation;
        event.personal_significance = (emotional_impact + learning_will_activation) / 2.0;
        
        self.store_event(event).await
    }

    /// Consolidate memories with optimized batch processing
    pub async fn consolidate_memories(&self) -> Result<()> {
        info!("Consolidating memories (optimized)...");
        
        let memory_store = self.memory_store.read().await;
        let events: Vec<_> = memory_store.iter().collect();
        drop(memory_store); // Release lock early
        
        // Batch process events
        let batch_size = 50;
        for chunk in events.chunks(batch_size) {
            for event in chunk {
                self.memory_system.store_memory(
                    crate::memory::guessing_spheres::SphereId(format!("event_{}", chrono::Utc::now().timestamp())),
                    event.content.clone(),
                    [0.0, 0.0, 0.0], // Default position
                    crate::memory::guessing_spheres::EmotionalVector::new(0.0, 0.0, 0.0, 0.0, 0.0),
                    format!("Learning will: {}", event.learning_will_activation),
                );
            }
        }
        
        debug!("Optimized memory consolidation completed");
        Ok(())
    }

    /// Retrieve memories with caching and optimized search
    pub async fn retrieve_memories(&self, query: &str) -> Result<Vec<OptimizedPersonalConsciousnessEvent>> {
        debug!("Retrieving memories for query: {}", query);
        
        // Check cache first
        {
            let cache = self.query_cache.lock();
            if let Some(cached_results) = cache.get(query) {
                return Ok(cached_results.clone());
            }
        }
        
        let memory_store = self.memory_store.read().await;
        let mut relevant_memories = Vec::new();
        
        // Optimized keyword matching with early termination
        let query_lower = query.to_lowercase();
        for event in memory_store.iter() {
            if event.content.to_lowercase().contains(&query_lower) ||
               event.event_type.to_lowercase().contains(&query_lower) {
                relevant_memories.push(event.clone());
                
                // Limit results to prevent memory bloat
                if relevant_memories.len() >= 100 {
                    break;
                }
            }
        }
        
        // Sort by personal significance
        relevant_memories.sort_by(|a, b| b.personal_significance.partial_cmp(&a.personal_significance).unwrap_or(std::cmp::Ordering::Equal));
        
        // Cache results
        {
            let mut cache = self.query_cache.lock();
            if cache.len() < 100 { // Limit cache size
                cache.insert(query.to_string(), relevant_memories.clone());
            }
        }
        
        Ok(relevant_memories)
    }

    /// Create a memory from conversation with optimized string handling
    pub async fn create_memory_from_conversation(
        &self,
        content: String,
        emotional_significance: f32,
    ) -> Result<PersonalMemory> {
        info!("Creating personal memory from conversation (optimized)");
        
        self.personal_memory_engine.create_memory_from_conversation(content, emotional_significance as f64)
    }

    /// Get emotional memories with optimized filtering
    pub fn get_emotional_memories(
        &self,
        emotion: &EmotionType,
        limit: usize,
    ) -> Vec<PersonalMemory> {
        self.personal_memory_engine.get_emotional_memories(emotion, limit)
    }

    /// Get personal insights with optimized access
    pub fn get_personal_insights(&self) -> Vec<PersonalInsight> {
        self.personal_memory_engine.get_personal_insights()
    }

    /// Export personal consciousness data with optimized serialization
    pub async fn export_personal_consciousness(&self, path: &std::path::Path) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        info!("Exporting personal consciousness data to: {}", path.display());

        // Serialize memory store and personal memories
        let export_data = serde_json::json!({
            "memory_store": self.memory_store.read().await.len(),
            "personal_memories": self.personal_memories.read().await.len(),
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "version": env!("CARGO_PKG_VERSION"),
        });

        let json_content = serde_json::to_string_pretty(&export_data)
            .map_err(|e| anyhow::anyhow!("Failed to serialize consciousness data: {}", e))?;

        let mut file = tokio::fs::File::create(path).await
            .map_err(|e| anyhow::anyhow!("Failed to create export file: {}", e))?;

        file.write_all(json_content.as_bytes()).await
            .map_err(|e| anyhow::anyhow!("Failed to write export data: {}", e))?;

        info!("Successfully exported consciousness data");
        Ok(())
    }

    /// Get memory statistics with optimized calculations
    pub async fn get_memory_stats(&self) -> OptimizedMemoryStats {
        let memory_store = self.memory_store.read().await;
        
        let total_events = memory_store.len();
        let total_emotional_impact: f32 = memory_store.iter().map(|e| e.emotional_impact).sum();
        let total_learning_will: f32 = memory_store.iter().map(|e| e.learning_will_activation).sum();
        let avg_personal_significance: f32 = if total_events > 0 {
            memory_store.iter().map(|e| e.personal_significance).sum::<f32>() / total_events as f32
        } else {
            0.0
        };
        
        OptimizedMemoryStats {
            total_events,
            total_emotional_impact,
            total_learning_will,
            avg_personal_significance,
            cache_hits: self.cache_hits.load(std::sync::atomic::Ordering::Relaxed),
            cache_misses: self.cache_misses.load(std::sync::atomic::Ordering::Relaxed),
            pool_hits: self.pool_hits.load(std::sync::atomic::Ordering::Relaxed),
            pool_misses: self.pool_misses.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Cleanup resources and clear caches
    pub fn cleanup(&self) {
        let mut cache = self.query_cache.lock();
        cache.clear();
        debug!("Optimized memory manager cleanup completed");
    }
}

impl Drop for OptimizedMemoryManager {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Optimized memory statistics
#[derive(Debug, Clone)]
pub struct OptimizedMemoryStats {
    pub total_events: usize,
    pub total_emotional_impact: f32,
    pub total_learning_will: f32,
    pub avg_personal_significance: f32,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub pool_hits: u64,
    pub pool_misses: u64,
}

impl OptimizedMemoryStats {
    pub fn calculate_efficiency(&self) -> f64 {
        let total_cache_operations = self.cache_hits + self.cache_misses;
        let total_pool_operations = self.pool_hits + self.pool_misses;
        
        if total_cache_operations == 0 && total_pool_operations == 0 {
            return 0.0;
        }
        
        let cache_hit_ratio = if total_cache_operations > 0 {
            self.cache_hits as f64 / total_cache_operations as f64
        } else {
            0.0
        };
        
        let pool_hit_ratio = if total_pool_operations > 0 {
            self.pool_hits as f64 / total_pool_operations as f64
        } else {
            0.0
        };
        
        (cache_hit_ratio + pool_hit_ratio) / 2.0
    }
}
