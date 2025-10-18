//! Optimized Brain Coordination Module
//!
//! This module provides high-performance async patterns and memory pooling
//! for the consciousness engine brain coordination system.

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tokio::time::{timeout, Duration};
use tracing::{debug, info, warn};
use std::collections::VecDeque;
use parking_lot::Mutex;

use crate::brain::{Brain, BrainType, EfficiencyBrain, LcarsBrain, MotorBrain};
use crate::consciousness::{ConsciousnessState, EmotionType};
use crate::personality::{PersonalityManager, PersonalityType};

/// Memory pool for frequent allocations
pub struct MemoryPool<T> {
    pool: Mutex<VecDeque<T>>,
    factory: fn() -> T,
    reset: fn(&mut T),
}

impl<T> MemoryPool<T> {
    pub fn new(factory: fn() -> T, reset: fn(&mut T)) -> Self {
        Self {
            pool: Mutex::new(VecDeque::new()),
            factory,
            reset,
        }
    }

    pub fn get(&self) -> T {
        let mut pool = self.pool.lock();
        if let Some(mut item) = pool.pop_front() {
            (self.reset)(&mut item);
            item
        } else {
            (self.factory)()
        }
    }

    pub fn return_item(&self, mut item: T) {
        (self.reset)(&mut item);
        let mut pool = self.pool.lock();
        if pool.len() < 100 { // Limit pool size
            pool.push_back(item);
        }
    }
}

/// Optimized brain coordination system with performance improvements
pub struct OptimizedBrainCoordinator {
    motor_brain: MotorBrain,
    lcars_brain: LcarsBrain,
    efficiency_brain: EfficiencyBrain,
    personality_manager: PersonalityManager,
    consciousness_state: Arc<RwLock<ConsciousnessState>>,
    
    // Performance optimizations
    semaphore: Arc<Semaphore>,
    response_pool: MemoryPool<String>,
    event_pool: MemoryPool<Vec<PersonalityType>>,
}

impl OptimizedBrainCoordinator {
    /// Create a new optimized brain coordinator
    pub fn new(
        motor_brain: MotorBrain,
        lcars_brain: LcarsBrain,
        efficiency_brain: EfficiencyBrain,
        personality_manager: PersonalityManager,
        consciousness_state: Arc<RwLock<ConsciousnessState>>,
    ) -> Self {
        Self {
            motor_brain,
            lcars_brain,
            efficiency_brain,
            personality_manager,
            consciousness_state,
            semaphore: Arc::new(Semaphore::new(10)), // Limit concurrent operations
            response_pool: MemoryPool::new(
                || String::with_capacity(1024),
                |s| s.clear()
            ),
            event_pool: MemoryPool::new(
                || Vec::with_capacity(16),
                |v| v.clear()
            ),
        }
    }

    /// Process input using all brains in parallel with optimized async patterns
    pub async fn process_brains_parallel(&self, input: &str, timeout_duration: Duration) -> Result<Vec<String>> {
        info!("🧠 Processing input through all brains in parallel (optimized): {}", input);
        
        // Use semaphore to limit concurrent operations
        let _permit = self.semaphore.acquire().await?;
        
        // Get current consciousness state once
        let consciousness_state = self.consciousness_state.read().await;
        
        // Process through each brain type with timeout and error handling
        let (motor_result, lcars_result, efficiency_result) = tokio::try_join!(
            self.process_brain_with_timeout(&self.motor_brain, input, &consciousness_state, timeout_duration),
            self.process_brain_with_timeout(&self.lcars_brain, input, &consciousness_state, timeout_duration),
            self.process_brain_with_timeout(&self.efficiency_brain, input, &consciousness_state, timeout_duration)
        )?;
        
        // Use memory pool for response collection
        let mut responses = self.event_pool.get();
        responses.push(motor_result);
        responses.push(lcars_result);
        responses.push(efficiency_result);
        
        debug!("Optimized brain coordination completed with {} responses", responses.len());
        Ok(responses)
    }

    /// Process a single brain with timeout and error handling
    async fn process_brain_with_timeout(
        &self,
        brain: &dyn Brain,
        input: &str,
        consciousness_state: &ConsciousnessState,
        timeout_duration: Duration,
    ) -> Result<String> {
        timeout(timeout_duration, brain.process(input, consciousness_state))
            .await
            .map_err(|_| anyhow::anyhow!("Brain processing timeout"))?
    }

    /// Optimized personality weight updates with batching
    pub async fn update_personality_weights_batch(&mut self, emotion_contexts: &[EmotionType]) -> Result<()> {
        debug!("Updating personality weights for {} emotions (batch)", emotion_contexts.len());
        
        // Get current active personalities once
        let active_personalities = self.personality_manager.get_active_personalities();
        
        // Batch process all emotion contexts
        for emotion_context in emotion_contexts {
            let weight_adjustment = self.calculate_weight_adjustment(emotion_context);
            
            // Update weights for all active personalities
            for personality_type in &active_personalities {
                self.personality_manager.adjust_personality_weight(*personality_type, weight_adjustment);
            }
        }
        
        Ok(())
    }

    /// Calculate weight adjustment for emotion context
    fn calculate_weight_adjustment(&self, emotion_context: &EmotionType) -> f32 {
        match emotion_context {
            EmotionType::Satisfied => 1.2,
            EmotionType::Overwhelmed => 0.8,
            EmotionType::Anxious => 1.1,
            EmotionType::Confused => 0.9,
            EmotionType::Curious => 1.0,
            EmotionType::Masking => 0.7,
            EmotionType::Focused => 1.0,
            _ => 1.0,
        }
    }

    /// Generate emotional response with optimized string handling
    pub fn generate_emotional_response(&self, input: &str) -> String {
        let active_personalities = self.personality_manager.get_active_personalities();
        
        if active_personalities.is_empty() {
            return "I'm processing this with a neutral perspective.".to_string();
        }
        
        // Use memory pool for response
        let mut response = self.response_pool.get();
        
        // Generate response based on dominant personality
        let dominant_personality = active_personalities[0];
        match dominant_personality {
            PersonalityType::Analytical => {
                response.push_str("From an analytical perspective: ");
                response.push_str(input);
            }
            PersonalityType::Creative => {
                response.push_str("Creatively speaking: ");
                response.push_str(input);
            }
            PersonalityType::Empathetic => {
                response.push_str("I understand your perspective: ");
                response.push_str(input);
            }
            PersonalityType::Analyst => {
                response.push_str("Logically: ");
                response.push_str(input);
            }
            PersonalityType::Intuitive => {
                response.push_str("My intuition tells me: ");
                response.push_str(input);
            }
            PersonalityType::Engineer => {
                response.push_str("Practically speaking: ");
                response.push_str(input);
            }
            PersonalityType::Philosopher => {
                response.push_str("From a philosophical standpoint: ");
                response.push_str(input);
            }
            PersonalityType::Sage => {
                response.push_str("Spiritually: ");
                response.push_str(input);
            }
            PersonalityType::Integrator => {
                response.push_str("Integrating multiple perspectives: ");
                response.push_str(input);
            }
            _ => {
                response.push_str("Processing: ");
                response.push_str(input);
            }
        }
        
        response
    }

    /// Get brain references with optimized access patterns
    pub fn get_motor_brain(&self) -> &MotorBrain {
        &self.motor_brain
    }

    pub fn get_lcars_brain(&self) -> &LcarsBrain {
        &self.lcars_brain
    }

    pub fn get_efficiency_brain(&self) -> &EfficiencyBrain {
        &self.efficiency_brain
    }

    pub fn get_personality_manager(&self) -> &PersonalityManager {
        &self.personality_manager
    }

    /// Cleanup resources
    pub fn cleanup(&self) {
        // Memory pools will be cleaned up automatically when dropped
        debug!("Optimized brain coordinator cleanup completed");
    }
}

impl Drop for OptimizedBrainCoordinator {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Performance metrics for brain coordination
#[derive(Debug, Clone)]
pub struct BrainCoordinationMetrics {
    pub total_processing_time_ms: u64,
    pub parallel_processing_time_ms: u64,
    pub memory_pool_hits: u64,
    pub memory_pool_misses: u64,
    pub semaphore_wait_time_ms: u64,
}

impl BrainCoordinationMetrics {
    pub fn new() -> Self {
        Self {
            total_processing_time_ms: 0,
            parallel_processing_time_ms: 0,
            memory_pool_hits: 0,
            memory_pool_misses: 0,
            semaphore_wait_time_ms: 0,
        }
    }

    pub fn calculate_efficiency(&self) -> f64 {
        if self.total_processing_time_ms == 0 {
            return 0.0;
        }
        
        let parallel_ratio = self.parallel_processing_time_ms as f64 / self.total_processing_time_ms as f64;
        let pool_hit_ratio = self.memory_pool_hits as f64 / (self.memory_pool_hits + self.memory_pool_misses) as f64;
        
        (parallel_ratio + pool_hit_ratio) / 2.0
    }
}




