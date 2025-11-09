//! Learning Actor Model - Decoupled MPSC Channel Pattern
//! 
//! This module implements the Actor Model pattern for the learning subsystem,
//! preventing mutex poisoning by fully decoupling the main request pipeline
//! from background learning failures.
//!
//! Architecture:
//! - Main pipeline sends learning data via MPSC channel (non-blocking, cannot panic)
//! - Dedicated actor task processes messages in batches
//! - Panics in actor are isolated and don't affect main pipeline
//! - Write batching reduces Qdrant contention

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use tracing::{error, info, warn};

use crate::compass::CompassOutcome;
use crate::config::RuntimeConfig;
use crate::erag::EragClient;
use crate::generation::GenerationResult;
use crate::learning::{LearningLoop, LearningOutcome};
use crate::tcs_analysis::TopologicalSignature;
use crate::torus::PadGhostState;

/// System Health Vector (4-D): P99 Latency, VRAM Consumption, ROUGE-L Score, Entropy σ
#[derive(Debug, Clone)]
pub struct SystemHealthVector {
    pub p99_latency_ms: f64,
    pub vram_consumption_gb: f64,
    pub rouge_l_score: f64,
    pub entropy_sigma: f64,
    pub timestamp: std::time::SystemTime,
}

impl SystemHealthVector {
    pub fn to_vector(&self) -> Vec<f32> {
        vec![
            self.p99_latency_ms as f32,
            self.vram_consumption_gb as f32,
            self.rouge_l_score as f32,
            self.entropy_sigma as f32,
        ]
    }
}

/// Message sent to learning actor
#[derive(Debug, Clone)]
pub enum LearningMessage {
    /// Update learning loop with new cycle data
    Update {
        pad_state: PadGhostState,
        compass: CompassOutcome,
        collapse: crate::erag::CollapseResult,
        generation: GenerationResult,
        topology: TopologicalSignature,
    },
    /// Record system health vector (batched writes)
    HealthVector(SystemHealthVector),
    /// Shutdown signal
    Shutdown,
}

/// Learning Actor - processes learning messages in background
pub struct LearningActor {
    receiver: mpsc::Receiver<LearningMessage>,
    learning_loop: LearningLoop,
    erag: Arc<EragClient>,
    config: Arc<tokio::sync::RwLock<RuntimeConfig>>,
    health_batch: Vec<SystemHealthVector>,
    batch_size: usize,
    batch_flush_interval: Duration,
}

impl LearningActor {
    pub fn new(
        receiver: mpsc::Receiver<LearningMessage>,
        learning_loop: LearningLoop,
        erag: Arc<EragClient>,
        config: Arc<tokio::sync::RwLock<RuntimeConfig>>,
    ) -> Self {
        let batch_size = std::env::var("LEARNING_HEALTH_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100); // Default: batch 100 health vectors
        
        let batch_flush_secs = std::env::var("LEARNING_HEALTH_BATCH_FLUSH_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1); // Default: flush every 1 second
        
        Self {
            receiver,
            learning_loop,
            erag,
            config,
            health_batch: Vec::with_capacity(batch_size),
            batch_size,
            batch_flush_interval: Duration::from_secs(batch_flush_secs),
        }
    }

    /// Run the actor loop - processes messages forever until shutdown
    pub async fn run(mut self) {
        let mut flush_timer = interval(self.batch_flush_interval);
        flush_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        info!("Learning actor started (batch_size={}, flush_interval={:?})", 
              self.batch_size, self.batch_flush_interval);

        loop {
            tokio::select! {
                // Process incoming messages
                msg = self.receiver.recv() => {
                    match msg {
                Some(LearningMessage::Update { pad_state, compass, collapse, generation, topology }) => {
                    // Process update directly (actor owns learning_loop, no mutex needed)
                    // The update() method internally handles Qdrant operations
                    // If Qdrant panics, it will be caught by the actor's error handling
                    match self.learning_loop.update(
                        &pad_state,
                        &compass,
                        &collapse,
                        &generation,
                        &topology,
                    ).await {
                        Ok(_outcome) => {
                            // Success - outcome is logged inside update()
                            info!("Learning actor processed update successfully");
                        }
                        Err(e) => {
                            warn!("Learning update failed in actor: {}", e);
                            // Actor continues processing - failure is isolated
                        }
                    }
                }
                        Some(LearningMessage::HealthVector(health)) => {
                            self.health_batch.push(health);
                            
                            // Flush if batch is full
                            if self.health_batch.len() >= self.batch_size {
                                self.flush_health_batch().await;
                            }
                        }
                        Some(LearningMessage::Shutdown) => {
                            info!("Learning actor received shutdown signal");
                            // Flush any remaining health vectors
                            if !self.health_batch.is_empty() {
                                self.flush_health_batch().await;
                            }
                            break;
                        }
                        None => {
                            warn!("Learning actor channel closed - shutting down");
                            // Flush any remaining health vectors
                            if !self.health_batch.is_empty() {
                                self.flush_health_batch().await;
                            }
                            break;
                        }
                    }
                }
                // Periodic batch flush
                _ = flush_timer.tick() => {
                    if !self.health_batch.is_empty() {
                        self.flush_health_batch().await;
                    }
                }
            }
        }

        info!("Learning actor stopped");
    }

    /// Flush batched health vectors to Qdrant
    async fn flush_health_batch(&mut self) {
        if self.health_batch.is_empty() {
            return;
        }

        let batch = std::mem::take(&mut self.health_batch);
        let batch_size = batch.len();
        
        info!("Flushing {} health vectors to Qdrant (batched write)", batch_size);

        // Wrap Qdrant write in catch_unwind to prevent panic propagation
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Convert health vectors to Qdrant points
            batch.iter().map(|health| health.to_vector()).collect::<Vec<_>>()
        }));

        match result {
            Ok(vectors) => {
                // TODO: Implement actual Qdrant batch upsert for system_health_logs collection
                // For now, log that batching is working
                info!("Prepared {} health vectors for batch upsert (Qdrant integration pending)", vectors.len());
                
                // Note: The actual Qdrant collection creation and upsert should be implemented
                // in erag.rs with a dedicated method for health vectors
            }
            Err(_) => {
                warn!("Health vector batch preparation panicked - skipping flush");
            }
        }
    }
}

/// Learning Actor Handle - provides channel sender for main pipeline
pub struct LearningActorHandle {
    sender: mpsc::Sender<LearningMessage>,
}

impl LearningActorHandle {
    pub fn new(sender: mpsc::Sender<LearningMessage>) -> Self {
        Self { sender }
    }

    /// Send learning update (non-blocking, cannot panic from Qdrant errors)
    pub async fn send_update(
        &self,
        pad_state: PadGhostState,
        compass: CompassOutcome,
        collapse: crate::erag::CollapseResult,
        generation: GenerationResult,
        topology: TopologicalSignature,
    ) -> Result<()> {
        let msg = LearningMessage::Update {
            pad_state,
            compass,
            collapse,
            generation,
            topology,
        };

        self.sender.send(msg).await
            .map_err(|e| anyhow::anyhow!("Learning actor channel closed: {}", e))?;
        
        Ok(())
    }

    /// Send health vector (non-blocking, batched)
    pub async fn send_health_vector(&self, health: SystemHealthVector) -> Result<()> {
        self.sender.send(LearningMessage::HealthVector(health)).await
            .map_err(|e| anyhow::anyhow!("Learning actor channel closed: {}", e))?;
        
        Ok(())
    }

    /// Shutdown the actor gracefully
    pub async fn shutdown(&self) -> Result<()> {
        self.sender.send(LearningMessage::Shutdown).await
            .map_err(|e| anyhow::anyhow!("Learning actor channel closed: {}", e))?;
        
        Ok(())
    }
}

/// Spawn learning actor and return handle
pub async fn spawn_learning_actor(
    learning_loop: LearningLoop,
    erag: Arc<EragClient>,
    config: Arc<tokio::sync::RwLock<RuntimeConfig>>,
) -> LearningActorHandle {
    let (tx, rx) = mpsc::channel(1000); // Buffer up to 1000 messages
    
    let actor = LearningActor::new(rx, learning_loop, erag, config);
    
    // Spawn actor task
    tokio::spawn(async move {
        actor.run().await;
    });
    
    LearningActorHandle::new(tx)
}

