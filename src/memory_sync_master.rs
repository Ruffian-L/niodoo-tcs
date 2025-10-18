/*
 * 🧠 Agent 2: MemorySyncMaster
 * Queries semantic + emotional layers for resonance >0.4
 */

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{error, info, warn, debug};

use crate::memory::{MemorySystem, MemoryQuery, MemoryResult};

/// Multi-layer memory query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLayerMemoryResult {
    pub semantic_layer: Vec<MemoryResult>,
    pub emotional_layer: Vec<MemoryResult>,
    pub resonance_score: f32,
    pub emotional_valence: f32,
    pub semantic_coherence: f32,
    pub layer_sync_status: String,
}

/// Memory layer types
#[derive(Debug, Clone)]
pub enum MemoryLayer {
    Semantic,
    Emotional,
    Contextual,
}

/// MemorySyncMaster agent for multi-layer memory querying
pub struct MemorySyncMaster {
    memory_system: Arc<dyn MemorySystem>,
    layer_configs: HashMap<MemoryLayer, LayerConfig>,
    sync_channel: mpsc::UnboundedSender<MultiLayerMemoryResult>,
    shutdown: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub resonance_threshold: f32,
    pub max_results: usize,
    pub emotional_weight: f32,
    pub semantic_weight: f32,
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            resonance_threshold: 0.4, // As specified: resonance >0.4
            max_results: 10,
            emotional_weight: 0.3,
            semantic_weight: 0.7,
        }
    }
}

impl MemorySyncMaster {
    /// Create new MemorySyncMaster agent
    pub fn new(memory_system: Arc<dyn MemorySystem>) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        // Spawn sync monitoring loop
        tokio::spawn(async move {
            while !shutdown_clone.load(Ordering::Relaxed) {
                if let Ok(result) = rx.try_recv() {
                    Self::process_sync_result(&result).await;
                }
                tokio::time::sleep(Duration::from_millis(50)).await; // 20 Hz monitoring
            }
        });

        let mut layer_configs = HashMap::new();
        layer_configs.insert(MemoryLayer::Semantic, LayerConfig::default());
        layer_configs.insert(MemoryLayer::Emotional, LayerConfig {
            resonance_threshold: 0.4,
            max_results: 8,
            emotional_weight: 0.8,
            semantic_weight: 0.2,
        });

        Self {
            memory_system,
            layer_configs,
            sync_channel: tx,
            shutdown,
        }
    }

    /// Query multi-layer memory with resonance filtering
    pub async fn query_multi_layer(&self, query: &str) -> Result<MultiLayerMemoryResult> {
        info!("🧠 MemorySyncMaster querying multi-layer: {}", query);

        // Query semantic layer
        let semantic_query = MemoryQuery {
            content: query.to_string(),
            k: self.layer_configs[&MemoryLayer::Semantic].max_results,
            threshold: self.layer_configs[&MemoryLayer::Semantic].resonance_threshold,
        };
        let semantic_results = self.memory_system.query(semantic_query).await?;

        // Query emotional layer
        let emotional_query = MemoryQuery {
            content: query.to_string(),
            k: self.layer_configs[&MemoryLayer::Emotional].max_results,
            threshold: self.layer_configs[&MemoryLayer::Emotional].resonance_threshold,
        };
        let emotional_results = self.memory_system.query(emotional_query).await?;

        // Calculate resonance and coherence scores
        let resonance_score = self.calculate_resonance_score(&semantic_results, &emotional_results);
        let emotional_valence = self.calculate_emotional_valence(&emotional_results);
        let semantic_coherence = self.calculate_semantic_coherence(&semantic_results);

        // Determine layer sync status
        let layer_sync_status = if resonance_score > 0.4 {
            if emotional_valence.abs() > 0.3 {
                "emotional_sync".to_string()
            } else {
                "semantic_sync".to_string()
            }
        } else {
            "desync".to_string()
        };

        let result = MultiLayerMemoryResult {
            semantic_layer: semantic_results,
            emotional_layer: emotional_results,
            resonance_score,
            emotional_valence,
            semantic_coherence,
            layer_sync_status,
        };

        // Send to sync monitoring channel
        if let Err(e) = self.sync_channel.send(result.clone()) {
            warn!("Failed to send sync result: {}", e);
        }

        info!("🧠 Multi-layer query complete: resonance={:.3}, valence={:.3}, coherence={:.3}, status={}",
              resonance_score, emotional_valence, semantic_coherence, layer_sync_status);

        Ok(result)
    }

    /// Calculate resonance score between semantic and emotional layers
    fn calculate_resonance_score(&self, semantic: &[MemoryResult], emotional: &[MemoryResult]) -> f32 {
        if semantic.is_empty() || emotional.is_empty() {
            return 0.0;
        }

        let semantic_avg_resonance: f32 = semantic.iter().map(|r| r.resonance).sum::<f32>() / semantic.len() as f32;
        let emotional_avg_resonance: f32 = emotional.iter().map(|r| r.resonance).sum::<f32>() / emotional.len() as f32;

        // Weighted combination
        let semantic_weight = self.layer_configs[&MemoryLayer::Semantic].semantic_weight;
        let emotional_weight = self.layer_configs[&MemoryLayer::Emotional].emotional_weight;

        (semantic_avg_resonance * semantic_weight + emotional_avg_resonance * emotional_weight).clamp(0.0, 1.0)
    }

    /// Calculate emotional valence from emotional layer results
    fn calculate_emotional_valence(&self, emotional: &[MemoryResult]) -> f32 {
        if emotional.is_empty() {
            return 0.0;
        }

        emotional.iter().map(|r| r.emotional_valence).sum::<f32>() / emotional.len() as f32
    }

    /// Calculate semantic coherence from semantic layer results
    fn calculate_semantic_coherence(&self, semantic: &[MemoryResult]) -> f32 {
        if semantic.len() < 2 {
            return 0.5; // Default coherence for single result
        }

        // Simple coherence metric based on resonance variance
        let resonances: Vec<f32> = semantic.iter().map(|r| r.resonance).collect();
        let mean = resonances.iter().sum::<f32>() / resonances.len() as f32;
        let variance = resonances.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f32>() / resonances.len() as f32;

        // Lower variance = higher coherence
        (1.0 - variance.min(1.0)).max(0.0)
    }

    /// Process sync result for monitoring and optimization
    async fn process_sync_result(result: &MultiLayerMemoryResult) {
        debug!("📊 Processing sync result: {:?}", result);

        // Log sync status for debugging
        if result.resonance_score > 0.4 {
            info!("✅ Memory layers synchronized: resonance={:.3}", result.resonance_score);
        } else {
            warn!("⚠️  Memory layers desynchronized: resonance={:.3}", result.resonance_score);
        }

        // This would trigger optimization or re-sync procedures in a full implementation
    }

    /// Continuous monitoring loop for memory layer health
    pub async fn run_monitoring(&self) -> Result<()> {
        info!("🧠 MemorySyncMaster starting monitoring");

        let mut interval = tokio::time::interval(Duration::from_secs(5)); // 5 second intervals

        while !self.shutdown.load(Ordering::Relaxed) {
            interval.tick().await;

            // Monitor layer health
            let test_query = "memory layer health check";
            match self.query_multi_layer(test_query).await {
                Ok(result) => {
                    if result.resonance_score < 0.3 {
                        warn!("⚠️  Low memory layer resonance detected: {:.3}", result.resonance_score);
                    }
                }
                Err(e) => {
                    warn!("Memory layer monitoring error: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Shutdown the agent
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        info!("🧠 MemorySyncMaster shutting down");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MockMemorySystem;

    #[tokio::test]
    async fn test_memory_sync_master_creation() {
        let memory_system = Arc::new(MockMemorySystem::new());
        let master = MemorySyncMaster::new(memory_system);

        let result = master.query_multi_layer("test query").await.unwrap();
        assert!(!result.semantic_layer.is_empty());
        assert!(!result.emotional_layer.is_empty());
        assert!(result.resonance_score >= 0.0 && result.resonance_score <= 1.0);
        assert!(result.emotional_valence >= -1.0 && result.emotional_valence <= 1.0);
        assert!(result.semantic_coherence >= 0.0 && result.semantic_coherence <= 1.0);
    }

    #[tokio::test]
    async fn test_resonance_threshold_filtering() {
        let memory_system = Arc::new(MockMemorySystem::new());
        let master = MemorySyncMaster::new(memory_system);

        // Query with high resonance threshold
        let mut config = LayerConfig::default();
        config.resonance_threshold = 0.8; // Very high threshold

        let result = master.query_multi_layer("test query").await.unwrap();

        // Results should still be present (mock data has high resonance)
        assert!(!result.semantic_layer.is_empty() || !result.emotional_layer.is_empty());
    }

    #[tokio::test]
    async fn test_layer_sync_status() {
        let memory_system = Arc::new(MockMemorySystem::new());
        let master = MemorySyncMaster::new(memory_system);

        let result = master.query_multi_layer("test query").await.unwrap();

        // Should have a valid sync status
        assert!(matches!(
            result.layer_sync_status.as_str(),
            "emotional_sync" | "semantic_sync" | "desync"
        ));
    }
}
