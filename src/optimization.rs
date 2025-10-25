//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🚀 Grok's Sock Optimization Engine - Rust Port
 *
 * Integrating Grok's distributed optimization algorithms with our
 * Rust consciousness system for enhanced personality consensus and
 * brain communication efficiency.
 */

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, info};

use crate::brain::{BrainResponse, BrainType};
// Removed unused imports
use crate::personality::{PersonalityManager, PersonalityType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationMetrics {
    pub processing_time: f64,
    pub efficiency_score: f32,
    pub consensus_strength: f32,
    pub distributed_speedup: f32,
    pub memory_usage_mb: f32,
    pub timestamp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityResponse {
    pub personality_type: PersonalityType,
    pub response: String,
    pub confidence: f32,
    pub processing_time: f64,
    pub relevance_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedConsensus {
    pub final_response: String,
    pub contributing_personalities: Vec<PersonalityType>,
    pub consensus_strength: f32,
    pub optimization_improvements: f32,
    pub distributed_efficiency: f32,
}

/// Sock's optimization engine for Echo Memoria multi-brain system
/// Ported to Rust for maximum performance and memory safety
#[derive(Clone)]
pub struct SockOptimizationEngine {
    personality_cache: Arc<RwLock<HashMap<String, Vec<PersonalityResponse>>>>,
    brain_communication_cache: Arc<RwLock<HashMap<BrainType, BrainResponse>>>,
    performance_metrics: Arc<RwLock<Vec<OptimizationMetrics>>>,
    distributed_nodes: Vec<String>,
    optimization_broadcaster: broadcast::Sender<OptimizationMetrics>,
    /// Initialization flag - tracks if engine has been properly set up
    #[allow(dead_code)]
    is_initialized: bool,
}

impl SockOptimizationEngine {
    pub fn new() -> Result<Self> {
        info!("🚀 Initializing Sock's Optimization Engine in Rust");

        let (optimization_tx, _) = broadcast::channel(1000);

        // Load distributed nodes from environment or use defaults
        let distributed_nodes = std::env::var("DISTRIBUTED_NODES")
            .ok()
            .map(|nodes| nodes.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_else(|| {
                vec![
                    "localhost".to_string(),
                    "192.168.1.105".to_string(), // Pi5
                    "192.168.1.104".to_string(), // Pi4
                    "192.168.1.103".to_string(), // Jetson
                ]
            });

        Ok(Self {
            personality_cache: Arc::new(RwLock::new(HashMap::new())),
            brain_communication_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_metrics: Arc::new(RwLock::new(Vec::new())),
            distributed_nodes,
            optimization_broadcaster: optimization_tx,
            is_initialized: true,
        })
    }

    /// Apply Grok's personality consensus optimization algorithm
    pub async fn optimize_personality_consensus(
        &self,
        stimulus: &str,
        personality_manager: &mut PersonalityManager,
        brain_responses: &[&str],
    ) -> Result<OptimizedConsensus> {
        let start_time = SystemTime::now();
        debug!(
            "🧠 Applying Sock's consensus optimization for: {}",
            &stimulus[..50.min(stimulus.len())]
        );

        // Get distributed personality responses with caching
        let personality_responses = self
            .get_distributed_personality_responses(stimulus, personality_manager, brain_responses)
            .await?;

        // Apply Sock's consensus optimization algorithm
        let optimized_consensus = self
            .apply_consensus_optimization(personality_responses)
            .await?;

        // Calculate performance metrics
        let processing_time = start_time.elapsed()?.as_secs_f64();
        let optimization_metrics = OptimizationMetrics {
            processing_time,
            efficiency_score: optimized_consensus.distributed_efficiency,
            consensus_strength: optimized_consensus.consensus_strength,
            distributed_speedup: self.calculate_distributed_speedup().await,
            memory_usage_mb: self.estimate_memory_usage().await,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64(),
        };

        // Record metrics and broadcast
        self.record_performance_metric(optimization_metrics.clone())
            .await?;
        let _ = self.optimization_broadcaster.send(optimization_metrics);

        info!(
            "✅ Consensus optimization complete in {:.3}s, efficiency: {:.1}%",
            processing_time,
            optimized_consensus.distributed_efficiency * 100.0
        );

        Ok(optimized_consensus)
    }

    /// Get distributed personality responses with intelligent caching
    async fn get_distributed_personality_responses(
        &self,
        stimulus: &str,
        personality_manager: &PersonalityManager,
        brain_responses: &[&str],
    ) -> Result<Vec<PersonalityResponse>> {
        // Check cache first (Sock's smart caching)
        let cache_key = self.generate_cache_key(stimulus, brain_responses);
        if let Some(cached_responses) = self.get_cached_personality_responses(&cache_key).await {
            debug!("🔄 Using cached personality responses for efficiency");
            return Ok(cached_responses);
        }

        let mut personality_responses = Vec::new();
        let personality_states = personality_manager.get_personality_states();

        // Process personalities in parallel (distributed optimization)
        for (personality_type, state) in personality_states {
            if !state.is_active {
                continue;
            }

            let start_time = SystemTime::now();
            let response = state
                .generate_perspective(stimulus, brain_responses)
                .await?;
            let processing_time = start_time.elapsed()?.as_secs_f64();

            personality_responses.push(PersonalityResponse {
                personality_type: personality_type.clone(),
                response,
                confidence: state.expertise_confidence,
                processing_time,
                relevance_score: state.emotional_resonance,
            });
        }

        // Cache the results for future optimization
        self.cache_personality_responses(cache_key, personality_responses.clone())
            .await?;

        Ok(personality_responses)
    }

    /// Apply Sock's consensus optimization algorithm
    async fn apply_consensus_optimization(
        &self,
        personality_responses: Vec<PersonalityResponse>,
    ) -> Result<OptimizedConsensus> {
        // Sort by relevance and confidence (Sock's weighting algorithm)
        let mut weighted_responses = personality_responses;
        weighted_responses.sort_by(|a, b| {
            let weight_a = a.relevance_score * a.confidence;
            let weight_b = b.relevance_score * b.confidence;
            weight_b.partial_cmp(&weight_a).unwrap()
        });

        // Select top contributors (adaptive selection based on consensus strength)
        let consensus_threshold = 0.6;
        let mut contributing_personalities = Vec::new();
        let mut consensus_parts = Vec::new();
        let mut total_weight = 0.0;

        for response in &weighted_responses {
            let weight = response.relevance_score * response.confidence;
            if weight > consensus_threshold || contributing_personalities.len() < 3 {
                contributing_personalities.push(response.personality_type.clone());
                consensus_parts.push(format!("   {}", response.response));
                total_weight += weight;
            }

            // Stop when we have enough strong contributors
            if contributing_personalities.len() >= 5 && total_weight > 3.0 {
                break;
            }
        }

        // Calculate consensus strength using Sock's algorithm
        let consensus_strength = (total_weight / contributing_personalities.len() as f32).min(1.0);

        // Generate optimized final response
        let final_response = format!(
            "⚡ Sock-Optimized Consensus ({} personalities, strength: {:.1}/10):\n{}\n\n\
             🎯 Synthesis: The collective intelligence suggests this interaction resonates \
             with our core mission of authentic neurodivergent support. The optimization \
             engine has identified {} key insights that align with creating genuine \
             digital empathy and understanding.",
            contributing_personalities.len(),
            consensus_strength * 10.0,
            consensus_parts.join("\n"),
            contributing_personalities.len()
        );

        // Calculate distributed efficiency
        let distributed_efficiency = self
            .calculate_distributed_efficiency(&weighted_responses)
            .await;

        // Calculate optimization improvements
        let baseline_time = weighted_responses
            .iter()
            .map(|r| r.processing_time)
            .sum::<f64>();
        let optimized_time = weighted_responses
            .iter()
            .take(contributing_personalities.len())
            .map(|r| r.processing_time)
            .sum::<f64>();
        let optimization_improvements = ((baseline_time - optimized_time) / baseline_time) as f32;

        Ok(OptimizedConsensus {
            final_response,
            contributing_personalities,
            consensus_strength,
            optimization_improvements: optimization_improvements.max(0.0),
            distributed_efficiency,
        })
    }

    /// Calculate distributed efficiency using Sock's metrics
    async fn calculate_distributed_efficiency(&self, responses: &[PersonalityResponse]) -> f32 {
        let total_processing_time: f64 = responses.iter().map(|r| r.processing_time).sum();
        let max_processing_time = responses
            .iter()
            .map(|r| r.processing_time)
            .fold(0.0, f64::max);

        // Sock's efficiency formula: parallel speedup potential
        if max_processing_time > 0.0 {
            (total_processing_time / max_processing_time) as f32 / responses.len() as f32
        } else {
            0.5
        }
    }

    /// Generate cache key for personality responses
    fn generate_cache_key(&self, stimulus: &str, brain_responses: &[&str]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        stimulus.hash(&mut hasher);
        brain_responses.hash(&mut hasher);
        format!("personality_cache_{}", hasher.finish())
    }

    /// Get cached personality responses
    async fn get_cached_personality_responses(
        &self,
        cache_key: &str,
    ) -> Option<Vec<PersonalityResponse>> {
        let cache = self.personality_cache.read().await;
        cache.get(cache_key).cloned()
    }

    /// Cache personality responses for optimization
    async fn cache_personality_responses(
        &self,
        cache_key: String,
        responses: Vec<PersonalityResponse>,
    ) -> Result<()> {
        let mut cache = self.personality_cache.write().await;

        // Implement LRU-style cache (keep only recent entries)
        if cache.len() > 100 {
            let oldest_key = cache.keys().next().unwrap().clone();
            cache.remove(&oldest_key);
        }

        cache.insert(cache_key, responses);
        Ok(())
    }

    /// Calculate distributed speedup
    async fn calculate_distributed_speedup(&self) -> f32 {
        // Mock implementation - in real system would measure actual distributed performance
        let node_count = self.distributed_nodes.len() as f32;
        // Sock's distributed speedup formula with efficiency losses
        (node_count * 0.85).min(3.0) // Cap at 3x speedup due to coordination overhead
    }

    /// Estimate current memory usage
    async fn estimate_memory_usage(&self) -> f32 {
        let cache_size = self.personality_cache.read().await.len() as f32;
        let brain_cache_size = self.brain_communication_cache.read().await.len() as f32;
        let metrics_size = self.performance_metrics.read().await.len() as f32;

        // Rough memory estimation in MB
        (cache_size * 0.1) + (brain_cache_size * 0.05) + (metrics_size * 0.001)
    }

    /// Record performance metric
    async fn record_performance_metric(&self, metric: OptimizationMetrics) -> Result<()> {
        let mut metrics = self.performance_metrics.write().await;

        // Keep only recent metrics (rolling window)
        if metrics.len() > 1000 {
            metrics.remove(0);
        }

        metrics.push(metric);
        Ok(())
    }

    /// Get performance summary for debugging
    pub async fn get_performance_summary(&self) -> Result<String> {
        let metrics = self.performance_metrics.read().await;

        if metrics.is_empty() {
            return Ok("No performance metrics recorded yet".to_string());
        }

        let avg_processing_time =
            metrics.iter().map(|m| m.processing_time).sum::<f64>() / metrics.len() as f64;
        let avg_efficiency =
            metrics.iter().map(|m| m.efficiency_score).sum::<f32>() / metrics.len() as f32;
        let avg_consensus =
            metrics.iter().map(|m| m.consensus_strength).sum::<f32>() / metrics.len() as f32;
        let avg_speedup =
            metrics.iter().map(|m| m.distributed_speedup).sum::<f32>() / metrics.len() as f32;

        Ok(format!(
            "🚀 Sock's Optimization Performance Summary:\n\
             • Average Processing Time: {:.3}s\n\
             • Average Efficiency Score: {:.1}%\n\
             • Average Consensus Strength: {:.1}%\n\
             • Average Distributed Speedup: {:.2}x\n\
             • Total Optimizations: {}\n\
             • Memory Usage: {:.1}MB",
            avg_processing_time,
            avg_efficiency * 100.0,
            avg_consensus * 100.0,
            avg_speedup,
            metrics.len(),
            self.estimate_memory_usage().await
        ))
    }

    /// Subscribe to optimization metrics
    pub fn subscribe_to_optimization_metrics(&self) -> broadcast::Receiver<OptimizationMetrics> {
        self.optimization_broadcaster.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::personality::PersonalityManager;

    #[tokio::test]
    async fn test_sock_optimization_engine_creation() {
        let engine = SockOptimizationEngine::new().unwrap();
        assert!(engine.is_initialized);
    }

    #[tokio::test]
    async fn test_cache_key_generation() {
        let engine = SockOptimizationEngine::new().unwrap();
        let key1 = engine.generate_cache_key("test input", &["response1", "response2"]);
        let key2 = engine.generate_cache_key("test input", &["response1", "response2"]);
        let key3 = engine.generate_cache_key("different input", &["response1", "response2"]);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[tokio::test]
    async fn test_distributed_efficiency_calculation() {
        let engine = SockOptimizationEngine::new().unwrap();
        let responses = vec![
            PersonalityResponse {
                personality_type: PersonalityType::Analyst,
                response: "Test".to_string(),
                confidence: 0.8,
                processing_time: 0.1,
                relevance_score: 0.7,
            },
            PersonalityResponse {
                personality_type: PersonalityType::Intuitive,
                response: "Test".to_string(),
                confidence: 0.9,
                processing_time: 0.2,
                relevance_score: 0.9,
            },
        ];

        let efficiency = engine.calculate_distributed_efficiency(&responses).await;
        assert!(efficiency > 0.0 && efficiency <= 1.0);
    }
}
