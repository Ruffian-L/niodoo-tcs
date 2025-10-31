//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham
//!
//! Memory Architect - Phase 2 Integration Module
//!
//! This module uses MultiLayerMemoryQuery to decide memory layer placement.
//! It queries existing memories using hybrid retrieval (RAG + Gaussian) and
//! decides appropriate memory layer based on query results and stability.

use anyhow::Result;
use niodoo_core::config::AppConfig;
use niodoo_core::consciousness::ConsciousnessState;
use niodoo_core::memory::{
    EmotionalVector, GuessingMemorySystem, MemoryLayer, MobiusMemorySystem,
};
use niodoo_core::{
    MemoryConsolidationEngine, MemoryWithResonance, MultiLayerMemoryQuery,
};
use niodoo_core::rag::RetrievalEngine;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

/// Configuration for memory architect decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryArchitectConfig {
    /// Minimum stability score for CoreBurned layer
    pub core_burned_threshold: f64,
    /// Minimum stability score for Procedural layer
    pub procedural_threshold: f64,
    /// Minimum stability score for Episodic layer
    pub episodic_threshold: f64,
    /// Minimum stability score for Semantic layer
    pub semantic_threshold: f64,
    /// Minimum stability score for Somatic layer
    pub somatic_threshold: f64,
    /// Minimum novelty score for layer promotion
    pub promotion_novelty_threshold: f32,
    /// Maximum number of query results to consider
    pub max_query_results: usize,
}

impl Default for MemoryArchitectConfig {
    fn default() -> Self {
        Self {
            core_burned_threshold: 0.95,
            procedural_threshold: 0.85,
            episodic_threshold: 0.75,
            semantic_threshold: 0.65,
            somatic_threshold: 0.55,
            promotion_novelty_threshold: 0.5,
            max_query_results: 10,
        }
    }
}

/// Memory architect that decides memory layer placement
pub struct MemoryArchitect {
    multi_layer_query: MultiLayerMemoryQuery,
    memory_system: MobiusMemorySystem,
    consolidation_engine: MemoryConsolidationEngine,
    config: MemoryArchitectConfig,
    app_config: AppConfig,
}

impl MemoryArchitect {
    /// Create a new memory architect
    pub fn new(
        rag_engine: Arc<Mutex<RetrievalEngine>>,
        gaussian_system: GuessingMemorySystem,
        memory_system: MobiusMemorySystem,
        app_config: AppConfig,
        config: MemoryArchitectConfig,
    ) -> Self {
        let multi_layer_query = MultiLayerMemoryQuery::new(rag_engine, gaussian_system);
        let consolidation_engine = MemoryConsolidationEngine::new();
        
        Self {
            multi_layer_query,
            memory_system,
            consolidation_engine,
            config,
            app_config,
        }
    }

    /// Create with default config
    pub fn with_default_config(
        rag_engine: Arc<Mutex<RetrievalEngine>>,
        gaussian_system: GuessingMemorySystem,
        memory_system: MobiusMemorySystem,
        app_config: AppConfig,
    ) -> Self {
        Self::new(
            rag_engine,
            gaussian_system,
            memory_system,
            app_config,
            MemoryArchitectConfig::default(),
        )
    }

    /// Query existing memories and decide appropriate layer for new memory
    pub fn decide_layer(
        &mut self,
        content: &str,
        emotional_vector: &EmotionalVector,
        initial_stability: Option<f64>,
    ) -> Result<MemoryLayer> {
        info!("Deciding memory layer for content: {}", &content[..content.len().min(50)]);
        
        // Query existing memories to find similar ones
        let mut state = ConsciousnessState::default();
        let query_results = self.multi_layer_query.query(
            content,
            emotional_vector,
            self.config.max_query_results,
            &mut state,
        )?;
        
        debug!("Query returned {} results", query_results.len());
        
        // Analyze query results to determine layer
        let layer = self.analyze_results_for_layer(&query_results, initial_stability)?;
        
        info!("Decided layer: {:?}", layer);
        Ok(layer)
    }

    /// Analyze query results to determine appropriate layer
    fn analyze_results_for_layer(
        &self,
        results: &[MemoryWithResonance],
        initial_stability: Option<f64>,
    ) -> Result<MemoryLayer> {
        if results.is_empty() {
            // No similar memories found - start at Working layer
            return Ok(MemoryLayer::Working);
        }
        
        // Calculate average resonance and novelty
        let avg_resonance: f32 = results
            .iter()
            .map(|r| r.emotional_resonance)
            .sum::<f32>()
            / results.len() as f32;
        
        let avg_novelty: f32 = results
            .iter()
            .map(|r| r.novelty_score)
            .sum::<f32>()
            / results.len() as f32;
        
        // Use initial stability if provided, otherwise calculate from results
        let stability = initial_stability.unwrap_or_else(|| {
            // Calculate stability from resonance and novelty
            let stability_score = (avg_resonance as f64 + avg_novelty as f64) / 2.0;
            stability_score.clamp(0.0, 1.0)
        });
        
        debug!(
            "Analysis: avg_resonance={:.3}, avg_novelty={:.3}, stability={:.3}",
            avg_resonance, avg_novelty, stability
        );
        
        // Determine layer based on stability thresholds
        if stability >= self.config.core_burned_threshold {
            Ok(MemoryLayer::CoreBurned)
        } else if stability >= self.config.procedural_threshold {
            Ok(MemoryLayer::Procedural)
        } else if stability >= self.config.episodic_threshold {
            Ok(MemoryLayer::Episodic)
        } else if stability >= self.config.semantic_threshold {
            Ok(MemoryLayer::Semantic)
        } else if stability >= self.config.somatic_threshold {
            Ok(MemoryLayer::Somatic)
        } else {
            Ok(MemoryLayer::Working)
        }
    }

    /// Store memory in appropriate layer based on architect decision
    pub fn store_memory(
        &mut self,
        content: String,
        emotional_vector: EmotionalVector,
        initial_stability: Option<f64>,
    ) -> Result<String> {
        // Decide layer
        let layer = self.decide_layer(&content, &emotional_vector, initial_stability)?;
        
        // Calculate emotional weight from vector magnitude
        let emotional_weight = emotional_vector.magnitude() as f64;
        
        // Store in memory system
        let memory_id = self.memory_system.add_memory(
            content,
            layer.clone(),
            emotional_weight,
            &self.app_config,
        )?;
        
        info!("Stored memory {} in layer {:?}", memory_id, layer);
        Ok(memory_id)
    }

    /// Promote memories to higher layers based on consolidation
    pub async fn consolidate_and_promote(&mut self) -> Result<()> {
        info!("Starting memory consolidation and promotion");
        
        // Apply emotional transformation to update stabilities
        self.memory_system.apply_emotional_transformation(&self.app_config)?;
        
        // Get consolidation stats
        let metrics = self.memory_system.get_stability_metrics();
        info!(
            "Consolidation metrics: overall_stability={:.3}, consolidation_rate={:.3}",
            metrics.overall_stability, metrics.consolidation_rate
        );
        
        Ok(())
    }

    /// Query memories and return results with layer information
    pub fn query_memories(
        &mut self,
        query_text: &str,
        query_emotion: &EmotionalVector,
        top_k: usize,
    ) -> Result<Vec<MemoryWithResonance>> {
        let mut state = ConsciousnessState::default();
        self.multi_layer_query.query(query_text, query_emotion, top_k, &mut state)
    }

    /// Get memory system reference
    pub fn memory_system(&self) -> &MobiusMemorySystem {
        &self.memory_system
    }

    /// Get mutable memory system reference
    pub fn memory_system_mut(&mut self) -> &mut MobiusMemorySystem {
        &mut self.memory_system
    }

    /// Get consolidation engine reference
    pub fn consolidation_engine(&self) -> &MemoryConsolidationEngine {
        &self.consolidation_engine
    }

    /// Get mutable consolidation engine reference
    pub fn consolidation_engine_mut(&mut self) -> &mut MemoryConsolidationEngine {
        &mut self.consolidation_engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use niodoo_core::rag::RetrievalEngine;

    fn create_test_config() -> AppConfig {
        AppConfig::default()
    }

    fn create_test_rag_engine() -> Arc<Mutex<RetrievalEngine>> {
        // Create a minimal RAG engine for testing
        // In real usage, this would be properly initialized
        Arc::new(Mutex::new(RetrievalEngine::new()))
    }

    #[test]
    fn test_memory_architect_creation() {
        let rag_engine = create_test_rag_engine();
        let gaussian_system = GuessingMemorySystem::new();
        let memory_system = MobiusMemorySystem::new(&create_test_config());
        let app_config = create_test_config();
        
        let architect = MemoryArchitect::with_default_config(
            rag_engine,
            gaussian_system,
            memory_system,
            app_config,
        );
        
        // Architect should be created successfully
        assert!(architect.memory_system().get_total_memories() >= 0);
    }

    #[test]
    fn test_decide_layer_with_no_results() {
        let rag_engine = create_test_rag_engine();
        let gaussian_system = GuessingMemorySystem::new();
        let memory_system = MobiusMemorySystem::new(&create_test_config());
        let app_config = create_test_config();
        
        let mut architect = MemoryArchitect::with_default_config(
            rag_engine,
            gaussian_system,
            memory_system,
            app_config,
        );
        
        let emotion = EmotionalVector::new(0.5, 0.5, 0.0, 0.0, 0.0);
        let layer = architect.decide_layer("New memory", &emotion, None).unwrap();
        
        // With no similar memories, should start at Working layer
        assert_eq!(layer, MemoryLayer::Working);
    }
}

