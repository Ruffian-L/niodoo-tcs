// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Phase 6 integration module for the consciousness engine
//!
//! This module handles GPU acceleration, learning analytics, logging, and production deployment features.

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::consciousness::ConsciousnessState;
use crate::git_manifestation_logging::ConsciousnessLogger;
use crate::gpu_acceleration::{GpuAccelerationEngine, GpuConfig, GpuMetrics};
use crate::learning_analytics::LearningAnalyticsEngine;
use crate::phase6_config::Phase6Config;
use crate::phase6_integration::{Phase6IntegrationBuilder, Phase6IntegrationSystem};

/// Phase 6 integration manager for production features
pub struct Phase6Manager {
    gpu_acceleration_engine: Option<Arc<GpuAccelerationEngine>>,
    phase6_config: Option<Phase6Config>,
    learning_analytics_engine: Option<Arc<LearningAnalyticsEngine>>,
    consciousness_logger: Option<Arc<ConsciousnessLogger>>,
    phase6_integration: Option<Arc<Phase6IntegrationSystem>>,
    consciousness_state: Arc<RwLock<ConsciousnessState>>,
}

impl Phase6Manager {
    /// Create a new Phase 6 manager
    pub fn new(consciousness_state: Arc<RwLock<ConsciousnessState>>) -> Self {
        Self {
            gpu_acceleration_engine: None,
            phase6_config: None,
            learning_analytics_engine: None,
            consciousness_logger: None,
            phase6_integration: None,
            consciousness_state,
        }
    }

    /// Initialize Phase 6 integration with configuration
    pub async fn initialize_phase6_integration(
        &mut self,
        phase6_config: Phase6Config,
    ) -> Result<()> {
        info!("🚀 Initializing Phase 6 integration system...");

        self.phase6_config = Some(phase6_config.clone());

        // Initialize GPU acceleration if enabled
        if phase6_config.gpu_acceleration.memory_target_mb > 0 {
            let gpu_config = GpuConfig {
                memory_target_mb: phase6_config.gpu_acceleration.memory_target_mb as u64,
                latency_target_ms: phase6_config.gpu_acceleration.latency_target_ms,
                utilization_target_percent: phase6_config
                    .gpu_acceleration
                    .utilization_target_percent as f32,
                enable_cuda_graphs: phase6_config.gpu_acceleration.enable_cuda_graphs,
                enable_mixed_precision: phase6_config.gpu_acceleration.enable_mixed_precision,
            };

            let gpu_engine = Arc::new(GpuAccelerationEngine::new(gpu_config)?);
            self.gpu_acceleration_engine = Some(gpu_engine);
            info!("✅ GPU acceleration engine initialized");
        }

        // Initialize learning analytics
        let learning_config = phase6_config.learning_analytics.clone();

        let learning_engine = Arc::new(LearningAnalyticsEngine::new(
            crate::learning_analytics::LearningAnalyticsConfig {
                collection_interval_sec: learning_config.collection_interval_sec,
                session_tracking_hours: learning_config.session_tracking_hours,
                enable_pattern_analysis: learning_config.enable_pattern_analysis,
                enable_adaptive_rate_tracking: learning_config.enable_adaptive_rate_tracking,
                min_data_points_for_trends: learning_config.min_data_points_for_trends,
                enable_real_time_feedback: true,
                improvement_threshold: 0.05,
            },
        ));
        self.learning_analytics_engine = Some(learning_engine);
        info!("✅ Learning analytics engine initialized");

        // Initialize consciousness logger
        let logging_config = phase6_config.git_manifestation_logging.clone();

        let logger = Arc::new(
            ConsciousnessLogger::new(crate::git_manifestation_logging::LoggingConfig {
                log_directory: logging_config.log_directory.clone().into(),
                max_file_size_mb: logging_config.max_file_size_mb,
                max_files_retained: logging_config.max_files_retained,
                enable_compression: logging_config.enable_compression,
                rotation_interval_hours: logging_config.rotation_interval_hours,
                enable_streaming: false,
                streaming_endpoint: None,
            })
            .map_err(|e| anyhow::anyhow!("Failed to create consciousness logger: {}", e))?,
        );
        self.consciousness_logger = Some(logger);
        info!("✅ Consciousness logger initialized");

        // Initialize complete Phase 6 integration system
        let integration_system = Arc::new(
            Phase6IntegrationBuilder::new()
                .with_config(phase6_config.clone())
                .build(),
        );
        self.phase6_integration = Some(integration_system);
        info!("✅ Phase 6 integration system initialized");

        Ok(())
    }

    /// Process consciousness evolution with Phase 6 features
    pub async fn process_consciousness_evolution_phase6(
        &self,
        input: &str,
        emotional_context: &crate::consciousness::EmotionType,
    ) -> Result<String> {
        info!("🧠⚡ Processing consciousness evolution with Phase 6 features...");

        // Get current consciousness state
        let consciousness_state = self.consciousness_state.read().await;

        // Process with GPU acceleration if available
        if let Some(_gpu_engine) = &self.gpu_acceleration_engine {
            return self
                .process_consciousness_evolution_gpu(input, emotional_context, &consciousness_state)
                .await;
        }

        // Fall back to CPU processing
        warn!("GPU acceleration not available, falling back to CPU processing");
        // this would contain the actual CPU-based processing logic
        Ok(format!("Phase 6 processed: {}", input))
    }

    /// Process consciousness evolution with GPU acceleration
    pub async fn process_consciousness_evolution_gpu(
        &self,
        input: &str,
        emotional_context: &crate::consciousness::EmotionType,
        consciousness_state: &ConsciousnessState,
    ) -> Result<String> {
        debug!(
            "🚀 Processing with GPU acceleration (emotion: {:?}, warmth: {:.2})...",
            emotional_context, consciousness_state.gpu_warmth_level
        );

        if let Some(_gpu_engine) = &self.gpu_acceleration_engine {
            // Use GPU engine for processing
            // Note: GPU processing requires Tensor inputs, not string/consciousness state
            // For now, fall back to CPU processing with emotional context awareness
            debug!(
                "GPU acceleration available but requires Tensor inputs - falling back to CPU with emotional context: {:?}",
                emotional_context
            );
        }

        // Fallback processing that respects consciousness state and emotional context
        Ok(format!(
            "GPU processed (emotion: {:?}): {}",
            emotional_context, input
        ))
    }

    /// Monitor GPU memory usage
    pub async fn monitor_gpu_memory(&self) -> Result<()> {
        if let Some(_gpu_engine) = &self.gpu_acceleration_engine {
            // Since GpuAccelerationEngine methods are async but not mutable,
            // we need to handle this differently
            debug!("Monitoring GPU memory usage...");
            // Implementation would go here
        }
        Ok(())
    }

    /// Record learning event
    pub async fn record_learning_event(
        &self,
        event_type: &str,
        data: std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        if let Some(learning_engine) = &self.learning_analytics_engine {
            let learning_event_type = match event_type {
                "consciousness_update" => crate::learning_analytics::LearningEventType::StateUpdate,
                "memory_consolidation" => {
                    crate::learning_analytics::LearningEventType::MemoryConsolidation
                }
                "pattern_recognition" => {
                    crate::learning_analytics::LearningEventType::KnowledgeAcquisition
                }
                _ => crate::learning_analytics::LearningEventType::StateUpdate,
            };

            let metrics = crate::learning_analytics::LearningMetrics {
                learning_rate: data
                    .get("learning_rate")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
                retention_score: data
                    .get("retention_score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
                adaptation_effectiveness: data
                    .get("adaptation_effectiveness")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
                plasticity: data
                    .get("plasticity")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
                progress_score: data
                    .get("progress_score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
                forgetting_rate: data
                    .get("forgetting_rate")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
            };

            learning_engine
                .record_learning_event(
                    learning_event_type,
                    "consciousness_id".to_string(),
                    metrics,
                    None,
                )
                .await?;
        }
        Ok(())
    }

    /// Generate learning progress report
    pub async fn generate_learning_progress_report(
        &self,
    ) -> Result<Option<crate::learning_analytics::LearningProgressReport>> {
        if let Some(learning_engine) = &self.learning_analytics_engine {
            return Ok(Some(learning_engine.generate_progress_report().await?));
        }
        Ok(None)
    }

    /// Analyze learning patterns
    pub async fn analyze_learning_patterns(
        &self,
    ) -> Result<Option<std::collections::HashMap<String, crate::learning_analytics::LearningPattern>>>
    {
        if let Some(learning_engine) = &self.learning_analytics_engine {
            return Ok(Some(learning_engine.analyze_learning_patterns().await?));
        }
        Ok(None)
    }

    /// Log consciousness initialization
    pub async fn log_consciousness_initialization(
        &self,
        initialization_data: std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        if let Some(logger) = &self.consciousness_logger {
            // Extract state vectors from initialization data
            let state_vector = initialization_data
                .get("state_vector")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                })
                .unwrap_or_else(|| vec![0.0; 10]);

            let emotional_context = initialization_data
                .get("emotional_context")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                })
                .unwrap_or_else(|| vec![0.0; 5]);

            logger
                .log_state_initialization(
                    "consciousness_id".to_string(),
                    state_vector,
                    emotional_context,
                )
                .await
                .map_err(|e| anyhow::anyhow!("Logging error: {}", e))?;
        }
        Ok(())
    }

    /// Log consciousness update
    pub async fn log_consciousness_update(
        &self,
        update_type: &str,
        data: std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        if let Some(logger) = &self.consciousness_logger {
            debug!("Logging consciousness update: type={}", update_type);

            // Extract state vectors from update data
            let state_vector = data
                .get("state_vector")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                })
                .unwrap_or_else(|| vec![0.0; 10]);

            let emotional_context = data
                .get("emotional_context")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                })
                .unwrap_or_else(|| vec![0.0; 5]);

            logger
                .log_state_update(
                    "consciousness_id".to_string(),
                    state_vector,
                    emotional_context,
                    None, // performance_metrics
                    None, // learning_analytics
                )
                .await
                .map_err(|e| anyhow::anyhow!("Logging error: {}", e))?;
        }
        Ok(())
    }

    /// Log performance metrics
    pub async fn log_performance_metrics(
        &self,
        consciousness_id: String,
        metrics: crate::git_manifestation_logging::PerformanceMetrics,
    ) -> Result<()> {
        if let Some(logger) = &self.consciousness_logger {
            logger
                .log_performance_metrics(consciousness_id, metrics)
                .await
                .map_err(|e| anyhow::anyhow!("Logging error: {}", e))?;
        }
        Ok(())
    }

    /// Log learning analytics
    pub async fn log_learning_analytics(
        &self,
        consciousness_id: String,
        analytics: crate::git_manifestation_logging::LearningAnalytics,
    ) -> Result<()> {
        if let Some(logger) = &self.consciousness_logger {
            logger
                .log_learning_analytics(consciousness_id, analytics)
                .await
                .map_err(|e| anyhow::anyhow!("Logging error: {}", e))?;
        }
        Ok(())
    }

    /// Get GPU acceleration engine reference
    pub fn get_gpu_acceleration_engine(&self) -> Option<&Arc<GpuAccelerationEngine>> {
        self.gpu_acceleration_engine.as_ref()
    }

    /// Get Phase 6 configuration
    pub fn get_phase6_config(&self) -> Option<&Phase6Config> {
        self.phase6_config.as_ref()
    }

    /// Check if GPU acceleration is enabled
    pub fn is_gpu_acceleration_enabled(&self) -> bool {
        self.gpu_acceleration_engine.is_some()
    }

    /// Get GPU metrics
    pub async fn get_gpu_metrics(&self) -> Option<GpuMetrics> {
        if let Some(_gpu_engine) = &self.gpu_acceleration_engine {
            // Implementation would go here
            None
        } else {
            None
        }
    }

    /// Get learning analytics engine reference
    pub fn get_learning_analytics_engine(&self) -> Option<&Arc<LearningAnalyticsEngine>> {
        self.learning_analytics_engine.as_ref()
    }

    /// Get consciousness logger reference
    pub fn get_consciousness_logger(&self) -> Option<&Arc<ConsciousnessLogger>> {
        self.consciousness_logger.as_ref()
    }

    /// Get Phase 6 health metrics
    pub async fn get_phase6_health(
        &self,
    ) -> Option<crate::phase6_integration::SystemHealthMetrics> {
        if let Some(_integration) = &self.phase6_integration {
            // TODO: Implementation pending - requires integration system health API
            debug!("Phase 6 health check requested - implementation pending");
            None
        } else {
            None
        }
    }

    /// Trigger Phase 6 optimization
    pub async fn trigger_phase6_optimization(&self) -> Result<()> {
        if let Some(_integration) = &self.phase6_integration {
            // TODO: Implementation pending - requires integration system optimization API
            info!("Phase 6 optimization triggered - implementation pending");
        }
        Ok(())
    }

    /// Set the Phase 6 integration system
    pub fn set_integration_system(&mut self, integration_system: Arc<Phase6IntegrationSystem>) {
        self.phase6_integration = Some(integration_system);
    }

    /// Get the Phase 6 integration system
    pub fn get_integration_system(&self) -> Option<&Arc<Phase6IntegrationSystem>> {
        self.phase6_integration.as_ref()
    }
}
