// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Plugin registry for Silicon Synapse monitoring
//!
//! This module provides a centralized registry for managing plugin instances.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::{atomic::AtomicU64, Arc};
use tokio::sync::{mpsc, RwLock};

use crate::silicon_synapse::plugins::collector::{Collector, CollectorConfig, CollectorHealth};
use crate::silicon_synapse::plugins::detector::{AnomalyDetector, DetectorConfig, DetectorHealth};
use crate::silicon_synapse::plugins::exporter::{ExporterConfig, ExporterHealth, MetricExporter};
use crate::silicon_synapse::telemetry_bus::TelemetrySender;

/// Error types for plugin registry operations
#[derive(Debug, thiserror::Error)]
pub enum PluginRegistryError {
    #[error("Plugin not found: {0}")]
    PluginNotFound(String),

    #[error("Plugin already exists: {0}")]
    PluginAlreadyExists(String),

    #[error("Plugin initialization failed: {0}")]
    PluginInitializationFailed(String),

    #[error("Plugin shutdown failed: {0}")]
    PluginShutdownFailed(String),

    #[error("Registry operation failed: {0}")]
    RegistryOperationFailed(String),
}

/// Plugin registry for managing all plugin instances
pub struct PluginRegistry {
    collectors: Arc<RwLock<HashMap<String, Arc<dyn Collector>>>>,
    detectors: Arc<RwLock<HashMap<String, Arc<dyn AnomalyDetector>>>>,
    exporters: Arc<RwLock<HashMap<String, Arc<dyn MetricExporter>>>>,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            collectors: Arc::new(RwLock::new(HashMap::new())),
            detectors: Arc::new(RwLock::new(HashMap::new())),
            exporters: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a collector plugin
    pub async fn register_collector(
        &self,
        mut collector: Box<dyn Collector>,
    ) -> Result<(), PluginRegistryError> {
        let id = collector.id().to_string();

        // Check if collector already exists
        {
            let collectors = self.collectors.read().await;
            if collectors.contains_key(&id) {
                return Err(PluginRegistryError::PluginAlreadyExists(id));
            }
        }

        // Initialize the collector
        collector
            .initialize()
            .await
            .map_err(|e| PluginRegistryError::PluginInitializationFailed(e.to_string()))?;

        // Register the collector - convert Box to Arc using raw pointers
        {
            let mut collectors = self.collectors.write().await;
            let raw_ptr = Box::into_raw(collector);
            let arc_collector = unsafe { Arc::from_raw(raw_ptr) };
            collectors.insert(id.clone(), arc_collector);
        }

        tracing::info!("Registered collector plugin: {}", id);
        Ok(())
    }

    /// Register a detector plugin
    pub async fn register_detector(
        &self,
        mut detector: Box<dyn AnomalyDetector>,
    ) -> Result<(), PluginRegistryError> {
        let id = detector.id().to_string();

        // Check if detector already exists
        {
            let detectors = self.detectors.read().await;
            if detectors.contains_key(&id) {
                return Err(PluginRegistryError::PluginAlreadyExists(id));
            }
        }

        // Initialize the detector
        detector
            .initialize()
            .await
            .map_err(|e| PluginRegistryError::PluginInitializationFailed(e.to_string()))?;

        // Register the detector - convert Box to Arc using raw pointers
        {
            let mut detectors = self.detectors.write().await;
            let raw_ptr = Box::into_raw(detector);
            let arc_detector = unsafe { Arc::from_raw(raw_ptr) };
            detectors.insert(id.clone(), arc_detector);
        }

        tracing::info!("Registered detector plugin: {}", id);
        Ok(())
    }

    /// Register an exporter plugin
    pub async fn register_exporter(
        &self,
        mut exporter: Box<dyn MetricExporter>,
    ) -> Result<(), PluginRegistryError> {
        let id = exporter.id().to_string();

        // Check if exporter already exists
        {
            let exporters = self.exporters.read().await;
            if exporters.contains_key(&id) {
                return Err(PluginRegistryError::PluginAlreadyExists(id));
            }
        }

        // Initialize the exporter
        exporter
            .initialize()
            .await
            .map_err(|e| PluginRegistryError::PluginInitializationFailed(e.to_string()))?;

        // Register the exporter - convert Box to Arc using raw pointers
        {
            let mut exporters = self.exporters.write().await;
            let raw_ptr = Box::into_raw(exporter);
            let arc_exporter = unsafe { Arc::from_raw(raw_ptr) };
            exporters.insert(id.clone(), arc_exporter);
        }

        tracing::info!("Registered exporter plugin: {}", id);
        Ok(())
    }

    /// Unregister a collector plugin
    pub async fn unregister_collector(&self, id: &str) -> Result<(), PluginRegistryError> {
        let _collector = {
            let mut collectors = self.collectors.write().await;
            collectors
                .remove(id)
                .ok_or_else(|| PluginRegistryError::PluginNotFound(id.to_string()))?
        };

        // Note: Cannot call shutdown on Arc<dyn Trait> since it requires &mut self
        // The collector will be dropped when this function returns
        tracing::info!("Unregistered collector plugin: {}", id);
        Ok(())
    }

    /// Unregister a detector plugin
    pub async fn unregister_detector(&self, id: &str) -> Result<(), PluginRegistryError> {
        let _detector = {
            let mut detectors = self.detectors.write().await;
            detectors
                .remove(id)
                .ok_or_else(|| PluginRegistryError::PluginNotFound(id.to_string()))?
        };

        // Note: Cannot call shutdown on Arc<dyn Trait> since it requires &mut self
        // The detector will be dropped when this function returns
        tracing::info!("Unregistered detector plugin: {}", id);
        Ok(())
    }

    /// Unregister an exporter plugin
    pub async fn unregister_exporter(&self, id: &str) -> Result<(), PluginRegistryError> {
        let _exporter = {
            let mut exporters = self.exporters.write().await;
            exporters
                .remove(id)
                .ok_or_else(|| PluginRegistryError::PluginNotFound(id.to_string()))?
        };

        // Note: Cannot call shutdown on Arc<dyn Trait> since it requires &mut self
        // The exporter will be dropped when this function returns
        tracing::info!("Unregistered exporter plugin: {}", id);
        Ok(())
    }

    /// Get a collector plugin by ID
    pub async fn get_collector(&self, id: &str) -> Result<Arc<dyn Collector>, PluginRegistryError> {
        let collectors = self.collectors.read().await;
        collectors
            .get(id)
            .cloned()
            .ok_or_else(|| PluginRegistryError::PluginNotFound(id.to_string()))
    }

    /// Get a detector plugin by ID
    pub async fn get_detector(
        &self,
        id: &str,
    ) -> Result<Arc<dyn AnomalyDetector>, PluginRegistryError> {
        let detectors = self.detectors.read().await;
        detectors
            .get(id)
            .cloned()
            .ok_or_else(|| PluginRegistryError::PluginNotFound(id.to_string()))
    }

    /// Get an exporter plugin by ID
    pub async fn get_exporter(
        &self,
        id: &str,
    ) -> Result<Arc<dyn MetricExporter>, PluginRegistryError> {
        let exporters = self.exporters.read().await;
        exporters
            .get(id)
            .cloned()
            .ok_or_else(|| PluginRegistryError::PluginNotFound(id.to_string()))
    }

    /// List all registered collector IDs
    pub async fn list_collectors(&self) -> Vec<String> {
        let collectors = self.collectors.read().await;
        collectors.keys().cloned().collect()
    }

    /// List all registered detector IDs
    pub async fn list_detectors(&self) -> Vec<String> {
        let detectors = self.detectors.read().await;
        detectors.keys().cloned().collect()
    }

    /// List all registered exporter IDs
    pub async fn list_exporters(&self) -> Vec<String> {
        let exporters = self.exporters.read().await;
        exporters.keys().cloned().collect()
    }

    /// Get health status of all collectors
    pub async fn get_collectors_health(&self) -> HashMap<String, CollectorHealth> {
        let collectors = self.collectors.read().await;
        let mut health_map = HashMap::new();

        for (id, collector) in collectors.iter() {
            if let Ok(health) = collector.health_check().await {
                health_map.insert(id.clone(), health);
            }
        }

        health_map
    }

    /// Get health status of all detectors
    pub async fn get_detectors_health(&self) -> HashMap<String, DetectorHealth> {
        let detectors = self.detectors.read().await;
        let mut health_map = HashMap::new();

        for (id, detector) in detectors.iter() {
            if let Ok(health) = detector.health_check().await {
                health_map.insert(id.clone(), health);
            }
        }

        health_map
    }

    /// Get health status of all exporters
    pub async fn get_exporters_health(&self) -> HashMap<String, ExporterHealth> {
        let exporters = self.exporters.read().await;
        let mut health_map = HashMap::new();

        for (id, exporter) in exporters.iter() {
            if let Ok(health) = exporter.health_check().await {
                health_map.insert(id.clone(), health);
            }
        }

        health_map
    }

    /// Shutdown all plugins
    pub async fn shutdown_all(&self) -> Result<(), PluginRegistryError> {
        tracing::info!("Shutting down all plugins");

        // Clear collectors (they will be dropped)
        {
            let mut collectors = self.collectors.write().await;
            collectors.clear();
        }

        // Clear detectors (they will be dropped)
        {
            let mut detectors = self.detectors.write().await;
            detectors.clear();
        }

        // Clear exporters (they will be dropped)
        {
            let mut exporters = self.exporters.write().await;
            exporters.clear();
        }

        tracing::info!("All plugins shutdown complete");
        Ok(())
    }

    /// Get registry statistics
    pub async fn get_stats(&self) -> RegistryStats {
        let collectors = self.collectors.read().await;
        let detectors = self.detectors.read().await;
        let exporters = self.exporters.read().await;

        RegistryStats {
            total_collectors: collectors.len(),
            total_detectors: detectors.len(),
            total_exporters: exporters.len(),
            total_plugins: collectors.len() + detectors.len() + exporters.len(),
        }
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub total_collectors: usize,
    pub total_detectors: usize,
    pub total_exporters: usize,
    pub total_plugins: usize,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Plugin factory for creating plugin instances
pub struct PluginFactory;

impl PluginFactory {
    /// Create a collector from configuration
    pub fn create_collector(
        config: CollectorConfig,
    ) -> Result<Box<dyn Collector>, PluginRegistryError> {
        match config.name.as_str() {
            "hardware" => {
                let telemetry_sender = TelemetrySender {
                    inner: mpsc::unbounded_channel().0,
                    dropped_events: Arc::new(AtomicU64::new(0)),
                };
                // Create HardwareConfig from CollectorConfig
                let hardware_config = crate::silicon_synapse::config::HardwareConfig::default();
                let bridge = crate::silicon_synapse::collectors::hardware::HardwareCollector::new(
                    hardware_config,
                    telemetry_sender,
                )
                .map_err(|e| {
                    PluginRegistryError::PluginInitializationFailed(format!(
                        "Hardware collector initialization failed: {}",
                        e
                    ))
                })?;
                Ok(Box::new(bridge))
            }
            "inference" => {
                // TODO: Create actual inference collector
                Err(PluginRegistryError::PluginInitializationFailed(
                    "Inference collector not implemented".to_string(),
                ))
            }
            "model_probe" => {
                // TODO: Create actual model probe collector
                Err(PluginRegistryError::PluginInitializationFailed(
                    "Model probe collector not implemented".to_string(),
                ))
            }
            _ => Err(PluginRegistryError::PluginInitializationFailed(format!(
                "Unknown collector type: {}",
                config.name
            ))),
        }
    }

    /// Create a detector from configuration
    pub fn create_detector(
        config: DetectorConfig,
    ) -> Result<Box<dyn AnomalyDetector>, PluginRegistryError> {
        match config.algorithm.as_str() {
            "statistical" => {
                use crate::silicon_synapse::plugins::detector::StatisticalDetector;
                Ok(Box::new(StatisticalDetector::new(config)))
            }
            "isolation_forest" => {
                // TODO: Implement Isolation Forest detector
                Err(PluginRegistryError::PluginInitializationFailed(
                    "Isolation Forest detector not implemented".to_string(),
                ))
            }
            "lstm_autoencoder" => {
                // TODO: Implement LSTM Autoencoder detector
                Err(PluginRegistryError::PluginInitializationFailed(
                    "LSTM Autoencoder detector not implemented".to_string(),
                ))
            }
            _ => Err(PluginRegistryError::PluginInitializationFailed(format!(
                "Unknown detector algorithm: {}",
                config.algorithm
            ))),
        }
    }

    /// Create an exporter from configuration
    pub fn create_exporter(
        config: ExporterConfig,
    ) -> Result<Box<dyn MetricExporter>, PluginRegistryError> {
        match config.format.as_str() {
            "json" => {
                use crate::silicon_synapse::plugins::exporter::JsonApiExporter;
                Ok(Box::new(JsonApiExporter::new(config)))
            }
            "prometheus" => {
                use crate::silicon_synapse::plugins::exporter::PrometheusExporter;
                Ok(Box::new(PrometheusExporter::new(config)))
            }
            "influxdb" => {
                // TODO: Implement InfluxDB exporter
                Err(PluginRegistryError::PluginInitializationFailed(
                    "InfluxDB exporter not implemented".to_string(),
                ))
            }
            _ => Err(PluginRegistryError::PluginInitializationFailed(format!(
                "Unknown exporter format: {}",
                config.format
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silicon_synapse::plugins::detector::{DetectorConfig, StatisticalDetector};

    #[tokio::test]
    async fn test_plugin_registry() {
        let registry = PluginRegistry::new();

        // Test empty registry
        assert_eq!(registry.list_collectors().await.len(), 0);
        assert_eq!(registry.list_detectors().await.len(), 0);
        assert_eq!(registry.list_exporters().await.len(), 0);

        // Test registering a detector
        let config = DetectorConfig {
            id: "test-detector".to_string(),
            name: "Test Detector".to_string(),
            algorithm: "statistical".to_string(),
            enabled: true,
            sensitivity: 0.5,
            min_samples: 10,
            parameters: std::collections::HashMap::new(),
        };

        let detector = Box::new(StatisticalDetector::new(config));
        registry.register_detector(detector).await.unwrap();

        assert_eq!(registry.list_detectors().await.len(), 1);
        assert!(registry
            .list_detectors()
            .await
            .contains(&"test-detector".to_string()));

        // Test getting detector
        let _detector = registry.get_detector("test-detector").await.unwrap();

        // Test unregistering detector
        registry.unregister_detector("test-detector").await.unwrap();
        assert_eq!(registry.list_detectors().await.len(), 0);
    }

    #[test]
    fn test_plugin_factory() {
        // Test creating statistical detector
        let config = DetectorConfig {
            id: "test-detector".to_string(),
            name: "Test Detector".to_string(),
            algorithm: "statistical".to_string(),
            enabled: true,
            sensitivity: 0.5,
            min_samples: 10,
            parameters: std::collections::HashMap::new(),
        };

        let detector = PluginFactory::create_detector(config).unwrap();
        assert_eq!(detector.id(), "test-detector");
        assert_eq!(detector.name(), "Test Detector");

        // Test unknown detector algorithm
        let unknown_config = DetectorConfig {
            algorithm: "unknown".to_string(),
            parameters: config.parameters.clone(),
        };

        let result = PluginFactory::create_detector(unknown_config);
        assert!(result.is_err());
    }
}
