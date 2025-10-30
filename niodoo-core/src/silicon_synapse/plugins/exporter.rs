// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Exporter plugin trait for Silicon Synapse monitoring
//!
//! This module defines the trait interface for pluggable metric exporters.

use crate::silicon_synapse::plugins::collector::CollectedMetrics;
use crate::silicon_synapse::plugins::detector::DetectionResult;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use uuid::Uuid;

/// Result type for exporter operations
pub type ExporterResult<T> = Result<T, ExporterError>;

/// Error types for exporter operations
#[derive(Debug, thiserror::Error)]
pub enum ExporterError {
    #[error("Exporter initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Export failed: {0}")]
    ExportFailed(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),
}

/// Configuration for an exporter plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExporterConfig {
    /// Unique identifier for this exporter instance
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Export format/type
    pub format: String,
    /// Export endpoint URL
    pub endpoint: String,
    /// Whether this exporter is enabled
    pub enabled: bool,
    /// Export batch size
    pub batch_size: usize,
    /// Export interval
    pub export_interval_ms: u64,
    /// Custom exporter parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

impl Default for ExporterConfig {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: "Unknown Exporter".to_string(),
            format: "json".to_string(),
            endpoint: "http://localhost:8080/metrics".to_string(),
            enabled: true,
            batch_size: 100,
            export_interval_ms: 5000,
            parameters: HashMap::new(),
        }
    }
}

/// Export data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportData {
    /// Exporter ID that created this export
    pub exporter_id: String,
    /// Export timestamp
    pub timestamp: SystemTime,
    /// Export format
    pub format: String,
    /// Raw export data
    pub data: Vec<u8>,
    /// Metadata about the export
    pub metadata: HashMap<String, String>,
}

/// Exporter health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExporterHealth {
    /// Whether the exporter is healthy
    pub is_healthy: bool,
    /// Last successful export timestamp
    pub last_export: Option<SystemTime>,
    /// Number of consecutive failures
    pub failure_count: u32,
    /// Error message if unhealthy
    pub error_message: Option<String>,
    /// Export statistics
    pub stats: ExporterStats,
}

/// Export statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExporterStats {
    /// Total number of exports
    pub total_exports: u64,
    /// Number of successful exports
    pub successful_exports: u64,
    /// Number of failed exports
    pub failed_exports: u64,
    /// Total bytes exported
    pub total_bytes_exported: u64,
    /// Average export duration
    pub avg_export_duration_ms: f64,
    /// Last export duration
    pub last_export_duration_ms: Option<f64>,
}

impl Default for ExporterStats {
    fn default() -> Self {
        Self {
            total_exports: 0,
            successful_exports: 0,
            failed_exports: 0,
            total_bytes_exported: 0,
            avg_export_duration_ms: 0.0,
            last_export_duration_ms: None,
        }
    }
}

/// Trait for pluggable metric exporters
#[async_trait]
pub trait MetricExporter: Send + Sync {
    /// Get the exporter's unique identifier
    fn id(&self) -> &str;

    /// Get the exporter's human-readable name
    fn name(&self) -> &str;

    /// Get the exporter's configuration
    fn config(&self) -> &ExporterConfig;

    /// Initialize the exporter
    async fn initialize(&mut self) -> ExporterResult<()>;

    /// Export collected metrics
    async fn export_metrics(&self, metrics: &[CollectedMetrics]) -> ExporterResult<ExportData>;

    /// Export detection results
    async fn export_detections(&self, detections: &[DetectionResult])
        -> ExporterResult<ExportData>;

    /// Export combined data (metrics + detections)
    async fn export_combined(
        &self,
        metrics: &[CollectedMetrics],
        detections: &[DetectionResult],
    ) -> ExporterResult<ExportData>;

    /// Shutdown the exporter
    async fn shutdown(&mut self) -> ExporterResult<()>;

    /// Check exporter health
    async fn health_check(&self) -> ExporterResult<ExporterHealth>;

    /// Update configuration
    async fn update_config(&mut self, config: ExporterConfig) -> ExporterResult<()>;
}

/// Base exporter implementation with common functionality
pub struct BaseExporter {
    config: ExporterConfig,
    stats: ExporterStats,
    last_export: Option<SystemTime>,
    failure_count: u32,
}

impl BaseExporter {
    /// Create a new base exporter
    pub fn new(config: ExporterConfig) -> Self {
        Self {
            config,
            stats: ExporterStats::default(),
            last_export: None,
            failure_count: 0,
        }
    }

    /// Update export statistics
    pub fn update_stats(
        &mut self,
        success: bool,
        bytes_exported: u64,
        duration: std::time::Duration,
    ) {
        self.stats.total_exports += 1;

        if success {
            self.stats.successful_exports += 1;
            self.stats.total_bytes_exported += bytes_exported;
            self.failure_count = 0;
            self.last_export = Some(SystemTime::now());
        } else {
            self.stats.failed_exports += 1;
            self.failure_count += 1;
        }

        let duration_ms = duration.as_millis() as f64;
        self.stats.last_export_duration_ms = Some(duration_ms);

        // Update average duration
        let total = self.stats.total_exports as f64;
        self.stats.avg_export_duration_ms =
            (self.stats.avg_export_duration_ms * (total - 1.0) + duration_ms) / total;
    }

    /// Get current health status
    pub fn get_health(&self) -> ExporterHealth {
        ExporterHealth {
            is_healthy: self.failure_count < 5, // Consider unhealthy after 5 consecutive failures
            last_export: self.last_export,
            failure_count: self.failure_count,
            error_message: if self.failure_count >= 5 {
                Some(format!(
                    "Exporter has failed {} consecutive times",
                    self.failure_count
                ))
            } else {
                None
            },
            stats: self.stats.clone(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ExporterConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ExporterConfig) {
        self.config = config;
    }
}

/// Example JSON API exporter
pub struct JsonApiExporter {
    base: BaseExporter,
    client: reqwest::Client,
}

impl JsonApiExporter {
    /// Create a new JSON API exporter
    pub fn new(config: ExporterConfig) -> Self {
        Self {
            base: BaseExporter::new(config),
            client: reqwest::Client::new(),
        }
    }

    /// Serialize data to JSON
    fn serialize_to_json<T: Serialize>(&self, data: &T) -> ExporterResult<Vec<u8>> {
        serde_json::to_vec(data).map_err(|e| ExporterError::SerializationError(e.to_string()))
    }

    /// Send data to the configured endpoint
    async fn send_data(&self, data: &[u8]) -> ExporterResult<()> {
        let response = self
            .client
            .post(&self.base.config().endpoint)
            .header("Content-Type", "application/json")
            .body(data.to_vec())
            .send()
            .await
            .map_err(|e| ExporterError::ConnectionError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(ExporterError::ExportFailed(format!(
                "HTTP error: {}",
                response.status()
            )));
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl MetricExporter for JsonApiExporter {
    fn id(&self) -> &str {
        &self.base.config().id
    }

    fn name(&self) -> &str {
        &self.base.config().name
    }

    fn config(&self) -> &ExporterConfig {
        self.base.config()
    }

    async fn initialize(&mut self) -> ExporterResult<()> {
        // Test connection to endpoint
        let test_data = serde_json::json!({
            "test": true,
            "timestamp": SystemTime::now()
        });

        let json_data = self.serialize_to_json(&test_data)?;
        self.send_data(&json_data).await?;

        Ok(())
    }

    async fn export_metrics(&self, metrics: &[CollectedMetrics]) -> ExporterResult<ExportData> {
        let start_time = std::time::Instant::now();

        let export_payload = serde_json::json!({
            "type": "metrics",
            "timestamp": SystemTime::now(),
            "count": metrics.len(),
            "data": metrics
        });

        let json_data = self.serialize_to_json(&export_payload)?;
        self.send_data(&json_data).await?;

        let duration = start_time.elapsed();
        // Note: In a real implementation, you'd update stats here

        Ok(ExportData {
            exporter_id: self.id().to_string(),
            timestamp: SystemTime::now(),
            format: "json".to_string(),
            data: json_data,
            metadata: HashMap::new(),
        })
    }

    async fn export_detections(
        &self,
        detections: &[DetectionResult],
    ) -> ExporterResult<ExportData> {
        let start_time = std::time::Instant::now();

        let export_payload = serde_json::json!({
            "type": "detections",
            "timestamp": SystemTime::now(),
            "count": detections.len(),
            "data": detections
        });

        let json_data = self.serialize_to_json(&export_payload)?;
        self.send_data(&json_data).await?;

        let duration = start_time.elapsed();
        // Note: In a real implementation, you'd update stats here

        Ok(ExportData {
            exporter_id: self.id().to_string(),
            timestamp: SystemTime::now(),
            format: "json".to_string(),
            data: json_data,
            metadata: HashMap::new(),
        })
    }

    async fn export_combined(
        &self,
        metrics: &[CollectedMetrics],
        detections: &[DetectionResult],
    ) -> ExporterResult<ExportData> {
        let start_time = std::time::Instant::now();

        let export_payload = serde_json::json!({
            "type": "combined",
            "timestamp": SystemTime::now(),
            "metrics_count": metrics.len(),
            "detections_count": detections.len(),
            "metrics": metrics,
            "detections": detections
        });

        let json_data = self.serialize_to_json(&export_payload)?;
        self.send_data(&json_data).await?;

        let duration = start_time.elapsed();
        // Note: In a real implementation, you'd update stats here

        Ok(ExportData {
            exporter_id: self.id().to_string(),
            timestamp: SystemTime::now(),
            format: "json".to_string(),
            data: json_data,
            metadata: HashMap::new(),
        })
    }

    async fn shutdown(&mut self) -> ExporterResult<()> {
        Ok(())
    }

    async fn health_check(&self) -> ExporterResult<ExporterHealth> {
        Ok(self.base.get_health())
    }

    async fn update_config(&mut self, config: ExporterConfig) -> ExporterResult<()> {
        self.base.update_config(config);
        Ok(())
    }
}

/// Example Prometheus exporter
pub struct PrometheusExporter {
    base: BaseExporter,
    /// Prometheus metrics registry (future: full Prometheus integration)
    #[allow(dead_code)]
    metrics_registry: prometheus::Registry,
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter
    pub fn new(config: ExporterConfig) -> Self {
        Self {
            base: BaseExporter::new(config),
            metrics_registry: prometheus::Registry::new(),
        }
    }

    /// Convert metrics to Prometheus format
    fn convert_to_prometheus(&self, metrics: &[CollectedMetrics]) -> String {
        let mut output = String::new();

        for collected_metrics in metrics {
            for (metric_name, metric_value) in &collected_metrics.metrics {
                let metric_line = match metric_value {
                    crate::silicon_synapse::plugins::collector::MetricValue::Counter(value) => {
                        format!("{}_total {}\n", metric_name, value)
                    }
                    crate::silicon_synapse::plugins::collector::MetricValue::Gauge(value) => {
                        format!("{} {}\n", metric_name, value)
                    }
                    crate::silicon_synapse::plugins::collector::MetricValue::Float(value) => {
                        format!("{} {}\n", metric_name, value)
                    }
                    crate::silicon_synapse::plugins::collector::MetricValue::Integer(value) => {
                        format!("{} {}\n", metric_name, value)
                    }
                    _ => continue, // Skip non-numeric values
                };
                output.push_str(&metric_line);
            }
        }

        output
    }
}

#[async_trait::async_trait]
impl MetricExporter for PrometheusExporter {
    fn id(&self) -> &str {
        &self.base.config().id
    }

    fn name(&self) -> &str {
        &self.base.config().name
    }

    fn config(&self) -> &ExporterConfig {
        self.base.config()
    }

    async fn initialize(&mut self) -> ExporterResult<()> {
        // Initialize Prometheus registry
        Ok(())
    }

    async fn export_metrics(&self, metrics: &[CollectedMetrics]) -> ExporterResult<ExportData> {
        let start_time = std::time::Instant::now();

        let prometheus_data = self.convert_to_prometheus(metrics);
        let data_bytes = prometheus_data.as_bytes().to_vec();

        let duration = start_time.elapsed();
        // Note: In a real implementation, you'd update stats here

        Ok(ExportData {
            exporter_id: self.id().to_string(),
            timestamp: SystemTime::now(),
            format: "prometheus".to_string(),
            data: data_bytes,
            metadata: HashMap::new(),
        })
    }

    async fn export_detections(
        &self,
        _detections: &[DetectionResult],
    ) -> ExporterResult<ExportData> {
        // Prometheus doesn't directly support detection results
        // Convert to metrics format
        Ok(ExportData {
            exporter_id: self.id().to_string(),
            timestamp: SystemTime::now(),
            format: "prometheus".to_string(),
            data: vec![],
            metadata: HashMap::new(),
        })
    }

    async fn export_combined(
        &self,
        metrics: &[CollectedMetrics],
        _detections: &[DetectionResult],
    ) -> ExporterResult<ExportData> {
        self.export_metrics(metrics).await
    }

    async fn shutdown(&mut self) -> ExporterResult<()> {
        Ok(())
    }

    async fn health_check(&self) -> ExporterResult<ExporterHealth> {
        Ok(self.base.get_health())
    }

    async fn update_config(&mut self, config: ExporterConfig) -> ExporterResult<()> {
        self.base.update_config(config);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_exporter_config_default() {
        let config = ExporterConfig::default();
        assert!(!config.id.is_empty());
        assert_eq!(config.name, "Unknown Exporter");
        assert_eq!(config.format, "json");
        assert_eq!(config.endpoint, "http://localhost:8080/metrics");
        assert!(config.enabled);
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.export_interval_ms, 5000);
    }

    #[test]
    fn test_base_exporter_stats() {
        let config = ExporterConfig::default();
        let mut exporter = BaseExporter::new(config);

        // Initial state
        let health = exporter.get_health();
        assert!(health.is_healthy);
        assert_eq!(health.stats.total_exports, 0);

        // Update stats
        exporter.update_stats(true, 1024, Duration::from_millis(100));
        let health = exporter.get_health();
        assert!(health.is_healthy);
        assert_eq!(health.stats.total_exports, 1);
        assert_eq!(health.stats.successful_exports, 1);
        assert_eq!(health.stats.failed_exports, 0);
        assert_eq!(health.stats.total_bytes_exported, 1024);
        assert_eq!(health.stats.avg_export_duration_ms, 100.0);

        // Multiple failures
        for _ in 0..6 {
            exporter.update_stats(false, 0, Duration::from_millis(50));
        }

        let health = exporter.get_health();
        assert!(!health.is_healthy);
        assert_eq!(health.failure_count, 6);
        assert!(health.error_message.is_some());
    }

    #[test]
    fn test_prometheus_conversion() {
        let config = ExporterConfig {
            format: "prometheus".to_string(),
            ..Default::default()
        };
        let exporter = PrometheusExporter::new(config);

        let mut metrics = HashMap::new();
        metrics.insert(
            "cpu_usage".to_string(),
            crate::silicon_synapse::plugins::collector::MetricValue::Gauge(75.5),
        );
        metrics.insert(
            "memory_total".to_string(),
            crate::silicon_synapse::plugins::collector::MetricValue::Counter(8192),
        );

        let collected_metrics = vec![CollectedMetrics {
            collector_id: "test-collector".to_string(),
            timestamp: SystemTime::now(),
            collection_duration: Duration::from_millis(10),
            metrics,
            metadata: HashMap::new(),
        }];

        let prometheus_output = exporter.convert_to_prometheus(&collected_metrics);
        assert!(prometheus_output.contains("cpu_usage 75.5"));
        assert!(prometheus_output.contains("memory_total_total 8192"));
    }
}
