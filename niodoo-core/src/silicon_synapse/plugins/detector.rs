// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Detector plugin trait for Silicon Synapse monitoring
//!
//! This module defines the trait interface for pluggable anomaly detection algorithms.

use crate::silicon_synapse::plugins::collector::CollectedMetrics;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use uuid::Uuid;

/// Result type for detector operations
pub type DetectorResult<T> = Result<T, DetectorError>;

/// Error types for detector operations
#[derive(Debug, thiserror::Error)]
pub enum DetectorError {
    #[error("Detector initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Detection failed: {0}")]
    DetectionFailed(String),

    #[error("Model training failed: {0}")]
    TrainingFailed(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    #[error("Model persistence error: {0}")]
    ModelPersistenceError(String),
}

/// Configuration for a detector plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorConfig {
    /// Unique identifier for this detector instance
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Detection algorithm type
    pub algorithm: String,
    /// Whether this detector is enabled
    pub enabled: bool,
    /// Sensitivity threshold (0.0 to 1.0)
    pub sensitivity: f64,
    /// Minimum samples required for detection
    pub min_samples: usize,
    /// Custom algorithm parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: "Unknown Detector".to_string(),
            algorithm: "statistical".to_string(),
            enabled: true,
            sensitivity: 0.5,
            min_samples: 100,
            parameters: HashMap::new(),
        }
    }
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Detector ID that produced this result
    pub detector_id: String,
    /// Timestamp when detection was performed
    pub timestamp: SystemTime,
    /// Whether an anomaly was detected
    pub anomaly_detected: bool,
    /// Anomaly score (0.0 to 1.0, higher = more anomalous)
    pub anomaly_score: f64,
    /// Confidence in the detection (0.0 to 1.0)
    pub confidence: f64,
    /// Affected metrics
    pub affected_metrics: Vec<String>,
    /// Detailed anomaly information
    pub anomalies: Vec<Anomaly>,
    /// Detection metadata
    pub metadata: HashMap<String, String>,
}

/// Individual anomaly details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Unique identifier for this anomaly
    pub id: Uuid,
    /// Metric name that triggered the anomaly
    pub metric_name: String,
    /// Observed value
    pub observed_value: f64,
    /// Expected value (from baseline)
    pub expected_value: f64,
    /// Deviation magnitude
    pub deviation: f64,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Severity level
    pub severity: Severity,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Types of anomalies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnomalyType {
    /// Statistical outlier (3-sigma rule)
    StatisticalOutlier,
    /// Trend anomaly (sudden change in trend)
    TrendAnomaly,
    /// Seasonal anomaly (unexpected seasonal pattern)
    SeasonalAnomaly,
    /// Correlation anomaly (unexpected correlation change)
    CorrelationAnomaly,
    /// Pattern anomaly (unexpected pattern)
    PatternAnomaly,
    /// Custom anomaly type
    Custom(String),
}

/// Severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum Severity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Detector health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorHealth {
    /// Whether the detector is healthy
    pub is_healthy: bool,
    /// Last successful detection timestamp
    pub last_detection: Option<SystemTime>,
    /// Number of consecutive failures
    pub failure_count: u32,
    /// Error message if unhealthy
    pub error_message: Option<String>,
    /// Detection statistics
    pub stats: DetectorStats,
    /// Model status
    pub model_status: ModelStatus,
}

/// Detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorStats {
    /// Total number of detections
    pub total_detections: u64,
    /// Number of successful detections
    pub successful_detections: u64,
    /// Number of failed detections
    pub failed_detections: u64,
    /// Number of anomalies detected
    pub anomalies_detected: u64,
    /// Average detection duration
    pub avg_detection_duration_ms: f64,
    /// Last detection duration
    pub last_detection_duration_ms: Option<f64>,
}

impl Default for DetectorStats {
    fn default() -> Self {
        Self {
            total_detections: 0,
            successful_detections: 0,
            failed_detections: 0,
            anomalies_detected: 0,
            avg_detection_duration_ms: 0.0,
            last_detection_duration_ms: None,
        }
    }
}

/// Model status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    /// Whether the model is trained
    pub is_trained: bool,
    /// Number of training samples used
    pub training_samples: usize,
    /// Last training timestamp
    pub last_training: Option<SystemTime>,
    /// Model accuracy (if available)
    pub accuracy: Option<f64>,
    /// Model version
    pub version: String,
}

impl Default for ModelStatus {
    fn default() -> Self {
        Self {
            is_trained: false,
            training_samples: 0,
            last_training: None,
            accuracy: None,
            version: "1.0.0".to_string(),
        }
    }
}

/// Trait for pluggable anomaly detectors
#[async_trait]
pub trait AnomalyDetector: Send + Sync {
    /// Get the detector's unique identifier
    fn id(&self) -> &str;

    /// Get the detector's human-readable name
    fn name(&self) -> &str;

    /// Get the detector's configuration
    fn config(&self) -> &DetectorConfig;

    /// Initialize the detector
    async fn initialize(&mut self) -> DetectorResult<()>;

    /// Train the detector with historical data
    async fn train(&mut self, training_data: &[CollectedMetrics]) -> DetectorResult<()>;

    /// Detect anomalies in the given metrics
    async fn detect(&self, metrics: &CollectedMetrics) -> DetectorResult<DetectionResult>;

    /// Batch detect anomalies in multiple metric sets
    async fn detect_batch(
        &self,
        metrics_batch: &[CollectedMetrics],
    ) -> DetectorResult<Vec<DetectionResult>>;

    /// Shutdown the detector
    async fn shutdown(&mut self) -> DetectorResult<()>;

    /// Check detector health
    async fn health_check(&self) -> DetectorResult<DetectorHealth>;

    /// Update configuration
    async fn update_config(&mut self, config: DetectorConfig) -> DetectorResult<()>;

    /// Save the detector model to persistent storage
    async fn save_model(&self, path: &str) -> DetectorResult<()>;

    /// Load the detector model from persistent storage
    async fn load_model(&mut self, path: &str) -> DetectorResult<()>;
}

/// Base detector implementation with common functionality
pub struct BaseDetector {
    config: DetectorConfig,
    stats: DetectorStats,
    model_status: ModelStatus,
    last_detection: Option<SystemTime>,
    failure_count: u32,
}

impl BaseDetector {
    /// Create a new base detector
    pub fn new(config: DetectorConfig) -> Self {
        Self {
            config,
            stats: DetectorStats::default(),
            model_status: ModelStatus::default(),
            last_detection: None,
            failure_count: 0,
        }
    }

    /// Update detection statistics
    pub fn update_stats(
        &mut self,
        success: bool,
        anomaly_detected: bool,
        duration: std::time::Duration,
    ) {
        self.stats.total_detections += 1;

        if success {
            self.stats.successful_detections += 1;
            self.failure_count = 0;
            self.last_detection = Some(SystemTime::now());

            if anomaly_detected {
                self.stats.anomalies_detected += 1;
            }
        } else {
            self.stats.failed_detections += 1;
            self.failure_count += 1;
        }

        let duration_ms = duration.as_millis() as f64;
        self.stats.last_detection_duration_ms = Some(duration_ms);

        // Update average duration
        let total = self.stats.total_detections as f64;
        self.stats.avg_detection_duration_ms =
            (self.stats.avg_detection_duration_ms * (total - 1.0) + duration_ms) / total;
    }

    /// Update model status after training
    pub fn update_model_status(&mut self, training_samples: usize, accuracy: Option<f64>) {
        self.model_status.is_trained = true;
        self.model_status.training_samples = training_samples;
        self.model_status.last_training = Some(SystemTime::now());
        self.model_status.accuracy = accuracy;
    }

    /// Get current health status
    pub fn get_health(&self) -> DetectorHealth {
        DetectorHealth {
            is_healthy: self.failure_count < 5, // Consider unhealthy after 5 consecutive failures
            last_detection: self.last_detection,
            failure_count: self.failure_count,
            error_message: if self.failure_count >= 5 {
                Some(format!(
                    "Detector has failed {} consecutive times",
                    self.failure_count
                ))
            } else {
                None
            },
            stats: self.stats.clone(),
            model_status: self.model_status.clone(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &DetectorConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: DetectorConfig) {
        self.config = config;
    }
}

/// Example statistical anomaly detector
pub struct StatisticalDetector {
    base: BaseDetector,
    baseline_stats: HashMap<String, MetricStats>,
}

#[derive(Debug, Clone)]
struct MetricStats {
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    sample_count: usize,
}

impl StatisticalDetector {
    /// Create a new statistical detector
    pub fn new(config: DetectorConfig) -> Self {
        Self {
            base: BaseDetector::new(config),
            baseline_stats: HashMap::new(),
        }
    }

    /// Calculate baseline statistics from training data
    fn calculate_baseline_stats(&mut self, training_data: &[CollectedMetrics]) {
        let mut metric_values: HashMap<String, Vec<f64>> = HashMap::new();

        // Collect all metric values
        for metrics in training_data {
            for (metric_name, metric_value) in &metrics.metrics {
                if let Some(value) = metric_value.as_f64() {
                    metric_values
                        .entry(metric_name.clone())
                        .or_default()
                        .push(value);
                }
            }
        }

        // Calculate statistics for each metric
        for (metric_name, values) in metric_values {
            if values.len() >= 2 {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
                let std_dev = variance.sqrt();

                let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                self.baseline_stats.insert(
                    metric_name,
                    MetricStats {
                        mean,
                        std_dev,
                        min,
                        max,
                        sample_count: values.len(),
                    },
                );
            }
        }
    }

    /// Detect anomalies using statistical methods
    fn detect_anomalies(&self, metrics: &CollectedMetrics) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();
        let threshold = self.base.config().sensitivity * 3.0; // Convert sensitivity to sigma threshold

        for (metric_name, metric_value) in &metrics.metrics {
            if let Some(value) = metric_value.as_f64() {
                if let Some(stats) = self.baseline_stats.get(metric_name) {
                    let deviation = (value - stats.mean).abs();
                    let sigma_deviation = if stats.std_dev > 0.0 {
                        deviation / stats.std_dev
                    } else {
                        0.0
                    };

                    if sigma_deviation > threshold {
                        let severity = if sigma_deviation > 6.0 {
                            Severity::Critical
                        } else if sigma_deviation > 4.0 {
                            Severity::High
                        } else if sigma_deviation > 3.0 {
                            Severity::Medium
                        } else {
                            Severity::Low
                        };

                        anomalies.push(Anomaly {
                            id: Uuid::new_v4(),
                            metric_name: metric_name.clone(),
                            observed_value: value,
                            expected_value: stats.mean,
                            deviation: sigma_deviation,
                            anomaly_type: AnomalyType::StatisticalOutlier,
                            severity,
                            context: HashMap::new(),
                        });
                    }
                }
            }
        }

        anomalies
    }
}

#[async_trait::async_trait]
impl AnomalyDetector for StatisticalDetector {
    fn id(&self) -> &str {
        &self.base.config().id
    }

    fn name(&self) -> &str {
        &self.base.config().name
    }

    fn config(&self) -> &DetectorConfig {
        self.base.config()
    }

    async fn initialize(&mut self) -> DetectorResult<()> {
        Ok(())
    }

    async fn train(&mut self, training_data: &[CollectedMetrics]) -> DetectorResult<()> {
        if training_data.len() < self.base.config().min_samples {
            return Err(DetectorError::InsufficientData(format!(
                "Need at least {} samples, got {}",
                self.base.config().min_samples,
                training_data.len()
            )));
        }

        self.calculate_baseline_stats(training_data);
        self.base.update_model_status(training_data.len(), None);

        Ok(())
    }

    async fn detect(&self, metrics: &CollectedMetrics) -> DetectorResult<DetectionResult> {
        let start_time = std::time::Instant::now();

        if !self.base.model_status.is_trained {
            return Err(DetectorError::InsufficientData(
                "Model not trained".to_string(),
            ));
        }

        let anomalies = self.detect_anomalies(metrics);
        let anomaly_detected = !anomalies.is_empty();
        let anomaly_score = if anomaly_detected {
            anomalies.iter().map(|a| a.deviation).fold(0.0, f64::max) / 10.0 // Normalize to 0-1
        } else {
            0.0
        };

        let confidence = if anomaly_detected {
            anomalies
                .iter()
                .map(|a| match a.severity {
                    Severity::Critical => 0.95,
                    Severity::High => 0.85,
                    Severity::Medium => 0.75,
                    Severity::Low => 0.65,
                })
                .fold(0.0, f64::max)
        } else {
            0.9 // High confidence in normal behavior
        };

        let affected_metrics: Vec<String> =
            anomalies.iter().map(|a| a.metric_name.clone()).collect();

        let duration = start_time.elapsed();
        // Note: In a real implementation, you'd update stats here

        Ok(DetectionResult {
            detector_id: self.id().to_string(),
            timestamp: SystemTime::now(),
            anomaly_detected,
            anomaly_score,
            confidence,
            affected_metrics,
            anomalies,
            metadata: HashMap::new(),
        })
    }

    async fn detect_batch(
        &self,
        metrics_batch: &[CollectedMetrics],
    ) -> DetectorResult<Vec<DetectionResult>> {
        let mut results = Vec::new();

        for metrics in metrics_batch {
            let result = self.detect(metrics).await?;
            results.push(result);
        }

        Ok(results)
    }

    async fn shutdown(&mut self) -> DetectorResult<()> {
        Ok(())
    }

    async fn health_check(&self) -> DetectorResult<DetectorHealth> {
        Ok(self.base.get_health())
    }

    async fn update_config(&mut self, config: DetectorConfig) -> DetectorResult<()> {
        self.base.update_config(config);
        Ok(())
    }

    async fn save_model(&self, _path: &str) -> DetectorResult<()> {
        // TODO: Implement model persistence
        Ok(())
    }

    async fn load_model(&mut self, _path: &str) -> DetectorResult<()> {
        // TODO: Implement model loading
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_detector_config_default() {
        let config = DetectorConfig::default();
        assert!(!config.id.is_empty());
        assert_eq!(config.name, "Unknown Detector");
        assert_eq!(config.algorithm, "statistical");
        assert!(config.enabled);
        assert_eq!(config.sensitivity, 0.5);
        assert_eq!(config.min_samples, 100);
    }

    #[tokio::test]
    async fn test_statistical_detector() {
        let config = DetectorConfig {
            id: "test-detector".to_string(),
            name: "Test Statistical Detector".to_string(),
            algorithm: "statistical".to_string(),
            enabled: true,
            sensitivity: 0.5,
            min_samples: 10,
            parameters: HashMap::new(),
        };

        let mut detector = StatisticalDetector::new(config);

        // Initialize
        detector.initialize().await.unwrap();

        // Create training data
        let mut training_data = Vec::new();
        for i in 0..20 {
            let mut metrics = HashMap::new();
            metrics.insert(
                "cpu_usage".to_string(),
                MetricValue::Float(50.0 + (i as f64 * 0.1)),
            );
            metrics.insert(
                "memory_usage".to_string(),
                MetricValue::Float(60.0 + (i as f64 * 0.05)),
            );

            training_data.push(CollectedMetrics {
                collector_id: "test-collector".to_string(),
                timestamp: SystemTime::now(),
                collection_duration: Duration::from_millis(10),
                metrics,
                metadata: HashMap::new(),
            });
        }

        // Train detector
        detector.train(&training_data).await.unwrap();

        // Test normal data
        let mut normal_metrics = HashMap::new();
        normal_metrics.insert("cpu_usage".to_string(), MetricValue::Float(52.0));
        normal_metrics.insert("memory_usage".to_string(), MetricValue::Float(61.0));

        let normal_data = CollectedMetrics {
            collector_id: "test-collector".to_string(),
            timestamp: SystemTime::now(),
            collection_duration: Duration::from_millis(10),
            metrics: normal_metrics,
            metadata: HashMap::new(),
        };

        let result = detector.detect(&normal_data).await.unwrap();
        assert!(!result.anomaly_detected);
        assert_eq!(result.anomaly_score, 0.0);

        // Test anomalous data
        let mut anomalous_metrics = HashMap::new();
        anomalous_metrics.insert("cpu_usage".to_string(), MetricValue::Float(200.0)); // Way outside normal range
        anomalous_metrics.insert("memory_usage".to_string(), MetricValue::Float(61.0));

        let anomalous_data = CollectedMetrics {
            collector_id: "test-collector".to_string(),
            timestamp: SystemTime::now(),
            collection_duration: Duration::from_millis(10),
            metrics: anomalous_metrics,
            metadata: HashMap::new(),
        };

        let result = detector.detect(&anomalous_data).await.unwrap();
        assert!(result.anomaly_detected);
        assert!(result.anomaly_score > 0.0);
        assert!(!result.anomalies.is_empty());
    }
}
