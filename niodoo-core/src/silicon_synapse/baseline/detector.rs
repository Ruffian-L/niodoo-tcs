// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Anomaly detector for Silicon Synapse
//!
//! This module implements the anomaly detection system that compares live metrics
//! against learned baselines to identify deviations from normal operational behavior.

use super::model::BaselineModel;
use crate::silicon_synapse::aggregation::AggregatedMetrics;
use std::collections::HashMap;
use std::time::SystemTime;
use uuid::Uuid;

/// Anomaly detector that compares live metrics against baseline
pub struct AnomalyDetector {
    /// Current baseline model
    baseline: Option<BaselineModel>,
    /// Detector configuration
    config: DetectorConfig,
    /// Whether the detector is currently running
    is_running: std::sync::atomic::AtomicBool,
    /// Learning mode state
    learning_mode: bool,
    /// Samples collected during learning
    learning_samples: Vec<AggregatedMetrics>,
    /// When learning started
    learning_start: Option<SystemTime>,
}

/// Configuration for anomaly detection
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Whether anomaly detection is enabled
    pub enabled: bool,
    /// Learning duration in hours
    pub learning_duration_hours: u64,
    /// Minimum samples required for baseline
    pub min_samples_for_baseline: usize,
    /// Sigma threshold for univariate detection
    pub univariate_threshold_sigma: f64,
    /// Correlation threshold for multivariate detection
    pub multivariate_correlation_threshold: f64,
    /// Whether to enable multivariate detection
    pub enable_multivariate_detection: bool,
    /// Whether to enable learning mode
    pub enable_learning_mode: bool,
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Unique identifier for the anomaly
    pub id: Uuid,
    /// When the anomaly was detected
    pub timestamp: SystemTime,
    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,
    /// Severity level
    pub severity: Severity,
    /// Human-readable description
    pub description: String,
    /// Affected metrics
    pub affected_metrics: Vec<String>,
    /// Deviation scores for each metric
    pub deviation_scores: HashMap<String, f64>,
    /// Confidence in the detection (0.0 to 1.0)
    pub confidence: f64,
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    /// Security threat (unusual power/compute patterns)
    SecurityThreat,
    /// Model instability (repetitive loops, high entropy)
    ModelInstability,
    /// Performance degradation (increasing latency over time)
    PerformanceDegradation,
    /// Emergent behavior (novel activation patterns)
    EmergentBehavior,
    /// Hardware failure (temperature spikes, power issues)
    HardwareFailure,
    /// Unknown anomaly type
    Unknown,
}

/// Severity levels for anomalies
#[derive(Debug, Clone, PartialEq)]
pub enum Severity {
    /// Low severity - informational
    Low,
    /// Medium severity - requires attention
    Medium,
    /// High severity - requires immediate action
    High,
    /// Critical severity - system at risk
    Critical,
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    pub fn new(config: DetectorConfig) -> Self {
        Self {
            baseline: None,
            config,
            is_running: std::sync::atomic::AtomicBool::new(false),
            learning_mode: true,
            learning_samples: Vec::new(),
            learning_start: None,
        }
    }

    /// Start the anomaly detector
    pub async fn start(&mut self) -> Result<(), String> {
        if self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("Anomaly detector is already running".to_string());
        }

        tracing::info!("Starting anomaly detector in learning mode");
        self.is_running
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.learning_mode = true;
        self.learning_samples.clear();
        self.learning_start = Some(SystemTime::now());

        Ok(())
    }

    /// Stop the anomaly detector
    pub async fn stop(&mut self) -> Result<(), String> {
        if !self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }

        tracing::info!("Stopping anomaly detector");
        self.is_running
            .store(false, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Process new metrics (learning or detection mode)
    pub fn process_metrics(&mut self, metrics: &AggregatedMetrics) -> Vec<Anomaly> {
        if !self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
            return Vec::new();
        }

        if self.learning_mode {
            self.learn_from_metrics(metrics);
            Vec::new() // No anomalies during learning
        } else {
            self.detect_anomalies(metrics)
        }
    }

    /// Learn from metrics during learning mode
    fn learn_from_metrics(&mut self, metrics: &AggregatedMetrics) {
        self.learning_samples.push(metrics.clone());

        if let Some(start) = &self.learning_start {
            if let Ok(elapsed) = start.elapsed() {
                if elapsed.as_secs() > self.config.learning_duration_hours * 3600 {
                    self.build_baseline_from_samples();
                    self.learning_mode = false;
                }
            }

            // Persist baseline
            let path = "./data/baseline.json";
            if let Some(baseline) = &self.baseline {
                if let Ok(mut file) = std::fs::File::create(path) {
                    if let Err(e) = serde_json::to_writer(&mut file, baseline) {
                        tracing::error!("Failed to persist baseline: {}", e);
                    }
                }
            }
        }

        // Check if we have enough samples to build baseline
        if self.learning_samples.len() >= self.config.min_samples_for_baseline {
            self.build_baseline_from_samples();
            self.learning_mode = false;
            tracing::info!("Baseline learning complete, switching to detection mode");
        }
    }

    /// Build baseline model from collected samples
    fn build_baseline_from_samples(&mut self) {
        let mut baseline = BaselineModel::new();

        // Process all learning samples
        for sample in &self.learning_samples {
            baseline.update(sample);
        }

        self.baseline = Some(baseline);
        tracing::info!(
            "Baseline model built with {} samples",
            self.learning_samples.len()
        );
    }

    /// Detect anomalies in current metrics
    fn detect_anomalies(&self, metrics: &AggregatedMetrics) -> Vec<Anomaly> {
        let baseline = match &self.baseline {
            Some(b) => b,
            None => return Vec::new(),
        };

        let mut anomalies = Vec::new();

        // Univariate anomaly detection
        anomalies.extend(self.detect_univariate_anomalies(metrics, baseline));

        // Multivariate anomaly detection
        if self.config.enable_multivariate_detection {
            anomalies.extend(self.detect_multivariate_anomalies(metrics, baseline));
        }

        anomalies
    }

    /// Detect univariate anomalies using 3-sigma rule
    fn detect_univariate_anomalies(
        &self,
        metrics: &AggregatedMetrics,
        baseline: &BaselineModel,
    ) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        // Check hardware metrics
        anomalies.extend(self.check_hardware_metrics(metrics, baseline));

        // Check inference metrics
        anomalies.extend(self.check_inference_metrics(metrics, baseline));

        // Check model metrics
        anomalies.extend(self.check_model_metrics(
            metrics,
            baseline,
            "Dummy prompt",
            "Dummy output",
        ));

        anomalies
    }

    /// Check hardware metrics for anomalies
    fn check_hardware_metrics(
        &self,
        metrics: &AggregatedMetrics,
        baseline: &BaselineModel,
    ) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        // GPU temperature anomaly
        if baseline.univariate_stats.contains_key("gpu_temperature")
            && metrics.hardware_metrics.gpu_temperature.count > 0
        {
            let current_temp = metrics.hardware_metrics.gpu_temperature.mean;
            let deviation = baseline.calculate_deviation_score("gpu_temperature", current_temp);

            if deviation.abs() > self.config.univariate_threshold_sigma {
                anomalies.push(Anomaly {
                    id: Uuid::new_v4(),
                    timestamp: SystemTime::now(),
                    anomaly_type: AnomalyType::HardwareFailure,
                    severity: self.classify_hardware_severity(deviation.abs()),
                    description: format!(
                        "GPU temperature anomaly: {:.1}°C (deviation: {:.1}σ)",
                        current_temp, deviation
                    ),
                    affected_metrics: vec!["gpu_temperature".to_string()],
                    deviation_scores: vec![("gpu_temperature".to_string(), deviation.abs())]
                        .into_iter()
                        .collect(),
                    confidence: self.calculate_confidence(deviation.abs()),
                });
            }
        }

        // GPU power anomaly
        if baseline.univariate_stats.contains_key("gpu_power")
            && metrics.hardware_metrics.gpu_power.count > 0
        {
            let current_power = metrics.hardware_metrics.gpu_power.mean;
            let deviation = baseline.calculate_deviation_score("gpu_power", current_power);

            if deviation.abs() > self.config.univariate_threshold_sigma {
                anomalies.push(Anomaly {
                    id: Uuid::new_v4(),
                    timestamp: SystemTime::now(),
                    anomaly_type: AnomalyType::SecurityThreat,
                    severity: self.classify_security_severity(deviation.abs()),
                    description: format!(
                        "GPU power anomaly: {:.1}W (deviation: {:.1}σ)",
                        current_power, deviation
                    ),
                    affected_metrics: vec!["gpu_power".to_string()],
                    deviation_scores: vec![("gpu_power".to_string(), deviation.abs())]
                        .into_iter()
                        .collect(),
                    confidence: self.calculate_confidence(deviation.abs()),
                });
            }
        }

        anomalies
    }

    /// Check inference metrics for anomalies
    fn check_inference_metrics(
        &self,
        metrics: &AggregatedMetrics,
        baseline: &BaselineModel,
    ) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        // TTFT (Time To First Token) anomaly
        if baseline.univariate_stats.contains_key("ttft_ms")
            && metrics.inference_metrics.ttft_ms.count > 0
        {
            let current_ttft = metrics.inference_metrics.ttft_ms.mean;
            let deviation = baseline.calculate_deviation_score("ttft_ms", current_ttft);

            if deviation > self.config.univariate_threshold_sigma {
                anomalies.push(Anomaly {
                    id: Uuid::new_v4(),
                    timestamp: SystemTime::now(),
                    anomaly_type: AnomalyType::PerformanceDegradation,
                    severity: self.classify_performance_severity(deviation),
                    description: format!(
                        "TTFT performance degradation: {:.1}ms (deviation: {:.1}σ)",
                        current_ttft, deviation
                    ),
                    affected_metrics: vec!["ttft_ms".to_string()],
                    deviation_scores: vec![("ttft_ms".to_string(), deviation)]
                        .into_iter()
                        .collect(),
                    confidence: self.calculate_confidence(deviation),
                });
            }
        }

        // Error rate anomaly
        if baseline.univariate_stats.contains_key("error_rate")
            && metrics.inference_metrics.error_rate.count > 0
        {
            let current_error_rate = metrics.inference_metrics.error_rate.mean;
            let deviation = baseline.calculate_deviation_score("error_rate", current_error_rate);

            if deviation > self.config.univariate_threshold_sigma {
                anomalies.push(Anomaly {
                    id: Uuid::new_v4(),
                    timestamp: SystemTime::now(),
                    anomaly_type: AnomalyType::ModelInstability,
                    severity: self.classify_instability_severity(deviation),
                    description: format!(
                        "High error rate: {:.2}% (deviation: {:.1}σ)",
                        current_error_rate * 100.0,
                        deviation
                    ),
                    affected_metrics: vec!["error_rate".to_string()],
                    deviation_scores: vec![("error_rate".to_string(), deviation)]
                        .into_iter()
                        .collect(),
                    confidence: self.calculate_confidence(deviation),
                });
            }
        }

        anomalies
    }

    /// Check model metrics for anomalies
    fn check_model_metrics(
        &self,
        metrics: &AggregatedMetrics,
        baseline: &BaselineModel,
        prompt_context: &str,
        output_context: &str,
    ) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        // Check entropy anomalies (emergent behavior)
        for (layer, entropy_summary) in &metrics.model_metrics.entropy_by_layer {
            let metric_name = format!("entropy_layer_{}", layer);
            if baseline.univariate_stats.contains_key(&metric_name) && entropy_summary.count > 0 {
                let current_entropy = entropy_summary.mean;
                let deviation = baseline.calculate_deviation_score(&metric_name, current_entropy);

                if deviation.abs() > self.config.univariate_threshold_sigma {
                    anomalies.push(Anomaly {
                        id: Uuid::new_v4(),
                        timestamp: SystemTime::now(),
                        anomaly_type: AnomalyType::EmergentBehavior,
                        severity: self.classify_emergent_severity(deviation.abs()),
                        description: format!(
                            "Unusual entropy in layer {}: {:.3} (deviation: {:.1}σ)",
                            layer, current_entropy, deviation
                        ),
                        affected_metrics: vec![metric_name.clone()],
                        deviation_scores: vec![(metric_name, deviation.abs())]
                            .into_iter()
                            .collect(),
                        confidence: self.calculate_confidence(deviation.abs()),
                    });
                }
            }
        }

        anomalies
    }

    /// Detect multivariate anomalies using correlation violations
    fn detect_multivariate_anomalies(
        &self,
        metrics: &AggregatedMetrics,
        baseline: &BaselineModel,
    ) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        if let Some(multivariate_anomaly) = baseline.detect_multivariate_anomaly(metrics) {
            let mut affected_metrics = Vec::new();
            let mut deviation_scores = HashMap::new();

            for violation in &multivariate_anomaly.violations {
                affected_metrics.push(violation.metric1.clone());
                affected_metrics.push(violation.metric2.clone());
                deviation_scores.insert(
                    format!("{}:{}", violation.metric1, violation.metric2),
                    violation.deviation,
                );
            }

            anomalies.push(Anomaly {
                id: Uuid::new_v4(),
                timestamp: SystemTime::now(),
                anomaly_type: AnomalyType::Unknown, // Multivariate anomalies need further analysis
                severity: self.convert_multivariate_severity(multivariate_anomaly.severity),
                description: format!(
                    "Multivariate anomaly detected with {} correlation violations",
                    multivariate_anomaly.violations.len()
                ),
                affected_metrics,
                deviation_scores,
                confidence: 0.8, // Moderate confidence for multivariate detection
            });
        }

        anomalies
    }

    /// Classify hardware anomaly severity
    pub fn classify_hardware_severity(&self, deviation: f64) -> Severity {
        match deviation {
            d if d > 5.0 => Severity::Critical,
            d if d > 4.0 => Severity::High,
            d if d > 3.0 => Severity::Medium,
            _ => Severity::Low,
        }
    }

    /// Classify security anomaly severity
    pub fn classify_security_severity(&self, deviation: f64) -> Severity {
        match deviation {
            d if d > 4.0 => Severity::Critical,
            d if d > 3.0 => Severity::High,
            d if d > 2.0 => Severity::Medium,
            _ => Severity::Low,
        }
    }

    /// Classify performance anomaly severity
    fn classify_performance_severity(&self, deviation: f64) -> Severity {
        match deviation {
            d if d > 4.0 => Severity::High,
            d if d > 3.0 => Severity::Medium,
            d if d > 2.0 => Severity::Low,
            _ => Severity::Low,
        }
    }

    /// Classify instability anomaly severity
    fn classify_instability_severity(&self, deviation: f64) -> Severity {
        match deviation {
            d if d > 5.0 => Severity::Critical,
            d if d > 4.0 => Severity::High,
            d if d > 3.0 => Severity::Medium,
            _ => Severity::Low,
        }
    }

    /// Classify emergent behavior severity
    fn classify_emergent_severity(&self, deviation: f64) -> Severity {
        match deviation {
            d if d > 4.0 => Severity::High,
            d if d > 3.0 => Severity::Medium,
            d if d > 2.0 => Severity::Low,
            _ => Severity::Low,
        }
    }

    /// Convert multivariate severity to standard severity
    fn convert_multivariate_severity(
        &self,
        severity: super::model::MultivariateSeverity,
    ) -> Severity {
        match severity {
            super::model::MultivariateSeverity::Critical => Severity::Critical,
            super::model::MultivariateSeverity::High => Severity::High,
            super::model::MultivariateSeverity::Medium => Severity::Medium,
            super::model::MultivariateSeverity::Low => Severity::Low,
        }
    }

    /// Calculate confidence score based on deviation
    pub fn calculate_confidence(&self, deviation: f64) -> f64 {
        // Higher deviation = higher confidence
        (deviation / 10.0).min(1.0).max(0.1)
    }

    /// Check if detector is in learning mode
    pub fn is_learning_mode(&self) -> bool {
        self.learning_mode
    }

    /// Get current baseline model
    pub fn get_baseline(&self) -> Option<&BaselineModel> {
        self.baseline.as_ref()
    }

    /// Get number of learning samples collected
    pub fn get_learning_sample_count(&self) -> usize {
        self.learning_samples.len()
    }

    /// Force switch to detection mode (for testing)
    pub fn force_detection_mode(&mut self) {
        if !self.learning_samples.is_empty() {
            self.build_baseline_from_samples();
            self.learning_mode = false;
        }
    }
}

impl From<crate::silicon_synapse::config::BaselineConfig> for DetectorConfig {
    fn from(config: crate::silicon_synapse::config::BaselineConfig) -> Self {
        Self {
            enabled: config.enabled,
            learning_duration_hours: config.learning_window_hours,
            min_samples_for_baseline: config.min_samples,
            univariate_threshold_sigma: 3.0, // Default value
            multivariate_correlation_threshold: config.correlation_threshold,
            enable_multivariate_detection: config.enable_multivariate_analysis,
            enable_learning_mode: true, // Default value
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silicon_synapse::aggregation::{AggregatedMetrics, StatisticalSummary};

    fn create_test_metrics() -> AggregatedMetrics {
        AggregatedMetrics {
            timestamp: SystemTime::now(),
            window_start: SystemTime::now(),
            window_duration: std::time::Duration::from_secs(60),
            hardware_metrics: crate::silicon_synapse::aggregation::HardwareAggregatedMetrics {
                gpu_temperature: StatisticalSummary {
                    count: 1,
                    mean: 75.0,
                    std_dev: 5.0,
                    min: 70.0,
                    max: 80.0,
                    percentiles: HashMap::new(),
                },
                gpu_power: StatisticalSummary {
                    count: 1,
                    mean: 200.0,
                    std_dev: 20.0,
                    min: 180.0,
                    max: 220.0,
                    percentiles: HashMap::new(),
                },
                gpu_utilization: StatisticalSummary::default(),
                gpu_memory_usage: StatisticalSummary::default(),
                cpu_temperature: StatisticalSummary::default(),
                cpu_utilization: StatisticalSummary::default(),
                system_memory_usage: StatisticalSummary::default(),
            },
            inference_metrics: crate::silicon_synapse::aggregation::InferenceAggregatedMetrics {
                ttft_ms: StatisticalSummary {
                    count: 1,
                    mean: 100.0,
                    std_dev: 10.0,
                    min: 90.0,
                    max: 110.0,
                    percentiles: HashMap::new(),
                },
                tpot_ms: StatisticalSummary::default(),
                throughput_tps: StatisticalSummary::default(),
                error_rate: StatisticalSummary::default(),
                active_requests: StatisticalSummary::default(),
                completed_requests: StatisticalSummary::default(),
            },
            model_metrics: crate::silicon_synapse::aggregation::ModelAggregatedMetrics {
                entropy_by_layer: HashMap::new(),
                activation_sparsity_by_layer: HashMap::new(),
                activation_magnitude_by_layer: HashMap::new(),
            },
            correlations: Vec::new(),
        }
    }

    fn create_anomalous_metrics() -> AggregatedMetrics {
        let mut metrics = create_test_metrics();
        // Create anomalous GPU temperature (3+ sigma away)
        metrics.hardware_metrics.gpu_temperature.mean = 95.0; // 4 sigma away from 75
        metrics
    }

    #[tokio::test]
    async fn test_detector_creation() {
        let config = DetectorConfig::default();
        let detector = AnomalyDetector::new(config);
        assert!(detector.is_learning_mode());
    }

    #[tokio::test]
    async fn test_detector_start_stop() {
        let config = DetectorConfig::default();
        let mut detector = AnomalyDetector::new(config);

        assert!(detector.start().await.is_ok());
        assert!(detector.stop().await.is_ok());
    }

    #[test]
    fn test_learning_mode() {
        let config = DetectorConfig {
            min_samples_for_baseline: 2,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);

        let metrics = create_test_metrics();

        // First sample - should be in learning mode
        let anomalies = detector.process_metrics(&metrics);
        assert!(anomalies.is_empty());
        assert!(detector.is_learning_mode());
        assert_eq!(detector.get_learning_sample_count(), 1);

        // Second sample - should switch to detection mode
        let anomalies = detector.process_metrics(&metrics);
        assert!(anomalies.is_empty());
        assert!(!detector.is_learning_mode());
        assert!(detector.get_baseline().is_some());
    }

    #[test]
    fn test_anomaly_detection() {
        let config = DetectorConfig {
            min_samples_for_baseline: 1,
            univariate_threshold_sigma: 3.0,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);

        // Build baseline
        let normal_metrics = create_test_metrics();
        detector.process_metrics(&normal_metrics);

        // Test with anomalous metrics
        let anomalous_metrics = create_anomalous_metrics();
        let anomalies = detector.process_metrics(&anomalous_metrics);

        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::HardwareFailure);
        assert_eq!(anomalies[0].severity, Severity::High);
    }

    #[test]
    fn test_severity_classification() {
        let config = DetectorConfig::default();
        let detector = AnomalyDetector::new(config);

        assert_eq!(detector.classify_hardware_severity(6.0), Severity::Critical);
        assert_eq!(detector.classify_hardware_severity(4.5), Severity::High);
        assert_eq!(detector.classify_hardware_severity(3.5), Severity::Medium);
        assert_eq!(detector.classify_hardware_severity(2.0), Severity::Low);
    }

    #[test]
    fn test_confidence_calculation() {
        let config = DetectorConfig::default();
        let detector = AnomalyDetector::new(config);

        assert_eq!(detector.calculate_confidence(5.0), 0.5);
        assert_eq!(detector.calculate_confidence(10.0), 1.0);
        assert_eq!(detector.calculate_confidence(1.0), 0.1);
    }
}
