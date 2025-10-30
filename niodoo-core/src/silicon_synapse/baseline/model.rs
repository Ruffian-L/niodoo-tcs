// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Baseline model for anomaly detection
//!
//! This module implements the baseline learning system that captures normal operational
//! patterns and provides statistical models for anomaly detection.

use crate::silicon_synapse::aggregation::AggregatedMetrics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// Baseline model for normal operational behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineModel {
    /// Model version for compatibility tracking
    pub version: u32,
    /// When the baseline was created
    pub created_at: SystemTime,
    /// When the baseline was last updated
    pub updated_at: SystemTime,
    /// Total number of samples used to build the baseline
    pub sample_count: usize,
    /// Per-metric statistical baselines
    pub univariate_stats: HashMap<String, MetricStats>,
    /// Correlation matrix for multivariate relationships
    pub correlation_matrix: CorrelationMatrix,
}

/// Statistical baseline for a specific metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    /// Metric name
    pub metric_name: String,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub stddev: f64,
    /// Minimum observed value
    pub min: f64,
    /// Maximum observed value
    pub max: f64,
    /// Percentile values
    pub percentiles: Percentiles,
    /// Number of samples used for this metric
    pub sample_count: usize,
    /// Last update timestamp
    pub last_updated: SystemTime,
}

/// Percentile values for baseline comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Percentiles {
    /// 50th percentile (median)
    pub p50: f64,
    /// 90th percentile
    pub p90: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
}

/// Correlation matrix for multivariate anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    /// Map of metric pairs to their correlation coefficients
    pub correlations: HashMap<String, f64>,
    /// Confidence threshold for significant correlations
    pub confidence_threshold: f64,
    /// Minimum sample count required for correlation calculation
    pub min_samples: usize,
}

impl BaselineModel {
    /// Create a new baseline model
    pub fn new() -> Self {
        Self {
            version: 1,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            sample_count: 0,
            univariate_stats: HashMap::new(),
            correlation_matrix: CorrelationMatrix::new(),
        }
    }

    /// Update baseline with new aggregated metrics
    pub fn update(&mut self, metrics: &AggregatedMetrics) {
        self.sample_count += 1;
        self.updated_at = SystemTime::now();

        // Update hardware metrics baselines
        self.update_hardware_baseline(&metrics.hardware_metrics);

        // Update inference metrics baselines
        self.update_inference_baseline(&metrics.inference_metrics);

        // Update model metrics baselines
        self.update_model_baseline(&metrics.model_metrics);

        // Update correlation matrix
        self.update_correlation_matrix(metrics);
    }

    /// Update hardware metrics baseline
    fn update_hardware_baseline(
        &mut self,
        metrics: &crate::silicon_synapse::aggregation::HardwareAggregatedMetrics,
    ) {
        self.update_metric_stats("gpu_temperature", &metrics.gpu_temperature);
        self.update_metric_stats("gpu_power", &metrics.gpu_power);
        self.update_metric_stats("gpu_utilization", &metrics.gpu_utilization);
        self.update_metric_stats("gpu_memory_usage", &metrics.gpu_memory_usage);
        self.update_metric_stats("cpu_temperature", &metrics.cpu_temperature);
        self.update_metric_stats("cpu_utilization", &metrics.cpu_utilization);
        self.update_metric_stats("system_memory_usage", &metrics.system_memory_usage);
    }

    /// Update inference metrics baseline
    fn update_inference_baseline(
        &mut self,
        metrics: &crate::silicon_synapse::aggregation::InferenceAggregatedMetrics,
    ) {
        self.update_metric_stats("ttft_ms", &metrics.ttft_ms);
        self.update_metric_stats("tpot_ms", &metrics.tpot_ms);
        self.update_metric_stats("throughput_tps", &metrics.throughput_tps);
        self.update_metric_stats("error_rate", &metrics.error_rate);
        self.update_metric_stats("active_requests", &metrics.active_requests);
        self.update_metric_stats("completed_requests", &metrics.completed_requests);
    }

    /// Update model metrics baseline
    fn update_model_baseline(
        &mut self,
        metrics: &crate::silicon_synapse::aggregation::ModelAggregatedMetrics,
    ) {
        for (layer, entropy) in &metrics.entropy_by_layer {
            let metric_name = format!("entropy_layer_{}", layer);
            self.update_metric_stats(&metric_name, entropy);
        }

        for (layer, sparsity) in &metrics.activation_sparsity_by_layer {
            let metric_name = format!("sparsity_layer_{}", layer);
            self.update_metric_stats(&metric_name, sparsity);
        }

        for (layer, magnitude) in &metrics.activation_magnitude_by_layer {
            let metric_name = format!("magnitude_layer_{}", layer);
            self.update_metric_stats(&metric_name, magnitude);
        }
    }

    /// Update a single metric's statistical baseline
    fn update_metric_stats(
        &mut self,
        metric_name: &str,
        summary: &crate::silicon_synapse::aggregation::StatisticalSummary,
    ) {
        if summary.count == 0 {
            return;
        }

        let stats = self
            .univariate_stats
            .entry(metric_name.to_string())
            .or_insert_with(|| MetricStats {
                metric_name: metric_name.to_string(),
                mean: 0.0,
                stddev: 0.0,
                min: f64::MAX,
                max: f64::MIN,
                percentiles: Percentiles {
                    p50: 0.0,
                    p90: 0.0,
                    p95: 0.0,
                    p99: 0.0,
                },
                sample_count: 0,
                last_updated: SystemTime::now(),
            });

        // Update statistics using exponential moving average
        let alpha = 0.1; // Learning rate
        stats.mean = alpha * summary.mean + (1.0 - alpha) * stats.mean;
        stats.stddev = alpha * summary.std_dev + (1.0 - alpha) * stats.stddev;

        // Update min/max
        stats.min = stats.min.min(summary.min);
        stats.max = stats.max.max(summary.max);

        // Update percentiles
        stats.percentiles.p50 = summary
            .percentiles
            .get("50.0")
            .copied()
            .unwrap_or(summary.mean);
        stats.percentiles.p90 = summary
            .percentiles
            .get("90.0")
            .copied()
            .unwrap_or(summary.mean);
        stats.percentiles.p95 = summary
            .percentiles
            .get("95.0")
            .copied()
            .unwrap_or(summary.mean);
        stats.percentiles.p99 = summary
            .percentiles
            .get("99.0")
            .copied()
            .unwrap_or(summary.mean);

        stats.sample_count += summary.count;
        stats.last_updated = SystemTime::now();
    }

    /// Update correlation matrix with new metrics
    fn update_correlation_matrix(&mut self, metrics: &AggregatedMetrics) {
        // Extract all metric values for correlation calculation
        let mut metric_values: HashMap<String, f64> = HashMap::new();

        // Hardware metrics
        metric_values.insert(
            "gpu_temperature".to_string(),
            metrics.hardware_metrics.gpu_temperature.mean,
        );
        metric_values.insert(
            "gpu_power".to_string(),
            metrics.hardware_metrics.gpu_power.mean,
        );
        metric_values.insert(
            "gpu_utilization".to_string(),
            metrics.hardware_metrics.gpu_utilization.mean,
        );
        metric_values.insert(
            "cpu_temperature".to_string(),
            metrics.hardware_metrics.cpu_temperature.mean,
        );
        metric_values.insert(
            "cpu_utilization".to_string(),
            metrics.hardware_metrics.cpu_utilization.mean,
        );

        // Inference metrics
        metric_values.insert(
            "ttft_ms".to_string(),
            metrics.inference_metrics.ttft_ms.mean,
        );
        metric_values.insert(
            "tpot_ms".to_string(),
            metrics.inference_metrics.tpot_ms.mean,
        );
        metric_values.insert(
            "throughput_tps".to_string(),
            metrics.inference_metrics.throughput_tps.mean,
        );
        metric_values.insert(
            "error_rate".to_string(),
            metrics.inference_metrics.error_rate.mean,
        );

        // Calculate correlations between all pairs
        let metric_names: Vec<String> = metric_values.keys().cloned().collect();
        for i in 0..metric_names.len() {
            for j in (i + 1)..metric_names.len() {
                let metric1 = &metric_names[i];
                let metric2 = &metric_names[j];

                if let (Some(val1), Some(val2)) =
                    (metric_values.get(metric1), metric_values.get(metric2))
                {
                    let correlation_key = format!("{}:{}", metric1, metric2);
                    let correlation = self.calculate_simple_correlation(*val1, *val2);

                    self.correlation_matrix
                        .correlations
                        .insert(correlation_key, correlation);
                }
            }
        }
    }

    /// Calculate simple correlation between two values
    fn calculate_simple_correlation(&self, val1: f64, val2: f64) -> f64 {
        // Simplified correlation calculation
        // In a real implementation, this would use historical data points
        if val1 == 0.0 && val2 == 0.0 {
            return 1.0;
        }

        let max_val = val1.max(val2);
        if max_val == 0.0 {
            return 0.0;
        }

        // Simple correlation approximation based on relative values
        1.0 - ((val1 - val2).abs() / max_val)
    }

    /// Check if a metric value is anomalous using 3-sigma rule
    pub fn is_anomalous(&self, metric: &str, value: f64) -> bool {
        if let Some(stats) = self.univariate_stats.get(metric) {
            if stats.stddev == 0.0 {
                return false;
            }

            let deviation = (value - stats.mean).abs();
            deviation > 3.0 * stats.stddev
        } else {
            false
        }
    }

    /// Calculate deviation score (z-score) for a metric value
    pub fn calculate_deviation_score(&self, metric: &str, value: f64) -> f64 {
        if let Some(stats) = self.univariate_stats.get(metric) {
            if stats.stddev == 0.0 {
                return 0.0;
            }

            (value - stats.mean) / stats.stddev
        } else {
            0.0
        }
    }

    /// Detect multivariate anomalies using correlation violations
    pub fn detect_multivariate_anomaly(
        &self,
        metrics: &AggregatedMetrics,
    ) -> Option<MultivariateAnomaly> {
        let mut violations = Vec::new();

        // Check for correlation violations
        for (correlation_key, expected_correlation) in &self.correlation_matrix.correlations {
            if let Some((metric1, metric2)) = self.parse_correlation_key(correlation_key) {
                if let (Some(val1), Some(val2)) = (
                    self.get_metric_value(metrics, &metric1),
                    self.get_metric_value(metrics, &metric2),
                ) {
                    let current_correlation = self.calculate_simple_correlation(val1, val2);
                    let correlation_diff = (current_correlation - expected_correlation).abs();

                    if correlation_diff > 0.3 {
                        // Threshold for correlation violation
                        violations.push(CorrelationViolation {
                            metric1: metric1.clone(),
                            metric2: metric2.clone(),
                            expected_correlation: *expected_correlation,
                            actual_correlation: current_correlation,
                            deviation: correlation_diff,
                        });
                    }
                }
            }
        }

        if violations.is_empty() {
            None
        } else {
            Some(MultivariateAnomaly {
                violations: violations.clone(),
                severity: self.calculate_multivariate_severity(&violations),
            })
        }
    }

    /// Parse correlation key into metric names
    fn parse_correlation_key(&self, key: &str) -> Option<(String, String)> {
        if let Some(pos) = key.find(':') {
            let metric1 = key[..pos].to_string();
            let metric2 = key[pos + 1..].to_string();
            Some((metric1, metric2))
        } else {
            None
        }
    }

    /// Get metric value from aggregated metrics
    fn get_metric_value(&self, metrics: &AggregatedMetrics, metric_name: &str) -> Option<f64> {
        match metric_name {
            "gpu_temperature" => Some(metrics.hardware_metrics.gpu_temperature.mean),
            "gpu_power" => Some(metrics.hardware_metrics.gpu_power.mean),
            "gpu_utilization" => Some(metrics.hardware_metrics.gpu_utilization.mean),
            "cpu_temperature" => Some(metrics.hardware_metrics.cpu_temperature.mean),
            "cpu_utilization" => Some(metrics.hardware_metrics.cpu_utilization.mean),
            "ttft_ms" => Some(metrics.inference_metrics.ttft_ms.mean),
            "tpot_ms" => Some(metrics.inference_metrics.tpot_ms.mean),
            "throughput_tps" => Some(metrics.inference_metrics.throughput_tps.mean),
            "error_rate" => Some(metrics.inference_metrics.error_rate.mean),
            _ => None,
        }
    }

    /// Calculate severity for multivariate anomalies
    fn calculate_multivariate_severity(
        &self,
        violations: &[CorrelationViolation],
    ) -> MultivariateSeverity {
        let max_deviation = violations.iter().map(|v| v.deviation).fold(0.0, f64::max);

        match max_deviation {
            d if d > 0.8 => MultivariateSeverity::Critical,
            d if d > 0.6 => MultivariateSeverity::High,
            d if d > 0.4 => MultivariateSeverity::Medium,
            _ => MultivariateSeverity::Low,
        }
    }

    /// Serialize baseline model to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize baseline model from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Check if baseline has sufficient data for anomaly detection
    pub fn is_ready_for_detection(&self) -> bool {
        self.sample_count >= 1000 && !self.univariate_stats.is_empty()
    }
}

impl CorrelationMatrix {
    /// Create a new correlation matrix
    pub fn new() -> Self {
        Self {
            correlations: HashMap::new(),
            confidence_threshold: 0.7,
            min_samples: 100,
        }
    }
}

/// Multivariate anomaly detection result
#[derive(Debug, Clone)]
pub struct MultivariateAnomaly {
    /// Correlation violations detected
    pub violations: Vec<CorrelationViolation>,
    /// Overall severity of the multivariate anomaly
    pub severity: MultivariateSeverity,
}

/// Correlation violation between two metrics
#[derive(Debug, Clone)]
pub struct CorrelationViolation {
    /// First metric name
    pub metric1: String,
    /// Second metric name
    pub metric2: String,
    /// Expected correlation coefficient
    pub expected_correlation: f64,
    /// Actual correlation coefficient
    pub actual_correlation: f64,
    /// Deviation magnitude
    pub deviation: f64,
}

/// Severity levels for multivariate anomalies
#[derive(Debug, Clone, PartialEq)]
pub enum MultivariateSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for BaselineModel {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CorrelationMatrix {
    fn default() -> Self {
        Self::new()
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

    #[test]
    fn test_baseline_model_creation() {
        let baseline = BaselineModel::new();
        assert_eq!(baseline.version, 1);
        assert_eq!(baseline.sample_count, 0);
        assert!(baseline.univariate_stats.is_empty());
    }

    #[test]
    fn test_baseline_update() {
        let mut baseline = BaselineModel::new();
        let metrics = create_test_metrics();

        baseline.update(&metrics);
        assert_eq!(baseline.sample_count, 1);
        assert!(!baseline.univariate_stats.is_empty());
    }

    #[test]
    fn test_anomaly_detection() {
        let mut baseline = BaselineModel::new();
        let metrics = create_test_metrics();

        // Build baseline
        baseline.update(&metrics);

        // Test normal value (should not be anomalous)
        assert!(!baseline.is_anomalous("gpu_temperature", 75.0));

        // Test anomalous value (3+ standard deviations away)
        assert!(baseline.is_anomalous("gpu_temperature", 95.0));
    }

    #[test]
    fn test_deviation_score() {
        let mut baseline = BaselineModel::new();
        let metrics = create_test_metrics();

        baseline.update(&metrics);

        // Test deviation score calculation
        let score = baseline.calculate_deviation_score("gpu_temperature", 85.0);
        assert_eq!(score, 2.0); // (85 - 75) / 5 = 2.0
    }

    #[test]
    fn test_serialization() {
        let mut baseline = BaselineModel::new();
        let metrics = create_test_metrics();
        baseline.update(&metrics);

        // Test JSON serialization
        let json = baseline.to_json().unwrap();
        let deserialized = BaselineModel::from_json(&json).unwrap();

        assert_eq!(baseline.version, deserialized.version);
        assert_eq!(baseline.sample_count, deserialized.sample_count);
    }

    #[test]
    fn test_readiness_check() {
        let mut baseline = BaselineModel::new();
        assert!(!baseline.is_ready_for_detection());

        // Simulate enough samples
        baseline.sample_count = 1000;
        assert!(baseline.is_ready_for_detection());
    }
}
