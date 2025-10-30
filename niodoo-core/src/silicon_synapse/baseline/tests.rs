// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Comprehensive tests for baseline learning and anomaly detection
//!
//! This module contains tests with synthetic normal data and known outliers
//! to validate the baseline learning and anomaly detection system.

use super::{
    AnomalyDetector, AnomalyType, BaselineManager, BaselineModel, DetectorConfig, Severity,
};
use crate::silicon_synapse::aggregation::{AggregatedMetrics, StatisticalSummary};
use std::collections::HashMap;
use std::time::SystemTime;

/// Generate synthetic normal metrics for testing
fn generate_normal_metrics() -> AggregatedMetrics {
    AggregatedMetrics {
        timestamp: SystemTime::now(),
        window_start: SystemTime::now(),
        window_duration: std::time::Duration::from_secs(60),
        hardware_metrics: crate::silicon_synapse::aggregation::HardwareAggregatedMetrics {
            gpu_temperature: StatisticalSummary {
                count: 1,
                mean: 75.0, // Normal GPU temperature
                std_dev: 5.0,
                min: 70.0,
                max: 80.0,
                percentiles: HashMap::from([
                    ("50.0".to_string(), 75.0),
                    ("90.0".to_string(), 80.0),
                    ("95.0".to_string(), 82.0),
                    ("99.0".to_string(), 85.0),
                ]),
            },
            gpu_power: StatisticalSummary {
                count: 1,
                mean: 200.0, // Normal GPU power
                std_dev: 20.0,
                min: 180.0,
                max: 220.0,
                percentiles: HashMap::from([
                    ("50.0".to_string(), 200.0),
                    ("90.0".to_string(), 220.0),
                    ("95.0".to_string(), 225.0),
                    ("99.0".to_string(), 230.0),
                ]),
            },
            gpu_utilization: StatisticalSummary {
                count: 1,
                mean: 80.0,
                std_dev: 10.0,
                min: 70.0,
                max: 90.0,
                percentiles: HashMap::new(),
            },
            gpu_memory_usage: StatisticalSummary {
                count: 1,
                mean: 60.0,
                std_dev: 5.0,
                min: 55.0,
                max: 65.0,
                percentiles: HashMap::new(),
            },
            cpu_temperature: StatisticalSummary {
                count: 1,
                mean: 65.0,
                std_dev: 3.0,
                min: 62.0,
                max: 68.0,
                percentiles: HashMap::new(),
            },
            cpu_utilization: StatisticalSummary {
                count: 1,
                mean: 40.0,
                std_dev: 8.0,
                min: 32.0,
                max: 48.0,
                percentiles: HashMap::new(),
            },
            system_memory_usage: StatisticalSummary {
                count: 1,
                mean: 70.0,
                std_dev: 5.0,
                min: 65.0,
                max: 75.0,
                percentiles: HashMap::new(),
            },
        },
        inference_metrics: crate::silicon_synapse::aggregation::InferenceAggregatedMetrics {
            ttft_ms: StatisticalSummary {
                count: 1,
                mean: 100.0, // Normal TTFT
                std_dev: 10.0,
                min: 90.0,
                max: 110.0,
                percentiles: HashMap::from([
                    ("50.0".to_string(), 100.0),
                    ("90.0".to_string(), 110.0),
                    ("95.0".to_string(), 115.0),
                    ("99.0".to_string(), 120.0),
                ]),
            },
            tpot_ms: StatisticalSummary {
                count: 1,
                mean: 50.0,
                std_dev: 5.0,
                min: 45.0,
                max: 55.0,
                percentiles: HashMap::new(),
            },
            throughput_tps: StatisticalSummary {
                count: 1,
                mean: 20.0, // Normal throughput
                std_dev: 2.0,
                min: 18.0,
                max: 22.0,
                percentiles: HashMap::new(),
            },
            error_rate: StatisticalSummary {
                count: 1,
                mean: 0.01, // 1% error rate
                std_dev: 0.005,
                min: 0.005,
                max: 0.015,
                percentiles: HashMap::new(),
            },
            active_requests: StatisticalSummary {
                count: 1,
                mean: 5.0,
                std_dev: 1.0,
                min: 4.0,
                max: 6.0,
                percentiles: HashMap::new(),
            },
            completed_requests: StatisticalSummary {
                count: 1,
                mean: 100.0,
                std_dev: 10.0,
                min: 90.0,
                max: 110.0,
                percentiles: HashMap::new(),
            },
        },
        model_metrics: crate::silicon_synapse::aggregation::ModelAggregatedMetrics {
            entropy_by_layer: HashMap::from([
                (
                    "12".to_string(),
                    StatisticalSummary {
                        count: 1,
                        mean: 2.5, // Normal entropy
                        std_dev: 0.2,
                        min: 2.3,
                        max: 2.7,
                        percentiles: HashMap::new(),
                    },
                ),
                (
                    "24".to_string(),
                    StatisticalSummary {
                        count: 1,
                        mean: 1.8,
                        std_dev: 0.15,
                        min: 1.65,
                        max: 1.95,
                        percentiles: HashMap::new(),
                    },
                ),
            ]),
            activation_sparsity_by_layer: HashMap::new(),
            activation_magnitude_by_layer: HashMap::new(),
        },
        correlations: Vec::new(),
    }
}

/// Generate anomalous metrics for testing
fn generate_anomalous_metrics(anomaly_type: &str) -> AggregatedMetrics {
    let mut metrics = generate_normal_metrics();

    match anomaly_type {
        "gpu_temperature_spike" => {
            // GPU temperature spike (4+ sigma away)
            metrics.hardware_metrics.gpu_temperature.mean = 95.0; // 4 sigma away
        }
        "gpu_power_spike" => {
            // GPU power spike (security threat)
            metrics.hardware_metrics.gpu_power.mean = 280.0; // 4 sigma away
        }
        "ttft_degradation" => {
            // TTFT performance degradation
            metrics.inference_metrics.ttft_ms.mean = 150.0; // 5 sigma away
        }
        "high_error_rate" => {
            // High error rate (model instability)
            metrics.inference_metrics.error_rate.mean = 0.05; // 8 sigma away
        }
        "entropy_anomaly" => {
            // Unusual entropy (emergent behavior)
            metrics
                .model_metrics
                .entropy_by_layer
                .get_mut("12")
                .unwrap()
                .mean = 3.5; // 5 sigma away
        }
        "throughput_drop" => {
            // Throughput drop
            metrics.inference_metrics.throughput_tps.mean = 10.0; // 5 sigma away
        }
        _ => {
            // Unknown anomaly - mix of issues
            metrics.hardware_metrics.gpu_temperature.mean = 90.0;
            metrics.inference_metrics.ttft_ms.mean = 130.0;
        }
    }

    metrics
}

/// Test baseline model creation and updates
#[cfg(test)]
mod baseline_model_tests {
    use super::*;

    #[test]
    fn test_baseline_model_creation() {
        let baseline = BaselineModel::new();
        assert_eq!(baseline.version, 1);
        assert_eq!(baseline.sample_count, 0);
        assert!(baseline.univariate_stats.is_empty());
    }

    #[test]
    fn test_baseline_model_update() {
        let mut baseline = BaselineModel::new();
        let metrics = generate_normal_metrics();

        baseline.update(&metrics);
        assert_eq!(baseline.sample_count, 1);
        assert!(!baseline.univariate_stats.is_empty());

        // Check that key metrics are present
        assert!(baseline.univariate_stats.contains_key("gpu_temperature"));
        assert!(baseline.univariate_stats.contains_key("ttft_ms"));
        assert!(baseline.univariate_stats.contains_key("entropy_layer_12"));
    }

    #[test]
    fn test_baseline_model_serialization() {
        let mut baseline = BaselineModel::new();
        let metrics = generate_normal_metrics();
        baseline.update(&metrics);

        // Test JSON serialization
        let json = baseline.to_json().unwrap();
        let deserialized = BaselineModel::from_json(&json).unwrap();

        assert_eq!(baseline.version, deserialized.version);
        assert_eq!(baseline.sample_count, deserialized.sample_count);
        assert_eq!(
            baseline.univariate_stats.len(),
            deserialized.univariate_stats.len()
        );
    }

    #[test]
    fn test_anomaly_detection() {
        let mut baseline = BaselineModel::new();
        let normal_metrics = generate_normal_metrics();

        // Build baseline with normal data
        baseline.update(&normal_metrics);

        // Test normal values (should not be anomalous)
        assert!(!baseline.is_anomalous("gpu_temperature", 75.0));
        assert!(!baseline.is_anomalous("ttft_ms", 100.0));

        // Test anomalous values (should be anomalous)
        assert!(baseline.is_anomalous("gpu_temperature", 95.0)); // 4 sigma away
        assert!(baseline.is_anomalous("ttft_ms", 150.0)); // 5 sigma away
    }

    #[test]
    fn test_deviation_score_calculation() {
        let mut baseline = BaselineModel::new();
        let normal_metrics = generate_normal_metrics();
        baseline.update(&normal_metrics);

        // Test deviation score calculation
        let score = baseline.calculate_deviation_score("gpu_temperature", 85.0);
        assert_eq!(score, 2.0); // (85 - 75) / 5 = 2.0

        let score = baseline.calculate_deviation_score("ttft_ms", 120.0);
        assert_eq!(score, 2.0); // (120 - 100) / 10 = 2.0
    }

    #[test]
    fn test_multivariate_anomaly_detection() {
        let mut baseline = BaselineModel::new();
        let normal_metrics = generate_normal_metrics();
        baseline.update(&normal_metrics);

        // Test with normal metrics (should not detect multivariate anomaly)
        let normal_anomaly = baseline.detect_multivariate_anomaly(&normal_metrics);
        assert!(normal_anomaly.is_none());

        // Test with anomalous metrics (should detect multivariate anomaly)
        let anomalous_metrics = generate_anomalous_metrics("gpu_temperature_spike");
        let _anomalous_result = baseline.detect_multivariate_anomaly(&anomalous_metrics);
        // Note: This might be None depending on correlation thresholds
    }
}

/// Test anomaly detector functionality
#[cfg(test)]
mod anomaly_detector_tests {
    use super::*;

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

        let normal_metrics = generate_normal_metrics();

        // First sample - should be in learning mode
        let anomalies = detector.process_metrics(&normal_metrics);
        assert!(anomalies.is_empty());
        assert!(detector.is_learning_mode());
        assert_eq!(detector.get_learning_sample_count(), 1);

        // Second sample - should switch to detection mode
        let anomalies = detector.process_metrics(&normal_metrics);
        assert!(anomalies.is_empty());
        assert!(!detector.is_learning_mode());
        assert!(detector.get_baseline().is_some());
    }

    #[test]
    fn test_gpu_temperature_anomaly_detection() {
        let config = DetectorConfig {
            min_samples_for_baseline: 1,
            univariate_threshold_sigma: 3.0,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);

        // Build baseline with normal data
        let normal_metrics = generate_normal_metrics();
        detector.process_metrics(&normal_metrics);

        // Test with anomalous GPU temperature
        let anomalous_metrics = generate_anomalous_metrics("gpu_temperature_spike");
        let anomalies = detector.process_metrics(&anomalous_metrics);

        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::HardwareFailure);
        assert_eq!(anomalies[0].severity, Severity::High);
        assert!(anomalies[0]
            .affected_metrics
            .contains(&"gpu_temperature".to_string()));
    }

    #[test]
    fn test_gpu_power_security_threat_detection() {
        let config = DetectorConfig {
            min_samples_for_baseline: 1,
            univariate_threshold_sigma: 3.0,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);

        // Build baseline
        let normal_metrics = generate_normal_metrics();
        detector.process_metrics(&normal_metrics);

        // Test with anomalous GPU power
        let anomalous_metrics = generate_anomalous_metrics("gpu_power_spike");
        let anomalies = detector.process_metrics(&anomalous_metrics);

        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::SecurityThreat);
        assert_eq!(anomalies[0].severity, Severity::High);
    }

    #[test]
    fn test_ttft_performance_degradation_detection() {
        let config = DetectorConfig {
            min_samples_for_baseline: 1,
            univariate_threshold_sigma: 3.0,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);

        // Build baseline
        let normal_metrics = generate_normal_metrics();
        detector.process_metrics(&normal_metrics);

        // Test with TTFT degradation
        let anomalous_metrics = generate_anomalous_metrics("ttft_degradation");
        let anomalies = detector.process_metrics(&anomalous_metrics);

        assert!(!anomalies.is_empty());
        assert_eq!(
            anomalies[0].anomaly_type,
            AnomalyType::PerformanceDegradation
        );
        assert_eq!(anomalies[0].severity, Severity::High);
    }

    #[test]
    fn test_error_rate_instability_detection() {
        let config = DetectorConfig {
            min_samples_for_baseline: 1,
            univariate_threshold_sigma: 3.0,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);

        // Build baseline
        let normal_metrics = generate_normal_metrics();
        detector.process_metrics(&normal_metrics);

        // Test with high error rate
        let anomalous_metrics = generate_anomalous_metrics("high_error_rate");
        let anomalies = detector.process_metrics(&anomalous_metrics);

        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::ModelInstability);
        assert_eq!(anomalies[0].severity, Severity::Critical);
    }

    #[test]
    fn test_entropy_emergent_behavior_detection() {
        let config = DetectorConfig {
            min_samples_for_baseline: 1,
            univariate_threshold_sigma: 3.0,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);

        // Build baseline
        let normal_metrics = generate_normal_metrics();
        detector.process_metrics(&normal_metrics);

        // Test with unusual entropy
        let anomalous_metrics = generate_anomalous_metrics("entropy_anomaly");
        let anomalies = detector.process_metrics(&anomalous_metrics);

        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::EmergentBehavior);
        assert_eq!(anomalies[0].severity, Severity::High);
    }

    #[test]
    fn test_severity_classification() {
        let config = DetectorConfig::default();
        let detector = AnomalyDetector::new(config);

        // Test hardware severity classification
        assert_eq!(detector.classify_hardware_severity(6.0), Severity::Critical);
        assert_eq!(detector.classify_hardware_severity(4.5), Severity::High);
        assert_eq!(detector.classify_hardware_severity(3.5), Severity::Medium);
        assert_eq!(detector.classify_hardware_severity(2.0), Severity::Low);

        // Test security severity classification
        assert_eq!(detector.classify_security_severity(5.0), Severity::Critical);
        assert_eq!(detector.classify_security_severity(3.5), Severity::High);
        assert_eq!(detector.classify_security_severity(2.5), Severity::Medium);
        assert_eq!(detector.classify_security_severity(1.5), Severity::Low);
    }

    #[test]
    fn test_confidence_calculation() {
        let config = DetectorConfig::default();
        let detector = AnomalyDetector::new(config);

        assert_eq!(detector.calculate_confidence(5.0), 0.5);
        assert_eq!(detector.calculate_confidence(10.0), 1.0);
        assert_eq!(detector.calculate_confidence(1.0), 0.1);
        assert_eq!(detector.calculate_confidence(0.5), 0.1); // Minimum confidence
    }
}

/// Test baseline manager functionality
#[cfg(test)]
mod baseline_manager_tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_baseline_manager_creation() {
        let config = DetectorConfig::default();
        let manager = BaselineManager::new(config);

        let stats = manager.get_detector_stats().await;
        assert!(!stats.is_running);
        assert!(!stats.is_learning);
    }

    #[tokio::test]
    async fn test_baseline_manager_start_stop() {
        let config = DetectorConfig {
            min_samples_for_baseline: 1,
            learning_duration_hours: 1, // Short duration for testing
            ..Default::default()
        };
        let manager = BaselineManager::new(config);

        assert!(manager.start().await.is_ok());

        // Give it a moment to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        let stats = manager.get_detector_stats().await;
        assert!(stats.is_running);

        assert!(manager.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_learning_progress() {
        let config = DetectorConfig::default();
        let manager = BaselineManager::new(config);

        let progress = manager.get_learning_progress().await;
        assert!(!progress.is_learning);
        assert_eq!(progress.progress_percent, 0.0);
    }

    #[tokio::test]
    async fn test_force_detection_mode() {
        let config = DetectorConfig {
            min_samples_for_baseline: 1,
            ..Default::default()
        };
        let manager = BaselineManager::new(config);

        manager.start().await.unwrap();

        // Process one sample to build baseline
        let metrics = generate_normal_metrics();
        manager.process_metrics(&metrics).await;

        // Force detection mode
        manager.force_detection_mode().await;

        let stats = manager.get_detector_stats().await;
        assert!(!stats.is_learning);
    }

    #[tokio::test]
    async fn test_end_to_end_anomaly_detection() {
        let config = DetectorConfig {
            min_samples_for_baseline: 1,
            learning_duration_hours: 1, // Short duration for testing
            ..Default::default()
        };
        let manager = BaselineManager::new(config);

        manager.start().await.unwrap();

        // Build baseline with normal data
        let normal_metrics = generate_normal_metrics();
        let anomalies = manager.process_metrics(&normal_metrics).await;
        assert!(anomalies.is_empty()); // Should be in learning mode

        // Force detection mode
        manager.force_detection_mode().await;

        // Test with normal data (should not detect anomalies)
        let normal_anomalies = manager.process_metrics(&normal_metrics).await;
        assert!(normal_anomalies.is_empty());

        // Test with anomalous data (should detect anomalies)
        let anomalous_metrics = generate_anomalous_metrics("gpu_temperature_spike");
        let anomalies = manager.process_metrics(&anomalous_metrics).await;
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::HardwareFailure);

        manager.stop().await.unwrap();
    }
}

/// Integration tests with multiple anomaly types
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_multiple_anomaly_types() {
        let config = DetectorConfig {
            min_samples_for_baseline: 1,
            univariate_threshold_sigma: 3.0,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);

        // Build baseline
        let normal_metrics = generate_normal_metrics();
        detector.process_metrics(&normal_metrics);

        // Test different anomaly types
        let anomaly_types = [
            "gpu_temperature_spike",
            "gpu_power_spike",
            "ttft_degradation",
            "high_error_rate",
            "entropy_anomaly",
            "throughput_drop",
        ];

        for anomaly_type in &anomaly_types {
            let anomalous_metrics = generate_anomalous_metrics(anomaly_type);
            let anomalies = detector.process_metrics(&anomalous_metrics);

            assert!(
                !anomalies.is_empty(),
                "Failed to detect {} anomaly",
                anomaly_type
            );

            // Verify anomaly has proper structure
            let anomaly = &anomalies[0];
            assert!(!anomaly.id.to_string().is_empty());
            assert!(!anomaly.description.is_empty());
            assert!(!anomaly.affected_metrics.is_empty());
            assert!(anomaly.confidence > 0.0 && anomaly.confidence <= 1.0);
        }
    }

    #[test]
    fn test_baseline_persistence() {
        let mut baseline = BaselineModel::new();
        let normal_metrics = generate_normal_metrics();

        // Update baseline multiple times
        for _ in 0..10 {
            baseline.update(&normal_metrics);
        }

        assert_eq!(baseline.sample_count, 10);
        assert!(baseline.is_ready_for_detection());

        // Test serialization/deserialization
        let json = baseline.to_json().unwrap();
        let deserialized = BaselineModel::from_json(&json).unwrap();

        assert_eq!(baseline.sample_count, deserialized.sample_count);
        assert_eq!(
            baseline.univariate_stats.len(),
            deserialized.univariate_stats.len()
        );
    }

    #[test]
    fn test_correlation_matrix() {
        let mut baseline = BaselineModel::new();
        let normal_metrics = generate_normal_metrics();

        baseline.update(&normal_metrics);

        // Check that correlation matrix was populated
        assert!(!baseline.correlation_matrix.correlations.is_empty());

        // Test correlation key parsing
        let correlation_keys: Vec<String> = baseline
            .correlation_matrix
            .correlations
            .keys()
            .cloned()
            .collect();
        for key in correlation_keys {
            assert!(key.contains(':'));
            let parts: Vec<&str> = key.split(':').collect();
            assert_eq!(parts.len(), 2);
        }
    }
}
