//! Anomaly detection system for Silicon Synapse

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::config::AnomalyDetectionConfig;
use crate::telemetry::{TelemetrySender, TelemetryEvent, AnomalyType, Severity};

/// Anomaly detection system
pub struct AnomalyDetector {
    config: AnomalyDetectionConfig,
    telemetry_sender: TelemetrySender,
    anomaly_counters: Arc<RwLock<HashMap<AnomalyType, u64>>>,
    is_running: Arc<RwLock<bool>>,
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct Anomaly {
    pub timestamp: u64,
    pub anomaly_type: AnomalyType,
    pub severity: Severity,
    pub affected_metrics: Vec<String>,
    pub deviation_magnitude: f64,
    pub context: HashMap<String, String>,
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    pub async fn new(
        config: AnomalyDetectionConfig,
        telemetry_sender: TelemetrySender,
    ) -> Result<Self> {
        Ok(Self {
            config,
            telemetry_sender,
            anomaly_counters: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(RwLock::new(false)),
        })
    }
    
    /// Start anomaly detection
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting anomaly detection");
        
        {
            let mut is_running = self.is_running.write().await;
            *is_running = true;
        }
        
        info!("Anomaly detection started successfully");
        Ok(())
    }
    
    /// Shutdown anomaly detection
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down anomaly detection");
        
        {
            let mut is_running = self.is_running.write().await;
            *is_running = false;
        }
        
        info!("Anomaly detection shutdown complete");
        Ok(())
    }
    
    /// Process telemetry events for anomaly detection
    pub async fn process_event(&self, event: TelemetryEvent) -> Result<()> {
        match event {
            TelemetryEvent::HardwareMetrics { timestamp, device_id, metrics } => {
                self.process_hardware_metrics(timestamp, device_id, metrics).await?;
            }
            TelemetryEvent::InferenceComplete { timestamp, request_id, total_tokens, total_duration_ms, ttft_ms, tpot_ms } => {
                self.process_inference_complete(timestamp, request_id, total_tokens, total_duration_ms, ttft_ms, tpot_ms).await?;
            }
            TelemetryEvent::ModelEntropy { timestamp, model_name, entropy, token_index } => {
                self.process_model_entropy(timestamp, model_name, entropy, token_index).await?;
            }
            _ => {
                // Other event types not processed for anomaly detection
            }
        }
        
        Ok(())
    }
    
    /// Process hardware metrics for anomalies
    async fn process_hardware_metrics(
        &self,
        timestamp: u64,
        device_id: String,
        metrics: crate::telemetry::HardwareMetrics,
    ) -> Result<()> {
        let mut anomaly_metrics = HashMap::new();
        
        // Check GPU temperature
        if let Some(temp) = metrics.gpu_temperature {
            anomaly_metrics.insert(format!("gpu_temperature_{}", device_id), temp as f64);
        }
        
        // Check GPU power
        if let Some(power) = metrics.gpu_power {
            anomaly_metrics.insert(format!("gpu_power_{}", device_id), power as f64);
        }
        
        // Check GPU utilization
        if let Some(util) = metrics.gpu_utilization {
            anomaly_metrics.insert(format!("gpu_utilization_{}", device_id), util as f64);
        }
        
        // Check CPU utilization
        if let Some(cpu_util) = metrics.cpu_utilization {
            anomaly_metrics.insert(format!("cpu_utilization_{}", device_id), cpu_util as f64);
        }
        
        // Detect anomalies in hardware metrics
        let anomalies = self.detect_hardware_anomalies(timestamp, &anomaly_metrics).await?;
        
        // Send anomaly events
        for anomaly in anomalies {
            self.telemetry_sender.send_anomaly_detected(
                anomaly.anomaly_type,
                anomaly.severity,
                anomaly.affected_metrics,
                anomaly.deviation_magnitude,
                anomaly.context,
            ).await?;
        }
        
        Ok(())
    }
    
    /// Process inference completion for anomalies
    async fn process_inference_complete(
        &self,
        timestamp: u64,
        request_id: String,
        total_tokens: usize,
        total_duration_ms: u64,
        ttft_ms: Option<u64>,
        tpot_ms: Option<f64>,
    ) -> Result<()> {
        let mut anomaly_metrics = HashMap::new();
        
        // Check TTFT
        if let Some(ttft) = ttft_ms {
            anomaly_metrics.insert(format!("ttft_{}", request_id), ttft as f64);
        }
        
        // Check TPOT
        if let Some(tpot) = tpot_ms {
            anomaly_metrics.insert(format!("tpot_{}", request_id), tpot);
        }
        
        // Check total duration
        anomaly_metrics.insert(format!("total_duration_{}", request_id), total_duration_ms as f64);
        
        // Detect anomalies in inference metrics
        let anomalies = self.detect_inference_anomalies(timestamp, &anomaly_metrics).await?;
        
        // Send anomaly events
        for anomaly in anomalies {
            self.telemetry_sender.send_anomaly_detected(
                anomaly.anomaly_type,
                anomaly.severity,
                anomaly.affected_metrics,
                anomaly.deviation_magnitude,
                anomaly.context,
            ).await?;
        }
        
        Ok(())
    }
    
    /// Process model entropy for anomalies
    async fn process_model_entropy(
        &self,
        timestamp: u64,
        model_name: String,
        entropy: f32,
        token_index: Option<usize>,
    ) -> Result<()> {
        let mut anomaly_metrics = HashMap::new();
        anomaly_metrics.insert(format!("entropy_{}", model_name), entropy as f64);
        
        // Detect anomalies in entropy
        let anomalies = self.detect_entropy_anomalies(timestamp, &anomaly_metrics).await?;
        
        // Send anomaly events
        for anomaly in anomalies {
            self.telemetry_sender.send_anomaly_detected(
                anomaly.anomaly_type,
                anomaly.severity,
                anomaly.affected_metrics,
                anomaly.deviation_magnitude,
                anomaly.context,
            ).await?;
        }
        
        Ok(())
    }
    
    /// Detect hardware anomalies
    async fn detect_hardware_anomalies(
        &self,
        timestamp: u64,
        metrics: &HashMap<String, f64>,
    ) -> Result<Vec<Anomaly>> {
        let mut anomalies = Vec::new();
        
        for (metric_name, value) in metrics {
            // Simple threshold-based detection for now
            let anomaly = self.check_hardware_thresholds(timestamp, metric_name, *value).await;
            if let Some(anomaly) = anomaly {
                anomalies.push(anomaly);
            }
        }
        
        Ok(anomalies)
    }
    
    /// Detect inference anomalies
    async fn detect_inference_anomalies(
        &self,
        timestamp: u64,
        metrics: &HashMap<String, f64>,
    ) -> Result<Vec<Anomaly>> {
        let mut anomalies = Vec::new();
        
        for (metric_name, value) in metrics {
            // Simple threshold-based detection for now
            let anomaly = self.check_inference_thresholds(timestamp, metric_name, *value).await;
            if let Some(anomaly) = anomaly {
                anomalies.push(anomaly);
            }
        }
        
        Ok(anomalies)
    }
    
    /// Detect entropy anomalies
    async fn detect_entropy_anomalies(
        &self,
        timestamp: u64,
        metrics: &HashMap<String, f64>,
    ) -> Result<Vec<Anomaly>> {
        let mut anomalies = Vec::new();
        
        for (metric_name, value) in metrics {
            // Simple threshold-based detection for now
            let anomaly = self.check_entropy_thresholds(timestamp, metric_name, *value).await;
            if let Some(anomaly) = anomaly {
                anomalies.push(anomaly);
            }
        }
        
        Ok(anomalies)
    }
    
    /// Check hardware thresholds
    async fn check_hardware_thresholds(
        &self,
        timestamp: u64,
        metric_name: &str,
        value: f64,
    ) -> Option<Anomaly> {
        // Simple threshold-based anomaly detection
        if metric_name.contains("gpu_temperature") && value > 85.0 {
            Some(Anomaly {
                timestamp,
                anomaly_type: AnomalyType::HardwareFailure,
                severity: Severity::High,
                affected_metrics: vec![metric_name.to_string()],
                deviation_magnitude: value - 85.0,
                context: HashMap::new(),
            })
        } else if metric_name.contains("gpu_power") && value > 300.0 {
            Some(Anomaly {
                timestamp,
                anomaly_type: AnomalyType::SecurityThreat,
                severity: Severity::Critical,
                affected_metrics: vec![metric_name.to_string()],
                deviation_magnitude: value - 300.0,
                context: HashMap::new(),
            })
        } else if metric_name.contains("gpu_utilization") && value > 95.0 {
            Some(Anomaly {
                timestamp,
                anomaly_type: AnomalyType::PerformanceDegradation,
                severity: Severity::Medium,
                affected_metrics: vec![metric_name.to_string()],
                deviation_magnitude: value - 95.0,
                context: HashMap::new(),
            })
        } else {
            None
        }
    }
    
    /// Check inference thresholds
    async fn check_inference_thresholds(
        &self,
        timestamp: u64,
        metric_name: &str,
        value: f64,
    ) -> Option<Anomaly> {
        // Simple threshold-based anomaly detection
        if metric_name.contains("ttft") && value > 1000.0 {
            Some(Anomaly {
                timestamp,
                anomaly_type: AnomalyType::PerformanceDegradation,
                severity: Severity::High,
                affected_metrics: vec![metric_name.to_string()],
                deviation_magnitude: value - 1000.0,
                context: HashMap::new(),
            })
        } else if metric_name.contains("tpot") && value > 100.0 {
            Some(Anomaly {
                timestamp,
                anomaly_type: AnomalyType::PerformanceDegradation,
                severity: Severity::Medium,
                affected_metrics: vec![metric_name.to_string()],
                deviation_magnitude: value - 100.0,
                context: HashMap::new(),
            })
        } else {
            None
        }
    }
    
    /// Check entropy thresholds
    async fn check_entropy_thresholds(
        &self,
        timestamp: u64,
        metric_name: &str,
        value: f64,
    ) -> Option<Anomaly> {
        // Simple threshold-based anomaly detection
        if value < 0.1 {
            Some(Anomaly {
                timestamp,
                anomaly_type: AnomalyType::ModelInstability,
                severity: Severity::High,
                affected_metrics: vec![metric_name.to_string()],
                deviation_magnitude: 0.1 - value,
                context: HashMap::new(),
            })
        } else if value > 0.9 {
            Some(Anomaly {
                timestamp,
                anomaly_type: AnomalyType::EmergentBehavior,
                severity: Severity::Medium,
                affected_metrics: vec![metric_name.to_string()],
                deviation_magnitude: value - 0.9,
                context: HashMap::new(),
            })
        } else {
            None
        }
    }
    
    /// Get anomaly statistics
    pub async fn get_anomaly_stats(&self) -> HashMap<AnomalyType, u64> {
        self.anomaly_counters.read().await.clone()
    }
    
    /// Check detector health
    pub async fn health_check(&self) -> Result<AnomalyDetectorHealth> {
        let mut health = AnomalyDetectorHealth::new();
        
        let is_running = *self.is_running.read().await;
        if !is_running {
            health.add_issue("Anomaly detector not running".to_string());
        }
        
        // Check if any anomaly types have excessive counts
        let counters = self.anomaly_counters.read().await;
        for (anomaly_type, count) in counters.iter() {
            if *count > 1000 {
                health.add_issue(format!("Excessive {} anomalies: {}", 
                                       format!("{:?}", anomaly_type), count));
            }
        }
        
        Ok(health)
    }
}

/// Anomaly detector health status
#[derive(Debug)]
pub struct AnomalyDetectorHealth {
    pub issues: Vec<String>,
}

impl AnomalyDetectorHealth {
    pub fn new() -> Self {
        Self {
            issues: Vec::new(),
        }
    }
    
    pub fn is_healthy(&self) -> bool {
        self.issues.is_empty()
    }
    
    pub fn add_issue(&mut self, issue: String) {
        self.issues.push(issue);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_anomaly_detector_creation() {
        let config = AnomalyDetectionConfig::default();
        let telemetry_sender = TelemetrySender {
            sender: tokio::sync::mpsc::unbounded_channel().0,
            stats: Arc::new(RwLock::new(crate::telemetry::TelemetryStats::default())),
        };
        
        let detector = AnomalyDetector::new(config, telemetry_sender).await;
        assert!(detector.is_ok());
    }

    #[tokio::test]
    async fn test_hardware_anomaly_detection() {
        let config = AnomalyDetectionConfig::default();
        let telemetry_sender = TelemetrySender {
            sender: tokio::sync::mpsc::unbounded_channel().0,
            stats: Arc::new(RwLock::new(crate::telemetry::TelemetryStats::default())),
        };
        
        let detector = AnomalyDetector::new(config, telemetry_sender).await.unwrap();
        
        // Test high GPU temperature
        let mut metrics = HashMap::new();
        metrics.insert("gpu_temperature_device_0".to_string(), 90.0);
        
        let anomalies = detector.detect_hardware_anomalies(1234567890, &metrics).await.unwrap();
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::HardwareFailure);
        assert_eq!(anomalies[0].severity, Severity::High);
    }

    #[tokio::test]
    async fn test_inference_anomaly_detection() {
        let config = AnomalyDetectionConfig::default();
        let telemetry_sender = TelemetrySender {
            sender: tokio::sync::mpsc::unbounded_channel().0,
            stats: Arc::new(RwLock::new(crate::telemetry::TelemetryStats::default())),
        };
        
        let detector = AnomalyDetector::new(config, telemetry_sender).await.unwrap();
        
        // Test high TTFT
        let mut metrics = HashMap::new();
        metrics.insert("ttft_request_123".to_string(), 1500.0);
        
        let anomalies = detector.detect_inference_anomalies(1234567890, &metrics).await.unwrap();
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::PerformanceDegradation);
        assert_eq!(anomalies[0].severity, Severity::High);
    }

    #[tokio::test]
    async fn test_entropy_anomaly_detection() {
        let config = AnomalyDetectionConfig::default();
        let telemetry_sender = TelemetrySender {
            sender: tokio::sync::mpsc::unbounded_channel().0,
            stats: Arc::new(RwLock::new(crate::telemetry::TelemetryStats::default())),
        };
        
        let detector = AnomalyDetector::new(config, telemetry_sender).await.unwrap();
        
        // Test low entropy
        let mut metrics = HashMap::new();
        metrics.insert("entropy_model_123".to_string(), 0.05);
        
        let anomalies = detector.detect_entropy_anomalies(1234567890, &metrics).await.unwrap();
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::ModelInstability);
        assert_eq!(anomalies[0].severity, Severity::High);
    }
}