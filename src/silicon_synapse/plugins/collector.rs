//! Collector plugin trait for Silicon Synapse monitoring
//!
//! This module defines the trait interface for pluggable metric collectors.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// Result type for collector operations
pub type CollectorResult<T> = Result<T, CollectorError>;

/// Error types for collector operations
#[derive(Debug, thiserror::Error)]
pub enum CollectorError {
    #[error("Collector initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Collection failed: {0}")]
    CollectionFailed(String),

    #[error("Shutdown failed: {0}")]
    ShutdownFailed(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Hardware access error: {0}")]
    HardwareAccessError(String),

    #[error("Resource unavailable: {0}")]
    ResourceUnavailable(String),
}

/// Configuration for a collector plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectorConfig {
    /// Unique identifier for this collector instance
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Collection interval
    pub interval: Duration,
    /// Whether this collector is enabled
    pub enabled: bool,
    /// Custom configuration parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: "Unknown Collector".to_string(),
            interval: Duration::from_secs(1),
            enabled: true,
            parameters: HashMap::new(),
        }
    }
}

/// Health status of a collector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectorHealth {
    /// Whether the collector is healthy
    pub is_healthy: bool,
    /// Last successful collection timestamp
    pub last_collection: Option<SystemTime>,
    /// Number of consecutive failures
    pub failure_count: u32,
    /// Error message if unhealthy
    pub error_message: Option<String>,
    /// Collection statistics
    pub stats: CollectorStats,
}

/// Collection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectorStats {
    /// Total number of collections
    pub total_collections: u64,
    /// Number of successful collections
    pub successful_collections: u64,
    /// Number of failed collections
    pub failed_collections: u64,
    /// Average collection duration
    pub avg_collection_duration_ms: f64,
    /// Last collection duration
    pub last_collection_duration_ms: Option<f64>,
}

impl Default for CollectorStats {
    fn default() -> Self {
        Self {
            total_collections: 0,
            successful_collections: 0,
            failed_collections: 0,
            avg_collection_duration_ms: 0.0,
            last_collection_duration_ms: None,
        }
    }
}

/// Trait for pluggable metric collectors
#[async_trait]
pub trait Collector: Send + Sync {
    /// Get the collector's unique identifier
    fn id(&self) -> &str;

    /// Get the collector's human-readable name
    fn name(&self) -> &str;

    /// Get the collector's configuration
    fn config(&self) -> &CollectorConfig;

    /// Initialize the collector
    async fn initialize(&mut self) -> CollectorResult<()>;

    /// Collect metrics
    async fn collect(&self) -> CollectorResult<CollectedMetrics>;

    /// Start the collector
    async fn start(&mut self) -> CollectorResult<()>;

    /// Stop the collector
    async fn stop(&mut self) -> CollectorResult<()>;

    /// Check if collector is running
    fn is_running(&self) -> bool;

    /// Shutdown the collector
    async fn shutdown(&mut self) -> CollectorResult<()>;

    /// Check collector health
    async fn health_check(&self) -> CollectorResult<CollectorHealth>;

    /// Update configuration
    async fn update_config(&mut self, config: CollectorConfig) -> CollectorResult<()>;
}

/// Metrics collected by a collector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectedMetrics {
    /// Collector ID that collected these metrics
    pub collector_id: String,
    /// Timestamp when metrics were collected
    pub timestamp: SystemTime,
    /// Collection duration
    pub collection_duration: Duration,
    /// Raw metric values
    pub metrics: HashMap<String, MetricValue>,
    /// Metadata about the collection
    pub metadata: HashMap<String, String>,
}

/// A single metric value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    /// Integer value
    Integer(i64),
    /// Floating point value
    Float(f64),
    /// String value
    String(String),
    /// Boolean value
    Boolean(bool),
    /// Counter value (monotonically increasing)
    Counter(u64),
    /// Gauge value (can go up or down)
    Gauge(f64),
    /// Histogram value (distribution of values)
    Histogram(Vec<f64>),
}

impl MetricValue {
    /// Convert to f64 for numeric operations
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetricValue::Integer(i) => Some(*i as f64),
            MetricValue::Float(f) => Some(*f),
            MetricValue::Counter(c) => Some(*c as f64),
            MetricValue::Gauge(g) => Some(*g),
            MetricValue::Histogram(h) => h.first().copied(),
            _ => None,
        }
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        match self {
            MetricValue::Integer(i) => i.to_string(),
            MetricValue::Float(f) => f.to_string(),
            MetricValue::String(s) => s.clone(),
            MetricValue::Boolean(b) => b.to_string(),
            MetricValue::Counter(c) => c.to_string(),
            MetricValue::Gauge(g) => g.to_string(),
            MetricValue::Histogram(h) => format!(
                "[{}]",
                h.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

/// Base collector implementation with common functionality
pub struct BaseCollector {
    config: CollectorConfig,
    stats: CollectorStats,
    last_collection: Option<SystemTime>,
    failure_count: u32,
}

impl BaseCollector {
    /// Create a new base collector
    pub fn new(config: CollectorConfig) -> Self {
        Self {
            config,
            stats: CollectorStats::default(),
            last_collection: None,
            failure_count: 0,
        }
    }

    /// Update collection statistics
    pub fn update_stats(&mut self, success: bool, duration: Duration) {
        self.stats.total_collections += 1;

        if success {
            self.stats.successful_collections += 1;
            self.failure_count = 0;
            self.last_collection = Some(SystemTime::now());
        } else {
            self.stats.failed_collections += 1;
            self.failure_count += 1;
        }

        let duration_ms = duration.as_millis() as f64;
        self.stats.last_collection_duration_ms = Some(duration_ms);

        // Update average duration
        let total = self.stats.total_collections as f64;
        self.stats.avg_collection_duration_ms =
            (self.stats.avg_collection_duration_ms * (total - 1.0) + duration_ms) / total;
    }

    /// Get current health status
    pub fn get_health(&self) -> CollectorHealth {
        CollectorHealth {
            is_healthy: self.failure_count < 5, // Consider unhealthy after 5 consecutive failures
            last_collection: self.last_collection,
            failure_count: self.failure_count,
            error_message: if self.failure_count >= 5 {
                Some(format!(
                    "Collector has failed {} consecutive times",
                    self.failure_count
                ))
            } else {
                None
            },
            stats: self.stats.clone(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &CollectorConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: CollectorConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_metric_value_conversions() {
        let int_val = MetricValue::Integer(42);
        assert_eq!(int_val.as_f64(), Some(42.0));
        assert_eq!(int_val.to_string(), "42");

        let float_val = MetricValue::Float(3.14);
        assert_eq!(float_val.as_f64(), Some(3.14));
        assert_eq!(float_val.to_string(), "3.14");

        let string_val = MetricValue::String("test".to_string());
        assert_eq!(string_val.as_f64(), None);
        assert_eq!(string_val.to_string(), "test");

        let bool_val = MetricValue::Boolean(true);
        assert_eq!(bool_val.as_f64(), None);
        assert_eq!(bool_val.to_string(), "true");
    }

    #[test]
    fn test_collector_config_default() {
        let config = CollectorConfig::default();
        assert!(!config.id.is_empty());
        assert_eq!(config.name, "Unknown Collector");
        assert_eq!(config.interval, Duration::from_secs(1));
        assert!(config.enabled);
        assert!(config.parameters.is_empty());
    }

    #[test]
    fn test_base_collector_stats() {
        let config = CollectorConfig::default();
        let mut collector = BaseCollector::new(config);

        // Initial state
        let health = collector.get_health();
        assert!(health.is_healthy);
        assert_eq!(health.stats.total_collections, 0);

        // Update stats
        collector.update_stats(true, Duration::from_millis(100));
        let health = collector.get_health();
        assert!(health.is_healthy);
        assert_eq!(health.stats.total_collections, 1);
        assert_eq!(health.stats.successful_collections, 1);
        assert_eq!(health.stats.failed_collections, 0);
        assert_eq!(health.stats.avg_collection_duration_ms, 100.0);

        // Multiple failures
        for _ in 0..6 {
            collector.update_stats(false, Duration::from_millis(50));
        }

        let health = collector.get_health();
        assert!(!health.is_healthy);
        assert_eq!(health.failure_count, 6);
        assert!(health.error_message.is_some());
    }
}
