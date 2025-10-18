//! Baseline learning and anomaly detection for Silicon Synapse
//!
//! This module implements baseline learning, anomaly detection, and classification
//! capabilities for identifying deviations from normal operational behavior.

// use crate::silicon_synapse::aggregation::AggregatedMetrics; // Currently unused
// use std::collections::HashMap; // Currently unused
// use std::time::SystemTime; // Currently unused
// use uuid::Uuid; // Currently unused

pub mod detector;
pub mod manager;
pub mod model;

#[cfg(test)]
mod tests;

pub use detector::{Anomaly, AnomalyDetector, AnomalyType, DetectorConfig, Severity};
pub use manager::{BaselineManager, DetectorStats, LearningProgress};
pub use model::{BaselineModel, CorrelationMatrix, MetricStats, Percentiles};
