//! Scale Testing Infrastructure
//!
//! Validates system at production scale (1K, 10K, 100K interactions).

pub mod load_generator;
pub mod metrics_collector;

pub use load_generator::*;
pub use metrics_collector::*;

use serde::{Deserialize, Serialize};

/// Scale test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleTestResult {
    pub interaction_count: usize,
    pub metrics: ScaleMetrics,
    pub timestamp: String,
}

/// Scale test metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleMetrics {
    pub rouge_scores: Vec<f64>,
    pub latency_ms: Vec<f64>,
    pub memory_usage_mb: f64,
    pub improvement_rate: f64,
    pub stability_score: f64, // 0-1, higher is more stable
}



