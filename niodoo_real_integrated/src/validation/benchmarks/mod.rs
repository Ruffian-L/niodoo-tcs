//! Comparative Benchmarks
//!
//! Head-to-head comparisons with established systems.

pub mod baseline_rag;
pub mod baseline_memgpt;
pub mod test_suites;

pub use baseline_rag::*;
pub use baseline_memgpt::*;
pub use test_suites::*;

use serde::{Deserialize, Serialize};

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub system_name: String,
    pub test_suite: String,
    pub metrics: BenchmarkMetrics,
    pub timestamp: String,
}

/// Benchmark metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub accuracy: f64,
    pub latency_ms: f64,
    pub rouge_score: f64,
    pub memory_usage_mb: f64,
}



