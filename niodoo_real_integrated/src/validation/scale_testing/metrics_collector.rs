//! Metrics Collector
//!
//! Collects and aggregates metrics during scale testing.

use super::ScaleMetrics;
use serde::{Deserialize, Serialize};

/// Collect metrics at milestones
pub fn collect_metrics_at_milestone(
    metrics: &ScaleMetrics,
    milestone: usize,
) -> MilestoneReport {
    let relevant_rouge = if metrics.rouge_scores.len() >= milestone {
        &metrics.rouge_scores[..milestone]
    } else {
        &metrics.rouge_scores[..]
    };

    let mean_rouge = relevant_rouge.iter().sum::<f64>() / relevant_rouge.len() as f64;
    let mean_latency = if metrics.latency_ms.len() >= milestone {
        metrics.latency_ms[..milestone].iter().sum::<f64>() / milestone as f64
    } else {
        metrics.latency_ms.iter().sum::<f64>() / metrics.latency_ms.len() as f64
    };

    MilestoneReport {
        milestone,
        mean_rouge,
        mean_latency_ms: mean_latency,
        stability_score: metrics.stability_score,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilestoneReport {
    pub milestone: usize,
    pub mean_rouge: f64,
    pub mean_latency_ms: f64,
    pub stability_score: f64,
}



