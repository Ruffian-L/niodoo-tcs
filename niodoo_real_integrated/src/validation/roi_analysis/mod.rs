//! ROI Analysis Framework
//!
//! Cost/benefit analysis for each component.

pub mod cost_tracker;
pub mod value_analyzer;

pub use cost_tracker::*;
pub use value_analyzer::*;

use serde::{Deserialize, Serialize};

/// Component ROI result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentROI {
    pub component_name: String,
    pub cost: ComponentCost,
    pub value: ComponentValue,
    pub roi: f64, // (value - cost) / cost
}

/// Component cost metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentCost {
    pub latency_ms: f64,
    pub memory_mb: f64,
    pub cpu_percent: f64,
    pub training_time_secs: Option<f64>,
}

/// Component value metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentValue {
    pub quality_improvement: f64, // Percentage improvement
    pub learning_rate_improvement: f64,
    pub user_satisfaction: Option<f64>,
}



