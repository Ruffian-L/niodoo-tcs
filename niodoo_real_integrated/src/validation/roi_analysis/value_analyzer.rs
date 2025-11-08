//! Value Analyzer
//!
//! Analyzes value provided by each component.

use super::{ComponentROI, ComponentValue};

/// Calculate ROI for a component
pub fn calculate_roi(
    component_name: &str,
    cost: &crate::validation::roi_analysis::ComponentCost,
    value: &ComponentValue,
) -> ComponentROI {
    let total_cost = cost.latency_ms + cost.memory_mb * 0.1; // Normalize memory to latency units
    let total_value = value.quality_improvement + value.learning_rate_improvement;

    let roi = if total_cost > 0.0 {
        (total_value - total_cost) / total_cost
    } else {
        0.0
    };

    ComponentROI {
        component_name: component_name.to_string(),
        cost: cost.clone(),
        value: value.clone(),
        roi,
    }
}



