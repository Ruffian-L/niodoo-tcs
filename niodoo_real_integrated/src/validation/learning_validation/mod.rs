//! Continuous Learning Validation
//!
//! Validates that real-time learning works without catastrophic forgetting.

pub mod forgetting_tests;
pub mod incremental_learning;
pub mod breakthrough_detection;
pub mod safety_validation;

pub use forgetting_tests::*;
pub use incremental_learning::*;
pub use breakthrough_detection::*;
pub use safety_validation::*;

use serde::{Deserialize, Serialize};

/// Learning validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningValidationResult {
    pub test_name: String,
    pub forgetting_rate: f64,
    pub improvement_rate: f64,
    pub breakthrough_precision: Option<f64>,
    pub safety_score_delta: Option<f64>,
    pub timestamp: String,
}



