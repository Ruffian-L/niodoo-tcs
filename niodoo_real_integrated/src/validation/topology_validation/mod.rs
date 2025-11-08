//! Topology Impact Validation
//!
//! Validates that TDA provides measurable value over standard methods.

pub mod betti_validation;
pub mod persistence_validation;
pub mod knot_validation;
pub mod retrieval_validation;

pub use betti_validation::*;
pub use persistence_validation::*;
pub use knot_validation::*;
pub use retrieval_validation::*;

use serde::{Deserialize, Serialize};

/// Topology validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyValidationResult {
    pub experiment_name: String,
    pub correlation: f64,
    pub improvement_pct: f64,
    pub statistical_significance: f64, // p-value
    pub timestamp: String,
}



