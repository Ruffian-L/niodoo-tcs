//! # Utility Modules
//!
//! Collection of utility modules for the Niodoo consciousness framework

pub mod capacity;
pub mod thresholds;

// Re-export commonly used types
pub use thresholds::{ThresholdCalculator, ThresholdConfig, TimeoutCriticality};

// Re-export convenience modules (avoiding name conflicts)
pub use capacity::convenience as capacity_convenience;
pub use thresholds::convenience as threshold_convenience;
