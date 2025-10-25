//! Adaptive metrics aggregation and failure signal utilities.

mod failure_signals;
mod retry_controller;

pub use failure_signals::{
    AdaptiveMetricsSnapshot, AggregatedFailureSignals, FailureSeverity, FailureSignal,
    FailureSignalAggregator, FailureSignalCode, FailureSignalThresholds,
};
pub use failure_signals::simple::FailureSignals;
pub use retry_controller::{
    AdaptiveRetryController, AdaptiveRetryDecision, AdaptiveRetryLevel, RetryControllerConfig,
};

