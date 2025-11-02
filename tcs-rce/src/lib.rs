//! tcs-rce: Recursive Connectome Engine primitives
//!
//! This crate provides core building blocks for the RCE layer:
//! - Persistent Laplacian wrappers built on tcs-tda
//! - β_meta composite metric computation
//! - Sheaf descriptor interfaces (read-only metrics for Phase 3)
//! - Lightweight metrics structures for integration with telemetry layers

pub mod laplacian;
pub mod beta_meta;
pub mod sheaf;
pub mod rce_metrics;

pub use beta_meta::{BetaMetaInputs, BetaMetaWeights};
pub use laplacian::{LaplacianAnalyzer, LaplacianSummary};
pub use rce_metrics::{BetaMetaSnapshot, RceMetricSeries};


