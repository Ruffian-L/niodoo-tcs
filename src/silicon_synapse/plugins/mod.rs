//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Plugin system for Silicon Synapse monitoring
//!
//! This module provides a trait-based plugin system that allows for extensible
//! metric collection, anomaly detection, and metric export.

pub mod collector;
pub mod detector;
pub mod exporter;
pub mod registry;

pub use collector::{Collector, CollectorError, CollectorResult};
pub use detector::{AnomalyDetector, DetectorError, DetectorResult};
pub use exporter::{ExporterError, ExporterResult, MetricExporter};
pub use registry::{PluginRegistry, PluginRegistryError};
