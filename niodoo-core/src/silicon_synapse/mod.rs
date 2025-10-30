// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! # Silicon Synapse Hardware Monitoring System
//!
//! A comprehensive hardware-grounded AI state monitoring system for the Niodoo-Feeling Gen 1 consciousness engine.

use std::sync::Arc;
use tokio::sync::watch;
use tracing::info;

// Module declarations
pub mod aggregation;
pub mod baseline;
pub mod collectors;
pub mod config;
pub mod exporters;
pub mod plugins;
pub mod telemetry_bus;

// Re-export key types for easy access
pub use aggregation::InferenceMetrics as AggregationInferenceMetrics;
pub use aggregation::{AggregatedMetrics, AggregationEngine};
pub use baseline::{
    Anomaly, AnomalyDetector as BaselineAnomalyDetector, AnomalyType, BaselineManager, Severity,
};
pub use collectors::hardware::{HardwareCollector, HardwareMetrics};
pub use collectors::inference::{InferenceCollector, InferenceMetrics};
pub use collectors::model::{ModelMetrics, ModelProbe};
pub use config::{
    BaselineConfig, Config, ExporterConfig, HardwareConfig, InferenceConfig, ModelProbeConfig,
};
pub use exporters::prometheus::PrometheusExporter;
pub use telemetry_bus::{TelemetryBus, TelemetryEvent, TelemetrySender};

/// Main Silicon Synapse monitoring system
pub struct SiliconSynapse {
    config: Config,
    telemetry_bus: TelemetryBus,
    hardware_collector: Option<HardwareCollector>,
    inference_collector: Option<InferenceCollector>,
    model_probe: Option<ModelProbe>,
    aggregation_engine: Arc<AggregationEngine>,
    baseline_manager: BaselineManager,
    prometheus_exporter: Option<PrometheusExporter>,
    is_running: Arc<tokio::sync::RwLock<bool>>,
    metrics_tx: Option<watch::Sender<AggregatedMetrics>>,
    metrics_rx: Option<watch::Receiver<AggregatedMetrics>>,
}

/// Error types for Silicon Synapse operations
#[derive(Debug, thiserror::Error)]
pub enum SiliconSynapseError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Telemetry bus error: {0}")]
    TelemetryBus(#[from] telemetry_bus::TelemetryBusError),

    #[error("Collector error: {0}")]
    Collector(String),

    #[error("Aggregation error: {0}")]
    Aggregation(String),

    #[error("Baseline error: {0}")]
    Baseline(String),

    #[error("Exporter error: {0}")]
    Exporter(String),

    #[error("Hardware monitor error: {0}")]
    HardwareMonitor(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Generic error: {0}")]
    Generic(String),
}

// Additional From implementations for common conversions
impl From<String> for SiliconSynapseError {
    fn from(err: String) -> Self {
        SiliconSynapseError::Generic(err)
    }
}

impl From<Box<dyn std::error::Error>> for SiliconSynapseError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        SiliconSynapseError::Generic(err.to_string())
    }
}

impl SiliconSynapse {
    /// Create a new Silicon Synapse monitoring system
    pub async fn new(config: Config) -> Result<Self, SiliconSynapseError> {
        info!("Initializing Silicon Synapse monitoring system");

        // 1. Create telemetry bus
        let telemetry_bus = TelemetryBus::new(config.telemetry.clone())?;

        // 2. Create collectors
        let hardware_collector =
            HardwareCollector::new(config.hardware.clone(), telemetry_bus.sender())?;
        let inference_collector =
            InferenceCollector::new(config.inference.clone(), telemetry_bus.sender())?;
        let model_probe =
            ModelProbe::new(config.model_probe.clone().into(), telemetry_bus.sender())?;

        // 3. Create aggregation engine
        let (metrics_tx, metrics_rx) = watch::channel(AggregatedMetrics::default());
        let aggregation_engine = Arc::new(
            AggregationEngine::new(
                config.aggregation.clone(),
                telemetry_bus.sender(),
                metrics_tx.clone(),
            )
            .map_err(SiliconSynapseError::Aggregation)?,
        );

        // 4. Create baseline manager (includes anomaly detector)
        let baseline_manager = BaselineManager::new(config.baseline.clone().into());

        // 5. Create Prometheus exporter
        let prometheus_exporter = PrometheusExporter::new(config.exporter.clone())
            .map_err(|e| SiliconSynapseError::Exporter(e.to_string()))?;

        Ok(Self {
            config,
            telemetry_bus,
            hardware_collector: Some(hardware_collector),
            inference_collector: Some(inference_collector),
            model_probe: Some(model_probe),
            aggregation_engine,
            baseline_manager,
            prometheus_exporter: Some(prometheus_exporter),
            is_running: Arc::new(tokio::sync::RwLock::new(false)),
            metrics_tx: Some(metrics_tx),
            metrics_rx: Some(metrics_rx),
        })
    }

    /// Start the monitoring system
    pub async fn start(&mut self) -> std::result::Result<(), SiliconSynapseError> {
        info!("Starting Silicon Synapse monitoring system");
        *self.is_running.write().await = true;

        // 1. Start telemetry bus first
        let receiver = self.telemetry_bus.take_receiver();
        let aggregation_clone = Arc::clone(&self.aggregation_engine);
        tokio::spawn(async move {
            let mut rx = receiver;
            while let Some(event) = rx.recv().await {
                TelemetryBus::route_event(event.clone()).await;
                aggregation_clone.process_event(event).await;
            }
        });

        // 2. Start collectors (hardware, inference, model)
        if let Some(ref mut collector) = self.hardware_collector {
            collector.start().await?;
        }
        if let Some(ref mut collector) = self.inference_collector {
            collector.start().await?;
        }
        if let Some(ref mut probe) = self.model_probe {
            probe.start().await?;
        }

        // 3. Start aggregation engine
        AggregationEngine::start_arc(&self.aggregation_engine)
            .await
            .map_err(SiliconSynapseError::Aggregation)?;

        // 4. Start baseline manager
        self.baseline_manager
            .start()
            .await
            .map_err(SiliconSynapseError::Baseline)?;

        // 5. Start Prometheus exporter last
        if let Some(ref mut exporter) = self.prometheus_exporter {
            exporter
                .start()
                .await
                .map_err(|e| SiliconSynapseError::Exporter(e.to_string()))?;
        }

        info!("Silicon Synapse monitoring system started successfully.");
        Ok(())
    }

    /// Check if the monitoring system is currently running
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    /// Shutdown the monitoring system
    pub async fn shutdown(&mut self) -> Result<(), SiliconSynapseError> {
        info!("Shutting down Silicon Synapse monitoring system");
        *self.is_running.write().await = false;

        // Stop in reverse order: exporter → baseline → aggregation → collectors → telemetry bus

        // 1. Stop Prometheus exporter first
        if let Some(ref mut exporter) = self.prometheus_exporter {
            exporter
                .stop()
                .await
                .map_err(|e| SiliconSynapseError::Exporter(e.to_string()))?;
        }

        // 2. Stop baseline manager
        self.baseline_manager
            .stop()
            .await
            .map_err(SiliconSynapseError::Baseline)?;

        // 3. Stop aggregation engine
        AggregationEngine::stop_arc(&self.aggregation_engine)
            .await
            .map_err(SiliconSynapseError::Aggregation)?;

        // 4. Stop collectors
        if let Some(ref mut collector) = self.hardware_collector {
            collector.stop().await?;
        }
        if let Some(ref mut collector) = self.inference_collector {
            collector.stop().await?;
        }
        if let Some(ref mut probe) = self.model_probe {
            probe.stop().await?;
        }

        // 5. Telemetry bus doesn't need explicit stopping

        info!("Silicon Synapse monitoring system shut down.");
        Ok(())
    }

    /// Get a sender for telemetry events
    pub fn telemetry_sender(&self) -> TelemetrySender {
        self.telemetry_bus.sender()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_silicon_synapse_initialization() {
        let config = Config::default();
        let synapse = SiliconSynapse::new(config).await;
        assert!(synapse.is_ok());
    }

    #[tokio::test]
    async fn test_silicon_synapse_disabled_config() {
        let mut config = Config::default();
        config.enabled = false;

        let mut synapse = SiliconSynapse::new(config).await.unwrap();

        // Should start successfully even when disabled
        synapse.start().await.unwrap();
        assert!(!synapse.is_running()); // Should not be running when disabled
    }

    #[tokio::test]
    async fn test_full_system_integration() {
        let config = Config::default();
        let mut synapse = SiliconSynapse::new(config).await.unwrap();

        // Start the system
        synapse.start().await.unwrap();

        // Emit test events
        let tx = synapse.telemetry_sender();

        // Emit inference events
        let request_id = Uuid::new_v4();
        tx.try_send(TelemetryEvent::InferenceStart {
            request_id,
            timestamp: std::time::Instant::now(),
            prompt_length: 100,
        })
        .unwrap();

        tx.try_send(TelemetryEvent::TokenGenerated {
            request_id,
            token_id: 0,
            timestamp: std::time::Instant::now(),
            logits: Some(vec![0.1, 0.2, 0.3]),
        })
        .unwrap();

        tx.try_send(TelemetryEvent::InferenceComplete {
            request_id,
            timestamp: std::time::Instant::now(),
            total_tokens: 1,
            error: None,
        })
        .unwrap();

        // Emit hardware metrics
        tx.try_send(TelemetryEvent::HardwareMetrics {
            timestamp: std::time::SystemTime::now(),
            gpu_temp_celsius: Some(72.5),
            gpu_power_watts: Some(245.3),
            gpu_fan_speed_percent: Some(65.0),
            vram_used_bytes: Some(8589934592),
            vram_total_bytes: Some(17179869184),
            gpu_utilization_percent: Some(85.0),
            cpu_utilization_percent: 45.0,
            ram_used_bytes: 16106127360,
        })
        .unwrap();

        // Emit model metrics
        tx.try_send(TelemetryEvent::ModelMetrics {
            timestamp: std::time::Instant::now(),
            layer_index: 12,
            entropy: Some(0.342),
            activation_sparsity: Some(0.87),
            activation_magnitude_mean: Some(0.45),
            activation_magnitude_std: Some(0.12),
        })
        .unwrap();

        // Wait for processing
        tokio::time::sleep(crate::utils::threshold_convenience::timeout(
            crate::utils::TimeoutCriticality::High,
        ))
        .await;

        // Verify metrics endpoint works (if exporter is running)
        if let Some(ref exporter) = synapse.prometheus_exporter {
            // Test that the exporter is running by checking its state
            // In a real test, we would make an HTTP request to the metrics endpoint
            assert!(exporter.is_running());
        }

        // Shutdown the system
        synapse.shutdown().await.unwrap();
    }
}

// Stub stop for TelemetryBus
impl TelemetryBus {
    pub async fn stop(&self) -> Result<(), SiliconSynapseError> {
        // Stub implementation
        info!("Telemetry bus stopped");
        Ok(())
    }
}
