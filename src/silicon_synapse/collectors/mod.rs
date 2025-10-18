//! # Collectors Module
//!
//! This module implements the collector manager and individual collectors
//! for hardware, inference, and model probing.

use anyhow::Result;
use tracing::info;

use crate::silicon_synapse::collectors::model::ModelProbeConfig;
use crate::silicon_synapse::config::{HardwareConfig, InferenceConfig};
use crate::silicon_synapse::telemetry_bus::TelemetrySender;
use crate::silicon_synapse::SiliconSynapseError;

pub mod hardware;
pub mod inference;
pub mod model;

/// Manages all active metric collectors
pub struct CollectorManager {
    hardware_collector: Option<hardware::HardwareCollector>,
    inference_collector: Option<inference::InferenceCollector>,
    model_probe: Option<model::ModelProbe>,
    /// Telemetry sender for broadcasting events
    #[allow(dead_code)]
    telemetry_sender: TelemetrySender,
}

impl CollectorManager {
    pub fn new(
        hardware_config: Option<HardwareConfig>,
        inference_config: Option<InferenceConfig>,
        model_probe_config: Option<ModelProbeConfig>,
        telemetry_sender: TelemetrySender,
    ) -> Result<Self, SiliconSynapseError> {
        let hardware_collector = if let Some(config) = hardware_config {
            Some(hardware::HardwareCollector::new(
                config,
                telemetry_sender.clone(),
            )?)
        } else {
            None
        };

        let inference_collector = if let Some(config) = inference_config {
            Some(inference::InferenceCollector::new(
                config,
                telemetry_sender.clone(),
            )?)
        } else {
            None
        };

        let model_probe = if let Some(config) = model_probe_config {
            Some(model::ModelProbe::new(config, telemetry_sender.clone())?)
        } else {
            None
        };

        Ok(Self {
            hardware_collector,
            inference_collector,
            model_probe,
            telemetry_sender,
        })
    }

    pub async fn start(&mut self) -> Result<(), SiliconSynapseError> {
        info!("Starting collector manager");
        if let Some(collector) = &mut self.hardware_collector {
            collector.start().await?;
            info!("Hardware collector started.");
        }
        if let Some(collector) = &mut self.inference_collector {
            collector.start().await?;
            info!("Inference collector started.");
        }
        if let Some(probe) = &mut self.model_probe {
            probe.start().await?;
            info!("Model probe started.");
        }
        Ok(())
    }

    pub async fn stop(&mut self) -> Result<(), SiliconSynapseError> {
        info!("Stopping collector manager");
        if let Some(collector) = &mut self.hardware_collector {
            collector.stop().await?;
            info!("Hardware collector stopped.");
        }
        if let Some(collector) = &mut self.inference_collector {
            collector.stop().await?;
            info!("Inference collector stopped.");
        }
        if let Some(probe) = &mut self.model_probe {
            probe.stop().await?;
            info!("Model probe stopped.");
        }
        Ok(())
    }
}
