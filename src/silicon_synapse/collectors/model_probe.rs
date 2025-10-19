//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! # Model Probe
//!
//! This module implements model internal state probing.

use std::sync::Arc;
use std::time::SystemTime;
use anyhow::Result;
use tokio::sync::RwLock;
use tracing::{info, error, warn};

use crate::silicon_synapse::config::ModelProbeConfig;
use crate::silicon_synapse::telemetry_bus::{TelemetryBus, TelemetrySender};
use crate::silicon_synapse::collectors::{Collector, ModelMetrics};

/// Model probe implementation
pub struct ModelProbe {
    config: ModelProbeConfig,
    telemetry_bus: Arc<TelemetryBus>,
    is_running: Arc<RwLock<bool>>,
}

impl ModelProbe {
    /// Create a new model probe
    pub async fn new(config: ModelProbeConfig, telemetry_bus: Arc<TelemetryBus>) -> Result<Self> {
        info!("Initializing model probe");
        
        Ok(Self {
            config,
            telemetry_bus,
            is_running: Arc::new(RwLock::new(false)),
        })
    }
    
    /// Probe model internal state
    pub async fn probe_model_state(&self, layer_name: String, activations: Vec<f32>) -> Result<()> {
        let event = crate::silicon_synapse::telemetry_bus::TelemetryEvent::LayerActivation {
            timestamp: SystemTime::now(),
            layer_name,
            activations,
        };
        
        self.telemetry_bus.send(event).await?;
        Ok(())
    }
}

impl Collector for ModelProbe {
    async fn start(&self) -> Result<()> {
        info!("Starting model probe");
        *self.is_running.write().await = true;
        Ok(())
    }
    
    async fn stop(&self) -> Result<()> {
        info!("Stopping model probe");
        *self.is_running.write().await = false;
        Ok(())
    }
    
    fn is_running(&self) -> bool {
        false
    }
}

impl Clone for ModelProbe {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            telemetry_bus: self.telemetry_bus.clone(),
            is_running: self.is_running.clone(),
        }
    }
}