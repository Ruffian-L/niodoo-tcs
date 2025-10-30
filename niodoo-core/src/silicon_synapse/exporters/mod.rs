// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! # Exporters Module
//!
//! This module implements the exporter manager and individual exporters
//! for Prometheus and a JSON API.

use crate::silicon_synapse::config::ExporterConfig;
use crate::silicon_synapse::SiliconSynapseError;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::info;

pub mod json_api;
pub mod prometheus;

/// Manager for all metric exporters
pub struct ExporterManager {
    config: ExporterConfig,
    prometheus_exporter: Option<Arc<Mutex<prometheus::PrometheusExporter>>>,
    json_api_exporter: Option<Arc<Mutex<json_api::JsonApiExporter>>>,
    is_running: Arc<RwLock<bool>>,
}

impl ExporterManager {
    /// Create a new exporter manager
    pub fn new(config: ExporterConfig) -> Result<Self, SiliconSynapseError> {
        let mut manager = Self {
            config,
            prometheus_exporter: None,
            json_api_exporter: None,
            is_running: Arc::new(RwLock::new(false)),
        };

        // Initialize exporters based on configuration
        if manager.config.exporter_type == "prometheus" || manager.config.exporter_type == "all" {
            manager.prometheus_exporter = Some(Arc::new(Mutex::new(
                prometheus::PrometheusExporter::new(manager.config.clone())?,
            )));
        }

        if manager.config.exporter_type == "json" || manager.config.exporter_type == "all" {
            manager.json_api_exporter = Some(Arc::new(Mutex::new(json_api::JsonApiExporter::new(
                manager.config.clone(),
            )?)));
        }

        Ok(manager)
    }

    /// Start all exporters
    pub async fn start(&mut self) -> Result<(), SiliconSynapseError> {
        if *self.is_running.read().await {
            return Err(SiliconSynapseError::Exporter(
                "Exporter manager is already running".to_string(),
            ));
        }

        info!("Starting exporter manager");

        // Start Prometheus exporter
        if let Some(exporter) = &self.prometheus_exporter {
            exporter.lock().await.start().await?;
            info!("Prometheus exporter started");
        }

        // Start JSON API exporter
        if let Some(exporter) = &self.json_api_exporter {
            exporter.lock().await.start().await?;
            info!("JSON API exporter started");
        }

        *self.is_running.write().await = true;
        Ok(())
    }

    /// Stop all exporters
    pub async fn stop(&mut self) -> Result<(), SiliconSynapseError> {
        if !*self.is_running.read().await {
            return Ok(());
        }

        info!("Stopping exporter manager");

        // Stop Prometheus exporter
        if let Some(exporter) = &self.prometheus_exporter {
            exporter.lock().await.stop().await?;
            info!("Prometheus exporter stopped");
        }

        // Stop JSON API exporter
        if let Some(exporter) = &self.json_api_exporter {
            exporter.lock().await.stop().await?;
            info!("JSON API exporter stopped");
        }

        *self.is_running.write().await = false;
        Ok(())
    }

    /// Get the Prometheus exporter
    pub fn prometheus_exporter(&self) -> Option<Arc<Mutex<prometheus::PrometheusExporter>>> {
        self.prometheus_exporter.clone()
    }

    /// Get the JSON API exporter
    pub fn json_api_exporter(&self) -> Option<Arc<Mutex<json_api::JsonApiExporter>>> {
        self.json_api_exporter.clone()
    }
}

// Note: SiliconSynapseError is now defined in silicon_synapse/mod.rs
// to avoid duplicate definitions and properly consolidate error handling.
