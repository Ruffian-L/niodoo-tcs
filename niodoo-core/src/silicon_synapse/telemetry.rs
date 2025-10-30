// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Telemetry bus for Silicon Synapse monitoring system
//!
//! This module implements the asynchronous telemetry event bus that decouples metric production
//! from collection, ensuring the consciousness engine never blocks on monitoring.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::mpsc;
use uuid::Uuid;
use crate::silicon_synapse::config::TelemetryConfig;

/// Telemetry event types emitted by consciousness engine components
#[derive(Debug, Clone)]
pub enum TelemetryEvent {
    /// Inference request started
    InferenceStart {
        request_id: Uuid,
        timestamp: Instant,
        prompt_length: usize,
    },
    
    /// Token generated during inference
    TokenGenerated {
        request_id: Uuid,
        timestamp: Instant,
        token_index: usize,
        token_length: usize,
    },
    
    /// Inference request completed
    InferenceComplete {
        request_id: Uuid,
        timestamp: Instant,
        total_tokens: usize,
        error: Option<String>,
    },
    
    /// Hardware metrics collected
    HardwareMetrics {
        timestamp: Instant,
        gpu_temperature: Option<f32>,
        gpu_power: Option<f32>,
        gpu_utilization: Option<f32>,
        gpu_memory_used: Option<u64>,
        gpu_memory_total: Option<u64>,
        cpu_temperature: Option<f32>,
        cpu_utilization: Option<f32>,
        system_memory_used: Option<u64>,
        system_memory_total: Option<u64>,
    },
    
    /// Model internal state metrics
    ModelMetrics {
        timestamp: Instant,
        layer_index: usize,
        entropy: Option<f32>,
        activation_sparsity: Option<f32>,
        activation_magnitude_mean: Option<f32>,
        activation_magnitude_std: Option<f32>,
    },
    
    /// Consciousness state update
    ConsciousnessStateUpdate {
        timestamp: Instant,
        state_id: String,
        state_value: f32,
        confidence: f32,
    },
    
    /// Anomaly detected
    AnomalyDetected {
        timestamp: Instant,
        anomaly_type: String,
        severity: String,
        description: String,
        metrics: Vec<(String, f32)>,
    },
}

/// Telemetry bus for managing event flow
pub struct TelemetryBus {
    config: TelemetryConfig,
    sender: TelemetrySender,
    receiver: Option<mpsc::UnboundedReceiver<TelemetryEvent>>,
    dropped_events: Arc<std::sync::atomic::AtomicUsize>,
    is_running: Arc<std::sync::atomic::AtomicBool>,
}

/// Sender for telemetry events
#[derive(Clone)]
pub struct TelemetrySender {
    inner: mpsc::UnboundedSender<TelemetryEvent>,
    dropped_events: Arc<std::sync::atomic::AtomicUsize>,
}

impl TelemetryBus {
    /// Create a new telemetry bus
    pub async fn new(config: TelemetryConfig) -> Result<Self, String> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let dropped_events = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        
        let telemetry_sender = TelemetrySender {
            inner: sender,
            dropped_events: dropped_events.clone(),
        };
        
        Ok(TelemetryBus {
            config,
            sender: telemetry_sender,
            receiver: Some(receiver),
            dropped_events,
            is_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }
    
    /// Start the telemetry bus
    pub async fn start(&mut self) -> Result<(), String> {
        if self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("Telemetry bus is already running".to_string());
        }
        
        self.is_running.store(true, std::sync::atomic::Ordering::Relaxed);
        
        if let Some(receiver) = self.receiver.take() {
            let config = self.config.clone();
            let dropped_events = self.dropped_events.clone();
            let is_running = self.is_running.clone();
            
            tokio::spawn(async move {
                Self::process_events(receiver, config, dropped_events, is_running).await;
            });
        }
        
        Ok(())
    }
    
    /// Stop the telemetry bus
    pub async fn stop(&mut self) -> Result<(), String> {
        self.is_running.store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
    
    /// Get the telemetry sender
    pub fn sender(&self) -> TelemetrySender {
        self.sender.clone()
    }
    
    /// Get the number of dropped events
    pub fn dropped_events(&self) -> usize {
        self.dropped_events.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// Process telemetry events
    async fn process_events(
        mut receiver: mpsc::UnboundedReceiver<TelemetryEvent>,
        config: TelemetryConfig,
        dropped_events: Arc<std::sync::atomic::AtomicUsize>,
        is_running: Arc<std::sync::atomic::AtomicBool>,
    ) {
        let mut batch = Vec::with_capacity(config.batch_size);
        let mut interval = tokio::time::interval(Duration::from_millis(config.processing_interval_ms));
        
        while is_running.load(std::sync::atomic::Ordering::Relaxed) {
            interval.tick().await;
            
            // Collect events in batches
            while let Ok(event) = receiver.try_recv() {
                batch.push(event);
                
                if batch.len() >= config.batch_size {
                    break;
                }
            }
            
            if !batch.is_empty() {
                // Process the batch
                Self::process_batch(batch.clone()).await;
                batch.clear();
            }
            
            // Check for dropped events
            if config.enable_dropped_event_tracking {
                let dropped = dropped_events.load(std::sync::atomic::Ordering::Relaxed);
                if dropped > 0 {
                    tracing::warn!("Dropped {} telemetry events", dropped);
                }
            }
        }
    }
    
    /// Process a batch of events
    async fn process_batch(events: Vec<TelemetryEvent>) {
        for event in events {
            match event {
                TelemetryEvent::InferenceStart { request_id, timestamp, prompt_length } => {
                    tracing::debug!("Inference started: {} (prompt: {} tokens)", request_id, prompt_length);
                },
                TelemetryEvent::TokenGenerated { request_id, timestamp, token_index, token_length } => {
                    tracing::debug!("Token generated: {} (index: {}, length: {})", request_id, token_index, token_length);
                },
                TelemetryEvent::InferenceComplete { request_id, timestamp, total_tokens, error } => {
                    if let Some(err) = error {
                        tracing::warn!("Inference failed: {} - {}", request_id, err);
                    } else {
                        tracing::debug!("Inference completed: {} ({} tokens)", request_id, total_tokens);
                    }
                },
                TelemetryEvent::HardwareMetrics { timestamp, .. } => {
                    tracing::trace!("Hardware metrics collected at {:?}", timestamp);
                },
                TelemetryEvent::ModelMetrics { timestamp, layer_index, .. } => {
                    tracing::trace!("Model metrics collected for layer {} at {:?}", layer_index, timestamp);
                },
                TelemetryEvent::ConsciousnessStateUpdate { timestamp, state_id, state_value, confidence } => {
                    tracing::debug!("Consciousness state update: {} = {} (confidence: {})", state_id, state_value, confidence);
                },
                TelemetryEvent::AnomalyDetected { timestamp, anomaly_type, severity, description, .. } => {
                    tracing::warn!("Anomaly detected: {} - {} - {}", anomaly_type, severity, description);
                },
            }
        }
    }
}

impl TelemetrySender {
    /// Send a telemetry event
    pub async fn send(&self, event: TelemetryEvent) -> Result<(), String> {
        self.inner.send(event).map_err(|_| "Failed to send telemetry event".to_string())
    }
    
    /// Try to send a telemetry event without blocking
    pub fn try_send(&self, event: TelemetryEvent) -> Result<(), String> {
        self.inner.send(event).map_err(|_| {
            self.dropped_events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            "Failed to send telemetry event".to_string()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    
    #[tokio::test]
    async fn test_telemetry_bus_creation() {
        let config = TelemetryConfig::default();
        let bus = TelemetryBus::new(config).await;
        assert!(bus.is_ok());
    }
    
    #[tokio::test]
    async fn test_telemetry_event_sending() {
        let config = TelemetryConfig::default();
        let mut bus = TelemetryBus::new(config).await.unwrap();
        bus.start().await.unwrap();
        
        let sender = bus.sender();
        let request_id = Uuid::new_v4();
        
        let event = TelemetryEvent::InferenceStart {
            request_id,
            timestamp: Instant::now(),
            prompt_length: 100,
        };
        
        let result = sender.send(event).await;
        assert!(result.is_ok());
        
        bus.stop().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_telemetry_bus_start_stop() {
        let config = TelemetryConfig::default();
        let mut bus = TelemetryBus::new(config).await.unwrap();
        
        assert!(bus.start().await.is_ok());
        assert!(bus.stop().await.is_ok());
    }
}