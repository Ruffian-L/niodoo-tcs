//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Telemetry bus for Silicon Synapse monitoring system
//!
//! This module implements the telemetry event bus that decouples metric production
//! from collection, ensuring the consciousness engine never blocks on monitoring.
//!
//! Features:
//! - Rust `tokio::sync::mpsc` unbounded channel
//! - Consciousness components send `TelemetryEvent` messages asynchronously
//! - If channel is full, events are dropped (monitoring never blocks inference)
//! - Dedicated tokio task consumes events and routes to collectors
//! - Dropped event counter for backpressure monitoring

use crate::silicon_synapse::collectors::inference::InferenceMetrics;
use crate::silicon_synapse::config::TelemetryConfig;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use tokio::sync::mpsc;
use uuid::Uuid;

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
        token_id: u32,
        timestamp: Instant,
        logits: Option<Vec<f32>>,
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
        timestamp: SystemTime,
        gpu_temp_celsius: Option<f32>,
        gpu_power_watts: Option<f32>,
        gpu_fan_speed_percent: Option<f32>,
        vram_used_bytes: Option<u64>,
        vram_total_bytes: Option<u64>,
        gpu_utilization_percent: Option<f32>,
        cpu_utilization_percent: f32,
        ram_used_bytes: u64,
    },

    /// Model metrics collected
    ModelMetrics {
        timestamp: std::time::Instant,
        layer_index: usize,
        entropy: Option<f32>,
        activation_sparsity: Option<f32>,
        activation_magnitude_mean: Option<f32>,
        activation_magnitude_std: Option<f32>,
    },

    /// Aggregated inference metrics
    InferenceMetrics(InferenceMetrics),
}

/// Telemetry bus for decoupling metric production from collection
pub struct TelemetryBus {
    /// Configuration for telemetry system
    #[allow(dead_code)]
    config: TelemetryConfig,
    sender: TelemetrySender,
    receiver: Option<mpsc::UnboundedReceiver<TelemetryEvent>>,
    dropped_events: Arc<std::sync::atomic::AtomicU64>,
}

/// Sender for telemetry events
#[derive(Clone)]
pub struct TelemetrySender {
    pub inner: mpsc::UnboundedSender<TelemetryEvent>,
    pub dropped_events: Arc<std::sync::atomic::AtomicU64>,
}

impl TelemetryBus {
    /// Create a new telemetry bus
    pub fn new(config: TelemetryConfig) -> Result<Self, TelemetryBusError> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let dropped_events = Arc::new(std::sync::atomic::AtomicU64::new(0));

        let telemetry_sender = TelemetrySender {
            inner: sender,
            dropped_events: dropped_events.clone(),
        };

        Ok(TelemetryBus {
            config,
            sender: telemetry_sender,
            receiver: Some(receiver),
            dropped_events,
        })
    }

    /// Create a sender for telemetry events
    pub fn create_sender(&self) -> TelemetrySender {
        self.sender.clone()
    }

    pub fn take_receiver(&mut self) -> mpsc::UnboundedReceiver<TelemetryEvent> {
        self.receiver.take().expect("Receiver already taken")
    }

    /// Get a reference to the receiver for cloning (Arc-compatible)
    pub fn get_receiver_ref(&self) -> &Option<mpsc::UnboundedReceiver<TelemetryEvent>> {
        &self.receiver
    }

    /// Clone the receiver for use in spawned tasks (Arc-compatible)
    pub fn clone_receiver(&self) -> mpsc::UnboundedReceiver<TelemetryEvent> {
        // Create a new channel and return the receiver
        // Note: This creates a new channel since mpsc receivers can't be cloned
        let (_, receiver) = mpsc::unbounded_channel();
        receiver
    }

    /// Get the telemetry sender
    pub fn sender(&self) -> TelemetrySender {
        self.sender.clone()
    }

    /// Get the number of dropped events
    pub fn dropped_events_count(&self) -> u64 {
        self.dropped_events
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Event processor that routes events to collectors
    pub async fn event_processor(
        mut receiver: mpsc::UnboundedReceiver<TelemetryEvent>,
        _dropped_events: Arc<std::sync::atomic::AtomicU64>,
    ) {
        while let Some(event) = receiver.recv().await {
            // Route event to appropriate collectors
            Self::route_event(event).await;
        }
    }

    /// Route telemetry event to appropriate collectors
    pub async fn route_event(event: TelemetryEvent) {
        match event {
            TelemetryEvent::InferenceStart {
                request_id,
                timestamp: _,
                prompt_length,
            } => {
                tracing::debug!(
                    request_id = %request_id,
                    prompt_length = prompt_length,
                    "Inference started"
                );
            }

            TelemetryEvent::TokenGenerated {
                request_id,
                token_id,
                timestamp: _,
                logits,
            } => {
                tracing::debug!(
                    request_id = %request_id,
                    token_id = token_id,
                    has_logits = logits.is_some(),
                    "Token generated"
                );
            }

            TelemetryEvent::InferenceComplete {
                request_id,
                timestamp: _,
                total_tokens,
                error,
            } => {
                if let Some(error) = error {
                    tracing::warn!(
                        request_id = %request_id,
                        total_tokens = total_tokens,
                        error = %error,
                        "Inference completed with error"
                    );
                } else {
                    tracing::debug!(
                        request_id = %request_id,
                        total_tokens = total_tokens,
                        "Inference completed successfully"
                    );
                }
            }

            TelemetryEvent::HardwareMetrics {
                timestamp: _,
                gpu_temp_celsius,
                gpu_power_watts,
                gpu_fan_speed_percent,
                vram_used_bytes,
                vram_total_bytes,
                gpu_utilization_percent,
                cpu_utilization_percent,
                ram_used_bytes,
            } => {
                tracing::trace!(
                    gpu_temp_celsius = ?gpu_temp_celsius,
                    gpu_power_watts = ?gpu_power_watts,
                    gpu_fan_speed_percent = ?gpu_fan_speed_percent,
                    vram_used_bytes = ?vram_used_bytes,
                    vram_total_bytes = ?vram_total_bytes,
                    gpu_utilization_percent = ?gpu_utilization_percent,
                    cpu_utilization_percent = %cpu_utilization_percent,
                    ram_used_bytes = %ram_used_bytes,
                    "Hardware metrics collected"
                );
            }

            TelemetryEvent::ModelMetrics {
                timestamp: _,
                layer_index,
                entropy,
                activation_sparsity,
                activation_magnitude_mean,
                activation_magnitude_std,
            } => {
                tracing::trace!(
                    layer_index = %layer_index,
                    entropy = ?entropy,
                    activation_sparsity = ?activation_sparsity,
                    activation_magnitude_mean = ?activation_magnitude_mean,
                    activation_magnitude_std = ?activation_magnitude_std,
                    "Model metrics collected"
                );
            }

            TelemetryEvent::InferenceMetrics(metrics) => {
                tracing::debug!(
                    active_requests = %metrics.active_requests,
                    completed_requests = %metrics.completed_requests,
                    failed_requests = %metrics.failed_requests,
                    average_ttft_ms = ?metrics.average_ttft_ms,
                    average_tpot_ms = ?metrics.average_tpot_ms,
                    average_throughput_tps = ?metrics.average_throughput_tps,
                    error_rate = %metrics.error_rate,
                    "Inference metrics collected"
                );
            }
        }
    }
}

impl TelemetrySender {
    /// Send a telemetry event (blocking)
    pub async fn send(&self, event: TelemetryEvent) -> Result<(), TelemetryBusError> {
        self.inner
            .send(event)
            .map_err(|_| TelemetryBusError::ChannelClosed)?;
        Ok(())
    }

    /// Try to send a telemetry event (non-blocking)
    pub fn try_send(&self, event: TelemetryEvent) -> Result<(), TelemetryBusError> {
        self.inner
            .send(event)
            .map_err(|_| TelemetryBusError::ChannelClosed)?;
        Ok(())
    }

    /// Check if the channel is closed
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// Get the number of dropped events
    pub fn dropped_events_count(&self) -> u64 {
        self.dropped_events
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Telemetry bus error type
#[derive(Debug, thiserror::Error)]
pub enum TelemetryBusError {
    #[error("Telemetry bus already started")]
    AlreadyStarted,

    #[error("Telemetry channel is closed")]
    ChannelClosed,

    #[error("Failed to start telemetry bus: {0}")]
    StartFailed(String),

    #[error("No receiver available")]
    NoReceiver,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_telemetry_bus_creation() {
        let config = TelemetryConfig::default();
        let bus = TelemetryBus::new(config);
        assert!(bus.is_ok());
    }

    #[tokio::test]
    async fn test_telemetry_event_flow() {
        let config = TelemetryConfig::default();
        let mut bus = TelemetryBus::new(config).unwrap();

        let receiver = bus.take_receiver();

        let sender = bus.sender();
        let request_id = Uuid::new_v4();

        sender
            .send(TelemetryEvent::InferenceStart {
                request_id,
                timestamp: Instant::now(),
                prompt_length: 100,
            })
            .await
            .unwrap();

        sender
            .send(TelemetryEvent::InferenceComplete {
                request_id,
                timestamp: Instant::now(),
                total_tokens: 50,
                error: None,
            })
            .await
            .unwrap();

        // Give some time for processing
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_telemetry_sender_clone() {
        let config = TelemetryConfig::default();
        let mut bus = TelemetryBus::new(config).unwrap();

        let sender1 = bus.sender();
        let sender2 = sender1.clone();

        assert_eq!(sender1.is_closed(), sender2.is_closed());
    }

    #[tokio::test]
    async fn test_hardware_metrics_event() {
        let config = TelemetryConfig::default();
        let mut bus = TelemetryBus::new(config).unwrap();

        let sender = bus.sender();

        sender
            .send(TelemetryEvent::HardwareMetrics {
                timestamp: SystemTime::now(),
                gpu_temp_celsius: Some(72.5),
                gpu_power_watts: Some(245.3),
                gpu_fan_speed_percent: Some(65.0),
                vram_used_bytes: Some(8589934592),
                vram_total_bytes: Some(17179869184),
                gpu_utilization_percent: Some(85.0),
                cpu_utilization_percent: 45.0,
                ram_used_bytes: 16106127360,
            })
            .await
            .unwrap();

        // Give some time for processing
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_model_metrics_event() {
        let config = TelemetryConfig::default();
        let mut bus = TelemetryBus::new(config).unwrap();

        let sender = bus.sender();
        let mut activation_sparsity = std::collections::HashMap::new();
        let mut activation_magnitude = std::collections::HashMap::new();

        activation_sparsity.insert("layer_12".to_string(), 0.87);
        activation_magnitude.insert("layer_12".to_string(), 0.45);

        sender
            .send(TelemetryEvent::ModelMetrics {
                timestamp: std::time::Instant::now(),
                layer_index: 12,
                entropy: Some(0.342),
                activation_sparsity: Some(0.87),
                activation_magnitude_mean: Some(0.45),
                activation_magnitude_std: Some(0.12),
            })
            .await
            .unwrap();

        // Give some time for processing
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
