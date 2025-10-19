//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Inference performance monitoring collector for Silicon Synapse
//!
//! This module implements inference performance monitoring including TTFT, TPOT,
//! throughput tracking, and error rate monitoring.

use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn};
use uuid::Uuid;

use crate::silicon_synapse::config::InferenceConfig;
use crate::silicon_synapse::telemetry_bus::{TelemetryEvent, TelemetrySender};
use crate::silicon_synapse::SiliconSynapseError;

/// Inference performance collector
pub struct InferenceCollector {
    config: InferenceConfig,
    telemetry_sender: TelemetrySender,
    active_requests: Arc<RwLock<HashMap<Uuid, InferenceRequest>>>,
    is_running: Arc<RwLock<bool>>,
    // Persistent counters for completed/failed requests
    completed_count: Arc<AtomicUsize>,
    failed_count: Arc<AtomicUsize>,
    // History buffers for TTFT/TPOT metrics (keep last 1000 entries)
    ttft_history: Arc<RwLock<VecDeque<f64>>>,
    tpot_history: Arc<RwLock<VecDeque<f64>>>,
}

/// Clone of InferenceCollector for background tasks
struct InferenceCollectorClone {
    active_requests: Arc<RwLock<HashMap<Uuid, InferenceRequest>>>,
    config: InferenceConfig,
}

impl InferenceCollectorClone {
    async fn cleanup_old_requests(&self) -> Result<(), SiliconSynapseError> {
        let mut active_requests = self.active_requests.write().await;
        let now = Instant::now();
        let timeout_duration = Duration::from_secs(self.config.request_timeout_seconds);

        let mut to_remove = Vec::new();

        for (request_id, request) in active_requests.iter() {
            if now.duration_since(request.start_time) > timeout_duration {
                to_remove.push(*request_id);
            }
        }

        for request_id in to_remove {
            active_requests.remove(&request_id);
            warn!("Removed timed out request: {}", request_id);
        }

        Ok(())
    }
}

/// Active inference request tracking
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub request_id: Uuid,
    pub start_time: Instant,
    pub prompt_length: usize,
    pub first_token_time: Option<Instant>,
    pub tokens_generated: usize,
    pub last_token_time: Option<Instant>,
    pub error: Option<String>,
}

/// Inference metrics data structure
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    pub timestamp: Instant,
    pub active_requests: usize,
    pub completed_requests: usize,
    pub failed_requests: usize,
    pub average_ttft_ms: Option<f64>,
    pub average_tpot_ms: Option<f64>,
    pub average_throughput_tps: Option<f64>,
    pub error_rate: f64,
}

impl InferenceCollector {
    /// Create a new inference collector
    pub fn new(
        config: InferenceConfig,
        telemetry_sender: TelemetrySender,
    ) -> Result<Self, SiliconSynapseError> {
        Ok(Self {
            config,
            telemetry_sender,
            active_requests: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(RwLock::new(false)),
            completed_count: Arc::new(AtomicUsize::new(0)),
            failed_count: Arc::new(AtomicUsize::new(0)),
            ttft_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            tpot_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
        })
    }

    /// Start the inference collector
    pub async fn start(&mut self) -> Result<(), SiliconSynapseError> {
        if *self.is_running.read().await {
            return Err(SiliconSynapseError::Config(
                "Inference collector is already running".to_string(),
            ));
        }

        info!("Starting inference collector");
        *self.is_running.write().await = true;

        // Start background cleanup task
        let collector_clone = self.clone_for_cleanup();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                if let Err(e) = collector_clone.cleanup_old_requests().await {
                    warn!("Error during cleanup: {}", e);
                }
            }
        });

        let active_requests_clone = Arc::clone(&self.active_requests);
        let completed_count_clone = Arc::clone(&self.completed_count);
        let failed_count_clone = Arc::clone(&self.failed_count);
        let ttft_history_clone = self.ttft_history.clone();
        let tpot_history_clone = self.tpot_history.clone();
        let telemetry_sender_clone = self.telemetry_sender.clone();
        let config_clone = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                interval.tick().await;

                // Create temporary collector for metrics calculation (without is_running check)
                let temp_collector = InferenceCollector {
                    config: config_clone.clone(),
                    telemetry_sender: telemetry_sender_clone.clone(),
                    active_requests: active_requests_clone.clone(),
                    is_running: Arc::new(RwLock::new(true)), // Force true for metrics
                    completed_count: completed_count_clone.clone(),
                    failed_count: failed_count_clone.clone(),
                    ttft_history: ttft_history_clone.clone(),
                    tpot_history: tpot_history_clone.clone(),
                };

                if let Ok(metrics) = temp_collector.get_metrics().await {
                    let event = TelemetryEvent::InferenceMetrics(metrics.clone());
                    if let Err(e) = telemetry_sender_clone.try_send(event) {
                        warn!("Failed to send inference metrics: {}", e);
                    }
                } else {
                    warn!("Failed to get inference metrics");
                }
            }
        });

        Ok(())
    }

    /// Create a clone of the collector for background tasks
    fn clone_for_cleanup(&self) -> InferenceCollectorClone {
        InferenceCollectorClone {
            active_requests: self.active_requests.clone(),
            config: self.config.clone(),
        }
    }

    /// Stop the inference collector
    pub async fn stop(&mut self) -> Result<(), SiliconSynapseError> {
        if !*self.is_running.read().await {
            return Ok(());
        }

        info!("Stopping inference collector");
        *self.is_running.write().await = false;

        Ok(())
    }

    /// Track inference start
    pub async fn track_inference_start(
        &self,
        request_id: Uuid,
        prompt_length: usize,
    ) -> Result<(), SiliconSynapseError> {
        if !*self.is_running.read().await {
            return Err(SiliconSynapseError::Config(
                "Inference collector is not running".to_string(),
            ));
        }

        let request = InferenceRequest {
            request_id,
            start_time: Instant::now(),
            prompt_length,
            first_token_time: None,
            tokens_generated: 0,
            last_token_time: None,
            error: None,
        };

        {
            let mut active_requests = self.active_requests.write().await;
            active_requests.insert(request_id, request);

            // Check if we've exceeded the maximum concurrent requests
            if active_requests.len() > self.config.max_concurrent_requests {
                warn!(
                    "Exceeded maximum concurrent requests: {}",
                    active_requests.len()
                );
            }
        }

        // Send telemetry event
        let event = TelemetryEvent::InferenceStart {
            request_id,
            timestamp: Instant::now(),
            prompt_length,
        };

        if let Err(e) = self.telemetry_sender.try_send(event) {
            warn!("Failed to send inference start event: {}", e);
        }

        Ok(())
    }

    /// Track token generation
    pub async fn track_token_generated(
        &self,
        request_id: Uuid,
        _token_length: usize,
    ) -> Result<(), SiliconSynapseError> {
        if !*self.is_running.read().await {
            return Err(SiliconSynapseError::Config(
                "Inference collector is not running".to_string(),
            ));
        }

        let mut active_requests = self.active_requests.write().await;

        if let Some(request) = active_requests.get_mut(&request_id) {
            let now = Instant::now();

            // Set first token time if this is the first token
            if request.first_token_time.is_none() {
                request.first_token_time = Some(now);
            }

            request.tokens_generated += 1;
            request.last_token_time = Some(now);

            // Send telemetry event
            let event = TelemetryEvent::TokenGenerated {
                request_id,
                timestamp: now,
                token_id: (request.tokens_generated - 1) as u32,
                logits: None, // We don't have logits in this context
            };

            if let Err(e) = self.telemetry_sender.try_send(event) {
                warn!("Failed to send token generated event: {}", e);
            }
        } else {
            warn!("Received token for unknown request: {}", request_id);
        }

        Ok(())
    }

    /// Track inference completion
    pub async fn track_inference_complete(
        &self,
        request_id: Uuid,
        error: Option<String>,
    ) -> Result<(), SiliconSynapseError> {
        if !*self.is_running.read().await {
            return Err(SiliconSynapseError::Config(
                "Inference collector is not running".to_string(),
            ));
        }

        let mut active_requests = self.active_requests.write().await;

        if let Some(mut request) = active_requests.remove(&request_id) {
            request.error = error.clone();

            // Increment counters atomically
            if error.is_some() {
                self.failed_count.fetch_add(1, Ordering::Relaxed);
            } else {
                self.completed_count.fetch_add(1, Ordering::Relaxed);

                // Store TTFT/TPOT in history buffers before removing from HashMap
                if let Some(first_token_time) = request.first_token_time {
                    let ttft_ms = first_token_time
                        .duration_since(request.start_time)
                        .as_millis() as f64;

                    // Add to TTFT history
                    {
                        let mut ttft_history = self.ttft_history.write().await;
                        ttft_history.push_back(ttft_ms);
                        // Keep history bounded (max 1000 entries)
                        if ttft_history.len() > 1000 {
                            ttft_history.pop_front();
                        }
                    }

                    // Calculate TPOT if we have multiple tokens
                    if request.tokens_generated > 1 {
                        if let Some(last_token_time) = request.last_token_time {
                            let total_time_ms =
                                last_token_time.duration_since(first_token_time).as_millis() as f64;
                            let tpot_ms = total_time_ms / (request.tokens_generated - 1) as f64;

                            // Add to TPOT history
                            let mut tpot_history = self.tpot_history.write().await;
                            tpot_history.push_back(tpot_ms);
                            // Keep history bounded (max 1000 entries)
                            if tpot_history.len() > 1000 {
                                tpot_history.pop_front();
                            }
                        }
                    }
                }
            }

            // Send telemetry event
            let event = TelemetryEvent::InferenceComplete {
                request_id,
                timestamp: Instant::now(),
                total_tokens: request.tokens_generated,
                error,
            };

            if let Err(e) = self.telemetry_sender.try_send(event) {
                warn!("Failed to send inference complete event: {}", e);
            }
        } else {
            warn!("Received completion for unknown request: {}", request_id);
        }

        Ok(())
    }

    /// Get current inference metrics
    pub async fn get_metrics(&self) -> Result<InferenceMetrics, SiliconSynapseError> {
        let active_requests = self.active_requests.read().await;
        let timestamp = Instant::now();

        // Get counts from persistent atomic counters
        let completed_count = self.completed_count.load(Ordering::Relaxed);
        let failed_count = self.failed_count.load(Ordering::Relaxed);
        let active_count = active_requests.len();

        // Calculate averages from history buffers
        let (average_ttft_ms, average_tpot_ms, average_throughput_tps) = {
            // Read TTFT history
            let ttft_history = self.ttft_history.read().await;
            let ttft_avg = if !ttft_history.is_empty() {
                Some(ttft_history.iter().sum::<f64>() / ttft_history.len() as f64)
            } else {
                None
            };

            // Read TPOT history
            let tpot_history = self.tpot_history.read().await;
            let tpot_avg = if !tpot_history.is_empty() {
                Some(tpot_history.iter().sum::<f64>() / tpot_history.len() as f64)
            } else {
                None
            };

            // Calculate throughput from active requests only (since completed requests are removed)
            let mut throughput_sum = 0.0;
            let mut throughput_count = 0;

            for request in active_requests.values() {
                if request.tokens_generated > 0 {
                    if let Some(last_token_time) = request.last_token_time {
                        let total_time_s = last_token_time
                            .duration_since(request.start_time)
                            .as_secs_f64();
                        if total_time_s > 0.0 {
                            let throughput_tps = request.tokens_generated as f64 / total_time_s;
                            throughput_sum += throughput_tps;
                            throughput_count += 1;
                        }
                    }
                }
            }

            let throughput_avg = if throughput_count > 0 {
                Some(throughput_sum / throughput_count as f64)
            } else {
                None
            };

            (ttft_avg, tpot_avg, throughput_avg)
        };

        let total_requests = completed_count + failed_count;
        let error_rate = if total_requests > 0 {
            failed_count as f64 / total_requests as f64
        } else {
            0.0
        };

        Ok(InferenceMetrics {
            timestamp,
            active_requests: active_count,
            completed_requests: completed_count,
            failed_requests: failed_count,
            average_ttft_ms,
            average_tpot_ms,
            average_throughput_tps,
            error_rate,
        })
    }

    /// Clean up old requests (called periodically)
    pub async fn cleanup_old_requests(&self) -> Result<(), SiliconSynapseError> {
        let mut active_requests = self.active_requests.write().await;
        let now = Instant::now();
        let timeout_duration = Duration::from_secs(self.config.request_timeout_seconds);

        let mut to_remove = Vec::new();

        for (request_id, request) in active_requests.iter() {
            if now.duration_since(request.start_time) > timeout_duration {
                to_remove.push(*request_id);
            }
        }

        for request_id in to_remove {
            active_requests.remove(&request_id);
            warn!("Removed timed out request: {}", request_id);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silicon_synapse::config::InferenceConfig;
    use crate::silicon_synapse::telemetry_bus::TelemetryBus;

    #[tokio::test]
    async fn test_inference_collector_creation() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let collector = InferenceCollector::new(config, telemetry_sender);
        assert!(collector.is_ok());
    }

    #[tokio::test]
    async fn test_inference_collector_start_stop() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");

        assert!(collector.start().await.is_ok());
        assert!(collector.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_inference_tracking() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");
        collector
            .start()
            .await
            .expect("Failed to start collector in test");

        let request_id = Uuid::new_v4();

        // Track inference start
        assert!(collector
            .track_inference_start(request_id, 100)
            .await
            .is_ok());

        // Track token generation
        assert!(collector.track_token_generated(request_id, 5).await.is_ok());
        assert!(collector.track_token_generated(request_id, 3).await.is_ok());

        // Track completion
        assert!(collector
            .track_inference_complete(request_id, None)
            .await
            .is_ok());

        collector
            .stop()
            .await
            .expect("Failed to stop collector in test");
    }

    #[tokio::test]
    async fn test_ttft_calculation() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");
        collector
            .start()
            .await
            .expect("Failed to start collector in test");

        let request_id = Uuid::new_v4();

        // Start inference
        assert!(collector
            .track_inference_start(request_id, 50)
            .await
            .is_ok());

        // Use synthetic timing - simulate 150ms delay before first token
        {
            let mut active_requests = collector.active_requests.write().await;
            if let Some(request) = active_requests.get_mut(&request_id) {
                // Simulate first token arriving after 150ms
                use std::time::{Duration, Instant};
                let synthetic_first_token_time = request.start_time + Duration::from_millis(150);
                request.first_token_time = Some(synthetic_first_token_time);
                request.tokens_generated = 1;
                request.last_token_time = Some(synthetic_first_token_time);
            }
        }

        // Complete inference
        assert!(collector
            .track_inference_complete(request_id, None)
            .await
            .is_ok());

        // Check metrics
        let metrics = collector
            .get_metrics()
            .await
            .expect("Failed to get metrics in test");
        assert!(metrics.average_ttft_ms.is_some());
        assert_eq!(
            metrics.average_ttft_ms.expect("TTFT should be present"),
            150.0
        ); // Should be exactly 150ms

        collector
            .stop()
            .await
            .expect("Failed to stop collector in test");
    }

    #[tokio::test]
    async fn test_tpot_calculation() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");
        collector
            .start()
            .await
            .expect("Failed to start collector in test");

        let request_id = Uuid::new_v4();

        // Start inference
        assert!(collector
            .track_inference_start(request_id, 50)
            .await
            .is_ok());

        // Use synthetic timing - simulate tokens with controlled timing
        {
            let mut active_requests = collector.active_requests.write().await;
            if let Some(request) = active_requests.get_mut(&request_id) {
                use std::time::{Duration, Instant};
                // First token arrives after 100ms
                let first_token_time = request.start_time + Duration::from_millis(100);
                request.first_token_time = Some(first_token_time);
                request.tokens_generated = 1;
                request.last_token_time = Some(first_token_time);

                // Second token arrives after another 75ms (total 175ms from start)
                let second_token_time = request.start_time + Duration::from_millis(175);
                request.tokens_generated = 2;
                request.last_token_time = Some(second_token_time);
            }
        }

        // Complete inference
        assert!(collector
            .track_inference_complete(request_id, None)
            .await
            .is_ok());

        // Check metrics - TPOT should be (175ms - 100ms) / 1 token = 75ms
        let metrics = collector
            .get_metrics()
            .await
            .expect("Failed to get metrics in test");
        assert!(metrics.average_tpot_ms.is_some());
        assert_eq!(
            metrics.average_tpot_ms.expect("TPOT should be present"),
            75.0
        ); // Should be exactly 75ms

        collector
            .stop()
            .await
            .expect("Failed to stop collector in test");
    }

    #[tokio::test]
    async fn test_throughput_calculation() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");
        collector
            .start()
            .await
            .expect("Failed to start collector in test");

        let request_id = Uuid::new_v4();

        // Start inference
        assert!(collector
            .track_inference_start(request_id, 50)
            .await
            .is_ok());

        // Use synthetic timing - simulate 5 tokens over 200ms total time
        {
            let mut active_requests = collector.active_requests.write().await;
            if let Some(request) = active_requests.get_mut(&request_id) {
                use std::time::{Duration, Instant};
                // First token arrives after 50ms
                let first_token_time = request.start_time + Duration::from_millis(50);
                request.first_token_time = Some(first_token_time);
                request.tokens_generated = 1;
                request.last_token_time = Some(first_token_time);

                // Last token arrives after 200ms total (5 tokens)
                let last_token_time = request.start_time + Duration::from_millis(200);
                request.tokens_generated = 5;
                request.last_token_time = Some(last_token_time);
            }
        }

        // Complete inference
        assert!(collector
            .track_inference_complete(request_id, None)
            .await
            .is_ok());

        // Check metrics - throughput should be 5 tokens / 200ms = 25 tokens/second
        let metrics = collector
            .get_metrics()
            .await
            .expect("Failed to get metrics in test");
        assert!(metrics.average_throughput_tps.is_some());
        assert_eq!(
            metrics
                .average_throughput_tps
                .expect("Throughput should be present"),
            25.0
        ); // Should be exactly 25 TPS

        collector
            .stop()
            .await
            .expect("Failed to stop collector in test");
    }

    #[tokio::test]
    async fn test_error_rate_calculation() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");
        collector
            .start()
            .await
            .expect("Failed to start collector in test");

        // Successful request
        let success_id = Uuid::new_v4();
        assert!(collector
            .track_inference_start(success_id, 50)
            .await
            .is_ok());
        assert!(collector.track_token_generated(success_id, 5).await.is_ok());
        assert!(collector
            .track_inference_complete(success_id, None)
            .await
            .is_ok());

        // Failed request
        let fail_id = Uuid::new_v4();
        assert!(collector.track_inference_start(fail_id, 50).await.is_ok());
        assert!(collector
            .track_inference_complete(fail_id, Some("Test error".to_string()))
            .await
            .is_ok());

        // Check metrics
        let metrics = collector
            .get_metrics()
            .await
            .expect("Failed to get metrics in test");
        assert_eq!(metrics.error_rate, 0.5); // 1 failed out of 2 total

        collector
            .stop()
            .await
            .expect("Failed to stop collector in test");
    }

    #[tokio::test]
    async fn test_timeout_cleanup() {
        let mut config = InferenceConfig::default();
        config.request_timeout_seconds = 1; // 1 second timeout

        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");
        collector
            .start()
            .await
            .expect("Failed to start collector in test");

        let request_id = Uuid::new_v4();

        // Start inference but don't complete it
        assert!(collector
            .track_inference_start(request_id, 50)
            .await
            .is_ok());

        // Use synthetic timing - simulate that the request started 2 seconds ago (beyond timeout)
        {
            let mut active_requests = collector.active_requests.write().await;
            if let Some(request) = active_requests.get_mut(&request_id) {
                use std::time::{Duration, Instant};
                // Simulate the request started 2 seconds ago (beyond 1 second timeout)
                let synthetic_start_time = Instant::now() - Duration::from_secs(2);
                request.start_time = synthetic_start_time;
            }
        }

        // Cleanup should remove the timed out request
        assert!(collector.cleanup_old_requests().await.is_ok());

        // Check that request is no longer active
        let metrics = collector
            .get_metrics()
            .await
            .expect("Failed to get metrics in test");
        assert_eq!(metrics.active_requests, 0);

        collector
            .stop()
            .await
            .expect("Failed to stop collector in test");
    }

    #[tokio::test]
    async fn test_concurrent_requests() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");
        collector
            .start()
            .await
            .expect("Failed to start collector in test");

        // Start multiple concurrent requests with synthetic timing
        for _i in 0..5 {
            let request_id = Uuid::new_v4();

            // Start inference
            assert!(collector
                .track_inference_start(request_id, 50)
                .await
                .is_ok());

            // Use synthetic timing for each request
            {
                let mut active_requests = collector.active_requests.write().await;
                if let Some(request) = active_requests.get_mut(&request_id) {
                    use std::time::{Duration, Instant};
                    // Simulate tokens arriving at controlled intervals
                    let token_times = [
                        request.start_time + Duration::from_millis(50), // First token
                        request.start_time + Duration::from_millis(100), // Second token
                        request.start_time + Duration::from_millis(150), // Third token
                    ];

                    request.first_token_time = Some(token_times[0]);
                    request.tokens_generated = 3;
                    request.last_token_time = Some(token_times[2]);
                }
            }

            // Complete inference
            assert!(collector
                .track_inference_complete(request_id, None)
                .await
                .is_ok());
        }

        // Check metrics
        let metrics = collector
            .get_metrics()
            .await
            .expect("Failed to get metrics in test");
        assert_eq!(metrics.active_requests, 0); // All should be completed
        assert_eq!(metrics.completed_requests, 5);

        collector
            .stop()
            .await
            .expect("Failed to stop collector in test");
    }

    #[tokio::test]
    async fn test_zero_tokens_request() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");
        collector
            .start()
            .await
            .expect("Failed to start collector in test");

        let request_id = Uuid::new_v4();

        // Start inference
        assert!(collector
            .track_inference_start(request_id, 50)
            .await
            .is_ok());

        // Complete immediately without any tokens (edge case)
        assert!(collector
            .track_inference_complete(request_id, None)
            .await
            .is_ok());

        // Check metrics - should handle zero tokens gracefully
        let metrics = collector
            .get_metrics()
            .await
            .expect("Failed to get metrics in test");
        assert_eq!(metrics.completed_requests, 1);
        assert_eq!(metrics.active_requests, 0);
        // TTFT and TPOT should be None since no tokens were generated
        assert!(metrics.average_ttft_ms.is_none());
        assert!(metrics.average_tpot_ms.is_none());

        collector
            .stop()
            .await
            .expect("Failed to stop collector in test");
    }

    #[tokio::test]
    async fn test_persistent_counters() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");
        collector
            .start()
            .await
            .expect("Failed to start collector in test");

        // Complete multiple requests
        for i in 0..3 {
            let request_id = Uuid::new_v4();
            assert!(collector
                .track_inference_start(request_id, 50)
                .await
                .is_ok());

            // Use synthetic timing
            {
                let mut active_requests = collector.active_requests.write().await;
                if let Some(request) = active_requests.get_mut(&request_id) {
                    use std::time::{Duration, Instant};
                    let token_time = request.start_time + Duration::from_millis(100 + i * 50);
                    request.first_token_time = Some(token_time);
                    request.tokens_generated = 2;
                    request.last_token_time = Some(token_time + Duration::from_millis(50));
                }
            }

            assert!(collector
                .track_inference_complete(request_id, None)
                .await
                .is_ok());

            // Verify counters are persistent after each completion
            let metrics = collector
                .get_metrics()
                .await
                .expect("Failed to get metrics in test");
            assert_eq!(metrics.completed_requests, (i + 1) as usize);
            assert_eq!(metrics.active_requests, 0); // Should be 0 since completed requests are removed
        }

        // Complete one failed request
        let fail_id = Uuid::new_v4();
        assert!(collector.track_inference_start(fail_id, 50).await.is_ok());
        assert!(collector
            .track_inference_complete(fail_id, Some("Test error".to_string()))
            .await
            .is_ok());

        // Verify failed counter also persists
        let metrics = collector
            .get_metrics()
            .await
            .expect("Failed to get metrics in test");
        assert_eq!(metrics.completed_requests, 3);
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.error_rate, 1.0 / 4.0); // 1 failed out of 4 total

        collector
            .stop()
            .await
            .expect("Failed to stop collector in test");
    }

    #[tokio::test]
    async fn test_history_buffer_bounds() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");
        collector
            .start()
            .await
            .expect("Failed to start collector in test");

        // Add more than 1000 entries to test bounds
        for _i in 0..1010 {
            let request_id = Uuid::new_v4();
            assert!(collector
                .track_inference_start(request_id, 50)
                .await
                .is_ok());

            // Use synthetic timing
            {
                let mut active_requests = collector.active_requests.write().await;
                if let Some(request) = active_requests.get_mut(&request_id) {
                    use std::time::{Duration, Instant};
                    let token_time = request.start_time + Duration::from_millis(100);
                    request.first_token_time = Some(token_time);
                    request.tokens_generated = 2;
                    request.last_token_time = Some(token_time + Duration::from_millis(50));
                }
            }

            assert!(collector
                .track_inference_complete(request_id, None)
                .await
                .is_ok());
        }

        // Check that history buffers are bounded to 1000 entries
        {
            let ttft_history = collector.ttft_history.read().await;
            let tpot_history = collector.tpot_history.read().await;

            // TTFT history should have 1000 entries (bounded)
            assert_eq!(ttft_history.len(), 1000);

            // TPOT history should have fewer entries since not all requests have multiple tokens
            assert!(tpot_history.len() <= 1000);
        } // Drop read locks before getting metrics and stopping

        // Verify metrics still work with bounded history
        let metrics = collector
            .get_metrics()
            .await
            .expect("Failed to get metrics in test");
        assert_eq!(metrics.completed_requests, 1010);
        assert!(metrics.average_ttft_ms.is_some()); // Should have average from last 1000 TTFT values

        collector
            .stop()
            .await
            .expect("Failed to stop collector in test");
    }

    #[tokio::test]
    async fn test_single_token_request() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");
        collector
            .start()
            .await
            .expect("Failed to start collector in test");

        let request_id = Uuid::new_v4();

        // Start inference
        assert!(collector
            .track_inference_start(request_id, 50)
            .await
            .is_ok());

        // Use synthetic timing for single token
        {
            let mut active_requests = collector.active_requests.write().await;
            if let Some(request) = active_requests.get_mut(&request_id) {
                use std::time::{Duration, Instant};
                let token_time = request.start_time + Duration::from_millis(100);
                request.first_token_time = Some(token_time);
                request.tokens_generated = 1;
                request.last_token_time = Some(token_time);
            }
        }

        // Complete inference
        assert!(collector
            .track_inference_complete(request_id, None)
            .await
            .is_ok());

        // Check metrics - single token should have TTFT but no TPOT
        let metrics = collector
            .get_metrics()
            .await
            .expect("Failed to get metrics in test");
        assert_eq!(metrics.completed_requests, 1);
        assert_eq!(
            metrics.average_ttft_ms.expect("TTFT should be present"),
            100.0
        );
        assert!(metrics.average_tpot_ms.is_none()); // No TPOT for single token

        collector
            .stop()
            .await
            .expect("Failed to stop collector in test");
    }

    #[tokio::test]
    async fn test_rapid_token_generation() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");
        collector
            .start()
            .await
            .expect("Failed to start collector in test");

        let request_id = Uuid::new_v4();

        // Start inference
        assert!(collector
            .track_inference_start(request_id, 50)
            .await
            .is_ok());

        // Use synthetic timing - simulate very rapid token generation (10 tokens in 100ms)
        {
            let mut active_requests = collector.active_requests.write().await;
            if let Some(request) = active_requests.get_mut(&request_id) {
                use std::time::{Duration, Instant};
                let first_token_time = request.start_time + Duration::from_millis(10);
                let last_token_time = request.start_time + Duration::from_millis(100);

                request.first_token_time = Some(first_token_time);
                request.tokens_generated = 10;
                request.last_token_time = Some(last_token_time);
            }
        }

        // Complete inference
        assert!(collector
            .track_inference_complete(request_id, None)
            .await
            .is_ok());

        // Check metrics - should calculate correct TPOT and throughput
        let metrics = collector
            .get_metrics()
            .await
            .expect("Failed to get metrics in test");
        assert_eq!(metrics.completed_requests, 1);
        assert_eq!(
            metrics.average_ttft_ms.expect("TTFT should be present"),
            10.0
        );
        // TPOT should be (100ms - 10ms) / 9 tokens = 10ms per token
        assert_eq!(
            metrics.average_tpot_ms.expect("TPOT should be present"),
            10.0
        );
        // Throughput should be 10 tokens / 100ms = 100 tokens/second
        assert_eq!(
            metrics
                .average_throughput_tps
                .expect("Throughput should be present"),
            100.0
        );

        collector
            .stop()
            .await
            .expect("Failed to stop collector in test");
    }

    #[tokio::test]
    async fn test_mixed_success_failure() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus =
            TelemetryBus::new(telemetry_config).expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");
        collector
            .start()
            .await
            .expect("Failed to start collector in test");

        // Successful request
        let success_id = Uuid::new_v4();
        assert!(collector
            .track_inference_start(success_id, 50)
            .await
            .is_ok());
        {
            let mut active_requests = collector.active_requests.write().await;
            if let Some(request) = active_requests.get_mut(&success_id) {
                use std::time::{Duration, Instant};
                let token_time = request.start_time + Duration::from_millis(100);
                request.first_token_time = Some(token_time);
                request.tokens_generated = 3;
                request.last_token_time = Some(token_time + Duration::from_millis(50));
            }
        }
        assert!(collector
            .track_inference_complete(success_id, None)
            .await
            .is_ok());

        // Failed request
        let fail_id = Uuid::new_v4();
        assert!(collector.track_inference_start(fail_id, 50).await.is_ok());
        assert!(collector
            .track_inference_complete(fail_id, Some("Test error".to_string()))
            .await
            .is_ok());

        // Request with no tokens but completed successfully
        let empty_id = Uuid::new_v4();
        assert!(collector.track_inference_start(empty_id, 50).await.is_ok());
        assert!(collector
            .track_inference_complete(empty_id, None)
            .await
            .is_ok());

        // Check metrics
        let metrics = collector
            .get_metrics()
            .await
            .expect("Failed to get metrics in test");
        assert_eq!(metrics.completed_requests, 2); // success and empty
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.error_rate, 1.0 / 3.0); // 1 failed out of 3 total

        collector
            .stop()
            .await
            .expect("Failed to stop collector in test");
    }

    #[tokio::test]
    async fn test_persistent_counters2() {
        let config = InferenceConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus = TelemetryBus::new(telemetry_config)
            .expect("Failed to create telemetry bus in test");
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = InferenceCollector::new(config, telemetry_sender)
            .expect("Failed to create InferenceCollector in test");
        collector
            .start()
            .await
            .expect("Failed to start collector in test");

        // Test completed counter
        let success_id = Uuid::new_v4();
        assert!(collector
            .track_inference_start(success_id, 50)
            .await
            .is_ok());
        {
            let mut active_requests = collector.active_requests.write().await;
            if let Some(request) = active_requests.get_mut(&success_id) {
                use std::time::{Duration, Instant};
                let token_time = request.start_time + Duration::from_millis(100);
                request.first_token_time = Some(token_time);
                request.tokens_generated = 3;
                request.last_token_time = Some(token_time + Duration::from_millis(50));
            }
        }
        assert!(collector
            .track_inference_complete(success_id, None)
            .await
            .is_ok());

        // Test failed counter
        let fail_id = Uuid::new_v4();
        assert!(collector.track_inference_start(fail_id, 50).await.is_ok());
        assert!(collector
            .track_inference_complete(fail_id, Some("Test error".to_string()))
            .await
            .is_ok());

        // Check that counters are persistent (not reset when requests are removed from active_requests)
        let metrics = collector
            .get_metrics()
            .await
            .expect("Failed to get metrics in test");
        assert_eq!(metrics.completed_requests, 1);
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.active_requests, 0); // All requests completed
        assert_eq!(metrics.error_rate, 0.5); // 1 failed out of 2 total

        // Verify TTFT and TPOT history buffers
        assert!(metrics.average_ttft_ms.is_some());
        assert!(metrics.average_tpot_ms.is_some());
        assert_eq!(
            metrics.average_ttft_ms.expect("TTFT should be present"),
            100.0
        );
        assert_eq!(
            metrics.average_tpot_ms.expect("TPOT should be present"),
            25.0
        ); // 50ms / 2 tokens = 25ms per token

        collector
            .stop()
            .await
            .expect("Failed to stop collector in test");
    }
}
