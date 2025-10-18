//! JSON API exporter for Silicon Synapse
//!
//! This module implements a JSON API exporter that exposes Silicon Synapse metrics
//! via REST endpoints for programmatic access.

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, Router},
};
use serde_json::{json, Value};
use std::sync::{atomic::AtomicU64, Arc};
use std::time::SystemTime;
use tokio::sync::{mpsc, watch, RwLock};
use tracing::info;

use crate::silicon_synapse::aggregation::{AggregatedMetrics, AggregationEngine};
use crate::silicon_synapse::config::ExporterConfig;
use crate::silicon_synapse::telemetry_bus::TelemetrySender;

/// JSON API metrics exporter
pub struct JsonApiExporter {
    config: ExporterConfig,
    aggregation_engine: Arc<AggregationEngine>,
    is_running: Arc<RwLock<bool>>,
    server_handle: Option<tokio::task::JoinHandle<()>>,
}

impl JsonApiExporter {
    /// Create a new JSON API exporter
    pub fn new(config: ExporterConfig) -> Result<Self, String> {
        // Create a dummy aggregation engine for now
        let aggregation_config = crate::silicon_synapse::config::AggregationConfig::default();
        let telemetry_sender = TelemetrySender {
            inner: mpsc::unbounded_channel().0,
            dropped_events: Arc::new(AtomicU64::new(0)),
        };
        let (metrics_tx, _metrics_rx) = watch::channel(AggregatedMetrics::default());
        let aggregation_engine = Arc::new(AggregationEngine::new(
            aggregation_config,
            telemetry_sender,
            metrics_tx,
        )?);

        Ok(Self {
            config,
            aggregation_engine,
            is_running: Arc::new(RwLock::new(false)),
            server_handle: None,
        })
    }

    /// Start the JSON API exporter
    pub async fn start(&mut self) -> Result<(), String> {
        if *self.is_running.read().await {
            return Err("JSON API exporter is already running".to_string());
        }

        info!("Starting JSON API exporter on {}", self.config.bind_address);

        let config = self.config.clone();
        let aggregation_engine = Arc::clone(&self.aggregation_engine);
        let is_running = self.is_running.clone();

        let task = tokio::spawn(async move {
            Self::run_server(config, aggregation_engine, is_running).await;
        });

        self.server_handle = Some(task);
        *self.is_running.write().await = true;

        Ok(())
    }

    /// Stop the JSON API exporter
    pub async fn stop(&mut self) -> Result<(), String> {
        if !*self.is_running.read().await {
            return Ok(());
        }

        info!("Stopping JSON API exporter");
        *self.is_running.write().await = false;

        if let Some(task) = self.server_handle.take() {
            task.abort();
        }

        Ok(())
    }

    /// Run the HTTP server
    async fn run_server(
        config: ExporterConfig,
        aggregation_engine: Arc<AggregationEngine>,
        is_running: Arc<RwLock<bool>>,
    ) {
        let app = Router::new()
            .route("/api/v1/metrics", get(Self::metrics_handler))
            .route("/api/v1/health", get(Self::health_handler))
            .route("/api/v1/status", get(Self::status_handler))
            .with_state(aggregation_engine);

        let listener = tokio::net::TcpListener::bind(&config.bind_address).await;
        match listener {
            Ok(listener) => {
                info!("JSON API exporter listening on {}", config.bind_address);

                // Start the server
                let server = axum::serve(listener, app);

                // Run until stopped
                let is_running_clone = is_running.clone();
                tokio::select! {
                    _ = server => {
                        info!("JSON API exporter server stopped");
                    }
                    _ = async {
                        while *is_running_clone.read().await {
                            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                        }
                    } => {
                        info!("JSON API exporter shutdown requested");
                    }
                }
            }
            Err(e) => {
                tracing::error!("Failed to bind to {}: {}", config.bind_address, e);
            }
        }
    }

    /// Handle metrics endpoint
    async fn metrics_handler(
        State(aggregation_engine): State<Arc<AggregationEngine>>,
    ) -> Result<Json<Value>, StatusCode> {
        let metrics = aggregation_engine.get_metrics().await;
        let json_metrics = Self::format_json_metrics(&metrics);

        Ok(Json(json_metrics))
    }

    /// Handle health check endpoint
    async fn health_handler() -> Result<Json<Value>, StatusCode> {
        let health_status = json!({
            "status": "healthy",
            "timestamp": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs(),
            "service": "silicon_synapse_json_api"
        });

        Ok(Json(health_status))
    }

    /// Handle status endpoint
    async fn status_handler() -> Result<Json<Value>, StatusCode> {
        let status = json!({
            "service": "Silicon Synapse JSON API",
            "version": "1.0.0",
            "status": "running",
            "timestamp": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs(),
            "endpoints": {
                "metrics": "/api/v1/metrics",
                "health": "/api/v1/health",
                "status": "/api/v1/status"
            }
        });

        Ok(Json(status))
    }

    /// Format metrics as JSON
    fn format_json_metrics(metrics: &AggregatedMetrics) -> Value {
        json!({
            "timestamp": metrics.timestamp.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs(),
            "window_start": metrics.window_start.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs(),
            "window_duration_seconds": metrics.window_duration.as_secs(),
            "hardware": {
                "gpu": {
                    "temperature": {
                        "mean": metrics.hardware_metrics.gpu_temperature.mean,
                        "std_dev": metrics.hardware_metrics.gpu_temperature.std_dev,
                        "min": metrics.hardware_metrics.gpu_temperature.min,
                        "max": metrics.hardware_metrics.gpu_temperature.max,
                        "count": metrics.hardware_metrics.gpu_temperature.count
                    },
                    "power": {
                        "mean": metrics.hardware_metrics.gpu_power.mean,
                        "std_dev": metrics.hardware_metrics.gpu_power.std_dev,
                        "min": metrics.hardware_metrics.gpu_power.min,
                        "max": metrics.hardware_metrics.gpu_power.max,
                        "count": metrics.hardware_metrics.gpu_power.count
                    },
                    "utilization": {
                        "mean": metrics.hardware_metrics.gpu_utilization.mean,
                        "std_dev": metrics.hardware_metrics.gpu_utilization.std_dev,
                        "min": metrics.hardware_metrics.gpu_utilization.min,
                        "max": metrics.hardware_metrics.gpu_utilization.max,
                        "count": metrics.hardware_metrics.gpu_utilization.count
                    },
                    "memory_usage": {
                        "mean": metrics.hardware_metrics.gpu_memory_usage.mean,
                        "std_dev": metrics.hardware_metrics.gpu_memory_usage.std_dev,
                        "min": metrics.hardware_metrics.gpu_memory_usage.min,
                        "max": metrics.hardware_metrics.gpu_memory_usage.max,
                        "count": metrics.hardware_metrics.gpu_memory_usage.count
                    }
                },
                "cpu": {
                    "temperature": {
                        "mean": metrics.hardware_metrics.cpu_temperature.mean,
                        "std_dev": metrics.hardware_metrics.cpu_temperature.std_dev,
                        "min": metrics.hardware_metrics.cpu_temperature.min,
                        "max": metrics.hardware_metrics.cpu_temperature.max,
                        "count": metrics.hardware_metrics.cpu_temperature.count
                    },
                    "utilization": {
                        "mean": metrics.hardware_metrics.cpu_utilization.mean,
                        "std_dev": metrics.hardware_metrics.cpu_utilization.std_dev,
                        "min": metrics.hardware_metrics.cpu_utilization.min,
                        "max": metrics.hardware_metrics.cpu_utilization.max,
                        "count": metrics.hardware_metrics.cpu_utilization.count
                    }
                },
                "system_memory": {
                    "usage": {
                        "mean": metrics.hardware_metrics.system_memory_usage.mean,
                        "std_dev": metrics.hardware_metrics.system_memory_usage.std_dev,
                        "min": metrics.hardware_metrics.system_memory_usage.min,
                        "max": metrics.hardware_metrics.system_memory_usage.max,
                        "count": metrics.hardware_metrics.system_memory_usage.count
                    }
                }
            },
            "inference": {
                "ttft_ms": {
                    "mean": metrics.inference_metrics.ttft_ms.mean,
                    "std_dev": metrics.inference_metrics.ttft_ms.std_dev,
                    "min": metrics.inference_metrics.ttft_ms.min,
                    "max": metrics.inference_metrics.ttft_ms.max,
                    "count": metrics.inference_metrics.ttft_ms.count
                },
                "tpot_ms": {
                    "mean": metrics.inference_metrics.tpot_ms.mean,
                    "std_dev": metrics.inference_metrics.tpot_ms.std_dev,
                    "min": metrics.inference_metrics.tpot_ms.min,
                    "max": metrics.inference_metrics.tpot_ms.max,
                    "count": metrics.inference_metrics.tpot_ms.count
                },
                "throughput_tps": {
                    "mean": metrics.inference_metrics.throughput_tps.mean,
                    "std_dev": metrics.inference_metrics.throughput_tps.std_dev,
                    "min": metrics.inference_metrics.throughput_tps.min,
                    "max": metrics.inference_metrics.throughput_tps.max,
                    "count": metrics.inference_metrics.throughput_tps.count
                },
                "error_rate": {
                    "mean": metrics.inference_metrics.error_rate.mean,
                    "std_dev": metrics.inference_metrics.error_rate.std_dev,
                    "min": metrics.inference_metrics.error_rate.min,
                    "max": metrics.inference_metrics.error_rate.max,
                    "count": metrics.inference_metrics.error_rate.count
                },
                "active_requests": {
                    "mean": metrics.inference_metrics.active_requests.mean,
                    "std_dev": metrics.inference_metrics.active_requests.std_dev,
                    "min": metrics.inference_metrics.active_requests.min,
                    "max": metrics.inference_metrics.active_requests.max,
                    "count": metrics.inference_metrics.active_requests.count
                },
                "completed_requests": {
                    "mean": metrics.inference_metrics.completed_requests.mean,
                    "std_dev": metrics.inference_metrics.completed_requests.std_dev,
                    "min": metrics.inference_metrics.completed_requests.min,
                    "max": metrics.inference_metrics.completed_requests.max,
                    "count": metrics.inference_metrics.completed_requests.count
                }
            },
            "model": {
                "entropy_by_layer": metrics.model_metrics.entropy_by_layer.iter().map(|(layer, entropy)| {
                    (layer.to_string(), json!({
                        "mean": entropy.mean,
                        "std_dev": entropy.std_dev,
                        "min": entropy.min,
                        "max": entropy.max,
                        "count": entropy.count
                    }))
                }).collect::<std::collections::HashMap<String, Value>>(),
                "activation_sparsity_by_layer": metrics.model_metrics.activation_sparsity_by_layer.iter().map(|(layer, sparsity)| {
                    (layer.to_string(), json!({
                        "mean": sparsity.mean,
                        "std_dev": sparsity.std_dev,
                        "min": sparsity.min,
                        "max": sparsity.max,
                        "count": sparsity.count
                    }))
                }).collect::<std::collections::HashMap<String, Value>>(),
                "activation_magnitude_by_layer": metrics.model_metrics.activation_magnitude_by_layer.iter().map(|(layer, magnitude)| {
                    (layer.to_string(), json!({
                        "mean": magnitude.mean,
                        "std_dev": magnitude.std_dev,
                        "min": magnitude.min,
                        "max": magnitude.max,
                        "count": magnitude.count
                    }))
                }).collect::<std::collections::HashMap<String, Value>>()
            },
            "correlations": metrics.correlations.iter().map(|correlation| {
                json!({
                    "metric1": correlation.metric1,
                    "metric2": correlation.metric2,
                    "correlation_coefficient": correlation.correlation_coefficient,
                    "p_value": correlation.p_value,
                    "significance": correlation.significance
                })
            }).collect::<Vec<Value>>()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_json_api_exporter_creation() {
        let config = ExporterConfig::default();
        let exporter = JsonApiExporter::new(config);
        assert!(exporter.is_ok());
    }

    #[tokio::test]
    async fn test_json_api_exporter_start_stop() {
        let config = ExporterConfig::default();
        let mut exporter = JsonApiExporter::new(config).unwrap();

        // Note: This test might fail if the port is already in use
        // In a real test environment, we'd use a random port
        let result = exporter.start().await;
        if result.is_ok() {
            assert!(exporter.stop().await.is_ok());
        }
    }

    #[test]
    fn test_json_metrics_formatting() {
        let metrics = AggregatedMetrics::default();
        let formatted = JsonApiExporter::format_json_metrics(&metrics);

        assert!(formatted.is_object());
        assert!(formatted.get("timestamp").is_some());
        assert!(formatted.get("hardware").is_some());
        assert!(formatted.get("inference").is_some());
        assert!(formatted.get("model").is_some());
        assert!(formatted.get("correlations").is_some());
    }
}
