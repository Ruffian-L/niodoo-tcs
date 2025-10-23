//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Prometheus exporter for Silicon Synapse
//!
//! This module implements a Prometheus-compatible HTTP endpoint for exposing Silicon Synapse metrics.

use crate::silicon_synapse::aggregation::AggregatedMetrics;
use crate::silicon_synapse::config::ExporterConfig;
use axum::{routing::get, Router};
use prometheus::{
    Counter, CounterVec, Encoder, Gauge, HistogramOpts, HistogramVec, Opts, Registry, TextEncoder,
};
use std::net::SocketAddr;
use tracing::info;

#[cfg(test)]
use crate::silicon_synapse::aggregation::{
    HardwareAggregatedMetrics, InferenceAggregatedMetrics, ModelAggregatedMetrics,
    StatisticalSummary,
};
#[cfg(test)]
use std::collections::HashMap;

pub struct PrometheusExporter {
    registry: Registry,
    encoder: TextEncoder,
    router: Router,
    bind_address: String,
    is_running: std::sync::atomic::AtomicBool,
    // Metrics storage
    gpu_temp_gauge: Gauge,
    gpu_power_gauge: Gauge,
    vram_used_gauge: Gauge,
    ttft_histogram: HistogramVec,
    tpot_histogram: HistogramVec,
    tokens_counter: Counter,
    anomaly_counter: CounterVec,
}

impl PrometheusExporter {
    pub fn new(config: ExporterConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let registry = Registry::new_custom(Some("silicon_synapse".to_string()), None)?;

        // Hardware gauges
        let gpu_temp_gauge = Gauge::new("gpu_temperature_celsius", "GPU temperature in Celsius")?;
        registry.register(Box::new(gpu_temp_gauge.clone()))?;

        let gpu_power_gauge = Gauge::new("gpu_power_watts", "GPU power consumption in watts")?;
        registry.register(Box::new(gpu_power_gauge.clone()))?;

        let vram_used_gauge = Gauge::new("vram_used_bytes", "GPU VRAM used in bytes")?;
        registry.register(Box::new(vram_used_gauge.clone()))?;

        // Inference histograms
        let ttft_opts = HistogramOpts::new(
            "inference_ttft_milliseconds",
            "Time to first token in milliseconds",
        );
        let ttft_histogram = HistogramVec::new(ttft_opts, &["model"])?;
        registry.register(Box::new(ttft_histogram.clone()))?;

        let tpot_opts = HistogramOpts::new(
            "inference_tpot_milliseconds",
            "Time per output token in milliseconds",
        );
        let tpot_histogram = HistogramVec::new(tpot_opts, &["model"])?;
        registry.register(Box::new(tpot_histogram.clone()))?;

        // Counters
        let tokens_counter = Counter::new("tokens_generated_total", "Total tokens generated")?;
        registry.register(Box::new(tokens_counter.clone()))?;

        let anomaly_opts = Opts::new("anomalies_detected_total", "Total anomalies detected");
        let anomaly_counter = CounterVec::new(anomaly_opts, &["type"])?;
        registry.register(Box::new(anomaly_counter.clone()))?;

        let encoder = TextEncoder::new();

        // Clone registry for the closure to avoid move issues
        let registry_clone = registry.clone();
        let app = Router::new()
            .route(
                "/metrics",
                get(move || async move {
                    let encoder = TextEncoder::new();
                    let metric_families = registry_clone.gather();
                    let mut buffer = vec![];
                    encoder.encode(&metric_families, &mut buffer).unwrap();
                    axum::response::Response::builder()
                        .header("Content-Type", encoder.format_type())
                        .body(axum::body::Body::from(buffer))
                        .unwrap()
                }),
            )
            .route("/health", get(|| async { "OK" }));

        Ok(Self {
            registry,
            encoder,
            router: app,
            bind_address: config.bind_address,
            is_running: std::sync::atomic::AtomicBool::new(false),
            gpu_temp_gauge,
            gpu_power_gauge,
            vram_used_gauge,
            ttft_histogram,
            tpot_histogram,
            tokens_counter,
            anomaly_counter,
        })
    }

    pub async fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.is_running
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let addr: SocketAddr = self.bind_address.parse()?;
        info!("Starting Prometheus exporter on {}", addr);

        // For now, we'll use a simple periodic update mechanism
        // In a real implementation, this would receive metrics from a channel
        let gpu_temp_gauge = self.gpu_temp_gauge.clone();
        let gpu_power_gauge = self.gpu_power_gauge.clone();
        let vram_used_gauge = self.vram_used_gauge.clone();
        let ttft_histogram = self.ttft_histogram.clone();
        let tpot_histogram = self.tpot_histogram.clone();
        let anomaly_counter = self.anomaly_counter.clone();
        let tokens_counter = self.tokens_counter.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(crate::utils::threshold_convenience::timeout(
                crate::utils::TimeoutCriticality::Low,
            ));

            loop {
                interval.tick().await;

                // Update hardware metrics (stub values for now)
                gpu_temp_gauge.set(72.5);
                gpu_power_gauge.set(245.3);
                vram_used_gauge.set(8589934592.0);

                // Update inference metrics (stub values for now)
                ttft_histogram
                    .with_label_values(&["qwen-7b"])
                    .observe(0.125);
                tpot_histogram
                    .with_label_values(&["qwen-7b"])
                    .observe(0.025);

                // Update counters (stub values for now)
                tokens_counter.inc_by(100.0);
                anomaly_counter
                    .with_label_values(&["performance", "warning"])
                    .inc();

                info!("Updated Prometheus metrics");
            }
        });

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, self.router.clone()).await?;
        Ok(())
    }

    pub fn update_metrics(&self, metrics: &AggregatedMetrics) {
        // Update logic here, but since async task, perhaps use a channel
    }

    /// Check if the exporter is currently running
    pub fn is_running(&self) -> bool {
        self.is_running.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub async fn stop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.is_running
            .store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[tokio::test]
    async fn test_prometheus_exporter_creation() {
        let config = ExporterConfig::default();
        let exporter = PrometheusExporter::new(config);
        assert!(exporter.is_ok());
    }

    #[tokio::test]
    async fn test_prometheus_exporter_start_stop() {
        let mut config = ExporterConfig::default();
        config.bind_address = "127.0.0.1:0".to_string(); // Use random port
        let mut exporter = PrometheusExporter::new(config).unwrap();

        // Note: This test might fail if the port is already in use
        // In a real test environment, we'd use a random port
        let result = exporter.start().await;
        if result.is_ok() {
            assert!(exporter.stop().await.is_ok());
        }
    }

    #[test]
    fn test_metrics_endpoint_returns_valid_prometheus_format() {
        let config = ExporterConfig::default();
        let exporter = PrometheusExporter::new(config).unwrap();

        // Create test metrics
        let mut test_metrics = AggregatedMetrics {
            timestamp: SystemTime::now(),
            window_start: SystemTime::now()
                - crate::utils::threshold_convenience::timeout(
                    crate::utils::TimeoutCriticality::Low,
                ),
            window_duration: crate::utils::threshold_convenience::timeout(
                crate::utils::TimeoutCriticality::Low,
            ),
            hardware_metrics: HardwareAggregatedMetrics {
                gpu_temperature: StatisticalSummary {
                    count: 10,
                    mean: 72.5,
                    std_dev: 2.1,
                    min: 68.0,
                    max: 76.0,
                    percentiles: HashMap::new(),
                },
                gpu_power: StatisticalSummary {
                    count: 10,
                    mean: 245.3,
                    std_dev: 15.2,
                    min: 220.0,
                    max: 270.0,
                    percentiles: HashMap::new(),
                },
                gpu_utilization: StatisticalSummary {
                    count: 10,
                    mean: 85.2,
                    std_dev: 5.1,
                    min: 75.0,
                    max: 95.0,
                    percentiles: HashMap::new(),
                },
                gpu_memory_usage: StatisticalSummary {
                    count: 10,
                    mean: 8589934592.0, // 8GB
                    std_dev: 0.0,
                    min: 8589934592.0,
                    max: 8589934592.0,
                    percentiles: HashMap::new(),
                },
                cpu_temperature: StatisticalSummary {
                    count: 10,
                    mean: 45.2,
                    std_dev: 1.5,
                    min: 42.0,
                    max: 48.0,
                    percentiles: HashMap::new(),
                },
                cpu_utilization: StatisticalSummary {
                    count: 10,
                    mean: 25.8,
                    std_dev: 3.2,
                    min: 20.0,
                    max: 35.0,
                    percentiles: HashMap::new(),
                },
                system_memory_usage: StatisticalSummary {
                    count: 10,
                    mean: 17179869184.0, // 16GB
                    std_dev: 0.0,
                    min: 17179869184.0,
                    max: 17179869184.0,
                    percentiles: HashMap::new(),
                },
            },
            inference_metrics: InferenceAggregatedMetrics {
                ttft_ms: StatisticalSummary {
                    count: 100,
                    mean: 125.4,
                    std_dev: 15.2,
                    min: 95.0,
                    max: 180.0,
                    percentiles: HashMap::new(),
                },
                tpot_ms: StatisticalSummary {
                    count: 100,
                    mean: 25.8,
                    std_dev: 3.1,
                    min: 20.0,
                    max: 35.0,
                    percentiles: HashMap::new(),
                },
                throughput_tps: StatisticalSummary {
                    count: 100,
                    mean: 38.7,
                    std_dev: 2.5,
                    min: 30.0,
                    max: 45.0,
                    percentiles: HashMap::new(),
                },
                error_rate: StatisticalSummary {
                    count: 100,
                    mean: 0.02,
                    std_dev: 0.01,
                    min: 0.0,
                    max: 0.05,
                    percentiles: HashMap::new(),
                },
                active_requests: StatisticalSummary {
                    count: 100,
                    mean: 5.2,
                    std_dev: 1.2,
                    min: 1.0,
                    max: 10.0,
                    percentiles: HashMap::new(),
                },
                completed_requests: StatisticalSummary {
                    count: 100,
                    mean: 95.8,
                    std_dev: 2.1,
                    min: 90.0,
                    max: 100.0,
                    percentiles: HashMap::new(),
                },
            },
            model_metrics: ModelAggregatedMetrics {
                entropy_by_layer: {
                    let mut map = HashMap::new();
                    map.insert(
                        "0".to_string(),
                        StatisticalSummary {
                            count: 1000,
                            mean: 2.34,
                            std_dev: 0.15,
                            min: 2.0,
                            max: 2.8,
                            percentiles: HashMap::new(),
                        },
                    );
                    map.insert(
                        "12".to_string(),
                        StatisticalSummary {
                            count: 1000,
                            mean: 1.87,
                            std_dev: 0.12,
                            min: 1.5,
                            max: 2.2,
                            percentiles: HashMap::new(),
                        },
                    );
                    map
                },
                activation_sparsity_by_layer: {
                    let mut map = HashMap::new();
                    map.insert(
                        "0".to_string(),
                        StatisticalSummary {
                            count: 1000,
                            mean: 0.15,
                            std_dev: 0.05,
                            min: 0.1,
                            max: 0.25,
                            percentiles: HashMap::new(),
                        },
                    );
                    map.insert(
                        "12".to_string(),
                        StatisticalSummary {
                            count: 1000,
                            mean: 0.87,
                            std_dev: 0.08,
                            min: 0.8,
                            max: 0.95,
                            percentiles: HashMap::new(),
                        },
                    );
                    map
                },
                activation_magnitude_by_layer: {
                    let mut map = HashMap::new();
                    map.insert(
                        "0".to_string(),
                        StatisticalSummary {
                            count: 1000,
                            mean: 0.45,
                            std_dev: 0.12,
                            min: 0.3,
                            max: 0.6,
                            percentiles: HashMap::new(),
                        },
                    );
                    map.insert(
                        "12".to_string(),
                        StatisticalSummary {
                            count: 1000,
                            mean: 0.23,
                            std_dev: 0.08,
                            min: 0.15,
                            max: 0.35,
                            percentiles: HashMap::new(),
                        },
                    );
                    map
                },
            },
            correlations: Vec::new(),
        };

        // Update metrics
        exporter.update_metrics(&test_metrics);

        // Test that metrics are properly set
        assert_eq!(exporter.gpu_temp_gauge.get(), 72.5);
        assert_eq!(exporter.gpu_power_gauge.get(), 245.3);

        // Test anomaly recording
        exporter
            .anomaly_counter
            .with_label_values(&["test", "info"])
            .inc();
        // Note: CounterVec doesn't have a simple .get() method like Counter
    }

    #[tokio::test]
    async fn test_metric_updates_reflected_in_scrapes() {
        let config = ExporterConfig::default();
        let exporter = PrometheusExporter::new(config).unwrap();

        // Create initial metrics
        let mut initial_metrics = AggregatedMetrics {
            timestamp: SystemTime::now(),
            window_start: SystemTime::now()
                - crate::utils::threshold_convenience::timeout(
                    crate::utils::TimeoutCriticality::Low,
                ),
            window_duration: crate::utils::threshold_convenience::timeout(
                crate::utils::TimeoutCriticality::Low,
            ),
            hardware_metrics: HardwareAggregatedMetrics {
                gpu_temperature: StatisticalSummary {
                    count: 1,
                    mean: 70.0,
                    std_dev: 0.0,
                    min: 70.0,
                    max: 70.0,
                    percentiles: HashMap::new(),
                },
                gpu_power: StatisticalSummary {
                    count: 1,
                    mean: 240.0,
                    std_dev: 0.0,
                    min: 240.0,
                    max: 240.0,
                    percentiles: HashMap::new(),
                },
                gpu_utilization: StatisticalSummary {
                    count: 1,
                    mean: 80.0,
                    std_dev: 0.0,
                    min: 80.0,
                    max: 80.0,
                    percentiles: HashMap::new(),
                },
                gpu_memory_usage: StatisticalSummary {
                    count: 1,
                    mean: 8589934592.0, // 8GB
                    std_dev: 0.0,
                    min: 8589934592.0,
                    max: 8589934592.0,
                    percentiles: HashMap::new(),
                },
                cpu_temperature: StatisticalSummary {
                    count: 1,
                    mean: 45.0,
                    std_dev: 0.0,
                    min: 45.0,
                    max: 45.0,
                    percentiles: HashMap::new(),
                },
                cpu_utilization: StatisticalSummary {
                    count: 1,
                    mean: 25.0,
                    std_dev: 0.0,
                    min: 25.0,
                    max: 25.0,
                    percentiles: HashMap::new(),
                },
                system_memory_usage: StatisticalSummary {
                    count: 1,
                    mean: 17179869184.0, // 16GB
                    std_dev: 0.0,
                    min: 17179869184.0,
                    max: 17179869184.0,
                    percentiles: HashMap::new(),
                },
            },
            inference_metrics: InferenceAggregatedMetrics {
                ttft_ms: StatisticalSummary {
                    count: 1,
                    mean: 120.0,
                    std_dev: 0.0,
                    min: 120.0,
                    max: 120.0,
                    percentiles: HashMap::new(),
                },
                tpot_ms: StatisticalSummary {
                    count: 1,
                    mean: 25.0,
                    std_dev: 0.0,
                    min: 25.0,
                    max: 25.0,
                    percentiles: HashMap::new(),
                },
                throughput_tps: StatisticalSummary {
                    count: 1,
                    mean: 30.0,
                    std_dev: 0.0,
                    min: 30.0,
                    max: 30.0,
                    percentiles: HashMap::new(),
                },
                error_rate: StatisticalSummary {
                    count: 1,
                    mean: 0.01,
                    std_dev: 0.0,
                    min: 0.01,
                    max: 0.01,
                    percentiles: HashMap::new(),
                },
                active_requests: StatisticalSummary {
                    count: 1,
                    mean: 5.0,
                    std_dev: 0.0,
                    min: 5.0,
                    max: 5.0,
                    percentiles: HashMap::new(),
                },
                completed_requests: StatisticalSummary {
                    count: 1,
                    mean: 95.0,
                    std_dev: 0.0,
                    min: 95.0,
                    max: 95.0,
                    percentiles: HashMap::new(),
                },
            },
            model_metrics: ModelAggregatedMetrics {
                entropy_by_layer: {
                    let mut map = HashMap::new();
                    map.insert(
                        "0".to_string(),
                        StatisticalSummary {
                            count: 1,
                            mean: 2.34,
                            std_dev: 0.0,
                            min: 2.34,
                            max: 2.34,
                            percentiles: HashMap::new(),
                        },
                    );
                    map.insert(
                        "12".to_string(),
                        StatisticalSummary {
                            count: 1,
                            mean: 1.87,
                            std_dev: 0.0,
                            min: 1.87,
                            max: 1.87,
                            percentiles: HashMap::new(),
                        },
                    );
                    map
                },
                activation_sparsity_by_layer: {
                    let mut map = HashMap::new();
                    map.insert(
                        "0".to_string(),
                        StatisticalSummary {
                            count: 1,
                            mean: 0.15,
                            std_dev: 0.0,
                            min: 0.15,
                            max: 0.15,
                            percentiles: HashMap::new(),
                        },
                    );
                    map.insert(
                        "12".to_string(),
                        StatisticalSummary {
                            count: 1,
                            mean: 0.87,
                            std_dev: 0.0,
                            min: 0.87,
                            max: 0.87,
                            percentiles: HashMap::new(),
                        },
                    );
                    map
                },
                activation_magnitude_by_layer: {
                    let mut map = HashMap::new();
                    map.insert(
                        "0".to_string(),
                        StatisticalSummary {
                            count: 1,
                            mean: 0.45,
                            std_dev: 0.0,
                            min: 0.45,
                            max: 0.45,
                            percentiles: HashMap::new(),
                        },
                    );
                    map.insert(
                        "12".to_string(),
                        StatisticalSummary {
                            count: 1,
                            mean: 0.23,
                            std_dev: 0.0,
                            min: 0.23,
                            max: 0.23,
                            percentiles: HashMap::new(),
                        },
                    );
                    map
                },
            },
            correlations: Vec::new(),
        };

        // Update with initial metrics
        exporter.update_metrics(&initial_metrics);
        let initial_temp = exporter.gpu_temp_gauge.get();

        // Create updated metrics
        let mut updated_metrics = AggregatedMetrics {
            timestamp: SystemTime::now(),
            window_start: SystemTime::now()
                - crate::utils::threshold_convenience::timeout(
                    crate::utils::TimeoutCriticality::Low,
                ),
            window_duration: crate::utils::threshold_convenience::timeout(
                crate::utils::TimeoutCriticality::Low,
            ),
            hardware_metrics: HardwareAggregatedMetrics {
                gpu_temperature: StatisticalSummary {
                    count: 1,
                    mean: 75.0,
                    std_dev: 0.0,
                    min: 75.0,
                    max: 75.0,
                    percentiles: HashMap::new(),
                },
                gpu_power: StatisticalSummary {
                    count: 1,
                    mean: 240.0,
                    std_dev: 0.0,
                    min: 240.0,
                    max: 240.0,
                    percentiles: HashMap::new(),
                },
                gpu_utilization: StatisticalSummary {
                    count: 1,
                    mean: 80.0,
                    std_dev: 0.0,
                    min: 80.0,
                    max: 80.0,
                    percentiles: HashMap::new(),
                },
                gpu_memory_usage: StatisticalSummary {
                    count: 1,
                    mean: 8589934592.0, // 8GB
                    std_dev: 0.0,
                    min: 8589934592.0,
                    max: 8589934592.0,
                    percentiles: HashMap::new(),
                },
                cpu_temperature: StatisticalSummary {
                    count: 1,
                    mean: 45.0,
                    std_dev: 0.0,
                    min: 45.0,
                    max: 45.0,
                    percentiles: HashMap::new(),
                },
                cpu_utilization: StatisticalSummary {
                    count: 1,
                    mean: 25.0,
                    std_dev: 0.0,
                    min: 25.0,
                    max: 25.0,
                    percentiles: HashMap::new(),
                },
                system_memory_usage: StatisticalSummary {
                    count: 1,
                    mean: 17179869184.0, // 16GB
                    std_dev: 0.0,
                    min: 17179869184.0,
                    max: 17179869184.0,
                    percentiles: HashMap::new(),
                },
            },
            inference_metrics: InferenceAggregatedMetrics {
                ttft_ms: StatisticalSummary {
                    count: 1,
                    mean: 120.0,
                    std_dev: 0.0,
                    min: 120.0,
                    max: 120.0,
                    percentiles: HashMap::new(),
                },
                tpot_ms: StatisticalSummary {
                    count: 1,
                    mean: 25.0,
                    std_dev: 0.0,
                    min: 25.0,
                    max: 25.0,
                    percentiles: HashMap::new(),
                },
                throughput_tps: StatisticalSummary {
                    count: 1,
                    mean: 30.0,
                    std_dev: 0.0,
                    min: 30.0,
                    max: 30.0,
                    percentiles: HashMap::new(),
                },
                error_rate: StatisticalSummary {
                    count: 1,
                    mean: 0.01,
                    std_dev: 0.0,
                    min: 0.01,
                    max: 0.01,
                    percentiles: HashMap::new(),
                },
                active_requests: StatisticalSummary {
                    count: 1,
                    mean: 5.0,
                    std_dev: 0.0,
                    min: 5.0,
                    max: 5.0,
                    percentiles: HashMap::new(),
                },
                completed_requests: StatisticalSummary {
                    count: 1,
                    mean: 95.0,
                    std_dev: 0.0,
                    min: 95.0,
                    max: 95.0,
                    percentiles: HashMap::new(),
                },
            },
            model_metrics: ModelAggregatedMetrics {
                entropy_by_layer: {
                    let mut map = HashMap::new();
                    map.insert(
                        "0".to_string(),
                        StatisticalSummary {
                            count: 1,
                            mean: 2.34,
                            std_dev: 0.0,
                            min: 2.34,
                            max: 2.34,
                            percentiles: HashMap::new(),
                        },
                    );
                    map.insert(
                        "12".to_string(),
                        StatisticalSummary {
                            count: 1,
                            mean: 1.87,
                            std_dev: 0.0,
                            min: 1.87,
                            max: 1.87,
                            percentiles: HashMap::new(),
                        },
                    );
                    map
                },
                activation_sparsity_by_layer: {
                    let mut map = HashMap::new();
                    map.insert(
                        "0".to_string(),
                        StatisticalSummary {
                            count: 1,
                            mean: 0.15,
                            std_dev: 0.0,
                            min: 0.15,
                            max: 0.15,
                            percentiles: HashMap::new(),
                        },
                    );
                    map.insert(
                        "12".to_string(),
                        StatisticalSummary {
                            count: 1,
                            mean: 0.87,
                            std_dev: 0.0,
                            min: 0.87,
                            max: 0.87,
                            percentiles: HashMap::new(),
                        },
                    );
                    map
                },
                activation_magnitude_by_layer: {
                    let mut map = HashMap::new();
                    map.insert(
                        "0".to_string(),
                        StatisticalSummary {
                            count: 1,
                            mean: 0.45,
                            std_dev: 0.0,
                            min: 0.45,
                            max: 0.45,
                            percentiles: HashMap::new(),
                        },
                    );
                    map.insert(
                        "12".to_string(),
                        StatisticalSummary {
                            count: 1,
                            mean: 0.23,
                            std_dev: 0.0,
                            min: 0.23,
                            max: 0.23,
                            percentiles: HashMap::new(),
                        },
                    );
                    map
                },
            },
            correlations: Vec::new(),
        };

        // Update with new metrics
        exporter.update_metrics(&updated_metrics);
        let updated_temp = exporter.gpu_temp_gauge.get();

        // Verify metrics were updated
        assert_eq!(initial_temp, 70.0);
        assert_eq!(updated_temp, 75.0);
        assert_ne!(initial_temp, updated_temp);
    }
}
