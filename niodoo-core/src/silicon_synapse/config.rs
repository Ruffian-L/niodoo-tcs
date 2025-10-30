// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! # Configuration Module
//!
//! This module handles configuration loading and validation for the Silicon Synapse system.
//! All configuration is TOML-based with environment variable overrides and hot-reload support.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Main configuration structure for Silicon Synapse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Enable Silicon Synapse monitoring
    pub enabled: bool,
    /// Hardware monitoring configuration
    pub hardware: HardwareConfig,
    /// Inference monitoring configuration
    pub inference: InferenceConfig,
    /// Model probe configuration
    pub model_probe: ModelProbeConfig,
    /// Baseline learning configuration
    pub baseline: BaselineConfig,
    /// Exporter configuration
    pub exporter: ExporterConfig,
    /// Telemetry bus configuration
    pub telemetry: TelemetryConfig,
    /// Aggregation configuration
    pub aggregation: AggregationConfig,
}

/// Hardware monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Enable hardware monitoring
    pub enabled: bool,
    /// GPU monitoring configuration
    pub gpu: GpuConfig,
    /// CPU monitoring configuration
    pub cpu: CpuConfig,
    /// Memory monitoring configuration
    pub memory: MemoryConfig,
    /// Collection interval in milliseconds
    pub collection_interval_ms: u64,
    /// Enable temperature monitoring
    pub enable_temperature: bool,
    /// Enable power monitoring
    pub enable_power: bool,
    /// Enable utilization monitoring
    pub enable_utilization: bool,
}

/// GPU monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Enable GPU monitoring
    pub enabled: bool,
    /// GPU vendor preference (nvidia, amd, auto)
    pub vendor: String,
    /// Maximum number of GPUs to monitor
    pub max_gpus: usize,
    /// Enable NVIDIA NVML monitoring
    pub enable_nvidia: bool,
    /// Enable AMD ROCm monitoring
    pub enable_amd: bool,
    /// Enable Intel GPU monitoring
    pub enable_intel: bool,
}

/// CPU monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuConfig {
    /// Enable CPU monitoring
    pub enabled: bool,
    /// Monitor all CPU cores
    pub monitor_all_cores: bool,
    /// Maximum number of cores to monitor
    pub max_cores: usize,
    /// Enable per-core temperature monitoring
    pub enable_per_core_temp: bool,
}

/// Memory monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Enable memory monitoring
    pub enabled: bool,
    /// Monitor system RAM
    pub monitor_system_ram: bool,
    /// Monitor GPU VRAM
    pub monitor_gpu_vram: bool,
    /// Monitor swap usage
    pub monitor_swap: bool,
}

/// Inference performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Enable inference monitoring
    pub enabled: bool,
    /// Track Time To First Token (TTFT)
    pub track_ttft: bool,
    /// Track Time Per Output Token (TPOT)
    pub track_tpot: bool,
    /// Track throughput (tokens/second)
    pub track_throughput: bool,
    /// Track error rates
    pub track_errors: bool,
    /// Track memory usage during inference
    pub track_memory: bool,
    /// Maximum number of concurrent requests to track
    pub max_concurrent_requests: usize,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
}

/// Model internal state probing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProbeConfig {
    /// Enable model probing
    pub enabled: bool,
    /// Probe softmax entropy
    pub probe_entropy: bool,
    /// Probe activation patterns
    pub probe_activations: bool,
    /// Probe attention weights
    pub probe_attention: bool,
    /// Probe layer outputs
    pub probe_layer_outputs: bool,
    /// Maximum number of layers to probe
    pub max_layers: usize,
    /// Sampling rate (1.0 = all tokens, 0.1 = 10% of tokens)
    pub sampling_rate: f64,
    /// Enable activation sparsity analysis
    pub enable_sparsity_analysis: bool,
    /// Enable activation magnitude analysis
    pub enable_magnitude_analysis: bool,
    /// Enable activation distribution analysis
    pub enable_distribution_analysis: bool,
}

/// Baseline learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineConfig {
    /// Enable baseline learning
    pub enabled: bool,
    /// Learning window duration in hours
    pub learning_window_hours: u64,
    /// Minimum samples required for baseline
    pub min_samples: usize,
    /// Maximum samples for baseline
    pub max_samples: usize,
    /// Baseline update interval in hours
    pub update_interval_hours: u64,
    /// Enable correlation analysis
    pub enable_correlation_analysis: bool,
    /// Correlation threshold for significance
    pub correlation_threshold: f64,
    /// Enable multivariate analysis
    pub enable_multivariate_analysis: bool,
    /// Principal component analysis dimensions
    pub pca_dimensions: usize,
}

/// Exporter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExporterConfig {
    /// Enable metrics export
    pub enabled: bool,
    /// Exporter type (prometheus, json, etc.)
    pub exporter_type: String,
    /// Bind address for the HTTP server
    pub bind_address: String,
    /// Metrics endpoint path
    pub metrics_path: String,
    /// Health check endpoint path
    pub health_path: String,
    /// Enable detailed metrics
    pub enable_detailed_metrics: bool,
    /// Enable anomaly metrics
    pub enable_anomaly_metrics: bool,
    /// Metrics update interval in seconds
    pub update_interval_seconds: u64,
}

/// Telemetry bus configuration with performance optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable telemetry bus
    pub enabled: bool,
    /// Maximum number of events in the buffer
    pub max_buffer_size: usize,
    /// Enable event routing
    pub enable_routing: bool,
    /// Enable dropped event tracking
    pub enable_dropped_event_tracking: bool,
    /// Event processing batch size
    pub batch_size: usize,
    /// Event processing interval in milliseconds
    pub processing_interval_ms: u64,
    /// Enable batch processing optimization
    pub enable_batch_processing: bool,
    /// Enable event compression
    pub enable_compression: bool,
    /// Enable adaptive backpressure
    pub enable_adaptive_backpressure: bool,
    /// Maximum events per second (rate limiting)
    pub max_events_per_second: Option<u64>,
    /// Compression cache size
    pub compression_cache_size: usize,
    /// Target processing latency in milliseconds
    pub target_latency_ms: f32,
}

/// Aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    /// Enable aggregation
    pub enabled: bool,
    /// Size of ring buffers
    pub buffer_size: usize,
    /// Aggregation window duration in seconds
    pub window_duration_seconds: u64,
    /// Enable statistical aggregation
    pub enable_statistical_aggregation: bool,
    /// Enable percentile calculation
    pub enable_percentiles: bool,
    /// Percentiles to calculate (e.g., [50.0, 95.0, 99.0])
    pub percentiles: Vec<f64>,
    /// Enable correlation analysis
    pub enable_correlation_analysis: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: true,
            hardware: HardwareConfig::default(),
            inference: InferenceConfig::default(),
            model_probe: ModelProbeConfig::default(),
            baseline: BaselineConfig::default(),
            exporter: ExporterConfig::default(),
            telemetry: TelemetryConfig::default(),
            aggregation: AggregationConfig::default(),
        }
    }
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            gpu: GpuConfig::default(),
            cpu: CpuConfig::default(),
            memory: MemoryConfig::default(),
            collection_interval_ms: 1000,
            enable_temperature: true,
            enable_power: true,
            enable_utilization: true,
        }
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            vendor: "auto".to_string(),
            max_gpus: 8,
            enable_nvidia: true,
            enable_amd: true,
            enable_intel: true,
        }
    }
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitor_all_cores: true,
            max_cores: 64,
            enable_per_core_temp: true,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitor_system_ram: true,
            monitor_gpu_vram: true,
            monitor_swap: true,
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            track_ttft: true,
            track_tpot: true,
            track_throughput: true,
            track_errors: true,
            track_memory: true,
            max_concurrent_requests: 1000,
            request_timeout_seconds: 300,
        }
    }
}

impl Default for ModelProbeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            probe_entropy: true,
            probe_activations: true,
            probe_attention: true,
            probe_layer_outputs: true,
            max_layers: 32,
            sampling_rate: 0.1,
            enable_sparsity_analysis: true,
            enable_magnitude_analysis: true,
            enable_distribution_analysis: true,
        }
    }
}

impl Default for BaselineConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            learning_window_hours: 24,
            min_samples: 1000,
            max_samples: 100000,
            update_interval_hours: 1,
            enable_correlation_analysis: true,
            correlation_threshold: 0.7,
            enable_multivariate_analysis: true,
            pca_dimensions: 10,
        }
    }
}

impl Default for ExporterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            exporter_type: "prometheus".to_string(),
            bind_address: "0.0.0.0:9090".to_string(),
            metrics_path: "/metrics".to_string(),
            health_path: "/health".to_string(),
            enable_detailed_metrics: true,
            enable_anomaly_metrics: true,
            update_interval_seconds: 15,
        }
    }
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_buffer_size: 10000,
            enable_routing: true,
            enable_dropped_event_tracking: true,
            batch_size: 100,
            processing_interval_ms: 100,
            enable_batch_processing: true,
            enable_compression: true,
            enable_adaptive_backpressure: true,
            max_events_per_second: Some(10000), // 10K events/sec limit
            compression_cache_size: 1000,
            target_latency_ms: 10.0, // Target 10ms processing latency
        }
    }
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            buffer_size: 1000,
            window_duration_seconds: 60,
            enable_statistical_aggregation: true,
            enable_percentiles: true,
            percentiles: vec![50.0, 90.0, 95.0, 99.0],
            enable_correlation_analysis: true,
        }
    }
}

impl Config {
    /// Load configuration from a TOML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a TOML file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load configuration with environment variable overrides
    pub fn load_with_env<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = Self::load(path)?;

        // Apply environment variable overrides
        if let Ok(enabled) = std::env::var("SILICON_SYNAPSE_ENABLED") {
            config.enabled = enabled.parse().unwrap_or(config.enabled);
        }

        if let Ok(bind_addr) = std::env::var("SILICON_SYNAPSE_BIND_ADDRESS") {
            config.exporter.bind_address = bind_addr;
        }

        if let Ok(interval) = std::env::var("SILICON_SYNAPSE_COLLECTION_INTERVAL_MS") {
            if let Ok(interval_ms) = interval.parse() {
                config.hardware.collection_interval_ms = interval_ms;
            }
        }

        Ok(config)
    }

    /// Validate configuration for consistency
    pub fn validate(&self) -> Result<(), String> {
        if self.hardware.collection_interval_ms == 0 {
            return Err("Hardware collection interval must be greater than 0".to_string());
        }

        if self.inference.max_concurrent_requests == 0 {
            return Err("Max concurrent requests must be greater than 0".to_string());
        }

        if self.model_probe.sampling_rate < 0.0 || self.model_probe.sampling_rate > 1.0 {
            return Err("Model probe sampling rate must be between 0.0 and 1.0".to_string());
        }

        if self.baseline.min_samples > self.baseline.max_samples {
            return Err(
                "Baseline min_samples must be less than or equal to max_samples".to_string(),
            );
        }

        if self.aggregation.buffer_size == 0 {
            return Err("Aggregation buffer size must be greater than 0".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert!(config.enabled);
        assert!(config.hardware.enabled);
        assert!(config.inference.enabled);
        assert!(config.model_probe.enabled);
        assert!(config.baseline.enabled);
        assert!(config.exporter.enabled);
        assert!(config.telemetry.enabled);
        assert!(config.aggregation.enabled);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let serialized = toml::to_string(&config).unwrap();
        let deserialized: Config = toml::from_str(&serialized).unwrap();

        assert_eq!(config.enabled, deserialized.enabled);
        assert_eq!(config.hardware.enabled, deserialized.hardware.enabled);
        assert_eq!(config.inference.enabled, deserialized.inference.enabled);
    }

    #[test]
    fn test_config_load_save() {
        let original_config = Config::default();

        let temp_file = NamedTempFile::new().unwrap();
        original_config.save(temp_file.path()).unwrap();

        let loaded_config = Config::load(temp_file.path()).unwrap();

        assert_eq!(original_config.enabled, loaded_config.enabled);
        assert_eq!(
            original_config.hardware.enabled,
            loaded_config.hardware.enabled
        );
        assert_eq!(
            original_config.inference.enabled,
            loaded_config.inference.enabled
        );
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        assert!(config.validate().is_ok());

        // Test invalid configurations
        config.hardware.collection_interval_ms = 0;
        assert!(config.validate().is_err());

        config.hardware.collection_interval_ms = 1000;
        config.inference.max_concurrent_requests = 0;
        assert!(config.validate().is_err());

        config.inference.max_concurrent_requests = 1000;
        config.model_probe.sampling_rate = 1.5;
        assert!(config.validate().is_err());
    }
}
