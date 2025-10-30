// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Model internal state probing for Silicon Synapse
//!
//! This module implements model probing to extract internal model states
//! during inference, including softmax entropy and activation patterns.

use candle_core::Tensor;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

use crate::silicon_synapse::config::ModelProbeConfig as ConfigModelProbeConfig;
use crate::silicon_synapse::telemetry_bus::{TelemetryEvent, TelemetrySender};
use crate::silicon_synapse::SiliconSynapseError;

/// Model probe for extracting internal states
pub struct ModelProbe {
    config: ModelProbeConfig,
    telemetry_sender: TelemetrySender,
    hooked_layers: HashMap<String, LayerHook>,
    entropy_calculator: EntropyCalculator,
    activation_analyzer: ActivationAnalyzer,
    is_running: Arc<std::sync::atomic::AtomicBool>,
}

/// Configuration for model probing
#[derive(Debug, Clone)]
pub struct ModelProbeConfig {
    pub enabled: bool,
    pub probe_entropy: bool,
    pub probe_activations: bool,
    pub probe_attention: bool,
    pub probe_layer_outputs: bool,
    pub max_layers: usize,
    pub sampling_rate: f64,
    pub enable_sparsity_analysis: bool,
    pub enable_magnitude_analysis: bool,
    pub enable_distribution_analysis: bool,
}

impl Default for ModelProbeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            probe_entropy: true,
            probe_activations: true,
            probe_attention: false,
            probe_layer_outputs: false,
            max_layers: 10,
            sampling_rate: 0.1,
            enable_sparsity_analysis: true,
            enable_magnitude_analysis: true,
            enable_distribution_analysis: false,
        }
    }
}

impl From<ConfigModelProbeConfig> for ModelProbeConfig {
    fn from(config: ConfigModelProbeConfig) -> Self {
        Self {
            enabled: config.enabled,
            probe_entropy: config.probe_entropy,
            probe_activations: config.probe_activations,
            probe_attention: config.probe_attention,
            probe_layer_outputs: config.probe_layer_outputs,
            max_layers: config.max_layers,
            sampling_rate: config.sampling_rate,
            enable_sparsity_analysis: config.enable_sparsity_analysis,
            enable_magnitude_analysis: config.enable_magnitude_analysis,
            enable_distribution_analysis: config.enable_distribution_analysis,
        }
    }
}

/// Hook for capturing layer activations
#[derive(Debug, Clone)]
pub struct LayerHook {
    pub layer_name: String,
    pub layer_index: usize,
    pub activation_data: Vec<ActivationData>,
}

/// Activation data captured from a layer
#[derive(Debug, Clone)]
pub struct ActivationData {
    pub timestamp: Instant,
    pub tensor_shape: Vec<usize>,
    pub activation_values: Vec<f32>,
    pub entropy: Option<f32>,
    pub sparsity: Option<f32>,
    pub magnitude_mean: Option<f32>,
    pub magnitude_std: Option<f32>,
}

/// Model metrics data structure
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    pub timestamp: Instant,
    pub layer_index: usize,
    pub entropy: Option<f32>,
    pub activation_sparsity: Option<f32>,
    pub activation_magnitude_mean: Option<f32>,
    pub activation_magnitude_std: Option<f32>,
    pub attention_entropy: Option<f32>,
    pub attention_sparsity: Option<f32>,
}

/// Entropy calculator for softmax distributions
pub struct EntropyCalculator {
    enabled: bool,
}

/// Activation analyzer for pattern analysis
pub struct ActivationAnalyzer {
    enabled: bool,
    sparsity_enabled: bool,
    magnitude_enabled: bool,
    /// Distribution analysis flag - future feature for advanced pattern analysis
    #[allow(dead_code)]
    distribution_enabled: bool,
}

/// Trait for models that can be probed
pub trait ModelWithLayers {
    fn get_layer_count(&self) -> usize;
    fn get_layer_name(&self, index: usize) -> Option<String>;
    fn hook_layer(&mut self, index: usize, hook: LayerHook) -> Result<(), String>;
    fn unhook_layer(&mut self, index: usize) -> Result<(), String>;
    fn get_layer_output(&self, index: usize) -> Option<Tensor>;
}

impl ModelProbe {
    /// Create a new model probe
    pub fn new(
        config: ModelProbeConfig,
        telemetry_sender: TelemetrySender,
    ) -> Result<Self, SiliconSynapseError> {
        Ok(Self {
            config: config.clone(),
            telemetry_sender,
            hooked_layers: HashMap::new(),
            entropy_calculator: EntropyCalculator::new(config.probe_entropy),
            activation_analyzer: ActivationAnalyzer::new(
                config.enable_sparsity_analysis,
                config.enable_magnitude_analysis,
                config.enable_distribution_analysis,
            ),
            is_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Start the model probe
    pub async fn start(&mut self) -> Result<(), SiliconSynapseError> {
        if self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(SiliconSynapseError::Config(
                "Model probe is already running".to_string(),
            ));
        }

        info!("Starting model probe");
        self.is_running
            .store(true, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Stop the model probe
    pub async fn stop(&mut self) -> Result<(), SiliconSynapseError> {
        if !self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }

        info!("Stopping model probe");
        self.is_running
            .store(false, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Probe a model layer for internal states
    pub async fn probe_layer(
        &self,
        layer_index: usize,
        tensor: &Tensor,
    ) -> Result<ModelMetrics, SiliconSynapseError> {
        if !self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(SiliconSynapseError::Config(
                "Model probe is not running".to_string(),
            ));
        }

        let timestamp = Instant::now();

        // Calculate entropy if enabled
        let entropy = if self.config.probe_entropy {
            self.entropy_calculator.calculate_entropy(tensor).ok()
        } else {
            None
        };

        // Analyze activations if enabled
        let (activation_sparsity, activation_magnitude_mean, activation_magnitude_std) =
            if self.config.probe_activations {
                self.activation_analyzer.analyze_activations(tensor)
            } else {
                (None, None, None)
            };

        // Attention probing implementation
        // TODO: Implement attention pattern analysis with proper tensor extraction
        let (attention_entropy, attention_sparsity) = if self.config.probe_attention {
            // For now, return None until we implement proper tensor-to-activations conversion
            (None, None)
        } else {
            (None, None)
        };

        let metrics = ModelMetrics {
            timestamp,
            layer_index,
            entropy,
            activation_sparsity,
            activation_magnitude_mean,
            activation_magnitude_std,
            attention_entropy,
            attention_sparsity,
        };

        // Send telemetry event
        let event = TelemetryEvent::ModelMetrics {
            timestamp: metrics.timestamp,
            layer_index: metrics.layer_index,
            entropy: metrics.entropy,
            activation_sparsity: metrics.activation_sparsity,
            activation_magnitude_mean: metrics.activation_magnitude_mean,
            activation_magnitude_std: metrics.activation_magnitude_std,
        };

        if let Err(e) = self.telemetry_sender.try_send(event) {
            warn!("Failed to send model metrics event: {}", e);
        }

        Ok(metrics)
    }

    /// Hook a model layer for continuous monitoring
    pub fn hook_layer(
        &mut self,
        layer_index: usize,
        layer_name: String,
    ) -> Result<(), SiliconSynapseError> {
        if self.hooked_layers.len() >= self.config.max_layers {
            return Err(SiliconSynapseError::Config(
                "Maximum number of hooked layers exceeded".to_string(),
            ));
        }

        let hook = LayerHook {
            layer_name: layer_name.clone(),
            layer_index,
            activation_data: Vec::new(),
        };

        self.hooked_layers.insert(layer_name.clone(), hook);
        debug!("Hooked layer {} at index {}", layer_name, layer_index);

        Ok(())
    }

    /// Unhook a model layer
    pub fn unhook_layer(&mut self, layer_name: &str) -> Result<(), SiliconSynapseError> {
        if self.hooked_layers.remove(layer_name).is_some() {
            debug!("Unhooked layer {}", layer_name);
            Ok(())
        } else {
            Err(SiliconSynapseError::Config(format!(
                "Layer {} not found",
                layer_name
            )))
        }
    }

    /// Get hooked layers
    pub fn get_hooked_layers(&self) -> Vec<String> {
        self.hooked_layers.keys().cloned().collect()
    }

    /// Analyze attention patterns from layer activations
    fn analyze_attention_patterns(activations: &Option<Vec<f32>>) -> Result<(f32, f32), String> {
        let acts = activations
            .as_ref()
            .ok_or_else(|| "No activation data available".to_string())?;

        if acts.is_empty() {
            return Err("Empty activation data".to_string());
        }

        // Calculate attention entropy
        let sum: f32 = acts.iter().map(|x| x.abs()).sum();
        if sum == 0.0 {
            return Ok((0.0, 0.0));
        }

        // Normalize and calculate entropy
        let normalized: Vec<f32> = acts.iter().map(|x| x.abs() / sum).collect();
        let entropy: f32 = -normalized
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f32>();

        // Calculate sparsity (proportion of near-zero values)
        let threshold = 0.01 * sum / acts.len() as f32;
        let sparse_count = acts.iter().filter(|&&x| x.abs() < threshold).count();
        let sparsity = sparse_count as f32 / acts.len() as f32;

        Ok((entropy, sparsity))
    }
}

impl EntropyCalculator {
    /// Create a new entropy calculator
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Calculate entropy of a tensor (assuming softmax distribution)
    pub fn calculate_entropy(&self, tensor: &Tensor) -> Result<f32, String> {
        if !self.enabled {
            return Err("Entropy calculation is disabled".to_string());
        }

        // Convert tensor to Vec<f32> for processing
        let values = tensor
            .to_vec1::<f32>()
            .map_err(|e| format!("Failed to convert tensor: {}", e))?;

        if values.is_empty() {
            return Err("Empty tensor".to_string());
        }

        // Apply softmax normalization
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = values.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_values.iter().sum();

        if sum_exp == 0.0 {
            return Err("Invalid softmax distribution".to_string());
        }

        let softmax_values: Vec<f32> = exp_values.iter().map(|&x| x / sum_exp).collect();

        // Calculate entropy: H = -sum(p * log(p))
        let entropy = -softmax_values
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.log2())
            .sum::<f32>();

        Ok(entropy)
    }
}

impl ActivationAnalyzer {
    /// Create a new activation analyzer
    pub fn new(
        sparsity_enabled: bool,
        magnitude_enabled: bool,
        distribution_enabled: bool,
    ) -> Self {
        Self {
            enabled: sparsity_enabled || magnitude_enabled || distribution_enabled,
            sparsity_enabled,
            magnitude_enabled,
            distribution_enabled,
        }
    }

    /// Analyze activation patterns in a tensor
    pub fn analyze_activations(&self, tensor: &Tensor) -> (Option<f32>, Option<f32>, Option<f32>) {
        if !self.enabled {
            return (None, None, None);
        }

        let values = match tensor.to_vec1::<f32>() {
            Ok(v) => v,
            Err(_) => return (None, None, None),
        };

        if values.is_empty() {
            return (None, None, None);
        }

        let sparsity = if self.sparsity_enabled {
            Some(self.calculate_sparsity(&values))
        } else {
            None
        };

        let (magnitude_mean, magnitude_std) = if self.magnitude_enabled {
            let mean = self.calculate_mean(&values);
            let std = self.calculate_std(&values, mean);
            (Some(mean), Some(std))
        } else {
            (None, None)
        };

        (sparsity, magnitude_mean, magnitude_std)
    }

    /// Calculate sparsity (fraction of near-zero values)
    fn calculate_sparsity(&self, values: &[f32]) -> f32 {
        let threshold = 1e-6;
        let near_zero_count = values.iter().filter(|&&x| x.abs() < threshold).count();
        near_zero_count as f32 / values.len() as f32
    }

    /// Calculate mean of values
    fn calculate_mean(&self, values: &[f32]) -> f32 {
        values.iter().sum::<f32>() / values.len() as f32
    }

    /// Calculate standard deviation of values
    fn calculate_std(&self, values: &[f32], mean: f32) -> f32 {
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nvml_wrapper::Device;

    #[tokio::test]
    async fn test_model_probe_creation() {
        let config = ModelProbeConfig::default();
        let telemetry_bus = TelemetryBus::new(config.telemetry.clone()).unwrap();
        let telemetry_sender = telemetry_bus.sender();

        let probe = ModelProbe::new(config, telemetry_sender);
        assert!(probe.is_ok());
    }

    #[tokio::test]
    async fn test_model_probe_start_stop() {
        let config = ModelProbeConfig::default();
        let telemetry_bus = TelemetryBus::new(config.telemetry.clone()).unwrap();
        let telemetry_sender = telemetry_bus.sender();

        let mut probe = ModelProbe::new(config, telemetry_sender).unwrap();

        assert!(probe.start().await.is_ok());
        assert!(probe.stop().await.is_ok());
    }

    #[test]
    fn test_entropy_calculator() {
        let calculator = EntropyCalculator::new(true);

        // Test with a simple tensor
        let device = Device::Cpu;
        let tensor = Tensor::new(&[1.0, 2.0, 3.0], &device).unwrap();

        let entropy = calculator.calculate_entropy(&tensor);
        assert!(entropy.is_ok());
        assert!(entropy.unwrap() > 0.0);
    }

    #[test]
    fn test_activation_analyzer() {
        let analyzer = ActivationAnalyzer::new(true, true, true);

        // Test with a simple tensor
        let device = Device::Cpu;
        let tensor = Tensor::new(&[1.0, 2.0, 3.0, 0.0], &device).unwrap();

        let (sparsity, mean, std) = analyzer.analyze_activations(&tensor);

        assert!(sparsity.is_some());
        assert!(mean.is_some());
        assert!(std.is_some());

        assert!(sparsity.unwrap() > 0.0); // Should have some sparsity due to 0.0
        assert!(mean.unwrap() > 0.0);
        assert!(std.unwrap() > 0.0);
    }
}
