//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🧮 SMART THRESHOLDS & TIMING UTILITIES 🧮
 *
 * This module provides dynamic threshold and timing calculations
 * based on system performance, Gaussian processes, and adaptive algorithms.
 * Eliminates hardcoded magic numbers throughout the codebase.
 */

use anyhow::Result;
use std::time::Duration;
use sysinfo::System;
use tracing::{debug, info};

/// Configuration for threshold calculation
#[derive(Debug, Clone)]
pub struct ThresholdConfig {
    /// Base confidence level for statistical calculations (0.95 = 95%)
    pub base_confidence: f64,
    /// Performance factor for adaptive thresholds (0.1-1.0)
    pub performance_factor: f64,
    /// Memory pressure factor for conservative thresholds (0.1-1.0)
    pub memory_pressure_factor: f64,
    /// Load factor for dynamic adjustment (0.1-2.0)
    pub load_factor: f64,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            base_confidence: 0.95,       // 95% confidence interval
            performance_factor: 0.8,     // 80% performance baseline
            memory_pressure_factor: 0.9, // 90% memory utilization baseline
            load_factor: 1.0,            // Normal load
        }
    }
}

/// Smart threshold calculator
pub struct ThresholdCalculator {
    system: System,
    config: ThresholdConfig,
}

impl ThresholdCalculator {
    /// Create a new threshold calculator
    pub fn new() -> Result<Self> {
        let mut system = System::new_all();
        system.refresh_all();

        Ok(Self {
            system,
            config: ThresholdConfig::default(),
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: ThresholdConfig) -> Result<Self> {
        let mut system = System::new_all();
        system.refresh_all();

        Ok(Self { system, config })
    }

    /// Get system load factor (0.1-2.0 based on CPU/memory usage)
    pub fn get_system_load_factor(&self) -> f64 {
        let cpu_usage = self.system.global_cpu_info().cpu_usage() as f64 / 100.0;
        let memory_usage =
            1.0 - (self.system.available_memory() as f64 / self.system.total_memory() as f64);

        // Combine CPU and memory usage with weighting
        let load_factor = (cpu_usage * 0.6) + (memory_usage * 0.4);

        // Scale to reasonable range (0.1-2.0)
        let scaled_load = 0.1 + (load_factor * 1.9);
        debug!(
            "System load factor: {:.3} (CPU: {:.1}%, Memory: {:.1}%)",
            scaled_load,
            cpu_usage * 100.0,
            memory_usage * 100.0
        );

        scaled_load
    }

    /// Get adaptive performance factor based on system capabilities
    pub fn get_performance_factor(&self) -> f64 {
        let logical_cores = self.system.cpus().len() as f64;
        let total_memory_gb = self.system.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

        // Base performance factor from cores and memory
        let base_perf = (logical_cores.log10() * 0.3) + (total_memory_gb.sqrt() * 0.1);

        // Adjust for current load
        let load_factor = self.get_system_load_factor();
        let adaptive_perf = base_perf * (2.0 - load_factor) * self.config.performance_factor;

        debug!(
            "Performance factor: {:.3} (cores: {}, memory: {:.1}GB, load: {:.3})",
            adaptive_perf, logical_cores, total_memory_gb, load_factor
        );

        adaptive_perf.clamp(0.1, 1.0)
    }

    /// Calculate confidence threshold using Gaussian process principles
    pub fn calculate_confidence_threshold(&self, base_confidence: f64) -> f64 {
        // Use 95% confidence interval as default if not specified
        let confidence = if base_confidence == 0.0 {
            self.config.base_confidence
        } else {
            base_confidence
        };

        // Adjust based on system performance and load
        let perf_factor = self.get_performance_factor();
        let load_factor = self.get_system_load_factor();

        // Lower confidence for high-load systems (more conservative)
        let adaptive_confidence = confidence * perf_factor * (2.0 - load_factor);

        debug!(
            "Confidence threshold: {:.3} (base: {:.3}, perf: {:.3}, load: {:.3})",
            adaptive_confidence, confidence, perf_factor, load_factor
        );

        adaptive_confidence.clamp(0.5, 0.99)
    }

    /// Calculate emotion threshold based on system state
    pub fn calculate_emotion_threshold(&self) -> f32 {
        let base_threshold = 0.7; // Default emotion sensitivity
        let adaptive_threshold = base_threshold * self.get_performance_factor();

        debug!("Emotion threshold: {:.3}", adaptive_threshold);
        adaptive_threshold.clamp(0.3, 0.9) as f32
    }

    /// Calculate memory threshold for consolidation
    pub fn calculate_memory_threshold(&self) -> f32 {
        let base_threshold = 0.6; // Default memory formation threshold
        let perf_factor = self.get_performance_factor();
        let memory_pressure =
            1.0 - (self.system.available_memory() as f64 / self.system.total_memory() as f64);

        let adaptive_threshold = base_threshold * perf_factor * (1.0 + memory_pressure * 0.2);

        debug!(
            "Memory threshold: {:.3} (perf: {:.3}, memory_pressure: {:.3})",
            adaptive_threshold, perf_factor, memory_pressure
        );
        adaptive_threshold.clamp(0.4, 0.8) as f32
    }

    /// Calculate pattern sensitivity threshold
    pub fn calculate_pattern_sensitivity(&self) -> f32 {
        let base_sensitivity = 0.7; // Default pattern recognition sensitivity
        let adaptive_sensitivity = base_sensitivity * self.get_performance_factor();

        debug!("Pattern sensitivity: {:.3}", adaptive_sensitivity);
        adaptive_sensitivity.clamp(0.5, 0.9) as f32
    }

    /// Calculate consciousness stability threshold
    pub fn calculate_stability_threshold(&self) -> f64 {
        let base_stability = 0.95; // Default 95% stability
        let adaptive_stability = base_stability * self.get_performance_factor();

        debug!("Stability threshold: {:.3}", adaptive_stability);
        adaptive_stability.clamp(0.8, 0.99)
    }

    /// Calculate timeout duration based on operation criticality
    pub fn calculate_timeout(&self, criticality: TimeoutCriticality) -> Duration {
        let base_duration = match criticality {
            TimeoutCriticality::Critical => Duration::from_millis(100),
            TimeoutCriticality::High => Duration::from_millis(500),
            TimeoutCriticality::Normal => Duration::from_secs(2),
            TimeoutCriticality::Low => Duration::from_secs(10),
        };

        // Adjust based on system performance
        let perf_factor = self.get_performance_factor();
        let load_factor = self.get_system_load_factor();

        // Scale timeout based on performance (faster systems can afford shorter timeouts)
        let scaled_duration =
            base_duration.as_millis() as f64 * perf_factor * (1.0 + load_factor * 0.5);
        let adaptive_duration = Duration::from_millis(scaled_duration as u64);

        debug!(
            "Timeout for {:?}: {:?} (base: {:?}, perf: {:.3}, load: {:.3})",
            criticality, adaptive_duration, base_duration, perf_factor, load_factor
        );

        adaptive_duration
    }

    /// Calculate retry delay with exponential backoff
    pub fn calculate_retry_delay(&self, attempt: u32) -> Duration {
        let base_delay = Duration::from_millis(100);
        let perf_factor = self.get_performance_factor();

        // Exponential backoff: delay = base_delay * (2^attempt) * perf_factor
        let exponential_factor = 2.0_f64.powi(attempt as i32);
        let scaled_delay = base_delay.as_millis() as f64 * exponential_factor * perf_factor;

        let delay = Duration::from_millis(scaled_delay as u64);

        debug!(
            "Retry delay for attempt {}: {:?} (perf: {:.3})",
            attempt, delay, perf_factor
        );
        delay
    }

    /// Print system information for debugging
    pub fn print_system_info(&self) {
        info!("🔧 System Information for Threshold Calculation:");
        info!(
            "  CPU Usage: {:.1}%",
            self.system.global_cpu_info().cpu_usage()
        );
        info!(
            "  Memory Usage: {:.1}%",
            (1.0 - (self.system.available_memory() as f64 / self.system.total_memory() as f64))
                * 100.0
        );
        info!("  Logical Cores: {}", self.system.cpus().len());
        info!(
            "  Total Memory: {:.1} GB",
            self.system.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0
        );
        info!("  Base Confidence: {}", self.config.base_confidence);
        info!("  Performance Factor: {:.3}", self.get_performance_factor());
        info!("  System Load Factor: {:.3}", self.get_system_load_factor());
    }
}

/// Timeout criticality levels for adaptive timeout calculation
#[derive(Debug, Clone, Copy)]
pub enum TimeoutCriticality {
    Critical, // 100ms baseline
    High,     // 500ms baseline
    Normal,   // 2s baseline
    Low,      // 10s baseline
}

/// Global threshold calculator instance
pub fn get_threshold_calculator() -> Result<ThresholdCalculator> {
    ThresholdCalculator::new()
}

/// Convenience functions for common threshold calculations
pub mod convenience {
    use super::*;

    /// Get emotion threshold
    pub fn emotion_threshold() -> f32 {
        get_threshold_calculator()
            .map(|calc| calc.calculate_emotion_threshold())
            .unwrap_or(0.7) // fallback to old default
    }

    /// Get memory threshold
    pub fn memory_threshold() -> f32 {
        get_threshold_calculator()
            .map(|calc| calc.calculate_memory_threshold())
            .unwrap_or(0.6) // fallback to old default
    }

    /// Get pattern sensitivity
    pub fn pattern_sensitivity() -> f32 {
        get_threshold_calculator()
            .map(|calc| calc.calculate_pattern_sensitivity())
            .unwrap_or(0.7) // fallback to old default
    }

    /// Get stability threshold
    pub fn stability_threshold() -> f64 {
        get_threshold_calculator()
            .map(|calc| calc.calculate_stability_threshold())
            .unwrap_or(0.95) // fallback to old default
    }

    /// Get confidence threshold
    pub fn confidence_threshold(base_confidence: f64) -> f64 {
        get_threshold_calculator()
            .map(|calc| calc.calculate_confidence_threshold(base_confidence))
            .unwrap_or(base_confidence) // fallback to input
    }

    /// Get timeout for criticality level
    pub fn timeout(criticality: TimeoutCriticality) -> Duration {
        get_threshold_calculator()
            .map(|calc| calc.calculate_timeout(criticality))
            .unwrap_or_else(|_| match criticality {
                TimeoutCriticality::Critical => Duration::from_millis(100),
                TimeoutCriticality::High => Duration::from_millis(500),
                TimeoutCriticality::Normal => Duration::from_secs(2),
                TimeoutCriticality::Low => Duration::from_secs(10),
            })
    }

    /// Get retry delay for attempt number
    pub fn retry_delay(attempt: u32) -> Duration {
        get_threshold_calculator()
            .map(|calc| calc.calculate_retry_delay(attempt))
            .unwrap_or(Duration::from_millis(100 * 2_u64.pow(attempt)))
    }

    /// Get memory vector threshold
    pub fn memory_vector() -> f32 {
        get_threshold_calculator()
            .map(|calc| calc.calculate_memory_threshold())
            .unwrap_or(0.6) // fallback to old default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_calculator_creation() {
        let calc = ThresholdCalculator::new();
        assert!(calc.is_ok());
    }

    #[test]
    fn test_system_factors() {
        let calc = ThresholdCalculator::new().unwrap();
        assert!(calc.get_system_load_factor() > 0.0);
        assert!(calc.get_performance_factor() > 0.0);
    }

    #[test]
    fn test_threshold_calculations() {
        let calc = ThresholdCalculator::new().unwrap();

        assert!(calc.calculate_emotion_threshold() > 0.0);
        assert!(calc.calculate_memory_threshold() > 0.0);
        assert!(calc.calculate_pattern_sensitivity() > 0.0);
        assert!(calc.calculate_stability_threshold() > 0.0);
        assert!(calc.calculate_confidence_threshold(0.95) > 0.0);
    }

    #[test]
    fn test_timeout_calculations() {
        let calc = ThresholdCalculator::new().unwrap();

        let critical_timeout = calc.calculate_timeout(TimeoutCriticality::Critical);
        let normal_timeout = calc.calculate_timeout(TimeoutCriticality::Normal);

        assert!(critical_timeout < normal_timeout);
        assert!(critical_timeout.as_millis() > 0);
        assert!(normal_timeout.as_secs() > 0);
    }

    #[test]
    fn test_retry_delay() {
        let calc = ThresholdCalculator::new().unwrap();

        let delay1 = calc.calculate_retry_delay(0);
        let delay2 = calc.calculate_retry_delay(1);
        let delay3 = calc.calculate_retry_delay(2);

        assert!(delay2 > delay1); // Should increase with attempts
        assert!(delay3 > delay2); // Should continue increasing
    }

    #[test]
    fn test_convenience_functions() {
        // These should not panic and return reasonable values
        let _emotion = convenience::emotion_threshold();
        let _memory = convenience::memory_threshold();
        let _pattern = convenience::pattern_sensitivity();
        let _stability = convenience::stability_threshold();
        let _confidence = convenience::confidence_threshold(0.95);
        let _timeout = convenience::timeout(TimeoutCriticality::Normal);
        let _retry = convenience::retry_delay(1);
    }
}
