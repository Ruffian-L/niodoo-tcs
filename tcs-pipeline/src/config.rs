//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Configuration parameters for the Topological Cognitive System pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCSConfig {
    pub takens_dimension: usize,
    pub takens_delay: usize,
    pub takens_data_dim: usize,
    pub homology_max_dimension: usize,
    pub homology_max_edge_length: f32,
    pub jones_cache_capacity: usize,
    pub consensus_threshold: f32,
    pub tqft_algebra_dimension: usize,
    pub persistence_event_threshold: f32,
    pub feature_sampling_limit: usize,
    pub knot_complexity_threshold: f32,
    pub default_resonance: f32,
    pub default_coherence: f32,
    pub enable_tqft_checks: bool,
}

impl Default for TCSConfig {
    fn default() -> Self {
        Self {
            takens_dimension: 3,
            takens_delay: 2,
            takens_data_dim: 512,
            homology_max_dimension: 2,
            homology_max_edge_length: 2.5,
            jones_cache_capacity: 256,
            consensus_threshold: 0.8,
            tqft_algebra_dimension: 2,
            persistence_event_threshold: 0.1,
            feature_sampling_limit: 3,
            knot_complexity_threshold: 1.0,
            default_resonance: 0.6,
            default_coherence: 0.7,
            enable_tqft_checks: true,
        }
    }
}

impl TCSConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)
            .with_context(|| format!("failed to read config file {}", path.display()))?;
        let config = toml::from_str(&content)
            .with_context(|| format!("failed to parse config file {}", path.display()))?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn defaults_align_with_previous_values() {
        let config = TCSConfig::default();
        assert_eq!(config.takens_dimension, 3);
        assert!((config.homology_max_edge_length - 2.5).abs() < f32::EPSILON);
        assert!(config.enable_tqft_checks);
    }

    #[test]
    fn config_serialization() {
        let config = TCSConfig::default();
        let serialized = toml::to_string(&config);
        assert!(serialized.is_ok());
    }

    #[test]
    fn config_deserialization() {
        let toml_str = r#"
            takens_dimension = 5
            takens_delay = 3
            takens_data_dim = 768
            homology_max_dimension = 3
            homology_max_edge_length = 3.0
            jones_cache_capacity = 512
            consensus_threshold = 0.85
            tqft_algebra_dimension = 4
            persistence_event_threshold = 0.15
            feature_sampling_limit = 5
            knot_complexity_threshold = 1.5
            default_resonance = 0.7
            default_coherence = 0.8
            enable_tqft_checks = false
        "#;

        let config: Result<TCSConfig, _> = toml::from_str(toml_str);
        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.takens_dimension, 5);
        assert_eq!(config.homology_max_dimension, 3);
        assert!(!config.enable_tqft_checks);
    }

    #[test]
    fn config_from_file() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");

        let config = TCSConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        fs::write(&config_path, toml_str).unwrap();

        let loaded_config = TCSConfig::from_file(&config_path);
        assert!(loaded_config.is_ok());

        let loaded = loaded_config.unwrap();
        assert_eq!(loaded.takens_dimension, config.takens_dimension);
        assert_eq!(loaded.consensus_threshold, config.consensus_threshold);
    }

    #[test]
    fn config_invalid_file() {
        let result = TCSConfig::from_file("/nonexistent/path/config.toml");
        assert!(result.is_err());
    }
}
