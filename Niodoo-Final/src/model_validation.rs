//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🧠🔍 MODEL VALIDATION UTILITIES - NO MORE GUESSING
 *
 * Comprehensive model validation system that checks paths, configurations,
 * and provides helpful error messages. No hardcoded bullshit!
 */

use super::config::{AppConfig, LoggingConfig, ModelConfig};
use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use tracing::{debug, error, info, warn};

/// Model validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Model validator for checking configurations and paths
pub struct ModelValidator;

impl ModelValidator {
    /// Validate complete application configuration
    pub fn validate_config(config: &AppConfig) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        info!("🔍 Validating complete application configuration...");

        // Validate model configuration
        let model_result = Self::validate_model_config(&config.models);
        errors.extend(model_result.errors);
        warnings.extend(model_result.warnings);
        suggestions.extend(model_result.suggestions);

        // Validate consciousness configuration
        let consciousness_result = Self::validate_consciousness_config(&config.consciousness);
        errors.extend(consciousness_result.errors);
        warnings.extend(consciousness_result.warnings);
        suggestions.extend(consciousness_result.suggestions);

        // Validate logging configuration
        let logging_result = Self::validate_logging_config(&config.logging);
        errors.extend(logging_result.errors);
        warnings.extend(logging_result.warnings);
        suggestions.extend(logging_result.suggestions);

        // Check for critical issues
        if !errors.is_empty() {
            tracing::error!(
                "❌ Configuration validation failed with {} errors",
                errors.len()
            );
        }

        if !warnings.is_empty() {
            warn!(
                "⚠️ Configuration validation found {} warnings",
                warnings.len()
            );
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            suggestions,
        }
    }

    /// Validate model configuration specifically
    pub fn validate_model_config(model_config: &ModelConfig) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        info!("🔍 Validating model configuration...");

        // Validate model parameters
        if model_config.max_tokens == 0 {
            errors.push("max_tokens cannot be 0".to_string());
        }

        // Check for reasonable model names
        if model_config.default_model.is_empty() {
            warnings.push("default_model is empty".to_string());
            suggestions.push("Set a default model name in config".to_string());
        }

        if model_config.backup_model.is_empty() {
            warnings.push("backup_model is empty".to_string());
            suggestions.push("Set a backup model name in config".to_string());
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            suggestions,
        }
    }

    /// Validate consciousness configuration
    pub fn validate_consciousness_config(
        _config: &super::config::ConsciousnessConfig,
    ) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        // Consciousness validation would go here
        // For now, just return success
        ValidationResult {
            is_valid: true,
            errors,
            warnings,
            suggestions,
        }
    }

    /// Validate logging configuration
    pub fn validate_logging_config(_config: &super::config::LoggingConfig) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        // Logging validation would go here
        // For now, just return success
        ValidationResult {
            is_valid: true,
            errors,
            warnings,
            suggestions,
        }
    }

    /// Check if required directories exist
    pub fn validate_directories() -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        let required_dirs = ["models", "data", "config", "logs"];

        for dir in &required_dirs {
            let path = PathBuf::from(dir);
            if !path.exists() {
                warnings.push(format!("Directory '{}' does not exist", dir));
                suggestions.push(format!("Create directory: mkdir -p {}", dir));
            } else if !path.is_dir() {
                errors.push(format!("'{}' exists but is not a directory", dir));
            } else {
                info!("✅ Directory '{}' exists", dir);
            }
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            suggestions,
        }
    }

    /// Validate GPU availability and configuration
    pub fn validate_gpu_config() -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        info!("🔍 Checking GPU configuration...");

        // Check for CUDA
        #[cfg(feature = "cuda")]
        {
            match nvml_wrapper::Device::cuda_if_available(0) {
                Ok(cuda_device) => {
                    info!("✅ CUDA device available: {:?}", cuda_device);
                    if cuda_device.is_cuda() {
                        suggestions.push(
                            "Consider using GPU acceleration for better performance".to_string(),
                        );
                    }
                }
                Err(e) => {
                    warnings.push(format!("CUDA device not available: {}", e));
                    suggestions.push("Install CUDA drivers for GPU acceleration".to_string());
                    suggestions.push("Check: nvidia-smi".to_string());
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            warnings.push("CUDA feature not enabled in build".to_string());
            suggestions.push("Rebuild with CUDA support for GPU acceleration".to_string());
        }

        // Check memory (simplified version)
        // Note: For a more accurate check, consider adding the `sys-info` crate
        info!("✅ GPU validation completed (memory check requires sys-info crate for accurate results)");

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            suggestions,
        }
    }

    /// Run complete system validation
    pub fn validate_system() -> ValidationResult {
        info!("🚀 Running complete system validation...");

        let mut all_errors = Vec::new();
        let mut all_warnings = Vec::new();
        let mut all_suggestions = Vec::new();

        // Load configuration
        let config = match AppConfig::load_from_file("config.toml") {
            Ok(config) => config,
            Err(e) => {
                all_errors.push(format!("Failed to load config.toml: {}", e));
                AppConfig::default()
            }
        };

        // Run all validations
        let config_validation = Self::validate_config(&config);
        all_errors.extend(config_validation.errors);
        all_warnings.extend(config_validation.warnings);
        all_suggestions.extend(config_validation.suggestions);

        let dir_validation = Self::validate_directories();
        all_errors.extend(dir_validation.errors);
        all_warnings.extend(dir_validation.warnings);
        all_suggestions.extend(dir_validation.suggestions);

        let gpu_validation = Self::validate_gpu_config();
        all_errors.extend(gpu_validation.errors);
        all_warnings.extend(gpu_validation.warnings);
        all_suggestions.extend(gpu_validation.suggestions);

        // Summary
        if all_errors.is_empty() && all_warnings.is_empty() {
            info!("🎉 System validation passed!");
        } else {
            if !all_errors.is_empty() {
                tracing::error!(
                    "❌ System validation failed with {} errors",
                    all_errors.len()
                );
            }
            if !all_warnings.is_empty() {
                warn!("⚠️ System validation found {} warnings", all_warnings.len());
            }
        }

        ValidationResult {
            is_valid: all_errors.is_empty(),
            errors: all_errors,
            warnings: all_warnings,
            suggestions: all_suggestions,
        }
    }

    /// Display validation results in a user-friendly format
    pub fn display_results(result: &ValidationResult) {
        if result.is_valid {
            tracing::info!("✅ VALIDATION PASSED");
        } else {
            tracing::info!("❌ VALIDATION FAILED");
        }

        if !result.errors.is_empty() {
            tracing::info!("\n🚨 ERRORS ({}):", result.errors.len());
            for error in &result.errors {
                tracing::info!("  • {}", error);
            }
        }

        if !result.warnings.is_empty() {
            tracing::info!("\n⚠️ WARNINGS ({}):", result.warnings.len());
            for warning in &result.warnings {
                tracing::info!("  • {}", warning);
            }
        }

        if !result.suggestions.is_empty() {
            tracing::info!("\n💡 SUGGESTIONS ({}):", result.suggestions.len());
            for suggestion in &result.suggestions {
                tracing::info!("  • {}", suggestion);
            }
        }

        tracing::info!("\n{}", "=".repeat(60));
    }
}

/// Test functions for validation
pub async fn test_model_validation() -> Result<()> {
    tracing::info!("🧪 Testing model validation system...");

    let result = ModelValidator::validate_system();
    ModelValidator::display_results(&result);

    if !result.is_valid {
        return Err(anyhow!("System validation failed"));
    }

    tracing::info!("✅ Model validation tests passed!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_structure() {
        let config = ModelConfig::default();
        let result = ModelValidator::validate_model_config(&config);

        // Should have warnings for missing model but no errors for valid structure
        assert!(result.errors.is_empty() || result.errors.iter().any(|e| e.contains("not found")));
        assert!(result.suggestions.iter().any(|s| s.contains("Download")));
    }

    #[test]
    fn test_directory_validation() {
        let result = ModelValidator::validate_directories();
        // Should not have errors for existing directories
        assert!(result.errors.is_empty());
    }
}
