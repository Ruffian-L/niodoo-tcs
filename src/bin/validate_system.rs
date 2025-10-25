//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
use tracing::{info, error, warn};
 * 🧠💖 SYSTEM VALIDATION BINARY
 *
 * Rust binary for validating the complete Niodoo system
 * Used by validate_system.sh script
 */

use anyhow::Result;
use niodoo_consciousness::{
    config::AppConfig,
    model_validation::{ModelValidator, ValidationResult},
};

fn main() -> Result<()> {
    tracing::info!("🔍 NIODOO SYSTEM VALIDATION (Rust Binary)");
    tracing::info!("=========================================");

    // Load configuration
    let config = match AppConfig::load_from_file("config.toml") {
        Ok(config) => {
            tracing::info!("✅ Loaded configuration from config.toml");
            config
        }
        Err(e) => {
            tracing::info!("⚠️ Failed to load config.toml: {}", e);
            tracing::info!("Using default configuration...");
            AppConfig::default()
        }
    };

    // Run comprehensive validation
    let result = ModelValidator::validate_system();

    // Display results
    ModelValidator::display_results(&result);

    // Exit with appropriate code
    if result.is_valid {
        tracing::info!("✅ System validation successful!");
        std::process::exit(0);
    } else {
        tracing::info!("❌ System validation failed!");
        std::process::exit(1);
    }
}

