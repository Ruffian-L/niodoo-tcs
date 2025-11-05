// Example: How to use the Niodoo-Feeling configuration system
// This demonstrates proper usage of config values instead of hardcoding

use anyhow::Result;
use niodoo_feeling::config::{AppConfig, PathConfig, TimingConfig};
use std::time::Duration;
use tracing::{info, warn};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("🌟 Niodoo-Feeling Configuration System Example 🌟");

    // ============================================================================
    // EXAMPLE 1: Loading Configuration
    // ============================================================================
    info!("\n📖 Example 1: Loading Configuration");

    // Load configuration based on environment
    let env = std::env::var("NIODOO_ENV").unwrap_or_else(|_| "development".to_string());
    let config_path = format!("config/{}.toml", env);

    info!("Loading config from: {}", config_path);
    let config = AppConfig::load_from_file(&config_path)?;

    info!("✅ Configuration loaded successfully");
    info!("   Environment: {}", env);
    info!("   Log level: {}", config.logging.level);

    // ============================================================================
    // EXAMPLE 2: Using PathConfig (NO hardcoded paths!)
    // ============================================================================
    info!("\n📁 Example 2: Using PathConfig");

    // ❌ WRONG: Hardcoded path
    // let db_path = "/var/lib/niodoo/data/knowledge.db";

    // ✅ RIGHT: Use PathConfig
    let db_path = config.paths.get_db_path("knowledge_graph.db");
    info!("Database path: {}", db_path.display());

    let model_path = config.paths.get_model_path("qwen3");
    info!("Model path: {}", model_path.display());

    let log_path = config.paths.get_log_path("niodoo.log");
    info!("Log path: {}", log_path.display());

    // Backup paths with automatic timestamps
    let backup_path = config.paths.get_backup_path("consciousness_state");
    info!("Backup path: {}", backup_path.display());

    // ============================================================================
    // EXAMPLE 3: Using TimingConfig (NO hardcoded durations!)
    // ============================================================================
    info!("\n⏱️  Example 3: Using TimingConfig");

    // ❌ WRONG: Hardcoded duration
    // let timeout = Duration::from_secs(30);

    // ✅ RIGHT: Use TimingConfig
    let api_timeout = Duration::from_millis(config.timing.api_timeout_ms);
    info!("API timeout: {:?}", api_timeout);

    // Get task-specific timeouts
    let critical_timeout = config.timing.get_task_timeout("critical");
    let normal_timeout = config.timing.get_task_timeout("normal");
    info!("Critical task timeout: {:?}", critical_timeout);
    info!("Normal task timeout: {:?}", normal_timeout);

    // Visualization frame duration (for FPS control)
    let frame_duration = config.timing.get_viz_frame_duration();
    info!("Viz frame duration: {:?} (~60 FPS)", frame_duration);

    // Retry delays with exponential backoff
    for attempt in 0..3 {
        let delay = config.timing.get_retry_delay(attempt);
        info!("Retry attempt {}: wait {:?}", attempt, delay);
    }

    // ============================================================================
    // EXAMPLE 4: Using Network Configuration (NO hardcoded URLs!)
    // ============================================================================
    info!("\n🌐 Example 4: Using Network Configuration");

    // ❌ WRONG: Hardcoded URL
    // let api_url = "http://localhost:11434";

    // ✅ RIGHT: Use NetworkConfig (from ApiConfig for now)
    let ollama_url = &config.api.ollama_url;
    info!("Ollama URL: {}", ollama_url);

    // WebSocket address
    // Note: NetworkConfig would be accessed via config.network if added to AppConfig
    info!("Architect endpoint: {}", config.qt.architect_endpoint);
    info!("Developer endpoint: {}", config.qt.developer_endpoint);

    // ============================================================================
    // EXAMPLE 5: Using Thresholds (NO hardcoded values!)
    // ============================================================================
    info!("\n🎯 Example 5: Using Thresholds");

    // ❌ WRONG: Hardcoded threshold
    // if confidence > 0.7 { ... }

    // ✅ RIGHT: Use configured threshold
    let test_confidence = 0.85;
    if test_confidence > config.consciousness.emotion_sensitivity {
        info!(
            "Emotion detected! (confidence: {:.2}, threshold: {:.2})",
            test_confidence, config.consciousness.emotion_sensitivity
        );
    }

    // Memory formation threshold
    info!(
        "Memory threshold: {}",
        config.consciousness.memory_threshold
    );

    // Pattern recognition
    info!(
        "Pattern sensitivity: {}",
        config.consciousness.pattern_sensitivity
    );

    // ============================================================================
    // EXAMPLE 6: Using Mathematical Constants (NO hardcoded values!)
    // ============================================================================
    info!("\n🔢 Example 6: Using Mathematical Constants");

    // ❌ WRONG: Hardcoded mathematical constant
    // let radius = 2.0;
    // let epsilon = 1e-6;

    // ✅ RIGHT: Use configured mathematical constants
    info!(
        "Torus major radius: {}",
        config.consciousness.default_torus_major_radius
    );
    info!(
        "Torus minor radius: {}",
        config.consciousness.default_torus_minor_radius
    );
    info!(
        "Parametric epsilon: {}",
        config.consciousness.parametric_epsilon
    );
    info!(
        "Gaussian kernel exponent: {}",
        config.consciousness.gaussian_kernel_exponent
    );

    // ============================================================================
    // EXAMPLE 7: Using Resource Limits (NO hardcoded limits!)
    // ============================================================================
    info!("\n💾 Example 7: Using Resource Limits");

    // ❌ WRONG: Hardcoded limits
    // let max_tokens = 200;
    // let max_history = 50;

    // ✅ RIGHT: Use configured limits
    info!("Max tokens per request: {}", config.models.max_tokens);
    info!("Max conversation history: {}", config.core.max_history);
    info!("Max context window: {}", config.models.context_window);

    // Performance limits
    info!("GPU usage target: {}%", config.performance.gpu_usage_target);
    info!(
        "Memory usage target: {}%",
        config.performance.memory_usage_target
    );

    // ============================================================================
    // EXAMPLE 8: Using Feature Flags (Enable/disable without recompiling!)
    // ============================================================================
    info!("\n🚩 Example 8: Using Feature Flags");

    // ❌ WRONG: Hardcoded feature check
    // #[cfg(feature = "consciousness")]

    // ✅ RIGHT: Use runtime feature flag
    if config.consciousness.enabled {
        info!("Consciousness processing enabled");

        if config.consciousness.reflection_enabled {
            info!("  - Reflection enabled");
        }
    } else {
        warn!("Consciousness processing disabled");
    }

    // RAG feature flag
    if config.rag.enabled {
        info!("RAG system enabled");
        info!("  - Chunk size: {}", config.rag.chunk_size);
        info!("  - Top-k: {}", config.rag.top_k);
    } else {
        info!("RAG system disabled");
    }

    // ============================================================================
    // EXAMPLE 9: Using Weights and Factors (NO hardcoded weights!)
    // ============================================================================
    info!("\n⚖️  Example 9: Using Weights and Factors");

    // ❌ WRONG: Hardcoded weights
    // let quality_weight = 0.6;
    // let confidence_weight = 0.4;

    // ✅ RIGHT: Use configured weights
    info!(
        "Quality metric weight: {}",
        config.consciousness.quality_score_metric_weight
    );
    info!(
        "Quality confidence weight: {}",
        config.consciousness.quality_score_confidence_weight
    );

    // Calculate weighted score
    let metric_value = 0.8;
    let confidence_value = 0.9;
    let quality_score = (metric_value * config.consciousness.quality_score_metric_weight as f64)
        + (confidence_value * config.consciousness.quality_score_confidence_weight as f64);
    info!("Calculated quality score: {:.2}", quality_score);

    // ============================================================================
    // EXAMPLE 10: Dynamic Configuration Updates
    // ============================================================================
    info!("\n🔄 Example 10: Dynamic Configuration Updates");

    // Create a mutable copy for demonstration
    let mut config_copy = config.clone();

    // Update configuration values
    info!(
        "Original emotion threshold: {}",
        config_copy.core.emotion_threshold
    );

    config_copy.set_value("core.emotion_threshold", toml::Value::Float(0.85))?;
    info!(
        "Updated emotion threshold: {}",
        config_copy.core.emotion_threshold
    );

    // Save updated configuration
    config_copy.save_to_file("config/custom.toml")?;
    info!("✅ Saved custom configuration");

    // ============================================================================
    // EXAMPLE 11: Environment Variable Overrides
    // ============================================================================
    info!("\n🌍 Example 11: Environment Variable Overrides");

    info!("Environment variables can override config values:");
    info!(
        "  NIODOO_LOG_LEVEL={}",
        std::env::var("NIODOO_LOG_LEVEL").unwrap_or_else(|_| "not set".to_string())
    );
    info!(
        "  NIODOO_MODEL_PATH={}",
        std::env::var("NIODOO_MODEL_PATH").unwrap_or_else(|_| "not set".to_string())
    );
    info!(
        "  NIODOO_OLLAMA_URL={}",
        std::env::var("NIODOO_OLLAMA_URL").unwrap_or_else(|_| "not set".to_string())
    );

    info!("\nTo override, set environment variables before running:");
    info!("  export NIODOO_LOG_LEVEL=DEBUG");
    info!("  export NIODOO_AGENTS_COUNT=100");
    info!("  cargo run");

    // ============================================================================
    // Summary
    // ============================================================================
    info!("\n✅ Configuration System Examples Complete!");
    info!("\nKey Takeaways:");
    info!("  1. NEVER hardcode values - always use config");
    info!("  2. Use PathConfig for all file paths");
    info!("  3. Use TimingConfig for all durations");
    info!("  4. Use thresholds, limits, and weights from config");
    info!("  5. Use feature flags for optional functionality");
    info!("  6. Environment variables override config files");
    info!("  7. Configuration is validated on load");
    info!("\nFor more info, see: config/README.md");

    Ok(())
}

// ============================================================================
// Helper Functions Demonstrating Config Usage
// ============================================================================

/// Example function: Process input with configured timeout
fn process_with_timeout(config: &AppConfig, input: &str) -> Result<String> {
    use std::time::Instant;

    let timeout = Duration::from_millis(config.timing.model_inference_timeout_ms);
    let start = Instant::now();

    // Simulate processing
    info!("Processing with timeout: {:?}", timeout);

    if start.elapsed() > timeout {
        return Err(anyhow::anyhow!("Processing timeout exceeded"));
    }

    Ok(format!("Processed: {}", input))
}

/// Example function: Check if value exceeds threshold
fn check_threshold(config: &AppConfig, value: f32, threshold_type: &str) -> bool {
    let threshold = match threshold_type {
        "emotion" => config.consciousness.emotion_sensitivity,
        "memory" => config.consciousness.memory_threshold,
        "pattern" => config.consciousness.pattern_sensitivity,
        _ => config.core.emotion_threshold,
    };

    value > threshold
}

/// Example function: Calculate weighted score using config weights
fn calculate_quality_score(config: &AppConfig, metric: f64, confidence: f64) -> f64 {
    let weighted_score = (metric * config.consciousness.quality_score_metric_weight as f64)
        + (confidence * config.consciousness.quality_score_confidence_weight as f64);

    weighted_score * config.consciousness.quality_score_factor as f64
}

/// Example function: Get retry delay with exponential backoff
fn get_retry_strategy(config: &AppConfig, attempt: u32) -> Duration {
    config.timing.get_retry_delay(attempt)
}

/// Example function: Build URL from config
fn build_api_url(config: &AppConfig, endpoint: &str) -> String {
    format!("{}/{}", config.api.ollama_url, endpoint)
}
