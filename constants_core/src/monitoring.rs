// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

// Monitoring and Detection Constants
// Configuration defaults with environment-variable overrides for anomaly detection and monitoring systems
// See CONSTANTS_AUDIT_CLEAN.md for the distinction between derived constants and configuration defaults

use std::env;

/// Default multivariate correlation threshold for anomaly detection
/// This is a tunable deployment default and should be adjusted per environment.
/// Value of 0.8 represents "strong positive correlation" per statistical convention:
/// - Pearson r ≥ 0.7 is generally considered "strong" in behavioral sciences
/// - 0.8 provides a balance between sensitivity and specificity for production systems
/// - Adjust higher (e.g., 0.9) for stricter anomaly detection or lower (e.g., 0.6) for more sensitivity
const DEFAULT_MULTIVARIATE_CORRELATION_THRESHOLD: f64 = 0.8;

/// Default sigma threshold for univariate anomaly detection
/// Uses the 3-sigma rule based on normal distribution properties:
/// - Approximately 99.73% of values lie within 3 standard deviations of the mean
/// - Common statistical threshold for outlier detection in quality control
/// - Balances false positive rate with anomaly detection sensitivity
pub const DEFAULT_UNIVARIATE_SIGMA: f64 = 3.0;

/// Minimum valid sample count for statistical analysis
/// Must be >= 2, recommended >= 30 for reliable statistics
pub fn min_valid_sample_count() -> usize {
    env::var("NIODOO_MIN_VALID_SAMPLE_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30)
}

/// Minimum samples required for baseline establishment
pub fn min_samples_for_baseline() -> usize {
    env::var("NIODOO_MIN_SAMPLES_FOR_BASELINE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100)
}

/// Sigma threshold for univariate anomaly detection
/// Uses the 3-sigma rule based on normal distribution properties for outlier detection
pub fn univariate_threshold_sigma() -> f64 {
    env::var("NIODOO_UNIVARIATE_THRESHOLD_SIGMA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_UNIVARIATE_SIGMA)
}

/// Correlation threshold for multivariate anomaly detection
pub fn multivariate_correlation_threshold() -> f64 {
    env::var("NIODOO_MULTIVARIATE_CORRELATION_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MULTIVARIATE_CORRELATION_THRESHOLD)
}

/// Learning duration in hours for baseline establishment
pub fn learning_duration_hours() -> u64 {
    env::var("NIODOO_LEARNING_DURATION_HOURS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24)
}

// ============================================================================
// Consciousness Logger Fallback Configuration
// ============================================================================

/// Fallback log directory when primary logger initialization fails
pub fn fallback_log_directory() -> String {
    env::var("NIODOO_FALLBACK_LOG_DIR").unwrap_or_else(|_| "./fallback_logs".to_string())
}

/// Fallback max file size in MB for logger
pub fn fallback_max_file_size_mb() -> usize {
    env::var("NIODOO_FALLBACK_MAX_FILE_SIZE_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10)
}

/// Fallback max files retained for logger
pub fn fallback_max_files_retained() -> usize {
    env::var("NIODOO_FALLBACK_MAX_FILES_RETAINED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5)
}

/// Fallback enable compression flag for logger
pub fn fallback_enable_compression() -> bool {
    env::var("NIODOO_FALLBACK_ENABLE_COMPRESSION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(false)
}

/// Fallback rotation interval in hours for logger
pub fn fallback_rotation_interval_hours() -> u64 {
    env::var("NIODOO_FALLBACK_ROTATION_INTERVAL_HOURS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24)
}
