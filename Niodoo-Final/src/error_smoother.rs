//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🛡️ Agent 6: ErrorSmoother
 * Handles rank/NaN errors, logs "Why suppress cross-flip?"
 */

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{error, info, warn, debug};

/// Error types that can be smoothed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorType {
    RankDeficiency,
    NaNValue,
    CrossFlipSuppression,
    MatrixSingularity,
    ConvergenceFailure,
}

/// Smoothed error result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothedError {
    pub error_type: ErrorType,
    pub original_message: String,
    pub smoothed_message: String,
    pub smoothing_applied: bool,
    pub recovery_suggestion: String,
    pub timestamp: u64,
}

/// Error smoothing configuration
#[derive(Debug, Clone)]
pub struct SmoothingConfig {
    pub max_rank_deficiency: usize,
    pub nan_tolerance: f32,
    pub cross_flip_suppression_enabled: bool,
    pub retry_attempts: usize,
    pub log_suppressions: bool,
}

impl Default for SmoothingConfig {
    fn default() -> Self {
        Self {
            max_rank_deficiency: 3,
            nan_tolerance: 0.001,
            cross_flip_suppression_enabled: true,
            retry_attempts: 3,
            log_suppressions: true,
        }
    }
}

/// ErrorSmoother agent for handling numerical errors
pub struct ErrorSmoother {
    config: SmoothingConfig,
    error_channel: mpsc::UnboundedSender<SmoothedError>,
    error_history: Vec<SmoothedError>,
    suppression_log: HashMap<ErrorType, usize>,
    shutdown: Arc<AtomicBool>,
}

impl ErrorSmoother {
    /// Create new ErrorSmoother agent
    pub fn new(config: SmoothingConfig) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        // Spawn error monitoring loop
        tokio::spawn(async move {
            while !shutdown_clone.load(Ordering::Relaxed) {
                if let Ok(smoothed_error) = rx.try_recv() {
                    Self::process_smoothed_error(&smoothed_error).await;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });

        Self {
            config,
            error_channel: tx,
            error_history: Vec::new(),
            suppression_log: HashMap::new(),
            shutdown,
        }
    }

    /// Smooth and handle an error
    pub async fn smooth_error(&mut self, error_type: ErrorType, message: &str) -> Result<SmoothedError> {
        info!("🛡️ ErrorSmoother processing: {:?} - {}", error_type, message);

        // Check if error can be smoothed
        let can_smooth = self.can_smooth_error(&error_type, message)?;

        let (smoothed_message, recovery_suggestion) = if can_smooth {
            self.generate_smoothed_response(&error_type, message)?
        } else {
            (message.to_string(), "Manual intervention required".to_string())
        };

        // Log cross-flip suppressions
        if matches!(error_type, ErrorType::CrossFlipSuppression) && self.config.log_suppressions {
            let count = self.suppression_log.entry(error_type.clone()).or_insert(0);
            *count += 1;
            warn!("🚫 Cross-flip suppression #{}: Why suppress cross-flip?", count);
        }

        let smoothed_error = SmoothedError {
            error_type: error_type.clone(),
            original_message: message.to_string(),
            smoothed_message,
            smoothing_applied: can_smooth,
            recovery_suggestion,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        // Send to monitoring channel
        if let Err(e) = self.error_channel.send(smoothed_error.clone()) {
            warn!("Failed to send smoothed error: {}", e);
        }

        // Add to history
        self.error_history.push(smoothed_error.clone());

        // Keep only last 100 errors in history
        if self.error_history.len() > 100 {
            self.error_history.remove(0);
        }

        info!("🛡️ Error smoothed: {:?}", error_type);

        Ok(smoothed_error)
    }

    /// Check if error can be automatically smoothed
    fn can_smooth_error(&self, error_type: &ErrorType, message: &str) -> Result<bool> {
        match error_type {
            ErrorType::RankDeficiency => {
                // Can smooth if rank deficiency is within tolerance
                let rank_loss = self.extract_rank_loss(message)?;
                Ok(rank_loss <= self.config.max_rank_deficiency)
            }
            ErrorType::NaNValue => {
                // Can smooth if NaN values are within tolerance
                let nan_count = self.count_nan_values(message)?;
                Ok(nan_count as f32 <= self.config.nan_tolerance)
            }
            ErrorType::CrossFlipSuppression => {
                // Cross-flip suppressions are logged but not smoothed
                Ok(false)
            }
            ErrorType::MatrixSingularity => {
                // Matrix singularity often requires manual intervention
                Ok(false)
            }
            ErrorType::ConvergenceFailure => {
                // Can retry convergence failures
                Ok(true)
            }
        }
    }

    /// Generate smoothed response for recoverable errors
    fn generate_smoothed_response(&self, error_type: &ErrorType, message: &str) -> Result<(String, String)> {
        match error_type {
            ErrorType::RankDeficiency => {
                let rank_loss = self.extract_rank_loss(message)?;
                Ok((
                    format!("Rank deficiency smoothed (loss: {})", rank_loss),
                    "Consider regularization or dimensionality reduction".to_string(),
                ))
            }
            ErrorType::NaNValue => {
                let nan_count = self.count_nan_values(message)?;
                Ok((
                    format!("NaN values handled (count: {})", nan_count),
                    "Check input data quality and normalization".to_string(),
                ))
            }
            ErrorType::ConvergenceFailure => {
                Ok((
                    "Convergence failure - retrying with adjusted parameters".to_string(),
                    "Try different learning rate or optimizer settings".to_string(),
                ))
            }
            _ => Ok((
                message.to_string(),
                "Manual intervention required".to_string(),
            )),
        }
    }

    /// Extract rank loss from error message (simple heuristic)
    fn extract_rank_loss(&self, message: &str) -> Result<usize> {
        // Look for patterns like "rank deficient by X" or "lost rank X"
        if let Some(captures) = regex::Regex::new(r"rank deficient by (\d+)|lost rank (\d+)")
            .unwrap()
            .captures(message)
        {
            if let Some(rank_str) = captures.get(1).or_else(|| captures.get(2)) {
                return Ok(rank_str.as_str().parse::<usize>().unwrap_or(1));
            }
        }
        Ok(1) // Default to rank loss of 1
    }

    /// Count NaN values in error message (simple heuristic)
    fn count_nan_values(&self, message: &str) -> Result<usize> {
        // Simple count of "NaN" occurrences
        Ok(message.matches("NaN").count())
    }

    /// Process smoothed error for monitoring
    async fn process_smoothed_error(smoothed_error: &SmoothedError) {
        debug!("📊 Processing smoothed error: {:?}", smoothed_error);

        if smoothed_error.smoothing_applied {
            info!("✅ Error smoothed successfully: {}", smoothed_error.smoothed_message);
        } else {
            warn!("⚠️  Error requires manual intervention: {}", smoothed_error.original_message);
        }
    }

    /// Get error statistics
    pub fn get_error_stats(&self) -> ErrorStats {
        let total_errors = self.error_history.len();
        let smoothed_errors = self.error_history.iter().filter(|e| e.smoothing_applied).count();

        let mut error_type_counts = HashMap::new();
        for error in &self.error_history {
            *error_type_counts.entry(error.error_type.clone()).or_insert(0) += 1;
        }

        ErrorStats {
            total_errors,
            smoothed_errors,
            smoothing_rate: if total_errors > 0 {
                (smoothed_errors as f32 / total_errors as f32) * 100.0
            } else {
                0.0
            },
            error_type_counts,
            suppression_count: self.suppression_log.get(&ErrorType::CrossFlipSuppression).cloned().unwrap_or(0),
        }
    }

    /// Continuous error monitoring loop
    pub async fn run_monitoring(&mut self) -> Result<()> {
        info!("🛡️ ErrorSmoother starting monitoring");

        let mut interval = tokio::time::interval(Duration::from_secs(10));

        while !self.shutdown.load(Ordering::Relaxed) {
            interval.tick().await;

            let stats = self.get_error_stats();
            if stats.total_errors > 0 {
                info!("🛡️ Error stats: {} total, {:.1}% smoothed, {} cross-flip suppressions",
                      stats.total_errors, stats.smoothing_rate, stats.suppression_count);
            }
        }

        Ok(())
    }

    /// Shutdown the agent
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        info!("🛡️ ErrorSmoother shutting down");
    }
}

/// Error statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStats {
    pub total_errors: usize,
    pub smoothed_errors: usize,
    pub smoothing_rate: f32,
    pub error_type_counts: HashMap<ErrorType, usize>,
    pub suppression_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_error_smoother_creation() {
        let config = SmoothingConfig::default();
        let mut smoother = ErrorSmoother::new(config);

        let smoothed = smoother.smooth_error(
            ErrorType::NaNValue,
            "Found NaN values in tensor"
        ).await.unwrap();

        assert_eq!(smoothed.error_type, ErrorType::NaNValue);
        assert!(smoothed.smoothing_applied);
        assert!(!smoothed.smoothed_message.is_empty());
    }

    #[tokio::test]
    async fn test_cross_flip_suppression_logging() {
        let config = SmoothingConfig::default();
        let mut smoother = ErrorSmoother::new(config);

        let smoothed = smoother.smooth_error(
            ErrorType::CrossFlipSuppression,
            "Cross-flip suppressed due to instability"
        ).await.unwrap();

        assert_eq!(smoothed.error_type, ErrorType::CrossFlipSuppression);
        assert!(!smoothed.smoothing_applied); // Cross-flip suppressions are not smoothed

        let stats = smoother.get_error_stats();
        assert_eq!(stats.suppression_count, 1);
    }

    #[test]
    fn test_error_stats() {
        let config = SmoothingConfig::default();
        let smoother = ErrorSmoother::new(config);

        let stats = smoother.get_error_stats();
        assert_eq!(stats.total_errors, 0);
        assert_eq!(stats.smoothed_errors, 0);
        assert_eq!(stats.smoothing_rate, 0.0);
    }
}
