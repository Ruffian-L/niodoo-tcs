//! Baseline manager for Silicon Synapse
//!
//! This module implements the baseline manager that coordinates learning and detection modes,
//! manages the 24-hour learning window, and provides a unified interface for anomaly detection.

use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::info;

use super::detector::{Anomaly, AnomalyDetector, DetectorConfig};
use super::model::BaselineModel;
use crate::silicon_synapse::aggregation::AggregatedMetrics;

/// Baseline manager that coordinates learning and detection
pub struct BaselineManager {
    /// Anomaly detector instance
    detector: Arc<RwLock<AnomalyDetector>>,
    /// Configuration
    config: DetectorConfig,
    /// Whether the manager is running
    is_running: Arc<std::sync::atomic::AtomicBool>,
    /// Learning start time
    learning_start_time: Arc<RwLock<Option<SystemTime>>>,
    /// Current baseline model
    baseline_model: Arc<RwLock<Option<BaselineModel>>>,
}

impl BaselineManager {
    /// Create a new baseline manager
    pub fn new(config: DetectorConfig) -> Self {
        let detector = AnomalyDetector::new(config.clone());

        Self {
            detector: Arc::new(RwLock::new(detector)),
            config,
            is_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            learning_start_time: Arc::new(RwLock::new(None)),
            baseline_model: Arc::new(RwLock::new(None)),
        }
    }

    /// Start the baseline manager
    pub async fn start(&self) -> Result<(), String> {
        if self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("Baseline manager is already running".to_string());
        }

        info!("Starting baseline manager");
        self.is_running
            .store(true, std::sync::atomic::Ordering::Relaxed);

        // Start the detector
        {
            let mut detector = self.detector.write().await;
            detector.start().await?;
        }

        // Record learning start time
        {
            let mut start_time = self.learning_start_time.write().await;
            *start_time = Some(SystemTime::now());
        }

        // Start the learning/detection task
        self.start_learning_detection_task().await;

        Ok(())
    }

    /// Stop the baseline manager
    pub async fn stop(&self) -> Result<(), String> {
        if !self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }

        info!("Stopping baseline manager");
        self.is_running
            .store(false, std::sync::atomic::Ordering::Relaxed);

        // Stop the detector
        {
            let mut detector = self.detector.write().await;
            detector.stop().await?;
        }

        Ok(())
    }

    /// Process new metrics
    pub async fn process_metrics(&self, metrics: &AggregatedMetrics) -> Vec<Anomaly> {
        if !self.is_running.load(std::sync::atomic::Ordering::Relaxed) {
            return Vec::new();
        }

        let mut detector = self.detector.write().await;
        detector.process_metrics(metrics)
    }

    /// Start the learning/detection task
    async fn start_learning_detection_task(&self) {
        let detector = Arc::clone(&self.detector);
        let config = self.config.clone();
        let learning_start_time = Arc::clone(&self.learning_start_time);
        let baseline_model = Arc::clone(&self.baseline_model);
        let is_running = Arc::clone(&self.is_running);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Check every minute

            while is_running.load(std::sync::atomic::Ordering::Relaxed) {
                interval.tick().await;

                // Check if learning period is complete
                if let Some(start_time) = *learning_start_time.read().await {
                    let elapsed = SystemTime::now()
                        .duration_since(start_time)
                        .unwrap_or_default();

                    if elapsed >= Duration::from_secs(config.learning_duration_hours * 3600) {
                        // Learning period complete, switch to detection mode
                        {
                            let mut detector = detector.write().await;
                            if detector.is_learning_mode() {
                                detector.force_detection_mode();
                                info!("Learning period complete, switched to detection mode");
                            }
                        }

                        // Save baseline model
                        {
                            let detector = detector.read().await;
                            if let Some(baseline) = detector.get_baseline() {
                                let mut baseline_model = baseline_model.write().await;
                                *baseline_model = Some(baseline.clone());
                                info!("Baseline model saved");
                            }
                        }
                    }
                }
            }
        });
    }

    /// Get current learning progress
    pub async fn get_learning_progress(&self) -> LearningProgress {
        let start_time = *self.learning_start_time.read().await;
        let detector = self.detector.read().await;

        if let Some(start) = start_time {
            let elapsed = SystemTime::now().duration_since(start).unwrap_or_default();

            let total_duration = Duration::from_secs(self.config.learning_duration_hours * 3600);
            let progress_percent =
                (elapsed.as_secs() as f64 / total_duration.as_secs() as f64 * 100.0).min(100.0);

            LearningProgress {
                is_learning: detector.is_learning_mode(),
                progress_percent,
                samples_collected: detector.get_learning_sample_count(),
                estimated_completion: start + total_duration,
            }
        } else {
            LearningProgress {
                is_learning: false,
                progress_percent: 0.0,
                samples_collected: 0,
                estimated_completion: SystemTime::now(),
            }
        }
    }

    /// Get current baseline model
    pub async fn get_baseline_model(&self) -> Option<BaselineModel> {
        let baseline = self.baseline_model.read().await;
        baseline.clone()
    }

    /// Check if baseline is ready for detection
    pub async fn is_baseline_ready(&self) -> bool {
        let detector = self.detector.read().await;
        detector
            .get_baseline()
            .map(|b| b.is_ready_for_detection())
            .unwrap_or(false)
    }

    /// Force switch to detection mode (for testing)
    pub async fn force_detection_mode(&self) {
        let mut detector = self.detector.write().await;
        detector.force_detection_mode();
    }

    /// Get detector statistics
    pub async fn get_detector_stats(&self) -> DetectorStats {
        let detector = self.detector.read().await;

        DetectorStats {
            is_running: self.is_running.load(std::sync::atomic::Ordering::Relaxed),
            is_learning: detector.is_learning_mode(),
            learning_samples: detector.get_learning_sample_count(),
            has_baseline: detector.get_baseline().is_some(),
        }
    }
}

/// Learning progress information
#[derive(Debug, Clone)]
pub struct LearningProgress {
    /// Whether currently in learning mode
    pub is_learning: bool,
    /// Progress percentage (0.0 to 100.0)
    pub progress_percent: f64,
    /// Number of samples collected
    pub samples_collected: usize,
    /// Estimated completion time
    pub estimated_completion: SystemTime,
}

/// Detector statistics
#[derive(Debug, Clone)]
pub struct DetectorStats {
    /// Whether the detector is running
    pub is_running: bool,
    /// Whether in learning mode
    pub is_learning: bool,
    /// Number of learning samples collected
    pub learning_samples: usize,
    /// Whether baseline model exists
    pub has_baseline: bool,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            learning_duration_hours: 24,
            min_samples_for_baseline: 1000,
            univariate_threshold_sigma: 3.0,
            multivariate_correlation_threshold: 0.3,
            enable_multivariate_detection: true,
            enable_learning_mode: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silicon_synapse::aggregation::AggregatedMetrics;

    #[tokio::test]
    async fn test_baseline_manager_creation() {
        let config = DetectorConfig::default();
        let manager = BaselineManager::new(config);

        let stats = manager.get_detector_stats().await;
        assert!(!stats.is_running);
        assert!(!stats.is_learning);
    }

    #[tokio::test]
    async fn test_baseline_manager_start_stop() {
        let config = DetectorConfig {
            min_samples_for_baseline: 1,
            learning_duration_hours: 1, // Short duration for testing
            ..Default::default()
        };
        let manager = BaselineManager::new(config);

        assert!(manager.start().await.is_ok());

        // Give it a moment to start
        sleep(Duration::from_millis(100)).await;

        let stats = manager.get_detector_stats().await;
        assert!(stats.is_running);

        assert!(manager.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_learning_progress() {
        let config = DetectorConfig::default();
        let manager = BaselineManager::new(config);

        let progress = manager.get_learning_progress().await;
        assert!(!progress.is_learning);
        assert_eq!(progress.progress_percent, 0.0);
    }

    #[tokio::test]
    async fn test_force_detection_mode() {
        let config = DetectorConfig {
            min_samples_for_baseline: 1,
            ..Default::default()
        };
        let manager = BaselineManager::new(config);

        manager.start().await.unwrap();

        // Process one sample to build baseline
        let metrics = AggregatedMetrics::default();
        manager.process_metrics(&metrics).await;

        // Force detection mode
        manager.force_detection_mode().await;

        let stats = manager.get_detector_stats().await;
        assert!(!stats.is_learning);
    }
}
