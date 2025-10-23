//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! # Persistent Learning Harness
//!
//! This module provides a persistent learning harness for running continuous
//! learning routines with metrics collection and reporting.

use crate::learning_analytics::LearningMetrics;
use anyhow::Result;
use std::collections::VecDeque;
use std::path::Path;
use std::time::Duration;

/// Learning routine trait for persistent learning harness
pub trait LearningRoutine {
    /// Get unique identifier for this learning routine
    fn identifier(&self) -> &str;

    /// Execute one step of learning and return metrics
    fn step(&mut self) -> Result<LearningMetrics>;
}

/// Configuration for persistent learning harness
#[derive(Debug, Clone)]
pub struct HarnessConfig {
    /// Directory to store persistent data
    pub data_dir: std::path::PathBuf,
    /// Step interval between learning iterations
    pub step_interval: Duration,
    /// Persist metrics every N steps
    pub persist_every: u64,
    /// Maximum number of steps to run (None for unlimited)
    pub max_steps: Option<u64>,
    /// Maximum number of metrics to keep in history (prevents unbounded growth)
    pub max_history_size: usize,
}

impl HarnessConfig {
    /// Create new harness configuration
    pub fn new(data_dir: &Path, step_interval: Duration) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
            step_interval,
            persist_every: 100,
            max_steps: None,
            max_history_size: 10000,
        }
    }

    /// Set persistence frequency
    pub fn with_persist_every(mut self, persist_every: u64) -> Self {
        self.persist_every = persist_every;
        self
    }

    /// Set maximum steps
    pub fn with_max_steps(mut self, max_steps: Option<u64>) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Set maximum history size
    pub fn with_max_history_size(mut self, max_history_size: usize) -> Self {
        self.max_history_size = max_history_size;
        self
    }
}

/// Metrics reporter trait
pub trait MetricsReporter: Send + Sync {
    /// Report metrics for a learning step
    fn report(&self, step: u64, metrics: &LearningMetrics) -> Result<()>;
}

/// Console metrics reporter that prints to stdout
pub struct ConsoleMetricsReporter {
    report_every: u64,
}

impl ConsoleMetricsReporter {
    /// Create new console reporter
    pub fn new(report_every: u64) -> Self {
        Self { report_every }
    }
}

impl MetricsReporter for ConsoleMetricsReporter {
    fn report(&self, step: u64, metrics: &LearningMetrics) -> Result<()> {
        if step % self.report_every == 0 {
            println!(
                "Step {}: learning_rate={:.4}, retention={:.4}, progress={:.4}",
                step, metrics.learning_rate, metrics.retention_score, metrics.progress_score
            );
        }
        Ok(())
    }
}

/// Disk metrics reporter that saves to files
pub struct DiskMetricsReporter {
    data_dir: std::path::PathBuf,
}

impl DiskMetricsReporter {
    /// Create new disk reporter
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
        }
    }
}

impl MetricsReporter for DiskMetricsReporter {
    fn report(&self, step: u64, metrics: &LearningMetrics) -> Result<()> {
        // TODO: Implement disk persistence
        println!("Persisting metrics for step {} to disk", step);
        Ok(())
    }
}

/// Persistent learning harness that runs learning routines
pub struct PersistentLearningHarness<R: LearningRoutine, T: MetricsReporter> {
    config: HarnessConfig,
    routine: R,
    reporter: T,
    step_count: u64,
    metrics_history: VecDeque<LearningMetrics>,
}

impl<R: LearningRoutine, T: MetricsReporter> PersistentLearningHarness<R, T> {
    /// Create new persistent learning harness
    pub fn new(config: HarnessConfig, routine: R, reporter: T) -> Result<Self> {
        Ok(Self {
            config,
            routine,
            reporter,
            step_count: 0,
            metrics_history: VecDeque::new(),
        })
    }

    /// Run the learning harness
    pub fn run(&mut self, max_steps: Option<u64>) -> Result<()> {
        let max_steps = max_steps.or(self.config.max_steps);

        loop {
            // Check if we've reached max steps
            if let Some(max) = max_steps {
                if self.step_count >= max {
                    break;
                }
            }

            // Execute learning step
            let metrics = self.routine.step()?;
            self.metrics_history.push_back(metrics.clone());

            // Apply retention policy: keep only max_history_size most recent metrics
            if self.metrics_history.len() > self.config.max_history_size {
                self.metrics_history.pop_front();
            }

            // Report metrics
            self.reporter.report(self.step_count, &metrics)?;

            // Persist if needed
            if self.step_count % self.config.persist_every == 0 {
                self.persist_metrics()?;
            }

            self.step_count += 1;

            // Sleep between steps
            std::thread::sleep(self.config.step_interval);
        }

        Ok(())
    }

    /// Persist metrics to disk
    fn persist_metrics(&self) -> Result<()> {
        // TODO: Implement actual persistence
        println!("Persisting {} metrics to disk", self.metrics_history.len());
        Ok(())
    }

    /// Get current step count
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Get metrics history
    pub fn metrics_history(&mut self) -> &[LearningMetrics] {
        self.metrics_history.make_contiguous()
    }
}