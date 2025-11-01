//! Persistent learning harness utilities shared across integration tests and production tooling.
//!
//! Provides a trait for pluggable learning routines, configurable execution harness, and
//! metrics reporters that stream results to stdout or disk for later analysis.

use crate::learning_analytics::LearningMetrics;
use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::VecDeque;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
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
    pub data_dir: PathBuf,
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
            max_history_size: 10_000,
        }
    }

    /// Set persistence frequency
    pub fn with_persist_every(mut self, persist_every: u64) -> Self {
        self.persist_every = persist_every.max(1);
        self
    }

    /// Set maximum steps
    pub fn with_max_steps(mut self, max_steps: Option<u64>) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Set maximum history size
    pub fn with_max_history_size(mut self, max_history_size: usize) -> Self {
        self.max_history_size = max_history_size.max(1);
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
        Self {
            report_every: report_every.max(1),
        }
    }
}

impl MetricsReporter for ConsoleMetricsReporter {
    fn report(&self, step: u64, metrics: &LearningMetrics) -> Result<()> {
        if step % self.report_every == 0 {
            println!(
                "Step {} :: learning_rate={:.4} retention={:.4} progress={:.4} loss={:.4}",
                step,
                metrics.learning_rate,
                metrics.retention_score,
                metrics.progress_score,
                metrics.loss
            );
        }
        Ok(())
    }
}

/// Disk metrics reporter that appends JSON lines for each step
pub struct DiskMetricsReporter {
    data_dir: PathBuf,
}

impl DiskMetricsReporter {
    /// Create new disk reporter
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
        }
    }

    fn metrics_file(&self) -> PathBuf {
        self.data_dir.join("learning_metrics.jsonl")
    }
}

impl MetricsReporter for DiskMetricsReporter {
    fn report(&self, step: u64, metrics: &LearningMetrics) -> Result<()> {
        fs::create_dir_all(&self.data_dir).with_context(|| {
            format!(
                "failed to create metrics directory at {}",
                self.data_dir.display()
            )
        })?;

        #[derive(Serialize)]
        struct MetricsRecord<'a> {
            step: u64,
            timestamp: chrono::DateTime<chrono::Utc>,
            metrics: &'a LearningMetrics,
        }

        let record = MetricsRecord {
            step,
            timestamp: chrono::Utc::now(),
            metrics,
        };

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.metrics_file())
            .with_context(|| "failed to open metrics log for append")?;

        serde_json::to_writer(&mut file, &record).context("failed to serialize metrics record")?;
        file.write_all(b"\n")
            .context("failed to write newline to metrics log")?;
        file.flush().ok();
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
        fs::create_dir_all(&config.data_dir).with_context(|| {
            format!(
                "failed to create data directory at {}",
                config.data_dir.display()
            )
        })?;

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
            if let Some(max) = max_steps {
                if self.step_count >= max {
                    break;
                }
            }

            let metrics = self.routine.step()?;
            self.metrics_history.push_back(metrics.clone());

            if self.metrics_history.len() > self.config.max_history_size {
                self.metrics_history.pop_front();
            }

            self.reporter.report(self.step_count, &metrics)?;

            if self.step_count % self.config.persist_every == 0 {
                self.persist_metrics()?;
            }

            self.step_count += 1;
            std::thread::sleep(self.config.step_interval);
        }

        Ok(())
    }

    /// Persist metrics snapshot to disk for later analysis
    pub fn persist_metrics(&mut self) -> Result<()> {
        #[derive(Serialize)]
        struct Snapshot<'a> {
            routine: &'a str,
            step_count: u64,
            generated_at: chrono::DateTime<chrono::Utc>,
            history: &'a [LearningMetrics],
        }

        let history = self.metrics_history.make_contiguous();
        let snapshot = Snapshot {
            routine: self.routine.identifier(),
            step_count: self.step_count,
            generated_at: chrono::Utc::now(),
            history,
        };

        let path = self.config.data_dir.join("metrics_snapshot.json");
        let data =
            serde_json::to_vec_pretty(&snapshot).context("failed to serialize metrics snapshot")?;
        fs::write(&path, data)
            .with_context(|| format!("failed to write metrics snapshot to {}", path.display()))?;
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
