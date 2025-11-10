//! Integration tests for the persistent learning harness.

use anyhow::Result;
use niodoo_consciousness::learning_analytics::LearningMetrics;
use niodoo_consciousness::persistent_learning::{
    ConsoleMetricsReporter, DiskMetricsReporter, HarnessConfig, LearningRoutine, PersistentLearningHarness,
};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Default)]
struct DummyRoutine {
    step: u64,
}

impl LearningRoutine for DummyRoutine {
    fn identifier(&self) -> &str {
        "dummy-routine"
    }

    fn step(&mut self) -> Result<LearningMetrics> {
        self.step += 1;
        Ok(LearningMetrics {
            learning_rate: 0.1,
            retention_score: 0.5 + (self.step as f32 * 0.01).min(0.4),
            adaptation_effectiveness: 0.4,
            plasticity: 0.6,
            progress_score: (self.step as f32 / 10.0).min(1.0),
            forgetting_rate: 0.1,
            loss: 1.0 / (1.0 + self.step as f32),
        })
    }
}

#[test]
fn harness_runs_smoke() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let config = HarnessConfig::new(tmp.path(), Duration::from_millis(5))
        .with_persist_every(2)
        .with_max_steps(Some(5));

    let routine = DummyRoutine::default();
    let reporter = ConsoleMetricsReporter::new(2);
    let mut harness = PersistentLearningHarness::new(config, routine, reporter)?;

    harness.run(Some(4))?;

    assert!(harness.step_count() >= 4);
    assert!(!harness.metrics_history().is_empty());
    Ok(())
}

#[test]
fn disk_reporter_persists_metrics() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let reporter = DiskMetricsReporter::new(tmp.path());

    let metrics = LearningMetrics {
        learning_rate: 0.2,
        retention_score: 0.7,
        adaptation_effectiveness: 0.6,
        plasticity: 0.5,
        progress_score: 0.8,
        forgetting_rate: 0.05,
        loss: 0.3,
    };

    reporter.report(1, &metrics)?;
    reporter.report(2, &metrics)?;

    let log_path = PathBuf::from(tmp.path()).join("learning_metrics.jsonl");
    let contents = std::fs::read_to_string(&log_path)?;
    assert!(contents.lines().count() >= 2);
    assert!(contents.contains("\"step\":1"));
    Ok(())
}