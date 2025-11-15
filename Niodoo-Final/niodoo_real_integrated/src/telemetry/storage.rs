//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Test Run Storage and Replay
//!
//! Handles persistence and replay of test runs.

use crate::pipeline::generation::topo_reasoning::{TopoCotEvaluation, TopoCotScore};
use crate::telemetry::test_run::TestRun;
use anyhow::{Context, Result};
use blake3;
use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use uuid::Uuid;

const TOPOCOT_LOG_DIR: &str = "logs/topocot";
const TOPOCOT_EXCERPT_LIMIT: usize = 512;

/// Storage backend for test runs
pub struct TestRunStorage {
    base_dir: PathBuf,
}

impl TestRunStorage {
    /// Create a new test run storage with base directory
    pub fn new(base_dir: impl AsRef<Path>) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        fs::create_dir_all(&base_dir).with_context(|| {
            format!(
                "Failed to create test run storage directory: {:?}",
                base_dir
            )
        })?;
        Ok(Self { base_dir })
    }

    /// Get the path for a test run file
    fn test_run_path(&self, test_id: &Uuid) -> PathBuf {
        self.base_dir.join(format!("{}.json", test_id))
    }

    /// Save a test run to disk
    pub fn save_test_run(&self, test_run: &TestRun) -> Result<()> {
        let path = self.test_run_path(&test_run.test_id);
        let json = serde_json::to_string_pretty(test_run)
            .context("Failed to serialize test run to JSON")?;
        fs::write(&path, json)
            .with_context(|| format!("Failed to write test run to {:?}", path))?;
        Ok(())
    }

    /// Load a test run from disk
    pub fn load_test_run(&self, test_id: &Uuid) -> Result<TestRun> {
        let path = self.test_run_path(test_id);
        let json = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read test run from {:?}", path))?;
        let test_run: TestRun = serde_json::from_str(&json)
            .with_context(|| format!("Failed to deserialize test run from {:?}", path))?;
        Ok(test_run)
    }

    /// List all test run IDs
    pub fn list_test_runs(&self) -> Result<Vec<Uuid>> {
        let mut test_ids = Vec::new();
        let entries = fs::read_dir(&self.base_dir)
            .with_context(|| format!("Failed to read test run directory: {:?}", self.base_dir))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(uuid) = Uuid::parse_str(stem) {
                        test_ids.push(uuid);
                    }
                }
            }
        }

        Ok(test_ids)
    }

    /// Delete a test run
    pub fn delete_test_run(&self, test_id: &Uuid) -> Result<()> {
        let path = self.test_run_path(test_id);
        fs::remove_file(&path).with_context(|| format!("Failed to delete test run {:?}", path))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TopoCotLogEntry {
    pub entry_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub prompt_digest: String,
    pub response_digest: String,
    pub response_excerpt: String,
    pub grade_overall: f64,
    pub score: TopoCotScore,
    pub issues: Vec<String>,
    pub raw_json: Option<String>,
    pub thinking_depth: f64,
    pub pivot_score: f64,
    pub betti_numbers: [usize; 3],
    pub topology_mode: String,
    pub has_payload: bool,
    pub final_grounding_excerpt: Option<String>,
}

impl TopoCotLogEntry {
    pub fn from_evaluation(
        evaluation: &TopoCotEvaluation,
        prompt: &str,
        response: &str,
        thinking_depth: f64,
        pivot_score: f64,
        betti_numbers: [usize; 3],
        topology_mode: &str,
    ) -> Self {
        let timestamp = Utc::now();
        let prompt_digest = blake3::hash(prompt.as_bytes()).to_hex().to_string();
        let response_digest = blake3::hash(response.as_bytes()).to_hex().to_string();
        let response_excerpt = truncate_to(response, TOPOCOT_EXCERPT_LIMIT);
        let final_grounding_excerpt = evaluation.payload.as_ref().map(|payload| {
            truncate_to(
                &payload.step_4_final_output_grounding,
                TOPOCOT_EXCERPT_LIMIT,
            )
        });

        Self {
            entry_id: Uuid::new_v4(),
            timestamp,
            prompt_digest,
            response_digest,
            response_excerpt,
            grade_overall: evaluation.score.overall,
            score: evaluation.score.clone(),
            issues: evaluation.issues.clone(),
            raw_json: evaluation.raw_json.clone(),
            thinking_depth,
            pivot_score,
            betti_numbers,
            topology_mode: topology_mode.to_string(),
            has_payload: evaluation.payload.is_some(),
            final_grounding_excerpt,
        }
    }
}

pub fn append_topocot_log(entry: &TopoCotLogEntry) -> Result<()> {
    fs::create_dir_all(TOPOCOT_LOG_DIR).with_context(|| {
        format!(
            "Failed to create TopoCoT log directory at {}",
            TOPOCOT_LOG_DIR
        )
    })?;

    let date_tag = entry.timestamp.format("%Y%m%d").to_string();
    let path = PathBuf::from(TOPOCOT_LOG_DIR).join(format!("topocot_{date_tag}.jsonl"));
    let json = serde_json::to_string(entry).context("Failed to serialise TopoCoT log entry")?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("Failed to open TopoCoT log file {:?}", path))?;
    writeln!(file, "{json}")
        .with_context(|| format!("Failed to write TopoCoT log entry to {:?}", path))?;
    Ok(())
}

fn truncate_to(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        text.to_string()
    } else {
        text.chars().take(limit).collect::<String>()
    }
}

/// Replay a test run by iterating through its packets
pub struct TestRunReplayer {
    test_run: TestRun,
    current_index: usize,
}

impl TestRunReplayer {
    /// Create a new replayer for a test run
    pub fn new(test_run: TestRun) -> Self {
        Self {
            test_run,
            current_index: 0,
        }
    }

    /// Get the next packet in the test run
    pub fn next(&mut self) -> Option<&crate::telemetry::EnhancedCognitiveStatePacket> {
        if self.current_index < self.test_run.iterations.len() {
            let packet = &self.test_run.iterations[self.current_index];
            self.current_index += 1;
            Some(packet)
        } else {
            None
        }
    }

    /// Reset to the beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Get the current index
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// Get the total number of iterations
    pub fn total_iterations(&self) -> usize {
        self.test_run.iterations.len()
    }

    /// Check if there are more packets
    pub fn has_next(&self) -> bool {
        self.current_index < self.test_run.iterations.len()
    }
}

/// Compare two test runs
pub struct TestRunComparator;

impl TestRunComparator {
    /// Compare two test runs and return differences
    pub fn compare(run1: &TestRun, run2: &TestRun) -> TestRunDiff {
        let mut diff = TestRunDiff {
            iteration_count_diff: run1.iterations.len() as i64 - run2.iterations.len() as i64,
            latency_diff: run1.average_latency_ms() - run2.average_latency_ms(),
            metric_diffs: HashMap::new(),
            status_diff: format!("{:?} vs {:?}", run1.status, run2.status),
        };

        // Compare evaluation metrics if both have evaluations
        if let (Some(eval1), Some(eval2)) = (&run1.evaluation, &run2.evaluation) {
            for (key, val1) in &eval1.metrics {
                if let Some(val2) = eval2.metrics.get(key) {
                    diff.metric_diffs.insert(key.clone(), val1 - val2);
                }
            }
        }

        diff
    }
}

/// Differences between two test runs
#[derive(Debug, Clone)]
pub struct TestRunDiff {
    pub iteration_count_diff: i64,
    pub latency_diff: f64,
    pub metric_diffs: std::collections::HashMap<String, f64>,
    pub status_diff: String,
}
