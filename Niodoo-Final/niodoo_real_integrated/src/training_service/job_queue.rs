//! Training Job Queue
//!
//! File-based job queue for training requests with JSON serialization,
//! job status tracking, and atomic operations.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Training job status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

/// Training job payload for Rust-based training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustTrainingPayload {
    pub samples: Vec<Vec<f32>>,
    pub epochs: usize,
    pub learning_rate: f32,
}

/// Training job payload for Python-based training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonTrainingPayload {
    pub buffer_path: String,
    pub config_path: String,
    pub base_adapter_path: String,
}

/// Training job type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "job_type")]
pub enum JobType {
    #[serde(rename = "rust")]
    Rust(RustTrainingPayload),
    #[serde(rename = "python")]
    Python(PythonTrainingPayload),
}

/// Training job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    pub job_id: String,
    #[serde(flatten)]
    pub job_type: JobType,
    pub status: JobStatus,
    pub created_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_path: Option<String>,
}

impl TrainingJob {
    pub fn new_rust(samples: Vec<Vec<f32>>, epochs: usize, learning_rate: f32) -> Self {
        Self {
            job_id: Uuid::new_v4().to_string(),
            job_type: JobType::Rust(RustTrainingPayload {
                samples,
                epochs,
                learning_rate,
            }),
            status: JobStatus::Pending,
            created_at: Utc::now(),
            completed_at: None,
            error: None,
            adapter_path: None,
        }
    }

    pub fn new_python(buffer_path: String, config_path: String, base_adapter_path: String) -> Self {
        Self {
            job_id: Uuid::new_v4().to_string(),
            job_type: JobType::Python(PythonTrainingPayload {
                buffer_path,
                config_path,
                base_adapter_path,
            }),
            status: JobStatus::Pending,
            created_at: Utc::now(),
            completed_at: None,
            error: None,
            adapter_path: None,
        }
    }
}

/// File-based job queue
pub struct JobQueue {
    queue_dir: PathBuf,
}

impl JobQueue {
    pub fn new(queue_dir: impl AsRef<Path>) -> Result<Self> {
        let queue_dir = queue_dir.as_ref().to_path_buf();
        fs::create_dir_all(&queue_dir).with_context(|| {
            format!("Failed to create queue directory: {}", queue_dir.display())
        })?;

        Ok(Self { queue_dir })
    }

    fn job_path(&self, job_id: &str) -> PathBuf {
        self.queue_dir.join(format!("{}.json", job_id))
    }

    /// Enqueue a new training job
    pub fn enqueue(&self, job: TrainingJob) -> Result<String> {
        let job_id = job.job_id.clone();
        let job_path = self.job_path(&job_id);

        // Atomic write: write to temp file, then rename
        let temp_path = self.queue_dir.join(format!(".{}.json.tmp", job_id));
        let job_json =
            serde_json::to_string_pretty(&job).context("Failed to serialize training job")?;
        fs::write(&temp_path, job_json)
            .with_context(|| format!("Failed to write job file: {}", temp_path.display()))?;
        fs::rename(&temp_path, &job_path)
            .with_context(|| format!("Failed to rename job file: {}", job_path.display()))?;

        info!(job_id = %job_id, "Enqueued training job");
        Ok(job_id)
    }

    /// Get a job by ID
    pub fn get(&self, job_id: &str) -> Result<Option<TrainingJob>> {
        let job_path = self.job_path(job_id);
        if !job_path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&job_path)
            .with_context(|| format!("Failed to read job file: {}", job_path.display()))?;
        let job: TrainingJob = serde_json::from_str(&content)
            .with_context(|| format!("Failed to deserialize job file: {}", job_path.display()))?;

        Ok(Some(job))
    }

    /// Update job status
    pub fn update_status(
        &self,
        job_id: &str,
        status: JobStatus,
        error: Option<String>,
        adapter_path: Option<String>,
    ) -> Result<()> {
        let mut job = self
            .get(job_id)?
            .ok_or_else(|| anyhow::anyhow!("Job not found: {}", job_id))?;

        job.status = status;
        if status == JobStatus::Completed || status == JobStatus::Failed {
            job.completed_at = Some(Utc::now());
        }
        if let Some(err) = error {
            job.error = Some(err);
        }
        if let Some(path) = adapter_path {
            job.adapter_path = Some(path);
        }

        let job_path = self.job_path(job_id);
        let temp_path = self.queue_dir.join(format!(".{}.json.tmp", job_id));
        let job_json =
            serde_json::to_string_pretty(&job).context("Failed to serialize training job")?;
        fs::write(&temp_path, job_json)
            .with_context(|| format!("Failed to write job file: {}", temp_path.display()))?;
        fs::rename(&temp_path, &job_path)
            .with_context(|| format!("Failed to rename job file: {}", job_path.display()))?;

        debug!(job_id = %job_id, ?status, "Updated job status");
        Ok(())
    }

    /// Get next pending job
    pub fn dequeue(&self) -> Result<Option<TrainingJob>> {
        let entries = fs::read_dir(&self.queue_dir).with_context(|| {
            format!(
                "Failed to read queue directory: {}",
                self.queue_dir.display()
            )
        })?;

        for entry in entries {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            // Skip temp files and non-JSON files
            if path.extension() != Some(std::ffi::OsStr::new("json"))
                || path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with('.'))
                    .unwrap_or(false)
            {
                continue;
            }

            let content = match fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "Failed to read job file, skipping");
                    continue;
                }
            };

            let job: TrainingJob = match serde_json::from_str(&content) {
                Ok(j) => j,
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "Failed to deserialize job file, skipping");
                    continue;
                }
            };

            if job.status == JobStatus::Pending {
                // Mark as processing
                self.update_status(&job.job_id, JobStatus::Processing, None, None)?;
                return Ok(Some(job));
            }
        }

        Ok(None)
    }

    /// List all jobs
    pub fn list_jobs(&self) -> Result<Vec<TrainingJob>> {
        let entries = fs::read_dir(&self.queue_dir).with_context(|| {
            format!(
                "Failed to read queue directory: {}",
                self.queue_dir.display()
            )
        })?;

        let mut jobs = Vec::new();
        for entry in entries {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            if path.extension() != Some(std::ffi::OsStr::new("json"))
                || path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with('.'))
                    .unwrap_or(false)
            {
                continue;
            }

            let content = match fs::read_to_string(&path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            if let Ok(job) = serde_json::from_str::<TrainingJob>(&content) {
                jobs.push(job);
            }
        }

        jobs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(jobs)
    }

    /// Delete a job file (after completion or failure)
    pub fn delete(&self, job_id: &str) -> Result<()> {
        let job_path = self.job_path(job_id);
        if job_path.exists() {
            fs::remove_file(&job_path)
                .with_context(|| format!("Failed to delete job file: {}", job_path.display()))?;
            debug!(job_id = %job_id, "Deleted job file");
        }
        Ok(())
    }
}
