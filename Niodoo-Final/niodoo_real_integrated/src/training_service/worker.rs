//! Training Worker
//!
//! Worker that polls the job queue, processes training jobs,
//! runs QLoRA training using LoRATrainer, and saves adapters.

use anyhow::{Context, Result};
use chrono::Utc;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use crate::training_service::adapter_storage::{AdapterMetadata, AdapterStorage};
use crate::training_service::job_queue::{JobQueue, JobStatus, TrainingJob};

/// Training worker that processes jobs from the queue
pub struct TrainingWorker {
    job_queue: Arc<JobQueue>,
    adapter_storage: Arc<AdapterStorage>,
    shutdown: Arc<tokio::sync::Notify>,
}

impl TrainingWorker {
    pub fn new(job_queue: Arc<JobQueue>, adapter_storage: Arc<AdapterStorage>) -> Self {
        Self {
            job_queue,
            adapter_storage,
            shutdown: Arc::new(tokio::sync::Notify::new()),
        }
    }

    /// Start the worker loop
    pub async fn run(&self) {
        info!("Training worker started");

        loop {
            tokio::select! {
                _ = self.shutdown.notified() => {
                    info!("Training worker received shutdown signal");
                    break;
                }
                _ = tokio::time::sleep(Duration::from_secs(1)) => {
                    if let Err(e) = self.process_next_job().await {
                        error!(error = %e, "Failed to process training job");
                    }
                }
            }
        }

        info!("Training worker stopped");
    }

    /// Process the next job from the queue
    async fn process_next_job(&self) -> Result<()> {
        let job = match self.job_queue.dequeue()? {
            Some(job) => job,
            None => return Ok(()), // No jobs available
        };

        info!(job_id = %job.job_id, "Processing training job");

        match &job.job_type {
            crate::training_service::job_queue::JobType::Rust(payload) => {
                self.process_rust_job(&job, payload).await
            }
            crate::training_service::job_queue::JobType::Python(_payload) => {
                warn!(job_id = %job.job_id, "Python training jobs not supported in Rust worker");
                self.job_queue.update_status(
                    &job.job_id,
                    JobStatus::Failed,
                    Some("Python training jobs not supported in Rust worker".to_string()),
                    None,
                )?;
                Ok(())
            }
        }
    }

    /// Process a Rust training job using Python QLoRA
    async fn process_rust_job(
        &self,
        job: &TrainingJob,
        payload: &crate::training_service::job_queue::RustTrainingPayload,
    ) -> Result<()> {
        // Convert Vec<Vec<f32>> to Vec<(Vec<f32>, Vec<f32>)>
        // Assuming samples are pairs of (input, target)
        let training_data: Vec<(Vec<f32>, Vec<f32>)> = if payload.samples.len() % 2 == 0 {
            payload
                .samples
                .chunks(2)
                .map(|chunk| {
                    if chunk.len() == 2 {
                        (chunk[0].clone(), chunk[1].clone())
                    } else {
                        // Fallback: duplicate first element
                        (chunk[0].clone(), chunk[0].clone())
                    }
                })
                .collect()
        } else {
            // Odd number of samples - pair them sequentially
            payload
                .samples
                .windows(2)
                .step_by(1)
                .map(|window| (window[0].clone(), window[1].clone()))
                .collect()
        };

        if training_data.is_empty() {
            self.job_queue.update_status(
                &job.job_id,
                JobStatus::Failed,
                Some("No valid training samples".to_string()),
                None,
            )?;
            return Ok(());
        }

        // Convert training data to JSONL format for Python QLoRA
        let train_file = std::env::temp_dir().join(format!("train_{}.jsonl", job.job_id));
        self.write_training_jsonl(&train_file, &training_data)
            .context("Failed to write training JSONL")?;

        // Determine Python script path
        let project_root = std::env::var("PROJECT_ROOT").unwrap_or_else(|_| {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .to_string_lossy()
                .to_string()
        });
        let python_script = PathBuf::from(&project_root)
            .join("niodoo-ai")
            .join("scripts")
            .join("train_from_service.py");

        // Output directory for adapters
        let output_dir = std::env::temp_dir().join(format!("adapters_{}", job.job_id));

        // Base model (configurable via env)
        let base_model = std::env::var("QLORA_BASE_MODEL")
            .unwrap_or_else(|_| "Qwen/Qwen2.5-Coder-7B-Instruct".to_string());

        // Run Python QLoRA training
        info!(
            job_id = %job.job_id,
            samples = training_data.len(),
            epochs = payload.epochs,
            "Starting Python QLoRA training"
        );

        let epochs = payload.epochs;
        let learning_rate = payload.learning_rate;

        let output_res = tokio::task::spawn_blocking({
            let python_script = python_script.clone();
            let train_file = train_file.clone();
            let output_dir = output_dir.clone();
            let base_model = base_model.clone();
            move || {
                std::process::Command::new("python3")
                    .arg(&python_script)
                    .arg("--train-file")
                    .arg(&train_file)
                    .arg("--output-dir")
                    .arg(&output_dir)
                    .arg("--base-model")
                    .arg(&base_model)
                    .arg("--epochs")
                    .arg(epochs.to_string())
                    .arg("--learning-rate")
                    .arg(learning_rate.to_string())
                    .arg("--lora-r")
                    .arg("64")
                    .arg("--lora-alpha")
                    .arg("16")
                    .arg("--max-seq-length")
                    .arg("2048")
                    .arg("--gradient-accumulation")
                    .arg("16")
                    .output()
            }
        })
        .await
        .context("Failed to spawn Python training subprocess")?;

        let output: std::process::Output =
            output_res.context("Python training process failed to run")?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let error_msg = format!(
                "Python QLoRA training failed: {}\nSTDOUT: {}",
                stderr, stdout
            );
            error!(job_id = %job.job_id, error = %error_msg, "Python QLoRA training failed");
            self.job_queue
                .update_status(&job.job_id, JobStatus::Failed, Some(error_msg), None)?;
            return Ok(());
        }

        info!(
            job_id = %job.job_id,
            samples = training_data.len(),
            epochs = payload.epochs,
            "Python QLoRA training completed"
        );

        // Adapters are saved in output_dir by Python script
        // Look for adapter files (typically adapter_model.safetensors or adapter_config.json indicates directory)
        let adapter_path = if output_dir.join("adapter_config.json").exists() {
            output_dir.clone()
        } else if output_dir.join("adapter_model.safetensors").exists() {
            output_dir.clone()
        } else {
            // Fallback: use output_dir as-is
            output_dir.clone()
        };

        // Create metadata
        let metadata = AdapterMetadata {
            version: job.job_id.clone(),
            timestamp: Utc::now(),
            adapter_type: "python_qlora".to_string(),
            sample_count: Some(training_data.len()),
            epochs: Some(payload.epochs),
            learning_rate: Some(payload.learning_rate),
            loss: None, // Python training doesn't return loss directly
            buffer_size: None,
            config_path: None,
        };

        // Save to versioned storage
        let versioned_path = self
            .adapter_storage
            .save_adapter(&adapter_path, metadata)
            .context("Failed to save adapter to versioned storage")?;

        // Clean up temp training file
        if let Err(e) = std::fs::remove_file(&train_file) {
            warn!(error = %e, path = %train_file.display(), "Failed to remove temp training file");
        }

        // Update job status
        self.job_queue.update_status(
            &job.job_id,
            JobStatus::Completed,
            None,
            Some(versioned_path.to_string_lossy().to_string()),
        )?;

        Ok(())
    }

    /// Write training data to JSONL format for Python QLoRA
    fn write_training_jsonl(
        &self,
        path: &std::path::Path,
        training_data: &[(Vec<f32>, Vec<f32>)],
    ) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)
            .with_context(|| format!("Failed to create training file: {}", path.display()))?;

        for (input, target) in training_data {
            // Convert f32 vectors to strings for instruction/input/output
            // For topology-aware training, we need instruction, input, output, and topology_features
            let instruction = "Learn from the given input-output pair";
            let input_text = format!(
                "Input: {}",
                input
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let output_text = format!(
                "Output: {}",
                target
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            // Use input as topology features (or could derive from topology analysis)
            let topology_features: Vec<f64> = input.iter().map(|&v| v as f64).collect();

            let record = serde_json::json!({
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "topology_features": topology_features,
            });

            writeln!(file, "{}", serde_json::to_string(&record)?)
                .with_context(|| format!("Failed to write record to {}", path.display()))?;
        }

        Ok(())
    }

    /// Signal the worker to shutdown
    pub fn shutdown(&self) {
        self.shutdown.notify_one();
    }
}
