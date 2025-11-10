//! Training Service Client
//!
//! Client for submitting training jobs, checking status, and retrieving adapter paths.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use tracing::{debug, error, warn};

/// Training service client
pub struct TrainingServiceClient {
    client: Client,
    base_url: String,
}

/// Training sample pair (input, target)
#[derive(Debug, Serialize)]
struct TrainingSample {
    input: Vec<f32>,
    target: Vec<f32>,
}

/// Request to submit a training job
#[derive(Debug, Serialize)]
struct SubmitJobRequest {
    samples: Vec<TrainingSample>,
    epochs: usize,
    learning_rate: f32,
}

/// Response for job submission
#[derive(Debug, Deserialize)]
struct SubmitJobResponse {
    job_id: String,
    status: String,
}

/// Job status response
#[derive(Debug, Deserialize)]
pub struct JobStatusResponse {
    pub job_id: String,
    pub status: String,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_path: Option<String>,
}

/// Adapter version information
#[derive(Debug, Deserialize)]
pub struct AdapterVersionInfo {
    pub version: String,
    pub path: String,
    pub timestamp: String,
    pub adapter_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss: Option<f32>,
}

/// Adapter list response
#[derive(Debug, Deserialize)]
struct AdapterListResponse {
    adapters: Vec<AdapterVersionInfo>,
}

/// Latest adapter response
#[derive(Debug, Deserialize)]
struct LatestAdapterResponse {
    path: String,
    version: String,
}

impl TrainingServiceClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.into(),
        }
    }

    /// Submit a training job
    pub async fn submit_training_job(
        &self,
        samples: Vec<(Vec<f32>, Vec<f32>)>,
        epochs: usize,
        learning_rate: f32,
    ) -> Result<String> {
        // Convert Vec<(Vec<f32>, Vec<f32>)> to Vec<TrainingSample>
        let samples_vec: Vec<TrainingSample> = samples
            .into_iter()
            .map(|(input, target)| TrainingSample { input, target })
            .collect();

        let request = SubmitJobRequest {
            samples: samples_vec,
            epochs,
            learning_rate,
        };

        let url = format!("{}/training/jobs", self.base_url);
        debug!(url = %url, samples = request.samples.len(), "Submitting training job");

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send training job request")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Training service returned error: {} - {}",
                status,
                text
            ));
        }

        let job_response: SubmitJobResponse = response
            .json()
            .await
            .context("Failed to parse job submission response")?;

        debug!(job_id = %job_response.job_id, "Training job submitted");
        Ok(job_response.job_id)
    }

    /// Check job status
    pub async fn check_job_status(&self, job_id: &str) -> Result<JobStatusResponse> {
        let url = format!("{}/training/jobs/{}", self.base_url, job_id);
        debug!(url = %url, "Checking job status");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send job status request")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Training service returned error: {} - {}",
                status,
                text
            ));
        }

        let status_response: JobStatusResponse = response
            .json()
            .await
            .context("Failed to parse job status response")?;

        Ok(status_response)
    }

    /// Get latest adapter path
    pub async fn get_latest_adapter_path(&self) -> Result<PathBuf> {
        let url = format!("{}/training/adapters/latest", self.base_url);
        debug!(url = %url, "Getting latest adapter path");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send latest adapter request")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Training service returned error: {} - {}",
                status,
                text
            ));
        }

        let adapter_response: LatestAdapterResponse = response
            .json()
            .await
            .context("Failed to parse latest adapter response")?;

        Ok(PathBuf::from(adapter_response.path))
    }

    /// List all adapter versions
    pub async fn list_adapter_versions(&self) -> Result<Vec<AdapterVersionInfo>> {
        let url = format!("{}/training/adapters", self.base_url);
        debug!(url = %url, "Listing adapter versions");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send adapter list request")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Training service returned error: {} - {}",
                status,
                text
            ));
        }

        let list_response: AdapterListResponse = response
            .json()
            .await
            .context("Failed to parse adapter list response")?;

        Ok(list_response.adapters)
    }
}

