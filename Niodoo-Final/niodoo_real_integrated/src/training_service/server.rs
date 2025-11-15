//! Training Service HTTP Server
//!
//! FastAPI-style HTTP server using axum with endpoints for job submission,
//! status checks, adapter listing, and health checks.

#[cfg(feature = "svc")]
use anyhow::{Context, Result};
#[cfg(feature = "svc")]
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use crate::training_service::adapter_storage::{AdapterStorage, AdapterVersion};
use crate::training_service::job_queue::{JobQueue, JobStatus, RustTrainingPayload, TrainingJob};

/// Server state
#[cfg(feature = "svc")]
pub struct ServerState {
    job_queue: Arc<JobQueue>,
    adapter_storage: Arc<AdapterStorage>,
}

/// Training Service HTTP Server
#[cfg(feature = "svc")]
pub struct TrainingServiceServer {
    state: Arc<ServerState>,
    port: u16,
}

#[cfg(feature = "svc")]
impl TrainingServiceServer {
    pub fn new(job_queue: Arc<JobQueue>, adapter_storage: Arc<AdapterStorage>, port: u16) -> Self {
        Self {
            state: Arc::new(ServerState {
                job_queue,
                adapter_storage,
            }),
            port,
        }
    }

    /// Start the HTTP server
    pub async fn start(&self) -> Result<()> {
        let app = Router::new()
            .route("/training/jobs", post(submit_job_handler))
            .route("/training/jobs/:job_id", get(get_job_handler))
            .route("/training/adapters", get(list_adapters_handler))
            .route("/training/adapters/latest", get(get_latest_adapter_handler))
            .route("/health", get(health_handler))
            .route("/metrics", get(metrics_handler))
            .with_state(self.state.clone());

        let addr = format!("0.0.0.0:{}", self.port);
        info!(
            port = self.port,
            "Starting training service server on {}", addr
        );

        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .with_context(|| format!("Failed to bind training service server to {}", addr))?;

        axum::serve(listener, app)
            .await
            .context("Training service server error")?;

        Ok(())
    }
}

/// Training sample pair (input, target)
#[derive(Debug, Deserialize)]
#[cfg(feature = "svc")]
struct TrainingSample {
    input: Vec<f32>,
    target: Vec<f32>,
}

/// Request to submit a training job
#[derive(Debug, Deserialize)]
#[cfg(feature = "svc")]
struct SubmitJobRequest {
    samples: Vec<TrainingSample>,
    epochs: usize,
    learning_rate: f32,
}

/// Response for job submission
#[derive(Debug, Serialize)]
#[cfg(feature = "svc")]
struct SubmitJobResponse {
    job_id: String,
    status: String,
}

/// Response for job status
#[derive(Debug, Serialize)]
#[cfg(feature = "svc")]
struct JobStatusResponse {
    job_id: String,
    status: String,
    created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    completed_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    adapter_path: Option<String>,
}

/// Response for adapter list
#[derive(Debug, Serialize)]
#[cfg(feature = "svc")]
struct AdapterListResponse {
    adapters: Vec<AdapterVersionInfo>,
}

/// Adapter version information for API
#[derive(Debug, Serialize)]
#[cfg(feature = "svc")]
struct AdapterVersionInfo {
    version: String,
    path: String,
    timestamp: String,
    adapter_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    loss: Option<f32>,
}

/// Response for latest adapter
#[derive(Debug, Serialize)]
#[cfg(feature = "svc")]
struct LatestAdapterResponse {
    path: String,
    version: String,
}

#[cfg(feature = "svc")]
async fn submit_job_handler(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<SubmitJobRequest>,
) -> Result<Json<SubmitJobResponse>, (StatusCode, String)> {
    // Convert TrainingSample pairs to Vec<Vec<f32>> format expected by job queue
    let samples_vec: Vec<Vec<f32>> = request
        .samples
        .into_iter()
        .flat_map(|s| vec![s.input, s.target])
        .collect();

    let job = TrainingJob::new_rust(samples_vec, request.epochs, request.learning_rate);

    let job_id = state.job_queue.enqueue(job).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to enqueue job: {}", e),
        )
    })?;

    Ok(Json(SubmitJobResponse {
        job_id,
        status: "pending".to_string(),
    }))
}

#[cfg(feature = "svc")]
async fn get_job_handler(
    State(state): State<Arc<ServerState>>,
    Path(job_id): Path<String>,
) -> Result<Json<JobStatusResponse>, (StatusCode, String)> {
    let job = state
        .job_queue
        .get(&job_id)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to get job: {}", e),
            )
        })?
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Job not found: {}", job_id)))?;

    Ok(Json(JobStatusResponse {
        job_id: job.job_id.clone(),
        status: format!("{:?}", job.status).to_lowercase(),
        created_at: job.created_at.to_rfc3339(),
        completed_at: job.completed_at.map(|d| d.to_rfc3339()),
        error: job.error.clone(),
        adapter_path: job.adapter_path.clone(),
    }))
}

#[cfg(feature = "svc")]
async fn list_adapters_handler(
    State(state): State<Arc<ServerState>>,
) -> Result<Json<AdapterListResponse>, (StatusCode, String)> {
    let versions = state.adapter_storage.list_versions().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to list adapters: {}", e),
        )
    })?;

    let adapters: Vec<AdapterVersionInfo> = versions
        .into_iter()
        .map(|v| AdapterVersionInfo {
            version: v.version,
            path: v.path.to_string_lossy().to_string(),
            timestamp: v.metadata.timestamp.to_rfc3339(),
            adapter_type: v.metadata.adapter_type,
            sample_count: v.metadata.sample_count,
            loss: v.metadata.loss,
        })
        .collect();

    Ok(Json(AdapterListResponse { adapters }))
}

#[cfg(feature = "svc")]
async fn get_latest_adapter_handler(
    State(state): State<Arc<ServerState>>,
) -> Result<Json<LatestAdapterResponse>, (StatusCode, String)> {
    let latest_path = state
        .adapter_storage
        .get_latest()
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to get latest adapter: {}", e),
            )
        })?
        .ok_or_else(|| (StatusCode::NOT_FOUND, "No adapters available".to_string()))?;

    // Get version from path or metadata
    let version = latest_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    Ok(Json(LatestAdapterResponse {
        path: latest_path.to_string_lossy().to_string(),
        version,
    }))
}

#[cfg(feature = "svc")]
async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "training_service"
    }))
}

#[cfg(feature = "svc")]
async fn metrics_handler() -> Result<String, (StatusCode, String)> {
    // Return Prometheus-style metrics
    // For now, return basic metrics
    Ok(format!(
        "# HELP training_jobs_total Total number of training jobs\n\
         # TYPE training_jobs_total counter\n\
         training_jobs_total 0\n"
    ))
}
