//! HTTP server for RL Execution Harness
//!
//! This module exposes an HTTP API endpoint that the Python RL trainer can call
//! to evaluate generated code and get rewards.

#[cfg(feature = "svc")]
use crate::rl_harness::{ExecutionHarness, RewardWeights, TrainingProblem};
use crate::config::CodeLanguage;
use anyhow::{Context, Result};
use axum::{
    extract::Json,
    http::StatusCode,
    response::Json as ResponseJson,
    routing::post,
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

#[cfg(feature = "svc")]
use crate::sandbox::manager::SandboxManager;
#[cfg(feature = "svc")]
use crate::tcs_analysis::TCSAnalyzer;

/// Request payload from Python RL trainer
#[derive(Debug, Deserialize)]
pub struct EvaluateRequest {
    pub code: String,
    pub language: String,
    pub problem: ProblemPayload,
}

/// Problem payload in request
#[derive(Debug, Deserialize)]
pub struct ProblemPayload {
    pub id: String,
    pub description: String,
    #[serde(default)]
    pub test_cases: Option<Vec<String>>,
}

/// Response payload with reward breakdown
#[derive(Debug, Serialize)]
pub struct EvaluateResponse {
    pub functional: f64,
    pub cqs: f64,
    pub topological: f64,
    pub total: f64,
}

/// Create HTTP router for RL harness endpoints
#[cfg(feature = "svc")]
pub fn create_rl_harness_router(
    harness: Arc<ExecutionHarness>,
) -> Router {
    Router::new().route("/rl/evaluate", post(evaluate_code_handler))
        .with_state(harness)
}

/// Handler for code evaluation endpoint
#[cfg(feature = "svc")]
async fn evaluate_code_handler(
    axum::extract::State(harness): axum::extract::State<Arc<ExecutionHarness>>,
    Json(request): Json<EvaluateRequest>,
) -> Result<ResponseJson<EvaluateResponse>, StatusCode> {
    info!(
        problem_id = %request.problem.id,
        language = %request.language,
        "Received code evaluation request"
    );

    // Parse language
    let language = match request.language.as_str() {
        "python" => CodeLanguage::Python,
        "typescript" | "ts" => CodeLanguage::TypeScript,
        _ => {
            warn!(language = %request.language, "Unknown language, defaulting to Python");
            CodeLanguage::Python
        }
    };

    // Create training problem
    let problem = TrainingProblem {
        id: request.problem.id,
        description: request.problem.description,
        language,
        test_cases: request.problem.test_cases,
        expected_output: None,
    };

    // Evaluate code
    let reward = harness
        .evaluate_code(&request.code, language, &problem)
        .await
        .map_err(|e| {
            warn!(error = %e, "Failed to evaluate code");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(ResponseJson(EvaluateResponse {
        functional: reward.functional,
        cqs: reward.cqs,
        topological: reward.topological,
        total: reward.total,
    }))
}

/// Create execution harness with default configuration
#[cfg(feature = "svc")]
pub async fn create_harness(
    sandbox_manager: Arc<SandboxManager>,
    tcs_analyzer: Option<Arc<Mutex<TCSAnalyzer>>>,
) -> Result<Arc<ExecutionHarness>> {
    let reward_weights = RewardWeights::default();
    let harness = ExecutionHarness::new(sandbox_manager, tcs_analyzer, reward_weights)
        .context("Failed to create execution harness")?;
    Ok(Arc::new(harness))
}


