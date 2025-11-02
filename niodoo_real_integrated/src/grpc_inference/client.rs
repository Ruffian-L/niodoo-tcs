#![cfg(feature = "svc")]
//! gRPC Inference Client for ONNX Model Serving
//! 
//! Provides client-side gRPC interface for distributed inference

use anyhow::{Context, Result};
use tonic::transport::Channel;
use tracing::{error, info};

use crate::grpc_inference::server::proto::{
    onnx_inference_service_client::OnnxInferenceServiceClient,
    InferenceRequest, InferenceResponse, LoadModelRequest, LoadModelResponse,
    HealthCheckRequest, HealthCheckResponse, BatchInferenceRequest, BatchInferenceResponse,
};

/// gRPC Inference Client
pub struct OnnxInferenceClient {
    client: OnnxInferenceServiceClient<Channel>,
}

impl OnnxInferenceClient {
    /// Create a new inference client
    pub async fn new(addr: String) -> Result<Self> {
        let client = OnnxInferenceServiceClient::connect(addr.clone())
            .await
            .with_context(|| format!("Failed to connect to gRPC server at {}", addr))?;

        info!(addr = %addr, "Connected to gRPC ONNX Inference Server");

        Ok(Self { client })
    }

    /// Load a model on the server
    pub async fn load_model(
        &mut self,
        model_name: String,
        model_path: String,
    ) -> Result<LoadModelResponse> {
        let request = LoadModelRequest {
            model_name,
            model_path,
            default_options: None,
        };

        let response = self
            .client
            .load_model(request)
            .await
            .context("Failed to load model")?
            .into_inner();

        if response.success {
            info!("Model loaded successfully");
        } else {
            error!(error = %response.error_message, "Failed to load model");
        }

        Ok(response)
    }

    /// Run inference
    pub async fn infer(&mut self, request: InferenceRequest) -> Result<InferenceResponse> {
        Ok(self
            .client
            .infer(request)
            .await
            .context("Failed to run inference")?
            .into_inner())
    }

    /// Run batch inference
    pub async fn batch_infer(
        &mut self,
        request: BatchInferenceRequest,
    ) -> Result<BatchInferenceResponse> {
        Ok(self
            .client
            .batch_infer(request)
            .await
            .context("Failed to run batch inference")?
            .into_inner())
    }

    /// Health check
    pub async fn health_check(&mut self) -> Result<HealthCheckResponse> {
        Ok(self
            .client
            .check(HealthCheckRequest {
                service: "onnx_inference".to_string(),
            })
            .await
            .context("Failed to check health")?
            .into_inner())
    }
}

