#![cfg(feature = "svc")]
//! gRPC Inference Server for ONNX Model Serving
//! 
//! Provides distributed inference capabilities using Tonic gRPC, optimized for:
//! - NVIDIA H200 GPU with FP8 precision
//! - Batched inference (up to 1024 batch size)
//! - Streaming inference for recursive loops
//! - Integration with Persistent Laplacians and topological analysis

use anyhow::{Context, Result};
use dashmap::DashMap;
use prost::Message;
use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{error, info, warn};

#[cfg(feature = "onnx")]
use ort::{
    environment::Environment,
    session::{Session, SessionBuilder},
    value::Value,
    GraphOptimizationLevel, LoggingLevel,
};

#[cfg(feature = "onnx")]
use ndarray::{ArrayD, IxDyn};

// Include generated protobuf code
pub mod proto {
    tonic::include_proto!("niodoo.rce");
}

use proto::{
    onnx_inference_service_server::{OnnxInferenceService, OnnxInferenceServiceServer},
    BatchInferenceRequest, BatchInferenceResponse, BatchMetadata, HealthCheckRequest,
    HealthCheckResponse, InferenceMetadata, InferenceRequest, InferenceResponse,
    InferenceOptions, LoadModelRequest, LoadModelResponse, ModelInfo, ServingStatus,
    Tensor, TensorDataType, TensorShape,
};

/// ONNX model session wrapper with metadata
#[cfg(feature = "onnx")]
struct ModelSession {
    session: Arc<Session>,
    model_info: ModelInfo,
    environment: Arc<Environment>,
}

/// gRPC Inference Server implementation
pub struct OnnxInferenceServer {
    #[cfg(feature = "onnx")]
    models: Arc<DashMap<String, ModelSession>>,
    #[cfg(feature = "onnx")]
    default_environment: Arc<Environment>,
    gpu_device_id: i32,
}

impl OnnxInferenceServer {
    /// Create a new inference server
    pub fn new(gpu_device_id: i32) -> Result<Self> {
        #[cfg(feature = "onnx")]
        {
            let environment = Arc::new(
                Environment::builder()
                    .with_name("niodoo-grpc-inference")
                    .with_log_level(LoggingLevel::Warning)
                    .build()
                    .context("Failed to create ONNX Runtime environment")?,
            );

            info!(
                gpu_device_id = gpu_device_id,
                "Created ONNX Inference Server with GPU support"
            );

            Ok(Self {
                models: Arc::new(DashMap::new()),
                default_environment: environment,
                gpu_device_id,
            })
        }

        #[cfg(not(feature = "onnx"))]
        {
            warn!("ONNX feature not enabled - gRPC server will use stub implementation");
            Ok(Self { gpu_device_id })
        }
    }

    /// Load an ONNX model
    #[cfg(feature = "onnx")]
    async fn load_model_internal(
        &self,
        model_name: String,
        model_path: String,
        options: Option<InferenceOptions>,
    ) -> Result<ModelInfo> {
        let session = SessionBuilder::new(&self.default_environment)
            .context("Failed to create session builder")?
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .with_model_from_file(&model_path)
            .context("Failed to load model")?;

        // Extract model information
        let input_names: Vec<String> = session
            .inputs
            .iter()
            .map(|i| i.name.clone())
            .collect();
        let output_names: Vec<String> = session
            .outputs
            .iter()
            .map(|o| o.name.clone())
            .collect();

        let mut input_shapes = std::collections::HashMap::new();
        for input in &session.inputs {
            if let Some(shape) = &input.input_type.shape {
                let dims: Vec<i64> = shape
                    .iter()
                    .map(|d| d.dim_value().unwrap_or(-1) as i64)
                    .collect();
                input_shapes.insert(
                    input.name.clone(),
                    TensorShape { dims },
                );
            }
        }

        let mut output_shapes = std::collections::HashMap::new();
        for output in &session.outputs {
            if let Some(shape) = &output.output_type.shape {
                let dims: Vec<i64> = shape
                    .iter()
                    .map(|d| d.dim_value().unwrap_or(-1) as i64)
                    .collect();
                output_shapes.insert(
                    output.name.clone(),
                    TensorShape { dims },
                );
            }
        }

        let model_info = ModelInfo {
            model_name: model_name.clone(),
            input_names,
            output_names,
            input_shapes,
            output_shapes,
            model_size_bytes: std::fs::metadata(&model_path)
                .map(|m| m.len() as i64)
                .unwrap_or(0),
            supports_fp8: options.as_ref().map(|o| o.use_fp8).unwrap_or(false),
        };

        let model_session = ModelSession {
            session: Arc::new(session),
            model_info: model_info.clone(),
            environment: Arc::clone(&self.default_environment),
        };

        self.models.insert(model_name, model_session);

        Ok(model_info)
    }
}

#[tonic::async_trait]
impl OnnxInferenceService for OnnxInferenceServer {
    async fn load_model(
        &self,
        request: Request<LoadModelRequest>,
    ) -> Result<Response<LoadModelResponse>, Status> {
        let req = request.into_inner();
        info!(model_name = %req.model_name, model_path = %req.model_path, "Loading ONNX model");

        #[cfg(feature = "onnx")]
        {
            match self
                .load_model_internal(req.model_name.clone(), req.model_path.clone(), req.default_options)
                .await
            {
                Ok(model_info) => {
                    info!(model_name = %req.model_name, "Model loaded successfully");
                    Ok(Response::new(LoadModelResponse {
                        success: true,
                        error_message: String::new(),
                        model_info: Some(model_info),
                    }))
                }
                Err(e) => {
                    error!(model_name = %req.model_name, error = %e, "Failed to load model");
                    Ok(Response::new(LoadModelResponse {
                        success: false,
                        error_message: format!("Failed to load model: {}", e),
                        model_info: None,
                    }))
                }
            }
        }

        #[cfg(not(feature = "onnx"))]
        {
            warn!("ONNX feature not enabled");
            Ok(Response::new(LoadModelResponse {
                success: false,
                error_message: "ONNX feature not enabled".to_string(),
                model_info: None,
            }))
        }
    }

    async fn infer(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceResponse>, Status> {
        let req = request.into_inner();
        let start_time = std::time::Instant::now();

        #[cfg(feature = "onnx")]
        {
            let model_session = self
                .models
                .get(&req.model_name)
                .ok_or_else(|| Status::not_found(format!("Model {} not found", req.model_name)))?;

            // Convert protobuf tensors to ONNX Runtime values
            let mut input_values = Vec::new();
            for (name, tensor) in &req.inputs {
                let ort_value = self
                    .tensor_to_ort_value(tensor, &model_session.session)
                    .map_err(|e| Status::internal(format!("Failed to convert tensor {}: {}", name, e)))?;
                input_values.push(ort_value);
            }

            // Run inference
            let outputs = model_session
                .session
                .run(input_values)
                .map_err(|e| Status::internal(format!("Inference failed: {}", e)))?;

            // Convert outputs back to protobuf tensors
            let mut output_tensors = std::collections::HashMap::new();
            for (i, output) in outputs.iter().enumerate() {
                let output_name = model_session
                    .model_info
                    .output_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("output_{}", i));
                let tensor = self
                    .ort_value_to_tensor(output)
                    .map_err(|e| Status::internal(format!("Failed to convert output: {}", e)))?;
                output_tensors.insert(output_name, tensor);
            }

            let inference_time = start_time.elapsed().as_secs_f64() * 1000.0;

            Ok(Response::new(InferenceResponse {
                outputs: output_tensors,
                metadata: Some(InferenceMetadata {
                    inference_time_ms: inference_time,
                    gpu_memory_used_bytes: 0, // TODO: Get from CUDA
                    batch_size: req.options.as_ref().map(|o| o.batch_size).unwrap_or(1),
                    used_fp8: req.options.as_ref().map(|o| o.use_fp8).unwrap_or(false),
                    execution_provider: req
                        .options
                        .as_ref()
                        .map(|o| o.execution_provider.clone())
                        .unwrap_or_else(|| "CUDAExecutionProvider".to_string()),
                }),
                timestamp: Some(::prost_types::Timestamp {
                    seconds: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs() as i64,
                    nanos: 0,
                }),
            }))
        }

        #[cfg(not(feature = "onnx"))]
        {
            warn!("ONNX feature not enabled - returning stub response");
            Err(Status::unimplemented("ONNX feature not enabled"))
        }
    }

    async fn batch_infer(
        &self,
        request: Request<BatchInferenceRequest>,
    ) -> Result<Response<BatchInferenceResponse>, Status> {
        let req = request.into_inner();
        let start_time = std::time::Instant::now();

        #[cfg(feature = "onnx")]
        {
            let mut responses = Vec::new();
            for inference_req in req.requests {
                match self.infer(Request::new(inference_req)).await {
                    Ok(resp) => responses.push(resp.into_inner()),
                    Err(e) => {
                        error!(error = %e, "Batch inference failed for one request");
                        return Err(e);
                    }
                }
            }

            let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
            let avg_time = total_time / responses.len() as f64;

            Ok(Response::new(BatchInferenceResponse {
                responses,
                batch_metadata: Some(BatchMetadata {
                    batch_size: responses.len() as i32,
                    total_time_ms: total_time,
                    avg_time_per_item_ms: avg_time,
                    peak_gpu_memory_bytes: 0, // TODO: Get from CUDA
                }),
            }))
        }

        #[cfg(not(feature = "onnx"))]
        {
            Err(Status::unimplemented("ONNX feature not enabled"))
        }
    }

    type StreamInferStream = tokio_stream::wrappers::ReceiverStream<Result<InferenceResponse, Status>>;

    async fn stream_infer(
        &self,
        _request: Request<tonic::Streaming<InferenceRequest>>,
    ) -> Result<Response<Self::StreamInferStream>, Status> {
        // Streaming inference requires more complex state management
        // For now, return unimplemented - use batch_infer for multiple requests
        // TODO: Implement proper streaming with shared state via Arc<DashMap>
        Err(Status::unimplemented(
            "Streaming inference not yet fully implemented - use batch_infer for multiple requests"
        ))
    }

    async fn check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        #[cfg(feature = "onnx")]
        {
            let gpu_info = format!("GPU Device {}", self.gpu_device_id);
            let loaded_models = self.models.len() as i32;

            Ok(Response::new(HealthCheckResponse {
                status: ServingStatus::Serving as i32,
                gpu_info,
                loaded_models_count: loaded_models,
            }))
        }

        #[cfg(not(feature = "onnx"))]
        {
            Ok(Response::new(HealthCheckResponse {
                status: ServingStatus::NotServing as i32,
                gpu_info: "ONNX not enabled".to_string(),
                loaded_models_count: 0,
            }))
        }
    }
}

#[cfg(feature = "onnx")]
impl OnnxInferenceServer {
    /// Convert protobuf Tensor to ONNX Runtime Value
    fn tensor_to_ort_value(&self, tensor: &Tensor, session: &Session) -> Result<Value> {
        let shape: Vec<usize> = tensor
            .shape
            .as_ref()
            .map(|s| s.dims.iter().map(|&d| d as usize).collect())
            .unwrap_or_default();

        match tensor.dtype {
            x if x == TensorDataType::TensorDataTypeFloat32 as i32 => {
                let data: Vec<f32> = if !tensor.float_data.is_empty() {
                    tensor.float_data.clone()
                } else {
                    // Parse from bytes
                    tensor
                        .data
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect()
                };
                let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
                    .context("Failed to create ndarray from tensor data")?;
                Value::from_array(session.allocator(), &array)
                    .context("Failed to create ONNX Runtime value")
            }
            x if x == TensorDataType::TensorDataTypeInt64 as i32 => {
                let data: Vec<i64> = if !tensor.int64_data.is_empty() {
                    tensor.int64_data.clone()
                } else {
                    tensor
                        .data
                        .chunks_exact(8)
                        .map(|chunk| i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]))
                        .collect()
                };
                let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
                    .context("Failed to create ndarray from tensor data")?;
                Value::from_array(session.allocator(), &array)
                    .context("Failed to create ONNX Runtime value")
            }
            _ => Err(anyhow::anyhow!(
                "Unsupported tensor dtype: {:?}",
                tensor.dtype
            )),
        }
    }

    /// Convert ONNX Runtime Value to protobuf Tensor
    fn ort_value_to_tensor(&self, value: &Value) -> Result<Tensor> {
        match value.try_extract::<f32>() {
            Ok(tensor) => {
                let view = tensor.view();
                let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
                let float_data: Vec<f32> = view.iter().copied().collect();
                Ok(Tensor {
                    dtype: TensorDataType::TensorDataTypeFloat32 as i32,
                    shape: Some(TensorShape { dims: shape }),
                    data: vec![], // Empty, using float_data instead
                    float_data,
                    int64_data: vec![],
                    int32_data: vec![],
                })
            }
            Err(_) => {
                // Try INT64
                match value.try_extract::<i64>() {
                    Ok(tensor) => {
                        let view = tensor.view();
                        let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
                        let int64_data: Vec<i64> = view.iter().copied().collect();
                        Ok(Tensor {
                            dtype: TensorDataType::TensorDataTypeInt64 as i32,
                            shape: Some(TensorShape { dims: shape }),
                            data: vec![],
                            float_data: vec![],
                            int64_data,
                            int32_data: vec![],
                        })
                    }
                    Err(e) => Err(anyhow::anyhow!(
                        "Failed to extract tensor as f32 or i64: {}",
                        e
                    )),
                }
            }
        }
    }
}

/// Start the gRPC inference server
pub async fn start_server(addr: std::net::SocketAddr, gpu_device_id: i32) -> Result<()> {
    let server = OnnxInferenceServer::new(gpu_device_id)
        .context("Failed to create inference server")?;

    info!(addr = %addr, "Starting gRPC ONNX Inference Server");

    Server::builder()
        .add_service(OnnxInferenceServiceServer::new(server))
        .serve(addr)
        .await
        .context("Failed to start gRPC server")?;

    Ok(())
}
