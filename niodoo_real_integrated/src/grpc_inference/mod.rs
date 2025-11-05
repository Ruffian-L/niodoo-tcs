//! gRPC Inference Module
//!
//! Provides gRPC-based ONNX inference server and client for distributed inference

#[cfg(feature = "svc")]
pub mod server;
#[cfg(feature = "svc")]
pub mod client;

#[cfg(feature = "svc")]
pub use server::{OnnxInferenceServer, start_server};
#[cfg(feature = "svc")]
pub use client::OnnxInferenceClient;

