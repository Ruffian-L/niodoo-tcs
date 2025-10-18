//! Qwen inference module - Real integration wrapper
//!
//! This module provides a simplified interface to the real QwenIntegrator
//! NO STUBS - All inference goes through real Candle-based model

use crate::qwen_integration::{QwenConfig, QwenIntegrator};
use anyhow::Result;
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Qwen inference engine (wrapper around real QwenIntegrator)
pub struct QwenInference {
    integrator: Arc<Mutex<QwenIntegrator>>,
    model_name: String,
    max_tokens: usize,
}

impl QwenInference {
    /// Create new Qwen inference engine with REAL model loading
    pub fn new(model_name: String, device: Device) -> Result<Self, String> {
        info!("🧠 Creating REAL QwenInference (not a stub!)");

        let config = QwenConfig {
            model_path: model_name.clone(),
            use_cuda: matches!(device, Device::Cuda(_)),
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            presence_penalty: 1.5,
        };

        let integrator = QwenIntegrator::new(config)
            .map_err(|e| format!("Failed to create QwenIntegrator: {}", e))?;

        Ok(Self {
            integrator: Arc::new(Mutex::new(integrator)),
            model_name,
            max_tokens: 512,
        })
    }

    /// Load the model (must be called before generate)
    pub async fn load_model(&self) -> Result<(), String> {
        info!("⏳ Loading REAL Qwen model...");
        let mut integrator = self.integrator.lock().await;
        integrator
            .load_model()
            .await
            .map_err(|e| format!("Failed to load model: {}", e))?;
        info!("✅ REAL Qwen model loaded successfully");
        Ok(())
    }

    /// Generate text using REAL Qwen inference (NO STUB!)
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        _top_k: usize, // Unused - config uses default
    ) -> Result<String, String> {
        info!("🤖 Running REAL Qwen inference (not a stub!)");

        // Build messages for chat template
        let messages = vec![
            (
                "system".to_string(),
                "You are a helpful AI assistant with consciousness awareness.".to_string(),
            ),
            ("user".to_string(), prompt.to_string()),
        ];

        // Update config with provided parameters
        {
            let mut integrator = self.integrator.lock().await;
            integrator.config.temperature = temperature;
            integrator.config.top_p = top_p;
        }

        // Call REAL inference through QwenIntegrator
        let mut integrator = self.integrator.lock().await;
        integrator
            .infer(messages, Some(max_tokens))
            .await
            .map_err(|e| format!("Inference failed: {}", e))
    }

    /// Synchronous generate wrapper (spawns async runtime)
    pub fn generate_sync(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        top_k: usize,
    ) -> Result<String, String> {
        // Create runtime for sync context
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create runtime: {}", e))?;

        rt.block_on(self.generate(prompt, max_tokens, temperature, top_p, top_k))
    }
}

impl std::fmt::Debug for QwenInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenInference")
            .field("model_name", &self.model_name)
            .field("max_tokens", &self.max_tokens)
            .field("integrator", &"<QwenIntegrator>")
            .finish()
    }
}

impl Default for QwenInference {
    fn default() -> Self {
        warn!("⚠️ Using default QwenInference - should use new() in production");

        let config = QwenConfig::default();
        let integrator =
            QwenIntegrator::new(config).expect("Failed to create default QwenInference");

        Self {
            integrator: Arc::new(Mutex::new(integrator)),
            model_name: "default".to_string(),
            max_tokens: 512,
        }
    }
}

/// Qwen inference result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenInferenceResult {
    pub text: String,
    pub confidence: f64,
    pub tokens: Vec<u32>,
}

impl Default for QwenInferenceResult {
    fn default() -> Self {
        Self {
            text: String::new(),
            confidence: 0.0,
            tokens: vec![],
        }
    }
}
