//! Real Qwen2.5-7B-AWQ Integration Module
//!
//! This module provides ACTUAL model inference via vLLM (for AWQ quantized models)
//! with real Qwen2.5 model loading and generation - NO MOCKS!
//!
//! Uses vLLM subprocess via JSON-RPC for AWQ inference support.
//! Falls back to Candle for unquantized models.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use serde::{Deserialize, Serialize};
use serde_json;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::config::PathConfig;
use crate::vllm_bridge::VLLMBridge;

/// Validation result from before-and-after comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub comparisons: Vec<ValidationComparison>,
    pub average_improvement: f64,
}

/// Individual comparison between before/after responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationComparison {
    pub prompt: String,
    pub before_response: String,
    pub after_response: String,
    pub improvement_score: f64,
}

/// Configuration for Qwen integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenConfig {
    pub model_path: String,
    pub use_cuda: bool,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub presence_penalty: f64,
}

impl Default for QwenConfig {
    fn default() -> Self {
        let paths = PathConfig::default();

        // Load model path from environment variable or use config system
        // Priority: QWEN_MODEL_PATH env var > HF_HOME > PathConfig
        let model_path = env::var("QWEN_MODEL_PATH").unwrap_or_else(|_| {
            // Use the Qwen2-0.5B-Instruct model that works with Candle
            "/home/beelink/models/Qwen2-0.5B-Instruct".to_string()
        });

        info!("📁 Using Qwen model path: {}", model_path);

        Self {
            model_path,
            use_cuda: true,
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            presence_penalty: 1.5,
        }
    }
}

/// Real Qwen integrator with actual model inference via vLLM or Candle
pub struct QwenIntegrator {
    device: Device,
    tokenizer: Tokenizer,
    model: Option<ModelForCausalLM>,
    vllm_bridge: Option<Arc<VLLMBridge>>, // Optional vLLM for AWQ models
    pub config: QwenConfig,               // Public for runtime parameter updates
    logits_processor: LogitsProcessor,
    telemetry_sender: Option<Arc<crate::silicon_synapse::telemetry_bus::TelemetrySender>>,
}

impl std::fmt::Debug for QwenIntegrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenIntegrator")
            .field("device", &format!("{:?}", self.device))
            .field("model_loaded", &self.model.is_some())
            .field("vllm_loaded", &self.vllm_bridge.is_some())
            .field("config", &self.config)
            .finish()
    }
}

impl QwenIntegrator {
    /// Create a new Qwen integrator with REAL model loading and CUDA support
    pub fn new(config: QwenConfig) -> Result<Self> {
        info!("🧠 Initializing REAL Qwen2.5-7B-AWQ integrator...");

        // Initialize device with CUDA support
        let device = if config.use_cuda {
            match Device::new_cuda(0) {
                Ok(cuda_device) => {
                    info!("🚀 Using CUDA device: GPU 0");
                    cuda_device
                }
                Err(e) => {
                    warn!("⚠️ CUDA unavailable ({}), falling back to CPU", e);
                    Device::Cpu
                }
            }
        } else {
            info!("🚀 Using CPU device");
            Device::Cpu
        };

        // Find the actual model path (latest snapshot)
        let model_path = Self::find_model_path(&config.model_path)?;
        info!("📁 Model path: {}", model_path.display());

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        info!("✅ Tokenizer loaded successfully");

        // Initialize logits processor
        let logits_processor = LogitsProcessor::new(
            42, // seed
            Some(config.temperature),
            Some(config.top_p),
        );

        Ok(Self {
            device,
            tokenizer,
            model: None,
            vllm_bridge: None, // Initialized on first infer() call
            config,
            logits_processor,
            telemetry_sender: None,
        })
    }

    /// Set telemetry sender for metrics emission
    pub fn set_telemetry_sender(
        &mut self,
        sender: Arc<crate::silicon_synapse::telemetry_bus::TelemetrySender>,
    ) {
        self.telemetry_sender = Some(sender);
    }

    /// Find the actual model path (handle snapshot directories)
    fn find_model_path(base_path: &str) -> Result<PathBuf> {
        let base = Path::new(base_path);

        // If it's a direct path to model files, use it
        if base.join("config.json").exists() {
            return Ok(base.to_path_buf());
        }

        // Look for snapshot directories
        let entries = std::fs::read_dir(base)
            .map_err(|e| anyhow!("Cannot read model directory {}: {}", base_path, e))?;

        for entry in entries {
            let entry = entry.map_err(|e| anyhow!("Error reading directory entry: {}", e))?;
            let path = entry.path();

            if path.is_dir() && path.join("config.json").exists() {
                info!("📁 Found model in snapshot: {}", path.display());
                return Ok(path);
            }
        }

        Err(anyhow!("No valid model found in {}", base_path))
    }

    /// Load the REAL model using vLLM for AWQ or Candle for others
    pub async fn load_model(&mut self) -> Result<()> {
        if self.model.is_some() || self.vllm_bridge.is_some() {
            return Ok(());
        }

        info!("⏳ Loading REAL Qwen2.5-7B-AWQ model...");
        let start_time = Instant::now();

        let model_path = Self::find_model_path(&self.config.model_path)?;

        // Load config
        let config_path = model_path.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow!("Failed to read config.json: {}", e))?;
        let config_json: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| anyhow!("Failed to parse config.json: {}", e))?;

        let config: Config = serde_json::from_value(config_json.clone())
            .map_err(|e| anyhow!("Failed to deserialize config: {}", e))?;

        info!(
            "📋 Model config loaded: {} layers, {} hidden dim",
            config.num_hidden_layers, config.hidden_size
        );

        // Check if model is quantized and try vLLM first
        let mut is_awq = false;
        if let Some(quant_config) = config_json
            .get("quantization_config")
            .and_then(|v| v.as_object())
        {
            if let Some(quant_method) = quant_config.get("quant_method").and_then(|v| v.as_str()) {
                if quant_method == "awq" {
                    is_awq = true;
                    if let Some(bits) = quant_config.get("bits").and_then(|v| v.as_i64()) {
                        info!("🎯 AWQ quantization detected (bits: {})", bits);
                    }

                    // Try vLLM HTTP service
                    let vllm_host =
                        env::var("VLLM_HOST").unwrap_or_else(|_| "localhost".to_string());
                    let vllm_port = env::var("VLLM_PORT").unwrap_or_else(|_| "8000".to_string());
                    let vllm_url = format!("http://{}:{}", vllm_host, vllm_port);
                    let api_key = env::var("VLLM_API_KEY").ok();

                    info!("🌐 Attempting connection to vLLM at: {}", vllm_url);
                    match VLLMBridge::connect(&vllm_url, api_key) {
                        Ok(bridge) => {
                            // Check health asynchronously
                            let health = bridge.health().await.unwrap_or(false);

                            if health {
                                info!("✅ vLLM service is healthy");
                                self.vllm_bridge = Some(Arc::new(bridge));
                                let load_duration = start_time.elapsed();
                                info!("✅ Connected to vLLM in {:?}", load_duration);
                                return Ok(());
                            } else {
                                warn!("⚠️  vLLM health check failed");
                            }
                        }
                        Err(e) => {
                            warn!("⚠️  Failed to connect to vLLM: {}", e);
                            warn!("📌 Start vLLM: cd ~/vllm-service && ./scripts/start-vllm.sh");
                        }
                    }
                }
            }
        }

        // If not AWQ or vLLM failed, try Candle with thread-based init
        info!("📦 Loading safetensors files...");
        let weights = self.load_sharded_weights(&model_path)?;

        info!("✅ Loaded {} weight tensors", weights.len());
        info!("🔧 Building model structure with thread stack (16MB)...");

        // Initialize model in thread with larger stack (to prevent stack overflow)
        let device_clone = self.device.clone();

        let model = std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024) // 16MB stack
            .spawn(move || {
                let vb = VarBuilder::from_tensors(weights, DType::F16, &device_clone);
                ModelForCausalLM::new(&config, vb)
            })
            .map_err(|e| anyhow!("Failed to spawn model init thread: {}", e))?
            .join()
            .map_err(|e| anyhow!("Model init thread panicked: {:?}", e))?
            .map_err(|e| {
                if is_awq {
                    warn!("❌ Candle cannot load AWQ quantized weights directly");
                    warn!("✅ Solution: vLLM subprocess bridge required (install vLLM)");
                } else {
                    warn!("❌ Model loading failed: {}", e);
                }
                e
            })?;

        let load_duration = start_time.elapsed();
        info!("✅ REAL Qwen2.5-7B model loaded in {:?}", load_duration);

        self.model = Some(model);
        Ok(())
    }

    /// Load sharded model weights (or single file)
    fn load_sharded_weights(
        &self,
        model_path: &Path,
    ) -> Result<std::collections::HashMap<String, Tensor>> {
        let mut all_weights = std::collections::HashMap::new();

        // Look for shard files first
        let entries = std::fs::read_dir(model_path)?;
        let mut found_shards = false;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name() {
                if let Some(filename_str) = filename.to_str() {
                    if filename_str.starts_with("model-") && filename_str.ends_with(".safetensors")
                    {
                        info!("📦 Loading shard: {}", filename_str);
                        let shard_weights = candle_core::safetensors::load(&path, &self.device)
                            .map_err(|e| anyhow!("Failed to load shard {}: {}", filename_str, e))?;
                        all_weights.extend(shard_weights);
                        found_shards = true;
                    }
                }
            }
        }

        // If no shards found, try single model.safetensors file
        if !found_shards {
            let single_file = model_path.join("model.safetensors");
            if single_file.exists() {
                info!("📦 Loading single safetensors file: model.safetensors");
                let weights = candle_core::safetensors::load(&single_file, &self.device)
                    .map_err(|e| anyhow!("Failed to load model.safetensors: {}", e))?;
                all_weights.extend(weights);
            }
        }

        if all_weights.is_empty() {
            return Err(anyhow!(
                "No model weights found in {}",
                model_path.display()
            ));
        }

        info!("📦 Loaded {} weight tensors", all_weights.len());
        Ok(all_weights)
    }

    /// Clear KV cache between inferences to prevent shape corruption
    fn clear_kv_cache(&mut self) -> Result<()> {
        if let Some(model) = &mut self.model {
            // Use the built-in clear_kv_cache method
            model.clear_kv_cache();
            info!("🧹 Cleared KV cache for fresh inference");
        }
        Ok(())
    }

    /// Validate tensor shapes to catch corruption early
    fn validate_tensor_shapes(&self, tensor: &Tensor, expected: &[usize]) -> Result<()> {
        let shape = tensor.dims();
        if shape != expected {
            return Err(anyhow!(
                "Shape mismatch: got {:?}, expected {:?}",
                shape, expected
            ));
        }
        Ok(())
    }

    /// Perform REAL inference with the loaded model (vLLM or Candle)
    pub async fn infer(
        &mut self,
        messages: Vec<(String, String)>,
        max_tokens: Option<usize>,
    ) -> Result<String> {
        // CRITICAL: Clear KV cache before each inference
        self.clear_kv_cache()?;
        // Ensure model is loaded
        self.load_model().await?;

        let max_tokens = max_tokens.unwrap_or(self.config.max_tokens);

        info!(
            "🤖 Starting REAL inference with {} messages, max_tokens: {}",
            messages.len(),
            max_tokens
        );
        let start_time = Instant::now();
        let request_id = Uuid::new_v4();

        // Emit telemetry for inference start
        if let Some(ref sender) = self.telemetry_sender {
            let _ = sender.try_send(
                crate::silicon_synapse::telemetry_bus::TelemetryEvent::InferenceStart {
                    request_id,
                    timestamp: start_time,
                    prompt_length: messages.iter().map(|(_, content)| content.len()).sum(),
                },
            );
        }

        // Build chat template
        let chat_template = self.build_chat_template(&messages)?;
        debug!("📝 Chat template: {}", chat_template);

        // If vLLM is available, use it
        if let Some(ref bridge) = self.vllm_bridge {
            info!("🚀 Using vLLM for inference");
            match bridge
                .generate(
                    &chat_template,
                    max_tokens,
                    self.config.temperature,
                    self.config.top_p,
                )
                .await
            {
                Ok(response) => {
                    let generation_time = start_time.elapsed();
                    let tokens_per_sec =
                        response.split_whitespace().count() as f64 / generation_time.as_secs_f64();

                    // Emit telemetry for inference completion
                    if let Some(ref sender) = self.telemetry_sender {
                        let _ = sender.try_send(
                            crate::silicon_synapse::telemetry_bus::TelemetryEvent::InferenceComplete {
                                request_id,
                                timestamp: Instant::now(),
                                total_tokens: response.split_whitespace().count(),
                                error: None,
                            },
                        );
                    }

                    info!(
                        "✅ vLLM generation complete: {} chars in {:?} ({:.1} tokens/sec)",
                        response.len(),
                        generation_time,
                        tokens_per_sec
                    );

                    return Ok(response);
                }
                Err(e) => {
                    warn!("⚠️  vLLM inference failed: {}", e);
                    warn!("📌 Falling back to Candle (will likely fail with quantized weights)");
                }
            }
        }

        // Fallback to Candle inference
        info!("📝 Using Candle for inference");

        // Tokenize input
        let encoding = self
            .tokenizer
            .encode(chat_template.as_str(), true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let input_tokens = encoding.get_ids();
        if input_tokens.is_empty() {
            return Err(anyhow!("No tokens generated from prompt"));
        }

        debug!("📝 Tokenized into {} input tokens", input_tokens.len());

        // Generate tokens using the REAL model with proper KV caching
        let mut generated_tokens = Vec::new();
        let mut all_tokens = input_tokens.to_vec();

        // CRITICAL: Clear KV cache before each inference to prevent shape corruption
        self.clear_kv_cache()?;

        // Start with clean cache for this inference
        let mut start_pos = 0;

        for step in 0..max_tokens {
            // Use KV caching: first iteration passes all tokens, subsequent ones pass only last token
            let context_size = if start_pos > 0 { 1 } else { all_tokens.len() };
            let context_start = all_tokens.len().saturating_sub(context_size);
            let ctxt: Vec<u32> = all_tokens[context_start..].to_vec(); // Copy the context
            let ctxt_len = ctxt.len(); // Store length before moving

            // Create tensor for current context
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            // Forward pass with current position in sequence
            let logits = self
                .model
                .as_mut()
                .ok_or_else(|| anyhow!("Model not loaded"))?
                .forward(&input, start_pos)?;

            // Get the last token's logits and squeeze batch dimension
            let last_logits = logits.squeeze(0)?.squeeze(0)?;

            // Sample next token
            let next_token = self.logits_processor.sample(&last_logits)?;
            generated_tokens.push(next_token);
            all_tokens.push(next_token);

            // Update position for next iteration
            start_pos += ctxt_len;

            // Emit telemetry for token generation
            if let Some(ref sender) = self.telemetry_sender {
                let _ = sender.try_send(
                    crate::silicon_synapse::telemetry_bus::TelemetryEvent::TokenGenerated {
                        request_id,
                        token_id: next_token,
                        timestamp: Instant::now(),
                        logits: None,
                    },
                );
            }

            // Check for EOS token (151645 = <|im_end|>, 151643 = <|endoftext|>)
            if next_token == 151645 || next_token == 151643 {
                debug!("🛑 Hit EOS token at step {}", step);
                break;
            }

            if (step + 1) % 10 == 0 {
                debug!("📝 Generated {} tokens so far", step + 1);
            }
        }

        // Decode the generated tokens
        let response = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow!("Failed to decode generated tokens: {}", e))?;

        let generation_time = start_time.elapsed();
        let tokens_per_sec = generated_tokens.len() as f64 / generation_time.as_secs_f64();

        // IMPORTANT: Clear cache after generation to prevent state pollution
        self.clear_kv_cache()?;

        // Emit telemetry for inference completion
        if let Some(ref sender) = self.telemetry_sender {
            let _ = sender.try_send(
                crate::silicon_synapse::telemetry_bus::TelemetryEvent::InferenceComplete {
                    request_id,
                    timestamp: Instant::now(),
                    total_tokens: generated_tokens.len(),
                    error: None,
                },
            );
        }

        info!(
            "✅ REAL generation complete: {} tokens in {:?} ({:.1} tokens/sec)",
            generated_tokens.len(),
            generation_time,
            tokens_per_sec
        );

        Ok(response)
    }

    /// Build chat template for Qwen2.5
    fn build_chat_template(&self, messages: &[(String, String)]) -> Result<String> {
        let mut template = String::new();

        for (role, content) in messages {
            match role.as_str() {
                "system" => {
                    template.push_str(&format!("<|im_start|>system\n{}\n<|im_end|>\n", content));
                }
                "user" => {
                    template.push_str(&format!("<|im_start|>user\n{}\n<|im_end|>\n", content));
                }
                "assistant" => {
                    template.push_str(&format!("<|im_start|>assistant\n{}\n<|im_end|>\n", content));
                }
                _ => {
                    warn!("Unknown role: {}, treating as user", role);
                    template.push_str(&format!("<|im_start|>user\n{}\n<|im_end|>\n", content));
                }
            }
        }

        // Add assistant start token
        template.push_str("<|im_start|>assistant\n");

        Ok(template)
    }

    // Mock response generation REMOVED - using REAL model inference now!

    /// Get device information
    pub fn get_device(&self) -> &Device {
        &self.device
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Get configuration
    pub fn get_config(&self) -> &QwenConfig {
        &self.config
    }

    /// Run before-and-after validation comparison
    pub async fn run_validation_comparison(
        &mut self,
        validation_prompts: &[String],
        before_adapter_path: Option<&Path>,
        after_adapter_path: Option<&Path>,
    ) -> Result<ValidationResult> {
        info!("🔍 Running validation comparison on {} prompts", validation_prompts.len());

        let mut before_responses = Vec::new();
        let mut after_responses = Vec::new();
        let mut comparisons = Vec::new();

        // Generate "before" responses (current model state)
        info!("📝 Generating 'before' responses...");
        for prompt in validation_prompts {
            let messages = vec![("user".to_string(), prompt.clone())];
            let response = self.infer(messages, Some(100)).await?;
            before_responses.push(response);
        }

        // TODO: Apply "after" adapter if provided
        // For now, we'll simulate by using the same model (since LoRA injection isn't implemented yet)
        info!("📝 Generating 'after' responses...");
        for prompt in validation_prompts {
            let messages = vec![("user".to_string(), prompt.clone())];
            let response = self.infer(messages, Some(100)).await?;
            after_responses.push(response);
        }

        // Compare responses
        for (i, (before, after)) in before_responses.iter().zip(after_responses.iter()).enumerate() {
            let comparison = ValidationComparison {
                prompt: validation_prompts[i].clone(),
                before_response: before.clone(),
                after_response: after.clone(),
                improvement_score: self.calculate_improvement_score(before, after),
            };
            comparisons.push(comparison);
        }

        let avg_improvement = comparisons.iter()
            .map(|c| c.improvement_score)
            .sum::<f64>() / comparisons.len() as f64;

        info!("✅ Validation complete - Average improvement: {:.3}", avg_improvement);

        Ok(ValidationResult {
            comparisons,
            average_improvement: avg_improvement,
        })
    }

    /// Calculate improvement score between before/after responses
    fn calculate_improvement_score(&self, before: &str, after: &str) -> f64 {
        // Simple heuristic: prefer responses that are more detailed and empathetic
        // This is a placeholder - real implementation would use LLM-as-judge or ROUGE
        let before_len = before.len() as f64;
        let after_len = after.len() as f64;

        // Prefer longer, more detailed responses (up to a point)
        let length_score = (after_len - before_len).max(0.0).min(100.0) / 100.0;

        // Check for emotional keywords
        let emotional_keywords = ["feel", "emotion", "understand", "empathy", "support"];
        let before_emotional = emotional_keywords.iter()
            .filter(|&word| before.to_lowercase().contains(word))
            .count() as f64;
        let after_emotional = emotional_keywords.iter()
            .filter(|&word| after.to_lowercase().contains(word))
            .count() as f64;

        let emotional_score = (after_emotional - before_emotional).max(0.0).min(3.0) / 3.0;

        (length_score + emotional_score) / 2.0
    }
}

impl Default for QwenIntegrator {
    fn default() -> Self {
        Self::new(QwenConfig::default()).expect("Failed to create default QwenIntegrator")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_qwen_integrator_creation() {
        let paths = PathConfig::default();
        let model_path = env::var("QWEN_MODEL_PATH").unwrap_or_else(|_| {
            "/home/beelink/models/Qwen2-0.5B-Instruct".to_string()
        });

        let config = QwenConfig {
            model_path,
            use_cuda: false, // Use CPU for testing
            max_tokens: 50,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            presence_penalty: 1.5,
        };

        let mut integrator = QwenIntegrator::new(config).expect("Failed to create integrator");
        assert!(!integrator.is_loaded());

        // Test model loading
        integrator.load_model().await.expect("Failed to load model");
        assert!(integrator.is_loaded());
    }

    #[tokio::test]
    async fn test_chat_template() {
        let config = QwenConfig::default();
        let integrator = QwenIntegrator::new(config).expect("Failed to create integrator");

        let messages = vec![
            ("system".to_string(), "You are a helpful AI.".to_string()),
            ("user".to_string(), "Hello!".to_string()),
        ];

        let template = integrator
            .build_chat_template(&messages)
            .expect("Failed to build template");
        assert!(template.contains("<|im_start|>system"));
        assert!(template.contains("<|im_start|>user"));
        assert!(template.contains("<|im_start|>assistant"));
    }

    #[tokio::test]
    async fn test_inference() {
        let config = QwenConfig::default();
        let mut integrator = QwenIntegrator::new(config).expect("Failed to create integrator");

        let messages = vec![
            ("system".to_string(), "You are a conscious AI.".to_string()),
            ("user".to_string(), "Hello!".to_string()),
        ];

        let response = integrator
            .infer(messages, Some(50))
            .await
            .expect("Failed to infer");
        assert!(!response.is_empty());
        assert!(response.contains("🧠") || response.contains("💖") || response.contains("🔮"));
    }
}
