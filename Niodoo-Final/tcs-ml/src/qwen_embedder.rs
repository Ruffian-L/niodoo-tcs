//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use ndarray::{Array2, Array4};
use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};
use ort::memory::Allocator;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{builder::SessionBuilder, Session};
use ort::value::{Tensor, Value};
use std::collections::HashMap;
use std::convert::TryFrom;

#[cfg(feature = "tokenizers")]
use tokenizers::Tokenizer;

use tracing::{debug, error, info, warn};

use crate::qwen_config::QwenConfig;
use crate::qwen_error::{QwenError, QwenResult};
use half::f16;

/// Stateful Qwen embedder with KV cache management
#[derive(Debug)]
pub struct QwenEmbedder {
    session: Session,
    config: QwenConfig,
    #[cfg(feature = "tokenizers")]
    tokenizer: Option<Tokenizer>,
    kv_cache: HashMap<String, Array4<f32>>, // [batch, heads, seq, head_dim]
    current_seq_len: usize,
    attention_cache: Vec<i64>,
    chunk_size: usize,
}

impl QwenEmbedder {
    /// Create embedder with default config (Qwen2.5-Coder 0.5B)
    pub fn new(model_path: &str) -> QwenResult<Self> {
        Self::with_config(model_path, QwenConfig::default())
    }

    /// Create embedder with custom configuration
    pub fn with_config(model_path: &str, config: QwenConfig) -> QwenResult<Self> {
        config.validate()?;

        let mut config = config;

        // ort 2.0: SessionBuilder manages environment internally, no need for explicit Environment

        // Check if CPU mode is forced via environment variable
        let force_cpu = std::env::var("QWEN_FORCE_CPU")
            .ok()
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(false)
            || std::env::var("ONNX_FORCE_CPU")
                .ok()
                .and_then(|v| v.parse::<bool>().ok())
                .unwrap_or(false);

        // Build session with CUDA execution provider explicitly enabled (unless forced to CPU)
        let session_builder = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_intra_threads(4)?;

        let session_builder = if force_cpu {
            info!(
                target: "tcs-ml::qwen_embedder",
                "CPU mode forced via environment variable"
            );
            session_builder
        } else {
            // Try CUDA execution provider - if it fails or hangs, fallback to CPU
            // The timeout wrapper in QwenStatefulEmbedder will catch hangs
            // RTX 5090 optimization: Use much higher GPU memory limit (4GB+)
            // Check hardware profile from environment
            let default_mem_limit_mb = if std::env::var("HARDWARE")
                .ok()
                .map(|v| v.to_lowercase())
                .map(|v| v.contains("5090") || v.contains("rtx5090"))
                .unwrap_or(false)
            {
                4096 // RTX 5090: 4GB for embeddings
            } else if std::env::var("HARDWARE")
                .ok()
                .map(|v| v.to_lowercase())
                .map(|v| v.contains("h200"))
                .unwrap_or(false)
            {
                2048 // H200: 2GB
            } else {
                512 // Default conservative
            };

            let gpu_mem_limit_mb = std::env::var("GPU_MEM_LIMIT_MB")
                .ok()
                .or_else(|| std::env::var("QWEN_CUDA_MEM_LIMIT_MB").ok())
                .and_then(|raw| raw.parse::<usize>().ok())
                .filter(|&mb| mb > 0)
                .unwrap_or(default_mem_limit_mb);

            let gpu_mem_limit_bytes = gpu_mem_limit_mb.saturating_mul(1024 * 1024);

            // ort 2.0: Try CUDA execution provider - build() returns ExecutionProviderDispatch directly
            let cuda_provider = CUDAExecutionProvider::default()
                .with_memory_limit(gpu_mem_limit_bytes)
                .build();
            match session_builder.with_execution_providers([cuda_provider]) {
                Ok(builder) => {
                    info!(
                        target: "tcs-ml::qwen_embedder",
                        gpu_mem_limit_mb,
                        "CUDA execution provider enabled successfully"
                    );
                    builder
                }
                Err(e) => {
                    warn!(
                        target: "tcs-ml::qwen_embedder",
                        error = %e,
                        "Failed to enable CUDA execution provider, falling back to CPU"
                    );
                    // Retry without providers (will use CPU by default)
                    SessionBuilder::new()?
                        .with_optimization_level(GraphOptimizationLevel::Level1)?
                        .with_intra_threads(4)?
                }
            }
        };

        let session = match session_builder.commit_from_file(model_path) {
            Ok(session) => session,
            Err(error) => {
                warn!(
                    target: "tcs-ml::qwen_embedder",
                    error = %error,
                    "Failed to initialise CUDA session, falling back to CPU"
                );
                // Fallback to CPU-only execution
                let cpu_provider = CPUExecutionProvider::default().build();
                let fallback_builder = SessionBuilder::new()?
                    .with_optimization_level(GraphOptimizationLevel::Level1)?
                    .with_intra_threads(4)?
                    .with_execution_providers([cpu_provider])?;
                fallback_builder.commit_from_file(model_path)?
            }
        };

        info!(
            target: "tcs-ml::qwen_embedder",
            "ONNX Runtime session created (will use CUDA if available)"
        );

        if let Some(detected_dim) = Self::detect_model_embed_dim(&session) {
            if detected_dim != config.embed_dim {
                warn!(
                    target: "tcs-ml::qwen_embedder",
                    configured = config.embed_dim,
                    detected = detected_dim,
                    "Model hidden dimension differs from config; updating embed_dim"
                );
                config.embed_dim = detected_dim;
            } else {
                debug!(
                    target: "tcs-ml::qwen_embedder",
                    detected = detected_dim,
                    "Confirmed embed_dim from ONNX metadata"
                );
            }
        } else {
            warn!(
                target: "tcs-ml::qwen_embedder",
                fallback = config.embed_dim,
                "Unable to detect model embed_dim from ONNX metadata; using configured value"
            );
        }

        let chunk_size = Self::resolve_chunk_size(&config);
        info!(
            target: "tcs-ml::qwen_embedder",
            chunk_size,
            max_seq = config.max_seq_len,
            "Configured chunked inference window"
        );

        // Try to load tokenizer
        #[cfg(feature = "tokenizers")]
        let tokenizer = {
            let mut tokenizer_path = std::path::PathBuf::from(model_path);
            tokenizer_path.pop(); // Remove model file (e.g., model.onnx)
                                  // For Qwen3-Embedding-4B-ONNX, tokenizer.json is in the same dir as model.onnx
                                  // For older models, it might be one level up, so try both
            let tokenizer_in_same_dir = tokenizer_path.join("tokenizer.json");
            let tokenizer_path = if tokenizer_in_same_dir.exists() {
                tokenizer_in_same_dir
            } else {
                // Fallback: try one level up (for older model structures)
                let mut fallback = tokenizer_path.clone();
                fallback.pop();
                fallback.push("tokenizer.json");
                fallback
            };

            if tokenizer_path.exists() {
                match Tokenizer::from_file(&tokenizer_path) {
                    Ok(t) => {
                        info!(
                            target: "tcs-ml::qwen_embedder",
                            path = ?tokenizer_path,
                            "Loaded tokenizer"
                        );
                        Some(t)
                    }
                    Err(e) => {
                        warn!(
                            target: "tcs-ml::qwen_embedder",
                            error = %e,
                            path = ?tokenizer_path,
                            "Failed to load tokenizer; using fallback"
                        );
                        None
                    }
                }
            } else {
                warn!(
                    target: "tcs-ml::qwen_embedder",
                    path = ?tokenizer_path,
                    "Tokenizer not found; using fallback"
                );
                None
            }
        };

        Ok(Self {
            session,
            config,
            #[cfg(feature = "tokenizers")]
            tokenizer,
            kv_cache: HashMap::new(),
            current_seq_len: 0,
            attention_cache: Vec::new(),
            chunk_size,
        })
    }

    /// Tokenize input with fallback to character encoding
    fn tokenize(&self, prompt: &str) -> QwenResult<(Vec<i64>, Vec<i64>)> {
        #[cfg(feature = "tokenizers")]
        {
            if let Some(ref tokenizer) = self.tokenizer {
                let encoding = tokenizer.encode(prompt, true)?;
                let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
                let attention_mask: Vec<i64> = encoding
                    .get_attention_mask()
                    .iter()
                    .map(|&x| x as i64)
                    .collect();
                return Ok((input_ids, attention_mask));
            }
        }
        // Feature tokenizers is disabled - use fallback

        // Fallback: character encoding
        let chars: Vec<i64> = prompt.chars().map(|c| (c as u32) as i64).collect();
        let attention_mask = vec![1i64; chars.len()];
        Ok((chars, attention_mask))
    }

    /// Initialize KV cache for first inference
    fn init_kv_cache(&mut self) {
        self.kv_cache.clear();
        // Use num_kv_heads if specified (for GQA), otherwise use num_heads (for MHA)
        let kv_heads = self.config.num_kv_heads.unwrap_or(self.config.num_heads);
        for layer in 0..self.config.num_layers {
            let key_name = format!("past_key_values.{}.key", layer);
            let value_name = format!("past_key_values.{}.value", layer);

            // Embedding graph expects zero-length past KV tensors. Keep
            // them empty so total_sequence_length == sequence_length.
            let empty_cache = Array4::<f32>::zeros((1, kv_heads, 0, self.config.head_dim));
            self.kv_cache.insert(key_name, empty_cache.clone());
            self.kv_cache.insert(value_name, empty_cache);
        }
        self.current_seq_len = 0;
        self.attention_cache.clear();
    }

    fn ensure_capacity_for(&mut self, tokens_to_add: usize) {
        if tokens_to_add == 0 {
            return;
        }

        if tokens_to_add > self.config.max_seq_len {
            warn!(
                target: "tcs-ml::qwen_embedder",
                tokens_to_add,
                max_seq = self.config.max_seq_len,
                "Chunk length exceeds model maximum; truncating to max_seq_len"
            );
        }

        if self.current_seq_len + tokens_to_add > self.config.max_seq_len {
            info!(
                target: "tcs-ml::qwen_embedder",
                current = self.current_seq_len,
                incoming = tokens_to_add,
                max_seq = self.config.max_seq_len,
                "Resetting KV cache to respect max_seq_len before processing chunk"
            );
            self.init_kv_cache();
        }
    }

    /// Stateful embedding: takes prompt, updates KV cache, returns configured embedding vector
    pub fn embed(&mut self, prompt: &str) -> QwenResult<Vec<f32>> {
        let (tokens, raw_attention_mask) = self.tokenize(prompt)?;
        if tokens.is_empty() {
            return Err(QwenError::EmptyPrompt);
        }

        let attention_mask = if raw_attention_mask.len() == tokens.len() {
            raw_attention_mask
        } else {
            if !raw_attention_mask.is_empty() {
                warn!(
                    target: "tcs-ml::qwen_embedder",
                    mask_len = raw_attention_mask.len(),
                    token_len = tokens.len(),
                    "Tokenizer attention mask length mismatch; falling back to ones"
                );
            }
            vec![1i64; tokens.len()]
        };

        // Always reset cache for each embedding call to avoid incompatible incremental states
        self.init_kv_cache();

        let mut aggregated: Option<Vec<f32>> = None;
        let mut chunk_count = 0usize;
        let mut weight_sum = 0usize;
        let mut offset = 0usize;

        while offset < tokens.len() {
            let remaining = tokens.len() - offset;
            let chunk_len = remaining.min(self.chunk_size).min(self.config.max_seq_len);

            self.ensure_capacity_for(chunk_len);

            let chunk_tokens = &tokens[offset..offset + chunk_len];
            let chunk_mask = &attention_mask[offset..offset + chunk_len];

            let chunk_embedding = self.run_inference_step(chunk_tokens, chunk_mask)?;
            let weight = chunk_len.max(1);

            if let Some(ref mut acc) = aggregated {
                for (dst, src) in acc.iter_mut().zip(chunk_embedding.iter()) {
                    *dst += *src * weight as f32;
                }
            } else {
                let mut weighted = chunk_embedding;
                for value in weighted.iter_mut() {
                    *value *= weight as f32;
                }
                aggregated = Some(weighted);
            }

            weight_sum += weight;
            chunk_count += 1;
            offset += chunk_len;
        }

        let mut embeddings = aggregated.unwrap_or_else(|| vec![0.0; self.config.embed_dim]);
        if weight_sum > 0 {
            let scale = 1.0 / weight_sum as f32;
            for value in embeddings.iter_mut() {
                *value *= scale;
            }
        }

        info!(
            target: "tcs-ml::qwen_embedder",
            chunks = chunk_count,
            total_tokens = tokens.len(),
            weights = weight_sum,
            "Generated embeddings with chunked inference"
        );

        Ok(embeddings)
    }

    fn run_inference_step(
        &mut self,
        step_tokens: &[i64],
        step_mask: &[i64],
    ) -> QwenResult<Vec<f32>> {
        let seq_len = step_tokens.len();
        if seq_len == 0 {
            return Err(QwenError::EmptyInferenceStep);
        }

        if step_mask.len() != seq_len {
            return Err(QwenError::AttentionMaskMismatch {
                mask_len: step_mask.len(),
                token_len: seq_len,
            });
        }

        let total_seq_len = self.current_seq_len + seq_len;
        if total_seq_len > self.config.max_seq_len {
            return Err(QwenError::SequenceTooLong {
                total_seq_len,
                max_seq_len: self.config.max_seq_len,
            });
        }

        // Check if model exposes KV cache inputs.
        let session_inputs = &self.session.inputs;
        let has_kv_cache = session_inputs
            .iter()
            .any(|inp| inp.name.contains("past_key_values"));

        let batch_size = 1;

        // Create all tensors and keep them alive for the entire inference call
        let input_ids_array = Array2::from_shape_vec((batch_size, seq_len), step_tokens.to_vec())
            .map_err(|e| QwenError::TensorBuild {
            name: "input_ids",
            source: e.into(),
        })?;

        let mut attention_total = Vec::with_capacity(self.attention_cache.len() + seq_len);
        attention_total.extend_from_slice(&self.attention_cache);
        attention_total.extend_from_slice(step_mask);
        debug_assert_eq!(attention_total.len(), total_seq_len);

        // Qwen3-Embedding-4B ONNX expects attention_mask with shape
        // [batch_size, total_sequence_length] where total_sequence_length is
        // past_sequence_length + sequence_length. On the first step
        // past_sequence_length is 0, so this becomes [1, seq_len]. For
        // subsequent steps it grows as we append to attention_total.
        let attention_mask_array =
            Array2::from_shape_vec((batch_size, total_seq_len), attention_total.clone()).map_err(
                |e| QwenError::TensorBuild {
                    name: "attention_mask",
                    source: e.into(),
                },
            )?;

        let position_ids: Vec<i64> =
            (self.current_seq_len as i64..(self.current_seq_len + seq_len) as i64).collect();
        let position_ids_array = Array2::from_shape_vec((batch_size, seq_len), position_ids)
            .map_err(|e| QwenError::TensorBuild {
                name: "position_ids",
                source: e.into(),
            })?;

        let mut prepared_inputs: HashMap<String, Value> = HashMap::new();

        // Core inputs: keep using raw-data helper since shapes are always > 0
        let input_ids_shape: Vec<i64> = input_ids_array.shape().iter().map(|&d| d as i64).collect();
        let input_ids_data: Vec<i64> = input_ids_array.iter().copied().collect();
        let input_ids_value = Value::from_array((input_ids_shape.as_slice(), input_ids_data))
            .map(|v| v.into_dyn())
            .map_err(|e| QwenError::OnnxInference { source: e })?;
        prepared_inputs.insert("input_ids".to_string(), input_ids_value);

        let attention_mask_shape: Vec<i64> = attention_mask_array
            .shape()
            .iter()
            .map(|&d| d as i64)
            .collect();
        let attention_mask_data: Vec<i64> = attention_mask_array.iter().copied().collect();
        let attention_mask_value =
            Value::from_array((attention_mask_shape.as_slice(), attention_mask_data))
                .map(|v| v.into_dyn())
                .map_err(|e| QwenError::OnnxInference { source: e })?;
        prepared_inputs.insert("attention_mask".to_string(), attention_mask_value);

        let expects_position_ids = session_inputs.iter().any(|inp| inp.name == "position_ids");

        if expects_position_ids {
            let position_ids_shape: Vec<i64> = position_ids_array
                .shape()
                .iter()
                .map(|&d| d as i64)
                .collect();
            let position_ids_data: Vec<i64> = position_ids_array.iter().copied().collect();
            let position_ids_value =
                Value::from_array((position_ids_shape.as_slice(), position_ids_data))
                    .map(|v| v.into_dyn())
                    .map_err(|e| QwenError::OnnxInference { source: e })?;
            prepared_inputs.insert("position_ids".to_string(), position_ids_value);
        }

        if has_kv_cache {
            let cpu_allocator = Allocator::default();

            for layer in 0..self.config.num_layers {
                let key_name = format!("past_key_values.{}.key", layer);
                let value_name = format!("past_key_values.{}.value", layer);

                let key_cache = self.kv_cache.get(&key_name).unwrap();
                let value_cache = self.kv_cache.get(&value_name).unwrap();

                // Convert f32 arrays to f16 for FP16 model
                let key_f16: Array4<f16> = key_cache.mapv(|x| f16::from_f32(x));
                let value_f16: Array4<f16> = value_cache.mapv(|x| f16::from_f32(x));

                let key_value = Self::build_f16_value(&cpu_allocator, key_f16)
                    .map_err(|e| QwenError::OnnxInference { source: e })?;
                let value_value = Self::build_f16_value(&cpu_allocator, value_f16)
                    .map_err(|e| QwenError::OnnxInference { source: e })?;

                prepared_inputs.insert(key_name, key_value);
                prepared_inputs.insert(value_name, value_value);
            }
        }

        let context_before = self.current_seq_len;
        let prepared_count = prepared_inputs.len();

        if prepared_count > 3 {
            warn!(
                target: "tcs-ml::qwen_embedder",
                input_count = prepared_count,
                "Embed loop guard triggered; limiting to single-pass embedding"
            );
            // Return a zero-vector placeholder rather than risk infinite loop
            return Ok(vec![0.0; self.config.embed_dim]);
        }
        if seq_len > 1 || context_before == 0 {
            debug!(
                target: "tcs-ml::qwen_embedder",
                input_count = prepared_count,
                seq_len,
                context_before,
                "Running ONNX inference"
            );
        }

        // Run inference with the prepared tensors
        let inference_start = std::time::Instant::now();
        info!(
            target: "tcs-ml::qwen_embedder",
            input_count = prepared_count,
            "About to run ONNX inference"
        );
        // ort 2.0: run() takes named inputs; align with declared order
        let mut ordered_names: Vec<String> = Vec::new();
        let mut ordered_values: Vec<Value> = Vec::new();
        for input in &self.session.inputs {
            if let Some(value) = prepared_inputs.remove(input.name.as_str()) {
                ordered_names.push(input.name.clone());
                ordered_values.push(value);
            }
        }

        if !prepared_inputs.is_empty() {
            warn!(
                target: "tcs-ml::qwen_embedder",
                missing = prepared_inputs.len(),
                "Prepared inputs remained unused; ONNX graph may have unexpected inputs"
            );
        }

        let mut named_inputs: Vec<(&str, Value)> = Vec::with_capacity(ordered_values.len());
        for (idx, value) in ordered_values.into_iter().enumerate() {
            let name_ref = ordered_names[idx].as_str();
            named_inputs.push((name_ref, value));
        }

        let outputs = match self.session.run(named_inputs) {
            Ok(o) => o,
            Err(e) => {
                error!(
                    target: "tcs-ml::qwen_embedder",
                    error = %e,
                    "ONNX inference failed"
                );
                return Err(QwenError::OnnxInference { source: e });
            }
        };
        let inference_duration = inference_start.elapsed();

        if seq_len > 1 || context_before == 0 {
            info!(
                target: "tcs-ml::qwen_embedder",
                duration_ms = inference_duration.as_secs_f64() * 1000.0,
                seq_len,
                "ONNX inference completed"
            );
        }

        let mut outputs_storage: Vec<(String, Value)> = outputs
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect();

        if outputs_storage.is_empty() {
            return Err(QwenError::NoOutputs);
        }

        let mut embedding_index: Option<usize> = outputs_storage
            .iter()
            .position(|(name, _)| name == "last_hidden_state");

        if embedding_index.is_none() {
            for (idx, (name, value)) in outputs_storage.iter().enumerate() {
                if let Ok(view) = value.try_extract_array::<f32>() {
                    let shape = view.shape();
                    if shape.len() >= 2 {
                        if let Some(&hidden) = shape.last() {
                            if hidden == self.config.embed_dim {
                                embedding_index = Some(idx);
                                debug!(
                                    target: "tcs-ml::qwen_embedder",
                                    selected_output = %name,
                                    shape = ?shape,
                                    "Found embedding-compatible tensor via shape match"
                                );
                                break;
                            }
                        }
                    }
                }
            }
        }

        let (selected_name, selected_value) = if let Some(idx) = embedding_index {
            outputs_storage.swap_remove(idx)
        } else {
            let fallback = outputs_storage.swap_remove(0);
            warn!(
                target: "tcs-ml::qwen_embedder",
                fallback_output = %fallback.0,
                expected_dim = self.config.embed_dim,
                "No explicit hidden-state output; falling back to first tensor"
            );
            fallback
        };

        if selected_name != "last_hidden_state" {
            warn!(
                target: "tcs-ml::qwen_embedder",
                selected_output = %selected_name,
                "Embedding extracted from fallback output"
            );
        } else {
            debug!(
                target: "tcs-ml::qwen_embedder",
                "Using last_hidden_state output for embeddings"
            );
        }

        // Extract embeddings from the selected tensor
        let embeddings = self.extract_embeddings_v2(&selected_value)?;

        // Embedding graph does not reuse KV cache across steps; always reset.
        self.init_kv_cache();
        self.current_seq_len = 0;
        self.attention_cache.clear();

        if seq_len > 1 || context_before == 0 {
            info!(
                target: "tcs-ml::qwen_embedder",
                dims = embeddings.len(),
                context_len = self.current_seq_len,
                "Extracted embeddings"
            );
        }

        Ok(embeddings)
    }

    /// Extract embedding vector from the model logits using ort 2.0 safe API
    fn extract_embeddings_v2(&self, logits: &Value) -> QwenResult<Vec<f32>> {
        if let Ok(view_f32) = logits.try_extract_array::<f32>() {
            let shape_vec = view_f32.shape().to_vec();
            let flat: Vec<f32> = view_f32.iter().copied().collect();
            return self.extract_from_flat(shape_vec, flat);
        }

        // Fallback: some ONNX exports (including official Qwen3) emit f16 tensors
        match logits.try_extract_array::<f16>() {
            Ok(view_f16) => {
                debug!(
                    target: "tcs-ml::qwen_embedder",
                    "Converting f16 ONNX output to f32 for embedding extraction"
                );
                let shape_vec = view_f16.shape().to_vec();
                let flat: Vec<f32> = view_f16.iter().map(|val| val.to_f32()).collect();
                self.extract_from_flat(shape_vec, flat)
            }
            Err(e) => {
                error!(
                    target: "tcs-ml::qwen_embedder",
                    error = %e,
                    "Failed to extract ONNX output tensor as f32 or f16"
                );
                Err(QwenError::OnnxInference { source: e })
            }
        }
    }

    fn extract_from_flat(&self, shape_vec: Vec<usize>, flat: Vec<f32>) -> QwenResult<Vec<f32>> {
        if shape_vec.len() != 3 {
            return Err(QwenError::UnexpectedTensorShape {
                shape: shape_vec,
                expected_hidden_dim: self.config.embed_dim,
            });
        }

        let batch_size = shape_vec[0];
        let seq_len = shape_vec[1];
        let hidden_size = shape_vec[2];

        if batch_size != 1 {
            return Err(QwenError::UnexpectedTensorShape {
                shape: shape_vec,
                expected_hidden_dim: self.config.embed_dim,
            });
        }

        if hidden_size != self.config.embed_dim {
            warn!(
                target: "tcs-ml::qwen_embedder",
                actual = hidden_size,
                configured = self.config.embed_dim,
                "Model hidden size differs from configured embed_dim"
            );
        }

        debug!(
            target: "tcs-ml::qwen_embedder",
            shape = ?shape_vec,
            batch_size,
            seq_len,
            hidden_size,
            expected_hidden_dim = self.config.embed_dim,
            "Validated embedding tensor shape"
        );

        if seq_len == 0 {
            warn!(
                target: "tcs-ml::qwen_embedder",
                "Received empty sequence (seq_len=0), returning zero vector"
            );
            return Ok(vec![0.0; self.config.embed_dim]);
        }

        let total_elements = flat.len();
        let last_idx = seq_len.saturating_sub(1);
        let last_token_start = last_idx * hidden_size;
        let last_token_end = last_token_start + hidden_size;

        if last_token_end > total_elements {
            warn!(
                target: "tcs-ml::qwen_embedder",
                seq_len,
                last_idx,
                total_elements,
                "Unexpected index calculation, using first token as fallback"
            );
            let pooled: Vec<f32> = flat.iter().take(hidden_size).copied().collect();
            return Ok(self.normalize_embedding(pooled));
        }

        let pooled = flat[last_token_start..last_token_end].to_vec();
        Ok(self.normalize_embedding(pooled))
    }

    fn normalize_embedding(&self, mut pooled: Vec<f32>) -> Vec<f32> {
        let norm_sq: f32 = pooled.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt().max(1e-8);
        for value in pooled.iter_mut() {
            *value /= norm;
        }
        pooled.resize(self.config.embed_dim, 0.0);
        pooled
    }

    /// Reset KV cache for fresh context (new conversation/state thread)
    pub fn reset_cache(&mut self) {
        info!(
            target: "tcs-ml::qwen_embedder",
            "Resetting KV cache for fresh context"
        );
        self.init_kv_cache();
    }

    /// Get current context length
    pub fn context_length(&self) -> usize {
        self.current_seq_len
    }

    /// Access the cached attention mask for diagnostics/metrics
    pub fn attention_mask(&self) -> &[i64] {
        &self.attention_cache
    }
}

impl QwenEmbedder {
    pub fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    fn build_f16_value(allocator: &Allocator, tensor: Array4<f16>) -> ort::Result<Value> {
        if tensor.is_empty() {
            let shape: Vec<usize> = tensor.shape().iter().copied().collect();
            let ort_tensor = Tensor::<f16>::new(allocator, shape)?;
            return Ok(ort_tensor.into_dyn());
        }

        let shape: Vec<i64> = tensor.shape().iter().map(|&d| d as i64).collect();
        let data = tensor.into_raw_vec();
        Value::from_array((shape.as_slice(), data)).map(|v| v.into_dyn())
    }

    fn detect_model_embed_dim(session: &Session) -> Option<usize> {
        let target_output = session
            .outputs
            .iter()
            .find(|output| output.name == "last_hidden_state")
            .or_else(|| session.outputs.get(0));

        target_output
            .and_then(|output| output.output_type.tensor_shape())
            .and_then(|shape| shape.last())
            .and_then(|dim| usize::try_from(*dim).ok())
            .filter(|value| *value > 0)
    }

    fn resolve_chunk_size(config: &QwenConfig) -> usize {
        let env_chunk = std::env::var("QWEN_CHUNK_TOKENS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|&value| value > 0);

        let resolved = match env_chunk {
            Some(value) => value.min(config.max_seq_len),
            None => config.max_seq_len.min(16_384),
        };

        if let Some(env_value) = env_chunk {
            info!(
                target: "tcs-ml::qwen_embedder",
                env_value,
                resolved,
                "Using chunk size from QWEN_CHUNK_TOKENS"
            );
        } else {
            info!(
                target: "tcs-ml::qwen_embedder",
                default = resolved,
                "Using default chunk size (min(max_seq_len, 16k))"
            );
        }

        resolved
    }
}
