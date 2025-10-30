// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use ndarray::{Array2, Array4, Axis, CowArray, concatenate, s};
use ort::{Environment, GraphOptimizationLevel, Session, SessionBuilder, Value};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "tokenizers")]
use tokenizers::Tokenizer;

use tracing::{debug, info, warn};

use crate::f16;
use crate::qwen_config::QwenConfig;
use crate::qwen_error::{QwenError, QwenResult};

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
}

impl QwenEmbedder {
    /// Create embedder with default config (Qwen2.5-Coder 0.5B)
    pub fn new(model_path: &str) -> QwenResult<Self> {
        Self::with_config(model_path, QwenConfig::default())
    }

    /// Create embedder with custom configuration
    pub fn with_config(model_path: &str, config: QwenConfig) -> QwenResult<Self> {
        config.validate()?;

        let env = Arc::new(Environment::builder().with_name("qwen_embedder").build()?);

        let session = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_intra_threads(4)?
            .with_model_from_file(model_path)?;

        // Try to load tokenizer
        #[cfg(feature = "tokenizers")]
        let tokenizer = {
            let mut tokenizer_path = std::path::PathBuf::from(model_path);
            tokenizer_path.pop(); // Remove model file
            tokenizer_path.pop(); // Go up from onnx/ to model root
            tokenizer_path.push("tokenizer.json");

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

    /// Initialize empty KV cache for first inference
    fn init_kv_cache(&mut self) {
        self.kv_cache.clear();
        for layer in 0..self.config.num_layers {
            let key_name = format!("past_key_values.{}.key", layer);
            let value_name = format!("past_key_values.{}.value", layer);

            // Empty cache: [batch=1, heads, seq_len=0, head_dim]
            let empty_cache =
                Array4::<f32>::zeros((1, self.config.num_heads, 0, self.config.head_dim));
            self.kv_cache.insert(key_name, empty_cache.clone());
            self.kv_cache.insert(value_name, empty_cache);
        }
        self.current_seq_len = 0;
        self.attention_cache.clear();
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

        if self.kv_cache.is_empty() {
            self.init_kv_cache();
        }

        // If we have no past context, run the whole prompt in a single pass.
        if self.current_seq_len == 0 {
            return self.run_inference_step(&tokens, &attention_mask);
        }

        // Otherwise stream tokens one by one to satisfy the ONNX incremental contract.
        let mut last_embeddings = Vec::new();
        for (token, mask_value) in tokens.iter().zip(attention_mask.iter()) {
            let token_slice = [*token];
            let mask_slice = [*mask_value];
            last_embeddings = self.run_inference_step(&token_slice, &mask_slice)?;
        }

        Ok(last_embeddings)
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

        let batch_size = 1;

        // Create all tensors and keep them alive for the entire inference call
        let input_ids_array = Array2::from_shape_vec((batch_size, seq_len), step_tokens.to_vec())
            .map_err(|e| QwenError::TensorBuild {
            name: "input_ids",
            source: e,
        })?;

        let mut attention_total = Vec::with_capacity(self.attention_cache.len() + seq_len);
        attention_total.extend_from_slice(&self.attention_cache);
        attention_total.extend_from_slice(step_mask);
        debug_assert_eq!(attention_total.len(), total_seq_len);

        let attention_mask_array =
            Array2::from_shape_vec((batch_size, attention_total.len()), attention_total.clone())
                .map_err(|e| QwenError::TensorBuild {
                    name: "attention_mask",
                    source: e,
                })?;

        let position_ids: Vec<i64> =
            (self.current_seq_len as i64..(self.current_seq_len + seq_len) as i64).collect();
        let position_ids_array = Array2::from_shape_vec((batch_size, seq_len), position_ids)
            .map_err(|e| QwenError::TensorBuild {
                name: "position_ids",
                source: e,
            })?;

        // Convert to CowArrays and keep them alive
        let input_ids_cow = CowArray::from(input_ids_array.into_dyn());
        let attention_mask_cow = CowArray::from(attention_mask_array.into_dyn());
        let position_ids_cow = CowArray::from(position_ids_array.into_dyn());

        // Store all KV cache CowArrays first
        let mut kv_cows = Vec::with_capacity(self.config.num_layers * 2);
        for layer in 0..self.config.num_layers {
            let key_name = format!("past_key_values.{}.key", layer);
            let value_name = format!("past_key_values.{}.value", layer);

            let key_cache = self.kv_cache.get(&key_name).unwrap();
            let value_cache = self.kv_cache.get(&value_name).unwrap();

            let key_cow = CowArray::from(key_cache.view().into_dyn());
            let value_cow = CowArray::from(value_cache.view().into_dyn());

            kv_cows.push(key_cow);
            kv_cows.push(value_cow);
        }

        // Now create Value objects from the stored CowArrays
        let input_ids_value = Value::from_array(self.session.allocator(), &input_ids_cow)
            .map_err(|e| QwenError::OnnxInference { source: e })?;
        let attention_mask_value = Value::from_array(self.session.allocator(), &attention_mask_cow)
            .map_err(|e| QwenError::OnnxInference { source: e })?;
        let position_ids_value = Value::from_array(self.session.allocator(), &position_ids_cow)
            .map_err(|e| QwenError::OnnxInference { source: e })?;

        let mut kv_values = Vec::with_capacity(self.config.num_layers * 2);
        for kv_cow in &kv_cows {
            let kv_value = Value::from_array(self.session.allocator(), kv_cow)
                .map_err(|e| QwenError::OnnxInference { source: e })?;
            kv_values.push(kv_value);
        }

        // Combine all input values
        let mut input_values = vec![input_ids_value, attention_mask_value, position_ids_value];
        input_values.extend(kv_values);

        let context_before = self.current_seq_len;
        if seq_len > 1 || context_before == 0 {
            debug!(
                target: "tcs-ml::qwen_embedder",
                input_count = input_values.len(),
                seq_len,
                context_before,
                "Running ONNX inference"
            );
        }

        // Run inference (all CowArrays are still alive here)
        let outputs = self.session.run(input_values)?;

        if outputs.is_empty() {
            return Err(QwenError::NoOutputs);
        }

        // Ensure we received logits + KV cache outputs
        let expected_outputs = 1 + self.config.num_layers * 2;
        if outputs.len() < expected_outputs {
            return Err(QwenError::UnexpectedOutputCount {
                expected: expected_outputs,
                actual: outputs.len(),
            });
        }

        // Extract embeddings from logits (first output)
        let embeddings = self.extract_embeddings(&outputs[0])?;

        // Update KV cache with present_key_values tensors
        let mut new_kv_entries = Vec::with_capacity(self.config.num_layers * 2);
        let mut updated_context_len = self.current_seq_len + seq_len;
        let previous_context_len = self.current_seq_len;

        for layer in 0..self.config.num_layers {
            let key_name = format!("past_key_values.{}.key", layer);
            let value_name = format!("past_key_values.{}.value", layer);

            let key_index = 1 + layer * 2;
            let value_index = key_index + 1;

            let present_key = Self::extract_kv_tensor(&outputs[key_index], &key_name)?;
            let present_value = Self::extract_kv_tensor(&outputs[value_index], &value_name)?;

            let previous_key = self.kv_cache.get(&key_name).unwrap();
            let previous_value = self.kv_cache.get(&value_name).unwrap();

            let merged_key = Self::merge_kv_cache(previous_key, present_key, seq_len, &key_name)?;
            let merged_value =
                Self::merge_kv_cache(previous_value, present_value, seq_len, &value_name)?;

            updated_context_len = merged_key.shape()[2];

            new_kv_entries.push((key_name, merged_key));
            new_kv_entries.push((value_name, merged_value));
        }

        for (name, tensor) in new_kv_entries {
            self.kv_cache.insert(name, tensor);
        }

        // Update sequence length for next inference
        self.current_seq_len = updated_context_len;
        self.attention_cache = attention_total;
        self.truncate_cache_if_needed();
        debug_assert_eq!(
            self.attention_cache.len(),
            self.current_seq_len,
            "attention cache and sequence length diverged"
        );

        if seq_len > 1 || previous_context_len == 0 {
            info!(
                target: "tcs-ml::qwen_embedder",
                dims = embeddings.len(),
                context_len = self.current_seq_len,
                "Extracted embeddings"
            );
        }

        Ok(embeddings)
    }

    /// Extract KV cache tensor as owned Array4<f32>
    fn extract_kv_tensor(value: &Value, name: &str) -> QwenResult<Array4<f32>> {
        if let Ok(tensor) = value.try_extract::<f32>() {
            let view = tensor.view();
            let dims = view.shape().to_vec();
            if dims.len() != 4 {
                return Err(QwenError::TensorRankMismatch {
                    name: name.to_string(),
                    shape: dims,
                });
            }
            let data: Vec<f32> = view.iter().copied().collect();
            return Array4::from_shape_vec((dims[0], dims[1], dims[2], dims[3]), data).map_err(
                |e| QwenError::TensorMaterialise {
                    name: name.to_string(),
                    source: e,
                },
            );
        }

        if let Ok(tensor) = value.try_extract::<f16>() {
            let view = tensor.view();
            let dims = view.shape().to_vec();
            if dims.len() != 4 {
                return Err(QwenError::TensorRankMismatch {
                    name: name.to_string(),
                    shape: dims,
                });
            }
            let data: Vec<f32> = view.iter().map(|&x| f16::to_f32(x)).collect();
            return Array4::from_shape_vec((dims[0], dims[1], dims[2], dims[3]), data).map_err(
                |e| QwenError::TensorMaterialise {
                    name: name.to_string(),
                    source: e,
                },
            );
        }

        Err(QwenError::UnsupportedTensorType {
            name: name.to_string(),
        })
    }

    /// Merge existing KV cache with newly returned present tensors
    fn merge_kv_cache(
        existing: &Array4<f32>,
        present: Array4<f32>,
        new_tokens: usize,
        name: &str,
    ) -> QwenResult<Array4<f32>> {
        let previous_len = existing.shape()[2];
        let present_len = present.shape()[2];

        if present_len == previous_len + new_tokens {
            // Model returned full sequence (past + present): trust it outright
            return Ok(present);
        }

        if present_len == new_tokens {
            // Model returned only the new tokens: concatenate with past cache
            let merged = concatenate(Axis(2), &[existing.view(), present.view()]).map_err(|e| {
                QwenError::KvConcat {
                    name: name.to_string(),
                    source: e,
                }
            })?;
            return Ok(merged);
        }

        if present_len >= previous_len {
            // Fallback: prefer present (assume model handled accumulation)
            return Ok(present);
        }

        Err(QwenError::InvalidKvShape {
            name: name.to_string(),
            previous: previous_len,
            new_tokens,
            present: present_len,
        })
    }

    fn truncate_cache_if_needed(&mut self) {
        let window = self.config.cache_window;
        if self.current_seq_len <= window {
            return;
        }

        let before = self.current_seq_len;
        let trim = before - window;

        for layer in 0..self.config.num_layers {
            let key_name = format!("past_key_values.{}.key", layer);
            if let Some(tensor) = self.kv_cache.get_mut(&key_name) {
                if tensor.shape()[2] > window {
                    let trimmed = tensor.slice(s![.., .., trim.., ..]).to_owned();
                    *tensor = trimmed;
                }
            }

            let value_name = format!("past_key_values.{}.value", layer);
            if let Some(tensor) = self.kv_cache.get_mut(&value_name) {
                if tensor.shape()[2] > window {
                    let trimmed = tensor.slice(s![.., .., trim.., ..]).to_owned();
                    *tensor = trimmed;
                }
            }
        }

        if trim >= self.attention_cache.len() {
            self.attention_cache.clear();
        } else {
            self.attention_cache.drain(0..trim);
        }

        self.current_seq_len = window;
        info!(
            target: "tcs-ml::qwen_embedder",
            before,
            window,
            trim,
            "Trimmed KV cache window"
        );
    }

    /// Extract embedding vector from the model logits
    fn extract_embeddings(&self, logits: &Value) -> QwenResult<Vec<f32>> {
        // Handle both f32 and f16 outputs
        let logits_vec: Vec<f32> = match logits.try_extract::<f32>() {
            Ok(tensor) => tensor.view().iter().copied().collect(),
            Err(_) => match logits.try_extract::<f16>() {
                Ok(tensor) => tensor.view().iter().map(|&x| f16::to_f32(x)).collect(),
                Err(_) => {
                    return Err(QwenError::UnsupportedTensorType {
                        name: "logits".to_string(),
                    });
                }
            },
        };

        // Logits shape should be [batch=1, seq_len, vocab_size]
        let total_elements = logits_vec.len();
        let vocab_size = self.config.vocab_size;
        if vocab_size == 0 {
            return Err(QwenError::InvalidLogitsShape {
                total_elements,
                vocab_size,
            });
        }
        let seq_len = total_elements / vocab_size;

        if total_elements != seq_len * vocab_size {
            return Err(QwenError::InvalidLogitsShape {
                total_elements,
                vocab_size,
            });
        }

        // Take last token's logits and extract configured embedding dimensions
        let last_token_start = (seq_len.saturating_sub(1)) * vocab_size;
        let embedding_size = self.config.embed_dim.min(vocab_size);

        if last_token_start + embedding_size > logits_vec.len() {
            return Err(QwenError::InsufficientLogits);
        }

        let mut embeddings: Vec<f32> =
            logits_vec[last_token_start..last_token_start + embedding_size].to_vec();

        // Ensure exact configured embedding dimensions
        embeddings.resize(self.config.embed_dim, 0.0);

        Ok(embeddings)
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn merge_returns_present_when_full_sequence() {
        let existing = Array4::from_shape_vec((1, 1, 3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let present =
            Array4::from_shape_vec((1, 1, 5, 1), vec![10.0, 11.0, 12.0, 13.0, 14.0]).unwrap();
        let present_clone = present.clone();

        let merged = QwenEmbedder::merge_kv_cache(&existing, present, 2, "layer").unwrap();
        assert_eq!(merged, present_clone);
    }

    #[test]
    fn merge_appends_when_incremental_present() {
        let existing = Array4::from_shape_vec((1, 1, 3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let present = Array4::from_shape_vec((1, 1, 2, 1), vec![3.0, 4.0]).unwrap();

        let merged = QwenEmbedder::merge_kv_cache(&existing, present, 2, "layer").unwrap();
        assert_eq!(merged.shape(), &[1, 1, 5, 1]);
        let values: Vec<f32> = merged.iter().copied().collect();
        let expected = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(values, expected);
    }

    #[test]
    fn merge_falls_back_when_present_expands_beyond_sum() {
        let existing = Array4::from_shape_vec((1, 1, 3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let present = Array4::from_shape_vec((1, 1, 6, 1), vec![5.0; 6]).unwrap();

        let merged = QwenEmbedder::merge_kv_cache(&existing, present.clone(), 1, "layer").unwrap();
        assert_eq!(merged, present);
    }

    #[test]
    fn merge_errors_when_present_shrinks_context() {
        let existing = Array4::from_shape_vec((1, 1, 3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let present = Array4::from_shape_vec((1, 1, 2, 1), vec![5.0, 6.0]).unwrap();
        let err = QwenEmbedder::merge_kv_cache(&existing, present, 1, "layer");
        assert!(err.is_err());
    }
}
