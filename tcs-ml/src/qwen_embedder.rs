use anyhow::{anyhow, Context, Result};
use ndarray::{concatenate, Array2, Array4, Axis, CowArray};
use ort::{Environment, GraphOptimizationLevel, Session, SessionBuilder, Value};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "tokenizers")]
use tokenizers::Tokenizer;

use crate::f16;
use crate::qwen_config::QwenConfig;

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
    pub fn new(model_path: &str) -> Result<Self> {
        Self::with_config(model_path, QwenConfig::default())
    }

    /// Create embedder with custom configuration
    pub fn with_config(model_path: &str, config: QwenConfig) -> Result<Self> {
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
                        println!("✓ Loaded tokenizer from: {:?}", tokenizer_path);
                        Some(t)
                    }
                    Err(e) => {
                        println!("⚠ Failed to load tokenizer: {}, using fallback", e);
                        None
                    }
                }
            } else {
                println!(
                    "⚠ Tokenizer not found at {:?}, using fallback",
                    tokenizer_path
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
    fn tokenize(&self, prompt: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        #[cfg(feature = "tokenizers")]
        {
            if let Some(ref tokenizer) = self.tokenizer {
                let encoding = tokenizer
                    .encode(prompt, true)
                    .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
                let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
                let attention_mask: Vec<i64> = encoding
                    .get_attention_mask()
                    .iter()
                    .map(|&x| x as i64)
                    .collect();
                return Ok((input_ids, attention_mask));
            }
        }

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
            let empty_cache = Array4::<f32>::zeros((1, self.config.num_heads, 0, self.config.head_dim));
            self.kv_cache.insert(key_name, empty_cache.clone());
            self.kv_cache.insert(value_name, empty_cache);
        }
        self.current_seq_len = 0;
        self.attention_cache.clear();
    }

    /// Stateful embedding: takes prompt, updates KV cache, returns 512-dim vec
    pub fn embed(&mut self, prompt: &str) -> Result<Vec<f32>> {
        let (tokens, raw_attention_mask) = self.tokenize(prompt)?;
        if tokens.is_empty() {
            return Err(anyhow!("Prompt produced no tokens"));
        }

        let attention_mask = if raw_attention_mask.len() == tokens.len() {
            raw_attention_mask
        } else {
            if !raw_attention_mask.is_empty() {
                println!(
                    "⚠️ Tokenizer attention mask length {} mismatched token count {}, falling back to ones",
                    raw_attention_mask.len(),
                    tokens.len()
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

    fn run_inference_step(&mut self, step_tokens: &[i64], step_mask: &[i64]) -> Result<Vec<f32>> {
        let seq_len = step_tokens.len();
        if seq_len == 0 {
            return Err(anyhow!("Inference step received no tokens"));
        }

        if step_mask.len() != seq_len {
            return Err(anyhow!(
                "Attention mask length {} does not match token count {}",
                step_mask.len(),
                seq_len
            ));
        }

        let total_seq_len = self.current_seq_len + seq_len;
        if total_seq_len > self.config.max_seq_len {
            return Err(anyhow!(
                "Sequence too long: total context {} would exceed max_seq_len {}",
                total_seq_len,
                self.config.max_seq_len
            ));
        }

        let batch_size = 1;

        // Create all tensors and keep them alive for the entire inference call
        let input_ids_array = Array2::from_shape_vec((batch_size, seq_len), step_tokens.to_vec())?;

        let mut attention_total = Vec::with_capacity(self.attention_cache.len() + seq_len);
        attention_total.extend_from_slice(&self.attention_cache);
        attention_total.extend_from_slice(step_mask);
        debug_assert_eq!(attention_total.len(), total_seq_len);

        let attention_mask_array =
            Array2::from_shape_vec((batch_size, attention_total.len()), attention_total.clone())?;

        let position_ids: Vec<i64> =
            (self.current_seq_len as i64..(self.current_seq_len + seq_len) as i64).collect();
        let position_ids_array = Array2::from_shape_vec((batch_size, seq_len), position_ids)?;

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
        let input_ids_value = Value::from_array(self.session.allocator(), &input_ids_cow)?;
        let attention_mask_value =
            Value::from_array(self.session.allocator(), &attention_mask_cow)?;
        let position_ids_value = Value::from_array(self.session.allocator(), &position_ids_cow)?;

        let mut kv_values = Vec::with_capacity(self.config.num_layers * 2);
        for kv_cow in &kv_cows {
            let kv_value = Value::from_array(self.session.allocator(), kv_cow)?;
            kv_values.push(kv_value);
        }

        // Combine all input values
        let mut input_values = vec![input_ids_value, attention_mask_value, position_ids_value];
        input_values.extend(kv_values);

        if seq_len > 1 || self.current_seq_len == 0 {
            println!("Running ONNX inference with {} inputs", input_values.len());
        }

        // Run inference (all CowArrays are still alive here)
        let outputs = self
            .session
            .run(input_values)
            .context("Failed to execute ONNX session")?;

        if outputs.is_empty() {
            return Err(anyhow!("ONNX model produced no outputs"));
        }

        // Ensure we received logits + KV cache outputs
        let expected_outputs = 1 + self.config.num_layers * 2;
        if outputs.len() < expected_outputs {
            return Err(anyhow!(
                "Expected at least {} outputs (logits + KV cache), got {}",
                expected_outputs,
                outputs.len()
            ));
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

        if seq_len > 1 || previous_context_len == 0 {
            println!(
                "✓ Extracted {}-dim embeddings, context length: {}",
                embeddings.len(),
                self.current_seq_len
            );
        }

        Ok(embeddings)
    }

    /// Extract KV cache tensor as owned Array4<f32>
    fn extract_kv_tensor(value: &Value, name: &str) -> Result<Array4<f32>> {
        if let Ok(tensor) = value.try_extract::<f32>() {
            let view = tensor.view();
            let dims = view.shape().to_vec();
            if dims.len() != 4 {
                return Err(anyhow!(
                    "{}: expected 4D f32 tensor, got shape {:?}",
                    name,
                    dims
                ));
            }
            let data: Vec<f32> = view.iter().copied().collect();
            return Array4::from_shape_vec((dims[0], dims[1], dims[2], dims[3]), data)
                .map_err(|e| anyhow!("{}: failed to construct f32 tensor: {}", name, e));
        }

        if let Ok(tensor) = value.try_extract::<f16>() {
            let view = tensor.view();
            let dims = view.shape().to_vec();
            if dims.len() != 4 {
                return Err(anyhow!(
                    "{}: expected 4D f16 tensor, got shape {:?}",
                    name,
                    dims
                ));
            }
            let data: Vec<f32> = view.iter().map(|&x| f16::to_f32(x)).collect();
            return Array4::from_shape_vec((dims[0], dims[1], dims[2], dims[3]), data)
                .map_err(|e| anyhow!("{}: failed to convert f16 tensor to f32: {}", name, e));
        }

        Err(anyhow!("{}: tensor is neither f32 nor f16", name))
    }

    /// Merge existing KV cache with newly returned present tensors
    fn merge_kv_cache(
        existing: &Array4<f32>,
        present: Array4<f32>,
        new_tokens: usize,
        name: &str,
    ) -> Result<Array4<f32>> {
        let previous_len = existing.shape()[2];
        let present_len = present.shape()[2];

        if present_len == previous_len + new_tokens {
            // Model returned full sequence (past + present): trust it outright
            return Ok(present);
        }

        if present_len == new_tokens {
            // Model returned only the new tokens: concatenate with past cache
            let merged = concatenate(Axis(2), &[existing.view(), present.view()])
                .map_err(|e| anyhow!("{}: failed to concatenate KV cache: {}", name, e))?;
            return Ok(merged);
        }

        if present_len >= previous_len {
            // Fallback: prefer present (assume model handled accumulation)
            return Ok(present);
        }

        Err(anyhow!(
            "{}: unexpected KV cache lengths (previous={}, new_tokens={}, present={})",
            name,
            previous_len,
            new_tokens,
            present_len
        ))
    }

    /// Extract 512-dimensional embeddings from logits
    fn extract_embeddings(&self, logits: &Value) -> Result<Vec<f32>> {
        // Handle both f32 and f16 outputs
        let logits_vec: Vec<f32> = match logits.try_extract::<f32>() {
            Ok(tensor) => tensor.view().iter().copied().collect(),
            Err(_) => {
                // Try f16 and convert to f32
                match logits.try_extract::<f16>() {
                    Ok(tensor) => tensor.view().iter().map(|&x| f16::to_f32(x)).collect(),
                    Err(e) => return Err(anyhow!("Failed to extract tensor as f32 or f16: {}", e)),
                }
            }
        };

        // Logits shape should be [batch=1, seq_len, vocab_size]
        let total_elements = logits_vec.len();
        let vocab_size = self.config.vocab_size;
        let seq_len = total_elements / vocab_size;

        if total_elements != seq_len * vocab_size {
            return Err(anyhow!(
                "Unexpected logits shape: total_elements={}, expected batch*seq*vocab",
                total_elements
            ));
        }

        // Take last token's logits and extract configured embedding dimensions
        let last_token_start = (seq_len - 1) * vocab_size;
        let embedding_size = self.config.embed_dim.min(vocab_size);

        if last_token_start + embedding_size > logits_vec.len() {
            return Err(anyhow!(
                "Cannot extract embedding: insufficient logits data"
            ));
        }

        let mut embeddings: Vec<f32> =
            logits_vec[last_token_start..last_token_start + embedding_size].to_vec();

        // Ensure exact configured embedding dimensions
        embeddings.resize(self.config.embed_dim, 0.0);

        Ok(embeddings)
    }

    /// Reset KV cache for fresh context (new conversation/consciousness thread)
    pub fn reset_cache(&mut self) {
        println!("🔄 Resetting KV cache for fresh context");
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

    #[test]
    fn test_kv_cache_initialization() {
        let config = QwenConfig::default();
        let mut embedder = QwenEmbedder::with_config("dummy_path", config.clone()).unwrap();
        embedder.init_kv_cache();

        assert_eq!(embedder.kv_cache.len(), config.num_layers * 2); // key + value per layer
        assert_eq!(embedder.current_seq_len, 0);

        for layer in 0..config.num_layers {
            let key_name = format!("past_key_values.{}.key", layer);
            let value_name = format!("past_key_values.{}.value", layer);

            let key_cache = embedder.kv_cache.get(&key_name).unwrap();
            let value_cache = embedder.kv_cache.get(&value_name).unwrap();

            assert_eq!(key_cache.shape(), &[1, config.num_heads, 0, config.head_dim]);
            assert_eq!(value_cache.shape(), &[1, config.num_heads, 0, config.head_dim]);
        }
    }

    #[test]
    fn test_tokenization_fallback() {
        let embedder = QwenEmbedder::new("dummy_path").unwrap();
        let (input_ids, attention_mask) = embedder.tokenize("Hello, world!").unwrap();

        assert!(!input_ids.is_empty());
        assert_eq!(input_ids.len(), attention_mask.len());
        assert!(attention_mask.iter().all(|&x| x == 1));
    }
}
