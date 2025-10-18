use anyhow::{anyhow, Context, Result};
use ndarray::{Array2, Array4, Axis, CowArray};
use ort::{Environment, GraphOptimizationLevel, Session, SessionBuilder, Value};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "tokenizers")]
use tokenizers::Tokenizer;

use crate::f16;

// Qwen2.5-Coder specifics (configurable via model config)
const NUM_LAYERS: usize = 24;
const NUM_HEADS: usize = 2; // Simplified for 0.5B model (not 7B)
const HEAD_DIM: usize = 64;
const HIDDEN_SIZE: usize = 1536; // Qwen2.5-0.5B hidden size
const MAX_SEQ_LEN: usize = 2048;
const EMBED_DIM: usize = 512; // TCS target dimension

/// Stateful Qwen embedder with KV cache management
#[derive(Debug)]
pub struct QwenEmbedder {
    session: Session,
    #[cfg(feature = "tokenizers")]
    tokenizer: Option<Tokenizer>,
    kv_cache: HashMap<String, Array4<f32>>, // [batch, heads, seq, head_dim]
    current_seq_len: usize,
}

impl QwenEmbedder {
    pub fn new(model_path: &str) -> Result<Self> {
        let env = Arc::new(Environment::builder()
            .with_name("qwen_embedder")
            .build()?);
        
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
                println!("⚠ Tokenizer not found at {:?}, using fallback", tokenizer_path);
                None
            }
        };

        Ok(Self {
            session,
            #[cfg(feature = "tokenizers")]
            tokenizer,
            kv_cache: HashMap::new(),
            current_seq_len: 0,
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
                let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
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
        for layer in 0..NUM_LAYERS {
            let key_name = format!("past_key_values.{}.key", layer);
            let value_name = format!("past_key_values.{}.value", layer);
            
            // Empty cache: [batch=1, heads, seq_len=0, head_dim]
            let empty_cache = Array4::<f32>::zeros((1, NUM_HEADS, 0, HEAD_DIM));
            self.kv_cache.insert(key_name, empty_cache.clone());
            self.kv_cache.insert(value_name, empty_cache);
        }
        self.current_seq_len = 0;
    }

    /// Create all 51 input tensors for ONNX model
    fn create_input_tensors(&self, input_ids: &[i64], attention_mask: &[i64]) -> Result<Vec<Value>> {
        let seq_len = input_ids.len();
        let batch_size = 1;

        let mut input_values = Vec::new();

        // 1. input_ids [batch, seq]
        let input_ids_array = ndarray::Array2::from_shape_vec((batch_size, seq_len), input_ids.to_vec())?;
        let input_ids_cow = CowArray::from(input_ids_array.into_dyn());
        let input_ids_value = Value::from_array(self.session.allocator(), &input_ids_cow)?;
        input_values.push(input_ids_value);

        // 2. attention_mask [batch, seq]
        let attention_mask_array = ndarray::Array2::from_shape_vec((batch_size, seq_len), attention_mask.to_vec())?;
        let attention_mask_cow = CowArray::from(attention_mask_array.into_dyn());
        let attention_mask_value = Value::from_array(self.session.allocator(), &attention_mask_cow)?;
        input_values.push(attention_mask_value);

        // 3. position_ids [batch, seq]
        let position_ids: Vec<i64> = (self.current_seq_len as i64..(self.current_seq_len + seq_len) as i64).collect();
        let position_ids_array = ndarray::Array2::from_shape_vec((batch_size, seq_len), position_ids)?;
        let position_ids_cow = CowArray::from(position_ids_array.into_dyn());
        let position_ids_value = Value::from_array(self.session.allocator(), &position_ids_cow)?;
        input_values.push(position_ids_value);

        // 4-51. past_key_values (48 tensors: 24 layers × 2)
        for layer in 0..NUM_LAYERS {
            let key_name = format!("past_key_values.{}.key", layer);
            let value_name = format!("past_key_values.{}.value", layer);
            
            let key_cache = self.kv_cache.get(&key_name).unwrap();
            let value_cache = self.kv_cache.get(&value_name).unwrap();
            
            let key_cow = CowArray::from(key_cache.view().into_dyn());
            let value_cow = CowArray::from(value_cache.view().into_dyn());
            
            let key_value = Value::from_array(self.session.allocator(), &key_cow)?;
            let value_value = Value::from_array(self.session.allocator(), &value_cow)?;
            
            input_values.push(key_value);
            input_values.push(value_value);
        }

        Ok(input_values)
    }

    /// Update KV cache with new key/value tensors from model output
    fn update_kv_cache(&mut self, outputs: &[Value]) -> Result<()> {
        // Skip logits (output 0), process present_key_values (outputs 1-48)
        let mut output_idx = 1;
        
        for layer in 0..NUM_LAYERS {
            let key_name = format!("past_key_values.{}.key", layer);
            let value_name = format!("past_key_values.{}.value", layer);
            
            if output_idx < outputs.len() {
                // Extract new key tensor
                let new_key_tensor = outputs[output_idx].try_extract::<f32>()
                    .context("Failed to extract key tensor as f32")?;
                let new_key_array = Array4::from_shape_vec(
                    (1, NUM_HEADS, new_key_tensor.shape()[2], HEAD_DIM),
                    new_key_tensor.view().to_vec()
                )?;
                
                // Extract new value tensor
                let new_value_tensor = outputs[output_idx + 1].try_extract::<f32>()
                    .context("Failed to extract value tensor as f32")?;
                let new_value_array = Array4::from_shape_vec(
                    (1, NUM_HEADS, new_value_tensor.shape()[2], HEAD_DIM),
                    new_value_tensor.view().to_vec()
                )?;
                
                // Concatenate with existing cache along sequence dimension (axis 2)
                let old_key = self.kv_cache.get(&key_name).unwrap();
                let old_value = self.kv_cache.get(&value_name).unwrap();
                
                let updated_key = if old_key.shape()[2] == 0 {
                    new_key_array
                } else {
                    ndarray::concatenate![Axis(2), old_key.view(), new_key_array.view()]
                };
                
                let updated_value = if old_value.shape()[2] == 0 {
                    new_value_array
                } else {
                    ndarray::concatenate![Axis(2), old_value.view(), new_value_array.view()]
                };
                
                self.kv_cache.insert(key_name, updated_key);
                self.kv_cache.insert(value_name, updated_value);
                
                output_idx += 2;
            }
        }
        
        Ok(())
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
        
        // Logits shape should be [batch=1, seq_len, vocab_size=151936]
        let total_elements = logits_vec.len();
        let vocab_size = 151936; // Qwen2.5 vocab size
        let seq_len = total_elements / vocab_size;
        
        if total_elements != seq_len * vocab_size {
            return Err(anyhow!("Unexpected logits shape: total_elements={}, expected batch*seq*vocab", total_elements));
        }
        
        // Take last token's logits and extract first 512 dimensions as embedding
        let last_token_start = (seq_len - 1) * vocab_size;
        let embedding_size = EMBED_DIM.min(vocab_size);
        
        if last_token_start + embedding_size > logits_vec.len() {
            return Err(anyhow!("Cannot extract embedding: insufficient logits data"));
        }
        
        let mut embeddings: Vec<f32> = logits_vec[last_token_start..last_token_start + embedding_size].to_vec();
        
        // Ensure exactly 512 dimensions
        embeddings.resize(EMBED_DIM, 0.0);
        
        Ok(embeddings)
    }

    /// Stateful embedding: takes prompt, updates KV cache, returns 512-dim vec
    pub fn embed(&mut self, prompt: &str) -> Result<Vec<f32>> {
        // Tokenize input
        let (input_ids, attention_mask) = self.tokenize(prompt)?;
        let seq_len = input_ids.len();
        
        if seq_len > MAX_SEQ_LEN {
            return Err(anyhow!("Sequence too long: {} > {}", seq_len, MAX_SEQ_LEN));
        }
        
        // Initialize cache if this is the first call
        if self.kv_cache.is_empty() {
            self.init_kv_cache();
        }
        
        // Create all 51 input tensors
        let input_values = self.create_input_tensors(&input_ids, &attention_mask)?;
        
        println!("Running ONNX inference with {} inputs", input_values.len());
        
        // Run inference
        let outputs = self.session.run(input_values)
            .context("Failed to execute ONNX session")?;
        
        if outputs.is_empty() {
            return Err(anyhow!("ONNX model produced no outputs"));
        }
        
        // Extract embeddings from logits (first output)
        let embeddings = self.extract_embeddings(&outputs[0])?;
        
        // Update KV cache with present_key_values
        if outputs.len() > 1 {
            self.update_kv_cache(&outputs).context("Failed to update KV cache")?;
        }
        
        // Update sequence length for next inference
        self.current_seq_len += seq_len;
        
        println!("✓ Extracted {}-dim embeddings, KV cache updated (seq_len={})", 
                 embeddings.len(), self.current_seq_len);
        
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_initialization() {
        let mut embedder = QwenEmbedder::new("dummy_path").unwrap();
        embedder.init_kv_cache();
        
        assert_eq!(embedder.kv_cache.len(), NUM_LAYERS * 2); // key + value per layer
        assert_eq!(embedder.current_seq_len, 0);
        
        for layer in 0..NUM_LAYERS {
            let key_name = format!("past_key_values.{}.key", layer);
            let value_name = format!("past_key_values.{}.value", layer);
            
            let key_cache = embedder.kv_cache.get(&key_name).unwrap();
            let value_cache = embedder.kv_cache.get(&value_name).unwrap();
            
            assert_eq!(key_cache.shape(), &[1, NUM_HEADS, 0, HEAD_DIM]);
            assert_eq!(value_cache.shape(), &[1, NUM_HEADS, 0, HEAD_DIM]);
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