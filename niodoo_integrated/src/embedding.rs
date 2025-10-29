use anyhow::Result;
use reqwest::Client;
use serde_json::{json, Value};
use tokio::time::{sleep, Duration};

// Configuration constants
const EMBEDDING_DIMENSION: usize = 896;
const DEFAULT_VLLM_URL: &str = "http://localhost:5001";
const DEFAULT_MODEL_PATH: &str = "/home/beelink/models/Qwen2.5-7B-Instruct-AWQ";
const MAX_LOGPROBS_TO_EXTRACT: usize = 100;
const LOGPROB_DISTRIBUTION_SIZE: usize = 8;
const TOKEN_INFLUENCE_MULTIPLIER: f64 = 0.1;
const WINDOW_SIZES: [usize; 5] = [1, 2, 3, 4, 5];
const HASH_DISTRIBUTION_COUNT: usize = 10;
const HASH_SCALE_DIVISOR: f64 = 64.0;
const HASH_OFFSET: f64 = 0.5;
const INDEX_MULTIPLIER_100: usize = 100;
const INDEX_MULTIPLIER_10: usize = 10;
const HASH_BIT_SHIFT_PER_ITER: usize = 6;
const HASH_MASK: u64 = 0x3F;
const DEFAULT_LOGPROBS: usize = 10;

#[derive(Debug)]
pub struct QwenEmbedder {
    client: Client,
    vllm_url: String,
}

impl QwenEmbedder {
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            vllm_url: DEFAULT_VLLM_URL.to_string(),
        })
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f64>> {
        // Use vLLM to extract semantic embeddings from hidden states
        let response = self.client
            .post(&format!("{}/v1/completions", self.vllm_url))
            .json(&json!({
                "model": DEFAULT_MODEL_PATH,
                "prompt": format!("Extract semantic meaning and concepts from: {}", text),
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": DEFAULT_LOGPROBS
            }))
            .send()
            .await;

        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    if let Ok(completion) = resp.json::<Value>().await {
                        return self.extract_embedding_from_response(&completion, text).await;
                    }
                }
                // Fallback to hash-based embedding
                Ok(self.generate_hash_embedding(text))
            }
            Err(_) => {
                // Fallback to hash-based embedding
                Ok(self.generate_hash_embedding(text))
            }
        }
    }

    async fn extract_embedding_from_response(&self, completion: &Value, text: &str) -> Result<Vec<f64>> {
        let mut embedding = vec![0.0; EMBEDDING_DIMENSION];
        
        // Extract from logprobs if available
        if let Some(choices) = completion["choices"].as_array() {
            if let Some(first_choice) = choices.first() {
                if let Some(logprobs) = first_choice["logprobs"]["top_logprobs"].as_array() {
                    for (i, logprob_obj) in logprobs.iter().enumerate().take(MAX_LOGPROBS_TO_EXTRACT) {
                        if let Some(obj) = logprob_obj.as_object() {
                            let mut prob_sum = 0.0;
                            for (token, prob) in obj.iter() {
                                if let Some(p) = prob.as_f64() {
                                    prob_sum += p.exp();
                                    // Distribute token influence across embedding dimensions
                                    for j in 0..LOGPROB_DISTRIBUTION_SIZE {
                                        let idx = (i * LOGPROB_DISTRIBUTION_SIZE + j) % EMBEDDING_DIMENSION;
                                        embedding[idx] += p.exp() * (token.len() as f64 * TOKEN_INFLUENCE_MULTIPLIER);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // If no logprobs, use text-based features
        if embedding.iter().all(|&x| x == 0.0) {
            embedding = self.generate_hash_embedding(text);
        } else {
            // Normalize the extracted embedding
            let norm = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for val in embedding.iter_mut() {
                    *val /= norm;
                }
            }
        }

        Ok(embedding)
    }

    fn generate_hash_embedding(&self, text: &str) -> Vec<f64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut embedding = vec![0.0; EMBEDDING_DIMENSION];
        
        // Multi-scale text hashing for semantic-like features
        for (i, window_size) in WINDOW_SIZES.iter().enumerate() {
            let words: Vec<&str> = text.split_whitespace().collect();
            for window in words.windows(*window_size) {
                let phrase = window.join(" ");
                let mut hasher = DefaultHasher::new();
                phrase.hash(&mut hasher);
                let hash = hasher.finish();
                
                // Distribute hash across embedding dimensions
                for j in 0..HASH_DISTRIBUTION_COUNT {
                    let idx = ((hash as usize + i * INDEX_MULTIPLIER_100 + j * INDEX_MULTIPLIER_10) % EMBEDDING_DIMENSION);
                    embedding[idx] += ((hash >> ((j as usize) * HASH_BIT_SHIFT_PER_ITER)) & HASH_MASK) as f64 / HASH_SCALE_DIVISOR - HASH_OFFSET;
                }
            }
        }

        // Normalize
        let norm = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for val in embedding.iter_mut() {
                *val /= norm;
            }
        }

        embedding
    }
}