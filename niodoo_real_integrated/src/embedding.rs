use anyhow::{Context, Result};
use once_cell::sync::OnceCell;
use blake3::hash;
use rand::{rngs::StdRng, Rng, SeedableRng};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::sleep;
use tracing::{debug, info, instrument, warn};

/// Wraps Ollama /api/embeddings API in an async-friendly interface.
#[derive(Clone)]
pub struct QwenStatefulEmbedder {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    expected_dim: usize,
    max_chunk_chars: usize,
    concurrency_limiter: Arc<Semaphore>,
    max_retries: u32,
    retry_delay_ms: u64,
}

#[derive(Serialize)]
struct OllamaEmbeddingRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct OllamaEmbeddingResponse {
    embedding: Vec<f32>,
}

impl QwenStatefulEmbedder {
    const DEFAULT_MAX_CONCURRENCY: usize = 8;
    const DEFAULT_MAX_RETRIES: u32 = 3;
    const DEFAULT_RETRY_DELAY_MS: u64 = 50;

    pub fn new(
        endpoint: &str,
        model: &str,
        expected_dim: usize,
        max_chunk_chars: usize,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("failed to create HTTP client for embeddings")?;

        let chunk_limit = max_chunk_chars.max(1);
        info!(chunk_limit, "initialized embedding client with chunk limit");

        static GLOBAL_EMBED_SEMAPHORE: OnceCell<Arc<Semaphore>> = OnceCell::new();

        let concurrency = std::env::var("EMBEDDING_MAX_CONCURRENCY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(Self::DEFAULT_MAX_CONCURRENCY);

        let semaphore = GLOBAL_EMBED_SEMAPHORE
            .get_or_init(|| {
                info!(limit = concurrency, "embedding concurrency limiter initialized");
                Arc::new(Semaphore::new(concurrency))
            })
            .clone();

        let max_retries = std::env::var("EMBEDDING_MAX_RETRIES")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(Self::DEFAULT_MAX_RETRIES);

        let retry_delay_ms = std::env::var("EMBEDDING_RETRY_DELAY_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(Self::DEFAULT_RETRY_DELAY_MS);

        Ok(Self {
            client,
            endpoint: endpoint.trim_end_matches('/').to_string(),
            model: model.to_string(),
            expected_dim,
            max_chunk_chars: chunk_limit,
            concurrency_limiter: semaphore,
            max_retries,
            retry_delay_ms,
        })
    }

    #[instrument(skip_all, fields(chars = prompt.len()))]
    pub async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        let chunks = chunk_text(prompt, self.max_chunk_chars);
        let chunk_count = chunks.len();
        if chunk_count > 1 {
            debug!(
                chunk_count,
                chunk_limit = self.max_chunk_chars,
                "embedding prompt split due to configured chunk limit"
            );
        }

        // Process chunks in parallel for better performance
        let embedding_results: Vec<Result<Vec<f32>, anyhow::Error>> = futures::future::join_all(
            chunks.iter().map(|chunk| self.fetch_embedding(chunk))
        ).await;
        
        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        for result in embedding_results {
            embeddings.push(result?);
        }
        
        // Accumulate embeddings
        let mut accum: Option<Vec<f32>> = None;
        let count = embeddings.len() as f32;
        
        for embedding in embeddings {
            if let Some(ref mut total) = accum {
                for (sum, value) in total.iter_mut().zip(embedding.iter()) {
                    *sum += *value;
                }
            } else {
                accum = Some(embedding);
            }
        }

        let mut combined = accum.unwrap_or_else(|| vec![0.0; self.expected_dim]);
        if count > 1.0 {
            for value in combined.iter_mut() {
                *value /= count;
            }
            normalize(&mut combined);
        }

        Ok(combined)
    }

    async fn fetch_embedding(&self, prompt: &str) -> Result<Vec<f32>> {
        let permit = self
            .concurrency_limiter
            .clone()
            .acquire_owned()
            .await
            .context("embedding concurrency limiter closed")?;

        let url = format!("{}/api/embeddings", self.endpoint);
        let request = OllamaEmbeddingRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
        };
        let mut attempt = 0;
        let mut last_error = None;

        loop {
            let response_result = self
                .client
                .post(&url)
                .json(&request)
                .send()
                .await;

            match response_result {
                Ok(response) => {
                    let status = response.status();
                    if !status.is_success() {
                        let error_text = response
                            .text()
                            .await
                            .unwrap_or_else(|_| String::from("<no body>"));

                        if status == StatusCode::NOT_FOUND {
                            warn!(
                                model = %self.model,
                                endpoint = %self.endpoint,
                                "Embedding model missing on Ollama endpoint; using deterministic fallback embedding"
                            );
                            drop(permit);
                            return Ok(self.fallback_embedding(prompt));
                        }

                        last_error = Some(anyhow::anyhow!(
                            "Ollama embeddings API returned {}: {}",
                            status,
                            error_text
                        ));
                    } else {
                        let embed_response: OllamaEmbeddingResponse = response
                            .json()
                            .await
                            .context("failed to parse Ollama embedding response")?;

                        drop(permit);

                        let mut embedding = embed_response.embedding;

                        if embedding.len() != self.expected_dim {
                            anyhow::bail!(
                                "Embedding dimension mismatch: expected {}, got {}. This indicates a model configuration error.",
                                self.expected_dim,
                                embedding.len()
                            );
                        }

                        normalize(&mut embedding);
                        return Ok(embedding);
                    }
                }
                Err(error) => {
                    last_error = Some(anyhow::anyhow!(
                        "failed to send embedding request to embedding backend: {}",
                        error
                    ));
                }
            }

            attempt += 1;
            if attempt > self.max_retries {
                drop(permit);
                return Err(last_error.unwrap_or_else(|| anyhow::anyhow!(
                    "embedding request failed after {} attempts",
                    self.max_retries + 1
                )));
            }

            let backoff_ms = self.retry_delay_ms * 2_u64.saturating_pow(attempt - 1);
            sleep(Duration::from_millis(backoff_ms)).await;
        }
    }

    fn fallback_embedding(&self, prompt: &str) -> Vec<f32> {
        let mut seed = [0u8; 32];
        seed.copy_from_slice(hash(prompt.as_bytes()).as_bytes());

        let mut rng = StdRng::from_seed(seed);
        let mut embedding = Vec::with_capacity(self.expected_dim);
        for _ in 0..self.expected_dim {
            embedding.push(rng.gen_range(-1.0..=1.0));
        }
        normalize(&mut embedding);
        embedding
    }
}

fn normalize(vec: &mut [f32]) {
    let norm = vec.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    if norm == 0.0 {
        return;
    }
    for v in vec.iter_mut() {
        *v = (*v as f64 / norm) as f32;
    }
    info!(dim = vec.len(), "normalized embedding to hypersphere");
}

fn chunk_text(input: &str, max_chars: usize) -> Vec<String> {
    if input.len() <= max_chars {
        return vec![input.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for word in input.split_whitespace() {
        if current.len() + word.len() + 1 > max_chars && !current.is_empty() {
            chunks.push(current.trim().to_string());
            current.clear();
        }

        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }

    if !current.is_empty() {
        chunks.push(current.trim().to_string());
    }

    chunks
}
