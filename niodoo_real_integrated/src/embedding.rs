use anyhow::{Context, Result};
use once_cell::sync::OnceCell;
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{error, info, instrument, warn};

/// Wraps Ollama /api/embeddings API in an async-friendly interface without fallbacks.
#[derive(Clone)]
pub struct QwenStatefulEmbedder {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    expected_dim: usize,
    max_chunk_chars: usize,
    concurrency_limiter: Arc<Semaphore>,
    mock_mode: bool,
}

#[derive(Deserialize)]
struct OllamaEmbeddingResponse {
    embedding: Vec<f32>,
}

impl QwenStatefulEmbedder {
    const DEFAULT_MAX_CONCURRENCY: usize = 8;

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
        info!(
            chunk_limit,
            model, endpoint, "Initialized Ollama embedding client"
        );

        static GLOBAL_EMBED_SEMAPHORE: OnceCell<Arc<Semaphore>> = OnceCell::new();

        let concurrency = std::env::var("EMBEDDING_MAX_CONCURRENCY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(Self::DEFAULT_MAX_CONCURRENCY);

        let semaphore = GLOBAL_EMBED_SEMAPHORE
            .get_or_init(|| {
                info!(
                    limit = concurrency,
                    "Embedding concurrency limiter initialized"
                );
                Arc::new(Semaphore::new(concurrency))
            })
            .clone();

        Ok(Self {
            client,
            endpoint: endpoint.trim_end_matches('/').to_string(),
            model: model.to_string(),
            expected_dim,
            max_chunk_chars: chunk_limit,
            concurrency_limiter: semaphore,
            mock_mode: false,
        })
    }

    pub fn set_mock_mode(&mut self, mock_mode: bool) {
        if mock_mode && !self.mock_mode {
            warn!("Embedding mock mode enabled; returning zero vectors");
        }
        self.mock_mode = mock_mode;
    }

    #[instrument(skip_all, fields(chars = prompt.len()))]
    pub async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        let dim = self.expected_dim;
        if prompt.is_empty() {
            return Ok(vec![0.0; dim]);
        }
        if self.mock_mode {
            return Ok(vec![0.0; self.expected_dim]);
        }
        let chunks = chunk_text(prompt, self.max_chunk_chars);
        let chunk_count = chunks.len();
        if chunk_count > 1 {
            info!(chunk_count, "Embedding prompt split due to chunk limit");
        }

        let embedding_results =
            futures::future::join_all(chunks.iter().map(|chunk| self.fetch_embedding(chunk))).await;

        let mut embeddings = Vec::with_capacity(embedding_results.len());
        for result in embedding_results {
            embeddings.push(result?);
        }

        let mut accumulated: Option<Vec<f32>> = None;
        let count = embeddings.len() as f32;

        for embedding in embeddings {
            if let Some(acc) = accumulated.as_mut() {
                for (acc_value, embedding_value) in acc.iter_mut().zip(embedding.iter()) {
                    *acc_value += *embedding_value;
                }
            } else {
                accumulated = Some(embedding);
            }
        }

        let mut combined = accumulated.unwrap_or_else(|| vec![0.0; self.expected_dim]);
        if count > 1.0 {
            for value in combined.iter_mut() {
                *value /= count;
            }
        }

        if combined.len() != self.expected_dim {
            let len = combined.len();
            warn!(
                expected = self.expected_dim,
                actual = len,
                "Aggregated embedding dimension mismatch; padding/truncating"
            );
            if len < self.expected_dim {
                combined.resize(self.expected_dim, 0.0);
            } else {
                combined.truncate(self.expected_dim);
            }
            return Ok(combined);
        }

        normalize(&mut combined);
        info!(dim = self.expected_dim, "Ollama Qwen embed complete");

        Ok(combined)
    }

    async fn fetch_embedding(&self, prompt: &str) -> Result<Vec<f32>> {
        if self.mock_mode {
            return Ok(vec![0.0; self.expected_dim]);
        }
        let permit = self
            .concurrency_limiter
            .clone()
            .acquire_owned()
            .await
            .context("embedding concurrency limiter closed")?;

        let url = format!("{}/api/embeddings", self.endpoint);
        let request_body = json!({
            "model": self.model,
            "prompt": prompt,
        });

        let response = self
            .client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .context("failed to contact Ollama embeddings endpoint")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            drop(permit);

            if status == StatusCode::NOT_FOUND {
                error!(
                    endpoint = %self.endpoint,
                    model = %self.model,
                    "Ollama down—run 'ollama serve && ollama pull qwen2:0.5b'"
                );
                anyhow::ensure!(
                    self.mock_mode,
                    "Ollama missing model {}; run 'ollama serve && ollama pull qwen2:0.5b'",
                    self.model
                );
                warn!("Embedding mock fallback engaged after 404");
                return Ok(vec![0.0; self.expected_dim]);
            }

            error!(%status, body = %body, "Ollama embeddings request failed");
            anyhow::ensure!(
                self.mock_mode,
                "Ollama embeddings request failed: status={status}, body={body}"
            );
            warn!(
                "Embedding mock fallback engaged after error status {}",
                status
            );
            return Ok(vec![0.0; self.expected_dim]);
        }

        let response_body: OllamaEmbeddingResponse = response
            .json()
            .await
            .context("failed to parse Ollama embedding response")?;

        drop(permit);

        if response_body.embedding.len() != self.expected_dim {
            let len = response_body.embedding.len();
            if len == 0 {
                warn!(
                    expected = self.expected_dim,
                    "Embedding service returned empty vector; substituting zeros"
                );
                return Ok(vec![0.0; self.expected_dim]);
            }

            warn!(
                expected = self.expected_dim,
                actual = len,
                "Embedding dimension mismatch; padding/truncating response"
            );
            let mut adjusted = response_body.embedding;
            if len < self.expected_dim {
                adjusted.resize(self.expected_dim, 0.0);
            } else {
                adjusted.truncate(self.expected_dim);
            }
            return Ok(adjusted);
        }

        Ok(response_body.embedding)
    }
}

fn normalize(vec: &mut [f32]) {
    let norm = vec
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return;
    }
    for value in vec.iter_mut() {
        *value = (*value as f64 / norm) as f32;
    }
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
