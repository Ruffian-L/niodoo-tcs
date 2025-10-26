use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

const MAX_EMBED_CHARS: usize = 4000;

/// Wraps Ollama /api/embeddings API in an async-friendly interface.
#[derive(Clone)]
pub struct QwenStatefulEmbedder {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    expected_dim: usize,
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
    pub fn new(endpoint: &str, model: &str, expected_dim: usize) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("failed to create HTTP client for embeddings")?;

        Ok(Self {
            client,
            endpoint: endpoint.trim_end_matches('/').to_string(),
            model: model.to_string(),
            expected_dim,
        })
    }

    #[instrument(skip_all, fields(chars = prompt.len()))]
    pub async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        let chunks = chunk_text(prompt, MAX_EMBED_CHARS);
        let chunk_count = chunks.len();
        if chunk_count > 1 {
            debug!(chunk_count, "embedding prompt split for Ollama context limits");
        }

        let mut accum: Option<Vec<f32>> = None;
        let mut count = 0f32;

        for chunk in chunks {
            let embedding = self.fetch_embedding(&chunk).await?;
            if let Some(ref mut total) = accum {
                for (sum, value) in total.iter_mut().zip(embedding.iter()) {
                    *sum += *value;
                }
            } else {
                accum = Some(embedding);
            }
            count += 1.0;
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
        let url = format!("{}/api/embeddings", self.endpoint);
        let request = OllamaEmbeddingRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send embedding request to Ollama")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<no body>"));
            anyhow::bail!("Ollama embeddings API returned {}: {}", status, error_text);
        }

        let embed_response: OllamaEmbeddingResponse = response
            .json()
            .await
            .context("failed to parse Ollama embedding response")?;

        let mut embedding = embed_response.embedding;

        if embedding.len() != self.expected_dim {
            if embedding.len() < self.expected_dim {
                embedding.resize(self.expected_dim, 0.0);
            } else {
                embedding.truncate(self.expected_dim);
            }
        }

        normalize(&mut embedding);
        Ok(embedding)
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
