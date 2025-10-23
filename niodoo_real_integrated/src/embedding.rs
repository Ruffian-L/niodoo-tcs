use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, instrument};

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

    #[instrument(skip_all, fields(tokens = prompt.len()))]
    pub async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
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
