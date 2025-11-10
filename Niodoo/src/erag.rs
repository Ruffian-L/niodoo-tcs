use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use crate::embedding::LocalEmbedder;

/// ERAG configuration loaded from TOML
#[derive(Debug, Deserialize)]
pub struct EragConfig {
    pub qdrant_url: String,
    pub collection: String,
    pub similarity_threshold: f64,
    pub limit: usize,
}

impl EragConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read ERAG config from {}", path))?;
        let config: EragConfig = toml::from_str(&content)
            .with_context(|| format!("failed to parse ERAG config from {}", path))?;
        Ok(config)
    }
}

/// Search result from Qdrant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: Option<String>,
    pub score: f64,
    pub payload: Value,
}

/// ERAG service for memory retrieval
pub struct EragService {
    config: EragConfig,
    http_client: reqwest::Client,
    embedder: LocalEmbedder,
}

impl EragService {
    /// Initialize ERAG service from config file
    pub async fn initialise(config_path: &str) -> Result<Self> {
        let config = EragConfig::from_file(config_path)?;
        let http_client = reqwest::Client::new();
        let embedder = LocalEmbedder::from_env()?;

        Ok(Self {
            config,
            http_client,
            embedder,
        })
    }

    /// Embed prompt and search Qdrant for similar memories
    pub async fn embed_and_search(
        &self,
        prompt: &str,
        compass_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        // Embed the prompt (returns Vec<f32>, convert to Vec<f64> for Qdrant)
        let embedding_f32 = self.embedder.embed(prompt)?;
        let embedding: Vec<f64> = embedding_f32.iter().map(|&x| x as f64).collect();

        // Build Qdrant search request
        let search_url = format!(
            "{}/collections/{}/points/search",
            self.config.qdrant_url,
            self.config.collection
        );

        // Build filter if compass is provided
        let mut filter: Option<Value> = None;
        if let Some(compass) = compass_filter {
            filter = Some(json!({
                "must": [
                    {
                        "key": "compass_quadrant",
                        "match": {
                            "value": compass
                        }
                    }
                ]
            }));
        }

        let search_request = json!({
            "vector": embedding,
            "limit": self.config.limit,
            "score_threshold": self.config.similarity_threshold,
            "with_payload": true,
            "with_vectors": false,
        });

        // Add filter if present
        let mut request_body = search_request;
        if let Some(ref f) = filter {
            request_body["filter"] = f.clone();
        }

        // Execute search
        let response = self
            .http_client
            .post(&search_url)
            .json(&request_body)
            .send()
            .await
            .with_context(|| format!("failed to search Qdrant at {}", search_url))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Qdrant search failed with status {}: {}", status, text);
        }

        let search_response: Value = response
            .json()
            .await
            .context("failed to parse Qdrant search response")?;

        // Parse results
        let mut results = Vec::new();
        if let Some(result_array) = search_response.get("result").and_then(|r| r.as_array()) {
            for hit in result_array {
                let id = hit.get("id").and_then(|id| {
                    id.as_str().map(|s| s.to_string())
                        .or_else(|| id.as_u64().map(|n| n.to_string()))
                });
                let score = hit
                    .get("score")
                    .and_then(|s| s.as_f64())
                    .unwrap_or(0.0);
                let payload = hit
                    .get("payload")
                    .cloned()
                    .unwrap_or_else(|| json!({}));

                results.push(SearchResult {
                    id,
                    score,
                    payload,
                });
            }
        }

        Ok(results)
    }
}
