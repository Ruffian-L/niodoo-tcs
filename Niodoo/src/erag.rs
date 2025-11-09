use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::header::{HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::embedding::LocalEmbedder;

/// ERAG configuration pulled from TOML (`config/erag.toml`).
#[derive(Debug, Deserialize, Clone)]
pub struct EragConfig {
    pub qdrant: QdrantConfig,
    pub hypersphere: HypersphereConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct QdrantConfig {
    pub http_url: String,
    pub grpc_url: Option<String>, // unused in REST mode but retained for compatibility
    pub api_key: Option<String>,
    pub collection: String,
    pub vector_size: usize,
    #[serde(default = "default_distance")]
    pub distance: String,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default = "default_top_k")]
    pub query_top_k: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct HypersphereConfig {
    #[serde(default = "default_max_radius")]
    pub max_radius: f32,
    #[serde(default = "default_min_score")]
    pub min_score: f32,
    #[serde(default = "default_compass_key")]
    pub compass_payload_key: String,
}

fn default_distance() -> String {
    "Cosine".to_string()
}

fn default_top_k() -> usize {
    8
}

fn default_max_radius() -> f32 {
    0.32
}

fn default_min_score() -> f32 {
    0.15
}

fn default_compass_key() -> String {
    "compass_quadrant".to_string()
}

impl EragConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read ERAG config from {}", path))?;
        toml::from_str(&content).context("failed to parse ERAG config")
    }
}

/// Thin wrapper around Qdrant REST API for ERAG searches.
#[derive(Debug)]
pub struct EragClient {
    http: reqwest::Client,
    config: EragConfig,
}

impl EragClient {
    pub fn connect(config: EragConfig) -> Result<Self> {
        let mut headers = HeaderMap::new();
        if let Some(api_key) = &config.qdrant.api_key {
            headers.insert(
                "api-key",
                HeaderValue::from_str(api_key)
                    .with_context(|| "invalid characters in Qdrant API key")?,
            );
        }

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(15))
            .build()
            .context("failed to build Qdrant HTTP client")?;

        Ok(Self { http, config })
    }

    pub async fn search(&self, query: &[f32], compass: Option<&str>) -> Result<Vec<SearchResult>> {
        let base_url = self.config.qdrant.http_url.trim_end_matches('/');
        let url = format!(
            "{}/collections/{}/points/search",
            base_url, self.config.qdrant.collection
        );

        let mut body = json!({
            "vector": query,
            "limit": self.config.qdrant.query_top_k,
            "with_payload": true,
        });

        if self.config.hypersphere.min_score > 0.0 {
            body["score_threshold"] = Value::from(self.config.hypersphere.min_score);
        }

        if let Some(quadrant) = compass {
            body["filter"] = json!({
                "must": [{
                    "key": self.config.hypersphere.compass_payload_key,
                    "match": { "value": quadrant }
                }]
            });
        }

        let response = self
            .http
            .post(url)
            .json(&body)
            .send()
            .await
            .context("failed to send Qdrant search request")?
            .error_for_status()
            .context("Qdrant search returned error status")?
            .json::<Value>()
            .await
            .context("failed to decode Qdrant response")?;

        let results = response
            .get("result")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("Qdrant response missing result array"))?;

        let parsed = results
            .iter()
            .map(|entry| {
                let id = entry
                    .get("id")
                    .map(|v| match v {
                        Value::String(s) => Some(s.clone()),
                        Value::Number(n) => Some(n.to_string()),
                        _ => None,
                    })
                    .flatten();
                let score = entry
                    .get("score")
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| anyhow!("Qdrant result missing score"))?
                    as f32;
                let payload = entry.get("payload").cloned().unwrap_or(Value::Null);
                Ok(SearchResult { id, score, payload })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(parsed)
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: Option<String>,
    pub score: f32,
    pub payload: Value,
}

/// Combined ERAG service: embeds prompts then searches Qdrant.
#[derive(Debug)]
pub struct EragService {
    embedder: LocalEmbedder,
    pub(crate) client: EragClient,
}

impl EragService {
    pub async fn initialise(config_path: &str) -> Result<Self> {
        let config = EragConfig::from_file(config_path)?;
        let embedder = LocalEmbedder::from_env()?;
        Ok(Self {
            client: EragClient::connect(config)?,
            embedder,
        })
    }

    pub async fn embed_and_search(
        &self,
        prompt: &str,
        compass_quadrant: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let mut embedding = self.embedder.embed(prompt)?;
        let expected = self.client.config.qdrant.vector_size;
        match embedding.len().cmp(&expected) {
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Less => {
                embedding.resize(expected, 0.0);
            }
            std::cmp::Ordering::Greater => {
                embedding.truncate(expected);
            }
        }

        self.client.search(&embedding, compass_quadrant).await
    }
}
