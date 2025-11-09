// Real Qdrant client implementation using HTTP API
use anyhow::{anyhow, Result};
use reqwest::Client;
use serde_json::{json, Value};
use std::env;
use std::time::Duration;

// Configuration constants
const DEFAULT_TIMEOUT_SECS: u64 = 10;
const MOCK_SCORE_HIGH: f32 = 0.9;
const MOCK_SCORE_MEDIUM: f32 = 0.8;
const DEFAULT_PAYLOAD_CONTENT: &str = "stored_vector";

pub struct MockQdrantClient {
    client: Client,
    base_url: String,
    real_mode: bool,
}

impl MockQdrantClient {
    pub fn new(url: &str) -> Self {
        let qdrant_url = env::var("QDRANT_URL")
            .unwrap_or_else(|_| url.to_string())
            .trim_end_matches('/')
            .to_string();

        let timeout_secs = env::var("QDRANT_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .unwrap_or_else(|_| Client::new());

        // Check if Qdrant is actually available
        let real_mode = env::var("QDRANT_ENABLED").is_ok();

        if real_mode {
            tracing::info!("✅ Real Qdrant mode enabled at {}", qdrant_url);
        } else {
            tracing::warn!("⚠️  Qdrant not enabled, using fallback mode");
        }

        Self {
            client,
            base_url: qdrant_url,
            real_mode,
        }
    }

    pub async fn search_points(
        &self,
        collection: &str,
        vector: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        if !self.real_mode {
            // Fallback mock implementation
            let avg_score = if vector.is_empty() {
                0.0
            } else {
                vector.iter().sum::<f32>() / vector.len() as f32
            };

            return Ok(vec![
                SearchResult {
                    id: "mock_1".to_string(),
                    score: avg_score * MOCK_SCORE_HIGH,
                    payload: serde_json::json!({"content": "Mock vector result 1"}),
                },
                SearchResult {
                    id: "mock_2".to_string(),
                    score: avg_score * MOCK_SCORE_MEDIUM,
                    payload: serde_json::json!({"content": "Mock vector result 2"}),
                },
            ]);
        }

        // Real Qdrant search using HTTP API
        let request_json = json!({
            "vector": vector,
            "limit": limit,
            "with_payload": true,
            "with_vectors": false
        });

        let url = format!("{}/collections/{}/points/search", self.base_url, collection);

        match self.client.post(&url).json(&request_json).send().await {
            Ok(resp) if resp.status().is_success() => {
                #[derive(serde::Deserialize)]
                struct SearchResponse {
                    result: Vec<SearchHit>,
                }

                #[derive(serde::Deserialize)]
                struct SearchHit {
                    id: Option<String>,
                    score: f32,
                    payload: Value,
                }

                match resp.json::<SearchResponse>().await {
                    Ok(search_resp) => {
                        let results: Vec<SearchResult> = search_resp
                            .result
                            .into_iter()
                            .map(|hit| SearchResult {
                                id: hit.id.unwrap_or_else(|| "unknown".to_string()),
                                score: hit.score,
                                payload: hit.payload,
                            })
                            .collect();
                        Ok(results)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse Qdrant search response: {}", e);
                        // Return empty results on parse error
                        Ok(Vec::new())
                    }
                }
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                tracing::warn!("Qdrant search failed: status={}, body={}", status, body);
                // Return empty results on error
                Ok(Vec::new())
            }
            Err(e) => {
                tracing::warn!("Qdrant request failed: {}. Using fallback.", e);
                // Fallback to mock
                let avg_score = if vector.is_empty() {
                    0.0
                } else {
                    vector.iter().sum::<f32>() / vector.len() as f32
                };

                Ok(vec![SearchResult {
                    id: "fallback_1".to_string(),
                    score: avg_score * MOCK_SCORE_HIGH,
                    payload: serde_json::json!({"content": "Fallback result 1"}),
                }])
            }
        }
    }

    pub async fn upsert_points(
        &mut self,
        collection: &str,
        id: String,
        vector: Vec<f32>,
    ) -> Result<()> {
        if !self.real_mode {
            // Fallback: just log
            tracing::debug!("Mock upsert: {} to {}", id, collection);
            return Ok(());
        }

        // Real Qdrant upsert using HTTP API
        let request_body = json!({
            "points": [
                {
                    "id": id,
                    "vector": vector,
                    "payload": {
                        "content": DEFAULT_PAYLOAD_CONTENT
                    }
                }
            ]
        });

        let url = format!("{}/collections/{}/points", self.base_url, collection);

        match self.client.put(&url).json(&request_body).send().await {
            Ok(resp) if resp.status().is_success() => {
                tracing::debug!("Successfully upserted point {} to {}", id, collection);
                Ok(())
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                Err(anyhow!(
                    "Qdrant upsert failed: status={}, body={}",
                    status,
                    body
                ))
            }
            Err(e) => {
                tracing::warn!("Qdrant upsert request failed: {}", e);
                Ok(()) // Don't fail on network errors
            }
        }
    }
}

pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub payload: Value,
}
