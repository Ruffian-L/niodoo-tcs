use anyhow::{Result, anyhow};
use async_trait::async_trait;
use base64::Engine;
use reqwest::Client;
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use std::collections::HashMap;
use std::convert::TryInto;
use std::time::Duration;
use std::time::SystemTime;
use tracing::info;

/// Document returned from vector store retrieval
#[derive(Debug, Clone)]
pub struct Document {
    pub content: String,
    pub metadata: HashMap<String, JsonValue>,
}

/// Trait for vector storage backends
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Retrieve similar documents
    async fn retrieve(&self, embedding: &[f32], k: usize) -> Result<Vec<Document>>;

    /// Upsert binary payload with embedding
    async fn upsert_binary(&self, id: &str, payload: &[u8], embedding: &[f32]) -> Result<()>;
}

/// Real Qdrant client implementing VectorStore with binary proto support
#[derive(Clone)]
pub struct RealQdrantClient {
    client: Client,
    base_url: String,
    collection: String,
}

impl RealQdrantClient {
    /// Create new RealQdrantClient
    pub fn new(url: &str, collection: &str) -> Result<Self> {
        let base_url = std::env::var("QDRANT_URL")
            .unwrap_or_else(|_| url.to_string())
            .trim_end_matches('/')
            .to_string();

        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|err| anyhow!("failed to build qdrant http client: {err}"))?;

        info!(
            collection = %collection,
            url = %base_url,
            "RealQdrantClient initialized"
        );

        Ok(Self {
            client,
            base_url,
            collection: collection.to_string(),
        })
    }
}

#[async_trait]
impl VectorStore for RealQdrantClient {
    async fn retrieve(&self, embedding: &[f32], k: usize) -> Result<Vec<Document>> {
        let request_json = json!({
            "vector": embedding.to_vec(),
            "limit": k.max(1).min(50),
            "with_payload": true,
            "with_vectors": false
        });

        let url = format!(
            "{}/collections/{}/points/search",
            self.base_url, self.collection
        );

        let response = self.client.post(&url).json(&request_json).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "Qdrant search failed: status={}, body={}",
                status,
                body
            ));
        }

        #[derive(serde::Deserialize)]
        struct SearchResponse {
            #[serde(default)]
            result: Vec<SearchHit>,
        }

        #[derive(serde::Deserialize)]
        struct SearchHit {
            #[serde(default)]
            payload: JsonMap<String, JsonValue>,
        }

        let search_resp: SearchResponse = response.json().await?;

        let docs = search_resp
            .result
            .into_iter()
            .map(|hit| {
                let payload = hit.payload;
                let mut metadata = HashMap::new();

                // Extract content if available
                let content = payload
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();

                // Store all payload as metadata
                for (key, value) in payload {
                    metadata.insert(key, value);
                }

                Document { content, metadata }
            })
            .collect();

        Ok(docs)
    }

    async fn upsert_binary(&self, id: &str, payload: &[u8], embedding: &[f32]) -> Result<()> {
        // Hash ID using blake3 for deterministic point IDs
        let hash = blake3::hash(id.as_bytes());
        let point_id = u64::from_le_bytes(
            hash.as_bytes()[..8]
                .try_into()
                .map_err(|_| anyhow!("Hash conversion failed"))?,
        );

        // Build payload map with binary proto blob as base64
        let mut payload_map = JsonMap::new();

        // Store binary proto as base64 string
        use base64::engine::general_purpose;
        payload_map.insert(
            "state_proto".to_string(),
            JsonValue::String(general_purpose::STANDARD.encode(payload)),
        );

        // Add timestamp
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        payload_map.insert("timestamp".to_string(), JsonValue::Number(timestamp.into()));

        // Add original ID for reference
        payload_map.insert("id".to_string(), JsonValue::String(id.to_string()));

        // Create upsert request
        let request_body = json!({
            "points": [
                {
                    "id": point_id,
                    "vector": embedding.to_vec(),
                    "payload": payload_map,
                }
            ]
        });

        let url = format!("{}/collections/{}/points", self.base_url, self.collection);
        let response = self.client.put(&url).json(&request_body).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "Qdrant upsert failed: status={}, body={}",
                status,
                body
            ));
        }

        info!(
            id = %id,
            size_bytes = payload.len(),
            "Upserted binary proto state to Qdrant"
        );

        Ok(())
    }
}

