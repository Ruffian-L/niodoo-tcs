use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::experience::Experience;

#[derive(Debug, Deserialize, Clone)]
pub struct MemoryConfig {
    pub http_url: String,
    pub collection: String,
    pub vector_size: usize,
}

impl MemoryConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read memory config from {}", path))?;
        toml::from_str(&content).context("failed to parse memory config")
    }
}

#[derive(Debug)]
pub struct ExperienceStore {
    client: Client,
    config: MemoryConfig,
}

impl ExperienceStore {
    pub async fn initialise(config_path: &str) -> Result<Self> {
        let config = MemoryConfig::from_file(config_path)?;
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(20))
            .build()
            .context("failed to build Qdrant HTTP client")?;

        let store = Self { client, config };
        store.ensure_collection().await?;
        Ok(store)
    }

    async fn ensure_collection(&self) -> Result<()> {
        let base = self.config.http_url.trim_end_matches('/');
        let url = format!("{}/collections/{}", base, self.config.collection);

        let create_body = json!({
            "vectors": {
                "size": self.config.vector_size,
                "distance": "Cosine"
            },
            "on_disk_payload": false
        });

        let response = self
            .client
            .put(&url)
            .json(&create_body)
            .send()
            .await
            .context("failed to ensure experience collection")?;

        if response.status().is_success() || response.status().as_u16() == 409 {
            return Ok(());
        }

        Err(anyhow::anyhow!(
            "failed to create collection {}: {}",
            self.config.collection,
            response.text().await.unwrap_or_default()
        ))
    }

    pub async fn upsert(&self, experience: &Experience, embedding: &[f32]) -> Result<()> {
        let mut vector = embedding.to_vec();
        match vector.len().cmp(&self.config.vector_size) {
            std::cmp::Ordering::Less => vector.resize(self.config.vector_size, 0.0),
            std::cmp::Ordering::Greater => vector.truncate(self.config.vector_size),
            std::cmp::Ordering::Equal => {}
        }

        let payload = experience.as_payload();
        let point = json!({
            "id": payload.id,
            "vector": vector,
            "payload": serde_json::to_value(&payload).expect("payload serialization"),
        });

        let base = self.config.http_url.trim_end_matches('/');
        let url = format!(
            "{}/collections/{}/points?wait=true",
            base, self.config.collection
        );

        let response = self
            .client
            .put(url)
            .json(&json!({ "points": [point] }))
            .send()
            .await
            .context("failed to upsert experience")?;

        if response.status().is_success() {
            Ok(())
        } else {
            let body = response.text().await.unwrap_or_default();
            Err(anyhow::anyhow!("experience upsert failed: {}", body))
        }
    }

    pub async fn search(&self, query: &[f32], limit: usize) -> Result<Value> {
        let base = self.config.http_url.trim_end_matches('/');
        let url = format!(
            "{}/collections/{}/points/search",
            base, self.config.collection
        );
        let body = json!({
            "vector": query,
            "limit": limit,
            "with_payload": true,
            "with_vectors": false
        });

        let response = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await
            .context("failed to search experiences")?
            .error_for_status()
            .context("experience search returned error")?
            .json::<Value>()
            .await
            .context("failed to decode experience search response")?;

        Ok(response)
    }

    pub fn vector_size(&self) -> usize {
        self.config.vector_size
    }
}
