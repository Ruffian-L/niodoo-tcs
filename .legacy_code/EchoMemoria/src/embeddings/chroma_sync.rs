//! ChromaDB synchronization for embedding storage
//! 
//! This module handles dual-write to both sidecar files and ChromaDB
//! for fast semantic search capabilities.

use std::path::Path;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use reqwest::Client;
use log::error;

use super::{EmbeddingEngine, EmbeddingMetadata};

/// ChromaDB synchronization client
pub struct ChromaSync {
    client: Client,
    base_url: String,
    collection_name: String,
    batch_size: usize,
}

/// Embedding record for ChromaDB
#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingRecord {
    pub id: String,
    pub embedding: Vec<f32>,
    pub metadata: EmbeddingMetadata,
    pub document: String,
}

/// ChromaDB response structure
#[derive(Serialize, Deserialize, Debug)]
pub struct ChromaResponse {
    pub ids: Vec<String>,
    pub embeddings: Option<Vec<Vec<f32>>>,
    pub metadatas: Option<Vec<serde_json::Value>>,
    pub documents: Option<Vec<String>>,
}

impl ChromaSync {
    /// Create a new ChromaDB sync client
    pub fn new(base_url: String, collection_name: String, batch_size: usize) -> Self {
        Self {
            client: Client::new(),
            base_url,
            collection_name,
            batch_size,
        }
    }

    /// Upsert a single embedding to ChromaDB
    pub async fn upsert_embedding(
        &self,
        file_path: &str,
        embedding: Vec<f32>,
        metadata: EmbeddingMetadata,
        document: String,
    ) -> Result<()> {
        let record = EmbeddingRecord {
            id: metadata.embedding_id.clone(),
            embedding,
            metadata,
            document,
        };

        let payload = serde_json::json!({
            "ids": [record.id],
            "embeddings": [record.embedding],
            "metadatas": [serde_json::to_value(record.metadata)?],
            "documents": [record.document]
        });

        let url = format!("{}/api/v1/collections/{}/upsert", self.base_url, self.collection_name);
        
        let response = self.client
            .post(&url)
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("ChromaDB upsert failed: {}", error_text));
        }

        Ok(())
    }

    /// Delete an embedding from ChromaDB
    pub async fn delete_embedding(&self, embedding_id: &str) -> Result<()> {
        let payload = serde_json::json!({
            "ids": [embedding_id]
        });

        let url = format!("{}/api/v1/collections/{}/delete", self.base_url, self.collection_name);
        
        let response = self.client
            .post(&url)
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("ChromaDB delete failed: {}", error_text));
        }

        Ok(())
    }

    /// Batch upsert multiple embeddings
    pub async fn batch_upsert(&self, records: Vec<EmbeddingRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        // Process in batches
        for chunk in records.chunks(self.batch_size) {
            let ids: Vec<String> = chunk.iter().map(|r| r.id.clone()).collect();
            let embeddings: Vec<Vec<f32>> = chunk.iter().map(|r| r.embedding.clone()).collect();
            let metadatas: Vec<serde_json::Value> = chunk.iter()
                .map(|r| serde_json::to_value(&r.metadata))
                .collect::<Result<Vec<_>, _>>()?;
            let documents: Vec<String> = chunk.iter().map(|r| r.document.clone()).collect();

            let payload = serde_json::json!({
                "ids": ids,
                "embeddings": embeddings,
                "metadatas": metadatas,
                "documents": documents
            });

            let url = format!("{}/api/v1/collections/{}/upsert", self.base_url, self.collection_name);
            
            let response = self.client
                .post(&url)
                .json(&payload)
                .send()
                .await?;

            if !response.status().is_success() {
                let error_text = response.text().await?;
                return Err(anyhow::anyhow!("ChromaDB batch upsert failed: {}", error_text));
            }
        }

        Ok(())
    }

    /// Query embeddings from ChromaDB
    pub async fn query_embeddings(
        &self,
        query_embedding: Vec<f32>,
        n_results: usize,
        where_clause: Option<serde_json::Value>,
    ) -> Result<ChromaResponse> {
        let mut payload = serde_json::json!({
            "query_embeddings": [query_embedding],
            "n_results": n_results
        });

        if let Some(where_clause) = where_clause {
            payload["where"] = where_clause;
        }

        let url = format!("{}/api/v1/collections/{}/query", self.base_url, self.collection_name);
        
        let response = self.client
            .post(&url)
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("ChromaDB query failed: {}", error_text));
        }

        let chroma_response: ChromaResponse = response.json().await?;
        Ok(chroma_response)
    }

    /// Check if ChromaDB is available
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/v1/heartbeat", self.base_url);
        
        match self.client.get(&url).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Get collection info
    pub async fn get_collection_info(&self) -> Result<serde_json::Value> {
        let url = format!("{}/api/v1/collections/{}", self.base_url, self.collection_name);
        
        let response = self.client
            .get(&url)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Failed to get collection info: {}", error_text));
        }

        let info: serde_json::Value = response.json().await?;
        Ok(info)
    }

    /// Create collection if it doesn't exist
    pub async fn ensure_collection(&self) -> Result<()> {
        // Check if collection exists
        match self.get_collection_info().await {
            Ok(_) => {
                // Collection exists, nothing to do
                return Ok(());
            }
            Err(_) => {
                // Collection doesn't exist, create it
                self.create_collection().await?;
            }
        }

        Ok(())
    }

    /// Create a new collection
    async fn create_collection(&self) -> Result<()> {
        let payload = serde_json::json!({
            "name": self.collection_name,
            "metadata": {
                "description": "Niodoo codebase embeddings",
                "created_by": "niodoo-embeddings-system"
            }
        });

        let url = format!("{}/api/v1/collections", self.base_url);
        
        let response = self.client
            .post(&url)
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Failed to create collection: {}", error_text));
        }

        Ok(())
    }
}

/// Dual-write manager for sidecar files and ChromaDB
pub struct DualWriteManager {
    engine: std::sync::Arc<EmbeddingEngine>,
    chroma_sync: ChromaSync,
    mcp_notifier: Option<std::sync::Arc<dyn super::McpNotifier + Send + Sync>>,
}

impl DualWriteManager {
    /// Create a new dual-write manager
    pub fn new(
        engine: std::sync::Arc<EmbeddingEngine>,
        chroma_sync: ChromaSync,
        mcp_notifier: Option<std::sync::Arc<dyn super::McpNotifier + Send + Sync>>,
    ) -> Self {
        Self {
            engine,
            chroma_sync,
            mcp_notifier,
        }
    }

    /// Write embedding to both sidecar file and ChromaDB
    pub async fn write_embedding(
        &self,
        path: &Path,
        embedding: Vec<f32>,
        document: String,
    ) -> Result<()> {
        // Write to sidecar file
        self.engine.save_embedding(path, embedding.clone()).await?;

        // Write to ChromaDB (async, don't fail on errors)
        let metadata = self.create_metadata(path)?;
        if let Err(e) = self.chroma_sync.upsert_embedding(
            &path.to_string_lossy(),
            embedding,
            metadata,
            document,
        ).await {
            error!("ChromaDB sync failed: {}", e);
            // Notify MCP server about sync failure
            if let Some(notifier) = &self.mcp_notifier {
                notifier.notify_error(&format!("ChromaDB sync failed: {}", e)).await.ok();
            }
        }

        Ok(())
    }

    /// Create metadata for embedding
    fn create_metadata(&self, path: &Path) -> Result<EmbeddingMetadata> {
        let file_hash = self.engine.calculate_file_hash(path)?;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        
        Ok(EmbeddingMetadata {
            file_path: path.to_string_lossy().to_string(),
            file_hash,
            timestamp,
            model: "all-MiniLM-L6-v2".to_string(),
            chunk_index: 0,
            total_chunks: 1,
            embedding_id: self.engine.generate_embedding_id(path),
            file_size: std::fs::metadata(path)?.len(),
        })
    }

    /// Initialize ChromaDB collection
    pub async fn initialize(&self) -> Result<()> {
        // Ensure ChromaDB is available
        if !self.chroma_sync.health_check().await? {
            return Err(anyhow::anyhow!("ChromaDB is not available"));
        }

        // Ensure collection exists
        self.chroma_sync.ensure_collection().await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs;

    #[tokio::test]
    async fn test_chroma_sync_creation() {
        let sync = ChromaSync::new(
            "http://localhost:8000".to_string(),
            "test_collection".to_string(),
            10,
        );

        assert_eq!(sync.base_url, "http://localhost:8000");
        assert_eq!(sync.collection_name, "test_collection");
        assert_eq!(sync.batch_size, 10);
    }

    #[tokio::test]
    async fn test_dual_write_manager() {
        let temp_dir = tempdir().unwrap();
        let engine = std::sync::Arc::new(EmbeddingEngine::new(temp_dir.path().to_path_buf()).unwrap());
        let chroma_sync = ChromaSync::new(
            "http://localhost:8000".to_string(),
            "test_collection".to_string(),
            10,
        );

        let manager = DualWriteManager::new(engine, chroma_sync, None);

        // Test file
        let test_file = temp_dir.path().join("test.rs");
        fs::write(&test_file, "fn main() { println!(\"Hello, world!\"); }").unwrap();

        // This will fail ChromaDB sync but should succeed sidecar write
        let embedding = vec![0.1; 384]; // Mock embedding
        let result = manager.write_embedding(&test_file, embedding, "test content".to_string()).await;
        
        // Should succeed despite ChromaDB failure
        assert!(result.is_ok());
    }
}

