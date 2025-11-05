use anyhow::Result;
use chrono::{DateTime, Utc};
use qdrant_client::client::{QdrantClient, Payload};
use qdrant_client::qdrant::{
    CreateCollection, VectorsConfig, VectorParams, Distance,
    PointStruct, vectors::VectorsOptions, SearchPoints,
    Value, vectors_config,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use uuid::Uuid;

/// Represents a single experience from the Executor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub id: Uuid,
    pub input: String,
    pub output: String,
    pub context: String,
    pub task_type: String,
    pub success_score: f32,
    pub timestamp: DateTime<Utc>,
    pub embedding: Option<Vec<f32>>,
    pub relevance_score: f32,  // Added for optimization
}

impl Experience {
    pub fn new(
        input: String,
        output: String,
        context: String,
        task_type: String,
        success_score: f32,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            input,
            output,
            context,
            task_type,
            success_score,
            timestamp: Utc::now(),
            embedding: None, // Will be set by embedder
            relevance_score: 0.0, // Will be calculated during retrieval
        }
    }

    /// Normalize embedding to unit hypersphere
    pub fn normalize_embedding(&mut self) {
        if let Some(ref mut embedding) = self.embedding {
            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in embedding.iter_mut() {
                    *val /= norm;
                }
            }
        }
    }
}

/// Configuration for the memory core
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub qdrant_url: String,
    pub collection_name: String,
    pub vector_dim: usize,
    pub max_memory_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            qdrant_url: std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string()),
            collection_name: "experiences".to_string(),
            vector_dim: 768, // Qwen 0.5B embedding dimension
            max_memory_size: 100000,
        }
    }
}

/// Core memory management system
pub struct MemoryCore {
    client: QdrantClient,
    config: MemoryConfig,
}

impl MemoryCore {
    /// Create a new memory core instance
    pub async fn new(config: MemoryConfig) -> Result<Self> {
        // Connect to Qdrant running on beelink
        let client = QdrantClient::new(Some(qdrant_client::client::QdrantClientConfig::from_url(&config.qdrant_url))).await?;

        // Ensure collection exists
        Self::ensure_collection(&client, &config).await?;

        Ok(Self { client, config })
    }

    /// Ensure the vector collection exists
    /// Ensure the vector collection exists
    async fn ensure_collection(client: &QdrantClient, config: &MemoryConfig) -> Result<()> {
        let collections = client.list_collections().await?;
        let collection_exists = collections
            .collections
            .iter()
            .any(|c| c.name == config.collection_name);

        if !collection_exists {
            // Create collection with cosine similarity
            let create_collection = CreateCollection {
                collection_name: config.collection_name.clone(),
                vectors_config: Some(VectorsConfig {
                    config: Some(vectors_config::Config::Params(VectorParams {
                        size: config.vector_dim as u64,
                        distance: Distance::Cosine.into(),
                        ..Default::default()
                    })),
                }),
                ..Default::default()
            };
            
            client
                .create_collection(&create_collection)
                .await?;
        }

        Ok(())
    }

    /// Store an experience in the vector database
    pub async fn store_experience(&self, experience: &Experience) -> Result<()> {
        let mut exp = experience.clone();
        exp.normalize_embedding();

        // Ensure we have an embedding
        let embedding = exp.embedding
            .ok_or_else(|| anyhow::anyhow!("Experience must have an embedding to be stored"))?;

        let mut payload = Payload::new();
        payload.insert("id", exp.id.to_string());
        payload.insert("input", exp.input.clone());
        payload.insert("output", exp.output.clone());
        payload.insert("context", exp.context.clone());
        payload.insert("task_type", exp.task_type.clone());
        payload.insert("success_score", exp.success_score as f64);
        payload.insert("timestamp", exp.timestamp.to_rfc3339());
        payload.insert("relevance_score", exp.relevance_score as f64);

        let points = vec![PointStruct::new(
            exp.id.to_string(),
            embedding,
            payload,
        )];

        self.client
            .upsert_points(&self.config.collection_name, points)
            .await?;

        Ok(())
    }

    /// Search for similar experiences using vector similarity
    pub async fn search_similar(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<Experience>> {
        let search_points = SearchPoints {
            collection_name: self.config.collection_name.clone(),
            vector: query_embedding.to_vec(),
            filter: None,
            limit: limit as u64,
            with_vectors: Some(true.into()),
            with_payload: Some(true.into()),
            ..Default::default()
        };
        
        let search_result = self.client
            .search_points(&search_points)
            .await?;

        let mut experiences = Vec::new();
        for point in search_result.result {
            let payload = point.payload;
            
            // Try to parse ID from payload, fallback to new UUID
            let id = payload.get("id")
                .and_then(|v| Self::value_as_str(v))
                .as_deref()
                .and_then(|s| Uuid::parse_str(s).ok())
                .unwrap_or(Uuid::new_v4());
            
            let embedding = match &point.vectors {
                Some(vectors) => {
                    // Try to extract vector data in various formats
                    if let Some(v) = vectors.vectors_options.as_ref() {
                        match v {
                            VectorsOptions::Vector(vec_data) => Some(vec_data.data.clone()),
                            _ => None,
                        }
                    } else {
                        None
                    }
                },
                None => None,
            };
            
            let exp = Experience {
                id,
                input: payload.get("input").and_then(|v| Self::value_as_str(v)).unwrap_or_default(),
                output: payload.get("output").and_then(|v| Self::value_as_str(v)).unwrap_or_default(),
                context: payload.get("context").and_then(|v| Self::value_as_str(v)).unwrap_or_default(),
                task_type: payload.get("task_type").and_then(|v| Self::value_as_str(v)).unwrap_or_default(),
                embedding,
                success_score: payload.get("success_score").and_then(|v| Self::value_as_f64(v)).unwrap_or(0.0) as f32,
                timestamp: payload.get("timestamp")
                    .and_then(|v| Self::value_as_str(v))
                    .as_deref()
                    .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or(Utc::now()),
                relevance_score: point.score,  // Use Qdrant's similarity score
            };
            experiences.push(exp);
        }

        Ok(experiences)
    }

    /// Compact the collection to optimize storage
    pub async fn compact_collection(&self) -> Result<()> {
        // Note: qdrant-client doesn't have a direct compact method in the public API
        // This would typically be handled by Qdrant's internal optimization
        // For now, we'll just log that compaction is requested
        tracing::info!("Collection compaction requested for {}", self.config.collection_name);
        Ok(())
    }

    /// Helper to extract string from qdrant Value
    fn value_as_str(v: &Value) -> Option<String> {
        v.kind.as_ref().and_then(|k| match k {
            qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.clone()),
            _ => None,
        })
    }

    /// Helper to extract f64 from qdrant Value
    fn value_as_f64(v: &Value) -> Option<f64> {
        v.kind.as_ref().and_then(|k| match k {
            qdrant_client::qdrant::value::Kind::DoubleValue(f) => Some(*f),
            qdrant_client::qdrant::value::Kind::IntegerValue(i) => Some(*i as f64),
            _ => None,
        })
    }

    /// Convert Qdrant payload back to Experience
    fn payload_to_experience(
        payload: std::collections::HashMap<String, JsonValue>,
        vectors: Vec<f32>,
    ) -> Result<Experience> {
        let id = payload
            .get("id")
            .and_then(|v| match v {
                JsonValue::String(s) => Uuid::parse_str(s).ok(),
                _ => None,
            })
            .unwrap_or_else(Uuid::new_v4);

        let input = payload
            .get("input")
            .and_then(|v| match v {
                JsonValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "".to_string());

        let output = payload
            .get("output")
            .and_then(|v| match v {
                JsonValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "".to_string());

        let context = payload
            .get("context")
            .and_then(|v| match v {
                JsonValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "".to_string());

        let task_type = payload
            .get("task_type")
            .and_then(|v| match v {
                JsonValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "".to_string());

        let success_score = payload
            .get("success_score")
            .and_then(|v| match v {
                JsonValue::Number(n) => n.as_f64().map(|d| d as f32),
                _ => None,
            })
            .unwrap_or(0.0);

        let timestamp = payload
            .get("timestamp")
            .and_then(|v| match v {
                JsonValue::String(s) => DateTime::parse_from_rfc3339(s).ok(),
                _ => None,
            })
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        // Use provided vector
        let embedding = Some(vectors);

        Ok(Experience {
            id,
            input,
            output,
            context,
            task_type,
            success_score,
            timestamp,
            embedding,
            relevance_score: 0.0, // Will be set during retrieval
        })
    }

    /// Get memory statistics
    pub async fn get_stats(&self) -> Result<HashMap<String, u64>> {
        let info = self.client.collection_info(&self.config.collection_name).await?;
        let mut stats = HashMap::new();
        
        // Access the result field which contains collection info
        if let Some(collection) = info.result {
            stats.insert("vectors_count".to_string(), collection.vectors_count);
            stats.insert("total_points".to_string(), collection.points_count);
        }
        
        Ok(stats)
    }

    /// Perform memory compaction (remove old/low-quality experiences)
    pub async fn compact_memory(&self, _keep_ratio: f32) -> Result<()> {
        // TODO: Implement compaction using qdrant-client scroll and delete
        // For now, just log
        tracing::info!("Memory compaction requested but not yet implemented with qdrant-client");
        Ok(())
    }

    /// Calculate recency factor (exponential decay)
    fn recency_factor(timestamp: DateTime<Utc>, now: DateTime<Utc>) -> f32 {
        let hours_old = (now - timestamp).num_hours() as f32;
        (-hours_old / 24.0).exp() // Half-life of 1 day
    }
}