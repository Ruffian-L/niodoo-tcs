use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;
use crate::experience::Experience;

/// Memory layer types in the 6-layer architecture
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryLayer {
    CoreBurned, // Layer 1: Core burned memories (highest stability)
    Procedural, // Layer 2: Procedural memories
    Episodic,   // Layer 3: Episodic memories
    Semantic,   // Layer 4: Semantic memories
    Somatic,    // Layer 5: Somatic memories
    Working,    // Layer 6: Working memory (lowest stability)
}

/// Memory entry with emotional and stability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub layer: MemoryLayer,
    pub emotional_weight: f64,
    pub stability: f64,
    pub timestamp: u64,
    pub access_count: u32,
    pub last_accessed: u64,
    pub emotional_vector: (f64, f64, f64), // RGB emotional encoding
    pub topology_position: (f64, f64, f64), // Position in K-Twist topology
}

impl Memory {
    pub fn new(content: String, layer: MemoryLayer) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: Uuid::new_v4().to_string(),
            content,
            layer: layer.clone(),
            emotional_weight: 0.5,
            stability: match layer {
                MemoryLayer::CoreBurned => 0.99,
                MemoryLayer::Procedural => 0.85,
                MemoryLayer::Episodic => 0.70,
                MemoryLayer::Semantic => 0.60,
                MemoryLayer::Somatic => 0.45,
                MemoryLayer::Working => 0.30,
            },
            timestamp,
            access_count: 0,
            last_accessed: timestamp,
            emotional_vector: (0.0, 0.0, 0.0),
            topology_position: (0.0, 0.0, 0.0),
        }
    }

    pub fn access(&mut self) {
        self.access_count += 1;
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
}

/// Memory system for storing and retrieving memories
pub struct MemorySystem {
    memories: HashMap<String, Memory>,
}

impl MemorySystem {
    pub fn new() -> Self {
        Self {
            memories: HashMap::new(),
        }
    }

    pub fn store(&mut self, memory: Memory) {
        self.memories.insert(memory.id.clone(), memory);
    }

    pub fn retrieve(&mut self, id: &str) -> Option<&mut Memory> {
        self.memories.get_mut(id).map(|m| {
            m.access();
            m
        })
    }

    pub fn search_by_layer(&self, layer: &MemoryLayer) -> Vec<&Memory> {
        self.memories
            .values()
            .filter(|m| &m.layer == layer)
            .collect()
    }
}

impl Default for MemorySystem {
    fn default() -> Self {
        Self::new()
    }
}

// ExperienceStore implementation (simplified HTTP REST version for lab, based on legacy EragClient patterns)

/// Experience store configuration
#[derive(Debug, Deserialize)]
pub struct ExperienceStoreConfig {
    pub qdrant_url: String,
    pub collection: String,
    pub vector_size: usize,
}

impl ExperienceStoreConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read experience store config from {}", path))?;
        let config: ExperienceStoreConfig = toml::from_str(&content)
            .with_context(|| format!("failed to parse experience store config from {}", path))?;
        Ok(config)
    }
}

/// Experience store backed by Qdrant
pub struct ExperienceStore {
    config: ExperienceStoreConfig,
    http_client: reqwest::Client,
}

impl ExperienceStore {
    /// Initialize experience store from config file
    pub async fn initialise(config_path: &str) -> Result<Self> {
        let config = ExperienceStoreConfig::from_file(config_path)?;
        let http_client = reqwest::Client::new();

        Ok(Self {
            config,
            http_client,
        })
    }

    /// Get vector size for this store
    pub fn vector_size(&self) -> usize {
        self.config.vector_size
    }

    /// Upsert experience with embedding into Qdrant
    pub async fn upsert(&self, experience: &Experience, embedding: &[f32]) -> Result<()> {
        let upsert_url = format!(
            "{}/collections/{}/points",
            self.config.qdrant_url,
            self.config.collection
        );

        // Convert f32 embedding to f64 for Qdrant
        let embedding_f64: Vec<f64> = embedding.iter().map(|&x| x as f64).collect();

        // Build payload
        let payload: Value = json!({
            "input": experience.input,
            "output": experience.output,
            "context": experience.context,
            "task_type": experience.task_type,
            "success_score": experience.success_score,
            "rouge_l": experience.rouge_l,
            "feedback": experience.feedback,
            "reward": experience.reward,
            "metadata": experience.metadata,
            "timestamp": experience.timestamp.to_rfc3339(),
        });

        let point = json!({
            "id": experience.id.to_string(),
            "vector": embedding_f64,
            "payload": payload,
        });

        let request_body = json!({
            "points": [point]
        });

        let response = self
            .http_client
            .put(&upsert_url)
            .json(&request_body)
            .send()
            .await
            .with_context(|| format!("failed to upsert experience to Qdrant at {}", upsert_url))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Qdrant upsert failed with status {}: {}", status, text);
        }

        Ok(())
    }
}

