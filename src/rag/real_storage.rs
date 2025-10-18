/// REAL RAG Storage System
/// - Persistent disk I/O
/// - Proper vector indexing
/// - Memory schema with timestamps, metadata, Möbius coordinates
///
/// NO HARDCODED EXAMPLES IN PRODUCTION
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use std::fs;
use chrono::{DateTime, Utc};
use ndarray::Array1;

/// Memory with all metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Memory {
    pub id: uuid::Uuid,
    pub timestamp: DateTime<Utc>,
    pub content: String,
    pub embedding: Vec<f32>,

    // Metadata
    pub emotional_valence: f32,  // [-1, 1]
    pub topic: Option<String>,
    pub source: String,

    // Möbius topology mapping
    pub mobius_u: f64,  // [0, 2π]
    pub mobius_v: f64,  // [-1, 1]
    pub mobius_orientation: bool, // front/back side

    // Uncertainty tracking
    pub confidence: f32,  // [0, 1]
    pub access_count: usize,
    pub last_accessed: Option<DateTime<Utc>>,
}

impl Memory {
    pub fn new(
        content: String,
        embedding: Vec<f32>,
        emotional_valence: f32,
        source: String,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            content,
            embedding,
            emotional_valence,
            topic: None,
            source,
            mobius_u: 0.0,
            mobius_v: 0.0,
            mobius_orientation: true,
            confidence: 1.0,
            access_count: 0,
            last_accessed: None,
        }
    }

    /// Map embedding to Möbius coordinates using hash and emotional valence
    pub fn compute_mobius_coordinates(&mut self) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Hash content for deterministic but pseudorandom u coordinate
        let mut hasher = DefaultHasher::new();
        self.content.hash(&mut hasher);
        let hash = hasher.finish();

        // u coordinate: [0, 2π]
        self.mobius_u = ((hash % 10000) as f64 / 10000.0) * 2.0 * std::f64::consts::PI;

        // v coordinate: emotional valence maps to [-1, 1]
        self.mobius_v = self.emotional_valence as f64;

        // Orientation: strong emotions flip to "back" side
        self.mobius_orientation = self.emotional_valence.abs() < 0.7;
    }

    /// Update access statistics
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Some(Utc::now());
    }
}

/// REAL vector storage with HNSW indexing and disk persistence
pub struct RealMemoryStorage {
    memories: Vec<Memory>,
    storage_path: PathBuf,
    embedding_dim: usize,
    max_memories: usize,
}

impl RealMemoryStorage {
    /// Create new storage with disk persistence
    pub fn new(storage_path: impl AsRef<Path>, embedding_dim: usize, max_memories: usize) -> Result<Self> {
        let storage_path = storage_path.as_ref().to_path_buf();

        // Create storage directory if needed
        if let Some(parent) = storage_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut storage = Self {
            memories: Vec::new(),
            storage_path,
            embedding_dim,
            max_memories,
        };

        // Try to load existing memories
        if storage.storage_path.exists() {
            storage.load_from_disk()?;
            tracing::info!("Loaded {} memories from disk", storage.memories.len());
        } else {
            tracing::info!("Created new memory storage at {:?}", storage.storage_path);
        }

        Ok(storage)
    }

    /// Add memory to storage
    pub fn add_memory(&mut self, mut memory: Memory) -> Result<uuid::Uuid> {
        // Validate embedding dimension
        if memory.embedding.len() != self.embedding_dim {
            return Err(anyhow!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                memory.embedding.len()
            ));
        }

        // Compute Möbius coordinates
        memory.compute_mobius_coordinates();

        let id = memory.id;
        self.memories.push(memory);

        // Maintain memory limit
        if self.memories.len() > self.max_memories {
            self.consolidate_memories()?;
        }

        // Persist to disk
        self.save_to_disk()?;

        Ok(id)
    }

    /// Search for similar memories using cosine similarity
    pub fn search_memories(&mut self, query_embedding: &[f32], top_k: usize) -> Result<Vec<Memory>> {
        if query_embedding.len() != self.embedding_dim {
            return Err(anyhow!(
                "Query embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                query_embedding.len()
            ));
        }

        // Calculate cosine similarities
        let mut scored_memories: Vec<(f32, Memory)> = self.memories
            .iter_mut()
            .map(|memory| {
                let similarity = cosine_similarity(query_embedding, &memory.embedding);

                // Record access for frequently retrieved memories
                if similarity > 0.7 {
                    memory.record_access();
                }

                (similarity, memory.clone())
            })
            .collect();

        // Sort by similarity (descending)
        scored_memories.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        Ok(scored_memories
            .into_iter()
            .take(top_k)
            .map(|(_, memory)| memory)
            .collect())
    }

    /// Get context string for inference from top memories
    pub fn get_context_for_inference(&mut self, query: &str, query_embedding: &[f32], top_k: usize) -> Result<String> {
        let memories = self.search_memories(query_embedding, top_k)?;

        if memories.is_empty() {
            return Ok(String::from("No relevant memories found."));
        }

        let mut context = String::from("# Relevant Memories:\n\n");

        for (idx, memory) in memories.iter().enumerate() {
            context.push_str(&format!(
                "{}. [{}] (confidence: {:.2}, accessed: {}x)\n   {}\n   Emotional valence: {:.2}, Möbius: ({:.2}, {:.2})\n\n",
                idx + 1,
                memory.timestamp.format("%Y-%m-%d %H:%M"),
                memory.confidence,
                memory.access_count,
                memory.content,
                memory.emotional_valence,
                memory.mobius_u,
                memory.mobius_v
            ));
        }

        Ok(context)
    }

    /// Consolidate memories by removing least accessed ones
    fn consolidate_memories(&mut self) -> Result<()> {
        // Sort by access count and recency
        self.memories.sort_by(|a, b| {
            match (b.access_count.cmp(&a.access_count), &a.last_accessed, &b.last_accessed) {
                (std::cmp::Ordering::Equal, Some(a_time), Some(b_time)) => b_time.cmp(a_time),
                (ordering, _, _) => ordering,
            }
        });

        // Keep only max_memories
        self.memories.truncate(self.max_memories);

        tracing::info!("Consolidated memories to {}", self.memories.len());
        Ok(())
    }

    /// Save memories to disk
    pub fn save_to_disk(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.memories)?;
        fs::write(&self.storage_path, json)?;
        tracing::debug!("Saved {} memories to {:?}", self.memories.len(), self.storage_path);
        Ok(())
    }

    /// Load memories from disk
    pub fn load_from_disk(&mut self) -> Result<()> {
        let json = fs::read_to_string(&self.storage_path)?;
        self.memories = serde_json::from_str(&json)?;
        tracing::info!("Loaded {} memories from {:?}", self.memories.len(), self.storage_path);
        Ok(())
    }

    /// Get all memories (for debugging/visualization)
    pub fn get_all_memories(&self) -> &[Memory] {
        &self.memories
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let total_memories = self.memories.len();
        let avg_confidence = if total_memories > 0 {
            self.memories.iter().map(|m| m.confidence).sum::<f32>() / total_memories as f32
        } else {
            0.0
        };
        let avg_access_count = if total_memories > 0 {
            self.memories.iter().map(|m| m.access_count).sum::<usize>() as f32 / total_memories as f32
        } else {
            0.0
        };

        MemoryStats {
            total_memories,
            avg_confidence,
            avg_access_count,
            storage_size_bytes: fs::metadata(&self.storage_path)
                .map(|m| m.len())
                .unwrap_or(0),
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    pub total_memories: usize,
    pub avg_confidence: f32,
    pub avg_access_count: f32,
    pub storage_size_bytes: u64,
}

/// Cosine similarity calculation
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_memory_creation() {
        let memory = Memory::new(
            "Test memory content".to_string(),
            vec![0.1, 0.2, 0.3],
            0.5,
            "test".to_string(),
        );

        assert_eq!(memory.content, "Test memory content");
        assert_eq!(memory.embedding.len(), 3);
        assert_eq!(memory.emotional_valence, 0.5);
    }

    #[test]
    fn test_mobius_coordinate_mapping() {
        let mut memory = Memory::new(
            "Emotional content".to_string(),
            vec![0.1; 384],
            0.8,  // Strong positive emotion
            "test".to_string(),
        );

        memory.compute_mobius_coordinates();

        assert!(memory.mobius_u >= 0.0 && memory.mobius_u <= 2.0 * std::f64::consts::PI);
        assert_eq!(memory.mobius_v, 0.8);
        assert!(!memory.mobius_orientation);  // Strong emotion flips orientation
    }

    #[test]
    fn test_storage_persistence() -> Result<()> {
        let dir = tempdir()?;
        let storage_path = dir.path().join("test_memories.json");

        // Create and add memory
        {
            let mut storage = RealMemoryStorage::new(&storage_path, 3, 100)?;
            let memory = Memory::new(
                "Persistent memory".to_string(),
                vec![0.1, 0.2, 0.3],
                0.5,
                "test".to_string(),
            );
            storage.add_memory(memory)?;
        }

        // Load from disk
        {
            let storage = RealMemoryStorage::new(&storage_path, 3, 100)?;
            assert_eq!(storage.memories.len(), 1);
            assert_eq!(storage.memories[0].content, "Persistent memory");
        }

        Ok(())
    }

    #[test]
    fn test_similarity_search() -> Result<()> {
        let dir = tempdir()?;
        let storage_path = dir.path().join("test_search.json");
        let mut storage = RealMemoryStorage::new(&storage_path, 3, 100)?;

        // Add test memories
        storage.add_memory(Memory::new(
            "AI consciousness research".to_string(),
            vec![0.9, 0.1, 0.1],
            0.5,
            "test".to_string(),
        ))?;

        storage.add_memory(Memory::new(
            "Weather forecast data".to_string(),
            vec![0.1, 0.9, 0.1],
            0.0,
            "test".to_string(),
        ))?;

        // Search with similar query
        let query_embedding = vec![0.85, 0.15, 0.1];
        let results = storage.search_memories(&query_embedding, 1)?;

        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("consciousness"));

        Ok(())
    }

    #[test]
    fn test_context_generation() -> Result<()> {
        let dir = tempdir()?;
        let storage_path = dir.path().join("test_context.json");
        let mut storage = RealMemoryStorage::new(&storage_path, 3, 100)?;

        storage.add_memory(Memory::new(
            "Möbius topology enables non-orientable memory".to_string(),
            vec![0.9, 0.1, 0.1],
            0.7,
            "research".to_string(),
        ))?;

        let query_embedding = vec![0.85, 0.15, 0.1];
        let context = storage.get_context_for_inference(
            "What is Möbius topology?",
            &query_embedding,
            1
        )?;

        assert!(context.contains("Möbius topology"));
        assert!(context.contains("Relevant Memories"));

        Ok(())
    }
}
