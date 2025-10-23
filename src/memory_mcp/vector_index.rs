// Copyright (c) 2025 Jason Van Pham (ruffian-l on GitHub) @ The Niodoo Collaborative
// Licensed under the MIT License - See LICENSE file for details
// Attribution required for all derivative works

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use usearch::Index;
use uuid::Uuid;

/// Represents a memory with its unique identifier and embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEmbedding {
    pub id: Uuid,
    pub embedding: Vec<f32>,
    pub content: String,
}

/// VectorIndex manages semantic search for memories using usearch
///
/// Thread-safe memory indexing for async operations
pub struct VectorIndex {
    index: Arc<Index>,
    id_to_memory: HashMap<u64, MemoryEmbedding>,
}

// SAFETY: VectorIndex is Send + Sync because:
// - Arc<Index> is Send + Sync (Index is thread-safe)
// - HashMap<u64, MemoryEmbedding> is Send + Sync (all contained types are Send + Sync)
unsafe impl Send for VectorIndex {}
unsafe impl Sync for VectorIndex {}

impl VectorIndex {
    /// Creates a new VectorIndex with a specified dimensionality
    pub fn new(dim: usize) -> Result<Self> {
        // USearch 2.x API: create IndexOptions with Default and override fields
        let mut options = usearch::IndexOptions::default();
        options.dimensions = dim;
        options.metric = usearch::MetricKind::Cos;
        options.quantization = usearch::ScalarKind::F32;
        options.connectivity = 16;
        options.expansion_add = 128;
        options.expansion_search = 64;
        options.multi = false;

        let index = Index::new(&options)
            .context("Failed to create usearch index")?;

        Ok(Self {
            index: Arc::new(index),
            id_to_memory: HashMap::new(),
        })
    }

    /// Adds a memory embedding to the index
    pub fn add_memory(&mut self, memory: MemoryEmbedding) -> Result<()> {
        let embedding_len = memory.embedding.len();

        // Ensure the embedding matches the index dimensions
        if embedding_len != self.index.dimensions() {
            return Err(anyhow::anyhow!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.index.dimensions(),
                embedding_len
            ));
        }

        // Get a unique internal ID
        let internal_id = self.id_to_memory.len() as u64;

        // Add to usearch index (v2.x API)
        self.index.add(internal_id, &memory.embedding)
            .map_err(|e| anyhow::anyhow!("Failed to add embedding: {:?}", e))?;

        // Store mapping for retrieval
        self.id_to_memory.insert(internal_id, memory);

        Ok(())
    }

    /// Performs k-NN semantic search
    pub fn search(&self, query_embedding: &[f32], k: usize) -> Result<Vec<MemoryEmbedding>> {
        // Validate query embedding matches index dimensions
        if query_embedding.len() != self.index.dimensions() {
            return Err(anyhow::anyhow!(
                "Query embedding dimension mismatch: expected {}, got {}",
                self.index.dimensions(),
                query_embedding.len()
            ));
        }

        // Perform k-NN search - usearch 2.x API: search(query, count)
        let results = self.index
            .search(query_embedding, k)
            .context("Failed to perform semantic search")?;

        // Map internal IDs to memory embeddings
        // Matches struct has public keys and distances fields
        let matched_memories = results.keys.iter()
            .filter_map(|key| {
                self.id_to_memory.get(key).cloned()
            })
            .collect();

        Ok(matched_memories)
    }

    /// Get total number of memories in the index
    pub fn len(&self) -> usize {
        self.id_to_memory.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.id_to_memory.is_empty()
    }
}

/// Extension trait for additional search capabilities
pub trait SemanticSearchExt {
    fn semantic_similarity(&self, other: &[f32]) -> f32;
}

impl SemanticSearchExt for Vec<f32> {
    /// Calculate cosine similarity between two embeddings
    fn semantic_similarity(&self, other: &[f32]) -> f32 {
        self.as_slice().semantic_similarity(other)
    }
}

impl SemanticSearchExt for [f32] {
    /// Calculate cosine similarity between two embeddings
    /// Returns a value between -1.0 and 1.0, where:
    /// - 1.0 means identical vectors
    /// - 0.0 means orthogonal vectors
    /// - -1.0 means opposite vectors
    fn semantic_similarity(&self, other: &[f32]) -> f32 {
        if self.len() != other.len() {
            return 0.0;
        }

        let dot_product: f32 = self.iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let norm_b: f32 = other.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_vector_index_basic_operations() -> Result<()> {
        // Create an index with 128-dimensional embeddings
        let mut index = VectorIndex::new(128)?;

        // Generate some test memories
        let mut rng = rand::thread_rng();
        let memories: Vec<MemoryEmbedding> = (0..10)
            .map(|_| MemoryEmbedding {
                id: Uuid::new_v4(),
                embedding: (0..128).map(|_| rng.random()).collect(),
                content: "Test memory".to_string(),
            })
            .collect();

        // Add memories to index
        for memory in memories.clone() {
            index.add_memory(memory)?;
        }

        assert_eq!(index.len(), 10);

        // Test search with the first memory's embedding
        let first_memory_embedding = memories[0].embedding.clone();
        let search_results = index.search(&first_memory_embedding, 3)?;

        assert!(!search_results.is_empty());

        Ok(())
    }

    #[test]
    fn test_semantic_similarity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let c = vec![0.0, 0.0, 0.0];

        assert_eq!(a.semantic_similarity(&b), 1.0);
        assert_eq!(a.semantic_similarity(&c), 0.0);
    }
}
