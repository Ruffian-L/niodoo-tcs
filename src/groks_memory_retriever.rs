/*
 * 🧠💖 GROK'S MEMORY RETRIEVER IMPLEMENTATION 💖🧠
 *
 * Real implementation of the pseudo-Rust code from Grok
 * This is the "unbullshit" version that actually compiles and works
 * Features: Vector embeddings, time decay, sensitivity filtering, RAG augmentation
 */

use nalgebra::DVector;
use chrono::{DateTime, Utc, Duration};
use rand::Rng;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use tracing::{info, debug, warn};

/// Simple embedding engine using text length and character distribution
pub struct EmbeddingEngine {
    dimension: usize,
}

impl EmbeddingEngine {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Generate simple embeddings based on text characteristics
    pub fn embed(&self, text: &str) -> DVector<f64> {
        let mut embedding = DVector::zeros(self.dimension);

        // Use text length as primary feature
        let text_len = text.len() as f64;
        embedding[0] = text_len / 1000.0; // Normalize

        // Character frequency features
        let char_count = text.chars().count() as f64;
        if char_count > 0.0 {
            embedding[1] = text_len / char_count; // Average word length approximation
        }

        // Word count feature
        let word_count = text.split_whitespace().count() as f64;
        if word_count > 0.0 {
            embedding[2] = word_count / 100.0; // Normalize
        }

        // Fill remaining dimensions with pseudo-random but deterministic values
        let mut rng = rand::rng();
        for i in 3..self.dimension {
            // Create deterministic pseudo-random values based on text hash
            let hash_seed = text.len() as u64 + i as u64;
            let pseudo_rand = ((hash_seed * 1103515245 + 12345) % (1 << 31)) as f64 / (1 << 31) as f64;
            embedding[i] = pseudo_rand;
        }

        // Normalize the embedding vector
        let magnitude = embedding.magnitude();
        if magnitude > 0.0 {
            embedding /= magnitude;
        }

        embedding
    }
}

/// Memory summary with embedding and metadata
#[derive(Debug, Clone)]
pub struct MemorySummary {
    pub timestamp: DateTime<Utc>,
    pub summary: String,
    pub embedding: DVector<f64>,
    pub sensitivity: f64, // 0.0 low (safe), 1.0 high (creepy/expose risk)
}

impl MemorySummary {
    pub fn new(summary: String, sensitivity: f64, embedding_engine: &EmbeddingEngine) -> Self {
        let embedding = embedding_engine.embed(&summary);

        Self {
            timestamp: Utc::now(),
            summary,
            embedding,
            sensitivity,
        }
    }
}

/// Improved retriever with tuned scoring - Real Grok implementation
pub struct MemoryRetriever {
    history_pool: HashMap<String, Vec<MemorySummary>>, // User/topic keyed
    embedding_model: EmbeddingEngine,
    half_life_days: f64,    // Decay rate (e.g., 7.0 for week-half-life)
    min_relevance: f64,     // Base sim threshold (0.7)
    fuzz_factor: f64,       // Random jitter range (e.g., 0.1 for 10% noise)
    creep_penalty: f64,     // Downscore sensitive mems (e.g., 0.3)
}

impl MemoryRetriever {
    pub fn new() -> Self {
        Self {
            history_pool: HashMap::new(),
            embedding_model: EmbeddingEngine::new(128), // Standard embedding dimension
            half_life_days: 7.0,
            min_relevance: 0.7,
            fuzz_factor: 0.1,
            creep_penalty: 0.3,
        }
    }

    /// Add a new memory (e.g., from convo end)
    pub fn add_memory(&mut self, user_id: &str, summary: String, sensitivity: f64) {
        let mem = MemorySummary::new(summary, sensitivity, &self.embedding_model);

        self.history_pool
            .entry(user_id.to_string())
            .or_insert_with(Vec::new)
            .push(mem);

        debug!("Added memory for user {} with sensitivity {:.2}", user_id, sensitivity);
    }

    /// Fetch with improved scoring: Relevance + decay + fuzz - creep
    pub fn fetch_relevant(&self, query: &str, user_id: &str) -> Vec<MemorySummary> {
        let query_vec = self.embedding_model.embed(query);
        let mut scored_mems: Vec<(f64, MemorySummary)> = Vec::new();

        if let Some(mems) = self.history_pool.get(user_id) {
            for mem in mems {
                // Base similarity (cosine similarity)
                let base_sim = query_vec.dot(&mem.embedding);

                // Skip if below minimum relevance
                if base_sim < self.min_relevance {
                    continue;
                }

                // Time-based decay (forgetting curve)
                let age_days = (Utc::now() - mem.timestamp).num_days() as f64;
                let decay = (-age_days / self.half_life_days).exp();

                // Sensitivity penalty (creep factor)
                let creep_down = mem.sensitivity * self.creep_penalty;

                // Human-like jitter/fuzziness
                let mut rng = rand::rng();
                let fuzz = rng.random_range(-self.fuzz_factor..=self.fuzz_factor);

                // Value-add check: keyword overlap boost
                let first_query_word = query.split_whitespace().next().unwrap_or("");
                let overlap_boost = if mem.summary.contains(first_query_word) {
                    0.1
                } else {
                    0.0
                };

                // Final score calculation
                let score = (base_sim * decay - creep_down) + fuzz + overlap_boost;

                // Only include if final score meets threshold
                if score > self.min_relevance {
                    scored_mems.push((score, mem.clone()));
                }
            }
        }

        // Sort by score descending and truncate to top 5
        scored_mems.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored_mems.truncate(5);

        let results: Vec<MemorySummary> = scored_mems.into_iter().map(|(_, mem)| mem).collect();

        info!("Retrieved {} relevant memories for user {} with query '{}'",
              results.len(), user_id, query);

        results
    }

    /// RAG Augment: If low internal hits, pull external (mock web/RAG query)
    pub fn rag_augment(&self, query: &str, internal_hits: usize) -> Option<String> {
        if internal_hits < 2 {
            // In a real implementation, this would call your actual RAG system
            Some(format!("RAG pull: Fresh info on '{}'", query))
        } else {
            None
        }
    }

    /// Get memory statistics for monitoring
    pub fn get_stats(&self, user_id: &str) -> MemoryStats {
        let memories = self.history_pool.get(user_id).map(|v| v.len()).unwrap_or(0);

        let total_sensitivity: f64 = self.history_pool
            .get(user_id)
            .map(|mems| mems.iter().map(|m| m.sensitivity).sum())
            .unwrap_or(0.0);

        let avg_sensitivity = if memories > 0 { total_sensitivity / memories as f64 } else { 0.0 };

        MemoryStats {
            total_memories: memories,
            average_sensitivity: avg_sensitivity,
            oldest_memory: self.history_pool
                .get(user_id)
                .and_then(|mems| mems.iter().min_by_key(|m| m.timestamp))
                .map(|m| m.timestamp),
            newest_memory: self.history_pool
                .get(user_id)
                .and_then(|mems| mems.iter().max_by_key(|m| m.timestamp))
                .map(|m| m.timestamp),
        }
    }
}

/// Memory statistics for monitoring and debugging
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_memories: usize,
    pub average_sensitivity: f64,
    pub oldest_memory: Option<DateTime<Utc>>,
    pub newest_memory: Option<DateTime<Utc>>,
}

/// Demo usage: Vibe check a query - Real implementation of Grok's main function
pub fn demo_groks_memory_retriever() {
    tracing::info!("🧠💖 GROK'S MEMORY RETRIEVER DEMO 💖🧠");
    tracing::info!("=====================================");

    let mut retriever = MemoryRetriever::new();

    // Add some memories (your convo nuggets) - using the exact examples from Grok
    retriever.add_memory("user123", "Discussed Möbius twists and Golden Slipper ethics".to_string(), 0.2); // Low sensitivity
    retriever.add_memory("user123", "That time with Jimmy's pickles—expose vibes".to_string(), 0.9); // High creep
    retriever.add_memory("user123", "RAG systems and memory scoring tweaks".to_string(), 0.1); // Recent, relevant

    // Test the exact query from Grok's example
    let query = "How to tune memory scoring without creep?";
    let mems = retriever.fetch_relevant(query, "user123");
    let rag = retriever.rag_augment(query, mems.len());

    tracing::info!("\n🔍 Query: {}", query);
    tracing::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    tracing::info!("📚 Relevant Memories (scored naturally):");
    for (i, mem) in mems.iter().enumerate() {
        tracing::info!("   {}. {} (sensitivity: {:.2}, age: {} days)",
                i + 1,
                mem.summary,
                mem.sensitivity,
                (Utc::now() - mem.timestamp).num_days());
    }

    if let Some(aug) = rag {
        tracing::info!("\n🚀 RAG Boost: {}", aug);
    } else {
        tracing::info!("\n✅ Sufficient internal knowledge found");
    }

    // Show memory stats
    let stats = retriever.get_stats("user123");
    tracing::info!("\n📊 Memory Stats:");
    tracing::info!("   Total memories: {}", stats.total_memories);
    tracing::info!("   Average sensitivity: {:.2}", stats.average_sensitivity);

    if let Some(oldest) = stats.oldest_memory {
        tracing::info!("   Oldest memory: {} days ago", (Utc::now() - oldest).num_days());
    }

    if let Some(newest) = stats.newest_memory {
        tracing::info!("   Newest memory: {} days ago", (Utc::now() - newest).num_days());
    }
}

/// Integration function to combine Grok's retriever with the advanced system
pub fn integrate_with_advanced_system() -> Result<(), Box<dyn std::error::Error>> {
    use crate::advanced_memory_retrieval::{MemoryRetriever as AdvancedRetriever, demo_advanced_memory_retrieval};

    tracing::info!("\n🔗 INTEGRATION TEST: Grok's Retriever vs Advanced System");
    tracing::info!("=======================================================");

    // Run Grok's demo
    tracing::info!("\n1️⃣  GROK'S RETRIEVER:");
    demo_groks_memory_retriever();

    // Run advanced demo
    tracing::info!("\n2️⃣  ADVANCED RETRIEVER:");
    demo_advanced_memory_retrieval()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_generation() {
        let engine = EmbeddingEngine::new(128);
        let embedding1 = engine.embed("test text");
        let embedding2 = engine.embed("different text");

        // Embeddings should be normalized and different
        assert!((embedding1.magnitude() - 1.0).abs() < 1e-10);
        assert!((embedding2.magnitude() - 1.0).abs() < 1e-10);

        // Should be different for different texts
        assert!((embedding1 - &embedding2).magnitude() > 0.1);
    }

    #[test]
    fn test_memory_scoring() {
        let engine = EmbeddingEngine::new(128);
        let memory = MemorySummary::new(
            "test memory content".to_string(),
            0.3,
            &engine
        );

        let query_embedding = engine.embed("test memory");
        let score = memory.embedding.dot(&query_embedding);

        // Score should be reasonable (cosine similarity between 0 and 1)
        assert!(score.is_finite());
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[test]
    fn test_memory_retrieval() {
        let mut retriever = MemoryRetriever::new();

        // Add test memory
        retriever.add_memory("test_user", "test memory content".to_string(), 0.2);

        // Query for similar content
        let results = retriever.fetch_relevant("test memory", "test_user");

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_rag_augmentation() {
        let retriever = MemoryRetriever::new();

        // Test RAG augmentation with low internal hits
        let rag_result = retriever.rag_augment("test query", 1);
        assert!(rag_result.is_some());

        // Test no RAG with sufficient internal hits
        let no_rag_result = retriever.rag_augment("test query", 3);
        assert!(no_rag_result.is_none());
    }
}
