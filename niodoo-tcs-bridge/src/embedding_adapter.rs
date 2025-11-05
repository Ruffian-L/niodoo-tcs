//! Embedding Adapter
//!
//! Maps 1536-dim Qwen embeddings to 5D emotional space (PAD: Pleasure/Arousal/Dominance)

use anyhow::Result;
use nalgebra::Vector5;
use niodoo_core::EmotionalVector;

/// Maps Qwen embeddings to emotional space
pub struct EmbeddingAdapter {
    // Placeholder for Qwen embedder
    // In real implementation, this would hold the TCS QwenEmbedder
}

impl EmbeddingAdapter {
    pub fn new() -> Self {
        Self {}
    }

    /// Embed text and map to emotional space
    pub async fn embed(&self, text: &str) -> Result<EmotionalVector> {
        // Placeholder: Generate mock embedding based on text length
        // In real implementation, this would call TCS QwenEmbedder
        let mock_embedding: Vec<f32> = (0..1536).map(|i| (text.len() as f32 * (i as f32).sin()) / 100.0).collect();

        // Map to 5D emotional space
        let emotion = self.to_emotional_vector(&mock_embedding);

        Ok(emotion)
    }

    /// Convert 1536-dim embedding to 5D emotional vector
    fn to_emotional_vector(&self, embedding: &[f32]) -> EmotionalVector {
        // Simple projection: average chunks of the embedding
        // In practice, this would be learned weights
        let chunk_size = embedding.len() / 5;

        let mut emotion = Vector5::zeros();
        for i in 0..5 {
            let start = i * chunk_size;
            let end = if i == 4 { embedding.len() } else { (i + 1) * chunk_size };
            let chunk_sum: f32 = embedding[start..end].iter().sum();
            emotion[i] = chunk_sum / (end - start) as f32;
        }

        // Create EmotionalVector with the 5 values
        EmotionalVector::new(
            emotion[0],
            emotion[1],
            emotion[2],
            emotion[3],
            emotion[4],
        )
    }
}