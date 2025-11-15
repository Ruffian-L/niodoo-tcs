//! Local embedding primitives supporting RetrievalEngine.

use anyhow::Result;
use blake3::Hasher as Blake3Hasher;
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Mutex;

const DEFAULT_EMBEDDING_DIM: usize = 384;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

impl Document {
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            embedding: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

pub trait EmbeddingModel {
    fn generate_embedding(&self, text: &str) -> Result<Vec<f32>>;
    fn get_dimension(&self) -> usize;
}

#[derive(Debug)]
pub struct MathematicalEmbeddingModel {
    embedding_dim: usize,
    frequency_bands: Vec<f32>,
}

impl Default for MathematicalEmbeddingModel {
    fn default() -> Self {
        Self::new(DEFAULT_EMBEDDING_DIM)
    }
}

impl MathematicalEmbeddingModel {
    pub fn new(embedding_dim: usize) -> Self {
        let frequency_bands = (0..embedding_dim)
            .map(|i| ((i as f32 + 1.0).ln() + 1.2).sin().abs() + 0.35)
            .collect();
        Self {
            embedding_dim,
            frequency_bands,
        }
    }

    fn project(&self, text: &str) -> Vec<f32> {
        let tokens = tokenize(text);
        let mut embedding = vec![0.0f32; self.embedding_dim];
        if tokens.is_empty() {
            return embedding;
        }

        for token in tokens {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            token.hash(&mut hasher);
            let raw = hasher.finish();
            let index = (raw as usize) % self.embedding_dim;
            let phase = ((raw >> 32) as f32 / u32::MAX as f32) * std::f32::consts::TAU;
            embedding[index] += phase.cos() * self.frequency_bands[index];
        }

        normalise(&mut embedding);
        embedding
    }
}

impl EmbeddingModel for MathematicalEmbeddingModel {
    fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.project(text))
    }

    fn get_dimension(&self) -> usize {
        self.embedding_dim
    }
}

#[derive(Debug, Clone)]
pub struct LocalEmbeddingConfig {
    pub model_id: String,
    pub max_seq_len: usize,
    pub embedding_dim: usize,
    pub use_pooled_output: bool,
    pub cache_size: usize,
}

impl Default for LocalEmbeddingConfig {
    fn default() -> Self {
        Self {
            model_id: "niodoo/local-mobius".to_string(),
            max_seq_len: 512,
            embedding_dim: DEFAULT_EMBEDDING_DIM,
            use_pooled_output: true,
            cache_size: 1024,
        }
    }
}

pub struct LocalEmbeddingGenerator {
    config: LocalEmbeddingConfig,
    baseline_model: MathematicalEmbeddingModel,
    cache: Mutex<LruCache<u64, Vec<f32>>>,
}

impl LocalEmbeddingGenerator {
    pub fn new(config: LocalEmbeddingConfig) -> Result<Self> {
        let capacity = config.cache_size.max(64);
        let cache_capacity = NonZeroUsize::new(capacity).unwrap_or_else(|| NonZeroUsize::new(1024).unwrap());
        Ok(Self {
            baseline_model: MathematicalEmbeddingModel::new(config.embedding_dim),
            cache: Mutex::new(LruCache::new(cache_capacity)),
            config,
        })
    }

    pub fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; self.config.embedding_dim]);
        }

        let fingerprint = fingerprint(text);
        if let Some(cached) = self.cache.lock().unwrap().get(&fingerprint) {
            return Ok(cached.clone());
        }

        let embedding = self.baseline_model.project(text);
        self.cache
            .lock()
            .unwrap()
            .put(fingerprint, embedding.clone());
        Ok(embedding)
    }

    pub fn dimension(&self) -> usize {
        self.config.embedding_dim
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|token| {
            let cleaned = token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if cleaned.is_empty() {
                None
            } else {
                Some(cleaned)
            }
        })
        .collect()
}

fn normalise(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

fn fingerprint(text: &str) -> u64 {
    let mut hasher = Blake3Hasher::new();
    hasher.update(text.as_bytes());
    let digest = hasher.finalize();
    let bytes = digest.as_bytes();
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_are_normalised() {
        let generator = LocalEmbeddingGenerator::new(LocalEmbeddingConfig::default()).unwrap();
        let embedding = generator
            .generate_embedding("Möbius empathy aligns with joy")
            .unwrap();
        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3);
    }

    #[test]
    fn cache_respects_identical_inputs() {
        let generator = LocalEmbeddingGenerator::new(LocalEmbeddingConfig::default()).unwrap();
        let a = generator.generate_embedding("topological love").unwrap();
        let b = generator.generate_embedding("topological love").unwrap();
        assert_eq!(a, b);
    }
}
