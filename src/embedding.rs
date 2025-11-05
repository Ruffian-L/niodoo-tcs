use anyhow::Result;
use rand::prelude::*;

pub struct QwenEmbedder {
    // Simplified mock embedder
}

impl QwenEmbedder {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f64>> {
        // Mock embedding: create 896D vector based on text hash
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(text, &mut hasher);
        let seed = std::hash::Hasher::finish(&hasher) as u64;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut embedding = Vec::with_capacity(896);

        for _ in 0..896 {
            embedding.push(rng.gen_range(-1.0..1.0));
        }

        Ok(embedding)
    }
}