//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Privacy configuration for embedding storage
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Enable content hashing before storage
    pub hash_embeddings: bool,

    /// Add differential privacy noise to embeddings
    pub add_differential_privacy: bool,

    /// Noise scale for differential privacy
    pub privacy_noise_scale: f32,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            hash_embeddings: false,
            add_differential_privacy: false,
            privacy_noise_scale: 0.1,
        }
    }
}

/// Utility for privacy-preserving embedding operations
#[derive(Clone)]
pub struct EmbeddingPrivacyShield {
    config: PrivacyConfig,
}

impl EmbeddingPrivacyShield {
    pub fn new(config: PrivacyConfig) -> Self {
        Self { config }
    }

    /// Hash the content to create a deterministic, privacy-preserving embedding identifier
    pub fn hash_content(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let result = hasher.finalize();
        format!("{:x}", result)
    }

    /// Add differential privacy noise to embedding
    pub fn add_noise(embedding: &Array1<f32>, noise_scale: f32) -> Array1<f32> {
        use rand_distr::{Distribution, Normal};
        use rand_chacha::ChaCha8Rng;
        use rand_chacha::rand_core::SeedableRng;

        // Use ChaCha8Rng which is compatible across rand versions
        let mut rng = ChaCha8Rng::from_entropy();
        let normal = Normal::new(0.0, noise_scale as f64).unwrap();

        Array1::from_vec(
            embedding
                .iter()
                .map(|&x| x + normal.sample(&mut rng) as f32)
                .collect(),
        )
    }

    /// Process embedding based on privacy configuration
    pub fn process_embedding(
        &self,
        content: &str,
        embedding: Array1<f32>,
    ) -> (String, Array1<f32>) {
        let content_hash = Self::hash_content(content);

        let processed_embedding = if self.config.add_differential_privacy {
            Self::add_noise(&embedding, self.config.privacy_noise_scale)
        } else {
            embedding
        };

        (content_hash, processed_embedding)
    }

    /// Verify if a document can be retrieved using hash
    pub fn verify_document_access(&self, access_token: &str, stored_content_hash: &str) -> bool {
        // Implement optional more complex access control
        access_token == stored_content_hash
    }
}

// Example usage in storage/embedding generation contexts
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_content_hashing() {
        let content = "Secret document about AI consciousness";
        let hash = EmbeddingPrivacyShield::hash_content(content);

        assert_eq!(hash.len(), 64); // SHA256 hash length
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_differential_privacy_noise() {
        let config = PrivacyConfig {
            hash_embeddings: true,
            add_differential_privacy: true,
            privacy_noise_scale: 0.1,
        };

        let shield = EmbeddingPrivacyShield::new(config);

        let original_embedding = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let (hash, noisy_embedding) =
            shield.process_embedding("Test content", original_embedding.clone());

        assert_ne!(original_embedding, noisy_embedding);
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn test_thread_safe_rng() {
        // Test that RNG works correctly with mutable borrow in closure
        let embedding = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let noisy = EmbeddingPrivacyShield::add_noise(&embedding, 0.1);

        // Verify all values are different (with high probability)
        assert_eq!(embedding.len(), noisy.len());

        // Check that noise was actually added (at least one value different)
        let has_difference = embedding
            .iter()
            .zip(noisy.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(has_difference, "Noise should be added to embedding");
    }

    #[test]
    fn test_privacy_shield_clone() {
        // Test that EmbeddingPrivacyShield can be cloned (thread-safe)
        let config = PrivacyConfig::default();
        let shield1 = EmbeddingPrivacyShield::new(config);
        let shield2 = shield1.clone();

        let content = "Test content";
        let embedding = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let (hash1, _) = shield1.process_embedding(content, embedding.clone());
        let (hash2, _) = shield2.process_embedding(content, embedding);

        // Hashes should be identical (deterministic)
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_large_embedding_noise() {
        // Test RNG with larger embeddings to ensure no borrow issues
        let large_embedding = Array1::from_vec(vec![1.0; 1024]);
        let noisy = EmbeddingPrivacyShield::add_noise(&large_embedding, 0.05);

        assert_eq!(large_embedding.len(), noisy.len());

        // Check statistical properties: noise should be roughly centered around original values
        let mean_diff: f32 = large_embedding
            .iter()
            .zip(noisy.iter())
            .map(|(a, b)| b - a)
            .sum::<f32>()
            / large_embedding.len() as f32;

        // Mean difference should be close to 0 (within 3 standard deviations)
        assert!(
            mean_diff.abs() < 0.15,
            "Mean noise should be near zero, got {}",
            mean_diff
        );
    }
}
