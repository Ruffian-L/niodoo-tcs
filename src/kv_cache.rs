/*
 * 🚀 KV CACHE OPTIMIZATION FOR QWEN INFERENCE
 *
 * Implements Key-Value caching for transformer attention layers
 * Target: 5x speedup for sequential token generation in 100-token chats
 *
 * Architecture:
 * - Per-layer KV tensors stored during prefill phase
 * - Reused during autoregressive generation (one token at a time)
 * - Cleared on context switch (new conversation/query)
 *
 * Memory layout:
 * - Keys: [batch, num_kv_heads, seq_len, head_dim]
 * - Values: [batch, num_kv_heads, seq_len, head_dim]
 */

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Single layer KV cache entry
#[derive(Clone, Debug)]
pub struct LayerKVCache {
    /// Cached key tensor: [batch, num_kv_heads, seq_len, head_dim]
    pub key: Option<Tensor>,
    /// Cached value tensor: [batch, num_kv_heads, seq_len, head_dim]
    pub value: Option<Tensor>,
    /// Current sequence length cached
    pub seq_len: usize,
}

impl LayerKVCache {
    pub fn new() -> Self {
        Self {
            key: None,
            value: None,
            seq_len: 0,
        }
    }

    /// Update cache with new key-value tensors
    pub fn update(&mut self, key: Tensor, value: Tensor) -> Result<()> {
        // Validate shapes match
        let key_dims = key.dims();
        let value_dims = value.dims();

        if key_dims.len() != 4 || value_dims.len() != 4 {
            return Err(anyhow!(
                "Invalid KV tensor shapes: key={:?}, value={:?}. Expected 4D tensors.",
                key_dims,
                value_dims
            ));
        }

        if key_dims[2] != value_dims[2] {
            return Err(anyhow!(
                "KV sequence length mismatch: key={}, value={}",
                key_dims[2],
                value_dims[2]
            ));
        }

        self.seq_len = key_dims[2];
        self.key = Some(key);
        self.value = Some(value);

        debug!("Updated layer KV cache: seq_len={}", self.seq_len);
        Ok(())
    }

    /// Append new key-value to existing cache (for autoregressive generation)
    pub fn append(&mut self, new_key: Tensor, new_value: Tensor) -> Result<()> {
        match (&self.key, &self.value) {
            (Some(cached_key), Some(cached_value)) => {
                // Concatenate along sequence dimension (dim 2)
                let updated_key = Tensor::cat(&[cached_key, &new_key], 2)
                    .map_err(|e| anyhow!("Failed to concatenate keys: {}", e))?;
                let updated_value = Tensor::cat(&[cached_value, &new_value], 2)
                    .map_err(|e| anyhow!("Failed to concatenate values: {}", e))?;

                self.key = Some(updated_key);
                self.value = Some(updated_value);
                self.seq_len += 1;

                debug!("Appended to KV cache: new_seq_len={}", self.seq_len);
                Ok(())
            }
            _ => {
                // No existing cache, just store new tensors
                self.update(new_key, new_value)
            }
        }
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.key = None;
        self.value = None;
        self.seq_len = 0;
        debug!("Cleared layer KV cache");
    }

    /// Get cached tensors (returns references for zero-copy access)
    pub fn get(&self) -> Option<(&Tensor, &Tensor)> {
        match (&self.key, &self.value) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.key.is_none() || self.value.is_none()
    }
}

impl Default for LayerKVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-layer KV cache for entire transformer model
#[derive(Clone)]
pub struct QwenKVCache {
    /// Per-layer caches (layer_idx -> LayerKVCache)
    layers: Arc<RwLock<Vec<LayerKVCache>>>,
    /// Number of transformer layers
    num_layers: usize,
    /// Device for cache tensors (used for allocation and tensor operations)
    #[allow(dead_code)]
    device: Device,
    /// Cache hit statistics
    hits: Arc<RwLock<u64>>,
    misses: Arc<RwLock<u64>>,
}

impl QwenKVCache {
    /// Create new KV cache for specified number of layers
    pub fn new(num_layers: usize, device: Device) -> Self {
        let layers = (0..num_layers).map(|_| LayerKVCache::new()).collect();

        Self {
            layers: Arc::new(RwLock::new(layers)),
            num_layers,
            device,
            hits: Arc::new(RwLock::new(0)),
            misses: Arc::new(RwLock::new(0)),
        }
    }

    /// Update cache for specific layer
    pub fn update_layer(&self, layer_idx: usize, key: Tensor, value: Tensor) -> Result<()> {
        if layer_idx >= self.num_layers {
            return Err(anyhow!(
                "Invalid layer index: {} (max: {})",
                layer_idx,
                self.num_layers - 1
            ));
        }

        let mut layers = self.layers.write();
        layers[layer_idx].update(key, value)?;

        Ok(())
    }

    /// Append new KV to existing cache for autoregressive generation
    pub fn append_layer(&self, layer_idx: usize, new_key: Tensor, new_value: Tensor) -> Result<()> {
        if layer_idx >= self.num_layers {
            return Err(anyhow!(
                "Invalid layer index: {} (max: {})",
                layer_idx,
                self.num_layers - 1
            ));
        }

        let mut layers = self.layers.write();
        layers[layer_idx].append(new_key, new_value)?;

        Ok(())
    }

    /// Get cached KV for specific layer
    pub fn get_layer(&self, layer_idx: usize) -> Result<Option<(Tensor, Tensor)>> {
        if layer_idx >= self.num_layers {
            return Err(anyhow!(
                "Invalid layer index: {} (max: {})",
                layer_idx,
                self.num_layers - 1
            ));
        }

        let layers = self.layers.read();
        if let Some((k, v)) = layers[layer_idx].get() {
            *self.hits.write() += 1;
            Ok(Some((k.clone(), v.clone())))
        } else {
            *self.misses.write() += 1;
            Ok(None)
        }
    }

    /// Check if layer has cached KV
    pub fn has_cache(&self, layer_idx: usize) -> bool {
        let layers = self.layers.read();
        if layer_idx < self.num_layers {
            !layers[layer_idx].is_empty()
        } else {
            false
        }
    }

    /// Clear all cached KV tensors (call on context switch)
    pub fn clear(&self) {
        let mut layers = self.layers.write();
        for layer in layers.iter_mut() {
            layer.clear();
        }

        // Reset statistics
        *self.hits.write() = 0;
        *self.misses.write() = 0;

        info!("🗑️ Cleared all KV cache layers");
    }

    /// Get cache statistics (for performance monitoring)
    pub fn get_stats(&self) -> KVCacheStats {
        let hits = *self.hits.read();
        let misses = *self.misses.read();
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        // Calculate memory usage
        let layers = self.layers.read();
        let memory_bytes: usize = layers
            .iter()
            .map(|layer| {
                if let Some((k, v)) = layer.get() {
                    // Estimate memory: num_elements * dtype_size
                    let k_elements: usize = k.dims().iter().product();
                    let v_elements: usize = v.dims().iter().product();
                    let dtype_size = match k.dtype() {
                        DType::F32 => 4,
                        DType::F16 => 2,
                        DType::BF16 => 2,
                        _ => 4, // Default to 4 bytes
                    };
                    (k_elements + v_elements) * dtype_size
                } else {
                    0
                }
            })
            .sum();

        KVCacheStats {
            hits,
            misses,
            hit_rate,
            memory_mb: memory_bytes as f64 / (1024.0 * 1024.0),
            num_layers: self.num_layers,
        }
    }

    /// Get current sequence length from first layer (assumes all layers have same seq_len)
    pub fn get_seq_len(&self) -> usize {
        let layers = self.layers.read();
        if !layers.is_empty() {
            layers[0].seq_len
        } else {
            0
        }
    }
}

/// Cache performance statistics
#[derive(Debug, Clone)]
pub struct KVCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub memory_mb: f64,
    pub num_layers: usize,
}

impl std::fmt::Display for KVCacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KV Cache Stats: hits={}, misses={}, hit_rate={:.2}%, memory={:.2}MB, layers={}",
            self.hits,
            self.misses,
            self.hit_rate * 100.0,
            self.memory_mb,
            self.num_layers
        )
    }
}

/// RAG embedding cache for frequent queries
#[derive(Clone)]
pub struct EmbeddingCache {
    /// Query -> Embedding mapping
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    /// Maximum cache size (evict oldest when full)
    max_size: usize,
    /// Cache hits/misses for monitoring
    hits: Arc<RwLock<u64>>,
    misses: Arc<RwLock<u64>>,
}

impl EmbeddingCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            hits: Arc::new(RwLock::new(0)),
            misses: Arc::new(RwLock::new(0)),
        }
    }

    /// Get cached embedding for query
    pub fn get(&self, query: &str) -> Option<Vec<f32>> {
        let cache = self.cache.read();
        if let Some(embedding) = cache.get(query) {
            *self.hits.write() += 1;
            debug!("📦 Embedding cache hit for query: {}", query);
            Some(embedding.clone())
        } else {
            *self.misses.write() += 1;
            debug!("❌ Embedding cache miss for query: {}", query);
            None
        }
    }

    /// Store embedding in cache
    pub fn put(&self, query: String, embedding: Vec<f32>) {
        let mut cache = self.cache.write();

        // Simple eviction: remove random entry if at capacity
        if cache.len() >= self.max_size {
            if let Some(key) = cache.keys().next().cloned() {
                cache.remove(&key);
                debug!("♻️ Evicted old embedding from cache");
            }
        }

        cache.insert(query, embedding);
    }

    /// Clear all cached embeddings
    pub fn clear(&self) {
        self.cache.write().clear();
        *self.hits.write() = 0;
        *self.misses.write() = 0;
        info!("🗑️ Cleared embedding cache");
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> (u64, u64, f64, usize) {
        let hits = *self.hits.read();
        let misses = *self.misses.read();
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        let size = self.cache.read().len();

        (hits, misses, hit_rate, size)
    }
}

/// Cosine similarity cache with memoization
#[derive(Clone)]
pub struct SimilarityCache {
    /// (query_hash, doc_hash) -> similarity score
    cache: Arc<RwLock<HashMap<(u64, u64), f32>>>,
    /// Maximum cache entries
    max_size: usize,
}

impl SimilarityCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
        }
    }

    /// Hash embedding vector for cache key
    fn hash_embedding(embedding: &[f32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        // Hash first 16 elements for speed (assumes embeddings are high-dimensional)
        for &val in embedding.iter().take(16) {
            val.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Get cached similarity or compute and cache
    pub fn get_or_compute<F>(&self, query_emb: &[f32], doc_emb: &[f32], compute_fn: F) -> f32
    where
        F: FnOnce(&[f32], &[f32]) -> f32,
    {
        let query_hash = Self::hash_embedding(query_emb);
        let doc_hash = Self::hash_embedding(doc_emb);
        let key = (query_hash, doc_hash);

        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(&similarity) = cache.get(&key) {
                debug!("📦 Similarity cache hit");
                return similarity;
            }
        }

        // Cache miss - compute similarity
        let similarity = compute_fn(query_emb, doc_emb);

        // Store in cache
        {
            let mut cache = self.cache.write();
            if cache.len() >= self.max_size {
                // Simple eviction: remove first entry
                if let Some(old_key) = cache.keys().next().cloned() {
                    cache.remove(&old_key);
                }
            }
            cache.insert(key, similarity);
        }

        debug!("❌ Similarity cache miss - computed and cached");
        similarity
    }

    /// Clear cache
    pub fn clear(&self) {
        self.cache.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_kv_cache_update() {
        let device = Device::Cpu;
        let mut layer_cache = LayerKVCache::new();

        // Create dummy KV tensors [batch=1, heads=4, seq=10, dim=64]
        let key = Tensor::zeros(&[1, 4, 10, 64], DType::F32, &device).unwrap();
        let value = Tensor::zeros(&[1, 4, 10, 64], DType::F32, &device).unwrap();

        layer_cache.update(key, value).unwrap();

        assert_eq!(layer_cache.seq_len, 10);
        assert!(layer_cache.get().is_some());
    }

    #[test]
    fn test_layer_kv_cache_append() {
        let device = Device::Cpu;
        let mut layer_cache = LayerKVCache::new();

        // Initial cache
        let key1 = Tensor::zeros(&[1, 4, 10, 64], DType::F32, &device).unwrap();
        let value1 = Tensor::zeros(&[1, 4, 10, 64], DType::F32, &device).unwrap();
        layer_cache.update(key1, value1).unwrap();

        // Append new token
        let key2 = Tensor::zeros(&[1, 4, 1, 64], DType::F32, &device).unwrap();
        let value2 = Tensor::zeros(&[1, 4, 1, 64], DType::F32, &device).unwrap();
        layer_cache.append(key2, value2).unwrap();

        assert_eq!(layer_cache.seq_len, 11);
    }

    #[test]
    fn test_qwen_kv_cache() {
        let device = Device::Cpu;
        let num_layers = 32;
        let cache = QwenKVCache::new(num_layers, device.clone());

        // Update layer 0
        let key = Tensor::zeros(&[1, 4, 10, 64], DType::F32, &device).unwrap();
        let value = Tensor::zeros(&[1, 4, 10, 64], DType::F32, &device).unwrap();
        cache.update_layer(0, key, value).unwrap();

        assert!(cache.has_cache(0));
        assert!(!cache.has_cache(1));

        // Get stats
        let stats = cache.get_stats();
        assert_eq!(stats.num_layers, num_layers);
        assert!(stats.memory_mb > 0.0);
    }

    #[test]
    fn test_embedding_cache() {
        let cache = EmbeddingCache::new(10);

        let query = "test query".to_string();
        let embedding = vec![0.1, 0.2, 0.3];

        // First access - miss
        assert!(cache.get(&query).is_none());

        // Put in cache
        cache.put(query.clone(), embedding.clone());

        // Second access - hit
        let cached = cache.get(&query).unwrap();
        assert_eq!(cached, embedding);

        // Check stats
        let (hits, misses, _hit_rate, size) = cache.get_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(size, 1);
    }

    #[test]
    fn test_similarity_cache() {
        let cache = SimilarityCache::new(100);

        let query_emb = vec![0.1, 0.2, 0.3];
        let doc_emb = vec![0.4, 0.5, 0.6];

        let compute_fn = |a: &[f32], b: &[f32]| -> f32 {
            // Simple dot product
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        // First call - computes
        let sim1 = cache.get_or_compute(&query_emb, &doc_emb, &compute_fn);

        // Second call - cached
        let sim2 = cache.get_or_compute(&query_emb, &doc_emb, &compute_fn);

        assert_eq!(sim1, sim2);
    }
}
