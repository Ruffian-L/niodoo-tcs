//! Simplified Caching System for Niodoo-Feeling
//!
//! This module provides a simplified caching system with:
//! - LRU cache for frequently accessed consciousness states
//! - Basic TTL support for time-sensitive data
//! - Simple cache statistics and monitoring

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{RwLock, Mutex};
use tracing::{debug, info, warn};

/// Simplified cache entry with basic metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub created_at: SystemTime,
    pub ttl_seconds: Option<u64>,
}

impl CacheEntry {
    /// Create a new cache entry
    pub fn new(key: String, value: Vec<u8>, ttl_seconds: Option<u64>) -> Self {
        Self {
            key,
            value,
            created_at: SystemTime::now(),
            ttl_seconds,
        }
    }

    /// Check if entry is expired
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl_seconds {
            let now = SystemTime::now();
            let elapsed = now.duration_since(self.created_at).unwrap_or_default();
            elapsed.as_secs() > ttl
        } else {
            false
        }
    }
}

/// Simplified LRU cache implementation
pub struct LruCache {
    capacity: usize,
    entries: HashMap<String, CacheEntry>,
    access_order: VecDeque<String>,
}

impl LruCache {
    /// Create a new LRU cache
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::new(),
            access_order: VecDeque::new(),
        }
    }

    /// Get value from cache
    pub fn get(&mut self, key: &str) -> Option<&CacheEntry> {
        if let Some(entry) = self.entries.get_mut(key) {
            // Check if expired
            if entry.is_expired() {
                self.remove(key);
                return None;
            }

            // Move to end of access order (most recently used)
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push_back(key.to_string());

            Some(entry)
        } else {
            None
        }
    }

    /// Put value in cache
    pub fn put(&mut self, key: String, value: Vec<u8>, ttl_seconds: Option<u64>) {
        // Remove expired entries first
        self.cleanup();

        // If at capacity, remove least recently used
        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
            if let Some(lru_key) = self.access_order.pop_front() {
                self.entries.remove(&lru_key);
            }
        }

        let entry = CacheEntry::new(key.clone(), value, ttl_seconds);
        self.entries.insert(key.clone(), entry);
        self.access_order.push_back(key);
    }

    /// Remove entry from cache
    pub fn remove(&mut self, key: &str) -> Option<CacheEntry> {
        if let Some(entry) = self.entries.remove(key) {
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            Some(entry)
        } else {
            None
        }
    }

    /// Remove expired entries
    pub fn cleanup(&mut self) {
        let mut expired_keys = Vec::new();

        for (key, entry) in &self.entries {
            if entry.is_expired() {
                expired_keys.push(key.clone());
            }
        }

        for key in expired_keys {
            self.remove(&key);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total_entries = self.entries.len();
        let total_size: usize = self.entries.values().map(|e| e.value.len()).sum();

        CacheStats {
            entries: total_entries,
            total_size_bytes: total_size,
            hit_rate: 0.0, // Simplified - no access tracking
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Simplified TTL-based cache
pub struct TtlCache {
    entries: HashMap<String, CacheEntry>,
    default_ttl_seconds: u64,
}

impl TtlCache {
    /// Create a new TTL cache
    pub fn new(default_ttl_seconds: u64) -> Self {
        Self {
            entries: HashMap::new(),
            default_ttl_seconds,
        }
    }

    /// Get value from cache
    pub fn get(&mut self, key: &str) -> Option<&CacheEntry> {
        if let Some(entry) = self.entries.get_mut(key) {
            // Check if expired
            if entry.is_expired() {
                self.entries.remove(key);
                return None;
            }
            Some(entry)
        } else {
            None
        }
    }

    /// Put value in cache with TTL
    pub fn put(&mut self, key: String, value: Vec<u8>, ttl_seconds: Option<u64>) {
        let ttl = ttl_seconds.unwrap_or(self.default_ttl_seconds);
        let entry = CacheEntry::new(key.clone(), value, Some(ttl));
        self.entries.insert(key, entry);
    }

    /// Remove expired entries
    pub fn cleanup(&mut self) {
        let mut expired_keys = Vec::new();

        for (key, entry) in &self.entries {
            if entry.is_expired() {
                expired_keys.push(key.clone());
            }
        }

        for key in expired_keys {
            self.entries.remove(&key);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total_entries = self.entries.len();
        let total_size: usize = self.entries.values().map(|e| e.value.len()).sum();

        CacheStats {
            entries: total_entries,
            total_size_bytes: total_size,
            hit_rate: 0.0, // Simplified - no access tracking
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Simplified cache manager combining LRU and TTL caches
pub struct SimpleCacheManager {
    lru_cache: Arc<Mutex<LruCache>>,
    ttl_cache: Arc<Mutex<TtlCache>>,
}

impl SimpleCacheManager {
    /// Create a new simple cache manager
    pub async fn new() -> Result<Self> {
        info!("💾 Initializing simplified cache manager");

        Ok(Self {
            lru_cache: Arc::new(Mutex::new(LruCache::new(1000))), // 1000 entries
            ttl_cache: Arc::new(Mutex::new(TtlCache::new(3600))), // 1 hour default TTL
        })
    }

    /// Get cached result
    pub async fn get_cached_result(&self, input: &str) -> Result<Option<String>> {
        let cache_key = Self::generate_cache_key(input);

        // Check LRU cache first
        if let Some(entry) = self.lru_cache.lock().await.get(&cache_key) {
            return Ok(Some(String::from_utf8_lossy(&entry.value).to_string()));
        }

        // Check TTL cache
        if let Some(entry) = self.ttl_cache.lock().await.get(&cache_key) {
            // Promote to LRU cache
            let value = String::from_utf8_lossy(&entry.value).to_string();
            self.lru_cache.lock().await.put(
                cache_key.clone(),
                entry.value.clone(),
                entry.ttl_seconds,
            );

            return Ok(Some(value));
        }

        Ok(None)
    }

    /// Cache result
    pub async fn cache_result(&self, input: &str, result: &str) -> Result<()> {
        let cache_key = Self::generate_cache_key(input);
        let value = result.as_bytes().to_vec();

        // Store in both caches
        self.lru_cache.lock().await.put(
            cache_key.clone(),
            value.clone(),
            Some(300), // 5 minute TTL for LRU
        );

        self.ttl_cache.lock().await.put(
            cache_key,
            value,
            Some(3600), // 1 hour TTL for TTL cache
        );

        Ok(())
    }

    /// Cleanup expired entries
    pub async fn cleanup(&self) -> Result<()> {
        self.lru_cache.lock().await.cleanup();
        self.ttl_cache.lock().await.cleanup();
        Ok(())
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> Result<HashMap<String, CacheStats>> {
        let mut stats = HashMap::new();
        stats.insert("lru_cache".to_string(), self.lru_cache.lock().await.stats());
        stats.insert("ttl_cache".to_string(), self.ttl_cache.lock().await.stats());
        Ok(stats)
    }

    /// Generate cache key from input
    fn generate_cache_key(input: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Clear all caches
    pub async fn clear_all(&self) -> Result<()> {
        self.lru_cache.lock().await.clear();
        self.ttl_cache.lock().await.clear();
        Ok(())
    }
}

/// Simplified cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub entries: usize,
    pub total_size_bytes: usize,
    pub hit_rate: f64,
}

// Legacy alias for backward compatibility
pub type AdvancedCacheManager = SimpleCacheManager;

/// Simplified cache warming strategy
pub struct CacheWarmingStrategy {
    warm_up_data: Vec<(String, String)>,
}

impl CacheWarmingStrategy {
    /// Create a new cache warming strategy
    pub fn new() -> Self {
        Self {
            warm_up_data: Vec::new(),
        }
    }

    /// Add warm-up data
    pub fn add_warm_up_data(&mut self, key: String, value: String) {
        self.warm_up_data.push((key, value));
    }

    /// Warm up cache with pre-loaded data
    pub async fn warm_up_cache(&self, cache_manager: &SimpleCacheManager) -> Result<()> {
        info!("🔥 Warming up cache with {} entries", self.warm_up_data.len());

        for (key, value) in &self.warm_up_data {
            cache_manager.cache_result(key, value).await?;
        }

        info!("✅ Cache warming complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache() {
        let mut cache = LruCache::new(2);

        // Test basic operations
        cache.put("key1".to_string(), b"value1".to_vec(), None);
        cache.put("key2".to_string(), b"value2".to_vec(), None);

        assert_eq!(cache.len(), 2);
        assert!(cache.get("key1").is_some());

        // Add third item should evict least recently used
        cache.put("key3".to_string(), b"value3".to_vec(), None);
        assert_eq!(cache.len(), 2);
        assert!(cache.get("key1").is_none()); // Should be evicted
        assert!(cache.get("key2").is_some()); // Should still be there
        assert!(cache.get("key3").is_some()); // Should be added
    }

    #[test]
    fn test_ttl_cache() {
        let mut cache = TtlCache::new(1); // 1 second TTL

        // Add entry
        cache.put("key1".to_string(), b"value1".to_vec(), Some(1));

        // Should be available immediately
        assert!(cache.get("key1").is_some());

        // Wait for expiration (in real test, would need to wait)
        // For testing, we simulate by manually setting expired state
        if let Some(entry) = cache.entries.get_mut("key1") {
            entry.created_at = SystemTime::UNIX_EPOCH; // Set to very old timestamp
        }

        // Should be expired and removed
        cache.cleanup();
        assert_eq!(cache.len(), 0);
    }

    #[tokio::test]
    async fn test_simple_cache_manager() {
        let cache_manager = SimpleCacheManager::new().await.unwrap();

        // Test basic caching
        let result = cache_manager.cache_result("test_key", "test_value").await;
        assert!(result.is_ok());

        let retrieved = cache_manager.get_cached_result("test_key").await.unwrap();
        assert_eq!(retrieved, Some("test_value".to_string()));

        // Test cache statistics
        let stats = cache_manager.get_stats().await.unwrap();
        assert!(stats.contains_key("lru_cache"));
        assert!(stats.contains_key("ttl_cache"));
    }
}
