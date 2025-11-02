use std::convert::TryInto;
use std::hash::Hasher;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ahash::AHasher;
use anyhow::{anyhow, Context, Result};
use blake3::hash as blake3_hash;
use bytemuck::cast_slice;
use lru::LruCache;
use lz4_flex::block::{compress_prepend_size, decompress_size_prepended};
use parking_lot::RwLock;
use tokio::sync::Mutex as AsyncMutex;
use tracing::warn;

use crate::erag::CollapseResult;
use crate::metrics::cache_metrics;

const CACHE_KEY_FAST_PATH_MAX_LEN: usize = 96;

#[derive(Debug, Clone)]
pub struct CacheHit<T> {
    pub value: T,
    pub compression_ratio: Option<f64>,
}

#[derive(Clone, Debug)]
struct CacheEntry<T> {
    value: T,
    inserted_at: Instant,
}

impl<T> CacheEntry<T> {
    fn new(value: T, inserted_at: Instant) -> Self {
        Self { value, inserted_at }
    }

    fn is_expired(&self, now: Instant, ttl: Duration) -> bool {
        now.duration_since(self.inserted_at) > ttl
    }
}

#[derive(Clone)]
pub struct PipelineCache<T> {
    ttl: Arc<RwLock<Duration>>,
    inner: Arc<AsyncMutex<LruCache<u64, CacheEntry<T>>>>,
}

impl<T> PipelineCache<T> {
    pub fn new(capacity: NonZeroUsize, ttl: Duration) -> Self {
        Self {
            ttl: Arc::new(RwLock::new(ttl)),
            inner: Arc::new(AsyncMutex::new(LruCache::new(capacity))),
        }
    }

    pub fn update_ttl(&self, ttl: Duration) {
        *self.ttl.write() = ttl;
    }

    fn ttl(&self) -> Duration {
        *self.ttl.read()
    }

    pub async fn get(&self, key: &u64, now: Instant) -> Option<T>
    where
        T: Clone,
    {
        let ttl = self.ttl();
        let mut guard = self.inner.lock().await;
        if let Some(entry) = guard.get(key) {
            if entry.is_expired(now, ttl) {
                guard.pop(key);
                None
            } else {
                Some(entry.value.clone())
            }
        } else {
            None
        }
    }

    pub async fn insert(&self, key: u64, value: T, now: Instant) {
        let mut guard = self.inner.lock().await;
        guard.put(key, CacheEntry::new(value, now));
    }

    pub async fn invalidate(&self, key: &u64) {
        self.inner.lock().await.pop(key);
    }
}

#[derive(Clone, Debug)]
struct CachedEmbedding {
    payload: Vec<u8>,
    len_f32: usize,
    raw_len: usize,
    compressed: bool,
}

impl CachedEmbedding {
    fn new(data: &[f32], min_bytes: usize) -> Self {
        let raw_bytes = cast_slice(data).to_vec();
        let raw_len = raw_bytes.len();
        let (payload, compressed) = if raw_len >= min_bytes {
            let compressed = compress_prepend_size(&raw_bytes);
            if compressed.len() < raw_len {
                (compressed, true)
            } else {
                (raw_bytes, false)
            }
        } else {
            (raw_bytes, false)
        };

        Self {
            payload,
            len_f32: data.len(),
            raw_len,
            compressed,
        }
    }

    fn compression_ratio(&self) -> Option<f64> {
        if self.compressed {
            Some(self.payload.len() as f64 / self.raw_len as f64)
        } else {
            None
        }
    }

    fn decode(&self) -> Result<Vec<f32>> {
        let bytes = if self.compressed {
            decompress_size_prepended(&self.payload)
                .context("Failed to decompress embedding cache entry")?
        } else {
            self.payload.clone()
        };

        let expected_bytes = self.len_f32 * std::mem::size_of::<f32>();
        if bytes.len() != expected_bytes {
            return Err(anyhow!(
                "Embedding cache decode length mismatch: expected {} bytes, got {}",
                expected_bytes,
                bytes.len()
            ));
        }

        let mut decoded = Vec::with_capacity(self.len_f32);
        for chunk in bytes.chunks_exact(4) {
            decoded.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(decoded)
    }
}

#[derive(Clone, Debug)]
struct CachedCollapse {
    payload: Vec<u8>,
    raw_len: usize,
    compressed: bool,
}

impl CachedCollapse {
    fn new(value: &CollapseResult, min_bytes: usize) -> Result<Self> {
        let raw_bytes = bincode::serialize(value).context("Failed to serialize collapse result")?;
        let raw_len = raw_bytes.len();
        let (payload, compressed) = if raw_len >= min_bytes {
            let compressed = compress_prepend_size(&raw_bytes);
            if compressed.len() < raw_len {
                (compressed, true)
            } else {
                (raw_bytes, false)
            }
        } else {
            (raw_bytes, false)
        };

        Ok(Self {
            payload,
            raw_len,
            compressed,
        })
    }

    fn compression_ratio(&self) -> Option<f64> {
        if self.compressed {
            Some(self.payload.len() as f64 / self.raw_len as f64)
        } else {
            None
        }
    }

    fn decode(&self) -> Result<CollapseResult> {
        let bytes = if self.compressed {
            decompress_size_prepended(&self.payload)
                .context("Failed to decompress collapse cache entry")?
        } else {
            self.payload.clone()
        };

        let result =
            bincode::deserialize(&bytes).context("Failed to deserialize collapse cache entry")?;
        Ok(result)
    }
}

#[derive(Clone)]
pub struct EmbeddingCache {
    inner: PipelineCache<CachedEmbedding>,
    compression_min_bytes: usize,
}

impl EmbeddingCache {
    pub fn new(capacity: NonZeroUsize, ttl: Duration, compression_min_bytes: usize) -> Self {
        Self {
            inner: PipelineCache::new(capacity, ttl),
            compression_min_bytes,
        }
    }

    pub fn update_ttl(&self, ttl: Duration) {
        self.inner.update_ttl(ttl);
    }

    pub async fn fetch(&self, key: &u64, now: Instant) -> Result<Option<CacheHit<Vec<f32>>>> {
        match self.inner.get(key, now).await {
            Some(entry) => {
                let ratio = entry.compression_ratio();
                let value = entry.decode()?;
                cache_metrics().record_embedding_hit(ratio);
                Ok(Some(CacheHit {
                    value,
                    compression_ratio: ratio,
                }))
            }
            None => {
                cache_metrics().record_embedding_miss();
                Ok(None)
            }
        }
    }

    pub async fn store(&self, key: u64, embedding: &[f32], now: Instant) -> Result<Option<f64>> {
        let entry = CachedEmbedding::new(embedding, self.compression_min_bytes);
        let ratio = entry.compression_ratio();
        cache_metrics().observe_embedding_entry(ratio.unwrap_or(1.0));
        self.inner.insert(key, entry, now).await;
        Ok(ratio)
    }

    pub async fn invalidate(&self, key: &u64) {
        self.inner.invalidate(key).await;
    }
}

#[derive(Clone)]
pub struct CollapseCache {
    inner: PipelineCache<CachedCollapse>,
    compression_min_bytes: usize,
}

impl CollapseCache {
    pub fn new(capacity: NonZeroUsize, ttl: Duration, compression_min_bytes: usize) -> Self {
        Self {
            inner: PipelineCache::new(capacity, ttl),
            compression_min_bytes,
        }
    }

    pub fn update_ttl(&self, ttl: Duration) {
        self.inner.update_ttl(ttl);
    }

    pub async fn fetch(&self, key: &u64, now: Instant) -> Result<Option<CacheHit<CollapseResult>>> {
        match self.inner.get(key, now).await {
            Some(entry) => {
                let ratio = entry.compression_ratio();
                let value = entry.decode()?;
                cache_metrics().record_collapse_hit(ratio);
                Ok(Some(CacheHit {
                    value,
                    compression_ratio: ratio,
                }))
            }
            None => {
                cache_metrics().record_collapse_miss();
                Ok(None)
            }
        }
    }

    pub async fn store(
        &self,
        key: u64,
        collapse: &CollapseResult,
        now: Instant,
    ) -> Result<Option<f64>> {
        let entry = CachedCollapse::new(collapse, self.compression_min_bytes)?;
        let ratio = entry.compression_ratio();
        cache_metrics().observe_collapse_entry(ratio.unwrap_or(1.0));
        self.inner.insert(key, entry, now).await;
        Ok(ratio)
    }

    pub async fn invalidate(&self, key: &u64) {
        self.inner.invalidate(key).await;
    }
}

pub fn cache_key(prompt: &str) -> u64 {
    if prompt.len() <= CACHE_KEY_FAST_PATH_MAX_LEN {
        let mut hasher = AHasher::default();
        hasher.write(prompt.as_bytes());
        hasher.finish()
    } else {
        let digest = blake3_hash(prompt.as_bytes());
        let mut hasher = AHasher::default();
        hasher.write(digest.as_bytes());
        hasher.finish()
    }
}
