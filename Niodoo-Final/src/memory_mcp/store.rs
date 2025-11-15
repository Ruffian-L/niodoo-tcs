// Copyright (c) 2025 Jason Van Pham (ruffian-l on GitHub) @ The Niodoo Collaborative
// Licensed under the MIT License - See LICENSE file for details
// Attribution required for all derivative works

//! Memory Store - RocksDB backed persistence layer
//!
//! This module provides a persistent storage layer for consciousness memory,
//! bridging the MemorySystem (adaptive TTL layers) with RocksDB persistence.
//!
//! NO HARDCODING POLICY:
//! - All paths derived from configuration
//! - All TTLs calculated from mathematical constants
//! - All thresholds derived from consciousness state
//! - All logging via log crate (no println)

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use log::{debug, info};
// use rocksdb::{DB, Options, WriteBatch};  // Temporarily disabled
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;

use crate::config::MemoryConfig;

/// Memory store with in-memory persistence (RocksDB temporarily disabled)
///
/// Provides key-value storage with:
/// - In-memory HashMap storage
/// - Basic statistics tracking
/// - Configuration support
/// - Access frequency tracking for spatial consolidation
///
/// Thread-safe for async operations (Send + Sync)
pub struct MemoryStore {
    /// In-memory storage (temporary replacement for RocksDB)
    storage: Arc<parking_lot::RwLock<HashMap<String, Value>>>,

    /// Configuration
    config: MemoryConfig,

    /// Statistics
    stats: Arc<parking_lot::RwLock<MemoryStats>>,

    /// Access frequency tracking for spatial consolidation
    access_frequency: Arc<parking_lot::RwLock<HashMap<String, (usize, Option<DateTime<Utc>>)>>>,
}

#[derive(Debug, Clone, Default)]
struct MemoryStats {
    /// Total reads
    reads: u64,
    /// Total writes
    writes: u64,
    /// Total deletes
    deletes: u64,
}

impl MemoryStore {
    /// Create a new memory store
    pub async fn new(config: &MemoryConfig) -> Result<Self> {
        info!("Initializing in-memory memory store (RocksDB temporarily disabled)...");

        // Ensure database directory exists (for future RocksDB use)
        let db_path = config.db_path.clone();
        std::fs::create_dir_all(&db_path)
            .context("Failed to create database directory")?;

        debug!("In-memory storage configuration:");
        debug!("  Path: {:?}", db_path);
        debug!("  Cache size: {} MB", config.max_cache_size_mb);

        info!("In-memory memory store initialized successfully");

        Ok(Self {
            storage: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            config: config.clone(),
            stats: Arc::new(parking_lot::RwLock::new(MemoryStats::default())),
            access_frequency: Arc::new(parking_lot::RwLock::new(HashMap::new())),
        })
    }

    /// Store a value with a key
    pub async fn store(&mut self, key: &str, value: &Value) -> Result<()> {
        debug!("Storing key: {}", key);

        // Store in in-memory HashMap
        let mut storage = self.storage.write();
        storage.insert(key.to_string(), value.clone());

        let mut stats = self.stats.write();
        stats.writes += 1;

        Ok(())
    }

    /// Retrieve a value by key
    pub async fn retrieve(&mut self, key: &str) -> Result<Value> {
        debug!("Retrieving key: {}", key);

        // Read from in-memory HashMap
        let storage = self.storage.read();
        let value = storage.get(key)
            .ok_or_else(|| anyhow::anyhow!("Key not found: {}", key))?
            .clone();

        // Record access for spatial consolidation
        self.record_access(key).await?;

        let mut stats = self.stats.write();
        stats.reads += 1;

        Ok(value)
    }

    /// Delete a value by key
    pub async fn delete(&mut self, key: &str) -> Result<()> {
        debug!("Deleting key: {}", key);

        // Delete from in-memory HashMap
        let mut storage = self.storage.write();
        storage.remove(key);

        let mut stats = self.stats.write();
        stats.deletes += 1;

        Ok(())
    }

    /// Batch store multiple key-value pairs atomically
    pub async fn batch_store(&mut self, entries: Vec<(String, Value)>) -> Result<()> {
        debug!("Batch storing {} entries", entries.len());

        // Store all entries in in-memory HashMap
        let mut storage = self.storage.write();
        for (key, value) in entries {
            storage.insert(key, value);
        }

        let mut stats = self.stats.write();
        stats.writes += 1;

        Ok(())
    }

    /// Check if a key exists
    pub async fn exists(&self, key: &str) -> Result<bool> {
        let storage = self.storage.read();
        Ok(storage.contains_key(key))
    }

    /// Track memory access for spatial consolidation
    pub async fn record_access(&self, memory_id: &str) -> Result<()> {
        let mut access_data = self.access_frequency.write();
        let current_time = Utc::now();

        let (count, last_access) = access_data.get(memory_id).unwrap_or(&(0, None));
        let new_count = count + 1;
        let new_last_access = Some(current_time);

        access_data.insert(memory_id.to_string(), (new_count, new_last_access));

        Ok(())
    }

    /// Get access frequency for a memory
    pub async fn get_access_frequency(&self, memory_id: &str) -> Result<(usize, Option<DateTime<Utc>>)> {
        let access_data = self.access_frequency.read();
        Ok(access_data.get(memory_id).copied().unwrap_or((0, None)))
    }

    /// Get access frequency normalized for spatial consolidation
    pub async fn get_normalized_access_frequency(&self, memory_id: &str, max_frequency: usize) -> Result<f64> {
        let (count, _) = self.get_access_frequency(memory_id).await?;
        Ok(if max_frequency > 0 {
            count as f64 / max_frequency as f64
        } else {
            0.0
        })
    }

    /// Get memory store statistics
    pub fn stats(&self) -> MemoryStoreStats {
        let stats = self.stats.read();
        MemoryStoreStats {
            reads: stats.reads,
            writes: stats.writes,
            deletes: stats.deletes,
            cache_size_mb: self.config.max_cache_size_mb,
        }
    }

    /// Compact database for optimal performance (no-op for in-memory storage)
    pub async fn compact(&self) -> Result<()> {
        info!("Compacting in-memory storage (no-op)...");
        info!("In-memory storage compaction complete");
        Ok(())
    }

    /// Flush writes to disk (no-op for in-memory storage)
    pub async fn flush(&self) -> Result<()> {
        debug!("Flushing in-memory storage (no-op)...");
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStoreStats {
    pub reads: u64,
    pub writes: u64,
    pub deletes: u64,
    pub cache_size_mb: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let config = MemoryConfig {
            max_entries: 10000,
            decay_rate: 0.01,
            consolidation_threshold: 0.5,
            persistent: true,
            storage_path: PathBuf::from("./test_db_store"),
            db_path: PathBuf::from("./test_db_store"),
            max_cache_size_mb: 64,
        };

        let mut store = MemoryStore::new(&config).await.unwrap();

        let key = "test_key";
        let value = json!({
            "data": "test_value",
            "number": 42
        });

        // Store
        store.store(key, &value).await.unwrap();

        // Retrieve
        let retrieved = store.retrieve(key).await.unwrap();
        assert_eq!(retrieved, value);

        // Cleanup
        std::fs::remove_dir_all("./test_db_store").ok();
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let config = MemoryConfig {
            max_entries: 10000,
            decay_rate: 0.01,
            consolidation_threshold: 0.5,
            persistent: true,
            storage_path: PathBuf::from("./test_db_batch"),
            db_path: PathBuf::from("./test_db_batch"),
            max_cache_size_mb: 64,
        };

        let mut store = MemoryStore::new(&config).await.unwrap();

        let entries = vec![
            ("key1".to_string(), json!({"val": 1})),
            ("key2".to_string(), json!({"val": 2})),
            ("key3".to_string(), json!({"val": 3})),
        ];

        // Batch store
        store.batch_store(entries).await.unwrap();

        // Verify
        assert!(store.exists("key1").await.unwrap());
        assert!(store.exists("key2").await.unwrap());
        assert!(store.exists("key3").await.unwrap());

        // Cleanup
        std::fs::remove_dir_all("./test_db_batch").ok();
    }
}
