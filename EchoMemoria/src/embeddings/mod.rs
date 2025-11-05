//! Active Embeddings System with MCP Integration
//! 
//! This module provides real-time code embedding capabilities that sync with the MCP server
//! to give the AI transformer a constantly-updated semantic map of the codebase.

use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};
use sha2::{Sha256, Digest};
use tokio::sync::RwLock;

/// Core embedding engine that handles text embedding and storage
pub struct EmbeddingEngine {
    model: fastembed::Embedding,
    db_path: PathBuf,
    embeddings_cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    mcp_notifier: Option<Arc<dyn McpNotifier + Send + Sync>>,
}

/// MCP notification trait for real-time updates
pub trait McpNotifier {
    async fn notify_embedding_created(&self, file_path: &str, embedding_id: &str) -> Result<()>;
    async fn notify_embedding_updated(&self, file_path: &str, embedding_id: &str) -> Result<()>;
    async fn notify_batch_complete(&self, total_files: usize, success_count: usize) -> Result<()>;
    async fn notify_error(&self, error: &str) -> Result<()>;
}

/// Embedding metadata for tracking and versioning
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingMetadata {
    pub file_path: String,
    pub file_hash: String,
    pub timestamp: u64,
    pub model: String,
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub embedding_id: String,
    pub file_size: u64,
}

/// Embedding file format for sidecar storage
#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingFile {
    pub metadata: EmbeddingMetadata,
    pub embedding: Vec<f32>,
}

/// Embedding statistics for monitoring
#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingStats {
    pub total_files: usize,
    pub embedded_files: usize,
    pub stale_embeddings: usize,
    pub avg_embed_time_ms: f64,
    pub last_sync: u64,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl EmbeddingEngine {
    /// Create a new embedding engine
    pub fn new(db_path: PathBuf) -> Result<Self> {
        let model = fastembed::Embedding::new(fastembed::InitOptions {
            model_name: fastembed::EmbeddingModel::AllMiniLmL6V2,
            show_download_progress: true,
            ..Default::default()
        })?;

        // Ensure embedding directory exists
        std::fs::create_dir_all(&db_path)
            .context("Failed to create embedding directory")?;

        Ok(Self {
            model,
            db_path,
            embeddings_cache: Arc::new(RwLock::new(HashMap::new())),
            mcp_notifier: None,
        })
    }

    /// Set MCP notifier for real-time updates
    pub fn set_mcp_notifier(&mut self, notifier: Arc<dyn McpNotifier + Send + Sync>) {
        self.mcp_notifier = Some(notifier);
    }

    /// Embed a single file and return the embedding vector
    pub async fn embed_file(&self, path: &Path) -> Result<Vec<f32>> {
        let start_time = SystemTime::now();
        
        // Read file content
        let content = std::fs::read_to_string(path)
            .context("Failed to read file")?;
        
        // Generate embedding
        let embedding = self.embed_text(&content)?;
        
        // Calculate embedding time
        let embed_time = start_time.elapsed()?.as_millis() as f64;
        
        // Update cache
        {
            let mut cache = self.embeddings_cache.write().await;
            cache.insert(path.to_string_lossy().to_string(), embedding.clone());
        }

        // Notify MCP server
        if let Some(notifier) = &self.mcp_notifier {
            let embedding_id = self.generate_embedding_id(path);
            notifier.notify_embedding_created(
                &path.to_string_lossy(),
                &embedding_id
            ).await.ok(); // Don't fail on notification errors
        }

        Ok(embedding)
    }

    /// Embed text content directly
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        // Chunk large texts if needed
        let chunks = self.chunk_text(text);
        let mut embeddings = Vec::new();
        
        for chunk in chunks {
            let embedding = self.model.embed(&[chunk], None)?;
            embeddings.extend(embedding.into_iter().flatten());
        }
        
        Ok(embeddings)
    }

    /// Save embedding to sidecar file
    pub async fn save_embedding(&self, path: &Path, embedding: Vec<f32>) -> Result<()> {
        let file_hash = self.calculate_file_hash(path)?;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();
        
        let metadata = EmbeddingMetadata {
            file_path: path.to_string_lossy().to_string(),
            file_hash,
            timestamp,
            model: "all-MiniLM-L6-v2".to_string(),
            chunk_index: 0,
            total_chunks: 1,
            embedding_id: self.generate_embedding_id(path),
            file_size: std::fs::metadata(path)?.len(),
        };

        let embedding_file = EmbeddingFile {
            metadata,
            embedding,
        };

        let sidecar_path = self.get_sidecar_path(path);
        
        // Save as binary for efficiency
        let file = std::fs::File::create(&sidecar_path)
            .context("Failed to create sidecar file")?;
        bincode::serialize_into(file, &embedding_file)
            .context("Failed to serialize embedding")?;

        Ok(())
    }

    /// Load embedding from sidecar file
    pub async fn load_embedding(&self, path: &Path) -> Result<Vec<f32>> {
        // Check cache first
        {
            let cache = self.embeddings_cache.read().await;
            if let Some(embedding) = cache.get(&path.to_string_lossy()) {
                return Ok(embedding.clone());
            }
        }

        let sidecar_path = self.get_sidecar_path(path);
        
        if !sidecar_path.exists() {
            return Err(anyhow::anyhow!("No embedding found for file"));
        }

        let file = std::fs::File::open(&sidecar_path)
            .context("Failed to open sidecar file")?;
        
        let embedding_file: EmbeddingFile = bincode::deserialize_from(file)
            .context("Failed to deserialize embedding")?;

        // Verify file hasn't changed
        let current_hash = self.calculate_file_hash(path)?;
        if embedding_file.metadata.file_hash != current_hash {
            return Err(anyhow::anyhow!("File has changed, embedding is stale"));
        }

        // Update cache
        {
            let mut cache = self.embeddings_cache.write().await;
            cache.insert(path.to_string_lossy().to_string(), embedding_file.embedding.clone());
        }

        Ok(embedding_file.embedding)
    }

    /// Check if embedding exists and is up-to-date
    pub fn is_embedding_current(&self, path: &Path) -> Result<bool> {
        let sidecar_path = self.get_sidecar_path(path);
        
        if !sidecar_path.exists() {
            return Ok(false);
        }

        let file = std::fs::File::open(&sidecar_path)?;
        let embedding_file: EmbeddingFile = bincode::deserialize_from(file)?;
        
        let current_hash = self.calculate_file_hash(path)?;
        Ok(embedding_file.metadata.file_hash == current_hash)
    }

    /// Get embedding statistics
    pub async fn get_stats(&self) -> Result<EmbeddingStats> {
        let cache = self.embeddings_cache.read().await;
        
        Ok(EmbeddingStats {
            total_files: 0, // Will be updated by watcher
            embedded_files: cache.len(),
            stale_embeddings: 0, // Will be calculated by watcher
            avg_embed_time_ms: 0.0, // Will be tracked by watcher
            last_sync: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            cache_hits: 0, // Will be tracked
            cache_misses: 0, // Will be tracked
        })
    }

    /// Generate unique embedding ID
    fn generate_embedding_id(&self, path: &Path) -> String {
        let mut hasher = Sha256::new();
        hasher.update(path.to_string_lossy().as_bytes());
        hasher.update(b"embedding");
        format!("{:x}", hasher.finalize())
    }

    /// Calculate file hash for change detection
    fn calculate_file_hash(&self, path: &Path) -> Result<String> {
        let content = std::fs::read(path)?;
        let mut hasher = Sha256::new();
        hasher.update(&content);
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Get sidecar file path
    fn get_sidecar_path(&self, path: &Path) -> PathBuf {
        let mut sidecar_path = path.to_path_buf();
        sidecar_path.set_extension("embedding");
        sidecar_path
    }

    /// Chunk text for large files
    fn chunk_text(&self, text: &str) -> Vec<String> {
        const MAX_CHUNK_SIZE: usize = 1000; // Adjust based on model limits
        
        if text.len() <= MAX_CHUNK_SIZE {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        
        while start < text.len() {
            let end = (start + MAX_CHUNK_SIZE).min(text.len());
            let chunk = &text[start..end];
            
            // Try to break at a natural boundary
            let break_point = if end < text.len() {
                chunk.rfind('\n').unwrap_or(chunk.rfind(' ').unwrap_or(chunk.len()))
            } else {
                chunk.len()
            };
            
            chunks.push(chunk[..break_point].to_string());
            start += break_point;
        }
        
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_embed_single_file() {
        let temp_dir = tempdir().unwrap();
        let engine = EmbeddingEngine::new(temp_dir.path().to_path_buf()).unwrap();
        
        let test_file = temp_dir.path().join("test.rs");
        fs::write(&test_file, "fn main() { println!(\"Hello, world!\"); }").unwrap();
        
        let embedding = engine.embed_file(&test_file).await.unwrap();
        assert!(!embedding.is_empty());
        assert_eq!(embedding.len(), 384); // all-MiniLM-L6-v2 dimension
    }

    #[tokio::test]
    async fn test_sidecar_roundtrip() {
        let temp_dir = tempdir().unwrap();
        let engine = EmbeddingEngine::new(temp_dir.path().to_path_buf()).unwrap();
        
        let test_file = temp_dir.path().join("test.rs");
        fs::write(&test_file, "fn main() { println!(\"Hello, world!\"); }").unwrap();
        
        let embedding = engine.embed_file(&test_file).await.unwrap();
        engine.save_embedding(&test_file, embedding.clone()).await.unwrap();
        
        let loaded_embedding = engine.load_embedding(&test_file).await.unwrap();
        assert_eq!(embedding, loaded_embedding);
    }

    #[tokio::test]
    async fn test_file_hash_detection() {
        let temp_dir = tempdir().unwrap();
        let engine = EmbeddingEngine::new(temp_dir.path().to_path_buf()).unwrap();
        
        let test_file = temp_dir.path().join("test.rs");
        fs::write(&test_file, "fn main() { println!(\"Hello, world!\"); }").unwrap();
        
        assert!(engine.is_embedding_current(&test_file).unwrap() == false);
        
        let embedding = engine.embed_file(&test_file).await.unwrap();
        engine.save_embedding(&test_file, embedding).await.unwrap();
        
        assert!(engine.is_embedding_current(&test_file).unwrap() == true);
        
        // Modify file
        fs::write(&test_file, "fn main() { println!(\"Goodbye, world!\"); }").unwrap();
        
        assert!(engine.is_embedding_current(&test_file).unwrap() == false);
    }
}

