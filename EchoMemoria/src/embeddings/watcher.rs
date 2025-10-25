//! File watcher service for real-time embedding updates
//! 
//! This module monitors file changes and automatically triggers embedding updates
//! with MCP notifications to keep the AI transformer's semantic map current.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use anyhow::Result;
use notify::{RecommendedWatcher, Watcher, RecursiveMode, Event, EventKind};
use tokio::sync::mpsc;
use tokio::time::{sleep, Instant};
use rayon::prelude::*;
use glob::Pattern;
use log::error;

use super::{EmbeddingEngine, McpNotifier, EmbeddingStats};

/// File watcher for automatic embedding updates
pub struct CodeWatcher {
    engine: Arc<EmbeddingEngine>,
    watcher: RecommendedWatcher,
    patterns: Vec<Pattern>,
    ignore_patterns: Vec<Pattern>,
    debounce_ms: u64,
    batch_size: usize,
    is_running: Arc<tokio::sync::RwLock<bool>>,
    stats: Arc<tokio::sync::RwLock<EmbeddingStats>>,
}

/// File change event for processing
#[derive(Debug, Clone)]
pub struct FileChangeEvent {
    pub path: PathBuf,
    pub kind: EventKind,
    pub timestamp: Instant,
}

impl CodeWatcher {
    /// Create a new code watcher
    pub fn new(
        root: PathBuf,
        engine: Arc<EmbeddingEngine>,
        patterns: Vec<String>,
        ignore_patterns: Vec<String>,
        debounce_ms: u64,
        batch_size: usize,
    ) -> Result<Self> {
        // Convert string patterns to glob patterns
        let patterns: Result<Vec<Pattern>> = patterns
            .into_iter()
            .map(|p| Pattern::new(&p).map_err(|e| anyhow::anyhow!("Invalid pattern {}: {}", p, e)))
            .collect();
        let patterns = patterns?;

        let ignore_patterns: Result<Vec<Pattern>> = ignore_patterns
            .into_iter()
            .map(|p| Pattern::new(&p).map_err(|e| anyhow::anyhow!("Invalid ignore pattern {}: {}", p, e)))
            .collect();
        let ignore_patterns = ignore_patterns?;

        // Create watcher
        let (tx, rx) = mpsc::channel(1000);
        let watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            if let Ok(event) = res {
                let _ = tx.try_send(event);
            }
        })?;

        Ok(Self {
            engine,
            watcher,
            patterns,
            ignore_patterns,
            debounce_ms,
            batch_size,
            is_running: Arc::new(tokio::sync::RwLock::new(false)),
            stats: Arc::new(tokio::sync::RwLock::new(EmbeddingStats {
                total_files: 0,
                embedded_files: 0,
                stale_embeddings: 0,
                avg_embed_time_ms: 0.0,
                last_sync: 0,
                cache_hits: 0,
                cache_misses: 0,
            })),
        })
    }

    /// Start watching for file changes
    pub async fn start(&mut self, root: PathBuf) -> Result<()> {
        // Watch the root directory
        self.watcher.watch(&root, RecursiveMode::Recursive)?;
        
        // Set running flag
        {
            let mut running = self.is_running.write().await;
            *running = true;
        }

        // Start the event processing loop
        self.process_events().await?;

        Ok(())
    }

    /// Stop watching
    pub async fn stop(&mut self) -> Result<()> {
        {
            let mut running = self.is_running.write().await;
            *running = false;
        }
        
        self.watcher.unwatch(Path::new("."))?;
        Ok(())
    }

    /// Process file change events with debouncing
    async fn process_events(&self) -> Result<()> {
        let mut pending_events: std::collections::HashMap<PathBuf, FileChangeEvent> = std::collections::HashMap::new();
        let mut last_process = Instant::now();

        loop {
            // Check if we should stop
            {
                let running = self.is_running.read().await;
                if !*running {
                    break;
                }
            }

            // Process pending events if debounce time has passed
            if last_process.elapsed() >= Duration::from_millis(self.debounce_ms) {
                if !pending_events.is_empty() {
                    let events: Vec<FileChangeEvent> = pending_events.values().cloned().collect();
                    pending_events.clear();
                    
                    self.process_batch(events).await?;
                    last_process = Instant::now();
                }
            }

            // Small delay to prevent busy waiting
            sleep(Duration::from_millis(100)).await;
        }

        Ok(())
    }

    /// Process a batch of file change events
    async fn process_batch(&self, events: Vec<FileChangeEvent>) -> Result<()> {
        let mut to_embed = Vec::new();

        for event in events {
            if self.should_process_file(&event.path) {
                match event.kind {
                    EventKind::Create(_) | EventKind::Modify(_) => {
                        to_embed.push(event.path);
                    }
                    EventKind::Remove(_) => {
                        // Handle file deletion - remove embedding
                        self.remove_embedding(&event.path).await?;
                    }
                    _ => {} // Ignore other event types
                }
            }
        }

        if !to_embed.is_empty() {
            self.embed_files_batch(to_embed).await?;
        }

        Ok(())
    }

    /// Check if a file should be processed
    fn should_process_file(&self, path: &Path) -> bool {
        // Check if file matches any pattern
        let matches_pattern = self.patterns.iter().any(|pattern| {
            pattern.matches_path(path)
        });

        if !matches_pattern {
            return false;
        }

        // Check if file should be ignored
        let should_ignore = self.ignore_patterns.iter().any(|pattern| {
            pattern.matches_path(path)
        });

        !should_ignore
    }

    /// Embed a batch of files in parallel
    async fn embed_files_batch(&self, paths: Vec<PathBuf>) -> Result<()> {
        let start_time = Instant::now();
        let mut success_count = 0;
        let mut error_count = 0;

        // Process files in parallel batches
        for chunk in paths.chunks(self.batch_size) {
            let results: Vec<Result<()>> = chunk
                .par_iter()
                .map(|path| {
                    tokio::runtime::Handle::current().block_on(async {
                        self.embed_single_file(path).await
                    })
                })
                .collect();

            for result in results {
                match result {
                    Ok(_) => success_count += 1,
                    Err(e) => {
                        error_count += 1;
                        error!("Failed to embed file: {}", e);
                    }
                }
            }
        }

        let total_time = start_time.elapsed().as_millis() as f64;
        let avg_time = total_time / (success_count + error_count) as f64;

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.avg_embed_time_ms = avg_time;
            stats.last_sync = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }

        // Notify MCP server about batch completion
        if let Some(notifier) = &self.engine.mcp_notifier {
            let total_files = success_count + error_count;
            notifier.notify_batch_complete(total_files, success_count).await.ok();
        }

        Ok(())
    }

    /// Embed a single file
    async fn embed_single_file(&self, path: &Path) -> Result<()> {
        // Check if file exists and is readable
        if !path.exists() {
            return Err(anyhow::anyhow!("File does not exist: {:?}", path));
        }

        // Check if embedding is current
        if self.engine.is_embedding_current(path)? {
            return Ok(()); // Skip if up-to-date
        }

        // Generate embedding
        let embedding = self.engine.embed_file(path).await?;
        
        // Save to sidecar file
        self.engine.save_embedding(path, embedding).await?;

        Ok(())
    }

    /// Remove embedding for deleted file
    async fn remove_embedding(&self, path: &Path) -> Result<()> {
        let sidecar_path = self.get_sidecar_path(path);
        
        if sidecar_path.exists() {
            std::fs::remove_file(&sidecar_path)?;
        }

        Ok(())
    }

    /// Get sidecar file path
    fn get_sidecar_path(&self, path: &Path) -> PathBuf {
        let mut sidecar_path = path.to_path_buf();
        sidecar_path.set_extension("embedding");
        sidecar_path
    }

    /// Batch embed entire codebase
    pub async fn batch_embed_all(&self, root: PathBuf) -> Result<()> {
        let mut total_files = 0;
        let mut success_count = 0;

        // Walk directory tree
        let mut entries = Vec::new();
        self.collect_files(&root, &mut entries)?;

        total_files = entries.len();

        // Process files in parallel batches
        for chunk in entries.chunks(self.batch_size) {
            let results: Vec<Result<()>> = chunk
                .par_iter()
                .map(|path| {
                    tokio::runtime::Handle::current().block_on(async {
                        self.embed_single_file(path).await
                    })
                })
                .collect();

            for result in results {
                match result {
                    Ok(_) => success_count += 1,
                    Err(e) => {
                        error!("Failed to embed file: {}", e);
                    }
                }
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_files = total_files;
            stats.embedded_files = success_count;
            stats.last_sync = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }

        // Notify MCP server
        if let Some(notifier) = &self.engine.mcp_notifier {
            notifier.notify_batch_complete(total_files, success_count).await.ok();
        }

        Ok(())
    }

    /// Collect files matching patterns
    fn collect_files(&self, root: &Path, entries: &mut Vec<PathBuf>) -> Result<()> {
        if !root.is_dir() {
            return Ok(());
        }

        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                // Skip ignored directories
                if self.ignore_patterns.iter().any(|pattern| pattern.matches_path(&path)) {
                    continue;
                }
                self.collect_files(&path, entries)?;
            } else if path.is_file() {
                // Check if file matches patterns
                if self.patterns.iter().any(|pattern| pattern.matches_path(&path)) {
                    entries.push(path);
                }
            }
        }

        Ok(())
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> EmbeddingStats {
        self.stats.read().await.clone()
    }

    /// Check if watcher is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs;

    #[tokio::test]
    async fn test_watcher_creation() {
        let temp_dir = tempdir().unwrap();
        let engine = Arc::new(EmbeddingEngine::new(temp_dir.path().to_path_buf()).unwrap());
        
        let watcher = CodeWatcher::new(
            temp_dir.path().to_path_buf(),
            engine,
            vec!["*.rs".to_string()],
            vec!["target/".to_string()],
            1000,
            10,
        ).unwrap();

        assert!(!watcher.is_running().await);
    }

    #[tokio::test]
    async fn test_file_collection() {
        let temp_dir = tempdir().unwrap();
        let engine = Arc::new(EmbeddingEngine::new(temp_dir.path().to_path_buf()).unwrap());
        
        let watcher = CodeWatcher::new(
            temp_dir.path().to_path_buf(),
            engine,
            vec!["*.rs".to_string()],
            vec!["target/".to_string()],
            1000,
            10,
        ).unwrap();

        // Create test files
        fs::write(temp_dir.path().join("test.rs"), "fn main() {}").unwrap();
        fs::write(temp_dir.path().join("test.cpp"), "int main() {}").unwrap();

        let mut entries = Vec::new();
        watcher.collect_files(temp_dir.path(), &mut entries).unwrap();

        // Should only collect .rs files
        assert_eq!(entries.len(), 1);
        assert!(entries[0].file_name().unwrap() == "test.rs");
    }
}

