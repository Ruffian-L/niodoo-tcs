//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Async File Logger for Telemetry
//!
//! Provides async JSONL file logging with rotation and compression support.

use crate::telemetry::EnhancedCognitiveStatePacket;
use anyhow::{Context, Result};
use serde_json;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::Mutex;
use tracing::{error, warn};

/// Configuration for file logger
#[derive(Debug, Clone)]
pub struct FileLoggerConfig {
    /// Base directory for log files
    pub log_dir: PathBuf,
    /// Maximum file size in bytes before rotation
    pub max_file_size: u64,
    /// Maximum number of rotated files to keep
    pub max_files: usize,
    /// Whether to compress rotated files
    pub compress: bool,
    /// Log file prefix
    pub file_prefix: String,
}

impl Default for FileLoggerConfig {
    fn default() -> Self {
        Self {
            log_dir: PathBuf::from("logs"),
            max_file_size: 100 * 1024 * 1024, // 100 MB
            max_files: 10,
            compress: true,
            file_prefix: "telemetry".to_string(),
        }
    }
}

/// Async file logger for telemetry packets
pub struct FileLogger {
    config: FileLoggerConfig,
    current_file: Arc<Mutex<Option<BufWriter<File>>>>,
    current_size: Arc<Mutex<u64>>,
    file_counter: Arc<Mutex<usize>>,
}

impl FileLogger {
    /// Create a new file logger
    pub async fn new(config: FileLoggerConfig) -> Result<Self> {
        // Ensure log directory exists
        tokio::fs::create_dir_all(&config.log_dir)
            .await
            .with_context(|| format!("Failed to create log directory: {:?}", config.log_dir))?;

        let logger = Self {
            config,
            current_file: Arc::new(Mutex::new(None)),
            current_size: Arc::new(Mutex::new(0)),
            file_counter: Arc::new(Mutex::new(0)),
        };

        // Open initial log file
        logger.rotate_file().await?;

        Ok(logger)
    }

    /// Get the current log file path
    fn current_file_path(&self) -> PathBuf {
        let counter = *self.file_counter.blocking_lock();
        if counter == 0 {
            self.config
                .log_dir
                .join(format!("{}.jsonl", self.config.file_prefix))
        } else {
            self.config
                .log_dir
                .join(format!("{}_{}.jsonl", self.config.file_prefix, counter))
        }
    }

    /// Rotate to a new log file
    async fn rotate_file(&self) -> Result<()> {
        // Close current file if open
        if let Some(mut writer) = self.current_file.lock().await.take() {
            writer.flush().await?;
            // Drop writer to close file
        }

        // Increment file counter
        let counter = {
            let mut c = self.file_counter.lock().await;
            *c += 1;
            *c
        };

        // Open new file
        let path = if counter == 1 {
            self.config
                .log_dir
                .join(format!("{}.jsonl", self.config.file_prefix))
        } else {
            self.config
                .log_dir
                .join(format!("{}_{}.jsonl", self.config.file_prefix, counter))
        };

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .with_context(|| format!("Failed to open log file: {:?}", path))?;

        let writer = BufWriter::new(file);
        *self.current_file.lock().await = Some(writer);
        *self.current_size.lock().await = 0;

        // Clean up old files if needed
        let _ = self.cleanup_old_files().await;

        Ok(())
    }

    /// Clean up old log files
    async fn clean_old_files(&self) -> Result<()> {
        let mut entries = Vec::new();
        let mut read_dir = tokio::fs::read_dir(&self.config.log_dir).await?;

        loop {
            let entry = match read_dir.next_entry().await? {
                Some(entry) => entry,
                None => break,
            };
            let path = entry.path();
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(&self.config.file_prefix))
                .unwrap_or(false)
            {
                if let Ok(metadata) = entry.metadata().await {
                    if let Ok(modified) = metadata.modified() {
                        entries.push((path, modified, metadata.len()));
                    }
                }
            }
        }

        // Sort by modification time (oldest first)
        entries.sort_by_key(|(_, modified, _)| *modified);

        // Remove oldest files if we exceed max_files
        if entries.len() > self.config.max_files {
            let to_remove = entries.len() - self.config.max_files;
            for (path, _, _) in entries.iter().take(to_remove) {
                if let Err(e) = tokio::fs::remove_file(path).await {
                    warn!(path = ?path, error = %e, "Failed to remove old log file");
                }
            }
        }

        Ok(())
    }

    /// Cleanup old files (wrapper to handle errors)
    async fn cleanup_old_files(&self) {
        if let Err(e) = self.clean_old_files().await {
            warn!(error = %e, "Failed to cleanup old log files");
        }
    }

    /// Log a telemetry packet
    pub async fn log(&self, packet: &EnhancedCognitiveStatePacket) -> Result<()> {
        let json = serde_json::to_string(packet).context("Failed to serialize packet to JSON")?;
        let line = format!("{}\n", json);
        let line_size = line.len() as u64;

        // Check if we need to rotate
        let mut current_size = self.current_size.lock().await;
        if *current_size + line_size > self.config.max_file_size {
            drop(current_size);
            self.rotate_file().await?;
            current_size = self.current_size.lock().await;
        }

        // Write to current file
        if let Some(ref mut writer) = *self.current_file.lock().await {
            writer.write_all(line.as_bytes()).await?;
            *current_size += line_size;
        } else {
            // File not open, try to open it
            self.rotate_file().await?;
            if let Some(ref mut writer) = *self.current_file.lock().await {
                writer.write_all(line.as_bytes()).await?;
                *current_size += line_size;
            }
        }

        Ok(())
    }

    /// Flush any pending writes
    pub async fn flush(&self) -> Result<()> {
        if let Some(ref mut writer) = *self.current_file.lock().await {
            writer.flush().await?;
        }
        Ok(())
    }
}
