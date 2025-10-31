//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham
//!
//! Conversation Log Storage - Phase 2 Integration Module
//!
//! This module provides conversation storage functionality for Phase 2 pipeline integration.
//! It wraps conversation storage with query capabilities by emotion, time, and content similarity.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use niodoo_core::memory::EmotionalVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// A single conversation entry recording user input and AI response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub user_input: String,
    pub ai_response: String,
    pub emotional_vector: EmotionalVector,
    pub emotion_state: String,
    pub metadata: HashMap<String, String>,
}

impl ConversationEntry {
    /// Create a new conversation entry
    pub fn new(
        user_input: String,
        ai_response: String,
        emotional_vector: EmotionalVector,
        emotion_state: String,
    ) -> Self {
        let id = format!("conv_{}", Utc::now().timestamp_nanos_opt().unwrap_or(0));
        Self {
            id,
            timestamp: Utc::now(),
            user_input,
            ai_response,
            emotional_vector,
            emotion_state,
            metadata: HashMap::new(),
        }
    }

    /// Calculate content similarity between two entries using simple string similarity
    pub fn content_similarity(&self, other: &Self) -> f32 {
        let self_text = format!("{} {}", self.user_input, self.ai_response);
        let other_text = format!("{} {}", other.user_input, other.ai_response);
        simple_text_similarity(&self_text, &other_text)
    }
}

/// Conversation log store for Phase 2 pipeline integration
pub struct ConversationLogStore {
    entries: Vec<ConversationEntry>,
    storage_path: PathBuf,
    auto_save_interval: usize,
}

impl ConversationLogStore {
    /// Create a new conversation log store
    pub fn new(storage_path: impl AsRef<Path>) -> Self {
        let path = PathBuf::from(storage_path.as_ref());
        Self {
            entries: Vec::new(),
            storage_path: path,
            auto_save_interval: 10,
        }
    }

    /// Create with custom auto-save interval
    pub fn with_auto_save_interval(
        storage_path: impl AsRef<Path>,
        auto_save_interval: usize,
    ) -> Self {
        let mut store = Self::new(storage_path);
        store.auto_save_interval = auto_save_interval;
        store
    }

    /// Load existing conversations from disk
    pub fn load(&mut self) -> Result<()> {
        if !self.storage_path.exists() {
            std::fs::create_dir_all(self.storage_path.parent().unwrap_or(Path::new(".")))
                .context("Failed to create storage directory")?;
            return Ok(());
        }

        let file = File::open(&self.storage_path).context("Failed to open storage file")?;
        let reader = BufReader::new(file);
        
        match serde_json::from_reader::<_, Vec<ConversationEntry>>(reader) {
            Ok(entries) => {
                self.entries = entries;
                info!("Loaded {} conversation entries from {:?}", self.entries.len(), self.storage_path);
                Ok(())
            }
            Err(e) => {
                warn!("Failed to parse conversation log: {}. Starting fresh.", e);
                Ok(())
            }
        }
    }

    /// Store a new conversation entry
    pub fn store(&mut self, entry: ConversationEntry) -> Result<()> {
        self.entries.push(entry.clone());
        
        // Auto-save periodically
        if self.entries.len() % self.auto_save_interval == 0 {
            self.save().context("Failed to auto-save conversation log")?;
        }
        
        debug!("Stored conversation entry: {}", entry.id);
        Ok(())
    }

    /// Save all entries to disk
    pub fn save(&self) -> Result<()> {
        std::fs::create_dir_all(self.storage_path.parent().unwrap_or(Path::new(".")))
            .context("Failed to create storage directory")?;
            
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.storage_path)
            .context("Failed to open storage file for writing")?;
            
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.entries)
            .context("Failed to serialize conversation log")?;
            
        debug!("Saved {} conversation entries to {:?}", self.entries.len(), self.storage_path);
        Ok(())
    }

    /// Query conversations by emotion similarity
    pub fn query_by_emotion(
        &self,
        query_emotion: &EmotionalVector,
        threshold: f32,
        limit: usize,
    ) -> Vec<&ConversationEntry> {
        let mut results: Vec<(&ConversationEntry, f32)> = self
            .entries
            .iter()
            .map(|entry| {
                let similarity = emotional_similarity(query_emotion, &entry.emotional_vector);
                (entry, similarity)
            })
            .filter(|(_, similarity)| *similarity >= threshold)
            .collect();
            
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(limit).map(|(entry, _)| entry).collect()
    }

    /// Query conversations by time range
    pub fn query_by_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<&ConversationEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.timestamp >= start && entry.timestamp <= end)
            .collect()
    }

    /// Query conversations by content similarity
    pub fn query_by_content(
        &self,
        query_text: &str,
        threshold: f32,
        limit: usize,
    ) -> Vec<&ConversationEntry> {
        let mut results: Vec<(&ConversationEntry, f32)> = self
            .entries
            .iter()
            .map(|entry| {
                let similarity = simple_text_similarity(
                    query_text,
                    &format!("{} {}", entry.user_input, entry.ai_response),
                );
                (entry, similarity)
            })
            .filter(|(_, similarity)| *similarity >= threshold)
            .collect();
            
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(limit).map(|(entry, _)| entry).collect()
    }

    /// Get all entries
    pub fn all_entries(&self) -> &[ConversationEntry] {
        &self.entries
    }

    /// Get entry count
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Clear all entries (use with caution)
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Calculate emotional similarity between two emotional vectors
fn emotional_similarity(a: &EmotionalVector, b: &EmotionalVector) -> f32 {
    let dot_product = a.joy * b.joy
        + a.sadness * b.sadness
        + a.anger * b.anger
        + a.fear * b.fear
        + a.surprise * b.surprise;
    
    let mag_a = a.magnitude();
    let mag_b = b.magnitude();
    
    if mag_a > 0.0 && mag_b > 0.0 {
        (dot_product / (mag_a * mag_b)).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Simple text similarity using Jaccard similarity on character n-grams
fn simple_text_similarity(a: &str, b: &str) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let ngram_size = 3;
    let a_grams: std::collections::HashSet<_> = a
        .chars()
        .collect::<Vec<_>>()
        .windows(ngram_size)
        .map(|w| w.iter().collect::<String>())
        .collect();
    let b_grams: std::collections::HashSet<_> = b
        .chars()
        .collect::<Vec<_>>()
        .windows(ngram_size)
        .map(|w| w.iter().collect::<String>())
        .collect();

    let intersection = a_grams.intersection(&b_grams).count();
    let union = a_grams.union(&b_grams).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_entry_creation() {
        let entry = ConversationEntry::new(
            "Hello".to_string(),
            "Hi there!".to_string(),
            EmotionalVector::new(0.8, 0.1, 0.0, 0.0, 0.1),
            "joyful".to_string(),
        );
        assert_eq!(entry.user_input, "Hello");
        assert_eq!(entry.ai_response, "Hi there!");
    }

    #[test]
    fn test_conversation_store() {
        let temp_dir = std::env::temp_dir();
        let store_path = temp_dir.join("test_conversations.json");
        let mut store = ConversationLogStore::new(&store_path);
        
        let entry = ConversationEntry::new(
            "Test".to_string(),
            "Response".to_string(),
            EmotionalVector::new(0.5, 0.5, 0.0, 0.0, 0.0),
            "neutral".to_string(),
        );
        
        assert!(store.store(entry).is_ok());
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn test_emotional_similarity() {
        let a = EmotionalVector::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let b = EmotionalVector::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let similarity = emotional_similarity(&a, &b);
        assert!(similarity > 0.99);
    }
}

