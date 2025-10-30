// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::consciousness::EmotionType;

/// Intent types for consciousness processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    HelpRequest,
    EmotionalQuery,
    TechnicalQuery,
    CreativeQuery,
    GeneralQuery,
}

#[async_trait]
pub trait IntentAnalyzer: Send + Sync {
    async fn analyze_intent(&self, input: &str) -> Result<(Intent, f32)>; // intent, dynamic confidence
    fn calculate_complexity(&self, input: &str) -> f32;
    fn extract_keywords(&self, input: &str) -> Vec<String>;
}

#[async_trait]
pub trait ResourceOptimizer: Send + Sync {
    async fn optimize_response(&self, input: &str, complexity: f32) -> Result<String>;
    fn detect_loops(&self, input_hash: &str) -> bool; // Simple hash-based, no history for now
}

#[async_trait]
pub trait CreativeSynthesizer: Send + Sync {
    async fn synthesize_creative(&self, input: &str, emotion: &EmotionType) -> Result<String>;
}

#[async_trait]
pub trait BrainBehavior: IntentAnalyzer + ResourceOptimizer + CreativeSynthesizer {}

// Unified impl extracting from existing brains
pub struct UnifiedBehavior {
    loop_history: Arc<RwLock<HashSet<String>>>, // Thread-safe loop detection with interior mutability
}

impl UnifiedBehavior {
    pub fn new() -> Self {
        Self {
            loop_history: Arc::new(RwLock::new(HashSet::new())),
        }
    }
}

#[async_trait]
impl IntentAnalyzer for UnifiedBehavior {
    async fn analyze_intent(&self, input: &str) -> Result<(Intent, f32)> {
        let input_lower = input.to_lowercase();
        let keyword_count = if input_lower.contains("help") || input_lower.contains("how") { 1 } else { 0 }
            + if input_lower.contains("feel") || input_lower.contains("emotion") { 1 } else { 0 }
            + if input_lower.contains("code") || input_lower.contains("function") { 1 } else { 0 }
            + if input_lower.contains("create") || input_lower.contains("imagine") { 1 } else { 0 };
        let confidence = (keyword_count as f32 / 4.0).min(1.0).max(0.5); // Dynamic based on matches

        let intent = if input_lower.contains("help") || input_lower.contains("how") {
            Intent::HelpRequest
        } else if input_lower.contains("feel") || input_lower.contains("emotion") {
            Intent::EmotionalQuery
        } else if input_lower.contains("code") || input_lower.contains("function") {
            Intent::TechnicalQuery
        } else if input_lower.contains("create") || input_lower.contains("imagine") {
            Intent::CreativeQuery
        } else {
            Intent::GeneralQuery
        };

        Ok((intent, confidence))
    }

    fn calculate_complexity(&self, input: &str) -> f32 {
        let words: Vec<&str> = input.split_whitespace().collect();
        if words.is_empty() { return 0.0; }
        let unique_words = words.iter().collect::<HashSet<_>>().len();
        let avg_word_length = words.iter().map(|w| w.len()).sum::<usize>() as f32 / words.len() as f32;
        let length_factor = (input.len() as f32 / 1000.0).min(1.0);
        let vocab_factor = (unique_words as f32 / words.len() as f32).min(1.0);
        let word_length_factor = (avg_word_length / 10.0).min(1.0);
        (length_factor * 0.4 + vocab_factor * 0.3 + word_length_factor * 0.3).min(1.0)
    }

    fn extract_keywords(&self, input: &str) -> Vec<String> {
        let technical_terms = ["consciousness", "memory", "brain", "neural", "cognitive", "emotion"];
        technical_terms.iter()
            .filter(|&&term| input.to_lowercase().contains(term))
            .map(|&term| term.to_string())
            .collect()
    }
}

#[async_trait]
impl ResourceOptimizer for UnifiedBehavior {
    async fn optimize_response(&self, input: &str, complexity: f32) -> Result<String> {
        let input_hash = format!("{:x}", (input.len() * 17) as u64 + (input.chars().count() as u64 * 31u64));
        let is_loop = self.detect_loops(&input_hash);
        if is_loop {
            return Ok("Cognitive loop detected: Suggest pivoting to new perspective.".to_string());
        }

        // Update loop history with current input hash (thread-safe)
        {
            let mut history = self.loop_history.write().await;
            history.insert(input_hash.clone());
            // Keep history size bounded to prevent memory growth
            if history.len() > 1000 {
                history.clear();
            }
        }

        if complexity > 0.7 {
            Ok(format!("High complexity ({}): Recommend chunking into smaller parts.", complexity))
        } else {
            Ok("Optimal processing path selected.".to_string())
        }
    }

    fn detect_loops(&self, input_hash: &str) -> bool {
        // Real implementation: Check if input hash already exists in history
        // Use try_read() to avoid blocking if write lock is held
        if let Ok(history) = self.loop_history.try_read() {
            history.contains(input_hash)
        } else {
            // If we can't acquire read lock, assume no loop to avoid blocking
            false
        }
    }
}

#[async_trait]
impl CreativeSynthesizer for UnifiedBehavior {
    async fn synthesize_creative(&self, input: &str, emotion: &EmotionType) -> Result<String> {
        let prefix = match emotion {
            EmotionType::GpuWarm => "Warm creative pathways:",
            EmotionType::Purposeful => "Purposeful synthesis:",
            _ => "Creative exploration:",
        };
        Ok(format!("{} Imagining innovative solutions for {}.", prefix, input))
    }
}

// Blanket implementation of BrainBehavior for UnifiedBehavior.
// BrainBehavior is a marker trait that requires:
// - IntentAnalyzer (implemented above, lines 36-78)
// - ResourceOptimizer (implemented above, lines 81-116)
// - CreativeSynthesizer (implemented above, lines 119-128)
//
// Since all super-traits are implemented, we provide the blanket impl.
// No additional methods are needed as BrainBehavior defines no new methods.
#[async_trait]
impl BrainBehavior for UnifiedBehavior {
    // This is intentionally empty - BrainBehavior is a marker trait
    // that combines the three behavior traits without adding new methods.
    // All functionality comes from the super-trait implementations above.
}









