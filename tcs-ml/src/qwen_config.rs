// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use serde::{Deserialize, Serialize};

/// Configuration for Qwen model architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// Maximum sequence length supported
    pub max_seq_len: usize,
    /// Maximum number of cached tokens to retain after each step
    #[serde(default = "default_cache_window")]
    pub cache_window: usize,
    /// Embedding dimension for TCS pipeline
    pub embed_dim: usize,
    /// Vocabulary size for logits extraction
    pub vocab_size: usize,
}

fn default_cache_window() -> usize {
    2048
}

impl QwenConfig {
    /// Qwen2.5-Coder 0.5B configuration
    pub fn qwen25_coder_05b() -> Self {
        Self {
            num_layers: 24,
            num_heads: 2, // Simplified for 0.5B model
            head_dim: 64,
            max_seq_len: 2048,
            cache_window: 2048,
            embed_dim: 512,
            vocab_size: 151936,
        }
    }

    /// Qwen2.5-Coder 7B configuration (for future use)
    pub fn qwen25_coder_7b() -> Self {
        Self {
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            max_seq_len: 4096,
            cache_window: 2048,
            embed_dim: 512,
            vocab_size: 151936,
        }
    }

    /// Load configuration from TOML file
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: QwenConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Validate configuration values
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.num_layers == 0 {
            return Err(anyhow::anyhow!("num_layers must be > 0"));
        }
        if self.num_heads == 0 {
            return Err(anyhow::anyhow!("num_heads must be > 0"));
        }
        if self.head_dim == 0 {
            return Err(anyhow::anyhow!("head_dim must be > 0"));
        }
        if self.max_seq_len == 0 {
            return Err(anyhow::anyhow!("max_seq_len must be > 0"));
        }
        if self.cache_window == 0 {
            return Err(anyhow::anyhow!("cache_window must be > 0"));
        }
        if self.cache_window > self.max_seq_len {
            return Err(anyhow::anyhow!(
                "cache_window ({}) cannot exceed max_seq_len ({})",
                self.cache_window,
                self.max_seq_len
            ));
        }
        if self.embed_dim == 0 {
            return Err(anyhow::anyhow!("embed_dim must be > 0"));
        }
        if self.vocab_size == 0 {
            return Err(anyhow::anyhow!("vocab_size must be > 0"));
        }
        Ok(())
    }
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self::qwen25_coder_05b()
    }
}
