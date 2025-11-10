//! Dynamic Tokenizer for TCS Analysis
//! 
//! Provides extended tokenization with promoted tokens for better point cloud generation.
//! Based on the triple-threat CRDT tokenizer from the main codebase.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

/// Dynamic tokenizer with extended vocabulary support
#[derive(Clone)]
pub struct DynamicTokenizer {
    base_tokenizer: Tokenizer,
    extended_vocab: HashMap<Vec<u8>, u32>,
    id_to_bytes: HashMap<u32, Vec<u8>>,
    next_token_id: u32,
    token_usage: HashMap<u32, u64>,
    max_extended_length: usize,
}

impl DynamicTokenizer {
    /// Create new dynamic tokenizer from base tokenizer
    pub fn new(base_tokenizer: Tokenizer) -> Self {
        let next_token_id = base_tokenizer.get_vocab_size(false) as u32;
        Self {
            base_tokenizer,
            extended_vocab: HashMap::new(),
            id_to_bytes: HashMap::new(),
            next_token_id,
            token_usage: HashMap::new(),
            max_extended_length: 20,
        }
    }

    /// Load tokenizer from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.as_ref())
            .map_err(|err| anyhow!("failed to load tokenizer: {err}"))?;
        Ok(Self::new(tokenizer))
    }

    /// Encode text with extended vocabulary (triple-threat: base + extended + CRDT)
    pub fn encode_extended(&mut self, text: &str) -> Result<Vec<u32>> {
        let bytes = text.as_bytes();
        let mut tokens = Vec::new();
        let mut index = 0;

        while index < bytes.len() {
            let mut matched = false;
            let start = index;
            let min_len = self.min_token_length();

            // Try extended vocab first (longest match)
            for len in (min_len..=self.max_extended_length).rev() {
                if start + len > bytes.len() {
                    continue;
                }

                let candidate = &bytes[start..start + len];
                if let Some(&token_id) = self.extended_vocab.get(candidate) {
                    tokens.push(token_id);
                    *self.token_usage.entry(token_id).or_insert(0) += 1;
                    index += len;
                    matched = true;
                    break;
                }
            }

            // Fall back to base tokenizer
            if !matched {
                let remaining = &text[index..];
                if remaining.is_empty() {
                    break;
                }

                let encoding = self
                    .base_tokenizer
                    .encode(remaining, false)
                    .map_err(|err| anyhow!("tokenizer encoding failed: {err}"))?;
                let ids = encoding.get_ids();

                if ids.is_empty() {
                    // Fallback: advance by one character to avoid infinite loop
                    let char_len = remaining
                        .chars()
                        .next()
                        .map(|ch| ch.len_utf8())
                        .unwrap_or(1);
                    let fallback_slice = &remaining[..char_len];
                    let fallback_ids = self
                        .base_tokenizer
                        .encode(fallback_slice, false)
                        .map_err(|err| anyhow!("tokenizer encoding failed: {err}"))?;
                    tokens.extend_from_slice(fallback_ids.get_ids());
                    index += char_len;
                    continue;
                }

                tokens.push(ids[0]);

                let consumed = encoding
                    .get_offsets()
                    .get(0)
                    .map(|(_, end)| *end)
                    .filter(|end| *end > 0)
                    .unwrap_or_else(|| {
                        remaining
                            .chars()
                            .next()
                            .map(|ch| ch.len_utf8())
                            .unwrap_or(1)
                    });

                index += consumed;
            }
        }

        Ok(tokens)
    }

    /// Decode token IDs back to strings
    pub fn decode_extended(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(token_bytes) = self.id_to_bytes.get(&id) {
                bytes.extend_from_slice(token_bytes);
            } else {
                let decoded = self
                    .base_tokenizer
                    .decode(&[id], false)
                    .map_err(|err| anyhow!("tokenizer decoding failed: {err}"))?;
                bytes.extend_from_slice(decoded.as_bytes());
            }
        }

        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    /// Decode single token ID to string
    pub fn decode_token(&self, id: u32) -> Result<String> {
        if let Some(token_bytes) = self.id_to_bytes.get(&id) {
            Ok(String::from_utf8_lossy(token_bytes).to_string())
        } else {
            self.base_tokenizer
                .decode(&[id], false)
                .map_err(|err| anyhow!("tokenizer decoding failed: {err}"))
        }
    }

    /// Add a promoted token to extended vocabulary
    pub fn add_token(&mut self, bytes: Vec<u8>, token_id: u32) {
        if !self.extended_vocab.contains_key(&bytes) {
            self.extended_vocab.insert(bytes.clone(), token_id);
            self.id_to_bytes.insert(token_id, bytes);
            self.token_usage.insert(token_id, 0);
            self.next_token_id = self.next_token_id.max(token_id + 1);
        }
    }

    fn min_token_length(&self) -> usize {
        4
    }
}

