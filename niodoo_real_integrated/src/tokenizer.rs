use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::SystemTime;

use anyhow::{anyhow, Context, Result};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;
use tracing::instrument;

use crate::erag::CollapseResult;
use crate::torus::PadGhostState;

#[derive(Debug, Clone)]
pub struct PromotedToken {
    pub token_id: u32,
    pub bytes: Vec<u8>,
    pub embedding: Vec<f32>,
    pub promotion_score: f64,
    pub introduced_at: SystemTime,
}

#[derive(Debug, Clone)]
pub struct TokenizerOutput {
    pub tokens: Vec<u32>,
    pub augmented_prompt: String,
    pub promoted_tokens: Vec<PromotedToken>,
    pub vocab_size: usize,
    pub oov_rate: f64,
}

pub struct TokenizerEngine {
    vocab: HashMap<String, u32>,
    inverse_vocab: HashMap<u32, String>,
    next_token_id: u32,
    mirage_sigma: f64,
}

#[derive(Debug, Deserialize)]
struct TokenizerModelSpec {
    vocab: HashMap<String, u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerSpec {
    model: TokenizerModelSpec,
}

impl TokenizerEngine {
    pub fn new(tokenizer_path: impl AsRef<Path>, mirage_sigma: f64) -> Result<Self> {
        let path = tokenizer_path.as_ref();
        let file = File::open(path).with_context(|| {
            format!(
                "failed to open tokenizer specification at {}",
                path.display()
            )
        })?;
        let reader = BufReader::new(file);
        let spec: TokenizerSpec = serde_json::from_reader(reader).with_context(|| {
            format!(
                "failed to parse tokenizer specification at {}",
                path.display()
            )
        })?;

        let vocab = spec.model.vocab;
        if vocab.is_empty() {
            return Err(anyhow!("tokenizer vocabulary is empty"));
        }
        let mut inverse = HashMap::new();
        let mut max_id = 0u32;
        for (token, id) in &vocab {
            inverse.insert(*id, token.clone());
            max_id = max_id.max(*id);
        }

        Ok(Self {
            vocab,
            inverse_vocab: inverse,
            next_token_id: max_id + 1,
            mirage_sigma,
        })
    }

    #[instrument(skip(self, collapse))]
    pub fn process(
        &mut self,
        prompt: &str,
        collapse: &CollapseResult,
        pad_state: &PadGhostState,
        entropy_mean: f64,
    ) -> Result<TokenizerOutput> {
        let augmented_prompt = build_augmented_prompt(prompt, collapse);

        let mut promoted_tokens = Vec::new();
        for (word, count) in discover_promotable_candidates(&augmented_prompt) {
            if self.vocab.contains_key(&word) {
                continue;
            }
            let token_id = self.next_token_id;
            self.next_token_id += 1;
            self.vocab.insert(word.clone(), token_id);
            self.inverse_vocab.insert(token_id, word.clone());

            let token = PromotedToken {
                token_id,
                bytes: word.clone().into_bytes(),
                embedding: Vec::new(),
                promotion_score: count as f64,
                introduced_at: SystemTime::now(),
            };
            promoted_tokens.push(token);
        }

        let (mut tokens, oov_count) = self.encode(&augmented_prompt);
        apply_rut_mirage(pad_state, entropy_mean, &mut tokens, self.mirage_sigma)?;

        let vocab_size = self.vocab.len();
        let oov_rate = if tokens.is_empty() {
            0.0
        } else {
            oov_count as f64 / tokens.len() as f64
        };

        Ok(TokenizerOutput {
            tokens,
            augmented_prompt,
            promoted_tokens,
            vocab_size,
            oov_rate,
        })
    }

    fn encode(&mut self, text: &str) -> (Vec<u32>, usize) {
        let mut tokens = Vec::new();
        let mut oov_count = 0usize;

        for word in tokenize(text) {
            if let Some(&id) = self.vocab.get(word) {
                tokens.push(id);
            } else {
                let id = self.next_token_id;
                self.next_token_id += 1;
                self.vocab.insert(word.to_string(), id);
                self.inverse_vocab.insert(id, word.to_string());
                tokens.push(id);
                oov_count += 1;
            }
        }

        (tokens, oov_count)
    }
}

fn tokenize(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .filter(|token| !token.is_empty())
}

fn build_augmented_prompt(prompt: &str, collapse: &CollapseResult) -> String {
    let prompt_snip = snippet(prompt, 80);

    let memory_lines: Vec<String> = collapse
        .top_hits
        .iter()
        .take(2)
        .map(|hit| {
            format!(
                "- {} (dH {:.2}->{:.2})",
                snippet(&hit.output, 35),
                hit.entropy_before,
                hit.entropy_after
            )
        })
        .collect();

    let memory_section = if memory_lines.is_empty() {
        "- none".to_string()
    } else {
        memory_lines.join("\n")
    };

    let context_section = snippet(&collapse.aggregated_context, 30);

    format!(
        "Prompt: {}\nMemory: {}\nContext: {}",
        prompt_snip, memory_section, context_section
    )
}

fn discover_promotable_candidates(prompt: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for token in tokenize(prompt) {
        let entry = counts.entry(token.to_lowercase()).or_insert(0);
        *entry += 1;
    }
    counts
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .collect()
}

fn snippet(text: &str, limit: usize) -> String {
    if text.is_empty() {
        return "∅".to_string();
    }
    let mut out = String::with_capacity(limit + 1);
    let mut count = 0;
    for ch in text.chars() {
        if count >= limit {
            out.push('…');
            break;
        }
        out.push(ch);
        count += 1;
    }
    out.trim().to_string()
}

fn apply_rut_mirage(
    pad_state: &PadGhostState,
    entropy_mean: f64,
    tokens: &mut [u32],
    mirage_sigma: f64,
) -> Result<()> {
    if tokens.is_empty() {
        return Ok(());
    }

    let mut rng = thread_rng();
    let normal = Normal::new(entropy_mean, mirage_sigma.max(1e-3))?;
    let jitter = normal.sample(&mut rng);
    let shift = ((pad_state.entropy - jitter) * 7.0).round() as i64;

    for token in tokens.iter_mut() {
        let new_val = (*token as i64 + shift).max(0) as u32;
        *token = new_val;
    }

    Ok(())
}
