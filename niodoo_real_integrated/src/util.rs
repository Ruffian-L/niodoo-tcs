use blake3::hash as blake3_hash;
use once_cell::sync::OnceCell;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashSet;

/// Compute Shannon entropy (base e) for a slice of probabilities.
pub fn shannon_entropy(probs: &[f64]) -> f64 {
    let mut entropy = 0.0;
    for &p in probs {
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Compute entropy from logprobs: -sum(p * ln(p)) where p = exp(logprob)/Z
pub fn entropy_from_logprobs(logprobs: &[f64]) -> f64 {
    if logprobs.is_empty() {
        return 0.0;
    }

    // Normalize logprobs by subtracting max for numerical stability
    let max_logprob = logprobs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let probs: Vec<f64> = logprobs
        .iter()
        .map(|&lp| (lp - max_logprob).exp())
        .collect();
    let z: f64 = probs.iter().sum();

    if z == 0.0 {
        return 0.0;
    }

    let normalized_probs: Vec<f64> = probs.iter().map(|&p| p / z).collect();
    shannon_entropy(&normalized_probs)
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (va, vb) in a.iter().zip(b.iter()) {
        let da = *va as f64;
        let db = *vb as f64;
        dot += da * db;
        norm_a += da * da;
        norm_b += db * db;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
}

/// ROUGE-L score between two strings.
/// Native Rust implementation.
pub fn rouge_l(candidate: &str, reference: &str) -> f64 {
    rouge_l_native(candidate, reference)
}

/// Native Rust implementation of ROUGE-L
fn rouge_l_native(candidate: &str, reference: &str) -> f64 {
    let cand_tokens: Vec<&str> = candidate.split_whitespace().collect();
    let ref_tokens: Vec<&str> = reference.split_whitespace().collect();

    if cand_tokens.is_empty() || ref_tokens.is_empty() {
        return 0.0;
    }

    let lcs = lcs_length(&cand_tokens, &ref_tokens) as f64;
    let recall = lcs / ref_tokens.len() as f64;
    let precision = lcs / cand_tokens.len() as f64;

    if precision + recall == 0.0 {
        return 0.0;
    }

    let beta = recall / (precision + 1e-9);
    ((1.0 + beta * beta) * precision * recall) / (recall + beta * beta * precision + 1e-9)
}

fn lcs_length(a: &[&str], b: &[&str]) -> usize {
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            if ai == bj {
                dp[i + 1][j + 1] = dp[i][j] + 1;
            } else {
                dp[i + 1][j + 1] = dp[i + 1][j].max(dp[i][j + 1]);
            }
        }
    }
    dp[a.len()][b.len()]
}

/// Returns unique tokens from text preserving insertion order.
pub fn unique_tokens(text: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for token in text.split_whitespace() {
        if seen.insert(token) {
            result.push(token.to_string());
        }
    }
    result
}

/// Seed manager for deterministic RNG across the integrated runtime
#[derive(Debug, Clone, Copy)]
pub struct SeedManager {
    master_seed: u64,
}

impl SeedManager {
    /// Create a new seed manager from a master seed
    pub fn new(master_seed: u64) -> Self {
        Self { master_seed }
    }

    /// Expose the configured master seed
    pub fn master_seed(&self) -> u64 {
        self.master_seed
    }

    /// Initialize from environment variables. Falls back to 42 if unset/invalid.
    /// Recognized vars: NIODOO_SEED, RNG_SEED
    pub fn from_env() -> Self {
        let seed = std::env::var("NIODOO_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .or_else(|| {
                std::env::var("RNG_SEED")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
            })
            .unwrap_or(42);
        Self::new(seed)
    }

    /// Derive a deterministic StdRng for a given scope string
    pub fn get_rng(&self, scope: &str) -> StdRng {
        let digest = blake3_hash(scope.as_bytes());
        let mut scope_seed_bytes = [0u8; 8];
        scope_seed_bytes.copy_from_slice(&digest.as_bytes()[..8]);
        let scope_seed = u64::from_le_bytes(scope_seed_bytes);
        StdRng::seed_from_u64(self.master_seed ^ scope_seed)
    }
}

static GLOBAL_SEED_MANAGER: OnceCell<SeedManager> = OnceCell::new();

/// Global access to the process-level seed manager, initialized from env
pub fn seed_manager() -> &'static SeedManager {
    GLOBAL_SEED_MANAGER.get_or_init(SeedManager::from_env)
}

/// Override the global seed manager with a specific seed. Subsequent calls to
/// `seed_manager()` will return a manager initialised with this seed. If the
/// seed manager was already initialised with a different seed, the override is
/// ignored to preserve determinism across threads.
pub fn set_global_seed(seed: u64) {
    if GLOBAL_SEED_MANAGER.set(SeedManager::new(seed)).is_err() {
        if let Some(existing) = GLOBAL_SEED_MANAGER.get() {
            if existing.master_seed() != seed {
                // Intentionally silent: callers decide how to handle conflicting seeds.
            }
        }
    }
}
