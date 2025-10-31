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
pub fn rouge_l(candidate: &str, reference: &str) -> f64 {
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

/// Seed manager for deterministic RNG
pub struct SeedManager {
    global_seed: u64,
}

impl SeedManager {
    pub fn new() -> Self {
        Self {
            global_seed: std::env::var("RNG_SEED")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(42),
        }
    }

    pub fn get_rng(&self, scope: &str) -> rand::rngs::StdRng {
        use rand::SeedableRng;
        let seed = self.global_seed.wrapping_add(scope.len() as u64);
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    /// Get master seed
    pub fn master_seed(&self) -> u64 {
        self.global_seed
    }
}

static SEED_MANAGER: once_cell::sync::Lazy<std::sync::Mutex<SeedManager>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(SeedManager::new()));

/// Get the global seed manager
pub fn seed_manager() -> std::sync::MutexGuard<'static, SeedManager> {
    SEED_MANAGER.lock().unwrap()
}

/// Set global seed for deterministic RNG
pub fn set_global_seed(seed: u64) {
    let mut manager = SEED_MANAGER.lock().unwrap();
    manager.global_seed = seed;
}

/// Compute entropy from log probabilities
pub fn entropy_from_logprobs(logprobs: &[f64]) -> f64 {
    // Convert logprobs to probabilities
    let probs: Vec<f64> = logprobs.iter().map(|&lp| lp.exp()).collect();
    shannon_entropy(&probs)
}
