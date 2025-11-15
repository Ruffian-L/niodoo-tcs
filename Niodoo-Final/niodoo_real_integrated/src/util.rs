use std::collections::HashSet;

use futures::future;

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

/// Phase 4.1: Parallel ROUGE scoring for multiple candidate-reference pairs
/// Uses tokio::join! to compute scores concurrently when enabled
pub async fn rouge_l_parallel(candidate: &str, reference: &str) -> f64 {
    // For now, use synchronous computation (rouge_l is CPU-bound and fast)
    // In the future, can spawn blocking tasks if needed for very large texts
    rouge_l(candidate, reference)
}

/// Phase 4.1: Parallel ROUGE scoring for multiple pairs
/// Returns scores in the same order as inputs
pub async fn rouge_l_batch_parallel(pairs: Vec<(&str, &str)>) -> Vec<f64> {
    if pairs.is_empty() {
        return Vec::new();
    }

    // Use tokio::join! to compute scores in parallel
    let mut futures = Vec::with_capacity(pairs.len());
    for (candidate, reference) in pairs {
        // Spawn blocking task for each ROUGE computation to avoid blocking async runtime
        let candidate = candidate.to_string();
        let reference = reference.to_string();
        futures.push(tokio::task::spawn_blocking(move || {
            rouge_l(&candidate, &reference)
        }));
    }

    // Collect all results
    let results = futures::future::join_all(futures).await;
    results
        .into_iter()
        .map(|r| r.unwrap_or(0.0)) // Fallback to 0.0 on panic
        .collect()
}

pub fn append_robust_log(scope: &str, message: &str) -> std::io::Result<()> {
    use std::fs::OpenOptions;
    use std::io::Write;

    let log_dir = std::path::Path::new("logs");
    std::fs::create_dir_all(log_dir)?;
    let log_path = log_dir.join("robust_audit.log");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    writeln!(
        file,
        "{} [{}] {}",
        chrono::Utc::now().to_rfc3339(),
        scope,
        message
    )?;
    Ok(())
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
    SEED_MANAGER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Set global seed for deterministic RNG
pub fn set_global_seed(seed: u64) {
    let mut manager = SEED_MANAGER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    manager.global_seed = seed;
}

/// Compute entropy from log probabilities
pub fn entropy_from_logprobs(logprobs: &[f64]) -> f64 {
    // Convert logprobs to probabilities
    let probs: Vec<f64> = logprobs.iter().map(|&lp| lp.exp()).collect();
    shannon_entropy(&probs)
}

/// Compute Jaccard similarity (intersection over union) between two token sets
pub fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let tokens_a: HashSet<&str> = a.split_whitespace().collect();
    let tokens_b: HashSet<&str> = b.split_whitespace().collect();

    if tokens_a.is_empty() && tokens_b.is_empty() {
        return 1.0;
    }
    if tokens_a.is_empty() || tokens_b.is_empty() {
        return 0.0;
    }

    let intersection = tokens_a.intersection(&tokens_b).count();
    let union = tokens_a.union(&tokens_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Compute pairwise cosine similarity matrix for a set of vectors
pub fn pairwise_cosine_matrix(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let mut matrix = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n {
            let sim = cosine_similarity(&vectors[i], &vectors[j]);
            matrix[i][j] = sim;
            matrix[j][i] = sim;
        }
    }
    matrix
}

/// Compute entropy of memory diversity based on pairwise similarities
pub fn diversity_entropy(similarities: &[Vec<f32>]) -> f64 {
    if similarities.is_empty() {
        return 0.0;
    }

    let mut probs = Vec::new();
    let mut total = 0.0;
    for row in similarities {
        for &sim in row {
            // Convert similarity to probability-like value
            let prob = (sim.max(0.0) as f64).min(1.0);
            probs.push(prob);
            total += prob;
        }
    }

    if total == 0.0 {
        return 0.0;
    }

    // Normalize to probabilities
    let probs: Vec<f64> = probs.iter().map(|&p| p / total).collect();
    shannon_entropy(&probs)
}
