use std::collections::HashSet;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

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
/// Uses PyO3 bridge to rouge-score library if available, falls back to native Rust impl.
pub fn rouge_l(candidate: &str, reference: &str) -> f64 {
    // Try Python rouge-score first for accuracy
    #[cfg(feature = "pyo3")]
    {
        if let Ok(score) = rouge_l_py(candidate, reference) {
            return score;
        }
    }

    // Fallback to native Rust implementation
    rouge_l_native(candidate, reference)
}

/// Python bridge for ROUGE-L using rouge-score library via PyO3
#[cfg(feature = "pyo3")]
fn rouge_l_py(candidate: &str, reference: &str) -> PyResult<f64> {
    Python::with_gil(|py| {
        let module = py.import_bound("rouge_score.rouge_scorer")?;
        let scorer_class = module.getattr("RougeScorer")?;
        let scorer = scorer_class.call1((vec!["rougeL"], Some("f")))?;

        let scores = scorer.call_method1("score", (reference, candidate))?;
        let rouge_dict = scores.downcast::<pyo3::types::PyDict>()?;
        let rouge_l = rouge_dict.get_item("rougeL")?.unwrap();
        let fmeasure = rouge_l.getattr("fmeasure")?;
        fmeasure.extract::<f64>()
    })
}

/// Native Rust implementation of ROUGE-L (fallback)
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
