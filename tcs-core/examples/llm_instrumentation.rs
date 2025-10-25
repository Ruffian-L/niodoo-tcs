//! Example: How to instrument LLM prompts with stuck/rogue detection
//! Add these hooks to your LLM inference

use tcs_core::{init_metrics, get_registry};
use prometheus::CounterVec;

pub fn instrumented_llm_prompt(
    prompt: &str,
    output: &str,
) -> Result<String, String> {
    init_metrics();
    let registry = get_registry();
    
    let llm_counter = registry.get_metric("tcs_llm_prompts_total")
        .map(|m| m.as_any().downcast_ref::<CounterVec>().unwrap().clone())
        .ok();
    
    // Detect prompt type
    let prompt_type = detect_prompt_type(prompt, output);
    
    // Record metric
    if let Some(ref counter) = llm_counter {
        counter.with_label_values(&[prompt_type]).inc();
    }
    
    Ok(output.to_string())
}

fn detect_prompt_type(prompt: &str, output: &str) -> &str {
    // Stuck detection: High entropy (repetitive output)
    let entropy = calculate_entropy(output);
    if entropy > 2.0 {
        return "stuck";
    }
    
    // Rogue detection: Anomalous output (high variance, unusual patterns)
    let variance = calculate_variance(output);
    if variance > 1.0 {
        return "rogue";
    }
    
    // Unstuck: Low variance, consistent output
    if variance < 0.1 {
        return "unstuck";
    }
    
    "normal"
}

fn calculate_entropy(text: &str) -> f64 {
    use std::collections::HashMap;
    let mut counts = HashMap::new();
    let total = text.len() as f64;
    
    for ch in text.chars() {
        *counts.entry(ch).or_insert(0) += 1;
    }
    
    counts.values()
        .map(|&count| {
            let p = count as f64 / total;
            if p > 0.0 { -p * p.log2() } else { 0.0 }
        })
        .sum()
}

fn calculate_variance(text: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    
    let chars: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
    let mean = chars.iter().sum::<f64>() / chars.len() as f64;
    let variance = chars.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / chars.len() as f64;
    
    variance.sqrt()
}

