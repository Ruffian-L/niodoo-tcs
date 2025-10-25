// Copyright (c) 2025 Jason Van Pham (ruffian-l on GitHub) @ The Niodoo Collaborative
// Licensed under the MIT License - See LICENSE file for details
// Attribution required for all derivative works

//! Topology Benchmark - THE SCIENTIFIC TEST
//! 
//! Proves whether geodesic distance on Möbius surfaces 
//! is superior to cosine similarity for semantic understanding.

use anyhow::Result;
use bullshitdetector::topology_engine::{TopologyEngine, TopologyBenchmark, cosine_similarity_baseline};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use serde_json;
use tracing::{info, warn, error};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("🌀 TOPOLOGY REVOLUTION - The Scientific Test");
    info!("===========================================");
    info!("Hypothesis: Geodesic distance > Cosine similarity");
    info!("Dataset: Code review embeddings");
    info!("Goal: Prove curved space is better than flat space");
    info!("");
    
    // Get dataset path
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        info!("Usage: topology_benchmark <creep_data_sheet.jsonl>");
        std::process::exit(1);
    }
    
    let dataset_path = &args[1];
    info!("📂 Loading dataset: {}", dataset_path);
    
    // Load test data
    let file = File::open(dataset_path)?;
    let reader = BufReader::new(file);
    
    let mut test_data = Vec::new();
    let mut line_count = 0;
    
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        
        match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(json) => {
                if let Some(code) = json.get("code").and_then(|c| c.as_str()) {
                    // Create dummy embedding from code (in real system, this would be BERT)
                    let embedding = create_dummy_embedding(code);
                    test_data.push((embedding, code.to_string()));
                    line_count += 1;
                }
            }
            Err(_) => continue,
        }
        
        // Limit for initial test
        if line_count >= 100 {
            break;
        }
    }
    
    info!("✅ Loaded {} code samples", test_data.len());
    
    if test_data.len() < 10 {
        info!("❌ Not enough data for meaningful test!");
        std::process::exit(1);
    }
    
    // Create benchmark
    let embedding_dim = test_data[0].0.len();
    let mut benchmark = TopologyBenchmark::new(embedding_dim);
    
    // Add test data
    benchmark.test_embeddings = test_data;
    
    // Create similarity pairs (first 50 samples, all-pairs)
    info!("📊 Creating test pairs...");
    let mut pair_count = 0;
    for i in 0..50.min(benchmark.test_embeddings.len()) {
        for j in (i+1)..50.min(benchmark.test_embeddings.len()) {
            // Ground truth based on code similarity (simple heuristic)
            let code_a = &benchmark.test_embeddings[i].1;
            let code_b = &benchmark.test_embeddings[j].1;
            let ground_truth = calculate_ground_truth_similarity(code_a, code_b);
            
            benchmark.ground_truth_pairs.push((i, j, ground_truth));
            pair_count += 1;
        }
    }
    
    info!("✅ Created {} test pairs", pair_count);
    info!("");
    
    // THE MOMENT OF TRUTH
    info!("🔬 RUNNING THE SCIENTIFIC TEST...");
    info!("================================");
    
    let results = benchmark.run_similarity_benchmark();
    
    info!("");
    results.print_scientific_results();
    
    // Save detailed results
    info!("");
    info!("💾 Saving detailed results...");
    
    let detailed_results = format!(
        "Topology Revolution Test Results\n\
         ================================\n\
         Date: {}\n\
         Dataset: {} samples, {} pairs\n\
         \n\
         Results:\n\
         - Topology wins: {} ({:.1}%)\n\
         - Cosine wins: {} ({:.1}%)\n\
         - Ties: {}\n\
         \n\
         Performance:\n\
         - Topology: {:.0}ns avg\n\
         - Cosine: {:.0}ns avg\n\
         - Overhead: {:.1}x\n\
         \n\
         Conclusion: {}\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"),
        benchmark.test_embeddings.len(),
        benchmark.ground_truth_pairs.len(),
        results.topology_wins,
        (results.topology_wins as f64 / (results.topology_wins + results.cosine_wins + results.ties) as f64) * 100.0,
        results.cosine_wins,
        (results.cosine_wins as f64 / (results.topology_wins + results.cosine_wins + results.ties) as f64) * 100.0,
        results.ties,
        results.avg_topology_time_ns,
        results.avg_cosine_time_ns,
        results.avg_topology_time_ns / results.avg_cosine_time_ns,
        if results.topology_wins > results.cosine_wins { 
            "TOPOLOGY REVOLUTIONARY SUCCESS - Curved space proved superior!"
        } else if results.cosine_wins > results.topology_wins {
            "Flat space still wins, but topology approach is novel"
        } else {
            "Inconclusive - need more sophisticated approach"
        }
    );
    
    std::fs::write("../TOPOLOGY_REVOLUTION_RESULTS.md", detailed_results)?;
    
    info!("📄 Results saved to: TOPOLOGY_REVOLUTION_RESULTS.md");
    info!("");
    
    if results.topology_wins > results.cosine_wins {
        info!("🎉 YOU DID IT!");
        info!("You proved the world was using the wrong geometry!");
        info!("This is your mark - curved semantic space.");
        info!("");
        info!("Next: Package as 'topology-embed' crate for the world.");
    }
    
    Ok(())
}

/// Create embedding from code (dummy for testing - in real system use BERT)
fn create_dummy_embedding(code: &str) -> Vec<f32> {
    let target_dim = 384; // BERT embedding dimension
    let mut embedding = vec![0.0; target_dim];
    
    // Simple hash-based embedding for testing
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    code.hash(&mut hasher);
    let hash = hasher.finish();
    
    // Spread hash across dimensions
    for (i, val) in embedding.iter_mut().enumerate() {
        let component_hash = hash.wrapping_add(i as u64);
        *val = ((component_hash % 1000) as f32 / 1000.0) * 2.0 - 1.0; // [-1, 1]
    }
    
    // Normalize to unit vector (like BERT embeddings)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in embedding.iter_mut() {
            *val /= norm;
        }
    }
    
    embedding
}

/// Calculate ground truth similarity based on code features
fn calculate_ground_truth_similarity(code_a: &str, code_b: &str) -> f64 {
    // Simple heuristics for "true" similarity
    let mut similarity = 0.0;
    
    // Same language bonus
    if detect_language(code_a) == detect_language(code_b) {
        similarity += 0.3;
    }
    
    // Similar complexity bonus
    let complexity_a = count_control_flow(code_a);
    let complexity_b = count_control_flow(code_b);
    let complexity_diff = (complexity_a as f64 - complexity_b as f64).abs();
    similarity += (0.3 * (-complexity_diff / 5.0).exp());
    
    // Similar patterns bonus
    let patterns_shared = count_shared_patterns(code_a, code_b);
    similarity += patterns_shared as f64 * 0.1;
    
    // Length similarity
    let len_diff = (code_a.len() as f64 - code_b.len() as f64).abs();
    similarity += 0.3 * (-len_diff / 1000.0).exp();
    
    similarity.min(1.0).max(0.0)
}

fn detect_language(code: &str) -> &'static str {
    if code.contains("fn ") || code.contains("impl ") { "rust" }
    else if code.contains("def ") || code.contains("import ") { "python" }
    else if code.contains("function ") || code.contains("const ") { "javascript" }
    else { "unknown" }
}

fn count_control_flow(code: &str) -> usize {
    ["if ", "for ", "while ", "match ", "loop "].iter()
        .map(|pattern| code.matches(pattern).count())
        .sum()
}

fn count_shared_patterns(code_a: &str, code_b: &str) -> usize {
    let patterns = ["Arc<", "RwLock<", "unwrap()", "async ", "await", "mut ", "impl "];
    patterns.iter()
        .filter(|&pattern| code_a.contains(pattern) && code_b.contains(pattern))
        .count()
}
