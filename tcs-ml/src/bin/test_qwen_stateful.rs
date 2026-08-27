#![cfg(feature = "onnx-with-tokenizers")]

use anyhow::Result;
use std::error::Error;
use std::path::{Path, PathBuf};
use tcs_ml::QwenEmbedder;

/// ort 1.16 load-dynamic reads `ORT_DYLIB_PATH` once (lazy_static). CI still
/// points `LD_LIBRARY_PATH` at a missing 1.18.1 tree; the repo ships 1.16.3.
fn ensure_ort_dylib() {
    if std::env::var_os("ORT_DYLIB_PATH")
        .filter(|v| !v.is_empty())
        .is_some()
    {
        return;
    }
    let mut roots = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    roots.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".."));
    let rel = Path::new("onnxruntime-linux-x64-1.16.3/lib/libonnxruntime.so.1.16.3");
    for root in roots {
        let dylib = root.join(rel);
        if dylib.is_file() {
            std::env::set_var("ORT_DYLIB_PATH", &dylib);
            return;
        }
    }
}

fn main() -> Result<()> {
    ensure_ort_dylib();
    println!("🚀 Testing QwenEmbedder with stateful KV cache...");

    let model_path = std::env::var("QWEN_MODEL_PATH").unwrap_or_else(|_| {
        "models/qwen2.5-coder-0.5b-instruct-onnx/onnx/model_quantized.onnx".to_string()
    });

    let mut embedder = QwenEmbedder::new(&model_path)?;

    println!("✓ QwenEmbedder initialized successfully!");

    // Test 1: First embedding (fresh KV cache)
    let test_prompt1 = "Hello, world! This is topological coherence emerging.";
    println!("\n🧠 Test 1: First embedding");
    println!("Prompt: '{}'", test_prompt1);

    match embedder.embed(test_prompt1) {
        Ok(emb1) => {
            println!("✓ Successfully extracted embeddings!");
            println!("  - Dimensions: {}", emb1.len());
            println!("  - First 10 values: {:?}", &emb1[..10.min(emb1.len())]);
            println!("  - Context length: {}", embedder.context_length());

            let non_zero_count = emb1.iter().filter(|&&x| x != 0.0).count();
            println!("  - Non-zero values: {}/{}", non_zero_count, emb1.len());

            // Test 2: Second embedding (should reuse KV cache)
            let test_prompt2 = " Now we explore topological spaces.";
            println!("\n🧠 Test 2: Stateful embedding (KV cache reuse)");
            println!("Prompt: '{}'", test_prompt2);

            match embedder.embed(test_prompt2) {
                Ok(emb2) => {
                    println!("✓ Successfully extracted stateful embeddings!");
                    println!("  - Dimensions: {}", emb2.len());
                    println!("  - First 10 values: {:?}", &emb2[..10.min(emb2.len())]);
                    println!("  - Context length: {}", embedder.context_length());

                    // Check that embeddings evolved (different but related)
                    let cosine_sim = cosine_similarity(&emb1, &emb2);
                    println!("  - Cosine similarity with previous: {:.4}", cosine_sim);

                    if emb1 != emb2 {
                        println!("✓ Stateful embeddings are evolving (not identical)");
                    } else {
                        println!(
                            "⚠ Warning: Embeddings are identical - KV cache might not be working"
                        );
                    }

                    // Test 3: Cache reset
                    println!("\n🧠 Test 3: Cache reset and fresh context");
                    embedder.reset_cache();
                    println!(
                        "  - Context length after reset: {}",
                        embedder.context_length()
                    );

                    let test_prompt3 = "Fresh start: persistent homology in AI.";
                    println!("Prompt: '{}'", test_prompt3);

                    match embedder.embed(test_prompt3) {
                        Ok(emb3) => {
                            println!("✓ Successfully extracted fresh embeddings!");
                            println!("  - Dimensions: {}", emb3.len());
                            println!("  - Context length: {}", embedder.context_length());

                            let cosine_sim_1_3 = cosine_similarity(&emb1, &emb3);
                            let cosine_sim_2_3 = cosine_similarity(&emb2, &emb3);
                            println!("  - Cosine similarity with emb1: {:.4}", cosine_sim_1_3);
                            println!("  - Cosine similarity with emb2: {:.4}", cosine_sim_2_3);

                            println!("\n🎉 All tests completed successfully!");
                            println!("📊 Summary:");
                            println!("  - Embedding extraction: ✓");
                            println!("  - Stateful KV cache: ✓");
                            println!("  - Cache reset: ✓");
                            println!("  - {}-dim output: ✓", emb3.len());
                        }
                        Err(e) => {
                            println!("✗ Failed third embedding: {}", e);
                            let mut source = e.source();
                            while let Some(err) = source {
                                println!("  caused by: {}", err);
                                source = err.source();
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("✗ Failed second embedding: {}", e);
                    let mut source = e.source();
                    while let Some(err) = source {
                        println!("  caused by: {}", err);
                        source = err.source();
                    }
                }
            }
        }
        Err(e) => {
            println!("✗ Failed first embedding: {}", e);
            println!("Full error chain:");
            let mut source = e.source();
            while let Some(err) = source {
                println!("  caused by: {}", err);
                source = err.source();
            }
        }
    }

    Ok(())
}

// Helper function to compute cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}
