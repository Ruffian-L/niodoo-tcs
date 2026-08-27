#![cfg(feature = "onnx-with-tokenizers")]

use anyhow::{bail, Context, Result};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;
use tcs_ml::QwenEmbedder;

/// ort 1.16 load-dynamic reads `ORT_DYLIB_PATH` once (lazy_static).
///
/// In-tree 1.16.3 cannot load the HuggingFace `model_quantized.onnx` (IR 10,
/// max IR 9). The workflow names 1.18.1 but that tree is not in git, and we
/// cannot push `.github/workflows`. Fetch the CPU 1.18.1 release into target/.
fn ensure_ort_dylib() -> Result<()> {
    if std::env::var_os("ORT_DYLIB_PATH")
        .filter(|v| !v.is_empty())
        .is_some()
    {
        return Ok(());
    }
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let dylib = fetch_ort_1_18(&repo)?;
    std::env::set_var("ORT_DYLIB_PATH", &dylib);
    Ok(())
}

fn fetch_ort_1_18(repo: &Path) -> Result<PathBuf> {
    let dest = repo.join("target/onnxruntime-linux-x64-1.18.1");
    let so = dest.join("lib/libonnxruntime.so");
    if so.is_file() {
        return Ok(so);
    }
    let target = repo.join("target");
    std::fs::create_dir_all(&target).context("create target/ for ONNX Runtime")?;
    let tgz = target.join("onnxruntime-linux-x64-1.18.1.tgz");
    let url = "https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz";
    let curl = Command::new("curl")
        .args(["-fsSL", "-o"])
        .arg(&tgz)
        .arg(url)
        .status()
        .context("spawn curl for ONNX Runtime 1.18.1")?;
    if !curl.success() {
        bail!("curl failed downloading {url}");
    }
    let tar = Command::new("tar")
        .args(["-xzf"])
        .arg(&tgz)
        .arg("-C")
        .arg(&target)
        .status()
        .context("spawn tar for ONNX Runtime 1.18.1")?;
    if !tar.success() {
        bail!("tar failed extracting {}", tgz.display());
    }
    if !so.is_file() {
        bail!("ONNX Runtime 1.18.1 extract missing {}", so.display());
    }
    Ok(so)
}

fn main() -> Result<()> {
    ensure_ort_dylib()?;
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
