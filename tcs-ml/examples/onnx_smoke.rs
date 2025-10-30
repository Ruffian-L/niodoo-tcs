// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

use std::env;

use anyhow::{Context, Result};
use tcs_ml::{Brain, MotorBrain};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut args = env::args().skip(1);
    let model_path = args
        .next()
        .context("expected model path as first argument")?;
    let prompt = args.collect::<Vec<String>>().join(" ");
    let prompt = if prompt.is_empty() {
        "Write a short Rust function that adds two numbers.".to_string()
    } else {
        prompt
    };

    println!("🧠 Testing Qwen2.5-Coder-0.5B-Instruct ONNX Model");
    println!("Model path: {}", model_path);
    println!("Prompt: {}", prompt);
    println!("{}", "=".repeat(60));

    let mut brain = MotorBrain::new()?;
    brain
        .load_model(&model_path)
        .await
        .with_context(|| format!("failed to load model at {}", model_path))?;

    // Test traditional text processing
    println!("\n🔤 Traditional Text Processing:");
    let response = brain
        .process(&prompt)
        .await
        .with_context(|| "failed to run motor brain inference".to_string())?;
    println!("{}", response);

    // Test embedding extraction for TCS integration
    println!("\n🔢 Embedding Extraction for TCS:");
    let embeddings = brain
        .extract_embeddings(&prompt)
        .await
        .with_context(|| "failed to extract embeddings".to_string())?;

    println!("Extracted {} embeddings", embeddings.len());

    // Show first 10 embeddings
    if embeddings.len() >= 10 {
        println!("First 10 embeddings: {:?}", &embeddings[0..10]);
    } else {
        println!("All embeddings: {:?}", embeddings);
    }

    // Show some statistics
    let mean: f32 = embeddings.iter().sum::<f32>() / embeddings.len() as f32;
    let min = embeddings.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = embeddings.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    println!("Embedding statistics:");
    println!("  Mean: {:.6}", mean);
    println!("  Min:  {:.6}", min);
    println!("  Max:  {:.6}", max);

    println!("\n✅ Test completed successfully!");
    Ok(())
}
