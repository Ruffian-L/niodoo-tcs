use std::env;

use anyhow::{Context, Result};
use tcs_ml::{Brain, MotorBrain};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    println!("🧠 Simple ONNX Test - Testing Basic Functionality");

    let mut args = env::args().skip(1);
    let model_path = args
        .next()
        .context("expected model path as first argument")?;
    let prompt = args.collect::<Vec<String>>().join(" ");
    let prompt = if prompt.is_empty() {
        "Hello, World!".to_string()
    } else {
        prompt
    };

    println!("Model path: {}", model_path);
    println!("Prompt: {}", prompt);

    let mut brain = MotorBrain::new()?;

    // Test without loading model first
    println!("\n📊 Testing basic embedding extraction (fallback mode):");
    let fallback_embeddings = brain.extract_embeddings(&prompt).await?;
    println!(
        "Generated {} fallback embeddings",
        fallback_embeddings.len()
    );
    println!(
        "First 5 embeddings: {:?}",
        &fallback_embeddings[0..5.min(fallback_embeddings.len())]
    );

    // Now try loading the model
    println!("\n🔧 Loading ONNX model...");
    match brain.load_model(&model_path).await {
        Ok(_) => {
            println!("✅ Model loaded successfully!");

            // Test text processing
            println!("\n🔤 Testing text processing:");
            let response = brain.process(&prompt).await?;
            println!("{}", response);

            // Test embedding extraction
            println!("\n🔢 Testing embedding extraction:");
            let embeddings = brain.extract_embeddings(&prompt).await?;
            println!("Extracted {} embeddings", embeddings.len());
            if embeddings.len() >= 5 {
                println!("First 5 embeddings: {:?}", &embeddings[0..5]);
            }

            let mean: f32 = embeddings.iter().sum::<f32>() / embeddings.len() as f32;
            let min = embeddings.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = embeddings.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            println!(
                "Statistics - Mean: {:.6}, Min: {:.6}, Max: {:.6}",
                mean, min, max
            );
        }
        Err(e) => {
            println!("❌ Failed to load model: {}", e);
            println!("This might be due to missing tokenizer dependencies or model format issues.");
            println!("Fallback embeddings are still available for TCS integration.");
        }
    }

    println!("\n✅ Test completed!");
    Ok(())
}
