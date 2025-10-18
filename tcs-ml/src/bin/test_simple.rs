use anyhow::Result;
use tcs_ml::InferenceModelBackend as ModelBackend;

fn main() -> Result<()> {
    println!("Testing Qwen2.5-Coder ONNX integration...");

    let model_path = std::env::var("QWEN_MODEL_PATH")
        .unwrap_or_else(|_| {
            "models/qwen2.5-coder-0.5b-instruct-onnx/onnx/model_quantized.onnx".to_string()
        });

    let model = ModelBackend::new("Qwen2.5-Coder")?;
    model.load(model_path)?;

    println!("✓ Model loaded successfully!");

    let test_prompt = "Hello, world! This is a test.";
    println!("Testing inference with prompt: '{}'", test_prompt);

    match model.extract_embeddings(test_prompt) {
        Ok(embeddings) => {
            println!("✓ Successfully extracted embeddings!");
            println!("  - Embedding dimensions: {}", embeddings.len());
            println!(
                "  - First 10 values: {:?}",
                &embeddings[..10.min(embeddings.len())]
            );

            // Check if embeddings are meaningful (not all zeros)
            let non_zero_count = embeddings.iter().filter(|&&x| x != 0.0).count();
            println!(
                "  - Non-zero values: {}/{}",
                non_zero_count,
                embeddings.len()
            );

            if non_zero_count > 0 {
                println!("✓ Embeddings contain meaningful values!");
            } else {
                println!("⚠ Warning: All embeddings are zero - this might indicate an issue");
            }
        }
        Err(e) => {
            println!("✗ Failed to extract embeddings: {}", e);
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
