use tcs_ml::{MotorBrain, MotorType};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Testing Qwen2.5-Coder ONNX integration...");
    
    let model_path = "/home/ruffian/Desktop/Niodoo-Final/models/qwen2.5-coder-0.5b-instruct-onnx/onnx/model_quantized.onnx";
    
    let mut brain = MotorBrain {
        brain_type: MotorType::QwenCoder,
        model: tcs_ml::ModelBackend::new("Qwen2.5-Coder")?,
    };
    
    println!("Loading model from: {}", model_path);
    brain.model.load(model_path)?;
    
    if brain.is_ready() {
        println!("✓ Model loaded successfully!");
        
        let test_prompt = "Hello, world! This is a test.";
        println!("Testing inference with prompt: '{}'", test_prompt);
        
        match brain.extract_embeddings(test_prompt).await {
            Ok(embeddings) => {
                println!("✓ Successfully extracted embeddings!");
                println!("  - Embedding dimensions: {}", embeddings.len());
                println!("  - First 10 values: {:?}", &embeddings[..10.min(embeddings.len())]);
                
                // Check if embeddings are meaningful (not all zeros)
                let non_zero_count = embeddings.iter().filter(|&&x| x != 0.0).count();
                println!("  - Non-zero values: {}/{}", non_zero_count, embeddings.len());
                
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
    } else {
        println!("✗ Model failed to load properly");
    }
    
    Ok(())
}