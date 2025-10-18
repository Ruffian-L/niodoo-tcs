/*
 * 🧠 Simple Qwen Inference Test
 */

use niodoo_consciousness::qwen_integration::{QwenConfig, QwenIntegrator};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    
    tracing::info!("🧠 Qwen Inference Test");
    tracing::info!("======================\n");

    let qwen_config = QwenConfig::default();
    let mut qwen = QwenIntegrator::new(qwen_config)?;
    
    tracing::info!("Loading model...");
    qwen.load_model().await?;
    tracing::info!("Model loaded successfully!\n");

    // Test 1: Simple greeting
    tracing::info!("TEST 1: Simple Greeting");
    tracing::info!("-----------------------");
    let messages = vec![
        ("system", "You are a helpful AI assistant."),
        ("user", "Hello! How are you?"),
    ];
    match qwen.infer(messages, Some(50)).await {
        Ok(result) => {
            tracing::info!("Response: {}\n", result.trim());
        }
        Err(e) => tracing::error!("Error: {}", e),
    }

    // Test 2: Code generation
    tracing::info!("TEST 2: Code Generation");
    tracing::info!("-----------------------");
    let messages = vec![
        ("system", "You are a Rust programming expert."),
        ("user", "Write a simple Rust function to add two numbers."),
    ];
    match qwen.infer(messages, Some(100)).await {
        Ok(result) => {
            tracing::info!("Response: {}\n", result.trim());
        }
        Err(e) => tracing::error!("Error: {}", e),
    }

    tracing::info!("✅ Tests completed!");
    Ok(())
}