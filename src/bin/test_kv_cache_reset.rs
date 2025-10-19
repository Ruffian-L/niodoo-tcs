//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Test KV cache reset between multiple inferences
//!
//! This test verifies that the KV cache is properly cleared between
//! successive inference calls, preventing shape mismatch errors.

use niodoo_core::qwen_integration::{QwenConfig, QwenIntegrator};
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    info!("🧪 Testing KV Cache Reset Between Multiple Inferences");
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Create integrator with small token limits for fast testing
    let config = QwenConfig {
        max_tokens: 20, // Small token count for fast iterations
        ..Default::default()
    };

    let mut integrator = QwenIntegrator::new(config)?;
    info!("✅ Integrator created successfully");

    // Load model once
    info!("⏳ Loading model...");
    integrator.load_model().await?;
    info!("✅ Model loaded successfully");

    // Test multiple inferences to verify KV cache is reset
    let test_prompts = vec![
        ("user".to_string(), "What is 2+2?".to_string()),
        ("user".to_string(), "Tell me about consciousness.".to_string()),
        ("user".to_string(), "Explain quantum mechanics.".to_string()),
        ("user".to_string(), "What is the meaning of life?".to_string()),
    ];

    info!("\n🔄 Running {} sequential inferences...", test_prompts.len());

    for (idx, messages) in test_prompts.iter().enumerate() {
        info!("\n📝 Inference {}/{}", idx + 1, test_prompts.len());
        info!("   Prompt: {}", messages.1);

        match integrator.infer(vec![messages.clone()], Some(20)).await {
            Ok(response) => {
                info!("   ✅ Success! Response: {}", response.chars().take(100).collect::<String>());
                if response.len() > 100 {
                    info!("   ... (truncated)");
                }
            }
            Err(e) => {
                error!("   ❌ FAILED: {}", e);
                error!("\n🚨 KV Cache reset appears to be broken!");
                error!("   Expected: All inferences succeed");
                error!("   Actual: Inference {} failed with: {}", idx + 1, e);
                return Err(e.into());
            }
        }
    }

    info!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("✅ ALL TESTS PASSED!");
    info!("   - {} inferences completed successfully", test_prompts.len());
    info!("   - KV cache is properly reset between inferences");
    info!("   - No shape mismatch errors occurred");

    Ok(())
}
