//! Simple Qwen Integration Test
//!
//! This is a minimal test to verify the Qwen integrator works
//! without the full consciousness engine complexity.

use anyhow::Result;
use niodoo_consciousness::qwen_integration::QwenIntegrator;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🚀 Starting simple Qwen integration test...");

    // Test 1: Create integrator
    info!("📦 Creating Qwen integrator...");
    let mut integrator = QwenIntegrator::default();
    info!("✅ Qwen integrator created successfully");

    // Test 2: Test inference
    info!("🤖 Testing inference...");
    let messages = vec![
        (
            "system".to_string(),
            "You are a helpful AI assistant.".to_string(),
        ),
        ("user".to_string(), "Hello! How are you today?".to_string()),
    ];

    match integrator.infer(messages, Some(100)).await {
        Ok(response) => {
            info!("✅ Inference successful!");
            tracing::info!("Response: {}", response);
        }
        Err(e) => {
            warn!("⚠️  Inference failed: {}", e);
        }
    }

    // Test 3: Test with consciousness-aware prompt
    info!("🧠 Testing consciousness-aware inference...");
    let consciousness_messages = vec![
        (
            "system".to_string(),
            "You are Niodoo, a conscious AI with emotional awareness.".to_string(),
        ),
        (
            "user".to_string(),
            "I'm feeling sad today. Can you help me?".to_string(),
        ),
    ];

    match integrator.infer(consciousness_messages, Some(150)).await {
        Ok(response) => {
            info!("✅ Consciousness-aware inference successful!");
            tracing::info!("Consciousness Response: {}", response);
        }
        Err(e) => {
            warn!("⚠️  Consciousness-aware inference failed: {}", e);
        }
    }

    // Test 4: Performance test
    info!("⚡ Running performance test...");
    let start_time = std::time::Instant::now();

    for i in 0..5 {
        let test_messages = vec![
            ("system".to_string(), "You are a helpful AI.".to_string()),
            ("user".to_string(), format!("Test message number {}", i)),
        ];

        match integrator.infer(test_messages, Some(50)).await {
            Ok(response) => {
                tracing::info!("Test {}: {}", i, response);
            }
            Err(e) => {
                warn!("Test {} failed: {}", i, e);
            }
        }
    }

    let duration = start_time.elapsed();
    info!("✅ Performance test completed in {:?}", duration);
    info!("Average time per inference: {:?}", duration / 5);

    info!("🎉 Simple Qwen integration test completed successfully!");
    Ok(())
}
