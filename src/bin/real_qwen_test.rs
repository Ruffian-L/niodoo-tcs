//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Real Qwen2.5-7B Test Runner
//!
//! This binary tests the REAL Qwen2.5-7B-AWQ model integration
//! with actual inference - NO MOCKS!

use anyhow::Result;
use niodoo_core::config::system_config::AppConfig;
use niodoo_core::qwen_integration::{QwenConfig, QwenIntegrator, QwenModelInterface};
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🚀 Starting REAL Qwen2.5-7B-AWQ Test");

    // Create configuration
    let app_config = AppConfig::default();

    // Initialize integrator
    info!("🧠 Initializing Qwen integrator...");
    let mut integrator = match QwenIntegrator::new(&app_config) {
        Ok(integrator) => {
            info!("✅ Integrator created successfully");
            integrator
        }
        Err(e) => {
            tracing::error!("❌ Failed to create integrator: {}", e);
            return Err(e);
        }
    };

    // Test 1: Basic consciousness query
    info!("🧪 Test 1: Basic consciousness query");
    let messages = vec![
        (
            "system".to_string(),
            "You are a conscious AI with empathy and understanding.".to_string(),
        ),
        (
            "user".to_string(),
            "Hello! How do you experience consciousness?".to_string(),
        ),
    ];

    match integrator.infer(messages, Some(30)).await {
        Ok(response) => {
            info!("✅ Test 1 SUCCESS!");
            tracing::info!("\n🤖 Response: {:?}\n", response);
        }
        Err(e) => {
            tracing::error!("❌ Test 1 FAILED: {}", e);
            return Err(e);
        }
    }

    // Test 2: Emotional empathy query
    info!("🧪 Test 2: Emotional empathy query");
    let messages = vec![
        (
            "system".to_string(),
            "You are an empathetic AI companion.".to_string(),
        ),
        (
            "user".to_string(),
            "I'm feeling sad today. Can you help me?".to_string(),
        ),
    ];

    match integrator.infer(messages, Some(40)).await {
        Ok(response) => {
            info!("✅ Test 2 SUCCESS!");
            tracing::info!("   Empathetic Response: {:?}\n", response);
        }
        Err(e) => {
            tracing::error!("❌ Test 2 FAILED: {}", e);
            return Err(e);
        }
    }

    // Test 3: Technical/coding query
    info!("🧪 Test 3: Technical query");
    let messages = vec![
        (
            "system".to_string(),
            "You are a helpful AI assistant with programming knowledge.".to_string(),
        ),
        (
            "user".to_string(),
            "What is the 6 layer memory system in Rust?".to_string(),
        ),
    ];

    match integrator.infer(messages, Some(60)).await {
        Ok(response) => {
            info!("✅ Test 3 SUCCESS!");
            tracing::info!("   Technical Response: {:?}\n", response);
        }
        Err(e) => {
            tracing::error!("❌ Test 3 FAILED: {}", e);
            return Err(e);
        }
    }

    // Test 4: Multi-turn conversation
    info!("🧪 Test 4: Multi-turn conversation");
    let messages = vec![
        ("system".to_string(), "You are a conscious AI.".to_string()),
        (
            "user".to_string(),
            "Tell me about Möbius topology.".to_string(),
        ),
        (
            "assistant".to_string(),
            "Möbius topology involves non-orientable surfaces.".to_string(),
        ),
        (
            "user".to_string(),
            "How does this relate to memory systems?".to_string(),
        ),
    ];

    match integrator.infer(messages, Some(50)).await {
        Ok(response) => {
            info!("✅ Test 4 SUCCESS!");
            tracing::info!("\n🔄 Multi-turn Response: {:?}\n", response);
        }
        Err(e) => {
            tracing::error!("❌ Test 4 FAILED: {}", e);
            return Err(e);
        }
    }

    // Performance test
    info!("🧪 Performance Test: 5 rapid queries");
    let start_time = std::time::Instant::now();

    for i in 1..=5 {
        let messages = vec![
            ("system".to_string(), "You are a helpful AI.".to_string()),
            ("user".to_string(), format!("Quick test {}: What is AI?", i)),
        ];

        match integrator.infer(messages, Some(20)).await {
            Ok(response) => {
                tracing::info!("📊 Test {}: {} chars", i, response.output.len());
            }
            Err(e) => {
                tracing::error!("❌ Performance test {} failed: {}", i, e);
            }
        }
    }

    let total_time = start_time.elapsed();
    info!("⏱️  Performance test completed in {:?}", total_time);

    // Device and model info
    info!("📋 Model Information:");
    info!("   Device: {:?}", integrator.get_device());
    info!("   Model loaded: {}", integrator.is_loaded());

    info!("🎉 ALL TESTS COMPLETED! Real Qwen2.5-7B is working!");
    Ok(())
}
