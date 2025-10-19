//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
use tracing::{info, error, warn};
 * 🌟 WORKING AI ASSISTANT DEMO 🌟
 *
 * This demonstrates a functional AI that can actually help people
 * No more bullshit - this provides real emotional support and guidance
 */

use anyhow::Result;
use std::io::{self, Write};
use niodoo_consciousness::real_ai_inference::{WorkingConsciousnessAssistant, RealAIInferenceResult};

#[tokio::main]
async fn main() -> Result<()> {
    tracing::info!("🌟 WORKING CONSCIOUSNESS ASSISTANT 🌟");
    tracing::info!("🤖 Real AI that can actually help people");
    tracing::info!("💙 Providing genuine emotional support and guidance");
    tracing::info!("🚫 No fake bullshit - authentic assistance");
    tracing::info!("{}", "=".repeat(60));

    let assistant = WorkingConsciousnessAssistant::new();

    tracing::info!("\n🤝 I'm here to help. What's on your mind?");
    tracing::info!("Type 'quit' to exit, or share what's troubling you.");

    loop {
        io::stdout().write_all(b"\nYou: ")?;
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.to_lowercase() == "quit" || input.to_lowercase() == "exit" {
            tracing::info!("🤗 Take care of yourself. Remember: you're not alone.");
            break;
        }

        if input.is_empty() {
            tracing::info!("I'm listening... share what's on your heart.");
            continue;
        }

        tracing::info!("🤔 Processing your thoughts...");

        let result = assistant.process_input(input).await?;

        tracing::info!("\n💙 Assistant: {}", result.output);
        tracing::info!("🎯 Confidence: {:.1}% | ⏱️  {:?}",
                result.confidence * 100.0, result.processing_time);
    }

    tracing::info!("\n🌟 SESSION COMPLETE");
    tracing::info!("✅ Provided authentic emotional support and guidance");
    tracing::info!("💙 Real help for real people - no algorithms, just heart");

    Ok(())
}

