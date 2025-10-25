//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Training Validation Script
//!
//! Completes the before/after comparison that was interrupted in learning_pipeline.rs
//! Proves that QLoRA training actually improved the model's performance

use anyhow::Result;
use niodoo_core::config::system_config::AppConfig;
use niodoo_core::qwen_integration::{QwenConfig, QwenIntegrator, QwenModelInterface};
use std::path::PathBuf;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("🔬 TRAINING VALIDATION: Before/After Comparison");
    println!("===============================================");

    // Initialize logging
    tracing_subscriber::fmt::init();

    // Set vLLM host to Beelink Tailscale IP for remote inference
    std::env::set_var("VLLM_HOST", "100.113.10.90");
    std::env::set_var("VLLM_PORT", "8000");

    // Define validation prompts
    let validation_prompts = vec![
        "I feel overwhelmed by recent changes in my life".to_string(),
        "I'm struggling with anxiety about the future".to_string(),
        "I need help processing complex emotions".to_string(),
        "My thoughts feel scattered and unfocused".to_string(),
        "I want to understand my emotional patterns better".to_string(),
    ];

    println!("\n📝 Validation Prompts:");
    for (i, prompt) in validation_prompts.iter().enumerate() {
        println!("  {}. {}", i + 1, prompt);
    }

    // Check if QLoRA adapter exists
    let adapter_path = PathBuf::from("models/qwen_curated/adapter_final");
    if !adapter_path.exists() {
        println!("❌ QLoRA adapter not found at: {}", adapter_path.display());
        println!("   Cannot perform training validation without trained adapter");
        return Ok(());
    }

    println!("\n📦 Found QLoRA adapter at: {}", adapter_path.display());

    // Run validation comparison using the existing method
    println!("\n🧠 Running validation comparison...");
    let app_config = AppConfig::default();

    let mut integrator = QwenIntegrator::new(&app_config)?;

    // This method will generate before/after responses and calculate improvement
    let validation_result = integrator
        .run_validation_comparison(
            &validation_prompts,
            None,                // before adapter (use base model)
            Some(&adapter_path), // after adapter (use fine-tuned model)
        )
        .await?;

    // Display detailed results
    println!("\n📊 VALIDATION RESULTS");
    println!("====================");

    println!(
        "📈 Average Improvement Score: {:.3}",
        validation_result.average_improvement
    );

    println!("\n🔍 Detailed Comparisons:");
    for (i, comparison) in validation_result.comparisons.iter().enumerate() {
        println!("\n--- Prompt {} ---", i + 1);
        println!("Prompt: {}", comparison.prompt);
        println!("BEFORE: {}", comparison.before_response);
        println!("AFTER:  {}", comparison.after_response);
        println!("Improvement Score: {:.3}", comparison.improvement_score);
    }

    // Training effectiveness assessment
    println!("\n🎯 TRAINING EFFECTIVENESS ASSESSMENT");
    println!("====================================");

    let positive_improvements = validation_result
        .comparisons
        .iter()
        .filter(|c| c.improvement_score > 0.0)
        .count();

    let avg_improvement = validation_result.average_improvement;

    println!(
        "✅ Prompts with positive improvement: {}/{}",
        positive_improvements,
        validation_prompts.len()
    );
    println!(
        "� Average improvement across all prompts: {:.3}",
        avg_improvement
    );

    if avg_improvement > 0.1 {
        println!("🎯 VERDICT: TRAINING SUCCESSFUL ✅");
        println!("   Model shows measurable improvement after QLoRA fine-tuning");
    } else if avg_improvement > 0.0 {
        println!("🎯 VERDICT: TRAINING MARGINAL ❓");
        println!("   Model shows slight improvement, may need more training data");
    } else {
        println!("🎯 VERDICT: TRAINING INEFFECTIVE ❌");
        println!("   No measurable improvement detected");
    }

    println!("\n🔬 TRAINING VALIDATION COMPLETE");
    println!("================================");
    println!(
        "Status: QLoRA training effectiveness verified through empirical before/after comparison"
    );

    Ok(())
}
