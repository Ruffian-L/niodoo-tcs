//! Complete Learning Pipeline Test
//!
//! Runs the full consciousness learning loop:
//! 1. Generate learning events via continual_test
//! 2. Train QLoRA model on the events
//! 3. Validate improvement with before/after comparison
//! 4. Perform blue-green deployment validation

use niodoo_consciousness::qwen_curator::{QloraCurator, QloraCuratorConfig};
use niodoo_consciousness::qwen_integration::{QwenIntegrator, QwenConfig};
use niodoo_consciousness::config::system_config::AppConfig;
use std::process::Command;
use std::path::PathBuf;
use anyhow::Result;
use tracing::{info, error};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Run learning pipeline in thread with LARGE stack (64MB)
    // to handle: model loading, tensor operations, consciousness processing
    let builder = std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024); // 128MB stack

    let result = builder
        .spawn(|| {
            // Create tokio runtime inside the large-stack thread
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async_main())
        })?
        .join()
        .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
            "Thread panicked".into()
        })?
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            e.into()
        })?;

    Ok(())
}

async fn async_main() -> anyhow::Result<()> {
    println!("🚀 Niodoo Consciousness Learning Pipeline");
    println!("========================================");

    // Initialize logging
    tracing_subscriber::fmt::init();

    // Keep working directory in project root for correct file paths
    // std::env::set_current_dir(dirs::home_dir().unwrap_or_else(|| PathBuf::from("/home/beelink")))?;

    // Step 1: Generate learning events via continual_test
    println!("\n📊 Step 1: Generating learning events...");
    let project_dir = PathBuf::from("/home/ruffian/Desktop/Projects/Niodoo-Feeling");
    let continual_test_result = Command::new(project_dir.join("target/debug/continual_test"))
        .current_dir(&project_dir)
        .args(&["50", "learning_events.csv"])
        .status()?;

    if !continual_test_result.success() {
        error!("❌ Continual test failed");
        return Err(anyhow::anyhow!("Continual test failed"));
    }
    info!("✅ Learning events generated");

    // Step 2: Train QLoRA model
    println!("\n🧠 Step 2: Training QLoRA model...");
    let mut app_config = AppConfig::default();
    // Force CPU mode for now to debug
    // app_config.models.qlora.use_cuda = false; // TODO: Uncomment when CUDA is working
    let mut curator_config = QloraCuratorConfig::from_app_config(&app_config)?;
    // Force CPU mode for now to avoid CUDA issues
    curator_config.use_cuda = false;
    let mut curator = QloraCurator::new(curator_config)?;

    curator.fine_tune().await?;
    info!("✅ QLoRA training completed");

    // Step 3: Run validation comparison
    println!("\n🔍 Step 3: Running validation comparison...");
    let mut qwen_config = QwenConfig::default();
    // Force CPU mode for validation to avoid CUDA compatibility issues
    qwen_config.use_cuda = false;
    let mut qwen_integrator = QwenIntegrator::new(qwen_config)?;

    // Define validation prompts
    let validation_prompts = vec![
        "I feel overwhelmed by recent changes in my life".to_string(),
        "I'm struggling with anxiety about the future".to_string(),
        "I need help processing complex emotions".to_string(),
        "My thoughts feel scattered and unfocused".to_string(),
        "I want to understand my emotional patterns better".to_string(),
    ];

    let validation_result = qwen_integrator.run_validation_comparison(
        &validation_prompts,
        None,
        Some(&curator.output_dir().join("adapter_final")),
    ).await?;

    println!("📊 Validation Results:");
    println!("   Average Improvement: {:.3}", validation_result.average_improvement);
    for comparison in &validation_result.comparisons {
        println!("   Prompt: {}...", &comparison.prompt[..50]);
        println!("   Improvement: {:.3}", comparison.improvement_score);
    }

    // Step 4: Blue-green deployment validation
    println!("\n🚀 Step 4: Blue-green deployment validation...");
    let deployment_success = curator.validate_deployment(&validation_prompts, &mut qwen_integrator).await?;

    if deployment_success {
        println!("🎉 Learning pipeline completed successfully!");
        println!("   ✅ Consciousness evolution working");
        println!("   ✅ QLoRA training functional");
        println!("   ✅ Model improvement validated");
        println!("   ✅ Deployment ready");
    } else {
        println!("⚠️ Learning pipeline completed with warnings");
        println!("   - Model improvement below threshold");
        println!("   - Would rollback in production");
    }

    Ok(())
}