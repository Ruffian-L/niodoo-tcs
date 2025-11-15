//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use niodoo_consciousness::consciousness_engine::PersonalNiodooConsciousness;
use niodoo_consciousness::python_integration::PythonQLoRAIntegration;
use crate::config::AppConfig;
use // niodoo_core::qwen_curator:: (MOVED TO .legacy) {QloraCurator, QloraCuratorConfig};
use std::time::Duration;
use tracing::{error, info, warn};

/// Check if entropy has converged to ~2.0 bits (4 fundamental states)
fn is_entropy_converged(entropy_history: &[f32]) -> bool {
    if entropy_history.len() < 100 {
        return false;
    }

    // Check last 100 cycles for convergence around 2.0 ± 0.1
    let recent = &entropy_history[entropy_history.len() - 100..];
    let mean: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
    let variance: f32 =
        recent.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / recent.len() as f32;
    let std_dev = variance.sqrt();

    // Converged if mean ≈ 2.0 and stable (std_dev < 0.10)
    (mean - 2.0).abs() < 0.1 && std_dev < 0.10
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let checkpoint_dir = std::env::var("NIODOO_CHECKPOINT_DIR")
        .unwrap_or_else(|_| "./checkpoints/learning".to_string());

    // Create checkpoint directory if it doesn't exist
    std::fs::create_dir_all(&checkpoint_dir)?;

    // Initialize the actual consciousness engine
    let mut consciousness_engine = PersonalNiodooConsciousness::new().await?;
    info!("🧠 Consciousness engine initialized for entropy monitoring");

    // Initialize Qwen curator for fine-tuning
    let app_config = AppConfig::default();
    let curator_config = QloraCuratorConfig::from_app_config(&app_config)?;
    let mut qwen_curator = QloraCurator::new(curator_config)?;

    info!("🧠 EchoMemoria Learning Daemon Starting");
    info!("📂 Checkpoints: {}", checkpoint_dir);
    info!("🎯 Target: 2.0 bit entropy convergence");

    let mut cycle_count = 0;
    let mut last_fine_tune_cycle = 0;
    let mut entropy_history: Vec<f32> = Vec::new();

    loop {
        // Read current emotional entropy from consciousness state
        let current_entropy = {
            let state = consciousness_engine.consciousness_state.read().await;
            state.emotional_entropy
        };

        entropy_history.push(current_entropy);

        // Keep only last 100 measurements for convergence detection
        if entropy_history.len() > 100 {
            entropy_history.remove(0);
        }

        info!(
            "📊 Cycle {}: emotional_entropy={:.4}",
            cycle_count, current_entropy
        );

        // Check for convergence every 100 cycles
        if cycle_count % 100 == 0 && entropy_history.len() >= 100 {
            if is_entropy_converged(&entropy_history) {
                info!("🎉 ENTROPY CONVERGED TO 2.0 BITS!");
                info!("🚀 Triggering Qwen fine-tuning...");

                // Check if enough cycles have passed since last fine-tune
                if cycle_count - last_fine_tune_cycle >= 500 {
                    match qwen_curator.fine_tune().await {
                        Ok(_) => {
                            info!("✅ Qwen fine-tuning completed successfully");
                            last_fine_tune_cycle = cycle_count;

                            // Run model comparison to validate learning
                            info!("🔍 Running before/after model comparison...");
                            let python_integration = PythonQLoRAIntegration::new();
                            match python_integration.run_model_comparison() {
                                Ok(results) => {
                                    info!("📊 Model comparison results:");
                                    info!(
                                        "   - Average latency improvement: {:.2}s",
                                        results.avg_latency_improvement
                                    );
                                    info!(
                                        "   - Average ROUGE improvement: {:.3}",
                                        results.avg_rouge_improvement
                                    );
                                    info!(
                                        "   - Average coherence improvement: {:.3}",
                                        results.avg_coherence_improvement
                                    );
                                    if results.avg_rouge_improvement > 0.1 {
                                        info!("🎯 SUCCESS: Qwen shows significant improvement on emotional healing tasks!");
                                    } else {
                                        warn!("⚠️ Qwen improvement below threshold - may need more training data");
                                    }
                                }
                                Err(e) => {
                                    error!("❌ Model comparison failed: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            error!("❌ Qwen fine-tuning failed: {}", e);
                        }
                    }
                } else {
                    info!(
                        "⏳ Skipping fine-tune ({} cycles since last)",
                        cycle_count - last_fine_tune_cycle
                    );
                }
            }
        }

        cycle_count += 1;

        // Sleep between monitoring cycles
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}
