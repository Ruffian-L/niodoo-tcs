use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use niodoo_real_integrated::config::{CliArgs, HardwareProfile};
use niodoo_real_integrated::pipeline::Pipeline;
use tracing::{info, warn};

fn default_prompts() -> Vec<&'static str> {
    vec![
        "Solve 2x + 3 = 7",
        "What is 15 * 4?",
        "Find x where x^2 = 16",
        "Explain the Pythagorean theorem in one sentence",
        "Convert 45 degrees to radians",
        "Simplify the fraction 18/24",
        "What is the derivative of x^2?",
        "Summarize why gravity keeps planets in orbit",
    ]
}

fn parse_cycles() -> usize {
    std::env::var("LEARNING_DEMO_CYCLES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|cycles| *cycles > 0)
        .unwrap_or(20)
}

fn parse_target_rouge() -> f64 {
    std::env::var("LEARNING_DEMO_TARGET_ROUGE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|score| *score > 0.0)
        .unwrap_or(0.7)
}

fn adapter_path() -> PathBuf {
    let raw = std::env::var("LORA_ADAPTER_PATH")
        .unwrap_or_else(|_| "lora_weights.safetensors".to_string());
    PathBuf::from(raw)
}

fn log_path() -> PathBuf {
    let raw =
        std::env::var("LEARNING_DEMO_LOG").unwrap_or_else(|_| "learning_demo_log.tsv".to_string());
    PathBuf::from(raw)
}

#[tokio::main]
async fn main() -> Result<()> {
    if tracing::dispatcher::has_been_set() {
        warn!("Tracing subscriber already initialised; continuing");
    } else {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .without_time()
            .init();
    }

    let mut args = CliArgs::default();
    args.hardware = HardwareProfile::H200;

    info!("Initialising pipeline for QLoRA learning demo");
    let mut pipeline = Pipeline::initialise(args).await?;

    let adapter_path = adapter_path();
    if adapter_path.exists() {
        info!(adapter = %adapter_path.display(), "Reloading existing LoRA adapter");
        pipeline
            .load_lora_adapter(&adapter_path)
            .await
            .context("failed to load existing LoRA adapter")?;
    }

    let log_path = log_path();
    let mut log_file = File::create(&log_path)
        .with_context(|| format!("unable to create log file at {}", log_path.display()))?;
    writeln!(
        log_file,
        "cycle\tprompt\trouge\tentropy\tthreat\thealing\tqlora_updates\tlatency_ms"
    )?;

    let prompts = default_prompts();
    let total_cycles = parse_cycles();
    let rouge_threshold = parse_target_rouge();

    let mut rouge_scores = Vec::with_capacity(total_cycles);
    let mut entropy_values = Vec::with_capacity(total_cycles);
    let mut threat_count = 0_usize;
    let mut healing_count = 0_usize;
    let mut qlora_update_total = 0_usize;

    let start = Instant::now();

    for cycle_index in 0..total_cycles {
        let prompt = prompts[cycle_index % prompts.len()];
        info!(
            cycle = cycle_index + 1,
            prompt, "Running prompt through pipeline"
        );

        let cycle_start = Instant::now();
        let result = pipeline
            .process_prompt(prompt)
            .await
            .with_context(|| format!("pipeline execution failed on cycle {}", cycle_index + 1))?;
        let latency_ms = cycle_start.elapsed().as_secs_f64() * 1_000.0;

        let rouge = result.rouge;
        let entropy = result.pad_state.entropy;
        let threat = result.compass.is_threat;
        let healing = result.compass.is_healing;
        let qlora_updates = result.learning.qlora_updates.len();

        rouge_scores.push(rouge);
        entropy_values.push(entropy);
        if threat {
            threat_count += 1;
        }
        if healing {
            healing_count += 1;
        }
        qlora_update_total += qlora_updates;

        writeln!(
            log_file,
            "{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{:.1}",
            cycle_index + 1,
            prompt.replace('\t', " "),
            rouge,
            entropy,
            threat,
            healing,
            qlora_updates,
            latency_ms
        )?;

        info!(
            cycle = cycle_index + 1,
            rouge = format!("{:.3}", rouge),
            entropy = format!("{:.3}", entropy),
            threat,
            healing,
            qlora_updates,
            "Cycle complete"
        );

        let should_persist = threat || rouge < rouge_threshold || qlora_updates > 0;
        if should_persist {
            pipeline
                .save_lora_adapter(&adapter_path)
                .await
                .context("failed to persist LoRA adapter during demo")?;
        }
    }

    pipeline
        .save_lora_adapter(&adapter_path)
        .await
        .context("failed to persist final LoRA adapter")?;

    let elapsed = start.elapsed().as_secs_f64();
    let midpoint = rouge_scores.len() / 2;
    let first_avg = if midpoint > 0 {
        rouge_scores[..midpoint].iter().sum::<f64>() / midpoint as f64
    } else {
        rouge_scores.first().copied().unwrap_or(0.0)
    };
    let second_avg = if rouge_scores.len() > midpoint {
        rouge_scores[midpoint..].iter().sum::<f64>() / (rouge_scores.len() - midpoint) as f64
    } else {
        first_avg
    };
    let avg_entropy = if entropy_values.is_empty() {
        0.0
    } else {
        entropy_values.iter().sum::<f64>() / entropy_values.len() as f64
    };

    println!("=== Learning Demo Summary ===");
    println!("Cycles run: {}", rouge_scores.len());
    println!(
        "Average latency: {:.1} ms",
        (elapsed * 1_000.0) / rouge_scores.len().max(1) as f64
    );
    println!("ROUGE (first half): {:.3}", first_avg);
    println!("ROUGE (second half): {:.3}", second_avg);
    println!("Average entropy: {:.3}", avg_entropy);
    println!("Threat cycles: {}", threat_count);
    println!("Healing cycles: {}", healing_count);
    println!("QLoRA updates observed: {}", qlora_update_total);
    println!("Log written to {}", log_path.display());
    println!(
        "Adapter saved to {} (set LORA_ADAPTER_PATH to change)",
        adapter_path.display()
    );

    Ok(())
}
use anyhow::Result;
use niodoo_real_integrated::config::{CliArgs, RuntimeConfig};
use niodoo_real_integrated::embedding::QwenStatefulEmbedder;
use niodoo_real_integrated::lora_trainer::{LoRAConfig, LoRATrainer};
use niodoo_real_integrated::pipeline::Pipeline;
use niodoo_real_integrated::util::rouge_l;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use tracing::{info, warn};

const LORA_WEIGHTS_PATH: &str = "./lora_weights.safetensors";
const TRAINING_THRESHOLD_ROUGE: f64 = 0.7;
const MIN_SAMPLES_FOR_TRAINING: usize = 3;

/// REAL learning demo that:
/// 1. Trains LoRA on bad responses using ideal answers as targets
/// 2. Saves/loads LoRA weights to persist learning
/// 3. Shows measurable improvement through training loss
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    niodoo_real_integrated::config::prime_environment();

    // Sample prompts with ideal answers (easy to score)
    let prompts_with_answers = vec![
        ("Solve 2x + 3 = 7", "x = 2"),
        ("What is 15 * 4?", "60"),
        ("Find x where x^2 = 16", "x = 4 or x = -4"),
        ("Calculate 100 / 5", "20"),
        ("What is 7 + 8?", "15"),
        ("Solve 3x - 5 = 10", "x = 5"),
        ("What is 12 * 3?", "36"),
        ("Find x where 2x = 18", "x = 9"),
        ("Calculate 50 + 25", "75"),
        ("What is 9 * 9?", "81"),
        ("Solve x + 5 = 12", "x = 7"),
        ("What is 20 / 4?", "5"),
        ("Find x where x - 3 = 8", "x = 11"),
        ("Calculate 6 * 7", "42"),
        ("What is 30 + 15?", "45"),
        ("Solve 4x = 20", "x = 5"),
        ("What is 64 / 8?", "8"),
        ("Find x where x^2 = 25", "x = 5 or x = -5"),
        ("Calculate 11 * 11", "121"),
        ("What is 100 - 25?", "75"),
    ];

    // Initialize pipeline
    let args = CliArgs::default();
    let mut pipeline = Pipeline::initialise(args).await?;

    // Initialize embedder for creating training samples
    let config = RuntimeConfig::load(&CliArgs::default())?;
    let mut embedder = QwenStatefulEmbedder::new(
        &config.ollama_endpoint,
        &config.embedding_model_name,
        config.qdrant_vector_dim,
        config.embedding_max_chars,
    )?;
    embedder.set_mock_mode(config.mock_mode);

    // Initialize LoRA trainer - load existing weights if available
    let mut lora_trainer = if Path::new(LORA_WEIGHTS_PATH).exists() {
        info!("Loading existing LoRA weights from {}", LORA_WEIGHTS_PATH);
        match LoRATrainer::load_adapter(LORA_WEIGHTS_PATH) {
            Ok(trainer) => {
                info!("Successfully loaded LoRA weights");
                trainer
            }
            Err(e) => {
                warn!("Failed to load LoRA weights: {}, creating new trainer", e);
                LoRATrainer::with_config(LoRAConfig {
                    rank: 8,
                    alpha: 16.0,
                    input_dim: config.qdrant_vector_dim,
                    output_dim: config.qdrant_vector_dim,
                })?
            }
        }
    } else {
        info!("No existing LoRA weights found, creating new trainer");
        LoRATrainer::with_config(LoRAConfig {
            rank: 8,
            alpha: 16.0,
            input_dim: config.qdrant_vector_dim,
            output_dim: config.qdrant_vector_dim,
        })?
    };

    // Accumulate training samples (prompt_embedding, ideal_answer_embedding)
    let mut training_buffer: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

    // Create log file
    let mut log_file = File::create("learning_log.txt")?;
    writeln!(log_file, "=== NIODOO REAL Learning Demo ===")?;
    writeln!(log_file, "Cycles: {}", prompts_with_answers.len())?;
    writeln!(log_file, "LoRA weights: {}", LORA_WEIGHTS_PATH)?;
    writeln!(log_file, "")?;

    let mut rouge_scores = Vec::new();
    let mut training_losses = Vec::new();
    let mut threat_counts = Vec::new();
    let start = Instant::now();

    println!("\n=== NIODOO REAL Learning Demo ===");
    println!("Training LoRA on bad responses with ideal answers as targets");
    println!("Running {} cycles...\n", prompts_with_answers.len());

    for (cycle_idx, (prompt, ideal_answer)) in prompts_with_answers.iter().enumerate() {
        let cycle_num = cycle_idx + 1;
        println!("\n--- Cycle {} ---", cycle_num);
        println!("Prompt: {}", prompt);

        let cycle_start = Instant::now();

        // Process prompt through full pipeline
        let cycle = pipeline.process_prompt(prompt).await?;

        let latency = cycle_start.elapsed().as_secs_f64() * 1000.0;

        // Score response against ideal answer
        let rouge = rouge_l(&cycle.hybrid_response, ideal_answer);
        rouge_scores.push(rouge);

        // Track threats
        let was_threat = cycle.compass.is_threat;
        threat_counts.push(if was_threat { 1 } else { 0 });

        println!("Response: {}", cycle.hybrid_response);
        println!("ROUGE vs ideal: {:.3}", rouge);
        println!("Threat detected: {}", was_threat);
        println!("Entropy: {:.3}", cycle.pad_state.entropy);
        println!("Latency: {:.0}ms", latency);

        // REAL TRAINING: If response is bad (low ROUGE), create training sample
        if rouge < TRAINING_THRESHOLD_ROUGE {
            println!("⚠️  Low ROUGE detected - creating training sample...");

            // Get embeddings for prompt and ideal answer
            let prompt_emb = embedder.embed(prompt).await?;
            let ideal_emb = embedder.embed(ideal_answer).await?;

            // Add to training buffer
            training_buffer.push((prompt_emb, ideal_emb));
            println!(
                "  Added training sample (buffer size: {})",
                training_buffer.len()
            );
        }

        // Train LoRA when we have enough samples
        if training_buffer.len() >= MIN_SAMPLES_FOR_TRAINING {
            println!("\n🔥 TRAINING LoRA on {} samples...", training_buffer.len());
            let train_start = Instant::now();

            match lora_trainer.train(&training_buffer, 3, 1e-3_f32) {
                Ok(loss) => {
                    let train_time = train_start.elapsed().as_secs_f64() * 1000.0;
                    training_losses.push(loss);
                    println!(
                        "  ✓ Training complete: loss={:.6}, time={:.0}ms",
                        loss, train_time
                    );

                    // Save weights after training
                    match lora_trainer.save_adapter(LORA_WEIGHTS_PATH) {
                        Ok(_) => println!("  ✓ Saved LoRA weights to {}", LORA_WEIGHTS_PATH),
                        Err(e) => warn!("Failed to save LoRA weights: {}", e),
                    }

                    // Clear buffer (keep last sample for continuity)
                    let last = training_buffer.pop();
                    training_buffer.clear();
                    if let Some(sample) = last {
                        training_buffer.push(sample);
                    }
                }
                Err(e) => {
                    warn!("LoRA training failed: {}", e);
                }
            }
        }

        // Log to file
        writeln!(
            log_file,
            "Cycle {}: prompt='{}', response='{}', ideal='{}', rouge={:.3}, threat={}, entropy={:.3}, latency={:.0}ms",
            cycle_num,
            prompt,
            cycle.hybrid_response,
            ideal_answer,
            rouge,
            was_threat,
            cycle.pad_state.entropy,
            latency
        )?;

        // Show progress every 5 cycles
        if cycle_num % 5 == 0 {
            let recent_rouge: f64 = rouge_scores.iter().rev().take(5).sum::<f64>() / 5.0;
            let early_count = 5.min(rouge_scores.len());
            let early_rouge: f64 = if early_count > 0 {
                rouge_scores.iter().take(early_count).sum::<f64>() / early_count as f64
            } else {
                0.0
            };

            println!("\n>>> Progress Check <<<");
            println!("Early cycles avg ROUGE: {:.3}", early_rouge);
            println!("Recent cycles avg ROUGE: {:.3}", recent_rouge);
            if recent_rouge > early_rouge {
                let improvement = ((recent_rouge - early_rouge) / early_rouge.max(0.01)) * 100.0;
                println!("✨ Improvement: +{:.1}%", improvement);
            }
            if !training_losses.is_empty() {
                let avg_loss: f64 = training_losses.iter().map(|&l| l as f64).sum::<f64>()
                    / training_losses.len() as f64;
                println!("Avg training loss: {:.6}", avg_loss);
            }
            println!();
        }
    }

    // Final summary
    let total_time = start.elapsed().as_secs_f64();
    let avg_rouge = rouge_scores.iter().sum::<f64>() / rouge_scores.len() as f64;
    let early_count = 5.min(rouge_scores.len());
    let early_avg = if early_count > 0 {
        rouge_scores.iter().take(early_count).sum::<f64>() / early_count as f64
    } else {
        0.0
    };
    let late_count = 5.min(rouge_scores.len());
    let late_avg = if late_count > 0 {
        rouge_scores.iter().rev().take(late_count).sum::<f64>() / late_count as f64
    } else {
        0.0
    };
    let total_threats: usize = threat_counts.iter().sum();
    let total_trainings = training_losses.len();

    println!("\n=== Learning Demo Complete ===");
    println!("Total cycles: {}", prompts_with_answers.len());
    println!("Total time: {:.1}s", total_time);
    println!("\nROUGE Scores:");
    println!("  Early (first 5): {:.3}", early_avg);
    println!("  Late (last 5): {:.3}", late_avg);
    println!("  Overall avg: {:.3}", avg_rouge);

    if late_avg > early_avg {
        let improvement = ((late_avg - early_avg) / early_avg.max(0.01)) * 100.0;
        println!("  🚀 Improvement: +{:.1}%", improvement);
    }

    println!("\nLoRA Training:");
    println!("  Training sessions: {}", total_trainings);
    if !training_losses.is_empty() {
        let first_loss = training_losses[0] as f64;
        let last_loss = training_losses[training_losses.len() - 1] as f64;
        let avg_loss: f64 =
            training_losses.iter().map(|&l| l as f64).sum::<f64>() / training_losses.len() as f64;
        println!("  First loss: {:.6}", first_loss);
        println!("  Last loss: {:.6}", last_loss);
        println!("  Avg loss: {:.6}", avg_loss);
        if last_loss < first_loss {
            let loss_improvement = ((first_loss - last_loss) / first_loss.max(1e-6)) * 100.0;
            println!("  ✓ Loss decreased by {:.1}%", loss_improvement);
        }
    }

    println!("\nSelf-Healing:");
    println!("  Threats detected: {}", total_threats);

    println!("\nLoRA weights saved to: {}", LORA_WEIGHTS_PATH);
    println!("Run again to load weights and continue learning!");

    // Write summary to log
    writeln!(log_file, "")?;
    writeln!(log_file, "=== Summary ===")?;
    writeln!(log_file, "Total cycles: {}", prompts_with_answers.len())?;
    writeln!(log_file, "Total time: {:.1}s", total_time)?;
    writeln!(log_file, "Early avg ROUGE: {:.3}", early_avg)?;
    writeln!(log_file, "Late avg ROUGE: {:.3}", late_avg)?;
    writeln!(log_file, "Overall avg ROUGE: {:.3}", avg_rouge)?;
    writeln!(log_file, "Training sessions: {}", total_trainings)?;
    if !training_losses.is_empty() {
        writeln!(log_file, "Training losses: {:?}", training_losses)?;
    }
    writeln!(log_file, "Threats detected: {}", total_threats)?;

    Ok(())
}
