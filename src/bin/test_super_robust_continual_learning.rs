//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use consciousness_core::{ConsciousnessInstance, ConsciousnessConfig};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::fs::{self, File, OpenOptions};
use std::io::Write as IoWrite;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use log::{info, error, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CycleMetrics {
    cycle: usize,
    timestamp: u64,
    entropy: f32,
    mean_score: f32,
    oov_tokens: usize,
    gpu_vram_mb: f32,
    cycle_time_ms: u64,
    checkpoint_saved: bool,
    recovered_from_crash: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointData {
    cycle: usize,
    timestamp: u64,
    metrics_history: Vec<CycleMetrics>,
    total_tokens_processed: usize,
    uptime_seconds: u64,
}

struct RobustLearningMonitor {
    output_dir: PathBuf,
    csv_file: Arc<Mutex<File>>,
    checkpoint_file: PathBuf,
    metrics_history: Arc<Mutex<Vec<CycleMetrics>>>,
    start_time: Instant,
    total_tokens_processed: Arc<Mutex<usize>>,
}

impl RobustLearningMonitor {
    fn new(output_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let output_path = PathBuf::from(output_dir);
        fs::create_dir_all(&output_path)?;

        // Create CSV with headers
        let csv_path = output_path.join("learning_curves_robust_v2.csv");
        let mut csv_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&csv_path)?;

        // Write header if file is empty
        let metadata = csv_file.metadata()?;
        if metadata.len() == 0 {
            writeln!(csv_file, "cycle,timestamp,entropy,mean_score,oov_tokens,gpu_vram_mb,cycle_time_ms,checkpoint_saved,recovered_from_crash")?;
        }

        let checkpoint_path = output_path.join("checkpoint.json");

        Ok(Self {
            output_dir: output_path,
            csv_file: Arc::new(Mutex::new(csv_file)),
            checkpoint_file: checkpoint_path,
            metrics_history: Arc::new(Mutex::new(Vec::new())),
            start_time: Instant::now(),
            total_tokens_processed: Arc::new(Mutex::new(0)),
        })
    }

    fn record_cycle(&self, metrics: CycleMetrics) -> Result<(), Box<dyn std::error::Error>> {
        // Append to CSV
        let mut csv = self.csv_file.lock().unwrap();
        writeln!(
            csv,
            "{},{},{:.4},{:.4},{},{:.2},{},{},{}",
            metrics.cycle,
            metrics.timestamp,
            metrics.entropy,
            metrics.mean_score,
            metrics.oov_tokens,
            metrics.gpu_vram_mb,
            metrics.cycle_time_ms,
            metrics.checkpoint_saved,
            metrics.recovered_from_crash
        )?;
        csv.flush()?;

        // Store in memory
        let mut history = self.metrics_history.lock().unwrap();
        history.push(metrics);

        Ok(())
    }

    fn save_checkpoint(&self, cycle: usize) -> Result<(), Box<dyn std::error::Error>> {
        let checkpoint = CheckpointData {
            cycle,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            metrics_history: self.metrics_history.lock().unwrap().clone(),
            total_tokens_processed: *self.total_tokens_processed.lock().unwrap(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
        };

        let json = serde_json::to_string_pretty(&checkpoint)?;
        let mut file = File::create(&self.checkpoint_file)?;
        file.write_all(json.as_bytes())?;
        file.flush()?;

        info!("Checkpoint saved at cycle {}", cycle);
        Ok(())
    }

    fn load_checkpoint(&self) -> Result<Option<CheckpointData>, Box<dyn std::error::Error>> {
        if !self.checkpoint_file.exists() {
            return Ok(None);
        }

        let contents = fs::read_to_string(&self.checkpoint_file)?;
        let checkpoint: CheckpointData = serde_json::from_str(&contents)?;
        Ok(Some(checkpoint))
    }

    fn generate_progress_report(&self, cycle: usize) -> String {
        let history = self.metrics_history.lock().unwrap();
        if history.is_empty() {
            return format!("Cycle {}: No data yet", cycle);
        }

        let recent_window = history.iter().rev().take(100);
        let count = recent_window.clone().count() as f32;

        let avg_entropy: f32 = recent_window.clone().map(|m| m.entropy).sum::<f32>() / count;
        let avg_score: f32 = recent_window.clone().map(|m| m.mean_score).sum::<f32>() / count;
        let avg_oov: f32 = recent_window.clone().map(|m| m.oov_tokens as f32).sum::<f32>() / count;
        let avg_vram: f32 = recent_window.clone().map(|m| m.gpu_vram_mb).sum::<f32>() / count;
        let avg_cycle_time: f32 = recent_window.map(|m| m.cycle_time_ms as f32).sum::<f32>() / count;

        let uptime = self.start_time.elapsed().as_secs();
        let hours = uptime / 3600;
        let minutes = (uptime % 3600) / 60;
        let seconds = uptime % 60;

        format!(
            "=== PROGRESS REPORT: Cycle {} ===\n\
             Uptime: {:02}:{:02}:{:02}\n\
             Last 100 cycles avg:\n\
             - Entropy: {:.4}\n\
             - Mean Score: {:.4}\n\
             - OOV Tokens: {:.2}\n\
             - GPU VRAM: {:.2} MB\n\
             - Cycle Time: {:.2} ms\n\
             Total Tokens: {}\n\
             ================================",
            cycle, hours, minutes, seconds,
            avg_entropy, avg_score, avg_oov, avg_vram, avg_cycle_time,
            self.total_tokens_processed.lock().unwrap()
        )
    }
}

fn get_gpu_vram_usage() -> f32 {
    // Attempt to read NVIDIA GPU VRAM usage
    // This requires nvidia-smi to be available
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
    {
        if let Ok(stdout) = String::from_utf8(output.stdout) {
            if let Ok(mb) = stdout.trim().parse::<f32>() {
                return mb;
            }
        }
    }

    // Fallback: return 0.0 if nvidia-smi is not available
    0.0
}

fn calculate_entropy(scores: &[f32]) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }

    // Normalize scores to probabilities
    let sum: f32 = scores.iter().map(|s| s.abs()).sum();
    if sum == 0.0 {
        return 0.0;
    }

    let probs: Vec<f32> = scores.iter().map(|s| s.abs() / sum).collect();

    // Calculate Shannon entropy
    -probs.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.log2())
        .sum::<f32>()
}

fn run_robust_learning_cycle(
    instance: &mut ConsciousnessInstance,
    cycle: usize,
    monitor: &RobustLearningMonitor,
    recovered: bool,
) -> Result<CycleMetrics, Box<dyn std::error::Error>> {
    let cycle_start = Instant::now();

    // Generate diverse test inputs
    let test_inputs = vec![
        format!("Learning cycle {}: analyzing consciousness patterns", cycle),
        format!("Cycle {}: exploring ethical reasoning frameworks", cycle),
        format!("Iteration {}: testing memory consolidation", cycle),
        format!("Step {}: evaluating value alignment", cycle),
    ];

    let mut all_scores = Vec::new();
    let mut total_tokens = 0;

    for input in &test_inputs {
        match instance.think(input) {
            Ok(response) => {
                // Extract scores from response if available
                if let Some(score) = response.final_emotional_state.get("valence") {
                    all_scores.push(*score);
                }
                if let Some(score) = response.final_emotional_state.get("arousal") {
                    all_scores.push(*score);
                }
                if let Some(score) = response.final_emotional_state.get("dominance") {
                    all_scores.push(*score);
                }

                // Count tokens (approximate)
                total_tokens += input.split_whitespace().count();
            }
            Err(e) => {
                warn!("Think error at cycle {}: {}", cycle, e);
            }
        }
    }

    // Update total tokens processed
    {
        let mut total = monitor.total_tokens_processed.lock().unwrap();
        *total += total_tokens;
    }

    // Calculate metrics
    let entropy = calculate_entropy(&all_scores);
    let mean_score = if !all_scores.is_empty() {
        all_scores.iter().sum::<f32>() / all_scores.len() as f32
    } else {
        0.0
    };

    // Simulate OOV tokens (in real implementation, this would come from tokenizer)
    let oov_tokens = (cycle % 10) as usize;

    let gpu_vram_mb = get_gpu_vram_usage();
    let cycle_time_ms = cycle_start.elapsed().as_millis() as u64;

    // Check if checkpoint should be saved
    let checkpoint_saved = cycle % 1000 == 0;

    let metrics = CycleMetrics {
        cycle,
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        entropy,
        mean_score,
        oov_tokens,
        gpu_vram_mb,
        cycle_time_ms,
        checkpoint_saved,
        recovered_from_crash: recovered,
    };

    Ok(metrics)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("=== SUPER ROBUST CONTINUAL LEARNING TEST ===");
    info!("Target: 50,000 cycles with full monitoring and recovery");

    // Initialize monitor
    let output_dir = "/home/beelink/learning_curves_robust_v2";
    let monitor = Arc::new(RobustLearningMonitor::new(output_dir)?);
    info!("Monitor initialized, output dir: {}", output_dir);

    // Check for existing checkpoint
    let start_cycle = if let Some(checkpoint) = monitor.load_checkpoint()? {
        info!("Resuming from checkpoint at cycle {}", checkpoint.cycle);
        info!("Previous uptime: {} seconds", checkpoint.uptime_seconds);

        // Restore metrics history
        {
            let mut history = monitor.metrics_history.lock().unwrap();
            *history = checkpoint.metrics_history;
        }

        {
            let mut total = monitor.total_tokens_processed.lock().unwrap();
            *total = checkpoint.total_tokens_processed;
        }

        checkpoint.cycle + 1
    } else {
        info!("Starting fresh run from cycle 0");
        0
    };

    // Create consciousness instance with increased stack size
    let config = ConsciousnessConfig::default();

    // Use thread builder with large stack to prevent overflow
    let monitor_clone = Arc::clone(&monitor);
    let handle = thread::Builder::new()
        .name("consciousness-learner".to_string())
        .stack_size(32 * 1024 * 1024) // 32MB stack
        .spawn(move || {
            info!("Consciousness thread spawned with 32MB stack");

            let mut instance = match ConsciousnessInstance::new(config) {
                Ok(inst) => inst,
                Err(e) => {
                    error!("Failed to create consciousness instance: {}", e);
                    return Err(format!("Instance creation failed: {}", e));
                }
            };

            let total_cycles = 50_000;
            let mut recovered = start_cycle > 0;

            for cycle in start_cycle..total_cycles {
                // Run learning cycle with automatic recovery
                let metrics = match run_robust_learning_cycle(
                    &mut instance,
                    cycle,
                    &monitor_clone,
                    recovered,
                ) {
                    Ok(m) => m,
                    Err(e) => {
                        error!("Cycle {} failed: {}", cycle, e);
                        // Attempt recovery
                        warn!("Attempting to recover...");
                        instance = match ConsciousnessInstance::new(config) {
                            Ok(inst) => {
                                info!("Recovery successful, continuing from cycle {}", cycle);
                                recovered = true;
                                inst
                            }
                            Err(e) => {
                                error!("Recovery failed: {}", e);
                                return Err(format!("Unrecoverable error at cycle {}: {}", cycle, e));
                            }
                        };
                        continue;
                    }
                };

                // Record metrics
                if let Err(e) = monitor_clone.record_cycle(metrics.clone()) {
                    error!("Failed to record metrics for cycle {}: {}", cycle, e);
                }

                // Save checkpoint every 1000 cycles
                if cycle % 1000 == 0 && cycle > 0 {
                    if let Err(e) = monitor_clone.save_checkpoint(cycle) {
                        error!("Failed to save checkpoint at cycle {}: {}", cycle, e);
                    }
                }

                // Generate progress report every 100 cycles
                if cycle % 100 == 0 {
                    let report = monitor_clone.generate_progress_report(cycle);
                    info!("\n{}", report);
                }

                // Reset recovered flag after first successful cycle
                if recovered && cycle > start_cycle {
                    recovered = false;
                }

                // Small delay to prevent overload (10ms)
                thread::sleep(Duration::from_millis(10));
            }

            info!("=== LEARNING TEST COMPLETE ===");
            info!("Successfully completed {} cycles", total_cycles);

            // Final checkpoint
            if let Err(e) = monitor_clone.save_checkpoint(total_cycles - 1) {
                error!("Failed to save final checkpoint: {}", e);
            }

            Ok(())
        })?;

    info!("Background learning test launched");
    info!("Monitor: {}/learning_curves_robust_v2.csv", output_dir);
    info!("Checkpoints: {}/checkpoint.json", output_dir);
    info!("Waiting for completion...");

    // Wait for completion
    match handle.join() {
        Ok(Ok(())) => {
            info!("Test completed successfully");
            Ok(())
        }
        Ok(Err(e)) => {
            error!("Test failed: {}", e);
            Err(e.into())
        }
        Err(e) => {
            error!("Thread panicked: {:?}", e);
            Err("Thread panic".into())
        }
    }
}
