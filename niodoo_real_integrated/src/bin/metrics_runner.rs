//! Metrics Runner for Validation Framework
//! 
//! CLI tool for running performance benchmarks, capturing baselines, and generating validation reports.
//! Supports load testing, baseline capture, and cognitive benchmark execution.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use niodoo_real_integrated::config::CliArgs;
use niodoo_real_integrated::metrics::metrics;
use niodoo_real_integrated::pipeline::Pipeline;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as AsyncMutex;
use tokio::time::sleep;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "metrics_runner")]
#[command(about = "NIODOO validation metrics runner for performance and cognitive benchmarks")]
struct Args {
    /// Test scenario to execute
    #[arg(long, value_enum, default_value = "load_test")]
    scenario: Scenario,

    /// Number of concurrent users for load test
    #[arg(long, default_value = "16")]
    concurrent_users: usize,

    /// Duration of load test in seconds
    #[arg(long, default_value = "60")]
    duration_secs: u64,

    /// Target tokens per request for load test
    #[arg(long, default_value = "2048")]
    target_tokens: usize,

    /// Output file for metrics report (JSON)
    #[arg(long, default_value = "metrics_report.json")]
    output: PathBuf,

    /// Baseline file to compare against (optional)
    #[arg(long)]
    baseline: Option<PathBuf>,

    /// Config file path
    #[arg(long)]
    config: Option<PathBuf>,

    /// Prompt file or single prompt text
    #[arg(long)]
    prompt: Option<String>,

    /// Prompt file path
    #[arg(long)]
    prompt_file: Option<PathBuf>,

    /// Mock mode (disable external services)
    #[arg(long)]
    mock_mode: bool,
}

#[derive(Debug, Clone, ValueEnum)]
enum Scenario {
    /// Load test: simulate concurrent users with target token generation
    LoadTest,
    /// Baseline capture: run full test suite and save golden metrics
    Baseline,
    /// Cognitive baseline: run cognitive benchmarks only
    Cognitive,
}

#[derive(Debug, Serialize, Deserialize)]
struct MetricsReport {
    timestamp: String,
    scenario: String,
    duration_secs: f64,
    concurrent_users: usize,
    latency: LatencyMetrics,
    throughput: ThroughputMetrics,
    quality_slis: QualitySLIMetrics,
    topological: TopologicalMetrics,
    cognitive: Option<CognitiveMetrics>,
    errors: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LatencyMetrics {
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    ttft_p99_ms: Option<f64>,
    stage_latencies: HashMap<String, StageLatency>,
}

#[derive(Debug, Serialize, Deserialize)]
struct StageLatency {
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ThroughputMetrics {
    requests_per_sec: f64,
    tokens_per_sec: f64,
    embeddings_per_sec: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct QualitySLIMetrics {
    tcs_stability_cv: Option<f64>,
    rce_beta_meta_compliance: Option<f64>,
    rce_beta_meta_current: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TopologicalMetrics {
    persistence_entropy: Option<f64>,
    spectral_gap: Option<f64>,
    beta_meta_current: Option<f64>,
    beta_meta_peak: Option<f64>,
    betti_0_median: Option<f64>,
    betti_1_median: Option<f64>,
    betti_2_median: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CognitiveMetrics {
    locomo_f1_single_hop: Option<f64>,
    locomo_f1_multi_hop: Option<f64>,
    locomo_f1_temporal: Option<f64>,
    locomo_f1_adversarial: Option<f64>,
    aqa_bench_success_rate: Option<f64>,
    docpuzzle_process_score: Option<f64>,
    counterbench_accuracy: Option<f64>,
    criticbench_generation: Option<f64>,
    criticbench_critique: Option<f64>,
    criticbench_correction: Option<f64>,
}

struct LoadTestMetrics {
    latencies: Vec<f64>,
    stage_latencies: HashMap<String, Vec<f64>>,
    request_count: Arc<AtomicU64>,
    token_count: Arc<AtomicU64>,
    start_time: Instant,
    errors: Arc<AsyncMutex<Vec<String>>>,
}

impl LoadTestMetrics {
    fn new() -> Self {
        Self {
            latencies: Vec::new(),
            stage_latencies: HashMap::new(),
            request_count: Arc::new(AtomicU64::new(0)),
            token_count: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
            errors: Arc::new(AsyncMutex::new(Vec::new())),
        }
    }

    fn record_latency(&mut self, latency_ms: f64) {
        self.latencies.push(latency_ms);
    }

    fn record_stage_latency(&mut self, stage: String, latency_ms: f64) {
        self.stage_latencies
            .entry(stage)
            .or_insert_with(Vec::new)
            .push(latency_ms);
    }

    fn compute_percentiles(&self, values: &[f64], percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (percentile * (sorted.len() - 1) as f64) as usize;
        sorted[index]
    }

    fn to_report(&self, duration_secs: f64, concurrent_users: usize) -> MetricsReport {
        let metrics_text = metrics().gather().unwrap_or_default();
        
        // Parse Prometheus metrics (simplified - would need proper parser in production)
        let quality_slis = self.extract_quality_slis(&metrics_text);
        let topological = self.extract_topological(&metrics_text);
        let throughput = ThroughputMetrics {
            requests_per_sec: self.request_count.load(Ordering::Relaxed) as f64 / duration_secs,
            tokens_per_sec: self.token_count.load(Ordering::Relaxed) as f64 / duration_secs,
            embeddings_per_sec: self.latencies.len() as f64 / duration_secs,
        };

        let latency = if !self.latencies.is_empty() {
            LatencyMetrics {
                p50_ms: self.compute_percentiles(&self.latencies, 0.50),
                p95_ms: self.compute_percentiles(&self.latencies, 0.95),
                p99_ms: self.compute_percentiles(&self.latencies, 0.99),
                mean_ms: self.latencies.iter().sum::<f64>() / self.latencies.len() as f64,
                min_ms: self.latencies.iter().copied().fold(f64::INFINITY, f64::min),
                max_ms: self.latencies.iter().copied().fold(0.0, f64::max),
                ttft_p99_ms: None, // Would extract from vLLM metrics
                stage_latencies: self
                    .stage_latencies
                    .iter()
                    .map(|(stage, latencies)| {
                        (
                            stage.clone(),
                            StageLatency {
                                p50_ms: self.compute_percentiles(latencies, 0.50),
                                p95_ms: self.compute_percentiles(latencies, 0.95),
                                p99_ms: self.compute_percentiles(latencies, 0.99),
                            },
                        )
                    })
                    .collect(),
            }
        } else {
            LatencyMetrics {
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                mean_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                ttft_p99_ms: None,
                stage_latencies: HashMap::new(),
            }
        };

        MetricsReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            scenario: "load_test".to_string(),
            duration_secs,
            concurrent_users,
            latency,
            throughput,
            quality_slis,
            topological,
            cognitive: None,
            errors: self.errors.blocking_lock().clone(),
        }
    }

    fn extract_quality_slis(&self, metrics_text: &str) -> QualitySLIMetrics {
        let mut result = QualitySLIMetrics {
            tcs_stability_cv: None,
            rce_beta_meta_compliance: None,
            rce_beta_meta_current: None,
        };

        for line in metrics_text.lines() {
            if line.starts_with("niodoo_quality_sli_tcs_stability_cv") {
                if let Some(value) = self.extract_gauge_value(line) {
                    result.tcs_stability_cv = Some(value);
                }
            } else if line.starts_with("niodoo_quality_sli_rce_beta_meta_compliance") {
                if let Some(value) = self.extract_gauge_value(line) {
                    result.rce_beta_meta_compliance = Some(value);
                }
            } else if line.starts_with("niodoo_rce_beta_meta_current") {
                if let Some(value) = self.extract_gauge_value(line) {
                    result.rce_beta_meta_current = Some(value);
                }
            }
        }

        result
    }

    fn extract_topological(&self, metrics_text: &str) -> TopologicalMetrics {
        let mut result = TopologicalMetrics {
            persistence_entropy: None,
            spectral_gap: None,
            beta_meta_current: None,
            beta_meta_peak: None,
            betti_0_median: None,
            betti_1_median: None,
            betti_2_median: None,
        };

        for line in metrics_text.lines() {
            if line.starts_with("niodoo_rce_persistence_entropy") {
                if let Some(value) = self.extract_gauge_value(line) {
                    result.persistence_entropy = Some(value);
                }
            } else if line.starts_with("niodoo_rce_laplacian_spectral_gap") {
                if let Some(value) = self.extract_gauge_value(line) {
                    result.spectral_gap = Some(value);
                }
            } else if line.starts_with("niodoo_rce_beta_meta_current") {
                if let Some(value) = self.extract_gauge_value(line) {
                    result.beta_meta_current = Some(value);
                }
            } else if line.starts_with("niodoo_rce_beta_meta_peak") {
                if let Some(value) = self.extract_gauge_value(line) {
                    result.beta_meta_peak = Some(value);
                }
            }
        }

        result
    }

    fn extract_gauge_value(&self, line: &str) -> Option<f64> {
        line.split_whitespace()
            .last()
            .and_then(|v| v.parse::<f64>().ok())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("🚀 Starting metrics runner with scenario: {:?}", args.scenario);

    // Build CLI args for pipeline initialization
    let mut cli_args = CliArgs::default();
    if let Some(config_path) = &args.config {
        cli_args.config = Some(config_path.to_string_lossy().to_string());
    }
    if let Some(prompt) = &args.prompt {
        cli_args.prompt = Some(prompt.clone());
    }
    if let Some(prompt_file) = &args.prompt_file {
        cli_args.prompt_file = Some(prompt_file.to_string_lossy().to_string());
    }
    if args.mock_mode {
        std::env::set_var("MOCK_MODE", "true");
    }

    // Initialize pipeline once
    info!("Initializing pipeline...");
    let mut pipeline = Pipeline::initialise(cli_args.clone())
        .await
        .context("Failed to initialize pipeline")?;
    info!("✅ Pipeline initialized");

    let report = match args.scenario {
        Scenario::LoadTest => {
            run_load_test(pipeline, args.concurrent_users, args.duration_secs, args.target_tokens).await?
        }
        Scenario::Baseline => {
            run_baseline(pipeline).await?
        }
        Scenario::Cognitive => {
            run_cognitive_baseline(pipeline).await?
        }
    };

    // Save report
    let report_json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&args.output, report_json)?;
    info!("✅ Metrics report saved to: {}", args.output.display());

    // Compare with baseline if provided
    if let Some(baseline_path) = &args.baseline {
        compare_with_baseline(&report, baseline_path)?;
    }

    Ok(())
}

async fn run_load_test(
    pipeline: Pipeline,
    concurrent_users: usize,
    duration_secs: u64,
    target_tokens: usize,
) -> Result<MetricsReport> {
    info!(
        "Running load test: {} concurrent users for {} seconds",
        concurrent_users, duration_secs
    );

    let metrics = Arc::new(AsyncMutex::new(LoadTestMetrics::new()));
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(duration_secs);

    // Generate test prompts designed to elicit target_tokens
    let test_prompts = generate_load_test_prompts(concurrent_users * 2, target_tokens);

    // Use a mutex to serialize pipeline access (pipeline is not thread-safe)
    let pipeline_mutex = Arc::new(AsyncMutex::new(pipeline));
    let mut handles = Vec::new();
    
    for user_id in 0..concurrent_users {
        let pipeline_clone = pipeline_mutex.clone();
        let metrics_clone = metrics.clone();
        let prompts_clone = test_prompts.clone();
        let request_count = metrics_clone.lock().await.request_count.clone();
        let errors_clone = metrics_clone.lock().await.errors.clone();

        let handle = tokio::spawn(async move {
            let mut prompt_idx = user_id;
            while Instant::now() < end_time {
                let prompt = &prompts_clone[prompt_idx % prompts_clone.len()];
                let req_start = Instant::now();

                let result = {
                    let mut p = pipeline_clone.lock().await;
                    p.process_prompt(prompt).await
                };

                match result {
                    Ok(cycle) => {
                        let latency_ms = req_start.elapsed().as_secs_f64() * 1000.0;
                        request_count.fetch_add(1, Ordering::Relaxed);
                        
                        let mut m = metrics_clone.lock().await;
                        m.record_latency(latency_ms);
                        
                        // Record stage latencies
                        m.record_stage_latency(
                            "embedding".to_string(),
                            cycle.stage_timings.embedding_ms,
                        );
                        m.record_stage_latency(
                            "erag".to_string(),
                            cycle.stage_timings.erag_ms,
                        );
                        if cycle.stage_timings.tcs_ms > 0.0 {
                            m.record_stage_latency(
                                "tcs".to_string(),
                                cycle.stage_timings.tcs_ms,
                            );
                        }
                    }
                    Err(e) => {
                        errors_clone.lock().await.push(format!("User {} error: {}", user_id, e));
                    }
                }

                prompt_idx += concurrent_users;
                sleep(Duration::from_millis(100)).await; // Small delay between requests
            }
        });
        handles.push(handle);
    }

    // Wait for all workers
    for handle in handles {
        let _ = handle.await;
    }

    let actual_duration = start_time.elapsed().as_secs_f64();
    let m = metrics.lock().await;
    Ok(m.to_report(actual_duration, concurrent_users))
}

async fn run_baseline(pipeline: Pipeline) -> Result<MetricsReport> {
    info!("Running baseline capture...");
    
    // Run a small load test as baseline
    run_load_test(pipeline, 16, 60, 2048).await
}

async fn run_cognitive_baseline(_pipeline: Pipeline) -> Result<MetricsReport> {
    info!("Running cognitive baseline...");
    
    // TODO: Implement cognitive benchmark execution
    // For now, return empty cognitive metrics
    Ok(MetricsReport {
        timestamp: chrono::Utc::now().to_rfc3339(),
        scenario: "cognitive".to_string(),
        duration_secs: 0.0,
        concurrent_users: 0,
        latency: LatencyMetrics {
            p50_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            mean_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            ttft_p99_ms: None,
            stage_latencies: HashMap::new(),
        },
        throughput: ThroughputMetrics {
            requests_per_sec: 0.0,
            tokens_per_sec: 0.0,
            embeddings_per_sec: 0.0,
        },
        quality_slis: QualitySLIMetrics {
            tcs_stability_cv: None,
            rce_beta_meta_compliance: None,
            rce_beta_meta_current: None,
        },
        topological: TopologicalMetrics {
            persistence_entropy: None,
            spectral_gap: None,
            beta_meta_current: None,
            beta_meta_peak: None,
            betti_0_median: None,
            betti_1_median: None,
            betti_2_median: None,
        },
        cognitive: Some(CognitiveMetrics {
            locomo_f1_single_hop: None,
            locomo_f1_multi_hop: None,
            locomo_f1_temporal: None,
            locomo_f1_adversarial: None,
            aqa_bench_success_rate: None,
            docpuzzle_process_score: None,
            counterbench_accuracy: None,
            criticbench_generation: None,
            criticbench_critique: None,
            criticbench_correction: None,
        }),
        errors: Vec::new(),
    })
}

fn generate_load_test_prompts(count: usize, target_tokens: usize) -> Vec<String> {
    // Generate prompts designed to elicit approximately target_tokens
    let base_prompt = format!(
        "Write a detailed explanation of {} including background, methodology, examples, and implications. Be thorough and comprehensive.",
        match target_tokens {
            0..=512 => "a simple concept",
            513..=1024 => "a complex topic",
            1025..=2048 => "an advanced subject with multiple perspectives",
            _ => "a comprehensive analysis of multiple interconnected topics",
        }
    );

    (0..count)
        .map(|i| format!("{} [Request #{}]", base_prompt, i))
        .collect()
}

fn compare_with_baseline(report: &MetricsReport, baseline_path: &PathBuf) -> Result<()> {
    let baseline_json = std::fs::read_to_string(baseline_path)?;
    let baseline: MetricsReport = serde_json::from_str(&baseline_json)?;

    info!("Comparing current run with baseline...");
    
    // Simple comparison (would use statistical analysis in production)
    let latency_diff = report.latency.p99_ms - baseline.latency.p99_ms;
    if latency_diff > 100.0 {
        warn!("⚠️  p99 latency increased by {:.2}ms", latency_diff);
    } else {
        info!("✅ p99 latency within acceptable range");
    }

    Ok(())
}

