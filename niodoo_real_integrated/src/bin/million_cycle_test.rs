use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::info;

use niodoo_real_integrated::config::{CliArgs, HardwareProfile, OutputFormat};
use niodoo_real_integrated::pipeline::{Pipeline, PipelineCycle};

#[derive(Parser, Debug)]
struct MillionTestArgs {
    /// Total number of tests to run
    #[arg(long, default_value_t = 1_000_000)]
    pub count: usize,

    /// Number of parallel workers
    #[arg(long, default_value_t = 128)]
    pub workers: usize,

    /// Batch size per worker
    #[arg(long, default_value_t = 100)]
    pub batch_size: usize,

    /// Output directory for results
    #[arg(long, default_value = "logs/million_cycle_test")]
    pub output_dir: String,

    /// Hardware profile (beelink, 5080q, or h200)
    #[arg(long, default_value = "h200")]
    pub hardware: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestResult {
    id: usize,
    prompt: String,
    entropy: f64,
    rouge: f64,
    latency_ms: f64,
    is_threat: bool,
    is_healing: bool,
    generation_source: String,
    failure_type: String,
    embeddings_cache_hit: bool,
    torus_time_ms: f64,
    tcs_time_ms: f64,
    erag_time_ms: f64,
    generation_time_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct TestSummary {
    total_tests: usize,
    completed: usize,
    failed: usize,
    avg_entropy: f64,
    entropy_std: f64,
    avg_rouge: f64,
    avg_latency_ms: f64,
    p50_latency_ms: f64,
    p95_latency_ms: f64,
    p99_latency_ms: f64,
    threat_rate: f64,
    healing_rate: f64,
    cache_hit_rate: f64,
    total_time_secs: f64,
    throughput_per_sec: f64,
    torus_bottleneck_ms: f64,
    tcs_bottleneck_ms: f64,
    erag_bottleneck_ms: f64,
    generation_bottleneck_ms: f64,
}

struct PipelinePool {
    pipelines: Vec<Arc<Mutex<Pipeline>>>,
}

impl PipelinePool {
    async fn new(size: usize, hardware: HardwareProfile) -> Result<Self> {
        let mut pipelines = Vec::with_capacity(size);

        for i in 0..size {
            let args = CliArgs {
                hardware,
                prompt: None,
                prompt_file: None,
                swarm: 1,
                output: OutputFormat::Csv,
                config: None,
            };

            let pipeline = Pipeline::initialise(args)
                .await
                .with_context(|| format!("Failed to initialize pipeline worker {}", i))?;

            pipelines.push(Arc::new(Mutex::new(pipeline)));

            if (i + 1) % 10 == 0 {
                info!("Initialized {}/{} pipeline workers", i + 1, size);
            }
        }

        info!("Pipeline pool initialized with {} workers", size);
        Ok(Self { pipelines })
    }

    async fn process_prompt(&self, worker_id: usize, prompt: &str) -> Result<PipelineCycle> {
        let pipeline = &self.pipelines[worker_id % self.pipelines.len()];
        let mut pipeline_guard = pipeline.lock().await;
        pipeline_guard.process_prompt(prompt).await
    }
}

fn generate_test_prompts(count: usize) -> Vec<String> {
    let base_templates = vec![
        "Write a Python function to solve: {}",
        "Explain the concept of {} in detail",
        "Design an algorithm for {}",
        "Analyze the relationship between {} and {}",
        "Create a system to handle {} efficiently",
        "What are the implications of {} for future technology?",
        "How would you implement {} using modern best practices?",
        "Provide a mathematical analysis of {}",
        "Discuss the trade-offs involved in {}",
        "Generate a comprehensive solution for {}",
    ];

    let topics = vec![
        "parallel processing",
        "memory management",
        "distributed systems",
        "machine learning",
        "topological data analysis",
        "consciousness modeling",
        "neural networks",
        "quantum computing",
        "information theory",
        "computational complexity",
        "optimization algorithms",
        "graph theory",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "embeddings",
        "attention mechanisms",
        "transformer architecture",
        "persistent homology",
        "knot theory",
        "manifold learning",
        "entropy measurement",
        "signal processing",
        "pattern recognition",
    ];

    let prompts: Vec<String> = (0..count)
        .map(|i| {
            let template = &base_templates[i % base_templates.len()];
            let topic1 = &topics[i % topics.len()];
            let topic2 = &topics[(i * 7) % topics.len()];

            if template.contains("{}") && template.matches("{}").count() == 2 {
                template.replace("{}", topic1).replace("{}", topic2)
            } else {
                template.replace("{}", topic1)
            }
        })
        .collect();

    prompts
}

fn calculate_percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (percentile.clamp(0.0, 1.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank]
}

async fn run_parallel_test(args: MillionTestArgs) -> Result<()> {
    let overall_start = Instant::now();

    info!("🚀 Starting 1M-cycle NIODOO test");
    info!(
        "Configuration: count={}, workers={}, batch_size={}",
        args.count, args.workers, args.batch_size
    );

    // Determine hardware profile
    let hardware = match args.hardware.as_str() {
        "beelink" => HardwareProfile::Beelink,
        "5080q" | "5080-q" => HardwareProfile::Laptop5080Q,
        "h200" | "H200" => HardwareProfile::H200,
        _ => {
            eprintln!(
                "Unknown hardware profile: {}, defaulting to h200",
                args.hardware
            );
            HardwareProfile::H200
        }
    };

    // Initialize pipeline pool
    info!(
        "Initializing pipeline pool with {} workers...",
        args.workers
    );
    let pool = PipelinePool::new(args.workers, hardware).await?;

    // Generate test prompts
    info!("Generating {} test prompts...", args.count);
    let prompts = generate_test_prompts(args.count);

    // Parallel processing
    info!("Starting parallel test execution...");
    let results: Vec<Result<TestResult>> = prompts
        .par_chunks(args.batch_size)
        .enumerate()
        .map(|(batch_idx, batch)| {
            let worker_id = batch_idx % args.workers;

            batch
                .iter()
                .enumerate()
                .map(|(local_idx, prompt)| {
                    let test_id = batch_idx * args.batch_size + local_idx;

                    // Synchronous processing block
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    let start = Instant::now();

                    let result =
                        rt.block_on(async { pool.process_prompt(worker_id, prompt).await });

                    let latency = start.elapsed().as_millis() as f64;

                    match result {
                        Ok(cycle) => Ok(TestResult {
                            id: test_id,
                            prompt: prompt.clone(),
                            entropy: cycle.entropy,
                            rouge: cycle.rouge,
                            latency_ms: latency,
                            is_threat: cycle.compass.is_threat,
                            is_healing: cycle.compass.is_healing,
                            generation_source: cycle.generation.source.clone(),
                            failure_type: cycle.failure.clone(),
                            embeddings_cache_hit: cycle.stage_timings.embedding_ms < 1.0,
                            torus_time_ms: cycle.stage_timings.torus_ms,
                            tcs_time_ms: cycle.stage_timings.tcs_ms,
                            erag_time_ms: cycle.stage_timings.erag_ms,
                            generation_time_ms: cycle.stage_timings.generation_ms,
                        }),
                        Err(e) => {
                            eprintln!("Test {} failed: {}", test_id, e);
                            Err(e)
                        }
                    }
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect();

    // Analyze results
    let mut completed = Vec::new();
    let mut failed = 0;

    for result in results {
        match result {
            Ok(res) => completed.push(res),
            Err(_) => failed += 1,
        }
    }

    info!("Completed: {}, Failed: {}", completed.len(), failed);

    // Calculate statistics
    let entropies: Vec<f64> = completed.iter().map(|r| r.entropy).collect();
    let rouges: Vec<f64> = completed.iter().map(|r| r.rouge).collect();
    let latencies: Vec<f64> = completed.iter().map(|r| r.latency_ms).collect();
    let threats: usize = completed.iter().filter(|r| r.is_threat).count();
    let healings: usize = completed.iter().filter(|r| r.is_healing).count();
    let cache_hits: usize = completed.iter().filter(|r| r.embeddings_cache_hit).count();

    let avg_entropy = entropies.iter().sum::<f64>() / entropies.len() as f64;
    let entropy_std = (entropies
        .iter()
        .map(|&e| (e - avg_entropy).powi(2))
        .sum::<f64>()
        / entropies.len() as f64)
        .sqrt();

    let avg_rouge = rouges.iter().sum::<f64>() / rouges.len() as f64;
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;

    let torus_times: Vec<f64> = completed.iter().map(|r| r.torus_time_ms).collect();
    let tcs_times: Vec<f64> = completed.iter().map(|r| r.tcs_time_ms).collect();
    let erag_times: Vec<f64> = completed.iter().map(|r| r.erag_time_ms).collect();
    let gen_times: Vec<f64> = completed.iter().map(|r| r.generation_time_ms).collect();

    let torus_bottleneck = torus_times.iter().sum::<f64>() / torus_times.len() as f64;
    let tcs_bottleneck = tcs_times.iter().sum::<f64>() / tcs_times.len() as f64;
    let erag_bottleneck = erag_times.iter().sum::<f64>() / erag_times.len() as f64;
    let gen_bottleneck = gen_times.iter().sum::<f64>() / gen_times.len() as f64;

    let total_time = overall_start.elapsed().as_secs_f64();
    let throughput = completed.len() as f64 / total_time;

    let summary = TestSummary {
        total_tests: args.count,
        completed: completed.len(),
        failed,
        avg_entropy,
        entropy_std,
        avg_rouge,
        avg_latency_ms: avg_latency,
        p50_latency_ms: calculate_percentile(&latencies, 0.50),
        p95_latency_ms: calculate_percentile(&latencies, 0.95),
        p99_latency_ms: calculate_percentile(&latencies, 0.99),
        threat_rate: threats as f64 / completed.len() as f64 * 100.0,
        healing_rate: healings as f64 / completed.len() as f64 * 100.0,
        cache_hit_rate: cache_hits as f64 / completed.len() as f64 * 100.0,
        total_time_secs: total_time,
        throughput_per_sec: throughput,
        torus_bottleneck_ms: torus_bottleneck,
        tcs_bottleneck_ms: tcs_bottleneck,
        erag_bottleneck_ms: erag_bottleneck,
        generation_bottleneck_ms: gen_bottleneck,
    };

    // Print summary
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🎯 Million-Cycle Test Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Total Tests: {}", summary.total_tests);
    println!(
        "Completed: {} ({:.1}%)",
        summary.completed,
        summary.completed as f64 / summary.total_tests as f64 * 100.0
    );
    println!(
        "Failed: {} ({:.1}%)",
        summary.failed,
        summary.failed as f64 / summary.total_tests as f64 * 100.0
    );
    println!("\n📊 Performance Metrics:");
    println!(
        "  Average Entropy: {:.3} ± {:.3}",
        summary.avg_entropy, summary.entropy_std
    );
    println!("  Average ROUGE-L: {:.3}", summary.avg_rouge);
    println!("  Average Latency: {:.1}ms", summary.avg_latency_ms);
    println!("  P50 Latency: {:.1}ms", summary.p50_latency_ms);
    println!("  P95 Latency: {:.1}ms", summary.p95_latency_ms);
    println!("  P99 Latency: {:.1}ms", summary.p99_latency_ms);
    println!("\n🧠 Consciousness Metrics:");
    println!("  Threat Rate: {:.1}%", summary.threat_rate);
    println!("  Healing Rate: {:.1}%", summary.healing_rate);
    println!("  Cache Hit Rate: {:.1}%", summary.cache_hit_rate);
    println!("\n⚡ Throughput:");
    println!("  Total Time: {:.1}s", summary.total_time_secs);
    println!("  Throughput: {:.1} tests/sec", summary.throughput_per_sec);
    println!("\n🔍 Bottleneck Analysis:");
    println!("  Torus Projection: {:.1}ms", summary.torus_bottleneck_ms);
    println!("  TCS Analysis: {:.1}ms", summary.tcs_bottleneck_ms);
    println!("  ERAG Retrieval: {:.1}ms", summary.erag_bottleneck_ms);
    println!("  Generation: {:.1}ms", summary.generation_bottleneck_ms);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Save results if requested
    if !args.output_dir.is_empty() {
        std::fs::create_dir_all(&args.output_dir)?;

        let summary_path = format!("{}/summary.json", args.output_dir);
        let summary_json = serde_json::to_string_pretty(&summary)?;
        std::fs::write(&summary_path, summary_json)?;
        info!("Summary saved to: {}", summary_path);

        // Sample first 1000 results for CSV export
        let sample: Vec<&TestResult> = completed.iter().take(1000).collect();
        let csv_path = format!("{}/sample_results.csv", args.output_dir);
        let mut writer = csv::Writer::from_path(&csv_path)?;

        for result in sample {
            writer.serialize(result)?;
        }

        writer.flush()?;
        info!("Sample results saved to: {}", csv_path);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Load environment
    niodoo_real_integrated::config::prime_environment();
    niodoo_real_integrated::config::init();

    let args = MillionTestArgs::parse();
    run_parallel_test(args).await
}
