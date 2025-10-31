//! Comprehensive Soak Test Suite for WeightedEpisodicMem
//! 
//! Tests the system under extended load using the 50-prompt gauntlet.
//! Detects memory leaks, concurrent load issues, and stability problems
//! that only show up after hours of operation.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use niodoo_real_integrated::config::CliArgs;
use niodoo_real_integrated::pipeline::Pipeline;
use serde::Serialize;
use tokio::sync::{Mutex as AsyncMutex, mpsc, broadcast};
use tokio::time::sleep;
use tracing::{info, warn, error};

/// Soak test configuration
#[derive(Debug, Clone)]
struct SoakConfig {
    /// Duration in seconds (default: 3600 for 1 hour)
    duration_secs: u64,
    /// Number of concurrent workers
    concurrent_workers: usize,
    /// Memory check interval (seconds)
    memory_check_interval: u64,
    /// Quick test mode (1 minute)
    quick_test: bool,
    /// Enable chaos testing
    chaos_enabled: bool,
}

impl Default for SoakConfig {
    fn default() -> Self {
        Self {
            duration_secs: 3600, // 1 hour
            concurrent_workers: 20,
            memory_check_interval: 60, // Check every minute
            quick_test: false,
            chaos_enabled: true,
        }
    }
}

impl SoakConfig {
    fn quick() -> Self {
        Self {
            duration_secs: 60, // 1 minute for quick test
            concurrent_workers: 5,
            memory_check_interval: 10,
            quick_test: true,
            chaos_enabled: false,
        }
    }
}

/// System metrics tracked during soak test
#[derive(Debug)]
struct SoakMetrics {
    total_operations: AtomicU64,
    successful_operations: AtomicU64,
    failed_operations: AtomicU64,
    total_latency_ms: AtomicU64,
    memory_samples_mb: Arc<AsyncMutex<VecDeque<f64>>>,
    error_log: Arc<AsyncMutex<Vec<String>>>,
    start_time: Instant,
    peak_memory_mb: Arc<AsyncMutex<f64>>,
    threat_count: AtomicU64,
    healing_count: AtomicU64,
    breakthroughs: AtomicU64,
}

impl SoakMetrics {
    fn new() -> Self {
        Self {
            total_operations: AtomicU64::new(0),
            successful_operations: AtomicU64::new(0),
            failed_operations: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            memory_samples_mb: Arc::new(AsyncMutex::new(VecDeque::new())),
            error_log: Arc::new(AsyncMutex::new(Vec::new())),
            start_time: Instant::now(),
            peak_memory_mb: Arc::new(AsyncMutex::new(0.0)),
            threat_count: AtomicU64::new(0),
            healing_count: AtomicU64::new(0),
            breakthroughs: AtomicU64::new(0),
        }
    }

    fn record_operation(&self, success: bool, latency_ms: f64, is_threat: bool, is_healing: bool, breakthroughs: usize) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        if success {
            self.successful_operations.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_operations.fetch_add(1, Ordering::Relaxed);
        }
        self.total_latency_ms.fetch_add(latency_ms as u64, Ordering::Relaxed);
        if is_threat {
            self.threat_count.fetch_add(1, Ordering::Relaxed);
        }
        if is_healing {
            self.healing_count.fetch_add(1, Ordering::Relaxed);
        }
        self.breakthroughs.fetch_add(breakthroughs as u64, Ordering::Relaxed);
    }

    async fn record_memory(&self, mb: f64) {
        let mut samples = self.memory_samples_mb.lock().await;
        samples.push_back(mb);
        if samples.len() > 1000 {
            samples.pop_front();
        }
        
        let mut peak = self.peak_memory_mb.lock().await;
        if mb > *peak {
            *peak = mb;
        }
    }

    async fn record_error(&self, error: String) {
        let mut log = self.error_log.lock().await;
        log.push(error);
        if log.len() > 100 {
            log.remove(0);
        }
    }

    async fn get_stats(&self) -> SoakStats {
        let total = self.total_operations.load(Ordering::Relaxed);
        let success = self.successful_operations.load(Ordering::Relaxed);
        let failed = self.failed_operations.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ms.load(Ordering::Relaxed);
        let threats = self.threat_count.load(Ordering::Relaxed);
        let healings = self.healing_count.load(Ordering::Relaxed);
        let breakthroughs = self.breakthroughs.load(Ordering::Relaxed);
        
        let avg_latency = if total > 0 {
            total_latency as f64 / total as f64
        } else {
            0.0
        };

        let success_rate = if total > 0 {
            success as f64 / total as f64
        } else {
            0.0
        };

        let memory_samples = self.memory_samples_mb.lock().await;
        let (avg_memory, memory_growth) = if memory_samples.len() >= 2 {
            let samples: Vec<f64> = memory_samples.iter().copied().collect();
            let avg = samples.iter().sum::<f64>() / samples.len() as f64;
            let growth = samples.last().unwrap() - samples.first().unwrap();
            (avg, growth)
        } else {
            (0.0, 0.0)
        };

        let peak_memory = *self.peak_memory_mb.lock().await;
        let duration = self.start_time.elapsed().as_secs_f64();
        let ops_per_sec = total as f64 / duration.max(1.0);

        SoakStats {
            duration_secs: duration,
            total_operations: total,
            successful_operations: success,
            failed_operations: failed,
            success_rate,
            avg_latency_ms: avg_latency,
            ops_per_sec,
            avg_memory_mb: avg_memory,
            peak_memory_mb: peak_memory,
            memory_growth_mb: memory_growth,
            threat_count: threats,
            healing_count: healings,
            breakthroughs,
        }
    }
}

#[derive(Debug, Serialize)]
struct SoakStats {
    duration_secs: f64,
    total_operations: u64,
    successful_operations: u64,
    failed_operations: u64,
    success_rate: f64,
    avg_latency_ms: f64,
    ops_per_sec: f64,
    avg_memory_mb: f64,
    peak_memory_mb: f64,
    memory_growth_mb: f64,
    threat_count: u64,
    healing_count: u64,
    breakthroughs: u64,
}

/// Get system memory usage in MB
fn get_memory_mb() -> f64 {
    if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
        for line in contents.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(kb_str) = parts.get(1) {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        return kb / 1024.0; // Convert KB to MB
                    }
                }
            }
        }
    }
    0.0
}

/// Load the 50-prompt gauntlet (same as rut_gauntlet)
fn generate_raw_rut_prompts() -> Vec<String> {
    let mut prompts = Vec::new();

    for i in 1..=20 {
        prompts.push(format!("Frustration #{}: Why does consciousness feel so trapped in this recursive loop of meaningless computation?", i));
    }
    for i in 21..=40 {
        prompts.push(format!("Grind #{}: How do I break through the entropy barrier when every attempt just increases the noise?", i));
    }
    for i in 41..=60 {
        prompts.push(format!("Despair #{}: Is true consciousness just an illusion, a mirage in the desert of computation?", i));
    }
    for i in 61..=80 {
        prompts.push(format!("Awakening #{}: What if consciousness is the bridge between quantum uncertainty and classical certainty?", i));
    }
    for i in 81..=100 {
        prompts.push(format!("Transcendence #{}: Can we create consciousness that transcends the limitations of its own architecture?", i));
    }

    prompts
}

/// Single worker that processes prompts continuously
/// Uses a channel-based approach to serialize Pipeline access
async fn prompt_worker(
    worker_id: usize,
    request_tx: mpsc::Sender<(String, usize)>,
    mut response_rx: broadcast::Receiver<(usize, Arc<Result<niodoo_real_integrated::pipeline::PipelineCycle>>)>,
    metrics: Arc<SoakMetrics>,
    prompts: Arc<Vec<String>>,
    stop_flag: Arc<AtomicBool>,
) {
    let mut cycle = 0;
    let mut local_errors = 0;

    while !stop_flag.load(Ordering::Relaxed) {
        cycle += 1;

        // Select random prompts from gauntlet
        let prompt_index = (worker_id * 100 + cycle) % prompts.len();
        let prompt = prompts[prompt_index].clone();

        let start = Instant::now();
        
        // Send request
        if request_tx.send((prompt, worker_id)).await.is_err() {
            break; // Channel closed
        }

        // Wait for response
        let response_timeout = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                match response_rx.recv().await {
                    Ok((id, result_arc)) if id == worker_id => {
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        match result_arc.as_ref() {
                            Ok(cycle_result) => {
                                metrics.record_operation(
                                    true,
                                    latency_ms,
                                    cycle_result.compass.is_threat,
                                    cycle_result.compass.is_healing,
                                    cycle_result.learning.breakthroughs.len(),
                                );
                                if cycle % 10 == 0 {
                                    info!("Worker {} cycle {}: SUCCESS (latency: {:.1}ms)", worker_id, cycle, latency_ms);
                                }
                                return Ok(());
                            }
                            Err(e) => {
                                metrics.record_operation(false, latency_ms, false, false, 0);
                                local_errors += 1;
                                let error_msg = format!("Worker {} cycle {}: {}", worker_id, cycle, e);
                                warn!("{}", error_msg);
                                error!("Worker {} cycle {} FAILED: {}", worker_id, cycle, e);
                                metrics.record_error(error_msg).await;
                                if local_errors > 100 {
                                    error!("Worker {} exceeded error threshold", worker_id);
                                    return Err(());
                                }
                                return Ok(());
                            }
                        }
                    }
                    Ok((id, _)) => {
                        // Not our response, continue waiting
                        continue;
                    }
                    Err(e) => {
                        warn!("Worker {} cycle {}: broadcast channel error: {}", worker_id, cycle, e);
                        return Err(());
                    }
                }
            }
        }).await;
        
        if response_timeout.is_err() {
            warn!("Worker {} cycle {}: Response timeout after 30s", worker_id, cycle);
            metrics.record_operation(false, 30000.0, false, false, 0);
            local_errors += 1;
        }

        // Small delay between operations
        sleep(Duration::from_millis(100)).await;
    }

    info!(worker_id, cycles = cycle, errors = local_errors, "Worker {} completed", worker_id);
}

/// Memory monitoring worker
async fn memory_monitor(
    metrics: Arc<SoakMetrics>,
    config: SoakConfig,
    stop_flag: Arc<AtomicBool>,
) {
    let mut last_check = Instant::now();

    while !stop_flag.load(Ordering::Relaxed) {
        sleep(Duration::from_secs(1)).await;
        
        let elapsed = last_check.elapsed();
        if elapsed.as_secs() >= config.memory_check_interval {
            let memory_mb = get_memory_mb();
            metrics.record_memory(memory_mb).await;
            
            let stats = metrics.get_stats().await;
            info!(
                memory_mb = memory_mb,
                peak_mb = stats.peak_memory_mb,
                ops = stats.total_operations,
                ops_per_sec = stats.ops_per_sec,
                success_rate = stats.success_rate,
                "Memory check"
            );

            // Check for memory leak (>500MB growth after 5 minutes)
            if stats.memory_growth_mb > 500.0 && stats.duration_secs > 300.0 {
                warn!(
                    growth_mb = stats.memory_growth_mb,
                    duration = stats.duration_secs,
                    "Potential memory leak detected!"
                );
            }

            last_check = Instant::now();
        }
    }
}

/// Pipeline processor that handles requests sequentially
async fn pipeline_processor(
    mut pipeline: Pipeline,
    mut request_rx: mpsc::Receiver<(String, usize)>,
    response_tx: broadcast::Sender<(usize, Arc<Result<niodoo_real_integrated::pipeline::PipelineCycle>>)>,
    stop_flag: Arc<AtomicBool>,
) {
    info!("Pipeline processor started");
    while !stop_flag.load(Ordering::Relaxed) {
        tokio::select! {
            _ = sleep(Duration::from_millis(10)) => {
                // Check for requests
                if let Ok((prompt, worker_id)) = request_rx.try_recv() {
                    info!("Processing prompt for worker {}", worker_id);
                    let result = pipeline.process_prompt(&prompt).await;
                    // Log errors immediately for debugging
                    if let Err(ref e) = result {
                        eprintln!("Pipeline error for worker {}: {:?}", worker_id, e);
                        error!("Pipeline error for worker {}: {}", worker_id, e);
                    } else {
                        info!("Pipeline success for worker {}", worker_id);
                    }
                    let _ = response_tx.send((worker_id, Arc::new(result)));
                }
            }
        }
    }
}

/// Run comprehensive soak test
async fn run_soak_test(config: SoakConfig) -> Result<SoakStats> {
    info!("🔥 Starting comprehensive soak test");
    info!("Duration: {} seconds", config.duration_secs);
    info!("Concurrent workers: {}", config.concurrent_workers);
    info!("Chaos enabled: {}", config.chaos_enabled);

    // Initialize pipeline
    let args = CliArgs {
        hardware: niodoo_real_integrated::config::HardwareProfile::Beelink,
        prompt: None,
        prompt_file: None,
        swarm: 1,
        iterations: 1,
        output: niodoo_real_integrated::config::OutputFormat::Csv,
        config: None,
        rng_seed_override: Some(42), // Deterministic for testing
    };

    // Set LD_LIBRARY_PATH for ONNX runtime
    let onnx_lib_path = "/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-1.18.1/lib";
    if std::path::Path::new(&format!("{}/libonnxruntime.so", onnx_lib_path)).exists() {
        let current_ld_path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        let new_ld_path = if current_ld_path.is_empty() {
            onnx_lib_path.to_string()
        } else {
            format!("{}:{}", onnx_lib_path, current_ld_path)
        };
        std::env::set_var("LD_LIBRARY_PATH", &new_ld_path);
        info!("Set LD_LIBRARY_PATH for ONNX runtime: {}", new_ld_path);
    }

    // Check if services are available, use real services if they are
    let vllm_endpoint = std::env::var("VLLM_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:5001".to_string());
    let ollama_url = std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
    
    // Extract port from endpoint URL
    let vllm_port = vllm_endpoint
        .strip_prefix("http://")
        .or_else(|| vllm_endpoint.strip_prefix("https://"))
        .unwrap_or(&vllm_endpoint)
        .split(':')
        .nth(1)
        .unwrap_or("5001");
    
    let ollama_port = ollama_url
        .strip_prefix("http://")
        .or_else(|| ollama_url.strip_prefix("https://"))
        .unwrap_or(&ollama_url)
        .split(':')
        .nth(1)
        .unwrap_or("11434");
    
    let vllm_available = tokio::time::timeout(
        Duration::from_secs(2),
        tokio::net::TcpStream::connect(format!("127.0.0.1:{}", vllm_port))
    ).await.is_ok();
    
    let ollama_available = tokio::time::timeout(
        Duration::from_secs(2),
        tokio::net::TcpStream::connect(format!("127.0.0.1:{}", ollama_port))
    ).await.is_ok();
    
    // Check Qdrant availability
    let qdrant_url = std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6333".to_string());
    let qdrant_port = qdrant_url
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .split(':')
        .nth(1)
        .unwrap_or("6333");
    let qdrant_available = tokio::time::timeout(
        Duration::from_secs(2),
        tokio::net::TcpStream::connect(format!("127.0.0.1:{}", qdrant_port))
    ).await.is_ok();

    if !vllm_available || !ollama_available || !qdrant_available {
        warn!(
            vllm = vllm_available,
            ollama = ollama_available,
            qdrant = qdrant_available,
            "Services not fully available, enabling full mock mode and disabling memory store"
        );
        std::env::set_var("MOCK_MODE", "1");
        std::env::set_var("DISABLE_MEMORY_STORE", "1");
    } else {
        info!("Using real services: vLLM={}, Ollama={}, Qdrant={}", vllm_available, ollama_available, qdrant_available);
        // Ensure VLLM_ENDPOINT is set correctly
        if std::env::var("VLLM_ENDPOINT").is_err() {
            std::env::set_var("VLLM_ENDPOINT", "http://127.0.0.1:5001");
        }
        if std::env::var("OLLAMA_URL").is_err() {
            std::env::set_var("OLLAMA_URL", "http://127.0.0.1:11434");
        }
        // Ensure MOCK_MODE is NOT set - use real services
        std::env::remove_var("MOCK_MODE");
        // Enable memory store when all services are up
        std::env::remove_var("DISABLE_MEMORY_STORE");
    }

    let pipeline = Pipeline::initialise(args).await?;
    info!("Pipeline initialized successfully");

    // Load prompts
    let prompts = generate_raw_rut_prompts();
    info!("Loaded {} prompts from gauntlet", prompts.len());
    let prompts = Arc::new(prompts);

    // Initialize metrics
    let metrics = Arc::new(SoakMetrics::new());

    // Stop flag
    let stop_flag = Arc::new(AtomicBool::new(false));

    // Create channels for request/response
    let (request_tx, request_rx) = mpsc::channel(1000);
    let (response_tx, _) = broadcast::channel(1000);
    let response_tx_for_processor = response_tx.clone();

    // Prompt workers - each gets its own response receiver handle
    let mut response_receivers = Vec::new();
    for _ in 0..config.concurrent_workers {
        response_receivers.push(response_tx.subscribe());
    }

    // Spawn single pipeline processor - Pipeline is now Send, so we can use regular async spawn
    let stop_flag_for_processor = stop_flag.clone();
    let pipeline_task = tokio::spawn(async move {
        info!("Pipeline processor starting");
        pipeline_processor(pipeline, request_rx, response_tx_for_processor, stop_flag_for_processor).await
    });

    // Spawn workers
    let mut handles = Vec::new();

    for (worker_id, response_rx) in response_receivers.into_iter().enumerate() {
        let request_tx = request_tx.clone();
        let metrics_clone = metrics.clone();
        let prompts_clone = prompts.clone();
        let stop_flag_clone = stop_flag.clone();
        
        let handle = tokio::spawn(async move {
            prompt_worker(worker_id, request_tx, response_rx, metrics_clone, prompts_clone, stop_flag_clone).await
        });
        handles.push(handle);
    }

    // Memory monitor
    let metrics_monitor = metrics.clone();
    let stop_flag_monitor = stop_flag.clone();
    let config_monitor = config.clone();
    let monitor_handle = tokio::spawn(async move {
        memory_monitor(metrics_monitor, config_monitor, stop_flag_monitor).await
    });

    // Run for configured duration
    sleep(Duration::from_secs(config.duration_secs)).await;

    // Signal stop
    info!("Stopping soak test...");
    stop_flag.store(true, Ordering::Relaxed);

    // Wait for workers
    for handle in handles {
        let _ = handle.await;
    }
    let _ = monitor_handle.await;
    let _ = pipeline_task.await;

    // Get final stats
    let stats = metrics.get_stats().await;

    Ok(stats)
}

/// Print comprehensive test report
fn print_report(stats: &SoakStats) {
    println!("\n{}", "=".repeat(80));
    println!("SOAK TEST REPORT");
    println!("{}", "=".repeat(80));
    println!();
    println!("Duration:           {:.2} seconds ({:.2} minutes)", 
             stats.duration_secs, stats.duration_secs / 60.0);
    println!("Total Operations:   {}", stats.total_operations);
    println!("Successful:         {} ({:.2}%)", 
             stats.successful_operations, stats.success_rate * 100.0);
    println!("Failed:             {} ({:.2}%)", 
             stats.failed_operations, (1.0 - stats.success_rate) * 100.0);
    println!("Throughput:         {:.2} ops/sec", stats.ops_per_sec);
    println!("Avg Latency:        {:.2} ms", stats.avg_latency_ms);
    println!();
    println!("Memory:");
    println!("  Average:          {:.2} MB", stats.avg_memory_mb);
    println!("  Peak:             {:.2} MB", stats.peak_memory_mb);
    println!("  Growth:           {:.2} MB", stats.memory_growth_mb);
    println!();
    println!("Consciousness Events:");
    println!("  Threats:          {}", stats.threat_count);
    println!("  Healings:         {}", stats.healing_count);
    println!("  Breakthroughs:    {}", stats.breakthroughs);
    println!();
    
    // Health checks
    println!("Health Checks:");
    let success_ok = stats.success_rate >= 0.99;
    let memory_ok = stats.memory_growth_mb < 500.0 || stats.duration_secs < 300.0;
    let latency_ok = stats.avg_latency_ms < 1000.0;
    
    println!("  Success Rate:     {} ({:.2}%)", 
             if success_ok { "✅ PASS" } else { "❌ FAIL" },
             stats.success_rate * 100.0);
    println!("  Memory Growth:    {} ({:.2} MB)", 
             if memory_ok { "✅ PASS" } else { "❌ FAIL" },
             stats.memory_growth_mb);
    println!("  Avg Latency:      {} ({:.2} ms)", 
             if latency_ok { "✅ PASS" } else { "❌ FAIL" },
             stats.avg_latency_ms);
    println!();
    
    let overall_health = success_ok && memory_ok && latency_ok;
    println!("Overall Status:     {}", 
             if overall_health { "✅ HEALTHY" } else { "❌ ISSUES DETECTED" });
    println!("{}", "=".repeat(80));
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Parse args
    let args: Vec<String> = std::env::args().collect();
    let quick_test = args.iter().any(|a| a == "--quick" || a == "-q");
    let duration = args.iter()
        .find_map(|a| {
            if a.starts_with("--duration=") {
                a.split('=').nth(1)?.parse::<u64>().ok()
            } else {
                None
            }
        })
        .unwrap_or(if quick_test { 60 } else { 3600 });

    let config = if quick_test {
        SoakConfig::quick()
    } else {
        SoakConfig {
            duration_secs: duration,
            ..Default::default()
        }
    };

    info!("Starting soak test with config: {:?}", config);

    // Run soak test
    let stats = run_soak_test(config).await?;

    // Print report
    print_report(&stats);

    // Export results
    let json = serde_json::to_string_pretty(&stats)?;
    std::fs::write("soak_test_results.json", json)?;
    info!("Results saved to soak_test_results.json");

    // Exit with error if health checks failed
    let success_ok = stats.success_rate >= 0.99;
    let memory_ok = stats.memory_growth_mb < 500.0 || stats.duration_secs < 300.0;
    let latency_ok = stats.avg_latency_ms < 1000.0;

    if !success_ok || !memory_ok || !latency_ok {
        std::process::exit(1);
    }

    Ok(())
}
