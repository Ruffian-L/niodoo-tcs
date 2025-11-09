//! Next-generation soak harness with curated exploration prompts and cycle-aware scheduling.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use niodoo_real_integrated::config::CliArgs;
use niodoo_real_integrated::pipeline::Pipeline;
use niodoo_real_integrated::util::rouge_l;
use serde::Serialize;
use tokio::sync::{broadcast, mpsc, Mutex as AsyncMutex};
use tokio::time::sleep;
use tracing::{error, info, warn};

mod soak_prompts_v2;

use soak_prompts_v2::{
    easy_prompts, hard_prompts, PromptDifficulty, PromptEntry, EASY_PER_CYCLE, PROMPTS_PER_CYCLE,
};

const DEFAULT_DURATION_SECS: u64 = 3600;
const DEFAULT_CONCURRENT_WORKERS: usize = 150;
const DEFAULT_MEMORY_CHECK_INTERVAL_SECS: u64 = 60;
const QUICK_DURATION_SECS: u64 = 120;
const QUICK_CONCURRENT_WORKERS: usize = 36;
const QUICK_MEMORY_CHECK_INTERVAL_SECS: u64 = 10;
const MEMORY_SAMPLE_WINDOW: usize = 2000;
const MAX_ERROR_LOG_ENTRIES: usize = 200;
const DEFAULT_RESPONSE_TIMEOUT_SECS: u64 = 60;
const INTER_OP_SLEEP_MS: u64 = 50;
const ERROR_THRESHOLD_PER_WORKER: u64 = 120;
const ROUGE_SCALE: f64 = 1_000_000.0;
const BASELINE_WIN_MARGIN: f64 = 0.01;
const BASELINE_ALERT_MARGIN: f64 = 0.05;

#[derive(Debug, Clone)]
struct SoakConfig {
    duration_secs: u64,
    concurrent_workers: usize,
    memory_check_interval_secs: u64,
    quick_test: bool,
    response_timeout_secs: u64,
}

impl Default for SoakConfig {
    fn default() -> Self {
        let concurrent_workers = std::env::var("SOAK_WORKERS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_CONCURRENT_WORKERS);
        Self {
            duration_secs: DEFAULT_DURATION_SECS,
            concurrent_workers,
            memory_check_interval_secs: DEFAULT_MEMORY_CHECK_INTERVAL_SECS,
            quick_test: false,
            response_timeout_secs: DEFAULT_RESPONSE_TIMEOUT_SECS,
        }
    }
}

impl SoakConfig {
    fn quick() -> Self {
        let concurrent_workers = std::env::var("SOAK_QUICK_WORKERS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(QUICK_CONCURRENT_WORKERS);
        Self {
            duration_secs: QUICK_DURATION_SECS,
            concurrent_workers,
            memory_check_interval_secs: QUICK_MEMORY_CHECK_INTERVAL_SECS,
            quick_test: true,
            response_timeout_secs: DEFAULT_RESPONSE_TIMEOUT_SECS,
        }
    }
}

fn resolve_timeout_env(keys: &[&str], default: u64) -> u64 {
    for key in keys {
        if let Ok(value) = std::env::var(key) {
            match value.parse::<u64>() {
                Ok(parsed) if parsed > 0 => {
                    info!(
                        env = *key,
                        timeout = parsed,
                        "Using soak response timeout override"
                    );
                    return parsed;
                }
                Ok(_) | Err(_) => {
                    warn!(
                        env = *key,
                        value, "Ignoring invalid soak response timeout override"
                    );
                }
            }
        }
    }
    default
}

#[derive(Debug, Clone)]
struct ScheduledPrompt {
    cycle_index: u64,
    slot_in_cycle: usize,
    entry: PromptEntry,
}

#[derive(Debug)]
struct PromptScheduler {
    easy_prompts: Vec<PromptEntry>,
    hard_prompts: Vec<PromptEntry>,
    easy_cursor: usize,
    hard_cursor: usize,
    cycle_index: u64,
    slot_in_cycle: usize,
}

impl PromptScheduler {
    fn new() -> Self {
        Self {
            easy_prompts: easy_prompts().to_vec(),
            hard_prompts: hard_prompts().to_vec(),
            easy_cursor: 0,
            hard_cursor: 0,
            cycle_index: 0,
            slot_in_cycle: 0,
        }
    }

    fn next_prompt(&mut self) -> ScheduledPrompt {
        if self.slot_in_cycle == PROMPTS_PER_CYCLE {
            self.slot_in_cycle = 0;
            self.cycle_index += 1;
        }

        let cycle_index = self.cycle_index;
        let slot_in_cycle = self.slot_in_cycle;

        let entry = if slot_in_cycle < EASY_PER_CYCLE {
            let entry = self.easy_prompts[self.easy_cursor].clone();
            self.easy_cursor = (self.easy_cursor + 1) % self.easy_prompts.len();
            entry
        } else {
            let entry = self.hard_prompts[self.hard_cursor].clone();
            self.hard_cursor = (self.hard_cursor + 1) % self.hard_prompts.len();
            entry
        };

        self.slot_in_cycle += 1;

        ScheduledPrompt {
            cycle_index,
            slot_in_cycle,
            entry,
        }
    }
}

#[derive(Debug)]
struct SoakMetrics {
    total_operations: AtomicU64,
    successful_operations: AtomicU64,
    failed_operations: AtomicU64,
    total_latency_ms: AtomicU64,
    dispatched_easy: AtomicU64,
    dispatched_hard: AtomicU64,
    dispatched_cycles: AtomicU64,
    completed_cycles: AtomicU64,
    easy_successes: AtomicU64,
    hard_successes: AtomicU64,
    easy_failures: AtomicU64,
    hard_failures: AtomicU64,
    memory_samples_mb: Arc<AsyncMutex<VecDeque<f64>>>,
    error_log: Arc<AsyncMutex<Vec<String>>>,
    start_time: Instant,
    peak_memory_mb: Arc<AsyncMutex<f64>>,
    threat_count: AtomicU64,
    healing_count: AtomicU64,
    breakthroughs: AtomicU64,
    baseline_prompt_rouge_sum: AtomicU64,
    hybrid_prompt_rouge_sum: AtomicU64,
    rouge_to_baseline_sum: AtomicU64,
    baseline_comparisons: AtomicU64,
    hybrid_wins: AtomicU64,
    baseline_wins: AtomicU64,
}

impl SoakMetrics {
    fn new() -> Self {
        Self {
            total_operations: AtomicU64::new(0),
            successful_operations: AtomicU64::new(0),
            failed_operations: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            dispatched_easy: AtomicU64::new(0),
            dispatched_hard: AtomicU64::new(0),
            dispatched_cycles: AtomicU64::new(0),
            completed_cycles: AtomicU64::new(0),
            easy_successes: AtomicU64::new(0),
            hard_successes: AtomicU64::new(0),
            easy_failures: AtomicU64::new(0),
            hard_failures: AtomicU64::new(0),
            memory_samples_mb: Arc::new(AsyncMutex::new(VecDeque::new())),
            error_log: Arc::new(AsyncMutex::new(Vec::new())),
            start_time: Instant::now(),
            peak_memory_mb: Arc::new(AsyncMutex::new(0.0)),
            threat_count: AtomicU64::new(0),
            healing_count: AtomicU64::new(0),
            breakthroughs: AtomicU64::new(0),
            baseline_prompt_rouge_sum: AtomicU64::new(0),
            hybrid_prompt_rouge_sum: AtomicU64::new(0),
            rouge_to_baseline_sum: AtomicU64::new(0),
            baseline_comparisons: AtomicU64::new(0),
            hybrid_wins: AtomicU64::new(0),
            baseline_wins: AtomicU64::new(0),
        }
    }

    fn record_dispatch(&self, scheduled: &ScheduledPrompt) {
        match scheduled.entry.difficulty {
            PromptDifficulty::Easy => {
                self.dispatched_easy.fetch_add(1, Ordering::Relaxed);
            }
            PromptDifficulty::Hard => {
                self.dispatched_hard.fetch_add(1, Ordering::Relaxed);
            }
        }

        if scheduled.slot_in_cycle == 0 {
            let cycle = self.dispatched_cycles.fetch_add(1, Ordering::Relaxed) + 1;
            info!(cycle, "Starting new prompt cycle");
        } else if scheduled.slot_in_cycle + 1 == PROMPTS_PER_CYCLE {
            self.completed_cycles.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_operation(
        &self,
        scheduled: &ScheduledPrompt,
        success: bool,
        latency_ms: f64,
        is_threat: bool,
        is_healing: bool,
        breakthroughs: usize,
        baseline_prompt_rouge: Option<f64>,
        hybrid_prompt_rouge: Option<f64>,
        rouge_to_baseline: Option<f64>,
    ) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ms
            .fetch_add(latency_ms as u64, Ordering::Relaxed);

        let counter = if success {
            self.successful_operations.fetch_add(1, Ordering::Relaxed);
            match scheduled.entry.difficulty {
                PromptDifficulty::Easy => &self.easy_successes,
                PromptDifficulty::Hard => &self.hard_successes,
            }
        } else {
            self.failed_operations.fetch_add(1, Ordering::Relaxed);
            match scheduled.entry.difficulty {
                PromptDifficulty::Easy => &self.easy_failures,
                PromptDifficulty::Hard => &self.hard_failures,
            }
        };

        counter.fetch_add(1, Ordering::Relaxed);

        if success {
            if let Some(rouge) = rouge_to_baseline {
                let scaled = (rouge.clamp(0.0, 1.0) * ROUGE_SCALE).round() as u64;
                self.rouge_to_baseline_sum
                    .fetch_add(scaled, Ordering::Relaxed);
            }

            if let (Some(base_rouge), Some(hybrid_rouge)) =
                (baseline_prompt_rouge, hybrid_prompt_rouge)
            {
                let base_scaled = (base_rouge.clamp(0.0, 1.0) * ROUGE_SCALE).round() as u64;
                let hybrid_scaled = (hybrid_rouge.clamp(0.0, 1.0) * ROUGE_SCALE).round() as u64;
                self.baseline_prompt_rouge_sum
                    .fetch_add(base_scaled, Ordering::Relaxed);
                self.hybrid_prompt_rouge_sum
                    .fetch_add(hybrid_scaled, Ordering::Relaxed);
                self.baseline_comparisons.fetch_add(1, Ordering::Relaxed);

                let delta = hybrid_rouge - base_rouge;
                if delta > BASELINE_WIN_MARGIN {
                    self.hybrid_wins.fetch_add(1, Ordering::Relaxed);
                } else if delta < -BASELINE_WIN_MARGIN {
                    self.baseline_wins.fetch_add(1, Ordering::Relaxed);
                }

                if delta < -BASELINE_ALERT_MARGIN {
                    warn!(
                        cycle = scheduled.cycle_index,
                        slot = scheduled.slot_in_cycle,
                        baseline_rouge = base_rouge,
                        hybrid_rouge,
                        prompt = %scheduled.entry.title,
                        "Hybrid response underperformed baseline beyond alert margin"
                    );
                }
            }
        }

        if is_threat {
            self.threat_count.fetch_add(1, Ordering::Relaxed);
        }
        if is_healing {
            self.healing_count.fetch_add(1, Ordering::Relaxed);
        }
        self.breakthroughs
            .fetch_add(breakthroughs as u64, Ordering::Relaxed);
    }

    async fn record_memory(&self, mb: f64) {
        let mut samples = self.memory_samples_mb.lock().await;
        samples.push_back(mb);
        if samples.len() > MEMORY_SAMPLE_WINDOW {
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
        if log.len() > MAX_ERROR_LOG_ENTRIES {
            log.remove(0);
        }
    }

    async fn stats(&self) -> SoakStats {
        let total = self.total_operations.load(Ordering::Relaxed);
        let success = self.successful_operations.load(Ordering::Relaxed);
        let failed = self.failed_operations.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ms.load(Ordering::Relaxed);
        let threats = self.threat_count.load(Ordering::Relaxed);
        let healings = self.healing_count.load(Ordering::Relaxed);
        let breakthroughs = self.breakthroughs.load(Ordering::Relaxed);
        let dispatched_easy = self.dispatched_easy.load(Ordering::Relaxed);
        let dispatched_hard = self.dispatched_hard.load(Ordering::Relaxed);
        let dispatched_cycles = self.dispatched_cycles.load(Ordering::Relaxed);
        let completed_cycles = self.completed_cycles.load(Ordering::Relaxed);

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
            // Safety: We checked len() >= 2 above, so first() and last() are guaranteed Some
            let growth = samples.last().expect("samples.len() >= 2 ensures last() exists") 
                - samples.first().expect("samples.len() >= 2 ensures first() exists");
            (avg, growth)
        } else {
            (0.0, 0.0)
        };

        let peak_memory = *self.peak_memory_mb.lock().await;
        let duration = self.start_time.elapsed().as_secs_f64();
        let ops_per_sec = if duration > 0.0 {
            total as f64 / duration
        } else {
            0.0
        };

        let rouge_sum = self.rouge_to_baseline_sum.load(Ordering::Relaxed);
        let baseline_sum = self.baseline_prompt_rouge_sum.load(Ordering::Relaxed);
        let hybrid_sum = self.hybrid_prompt_rouge_sum.load(Ordering::Relaxed);
        let comparisons = self.baseline_comparisons.load(Ordering::Relaxed);
        let hybrid_wins = self.hybrid_wins.load(Ordering::Relaxed);
        let baseline_wins = self.baseline_wins.load(Ordering::Relaxed);

        let avg_rouge_to_baseline = if success > 0 {
            rouge_sum as f64 / ROUGE_SCALE / success as f64
        } else {
            0.0
        };

        let avg_baseline_prompt_rouge = if comparisons > 0 {
            baseline_sum as f64 / ROUGE_SCALE / comparisons as f64
        } else {
            0.0
        };

        let avg_hybrid_prompt_rouge = if comparisons > 0 {
            hybrid_sum as f64 / ROUGE_SCALE / comparisons as f64
        } else {
            0.0
        };

        let baseline_win_rate = if comparisons > 0 {
            baseline_wins as f64 / comparisons as f64
        } else {
            0.0
        };

        let hybrid_win_rate = if comparisons > 0 {
            hybrid_wins as f64 / comparisons as f64
        } else {
            0.0
        };

        let tie_rate = (1.0 - baseline_win_rate - hybrid_win_rate).clamp(0.0, 1.0);

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
            dispatched_easy,
            dispatched_hard,
            dispatched_cycles,
            completed_cycles,
            easy_successes: self.easy_successes.load(Ordering::Relaxed),
            hard_successes: self.hard_successes.load(Ordering::Relaxed),
            easy_failures: self.easy_failures.load(Ordering::Relaxed),
            hard_failures: self.hard_failures.load(Ordering::Relaxed),
            avg_rouge_to_baseline,
            avg_baseline_prompt_rouge,
            avg_hybrid_prompt_rouge,
            baseline_comparisons: comparisons,
            baseline_wins,
            hybrid_wins,
            baseline_win_rate,
            hybrid_win_rate,
            tie_rate,
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
    dispatched_easy: u64,
    dispatched_hard: u64,
    dispatched_cycles: u64,
    completed_cycles: u64,
    easy_successes: u64,
    hard_successes: u64,
    easy_failures: u64,
    hard_failures: u64,
    avg_rouge_to_baseline: f64,
    avg_baseline_prompt_rouge: f64,
    avg_hybrid_prompt_rouge: f64,
    baseline_comparisons: u64,
    baseline_wins: u64,
    hybrid_wins: u64,
    baseline_win_rate: f64,
    hybrid_win_rate: f64,
    tie_rate: f64,
}

fn memory_usage_mb() -> f64 {
    if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
        for line in contents.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(kb_str) = parts.get(1) {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        return kb / 1024.0;
                    }
                }
            }
        }
    }
    0.0
}

async fn prompt_worker(
    worker_id: usize,
    request_tx: mpsc::Sender<(String, usize)>,
    mut response_rx: broadcast::Receiver<(
        usize,
        Arc<Result<niodoo_real_integrated::pipeline::PipelineCycle>>,
    )>,
    metrics: Arc<SoakMetrics>,
    scheduler: Arc<AsyncMutex<PromptScheduler>>,
    stop_flag: Arc<AtomicBool>,
    response_timeout_secs: u64,
) {
    let mut local_errors = 0;

    while !stop_flag.load(Ordering::Relaxed) {
        let scheduled = {
            let mut guard = scheduler.lock().await;
            let scheduled = guard.next_prompt();
            metrics.record_dispatch(&scheduled);
            scheduled
        };

        let prompt = scheduled.entry.to_prompt();
        let start = Instant::now();

        if request_tx.send((prompt.clone(), worker_id)).await.is_err() {
            warn!(worker_id, "Request channel closed");
            break;
        }

        let response_timeout =
            tokio::time::timeout(Duration::from_secs(response_timeout_secs), async {
                loop {
                    match response_rx.recv().await {
                        Ok((id, result_arc)) if id == worker_id => {
                            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                            match result_arc.as_ref() {
                                Ok(cycle_result) => {
                                    let baseline_prompt_rouge = rouge_l(
                                        &cycle_result.generation.baseline_response,
                                        &cycle_result.prompt,
                                    );
                                    let hybrid_prompt_rouge = rouge_l(
                                        &cycle_result.generation.hybrid_response,
                                        &cycle_result.prompt,
                                    );
                                    metrics.record_operation(
                                        &scheduled,
                                        true,
                                        latency_ms,
                                        cycle_result.compass.is_threat,
                                        cycle_result.compass.is_healing,
                                        cycle_result.learning.breakthroughs.len(),
                                        Some(baseline_prompt_rouge),
                                        Some(hybrid_prompt_rouge),
                                        Some(cycle_result.rouge),
                                    );
                                    if !cycle_result.learning.breakthroughs.is_empty() {
                                        info!(
                                            worker_id,
                                            cycle = scheduled.cycle_index,
                                            breakthroughs =
                                                cycle_result.learning.breakthroughs.len(),
                                            "Learning loop breakthroughs applied"
                                        );
                                    }
                                    if scheduled.slot_in_cycle == 0 {
                                        info!(
                                            worker_id,
                                            cycle = scheduled.cycle_index,
                                            prompt = prompt,
                                            latency_ms,
                                            "Cycle lead prompt executed"
                                        );
                                    }
                                    return Ok(());
                                }
                                Err(e) => {
                                    metrics.record_operation(
                                        &scheduled, false, latency_ms, false, false, 0, None, None,
                                        None,
                                    );
                                    local_errors += 1;
                                    let error_msg = format!(
                                        "Worker {} cycle {} slot {} error: {}",
                                        worker_id,
                                        scheduled.cycle_index,
                                        scheduled.slot_in_cycle,
                                        e
                                    );
                                    error!(
                                        worker_id,
                                        cycle = scheduled.cycle_index,
                                        slot = scheduled.slot_in_cycle,
                                        "{}",
                                        error_msg
                                    );
                                    metrics.record_error(error_msg).await;
                                    return Ok(());
                                }
                            }
                        }
                        Ok((_, _)) => continue,
                        Err(e) => {
                            warn!(worker_id, "Broadcast channel error: {}", e);
                            return Err(());
                        }
                    }
                }
            })
            .await;

        if response_timeout.is_err() {
            warn!(
                worker_id,
                cycle = scheduled.cycle_index,
                slot = scheduled.slot_in_cycle,
                timeout_secs = response_timeout_secs,
                "Response timeout"
            );
            metrics.record_operation(
                &scheduled,
                false,
                response_timeout_secs as f64 * 1000.0,
                false,
                false,
                0,
                None,
                None,
                None,
            );
            local_errors += 1;
        }

        if local_errors > ERROR_THRESHOLD_PER_WORKER {
            error!(
                worker_id,
                local_errors, "Error threshold exceeded, shutting down worker"
            );
            break;
        }

        sleep(Duration::from_millis(INTER_OP_SLEEP_MS)).await;
    }

    info!(worker_id, local_errors, "Worker exiting");
}

async fn memory_monitor(metrics: Arc<SoakMetrics>, config: SoakConfig, stop_flag: Arc<AtomicBool>) {
    let mut last_check = Instant::now();

    while !stop_flag.load(Ordering::Relaxed) {
        sleep(Duration::from_secs(1)).await;
        if last_check.elapsed().as_secs() >= config.memory_check_interval_secs {
            let memory_mb = memory_usage_mb();
            metrics.record_memory(memory_mb).await;
            let stats = metrics.stats().await;
            info!(
                memory_mb = memory_mb,
                peak_mb = stats.peak_memory_mb,
                dispatched_cycles = stats.dispatched_cycles,
                completed_cycles = stats.completed_cycles,
                total_ops = stats.total_operations,
                success_rate = stats.success_rate,
                "Memory monitor"
            );
            last_check = Instant::now();
        }
    }
}

async fn pipeline_processor(
    mut pipeline: Pipeline,
    mut request_rx: mpsc::Receiver<(String, usize)>,
    response_tx: broadcast::Sender<(
        usize,
        Arc<Result<niodoo_real_integrated::pipeline::PipelineCycle>>,
    )>,
    stop_flag: Arc<AtomicBool>,
) {
    info!("Pipeline processor online");

    while !stop_flag.load(Ordering::Relaxed) {
        tokio::select! {
            maybe_request = request_rx.recv() => {
                if let Some((prompt, worker_id)) = maybe_request {
                    let result = pipeline.process_prompt(&prompt).await;
                    if let Err(ref e) = result {
                        error!(worker_id, "Pipeline error: {}", e);
                    }
                    let _ = response_tx.send((worker_id, Arc::new(result)));
                } else {
                    sleep(Duration::from_millis(5)).await;
                }
            }
        }
    }
}

async fn configure_services() {
    let vllm_endpoint =
        std::env::var("VLLM_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:5001".to_string());
    let ollama_url =
        std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6333".to_string());

    let vllm_port = extract_port(&vllm_endpoint, "5001");
    let ollama_port = extract_port(&ollama_url, "11434");
    let qdrant_port = extract_port(&qdrant_url, "6333");

    let vllm_available = endpoint_available("127.0.0.1", vllm_port).await;
    let ollama_available = endpoint_available("127.0.0.1", ollama_port).await;
    let qdrant_available = endpoint_available("127.0.0.1", qdrant_port).await;

    if vllm_available && qdrant_available {
        info!(
            vllm_endpoint,
            ollama_url, qdrant_url, "Using live service endpoints for soak test v2"
        );
        if std::env::var("VLLM_ENDPOINT").is_err() {
            std::env::set_var("VLLM_ENDPOINT", &vllm_endpoint);
        }
        if ollama_available && std::env::var("OLLAMA_URL").is_err() {
            std::env::set_var("OLLAMA_URL", &ollama_url);
        }
        std::env::remove_var("MOCK_MODE");
        std::env::remove_var("DISABLE_MEMORY_STORE");
    } else {
        warn!(
            vllm_available,
            ollama_available, qdrant_available, "Falling back to mock mode for soak test v2"
        );
        std::env::set_var("MOCK_MODE", "1");
        std::env::set_var("DISABLE_MEMORY_STORE", "1");
    }

    ensure_embedding_model_path();
}

fn extract_port(endpoint: &str, default_port: &str) -> String {
    endpoint
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .split(':')
        .nth(1)
        .unwrap_or(default_port)
        .to_string()
}

async fn endpoint_available(host: &str, port: String) -> bool {
    tokio::time::timeout(
        Duration::from_secs(2),
        tokio::net::TcpStream::connect(format!("{}:{}", host, port)),
    )
    .await
    .is_ok()
}

fn ensure_embedding_model_path() {
    const CANDIDATE_PATHS: &[&str] = &[
        "/workspace/models/Qwen2.5-0.5B-Instruct/onnx/model_fp16.onnx",
        "/workspace/models/Qwen2-0.5B-Instruct/onnx/model_fp16.onnx",
        "/workspace/Niodoo-Final/models/Qwen2.5-0.5B-Instruct/onnx/model_fp16.onnx",
        "/workspace/Niodoo-Final/models/qwen2.5-coder-0.5b-instruct-onnx/onnx/model_fp16.onnx",
    ];

    if let Ok(current) = std::env::var("EMBEDDING_MODEL_NAME") {
        let path = Path::new(&current);
        if path.exists() {
            info!(model = %path.display(), "Embedding model path resolved from environment");
            propagate_embed_dir(path);
            return;
        }
        warn!(
            model = current,
            "EMBEDDING_MODEL_NAME set but path missing; searching for defaults"
        );
    }

    for candidate in CANDIDATE_PATHS {
        let candidate_path = Path::new(candidate);
        if candidate_path.exists() {
            std::env::set_var("EMBEDDING_MODEL_NAME", candidate);
            info!(model = %candidate_path.display(), "Embedding model path discovered automatically");
            propagate_embed_dir(candidate_path);
            return;
        }
    }

    warn!("Unable to locate embedding ONNX model automatically. Set EMBEDDING_MODEL_NAME to the absolute path of model_fp16.onnx");
}

fn propagate_embed_dir(model_path: &Path) {
    if std::env::var("EMBED_MODEL_DIR").is_err() {
        if let Some(parent) = model_path.parent() {
            if parent.exists() {
                std::env::set_var("EMBED_MODEL_DIR", parent.to_string_lossy().to_string());
                info!(dir = %parent.display(), "Set EMBED_MODEL_DIR based on embedding path");
            }
        }
    }
}

fn configure_onnx_runtime() {
    const GPU_LIB_CANDIDATES: &[&str] = &[
        "/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.18.1/lib",
        "/workspace/onnxruntime-linux-x64-gpu-1.18.1/lib",
        "/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.16.3/lib",
        "/workspace/onnxruntime-linux-x64-gpu-1.16.3/lib",
    ];
    const CPU_LIB_CANDIDATES: &[&str] = &[
        "/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-1.18.1/lib",
        "/workspace/onnxruntime-linux-x64-1.18.1/lib",
        "/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-1.16.3/lib",
        "/workspace/onnxruntime-linux-x64-1.16.3/lib",
    ];

    let gpu_path = GPU_LIB_CANDIDATES
        .iter()
        .find(|candidate| {
            Path::new(&format!("{}/libonnxruntime_providers_cuda.so", candidate)).exists()
        })
        .map(|p| p.to_string());
    let cpu_path = CPU_LIB_CANDIDATES
        .iter()
        .find(|candidate| Path::new(&format!("{}/libonnxruntime.so", candidate)).exists())
        .map(|p| p.to_string());

    let mut base_runtime_path = None;

    let onnx_lib_path = if let Some(path) = gpu_path {
        info!("Using CUDA-enabled ONNX runtime");
        base_runtime_path = Some(path.clone());
        let compat_path = format!("{}/cuda_compat", path);
        if std::path::Path::new(&compat_path).exists() {
            format!("{}:{}", path, compat_path)
        } else {
            path
        }
    } else if let Some(path) = cpu_path {
        warn!("CUDA build not found, using CPU ONNX runtime");
        base_runtime_path = Some(path.clone());
        path
    } else {
        warn!("ONNX runtime binaries not located");
        String::new()
    };

    if !onnx_lib_path.is_empty() {
        if let Some(base_path) = base_runtime_path {
            let dylib_path = format!("{}/libonnxruntime.so", base_path);
            std::env::set_var("ORT_DYLIB_PATH", &dylib_path);
            std::env::set_var("ORT_DYLIB_DEFAULT_PATH", &base_path);
            info!(
                ort_dylib = %dylib_path,
                ort_default = %base_path,
                "Configured ORT dynamic library search paths"
            );
        }

        let cudnn_extract = "/tmp/cudnn8_extract/cudnn-linux-x86_64-8.9.7.29_cuda11-archive/lib";
        let cuda11 = "/usr/local/cuda-11.8/lib64";
        let cuda12 = "/usr/local/cuda-12.8/lib64";
        let current = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();

        let new_ld_path = if current.is_empty() {
            format!("{}:{}:{}:{}", cudnn_extract, onnx_lib_path, cuda11, cuda12)
        } else {
            format!(
                "{}:{}:{}:{}:{}",
                cudnn_extract, onnx_lib_path, cuda11, cuda12, current
            )
        };

        std::env::set_var("LD_LIBRARY_PATH", &new_ld_path);
        info!("LD_LIBRARY_PATH configured for ONNX runtime");
    }
}

async fn run_soak_test(config: SoakConfig) -> Result<SoakStats> {
    info!(
        duration_secs = config.duration_secs,
        concurrent_workers = config.concurrent_workers,
        quick_mode = config.quick_test,
        "Starting soak test v2"
    );

    configure_onnx_runtime();
    configure_services().await;

    let args = CliArgs {
        hardware: niodoo_real_integrated::config::HardwareProfile::Beelink,
        prompt: None,
        prompt_file: None,
        swarm: 1,
        iterations: 1,
        output: niodoo_real_integrated::config::OutputFormat::Csv,
        config: None,
        rng_seed_override: Some(84),
    };

    let pipeline = Pipeline::initialise(args).await?;
    info!("Pipeline initialised");

    let metrics = Arc::new(SoakMetrics::new());
    let scheduler = Arc::new(AsyncMutex::new(PromptScheduler::new()));
    let stop_flag = Arc::new(AtomicBool::new(false));

    let (request_tx, request_rx) = mpsc::channel::<(String, usize)>(config.concurrent_workers * 4);
    let (response_tx, _) = broadcast::channel::<(
        usize,
        Arc<Result<niodoo_real_integrated::pipeline::PipelineCycle>>,
    )>(config.concurrent_workers * 4);

    let response_tx_for_processor = response_tx.clone();
    let stop_flag_for_processor = stop_flag.clone();
    let pipeline_task = tokio::spawn(async move {
        pipeline_processor(
            pipeline,
            request_rx,
            response_tx_for_processor,
            stop_flag_for_processor,
        )
        .await
    });

    let mut worker_handles = Vec::new();
    let response_timeout_secs = config.response_timeout_secs;

    for worker_id in 0..config.concurrent_workers {
        let request_tx = request_tx.clone();
        let response_rx = response_tx.subscribe();
        let metrics_clone = metrics.clone();
        let scheduler_clone = scheduler.clone();
        let stop_flag_clone = stop_flag.clone();

        worker_handles.push(tokio::spawn(async move {
            prompt_worker(
                worker_id,
                request_tx,
                response_rx,
                metrics_clone,
                scheduler_clone,
                stop_flag_clone,
                response_timeout_secs,
            )
            .await
        }));
    }

    let metrics_monitor = metrics.clone();
    let stop_flag_monitor = stop_flag.clone();
    let config_monitor = config.clone();
    let monitor_handle = tokio::spawn(async move {
        memory_monitor(metrics_monitor, config_monitor, stop_flag_monitor).await
    });

    sleep(Duration::from_secs(config.duration_secs)).await;
    stop_flag.store(true, Ordering::Relaxed);

    for handle in worker_handles {
        let _ = handle.await;
    }
    let _ = monitor_handle.await;
    let _ = pipeline_task.await;

    let stats = metrics.stats().await;
    Ok(stats)
}

fn print_report(stats: &SoakStats) {
    println!("\n{}", "=".repeat(100));
    println!("SOAK TEST V2 REPORT");
    println!("{}", "=".repeat(100));
    println!(
        "Duration: {:.2} seconds ({:.2} minutes)",
        stats.duration_secs,
        stats.duration_secs / 60.0
    );
    println!("Total operations: {}", stats.total_operations);
    println!(
        "Success rate: {:.2}% ({} success / {} failed)",
        stats.success_rate * 100.0,
        stats.successful_operations,
        stats.failed_operations
    );
    println!("Throughput: {:.2} ops/sec", stats.ops_per_sec);
    println!("Average latency: {:.2} ms", stats.avg_latency_ms);
    println!(
        "Prompt mix dispatched: {} easy / {} hard (cycles started: {}, completed: {})",
        stats.dispatched_easy,
        stats.dispatched_hard,
        stats.dispatched_cycles,
        stats.completed_cycles
    );
    println!(
        "Outcome mix: easy success {} | easy failure {} | hard success {} | hard failure {}",
        stats.easy_successes, stats.easy_failures, stats.hard_successes, stats.hard_failures
    );
    println!(
        "Threats: {} | Healings: {} | Breakthroughs: {}",
        stats.threat_count, stats.healing_count, stats.breakthroughs
    );
    println!(
        "Memory (avg / peak / growth MB): {:.2} / {:.2} / {:.2}",
        stats.avg_memory_mb, stats.peak_memory_mb, stats.memory_growth_mb
    );
    println!(
        "Prompt ROUGE (baseline vs hybrid vs delta): {:.3} / {:.3} / {:+.3}",
        stats.avg_baseline_prompt_rouge,
        stats.avg_hybrid_prompt_rouge,
        stats.avg_hybrid_prompt_rouge - stats.avg_baseline_prompt_rouge
    );
    println!(
        "Hybrid vs baseline similarity (ROUGE-L): {:.3}",
        stats.avg_rouge_to_baseline
    );
    println!(
        "Baseline wins {} ({:.1}%), Hybrid wins {} ({:.1}%), ties {:.1}%",
        stats.baseline_wins,
        stats.baseline_win_rate * 100.0,
        stats.hybrid_wins,
        stats.hybrid_win_rate * 100.0,
        stats.tie_rate * 100.0
    );

    let success_ok = stats.success_rate >= 0.95;
    let memory_ok = stats.memory_growth_mb < 500.0 || stats.duration_secs < 300.0;
    let latency_ok = stats.avg_latency_ms < 1000.0;
    let hybrid_ok = stats.hybrid_win_rate >= 0.5;

    println!(
        "Health checks: success {} | memory {} | latency {} | hybrid_vs_baseline {}",
        if success_ok { "✅" } else { "❌" },
        if memory_ok { "✅" } else { "❌" },
        if latency_ok { "✅" } else { "❌" },
        if hybrid_ok { "✅" } else { "❌" }
    );
    println!("{}", "=".repeat(100));

    if !(success_ok && memory_ok && latency_ok && hybrid_ok) {
        eprintln!("Warning: one or more health checks failed");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick" || a == "-q");
    let duration = args
        .iter()
        .find_map(|arg| {
            if let Some(value) = arg.strip_prefix("--duration=") {
                value.parse::<u64>().ok()
            } else {
                None
            }
        })
        .unwrap_or(if quick {
            QUICK_DURATION_SECS
        } else {
            DEFAULT_DURATION_SECS
        });

    let mut config = if quick {
        let mut quick_cfg = SoakConfig::quick();
        quick_cfg.duration_secs = duration;
        quick_cfg
    } else {
        SoakConfig {
            duration_secs: duration,
            ..Default::default()
        }
    };

    let timeout_keys = if quick {
        ["SOAK_QUICK_RESPONSE_TIMEOUT", "SOAK_RESPONSE_TIMEOUT"]
    } else {
        ["SOAK_RESPONSE_TIMEOUT", "SOAK_QUICK_RESPONSE_TIMEOUT"]
    };
    config.response_timeout_secs =
        resolve_timeout_env(&timeout_keys, DEFAULT_RESPONSE_TIMEOUT_SECS);

    info!(?config, "Launching soak test v2");

    let stats = run_soak_test(config).await?;
    print_report(&stats);

    let json = serde_json::to_string_pretty(&stats)?;
    std::fs::write("soak_test_v2_results.json", json)?;
    info!("Results saved to soak_test_v2_results.json");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_emits_expected_mix_per_cycle() {
        let mut scheduler = PromptScheduler::new();
        let mut easy_counts = Vec::new();
        let mut hard_counts = Vec::new();

        for _ in 0..5 {
            let mut easy = 0;
            let mut hard = 0;
            for _ in 0..PROMPTS_PER_CYCLE {
                let scheduled = scheduler.next_prompt();
                match scheduled.entry.difficulty {
                    PromptDifficulty::Easy => easy += 1,
                    PromptDifficulty::Hard => hard += 1,
                }
            }
            easy_counts.push(easy);
            hard_counts.push(hard);
        }

        assert!(easy_counts.iter().all(|&count| count == EASY_PER_CYCLE));
        assert!(hard_counts
            .iter()
            .all(|&count| count == soak_prompts_v2::HARD_PER_CYCLE));
    }

    #[test]
    fn scheduler_wraps_prompts() {
        let mut scheduler = PromptScheduler::new();
        let total_prompts = easy_prompts().len() + hard_prompts().len();
        let mut seen_ids = std::collections::HashSet::new();

        for _ in 0..total_prompts * 2 {
            let scheduled = scheduler.next_prompt();
            seen_ids.insert(scheduled.entry.id);
        }

        assert_eq!(seen_ids.len(), total_prompts);
    }
}
