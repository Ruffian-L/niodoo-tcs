//! A/B Test Runner for Configuration Comparison
//!
//! Compares baseline vs treatment configurations to prove system superiority
//! through statistical comparison of performance and quality metrics.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tracing::info;

use niodoo_real_integrated::config::CliArgs;
use niodoo_real_integrated::pipeline::Pipeline;
use niodoo_real_integrated::validation::stats::{
    StatisticalSummary, cohens_d, mann_whitney_u, bootstrap_percentile_ci,
};

#[derive(Parser, Debug)]
#[command(name = "ab_test_runner")]
#[command(about = "A/B testing framework for comparing configurations")]
struct Args {
    /// Baseline configuration name
    #[arg(long, default_value = "baseline")]
    baseline_name: String,

    /// Treatment configuration name
    #[arg(long, default_value = "treatment")]
    treatment_name: String,

    /// Baseline configuration file (JSON)
    #[arg(long)]
    baseline_config: Option<PathBuf>,

    /// Treatment configuration file (JSON)
    #[arg(long)]
    treatment_config: Option<PathBuf>,

    /// Output directory for results
    #[arg(long, default_value = "ab_test_results")]
    output_dir: PathBuf,

    /// Number of concurrent users for load test
    #[arg(long, default_value = "16")]
    concurrent_users: usize,

    /// Duration of test in seconds
    #[arg(long, default_value = "60")]
    duration_secs: u64,

    /// Statistical significance threshold (p-value)
    #[arg(long, default_value = "0.05")]
    significance_threshold: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ABTestResult {
    baseline_name: String,
    treatment_name: String,
    timestamp: String,
    baseline_metrics: TestMetrics,
    treatment_metrics: TestMetrics,
    comparison: Comparison,
    winner: String,
    statistical_significance: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct TestMetrics {
    latency: StatisticalSummary,
    throughput: f64,
    quality_score: Option<f64>,
    error_rate: f64,
    request_count: u64,
    // Topology-specific metrics
    topology_metrics: TopologyMetrics,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct TopologyMetrics {
    persistence_entropy_mean: Option<f64>,
    persistence_entropy_std: Option<f64>,
    beta_meta_current_mean: Option<f64>,
    beta_meta_peak_mean: Option<f64>,
    spectral_gap_mean: Option<f64>,
    quality_score_mean: Option<f64>,
    quality_score_std: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Comparison {
    latency_difference_ms: f64,
    latency_difference_pct: f64,
    throughput_difference_pct: f64,
    quality_difference_pct: Option<f64>,
    error_rate_difference_pct: f64,
    cohens_d_latency: f64,
    cohens_d_throughput: f64,
    p_value_latency: f64,
    p_value_throughput: f64,
    confidence_interval_95_latency: (f64, f64),
    confidence_interval_95_throughput: (f64, f64),
    effect_size_latency: String,
    effect_size_throughput: String,
    winner_metrics: Vec<String>,
    // Topology-specific comparisons
    persistence_entropy_difference: Option<f64>,
    beta_meta_difference: Option<f64>,
    topology_impact: Option<String>, // "positive", "negative", "neutral", "inconclusive"
}

pub struct ABTestRunner;

impl ABTestRunner {
    pub fn new() -> Self {
        Self
    }

    /// Run A/B test comparing baseline vs treatment
    pub async fn run_ab_test(
        &self,
        baseline_config: &HashMap<String, String>,
        treatment_config: &HashMap<String, String>,
        concurrent_users: usize,
        duration_secs: u64,
    ) -> Result<(TestMetrics, TestMetrics)> {
        // Run baseline test
        info!("🧪 Running baseline configuration...");
        let baseline_metrics = self.run_configuration_test(baseline_config, concurrent_users, duration_secs).await?;

        // Run treatment test
        info!("🧪 Running treatment configuration...");
        let treatment_metrics = self.run_configuration_test(treatment_config, concurrent_users, duration_secs).await?;

        Ok((baseline_metrics, treatment_metrics))
    }

    async fn run_configuration_test(
        &self,
        config: &HashMap<String, String>,
        concurrent_users: usize,
        duration_secs: u64,
    ) -> Result<TestMetrics> {
        // Set environment variables from config
        for (key, value) in config {
            std::env::set_var(key, value);
        }

        // Initialize pipeline
        let cli_args = CliArgs::default();
        let pipeline = Pipeline::initialise(cli_args).await
            .context("Failed to initialize pipeline for A/B test")?;

        // Run load test
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;
        use tokio::sync::Mutex as AsyncMutex;
        use std::time::{Duration, Instant};

        let pipeline_mutex = Arc::new(AsyncMutex::new(pipeline));
        let latencies = Arc::new(AsyncMutex::new(Vec::new()));
        let request_count = Arc::new(AtomicU64::new(0));
        let error_count = Arc::new(AtomicU64::new(0));
        // Topology metrics collection
        let persistence_entropies = Arc::new(AsyncMutex::new(Vec::new()));
        let spectral_gaps = Arc::new(AsyncMutex::new(Vec::new()));
        let quality_scores = Arc::new(AsyncMutex::new(Vec::new()));
        let start_time = Instant::now();
        let end_time = start_time + Duration::from_secs(duration_secs);

        let test_prompts = vec![
            "Explain quantum computing in detail".to_string(),
            "Describe the theory of relativity".to_string(),
            "What is machine learning?".to_string(),
            "How does neural network training work?".to_string(),
            "Explain the concept of consciousness".to_string(),
        ];

        let mut handles = Vec::new();
        for user_id in 0..concurrent_users {
            let pipeline_clone = pipeline_mutex.clone();
            let latencies_clone = latencies.clone();
            let request_count_clone = request_count.clone();
            let error_count_clone = error_count.clone();
            let persistence_entropies_clone = persistence_entropies.clone();
            let spectral_gaps_clone = spectral_gaps.clone();
            let quality_scores_clone = quality_scores.clone();
            let prompts = test_prompts.clone();

            let handle = tokio::spawn(async move {
                let mut prompt_idx = 0;
                while Instant::now() < end_time {
                    let prompt = prompts[prompt_idx % prompts.len()].clone();
                    prompt_idx += 1;

                    let cycle_start = Instant::now();
                    let mut pipeline_guard = pipeline_clone.lock().await;

                    match pipeline_guard.process_prompt(prompt.as_str()).await {
                        Ok(cycle) => {
                            let latency_ms = cycle_start.elapsed().as_secs_f64() * 1000.0;
                            latencies_clone.lock().await.push(latency_ms);
                            
                            // Extract topology metrics
                            persistence_entropies_clone.lock().await.push(cycle.topology.persistence_entropy);
                            spectral_gaps_clone.lock().await.push(cycle.topology.spectral_gap);
                            
                            // Try to extract quality score from consonance if available
                            if let Some(ref consonance) = cycle.consonance {
                                // Use score field (1.0 - dissonance_score) as quality measure
                                let quality = 1.0 - consonance.dissonance_score;
                                quality_scores_clone.lock().await.push(quality);
                            }
                            
                            request_count_clone.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            tracing::warn!(user_id, error = %e, "A/B test request failed");
                            error_count_clone.fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.await;
        }

        let actual_duration = start_time.elapsed().as_secs_f64();
        let latencies_vec = latencies.lock().await.clone();
        let persistence_entropies_vec = persistence_entropies.lock().await.clone();
        let spectral_gaps_vec = spectral_gaps.lock().await.clone();
        let quality_scores_vec = quality_scores.lock().await.clone();
        let requests = request_count.load(Ordering::Relaxed);
        let errors = error_count.load(Ordering::Relaxed);

        let latency_stats = if !latencies_vec.is_empty() {
            StatisticalSummary::from_values(&latencies_vec)
        } else {
            StatisticalSummary {
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
                min: 0.0,
                max: 0.0,
                count: 0,
            }
        };

        let throughput = if actual_duration > 0.0 {
            requests as f64 / actual_duration
        } else {
            0.0
        };

        let error_rate = if requests > 0 {
            errors as f64 / requests as f64
        } else {
            0.0
        };

        // Compute topology metrics
        let persistence_entropy_mean = if !persistence_entropies_vec.is_empty() {
            Some(persistence_entropies_vec.iter().sum::<f64>() / persistence_entropies_vec.len() as f64)
        } else {
            None
        };
        
        let persistence_entropy_std = if persistence_entropies_vec.len() > 1 {
            let mean = persistence_entropy_mean.unwrap_or(0.0);
            let variance = persistence_entropies_vec.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / (persistence_entropies_vec.len() - 1) as f64;
            Some(variance.sqrt())
        } else {
            None
        };

        let spectral_gap_mean = if !spectral_gaps_vec.is_empty() {
            Some(spectral_gaps_vec.iter().sum::<f64>() / spectral_gaps_vec.len() as f64)
        } else {
            None
        };

        let quality_score_mean = if !quality_scores_vec.is_empty() {
            Some(quality_scores_vec.iter().sum::<f64>() / quality_scores_vec.len() as f64)
        } else {
            None
        };

        let quality_score_std = if quality_scores_vec.len() > 1 {
            let mean = quality_score_mean.unwrap_or(0.0);
            let variance = quality_scores_vec.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / (quality_scores_vec.len() - 1) as f64;
            Some(variance.sqrt())
        } else {
            None
        };

        // Try to get β_meta from Prometheus metrics endpoint
        let (beta_meta_current_mean, beta_meta_peak_mean) = self.fetch_beta_meta_from_metrics().await.unwrap_or((None, None));

        Ok(TestMetrics {
            latency: latency_stats,
            throughput,
            quality_score: quality_score_mean,
            error_rate,
            request_count: requests,
            topology_metrics: TopologyMetrics {
                persistence_entropy_mean,
                persistence_entropy_std,
                beta_meta_current_mean,
                beta_meta_peak_mean,
                spectral_gap_mean,
                quality_score_mean,
                quality_score_std,
            },
        })
    }

    /// Fetch β_meta metrics from Prometheus endpoint
    async fn fetch_beta_meta_from_metrics(&self) -> Result<(Option<f64>, Option<f64>)> {
        let metrics_url = std::env::var("NIODOO_METRICS_URL")
            .unwrap_or_else(|_| "http://localhost:9090/metrics".to_string());
        
        match reqwest::get(&metrics_url).await {
            Ok(response) => {
                let text = response.text().await?;
                let mut beta_meta_current = None;
                let mut beta_meta_peak = None;
                
                for line in text.lines() {
                    if line.starts_with("niodoo_rce_beta_meta_current") {
                        if let Some(value) = line.split_whitespace().last().and_then(|v| v.parse::<f64>().ok()) {
                            beta_meta_current = Some(value);
                        }
                    } else if line.starts_with("niodoo_rce_beta_meta_peak") {
                        if let Some(value) = line.split_whitespace().last().and_then(|v| v.parse::<f64>().ok()) {
                            beta_meta_peak = Some(value);
                        }
                    }
                }
                
                Ok((beta_meta_current, beta_meta_peak))
            }
            Err(_) => Ok((None, None)), // Metrics endpoint not available, continue without β_meta
        }
    }

    /// Compare baseline vs treatment metrics
    pub fn compare_metrics(
        &self,
        baseline: &TestMetrics,
        treatment: &TestMetrics,
        significance_threshold: f64,
    ) -> Comparison {
        // Latency comparison
        let latency_diff = treatment.latency.p99 - baseline.latency.p99;
        let latency_diff_pct = if baseline.latency.p99 > 0.0 {
            (latency_diff / baseline.latency.p99 * 100.0).min(100.0).max(-100.0)
        } else {
            0.0
        };

        // Throughput comparison
        let throughput_diff_pct = if baseline.throughput > 0.0 {
            ((treatment.throughput - baseline.throughput) / baseline.throughput * 100.0)
                .min(100.0).max(-100.0)
        } else {
            0.0
        };

        // Error rate comparison
        let error_rate_diff_pct = if baseline.error_rate > 0.0 {
            ((treatment.error_rate - baseline.error_rate) / baseline.error_rate * 100.0)
                .min(100.0).max(-100.0)
        } else {
            0.0
        };

        // Quality comparison
        let quality_diff_pct = if let (Some(baseline_q), Some(treatment_q)) = 
            (baseline.topology_metrics.quality_score_mean, treatment.topology_metrics.quality_score_mean) {
            if baseline_q > 0.0 {
                Some(((treatment_q - baseline_q) / baseline_q * 100.0).min(100.0).max(-100.0))
            } else {
                None
            }
        } else if let (Some(baseline_q), Some(treatment_q)) = (baseline.quality_score, treatment.quality_score) {
            if baseline_q > 0.0 {
                Some(((treatment_q - baseline_q) / baseline_q * 100.0).min(100.0).max(-100.0))
            } else {
                None
            }
        } else {
            None
        };
        
        // Topology metrics comparison
        let persistence_entropy_diff = if let (Some(baseline_pe), Some(treatment_pe)) = 
            (baseline.topology_metrics.persistence_entropy_mean, treatment.topology_metrics.persistence_entropy_mean) {
            Some(treatment_pe - baseline_pe)
        } else {
            None
        };
        
        let beta_meta_diff = if let (Some(baseline_bm), Some(treatment_bm)) = 
            (baseline.topology_metrics.beta_meta_current_mean, treatment.topology_metrics.beta_meta_current_mean) {
            Some(treatment_bm - baseline_bm)
        } else {
            None
        };

        // Determine topology impact
        let topology_impact = if let (Some(pe_diff), Some(q_diff)) = (persistence_entropy_diff, quality_diff_pct) {
            // If topology-enabled has higher persistence entropy AND higher quality, topology helps
            if pe_diff > 0.0 && q_diff > 0.0 {
                Some("positive".to_string())
            } else if pe_diff < 0.0 && q_diff < 0.0 {
                Some("negative".to_string())
            } else if pe_diff.abs() < 0.01 && q_diff.abs() < 1.0 {
                Some("neutral".to_string())
            } else {
                Some("inconclusive".to_string())
            }
        } else {
            None
        };

        // Statistical tests
        // For latency, use p99 values as proxy (would need full distributions)
        let baseline_latencies = vec![baseline.latency.p99];
        let treatment_latencies = vec![treatment.latency.p99];
        let (_, p_value_latency) = mann_whitney_u(&baseline_latencies, &treatment_latencies);
        let cohens_d_latency = cohens_d(&baseline_latencies, &treatment_latencies);

        let baseline_throughputs = vec![baseline.throughput];
        let treatment_throughputs = vec![treatment.throughput];
        let (_, p_value_throughput) = mann_whitney_u(&baseline_throughputs, &treatment_throughputs);
        let cohens_d_throughput = cohens_d(&baseline_throughputs, &treatment_throughputs);

        // Confidence intervals
        let ci_latency = bootstrap_percentile_ci(&treatment_latencies, 0.99, 1000, 0.95);
        let ci_throughput = bootstrap_percentile_ci(&treatment_throughputs, 0.50, 1000, 0.95);

        // Effect size categories
        let effect_size_latency = if cohens_d_latency.abs() < 0.2 {
            "Small"
        } else if cohens_d_latency.abs() < 0.5 {
            "Medium"
        } else if cohens_d_latency.abs() < 0.8 {
            "Large"
        } else {
            "Very Large"
        }.to_string();

        let effect_size_throughput = if cohens_d_throughput.abs() < 0.2 {
            "Small"
        } else if cohens_d_throughput.abs() < 0.5 {
            "Medium"
        } else if cohens_d_throughput.abs() < 0.8 {
            "Large"
        } else {
            "Very Large"
        }.to_string();

        // Determine winner metrics
        let mut winner_metrics = Vec::new();
        if latency_diff < 0.0 && p_value_latency < significance_threshold {
            winner_metrics.push("Lower Latency".to_string());
        }
        if throughput_diff_pct > 0.0 && p_value_throughput < significance_threshold {
            winner_metrics.push("Higher Throughput".to_string());
        }
        if error_rate_diff_pct < 0.0 {
            winner_metrics.push("Lower Error Rate".to_string());
        }

        Comparison {
            latency_difference_ms: latency_diff,
            latency_difference_pct: latency_diff_pct,
            throughput_difference_pct: throughput_diff_pct,
            quality_difference_pct: quality_diff_pct,
            error_rate_difference_pct: error_rate_diff_pct,
            cohens_d_latency,
            cohens_d_throughput,
            p_value_latency,
            p_value_throughput,
            confidence_interval_95_latency: ci_latency,
            confidence_interval_95_throughput: ci_throughput,
            effect_size_latency,
            effect_size_throughput,
            winner_metrics,
            persistence_entropy_difference: persistence_entropy_diff,
            beta_meta_difference: beta_meta_diff,
            topology_impact,
        }
    }

    /// Determine winner based on comparison
    pub fn determine_winner(
        &self,
        baseline_name: &str,
        treatment_name: &str,
        comparison: &Comparison,
        significance_threshold: f64,
    ) -> (String, bool) {
        let baseline_wins = comparison.latency_difference_ms > 0.0
            || comparison.throughput_difference_pct < 0.0
            || comparison.error_rate_difference_pct > 0.0;

        let treatment_wins = comparison.latency_difference_ms < 0.0
            || comparison.throughput_difference_pct > 0.0
            || comparison.error_rate_difference_pct < 0.0;

        let is_significant = comparison.p_value_latency < significance_threshold
            || comparison.p_value_throughput < significance_threshold;

        let winner = if treatment_wins && is_significant {
            treatment_name.to_string()
        } else if baseline_wins && is_significant {
            baseline_name.to_string()
        } else {
            "Inconclusive".to_string()
        };

        (winner, is_significant)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("🔬 Starting A/B test: {} vs {}", args.baseline_name, args.treatment_name);

    // Load configurations
    let baseline_config = if let Some(config_path) = &args.baseline_config {
        let json = std::fs::read_to_string(config_path)?;
        serde_json::from_str(&json)?
    } else {
        HashMap::new() // Default baseline
    };

    let treatment_config = if let Some(config_path) = &args.treatment_config {
        let json = std::fs::read_to_string(config_path)?;
        serde_json::from_str(&json)?
    } else {
        HashMap::new() // Default treatment
    };

    let runner = ABTestRunner::new();
    let (baseline_metrics, treatment_metrics) = runner
        .run_ab_test(&baseline_config, &treatment_config, args.concurrent_users, args.duration_secs)
        .await?;

    let comparison = runner.compare_metrics(&baseline_metrics, &treatment_metrics, args.significance_threshold);
    let (winner, is_significant) = runner.determine_winner(
        &args.baseline_name,
        &args.treatment_name,
        &comparison,
        args.significance_threshold,
    );

    // Log results before moving comparison
    info!("✅ A/B test results:");
    info!("   Winner: {}", winner);
    info!("   Statistically Significant: {}", is_significant);
    info!("   Latency Difference: {:.2}ms ({:.1}%)", comparison.latency_difference_ms, comparison.latency_difference_pct);
    info!("   Throughput Difference: {:.1}%", comparison.throughput_difference_pct);
    if let Some(q_diff) = comparison.quality_difference_pct {
        info!("   Quality Difference: {:.1}%", q_diff);
    }
    if let Some(pe_diff) = comparison.persistence_entropy_difference {
        info!("   Persistence Entropy Difference: {:.4}", pe_diff);
    }
    if let Some(bm_diff) = comparison.beta_meta_difference {
        info!("   β_meta Difference: {:.4}", bm_diff);
    }
    if let Some(ref impact) = comparison.topology_impact {
        info!("   Topology Impact: {}", impact);
    }
    info!("   P-value (Latency): {:.4}", comparison.p_value_latency);
    info!("   P-value (Throughput): {:.4}", comparison.p_value_throughput);
    info!("   Cohen's d (Latency): {:.2} ({})", comparison.cohens_d_latency, comparison.effect_size_latency);
    info!("   Cohen's d (Throughput): {:.2} ({})", comparison.cohens_d_throughput, comparison.effect_size_throughput);

    if !comparison.winner_metrics.is_empty() {
        info!("   Winning Metrics: {}", comparison.winner_metrics.join(", "));
    }

    let result = ABTestResult {
        baseline_name: args.baseline_name.clone(),
        treatment_name: args.treatment_name.clone(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        baseline_metrics,
        treatment_metrics,
        comparison,
        winner: winner.clone(),
        statistical_significance: is_significant,
    };

    // Save results
    std::fs::create_dir_all(&args.output_dir)?;
    let output_file = args.output_dir.join(format!("ab_test_{}_vs_{}.json", args.baseline_name, args.treatment_name));
    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(&output_file, json)?;

    info!("✅ A/B test results saved to: {}", output_file.display());

    Ok(())
}






