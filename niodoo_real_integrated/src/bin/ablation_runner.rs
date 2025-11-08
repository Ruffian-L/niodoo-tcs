//! Ablation Runner for Systematic Component Testing
//!
//! Programmatically sets ablation flags and runs comparative tests to quantify
//! component contributions to performance and cognitive capabilities.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use tracing::info;

use niodoo_real_integrated::config::CliArgs;
use niodoo_real_integrated::pipeline::Pipeline;
use niodoo_real_integrated::validation::stats::{
    StatisticalSummary, cohens_d, mann_whitney_u, bootstrap_percentile_ci,
    requires_regression_action,
};

#[derive(Parser, Debug)]
#[command(name = "ablation_runner")]
#[command(about = "Systematic ablation testing for niodoo_real_integrated components")]
struct Args {
    /// Ablation experiment to run
    #[arg(long, value_enum)]
    experiment: AblationExperiment,

    /// Output directory for results
    #[arg(long, default_value = "ablation_results")]
    output_dir: PathBuf,

    /// Baseline metrics file to compare against
    #[arg(long)]
    baseline: Option<PathBuf>,

    /// Number of concurrent users for load test
    #[arg(long, default_value = "16")]
    concurrent_users: usize,

    /// Duration of test in seconds
    #[arg(long, default_value = "60")]
    duration_secs: u64,
}

#[derive(Debug, Clone, ValueEnum)]
enum AblationExperiment {
    /// ABL-001: Disable RCE layer
    DisableRce,
    /// ABL-002: Bypass nTokens layer
    BypassNTokens,
    /// ABL-003: Disable GPU acceleration for TCS
    DisableTcsGpu,
    /// ABL-004: Disable GPU fitness calculation
    DisableGpuFitness,
    /// ABL-005: Disable Curator
    DisableCurator,
    /// ABL-006: Bypass ERAG (zero-shot mode)
    BypassErag,
    /// ABL-007: Disable Compass engine
    DisableCompass,
    /// ABL-008: Disable Learning loop
    DisableLearning,
    /// ABL-009: Disable TCS topology analysis
    DisableTcs,
    /// ABL-010: Disable Tokenizer promotion
    DisableTokenizer,
    /// ABL-011: Multi-component: Disable RCE + nTokens
    DisableRceAndNTokens,
    /// ABL-012: Multi-component: Disable Curator + ERAG
    DisableCuratorAndErag,
}

#[derive(Debug, Serialize, Deserialize)]
struct AblationResult {
    experiment: String,
    timestamp: String,
    config: AblationConfig,
    metrics: AblationMetrics,
    comparison: Option<ComparisonResult>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AblationConfig {
    rce_enabled: bool,
    n_tokens_bypass: bool,
    tcs_gpu_enabled: bool,
    use_gpu_fitness: bool,
    enable_curator: bool,
    erag_bypass: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct AblationMetrics {
    latency: StatisticalSummary,
    throughput: f64,
    quality_slis: QualitySLISummary,
    topological: TopologicalSummary,
}

#[derive(Debug, Serialize, Deserialize)]
struct QualitySLISummary {
    tcs_stability_cv: Option<f64>,
    rce_beta_meta_compliance: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TopologicalSummary {
    persistence_entropy: Option<f64>,
    beta_meta_current: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ComparisonResult {
    latency_change_p99_ms: f64,
    latency_change_pct: f64,
    throughput_change_pct: f64,
    cohens_d_latency: f64,
    p_value: f64,
    confidence_interval_95: (f64, f64),
    regression_detected: bool,
    component_contribution_score: f64,
    superiority_proof: SuperiorityProof,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SuperiorityProof {
    component_name: String,
    performance_impact_pct: f64,
    quality_impact_pct: f64,
    cognitive_impact_pct: f64,
    statistical_significance: bool,
    effect_size_category: String,
    recommendation: String,
}

pub struct AblationRunner;

impl AblationRunner {
    pub fn new() -> Self {
        Self
    }

    /// Run a single ablation experiment
    pub async fn run_experiment(
        &self,
        experiment: AblationExperiment,
        concurrent_users: usize,
        duration_secs: u64,
    ) -> Result<AblationResult> {
        // Set environment variables based on experiment
        let config = self.setup_experiment_config(&experiment)?;

        // Initialize pipeline with ablation config
        let cli_args = CliArgs::default();
        
        let pipeline = Pipeline::initialise(cli_args).await
            .context("Failed to initialize pipeline for ablation test")?;

        // Run actual load test by executing pipeline cycles
        // This replaces the placeholder metrics with real execution data
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;
        use tokio::sync::Mutex as AsyncMutex;
        use std::time::{Duration, Instant};
        
        let pipeline_mutex = Arc::new(AsyncMutex::new(pipeline));
        let latencies = Arc::new(AsyncMutex::new(Vec::new()));
        let request_count = Arc::new(AtomicU64::new(0));
        let start_time = Instant::now();
        let end_time = start_time + Duration::from_secs(duration_secs);
        
        // Generate test prompts
        let test_prompts = vec![
            "Explain quantum computing in detail".to_string(),
            "Describe the theory of relativity".to_string(),
            "What is machine learning?".to_string(),
        ];
        
        // Run concurrent load test
        let mut handles = Vec::new();
        for user_id in 0..concurrent_users {
            let pipeline_clone = pipeline_mutex.clone();
            let latencies_clone = latencies.clone();
            let request_count_clone = request_count.clone();
            let prompts = test_prompts.clone();
            
            let handle = tokio::spawn(async move {
                let mut prompt_idx = 0;
                while Instant::now() < end_time {
                    let prompt = prompts[prompt_idx % prompts.len()].clone();
                    prompt_idx += 1;
                    
                    let cycle_start = Instant::now();
                    let mut pipeline_guard = pipeline_clone.lock().await;
                    
                    match pipeline_guard.process_prompt(prompt.as_str()).await {
                        Ok(_cycle) => {
                            let latency_ms = cycle_start.elapsed().as_secs_f64() * 1000.0;
                            latencies_clone.lock().await.push(latency_ms);
                            request_count_clone.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            tracing::warn!(user_id, error = %e, "Ablation test request failed");
                        }
                    }
                    
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            });
            handles.push(handle);
        }
        
        // Wait for all workers
        for handle in handles {
            let _ = handle.await;
        }
        
        let actual_duration = start_time.elapsed().as_secs_f64();
        let latencies_vec = latencies.lock().await.clone();
        let requests = request_count.load(Ordering::Relaxed);
        
        // Compute real metrics from execution
        let latency_stats = if !latencies_vec.is_empty() {
            let mut sorted = latencies_vec.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p50_idx = (sorted.len() as f64 * 0.5) as usize;
            let p95_idx = (sorted.len() as f64 * 0.95) as usize;
            let p99_idx = (sorted.len() as f64 * 0.99).min(sorted.len() as f64 - 1.0) as usize;
            
            let mean = latencies_vec.iter().sum::<f64>() / latencies_vec.len() as f64;
            let variance = latencies_vec.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / latencies_vec.len() as f64;
            let std_dev = variance.sqrt();
            let median_idx = sorted.len() / 2;
            let median = if sorted.len() % 2 == 0 {
                (sorted[median_idx - 1] + sorted[median_idx]) / 2.0
            } else {
                sorted[median_idx]
            };
            
            StatisticalSummary {
                mean,
                median,
                std_dev,
                p50: sorted[p50_idx],
                p95: sorted[p95_idx],
                p99: sorted[p99_idx],
                min: sorted[0],
                max: sorted[sorted.len() - 1],
                count: latencies_vec.len(),
            }
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
        
        let metrics = AblationMetrics {
            latency: latency_stats,
            throughput,
            quality_slis: QualitySLISummary {
                tcs_stability_cv: None, // Would need to extract from pipeline metrics
                rce_beta_meta_compliance: None,
            },
            topological: TopologicalSummary {
                persistence_entropy: None, // Would need to extract from pipeline metrics
                beta_meta_current: None,
            },
        };

        Ok(AblationResult {
            experiment: format!("{:?}", experiment),
            timestamp: chrono::Utc::now().to_rfc3339(),
            config,
            metrics,
            comparison: None,
        })
    }

    fn setup_experiment_config(&self, experiment: &AblationExperiment) -> Result<AblationConfig> {
        // Default config (all enabled)
        let mut config = AblationConfig {
            rce_enabled: true,
            n_tokens_bypass: false,
            tcs_gpu_enabled: true,
            use_gpu_fitness: true,
            enable_curator: true,
            erag_bypass: false,
        };

        // Apply experiment-specific changes
        match experiment {
            AblationExperiment::DisableRce => {
                config.rce_enabled = false;
                std::env::set_var("RCE_ENABLED", "0");
            }
            AblationExperiment::BypassNTokens => {
                config.n_tokens_bypass = true;
                std::env::set_var("N_TOKENS_BYPASS", "1");
            }
            AblationExperiment::DisableTcsGpu => {
                config.tcs_gpu_enabled = false;
                std::env::set_var("TCS_ENABLE_GPU", "0");
            }
            AblationExperiment::DisableGpuFitness => {
                config.use_gpu_fitness = false;
                std::env::set_var("USE_GPU_FITNESS", "0");
            }
            AblationExperiment::DisableCurator => {
                config.enable_curator = false;
                std::env::set_var("ENABLE_CURATOR", "0");
            }
            AblationExperiment::BypassErag => {
                config.erag_bypass = true;
                std::env::set_var("ERAG_BYPASS", "1");
            }
            AblationExperiment::DisableCompass => {
                std::env::set_var("COMPASS_BYPASS", "1");
            }
            AblationExperiment::DisableLearning => {
                std::env::set_var("LEARNING_BYPASS", "1");
            }
            AblationExperiment::DisableTcs => {
                std::env::set_var("TOPOLOGY_MODE", "baseline");
            }
            AblationExperiment::DisableTokenizer => {
                std::env::set_var("TOKENIZER_BYPASS", "1");
            }
            AblationExperiment::DisableRceAndNTokens => {
                config.rce_enabled = false;
                config.n_tokens_bypass = true;
                std::env::set_var("RCE_ENABLED", "0");
                std::env::set_var("N_TOKENS_BYPASS", "1");
            }
            AblationExperiment::DisableCuratorAndErag => {
                config.enable_curator = false;
                config.erag_bypass = true;
                std::env::set_var("ENABLE_CURATOR", "0");
                std::env::set_var("ERAG_BYPASS", "1");
            }
        }

        Ok(config)
    }

    /// Compare ablation result with baseline metrics using statistical tests
    pub fn compare_with_baseline(
        &self,
        ablation: &AblationResult,
        baseline_latencies: &[f64],
        baseline_throughput: f64,
    ) -> ComparisonResult {
        let ablation_latencies = vec![ablation.metrics.latency.p99]; // Simplified - would need full distribution
        let baseline_p99 = baseline_latencies.iter().fold(0.0, |acc, x| acc.max(*x));
        
        let latency_change = ablation.metrics.latency.p99 - baseline_p99;
        let latency_change_pct = if baseline_p99 > 0.0 {
            (latency_change / baseline_p99 * 100.0).min(100.0).max(-100.0)
        } else {
            0.0
        };

        let throughput_change = if baseline_throughput > 0.0 {
            (ablation.metrics.throughput - baseline_throughput) 
                / baseline_throughput * 100.0
        } else {
            0.0
        };

        // Statistical significance testing
        let (_, p_value) = mann_whitney_u(baseline_latencies, &ablation_latencies);
        let cohens_d_val = cohens_d(baseline_latencies, &ablation_latencies);
        
        // Bootstrap confidence interval
        let confidence_interval_95 = bootstrap_percentile_ci(
            &ablation_latencies,
            0.99,
            1000,
            0.95,
        );

        // Regression detection with statistical rigor
        let regression_detected = requires_regression_action(
            baseline_latencies,
            &ablation_latencies,
            0.05, // p-value threshold
            0.5,  // effect size threshold
        ) || latency_change > 100.0 || latency_change_pct > 20.0;

        // Component contribution scoring (higher = more important)
        let component_contribution_score = cohens_d_val.abs() * (1.0 - p_value.min(1.0));

        // Determine effect size category
        let effect_size_category = if cohens_d_val.abs() < 0.2 {
            "Small".to_string()
        } else if cohens_d_val.abs() < 0.5 {
            "Medium".to_string()
        } else if cohens_d_val.abs() < 0.8 {
            "Large".to_string()
        } else {
            "Very Large".to_string()
        };

        // Generate superiority proof
        let component_name = ablation.experiment.clone();
        let performance_impact_pct = -latency_change_pct; // Negative = worse performance
        let quality_impact_pct = 0.0; // Would need quality metrics
        let cognitive_impact_pct = 0.0; // Would need cognitive metrics
        
        let recommendation = if regression_detected && p_value < 0.05 {
            format!("CRITICAL: Component is essential. Removing it causes significant degradation (p={:.4}, d={:.2})", p_value, cohens_d_val)
        } else if cohens_d_val.abs() > 0.5 {
            format!("IMPORTANT: Component provides substantial value (effect size: {})", effect_size_category)
        } else {
            "OPTIONAL: Component has minimal measurable impact".to_string()
        };

        ComparisonResult {
            latency_change_p99_ms: latency_change,
            latency_change_pct,
            throughput_change_pct: throughput_change,
            cohens_d_latency: cohens_d_val,
            p_value,
            confidence_interval_95,
            regression_detected,
            component_contribution_score,
            superiority_proof: SuperiorityProof {
                component_name,
                performance_impact_pct,
                quality_impact_pct,
                cognitive_impact_pct,
                statistical_significance: p_value < 0.05,
                effect_size_category,
                recommendation,
            },
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("🧪 Running ablation experiment: {:?}", args.experiment);

    let runner = AblationRunner::new();
    let mut result = runner
        .run_experiment(args.experiment.clone(), args.concurrent_users, args.duration_secs)
        .await?;

    // Compare with baseline if provided
    if let Some(baseline_path) = &args.baseline {
        let baseline_json = std::fs::read_to_string(baseline_path)?;
        let baseline: serde_json::Value = serde_json::from_str(&baseline_json)?;
        
        // Extract baseline latencies array if available, otherwise use single value
        let baseline_latencies = if let Some(latencies_array) = baseline["latency"]["values"].as_array() {
            latencies_array.iter()
                .filter_map(|v| v.as_f64())
                .collect::<Vec<f64>>()
        } else {
            // Fallback to single p99 value
            let p99 = baseline["latency"]["p99_ms"].as_f64().unwrap_or(0.0);
            vec![p99]
        };
        
        let baseline_throughput = baseline["throughput"]["tokens_per_sec"].as_f64().unwrap_or(0.0);
        
        let comparison = runner.compare_with_baseline(&result, &baseline_latencies, baseline_throughput);
        result.comparison = Some(comparison);
    }

    // Save results
    std::fs::create_dir_all(&args.output_dir)?;
    let output_file = args.output_dir.join(format!("ablation-{:?}.json", args.experiment));
    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(&output_file, json)?;

    info!("✅ Ablation results saved to: {}", output_file.display());

    if let Some(ref comparison) = result.comparison {
        info!("📊 Statistical Analysis:");
        info!("   P-value: {:.4}", comparison.p_value);
        info!("   Cohen's d: {:.2} ({})", comparison.cohens_d_latency, comparison.superiority_proof.effect_size_category);
        info!("   95% CI: [{:.2}, {:.2}]", comparison.confidence_interval_95.0, comparison.confidence_interval_95.1);
        info!("   Component Contribution Score: {:.2}", comparison.component_contribution_score);
        
        if comparison.regression_detected {
            eprintln!("⚠️  REGRESSION DETECTED: Latency increased by {:.2}ms ({:.1}%)", 
                comparison.latency_change_p99_ms,
                comparison.latency_change_pct);
        } else {
            info!("✅ No significant regression detected");
        }
        
        info!("💡 Recommendation: {}", comparison.superiority_proof.recommendation);
    }

    Ok(())
}

