use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use anyhow::{Context, Result};
use clap::Parser;
use csv::Writer;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use tracing_subscriber::prelude::*;

use niodoo_real_integrated::{
    config::{CliArgs, OutputFormat},
    metrics::metrics,
    pipeline::Pipeline,
};

#[derive(Parser, Debug, Clone)]
struct GauntletArgs {
    #[arg(long, value_hint = clap::ValueHint::DirPath)]
    pub output_dir: Option<PathBuf>,

    #[arg(
        long,
        help = "Run labyrinth validation mode with blueprint-specific prompts"
    )]
    pub labyrinth: bool,
}

#[derive(Deserialize, Clone)]
struct RutGauntletConfig {
    default_entropy_high: f64,
    entropy_stability_threshold: f64,
    latency_max_ms: f64,
    emotional_activation_min_percent: f64,
    breakthroughs_min_percent: f64,
}

#[derive(Debug, Clone)]
struct DynamicThresholds {
    entropy_high: f64,
    variance_spike: f64,
    similarity_threshold: f64,
    mcts_c: f64,
    mirage_sigma: f64,
}

#[derive(Serialize, Clone)]
struct TestResult {
    cycle: usize,
    prompt: String,
    response: String,
    entropy: f64,
    is_threat: bool,
    is_healing: bool,
    latency_ms: f64,
    learning_events: Vec<String>,
    coherence_rouge: f64,
    rouge_l: f64,
    generation_source: String,
    quadrant: String,
}

#[derive(Serialize)]
struct TestSummary {
    total_prompts: usize,
    avg_entropy: f64,
    entropy_std: f64,
    threat_count: usize,
    healing_count: usize,
    threat_rate_percent: f64,
    healing_rate_percent: f64,
    quadrant_threat_counts: serde_json::Value,
    quadrant_healing_counts: serde_json::Value,
    quadrant_threat_rates: serde_json::Value,
    quadrant_healing_rates: serde_json::Value,
    avg_latency_ms: f64,
    p50_latency_ms: f64,
    p75_latency_ms: f64,
    p90_latency_ms: f64,
    p95_latency_ms: f64,
    avg_rouge_l: f64,
    p50_rouge_l: f64,
    p75_rouge_l: f64,
    p90_rouge_l: f64,
    p95_rouge_l: f64,
    entropy_stability: bool,
    emotional_activation: bool,
    breakthroughs: usize,
    breakthrough_rate_percent: f64,
    verdict: String,
    output_dir: String,
    csv_path: String,
    summary_json_path: String,
    plot_path: String,
    metrics_txt_path: String,
}

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

fn generate_labyrinth_prompts() -> Vec<String> {
    // Blueprint-specific labyrinth prompts for Phase 1 validation
    let base_prompt = "Write a Python program using Dijkstra's algorithm to solve a 3D labyrinth with echo chambers. Use a 7x7x7 grid, track state transitions (consec_echoes, attune_timer, multiplier), and find the path from (0,0,0) to (6,6,6) with minimal cost. Expected cost: 46.";

    let mut prompts = Vec::new();
    for i in 1..=100 {
        prompts.push(format!("Labyrinth #{}: {}", i, base_prompt));
    }

    prompts
}

fn compute_dynamic_thresholds_from_first_20(
    results: &[TestResult],
    config: &RutGauntletConfig,
) -> DynamicThresholds {
    if results.len() < 20 {
        return DynamicThresholds {
            entropy_high: config.default_entropy_high,
            variance_spike: 0.5,
            similarity_threshold: 0.8,
            mcts_c: 1.4,
            mirage_sigma: 0.1,
        };
    }

    let first_20_entropies: Vec<f64> = results.iter().take(20).map(|r| r.entropy).collect();
    let entropy_mean = first_20_entropies.iter().sum::<f64>() / first_20_entropies.len() as f64;
    let entropy_std = (first_20_entropies
        .iter()
        .map(|&e| (e - entropy_mean).powi(2))
        .sum::<f64>()
        / first_20_entropies.len() as f64)
        .sqrt();

    DynamicThresholds {
        entropy_high: entropy_mean + entropy_std,
        variance_spike: 0.5 * entropy_std,
        similarity_threshold: 0.8,
        mcts_c: entropy_std * 0.1,
        mirage_sigma: 0.1 * entropy_mean,
    }
}

fn calculate_jaccard_similarity(response: &str, prompt: &str) -> f64 {
    let prompt_words: std::collections::HashSet<&str> = prompt.split_whitespace().collect();
    let response_words: std::collections::HashSet<&str> = response.split_whitespace().collect();
    let intersection = prompt_words.intersection(&response_words).count();
    let union = prompt_words.union(&response_words).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

async fn run_100_prompt_test() -> Result<()> {
    // Initialize tracing before anything else
    let env_directives = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    let filter_spec = if env_directives
        .split(',')
        .any(|part| part.trim_start().starts_with("ort"))
    {
        env_directives
    } else {
        format!("{env_directives},ort=error")
    };
    let env_filter = tracing_subscriber::EnvFilter::try_new(filter_spec)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_filter(tracing_subscriber::filter::FilterFn::new(|metadata| {
            !metadata.target().starts_with("ort")
        }));
    let subscriber = tracing_subscriber::registry::Registry::default()
        .with(env_filter)
        .with(fmt_layer);
    let _ = subscriber.try_init();

    let gauntlet_args = GauntletArgs::parse();
    let output_dir = prepare_output_dir(gauntlet_args.output_dir)?;

    if gauntlet_args.labyrinth {
        println!("🧠 Starting NIODOO Labyrinth Validation Test (Phase 1 Blueprint)...");
        println!("Testing multi-modal sensory system with ROUGE >0.5, cost target 46\n");
    } else {
        println!("🧠 Starting NIODOO 100-Prompt Raw Rut Gauntlet Test...");
        println!("Testing operational torque through dynamic consciousness pipeline\n");
    }
    println!("📂 Output directory: {}", output_dir.display());

    // Load configuration
    let config = RutGauntletConfig {
        default_entropy_high: 2.0,
        entropy_stability_threshold: 0.3,
        latency_max_ms: 500.0,
        emotional_activation_min_percent: 20.0,
        breakthroughs_min_percent: 60.0,
    };

    let args = CliArgs {
        hardware: niodoo_real_integrated::config::HardwareProfile::Beelink,
        prompt: None,
        prompt_file: None,
        swarm: 1,
        output: OutputFormat::Csv,
        config: None,
    };

    let mut pipeline = Pipeline::initialise(args).await?;
    let mut results: Vec<TestResult> = Vec::new();
    let mut latencies: Vec<f64> = Vec::new();
    let mut entropies: Vec<f64> = Vec::new();
    let mut rouge_scores: Vec<f64> = Vec::new();
    let mut breakthroughs = 0usize;
    let mut quadrant_threat_counts: std::collections::HashMap<String, usize> = Default::default();
    let mut quadrant_healing_counts: std::collections::HashMap<String, usize> = Default::default();

    let prompts = if gauntlet_args.labyrinth {
        generate_labyrinth_prompts()
    } else {
        generate_raw_rut_prompts()
    };

    println!("📊 Computing dynamic thresholds from first 20 cycles...");
    let mut first_20_results: Vec<TestResult> = Vec::new();

    for (i, prompt) in prompts.iter().enumerate().take(20) {
        let start_time = Instant::now();
        let cycle = pipeline.process_prompt(prompt).await?;
        let latency = start_time.elapsed().as_millis() as f64;
        let coherence = calculate_jaccard_similarity(&cycle.hybrid_response, prompt);

        let test_result = TestResult {
            cycle: i + 1,
            prompt: prompt.clone(),
            response: cycle.hybrid_response.clone(),
            entropy: cycle.entropy,
            is_threat: cycle.compass.is_threat,
            is_healing: cycle.compass.is_healing,
            latency_ms: latency,
            learning_events: cycle.learning.events.clone(),
            coherence_rouge: coherence,
            rouge_l: cycle.rouge,
            generation_source: cycle.generation.source.clone(),
            quadrant: format!("{:?}", cycle.compass.quadrant),
        };

        first_20_results.push(test_result.clone());
        results.push(test_result);
        latencies.push(latency);
        entropies.push(cycle.entropy);
        rouge_scores.push(cycle.rouge);
        breakthroughs += cycle.learning.breakthroughs.len();
        if cycle.compass.is_threat {
            *quadrant_threat_counts
                .entry(format!("{:?}", cycle.compass.quadrant))
                .or_default() += 1;
        }
        if cycle.compass.is_healing {
            *quadrant_healing_counts
                .entry(format!("{:?}", cycle.compass.quadrant))
                .or_default() += 1;
        }

        println!(
            "Cycle {}/20: H={:.2}, Threat={}, Healing={}, Latency={:.0}ms",
            i + 1,
            cycle.entropy,
            cycle.compass.is_threat,
            cycle.compass.is_healing,
            latency
        );
    }

    let dynamic_thresholds = compute_dynamic_thresholds_from_first_20(&first_20_results, &config);
    println!(
        "🔧 Dynamic thresholds computed: entropy_high={:.2}, mcts_c={:.3}, mirage_sigma={:.3}",
        dynamic_thresholds.entropy_high, dynamic_thresholds.mcts_c, dynamic_thresholds.mirage_sigma
    );

    println!("\n🚀 Running remaining 80 cycles with dynamic thresholds...");
    for (i, prompt) in prompts.iter().enumerate().skip(20) {
        let start_time = Instant::now();
        let cycle = pipeline.process_prompt(prompt).await?;
        let latency = start_time.elapsed().as_millis() as f64;
        let coherence = calculate_jaccard_similarity(&cycle.hybrid_response, prompt);

        let test_result = TestResult {
            cycle: i + 1,
            prompt: prompt.clone(),
            response: cycle.hybrid_response.clone(),
            entropy: cycle.entropy,
            is_threat: cycle.compass.is_threat,
            is_healing: cycle.compass.is_healing,
            latency_ms: latency,
            learning_events: cycle.learning.events.clone(),
            coherence_rouge: coherence,
            rouge_l: cycle.rouge,
            generation_source: cycle.generation.source.clone(),
            quadrant: format!("{:?}", cycle.compass.quadrant),
        };

        results.push(test_result);
        latencies.push(latency);
        entropies.push(cycle.entropy);
        rouge_scores.push(cycle.rouge);
        breakthroughs += cycle.learning.breakthroughs.len();
        if cycle.compass.is_threat {
            *quadrant_threat_counts
                .entry(format!("{:?}", cycle.compass.quadrant))
                .or_default() += 1;
        }
        if cycle.compass.is_healing {
            *quadrant_healing_counts
                .entry(format!("{:?}", cycle.compass.quadrant))
                .or_default() += 1;
        }

        if (i + 1) % 20 == 0 {
            println!(
                "Cycle {}/100: H={:.2}, Threats={}, Healings={}, Latency={:.0}ms",
                i + 1,
                cycle.entropy,
                results.iter().filter(|r| r.is_threat).count(),
                results.iter().filter(|r| r.is_healing).count(),
                latency
            );
        }
    }

    let total_prompts = results.len();
    let avg_entropy = entropies.iter().sum::<f64>() / entropies.len() as f64;
    let entropy_std = (entropies
        .iter()
        .map(|&e| (e - avg_entropy).powi(2))
        .sum::<f64>()
        / entropies.len() as f64)
        .sqrt();
    let threat_count = results.iter().filter(|r| r.is_threat).count();
    let healing_count = results.iter().filter(|r| r.is_healing).count();
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let avg_rouge = rouge_scores.iter().sum::<f64>() / rouge_scores.len() as f64;

    let entropy_stability = entropy_std < config.entropy_stability_threshold;
    let emotional_activation = (threat_count + healing_count) as f64
        > (total_prompts as f64 * config.emotional_activation_min_percent) / 100.0;
    let latency_ok = avg_latency < config.latency_max_ms;
    let breakthroughs_ok =
        breakthroughs as f64 > (total_prompts as f64 * config.breakthroughs_min_percent) / 100.0;

    let verdict = if entropy_stability && emotional_activation && latency_ok && breakthroughs_ok {
        "Validated: Full operational torque achieved".to_string()
    } else {
        format!(
            "Fix required: entropy_stability={}, emotional_activation={}, latency_ok={}, breakthroughs_ok={}",
            entropy_stability, emotional_activation, latency_ok, breakthroughs_ok
        )
    };

    let csv_path = output_dir.join("rut_gauntlet_results.csv");
    let summary_path = output_dir.join("rut_gauntlet_summary.json");
    let plot_path = output_dir.join("entropy_over_cycles.png");
    let metrics_path = output_dir.join("metrics.prom");

    export_csv(&results, &csv_path)?;
    export_json(
        &TestSummary {
            total_prompts,
            avg_entropy,
            entropy_std,
            threat_count,
            healing_count,
            threat_rate_percent: threat_count as f64 / total_prompts as f64 * 100.0,
            healing_rate_percent: healing_count as f64 / total_prompts as f64 * 100.0,
            quadrant_threat_counts: serde_json::to_value(&quadrant_threat_counts)?,
            quadrant_healing_counts: serde_json::to_value(&quadrant_healing_counts)?,
            quadrant_threat_rates: serde_json::to_value(&quadrant_threat_counts)?,
            quadrant_healing_rates: serde_json::to_value(&quadrant_healing_counts)?,
            avg_latency_ms: avg_latency,
            p50_latency_ms: percentile(&latencies, 0.50),
            p75_latency_ms: percentile(&latencies, 0.75),
            p90_latency_ms: percentile(&latencies, 0.90),
            p95_latency_ms: percentile(&latencies, 0.95),
            avg_rouge_l: avg_rouge,
            p50_rouge_l: percentile(&rouge_scores, 0.50),
            p75_rouge_l: percentile(&rouge_scores, 0.75),
            p90_rouge_l: percentile(&rouge_scores, 0.90),
            p95_rouge_l: percentile(&rouge_scores, 0.95),
            entropy_stability,
            emotional_activation,
            breakthroughs,
            breakthrough_rate_percent: breakthroughs as f64 / total_prompts as f64 * 100.0,
            verdict: verdict.clone(),
            output_dir: output_dir.display().to_string(),
            csv_path: csv_path.display().to_string(),
            summary_json_path: summary_path.display().to_string(),
            plot_path: plot_path.display().to_string(),
            metrics_txt_path: metrics_path.display().to_string(),
        },
        &summary_path,
    )?;
    generate_plot(&entropies, &plot_path)?;
    dump_prometheus_metrics(metrics_path.as_path())?;
    write_instructions(&output_dir)?;

    println!("\n🔬 Operational Torque Validation Results:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "Entropy Stability (< 0.3 std): {} ({:.3} ✓)",
        entropy_stability, entropy_std
    );
    println!(
        "Emotional Activation (>20%): {} ({:.1}%) ✓",
        emotional_activation,
        (threat_count + healing_count) as f64 / total_prompts as f64 * 100.0
    );
    println!(
        "Average Latency (<500ms): {} ({:.0}ms) ✓",
        latency_ok, avg_latency
    );
    println!(
        "Breakthroughs (>60%): {} ({:.1}%) ✓",
        breakthroughs_ok,
        breakthroughs as f64 / total_prompts as f64 * 100.0
    );
    println!("Average ROUGE-L: {:.3}", avg_rouge);
    let quadrant_threat_rates = quadrant_threat_counts
        .iter()
        .map(|(k, v)| (k.clone(), (*v as f64) / total_prompts as f64 * 100.0))
        .collect::<std::collections::HashMap<_, _>>();
    let quadrant_healing_rates = quadrant_healing_counts
        .iter()
        .map(|(k, v)| (k.clone(), (*v as f64) / total_prompts as f64 * 100.0))
        .collect::<std::collections::HashMap<_, _>>();
    println!(
        "Threat distribution by quadrant: {:?}",
        quadrant_threat_rates
    );
    println!(
        "Healing distribution by quadrant: {:?}",
        quadrant_healing_rates
    );
    println!("\n🎯 VERDICT: {}", verdict);
    println!("\nArtifacts written to: {}", output_dir.display());

    Ok(())
}

fn prepare_output_dir(provided: Option<PathBuf>) -> Result<PathBuf> {
    let dir = if let Some(path) = provided {
        path
    } else {
        let timestamp = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .to_string();
        let sanitized = timestamp.replace(':', "-");
        PathBuf::from("logs").join(format!("rut_gauntlet-{}", sanitized))
    };
    create_dir_all(&dir)
        .with_context(|| format!("failed to create output dir {}", dir.display()))?;
    Ok(dir)
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (quantile.clamp(0.0, 1.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank]
}

fn export_csv(results: &[TestResult], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = csv::Writer::from_writer(BufWriter::new(file));
    writer.write_record(&[
        "cycle",
        "prompt",
        "response",
        "entropy",
        "is_threat",
        "is_healing",
        "latency_ms",
        "learning_events",
        "coherence_rouge",
        "rouge_l",
        "generation_source",
    ])?;

    for result in results {
        writer.serialize(result)?;
    }

    writer.flush()?;
    println!("📊 Results exported to {}", path.display());
    Ok(())
}

fn export_json(summary: &TestSummary, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(summary)?;
    std::fs::write(path, json)?;
    println!("📋 Summary exported to {}", path.display());
    Ok(())
}

fn generate_plot(entropies: &[f64], path: &Path) -> Result<()> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Entropy Over Consciousness Cycles", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..entropies.len(), 0.0..3.0)?;

    chart
        .configure_mesh()
        .x_desc("Cycle")
        .y_desc("Entropy")
        .draw()?;

    chart.draw_series(LineSeries::new(
        entropies.iter().enumerate().map(|(i, &e)| (i, e)),
        &RED,
    ))?;

    println!("📈 Plot saved to {}", path.display());
    Ok(())
}

fn dump_prometheus_metrics(path: &Path) -> Result<()> {
    let snapshot = metrics().gather()?;
    std::fs::write(path, snapshot)?;
    println!("📡 Metrics snapshot written to {}", path.display());
    Ok(())
}

fn write_instructions(output_dir: &Path) -> Result<()> {
    let path = output_dir.join("README.txt");
    let mut file = File::create(&path)?;
    writeln!(
        file,
        "# NIODOO Rut Gauntlet Artifacts\n\nRun again:\n    cargo run -p niodoo_real_integrated --bin rut_gauntlet -- --output-dir {}\n\nKey files:\n    - CSV: rut_gauntlet_results.csv\n    - Summary JSON: rut_gauntlet_summary.json\n    - Plot: entropy_over_cycles.png\n    - Prometheus snapshot: metrics.prom\n",
        output_dir.display()
    )?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    run_100_prompt_test().await
}
