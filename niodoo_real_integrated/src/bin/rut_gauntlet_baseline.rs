use std::fs::{File, create_dir_all};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use plotters::prelude::*;
use serde::Serialize;
use tracing_subscriber::prelude::*;

use niodoo_real_integrated::config::{CliArgs, OutputFormat, RuntimeConfig};
use niodoo_real_integrated::generation::GenerationEngine;
use niodoo_real_integrated::util::rouge_l;

#[derive(Parser, Debug, Clone)]
struct BaselineArgs {
    #[arg(long, value_hint = clap::ValueHint::DirPath)]
    pub output_dir: Option<PathBuf>,
}

#[derive(Serialize, Clone)]
struct BaselineResult {
    cycle: usize,
    prompt: String,
    response: String,
    latency_ms: f64,
    rouge_l: f64,
}

#[derive(Serialize)]
struct BaselineSummary {
    total_prompts: usize,
    avg_latency_ms: f64,
    p50_latency_ms: f64,
    p90_latency_ms: f64,
    avg_rouge_l: f64,
    output_dir: String,
    csv_path: String,
    summary_json_path: String,
    plot_path: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = BaselineArgs::parse();
    let output_dir = prepare_output_dir(args.output_dir)?;

    let cli_args = CliArgs {
        hardware: niodoo_real_integrated::config::HardwareProfile::Beelink,
        prompt: None,
        prompt_file: None,
        swarm: 1,
        iterations: 1,
        output: OutputFormat::Csv,
        config: None,
        rng_seed_override: None,
    };

    let config = RuntimeConfig::load(&cli_args)?;

    let env_directives = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    let env_filter = tracing_subscriber::EnvFilter::try_new(env_directives)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let fmt_layer = tracing_subscriber::fmt::layer().with_target(false);
    let subscriber = tracing_subscriber::registry::Registry::default()
        .with(env_filter)
        .with(fmt_layer);
    let _ = subscriber.try_init();

    let mut engine = GenerationEngine::new_with_config(
        &config.vllm_endpoint,
        &config.vllm_model,
        config.generation_timeout_secs,
        config.generation_max_tokens,
        config.dynamic_token_min,
        config.dynamic_token_max,
        config.prompt_max_chars,
        config.consistency_variance_threshold,
    )?;
    engine.apply_runtime_from_config(&config);
    engine.set_mock_mode(config.mock_mode);

    let prompts = generate_raw_rut_prompts();
    let mut results = Vec::new();
    let mut latencies = Vec::new();
    let mut rouges = Vec::new();

    for (cycle, prompt) in prompts.iter().enumerate() {
        let start_time = Instant::now();
        let generation = engine
            .generate_with_params(prompt, config.temperature, config.top_p)
            .await
            .with_context(|| format!("generation failed for cycle {}", cycle + 1))?;

        let latency = generation
            .latency_ms
            .max(start_time.elapsed().as_millis() as f64);
        let rouge = rouge_l(&generation.hybrid_response, prompt);

        results.push(BaselineResult {
            cycle: cycle + 1,
            prompt: prompt.clone(),
            response: generation.hybrid_response.clone(),
            latency_ms: latency,
            rouge_l: rouge,
        });

        latencies.push(latency);
        rouges.push(rouge);
        println!(
            "Cycle {}/{}: latency={:.0}ms, ROUGE-L={:.3}",
            cycle + 1,
            prompts.len(),
            latency,
            rouge
        );
    }

    let csv_path = output_dir.join("rut_gauntlet_baseline_results.csv");
    export_csv(&results, &csv_path)?;

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_latency = latencies.iter().copied().sum::<f64>() / latencies.len() as f64;
    let p50_latency = percentile(&latencies, 0.5);
    let p90_latency = percentile(&latencies, 0.9);
    let avg_rouge = rouges.iter().copied().sum::<f64>() / rouges.len() as f64;

    let summary_path = output_dir.join("rut_gauntlet_baseline_summary.json");
    let summary = BaselineSummary {
        total_prompts: results.len(),
        avg_latency_ms: avg_latency,
        p50_latency_ms: p50_latency,
        p90_latency_ms: p90_latency,
        avg_rouge_l: avg_rouge,
        output_dir: output_dir.display().to_string(),
        csv_path: csv_path.display().to_string(),
        summary_json_path: summary_path.display().to_string(),
        plot_path: output_dir
            .join("latency_over_cycles.png")
            .display()
            .to_string(),
    };
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;

    plot_latency(&results, &output_dir.join("latency_over_cycles.png"))?;

    println!("Baseline results exported to {}", output_dir.display());

    Ok(())
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

fn prepare_output_dir(dir: Option<PathBuf>) -> Result<PathBuf> {
    let dir = dir.unwrap_or_else(|| {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        PathBuf::from(format!("logs/rut_gauntlet_baseline/{timestamp}"))
    });
    if !dir.exists() {
        create_dir_all(&dir)?;
    }
    Ok(dir)
}

fn export_csv(results: &[BaselineResult], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = csv::Writer::from_writer(BufWriter::new(file));
    for result in results {
        writer.serialize(result)?;
    }
    writer.flush()?;
    Ok(())
}

fn percentile(values: &[f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() - 1) as f64 * fraction).round() as usize;
    values[index]
}

fn plot_latency(results: &[BaselineResult], path: &Path) -> Result<()> {
    let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    let latencies: Vec<f64> = results.iter().map(|r| r.latency_ms).collect();
    let max_latency = latencies.iter().cloned().fold(0.0_f64, f64::max).max(1.0);
    let x_end = results.len() as i32 + 1;
    let mut chart = ChartBuilder::on(&root)
        .caption("Baseline Latency Over Cycles", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(1i32..x_end, 0f64..(max_latency * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("Cycle")
        .y_desc("Latency (ms)")
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            results.iter().map(|r| (r.cycle as i32, r.latency_ms)),
            &BLUE,
        ))?
        .label("Latency")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
