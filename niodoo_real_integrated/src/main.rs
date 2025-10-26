use std::fs::File;
use std::io::{self, BufRead, BufReader};

use anyhow::Result;
use clap::Parser;
use csv::WriterBuilder;
use serde::Serialize;
use tracing::info;
use tracing_subscriber::{
    filter::FilterFn,
    fmt,
    layer::{Layer, SubscriberExt},
    registry::Registry,
    util::SubscriberInitExt,
    EnvFilter,
};

use niodoo_real_integrated::config::{CliArgs, OutputFormat};
use niodoo_real_integrated::metrics::metrics;
use niodoo_real_integrated::pipeline::{Pipeline, PipelineCycle};

#[derive(Serialize)]
struct CsvRecord {
    prompt: String,
    baseline: String,
    hybrid: String,
    entropy: f64,
    rouge: f64,
    latency_ms: f64,
    compass: String,
    threat: bool,
    healing: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tcs_core::metrics::init_metrics();

    let args = CliArgs::parse();
    // Load env files early so RuntimeConfig can pick up `.env.production` or `.env`
    niodoo_real_integrated::config::prime_environment();
    init_tracing();

    let mut pipeline = Pipeline::initialise(args.clone()).await?;
    let prompts = collect_prompts(&args, &pipeline)?;

    info!(
        count = prompts.len(),
        "processing prompts through NIODOO pipeline"
    );

    let mut cycles = Vec::new();
    for prompt in prompts {
        let cycle = pipeline.process_prompt(&prompt.text).await?;
        cycles.push(cycle);
    }

    match args.output {
        OutputFormat::Csv => emit_csv(&cycles)?,
        OutputFormat::Json => emit_json(&cycles)?,
    }

    emit_summary(&cycles);

    let metrics_dump = metrics().gather()?;
    println!("\n# Prometheus Metrics\n{}", metrics_dump);

    Ok(())
}

fn init_tracing() {
    let env_directives = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    let filter_spec = if env_directives
        .split(',')
        .any(|part| part.trim_start().starts_with("ort"))
    {
        env_directives
    } else {
        format!("{env_directives},ort=error")
    };

    let env_filter = EnvFilter::try_new(filter_spec).unwrap_or_else(|_| EnvFilter::new("info"));

    let fmt_layer = fmt::layer()
        .with_target(false)
        .with_filter(FilterFn::new(|metadata| {
            !metadata.target().starts_with("ort")
        }));

    let subscriber = Registry::default().with(env_filter).with(fmt_layer);
    let _ = subscriber.try_init();
}

fn collect_prompts(
    args: &CliArgs,
    pipeline: &Pipeline,
) -> Result<Vec<niodoo_real_integrated::data::RutPrompt>> {
    if let Some(ref prompt) = args.prompt {
        return Ok(vec![niodoo_real_integrated::data::RutPrompt {
            index: 0,
            category: niodoo_real_integrated::data::RutCategory::Wildcard,
            text: prompt.clone(),
        }]);
    }

    if let Some(ref path) = args.prompt_file {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut prompts = Vec::new();
        for (idx, line) in reader.lines().enumerate() {
            let text = line?;
            if text.trim().is_empty() {
                continue;
            }
            prompts.push(niodoo_real_integrated::data::RutPrompt {
                index: idx + 1,
                category: niodoo_real_integrated::data::RutCategory::Wildcard,
                text,
            });
        }
        return Ok(prompts);
    }

    Ok(pipeline.rut_prompts())
}

fn emit_csv(cycles: &[PipelineCycle]) -> Result<()> {
    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_writer(io::stdout());
    for cycle in cycles {
        writer.serialize(CsvRecord {
            prompt: cycle.prompt.clone(),
            baseline: cycle.baseline_response.clone(),
            hybrid: cycle.hybrid_response.clone(),
            entropy: cycle.entropy,
            rouge: cycle.rouge,
            latency_ms: cycle.latency_ms,
            compass: format!("{:?}", cycle.compass.quadrant),
            threat: cycle.compass.is_threat,
            healing: cycle.compass.is_healing,
        })?;
    }
    writer.flush()?;
    Ok(())
}

fn emit_json(cycles: &[PipelineCycle]) -> Result<()> {
    let records: Vec<CsvRecord> = cycles
        .iter()
        .map(|cycle| CsvRecord {
            prompt: cycle.prompt.clone(),
            baseline: cycle.baseline_response.clone(),
            hybrid: cycle.hybrid_response.clone(),
            entropy: cycle.entropy,
            rouge: cycle.rouge,
            latency_ms: cycle.latency_ms,
            compass: format!("{:?}", cycle.compass.quadrant),
            threat: cycle.compass.is_threat,
            healing: cycle.compass.is_healing,
        })
        .collect();
    let json = serde_json::to_string_pretty(&records)?;
    println!("{}", json);
    Ok(())
}

fn emit_summary(cycles: &[PipelineCycle]) {
    if cycles.is_empty() {
        return;
    }

    let entropy_avg = cycles.iter().map(|c| c.entropy).sum::<f64>() / cycles.len() as f64;
    let entropy_std = (cycles
        .iter()
        .map(|c| (c.entropy - entropy_avg).powi(2))
        .sum::<f64>()
        / cycles.len() as f64)
        .sqrt();
    let rouge_avg = cycles.iter().map(|c| c.rouge).sum::<f64>() / cycles.len() as f64;
    let latency_avg = cycles.iter().map(|c| c.latency_ms).sum::<f64>() / cycles.len() as f64;
    let threats = cycles.iter().filter(|c| c.compass.is_threat).count();
    let healings = cycles.iter().filter(|c| c.compass.is_healing).count();

    eprintln!("=== NIODOO Pipeline Summary ===");
    eprintln!("Prompts processed: {}", cycles.len());
    eprintln!("Entropy avg/std: {:.3} / {:.3}", entropy_avg, entropy_std);
    eprintln!(
        "Threat cycles: {} ({}%)",
        threats,
        threats as f64 / cycles.len() as f64 * 100.0
    );
    eprintln!(
        "Healing cycles: {} ({}%)",
        healings,
        healings as f64 / cycles.len() as f64 * 100.0
    );
    eprintln!("Latency avg: {:.2} ms", latency_avg);
    eprintln!("ROUGE-L avg: {:.3}", rouge_avg);

    eprintln!("\nSample outputs:");
    for cycle in cycles.iter().take(3) {
        eprintln!("- Prompt: {}", cycle.prompt);
        eprintln!("  Baseline: {}", cycle.baseline_response);
        eprintln!("  Hybrid: {}", cycle.hybrid_response);
    }
}
