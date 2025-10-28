use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use csv::WriterBuilder;
use futures::future::{join_all, BoxFuture};
use rayon::prelude::*;
use serde::Serialize;
use tokio::sync::Mutex as AsyncMutex;
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
    raw_stds: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tcs_core::metrics::init_metrics();

    let args = CliArgs::parse();
    if let Some(seed) = args.rng_seed_override {
        std::env::set_var("RNG_SEED", seed.to_string());
    }
    // Load env files early so RuntimeConfig can pick up `.env.production` or `.env`
    niodoo_real_integrated::config::prime_environment();
    niodoo_real_integrated::config::init();
    init_tracing();

    // Seed RNG from env in main and pass into pipeline
    let seed = std::env::var("RNG_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);
    let mut pipeline = Pipeline::initialise_with_seed(args.clone(), seed).await?;
    let prompts = collect_prompts(&args, &pipeline)?;
    let prompt_count = prompts.len();

    info!(
        count = prompt_count,
        "processing prompts through NIODOO pipeline"
    );

    let cycles = if args.swarm > 1 {
        let mut shared = Vec::with_capacity(args.swarm);
        shared.push(Arc::new(AsyncMutex::new(pipeline)));
        for offset in 1..args.swarm {
            let extra_seed = seed.wrapping_add(offset as u64);
            let extra = Pipeline::initialise_with_seed(args.clone(), extra_seed).await?;
            shared.push(Arc::new(AsyncMutex::new(extra)));
        }
        let pipelines = Arc::new(shared);
        let pipelines_len = pipelines.len();

        let tasks: Vec<BoxFuture<'static, anyhow::Result<(usize, PipelineCycle)>>> = prompts
            .into_par_iter()
            .enumerate()
            .map(|(idx, prompt)| {
                let pipelines = pipelines.clone();
                Box::pin(async move {
                    let slot = idx % pipelines_len;
                    let pipeline = pipelines[slot].clone();
                    let mut guard = pipeline.lock().await;
                    let cycle = guard.process_prompt(&prompt.text).await?;
                    Ok::<(usize, PipelineCycle), anyhow::Error>((idx, cycle))
                }) as BoxFuture<'static, anyhow::Result<(usize, PipelineCycle)>>
            })
            .collect();

        let mut indexed = Vec::with_capacity(prompt_count);
        for result in join_all(tasks).await {
            let (idx, cycle) = result?;
            indexed.push((idx, cycle));
        }
        indexed.sort_by_key(|(idx, _)| *idx);
        indexed.into_iter().map(|(_, cycle)| cycle).collect()
    } else {
        let mut sequential_cycles = Vec::with_capacity(prompt_count);
        for prompt in prompts {
            let cycle = pipeline.process_prompt(&prompt.text).await?;
            sequential_cycles.push(cycle);
        }
        sequential_cycles
    };

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
    let base_prompts = if let Some(ref prompt) = args.prompt {
        vec![niodoo_real_integrated::data::RutPrompt {
            index: 0,
            category: niodoo_real_integrated::data::RutCategory::Wildcard,
            text: prompt.clone(),
        }]
    } else if let Some(ref path) = args.prompt_file {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut prompts = Vec::new();
        for line in reader.lines() {
            let text = line?;
            if text.trim().is_empty() {
                continue;
            }
            prompts.push(niodoo_real_integrated::data::RutPrompt {
                index: 0,
                category: niodoo_real_integrated::data::RutCategory::Wildcard,
                text,
            });
        }
        prompts
    } else {
        pipeline.rut_prompts()
    };

    let iterations = args.iterations.max(1);
    let mut expanded = Vec::with_capacity(base_prompts.len() * iterations);
    for prompt in base_prompts {
        for _ in 0..iterations {
            expanded.push(niodoo_real_integrated::data::RutPrompt {
                index: 0,
                category: prompt.category,
                text: prompt.text.clone(),
            });
        }
    }

    for (idx, prompt) in expanded.iter_mut().enumerate() {
        prompt.index = idx + 1;
    }

    Ok(expanded)
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
            raw_stds: format!("{:?}", cycle.pad_state.raw_stds),
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
            raw_stds: format!("{:?}", cycle.pad_state.raw_stds),
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
