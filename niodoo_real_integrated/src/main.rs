use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result, bail};
use clap::Parser;
use niodoo_real_integrated::config::{self, env_value, set_env_override, CliArgs};
use niodoo_real_integrated::pipeline::Pipeline;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // Compose subscriber with optional OpenTelemetry layer if enabled and configured
    let fmt_layer = tracing_subscriber::fmt::layer().with_target(false);

    #[cfg(feature = "otel")]
    let registry = {
        let mut base = tracing_subscriber::registry().with(env_filter).with(fmt_layer);
        if let Ok(endpoint) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
            match opentelemetry_otlp::new_pipeline()
                .tracing()
                .with_exporter(opentelemetry_otlp::new_exporter().tonic().with_endpoint(endpoint))
                .install_simple()
            {
                Ok(tracer) => {
                    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
                    base = base.with(otel_layer);
                }
                Err(err) => {
                    warn!(error = %err, "Failed to initialize OpenTelemetry; continuing without it");
                }
            }
        }
        base
    };

    #[cfg(not(feature = "otel"))]
    let registry = tracing_subscriber::registry().with(env_filter).with(fmt_layer);

    registry.init();

    config::prime_environment();

    let args = CliArgs::parse();

    let rng_seed = args
        .rng_seed_override
        .or_else(|| env_value("RNG_SEED").and_then(|value| value.parse::<u64>().ok()))
        .unwrap_or_else(|| {
            let default_seed = 42_u64;
            warn!(
                seed = default_seed,
                "RNG_SEED not specified; using default seed"
            );
            default_seed
        });

    set_env_override("RNG_SEED", rng_seed.to_string());

    let prompts = gather_prompts(&args)?;
    let iterations = args.iterations.max(1);
    let mut expanded_prompts = Vec::with_capacity(prompts.len() * iterations);
    for prompt in prompts {
        for _ in 0..iterations {
            expanded_prompts.push(prompt.clone());
        }
    }

    if expanded_prompts.is_empty() {
        bail!("No prompts provided. Use --prompt or --prompt-file to supply input.");
    }

    if args.swarm > 1 {
        warn!(
            swarm = args.swarm,
            "Swarm concurrency is not yet implemented; running sequentially"
        );
    }

    info!(
        total_prompts = expanded_prompts.len(),
        iterations,
        seed = rng_seed,
        "Starting NIODOO pipeline run"
    );

    let mut pipeline = Pipeline::initialise_with_seed(args.clone(), rng_seed).await?;

    let mut latencies = Vec::with_capacity(expanded_prompts.len());
    let mut rouges = Vec::with_capacity(expanded_prompts.len());
    let mut failure_counts: HashMap<String, usize> = HashMap::new();
    let mut promoted_total = 0usize;
    let mut promoted_max = 0usize;

    for (index, prompt) in expanded_prompts.iter().enumerate() {
        info!(index, prompt = %prompt, "Processing prompt");
        let cycle = pipeline.process_prompt(prompt).await?;

        latencies.push(cycle.latency_ms);
        rouges.push(cycle.rouge);

        if cycle.failure != "none" {
            *failure_counts.entry(cycle.failure.clone()).or_default() += 1;
        }

        let promoted = cycle.tokenizer.promoted_tokens.len();
        promoted_total += promoted;
        promoted_max = promoted_max.max(promoted);

        info!(
            latency_ms = cycle.latency_ms,
            rouge = cycle.rouge,
            promoted_tokens = promoted,
            failure = %cycle.failure,
            "Cycle complete"
        );
    }

    let latency_summary = Summary::from(&mut latencies);
    let rouge_summary = Summary::from(&mut rouges);

    info!("=== Run Summary ===");
    info!(
        count = latency_summary.count,
        avg_latency_ms = latency_summary.mean,
        median_latency_ms = latency_summary.median,
        p95_latency_ms = latency_summary.p95,
        p99_latency_ms = latency_summary.p99,
        min_latency_ms = latency_summary.min,
        max_latency_ms = latency_summary.max,
        "Latency metrics"
    );
    info!(
        avg_rouge = rouge_summary.mean,
        median_rouge = rouge_summary.median,
        p05_rouge = rouge_summary.p05,
        p95_rouge = rouge_summary.p95,
        min_rouge = rouge_summary.min,
        max_rouge = rouge_summary.max,
        "ROUGE metrics"
    );
    info!(promoted_total, promoted_max, "Token promotion metrics");

    if !failure_counts.is_empty() {
        for (failure, count) in failure_counts.iter() {
            info!(failure = %failure, count, "Failure occurrences");
        }
    } else {
        info!("No failure tiers triggered");
    }

    Ok(())
}

fn gather_prompts(args: &CliArgs) -> Result<Vec<String>> {
    let mut prompts = Vec::new();

    if let Some(prompt_file) = &args.prompt_file {
        let file = File::open(prompt_file)
            .with_context(|| format!("unable to open prompt file {prompt_file}"))?;
        for line in BufReader::new(file).lines() {
            let line = line?;
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                prompts.push(trimmed.to_string());
            }
        }
    }

    if let Some(prompt) = &args.prompt {
        if prompt.trim().is_empty() {
            bail!("Provided prompt is empty");
        }
        prompts.push(prompt.clone());
    }

    Ok(prompts)
}

#[derive(Default)]
struct Summary {
    count: usize,
    mean: f64,
    median: f64,
    min: f64,
    max: f64,
    p05: f64,
    p95: f64,
    p99: f64,
}

impl Summary {
    fn from(values: &mut Vec<f64>) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let count = values.len();
        let sum: f64 = values.iter().copied().sum();
        let mean = sum / count as f64;
        let median = percentile(values, 0.5);
        let min = values[0];
        let max = values[count - 1];
        let p05 = percentile(values, 0.05);
        let p95 = percentile(values, 0.95);
        let p99 = percentile(values, 0.99);

        Self {
            count,
            mean,
            median,
            min,
            max,
            p05,
            p95,
            p99,
        }
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }

    let idx = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[idx]
}
