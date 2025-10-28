use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use niodoo_real_integrated::config::{prime_environment, CliArgs, OutputFormat};
use niodoo_real_integrated::metrics::metrics;
use niodoo_real_integrated::pipeline::Pipeline;
use serde::Serialize;
use tokio::runtime::Runtime;
use tracing::info;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Topology vs baseline evaluation harness", long_about = None)]
struct HarnessArgs {
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    #[arg(long, default_value = "baseline_toggle.csv")]
    output: PathBuf,
    /// Subsets to run: list of labels (topology_full, topology_off, betti_only, entropy_only)
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "topology_full,topology_off,betti_only,entropy_only"
    )]
    modes: Vec<String>,
    #[arg(long)]
    skip_promotion: bool,
}

#[derive(Serialize)]
struct HarnessRecord {
    prompt: String,
    baseline: String,
    hybrid: String,
    entropy: f64,
    rouge: f64,
    latency_ms: f64,
    compass: String,
    threat: bool,
    healing: bool,
    mode: String,
}

fn main() -> Result<()> {
    eprintln!("Starting topology baseline harness...");
    prime_environment();
    niodoo_real_integrated::config::init();
    let args = HarnessArgs::parse();
    eprintln!(
        "Parsed args: output={:?}, modes={:?}",
        args.output, args.modes
    );
    let rt = Runtime::new()?;

    let suites = vec![
        ("topology_full", vec![("TOPOLOGY_MODE", "full")]),
        ("topology_off", vec![("TOPOLOGY_MODE", "off")]),
        ("betti_only", vec![("TOPOLOGY_MODE", "betti")]),
        ("entropy_only", vec![("TOPOLOGY_MODE", "entropy")]),
    ];

    let mut writer = csv::Writer::from_path(&args.output)?;

    for (label, env_overrides) in suites {
        if !args.modes.iter().any(|m| m == label) {
            continue;
        }
        eprintln!("Running mode: {}", label);
        info!(mode = label, "Starting topology comparison run");
        for (key, value) in &env_overrides {
            std::env::set_var(key, value);
        }
        if args.skip_promotion {
            std::env::set_var("TOKEN_PROMOTION_DISABLED", "1");
        }
        eprintln!("Calling run_suite for {}", label);
        let mut results = rt.block_on(run_suite(&args, label))?;
        eprintln!("Got {} results for {}", results.len(), label);
        for record in results.drain(..) {
            writer.serialize(record)?;
            writer.flush()?; // Flush after each record to see progress
        }
        std::env::remove_var("TOKEN_PROMOTION_DISABLED");
        for key in env_overrides.iter().map(|(k, _)| k) {
            std::env::remove_var(key);
        }
    }

    writer.flush()?;
    Ok(())
}

async fn run_suite(args: &HarnessArgs, label: &str) -> Result<Vec<HarnessRecord>> {
    let mut cli_args = CliArgs::parse_from(["topology-baseline", "--output", "json"]);
    cli_args.output = OutputFormat::Json;
    if let Some(path) = args.prompt_file.clone() {
        cli_args.prompt_file = Some(path.to_string_lossy().into_owned());
    }
    eprintln!("Initializing pipeline...");
    let mut pipeline = Pipeline::initialise(cli_args.clone()).await?;
    eprintln!("Pipeline initialized, loading prompts...");
    let all_prompts = pipeline.rut_prompts();
    eprintln!("Loaded {} prompts", all_prompts.len());
    // Process all prompts
    let prompts: Vec<_> = all_prompts.iter().collect();
    eprintln!("Processing {} prompts", prompts.len());
    let mut records = Vec::new();
    for (idx, prompt) in prompts.iter().enumerate() {
        eprintln!("Processing prompt {}/{}", idx + 1, prompts.len());
        let cycle = match pipeline.process_prompt(&prompt.text).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error processing prompt {}: {}", idx + 1, e);
                return Err(e);
            }
        };
        records.push(HarnessRecord {
            prompt: cycle.prompt,
            baseline: cycle.baseline_response,
            hybrid: cycle.hybrid_response,
            entropy: cycle.entropy,
            rouge: cycle.rouge,
            latency_ms: cycle.latency_ms,
            compass: format!("{:?}", cycle.compass.quadrant),
            threat: cycle.compass.is_threat,
            healing: cycle.compass.is_healing,
            mode: label.to_string(),
        });
    }
    let scrape = metrics().gather()?;
    std::fs::write(format!("metrics_{label}.prom"), scrape)?;
    Ok(records)
}
