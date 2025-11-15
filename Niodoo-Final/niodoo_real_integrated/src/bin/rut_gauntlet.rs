use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use niodoo_real_integrated::data::{
    load_rut_gauntlet_prompts, sample_prompts, RutCategory, RutPrompt,
};
use niodoo_real_integrated::util::shannon_entropy;
use serde::Serialize;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(
    name = "rut_gauntlet",
    author = "Niodoo Torque Team",
    version,
    about = "Explore and analyse the 100 prompt Rut Gauntlet dataset."
)]
struct Cli {
    /// Optional category filter (defaults to all categories)
    #[arg(long, value_enum)]
    category: Option<CategoryArg>,

    /// Sample N prompts (with optional category) and write them to stdout
    #[arg(long)]
    sample: Option<usize>,

    /// Render the full prompt list as JSON to the given file
    #[arg(long)]
    output: Option<PathBuf>,

    /// Emit dataset level statistics (entropy, variance, coherence estimates)
    #[arg(long)]
    stats: bool,

    /// Include the raw prompt text when printing stats
    #[arg(long)]
    verbose: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CategoryArg {
    Frustration,
    Grind,
    Breakthrough,
    Flow,
    Wildcard,
}

impl From<CategoryArg> for RutCategory {
    fn from(value: CategoryArg) -> RutCategory {
        match value {
            CategoryArg::Frustration => RutCategory::Frustration,
            CategoryArg::Grind => RutCategory::Grind,
            CategoryArg::Breakthrough => RutCategory::Breakthrough,
            CategoryArg::Flow => RutCategory::Flow,
            CategoryArg::Wildcard => RutCategory::Wildcard,
        }
    }
}

#[derive(Debug, Serialize)]
struct PromptRecord<'a> {
    index: usize,
    category: &'a str,
    text: &'a str,
    entropy: f64,
    char_variance: f64,
    length: usize,
}

#[derive(Debug, Serialize)]
struct DatasetReport {
    prompt_count: usize,
    categories: Vec<CategorySummary>,
    overall: AggregateStats,
}

#[derive(Debug, Serialize)]
struct CategorySummary {
    name: String,
    count: usize,
    entropy_mean: f64,
    variance_mean: f64,
    average_len: f64,
}

#[derive(Debug, Serialize)]
struct AggregateStats {
    entropy_mean: f64,
    entropy_std: f64,
    variance_mean: f64,
    variance_std: f64,
    avg_length: f64,
    length_std: f64,
}

#[cfg(feature = "cli_bins")]
fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let mut prompts = load_rut_gauntlet_prompts();

    if let Some(category) = cli.category {
        let target = category.into();
        prompts.retain(|p| p.category == target);
        if prompts.is_empty() {
            warn!("No prompts matched the requested category; exiting.");
            return Ok(());
        }
    }

    if let Some(count) = cli.sample {
        emit_samples(&prompts, count, cli.verbose)?;
    }

    if cli.stats {
        emit_stats(&prompts, cli.verbose)?;
    }

    if let Some(path) = cli.output.as_ref() {
        write_json(&prompts, path)?;
        info!("Wrote {} prompts to {}", prompts.len(), path.display());
    }

    if cli.sample.is_none() && !cli.stats && cli.output.is_none() {
        emit_samples(&prompts, 5, cli.verbose)?;
    }

    Ok(())
}

#[cfg(not(feature = "cli_bins"))]
fn main() {
    eprintln!("Enable the `cli_bins` feature to run `rut_gauntlet`.");
}

fn emit_samples(prompts: &[RutPrompt], count: usize, verbose: bool) -> Result<()> {
    let chosen = sample_prompts(prompts, count.min(prompts.len()));
    info!("--- Rut Gauntlet sample ({} prompts) ---", chosen.len());
    for prompt in &chosen {
        println!(
            "[#{:03} {:?}] {}",
            prompt.index, prompt.category, prompt.text
        );
        if verbose {
            let (entropy, variance) = prompt_metrics(prompt);
            println!("    entropy={entropy:.4} variance={variance:.4}");
        }
    }
    Ok(())
}

fn emit_stats(prompts: &[RutPrompt], verbose: bool) -> Result<()> {
    let start = Instant::now();
    let per_category: Vec<PromptRecord<'_>> = prompts
        .iter()
        .map(|prompt| {
            let (entropy, variance) = prompt_metrics(prompt);
            PromptRecord {
                index: prompt.index,
                category: category_name(prompt.category),
                text: &prompt.text,
                entropy,
                char_variance: variance,
                length: prompt.text.chars().count(),
            }
        })
        .collect();

    let report = build_dataset_report(&per_category);

    if verbose {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{}", serde_json::to_string(&report)?);
    }

    info!(
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "Computed dataset statistics"
    );
    Ok(())
}

fn write_json(prompts: &[RutPrompt], path: &PathBuf) -> Result<()> {
    let mut file =
        File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let records: Vec<_> = prompts
        .iter()
        .map(|p| {
            let (entropy, variance) = prompt_metrics(p);
            PromptRecord {
                index: p.index,
                category: category_name(p.category),
                text: &p.text,
                entropy,
                char_variance: variance,
                length: p.text.chars().count(),
            }
        })
        .collect();
    let json = serde_json::to_vec_pretty(&records)?;
    file.write_all(&json)?;
    Ok(())
}

fn prompt_metrics(prompt: &RutPrompt) -> (f64, f64) {
    let entropy = {
        let mut histogram = [0usize; 256];
        let mut total = 0usize;
        for byte in prompt.text.bytes() {
            histogram[byte as usize] += 1;
            total += 1;
        }
        let mut probs = Vec::new();
        if total > 0 {
            for &count in &histogram {
                if count > 0 {
                    probs.push(count as f64 / total as f64);
                }
            }
        }
        shannon_entropy(&probs)
    };

    let bytes: Vec<f64> = prompt.text.bytes().map(|b| b as f64).collect();
    let variance = if bytes.is_empty() {
        0.0
    } else {
        let mu = mean_slice(&bytes);
        bytes.iter().map(|value| (value - mu).powi(2)).sum::<f64>() / bytes.len() as f64
    };

    (entropy, variance)
}

fn build_dataset_report(records: &[PromptRecord<'_>]) -> DatasetReport {
    let mut by_category: std::collections::BTreeMap<&str, Vec<&PromptRecord<'_>>> =
        std::collections::BTreeMap::new();
    for record in records {
        by_category.entry(record.category).or_default().push(record);
    }

    let categories = by_category
        .into_iter()
        .map(|(name, prompts)| {
            let count = prompts.len();
            let entropies: Vec<f64> = prompts.iter().map(|p| p.entropy).collect();
            let variances: Vec<f64> = prompts.iter().map(|p| p.char_variance).collect();
            let lengths: Vec<f64> = prompts.iter().map(|p| p.length as f64).collect();
            CategorySummary {
                name: name.to_string(),
                count,
                entropy_mean: mean_slice(&entropies),
                variance_mean: mean_slice(&variances),
                average_len: mean_slice(&lengths),
            }
        })
        .collect();

    let entropies: Vec<f64> = records.iter().map(|r| r.entropy).collect();
    let variances: Vec<f64> = records.iter().map(|r| r.char_variance).collect();
    let lengths: Vec<f64> = records.iter().map(|r| r.length as f64).collect();

    let overall = AggregateStats {
        entropy_mean: mean_slice(&entropies),
        entropy_std: std_dev_slice(&entropies),
        variance_mean: mean_slice(&variances),
        variance_std: std_dev_slice(&variances),
        avg_length: mean_slice(&lengths),
        length_std: std_dev_slice(&lengths),
    };

    DatasetReport {
        prompt_count: records.len(),
        categories,
        overall,
    }
}

fn mean_slice(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().sum();
    sum / values.len() as f64
}

fn std_dev_slice(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = mean_slice(values);
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn category_name(category: RutCategory) -> &'static str {
    match category {
        RutCategory::Frustration => "Frustration",
        RutCategory::Grind => "Grind",
        RutCategory::Breakthrough => "Breakthrough",
        RutCategory::Flow => "Flow",
        RutCategory::Wildcard => "Wildcard",
    }
}
