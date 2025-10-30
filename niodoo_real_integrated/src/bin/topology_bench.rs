use std::fs::{self, File};
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use blake3::hash as blake3_hash;
use chrono::Utc;
use clap::Parser;
use csv::ReaderBuilder;
use niodoo_real_integrated::config::{CliArgs, HardwareProfile, TopologyMode};
use niodoo_real_integrated::pipeline::Pipeline;
use niodoo_real_integrated::util::cosine_similarity;
use serde::Serialize;
use tracing::{info, warn};

#[derive(Debug, Parser)]
#[command(
    name = "topology_bench",
    about = "Benchmark topology-enabled inference against baseline mode."
)]
struct Args {
    /// Path to GoEmotions TSV dataset.
    #[arg(long, default_value = "data/goemotions_test.tsv")]
    dataset: String,

    /// Number of cycles to execute in each mode.
    #[arg(long, default_value_t = 50)]
    cycles: usize,

    /// Optional explicit runtime config file for the pipeline.
    #[arg(long)]
    config: Option<String>,

    /// Hardware profile used for pipeline initialization.
    #[arg(long)]
    hardware: Option<HardwareProfile>,

    /// Directory to store benchmark outputs.
    #[arg(long, default_value = "results/benchmarks/topology")]
    output_dir: String,
}

#[derive(Debug, Serialize)]
struct CycleRecord {
    cycle: usize,
    prompt: String,
    label: String,
    latency_baseline_ms: f64,
    latency_hybrid_ms: f64,
    entropy_baseline: f64,
    entropy_hybrid: f64,
    rouge_baseline: f64,
    rouge_hybrid: f64,
    pad_similarity_baseline: f64,
    pad_similarity_hybrid: f64,
    betti_0_baseline: usize,
    betti_1_baseline: usize,
    betti_2_baseline: usize,
    betti_0_hybrid: usize,
    betti_1_hybrid: usize,
    betti_2_hybrid: usize,
    knot_complexity_baseline: f64,
    knot_complexity_hybrid: f64,
    persistence_entropy_baseline: f64,
    persistence_entropy_hybrid: f64,
    spectral_gap_baseline: f64,
    spectral_gap_hybrid: f64,
    confidence_baseline: f64,
    confidence_hybrid: f64,
    baseline_preview: String,
    hybrid_preview: String,
    baseline_hash: String,
    hybrid_hash: String,
}

#[derive(Debug, Serialize, Default)]
struct SummaryMetrics {
    avg_latency_baseline_ms: f64,
    avg_latency_hybrid_ms: f64,
    avg_entropy_baseline: f64,
    avg_entropy_hybrid: f64,
    avg_rouge_baseline: f64,
    avg_rouge_hybrid: f64,
    avg_pad_similarity_baseline: f64,
    avg_pad_similarity_hybrid: f64,
    avg_knot_complexity_baseline: f64,
    avg_knot_complexity_hybrid: f64,
    avg_persistence_entropy_baseline: f64,
    avg_persistence_entropy_hybrid: f64,
    avg_spectral_gap_baseline: f64,
    avg_spectral_gap_hybrid: f64,
}

impl SummaryMetrics {
    fn from_records(records: &[CycleRecord]) -> Self {
        let total = records.len() as f64;
        if total == 0.0 {
            return Self::default();
        }

        let mut summary = Self::default();
        for record in records {
            summary.avg_latency_baseline_ms += record.latency_baseline_ms;
            summary.avg_latency_hybrid_ms += record.latency_hybrid_ms;
            summary.avg_entropy_baseline += record.entropy_baseline;
            summary.avg_entropy_hybrid += record.entropy_hybrid;
            summary.avg_rouge_baseline += record.rouge_baseline;
            summary.avg_rouge_hybrid += record.rouge_hybrid;
            summary.avg_pad_similarity_baseline += record.pad_similarity_baseline;
            summary.avg_pad_similarity_hybrid += record.pad_similarity_hybrid;
            summary.avg_knot_complexity_baseline += record.knot_complexity_baseline;
            summary.avg_knot_complexity_hybrid += record.knot_complexity_hybrid;
            summary.avg_persistence_entropy_baseline += record.persistence_entropy_baseline;
            summary.avg_persistence_entropy_hybrid += record.persistence_entropy_hybrid;
            summary.avg_spectral_gap_baseline += record.spectral_gap_baseline;
            summary.avg_spectral_gap_hybrid += record.spectral_gap_hybrid;
        }

        summary.avg_latency_baseline_ms /= total;
        summary.avg_latency_hybrid_ms /= total;
        summary.avg_entropy_baseline /= total;
        summary.avg_entropy_hybrid /= total;
        summary.avg_rouge_baseline /= total;
        summary.avg_rouge_hybrid /= total;
        summary.avg_pad_similarity_baseline /= total;
        summary.avg_pad_similarity_hybrid /= total;
        summary.avg_knot_complexity_baseline /= total;
        summary.avg_knot_complexity_hybrid /= total;
        summary.avg_persistence_entropy_baseline /= total;
        summary.avg_persistence_entropy_hybrid /= total;
        summary.avg_spectral_gap_baseline /= total;
        summary.avg_spectral_gap_hybrid /= total;

        summary
    }
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    generated_at: String,
    dataset_path: String,
    total_cycles: usize,
    hardware: String,
    summary: SummaryMetrics,
    records: Vec<CycleRecord>,
}

#[derive(Debug)]
struct EmotionSample {
    text: String,
    label: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let samples = load_dataset(&args.dataset)?;
    if samples.is_empty() {
        anyhow::bail!("dataset {} is empty", args.dataset);
    }

    let mut cli_args = CliArgs::default();
    cli_args.config = args.config.clone();
    if let Some(hw) = args.hardware {
        cli_args.hardware = hw;
    }

    let baseline_args = cli_args.clone();
    let hybrid_args = cli_args.clone();

    info!(
        dataset = %args.dataset,
        cycles = args.cycles,
        hardware = %cli_args.hardware,
        "Starting topology benchmark"
    );

    let mut baseline_pipeline =
        Pipeline::initialise_with_mode(baseline_args, TopologyMode::Baseline).await?;
    let mut hybrid_pipeline =
        Pipeline::initialise_with_mode(hybrid_args, TopologyMode::Hybrid).await?;

    let mut records = Vec::with_capacity(args.cycles);
    for cycle_idx in 0..args.cycles {
        let sample = &samples[cycle_idx % samples.len()];

        info!(cycle = cycle_idx, "Processing prompt");

        let baseline_cycle = baseline_pipeline
            .process_prompt(&sample.text)
            .await
            .with_context(|| format!("baseline pipeline failed on cycle {}", cycle_idx))?;
        let hybrid_cycle = hybrid_pipeline
            .process_prompt(&sample.text)
            .await
            .with_context(|| format!("hybrid pipeline failed on cycle {}", cycle_idx))?;

        let baseline_text = baseline_cycle.hybrid_response.trim();
        let hybrid_text = hybrid_cycle.hybrid_response.trim();

        ensure!(
            !baseline_text.is_empty(),
            "baseline response empty for cycle {} (prompt: {})",
            cycle_idx,
            sample.text
        );
        ensure!(
            !hybrid_text.is_empty(),
            "hybrid response empty for cycle {} (prompt: {})",
            cycle_idx,
            sample.text
        );
        ensure!(
            baseline_text != hybrid_text,
            "baseline and hybrid responses identical for cycle {} (prompt: {})",
            cycle_idx,
            sample.text
        );
        ensure!(
            baseline_cycle.generation.source != "mock",
            "baseline generation used mock fallback for cycle {} (prompt: {})",
            cycle_idx,
            sample.text
        );
        ensure!(
            hybrid_cycle.generation.source != "mock",
            "hybrid generation used mock fallback for cycle {} (prompt: {})",
            cycle_idx,
            sample.text
        );
        ensure!(
            !baseline_text.starts_with("Mock response:"),
            "mock placeholder detected in baseline response for cycle {} (prompt: {})",
            cycle_idx,
            sample.text
        );
        ensure!(
            !hybrid_text.starts_with("Mock response:"),
            "mock placeholder detected in hybrid response for cycle {} (prompt: {})",
            cycle_idx,
            sample.text
        );
        ensure!(
            baseline_text != "Generation failed",
            "generation failure placeholder detected in baseline response for cycle {} (prompt: {})",
            cycle_idx,
            sample.text
        );
        ensure!(
            hybrid_text != "Generation failed",
            "generation failure placeholder detected in hybrid response for cycle {} (prompt: {})",
            cycle_idx,
            sample.text
        );

        let baseline_hash = response_hash(baseline_text);
        let hybrid_hash = response_hash(hybrid_text);
        let baseline_preview = response_preview(baseline_text);
        let hybrid_preview = response_preview(hybrid_text);

        let ground_truth = emotion_to_pad(&sample.label);
        let pad_similarity_baseline =
            cosine_similarity_f64(&baseline_cycle.pad_state, &ground_truth);
        let pad_similarity_hybrid = cosine_similarity_f64(&hybrid_cycle.pad_state, &ground_truth);

        let confidence_baseline = pipeline_confidence(baseline_cycle.entropy);
        let confidence_hybrid = pipeline_confidence(hybrid_cycle.entropy);

        records.push(CycleRecord {
            cycle: cycle_idx,
            prompt: sample.text.clone(),
            label: sample.label.clone(),
            latency_baseline_ms: baseline_cycle.latency_ms,
            latency_hybrid_ms: hybrid_cycle.latency_ms,
            entropy_baseline: baseline_cycle.entropy,
            entropy_hybrid: hybrid_cycle.entropy,
            rouge_baseline: baseline_cycle.rouge,
            rouge_hybrid: hybrid_cycle.rouge,
            pad_similarity_baseline,
            pad_similarity_hybrid,
            betti_0_baseline: baseline_cycle.topology.betti_numbers[0],
            betti_1_baseline: baseline_cycle.topology.betti_numbers[1],
            betti_2_baseline: baseline_cycle.topology.betti_numbers[2],
            betti_0_hybrid: hybrid_cycle.topology.betti_numbers[0],
            betti_1_hybrid: hybrid_cycle.topology.betti_numbers[1],
            betti_2_hybrid: hybrid_cycle.topology.betti_numbers[2],
            knot_complexity_baseline: baseline_cycle.topology.knot_complexity,
            knot_complexity_hybrid: hybrid_cycle.topology.knot_complexity,
            persistence_entropy_baseline: baseline_cycle.topology.persistence_entropy,
            persistence_entropy_hybrid: hybrid_cycle.topology.persistence_entropy,
            spectral_gap_baseline: baseline_cycle.topology.spectral_gap,
            spectral_gap_hybrid: hybrid_cycle.topology.spectral_gap,
            confidence_baseline,
            confidence_hybrid,
            baseline_preview,
            hybrid_preview,
            baseline_hash,
            hybrid_hash,
        });
    }

    let summary = SummaryMetrics::from_records(&records);
    let report = BenchmarkReport {
        generated_at: Utc::now().to_rfc3339(),
        dataset_path: args.dataset.clone(),
        total_cycles: records.len(),
        hardware: cli_args.hardware.to_string(),
        summary,
        records,
    };

    persist_report(&args.output_dir, &report)?;

    info!("Benchmark complete");
    Ok(())
}

fn load_dataset(path: &str) -> Result<Vec<EmotionSample>> {
    let file = File::open(path).with_context(|| format!("unable to open dataset at {path}"))?;
    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_reader(file);

    let mut samples = Vec::new();
    for (index, record) in reader.records().enumerate() {
        let record = record.with_context(|| format!("failed to read TSV record {index}"))?;
        if record.len() < 2 {
            warn!(row = index, "Skipping malformed TSV row");
            continue;
        }
        let text = record.get(0).unwrap_or_default().trim();
        let label = record.get(1).unwrap_or("neutral").trim();
        if text.is_empty() {
            continue;
        }
        samples.push(EmotionSample {
            text: text.to_string(),
            label: label.to_string(),
        });
    }

    Ok(samples)
}

fn emotion_to_pad(emotion: &str) -> [f64; 7] {
    match emotion.to_ascii_lowercase().as_str() {
        "joy" | "excitement" => [0.8, 0.9, 0.7, 0.2, -0.1, 0.1, -0.05],
        "sadness" | "disappointment" => [-0.7, -0.3, -0.4, -0.2, -0.15, -0.1, -0.05],
        "anger" | "annoyance" => [-0.8, 0.9, 0.6, 0.3, -0.05, 0.1, -0.1],
        "fear" | "nervousness" => [-0.6, 0.8, -0.7, -0.25, 0.05, -0.1, -0.15],
        "surprise" => [0.3, 0.9, 0.0, 0.1, 0.0, 0.05, -0.05],
        "love" => [0.9, 0.6, 0.8, 0.3, 0.1, 0.05, 0.0],
        "gratitude" => [0.7, 0.3, 0.5, 0.15, 0.05, 0.1, 0.0],
        "pride" => [0.8, 0.7, 0.9, 0.25, 0.05, 0.15, 0.05],
        "neutral" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        other => {
            warn!(
                label = other,
                "Unknown emotion label; defaulting to neutral PAD"
            );
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    }
}

fn cosine_similarity_f64(
    state: &niodoo_real_integrated::torus::PadGhostState,
    truth: &[f64; 7],
) -> f64 {
    let actual: Vec<f32> = state.pad.iter().map(|value| *value as f32).collect();
    let expected: Vec<f32> = truth.iter().map(|value| *value as f32).collect();
    cosine_similarity(&actual, &expected) as f64
}

fn pipeline_confidence(entropy: f64) -> f64 {
    (1.0 - (entropy / 3.0)).clamp(0.0, 1.0)
}

fn response_hash(text: &str) -> String {
    blake3_hash(text.as_bytes()).to_hex().to_string()
}

fn response_preview(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    const MAX_CHARS: usize = 160;
    let mut preview = String::new();
    let mut count = 0usize;
    for ch in trimmed.chars() {
        if count >= MAX_CHARS {
            preview.push('…');
            break;
        }
        preview.push(ch);
        count += 1;
    }
    preview
}

fn persist_report(output_dir: &str, report: &BenchmarkReport) -> Result<()> {
    let dir = PathBuf::from(output_dir);
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create output dir {output_dir}"))?;

    let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
    let json_path = dir.join(format!("topology_benchmark_{timestamp}.json"));
    let csv_path = dir.join(format!("topology_benchmark_{timestamp}.csv"));

    let json_file = File::create(&json_path)
        .with_context(|| format!("failed to create report file {}", json_path.display()))?;
    serde_json::to_writer_pretty(json_file, report)?;

    let mut csv_writer = csv::Writer::from_path(&csv_path)
        .with_context(|| format!("failed to create CSV file {}", csv_path.display()))?;
    for record in &report.records {
        csv_writer.serialize(record)?;
    }
    csv_writer.flush()?;

    info!(json = %json_path.display(), csv = %csv_path.display(), "Benchmark artifacts written");
    Ok(())
}
