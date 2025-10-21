use std::env;
use std::fmt;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

/// CLI arguments for the integrated pipeline binary.
///
/// The binary can operate on a single prompt or over a full rut-gauntlet batch.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "niodoo_real_integrated",
    version,
    about = "Real NIODOO torque pipeline"
)]
pub struct CliArgs {
    /// Single prompt to process through the pipeline.
    #[arg(short, long)]
    pub prompt: Option<String>,

    /// Optional path to a newline-delimited prompt list (rut gauntlet).
    #[arg(long)]
    pub prompt_file: Option<String>,

    /// Number of swarm instances to process prompts in parallel.
    #[arg(short, long, default_value_t = 1)]
    pub swarm: usize,

    /// Output format for results: csv or json.
    #[arg(short, long, default_value = "csv")]
    pub output: OutputFormat,

    /// Hardware profile used to tune batching/latency assumptions.
    #[arg(long = "hardware", default_value_t = HardwareProfile::Beelink)]
    pub hardware: HardwareProfile,

    /// Optional explicit config file (YAML) overriding env defaults.
    #[arg(long)]
    pub config: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
pub enum OutputFormat {
    #[serde(rename = "csv")]
    Csv,
    #[serde(rename = "json")]
    Json,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
pub enum HardwareProfile {
    #[serde(rename = "beelink")]
    Beelink,
    #[serde(rename = "5080q")]
    #[value(alias = "5080-q")]
    Laptop5080Q,
}

impl fmt::Display for HardwareProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            HardwareProfile::Beelink => "beelink",
            HardwareProfile::Laptop5080Q => "5080q",
        };
        f.write_str(label)
    }
}

impl HardwareProfile {
    pub fn batch_size(self) -> usize {
        match self {
            HardwareProfile::Beelink => 8,
            HardwareProfile::Laptop5080Q => 4,
        }
    }

    pub fn latency_budget_ms(self) -> f64 {
        match self {
            HardwareProfile::Beelink => 100.0,
            HardwareProfile::Laptop5080Q => 180.0,
        }
    }

    pub fn max_kv_cache_tokens(self) -> usize {
        match self {
            HardwareProfile::Beelink => 128_000,
            HardwareProfile::Laptop5080Q => 256_000,
        }
    }
}

/// Runtime configuration resolved from CLI arguments, environment variables,
/// and optional YAML configuration file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub vllm_endpoint: String,
    pub vllm_model: String,
    pub qdrant_url: String,
    pub qdrant_collection: String,
    pub qdrant_vector_dim: usize,
    pub ollama_endpoint: String,
    pub training_data_path: String,
    pub emotional_seed_path: String,
    pub rut_gauntlet_path: Option<String>,
    pub entropy_cycles_for_baseline: usize,
}

impl RuntimeConfig {
    pub fn load(args: &CliArgs) -> Result<Self> {
        if let Some(ref config_path) = args.config {
            let file = std::fs::read_to_string(config_path)
                .with_context(|| format!("unable to read config file {config_path}"))?;
            let cfg: RuntimeConfig = serde_yaml::from_str(&file)
                .with_context(|| format!("invalid YAML in {config_path}"))?;
            return Ok(cfg);
        }

        let vllm_endpoint = env::var("VLLM_ENDPOINT")
            .unwrap_or_else(|_| "http://100.113.10.90:8000/v1/chat/completions".to_string());
        let vllm_model = env::var("VLLM_MODEL")
            .unwrap_or_else(|_| "/home/beelink/models/Qwen2.5-7B-Instruct-AWQ".to_string());
        let qdrant_url =
            env::var("QDRANT_URL").unwrap_or_else(|_| "http://100.113.10.90:6333".to_string());
        let qdrant_collection =
            env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "experiences".to_string());
        let qdrant_vector_dim = env::var("QDRANT_VECTOR_DIM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(896);
        let ollama_endpoint =
            env::var("OLLAMA_ENDPOINT").unwrap_or_else(|_| "http://localhost:11434".to_string());
        let training_data_path = env::var("TRAINING_DATA_PATH").unwrap_or_else(|_| {
            "/home/beelink/Niodoo-Final/data/training_data/emotion_training_data.json".to_string()
        });
        let emotional_seed_path = env::var("CONSCIOUSNESS_TRAINING_DATA").unwrap_or_else(|_| {
            "/home/beelink/Niodoo-Final/data/training_data/existing_continual_training_data.json".to_string()
        });
        let rut_gauntlet_path = args
            .prompt_file
            .clone()
            .or_else(|| env::var("RUT_GAUNTLET_PATH").ok());
        let entropy_cycles_for_baseline = env::var("ENTROPY_BASELINE_CYCLES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);

        Ok(Self {
            vllm_endpoint,
            vllm_model,
            qdrant_url,
            qdrant_collection,
            qdrant_vector_dim,
            ollama_endpoint,
            training_data_path,
            emotional_seed_path,
            rut_gauntlet_path,
            entropy_cycles_for_baseline,
        })
    }
}
