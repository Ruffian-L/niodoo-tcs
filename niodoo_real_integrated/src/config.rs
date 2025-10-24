use std::collections::HashSet;
use std::env;
use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use tracing::warn;

pub fn prime_environment() {
    let mut roots: HashSet<PathBuf> = HashSet::new();

    if let Ok(project_root) = env::var("PROJECT_ROOT") {
        if !project_root.trim().is_empty() {
            roots.insert(PathBuf::from(project_root));
        }
    }

    if let Ok(current) = std::env::current_dir() {
        roots.insert(current);
    }

    roots.insert(PathBuf::from("."));

    let env_files = [".env.production", ".env"];
    let mut seen_paths = HashSet::new();

    for root in roots {
        for file in env_files {
            let path = root.join(file);
            if !path.is_file() {
                continue;
            }
            if !seen_paths.insert(path.clone()) {
                continue;
            }
            if let Err(error) = load_env_file(&path) {
                warn!(path = %path.display(), ?error, "failed to load environment file");
            }
        }
    }
}

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

/// Generation backend type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BackendType {
    #[serde(rename = "vllm_gpu")]
    VllmGpu,
    #[serde(rename = "ollama_cpu")]
    OllamaCpu,
    #[serde(rename = "cascade")]
    Cascade,
}

impl Default for BackendType {
    fn default() -> Self {
        BackendType::VllmGpu
    }
}

impl BackendType {
    pub fn from_env() -> Self {
        std::env::var("GENERATION_BACKEND")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "vllm_gpu" => Some(BackendType::VllmGpu),
                "ollama_cpu" => Some(BackendType::OllamaCpu),
                "cascade" => Some(BackendType::Cascade),
                _ => None,
            })
            .unwrap_or_default()
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
    #[serde(default)]
    pub enable_consistency_voting: bool,

    // Generation backend configuration
    #[serde(default)]
    pub generation_backend: BackendType,

    // Curator configuration
    #[serde(default)]
    pub enable_curator: bool,
    pub curator_model_name: String,
    pub curator_quality_threshold: f32,
    pub curator_minimum_threshold: f32,
    pub curator_timeout_secs: u64,
    pub curator_temperature: f64,
    pub curator_max_tokens: usize,
    pub assessment_prompt_template: String,

    // Generation timeout and token configuration
    pub generation_timeout_secs: u64,
    pub generation_max_tokens: usize,
    pub dynamic_token_min: usize,
    pub dynamic_token_max: usize,
}

impl RuntimeConfig {
    pub fn load(args: &CliArgs) -> Result<Self> {
        prime_environment();

        if let Some(ref config_path) = args.config {
            let file = std::fs::read_to_string(config_path)
                .with_context(|| format!("unable to read config file {config_path}"))?;
            let cfg: RuntimeConfig = serde_yaml::from_str(&file)
                .with_context(|| format!("invalid YAML in {config_path}"))?;
            return Ok(cfg);
        }

        let mut vllm_keys: Vec<&str> = vec!["VLLM_ENDPOINT"];
        if matches!(args.hardware, HardwareProfile::Laptop5080Q) {
            vllm_keys.insert(0, "VLLM_ENDPOINT_TAILSCALE");
        } else {
            vllm_keys.push("VLLM_ENDPOINT_TAILSCALE");
        }
        vllm_keys.push("TEST_ENDPOINT_VLLM");
        let vllm_endpoint = env_with_fallback(&vllm_keys)
            .unwrap_or_else(|| "http://127.0.0.1:8000".to_string())
            .trim()
            .trim_end_matches('/')
            // Strip common API paths if present (curator appends its own)
            .replace("/v1/chat/completions", "")
            .replace("/v1/completions", "")
            .replace("/v1/embeddings", "")
            .trim_end_matches('/')
            .to_string();

        let vllm_model = env_with_fallback(&["VLLM_MODEL", "VLLM_MODEL_ID", "VLLM_MODEL_PATH"])
            .unwrap_or_else(|| {
                "/workspace/models/hf_cache/models--Qwen--Qwen2.5-7B-Instruct-AWQ".to_string()
            });

        let mut qdrant_keys: Vec<&str> = vec!["QDRANT_URL"];
        if matches!(args.hardware, HardwareProfile::Laptop5080Q) {
            qdrant_keys.insert(0, "QDRANT_URL_TAILSCALE");
        } else {
            qdrant_keys.push("QDRANT_URL_TAILSCALE");
        }
        qdrant_keys.push("TEST_ENDPOINT_QDRANT");
        let qdrant_url = env_with_fallback(&qdrant_keys)
            .unwrap_or_else(|| "http://127.0.0.1:6333".to_string())
            .trim()
            .trim_end_matches('/')
            .to_string();

        let qdrant_collection = env_with_fallback(&["QDRANT_COLLECTION", "QDRANT_COLLECTION_NAME"])
            .unwrap_or_else(|| "experiences".to_string());

        let qdrant_vector_dim = env_with_fallback(&["QDRANT_VECTOR_DIM", "QDRANT_VECTOR_SIZE"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(896);

        let ollama_endpoint = env_with_fallback(&["OLLAMA_ENDPOINT", "OLLAMA_ENDPOINT_TAILSCALE"])
            .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());

        let training_data_path = env_with_fallback(&["TRAINING_DATA_PATH"]).unwrap_or_else(|| {
            "/workspace/Niodoo-Final/data/training_data/emotion_training_data.json".to_string()
        });

        let emotional_seed_path = env_with_fallback(&[
            "CONSCIOUSNESS_TRAINING_DATA",
            "EMOTIONAL_SEED_PATH",
        ])
        .unwrap_or_else(|| {
            "/workspace/Niodoo-Final/data/training_data/existing_continual_training_data.json"
                .to_string()
        });

        let rut_gauntlet_path = args
            .prompt_file
            .clone()
            .or_else(|| env_with_fallback(&["RUT_GAUNTLET_PATH", "RUT_PROMPT_FILE"]));

        let entropy_cycles_for_baseline = env_with_fallback(&["ENTROPY_BASELINE_CYCLES"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(20);

        let enable_consistency_voting = env_with_fallback(&["ENABLE_CONSISTENCY_VOTING"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(false);

        let generation_backend = BackendType::from_env();

        let enable_curator = env_with_fallback(&["ENABLE_CURATOR"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(true); // Enabled by default

        let curator_model_name = env_with_fallback(&["CURATOR_MODEL_NAME"])
            .unwrap_or_else(|| "qwen2.5-coder:1.5b".to_string());

        let curator_quality_threshold = env_with_fallback(&["CURATOR_QUALITY_THRESHOLD"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.7);

        let curator_minimum_threshold = env_with_fallback(&["CURATOR_MINIMUM_THRESHOLD"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.5);

        let curator_timeout_secs = env_with_fallback(&["CURATOR_TIMEOUT_SECS"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(30); // Increased from 10 to 30 seconds

        let curator_temperature = env_with_fallback(&["CURATOR_TEMPERATURE"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.7);

        let curator_max_tokens = env_with_fallback(&["CURATOR_MAX_TOKENS"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(256);

        // Generation timeout and token configuration from env
        let generation_timeout_secs = env_with_fallback(&["GENERATION_TIMEOUT_SECS", "TIMEOUT_SECS"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(60); // Default to 60s (reasonable for API calls)

        let generation_max_tokens = env_with_fallback(&["GENERATION_MAX_TOKENS", "MAX_TOKENS"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(2048); // Default to 2048 (sufficient for complex code generation)

        let dynamic_token_min = env_with_fallback(&["DYNAMIC_TOKEN_MIN"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(256); // Default dynamic clamp minimum

        let dynamic_token_max = env_with_fallback(&["DYNAMIC_TOKEN_MAX"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(512); // Default dynamic clamp maximum

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
            enable_consistency_voting,
            generation_backend,
            enable_curator,
            curator_model_name,
            curator_quality_threshold,
            curator_minimum_threshold,
            curator_timeout_secs,
            curator_temperature,
            curator_max_tokens,
            // Enhanced prompt with strict output format
            assessment_prompt_template: "Score this response (0.0-1.0) for emotional breakthrough potential.\nConsider: breakthrough→high score, stagnation→low score, LearningWill advance→boost score.\n\nPrompt: {}\nResponse: {}\nEntropy: {:.3}, Quadrant: {}\n\nOUTPUT FORMAT: Respond with ONLY a single number (e.g., '0.85'). No text, no explanation, no JSON, just the number.:".to_string(),
            generation_timeout_secs,
            generation_max_tokens,
            dynamic_token_min,
            dynamic_token_max,
        })
    }
}

/// Curator configuration derived from runtime config
#[derive(Debug, Clone)]
pub struct CuratorConfig {
    pub vllm_endpoint: String,
    pub model_name: String,
    pub embedding_dim: usize,
    pub max_context_length: usize,
    pub distillation_batch_size: usize,
    pub clustering_threshold: f32,
    pub quality_threshold: f32,
    pub minimum_threshold: f32,
    pub timeout_secs: u64,
    pub temperature: f64,
    pub max_tokens: usize,
    pub assessment_prompt_template: String,
    pub parse_mode: crate::curator_parser::ParserMode,
    // Heuristic parser configuration
    pub heuristic_max_length: usize,
    pub heuristic_optimal_entropy_low: f64,
    pub heuristic_optimal_entropy_high: f64,
    pub heuristic_optimal_entropy_score: f32,
    pub heuristic_suboptimal_entropy_score: f32,
    pub heuristic_length_weight: f32,
}

impl CuratorConfig {
    pub fn from_runtime_config(config: &RuntimeConfig) -> Self {
        Self {
            vllm_endpoint: config.ollama_endpoint.clone(), // Curator uses Ollama with small model
            model_name: config.curator_model_name.clone(),
            embedding_dim: config.qdrant_vector_dim,
            max_context_length: 2048,
            distillation_batch_size: 32,
            clustering_threshold: 0.8,
            quality_threshold: config.curator_quality_threshold,
            minimum_threshold: config.curator_minimum_threshold,
            timeout_secs: config.curator_timeout_secs,
            temperature: config.curator_temperature,
            max_tokens: config.curator_max_tokens,
            assessment_prompt_template: config.assessment_prompt_template.clone(),
            parse_mode: crate::curator_parser::ParserMode::from_env(),
            // Heuristic parser defaults (configurable via env if needed)
            heuristic_max_length: env_with_fallback(&["CURATOR_HEURISTIC_MAX_LENGTH"])
                .and_then(|v| v.parse().ok())
                .unwrap_or(500),
            heuristic_optimal_entropy_low: env_with_fallback(&["CURATOR_HEURISTIC_ENTROPY_LOW"])
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.5),
            heuristic_optimal_entropy_high: env_with_fallback(&["CURATOR_HEURISTIC_ENTROPY_HIGH"])
                .and_then(|v| v.parse().ok())
                .unwrap_or(2.5),
            heuristic_optimal_entropy_score: env_with_fallback(&[
                "CURATOR_HEURISTIC_OPTIMAL_SCORE",
            ])
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.9),
            heuristic_suboptimal_entropy_score: env_with_fallback(&[
                "CURATOR_HEURISTIC_SUBOPTIMAL_SCORE",
            ])
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.6),
            heuristic_length_weight: env_with_fallback(&["CURATOR_HEURISTIC_LENGTH_WEIGHT"])
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.4),
        }
    }
}

fn load_env_file(path: &Path) -> Result<()> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("unable to read env file {}", path.display()))?;

    for (line_index, line) in contents.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let mut parts = trimmed.splitn(2, '=');
        let key = parts.next().unwrap_or("").trim();
        if key.is_empty() {
            continue;
        }
        let raw_value = parts.next().unwrap_or("").trim();
        let value = normalise_env_value(raw_value);
        env::set_var(key, value);
    }

    Ok(())
}

fn normalise_env_value(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.len() >= 2 {
        let first = trimmed.as_bytes()[0] as char;
        let last = trimmed.as_bytes()[trimmed.len() - 1] as char;
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            return trimmed[1..trimmed.len() - 1].trim().to_string();
        }
    }
    trimmed.trim_end_matches('\r').to_string()
}
fn env_with_fallback(keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Ok(value) = env::var(key) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}
