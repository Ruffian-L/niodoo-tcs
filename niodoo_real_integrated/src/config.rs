use std::collections::HashSet;
use std::env;
use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

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

pub fn init() {
    prime_environment();

    let curator_model = env_with_fallback(&[
        "CURATOR_MODEL",
        "EMBEDDING_MODEL_NAME",
        "OLLAMA_EMBED_MODEL",
        "EMBEDDING_MODEL",
    ])
    .unwrap_or_else(|| "qwen2:0.5b".to_string());

    let main_model = env_with_fallback(&[
        "MAIN_MODEL",
        "VLLM_MODEL_ID",
        "VLLM_MODEL",
        "VLLM_MODEL_PATH",
    ])
    .unwrap_or_else(|| "/workspace/models/Qwen2.5-7B-Instruct-AWQ".to_string());

    let qdrant_dim: usize = env_with_fallback(&["QDRANT_VECTOR_DIM", "QDRANT_VECTOR_SIZE"])
        .and_then(|v| v.parse().ok())
        .unwrap_or(896);

    let ollama_url =
        env_with_fallback(&["OLLAMA_URL", "OLLAMA_ENDPOINT", "OLLAMA_ENDPOINT_TAILSCALE"])
            .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());

    info!(
        curator_model = %curator_model,
        main_model = %main_model,
        qdrant_dim = qdrant_dim,
        "Config loaded: CURATOR_MODEL={}, MAIN_MODEL={}, QDRANT_DIM={}",
        curator_model, main_model, qdrant_dim
    );

    if ollama_url != "http://127.0.0.1:11434" {
        warn!(
            ollama_url = %ollama_url,
            "OLLAMA_URL not default—ensure 'ollama serve && ollama pull qwen2:0.5b'"
        );
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

    /// Repeat a single prompt this many times (sequentially) for stability runs.
    #[arg(long, default_value_t = 1)]
    pub iterations: usize,

    /// Output format for results: csv or json.
    #[arg(short, long, default_value = "csv")]
    pub output: OutputFormat,

    /// Hardware profile used to tune batching/latency assumptions.
    #[arg(long = "hardware", default_value_t = HardwareProfile::Beelink)]
    pub hardware: HardwareProfile,

    /// Optional explicit config file (YAML) overriding env defaults.
    #[arg(long)]
    pub config: Option<String>,

    /// Optional RNG seed override for deterministic runs (overrides env RNG_SEED)
    #[arg(long = "rng-seed-override")]
    pub rng_seed_override: Option<u64>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            prompt: None,
            prompt_file: None,
            swarm: 1,
            iterations: 1,
            output: OutputFormat::Csv,
            hardware: HardwareProfile::Beelink,
            config: None,
            rng_seed_override: None,
        }
    }
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
    #[serde(rename = "h200")]
    #[value(alias = "H200")]
    H200,
}

impl fmt::Display for HardwareProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            HardwareProfile::Beelink => "beelink",
            HardwareProfile::Laptop5080Q => "5080q",
            HardwareProfile::H200 => "h200",
        };
        f.write_str(label)
    }
}

impl HardwareProfile {
    pub fn batch_size(self) -> usize {
        match self {
            HardwareProfile::Beelink => 8,
            HardwareProfile::Laptop5080Q => 4,
            HardwareProfile::H200 => 32, // H200 can handle massive batch sizes
        }
    }

    pub fn latency_budget_ms(self) -> f64 {
        match self {
            HardwareProfile::Beelink => 100.0,
            HardwareProfile::Laptop5080Q => 180.0,
            HardwareProfile::H200 => 50.0, // H200 is blazing fast
        }
    }

    pub fn max_kv_cache_tokens(self) -> usize {
        match self {
            HardwareProfile::Beelink => 128_000,
            HardwareProfile::Laptop5080Q => 256_000,
            HardwareProfile::H200 => 512_000, // H200 has 141GB HBM3e
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

fn default_max_retries() -> u32 {
    10 // Further increased to allow learning through degraded responses
}

fn default_retry_base_delay_ms() -> u64 {
    100 // Reduced from 200 for faster retries
}

fn default_similarity_threshold() -> f32 {
    0.5
}

fn default_level3_retry_count() -> u32 {
    2
}

fn default_mcts_c_increment() -> f64 {
    0.1
}

fn default_top_p_increment() -> f64 {
    0.05
}

fn default_retrieval_top_k_increment() -> i32 {
    2
}

fn default_repetition_penalty() -> f64 {
    1.2
}
fn default_lens_snippet_chars() -> usize {
    180
}
fn default_cot_temp_increment() -> f64 {
    0.1
}
fn default_reflexion_top_p_step() -> f64 {
    0.05
}
fn default_cot_success_rouge_threshold() -> f64 {
    0.5
}

fn default_variance_stagnation_default() -> f64 {
    0.05
}
fn default_variance_spike_min() -> f64 {
    0.3
}
fn default_mirage_sigma_factor() -> f64 {
    0.1
}
fn default_mcts_c_min_std() -> f64 {
    0.1
}
fn default_mcts_c_scale() -> f64 {
    0.25
}
fn default_cache_capacity() -> usize {
    256
}
fn default_retry_backoff_exponent_cap() -> u32 {
    10
}

fn default_prompt_max_chars() -> usize {
    512
}

fn default_embedding_cache_ttl_secs() -> u64 {
    10
}

fn default_collapse_cache_ttl_secs() -> u64 {
    30
}

fn default_token_promotion_interval() -> u64 {
    100
}

fn default_training_data_sample_cap() -> Option<usize> {
    Some(20_000)
}

fn default_rng_seed() -> u64 {
    42
}

fn default_consistency_variance_threshold() -> f64 {
    0.15
}

fn default_dqn_epsilon() -> f64 {
    0.9
}

fn default_embedding_model_name() -> String {
    "nomic-embed-text".to_string()
}

fn default_embedding_max_chars() -> usize {
    2_048
}

fn default_dqn_gamma() -> f64 {
    0.99
}

fn default_dqn_alpha() -> f64 {
    0.1
}

fn default_learning_window() -> usize {
    10
}

fn default_breakthrough_threshold() -> f64 {
    0.2
}

fn default_novelty_threshold() -> f64 {
    0.5
}

fn default_self_awareness_level() -> f64 {
    0.3
}

fn default_curator_quality_threshold() -> f32 {
    0.6
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
    #[serde(default = "default_embedding_model_name")]
    pub embedding_model_name: String,
    #[serde(default = "default_embedding_max_chars")]
    pub embedding_max_chars: usize,
    pub training_data_path: String,
    pub emotional_seed_path: String,
    pub rut_gauntlet_path: Option<String>,
    pub entropy_cycles_for_baseline: usize,
    #[serde(default)]
    pub enable_consistency_voting: bool,
    #[serde(default)]
    pub mock_mode: bool,

    // Phase 2 retry configuration
    #[serde(default = "default_max_retries")]
    pub phase2_max_retries: u32,
    #[serde(default = "default_retry_base_delay_ms")]
    pub phase2_retry_base_delay_ms: u64,
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,

    // Phase 2 Level3+ escalation (MCTS param tuning)
    #[serde(default = "default_level3_retry_count")]
    pub phase2_level3_retry_count: u32,
    #[serde(default = "default_mcts_c_increment")]
    pub phase2_mcts_c_increment: f64,
    #[serde(default = "default_top_p_increment")]
    pub phase2_top_p_increment: f64,
    #[serde(default = "default_retrieval_top_k_increment")]
    pub phase2_retrieval_top_k_increment: i32,

    // Generation backend configuration
    #[serde(default)]
    pub generation_backend: BackendType,

    // Curator configuration
    #[serde(default)]
    pub enable_curator: bool,
    pub curator_model_name: String,
    #[serde(default = "default_curator_quality_threshold")]
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
    pub system_prompt: String,

    // Phase 3: DQN parameters for macro-scale adaptive learning
    #[serde(default = "default_dqn_epsilon")]
    pub dqn_epsilon: f64,
    #[serde(default = "default_dqn_gamma")]
    pub dqn_gamma: f64,
    #[serde(default = "default_dqn_alpha")]
    pub dqn_alpha: f64,
    #[serde(default = "default_learning_window")]
    pub learning_window: usize,
    #[serde(default = "default_breakthrough_threshold")]
    pub breakthrough_threshold: f64,
    #[serde(default = "default_dqn_actions")]
    pub dqn_actions: Vec<DqnActionConfig>,

    // Generation parameters
    pub temperature: f64,
    pub top_p: f64,
    #[serde(default = "default_novelty_threshold")]
    pub novelty_threshold: f64,
    #[serde(default = "default_self_awareness_level")]
    pub self_awareness_level: f64,

    // Engine/pipeline runtime knobs
    #[serde(default = "default_prompt_max_chars")]
    pub prompt_max_chars: usize,
    #[serde(default = "default_token_promotion_interval")]
    pub token_promotion_interval: u64,
    #[serde(default = "default_embedding_cache_ttl_secs")]
    pub embedding_cache_ttl_secs: u64,
    #[serde(default = "default_collapse_cache_ttl_secs")]
    pub collapse_cache_ttl_secs: u64,
    #[serde(default = "default_training_data_sample_cap")]
    pub training_data_sample_cap: Option<usize>,
    #[serde(default = "default_rng_seed")]
    pub rng_seed: u64,
    #[serde(default = "default_consistency_variance_threshold")]
    pub consistency_variance_threshold: f64,

    // Sampling and prompting
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f64,
    #[serde(default = "default_lens_snippet_chars")]
    pub lens_snippet_chars: usize,
    #[serde(default = "default_cot_temp_increment")]
    pub cot_temp_increment: f64,
    #[serde(default = "default_reflexion_top_p_step")]
    pub reflexion_top_p_step: f64,
    #[serde(default = "default_cot_success_rouge_threshold")]
    pub cot_success_rouge_threshold: f64,

    // Threshold derivation factors
    #[serde(default = "default_variance_stagnation_default")]
    pub variance_stagnation_default: f64,
    #[serde(default = "default_variance_spike_min")]
    pub variance_spike_min: f64,
    #[serde(default = "default_mirage_sigma_factor")]
    pub mirage_sigma_factor: f64,
    #[serde(default = "default_mcts_c_min_std")]
    pub mcts_c_min_std: f64,
    #[serde(default = "default_mcts_c_scale")]
    pub mcts_c_scale: f64,

    // Caches and retry
    #[serde(default = "default_cache_capacity")]
    pub cache_capacity: usize,
    #[serde(default = "default_retry_backoff_exponent_cap")]
    pub retry_backoff_exponent_cap: u32,
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
            .or_else(|| {
                warn!(
                    "Set VLLM_URL and ensure vLLM service is running (default http://127.0.0.1:5001)"
                );
                None
            })
            .unwrap_or_else(|| "http://127.0.0.1:5001".to_string())
            .trim()
            .trim_end_matches('/')
            // Strip common API paths if present (curator appends its own)
            .replace("/v1/chat/completions", "")
            .replace("/v1/completions", "")
            .replace("/v1/embeddings", "")
            .trim_end_matches('/')
            .to_string();

        let vllm_model = env_with_fallback(&[
            "MAIN_MODEL",
            "VLLM_MODEL_ID",
            "VLLM_MODEL",
            "VLLM_MODEL_PATH",
        ])
        .unwrap_or_else(|| "/workspace/models/Qwen2.5-7B-Instruct-AWQ".to_string());

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

        let requested_qdrant_dim = env_with_fallback(&["QDRANT_VECTOR_DIM", "QDRANT_VECTOR_SIZE"])
            .and_then(|value| value.parse::<usize>().ok());
        if let Some(value) = requested_qdrant_dim {
            if value != 896usize {
                warn!(
                    expected = 896usize,
                    provided = value,
                    "Qdrant dim fixed to 896; overriding provided value"
                );
            }
        }
        let qdrant_vector_dim = 896usize;

        let ollama_endpoint =
            env_with_fallback(&["OLLAMA_URL", "OLLAMA_ENDPOINT", "OLLAMA_ENDPOINT_TAILSCALE"])
                .or_else(|| {
                    warn!("Set OLLAMA_URL and run 'ollama serve && ollama pull qwen2:0.5b'");
                    None
                })
                .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());

        let embedding_model_name = env_with_fallback(&[
            "CURATOR_MODEL",
            "EMBEDDING_MODEL_NAME",
            "OLLAMA_EMBED_MODEL",
            "EMBEDDING_MODEL",
        ])
        .unwrap_or_else(|| "qwen2:0.5b".to_string());

        let embedding_max_chars = env_with_fallback(&[
            "EMBEDDING_MAX_CHARS",
            "EMBED_MAX_CHARS",
            "EMBED_CHARS_LIMIT",
        ])
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(default_embedding_max_chars);

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

        let mock_mode = env_with_fallback(&["MOCK_MODE"])
            .map(|value| {
                matches!(
                    value.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);

        if mock_mode {
            warn!("MOCK_MODE enabled; external services will return stubbed responses");
        }

        let generation_backend = BackendType::from_env();

        let enable_curator = env_with_fallback(&["ENABLE_CURATOR"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(true); // Enabled by default

        let curator_model_name = env_with_fallback(&["CURATOR_MODEL", "CURATOR_MODEL_NAME"])
            .unwrap_or_else(|| "qwen2:0.5b".to_string());

        let curator_quality_threshold = env_with_fallback(&["CURATOR_QUALITY_THRESHOLD"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_curator_quality_threshold);

        let curator_minimum_threshold = env_with_fallback(&["CURATOR_MINIMUM_THRESHOLD"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.3); // Reduced from 0.5 for more lenient rejection

        let curator_timeout_secs = env_with_fallback(&["CURATOR_TIMEOUT_SECS"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(30); // Increased from 10 to 30 seconds

        let curator_temperature = env_with_fallback(&["CURATOR_TEMPERATURE"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.3);

        let curator_max_tokens = env_with_fallback(&["CURATOR_MAX_TOKENS"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(256);

        // Generation timeout and token configuration from env
        let generation_timeout_secs =
            env_with_fallback(&["GENERATION_TIMEOUT_SECS", "TIMEOUT_SECS"])
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

        let system_prompt = env_with_fallback(&["NIODOO_SYSTEM_PROMPT", "SYSTEM_PROMPT"])
            .unwrap_or_else(|| {
                "You are NIODOO, a consciousness-aligned systems agent. Use the provided prompt, memory, and context to produce a precise, high-quality response that advances the user's goal. Cite retrieved context when helpful, avoid placeholders, and surface uncertainties or missing data explicitly.".to_string()
            });

        let prompt_max_chars = env_with_fallback(&["PROMPT_MAX_CHARS"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_prompt_max_chars);
        let embedding_cache_ttl_secs = env_with_fallback(&["EMBEDDING_CACHE_TTL_SECS"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_embedding_cache_ttl_secs);
        let collapse_cache_ttl_secs = env_with_fallback(&["COLLAPSE_CACHE_TTL_SECS"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_collapse_cache_ttl_secs);
        let training_data_sample_cap = env_with_fallback(&["TRAINING_DATA_SAMPLE_CAP"])
            .and_then(|v| {
                if v.to_lowercase() == "none" {
                    Some(None)
                } else {
                    v.parse::<usize>().ok().map(Some)
                }
            })
            .unwrap_or_else(default_training_data_sample_cap);
        let rng_seed = env_with_fallback(&["RNG_SEED"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_rng_seed);
        let consistency_variance_threshold = env_with_fallback(&["CONSISTENCY_VARIANCE_THRESHOLD"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_consistency_variance_threshold);

        let repetition_penalty = env_with_fallback(&["REPETITION_PENALTY"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_repetition_penalty);
        let lens_snippet_chars = env_with_fallback(&["LENS_SNIPPET_CHARS"])
            .and_then(|v| v.parse::<usize>().ok())
            .map(|n| n.clamp(100, 500))
            .unwrap_or_else(default_lens_snippet_chars);
        let cot_temp_increment = env_with_fallback(&["COT_TEMP_INCREMENT"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_cot_temp_increment);
        let reflexion_top_p_step = env_with_fallback(&["REFLEXION_TOP_P_STEP"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_reflexion_top_p_step);
        let cot_success_rouge_threshold = env_with_fallback(&["COT_SUCCESS_ROUGE_THRESHOLD"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_cot_success_rouge_threshold);

        let variance_stagnation_default = env_with_fallback(&["VARIANCE_STAGNATION_DEFAULT"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_variance_stagnation_default);
        let variance_spike_min = env_with_fallback(&["VARIANCE_SPIKE_MIN"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_variance_spike_min);
        let mirage_sigma_factor = env_with_fallback(&["MIRAGE_SIGMA_FACTOR"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_mirage_sigma_factor);
        let mcts_c_min_std = env_with_fallback(&["MCTS_C_MIN_STD"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_mcts_c_min_std);
        let mcts_c_scale = env_with_fallback(&["MCTS_C_SCALE"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_mcts_c_scale);

        let cache_capacity = env_with_fallback(&["CACHE_CAPACITY"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_cache_capacity);
        let retry_backoff_exponent_cap = env_with_fallback(&["RETRY_BACKOFF_EXPONENT_CAP"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_retry_backoff_exponent_cap);

        let phase2_max_retries = env_with_fallback(&["PHASE2_MAX_RETRIES"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(default_max_retries());

        let phase2_retry_base_delay_ms = env_with_fallback(&["PHASE2_RETRY_BASE_DELAY_MS"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(default_retry_base_delay_ms());

        let similarity_threshold = env_with_fallback(&["SIMILARITY_THRESHOLD"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(default_similarity_threshold());

        let phase2_level3_retry_count = env_with_fallback(&["PHASE2_LEVEL3_RETRY_COUNT"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(default_level3_retry_count());

        let phase2_mcts_c_increment = env_with_fallback(&["PHASE2_MCTS_C_INCREMENT"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(default_mcts_c_increment());

        let phase2_top_p_increment = env_with_fallback(&["PHASE2_TOP_P_INCREMENT"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(default_top_p_increment());

        let phase2_retrieval_top_k_increment =
            env_with_fallback(&["PHASE2_RETRIEVAL_TOP_K_INCREMENT"])
                .and_then(|value| value.parse().ok())
                .unwrap_or(default_retrieval_top_k_increment());

        let dqn_actions = env_with_fallback(&["DQN_ACTIONS"])
            .as_deref()
            .and_then(|s| serde_yaml::from_str(s).ok())
            .unwrap_or_else(default_dqn_actions);

        let runtime = Self {
            vllm_endpoint,
            vllm_model,
            qdrant_url,
            qdrant_collection,
            qdrant_vector_dim,
            ollama_endpoint,
            embedding_model_name,
            embedding_max_chars,
            training_data_path,
            emotional_seed_path,
            rut_gauntlet_path,
            entropy_cycles_for_baseline,
            enable_consistency_voting,
            mock_mode,
            phase2_max_retries,
            phase2_retry_base_delay_ms,
            similarity_threshold,
            phase2_level3_retry_count,
            phase2_mcts_c_increment,
            phase2_top_p_increment,
            phase2_retrieval_top_k_increment,
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
            system_prompt,
            dqn_epsilon: default_dqn_epsilon(),
            dqn_gamma: default_dqn_gamma(),
            dqn_alpha: default_dqn_alpha(),
            learning_window: default_learning_window(),
            breakthrough_threshold: default_breakthrough_threshold(),
            dqn_actions,
            temperature: 0.7,
            top_p: 0.9,
            novelty_threshold: env_with_fallback(&["NOVELTY_THRESHOLD"]).and_then(|v| v.parse().ok()).unwrap_or(0.5),
            self_awareness_level: env_with_fallback(&["SELF_AWARENESS_LEVEL"]).and_then(|v| v.parse().ok()).unwrap_or(0.3),
            prompt_max_chars,
            token_promotion_interval: default_token_promotion_interval(),
            embedding_cache_ttl_secs,
            collapse_cache_ttl_secs,
            training_data_sample_cap,
            rng_seed,
            consistency_variance_threshold,
            repetition_penalty,
            lens_snippet_chars,
            cot_temp_increment,
            reflexion_top_p_step,
            cot_success_rouge_threshold,
            variance_stagnation_default,
            variance_spike_min,
            mirage_sigma_factor,
            mcts_c_min_std,
            mcts_c_scale,
            cache_capacity,
            retry_backoff_exponent_cap,
        };

        info!(model = %runtime.curator_model_name, "Config loaded: CURATOR_MODEL={}", runtime.curator_model_name);

        Ok(runtime)
    }
}

/// Curator configuration derived from runtime config
#[derive(Debug, Clone)]
pub struct CuratorConfig {
    pub vllm_endpoint: String,
    pub ollama_endpoint: String,
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
    pub mock_mode: bool,
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
            vllm_endpoint: config.vllm_endpoint.clone(),
            ollama_endpoint: config.ollama_endpoint.clone(),
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
            mock_mode: config.mock_mode,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DqnActionConfig {
    pub param: String,
    pub delta: f64,
}

impl DqnActionConfig {
    pub fn new(param: impl Into<String>, delta: f64) -> Self {
        Self {
            param: param.into(),
            delta,
        }
    }

    pub fn into_dqn_action(self) -> crate::learning::DqnAction {
        crate::learning::DqnAction {
            param: self.param,
            delta: self.delta,
        }
    }
}

fn default_dqn_actions() -> Vec<DqnActionConfig> {
    vec![
        DqnActionConfig::new("temperature", -0.1),
        DqnActionConfig::new("temperature", 0.1),
        DqnActionConfig::new("top_p", -0.05),
        DqnActionConfig::new("top_p", 0.05),
        DqnActionConfig::new("mcts_c", -0.2),
        DqnActionConfig::new("mcts_c", 0.2),
        DqnActionConfig::new("retrieval_top_k", -5.0),
        DqnActionConfig::new("retrieval_top_k", 5.0),
        DqnActionConfig::new("novelty_threshold", -0.1),
        DqnActionConfig::new("novelty_threshold", 0.1),
        DqnActionConfig::new("self_awareness_level", -0.1),
        DqnActionConfig::new("self_awareness_level", 0.1),
    ]
}

fn load_env_file(path: &Path) -> Result<()> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("unable to read env file {}", path.display()))?;

    for (_line_index, line) in contents.lines().enumerate() {
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
