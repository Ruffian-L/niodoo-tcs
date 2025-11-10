use std::collections::{HashMap, HashSet};
use std::env;
use std::fmt;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Parser, ValueEnum};
use once_cell::sync::OnceCell;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

static ENV_OVERRIDES: OnceCell<RwLock<HashMap<String, String>>> = OnceCell::new();

fn env_store() -> &'static RwLock<HashMap<String, String>> {
    ENV_OVERRIDES.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn set_env_override<K, V>(key: K, value: V)
where
    K: Into<String>,
    V: Into<String>,
{
    let key = key.into();
    let value = value.into();
    env_store().write().insert(key.clone(), value.clone());

    if let Err(error) = append_config_audit_log(&key, &value) {
        warn!(%key, ?error, "failed to record configuration override");
    }
}

fn append_config_audit_log(key: &str, value: &str) -> Result<()> {
    let path = PathBuf::from("./logs/config_audit.log");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "unable to create config audit directory at {}",
                parent.display()
            )
        })?;
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("unable to open config audit log at {}", path.display()))?;

    let timestamp = Utc::now().to_rfc3339();
    let digest = blake3::hash(value.as_bytes());
    writeln!(
        file,
        "{timestamp} key={key} value_hash={} char_count={}",
        digest.to_hex(),
        value.chars().count()
    )?;
    Ok(())
}

pub fn env_value(key: &str) -> Option<String> {
    env_store()
        .read()
        .get(key)
        .cloned()
        .or_else(|| env::var(key).ok())
}

pub fn env_var(key: &str) -> std::result::Result<String, std::env::VarError> {
    if let Some(value) = env_store().read().get(key) {
        return Ok(value.clone());
    }
    env::var(key)
}

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
    .unwrap_or_else(|| "/workspace/models/hf_cache/models--Qwen--Qwen2.5-7B-Instruct-AWQ".to_string());
    
    info!("Config: main_model={}", main_model);  // ADD THIS

    // DEBUG: Log model ID source
    if std::env::var("VLLM_MODEL_ID").is_ok() {
        tracing::info!("Model ID from VLLM_MODEL_ID env var: {}", main_model);
    } else if std::env::var("VLLM_MODEL").is_ok() {
        tracing::info!("Model ID from VLLM_MODEL env var: {}", main_model);
    } else if std::env::var("MAIN_MODEL").is_ok() {
        tracing::info!("Model ID from MAIN_MODEL env var: {}", main_model);
    } else {
        tracing::info!("Model ID using default: {}", main_model);
    }

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
    #[serde(rename = "5090")]
    #[value(alias = "RTX5090")]
    #[value(alias = "rtx5090")]
    RTX5090,
}

impl fmt::Display for HardwareProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            HardwareProfile::Beelink => "beelink",
            HardwareProfile::Laptop5080Q => "5080q",
            HardwareProfile::H200 => "h200",
            HardwareProfile::RTX5090 => "5090",
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
            HardwareProfile::RTX5090 => 64, // RTX 5090 Blackwell - aggressive batching
        }
    }

    pub fn latency_budget_ms(self) -> f64 {
        match self {
            HardwareProfile::Beelink => 100.0,
            HardwareProfile::Laptop5080Q => 180.0,
            HardwareProfile::H200 => 50.0, // H200 is blazing fast
            HardwareProfile::RTX5090 => 30.0, // RTX 5090 is even faster
        }
    }

    pub fn max_kv_cache_tokens(self) -> usize {
        match self {
            HardwareProfile::Beelink => 128_000,
            HardwareProfile::Laptop5080Q => 256_000,
            HardwareProfile::H200 => 512_000, // H200 has 141GB HBM3e
            HardwareProfile::RTX5090 => 512_000, // RTX 5090 has 32GB GDDR7 - match H200 capacity
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CuratorBackend {
    #[serde(rename = "ollama")]
    Ollama,
    #[serde(rename = "vllm")]
    Vllm,
}

impl Default for CuratorBackend {
    fn default() -> Self {
        // Default to vLLM for better reliability
        CuratorBackend::Vllm
    }
}

/// Qdrant quantization type for vector compression
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum QuantizationType {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "scalar_pq4")]
    ScalarPQ4,
}

impl Default for QuantizationType {
    fn default() -> Self {
        QuantizationType::None
    }
}

impl CuratorBackend {
    pub fn from_env() -> Self {
        match env_with_fallback(&["CURATOR_BACKEND", "CURATOR_TYPE"]) {
            Some(value) => match value.to_ascii_lowercase().as_str() {
                "vllm" | "vllm_gpu" => CuratorBackend::Vllm,
                "ollama" | "ollama_cpu" => CuratorBackend::Ollama,
                _ => {
                    warn!(%value, "Invalid curator backend; defaulting to Ollama for mini Qwen");
                    CuratorBackend::Ollama  // Change default to Ollama
                }
            },
            None => CuratorBackend::Ollama, // Default to Ollama for mini model
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TopologyMode {
    #[serde(rename = "hybrid")]
    Hybrid,
    #[serde(rename = "baseline")]
    Baseline,
}

impl Default for TopologyMode {
    fn default() -> Self {
        TopologyMode::Hybrid
    }
}

impl TopologyMode {
    pub fn from_env() -> Self {
        match env_with_fallback(&["TOPOLOGY_MODE", "TCS_TOPOLOGY_MODE"]) {
            Some(value) => match TopologyMode::from_str(&value) {
                Ok(mode) => mode,
                Err(error) => {
                    warn!(%value, %error, "Invalid topology mode; defaulting to hybrid");
                    TopologyMode::Hybrid
                }
            },
            None => TopologyMode::Hybrid,
        }
    }
}

impl FromStr for TopologyMode {
    type Err = anyhow::Error;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input.trim().to_ascii_lowercase().as_str() {
            "hybrid" => Ok(TopologyMode::Hybrid),
            "baseline" => Ok(TopologyMode::Baseline),
            other => Err(anyhow::anyhow!("unsupported topology mode '{other}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RceBetaMetaWeights {
    pub alpha_betti: f64,
    pub alpha_meta: f64,
    pub alpha_motif: f64,
    pub alpha_sheaf: f64,
}

impl Default for RceBetaMetaWeights {
    fn default() -> Self {
        Self {
            alpha_betti: 1.0,
            alpha_meta: 1.0,
            alpha_motif: 1.0,
            alpha_sheaf: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RceConsensusConfig {
    pub enabled: bool,
    pub analyzers: usize,
    pub quorum: usize,
}

fn default_security_rate_limit_window_secs() -> u64 {
    60
}

fn default_security_rate_limit_max_requests() -> u32 {
    45
}

fn default_security_allow_control_chars() -> bool {
    false
}

fn default_security_banned_patterns() -> Vec<String> {
    vec![
        r"(?i)\b(?:drop|delete)\s+(?:table|database)\b".to_string(),
        r"(?i)\bunion\s+select\b".to_string(),
        r"(?i)<script".to_string(),
        r"(?i)\brm\s+-rf\s+/".to_string(),
    ]
}

fn default_security_audit_log_path() -> String {
    "./logs/security_audit.log".to_string()
}

fn default_security_prompt_max_chars() -> usize {
    0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    #[serde(default = "default_security_rate_limit_window_secs")]
    pub rate_limit_window_secs: u64,
    #[serde(default = "default_security_rate_limit_max_requests")]
    pub rate_limit_max_requests: u32,
    #[serde(default = "default_security_allow_control_chars")]
    pub allow_control_chars: bool,
    #[serde(default = "default_security_banned_patterns")]
    pub banned_patterns: Vec<String>,
    #[serde(default = "default_security_audit_log_path")]
    pub audit_log_path: String,
    #[serde(default = "default_security_prompt_max_chars")]
    pub prompt_max_chars: usize,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            rate_limit_window_secs: default_security_rate_limit_window_secs(),
            rate_limit_max_requests: default_security_rate_limit_max_requests(),
            allow_control_chars: default_security_allow_control_chars(),
            banned_patterns: default_security_banned_patterns(),
            audit_log_path: default_security_audit_log_path(),
            prompt_max_chars: default_security_prompt_max_chars(),
        }
    }
}

impl SecurityConfig {
    pub fn finalize(&mut self, prompt_max_chars: usize) {
        if self.prompt_max_chars == 0 {
            self.prompt_max_chars = prompt_max_chars;
        }
        if self.banned_patterns.is_empty() {
            self.banned_patterns = default_security_banned_patterns();
        }
        if self.audit_log_path.trim().is_empty() {
            self.audit_log_path = default_security_audit_log_path();
        }
        if self.rate_limit_window_secs == 0 {
            self.rate_limit_window_secs = default_security_rate_limit_window_secs();
        }
        if self.rate_limit_max_requests == 0 {
            self.rate_limit_max_requests = default_security_rate_limit_max_requests();
        }
    }

    pub fn parse_patterns(raw: &str) -> Vec<String> {
        raw.split(|c| c == ',' || c == ';')
            .map(|pattern| pattern.trim())
            .filter(|pattern| !pattern.is_empty())
            .map(|pattern| pattern.to_string())
            .collect()
    }
}

fn default_max_retries() -> u32 {
    3 // Keep retry budget tight to avoid runaway latency
}

fn default_retry_base_delay_ms() -> u64 {
    100 // Reduced from 200 for faster retries
}

fn default_phase2_cot_iterations() -> u32 {
    1
}

fn default_phase2_retry_backoff_cap_ms() -> u64 {
    1_500
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
fn default_cache_compression_min_bytes() -> usize {
    2048
}
fn default_cache_prefetch_prompts() -> usize {
    8
}
fn default_cache_prefetch_top_hits() -> usize {
    3
}
fn default_cache_prefetch_parallelism() -> usize {
    2
}
fn default_cache_prefetch_enabled() -> bool {
    true
}
fn default_retry_backoff_exponent_cap() -> u32 {
    10
}

fn default_tokenizer_json() -> Option<String> {
    for key in ["TOKENIZER_JSON", "QWEN_TOKENIZER"] {
        if let Some(path) = env_value(key) {
            let candidate = PathBuf::from(&path);
            if candidate.exists() {
                return Some(path);
            }
        }
    }
    None
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

fn default_rce_enabled() -> bool { true }
fn default_rce_shadow_mode() -> bool { true }
fn default_rce_actions_enabled() -> bool { false }
fn default_rce_window_seconds() -> u64 { 10 }
fn default_rce_stride_seconds() -> u64 { 2 }
fn default_rce_beta_meta_weights() -> RceBetaMetaWeights { RceBetaMetaWeights::default() }
fn default_rce_breakthrough_threshold() -> f64 { 0.5 }
fn default_rce_erag_lambda() -> f64 { 0.0 }
fn default_rce_archive_backend() -> String { "Qdrant".to_string() }

fn default_telemetry_enabled() -> bool {
    env_value("NIODOO_TELEMETRY_ENABLED")
        .and_then(|v| v.parse().ok())
        .unwrap_or(false)
}

fn default_telemetry_port() -> u16 {
    env_value("NIODOO_TELEMETRY_PORT")
        .and_then(|v| v.parse().ok())
        .unwrap_or(9999)
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

fn default_breakthrough_rouge_min() -> f64 {
    0.65
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

fn default_curator_autonomous() -> bool {
    true
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
    #[serde(default)]
    pub qdrant_embedded: bool,
    pub ollama_endpoint: String,
    #[serde(default = "default_embedding_model_name")]
    pub embedding_model_name: String,
    #[serde(default)]
    pub embed_with_candle: bool,
    #[serde(default)]
    pub embed_model_dir: Option<String>,
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
    #[serde(default)]
    pub topology_mode: TopologyMode,

    // RCE (Recursive Connectome Engine) configuration
    #[serde(default = "default_rce_enabled")]
    pub rce_enabled: bool,
    #[serde(default = "default_rce_shadow_mode")]
    pub rce_shadow_mode: bool,
    #[serde(default = "default_rce_actions_enabled")]
    pub rce_actions_enabled: bool,
    #[serde(default = "default_rce_window_seconds")]
    pub rce_window_seconds: u64,
    #[serde(default = "default_rce_stride_seconds")]
    pub rce_stride_seconds: u64,
    #[serde(default = "default_rce_beta_meta_weights")]
    pub rce_beta_meta_weights: RceBetaMetaWeights,
    #[serde(default = "default_rce_breakthrough_threshold")]
    pub rce_breakthrough_threshold: f64,
    #[serde(default)]
    pub rce_consensus: RceConsensusConfig,
    #[serde(default = "default_rce_erag_lambda")]
    pub rce_erag_lambda: f64,
    #[serde(default = "default_rce_archive_backend")]
    pub rce_archive_backend: String,

    // Telemetry configuration
    #[serde(default = "default_telemetry_enabled")]
    pub telemetry_enabled: bool,
    #[serde(default = "default_telemetry_port")]
    pub telemetry_port: u16,

    // Phase 2 retry configuration
    #[serde(default = "default_max_retries")]
    pub phase2_max_retries: u32,
    #[serde(default = "default_retry_base_delay_ms")]
    pub phase2_retry_base_delay_ms: u64,
    #[serde(default = "default_phase2_cot_iterations")]
    pub phase2_cot_iterations: u32,
    #[serde(default = "default_phase2_retry_backoff_cap_ms")]
    pub phase2_retry_backoff_cap_ms: u64,
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
    #[serde(default = "default_curator_autonomous")]
    pub curator_autonomous: bool,
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
    #[serde(default = "default_breakthrough_rouge_min")]
    pub breakthrough_rouge_min: f64,
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
    #[serde(default = "default_tokenizer_json")]
    pub tokenizer_json: Option<String>,
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
    #[serde(default = "default_cache_compression_min_bytes")]
    pub cache_compression_min_bytes: usize,
    #[serde(default = "default_cache_prefetch_enabled")]
    pub cache_prefetch_enabled: bool,
    #[serde(default = "default_cache_prefetch_prompts")]
    pub cache_prefetch_prompts: usize,
    #[serde(default = "default_cache_prefetch_top_hits")]
    pub cache_prefetch_top_hits: usize,
    #[serde(default = "default_cache_prefetch_parallelism")]
    pub cache_prefetch_parallelism: usize,
    #[serde(default = "default_retry_backoff_exponent_cap")]
    pub retry_backoff_exponent_cap: u32,

    // Weighted Episodic Memory configuration
    #[serde(default)]
    pub weighted_memory_config: WeightedMemoryConfig,
    /// Disable memory storage to ERAG/Qdrant (best-effort store becomes a no-op)
    #[serde(default)]
    pub disable_memory_store: bool,

    // Resource budget and degradation configuration
    #[serde(default)]
    pub resource_budget_config: ResourceBudgetConfig,
    #[serde(default)]
    pub degradation_config: DegradationConfig,
    #[serde(default)]
    pub temporal_tda_config: TemporalTDAConfig,
    #[serde(default)]
    pub security: SecurityConfig,

    // Phase 1-6: Back-half pipeline optimizations
    #[serde(default)]
    pub optimized_erag: bool,
    #[serde(default = "default_erag_batch_size")]
    pub erag_batch_size: usize,
    #[serde(default = "default_erag_batch_flush_ms")]
    pub erag_batch_flush_ms: u64,
    #[serde(default)]
    pub qdrant_quantization: Option<QuantizationType>,
    #[serde(default)]
    pub use_approximate_tda: bool,
    #[serde(default = "default_fp16_qlora_adapters")]
    pub fp16_qlora_adapters: bool,
    #[serde(default = "default_parallel_curator_rouge")]
    pub parallel_curator_rouge: bool,

    // Training service configuration
    #[serde(default = "default_training_service_enabled")]
    pub training_service_enabled: bool,
    #[serde(default = "default_training_service_url")]
    pub training_service_url: String,
    #[serde(default = "default_training_service_use_grpc")]
    pub training_service_use_grpc: bool,
    #[serde(default = "default_adapter_storage_path")]
    pub adapter_storage_path: String,
    #[serde(default = "default_training_queue_path")]
    pub training_queue_path: String,
    #[serde(default)]
    pub use_gpu_fitness: bool,

    // Ablation testing flags (for validation framework)
    /// Bypass ERAG retrieval (zero-shot mode) for ablation testing
    #[serde(default)]
    pub erag_bypass: bool,
    /// Bypass nTokens layer for ablation testing
    #[serde(default)]
    pub n_tokens_bypass: bool,

    // Pipeline runtime configuration
    /// Curator feedback controller window size (number of responses to track)
    #[serde(default = "default_curator_feedback_window_size")]
    pub curator_feedback_window_size: usize,
    /// Curator feedback - threshold adjustment percentage per trend unit
    #[serde(default = "default_curator_feedback_threshold_adjustment")]
    pub curator_feedback_threshold_adjustment: f32,
    /// Curator feedback - adaptive threshold minimum bound
    #[serde(default = "default_curator_feedback_threshold_min")]
    pub curator_feedback_threshold_min: f32,
    /// Curator feedback - adaptive threshold maximum bound
    #[serde(default = "default_curator_feedback_threshold_max")]
    pub curator_feedback_threshold_max: f32,
    /// Curator feedback - quality trend threshold for parameter adjustments
    #[serde(default = "default_curator_feedback_quality_trend_threshold")]
    pub curator_feedback_quality_trend_threshold: f32,
    /// Curator feedback - temperature adjustment multiplier
    #[serde(default = "default_curator_feedback_temp_adjustment_multiplier")]
    pub curator_feedback_temp_adjustment_multiplier: f32,
    /// Curator feedback - learned rate threshold for top_p adjustment (low threshold)
    #[serde(default = "default_curator_feedback_learned_rate_low")]
    pub curator_feedback_learned_rate_low: f32,
    /// Curator feedback - quality threshold for top_p adjustment (low threshold)
    #[serde(default = "default_curator_feedback_quality_low")]
    pub curator_feedback_quality_low: f32,
    /// Curator feedback - top_p increase adjustment for low learned rate
    #[serde(default = "default_curator_feedback_top_p_increase")]
    pub curator_feedback_top_p_increase: f64,
    /// Curator feedback - learned rate threshold for top_p adjustment (high threshold)
    #[serde(default = "default_curator_feedback_learned_rate_high")]
    pub curator_feedback_learned_rate_high: f32,
    /// Curator feedback - quality threshold for top_p adjustment (high threshold)
    #[serde(default = "default_curator_feedback_quality_high")]
    pub curator_feedback_quality_high: f32,
    /// Curator feedback - top_p decrease adjustment for high learned rate
    #[serde(default = "default_curator_feedback_top_p_decrease")]
    pub curator_feedback_top_p_decrease: f64,
    /// Curator feedback - quality threshold for retrieval_top_k increase
    #[serde(default = "default_curator_feedback_retrieval_quality_threshold")]
    pub curator_feedback_retrieval_quality_threshold: f32,
    /// Curator feedback - retrieval_top_k increase adjustment
    #[serde(default = "default_curator_feedback_retrieval_top_k_increase")]
    pub curator_feedback_retrieval_top_k_increase: f64,
    /// Curator feedback - quality threshold for retrieval_top_k decrease (high threshold)
    #[serde(default = "default_curator_feedback_retrieval_quality_high")]
    pub curator_feedback_retrieval_quality_high: f32,
    /// Curator feedback - learned rate threshold for retrieval_top_k decrease
    #[serde(default = "default_curator_feedback_retrieval_learned_rate_high")]
    pub curator_feedback_retrieval_learned_rate_high: f32,
    /// Curator feedback - retrieval_top_k decrease adjustment
    #[serde(default = "default_curator_feedback_retrieval_top_k_decrease")]
    pub curator_feedback_retrieval_top_k_decrease: f64,
    /// Pipeline - retrieval top_k minimum limit
    #[serde(default = "default_pipeline_retrieval_top_k_min")]
    pub pipeline_retrieval_top_k_min: usize,
    /// Pipeline - retrieval top_k maximum limit
    #[serde(default = "default_pipeline_retrieval_top_k_max")]
    pub pipeline_retrieval_top_k_max: usize,
    /// Pipeline - timing split ratio for compass/erag parallel execution (default 0.5 = 50/50)
    #[serde(default = "default_pipeline_timing_split_ratio")]
    pub pipeline_timing_split_ratio: f64,
    /// Pipeline - healing state knot complexity threshold
    #[serde(default = "default_pipeline_healing_knot_threshold")]
    pub pipeline_healing_knot_threshold: f64,
    /// Pipeline - healing state spectral gap threshold
    #[serde(default = "default_pipeline_healing_spectral_gap_threshold")]
    pub pipeline_healing_spectral_gap_threshold: f64,
    /// Pipeline - UCB1 score maximum clamp value
    #[serde(default = "default_pipeline_ucb1_max_clamp")]
    pub pipeline_ucb1_max_clamp: f64,
    /// Pipeline - curator quality score increment for refinement passes
    #[serde(default = "default_pipeline_quality_score_increment")]
    pub pipeline_quality_score_increment: f32,
    /// Pipeline - parameter adjustment minimum bounds (temperature/top_p)
    #[serde(default = "default_pipeline_param_min")]
    pub pipeline_param_min: f64,
    /// Pipeline - parameter adjustment maximum bounds (temperature/top_p)
    #[serde(default = "default_pipeline_param_max")]
    pub pipeline_param_max: f64,
    /// Pipeline - retrieval top_k increment minimum bound
    #[serde(default = "default_pipeline_retrieval_top_k_increment_min")]
    pub pipeline_retrieval_top_k_increment_min: f64,
    /// Pipeline - retrieval top_k increment maximum bound
    #[serde(default = "default_pipeline_retrieval_top_k_increment_max")]
    pub pipeline_retrieval_top_k_increment_max: f64,
    /// Topology memory analyzer - similarity threshold
    #[serde(default = "default_topology_memory_analyzer_threshold")]
    pub topology_memory_analyzer_threshold: f64,
    /// Discovery buffer processing interval in seconds
    #[serde(default = "default_discovery_buffer_interval_secs")]
    pub discovery_buffer_interval_secs: u64,
    /// Embedding cache capacity (number of entries)
    #[serde(default = "default_embedding_cache_capacity")]
    pub embedding_cache_capacity: usize,
    /// Collapse cache capacity (number of entries)
    #[serde(default = "default_collapse_cache_capacity")]
    pub collapse_cache_capacity: usize,
    /// MCTS exploration constant (UCB1 exploration parameter, typically sqrt(2))
    #[serde(default = "default_mcts_exploration_constant")]
    pub mcts_exploration_constant: f64,
    /// MCTS search depth (maximum tree depth)
    #[serde(default = "default_mcts_depth")]
    pub mcts_depth: usize,
    /// Discovery buffer threshold (batch size for processing discoveries)
    #[serde(default = "default_discovery_buffer_threshold")]
    pub discovery_buffer_threshold: usize,
    /// GPU fitness refresh interval in seconds
    #[serde(default = "default_gpu_fitness_refresh_interval_secs")]
    pub gpu_fitness_refresh_interval_secs: u64,
    /// Learning loop timeout in seconds
    #[serde(default = "default_learning_timeout_secs")]
    pub learning_timeout_secs: u64,
    /// Context truncation limit (maximum context items to keep)
    #[serde(default = "default_context_truncation_limit")]
    pub context_truncation_limit: usize,
    /// Base retrieval top_k (minimum number of results to retrieve)
    #[serde(default = "default_base_retrieval_top_k")]
    pub base_retrieval_top_k: i32,
    /// Delay threshold in milliseconds (for various delay checks)
    #[serde(default = "default_delay_threshold_ms")]
    pub delay_threshold_ms: u64,
    /// Generation HTTP client timeout in seconds
    #[serde(default = "default_generation_client_timeout_secs")]
    pub generation_client_timeout_secs: u64,
    /// Memory upsert timeout in seconds
    #[serde(default = "default_memory_upsert_timeout_secs")]
    pub memory_upsert_timeout_secs: u64,
    /// ROUGE acceptable threshold (minimum ROUGE score for soft failure bypass)
    #[serde(default = "default_rouge_acceptable_threshold")]
    pub rouge_acceptable_threshold: f64,
    /// ROUGE improvement threshold for retry success (delta improvement)
    #[serde(default = "default_rouge_improvement_threshold")]
    pub rouge_improvement_threshold: f64,
    /// UCB1 score boost threshold (minimum score when ROUGE improves)
    #[serde(default = "default_ucb1_boost_threshold")]
    pub ucb1_boost_threshold: f64,
    /// UCB1 score relaxation threshold (after multiple retries)
    #[serde(default = "default_ucb1_relaxation_threshold")]
    pub ucb1_relaxation_threshold: f64,
    /// Retry count threshold for UCB1 relaxation
    #[serde(default = "default_retry_count_for_relaxation")]
    pub retry_count_for_relaxation: u32,
    /// Quality calculation base score
    #[serde(default = "default_quality_base_score")]
    pub quality_base_score: f32,
    /// Quality calculation maximum length for length factor
    #[serde(default = "default_quality_max_length")]
    pub quality_max_length: usize,
    /// Quality calculation length factor weight
    #[serde(default = "default_quality_length_factor_weight")]
    pub quality_length_factor_weight: f32,
    /// Quality calculation entropy threshold (below this gets bonus)
    #[serde(default = "default_quality_entropy_threshold")]
    pub quality_entropy_threshold: f64,
    /// Quality calculation entropy factor weight
    #[serde(default = "default_quality_entropy_factor_weight")]
    pub quality_entropy_factor_weight: f32,
    /// Knot complexity threshold for quality penalty
    #[serde(default = "default_knot_complexity_penalty_threshold")]
    pub knot_complexity_penalty_threshold: f64,
    /// Knot complexity quality penalty multiplier
    #[serde(default = "default_knot_complexity_penalty_multiplier")]
    pub knot_complexity_penalty_multiplier: f32,
    /// Spectral gap threshold for quality bonus
    #[serde(default = "default_spectral_gap_bonus_threshold")]
    pub spectral_gap_bonus_threshold: f64,
    /// Spectral gap quality bonus multiplier
    #[serde(default = "default_spectral_gap_bonus_multiplier")]
    pub spectral_gap_bonus_multiplier: f32,
    /// Betti-1 threshold for quality adjustment
    #[serde(default = "default_betti1_quality_threshold")]
    pub betti1_quality_threshold: usize,
    /// Betti-1 quality bonus multiplier (in Discover quadrant)
    #[serde(default = "default_betti1_bonus_multiplier")]
    pub betti1_bonus_multiplier: f32,
    /// Betti-1 quality penalty multiplier (in other quadrants)
    #[serde(default = "default_betti1_penalty_multiplier")]
    pub betti1_penalty_multiplier: f32,
    /// Persistence entropy threshold for quality bonus (low entropy = stable)
    #[serde(default = "default_persistence_entropy_quality_threshold")]
    pub persistence_entropy_quality_threshold: f64,
    /// Persistence entropy quality bonus multiplier
    #[serde(default = "default_persistence_entropy_bonus_multiplier")]
    pub persistence_entropy_bonus_multiplier: f32,
    /// Topology refinement knot complexity threshold
    #[serde(default = "default_topology_refinement_knot_threshold")]
    pub topology_refinement_knot_threshold: f64,
    /// Topology refinement Betti-1 threshold
    #[serde(default = "default_topology_refinement_betti1_threshold")]
    pub topology_refinement_betti1_threshold: usize,
    /// Topology refinement persistence entropy threshold
    #[serde(default = "default_topology_refinement_entropy_threshold")]
    pub topology_refinement_entropy_threshold: f64,
    /// Autonomous refinement temperature (low for stability)
    #[serde(default = "default_autonomous_refinement_temperature")]
    pub autonomous_refinement_temperature: f64,
    /// Autonomous refinement top_p
    #[serde(default = "default_autonomous_refinement_top_p")]
    pub autonomous_refinement_top_p: f64,
    /// Autonomous refinement improvement weight
    #[serde(default = "default_autonomous_refinement_improvement_weight")]
    pub autonomous_refinement_improvement_weight: f32,
    /// Autonomous refinement improvement threshold
    #[serde(default = "default_autonomous_refinement_improvement_threshold")]
    pub autonomous_refinement_improvement_threshold: f64,
    /// Second pass refinement improvement threshold
    #[serde(default = "default_second_pass_refinement_threshold")]
    pub second_pass_refinement_threshold: f64,
    /// Second pass refinement temperature
    #[serde(default = "default_second_pass_refinement_temperature")]
    pub second_pass_refinement_temperature: f64,
    /// Second pass refinement top_p
    #[serde(default = "default_second_pass_refinement_top_p")]
    pub second_pass_refinement_top_p: f64,
    /// Enhancement prompt temperature (for healing state)
    #[serde(default = "default_enhancement_temperature")]
    pub enhancement_temperature: f64,
    /// Enhancement prompt top_p (for healing state)
    #[serde(default = "default_enhancement_top_p")]
    pub enhancement_top_p: f64,
    /// Reward calculation ROUGE weight
    #[serde(default = "default_reward_rouge_weight")]
    pub reward_rouge_weight: f64,
    /// Reward calculation entropy weight
    #[serde(default = "default_reward_entropy_weight")]
    pub reward_entropy_weight: f64,
    /// Default curator quality for consistency voting
    #[serde(default = "default_consistency_voting_quality")]
    pub consistency_voting_quality: f64,
    /// Failure signal thresholds configuration
    #[serde(default)]
    pub failure_signal_thresholds: FailureSignalThresholds,
    /// RCE-ERAG cosine similarity weight (for ranking)
    #[serde(default = "default_rce_erag_cosine_weight")]
    pub rce_erag_cosine_weight: f64,
    /// RCE-ERAG entropy score weight (for ranking)
    #[serde(default = "default_rce_erag_entropy_weight")]
    pub rce_erag_entropy_weight: f64,
    /// RCE adaptation persistence entropy threshold
    #[serde(default = "default_rce_adaptation_entropy_threshold")]
    pub rce_adaptation_entropy_threshold: f64,
    /// RCE adaptation spectral gap threshold
    #[serde(default = "default_rce_adaptation_spectral_gap_threshold")]
    pub rce_adaptation_spectral_gap_threshold: f64,
    /// RCE circuit breaker streak threshold
    #[serde(default = "default_rce_circuit_breaker_streak")]
    pub rce_circuit_breaker_streak: u32,
    /// Tough knots query multiplier (fetch N * multiplier samples)
    #[serde(default = "default_tough_knots_multiplier")]
    pub tough_knots_multiplier: usize,
    /// Tough knots query maximum fetch size
    #[serde(default = "default_tough_knots_max_fetch")]
    pub tough_knots_max_fetch: usize,
    /// Tough knots knot complexity threshold
    #[serde(default = "default_tough_knots_knot_threshold")]
    pub tough_knots_knot_threshold: f64,
    /// Tough knots curator quality threshold
    #[serde(default = "default_tough_knots_quality_threshold")]
    pub tough_knots_quality_threshold: f64,
    /// Tough knots knot complexity multiplier (for scoring)
    #[serde(default = "default_tough_knots_knot_multiplier")]
    pub tough_knots_knot_multiplier: f64,
    /// Compass PAD adjustment - H1 persistence normalization divisor
    #[serde(default = "default_compass_h1_persistence_divisor")]
    pub compass_h1_persistence_divisor: f64,
    /// Compass PAD adjustment - H1 penalty scale factor
    #[serde(default = "default_compass_h1_penalty_scale")]
    pub compass_h1_penalty_scale: f64,
    /// Compass PAD adjustment - sheaf energy threshold for boost
    #[serde(default = "default_compass_sheaf_energy_threshold")]
    pub compass_sheaf_energy_threshold: f64,
    /// Compass PAD adjustment - sheaf boost multiplier
    #[serde(default = "default_compass_sheaf_boost_multiplier")]
    pub compass_sheaf_boost_multiplier: f64,
    /// Compass PAD adjustment - dominance penalty multiplier
    #[serde(default = "default_compass_dominance_penalty_multiplier")]
    pub compass_dominance_penalty_multiplier: f64,
    /// Compass PAD adjustment - dominance boost multiplier
    #[serde(default = "default_compass_dominance_boost_multiplier")]
    pub compass_dominance_boost_multiplier: f64,
    /// Compass PAD adjustment - arousal penalty multiplier
    #[serde(default = "default_compass_arousal_penalty_multiplier")]
    pub compass_arousal_penalty_multiplier: f64,
    /// Compass PAD adjustment - random noise range
    #[serde(default = "default_compass_random_noise_range")]
    pub compass_random_noise_range: f64,
    /// Compass PAD adjustment - pleasure boost probability
    #[serde(default = "default_compass_pleasure_boost_probability")]
    pub compass_pleasure_boost_probability: f64,
    /// Compass PAD adjustment - pleasure boost multiplier
    #[serde(default = "default_compass_pleasure_boost_multiplier")]
    pub compass_pleasure_boost_multiplier: f64,
    /// Compass threat detection - base threat arousal threshold
    #[serde(default = "default_compass_base_threat_arousal_threshold")]
    pub compass_base_threat_arousal_threshold: f64,
    /// Compass threat detection - variance spike multiplier
    #[serde(default = "default_compass_variance_spike_multiplier")]
    pub compass_variance_spike_multiplier: f64,
    /// Compass threat detection - random threat probability
    #[serde(default = "default_compass_random_threat_probability")]
    pub compass_random_threat_probability: f64,
    /// Compass threat detection - random threat arousal threshold
    #[serde(default = "default_compass_random_threat_arousal_threshold")]
    pub compass_random_threat_arousal_threshold: f64,
    /// Compass threat detection - random threat pleasure threshold
    #[serde(default = "default_compass_random_threat_pleasure_threshold")]
    pub compass_random_threat_pleasure_threshold: f64,
    /// Compass healing detection - pleasure threshold
    #[serde(default = "default_compass_healing_pleasure_threshold")]
    pub compass_healing_pleasure_threshold: f64,
    /// Compass healing detection - dominance threshold
    #[serde(default = "default_compass_healing_dominance_threshold")]
    pub compass_healing_dominance_threshold: f64,
    /// Compass quadrant thresholds - panic pleasure threshold
    #[serde(default = "default_compass_quadrant_panic_pleasure_threshold")]
    pub compass_quadrant_panic_pleasure_threshold: f64,
    /// Compass quadrant thresholds - panic arousal threshold
    #[serde(default = "default_compass_quadrant_panic_arousal_threshold")]
    pub compass_quadrant_panic_arousal_threshold: f64,
    /// Compass quadrant thresholds - persist arousal threshold
    #[serde(default = "default_compass_quadrant_persist_arousal_threshold")]
    pub compass_quadrant_persist_arousal_threshold: f64,
    /// Compass intrinsic reward - panic to discover base reward
    #[serde(default = "default_compass_reward_panic_to_discover")]
    pub compass_reward_panic_to_discover: f64,
    /// Compass intrinsic reward - panic to persist base reward
    #[serde(default = "default_compass_reward_panic_to_persist")]
    pub compass_reward_panic_to_persist: f64,
    /// Compass intrinsic reward - panic to master base reward
    #[serde(default = "default_compass_reward_panic_to_master")]
    pub compass_reward_panic_to_master: f64,
    /// Compass intrinsic reward - master to panic base reward
    #[serde(default = "default_compass_reward_master_to_panic")]
    pub compass_reward_master_to_panic: f64,
    /// Compass intrinsic reward - default base reward
    #[serde(default = "default_compass_reward_default")]
    pub compass_reward_default: f64,
    /// Compass intrinsic reward - entropy delta multiplier
    #[serde(default = "default_compass_reward_entropy_multiplier")]
    pub compass_reward_entropy_multiplier: f64,
    /// Compass MCTS branch - H1 bonus cap and multiplier
    #[serde(default = "default_compass_mcts_h1_bonus_cap")]
    pub compass_mcts_h1_bonus_cap: f64,
    #[serde(default = "default_compass_mcts_h1_bonus_multiplier")]
    pub compass_mcts_h1_bonus_multiplier: f64,
    /// Compass MCTS branch - persistence bonus divisor and multiplier
    #[serde(default = "default_compass_mcts_persistence_divisor")]
    pub compass_mcts_persistence_divisor: f64,
    #[serde(default = "default_compass_mcts_persistence_multiplier")]
    pub compass_mcts_persistence_multiplier: f64,
    /// Compass MCTS branch - knot bonus multiplier and cap
    #[serde(default = "default_compass_mcts_knot_multiplier")]
    pub compass_mcts_knot_multiplier: f64,
    #[serde(default = "default_compass_mcts_knot_multiplier_cap")]
    pub compass_mcts_knot_multiplier_cap: f64,
    #[serde(default = "default_compass_mcts_knot_weight")]
    pub compass_mcts_knot_weight: f64,
    /// Compass MCTS branch - gap bonus multiplier
    #[serde(default = "default_compass_mcts_gap_multiplier")]
    pub compass_mcts_gap_multiplier: f64,
    /// Compass MCTS branch - entropy bonus multiplier and cap
    #[serde(default = "default_compass_mcts_entropy_multiplier")]
    pub compass_mcts_entropy_multiplier: f64,
    #[serde(default = "default_compass_mcts_entropy_multiplier_cap")]
    pub compass_mcts_entropy_multiplier_cap: f64,
    #[serde(default = "default_compass_mcts_entropy_weight")]
    pub compass_mcts_entropy_weight: f64,
    /// Compass MCTS branch - H0 bonus cap and multiplier
    #[serde(default = "default_compass_mcts_h0_bonus_cap")]
    pub compass_mcts_h0_bonus_cap: f64,
    #[serde(default = "default_compass_mcts_h0_bonus_multiplier")]
    pub compass_mcts_h0_bonus_multiplier: f64,
    /// Compass MCTS branch - default exploration bonus base and divisor
    #[serde(default = "default_compass_mcts_default_exploration_base")]
    pub compass_mcts_default_exploration_base: f64,
    #[serde(default = "default_compass_mcts_default_exploration_divisor")]
    pub compass_mcts_default_exploration_divisor: f64,
    /// Compass cascade - minimum consonance threshold
    #[serde(default = "default_compass_cascade_min_consonance")]
    pub compass_cascade_min_consonance: f64,
    /// Compass cascade - recognition to satisfaction consonance threshold
    #[serde(default = "default_compass_cascade_recognition_satisfaction_consonance")]
    pub compass_cascade_recognition_satisfaction_consonance: f64,
    /// Compass cascade - calm to motivation consonance threshold
    #[serde(default = "default_compass_cascade_calm_motivation_consonance")]
    pub compass_cascade_calm_motivation_consonance: f64,
    /// Learning loop - executor memory limit
    #[serde(default = "default_learning_executor_memory_limit")]
    pub learning_executor_memory_limit: usize,
    /// Learning loop - executor cluster threshold
    #[serde(default = "default_learning_executor_cluster_threshold")]
    pub learning_executor_cluster_threshold: f32,
    /// Learning loop - reward threshold for QLoRA trigger
    #[serde(default = "default_learning_reward_threshold")]
    pub learning_reward_threshold: f64,
    /// Learning loop - reptile episode interval
    #[serde(default = "default_learning_reptile_episode_interval")]
    pub learning_reptile_episode_interval: u32,
    /// Learning loop - evolution episode interval
    #[serde(default = "default_learning_evolution_episode_interval")]
    pub learning_evolution_episode_interval: u32,
    /// Learning loop - reptile batch size
    #[serde(default = "default_learning_reptile_batch_size")]
    pub learning_reptile_batch_size: usize,
    /// Learning loop - QLoRA low reward threshold
    #[serde(default = "default_learning_qlora_low_reward_threshold")]
    pub learning_qlora_low_reward_threshold: f64,
    /// Learning loop - QLoRA sample count
    #[serde(default = "default_learning_qlora_sample_count")]
    pub learning_qlora_sample_count: usize,
    /// Learning loop - QLoRA max samples
    #[serde(default = "default_learning_qlora_max_samples")]
    pub learning_qlora_max_samples: usize,
    /// Learning loop - epsilon decay rate
    #[serde(default = "default_learning_epsilon_decay_rate")]
    pub learning_epsilon_decay_rate: f64,
    /// Learning loop - epsilon minimum
    #[serde(default = "default_learning_epsilon_minimum")]
    pub learning_epsilon_minimum: f64,
    /// Learning loop - alpha decay rate
    #[serde(default = "default_learning_alpha_decay_rate")]
    pub learning_alpha_decay_rate: f64,
    /// Learning loop - alpha minimum
    #[serde(default = "default_learning_alpha_minimum")]
    pub learning_alpha_minimum: f64,
    /// Learning loop - evolution old episodes ratio
    #[serde(default = "default_learning_evolution_old_episodes_ratio")]
    pub learning_evolution_old_episodes_ratio: f64,
    /// Learning loop - evolution old episodes min
    #[serde(default = "default_learning_evolution_old_episodes_min")]
    pub learning_evolution_old_episodes_min: usize,
    /// Learning loop - evolution old episodes max
    #[serde(default = "default_learning_evolution_old_episodes_max")]
    pub learning_evolution_old_episodes_max: usize,
    /// Learning loop - tough knots ratio
    #[serde(default = "default_learning_tough_knots_ratio")]
    pub learning_tough_knots_ratio: f64,
    /// Learning loop - TCS reward shaping knot penalty
    #[serde(default = "default_learning_tcs_knot_penalty")]
    pub learning_tcs_knot_penalty: f64,
    /// Learning loop - TCS reward shaping Betti1 penalty
    #[serde(default = "default_learning_tcs_betti1_penalty")]
    pub learning_tcs_betti1_penalty: f64,
    /// Learning loop - TCS reward shaping entropy penalty
    #[serde(default = "default_learning_tcs_entropy_penalty")]
    pub learning_tcs_entropy_penalty: f64,
    /// Learning loop - TCS reward shaping discover mode weight
    #[serde(default = "default_learning_tcs_discover_weight")]
    pub learning_tcs_discover_weight: f64,
    /// Learning loop - TCS reward shaping spectral gap threshold
    #[serde(default = "default_learning_tcs_spectral_gap_threshold")]
    pub learning_tcs_spectral_gap_threshold: f64,
    /// Learning loop - TCS reward shaping convergence bonus
    #[serde(default = "default_learning_tcs_convergence_bonus")]
    pub learning_tcs_convergence_bonus: f64,
    /// Learning loop - TCS reward shaping convergence penalty
    #[serde(default = "default_learning_tcs_convergence_penalty")]
    pub learning_tcs_convergence_penalty: f64,
    /// Learning loop - TCS reward shaping novelty threshold
    #[serde(default = "default_learning_tcs_novelty_threshold")]
    pub learning_tcs_novelty_threshold: f64,
    /// Learning loop - TCS reward shaping novelty bonus
    #[serde(default = "default_learning_tcs_novelty_bonus")]
    pub learning_tcs_novelty_bonus: f64,
    /// Learning loop - DQN batch size
    #[serde(default = "default_learning_dqn_batch_size")]
    pub learning_dqn_batch_size: usize,
    /// Learning loop - DQN parameter adjustment multipliers
    #[serde(default = "default_learning_dqn_temp_multiplier")]
    pub learning_dqn_temp_multiplier: f64,
    #[serde(default = "default_learning_dqn_top_p_multiplier")]
    pub learning_dqn_top_p_multiplier: f64,
    #[serde(default = "default_learning_dqn_mcts_c_multiplier")]
    pub learning_dqn_mcts_c_multiplier: f64,
    #[serde(default = "default_learning_dqn_retrieval_multiplier")]
    pub learning_dqn_retrieval_multiplier: f64,
    #[serde(default = "default_learning_dqn_novelty_multiplier")]
    pub learning_dqn_novelty_multiplier: f64,
    #[serde(default = "default_learning_dqn_awareness_multiplier")]
    pub learning_dqn_awareness_multiplier: f64,
    /// Learning loop - Reptile inner gradient multiplier
    #[serde(default = "default_learning_reptile_inner_gradient_multiplier")]
    pub learning_reptile_inner_gradient_multiplier: f64,
    /// Learning loop - Evolution fitness multipliers
    #[serde(default = "default_learning_evolution_temp_multiplier")]
    pub learning_evolution_temp_multiplier: f64,
    #[serde(default = "default_learning_evolution_alpha_multiplier")]
    pub learning_evolution_alpha_multiplier: f64,
    /// Learning loop - Evolution mutation std multipliers
    #[serde(default = "default_learning_evolution_mutation_reduce_multiplier")]
    pub learning_evolution_mutation_reduce_multiplier: f64,
    #[serde(default = "default_learning_evolution_mutation_increase_multiplier")]
    pub learning_evolution_mutation_increase_multiplier: f64,
    /// Generation - reflexion temperature base multiplier
    #[serde(default = "default_generation_reflexion_temp_base_multiplier")]
    pub generation_reflexion_temp_base_multiplier: f64,
    /// Generation - reflexion temperature stability multiplier
    #[serde(default = "default_generation_reflexion_temp_stability_multiplier")]
    pub generation_reflexion_temp_stability_multiplier: f64,
    /// Generation - reflexion top_p increment
    #[serde(default = "default_generation_reflexion_top_p_increment")]
    pub generation_reflexion_top_p_increment: f64,
    /// Generation - reflexion top_p stability increment
    #[serde(default = "default_generation_reflexion_top_p_stability_increment")]
    pub generation_reflexion_top_p_stability_increment: f64,
    /// Generation - reflexion top_p maximum
    #[serde(default = "default_generation_reflexion_top_p_max")]
    pub generation_reflexion_top_p_max: f64,
    /// Generation - CoT repair temperature base multiplier
    #[serde(default = "default_generation_cot_repair_temp_base_multiplier")]
    pub generation_cot_repair_temp_base_multiplier: f64,
    /// Generation - CoT repair temperature iteration increment
    #[serde(default = "default_generation_cot_repair_temp_iteration_increment")]
    pub generation_cot_repair_temp_iteration_increment: f64,
    /// Generation - CoT repair top_p increment
    #[serde(default = "default_generation_cot_repair_top_p_increment")]
    pub generation_cot_repair_top_p_increment: f64,
    /// Generation - CoT repair top_p maximum
    #[serde(default = "default_generation_cot_repair_top_p_max")]
    pub generation_cot_repair_top_p_max: f64,
    /// Generation - CoT repair temperature min/max
    #[serde(default = "default_generation_cot_repair_temp_min")]
    pub generation_cot_repair_temp_min: f64,
    #[serde(default = "default_generation_cot_repair_temp_max")]
    pub generation_cot_repair_temp_max: f64,
    /// ERAG - similarity boost multiplier
    #[serde(default = "default_erag_similarity_boost_multiplier")]
    pub erag_similarity_boost_multiplier: f64,
    /// ERAG - similarity boost maximum
    #[serde(default = "default_erag_similarity_boost_max")]
    pub erag_similarity_boost_max: f64,
}

/// Weighted Episodic Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedMemoryConfig {
    /// Fitness weights [temporal, pad, beta1, retrieval, consonance, resource_penalty]
    #[serde(default = "default_fitness_weights")]
    pub fitness_weights: [f32; 6],
    /// Enable weight evolution
    #[serde(default = "default_weight_evolution_enabled")]
    pub weight_evolution_enabled: bool,
    /// Minimum discoveries needed to trigger weight update
    #[serde(default = "default_weight_evolution_update_threshold")]
    pub weight_evolution_update_threshold: usize,
    /// Enable MCTS daydreaming
    #[serde(default = "default_daydreaming_enabled")]
    pub daydreaming_enabled: bool,
    /// Daydreaming duration in seconds
    #[serde(default = "default_daydreaming_duration_seconds")]
    pub daydreaming_duration_seconds: u64,
    /// Topology update interval in seconds
    #[serde(default = "default_topology_update_interval_seconds")]
    pub topology_update_interval_seconds: u64,
    /// Enable memory consolidation
    #[serde(default = "default_consolidation_enabled")]
    pub consolidation_enabled: bool,
    /// GPU device preference ("cuda", "cpu", "auto")
    #[serde(default = "default_gpu_device")]
    pub gpu_device: String,
}

fn default_fitness_weights() -> [f32; 6] {
    [0.20, 0.18, 0.18, 0.13, 0.18, 0.13] // temporal, pad, beta1, retrieval, consonance, resource_penalty
}

fn default_weight_evolution_enabled() -> bool {
    true
}

fn default_weight_evolution_update_threshold() -> usize {
    10
}

fn default_daydreaming_enabled() -> bool {
    true
}

fn default_daydreaming_duration_seconds() -> u64 {
    60
}

fn default_topology_update_interval_seconds() -> u64 {
    3600 // 1 hour
}

fn default_consolidation_enabled() -> bool {
    true
}

fn default_gpu_device() -> String {
    "cpu".to_string()
}

/// Resource budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudgetConfig {
    /// Maximum token budget
    #[serde(default = "default_tokens_max")]
    pub tokens_max: u64,
    /// Maximum API rate limit per window
    #[serde(default = "default_api_rate_limit_max")]
    pub api_rate_limit_max: u64,
    /// Maximum compute cycles (for normalization)
    #[serde(default = "default_compute_cycles_max")]
    pub compute_cycles_max: u64,
    /// Maximum memory bandwidth (for normalization)
    #[serde(default = "default_memory_bandwidth_max")]
    pub memory_bandwidth_max: u64,
}

fn default_tokens_max() -> u64 {
    100_000
}

fn default_api_rate_limit_max() -> u64 {
    100
}

fn default_compute_cycles_max() -> u64 {
    1_000_000
}

fn default_memory_bandwidth_max() -> u64 {
    100_000
}

impl Default for ResourceBudgetConfig {
    fn default() -> Self {
        Self {
            tokens_max: default_tokens_max(),
            api_rate_limit_max: default_api_rate_limit_max(),
            compute_cycles_max: default_compute_cycles_max(),
            memory_bandwidth_max: default_memory_bandwidth_max(),
        }
    }
}

/// Graceful degradation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationConfig {
    /// Tier 1 threshold (70-100% resources)
    #[serde(default = "default_tier1_threshold")]
    pub tier1_threshold: f32,
    /// Tier 2 threshold (50-70% resources)
    #[serde(default = "default_tier2_threshold")]
    pub tier2_threshold: f32,
    /// Tier 3 threshold (30-50% resources)
    #[serde(default = "default_tier3_threshold")]
    pub tier3_threshold: f32,
    /// Tier 4 threshold (0-30% resources)
    #[serde(default = "default_tier4_threshold")]
    pub tier4_threshold: f32,
    /// Resource penalty multiplier for tier 1
    #[serde(default = "default_tier1_multiplier")]
    pub tier1_multiplier: f32,
    /// Resource penalty multiplier for tier 2
    #[serde(default = "default_tier2_multiplier")]
    pub tier2_multiplier: f32,
    /// Resource penalty multiplier for tier 3
    #[serde(default = "default_tier3_multiplier")]
    pub tier3_multiplier: f32,
    /// Resource penalty multiplier for tier 4
    #[serde(default = "default_tier4_multiplier")]
    pub tier4_multiplier: f32,
}

fn default_tier1_threshold() -> f32 {
    0.70
}

fn default_tier2_threshold() -> f32 {
    0.50
}

fn default_tier3_threshold() -> f32 {
    0.30
}

fn default_tier4_threshold() -> f32 {
    0.0
}

fn default_tier1_multiplier() -> f32 {
    1.2
}

fn default_tier2_multiplier() -> f32 {
    2.0
}

fn default_tier3_multiplier() -> f32 {
    5.0
}

fn default_tier4_multiplier() -> f32 {
    10.0
}

impl Default for DegradationConfig {
    fn default() -> Self {
        Self {
            tier1_threshold: default_tier1_threshold(),
            tier2_threshold: default_tier2_threshold(),
            tier3_threshold: default_tier3_threshold(),
            tier4_threshold: default_tier4_threshold(),
            tier1_multiplier: default_tier1_multiplier(),
            tier2_multiplier: default_tier2_multiplier(),
            tier3_multiplier: default_tier3_multiplier(),
            tier4_multiplier: default_tier4_multiplier(),
        }
    }
}

/// Temporal TDA configuration for failure chain detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalTDAConfig {
    /// Maximum history window size for topological snapshots
    #[serde(default = "default_temporal_tda_window_size")]
    pub window_size: usize,
    /// Wasserstein distance threshold for detecting transitions
    #[serde(default = "default_temporal_tda_wasserstein_threshold")]
    pub wasserstein_threshold: f32,
    /// Minimum severity score to trigger failure detection
    #[serde(default = "default_temporal_tda_severity_threshold")]
    pub severity_threshold: f32,
    /// Maximum number of failure chains to track
    #[serde(default = "default_temporal_tda_max_chains")]
    pub max_chains: usize,
    /// Enable temporal TDA detection
    #[serde(default = "default_temporal_tda_enabled")]
    pub enabled: bool,
}

fn default_temporal_tda_window_size() -> usize {
    20
}

fn default_temporal_tda_wasserstein_threshold() -> f32 {
    0.5
}

fn default_temporal_tda_severity_threshold() -> f32 {
    5.0
}

fn default_temporal_tda_max_chains() -> usize {
    10
}

fn default_temporal_tda_enabled() -> bool {
    true
}

impl Default for TemporalTDAConfig {
    fn default() -> Self {
        Self {
            window_size: default_temporal_tda_window_size(),
            wasserstein_threshold: default_temporal_tda_wasserstein_threshold(),
            severity_threshold: default_temporal_tda_severity_threshold(),
            max_chains: default_temporal_tda_max_chains(),
            enabled: default_temporal_tda_enabled(),
        }
    }
}

impl Default for WeightedMemoryConfig {
    fn default() -> Self {
        Self {
            fitness_weights: default_fitness_weights(),
            weight_evolution_enabled: true,
            weight_evolution_update_threshold: 10,
            daydreaming_enabled: true,
            daydreaming_duration_seconds: 60,
            topology_update_interval_seconds: 3600,
            consolidation_enabled: true,
            gpu_device: "cpu".to_string(),
        }
    }
}

impl RuntimeConfig {
    pub fn load(args: &CliArgs) -> Result<Self> {
        prime_environment();

        if let Some(ref config_path) = args.config {
            let file = std::fs::read_to_string(config_path)
                .with_context(|| format!("unable to read config file {config_path}"))?;
            let mut cfg: RuntimeConfig = serde_yaml::from_str(&file)
                .with_context(|| format!("invalid YAML in {config_path}"))?;
            cfg.security.finalize(cfg.prompt_max_chars);
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
        .unwrap_or_else(|| "/workspace/models/hf_cache/models--Qwen--Qwen2.5-7B-Instruct-AWQ".to_string());

        info!("RuntimeConfig: vllm_model={}", vllm_model);  // ADD THIS

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

        let qdrant_embedded = env_with_fallback(&["QDRANT_EMBEDDED"])
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);

        let qdrant_collection = env_with_fallback(&["QDRANT_COLLECTION", "QDRANT_COLLECTION_NAME"])
            .unwrap_or_else(|| "experiences".to_string());

        let requested_qdrant_dim = env_with_fallback(&["QDRANT_VECTOR_DIM", "QDRANT_VECTOR_SIZE"])
            .and_then(|value| value.parse::<usize>().ok());
        
        // Use requested dimension or default to 896
        // Note: The embedding model determines the actual vector dimension
        // This allows override but warns if mismatch with expected value
        let qdrant_vector_dim = requested_qdrant_dim.unwrap_or(896);
        
        if qdrant_vector_dim != 896 {
            warn!(
                requested = qdrant_vector_dim,
                expected = 896,
                "Qdrant vector dimension ({}) differs from expected (896). Ensure embedding model matches.",
                qdrant_vector_dim
            );
        }

        let ollama_endpoint =
            env_with_fallback(&["OLLAMA_URL", "OLLAMA_ENDPOINT", "OLLAMA_ENDPOINT_TAILSCALE"])
                .or_else(|| {
                    warn!("Set OLLAMA_URL and run 'ollama serve && ollama pull qwen2:0.5b'");
                    None
                })
                .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());

        let embedding_model_name = env_with_fallback(&[
            "EMBEDDING_MODEL_NAME",
            "EMBEDDING_MODEL",
            "ONNX_EMBED_MODEL_PATH",
        ]) // Prioritize embedding-specific vars, exclude CURATOR_MODEL
        .unwrap_or_else(|| "/workspace/models/Qwen2.5-0.5B-Instruct/onnx/model_fp16.onnx".to_string());

        let embed_with_candle = env_with_fallback(&["EMBED_WITH_CANDLE"])
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);

        let embed_model_dir = env_with_fallback(&["EMBED_MODEL_DIR"])
            .or_else(|| Some("./models/bge-small-en".to_string()));

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

        let topology_mode = TopologyMode::from_env();

        let generation_backend = BackendType::from_env();

        let enable_curator = env_with_fallback(&["ENABLE_CURATOR"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(false); // Default to autonomous mode unless explicitly enabled

        let curator_model_name = env_with_fallback(&["CURATOR_MODEL", "CURATOR_MODEL_NAME"])
            .unwrap_or_else(|| "qwen2:0.5b".to_string()); // Keep for Ollama curator

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

        let curator_autonomous = env_with_fallback(&["CURATOR_AUTONOMOUS"])
            .map(|value| {
                matches!(
                    value.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on" | "auto"
                )
            })
            .unwrap_or(default_curator_autonomous());

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
        let security_rate_limit_window_secs = env_with_fallback(&[
            "SECURITY_PROMPT_RATE_WINDOW_SECS",
            "PIPELINE_RATE_LIMIT_WINDOW_SECS",
        ])
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or_else(default_security_rate_limit_window_secs);
        let security_rate_limit_max_requests = env_with_fallback(&[
            "SECURITY_PROMPT_RATE_LIMIT",
            "SECURITY_RATE_LIMIT_MAX_REQUESTS",
            "PIPELINE_RATE_LIMIT_MAX_REQUESTS",
        ])
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or_else(default_security_rate_limit_max_requests);
        let security_allow_control_chars = env_with_fallback(&["SECURITY_ALLOW_CONTROL_CHARS"])
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or_else(default_security_allow_control_chars);
        let mut security_banned_patterns = env_with_fallback(&["SECURITY_BANNED_PATTERNS"])
            .map(|raw| SecurityConfig::parse_patterns(&raw))
            .unwrap_or_else(default_security_banned_patterns);
        if security_banned_patterns.is_empty() {
            security_banned_patterns = default_security_banned_patterns();
        }
        let security_audit_log_path = env_with_fallback(&["SECURITY_AUDIT_LOG_PATH"])
            .unwrap_or_else(default_security_audit_log_path);
        let mut security = SecurityConfig {
            rate_limit_window_secs: security_rate_limit_window_secs,
            rate_limit_max_requests: security_rate_limit_max_requests,
            allow_control_chars: security_allow_control_chars,
            banned_patterns: security_banned_patterns,
            audit_log_path: security_audit_log_path,
            prompt_max_chars,
        };
        security.finalize(prompt_max_chars);
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
            .unwrap_or_else(default_lens_snippet_chars);
        
        // Validate lens_snippet_chars is within reasonable bounds
        // Allow override via environment variable but warn if outside expected range
        if lens_snippet_chars < 50 || lens_snippet_chars > 1000 {
            warn!(
                value = lens_snippet_chars,
                "LENS_SNIPPET_CHARS ({}) outside typical range (50-1000). This may impact performance.",
                lens_snippet_chars
            );
        }
        let cot_temp_increment = env_with_fallback(&["COT_TEMP_INCREMENT"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_cot_temp_increment);
        let reflexion_top_p_step = env_with_fallback(&["REFLEXION_TOP_P_STEP"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_reflexion_top_p_step);
        let cot_success_rouge_threshold = env_with_fallback(&["COT_SUCCESS_ROUGE_THRESHOLD"])
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or_else(default_cot_success_rouge_threshold);
        let breakthrough_rouge_min = env_with_fallback(&["BREAKTHROUGH_ROUGE_MIN"])
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or_else(default_breakthrough_rouge_min);

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
        let cache_compression_min_bytes = env_with_fallback(&["CACHE_COMPRESSION_MIN_BYTES"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_cache_compression_min_bytes);
        let cache_prefetch_enabled = env_with_fallback(&["CACHE_PREFETCH_ENABLED"])
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or_else(default_cache_prefetch_enabled);
        let cache_prefetch_prompts = env_with_fallback(&["CACHE_PREFETCH_PROMPTS"])
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(default_cache_prefetch_prompts);
        let cache_prefetch_top_hits = env_with_fallback(&["CACHE_PREFETCH_TOP_HITS"])
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(default_cache_prefetch_top_hits);
        let cache_prefetch_parallelism = env_with_fallback(&["CACHE_PREFETCH_PARALLELISM"])
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v.clamp(1, 16))
            .unwrap_or_else(default_cache_prefetch_parallelism);
        let retry_backoff_exponent_cap = env_with_fallback(&["RETRY_BACKOFF_EXPONENT_CAP"])
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(default_retry_backoff_exponent_cap);

        let phase2_max_retries = env_with_fallback(&["PHASE2_MAX_RETRIES"])
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(default_max_retries());
        let phase2_retry_base_delay_ms = env_with_fallback(&["PHASE2_RETRY_BASE_DELAY_MS"])
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(default_retry_base_delay_ms());
        let phase2_cot_iterations = env_with_fallback(&["PHASE2_COT_ITERATIONS"])
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(default_phase2_cot_iterations());
        let phase2_retry_backoff_cap_ms = env_with_fallback(&["PHASE2_RETRY_BACKOFF_CAP_MS"])
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(default_phase2_retry_backoff_cap_ms());

        let similarity_threshold = env_with_fallback(&["SIMILARITY_THRESHOLD"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(default_similarity_threshold());

        let phase2_level3_retry_count = env_with_fallback(&["PHASE2_LEVEL3_RETRY_COUNT"])
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(default_level3_retry_count());

        let phase2_mcts_c_increment = env_with_fallback(&["PHASE2_MCTS_C_INCREMENT"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(default_mcts_c_increment());

        let phase2_top_p_increment = env_with_fallback(&["PHASE2_TOP_P_INCREMENT"])
            .and_then(|value| value.parse().ok())
            .unwrap_or(default_top_p_increment());

        let phase2_retrieval_top_k_increment =
            env_with_fallback(&["PHASE2_RETRIEVAL_TOP_K_INCREMENT"])
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or(default_retrieval_top_k_increment());

        let dqn_actions = env_with_fallback(&["DQN_ACTIONS"])
            .as_deref()
            .and_then(|s| serde_yaml::from_str(s).ok())
            .unwrap_or_else(default_dqn_actions);

        let mut runtime = Self {
            vllm_endpoint,
            vllm_model,
            qdrant_url,
            qdrant_collection,
            qdrant_vector_dim,
            qdrant_embedded,
            ollama_endpoint,
            embedding_model_name,
            embed_with_candle,
            embed_model_dir,
            embedding_max_chars,
            training_data_path,
            emotional_seed_path,
            rut_gauntlet_path,
            entropy_cycles_for_baseline,
            enable_consistency_voting,
            mock_mode,
            topology_mode,
            rce_enabled: env_with_fallback(&["RCE_ENABLED"])
                .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
                .unwrap_or_else(default_rce_enabled),
            rce_shadow_mode: default_rce_shadow_mode(),
            rce_actions_enabled: default_rce_actions_enabled(),
            rce_window_seconds: default_rce_window_seconds(),
            rce_stride_seconds: default_rce_stride_seconds(),
            rce_beta_meta_weights: default_rce_beta_meta_weights(),
            rce_breakthrough_threshold: default_rce_breakthrough_threshold(),
            rce_consensus: RceConsensusConfig::default(),
            rce_erag_lambda: default_rce_erag_lambda(),
            rce_archive_backend: default_rce_archive_backend(),
            telemetry_enabled: default_telemetry_enabled(),
            telemetry_port: default_telemetry_port(),
            phase2_max_retries,
            phase2_retry_base_delay_ms,
            phase2_cot_iterations,
            phase2_retry_backoff_cap_ms,
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
            curator_autonomous,
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
            breakthrough_rouge_min,
            dqn_actions,
            temperature: 0.7,
            top_p: 0.9,
            novelty_threshold: env_with_fallback(&["NOVELTY_THRESHOLD"]).and_then(|v| v.parse().ok()).unwrap_or(0.5),
            self_awareness_level: env_with_fallback(&["SELF_AWARENESS_LEVEL"]).and_then(|v| v.parse().ok()).unwrap_or(0.3),
            prompt_max_chars,
            tokenizer_json: default_tokenizer_json(),
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
            cache_compression_min_bytes,
            cache_prefetch_enabled,
            cache_prefetch_prompts,
            cache_prefetch_top_hits,
            cache_prefetch_parallelism,
            retry_backoff_exponent_cap,
            security,
            weighted_memory_config: WeightedMemoryConfig::default(),
            disable_memory_store: env_with_fallback(&["DISABLE_MEMORY_STORE"])
                .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
                .unwrap_or(false),
            resource_budget_config: ResourceBudgetConfig::default(),
            degradation_config: DegradationConfig::default(),
            temporal_tda_config: TemporalTDAConfig::default(),
            // Phase 1-6: Back-half pipeline optimizations
            optimized_erag: env_with_fallback(&["OPTIMIZED_ERAG"])
                .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
                .unwrap_or(false),
            erag_batch_size: default_erag_batch_size(),
            erag_batch_flush_ms: default_erag_batch_flush_ms(),
            qdrant_quantization: env_with_fallback(&["QDRANT_QUANTIZATION"])
                .and_then(|v| match v.to_ascii_lowercase().as_str() {
                    "scalar_pq4" | "pq4" => Some(QuantizationType::ScalarPQ4),
                    "none" | "" => None,
                    _ => None,
                }),
            use_approximate_tda: env_with_fallback(&["USE_APPROXIMATE_TDA"])
                .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
                .unwrap_or(false),
            fp16_qlora_adapters: default_fp16_qlora_adapters(),
            parallel_curator_rouge: default_parallel_curator_rouge(),
            training_service_enabled: default_training_service_enabled(),
            training_service_url: default_training_service_url(),
            training_service_use_grpc: default_training_service_use_grpc(),
            adapter_storage_path: default_adapter_storage_path(),
            training_queue_path: default_training_queue_path(),
            use_gpu_fitness: env_with_fallback(&["USE_GPU_FITNESS"])
                .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
                .unwrap_or(false),
            // Ablation testing flags
            erag_bypass: env_with_fallback(&["ERAG_BYPASS"])
                .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
                .unwrap_or(false),
            n_tokens_bypass: env_with_fallback(&["N_TOKENS_BYPASS"])
                .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
                .unwrap_or(false),
            // Pipeline runtime configuration
            curator_feedback_window_size: default_curator_feedback_window_size(),
            curator_feedback_threshold_adjustment: default_curator_feedback_threshold_adjustment(),
            curator_feedback_threshold_min: default_curator_feedback_threshold_min(),
            curator_feedback_threshold_max: default_curator_feedback_threshold_max(),
            curator_feedback_quality_trend_threshold: default_curator_feedback_quality_trend_threshold(),
            curator_feedback_temp_adjustment_multiplier: default_curator_feedback_temp_adjustment_multiplier(),
            curator_feedback_learned_rate_low: default_curator_feedback_learned_rate_low(),
            curator_feedback_quality_low: default_curator_feedback_quality_low(),
            curator_feedback_top_p_increase: default_curator_feedback_top_p_increase(),
            curator_feedback_learned_rate_high: default_curator_feedback_learned_rate_high(),
            curator_feedback_quality_high: default_curator_feedback_quality_high(),
            curator_feedback_top_p_decrease: default_curator_feedback_top_p_decrease(),
            curator_feedback_retrieval_quality_threshold: default_curator_feedback_retrieval_quality_threshold(),
            curator_feedback_retrieval_top_k_increase: default_curator_feedback_retrieval_top_k_increase(),
            curator_feedback_retrieval_quality_high: default_curator_feedback_retrieval_quality_high(),
            curator_feedback_retrieval_learned_rate_high: default_curator_feedback_retrieval_learned_rate_high(),
            curator_feedback_retrieval_top_k_decrease: default_curator_feedback_retrieval_top_k_decrease(),
            pipeline_retrieval_top_k_min: default_pipeline_retrieval_top_k_min(),
            pipeline_retrieval_top_k_max: default_pipeline_retrieval_top_k_max(),
            pipeline_timing_split_ratio: default_pipeline_timing_split_ratio(),
            pipeline_healing_knot_threshold: default_pipeline_healing_knot_threshold(),
            pipeline_healing_spectral_gap_threshold: default_pipeline_healing_spectral_gap_threshold(),
            pipeline_ucb1_max_clamp: default_pipeline_ucb1_max_clamp(),
            pipeline_quality_score_increment: default_pipeline_quality_score_increment(),
            pipeline_param_min: default_pipeline_param_min(),
            pipeline_param_max: default_pipeline_param_max(),
            pipeline_retrieval_top_k_increment_min: default_pipeline_retrieval_top_k_increment_min(),
            pipeline_retrieval_top_k_increment_max: default_pipeline_retrieval_top_k_increment_max(),
            topology_memory_analyzer_threshold: default_topology_memory_analyzer_threshold(),
            discovery_buffer_interval_secs: default_discovery_buffer_interval_secs(),
            embedding_cache_capacity: default_embedding_cache_capacity(),
            collapse_cache_capacity: default_collapse_cache_capacity(),
            mcts_exploration_constant: default_mcts_exploration_constant(),
            mcts_depth: default_mcts_depth(),
            discovery_buffer_threshold: default_discovery_buffer_threshold(),
            gpu_fitness_refresh_interval_secs: default_gpu_fitness_refresh_interval_secs(),
            learning_timeout_secs: default_learning_timeout_secs(),
            context_truncation_limit: default_context_truncation_limit(),
            base_retrieval_top_k: default_base_retrieval_top_k(),
            delay_threshold_ms: default_delay_threshold_ms(),
            generation_client_timeout_secs: default_generation_client_timeout_secs(),
            memory_upsert_timeout_secs: default_memory_upsert_timeout_secs(),
            rouge_acceptable_threshold: default_rouge_acceptable_threshold(),
            rouge_improvement_threshold: default_rouge_improvement_threshold(),
            ucb1_boost_threshold: default_ucb1_boost_threshold(),
            ucb1_relaxation_threshold: default_ucb1_relaxation_threshold(),
            retry_count_for_relaxation: default_retry_count_for_relaxation(),
            quality_base_score: default_quality_base_score(),
            quality_max_length: default_quality_max_length(),
            quality_length_factor_weight: default_quality_length_factor_weight(),
            quality_entropy_threshold: default_quality_entropy_threshold(),
            quality_entropy_factor_weight: default_quality_entropy_factor_weight(),
            knot_complexity_penalty_threshold: default_knot_complexity_penalty_threshold(),
            knot_complexity_penalty_multiplier: default_knot_complexity_penalty_multiplier(),
            spectral_gap_bonus_threshold: default_spectral_gap_bonus_threshold(),
            spectral_gap_bonus_multiplier: default_spectral_gap_bonus_multiplier(),
            betti1_quality_threshold: default_betti1_quality_threshold(),
            betti1_bonus_multiplier: default_betti1_bonus_multiplier(),
            betti1_penalty_multiplier: default_betti1_penalty_multiplier(),
            persistence_entropy_quality_threshold: default_persistence_entropy_quality_threshold(),
            persistence_entropy_bonus_multiplier: default_persistence_entropy_bonus_multiplier(),
            topology_refinement_knot_threshold: default_topology_refinement_knot_threshold(),
            topology_refinement_betti1_threshold: default_topology_refinement_betti1_threshold(),
            topology_refinement_entropy_threshold: default_topology_refinement_entropy_threshold(),
            autonomous_refinement_temperature: default_autonomous_refinement_temperature(),
            autonomous_refinement_top_p: default_autonomous_refinement_top_p(),
            autonomous_refinement_improvement_weight: default_autonomous_refinement_improvement_weight(),
            autonomous_refinement_improvement_threshold: default_autonomous_refinement_improvement_threshold(),
            second_pass_refinement_threshold: default_second_pass_refinement_threshold(),
            second_pass_refinement_temperature: default_second_pass_refinement_temperature(),
            second_pass_refinement_top_p: default_second_pass_refinement_top_p(),
            enhancement_temperature: default_enhancement_temperature(),
            enhancement_top_p: default_enhancement_top_p(),
            reward_rouge_weight: default_reward_rouge_weight(),
            reward_entropy_weight: default_reward_entropy_weight(),
            consistency_voting_quality: default_consistency_voting_quality(),
            failure_signal_thresholds: FailureSignalThresholds::default(),
            rce_erag_cosine_weight: default_rce_erag_cosine_weight(),
            rce_erag_entropy_weight: default_rce_erag_entropy_weight(),
            rce_adaptation_entropy_threshold: default_rce_adaptation_entropy_threshold(),
            rce_adaptation_spectral_gap_threshold: default_rce_adaptation_spectral_gap_threshold(),
            rce_circuit_breaker_streak: default_rce_circuit_breaker_streak(),
            tough_knots_multiplier: default_tough_knots_multiplier(),
            tough_knots_max_fetch: default_tough_knots_max_fetch(),
            tough_knots_knot_threshold: default_tough_knots_knot_threshold(),
            tough_knots_quality_threshold: default_tough_knots_quality_threshold(),
            tough_knots_knot_multiplier: default_tough_knots_knot_multiplier(),
            compass_h1_persistence_divisor: default_compass_h1_persistence_divisor(),
            compass_h1_penalty_scale: default_compass_h1_penalty_scale(),
            compass_sheaf_energy_threshold: default_compass_sheaf_energy_threshold(),
            compass_sheaf_boost_multiplier: default_compass_sheaf_boost_multiplier(),
            compass_dominance_penalty_multiplier: default_compass_dominance_penalty_multiplier(),
            compass_dominance_boost_multiplier: default_compass_dominance_boost_multiplier(),
            compass_arousal_penalty_multiplier: default_compass_arousal_penalty_multiplier(),
            compass_random_noise_range: default_compass_random_noise_range(),
            compass_pleasure_boost_probability: default_compass_pleasure_boost_probability(),
            compass_pleasure_boost_multiplier: default_compass_pleasure_boost_multiplier(),
            compass_base_threat_arousal_threshold: default_compass_base_threat_arousal_threshold(),
            compass_variance_spike_multiplier: default_compass_variance_spike_multiplier(),
            compass_random_threat_probability: default_compass_random_threat_probability(),
            compass_random_threat_arousal_threshold: default_compass_random_threat_arousal_threshold(),
            compass_random_threat_pleasure_threshold: default_compass_random_threat_pleasure_threshold(),
            compass_healing_pleasure_threshold: default_compass_healing_pleasure_threshold(),
            compass_healing_dominance_threshold: default_compass_healing_dominance_threshold(),
            compass_quadrant_panic_pleasure_threshold: default_compass_quadrant_panic_pleasure_threshold(),
            compass_quadrant_panic_arousal_threshold: default_compass_quadrant_panic_arousal_threshold(),
            compass_quadrant_persist_arousal_threshold: default_compass_quadrant_persist_arousal_threshold(),
            compass_reward_panic_to_discover: default_compass_reward_panic_to_discover(),
            compass_reward_panic_to_persist: default_compass_reward_panic_to_persist(),
            compass_reward_panic_to_master: default_compass_reward_panic_to_master(),
            compass_reward_master_to_panic: default_compass_reward_master_to_panic(),
            compass_reward_default: default_compass_reward_default(),
            compass_reward_entropy_multiplier: default_compass_reward_entropy_multiplier(),
            compass_mcts_h1_bonus_cap: default_compass_mcts_h1_bonus_cap(),
            compass_mcts_h1_bonus_multiplier: default_compass_mcts_h1_bonus_multiplier(),
            compass_mcts_persistence_divisor: default_compass_mcts_persistence_divisor(),
            compass_mcts_persistence_multiplier: default_compass_mcts_persistence_multiplier(),
            compass_mcts_knot_multiplier: default_compass_mcts_knot_multiplier(),
            compass_mcts_knot_multiplier_cap: default_compass_mcts_knot_multiplier_cap(),
            compass_mcts_knot_weight: default_compass_mcts_knot_weight(),
            compass_mcts_gap_multiplier: default_compass_mcts_gap_multiplier(),
            compass_mcts_entropy_multiplier: default_compass_mcts_entropy_multiplier(),
            compass_mcts_entropy_multiplier_cap: default_compass_mcts_entropy_multiplier_cap(),
            compass_mcts_entropy_weight: default_compass_mcts_entropy_weight(),
            compass_mcts_h0_bonus_cap: default_compass_mcts_h0_bonus_cap(),
            compass_mcts_h0_bonus_multiplier: default_compass_mcts_h0_bonus_multiplier(),
            compass_mcts_default_exploration_base: default_compass_mcts_default_exploration_base(),
            compass_mcts_default_exploration_divisor: default_compass_mcts_default_exploration_divisor(),
            compass_cascade_min_consonance: default_compass_cascade_min_consonance(),
            compass_cascade_recognition_satisfaction_consonance: default_compass_cascade_recognition_satisfaction_consonance(),
            compass_cascade_calm_motivation_consonance: default_compass_cascade_calm_motivation_consonance(),
            learning_executor_memory_limit: default_learning_executor_memory_limit(),
            learning_executor_cluster_threshold: default_learning_executor_cluster_threshold(),
            learning_reward_threshold: default_learning_reward_threshold(),
            learning_reptile_episode_interval: default_learning_reptile_episode_interval(),
            learning_evolution_episode_interval: default_learning_evolution_episode_interval(),
            learning_reptile_batch_size: default_learning_reptile_batch_size(),
            learning_qlora_low_reward_threshold: default_learning_qlora_low_reward_threshold(),
            learning_qlora_sample_count: default_learning_qlora_sample_count(),
            learning_qlora_max_samples: default_learning_qlora_max_samples(),
            learning_epsilon_decay_rate: default_learning_epsilon_decay_rate(),
            learning_epsilon_minimum: default_learning_epsilon_minimum(),
            learning_alpha_decay_rate: default_learning_alpha_decay_rate(),
            learning_alpha_minimum: default_learning_alpha_minimum(),
            learning_evolution_old_episodes_ratio: default_learning_evolution_old_episodes_ratio(),
            learning_evolution_old_episodes_min: default_learning_evolution_old_episodes_min(),
            learning_evolution_old_episodes_max: default_learning_evolution_old_episodes_max(),
            learning_tough_knots_ratio: default_learning_tough_knots_ratio(),
            learning_tcs_knot_penalty: default_learning_tcs_knot_penalty(),
            learning_tcs_betti1_penalty: default_learning_tcs_betti1_penalty(),
            learning_tcs_entropy_penalty: default_learning_tcs_entropy_penalty(),
            learning_tcs_discover_weight: default_learning_tcs_discover_weight(),
            learning_tcs_spectral_gap_threshold: default_learning_tcs_spectral_gap_threshold(),
            learning_tcs_convergence_bonus: default_learning_tcs_convergence_bonus(),
            learning_tcs_convergence_penalty: default_learning_tcs_convergence_penalty(),
            learning_tcs_novelty_threshold: default_learning_tcs_novelty_threshold(),
            learning_tcs_novelty_bonus: default_learning_tcs_novelty_bonus(),
            learning_dqn_batch_size: default_learning_dqn_batch_size(),
            learning_dqn_temp_multiplier: default_learning_dqn_temp_multiplier(),
            learning_dqn_top_p_multiplier: default_learning_dqn_top_p_multiplier(),
            learning_dqn_mcts_c_multiplier: default_learning_dqn_mcts_c_multiplier(),
            learning_dqn_retrieval_multiplier: default_learning_dqn_retrieval_multiplier(),
            learning_dqn_novelty_multiplier: default_learning_dqn_novelty_multiplier(),
            learning_dqn_awareness_multiplier: default_learning_dqn_awareness_multiplier(),
            learning_reptile_inner_gradient_multiplier: default_learning_reptile_inner_gradient_multiplier(),
            learning_evolution_temp_multiplier: default_learning_evolution_temp_multiplier(),
            learning_evolution_alpha_multiplier: default_learning_evolution_alpha_multiplier(),
            learning_evolution_mutation_reduce_multiplier: default_learning_evolution_mutation_reduce_multiplier(),
            learning_evolution_mutation_increase_multiplier: default_learning_evolution_mutation_increase_multiplier(),
            generation_reflexion_temp_base_multiplier: default_generation_reflexion_temp_base_multiplier(),
            generation_reflexion_temp_stability_multiplier: default_generation_reflexion_temp_stability_multiplier(),
            generation_reflexion_top_p_increment: default_generation_reflexion_top_p_increment(),
            generation_reflexion_top_p_stability_increment: default_generation_reflexion_top_p_stability_increment(),
            generation_reflexion_top_p_max: default_generation_reflexion_top_p_max(),
            generation_cot_repair_temp_base_multiplier: default_generation_cot_repair_temp_base_multiplier(),
            generation_cot_repair_temp_iteration_increment: default_generation_cot_repair_temp_iteration_increment(),
            generation_cot_repair_top_p_increment: default_generation_cot_repair_top_p_increment(),
            generation_cot_repair_top_p_max: default_generation_cot_repair_top_p_max(),
            generation_cot_repair_temp_min: default_generation_cot_repair_temp_min(),
            generation_cot_repair_temp_max: default_generation_cot_repair_temp_max(),
            erag_similarity_boost_multiplier: default_erag_similarity_boost_multiplier(),
            erag_similarity_boost_max: default_erag_similarity_boost_max(),
        };

        runtime.apply_hardware_overrides(args.hardware);

        info!(model = %runtime.curator_model_name, "Config loaded: CURATOR_MODEL={}", runtime.curator_model_name);
        info!(mode = ?runtime.topology_mode, "Topology mode configured");

        Ok(runtime)
    }

    fn apply_hardware_overrides(&mut self, hardware: HardwareProfile) {
        match hardware {
            HardwareProfile::RTX5090 => {
                info!("Applying RTX 5090 hardware overrides - MAXIMUM GPU UTILIZATION");

                // Force GPU usage everywhere
                self.use_gpu_fitness = true;
                self.optimized_erag = true;
                self.cache_prefetch_enabled = true;
                self.cache_prefetch_parallelism = self.cache_prefetch_parallelism.max(16);
                self.cache_prefetch_prompts = self.cache_prefetch_prompts.max(32);
                self.cache_prefetch_top_hits = self.cache_prefetch_top_hits.max(16);
                self.erag_batch_size = self.erag_batch_size.max(512); // Larger batches for RTX 5090
                self.erag_batch_flush_ms = self.erag_batch_flush_ms.min(100);
                self.generation_max_tokens = self.generation_max_tokens.max(8192);
                self.dynamic_token_max = self.dynamic_token_max.max(2048);
                self.token_promotion_interval = self.token_promotion_interval.min(20);
                self.cache_capacity = self.cache_capacity.max(8_192);
                
                // FORCE CUDA - no fallbacks
                self.weighted_memory_config.gpu_device = "cuda".to_string();

                if self.parallel_curator_rouge == false {
                    self.parallel_curator_rouge = true;
                }

                info!(
                    use_gpu_fitness = self.use_gpu_fitness,
                    optimized_erag = self.optimized_erag,
                    gpu_device = %self.weighted_memory_config.gpu_device,
                    erag_batch_size = self.erag_batch_size,
                    generation_max_tokens = self.generation_max_tokens,
                    token_promotion_interval = self.token_promotion_interval,
                    cache_prefetch_parallelism = self.cache_prefetch_parallelism,
                    "RTX 5090 MAXIMUM GPU overrides applied"
                );
            }
            HardwareProfile::H200 => {
                info!("Applying H200 hardware overrides");

                self.use_gpu_fitness = true;
                self.optimized_erag = true;
                self.cache_prefetch_enabled = true;
                self.cache_prefetch_parallelism = self.cache_prefetch_parallelism.max(12);
                self.cache_prefetch_prompts = self.cache_prefetch_prompts.max(16);
                self.cache_prefetch_top_hits = self.cache_prefetch_top_hits.max(8);
                self.erag_batch_size = self.erag_batch_size.max(256);
                self.erag_batch_flush_ms = self.erag_batch_flush_ms.min(150);
                self.generation_max_tokens = self.generation_max_tokens.max(4096);
                self.dynamic_token_max = self.dynamic_token_max.max(1024);
                self.token_promotion_interval = self.token_promotion_interval.min(30);
                self.cache_capacity = self.cache_capacity.max(4_096);
                if !self
                    .weighted_memory_config
                    .gpu_device
                    .eq_ignore_ascii_case("cuda")
                {
                    self.weighted_memory_config.gpu_device = "cuda".to_string();
                }

                if self.parallel_curator_rouge == false {
                    self.parallel_curator_rouge = true;
                }

                info!(
                    use_gpu_fitness = self.use_gpu_fitness,
                    optimized_erag = self.optimized_erag,
                    gpu_device = %self.weighted_memory_config.gpu_device,
                    erag_batch_size = self.erag_batch_size,
                    generation_max_tokens = self.generation_max_tokens,
                    token_promotion_interval = self.token_promotion_interval,
                    "H200 overrides applied"
                );
            }
            HardwareProfile::Laptop5080Q => {
                if self.weighted_memory_config.gpu_device == "cpu" {
                    self.weighted_memory_config.gpu_device = "auto".to_string();
                }
                self.use_gpu_fitness = true;
                self.optimized_erag = true;
                self.cache_prefetch_enabled = true;
            }
            HardwareProfile::Beelink => {
                // No overrides; defaults tuned for Beelink edge boxes.
            }
        }
    }

    /// Serialize the active runtime configuration to a JSON file on disk.
    /// Intended for baseline freezes and reproducibility logs.
    pub fn snapshot_to_json<P: AsRef<std::path::Path>>(&self, path: P) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// Curator configuration derived from runtime config
#[derive(Debug, Clone)]
pub struct CuratorConfig {
    pub vllm_endpoint: String,
    pub ollama_endpoint: String,
    pub model_name: String,
    pub curator_backend: CuratorBackend, // NEW: Backend selection
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
        // Determine curator backend from env or default to vLLM
        let curator_backend = CuratorBackend::from_env();

        // If vLLM backend, use separate endpoint if configured, otherwise use main vLLM endpoint
        let curator_vllm_endpoint =
            env_with_fallback(&["CURATOR_VLLM_ENDPOINT", "CURATOR_ENDPOINT"])
                .unwrap_or_else(|| config.vllm_endpoint.clone());

        Self {
            vllm_endpoint: curator_vllm_endpoint,
            ollama_endpoint: config.ollama_endpoint.clone(),
            model_name: config.curator_model_name.clone(),
            curator_backend,
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
        set_env_override(key, value);
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

// Phase 1-6: Back-half pipeline optimization defaults
fn default_erag_batch_size() -> usize {
    env_with_fallback(&["ERAG_BATCH_SIZE"])
        .and_then(|v| v.parse().ok())
        .unwrap_or(128)
}

fn default_erag_batch_flush_ms() -> u64 {
    env_with_fallback(&["ERAG_BATCH_FLUSH_MS"])
        .and_then(|v| v.parse().ok())
        .unwrap_or(300)
}

fn default_fp16_qlora_adapters() -> bool {
    env_with_fallback(&["FP16_QLORA_ADAPTERS"])
        .and_then(|v| v.parse().ok())
        .unwrap_or(true) // Default to true for optimization
}

fn default_training_service_enabled() -> bool {
    env_with_fallback(&["TRAINING_SERVICE_ENABLED"])
        .and_then(|v| v.parse().ok())
        .unwrap_or(false) // Default to false for backward compatibility
}

fn default_training_service_url() -> String {
    env_with_fallback(&["TRAINING_SERVICE_URL"])
        .unwrap_or_else(|| "http://localhost:8001".to_string())
}

fn default_training_service_use_grpc() -> bool {
    env_with_fallback(&["TRAINING_SERVICE_USE_GRPC"])
        .and_then(|v| v.parse().ok())
        .unwrap_or(false) // Default to HTTP REST
}

fn default_adapter_storage_path() -> String {
    env_with_fallback(&["ADAPTER_STORAGE_PATH"])
        .unwrap_or_else(|| "models/system2_adapters".to_string())
}

fn default_training_queue_path() -> String {
    env_with_fallback(&["TRAINING_QUEUE_PATH"])
        .unwrap_or_else(|| "data/training_queue".to_string())
}

fn default_parallel_curator_rouge() -> bool {
    env_with_fallback(&["PARALLEL_CURATOR_ROUGE"])
        .and_then(|v| v.parse().ok())
        .unwrap_or(true) // Default to true for optimization
}

fn default_curator_feedback_window_size() -> usize {
    20
}

fn default_curator_feedback_threshold_adjustment() -> f32 {
    0.05
}

fn default_curator_feedback_threshold_min() -> f32 {
    0.3
}

fn default_curator_feedback_threshold_max() -> f32 {
    0.9
}

fn default_curator_feedback_quality_trend_threshold() -> f32 {
    0.05
}

fn default_curator_feedback_temp_adjustment_multiplier() -> f32 {
    0.1
}

fn default_curator_feedback_learned_rate_low() -> f32 {
    0.3
}

fn default_curator_feedback_quality_low() -> f32 {
    0.6
}

fn default_curator_feedback_top_p_increase() -> f64 {
    0.05
}

fn default_curator_feedback_learned_rate_high() -> f32 {
    0.7
}

fn default_curator_feedback_quality_high() -> f32 {
    0.7
}

fn default_curator_feedback_top_p_decrease() -> f64 {
    -0.02
}

fn default_curator_feedback_retrieval_quality_threshold() -> f32 {
    0.5
}

fn default_curator_feedback_retrieval_top_k_increase() -> f64 {
    1.0
}

fn default_curator_feedback_retrieval_quality_high() -> f32 {
    0.8
}

fn default_curator_feedback_retrieval_learned_rate_high() -> f32 {
    0.6
}

fn default_curator_feedback_retrieval_top_k_decrease() -> f64 {
    -0.5
}

fn default_pipeline_retrieval_top_k_min() -> usize {
    1
}

fn default_pipeline_retrieval_top_k_max() -> usize {
    50
}

fn default_pipeline_timing_split_ratio() -> f64 {
    0.5
}

fn default_pipeline_healing_knot_threshold() -> f64 {
    0.4
}

fn default_pipeline_healing_spectral_gap_threshold() -> f64 {
    0.6
}

fn default_pipeline_ucb1_max_clamp() -> f64 {
    1.0
}

fn default_pipeline_quality_score_increment() -> f32 {
    0.1
}

fn default_pipeline_param_min() -> f64 {
    0.1
}

fn default_pipeline_param_max() -> f64 {
    1.0
}

fn default_pipeline_retrieval_top_k_increment_min() -> f64 {
    0.0
}

fn default_pipeline_retrieval_top_k_increment_max() -> f64 {
    10.0
}

fn default_topology_memory_analyzer_threshold() -> f64 {
    0.3
}

fn default_discovery_buffer_interval_secs() -> u64 {
    1
}

fn default_embedding_cache_capacity() -> usize {
    1000
}

fn default_collapse_cache_capacity() -> usize {
    500
}

fn default_mcts_exploration_constant() -> f64 {
    1.414 // sqrt(2) - standard UCB1 exploration constant
}

fn default_mcts_depth() -> usize {
    5
}

fn default_discovery_buffer_threshold() -> usize {
    10
}

fn default_gpu_fitness_refresh_interval_secs() -> u64 {
    30
}

fn default_learning_timeout_secs() -> u64 {
    10
}

fn default_context_truncation_limit() -> usize {
    100
}

fn default_base_retrieval_top_k() -> i32 {
    3
}

fn default_delay_threshold_ms() -> u64 {
    100
}

fn default_generation_client_timeout_secs() -> u64 {
    60
}

fn default_memory_upsert_timeout_secs() -> u64 {
    5
}

fn default_rouge_acceptable_threshold() -> f64 {
    0.25
}

fn default_rouge_improvement_threshold() -> f64 {
    0.1
}

fn default_ucb1_boost_threshold() -> f64 {
    0.2
}

fn default_ucb1_relaxation_threshold() -> f64 {
    0.15
}

fn default_retry_count_for_relaxation() -> u32 {
    3
}

fn default_quality_base_score() -> f32 {
    0.5
}

fn default_quality_max_length() -> usize {
    1000
}

fn default_quality_length_factor_weight() -> f32 {
    0.2
}

fn default_quality_entropy_threshold() -> f64 {
    0.5
}

fn default_quality_entropy_factor_weight() -> f32 {
    0.15
}

fn default_knot_complexity_penalty_threshold() -> f64 {
    0.6
}

fn default_knot_complexity_penalty_multiplier() -> f32 {
    0.9
}

fn default_spectral_gap_bonus_threshold() -> f64 {
    0.7
}

fn default_spectral_gap_bonus_multiplier() -> f32 {
    1.1
}

fn default_betti1_quality_threshold() -> usize {
    3
}

fn default_betti1_bonus_multiplier() -> f32 {
    1.05
}

fn default_betti1_penalty_multiplier() -> f32 {
    0.95
}

fn default_persistence_entropy_quality_threshold() -> f64 {
    0.3
}

fn default_persistence_entropy_bonus_multiplier() -> f32 {
    1.05
}

fn default_topology_refinement_knot_threshold() -> f64 {
    0.7
}

fn default_topology_refinement_betti1_threshold() -> usize {
    5
}

fn default_topology_refinement_entropy_threshold() -> f64 {
    0.8
}

fn default_autonomous_refinement_temperature() -> f64 {
    0.22
}

fn default_autonomous_refinement_top_p() -> f64 {
    0.82
}

fn default_autonomous_refinement_improvement_weight() -> f32 {
    0.35
}

fn default_autonomous_refinement_improvement_threshold() -> f64 {
    0.05
}

fn default_second_pass_refinement_threshold() -> f64 {
    0.25
}

fn default_second_pass_refinement_temperature() -> f64 {
    0.28
}

fn default_second_pass_refinement_top_p() -> f64 {
    0.78
}

fn default_enhancement_temperature() -> f64 {
    0.3
}

fn default_enhancement_top_p() -> f64 {
    0.95
}

fn default_reward_rouge_weight() -> f64 {
    0.5
}

fn default_reward_entropy_weight() -> f64 {
    0.5
}

fn default_consistency_voting_quality() -> f64 {
    0.8
}

/// Failure signal thresholds for hard/soft failure detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureSignalThresholds {
    /// Hard failure ROUGE threshold (below this triggers hard failure)
    #[serde(default = "default_hard_rouge_threshold")]
    pub hard_rouge_threshold: f64,
    /// Hard failure entropy delta threshold (above this triggers hard failure)
    #[serde(default = "default_hard_entropy_delta_threshold")]
    pub hard_entropy_delta_threshold: f64,
    /// Hard failure curator threshold (below this triggers hard failure)
    #[serde(default = "default_hard_curator_threshold")]
    pub hard_curator_threshold: f64,
    /// Soft failure UCB1 threshold (below this triggers soft failure)
    #[serde(default = "default_soft_ucb_threshold")]
    pub soft_ucb_threshold: f64,
    /// Soft failure average similarity threshold (below this triggers soft failure)
    #[serde(default = "default_soft_avg_similarity_threshold")]
    pub soft_avg_similarity_threshold: f32,
    /// Soft failure OOV rate threshold (above this triggers soft failure)
    #[serde(default = "default_soft_oov_threshold")]
    pub soft_oov_threshold: f64,
    /// Low quality hits threshold (above this triggers soft failure)
    #[serde(default = "default_low_quality_hits_threshold")]
    pub low_quality_hits_threshold: usize,
}

impl Default for FailureSignalThresholds {
    fn default() -> Self {
        Self {
            hard_rouge_threshold: default_hard_rouge_threshold(),
            hard_entropy_delta_threshold: default_hard_entropy_delta_threshold(),
            hard_curator_threshold: default_hard_curator_threshold(),
            soft_ucb_threshold: default_soft_ucb_threshold(),
            soft_avg_similarity_threshold: default_soft_avg_similarity_threshold(),
            soft_oov_threshold: default_soft_oov_threshold(),
            low_quality_hits_threshold: default_low_quality_hits_threshold(),
        }
    }
}

fn default_hard_rouge_threshold() -> f64 {
    0.5
}

fn default_hard_entropy_delta_threshold() -> f64 {
    0.1
}

fn default_hard_curator_threshold() -> f64 {
    0.7
}

fn default_soft_ucb_threshold() -> f64 {
    0.3
}

fn default_soft_avg_similarity_threshold() -> f32 {
    0.4
}

fn default_soft_oov_threshold() -> f64 {
    0.2
}

fn default_low_quality_hits_threshold() -> usize {
    3
}

fn default_rce_erag_cosine_weight() -> f64 {
    0.7
}

fn default_rce_erag_entropy_weight() -> f64 {
    0.3
}

fn default_rce_adaptation_entropy_threshold() -> f64 {
    0.7
}

fn default_rce_adaptation_spectral_gap_threshold() -> f64 {
    0.7
}

fn default_rce_circuit_breaker_streak() -> u32 {
    3
}

fn default_tough_knots_multiplier() -> usize {
    4
}

fn default_tough_knots_max_fetch() -> usize {
    512
}

fn default_tough_knots_knot_threshold() -> f64 {
    0.4
}

fn default_tough_knots_quality_threshold() -> f64 {
    0.5
}

fn default_tough_knots_knot_multiplier() -> f64 {
    2.0
}

// Compass PAD adjustment defaults
fn default_compass_h1_persistence_divisor() -> f64 {
    2.5
}

fn default_compass_h1_penalty_scale() -> f64 {
    0.3
}

fn default_compass_sheaf_energy_threshold() -> f64 {
    0.3
}

fn default_compass_sheaf_boost_multiplier() -> f64 {
    0.5
}

fn default_compass_dominance_penalty_multiplier() -> f64 {
    0.7
}

fn default_compass_dominance_boost_multiplier() -> f64 {
    0.8
}

fn default_compass_arousal_penalty_multiplier() -> f64 {
    0.5
}

fn default_compass_random_noise_range() -> f64 {
    0.4
}

fn default_compass_pleasure_boost_probability() -> f64 {
    0.15
}

fn default_compass_pleasure_boost_multiplier() -> f64 {
    1.1
}

// Compass threat detection defaults
fn default_compass_base_threat_arousal_threshold() -> f64 {
    0.05
}

fn default_compass_variance_spike_multiplier() -> f64 {
    1.2
}

fn default_compass_random_threat_probability() -> f64 {
    0.45
}

fn default_compass_random_threat_arousal_threshold() -> f64 {
    -0.2
}

fn default_compass_random_threat_pleasure_threshold() -> f64 {
    0.35
}

// Compass healing detection defaults
fn default_compass_healing_pleasure_threshold() -> f64 {
    0.25
}

fn default_compass_healing_dominance_threshold() -> f64 {
    0.05
}

// Compass quadrant thresholds defaults
fn default_compass_quadrant_panic_pleasure_threshold() -> f64 {
    -0.1
}

fn default_compass_quadrant_panic_arousal_threshold() -> f64 {
    0.2
}

fn default_compass_quadrant_persist_arousal_threshold() -> f64 {
    0.2
}

// Compass intrinsic reward defaults
fn default_compass_reward_panic_to_discover() -> f64 {
    10.0
}

fn default_compass_reward_panic_to_persist() -> f64 {
    -1.0
}

fn default_compass_reward_panic_to_master() -> f64 {
    10.0
}

fn default_compass_reward_master_to_panic() -> f64 {
    -5.0
}

fn default_compass_reward_default() -> f64 {
    1.0
}

fn default_compass_reward_entropy_multiplier() -> f64 {
    5.0
}

// Compass MCTS branch defaults
fn default_compass_mcts_h1_bonus_cap() -> f64 {
    5.0
}

fn default_compass_mcts_h1_bonus_multiplier() -> f64 {
    0.1
}

fn default_compass_mcts_persistence_divisor() -> f64 {
    3.0
}

fn default_compass_mcts_persistence_multiplier() -> f64 {
    0.15
}

fn default_compass_mcts_knot_multiplier() -> f64 {
    2.0
}

fn default_compass_mcts_knot_multiplier_cap() -> f64 {
    1.0
}

fn default_compass_mcts_knot_weight() -> f64 {
    0.2
}

fn default_compass_mcts_gap_multiplier() -> f64 {
    0.15
}

fn default_compass_mcts_entropy_multiplier() -> f64 {
    1.5
}

fn default_compass_mcts_entropy_multiplier_cap() -> f64 {
    1.0
}

fn default_compass_mcts_entropy_weight() -> f64 {
    0.12
}

fn default_compass_mcts_h0_bonus_cap() -> f64 {
    5.0
}

fn default_compass_mcts_h0_bonus_multiplier() -> f64 {
    0.1
}

fn default_compass_mcts_default_exploration_base() -> f64 {
    0.05
}

fn default_compass_mcts_default_exploration_divisor() -> f64 {
    3.0
}

// Compass cascade defaults
fn default_compass_cascade_min_consonance() -> f64 {
    0.7
}

fn default_compass_cascade_recognition_satisfaction_consonance() -> f64 {
    0.8
}

fn default_compass_cascade_calm_motivation_consonance() -> f64 {
    0.75
}

// Learning loop defaults
fn default_learning_executor_memory_limit() -> usize {
    256
}

fn default_learning_executor_cluster_threshold() -> f32 {
    0.82
}

fn default_learning_reward_threshold() -> f64 {
    -0.5
}

fn default_learning_reptile_episode_interval() -> u32 {
    5
}

fn default_learning_evolution_episode_interval() -> u32 {
    50
}

fn default_learning_reptile_batch_size() -> usize {
    32
}

fn default_learning_qlora_low_reward_threshold() -> f64 {
    -0.5
}

fn default_learning_qlora_sample_count() -> usize {
    16
}

fn default_learning_qlora_max_samples() -> usize {
    64
}

fn default_learning_epsilon_decay_rate() -> f64 {
    0.001
}

fn default_learning_epsilon_minimum() -> f64 {
    0.01
}

fn default_learning_alpha_decay_rate() -> f64 {
    0.0005
}

fn default_learning_alpha_minimum() -> f64 {
    0.001
}

fn default_learning_evolution_old_episodes_ratio() -> f64 {
    0.3
}

fn default_learning_evolution_old_episodes_min() -> usize {
    10
}

fn default_learning_evolution_old_episodes_max() -> usize {
    50
}

fn default_learning_tough_knots_ratio() -> f64 {
    0.2
}

fn default_learning_tcs_knot_penalty() -> f64 {
    0.5
}

fn default_learning_tcs_betti1_penalty() -> f64 {
    0.2
}

fn default_learning_tcs_entropy_penalty() -> f64 {
    0.1
}

fn default_learning_tcs_discover_weight() -> f64 {
    0.5
}

fn default_learning_tcs_spectral_gap_threshold() -> f64 {
    0.5
}

fn default_learning_tcs_convergence_bonus() -> f64 {
    0.3
}

fn default_learning_tcs_convergence_penalty() -> f64 {
    -0.2
}

fn default_learning_tcs_novelty_threshold() -> f64 {
    0.1
}

fn default_learning_tcs_novelty_bonus() -> f64 {
    0.2
}

fn default_learning_dqn_batch_size() -> usize {
    32
}

fn default_learning_dqn_temp_multiplier() -> f64 {
    0.05
}

fn default_learning_dqn_top_p_multiplier() -> f64 {
    0.1
}

fn default_learning_dqn_mcts_c_multiplier() -> f64 {
    0.1
}

fn default_learning_dqn_retrieval_multiplier() -> f64 {
    0.01
}

fn default_learning_dqn_novelty_multiplier() -> f64 {
    0.05
}

fn default_learning_dqn_awareness_multiplier() -> f64 {
    0.03
}

fn default_learning_reptile_inner_gradient_multiplier() -> f64 {
    0.01
}

fn default_learning_evolution_temp_multiplier() -> f64 {
    0.2
}

fn default_learning_evolution_alpha_multiplier() -> f64 {
    0.1
}

fn default_learning_evolution_mutation_reduce_multiplier() -> f64 {
    0.7
}

fn default_learning_evolution_mutation_increase_multiplier() -> f64 {
    1.3
}

// Generation defaults
fn default_generation_reflexion_temp_base_multiplier() -> f64 {
    0.7
}

fn default_generation_reflexion_temp_stability_multiplier() -> f64 {
    0.3
}

fn default_generation_reflexion_top_p_increment() -> f64 {
    0.05
}

fn default_generation_reflexion_top_p_stability_increment() -> f64 {
    0.2
}

fn default_generation_reflexion_top_p_max() -> f64 {
    0.99
}

fn default_generation_cot_repair_temp_base_multiplier() -> f64 {
    0.6
}

fn default_generation_cot_repair_temp_iteration_increment() -> f64 {
    0.1
}

fn default_generation_cot_repair_top_p_increment() -> f64 {
    0.05
}

fn default_generation_cot_repair_top_p_max() -> f64 {
    0.98
}

fn default_generation_cot_repair_temp_min() -> f64 {
    0.1
}

fn default_generation_cot_repair_temp_max() -> f64 {
    1.2
}

// ERAG defaults
fn default_erag_similarity_boost_multiplier() -> f64 {
    1.2
}

fn default_erag_similarity_boost_max() -> f64 {
    1.0
}

fn env_with_fallback(keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(value) = env_value(key) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}
