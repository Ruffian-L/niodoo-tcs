use std::env;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

mod environment;

pub use environment::{
    env_value, env_var, env_with_fallback, prime_environment, set_env_override,
};

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

impl CuratorBackend {
    pub fn from_env() -> Self {
        match env_with_fallback(&["CURATOR_BACKEND", "CURATOR_TYPE"]) {
            Some(value) => match value.to_ascii_lowercase().as_str() {
                "vllm" | "vllm_gpu" => CuratorBackend::Vllm,
                "ollama" | "ollama_cpu" => CuratorBackend::Ollama,
                _ => {
                    warn!(%value, "Invalid curator backend; defaulting to vLLM");
                    CuratorBackend::Vllm
                }
            },
            None => CuratorBackend::Vllm, // Default to vLLM
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

        let qdrant_embedded = env_with_fallback(&["QDRANT_EMBEDDED"])
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);

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
            .map(|n| n.clamp(100, 500))
            .unwrap_or_else(default_lens_snippet_chars);
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

        let runtime = Self {
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
        };

        info!(model = %runtime.curator_model_name, "Config loaded: CURATOR_MODEL={}", runtime.curator_model_name);
        info!(mode = ?runtime.topology_mode, "Topology mode configured");

        runtime.validate()?;
        Ok(runtime)
    }

    pub fn validate(&self) -> Result<()> {
        // Validate numeric ranges
        if self.prompt_max_chars > 1_000_000 {
            return Err(anyhow::anyhow!(
                "prompt_max_chars ({}) exceeds maximum allowed value (1,000,000)",
                self.prompt_max_chars
            ));
        }

        if self.generation_max_tokens > 100_000 {
            return Err(anyhow::anyhow!(
                "generation_max_tokens ({}) exceeds maximum allowed value (100,000)",
                self.generation_max_tokens
            ));
        }

        if self.generation_timeout_secs > 3600 {
            return Err(anyhow::anyhow!(
                "generation_timeout_secs ({}) exceeds maximum allowed value (3600)",
                self.generation_timeout_secs
            ));
        }

        if self.temperature < 0.0 || self.temperature > 2.0 {
            return Err(anyhow::anyhow!(
                "temperature ({}) must be between 0.0 and 2.0",
                self.temperature
            ));
        }

        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(anyhow::anyhow!(
                "top_p ({}) must be between 0.0 and 1.0",
                self.top_p
            ));
        }

        // Validate paths exist (if not mock mode)
        if !self.mock_mode {
            if !Path::new(&self.training_data_path).exists() {
                warn!(path = %self.training_data_path, "training_data_path does not exist");
            }
            if !Path::new(&self.emotional_seed_path).exists() {
                warn!(path = %self.emotional_seed_path, "emotional_seed_path does not exist");
            }
        }

        // Validate URLs
        if !self.vllm_endpoint.starts_with("http://") && !self.vllm_endpoint.starts_with("https://")
        {
            return Err(anyhow::anyhow!(
                "vllm_endpoint ({}) must be a valid HTTP(S) URL",
                self.vllm_endpoint
            ));
        }

        if !self.qdrant_url.starts_with("http://") && !self.qdrant_url.starts_with("https://") {
            return Err(anyhow::anyhow!(
                "qdrant_url ({}) must be a valid HTTP(S) URL",
                self.qdrant_url
            ));
        }

        if !self.ollama_endpoint.starts_with("http://")
            && !self.ollama_endpoint.starts_with("https://")
        {
            return Err(anyhow::anyhow!(
                "ollama_endpoint ({}) must be a valid HTTP(S) URL",
                self.ollama_endpoint
            ));
        }

        // Validate security config
        if self.security.prompt_max_chars > 0
            && self.prompt_max_chars > self.security.prompt_max_chars
        {
            return Err(anyhow::anyhow!(
                "prompt_max_chars ({}) exceeds security.prompt_max_chars ({})",
                self.prompt_max_chars,
                self.security.prompt_max_chars
            ));
        }

        if self.security.rate_limit_window_secs == 0 {
            return Err(anyhow::anyhow!(
                "security.rate_limit_window_secs must be greater than 0"
            ));
        }

        // Validate Qdrant vector dimension
        if self.qdrant_vector_dim == 0 || self.qdrant_vector_dim > 65536 {
            return Err(anyhow::anyhow!(
                "qdrant_vector_dim ({}) must be between 1 and 65536",
                self.qdrant_vector_dim
            ));
        }

        // Validate cache capacity
        if self.cache_capacity == 0 {
            return Err(anyhow::anyhow!(
                "cache_capacity ({}) must be greater than 0",
                self.cache_capacity
            ));
        }

        // Validate retry configurations
        if self.phase2_max_retries > 100 {
            return Err(anyhow::anyhow!(
                "phase2_max_retries ({}) exceeds maximum allowed value (100)",
                self.phase2_max_retries
            ));
        }

        if self.phase2_retry_base_delay_ms == 0 {
            return Err(anyhow::anyhow!(
                "phase2_retry_base_delay_ms ({}) must be greater than 0",
                self.phase2_retry_base_delay_ms
            ));
        }

        // Validate similarity threshold
        if self.similarity_threshold < 0.0 || self.similarity_threshold > 1.0 {
            return Err(anyhow::anyhow!(
                "similarity_threshold ({}) must be between 0.0 and 1.0",
                self.similarity_threshold
            ));
        }

        // Validate curator thresholds
        if self.curator_quality_threshold < 0.0 || self.curator_quality_threshold > 1.0 {
            return Err(anyhow::anyhow!(
                "curator_quality_threshold ({}) must be between 0.0 and 1.0",
                self.curator_quality_threshold
            ));
        }

        if self.curator_minimum_threshold < 0.0 || self.curator_minimum_threshold > 1.0 {
            return Err(anyhow::anyhow!(
                "curator_minimum_threshold ({}) must be between 0.0 and 1.0",
                self.curator_minimum_threshold
            ));
        }

        // Validate timeout values
        if self.curator_timeout_secs == 0 {
            return Err(anyhow::anyhow!(
                "curator_timeout_secs ({}) must be greater than 0",
                self.curator_timeout_secs
            ));
        }

        // Validate cache TTL values
        if self.embedding_cache_ttl_secs == 0 {
            return Err(anyhow::anyhow!(
                "embedding_cache_ttl_secs ({}) must be greater than 0",
                self.embedding_cache_ttl_secs
            ));
        }

        if self.collapse_cache_ttl_secs == 0 {
            return Err(anyhow::anyhow!(
                "collapse_cache_ttl_secs ({}) must be greater than 0",
                self.collapse_cache_ttl_secs
            ));
        }

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

