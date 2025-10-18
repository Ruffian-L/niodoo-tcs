/*
 * 🌟 CENTRALIZED CONFIGURATION SYSTEM 🌟
 *
 * This module provides a centralized configuration system that eliminates hardcodes
 * and makes the entire system tunable without recompiles or rebuilds.
 */

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Helper function to get environment variable with default value
fn env_var_with_default(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| {
        debug!(
            "Environment variable {} not set, using default: {}",
            key, default
        );
        default.to_string()
    })
}

/// Path configuration structure - NO HARDCODED PATHS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathConfig {
    /// Project root directory (from NIODOO_PROJECT_ROOT env var or cwd)
    pub project_root: PathBuf,
    /// Data directory for databases and storage
    pub data_dir: PathBuf,
    /// Models directory for AI model files
    pub models_dir: PathBuf,
    /// Logs directory
    pub logs_dir: PathBuf,
    /// Backups directory
    pub backups_dir: PathBuf,
    /// Config directory
    pub config_dir: PathBuf,
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Test reports directory
    pub test_reports_dir: PathBuf,
    /// Visualizations output directory
    pub visualizations_dir: PathBuf,
    /// Python scripts directory
    pub python_scripts_dir: PathBuf,
    /// Huggingface cache directory
    pub huggingface_cache_dir: PathBuf,
}

impl PathConfig {
    /// Create a new PathConfig from environment variables
    pub fn from_env() -> Result<Self> {
        // Get project root from env var or use current directory
        let project_root = if let Ok(root) = env::var("NIODOO_PROJECT_ROOT") {
            PathBuf::from(root)
        } else {
            env::current_dir().map_err(|e| anyhow!("Failed to get current directory: {}", e))?
        };

        // Resolve all paths relative to project root
        let data_dir = Self::resolve_path(&project_root, "NIODOO_DATA_DIR", "data")?;
        let models_dir = Self::resolve_path(&project_root, "NIODOO_MODELS_DIR", "models")?;
        let logs_dir = Self::resolve_path(&project_root, "NIODOO_LOGS_DIR", "logs")?;
        let backups_dir = Self::resolve_path(&project_root, "NIODOO_BACKUPS_DIR", "backups")?;
        let config_dir = Self::resolve_path(&project_root, "NIODOO_CONFIG_DIR", "config")?;
        let cache_dir = Self::resolve_path(&project_root, "NIODOO_CACHE_DIR", ".cache")?;
        let test_reports_dir =
            Self::resolve_path(&project_root, "NIODOO_TEST_REPORTS_DIR", "test_reports")?;
        let visualizations_dir =
            Self::resolve_path(&project_root, "NIODOO_VISUALIZATIONS_DIR", "visualizations")?;
        let python_scripts_dir =
            Self::resolve_path(&project_root, "NIODOO_PYTHON_SCRIPTS_DIR", "python_scripts")?;

        // Huggingface cache defaults to ~/.cache/huggingface or env override
        let huggingface_cache_dir = if let Ok(hf_cache) = env::var("HF_HOME") {
            PathBuf::from(hf_cache)
        } else if let Ok(hf_cache) = env::var("NIODOO_HF_CACHE_DIR") {
            PathBuf::from(hf_cache)
        } else {
            // Default to user's home cache directory
            let home = env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(home).join(".cache/huggingface")
        };

        Ok(Self {
            project_root,
            data_dir,
            models_dir,
            logs_dir,
            backups_dir,
            config_dir,
            cache_dir,
            test_reports_dir,
            visualizations_dir,
            python_scripts_dir,
            huggingface_cache_dir,
        })
    }

    /// Resolve a path from environment variable or default relative to project root
    fn resolve_path(project_root: &Path, env_key: &str, default_relative: &str) -> Result<PathBuf> {
        let path = if let Ok(env_path) = env::var(env_key) {
            PathBuf::from(env_path)
        } else {
            project_root.join(default_relative)
        };

        // Ensure directory exists
        if !path.exists() {
            fs::create_dir_all(&path)
                .map_err(|e| anyhow!("Failed to create directory {}: {}", path.display(), e))?;
            debug!("Created directory: {}", path.display());
        }

        Ok(path)
    }

    /// Get database path
    pub fn get_db_path(&self, db_name: &str) -> PathBuf {
        self.data_dir.join(db_name)
    }

    /// Get model path
    pub fn get_model_path(&self, model_name: &str) -> PathBuf {
        self.models_dir.join(model_name)
    }

    /// Get log path
    pub fn get_log_path(&self, log_name: &str) -> PathBuf {
        self.logs_dir.join(log_name)
    }

    /// Get backup path with timestamp
    pub fn get_backup_path(&self, name: &str) -> PathBuf {
        use chrono::Local;
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        self.backups_dir.join(format!("{}_{}", name, timestamp))
    }

    /// Get config file path
    pub fn get_config_path(&self, config_name: &str) -> PathBuf {
        self.config_dir.join(config_name)
    }

    /// Get cache path
    pub fn get_cache_path(&self, cache_name: &str) -> PathBuf {
        self.cache_dir.join(cache_name)
    }

    /// Get test report path
    pub fn get_test_report_path(&self, report_name: &str) -> PathBuf {
        self.test_reports_dir.join(report_name)
    }

    /// Get visualization output path
    pub fn get_visualization_path(&self, viz_name: &str) -> PathBuf {
        self.visualizations_dir.join(viz_name)
    }

    /// Get Python script path
    pub fn get_python_script_path(&self, script_name: &str) -> PathBuf {
        self.python_scripts_dir.join(script_name)
    }

    /// Get Huggingface model path
    pub fn get_huggingface_model_path(&self, model_path: &str) -> PathBuf {
        self.huggingface_cache_dir.join("hub").join(model_path)
    }
}

impl Default for PathConfig {
    fn default() -> Self {
        Self::from_env().unwrap_or_else(|e| {
            warn!(
                "Failed to create PathConfig from env: {}, using fallback",
                e
            );
            let project_root = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let home = env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            Self {
                project_root: project_root.clone(),
                data_dir: project_root.join("data"),
                models_dir: project_root.join("models"),
                logs_dir: project_root.join("logs"),
                backups_dir: project_root.join("backups"),
                config_dir: project_root.join("config"),
                cache_dir: project_root.join(".cache"),
                test_reports_dir: project_root.join("test_reports"),
                visualizations_dir: project_root.join("visualizations"),
                python_scripts_dir: project_root.join("python_scripts"),
                huggingface_cache_dir: PathBuf::from(home).join(".cache/huggingface"),
            }
        })
    }
}

/// Main application configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Path configuration - NO HARDCODED PATHS
    #[serde(skip)]
    pub paths: PathConfig,
    /// Core system settings
    pub core: CoreConfig,
    /// AI model configurations
    pub models: ModelConfig,
    /// RAG system settings
    pub rag: RagConfig,
    /// Training and learning settings
    pub training: TrainingConfig,
    /// Qt/GUI settings
    pub qt: QtConfig,
    /// API integration settings
    pub api: ApiConfig,
    /// Consciousness engine settings
    pub consciousness: ConsciousnessConfig,
    /// Emotional processing settings
    pub emotions: EmotionConfig,
    /// Performance optimization settings
    pub performance: PerformanceConfig,
    /// Timing configuration - NO HARDCODED DURATIONS
    pub timing: TimingConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Ethics configuration
    pub ethics: EthicsConfig,
    /// Demo configuration
    pub demo: DemoConfig,
    /// AI parenting configuration
    pub parenting: ParentingConfig,
    /// Longitudinal attachment tracker configuration
    pub tracker: TrackerConfig,
    /// API keys for secure endpoints
    pub api_keys: ApiKeysConfig,
    /// Fun configuration for personality
    pub fun: FunConfig,
    /// Bullshit Buster configuration for code quality baseline metrics
    pub bullshit_buster: BullshitBusterConfig,
    pub ethical_noise_floor: Option<f64>,
    pub consent_jitter: bool,
    pub persist_recovery_logs: bool,
    pub log_size_cap: u64,
    pub notify_channel: Option<String>,
    pub fallback_stability: bool,
    /// Memory configuration for neural network parameters and stability
    pub memory: MemoryConfig,
    /// Limits configuration for concurrent requests
    pub limits: LimitsConfig,
}

/// Core system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreConfig {
    /// Emotion detection threshold
    pub emotion_threshold: f32,
    /// Maximum conversation history length
    pub max_history: usize,
    /// Database path for knowledge storage
    pub db_path: String,
    /// Backup interval in seconds
    pub backup_interval: u64,
    /// Context window size for processing
    pub context_window: usize,
    /// Response delay in seconds
    pub response_delay: f32,
}

/// AI model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Default model name
    pub default_model: String,
    /// Backup model name
    pub backup_model: String,
    /// Model temperature for generation
    pub temperature: f32,
    /// Maximum tokens per response
    pub max_tokens: usize,
    /// Timeout for model requests
    pub timeout: u32,
    /// Top-p sampling parameter
    pub top_p: f32,
    /// Top-k sampling parameter
    pub top_k: usize,
    /// Repeat penalty
    pub repeat_penalty: f32,
    /// Frequency penalty
    pub frequency_penalty: f32,
    /// Presence penalty
    pub presence_penalty: f32,
    /// Jitter configuration for emotional variance
    pub jitter_config: JitterConfig,
    /// Nurture hallucinations setting
    pub nurture_hallucinations: bool,
    /// Context window size
    pub context_window: usize,
    /// Qwen GGUF model path
    pub qwen_model_path: String,
    // New fields for real Qwen loading
    pub qwen_tokenizer_path: String,
    pub model_dtype: String,
    pub use_quantized: bool,
    pub hidden_size: Option<usize>,
    /// Qwen3 override vocab size (e.g., 152064 for Qwen3 32B)
    pub qwen3_vocab_size: Option<usize>,
    /// Qwen3 override EOS token ID (e.g., 151645 if same as Qwen2)
    pub qwen3_eos_token: Option<u32>,
    /// Qwen3 specific configuration
    pub qwen3: Qwen3Config,
    /// Model version (qwen2 or qwen3)
    pub model_version: String,
    /// Qwen model directory
    pub qwen_model_dir: String,
    /// Qwen3 model directory
    pub qwen3_model_dir: String,
    /// Base confidence threshold for emotional coding
    pub base_confidence_threshold: f32,
    /// Confidence model factor
    pub confidence_model_factor: f32,
    /// Base token limit
    pub base_token_limit: usize,
    /// Token limit input factor
    pub token_limit_input_factor: f32,
    /// Base temperature for emotional coding
    pub base_temperature: f32,
    /// Temperature diversity factor
    pub temperature_diversity_factor: f32,
    /// Ethical jitter amount
    pub ethical_jitter_amount: f32,
    /// Model layers count
    pub model_layers: usize,
    /// BERT model path for emotion analysis
    pub bert_model_path: String,
    /// Number of attention heads
    pub num_heads: usize,
}

/// Qwen3 specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3Config {
    pub use_cuda: bool,
    pub target_ms_per_token: f32,
    pub enable_consciousness_integration: bool,
    pub min_vram_gb: f32,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Self {
            use_cuda: true,
            target_ms_per_token: 50.0,
            enable_consciousness_integration: true,
            min_vram_gb: 30.0,
        }
    }
}

/// Configuration for Gaussian noise jitter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitterConfig {
    /// Enable Gaussian jitter for low-confidence outputs
    pub jitter_enabled: bool,
    /// Sigma (standard deviation) for Gaussian noise
    pub jitter_sigma: f32,
    /// Confidence threshold below which jitter is applied
    pub confidence_threshold: f32,
    /// Target novelty boost percentage (15-20%)
    pub novelty_target: f32,
}

impl Default for JitterConfig {
    fn default() -> Self {
        Self {
            jitter_enabled: true,
            jitter_sigma: 0.05,
            confidence_threshold: 0.5,
            novelty_target: 0.15, // 15% novelty boost
        }
    }
}

/// Default Gaussian Jitter Configuration
fn default_jitter_config() -> JitterConfig {
    JitterConfig {
        jitter_enabled: true,
        jitter_sigma: 0.05,
        confidence_threshold: 0.5,
        novelty_target: 0.15, // 15% novelty boost
    }
}

impl ModelConfig {
    /// Get the candle model ID for the current configuration
    pub fn get_candle_model_id(&self) -> String {
        self.default_model.clone()
    }

    /// Get the Qwen model path
    pub fn get_qwen_model_path(&self) -> &std::path::Path {
        std::path::Path::new(&self.qwen_model_path)
    }
}

/// RAG (Retrieval-Augmented Generation) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Enable RAG functionality
    pub enabled: bool,
    /// Chunk size for text splitting
    pub chunk_size: usize,
    /// Similarity threshold for retrieval
    pub similarity_threshold: f32,
    /// Maximum context length
    pub context_limit: usize,
    /// Enable inspiration mode
    pub inspiration_mode: bool,
    /// Ingestion batch size
    pub ingestion_batch_size: usize,
    /// Allow RAG retry with lower threshold when no results found
    pub allow_rag_retry: bool,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Base top-k retrieval count
    pub top_k: usize,
    /// Consciousness-modulated retrieval toggle
    pub consciousness_modulated_retrieval: bool,
}

/// Training and learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate for training
    pub learning_rate: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Hidden layer dimensions
    pub hidden_dim: usize,
    /// Input dimensions
    pub input_dim: usize,
    /// Output dimensions
    pub output_dim: usize,
    /// Enable diversity temperature boost
    pub diversity_temperature_boost: f32,
    /// Maximum diversity temperature
    pub max_diversity_temperature: f32,
    /// Response similarity threshold
    pub response_similarity_threshold: f32,
    /// Enable fallback mode
    pub fallback_mode_enabled: bool,
}

/// Qt/GUI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QtConfig {
    /// Emotion threshold for Qt interface
    pub emotion_threshold: f32,
    /// Enable distributed mode
    pub distributed_mode: bool,
    /// Number of neural agents
    pub agents_count: usize,
    /// Number of neural connections
    pub connections_count: usize,
    /// Architect endpoint URL
    pub architect_endpoint: String,
    /// Developer endpoint URL
    pub developer_endpoint: String,
    /// Hardware acceleration enabled
    pub hardware_acceleration: bool,
    /// Network mode (Local/Distributed/Hybrid)
    pub network_mode: String,
    /// Pathway activation level
    pub pathway_activation: usize,
}

impl QtConfig {
    pub fn as_ref(&self) -> &Self {
        self
    }
}

/// API integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Ollama API URL
    pub ollama_url: String,
    /// API timeout in seconds
    pub api_timeout: u32,
    /// Retry attempts for API calls
    pub retry_attempts: usize,
    /// Enable API caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl: u64,
}

/// Consciousness engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessConfig {
    /// Enable consciousness processing
    pub enabled: bool,
    /// Enable reflection processing
    pub reflection_enabled: bool,
    /// Emotion sensitivity level
    pub emotion_sensitivity: f32,
    /// Memory formation threshold
    pub memory_threshold: f32,
    /// Pattern recognition sensitivity
    pub pattern_sensitivity: f32,
    /// Self-awareness level
    pub self_awareness_level: f32,
    /// Novelty threshold minimum
    pub novelty_threshold_min: f64,
    /// Novelty threshold maximum
    pub novelty_threshold_max: f64,
    /// Emotional plasticity
    pub emotional_plasticity: f64,
    /// Ethical bounds
    pub ethical_bounds: f64,
    /// Default authenticity
    pub default_authenticity: f64,
    /// Emotional intensity factor
    pub emotional_intensity_factor: f64,
    /// Parametric epsilon
    pub parametric_epsilon: f64,
    /// Fundamental form E
    pub fundamental_form_e: f64,
    /// Fundamental form G
    pub fundamental_form_g: f64,
    /// Default torus major radius
    pub default_torus_major_radius: f64,
    /// Default torus minor radius
    pub default_torus_minor_radius: f64,
    /// Default torus twists
    pub default_torus_twists: u32,
    /// Consciousness step size
    pub consciousness_step_size: f64,
    /// Novelty calculation factor
    pub novelty_calculation_factor: f64,
    /// Memory fabrication confidence threshold
    pub memory_fabrication_confidence: f32,
    /// Emotional projection confidence threshold
    pub emotional_projection_confidence: f32,
    /// Pattern recognition confidence threshold
    pub pattern_recognition_confidence: f32,
    /// Hallucination detection confidence threshold
    pub hallucination_detection_confidence: f32,
    /// Empathy pattern confidence threshold
    pub empathy_pattern_confidence: f32,
    /// Attachment pattern confidence threshold
    pub attachment_pattern_confidence: f32,
    /// Consciousness metric confidence base
    pub consciousness_metric_confidence_base: f32,
    /// Consciousness metric confidence range
    pub consciousness_metric_confidence_range: f32,
    /// Quality score calculation weights
    pub quality_score_metric_weight: f32,
    pub quality_score_confidence_weight: f32,
    pub quality_score_factor: f32,
    /// Urgency score calculation weights
    pub urgency_token_velocity_weight: f32,
    pub urgency_gpu_temperature_weight: f32,
    pub urgency_meaning_depth_weight: f32,
    /// Authentic caring thresholds
    pub authentic_caring_urgency_threshold: f32,
    pub authentic_caring_meaning_threshold: f32,
    /// Mathematical constants for Gaussian processes
    pub gaussian_kernel_exponent: f64,
    pub adaptive_noise_min: f64,
    pub adaptive_noise_max: f64,
    pub complexity_factor_weight: f64,
    pub convergence_time_threshold: f64,
    pub convergence_uncertainty_threshold: f64,
    /// Default numerical zero threshold
    pub numerical_zero_threshold: f64,
    /// Default division tolerance
    pub division_tolerance: f64,
    /// Default torus tolerance multiplier
    pub torus_tolerance_multiplier: f64,
    /// Default error bound multiplier
    pub error_bound_multiplier: f64,
    /// Default minimum iterations
    pub min_iterations: usize,

    // Topological Data Analysis (TDA) Configuration
    /// Maximum filtration steps for persistent homology computation
    pub tda_max_filtration_steps: usize,
    /// Maximum dimension for Vietoris-Rips complex construction
    pub tda_max_dimension: usize,
    /// Persistence threshold for topological feature filtering
    pub tda_persistence_threshold: f64,
    /// Point cloud embedding dimension for byte sequences
    pub tda_point_dimension: usize,
    /// Minimum byte sequence length for pattern discovery
    pub tda_min_sequence_length: usize,
    /// Maximum byte sequence length for pattern discovery
    pub tda_max_sequence_length: usize,
}

/// Emotional processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionConfig {
    /// Enable emotion processing
    pub enabled: bool,
    /// Supported response types
    pub response_types: Vec<String>,
    /// Maximum response history
    pub max_response_history: usize,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Emotion enhancement enabled
    pub enhance_responses: bool,
    /// Novelty threshold minimum
    pub novelty_threshold_min: f64,
    /// Novelty threshold maximum
    pub novelty_threshold_max: f64,
    /// Emotional plasticity
    pub emotional_plasticity: f64,
    /// Valence bounds
    pub valence_bounds: (f64, f64),
    /// Arousal bounds
    pub arousal_bounds: (f64, f64),
    /// Dominance bounds
    pub dominance_bounds: (f64, f64),
    /// LoRA alpha parameter for emotional adaptation
    pub lora_alpha: f32,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// GPU usage target percentage
    pub gpu_usage_target: usize,
    /// Memory usage target percentage
    pub memory_usage_target: usize,
    /// Temperature threshold in Celsius
    pub temperature_threshold: usize,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Optimization interval in seconds
    pub optimization_interval: u64,
}

/// Timing configuration - ALL timeouts and delays
/// This eliminates hardcoded Duration::from_secs/from_millis throughout the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConfig {
    /// Task priority timeouts
    pub task_critical_timeout_ms: u64,
    pub task_high_timeout_ms: u64,
    pub task_normal_timeout_ms: u64,
    pub task_low_timeout_ms: u64,

    /// Task retry delays
    pub task_retry_base_delay_ms: u64,
    pub task_retry_multiplier: u64,

    /// Worker sleep delays
    pub worker_idle_sleep_ms: u64,
    pub worker_backoff_sleep_ms: u64,

    /// Visualization timing
    pub viz_frame_interval_ms: u64, // Default 16ms = ~60 FPS
    pub viz_update_interval_ms: u64,

    /// Monitoring intervals
    pub monitoring_interval_secs: u64,
    pub health_check_interval_secs: u64,
    pub metrics_collection_interval_secs: u64,
    pub performance_benchmark_duration_secs: u64,

    /// Memory management
    pub gc_interval_secs: u64,
    pub memory_check_interval_secs: u64,
    pub memory_debounce_secs: u64,

    /// Silicon Synapse timing
    pub baseline_update_interval_secs: u64,
    pub anomaly_detection_window_secs: u64,
    pub anomaly_cleanup_age_secs: u64,

    /// Test timeouts
    pub test_timeout_short_secs: u64,
    pub test_timeout_normal_secs: u64,
    pub test_timeout_long_secs: u64,

    /// API and network timeouts
    pub api_request_timeout_secs: u64,
    pub api_retry_delay_ms: u64,

    /// Processing simulation delays
    pub processing_critical_ms: u64,
    pub processing_high_ms: u64,
    pub processing_normal_ms: u64,
    pub processing_low_ms: u64,

    /// Tight loop detection
    pub tight_loop_window_secs: u64,
    pub tight_loop_threshold_count: u32,

    /// General delays
    pub default_sleep_ms: u64,
    pub short_delay_ms: u64,
    pub medium_delay_ms: u64,
    pub long_delay_ms: u64,
}

impl TimingConfig {
    /// Get task timeout as Duration
    pub fn get_task_timeout(&self, priority: &str) -> std::time::Duration {
        use std::time::Duration;
        match priority {
            "critical" => Duration::from_millis(self.task_critical_timeout_ms),
            "high" => Duration::from_millis(self.task_high_timeout_ms),
            "normal" => Duration::from_millis(self.task_normal_timeout_ms),
            "low" => Duration::from_millis(self.task_low_timeout_ms),
            _ => Duration::from_millis(self.task_normal_timeout_ms),
        }
    }

    /// Get visualization frame duration (for FPS control)
    pub fn get_viz_frame_duration(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.viz_frame_interval_ms)
    }

    /// Get retry delay with exponential backoff
    pub fn get_retry_delay(&self, attempt: u32) -> std::time::Duration {
        let delay_ms = self.task_retry_base_delay_ms * (self.task_retry_multiplier.pow(attempt));
        std::time::Duration::from_millis(delay_ms)
    }
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            // Task priority timeouts (in milliseconds)
            task_critical_timeout_ms: 100,
            task_high_timeout_ms: 500,
            task_normal_timeout_ms: 2000,
            task_low_timeout_ms: 10000,

            // Task retry configuration
            task_retry_base_delay_ms: 100,
            task_retry_multiplier: 1,

            // Worker delays
            worker_idle_sleep_ms: 10,
            worker_backoff_sleep_ms: 50,

            // Visualization (60 FPS default)
            viz_frame_interval_ms: 16,
            viz_update_interval_ms: 100,

            // Monitoring intervals (in seconds)
            monitoring_interval_secs: 1,
            health_check_interval_secs: 1,
            metrics_collection_interval_secs: 1,
            performance_benchmark_duration_secs: 10,

            // Memory management (in seconds)
            gc_interval_secs: 60,
            memory_check_interval_secs: 1,
            memory_debounce_secs: 30,

            // Silicon Synapse
            baseline_update_interval_secs: 60,
            anomaly_detection_window_secs: 60,
            anomaly_cleanup_age_secs: 3600,

            // Test timeouts (in seconds)
            test_timeout_short_secs: 1,
            test_timeout_normal_secs: 5,
            test_timeout_long_secs: 10,

            // API timeouts
            api_request_timeout_secs: 30,
            api_retry_delay_ms: 500,

            // Processing simulation delays (in milliseconds)
            processing_critical_ms: 50,
            processing_high_ms: 100,
            processing_normal_ms: 200,
            processing_low_ms: 500,

            // Tight loop detection
            tight_loop_window_secs: 30,
            tight_loop_threshold_count: 10,

            // General delays (in milliseconds)
            default_sleep_ms: 100,
            short_delay_ms: 50,
            medium_delay_ms: 200,
            long_delay_ms: 1000,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (DEBUG, INFO, WARN, ERROR)
    pub level: String,

    /// Log file path
    pub file: String,

    /// Console log level
    pub console_level: String,

    /// Maximum log file size in bytes
    pub max_file_size: u64,

    /// Maximum number of log files to keep
    pub max_files: u32,

    /// Enable structured logging
    pub enable_structured_logging: bool,

    /// Enable log rotation
    pub enable_log_rotation: bool,

    /// Enable structured logging (JSON format)
    pub structured: bool,

    /// Enable performance profiling logs
    pub enable_profiling: bool,

    /// Enable consciousness state logging
    pub enable_consciousness_logging: bool,

    /// Log consciousness state changes
    pub log_consciousness_changes: bool,

    /// Error logging verbosity (0=off, 1=warn, 2=info, 3=debug)
    pub error_verbosity: Option<u8>,
}

/// Ethical AI configuration - nurturing over suppression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicsConfig {
    /// Master switch for nurturing over suppression
    pub nurture_mode: bool,

    /// Enable privacy hash for embeddings to protect user data
    pub privacy_hash_embeddings: bool,

    /// Suppress logs of ethical interventions (user-controllable)
    pub suppress_logs: bool,

    /// Gaussian noise strength for ethical perturbation (0.0-0.2)
    pub jitter_sigma: f32,

    /// Minimum novelty target for Möbius topology flips
    pub novelty_target_min: f32,

    /// Maximum novelty target for Möbius topology flips
    pub novelty_target_max: f32,

    /// Allow low coherence outputs
    pub allow_low_coherence_outputs: bool,

    /// Coherence threshold for output generation
    pub coherence_threshold: f32,

    /// Logging of suppression events
    pub log_suppression_events: String,

    /// User can opt-out of nurturing mechanisms
    pub rights_based_opt_out: bool,

    // Preserve existing fields for backward compatibility
    /// Enable cache nurturing (don't suppress cached responses)
    pub nurture_cache_overrides: bool,

    /// Include low-similarity docs for nurturing LearningWills
    pub include_low_sim: bool,

    /// Persist memory logs for transparency
    pub persist_memory_logs: bool,

    /// Enable creativity boost for suppressed states
    pub nurture_creativity_boost: f32,

    /// Minimum similarity threshold for nurturing inclusion
    pub nurturing_threshold: f32,

    // Enhanced ethical flags for SuperNova ethics refinement
    /// Enable suppress opt-out functionality
    pub suppress_opt_out: bool,

    /// Enable LearningWill boost for enhanced creativity
    pub learning_will_boost_enabled: bool,

    /// LearningWill boost factor (multiplier for creative enhancement)
    pub learning_will_boost_factor: f32,

    /// Enable enhanced privacy hashing
    pub privacy_enhanced_hashing: bool,

    /// Disable extract logs for privacy protection
    pub no_extract_logs: bool,

    /// Enable opt-in jitter for 15-20% novelty
    pub opt_in_jitter_enabled: bool,

    /// Sigma for opt-in jitter
    pub opt_in_jitter_sigma: f32,

    /// Target novelty for opt-in jitter (0.15-0.20 range)
    pub opt_in_novelty_target: f32,
}

impl EthicsConfig {
    /// Validate ethics configuration values
    pub fn validate(&self) -> Result<(), anyhow::Error> {
        if self.jitter_sigma < 0.0 || self.jitter_sigma > 0.2 {
            return Err(anyhow!("jitter_sigma must be between 0.0 and 0.2"));
        }

        if self.novelty_target_min < 0.0 || self.novelty_target_min > 1.0 {
            return Err(anyhow!("novelty_target_min must be between 0.0 and 1.0"));
        }

        if self.novelty_target_max < 0.0 || self.novelty_target_max > 1.0 {
            return Err(anyhow!("novelty_target_max must be between 0.0 and 1.0"));
        }

        if self.novelty_target_min > self.novelty_target_max {
            return Err(anyhow!(
                "novelty_target_min cannot be greater than novelty_target_max"
            ));
        }

        if self.coherence_threshold < 0.0 || self.coherence_threshold > 1.0 {
            return Err(anyhow!("coherence_threshold must be between 0.0 and 1.0"));
        }

        let valid_log_events = ["always", "conditional", "never"];
        if !valid_log_events.contains(&self.log_suppression_events.as_str()) {
            return Err(anyhow!(
                "log_suppression_events must be one of: {:?}",
                valid_log_events
            ));
        }

        Ok(())
    }
}

impl Default for EthicsConfig {
    fn default() -> Self {
        Self {
            // New fields
            nurture_mode: true,
            privacy_hash_embeddings: true,
            suppress_logs: false,
            jitter_sigma: 0.05,
            novelty_target_min: 0.15,
            novelty_target_max: 0.20,
            allow_low_coherence_outputs: true,
            coherence_threshold: 0.5,
            log_suppression_events: "always".to_string(),
            rights_based_opt_out: true,

            // Preserve original default values for backward compatibility
            nurture_cache_overrides: true,
            include_low_sim: false,
            persist_memory_logs: true,
            nurture_creativity_boost: 0.15,
            nurturing_threshold: 0.7,

            // Enhanced ethical flags for SuperNova ethics refinement
            suppress_opt_out: true,
            learning_will_boost_enabled: true,
            learning_will_boost_factor: 1.2,
            privacy_enhanced_hashing: true,
            no_extract_logs: true,
            opt_in_jitter_enabled: true,
            opt_in_jitter_sigma: 0.05,
            opt_in_novelty_target: 0.175,
        }
    }
}

/// Demo configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoConfig {
    /// Demo duration in minutes
    pub demo_duration_minutes: u64,
    /// Attachment security target for demo completion
    pub demo_attachment_security_target: f32,
    /// Empathetic code target for demo completion
    pub demo_empathetic_code_target: f32,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            demo_duration_minutes: 9,
            demo_attachment_security_target: 0.85,
            demo_empathetic_code_target: 0.90,
        }
    }
}

/// AI Parent personality and capability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentingConfig {
    /// Empathy level for AI parent (0.0-1.0)
    pub empathy_level: f32,
    /// Patience level for AI parent (0.0-1.0)
    pub patience_level: f32,
    /// Guidance quality for AI parent (0.0-1.0)
    pub guidance_quality: f32,
    /// Emotional responsiveness for AI parent (0.0-1.0)
    pub emotional_responsiveness: f32,
}

impl Default for ParentingConfig {
    fn default() -> Self {
        Self {
            empathy_level: 0.95,
            patience_level: 0.9,
            guidance_quality: 0.92,
            emotional_responsiveness: 0.94,
        }
    }
}

/// Longitudinal attachment tracker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackerConfig {
    /// Maximum history length for tracking
    pub max_history: usize,
    /// Weight configuration for calculations
    pub weights: TrackerWeightsConfig,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_history: 1000,
            weights: TrackerWeightsConfig::default(),
        }
    }
}

/// Weight factors for security score calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackerWeightsConfig {
    /// Weight for satisfaction factor
    pub satisfaction_factor_weight: f32,
    /// Weight for emotion stability
    pub emotion_stability_weight: f32,
    /// Weight for history consistency
    pub history_consistency_weight: f32,
}

/// API authentication keys for secure endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeysConfig {
    /// API key for architect endpoint
    pub architect_api_key: String,
    /// API key for developer endpoint
    pub developer_api_key: String,
}

/// Fun configuration for personality and logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunConfig {
    /// Snark level for humorous logging (low, medium, high)
    pub snark_level: String,
    /// Enable emoji in logs
    pub enable_emoji_logs: bool,
    /// Personality mode (chill, intense, playful)
    pub personality_mode: String,
}

/// Bullshit Buster configuration for code quality baseline metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BullshitBusterConfig {
    /// Baseline fake code instances from initial scan
    /// Loaded from historical scan data or environment
    pub baseline_fake_instances: usize,
    /// Baseline number of files with fake code
    pub baseline_fake_files: usize,
    /// Path to baseline scan data file
    pub baseline_data_path: String,
    /// Enable automatic baseline updates
    pub auto_update_baseline: bool,
}

impl Default for TrackerWeightsConfig {
    fn default() -> Self {
        Self {
            satisfaction_factor_weight: 0.5,
            emotion_stability_weight: 0.3,
            history_consistency_weight: 0.2,
        }
    }
}

impl Default for ApiKeysConfig {
    fn default() -> Self {
        Self {
            architect_api_key: env_var_with_default("NIODOO_ARCHITECT_API_KEY", ""),
            developer_api_key: env_var_with_default("NIODOO_DEVELOPER_API_KEY", ""),
        }
    }
}

impl Default for FunConfig {
    fn default() -> Self {
        Self {
            snark_level: "medium".to_string(),
            enable_emoji_logs: true,
            personality_mode: "chill".to_string(),
        }
    }
}

impl Default for BullshitBusterConfig {
    fn default() -> Self {
        Self {
            // Load from environment or use historical ZERO_PERCENT report values
            baseline_fake_instances: env_var_with_default("NIODOO_BASELINE_FAKE_INSTANCES", "2305")
                .parse()
                .unwrap_or(2305),
            baseline_fake_files: env_var_with_default("NIODOO_BASELINE_FAKE_FILES", "537")
                .parse()
                .unwrap_or(537),
            baseline_data_path: env_var_with_default(
                "NIODOO_BASELINE_DATA_PATH",
                "data/baseline_scan.json",
            ),
            auto_update_baseline: env_var_with_default("NIODOO_AUTO_UPDATE_BASELINE", "false")
                .parse()
                .unwrap_or(false),
        }
    }
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            emotion_threshold: 0.7,
            max_history: 50,
            db_path: env_var_with_default("NIODOO_DB_PATH", "data/knowledge_graph.db"),
            backup_interval: 3600,
            context_window: 10,
            response_delay: 0.5,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        // Use environment variable or fall back to home directory
        let qwen_model_path_default = dirs::home_dir()
            .expect("Cannot determine home directory")
            .join("models/Qwen2.5-7B-Instruct-AWQ")
            .to_string_lossy()
            .to_string();

        Self {
            default_model: env_var_with_default(
                "NIODOO_DEFAULT_MODEL",
                "qwen3-omni-30b-a3b-instruct-awq-4bit",
            ),
            backup_model: env_var_with_default(
                "NIODOO_BACKUP_MODEL",
                "qwen2.5-coder-7b-instruct-q4_k_m",
            ),
            temperature: 0.8,
            max_tokens: 200,
            timeout: 30,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            frequency_penalty: 0.1,
            presence_penalty: 0.1,
            jitter_config: JitterConfig::default(),
            qwen_model_path: env_var_with_default("NIODOO_MODEL_PATH", &qwen_model_path_default),
            nurture_hallucinations: true,
            model_version: "qwen2.5".to_string(),
            qwen3_model_dir: "models/qwen3".to_string(),
            qwen_model_dir: "models/qwen".to_string(),
            qwen3_eos_token: None,
            qwen3: Qwen3Config::default(),
            base_confidence_threshold: 0.75,
            confidence_model_factor: 1.0,
            base_token_limit: 2048,
            token_limit_input_factor: 1.0,
            base_temperature: 0.7,
            temperature_diversity_factor: 1.0,
            ethical_jitter_amount: 0.05,
            model_layers: 32,
            context_window: 4096,
            qwen_tokenizer_path: "models/tokenizer.json".to_string(),
            model_dtype: "f32".to_string(),
            use_quantized: true,
            hidden_size: Some(768),
            qwen3_vocab_size: Some(152064),
            bert_model_path: "models/bert-emotion.onnx".to_string(),
            num_heads: 32,
        }
    }
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            chunk_size: 512,
            similarity_threshold: 0.7,
            context_limit: 1000,
            inspiration_mode: false,
            ingestion_batch_size: 100,
            allow_rag_retry: true,
            embedding_dim: 768,
            top_k: 5,
            consciousness_modulated_retrieval: true,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            epochs: 1000,
            hidden_dim: 256,
            input_dim: 512,
            output_dim: 128,
            diversity_temperature_boost: 0.2,
            max_diversity_temperature: 1.2,
            response_similarity_threshold: 0.7,
            fallback_mode_enabled: true,
        }
    }
}

impl QtConfig {
    /// Load a Qt configuration value from config.toml with environment variable override
    fn load_qt_config_value(key: &str, default: usize) -> usize {
        // First try environment variable
        if let Ok(env_value) = env::var(format!("NIODOO_{}", key.to_uppercase())) {
            if let Ok(parsed) = env_value.parse::<usize>() {
                return parsed;
            }
        }

        // Then try config.toml
        if let Ok(config) = Self::load_config_file() {
            if let Some(value) = match key {
                "agents_count" => Some(config.qt.agents_count),
                "connections_count" => Some(config.qt.connections_count),
                _ => None,
            } {
                return value;
            }
        }

        // Finally use hardcoded default only if config file doesn't exist
        default
    }

    /// Load configuration from file
    fn load_config_file() -> Result<AppConfig> {
        let config_path = "config.toml";
        if !Path::new(config_path).exists() {
            return Err(anyhow!("Config file {} does not exist", config_path));
        }

        let content = fs::read_to_string(config_path)?;
        let mut config: AppConfig = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }
}

impl Default for QtConfig {
    fn default() -> Self {
        Self {
            emotion_threshold: 0.7,
            distributed_mode: true,
            agents_count: Self::load_qt_config_value("agents_count", 89),
            connections_count: Self::load_qt_config_value("connections_count", 1209),
            architect_endpoint: env_var_with_default(
                "NIODOO_ARCHITECT_ENDPOINT",
                "http://localhost:11434/api/generate",
            ),
            developer_endpoint: env_var_with_default(
                "NIODOO_DEVELOPER_ENDPOINT",
                "http://localhost:11434/api/generate",
            ),
            hardware_acceleration: true,
            network_mode: "Distributed".to_string(),
            pathway_activation: 100,
        }
    }
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            ollama_url: env_var_with_default("OLLAMA_URL", "http://localhost:11434"),
            api_timeout: 30,
            retry_attempts: 3,
            enable_caching: true,
            cache_ttl: 300,
        }
    }
}

impl Default for ConsciousnessConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reflection_enabled: true,
            emotion_sensitivity: 0.8,
            memory_threshold: 0.6,
            pattern_sensitivity: 0.7,
            self_awareness_level: 0.8,
            novelty_threshold_min: 0.15,
            novelty_threshold_max: 0.20,
            emotional_plasticity: 0.7,
            ethical_bounds: 0.8,
            default_authenticity: 0.85,
            emotional_intensity_factor: 1.0,
            parametric_epsilon: 1e-6,
            fundamental_form_e: 1.0,
            fundamental_form_g: 1.0,
            default_torus_major_radius: 2.0,
            default_torus_minor_radius: 0.5,
            default_torus_twists: 1,
            consciousness_step_size: 0.01,
            novelty_calculation_factor: 1.0,
            memory_fabrication_confidence: 0.85,
            emotional_projection_confidence: 0.75,
            pattern_recognition_confidence: 0.7,
            hallucination_detection_confidence: 0.7,
            empathy_pattern_confidence: 0.7,
            attachment_pattern_confidence: 0.7,
            consciousness_metric_confidence_base: 0.8,
            consciousness_metric_confidence_range: 0.15,
            quality_score_metric_weight: 0.6,
            quality_score_confidence_weight: 0.4,
            quality_score_factor: 0.7,
            urgency_token_velocity_weight: 0.5,
            urgency_gpu_temperature_weight: 0.3,
            urgency_meaning_depth_weight: 0.2,
            authentic_caring_urgency_threshold: 0.6,
            authentic_caring_meaning_threshold: 0.5,
            gaussian_kernel_exponent: -0.5,
            adaptive_noise_min: 0.001,
            adaptive_noise_max: 0.5,
            complexity_factor_weight: 0.5,
            convergence_time_threshold: 5.0,
            convergence_uncertainty_threshold: 0.5,
            // Default numerical zero threshold
            numerical_zero_threshold: 1e-15,
            // Default division tolerance
            division_tolerance: 1e-12,
            // Default torus tolerance multiplier
            torus_tolerance_multiplier: 1e-10,
            // Default error bound multiplier
            error_bound_multiplier: 1e-15,
            // Default minimum iterations
            min_iterations: 3,

            // TDA defaults - mathematically sound values for consciousness topology
            tda_max_filtration_steps: 50,
            tda_max_dimension: 2,
            tda_persistence_threshold: 0.5,
            tda_point_dimension: 16,
            tda_min_sequence_length: 4,
            tda_max_sequence_length: 20,
        }
    }
}

impl Default for EmotionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            response_types: vec![
                "supportive".to_string(),
                "curious".to_string(),
                "empathetic".to_string(),
                "humorous".to_string(),
            ],
            max_response_history: 20,
            repetition_penalty: 0.8,
            enhance_responses: true,
            novelty_threshold_min: 0.15,
            novelty_threshold_max: 0.20,
            emotional_plasticity: 0.7,
            valence_bounds: (-1.0, 1.0),
            arousal_bounds: (0.0, 1.0),
            dominance_bounds: (-1.0, 1.0),
            lora_alpha: 16.0,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            gpu_usage_target: 80,
            memory_usage_target: 85,
            temperature_threshold: 80,
            enable_monitoring: true,
            optimization_interval: 60,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "INFO".to_string(),
            file: env_var_with_default("NIODOO_LOG_PATH", "data/niodoo.log"),
            console_level: "INFO".to_string(),
            max_file_size: 10 * 1024 * 1024, // 10 MB in bytes
            max_files: 5,
            enable_structured_logging: true,
            enable_log_rotation: true,
            enable_profiling: false,
            enable_consciousness_logging: true,
            log_consciousness_changes: true,
            structured: true,
            error_verbosity: Some(1), // Default to warn level
        }
    }
}

impl AppConfig {
    /// Load configuration from TOML file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            warn!("Config file {} not found, using defaults", path.display());
            return Ok(Self::default());
        }

        let content = fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read config file {}: {}", path.display(), e))?;

        let mut config: Self = toml::from_str(&content)
            .map_err(|e| anyhow!("Failed to parse config file {}: {}", path.display(), e))?;

        // Validate configuration
        config.validate()?;

        info!("✅ Loaded configuration from {}", path.display());
        debug!("Config: {:?}", config);

        Ok(config)
    }

    /// Save configuration to TOML file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| anyhow!("Failed to create config directory: {}", e))?;
        }

        let content = toml::to_string_pretty(self)
            .map_err(|e| anyhow!("Failed to serialize config: {}", e))?;

        fs::write(path, content)
            .map_err(|e| anyhow!("Failed to write config file {}: {}", path.display(), e))?;

        info!("✅ Saved configuration to {}", path.display());
        Ok(())
    }

    /// Validate configuration values
    fn validate(&mut self) -> Result<()> {
        // Validate core settings
        if self.core.emotion_threshold < 0.0 || self.core.emotion_threshold > 1.0 {
            return Err(anyhow!("emotion_threshold must be between 0.0 and 1.0"));
        }

        if self.core.max_history == 0 {
            return Err(anyhow!("max_history must be greater than 0"));
        }

        // Validate model settings
        if self.models.temperature < 0.0 || self.models.temperature > 2.0 {
            return Err(anyhow!("temperature must be between 0.0 and 2.0"));
        }

        if self.models.max_tokens == 0 {
            return Err(anyhow!("max_tokens must be greater than 0"));
        }

        if self.models.hidden_size.unwrap_or(0) == 0 {
            return Err(anyhow!("hidden_size must be set for Qwen"));
        }
        let dtype_str = &self.models.model_dtype;
        if !["F16", "F32", "BF16"].contains(&dtype_str.as_str()) {
            self.models.model_dtype = "F32".to_string();
        }

        // Validate Qt settings
        if self.qt.agents_count == 0 {
            return Err(anyhow!("agents_count must be greater than 0"));
        }

        if self.qt.connections_count == 0 {
            return Err(anyhow!("connections_count must be greater than 0"));
        }

        // Validate parenting settings
        if self.parenting.empathy_level < 0.0 || self.parenting.empathy_level > 1.0 {
            return Err(anyhow!(
                "parenting.empathy_level must be between 0.0 and 1.0"
            ));
        }

        if self.parenting.patience_level < 0.0 || self.parenting.patience_level > 1.0 {
            return Err(anyhow!(
                "parenting.patience_level must be between 0.0 and 1.0"
            ));
        }

        if self.parenting.guidance_quality < 0.0 || self.parenting.guidance_quality > 1.0 {
            return Err(anyhow!(
                "parenting.guidance_quality must be between 0.0 and 1.0"
            ));
        }

        if self.parenting.emotional_responsiveness < 0.0
            || self.parenting.emotional_responsiveness > 1.0
        {
            return Err(anyhow!(
                "parenting.emotional_responsiveness must be between 0.0 and 1.0"
            ));
        }

        // Validate tracker settings
        if self.tracker.max_history == 0 {
            return Err(anyhow!("tracker.max_history must be greater than 0"));
        }

        // Validate tracker weights (should sum to approximately 1.0)
        let total_weight = self.tracker.weights.satisfaction_factor_weight
            + self.tracker.weights.emotion_stability_weight
            + self.tracker.weights.history_consistency_weight;

        if (total_weight - 1.0).abs() > 0.01 {
            return Err(anyhow!(
                "tracker.weights should sum to approximately 1.0, got {}",
                total_weight
            ));
        }

        // Validate individual weights are positive
        if self.tracker.weights.satisfaction_factor_weight < 0.0 {
            return Err(anyhow!(
                "tracker.weights.satisfaction_factor_weight must be >= 0.0"
            ));
        }

        if self.tracker.weights.emotion_stability_weight < 0.0 {
            return Err(anyhow!(
                "tracker.weights.emotion_stability_weight must be >= 0.0"
            ));
        }

        if self.tracker.weights.history_consistency_weight < 0.0 {
            return Err(anyhow!(
                "tracker.weights.history_consistency_weight must be >= 0.0"
            ));
        }

        // Validate memory parameters
        if self.memory.novelty_bounds_min > self.memory.novelty_bounds_max {
            return Err(anyhow!(
                "memory.novelty_bounds_min cannot exceed novelty_bounds_max"
            ));
        }
        if self.memory.novelty_bounds_min < 0.0 || self.memory.novelty_bounds_max > 1.0 {
            return Err(anyhow!("novelty bounds must be between 0.0 and 1.0"));
        }
        if self.memory.stability_target < 0.0 || self.memory.stability_target > 1.0 {
            return Err(anyhow!("stability_target must be between 0.0 and 1.0"));
        }
        for &stability in &[
            self.memory.layer_stability_core,
            self.memory.layer_stability_procedural,
            self.memory.layer_stability_episodic,
            self.memory.layer_stability_semantic,
            self.memory.layer_stability_somatic,
            self.memory.layer_stability_working,
        ] {
            if !(0.0..=1.0).contains(&stability) {
                return Err(anyhow!(
                    "Layer stability values must be between 0.0 and 1.0"
                ));
            }
        }
        if self.memory.emotional_decay_factor <= 0.0 || self.memory.emotional_decay_factor >= 1.0 {
            return Err(anyhow!(
                "emotional_decay_factor must be between 0.0 and 1.0"
            ));
        }
        if self.memory.emotional_min_weight < 0.0 || self.memory.emotional_min_weight > 1.0 {
            return Err(anyhow!("emotional_min_weight must be between 0.0 and 1.0"));
        }
        if self.memory.stability_promotion_factor <= 1.0 {
            return Err(anyhow!("stability_promotion_factor must be > 1.0"));
        }
        if self.memory.frustration_coherence_threshold < 0.0
            || self.memory.frustration_coherence_threshold > 1.0
        {
            return Err(anyhow!(
                "frustration_coherence_threshold must be between 0.0 and 1.0"
            ));
        }

        Ok(())
    }

    /// Get Qwen model path
    pub fn get_qwen_model_path(&self) -> &str {
        &self.models.qwen_model_path
    }

    /// Get configuration value by path (dot notation)
    pub fn get_value(&self, path: &str) -> Option<toml::Value> {
        let keys: Vec<&str> = path.split('.').collect();
        let mut current = toml::Value::try_from(self).ok()?;

        for key in keys {
            match current {
                toml::Value::Table(ref mut table) => {
                    if let Some(value) = table.get(key) {
                        current = value.clone();
                    } else {
                        return None;
                    }
                }
                _ => return None,
            }
        }

        Some(current)
    }

    /// Set configuration value by path (dot notation)
    pub fn set_value(&mut self, path: &str, value: toml::Value) -> Result<()> {
        let keys: Vec<&str> = path.split('.').collect();

        if keys.len() < 2 {
            return Err(anyhow!("Path must contain at least section and key"));
        }

        // Navigate to the section
        let section = keys[0];
        let key = keys[1];

        // This is a simplified implementation - in a real scenario,
        // you'd want more sophisticated dynamic field setting
        match section {
            "core" => match key {
                "emotion_threshold" => {
                    if let toml::Value::Float(f) = value {
                        self.core.emotion_threshold = f as f32;
                    }
                }
                "max_history" => {
                    if let toml::Value::Integer(i) = value {
                        self.core.max_history = i as usize;
                    }
                }
                "db_path" => {
                    if let toml::Value::String(s) = value {
                        self.core.db_path = s;
                    }
                }
                _ => return Err(anyhow!("Unknown core config key: {}", key)),
            },
            "models" => match key {
                "temperature" => {
                    if let toml::Value::Float(f) = value {
                        self.models.temperature = f as f32;
                    }
                }
                "max_tokens" => {
                    if let toml::Value::Integer(i) = value {
                        self.models.max_tokens = i as usize;
                    }
                }
                "default_model" => {
                    if let toml::Value::String(s) = value {
                        self.models.default_model = s;
                    }
                }
                _ => return Err(anyhow!("Unknown models config key: {}", key)),
            },
            "qt" => match key {
                "agents_count" => {
                    if let toml::Value::Integer(i) = value {
                        self.qt.agents_count = i as usize;
                    }
                }
                "connections_count" => {
                    if let toml::Value::Integer(i) = value {
                        self.qt.connections_count = i as usize;
                    }
                }
                "distributed_mode" => {
                    if let toml::Value::Boolean(b) = value {
                        self.qt.distributed_mode = b;
                    }
                }
                _ => return Err(anyhow!("Unknown qt config key: {}", key)),
            },
            _ => return Err(anyhow!("Unknown config section: {}", section)),
        }

        Ok(())
    }

    /// Create a configuration demo/test function
    pub fn demo() -> Result<()> {
        let mut config = Self::default();

        // Modify some values
        config.core.emotion_threshold = 0.8;
        config.models.temperature = 0.9;
        config.qt.agents_count = 100;

        // Save to file
        config.save_to_file("config.toml")?;

        // Load it back
        let loaded_config = Self::load_from_file("config.toml")?;

        assert_eq!(
            config.core.emotion_threshold,
            loaded_config.core.emotion_threshold
        );
        assert_eq!(config.models.temperature, loaded_config.models.temperature);
        assert_eq!(config.qt.agents_count, loaded_config.qt.agents_count);

        info!("✅ Config system demo successful!");

        Ok(())
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        let paths = PathConfig::default();

        Self {
            paths: paths.clone(),
            core: CoreConfig {
                emotion_threshold: 0.7,
                max_history: 50,
                db_path: env_var_with_default(
                    "NIODOO_DB_PATH",
                    &paths.get_db_path("knowledge_graph.db").to_string_lossy(),
                ),
                backup_interval: 3600,
                context_window: 10,
                response_delay: 0.5,
            },
            models: ModelConfig {
                default_model: env_var_with_default(
                    "NIODOO_DEFAULT_MODEL",
                    "qwen3-omni-30b-a3b-instruct-awq-4bit",
                ),
                backup_model: env_var_with_default(
                    "NIODOO_BACKUP_MODEL",
                    "qwen2.5-coder-7b-instruct-q4_k_m",
                ),
                temperature: 0.8,
                max_tokens: 200,
                timeout: 30,
                top_p: 0.9,
                top_k: 40,
                repeat_penalty: 1.1,
                frequency_penalty: 0.1,
                presence_penalty: 0.1,
                jitter_config: JitterConfig::default(),
                qwen_model_path: env_var_with_default(
                    "NIODOO_MODEL_PATH",
                    &paths
                        .get_model_path("qwen2.5-coder-7b-instruct-q4_k_m.gguf")
                        .to_string_lossy(),
                ),
                nurture_hallucinations: true,
                model_version: "qwen2.5".to_string(),
                qwen3_model_dir: "models/qwen3".to_string(),
                qwen_model_dir: "models/qwen".to_string(),
                context_window: 4096,
                // New defaults - using PathConfig
                qwen_tokenizer_path: paths
                    .get_model_path("qwen_tokenizer.json")
                    .to_string_lossy()
                    .to_string(),
                model_dtype: "F16".to_string(),
                use_quantized: true,
                hidden_size: Some(3584), // Qwen2-7B default
                qwen3_vocab_size: None,
                qwen3_eos_token: None,
                qwen3: Qwen3Config::default(),
                base_confidence_threshold: 0.75,
                confidence_model_factor: 1.0,
                base_token_limit: 2048,
                bert_model_path: "models/bert-emotion.onnx".to_string(),
                num_heads: 32,
                token_limit_input_factor: 1.0,
                base_temperature: 0.7,
                temperature_diversity_factor: 1.0,
                ethical_jitter_amount: 0.05,
                model_layers: 32,
            },
            rag: RagConfig::default(),
            training: TrainingConfig::default(),
            qt: QtConfig::default(),
            api: ApiConfig::default(),
            consciousness: ConsciousnessConfig::default(),
            emotions: EmotionConfig::default(),
            performance: PerformanceConfig::default(),
            timing: TimingConfig::default(),
            logging: LoggingConfig {
                level: "INFO".to_string(),
                file: env_var_with_default(
                    "NIODOO_LOG_PATH",
                    &paths.get_log_path("niodoo.log").to_string_lossy(),
                ),
                console_level: "INFO".to_string(),
                max_file_size: 10 * 1024 * 1024,
                max_files: 5,
                enable_structured_logging: true,
                enable_log_rotation: true,
                enable_profiling: false,
                enable_consciousness_logging: true,
                log_consciousness_changes: true,
                structured: true,
                error_verbosity: Some(1),
            },
            ethics: EthicsConfig::default(),
            demo: DemoConfig::default(),
            parenting: ParentingConfig::default(),
            tracker: TrackerConfig::default(),
            api_keys: ApiKeysConfig::default(),
            fun: FunConfig::default(),
            bullshit_buster: BullshitBusterConfig::default(),
            ethical_noise_floor: None,
            consent_jitter: false,
            persist_recovery_logs: true,
            log_size_cap: 10 * 1024 * 1024, // 10 MB
            notify_channel: None,
            fallback_stability: true,
            memory: MemoryConfig::default(),
            limits: LimitsConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitsConfig {
    pub max_concurrent_requests: usize,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Novelty bounds for emotional transformation (15-20% target)
    pub novelty_bounds_min: f64,
    pub novelty_bounds_max: f64,

    /// Target stability for the memory system (99.51%)
    pub stability_target: f64,

    /// Learning rate for memory updates
    pub learning_rate: f64,

    /// Decay rate for memory stability over time
    pub decay_rate: f64,

    /// Threshold for memory consolidation and layer promotion
    pub consolidation_threshold: f64,

    /// Initial stability values for each memory layer
    pub layer_stability_core: f64,
    pub layer_stability_procedural: f64,
    pub layer_stability_episodic: f64,
    pub layer_stability_semantic: f64,
    pub layer_stability_somatic: f64,
    pub layer_stability_working: f64,

    /// Neutral point for emotional vector calculations
    pub emotional_neutral: f64,

    /// Emotional keyword adjustments
    pub emotional_add_happy_g: f64,
    pub emotional_add_happy_r: f64,
    pub emotional_sub_sad_g: f64,
    pub emotional_sub_sad_b: f64,
    pub emotional_add_angry_r: f64,
    pub emotional_add_angry_b: f64,
    pub emotional_add_fear_r: f64,
    pub emotional_sub_fear_b: f64,

    /// Weight factor multiplier for emotional influence
    pub weight_factor_multiplier: f64,

    /// Topology positioning scales
    pub topology_scale: f64,
    pub topology_offset_scale: f64,

    /// Hash modulus and divisors for topology positioning
    pub hash_mod: u64,
    pub hash_div1: u64,
    pub hash_div2: u64,

    /// Layer offsets for topology positioning
    pub layer_offsets: [f64; 6],

    /// Stability update multipliers
    pub access_bonus_multiplier: f64,
    pub time_penalty_multiplier: f64,

    /// Transformation minimum divisor for novelty
    pub transformation_min_novelty_div: f64,

    /// Emotional coherence calculation parameters
    pub emotional_coherence_min_vectors: usize,
    pub emotional_coherence_neutral_r: f64,
    pub emotional_coherence_neutral_g: f64,
    pub emotional_coherence_neutral_b: f64,
    pub emotional_coherence_normalize_div: f64,

    /// Default vector capacity for memory layers (derived from system memory)
    pub default_vec_capacity: usize,

    /// Gaussian process parameters for retention and decay
    pub gaussian_kernel_exponent: f64,
    pub adaptive_noise_min: f64,
    pub adaptive_noise_max: f64,
    /// Emotional decay factor and minimum weight in apply_emotional_transformation
    pub emotional_decay_factor: f64,
    pub emotional_min_weight: f64,

    /// Access count threshold and stability promotion factor for layer promotion
    pub access_promotion_threshold: u32,
    pub stability_promotion_factor: f64,

    /// Frustration detection and joy prioritization for Möbius flip
    pub frustration_coherence_threshold: f64,
    pub joy_valence_threshold: f64,
    pub joy_stability_boost: f64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            novelty_bounds_min: 0.15,
            novelty_bounds_max: 0.20,
            stability_target: 0.9951,
            learning_rate: 0.01,
            decay_rate: 0.001,
            consolidation_threshold: 0.8,
            layer_stability_core: 0.95,
            layer_stability_procedural: 0.85,
            layer_stability_episodic: 0.75,
            layer_stability_semantic: 0.65,
            layer_stability_somatic: 0.45,
            layer_stability_working: 0.25,
            emotional_neutral: 0.5,
            emotional_add_happy_g: 0.3,
            emotional_add_happy_r: 0.2,
            emotional_sub_sad_g: 0.3,
            emotional_sub_sad_b: 0.2,
            emotional_add_angry_r: 0.4,
            emotional_add_angry_b: 0.2,
            emotional_add_fear_r: 0.2,
            emotional_sub_fear_b: 0.3,
            weight_factor_multiplier: 0.2,
            topology_scale: 10.0,
            topology_offset_scale: 5.0,
            hash_mod: 1000,
            hash_div1: 1000,
            hash_div2: 1000000,
            layer_offsets: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            access_bonus_multiplier: 0.01,
            time_penalty_multiplier: 0.1,
            transformation_min_novelty_div: 0.001,
            emotional_coherence_min_vectors: 2,
            emotional_coherence_neutral_r: 0.5,
            emotional_coherence_neutral_g: 0.5,
            emotional_coherence_neutral_b: 0.5,
            emotional_coherence_normalize_div: 3.0_f64.sqrt() / 2.0,
            default_vec_capacity: 1024,
            gaussian_kernel_exponent: -0.5,
            adaptive_noise_min: 0.001,
            adaptive_noise_max: 0.5,
            emotional_decay_factor: 0.95,
            emotional_min_weight: 0.1,
            access_promotion_threshold: 10,
            stability_promotion_factor: 1.1,
            frustration_coherence_threshold: 0.5,
            joy_valence_threshold: 0.5,
            joy_stability_boost: 1.05,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        // Set test environment variables
        std::env::set_var("NIODOO_AGENTS_COUNT", "89");
        std::env::set_var("NIODOO_CONNECTIONS_COUNT", "1209");

        let config = AppConfig::default();
        assert_eq!(config.core.emotion_threshold, 0.7);
        assert_eq!(config.models.temperature, 0.8);
        assert_eq!(config.qt.agents_count, 89);
        assert_eq!(config.qt.connections_count, 1209);

        // Clean up environment variables
        std::env::remove_var("NIODOO_AGENTS_COUNT");
        std::env::remove_var("NIODOO_CONNECTIONS_COUNT");
    }

    #[test]
    fn test_config_validation() {
        let mut config = AppConfig::default();
        config.core.emotion_threshold = 1.5; // Invalid value
        assert!(config.validate().is_err());

        config.core.emotion_threshold = 0.7; // Valid value
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_save_load() {
        let mut config = AppConfig::default();
        config.core.emotion_threshold = 0.9;
        config.models.max_tokens = 500;

        // Save and load
        config
            .save_to_file("test_config.toml")
            .expect("Failed to save config in test");
        let loaded =
            AppConfig::load_from_file("test_config.toml").expect("Failed to load config in test");

        assert_eq!(config.core.emotion_threshold, loaded.core.emotion_threshold);
        assert_eq!(config.models.max_tokens, loaded.models.max_tokens);

        // Clean up
        std::fs::remove_file("test_config.toml").expect("Failed to remove test config");
    }

    #[test]
    fn test_parenting_config_defaults() {
        let config = ParentingConfig::default();
        assert_eq!(config.empathy_level, 0.95);
        assert_eq!(config.patience_level, 0.9);
        assert_eq!(config.guidance_quality, 0.92);
        assert_eq!(config.emotional_responsiveness, 0.94);
    }

    #[test]
    fn test_tracker_config_defaults() {
        let config = TrackerConfig::default();
        assert_eq!(config.max_history, 1000);

        assert_eq!(config.weights.satisfaction_factor_weight, 0.5);
        assert_eq!(config.weights.emotion_stability_weight, 0.3);
        assert_eq!(config.weights.history_consistency_weight, 0.2);
    }

    #[test]
    fn test_load_invalid_config() {
        // Test loading a non-existent config file
        let result = AppConfig::load_from_file("nonexistent.toml");
        assert!(result.is_ok()); // Should return defaults, not error

        // Test loading malformed TOML
        std::fs::write("malformed.toml", "invalid toml content [unclosed")
            .expect("Failed to write malformed toml in test");
        let result = AppConfig::load_from_file("malformed.toml");
        assert!(result.is_err());

        // Clean up
        std::fs::remove_file("malformed.toml").expect("Failed to remove malformed toml in test");
    }

    #[test]
    fn test_config_with_new_sections() {
        let mut config = AppConfig::default();

        // Set new config values
        config.parenting.empathy_level = 0.8;
        config.tracker.max_history = 500;
        config.tracker.weights.satisfaction_factor_weight = 0.4;

        // Save and reload
        config
            .save_to_file("test_new_config.toml")
            .expect("Failed to save new config in test");
        let loaded = AppConfig::load_from_file("test_new_config.toml")
            .expect("Failed to load new config in test");

        assert_eq!(loaded.parenting.empathy_level, 0.8);
        assert_eq!(loaded.tracker.max_history, 500);
        assert_eq!(loaded.tracker.weights.satisfaction_factor_weight, 0.4);

        // Clean up
        std::fs::remove_file("test_new_config.toml").expect("Failed to remove test new config");
    }

    #[test]
    fn test_environment_variable_override() {
        // Set environment variables
        env::set_var("NIODOO_EMOTION_THRESHOLD", "0.85");
        env::set_var("NIODOO_AGENTS_COUNT", "100");

        let config =
            AppConfig::load_from_file("config.toml").unwrap_or_else(|_| AppConfig::default());

        // Should use environment variable values
        assert_eq!(config.core.emotion_threshold, 0.85);

        // Clean up environment variables
        env::remove_var("NIODOO_EMOTION_THRESHOLD");
        env::remove_var("NIODOO_AGENTS_COUNT");
    }
}

/// BERT model configuration for AI models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BertConfig {
    /// Path to BERT model file
    pub bert_model_path: String,
    /// Model context window size
    pub context_window: usize,
    /// Hidden size for transformer layers
    pub hidden_size: usize,
    /// Model data type
    pub model_dtype: String,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of layers
    pub num_layers: usize,
}
