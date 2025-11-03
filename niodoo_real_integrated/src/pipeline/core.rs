use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

use crate::compass::{CascadeTracker, CompassEngine, CompassOutcome, CompassQuadrant};
use crate::config::{
    env_value, set_env_override, CliArgs, CuratorConfig, HardwareProfile, RuntimeConfig,
    TopologyMode,
};
use crate::consonance::{compute_consonance, ConsonanceMetrics};
use crate::curator::Curator;
use crate::data::{
    compute_dataset_stats, load_emotional_dataset, load_rut_gauntlet_prompts, DatasetStats,
    Experience, RutPrompt,
};
use crate::embedding::QwenStatefulEmbedder;
use crate::erag::{CollapseResult, EragClient};
use crate::generation::{GenerationEngine, GenerationResult};
use crate::gpu_fitness::GPUMemoryFitnessCalculator;
use crate::hyperfocus::{HyperfocusDetector, HyperfocusEvent};
use crate::learning::{LearningLoop, LearningOutcome};
use crate::mcts::MctsDaydreamer;
use crate::memory_consolidation::MemoryConsolidationManager;
use crate::metrics::{metrics, weighted_memory_metrics};
use crate::security::PromptSecurityManager;
use crate::signals::FailureSignals;
use crate::tcs_analysis::{TCSAnalyzer, TopologicalSignature};
use crate::rce::analyzer::RceAnalyzer;
use crate::token_manager::{DynamicTokenizerManager, TokenizerOutput};
use crate::topology_memory::TopologyMemoryAnalyzer;
use crate::torus::{PadGhostState, TorusPadMapper};
use crate::util::{rouge_l, seed_manager, set_global_seed};
use crate::weight_evolution::{Discovery, SmoothWeightEvolution};
use parking_lot::RwLock;
#[allow(unused_imports)]
use qdrant_client::qdrant::{CreateCollection, Distance, VectorsConfig};
use rand::RngCore;
use tcs_core::PersistentFeature;
use tokio::sync::Mutex as AsyncMutex;
use tracing::{info, warn};

use crate::pipeline::cache::{cache_key, PipelineCache};
use crate::pipeline::metrics::StageTimings;
use crate::pipeline::state::{CuratedExperience, CuratorFeedbackController, PipelineCycle, Thresholds, TorusSeedStrategy};

// Proto module - include generated proto code from OUT_DIR during build
#[allow(dead_code)]
pub mod proto {
    // Stub proto module - requires build.rs to generate niodoo.rs
    // For now, define minimal types needed for compilation
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct ConsciousnessState {
        pub entropy: f64,
        pub quadrant: String,
        pub threat: bool,
        pub healing: bool,
    }
}

pub struct Pipeline {
    pub config: RuntimeConfig,
    pub(crate) config_arc: Arc<RwLock<RuntimeConfig>>,
    pub args: CliArgs,
    pub thresholds: Thresholds,
    pub dataset_stats: DatasetStats,
    pub(crate) embedder: QwenStatefulEmbedder,
    torus_strategy: TorusSeedStrategy,
    torus_counter: AtomicU64,
    pub(crate) compass: Arc<AsyncMutex<CompassEngine>>,
    pub(crate) erag: Arc<EragClient>,
    pub(crate) tokenizer: Arc<DynamicTokenizerManager>,
    pub(crate) generator: GenerationEngine,
    pub(crate) learning: AsyncMutex<LearningLoop>,
    pub(crate) curator: Option<Curator>,
    pub(crate) tcs_analyzer: Option<TCSAnalyzer>,
    pub(crate) rce_analyzer: Option<RceAnalyzer>,
    pub(crate) embedding_cache: PipelineCache<Vec<f32>>,
    pub(crate) collapse_cache: PipelineCache<CollapseResult>,
    pub(crate) retry_count: Arc<AtomicU32>,
    pub(crate) cascade_tracker: Arc<AsyncMutex<CascadeTracker>>, // Cascade tracking
    pub(crate) hyperfocus_detector: Arc<HyperfocusDetector>,     // Hyperfocus detection
    pub(crate) last_compass_outcome: Arc<AsyncMutex<Option<CompassOutcome>>>, // Track last compass for cascade
    #[allow(dead_code)]
    qdrant_process: Option<Arc<tokio::sync::Mutex<tokio::process::Child>>>,
    // Weighted Episodic Memory components
    weight_evolver: Arc<SmoothWeightEvolution>,
    gpu_fitness_calculator: Arc<GPUMemoryFitnessCalculator>,
    topology_analyzer: Arc<TopologyMemoryAnalyzer>,
    consolidation_manager: Arc<AsyncMutex<MemoryConsolidationManager>>,
    mcts_daydreamer: Arc<MctsDaydreamer>,
    discovery_queue: Arc<AsyncMutex<tokio::sync::mpsc::UnboundedSender<Discovery>>>,
    security: Arc<PromptSecurityManager>,
    // Phase 4.2: Curator feedback controller
    pub(crate) curator_feedback: Option<Arc<AsyncMutex<CuratorFeedbackController>>>,
    // RCE: spike circuit breaker streak counter
    pub(crate) rce_spike_streak: Arc<AtomicU32>,
}

impl Pipeline {
    pub async fn initialise(args: CliArgs) -> Result<Self> {
        Self::initialise_with_topology(args, None, None).await
    }

    pub async fn initialise_with_mode(args: CliArgs, mode: TopologyMode) -> Result<Self> {
        Self::initialise_with_topology(args, Some(mode), None).await
    }

    async fn initialise_with_topology(
        mut args: CliArgs,
        override_mode: Option<TopologyMode>,
        seed_override: Option<u64>,
    ) -> Result<Self> {
        if let Some(seed) = seed_override {
            args.rng_seed_override = Some(seed);
            set_env_override("RNG_SEED", seed.to_string());
        }

        let mut config = RuntimeConfig::load(&args)?;
        if let Some(seed) = seed_override {
            config.rng_seed = seed;
        }
        if let Some(mode) = override_mode {
            config.topology_mode = mode;
        }

        let samples =
            load_emotional_dataset(&config.training_data_path, config.training_data_sample_cap)?;
        let stats = compute_dataset_stats(&samples);

        let thresholds = Thresholds {
            entropy_mean: stats.entropy_mean,
            entropy_high: stats.entropy_mean + stats.entropy_std,
            variance_stagnation: config.variance_stagnation_default,
            variance_spike: stats.variance_std.max(config.variance_spike_min),
            mirage_sigma: config.mirage_sigma_factor * stats.entropy_mean,
            mcts_c: stats.entropy_std.max(config.mcts_c_min_std) * config.mcts_c_scale,
        };

        // Mock mode is handled via MOCK_MODE environment variable in QwenStatefulEmbedder::new
        if config.mock_mode {
            std::env::set_var("MOCK_MODE", "true");
        }
        let embedder =
            QwenStatefulEmbedder::new(&config.embedding_model_name, config.qdrant_vector_dim)?;
        // Note: Candle support would need to be added to QwenStatefulEmbedder if needed
        info!(
            endpoint = %config.ollama_endpoint,
            model = %config.embedding_model_name,
            mock_mode = config.mock_mode,
            "Initialized embedding client"
        );
        let embedder_arc = Arc::new(embedder.clone());
        let torus_strategy = if let Some(seed) = seed_override {
            info!(
                seed,
                "Initializing torus pad mapper with fixed seed override"
            );
            TorusSeedStrategy::Fixed(seed)
        } else if let Some(value) = env_value("TORUS_SEED") {
            match value.parse::<u64>() {
                Ok(seed) => {
                    info!(seed, "Initializing torus pad mapper with fixed seed");
                    TorusSeedStrategy::Fixed(seed)
                }
                Err(_) => {
                    warn!(value = %value, "Invalid TORUS_SEED value; using entropy seeding");
                    TorusSeedStrategy::Random
                }
            }
        } else {
            info!("Initializing torus pad mapper with entropy seed");
            TorusSeedStrategy::Random
        };
        let compass = Arc::new(AsyncMutex::new(CompassEngine::new(
            thresholds.mcts_c,
            thresholds.variance_spike,
            thresholds.variance_stagnation,
        )));

        // Optional embedded Qdrant startup (spawns Qdrant as child process)
        #[cfg(feature = "embedded-qdrant")]
        let qdrant_process: Option<Arc<tokio::sync::Mutex<tokio::process::Child>>> = if config
            .qdrant_embedded
        {
            info!("QDRANT_EMBEDDED enabled: spawning embedded Qdrant process");
            match crate::embedded_qdrant::spawn_embedded_qdrant().await {
                Ok(child) => Some(Arc::new(tokio::sync::Mutex::new(child))),
                Err(e) => {
                    warn!(error = %e, "Failed to spawn embedded Qdrant; falling back to external Qdrant");
                    None
                }
            }
        } else {
            None
        };

        #[cfg(not(feature = "embedded-qdrant"))]
        let qdrant_process: Option<Arc<tokio::sync::Mutex<tokio::process::Child>>> = None;

        // Initialize Weighted Episodic Memory components ahead of ERAG setup
        let weighted_config = &config.weighted_memory_config;
        let weight_evolver = Arc::new(SmoothWeightEvolution::new());
        let gpu_fitness_calculator =
            Arc::new(GPUMemoryFitnessCalculator::new(&weighted_config.gpu_device));
        let gpu_fitness_calc = if config.use_gpu_fitness {
            Some(gpu_fitness_calculator.clone())
        } else {
            None
        };

        let erag = if config.optimized_erag {
            EragClient::new_with_config_and_quantization(
                &config.qdrant_url,
                &config.qdrant_collection,
                config.qdrant_vector_dim,
                config.similarity_threshold,
                config.optimized_erag,
                config.erag_batch_size,
                config.erag_batch_flush_ms,
                config.qdrant_quantization,
                gpu_fitness_calc.clone(), // Phase 4.3: Pass GPU calculator
            )
            .await?
        } else {
            EragClient::new_with_config(
                &config.qdrant_url,
                &config.qdrant_collection,
                config.qdrant_vector_dim,
                config.similarity_threshold,
                config.optimized_erag,
                config.erag_batch_size,
                config.erag_batch_flush_ms,
                gpu_fitness_calc.clone(), // Phase 4.3: Pass GPU calculator
            )
            .await?
        };

        // Log collection state for diagnostics
        if !config.mock_mode {
            if let Err(e) = erag.check_collection_info().await {
                warn!(error = %e, "Failed to check Qdrant collection info");
            }
        }
        let tokenizer_file = tokenizer_path(&config)?;
        info!(path = %tokenizer_file.display(), "initializing dynamic tokenizer");
        let tokenizer = Arc::new(
            DynamicTokenizerManager::initialise(
                tokenizer_file.as_path(),
                env_value("NODE_ID").unwrap_or_else(|| "niodoo_real_integrated".to_string()),
                config.token_promotion_interval,
            )
            .await?,
        );
        tokenizer.spawn_maintenance().await;
        let mut generator = GenerationEngine::new_with_config(
            &config.vllm_endpoint,
            &config.vllm_model,
            config.generation_max_tokens,
            config.consistency_variance_threshold,
        )?;
        info!(model = %config.vllm_model, endpoint = %config.vllm_endpoint, "Initialized GenerationEngine with vLLM model");
        generator.set_mock_mode(config.mock_mode);
        generator.set_system_prompt(config.system_prompt.clone());
        let config_arc = Arc::new(parking_lot::RwLock::new(config.clone()));
        let security_manager = Arc::new(PromptSecurityManager::new(config.security.clone())?);
        security_manager.audit_config_snapshot(&config);

        // Phase 4.2: Initialize curator feedback controller
        let curator_feedback = Arc::new(AsyncMutex::new(CuratorFeedbackController::new(
            config.curator_quality_threshold,
            20, // window_size: track last 20 curator responses
        )));

        let erag_arc = Arc::new(erag.clone());
        let learning = LearningLoop::new(
            config.learning_window,
            config.breakthrough_threshold,
            config.breakthrough_rouge_min,
            config.dqn_epsilon,
            config.dqn_gamma,
            config.dqn_alpha,
            erag_arc.clone(),
            config_arc.clone(),
            tokenizer.clone(),
            config.rng_seed,
        );

        // Initialize TCS analyzer only when topology mode requires it
        let tcs_analyzer = if matches!(config.topology_mode, TopologyMode::Hybrid) {
            match TCSAnalyzer::new_with_config(config.use_approximate_tda) {
                Ok(analyzer) => {
                    info!(
                        "TCS topology analyzer initialized (approximate_tda: {})",
                        config.use_approximate_tda
                    );
                    Some(analyzer)
                }
                Err(error) => {
                    warn!(%error, "Failed to initialize TCS analyzer; using analytic fallback");
                    None
                }
            }
        } else {
            info!("Topology mode set to baseline; skipping TCS analyzer initialization");
            None
        };

        // Initialize curator if enabled
        let curator = if config.enable_curator {
            let curator_config = CuratorConfig::from_runtime_config(&config);
            match Curator::new(curator_config) {
                Ok(c) => {
                    info!("Curator initialized successfully");
                    Some(c)
                }
                Err(e) => {
                    warn!(
                        "Failed to initialize curator: {}, continuing without curator",
                        e
                    );
                    None
                }
            }
        } else {
            info!("Curator disabled via config");
            None
        };

        let embedding_cache = PipelineCache::new(
            NonZeroUsize::new(1000).unwrap(),
            Duration::from_secs(config.embedding_cache_ttl_secs),
        );
        let collapse_cache = PipelineCache::new(
            NonZeroUsize::new(500).unwrap(),
            Duration::from_secs(config.collapse_cache_ttl_secs),
        );

        let topology_analyzer = Arc::new(TopologyMemoryAnalyzer::new(0.3));
        let consolidation_manager = Arc::new(AsyncMutex::new(MemoryConsolidationManager::new()));
        let mcts_daydreamer = Arc::new(MctsDaydreamer::new(1.414, 5)); // sqrt(2) exploration, depth 5

        // Create discovery queue for async processing
        let (discovery_tx, mut discovery_rx) = tokio::sync::mpsc::unbounded_channel::<Discovery>();
        let discovery_queue = Arc::new(AsyncMutex::new(discovery_tx));

        // Clone components for background tasks
        let weight_evolver_clone = Arc::clone(&weight_evolver);
        let discovery_queue_clone = Arc::clone(&discovery_queue);

        // Spawn background discovery processor
        tokio::spawn(async move {
            let mut discovery_buffer = Vec::new();
            loop {
                tokio::select! {
                    discovery = discovery_rx.recv() => {
                        if let Some(disc) = discovery {
                            discovery_buffer.push(disc);
                            if discovery_buffer.len() >= 10 {
                                // Process batch
                                for disc in discovery_buffer.drain(..) {
                                    weight_evolver_clone.register_discovery(disc).await;
                                }
                            }
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {
                        // Process remaining discoveries every second
                        if !discovery_buffer.is_empty() {
                            for disc in discovery_buffer.drain(..) {
                                weight_evolver_clone.register_discovery(disc).await;
                            }
                        }
                    }
                }
            }
        });

        // Weighted memory GPU evolution background task (Phase 4.3)
        let gpu_fitness_calc_clone = gpu_fitness_calc.clone();
        let erag_refresh = erag_arc.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                if let Some(ref calc) = gpu_fitness_calc_clone {
                    if let Err(e) = calc.refresh_metrics().await {
                        warn!(error = %e, "GPU fitness metrics refresh failed");
                    }
                }
                if let Err(e) = erag_refresh.refresh_weighted_memory().await {
                    warn!(error = %e, "Failed to refresh weighted memory cache");
                }
            }
        });

        // Ensure Prometheus metrics are initialised for observability
        let _ = crate::metrics::metrics();
        let _ = crate::metrics::weighted_memory_metrics();

        Ok(Self {
            config: config.clone(),
            config_arc: config_arc.clone(),
            args,
            thresholds,
            dataset_stats: stats,
            embedder,
            torus_strategy,
            torus_counter: AtomicU64::new(0),
            compass,
            erag: erag_arc.clone(),
            tokenizer: tokenizer.clone(),
            generator,
            learning: AsyncMutex::new(learning),
            curator,
            tcs_analyzer,
            rce_analyzer: None,
            embedding_cache,
            collapse_cache,
            retry_count: Arc::new(AtomicU32::new(0)),
            cascade_tracker: Arc::new(AsyncMutex::new(CascadeTracker::new())),
            hyperfocus_detector: Arc::new(HyperfocusDetector::new()),
            last_compass_outcome: Arc::new(AsyncMutex::new(None)),
            qdrant_process,
            // Weighted Episodic Memory components
            weight_evolver,
            gpu_fitness_calculator,
            topology_analyzer,
            consolidation_manager,
            mcts_daydreamer,
            discovery_queue,
            security: security_manager,
            curator_feedback: Some(curator_feedback), // Phase 4.2: Curator feedback controller
            rce_spike_streak: Arc::new(AtomicU32::new(0)),
        })
    }

    /// Phase 4.2: Helper to adjust runtime parameters based on curator feedback
    pub(crate) fn adjust_runtime_param(
        config: &mut RuntimeConfig,
        param: &str,
        delta: f64,
    ) {
        match param {
            "temperature" => {
                config.temperature = (config.temperature + delta).clamp(0.1, 1.0);
            }
            "top_p" => {
                config.top_p = (config.top_p + delta).clamp(0.1, 1.0);
            }
            "retrieval_top_k" => {
                let updated =
                    (config.phase2_retrieval_top_k_increment as f64 + delta).clamp(0.0, 10.0);
                config.phase2_retrieval_top_k_increment = updated.round() as i32;
            }
            _ => {
                // Unknown parameter, ignore
            }
        }
    }

    pub fn set_topology_mode(&mut self, mode: TopologyMode) -> Result<()> {
        if self.config.topology_mode == mode {
            return Ok(());
        }

        self.config.topology_mode = mode;
        {
            let mut guard = self.config_arc.write();
            guard.topology_mode = mode;
        }

        self.tcs_analyzer = match mode {
            TopologyMode::Hybrid => {
                match TCSAnalyzer::new_with_config(self.config.use_approximate_tda) {
                    Ok(analyzer) => {
                        info!(
                            "TCS analyzer re-initialized for hybrid mode (approximate_tda: {})",
                            self.config.use_approximate_tda
                        );
                        Some(analyzer)
                    }
                    Err(error) => {
                        warn!(%error, "Failed to initialize TCS analyzer; analytic fallback remains active");
                        None
                    }
                }
            }
            TopologyMode::Baseline => {
                info!("Topology mode changed to baseline; disabling TCS analyzer");
                None
            }
        };

        Ok(())
    }

    pub async fn initialise_with_seed(args: CliArgs, seed: u64) -> Result<Self> {
        set_global_seed(seed);
        let manager = seed_manager();
        if manager.master_seed() != seed {
            warn!(
                existing = manager.master_seed(),
                requested = seed,
                "Seed override ignored; global seed already initialised"
            );
        }
        Self::initialise_with_topology(args, None, Some(seed)).await
    }

    pub fn next_torus_mapper(&self) -> TorusPadMapper {
        // Derive a fresh mapper using the global seed manager and a stable scope
        // Include topology mode in scope to ensure baseline and hybrid produce different PAD states
        let counter = self.torus_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let mode_str = match self.config.topology_mode {
            TopologyMode::Baseline => "baseline",
            TopologyMode::Hybrid => "hybrid",
        };
        let scope = match self.torus_strategy {
            TorusSeedStrategy::Fixed(seed) => format!("torus/fixed/{seed}/{mode_str}/{counter}"),
            TorusSeedStrategy::Random => format!("torus/derived/{mode_str}/{counter}"),
        };
        let mut derived_rng = crate::util::seed_manager().get_rng(&scope);
        // Extract u64 seed by sampling from the derived RNG to initialize mapper RNG deterministically
        let derived_seed: u64 = derived_rng.next_u64();
        TorusPadMapper::new(derived_seed)
    }

    /// Recompute thresholds from updated config (called after learning updates)
    pub fn recompute_thresholds(&mut self) {
        let updated_thresholds = Thresholds {
            entropy_mean: self.thresholds.entropy_mean, // Keep static
            entropy_high: self.thresholds.entropy_high, // Keep static
            variance_stagnation: self.config.variance_stagnation_default,
            variance_spike: self
                .dataset_stats
                .variance_std
                .max(self.config.variance_spike_min),
            mirage_sigma: self.config.mirage_sigma_factor * self.dataset_stats.entropy_mean,
            mcts_c: self
                .dataset_stats
                .entropy_std
                .max(self.config.mcts_c_min_std)
                * self.config.mcts_c_scale,
        };
        self.thresholds = updated_thresholds;
    }

    pub fn rut_prompts(&self) -> Vec<RutPrompt> {
        load_rut_gauntlet_prompts()
    }

    pub async fn save_lora_adapter(&self, path: impl AsRef<Path>) -> Result<()> {
        let path_buf = path.as_ref().to_path_buf();
        let guard = self.learning.lock().await;
        guard.save_lora_adapter(&path_buf)?;
        info!(adapter = %path_buf.display(), "Pipeline persisted LoRA adapter");
        Ok(())
    }

    pub async fn load_lora_adapter(&self, path: impl AsRef<Path>) -> Result<()> {
        let path_buf = path.as_ref().to_path_buf();
        let mut guard = self.learning.lock().await;
        guard.load_lora_adapter(&path_buf)?;
        info!(adapter = %path_buf.display(), "Pipeline reloaded LoRA adapter");
        Ok(())
    }

    // NOTE: process_prompt is implemented in stages.rs, not here
    // This prevents duplicate method definitions
}

fn tokenizer_path(config: &RuntimeConfig) -> Result<PathBuf> {
    if let Some(path_str) = config.tokenizer_json.as_deref() {
        let candidate = PathBuf::from(path_str);
        if candidate.exists() {
            return Ok(candidate);
        } else {
            bail!(
                "configured tokenizer_json path does not exist: {}",
                candidate.display()
            );
        }
    }

    tokenizer_path_from_env()
}

fn tokenizer_path_from_env() -> Result<PathBuf> {
    if let Some(value) = env_value("TOKENIZER_JSON") {
        let path = PathBuf::from(value);
        if path.exists() {
            return Ok(path);
        }
    }

    if let Some(value) = env_value("QWEN_TOKENIZER") {
        let path = PathBuf::from(value);
        if path.exists() {
            return Ok(path);
        }
    }

    if let Some(models_dir) = env_value("MODELS_DIR") {
        let path = PathBuf::from(models_dir).join("tokenizer.json");
        if path.exists() {
            return Ok(path);
        }
    }

    // Fallback: try common locations
    let fallback_paths = vec![
        "/workspace/models/Qwen2.5-7B-Instruct-AWQ/tokenizer.json",
        "/workspace/models/Qwen2-0.5B-Instruct/tokenizer.json",
        "./models/tokenizer.json",
        "models/tokenizer.json",
    ];

    for path_str in fallback_paths {
        let path = PathBuf::from(path_str);
        if path.exists() {
            return Ok(path);
        }
    }

    // If VLLM_MODEL_PATH is set, try tokenizer.json in that directory
    if let Some(model_path) = env_value("VLLM_MODEL_PATH") {
        let path = PathBuf::from(&model_path).join("tokenizer.json");
        if path.exists() {
            return Ok(path);
        }
        // Also try parent directory if model_path is a file
        if let Some(parent) = PathBuf::from(&model_path).parent() {
            let path = parent.join("tokenizer.json");
            if path.exists() {
                return Ok(path);
            }
        }
    }

    bail!(
        "tokenizer.json not found; set TOKENIZER_JSON, QWEN_TOKENIZER, or configure runtime.tokenizer_json"
    )
}
fn baseline_topological_signature(
    pad_state: &PadGhostState,
    embedding: &[f32],
) -> TopologicalSignature {
    let analysis_start = Instant::now();

    let pad: Vec<f64> = pad_state.pad.iter().map(|v| *v as f64).collect();
    let mu: Vec<f64> = pad_state.mu.iter().map(|v| *v as f64).collect();
    let sigma: Vec<f64> = pad_state.sigma.iter().map(|v| *v as f64).collect();

    let pad_min = pad.iter().fold(f64::INFINITY, |acc, value| acc.min(*value));
    let pad_max = pad
        .iter()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));
    let mu_min = mu.iter().fold(f64::INFINITY, |acc, value| acc.min(*value));
    let mu_max = mu
        .iter()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));
    let sigma_min = sigma
        .iter()
        .fold(f64::INFINITY, |acc, value| acc.min(*value));
    let sigma_max = sigma
        .iter()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));

    let persistence_features = vec![
        PersistentFeature {
            birth: pad_min as f32,
            death: pad_max as f32,
            dimension: 0,
        },
        PersistentFeature {
            birth: mu_min as f32,
            death: mu_max as f32,
            dimension: 1,
        },
        PersistentFeature {
            birth: sigma_min as f32,
            death: sigma_max as f32,
            dimension: 2,
        },
    ];

    let betti0 = pad.iter().filter(|value| **value >= 0.0).count();
    let betti1 = pad.iter().filter(|value| **value < 0.0).count();
    let sigma_threshold = if sigma.is_empty() {
        0.0
    } else {
        sigma.iter().sum::<f64>() / sigma.len() as f64
    };
    let betti2 = sigma
        .iter()
        .zip(pad_state.sigma.iter())
        .filter(|(sigma_value, raw_std)| {
            **sigma_value > sigma_threshold && **sigma_value > **raw_std
        })
        .count();

    let knot_complexity = if pad.len() > 1 {
        pad.windows(2)
            .map(|window| (window[1] - window[0]).abs())
            .sum::<f64>()
            / (pad.len() - 1) as f64
    } else {
        0.0
    };

    let pad_mean = if pad.is_empty() {
        0.0
    } else {
        pad.iter().sum::<f64>() / pad.len() as f64
    };
    let pad_variance = if pad.len() > 1 {
        pad.iter()
            .map(|value| (value - pad_mean).powi(2))
            .sum::<f64>()
            / (pad.len() - 1) as f64
    } else {
        0.0
    };

    let knot_polynomial = format!("λ² + {:.3}λ + {:.3}", pad_mean, pad_variance);

    let pad_energy = pad
        .iter()
        .map(|value| value.abs())
        .sum::<f64>()
        .max(f64::EPSILON);
    let persistence_entropy = pad
        .iter()
        .map(|value| {
            let p = value.abs() / pad_energy;
            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum::<f64>();

    let mut spectral_basis: Vec<f64> = embedding
        .iter()
        .map(|value| (*value as f64).abs())
        .collect();
    spectral_basis.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let spectral_gap = match spectral_basis.len() {
        0 => 0.0,
        1 => spectral_basis[0],
        _ => spectral_basis[0] - spectral_basis[1],
    };

    let computation_time_ms = analysis_start.elapsed().as_secs_f64() * 1000.0;

    TopologicalSignature::new(
        persistence_features,
        [betti0, betti1, betti2],
        knot_complexity,
        knot_polynomial,
        2,
        None,
        computation_time_ms,
        persistence_entropy,
        spectral_gap,
        0.0, // euler_characteristic - placeholder
        0.0, // total_persistence - placeholder
        0.0, // max_persistence - placeholder
        0.0, // mean_persistence - placeholder
        0.0, // laplacian_spectral_radius - placeholder
    )
}
