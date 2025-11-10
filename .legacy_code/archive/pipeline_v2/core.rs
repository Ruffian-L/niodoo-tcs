use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use anyhow::Result;

use crate::compass::{CascadeTracker, CompassEngine, CompassOutcome, CompassQuadrant};
use crate::config::{
    env_value, set_env_override, CliArgs, CuratorConfig, HardwareProfile, RuntimeConfig,
    TopologyMode,
};
use crate::curator::Curator;
use crate::data::{
    compute_dataset_stats, load_emotional_dataset, load_rut_gauntlet_prompts, DatasetStats,
    EmotionalSample, RutPrompt,
};
use crate::embedding::QwenStatefulEmbedder;
use crate::erag::{CollapseResult, EragClient};
use crate::generation::GenerationEngine;
use crate::gpu_fitness::GPUMemoryFitnessCalculator;
use crate::hyperfocus::HyperfocusDetector;
use crate::learning::LearningLoop;
use crate::mcts::MctsDaydreamer;
use crate::memory_consolidation::MemoryConsolidationManager;
use crate::metrics::cache_metrics;
use crate::security::PromptSecurityManager;
use crate::tcs_analysis::TCSAnalyzer;
use crate::token_manager::DynamicTokenizerManager;
use crate::topology_memory::TopologyMemoryAnalyzer;
use crate::torus::TorusPadMapper;
use crate::util::{seed_manager, set_global_seed};
use crate::weight_evolution::{Discovery, SmoothWeightEvolution};
use parking_lot::RwLock;
#[allow(unused_imports)]
use qdrant_client::qdrant::{CreateCollection, Distance, VectorsConfig};
use rand::RngCore;
use tcs_core::PersistentFeature;
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tracing::{info, warn};

use super::cache::{CollapseCache, EmbeddingCache};
use super::state::{Thresholds, TorusSeedStrategy};

const DEFAULT_CACHE_CAPACITY: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(256) };

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

fn preview_prompt(prompt: &str) -> String {
    let trimmed = prompt.trim();
    if trimmed.len() <= 64 {
        trimmed.to_string()
    } else {
        format!("{}...", &trimmed[..64])
    }
}

pub struct Pipeline {
    pub config: RuntimeConfig,
    pub config_arc: Arc<RwLock<RuntimeConfig>>,
    pub args: CliArgs,
    pub thresholds: Thresholds,
    pub dataset_stats: DatasetStats,
    pub embedder: QwenStatefulEmbedder,
    pub torus_strategy: TorusSeedStrategy,
    pub torus_counter: AtomicU64,
    pub compass: Arc<AsyncMutex<CompassEngine>>,
    pub erag: Arc<EragClient>,
    pub tokenizer: Arc<DynamicTokenizerManager>,
    pub generator: GenerationEngine,
    pub learning: AsyncMutex<LearningLoop>,
    pub curator: Option<Curator>,
    pub tcs_analyzer: Option<TCSAnalyzer>,
    pub embedding_cache: EmbeddingCache,
    pub collapse_cache: CollapseCache,
    pub retry_count: Arc<AtomicU32>,
    pub cascade_tracker: Arc<AsyncMutex<CascadeTracker>>, // Cascade tracking
    pub hyperfocus_detector: Arc<HyperfocusDetector>,     // Hyperfocus detection
    pub last_compass_outcome: Arc<AsyncMutex<Option<CompassOutcome>>>, // Track last compass for cascade
    #[allow(dead_code)]
    pub qdrant_process: Option<Arc<tokio::sync::Mutex<tokio::process::Child>>>,
    // Weighted Episodic Memory components
    pub weight_evolver: Arc<SmoothWeightEvolution>,
    pub gpu_fitness_calculator: Arc<GPUMemoryFitnessCalculator>,
    pub topology_analyzer: Arc<TopologyMemoryAnalyzer>,
    pub consolidation_manager: Arc<AsyncMutex<MemoryConsolidationManager>>,
    pub mcts_daydreamer: Arc<MctsDaydreamer>,
    pub discovery_queue: Arc<AsyncMutex<tokio::sync::mpsc::UnboundedSender<Discovery>>>,
    pub security: Arc<PromptSecurityManager>,
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

        let mut embedder =
            QwenStatefulEmbedder::new(&config.embedding_model_name, config.qdrant_vector_dim)?;
        // Set mock mode after initialization (embedder handles this gracefully)
        embedder.set_mock_mode(config.mock_mode);
        if config.embed_with_candle {
            if let Some(dir) = &config.embed_model_dir {
                embedder.enable_candle(std::path::Path::new(dir));
            }
        }
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

        let erag = EragClient::new(
            &config.qdrant_url,
            &config.qdrant_collection,
            config.qdrant_vector_dim,
            config.similarity_threshold,
        )
        .await?;

        // Log collection state for diagnostics
        if !config.mock_mode {
            if let Err(e) = erag.check_collection_info().await {
                warn!(error = %e, "Failed to check Qdrant collection info");
            }
        }
        let tokenizer = Arc::new(
            DynamicTokenizerManager::initialise(
                &tokenizer_path()?,
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
        generator.set_mock_mode(config.mock_mode);
        generator.set_system_prompt(config.system_prompt.clone());
        let config_arc = Arc::new(parking_lot::RwLock::new(config.clone()));
        let security_manager = Arc::new(PromptSecurityManager::new(config.security.clone())?);
        security_manager.audit_config_snapshot(&config);
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
            match TCSAnalyzer::new() {
                Ok(analyzer) => {
                    info!("TCS topology analyzer initialized");
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

        let cache_capacity =
            NonZeroUsize::new(config.cache_capacity).unwrap_or(DEFAULT_CACHE_CAPACITY);
        let embedding_cache = EmbeddingCache::new(
            cache_capacity,
            Duration::from_secs(config.embedding_cache_ttl_secs),
            config.cache_compression_min_bytes,
        );
        let collapse_cache = CollapseCache::new(
            cache_capacity,
            Duration::from_secs(config.collapse_cache_ttl_secs),
            config.cache_compression_min_bytes,
        );

        let prefetch_prompts = if config.cache_prefetch_enabled {
            Some(Self::select_prefetch_prompts(
                &samples,
                config.cache_prefetch_prompts,
            ))
        } else {
            None
        };

        // Initialize Weighted Episodic Memory components
        let weighted_config = &config.weighted_memory_config;
        let weight_evolver = Arc::new(SmoothWeightEvolution::new());
        let gpu_fitness_calculator =
            Arc::new(GPUMemoryFitnessCalculator::new(&weighted_config.gpu_device));
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

        // Spawn weight update monitor (updates EragClient weights every 5 seconds)
        let erag_arc_clone = Arc::clone(&erag_arc);
        let weight_evolver_monitor = Arc::clone(&weight_evolver);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            loop {
                interval.tick().await;
                let new_weights = weight_evolver_monitor.get_current_weights();
                // Update ERAG client weights (would need to add setter method)
                // For now, weights are accessed via weight_evolver when needed
            }
        });

        let mut pipeline = Self {
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
        };

        if let Some(prompts) = prefetch_prompts {
            pipeline.spawn_cache_prefetch(
                prompts,
                config.cache_prefetch_top_hits,
                config.cache_prefetch_parallelism,
                Duration::from_secs(config.embedding_cache_ttl_secs),
                Duration::from_secs(config.collapse_cache_ttl_secs),
            );
        }

        Ok(pipeline)
    }

    fn spawn_cache_prefetch(
        &self,
        prompts: Vec<String>,
        top_hits: usize,
        parallelism: usize,
        embedding_ttl: Duration,
        collapse_ttl: Duration,
    ) {
        let prompts: Vec<String> = prompts
            .into_iter()
            .filter(|p| !p.trim().is_empty())
            .collect();
        if prompts.is_empty() {
            return;
        }

        let parallelism = parallelism.clamp(1, 16);
        let semaphore = Arc::new(Semaphore::new(parallelism));
        let embedder = self.embedder.clone();
        let erag = Arc::clone(&self.erag);
        let embedding_cache = self.embedding_cache.clone();
        let collapse_cache = self.collapse_cache.clone();
        let top_hits = top_hits.clamp(1, 50);

        tokio::spawn(async move {
            for prompt in prompts {
                let permit = match semaphore.clone().acquire_owned().await {
                    Ok(permit) => permit,
                    Err(_) => break,
                };
                let _permit = permit;

                cache_metrics().record_prefetch_job();
                let key = super::cache::cache_key(&prompt);
                let now = Instant::now();

                if let Err(err) = async {
                    embedding_cache.update_ttl(embedding_ttl);
                    let mut embedding = if let Some(hit) = embedding_cache.fetch(&key, now).await? {
                        hit.value
                    } else {
                        let emb = embedder
                            .embed(&prompt)
                            .await
                            .context("Prefetch embedding computation failed")?;
                        embedding_cache.store(key, &emb, now).await?;
                        emb
                    };

                    collapse_cache.update_ttl(collapse_ttl);
                    if collapse_cache.fetch(&key, now).await?.is_none() {
                        let collapse = erag
                            .collapse_with_limit(&embedding, top_hits)
                            .await
                            .context("Prefetch ERAG collapse failed")?;
                        collapse_cache.store(key, &collapse, now).await?;
                    }

                    // Reuse embedding to keep borrow checker happy
                    embedding.clear();
                    Ok::<(), anyhow::Error>(())
                }
                .await
                {
                    cache_metrics().record_prefetch_failure();
                    warn!(target: "pipeline::prefetch", error = %err, prompt_preview = %preview_prompt(&prompt), "Cache prefetch task failed");
                }
            }
        });
    }

    fn select_prefetch_prompts(samples: &[EmotionalSample], limit: usize) -> Vec<String> {
        if limit == 0 {
            return Vec::new();
        }

        let mut prompts = Vec::with_capacity(limit);
        for sample in samples.iter().take(limit * 2) {
            let trimmed = sample.text.trim();
            if !trimmed.is_empty() {
                prompts.push(trimmed.to_string());
            }
            if prompts.len() >= limit {
                break;
            }
        }

        if prompts.len() < limit {
            for rut in load_rut_gauntlet_prompts() {
                let trimmed = rut.text.trim();
                if !trimmed.is_empty() {
                    prompts.push(trimmed.to_string());
                }
                if prompts.len() >= limit {
                    break;
                }
            }
        }

        prompts.truncate(limit);
        prompts
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
            TopologyMode::Hybrid => match TCSAnalyzer::new() {
                Ok(analyzer) => {
                    info!("TCS analyzer re-initialized for hybrid mode");
                    Some(analyzer)
                }
                Err(error) => {
                    warn!(%error, "Failed to initialize TCS analyzer; analytic fallback remains active");
                    None
                }
            },
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

    pub(crate) fn next_torus_mapper(&self) -> TorusPadMapper {
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
    pub(crate) fn recompute_thresholds(&mut self) {
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

    pub fn hardware_profile(&self) -> HardwareProfile {
        self.args.hardware
    }
}

pub(crate) fn tokenizer_path() -> Result<PathBuf> {
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

    anyhow::bail!("Tokenizer JSON not found; set TOKENIZER_JSON or QWEN_TOKENIZER")
}
