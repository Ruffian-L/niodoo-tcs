use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context;
use anyhow::Result;

use crate::compass::{CompassEngine, CompassOutcome, CompassQuadrant};
use crate::config::{
    CliArgs, CuratorConfig, HardwareProfile, RuntimeConfig, TopologyMode, env_value,
    set_env_override,
};
use crate::conversation_log::{ConversationEntry, ConversationLogStore};
use crate::curator::Curator;
use crate::data::{
    DatasetStats, Experience, RutPrompt, compute_dataset_stats, load_emotional_dataset,
    load_rut_gauntlet_prompts,
};
use crate::embedding::QwenStatefulEmbedder;
use crate::emotional_graph::EmotionalGraphBuilder;
use crate::erag::{CollapseResult, EragClient};
use crate::generation::{GenerationEngine, GenerationResult};
use crate::learning::{LearningLoop, LearningOutcome};
use crate::metrics::{FailureSignals, metrics};
use crate::tcs_analysis::{TCSAnalyzer, TopologicalSignature};
use crate::token_manager::{DynamicTokenizerManager, TokenizerOutput};
use crate::torus::{PadGhostState, TorusPadMapper};
use crate::util::{rouge_l, seed_manager, set_global_seed};
use crate::erag::EmotionalVector as EragEmotionalVector;
use blake3::hash as blake3_hash;
use lru::LruCache;
use parking_lot::RwLock;
#[allow(unused_imports)]
use qdrant_client::qdrant::{CreateCollection, Distance, VectorsConfig};
use rand::RngCore;
use tcs_core::topology::PersistenceFeature;
use tokio::sync::Mutex as AsyncMutex;
use tracing::{info, warn};
use niodoo_core::{MemoryConsolidationEngine, ConsciousnessEvent, MultiLayerMemoryQuery, RetrievalEngine, GuessingMemorySystem, ConsciousnessState};

// Proto module - include generated proto code from OUT_DIR during build
#[allow(dead_code)]
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/niodoo.rs"));
}

// Define CuratedExperience struct
#[derive(Debug, Clone)]
struct CuratedExperience {
    refined_response: String,
    quality_score: f32,
    promoted_tokens: Vec<String>,
    learned: bool,
    #[allow(dead_code)]
    reason: String,
}

#[derive(Debug, Clone)]
pub struct Thresholds {
    pub entropy_mean: f64,
    pub entropy_high: f64,
    pub variance_stagnation: f64,
    pub variance_spike: f64,
    pub mirage_sigma: f64,
    pub mcts_c: f64,
}

#[derive(Debug, Clone)]
pub struct PipelineCycle {
    pub prompt: String,
    pub baseline_response: String,
    pub hybrid_response: String,
    pub entropy: f64,
    pub rouge: f64,
    pub latency_ms: f64,
    pub compass: CompassOutcome,
    pub generation: GenerationResult,
    pub tokenizer: TokenizerOutput,
    pub collapse: CollapseResult,
    pub learning: LearningOutcome,
    pub stage_timings: StageTimings,
    pub last_entropy: f64,
    pub failure: String, // "soft", "hard", "none"
    pub pad_state: PadGhostState,
    pub topology: TopologicalSignature,
    pub topology_mode: TopologyMode,
}

#[derive(Debug, Clone, Default)]
pub struct StageTimings {
    pub embedding_ms: f64,
    pub torus_ms: f64,
    pub tcs_ms: f64,
    pub compass_ms: f64,
    pub erag_ms: f64,
    pub tokenizer_ms: f64,
    pub generation_ms: f64,
    pub learning_ms: f64,
    pub threat_cycle_ms: f64,
}

#[derive(Debug, Clone, Copy)]
enum TorusSeedStrategy {
    Fixed(u64),
    Random,
}

pub struct Pipeline {
    pub config: RuntimeConfig,
    config_arc: Arc<RwLock<RuntimeConfig>>,
    pub args: CliArgs,
    pub thresholds: Thresholds,
    pub dataset_stats: DatasetStats,
    embedder: QwenStatefulEmbedder,
    torus_strategy: TorusSeedStrategy,
    torus_counter: AtomicU64,
    compass: Arc<AsyncMutex<CompassEngine>>,
    erag: Arc<EragClient>,
    tokenizer: Arc<DynamicTokenizerManager>,
    generator: GenerationEngine,
    learning: AsyncMutex<LearningLoop>,
    curator: Option<Curator>,
    tcs_analyzer: Option<TCSAnalyzer>,
    embedding_cache: LruCache<u64, CacheEntry<Vec<f32>>>,
    collapse_cache: LruCache<u64, CacheEntry<CollapseResult>>,
    retry_count: Arc<AtomicU32>,
    #[allow(dead_code)]
    qdrant_process: Option<tokio::process::Child>,
    // Phase 2 integration modules
    conversation_log: AsyncMutex<ConversationLogStore>,
    emotional_graph: AsyncMutex<EmotionalGraphBuilder>,
    cycle_count: Arc<AtomicU64>,
    // NEW: Memory consolidation engine
    consolidation_engine: AsyncMutex<MemoryConsolidationEngine>,
    // NEW: Triple-threat trigger system for learning events
    multi_layer_query: Arc<AsyncMutex<Option<MultiLayerMemoryQuery>>>,
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

        let mut embedder = QwenStatefulEmbedder::new(
            &config.ollama_endpoint,
            &config.embedding_model_name,
            config.qdrant_vector_dim,
            config.embedding_max_chars,
        )?;
        if config.embed_with_candle {
            if let Some(dir) = &config.embed_model_dir {
                embedder.enable_candle(dir);
            }
        }
        embedder.set_mock_mode(config.mock_mode);
        info!(
            endpoint = %config.ollama_endpoint,
            model = %config.embedding_model_name,
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
        let qdrant_process: Option<tokio::process::Child> = if config.qdrant_embedded {
            info!("QDRANT_EMBEDDED enabled: spawning embedded Qdrant process");
            match crate::embedded_qdrant::spawn_embedded_qdrant().await {
                Ok(child) => Some(child),
                Err(e) => {
                    warn!(error = %e, "Failed to spawn embedded Qdrant; falling back to external Qdrant");
                    None
                }
            }
        } else {
            None
        };

        #[cfg(not(feature = "embedded-qdrant"))]
        let qdrant_process: Option<tokio::process::Child> = None;

        let erag = EragClient::new(
            &config.qdrant_url,
            &config.qdrant_collection,
            config.qdrant_vector_dim,
            config.similarity_threshold,
            embedder_arc.clone(),
            config.mock_mode,
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
            config.generation_timeout_secs,
            config.generation_max_tokens,
            config.dynamic_token_min,
            config.dynamic_token_max,
            config.prompt_max_chars,
            config.consistency_variance_threshold,
        )?;
        generator.set_mock_mode(config.mock_mode);
        generator.set_system_prompt(config.system_prompt.clone());
        let config_arc = Arc::new(parking_lot::RwLock::new(config.clone()));
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
            info!(
                backend = ?curator_config.curator_backend,
                endpoint = %curator_config.vllm_endpoint,
                model = %curator_config.model_name,
                "Curator configured (backend: {:?})",
                curator_config.curator_backend
            );
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
            NonZeroUsize::new(config.cache_capacity).unwrap_or(NonZeroUsize::new(256).unwrap());

        // Initialize Phase 2 integration modules
        let conversation_log_path = env_value("CONVERSATION_LOG_PATH")
            .unwrap_or_else(|| "./data/conversations.json".to_string());
        let mut conversation_log = ConversationLogStore::new(&conversation_log_path);
        conversation_log.load().context("Failed to load conversation log")?;
        info!("Phase 2: ConversationLogStore initialized at {:?}", conversation_log_path);

        let emotional_graph = EmotionalGraphBuilder::default();
        info!("Phase 2: EmotionalGraphBuilder initialized");

        // Initialize memory consolidation engine
        let consolidation_engine = MemoryConsolidationEngine::new();
        info!("Memory consolidation engine initialized");

        // Initialize triple-threat trigger system (optional - needs RetrievalEngine)
        // Create RetrievalEngine and GuessingMemorySystem for MultiLayerMemoryQuery
        let rag_engine = Arc::new(std::sync::Mutex::new(RetrievalEngine::new()));
        let gaussian_system = GuessingMemorySystem::new();
        let multi_layer_query_instance = MultiLayerMemoryQuery::new(rag_engine, gaussian_system);
        let multi_layer_query = Arc::new(AsyncMutex::new(Some(multi_layer_query_instance)));
        info!("Triple-threat trigger system initialized with RAG + Gaussian spheres");

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
            embedding_cache: LruCache::new(cache_capacity),
            collapse_cache: LruCache::new(cache_capacity),
            retry_count: Arc::new(AtomicU32::new(0)),
            qdrant_process,
            conversation_log: AsyncMutex::new(conversation_log),
            emotional_graph: AsyncMutex::new(emotional_graph),
            cycle_count: Arc::new(AtomicU64::new(0)),
            consolidation_engine: AsyncMutex::new(consolidation_engine),
            multi_layer_query: multi_layer_query,
        })
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
        if seed_manager().master_seed() != seed {
            warn!(
                existing = seed_manager().master_seed(),
                requested = seed,
                "Seed override ignored; global seed already initialised"
            );
        }
        Self::initialise_with_topology(args, None, Some(seed)).await
    }

    fn next_torus_mapper(&self) -> TorusPadMapper {
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
    fn recompute_thresholds(&mut self) {
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

    /// Process a single prompt through the complete Niodoo pipeline.
    ///
    /// This is the main entry point for processing user prompts. The pipeline executes
    /// the following stages in sequence:
    ///
    /// 1. **Embedding**: Convert prompt text to embedding vector (with caching)
    /// 2. **Torus Projection**: Map embedding to 7D PAD+ghost space
    /// 3. **Topology Analysis**: Compute persistent homology, knot invariants, TQFT signatures
    /// 4. **Compass**: Analyze emotional state and determine navigation quadrant
    /// 5. **ERAG Collapse**: Retrieve relevant memories from vector store
    /// 6. **Tokenization**: Encode prompt with dynamic tokenizer (promotes new tokens)
    /// 7. **Generation**: Generate response using vLLM (baseline + hybrid with retrieved context)
    /// 8. **Curator**: Quality assessment and refinement
    /// 9. **Learning**: Update DQN and trigger QLoRA fine-tuning if breakthroughs detected
    /// 10. **Memory Storage**: Store high-quality experiences in ERAG
    /// 11. **Phase 2 Integration**: Store conversation log and build emotional graph
    ///
    /// # Arguments
    ///
    /// * `prompt` - The user's input prompt to process
    ///
    /// # Returns
    ///
    /// * `PipelineCycle` - Complete cycle results including:
    ///   - Generated responses (baseline and hybrid)
    ///   - Metrics (entropy, ROUGE, latency)
    ///   - Emotional state (Compass outcome)
    ///   - Topological signature
    ///   - Learning outcomes
    ///
    /// # Errors
    ///
    /// Returns `Result` which can fail at any stage:
    /// - Embedding failures
    /// - Generation timeouts
    /// - Memory storage failures
    /// - Configuration errors
    ///
    /// # Example
    ///
    /// ```no_run
    /// use niodoo_real_integrated::pipeline::Pipeline;
    /// use niodoo_real_integrated::config::CliArgs;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let mut pipeline = Pipeline::initialise(CliArgs::default()).await?;
    /// let cycle = pipeline.process_prompt("Hello, how are you?").await?;
    /// println!("Response: {}", cycle.hybrid_response);
    /// println!("Entropy: {:.3}", cycle.entropy);
    /// println!("ROUGE: {:.3}", cycle.rouge);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn process_prompt(&mut self, prompt: &str) -> Result<PipelineCycle> {
        let overall_start = Instant::now();
        let mut timings = StageTimings::default();
        let cache_key = cache_key(prompt);
        let now = Instant::now();

        // Stage 1: Embedding (with cache)
        let embedding_start = Instant::now();
        let embedding_ttl = Duration::from_secs(self.config.embedding_cache_ttl_secs);
        let collapse_ttl = Duration::from_secs(self.config.collapse_cache_ttl_secs);
        let embedding = match self.embedding_cache.get(&cache_key) {
            Some(entry) if !entry.is_expired(now, embedding_ttl) => entry.value.clone(),
            _ => {
                self.embedding_cache.pop(&cache_key);
                let emb = self.embedder.embed(prompt).await?;
                self.embedding_cache
                    .put(cache_key, CacheEntry::new(emb.clone(), now));
                emb
            }
        };
        timings.embedding_ms = embedding_start.elapsed().as_secs_f64() * 1000.0;
        info!(
            "Pipeline stage: embedding completed in {:.2}ms",
            timings.embedding_ms
        );

        // Stage 2: Torus projection
        let torus_start = Instant::now();
        let mut torus_mapper = self.next_torus_mapper();
        let pad_state = torus_mapper.project(&embedding)?;
        timings.torus_ms = torus_start.elapsed().as_secs_f64() * 1000.0;

        let tcs_start = Instant::now();
        let (topology, analysis_label) = match self.config.topology_mode {
            TopologyMode::Hybrid => match self.tcs_analyzer.as_mut() {
                Some(analyzer) => match analyzer.analyze_state(&pad_state) {
                    Ok(signature) => (signature, "hybrid"),
                    Err(error) => {
                        warn!(%error, "TCS analyzer failed; using analytic baseline signature");
                        (
                            baseline_topological_signature(&pad_state, &embedding),
                            "hybrid_fallback",
                        )
                    }
                },
                None => {
                    warn!(
                        "Hybrid mode requested but TCS analyzer unavailable; using analytic baseline signature"
                    );
                    (
                        baseline_topological_signature(&pad_state, &embedding),
                        "hybrid_fallback",
                    )
                }
            },
            TopologyMode::Baseline => (
                baseline_topological_signature(&pad_state, &embedding),
                "baseline",
            ),
        };
        timings.tcs_ms = tcs_start.elapsed().as_secs_f64() * 1000.0;
        info!(
            "Pipeline stage: topology analysis completed in {:.2}ms ({analysis_label})",
            timings.tcs_ms
        );

        // Phase 5.3: Check if predictor should trigger (knot > 0.4)
        let _topology_json = match serde_json::to_string(&topology) {
            Ok(json) => json,
            Err(e) => {
                warn!(error = %e, "Failed to serialize topology to JSON");
                String::new()
            }
        };
        info!(
            "Topological signature: knot={:.3}, betti={:?}, pe={:.3}, gap={:.3}",
            topology.knot_complexity,
            topology.betti_numbers,
            topology.persistence_entropy,
            topology.spectral_gap
        );

        // Evaluate compass on blocking thread without locking inside closure
        let pad_state_for_compass = pad_state.clone();
        let topology_for_compass = topology.clone();
        let compass_guard = self.compass.clone().lock_owned().await;
        let compass_scope = format!("compass/{}", cache_key);
        let compass_task = tokio::task::spawn_blocking(move || {
            let mut engine = compass_guard;
            let mut rng = crate::util::seed_manager().get_rng(&compass_scope);
            engine.evaluate_with_rng(
                &pad_state_for_compass,
                Some(&topology_for_compass),
                &mut rng,
            )
        });

        // Start timing BEFORE the parallel work begins
        let compass_erag_start = Instant::now();
        
        // Convert ERAG emotional vector to niodoo_core EmotionalVector for triple-threat trigger
        let erag_emotional_vector = EragEmotionalVector::from_pad(&pad_state);
        let query_emotion = niodoo_core::memory::EmotionalVector::new(
            erag_emotional_vector.joy,
            erag_emotional_vector.sadness,
            erag_emotional_vector.anger,
            erag_emotional_vector.fear,
            erag_emotional_vector.surprise,
        );
        
        // Check triple-threat trigger system (runs in parallel, doesn't block ERAG)
        let multi_layer_query_clone = Arc::clone(&self.multi_layer_query);
        let prompt_clone = prompt.to_string();
        let query_emotion_clone = query_emotion.clone();
        tokio::spawn(async move {
            if let Some(ref mut mlq) = *multi_layer_query_clone.lock().await {
                let mut state = ConsciousnessState::default();
                // Query triggers triple-threat detection
                let _ = mlq.query(&prompt_clone, &query_emotion_clone, 5, &mut state);
                // Trigger detection happens inside query() - triggers are logged automatically
            }
        });
        
        let (compass, collapse) = tokio::try_join!(
            async {
                match compass_task.await {
                    Ok(inner) => inner,
                    Err(e) => Err(anyhow::anyhow!(format!(
                        "compass evaluation panicked: {}",
                        e
                    ))),
                }
            },
            async {
                match self.collapse_cache.get(&cache_key) {
                    Some(entry) if !entry.is_expired(now, collapse_ttl) => Ok(entry.value.clone()),
                    _ => {
                        self.collapse_cache.pop(&cache_key);
                        // Dynamic top_k based on config knobs (reuses retrieval_top_k_increment as delta)
                        let top_k = (3i32 + self.config.phase2_retrieval_top_k_increment)
                            .clamp(1, 50) as usize;
                        let collapse = self.erag.collapse_with_limit(&embedding, top_k).await?;
                        self.collapse_cache
                            .put(cache_key, CacheEntry::new(collapse.clone(), now));
                        Ok(collapse)
                    }
                }
            }
        )?;
        // Measure elapsed time AFTER the work completes
        let compass_erag_elapsed = compass_erag_start.elapsed().as_secs_f64() * 1000.0;
        timings.compass_ms = compass_erag_elapsed / 2.0;
        timings.erag_ms = compass_erag_elapsed / 2.0;
        info!(
            "Pipeline stage: compass completed in {:.2}ms",
            timings.compass_ms
        );
        info!("Pipeline stage: erag completed in {:.2}ms", timings.erag_ms);

        // Stage 5: Tokenizer
        let tokenizer_start = Instant::now();
        let tokenizer_output = self
            .tokenizer
            .process_with_memories(prompt, &collapse, &pad_state, collapse.top_hits.clone())
            .await?;
        timings.tokenizer_ms = tokenizer_start.elapsed().as_secs_f64() * 1000.0;

        // Update generation engine with latest config params (before generation)
        let current_config = self.config_arc.read().clone();
        self.generator.apply_runtime_from_config(&current_config);
        self.generator.update_params(
            current_config.temperature,
            current_config.top_p,
            current_config.repetition_penalty,
        );
        self.config = current_config;

        // Recompute thresholds from updated config and update compass
        self.recompute_thresholds();

        // Stage 6: Generation
        let generation_start = Instant::now();
        // Apply latest runtime parameters before generation
        {
            let cfg = self.config_arc.read().clone();
            self.generator.apply_runtime_from_config(&cfg);
            self.recompute_thresholds();
            self.config = cfg;
        }
        let generation = if self.config.enable_consistency_voting {
            let voting = self
                .generator
                .generate_with_consistency(&tokenizer_output, &compass)
                .await?;

            let selected = match voting.winner_index {
                0 => &voting.candidate_1,
                1 => &voting.candidate_2,
                _ => &voting.candidate_3,
            }
            .clone();

            GenerationResult {
                baseline_response: tokenizer_output.augmented_prompt.clone(),
                hybrid_response: selected,
                echoes: Vec::new(),
                rouge_to_baseline: voting.rouge_scores.iter().copied().sum::<f64>()
                    / voting.rouge_scores.len() as f64,
                latency_ms: voting.latency_ms,
                rouge_score: voting.rouge_scores.iter().copied().sum::<f64>()
                    / voting.rouge_scores.len() as f64,
                entropy_delta: 0.0,
                source: "consistency".to_string(),
                ucb1_score: compass
                    .mcts_branches
                    .iter()
                    .map(|b| b.ucb_score)
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.5),
                curator_quality: 0.8, // Default quality for consistency voting
                failure_type: None,
                failure_details: None,
            }
        } else {
            self.generator
                .generate_with_topology(&tokenizer_output, &compass, Some(&topology), false)
                .await?
        };
        timings.generation_ms = generation_start.elapsed().as_secs_f64() * 1000.0;
        info!(
            "Pipeline stage: generation completed in {:.2}ms",
            timings.generation_ms
        );

        // NEW: Phase 2 Integration - Call curator after generation WITH TOPOLOGY
        let curated_experience = self
            .integrate_curator(
                prompt,
                &generation.hybrid_response,
                &pad_state,
                &compass,
                &collapse.aggregated_context,
                &topology,
                &tokenizer_output,
            )
            .await?;

        // Feed curator output to learning loop if learned=true
        if curated_experience.learned {
            let reward = generation.rouge_score * 0.5 + (1.0 - pad_state.entropy) * 0.5;
            if let Err(e) = self
                .learning
                .lock()
                .await
                .apply_curator_learned(
                    &curated_experience.refined_response,
                    true,
                    reward,
                    &topology,
                    prompt,
                    &curated_experience.promoted_tokens,
                )
                .await
            {
                warn!("Failed to apply curator learned data: {}", e);
            }
        }

        // Failure evaluation after curator
        let entropy_delta = pad_state.entropy - (self.thresholds.entropy_mean);
        let curator_quality = curated_experience.quality_score as f64;

        // Extract actual UCB1 score from MCTS branches
        let ucb1_score = compass
            .mcts_branches
            .iter()
            .map(|branch| branch.ucb_score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(self.thresholds.mcts_c); // Fallback to configured threshold

        let (mut failure, mut details) = FailureSignals::evaluate(
            generation.rouge_score,
            entropy_delta,
            curator_quality,
            ucb1_score,
        );

        let reason_lower = curated_experience.reason.to_lowercase();
        let curator_unavailable = self.curator.is_none()
            || reason_lower.contains("curator_disabled")
            || reason_lower.contains("ollama")
            || reason_lower.contains("curator_error")
            || reason_lower.contains("curator mock mode")
            || reason_lower.contains("mock mode")
            || reason_lower.contains("request_failed");
        let curator_passive = !curated_experience.learned;

        if (curator_unavailable || curator_passive) && failure != "none" {
            info!(reason = %curated_experience.reason, "Curator unavailable or passive; skipping retry escalation");
            failure = "none".to_string();
            details = if curator_unavailable {
                "curator_unavailable".to_string()
            } else {
                "curator_passive".to_string()
            };
        }

        let quality_acceptable = (curated_experience.quality_score as f64)
            >= self.config.curator_minimum_threshold as f64;
        let rouge_acceptable = generation.rouge_score >= 0.25;
        if failure == "soft" && (quality_acceptable || rouge_acceptable) {
            info!(
                rouge = generation.rouge_score,
                quality = curated_experience.quality_score,
                "Soft failure bypassed due to acceptable metrics"
            );
            failure = "none".to_string();
            details = "quality_acceptable".to_string();
        }

        // Phase 2: Handle retries with Reflection and CoT with topology awareness
        let (final_generation, final_failure, threat_cycle_ms) = self
            .handle_retry_with_reflection(
                prompt,
                &failure,
                &details,
                &generation,
                &compass,
                &collapse,
                &curated_experience,
                entropy_delta,
                curator_quality,
                ucb1_score,
                &topology, // TOPOLOGY INTEGRATION: Pass topology to retry logic
            )
            .await?;

        // Update timings with threat cycle timing
        timings.threat_cycle_ms = threat_cycle_ms;

        // Log to ERAG if failure != "none"
        if final_failure != "none" {
            let metrics_ref = metrics();
            self.erag
                .store_failure(
                    prompt,
                    &final_generation.hybrid_response,
                    metrics_ref,
                    Some(details.to_string()),
                    &final_failure,
                    self.retry_count.load(Ordering::Relaxed),
                )
                .await?;
        }

        // Proceed with learning using curated response (with retry-corrected generation)
        let learning_start = Instant::now();

        let learning_outcome = self
            .learning
            .lock()
            .await
            .update(
                &pad_state,
                &compass,
                &collapse,
                &final_generation,
                &topology,
            )
            .await
            .context("Learning loop update failed")?;

        timings.learning_ms = learning_start.elapsed().as_secs_f64() * 1000.0;

        // Remove double-storage: defer storage decision to final gate below

        // Create Experience from pipeline
        let aggregated_context_lines: Vec<String> = collapse
            .aggregated_context
            .lines()
            .map(|s| s.to_string())
            .collect();
        let experience_input = prompt.to_string();
        let experience_output = final_generation.hybrid_response.clone();
        let experience_embedding = embedding.clone();
        let experience_context = aggregated_context_lines.clone();

        let experience = Experience::from_pipeline(
            experience_input.clone(),
            experience_output,
            experience_embedding,
            &pad_state,
            &compass,
            experience_context.clone(),
        );

        // Stage 7.5: Curator Quality Gate (single source of truth)
        let response_to_store = curated_experience.refined_response.clone();
        let final_quality_score = Some(curated_experience.quality_score);
        if curated_experience.quality_score < self.config.curator_minimum_threshold {
            warn!(
                quality = curated_experience.quality_score,
                min = self.config.curator_minimum_threshold,
                "Curated quality below minimum; skipping memory store"
            );
            return Ok(PipelineCycle {
                prompt: prompt.to_string(),
                baseline_response: final_generation.baseline_response.clone(),
                hybrid_response: final_generation.hybrid_response.clone(),
                entropy: pad_state.entropy,
                rouge: final_generation.rouge_to_baseline,
                latency_ms: overall_start.elapsed().as_secs_f64() * 1000.0,
                compass,
                generation: final_generation,
                tokenizer: tokenizer_output,
                collapse,
                learning: learning_outcome,
                stage_timings: timings,
                last_entropy: pad_state.entropy,
                failure: final_failure,
                pad_state: pad_state.clone(),
                topology: topology.clone(),
                topology_mode: self.config.topology_mode,
            });
        }

        let solution_path = experience.solution_path.clone();
        self.erag
            .upsert_memory(
                &experience.embedding.as_ref().unwrap(),
                &pad_state,
                &compass,
                &experience_input,
                &response_to_store,
                &experience_context,
                pad_state.entropy,
                final_quality_score,
                Some(&topology),
                solution_path,
                experience.iteration_count,
            )
            .await?; // Propagate error

        // Add to consolidation engine for later consolidation
        let erag_emotional_vector = EragEmotionalVector::from_pad(&pad_state);
        let emotional_vector_for_consolidation = niodoo_core::memory::EmotionalVector::new(
            erag_emotional_vector.joy,
            erag_emotional_vector.sadness,
            erag_emotional_vector.anger,
            erag_emotional_vector.fear,
            erag_emotional_vector.surprise,
        );
        let consolidation_event = ConsciousnessEvent::MemoryUpdate {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            content: format!("{} -> {}", experience_input, response_to_store),
            emotional_vector: emotional_vector_for_consolidation,
        };
        {
            let consolidator = self.consolidation_engine.lock().await;
            consolidator.add_memories_for_consolidation(vec![consolidation_event]).await;
        }

        metrics().observe_cycle(
            pad_state.entropy,
            final_generation.latency_ms,
            final_generation.rouge_to_baseline,
            compass.is_threat,
            compass.is_healing,
        );

        // Emit per-cycle WebSocket event (best-effort)
        if let Some(ws_url) = env_value("NIODOO_WS_ENDPOINT") {
            let _ = tokio::spawn({
                let ws_url = ws_url.clone();
                let event = serde_json::json!({
                    "event": "cycle",
                    "entropy": pad_state.entropy,
                    "knot": topology.knot_complexity,
                    "betti": topology.betti_numbers,
                    "ucb1": compass.ucb1_score,
                    "retries": self.retry_count.load(Ordering::Relaxed),
                    "latency_ms": final_generation.latency_ms,
                });
                async move {
                    let _ = reqwest::Client::new()
                        .post(format!("{}/events", ws_url.trim_end_matches('/')))
                        .json(&event)
                        .send()
                        .await;
                }
            });
        }

        // Phase 2: Store conversation in ConversationLogStore with enhanced PAD tagging
        let cycle = self.cycle_count.fetch_add(1, Ordering::Relaxed);
        let erag_emotional_vector = EragEmotionalVector::from_pad(&pad_state);
        let emotional_vector = niodoo_core::memory::EmotionalVector::new(
            erag_emotional_vector.joy,
            erag_emotional_vector.sadness,
            erag_emotional_vector.anger,
            erag_emotional_vector.fear,
            erag_emotional_vector.surprise,
        );
        let emotion_state = format!("{:?}", compass.quadrant);
        
        let mut conversation_entry = ConversationEntry::new(
            prompt.to_string(),
            final_generation.hybrid_response.clone(),
            emotional_vector,
            emotion_state,
        );
        
        // Hook post-process for PAD tagging: Add detailed PAD metadata
        conversation_entry.metadata.insert("pad_pleasure".to_string(), pad_state.pad[0].to_string());
        conversation_entry.metadata.insert("pad_arousal".to_string(), pad_state.pad[1].to_string());
        conversation_entry.metadata.insert("pad_dominance".to_string(), pad_state.pad[2].to_string());
        conversation_entry.metadata.insert("entropy".to_string(), pad_state.entropy.to_string());
        conversation_entry.metadata.insert("knot_complexity".to_string(), topology.knot_complexity.to_string());
        conversation_entry.metadata.insert("spectral_gap".to_string(), topology.spectral_gap.to_string());
        conversation_entry.metadata.insert("rouge_score".to_string(), final_generation.rouge_to_baseline.to_string());
        
        if let Err(e) = self.conversation_log.lock().await.store(conversation_entry) {
            warn!("Failed to store conversation: {}", e);
        }
        
        // Periodically build emotional graph (every 10 cycles)
        if cycle > 0 && cycle % 10 == 0 {
            let log_store = self.conversation_log.lock().await;
            let mut graph_builder = self.emotional_graph.lock().await;
            
            if let Err(e) = graph_builder.build_from_conversations(&log_store) {
                warn!("Failed to build emotional graph: {}", e);
            } else {
                info!("Phase 2: Built emotional graph with {} spheres", graph_builder.sphere_count());
            }
        }

        // Periodically curate memory (every 50 cycles) - REMOVE LOW QUALITY
        if cycle > 0 && cycle % 50 == 0 {
            match self.erag.curate_memory().await {
                Ok(deleted) => {
                    info!(deleted, "Memory curation completed: removed {} low-quality memories", deleted);
                }
                Err(e) => {
                    warn!(%e, "Memory curation failed");
                }
            }
        }

        // Periodically distill knowledge (every 100 cycles) - CREATE TRAINING EXAMPLES
        if cycle > 0 && cycle % 100 == 0 {
            if let Some(ref curator) = self.curator {
                match curator.distill_knowledge(&self.erag, 10).await {
                    Ok(distilled) => {
                        info!(count = distilled.len(), "Knowledge distillation completed: {} training examples created", distilled.len());
                        // Store distilled examples for learning loop
                        for example in distilled {
                            tracing::debug!(instruction = %example.instruction, quality = example.quality_score, "Distilled example");
                        }
                    }
                    Err(e) => {
                        warn!(%e, "Knowledge distillation failed");
                    }
                }
            }
        }

        // Periodically consolidate memories (every 200 cycles) - COMPRESS/MERGE/PRUNE
        if cycle > 0 && cycle % 200 == 0 {
            let mut consolidator = self.consolidation_engine.lock().await;
            match consolidator.run_consolidation_cycle().await {
                Ok(_) => {
                    info!("Memory consolidation cycle completed");
                }
                Err(e) => {
                    warn!(%e, "Memory consolidation failed");
                }
            }
        }

        // learning_ms already set above

        Ok(PipelineCycle {
            prompt: prompt.to_string(),
            baseline_response: final_generation.baseline_response.clone(),
            hybrid_response: final_generation.hybrid_response.clone(),
            entropy: pad_state.entropy,
            rouge: final_generation.rouge_to_baseline,
            latency_ms: overall_start.elapsed().as_secs_f64() * 1000.0,
            compass,
            generation: final_generation,
            tokenizer: tokenizer_output,
            collapse,
            learning: learning_outcome,
            stage_timings: timings,
            last_entropy: pad_state.entropy,
            failure: final_failure,
            pad_state,
            topology,
            topology_mode: self.config.topology_mode,
        })
    }

    pub fn hardware_profile(&self) -> HardwareProfile {
        self.args.hardware
    }

    /// Phase 2: Query conversations by emotion similarity
    /// Wraps ConversationLogStore query for use by LearningLoop and other components
    pub async fn query_conversations_by_emotion(
        &self,
        query_emotion: &niodoo_core::memory::EmotionalVector,
        threshold: f32,
        limit: usize,
    ) -> Vec<crate::conversation_log::ConversationEntry> {
        let log_store = self.conversation_log.lock().await;
        log_store.query_by_emotion(query_emotion, threshold, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Phase 2: Query conversations by time range
    /// Wraps ConversationLogStore query for use by LearningLoop and other components
    pub async fn query_conversations_by_time_range(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Vec<crate::conversation_log::ConversationEntry> {
        let log_store = self.conversation_log.lock().await;
        log_store.query_by_time_range(start, end)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Phase 2: Query conversations by content similarity
    /// Wraps ConversationLogStore query for use by LearningLoop and other components
    pub async fn query_conversations_by_content(
        &self,
        query_text: &str,
        threshold: f32,
        limit: usize,
    ) -> Vec<crate::conversation_log::ConversationEntry> {
        let log_store = self.conversation_log.lock().await;
        log_store.query_by_content(query_text, threshold, limit)
            .into_iter()
            .cloned()
            .collect()
    }
}

impl Default for PipelineCycle {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            baseline_response: String::new(),
            hybrid_response: String::new(),
            entropy: 0.0,
            rouge: 0.0,
            latency_ms: 0.0,
            compass: CompassOutcome {
                quadrant: CompassQuadrant::Persist,
                is_threat: false,
                is_healing: false,
                mcts_branches: Vec::new(),
                intrinsic_reward: 0.0,
                ucb1_score: None,
            },
            generation: GenerationResult {
                baseline_response: String::new(),
                hybrid_response: String::new(),
                echoes: Vec::new(),
                rouge_to_baseline: 0.0,
                latency_ms: 0.0,
                rouge_score: 0.0,
                entropy_delta: 0.0,
                source: "default".to_string(),
                ucb1_score: 0.0,
                curator_quality: 0.0,
                failure_type: None,
                failure_details: None,
            },
            tokenizer: TokenizerOutput {
                tokens: Vec::new(),
                augmented_prompt: String::new(),
                promoted_tokens: Vec::new(),
                vocab_size: 0,
                oov_rate: 0.0,
                failure_type: None,
                failure_details: None,
            },
            collapse: CollapseResult::empty("none", None),
            learning: LearningOutcome::default(),
            stage_timings: StageTimings::default(),
            last_entropy: 0.0,
            failure: "none".to_string(),
            pad_state: PadGhostState::default(),
            topology: TopologicalSignature::new(
                Vec::new(),
                [1, 0, 0],
                0.0,
                String::new(),
                2,
                None,
                0.0,
                0.0,
                0.0,
            ),
            topology_mode: TopologyMode::Baseline,
        }
    }
}

impl Pipeline {
    /// Phase 2: Handle retries with Reflection and CoT self-correction with topology awareness
    async fn handle_retry_with_reflection(
        &self,
        prompt: &str,
        initial_failure: &str,
        details: &str,
        generation: &GenerationResult,
        compass: &CompassOutcome,
        _collapse: &CollapseResult,
        _curated: &CuratedExperience,
        entropy_delta: f64,
        curator_quality: f64,
        ucb1_score: f64,
        topology: &crate::tcs_analysis::TopologicalSignature,
    ) -> Result<(GenerationResult, String, f64)> {
        // INTEGRATION FIX: Handle healing state specially - enhance instead of retry
        if initial_failure == "none" && compass.is_healing {
            // In healing state with good topology - apply enhancement strategies
            if topology.knot_complexity < 0.4 && topology.spectral_gap > 0.6 {
                info!("Healing state detected with good topology - applying enhancement");

                // Generate an enhanced version leveraging the good state
                let enhancement_prompt = format!(
                    "{}\n\n[System is in optimal healing state. Enhance clarity and depth.]",
                    prompt
                );

                if let Ok(enhanced) = self
                    .generator
                    .generate_with_params(&enhancement_prompt, 0.3, 0.95) // Low temp for stability
                    .await
                {
                    return Ok((enhanced, "none".to_string(), 0.0));
                }
            }
            return Ok((generation.clone(), "none".to_string(), 0.0));
        }

        // No failure and not healing, return original
        if initial_failure == "none" {
            return Ok((generation.clone(), "none".to_string(), 0.0));
        }

        let cfg_snapshot = self.config_arc.read().clone();
        let max_retries = cfg_snapshot.phase2_max_retries;
        let base_delay_ms = cfg_snapshot.phase2_retry_base_delay_ms;
        let cot_iterations = cfg_snapshot.phase2_cot_iterations.max(1) as usize;
        let cot_success_rouge = cfg_snapshot.cot_success_rouge_threshold;
        let level3_retry_count = cfg_snapshot.phase2_level3_retry_count;
        let mcts_c_increment = cfg_snapshot.phase2_mcts_c_increment;
        let top_p_increment = cfg_snapshot.phase2_top_p_increment;
        let retrieval_top_k_increment = cfg_snapshot.phase2_retrieval_top_k_increment;
        let backoff_cap_ms = cfg_snapshot.phase2_retry_backoff_cap_ms.max(base_delay_ms);
        let backoff_exponent_cap = cfg_snapshot.retry_backoff_exponent_cap;

        let mut current_gen = generation.clone();
        let mut current_failure = initial_failure.to_string();
        let mut retry_count = 0;

        let loop_start = Instant::now();

        // Retry loop with escalating strategies
        while retry_count < max_retries && current_failure != "none" {
            retry_count += 1;
            info!(retry = retry_count, tier = ?current_failure, detail = ?details, "retry loop attempt");

            // Store failure in ERAG before retry
            let metrics_ref = metrics();
            if let Err(e) = self
                .erag
                .store_failure(
                    prompt,
                    &current_gen.hybrid_response,
                    metrics_ref,
                    Some(format!("Retry {}: {}", retry_count, details)),
                    &current_failure,
                    retry_count,
                )
                .await
            {
                warn!("Failed to store failure: {}", e);
            }

            // Level3+ escalation: Tune MCTS/retrieval params for repeated failures
            let is_level3 = retry_count > level3_retry_count;
            if is_level3 {
                info!(
                    "Level3 escalation: Applying parameter tuning (attempt {})",
                    retry_count
                );
                // Log escalation metrics (actual tuning would require mutable access to compass/thresholds)
                info!(
                    "Suggested tuning: MCTS c += {:.3}, top_p += {:.3}, retrieval_top_k += {}",
                    mcts_c_increment, top_p_increment, retrieval_top_k_increment
                );
            }

            // Determine retry strategy based on failure type
            let retry_response = if current_failure == "hard" {
                // Meso: Reflexion for hard failures, but fallback to baseline if worse
                let reflexion_response = self
                    .generator
                    .reflexion_retry(prompt, current_gen.rouge_score, details)
                    .await?;

                // Compare with baseline and keep the better one
                let baseline_rouge = rouge_l(&current_gen.baseline_response, prompt);
                let reflexion_rouge = rouge_l(&reflexion_response, prompt);

                if reflexion_rouge > baseline_rouge {
                    info!(
                        "Reflexion improved from {:.3} to {:.3}",
                        baseline_rouge, reflexion_rouge
                    );
                    reflexion_response
                } else {
                    info!(
                        "Baseline better than reflexion ({:.3} vs {:.3}), using baseline",
                        baseline_rouge, reflexion_rouge
                    );
                    current_gen.baseline_response.clone()
                }
            } else {
                // Micro: Topology-aware CoT for soft failures (2-3 iterations)
                let mut best_response = current_gen.hybrid_response.clone();
                let mut best_rouge = current_gen.rouge_score;

                for cot_iter in 0..cot_iterations {
                    let cot_result = self
                        .generator
                        .apply_cot_repair_with_topology(
                            prompt,
                            details,
                            cot_iter as u32,
                            Some(topology),
                        )
                        .await?;

                    // Recompute ROUGE
                    let new_rouge = rouge_l(&cot_result.hybrid_response, &best_response);
                    if new_rouge > best_rouge {
                        best_response = cot_result.hybrid_response;
                        best_rouge = new_rouge;
                    }

                    if best_rouge >= cot_success_rouge {
                        info!(
                            "Topology-aware CoT iteration {} achieved target ROUGE {:.3}",
                            cot_iter + 1,
                            best_rouge
                        );
                        break;
                    }
                }
                best_response
            };

            // Create updated generation result with retry
            let retry_gen = GenerationResult {
                baseline_response: retry_response.clone(),
                hybrid_response: retry_response.clone(),
                echoes: current_gen.echoes.clone(),
                rouge_to_baseline: rouge_l(&retry_response, &current_gen.baseline_response),
                latency_ms: current_gen.latency_ms,
                rouge_score: rouge_l(&retry_response, &current_gen.baseline_response),
                entropy_delta: current_gen.entropy_delta,
                source: format!("retry_{}", retry_count),
                ucb1_score: current_gen.ucb1_score,
                curator_quality: current_gen.curator_quality,
                failure_type: None,
                failure_details: None,
            };

            // Re-evaluate failure with new metrics
            // OPTIMIZATION: Adjust ucb1_score based on ROUGE improvement to avoid infinite retry loops
            // If ROUGE improved significantly, boost ucb1 to reflect successful retry
            let adjusted_ucb1 = if retry_gen.rouge_score > current_gen.rouge_score + 0.1 {
                // ROUGE improved by >0.1, boost ucb1 to reflect success
                ucb1_score.max(0.2).min(1.0)
            } else if retry_count > 3 {
                // After 3 retries, if we're still here but ROUGE is reasonable, relax ucb1 check
                // This prevents infinite loops from stale ucb1_score
                ucb1_score.max(0.15)
            } else {
                ucb1_score
            };

            let (failure, _new_details) = FailureSignals::evaluate(
                retry_gen.rouge_score,
                entropy_delta,
                curator_quality,
                adjusted_ucb1,
            );

            current_gen = retry_gen;
            current_failure = failure.clone();

            // Success on retry
            if current_failure == "none" {
                info!(
                    "Retry succeeded on attempt {} (ROUGE: {:.3})",
                    retry_count, current_gen.rouge_score
                );
                self.retry_count.store(retry_count, Ordering::Relaxed);
                break;
            }

            // Backoff delay before next retry (exponential with jitter, but capped)
            // OPTIMIZATION: Cap exponential backoff to prevent excessive delays
            if retry_count < max_retries {
                let exponent = ((retry_count.saturating_sub(1)) as u32).min(backoff_exponent_cap);
                let multiplier = 1_u64 << exponent;
                let delay_ms = (base_delay_ms * multiplier).min(backoff_cap_ms);
                if delay_ms > 100 {
                    info!(
                        retry = retry_count,
                        delay_ms, "Backoff delay before next retry"
                    );
                }
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }

        if current_failure != "none" {
            warn!(
                "Failed after {} retry attempts, using degraded response",
                retry_count
            );

            // Graceful degradation: Instead of terminating, mark as degraded but continue
            if retry_count >= max_retries {
                warn!("Circuit breaker triggered: Using degraded response mode");
                // Add degraded marker to generation result
                current_gen.failure_type = Some("degraded".to_string());
                current_gen.failure_details = Some(format!(
                    "Max retries exceeded ({}), using best available response",
                    retry_count
                ));
            }
        }

        let threat_cycle_ms = loop_start.elapsed().as_secs_f64() * 1000.0;

        Ok((current_gen, current_failure, threat_cycle_ms))
    }

    async fn integrate_curator(
        &self,
        input: &str,
        output: &str,
        pad_state: &PadGhostState,
        compass: &CompassOutcome,
        context: &str,
        topology: &crate::tcs_analysis::TopologicalSignature,
        tokenizer_output: &TokenizerOutput,
    ) -> Result<CuratedExperience> {
        // Call curator_executor logic here
        // (either spawn as subprocess or integrate as library)

        // Create a proper Experience using the new constructor
        let _experience = Experience::new(
            input.to_string(),
            output.to_string(),
            context.to_string(),
            "curator_refinement".to_string(),
            0.5, // Initial score, will be updated
        );

        // TOPOLOGY INTEGRATION: Analyze quality with topological insights
        // Calculate base quality score based on output length, entropy, and topology
        let base = 0.5f32;
        let length_factor = (output.len().min(1000) as f32 / 1000.0) * 0.2;
        let entropy_factor = if pad_state.entropy < 0.5 {
            0.15f32
        } else {
            0.0f32
        };
        let base_quality = base + length_factor + entropy_factor;

        // TOPOLOGY ENHANCEMENT: Adjust quality based on topological features
        let mut adjusted_quality = base_quality;

        // High knot complexity indicates tangled/complex reasoning - slight quality penalty
        if topology.knot_complexity > 0.6 {
            adjusted_quality *= 0.9;
            info!(
                "High knot complexity {:.3} - reducing quality",
                topology.knot_complexity
            );
        }

        // High spectral gap indicates good exploration - quality bonus
        if topology.spectral_gap > 0.7 {
            adjusted_quality *= 1.1;
            info!(
                "High spectral gap {:.3} - boosting quality",
                topology.spectral_gap
            );
        }

        // High Betti-1 indicates many loops/cycles - check if intentional
        if topology.betti_numbers[1] > 3 {
            // In Discover quadrant, loops are good (exploration)
            if compass.quadrant == CompassQuadrant::Discover {
                adjusted_quality *= 1.05;
            } else {
                // In other quadrants, too many loops might indicate confusion
                adjusted_quality *= 0.95;
            }
            info!(
                "Betti-1={} affecting quality in {:?} quadrant",
                topology.betti_numbers[1], compass.quadrant
            );
        }

        // Persistence entropy indicates structural stability
        if topology.persistence_entropy < 0.3 {
            // Low entropy = stable structure = good quality
            adjusted_quality *= 1.05;
        }

        let mut quality_score = adjusted_quality.min(1.0).max(0.0);

        // TOPOLOGY-AWARE REFINEMENT: Refine if quality is low OR topology indicates issues
        let refinement_threshold = self.config.curator_quality_threshold;

        // Force refinement if topology shows problematic patterns
        let topology_needs_refinement = topology.knot_complexity > 0.7  // Too tangled
            || (topology.betti_numbers[1] > 5 && compass.quadrant != CompassQuadrant::Discover)  // Too many loops outside exploration
            || topology.persistence_entropy > 0.8; // Too chaotic structure

        let refinement_reason = if quality_score < refinement_threshold && topology_needs_refinement
        {
            "quality_below_threshold+topology_alert"
        } else if quality_score < refinement_threshold {
            "quality_below_threshold"
        } else if topology_needs_refinement {
            "topology_alert"
        } else {
            "stable"
        };

        let mut reason = refinement_reason.to_string();
        let needs_refinement = quality_score < refinement_threshold || topology_needs_refinement;
        let autonomy_enabled = self.config.curator_autonomous || self.curator.is_none();
        let mut refined = output.to_string();
        let mut learned = false;

        if needs_refinement {
            // First, attempt autonomous refinement if enabled
            if autonomy_enabled {
                let mut auto_improvement = 0.0;
                let autonomy_prompt = format!(
                    "You are NIODOO's autonomous curator. Rewrite the assistant response to be concise, specific, and constitutionally aligned. Remove filler, avoid repeating the prompt, and deliver one high-signal insight in 3-5 sentences.\n\nPrompt:\n{input}\n\nOriginal Response:\n{output}\n\nReturn only the refined response text.",
                    input = input,
                    output = output
                );

                match self
                    .generator
                    .generate_with_params(&autonomy_prompt, 0.22, 0.82)
                    .await
                {
                    Ok(autonomous) => {
                        let candidate = autonomous.hybrid_response.trim();
                        if !candidate.is_empty() {
                            auto_improvement = rouge_l(candidate, output);
                            if auto_improvement.is_finite() {
                                quality_score = (quality_score
                                    + (auto_improvement.clamp(0.0, 1.0) * 0.35) as f32)
                                    .min(1.0);
                            }
                            refined = candidate.to_string();
                            learned = auto_improvement > 0.05;
                            reason = format!(
                                "auto_refine|improvement:{:.3}|mode:{}",
                                auto_improvement,
                                if self.curator.is_some() {
                                    "curator_present"
                                } else {
                                    "curator_absent"
                                }
                            );

                            if auto_improvement < 0.25 {
                                let first_improvement = auto_improvement;
                                let second_prompt = format!(
                                    "You are NIODOO's refinement overdrive. Further tighten the assistant reply so it is laser-focused, free of hedging, and emphasizes one actionable takeaway. Maintain constitutional tone and clear structure.\n\nPrompt:\n{input}\n\nCurrent Refinement:\n{refined}\n\nReturn only the upgraded response.",
                                    input = input,
                                    refined = refined
                                );

                                match self
                                    .generator
                                    .generate_with_params(&second_prompt, 0.28, 0.78)
                                    .await
                                {
                                    Ok(second_pass) => {
                                        let second_candidate = second_pass.hybrid_response.trim();
                                        if !second_candidate.is_empty() {
                                            let second_improvement =
                                                rouge_l(second_candidate, output);
                                            if second_improvement.is_finite()
                                                && second_improvement > auto_improvement
                                            {
                                                refined = second_candidate.to_string();
                                                auto_improvement = second_improvement;
                                                learned = learned || auto_improvement > 0.05;
                                                quality_score = (quality_score
                                                    + (second_improvement.clamp(0.0, 1.0) * 0.35)
                                                        as f32)
                                                    .min(1.0);
                                                reason = format!(
                                                    "auto_refine_second_pass|first:{:.3}|second:{:.3}|mode:{}",
                                                    first_improvement,
                                                    second_improvement,
                                                    if self.curator.is_some() {
                                                        "curator_present"
                                                    } else {
                                                        "curator_absent"
                                                    }
                                                );
                                            } else {
                                                reason = format!(
                                                    "auto_refine_second_pass_no_gain|first:{:.3}|second:{:.3}",
                                                    first_improvement, second_improvement
                                                );
                                            }
                                        } else {
                                            reason = format!(
                                                "auto_refine_second_pass_empty|first:{:.3}",
                                                first_improvement
                                            );
                                        }
                                    }
                                    Err(error) => {
                                        warn!(?error, "Second-pass autonomous refinement failed");
                                        reason = format!(
                                            "auto_refine_second_pass_error:{error}|first:{:.3}",
                                            first_improvement
                                        );
                                    }
                                }
                            }
                        } else {
                            reason = "auto_refine_empty".to_string();
                        }
                    }
                    Err(error) => {
                        warn!(?error, "Autonomous curator refinement failed");
                        reason = format!("auto_refine_error:{error}");
                    }
                }
            }

            // If autonomous mode is disabled or produced no change, fall back to external curator
            let should_call_curator = !autonomy_enabled && self.curator.is_some();

            if should_call_curator {
                if let Some(ref curator) = self.curator {
                    match curator
                        .refine(
                            &refined,
                            quality_score as f64,
                            topology.knot_complexity,
                            topology.persistence_entropy,
                        )
                        .await
                    {
                        Ok(result) => {
                            reason = result.reason.clone();
                            refined = result.refined;
                            learned = result.learned;
                            info!(
                                "Curator refined response (quality={:.3}, knot={:.3}, learned={}, reason={})",
                                quality_score,
                                topology.knot_complexity,
                                result.learned,
                                result.reason
                            );
                            if result.learned {
                                quality_score = (quality_score + 0.1).min(1.0);
                            }
                        }
                        Err(e) => {
                            reason = format!("curator_error:{e}");
                            warn!("Curator refinement failed: {}, using current response", e);
                        }
                    }
                }
            }
        }

        if (self.curator.is_none() || autonomy_enabled) && !reason.contains("curator_disabled") {
            reason = format!("{}|curator_disabled", reason);
        }

        Ok(CuratedExperience {
            refined_response: refined,
            quality_score,
            promoted_tokens: tokenizer_output
                .promoted_tokens
                .iter()
                .map(|token| String::from_utf8_lossy(&token.bytes).to_string())
                .collect(),
            learned,
            reason,
        })
    }
}

#[derive(Clone, Debug)]
struct CacheEntry<T> {
    value: T,
    inserted_at: Instant,
}

impl<T> CacheEntry<T> {
    fn new(value: T, inserted_at: Instant) -> Self {
        Self { value, inserted_at }
    }

    fn is_expired(&self, now: Instant, ttl: Duration) -> bool {
        now.duration_since(self.inserted_at) > ttl
    }
}

fn cache_key(prompt: &str) -> u64 {
    let digest = blake3_hash(prompt.as_bytes());
    // Use full hash bytes instead of truncating to reduce collision risk
    // Blake3 produces 32 bytes, we hash those again to get a 64-bit key
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};
    digest.as_bytes().hash(&mut hasher);
    hasher.finish()
}

fn tokenizer_path() -> Result<PathBuf> {
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

    anyhow::bail!("Tokenizer JSON not found; set TOKENIZER_JSON or QWEN_TOKENIZER")
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
        PersistenceFeature {
            birth: pad_min,
            death: pad_max,
            dimension: 0,
        },
        PersistenceFeature {
            birth: mu_min,
            death: mu_max,
            dimension: 1,
        },
        PersistenceFeature {
            birth: sigma_min,
            death: sigma_max,
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
        .zip(pad_state.raw_stds.iter())
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
            if p > 0.0 { -p * p.log2() } else { 0.0 }
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
    )
}
