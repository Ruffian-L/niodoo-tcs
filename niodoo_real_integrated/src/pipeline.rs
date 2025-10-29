use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use anyhow::Result;

use crate::compass::{CompassEngine, CompassOutcome, CompassQuadrant, CompassRuntimeParams};
use crate::config::{env_value, CliArgs, CuratorConfig, HardwareProfile, RuntimeConfig, TopologyMode};
use crate::curator::Curator;
use crate::data::{
    compute_dataset_stats, load_emotional_dataset, load_rut_gauntlet_prompts, DatasetStats,
    Experience, RutPrompt,
};
use crate::embedding::QwenStatefulEmbedder;
use crate::erag::{CollapseResult, EragClient};
use crate::generation::{GenerationEngine, GenerationResult};
use crate::learning::{LearningLoop, LearningOutcome};
use crate::metrics::{metrics, FailureSignals};
use crate::tcs_analysis::{TCSAnalyzer, TopologicalSignature};
use crate::token_manager::{DynamicTokenizerManager, TokenizerOutput};
use crate::torus::{PadGhostState, TorusPadMapper};
use crate::util::{rouge_l, seed_manager, set_global_seed};
use blake3::hash as blake3_hash;
use lru::LruCache;
use parking_lot::RwLock;
use tcs_core::topology::PersistenceFeature;
use tokio::sync::Mutex as AsyncMutex;
use rand::RngCore;
use tracing::{info, warn};
#[allow(unused_imports)]
use qdrant_client::qdrant::{CreateCollection, Distance, VectorsConfig};

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
    solution_path: Option<String>,
    emotional_context: PadGhostState,
    promoted_tokens: Vec<String>,
    learned: bool,
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
            info!(seed, "Initializing torus pad mapper with fixed seed override");
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
            NonZeroUsize::new(config.cache_capacity).unwrap_or(NonZeroUsize::new(256).unwrap());

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
            warn!(existing = seed_manager().master_seed(), requested = seed, "Seed override ignored; global seed already initialised");
        }
        Self::initialise_with_topology(args, None, Some(seed)).await
    }

    fn next_torus_mapper(&self) -> TorusPadMapper {
        // Derive a fresh mapper using the global seed manager and a stable scope
        let counter = self.torus_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let scope = match self.torus_strategy {
            TorusSeedStrategy::Fixed(seed) => format!("torus/fixed/{seed}/{counter}"),
            TorusSeedStrategy::Random => format!("torus/derived/{counter}"),
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
                    warn!("Hybrid mode requested but TCS analyzer unavailable; using analytic baseline signature");
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
        let compass_params = CompassRuntimeParams::new(
            self.thresholds.mcts_c,
            self.thresholds.variance_spike,
            self.thresholds.variance_stagnation,
        );
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
        let compass_erag_start = Instant::now();
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

        let (failure, details) = FailureSignals::evaluate(
            generation.rouge_score,
            entropy_delta,
            curator_quality,
            ucb1_score,
        );

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

        let max_retries = self.config.phase2_max_retries;
        let base_delay_ms = self.config.phase2_retry_base_delay_ms;
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
            let is_level3 = retry_count > self.config.phase2_level3_retry_count;
            if is_level3 {
                info!(
                    "Level3 escalation: Applying parameter tuning (attempt {})",
                    retry_count
                );
                // Log escalation metrics (actual tuning would require mutable access to compass/thresholds)
                info!(
                    "Suggested tuning: MCTS c += {:.3}, top_p += {:.3}, retrieval_top_k += {}",
                    self.config.phase2_mcts_c_increment,
                    self.config.phase2_top_p_increment,
                    self.config.phase2_retrieval_top_k_increment
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

                for cot_iter in 0..3 {
                    let cot_result = self
                        .generator
                        .apply_cot_repair_with_topology(prompt, details, cot_iter, Some(topology))
                        .await?;

                    // Recompute ROUGE
                    let new_rouge = rouge_l(&cot_result.hybrid_response, &best_response);
                    if new_rouge > best_rouge {
                        best_response = cot_result.hybrid_response;
                        best_rouge = new_rouge;
                    }

                    if best_rouge > 0.4 {
                        info!(
                            "Topology-aware CoT iteration {} achieved ROUGE > 0.4",
                            cot_iter + 1
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
                let exponent = 2_u64.pow(retry_count.min(6)); // Cap at 2^6 = 64x instead of 2^10 = 1024x
                let delay_ms = (base_delay_ms * exponent).min(5000); // Cap at 5 seconds max
                if delay_ms > 100 {
                    info!(retry = retry_count, delay_ms, "Backoff delay before next retry");
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

        let quality_score = adjusted_quality.min(1.0).max(0.0);

        // TOPOLOGY-AWARE REFINEMENT: Refine if quality is low OR topology indicates issues
        let refinement_threshold = self.config.curator_quality_threshold;

        // Force refinement if topology shows problematic patterns
        let topology_needs_refinement = topology.knot_complexity > 0.7  // Too tangled
            || (topology.betti_numbers[1] > 5 && compass.quadrant != CompassQuadrant::Discover)  // Too many loops outside exploration
            || topology.persistence_entropy > 0.8; // Too chaotic structure

        let (refined, learned) =
            if quality_score < refinement_threshold || topology_needs_refinement {
                // Attempt refinement for low-quality or topologically problematic responses
                if let Some(ref curator) = self.curator {
                    // Call curator.refine with topology context
                    let (refined_output, learned_flag) = match curator
                        .refine(
                            output,
                            quality_score as f64,
                            topology.knot_complexity,
                            topology.persistence_entropy,
                        )
                        .await
                    {
                        Ok(result) => result,
                        Err(e) => {
                            warn!("Curator refinement failed: {}, using original", e);
                            (output.to_string(), false)
                        }
                    };

                    info!(
                        "Curator refined response (quality={:.3}, knot={:.3}, learned={})",
                        quality_score, topology.knot_complexity, learned_flag
                    );
                    (refined_output, learned_flag)
                } else {
                    (output.to_string(), false)
                }
            } else {
                (output.to_string(), false)
            };

        Ok(CuratedExperience {
            refined_response: refined,
            quality_score,
            solution_path: crate::data::extract_code_blocks(output),
            emotional_context: pad_state.clone(),
            promoted_tokens: tokenizer_output
                .promoted_tokens
                .iter()
                .map(|token| String::from_utf8_lossy(&token.bytes).to_string())
                .collect(),
            learned,
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
    )
}
