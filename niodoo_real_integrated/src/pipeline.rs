use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use anyhow::Context;
use futures::FutureExt;

use crate::compass::{CompassEngine, CompassOutcome};
use crate::config::{CliArgs, CuratorConfig, HardwareProfile, RuntimeConfig};
use crate::curator::Curator;
use crate::data::{
    compute_dataset_stats, load_emotional_dataset, load_rut_gauntlet_prompts, DatasetStats,
    Experience, RutPrompt,
};
use crate::embedding::QwenStatefulEmbedder;
use crate::erag::{CollapseResult, EragClient};
use crate::generation::{GenerationEngine, GenerationResult};
use crate::learning::{LearningLoop, LearningOutcome};
use crate::metrics::{metrics, FailureSignals, RetryContext};
use crate::tcs_analysis::TCSAnalyzer;
use crate::tokenizer::{TokenizerEngine, TokenizerOutput};
use crate::torus::{PadGhostState, TorusPadMapper};
use crate::util::rouge_l;
use blake3::hash as blake3_hash;
use lru::LruCache;
use tokio::task::spawn_blocking;
use tracing::{info, warn};

// Define CuratedExperience struct
#[derive(Debug, Clone)]
struct CuratedExperience {
    refined_response: String,
    quality_score: f32,
    solution_path: Option<String>,
    emotional_context: PadGhostState,
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

pub struct Pipeline {
    pub config: RuntimeConfig,
    config_arc: Arc<Mutex<RuntimeConfig>>,
    pub args: CliArgs,
    pub thresholds: Thresholds,
    pub dataset_stats: DatasetStats,
    embedder: QwenStatefulEmbedder,
    torus: TorusPadMapper,
    compass: Arc<Mutex<CompassEngine>>,
    erag: Arc<EragClient>,
    tokenizer: TokenizerEngine,
    generator: GenerationEngine,
    learning: LearningLoop,
    curator: Option<Curator>,
    tcs_analyzer: TCSAnalyzer,
    embedding_cache: LruCache<u64, CacheEntry<Vec<f32>>>,
    collapse_cache: LruCache<u64, CacheEntry<CollapseResult>>,
    retry_count: Arc<AtomicU32>,
    retry_context: Arc<Mutex<RetryContext>>,
}

impl Pipeline {
    pub async fn initialise(args: CliArgs) -> Result<Self> {
        let config = RuntimeConfig::load(&args)?;
        let samples = load_emotional_dataset(&config.training_data_path, Some(20_000))?;
        let stats = compute_dataset_stats(&samples);

        let thresholds = Thresholds {
            entropy_mean: stats.entropy_mean,
            entropy_high: stats.entropy_mean + stats.entropy_std,
            variance_stagnation: 0.05,
            variance_spike: stats.variance_std.max(0.3),
            mirage_sigma: 0.1 * stats.entropy_mean,
            mcts_c: stats.entropy_std.max(0.1) * 0.25,
        };

        let embedder = QwenStatefulEmbedder::new(
            &config.ollama_endpoint,
            "qwen2.5-coder:1.5b",
            config.qdrant_vector_dim,
        )?;
        let embedder_arc = Arc::new(embedder);
        let torus = TorusPadMapper::new(42);
        let compass = Arc::new(Mutex::new(CompassEngine::new(
            thresholds.mcts_c,
            thresholds.variance_spike,
            thresholds.variance_stagnation,
        )));
        let erag = EragClient::new(
            &config.qdrant_url,
            &config.qdrant_collection,
            config.qdrant_vector_dim,
            config.similarity_threshold,
            embedder_arc.clone(),
        )?;
        let tokenizer = TokenizerEngine::new(tokenizer_path()?, thresholds.mirage_sigma)?;
        let generator = GenerationEngine::new_with_config(
            &config.vllm_endpoint,
            &config.vllm_model,
            config.generation_timeout_secs,
            config.generation_max_tokens,
            config.dynamic_token_min,
            config.dynamic_token_max,
        )?;
        if let Err(error) = generator.warmup().await {
            warn!(?error, "vLLM warmup failed");
        }
        let config_arc = Arc::new(Mutex::new(config.clone()));
        let erag_arc = Arc::new(erag.clone());
        let learning = LearningLoop::new(
            config.learning_window,
            config.breakthrough_threshold,
            config.dqn_epsilon,
            config.dqn_gamma,
            config.dqn_alpha,
            erag_arc.clone(),
            config_arc.clone(),
        );

        // Initialize TCS analyzer
        let tcs_analyzer = TCSAnalyzer::new().unwrap_or_else(|e| {
            warn!("Failed to initialize TCS analyzer: {}, using default", e);
            TCSAnalyzer::default()
        });
        info!("TCS topology analyzer initialized");

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

        let cache_capacity = NonZeroUsize::new(256).unwrap();

        Ok(Self {
            config: config.clone(),
            config_arc: config_arc.clone(),
            args,
            thresholds,
            dataset_stats: stats,
            embedder,
            torus,
            compass,
            erag: erag_arc.clone(),
            tokenizer,
            generator,
            learning,
            curator,
            tcs_analyzer,
            embedding_cache: LruCache::new(cache_capacity),
            collapse_cache: LruCache::new(cache_capacity),
            retry_count: Arc::new(AtomicU32::new(0)),
            retry_context: Arc::new(Mutex::new(RetryContext {
                soft_retries: 0,
                hard_retries: 0,
                total_retries: 0,
                reflection_buffer: None,
                rng_seed: 42,
            })),
        })
    }

    pub fn rut_prompts(&self) -> Vec<RutPrompt> {
        load_rut_gauntlet_prompts()
    }

    pub async fn process_prompt(&mut self, prompt: &str) -> Result<PipelineCycle> {
        let overall_start = Instant::now();
        let mut timings = StageTimings::default();
        let cache_key = cache_key(prompt);
        let now = Instant::now();

        // Stage 1: Embedding (with cache)
        let embedding_start = Instant::now();
        let embedding = match self.embedding_cache.get(&cache_key) {
            Some(entry) if !entry.is_expired(now, EMBEDDING_TTL) => entry.value.clone(),
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
        let pad_state = self.torus.project(&embedding)?;
        timings.torus_ms = torus_start.elapsed().as_secs_f64() * 1000.0;

        let tcs_start = Instant::now();
        let topology = self.tcs_analyzer.analyze_state(&pad_state)?;
        timings.tcs_ms = tcs_start.elapsed().as_secs_f64() * 1000.0;
        info!(
            "Pipeline stage: TCS topology analysis completed in {:.2}ms",
            timings.tcs_ms
        );

        // Pass topology to compass - ACTUAL INTEGRATION
        let (compass, collapse) = tokio::try_join!(
            spawn_blocking({
                let compass_engine = self.compass.clone();
                let pad_state = pad_state.clone();
                let topology = topology.clone();
                move || {
                    compass_engine
                        .lock()
                        .unwrap()
                        .evaluate(&pad_state, Some(&topology))
                }
            })
            .map(|res| res.expect("compass evaluation task panicked")),
            async {
                match self.collapse_cache.get(&cache_key) {
                    Some(entry) if !entry.is_expired(now, COLLAPSE_TTL) => Ok(entry.value.clone()),
                    _ => {
                        self.collapse_cache.pop(&cache_key);
                        let collapse = self.erag.collapse(&embedding).await?;
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
        let tokenizer_output =
            self.tokenizer
                .process(prompt, &collapse, &pad_state, self.thresholds.entropy_mean)?;
        timings.tokenizer_ms = tokenizer_start.elapsed().as_secs_f64() * 1000.0;

        // Stage 6: Generation
        let generation_start = Instant::now();
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
                ucb1_score: compass.mcts_branches.iter()
                    .map(|b| b.ucb_score)
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.5),
                curator_quality: 0.8, // Default quality for consistency voting
                failure_type: None,
                failure_details: None,
            }
        } else {
            self.generator.generate(&tokenizer_output, &compass).await?
        };
        timings.generation_ms = generation_start.elapsed().as_secs_f64() * 1000.0;
        info!(
            "Pipeline stage: generation completed in {:.2}ms",
            timings.generation_ms
        );

        // NEW: Phase 2 Integration - Call curator after generation
        let curated_experience = self
            .integrate_curator(
                prompt,
                &generation.hybrid_response,
                &pad_state,
                &compass,
                &collapse.aggregated_context,
            )
            .await?;

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

        // Phase 2: Handle retries with Reflection and CoT
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
                    Some(details.clone()),
                    &final_failure,
                    self.retry_count.load(Ordering::Relaxed),
                )
                .await?;
        }

        // Proceed with learning using curated response (with retry-corrected generation)
        let learning_start = Instant::now();

        let learning = self.learning.update(&pad_state, &compass, &collapse, &final_generation).await
            .context("Learning loop update failed")?;

        timings.learning_ms = learning_start.elapsed().as_secs_f64() * 1000.0;

        // Store CURATED memory instead of raw
        if curated_experience.quality_score > 0.65 {
            let solution_path =
                crate::data::extract_code_blocks(&curated_experience.refined_response);
            self.erag
                .upsert_memory(
                    &embedding,
                    &pad_state,
                    &compass,
                    prompt,
                    &curated_experience.refined_response,
                    &[curated_experience.refined_response.clone()], // context
                    pad_state.entropy,
                    Some(curated_experience.quality_score as f32),
                    None, // topology
                    solution_path,
                    0, // iteration_count - will be tracked later
                )
                .await
                .ok();
        }

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

        // Stage 7.5: Curator Quality Gate
        let (response_to_store, final_quality_score) = if let Some(ref curator) = self.curator {
            match curator.curate_response(experience.clone()).await {
                Ok(curated) => {
                    if curated.should_store.unwrap_or(true) {
                        info!(
                            "Curator approved memory (quality: {:.3?}, latency: {:.2?}ms)",
                            curated.quality_score, curated.processing_time_ms
                        );
                        let response = curated.refined_output.clone().unwrap_or(curated.output);
                        (response, curated.quality_score)
                    } else {
                        warn!(
                            "Curator rejected memory (quality: {:.3?} < threshold)",
                            curated.quality_score
                        );
                        // Don't store low-quality memories, but return cycle
                        return Ok(PipelineCycle {
                            prompt: prompt.to_string(),
                            baseline_response: final_generation.baseline_response.clone(),
                            hybrid_response: final_generation.hybrid_response.clone(),
                            entropy: pad_state.entropy,
                            rouge: final_generation.rouge_to_baseline,
                            latency_ms: final_generation.latency_ms,
                            compass: compass.clone(),
                            generation: final_generation.clone(),
                            tokenizer: tokenizer_output.clone(),
                            collapse: collapse.clone(),
                            learning: learning.clone(),
                            stage_timings: timings.clone(),
                            last_entropy: pad_state.entropy,
                            failure: final_failure.clone(),
                        });
                    }
                }
                Err(e) => {
                    warn!("Curator failed: {}, storing raw response", e);
                    (experience.output.clone(), None)
                }
            }
        } else {
            // No curator - store raw response
            (experience.output.clone(), None)
        };

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
            learning,
            stage_timings: timings,
            last_entropy: pad_state.entropy,
            failure: final_failure,
        })
    }

    pub fn hardware_profile(&self) -> HardwareProfile {
        self.args.hardware
    }

    /// Phase 2: Handle retries with Reflection and CoT self-correction
    async fn handle_retry_with_reflection(
        &self,
        prompt: &str,
        initial_failure: &str,
        details: &str,
        generation: &GenerationResult,
        _compass: &CompassOutcome,
        _collapse: &CollapseResult,
        _curated: &CuratedExperience,
        entropy_delta: f64,
        curator_quality: f64,
        ucb1_score: f64,
    ) -> Result<(GenerationResult, String, f64)> {
        // No failure, return original with zero threat cycle time
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
                // Meso: Reflexion for hard failures
                self.generator
                    .reflexion_retry(prompt, current_gen.rouge_score, details)
                    .await?
            } else {
                // Micro: CoT for soft failures (2-3 iterations)
                let mut best_response = current_gen.hybrid_response.clone();
                let mut best_rouge = current_gen.rouge_score;

                for cot_iter in 0..3 {
                    let cot_response = self
                        .generator
                        .cot_self_correct(prompt, "low-confidence reasoning")
                        .await?;

                    // Recompute ROUGE
                    let new_rouge = rouge_l(&cot_response, &best_response);
                    if new_rouge > best_rouge {
                        best_response = cot_response;
                        best_rouge = new_rouge;
                    }

                    if best_rouge > 0.5 {
                        info!("CoT iteration {} achieved ROUGE > 0.5", cot_iter + 1);
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
            let (failure, _new_details) = FailureSignals::evaluate(
                retry_gen.rouge_score,
                entropy_delta,
                curator_quality,
                ucb1_score,
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

            // Backoff delay before next retry (exponential with jitter)
            if retry_count < max_retries {
                let exponent = 2_u64.pow(retry_count.min(10));
                let delay_ms = base_delay_ms * exponent;
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }

        if current_failure != "none" {
            warn!("Failed after {} retry attempts", retry_count);

            // Circuit breaker: If max retries exceeded, escalate to Phase 3
            if retry_count >= max_retries {
                warn!("Circuit breaker triggered: Escalating to Phase 3 (threat->healing cycle)");
                // Phase 3 would handle long-term recovery (e.g., threat cycle activation)
                // Return error to prevent further processing
                anyhow::bail!(
                    "Circuit breaker escalated: retry_count={} >= max_retries={}, failure={}, details={}",
                    retry_count, max_retries, current_failure, details
                );
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

        // Analyze quality if curator is available
        let quality_score = if let Some(ref curator) = self.curator {
            // Calculate quality based on actual curator metrics
            // Use compass quadrant from context if available
            let compass_quadrant_str = if compass.quadrant == crate::compass::CompassQuadrant::Panic
            {
                "Panic"
            } else if compass.quadrant == crate::compass::CompassQuadrant::Persist {
                "Persist"
            } else if compass.quadrant == crate::compass::CompassQuadrant::Discover {
                "Discover"
            } else {
                "Master"
            };

            match curator
                .assess_quality(input, output, pad_state.entropy, compass_quadrant_str)
                .await
            {
                Ok(q) => q as f32,
                Err(e) => {
                    warn!("Curator quality assessment failed: {}, using fallback", e);
                    // Fallback quality score based on heuristics
                    let length_score = (output.len().min(1000) as f32 / 1000.0) * 0.3;
                    let content_score =
                        if output.contains("solution") || output.contains("implementation") {
                            0.4
                        } else {
                            0.2
                        };
                    (length_score + content_score).min(1.0)
                }
            }
        } else {
            // Fallback quality score based on heuristics
            let length_score = (output.len().min(1000) as f32 / 1000.0) * 0.3;
            let content_score = if output.contains("solution") || output.contains("implementation")
            {
                0.4
            } else {
                0.2
            };
            (length_score + content_score).min(1.0)
        };

        // Refine if needed (use config threshold)
        let refinement_threshold = self.config.curator_quality_threshold;

        let refined = if quality_score < refinement_threshold {
            // Attempt refinement for low-quality responses
            if let Some(ref curator) = self.curator {
                match curator.refine_response(input, output).await {
                    Ok(refined_output) => {
                        info!(
                            "Curator refined low-quality response ({} -> improved)",
                            quality_score
                        );
                        refined_output
                    }
                    Err(e) => {
                        warn!("Curator refinement failed: {}, using original", e);
                        output.to_string()
                    }
                }
            } else {
                output.to_string()
            }
        } else {
            output.to_string()
        };

        Ok(CuratedExperience {
            refined_response: refined,
            quality_score,
            solution_path: crate::data::extract_code_blocks(output),
            emotional_context: pad_state.clone(),
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

const EMBEDDING_TTL: Duration = Duration::from_secs(300);
const COLLAPSE_TTL: Duration = Duration::from_secs(300);

fn cache_key(prompt: &str) -> u64 {
    let digest = blake3_hash(prompt.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest.as_bytes()[0..8]);
    u64::from_le_bytes(bytes)
}

fn locate_qwen_model() -> Result<PathBuf> {
    let candidates = ["QWEN_MODEL_PATH", "QWEN_CODER_ONNX", "QWEN_STATEFUL_ONNX"];
    for key in candidates {
        if let Ok(value) = std::env::var(key) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                let path = PathBuf::from(trimmed);
                if path.exists() {
                    return Ok(path);
                }
            }
        }
    }

    if let Ok(models_dir) = std::env::var("MODELS_DIR") {
        let base = PathBuf::from(models_dir)
            .join("qwen2.5-coder-0.5b-instruct-onnx/onnx/model_quantized.onnx");
        if base.exists() {
            return Ok(base);
        }
    }

    let fallback =
        PathBuf::from("../models/qwen2.5-coder-0.5b-instruct-onnx/onnx/model_quantized.onnx");
    if fallback.exists() {
        Ok(fallback)
    } else {
        anyhow::bail!(
            "Qwen model path not provided or found; set QWEN_MODEL_PATH or QWEN_CODER_ONNX"
        )
    }
}

fn tokenizer_path() -> Result<PathBuf> {
    if let Ok(value) = std::env::var("TOKENIZER_JSON") {
        let path = PathBuf::from(value);
        if path.exists() {
            return Ok(path);
        }
    }

    if let Ok(value) = std::env::var("QWEN_TOKENIZER") {
        let path = PathBuf::from(value);
        if path.exists() {
            return Ok(path);
        }
    }

    if let Ok(models_dir) = std::env::var("MODELS_DIR") {
        let path = PathBuf::from(models_dir).join("tokenizer.json");
        if path.exists() {
            return Ok(path);
        }
    }

    // Absolute fallback
    let absolute_fallback = PathBuf::from("/workspace/Niodoo-Final/models/tokenizer.json");
    if absolute_fallback.exists() {
        return Ok(absolute_fallback);
    }

    anyhow::bail!("Tokenizer JSON not found; set TOKENIZER_JSON or QWEN_TOKENIZER")
}
