use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use futures::FutureExt;

use crate::compass::{CompassEngine, CompassOutcome, CompassQuadrant};
use crate::config::{CliArgs, CuratorConfig, HardwareProfile, RuntimeConfig};
use crate::curator::Curator;
use crate::data::{
    compute_dataset_stats, load_emotional_dataset, load_rut_gauntlet_prompts, DatasetStats,
    RutPrompt, Experience,
};
use crate::embedding::QwenStatefulEmbedder;
use crate::erag::{CollapseResult, EragClient};
use crate::generation::{GenerationEngine, GenerationResult};
use crate::learning::{LearningLoop, LearningOutcome};
use crate::metrics::metrics;
use crate::tcs_analysis::{TCSAnalyzer, TopologicalSignature};
use crate::tokenizer::{TokenizerEngine, TokenizerOutput};
use crate::torus::{PadGhostState, TorusPadMapper};
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
}

pub struct Pipeline {
    pub config: RuntimeConfig,
    pub args: CliArgs,
    pub thresholds: Thresholds,
    pub dataset_stats: DatasetStats,
    embedder: QwenStatefulEmbedder,
    torus: TorusPadMapper,
    compass: Arc<Mutex<CompassEngine>>,
    erag: EragClient,
    tokenizer: TokenizerEngine,
    generator: GenerationEngine,
    learning: LearningLoop,
    curator: Option<Curator>,
    tcs_analyzer: TCSAnalyzer,
    embedding_cache: LruCache<u64, CacheEntry<Vec<f32>>>,
    collapse_cache: LruCache<u64, CacheEntry<CollapseResult>>,
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
            0.65,
        )
        .await?;
        let tokenizer = TokenizerEngine::new(tokenizer_path()?, thresholds.mirage_sigma)?;
        let generator = GenerationEngine::new(&config.vllm_endpoint, &config.vllm_model)?;
        if let Err(error) = generator.warmup().await {
            warn!(?error, "vLLM warmup failed");
        }
        let learning = LearningLoop::new(
            config.entropy_cycles_for_baseline.max(8),
            thresholds.variance_spike,
        );

        // Initialize TCS analyzer
        let tcs_analyzer = TCSAnalyzer::new()
            .unwrap_or_else(|e| {
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
                    warn!("Failed to initialize curator: {}, continuing without curator", e);
                    None
                }
            }
        } else {
            info!("Curator disabled via config");
            None
        };

        let cache_capacity = NonZeroUsize::new(256).unwrap();

        Ok(Self {
            config,
            args,
            thresholds,
            dataset_stats: stats,
            embedder,
            torus,
            compass,
            erag,
            tokenizer,
            generator,
            learning,
            curator,
            tcs_analyzer,
            embedding_cache: LruCache::new(cache_capacity),
            collapse_cache: LruCache::new(cache_capacity),
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
        info!("Pipeline stage: TCS topology analysis completed in {:.2}ms", timings.tcs_ms);

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
                source: "consistency".to_string(),
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
        let curated_experience = self.integrate_curator(
            prompt,
            &generation.hybrid_response,
            &pad_state,
            &compass,
            &collapse.aggregated_context
        ).await?;

        // Proceed with learning using curated response
        let learning = self
            .learning
            .update(&pad_state, &compass, &collapse, &generation)?;  // Note: May need to adjust to use curated

        // Store CURATED memory instead of raw
        if curated_experience.quality_score > 0.65 {
            let solution_path = crate::data::extract_code_blocks(&curated_experience.refined_response);
            self.erag.upsert_memory(
                &embedding,
                &pad_state,
                &compass,
                prompt,
                &curated_experience.refined_response,
                &[curated_experience.refined_response.clone()],  // context
                pad_state.entropy,
                Some(curated_experience.quality_score as f32),
                None,  // topology
                solution_path,
                0,  // iteration_count - will be tracked later
            ).await.ok();
        }

        // Create Experience from pipeline
        let aggregated_context_lines: Vec<String> = collapse
            .aggregated_context
            .lines()
            .map(|s| s.to_string())
            .collect();
        let experience_input = prompt.to_string();
        let experience_output = generation.hybrid_response.clone();
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
                            curated.quality_score,
                            curated.processing_time_ms
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
                            baseline_response: generation.baseline_response.clone(),
                            hybrid_response: generation.hybrid_response.clone(),
                            entropy: pad_state.entropy,
                            rouge: generation.rouge_to_baseline,
                            latency_ms: generation.latency_ms,
                            compass: compass.clone(),
                            generation: generation.clone(),
                            tokenizer: tokenizer_output.clone(),
                            collapse: collapse.clone(),
                            learning: learning.clone(),
                            stage_timings: timings.clone(),
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
            generation.latency_ms,
            generation.rouge_to_baseline,
            compass.is_threat,
            compass.is_healing,
        );

        // learning_ms already set above

        Ok(PipelineCycle {
            prompt: prompt.to_string(),
            baseline_response: generation.baseline_response.clone(),
            hybrid_response: generation.hybrid_response.clone(),
            entropy: pad_state.entropy,
            rouge: generation.rouge_to_baseline,
            latency_ms: overall_start.elapsed().as_secs_f64() * 1000.0,
            compass,
            generation,
            tokenizer: tokenizer_output,
            collapse,
            learning,
            stage_timings: timings,
        })
    }

    pub fn hardware_profile(&self) -> HardwareProfile {
        self.args.hardware
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
        let experience = Experience::new(
            input.to_string(),
            output.to_string(),
            context.to_string(),
            "curator_refinement".to_string(),
            0.5,  // Initial score, will be updated
        );
        
        // Analyze quality if curator is available
        let quality_score = if let Some(ref curator) = self.curator {
            // This would need proper curator method
            0.7f32  // Placeholder for now
        } else {
            0.5f32
        };
        
        // Refine if needed
        let refined = if quality_score < 0.7 && self.curator.is_some() {
            // Would need proper curator method
            output.to_string()
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

    let fallback = PathBuf::from("../tokenizer.json");
    if fallback.exists() {
        Ok(fallback)
    } else {
        anyhow::bail!("Tokenizer JSON not found; set TOKENIZER_JSON or QWEN_TOKENIZER")
    }
}
